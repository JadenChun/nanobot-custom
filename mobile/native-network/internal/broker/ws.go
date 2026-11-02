package broker

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/coder/websocket"
)

type WSRequest struct {
	Operation string            `json:"operation"`
	Path      string            `json:"path"`
	Headers   map[string]string `json:"headers,omitempty"`
}

type wsOut struct {
	kind websocket.MessageType
	data []byte
}

type wsSession struct {
	id     uint64
	core   *Core
	conn   *websocket.Conn
	ctx    context.Context
	cancel context.CancelFunc
	out    chan wsOut
	once   sync.Once
}

func (c *Core) WSOpen(input WSRequest) (uint64, error) {
	c.mu.Lock()
	state := c.state
	started := c.started
	testMode := c.testMode
	baseContext := c.ctx
	busy := c.ws != nil
	c.mu.Unlock()
	if busy {
		return 0, ErrBusy
	}
	if !testMode && (!started || state != "ready") {
		return 0, ErrUnavailable
	}
	if baseContext == nil {
		baseContext = context.Background()
	}
	target, route, err := resolveRoute(&c.cfg, c.origin, input.Operation, http.MethodGet, input.Path)
	if err != nil {
		return 0, err
	}
	if !route.ws {
		return 0, fmt.Errorf("%w: operation is not a WebSocket route", ErrInvalid)
	}
	headers, err := sanitizeHeaders(input.Headers)
	if err != nil {
		return 0, err
	}

	handshakeContext, cancelHandshake := context.WithTimeout(baseContext, 20*time.Second)
	defer cancelHandshake()
	conn, response, err := websocket.Dial(handshakeContext, target.String(), &websocket.DialOptions{
		HTTPClient:      c.httpClient,
		HTTPHeader:      headers,
		CompressionMode: websocket.CompressionDisabled,
	})
	if err != nil {
		status := 0
		if response != nil {
			status = response.StatusCode
		}
		_ = c.emit(map[string]any{
			"type":    "ws_error",
			"ws_id":   0,
			"code":    "handshake",
			"status":  status,
			"message": err.Error(),
		})
		return 0, fmt.Errorf("%w: WebSocket handshake failed", ErrUnavailable)
	}
	conn.SetReadLimit(int64(c.cfg.MaxWSFrameBytes))

	sessionContext, cancel := context.WithCancel(baseContext)
	session := &wsSession{
		id:     1,
		core:   c,
		conn:   conn,
		ctx:    sessionContext,
		cancel: cancel,
		out:    make(chan wsOut, defaultSendQueue),
	}
	c.mu.Lock()
	if c.ws != nil {
		c.mu.Unlock()
		cancel()
		conn.Close(websocket.StatusPolicyViolation, "another WebSocket is active")
		return 0, ErrBusy
	}
	c.ws = session
	c.mu.Unlock()

	_ = c.emit(map[string]any{
		"type":  "ws_open",
		"ws_id": session.id,
	})
	go session.readLoop()
	go session.writeLoop()
	return session.id, nil
}

func (c *Core) WSSend(wsID uint64, kind int, data []byte) error {
	if kind != 0 && kind != 1 {
		return fmt.Errorf("%w: WebSocket frame kind must be 0 (text) or 1 (binary)", ErrInvalid)
	}
	if len(data) > c.cfg.MaxWSFrameBytes {
		return ErrTooLarge
	}
	c.mu.Lock()
	session := c.ws
	c.mu.Unlock()
	if session == nil || session.id != wsID {
		return ErrNotFound
	}
	messageType := websocket.MessageText
	if kind == 1 {
		messageType = websocket.MessageBinary
	}
	copied := append([]byte(nil), data...)
	select {
	case session.out <- wsOut{kind: messageType, data: copied}:
		return nil
	default:
		return ErrQueueFull
	}
}

func (c *Core) WSClose(wsID uint64) error {
	c.mu.Lock()
	session := c.ws
	if session == nil || session.id != wsID {
		c.mu.Unlock()
		return ErrNotFound
	}
	c.ws = nil
	c.mu.Unlock()
	session.close(websocket.StatusNormalClosure, "client closed")
	return nil
}

func (s *wsSession) readLoop() {
	for {
		messageType, data, err := s.conn.Read(s.ctx)
		if err != nil {
			code := websocket.CloseStatus(err)
			if code == -1 {
				code = websocket.StatusAbnormalClosure
			}
			s.close(code, "read loop ended")
			return
		}
		if len(data) > s.core.cfg.MaxWSFrameBytes {
			s.close(websocket.StatusMessageTooBig, "frame exceeds limit")
			_ = s.core.emit(map[string]any{
				"type":  "ws_error",
				"ws_id": s.id,
				"code":  "too_large",
			})
			return
		}
		kind := "text"
		if messageType == websocket.MessageBinary {
			kind = "binary"
		}
		if err := s.core.emit(map[string]any{
			"type":        "ws_frame",
			"ws_id":       s.id,
			"kind":        kind,
			"data_base64": base64.StdEncoding.EncodeToString(data),
			"bytes":       len(data),
		}); err != nil {
			s.close(websocket.StatusPolicyViolation, "event queue full")
			_ = s.core.emit(map[string]any{
				"type":  "ws_error",
				"ws_id": s.id,
				"code":  "queue_full",
			})
			return
		}
	}
}

func (s *wsSession) writeLoop() {
	for {
		select {
		case frame := <-s.out:
			if err := s.conn.Write(s.ctx, frame.kind, frame.data); err != nil {
				s.close(websocket.StatusAbnormalClosure, "write failed")
				return
			}
		case <-s.ctx.Done():
			return
		}
	}
}

func (s *wsSession) close(code websocket.StatusCode, reason string) {
	s.once.Do(func() {
		s.cancel()
		_ = s.conn.Close(code, reason)
		_ = s.core.emit(map[string]any{
			"type":   "ws_closed",
			"ws_id":  s.id,
			"code":   int(code),
			"reason": reason,
		})
		s.core.mu.Lock()
		if s.core.ws == s {
			s.core.ws = nil
		}
		s.core.mu.Unlock()
	})
}
