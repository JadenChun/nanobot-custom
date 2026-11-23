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
	target, route, err := resolveRoute(&c.cfg, c.origin, input.Operation, http.MethodGet, input.Path, false)
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
	c.nextWSID++
	session.id = c.nextWSID
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

// WSPing sends one protocol ping and waits for its pong within five seconds.
// If the ping fails, the dead session is closed so subsequent WSOpen calls do
// not fail with ErrBusy against a zombie socket.
func (c *Core) WSPing(wsID uint64) error {
	c.mu.Lock()
	session := c.ws
	c.mu.Unlock()
	if session == nil || session.id != wsID {
		return ErrNotFound
	}
	ctx, cancel := context.WithTimeout(session.ctx, 5*time.Second)
	defer cancel()
	if err := session.conn.Ping(ctx); err != nil {
		session.close(websocket.StatusAbnormalClosure, "ping failed")
		return fmt.Errorf("%w: WebSocket ping failed", ErrUnavailable)
	}
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
			_ = s.core.emit(map[string]any{
				"type":  "ws_error",
				"ws_id": s.id,
				"code":  "too_large",
			})
			s.close(websocket.StatusMessageTooBig, "frame exceeds limit")
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
			// Report the cause first. The close is the single terminal result
			// reserved for this admitted session, and it replaces the error
			// when the normal queue is full.
			_ = s.core.emit(map[string]any{
				"type":  "ws_error",
				"ws_id": s.id,
				"code":  "queue_full",
			})
			s.close(websocket.StatusPolicyViolation, "event queue full")
			return
		}
	}
}

const (
	wsWriteTimeout      = 15 * time.Second
	wsKeepaliveInterval = 25 * time.Second
	wsKeepaliveTimeout  = 10 * time.Second
)

func (s *wsSession) writeLoop() {
	keepalive := time.NewTicker(wsKeepaliveInterval)
	defer keepalive.Stop()
	for {
		select {
		case frame := <-s.out:
			writeCtx, cancel := context.WithTimeout(s.ctx, wsWriteTimeout)
			err := s.conn.Write(writeCtx, frame.kind, frame.data)
			cancel()
			if err != nil {
				s.close(websocket.StatusAbnormalClosure, "write failed")
				return
			}
		case <-keepalive.C:
			pingCtx, cancel := context.WithTimeout(s.ctx, wsKeepaliveTimeout)
			err := s.conn.Ping(pingCtx)
			cancel()
			if err != nil {
				s.close(websocket.StatusAbnormalClosure, "keepalive ping failed")
				return
			}
		case <-s.ctx.Done():
			return
		}
	}
}

func (s *wsSession) close(code websocket.StatusCode, reason string) {
	s.once.Do(func() {
		// Release c.ws before closing the underlying connection or emitting
		// ws_closed so an immediate reconnect triggered by ws_closed never
		// races with the retiring session and fails with ErrBusy.
		s.core.mu.Lock()
		if s.core.ws == s {
			s.core.ws = nil
		}
		s.core.mu.Unlock()
		s.cancel()
		_ = s.conn.Close(code, reason)
		_ = s.core.emit(map[string]any{
			"type":   "ws_closed",
			"ws_id":  s.id,
			"code":   int(code),
			"reason": reason,
		})
	})
}
