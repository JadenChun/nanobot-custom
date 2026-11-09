package broker

import (
	"encoding/json"
	"sync"
)

// PollEvent copies one queued event into out. It returns 0 when the queue is
// empty, the event size when copied, or ErrBufferSmall when out cannot hold the
// next event. A too-small event is retained for the next poll.
func (c *Core) PollEvent(out []byte) (int, error) {
	c.eventMu.Lock()
	defer c.eventMu.Unlock()
	if len(c.pending) > 0 {
		return c.copyPending(out)
	}
	if event, ok := c.popTerminal(); ok {
		c.pending = event
		return c.copyPending(out)
	}
	select {
	case event := <-c.events:
		c.pending = event
		return c.copyPending(out)
	default:
		return 0, nil
	}
}

func (c *Core) copyPending(out []byte) (int, error) {
	if len(c.pending) > len(out) {
		return 0, ErrBufferSmall
	}
	n := copy(out, c.pending)
	c.pending = nil
	return n, nil
}

func (c *Core) emit(event map[string]any) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	select {
	case c.events <- data:
		return nil
	default:
		if id, terminal := terminalRequestID(data); terminal {
			if c.retainTerminal(id, data) {
				return nil
			}
		}
		return ErrQueueFull
	}
}

func terminalRequestID(data []byte) (uint64, bool) {
	var event struct {
		Type      string  `json:"type"`
		RequestID float64 `json:"request_id"`
	}
	if err := json.Unmarshal(data, &event); err != nil {
		return 0, false
	}
	if event.Type != "http_response" && event.Type != "http_error" {
		return 0, false
	}
	if event.RequestID <= 0 || event.RequestID != float64(uint64(event.RequestID)) {
		return 0, false
	}
	return uint64(event.RequestID), true
}

func (c *Core) retainTerminal(id uint64, data []byte) bool {
	c.terminalMu.Lock()
	defer c.terminalMu.Unlock()
	if existing, ok := c.terminalEvents[id]; ok {
		c.terminalBytes -= len(existing)
	}
	if len(data) > c.terminalByteLimit {
		c.removeTerminalLocked(id)
		return false
	}
	if _, exists := c.terminalEvents[id]; !exists {
		c.terminalOrder = append(c.terminalOrder, id)
	}
	c.terminalEvents[id] = data
	c.terminalBytes += len(data)
	for len(c.terminalEvents) > c.cfg.MaxRequests ||
		c.terminalBytes > c.terminalByteLimit {
		if len(c.terminalOrder) == 0 {
			break
		}
		oldest := c.terminalOrder[0]
		c.terminalOrder = c.terminalOrder[1:]
		c.removeTerminalLocked(oldest)
	}
	return true
}

func (c *Core) removeTerminalLocked(id uint64) {
	if existing, ok := c.terminalEvents[id]; ok {
		delete(c.terminalEvents, id)
		c.terminalBytes -= len(existing)
	}
}

func (c *Core) popTerminal() ([]byte, bool) {
	c.terminalMu.Lock()
	defer c.terminalMu.Unlock()
	for len(c.terminalOrder) > 0 {
		id := c.terminalOrder[0]
		c.terminalOrder = c.terminalOrder[1:]
		event, ok := c.terminalEvents[id]
		if !ok {
			continue
		}
		c.removeTerminalLocked(id)
		return event, true
	}
	return nil, false
}

// A tiny mutex wrapper keeps the event queue logic readable without exposing
// the queue to callers.
type eventQueue struct {
	mu sync.Mutex
}
