package broker

import (
	"encoding/json"
)

// PollEvent copies one queued event into out. It returns 0 when the queue is
// empty, the event size when copied, or ErrBufferSmall when out cannot hold the
// next event. A too-small event is retained for the next poll. Retained
// terminal results are drained before the normal queue so an admitted request
// always settles even when the queue was full.
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

// PendingEventSize returns the size of the event retained after a short poll.
func (c *Core) PendingEventSize() int {
	c.eventMu.Lock()
	defer c.eventMu.Unlock()
	return len(c.pending)
}

func (c *Core) copyPending(out []byte) (int, error) {
	if len(c.pending) > len(out) {
		return 0, ErrBufferSmall
	}
	n := copy(out, c.pending)
	c.pending = nil
	return n, nil
}

// terminalKey identifies one admitted request or WebSocket session. HTTP
// terminal events and WebSocket close/error events share the terminal budget
// but never share an identity.
type terminalKey struct {
	ws bool
	id uint64
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
		if key, terminal := terminalEventKey(data); terminal {
			if c.retainTerminal(key, data) {
				return nil
			}
		}
		return ErrQueueFull
	}
}

func terminalEventKey(data []byte) (terminalKey, bool) {
	var event struct {
		Type      string  `json:"type"`
		RequestID float64 `json:"request_id"`
		WSID      float64 `json:"ws_id"`
	}
	if err := json.Unmarshal(data, &event); err != nil {
		return terminalKey{}, false
	}
	switch event.Type {
	case "http_response", "http_error":
		if event.RequestID <= 0 || event.RequestID != float64(uint64(event.RequestID)) {
			return terminalKey{}, false
		}
		return terminalKey{id: uint64(event.RequestID)}, true
	case "ws_closed", "ws_error":
		if event.WSID <= 0 || event.WSID != float64(uint64(event.WSID)) {
			return terminalKey{}, false
		}
		return terminalKey{ws: true, id: uint64(event.WSID)}, true
	default:
		return terminalKey{}, false
	}
}

// retainTerminal stores one bounded terminal result per identity. A repeat
// result for the same identity replaces the earlier one; an admitted result for
// a different identity is never evicted to make room.
func (c *Core) retainTerminal(key terminalKey, data []byte) bool {
	c.terminalMu.Lock()
	defer c.terminalMu.Unlock()
	if existing, ok := c.terminalEvents[key]; ok {
		c.terminalBytes -= len(existing)
	}
	if len(data) > c.terminalByteLimit {
		c.removeTerminalLocked(key)
		return false
	}
	if _, exists := c.terminalEvents[key]; !exists {
		c.terminalOrder = append(c.terminalOrder, key)
	}
	c.terminalEvents[key] = data
	c.terminalBytes += len(data)
	// Admission reserves one terminal slot per request. Never evict an
	// earlier admitted result to store a newer one.
	if c.terminalBytes > c.terminalByteLimit {
		c.removeTerminalLocked(key)
		return false
	}
	return true
}

// hasTerminalCapacityLocked reports whether one more admitted request fits the
// terminal budget. The caller holds c.mu, so the reserved WebSocket slot is
// read safely.
func (c *Core) hasTerminalCapacityLocked() bool {
	c.terminalMu.Lock()
	defer c.terminalMu.Unlock()
	reserved := 0
	if c.ws != nil {
		// One admitted WebSocket always leaves room for its close/error.
		reserved = 1
	}
	return len(c.requests)+len(c.terminalEvents)+reserved < c.maxTerminalResults
}

func (c *Core) removeTerminalLocked(key terminalKey) {
	if existing, ok := c.terminalEvents[key]; ok {
		delete(c.terminalEvents, key)
		c.terminalBytes -= len(existing)
	}
}

func (c *Core) popTerminal() ([]byte, bool) {
	c.terminalMu.Lock()
	defer c.terminalMu.Unlock()
	for len(c.terminalOrder) > 0 {
		key := c.terminalOrder[0]
		c.terminalOrder = c.terminalOrder[1:]
		event, ok := c.terminalEvents[key]
		if !ok {
			continue
		}
		c.removeTerminalLocked(key)
		return event, true
	}
	return nil, false
}
