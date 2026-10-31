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
		return ErrQueueFull
	}
}

// A tiny mutex wrapper keeps the event queue logic readable without exposing
// the queue to callers.
type eventQueue struct {
	mu sync.Mutex
}
