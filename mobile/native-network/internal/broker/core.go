package broker

import (
	"context"
	"encoding/base64"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"sync"
	"time"

	"tailscale.com/client/local"
	"tailscale.com/tsnet"
)

type DialFunc func(ctx context.Context, network, address string) (net.Conn, error)

type Core struct {
	cfg    Config
	origin *url.URL

	mu         sync.Mutex
	state      string
	loginURL   string
	node       *tsnet.Server
	httpClient *http.Client
	dial       DialFunc
	ctx        context.Context
	cancel     context.CancelFunc
	started    bool
	stopped    bool
	testMode   bool

	requests    map[uint64]context.CancelFunc
	nextRequest uint64
	ws          *wsSession

	events  chan []byte
	pending []byte
	eventMu sync.Mutex
}

func NewCore(cfg Config) (*Core, error) {
	if err := cfg.normalize(); err != nil {
		return nil, err
	}
	origin, err := url.Parse(cfg.Origin)
	if err != nil {
		return nil, fmt.Errorf("%w: origin cannot be parsed", ErrInvalid)
	}
	return &Core{
		cfg:      cfg,
		origin:   origin,
		state:    "new",
		requests: make(map[uint64]context.CancelFunc),
		events:   make(chan []byte, cfg.QueueSize),
	}, nil
}

// NewTestCore is available only to Go tests. It uses a caller-supplied dialer
// and never creates a Tailscale node. The C ABI has no path to this function.
func NewTestCore(cfg Config, dial DialFunc) (*Core, error) {
	core, err := NewCore(cfg)
	if err != nil {
		return nil, err
	}
	core.testMode = true
	core.dial = dial
	core.state = "ready"
	core.ctx, core.cancel = context.WithCancel(context.Background())
	core.httpClient = newHTTPClient(dial)
	return core, nil
}

func (c *Core) Start() error {
	c.mu.Lock()
	if c.started && !c.stopped {
		c.mu.Unlock()
		return ErrBusy
	}
	c.mu.Unlock()

	key, err := base64.StdEncoding.DecodeString(c.cfg.StateKeyBase64)
	if err != nil || len(key) != 32 {
		return fmt.Errorf("%w: state_key_base64 must contain a 32-byte key", ErrInvalid)
	}
	store, err := NewEncryptedFileStore(c.cfg.StateDir, key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(c.cfg.StateDir, 0o700); err != nil {
		return err
	}
	// Android has no writable temp/cache default for Tailscale's log state.
	_ = os.Setenv("TS_LOGS_DIR", c.cfg.StateDir)

	c.mu.Lock()
	c.started = true
	c.stopped = false
	c.state = "starting"
	c.ctx, c.cancel = context.WithCancel(context.Background())
	c.mu.Unlock()

	_ = c.emit(map[string]any{"type": "node_state", "state": "starting", "origin": c.cfg.Origin})
	node := &tsnet.Server{
		Dir:        c.cfg.StateDir,
		Store:      store,
		Hostname:   c.cfg.Hostname,
		AuthKey:    c.cfg.AuthKey,
		ControlURL: c.cfg.ControlURL,
		UserLogf:   func(string, ...any) {},
	}
	c.mu.Lock()
	c.node = node
	c.dial = node.Dial
	c.httpClient = newHTTPClient(node.Dial)
	c.mu.Unlock()
	go c.runNode(node)
	return nil
}

func (c *Core) runNode(node *tsnet.Server) {
	upCtx := c.ctx
	upResult := make(chan error, 1)
	go func() {
		_, err := node.Up(upCtx)
		upResult <- err
	}()

	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	var localClient *local.Client
	for {
		select {
		case err := <-upResult:
			if err != nil {
				if c.ctx.Err() != nil {
					c.setState("stopped", "")
				} else {
					c.setStateWithExtra("error", "", map[string]any{"message": err.Error()})
				}
				return
			}
			ips := []string{}
			ip4, ip6 := node.TailscaleIPs()
			if ip4.IsValid() {
				ips = append(ips, ip4.String())
			}
			if ip6.IsValid() {
				ips = append(ips, ip6.String())
			}
			c.setStateWithExtra("ready", "", map[string]any{"tailscale_ips": ips})
			return
		case <-ticker.C:
			if localClient == nil {
				client, err := node.LocalClient()
				if err != nil {
					continue
				}
				localClient = client
			}
			status, err := localClient.Status(c.ctx)
			if err != nil || status.AuthURL == "" {
				continue
			}
			approved, err := validateLoginURL(status.AuthURL)
			if err != nil {
				c.setStateWithExtra("error", "", map[string]any{"message": "unapproved login URL"})
				return
			}
			c.setState("needs_login", approved)
		case <-c.ctx.Done():
			return
		}
	}
}

func (c *Core) setState(state, loginURL string) {
	c.setStateWithExtra(state, loginURL, nil)
}

func (c *Core) setStateWithExtra(state, loginURL string, extra map[string]any) {
	c.mu.Lock()
	if c.state == state && c.loginURL == loginURL {
		c.mu.Unlock()
		return
	}
	c.state = state
	c.loginURL = loginURL
	c.mu.Unlock()

	event := map[string]any{"type": "node_state", "state": state}
	if loginURL != "" {
		event["login_url"] = loginURL
	}
	for key, value := range extra {
		event[key] = value
	}
	_ = c.emit(event)
}

func (c *Core) Status() map[string]any {
	c.mu.Lock()
	defer c.mu.Unlock()
	status := map[string]any{
		"type":            "status",
		"state":           c.state,
		"origin":          c.cfg.Origin,
		"installation_id": c.cfg.InstallationID,
		"queue_depth":     len(c.events),
	}
	if c.loginURL != "" {
		status["login_url"] = c.loginURL
	}
	return status
}

func (c *Core) LoginURL() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.loginURL
}

func (c *Core) Stop() error {
	c.mu.Lock()
	if !c.started || c.stopped {
		c.stopped = true
		c.mu.Unlock()
		return nil
	}
	c.stopped = true
	c.state = "stopping"
	cancel := c.cancel
	node := c.node
	ws := c.ws
	c.ws = nil
	for _, requestCancel := range c.requests {
		requestCancel()
	}
	c.requests = make(map[uint64]context.CancelFunc)
	c.mu.Unlock()

	if ws != nil {
		ws.close(1000, "transport stopped")
	}
	if node != nil {
		_ = node.Close()
	}
	if cancel != nil {
		cancel()
	}
	c.setState("stopped", "")
	return nil
}

func (c *Core) Cancel(requestID uint64) error {
	c.mu.Lock()
	cancel, ok := c.requests[requestID]
	if ok {
		delete(c.requests, requestID)
	}
	c.mu.Unlock()
	if !ok {
		return ErrNotFound
	}
	cancel()
	return nil
}

func (c *Core) nextID() uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.nextRequest++
	return c.nextRequest
}

func newHTTPClient(dial DialFunc) *http.Client {
	transport := &http.Transport{
		DialContext:           dial,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          8,
		IdleConnTimeout:       30 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 20 * time.Second,
	}
	return &http.Client{
		Transport: transport,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}
