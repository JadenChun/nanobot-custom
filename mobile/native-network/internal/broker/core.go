package broker

import (
	"context"
	"encoding/base64"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"tailscale.com/client/local"
	"tailscale.com/ipn/ipnstate"
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
	nextWSID    uint64
	ws          *wsSession

	events  chan []byte
	pending []byte
	eventMu sync.Mutex

	terminalMu         sync.Mutex
	terminalEvents     map[uint64][]byte
	terminalOrder      []uint64
	terminalBytes      int
	terminalByteLimit  int
	maxTerminalResults int

	peerMu  sync.Mutex
	peerIPs map[string]string
	peerAt  time.Time

	mediaToken   string
	mediaStreams map[uint64]*mediaStream
	nextMediaID  uint64
}

func NewCore(cfg Config) (*Core, error) {
	if err := cfg.normalize(); err != nil {
		return nil, err
	}
	origin, err := url.Parse(cfg.Origin)
	if err != nil {
		return nil, fmt.Errorf("%w: origin cannot be parsed", ErrInvalid)
	}
	// A terminal event carries base64-encoded response bytes plus JSON
	// overhead. Reserve one terminal slot for each of several maximum
	// responses; admission refuses when all reserved slots are in use.
	perResponse := base64.StdEncoding.EncodedLen(cfg.MaxResponseBytes) + 4096
	terminalLimit := perResponse * 4
	if terminalLimit < 16<<20 {
		terminalLimit = 16 << 20
	}
	if terminalLimit > 64<<20 {
		terminalLimit = 64 << 20
	}
	maxTerminalResults := terminalLimit / perResponse
	if maxTerminalResults < 2 {
		maxTerminalResults = 2
	}
	if maxTerminalResults > cfg.MaxRequests {
		maxTerminalResults = cfg.MaxRequests
	}
	return &Core{
		cfg:                cfg,
		origin:             origin,
		state:              "new",
		requests:           make(map[uint64]context.CancelFunc),
		events:             make(chan []byte, cfg.QueueSize),
		mediaStreams:       make(map[uint64]*mediaStream),
		terminalEvents:     make(map[uint64][]byte),
		terminalByteLimit:  terminalLimit,
		maxTerminalResults: maxTerminalResults,
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
	c.httpClient = newHTTPClient(c.tailnetDial(node))
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
			c.waitForPeers(node)
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

// tailnetDial resolves tailnet names from the node's peer list before dialing.
// tsnet hands names it does not know to the host resolver, which cannot see
// MagicDNS names on Android, so a name-based origin fails with "no such host"
// even while the node itself is running. Dialing the peer address directly
// keeps the original host name for TLS SNI and certificate verification.
func (c *Core) tailnetDial(node *tsnet.Server) DialFunc {
	return func(ctx context.Context, network, address string) (net.Conn, error) {
		host, port, err := net.SplitHostPort(address)
		if err == nil && net.ParseIP(host) == nil {
			if ip, ok := c.peerIP(ctx, node, host); ok {
				address = net.JoinHostPort(ip, port)
			}
		}
		return node.Dial(ctx, network, address)
	}
}

const peerCacheTTL = 30 * time.Second

// waitForPeers waits briefly until the netmap lists other nodes, because the
// peer addresses can arrive after the node reports itself running and the
// first request would otherwise fail name resolution.
func (c *Core) waitForPeers(node *tsnet.Server) {
	deadline := time.Now().Add(10 * time.Second)
	for {
		client, err := node.LocalClient()
		if err == nil {
			if status, statusErr := client.Status(c.ctx); statusErr == nil {
				if peers := peerAddresses(status); len(peers) > 0 {
					c.storePeers(peers)
					return
				}
			}
		}
		if time.Now().After(deadline) || c.ctx.Err() != nil {
			return
		}
		select {
		case <-c.ctx.Done():
			return
		case <-time.After(250 * time.Millisecond):
		}
	}
}

// peerIP maps a tailnet host name to the node's Tailscale address. Results are
// cached briefly so a stalled lookup does not hammer the local client.
func (c *Core) peerIP(ctx context.Context, node *tsnet.Server, host string) (string, bool) {
	name := strings.ToLower(strings.TrimSuffix(host, "."))
	c.peerMu.Lock()
	peers, cached := c.peerIPs, time.Since(c.peerAt) < peerCacheTTL
	c.peerMu.Unlock()
	if cached {
		ip, ok := peers[name]
		return ip, ok
	}
	client, err := node.LocalClient()
	if err != nil {
		return "", false
	}
	status, err := client.Status(ctx)
	if err != nil {
		return "", false
	}
	peers = peerAddresses(status)
	if len(peers) > 0 {
		c.storePeers(peers)
	}
	ip, ok := peers[name]
	return ip, ok
}

func (c *Core) storePeers(peers map[string]string) {
	c.peerMu.Lock()
	c.peerIPs = peers
	c.peerAt = time.Now()
	c.peerMu.Unlock()
}

// peerAddresses indexes the netmap by full name, short name and hostname.
func peerAddresses(status *ipnstate.Status) map[string]string {
	peers := make(map[string]string, len(status.Peer))
	for _, peer := range status.Peer {
		if peer == nil || len(peer.TailscaleIPs) == 0 {
			continue
		}
		ip := peer.TailscaleIPs[0]
		for _, candidate := range peer.TailscaleIPs {
			if candidate.Is4() {
				ip = candidate
				break
			}
		}
		address := ip.String()
		if peer.DNSName != "" {
			fqdn := strings.ToLower(strings.TrimSuffix(peer.DNSName, "."))
			peers[fqdn] = address
			if label, _, found := strings.Cut(fqdn, "."); found {
				peers[label] = address
			}
		}
		if peer.HostName != "" {
			peers[strings.ToLower(peer.HostName)] = address
		}
	}
	return peers
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
