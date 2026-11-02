package broker

import (
	"errors"
	"fmt"
	"net/url"
	"strings"
)

const (
	abiVersionValue   = 1
	defaultQueueSize  = 128
	defaultBodyLimit  = 1 << 20
	defaultRespLimit  = 8 << 20
	defaultFrameLimit = 37_748_736
	defaultSendQueue  = 16
	maxRequestTimeout = 120_000
)

var (
	ErrInvalid     = errors.New("invalid request")
	ErrDenied      = errors.New("request denied by policy")
	ErrBusy        = errors.New("operation already in progress")
	ErrNotFound    = errors.New("handle not found")
	ErrQueueFull   = errors.New("bounded queue is full")
	ErrTooLarge    = errors.New("payload exceeds the configured limit")
	ErrUnavailable = errors.New("transport is not ready")
	ErrCancelled   = errors.New("request cancelled")
	ErrBufferSmall = errors.New("output buffer is too small")
	ErrClosed      = errors.New("transport is closed")
	ErrLoginURL    = errors.New("login URL is not an approved Tailscale URL")
)

// Config is the only input accepted by the native transport. It contains no
// arbitrary dial, URL or admin surfaces.
type Config struct {
	Origin         string `json:"origin"`
	InstallationID string `json:"installation_id"`
	Hostname       string `json:"hostname"`
	StateDir       string `json:"state_dir"`
	AuthKey        string `json:"auth_key,omitempty"`
	ControlURL     string `json:"control_url,omitempty"`
	Mode           string `json:"mode,omitempty"`
	StateKeyBase64 string `json:"state_key_base64,omitempty"`

	QueueSize        int `json:"queue_size,omitempty"`
	MaxResponseBytes int `json:"max_response_bytes,omitempty"`
	MaxWSFrameBytes  int `json:"max_ws_frame_bytes,omitempty"`

	// Test-only; never populated from the C ABI or Capacitor config.
	AllowTestRoutes       bool `json:"-"`
	AllowInsecureLoopback bool `json:"-"`
}

func (c *Config) normalize() error {
	c.Origin = strings.TrimSpace(c.Origin)
	if c.Origin == "" {
		return fmt.Errorf("%w: origin is required", ErrInvalid)
	}
	u, err := url.Parse(c.Origin)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return fmt.Errorf("%w: origin is not an absolute URL", ErrInvalid)
	}
	if u.User != nil || u.Fragment != "" || u.RawQuery != "" || (u.Path != "" && u.Path != "/") {
		return fmt.Errorf("%w: origin must not carry credentials, query or path", ErrInvalid)
	}
	if u.Scheme != "https" && !(c.AllowInsecureLoopback && u.Scheme == "http" && isLoopbackHost(u.Hostname())) {
		return fmt.Errorf("%w: origin must use https", ErrInvalid)
	}
	if strings.TrimSpace(c.InstallationID) == "" {
		return fmt.Errorf("%w: installation_id is required", ErrInvalid)
	}
	if c.Mode != "" && c.Mode != "tsnet" {
		return fmt.Errorf("%w: only the tsnet transport is accepted", ErrInvalid)
	}
	if strings.TrimSpace(c.Hostname) == "" {
		return fmt.Errorf("%w: hostname is required", ErrInvalid)
	}
	if strings.TrimSpace(c.StateDir) == "" {
		return fmt.Errorf("%w: state_dir is required", ErrInvalid)
	}
	if c.QueueSize <= 0 {
		c.QueueSize = defaultQueueSize
	}
	if c.MaxResponseBytes <= 0 {
		c.MaxResponseBytes = defaultRespLimit
	}
	if c.MaxWSFrameBytes <= 0 {
		c.MaxWSFrameBytes = defaultFrameLimit
	}
	if c.QueueSize > 1024 {
		return fmt.Errorf("%w: queue_size is too large", ErrInvalid)
	}
	if c.MaxResponseBytes > 16<<20 || c.MaxWSFrameBytes > 41_943_040 {
		return fmt.Errorf("%w: configured limit is too large", ErrInvalid)
	}
	return nil
}

func isLoopbackHost(host string) bool {
	return host == "127.0.0.1" || host == "::1" || host == "localhost"
}
