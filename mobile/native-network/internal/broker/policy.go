package broker

import (
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strings"
)

type route struct {
	method   string
	pattern  *regexp.Regexp
	queries  map[string]bool
	ws       bool
	testOnly bool
}

var routes = map[string]route{
	"fixture_bootstrap": {
		method:  http.MethodPost,
		pattern: regexp.MustCompile(`^/fixture/v1/bootstrap$`),
	},
	"remote_health": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/remote/v1/health$`),
	},
	"remote_status": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/remote/v1/status$`),
	},
	"remote_refresh": {
		method:  http.MethodPost,
		pattern: regexp.MustCompile(`^/remote/v1/refresh$`),
	},
	"remote_devices": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/remote/v1/devices$`),
	},
	"remote_device_revoke": {
		method:  http.MethodPost,
		pattern: regexp.MustCompile(`^/remote/v1/devices/[A-Za-z0-9._-]{1,128}/revoke$`),
	},
	"remote_pair_collect": {
		method:  http.MethodPost,
		pattern: regexp.MustCompile(`^/remote/v1/pair/collect$`),
	},
	"remote_pair_poll": {
		method:  http.MethodPost,
		pattern: regexp.MustCompile(`^/remote/v1/pair/poll$`),
	},
	"mobile_handshake": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/handshake$`),
	},
	"mobile_read": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/read$`),
	},
	"mobile_model_settings": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/model-settings$`),
	},
	"mobile_workspaces": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/workspaces$`),
	},
	"mobile_api": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/api$`),
		queries: map[string]bool{"path": true},
	},
	"mobile_dispatch": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/dispatch$`),
		queries: map[string]bool{"path": true},
	},
	"mobile_feature": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/feature$`),
		queries: map[string]bool{"path": true},
	},
	"mobile_thread": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/sessions/[^/]+/thread$`),
		queries: map[string]bool{
			"limit":     true,
			"direction": true,
			"before":    true,
		},
	},
	"mobile_media": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/media/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$`),
	},
	"mobile_ws": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/ws$`),
		ws:      true,
	},
	"mobile_pair_collect": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/pair/collect$`),
	},
	"mobile_pair_poll": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/pair/poll$`),
	},
	"mobile_pair_code": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/pair/code$`),
	},
	"mobile_pair_prove": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/pair/prove$`),
	},
	"mobile_pair_recover": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/pair/recover$`),
	},
	"mobile_pair_recover_complete": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/pair/recover/complete$`),
	},
	"mobile_refresh_get": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/mobile/v1/refresh$`),
	},
	"mobile_refresh": {
		method:  http.MethodPost,
		pattern: regexp.MustCompile(`^/mobile/v1/refresh$`),
	},
	"mobile_mutation": {
		method:  http.MethodPost,
		pattern: regexp.MustCompile(`^/mobile/v1/mutation$`),
	},
	"mobile_logout": {
		method:  http.MethodPost,
		pattern: regexp.MustCompile(`^/mobile/v1/logout$`),
	},
	"fixture_read": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/fixture/v1/read$`),
		queries: map[string]bool{"cursor": true},
	},
	"fixture_refresh": {
		method:  http.MethodPost,
		pattern: regexp.MustCompile(`^/fixture/v1/refresh$`),
	},
	"fixture_mutation": {
		method:  http.MethodPost,
		pattern: regexp.MustCompile(`^/fixture/v1/mutation$`),
	},
	"fixture_counter": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/fixture/v1/counter$`),
	},
	"fixture_media": {
		method:  http.MethodGet,
		pattern: regexp.MustCompile(`^/fixture/v1/media/[A-Za-z0-9._-]{1,64}$`),
	},
	"fixture_attachment": {
		method:  http.MethodPost,
		pattern: regexp.MustCompile(`^/fixture/v1/attachment$`),
	},
	"fixture_redirect": {
		method:   http.MethodGet,
		pattern:  regexp.MustCompile(`^/fixture/v1/redirect$`),
		queries:  map[string]bool{"to": true},
		testOnly: true,
	},
	"fixture_ws": {
		method:   http.MethodGet,
		pattern:  regexp.MustCompile(`^/fixture/v1/ws$`),
		ws:       true,
		testOnly: true,
	},
}

var allowedHeaders = map[string]bool{
	"authorization":              true,
	"content-type":               true,
	"accept":                     true,
	"range":                      true,
	"if-none-match":              true,
	"x-fixture-bootstrap":        true,
	"x-nanobot-pairing-id":       true,
	"x-nanobot-pairing-nonce":    true,
	"x-nanobot-pairing-secret":   true,
	"x-nanobot-pairing-code":     true,
	"x-nanobot-code":             true,
	"x-nanobot-installation":     true,
	"x-nanobot-label":            true,
	"x-nanobot-refresh-token":    true,
	"x-nanobot-device-key":       true,
	"x-nanobot-device-signature": true,
	"x-nanobot-proof-claims":     true,
}

func resolveRoute(cfg *Config, origin *url.URL, operation, method, rawPath string) (*url.URL, route, error) {
	selected, ok := routes[operation]
	if !ok {
		return nil, route{}, fmt.Errorf("%w: unknown operation", ErrInvalid)
	}
	if selected.testOnly && !cfg.AllowTestRoutes {
		return nil, route{}, fmt.Errorf("%w: operation is not enabled", ErrDenied)
	}
	if strings.ToUpper(method) != selected.method {
		return nil, route{}, fmt.Errorf("%w: method does not match operation", ErrInvalid)
	}
	parsed, err := url.ParseRequestURI(rawPath)
	if err != nil || parsed.Scheme != "" || parsed.Host != "" || parsed.User != nil || parsed.Fragment != "" {
		return nil, route{}, fmt.Errorf("%w: path must be an absolute request path", ErrInvalid)
	}
	if !selected.pattern.MatchString(parsed.Path) {
		return nil, route{}, fmt.Errorf("%w: path is not allowlisted", ErrDenied)
	}
	if len(parsed.RawQuery) > 2048 {
		return nil, route{}, fmt.Errorf("%w: query is too large", ErrInvalid)
	}
	for key, values := range parsed.Query() {
		if !selected.queries[key] {
			return nil, route{}, fmt.Errorf("%w: query key is not allowlisted", ErrDenied)
		}
		for _, value := range values {
			if len(value) > 1024 {
				return nil, route{}, fmt.Errorf("%w: query value is too large", ErrInvalid)
			}
		}
	}
	return &url.URL{Scheme: origin.Scheme, Host: origin.Host, Path: parsed.Path, RawQuery: parsed.RawQuery}, selected, nil
}

func sanitizeHeaders(input map[string]string) (http.Header, error) {
	headers := make(http.Header, len(input))
	for key, value := range input {
		lower := strings.ToLower(strings.TrimSpace(key))
		if !allowedHeaders[lower] {
			return nil, fmt.Errorf("%w: header %q is not allowlisted", ErrDenied, key)
		}
		if len(value) > 8192 {
			return nil, fmt.Errorf("%w: header value is too large", ErrInvalid)
		}
		if strings.ContainsAny(value, "\r\n") {
			return nil, fmt.Errorf("%w: header value contains a line break", ErrInvalid)
		}
		headers.Set(lower, value)
		if len(headers) > 16 {
			return nil, fmt.Errorf("%w: too many headers", ErrInvalid)
		}
	}
	return headers, nil
}

func redirectTarget(cfg *Config, origin *url.URL, current *url.URL, location string) (*url.URL, error) {
	next, err := current.Parse(location)
	if err != nil {
		return nil, fmt.Errorf("%w: malformed redirect", ErrDenied)
	}
	if next.Scheme != origin.Scheme || next.Host != origin.Host {
		return nil, fmt.Errorf("%w: cross-origin redirect", ErrDenied)
	}
	if strings.TrimSpace(next.Path) == "" {
		return nil, fmt.Errorf("%w: redirect path is empty", ErrDenied)
	}
	resolved, _, err := resolveRoute(cfg, origin, "fixture_read", http.MethodGet, next.RequestURI())
	if err != nil {
		// A redirect to a non-read route is still safe if the exact GET route is
		// otherwise allowlisted. Probe the route table directly.
		for operation, candidate := range routes {
			if candidate.method != http.MethodGet || candidate.ws || (candidate.testOnly && !cfg.AllowTestRoutes) {
				continue
			}
			if candidate.pattern.MatchString(next.Path) {
				resolved, _, err = resolveRoute(cfg, origin, operation, http.MethodGet, next.RequestURI())
				break
			}
		}
	}
	if err != nil || resolved == nil {
		return nil, fmt.Errorf("%w: redirect target is not allowlisted", ErrDenied)
	}
	return resolved, nil
}

func validateLoginURL(raw string) (string, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.Scheme != "https" || u.Host == "" || u.User != nil || u.Fragment != "" {
		return "", ErrLoginURL
	}
	switch u.Hostname() {
	case "login.tailscale.com", "controlplane.tailscale.com":
	default:
		return "", ErrLoginURL
	}
	if strings.TrimSpace(u.Path) == "" {
		return "", ErrLoginURL
	}
	return u.String(), nil
}
