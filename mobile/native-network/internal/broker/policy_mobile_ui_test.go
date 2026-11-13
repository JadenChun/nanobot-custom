package broker

import (
	"net/http"
	"net/url"
	"testing"
)

func TestMobileUIReadRoutesAreAllowlisted(t *testing.T) {
	origin, err := url.Parse("https://fedora.example.test")
	if err != nil {
		t.Fatal(err)
	}
	cfg := &Config{}
	cases := []struct {
		operation string
		path      string
	}{
		{"mobile_model_settings", "/mobile/v1/model-settings"},
		{"mobile_workspaces", "/mobile/v1/workspaces"},
		{"mobile_thread", "/mobile/v1/sessions/websocket%3Achat-1/thread?limit=2&direction=latest"},
		{"mobile_delegated_children", "/mobile/v1/sessions/websocket%3Achat-1/delegated-children?limit=20&before=eyJ2IjoxfQ"},
		{"mobile_api", "/mobile/v1/api?path=%2Fapi%2Fcommands"},
		{"mobile_dispatch", "/mobile/v1/dispatch?path=%2Fapi%2Fsettings"},
	}
	for _, tc := range cases {
		resolved, route, err := resolveRoute(cfg, origin, tc.operation, http.MethodGet, tc.path, false)
		if err != nil {
			t.Fatalf("%s rejected: %v", tc.operation, err)
		}
		if resolved == nil || resolved.Host != origin.Host {
			t.Fatalf("%s resolved to %#v", tc.operation, resolved)
		}
		if route.method != http.MethodGet {
			t.Fatalf("%s method = %s", tc.operation, route.method)
		}
	}
}

func TestMobileThreadRejectsUnknownQuery(t *testing.T) {
	origin, _ := url.Parse("https://fedora.example.test")
	cfg := &Config{}
	if _, _, err := resolveRoute(
		cfg,
		origin,
		"mobile_thread",
		http.MethodGet,
		"/mobile/v1/sessions/websocket%3Achat-1/thread?path=/etc/passwd",
		false,
	); err == nil {
		t.Fatal("expected an unallowlisted query key to be denied")
	}
}

func TestMobileDelegatedChildrenRejectsUnknownQuery(t *testing.T) {
	origin, _ := url.Parse("https://fedora.example.test")
	cfg := &Config{}
	if _, _, err := resolveRoute(
		cfg,
		origin,
		"mobile_delegated_children",
		http.MethodGet,
		"/mobile/v1/sessions/websocket%3Achat-1/delegated-children?limit=2&path=/etc/passwd",
		false,
	); err == nil {
		t.Fatal("expected an unallowlisted query key to be denied")
	}
}

func TestMobileUIReadRoutesRejectWriteMethods(t *testing.T) {
	origin, _ := url.Parse("https://fedora.example.test")
	cfg := &Config{}
	if _, _, err := resolveRoute(
		cfg,
		origin,
		"mobile_model_settings",
		http.MethodPost,
		"/mobile/v1/model-settings",
		false,
	); err == nil {
		t.Fatal("expected POST to be rejected")
	}
}

func TestOwnerControlRoutesAreIndependentlyAllowlisted(t *testing.T) {
	origin, _ := url.Parse("https://fedora.example.test")
	cfg := &Config{}
	cases := []struct {
		operation string
		method    string
		path      string
		native    bool
	}{
		{"remote_status", http.MethodGet, "/remote/v1/status", false},
		{"remote_refresh", http.MethodPost, "/remote/v1/refresh", true},
		{"remote_devices", http.MethodGet, "/remote/v1/devices", false},
		{"remote_device_revoke", http.MethodPost, "/remote/v1/devices/install-0001/revoke", false},
	}
	for _, tc := range cases {
		resolved, route, err := resolveRoute(cfg, origin, tc.operation, tc.method, tc.path, tc.native)
		if err != nil {
			t.Fatalf("%s rejected: %v", tc.operation, err)
		}
		if resolved == nil || route.method != tc.method {
			t.Fatalf("%s resolved to %#v method=%s", tc.operation, resolved, route.method)
		}
	}
	if _, _, err := resolveRoute(
		cfg,
		origin,
		"remote_device_revoke",
		http.MethodPost,
		"/remote/v1/devices/../status/revoke",
		false,
	); err == nil {
		t.Fatal("expected a path escape to be rejected")
	}
}

func TestMobilePairingProofHeadersAndRoutesAreAllowlisted(t *testing.T) {
	headers, err := sanitizeHeaders(map[string]string{
		"X-Nanobot-Device-Key":       "public-key",
		"X-Nanobot-Device-Signature": "signature",
		"X-Nanobot-Proof-Claims":     "claims",
	})
	if err != nil {
		t.Fatalf("proof headers rejected: %v", err)
	}
	if headers.Get("X-Nanobot-Device-Key") != "public-key" {
		t.Fatalf("device key header was not preserved: %#v", headers)
	}

	origin, _ := url.Parse("https://fedora.example.test")
	cfg := &Config{}
	cases := []struct {
		operation string
		path      string
	}{
		{"mobile_pair_prove", "/mobile/v1/pair/prove"},
		{"mobile_pair_recover", "/mobile/v1/pair/recover"},
		{"mobile_pair_recover_complete", "/mobile/v1/pair/recover/complete"},
	}
	for _, tc := range cases {
		resolved, route, err := resolveRoute(cfg, origin, tc.operation, http.MethodGet, tc.path, true)
		if err != nil {
			t.Fatalf("%s rejected: %v", tc.operation, err)
		}
		if resolved == nil || resolved.Host != origin.Host || route.method != http.MethodGet {
			t.Fatalf("%s resolved to %#v via %#v", tc.operation, resolved, route)
		}
	}
}
