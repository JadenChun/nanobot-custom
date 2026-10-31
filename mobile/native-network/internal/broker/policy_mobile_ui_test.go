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
	}
	for _, tc := range cases {
		resolved, route, err := resolveRoute(cfg, origin, tc.operation, http.MethodGet, tc.path)
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
	); err == nil {
		t.Fatal("expected POST to be rejected")
	}
}
