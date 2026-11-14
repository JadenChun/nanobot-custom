package broker

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
)

func testCore(t *testing.T, server *httptest.Server, mutate func(*Config)) *Core {
	t.Helper()
	cfg := Config{
		Origin:          server.URL,
		InstallationID:  "fixture-installation",
		Hostname:        "fixture-host",
		StateDir:        t.TempDir(),
		AllowTestRoutes: true,
		QueueSize:       64,
	}
	if mutate != nil {
		mutate(&cfg)
	}
	core, err := NewTestCore(cfg, func(ctx context.Context, network, address string) (net.Conn, error) {
		return net.Dial("tcp", strings.TrimPrefix(server.URL, "https://"))
	})
	if err != nil {
		t.Fatalf("NewTestCore: %v", err)
	}
	// Trust the httptest certificate exactly as its own client does; production
	// still relies on the system roots and standard hostname/SNI validation.
	testTransport := server.Client().Transport.(*http.Transport)
	coreTransport := core.httpClient.Transport.(*http.Transport)
	coreTransport.TLSClientConfig = testTransport.TLSClientConfig.Clone()
	t.Cleanup(func() { _ = core.Stop() })
	return core
}

func testServer(t *testing.T) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/fixture/v1/bootstrap", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"fixture":true}`))
	})
	mux.HandleFunc("/fixture/v1/read", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("cursor") == "slow" {
			time.Sleep(3 * time.Second)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"items":[]}`))
	})
	mux.HandleFunc("/fixture/v1/counter", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(strings.Repeat("x", 4096)))
	})
	mux.HandleFunc("/fixture/v1/media/sample.png", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Accept-Ranges", "bytes")
		w.Header().Set("Content-Range", "bytes 0-3/8")
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write([]byte("PNG!"))
	})
	mux.HandleFunc("/fixture/v1/redirect", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "https://evil.example/steal", http.StatusFound)
	})
	mux.HandleFunc("/mobile/v1/pair/collect", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Nanobot-Pairing-Id") != "pair-1" ||
			r.Header.Get("X-Nanobot-Pairing-Nonce") != "nonce-1" ||
			r.Header.Get("X-Nanobot-Pairing-Secret") != "secret-1" ||
			r.Header.Get("X-Nanobot-Installation") != "install-1" ||
			r.Header.Get("X-Nanobot-Label") != "phone" {
			http.Error(w, "missing pairing headers", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"pairing_id":"pair-1","comparison_code":"123456"}`))
	})
	mux.HandleFunc("/mobile/v1/pair/code", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Nanobot-Pairing-Code") != "12345678" ||
			r.Header.Get("X-Nanobot-Installation") != "install-1" ||
			r.Header.Get("X-Nanobot-Label") != "phone" {
			http.Error(w, "missing code pairing headers", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"pairing_id":"pair-code"}`))
	})
	mux.HandleFunc("/mobile/v1/refresh", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Nanobot-Installation") != "install-1" ||
			r.Header.Get("X-Nanobot-Refresh-Token") != "refresh-1" {
			http.Error(w, "missing refresh headers", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"access_token":"nbm_at_new","refresh_token":"nbm_rt_new","expires_in":600}`))
	})
	mux.HandleFunc("/fixture/v1/ws", func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		conn.SetReadLimit(37_748_736)
		defer conn.Close(websocket.StatusNormalClosure, "")
		for {
			typ, data, err := conn.Read(r.Context())
			if err != nil {
				return
			}
			if err := conn.Write(r.Context(), typ, data); err != nil {
				return
			}
		}
	})
	server := httptest.NewTLSServer(mux)
	t.Cleanup(server.Close)
	return server
}

func waitEvent(t *testing.T, core *Core, eventType string) map[string]any {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	buffer := make([]byte, 12*1024*1024)
	for time.Now().Before(deadline) {
		n, err := core.PollEvent(buffer)
		if err != nil {
			t.Fatalf("PollEvent: %v", err)
		}
		if n == 0 {
			time.Sleep(10 * time.Millisecond)
			continue
		}
		var event map[string]any
		if err := json.Unmarshal(buffer[:n], &event); err != nil {
			t.Fatalf("event JSON: %v", err)
		}
		if event["type"] == eventType {
			return event
		}
	}
	t.Fatalf("event %q not observed", eventType)
	return nil
}

func TestHTTPReadAndRange(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, nil)

	id, err := core.HTTP(HTTPRequest{
		Operation: "fixture_read",
		Method:    "GET",
		Path:      "/fixture/v1/read?cursor=next",
	})
	if err != nil || id == 0 {
		t.Fatalf("HTTP: id=%d err=%v", id, err)
	}
	event := waitEvent(t, core, "http_response")
	if event["status"].(float64) != 200 {
		t.Fatalf("status: %v", event["status"])
	}
	if event["request_id"].(float64) != float64(id) {
		t.Fatalf("request id: %v", event["request_id"])
	}
	body, err := base64.StdEncoding.DecodeString(event["body_base64"].(string))
	if err != nil || string(body) != `{"items":[]}` {
		t.Fatalf("body: %q err=%v", body, err)
	}

	id, err = core.HTTP(HTTPRequest{
		Operation: "fixture_media",
		Method:    "GET",
		Path:      "/fixture/v1/media/sample.png",
		Headers:   map[string]string{"Range": "bytes=0-3"},
	})
	if err != nil || id == 0 {
		t.Fatalf("media HTTP: id=%d err=%v", id, err)
	}
	event = waitEvent(t, core, "http_response")
	headers := event["headers"].(map[string]any)
	if headers["content-range"] != "bytes 0-3/8" {
		t.Fatalf("content-range: %v", headers)
	}
}

func TestHTTPPolicyDeniesArbitraryTargets(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, nil)

	cases := []HTTPRequest{
		{Operation: "fixture_read", Method: "POST", Path: "/fixture/v1/read"},
		{Operation: "fixture_read", Method: "GET", Path: "/admin"},
		{Operation: "unknown", Method: "GET", Path: "/fixture/v1/read"},
		{Operation: "fixture_read", Method: "GET", Path: "https://evil.example/fixture/v1/read"},
		{Operation: "fixture_read", Method: "GET", Path: "/fixture/v1/read?cursor=next&extra=1"},
		{Operation: "fixture_read", Method: "GET", Path: "/fixture/v1/read", Headers: map[string]string{"X-Fixture-Admin": "fixture-admin"}},
	}
	for index, request := range cases {
		if _, err := core.HTTP(request); err == nil {
			t.Fatalf("case %d was accepted", index)
		}
	}
	if _, err := core.HTTP(HTTPRequest{Operation: "fixture_read", Method: "GET", Path: "/fixture/v1/read", BodyBase64: base64.StdEncoding.EncodeToString(make([]byte, defaultBodyLimit+1))}); err == nil {
		t.Fatal("oversized body was accepted")
	}
}

func TestHTTPResponseTooLarge(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, func(cfg *Config) { cfg.MaxResponseBytes = 32 })
	if _, err := core.HTTP(HTTPRequest{Operation: "fixture_counter", Method: "GET", Path: "/fixture/v1/counter"}); err != nil {
		t.Fatalf("HTTP: %v", err)
	}
	event := waitEvent(t, core, "http_error")
	if event["code"] != "too_large" {
		t.Fatalf("code: %v", event)
	}
}

func TestHTTPCrossOriginRedirectDenied(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, nil)
	if _, err := core.HTTP(HTTPRequest{Operation: "fixture_redirect", Method: "GET", Path: "/fixture/v1/redirect?to=https://evil.example/x"}); err != nil {
		t.Fatalf("HTTP: %v", err)
	}
	event := waitEvent(t, core, "http_error")
	if event["code"] != "redirect_denied" {
		t.Fatalf("code: %v", event)
	}
}

func TestHTTPCancellationAndTLSMismatch(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, nil)
	id, err := core.HTTP(HTTPRequest{Operation: "fixture_read", Method: "GET", Path: "/fixture/v1/read?cursor=slow", TimeoutMS: 5000})
	if err != nil {
		t.Fatalf("HTTP: %v", err)
	}
	time.Sleep(50 * time.Millisecond)
	if err := core.Cancel(id); err != nil {
		t.Fatalf("Cancel: %v", err)
	}
	event := waitEvent(t, core, "http_error")
	if event["code"] != "cancelled" {
		t.Fatalf("code: %v", event)
	}

	mismatch := testCore(t, server, func(cfg *Config) {
		cfg.Origin = strings.Replace(server.URL, "127.0.0.1", "wrong.example", 1)
	})
	if _, err := mismatch.HTTP(HTTPRequest{Operation: "fixture_read", Method: "GET", Path: "/fixture/v1/read"}); err != nil {
		t.Fatalf("HTTP: %v", err)
	}
	event = waitEvent(t, mismatch, "http_error")
	if event["code"] != "tls" {
		t.Fatalf("TLS mismatch code: %v", event)
	}
}

func TestWebSocketEchoAndClose(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, nil)
	wsID, err := core.WSOpen(WSRequest{Operation: "fixture_ws", Path: "/fixture/v1/ws"})
	if err != nil || wsID != 1 {
		t.Fatalf("WSOpen: id=%d err=%v", wsID, err)
	}
	waitEvent(t, core, "ws_open")
	if err := core.WSSend(wsID, 0, []byte("ping")); err != nil {
		t.Fatalf("WSSend: %v", err)
	}
	event := waitEvent(t, core, "ws_frame")
	if event["kind"] != "text" {
		t.Fatalf("kind: %v", event)
	}
	data, err := base64.StdEncoding.DecodeString(event["data_base64"].(string))
	if err != nil || string(data) != "ping" {
		t.Fatalf("frame: %q err=%v", data, err)
	}
	if err := core.WSClose(wsID); err != nil {
		t.Fatalf("WSClose: %v", err)
	}
	waitEvent(t, core, "ws_closed")
}

func TestBoundedQueuesAndBufferRetention(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, func(cfg *Config) { cfg.QueueSize = 1 })
	if err := core.emit(map[string]any{"type": "one"}); err != nil {
		t.Fatalf("emit one: %v", err)
	}
	if err := core.emit(map[string]any{"type": "two"}); err != ErrQueueFull {
		t.Fatalf("second emit: %v", err)
	}
	small := make([]byte, 2)
	if _, err := core.PollEvent(small); err != ErrBufferSmall {
		t.Fatalf("small poll: %v", err)
	}
	buffer := make([]byte, 512)
	n, err := core.PollEvent(buffer)
	if err != nil || n == 0 {
		t.Fatalf("retained poll: n=%d err=%v", n, err)
	}
	if string(buffer[:n]) != `{"type":"one"}` {
		t.Fatalf("retained event: %s", buffer[:n])
	}
}

func TestLifecycleAndLoginURLValidation(t *testing.T) {
	core := &Core{state: "new", events: make(chan []byte, 1)}
	if err := core.Stop(); err != nil {
		t.Fatalf("stop new: %v", err)
	}
	if err := core.Stop(); err != nil {
		t.Fatalf("duplicate stop: %v", err)
	}
	busy := &Core{started: true, stopped: false, events: make(chan []byte, 1)}
	if err := busy.Start(); err != ErrBusy {
		t.Fatalf("duplicate start: %v", err)
	}
	unavailable := &Core{
		cfg:    Config{Origin: "https://fixture.example:8443", InstallationID: "i", Hostname: "h", StateDir: t.TempDir()},
		state:  "starting",
		events: make(chan []byte, 4),
	}
	unavailable.origin, _ = url.Parse(unavailable.cfg.Origin)
	if _, err := unavailable.HTTP(HTTPRequest{Operation: "fixture_read", Method: "GET", Path: "/fixture/v1/read"}); err != ErrUnavailable {
		t.Fatalf("unavailable transport accepted HTTP: %v", err)
	}
	for _, valid := range []string{
		"https://login.tailscale.com/a/abc",
		"https://controlplane.tailscale.com/a/abc",
	} {
		if _, err := validateLoginURL(valid); err != nil {
			t.Fatalf("valid login URL rejected: %s: %v", valid, err)
		}
	}
	for _, invalid := range []string{
		"http://login.tailscale.com/a/abc",
		"https://evil.example/a/abc",
		"https://login.tailscale.com.evil.example/a/abc",
		"https://user@login.tailscale.com/a/abc",
		"https://login.tailscale.com/a/abc#fragment",
	} {
		if _, err := validateLoginURL(invalid); err == nil {
			t.Fatalf("invalid login URL accepted: %s", invalid)
		}
	}
}

func TestWebSocketBoundsAndDuplicateOpen(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, func(cfg *Config) { cfg.MaxWSFrameBytes = 16 })
	wsID, err := core.WSOpen(WSRequest{Operation: "fixture_ws", Path: "/fixture/v1/ws"})
	if err != nil {
		t.Fatalf("WSOpen: %v", err)
	}
	waitEvent(t, core, "ws_open")
	if _, err := core.WSOpen(WSRequest{Operation: "fixture_ws", Path: "/fixture/v1/ws"}); err != ErrBusy {
		t.Fatalf("duplicate WSOpen: %v", err)
	}
	if err := core.WSSend(wsID, 0, make([]byte, 17)); err != ErrTooLarge {
		t.Fatalf("oversize WSSend: %v", err)
	}
	if err := core.WSSend(wsID, 2, []byte("x")); err == nil {
		t.Fatal("invalid frame kind accepted")
	}
	if err := core.WSClose(wsID + 1); err != ErrNotFound {
		t.Fatalf("unknown WSClose: %v", err)
	}

	session := &wsSession{id: 7, core: core, out: make(chan wsOut, 1)}
	core.mu.Lock()
	core.ws = session
	core.mu.Unlock()
	if err := core.WSSend(7, 0, []byte("one")); err != nil {
		t.Fatalf("first queued send: %v", err)
	}
	if err := core.WSSend(7, 0, []byte("two")); err != ErrQueueFull {
		t.Fatalf("send queue bound: %v", err)
	}
	core.mu.Lock()
	core.ws = nil
	core.mu.Unlock()
}

func TestHTTPQueueFull(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, func(cfg *Config) { cfg.QueueSize = 1 })
	if err := core.emit(map[string]any{"type": "filler"}); err != nil {
		t.Fatalf("filler: %v", err)
	}
	if _, err := core.HTTP(HTTPRequest{Operation: "fixture_read", Method: "GET", Path: "/fixture/v1/read"}); err != ErrQueueFull {
		t.Fatalf("HTTP queue bound: %v", err)
	}
}

func TestHTTPPairingRouteAllowsOnlyBoundedHeaders(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, nil)
	id, err := core.HTTP(HTTPRequest{
		Operation: "mobile_pair_collect",
		Method:    http.MethodGet,
		Path:      "/mobile/v1/pair/collect",
		Native:    true,
		Headers: map[string]string{
			"X-Nanobot-Pairing-Id":     "pair-1",
			"X-Nanobot-Pairing-Nonce":  "nonce-1",
			"X-Nanobot-Pairing-Secret": "secret-1",
			"X-Nanobot-Installation":   "install-1",
			"X-Nanobot-Label":          "phone",
		},
	})
	if err != nil || id == 0 {
		t.Fatalf("pairing HTTP: id=%d err=%v", id, err)
	}
	event := waitEvent(t, core, "http_response")
	if event["status"].(float64) != 200 {
		t.Fatalf("status: %v", event["status"])
	}
	body, err := base64.StdEncoding.DecodeString(event["body_base64"].(string))
	if err != nil || !strings.Contains(string(body), `"comparison_code":"123456"`) {
		t.Fatalf("body: %q err=%v", body, err)
	}
	if _, err := core.HTTP(HTTPRequest{
		Operation: "mobile_pair_collect",
		Method:    http.MethodGet,
		Path:      "/mobile/v1/pair/collect",
		Native:    true,
		Headers:   map[string]string{"X-Admin": "1"},
	}); err == nil {
		t.Fatal("unlisted pairing header was accepted")
	}
	id, err = core.HTTP(HTTPRequest{
		Operation: "mobile_pair_code",
		Method:    http.MethodGet,
		Path:      "/mobile/v1/pair/code",
		Native:    true,
		Headers: map[string]string{
			"X-Nanobot-Pairing-Code": "12345678",
			"X-Nanobot-Installation": "install-1",
			"X-Nanobot-Label":        "phone",
		},
	})
	if err != nil || id == 0 {
		t.Fatalf("code pairing HTTP: id=%d err=%v", id, err)
	}
	codeEvent := waitEvent(t, core, "http_response")
	if codeEvent["status"].(float64) != 200 {
		t.Fatalf("code pairing status: %v", codeEvent["status"])
	}
}

func TestWebSocketAllowsAttachmentSizedFrame(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, func(cfg *Config) { cfg.MaxWSFrameBytes = 37_748_736 })
	wsID, err := core.WSOpen(WSRequest{Operation: "fixture_ws", Path: "/fixture/v1/ws"})
	if err != nil {
		t.Fatalf("WSOpen: %v", err)
	}
	waitEvent(t, core, "ws_open")
	payload := make([]byte, 3*1024*1024)
	for i := range payload {
		payload[i] = 'x'
	}
	if err := core.WSSend(wsID, 0, payload); err != nil {
		t.Fatalf("attachment-sized WSSend: %v", err)
	}
	event := waitEvent(t, core, "ws_frame")
	if event["bytes"].(float64) != float64(len(payload)) {
		t.Fatalf("ws frame bytes: %v", event["bytes"])
	}
}

func TestHTTPRefreshRouteUsesBoundedHeaders(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, nil)
	id, err := core.HTTP(HTTPRequest{
		Operation: "mobile_refresh_get",
		Method:    http.MethodGet,
		Path:      "/mobile/v1/refresh",
		Native:    true,
		Headers: map[string]string{
			"X-Nanobot-Installation":  "install-1",
			"X-Nanobot-Refresh-Token": "refresh-1",
		},
	})
	if err != nil || id == 0 {
		t.Fatalf("refresh HTTP: id=%d err=%v", id, err)
	}
	event := waitEvent(t, core, "http_response")
	if event["status"].(float64) != 200 {
		t.Fatalf("status: %v", event["status"])
	}
	body, err := base64.StdEncoding.DecodeString(event["body_base64"].(string))
	if err != nil || !strings.Contains(string(body), `"access_token":"nbm_at_new"`) {
		t.Fatalf("body: %q err=%v", body, err)
	}
}

func TestHTTPCredentialRouteRequiresTrustedNativeProvenance(t *testing.T) {
	requests := make(chan string, 4)
	mux := http.NewServeMux()
	mux.HandleFunc("/mobile/v1/pair/prove", func(w http.ResponseWriter, r *http.Request) {
		requests <- r.Method + " " + r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"access_token":"native-secret"}`))
	})
	mux.HandleFunc("/mobile/v1/handshake", func(w http.ResponseWriter, r *http.Request) {
		requests <- r.Method + " " + r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"mobile_protocol":1}`))
	})
	server := httptest.NewTLSServer(mux)
	defer server.Close()
	core := testCore(t, server, nil)
	proof := HTTPRequest{
		Operation: "mobile_pair_prove",
		Method:    http.MethodGet,
		Path:      "/mobile/v1/pair/prove",
	}
	if id, err := core.HTTP(proof); id != 0 || !errors.Is(err, ErrDenied) {
		t.Fatalf("WebView proof request was not denied: id=%d err=%v", id, err)
	}
	select {
	case request := <-requests:
		t.Fatalf("denied WebView proof reached transport: %s", request)
	default:
	}

	proof.Native = true
	proofID, err := core.HTTP(proof)
	if err != nil || proofID == 0 {
		t.Fatalf("trusted native proof request was denied: id=%d err=%v", proofID, err)
	}
	proofEvent := waitEvent(t, core, "http_response")
	if proofEvent["request_id"].(float64) != float64(proofID) {
		t.Fatalf("proof response ID = %v, want %d", proofEvent["request_id"], proofID)
	}
	proofBody, err := base64.StdEncoding.DecodeString(proofEvent["body_base64"].(string))
	if err != nil || !strings.Contains(string(proofBody), `"access_token":"native-secret"`) {
		t.Fatalf("trusted native proof response = %q, err=%v", proofBody, err)
	}
	if request := <-requests; request != "GET /mobile/v1/pair/prove" {
		t.Fatalf("native proof request = %q", request)
	}

	handshakeID, err := core.HTTP(HTTPRequest{
		Operation: "mobile_handshake",
		Method:    http.MethodGet,
		Path:      "/mobile/v1/handshake",
	})
	if err != nil || handshakeID == 0 {
		t.Fatalf("WebView handshake was denied: id=%d err=%v", handshakeID, err)
	}
	handshakeEvent := waitEvent(t, core, "http_response")
	if handshakeEvent["request_id"].(float64) != float64(handshakeID) {
		t.Fatalf("handshake response ID = %v, want %d", handshakeEvent["request_id"], handshakeID)
	}
	handshakeBody, err := base64.StdEncoding.DecodeString(handshakeEvent["body_base64"].(string))
	if err != nil || !strings.Contains(string(handshakeBody), `"mobile_protocol":1`) {
		t.Fatalf("WebView handshake response = %q, err=%v", handshakeBody, err)
	}
	if request := <-requests; request != "GET /mobile/v1/handshake" {
		t.Fatalf("handshake request = %q", request)
	}
}

func TestWebSocketIDsAreUniqueAcrossConnections(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, nil)
	first, err := core.WSOpen(WSRequest{Operation: "fixture_ws", Path: "/fixture/v1/ws"})
	if err != nil || first != 1 {
		t.Fatalf("first WSOpen: id=%d err=%v", first, err)
	}
	waitEvent(t, core, "ws_open")
	if err := core.WSClose(first); err != nil {
		t.Fatalf("first WSClose: %v", err)
	}
	waitEvent(t, core, "ws_closed")

	second, err := core.WSOpen(WSRequest{Operation: "fixture_ws", Path: "/fixture/v1/ws"})
	if err != nil || second != 2 {
		t.Fatalf("second WSOpen: id=%d err=%v", second, err)
	}
	waitEvent(t, core, "ws_open")
	if err := core.WSClose(first); err != ErrNotFound {
		t.Fatalf("retired WSClose: %v", err)
	}
	if err := core.WSSend(second, 0, []byte("new")); err != nil {
		t.Fatalf("second WSSend: %v", err)
	}
	event := waitEvent(t, core, "ws_frame")
	if event["ws_id"] != float64(second) && event["ws_id"] != second {
		t.Fatalf("frame routed to stale id: %v", event)
	}
	if err := core.WSClose(second); err != nil {
		t.Fatalf("second WSClose: %v", err)
	}
	waitEvent(t, core, "ws_closed")
}

func TestWebSocketPingTracksTheLiveSocket(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, nil)
	wsID, err := core.WSOpen(WSRequest{Operation: "fixture_ws", Path: "/fixture/v1/ws"})
	if err != nil {
		t.Fatalf("WSOpen: %v", err)
	}
	waitEvent(t, core, "ws_open")
	if err := core.WSPing(wsID); err != nil {
		t.Fatalf("WSPing live socket: %v", err)
	}
	if err := core.WSClose(wsID); err != nil {
		t.Fatalf("WSClose: %v", err)
	}
	waitEvent(t, core, "ws_closed")
	if err := core.WSPing(wsID); err != ErrNotFound {
		t.Fatalf("WSPing retired socket: %v", err)
	}
}

func TestQueueFullRetainsTerminalResult(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, func(cfg *Config) { cfg.QueueSize = 1 })
	if err := core.emit(map[string]any{"type": "filler"}); err != nil {
		t.Fatalf("filler: %v", err)
	}
	if err := core.emit(map[string]any{
		"type":       "http_response",
		"request_id": 7,
		"status":     200,
	}); err != nil {
		t.Fatalf("terminal emit: %v", err)
	}
	buffer := make([]byte, 512)
	n, err := core.PollEvent(buffer)
	if err != nil || n == 0 {
		t.Fatalf("terminal poll: n=%d err=%v", n, err)
	}
	if string(buffer[:n]) != `{"request_id":7,"status":200,"type":"http_response"}` {
		t.Fatalf("terminal event: %s", buffer[:n])
	}
}

func TestTerminalBudgetHoldsEveryAdmittedResult(t *testing.T) {
	release := make(chan struct{})
	releaseResponses := sync.OnceFunc(func() { close(release) })
	written := make(chan struct{}, 5)
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-release
		marker := r.URL.Query().Get("cursor")
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Header().Set("ETag", `"fixture-`+marker+`"`)
		_, _ = w.Write(bytes.Repeat([]byte(marker), 8<<20))
		written <- struct{}{}
	}))
	defer server.Close()
	defer releaseResponses()
	core := testCore(t, server, func(cfg *Config) {
		cfg.QueueSize = 1
		cfg.MaxRequests = 4
	})
	expected := make(map[uint64]byte)
	for _, marker := range []byte("abcd") {
		id, err := core.HTTP(HTTPRequest{
			Operation: "fixture_read",
			Method:    http.MethodGet,
			Path:      "/fixture/v1/read?cursor=" + string(marker),
		})
		if err != nil {
			t.Fatalf("admit %q: %v", marker, err)
		}
		expected[id] = marker
	}
	refused, err := core.HTTP(HTTPRequest{
		Operation: "fixture_read",
		Method:    http.MethodGet,
		Path:      "/fixture/v1/read?cursor=e",
	})
	if refused != 0 || err != ErrQueueFull {
		t.Fatalf("expected synchronous refusal, got id=%d err=%v", refused, err)
	}
	releaseResponses()
	deadline := time.After(10 * time.Second)
	for range expected {
		select {
		case <-written:
		case <-deadline:
			t.Fatal("fixture responses did not complete")
		}
	}
	buffer := make([]byte, 12<<20)
	seen := map[uint64]bool{}
	for len(seen) < len(expected) {
		n, err := core.PollEvent(buffer)
		if err != nil {
			t.Fatalf("poll: %v", err)
		}
		if n == 0 {
			select {
			case <-deadline:
				t.Fatalf("terminal results did not settle: %v", seen)
			case <-time.After(10 * time.Millisecond):
			}
			continue
		}
		var event struct {
			Type      string            `json:"type"`
			RequestID uint64            `json:"request_id"`
			Headers   map[string]string `json:"headers"`
			Body      string            `json:"body_base64"`
		}
		if err := json.Unmarshal(buffer[:n], &event); err != nil {
			t.Fatalf("decode: %v", err)
		}
		marker, admitted := expected[event.RequestID]
		if !admitted || seen[event.RequestID] || event.Type != "http_response" {
			t.Fatalf("unexpected terminal result: id=%d type=%s", event.RequestID, event.Type)
		}
		body, err := base64.StdEncoding.DecodeString(event.Body)
		if err != nil || len(body) != 8<<20 || body[0] != marker || event.Headers["etag"] != `"fixture-`+string(marker)+`"` {
			t.Fatalf("wrong response for request %d", event.RequestID)
		}
		seen[event.RequestID] = true
	}
	if n, err := core.PollEvent(buffer); n != 0 || err != nil {
		t.Fatalf("extra terminal result: n=%d err=%v", n, err)
	}
}

func TestHTTPRequestCountIsBounded(t *testing.T) {
	server := testServer(t)
	core := testCore(t, server, func(cfg *Config) { cfg.MaxRequests = 1 })
	core.mu.Lock()
	core.requests[99] = func() {}
	core.mu.Unlock()
	if _, err := core.HTTP(HTTPRequest{
		Operation: "fixture_read",
		Method:    http.MethodGet,
		Path:      "/fixture/v1/read",
	}); err != ErrQueueFull {
		t.Fatalf("request count bound: %v", err)
	}
}
