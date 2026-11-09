package broker

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type HTTPRequest struct {
	Operation  string            `json:"operation"`
	Method     string            `json:"method"`
	Path       string            `json:"path"`
	Headers    map[string]string `json:"headers,omitempty"`
	BodyBase64 string            `json:"body_base64,omitempty"`
	TimeoutMS  int               `json:"timeout_ms,omitempty"`
}

func (c *Core) HTTP(input HTTPRequest) (uint64, error) {
	c.mu.Lock()
	state := c.state
	started := c.started
	testMode := c.testMode
	baseContext := c.ctx
	c.mu.Unlock()
	if !testMode && (!started || state != "ready") {
		return 0, ErrUnavailable
	}
	if baseContext == nil {
		baseContext = context.Background()
	}

	target, route, err := resolveRoute(&c.cfg, c.origin, input.Operation, input.Method, input.Path)
	if err != nil {
		return 0, err
	}
	headers, err := sanitizeHeaders(input.Headers)
	if err != nil {
		return 0, err
	}
	var body []byte
	if input.BodyBase64 != "" {
		body, err = base64.StdEncoding.DecodeString(input.BodyBase64)
		if err != nil {
			return 0, fmt.Errorf("%w: body is not base64", ErrInvalid)
		}
	}
	if len(body) > defaultBodyLimit {
		return 0, ErrTooLarge
	}
	if len(c.events) >= cap(c.events) {
		return 0, ErrQueueFull
	}

	timeout := time.Duration(input.TimeoutMS) * time.Millisecond
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	if timeout > maxRequestTimeout*time.Millisecond {
		timeout = maxRequestTimeout * time.Millisecond
	}
	ctx, cancel := context.WithTimeout(baseContext, timeout)
	id := c.nextID()
	c.mu.Lock()
	if len(c.requests) >= c.cfg.MaxRequests {
		c.mu.Unlock()
		cancel()
		return 0, ErrQueueFull
	}
	c.requests[id] = cancel
	c.mu.Unlock()
	go c.doHTTP(ctx, id, target, route, headers, body)
	return id, nil
}

func (c *Core) doHTTP(
	ctx context.Context,
	id uint64,
	target *url.URL,
	route route,
	headers http.Header,
	body []byte,
) {
	defer c.removeRequest(id)
	client := c.httpClient
	if client == nil {
		c.emitHTTPError(id, "unavailable", "HTTP transport is not initialized", 0)
		return
	}

	current := target
	method := route.method
	payload := body
	for hop := 0; hop <= 3; hop++ {
		var reader io.Reader
		if len(payload) > 0 {
			reader = bytes.NewReader(payload)
		}
		request, err := http.NewRequestWithContext(ctx, method, current.String(), reader)
		if err != nil {
			c.emitHTTPError(id, "invalid", "request could not be built", 0)
			return
		}
		request.Header = headers.Clone()

		response, err := client.Do(request)
		if err != nil {
			c.emitHTTPError(id, classifyHTTPError(err), err.Error(), 0)
			return
		}
		if isRedirect(response.StatusCode) {
			location := response.Header.Get("Location")
			if location == "" || method != http.MethodGet || hop == 3 {
				response.Body.Close()
				c.emitHTTPError(id, "redirect_denied", "redirect cannot be followed safely", response.StatusCode)
				return
			}
			next, err := redirectTarget(&c.cfg, c.origin, current, location)
			if err != nil {
				response.Body.Close()
				c.emitHTTPError(id, "redirect_denied", err.Error(), response.StatusCode)
				return
			}
			response.Body.Close()
			current = next
			continue
		}

		limited := io.LimitReader(response.Body, int64(c.cfg.MaxResponseBytes)+1)
		responseBody, readErr := io.ReadAll(limited)
		response.Body.Close()
		if readErr != nil {
			c.emitHTTPError(id, classifyHTTPError(readErr), readErr.Error(), response.StatusCode)
			return
		}
		if len(responseBody) > c.cfg.MaxResponseBytes {
			c.emitHTTPError(id, "too_large", "response body exceeds the configured limit", response.StatusCode)
			return
		}
		c.emit(map[string]any{
			"type":        "http_response",
			"request_id":  id,
			"status":      response.StatusCode,
			"headers":     safeResponseHeaders(response.Header),
			"body_base64": base64.StdEncoding.EncodeToString(responseBody),
			"bytes":       len(responseBody),
		})
		return
	}
	c.emitHTTPError(id, "redirect_denied", "redirect limit exceeded", 0)
}

func (c *Core) removeRequest(id uint64) {
	c.mu.Lock()
	delete(c.requests, id)
	c.mu.Unlock()
}

func (c *Core) emitHTTPError(id uint64, code, message string, status int) {
	_ = c.emit(map[string]any{
		"type":       "http_error",
		"request_id": id,
		"code":       code,
		"message":    message,
		"status":     status,
	})
}

func classifyHTTPError(err error) string {
	switch {
	case errors.Is(err, context.Canceled):
		return "cancelled"
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	case strings.Contains(err.Error(), "x509:"), strings.Contains(err.Error(), "tls:"),
		strings.Contains(err.Error(), "certificate"):
		return "tls"
	default:
		return "network"
	}
}

func isRedirect(status int) bool {
	switch status {
	case 301, 302, 303, 307, 308:
		return true
	default:
		return false
	}
}

func safeResponseHeaders(headers http.Header) map[string]string {
	allowed := map[string]bool{
		"content-type":   true,
		"content-length": true,
		"content-range":  true,
		"accept-ranges":  true,
		"etag":           true,
		"cache-control":  true,
	}
	out := make(map[string]string)
	for key, values := range headers {
		lower := strings.ToLower(key)
		if !allowed[lower] || len(values) == 0 {
			continue
		}
		value := values[0]
		if len(value) > 8192 {
			value = value[:8192]
		}
		out[lower] = value
	}
	return out
}
