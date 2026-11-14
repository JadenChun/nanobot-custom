package broker

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// testMediaServer serves one full media body at the path OpenMediaStream uses.
// withLength selects a declared Content-Length or a chunked body of unknown
// length; both must stay within the reserved cache budget.
func testMediaServer(t *testing.T, size int64, withLength bool) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/mobile/v1/media/sig/payload-image", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		if withLength {
			w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
		}
		chunk := bytes.Repeat([]byte{0xAB}, 4096)
		var written int64
		for written < size {
			n := int64(len(chunk))
			if remaining := size - written; remaining < n {
				n = remaining
			}
			if _, err := w.Write(chunk[:n]); err != nil {
				return
			}
			written += n
			if !withLength {
				if flusher, ok := w.(http.Flusher); ok {
					flusher.Flush()
				}
			}
		}
	})
	server := httptest.NewTLSServer(mux)
	t.Cleanup(server.Close)
	return server
}

// mediaCacheBytes reports the bytes held by cached bodies, excluding metadata.
func mediaCacheBytes(t *testing.T, stateDir string) int64 {
	t.Helper()
	root := filepath.Join(stateDir, mediaCacheDirName)
	var total int64
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info == nil || info.IsDir() {
			return nil
		}
		if strings.HasSuffix(path, ".meta.json") {
			return nil
		}
		total += info.Size()
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		t.Fatalf("walk cache: %v", err)
	}
	return total
}

func mediaTempFiles(t *testing.T, stateDir string) int {
	t.Helper()
	root := filepath.Join(stateDir, mediaCacheDirName)
	count := 0
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info == nil || info.IsDir() {
			return nil
		}
		if strings.Contains(info.Name(), ".part-") {
			count++
		}
		return nil
	})
	if err != nil && !os.IsNotExist(err) {
		t.Fatalf("walk cache: %v", err)
	}
	return count
}

// Requirement (A3-7 AC7.3): cache pressure stays within the reserved bytes.
// A body larger than the remaining budget must still stream every byte, must
// not write past the budget, and must not leave a partial cache file behind.
func TestMediaCacheStaysWithinBudgetWhileStreaming(t *testing.T) {
	const payloadSize = 512 << 10
	cases := []struct {
		name       string
		withLength bool
	}{
		{name: "declared-length", withLength: true},
		{name: "unknown-length", withLength: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := testMediaServer(t, payloadSize, tc.withLength)
			core := testCore(t, server, nil)
			core.mediaCacheBudget = 128 << 10
			budget := core.mediaCacheBudget

			id, err := core.OpenMediaStream("/mobile/v1/media/sig/payload-image", "")
			if err != nil {
				t.Fatalf("OpenMediaStream: %v", err)
			}
			buffer := make([]byte, 8<<10)
			var delivered int
			for delivered < payloadSize/2 {
				n := core.ReadMediaStream(id, buffer)
				if n < 0 {
					t.Fatalf("ReadMediaStream failed after %d bytes", delivered)
				}
				if n == 0 {
					break
				}
				delivered += n
				if used := mediaCacheBytes(t, core.cfg.StateDir); used > budget {
					t.Fatalf("cache held %d bytes mid-stream, budget %d", used, budget)
				}
			}
			for {
				n := core.ReadMediaStream(id, buffer)
				if n < 0 {
					t.Fatalf("ReadMediaStream failed after %d bytes", delivered)
				}
				if n == 0 {
					break
				}
				delivered += n
			}
			if delivered != payloadSize {
				t.Fatalf("stream delivered %d bytes, want %d", delivered, payloadSize)
			}
			if used := mediaCacheBytes(t, core.cfg.StateDir); used > budget {
				t.Fatalf("cache held %d bytes, budget %d", used, budget)
			}
			if err := core.CloseMediaStream(id); err != 0 {
				t.Fatalf("CloseMediaStream: %d", err)
			}
			if used := mediaCacheBytes(t, core.cfg.StateDir); used > budget {
				t.Fatalf("cache held %d bytes after close, budget %d", used, budget)
			}
			if parts := mediaTempFiles(t, core.cfg.StateDir); parts != 0 {
				t.Fatalf("%d partial cache files left behind", parts)
			}
		})
	}
}
