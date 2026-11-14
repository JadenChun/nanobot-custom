package broker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Media streaming for the packaged WebView.
//
// The WebView cannot load signed media URLs directly on the phone: the signed
// path is served by the gateway over the tailnet, not by the local asset server.
// Instead the Android WebViewClient intercepts /api/media/... and streams the
// body through this API, so images and video are fetched with byte ranges and
// never base64-encoded through the JS bridge.
const (
	mediaCacheBudget  = 512 << 20
	mediaCacheDirName = "media-cache"
	mediaStreamLimit  = 8
)

var mediaPathPattern = regexp.MustCompile(
	`^/mobile/v1/media/([A-Za-z0-9_-]{1,128})/([A-Za-z0-9_-]{1,512})$`,
)

var errMediaStream = errors.New("media stream is not available")

type mediaStream struct {
	body    io.ReadCloser
	cancel  context.CancelFunc
	status  int
	headers http.Header
	closer  func()
	cached  bool
	once    sync.Once
}

// SetMediaToken stores the owner access token used for media requests. It is
// refreshed by the platform layer whenever the access token rotates.
func (c *Core) SetMediaToken(token string) {
	c.mu.Lock()
	c.mediaToken = strings.TrimSpace(token)
	c.mu.Unlock()
}

// OpenMediaStream starts one ranged media request and returns its stream id.
func (c *Core) OpenMediaStream(path, rangeHeader string) (uint64, error) {
	match := mediaPathPattern.FindStringSubmatch(path)
	if match == nil {
		return 0, ErrInvalid
	}
	sig, payload := match[1], match[2]

	if stream, err := c.openCachedMedia(sig, payload, rangeHeader); err == nil && stream != nil {
		return c.storeMediaStream(stream)
	}

	c.mu.Lock()
	client := c.httpClient
	token := c.mediaToken
	ctx := c.ctx
	c.mu.Unlock()
	if client == nil || ctx == nil {
		return 0, ErrUnavailable
	}
	target := &url.URL{
		Scheme: c.origin.Scheme,
		Host:   c.origin.Host,
		Path:   path,
	}
	reqCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	request, err := http.NewRequestWithContext(reqCtx, http.MethodGet, target.String(), nil)
	if err != nil {
		cancel()
		return 0, ErrInvalid
	}
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	if rangeHeader != "" {
		request.Header.Set("Range", rangeHeader)
	}
	response, err := client.Do(request)
	if err != nil {
		cancel()
		return 0, fmt.Errorf("%w: media request failed", ErrUnavailable)
	}
	stream := &mediaStream{
		body:    response.Body,
		cancel:  cancel,
		status:  response.StatusCode,
		headers: response.Header.Clone(),
	}
	if response.StatusCode == http.StatusOK && rangeHeader == "" {
		stream.body, stream.closer = c.cachingBody(sig, payload, response.Body, response.Header)
	}
	return c.storeMediaStream(stream)
}

// MediaStreamInfo reports the upstream status and response headers.
func (c *Core) MediaStreamInfo(id uint64) map[string]any {
	c.mu.Lock()
	stream := c.mediaStreams[id]
	c.mu.Unlock()
	if stream == nil {
		return nil
	}
	info := map[string]any{
		"status":  stream.status,
		"cached":  stream.cached,
		"headers": map[string]string{},
	}
	headers := info["headers"].(map[string]string)
	for _, key := range []string{
		"Content-Type", "Content-Length", "Content-Range", "Accept-Ranges",
		"ETag", "Cache-Control", "Last-Modified",
	} {
		if value := stream.headers.Get(key); value != "" {
			headers[key] = value
		}
	}
	return info
}

// ReadMediaStream copies up to len(dst) bytes. It returns 0 at EOF and a
// negative value on error.
func (c *Core) ReadMediaStream(id uint64, dst []byte) int {
	c.mu.Lock()
	stream := c.mediaStreams[id]
	c.mu.Unlock()
	if stream == nil {
		return -1
	}
	n, err := stream.body.Read(dst)
	if n > 0 {
		return n
	}
	if err == nil {
		return 0
	}
	if errors.Is(err, io.EOF) {
		return 0
	}
	return -1
}

// CloseMediaStream releases one stream and finalizes its cache entry.
func (c *Core) CloseMediaStream(id uint64) int {
	c.mu.Lock()
	stream := c.mediaStreams[id]
	delete(c.mediaStreams, id)
	c.mu.Unlock()
	if stream == nil {
		return -1
	}
	stream.once.Do(func() {
		_ = stream.body.Close()
		if stream.closer != nil {
			stream.closer()
		}
		if stream.cancel != nil {
			stream.cancel()
		}
	})
	return 0
}

// PurgeMediaCache removes every cached media file.
func (c *Core) PurgeMediaCache() int {
	root := c.mediaCacheRoot()
	if root == "" {
		return -1
	}
	if err := os.RemoveAll(root); err != nil {
		return -1
	}
	return 0
}

func (c *Core) storeMediaStream(stream *mediaStream) (uint64, error) {
	c.mu.Lock()
	if len(c.mediaStreams) >= mediaStreamLimit {
		c.mu.Unlock()
		_ = stream.body.Close()
		if stream.cancel != nil {
			stream.cancel()
		}
		return 0, ErrBusy
	}
	c.nextMediaID++
	id := c.nextMediaID
	c.mediaStreams[id] = stream
	c.mu.Unlock()
	return id, nil
}

func (c *Core) mediaCacheRoot() string {
	stateDir := strings.TrimSpace(c.cfg.StateDir)
	if stateDir == "" {
		return ""
	}
	return filepath.Join(stateDir, mediaCacheDirName)
}

func (c *Core) mediaCachePath(sig, payload string) string {
	root := c.mediaCacheRoot()
	if root == "" {
		return ""
	}
	return filepath.Join(root, sig, payload)
}

// openCachedMedia serves a previously cached body, honouring one byte range.
func (c *Core) openCachedMedia(sig, payload, rangeHeader string) (*mediaStream, error) {
	path := c.mediaCachePath(sig, payload)
	if path == "" {
		return nil, errMediaStream
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, errMediaStream
	}
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		_ = file.Close()
		return nil, errMediaStream
	}
	meta := readMediaMeta(path)
	status := http.StatusOK
	headers := http.Header{}
	if meta != nil {
		for key, value := range meta.Headers {
			headers.Set(key, value)
		}
	}
	headers.Set("Accept-Ranges", "bytes")
	if headers.Get("Content-Type") == "" {
		headers.Set("Content-Type", mime.TypeByExtension(filepath.Ext(path)))
	}
	start, length, partial, rangeErr := parseSingleRange(rangeHeader, info.Size())
	if rangeErr != nil {
		_ = file.Close()
		return nil, errMediaStream
	}
	if partial {
		if _, err := file.Seek(start, io.SeekStart); err != nil {
			_ = file.Close()
			return nil, errMediaStream
		}
		status = http.StatusPartialContent
		headers.Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, start+length-1, info.Size()))
	} else {
		length = info.Size()
	}
	headers.Set("Content-Length", strconv.FormatInt(length, 10))
	_ = os.Chtimes(path, time.Now(), time.Now())
	body := io.NopCloser(io.LimitReader(file, length))
	return &mediaStream{
		body:    body,
		status:  status,
		headers: headers,
		closer:  func() { _ = file.Close() },
		cached:  true,
	}, nil
}

// cachingBody mirrors one full response into the media cache and finalizes it
// when the reader reaches EOF. Bytes are reserved against the cache budget
// before they are written. If the budget cannot hold the body, the copy stops
// and the caller keeps streaming from the source.
func (c *Core) cachingBody(
	sig, payload string,
	body io.ReadCloser,
	headers http.Header,
) (io.ReadCloser, func()) {
	path := c.mediaCachePath(sig, payload)
	if path == "" {
		return body, nil
	}
	known := false
	var expected int64
	if raw := strings.TrimSpace(headers.Get("Content-Length")); raw != "" {
		if parsed, parseErr := strconv.ParseInt(raw, 10, 64); parseErr == nil && parsed >= 0 {
			known = true
			expected = parsed
		}
	}
	if known && !c.reserveMediaCache(expected) {
		return body, nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		c.releaseMediaCache(expected)
		return body, nil
	}
	temp, err := os.CreateTemp(filepath.Dir(path), payload+".part-*")
	if err != nil {
		c.releaseMediaCache(expected)
		return body, nil
	}
	reader := &cacheTee{
		source:   body,
		sink:     temp,
		core:     c,
		known:    known,
		reserved: expected,
	}
	reader.finalize = func() {
		reader.mu.Lock()
		complete := reader.complete
		abandoned := reader.abandoned
		writeErr := reader.writeErr
		written := reader.written
		reserved := reader.reserved
		reader.mu.Unlock()
		full := !reader.known || written == reserved
		if complete && !abandoned && writeErr == nil && full {
			_ = temp.Sync()
			_ = temp.Close()
			if err := os.Rename(temp.Name(), path); err == nil {
				writeMediaMeta(path, headers)
				c.releaseMediaCache(reader.takeReserved())
				c.trimMediaCache()
				return
			}
		}
		_ = temp.Close()
		_ = os.Remove(temp.Name())
		c.releaseMediaCache(reader.takeReserved())
	}
	return reader, reader.finalize
}

type cacheTee struct {
	source     io.ReadCloser
	sink       *os.File
	core       *Core
	known      bool
	mu         sync.Mutex
	reserved   int64
	written    int64
	complete   bool
	abandoned  bool
	writeErr   error
	finalize   func()
	finalizing sync.Once
}

func (t *cacheTee) Read(p []byte) (int, error) {
	n, err := t.source.Read(p)
	if n > 0 {
		t.absorb(p[:n])
	}
	if errors.Is(err, io.EOF) {
		t.mu.Lock()
		t.complete = true
		t.mu.Unlock()
		t.done()
	}
	return n, err
}

// absorb mirrors one chunk into the sink only after the budget covers it.
func (t *cacheTee) absorb(chunk []byte) {
	size := int64(len(chunk))
	t.mu.Lock()
	skip := t.abandoned || t.writeErr != nil
	t.mu.Unlock()
	if skip {
		return
	}
	if !t.known {
		if !t.core.reserveMediaCache(size) {
			t.stopCaching()
			return
		}
		t.mu.Lock()
		t.reserved += size
		t.mu.Unlock()
	}
	t.mu.Lock()
	if t.known && t.written+size > t.reserved {
		// The declared length was shorter than the body.
		t.mu.Unlock()
		t.stopCaching()
		return
	}
	_, writeErr := t.sink.Write(chunk)
	if writeErr == nil {
		t.written += size
	} else {
		t.writeErr = writeErr
	}
	t.mu.Unlock()
	if writeErr != nil {
		t.stopCaching()
	}
}

// stopCaching abandons the cached copy and releases its reservation. The
// caller keeps streaming every byte from the source.
func (t *cacheTee) stopCaching() {
	t.mu.Lock()
	if t.abandoned {
		t.mu.Unlock()
		return
	}
	t.abandoned = true
	reserved := t.reserved
	t.reserved = 0
	t.mu.Unlock()
	_ = t.sink.Close()
	_ = os.Remove(t.sink.Name())
	if reserved > 0 {
		t.core.releaseMediaCache(reserved)
	}
}

func (t *cacheTee) takeReserved() int64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	reserved := t.reserved
	t.reserved = 0
	return reserved
}

func (t *cacheTee) Close() error {
	err := t.source.Close()
	t.done()
	return err
}

func (t *cacheTee) done() {
	t.finalizing.Do(func() {
		if t.finalize != nil {
			t.finalize()
		}
	})
}

// mediaCacheUsage reports the bytes held by finalized cache bodies. Temp files
// are excluded because in-flight writes are tracked by mediaCacheReserve.
func (c *Core) mediaCacheUsage() int64 {
	root := c.mediaCacheRoot()
	if root == "" {
		return 0
	}
	var total int64
	_ = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info == nil || !info.Mode().IsRegular() {
			return nil
		}
		name := info.Name()
		if strings.HasSuffix(name, ".meta.json") || strings.Contains(name, ".part-") {
			return nil
		}
		total += info.Size()
		return nil
	})
	return total
}

// evictOldestMediaCache removes the least recently used cached body. It reports
// false when nothing can be evicted. The caller holds c.mu.
func (c *Core) evictOldestMediaCache() bool {
	root := c.mediaCacheRoot()
	if root == "" {
		return false
	}
	oldest := ""
	var oldestMod time.Time
	_ = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info == nil || !info.Mode().IsRegular() {
			return nil
		}
		name := info.Name()
		if strings.HasSuffix(name, ".meta.json") || strings.Contains(name, ".part-") {
			return nil
		}
		if oldest == "" || info.ModTime().Before(oldestMod) {
			oldest = path
			oldestMod = info.ModTime()
		}
		return nil
	})
	if oldest == "" {
		return false
	}
	_ = os.Remove(oldest)
	_ = os.Remove(oldest + ".meta.json")
	return true
}

// reserveMediaCache reserves bytes for one body before it is written. It trims
// the oldest cached bodies first. It reports false when the budget cannot hold
// the body, in which case the caller streams without caching.
func (c *Core) reserveMediaCache(n int64) bool {
	if n <= 0 {
		return true
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.mediaCacheBudget <= 0 || n > c.mediaCacheBudget {
		return false
	}
	for c.mediaCacheUsage()+c.mediaCacheReserve+n > c.mediaCacheBudget {
		if !c.evictOldestMediaCache() {
			return false
		}
	}
	c.mediaCacheReserve += n
	return true
}

func (c *Core) releaseMediaCache(n int64) {
	if n <= 0 {
		return
	}
	c.mu.Lock()
	c.mediaCacheReserve -= n
	if c.mediaCacheReserve < 0 {
		c.mediaCacheReserve = 0
	}
	c.mu.Unlock()
}

type mediaMeta struct {
	Headers map[string]string `json:"headers"`
}

func readMediaMeta(path string) *mediaMeta {
	raw, err := os.ReadFile(path + ".meta.json")
	if err != nil {
		return nil
	}
	var meta mediaMeta
	if err := json.Unmarshal(raw, &meta); err != nil {
		return nil
	}
	return &meta
}

func writeMediaMeta(path string, headers http.Header) {
	meta := mediaMeta{Headers: map[string]string{}}
	for _, key := range []string{
		"Content-Type", "ETag", "Cache-Control", "Last-Modified",
	} {
		if value := headers.Get(key); value != "" {
			meta.Headers[key] = value
		}
	}
	raw, err := json.Marshal(meta)
	if err != nil {
		return
	}
	_ = os.WriteFile(path+".meta.json", raw, 0o600)
}

// trimMediaCache enforces the cache budget, evicting the oldest entries first.
func (c *Core) trimMediaCache() {
	root := c.mediaCacheRoot()
	if root == "" {
		return
	}
	type entry struct {
		path string
		size int64
		mod  time.Time
	}
	var entries []entry
	var total int64
	_ = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info == nil || !info.Mode().IsRegular() {
			return nil
		}
		if strings.HasSuffix(path, ".meta.json") {
			return nil
		}
		entries = append(entries, entry{path: path, size: info.Size(), mod: info.ModTime()})
		total += info.Size()
		return nil
	})
	if total <= c.mediaCacheBudget {
		return
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].mod.Before(entries[j].mod) })
	for _, item := range entries {
		if total <= c.mediaCacheBudget {
			break
		}
		_ = os.Remove(item.path)
		_ = os.Remove(item.path + ".meta.json")
		total -= item.size
	}
}

// parseSingleRange returns the offset and length of one byte range request.
func parseSingleRange(header string, size int64) (int64, int64, bool, error) {
	header = strings.TrimSpace(header)
	if header == "" {
		return 0, size, false, nil
	}
	if !strings.HasPrefix(header, "bytes=") || strings.Contains(header, ",") {
		return 0, 0, false, ErrInvalid
	}
	spec := strings.TrimPrefix(header, "bytes=")
	startRaw, endRaw, ok := strings.Cut(spec, "-")
	if !ok {
		return 0, 0, false, ErrInvalid
	}
	if startRaw == "" {
		// Suffix range: last N bytes.
		suffix, err := strconv.ParseInt(strings.TrimSpace(endRaw), 10, 64)
		if err != nil || suffix <= 0 {
			return 0, 0, false, ErrInvalid
		}
		if suffix > size {
			suffix = size
		}
		return size - suffix, suffix, true, nil
	}
	start, err := strconv.ParseInt(strings.TrimSpace(startRaw), 10, 64)
	if err != nil || start < 0 || start >= size {
		return 0, 0, false, ErrInvalid
	}
	end := size - 1
	if strings.TrimSpace(endRaw) != "" {
		parsed, err := strconv.ParseInt(strings.TrimSpace(endRaw), 10, 64)
		if err != nil || parsed < start {
			return 0, 0, false, ErrInvalid
		}
		if parsed < end {
			end = parsed
		}
	}
	return start, end - start + 1, true, nil
}
