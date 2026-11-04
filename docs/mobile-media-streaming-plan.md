# Mobile large-payload and media streaming plan

Status: implemented; on-device verification pending (see §11).

## 1. Problem

A single mobile session could not be opened: `@kevu` (`websocket:c78c5160-…`,
95 messages). Measured on the current staging build:

| Measurement | Value |
|---|---|
| Thread JSON projection (`limit=80&direction=latest`) | **30.5 MiB** |
| Of that, `toolEvents` | 30.4 MiB |
| Largest tool event | 5.5 MiB (`read_file` returning a 3 MiB base64 PNG) |
| Transcript segment files for that session | **103 MiB** |
| Phone response guard (`MOBILE_API_MAX_RESPONSE_BYTES`) | 2 MiB → HTTP 413 |
| Native broker body buffer (`MaxResponseBytes`) | 8 MiB |

The desktop parsed the 30 MiB payload slowly; the phone rejected it and the
session never rendered. The same over-large history is also why that session
reports "Context compaction failed".

Interim mitigation shipped: tool-event results are redacted/truncated in the
thread projection (`23aafaf9`), bringing the same payload to **238 KiB** and
letting the phone render the session. That is a stop-gap, not the architecture.

## 2. Goals

1. A session is always viewable on every client, regardless of how much the
   agent produced: long answers, many attachments, large tool output.
2. Media (images, audio, video, attachments) loads lazily, progressively and
   with byte-range seek on the phone — not as base64 through the JS bridge.
3. Large files transfer in chunks with progress, without a whole-body buffer.
4. The response cap stops being product behaviour. Remaining limits are memory
   guards with a clear failure mode, never a reason a session cannot open.
5. The desktop and the phone render identical content from the same projection
   rules.

Non-goals for this plan: client-side caching architecture, offline mode,
encryption-at-rest changes, thumbnail generation service.

## 3. Current design and why the caps exist

Three limits guard a transport that buffers whole bodies:

- `nanobot/remote/edge.py`: `MOBILE_API_MAX_RESPONSE_BYTES = 2 MiB`.
- `mobile/native-network/internal/broker/config.go`: `MaxResponseBytes = 8 MiB`.
- `mobile/native-network/internal/broker/config.go`: `MaxWSFrameBytes ≈ 37 MB`.

`broker/http.go` reads the whole body (`io.ReadAll`), base64-encodes it into one
`http_response` event, and the JS side decodes it again. Peak memory is roughly
4–6× the payload, which is why unbounded responses are unsafe. The caps are
consequences of that buffering, not design choices about content.

Media today goes through the same buffered path: `webui/src/lib/mobile-media.ts`
fetches the entire signed media body, base64-decodes it in JS and turns it into
a blob URL. This also caps media at 8 MiB (`MAX_MEDIA_RESPONSE_BYTES`) and rules
out range requests, progressive video and native caching.

## 4. Target architecture

Two planes, with different rules.

### Control plane (JSON)

- One server-side projection. The same JSON goes to the desktop and the phone;
  there is no per-client shaping and no mobile-specific variant.
- Thread/message JSON carries **references** for bulk that the UI does not
  render (raw tool blobs, inline base64), never those bytes themselves.
- Content the UI *does* render is always shipped in full: message text, media,
  trace lines, and the fields the UI reads from tool results
  (`webSearchSources`, `describeTraceLine`). If such content is genuinely too
  large, it is paged or fetched on demand by both clients — never replaced by a
  truncated variant for one client.
- History stays paged: `limit` + `direction=latest` + `before` (already
  implemented in `webui/src/hooks/useSessions.ts`).
- JSON is never chunked: clients buffer before parsing, so chunking buys nothing
  here.

### Binary plane (media, attachments, artifacts)

- The WebView loads media URLs **directly** over HTTP with `Range` support, so
  `<img>`/`<video>` render lazily and seek progressively with disk caching.
- The bytes stream through the embedded tsnet transport without a whole-body
  buffer and without base64 crossing the JS bridge.
- Large uploads use the same streaming channel.

### Mechanism decision

Use a **loopback streaming proxy inside the Go broker** rather than
`WebViewClient.shouldInterceptRequest` or a JNI read API:

- No WebView internals to subclass (Capacitor owns `BridgeWebViewClient`).
- Native HTTP semantics (Range, ETag, caching) come for free from the WebView.
- One JNI surface (return the base URL), no per-chunk JNI traffic.
- The proxy can only reach the allowlisted media route through the tsnet dialer.

Security: bind `127.0.0.1` only, require a per-process random path token, accept
only `GET`/`HEAD` for `/media/<sig>/<payload>`, forward only allowlisted headers,
and never proxy arbitrary URLs.

## 5. Work packages

### P2 — Projection budget, never 413 (server)

- Keep one budget for every client (`PROJECTION_BUDGET_BYTES`, target ≈ 1.5 MiB).
  When exceeded, reduce the page (fewer turns) and/or drop non-rendered bulk —
  never rendered content — so the response stays identical for everyone.
- `nanobot/webui/ws_http.py` / `transcript.py`: return the bounded payload with
  `200`; never `413` for a normal session.
- The mobile paging request is the same as the desktop's (same shape, optionally
  a smaller initial `limit`), so a smaller page never means different content.
- Acceptance: every existing session projects under the budget; a synthetic
  session with 500 turns still opens and pages.

### P3 — Streaming media proxy (native)

- `mobile/native-network/internal/broker/`: add a loopback `net/http` server
  started with the core; handlers proxy `GET|HEAD /<token>/media/{sig}/{payload}`
  through the existing dialer with `io.Copy`, forwarding `Range`, `If-Range`,
  `Content-Length`, `Content-Range`, `Accept-Ranges`, `Content-Type`, `ETag`,
  `Cache-Control`.
- `cmd/lib/main.go`: expose `nanobot_network_media_base(handle, out, cap)` →
  `{"base_url": "http://127.0.0.1:<port>/<token>"}`.
- `NativeNetworkPlugin.kt`: `@PluginMethod fun mediaBase(call)`; include the base
  URL in `status`/`handshake` so the WebUI can use it without an extra round trip.
- Acceptance: `curl -r 0-1023 <base>/media/...` returns `206` with the range;
  a 50 MiB video streams with flat RSS on the phone.

### P4 — Client media path (WebUI)

- `webui/src/lib/mobile-runtime.ts`: expose `mediaBaseUrl`.
- `webui/src/lib/mobile-media.ts`: return
  `${mediaBaseUrl}/media/${sig}/${payload}` directly instead of fetching a blob.
- Remove the blob/base64 path and `MaxResponseBytes` pressure from media.
- Acceptance: opening `@kevu` on the phone loads images natively; no
  `mobile_media` HTTP request appears in logcat; `<video>` seeks.

### P5 — Limits become guards (server + native)

- After P2, raise `MOBILE_API_MAX_RESPONSE_BYTES` to a safety ceiling
  (e.g. 64 MiB) and emit a structured `response_too_large` only for genuinely
  pathological payloads; the client shows a retry/paging hint instead of a blank
  thread.
- Media no longer uses the buffered path, so `MAX_MEDIA_RESPONSE_BYTES` stops
  bounding playback; keep only a per-request proxy guard.
- Acceptance: no session in the current data set reaches the ceiling; a forced
  64 MiB+ payload fails cleanly with a message, not an OOM.

### P8 — On-device media cache

Separate change on top of P3/P4 (own tests, own rollback), because it addresses
bandwidth/latency rather than memory, and because it carries a security
lifecycle of its own.

- Cache location: native, beside the transport state (not the WebView HTTP cache,
  which is small, evictable and not guaranteed across restarts).
- Key: the request path including the signature (`/media/{sig}/{payload}`). The
  signature is an HMAC over the relative media path, so identical files map to
  identical keys and the entry is immutable by construction.
- Policy: honour the existing `Cache-Control: private, max-age=31536000,
  immutable` plus `ETag`; revalidate with `If-None-Match` and accept `304`; LRU
  eviction with a configurable budget (default ~512 MiB) and a byte counter.
- Purge: hook `CustodyStore.purgeMedia()` (already called on logout and custody
  erase) so a revoked or unpaired device never keeps media on disk.
- Metrics: hit ratio and bytes served from cache, exposed in the diagnostics
  payload so we can tune the budget with real session data.
- Acceptance: reopening `@kevu` transfers zero media bytes on the second visit;
  a cache hit renders from disk with no network request in the gateway log; an
  unpair or logout empties the cache directory.

Without P8 the WebView's HTTP cache already avoids some repeats, but it is not a
guarantee ("loads once" needs the native cache).

### P9 — Media loading placeholder (UI)

While media bytes are in flight, show a spinner over the tile instead of a
broken-image icon. Implemented at the element level (`onLoad`/`onError`), so it
works for both surfaces and survives the native streaming path.

- `webui/src/components/MessageBubble.tsx` (image cell) and
  `AttachmentTile.tsx` keep a muted tile with a spinner until `onLoad`, and fall
  back to the existing placeholder on `onError`.
- No per-client behaviour: the same markup renders on desktop and phone.

## 5b. Out of scope (deferred, not needed for this fix)

- **P1 write-time artifact references** — stops new sessions from accumulating inline blobs on disk. Display works without it; the projection already bounds responses.
- **P6 segment cleanup / streamed uploads** — one-off disk reclaim for historical segments, and streamed attachment uploads for very large files.
- **P7 image derivatives (thumbnails)** — bounded-size renditions for very large images.

Revisit only when a measurement shows the symptom (disk growth or slow opens, large upload failures, slow large-image rendering).

## 6. Contracts

- Media base URL (native → JS): `{ base_url }` from the plugin; path layout
  `/<token>/media/{sig}/{payload}`; token is 32 random bytes, regenerated per
  transport start.
- Proxy allowlist: only `GET`/`HEAD`, only `/media/{sig}/{payload}`, only the
  configured origin. Requests forward `Range`, `If-Range` and `If-None-Match`;
  responses forward `Content-Length`, `Content-Range`, `Accept-Ranges`,
  `Content-Type`, `ETag`, `Cache-Control` and relay `304`.
- Device cache (P8): key = full media path, immutable, LRU with a byte budget,
  purged by `CustodyStore.purgeMedia()`.
- Thread payload: unchanged shape; additions are `truncated` markers and
  artifact references (`{type:"artifact", path, mime, bytes, preview}`).
- Paging: unchanged (`limit`, `direction=latest`, `before`).

## 7. Verification

- Unit: projection budget and artifact normalization (`tests/webui`), proxy
  handler and Range parsing (`mobile/native-network` Go tests).
- Device: open `@kevu` on the A142; check payload size in the gateway log,
  absence of `mobile_media` base64 fetches, image render, video seek, and
  `dumpsys meminfo` during a 50 MiB stream.
- Cross-client: the same session renders identically on desktop and phone;
  paging older turns produces identical message ids.
- Regression: full `tests/webui`, `nanobot/channels/websocket/tests`, WebUI
  `bun run test`, Go `go test ./...` in `mobile/native-network`.

## 8. Rollout

1. P2 (server only, no client change) — budget + never-413, safe, immediate.
2. P3 + P4 (native + WebUI) — one coordinated app build; no back-compat needed
   for the first app version.
3. P5 (ceiling) — after P2/P4 are in the field build.
4. P8 (device cache) — separate change after P3/P4 land; independent rollback.

Rollback: P1/P2 revert cleanly; P3/P4 roll back to the buffered media path by
reverting the plugin method and `mobile-media.ts` together.

## 9. Risks

- The loopback proxy adds a local listener: bind loopback, random token,
  allowlist, and shut it down with the core.
- Redaction/references must be deterministic and identical for every client, and
  must never remove content the UI renders.
- Budget tuning is empirical: verify with the largest real sessions before
  fixing the constant.

## 11. Implementation status

| Item | State |
|---|---|
| P2 projection budget | implemented (`nanobot/webui/ws_http.py`, `THREAD_PROJECTION_BUDGET_BYTES`) |
| P3 media streaming core | implemented (`mobile/native-network/internal/broker/media.go`, C ABI + JNI) |
| P4 same-origin intercept | implemented (`NativeMediaInterceptor.kt`, `NanobotWebViewClient.java`, direct media URLs in the WebUI) |
| P5 cap → safety ceiling | implemented (64 MiB, `nanobot/remote/edge.py`) |
| P9 loading placeholder | implemented |
| P8 device cache | implemented in the core (LRU, 512 MiB, purge on logout/revocation); hit-ratio metrics pending |
| On-device verification | **done for the session/media path** (see below); video seek and the loading placeholder still need a visual check |

Verified on the A142 (release APK `11c86ccf` + tracing build):

- `@kevu` opens (thread payload 238 KiB, no 413) and its images render.
- Media is served by the native interceptor, not the JS bridge:
  `NanobotMedia: open https://localhost/api/media/... status=200 cached=true`.
- The device cache persists across app updates: after reinstalling, every image
  opened from cache (`cached=true`) with no upstream fetch.
- No `mobile_media` request is issued by the client any more.

Still to check by eye: the loading spinner during a cold first fetch, and video
range seeking (`206`) on a session with a video attachment.

## 10. Decisions and open questions

Decided:

- One projection for every client; no per-client shaping.
- Tool result display stays as-is (the UI reads only web-search sources and trace
  text from `result`), so no full-result fetch is planned.
- Optional work (P1/P6/P7) is deferred; the plan is P2 → P3+P4 → P5 → P8.

Open:

1. Budget granularity: 1.5 MiB per thread response, or per turn? Confirm against
   the largest real sessions before fixing the constant.
