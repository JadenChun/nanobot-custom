# Mobile Connectivity Reliability & UX Improvement Plan

**Date:** 2026-11-22\
**Status:** Finalized Plan\
**Related:** [Mobile Notification Architecture Plan](./mobile-notification-architecture-plan.md), [Mobile Client Plan](./mobile-client-plan.md)

---

## 1. Architectural Direction (Confirmed)

We are **keeping the self-contained embedded `tsnet` architecture** (`mobile/native-network` Go broker + C ABI + Android Capacitor shell + shared `webui` SPA + desktop `tailscale serve` private tailnet).

### Constraints & Rationale
- **Zero public internet exposure:** The desktop host must remain private-only on the user's tailnet (no public Funnel/Cloudflare Tunnel hostname, no open router ports, no third-party TLS termination).
- **Fully self-contained Kohaku app:** No separate Tailscale app required on the phone.
- **No SaaS console or hosted relay to operate:** All connectivity and auth remain directly between the paired phone and desktop host.
- **Future iOS compatibility:** The C ABI boundary (`mobile/native-network/include/nanobot_network.h`) allows compiling the same Go `internal/broker` into an iOS `.xcframework` later (paired with Swift `NWPathMonitor` + Keychain/Secure Enclave).
- **Notifications scope:** Android stays on `RunKeepAliveService` only; iOS direct-APNs push is deferred and documented in [mobile-notification-architecture-plan.md](./mobile-notification-architecture-plan.md).

---

## 2. Root-Cause Summary (From Full-Stack Audit)

Five cross-layer failure chains account for the poor mobile experience today:

1. **Silent, permanent disconnects on network handoffs or brief backgrounding**
   - Android lacks `ACCESS_NETWORK_STATE` and `ConnectivityManager.NetworkCallback`; Go `interface_android.go` masks link changes with a hardcoded `"android"` interface.
   - Go WS has no automatic keepalive or read/write deadlines, and `WSPing` timeout leaves `c.ws` occupied.
   - WebUI `MobileWebSocket.recover()` leaves dead sockets in `OPEN` state if `refresh()` fails; `NanobotClient.scheduleReconnect()` permanently aborts auto-reconnect after 3.5s (`authRetryFailures >= 3`) on any transient error; `recoverMobileRuntime()` is a no-op once the client enters `"error"`.
   - `MobileConnectionNotice` never subscribes to `client.onStatus()`, so the user sees **no disconnect banner and no Retry button** while outbound messages silently queue.

2. **Mid-session drops every 7–10 minutes & during LLM streaming**
   - WebUI proactively closes live WebSockets every ~7 min (`0.7 * access_ttl_seconds`) to rotate tokens, wiping temporary chats, aborting uploads/transcriptions, and double-rotating tokens on reconnect.
   - Server pins the 10-minute access token at WS upgrade with no in-band refresh and closes with overloaded `1008`.
   - Every outbound WS frame runs uncached Unix-socket IPC + two JSON disk reads on the server; combined with a 256-frame outbound queue (no delta coalescing), a 1-event/20ms Android poller (50 events/sec cap, no `try/catch`), and a 128-event Go queue that kills WS on overflow, streaming bursts or brief stalls disconnect active turns.

3. **Sluggish cold starts & dead-end lockout screens**
   - Cold start blocks up to 90s on a static `.boot-splash`, then executes redundant sequential handshakes and double `transport.refresh()`.
   - React boot errors render a dead-end desktop hint page with no Retry button.
   - `presentReauth` has no Unpair / Switch Server button (and Settings hides Logout on mobile); Keystore/custody corruption traps the user in an unrecoverable error loop.

4. **Missed messages & stale state after resuming from background**
   - No WS event sequence numbers, replay buffer, or in-flight stream resume on `attach`/`hydrate`.
   - `useSessions` never resyncs on reconnect/resume; `ThreadShell` skips idle-chat refresh on return.
   - `RunKeepAliveService` is disabled by default (`browserNotifications: false`), lacks `POST_NOTIFICATIONS` runtime permission request, and has an `onDestroy()` registry wipe race.

5. **Fragile pairing & remote host setup**
   - Server's 120s device-proof TTL starts at `collect` (before desktop approval) and is never extended on `approve()` or `renew()`, causing `410 pairing_expired`.
   - Android `jsonRequest()` strips `pairing_rejected` and `upgrade_required`; desktop pairing `<Dialog>` hides errors behind the modal backdrop.
   - `tailscale_status()` ignores the configured gateway port and blocks the asyncio event loop up to 6s.

---

## 3. Phased Implementation Plan

Each phase is scoped to be independently testable and reviewable.

### Phase 1 — Stop Silent Disconnects, Escape Dead-Ends & Fix Android Keep-Alive *(Done)*
**Goal:** Eliminate silent offline hangs, lockout screens, and broken mobile status feedback (no Go/JNI ABI changes required).

- **WebUI live status & reconnect reliability**
  - [x] Subscribe `Shell` / `MobileConnectionNotice` to `client.onStatus()` (`webui/src/App.tsx`, `webui/src/components/MobileConnectionNotice.tsx`) so mobile always reflects `connecting` / `reconnecting` / `error` with a working Retry action.
  - [x] Separate transient network errors from genuine auth revocation in `App.tsx` `onReauth` and `NanobotClient.scheduleReconnect()` (`webui/src/lib/nanobot-client.ts`) so a >3.5s drop uses full backoff instead of permanently aborting.
  - [x] Fix `MobileWebSocket.recover()` (`webui/src/lib/mobile-runtime.ts`) to always close dead sockets when `wsPing()` fails even if `refresh()` fails.
  - [x] Fix `NanobotClient.retry()` to force-close unhealthy/half-open sockets, and wire `visibilitychange` / `online` / `nanobot-native-resume` (`webui/src/main.tsx`, `webui/src/App.tsx`) to revive clients in `"error"` state.
- **Escape dead-end lockout & boot screens**
  - [x] Add Retry + Unpair/Re-pair actions to the React boot error screen (`webui/src/App.tsx`) and handle `handshake_status_401` cleanly.
  - [x] Add **"Unpair / Switch server"** to `presentReauth` (`mobile/mobile-bootstrap.js`) and expose a mobile Unpair/Disconnect action in Settings (`webui/src/components/settings/SettingsSidebar.tsx`, `webui/src/App.tsx`) calling native `plugin.logout()`.
  - [x] Add a clean custody/Keystore reset escape hatch in `mobile/mobile-bootstrap.js` when `CustodyStore` throws invalid/corrupt errors, and fix `presentReauth` (disable Retry while in-flight, remove `.boot-splash`, auto-resume polling after Tailscale login).
- **Android `RunKeepAliveService` reliability**
  - [x] Decouple `startMobileRunKeepAlive()` (`webui/src/lib/mobile-run-keep-alive.ts`) from `browserNotifications: false` so phone-started runs keep alive by default on Android.
  - [x] Request `POST_NOTIFICATIONS` runtime permission on Android 13+ in `MainActivity.java`.
  - [x] Fix `RunKeepAliveService.onDestroy()` unconditionally clearing `RunKeepAliveRegistry` (`RunKeepAliveService.kt`) and guard start/stop ordering.
- **Verification:** `cd webui && bun run test` (new/updated tests in `mobile-socket-recovery.test.ts`, `mobile-runtime.test.ts`, `mobile-shell.test.tsx`) + Android unit tests.

---

### Phase 2 — Harden Android Bridge & Go Native Network Broker *(Core Complete)*
**Goal:** Make Wi-Fi ↔ cellular handoffs, Doze wakeups, streaming throughput, and media loading rock-solid.

- **Android poller, correlator & network callbacks**
  - [x] Wrap `NativeNetworkPlugin.startPoller()` (`NativeNetworkPlugin.kt`) in `try/catch` and drain up to 64 queued events per tick (instead of 1 event / 20ms).
  - [x] Add `ACCESS_NETWORK_STATE` (`AndroidManifest.xml`), register `ConnectivityManager.NetworkCallback`, and forward network changes + app foreground/resume to the Go broker via `nanobot_network_network_changed` JNI/C ABI hook.
  - [x] Fix `NativeRequestCorrelator` leak on `cancel()` (`NativeNetworkPlugin.kt`).
  - [x] Preserve `pairing_rejected` and `upgrade_required` in `jsonRequest()`.
  - [x] Fix `NativeMediaInterceptor.kt` case-insensitive `Content-Type` lookup and stop mapping transient/transport errors to `404`.
  - [x] Cache decrypted custody in memory (keyed by file mtime+length, invalidated on `save()`/`clear()` in `CustodyStore.kt`) and serialize chunked `wsSend` frames per `wsId` (`mobile/mobile-bootstrap.js`).
- **Go broker hardening (`mobile/native-network/`)**
  - [x] Expose C ABI `nanobot_network_network_changed` hook (`nanobot_network.h`, `cmd/lib/main.go`); report empty interface list when offline in `interface_android.go`; flush `httpClient.CloseIdleConnections()` and invalidate peer DNS cache.
  - [x] Add automatic WS ping/pong keepalive (25s/10s) + 15s write deadline (`ws.go`), close `wsSession` on ping failure, and clear `c.ws` *before* emitting `ws_closed`.
  - [x] Add 15s dial timeout to `tailnetDial` (`core.go`), retry idempotent `GET` on transient `EOF`/`ECONNRESET` (`http.go`), cancel request timers in `removeRequest`, and fix `waitForPeers` / 30s negative peer DNS cache (`core.go`).
  - [x] Move media cache filesystem walks off global `c.mu` (`mediaCacheMu` in `media.go`), enforce `"ready"` check and pre-request `mediaStreamLimit` in `OpenMediaStream`, close media streams in `Core.Stop()`, skip `.part-*` in `trimMediaCache()`, and cancel `c.ctx` before `node.Close()`.
  - [ ] Add post-startup `tsnet` `BackendState` / peer-online monitoring and richer failure classification in `Status()` / `classifyHTTPError()`.
- **Verification:** `go test ./...` in `mobile/native-network`, `mobile/scripts/test-cabi.sh`, Android unit/instrumented tests, rebuild `libnanobot_network.so`.

---

### Phase 3 — Eliminate 7–10 Minute Drops, Streaming Lag & Auth Cascades (Server + WebUI) *(In Progress)*
**Goal:** Keep long mobile sessions alive smoothly across token boundaries and heavy streaming without stalling the gateway.

- **In-band WS token renewal & auth error codes**
  - [x] Return distinct pairing error codes (`upgrade_required` → 426, `serve_unverified` → 503) in `nanobot/remote/edge.py`.
  - [x] Never convert transient `OwnerUnavailableError` into WS `1008` / scope wipe in `nanobot/webui/gateway_endpoint.py`.
  - [x] Add an in-band WS `auth_refresh` frame (`nanobot/webui/gateway_endpoint.py`, `nanobot/channels/websocket/runtime.py`, `nanobot/webui/inbound_commands.py`, `NativeNetworkPlugin.kt`, `mobile-bootstrap.js`, `webui/src/lib/mobile-runtime.ts`) so healthy mobile WebSockets rotate access tokens without closing the socket, exposing tokens to JS, wiping temporary chats, or aborting uploads/transcriptions.
  - [x] Remove redundant double `transport.refresh()` during reconnect (`mobile-runtime.ts`) and deduplicate concurrent `/mobile/v1/pair/recover` calls per installation (`nanobot/remote/pairing.py`).
- **Server streaming & event-loop performance**
  - [x] Increase mobile outbound queue headroom (`_OUTBOUND_QUEUE_MAX_FRAMES`: 256 → 1024) and move `_mobile_delivery_allowed` out of `_OUTBOUND_SEND_TIMEOUT_S` in `nanobot/channels/websocket/runtime.py`.
  - [x] Pass `expected_port` into `tailscale_status()` in `nanobot/webui/settings_remote.py`.
- **Verification:** `pytest` on `tests/` for `remote`, `webui`, `websocket`, `ruff check nanobot/`, `uv run --no-sync basedpyright`, `cd webui && bun run test`.

---

### Phase 4 — Seamless Resume, Faster Cold Start & Pairing/Shell Polish *(In Progress)*
**Goal:** Make app resume, cold boot, pairing, and daily mobile navigation feel fast and native.

- **Resume & state sync**
  - [x] Resync `useSessions()` on WS `open` after disconnect and foreground return (`webui/src/hooks/useSessions.ts`).
  - [x] Refresh idle threads on foreground return on mobile (`webui/src/components/thread/ThreadShell.tsx`).
  - [x] Eliminate duplicate `MobileRuntime` instantiation (`webui/src/main.tsx`).
  - [x] Replay in-flight stream buffers (`_stream_text_buffers` / `_reasoning_text_buffers`) on `attach` / `hydrate` scoped strictly to the attaching connection (`nanobot/channels/websocket/runtime.py`, `nanobot/webui/inbound_commands.py`).
  - [x] Show live boot progress on `.boot-splash` and shorten silent cold-start wait from 90s to 20s before showing actionable `presentReauth` (`mobile/mobile-bootstrap.js`).
- **Pairing & mobile shell polish**
  - [x] Extend `proof_expires_at` on server `approve()` and `renew()` (`nanobot/remote/pairing.py`).
  - [x] Render pairing errors inside the desktop `<Dialog>` and change `"Connected until revoked"` to `"Authorized until revoked"` (`RemoteAccessSettings.tsx`).
  - [x] Fix `presentPairing()` overwriting `ui.node` after `"ready"` and surface `pendingPairingLink()` parse errors in `mobile/mobile-bootstrap.js`.
  - [x] Stop rendering desktop `<HostChrome>` / macOS traffic-light padding on mobile (`webui/src/App.tsx`), fix Android hardware Back hash sync & open dialog handling (`App.tsx`), and skip `/api/pairing/requests` polling for non-owner mobile clients (`App.tsx`).
  - [ ] Show the real gateway port in `RemoteAccessSettings.tsx` and add reconnect-triggered retry for failed media thumbnails (`mobile-media.ts`).
- **Verification:** Full `pytest`, `bun run test`, `bun run build`, and emulator/device qualification (`mobile/scripts/qualify-emulator.sh`).
