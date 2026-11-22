# Mobile Background Notification Architecture Plan

**Date:** 2026-11-22\
**Status:** Decided (Deferred Implementation — Future Session)\
**Related:** [Mobile Connectivity Reliability & UX Plan](./mobile-connectivity-reliability-ux-plan.md)

---

## 1. Product & Hosting Constraints

All mobile notification decisions must respect these core constraints:

1. **No public internet exposure for the desktop host:** The gateway stays private-only on the user's Tailscale tailnet (`tailscale serve`, no Funnel / public tunnel).
2. **Fully self-contained Kohaku app:** No separate VPN app required on the phone.
3. **No SaaS platform console or hosted push relay to manage:** This is a personal/self-signed build; the desktop talks directly to OS push endpoints (outbound only) or relies on local foreground keep-alive.

---

## 2. Platform Decisions

### Android — Stay on `RunKeepAliveService` Only (No FCM)
- **Decision:** Android will **not** add Firebase Cloud Messaging (FCM) or any external push service. It will continue using the local Android foreground service (`RunKeepAliveService`) for active turns started from the phone.
- **Why:** Keeps Android 100% self-contained with zero external accounts/consoles.
- **Required fixes (tracked in Phase 1 of the Connectivity Plan):**
  - Decouple `startMobileRunKeepAlive()` (`webui/src/lib/mobile-run-keep-alive.ts`) from `browserNotifications: false` so active phone-started turns keep alive by default.
  - Request `POST_NOTIFICATIONS` runtime permission on Android 13+ in `MainActivity.java`.
  - Fix the `RunKeepAliveService.onDestroy()` race unconditionally clearing `RunKeepAliveRegistry`, and handle Android 15 `dataSync` `onTimeout()` cleanly.
- **Idle / scheduled tasks on Android:** Users who want notifications while the app is closed can use nanobot's existing outbound chat channels (Telegram, Slack, Discord, or self-hosted `ntfy`).

### iOS (Future Personal Self-Signed Build) — Direct Desktop-to-APNs Push
- **Why iOS differs:** iOS suspends normal apps ~30 seconds after backgrounding and freezes `tsnet`/WebSocket connections; there is no equivalent of Android's `RunKeepAliveService`.
- **Decision:** Because Kohaku iOS will be a **personal self-signed build**, the desktop nanobot gateway will send **outbound HTTP/2 push notifications directly to Apple APNs**—no intermediate push relay and no inbound public port on the desktop.

```text
Desktop nanobot ──(outbound HTTP/2 + local .p8 JWT)──▶ api.push.apple.com ──▶ iPhone wakes & shows alert
                                                                                  │
                                                         User taps ──▶ App connects over private Tailscale
```

- **Signing requirement caveat:**
  - **Paid Apple Developer account (\$99/yr):** Grants the `Push Notifications` entitlement and `.p8` APNs Auth Key required for direct desktop → APNs pushes.
  - **Free Apple ID (7-day Xcode signing):** Does not allow the Push Notifications entitlement. Fallback is an outbound webhook/channel from desktop to **Bark**, **ntfy**, or **Telegram**.

---

## 3. Planned Direct-APNs Design (For Future iOS Session)

When implementing iOS background notifications later:

1. **Desktop Config (`nanobot/config/schema.py`):**
   - Add an optional `remote.apns` config section (all local on desktop, never transmitted):
     - `enabled: bool`
     - `teamId: str`
     - `keyId: str`
     - `bundleId: str`
     - `keyPath: str` (path to local `.p8` file)
     - `useSandbox: bool` (switches between `api.sandbox.push.apple.com` and `api.push.apple.com`)
2. **Device Token Registration (`nanobot/remote/`):**
   - Extend paired device metadata in `nanobot/remote/store.py` to store an optional per-device `push_token` and `push_platform` (`"apns"`).
   - Add an authenticated endpoint `POST /mobile/v1/push-token` (over private Tailscale) called by the iOS app after pairing and whenever `didRegisterForRemoteNotificationsWithDeviceToken` fires.
   - Clear the stored token on device revocation or APNs `410 Unregistered`.
3. **Trigger Conditions (Desktop Gateway):**
   - Only fire an APNs push when a paired iOS device has a registered token **and** has no active foreground WebSocket attached (avoid duplicate alerts while the user is actively viewing the app).
   - Trigger events:
     - Assistant turn completed (`turn_end` / final message)
     - Tool/action approval required (`approval_request`)
     - Scheduled automation / cron job output
4. **Privacy-Preserving Payload:**
   - Default to an **opaque wake-up notification** (e.g., `"Kohaku finished a task"` / `"Approval needed"` + `chat_id` identifier) so private conversation content never traverses Apple's push servers.
   - Deep-link tapping the notification directly to the relevant `chat_id`, where the app reconnects over private Tailscale and loads the latest transcript (relying on the Connectivity Plan's fast resume + session/thread resync).

---

## 4. Dependencies on the Connectivity Plan

The following items from [mobile-connectivity-reliability-ux-plan.md](./mobile-connectivity-reliability-ux-plan.md) directly lay the groundwork for iOS push later:
- **Fast wake & reconnect** (Phases 1–2) so tapping a push notification opens directly into a live session without a 90s splash or dead-end error.
- **In-flight stream replay & foreground session/thread resync** (Phase 4) so waking from an opaque push immediately shows the completed response or pending approval card.
- **C ABI `network_changed` / `app_resume` hook** (Phase 2) so iOS `NWPathMonitor` and `sceneDidBecomeActive` plug directly into the same Go broker interface as Android.
