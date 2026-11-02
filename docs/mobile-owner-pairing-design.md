# Owner-phone pairing — solution design draft

Date: 2026-11-03
Status: implemented (2026-11-03) — backend, desktop and mobile owner pairing
Scope: new mobile pairing path; owner-only

## Decisions (2026-11-03)

- New pairings are **owner-only**. There is no chat-only pairing option.
- The QR flow is primary; an **8-digit pairing code** is the fallback.
- **Tailscale direct is the only transport in v1**. Relay is deferred.
- No biometric/app-lock requirement in v1.
- Existing chat-only devices must re-pair as owner devices; they are not
  grandfathered into the new owner flow.
- The owner session has **no 30-day limit**. It remains connected until the
  device is revoked.

## 1. Goal

Make pairing a phone to the desktop feel like one action, while granting **full
owner control** deliberately.

The new flow is owner-only:

- the phone can use the same chats, settings, apps, skills, automations,
  workspace actions and files as the desktop
- there is no chat-only pairing mode in the new UI or API
- existing chat-only devices are forced to re-pair as owner devices

Primary UX target:

```text
Desktop: Pair owner phone → show QR
Phone:   Scan
Desktop: “Pixel 10 — owner control requested” → Approve
Phone:   Connected until revoked
```

## 2. Why change

Current issues (see `docs/mobile-remote-pairing-review.md`):

- the challenge expires after **120 seconds**
- the desktop has no pending-request state and no “phone scanned” signal
- the QR path still requires a second manual comparison-code approval
- the origin is typed manually
- the link carries an unused random installation UUID
- Tailscale login is mixed into pairing errors
- there is no owner-only pairing experience

## 3. Trust model

Three independent layers:

1. **Transport identity**
   - Tailscale direct only in v1: the phone and desktop are tailnet nodes.
   - A relay transport is a future extension, not part of this design.
2. **Pairing authorization**
   - a one-time, high-entropy challenge shown only on the desktop
   - the phone must collect it and the desktop must approve the request
3. **Owner capability**
   - an owner-scoped device token issued only after approval
   - short-lived access token + non-expiring rotating refresh token
   - device-bound, revocable, audited

Tailnet membership is **not** sufficient for owner control. A device on the same
tailnet still needs a valid owner token.

## 4. Pairing state machine

### Desktop

```text
idle
  └─ start ─→ offer_created
                 ├─ phone collected ─→ pending_approval
                 │                       ├─ approve ─→ approved
                 │                       └─ deny ────→ denied
                 ├─ cancel ─────────────→ cancelled
                 └─ timeout ────────────→ expired
```

- The offer lives on the backend, not only in the React dialog.
- Closing and reopening Remote Access settings must still show the pending
  request.
- The desktop can list pending requests and approve/deny any one.

### Phone

```text
idle
  └─ scan/paste ─→ connecting
                    ├─ transport login required ─→ login_required
                    ├─ transport error ──────────→ error (retry)
                    └─ collected ────────────────→ waiting
                                                   ├─ approved ─→ connected
                                                   ├─ denied ───→ rejected
                                                   └─ timeout ──→ expired
```

- The phone never chooses an origin manually for the QR path.
- The phone stores the issued refresh token in Keystore custody. The refresh
  token remains valid until the device is revoked.
- The owner consent is shown once, before scanning; the desktop approval is the
  final authority.

## 5. Desktop UX

### 5.1 Pair owner phone

One screen, not a multi-step wizard:

```text
Pair owner phone
This phone will be able to control this desktop, including files,
commands, settings and existing chats. Only pair a device you own.

[ QR code ]                    Waiting for phone…
                               expires in 4:37

[ Copy link ]  [ Show 8-digit code ]

Tailscale: Connected · Serve mapping valid
Server: https://fedora.taila1e707.ts.net
```

Rules:

- Owner mode is the only mode. The consent text is explicit and cannot be
  dismissed without checking “I understand”.
- The QR/link is valid for **5 minutes**, single use.
- A countdown is always visible.
- “Refresh link” rotates only the challenge, not the device identity.
- No comparison code is shown in the QR path.
- An **8-digit pairing code** is shown as the fallback. The phone enters that
  code manually; the desktop still approves the request.

### 5.2 Pending request

When the phone collects:

```text
Pixel 10 · Android 16
Owner control requested
Files, commands, settings, existing chats
192.0.2.44 · just now

[ Deny ]  [ Approve owner access ]
```

- Device name, platform, requested scopes, origin/transport and time are shown.
- Approval is one click; no code typing.
- The request is visible even if the pairing dialog was closed and reopened.
- Deny/cancel writes an audit event.

### 5.3 Active device

Device list shows:

- label, platform, last seen
- owner scope badge
- rename
- revoke with confirmation
- “Re-authenticate” if the session is stale

## 6. Phone UX

- Scan QR → show “Connecting to <host>”.
- If Tailscale needs login, show a dedicated **Connect Tailscale** step. The
  pairing timer is paused or the offer is refreshed after login; the user does
  not start over.
- Show “Waiting for approval on the desktop” with a countdown.
- On approval, go straight to the existing owner UI.
- On denial/expiry, show **Scan again** without losing the app state.
- Manual code path: enter the 8-digit pairing code shown on the desktop. The
  desktop still approves the request; no separate comparison code is needed.

## 6.1 Desktop delivery

The installed Tauri shell is a setup/status shell. The Remote access screen is
part of the shared WebUI served from `nanobot/web/dist`, not the Tauri setup UI.
Every desktop build must rebuild that bundle and verify the desktop shell serves
the new assets.

Required check:

```text
bun run build (webui)
→ nanobot/web/dist includes Remote access
→ restart the installed desktop/gateway
→ Settings sidebar shows Remote access
```

A stale `nanobot/web/dist` was the reason Remote access was missing from the
installed desktop after the merge. This is now an explicit acceptance item.

## 7. Protocol/API changes

### 7.1 Pairing start

`settings.remote_access.pairing.start` returns:

```json
{
  "pairing_id": "…",
  "challenge": "…",
  "transport": "tailscale",
  "origin": "https://fedora.taila1e707.ts.net",
  "scopes": ["owner", "chat.read", "chat.write", "media.read", "attachment.write"],
  "expires_at": 1760000000,
  "expires_in": 300,
  "qr_link": "nanobot://pair?origin=…&challenge=…"
}
```

- `scopes` is always owner-complete for the new flow.
- `transport` is always `tailscale` in v1.
- No `installation` parameter is placed in the link. The phone supplies its own
  installation ID during collection; the host binds the challenge to it.
- The same offer carries an 8-digit fallback code for manual entry.

### 7.2 Manual code fallback

The offer always carries an 8-digit code alongside the QR/link.

```text
Phone manual entry:
  Server: https://fedora.taila1e707.ts.net   (shown by desktop)
  Pairing code: 4827 1936
```

- Code: 8 digits, 5-minute TTL, single use, rate-limited.
- The code is a lookup/collect secret, not an owner grant.
- The desktop must still approve the pending device.
- QR and code resolve to the same pairing offer; only one can be used.
- The desktop displays the host and code together so the phone has both pieces.

### 7.3 Pending status

New read operation:

```text
settings.remote_access.pairing.pending
→ {
    pairing_id,
    state: "offer_created" | "pending_approval" | "approved" | "denied" | "expired",
    expires_at,
    device: { installation_id, label, platform, model },
    scopes,
    origin,
    transport,
    requested_at
  }
```

The desktop uses this to render the pending card and restore state after a
reload. It can poll every 1–2 seconds or subscribe through the existing settings
WebSocket events.

### 7.4 Collect

Existing `mobile_pair_collect` remains, but:

- returns pending state and device metadata
- does not expose a separate comparison code to the phone
- validates the requested owner scopes against the offer
- the manual path submits the 8-digit `pairingCode` to bind the collect request

### 7.5 Approve / deny

```text
settings.remote_access.pairing.approve
{ pairingId, decision: "approve", scopes: [...] }
```

- QR path: no code required.
- Manual-code path: the 8-digit `pairingCode` is required.
- Approval is only possible from `pending_approval`.
- Owner scope cannot be added at approval; it is part of the offer and consent.

```text
settings.remote_access.pairing.deny
{ pairingId }
```

### 7.6 Token issuance

Existing token service remains:

- access TTL: 600 s
- refresh: no idle expiry; valid until device revoke
- owner scope bound to installation ID
- refresh rotation and revocation unchanged
- audit: pairing start, collect, approve, deny, revoke

Protocol/contract changes required for this decision:

- replace the fixed `REFRESH_IDLE_SECONDS` owner-device expiry with a
  device-level `refresh_expires_at = 0` meaning “no expiry”
- allow `refresh_expires_in: 0` in the pairing/token payload, or make it
  nullable; document the meaning
- remove idle-age revocation from the owner device store; only explicit revoke,
  scope reduction or token invalidation removes an owner session
- existing chat-only device expiry behavior can remain for legacy records until
  they are revoked/re-paired

## 8. Transport design

### 8.1 Tailscale direct (first implementation)

```text
Desktop owner process
  └─ tailscale serve /mobile/v1 → gateway WS/HTTP
Phone
  └─ embedded tsnet node
       └─ https://<host>.<tailnet>.ts.net/mobile/v1
```

Desktop preparation:

1. Detect `tailscale status`.
2. Verify Serve maps `/mobile/v1` to the gateway.
3. Derive the public origin automatically.
4. Show status before creating the offer; do not create a 5-minute offer while
   Tailscale is signed out.

Phone preparation:

1. Start tsnet.
2. If login is required, show **Connect Tailscale** and wait.
3. Only then collect the challenge.
4. Keep the offer alive while login is in progress (extend or refresh).

### 8.2 Transport scope

Tailscale direct is the only transport in v1. Do not implement or expose a relay
transport in this tranche. The pairing payload may carry `transport: "tailscale"`
so a later relay can be added without changing the phone/desktop states.

## 9. Security requirements

- Challenge: 256-bit random, single use, 5-minute TTL, rate-limited.
- Challenge is bound to host identity, transport and requested scopes.
- Device binding: phone installation ID is recorded at collect; approval cannot
  silently switch devices.
- Owner approval must be a local desktop action; a remote phone cannot approve
  another phone.
- Owner-mode consent explicitly lists files, commands, settings and existing
  chats.
- No owner token is issued until the desktop approves.
- Access tokens are short-lived; refresh tokens are hashed at rest, rotated on
  use, and remain valid until revoked. There is no 30-day idle cutoff.
- Revoke is immediate; all sessions and refresh tokens for that device die.
- Audit records: start, collect, approve, deny, revoke, acting device.
- Tailnet ACLs are recommended defense in depth, not the only authorization.
- No biometric/app-lock requirement in v1.

## 10. Compatibility and migration

- Existing owner devices remain valid.
- Existing chat-only devices are revoked on upgrade and must re-pair as owner
  devices.
- The new UI and API accept only owner-scoped pairing.
- Remove the unused `installation` parameter from new links.
- The 8-digit code is the manual fallback; no separate comparison code is kept.

## 11. Acceptance criteria

1. A fresh phone can pair in one scan plus one desktop approval within 60
   seconds on a working tailnet.
2. The desktop shows the phone-scanned request after closing/reopening settings.
3. The QR path requires no code typing.
4. The 8-digit fallback completes without a QR scan and still requires desktop
   approval.
5. The owner phone receives owner scope and can use settings/workspace actions.
6. The phone survives a 5-minute window even if Tailscale login takes time.
7. Expired/denied requests can be retried without restarting the app.
8. The owner session persists beyond 30 days without re-pairing.
9. A token-store cleanup pass does not age out an owner session; only revoke does.
10. Revoke immediately invalidates access and the refresh token.
11. Existing owner devices continue to work; existing chat-only devices are
    rejected and must re-pair as owner devices.
12. No chat-only mode is offered in the new pairing UI or API.

## 12. Resolved implementation details

- Desktop shows QR and the 8-digit code side-by-side above 480 px, stacked below
  it; both are always visible while the offer is live.
- Manual entry takes the full HTTPS origin plus the 8-digit code. The desktop
  shows both; the phone form accepts both.
- Forced re-pair copy: “This desktop now requires owner pairing. Re-pair this
  phone to continue.”
- Legacy chat-only scope constants remain internally for reading old records;
  the new owner pairing API rejects new chat-only starts.
- Tailscale status detection is read-only and never writes Serve or auth state.
- `remote_access` is included in the settings section keys so the desktop route
  and state restoration work.

All product decisions are fixed and implemented. Verification for this revision:

- `tests/remote` + `tests/webui/test_remote_access_settings.py`: green (111 cases,
  owner pairing, 300-second TTL, 8-digit code, no-expiry refresh, legacy revocation).
- WebUI: `bun run test`, lint and `tsc` green; installed desktop bundle rebuilt from
  `nanobot/web/dist` and `Settings → Remote access` verified against the running gateway.
- Manual fallback end-to-end: desktop produced an 8-digit code, phone edge collected by
  origin+code, desktop approved, poll returned `refresh_expires_in: 0` and owner scopes,
  revoke returned HTTP 401 on the next access attempt.
- Signed Android release: `mobile/evidence/20261103-m22-release/nanobot-mobile-release.apk`
  (v2 signature verified, SHA-256 `4c9e7ef5…140780`). A142 release installation succeeded
  in place; the pulled APK hash matches the signed artifact.
