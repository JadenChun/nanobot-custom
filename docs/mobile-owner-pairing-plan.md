# Owner-phone pairing — implementation plan

Date: 2026-11-03
Design: `docs/mobile-owner-pairing-design.md`
Scope: owner-only mobile pairing, QR + 8-digit fallback, Tailscale direct only
Status: implemented 2026-11-03

## 1. Deliverable

Replace the current 120-second, comparison-code-first pairing flow with:

```text
Desktop: Pair owner phone → QR + 8-digit fallback (5 min)
Phone:   Scan or enter code → Tailscale ready
Desktop: “Pixel 10 — owner control requested” → Approve
Phone:   Connected until revoked
```

Non-negotiable decisions:

- owner scope only; no chat-only pairing
- QR primary, 8-digit code fallback
- Tailscale direct only in v1
- no biometric gate
- refresh token does not expire after 30 days; only revoke ends the connection
- existing chat-only devices are forced to re-pair

## 2. Work breakdown

### Phase 0 — Contract and UI decisions (COMPLETE)

Decisions:

- Pairing status uses a single `pairing.pending` read with state, device,
  scopes, origin and expiry.
- Manual entry is **full HTTPS origin + 8-digit code**. The desktop displays
  both together; the phone form accepts both.
- Desktop layout is QR and code side-by-side above 480 px, stacked below it.
- Forced re-pair copy: “This desktop now requires owner pairing. Re-pair this
  phone to continue.”
- Legacy chat-only scope constants remain internally for reading old records,
  but new owner pairing rejects chat-only starts.
- Tailscale detection is a read-only server helper and never executes writes.

**Exit:** API/UX decisions frozen; no schema churn after Phase 1.

### Phase 1 — Protocol, pairing state, token lifetime

Files:

- `nanobot/remote/protocol.py`
- `nanobot/remote/pairing.py`
- `nanobot/remote/auth.py`
- `nanobot/remote/store.py`
- `nanobot/remote/owner.py`
- `mobile/contracts/mobile-v1.json`
- `nanobot/remote/contracts/operations-v1.json`
- `tests/remote/test_protocol.py`
- `tests/remote/test_pairing.py`
- `tests/remote/test_auth.py`
- `tests/remote/test_remote_store.py`

Tasks:

1. Raise `CHALLENGE_TTL_SECONDS` from 120 to 300.
2. Make the new pairing API owner-only:
   - require `full_control_consent` and the `owner` scope
   - reject chat-only `pairing.start` requests for the owner path
   - keep legacy scope constants only for reading old records
3. Add an 8-digit fallback code to `PairingOffer`:
   - random 8 digits
   - unique among live offers
   - 5-minute lifetime
   - single use
   - rate-limited lookup
4. Extend pending state:
   - `PairingPending` carries installation ID, label, platform/model,
     requested scopes, origin, transport, created time and state
   - `pairing.status` can return `offer_created`, `pending_approval`,
     `approved`, `denied`, `expired`
5. Split approval semantics:
   - QR path: approve by `pairing_id` only; no comparison code
   - manual path: approve by `pairing_id` + 8-digit code
   - deny/cancel supported
6. Remove the unused random installation UUID from pairing links; the phone
   supplies its own installation ID at collect and the host binds it.
7. Owner refresh lifetime:
   - add device-level `refresh_expires_at = 0` meaning no expiry
   - allow `refresh_expires_in: 0`/nullable in contract
   - make owner-store cleanup skip `refresh_expires_at == 0`
   - keep legacy chat-only expiry behavior for old records
8. Migration:
   - on owner-store load/upgrade, revoke active devices without `owner` scope
   - leave active owner devices untouched
   - reject new chat-only pairing starts with a stable error code
9. Tests for TTL, owner-only rejection, code uniqueness/expiry/rate limit,
   QR approval without code, manual approval, pending status, no-expiry cleanup,
   migration.

**Exit:** backend tests green under the existing Python environment; contract
matches.

### Phase 2 — Owner IPC and settings API

Files:

- `nanobot/remote/management.py`
- `nanobot/remote/client.py`
- `nanobot/webui/settings_remote.py`
- `nanobot/webui/settings_routes.py`
- `nanobot/webui/ws_http.py`
- `tests/remote/test_management.py`
- `tests/remote/test_owner_ipc.py`
- `tests/webui/test_remote_access_settings.py`

Tasks:

1. Add `pairing.status` / pending read to the owner IPC and manager.
2. Add approve/deny/cancel with the QR/manual distinction.
3. Add `settings.remote_access.pairing.pending` read route.
4. Update mutation allowlists and mobile operation registry.
5. Ensure pairing mutations are local/owner-admin only; a remote phone cannot
   approve another phone.
6. Map owner errors to stable API errors:
   - `owner_required`
   - `pairing_expired`
   - `pairing_code_invalid`
   - `pairing_denied`
   - `pairing_rate_limited`
7. Tests for pending status, remote approval denial, expiry, denied request,
   code mismatch, and status after dialog reload.

**Exit:** owner IPC and settings API tests green.

### Phase 3 — Desktop pairing UI

Files:

- `webui/src/components/settings/RemoteAccessSettings.tsx`
- `webui/src/lib/remote-access-api.ts`
- new `webui/src/components/settings/OwnerPairingCard.tsx` (or equivalent)
- `webui/src/tests/remote-access-settings.test.tsx`
- new UI tests

Tasks:

1. Replace the multi-step chat/owner wizard with one owner-only screen:
   - owner scope summary and explicit consent
   - QR code, 8-digit fallback, copy-link
   - 5-minute countdown
2. Add Tailscale readiness panel:
   - status, Serve mapping, detected origin
   - clear local fix commands if not ready
   - do not create an offer while invalid
3. Auto-fill origin from the current HTTPS origin or Tailscale-derived host;
   remove free-form manual typing for the normal path.
4. Add pending request card:
   - device label/platform, requested scopes, origin/transport, time
   - Approve / Deny
   - survives dialog close/reload
5. Add device list improvements: owner badge, last seen, rename, revoke.
6. Remove chat-only consent and scope selection from the owner UI.
7. Rebuild `nanobot/web/dist` from the merged WebUI and verify the installed
   desktop shell serves the new bundle. The desktop setup shell reads
   `nanobot/web/dist`; a stale bundle is why Remote access was missing after the
   merge.
   - Add `remote_access` to `SETTINGS_SECTION_KEYS` in `webui/src/App.tsx` so
     `/settings?section=remote_access` and state restoration work; the sidebar
     item already exists.
8. Tests for countdown, code fallback, pending approval, deny, reload restore,
   invalid Tailscale state and owner-only copy.

**Exit:** WebUI tests and lint green; installed desktop Settings shows Remote
access; manual desktop walkthrough works.

### Phase 4 — Mobile pairing flow

Files:

- `mobile/mobile-bootstrap.js`
- `mobile/android/app/src/main/java/ai/nanobot/mobile/network/NativeNetworkPlugin.kt`
- `mobile/android/app/src/main/AndroidManifest.xml` (if manual-link entry needs it)
- `mobile/android/app/src/androidTest/java/ai/nanobot/mobile/network/NativeNetworkPairingInstrumentedTest.kt`
- `mobile/contracts/mobile-v1.json`
- pairing-related WebUI bootstrap tests

Tasks:

1. Deep link supports QR and manual entry:
   - QR link: origin + challenge
   - manual: origin + 8-digit code
2. State machine:
   - idle → connecting → tailscale_login_required → waiting → connected
   - denied/expired/error states with retry/scan again
3. Owner consent before scanning; copy explains files, commands, settings,
   existing chats.
4. Remove the comparison-code display from the QR path.
5. Keep the desktop approval requirement for both paths.
6. Persist the refresh token with no expiry; reconnect until revoke.
7. Tailscale login is a dedicated step; do not expire the offer while the user
   signs in.
8. Tests for QR, manual code, wrong code, denied, expired, login-required,
   revoke-to-pairing, and restart persistence.

**Exit:** emulator instrumentation green; A142 live pairing green.

### Phase 5 — Tailscale readiness and origin

Files:

- `nanobot/remote/tailscale.py` (new read-only helper) or settings helper
- `nanobot/webui/settings_remote.py`
- `webui/src/components/settings/OwnerPairingCard.tsx`
- tests

Tasks:

1. Read `tailscale status --json` and `tailscale serve status --json`.
2. Return:
   - installed/available
   - connected
   - self DNS name
   - `/mobile/v1` Serve mapping present
   - derived HTTPS origin
3. Show actionable status; never expose a raw shell to the browser.
4. Reject pairing start if Tailscale is unavailable or the origin is not HTTPS.
5. Document the Tailscale ACL recommendation in the UI/help.
6. Unit tests with fixture JSON; no live `tailscale` dependency in unit tests.

**Exit:** desktop can pair without the user typing an origin.

### Phase 6 — Qualification and rollout

1. Python focused/full suites:
   - `tests/remote`, `tests/webui`, pairing/token tests
2. WebUI:
   - `bun run test`, lint, build
3. Android:
   - instrumentation 29 existing + new pairing cases
   - A142 live: QR, 8-digit fallback, approval, revoke, indefinite reconnect
4. Evidence:
   - pairing screenshots, pending request, fallback code, revoke
   - no comparison code in QR path
   - refresh survives restart and a simulated >30-day store cleanup
5. Rollout:
   - migration notice for existing chat-only devices
   - owner devices remain
   - `.nanobot` activation remains deferred until after merge
6. Update `mobile/README.md`, `docs/mobile-owner-pairing-design.md` and
   `mobile/evidence/`.

## 3. Suggested PR split

1. **PR A — protocol/pairing/token lifetime**
   - Phase 1 + tests
2. **PR B — owner IPC/settings API**
   - Phase 2 + tests
3. **PR C — desktop UI**
   - Phase 3 + WebUI tests
4. **PR D — mobile flow**
   - Phase 4 + instrumentation
5. **PR E — Tailscale readiness + qualification**
   - Phase 5/6 + evidence

Each PR should be independently reviewable and keep the old flow behind a
migration boundary until the new flow is green.

## 4. Acceptance criteria

1. Fresh owner phone pairs with one QR scan + one desktop approval in under
   60 seconds on a valid tailnet.
2. Desktop pending request is visible after closing/reopening settings.
3. QR path never asks for a comparison code.
4. 8-digit fallback pairs successfully and still requires desktop approval.
5. Owner phone receives owner scope and can use settings/workspace actions.
6. The connection survives app restart and a store cleanup older than 30 days;
   only revoke ends it.
7. Revoke immediately invalidates access and refresh.
8. Existing owner devices remain; existing chat-only devices are revoked and
   must re-pair as owner devices.
9. No chat-only option exists in the new pairing UI or API.
10. Tailscale status/Serve/origin are detected; no manual origin typing needed.
11. The installed desktop’s WebUI bundle is rebuilt and its Settings sidebar
    shows **Remote access**; `/settings?section=remote_access` resolves to the
    page; no stale `nanobot/web/dist` is served.
12. All Python, WebUI, Android instrumentation and A142 live checks pass.
13. No `.nanobot`/`.nanobot-staging` modification during development.

## 5. Risks

- **Refresh-token lifetime:** indefinite owner sessions increase the impact of a
  stolen phone. Mitigate with rotation, device binding, audit, immediate revoke
  and Tailscale node revocation.
- **8-digit code:** only ~26.6 bits; 5-minute TTL, single use, rate limiting and
  desktop approval are mandatory. Ensure codes are unique among live offers.
- **Tailscale CLI dependency:** read-only helper must degrade clearly if the CLI
  is missing or the daemon is down.
- **Migration:** revoking chat-only devices can surprise users; show a clear
  re-pair notice and keep owner devices untouched.
- **Contract compatibility:** `refresh_expires_in: 0` and owner-only errors need
  updates in mobile/native contracts and tests.
- **Scope:** relay, biometrics and chat-only are explicitly out of v1.

## 6. Implementation status (2026-11-03)

Phases 0–5 are implemented in the working tree: owner-only offer/approval, pending
state, 8-digit fallback, no-expiry owner refresh, forced legacy re-pair, desktop
QR/countdown/pending UI, Tailscale readiness/origin detection, mobile manual-code
entry, and signed release packaging. Phase 6 evidence is recorded in
`mobile/evidence/20261103-m22-release/`; A142 live installation is blocked only by
device availability (ADB and tailnet both report the phone offline).

## 7. First implementation slice

Start with Phase 1 only:

- 300-second TTL
- owner-only pairing request
- 8-digit fallback code
- pending status metadata
- QR approval without comparison code
- no-expiry owner refresh store
- migration/revocation tests

Do not touch the mobile UI until the backend contract and tests are frozen.
