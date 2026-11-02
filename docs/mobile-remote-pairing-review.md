# Mobile remote pairing review

Date: 2026-11-02
Status: historical review — superseded by owner-only design (2026-11-03)

## Current flow

1. Desktop opens **Settings → Remote access → Set up remote access**.
2. The user types an HTTPS origin (normally the Tailscale Serve hostname), selects
   full-owner control and media consent, then selects **Get started**.
3. Desktop shows a QR code, a six-digit comparison code, and a `nanobot://pair` link.
4. Phone scans the QR or pastes the link. The embedded Tailscale node starts,
   collects the challenge, and displays the six-digit code.
5. The user must return to the desktop and press **Approve phone** while the
   dialog is still open.
6. Phone polls every two seconds until approval, then stores the refresh token.

Relevant code:

- `webui/src/components/settings/RemoteAccessSettings.tsx`
- `webui/src/lib/remote-access-api.ts`
- `mobile/mobile-bootstrap.js`
- `nanobot/remote/pairing.py`
- `nanobot/remote/protocol.py`

## Findings

### P0 — 120-second challenge lifetime

`CHALLENGE_TTL_SECONDS = 120` (`nanobot/remote/protocol.py:24`). A first-time
pairing includes installing/opening the app, granting camera permission, starting
the embedded Tailscale node, and signing in. The link can expire before the phone
ever reaches the server. The UI has no countdown and no way to extend the current
offer; **Refresh code** creates a new offer and invalidates the old QR/link.

OpenCode uses a **5-minute, single-use** link. Codex keeps the host setup alive
while the phone completes account/MFA setup.

### P0 — Desktop is blind to the phone

The phone collects the challenge, but the desktop only knows because its dialog
is still open. There is no pending-request list, no “phone scanned” state, and no
notification. Closing the dialog loses the approval context. The user has to
notice the phone, then manually press **Approve phone** using the desktop’s copy
of the code.

Codex shows the host/device request in the desktop app and lets the user approve
it there. OpenCode requires no second desktop approval because the one-time QR
link itself is the proof.

### P1 — Manual origin entry

The origin field starts empty unless the WebUI is already on HTTPS. The user must
know and type the Tailscale Serve hostname. There is no validation that the
origin is reachable, that Serve maps `/mobile/v1`, or that the origin matches the
server. A typo produces an unhelpful failure on the phone.

### P1 — Misleading installation ID in the link

`buildMobilePairingLink()` inserts `crypto.randomUUID()` as the link’s
`installation` parameter (`webui/src/lib/remote-access-api.ts`). The phone ignores
that value and uses its own local installation ID in `pairStart`
(`mobile/mobile-bootstrap.js`). This caused confusion during physical testing and
can surface as `custody is bound to a different installation` when the app’s
custody state and link disagree. Either remove the parameter or make the phone
use it consistently, but do not include an unused identity.

### P1 — Full owner control needs a clear owner-vs-guest choice

The desktop checkbox defaults to true and the primary CTA enables it. That is
reasonable for a personal owner phone, but only if the UI states plainly that the
phone can control the desktop. For guests, chat-only must be the default. Treat
“Owner phone” and “Chat-only phone” as two explicit pairing modes rather than one
ambiguous checkbox, and require a second confirmation for owner scope.

### P1 — Tailscale login is mixed into pairing errors

The phone can fail with `tailnet login required` after the pairing link was
created. The user then has to open a separate login, while the 120-second offer
expires. Tailscale readiness should be a prerequisite step with its own status,
not an error inside the pairing attempt.

### P2 — No code fallback / weak recovery

The phone accepts a full `nanobot://pair` link or a bare challenge when the origin
is already known, but the desktop does not offer a short human code. If QR
scanning and link sharing both fail, there is no simple fallback. A short code
plus origin is also easier for desktop-to-phone handoff.

### P2 — Device presentation is thin

The device list shows label/status/scopes but not last seen, platform, or a
rename control. Pairing labels are generic (`phone`). Codex and OpenCode expose
device/session management as part of the remote feature.

### P2 — No relay/remote option

Nanobot currently depends on the user’s own Tailscale Serve setup. OpenCode has
an explicit opt-in `pair --remote` relay; Codex uses a relay layer so devices do
not need a manually exposed server. This is a larger feature, but the pairing
design should leave room for it.

## Reference flows

### OpenCode

```text
opencode pair
  → one-time link + QR, valid 5 minutes
  → scan/paste into the app
  → session lasts 30 days
  → localhost by default; `pair --remote` opts into an OpenTunnel relay
```

Key properties: one action on the host, one scan on the phone, no second
approval, short-lived single-use secret, long-lived session after pairing.

### Codex

```text
Desktop Settings → Connections → Control this Mac/PC → Set up
  → QR code
Phone scans → signs into the same ChatGPT account/workspace → MFA
  → host appears in Codex/Remote
```

Key properties: QR binds the phone to the host; account/workspace identity and
MFA establish trust; the desktop manages connected devices; a relay keeps hosts
reachable without exposing the server.

## Recommended target flow

### Fast path

1. **Desktop: Pair a phone**
   - Choose **Chat only** (default), **Media**, or **Full owner control**
     (explicit, second confirmation).
   - Origin is derived from the current HTTPS WebUI origin or a configured
     public URL; the app validates reachability and Serve mapping.
   - Generate a single-use, 5-minute pairing token and show QR/link with a
     countdown.
2. **Phone: Scan**
   - App opens the link, starts Tailscale, and presents the device identity.
   - Desktop switches to **Phone connected — approve?** with device name,
     platform, requested scopes, and IP/last-seen.
3. **Desktop: Approve once**
   - One **Approve** action grants the selected scopes. No comparison code in
     the QR path: the QR token is already an out-of-band secret.
   - Optional **Deny** and audit entry.
4. **Phone: Connected**
   - Stores a 30-day refresh token; reconnects without re-pairing.
   - Desktop device list supports rename, last-seen, revoke.

### Fallback path

- Desktop can show an 8-digit code and origin.
- Phone enters the code manually; desktop shows a pending request and approves.
- If the code path is used without a camera, keep the six-digit comparison code
  as an optional confirmation.

### Onboarding

- Before generating a link, show:
  - Tailscale: connected / sign in required / error.
  - Serve mapping: valid / missing.
  - Origin: detected / invalid.
- Do not start the 5-minute timer until the phone is actually able to reach the
  host, or pause/refresh the timer while Tailscale login is in progress.

### Security

- High-entropy single-use token; bind to phone installation ID and scopes.
- Owner scope explicit and confirmed; chat-only remains the guest mode. A personal
  owner phone may default to owner mode, but the consent screen must say it can
  control the desktop.
- Rate-limit pending requests; show requesting device; audit approve/deny/revoke.
- Keep refresh tokens hashed server-side; short access TTL; immediate revoke.
- Optional relay must be E2E encrypted and opt-in, like OpenCode `--remote`.

## Transport comparison: Tailscale vs relay

Tailscale is a *transport*, not a pairing design. It gives strong properties:

- no public inbound port; WireGuard encryption end to end
- device identity, tailnet ACLs, MagicDNS, NAT traversal and DERP fallback
- direct peer-to-peer path with low latency
- self-hostable control plane (Headscale or Tailscale)

But the current cost is high: the phone embeds its own `tsnet` node, the host must
run `tailscale serve`, both devices must be logged into the same tailnet, and the
user must understand the origin/mapping. The embedded node also introduced the
`needs_login`/DNS failures seen during physical testing.

OpenCode and Codex avoid that onboarding cost by using an outbound relay or
account-based connection:

| | Nanobot (Tailscale) | OpenCode | Codex |
| --- | --- | --- | --- |
| Inbound exposure | none; Serve on tailnet | localhost; OpenTunnel for remote | none; cloud relay |
| Trust | tailnet membership + app pairing | one-time secret + password | ChatGPT account/workspace + MFA |
| Onboarding | Tailscale + Serve + origin | one command + scan | sign in + scan |
| Path | direct WireGuard/P2P | local or relay | relay |
| Full control | app-level owner scope | full server API | full host |
| Dependency | Tailscale account/daemon | OpenTunnel | OpenAI relay |

For **full owner control**, Tailscale is arguably the safer default: the host is
not exposed publicly and tailnet ACLs can restrict the port to owner devices.
The problem is not Tailscale itself; it is that pairing currently exposes all of
Tailscale's setup complexity to the user.

Recommended transport model:

1. **Tailscale direct** as the preferred transport for owner phones, with
   automatic status detection, login, Serve validation and origin selection.
2. **Optional E2E relay** as a fallback for phones outside the tailnet or users
   who do not want Tailscale. Pairing and scopes stay the same; only the
   transport changes.
3. Never make tailnet membership the authorization for full control. Keep the
   owner scope, device binding, short access tokens, audit and revocation in the
   application layer.

This is the same practical split as OpenCode (`pair` local vs `pair --remote`) and
Codex (host/relay), while preserving Nanobot's self-hosted privacy advantage.

## Suggested implementation order

1. Fix the immediate UX/security defects:
   - 5-minute TTL + countdown.
   - Remove the unused random installation parameter.
   - Auto-fill/validate origin.
   - Chat-only default; explicit owner confirmation.
   - Show Tailscale/Serve readiness before pairing.
2. Add pending-request state and one-tap approval:
   - `pairing.status` read (pending/collected/approved/expired).
   - Desktop notification/state when the phone collects.
   - Device name/platform in the request.
3. Add code fallback and device rename/last-seen.
4. Add optional relay/tunnel and long-lived session management.

## Decision needed

The main product choice is the security/UX tradeoff:

- **Fast path:** QR scan + one desktop approval, no comparison code.
- **Strict path:** QR scan + desktop approval + comparison code (current design,
  but with a 5-minute TTL and pending state).

Recommendation: fast path for the QR route, strict comparison only for manual
code entry. For a personal owner phone, default to **owner/full control with a
clear one-time confirmation**; offer **chat-only** as the guest mode. Use
Tailscale direct as the preferred transport and add an opt-in E2E relay fallback
so full control is still easy to reach outside the tailnet.

**Superseded 2026-11-03:** the accepted decision is owner-only pairing with no
chat-only option, no relay and no biometric gate. The review is retained as the
problem statement, not as current product direction.
