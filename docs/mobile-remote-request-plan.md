# Mobile remote correction and request settings plan

Planning baseline: 2026-11-03, `a90191a41b025884e76ce029b98a5d5349171f2e`,
branch `docs/mobile-remote-request-plan`. This is an implementation specification,
not implementation acceptance, release qualification, or permission to install.

## Authority and scope

This plan supersedes older mobile-plan statements about chat-scoped Full,
persistent-only Full, shared unfinished composer selections, and preserving legacy
chat-only access. The accepted owner-only cutoff remains: old chat-only devices
are revoked and must pair again with explicit desktop owner approval.
The release/installed-transition gates in [mobile-client-plan.md](mobile-client-plan.md)
and [mobile-development.md](mobile-development.md), including M38, remain applicable.

Keep the shared WebUI and current desktop agent/session/feature handlers. Native
Go/Kotlin owns Tailscale/TLS and device credential custody; `RemoteAccessOwner`
owns device identity, pairing, refresh and revoke; `DesktopPermissionOwner` and
existing folder/native supervision owners alone decide execution permissions.
Tailnet membership grants no Nanobot scope. Owner refresh rotates indefinitely
until revoke; new-device approval remains desktop-local. Remote Start, relays,
biometrics, another client/runtime/permission database, dependencies, offline
private history/write queues and incidental refactoring are outside this change.

Finish means the findings below are corrected, ordinary and temporary composers
offer the same supported request controls, queued submissions retain their chosen
context, and the current signed candidate has actual recovery/layout evidence.
Persistent chats reconcile after reconnect. A temporary chat that disconnected
has ended; smooth recovery must never imply its history or authority was restored.

## Session context lifecycle integration

The overall implementation also includes the coordinated backend track in
[session-recovery-architecture.md](session-recovery-architecture.md), updated after the DSH and
OpenCode comparison. It prioritizes actual-route capabilities and consistent prepared-request
accounting, reserved compaction capacity, model-derived summary budgets and a verified historical
prefix where compatible. Bounded emergency recovery follows correction of the ordinary lifecycle.
Keep source/checkpoint authority, per-request permission snapshots and temporary disposal intact.

Mobile A-D can proceed independently of backend P1/P2. Serialize overlapping configuration/core
edits with mobile E, and any compaction-status UI delta with F under one shared-file owner.
Reconnect reconciles canonical history/status; it does not replay a failed or uncertain request.
Keep backend continuity evidence separate from mobile recovery/layout qualification and reuse the
linked plan's focused use-case validation instead of duplicating tests. Both tracks need their
own accepted outcomes before claiming the expanded work complete; retained release gates apply.

## Evidence and current limits

Read `AGENTS.md`, `.agent/{design,security,gotchas}.md`, relevant contribution
checks and `docs/migration/p6-error-attribution.md`. Keep the agent core small,
validate dynamic inputs at owning edges, preserve atomic writes, run no `ruff format`.
The source review at the baseline found the following concrete gaps:

| ID | Observed behavior / owning code | Producing package |
| --- | --- | --- |
| R1 | `remote/edge.py` validates before awaited HTTP reads, then delivers without fresh authorization; a real-owner reproduction returned private data after revoke acknowledgment | B |
| R2 | `mobile-runtime.ts` keeps remotely closed subscriptions; broker `ws.go` always assigns socket ID 1; a reproduction delivered a frame to old and new callbacks | A |
| R3 | `remote/http.py` already serves independent owner status/refresh/devices/revoke; broker policy, Kotlin refresh and bootstrap still require gateway mobile routes | C |
| R4 | Native socket open failure emits error without close, so `NanobotClient` never enters its close-based retry path | A |
| R5 | Incoming WS base64 is decoded with raw `atob`, corrupting `你好 👋` in a reproduction | A |
| R6 | `RemoteAccessSettings.tsx` clears pending context on dialog close; reopening does not trigger restoration | D |
| R7 | `remote/pairing.py` fixes a 300-second deadline; `mobile-bootstrap.js` retries the same expired invitation after Tailscale login | D |
| UX1 | No native foreground recovery; overlapping refresh rotations, stale-open sockets, and history refresh limited to already-open active/unsettled chats | C |
| UX2 | Attachment/project/permission triggers differ; narrow-container CSS can shrink targets below 44 px | F |
| UX3 | App/server lock focus during work; Full requires an idle persistent chat; composer queue lacks settings snapshots | E, F |
| T | Literal operation totals, overlapping source-layout routing tests, test-only `ws_http.mobile_mutation_payload`, misleading reopen coverage and unstubbed localhost fetch failures | B, D, G |
| API | Shared feature handlers exist, but mobile reads use `/mobile/v1/{api,feature}?path=...` wrappers | G, separately after corrections |

Relevant foundations already exist: `RequestContext.workspace_scope`, immutable
`TurnRequest`, `WorkspaceScope`, private permission references and native retirement.
`agent/loop.py` binds workspace and references around execution near 2413–2465;
its `_source_execution` currently rejects temporary Full and must change narrowly.
`webui/workspaces.py` promotes restricted envelopes back to a chat's proven Full;
that behavior cannot implement explicit per-request Sandbox. `temporary_chats.py`
and `inbound_commands.cleanup_connection` currently dispose connection-owned chats.

Prior review evidence: 127 focused backend tests, 44 WebUI tests across eight
mobile/settings files, and cached broker Go tests passed. Diagnostic probes
demonstrated R1/R2/R4/R5; they are not committed regression cases. Source tracing
does not prove R7 or long-background recovery on a phone. There is no fresh
physical-phone, signed-candidate, strict-type or full-suite acceptance for this plan.
The plan is self-contained; the review's scratch files need not survive dispatch.

## Request settings and authority contract

### Four distinct states

1. **Remembered chat defaults:** last accepted submission's project and requested
   mode, plus a monotonically increasing defaults revision. They are preferences.
2. **Device-local composer:** text, attachments, selections and dirty state keyed
   by selected instance/chat on that device. Opening a chat seeds a clean composer
   from defaults; peer events can update its defaults cache, never dirty input.
3. **Submitted snapshot:** stable existing `turn_id`, text/media/mentions/quote,
   exact workspace/mode, source token and confirmation correlation, captured at
   the first submit/queue action, before asynchronous attachment conversion.
4. **Admitted execution:** current server-validated scope and opaque owner-issued
   proof bound to that exact new request and source lifetime. Only this is authority.

Selection changes remain usable while answering and say “For the next message”.
They do not invoke the old chat-wide mode/focus mutation. An accepted submission
may update defaults once, in server acceptance order; retries/replays cannot
update them again. Revision notifications never rewrite another device's local
draft, already queued snapshot or running request. No global draft synchronization.
Persistent default preferences may use additive versioned session metadata through
the existing session writer; temporary defaults and drafts remain in memory.
Do not persist temporary queued text/media in composer localStorage. No new
attachment/blob persistence or private offline outbox is required for ordinary chats.

### Public boundary and Full consent

Extend the existing authenticated `message` envelope rather than inventing a new
turn API. Keep `workspace_scope` and `turn_id`; add `request_settings_version: 1`,
`source_token`, `request_intent: "new" | "steer"`, optional `target_turn_id` for
steering and optional `full_confirmation_id`. Workspace display fields are not
trusted; the server builds effective `WorkspaceScope` using existing resolvers.
`access_mode: "restricted"` means Sandbox even if a prior request used Full.
Never promote it from a chat label or silently downgrade requested Full on failure.

Reuse `workspace.access.prepare_full` and `workspace.access.mode.set` with an
explicit `request_settings_version: 1` payload variant. Preparation returns the
current source token, runtime/revision and availability/reason. Full confirmation
includes chat ID, source token and the existing expected runtime/revision fields;
it returns a non-authorizing `full_confirmation_id`. The existing owner delegation
adapter authenticates remote Full confirmation; local WebUI authentication remains
the other valid boundary. The permission owner stores consent in process memory,
bound to authenticated device/client identity, chat/source lifetime and runtime.
An ID copied to another device/source cannot authorize or confirm anything.

This confirmation permits following requests in that chat to request Full without
another dialog while its consent remains current. It does not issue a reusable
execution proof. Normal access-token rotation and reconnect preserve the same
device/source-bound consent; they do not change submission or operation identity.
Each new request still needs fresh admission. Focus changes do
not grant folders; Full confirmation explains its existing broad access meaning.
For an ordinary unsaved chat, reuse explicit ordinary-chat preparation to establish
its real canonical source, initially Sandbox; do not persist its unfinished focus
or mode. Temporary preparation uses only its already-active transient owner.

Persistent Full **preference** survives reconnect/restart, but runtime restart,
source reset/replacement/deletion, explicit revoke or loss of the confirming device
invalidates consent. Show “Full selected · confirmation required”; sending waits
for explicit reconfirmation. Do not reconstruct consent/proof from metadata.
Other devices may seed the Full preference but must confirm for themselves.
Choosing Sandbox changes only future input. Keep explicit access revocation and
Stop as distinct actions; the old restricted-mode authority action can remain
the explicit revoke path, with existing fencing and retirement behavior.

### Binding, admission and retirement

Persistent bindings use the current canonical store identity/incarnation/generation.
Temporary bindings come from `WebUITemporaryChats`: private runtime-owned identity
and a fresh incarnation for that active chat/connection, with a generation fence.
Reuse `PermissionSourceBinding`'s shape with a typed internal lifetime distinction;
never manufacture a canonical source, source file or persistent session for Full.
Only the existing transient owner can resolve that binding; client fields cannot.

Separate consent from request Full references inside `DesktopPermissionOwner`.
Bind each reference to source lifetime, admitted `turn_id`/input identity and epoch;
validate that same identity where the current turn accepts/binds the reference.
Retain object-identity proofs and launch-edge `with_current_reference` checks.
At most one primary execution owns a chat. Reuse current task/Full retirement,
residual process accounting and source lifecycle fencing; no new permission store.
The native ledger can keep its current shape and historical `chat_full` kind:
use request identity in `owner_id`, retain `task_input_event_id: null`, and keep
per-session residual counts spanning source generations. That accounting record
does not confer authority; opaque temporary identifiers contain no transcript.
Do not add a new native ledger version just to rename the historical kind.

Ingress validates and freezes intent. At the actual new-request admission boundary,
after any queue/wait and before model/tool dispatch, recheck current device/owner
authorization, source lifetime, Full consent, folder/path policy, configured
sandbox, supervisor availability, permission proofs/revision and old-process state.
Use the existing controller/permission owner and narrow existing admission hook;
the loop change is limited to this handoff/validation and the transient branch.
Issue the request proof there and bind it through existing request/run contexts.
No receipt, process label, remembered mode or client-supplied reference is proof.

Changes between preparation and commit fail with current projections and a reason
(`source_changed`, `permission_changed`, `confirmation_required`, policy denial).
Revalidate live policy at native/file operation boundaries as today. Defaults
revision changes alone do not retarget or reject an explicit submitted snapshot.
Unsafe active/unknown prior commands yield a visible waiting/blocked admission;
selection remains editable. Do not clear reservations, waive supervision or treat
“no current turn” as quiescence. Existing Once grants remain bound to their exact
task/input, with their current expiry; they never become a sticky default.
An admitted run and its children keep their captured context across picker changes.
At completion/cancel/reset/discard, retire its proof using existing rules; drain or
fence residual processes and require safe admission before the next incompatible run.
Revocation can fence future launches; choosing Sandbox does not confine an old process.

### Queueing, steering and replay

Reuse the composer queue and current server request ownership; do not activate the
dormant generic busy-inbox subsystem or add a durable WebUI request queue.
Every queue entry contains its complete submitted settings. `ThreadShell` and
`SendOptions` must forward that snapshot, never substitute current picker state.
Same-context guidance may use existing steering only for its exact active turn.
A different workspace/mode is a NEW request queued behind it; the server independently
checks equality and target identity and never silently turns it into steering.
If another device starts work before arrival, a non-admitted busy response leaves
the item waiting with its snapshot and no user-message/default/agent effect.
Use an explicit `not_admitted` disposition so a later deliberate admission attempt
can use a new transport operation ID with the same stable submission identity.

Keep existing operation IDs, payload digests, replay cache and unknown-outcome rules.
Include settings, source/intent/target and attachment identities in the digest.
Same-ID/different-payload conflicts; same-ID replay cannot issue a second proof,
start another run or update defaults twice. Authenticate again before replay join,
new admission and result delivery, using current authentication rather than a
frozen access token captured with the submission. Tokens, expiries and authentication
rotation/version counters stay outside the immutable payload digest; stable
device/submission identity and existing replay epoch/generation fences remain.
Unknown/evicted/old-generation outcomes require
canonical reconciliation or explicit user retry, never automatic fresh submission.
Reconnect does not authorize requeueing an uncertain mutation. A rejected stale
Full/source snapshot stays visible for review; reconfirmation creates a new explicit
submission without changing any earlier queued request behind the user's back.

### Temporary lifetime

Keep connection ownership and disposal on actual disconnect, explicit discard,
revoke/logout or gateway shutdown. Cancellation, media deletion, session invalidation
and late-event tombstones remain effective; no attach to a discarded temporary ID.
If foregrounding finds the same healthy connection, continue that live chat.
If reconnection was necessary, show that the temporary chat ended. Preserve unsent
input locally in memory while the current app survives, mark old queued submissions
unavailable, and offer explicit creation/send in a new temporary chat. Never replay
automatically, retain offline temporary history or promise survival of process death.
The screen may retain an ended-chat notice; it must not present deleted server
attachments/history or lost Full consent as recovered. This is the privacy exception
to persistent-chat history reattachment, not an invisible reconnect failure.

## Transport, delivery and pairing contracts

- Every socket terminal path unsubscribes once and produces one close, including
  open failure and close during an outstanding open. Reject late open/frame/close
  events for retired instances. Broker IDs are unique across connections and
  transport restarts in a live bridge, or paired with a distinct transport generation.
  Guard shared-client callbacks by current socket identity. Decode text with the
  existing UTF-8 `decodeBase64`; preserve frame/queue/time/byte bounds.
- Gate all private HTTP results at the final mobile response boundary, including
  thread/feature/media, cached results and post-wait errors containing private data.
  Revalidate against owner IPC **without the admission cache** immediately before
  authorizing delivery. An acknowledged revoke preceding that check yields no
  private payload; owner unavailability fails closed. Preserve task cancellation
  semantics and response bounds. Bytes already delivered before revoke cannot be
  recalled; a genuinely streamed body needs authorization at its delivery chunks.
- Native named operations expose only existing `/remote/v1/status`, `/refresh`,
  `/devices` and exact device-revoke routes, with fixed methods/limits/credentials.
  Bootstrap can open an owner-control screen with the gateway absent; feature/chat
  readiness remains a separate gateway handshake. No remote Start or arbitrary URL.
- Serialize refresh natively for the exact installation/custody generation: all
  bootstrap, timer, foreground and 401 callers share one in-flight rotation. Save
  the rotated pair atomically before releasing waiters; logout/revoke prevents an
  older completion from restoring custody. Expose expiry/status, never tokens, to JS.
- Route native foreground and DOM visibility through one in-flight recovery path.
  Probe an apparently open socket using native `wsPing(wsId)`: the current Go
  `websocket.Conn.Ping` waits for Pong with a five-second context deadline through
  the narrow C ABI/JNI/Kotlin bridge. This adds no application-message ping service
  or new socket. A successful probe with still-valid socket authentication keeps
  the connection. Otherwise renew credentials as needed, retire it once, open
  one replacement, reattach persistent chats and reconcile canonical history/run
  state, including idle chats changed elsewhere. Keep current screen, draft,
  attachments and scroll anchor; ordinary outages do not reload/remount the app.
  Distinguish reconnecting/offline, tailnet sign-in, missing credentials, gateway
  stopped, update-required and confirmed revoked states. Outage preserves pairing;
  confirmed revoke fences protected UI, clears credential/private media custody and
  leaves any safe unsent draft unsent. Do not label generic 401 as confirmed revoke.
- Closing the desktop pairing dialog is presentation only. Reopening loads current
  pending state; explicit cancel/deny are separate. Preserve mount restoration.
- Retain the same unconsumed QR/code/nonce/collector secret for login continuation.
  Add a **desktop-local explicit Renew setup** action on that pending offer: extend
  its deadline to `min(now + 300 seconds, created_at + 1800 seconds)` under the
  existing pairing lock, retaining any collector/device binding. Never renew after
  deny/cancel/approval/consumption or beyond the 30-minute overall setup limit.
  Retain eligible expired offers only until that limit; expiry remains enforced
  until local renewal. Approval still compares the actual collected phone/code.
  The phone retains its pending setup in native custody, retries the same offer
  after Tailscale readiness, and resumes polling without rescanning once renewed.
  If still expired, explain “Continue setup on the desktop”; use bounded/rate-limited
  retries. After the overall limit a new offer is required and is clearly reported.
  No public renewal endpoint, remote authorization by expired code, indefinite
  invitation or renewal after a single-use result exists.

## Versioning and compatibility

Advertise a request-settings contract version, owner-control support and later
standard-feature-dispatch support through existing versioned handshake projections;
update strict contract parsers, native bridge projections and paired copies together.
Treat permission projection, selected defaults and active request scope as distinct
fields. Missing support disables submission/unsupported controls with a visible
update-required reason; no best-effort Full, invented defaults or forged localhost.

Server rollout first adds negotiation and safe adapters; request-settings activation
then requires version 1 on WebUI message/control submissions using the new semantics.
Older owner clients may retain compatible reads/owner control, but context-mutating
legacy submissions/Full activation receive explicit update-required, not reinterpretation.
New clients facing an old server keep pairing and explain the required backend update;
they do not submit the new contract through old Full actions. The owner-only security
cutoff is permanent and not a compatibility adapter. Cover these mixed-version
outcomes before activation. Existing request IDs and replay epochs remain fenced.
Preferences metadata is additive/versioned; old Full labels never migrate into consent.
If a future change actually needs a stored schema version, specify old/new reader
behavior and downgrade handling first; do not silently rewrite permission/native stores.

## Acceptance criteria

| AC | Binary observable finish line |
| --- | --- |
| 1 | Revoke acknowledged before delivery authorization prevents every private HTTP result, even with a warm admission cache; owner loss fails closed |
| 2 | Repeated disconnect/open failures recover with one live socket, one delivery per frame, correct Unicode and no retired listener effects |
| 3 | A paired phone can status/refresh/list/revoke through the independent owner with gateway stopped; Start is visibly unsupported |
| 4 | Closing/reopening restores the pending pairing; a locally renewed offer resumes after login without rescan; expired/denied/consumed/out-of-limit offers cannot authorize |
| 5 | Chat defaults are remembered; new chats start Sandbox; peer updates never replace dirty local drafts/choices; restart requires fresh Full consent |
| 6 | Settings remain editable during streaming; queued/attachment-delayed submissions execute their original snapshot; context-changing guidance becomes a new request |
| 7 | Both chat types admit confirmed Full through the same permission owner; forged/stale preferences or consent cannot grant it; temporary work creates no durable chat/grant |
| 8 | Focus alone grants nothing; active/unknown earlier commands block unsafe admission visibly; admitted work/children retain context; Stop/revoke stay distinct |
| 9 | Foreground/network/token recovery preserves persistent UI/input/scroll and canonical history without reload; a disconnected temporary chat is explicitly ended, never auto-replayed |
| 10 | Attachment/project/permission use consistent existing primitives, compact two-row layout and measured >=44 px targets at narrow phone widths; project text truncates first; Full and unavailable reasons are understandable and accessible |
| 11 | Matching and mixed versions behave as specified; replay has one effect or explicit unknown outcome; new standard dispatcher preserves auth, bounds and handler behavior |
| 12 | Current source/build/live/signing evidence is distinguished; required M38 review and concrete installed-transition opt-in precede any promotion |

## Bounded work packages and ownership

Default to sequential writers. A and B can run in parallel only with disjoint
test files and explicit ownership; all later shared-file handoffs are sequential.
The orchestrator accepts this specification, checks worker availability at dispatch,
reviews actual deltas/evidence and returns corrections to the owning worker.
No automatic extra planner/reviewer/test pass is required for each package.

**A — Socket correctness (start here; R2/R4/R5, AC2).** Own
`webui/src/lib/mobile-runtime.ts`, socket lifecycle in `nanobot-client.ts`, broker
`core.go`/`ws.go`, and their closest existing WebUI/Go test files. Inputs are the
socket contract above and independently authored public-boundary cases. Preserve
current retry policy; handle terminal/open races and identity without transcript
deduplication. Run, from `webui`, `bun run test -- src/tests/mobile-runtime.test.ts
src/tests/nanobot-client.test.ts`; from repository root source
`mobile/scripts/mobile-env.sh`, then in `mobile/native-network` run
`go test ./internal/broker`. Completion includes actual commands/exits and AC2
evidence bound to final file hashes. No Android installation is needed for A.

**B — Delivery authorization (R1, related T; AC1/11).** Own
`nanobot/remote/edge.py`, the final `webui/gateway_endpoint.py` delivery boundary
as needed, and closest `tests/remote`/mobile HTTP boundary tests. Use uncached owner
validation and preserve existing cancellation/replay semantics. Relocate useful
owner-denial/local-enrollment cases to the authenticated public WS entry before
removing the test-only `webui/ws_http.py` helper; do not remove their protection.
Focused command: `.venv/bin/python -m pytest tests/remote/test_edge.py
tests/remote/test_edge_integration.py tests/webui/test_owner_mobile_api.py
tests/webui/test_mobile_owner_feature_api.py tests/webui/test_mobile_commands.py -q`.

**C — Owner wiring and foreground recovery (after A/B; R3/UX1, AC2/3/9/11).**
Own native `NativeNetworkPlugin.kt`, custody synchronization in `CustodyStore.kt`,
`MainActivity.java`, broker `policy.go`, `mobile/mobile-bootstrap.js`, shared
`mobile-runtime.ts`/`bootstrap.ts`/`nanobot-client.ts`, and the recovery-only
`App.tsx`/`ThreadShell.tsx` delta. The Ping seam also owns the broker connection
method, existing Go C ABI exports/header, `android/app/src/main/cpp/native_network_jni.c`
and `network/NativeNetwork.kt` declaration, with matched ABI/bridge capability checks.
Add only named owner operations, native refresh
single-flight and the shared recovery path; reuse existing thread reconciliation.
Use the bounded native Ping/Pong contract above. Focused checks:
WebUI `mobile-runtime`, `mobile-http-bridge`, `mobile-wiring`, `mobile-shell`,
`bootstrap`, `nanobot-client`, `thread-shell` test files; backend
`tests/remote/test_listener.py tests/remote/test_auth.py`; broker command from A.
Kotlin/bridge changes also need the build/instrumentation checks below.

**D — Pairing continuity (after C; R6/R7, related T; AC4).** Own
`remote/pairing.py`, its local IPC/settings operation and bounded contract changes,
`RemoteAccessSettings.tsx`, bootstrap/Kotlin pending setup state and closest pairing
tests. The renewal boundary is local authenticated desktop control only. Replace
misnamed reopen coverage with the actual interaction, preserve mount restoration,
and isolate all background status/pending fetches to eliminate localhost leaks.
Focused checks: `.venv/bin/python -m pytest tests/remote/test_pairing.py
tests/remote/test_owner_ipc.py tests/webui/test_remote_access_settings.py -q`;
WebUI `remote-access-settings.test.tsx mobile-pairing-identity.test.ts`; applicable
existing Android `NativeNetworkPairingInstrumentedTest` and pairing-link checks.

**E — Request authority/backend contract (after D; UX3, AC5–8/11).** Own
`security/desktop_permissions.py`, `webui/workspaces.py`, `temporary_chats.py`,
`inbound_commands.py`, permission/delegation adapters and versioned contracts,
session preference/lifecycle integration, and the minimal admission/context delta
in `agent/loop.py`, `agent/turn.py`, `agent/tools/context.py` and WS runtime.
Do not rewrite the loop/runner or activate busy-inbox routing. Implement the four
states, ephemeral binding, consent and request proof before F consumes them.
Focused command: `.venv/bin/python -m pytest
tests/security/test_permission_v2_authority.py tests/security/test_native_admission.py
tests/webui/test_permission_v2_api.py tests/utils/test_webui_workspaces.py
tests/agent/test_workspace_scope.py tests/session/test_p5_stage2_temporary_sessions.py
tests/remote/test_owner_permissions.py tests/remote/test_owner_delegation.py -q`.
Include the affected temporary/workspace public WS boundary cases in
`nanobot/channels/websocket/tests/test_websocket_channel.py`; run its affected
existing selectors plus independently authored new use cases, not unrelated channels.

**F — Composer settings and presentation (after E and C handoff; UX2/UX3,
AC5/6/9/10/11).** Own `App.tsx`, `ThreadShell.tsx`, `ThreadComposer.tsx`,
`WorkspaceControls.tsx`, `hooks/useNanobotStream.ts`, client wire/types, existing
composer CSS and required translations. Store local selections/dirty state,
snapshot before attachments/queueing, distinguish steering/new input, preserve
temporary privacy and render selected versus active authority accurately.
Use existing Button/menu/popover/dialog primitives and shared classes; keep model
row and folder/Full flows. Align target/icon/border/focus/menu treatment without
shrinking controls; expose selected Full, pressed/loading state and unsupported reasons.
Run existing `thread-composer`, `thread-composer-attach`, `thread-shell`,
`app-layout`, `useNanobotStream`, `nanobot-client` test files and `bun run build`.
Visual/touch/keyboard/accessible-name evidence is required; class assertions alone
cannot establish AC10. Preserve desktop layout and existing locale conventions.

**G — Standard feature dispatcher (separate later change after A–F; API/T,
AC11).** Own shared API transport dispatch, mobile adapters, exact broker operation
registry, `mobile/contracts/{desktop-parity,operations-v1}.json`, server operation
copy and corresponding checks. Admit authenticated native requests to the same
standard method/path/WS-action dispatcher and existing feature handlers as desktop;
no arbitrary-path proxy. New capable clients use standard routes; retain thin
version-negotiated wrapper adapters only for supported owner builds until the
explicit tested update-required cutoff. Preserve method/query/encoded-path/media
bounds, local-only enrollment and handler/error/auth/replay semantics. This is
not a prerequisite for urgent transport/admission fixes; it is required before
claiming the target architecture/full parity complete. Replace literal totals and
duplicate AST/string routing checks with one contract-drift check and real route
outcomes. Commands: `.venv/bin/python mobile/scripts/check-desktop-parity.py`,
`.venv/bin/python -m pytest tests/remote/test_operations.py -q`, affected
mobile API boundary tests and `mobile-http-bridge`/`api` WebUI tests. Regenerate
both operation copies with the existing builder only when their inventory changes.

## Validation, qualification and release

Reuse prior passing evidence only while its relevant source, inputs and dependency
environment remain identical. Do not rerun the earlier 127/44 checks for this
documentation edit. Add/change cases only for the concrete AC gaps above, from a
fresh `REQUIREMENTS_ONLY_TEST_AUTHOR` session with no parent history/repository/code,
tests/diffs/logs/implementation analysis. Supply only accepted user outcomes and
the explicit public UI/protocol inputs/outputs/error semantics. Preserve its
artifact/session ID; workers may wire fixtures/runners mechanically, not change
scenarios or expectations to fit code. No executable case matrix is prescribed here.

For changed Python, run `.venv/bin/ruff check` on owned runtime files. Preserve the
mandatory type-regression gate: freeze source/environment/checker/dependency hashes,
run before/after full-source `uv run --no-sync basedpyright --outputjson` under the
same environment, retain commands/exits/reports and compare with
`scripts/check_type_regressions.py BEFORE.json AFTER.json --baseline-root BEFORE_SOURCE
--current-root AFTER_SOURCE`. Zero introduced signatures across all files is required;
counts alone are insufficient. CONTRIBUTING setup is `uv sync --all-extras --dev`
then `uv run --no-sync python -m scripts.install_channel_dependencies --all-channels`;
finish setup before freezing both reports, never between them. Local `uv` is available
through `mobile/scripts/mobile-env.sh`; tool presence is not checker/test evidence.
Do not absorb errors by refreshing a baseline. Full unchanged strict exit zero and
the required full local `uv run --no-sync python -m pytest` are required to close P6
under AGENTS.md. A bounded correction needs its affected checks and type-regression
gate; it does not itself require or establish P6 closure. Retain applicable CI and
the existing mobile integrated-release checks; unrelated failures remain reported
and unwaived under the gate they actually affect.
Run required WebUI lint/build once on integrated changed inputs; broaden other
checks only for a demonstrated regression risk or retained release gate.

Native changes: use `mobile/scripts/build-native-network.sh` and
`mobile/scripts/test-cabi.sh`; these include existing Go test/vet and ABI checks.
Build shared assets with `mobile/scripts/build-shared-webui.sh`, sync Android from
`mobile` with `bunx cap sync android`, and build with `./gradlew --no-daemon
:app:assembleDebug` from `mobile/android`. Use existing
`mobile/scripts/qualify-emulator.sh --evidence-dir <private-run-directory> --native-tests`
on an opted-in disposable emulator. Existing refresh/WS/custody instrumentation
provides foundations, not automatic proof of foreground journeys; execute only
applicable classes with `:app:connectedDebugAndroidTest` and the runner class filter.

Qualify on an actual supported Android phone: long background/Doze beyond the
10-minute access lifetime, return during/after work and an idle desktop-side change,
Wi-Fi/cellular/offline recovery, tailnet re-login, gateway absent/restarted, token
rotation and revoke, while retaining draft/attachments/screen/scroll. Include the
temporary-ended distinction and narrow toolbar interaction/accessibility. Record
OS/WebView/device, durations, before/after state, actual public outcomes and failures.
Emulator passes, arm64 compilation and source inspection are separate evidence.
No new general device harness or proof from a mocked lifecycle callback is needed.

Use `mobile/scripts/build-release.sh` only with an available authorized signing
configuration; do not invent/read out keys. Record final APK at
`mobile/android/app/build/outputs/apk/release/app-release.apk`, retained
`nanobot-mobile-release.apk`, SHA-256, certificate/`apksigner` and `zipalign` evidence.
An exact signed-candidate smoke and in-place update/custody check are release gates;
debug `test-update-persistence.sh` is supplemental and may uninstall only disposable,
explicitly authorized app data. Never invoke its destructive opt-in on the daily app.

Stage isolated backend/owner compatibility first, then matching shared/native client,
then activate request semantics; qualify G separately and integrate its affected
evidence before full-parity acceptance. Freeze server/package/WebUI/APK hashes,
protocol/registry/schema versions and selected isolated instance/environment.
Preserve the genuine **M38 independent integrated release review**, required by
the existing mobile plans; it is not a reviewer gate for each repair. Changes after
review invalidate affected qualification/review only. M39 installation is a later
concrete opt-in with actual instance/config/data/executable/launcher/supervisor and
owned mapping transitions, backup/compatibility and rollback prepared for review.
No planning step changes installed services, credentials, Tailscale or runtime data.

Rollback disables incompatible remote ingress and stops the prior writer before
changing versions. Retain identity, revocations, audit, sessions and authorized writes;
never restore revoked credentials or durable Full from an old backup. Preserve
temporary disposal. An older backend may resume remote access only if its readers,
protocol and permission semantics are compatible; otherwise retain local recovery
with ingress off. Remove only owned mappings and explicitly disposable fixtures.

Planning readiness: ready for bounded implementation, starting with A/B. Product
decisions are settled; no routine implementation permission question is needed.
Signing/physical-device availability and exact external installation/network targets
remain unqualified execution inputs, not reasons to stop isolated corrections.
Each worker reports owned delta, AC/finding results, final hashes and actual checks,
failures/limitations and remaining blockers. The orchestrator accepts implementation;
this document does not claim that any product correction has already passed.
