# First-release product decisions

**Status: first-release scope, interaction/UI and all four product defaults selected by
the user. Frozen as `A3-PLAN-2026-11-11`; implementation acceptance remains separate.**

This extends the [accepted architectural direction](plan-review.md#user-acceptance).
It is a planning record. No nanobot implementation source was inspected and no product
tests were executed to prepare these choices.

## Selected: first release without legacy migration

On 2026-11-11 the user said: "for the migration, i think we probably dosent need it
since we are developing the first version only". The current plan therefore targets one
initial supported protocol and storage format, without a released-version migration
requirement.

Do not require a production v1-to-v2 cutover, legacy bearer coexistence, forced re-pairing
of an installed user population, historical executable readers or a two-version APK
upgrade rehearsal for this first release. Version the initial wire and persisted formats
explicitly so later incompatible changes can be identified. A candidate's development
version number does not establish a previously released compatibility obligation.

Development state can be explicitly exported or reset if necessary. This planning choice
does not execute a reset or silently discard histories, revocations or uncertain effects.
Ordinary restart/reconnect, cancellation, lost-response recovery, revocation and one live
writer remain first-release requirements. An older development build must not revive
authority from current state; disabling an unsupported path is sufficient when it cannot
read that state safely.

Historical audit packets remain unchanged. Their legacy migration/upgrade criteria are
outside the selected first-release scope. Findings concerning current authority,
recovery, data integrity and duplicate live owners retain their applicable criteria.
The strict-checker regression contract also remains applicable; its migration label is
independent of a released-product migration requirement.

## Subagents controlled through the main chat

"Child" means a delegated subagent. The user questioned the earlier direct-control
proposal and described using the original chat/main agent to control subagents, as in
Codex and OpenCode. The [source comparison](reference-design.md#d3-follow-up-main-chat-control-in-codex-and-opencode)
supports that interaction. The earlier six-control table is superseded as a public
interface requirement.

These are subagents working within a normal session. Future SuperAgent coordination
works between normal task sessions; each keeps using its ordinary subagents. The user
explicitly required that future direction not affect current subagent behavior. The
[future extension contract](future-harness-extension.md#session-coordination-and-subagents-are-separate-layers)
records that separation without changing the current delegation contract.

The user talks to the main agent. It launches subagents, assigns work, receives results,
routes follow-ups and requests cancellation through internal tools. Main-chat activity
shows a subagent's task, status and result; optional authorized read-only inspection
helps debugging. A separate subagent composer, sidebar task manager, manual archive/
close/resume controls and public mid-run steering API are outside the initial requirement.

Use canonical child sessions and the normal runner/history path, with a separate identity
for each active run. Durable child history does not require an independently managed user
conversation. Supply the native nanobot backend for this release; external agent backends
and a general start/resume/send/steer/interrupt/wait/close adapter framework are unnecessary
for this initial scope. Governing instructions, disabled capabilities and narrowed
authority apply independently of the child's prompt/history selection.

| Main-chat behavior | Minimal internal responsibility |
| --- | --- |
| Delegate work | Admit one bounded assignment under the parent and launch it through the normal child runner. |
| Report progress/results | Return child status and results to the parent. Wait/collect without a duplicate run or holding an unused model-request slot. The main agent integrates the result. |
| Change an assignment | The main agent sends a durable ordered follow-up to the same owned child through normal input admission. Start a turn when idle; queue at the next safe model boundary when busy, without rewriting completed effects or running concurrent turns. A separate public steer protocol is unnecessary. |
| Stop work | The existing parent Stop fences new admission and cancels its owned active/queued subtree without waiting for another model response. Targeted child cancellation can be requested through the main agent; siblings remain independent. |
| Finish/release resources | The runtime releases settled execution resources and preserves canonical history. This does not require a user-facing close/archive operation. |

Internal mutations target the current child incarnation/run under current owner and
parent authority. Duplicate attempts reconcile by identity. Accepted/queued remains
distinct from applied/settled. Risky child effects use the same main-chat approval flow,
bound to the actual child's effect; a model message cannot create human consent.

Awaited children are the selected default. The main agent may explicitly use internal
background execution for independent parallel work; completion returns to the same
parent chat. A normal parent reply is different from Stop, and may state that delegated
work is still running. The parent remains responsible for that work. This does not create
a separate user-managed background-task product. Shutdown attempts owned settlement and
preserves uncertainty where settlement is unproven.

Restart reopens history and reconciles owned work. It does not automatically restart a
child, replay effects or restore Full consent. Continuation requested through the main
chat requires fresh admission and reconciliation of outstanding effects, with a new run
identity. Required uncertain-effect evidence remains visible until reconciled.

This narrows the product surface while retaining canonical source, exact consent, live
revocation, bounded admission and real cancellation/settlement obligations. The reference
implementations are engineering evidence, not a feature-parity requirement.

## Client UI behavior for subagents

On 2026-11-11 the user approved the main-chat control direction and required the client
UI to align with the reference harness behavior. Follow Codex's documented main-thread
activity/inspection flow and OpenCode's child status/back-to-parent pattern, as recorded
in [the UI reference comparison](reference-design.md#d3-client-ui-behavior).
Use existing client conversation, approval and navigation surfaces. The UI selection
preceded the later explicit selection of lifetime, budget and retention defaults below.
Read-only inspection here applies to delegated-helper detail. Normal/direct conversations
retain their own composers. A future managed task opens its actual normal session for
authorized direct collaboration; it is not a separately controllable subagent view.

| User interaction | Required behavior |
| --- | --- |
| Follow delegated work | Show a compact activity item in the original conversation with a meaningful task label, current state and brief result or failure. The main agent integrates the answer; do not copy every child message into the parent transcript. Refused admission must not appear as running work. |
| Inspect details, when offered | Open authorized read-only child activity/history from that item. Show known status and a clear return to the original chat, preserving its draft and position. The child view has no independent composer or lifecycle management panel. |
| Give a follow-up | Use the original chat composer. The main agent routes the instruction through normal child admission. Queued/accepted feedback must not claim the child has applied it. |
| Stop owned work | Keep the original chat's Stop available while its owned delegated work is active, including after an ordinary parent reply during explicitly requested background execution. Stop reaches the runtime directly, fences admission and cancels the owned subtree. Show stopping until settlement is proven; incomplete proof remains visibly uncertain. A targeted request goes through the main agent. |
| Approve a child effect | Use the existing approval flow in the original chat. Identify the subagent task, action and effective target/scope being approved. Bind the decision to that pending child effect. Stale, cancelled or revoked approvals cannot apply; there is no separate child permission editor. |
| Reconnect or reopen | Restore the same canonical activity/results and current owner status, merging duplicate or late events without duplicate rows, runs, input or approvals. While disconnected, show that status is unavailable rather than infer completion. History reopening does not restart a child or restore Full consent. |
| Understand completion | Distinguish working, waiting for approval, stopping, completed, failed/interrupted and outcome needing reconciliation as applicable. A parent reply, transport acknowledgement or root-process exit is insufficient to mark all delegated effects completed. Unknown outcomes remain visible. |
| Use another client | WebUI/desktop, TUI and Android remote preserve these meanings in their existing layouts. Compact/touch navigation may replace a desktop detail pane. Task/status text and keyboard/touch access must carry the meaning without relying on color or hover alone. Android remains a remote client. |

Activity is a projection of canonical child history and current execution state, not a
second client-owned task store. Stable run identities and ordering stay in the protocol;
ordinary UI uses task names and clear states. Read-only detail is optional; a separate
sidebar task product, per-child control buttons and archive/close/resume UI are unnecessary.
Backend authorization still governs every read and mutation, independently of visible controls.

### Later UI acceptance

After the plan is frozen, verify the following through the real client/runtime boundary:

| Scenario | Observable criterion |
| --- | --- |
| Two siblings complete or one fails | Activity remains attributed to each task, with an integrated parent answer and no duplicated child transcript. Capacity refusal creates no running item. |
| Inspect and return, if details are offered | Authorized detail is read-only; return preserves parent context/draft. A historical child cannot control a current run. Check keyboard and narrow/touch navigation. |
| Follow-up and risky action | The original composer routes the follow-up once; its accepted state differs from application. The child effect waits for one exact approval in the existing parent flow; stale/revoked decisions start no effect. |
| Stop during a child tool or pending approval | Original-chat Stop works without another model response, prevents late admission/approval, and keeps stopping/uncertainty visible until the owner establishes settlement. Include a nested child and explicit background work after a parent reply. |
| Disconnect, duplicate/late events and restart | Rehydration gives one canonical activity/result, no replay or restored consent. A late old-run event cannot overwrite current state or falsely report success. |

These are planning acceptance criteria, not claims about the current nanobot UI. Codex's
private desktop source was not inspected. The reference UI components do not establish
nanobot's cancellation, approval or restart guarantees; those follow its accepted owner
contracts and require later implementation evidence. A3-4 owns shared client admission/
Stop/approval behavior, A3-5 owns subagent projection and lifecycle integration, and A3-7
owns Android's remote-client alignment. Reuse matching Linux/Android evidence later;
macOS remains the outstanding executor qualification.

## Selected: finite child admission defaults

| Budget | Selected default | Reason |
| --- | ---: | --- |
| Maximum ancestry depth | 2, with the root at 0 | Permits a child and one nested delegation level. |
| Active immediate child runs per parent | 2 | Permits sibling work without uncontrolled fan-out. |
| Active descendant runs per root | 4 | Bounds a complete delegated task tree. |
| Active child runs across the gateway | 4, excluding primary/root runs | Keeps multiple roots from multiplying child concurrency. |

These are conservative initial defaults for a personal agent, not measured optimal
values or copied reference requirements. The historical candidate's observed 2/2/4
values did not supply acceptance. The selected gateway cap additionally bounds roots
together. Configuration may change finite limits explicitly; children cannot raise them.

Reserve before launch. Active orchestration/waiting children count as child runs; idle
durable histories do not. Known non-start and settled runs release their reservation.
Uncertain work retains ownership/reservation until reconciled. Capacity exhaustion
returns an explicit refusal; it does not create an unbounded hidden spawn queue.
The caller may choose serial work within its existing authority.

These are selected limits on actual subagent runs. A future normal task session does not
consume a child-depth level or child reservation merely because the main conversation
routes work to it. Actual subagents inside any session keep using normal subagent limits
and behavior. Future task admission and provider capacity require separate policies;
this distinction does not change the selected numerical defaults or ordinary delegation behavior.

Preserve active provider/backend request controls and existing tool, input-queue, process
and output limits from the selected staging baseline. This does not restore a retired
all-provider turn semaphore. Provider-request capacity remains a separate owner. Creating a child
does not bypass the governing task's authority or configured aggregate budgets. This
selection makes no token-price claim or performance promise.

## Selected: initial Android device-proof contract

Use device-key pairing as the initial authentication contract. A non-exportable Android
Keystore P-256 key produces ECDSA/SHA-256 signatures. Public keys use X.509 SPKI DER,
signatures use ASN.1 DER, binary transport uses unpadded base64url, and the key fingerprint
is SHA-256 of the exact public-key DER bytes. The proof binds
protocol/purpose, instance, expected TLS origin, installation/device generation, challenge,
nonce and expiry. A fresh expiring challenge is consumed once before scoped token issuance.
Private keys and bearer tokens stay outside the WebView.

Signed claims use a fixed typed schema and UTF-8 JSON with sorted field names, compact
separators and no extra whitespace. Require every declared claim, reject unknown fields,
use integer Unix seconds for expiry and forbid floating-point claims. The expected TLS
origin is fixed by the admitted owner/connection before signing, not supplied by an
untrusted forwarded header. Shared native/server byte vectors define exact serialization,
encoding and rejection behavior before implementation acceptance.

Lost token delivery recovers using a fresh challenge and the same approved key.
Revocation prevents issuance and use. Cancelled or superseded pending attempts cannot be
revived by late responses or restart. Unknown versions and unsupported operations refuse
explicitly. Define and freeze canonical byte vectors before later implementation review.

This follows the first-release security portion of [A3-7](../../audits/implementation-packets/A3-7-mobile-pairing.md#1-cut-over-the-pairing-and-access-protocol--r01-r03-r11).
Its legacy cutover and re-pairing sections are superseded by the selected first-release
scope above. Selecting an initial device proof is separate from migrating old devices.

## Selected: output and retention

Keep existing finite output-size and admission limits. Live polling uses a bounded view
with explicit cursor/loss/truncation. Completion inspection uses the retained bounded
transcript or durable tool-result summary, independently of output already polled.
Raw process buffers can be released after required result recording and settlement;
their reclamation does not delete canonical tool/session evidence.

Regular conversations and child histories have no automatic age-based deletion in this
release: keep them until explicit deletion through their owner. Existing explicitly
temporary-session behavior remains separate. Releasing subagent execution resources does
not delete history.
Attachments are reclaimed only after complete reference checks prove them unreferenced.

Retain unresolved effect/source/receipt and revocation evidence. Storage pressure can
refuse new admission before effects and offer explicit cleanup through the existing owner.
Do not evict unresolved evidence, silently trim canonical history or add an arbitrary
retention duration to make room. A future automatic expiry policy is a separate change.

## Final selections for plan freeze

On 2026-11-11 the user said "ok lets do 1" in response to the numbered next steps.
Step 1 was to settle this exact four-default table; step 2 was to commit/freeze the
documentation and merge it into staging; step 3 was implementation review. This selects
all four defaults below. The user subsequently authorized step 2 with "ok lets do it";
the [freeze record](plan-review.md#frozen-baseline-and-subsequent-work) identifies the
resulting baseline. Implementation review remains the next phase.
The [evidence ledger](evidence.json) records the selection and
the digest of the preceding proposal document.

The architectural direction, first-release scope and main-chat client/UI interaction
were already selected. The [future extension contract](future-harness-extension.md)
keeps SuperAgent session coordination separate from ordinary subagent behavior; that
boundary remains intact. No future coordinator implementation is added to v1.

| Selected choice | First-release selection | Reason |
| --- | --- | --- |
| Subagent lifetime | Awaited by default; the main agent explicitly selects background work when useful, with results returned to the original chat. Stop cancels owned work; restart reconciles history without replay. | Supports the accepted main-chat model without a separate background-task product or a promise of automatic resumption. |
| Finite child budgets | Depth 2; two active immediate children per parent; four active descendants per root; four active child runs across the gateway, excluding primary runs. Keep limits configurable and reserve before launch. | Permits siblings and bounded nesting. These are adjustable engineering defaults, not measured performance limits; uncertainty continues to consume its reservation. |
| Initial Android device proof | Android Keystore P-256, ECDSA/SHA-256 and the SPKI/DER/base64url and exact-claim contract specified above. Fresh one-use challenges and live revocation govern issuance and use. | Uses the native key store and standard primitives. No legacy-device migration is required. Shared canonical byte vectors remain an explicit protocol deliverable before code review. |
| History and output retention | Preserve existing finite output/admission limits; keep regular parent/child histories until explicit deletion. Release settled execution buffers separately and retain unresolved proof. | Avoids silently deleting conversation history or evidence. Existing temporary sessions keep their separate behavior. |

The defaults are not taken from the implementation candidate. Numerical tuning can be a
later bounded configuration change; a later change to lifetime, authentication or deletion
semantics needs an explicit decision update. No separate child-management UI, automatic
restart/resume, legacy migration or new executor platform is added by these selections.

## Owner and failure walkthroughs

These are paper walkthroughs of the selected contract, not executed tests. They establish
which existing owner must resolve each scenario and what later verification must observe.

| Scenario and precondition | Authoritative owner and allowed action | Durable outcome and client result | Replay or escalation forbidden |
| --- | --- | --- | --- |
| Duplicate or late approval for a pending child effect | Existing approval admission selects one exact occurrence and rechecks current child/parent authority at dispatch. A concurrent Stop/revocation before dispatch refuses it; an effect already started stays owned. | Record the one consumed decision/effect disposition, or refusal. The original chat shows the child action and its actual result/uncertainty. | A repeated decision, changed target or model-written "yes" cannot authorize another occurrence. |
| Parent Stop races child spawn or tool execution | Parent/delegation admission fences the owned subtree. A reservation either proves non-start or remains owned through execution/process settlement. | Record cancellation/non-start or unresolved work. UI shows stopping until settlement is proven, then the corresponding terminal or uncertain result. | No late child launch, queued follow-up or pending approval can evade the fence. No success inferred from root exit. |
| Parent replies while explicitly selected background work runs | Canonical child execution retains ownership and reports completion to its parent. Normal parent reply does not invoke Stop. | Parent activity continues showing the owned task; completion is recorded and integrated once. The original-chat Stop remains available. | Parent completion cannot abandon a process, create a second task history or imply all children settled. |
| Gateway restarts with unfinished child work | Session/source and execution owners reopen canonical history and reconcile process/effect proof. Main-chat continuation, if requested, passes fresh admission with a new run identity. | Preserve historical inputs/results and unresolved evidence; show current reconciled state. Full consent remains absent after restart. | No automatic child restart, blind effect replay, revived runtime capability or old approval. |
| MCP reload/revocation overlaps a sampled tool call | Request/connection owners retain the captured definition and exact binding through settlement. New requests may use a ready replacement; revocation is checked live. | Execute the still-authorized original binding or record explicit refusal. Failed replacement preserves a usable authorized old binding. | No name-based substitution and no close while an admitted old request still owns the resource. |
| Required source write fails before a dependent effect | Canonical source writer fences the affected invocation and records/reconciles through its existing owner. Pre-effect failure admits no dependent effect. | Report the failed admission without claiming success. If an effect had already started before a later write failure, retain its uncertain disposition and ownership. | No dispatch after failed mandatory admission; no transport acknowledgement used as commit proof. |
| Effect applies but its response is lost | The effect/delivery owner reads back the same admitted identity and receipt. Unknown outcomes require reconciliation. | Return the committed result or explicit uncertainty; reconnect updates the same activity item. | A timeout or duplicate request cannot blindly perform the effect again. |
| Child capacity is exhausted or launch is uncertain | Delegation owner reserves before launch under all finite limits. Known non-start/settlement releases capacity; uncertainty retains it. | Refuse without a hidden unbounded queue, or retain the unresolved reservation. UI does not show refusal as running. Serial work may be chosen. | No fan-out beyond a configured cap, child-raised budget or premature uncertain-reservation release. |
| Client reconnects or receives an old-run event | Existing client edge projects canonical admitted identities and ordering. Rehydrate results/status without starting execution. | One activity/result per admitted run; old-run events do not overwrite current state. Disconnection means unavailable status, not completion. | No duplicate input, approval or launch, and no control authority derived from historical display text. |
| Storage fills or cleanup encounters unresolved evidence | Source/retention owner refuses new admission before effects when required recording cannot fit. Reclaim settled buffers and proven unreferenced data through their owners. | Preserve regular history under the selected deletion policy and retain unresolved effect/receipt/revocation evidence. Offer explicit cleanup. | No silent history eviction, deletion of unresolved proof or attachment deletion without complete reference checks. |
| Pre-release rollback cannot represent current authority/proof | Stop/drain the current owner or preserve uncertainty, verify an isolated copy and use a compatible state path or refuse unsupported execution. | Keep current revocations, pending-cancel fences and unknown effects readable/exportable. Unsupported remote/scheduler paths stay disabled. | No second live owner, silent reset or rollback that revives revoked authority. No historical-release qualification is invented. |

## Freeze boundary and subsequent work

All four product choices are selected and frozen as `A3-PLAN-2026-11-11`. Use the
[frozen documentation revision](plan-review.md#frozen-baseline-and-subsequent-work)
and latest staging as the implementation-review baseline. Complete and pin the initial fixed protocol schema/shared byte
vectors before reviewing the device-proof implementation boundary. This technical
deliverable is separate from the user's completed product selections.

Later work compares the separate refactor implementation against the frozen plan and
latest staging, supplies the protocol vectors before reviewing that boundary, and checks
the owner/UI scenarios through the production entry points. Platform verification focuses
on macOS; reuse matching Linux/Android evidence under their original scope. Documentation
publication, implementation acceptance and macOS verification remain separate facts.
