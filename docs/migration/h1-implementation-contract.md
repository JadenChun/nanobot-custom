# H1: offline implementation and functional acceptance

Status: **IN_PROGRESS, NOT_ACCEPTED** (2026-10-15). ROOT_DIRECT.
The human accepted the five recommendations on priority, reply destinations,
temporary recovery/privacy, bounded control inference and offline-first activation.
This is one functional stage; its internal work packages are not separate stages.

## Accepted decisions

- Channels are authenticated transports for one assistant. Direct SDK/API session
  and history contracts remain unchanged. No additional task-working agent.
- A steer supplies the full original message exactly once at a safe pre-model
  boundary. Independent tasks use a distinct FIFO next-turn lane. Stop is fenced
  to the observed run generation. Human work precedes automatic continuation;
  automation receives a bounded, non-starving opportunity at slice boundaries.
- The active task's full result stays on its original route. A cross-channel steer
  receives its own receipt and completion notice, not the full result. Queued
  tasks reply to their submitting route, including thread/reply context.
- Regular admission is durable. Temporary recovery exists only for that
  temporary session's lifetime, never enters regular context/long-term memory,
  and is retired only after accepted work has a visible disposition. Uncertain
  external effects are not silently discarded or replayed.
- A separate stateless, tool-free control request proposes a route. The owner
  coordinator, not the model, commits it. Errors/timeouts/capacity failures safely
  propose next-turn; ambiguity asks for clarification. No accepted message is
  dropped; storage capacity is checked before an acceptance receipt.
- Implementation and validation are offline. The migration guard, credentials,
  services, live providers/channels, cron/Dream/Heartbeat/callbacks/idle and P8
  remain outside this authorization. D9/P7 require their separate activation gate.

## Engineering limits and ownership

Control inference starts with explicitly configurable engineering limits:
2 simultaneous calls, a 3-second routing wait, 30 calls per 60-second window,
16 KiB user input, 128 output tokens and a 4 KiB response. These are offline
defaults, not production latency, billing or workload certification. A timed-out
provider retains its slot until it actually terminates, including when it resists
cancellation. No retry or fallback provider is automatically constructed.
Select a separate provider instance and actual model explicitly before live use.

Keep durable inbox transactions under the existing admitted session namespace
and metadata lock. Source binding must be reconciled by ingress/event identity;
do not generalize P4 receipt writers, create a competing session authority, or
discard existing WebUI follow-up journals. The temporary journal needs a disjoint
metadata-only lifetime and must not become a regular conversation.

The dormant ledger now uses content-free closed-lifetime records for temporary
retirement. This prevents stale wrappers/transport retries from recreating the
same lifetime, without acquiring a lifecycle mutation beneath the files lock.
It is logical content erasure, not a forensic secure-erase or backup-retention
guarantee. Generic session reset/delete/conversion require a settled regular
inbox under their existing mutation transaction; temporary lifetime records
belong to the dedicated retirement operation and are hidden from chat listings.

Clarifications retain the original ingress and result destination, are bounded
to eight replies per ingress by default, and acquire a new FIFO eligibility
position. Their scopes intersect with the original scope, never expand it.
Each reply phase has a stable intent identity even across repeated orphan
recovery. Only definite-unsent transport outcomes return to pending delivery;
uncertain effects fence subsequent work and uncertain sends are not replayed.
Admission reserves transition space; records/dedupe remain finitely capped,
and saturation is an explicit rejection before acceptance, not silent eviction.

These storage mechanisms are not runner/channel integration. Canonical source
binding, actual control effects, one owner consumer, bounded automation fairness,
channel authorization/destinations and old WebUI journal reconciliation still
need whole-stage evidence. H1 remains NOT_ACCEPTED until the criteria below pass.

The runner integration connects the real runner's safe boundaries to the
ledger through an explicitly admitted `BusySteering` adapter. Generic callbacks
remain separate and cannot supply source identities. The actual root writer
allocates content-bound source rows and reserved per-ingress markers; the adapter
verifies them before recording the inbox manifest and exposing model input.
Owning recovery can reconcile complete or partial source rows by those markers,
even if the manifest write was interrupted, without promoting uncertain work to
completed or replaying it. The originating dispatch run is retained for proof.
Main messages and all clarification replies retain their complete text/media.
Whole-ingress drain limits leave excess work durable, and exhausted iteration
budgets do not claim input for a nonexistent next model round. A dispatch marker
precedes rendering/source work; write failure or cancellation is not completion
and orphan recovery fences it as uncertain rather than blindly replaying it.
Draining leaves the ingress claimed: terminal settlement belongs to the owning
turn after its real persistence/effect boundary. This package does not activate
the bus consumer or activate the whole H1 stage. The separate Telegram package
below supplies an explicit adapter registration seam, which remains dormant unless
trusted composition registers it.

Canonical steering admission requires the actual source transcript before any
claim or provider call; supplying only initial messages with a writer is not
sufficient. Legacy steering continues to support its existing transcript path.
The optional steering field follows every pre-existing `AgentRunSpec` field so
existing positional construction retains its argument meanings.

The explicit offline owning-consumer seam now connects the ledger to actual
`AgentLoop.execute_turn` admission, source ownership and delivery. It does not
register itself from configuration or inbound metadata. A registered session
rejects raw legacy bus dispatch until its authenticated channel adapter exists;
it never enters the old 20-item injection queue or overflow/republication path.
One consumer task claims human next-turn work in eligibility order and passes
the claim's intersected tool-name scope as an explicit execution ceiling. Full
original/clarification source rows use the same owned binding path as steers.
Claims stay retained through actual turn persistence, response delivery and the
terminal run-record save; non-completion or ambiguous failure is not success.
Unclaimed obsolete steers become next-turn. A delivery failure is recorded
separately and never grants permission to repeat completed task effects.

Trusted cron/local-trigger submissions use a separate bounded lane on that same
consumer. A waiting automation receives an opportunity after at most three human
turns; it cannot become a mid-tool injection. Its future observes the real terminal
execution/delivery outcome, and shutdown terminates queued waiters. Registration,
recovery and teardown remain under the actual AgentLoop/session authority.
Completed slices yield to routing and occurrence waiters before younger work.
Ordinary turn settlement preserves independently active notice sends; only
explicit restart recovery marks abandoned sends uncertain. The inbox path uses
its durable dispatch marker instead of the incompatible legacy runtime checkpoint
writer, whose exact revision is advanced by new durable input admissions.
The separate offline Telegram package now supplies authenticated channel admission
and exact-route notice/result delivery through the explicit edge below. Temporary
execution and WebUI journal adoption remain separate integration work. The
Telegram edge stays dormant unless trusted composition registers it, and this is
not whole-H1 acceptance.

The offline owner now applies revisioned stop/status/stop-and-replace controls to
an exact live slice handle. Stop intent is durable before a one-shot cooperative
cancellation request, with no await between checking ownership and requesting it.
The bounded reply wait observes that slice's settlement event; it cannot cancel
a younger slice. Facts report running, stopping or blocked ownership and actual
queue/recovery/terminal state. A default three-second reply wait (explicitly bounded
to zero through thirty seconds) is an offline engineering limit, not a promise
that an in-flight effect can be forcibly terminated. A cancelled waiter leaves
settlement owned by the consumer. No control-model answer or status text becomes
task history. Subordinate cleanup stays beneath the actual session admission lock,
and stop checks prevent fresh model/tool dispatch after suppressed cancellation.

Replacement retains the complete original input, clarifications, media, intersected
scope and reply route. It becomes FIFO work only after the old slice quiesces with
a safe disposition. An explicitly stopped undispatched claim can be cancelled
without inventing a RunRecord: managed privacy deliberately omits unadmitted
cancellation records. Dispatched cancellation, interrupted automation, terminal
write/cleanup faults or abandoned started controls retain uncertainty and fence
execution for owning recovery. A computed result remains completed if stop races
delivery; delivery uncertainty does not permit repeating its effects. Durable
control completion records control bookkeeping separately from the observed task
status. The opt-in Telegram package now binds authenticated ingress to this seam,
sends factual notices independently, and delivers immutable task results through
exact thread/reply receipts. Configuration and startup do not activate the adapter;
trusted composition must register it explicitly. Temporary execution and whole-H1
acceptance remain outstanding.

On 2026-10-15 the offline Telegram package adds a channel-local edge bound to one
registered owner inbox, consumer, control router, finite tool scope, Telegram
account, and explicit sender/chat bindings. It admits complete assembled input
before sending its acceptance receipt, correlates clarification replies only to
their persisted notice receipt, and retains stable album/member identities. Its
independent notice pump sends factual stage notices; immutable full results use
the initial input's exact thread/reply route and receipt. Cross-route steers
receive their own factual notices, never the task result. Authorization is
rechecked before each provider call, and uncertain sends never replay task work.
This package is tested offline and remains dormant in normal configuration and
startup. It is Telegram-only evidence and does not establish whole-H1 acceptance.
Local output media is bound to its downloaded byte digest. Guarded HTTP(S) output
media remains an exact URL reference whose digest identifies the URL; it does not
identify the bytes served there, so a remote server may change the media at the
same URL between delivery attempts. The adapter does not fetch or cache remote
media while retaining or retrying a result.

## Exit criteria for the whole H1 stage

- AC1: Authenticate owner/origin, deduplicate stable transport IDs, durably save
  full text/media and reply route before truthful acceptance; fail before receipt
  on capacity/storage error. Bursts above 20 never create another owner runner.
- AC2: While a task provider/tool is blocked, independently classify and reply;
  bound rate, concurrency, wait and input/output; no tools, task provider state,
  task-history row or substantive answer from the control model.
- AC3: Revision and run-generation checks reject late/stale decisions. An
  unclaimed steer becomes next-turn if its run finishes. Ambiguity suspends
  without permanently blocking unrelated tasks; clarification retains identity.
- AC4: Ordered lanes and one consumer; no overflow task dispatch/republication
  races. Human/automation fairness and automation terminal futures are tested.
- AC5: Canonical-source and legacy steering include the complete message once
  before model dispatch, retain original authority and never inject mid-tool.
- AC6: Stop/status/stop-and-replace use actual events and cooperative ownership;
  no stop of a newer run, fake completion or blind replay of uncertain effects.
- AC7: Crash/fault tests cover acceptance, routing, outbox, claim, source append
  and effect/delivery boundaries, including existing follow-up reconciliation.
- AC8: Telegram first and each enabled IM adapter retain authorization, stable
  IDs and exact originating destinations. Delivery failure does not replay work.
- AC9: Temporary mode and restart recovery stay separate from regular context
  and memory. Close/erase handles pending and uncertain work visibly.
- AC10: Held-out multilingual/quoted/forwarded/ambiguous routing cases, receipt
  latency, auxiliary usage and task-completion observations are published with
  their limitations. Scripted tests do not establish real-model accuracy.
- AC11: Existing P1-P6 preserved contracts, current strict/lint and relevant
  integration regressions pass without suppressions, exclusions or guard changes.
- AC12: Exact scoped diff/evidence and a final H1 verdict. P6 local acceptance is
  retained as immutable history, not relabeled after new H1 changes.

## Validation discipline

Add deterministic red tests before implementation. Validate in fresh private
source snapshots using the existing frozen environment, network/home denial and
owned-process runner. Preserve every original source/test byte outside the named
H1 boundary. Record actual outcomes and remaining criteria across working rounds;
do not call a prototype, an imported module or a green unit selection H1 complete.

Authoritative prior local P6 acceptance: SHA256
`33f6ba53a9cc268a818e2e045879f8b06abde7f1d33a5015ffcbff99f0fb696f`.
Historical in-repository P6 failure checkpoints remain historical. The remaining
activation checklist and [busy-inbox design](h1-busy-inbox-design.md) remain in force.

On 2026-10-15 the temporary QA directory was found missing. The isolated candidate
and frozen dependency records survived. The recorded descriptor-relative reader
and owned-process runner were recovered verbatim from the chat history; historical
P6 acceptance and H1 ledger proof files were not recreated. Fresh before/after
validation uses a new current-candidate capture, including the repository's existing
type stubs, with network/home denial and unchanged dependency installation records.
It establishes the bounded followup's current behavior and source preservation
from that capture, not preservation back to the missing historical inventories.
The prior broad run's chat record reports 7,488 passed, two existing platform skips,
three shell-denial failures and two local-socket setup errors. This is not a
whole-suite pass. P6 remains immutable accepted history; whole-H1 preservation
and final acceptance evidence remain outstanding.
