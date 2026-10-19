# P3 cron migration checkpoint

This is a historical implementation log. Dated gap/open/unresolved statements
below record the state of their individual slices and are superseded for
current P3 status by the final closure conclusion.

The data adapter restores ordered/deduplicated additional destinations, historical run/occurrence identifiers, and canonical run status. Store/action decoding and explicit service serialization preserve those fields and execution mode. Action decoding does not mutate its input.

Service preparation now uses `prepare_staging_job`: a legacy record without an explicit binding gets `cron:{id}`, staging cli/direct route defaults, and `legacy_fresh` mode. Complete existing bindings retain their mode and origin. Partial/conflicting bindings and unsupported modes remain unchanged with a separate admission diagnostic. Migration does not rewrite enabled/next-fire/timestamps/history/delivery fields. Explicit user updates and enable/disable operations retain their ordinary scheduling changes.

Binding completeness is distinct from execution admission. Until canonical fresh-history execution and delivery adapters are installed, legacy fresh jobs and retained legacy delivery policy are fenced. Diagnostics do not replace historical last_error. Recompute, next-wake, timer selection, manual/forced run, direct execution, gateway callback, and bound runner enforce admission. WebUI run returns409 with the diagnostic instead of reporting a task accepted when it cannot execute. Complete modern jobs and system events retain execution.

Validation: new admission regressions first reproduced destructive load behavior. The final focused admission/field selection passes25 tests with network denied; Ruff passes for cron, gateway runtime, WebUI HTTP handler and the new tests. Coverage includes repeated load, one-shot preservation, no callback/run records, mixed blocked/system due jobs, action merge/update/enable, each retained delivery category, unsupported modes, runner rejection before side effects, and actual HTTP handler409, and extracted real gateway callbacks named dream/heartbeat rejecting before any agent/model/delivery access. Review found and closed the callback ordering defect by moving admission before all name-based branches.

The wider selection of service/persistence/session-delivery/delivery/admission/field tests yielded89 passed and8 failed before the final12 admission cases were added. The unchanged upstream baseline failures are:

- test_unbound_agent_jobs_are_disabled_on_add
- test_unbound_agent_jobs_are_disabled_on_load
- test_add_job_migrates_legacy_delivery_context
- test_load_store_migrates_legacy_delivery_context
- test_load_store_disables_malformed_legacy_payload
- test_list_bound_agent_jobs_includes_migrated_legacy_delivery_payloads
- test_channel_meta_and_session_key_survive_store_reload
- test_update_job_migrates_legacy_delivery_target

These assert destructive disable/route clearing or origin-session adoption that conflicts with preservation requirements. They remain recorded failures, not a green full-suite result. Their migration-aware replacements and final baseline reconciliation remain required.

Historical pre-composition gap list (superseded by the closure conclusion below): canonical fresh-session restore under the turn lock, no prior checkpoint/followup reuse, canonical outcome and explicit-message ledger, suppressed ordinary streaming/final publication, one computation per stable occurrence, destination fanout/dedup, durable receipts and uncertain-send reconciliation, concurrent edits/deletes without resurrection, callback lifecycle propagation and complete cron-tool fields. Fresh input still had to preserve run detail history. The checkpoint prohibited a parallel legacy runner or activation before contract integration. Full migration remained guarded, and no live cron state was read or changed.

2026-09-29 dormant text-result settlement slice: `nanobot.cron.canonical_result.complete_text_cron_result` accepts an exact claimed occurrence and a completed canonical `TurnResult` with its stable run ID, non-empty bounded text, and an empty authoritative explicit-message ledger. It durably completes the occurrence and prepares the E2 outbox through the injected E3 authority, preserving the claim's frozen destination policy and exact ordering. It never starts an agent, arms the scheduler, links visible chat, or sends to a channel. Unsupported statuses, mismatched IDs, explicit sends and oversized text fail before completion; a failed outbox preparation after completion recovers by projecting and preparing that occurrence, never by recomputing. Fake-sender tests exercise both destinations only after separately claiming them. At this historical slice, ordinary publication/message tools still needed suppression before an actual turn, media and channel-specific explicit sends needed separate treatment, and approval policy was unchosen; the final closure below supersedes that policy status. Both old fork cron tests remain fenced; the production migration guard is unchanged.

2026-09-29 dormant silent fresh-turn slice: `AgentLoop.execute_silent_cron_turn` admits a provenance-bound fresh-history request through the real canonical runner without a bus completion, channel progress/stream/lifecycle events, direct `MessageTool` availability, or a remembered visible route in the cron detail session. It refuses explicit grants, callbacks, media, tool overrides and pending human approvals before the model. Existing detail remains stored but out of the model-facing history; a new run writes its own detail and typed run record. At this historical slice, the primitive did **not** accept or verify an occurrence claim, fence another process, choose a production approval policy, settle a failed/non-text result, link a visible-chat result, or call the E2 result adapter. Later P3 slices and the final closure below supersede those implementation and policy gaps; cron admission and startup remain unwired.

2026-09-29 dormant one-shot claimed execution slice: `execute_claimed_fresh_text` validates a complete frozen legacy-fresh definition and destination policy, reserves the exact live process-local occurrence via `begin_execution`, runs the silent canonical fresh turn with the stable occurrence run ID, then durably completes a supported text result and prepares its outbox. Actual model and fake-agent tests cover no ordinary bus output, retained-but-excluded old detail, pending additional destinations (including an additional-only route), concurrent exact claim retry, copied/tampered/stale claims, unsupported result, cancellation, model failure, and outbox recovery without model replay. The one-shot start is consumed even if the model or settlement fails; a failure before completion leaves a claimed record requiring explicit operator reconciliation, while a post-completion outbox failure is recovered from durable completion without another model run. At this historical slice, the cross-process stopped-owner/turn fence, real transport, result/media semantics, visible-chat linking, callback lifecycle, and scheduled risky-action approval decision were open; later slices and the final closure supersede those P3 gaps. The scheduler and admission diagnostic still reject legacy fresh execution, and the global migration guard remains active.

2026-09-29 dormant visible-chat awareness slice: `link_completed_text_result` accepts only a durable, validated text-only completion ID, revalidates its frozen legacy-fresh definition and origin route, and writes a bounded run reference plus excerpts to that origin chat's metadata. It requires the exact enriched text-settlement and delivery schemas, so contradictory error, explicit-send, media, provenance, or unknown evidence fails before session mutation. It never appends the scheduled instruction/result to visible transcript or sends to a channel. An explicit `visible_sessions` authority optionally invokes this after computation completion and outbox preparation; the standalone linker retries after a session-write failure from the durable occurrence without recomputing. Stale concurrent session views use a bounded reload-and-retry and retain the independently written transcript. The next ordinary chat turn receives a bounded model-facing projection, and legacy scheduled wrapper pairs are excluded from that projection without rewriting old history. The full run remains in `cron:{job_id}` detail. Completion awareness explicitly does **not** attest delivery; additional destinations do not acquire chat references, and explicit/media/non-text results cannot use this text-only linker. Focused tests exercise claimed/failed/malformed rejection, real synthetic-provider computation followed by a visible-chat turn, idempotent retry, bounded ring, old-wrapper exclusion, concurrent chat writes, and save-failure recovery. The wider cron/session/runner selection passed **303** tests; scoped Ruff and diff checks passed. A fresh read-only review found the malformed-completion gap, verified its exact-schema correction, and recommended PASS. Type checking was unavailable because `basedpyright` is not installed in this environment. At this historical slice, the process/turn fence, channel sender, risky-action approval decision, and callbacks were unresolved; the final P3 closure below supersedes those P3 gaps. Legacy admission and the global migration guard remain unchanged. No live cron store or channel was touched.

2026-09-29 dormant per-occurrence execution/reconciliation fence slice: every
canonical claimed execution now holds one persistent nonblocking OS lock for
its exact occurrence from before `begin_execution` through the silent model
turn, claimed-to-completed mutation, projection/outbox preparation, and
optional visible-chat linking. Explicit reconciliation acquires that identical
lock before rereading and CASing the exact predecessor, so live execution is
reported busy and never advanced. The task/PID/store/claim-bound capability is
nonserializable, requires the same active started fence for completion, and is
invalidated on release and fork; a child closes inherited descriptors without
unlocking or retaining the parent's authority. Constructor and operation paths
validate the permanent `.execution.<64hex>.lock` files as exact private
`0600`, one-link regular files with descriptor/path/root binding checks and
durable first creation. Failure/cancellation consumes a begun local start and
leaves the durable claim reconciliation-required; completed post-settlement
failures recover without a model replay. Deterministic tests cover contention
during the model, settlement, outbox, and visible-link windows; different
occurrences; capability misuse; unsafe filesystem entries and fsync failure;
and fork-child descriptor behavior. The required cron/session selection passes
**207 tests**. Scoped Ruff and `git diff --check` pass. Type checking was not
run because `basedpyright` is unavailable. Detached effects after process
death remain operator evidence/compensation responsibility. Scheduler wiring,
transport, approvals, callbacks, admission diagnostics, the legacy admission
fence, and the global migration guard are unchanged; no production activation
path or live-state access was added.

2026-09-29 dormant per-destination delivery send/reconciliation fence slice:
every E3 dispatch now holds one permanent nonblocking OS lock for its exact
durable destination from before `mark_sending` through detached request
construction, the coalesced sender invocation, cancellation/timeout join,
durable settlement or settlement failure, and release. Explicit operator
reconciliation obtains the identical lock internally before its first durable
predecessor read and retains it through publication/reproof; a live sender or
reconciler reports `DeliverySendBusyError` without mutation or successor
authority. The capability is bound to the exact store, claim object and frozen
fields, PID, asyncio task, held root binding, and lock descriptor/inode;
copied, serialized, reconstructed, stale, foreign, inactive, wrong-task, and
fork-child use fails closed. Permanent `.send.<64hex-destination-id>.lock`
files are exact private `0600`, one-link regular files; every new or residue
acquisition reproves binding, fsyncs file and root with reproof between
boundaries, and only then attempts nonblocking flock. Lock order is destination
fence then ordinary outbox lock, and no ordinary lock crosses the provider
await. Runtime/service reconciliation forwarding preserves the exact
predecessor, action, request digest, revision, and evidence parameters without
accepting a caller fence. The integrated dormant Telegram-only receipt path
uses an exact resolver and direct provider call, never the bus or channel
manager's ordinary retry path. Provider acceptance does not attest user read;
bounded content-free receipts remain bound to the exact request and target,
while partial or ambiguous effects become uncertain and unsupported explicit
dormant transports become permanently not sent. Telegram locally serializes
bundles per target, but this is not cross-process or cross-host ordering. This
remains dormant: no scheduler, gateway, bus, CLI/tool, production composition,
other-channel qualification, admission, or migration-guard wiring changed. Abrupt process
death releases the kernel lock but leaves claimed/sending/uncertain state for
explicit evidence; a detached provider effect can still exist and cannot be
inferred, leased, timed out, or retried automatically. Independent integrated
selections passed **135** delivery/fence/receipt tests, **91**
occurrence/canonical/visible-link/execution-owner regressions, and **178**
Telegram regressions; scoped Ruff, compileall, and `git diff --check` passed.
`basedpyright` was unavailable and was not installed. The overlapping
selections are not summed.

2026-09-29 offline cron-delivery reconciliation slice: a separately registered
`cron-delivery` Typer group now provides explicit existing-only inspection,
private content-addressed evidence import, immutable plan/review, and doubly
confirmed apply through E2's internally fenced exact reconciliation API. All
commands require absolute non-overlapping outbox and artifact roots plus the
exact namespace; no production path is inferred. The stage permits only
claimed-to-confirmed-not-sent, sending/uncertain-to-confirmed-delivered with a
complete strict Telegram receipt, and conservative abandon. Artifacts and CLI
output exclude message content and raw target; provider acknowledgement means
Telegram acceptance rather than user read. Apply publishes a separate
content-free application receipt, and an exact replay after publication loss
uses E2's retained idempotent resolution without a second transition. This is
not activation: it constructs no channel/sender/provider/bus/scheduler,
dispatches and claims nothing, and leaves the global migration guard and
runtime admission unchanged.

Current P3 closure conclusion (2026-09-29): the dormant cron adapter contract,
including end-to-end claimed composition, is accepted. Unattended execution
has no implicit or stored human approval and persists no pending request.
Composition is bounded to canonical text, an authoritative empty explicit-send
ledger, and exact Telegram receipt transport; richer outcomes and other
transports fail closed or remain reconciliation-required. Durable content-free
callback observation is exact-once with no automatic retry: callback failure
never replays computation and delivery may continue. The separate cron
acceptance commands passed **241** and **25** tests respectively; the counts
are not summed. Eight legacy tests still assert destructive old behavior and
remain recorded failures. See [the P3 closure checkpoint](p3-closure-checkpoint.md)
and [the composition contract](p3-cron-composition-contract.md). This is not
production activation: scheduler wiring and legacy-fresh admission remain
closed, the global migration guard remains active, the POSIX/`flock` callback
backend fails closed elsewhere, and no live cron state or channel was touched.
