# P4 source persistence work in progress

The migration remains guarded and incomplete. Dormant source-event and context-
checkpoint validation contracts, strict JSONL source transactions, and legacy
session write protection are implemented. Governed execution does not yet use
these transactions; execution ownership and mandatory runner persistence remain
to implement.

## Implemented contract

`nanobot/session/source.py` assigns runtime event/run identities to detached
payloads, discards supplied identity fields, validates unique event IDs, and merges
ordered retry suffixes without content deduplication. Distinct events with equal
content remain distinct. Retrying a different payload under a committed identity,
interleaving new events with committed ones, or retrying a non-suffix is rejected.
The caller must first compare storage state; retry merging does not implement CAS.

Digest version 1 hashes exact ordered persisted events using UTF-8 JSON, sorted
string keys, fixed separators, no Unicode normalization, and SHA-256. Non-JSON
values, nonfinite numbers, non-string keys, and malformed identities are rejected.
Internal fields participate in the digest. Errors are content-free.

Runtime `_execution_ids` align with assistant `tool_calls`; tool `_execution_id`
references that call's unique execution and must also match its provider call ID.
Repeated provider call IDs across distinct executions are legal. A checkpoint
boundary cannot include an unresolved call or an unmatched/repeated result.
The runtime link allocator and runner persistence sink remain to implement.

`nanobot/session/context_checkpoint.py` validates explicit version-1 fields,
state/provenance, exact exclusive boundary/event/digest, timezone-aware timestamp,
nonempty summary (hard schema ceiling 64,000 characters), and closed executions.
Artifact references must be empty pending artifact authorization. Auxiliary usage
is a nonempty map of nonnegative integer counters, or explicit null when unknown;
booleans and fabricated defaults are rejected. This ceiling is not a summary
request budget. Validation returns a detached candidate and never edits source,
advances consolidation, creates continuation rows, or writes a checkpoint.

## Evidence

52 network-denied tests cover identities, ordered retries, equal-content events,
canonical representation, execution provenance, split boundaries, required schema,
stale state, malformed candidates, usage, and detached validation. Scoped Ruff
passes. Independent review found digest round-trip ambiguity, missing provider
call/result matching, and incomplete usage validation and malformed root handling; these have regression tests.

## Legacy write foundation

The JSONL store now persists revision, generation, and an incarnation identity.
Loaded objects retain a detached committed base. Writes reread under the existing
cross-process lock, merge only independent append suffixes, and reject unsupported
prefix edits. New append identities remain stable across failed writes; equal text
from independent writers is not deduplicated. Legacy rows retain their positions
and content without write-on-load identity assignment.

Explicit reset/replacement requires the base revision, generation, and incarnation;
it advances generation once on successful commit and clears checkpoint/provider
state. Legacy hidden-summary insertion is separately revision checked and does not
advance generation. Deletion plus recreation assigns a new incarnation, so an old
object cannot recreate a missing session or overwrite its replacement. Fork saves
are create-only. Full-write success publishes new caller bookkeeping only after
atomic replacement and requested fsync complete. Strict governed writes will
require fsync rather than relying on the legacy optional flag.

Metadata uses a three-way merge, preserves reserved fields, and retains the
scheduled run ring by run ID. Metadata-only transactions advance revision, preserve
activity time, and materialize valid recovery sidecars before replacing the main
snapshot. Cached metadata applies the same filtering and records committed changes
so they are not replayed over later writers. Provider state and recovery state have
explicit conflict checks; metadata-only saves preserve authoritative state.
Independent transcript branches invalidate provider handles, but active recovery
evidence causes a conflict instead of being discarded to permit merging.

Legacy sidecars are now version 2, bound to main revision/generation/incarnation as
well as the previous timestamp/count. Writes also compare the current checkpoint
and provider state with the caller's base, detecting competing sidecars even when
the main revision is unchanged. This is storage conflict detection, not an execution
lease. Version-1 sidecars are not admitted by this format; copied-state migration
and uncertain-outcome reconciliation remain required before rollout.

Evidence: 268 network-denied session/history/atomic/actual-engine/detail tests pass,
including 28 new conflict tests and two real processes. Another 117 network-denied
canonical/bus/SDK/policy/trace fixtures and 25 mocked-provider API tests pass. Ruff
and diff checks pass. Independent review accepted the corrected recovery path;
review findings for sidecar preservation, provider-state merging, cache filtering,
activity timestamps, and stale recovery publication have regression tests.

An exploratory broader legacy auto-compaction/loop run produced 265 passes and 94
failures. Two failures caused by this change were corrected (test-only fixed
random-token collision and preview budget accounting for additive event identity)
and pass in the accepted suites. The remaining legacy fixture/API/guard/default-TTL
failures are not counted as passing or as production acceptance. Idle TTL remains
zero and the production guard remains in place.

## Strict source transactions

`JsonlSessionStore.source` exposes the concrete JSONL source API in
`nanobot/session/source_store.py`: explicit upgrade, strict snapshot load, append,
checkpoint commit, reset, unrelated metadata update, and consolidation-cursor CAS.
There is no fallback to ordinary `Session.save` or an unqualified third-party store.

Source state compares storage-root identity, session key, incarnation, revision,
generation, the raw checkpoint ID, and a canonical fingerprint of the complete raw
checkpoint value. A snapshot from a copied/relocated store must be reloaded before
writing. All source mutation CAS checks run under the existing process-safe file
lock and reject even append-only revision drift. An idempotent append retry is
examined only after CAS succeeds against a freshly loaded state.

Explicit upgrade strictly parses the original bytes, preserves event content and
unrelated metadata, assigns missing event IDs, and does no legacy exec-name or
argument normalization. Structured tool call IDs are paired with exactly one
unresolved earlier call, and assigned execution IDs. Sequential repeated provider
IDs remain distinct. Orphan/duplicate/ambiguous results and overlapping legacy
provider IDs reject the upgrade without a write. An unmatched trailing call remains
unresolved and prevents a checkpoint from covering it. Unknown historical run
attribution is not inferred. Provider handles are discarded by upgrade; existing
recovery sidecars or active legacy recovery metadata require separate reconciliation.

The source parser accepts only the supported schema and explicit state fields,
unique IDs, valid execution links, and JSON-safe retained evidence. It performs no
repair, sidecar overlay, or write-on-read migration. A malformed or unsupported
checkpoint is retained in raw metadata and CAS fingerprinting but ignored for
projection, with a content-free diagnostic. A loaded checkpoint's source revision
must precede the containing file revision; later appends or metadata writes do not
make its unchanged covered prefix invalid.

Checkpoint commit validates against the exact current snapshot and atomically
writes source plus checkpoint, advancing revision only. It inserts no continuation
row, edits no source event, and leaves archive cursor/activity time unchanged.
Reset advances generation, clears checkpoint/recovery state, and invalidates old
snapshots. The separate consolidation operation advances both `last_archived` and
`last_consolidated` compatibility aliases together; these aliases describe the
memory archive cursor, independently of `context_checkpoint.boundary`. Upgrade
preserves legacy headers containing either alias.

Source writes use the existing private-file atomic writer with file fsync before
replace and directory fsync afterward. Platforms/filesystems that do not support
directory fsync retain the existing documented weaker guarantee; deployment
qualification remains required. Results are detached and returned only after the
write succeeds. Failure before replacement retains the old complete snapshot;
failure after replacement can be resolved by reloading the new complete snapshot.
These primitives do not publish a speculative runtime projection or cache entry.

Compatibility readers recognize source format and use the strict source reader,
so they do not rewrite exec records or overlay old sidecars. Ordinary full writes
and runtime-sidecar writes reject source sessions. Metadata updates use the strict
source transaction and the same ownership filter in disk and cache. A pre-upgrade
cached legacy object is invalidated when a metadata update observes source format.

Ordinary readers now require an intact, unambiguous first metadata row to identify
the format. An empty/truncated/malformed/null header cannot silently become an empty
or repaired legacy session. Explicit legacy repair remains a separate read-only
recovery operation; automatic legacy-tail recovery requires an intact legacy header.
Tests formerly expecting automatic recovery of unidentified headers now verify
failure without altering evidence and, where possible, explicit salvage instead.

Evidence: 304 network-denied session/atomic/history/actual-engine/detail tests,
117 network-denied canonical/bus/SDK/policy/trace tests, and 25 mocked-provider API
tests pass (446 total). The 36 strict-source tests include actual two-process
checkpoint contention, store binding, delete/recreate identity, before/after replace
failures, fsync ordering, malformed headers/counters, legacy exact pairing, checkpoint
persistence through later revisions, cursor independence, and compatibility/cache
paths. Ruff/diff checks pass. Independent review accepted the corrected archive
aliases, unidentified-header handling, cache filtering, and store binding.

These tests also found and corrected an earlier legacy append-identity corner case:
two appended positions referencing the same Python dictionary now receive distinct
identities instead of collapsing into one event.

## Next execution package

Integrate cross-process execution ownership and fencing, mandatory durable user/call/
result events, source-based projection provenance, and provider-state invalidation.
The existing storage CAS does not fence tool dispatch or reconcile uncertain external
outcomes. Consolidation's source transaction is implemented, but memory receipt and
runner wiring still require acceptance. Offloading, context governance, idle
compaction, provider coordination, cron/auth/startup work, and overall migration
validation remain open. No runtime guard has been removed.

## P4B execution ownership primitive

The dormant source store now has a per-session cross-process execution lock and a
nonserializable, task/PID-bound `ExecutionOwner`. Acquisition persists an exact
lease before returning authority. The lock spans dispatch validation and effects;
there is no lease expiry or takeover of a live owner. Registered child tasks may
share authority; unregistered children cannot dispatch, mutate, or self-register.
Root cleanup stops new dispatch and drains registered tasks even when cancelled.
Lock files remain in place across deletion so competing processes share one inode.

Source append/checkpoint/consolidation/reset/delete transactions take execution
authority before the short storage lock. Owned writes retain exact revision CAS;
metadata-only writes preserve reserved ownership fields. Destructive owned writes
require root quiescence and invalidate the capability. Ownerless writes reject
live locks, abandoned leases, and durable recovery markers. Abandoned ownership or
unresolved calls prevent dispatch until explicit reconciliation; release preserves
recovery evidence. Persistence failures before/after replacement never authorize
an acquired capability or permit retry of uncertain effects.

Validation: 327 network-denied session/atomic/history/actual-engine/detail tests,
71 network-denied canonical/bus/SDK/policy/trace tests, and 25 mocked-provider API
tests pass (423 total for this checkpoint). The 23 ownership tests include actual
process contention and abrupt process exit, stale lease tokens/incarnation/generation,
child registration, cancellation while draining, reset/delete fences, persistent
lock inode, and acquisition/release persistence failures. Ruff and diff checks pass.
Independent review passed the child-task and durable-recovery-marker corrections.

This primitive is not yet acquired by the canonical runner. Mandatory durable
user/intent/per-result/final events and dispatch wiring are the next package.
Production runtime remains guarded; these results do not establish migration readiness.

## P4C mandatory result recording boundary

`AgentRunSpec.tool_result_callback` now receives one detached `KnownToolResult`
per known result before post-tool/error observers. It retains batch ordinal,
provider call ID/name, full raw application evidence, classified result/event, and
whether dispatch occurred. Preparation/repeated-lookup rejection, policy rejection,
and skipped calls are recorded without claiming dispatch. Tool exceptions retain
structured type/message evidence; cancellation never fabricates a result.

A trusted typed `TurnContext.execution_owner` is checked immediately before tool
execution, outside tool exception handling. An owner without a recording callback
fails before provider/tool work. Callback-enabled batches are sequential. Recording
failures propagate once, preventing later tools and model continuation; observers
cannot erase already recorded evidence. The async callback permits a source sink
to perform a synchronous atomic transaction before any suspension. Existing
no-owner/no-callback execution retains its concurrency behavior.

Evidence: 414 network-denied session/migration/runner tests passed, with two existing
plugin-loader tests deselected because the unchanged production runtime guard
rejects their loader. A further real-source cancellation regression passed in the
17-test focused result-recording suite: the first result remains durable when the
second call is cancelled, and unresolved intent prevents later dispatch. Ruff/diff
checks pass. Independent review accepted the bounded recording seam.

Canonical acquisition and the production source sink are not wired yet. Application
execution identities must be bound to durable assistant intent by that sink; batch
ordinals alone are not durable identities. User/final event persistence, source
projection, nested delegation, and recovery admission remain integration gates.

## P4D canonical source execution

Explicitly upgraded default-store sessions (`source_schema_version=1`) now execute
through the real canonical stages using `SourceTurnWriter`. Source ownership is
acquired for the resolved approval key before approval consumption and retained
through stage/delivery cleanup. Recovery-required sessions cannot request a model
or dispatch tools; the explicit `/new` path quiesces and resets under ownership.
Legacy sessions are not automatically converted and retain their existing path.

The writer assigns fresh event, stream, and execution identities independently of
caller run IDs and provider call IDs. A shared transaction coordinator supports
separate parent/delegate streams without overwriting parent batch bindings. User
input is committed before generation; known assistant responses/call intent are
committed before post-response observers; each known tool result is committed
before its observers. Synthetic terminal responses are separate source events.
Recording failures poison the whole invocation so a failed child cannot be turned
into a recoverable parent result and followed by more effects. No automatic write
or effect retries are introduced. Text remains complete in source; existing privacy
behavior is retained for inline image data and reasoning blocks, with explicit
unavailable image placeholders rather than new unapproved binary retention.

Source projection reads retained root events independent of the archival cursor,
removes explicitly marked old runtime injections, preserves media breadcrumbs,
excludes child streams and command rows, and carries source identity provenance.
Fresh mode adds durable events while projecting only the current invocation.
Root traces bind to a trusted stream ID so child details and prior invocations
with a reused run ID cannot leak into the current trace. Source finalization emits
metadata deltas, not stale whole-metadata snapshots or legacy full-session saves.

This is the first working source execution slice, not final context governance:
requests retain full source and fail at the hard budget boundary. Legacy reduction,
offloading, idle summary publication, recovery sidecars, and provider-state resume
are excluded from this path. Source checkpoints are not yet used for projection.
Main and nested planner/scoped worker paths run through actual factories/runner/tools
in network- and process-blocked fixtures. `/new`, `/model`, `/help`, `/history`, and
`/status` have source-compatible paths; `/compact` and other commands give an
explicit candidate limitation until their source adapters are completed. New/reset
archival receipts, scheduled/injection/continuation integration, command parity,
checkpoint/offload/provider coordination, copied-state conversion, and activation
qualification remain mandatory subsequent work. The production guard remains.

Evidence: 496 network-denied session/migration/runner/atomic/history tests passed
with two guarded plugin-loader cases deselected; 25 mocked-provider API tests
passed. Subsequent targeted additions pass in 18 real source-engine tests and 11
writer tests, covering ownership wait before approval consumption, concurrent title
updates, writer/dispatch owner matching, source write failures, nested worker writes, policy terminals, large raw
text, reused run IDs, source history, and observer/cancellation failure boundaries.
Ruff and diff checks pass. Independent review accepted the source execution
corrections. Admission test hosts now provide the newly required SessionManager
or explicit legacy-store capability instead of uninitialized mock dependencies.

Broader qualification is still red: an additional runner/governance/autocompact/
provider-state/model-command selection produced **58 failures, 317 passes, 6 setup
errors, and 4 deselections**. Running the identical selection against an exported
pre-change `0436c0d6` tree produced the identical failing node set and counts.
Unresolved nodes by file: `test_runner.py` 18, `test_runner_context_trim.py` 6,
`test_runner_fallback.py` 3, `test_runner_governance.py` 1,
`test_runner_injections.py` 18, `test_runner_progress_deltas.py` 3,
`test_model_command.py` 15. These include legacy ABI expectations and guarded
startup tests; they are not waived and must be reconciled before migration success.

### Source projection foundation (implementation resumed)

`session/source_projection.py` now provides detached event-bearing nodes, scoped
root checkpoint projection, explicit stream-local projection, newest-user pinning,
and eligible closed-prefix selection. Only validated checkpoints marked
`projection_scope="root"`, `projection_version=1` are applied; unscoped or
incompatible checkpoints fall back to source. Child/fresh stream selection never
inherits a root checkpoint. Checkpoint text is historical evidence in an assistant
message, never system policy. Public source sanitization was extracted unchanged
from `SourceTurnWriter` so projection does not import agent startup.

Boundary selection uses exact event identities, global execution closure for roots,
and local closure for isolated streams. Invalid source linkage fails explicitly;
unacknowledged visible events prevent advancement. No storage, summary generation,
or provider calls occur in these helpers. The current runner still uses full source:
governor request-ticket acknowledgment, bounded auxiliary generation, pressure
compaction, checkpoint publication, and authorized model-facing instruction binding
remain unwired. This slice does not enable migration activation or offloading.

Evidence: 296 session/source-engine tests passed with network denied; Ruff and
diff checks passed. Fourteen projection tests cover scope, pinning, incompatible
checkpoint fallback, repeated-content identity, global/local tool closure,
corruption rejection, rollback projection, and checkpoint reuse after real store
reload. Read-only review accepted the foundation (12 tests before the two final
storage/corruption additions). Isolated writer-test collection still encounters the
existing missing `ExecToolConfig` startup import; combined session/source-engine
collection passes. This broader import/ABI qualification remains unresolved.

### Bounded source-summary operation

`Consolidator.summarize_source_chunk` supplies one storage-free auxiliary operation
for governor-selected evidence. It encodes evidence as data, excludes system-role
inputs, caps summary output at 2048 tokens, checks input plus output reserve plus
1024 tokens of safety headroom, and bounds execution to at most 60 seconds.
It rejects empty, errored, truncated, oversized, or tool-bearing responses without
raw fallback. Cancellation propagates. Typed results retain usage or unknown usage,
captured provider/model, status, and call count. No archive or receipt method runs.

`LLMProvider.chat_once` retains observation but bypasses framework retry/recovery.
Fallback wrappers retain the captured primary; key pools make one paced credential
attempt without rotating. Summary reasoning is explicitly disabled to prevent
Anthropic from enlarging the reserved output budget. Transport-internal retries
remain governed by the transport and bounded by the enclosing timeout.

Evidence: 309 combined session/source-engine/summary tests passed; 13 summary tests
also passed independent review after bounds corrections. Ruff/diff checks passed.
A broader pool/fallback/observer selection produced 81 passes and 24 failures:
the legacy `test_fallback_provider.py` fixture omits the now-required provider_name
constructor argument. These tests still require ABI reconciliation, not waiver.

This auxiliary operation remains unwired: the governor still must select exact
closed chunks, limit an attempt to four calls, aggregate all auxiliary usage,
validate the finalized rebuilt main request, and publish under strict source CAS.
Provider-specific finalized-request budget validation also remains an integration
gate. The runtime activation guard remains unchanged.

### Provider regression reconciliation and startup import qualification

The 24 old fallback tests now exercise `LegacyFallbackProvider` through its
preset/factory boundary with explicitly named synthetic providers. Checks retain
quota/rate-only traversal, primary-only model overrides, generation propagation,
and no fallback after streaming output. Cooldown assertions use the pinned staging
code's actual five-minute behavior (the old comments incorrectly said twelve
hours). The provider/pool/observer selection passes 105 tests.

`SourceTurnWriter.record_tool_result_event` now returns the exact committed result
event ID; the existing callback remains return-free for execution compatibility.
This supplies explicit ordinal-bound provenance for subsequent governor wiring.

Agent and configuration package exports now load lazily, preserving exported
names and canonical identities. Delegation imports tool configuration classes from
their owning modules. Schema aliases resolve lazily; Config and ToolsConfig supply
canonical tool types during Pydantic first-use rebuild without changing field
types or suppressing validation failures. These changes remove observed startup
cycles without requiring test import warmups or disabling the runtime guard.

Evidence: 31 fresh-interpreter cases pass, covering independent first-use model
validation, JSON validation, schema generation, alias lookup, export identity, and
tool/path imports avoiding agent-loop/schema initialization. Independent review
also ran these 31 cases. Standalone legacy-policy and source-writer files pass
20 and 11 tests respectively. The combined source/provider/import selection passes
465 tests; Ruff/diff checks pass. Automatic first-use `TypeAdapter(Config)` after a
delegation-first import remains unsupported; no production caller was found. It is
not covered by these startup acceptance claims.

Broader configuration qualification remains red: `tests/config` has 113 passes and
9 failures (legacy token object/save expectations, empty legacy context paths,
timezone detection, and guard-blocked onboarding). Those require reconciliation
before overall migration success. Source governor integration remains pending.

### Configuration qualification follow-up

The nine configuration failures above are now reconciled. A real normalization
defect was corrected: scalar `maxTokens` plus legacy `maxInputTokens` must retain
staging's precedence over `contextWindowTokens`; leaving the camel-case alias in
the input caused Pydantic to choose the wrong capacity. The conflicting-values
test now verifies 100000 input / 2222 output through save and reload. Empty legacy
`contextPaths` are omitted on save, as in staging; nonempty settings remain intact.

Legacy token tests now assert the accepted upstream scalar-output and separate
context-window representation. Missing timezone remains UTC/manual as required
by the foundation contract; explicit auto mode still detects the backend zone.
Onboarding tests exercise temporary config/workspace paths with scoped test-only
activation overrides and the current channel-plugin discovery boundary. Production
guard code and activation sites remain unchanged.

`tests/config` passes all 122 tests without deselection. Scoped Ruff/diff checks
pass; read-only review found no regression. This closes the previously listed
configuration-test failures, not the outstanding source governor, startup rollout,
or overall migration acceptance gates.

### Explicit request ledger

`session/source_request.py` now supplies detached source/runtime/checkpoint
transcripts and invocation-local request tickets. Initial eligible IDs must be
represented source nodes. Issuing a ticket grants no eligibility; acknowledgment
consumes the exact local ticket and includes only its explicitly bound source IDs,
never output appended after issuance. Discarded, reused, forged, cross-ledger, and
pre-projection-replacement tickets fail. Request-only extensions may contain
runtime instructions but cannot omit or edit the active projection. Projection
replacement invalidates pending tickets while retaining acknowledged evidence.

Evidence: eight ledger tests plus fourteen projection tests pass; independent
review ran all 22. The combined session/source-engine/summary selection passes
317 tests. Ruff/diff checks pass. This is provenance machinery, not activated
compaction: runner/governor call sites still must supply authoritative bindings
and acknowledge only after validated success and mandatory recording.

Next integration must preserve IDs at `_build_source_turn` and early user
persistence, bind the current model-facing user instruction separately from
persisted evidence, and carry a typed source transcript in `AgentRunSpec`.
Assistant response recording must return its event ID through the request result;
tool replay must bind the ID from `record_tool_result_event` by batch ordinal.
Length/continuation/malformed/finalization notes are explicit runtime nodes.
Each request stack frame must retain its own ticket: recursive malformed retries
discard the failed ticket and acknowledge only their validated replacement, never
a mutable shared last-ticket slot. Only then can the governor select eligible
closed prefixes, run bounded summaries, and publish checkpoints after rebuilt
request validation and strict source CAS. Production activation remains guarded.

### Initial source transcript integration

Canonical source admission now retains history nodes before writing the current
input, and stores the current user's actual writer-returned event ID on the trusted
run context. `ContextBuilder.build_source_transcript` constructs the system runtime
node, explicit source history, and authorized current-user node directly. A
synthetic unpersisted continuation is a runtime node. `AgentRunSpec` carries this
typed transcript and explicit initially eligible history IDs; the source runner
renders it instead of reconstructing identity from the legacy message builder.
Delegation binds its separately persisted assignment event to its own source node.
Fresh history includes only the current invocation's input. Checkpoint reuse is
still disabled in this initial integration.

Evidence: 318 session/source-engine/summary tests pass. Independent review ran all
19 real source-engine tests and accepted initial binding, including distinct IDs
for identical inputs and fresh scope. Ruff/diff checks pass. The broader migration
and runner-tool selection has 181 passes and the same two activation-guard failures
in entry-point plugin fixtures; neither fixture reached the changed source path.
Response/tool/runtime append bindings, ticket issuance/acknowledgment, and pressure
compaction remain required. This step changes initial construction only and does
not enable migration runtime activation.

### Recorded assistant response identities

Runner requests now return a `RecordedResponse` containing the provider response
and its exact durable event ID (or None when no response event was recorded).
Main and recursive malformed-request paths propagate the record belonging to the
response actually returned. No-tools and finalization paths preserve that identity;
no-tools mandatory recording now precedes the governor observer. Existing writes
remain single writes, and provider response objects carry no invented private IDs.

An actual-engine malformed → malformed → no-tools regression verifies four source
events exactly once and the final answer's event ID through both recursive returns.
Independent review ran 31 engine/writer tests and accepted the identity flow.
The combined session/migration/runner/summary selection passes 481 tests with the
two previously documented activation-guard plugin fixtures deselected. Ruff/diff
checks pass. Replay-node binding, ticket issuance/acknowledgment, checkpoint reuse,
and pressure compaction remain unfinished; production activation remains guarded.

### Tool replay bindings in runner request state

The runner now creates an invocation-local source request ledger from the admitted
typed transcript and explicit eligible history IDs. Tool intent replay uses the
recorded assistant event ID. The mandatory result callback captures each result's
durable ID by batch ordinal; subsequent tool-message replay binds that exact ID.
Repeated provider call IDs cannot collapse two results, child writers retain their
own callbacks, and missing bindings fail closed. Execution callbacks continue to
return no value. These bindings do not acknowledge requests by themselves.

An actual-engine repeated-call-ID test verifies distinct source result identities
in the second request ledger. Independent review ran 32 engine/writer tests and
accepted callback ordering and identity handling. The combined session/migration/
runner/summary selection passes 482 tests, with the same two guarded plugin
fixtures deselected. Ruff/diff checks pass. Other assistant/runtime replay append
paths and request-ticket issuance/acknowledgment are still pending, so this ledger
is not yet used to reduce context or publish checkpoints. Activation remains guarded.

### Assistant and runtime replay bindings

Source ledger replay now binds ordinary assistant replies, length-recovery
segments, synthetic error/empty terminal events, and max-iteration finalization
to their exact recorded IDs. Length and caller-requested continuation instructions
are detached runtime nodes. Continuation replay binds the preceding assistant
without writing it again. Typed source-mode real user injections reject missing
durable bindings; their persistence adapter remains required before enabling that
path. Legacy terminal replacement/deduplication helpers remain unchanged for
non-source runs, following a review correction and dedicated regressions.

Actual-engine length and continuation tests verify the ledger renders the second
request exactly while runtime instructions stay out of source. Independent review
accepted the source changes and verified both legacy terminal regressions. The
combined session/migration/runner/summary selection passes 486 tests with the same
two guarded plugin fixtures deselected; Ruff/diff checks pass. Request-only retry
views, ticket acknowledgment, checkpoint publication, and compaction remain
unfinished. The production activation guard is unchanged.

### Validated source request acknowledgment

Source requests now verify their rendered explicit view before transport, issue a
local ticket after budget validation, and acknowledge it only after mandatory
response recording and a recognized successful finish. Malformed calls invalidate
their attempt even when normalization changes the finish status; no-tools requests
cannot acknowledge tool-bearing responses. Transport cancellation/errors and
mandatory recording failures discard their tickets. Newly returned assistant/tool
evidence remains pending until actually represented in a later successful request.

Malformed, finalization, and iteration-budget retry instructions are explicit
request-only runtime nodes. Recursive requests carry their own views and tickets;
the stored last view is only a projection input, never a shared mutable last-ticket
slot. Unbound hook edits fail before transport. Source preparation still preserves
the whole request: compaction integration must issue the ticket from the governor's
resulting projection once that projection can change.

Actual-engine tests cover pending tool evidence, failed/malformed attempts,
unbound hook mutation, cancellation cleanup, and continuation/retry paths.
The focused engine/writer suite passes 40 tests; independent review ran 35
engine/request-ledger cases before the final cancellation/hook additions. The
combined selection passes 490 tests with the same two guarded plugin fixtures
deselected. Ruff/diff checks pass. Checkpoint reuse/publication, pressure compaction,
offloading, and remaining migration acceptance work are still unfinished.

### Scoped checkpoint reuse across turns

Root source turns now select the validated checkpoint projection after persisting
current input. The current input is excluded from history by its durable event ID
and added once by the context builder; the previous user is no longer replayed as
the pinned instruction. Compatible summaries remain historical assistant data.
Incompatible checkpoint scopes fall back to canonical source, and fresh turns
bypass root checkpoints. No auxiliary request or source rewrite occurs on reuse.

Actual-engine restart tests exercise compatible/incompatible scope, current-user
uniqueness, fresh isolation, and unchanged source prefixes. Independent review
accepted the slice and ran all 31 engine tests. The combined session/migration/
runner/summary selection passes 492 tests with the same two guarded plugin
fixtures deselected. Ruff and diff checks pass. Automatic pressure compaction,
checkpoint publication from the governor, offloading, and whole-migration
acceptance remain pending; production activation remains guarded.

### Captured pressure-compaction candidates

`session/source_compaction.py` now captures a detached source snapshot and typed
active projection for the existing governor. Root boundaries come from compatible
stored checkpoints and count all canonical events; local boundaries belong only
to the explicit fresh/child stream. Advancing boundaries use validated eligibility
and execution closure. New summary evidence comes from disjoint canonical slices,
never the expanded model request or an already-covered pinned instruction.

Rebuilding preserves active runtime nodes and surviving authorized source nodes,
replaces the historical checkpoint, and retains the newest canonical user by its
exact durable ID. Capture rejects foreign, duplicate, omitted, reordered, or
unexpected covered source nodes and checkpoint drift. Hidden-only advancement
cannot publish a summary with no new visible evidence. Independent review found
and corrected a pin-selection risk in reordered projections; regressions now prove
canonical ordering and role-independent pin selection. All 13 helper tests pass.

This helper performs no provider calls, token reduction, writes, or ledger
publication. Governor orchestration remains the next required integration:

- Capture source writer, immutable runtime, admission-derived root/local scope,
  local checkpoint state, and the runner's actual raw message list reference.
- Validate the explicit request view before transport without issuing a ticket;
  separate request-only runtime suffixes from the active ledger projection.
- Measure the complete request including tools and retry notes; select bounded
  disjoint chunks, fold the prior summary, and accumulate every auxiliary call's
  usage. At most four summary calls are allowed per attempt.
- Rebuild and validate the complete candidate request before durable root CAS.
  Use the originally captured full SourceState, including metadata revision and
  raw parent checkpoint identity; do not refresh/rebase after generation.
- For fresh/child candidates, revalidate ownership and exact local stream evidence
  without writing root checkpoint metadata. Unrelated sibling appends need not
  invalidate an unchanged local stream.
- Publish the ledger projection and replace the runner's raw list by slice without
  an intervening await. Keep retry notes request-only, clear usage/provider-state
  assumptions, and issue tickets from the resulting view. Preserve the production
  guard until all integrated acceptance requirements pass.

Validation after the review correction: 505 combined session/migration/runner/
summary tests pass, with the same two guarded plugin fixtures deselected. Scoped
Ruff and diff checks pass. This is candidate-selection acceptance only, not
pressure-compaction or whole-migration acceptance.

### Governor resulting-view handoff

The source request ledger now validates a request view without issuing a ticket or
advancing eligibility and returns detached request-only runtime extensions. The
source governor accepts that typed view, checks its exact rendered payload and
active projection, and publishes the resulting view alongside prepared messages.
Both runner transport paths issue tickets from that resulting view. Recursive
malformed retries and the no-tools fallback therefore inherit a replacement
projection rather than the pre-governor view.

Synthetic governor-publication tests run through the real engine and replace the
view independently at the initial, malformed retry, and no-tools request. They
verify checkpoint identities on every ticket, exact provider payload/view parity,
request-only retry notes, no acknowledgment of malformed output, and unchanged
canonical response recording. They isolate the handoff contract and do not claim
summary generation or durable publication is integrated.

Independent review accepted this slice and ran 43 engine/ledger tests. The broader
session/migration/runner/summary selection passes 509 tests with the same two
guarded plugin fixtures deselected. Both legacy context-governance tests pass;
scoped Ruff and diff checks pass. Pressure-triggered auxiliary generation, raw-list
publication, strict checkpoint CAS orchestration, and all remaining migration
acceptance work remain required. Production activation remains guarded.

### Owner-fenced governor checkpoint publication

`ModelRequestState` now holds invocation-owned source compaction state: the trusted
writer, captured runtime, actual runner message-list reference, admission-derived
root/local scope, and temporary local checkpoint state. Full-history roots use
root scope; fresh runs and child writers use their own stream IDs. An actual-engine
test checks root/fresh scope and raw-list identity at request preparation.

The existing governor captures canonical candidates, projection epoch, request-only
notes, and tool definitions before auxiliary work. Its publication method rejects
ledger/raw/epoch drift or outstanding tickets, validates the bounded summary and
complete rebuilt request (including tools/retry notes), and commits root checkpoints
with the originally captured SourceState and execution owner. Metadata-only drift
fails CAS. Local publication instead checks ownership, generation/incarnation and
exact scoped events, allows unrelated sibling/root appends, and never writes root
checkpoint metadata. Publication has no await between commit and ledger/raw-list
replacement; it clears usage assumptions and provider continuation state.

Real-store tests verify durable root reload, writer append after checkpoint commit,
retained source prefixes, local checkpoint reuse, preservation of an existing root
checkpoint, pending/metadata/source/epoch/raw drift, oversized complete requests,
and write failure. Independent review accepted the slice and ran 13 publication
tests; the final strengthened local-root preservation and same-stream drift cases
pass in the 14-test focused suite. The broader selection passed 525 tests with two
guarded plugin fixtures deselected before that final additional regression. Scoped
Ruff and diff checks pass.

Automatic pressure detection and bounded auxiliary generation still do not invoke
this publication method. Their usage/call accounting, cancellation handling and
actual-engine pressure/retry acceptance are the next integration. Production
activation, offloading, provider-state enablement and whole-migration acceptance
remain guarded and incomplete.

### Automatic source pressure compaction

The existing governor now invokes source compaction at 90% of available input
capacity, targets 60%, and retains the hard request guard. It sizes auxiliary
output against the rebuilt policy/tools/pin/runtime/suffix payload, capped at 2,048
tokens. Exact summary-message construction is shared with the bounded
Consolidator operation so auxiliary preflight measures the actual payload.

Before spending auxiliary calls, the governor checks every indivisible closed
group in the intended prefix for feasibility. It then chooses the largest fitting
closed disjoint chunk, folds the preceding summary, checks each folded payload,
and makes at most four sequential calls per attempt. A fitting partial projection
may publish after that cap; 60% is a soft target, while the hard budget must pass.
No eligible evidence leaves the prior view subject to the hard guard. Invalid
summaries, cancellation, infeasible chunks, oversized rebuilt requests, and stale
CAS do not publish a candidate or initiate a main transport from this method.

Each attempt retains content-free scope, boundary, before/after estimates, calls,
known usage, explicit unknown-usage status, latency and outcome. Debug telemetry
contains those metrics only. Root checkpoint usage is null if any auxiliary call
has unknown usage; known partial counts remain in attempt metrics. Summary calls
continue through the provider observer. These metrics are not yet integrated into
all outer run-result/UI cost displays, which remain a reporting qualification gap.

Tests exercise disjoint multi-call summaries, root/local publication, the four-call
cap with an oversized remainder, mixed known/unknown usage, repeated compaction in
one invocation, invalid output, cancellation, metadata drift, oversized fixed
requests, and a fitting early group followed by an impossible group with zero
auxiliary calls. An actual-engine test verifies pressure-triggered generation,
root publication, unchanged canonical prefixes, exact current-user retention, and
restart reuse with no extra summary call. Independent review found and corrected
the advance feasibility gap and accepted the slice after 44 pressure/engine tests.
The subsequent unknown-usage/repeated-compaction regressions pass in the 10-test
focused pressure suite. The combined selection passes 537 tests with the same two
guarded plugin fixtures deselected; scoped Ruff and diff checks pass.

Production activation remains guarded. Actual-engine local/retry pressure stress,
transport-finalized budgeting, factual summary quality/performance qualification,
outer usage reporting, offloading, provider-state coordination, idle/command and
source-injection parity, and remaining whole-migration acceptance are unfinished.

### Actual-engine fresh and delegated pressure qualification

Fresh and delegated planner runs now have actual-engine pressure tests spanning
two length-limited responses, local summarization, and successful continued
generation. They snapshot the provider requests and compare each local payload
with its governed view. Every local request excludes seeded root history, and the
post-compaction request retains both exact length-recovery instructions once.
Canonical segments remain once each in the correct stream; delegated intent/result
identities match and executions close. Root checkpoint metadata and existing source
prefixes remain unchanged. These tests qualify existing production code; they add
no new runtime behavior.

Independent review identified initially weak runtime/history assertions; the
strengthened request snapshots, exact instructions, source counts and execution
binding assertions address them. All 38 source-engine tests pass. The broader
selection passed 539 tests with the same two guarded plugin fixtures deselected
before the assertion strengthening; scoped Ruff and diff checks pass.

The next offloading packages and their unresolved integration gates are recorded
in [the artifact offloading contract](p4-artifact-offloading-contract.md). Artifact
storage/retrieval/projection is not implemented by this qualification slice.
Production activation and full migration acceptance remain incomplete.
The narrow review recheck accepted the strengthened assertions. Token pressure
uses a deterministic synthetic estimator; these cases do not establish local
crash/restart behavior or production summary quality.

### Artifact descriptor and source-binding foundation

The first offloading slice adds strict typed artifact descriptors and opaque
reference parsing, with exact canonical text/execution binding. It neither stores
blobs nor grants retrieval access. [The offloading contract](p4-artifact-offloading-contract.md)
records the fields, reviewed source-intent correction, 60 passing artifact cases
and 360 passing session tests. Private storage/publication and all authorization/
projection integrations remain next; source inline retention and the production
migration guard are unchanged.

### Private immutable artifact byte store

The dormant artifact store now provides no-follow private filesystem operations,
namespace binding, immutable verified blobs, bounded byte ranges, and store-lock
transactions. Reviewed corrections address concurrent macOS lock creation,
adopted-directory durability and interrupted hard-link installation. Actual
multi-process tests cover concurrent writes and crash recovery. All 394 session
tests pass, including 34 blob-store tests independently rerun in review; Ruff/diff
checks pass. [The offloading contract](p4-artifact-offloading-contract.md) records
platform/capacity limits and remaining integration gates. This store does not
publish descriptors, authorize reads, expose references or change the production
guard. Owned source metadata publication is the next storage integration.

### Owned artifact descriptor publication

The dormant source publisher now validates the exact configured private store,
blob durability, dispatch-capable execution ownership, strict source CAS before
idempotency, and immutable manifest/source bindings. Publication increments only
the metadata revision; canonical events, activity time, archive cursors and
checkpoint remain unchanged. Generic metadata updates cannot replace the manifest;
reset clears it and deletion removes source authority while retaining blob bytes.

Focused read-only review found no implementation defect and requested additional
failure coverage. Seventeen publication tests now cover reload/retry/stale CAS,
identity/origin conflicts, missing/corrupt/closed/wrong-store blobs, pre-publication
disk failure, invalid manifest, missing/closing/recovery-required owners,
namespace replacement before and after verification, stale legacy save, reset
and deletion. Manifest parser tests independently cover duplicate bindings and
malformed schemas. These tests do not establish authenticated producer provenance
or read authorization. Runtime store configuration, trusted producer receipts,
retrieval/grants and projection/checkpoint integration remain required. The
production migration guard remains enabled.

Final validation: all 417 session tests passed with network denied; scoped Ruff
and diff checks passed. The reviewer independently reran all 17 publication tests
and accepted this dormant package. This is package acceptance, not migration
completion or runtime enablement.

### Actual filesystem producer evidence

Contextual ordinary-text reads now persist execution-bound producer receipts via
registry, execution callback and source writer. Receipts bind the exact returned
text, policy locator and root identity; provider projections omit them. The
artifact attestation validator rejects missing/mismatching provenance and origin
relabeling. Legacy conversion removes untrusted receipt metadata. Independent
review accepted this correction. See the offloading contract for qualified-read
limits and unsupported adapters. Retrieval authorization, grants, runtime artifact
publication and projection integration remain unfinished; the migration guard is
unchanged.

### Scoped artifact retrieval

An explicitly supplied runner factory now provides per-invocation artifact reads
with current policy, source membership, producer evidence, stream and private blob
integrity checks. Historical roots survive restart; fresh and child streams cannot
read root history or sibling evidence implicitly. Reads use bounded UTF-8 ranges
and a complete 16384-byte JSON output cap. Trusted retrieval markers remain private
canonical metadata for future recursion exclusion. Source restoration preserves
admitted session policy, including explicit tool disables. Combined validation:
603 passed, two existing guarded plugin fixtures deselected; scoped lint/diff pass.
Runtime config/default factory, grants, rematerialization, offloading, summary
reference dependencies and whole-migration acceptance remain pending.
Read-only review accepted the correction bundle and independently passed 32
focused artifact/retrieval tests. This accepts the explicit-factory retrieval
package only; the production migration guard remains enabled.

### Automatic main-request offloading with an admitted factory

Large attested scalar results now become typed, verified reference projections
before pressure preparation while canonical text remains inline. The actual
engine test covers two large reads sharing a provider call ID, distinct artifact
identities, bounded previews, retrieval without recursive publication, unchanged
canonical prefix and restart reuse without any descriptor publication. Focused
checks cover stable revisions, runtime retry notes, no-tools inline fallback and
hard budget failure, storage corruption, policy revocation, stale CAS, owner loss
and canonical summary evidence. Reference-aware summaries remain blocked by an
explicit trusted-retrieval dependency guard; configuration defaults, general-tool
adapters, grants and whole-migration activation remain pending.
Final validation: 643 combined session, source-engine, artifact, filesystem,
pressure/publication and runner tool tests passed with network denied; two
existing guarded plugin fixtures remain deselected, not waived. Scoped Ruff and
diff checks passed. Read-only review independently passed 98 focused tests and
accepted this bounded package. A final canonical-order regression also passes.
The production migration guard remains enabled.

### Dormant foreground idle activity admission

The source store now accepts an optional current `ExecutionOwner` when recording
one already admitted foreground-user activity update. The owner-aware path
reuses the turn's existing execution transaction, advances the typed activity
revision and timestamp durably, and rejects stale or foreign owners. Ownerless
callers remain available for the dormant store fixture path; a governed turn
never opens a second ownerless transaction while it owns the session.

`AgentLoop` has an explicitly injected, disabled-by-default seam for a complete
idle integration. For `DIRECT`, `GATEWAY`, `SDK`, and `API` user turns, source
activity is durable before `foreground_admitted(session_key)` is awaited. A
callback failure or cancellation prevents model/tool stages. `CRON`,
`HEARTBEAT`, `SYSTEM_COMPAT`, subagent activity, and partial or disabled
integrations produce no activity write or callback. A foreground follow-up
accepted while a source turn owns the session is admitted by an
owner-registered worker before first-turn cleanup. The source runner clears its
pending queue, so cleanup republishes that accepted message for a second
canonical turn; the internal message-identity marker suppresses a duplicate
activity write and callback when that turn starts. Durable activity commit and
callback success are tracked separately: a callback failure or cancellation
still retains the marker, while the current turn stops before model work. The
marker is consumed on dispatch and cleared on cancellation, enqueue failure,
cleanup failure, and shutdown.

This package adds no scanner, summarizer, provider call, checkpoint publisher,
configuration default, guard change, or legacy `AutoCompact` behavior. Focused
network-denied validation covers owner reuse/fencing, source filtering,
durability-before-callback ordering, callback failure/cancellation, disabled
wiring, and serialized follow-up admission. The production migration guard
and idle scheduler remain dormant.

### Dormant dependency-aware idle checkpoint substrate (I2)

The new `nanobot.session.idle_checkpoint` module adds an explicit integration
seam for idle checkpoint capture and one-shot publication. Capture returns the
existing typed `IdleCaptureToken`, a detached `SourceSnapshot`, and caller-
supplied runtime/policy fingerprints. When a configured artifact store and
current policy binding are supplied, it also seals a nonserializable
`IdleDependencyAccess` to the exact source store, session/incarnation/generation,
capture identity, and the descriptors already published from root-history
events. The capability exposes no reader, grant set, tool, arbitrary path, or
blob read method.

Publication computes and validates the candidate/proof outside storage locks,
then acquires the configured artifact transaction before the nonblocking
ownerless execution/session transaction. It revalidates the idle token,
activity and pending/recovery state, exact schema/proposal/proof, source and
checkpoint identity, current policy receipt, artifact namespace, and durable
blob immediately before one atomic schema-v2 write. Dependency-free candidates
use schema v1 (or a dependency-free schema-v2 proposal); dependency-bearing
candidates require an exact root-history proof and schema v2. A stale,
foreign, revoked, corrupted, active-owner, recovery, or pending state fails
closed without retry, rebase, owner fabrication, or source write. The shared
receipt/blob verifier is also used by `ArtifactReader` after its existing
owner, invocation, stream, and private-root checks, so this extraction does
not widen ordinary artifact reads.

The source-store additions are delegates only. No scanner, factory, provider,
summarizer, configuration default, migration-guard change, or runtime activation
was added. Focused network-denied tests cover detached/nonserializable capture,
exact dependency references, forged/stale proof and capture rejection, policy
and blob failure, active-owner/recovery nonblocking behavior, activity-byte
preservation, and no-write failure paths. Broader session/artifact validation
and independent review remain required before any future wiring or enablement.
