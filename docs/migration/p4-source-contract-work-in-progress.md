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
