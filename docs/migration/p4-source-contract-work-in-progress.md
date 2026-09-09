# P4 source persistence work in progress

The migration remains guarded and incomplete. Dormant source-event and context-
checkpoint validation contracts are implemented, alongside legacy session write
protection. No production caller uses the strict source/checkpoint helpers yet;
governed storage transactions and execution ownership remain to implement.

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

## Next storage package

Add explicit locked source upgrade and strict append/reset/checkpoint transactions,
including revision/generation/incarnation/parent-checkpoint CAS. Restore the separate
generation-checked consolidation-offset transaction. Source-schema sessions must
reject legacy recovery overlays and all mutation paths that could rewrite source
without strict validation; ordinary full saves already reject that schema. Add
strict malformed-source handling, two-process stale checkpoint tests, and fsync/cache
publication failure tests for governed snapshots. Do not use content deduplication
as source identity or treat a legacy hidden summary marker as a source boundary.

Execution leases, durable runner events, projection provenance, provider-state
coordination, offloading, idle compaction, and remaining migration packages are
still open. The accepted legacy write tests do not establish governed persistence
or production readiness.
