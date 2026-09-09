# P4 source persistence work in progress

The migration remains guarded and incomplete. This checkpoint implements dormant
source-event and context-checkpoint validation contracts only. No production
caller uses these helpers; session saves still require revision/generation/CAS
restoration before governed execution can be enabled.

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

## Next storage package

Restore revision/generation behavior under the existing file lock with loaded-base
tracking, reject unsupported prefix edits, preserve scheduled-ring merge semantics,
and prevent stale reset/delete resurrection. Add explicit locked source upgrade,
strict append/reset/checkpoint transactions, protected metadata, fail-closed corrupt
reads, and atomic fsync/cache publication tests including two real processes.
Do not port staging content-prefix deduplication as source identity. Source-schema
sessions must reject ordinary save and legacy recovery-sidecar overlays.

Execution leases, durable runner events, projection provenance, provider-handle
invalidation, offloading, idle compaction, and remaining migration packages are
still open. These tests do not establish durability or production readiness.
