# P3 version-2 cron action-store contract

`nanobot.cron.action_store.CronActionStore` is an isolated private POSIX primitive. It is not imported by the cron service and does not parse, normalize, execute, or deliver cron jobs. Package B will adapt legacy service actions, Package C will bind a replay candidate to the canonical cron store, Package D will establish occurrence authority, and Package E will add destination receipts and uncertain-send reconciliation. The migration guard remains closed until those packages and their integrated acceptance evidence exist.

## Public bounded API

Construction requires an absolute, non-root, private directory and a UTF-8 store namespace of at most 512 bytes:

```python
store = CronActionStore(root, namespace="cron:workspace-a")
receipt = store.append(action_id="opaque-request-id", request={...})
state = store.replay()
settled = store.checkpoint(expected_generation=state.generation, candidate={...})
```

`append` accepts an opaque JSON object only. It returns an `ActionReceipt` containing namespace, action ID, canonical request digest, sequence, and the journal generation that established it. Reusing an action ID is accepted only when the canonical request digest is identical. `replay` returns detached data: the prior opaque candidate (or `None`), ordered pending opaque actions, all retained receipts, and an `ActionStoreGeneration` CAS token. The module never applies `add`, `update`, or `del` semantics. A later adapter derives its candidate from the returned values.

`checkpoint` accepts only the exact generation returned by `replay`. The token includes the journal generation, snapshot generation, snapshot watermark, and latest visible action sequence, so an append after replay makes a checkpoint stale. It writes a detached opaque candidate after all actions visible to that generation, then compacts the journal. A stale generation fails. Retrying the same token and canonical candidate after a durability error is safe: a published snapshot records its prior generation and is settled or re-proved. Retrying that token with different candidate content fails as a conflict.

The hard limits are 1 MiB per snapshot or journal record, 256 KiB per opaque request/candidate JSON encoding, 8 MiB for the entire current journal, 4,096 retained action receipts, 512 UTF-8 bytes for namespace and action ID, and action IDs matching `[A-Za-z0-9][A-Za-z0-9._:-]{0,511}`. Reaching a receipt or size limit fails closed and requires a later, reviewed export/retention policy; this package must not discard idempotency evidence itself.

## Version-2 durable wire protocol

`journal.jsonl` begins with exactly one strict JSON header:

```json
{"schema_version":2,"kind":"journal_header","store_namespace":"…","journal_generation":1,"base_sequence":0}
```

Each subsequent strict JSON line is an opaque action with that namespace and journal generation, a contiguous sequence of `base_sequence + 1`, an action ID, a SHA-256 canonical-request digest, and the request object. The canonical digest is SHA-256 over `nanobot-cron-action-v2\0` followed by canonical sorted, finite JSON. No field has a default. Every schema version, generation, sequence, base, and watermark field is an exact Python/JSON integer: booleans and `2.0` are corruption even when equality would otherwise succeed. Unknown fields, duplicate object keys, non-finite values, malformed UTF-8, missing final newlines, sequence gaps, and duplicate action IDs are also corruption.

`snapshot.json` is strict version-2 JSON with a namespace, target journal generation, monotonic snapshot generation, base/watermark, prior generation, opaque candidate, and a complete contiguous receipt list through its watermark. A snapshot receipt preserves the original action ID, digest, sequence, and journal generation. Receipt generations must be positive, never later than the prior journal generation, nondecreasing by sequence, older than the current journal for a previously compacted prefix, and exactly the prior journal generation for the new suffix. This makes action retries exact even after compaction.

On checkpoint the store first writes, file-fsyncs, atomically replaces, and directory-fsyncs the snapshot. Its fsynced staging descriptor remains open across the descriptor-relative rename; the destination entry must match that exact inode before and after directory fsync. Only then does it write, file-fsync, replace, and directory-fsync a new header-only journal whose generation and base equal the snapshot's target generation and watermark, using the same staged-descriptor proof. A restart which sees the new snapshot but old journal accepts it only when the journal is the exact receipt prefix covered by that snapshot and contains no later action. It re-fsyncs the snapshot and directory before finishing journal cleanup. Any other mixed state is ambiguous and fails closed.

The action header and every append are independently file- and directory-fsynced before acknowledgement. An append retry with the same action ID and digest re-fsyncs its source before returning its old receipt. These retry rules make failures after writes, renames, or directory fsyncs safe without silently treating visible data as durable.

## Filesystem and migration boundaries

This package is qualified only on POSIX systems offering `flock`, held descriptor-relative operations, no-follow opens, and directory fsync. It imports on Windows but construction fails closed. It traverses the configured absolute physical path with held no-follow directory descriptors and binds root/lock inodes. Before and after every file or directory fsync that may establish durability or return an acknowledgement, it re-traverses that configured path and requires its root inode to equal the held root. Journal and snapshot descriptors are also checked against their current directory entries before and after fsync, so a same-owner/mode replacement cannot be acknowledged. Root is mode `0700`; lock, journal, snapshot, and temporary files are owned regular single-link files mode `0600`. Existing unsafe paths, symlinks, hard links, mode/owner changes, and root/lock replacement are corruption and are never repaired in place.

The root is dedicated to this primitive. The legacy `action.jsonl` filename is detected and raises `ActionStoreConversionRequiredError`; no conversion is attempted. `jobs.json` is deliberately outside this primitive's ownership. It is neither read nor changed, so an unreadable legacy jobs snapshot cannot prevent an offline action create from being durably queued or be accidentally overwritten. Packages B and C must supply the explicit legacy conversion, job-snapshot validation, and service wiring before any production use.
