# P4 read-only terminal semantic proof

The offline finalizer and any uncertain-unlink replay before normal runtime
resumes call
`prove_terminal_complete_under_locks(authority, operation, checked_session,
workspace_id, receipt_evidence) -> TerminalCompleteProof` from
`nanobot.agent.memory_recovery_transaction_terminal`. The caller already holds
an active workspace memory authority/operation and an issued checked recovery
session, in that lock order. Do not acquire a second workspace lock. The last
argument is exactly an authority-issued live `MemoryTargetSnapshot` of
`.consolidation.pending.json` **or** an entered
`ValidatedQuarantineCopy` from `open_validated_quarantine_copy(...,
require_phase="copy-ready" | "committed", require_pending_absent=True)`.
The copy may be copy-ready before committed publication. A structural marker,
boolean, pathname, or caller-manufactured snapshot is not receipt evidence.

The proof requires the exact seven typed, predecessor-linked transaction
records through `COMPLETE`, replays the receipt-tail conversion's six records
through its `COMPLETE`, verifies frozen conversion blobs/output targets,
canonical marker and transaction target transitions, archive backup and source
absence, and the exact checked session source bytes/cursor. It compares the
issued live receipt's current named bytes and all nine stat fields throughout.
Copy mode rechecks the Q1 lease before/after semantic reads, requires named
pending absence, compares Q1's **original pending** stat (not the private copy
inode) to transaction `PREPARED`, and matches Q1 transaction, COMPLETE,
receipt, workspace ID, workspace path, and sessions root to the actual
manifests and checked session candidate. The backup-only verifier uses an
issued `history.jsonl` anchor, checks frozen one-link backup metadata/bytes
and source absence on both sides of a parent-directory fsync. It never links,
unlinks, appends, or publishes evidence. The existing archive application and
terminal reconciliation entrypoints retain their live-receipt behavior.

The returned operation-bound, immutable `TerminalCompleteProof` exposes
`transaction_identity`, `complete_digest`, `receipt_digest`, `receipt_size`,
`receipt_stat` (nine values in device, inode, mode, uid, gid, links, size,
mtime_ns, ctime_ns order), `checked_session_source_digest`,
`conversion_complete_digest`, `semantic_seal_claims` (detached mapping), and
`semantic_seal_digest`. Call `.require_current()` under the same held locks
immediately before consuming any field; it re-runs the semantic proof and
requires the exact same canonical claims. Closed leases, drift, or changed
records fail with the fixed content-free terminal error. It is not a durable
token to cache across calls. A fresh proof (with a fresh checked session and
copy lease) is possible on replay only while the old memory/session targets
remain unchanged. This current-target proof is distinct from the implemented
historical startup-admission protocol in
`memory_recovery_startup_admission.py`, which is used after normal runtime has
resumed. See the [final P4 closure checkpoint](p4-closure-checkpoint.md) for
current status. P4 acceptance does not authorize runtime activation, guard
removal, cron scheduling, deployment, live traffic/state migration, or
Windows/release qualification.

## Canonical historical semantic seal v1

`semantic_seal_digest = SHA256(UTF-8 JSON + "\n")`, where JSON is sorted by
key, compact (`separators=(",", ":")`), ASCII escaped, and contains exactly
the following claims. The `receipt_original_stat` array uses the order above.
Null is allowed only for an explicitly absent new target's blob digest or
optional stat timestamp, not for a present target's digest.

```json
{
  "archive_backup_digest": "<sha256 of frozen HISTORY.md bytes>",
  "archive_backup_name": "HISTORY.md.bak[.n]",
  "conversion_complete_digest": "<conversion COMPLETE record sha256>",
  "prepared_target_new_digests": {
    ".cursor": "<sha256>",
    "MEMORY.md": "<sha256 or null if absent>",
    "history.jsonl": "<sha256>"
  },
  "receipt_digest": "<sha256>",
  "receipt_original_stat": [0, 0, 0, 0, 0, 1, 0, 0, 0],
  "receipt_size": 0,
  "schema_version": 1,
  "session_source_new_digest": "<sha256>",
  "transaction_complete_digest": "<transaction COMPLETE record sha256>",
  "transaction_identity": "<transaction identity digest>"
}
```

The sample numbers are placeholders, not valid evidence. The seal contains no
raw receipt, session text, or session key. It records claims over immutable
manifest/blob evidence *after* full current-target verification; it is a
historical semantic claim. Later legitimate edits to memory/session targets
must not be judged against these old mutable bytes. Later startup validates
the durable seal, immutable transaction/conversion manifests and blobs, and
the exact private receipt copy without requiring historical memory or session
targets to remain byte-identical. The historical admission protocol is
implemented outside this package in `memory_recovery_startup_admission.py`; it
does not call this current-target proof after those edits. A
committed quarantine index alone cannot substitute for historical semantic
validation or authorize receipt disposition. The Q1 private exact receipt copy
is retained indefinitely; its removal is an explicit operator-only offline
concern.
