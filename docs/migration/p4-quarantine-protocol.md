# Private receipt-quarantine storage protocol (storage only)

`nanobot.agent.memory_recovery_quarantine_storage` owns the separate
`memory/.recovery/receipt-quarantine/<transaction-identity>/` index. It does
not change the transaction/conversion manifests or remove a pending receipt.
Startup admission now consumes this evidence through
`memory_recovery_startup_admission.py`; constructors and entrypoints do not call
the storage API directly. This storage protocol retains raw receipt bytes
indefinitely, including unknown fields; cleanup is a deliberate manual operator
task outside this API. The copy must not be surfaced in prompts, WebUI, Git, or
logs. See the [final P4 closure checkpoint](p4-closure-checkpoint.md) for current
acceptance and activation status. P4 acceptance does not authorize runtime
activation, guard removal, cron scheduling, deployment, live traffic/state
migration, or Windows/release qualification.

## Read-only inspection

`inspect_receipt_quarantine(workspace, *, expected=None)` creates no directory,
file, or lock and does not fsync. It returns `QuarantineInspection(clean,
incomplete, committed_identities, sealed_identities)`. Structurally committed
v1 records remain in `committed_identities`; only v2 sealed records appear in
`sealed_identities`. Neither collection is startup admission. Only absence of both a pending receipt and
the index is clean, including an absent workspace or memory directory. An
existing empty index, partial chain/copy, unknown entry, or live receipt is
never clean. Unsafe entries, malformed records, unsafe permissions, unsupported
POSIX descriptor capabilities with present evidence, or identity conflicts
raise a fixed, content-free `QuarantineError`. On unsupported platforms the
absent-evidence case remains inspectable, but present evidence cannot be
admitted. Inspection bounds each record to 4096 bytes, the copy/receipt to
64 MiB (the existing memory-target storage bound), and each transaction
directory to four entries. The index uses streaming descriptor-relative
enumeration without a lifetime identity ceiling; committed identities are
sorted for deterministic results. Its time is linear in total index entries
plus sorting committed identities; retained result memory is proportional to
committed identities, not incomplete or unknown names, while per-file reads
remain bounded. Unexpected names fail immediately with a fixed error.
Inspection rechecks observed directory, index, and pending-entry state before
returning, detecting in-call creation/replacement or mutation; it cannot
authenticate an arbitrary same-UID process that races after the final check.
It reads no receipt bytes into an exception. A structurally committed identity
is **not** proof of transaction/session semantics and is
never itself startup-admissible.

`expected=QuarantineBinding(...)` optionally checks the named transaction's
workspace ID, sessions root, COMPLETE digest, receipt digest, and size while
still enumerating other committed identities. The workspace and sessions root
are absolute path plus device/inode/mode/owner/group bindings, not a claim that
their current contents or markers are semantically valid. A later admission layer must
independently verify terminal transaction and checked session evidence.

## Locked publication

Create `QuarantineBinding(workspace, sessions_root, workspace_id,
transaction_identity, complete_digest, receipt_digest, receipt_size)` from
already validated recovery evidence. Under an existing
`open_workspace_memory_lock(workspace)` gate, use
`with ReceiptQuarantineStore(gate, binding) as store:`. Its entry validates the
live gate and descriptor-bound workspace/memory/sessions-root identity and
exclusively creates private `0700` recovery/index/identity directories. It
never enters another lock. A failed or interrupted stage leaves its residue
intact; no stage overwrites, auto-repairs, resumes, or deletes evidence.

1. `store.prepare(raw_receipt_bytes)` re-reads and binds the current private
   `0600`, single-link pending receipt, verifies exact bytes/digest/size, and
   exclusively publishes `prepared.json`.
2. `store.copy_receipt()` re-reads the same stat-bound receipt, exclusively
   writes `receipt.bin` with its exact raw bytes, fsyncs the copy and its
   directory, then re-reads and checks bytes and stat before publishing
   `copy-ready.json`. A partial copy remains visible and blocks replay.
3. A separate finalizer may dispose of the original only after its independent
   transaction/session proof. Once it has done so, `store.prove_durable_absence()`
   checks pathname absence around a bound memory-directory fsync and issues
   a same-store `DurableAbsenceProof`. An operation-backed store then opens an
   existing copy-ready lease and obtains a fresh `TerminalCompleteProof` from
   terminal verification *after* durable absence. `store.commit(absence_proof,
   terminal_proof)` requires the exact issued proof under that store's authority
   and operation, rechecks semantics immediately before publication, and binds
   COMPLETE, transaction identity, original receipt stat/digest/size, copy and
   canonical seal digest. A gate-only store can prepare/copy, not sealed-commit.
   It does not accept a boolean assertion or itself unlink/rename the pending receipt.

Records are canonical sorted compact JSON plus one newline. Prepared and
copy-ready remain version 1; newly published committed records are version 2,
with the same exact fields plus one lowercase SHA-256 `semantic_seal_digest`.
Old v1 committed records remain structural residue, never sealed evidence.
Strict fields and predecessor SHA-256 digests bind the canonical workspace
path and device/inode/mode/owner/group, workspace ID, sessions-root path and
the same stat fields, transaction identity, COMPLETE record digest, receipt SHA-256/size and original
stat, and (from `copy-ready` onward) the fixed copy name and stat. They contain
no receipt text. Directory entries and files are opened no-follow relative to
checked descriptors; files are `0600`, owner-only, single-link. Each record is
written with exclusive create, file fsync, parent fsync, and post-sync re-read.
An interrupted direct exclusive publication can leave an incomplete final
entry; inspectors must block rather than infer a clean state. A complete
record whose final fsync reports an error may be structurally visible despite
unproven durability. Neither a structurally committed record nor a failed
publication return is admission authority: startup admission revalidates
transaction/session semantics and durability independently.
Cooperative writers must use the workspace lock. This is not an authenticity or rollback
mechanism against an arbitrary same-UID process replacing the entire tree.

## One-lock composition and targeted copy reads

When the caller already holds an active `WorkspaceMemoryAuthority` and its
`WorkspaceMemoryOperation`, `ReceiptQuarantineStore.from_operation(authority,
operation, binding)` borrows that exact lock and pinned workspace descriptor.
It does not acquire a nested lock or issue a second authority. The gate-based
constructor remains available for lock-only callers. Both paths recheck the
workspace, memory/recovery/index/identity ancestry and sessions-root binding
around I/O; a foreign, closed or cross-thread operation fails closed.

`open_validated_quarantine_copy(authority, operation, binding,
require_phase="copy-ready" | "committed", require_pending_absent=True)` is a
read-only context-managed lease on one identity. It opens only existing
entries; it neither creates nor repairs files, acquires another lock, fsyncs,
appends evidence or disposes of the pending receipt. The lease validates the
canonical predecessor-linked records, exact private copy, original nine-field
receipt stat and distinct copy stat, binding and directory ancestry. Its
typed `raw_bytes`, `original_stat`, `copy_stat`, `phase`, and `binding` properties
revalidate before use, as does `require_current()`. A `copy-ready` identity
whose pending receipt is already absent remains readable for replay, before a
`committed` marker is published. An incomplete or drifting identity is not
readable. This structural evidence is **not** semantic transaction/session
proof and never grants startup admission or authorizes receipt disposition.

`open_existing_workspace_memory_lock(workspace)` provides a separate
existing-only gate for `establish_durable_quarantine_seals(gate)`. Under that
already-held lock the barrier rejects live pending, incomplete, unsafe,
unknown, v1-unsealed or drifting entries across *all* identities. It checks
bounded exact copy/record bytes, stats, predecessor chains and bindings;
fsyncs each copy and record, each identity directory, then index, recovery,
memory and workspace directories; and rechecks after the final sync. It does
not create, rewrite, delete, repair or take a nested lock. It streams identities
with bounded per-file reads and descriptor use. The returned
`DurableQuarantineEvidence` is storage durability evidence only, not semantic
startup admission. A complete v2 marker visible after a failed publication
fsync can pass a later successful barrier; a partial marker remains blocking.
A missing workspace or lock fails without creation; a present lock uses the existing
no-follow, private owner/mode/one-link and descriptor revalidation protocol.
Startup admission invokes this validation through the separate
`memory_recovery_startup_admission.py` policy; the original create-if-needed gate
remains unchanged.

The finalizer must reprove transaction `COMPLETE` under memory-to-checked-session
lock order before `prepared`, publish `copy-ready` before receipt disposition,
then reprove terminal and session evidence after durable absence before
`committed`. The finalizer and startup-admission wiring live in their separate
recovery modules rather than in this storage package.
