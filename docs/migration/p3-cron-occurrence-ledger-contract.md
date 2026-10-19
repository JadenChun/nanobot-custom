# P3 private scheduled-occurrence ledger contract

`nanobot.cron.occurrence_store.CronOccurrenceStore` is an isolated, private
ledger for synthetic scheduled fires. It is not wired to cron dispatch while
the migration guard and cron admission fence remain in place. The low-level
store supplies an exact scheduled claim and, in the later dormant extension
below, a one-shot computation-start gate. Package D's `CronOccurrenceRuntime`
supplies the shared identity and reconciliation seam for timer, manual, and
system attempts; none of those sources is automatically dispatched.

Each occurrence ID is a SHA-256 digest of the schema tag and these typed
values: store namespace, job ID, job revision, and scheduled fire time in
milliseconds. The digest, rather than any caller-controlled ID, is the sole
record filename. A first claim records a canonical detached job snapshot and
destination policy, an opaque owner token, and generation 1. Reusing the same
identity with changed frozen inputs fails closed. Claim creation uses a private
ledger directory, process lock, exclusive create, file fsync, and directory
fsync before returning computation authority. The encoded record has a hard
size limit and is rejected before any claim, completion, or reconciliation
replacement is opened for writing.

The legal durable path is:

```text
absent --exclusive claim(owner, generation 1)--> claimed
claimed --persist canonical completion--> completed
claimed --explicit reconciled reclaim--> claimed(owner', generation + 1)
```

Only the exact claim object issued by the same store instance and process for
the current owner and generation can complete the record. Reconstructed and
fork-inherited claims are rejected. Completion is accepted again solely when
its canonical JSON content is identical. An exact retry of an outstanding
claim in that same process returns the originally issued capability; an
independent process still receives a reconciliation-required error. A
completion retry rereads, verifies, and
fsyncs both the record and held directory before it returns. This lets a retry
establish durability after a post-rename directory-fsync error. A stale owner
cannot complete a successor generation.

Opening the ledger is read-only with respect to final records. It validates all
records and exposes every outstanding claim via `reconciliation_required()`.
A new process must treat those records as non-executable and `claim_scheduled`
returns a reconciliation-required error for the same identity. Opening another
process never invalidates a live owner. An authorized reconciliation flow may
call `reclaim_after_reconciliation` only after external evidence establishes
`confirmed_not_started` or `confirmed_compensated`; it must supply the
predecessor owner and generation. The store non-blockingly acquires the same
persistent per-occurrence execution lock used by canonical execution before it
rereads and replaces that exact predecessor. A live execution therefore
returns an explicit busy error without changing the record. The short ledger
lock is acquired only after the per-occurrence lock and never spans a model
await.

Computation must call `complete()` successfully before a future delivery layer
derives sends from the frozen policy. Replace or fsync errors return no claim or
completion result and therefore grant no execution or delivery authority.
Malformed, permission-weakened, conflicting, hardlinked, or integrity-invalid
final records fail closed and are never replaced. The ledger uses held POSIX
directory descriptors, no-follow/nonblocking opens, inode binding checks, and
private owner/type/link-count checks for root, lock, records, and staging files.
Existing roots and locks are rejected when unsafe; they are never chmod-repaired.
The instance detects root and lock swaps after opening. Ledger roots are mode
`0700`; lock, record, and temporary files are mode `0600`, so snapshots,
destinations, and canonical outcomes stay private on the local filesystem.

The implementation is qualified only on POSIX systems with descriptor-relative
no-follow operations and `flock`; it imports on Windows but construction fails
closed with a clear unsupported-platform error. It walks ancestors using held
descriptors, but does not claim to validate or own every ancestor; deployment
must supply a trusted absolute ancestor path.

Every construction walks that absolute path again. It validates each visible
component against the parent descriptor and fsyncs both the child directory and
its parent entry, even when the component already exists. A persistent parent
fsync failure therefore fences every retry and concurrent initializer rather
than becoming bypassable after an earlier partial root creation. A newly opened
child descriptor is closed before ownership transfer on any chmod or fsync
failure.

This package does not establish external exactly-once delivery. The later cron
runtime package must use the durable canonical completion as its only input to
destination receipt and uncertain-send reconciliation. Explicit occurrence
reclamation still requires exact predecessor evidence; the store itself now
enforces exclusion from any live canonical execution with the per-occurrence
OS lock.

2026-09-29 dormant per-occurrence execution-fence extension: canonical
execution first acquires `.execution.<occurrence-id>.lock`, a persistent private
regular owner file whose validated name is derived only from the 64-hex
occurrence ID. It retains the nonblocking OS lock through start reproof, the
model turn, durable completion, projection/outbox preparation, and optional
visible-chat linking. The file is created descriptor-relatively with
`O_NOFOLLOW|O_NONBLOCK`, exact mode `0600`, one link, file and directory fsync,
and pre/post-flock plus release binding checks; it is never unlinked. Existing
execution-lock entries are constructor-validated with the same exact grammar
and safety properties. Every acquisition repeats both fsync barriers before
flock authority, including an existing residue left visible by an earlier
failed create or fsync acknowledgement.

The store issues a nonserializable `OccurrenceExecutionFence` only for its
exact process-local claim. Store-private issuance state binds the capability to
the occurrence ID, owner/generation, canonical frozen snapshot and policy,
acquisition PID and asyncio task, active/started state, held root binding, and
lock descriptor/inode. Copies, reconstruction, serialization, foreign-store or
foreign-task use, mutation, inactive reuse, stale claims, and fork inheritance
fail closed. A process-global weak registry and child fork hook close inherited
descriptors without `LOCK_UN`, so the child cannot act or prolong the parent's
lock.

`begin_execution(claim, fence)` reproves that exact active fence and the
still-claimed durable owner/generation/frozen inputs under the short ledger
lock before consuming the one-shot local start marker. Claimed-to-completed
mutation requires the same active, started fence; identical completion retry
remains available only while it is held. Failure or cancellation releases the
OS lock, retains the consumed local marker, and leaves an incomplete durable
claim for explicit reconciliation. A completed occurrence recovers by durable
projection, outbox preparation, and visible linking without recomputation.
There is no timeout, lease expiry, waiting takeover, inferred resolution, or
automatic reclaim.

Process death can still leave detached external effects whose completion is
not represented in this private ledger. Those effects remain operator evidence
and compensation responsibility; the store never infers their outcome. No
production scheduling or automatic reconciliation is authorized by this API.
