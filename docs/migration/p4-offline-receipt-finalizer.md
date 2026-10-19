# Operator-only offline terminal receipt finalization

`finalize_pending_memory_receipt_offline(workspace, *, sessions_root)` is an
explicit operator action while the assistant is stopped. It does not run at
startup, in a channel, or from a constructor. The pending receipt blocks normal
runtime until this action has returned successfully. There is no automatic
recovery or cleanup. Its fixed content-free failure requires inspection by an
operator; failure never deletes an incomplete private copy.

The finalizer opens one existing-only workspace memory authority and operation,
then locates the existing checked session from the exact private receipt. The
checked workspace marker supplies `workspace_id`; the caller cannot specify
one. The workspace lock precedes checked-session locks. Before creating any
quarantine entry, the operation-bound terminal proof checks all seven
transaction records and six receipt-tail conversion records through COMPLETE,
immutable blobs, current targets, archive backup and source absence, and the
checked session's exact source. An existing-only authority does not mkdir or
repair missing layout, even when terminal verification opens evidence namespaces.

Only after that live proof does the operation-backed quarantine store exclusively
publish `prepared.json`, the exact private `receipt.bin`, and `copy-ready.json`
with file/parent fsync. Immediately before disposal the proof is revalidated,
then the pinned memory-parent borrow compares the fixed pending basename's
no-follow bytes and all nine original stat fields. It requires a single-link
owner-only `0600` regular file. The sole unlink is descriptor-relative on that
fixed basename. There is no stat-then-unlink atomicity guarantee against an
uncooperative same-UID actor. An unlink error with an absent name may proceed
only through the full absence and copy-mode proofs; a still-present name fails.

The store checks absence around a memory-parent fsync. Under the same checked
session and operation, a newly opened validated copy-ready lease supplies a
fresh copy-mode semantic proof. Its historical seal and original pending stat
must equal the pre-unlink proof. Only then can the store publish a sealed v2
`committed.json`. The exact raw private copy remains indefinitely; no finalizer
code removes it or edits transaction/conversion manifests.

`replay_unlinked_memory_receipt_offline(workspace, *, sessions_root,
transaction_identity)` is a separate, explicit operator action, allowed only
before ordinary runtime changes historical targets. It demands the exact
lowercase 64-hex identity, an intact targeted prepared/copy-ready chain and
private copy, and absent pending name. A descriptor-relative, existing-only
binding reader supplies bootstrap fields, never semantic authorization. The
copy gives the session key; the checked marker must match the stored workspace
ID. One existing-only operation re-proves the full terminal semantics, fsyncs
absence, obtains a fresh copy-mode proof, and seals the same identity. It never
guesses among incomplete identities or repairs partial entries.

Fault disposition is deliberately conservative:

| Observed state | Operator result |
| --- | --- |
| No quarantine identity and live pending | Fresh live finalization may begin |
| Prepared, partial copy, or uncertain copy with pending | Stop; manual reconciliation |
| Copy-ready with pending | Stop; never unlink on replay |
| Intact copy-ready with absent pending | Explicit identity-targeted replay, before runtime resumes |
| v1 or sealed v2 committed, conflicting or malformed evidence | Stop; never overwrite or clean up |

Directory, lock, marker, session, pending, blob, or manifest drift fails closed.
The workspace lock coordinates cooperative writers; these records are neither
an authenticity protocol nor a defense against arbitrary same-UID tree
replacement. The separate startup-admission protocol must verify its own durable
historical evidence and cannot treat structural quarantine state alone as
admission. No live production use is implied by this module.

## Candidate integration checkpoint (2026-09-28)

The existing candidate implementation now has a scoped functional acceptance
check from offline finalization and replay through historical-seal validation,
constructor admission, and CLI/gateway startup preflight. The focused
quarantine/terminal/finalizer/admission/entry-point suite passed 221 tests;
recovery conversion, lock, store, and session-memory integration added 58
passes. Scoped Ruff and `git diff --check` passed. No production code changed
in this validation round. An exploratory legacy CLI sweep remains red because
older fixtures patch removed entry-point symbols or expect execution past the
intentional migration guard; it is not a release-acceptance suite and must be
reconciled by the P5 integration owner. This checkpoint accepts only the
bounded recovery capability, not all of P4/P5, deployment, or startup of the
candidate. `MigrationIncompleteError` remains active.
