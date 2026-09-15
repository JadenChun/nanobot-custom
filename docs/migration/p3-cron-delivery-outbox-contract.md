# P3 Package E1 cron delivery provenance contract

Package E1 hardens the boundary between the completed occurrence ledger and
the later delivery outbox. It does not create an outbox, send a channel
message, call a provider, or grant delivery authority.

## Durable completed reads

`CronOccurrenceStore.get_completed_durable(occurrence_id)` runs under the
ledger lock and returns a detached `OccurrenceRecord` only after the exact
record and its containing directory have been fsynced and their descriptor
bindings rechecked. A claimed occurrence raises an explicit reconciliation
error; a corrupt, replaced, missing, or path-swapped record fails closed. The
returned record is evidence only and cannot be used as a computation or
delivery capability.

## Canonical provenance

When completion receives `provenance`, it must be a finite JSON object. The
projector copies it into the canonical occurrence completion under the reserved
`provenance` key before calling `CronOccurrenceStore.complete()`. If the
completion object already contains that key, the supplied value must be
canonically identical. This makes provenance part of the occurrence's durable
completion identity and keeps it available when the separate history
projection is missing.

Projection recovery reconstructs missing projections from canonical completion.
An explicit retry provenance must match that canonical value. A historical
Package D completion that has no canonical `provenance` remains readable and
reconstructable with empty/unavailable provenance; E1 never rewrites that
ledger record or invents provenance for it. If an immutable, strictly valid
legacy D projection already exists, an omitted-provenance retry may return its
historical nonempty projection evidence after verifying its exact identity,
canonical completion, and durability. That evidence is preserved as-is and
does not become canonical provenance.

An explicit retry is always compared with canonical completion: a nonempty
value fails when the canonical key is absent, and an explicit `{}` still fails
when an existing legacy projection carries nonempty historical evidence. E2
must derive provenance only from `get_completed_durable()`'s canonical
completion, preserve absence of the reserved key, and never use projection
provenance.

## Recovery and boundary

The ledger completion is durable before the separate fenced run projection is
written. If projection fails, a later projector instance reads the completed
ledger record durably and reproduces the exact canonical provenance without
recomputing the occurrence. Projection remains `delivery_state: "fenced"`.

Outbox creation, destination receipts, uncertain-send reconciliation, channel
fanout, and sender/provider/bus integration remain later delivery-package
responsibilities.

## Acceptance evidence

E1 tests cover canonical provenance surviving projection failure and restart,
reserved-key conflicts, strict finite JSON validation, historical completions
without provenance, immutable legacy projection evidence, durable
record/directory reproof, record replacement, in-place mutation, root swaps,
claimed-state rejection, and corrupt-record rejection. Package D and the
existing C1–C3 ledger suites continue to provide regression coverage for
identity, admission, projection, reconciliation, and execution fences.

## Package E2 durable outbox authority

`nanobot.cron.delivery_store.CronDeliveryOutboxStore` is the dormant E2
authority for one completed occurrence. It records delivery work and the
durable per-destination fence; it never invokes a sender, channel, provider,
message bus, service, tool, CLI, or scheduler. The migration guard remains
closed while this store is unconnected to runtime dispatch.

The preferred preparation call is:

```python
outbox = CronDeliveryOutboxStore(root / "outbox", namespace=namespace)
record = outbox.prepare_completed(occurrence_store, occurrence_id)
```

Preparation calls `get_completed_durable()` immediately before candidate
derivation and before taking the outbox lock, so the ledger and outbox locks
are never nested. A detached `OccurrenceRecord` is never accepted as
preparation authority: every call must supply the authoritative
`CronOccurrenceStore` and `occurrence_id`. A claimed, incomplete, corrupt,
replaced, or unavailable occurrence cannot create an outbox. The completion
and provenance used by E2 come only from that fresh durable read; no run
projection, definition store, or current definition is consulted. If
completion has no `provenance` key, E2 retains that absence and exposes
provenance as unavailable rather than inventing an empty value.

Completion must carry an explicit, strict text-only envelope under
`completion["delivery"]`:

```json
{
  "version": 1,
  "deliver": true,
  "messages": [{"content": "text", "media": []}]
}
```

The envelope has exactly those fields. Every message has exactly `content`
and `media`; content is a string and media is an empty list. A missing
envelope, unknown field, non-finite value, non-text message, or non-empty media
is unsupported. `deliver: false`, an empty message list, or a policy with no
usable destinations still creates a durable outbox record with a `no_work`
aggregate view. A destination policy is frozen from the occurrence: the
primary pair comes first, an absent or false primary channel uses `cli`, and
additional destinations follow. Pairs are deduplicated by exact channel and
target strings in first-seen order; no trimming, case folding, or other
normalization occurs. Later definition edits therefore cannot change a
prepared outbox.

The outbox ID, destination IDs, attempt action IDs, and request digests are
deterministic SHA-256 values over their versioned canonical JSON domains. The
record retains the complete immutable occurrence input, ordered destination
history, ordered attempt history, and exact transition history. Every mutation
increments the exact integer `record_revision` and appends one contiguous
transition entry carrying the affected destination, attempt, operation,
outcome, receipt, and reconciliation evidence. The parser replays that
history and rejects a revision or attempt history that cannot be derived from
it. Aggregate status is derived as `no_work`,
`pending`, `partial`, `delivered`, `failed`, or
`reconciliation-required`. Claimed, sending, and uncertain destinations are
always reconciliation-required; the store never applies a timeout, lease, or
automatic retry.

## E2 transition and reconciliation contract

`claim_destination(outbox_id, destination_id, expected_revision, owner=None)`
accepts only a pending destination. It durably appends the next contiguous
attempt and increments `record_revision` before returning a sealed
process-local `DeliveryClaim`. The claim is bound to the issuing store
object, process ID, destination, owner, and attempt. Copies, reconstructed
claims, foreign stores, forked processes, stale revisions, and consumed
claims have no authority. An exact reservation retry in the same process
returns the original claim after reproving the durable record. Another process
which sees a claimed destination receives a reconciliation-required or stale
revision error and never executes it.

`mark_sending(claim)` durably changes the retained attempt to `sending` and
keeps the same capability for settlement. `settle(claim, result)` consumes
that capability. A strict receipt is required for `confirmed_delivered`,
which produces `delivered`; `confirmed_not_sent` produces `pending` so a
future revision may explicitly claim a new attempt; `permanent_not_sent`
produces `failed`; and `unknown` produces `uncertain` with reconciliation
required. A live claim is fenced by its exact destination, attempt, owner,
action, request digest, and current attempt state; an unrelated destination
revision does not invalidate it. If persistence of the sending fence fails,
the claim is invalidated and no send authority is returned. A settled claim
never grants reinvocation.

`resolve_after_reconciliation` is a trusted store-only transition and does
not perform the external fence. It requires the exact predecessor outbox,
destination, attempt, and owner, the expected revision, the stable attempt
`action_id`, its canonical request digest, a bounded `evidence_ref`, and a
bounded `predecessor_fenced_ref`. An exact action retry reproves the durable
record before stale-revision checking; reuse with a conflicting digest or
other request fields is rejected. A claimed predecessor may resolve only to
`confirmed_not_sent` after the fence. Sending or uncertain predecessors may
resolve to `confirmed_delivered`, `confirmed_not_sent`,
`confirmed_compensated`, or `abandon`. Those outcomes retain the resolution
receipt and evidence and lead to delivered, pending, pending, or failed
respectively. Only `confirmed_not_sent` and `confirmed_compensated` permit a
successor attempt; delivered and permanent failure histories reject later
attempts. A stale prior settlement is rejected. Attempts, transition
evidence, and resolution receipts are retained without pruning, and the parser
requires every retained receipt and evidence field to agree with its
corresponding transition.

## E2 strict wire and filesystem boundary

Each final record is one strict UTF-8 JSON object named by its outbox digest.
The exact top-level fields are `schema_version`, `kind`,
`store_namespace`, `outbox_id`, `immutable_input`, `destinations`,
`record_revision`, `transition_history`, and `integrity`. Immutable input preserves completion,
delivery, occurrence identity, generation, owner, job snapshot, destination
policy, and an explicit provenance-availability bit. Destination and attempt
objects use exact schemas, contiguous histories, deterministic IDs/digests,
and state-derived aggregate invariants. Duplicate JSON keys, unknown fields,
non-finite values, booleans in integer fields, malformed hashes, impossible
state combinations, mismatched integrity, and namespace or identity conflicts
fail closed.

The encoded record is capped at 1 MiB; no outbox has more than 256
destinations or more than 256 retained attempts for one destination; and
owners, action IDs, request digests, destination identifiers, and evidence
references are capped at 512 UTF-8 bytes where applicable. The outbox root is
created and retained as mode `0700`; lock, record, and staging files are
single-link regular files with mode `0600`. Reads reject oversized data before
allocation and recheck descriptor metadata after reading. Mutations use
held no-follow descriptor chains, descriptor-relative exclusive create or
atomic replacement, file fsync, directory fsync, and root/lock/record/staging
inode, metadata, and content reproof across every acknowledgement boundary.
Mutations retain the predecessor record descriptor, inode, metadata, and
content signature through staging and reprove it before replacement, so an
identical private replacement fails closed. The intended staging signature and
content are captured before file fsync and compared after file fsync, rename,
and directory fsync; an in-place mutation at any point is never
acknowledged. Symlinks, hardlinks, permission weakening, root or lock swaps,
record replacement, and unsafe staging entries fail closed. Windows import
succeeds, but construction raises a clear unsupported-platform persistence
error when the required POSIX primitives are unavailable.

E2 acceptance evidence covers deterministic destination ordering and IDs,
false delivery and zero-work records, immutable old policy, provenance
absence, idempotent preparation and post-rename durability reproof, two
process reservation and process-local capability fences, all store-only
transitions and aggregate states, reconciliation retry/conflict/stale-owner
checks, strict schemas and limits, and root/lock/staging/record filesystem
attacks, including pre-publication settlement failure, reconciliation, and
lost-ack retries. No E2 test grants or invokes real delivery authority.

If a settlement or replacement fails before publication, the issued capability
is consumed and no sender authority is returned. A later authoritative read
sees either the unchanged predecessor or the durably published successor; an
acknowledgement lost after publication is retried only by reproving the exact
record and transition, while an in-flight destination remains subject to
explicit reconciliation.

## Package E3 dormant delivery runtime

Package E3 provides the dormant runtime substrate for one explicit,
process/store-scoped `DeliveryClaim`. `CronDeliveryRuntime` accepts only a
`CronDeliveryOutboxStore` and, when preparation delegation is needed, the
explicit matching `CronOccurrenceStore`. Their namespaces must be identical;
the runtime never constructs either authority implicitly. `CronService` may
receive that runtime or outbox store only through explicit injection with the
same occurrence authority and namespace. A management-only service therefore
does not create an outbox, occurrence ledger, sender, or delivery directory.

The sender boundary is intentionally small and typed:

```python
@runtime_checkable
class Sender(Protocol):
    async def send(self, request: DispatchRequest) -> SenderResult: ...
```

`DispatchRequest` is frozen and detached. It carries the exact outbox,
destination, attempt, action, and request digest identities, the exact
channel and target strings, and an ordered tuple of text-only messages with
empty media. `ConfirmedDelivered` requires a nonempty finite JSON receipt;
`ConfirmedNotSent`, `PermanentNotSent`, and `UnknownOutcome` each carry a
bounded reason code. Unknown or malformed sender returns are rejected.

`dispatch()` first asks E2 to durably mark the issued claim as `sending`.
Only that successful acknowledgement permits request construction and the
single `sender.send()` call. The outbox lock is not held while the sender is
awaited, so a sender can independently read the outbox. Results settle as
`confirmed_delivered`, `confirmed_not_sent`, `permanent_not_sent`, or
`unknown`, producing the corresponding E2 destination state. Timeout,
cancellation, sender exception, invalid result, and partial-bundle signals
settle `unknown` before the condition is propagated. Sender invocation
ownership is retained on the exact outbox store, so concurrent dispatches
through separate runtime adapters share one task and an already-dispatched
claim cannot start a second sender. A timeout or caller cancellation is
recorded independently of sender behavior: a sender that suppresses
cancellation and returns a success result is still settled as `unknown`.
Cleanup joins the sender task through repeated cancellation before the caller
receives its timeout or cancellation. A settlement failure never retries the
sender: the runtime raises a settlement error retaining the original
classification, and the destination remains available through
reconciliation-required state. A claim that never reaches durable `sending`,
or a stale, copied, reconstructed, or foreign claim, never reaches the
sender. A partial bundle makes the whole destination uncertain; a later retry
is possible only through explicit operator reconciliation and does not replay
the bundle automatically. Same-task or inherited-child-task reentrancy and
malformed result variants fail closed and are durably classified as unknown.
If unknown settlement
itself fails during caller cancellation, cancellation remains the public
result while the retained invocation records the settlement failure for
reconciliation.

The invocation registry is bound to the exact in-memory outbox store object
and process ID. Shallow store copies and fork-inherited registries therefore
get isolated registries, while E2's exact-store and process claim checks reject
inherited claims before a copied or forked adapter can invoke a sender.

These APIs expose preparation, detached reads and status, claims, dispatch,
and reconciliation-required inspection only. There is no scheduler, timer,
channel/provider factory, bus, CLI, tool, or real sender wiring in E3.

## Remaining E3 activation gates

This package accepts the fake-sender dormant substrate and its deterministic
durability tests. Production activation still requires the stopped-writer and
process/turn lifecycle fence, real transport and operator reconciliation
evidence, channel/provider ownership, media and result-mode support, and an
explicit migration-guard decision. Those gates remain closed; E3 does not
arm scheduling, invoke a channel, publish to a bus, or make external delivery
available merely by being importable or injected for tests.
