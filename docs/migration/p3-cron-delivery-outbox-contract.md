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
