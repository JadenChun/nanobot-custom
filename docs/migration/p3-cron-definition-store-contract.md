# P3 cron definition-store contract

`nanobot.cron.definition_store.CronDefinitionStore` is Package C2's typed,
in-memory adapter over an injected `CronActionStore`. It converts the opaque
action-store replay into a durable, definition-only candidate. It does not
open files itself, import `CronService`, create timers, execute jobs, deliver
messages, or change the migration guard. Timer, manual, system, and service
execution remain fenced until later migration packages establish their
separate authorities.

## Definition candidate

The checkpointed candidate has this exact shape:

```json
{
  "schema_version": 1,
  "kind": "cron_definition_candidate_v1",
  "definitions": ["definition-only canonical jobs"],
  "applied_action_ids": ["durable receipt IDs in order"],
  "legacy_conversion_evidence": {"preserved exactly when present"}
}
```

Each definition contains only `id`, `name`, `enabled`, `schedule`, `payload`,
`deleteAfterRun`, `jobInstanceId`, and `definitionRevision`. Runtime state,
run history, next/last-run values, and creation/update timestamps are rejected
from new typed definitions and are never written into this candidate.
`jobInstanceId` is a caller-supplied lowercase SHA-256 value; it is required
for new definitions and an update cannot change it. `definitionRevision` is
the SHA-256 digest of canonical sorted JSON for the definition fields plus
`jobInstanceId`, matching C1's legacy conversion calculation.

## Typed action protocol

`CronDefinitionAction.create`, `.update`, and `.delete` create strict JSON
requests with kind `cron_definition_action_v1` and schema version 1. Unknown
fields, non-finite JSON, invalid field types, runtime fields, and unknown
operations fail closed.

Create fails if the job already exists. Update and delete require the exact
current instance ID and definition revision. An update keeps the same instance
ID. The store applies actions in durable receipt order only while deriving a
replay; `append` merely records a request. A checkpoint accepts only a replay
created by that same definition store and only when the candidate's applied
action IDs exactly match all replay receipt IDs. It delegates the generation
CAS and durability boundary to `CronActionStore`, then replays the settled
typed candidate.

## C1 conversion seam

When C1 has produced `legacy_cron_snapshot_candidate`, C2 requires
`runtime_state_policy: "snapshot_wins"`, projects its snapshot jobs to
definitions, and validates that every embedded legacy operation has the same
action ID, receipt ordering, and C1 canonical request digest as the durable
receipt. It applies C1 `upsert` and `delete` records only during this one-time
projection. An upsert retains the canonical embedded stable instance ID and
definition revision while discarding its runtime state.

The exact `legacy_conversion_evidence` object is carried into the C2
candidate unchanged. Once checkpointed, the C2 candidate replaces the C1
snapshot candidate, so its `operations` list is never replayed again. The
action-store conversion manifest, completion provenance, backups, and closed
migration guard continue to fence the converted sources independently.
