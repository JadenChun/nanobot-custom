# P3 C3 definition-only cron service contract

`nanobot.cron.service.CronService` is the management adapter over the
committed C2 `CronDefinitionStore`, which in turn owns the injected C1
`CronActionStore`. The service retains its historical `store_path` argument
for callers, but never reads or writes that `jobs.json` path or its sibling
`action.jsonl`. A default construction binds the action store to the cron
directory; callers may inject an already configured `CronDefinitionStore`.
The stopped-writer legacy converter is an explicit operation and is never
called during construction or startup. If an unconverted `jobs.json` is
present, management fails closed with an explicit-conversion diagnostic rather
than presenting an empty list.

The service projects each typed definition to the existing `CronJob` shape for
list/get and management callers. Projections carry the exact
`job_instance_id` and `definition_revision`, while `state`, run history, and
timestamps are empty because C2 owns definitions only. Those fields are never
serialized into a definition action or candidate. New jobs receive a stable
lowercase SHA-256 instance identity; updates retain it and C2 recomputes the
revision from canonical definition JSON. Create, update, enable, and delete
actions are typed `cron_definition_action_v1` requests. Updates and deletes
require both exact instance and revision tokens. The legacy method signatures
remain available with token keywords added; a caller that omits required
tokens receives a `CronDefinitionTokenRequiredError` and no action is
appended. The old startup `remove_system_job(job_id)` call is a safe no-op
without tokens, while token-bearing system deletion is still protected.

The cron tool admits at most 256 additional destinations, requiring each item
to be exactly `{channel,to}` with trimmed, nonblank string identifiers no longer
than 512 UTF-8 bytes. It preserves input order and raw duplicate entries in
the definition; `CronPayload.delivery_destinations()` later places the primary
destination first and removes only exact duplicate pairs. Contextual adds use
the immutable turn request route, map the unified session to its concrete
`channel:chat` route, detach request metadata, and reject cron-origin nested
scheduling. The service deep-copies the admitted destination list before the
definition is serialized.

Each mutation derives its action ID from the exact typed request. It retries a
stale action-store generation only before `append` acknowledges a durable
receipt. After acknowledgement, a stale checkpoint raises
`CronDefinitionConflictError`; the receipt remains durable for a later replay
and no second action is appended. C2 replay is checkpointed after the first
management read when converted embedded operations or pending typed actions
are present. This replaces the legacy conversion candidate with the typed C2
candidate, so converted operations are applied once and retained conversion
evidence stays under the action store's authority.

The migration fence is intentionally closed. `start()` loads definitions only
and never arms a timer. Timer ticks, manual/forced runs, system-event
execution, automatic scheduling, occurrence execution, completion/run-record
projection, and delivery all remain fenced with the stable
`CRON_DEFINITION_ONLY_DIAGNOSTIC` until Packages D/E install their separate
runtime, occurrence, and delivery authorities. `admission_diagnostic()`
returns that same diagnostic for every job kind, including system events.

Legacy management callers that list definitions continue to receive detached
`CronJob` projections. Callers that mutate by ID without the C2 concurrency
tokens must be updated to carry the identity and revision from the list/get
response; silently mutating through the old ID-only interface would violate
the definition authority and is therefore rejected or made a no-op at the
compatibility startup seam.
