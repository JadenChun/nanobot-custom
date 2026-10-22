# P3 Package D cron occurrence and completion contract

Package D installs the runtime substrate between the committed C2 definition
candidate and the later delivery package.  It does not arm a timer, execute an
agent turn, run a manual or system callback, publish a channel message, or
change the migration guard.  `CronService` continues to return the C3
definition-only admission diagnostic for every execution entrypoint.

## Definition-bound admission

`nanobot.cron.occurrence.CronOccurrenceRuntime` replays the injected
`CronDefinitionStore` for every admission.  A candidate with pending durable
actions is not executable until C3 checkpoints it.  The requested job must
exist, be enabled, and match any supplied `job_instance_id` and
`definition_revision` tokens.  A deleted, disabled, changed, corrupt, or
unavailable definition fails closed before a claim is returned.  The definition
is checked again after the durable occurrence claim; a cross-store change
leaves an explicit claimed record for reconciliation and returns no execution
authority.

Timer, manual, and system attempts all call the same `claim` method.  Their
identity is only the occurrence-store namespace, job ID, committed definition
revision, and scheduled fire time in milliseconds.  Source kind is provenance,
not identity.  Thus the same fire cannot be claimed twice by different owners.
An exact retry in the issuing process returns the original claim object; an
independent process sees `OccurrenceReconciliationRequiredError` until an
operator-controlled, externally fenced reconciliation supplies the predecessor
owner and generation.  The existing occurrence ledger remains the durable
authority for exclusive creation, generation changes, canonical completion,
and crash-state reconciliation.

The claim freezes a detached definition wire snapshot and destination policy.
Those values are evidence for a future delivery adapter; they do not authorize
a send.  Definition candidates, action journals, and legacy `jobs.json` or
`action.jsonl` inputs are never updated by occurrence admission.

## Completion and history projection

`nanobot.cron.completion.CronCompletionProjector` derives one stable run ID
from the exact occurrence ID and claim generation:

```text
cron-occurrence-<occurrence_id>-<generation>
```

Completion input must be a JSON object with `status` `ok`, `error`, or
`skipped`.  The projector binds `run_id`, `occurrence_id`, `job_id`,
`job_revision`, and generation into the canonical completion.  Any supplied
identity or session-turn provenance that disagrees with those values is
rejected.  The occurrence ledger is completed first.  Only after that durable
canonical outcome exists does the projector atomically write a private
`cron_run_projection_v1` record under the separate `runs` directory.  The
record carries `delivery_state: "fenced"`; no channel or external tool is
called.

Retries with identical canonical content are idempotent and re-prove file and
directory durability.  A projection failure after the occurrence rename may
leave a visible completed ledger record with no run projection; a later process
can call `project_completed(occurrence_id)` to recover it.  A claimed ledger
record is never projected as success: restart inspection exposes it through
`reconciliation_required()` and `project_completed` raises an incomplete-state
error.  A conflicting run ID, completion, or provenance fails closed.

Runtime/history projection is intentionally read-side evidence.  The C2
definition candidate remains the only source for `CronService.list_jobs()` and
`get_job()`; run projections cannot resurrect, delete, enable, or revise a
definition.  No legacy action or jobs file is dual-written.

## Durability and boundaries

The occurrence ledger's POSIX descriptor, lock, exclusive-create, fsync, and
integrity checks remain authoritative.  Completion projection uses a private
lock, private `0600` files, temporary-file fsync, atomic replacement, and
directory fsync.  A write, replace, or fsync error returns an error and never
grants delivery authority.  The implementation is qualified only on POSIX
systems with `flock` and no-follow operations.

The exact external side-effect boundary is still open for Package E.  A model
or channel provider may have performed an irreversible effect before a process
crashed, and Package D does not infer success or compensate it from a run
projection.  Destination receipts, uncertain-send reconciliation, channel
fanout, and the process/turn fence required before reclaiming an outstanding
claim remain delivery-package responsibilities.

## Acceptance evidence

Focused Package D tests cover exact retry, timer/manual/system identity
sharing, stale revision and deletion rejection, idempotent run projection,
projection failure recovery, explicit claimed restart state, and the unchanged
execution fence.  The existing occurrence-ledger suite continues to cover
exclusive cross-process claims, restart reconciliation, fork-inherited claim
rejection, and filesystem failure injection.
