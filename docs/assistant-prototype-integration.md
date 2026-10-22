# Personal assistant prototype integration

Updated 2026-10-23. Implementation branch: `codex/complete-remaining-work`, based on
`79934a5630ca419e02e124693efa938d9d3407f0`. The client's running installation has
not been changed. This is a candidate for a copied-state rehearsal, not a
production cutover record.

## What is implemented

| Area | Result |
| --- | --- |
| Repository discovery | `repository_catalog` remembers existing local Git roots, purposes and project-to-knowledge associations. Registration neither clones repositories nor grants access |
| Context | The prompt receives a bounded catalog. The assistant chooses files to read and uses explicit paths/cwd when switching projects. Existing configured context adapters remain compatible |
| Knowledge saving | `save_knowledge` appends selected durable facts and decisions to the designated repository, with idempotent appends, atomic writes and the existing filesystem/store/proposal policy. Agent instructions call for saving useful knowledge before replying |
| Knowledge Git sync | Only opted-in knowledge repositories and path patterns sync automatically. Local commits survive network or merge failure. Unrelated staged changes stay separate; selected paths with user-staged versions refuse automatic sync. Conflicts retain both histories; there is no reset/reapply recovery |
| Background processes | `exec(notify_on_exit=true)` journals a long-running external process and queues its completion to the same assistant. Completion retains the original project, conversation, instructions, output identity and reply route |
| Busy Telegram input | Explicit gateway composition registers the existing inbox after bot authentication and before polling. Status, steering, queued work and `/stop` use one owner. A separate, tool-free control model proposes routes |
| Marketing scheduling | An explicitly enabled gateway owns the timer, lifetime lease, canonical fresh turns, external checks, receipt delivery, owner alerts and post-delivery commands |
| Telegram topics | The frozen primary topic travels through execution and delivery. A receipt confirms the requested topic; additional destinations do not inherit it |
| Codex quota recovery | Primary reset plus 120 seconds, at most three consecutive reschedules, no exhausted weekly window or wait beyond 24 hours. Retry intents/budget persist; confirmed client delivery resets the budget |
| Optional integrations | Crawler provider setup/instructions/workflow and crawler/VPN registrations live under `nanobot/extensions/`. Existing tools, flags and capability restrictions remain available. Legacy manager fields/methods are compatibility adapters |
| Compatibility fixes | Invalid manifests fail closed; external library skills appear in discovery; the old Copilot token location is supported; scheduler reads tolerate access-time changes |

Knowledge selection is agent-driven. This does not add an automatic transcript
extractor, semantic task-context manager, required context-repository schema,
duplicate checkout, second background model worker or Rust runtime.

## Catalog and knowledge workflow

With filesystem tools enabled, ask the assistant to remember the existing
project and knowledge-repository paths. The tool calls have this shape:

```json
{"action":"remember","name":"notes","path":"/absolute/existing/knowledge","kind":"knowledge","purpose":"Shared project decisions","auto_sync":true,"sync_paths":["decisions/**","notes/**"]}
```

```json
{"action":"remember","name":"project-a","path":"/absolute/existing/project-a","purpose":"Project A code","knowledge_repos":["notes"]}
```

Then a selected decision can be saved with:

```json
{"repository":"notes","path":"decisions/project-a.md","content":"The agreed durable decision."}
```

The result distinguishes local saving from remote synchronization. A pending
sync means the local note remains available; resolve connectivity or the Git
conflict before retrying. Registering a code repository cannot opt it into this
automatic knowledge synchronization.

Catalog, process and scheduling state live under the captured instance runtime
directory, separately from repository knowledge. Existing filesystem roots,
manifest protections and the selected shell/sandbox mode continue to apply.
A catalog entry does not expand those permissions. Repository metadata is small;
full knowledge files are read on demand. The legacy configured context adapter
still loads its configured entrypoints for compatibility.

## Explicit gateway settings

All three new gateway integrations are disabled by default. Enabling cron does
not enable Dream, heartbeat or idle compaction. Direct SDK/API construction does
not start these gateway timers or channel handlers.

The relevant configuration fragment is:

```json
{
  "gateway": {
    "processContinuations": true,
    "cron": {
      "enabled": true,
      "pollIntervalS": 1.0,
      "readinessTimeoutS": 60.0,
      "drainTimeoutS": 5.0,
      "adapterCwd": "/absolute/existing/script-working-directory",
      "codexQuotaRetry": true
    },
    "telegramBusy": {
      "enabled": true,
      "sessionKey": "telegram:123456789",
      "ownerSenderIds": ["123456789"],
      "ownerChatIds": ["123456789"],
      "controlProvider": "your-configured-provider",
      "controlModel": "your-explicit-control-model"
    }
  }
}
```

These example IDs and paths are placeholders. Busy input requires the enabled
Telegram channel's numeric `allowFrom` entries to belong to the configured
owners. The selected implementation uses a persistent Telegram session with
`agents.defaults.unifiedSession=false`. Temporary and shared cross-channel
inboxes remain outside this integration; existing WebUI recovery is retained.
The control model has no tools or task transcript. Scope defaults capture a
finite snapshot of currently registered tools; explicit `toolScope` and
`automationScope` lists can restrict that snapshot further.

For background execution, pass `notify_on_exit=true` to `exec`. Inline completion
does not start another turn. Cancellation/termination does not schedule a
continuation. An interrupted process or uncertain continuation is retained as
unknown after restart and is not blindly repeated. `exec_session` can retrieve
the retained result by its process identity within the original conversation.

## Importing old marketing jobs on a copy

From this source checkout:

```bash
python -m scripts.import_cron_copy \
  --source /absolute/stopped-snapshot/cron \
  --destination /absolute/candidate-workspace/cron \
  --timezone Asia/Kuala_Lumpur
```

The destination must be unused and separate from the source tree. Conversion
runs in a private staging directory and publishes the completed store atomically.
It retains source hashes, original history/retry state, conversion backups and
a migration receipt. Repeating the same import is idempotent; changed inputs or
changed imported definitions are refused. Existing explicitly session-bound
jobs require review rather than an implicit change in execution semantics.

Legacy unbound jobs receive an explicit `legacy_fresh` execution identity and
committed scheduling evidence. New jobs created through an enabled gateway use
that supported fresh-turn contract. Origin chat/topic, configured checks,
planning overrides, owner alerts and post-delivery commands are preserved.
Unsupported active jobs are refused before activation.

External checks run against the exact outgoing text. `skipVerification` does
not bypass configured external checks. Failed verification blocks client output.
Post-delivery commands receive `date`, `ack_status` and `repo_root` substitutions;
their failure after successful delivery produces an owner alert without resending
the client's result. The harness contains no marketing-specific validation or
rotation rules.

On startup, past scheduled fires are recorded as missed. The gateway resumes
future fires and does not catch up by replaying old work. While the gateway is
running, new committed revisions get their exact commit boundary without
discarding another job's due work. Computation and delivery with uncertain
outcomes require reconciliation. Imported legacy retry budgets are seeded once without resetting their counts.
Pending old retries are marked unknown for operator review, since an old error
alone does not prove that no tools ran. The stopped-writer
`CronQuotaRecovery.reconcile_pending` API records external evidence and can
abandon a pending retry, or requeue one confirmed not to have started; it refuses
review while the scheduler lease is active.

Native quota recovery only applies to actual Codex rate-limit failures with no
tool or explicit message effects. Ordinary provider transport retries remain
separate. Unreadable quota, exhausted budget, weekly exhaustion, tool effects,
verification failure and uncertain delivery cannot trigger whole-job replay.

## Rehearsal and production release

1. Capture the deployed commit, configuration drift, context-repository revision
   and consistent stopped copies of configuration, sessions, memory and cron.
2. Preserve those originals. Prepare a candidate workspace/instance with isolated
   credentials, providers, routes and script adapters. A copied configuration may
   still point at real client destinations or writable external systems; review
   those references before running it.
3. Import the rehearsed job copy and check its receipt and any pending-retry
   review items. Reopen historical sessions/memory through the existing migration
   admission tools. Keep the old token-budget configuration backup because
   serialization normalizes its old object shape.
4. Exercise each inventoried marketing workflow with controlled delivery,
   verification failure, bookkeeping failure, quota recovery, stop, shutdown and
   restart. Confirm the old installation can still read its untouched state.
5. Choose a cutover boundary with one active scheduler. Reconcile pending or
   uncertain occurrences and receipts before switching or rolling back; restarting
   the old runtime against stale state can repeat already delivered work.

Live deployed-version drift and production-specific scripts/destinations are
release checks. Local test adapters do not establish their live behavior. No
production upgrade or client delivery was performed in this implementation.

## Validation record

The private local evidence directory is
`.assets/validation/remaining-work-2026-10-22/`. Each implementation batch retains
before/after full-source checker reports, source hashes and affected test output
under the same frozen Python dependency environment. All completed batches have
zero introduced diagnostic signatures and a zero-error strict report.

The normal gateway integration test runs a due job through the canonical runner
and acknowledges it through a local receipt adapter. Additional checks cover
topic binding, duplicate/restart behavior, busy input, A/B project context,
background A completion while B is selected, managed knowledge policy, Git index
preservation and conflicts. Final local qualification passed:

- Python: **16,441 passed**, 22 skipped, plus **128 passed** in the serial CLI run;
  6 subtests also passed.
- WebUI: **1,186 passed** across 75 files; production build succeeded.
- Unchanged strict checker: **483 files, 0 errors, 0 warnings**. The diagnostic
  comparator reports `PASS_NO_NEW_DIAGNOSTICS`; the frozen 420-distribution
  checker environment remained unchanged.
- Full Python/test lint and whitespace checks passed.
- Wheel and sdist built; all 483 packaged Python files and the tool contract match
  the validated source, and the WebUI bundle is included.
- Fresh isolated wheel installation loaded all 17 channel plugins, checked their
  dependencies, imported the new integrations and ran CLI help. New automation
  defaults remain disabled.
- Copied session migration/rollback and cron conversion rehearsals passed;
  configuration compatibility, memory recovery, retained legacy budgets and
  stopped-writer retry review passed in the test suite.

See the [validation record](assistant-prototype-validation.md) for commands,
source/artifact hashes and the limits of this local evidence.

No end-to-end production latency or efficiency improvement is claimed. Context
selection and compact metadata avoid requiring all project knowledge in every
turn, but representative live token/call/latency measurements remain useful
before changing runtime language or pursuing further optimization.
