# Marketing agent scheduling compatibility plan

Original plan: 2026-10-22. Implementation update: **2026-10-23**.
The candidate now implements explicit gateway scheduling, job contracts,
checks/receipts/hooks, bounded quota recovery and copied-state import. The
starting plan below remains the acceptance contract. Current configuration,
local evidence and remaining production release checks are documented in the
[prototype integration guide](assistant-prototype-integration.md). Production
has not been activated or upgraded.

The next milestone is to make the migration candidate execute the existing
marketing agent's scheduled work reliably. Scheduled execution is required for
a marketing-agent upgrade. Loading its configuration or passing foreground
tests does not establish that compatibility.

## Scope and direction

Keep the existing marketing runtime serving the client while the candidate is
qualified separately. Use this repository as the prototype for our own harness;
upstream feature parity is not an acceptance criterion for this work.

Workspace organization, agent-managed context repositories, memory retrieval,
Git-sync redesign, Rust, and another upstream refresh are deferred. Existing
context paths, scripts, permissions, store adapters and post-delivery behavior
needed by a scheduled marketing job remain compatibility requirements.

This is a work plan, not a deployment instruction. No runtime, configuration,
production state, or activation guard is changed by publishing it.

## Current evidence

The inspected candidate is `chore/upstream-migration` at `79934a56`, with local
work in progress. GitHub comparison snapshots checked on 2026-10-22 were
[`main@0976742`](https://github.com/JadenChun/nanobot-custom/commit/09767426d64ab4c346c94bf04be59684aa5d13ba)
and
[`staging@af1f33a9`](https://github.com/JadenChun/nanobot-custom/commit/af1f33a9935bc8cdc77f7c527fda5b3b99fb7f84).
These identify source references; capture the actual deployed commit and any
local changes before implementation and rehearsal.

| Surface | Current candidate behavior | Work needed |
| --- | --- | --- |
| Gateway startup | Announces `Cron: definitions only; runtime inactive` | Own and start a qualified scheduler during normal gateway startup |
| Schedule service | Manages definitions; does not read or write the legacy `jobs.json` as its authority | Explicitly import existing jobs and state into the new stores |
| Scheduler/runtime | Implemented behind explicit offline injection; acquiring it alone does not start its timer | Define supported runtime inputs, activation and shutdown ownership |
| Marketing verification | Newer main-branch verifier, owner-alert and post-delivery contracts are missing | Preserve their behavior at the execution and delivery boundaries |
| Quota recovery | Five-hour Codex job rescheduling and persisted retry fields are missing | Preserve bounded retry scheduling and state across restarts |

Source references: [gateway startup](../nanobot/cli/gateway_runtime.py),
[runtime composition](../nanobot/runtime_composition.py),
[definition service](../nanobot/cron/service.py),
[cron runtime](../nanobot/cron/runtime.py), and
[legacy conversion validation](../nanobot/cron/action_store.py).
The [local runtime closeout](migration/local-runtime-closeout.md) qualifies
foreground behavior and explicitly leaves autonomous activation separate.

## Required outcome

The candidate must run the same accepted marketing workflows, with their
configured timing, evidence checks and destinations. Examples currently
documented are Meta analytics sync, trend research, content ideas and weekly
performance reviews; the deployed job inventory determines the final list.

Acceptance requires all of the following:

1. **Existing schedules survive.** Preserve job identity, enabled state,
   schedule, `Asia/Kuala_Lumpur` or other configured timezone, prompt, workspace,
   origin session, Telegram chat/topic, verification settings and alert routes.
   Map newer fields explicitly; never silently discard unsupported values.
2. **The gateway owns scheduled execution.** Start one scheduler after required
   resources are ready, expose its actual status, and stop and join its work on
   shutdown. Direct SDK/API and foreground-only use must not start timers.
3. **One canonical turn path executes the work.** Scheduled jobs use the same
   owned session, runner, context, tool policy and cancellation boundaries as
   interactive work. Coordinate busy sessions without duplicate turns or an
   unbounded wait behind automatic work. Do not restore a competing legacy
   execution engine.
4. **Client output passes the existing checks.** Preserve configured external
   verifiers against the actual outgoing result, concise result-only Telegram
   delivery, destination/thread metadata, delivery acknowledgements, and
   post-delivery bookkeeping. Verification failure blocks client output.
   Operational failures produce the appropriate owner alert.
5. **Retry behavior remains bounded.** Preserve the main-branch five-hour
   Codex recovery contract: retry after the primary reset plus 120 seconds,
   at most three consecutive reschedules, no exhausted-weekly-window retry,
   no reset more than 24 hours away, and clear the retry budget on successful
   delivery. Persist the retry state. Separately verify equivalent handling
   of the transient Codex transport fix now on `main`; provider retry and
   whole-job rescheduling must not multiply retries or repeat completed effects.
6. **Recovery preserves known outcomes.** Restart must not lose accepted work,
   rerun a confirmed completed occurrence, or resend a confirmed delivered
   result. Uncertain tool effects or delivery outcomes require reconciliation;
   a retry must not assume that nothing happened. Document and test overdue-job
   and catch-up behavior against the existing runtime before activation.

## Work sequence

| Step | Deliverable | Exit condition |
| --- | --- | --- |
| 1. Freeze the marketing contract | Redacted job/config inventory, deployed source identity, context-repo revision, required scripts and baseline behavior cases | Every active job has named inputs, outputs, verifiers, destinations, retry rules and required side effects |
| 2. Preserve job data and business contracts | Explicit field mapping, idempotent import on copied state, verifier/alert/post-delivery adapters | No required field is lost; repeated import is safe; unsupported records fail visibly before activation |
| 3. Integrate gateway scheduling | Supported runtime construction, scheduler timer, canonical dispatch, status and owned cleanup | A controlled due job runs through real gateway composition; disabled/foreground paths remain inactive |
| 4. Verify execution and recovery | Integration cases for client verification, delivery, quota recovery, busy sessions and crash boundaries | Success, refusal, failure, restart and ambiguous-outcome cases meet the acceptance criteria |
| 5. Rehearse and prepare release | Copied-state rehearsal, reviewed configuration changes, evidence and rollback procedure | Required workflows pass before proposing a production cutover |

Reuse the existing scheduler, occurrence, session and delivery components where
they satisfy these contracts. Do not merely pass the offline injection object
from the CLI or remove activation guards to make a timer run. First define and
validate the production ownership and recovery inputs that the offline gate
currently substitutes for.

## Validation and evidence

Start with controlled providers, local test destinations and representative
script fixtures. Exercise scheduling through the actual gateway composition,
not only by calling a job runner manually. Include:

- Due-time and timezone behavior; pause, edit and cancel; bounded overdue work.
- Existing job import, repeated import, unsupported payloads and retained retry
  state. Use copied configuration and stores, preserving the originals.
- Verified success, failed verifier, failed owner alert, failed bookkeeping,
  definite delivery failure and ambiguous delivery acknowledgement.
- A busy interactive session, stop during work, graceful shutdown and restart
  around occurrence claim, tool effects, verification and delivery.
- Codex primary-window recovery, weekly exhaustion, retry-budget exhaustion,
  unreadable usage and transient transport failures.

Extend the nearest coverage under `tests/cron/`,
`tests/cli/test_gateway_automation_boundary.py`, the relevant gateway lifecycle
tests and `tests/agent/test_migration_*.py`. The current gateway boundary test
deliberately proves inactivity; add positive scheduled-execution cases while
retaining the tests for paths that must remain inactive.

For later Python changes, follow `AGENTS.md` and the
[type-regression evidence policy](migration/p6-error-attribution.md): retain
before/after checker inputs under the same environment and introduce no new
diagnostic signatures. Run affected tests and the broader required checks.
Existing foreground results are historical evidence, not scheduling acceptance.

Report commands, source revisions/hashes, environment, actual results and
remaining limitations. Measure runtime overhead and provider/tool waiting
separately before claiming an efficiency improvement.

## Production acceptance and rollback

Rehearse all inventoried marketing workflows on isolated copied state with
test-only delivery. Any later provider-backed rehearsal must keep production
destinations and external mutations isolated; a manual successful reply alone
does not qualify the scheduled workflow.

Prepare a cutover with only one active scheduler for the client's jobs. Record
the cutover boundary, pending occurrences and delivery outcomes. Keep the old
code, original configuration and state backup available. Test rollback against
the candidate's state changes; simply restarting the old version must not
replay work already completed or sent by the candidate.

Call scheduling compatibility complete only when the normal gateway starts
scheduled execution, all required job contracts pass, copied-state recovery and
rollback are demonstrated, and the required checks pass. Production activation
remains a separate release decision. Then return to the deferred workspace,
context-repository and memory design discussion.

Related research: [migration branch review](migration-branch-review.md),
[customization review](customization-and-context-repo-review.md), and
[prototype direction](harness-prototype-direction.md). Their configuration
compatibility observations do not override the scheduling acceptance criteria
in this plan.
