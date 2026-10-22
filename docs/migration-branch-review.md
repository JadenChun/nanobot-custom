# Migration branch review: features, design, merge path

**Historical review; current status is consolidated in
[remaining work](nanobot-remaining-work.md) and the
[corrected feature inventory](customization-feature-inventory.md).** In
particular, candidate Codex transient retry support exists in a different
implementation; H1's Telegram seam is not registered by normal startup; and
imported background subagent code exists but is excluded from registration.
The original text below is retained for traceability, not as the current
implementation checklist or release procedure.

**Status:** static review of `chore/upstream-migration` @ `79934a56`, 2026-10-22.
Not a runtime qualification. Evidence is the source tree, the migration
checkpoints, and a config-compatibility test against a redacted copy of the live
production config.

## 1. What the branch is

540 Python files, 238,559 lines. It is an upstream import plus a rebuilt harness,
not a fork of production: the histories share no merge base, 1,827 paths exist
only here, and 50 exist only on `main`.

| Subsystem | Files | Lines | Role |
| --- | --- | --- | --- |
| `channels/` | 149 | 66,687 | Packaged channel runtimes (Telegram, Slack, …) |
| `agent/` | 113 | 58,693 | Turn loop, runner, policy, delegation, memory, context |
| `session/` | 46 | 23,533 | Session authority, modes, source/artifacts, recovery |
| `providers/` | 33 | 20,800 | Provider adapters and registry |
| `webui/` | 47 | 19,315 | Server-side WebUI protocol and transcript views |
| `cron/` | 28 | 16,039 | Definition store, occurrences, scheduler, delivery |
| `cli/` | 23 | 9,531 | Entrypoints and command surface |
| `utils/` | 25 | 5,671 | Git sync, budgets, prompting helpers |
| `context_repo/` | 3 | 839 | Manifest, policy, managed repositories |

The previous stages recorded 3,637 passing tests in the final batches and a
strict checker at zero (`docs/migration/local-runtime-closeout.md`). That claim is
theirs; this review did not re-run it.

## 2. Feature inventory

Status legend: **Active** = wired in the default foreground composition;
**Dormant** = implemented and tested but only reachable through explicit
injection; **Absent** = no implementation on this branch.

| Capability | Status | Evidence |
| --- | --- | --- |
| Turn engine (`AgentLoop` → `AgentRunner`, streaming, tool loop) | Active | `agent/loop.py`, `agent/runner.py` |
| Whole-batch policy preflight + terminal tool outcomes | Active | `agent/policy.py`, `runner._execute_tools` |
| Busy-input admission, steering, stop, recovery (H1) | Active | `agent/busy_*.py`, `session/busy_inbox.py`, `session/execution_owner.py` |
| Session authority: lease, generation, execution owner | Active | `session/runtime_authority.py`, `session/keys.py` |
| Session modes + legacy→canonical conversion | Active | `session/session_mode.py`, `session/mode_storage.py` |
| Source-bound transcripts (every model input traceable) | Active | `session/source.py`, `session/turn_writer.py`, `agent/turn.py` |
| Context governance and budget enforcement | Active | `agent/context_governance.py`, `agent/context_budget.py` |
| Memory archival + receipt recovery | Active | `agent/memory.py`, `agent/memory_recovery*.py` |
| Context repositories (manifest, policy, write guard) | Active | `context_repo/manager.py`, `context_repo/policy.py` |
| Foreground delegation (`plan_task`, `delegate_task`, `review_work`, `explore`) | Active | `agent/delegation.py`, `agent/tools/fork_factories.py` |
| Cron **definitions** (create/list/update, durable store) | Active | `cron/service.py`, `cron/definition_store.py` |
| Cron **scheduling and execution** | **Dormant** | `cron/scheduler.py`, `cron/runtime.py` — `CronRuntime` is only built when `cron_injection` is passed; no CLI path passes it |
| Context-repo auto-sync after a turn | **Absent** | no `_maybe_sync_context_repo`; `utils/git_sync.py` exists unused |
| Heartbeat service | **Absent** | no `nanobot/heartbeat/` |
| Dream / idle compaction / scheduled callbacks | Dormant | modules exist, preflight rejects enabling idle |
| Background subagents, `spawn_pipeline`, workflow state | **Absent by design** | replaced by foreground delegation |
| Codex transient-retry hotfix (live on production) | **Absent** | 0 references to `_CODEX_TRANSIENT_RETRY_AFTER` |
| Channels (Telegram incl. `fastCommands`/`allowChats`, Slack, …) | Active | `channels/*/runtime.py` |
| WebUI, API, Python SDK | Active | `webui/`, `api/`, `sdk/` |

## 3. How the harness is designed

Six decisions define this branch. They are coherent, and they are the reason it
is both safer and heavier than `main`.

1. **One authority per session.** A lease, an execution owner and a runtime
   authority decide who may mutate a session. The bus is transport, not state.
   This is what makes "two runners for one session" impossible rather than
   unlikely.
2. **Durable admission before acknowledgement.** Inbound work lands in a
   persistent inbox with an ingress sequence before any visible receipt; the
   in-memory queue is a wake-up cache. Capacity is checked before saying
   "accepted".
3. **Source-bound transcripts.** User rows, assistant rows and tool results are
   bound to a source writer and projected into the model transcript under an
   identity, so recovery can answer "did this reach the model?" without guessing.
4. **Whole-batch policy preflight.** Every tool call in a batch is authorised
   before any executes, and a terminal outcome stops the batch. No half-executed
   turns.
5. **Composition plus preflight, not ambient config.** `RuntimeComposition.define()`
   builds the graph; `runtime_preflight()` rejects unsupported state (a non-zero
   session TTL, idle compaction enabled, conflicting session headers) before
   anything starts.
6. **Dormant by default.** Anything with external effects — cron firing,
   artifacts, Dream, idle work — needs explicit injection. Absence of a feature is
   a decided state, not an oversight.

**Strengths.** Recovery is a first-class concept rather than a retry helper;
policy is enforceable at the boundary; every mutation has an owner; the migration
is auditable. For a personal assistant that must not duplicate side effects or
lose messages, this is the right shape.

**Costs.** 46 session modules and 28 cron modules is a lot of surface for one
assistant. Debugging requires holding the authority chain in your head, and the
tests are effectively the specification. A new contributor cannot read this in an
afternoon.

## 4. The holes that matter for us

### 4.1 Cron does not run

`cron/service.py` is explicitly definition-only: it never reads or writes
`jobs.json`, and "scheduling, execution, runtime state mutation, and delivery
remain fenced". The scheduler exists (`cron/scheduler.py`, `cron/runtime.py`) but
`CronRuntime` is constructed only when a `CronRuntimeInjection` is supplied, and
no CLI path supplies one. Tests do.

Consequence: **the marketing agent's four jobs would silently never fire.** This
is the single biggest gap between "validated harness" and "usable product".

### 4.2 Nothing migrates the existing cron store

`cron/migration.py` prepares *staging* records in memory for bound execution; it
does not import the deployed `workspace/cron/jobs.json`. A migration step has to
read the legacy file, map each job's payload/schedule/delivery, and write the new
definition store before the first run.

### 4.3 Auto-sync is missing

Production commits and pushes the context repo after a turn. The candidate has
`utils/git_sync.py` but no caller. We just enabled that path in production, so
adopting the candidate as-is would silently turn it off again.

### 4.4 Config format changes on first save

The loader accepts `maxTokens: {input, output}` and maps it to
`max_tokens: 20000` + `context_window_tokens: 200000`. A `model_dump` round-trip
writes the new shape, which the production branch does not expect. Any shared
config file is therefore a one-way conversion.

### 4.5 The Codex retry hotfix is not here

Production retries transient Codex stream failures; the candidate still raises
`RuntimeError("Codex response failed")`. Because the eventual merge replaces
`main`'s tree with this one, the hotfix must be ported **before** the merge or it
disappears.

## 5. Path to making this our base

**Gate 1 — port production behaviours (small, must precede any merge).**
1. Codex transient retry (lift from `main@0976742`).
2. Context-repo auto-sync call site.
3. Cron 5-hour rate-limit reschedule (needs cron execution first, see Gate 2).

**Gate 2 — activate cron (the real work).**
4. Wire `CronRuntime` into the foreground composition behind a config flag.
5. Import legacy `jobs.json` into the definition store, one-shot and reversible.
6. Run one real job in a sandbox copy and compare artifacts with production.

**Gate 3 — data migration on a copy.**
7. Copy `~/.nanobot` (sessions, memory, cron, workspace) to a scratch instance;
   start the candidate there; verify legacy session conversion, memory receipts
   and context-repo policy against real shapes.

Only when Gates 1–3 pass does the branch become "our nanobot" rather than "the
migration candidate".

## 6. Merging to main

Because the histories are unrelated, a normal merge produces conflict noise with
no information. The clean operation is a merge that **records both histories and
takes this tree**:

```bash
git tag prod-2026-10-22 0976742          # rollback anchor
git checkout -b adopt/harness 79934a56
git merge -s ours --allow-unrelated-histories main   # tree stays candidate
git checkout main
git merge --ff-only adopt/harness        # main now points at the merged history
```

Result: `main`'s content is the harness, `main`'s history still contains every
production commit, and `git log` shows the adoption explicitly. The deployment
stays on tag `prod-2026-10-22` until Gate 2/3 pass, then switches to `main`.

**What this deletes.** 50 paths that exist only on `main`, mostly intentional:
the 8 monolithic `channels/*.py` (replaced by packaged runtimes), their 12 tests,
`heartbeat/` (2 files), `pipeline.py` + `workflow.py`, bridge sources, and a few
deploy scripts. The only loss that is *not* intentional is the Codex hotfix,
which Gate 1 covers.

**Rollback.** `git checkout prod-2026-10-22` and redeploy, or re-point the service
at the old checkout. Nothing in the merge touches the running instance.

## 7. Verdict

Use this branch as the base. It is the only one with the harness properties we
actually want, it loads our production config, and its gaps are enumerable —
which is the important part. The branch is not ready to replace production today
for one reason: **scheduled work does not execute**. Fix cron activation, import
the job store, port the three small behaviours, run one job on a copy, and the
merge is a formality.
