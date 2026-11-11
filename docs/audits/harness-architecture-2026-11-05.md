# Staging harness architecture audit — 2026-11-05

Historical record: the first architecture pass, capability comparison, focused baseline
tests and isolated responsiveness reproduction below retain their original pin. They do
not certify feature parity, current stability or completion of the wider
[code-quality audit](../code-quality-audit-2026-11-04.md). Local evidence paths and copied
reference locations in this historical body are optional historical coordinates; current
source discovery does not depend on their availability.

Current continuation is the [project audit plan](project-audit-plan.md),
[current discovery map](codebase-discovery/README.md) and
[PLAN-009 completion record](codebase-discovery/completion-2026-11-06.md). At unchanged
`origin/staging` `27adced0aacd55d61791bb88a7240d7ad7bd5b75`, all 2,455 baseline paths are
assigned and all 919 source targets have module/relevant-body evidence: 224 current reads
plus 695 identical-hash carried reads, with zero mapped-only source. All 47 changed
implementation paths and 139 completion assignments are covered. Current origin diverges
from prior local repair/discovery history; findings below remain historical.

The active sequence is completed source discovery → source comparison and architecture
discussion → bounded audit/optimisation, with Sandbox first within the later audit. Source
comparison, new architecture decisions and detailed audit remain pending. Continued work
uses source inspection and document processing, without tests or runtime qualification.
The historical Decision heading and all following bytes are preserved.

## Decision

Retain nanobot's existing runner, canonical session ownership, provider snapshots, and
tool/channel extension seams. Improve the existing architecture incrementally. There is
no evidence here that replacing the Python harness with DSH's plugin framework, adding
a second agent loop, or introducing a general task framework would improve reliability.

The useful DSH reference is its explicit ownership and lifecycle behavior: one driver
per session, durable input identities, request/history consistency, scoped resources,
and testable teardown. Its optional continuable subagents and the separate SuperAgent
application offer concrete future capabilities that nanobot does not currently expose.

## Frozen sources and separation

| Item | Inspected state |
| --- | --- |
| Nanobot | Remote `origin/staging`, `b94b12527b014785c1454a04ef3b8138ee6070a5` |
| Nanobot source tree | `9e15246bb0710590abb3351321aec29688c16285` |
| Audit branch | `codex/staging-harness-audit` |
| Audit checkout | `/Users/jaden/.codex/worktrees/staging-harness-audit/nanobot` |
| Separate repair checkout | `/Users/jaden/Documents/github/nanobot`, branch `codex/feature-audit-repairs`, observed HEAD `364259daed2c11700d4064e7ca58743fd037b0b0` plus pending changes |
| DeepSeek Harness | Public HEAD `fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159`, downloaded for source inspection |
| DSH SuperAgent prototype | Public HEAD `8c32a9dc5d72c033816c6a8110b37f40cf94fba4`, downloaded for source inspection |

Remote staging was checked and fetched before creating this worktree. Its two newer
commits are the workspace-input exec fix and its merge. The local `staging` branch and
the repair branch were not moved. Pending repair edits were not copied into this audit.
The audit changes only audit documentation; no application code, tests, dependencies,
or running configuration were changed by this audit.

A Git worktree isolates tracked files and its index. It still shares Git objects and
refs; it also does not isolate home-directory state, ports, or environments automatically.
This audit launched no gateway or DSH runtime. Nanobot checks used the existing repair
checkout's Python environment without syncing it, and verified imports resolved to the
audit checkout. Test data and reproductions used temporary directories.

## What DSH establishes

DSH's pinned [README][dsh-readme] calls the project a **developer preview** and warns of
breaking changes. The [SuperAgent README][super-readme] calls that separate application
a research prototype, not an official DeepSeek product. They are architectural examples;
neither description establishes a production-stability guarantee.

This inspection updates the DSH core reference from `903a6dd9` in the earlier
[SuperAgent study](../superagent-research-and-session-direction.md). That document's
product direction still applies: stabilise the current assistant first, retain direct
access to real working sessions, and defer managed-session implementation. The prototype
snapshot is unchanged. The design essay's proposed/internal features are not counted as
shipped harness behavior.

| Capability | DSH source evidence | Nanobot staging evidence and result |
| --- | --- | --- |
| Shared model/tool loop | `ReactLoopAgent` owns one phase/driver; provider dispatch uses a prepared adapter call ([driver][dsh-driver], [LLM runtime][dsh-llm]) | [AgentLoop](../../nanobot/agent/loop.py) coordinates turns and [AgentRunner](../../nanobot/agent/runner.py) executes model/tool work. Reuse this split. Focused runner/error tests pass. |
| Input ordering and steering | [Inbox][dsh-inbox] projects durable `next-turn` and `next-step` input and rejects duplicate pending identities | Nanobot has per-session pending queues, runner injection, source activity admission, and [execution ownership](../../nanobot/session/execution_owner.py). Ownership/injection tests pass; full cross-channel crash/reconnect equivalence was not established. |
| Durable history and request consistency | [Architecture][dsh-architecture] requires model-visible input to be reconstructable from session events; request reconstruction has an owning test suite | Nanobot has canonical source events, [SourceTurnWriter](../../nanobot/session/turn_writer.py), and [SourceRequestLedger](../../nanobot/session/source_request.py). Keep these; this pass did not qualify exact reconstruction of every provider request/runtime prompt after restart. |
| Context and long sessions | Separate compaction and projection packages, with tool-pairing and runtime-context tests | [Context governance](../../nanobot/agent/context_governance.py), provider budgets, source checkpoints, and memory recovery already exist. Budget/recovery tests pass. Long-duration semantic continuity and cost remain unmeasured. |
| Background process continuation | [Local jobs][dsh-jobs] run in-process with bounded retained output and owner lifecycle; process restart ends those jobs | [ProcessCompletionRuntime](../../nanobot/process_completion.py) journals ownership/project/result and re-enters the same assistant; interrupted uncertain work becomes `unknown`. It is opt-in through `gateway.process_continuations`, default false. Real local-process and project-A/project-B tests pass. Journal blocking is finding A1. This does not promise process survival across restart. |
| Delegated work | [Subagent manager][dsh-subagents] supports cataloged local child sessions, follow-ups, cold resume, activation capacity, and cleanup; external children have different capabilities | Active `plan_task`, `delegate_task`, `review_work`, and `explore` use [ForegroundAgentManager](../../nanobot/agent/delegation.py). Child source streams record bounded work inside an outer turn; they are not independently addressable persistent task sessions. The legacy `SubagentManager` adapter is dormant (A3). |
| Managed ongoing tasks | The separate prototype owns Tickets, Assignments, subscriptions, worker sessions, reports, and restart reconciliation ([execution][super-execution], [executor][super-executor]) | Sustained goals, cron occurrences, and process records solve narrower problems. They do not provide task identity, multi-source follow-up routing, and an accessible persistent worker session. This is a future capability gap, not a regression or authorization to implement orchestration. |
| Extensions and resources | Plugin services, explicit disposal, and separate live versus durable events | Nanobot already has explicit foreground factories, staged entry-point construction, MCP lifecycle ownership, optional extensions, and typed events. Entry-point tests pass. Preserve these simpler seams rather than copying DSH's full plugin hierarchy. |
| Capacity | Session-driver ownership, tool fan-out, and subagent activation limits are separate controls | Keep per-session correctness and delegated/tool limits separate from provider pacing. Existing provider key-pool controls own backend limits. Local multi-request inference responsiveness is not established by either repository's source. |

## Findings

### A1 — P1 responsiveness defect: process journal contention blocks the event loop

The real async [exec-session start/exit paths](../../nanobot/agent/tools/exec_session.py)
call synchronous completion-sink methods (`started` near line 400, `finished` near line
470). [ProcessCompletionRuntime](../../nanobot/process_completion.py) opens SQLite
connections, uses `synchronous=FULL`, and reads/commits transactions inline in those
methods (lines 70–140). Its async continuation also calls these synchronous operations.

Trigger: with process continuations enabled, another connection holds the journal's
write lock when a start/completion is recorded. The event loop waits inside SQLite,
delaying unrelated streaming, input, and stop handling until the call returns.

An isolated real SQLite writer held `BEGIN IMMEDIATE` for 150 ms. Calling the actual
`started()` method from an asyncio task took **162.6 ms**; a concurrent 5 ms heartbeat
had a **163.6 ms** maximum gap. The reproduction used a private journal and launched no
subprocess or provider. This demonstrates blocking under contention, not its normal
frequency or uncontended overhead.

Proposed correction: give this runtime an ordered asynchronous persistence path, moving
whole transactions off the event-loop thread while keeping durable registration before
process launch. Preserve compare-and-swap transitions, `synchronous=FULL`, owner identity,
and cancellation/shutdown settlement. Independent `to_thread` calls without ordering
would change these guarantees and are not an adequate design by themselves.

Acceptance: the heartbeat continues while the isolated journal is locked; the existing
duplicate completion, project ownership, interruption, cancellation, and shutdown tests
still pass; no uncertain external work is automatically replayed. This is a proposed
bounded fix, not an implemented or accepted async-storage design.

### A2 — P1 architectural debt: execution and recovery still consume delivery details

Verified examples are [AgentLoop](../../nanobot/agent/loop.py) importing WebUI metadata
at line 194 and interpreting UI source/turn/installation fields during source execution
near lines 3254–3358; [session recovery](../../nanobot/session/recovery.py) importing
WebUI identity, listing, and transcript helpers; and
[TurnDelivery](../../nanobot/agent/turn_delivery.py) importing channel notification
metadata at line 22. These are real dependency directions, not merely similarly named
modules. They make changes to UI request identity and permissions cross core execution
and recovery code.

The repository's [design constraints](../../.agent/design.md) already assign WebUI wire
behavior to `WebuiTurnCoordinator` or an edge adapter. That coordinator's WebUI imports
are an intentional adapter exception; a blanket ban on all session-to-WebUI imports
would misclassify it. The dormant session-search tool's UI access dependency also does
not establish an active foreground defect.

Proposed correction: first move the smallest shared identity/metadata definitions to
their existing neutral turn/session owner, preserving wire names and compatibility
imports. Then let the WebUI adapter parse its request fields and pass explicit values
to execution/recovery. Keep source-generation and permission checks in their existing
owners. Do not relocate checks merely to make an import graph look cleaner.

Acceptance: shared CLI/API/cron paths have no new dependency on UI request processing;
stale-source, temporary/full-permission, steering, transcript, and recovery behavior is
preserved. Ratchet concrete forbidden imports with an explicit adapter exception after
the first extraction. This finding is structural debt, not a demonstrated permission
bypass.

### A3 — P2 simplification: a retired execution adapter still has default lifecycle cost

[AgentLoop](../../nanobot/agent/loop.py) constructs `SubagentManager` at line 705 and
wires its refresh, stop, idle-count, and shutdown paths. The adapter's tool factory calls
`ToolLoader.load(scope="subagent")`, while the current
[loader](../../nanobot/agent/tools/loader.py) deliberately accepts only `core`. The
passing `test_real_legacy_tool_factory_rejects_unsupported_scope` confirms that retired
path. Other lifecycle tests inject tools/runners and establish bookkeeping, not an
enabled background-delegation capability.

Active role delegation has a separate owner, `ForegroundAgentManager`, and shares the
generic runner. Treating these managers as two working alternatives would give a false
feature inventory. `SubagentManager` remains publicly exported, so absence of internal
spawn callers is insufficient authorization to delete its API.

Proposed correction: isolate or lazily construct the compatibility adapter after tracing
supported SDK callers, while retaining default stop/shutdown semantics and active role
delegation. If public removal is desired, make that a separate explicit compatibility
decision. No startup-speed gain is claimed: the measured loop import is already modest.

## Minimality and efficiency baseline

The read-only Python AST inventory covers **538 production Python files / 236,477
physical lines**, excluding paths containing a `tests` segment. Largest relevant modules
are `agent/loop.py` (5,133), `session/session_mode.py` (3,576), `session/manager.py`
(3,525), and `agent/runner.py` (2,496). The loop has 81 distinct static runtime dependency
targets including lazy imports; `_run_core` spans 735 physical lines.

These are investigation priorities, not complexity scores or deletion targets. Static
cycles include lazy imports and are not evidence of failing import-time initialization.
The previous size charter uses different counts; these measurements do not establish
that the codebase grew or shrank. This pass did not perform a whole-tree dead-code,
duplication, churn, coverage, or cognitive-complexity analysis.

Three fresh-interpreter imports of `nanobot.agent.loop` took 0.468, 0.347, and 0.371 s
(median **0.371 s**) and reached about 75 MiB peak RSS. Filesystem caches and other host
work were uncontrolled. This measures imports only, not gateway startup, idle runtime,
turn latency, model quality, or efficiency relative to DSH. A1 provides a specific
responsiveness result; there is no evidence for a general performance rewrite.

Keep schedules, occurrences, delivery attempts, session source, checkpoints, and memory
recovery receipts distinct until each owned invariant and failure mode is mapped. Several
stores or recovery files do not by themselves prove duplicate ownership. Preserve atomic
writes, source fencing, and no-repeat semantics while reducing incidental coupling.

The unbounded bus/pending queues and retained process records also need a measured
capacity policy before an always-running orchestration rollout. No overload/OOM failure
was reproduced. Adding `Queue(maxsize=...)` blindly could break accepted-input durability
or stop responsiveness; admission, acknowledgement, and retention need to be designed
together.

## Interaction with repairs already in progress

| Pending repair area observed | Audit relationship |
| --- | --- |
| `fork_factories.py`, `long_task.py`, runtime-control tests | Adds missing goal/self tool registrations and persistent-session guards. Record as an existing repair, not another architecture project or proof that the baseline goal tools are active. |
| Cron gateway/runtime/service/types and session automation adapters | Changes scheduling and outcome handling. Recheck cron execution, interruption recovery, and delivery ownership against the merged result before changing related architecture. |
| Gateway runtime, local trigger store/runner, config | Wires local triggers and their lifecycle. This overlaps the same background-input admission paths that a future task coordinator would need. |
| Generated assets, skills refresh, settings, documentation | Mostly separate from runner/session design; metadata, lifecycle events, and capability claims remain integration points. |
| Remote IPC and permission/exec integration tests | These appeared in the moving repair inventory during this audit. Recheck transport identity and permission behavior before changing those boundaries; they were not reviewed as part of this frozen baseline. |
| New staging exec workspace-input fix | Already included in the frozen audit source; any workspace/tool refactor must preserve it. |

The correct answer is **file isolation, with architectural reconciliation**. The current
repairs do not prevent this audit, and broad ownership recommendations remain useful.
They can change concrete findings, call paths, defaults, tests, and the safest patch.
Do not merge both lines of work independently and assume that a clean textual merge
establishes compatible behavior.

Before implementing an audit recommendation, record the then-current staging SHA,
compare touched owners and behavior with this baseline, mark findings already resolved
by repairs, and rerun the affected tests on the integrated source. Keep each architecture
change separate from unrelated repair cleanup.

## Sandbox scope qualification

This first pass did not establish sandbox stability. Its focused workspace and process
tests are not a real confinement or integrated permission-mode qualification suite.
The follow-up source mapping found that managed WebUI shell Sandbox requires Linux
with working Bubblewrap; native whole-process-tree supervision is also Linux-specific.
The separate feature-repair audit reports macOS native/sandbox failures that still need
classification against their exact source and environment.

Legacy `tools.exec.sandbox`, managed WebUI Sandbox, file-tool grants, native process
supervision and trusted MCP integrations have distinct contracts. In particular, the
legacy nonmanaged Windows fallback is not evidence of a managed-mode bypass. The
[current sandbox source review](project-audit-plan.md#sandbox-first-source-review)
examines the implementation of these boundaries, grants/Once, mode changes, children,
stop/restart, toolchain wiring and error paths. It does not establish operational
stability; the first pass did not qualify sandbox enforcement.

## Proposed work order and gates

These architecture-specific proposals record the original first-pass method. The
living project plan supersedes their qualification gates, prioritises sandbox source
review and reconciles these items with ongoing repairs.

1. Finish and reconcile the existing feature repairs. Preserve their focused regression
   evidence and refresh the capability inventory on the resulting staging commit.
2. Address A1 as a bounded journal-responsiveness change, with a failing nonblocking
   reproduction first and transaction/cancellation guarantees specified before edits.
3. Reduce A2 at one concrete dependency seam, then consider A3 independently. Prefer
   moving definitions/removing obsolete default wiring over adding framework layers.
4. Qualify the existing harness through restart, pending input, stop, tool failure,
   reconnect, project switching, compaction, and provider exhaustion scenarios. Use fake
   providers/local processes for deterministic gates, then separately authorized live
   provider/deployment checks. Type repairs must follow the frozen full-source before/
   after policy and `scripts/check_type_regressions.py`; this audit closes no P6 gate.
5. Only after the existing runtime is qualified, specify optional managed task sessions:
   durable task identity separate from execution attempts, one real session per worker,
   direct user access, explicit follow-up routing, typed results/attention, cancellation,
   restart reconciliation, and capacity at the actual backend. Reuse canonical turn
   ownership and existing session storage. A schema, routing authority, and inference
   capacity policy remain product/design decisions, not selected implementations here.

The target is an assistant with DSH-like useful outcomes and observable recovery, while
retaining nanobot's simpler Python extension model. Full feature parity is broader than
this audit and requires an agreed acceptance matrix and runtime evidence.

## Validation and retained evidence

Python 3.13.2 and pytest 9.1.1 from the existing `.venv` were used without dependency
changes. The focused command was `python -m pytest -q --durations=12 --basetemp <temp>`
over these complete modules:

```text
tests/session/test_execution_owner.py
tests/session/test_turn_continuation.py
tests/agent/test_runner_tool_execution.py
tests/agent/test_runner_errors.py
tests/agent/test_runner_injections.py
tests/agent/test_finalized_budget_recovery.py
tests/tools/test_process_completion.py
tests/agent/test_process_completion_scope.py
tests/agent/test_subagent_lifecycle.py
tests/agent/test_tool_loader_entrypoints.py
tests/agent/test_tool_loader_scopes.py
tests/agent/test_workspace_scope.py
tests/cron/test_interruption_recovery.py
tests/cron/test_interruption_recovery_correction.py
tests/agent/test_memory_recovery_target_snapshot.py
tests/agent/test_memory_recovery_receipt_finalizer.py
```

Result: **333 passed, 1 skipped, exit 0, 73.15 s**. The skipped special-permission mode
case was individually checked with `-rs`: the host filesystem does not preserve that
mode. It remains unqualified. These tests do not establish live-provider reliability,
all-channel parity, or enabled background-subagent behavior. No DSH tests/runtime,
full nanobot suite, strict checker, or frontend build were run for this report-only pass.

Retained local evidence is at
`/Users/jaden/.codex/audit-artifacts/nanobot-harness-2026-11-05/`:
`inventory.py`/`inventory.json` (source hashes, dependency versions, static imports),
`checks.py`/`focused-tests.json`/`focused-tests.log` (exact command and result),
`probe_journal_latency.py`/`journal-latency.json`, and
`import_profile.py`/`import-profile.json`/`import-profile-*.log`.
The manifest records the inspected references and selected test hashes. These are
local audit evidence outside the tracked delivery, not an application dependency.

[dsh-readme]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/README.md
[dsh-architecture]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/docs/architecture.md
[dsh-driver]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/core/agent-loop/src/agent.ts
[dsh-inbox]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/core/agent-loop/src/inbox.ts
[dsh-llm]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/llm/llm/src/index.ts
[dsh-jobs]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/jobs/jobs-local/README.md
[dsh-subagents]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/subagent/subagent/src/manager.ts
[super-readme]: https://github.com/deepseek-ai/dsh-superagent-prototype/blob/8c32a9dc5d72c033816c6a8110b37f40cf94fba4/README.md
[super-execution]: https://github.com/deepseek-ai/dsh-superagent-prototype/blob/8c32a9dc5d72c033816c6a8110b37f40cf94fba4/docs/003-execution.md
[super-executor]: https://github.com/deepseek-ai/dsh-superagent-prototype/blob/8c32a9dc5d72c033816c6a8110b37f40cf94fba4/src/executor/dsh.ts
