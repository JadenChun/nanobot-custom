# Customization inventory: original nanobot, main and migration

Date: 2026-10-22. Status: **source comparison with focused local checks**.
This inventories implemented behavior and its origin; it does not qualify a
production upgrade or authorize changes to the running marketing agent.

## Comparison method

Two different upstream baselines are needed. Comparing the migration only with
our old main branch would incorrectly count upstream's newer features as our
custom engineering.

| Snapshot | Frozen revision | Purpose |
| --- | --- | --- |
| Original upstream basis | [`HKUDS/nanobot@63d646f7`](https://github.com/HKUDS/nanobot/commit/63d646f731a505731e97909d114ce257c6de36bc), 2026-03-31 | Confirmed ancestor of our main branch; identifies the original fork additions |
| Our main branch | [`JadenChun/nanobot-custom@0976742`](https://github.com/JadenChun/nanobot-custom/commit/09767426d64ab4c346c94bf04be59684aa5d13ba) | GitHub main checked for this review; includes the Codex transient-retry fix |
| Imported upstream basis | [`HKUDS/nanobot@f47b8f2`](https://github.com/HKUDS/nanobot/commit/f47b8f23155ddec23cfb80ec001b544d495df89c), 2026-09-08 | Separates imported upstream work from migration customization |
| Our migration branch | `chore/upstream-migration@79934a56` | Committed candidate tree inspected locally |

The complete tree differences are 136 changed paths for original-to-main
(69 package paths, 53 test paths, 14 other paths), and 889 changed paths for
imported-upstream-to-migration (258 package paths, 480 test paths, 151 other
paths). Package paths include source and bundled assets. These are file counts,
not feature counts, complexity scores or performance measurements.

The pre-existing working changes in `nanobot/session/mode_storage.py` and
`tests/session/test_live_namespace_inventory.py` are separate from the committed
candidate comparison. Other uncommitted research documents are supporting
discussion, not proof of runtime compatibility.

In the tables, **present** means the implementation was found; **wired** means
its execution path was traced. Neither means live production qualification.
**Adapted** means a different implementation carries the capability, with any
remaining equivalence gap called out explicitly.

## What comes from nanobot

The original project already supplied the agent loop/runner foundation, message
bus, normal chat channels, provider abstractions, file/shell/web tools, MCP,
sessions, skills, basic memory, cron, background subagents and an HTTP API.
Our work extends these foundations; their basic existence is not a custom feature.

The later upstream snapshot additionally supplies much of the migration's
WebUI/TUI, packaged channel architecture, context governor, provider conversation
state, Responses/native compaction support, image-generation framework, document
readers, exec-session machinery, Dream and sustained-goal code. Our modifications
to their integration and safety boundaries are separate from those foundations.

Examples verified by blob comparison: `session/summary.py`,
`session/goal_state.py`, and the source modules for `session_messages`, `self`
and `apply_patch` are unchanged from the imported upstream snapshot. Their
presence does not establish registration: the migration uses an explicit
[core tool factory](../nanobot/agent/tools/fork_factories.py) rather than enabling
every imported module.

## Customizations on main and their migration disposition

| Customization | What main adds to the original basis | Migration state |
| --- | --- | --- |
| **Context repository packages** | `contextRepos`/`contextPaths`, `nanobot.context.json`, context entrypoints, modules, stores and repository summaries | Retained through a shared manager used by prompt construction, tools and foreground roles |
| **Repository and target-repository policies** | Writable/protected/proposal paths, script-owned stores, target repositories and path overrides | Retained/adapted; filesystem boundaries exist, but arbitrary shell commands do not provide equivalent containment |
| **Context Git synchronization** | Completion-path calls to the sync helper, selected sync paths and configured push behavior | **Integration missing:** the helper remains, but no package runtime caller invokes it |
| **External skills and bootstrap context** | Repository skill libraries and precedence, custom identity/tool guidance and supplemental repository memory/run files | Retained/adapted; ordinary `library` skills are currently omitted from the discovery summary |
| **Internal planning and verification** | Configurable planner decisions, internal model verification and bounded corrective work in the main loop | **Changed design:** optional explicit foreground planner/worker/reviewer/explorer roles replace that embedded workflow |
| **Background delegation safety** | Background spawn with optional review, session attribution, write-scope reservations and overlap rejection; foreground exploration | **Excluded from normal registration.** Scoped foreground delegation waits for completion. This is separate from the requested background-process/completion-continuation capability below |
| **Sequential pipeline surface** | A `spawn_pipeline` tool and workflow-state module | Main's actual manager method returns an unavailable error pending write-scope safety. The migration does not expose this tool |
| **Risky-action approval and file locks** | Tool-batch policy checks, contextual approval rules and coordinated writes | Retained and extended into explicit turn context, capability profiles, scoped writes and owned dispatch |
| **Context pressure handling** | Total-input/output budget configuration, older tool-result reduction, turn trimming and token-aware consolidation changes | Adapted onto upstream governance plus custom source/checkpoint and finalized-request safeguards; old trimming is not the intended canonical path |
| **Provider API-key pools and pacing** | Multiple keys, rotation, cooldowns, request ordering and requests-per-minute control inside the compatible provider | Retained/reorganized into shared `KeyPoolCoordinator` and `PooledProvider`; this capability already existed on main |
| **Provider fallback policy** | Configured alternate providers/models for quota and rate-limit conditions | Adapted onto upstream state-aware providers, with an explicit legacy policy adapter |
| **Provider/model compatibility** | Responses routing, Copilot/OpenCode Go behavior, reasoning/media normalization and vision fallback | Substantial newer upstream overlap; custom compatibility is integrated around that foundation rather than retaining main's provider implementation verbatim |
| **OAuth credential helpers** | Codex/Copilot import, usage reporting and authentication helpers | Adapted into coordinated credential storage/refresh. Old Copilot default-location discovery remains a compatibility gap |
| **Codex transient recovery** | Latest main fix marks transport and streamed server failures retryable, with a configurable retry delay | Structured retry support already exists in the candidate. Exact delay, environment and exhausted-retry parity still need comparison; do not classify all retry support as absent |
| **Image generation adapters** | OpenRouter generation plus hosted Codex/OAuth and Codex CLI paths with output-file handling | Adapted into upstream's `generate_image` framework with legacy configuration/backend support |
| **Telegram client access and fast commands** | Additional group allowlist, configured script shortcuts, command-mention normalization and quota/reset display | Ported into the packaged runtime; sender authorization remains required in addition to group access rules |
| **Client-facing output control** | `taskUpdateMode`, result-only delivery, terminal streaming behavior and explicit media/caption handling | Retained/adapted in channel and manager paths; not a claim of live channel qualification |
| **Cron delivery and acknowledgement** | Multiple destinations, deduplication and runtime delivery acknowledgements with explicit unknown outcomes | Reworked into occurrence/outbox/reconciliation components, but **normal gateway scheduling remains inactive** |
| **Marketing verification and owner alerts** | External verifiers, per-job planning/verification overrides, owner-alert destinations and post-delivery commands | **Missing contracts:** candidate payloads omit these fields and strict conversion rejects newer job payloads |
| **Five-hour Codex job rescheduling** | Primary-window reset plus buffer, bounded retry count, persistent retry state and successful-delivery reset | **Missing:** neither the job-level behavior nor its persisted fields is preserved by the current candidate |
| **Browser, device and desktop tools** | Browser/device wrappers, desktop input/screenshots, platform settings and humanized typing | Retained with stronger foreground role restrictions; availability still depends on configuration and platform dependencies |
| **VPN trend tools** | Bounded VPN/VPN Gate research tools and related deployment flags | Retained behind flags; presence does not imply use by the client's current approved workflows |
| **Custom practical skills** | Desktop use, mobile testing/dogfooding, webapp testing and video-understanding assets/scripts | Carried into the migration; separate from upstream's standard skill catalogue |
| **Marketing deployment and logging** | Service user/directories, gateway deployment/update scripts, fast-command environment and `--log-file` behavior | Retained/adapted; clean installation of enabled channel dependencies remains an operational qualification gap |

Representative main-branch evidence:
[planning, verification and sync](https://github.com/JadenChun/nanobot-custom/blob/09767426d64ab4c346c94bf04be59684aa5d13ba/nanobot/agent/loop.py),
[background spawn and rejected pipeline](https://github.com/JadenChun/nanobot-custom/blob/09767426d64ab4c346c94bf04be59684aa5d13ba/nanobot/agent/subagent.py#L389),
[key rotation and pacing](https://github.com/JadenChun/nanobot-custom/blob/09767426d64ab4c346c94bf04be59684aa5d13ba/nanobot/providers/openai_compat_provider.py),
[marketing cron payload](https://github.com/JadenChun/nanobot-custom/blob/09767426d64ab4c346c94bf04be59684aa5d13ba/nanobot/cron/types.py),
and [job-level quota recovery](https://github.com/JadenChun/nanobot-custom/commit/af1f33a9935bc8cdc77f7c527fda5b3b99fb7f84).

## Additional custom engineering present in the migration

These implementations are absent from inspected main and add to, or materially
adapt, the newer upstream foundation. Some came through the historical staging
line; this inventory does not claim they were first written during the merge.

| Area | Custom contribution | Qualification boundary |
| --- | --- | --- |
| **Canonical turn contract** | `TurnRequest`/`TurnResult`, typed run records, source/history modes, callbacks, sanitized opt-in traces and one execution boundary | Integrated foreground path; external callers and old state still require compatibility validation |
| **Foreground role system** | Explicit planner, worker, reviewer, explorer and crawler managers; immutable tool profiles, shared repository policy, call budgets and write scopes | Core foreground roles are wired; crawler is separately gated |
| **Session ownership and persistence** | Workspace/session namespaces, generations/revisions, execution owners, leases, canonical/legacy modes and conversion/export boundaries | Local implementation and tests; do not infer arbitrary live-data conversion safety |
| **Durable source and context checkpoints** | Source-bound transcript writers, projection provenance, checkpoint dependencies and guarded publication integrated with upstream governance | Custom durability layer; the governor and basic summary schema themselves are upstream foundations |
| **Private tool-result artifacts** | Scoped artifact storage, integrity checks, grants, read provenance, handoff and rematerialization | Preserves evidence outside prompt projections; not a reason to mix runtime storage into context Git repositories |
| **Legacy memory recovery** | Receipt-based preparation, checked conversion, transaction phases, quarantine and startup admission before legacy archival changes | Migration/recovery machinery rather than a new business-facing memory product |
| **Busy conversational control (H1)** | Durable inbox, bounded stateless control inference, steering/next-turn lanes, status/stop coordination and origin-aware reply intents | Explicit offline integration has local evidence; normal Telegram startup does not register the adapter. Whole-H1 and live cross-channel acceptance remain separate |
| **Provider execution safeguards** | Immutable runtime snapshots, shared key coordination, concrete-provider state boundaries, finalized-request budget checks and coordinated OAuth publication | Added integration/correctness work; not all underlying providers or Responses support are custom |
| **Durable scheduled-work substrate** | Definition/action/occurrence stores, scheduler ownership, delivery outbox and recovery/reconciliation identities | Implemented but dormant in the normal gateway; missing marketing contracts remain blockers |
| **Owned runtime lifecycle** | Two-phase composition/preflight, explicit acquisition, reverse cleanup, startup admission and restricted plugin/tool registration | Default foreground lifecycle is wired; it deliberately does not activate every imported autonomous subsystem |
| **Isolated crawler** | Foreground crawler role plus a non-LLM Crawl4AI worker, Unix socket, operation/domain limits and separate model budget | Not present on inspected main; disabled unless both configuration and environment gates permit it |
| **Owned subprocess support** | Shared process creation/cleanup used by custom integrations, alongside upstream exec sessions | A support module, not a separately registered user-facing `process` tool |
| **Idle-compaction contracts** | Additional activity/checkpoint ownership and cancellation safeguards | Dormant; normal composition rejects enabling idle compaction |

Migration source map:
[turn contract](../nanobot/agent/turn.py),
[foreground composition](../nanobot/agent/foreground.py),
[capability profiles](../nanobot/agent/capabilities.py),
[source store](../nanobot/session/source_store.py),
[checkpoint model](../nanobot/session/context_checkpoint.py),
[artifact store](../nanobot/session/artifact_store.py),
[memory recovery](../nanobot/agent/memory_recovery.py),
[busy control](../nanobot/agent/busy_control.py),
[runtime composition](../nanobot/runtime_composition.py), and
[cron runtime](../nanobot/cron/runtime.py).

## Corrections to broad feature claims

- A new filename does not imply a new capability: main already implemented key
  pools/pacing inside its compatible provider.
- A tool module does not imply a usable feature: main's pipeline manager refuses
  execution, and the migration's core registry excludes imported background
  spawn/goal tools.
- A missing main-branch constant does not imply missing retry semantics:
  candidate Codex failures use structured retry metadata and transient detection.
- The preserved sync utility does not establish automatic synchronization:
  main calls it after work; the candidate currently has no runtime caller.
- Config parsing and earlier foreground test passes do not establish marketing
  scheduling compatibility. The gateway explicitly reports that runtime inactive.
- Context repositories are already native runtime integration. The remaining
  design work is repository lifecycle management and knowledge/memory ownership,
  retrieval and write-back, not starting from a skill-only implementation.

## Background work and conversational responsiveness

Follow-up review on 2026-10-22: receiving messages during a task, controlling that
task, running a subprocess, and delegating an independent agent task are different
capabilities. None establishes the others.

**User clarification:** the requested behavior is a background process or job
followed by continuation of the same assistant when the work completes. A second
LLM session or background subagent is not required. The earlier review placed
too much emphasis on subagent delegation when describing this requirement.

| Capability | Current migration behavior |
| --- | --- |
| Receive input while the model or a tool is waiting | The asynchronous bus continues receiving. Same-session follow-ups enter the active turn's pending-input path rather than opening a competing conversation turn |
| Status, stop, steering and a queue for independent work | H1 provides an explicitly registered owning consumer and Telegram adapter. Offline tests exercise these paths; normal gateway composition does not register that adapter |
| Let a shell process continue after its tool call returns | Config-enabled `exec` supports `yield_time_ms`, returning a session ID; `exec_session` and `list_exec_sessions` manage it. This is process execution, not an independent LLM task or a durable completion-notification service |
| Automatically continue the assistant when that process completes | No completion-to-assistant-turn connection was found for `ExecSessionManager`. Output and exit state are collected through session polling. Local-trigger coordination exists separately, but it does not subscribe to exec-session completion |
| Delegate work to another LLM session | The normal registry exposes foreground roles that await their results. Imported background `spawn` code is excluded from core registration. Restoring it is not necessary for the clarified process/continuation requirement |

The bus can dispatch different session keys concurrently, subject to configured
capacity. The gap relevant to this request is the end-to-end process lifecycle:
start work, return control, retain its ownership and task context, and wake or
queue a continuation for the assistant when the work completes. A scheduled due
time and a process exit are different trigger sources; neither requires the
assistant to keep making model calls while waiting.

The [H1 contract](migration/h1-implementation-contract.md) explicitly chooses no
additional task-working agent and a FIFO next-turn lane for independent tasks.
Its tool-free control inference can propose a route while the task provider is
blocked, but it does not perform a second substantive user task. The clarified
background-process requirement can preserve that single-assistant ownership:
the initial turn ends or explicitly yields after launching the process; a
completion later enters through the owning session's continuation path. If the
assistant is busy, the completion waits for a safe boundary rather than starting
a competing transcript writer. Steering takes effect at a safe runner boundary;
accepting it does not interrupt an in-flight tool immediately. H1 remains
**NOT_ACCEPTED** as a whole.

Evidence: [bus admission](../nanobot/agent/loop.py),
[owning consumer](../nanobot/agent/busy_dispatch.py),
[foreground role execution](../nanobot/agent/delegation.py),
[explicit core registry](../nanobot/agent/tools/fork_factories.py),
[reserved plugin names](../nanobot/agent/tools/loader.py), and
[process sessions](../nanobot/agent/tools/exec_session.py). The separate
[local-trigger coordinator](../nanobot/triggers/local_turns.py) is an integration
building block, not an already connected process-completion service.
Only the registration definition and its internal call to the channel's
registration method occur in the package source; no normal startup caller was
found for `register_telegram_busy_inbox`.

The relevant reference is background tool execution and completion delivery.
[OpenCode's v2 tools documentation](https://dev.opencode.ai/v2/docs/tools/)
describes background shell calls that return immediately and notify the session
on completion.
[DeepSeek's background-job tools](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/jobs/tool-jobs/README.md)
provide job listing, output and cancellation, with completion delivery to busy
or idle owners. Those contracts are reference designs, not evidence that our
candidate implements them or that every completion race is solved upstream.

Recommendation for discussion: extend the existing process/job boundary with a
completion event and assistant continuation, reusing the canonical runner and
session owner. Retain job identity, original instructions or evidence references,
project path, originating session and reply destination independently of the
current chat topic. Provide status/output/cancellation and bounded output. Wake
an idle assistant or queue the event for its owner if busy; handle completion
once, including across restart or an uncertain outcome. Switching from project A
to B while a job runs must not make A's completion operate in B. Git, crawling,
VPN and marketing steps remain tool/skill behavior around that generic contract.

The acceptance example is an analytics script running while the user gets an
answer to another question, checks status, or requests cancellation. When the
script finishes, the assistant reads its output and continues the original task
in the correct project, delivering the result to its original destination.
There is no separate model worker running while the script is pending. This
records a proposed generic integration, not an implementation change or
scheduling activation.

## Validation and limits

This review compared all four committed trees in an isolated Git object store,
read the relevant callers, and checked eleven deterministic provenance facts,
including ancestry, inherited blob identity, main's rejected pipeline, key-pool
origin, missing completion-sync callers and the inactive scheduler message.

Fresh focused checks:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/providers/test_openai_codex_provider.py \
  tests/providers/test_key_pool_coordinator.py
```

Result: **33 passed**. These cover candidate Codex behavior and shared key
coordination; they are not a full main-to-candidate differential test or proof
that main's exact retry-delay policy is preserved. The earlier 57-test context,
capability and foreground selection is separate evidence from the preceding
review. No original-main suite, live service, complete migration suite or
performance benchmark was run for this inventory.

The background/responsiveness follow-up additionally ran:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/migration/test_h1_busy_control.py \
  tests/migration/test_h1_busy_dispatch.py \
  tests/agent/test_session_inputs.py \
  tests/agent/test_migration_foreground.py
```

Result: **89 passed**. A second selection covered
`tests/migration/test_h1_telegram_inbox.py`,
`tests/tools/test_exec_session_tools.py` and
`tests/agent/test_exec_session_isolation.py`: **78 passed**. The repository's
existing environment lacks the optional Telegram package, so initial collection
of that selection was blocked. It passed with the manifest's Telegram dependency
ranges installed into a disposable `/tmp` target supplied through `PYTHONPATH`;
the existing environment was not modified. The target resolved
`python-telegram-bot==22.8`. Evidence is retained under
`/tmp/nanobot-background-review-o2to0ffn/`, including the install log, command,
test output and matching before/after hashes for 543 runtime/selected-test files.
These **167 focused offline passes** do not qualify live startup, all of H1,
automatic process-completion continuation, or marketing scheduling.

The historical [158-path staging audit](migration/p0-compatibility-inventory.tsv)
remains useful but covers a different staging snapshot. It does not account for
all later main-branch marketing changes. The private marketing context repository
also owns business knowledge, schemas, scripts and outputs outside this runtime;
those must be versioned as a separate dependency of a working client workflow.

## What this means for the project

The main line contributes the client-operating behavior: context repositories,
marketing delivery rules, verified scheduled work, provider resilience and
deployment. The candidate adds our stronger harness and durable execution model
on a newer upstream foundation. Neither line alone is evidence of a complete,
production-compatible successor.

Keep [scheduled marketing-work compatibility](marketing-agent-scheduling-plan.md)
as the next required milestone. Track automatic repository sync, skill discovery,
credential-location compatibility and repository-management/memory design as
separate bounded follow-ups. This inventory does not expand that milestone or
activate any dormant feature. The background-process/continuation capability
above is an additional product/design follow-up; it does not require restoring
background subagents or changing H1 into a multi-agent system.

The subsequent discussion accepted a simpler direction: catalog-assisted,
agent-driven context discovery through existing tools, without assuming a new
semantic task-context module. Useful knowledge should be saved automatically;
automatic commit/push applies only to designated knowledge repositories. The
[direction and existing-capability assessment](harness-prototype-direction.md)
records those outcomes and distinguishes the configured repository index from
the still-unimplemented automatic catalog/save/sync workflow.
