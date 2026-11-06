# Project audit plan and coverage ledger

Updated: 2026-11-06. **Method: source inspection of implementation, architecture and
code quality. Current sequence: discovery → source comparison and architecture discussion
→ bounded audit/optimisation.** Whole-project scope and the reliable personal-agent goal
remain accepted. Nine historical subsystem maps, three current-origin differential maps
and three completion reports are published with exact read lists. All 2,455 baseline paths
are assigned; all 919 source targets have evidence: **224 current reads + 695 identical-hash
carried reads, zero mapped-only source**. All 47 changed implementation paths have current
evidence. **Module-level/relevant-body source discovery is complete (PLAN-009)**; source
comparison, architecture decisions and detailed audit remain pending. Sandbox keeps priority
within the later audit.

The current [discovery entry point](codebase-discovery/README.md),
[differential record](codebase-discovery/refresh-2026-11-06.md),
[completion record](codebase-discovery/completion-2026-11-06.md) and
[coverage summary](codebase-discovery/coverage-summary.json) distinguish current inspection
and byte-identical carried evidence. The [continuation contract](codebase-discovery/continuation-plan.md)
defines the next packages. Completion retains the bounded refresh checkpoint and the user's
accepted source-only method.

This is the living continuation plan for the [code-quality audit charter](../code-quality-audit-2026-11-04.md).
The [first harness architecture pass](harness-architecture-2026-11-05.md) retains its
historical findings and execution results. Those results do not define the current
method. This plan supersedes earlier test-based audit and qualification gates.

## Objective and working baseline

The agreed goal is a **reliable agentic loop for personal-agent use, implemented in a
clean, minimal, efficient and maintainable codebase**. Review how the actual code
supports that goal: its ownership, dependency direction, lifecycle, error handling,
state management, authority and resource use. Identify design defects, abandoned changes,
unnecessary complexity and concrete opportunities to simplify the implementation.

The audit uses source inspection and reasoning. Test execution, runtime qualification,
benchmarks, live-provider exercises and workflow acceptance testing are outside this
audit method. Passing tests and coverage percentages are not audit completion criteria.
Existing test files remain in the project inventory as support code; their fixtures,
helpers or obsolete imports may be relevant to source quality and caller tracing.

| Item | Planning baseline |
| --- | --- |
| Historical planning/first-pass snapshot | `b94b12527b014785c1454a04ef3b8138ee6070a5` |
| Historical source tree | `9e15246bb0710590abb3351321aec29688c16285` |
| Current discovery baseline | Fetched `origin/staging`, `27adced0aacd55d61791bb88a7240d7ad7bd5b75`; tree `d49aca97c9a34a2d17eae53d672fa591652457f1` |
| Previous discovery baseline | `4e2bf4d8d0b34f133c01591b67826ca956fed384`; tree `d5cc262acda5fb8051ffac9996ee9ab189c361cd`; retained historical nine-family map |
| Current discovery branch | `codex/staging-discovery-20261106`; historical `codex/staging-harness-audit` retains the prior discovery pin |
| Audit checkout | `/Users/jaden/.codex/worktrees/staging-harness-audit/nanobot` |
| Historical separate repair checkout | `/Users/jaden/Documents/github/nanobot`, `codex/feature-audit-repairs`; its repair merge is not assumed present in current origin |
| DSH source comparison | `fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159` |
| Codex source comparison | Public `openai/codex` release `0.166.0` (`rust-v0.166.0`), `6ff1e1ca3b30fc8b914720681fcf5fe62e95f856`; tree `03f3b46689faf274f126dc78508ed238564da8ba` |
| Separate SuperAgent prototype | `8c32a9dc5d72c033816c6a8110b37f40cf94fba4` |
| Portable discovery evidence | [Reports/index](codebase-discovery/reports/README.md), [baseline](codebase-discovery/baseline.json), [methodology](codebase-discovery/methodology.md); historical local artifacts are optional |

The user authorised codebase discovery and then the current-origin sync/refresh. Current
origin diverges from the prior discovery at `b94b12527b014785c1454a04ef3b8138ee6070a5`:
seven prior local-only commits are absent and eleven remote-only commits are present.
The source map treats both omissions and additions as changes. Historical repair claims
and execution results retain their original pin; they are not automatically current.

The refresh accounts for 95 changed current paths, four additions within those 95, and
16 deleted prior paths. All 47 changed current implementation paths have relevant-section
source evidence. Use one pinned commit/tree for each later package and recheck affected
callers when the baseline advances. Preserve valid earlier evidence with its original
revision and inherit inspection only when the SHA256 matches.

## Architecture reference and decision policy

Architecture reference revision: **`AR-2026-11-06.2`**. This metadata revision records complete
module/relevant-body source discovery at the unchanged baseline and the next comparison/discussion stage. `AR-2026-11-05.2`
established the source-only method and remains historical. No new product or architecture
choice is accepted here. Each later package uses the same working direction and unchanged
pinned DSH/Codex implementation references below.

### Goal and established direction

The accepted broad goal is a dependable personal agent. The working product description
is an assistant that carries work across ordinary projects, files, tools and services,
preserves useful context, supports background continuation and keeps the user in control.
Specific platform, task-session, compatibility and resource-policy choices remain discussion
topics under that goal. No acceptance-test workflows are required for source comparison
or architecture discussion.

Existing inputs are the repository's [design constraints](../../.agent/design.md),
[runtime source map](../architecture.md), [personal-assistant direction](../harness-prototype-direction.md),
[session lifecycle contracts](../session-recovery-architecture.md), and
[future managed-session direction](../superagent-research-and-session-direction.md).
Compare these documents with the pinned implementation. Record discrepancies and their
consequences; do not assume either old documentation or current code is automatically correct.

The working recommendation is to retain `AgentLoop` for coordination, `AgentRunner`
for model/tool iteration, and canonical session/source ownership. Keep optional and
business capabilities at tool, channel, skill and MCP boundaries. The first pass supports
incremental correction; source evidence of a design failure may change that recommendation.
Use DSH and Codex to examine concrete mechanisms rather than copy their framework size
or implementation language.

### Architecture properties to inspect

| ID | Intended responsibility / invariant | Source evidence to examine |
| --- | --- | --- |
| AR1 | Small generic core with explicit adapters at the edges | Dependency direction, actual imports/callers, typed boundaries, composition and justified adapter exceptions; avoid competing loops or owners |
| AR2 | One execution owner per session/source with explicit ordering and steering | Admission call chains, identity checks, queues, locks, cancellation and release paths; REF1 |
| AR3 | History, requests, checkpoints and recovery describe the same accepted work | All writers, commit ordering, schemas, reconstruction and compaction paths, tool pairing and uncertain-effect handling; REF2/REF3 |
| AR4 | Tool authority is enforced for each attempt by runtime code | Path canonicalisation, grants/Once/source identity, retries, backend wrapping, native boundaries and unavailable/denied branches; REF4 |
| AR5 | Tasks, processes, connections and retained data have explicit ownership and lifetime | Creation, cancellation, teardown, shutdown, buffers, retention, blocking calls and lock scope; REF3/REF5 |
| AR6 | Background inputs and completions use canonical admission and original authority/routing | Cron/goal/completion callers, duplicate handling, project/session identity, recovery and effect boundaries |
| AR7 | Long-lived context has coherent ownership and bounded growth | Context assembly, compaction, memory writes, folder instructions, knowledge policy and private-state separation; REF2 |
| AR8 | Providers, tools and channels have clear capability and lifecycle contracts | Registration/defaults, enabled/retired paths, optional dependencies, cleanup, public APIs and compatibility obligations; REF4/REF6 |
| AR9 | Simplicity and efficiency follow from the implementation | Duplicate owners and abstractions, coupling, algorithmic work, repeated I/O, buffering, scans and avoidable allocations; distinguish observed code from expected benefit |

Use adapters for real wire/SDK differences, explicit state transitions for lifecycle
rules and scoped ownership for async resources. Each pattern must protect a named
invariant or reduce demonstrated complexity. Prefer existing narrow seams; small
channel/provider duplication may be simpler than a coupled shared abstraction.

### Design defects, obsolete work and optimisation

Connect each recommendation as **required outcome → nanobot implementation → applicable
DSH/Codex mechanism → defect or justified difference → smallest proposed correction**.
Include concrete source locations and the conditions that make the issue relevant.
A difference from a reference is not automatically a defect. Consider supported scope,
configuration, language/runtime and ownership before choosing a design.

Before deleting deprecated or abandoned code, trace registration, config readers,
public/SDK exports, callers, persisted records and recovery paths. Separate unused
internals from compatibility obligations. Explain which concepts, writers, dependencies
or repeated work a cleanup removes. A shorter file or a pattern name alone is not an
improvement. State efficiency benefits as source-based expectations; do not claim a
measured speedup, memory reduction or operational stability from source inspection.

### Future direction and discussions

Stabilise the existing assistant first. The longer-term direction is one assistant
managing directly accessible task sessions while preserving ordinary folders, context
and background continuation. Task/session schemas, routing and scheduling remain deferred.
Independent model sessions and waiting for a process are different capabilities; REF6
helps compare their implementations without selecting a new framework.

| Decision area | Current position | Source analysis and discussion needed |
| --- | --- | --- |
| Cross-platform managed Sandbox | Historical managed shell requires Linux/Bubblewrap; DSH and Codex use platform-specific mechanisms | Trace backend selection, callers, policy and error paths; discuss desired native/executor platforms and backend tradeoffs before adding support |
| Managed task sessions | Future direction exists; implementation deferred | Compare task/attempt/session identity, follow-up authority, direct access, reconciliation and provider-capacity assumptions |
| Compatibility retirement | Dormant code may retain public or persisted obligations | Establish callers/state obligations and discuss changed user behavior or migrations before removal |
| Resource policy | No new budgets selected | Identify unbounded structures, backpressure and capacity assumptions; discuss policy changes that affect capability or user experience |

Resolve routine findings when the contract and code are clear. Bring consequential
unsettled choices here with the source evidence, viable options, tradeoffs and a reasoned
recommendation before implementing them. Record the answer and update affected lane
contracts. Do not ask the user to reconfirm accepted scope or every bounded cleanup.

## Coverage register

The current [path inventory](codebase-discovery/path-inventory.tsv) assigns **all 2,455
tracked paths once**, with zero unassigned or duplicate paths. It includes implementation,
runtime instructions, tests, clients, assets, deployment/support files and documentation.
The historical 2,451-path planning snapshot and 2,467-path previous discovery are retained
in their original evidence roots; neither is the current denominator.

There are **919 source targets**: 878 implementation/tooling files and 41 runtime instruction
files. Exactly **224** have current relevant-section inspection and **695** carry prior
inspection with identical SHA256; **zero source targets remain mapped-only**. Current
inspection covers **47/47 changed implementation paths**. Across all roles the depth totals
are 232 current, 808 carried and 1,415 mapped-only support paths. Roles outside the source denominator comprise 997 test-support
files, 54 assets/locks, 290 configuration/support artifacts and 195 generic documents.

These are assignment and discovery-depth counts, not completed semantic audit. The
[coverage summary](codebase-discovery/coverage-summary.json) lists the exact 139 completed
source paths, each deleted prior row, per-family/per-stream counts and portable provenance.
Current inspection comes from exact refresh lists and completion assigned/inspected lists;
incidental completion context paths add no coverage. Static indexes remain navigation.
Shared reads do not duplicate primary assignments, and P01 spans every stream.

In the table, **C / P / M** are source-target counts for current relevant-section inspection,
prior inspection carried by identical hash, and mapped-only detail respectively.

| ID | Primary ownership | Current paths | Source C / P / M | Later bounded source-audit slice |
| --- | --- | ---: | ---: | --- |
| C00 | Entrypoints, CLI, command router, config, gateway composition and shared utilities | 165 | 49 / 21 / 0 | Config → active wiring; SDK/CLI startup/shutdown; defaults, flags and migrations |
| S01 | Filesystem tools, folder grants, workspace/private-state policy and security | 31 | 2 / 13 / 0 | Canonical paths, read/write authority, private stores, stale/Once approvals and actual grant readers |
| S02 | Shell, sandbox backends, native admission, exec sessions and supervision | 15 | 2 / 4 / 0 | Platform/backend selection, policy/error paths, process ownership and cleanup |
| H01 | Agent loop, runner, bus, hooks, turn admission and delivery | 167 | 16 / 13 / 0 | Source/turn owners, ordering/steering, tool batches, recovery, cancellation and adapter seams |
| H02 | Sessions, source events, ledgers, history, memory, compaction and recovery | 228 | 35 / 39 / 0 | Authoritative writers, request reconstruction, transaction phases, reset/delete and receipts |
| H03 | Providers, models, budgets, key pools, usage and accounting | 141 | 0 / 45 / 0 | Logical request identity, retries/fallbacks, stream/tool pairing, credentials and accounting |
| T01 | Tool construction, plugins, MCP, roles, delegation and compatibility adapters | 109 | 6 / 38 / 0 | Registration/defaults, lifecycle/trust, child ownership, legacy callers and public exports |
| A01 | Cron, local triggers, process completions and retained goals | 103 | 16 / 31 / 0 | Definition/occurrence/run/completion/receipt owners, activation, interrupted work and duplicates |
| K01 | Context repositories, knowledge, runtime skills, prompts and templates | 62 | 45 / 11 / 0 | Policy, Git sync, instruction loading, bounded context and trust |
| E01 | Channel framework and platform adapters, excluding WebSocket | 321 | 1 / 94 / 0 | Per-adapter auth, inbound/reply/media mapping, retries, reconnect and teardown |
| E02 | HTTP API, SDK, WebSocket, backend WebUI and shared client contracts | 144 | 13 / 55 / 0 | Auth/source/scope identity, transcript projection, reconnect/cancel and mutation schemas |
| U01 | React WebUI and browser assets | 328 | 28 / 175 / 0 | Canonical/optimistic state, queueing, permissions, notification ownership, assets and settings |
| M01 | Attachments, audio/images and media ingress/adoption | 19 | 0 / 14 / 0 | Media identity/lifetime, paths, limits, conversion, delivery and cleanup |
| R01 | Remote IPC/auth/pairing and Android/native clients | 189 | 9 / 61 / 0 | Peer/device identity, revocation, native transport/custody and keepalive lifetime |
| D01 | Python desktop adapter, Tauri shell and terminal UI | 101 | 1 / 46 / 0 | Process/port/profile owners, packaging assumptions, lifecycle and native/client state |
| X01 | Optional apps, crawler and extensions | 15 | 1 / 10 / 0 | Registration/callers, resource handling, isolation and optional dependencies |
| Q01 | Builds/dependencies/CI/deployment, migration tools, stubs, support and docs | 317 | 0 / 25 / 0 | Artifact/dependency identity, installers, tooling implementation and documentation consistency |
| P01 | Efficiency across all workstreams | Shared | Shared | Blocking async work, lock scope, repeated I/O/scans, algorithmic cost and buffer/queue/cache growth |

The nine discovery families group these eighteen streams in the
[discovery entry point](codebase-discovery/README.md#nine-subsystem-families). Completion
closed C00 (43), H01 (14), H02 (35), K01 (45), R01 (1: Android Gradle wrapper) and D01
(1: desktop CSS), with one responsibility row per assigned path. No mapped-only source
remains. Current/carried inspection does not imply every semantic branch was read or that
the detailed source audit is complete.

Each E01 adapter retains a separate source boundary: `dingtalk`, `discord`, `email`,
`feishu`, `matrix`, `mattermost`, `mochat`, `msteams`, `napcat`, `qq`, `signal`, `slack`,
`telegram`, `wecom`, `weixin`, `whatsapp`. WebSocket is the seventeenth channel package,
owned by E02. H03 tracks Anthropic, OpenAI-compatible, Responses, Azure, Bedrock, Copilot,
Codex and xAI plus pooling/fallback/auth. M01 cross-links image/transcription providers;
R01/D01 retain each shipped native target. Shared-base reading does not cover an adapter.

## Current activation checkpoint

Current source [reconciliation](codebase-discovery/refresh-2026-11-06.md#connected-current-behavior)
supersedes earlier local-only activation claims. Tool loading uses the explicit manifest
plus admitted entry points; `create_goal`, `update_goal` and `my` are absent from the core
manifest. Retained goal handlers/state and router registration do not override canonical
source command restrictions (`/goal`, `/trigger`, `/dream` are unavailable there).

Cron remains gateway-owned/opt-in and otherwise definitions-only. Scheduled fresh-turn
computation and durable receipt delivery are separate owners. Manual `run_job`/HTTP run
admission are fenced; per-job WebUI serialization reads compatibility state without a live
joined execution projection. Configured Dream/Heartbeat settings and schedule language
coexist with inactive/rejected runtime controllers. Local enqueue persists records, but
the gateway local-trigger field, store injection and production filesystem-queue consumer
are absent. Independently opt-in process completion submits directly through local-turn
coordination; its done state is not a delivery receipt.

Ordinary WebUI messages now reject client scope/source-token/Full-confirmation controls;
server workspace/session state and internal source fencing govern admission. Explicit
scope/new-chat/Full permission mutations remain separate. Resource policy and private
owner-issued authority still enforce access; Once and Full both require the currently
Linux-only native whole-process supervisor. Transcript hydration may synthesize display
completion from session history, distinct from canonical terminal proof or persisted display
completion. Browser notification ownership moved above thread views; Android keepalive is
nonsticky/process-local, has no wake lock or push path, and does not survive process loss.

Current IPC requires `SO_PEERCRED`; the old Darwin branch is absent. Asset continuation,
skills credential/error state and mixed clipboard text handling also differ from the old
local repairs. See the differential record for exact callers, source anchors, documentation
discrepancies and unknowns. Deployment configuration, plugins, platform behavior and actual
transport effects remain unobserved; source differences alone are not regression grades.

## Source audit method and batch evidence

A batch covers one bounded responsibility or invariant and its actual callers. Trace:

1. Entrypoints, composition, registration, defaults, enabled branches and public exports.
2. Call paths, dependency direction, cohesion, typed contracts, competing owners and
   duplicated logic across the boundary.
3. State readers/writers, identity, lock/transaction ordering, persistence, compaction,
   recovery and uncertain external effects.
4. Error propagation, retries, async/blocking work, cancellation, cleanup and resource
   lifetimes, including early returns and partial failures.
5. Authority checks, path handling, trust transitions and platform/backend branches.
6. Dead or deprecated code, unnecessary abstractions, compatibility obligations and
   source-visible efficiency opportunities.
7. The relevant pinned DSH/Codex implementation mechanisms and the reasons to preserve,
   simplify or change nanobot's design.

Each batch checkpoint records its ID, pinned commit/tree, reviewed paths/symbols and
callers, AR/REF references, owner/writer map, findings, unresolved questions and the exact
next slice. Mark shared dependencies and any code that was mapped but not yet examined.
Use source hashes for dirty inputs if needed; do not run application code to create
audit evidence.

A finding needs **path:line or symbol, call chain and relevant condition, implementation
reasoning, impact, smallest proposed correction, dependencies and uncertainty**. Identify
the governing product/design requirement. Record whether a decision needs discussion.
Do not infer broken behavior merely from a long file, missing test or unfamiliar pattern.

| Classification | Meaning |
| --- | --- |
| Source-established defect | A concrete code path contradicts an identified contract; show the condition and causal chain |
| Structural/design debt | Ownership, coupling or lifecycle design creates an explained maintenance/correctness risk |
| Cleanup opportunity | Proven unused work, duplication or needless abstraction can be removed while preserving obligations |
| Capability difference | A reference supports something nanobot does not; a product choice determines whether to add it |
| Open question | Source or agreed requirements do not settle the conclusion; state exactly what is unknown |

Use the charter's P0/P1/P2 severity and Keep/Simplify/Merge/Delete/Split/Rewrite
dispositions where applicable. Separate priority, certainty and completion status.

| Review state | Meaning |
| --- | --- |
| Mapped | Paths, entrypoints, activation and candidate owners identified |
| Reviewed | Named implementation and callers examined, with a retained review boundary |
| Reconciled | Related findings, reference comparisons and shared-contract disagreements resolved or explicitly recorded |
| Recommendation ready | Smallest correction, compatibility obligations, dependencies and discussion status are concrete |

A source review may finish with defects or deferred choices in its backlog. These states
do not mean repairs have been implemented. A question that source cannot answer remains
explicit; it does not automatically create a runtime/test phase or block completion of
the source audit.

## Sequence and source-review completion

The accepted sequence is discovery, then implementation comparison and architecture
discussion, then bounded audit/optimisation. Current path assignment, changed-source refresh
and all 139 remaining source-detail reads are complete at module/relevant-body depth.
[PLAN-009](codebase-discovery/completion-2026-11-06.md) records that boundary; comparison,
architecture choices and detailed audit remain pending.
Sandbox retains priority in the audit stage. P01 and documentation consistency accompany
every package. Prioritise concrete security, data-loss and ownership defects over cosmetic
cleanup within the agreed stage and scope.

| Stage | Work | Completion boundary |
| --- | --- | --- |
| 0 — Codebase discovery — complete | Current inventory/activation/owner map and fifteen published reports across all nine families | 2,455 paths assigned once; 919 source targets with 224 current/695 carried/0 mapped-only; 139 closures and unknowns retained; no full semantic-audit assertion |
| 1 — Source comparison and architecture discussion | Pinned DSH/Codex implementation mechanisms against current nanobot owners/callers and AR/REF properties | Differences, governing requirements, justified scope choices and consequential decisions recorded; no automatic architecture adoption |
| 2 — Sandbox and authority audit | S01/S02 with E02/U01/R01 callers; source slices below | Backend, authority/lifetime/error paths reviewed; defects, platform limits and choices recorded |
| 3 — Harness and durable state audit | H01/H02/H03 and C00 composition | Session/source owners, requests, writers, provider boundaries and failure branches reviewed |
| 4 — Background and capability audit | A01/T01/K01/M01 | Activation/admission, resources, compatibility and deprecated paths reviewed |
| 5 — Delivery/client audit | E01/E02/U01/R01/D01/X01 | Per-adapter/client implementations and shared contracts reconciled |
| 6 — Cross-project synthesis | Q01/P01 and shared boundaries | Review dispositions accounted for; findings deduplicated; bounded corrections and dependencies reviewable |

### Continuation work packages

These packages define ownership and dependencies for subsequent bounded instructions.
They do not assign product code or turn a discovery map into audit acceptance.

| Package | Ownership / inputs | Dependency and required output | Deterministic validation |
| --- | --- | --- | --- |
| DISC-DETAIL — complete | Three source readers; one publication owner; exact C00/H01/H02/K01/R01/D01 assignments | PLAN-009 closes all 139 paths at the unchanged pin; each has responsibility/caller/activation/state-resource evidence; no product changes | Exact disjoint assigned/inspected union, 139 rows, 137 primary anchors plus two empty markers, hashes and final depth counts; source-only |
| REF-COMPARE | Bounded comparison owners use current nanobot map and unchanged DSH/Codex pins; coordinator owns shared conclusions | Discovery boundary established; trace REF1–REF6 mechanisms, supported activation, writers, authority and lifetime; record justified differences and open contracts | Pinned source symbols/callers and requirement-to-mechanism mapping; no runtime/stability inference |
| ARCH-DISCUSS | Coordinator owns accepted decisions and canonical ledger; source owners provide bounded evidence | Comparison inputs ready; discuss platform/session/compatibility/resource choices that affect observable behavior, schemas or authority before a repair contract | Decision record separates accepted choices, recommendations and unresolved gates |
| AUDIT-BOUNDED | One owner per invariant; first priority S02-01 with S01/E02/U01/R01 callers | Discovery/comparison/discussion prerequisites met; review concrete source contracts, then propose smallest correction with exact write ownership | Source evidence and caller/state/resource map satisfy AR/REF criteria; later code repairs use separate repository validation obligations |

A future code repair receives objective/non-goals, accepted decisions, exact owned paths,
public/persistence compatibility, acceptance criteria, focused validation and rollout/rollback
appropriate to that change. No new schema, migration, platform promise or resource policy
is selected by this plan. Source synthesis remains distinct from release qualification,
benchmarks or a test gate.

<a id="sandbox-first-source-review"></a>

## Sandbox: priority within the later audit

The sandbox remains in scope. Review its implementation and code quality, including
policy construction, enforcement delegation, platform branches and lifecycle ownership.
Recent reported failures are investigation leads; trace their relevant source paths in
a later bounded audit after discovery, comparison and architecture discussion. The current
source baseline is already pinned. Do not initiate reproductions under this method.

[Sandbox and folders](../sandbox-folders.md) distinguishes managed WebUI Sandbox from
legacy CLI/channel restrictions and `tools.exec.sandbox`. Historical source describes
ordinary OS-account reads with private-state exclusions and a write boundary, not general
network isolation. Configured MCP servers have a separate trust boundary.

| Environment/path | Historical source observation to recheck | Implementation questions |
| --- | --- | --- |
| Managed shell on Linux/Bubblewrap | Availability probe, read-only root, approved write overlays, private masks and own PID namespace | Is availability checked before launch? Are policy inputs canonical and current? Do wrapping and errors preserve the authority contract? |
| Linux with missing/unusable Bubblewrap | Managed confinement reports unavailable | Trace missing binary, denied namespace and runner errors; inspect whether any branch launches without required confinement |
| Managed shell on macOS/Windows | `managed_bwrap_available()` rejects non-Linux at this snapshot | Trace callers, fallback behavior and truthful UI/API errors; native support remains a design choice |
| Legacy nonmanaged Windows sandbox setting | Logs unsandboxed execution and proceeds | Establish its distinct contract and reachable callers; do not conflate it with managed-mode enforcement |
| Native process ownership | `native_supervisor_available()` requires Linux POSIX for whole-tree supervision | Inspect child identity, ownership receipts, cancellation, retirement and other-platform branches |
| Containers, desktop and remote/mobile routes | Executor host and runtime policy determine the backend | Trace host/client identity, mount/runtime assumptions and propagation of grants/mode |

| Batch | Bounded source slice | Required review output |
| --- | --- | --- |
| S02-01 — Availability and errors | Config/entrypoint → platform/backend choice → admission → command construction → API/UI error handling | Call map separating unavailable backend, policy denial, runner failure, command error and supervision limits; recent issue leads matched where possible |
| S01-01 — File authority/private state | File-tool and shell grant readers, canonical/symlink handling, private masks and approved input snapshots | One authority/writer map; inconsistent or stale policy paths identified; preserve the staging workspace-input correction |
| S01-02 — Approval authority | Grant persistence/revocation, Once scope/consumption, source/revision/session identity and retry paths | Explain how authority changes reach each attempt; identify stale, duplicate or inherited grants in code |
| S02-02 — Process lifetime | Foreground/background spawn, descendants, output retention, stop/timeout/disconnect/shutdown, recovery and mode changes | Lifecycle/ownership map and resource/effect boundaries; distinguish launch authority from later mode state |
| S02-03 — Toolchain and integration wiring | PATH/caches/temp, DNS inputs, context repos, local/private-URL checks and MCP authority | Configuration and dependency assumptions, incompatible branches, duplication and unclear trust boundaries |
| S02-04 — Cross-boundary reconciliation | S01/S02 findings against gateway, WebUI and remote callers | Reconciled identities, authority, errors and lifecycle contracts; smallest proposed fixes and unresolved platform choices |

Begin the later audit with S02-01 after the discovery/comparison/discussion sequence.
A source review can establish a policy construction bug or missing backend; it cannot
certify OS enforcement or operational parity with DSH.
Keep that limit in the conclusion rather than adding native-execution gates to this audit.

## DSH and Codex implementation comparison

Use the actual pinned **DSH and Codex codebases** as reference implementations for the
harness qualities the user wants. Trace their active mechanisms, configuration and
callers. Codex here means the public `openai/codex` source at the published non-prerelease
[0.166.0 release][codex-release]. Earlier development HEAD `86ac0916` remains historical
research, not the primary pin. Private hosted/desktop components are outside this source.
DSH's pinned snapshot is developer preview; the separate SuperAgent app is a research
prototype. Compare concrete designs without treating their presence as proof of stability.

| ID / responsibility | DSH implementation | Codex implementation | Nanobot comparison |
| --- | --- | --- | --- |
| REF1 — Session admission/steering | [Agent driver][dsh-agent-code], [inbox][dsh-inbox-code] | [Session][codex-session], [InputQueue][codex-input] | H01/H02: admission, identity, ordering, cancellation and ownership release |
| REF2 — Requests/history/compaction | [Driver][dsh-agent-code], [compaction][dsh-compact-code] | [Turn execution][codex-turn] | H02/H03: request construction, tool pairing, checkpoints, resume and provider failure paths |
| REF3 — Persistence/shutdown | [JSONL persistence owner][dsh-persistence-code] | [RolloutRecorder][codex-recorder] | H02/A01: writer ownership, ordered writes, responsiveness, acknowledgement/error and shutdown paths |
| REF4 — Authority/confinement | [Sandbox backend][dsh-sandbox-code], [policy resolver][dsh-policy-code] | [ToolOrchestrator][codex-tool-orchestrator], [SandboxManager][codex-sandbox-manager] | S01/S02/E02/U01: attempt authority, backend availability, retry scope and platform/error handling |
| REF5 — Process/resource lifecycle | [LocalJobRegistry][dsh-jobs-code] | [UnifiedExecProcessManager][codex-processes] | S02/A01/P01: ownership, output bounds, admission, cleanup and restart limits |
| REF6 — Delegation/future sessions | [SubagentManager][dsh-subagents-code] | [LocalAgentControl spawn][codex-agent-spawn], [resume][codex-agent-resume] | T01/A01: bounded foreground roles versus addressable sessions, authority and capacity assumptions |

For A1, compare Codex's bounded command channel, writer task and explicit persist/flush/
shutdown acknowledgements with nanobot's synchronous journal call chain. Their guarantees
are not automatically identical to SQLite transactions or nanobot's fsync obligations.
Compare mechanisms and required ordering before proposing a smaller correction.

DSH's [local backend][dsh-sandbox] applies shared [policy][dsh-policy] through Linux
Bubblewrap/Landlock, macOS Seatbelt or a Windows ACL restricted-token runner. Its candidate
selection/probing and partial-enforcement branches matter. Codex also selects native
platform backends under configured policy. Inspect wrapping and callers alongside backend
selection; a matching policy flag does not establish an equivalent implementation.

The [portable implementation index](codebase-discovery/provenance/source-reference-map.json)
retains pinned path/hash/URL metadata and existing bounded excerpts, without local-copy or
temporary-clone dependencies. The active REF matrix uses implementation sources only;
REF6 remains mapped rather than fully reviewed. Read actual caller bodies during comparison.
Historical maps/copies may be retained locally, but essential discovery does not require them.

## Parallel audits, repairs and integration

The user manages **this chat only**. Earlier authorised discovery produced nine family
maps; the authorised origin refresh produced three current differential maps; three completion
readers closed the 139 detail assignments. All fifteen reports and exact lists are published.
Those source-discovery workers are distinct from the later detailed audit lanes below. The former
merge hold is historical, and current origin was fetched/pinned for this explicit request.

For later source packages:

1. Use the recorded current commit/tree, instructions, inventory and exact coverage gaps;
   reconcile source changes at explicit checkpoints rather than silently refreshing a pin.
2. Follow discovery → comparison/discussion → bounded audit. Preserve raw reports and
   separate source facts, historical findings, recommendations and accepted decisions.
3. Assign one owner per responsibility/invariant and disjoint report/write locations;
   internal source workers may read shared callers and report disagreements here.
4. Keep this canonical ledger under coordinator ownership. No separate user-owned chats
   or product-code edits follow merely from this continuation protocol.

| Initial lane / proposed branch | First source batch | Shared boundaries |
| --- | --- | --- |
| Sandbox — `codex/audit-sandbox` | S02-01: backend selection, admission and error paths | S01 authority, exec/native supervision, API/UI and executor platform |
| Core state — `codex/audit-core-state` | H01-01: turn admission, ordering, ownership and cancellation | H02 execution/source owners, runner, stop/teardown and A01 callers |
| Provider contracts — `codex/audit-provider-contracts` | H03-01: request identity, retries/fallbacks and stream errors | Factory/wrappers, request ledger, tool pairing and usage |

These are proposed detailed audit packages, separate from completed discovery maps.
Refill later audit ownership with bounded provider, channel, client, context or media
slices as dependencies permit. Shared-source disagreements return to the coordinator;
workers preserve unrelated edits and do not redefine accepted architecture.

The subsystems are related: scheduled/foreground/completion inputs share ownership;
file/shell tools share authority; providers consume session-built requests; clients and
channels expose the same lifecycle. Separate worktrees isolate edits, not those contracts.
Keep each audit wave's baseline frozen and reconcile staging advances at explicit
checkpoints instead of allowing workers to pull different revisions independently.

**Audit in parallel; fix independent contracts in parallel; integrate in dependency
order.** Fixes need not all be authored in one worktree. For later repairs, assign exact
write sets and one owner per invariant; different filenames alone do not imply independence.

| Relationship | Repair arrangement |
| --- | --- |
| Disjoint implementation with unchanged shared contract | Separate repair branches/worktrees can proceed together |
| Read overlap with disjoint writes | Proceed together once shared caller/compatibility obligations are explicit |
| Same state owner, schema, authority or public contract | One repair owner, or land the common contract first |
| Dependent correction | Start/rebase the dependent work from the accepted prerequisite commit |

Preserve audit evidence before reusing an audit checkout for a repair. Reconcile active
repair ownership so another worker does not duplicate it. Integrate accepted commits in
a designated checkout in dependency order and review the combined callers, state/wire
formats, defaults and authority paths. A conflict-free Git merge does not resolve a
semantic contract mismatch; mark affected source conclusions stale until re-reviewed.

This plan defines the source audit. Later code repairs retain the repository's
[contribution requirements](../../CONTRIBUTING.md) and, when applicable,
[P6 migration validation requirements](../migration/p6-error-attribution.md). Those are
separate repair obligations, not execution phases or completion gates of this audit.

## Findings carried forward and checkpoints

Historical findings below must be reassessed against the current pinned source before
reuse. Preserve their original evidence rather than presenting it as a new source review
or treating an absent local repair as current origin behavior.

| ID / classification | Historical evidence | Source-review continuation |
| --- | --- | --- |
| A1 — P1 responsiveness defect | Synchronous SQLite process-completion journal and direct async callers; historical isolated contention probe observed about 163 ms blocking | A01-03: inspect ordered async persistence options, durable-before-launch, cancellation/shutdown and current repair overlap |
| A2 — P1 structural debt | Core/recovery interpret delivery/WebUI details; intentional `session/webui_turns.py` adapter exception | H01-04 with H02/E02: reassess imports/callers and propose neutral identity/metadata ownership preserving source/permission checks |
| A3 — P2 cleanup opportunity | Dormant legacy `SubagentManager` remains wired; active `ForegroundAgentManager` is separate | T01-02: trace SDK/public/persisted obligations before lazy construction or removal |
| SB-Q1 — implementation/platform question | Historical managed shell is Linux/Bubblewrap-only; reported macOS native/sandbox failures remain leads | S02-01: examine selection, caller assumptions, unavailable/error branches and platform choices |
| SB-Q2 — documentation inconsistency | `.agent/security.md` sandbox guidance is incomplete relative to current managed documentation | S02-01/Q01: reconcile claims with implemented modes and callers |
| CAP-Q1 — capability difference | Persistent addressable child sessions differ from bounded foreground roles | T01/A01: compare REF6; defer implementation until product direction is settled |

| Checkpoint | Scope | Status / continuation |
| --- | --- | --- |
| HA-001 | Historical first pass; 16 focused modules, 333 passed / 1 skipped, isolated journal probe and import measurements | Historical bounded results; do not complete a whole-project stream or validate current origin |
| PLAN-001 | Historical 2,451-path assignments and original sequence | Retained under `project-plan/`; current inventory has 2,455 paths |
| PLAN-002 | Original parallel audit/repair ownership protocol | Historical planning; later detailed audit packages remain separate from discovery maps |
| PLAN-003 | AR-2026-11-05.1 and pinned source/test mapping | Historical mapping retained; current method uses implementation references |
| PLAN-004 | Original single-chat worker protocol and merge hold | Historical hold superseded by authorised discovery and current-origin refresh; no current merge assertion inferred |
| PLAN-005 | Whole-project scope and reliable personal-agent goal | Accepted broad goal retained; consequential architecture choices remain discussion topics |
| PLAN-006 | Source-only implementation/code-quality audit method; AR-2026-11-05.2 | Runtime/test/benchmark qualification removed from the audit method; earlier executions remain historical |
| PLAN-007 | Authorised nine-family discovery at `4e2bf4d8`; 2,467 paths | Historical readable reports/exact lists now published under `codebase-discovery/reports/prior-2026-11-05/`; module detail was unfinished at that checkpoint |
| PLAN-008 | Authorised origin sync and three-map differential refresh at `27adced0`; AR-2026-11-06.1 | 2,455 paths assigned; 47/47 changed implementation paths currently inspected; At refresh checkpoint: 85 current + 695 carried source inspections; 139 mapped-only source gaps preserved |
| PLAN-009 | Source-discovery completion at unchanged `27adced0`; AR-2026-11-06.2 metadata | 139/139 disjoint completion reads and rows; 919 source targets = 224 current + 695 carried, zero source gaps; portable reports/previous hashes; no comparison, architecture choice or runtime qualification |
| DISC-DETAIL | C00/H01/H02/K01/R01/D01 source detail | Complete at module/relevant-body depth; exact 139 closed paths in current coverage JSON and three completion reports |
| REF-COMPARE / ARCH-DISCUSS | Pinned implementation comparison and architecture decisions | Later stage after discovery boundary; discovery publication performs neither comparison nor new design selection |
| S02-01 | Sandbox selection/admission/errors | First later audit priority after discovery/comparison/discussion |
| H01-01 | Admission/ownership/ordering/cancellation | Later bounded audit; reconcile historical findings and new runner recovery |
| H03-01 | Logical request/retry/fallback/stream contracts | Later bounded audit; preserve captured runtime/request-ledger dependencies |

## Source audit closure and resumption

The source audit closes against a recorded staging commit when:

- Every current subsystem and enabled/optional public capability has an accounted review
  disposition; implementation/callers are reviewed or explicitly deferred with a reason.
  Support files and per-adapter/client rows are accounted for rather than silently dropped.
- Ownership, dependencies, state writers, authority and lifecycle paths are documented
  for the reviewed responsibilities, including relevant error and cancellation branches.
- Findings have concrete source evidence, impact, certainty, disposition and repair
  dependencies. Shared root causes and conflicting recommendations are reconciled.
- Applicable DSH/Codex implementation comparisons explain defects, justified differences
  and future capability choices without requiring architectural imitation.
- Cleanup and efficiency proposals identify the actual complexity or work they remove;
  public compatibility and persistence obligations are explicit.
- Consequential open design decisions are recorded for discussion, and the proposed
  repair sequence is reviewable. Unresolved defects remain visible in the backlog;
  completion of source review does not require every repair to have landed.

The deliverable is an implementation/architecture assessment and prioritised correction
plan. It does not certify production reliability, measured performance or sandbox
operational parity. Source-inspection limits are conclusions to record, not additional
test or runtime gates.

On continuation, read the discovery entry point, PLAN-009 and the
[continuation contract](codebase-discovery/continuation-plan.md), verify the recorded source
revision and choose a bounded REF-COMPARE package. Retain discovery-depth boundaries and
exact next steps so later sessions can continue without restarting the map. Source comparison,
architecture decisions and detailed audit remain pending despite complete source discovery.
If staging changes, reassess affected owners and callers and preserve byte-identical evidence with its original revision.

[Portable discovery evidence](codebase-discovery/methodology.md) retains current/prior pins,
path hashes and exact reports inside Git. Optional historical local supplements include
`project-plan/`, `parallel-plan/`, `architecture-reference/`, `execution-hold/`,
`goal-alignment/`, `source-audit-method/` and original discovery/refresh/completion checkpoints.
Their before-document snapshots and local machinery are not required current inputs.
Document/link/hash checks add no application audit execution. Earlier execution results remain historical, and later repair
tests/P6 checks remain separate obligations.

[dsh-sandbox]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/sandbox/sandbox-local/README.md
[dsh-policy]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/sandbox/sandbox-policy/README.md
[dsh-agent-code]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/core/agent-loop/src/agent.ts
[dsh-inbox-code]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/core/agent-loop/src/inbox.ts
[dsh-compact-code]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/compaction/compaction-basic/src/index.ts
[dsh-persistence-code]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/session/session-persistence-jsonl/src/index.ts
[dsh-sandbox-code]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/sandbox/sandbox-local/src/index.ts
[dsh-policy-code]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/sandbox/sandbox-policy/src/index.ts
[dsh-jobs-code]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/jobs/jobs-local/src/index.ts
[dsh-subagents-code]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/subagent/subagent/src/manager.ts
[codex-session]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/session/session.rs
[codex-input]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/session/input_queue.rs
[codex-turn]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/session/turn.rs
[codex-recorder]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/rollout/src/recorder.rs
[codex-tool-orchestrator]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/tools/orchestrator.rs
[codex-sandbox-manager]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/sandboxing/src/manager.rs
[codex-processes]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/unified_exec/process_manager.rs
[codex-agent-spawn]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/agent/control/spawn.rs
[codex-agent-resume]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/agent/control/resume.rs
[codex-release]: https://github.com/openai/codex/releases/tag/rust-v0.166.0
