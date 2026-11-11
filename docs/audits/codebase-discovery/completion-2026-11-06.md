# Source-discovery completion — 2026-11-06

Module-level/relevant-body source discovery is complete at `27adced0aacd55d61791bb88a7240d7ad7bd5b75`,
tree `d49aca97c9a34a2d17eae53d672fa591652457f1`. PLAN-009 closes DISC-DETAIL without
changing the source pin, product behavior, architecture decisions or audit method.
The [current map](README.md) is canonical; the [refresh record](refresh-2026-11-06.md)
remains the historical checkpoint for changed-source reconciliation.

## Objective, scope and accepted decisions

Finish the 139 retained source-detail assignments, publish their findings and provenance,
and make the next source-comparison/discussion work executable. Accepted constraints are
whole-project inventory, the reliable personal-agent goal, source-only evidence, one
primary assignment per baseline path, hash-qualified inheritance, explicit supersession,
and discovery before comparison/discussion before bounded audit. This completion selects
no product fix, severity grade, replacement framework, migration or new platform promise.

Complete source discovery is a module/relevant-body coverage statement. It does not mean
every semantic branch was audited, support artifacts were deeply reviewed, DSH/Codex were
fully compared, architecture was accepted anew, or runtime reliability was qualified.
No repository imports, source scripts, tests, builds, checkers, dependencies, applications
or live services were executed. Product skills/templates were read as source data.

## Exact closures and coverage

The exact completion assignments are disjoint and equal the 139 mapped-only source
paths retained at refresh checkpoint. Each path has its own responsibility/caller/activation/
state-resource row in the linked report. Incidental `extra_context_paths` receive no new
coverage credit. All assigned hashes and byte sizes matched the frozen source.

| Completion report | Exact assigned/inspected paths | Primary streams | Responsibility rows |
| --- | ---: | --- | ---: |
| [Startup/core helpers](reports/completion-2026-11-06/01-startup-core.md) / [exact list](reports/completion-2026-11-06/01-startup-core-coverage.json) | 59 | C00 43; H01 14; R01 1; D01 1 | 59 |
| [State/recovery helpers](reports/completion-2026-11-06/02-state-recovery.md) / [exact list](reports/completion-2026-11-06/02-state-recovery-coverage.json) | 35 | H02 35 | 35 |
| [Knowledge/instructions](reports/completion-2026-11-06/03-knowledge-instructions.md) / [exact list](reports/completion-2026-11-06/03-knowledge-instructions-coverage.json) | 45 | K01 45 | 45 |
| Total | 139 | Six primary streams | 139 |

The ledger and [coverage summary](coverage-summary.json) retain the exact completed path
set, hashes, report memberships and derived per-family/per-stream counts. There are 137
valid primary source line anchors and two zero-byte package markers, linked without line
anchors. The [report index](reports/README.md) links all fifteen reports and exact lists.

| Coverage at this source pin | Current reads | Identical-hash carried reads | Mapped-only |
| --- | ---: | ---: | ---: |
| Source targets: 878 implementation/tooling + 41 runtime instructions | 224 | 695 | 0 |
| All 2,455 baseline paths | 232 | 808 | 1,415 |

The other roles remain 997 test-support paths, 54 assets/locks, 290 configuration/support
artifacts and 195 generic documents. Their assignment and any recorded reads do not imply
deep support review. The 47 changed implementation paths remain fully covered by current
refresh evidence. The original 85/695/139 source checkpoint is retained separately in
[refresh-coverage-summary.json](provenance/refresh-coverage-summary.json).

## Connected findings added by completion

Startup begins with admission. Clean absence or verified historical seals can permit
normal construction; retained uncertain receipt/quarantine state cannot. Explicit offline
receipt recovery moves through conversion preparation, JSONL/cursors, legacy finalization,
transaction preparation, history/memory/source-offset application, receipt-committed and
complete phases. Transaction COMPLETE still has a present receipt. Separate receipt
finalization copies it, verifies unchanged bytes/stat, unlinks, proves durable absence,
reconstructs terminal semantics through the retained copy and seals commit. Startup uses
frozen historical evidence, not a rerun of the offline coordinator against current live
memory. No production caller for coordinator/finalizer APIs was located; external callers
remain unknown. See the [state report](reports/completion-2026-11-06/02-state-recovery.md).

The runtime writer lease, managed namespace access, workspace recovery operation, checked
source authority and artifact reader are distinct capabilities. Artifact projection defaults
disabled and requires matching source/writer/reader/tool admission; canonical tool text
remains intact while a detached request may contain an opaque-reference preview. Copied-state
rematerialization is wired through an explicit CLI, despite its dormant description.
Session-handle lookup can persist metadata; a read-oriented API name is not a no-write guarantee.
Canonical follow-ups use in-process admission identity and validated restart reconstruction.

Startup/helper discovery also distinguishes callable implementation from a reachable normal
entrypoint. Bare CLI prints help; retained Desktop-picker helpers do not establish dispatch.
Classic onboarding/background/service paths have migration fences. Direct onboarding helpers
can nevertheless acquire/commit credentials or channel login before overall draft save.
Legacy context-budget, evaluator and copied-state diagnostic helpers do not grant current
runtime admission. File-edit snapshots require the accepting turn sink. Knowledge Git sync
uses selected-path/common-directory locking, while memory GitStore has a different local
index/locking contract. See the [startup/core report](reports/completion-2026-11-06/01-startup-core.md).

The instruction chain is workspace skills, context libraries, enabled plugins, then bundled
skills. Binary/environment prerequisite checks do not establish OS support, tool registration,
external command usability or credentials. None of the assigned bundled skills declares
`always`; references and helpers require explicit follow-up reads/execution. Main-context,
foreground-role and legacy-subagent loaders pass different disabled-skill/policy/context
options. Template sync copies missing profiles without overwriting, while packaged scenario
prompts render directly; managed project AGENTS and global profiles have separate checks.
Repository discovery reports candidates/access metadata without granting access or persisting
the catalog. See the [knowledge/instruction report](reports/completion-2026-11-06/03-knowledge-instructions.md).

The changed-source facts in the refresh remain current: explicit manifest omits goal/self
tools; local queue activation/config is absent; cron timers are opt-in with manual run fenced;
process completion is independently opt-in; Dream/Heartbeat remain inactive; canonical source
commands reject goal/trigger/dream; ordinary generic tool-error recovery allows at most two
recoveries; ordinary messages use server-owned scope while explicit controls mutate state;
transcript repair synthesizes response-only completion; managed shell/native Once/Full are
Linux-only; IPC uses SO_PEERCRED only; Android keepalive is client/process-local, nonsticky
`dataSync`, with no wake lock or push path. Missing local-only fixes remain source divergence,
not automatic regression grades.

## Remaining comparison and discussion agenda

These are questions for later source comparison/audit or consequential product discussion;
they are not remaining source-discovery gaps.

| Area | Current evidence and next question | Next owner/package |
| --- | --- | --- |
| Admission, request consistency and recovery | Compare source/execution/ticket identities, ordinary error classification, follow-up recovery and isolated-child boundaries against actual REF1/REF2 callers. | REF-COMPARE, H01/H02/H03 |
| Persistence and completion | Compare writer/ack/shutdown ordering and source/receipt/projection truth; include memory GitStore index behavior and offline recovery's separate capabilities. | REF3/REF5, H02/A01/P01 |
| Authority and platforms | Linux-only managed shell/native supervision and SO_PEERCRED-only IPC remain source limits. Desired executor/client platforms and backend obligations require discussion. | REF4, ARCH-DISCUSS, later S02-01 |
| Future task sessions | Bounded foreground roles differ from persistent addressable sessions. REF6 is mapped, not fully reviewed; task/attempt/session identity, capacity and authority remain undecided. | REF6, ARCH-DISCUSS |
| Activation and compatibility | Reconcile settings/prompt claims with inactive Dream/Heartbeat, omitted My/goal tools, absent local queue consumer, manual cron fence, copied-state APIs and other retained compatibility paths before removal or rewiring. | REF-COMPARE, A01/T01/K01/Q01 |
| Instruction fidelity | Config substitution guidance, cron origin/timezone text, tool/skill availability, browser `--cdp`, missing skill-creator references, literal tmux baseDir/private socket, My scratchpad lifetime and semantic-storyboard fallback all need requirement-based disposition. | K01/Q01, ARCH-DISCUSS if behavior changes |
| Clients and resources | Assess source terminal proof versus synthesized display completion, scope capture/steering, asset caps/request ownership, skills credential/events, mixed paste, Android process/service lifetime and external-provider capacity. | E02/U01/R01/P01; resource policy discussion |

No installed configuration, plugins, credentials, provider availability, deployed artifacts,
OS enforcement, crash behavior or actual delivery was observed. The next executable contract
is [continuation-plan.md](continuation-plan.md). Consequential platform, task-session,
compatibility-retirement and resource-policy decisions stay pending with the coordinator.

## Validation and publication boundary

[Methodology](methodology.md), [report integrity](provenance/report-manifest.json),
[baseline](baseline.json), [path ledger](path-inventory.tsv),
[change record](provenance/changed-paths.tsv) and [validation](provenance/publication-validation.json)
allow independent source/document checks using standard Git and Python's standard library.
The historical harness Decision heading and all following bytes are unchanged. The
frozen payload after the literal heading has SHA256
`654f4c367fbf7bb0ddc698122ecd0326e8fb288da94171d554493123fa3c181c`; including the heading,
the digest is `75a8b177693f48d8c2cd97c58600f7c473e4a3208b1ca722c9fd619f497ce96e`.
Documentation checks are not application tests. Coordinator acceptance is separate from the
writers' completion claims. Reopening this record requires a named evidence gap or an
explicit source-pin advance, not silent baseline refresh.
