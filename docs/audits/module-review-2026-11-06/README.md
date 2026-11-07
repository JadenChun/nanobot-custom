# Whole-project module source review — 2026-11-06

**ACTIVE — resumed from the saved web-continuation checkpoint.**
This directory is the current publication and coverage ledger. `staging` was freshly
reverified unchanged at `9562e57f8333d7f14b6aaeae837660723a9b2650`, tree
`bbe2f237efe0cd7b935946b533f66ed0097a3c93`. [S01](reports/S01.md) and
[S02](reports/S02.md) retain coordinator **PASS**. [H01](reports/H01.md) now also has
coordinator **PASS** for its source-review publication: seven classified findings plus the
open H01-F8 registration contract and six retained questions; its acceptance publication is
checkpointed at `f6a41b840c7e50c8f757e8889a880bc3ce77d47d`. Integrated review and product/
architecture decisions remain pending. C00 is now coordinator-checked from its saved 70/70 handoff and checkpointed at
`85c59330c91e7982574d79f92602cc7e605acb4c`. H02 is now coordinator-checked from the
superseding F74/P0/R0 handoff and checkpointed at `08ba3c3d48c9c96c1d04c25f5815a5efd17e20f4`;
H03 is now coordinator-checked at final F45/P0/R0 with seven promoted findings and seven retained questions. U01 is now coordinator-checked at final F203/P0/R0 with six qualified P2 findings and six retained questions, checkpointed at `b53644a2e20086a5d6a48e237e571d1720151270`. D01 is now coordinator-checked at final F47/P0/R0 with two qualified P2 findings and five retained questions, checkpointed at `9db80a30be08be3428ca33fda1532031dabd6a7f`. T01 is now coordinator-checked at final **F44/P0/R0** with twelve promoted findings and eight retained questions. A01 is now coordinator-checked at final **F47/P0/R0** with nine promoted findings and seven retained questions, checkpointed at `9bdcc59a5938084e843b61509e64bfb745b984a7`. K01 is coordinator-checked **PASS** at **F56/P0/R0** with thirteen promoted findings and eight final questions. E02-A/B/C/D/E/F have advanced the next stream to **F41/P0/R27** with ten provisional conditional findings; E02-F added three P2 findings, E02 remains unaccepted, and six later streams remain queued. No repair/design is
accepted; publication/acceptance, integrated review and Git preservation remain distinct.

The foundation was independently verified **PASS** by the coordinator and committed/pushed
at `5f0d14479b6ecd7a334ae72476a7c0ad5065be66`. That supplied acceptance applies to the
publication foundation, not S01/S02 or the overall architecture assessment.

Start with [status.json](status.json), the exact [scope](scope.json), the
[report template](report-template.md), [coverage template](coverage-template.json),
[pending synthesis](synthesis.md) and [structural validation](validation.json).
Future report paths are reserved in status.json and become links only after publication.
S01/S02 exact evidence is in [S01 coverage](reports/S01-coverage.json) and
[S02 coverage](reports/S02-coverage.json); H01 evidence is in [H01 coverage](reports/H01-coverage.json).

The [durable handoff index](handoffs/index.json) preserves [C00](handoffs/C00.md),
[H02](handoffs/H02.md) and [H03](handoffs/H03.md). C00 has now been published/checked as
[C00 report](reports/C00.md), and H02 has now been published/checked as [H02 report](reports/H02.md).
H03 has now been published/checked as [H03 report](reports/H03.md), U01 as [U01 report](reports/U01.md), and D01 as [D01 report](reports/D01.md). H02 supplements 5A/5B supersede its initial F33/P4/R37 attribution
with current F74/P0/R0. Pipeline states below are preserved. Resumption verified `staging` identical to pinned `9562`; no affected source evidence needed
refresh. H01, C00, H02, H03, U01, D01, T01 and K01 are now coordinator-checked. T01 is closed at F44/P0/R0 with twelve promoted findings and eight retained questions. A01 is closed at F47/P0/R0 with nine promoted findings and seven retained questions. F06/F08/F09 remain explicitly conditional. K01-A/B/C/D/E are published and coordinator-checked at F56/P0/R0 with thirteen promoted findings and eight final questions. E02-A/B/C/D/E/F are saved at F41/P0/R27 with ten provisional conditional findings and zero publication credit; next is E02-G small ancillary/session-service slice only before E01/R01/M01/X01/Q01/P01. A fresh integrated review and D1–D7 discussion remain pending. The coordinator
owns final structural verification and Git preservation. Continuation now proceeds stage-by-stage: keep each session to one bounded stage or substage, save its evidence/gaps/exact next action, and checkpoint before expanding scope.

## Scope and current progress

The immutable [discovery inventory](../codebase-discovery/path-inventory.tsv) supplies
**919 primary targets: 878 implementation/tooling sources and 41 runtime instruction
sources**, assigned once across seventeen primary streams. P01 examines efficiency
across those streams and adds no denominator. The other 1,536 inventory paths are support,
test, configuration, asset or generic documentation context; they remain available for
explicit supplementary reads.

| Stream | Responsibility | Primary targets | Current state |
| --- | --- | ---: | --- |
| C00 | Entrypoints, CLI, commands, config, gateway composition, shared utilities | 70 | coordinator-checked |
| H01 | Loop, runner, bus, hooks, turn admission and delivery | 29 | coordinator-checked |
| H02 | Sessions, ledgers, history, memory, compaction and recovery | 74 | coordinator-checked |
| H03 | Providers, models, budgets, keys, usage and accounting | 45 | coordinator-checked |
| S01 | Filesystem authority, folder grants, private-state policy and security | 15 | coordinator-checked |
| S02 | Shell, sandbox, native admission, exec sessions and supervision | 6 | coordinator-checked |
| T01 | Tools, plugins, MCP, roles, delegation and compatibility | 44 | coordinator-checked |
| A01 | Cron, triggers, process completion and retained goals | 47 | coordinator-checked |
| K01 | Context repositories, knowledge, skills, prompts and templates | 56 | coordinator-checked (F56/P0/R0) |
| E01 | Channel framework and sixteen non-WebSocket adapters | 95 | queued |
| E02 | API, SDK, WebSocket, backend WebUI and client contracts | 68 | in-progress (E02-A/B/C/D/E/F F41/P0/R27; unaccepted) |
| U01 | React WebUI and browser sources | 203 | coordinator-checked |
| M01 | Attachments, audio/images, ingress and adoption | 14 | queued |
| R01 | Remote IPC/auth/pairing and Android/native clients | 70 | queued |
| D01 | Desktop adapter, Tauri shell and terminal UI | 47 | coordinator-checked |
| X01 | Optional apps, crawler and extensions | 11 | queued |
| Q01 | Build/install/dependency/CI/deployment/migration tooling and documentation | 25 | queued |
| P01 | Efficiency and simplification across every stream | 0 (shared) | queued |

These are assignment counts. Published S01/S02/H01/C00/H02/H03/U01/D01/T01/A01/K01 evidence covers **636 primary bodies**. A01's F47/P0/R0 source review is now published and coordinator-checked, adding 47 bodies to the accepted denominator. Mechanical
existence, hash agreement, prior discovery and the earlier comparison supply no new read credit.
Each adapter, provider and native client needs its own accounted disposition; reading a
shared base does not review its implementations. No path count certifies all-branch semantics.

## Evidence and status rules

| Status | Required evidence / authority |
| --- | --- |
| queued | Assigned scope; no dispatched deep-review report yet |
| in-progress | Coordinator dispatch recorded; findings and completion remain unaccepted |
| report-written | Supplied report and coverage published with pins, limitations and exact reads |
| coordinator-checked | Coordinator explicitly records its check and outstanding issues |
| integrated-review-complete | Coordinator records cross-module reconciliation, material gaps/dispositions and checkpoint evidence |

The publication owner may record supplied dispatch and report evidence, but cannot award
coordinator or integrated acceptance. A Git checkpoint preserves a module's state; it does
not by itself establish semantic acceptance. Finding/question counts stay unknown for
streams without published analyst evidence. S01-F03 is shared with S02 and counted once;
preliminary coordinator corroboration of specified mechanics is separate from report acceptance.

Coverage separates bounded implementation/instruction body reads from caller context,
search-only results, test-source excerpts, support reads, justified exclusions and material
unread gaps. Reports retain exact ranges or symbols, evidence origin and limits. Shared
reads do not duplicate primary ownership. Source excerpts and tests may be read as text;
no test, build, checker or runtime execution belongs to this review.

## Pins and preserved history

| Source | Commit / tree |
| --- | --- |
| Nanobot | `9562e57f8333d7f14b6aaeae837660723a9b2650` / `bbe2f237efe0cd7b935946b533f66ed0097a3c93` |
| DSH | `fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159` / `6cccb3ee8c1501c9d623258a660f7413761dd0f0` |
| Public Codex | `6ff1e1ca3b30fc8b914720681fcf5fe62e95f856` / `03f3b46689faf274f126dc78508ed238564da8ba` (`0.166.0` / `rust-v0.166.0`) |

The frozen discovery and [first architecture comparison](../architecture-review-2026-11-06/README.md)
remain at nanobot `27adced0aacd55d61791bb88a7240d7ad7bd5b75`, tree
`d49aca97c9a34a2d17eae53d672fa591652457f1`. The latter is an accepted **targeted
33-row REF1–REF6 checkpoint**, not this whole-project deep review.

The named staging delta changes `nanobot/agent/runner.py`,
`nanobot/providers/fallback_provider.py`, `tests/agent/test_pre_dispatch_recovery.py`,
`tests/agent/test_runner_errors.py` and `tests/agent/test_runner_fallback.py`.
Scope retains current Git blob/SHA256 identities and old discovery hashes: 917 primary
targets are byte-identical and two changed. This permits mechanical provenance comparison,
not automatic semantic carry-forward. H01/H03 must reconcile affected bodies and callers.
The three changed tests are supplementary source context, outside the primary denominator.
No prior inventory, report, manifest or historical harness Decision has been refreshed.

DSH/Codex are pinned engineering references, with no feature-parity requirement. Each
report must compare an actual relevant mechanism or explain nonapplicability. Hermes
remains secondary product-direction context; no new Hermes source study is authorized.
Earlier architectural changes need explicit history/caller/compatibility evidence, not
automatic preservation or reversal.

## Review and discussion boundary

Each report assesses responsibility, actual owners/callers/state/lifetime, strengths,
conditional defects, smaller alternatives and concrete removal/combination/move/retirement
options. It records authority, durability and compatibility consequences, prior-change
evidence and uncertainty. The publisher records supplied evidence without inventing findings.

D1–D7 are **explicitly deferred to the final consolidated discussion**, as authorized
review sequencing. This does not accept a behavior, platform, schema, migration, budget
or compatibility removal. No open decision blocks source inspection. [Synthesis](synthesis.md)
stays pending until all eighteen streams and their shared boundaries are accounted for.
All conclusions are source-based design/efficiency assessments; there is no measured
benefit, runtime stability, sandbox enforcement or delivery certification.
