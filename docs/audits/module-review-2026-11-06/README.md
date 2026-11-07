# Whole-project module source review — 2026-11-06

**ACTIVE — resumed from the saved web-continuation checkpoint.**
This directory is the current publication and coverage ledger. The frozen audit source remains
`9562e57f8333d7f14b6aaeae837660723a9b2650`, tree `bbe2f237efe0cd7b935946b533f66ed0097a3c93`.
During R01-A and rechecked at the R01-B boundary, current `staging` is `29612c3df01dc3dce3501fe349bcf555cd6c58b6` (two commits ahead); only `nanobot/agent/skills.py` and its test changed. No R01 primary changed; K01 current-origin skill/test source refresh was completed as a separate 2026-11-07 source-only addendum. [S01](reports/S01.md) and
[S02](reports/S02.md) retain coordinator **PASS**. [H01](reports/H01.md) now also has
coordinator **PASS** for its source-review publication: seven classified findings plus the
open H01-F8 registration contract and six retained questions; its acceptance publication is
checkpointed at `f6a41b840c7e50c8f757e8889a880bc3ce77d47d`. Integrated review and product/
architecture decisions remain pending. C00 is now coordinator-checked from its saved 70/70 handoff and checkpointed at
`85c59330c91e7982574d79f92602cc7e605acb4c`. H02 is now coordinator-checked from the
superseding F74/P0/R0 handoff and checkpointed at `08ba3c3d48c9c96c1d04c25f5815a5efd17e20f4`;
H03 is now coordinator-checked at final F45/P0/R0 with seven promoted findings and seven retained questions. U01 is now coordinator-checked at final F203/P0/R0 with six qualified P2 findings and six retained questions, checkpointed at `b53644a2e20086a5d6a48e237e571d1720151270`. D01 is now coordinator-checked at final F47/P0/R0 with two qualified P2 findings and five retained questions, checkpointed at `9db80a30be08be3428ca33fda1532031dabd6a7f`. T01 is now coordinator-checked at final **F44/P0/R0** with twelve promoted findings and eight retained questions. A01 is now coordinator-checked at final **F47/P0/R0** with nine promoted findings and seven retained questions, checkpointed at `9bdcc59a5938084e843b61509e64bfb745b984a7`. K01 is coordinator-checked **PASS** at **F56/P0/R0** with thirteen promoted findings and eight final questions. E02 is coordinator-checked **PASS** at **F68/P0/R0** with twenty-three promoted findings and nine final questions. E01 is now **closed as a limited-depth/non-priority exception** by explicit user decision on 2026-11-07. E01-A..F preserve 40 deep primary-body reads and twenty-six findings; the remaining 55 channel primaries received only a quick high-signal sanity screen. This closure is not coordinator PASS or deep body credit, and channel functionality is intentionally left untouched. R01 is now coordinator-checked **PASS** at **F70/P0/R0** with twenty promoted findings (one conditional P1, nineteen P2) and ten final discussion questions. P01 cross-cutting synthesis is coordinator-checked, and the two-file K01 current-origin staging refresh is now source-checked; remaining work is the fresh integrated cross-module review and D1–D7 discussion. No repair/design is
accepted; publication/acceptance, integrated review and Git preservation remain distinct.

The foundation was independently verified **PASS** by the coordinator and committed/pushed
at `5f0d14479b6ecd7a334ae72476a7c0ad5065be66`. That supplied acceptance applies to the
publication foundation, not S01/S02 or the overall architecture assessment.


**2026-11-07 M01 closure and higher-depth recheck:** [M01 report](reports/M01.md) and [M01 coverage](reports/M01-coverage.json) are now coordinator-checked PASS at **F14/P0/R0**. **Six** qualified P2 findings and **six** final questions are retained after higher-depth second-pass verification; [M01-A handoff](handoffs/M01-A.md) is superseded as a partial checkpoint, not erased. Accepted deep evidence is now **788/824** (95.6%), with only X01 (11) and Q01 (25) remaining; P01 is cross-cutting with zero additional primary targets. E01 remains an explicit limited-depth exception. Next: **X01 only**. All 14 source SHA-256 hashes were independently recomputed and matched during the second pass; there was no independent peer or runtime test. Current staging delta remains K01-only; integrated review and D1–D7 decisions remain pending.

**2026-11-07 X01 closure (latest checkpoint):** [X01 report](reports/X01.md) and [X01 coverage](reports/X01-coverage.json) are coordinator-checked **source-review PASS at F11/P0/R0**. All 11 assigned primary bodies were independently SHA-256 verified (137,653 UTF-8 bytes, 3,570 lines). Four qualified findings: two **conditional P1** (CLI App subprocess permission scope, crawler DOM expand click), two **P2** (subprocess output capture, catalog-source collision); six questions retained. Current deep-reviewed evidence is **799/824 (97.0%)**, with **Q01's 25 primary bodies only** remaining; P01 is cross-cutting, zero additional primary bodies. X01 and M01 are same-assistant source checks, not independent peer review or runtime tests. **Next: Q01 only**, followed by P01, K01 current-origin refresh, and integrated D1–D7 decisions. The earlier M01 checkpoint above is historical and its X01-next line is superseded.

**2026-11-07 Q01 closure (historical checkpoint):** [Q01 report](reports/Q01.md) and [Q01 coverage](reports/Q01-coverage.json) now hold **F25/P0/R0** source-review PASS with 25/25 independently SHA-256/byte/blob checked primary bodies (141,582 UTF-8 bytes; 3,952 physical lines), **seven qualified findings** (one conditional P1 and six P2) and **seven final questions**. All required deep source streams are now complete at **824/824**; E01's deliberate 95-target limited-depth exception remains separate. Aggregate: **136** findings and **105** questions across 16 coordinator-checked deep source reports. **Next P01 only** (0 new primary targets), then K01 staging-delta refresh, fresh integrated review and D1–D7 decisions. No runtime tests or independent peer/integrated acceptance. Earlier M01 and X01 next-step lines below are historical/superseded.

**2026-11-07 P01 cross-module simplification closure (historical checkpoint):** [P01 report](reports/P01.md) and [P01 coverage](reports/P01-coverage.json) are now published and same-assistant source-synthesis checked PASS. The report proposes **10 bounded existing-owner simplification packages**, explicitly distinguishes **7 misleading cross-stream merge pairs**, and does **not** add new source bodies, findings or independently counted questions. All **824/824** deep-required primary source bodies remain accepted across the original 16 primary streams, plus the deliberate **95-body E01 limited-depth exception**. The deep-source findings/questions remain **136/105**; the seventeenth coordinator-checked report is P01, with zero primary-body denominator. **All 18 planned reports are now published** (16 deep-source, E01 limited-depth, P01 cross-cutting). No product repair or D1–D7 architecture decision is accepted. **Next stage: K01 current-origin source refresh only**, then separate integrated reconciliation and final decisions. Earlier Q01-next instructions below are historical and superseded; no code or runtime operations were performed.

**2026-11-07 K01 staging refresh (prior checkpoint):** The [K01 report addendum](reports/K01.md) and [K01 current-origin coverage overlay](reports/K01-coverage.json) now document the exact frozen `9562e57` → staging `29612c3` two-file diff. Both full versions of `nanobot/agent/skills.py` and `tests/agent/test_skills_loader.py` were independently SHA-256 checked; frozen hashes and bytes match the inventory. The changed code only formats bundled skill paths in the summary; **K01-F03 remains unresolved**. A new **K01-STG-F01 (conditional P2; staging-only)** records that bundled paths advertised as `skills/<name>/SKILL.md` can resolve to a same-name existing file in a selected project before the bundled read fallback, confusing the claimed instruction source. It is **not a proven filesystem policy bypass or a running attack**. The frozen **13 K01 findings / 136 deep-source findings / 105 questions / 824 of 824 deep primary bodies** are unchanged; the new issue is a separately qualified staging-only entry for integrated review. The original 95 E01 channel primary files remain limited-depth. **Next: fresh integrated cross-module review**, then D1–D7 choices. No tests or product changes were made.

**2026-11-07 I01 integrated authority/capability substage (prior checkpoint):** [I01 source-evidence report](integrated/I01-authority.md) and [I01 evidence map](integrated/I01-authority-evidence.json) reconcile **28 distinct original frozen finding IDs plus one staging-only K01-STG-F01** across **8 boundary groups**: path/provenance, native-risk approval, delegated read-only, Windows browser-worker peer identity, listener/mobile authorization, multi-instance credentials/roots, external trust exits/secrets, and prompt authority/audit. They identify repair owners and 8 clear false-merge cases. This is a **bounded same-assistant source synthesis checkpoint**, not full integration, no product repair and no D1–D7 decision. Original **824/824 deep source coverage, 136 findings and 105 questions are unchanged**; E01 still holds the user-authorized 95-file limited-depth exception. The next stage is **I02 only** (durable effects, retry/cancellation/recovery/completion), followed by other bounded integration and final decisions. No source tests or application executions took place.

**2026-11-07 I02 integrated durability/recovery substage (prior checkpoint):** [I02 source review](integrated/I02-durability.md) and [I02 evidence map](integrated/I02-durability-evidence.json) are published and same-assistant source-synthesis checked PASS for their limited remit. Nine groups reconcile **53 distinct existing ledger entries** from H01/H02/H03/A01/S02/E02/R01/M01/C00, including three **H02 optional structural/retirement proposals rather than newly verified live defects**. Original finding **S02-F02** also appears in I01; cross-substage references do not inflate totals. Critical distinctions: required source writes versus accepted but stale response, known-applied mutation versus lost acknowledgement, missing provider terminal versus failed transport, prelaunch reservation versus still-running child, initial device pairing versus refresh custody, and complete reference proof versus unsafe attachment pruning. The source-only result recommends typed per-owner outcomes and explicit fail-closed recovery, **not one shared transaction engine or unconditional retries**. Frozen **824/824 required deep source bodies, 136 findings, 105 questions and 95-file E01 limited-depth exception** remain unchanged; no D1–D7 decision, runtime test or product repair. **Next I03 only:** frontend/client, registered capabilities, route/skill guidance, compatibility and feature activation. Additional resource/release/efficiency integration and D1–D7 remain separate.

**2026-11-07 I03 client/capability integration (prior checkpoint):** [I03 report](integrated/I03-client-capabilities.md) and [I03 source/ID ledger](integrated/I03-client-capabilities-evidence.json) source-reconcile **32 distinct existing original findings** from U01, D01, T01, K01, A01, E02, R01 and C00 in **8 groups**. They separate queued browser attachment-only auto-flush from TUI dequeue-before-send, stale async client results, failed optimistic sidebar persistence, completion metadata misattribution, core tool instructions for unregistered tools, missing goal tools/inactive local-trigger consumer, skill guidance discrepancies, pairing-renew settings route omission, mobile declared HTTP vs live WS/edge routes, plugin/MCP readiness and optional/gated compatibility states. **Known overlaps:** K01-F03 was already in I01; A01-F07 was already in I02; K01-STG-F01 is a separate staging-only overlay, not a frozen source finding. Current onboarding wizard F04/F05 remains gated off and multi-chat cron scope A01-F06 is conditional. The source-only checkpoint leaves original **824/824 required deep bodies, 136 findings, 105 questions**, and **95 E01 limited-depth** bodies unchanged. No tests or source repairs or D1–D7 choices. **Next: I04 only** for release/toolchain/deployment/qualification and residual resource/compatibility; whole-project final reconciliation and D1–D7 later as separate stages.

**2026-11-07 I04 release/residual integration (latest checkpoint):** [I04 report](integrated/I04-release-residuals.md) and [I04 evidence ledger](integrated/I04-release-residuals-evidence.json) reconcile the **exact 26 original finding IDs not yet referenced by I01–I03**. Seven groups cover install/build artifact identity, deploy/rollback and CI/upgrade qualification, remaining process/network/output bounds, constructor/script validation, approval target/provenance, runtime capability/credential generations, and compatibility/knowledge ownership. Mechanical coverage now shows **I01–I04 map all 136/136 original findings at least once**. This is **mapping completion only**: no final duplicate collapse, no reprioritized repair backlog, no D1–D7 decision, no runtime/build/deploy validation. Original **824/824 deep-required bodies, 136 findings, 105 questions**, the **95-file E01 limited-depth exception**, and separate **K01-STG-F01** retain their status. **Next: I05 only** for project-wide true-duplicate vs non-merge reconciliation, repair ownership and priority ordering; D1–D7 remain a later separate discussion.

Start with [status.json](status.json), the exact [scope](scope.json), the
[report template](report-template.md), [coverage template](coverage-template.json),
[pending synthesis](synthesis.md) and [structural validation](validation.json).
Future report paths are reserved in status.json and become links only after publication.
S01/S02 exact evidence is in [S01 coverage](reports/S01-coverage.json) and
[S02 coverage](reports/S02-coverage.json); H01 evidence is in [H01 coverage](reports/H01-coverage.json), E02 evidence is in [E02 coverage](reports/E02-coverage.json), R01 in [R01 coverage](reports/R01-coverage.json), X01 in [X01 report](reports/X01.md) and [X01 coverage](reports/X01-coverage.json), and Q01 in [Q01 report](reports/Q01.md) and [Q01 coverage](reports/Q01-coverage.json).

The [durable handoff index](handoffs/index.json) preserves [C00](handoffs/C00.md),
[H02](handoffs/H02.md) and [H03](handoffs/H03.md). C00 has now been published/checked as
[C00 report](reports/C00.md), and H02 has now been published/checked as [H02 report](reports/H02.md).
H03 has now been published/checked as [H03 report](reports/H03.md), U01 as [U01 report](reports/U01.md), D01 as [D01 report](reports/D01.md), E02 as [E02 report](reports/E02.md), and R01 as [R01 report](reports/R01.md). H02 supplements 5A/5B supersede its initial F33/P4/R37 attribution
with current F74/P0/R0. Pipeline states below are preserved. Earlier resumption verified `staging` identical to pinned `9562`; R01-A/B/C/D/E/F/G/H/I/J/K observed and rechecked `staging` at `29612c3`, with only the K01 skill-display source/test changed. All R01 primary evidence remains byte-current; K01 current-origin refresh is pending. H01, C00, H02, H03, U01, D01, T01 and K01 are now coordinator-checked. T01 is closed at F44/P0/R0 with twelve promoted findings and eight retained questions. A01 is closed at F47/P0/R0 with nine promoted findings and seven retained questions. F06/F08/F09 remain explicitly conditional. K01-A/B/C/D/E are published and coordinator-checked at F56/P0/R0 with thirteen promoted findings and eight final questions. E02-A/B/C/D/E/F/G/H/I/J/K/L/M are published and coordinator-checked at F68/P0/R0 with twenty-three promoted findings and nine final questions. E01-A/B/C/D/E/F remain preserved at F40/P0/R55 with twenty-six findings. The former E01-G-and-later deep-adapter plan is cancelled by the 2026-11-07 scope decision: all 55 remaining E01 primaries received a quick sanity screen only, E01 is closed as limited-depth/non-priority, R01-A/B/C/D/E/F/G/H/I/J/K are published and coordinator-checked at F70/P0/R0 with twenty promoted findings and ten final questions; M01 is complete/rechecked, X01 is now also source-reviewed and coordinator-checked; next is Q01 only, before P01. A fresh integrated review and D1–D7 discussion remain pending. The coordinator
owns final structural verification and Git preservation. Continuation now proceeds stage-by-stage: keep each session to one bounded stage or substage, save its evidence/gaps/exact next action, and checkpoint before expanding scope.

## Scope and current progress

The immutable [discovery inventory](../codebase-discovery/path-inventory.tsv) supplies
**919 primary targets: 878 implementation/tooling sources and 41 runtime instruction
sources**, assigned once across seventeen primary streams. The frozen inventory remains unchanged. By explicit 2026-11-07 scope decision, E01's 95 channel targets are a limited-depth exception, leaving **824 targets in the current deep-review-required denominator**. P01 examines efficiency
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
| E01 | Channel framework and sixteen non-WebSocket adapters | 95 | limited-depth closed (40 deep reads preserved + 55 quick-screened; excluded from current deep-review denominator) |
| E02 | API, SDK, WebSocket, backend WebUI and client contracts | 68 | coordinator-checked (F68/P0/R0) |
| U01 | React WebUI and browser sources | 203 | coordinator-checked |
| M01 | Attachments, audio/images, ingress and adoption | 14 | coordinator-checked (F14/P0/R0; six findings after second pass) |
| R01 | Remote IPC/auth/pairing and Android/native clients | 70 | coordinator-checked (F70/P0/R0) |
| D01 | Desktop adapter, Tauri shell and terminal UI | 47 | coordinator-checked |
| X01 | Optional apps, crawler and extensions | 11 | coordinator-checked (F11/P0/R0; two conditional P1, two P2 findings) |
| Q01 | Build/install/dependency/CI/deployment/migration tooling and documentation | 25 | coordinator-checked (F25/P0/R0; seven findings) |
| P01 | Efficiency and simplification across every stream | 0 (shared) | queued |

These are assignment counts. Published S01/S02/H01/C00/H02/H03/U01/D01/T01/A01/K01/E02/R01/M01/X01/Q01 evidence covers **824 primary bodies** at the frozen audit pin, so **0 of the 824 current deep-review-required targets** still lack coordinator-checked source evidence. The remaining 95 inventory primaries belong to E01's explicit limited-depth exception; E01 remains the explicit 95-target limited-depth exception. Mechanical
existence, hash agreement, prior discovery and the earlier comparison supply no new read credit.
Each adapter, provider and native client normally needs its own accounted disposition; reading a
shared base does not review its implementations. **E01 is the explicit exception:** chat adapters are currently non-priority, so 55 remaining channel primaries were quick-screened rather than deeply traced. No path count or quick screen certifies all-branch semantics.

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


## R01 published checkpoint

R01 is published and coordinator-checked **PASS** at **F70/P0/R0**. All twenty provisional findings were retained: **R01-F02** is the single conditional P1 mobile-WebSocket scope bypass; R01-F01/F03-F20 are nineteen P2 findings with their conditional/unconditional boundaries preserved. The coordinator reconciled all 70 primary paths against the immutable inventory (**549,841 bytes / 15,397 physical lines**), found no R01 source drift in current staging, and retained ten final discussion questions. See [R01 report](reports/R01.md), [R01 coverage](reports/R01-coverage.json) and [R01 handoff](handoffs/R01.md). Integrated review and D1–D7 remain pending.
