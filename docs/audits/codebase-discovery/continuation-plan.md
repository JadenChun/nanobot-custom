# Source comparison and architecture discussion: continuation contract

This is the executable next-stage plan after [PLAN-009 source-discovery completion](completion-2026-11-06.md).
It changes audit documentation within the established architecture. It assigns no product
implementation, schema change, migration, runtime exercise or release action.

**Continuation checkpoint:** the [2026-11-06 architecture review](../architecture-review-2026-11-06/README.md)
now records the four comparison packages below as 33 targeted REF1–REF6 mechanism/caller
rows. This original contract and its input-state descriptions remain retained. The new
[decision record](../architecture-review-2026-11-06/decisions.md) keeps consequential choices
pending; the [backlog](../architecture-review-2026-11-06/audit-backlog.md) supplies the current
all-eighteen-stream continuation and first later AUDIT-S02-01 contract. This is not whole-project
detailed semantic-audit completion or independent coordinator acceptance.

## Objective and non-goals

Explain how nanobot's pinned ownership, admission, state, authority and resource mechanisms
support the accepted reliable personal-agent goal. Compare actual DSH/Codex implementation
mechanisms, record justified differences and unresolved governing contracts, and prepare
bounded audit packages. Keep Sandbox first within the later detailed audit.

Non-goals are implementing repairs, choosing a replacement harness, promising cross-platform
parity, accepting task-session schemas, removing compatibility paths, selecting new resource
budgets, runtime testing, benchmarks, dependency setup, provider exercises or stability grading.
A source difference alone is not a defect.

## Verified current behavior and accepted inputs

Nanobot source is `27adced0aacd55d61791bb88a7240d7ad7bd5b75`, tree
`d49aca97c9a34a2d17eae53d672fa591652457f1`. Discovery accounts for 2,455 paths once and
919 source targets with 224 current/695 identical-hash carried reads and no source gaps.
The [current map](README.md), [fifteen reports](reports/README.md), exact coverage lists,
[ledger](path-inventory.tsv) and [methodology](methodology.md) are the required input set.
The [completion agenda](completion-2026-11-06.md#remaining-comparison-and-discussion-agenda)
provides current questions. Historical findings retain their original pin and certainty.

Accepted decisions are whole-project scope, source-only inspection, the reliable personal-agent
goal, discovery → comparison/discussion → bounded audit, no automatic adoption of a reference
architecture, and one owner per invariant/write set. Established repository constraints favor
small loop/runner changes and edge capabilities, explicit configuration and typed boundaries.
The existing direction to retain loop/runner/canonical ownership is a working recommendation;
this publication accepts no further architecture choice.

References stay pinned: DSH `fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159`; public Codex
0.166.0 / `rust-v0.166.0`, commit `6ff1e1ca3b30fc8b914720681fcf5fe62e95f856`, tree
`03f3b46689faf274f126dc78508ed238564da8ba`. Use the [portable implementation index](provenance/source-reference-map.json)
and the [REF matrix](../project-audit-plan.md#dsh-and-codex-implementation-comparison).
The index's old excerpts and REF6-mapped status are historical bounded input evidence;
the linked checkpoint now provides targeted REF6 body/caller comparison, not a full semantic audit.
Temporary clones, local-only prior commits and historical runtime evidence are optional.

## Proposed documentation behavior and interfaces

A comparison row must contain: required outcome or open requirement; nanobot source pin,
symbol/caller/configuration; applicable reference pin/symbol/caller; writers and identities;
authority/lifecycle/error boundary; concrete difference; governing requirement; justified
scope difference or candidate issue; uncertainty; dependencies and proposed next disposition.
State observations, recommendations and accepted decisions in separate fields. Do not fill
an unknown requirement with a presumed product choice.

A decision record must identify options, source evidence, compatibility/persistence/security/
resource consequences, recommendation, human/coordinator acceptance and affected packages.
A repair contract is created only after material gates are resolved and must include exact
write ownership, accepted observable behavior, non-goals, acceptance criteria, focused
validation and rollout/rollback. No API/schema/interface change is authorized by this plan.

## Work packages, ownership and dependencies

The coordinator owns canonical ledger/decisions and acceptance. Each later dispatch names
its document owner and permitted paths; none inherits product-code ownership. Source analysts
remain read-only unless assigned report-document ownership. Independent reads may share
callers, but overlapping recommendations return to the coordinator for reconciliation.

| Package | Owner and bounded responsibility | Required inputs/dependencies | Reviewable output |
| --- | --- | --- | --- |
| REF-ADMISSION | H01/H02/H03 comparison owner: REF1/REF2 admission, steering, requests, compaction and tool recovery | Accepted PLAN-009; current nanobot reports; DSH agent/inbox/compaction and Codex session/input/turn bodies | Mechanism rows tracing admitted work, source/ticket identity, ordering, cancellation, provider state and completion; governing contracts and uncertainty |
| REF-PERSISTENCE | H02/A01/S02/P01 comparison owner: REF3/REF5 writer, journal/process lifecycle, shutdown and receipts | Accepted PLAN-009; state/recovery and automation reports; DSH persistence/jobs and Codex recorder/process manager | Writer/ack/durability/shutdown map; blocking/buffering/retention expectations; process completion versus receipt/display truth; no measured-performance assertion |
| REF-AUTHORITY | S01/S02/E02/U01/R01 comparison owner: REF4 per-attempt authority and backend/platform selection | Accepted PLAN-009; grants/native/scope reports; DSH sandbox/policy and Codex orchestrator/sandbox manager | Config → admission → wrapper/native owner → API/UI outcome map for supported/unavailable paths; platform and trust decisions surfaced |
| REF-DELEGATION | T01/A01/K01 comparison owner: REF6 bounded roles versus addressable sessions | Accepted PLAN-009; current explicit manifest/loader findings; full caller reads of mapped DSH/Codex spawn/resume mechanisms | Identity/lifetime/authority/capacity comparison; preserved SDK/persisted obligations; no task-session design selection |
| REF-SYNTHESIS | Coordinator with source owners | All required comparison rows above; disagreements and shared boundaries | Deduplicated requirement-to-mechanism matrix, justified differences, candidate issues and open consequential choices; no premature severity or repair acceptance |
| ARCH-DISCUSS | Coordinator owns accepted decisions; human resolves material product direction | REF-SYNTHESIS with concrete options/tradeoffs | Platform/executor, task-session, compatibility-retirement and resource-policy decisions or explicit deferrals; accepted versus recommended state recorded |
| AUDIT-S02-01 | One later source-audit owner for Sandbox backend selection/admission/error invariant | Comparison/discussion prerequisites accepted; frozen source and decision record | Bounded audit with S01/E02/U01/R01 callers, concrete evidence/impact/certainty/disposition and smallest proposed correction; no product edit from this contract |

The first four packages may be sequenced or read in parallel only when explicitly authorized
and ownership is disjoint. This plan creates no agents, tasks or branches. REF-SYNTHESIS
joins required rows before architecture discussion; later audit packages depend on relevant
accepted decisions. Do not perform repairs while material behavior/authority/persistence
choices remain unresolved.

## Acceptance criteria and deterministic validation

1. Baseline commit/tree matches the recorded source; report/path hashes and memberships
   remain valid. Any source advance has a named differential checkpoint and affected
   caller reinspection rather than silently refreshed counts.
2. Every REF1–REF6 row traces actual pinned implementation and active callers/configuration;
   retained helpers, tests, documents or mapped references are not promoted to full review.
3. Each candidate issue has a requirement, concrete source evidence, conditions, uncertainty
   and disposition; differences without a governing requirement remain questions.
4. Shared writer/authority/lifecycle findings are reconciled once. Accepted decisions,
   recommendations and unresolved gates are distinguishable. Consequential choices have
   explicit acceptance or remain pending.
5. Next audit contracts name responsibility, owned documents, inputs/dependencies, acceptance
   boundary and source-only validation. Product repair contracts remain a separate step.
6. Document links resolve, current line anchors are within the pinned file, source references
   identify exact pins, and only explicitly owned audit documents change.

Use Git metadata/blob reads, targeted source reads, exact hash/list comparisons and
standard-library JSON/TSV/link parsing. A conventional red-green product test cycle is
inapplicable to source-comparison documentation; deterministic before/after evidence is
the compensating verifier. Record commands/results without executing repository scripts,
imports, tests, builds, checkers or dependencies. A documentation PASS does not establish
application behavior or a completed semantic audit.

## Rollout, rollback and open decisions

Publish only after coordinator acceptance of the concrete document package. Keep current
navigation, historical pins and supersession explicit. No runtime rollout, data migration,
configuration change or compatibility removal occurs. If a document is rejected, correct
the owned report/navigation and rerun affected integrity/link checks; restore the frozen
prior documentation if necessary while preserving source evidence and unrelated edits.
If staging advances, freeze a new pin and delta, carry only matching hashes, and reopen
affected comparisons. Do not absorb changed bytes into this baseline.

Open material choices are desired managed/executor platforms, persistent task-session
semantics, compatibility retirement/migration obligations and resource/capacity policy.
Additional requirement questions concern synthesized display completion, tool-error recovery,
cron status/manual admission, skills/asset/clipboard behavior and Android lifetime claims.
Gather discriminating source evidence before discussion; ask the coordinator/human when a
choice changes behavior, persistence, authority, public compatibility or rollout. No such
choice blocks completing this documentation-only contract.

## Requirement traceability

| Requirement | Evidence/output |
| --- | --- |
| Complete established-architecture discovery before comparison | PLAN-009, [coverage summary](coverage-summary.json), exact 139 closures |
| Compare mechanisms under the reliable personal-agent goal | REF-ADMISSION/PERSISTENCE/AUTHORITY/DELEGATION rows, AR1–AR9 and REF1–REF6 |
| Preserve consequential decisions for discussion | REF-SYNTHESIS → ARCH-DISCUSS decision record |
| Sandbox remains first detailed audit priority | AUDIT-S02-01 after prerequisite acceptance |
| Source-only, portable and independently verifiable evidence | [Methodology](methodology.md), pinned source links, exact lists/hashes and documented validation |
