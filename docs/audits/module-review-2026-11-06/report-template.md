# <STREAM_ID> — <module / responsibility>

Publication state: report-written only after supplied evidence is complete enough to
publish. Coordinator check, integration and Git checkpoint: pending unless the coordinator
supplies an explicit outcome. Record analyst identity, evidence handoff and report date.

## Pin, scope and evidence boundary

Record nanobot, DSH and Codex commit/tree identities; assigned primary targets; actual
bounded read ranges/symbols; caller-context/search/test/support reads; justified exclusions;
material unread gaps; and the paired coverage JSON. Hash/existence checks are mechanical
provenance and supply no semantic read credit. Do not assert exhaustive branches from counts.
Reconcile the named staging delta and retain the previous evidence's original pin.

## Responsibility and actual ownership

| Responsibility / entrypoint | Actual caller and activation | Owner / identity | State readers and writers | Resource lifetime, cancellation and errors |
| --- | --- | --- | --- | --- |
| <bounded source evidence> | <composition / defaults / optional path> | <one owner or competing owners> | <storage / ordering / recovery> | <create / release / partial failure> |

Assess dependency direction and the real boundaries, including public/SDK/plugin callers
and shared contracts. Name authority checks, durability/acknowledgement and compatibility
obligations. Record unknown deployment or external-caller facts as unknown.

## Architecture judgement and strengths

State the current judgement and whether it is provisional. Explain which observed
mechanisms should be retained and the invariants they protect. Separate source facts,
analyst interpretation, recommendations and accepted decisions. A module can be reviewed
with defects, deferred choices and explicit gaps; publication is not coordinator acceptance.

## Conditional findings

For each finding record an ID, classification, governing requirement, pinned source
location/symbol and actual call chain; concrete trigger/configuration; implementation reason;
impact; confidence and uncertainty; priority separately from confidence; smallest alternative;
shared dependencies; and whether a final-discussion choice governs a later repair.
Absence of a test, file length or a reference difference alone is not a defect.

## Smaller alternatives and simplification

Explain what can be removed, combined, moved, retired or retained, and the specific owner,
abstraction, writer, dependency or repeated work that would disappear. Trace registration,
callers/exports, persisted records and recovery before proposing retirement. Compare viable
smaller alternatives and their compatibility/authority/durability consequences. State expected
source-visible efficiency effects without measured speed, memory or stability claims.

## Prior architecture and change history

Record relevant commits/diffs/documented intent at their own pins, including earlier
architectural changes. Explain current callers and obligations before recommending preservation,
reversal or retirement. Distinguish verified history from inferred intent and retain uncertainty.
Do not silently refresh old evidence or infer that a historical repair is present now.

## Pinned reference mechanism

Use actual relevant DSH/Codex implementation bodies and callers, retaining commit/tree,
paths/ranges and mechanism differences. Connect required outcome → nanobot implementation
→ reference mechanism → defect or justified difference → smallest alternative. Explain
nonapplicability where no reference mechanism helps. No parity requirement or new Hermes study.

## Cross-module reconciliation and final questions

Name shared invariants, report dependencies, conflicts and material gaps. Retain questions
for the final consolidated discussion, mapping D1–D7 where relevant. Decisions are deferred
for review sequencing; inspection continues and no implementation choice is accepted.

## Validation and limits

List exact allowed Git and Python-standard-library metadata/hash/link/document commands,
outcomes and raw failures. No repository imports/scripts, tests/builds/type/lint/dependency
operations, apps/services/provider/sandbox launches, probes or benchmarks. Source inspection
does not attest installed configuration, runtime effects, OS enforcement or measured benefits.
