# Architecture planning review and accepted direction

**Status: platform scope, architectural direction and four first-release defaults
accepted by the user. Frozen as `A3-PLAN-2026-11-11` for implementation review.**

This review concerns intended architecture, reasons, alternatives and work order, against
staging `ce8fb74f41e8d4fe7dcd1b5f7740ba29a68a2dff`. Inputs are the existing audit/design
records, pinned DSH/public Codex reference source and official desktop documentation.
[Reference answers](reference-design.md) distinguishes observations from adaptations.
No nanobot implementation source inspection or product-test execution belongs to this
planning review.

The required order is: settle and record the architecture/plan, then check the separate
implementation against that frozen plan and latest staging. Documentation structure,
runtime test results and user selection of requirements are separate facts.

## User acceptance

On 2026-11-11 the user accepted the preceding recommendations: "yes the recommendation
direction is ok". This accepts the six D2–D7 directions below and the recommendation to
preserve the existing Full-consent restart behavior. Together with the earlier D1
selection, these are the governing architectural direction for the plan.

The acceptance covers permission separation, canonical child sessions, request-owned
bindings, owner-local recovery, bounded output with separate durable retention, and
versioned compatibility with explicit migrations. That initial acceptance did not select
previously unspecified child controls/budgets, Android protocol changes or retention values.
Implementation acceptance and the final detailed-plan freeze were separate subsequent steps.

The user subsequently selected a first-release baseline without legacy migration.
[The first-release product contract](v1-product-contract.md) records that selection and
the subsequently accepted child, resource, initial device-proof and retention defaults.
Historical upgrade/cutover criteria do not create a released-user population.

The user also clarified that "child" means a subagent controlled through the original
chat/main agent, and questioned direct child management. The Codex/OpenCode
[source follow-up](reference-design.md#d3-follow-up-main-chat-control-in-codex-and-opencode)
supports narrowing v1 to main-chat-mediated control and optional read-only inspection.
The earlier separate public child-control proposal is superseded.

The user then approved that interaction direction and explicitly required reference-aligned
client UI behavior. The [client UI contract](v1-product-contract.md#client-ui-behavior-for-subagents)
records original-chat activity/results, optional read-only detail with return navigation,
main-chat follow-ups/approvals, subtree Stop and canonical reconnect behavior across
desktop/WebUI, TUI and Android remote. That UI decision selected the interaction model;
the numerical, lifetime, initial proof and retention defaults were selected later.

Before freezing, the user required awareness of future coding-harness and SuperAgent
implementation, then clarified that SuperAgent coordinates sessions and must not change
ordinary subagent behavior. The [extension contract](future-harness-extension.md) records
these separate layers, existing foundation/stage boundaries and future acceptance
scenarios. This is a design constraint on extensibility, not authorization to enable
session orchestration or to change subagent behavior.

The user then chose step 1: "ok lets do 1". This explicitly selects the
[four final defaults](v1-product-contract.md#final-selections-for-plan-freeze): awaited
subagents with explicit internal background work; configurable depth 2 / two immediate
children / four descendants / four gateway-wide child runs; the initial Android
Keystore P-256 proof contract; and regular history retained until explicit deletion with
existing finite output/admission limits and unresolved evidence preserved. Product
selection was complete. The user then said "ok lets do it", authorizing step 2:
freeze and commit the documentation, then merge/push it directly to staging. The
[frozen baseline](#frozen-baseline-and-subsequent-work) below records that decision.
Implementation review remains the next phase.

## Selected platform scope: D1

The user selected this scope on 2026-11-11:

| Required platform | Role | Current verification record | Remaining work |
| --- | --- | --- | --- |
| Linux | Executor/gateway | Previously verified, confirmed by the user. Historical receipts retain their original source/environment scope. | Reuse matching evidence later; requalify changes that invalidate it. |
| macOS | Executor/gateway | Remaining verification target. | Verify the selected contracts on an actual macOS host in the later implementation phase. |
| Android | Remote client of an admitted gateway | Previously verified, confirmed by the user. Historical receipts retain their original client/executor/artifact scope. | Reuse matching remote-client evidence later; requalify affected changes. |

Windows is outside this review's required qualification matrix. Existing compatibility
is preserved unless separately changed. A cancelled historical Windows job remains a
historical result; it is not a new platform requirement for this plan.

Client support and executor guarantees stay separate. The remaining macOS verification
must cover the following planned contracts, with exact host/source identities later:

| Contract | Planned macOS acceptance scenario |
| --- | --- |
| Local use and canonical sessions | CLI/SDK/gateway and desktop/WebUI entry points admit work through the same session/source owner; restart preserves readable history without replaying uncertain effects. |
| Selected workspace/Once/Full modes | Declared path/network enforcement matches actual effects. Missing required enforcement refuses explicitly; Full accurately states its OS-account authority. DSH/Codex Seatbelt is a native design reference, not evidence about nanobot. |
| Process ownership | Hard deadlines work without polling; Stop/shutdown settle owned descendants or preserve visible uncertainty. Root exit alone is insufficient. |
| Local owner authentication | Only the authorized local owner mutates execution/source authority through an actual macOS-supported peer-authentication boundary. |
| Android remote interaction | A remote request targeting a macOS executor uses the same scope, consent, cancellation and session rules; the phone does not become the executor. Existing Linux/Android evidence does not establish this particular pairing. |

These are planned scenarios. This review has not executed them.

## Architectural direction

The original small-core direction and stage dependency structure are coherent. Retain
AgentLoop/AgentRunner and capabilities at existing channel/tool/provider/client edges.
The reference study supports these target contracts:

1. One canonical session/source authority, including delegated subagent execution history.
   Required recording precedes dependent effects
   and acknowledgement; client projections do not become another history authority.
2. Distinct permission profile, approval scope and concrete OS enforcement. An effect
   uses its prepared target and live authority; delegation only narrows authority.
3. Distinct durable session/child identity and active run/process lifetime. Bound
   admission before launch; define Stop, parent completion, restart and uncertainty.
4. Request-owned tool definitions, bindings and resource lifetime through settlement,
   with current revocation checks and ready replacements for future requests.
5. Owner-local effect/recovery outcomes, separate from transport acknowledgement.
   Reconcile uncertain effects before replay; preserve compatible raw results.
6. Bounded live output and explicit completion provenance, with durable evidence and
   unresolved effects governed by separate retention rules.

This uses existing owners and narrow adapters. Future coding-harness capabilities and
coordination between normal sessions extend these edges while ordinary subagents retain
their normal behavior. A second execution-history authority, global generation service
or generalized release platform is not selected by that direction.

## D2–D7 direction from the references

The user authorized using DSH and Codex/ChatGPT desktop design as references and has
accepted the resulting architectural directions below. The later four-default selection
completes the current product choices; neither decision establishes implementation conformance.

| Decision | Accepted architectural direction | Rationale and selected defaults |
| --- | --- | --- |
| D2: permissions | Separate sandbox/profile, Once versus standing consent and actual backend facts. Recheck live authority per effect. Keep Full consent volatile; loading history does not restore it. | DSH persists session policy; desktop docs separate sandbox and approvals. Persisted policy cannot revive old runtime grants. Describe Full's actual OS-account authority and any separately enforced private-path exclusions. |
| D3: delegation | Use canonical subagent sessions with separate run identities, narrowed policy and bounded admission; reuse the normal runner/history path. The main agent owns control through the original chat. | Codex/OpenCode supports internal lifecycle tools and optional inspection. Awaited default, explicit background work and finite 2/2/4 plus four gateway-wide child runs are now selected in the product contract. No separate public child management interface. |
| D4: compatibility | Version the first release's protocol/state explicitly and use one live owner. The user selected no legacy migration for this first release. | Do not require a v1-to-v2 cutover, forced re-pairing, historical executable reader or two-version upgrade rehearsal. Initial device proof, restart recovery and revocation still need their normal contracts. |
| D5: request lifetime | Capture exact tool/MCP bindings for each sampling request, retain resources through settlement, and keep revocation live. Publish ready replacements for new requests. | DSH validates registration identity; Codex captures exact MCP/router state. A global reload pause is not required by this contract. |
| D6: recovery | Preserve raw compatible outputs with small owner-local typed dispositions; distinguish never-started, applied, partial and uncertain effects from acknowledgement. | DSH explicitly distinguishes unstarted and outcome-unknown calls. Ordinary feedback can permit bounded recovery; cancellation/denial/mandatory-write failure and uncertain replay need distinct handling. |
| D7: output/resources | Separate bounded polling output, completion transcript, execution capacity and durable retention. Expose loss/truncation and retain unresolved evidence. | Preserve active finite output/admission limits. The user selected no automatic age expiry for regular history, explicit deletion and retention of unresolved proof. Child budgets are selected separately from provider capacity. |

The detailed facts, source links, design adaptations and limits are recorded in
[reference-design.md](reference-design.md). The earlier proposal to preserve only
existing features and defer all new child/protocol scope is superseded as a blanket
recommendation. Each affected capability now gets an explicit planning disposition.

## Work order and planning prerequisites

The existing A3 labels retain audit traceability. This table defines intended sequencing;
it does not declare a stage implemented or accepted.

| Stage | Intended contract | Planning prerequisite |
| --- | --- | --- |
| A3-0 | Freeze latest staging identity, selected modes, outputs/limits and evidence method. | D1 selected; remaining contracts affecting the next slice recorded. |
| A3-1 | Effective target/instance facts, mandatory source admission and owner-local recovery. | One owner per invariant; pre-effect refusal and uncertain-outcome rules. |
| A3-2 / A3-3 | Process/deadline/termination and canonical session/incarnation ownership. | A3-1; Linux/macOS contracts explicitly separated from Android client behavior. |
| A3-4 | Exact-effect consent, queued admission, Stop and reconnect/readback in the existing client surfaces. | Process/session contracts and approval/cancellation race walkthroughs, including child effects shown in the original chat. |
| A3-5 | Canonical subagent execution/source, narrowed policy, parent-mediated lifecycle and main-chat activity/detail UI. | A3-4; selected D3 runtime lifetime/defaults and the client UI contract. No separate public child management interface. |
| A3-6 | Request-bound tool/MCP readiness, provider terminal/fallback and credential lifetime. | D5 contract; concurrent capture/reload/revocation walkthroughs. |
| A3-7 | Initial Android-remote authority, device-key pairing, native cancellation and the same main-chat subagent UI semantics. | D1/D2/D4, the initial proof and client UI contracts; no legacy cutover/re-pairing requirement. |
| A3-8 | Retained evidence, continuation identity, deletion completeness and safe maintenance preview. | Source/process/request producers identified; selected output/retention policy. |
| A3-9 | Supported-target, first-release artifact and restart/rollback qualification. | Linux/macOS executors and Android-remote scope; new proof focuses on macOS, with matching earlier receipts reused appropriately. No historical-release or two-version upgrade requirement. |
| A3-10 | Pre-release cleanup of duplicate/obsolete paths. | Current callers/state and one live owner; no retained adapter solely for an unreleased legacy format. |

Retain all 136 original finding IDs and the separate 26 E01 findings, including their
coverage qualifications. Each needs a planned treatment and observable criterion for its
selected scope. Deferral does not close or erase a finding. Per-finding implementation
acceptance remains UNVERIFIED; it is not the measure of whether planning discussion is
complete. Historical audit packets remain byte-exact when the current plan is amended.

## Frozen baseline and subsequent work

On 2026-11-11 the user authorized formal freeze and direct publication to staging after
selecting all four defaults. **`A3-PLAN-2026-11-11` is the frozen architecture and work-order
baseline.** Its exact revision is the Git commit that first introduces this freeze record
into staging. The commit's parent is pre-refactor staging
`ce8fb74f41e8d4fe7dcd1b5f7740ba29a68a2dff`; later implementation review also compares
against latest staging so subsequent repairs are included. No documentation PR is required.

This freeze covers the selected D1–D7 direction, v1 scope/defaults, main-chat subagent UI,
owner/failure walkthroughs, A3 stage contracts and future coding-harness/session-based
SuperAgent extension boundaries. The matching machine-readable record is
`planning_review.document_freeze` in [the evidence ledger](evidence.json).

The [final selection table](v1-product-contract.md#final-selections-for-plan-freeze)
now records all four user-selected defaults and their reasons. No product selection
remains open for this v1 scope. Legacy migration remains outside scope, Full consent
remains volatile and future SuperAgent session coordination stays separate from ordinary
subagent behavior.

The [owner and failure walkthroughs](v1-product-contract.md#owner-and-failure-walkthroughs)
now record approval races, cancellation, restart, reload, lost acknowledgement, capacity,
storage pressure and rollback on paper, with owners, permitted effects, durable outcomes,
visible results and forbidden replay. These are design reasoning and later test criteria,
not execution evidence. Review the separate implementation against this frozen baseline
and latest staging in the next phase. Initial fixed protocol schema/byte vectors remain
a technical deliverable before the device-proof boundary review.
The [future extension stage mapping](future-harness-extension.md#mapping-to-a3-and-later-review)
adds compatibility criteria for that later code review. It does not require future task
creation/routing/UI features to be built before accepting the current refactor.

This freeze does not close findings or attest runtime behavior. macOS qualification and
matching implementation evidence remain later work. Changes to these selected contracts
require an explicit decision update and a new documentation baseline.

## Review provenance

Coordinator planning review dated 2026-11-11, with subsequent user acceptance of the
architectural direction and the later explicit four-default selection recorded above.
Those are user decisions; the reference rationale uses the pinned comparisons and official
documentation recorded in [the evidence ledger](evidence.json). This is not an independent
implementation review, a product-parameter selection beyond the stated scope, or runtime
acceptance.
