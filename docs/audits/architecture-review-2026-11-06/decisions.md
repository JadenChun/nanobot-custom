# Decisions and recommendations

This record separates accepted review policy from architecture recommendations and pending
product choices. Completing the documentation does not accept a new behavior, platform,
schema, migration, budget or compatibility removal. Coordinator review of this package is
also separate from human acceptance of consequential choices.

| Item | Status and accepted boundary |
| --- | --- |
| Review policy | **USER-ACCEPTED:** whole-project scope; a reliable personal agent in a clean, minimal, efficient codebase; source-only review; discovery → comparison/discussion → bounded audit; one owner per invariant; discuss consequential architecture before implementation. |
| Reference policy | **USER-ACCEPTED:** real pinned DSH/Codex source as engineering references, not identical products or feature-parity targets. Hermes is secondary product-direction awareness only, with no full code audit or mechanism evidence in this package. |
| Existing discovery | **COMPLETE at module/relevant-body depth:** 919 source targets, 224 current + 695 identical-hash carried, across the 2,455-path baseline. This review adds targeted mechanism comparisons, not new discovery counts or a complete semantic audit. |
| Core direction | **RECOMMENDATION:** retain AgentLoop/AgentRunner and canonical session/source ownership; correct narrow boundaries and keep optional capabilities at the edges. No new product-design acceptance is inferred. |
| Product choices below | **PENDING:** acceptance is not supplied by an analyst, reference implementation, documentation check or this publication. Explicit deferral permits source audit but does not authorize behavior-changing repair. |

<a id="d1-supported-executor-and-client-platforms"></a>

## D1 — Supported executor and client platforms

| Field | Record |
| --- | --- |
| Options | Keep the present managed executor limits and document them; add selected native executor capabilities; or use a selected remote Linux executor for unsupported clients. A client OS does not itself determine the executor host. |
| Evidence | [S3/S9](authority-sandbox.md): Linux/Bubblewrap managed shell, Linux native supervisor and SO_PEERCRED owner IPC are distinct current limits. DSH/Codex have platform-specific mechanisms with different partial/failure branches. [P2](persistence-lifecycle.md) adds lease/teardown obligations. |
| Consequences | Backend availability, process ownership, private owner authentication, lease semantics, packaging and truthful API/UI outcomes all need their own contract. Adding one backend does not establish all three capabilities. Weaker fallback would change authority. |
| Recommendation | First publish the reachable capability/error matrix in AUDIT-S02-01 and preserve fail-closed behavior. Choose required executor/client combinations before platform implementation; do not copy the reference platform list. |
| Acceptance/dependencies | **PENDING** human/coordinator decision or explicit deferral; S01/S02/R01/D01/E02/U01/Q01. No current documentation gate. |

<a id="d2-full-authority-and-private-boundaries"></a>

## D2 — Full authority and private boundaries

| Field | Record |
| --- | --- |
| Options | Full means normal OS-account subprocess access, with honest tool/instruction boundaries; or Full still confines selected private paths/network effects through an accepted native policy. Separately, retain volatile Full proof or design an explicit durable restoration mechanism. |
| Evidence | [S2/S6](authority-sandbox.md): nonserialized source/task-bound proof and volatile restart state; path tools retain private checks, while Full shell disables managed masks. UI describes normal OS access but context language can imply a stronger guarantee. Text path/URL checks cannot confine arbitrary computed effects. |
| Consequences | Native private-path confinement changes the security/product contract and platform requirements. Durable restoration changes persistence/recovery/consent semantics. Merely changing a label or restoring saved mode does not grant valid authority. |
| Recommendation | Preserve current private proofs and launch checks. Make intended normal-account versus retained-confinement semantics explicit before changing instructions; do not claim kernel isolation from text guards. Treat durable restoration as a separate decision. |
| Acceptance/dependencies | **PENDING**; S01/S02/H02/E02/U01/K01, with D1 for platform enforcement. No exploit or mandatory new confinement requirement is asserted. |

<a id="d3-managed-task-sessions"></a>

## D3 — Managed task sessions and delegation

| Field | Record |
| --- | --- |
| Options | Keep bounded foreground roles; add an explicitly specified addressable task/session capability; or retain the future direction without implementation. |
| Evidence | [REF6 comparison](delegation-context.md) distinguishes foreground role lifetime, retained compatibility managers and addressable reference sessions. Process waiting/notification is a different capability. |
| Consequences | Persistent identity, task/attempt/session ownership, direct user access, authority inheritance/revocation, follow-up routing, cancellation/recovery and provider capacity would need accepted contracts. A task catalog or public schema is not chosen here. |
| Recommendation | Retain bounded roles for current work and use the comparison to ask discriminating requirement questions. Stabilize the existing source/effect boundaries before selecting long-lived task semantics. |
| Acceptance/dependencies | **PENDING**; T01/A01/H01/H02/K01/E02/U01/P01. No reference feature-parity obligation. |

<a id="d4-compatibility-and-retirement"></a>

## D4 — Compatibility and retirement

| Field | Record |
| --- | --- |
| Options | Preserve retained APIs while clarifying activation; lazy-construct proven unused internals; or retire a capability through an accepted migration after caller/state obligations are known. |
| Evidence | [Delegation/context](delegation-context.md), [P8](persistence-lifecycle.md) and the [discovery activation map](../codebase-discovery/README.md#effective-capabilities-defaults-and-compatibility): absent core registration does not erase SDK exports, construction, config, persisted data or recovery obligations. Canonical manual Dream/goal/trigger paths and local queue activation have distinct fences. |
| Consequences | Deletion can break external callers, installed plugins, saved state or recovery even when a normal entrypoint is inactive. External deployment and caller presence were not observed. |
| Recommendation | Trace registration → callers/exports → persisted/recovery records, document effective activation, then propose the smallest cleanup with explicit compatibility disposition. No automatic removal based on omitted manifest entries. |
| Acceptance/dependencies | **PENDING** for changed public behavior/migration; T01/A01/H02/K01/E02/Q01. Source-only compatibility tracing can proceed. |

<a id="d5-request-tool-registration-lifetime"></a>

## D5 — Request/tool registration lifetime

| Field | Record |
| --- | --- |
| Options | Defer MCP reload until an established safe boundary; or bind each request to registration/connection identity and reject replacement, while keeping revocation live. |
| Evidence | [A3](admission-requests.md): captured definitions can overlap live replacement; shared and filtered registries have different stale/closed-wrapper paths. DSH validates current registration identity; Codex retains exact originating step bindings. |
| Consequences | Deferral affects reload responsiveness; binding requires connection lifetime/revocation ownership and compatibility with recovery. Static schemas alone do not bind execution identity. No authority bypass is demonstrated. |
| Recommendation | Audit actual reload/admission callers, then choose the smallest rule that preserves request identity and live revocation. Keep the rule in one T01/H01 owner rather than layering another core scheduler. |
| Acceptance/dependencies | **PENDING** implementation choice; H01/T01/E02. Separate repair contract after bounded source audit. |

<a id="d6-attempt-outcomes-and-recovery"></a>

## D6 — Attempt outcomes and recovery

| Field | Record |
| --- | --- |
| Options | Retain current compatible outputs with a small internal typed disposition/helper; or specify a broader public error/result contract separately if a demonstrated caller requires it. |
| Evidence | [S7](authority-sandbox.md) distinguishes backend/runner/child failure; [A7](admission-requests.md) identifies bounded recovery and scattered text exclusions. Mandatory result recording and security/unavailable stops remain strengths. |
| Consequences | Classifying every nonzero exit as an error changes ordinary shell semantics. Widening authority on retry changes security. A public taxonomy affects providers/tools/UI/plugins and is not justified merely by reference types. |
| Recommendation | Preserve raw outputs/legacy fields and use one internal attempt-to-recovery owner if the bounded audit confirms benefit. Keep denials, unavailable states, cancellation and ordinary child outcomes distinct; no automatic unsandboxed retry. |
| Acceptance/dependencies | **PENDING** internal behavior contract; S02/H01/T01/E02. No new public schema or configuration knob is accepted. |

<a id="d7-completion-and-resource-policy"></a>

## D7 — Completion and resource policy

| Field | Record |
| --- | --- |
| Options | Define completion output as unread tail, bounded transcript or summary; define inferred display completion separately from canonical terminal/receipt proof; preserve or deliberately change retention/admission/capacity policy after the required view is known. |
| Evidence | [P4–P6/P9](persistence-lifecycle.md): drained output may already exist in tool history; journal caps safely refuse before launch; settled/unknown records differ; transcript repair synthesizes presentation completion. [REF6](delegation-context.md) adds separate role/model capacity. |
| Consequences | More retained output/records increase privacy/storage cost; reclamation must not discard uncertain effects. Public completion/provenance changes affect client expectations. Wider parallelism is not automatically faster or cheaper. |
| Recommendation | State each promised view and owner first. Keep bounded output, conservative unknown-effect recovery and current refusal. Split settled inspection/export/reclaim, lock narrowing and capacity decisions into separate contracts. |
| Acceptance/dependencies | **PENDING**; A01/S02/H01/H02/T01/E02/U01/P01. No new counts, budgets, eviction policy or public payload chosen. |

## Evidence and implementation gates

Mandatory source-writer failure propagation (A8), yielded hard deadline ownership (P3)
and allowed-version GitStore index scope (P8) have concrete conditional source chains.
Their [backlog contracts](audit-backlog.md) still require bounded caller reconciliation
before a repair. Reference mechanisms do not determine their nanobot-specific obligations.
Source-audit documents may record an explicit decision deferral; consequential repair
remains blocked only on the decision actually governing that repair.

No pending choice blocks completing this documentation package. New product work must
receive a separate contract/branch with exact writes, accepted behavior, compatibility,
focused checks and rollout/rollback; no code changes follow automatically from this record.
