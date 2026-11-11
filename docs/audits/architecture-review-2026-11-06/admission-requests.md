# Admission, requests and context consistency — REF1/REF2

The reviewed paths support retaining canonical admission, distinct execution/source/request
identities, mandatory result recording and portable source checkpoints. The strongest
conditional correction candidate is failure propagation from a mandatory source follow-up;
live tool registration also needs an explicit lifetime contract. Stream interruption and
broader parallel tool execution remain product choices.

These are targeted source comparisons at the [three frozen pins](README.md#source-pins).
Line spans below name relevant bodies, not whole-file semantic coverage. Exact body-read
and search-only paths are separate in [provenance.json](provenance.json). Tests mentioned
were read as source and never run. Coordinator acceptance is separate from this report.

## A1 — Preserve canonical admission and distinct identities

| Field | Comparison |
| --- | --- |
| Required outcome | One live execution owner for accepted canonical work; stale or recovery-held work cannot write or execute. |
| Nanobot implementation/configuration | [AgentLoop.execute_turn](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/loop.py#L3487) reaches `_admit_after_bus_cleanup` and `_source_execution` (3559–3579); canonical admission binds source/execution ownership (3322–3339, 3384–3386). `SessionManager.source_backend` (manager.py:2955) selects canonical versus narrower ephemeral/legacy paths. |
| Reference mechanism | DSH [ReactLoopAgent phase driver](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/core/agent-loop/src/agent.ts#L230) serializes phases (53–61, 327–432), constructed by `index.ts:572–576`. Codex [Session](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/session/session.rs#L1891) owns locks/input queue, with an atomic active-turn check in `turn_input.rs:869–911`. |
| Writer/identity/lifetime/authority/error boundary | [ExecutionOwner validation](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/session/execution_owner.py#L78) binds store, session, incarnation, generation, lease, PID and task; closing/recovery paths reject validation (78–106, 131–135). These identities have different lifetimes. |
| Actual difference/disposition | **Preserve.** The mechanisms differ, but each has a single admission/iteration owner. DSH/Codex do not establish nanobot's specific source-store obligations. Collapsing identities would discard fences, rather than prove simplification. |
| Conditions/impact/confidence/next step | High source confidence for the named canonical path. Preserve it in H01/H02 audits; inspect alternate entrypoints separately. No replacement admission layer is recommended. |

## A2 — Steering boundaries are an open interaction contract

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Keep follow-ups ordered and bounded. Must a new message interrupt an active provider stream, or is the next safe model/tool boundary sufficient? This requirement is pending. |
| Nanobot implementation/configuration | [Pending routing](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/loop.py#L2449) covers 2449–2689; WebUI steering rejects stale identity (2139, 2485, 2853), caps pending messages at 20 (2714), and republishes retained work (2753–2768, 2840–2850). [Runner drains](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/runner.py#L1217) occur after tools and before final return (1343–1370), bounded to three messages per drain/five cycles (135–136). This path does not interrupt an active stream. |
| Reference mechanism | DSH [followup/steer/inject](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/core/agent-loop/src/agent.ts#L170) separates next-turn, next-step and non-waking injection; `inbox.ts:108–112,185–230` records, then wakes, and prioritizes next-step input. Those calls alone do not prove fsync durability. Codex [turn input](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/session/turn_input.rs#L641) checks active state (859–950); `InstantInterrupt` is conditional (`session/mod.rs:4233–4237`). `turn.rs:2843–2879` drains/drops transport and retains a partial reconciliation TODO. Its default feature state was not established. |
| Writer/identity/lifetime/authority/error boundary | Nanobot pending-journal/source identity and runner drain budgets differ from transport cancellation. Retained input and republish are not evidence of immediate consumption or response. |
| Actual difference/disposition | **Product discussion.** Boundary steering may already satisfy the personal-agent requirement. Reference interruption behavior is not a parity target or evidence that nanobot loses input. |
| Conditions/impact/confidence/next step | High confidence in these boundaries; user interaction requirements remain unknown. Decide whether interruption is needed before designing transport cancellation/reconciliation. Preserve bounded drains and stale-source checks. |

## A3 — Bind advertised tools to an execution registration lifetime

| Field | Comparison |
| --- | --- |
| Required outcome/open question | A request's advertised tool contract must correspond to the implementation later called, while revocation remains live. Which owner should serialize or reject replacement during active work? |
| Nanobot implementation/configuration | [Runner schema capture](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/runner.py#L1663) precedes awaits and send (1790–1805). [ToolRegistry.prepare_call](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/tools/registry.py#L246) resolves current `self.get(name)`; definition caching is separate (222–244), context/parameter validation is at 280–286, and invocation is in `execute_prepared` (339–344), reached through `execute` (431–454). Live gateway [MCP reload](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/cli/gateway_runtime.py#L396) is reachable from `settings_routes.py:545–555`. [MCP replacement](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/tools/mcp.py#L1747) unregisters/closes then replaces (1747–1784; recovery 1854–1904). No shared turn-admission lock was found. |
| Reference mechanism | DSH [freezes a resolved tool set](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/core/agent-loop/src/agent.ts#L287) (287–312, execution 590–592) and rejects `agent/request` hook divergence from the originating step's already fixed provider/model/effort (635–639). [AgentModel.select](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/core/agent/src/model-selection.ts#L112) appends a selection without an active-work guard (112–115); this is step/request consistency, not a global freeze while active. `ResolvedToolSet.view` [checks current registration identity/definition](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/core/tools/src/index.ts#L264), so removal/replacement cannot be resurrected. Codex [StepContext](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/session/step_context.rs#L22) plus `session/mod.rs:4055–4072,4211–4248` freezes settings/router/exact MCP bindings; `tools/parallel.rs:49–54,176–180` retains the originating step. Current grants remain separate (step_context.rs:61–76). |
| Writer/identity/lifetime/authority/error boundary | Some nanobot turns share `self.tools` (`loop.py:2217,3678–3679`). Filtered registries copy names while retaining tool objects (`registry.py:473–481`, `loop.py:4185–4230`); artifact setup may create a per-run registry (`runner.py:573–611`). Thus shared paths may execute replacement B after advertising schema A, while filtered paths may retain a closed wrapper. Registry and connection lifetimes both matter. |
| Actual difference/disposition | **Conditional improvement candidate C3.** High static possibility on concurrent reload; actual frequency is unverified. No authority bypass is demonstrated. A schema snapshot alone does not settle executable/connection identity. |
| Conditions/impact/confidence/next step | Active request overlapping MCP reload/recovery is the trigger; possible schema mismatch or closed-wrapper failure. Choose boundary-deferred reload or request-bound registration identity with replacement rejection, retaining live revocation. H01/T01/E02 own the shared invariant together; [D5](decisions.md#d5-request-tool-registration-lifetime) remains pending. |

## A4 — Preserve source provenance and acknowledgement ordering

| Field | Comparison |
| --- | --- |
| Required outcome | Acknowledged source input must correspond to a usable model result that has passed mandatory recording. |
| Nanobot implementation/configuration | [SourceRequestLedger](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/session/source_request.py#L47) separates source/runtime notes (47–60,129–155); tickets bind epoch/exact source (157–167), acknowledgement binds the exact pending ticket (169–175), replacement invalidates it (183–188), and eligibility uses acknowledged IDs (125–127). [Runner records before acknowledgement](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/runner.py#L1606) (1606–1627), discards on write failure (1616–1620), and invalidates cancellation/error paths (1590–1604,1812–1815). |
| Reference mechanism | DSH [request/assistant record ordering](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/core/agent-loop/src/agent.ts#L726) (726–742,574–592) freezes before the batch. Codex's originating step consistency is analogous, not a specification of nanobot's ledger. |
| Writer/identity/lifetime/authority/error boundary | Source ticket epoch/exact input and mandatory source writer have different roles. A provider response alone is not an acknowledgement. |
| Actual difference/disposition | **Preserve.** Nanobot-specific provenance is a strength and must survive any request or registry correction. |
| Conditions/impact/confidence/next step | High confidence in the traced ordering. H02/H03 audits should preserve failure-before-ack behavior rather than substitute reference data models. |

## A5 — Preserve portable checkpoint and compaction ownership

| Field | Comparison |
| --- | --- |
| Required outcome | Request context must fit the selected provider while preserving canonical source/tool provenance and committing only a valid replacement. |
| Nanobot implementation/configuration | [Request governance](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/context_governance.py#L2180) constructs source view/checkpoint/artifact/image/compaction/final-fit (2180–2275), copies messages/schemas and validates ownership (2239–2260); source checkpoints disable opaque resume. Capture/commit bind epoch (432,523–528), dependencies (547–556) and fit-before-commit (568–610). Pressure/target are 90%/60% (`loop.py:2341–2351`, governance.py:286–289). |
| Reference mechanism | DSH [compaction region](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/compaction/compaction-basic/src/region.ts#L206) checks stability, shrink and provenance (206–275,417–420,430–508); `index.ts:213–256` bounds overflow/replacement progress. Codex [turn compaction failure](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/session/turn.rs#L198) still records input (198–241) and consumes steering after compaction (698–737). |
| Writer/identity/lifetime/authority/error boundary | Canonical source and detached provider context differ. Epoch/dependency checks protect checkpoint commit; an opaque provider continuation is not universally portable source truth. |
| Actual difference/disposition | **Preserve; bounded adapter audit queued.** The comparison supports explicit replacement validation, without proving every provider's final transport adapter. |
| Conditions/impact/confidence/next step | High confidence for named governance bodies; provider serialization/continuation variants remain H03 work. Do not add a second compaction owner or assume all provider adapters were reviewed. |

## A6 — Preserve conservative effect boundaries before replanning

| Field | Comparison |
| --- | --- |
| Required outcome | Completed effects and undispatched tail calls must be recorded accurately before recovery or replanning; cancellation must not create blind replay. |
| Nanobot implementation/configuration | Production enables fail-on-tool-error and concurrent flags (`loop.py:2327,2335`), but [execution](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/tools/execution.py#L135) disables parallel dispatch for source context, policy, fail-fast or mandatory callbacks (143–145), checks policy before effects (135–141), and records undispatched tail as skipped (169–225). [Turn writer callbacks](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/session/turn_writer.py#L221) bind per-ordinal execution identity (221–294). Runner appends results/checkpoints before recovery guidance (1001–1096); it does not replay the original tail. |
| Reference mechanism | DSH [rolling pool/exclusive barriers](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/core/tools/src/batch.ts#L50) replenish after ordinary failure (50–95); cancellation drains (219–229), while `session/repair.ts:105–196` separates started-unknown from unstarted work. Codex [parallel tools](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/tools/parallel.rs#L129) distinguishes ordinary feedback/fatal errors (129–141,285–302) and uses read/write locking. |
| Writer/identity/lifetime/authority/error boundary | Nanobot's source result writer, call ordinal and pre-dispatch policy govern safe progress. An enabled concurrency flag is not proof that a source/policy-bound batch executes concurrently. |
| Actual difference/disposition | **Preserve; optional local clarity improvement.** Broader concurrency is an unresolved capability/resource choice, not an automatic efficiency gain. |
| Conditions/impact/confidence/next step | High confidence for effective serialization and recording. Clarify the flag/effective-condition distinction locally if useful; behavioral parallelization requires an accepted effect/recovery contract and A01/S02 dependencies. |

## A7 — Make recovery disposition explicit inside the owner

| Field | Comparison |
| --- | --- |
| Required outcome | Allow bounded correction of ordinary tool mistakes while stopping security, unavailable or repeated failures. |
| Nanobot implementation/configuration | [Runner recovery](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/runner.py#L959) defaults to two recoveries (225–228); repeat signatures use name/arguments/error rather than call IDs (150–167). Exclusions include preparation, policy, security, unavailable and Dream (959–985). Trusted pre-dispatch recovery is separate (`loop.py:4562–4570`, execution.py:429–439). Registry schema/shape failures are generic `ToolResult.error` (282–297); unknown/withheld paths are typed (254–275). |
| Reference mechanism | The DSH batch and Codex parallel mechanisms in A6 distinguish continuation/fatal/cancellation boundaries differently. They do not require a nanobot public result-schema change or a new configuration knob. |
| Writer/identity/lifetime/authority/error boundary | Recovery belongs after mandatory result/checkpoint recording. Error classification must preserve security denial and fail-closed availability behavior rather than rely on broad retry. |
| Actual difference/disposition | **Bounded improvement candidate C6.** A small internal typed disposition/helper could consolidate scattered text conditions while retaining `ToolResult` compatibility. Deliberate model correction of schema mistakes can remain valid. |
| Conditions/impact/confidence/next step | High confidence in current bounded behavior; maintenance benefit is source-based, unmeasured. `tests/agent/test_runner_errors.py:334–451` expresses ordinary recovery/repeat-stop/bounds but was only read. Review with S7's attempt-outcome contract; no new public taxonomy is selected. |

## A8 — Propagate failure of mandatory source follow-up consumption

| Field | Comparison |
| --- | --- |
| Required outcome | Failure of the active mandatory source writer must stop provider continuation and successful completion on that writer. Optional/legacy callback failures may retain their established best-effort behavior. |
| Nanobot implementation/configuration | The [source injection callback](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/loop.py#L2118) consumes pending follow-up (2118–2133); conversion retains failed input for republish and rethrows (2146–2151). [SourceTurnWriter](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/session/turn_writer.py#L140) marks the shared writer failed before raising (140–169; failed semantics 69–76). [Runner callback handling](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/runner.py#L503) catches `Exception` and returns `[]` (503–518); `_try` treats it as no injection (428–429), and the after-tool path continues (1218–1225). |
| Reference applicability | A1/A4 establish the general admission/request boundary. This is a **nanobot-specific mandatory-writer obligation**; neither reference is claimed to prove its exact failure semantics or supply a matching defect. |
| Writer/identity/lifetime/authority/error boundary | Ticket issuance validates projection, not `writer.failed` (`source_request.py:157–167`). With artifacts disabled by default (`schema.py:929–932`), below pressure (`context_governance.py:821–826`) and no artifact projection (395–398), later governance need not fence the failed writer. Live-owner checks (`context_governance.py:2248`, execution_owner.py:86–106) are a different invariant. Owner transaction/source-store paths (execution_owner.py:109–123; source_store.py:1693–1743) do not automatically mark recovery for this before-write error. |
| Actual difference/disposition | **Static conditional defect candidate C2.** A transient **before-write** consumption failure can be swallowed, allowing another request (`runner.py:1790–1810`) on the old ledger. Later nonempty model/tool intent fails mandatory recording before local effects (1546–1557,1606–1620). A final-response drain (1346–1360) can still return completed (1455–1465,1518–1525) if later metadata/cleanup succeeds: canonical checkpoint callback is a no-op (`loop.py:2023–2025`), metadata persistence lacks the failed-writer check (4607–4642), and result mapping accepts completed (3692–3697). |
| Conditions/impact/confidence/uncertainty | High static call-chain confidence, no runtime verification. Persistent storage failure, owner revocation, artifact checks or required compaction can block continuation. **No lost-input or replay claim:** journal retention and republish protect the input. `tests/session/test_canonical_followups.py:146–161` was read and covers retention/failed flag, not the integrated runner path. |
| Smallest correction/dependencies | In a separate H01/H02 repair contract, propagate callback exceptions when they fail the active mandatory source writer, preserving optional/legacy callback behavior; consider a defensive writer-health check before provider send/success. First finish the bounded source-audit check of caller variants. Do not broaden this into a new callback framework or change acknowledgement/compaction owners. |

The deduplicated priorities and shared dependencies are in [README.md](README.md#improvement-roadmap)
and [audit-backlog.md](audit-backlog.md). No application behavior was changed or qualified.
