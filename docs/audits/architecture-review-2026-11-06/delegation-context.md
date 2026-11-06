# Delegation, context and future sessions — REF6

Retain bounded awaited roles while correcting narrow context/authority boundaries. Current
foreground roles are not independently addressable persistent child sessions. The strongest
current candidate is foreground skill admission; selected-project write scope also has a
fail-closed usability mismatch. A minimum governing-instruction contract needs discussion.

This is the targeted REF6 body/caller comparison at the [frozen pins](README.md#source-pins),
superseding the earlier **mapped-only REF6** status for this mechanism slice. It does not
turn complete discovery into a full semantic audit. Exact body/search-only paths are in
[provenance.json](provenance.json); no tests ran and no Hermes source audit was performed.

## G1 — Preserve bounded roles versus addressable session distinction

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Keep current delegation ownership explicit. Does the product require independent addressable task sessions, or are awaited roles sufficient? Persistent task/session semantics remain pending. |
| Nanobot implementation/configuration | [ForegroundAgentManager](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/delegation.py#L122) runs role tools inline/awaited (`fork_factories.py:85–90`). `_run_role` (485,509–545) records logging call ID, child source stream and explicit artifact grants, rather than an independent resumable session. |
| Reference mechanism | DSH [startLocal](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/subagent/subagent/src/manager.ts#L302) retains durable childSessionId, parent catalog, ancestry and accepted input (302–414), separately from activation runId (844–875). Codex [multi-agent v2 spawn](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/tools/handlers/multi_agents_v2/spawn.rs#L221) selects history (221–233); control/spawn.rs:609–868 reserves capacity, creates, persists the edge and admits input before commit. |
| Writer/identity/lifetime/authority/error boundary | A foreground call ID/source substream belongs to the outer turn. DSH durable identity/transient activation and Codex persistent identity/lazy residency are different ownership models. Neither establishes nanobot's future task schema. |
| Actual difference/disposition | **Preserve current scope; product discussion D3.** Call ID must not be promoted to persistent identity. Waiting for a process and owning a model session are separate capabilities. |
| Conditions/impact/confidence/next step | High source confidence. Keep awaited roles during stabilization; an addressable capability would need accepted identity, human-control authorization, ordered input acknowledgements, persistence, stop/recovery and capacity contracts. No framework is selected. |

## G2 — Align admitted project root with worker scope and tool resolution

| Field | Comparison |
| --- | --- |
| Required outcome | A worker's declared relative write scope and file-tool targets must use the same admitted project root, without weakening path/grant containment. |
| Nanobot implementation/configuration | Loop constructs the manager from `AgentLoop.workspace` (`loop.py:665–670`, foreground.py:52–68). [run_worker](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/delegation.py#L670) normalizes scope against `self.workspace`; [WriteScope.from_raw](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/write_guard.py#L18) rejects absolute/outside-base paths (18–36). File tools resolve against selected `scope.project_path` (`filesystem.py:266–280`, workspace_access.py:443–457) and apply final scope after policy (`filesystem.py:355–385`). Worker registry is narrow read/list/write/edit (delegation.py:415–459); write_guard.py:66–87 locks individual in-process mutations, not task ownership. |
| Reference mechanism | DSH [child policy capture](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/subagent/subagent/src/child-agent.ts#L249) (249–279) and Codex [child config](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/agent/child_config.rs#L84) reapply authority after role configuration (84–85; role.rs:91–126 is disable-only). An equivalent nanobot assignment-scope mechanism was not established. |
| Writer/identity/lifetime/authority/error boundary | For an authorized selected project **outside manager home**, `write_scope=['src/file.py']` covers home/src/file.py while the tool resolves project/src/file.py. Final scope denies it. Focus/project selection, policy authority and exact write scope are separate boundaries. |
| Actual difference/disposition | **Conditional fail-closed usability candidate C15.** High static confidence; no execution or write escape is demonstrated. |
| Conditions/impact/confidence/next step | Use one admitted project root for prompt/scope/tool resolution, retaining policy, grants, symlink checks and non-elevatable roles. Establish stable-or-reject focus-mutation semantics with REF4 before a repair; do not broaden containment to make the operation pass. |

## G3 — Define the minimum governing instructions for a role

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Which admitted project/governing constraints must every role receive, and with what precedence? Absence of all personal-profile text is not itself a defect. |
| Nanobot implementation/configuration | [Role base prompt](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/delegation.py#L286) includes role/home/catalog/skills/assignment (286–305), rather than main bootstrap/profile/project-instruction bodies. Main context uses separate assembly (context.py:149–252), checked project AGENTS (197–206) and profiles (379–420). |
| Reference mechanism | DSH can spawn fresh or [fork through the last completed turn](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/subagent/subagent-fork-in-process/src/index.ts#L46) (46–52,72–77). Codex v2 defaults to full-history fork with `fork_turns=none\|all` (handler:221–233); [instruction snapshots/descendant policy](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/thread_manager.rs#L2010) appear at 2010–2047,2448–2459. `agent/history.rs:43–128,284–298` has path-dependent filters. These are not a universal instruction inheritance guarantee. |
| Writer/identity/lifetime/authority/error boundary | Assignment text, admitted governing instructions, personal memory and parent transcript carry different trust/context. Context visibility is not file authority. |
| Actual difference/disposition | **Bounded improvement/open contract C16.** A child may miss a governing constraint omitted from its assignment, but cloning all memory/profile/history is not automatically correct. |
| Conditions/impact/confidence/next step | High confidence in loader difference; required subset/precedence unresolved. Define the minimum admitted governing subset, preserving isolated transcript and downward artifact grants. No default full-memory clone or persistent-session feature is recommended. |

## G4 — Carry skill admission into foreground construction and updates

| Field | Comparison |
| --- | --- |
| Required outcome | Disabled or read-denied custom skills must not regain prompt admission through a role-specific loader. Tool read policy cannot undo a prompt read that already occurred. |
| Nanobot implementation/configuration | [Main ContextBuilder](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/context.py#L142) passes disabled names, context roots and resource policy (142–147). [Foreground _skills_section](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/delegation.py#L270) passes workspace/extra roots only (270–284), then injects always bodies. [SkillsLoader defaults](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/skills.py#L60) use empty disabled names/no policy (60–74); missing policy permits reads (166–169), direct reads occur at 148–164. gateway_runtime.py:391–394 updates main and legacy loaders, not foreground construction. |
| Reference mechanism | Codex roles are [disable-only](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/agent/role.rs#L91) (91–126), with disabled-plugin inheritance independent of fresh history (`control/spawn.rs:676–688,715–750`). DSH child-agent.ts:187–218,249–279 restores child composition/policy; equivalent DSH skill-loader behavior was **not audited**. |
| Writer/identity/lifetime/authority/error boundary | Custom skill metadata/body admission is a prompt read before tool enforcement. Managed denied skills, including resolving symlinks, can enter role context, especially `always` bodies; a disabled custom skill can reappear. Bundled-skill exemptions are a distinct existing policy. |
| Actual difference/disposition | **Strong conditional current skill-admission candidate C14.** High source confidence under custom skills/managed policy. No actual disclosure was observed. It does not depend on future persistent task sessions. |
| Conditions/impact/confidence/next step | Propagate admitted disabled state and shared read policy, including live-update behavior, into the foreground loader; preserve bundled exemption and non-elevatable roles. One S01/K01/T01 owner must reconcile loader construction and update paths before a narrow repair. |

## G5 — Preserve parent cancellation and shared lifecycle

| Field | Comparison |
| --- | --- |
| Required outcome | Awaited roles must remain within outer-turn cancellation/source-write lifetime, and shared MCP resources must close only after owned work drains. |
| Nanobot implementation/configuration | [Parent stop](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/loop.py#L1460) cancels/joins and shields legacy exec cleanup (1460–1504); runner.py:631–638 marks child writer failed and rethrows cancellation. Nested terminal metadata uses `_role_result` (delegation.py:599–624). Shutdown drains (loop.py:3010–3091) before foreground MCP close (delegation.py:831–835). |
| Reference mechanism | DSH manager.ts:1155–1188 serializes child close/delivery, drains at 233–237 and settles residents (964–1105); final flush (1133–1140) is best-effort. Codex [provisional rollback](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/agent/control/spawn_guard.rs#L64) (64–114) and runtime tree shutdown (runtime.rs:279–305) fence admission. DSH interrupt acknowledgement is not quiescence; Codex close/unload/runtime identity differ. |
| Writer/identity/lifetime/authority/error boundary | Parent task, child source writer, terminal metadata, child runtime and shared MCP connection have different closing obligations. |
| Actual difference/disposition | **Preserve.** No current nanobot cancellation defect is established in this slice. Persistent children would add stop/recovery obligations, not automatically simplify them. |
| Conditions/impact/confidence/next step | High source confidence in named paths; runtime cancellation behavior unverified. Preserve these obligations during skill/scope corrections and any D3 future design. |

## G6 — Preserve runtime capture and distinguish capacity domains

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Delegated work uses a coherent admitted model runtime and bounded role work. What provider/workload-wide capacity policy is needed remains pending. |
| Nanobot implementation/configuration | [Role limits](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/delegation.py#L142) default to six role calls/two worker calls/two parallel explorers (142–155); turn budgets are 227–255 and loop.py:3527–3535. Immutable request-runtime baseline/overrides are captured at delegation.py:498–508. |
| Reference mechanism | DSH [ActivationPool.reserve](https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/subagent/subagent/src/activation.ts#L23) (23–34) limits the live tree, with defaults eight children/depth one (index.ts:195–197); model/policy capture precedes awaits (manager.ts:302–414,child-agent.ts:68–118). Codex [execution limiter](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/agent/control/execution.rs#L82) applies only to V2SubAgent (82–87); residency/idle unload are separate (residency.rs:194–208,323–336). |
| Writer/identity/lifetime/authority/error boundary | A nanobot turn budget is not a provider-wide gate; Codex's subagent execution limiter is not a global serving gate. Residency, active execution, request rate and cost are different resource domains. |
| Actual difference/disposition | **Preserve; resource discussion D7.** No measured capacity assertion or automatic reference-budget adoption. |
| Conditions/impact/confidence/next step | High confidence in configured/captured boundaries; deployment workload/provider capacity unknown. Keep runtime capture; justify new budgets from an accepted product/workload need. |

## G7 — Preserve the intentional legacy admission fence

| Field | Comparison |
| --- | --- |
| Required outcome/open question | Retained construction/public exports do not authorize reopening retired background admission. Retirement/reactivation requires a compatibility inventory and accepted behavior. |
| Nanobot implementation/configuration | [Public lazy export](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/__init__.py#L33) (33,57) and loop.py:705–715 retain adapter lifecycle/config. Legacy `_build_tools` requests `scope='subagent'` (subagent.py:207–227); [ToolLoader](https://github.com/JadenChun/nanobot-custom/blob/27adced0aacd55d61791bb88a7240d7ad7bd5b75/nanobot/agent/tools/loader.py#L203) rejects noncore scope (203–219). Test source `tests/agent/test_subagent_lifecycle.py:1–4,38–40,80–88` describes the dormant adapter, intentionally absent background scope and expected real ValueError; it was read only and corroborates the primary code contract. |
| Reference mechanism | Codex active [legacy compatibility](https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/agent/control/legacy.rs#L1) (1–127; spawn.rs:915–988) differs from nanobot's fence. DSH resident/external manager scope differs too. Presence in a reference is not a requirement to reactivate this adapter. |
| Writer/identity/lifetime/authority/error boundary | SDK/public consumers, persisted records, construction/config and tool admission are separate compatibility evidence. External consumers were not observed. |
| Actual difference/disposition | **Intentional fence, not an accidental bug.** Preserve it pending D4; do not “fix” a retired admission path by making scope registration succeed. |
| Conditions/impact/confidence/next step | High confidence in the fence; external compatibility obligations unknown. Trace consumers before lazy construction, disclosure, retirement or reactivation. Keep that work separate from current foreground skill/scope corrections. |

The next [source-only contracts](audit-backlog.md) separate skill construction/live updates,
project-root/focus agreement, governing-instruction precedence and compatibility inventory.
Source comparisons do not attest external Codex app-server human authorization; that path
was search-only. They do not select a new task/session schema or framework.
