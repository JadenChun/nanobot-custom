# Reference answers for the architecture plan

Reviewed 2026-11-11. This document answers planning questions using pinned DSH and
public Codex source plus official documentation for Codex desktop and client/server behavior.
It does not inspect or qualify the nanobot refactor implementation. D1 is selected
by the user. On 2026-11-11 the user accepted the D2–D7 architectural direction below,
including retaining the existing Full-consent restart behavior. The
[acceptance record](plan-review.md#user-acceptance) distinguishes that decision from
the later explicit four-default selection from implementation acceptance. The user chose
step 1 on 2026-11-11, selecting the stated lifetime/budget, device-proof and retention defaults.

## Reference basis

| Source | Identity and scope |
| --- | --- |
| DSH | Public `deepseek-ai/deepseek-harness` at `fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159`. Bounded source reads, building on the frozen audit comparison. |
| Codex | Public `openai/codex` at `6ff1e1ca3b30fc8b914720681fcf5fe62e95f856`, the existing `0.166.0` / `rust-v0.166.0` comparison pin. Bounded source reads; not private desktop source. |
| OpenCode, subagent follow-up only | Public `anomalyco/opencode` at `e56eff2bc01649b3aa21224876b8874e77dea4b0`, resolved from `dev` on 2026-11-11. Bounded subagent-tool, parent-completion and app-composer reads; not a whole-harness audit. |
| Desktop behavior | Official [permissions][permissions], [sandbox][sandbox], [subagents][subagents] and [app-server][app-server] documentation retrieved on 2026-11-11. These are current documentation snapshots, separate from the older source pin. |

The initial study retrieved fourteen reference source files and four official Markdown
pages. The later D3 follow-up adds separately recorded Codex/OpenCode source reads.
SHA-256 identities, retrieval-only qualifications and read bounds are retained in
[the planning evidence](evidence.json).
The source observations below establish mechanisms, not universal reliability or
equivalence between DSH, Codex and nanobot. Private Codex/ChatGPT desktop internals are
not inferred from public source or client documentation.

## D1: selected platform scope

The user selected Linux and macOS execution, with Android as a remote client. Windows
is outside this review's required qualification scope; this does not authorize deleting
existing compatibility. The user confirms Linux and Android have already been verified.
macOS is the remaining verification target. Those are user-confirmed baseline states,
not new matching-commit test receipts produced by this planning review.

DSH and Codex demonstrate native macOS confinement through Seatbelt, and official
[sandbox documentation][sandbox] describes the same platform distinction. That supplies
a design reference for a macOS adapter. It does not establish that nanobot already has
that adapter or that existing macOS runtime/owner contracts pass. Later verification must
cover the selected workspace/Full modes, process ownership, owner authentication and
session behavior separately. Android remains a client of the admitted executor.

## D2: separate permissions, consent and enforcement

**Reference facts.** Official [permissions][permissions] distinguishes the technical
sandbox from the approval policy. Changing who reviews an action does not expand the
sandbox. Full access is described as unrestricted file/network access within the
effective environment. DSH [Full execution][dsh-full] bypasses its confinement wrapper.
DSH also [persists per-session mode policy][dsh-mode]; its persistence is a policy
decision, not a serialized native launch capability. The pinned Codex comparison
retains some denied-read restrictions under escalation, so the two implementations
must not be described as identical.

**Accepted direction for nanobot.** Keep a permission profile, approval scope and concrete
backend enforcement as distinct facts. Resolve an effective target before approval and
recheck current authority at dispatch. Once consent covers one selected occurrence;
standing session policy has its own explicit scope and revocation. Delegation cannot
turn a role label into broader authority. Full must describe its actual OS-account
access and any separately enforced exclusions truthfully.

Persisted mode/policy settings may be restored as policy, but old runtime capabilities,
Once allowances and completed effects are not revived by loading them. Reconstruct
admission against the current owner, instance and policy after restart. DSH demonstrates
persisted session policy, but the references leave consent persistence as a product
choice. The accepted nanobot direction keeps the existing volatile Full-consent contract:
restart does not restore prior consent.

## D3: canonical child sessions with separate execution lifetime

**Reference facts.** DSH [child creation][dsh-child] records child identity, parent
relationship and accepted input, separately from an activation/run identity. Its
caller-delivery path can use optional persistence; not every DSH child is universally
durable. Codex [spawn admission][codex-spawn] retains provisional ownership through
child/relationship creation and input admission before committing capacity. Official
[subagent documentation][subagents] describes asking the main agent to steer/stop children,
with optional child-thread inspection. The [app-server contract][app-server] separates thread
start/resume/fork from starting, steering and interrupting a turn.

**Accepted direction for nanobot.** Use the existing canonical session/history machinery
for addressable children. Separate child identity/history from its current run, parent
relationship, policy and resource reservation. The parent orchestrates and collects
results; child activity has its own authoritative tool intents, results and terminal or
uncertain outcomes. Do not add a second task-history authority.

Child authority is bounded by live parent authority, role restrictions and assignment
scope. Required governing instructions and disabled capabilities remain effective even
with a fresh history. [Child policy construction][codex-child-policy] and the official
[subagent permission contract][subagents] support keeping policy independent of prompt
role text. Existing awaited roles can use the same execution path without requiring
every helper to become a separately exposed persistent task.

Keep user interaction in the main chat, with internal start, collect/wait, follow-up and
cancellation responsibilities. Optional inspection does not require a separate child
composer or management interface. A normal parent reply differs from explicit parent
Stop and runtime shutdown. Bound admission before launch. Preserve recoverable history
after restart; durable identity does not prove that an OS process survives or authorize
automatic replay. External child backends are outside the first-release scope.

This supports including canonical child-session architecture in the plan. The earlier
blanket recommendation to defer it is superseded by this reference-based recommendation;
the user has accepted that direction and clarified a preference for main-chat control.
The user subsequently selected the runtime lifetime and finite numerical defaults in
the [product contract](v1-product-contract.md#final-selections-for-plan-freeze), without
requiring the earlier broad public child-control proposal.

## D3 follow-up: main-chat control in Codex and OpenCode

On 2026-11-11 the user asked whether direct child control was overengineering and named
Codex/OpenCode as implementation references. Here "child" means a subagent, not a new
independent user conversation. Bounded source reads support the following distinction:

| Reference observation | Evidence and limit |
| --- | --- |
| Codex gives the model collaboration operations. | Its [v2 handlers][codex-collaboration-handlers] implement follow-up, messaging, interrupt and wait. The [follow-up tool specification][codex-followup-spec] starts an idle child turn or delivers at supported boundaries. These are public-source tool mechanisms, not private desktop source. |
| Codex directs user requests through the main agent. | Official [subagent documentation][subagents] says orchestration includes spawning, follow-ups, waiting and closing, and tells users to ask Codex to steer/stop children. It also offers optional child inspection. This does not imply every client lacks direct controls. |
| OpenCode uses one parent-facing subagent tool for new and continued work. | [Tool input/description][opencode-subagent-tool] accepts an existing session ID for follow-up, defaults to foreground and offers explicit background mode. [Parent validation][opencode-child-parent] restricts reuse to the current parent's child. |
| OpenCode returns completion to the parent. | [Execution][opencode-subagent-execution] runs or joins child work, while [completion delivery][opencode-child-completion] targets `parentSessionID`. Foreground cancellation and background notification are distinct paths. This is not universal proof of crash-safe external-effect settlement. |
| OpenCode's app provides child status and a path back to the parent. | The [composer gate][opencode-child-composer] excludes child sessions from the ordinary composer, and [the child bar][opencode-child-status] shows a back-to-parent control. This observation is about that app component, not all OpenCode APIs/clients. |

**Revised nanobot proposal.** Keep the original chat as the control surface. The main
agent owns delegation, follow-ups, result integration and targeted cancellation; the
existing parent Stop also cancels owned work without requiring another model response.
Show activity/results and optionally authorized read-only child detail. Use the existing
canonical session/runner/approval owners internally. Require neither a separate child
composer/management panel nor a universal backend API with public steer/close/resume.

Internal lifecycle operations still matter. For example, Codex's [v1 close operation][codex-close-spec]
shuts down an agent/descendants and counts open agents toward capacity, while its
[v2 interrupt][codex-interrupt-spec] leaves the agent available for further tasks. Neither
definition establishes the earlier proposed nanobot user-facing archive operation.
Choose the minimal native lifecycle needed by nanobot; preserve canonical history when
settled resources are released.

The pinned references do not require identical permission models, budgets or restart
promises. Nanobot retains its accepted live-authority intersection, exact effect consent
and uncertain-effect rules. The [first-release contract](v1-product-contract.md) records
the smaller product scope. Initial finite limits and background lifetime were selected
through the later four-default decision; no implementation conformance is claimed here.

## D3 client UI behavior

The user subsequently approved main-chat control and requested that the client UI align
with the reference harness behavior. The inspected evidence supports a small interaction
surface:

| Observed behavior | Reference and scope |
| --- | --- |
| Activity in the original chat opens optional child inspection; users ask the main agent to steer or stop work. | Official [Codex subagent documentation][subagents], desktop section. This describes documented behavior, not inspected private desktop source, and does not rule out other client controls. |
| A child view replaces the ordinary composer with status and a return path. | OpenCode's [composer gate][opencode-child-composer] and [child bar][opencode-child-status]. The [state rendering][opencode-child-state] distinguishes running, succeeded, failed and interrupted, with elapsed time. This is a bounded app-component observation. |
| Child results return to their parent. | OpenCode's [parent completion delivery][opencode-child-completion] and Codex's documented consolidated response. This supports parent result integration without mirroring every child message into the original transcript. |

**Selected nanobot interaction.** Keep the original chat composer and compact delegated
activity/results. Optional authorized read-only detail has a clear way back; it does not
become a separately managed chat. Apply the same meaning to desktop/WebUI, TUI and Android
remote using their existing layouts, with keyboard/touch-accessible status and navigation.
No separate child composer, task sidebar product or manual lifecycle UI is required.

The [client UI contract](v1-product-contract.md#client-ui-behavior-for-subagents) also
requires exact child-effect approval through the original chat, immediate runtime Stop,
truthful stopping/uncertainty and canonical reconnect/readback. These are nanobot
adaptations of its accepted authority/recovery contracts, not crash-safety claims proved
by the reference UI components. The same document lists later observable acceptance
scenarios. No nanobot UI source or running client was inspected for this planning update.

## D4: versioned compatibility and explicit migration

**Reference facts.** The official [app-server contract][app-server] distinguishes stored
history reads from live resume and documents version/feature gates and deprecated
history responses. The original [compatibility comparison](../../audits/architecture-review-2026-11-06/decisions.md#d4-compatibility-and-retirement)
preserves external callers and stored-state obligations. Neither reference supplies
nanobot's Android pairing/cutover protocol or an installed historical-reader receipt.

**Accepted direction and subsequent scope selection.** Keep one current live owner and
explicit versions for wire/state formats. The user selected a first-release baseline
without legacy migration: no v1-to-v2 cutover, forced re-pairing, historical-release
reader or two-version upgrade rehearsal is required. This is a nanobot product decision,
not a fact supplied by DSH/Codex. See [the first-release contract](v1-product-contract.md).

Specify the initial device-proof contract independently of legacy migration. Ordinary
restart/reconnect, cancellation, revocation and uncertain-effect recovery remain required.
An older development build must retain those facts or refuse an unsupported path.
Future changes to a released format can select narrow readers/adapters and explicit
migration/rollback when those obligations actually exist.

## D5: request-owned bindings with live revocation

**Reference facts.** DSH [resolved tool sets][dsh-tools] retain admitted definitions and
check current registration identity before presenting a callable view. Codex
[`StepContext`][codex-step] captures immutable request settings, exact MCP bindings and
the tool router, while grant access remains separately evaluated.

**Accepted direction for nanobot.** A model sampling request owns its tool definitions,
execution bindings and connection lifetime through settlement. A ready replacement is
published for new requests. Existing requests use their original still-authorized
binding or fail explicitly when it is revoked/unavailable; they never execute a new
registration merely because its name matches. Keep old resources until their admitted
users settle, then close them once. Failed replacement preserves an authorized usable
old binding. Revocation remains live.

This is the recommended design over a global pause of all work during reload. Safe
boundary deferral remains a smaller implementation strategy only if it satisfies the
same request-identity and revocation contract. No global generation framework is needed.

## D6: owner-local recovery with explicit uncertainty

**Reference facts.** DSH [tool recovery][dsh-recovery] distinguishes a call that never
started from one whose outcome is unknown after interruption. Codex
[tool execution][codex-errors] separates fatal failures from ordinary tool feedback.
Official [turn lifecycle documentation][app-server] distinguishes request acceptance,
streamed items and the final turn status.

**Accepted direction for nanobot.** Preserve raw compatible results and add the smallest
typed internal disposition required by the owner. Record whether an effect did not
start, applied, partly applied or remains uncertain; delivery acknowledgement is a
separate fact. Cancellation, authority denial and mandatory source-write failure stop
the affected path. Ordinary tool feedback can permit bounded reasoning/recovery after
required recording. Uncertain effects need readback/reconciliation rather than blind
retry; retry never implicitly widens authority.

Neither a repaired transcript nor a provider's success response proves an external
effect was delivered. Keep nanobot's stronger existing source/receipt obligations even
where a reference has a weaker flush or process-local notice contract.

## D7: bounded output, explicit completion and separate retention

**Reference facts.** DSH's [output ring][dsh-output] uses absolute offsets and reports
loss after bounded eviction. Codex [output buffers][codex-output] separate pending
poll output from a retained completion transcript, and its [completion watcher][codex-completion]
uses that transcript. These process buffers are not a durable session retention policy.
DSH's [storage boundary][dsh-storage] separately exposes durable append/flush semantics.

**Accepted direction for nanobot.** Give live output a bounded view with explicit cursor,
truncation/loss and provenance. Completion inspection reads a retained bounded transcript
or explicit summary, independently of what polling already consumed. Keep model-turn
completion, process-tree settlement and external delivery status distinct. A Stop request
is acknowledged before it necessarily settles all owned effects.

Separate live process capacity, child execution capacity, model request budgets and
durable evidence retention. Refuse admission before effects when capacity is unavailable.
Reclaim settled output under a named policy; preserve unresolved effect/source/receipt
evidence. A memory-buffer eviction cannot justify deleting durable history or unknown
effects. Preserve existing finite limits where applicable; choose new child limits and
retention durations for nanobot's workload instead of copying reference numbers.

## Nanobot-specific product choices

The [future extension record](future-harness-extension.md) connects A3 to the existing
coding-harness and SuperAgent planning documents. It preserves session-based coordination
and ordinary subagent behavior as separate layers. That comparison uses dated local
planning records; it does not reverify their implementation claims or add a fresh audit
of the SuperAgent prototype to the pinned source studies above.

The references provide strong answers for ownership, permission separation, child
identity, request binding and recovery/output boundaries. They do not select every
product parameter. The user selected the four concrete defaults in
[the first-release contract](v1-product-contract.md#final-selections-for-plan-freeze): internal subagent lifetime,
finite child budgets, initial device proof and output/retention policy. Legacy migration is
outside the selected first-release scope. Existing volatile Full-consent restart behavior
is retained by the user's accepted direction. Other choices can retain an explicit
existing behavior or be specified separately; none should be inferred from a code
candidate. These selections are recorded in the
[frozen plan](plan-review.md#frozen-baseline-and-subsequent-work).

[permissions]: https://learn.chatgpt.com/docs/permission-modes
[sandbox]: https://learn.chatgpt.com/docs/sandboxing
[subagents]: https://learn.chatgpt.com/docs/agent-configuration/subagents
[app-server]: https://learn.chatgpt.com/docs/app-server
[opencode-child-state]: https://github.com/anomalyco/opencode/blob/e56eff2bc01649b3aa21224876b8874e77dea4b0/packages/app/src/session/composer/subagent-bar.tsx#L46-L77
[dsh-full]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/shell/bash-sandbox/src/index.ts#L89-L96
[dsh-mode]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/sandbox/sandbox-policy/src/session-mode.ts#L1-L18
[dsh-child]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/subagent/subagent/src/manager.ts#L302-L414
[codex-spawn]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/agent/control/spawn.rs#L799-L868
[codex-child-policy]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/agent/child_config.rs#L65-L85
[dsh-tools]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/core/tools/src/index.ts#L264-L296
[codex-step]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/session/step_context.rs#L22-L76
[dsh-recovery]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/core/session/src/repair.ts#L157-L196
[codex-errors]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/tools/parallel.rs#L129-L141
[dsh-output]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/jobs/jobs-local/src/ring.ts#L75-L111
[codex-output]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/unified_exec/process.rs#L43-L55
[codex-completion]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/unified_exec/async_watcher.rs#L327-L350
[dsh-storage]: https://github.com/deepseek-ai/deepseek-harness/blob/fcc0ad462172522aaf5c93a7b2e01f6b9f0b0159/packages/session/session-persistence-jsonl/src/storage.ts#L187-L211
[codex-collaboration-handlers]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/tools/handlers/multi_agents_v2.rs#L28-L42
[codex-followup-spec]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/tools/handlers/multi_agents_spec.rs#L236-L262
[codex-close-spec]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/tools/handlers/multi_agents_spec.rs#L336-L357
[codex-interrupt-spec]: https://github.com/openai/codex/blob/6ff1e1ca3b30fc8b914720681fcf5fe62e95f856/codex-rs/core/src/tools/handlers/multi_agents_spec.rs#L361-L382
[opencode-subagent-tool]: https://github.com/anomalyco/opencode/blob/e56eff2bc01649b3aa21224876b8874e77dea4b0/packages/core/src/tool/plugin/subagent.ts#L30-L62
[opencode-child-parent]: https://github.com/anomalyco/opencode/blob/e56eff2bc01649b3aa21224876b8874e77dea4b0/packages/core/src/tool/plugin/subagent.ts#L153-L168
[opencode-subagent-execution]: https://github.com/anomalyco/opencode/blob/e56eff2bc01649b3aa21224876b8874e77dea4b0/packages/core/src/tool/plugin/subagent.ts#L184-L264
[opencode-child-completion]: https://github.com/anomalyco/opencode/blob/e56eff2bc01649b3aa21224876b8874e77dea4b0/packages/core/src/session/subagent-completion.ts#L21-L45
[opencode-child-composer]: https://github.com/anomalyco/opencode/blob/e56eff2bc01649b3aa21224876b8874e77dea4b0/packages/app/src/session/composer/session-composer-region.tsx#L36-L44
[opencode-child-status]: https://github.com/anomalyco/opencode/blob/e56eff2bc01649b3aa21224876b8874e77dea4b0/packages/app/src/session/composer/subagent-bar.tsx#L97-L109
