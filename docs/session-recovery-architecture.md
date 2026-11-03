# Session context lifecycle: preventive budgeting and authorized media inputs

Diagnosis/planning baseline: 2026-11-03, source `a90191a41b025884e76ce029b98a5d5349171f2e`.
Direction updated on 2026-11-03 after the DSH/OpenCode comparison: make ordinary long sessions
work through consistent route accounting, reserved compaction capacity and model-derived summary
budgets. Bounded retries are secondary recovery. This is an implementation plan; source fixes,
provider calls, live-session edits, installation and rollout have not occurred.
It supplements [mobile-remote-request-plan.md](mobile-remote-request-plan.md). Repository
instructions, source/checkpoint authority and permission ownership remain controlling.

## Findings and confidence

The inspected turn **failed**; it is not an active execution deadlock. Sanitized runtime evidence
at approximately 15:30 local shows 271 canonical events, no checkpoint or execution lease,
no goal-state metadata and no unmatched tool-call intents. History/tool closure survived.

| Finding | Evidence and interpretation |
| --- | --- |
| Main request exceeded its finalized input budget | Q6 route: context 165,888, completion reserve 8,192, safety buffer 1,024; input limit 156,672. Finalized input 156,778 exceeded it by 106. |
| Recovery failed before the original rejection was logged | One auxiliary call reported 123,445 input / 2,048 output, 218,307 ms, `summary_truncated`, boundary 0. This is not evidence of a second main rejection. No checkpoint advanced. |
| Pressure measurement differed from final measurement | Before: 133,255 using `tiktoken_compatibility`, below the roughly 141,005 proactive trigger. Final: `estimated_compat` plus inline-image allowances. The 23,523 difference cannot all be attributed to images. |
| Wrapper capabilities explain the canonical fallback | Factory deduplicates the session's Q6 override from global fallbacks, leaving DeepSeek inside `FallbackProvider`. That wrapper declines preview with multiple routes and exposes no source-summary capability; the log records `summary_provider_missing_capability`. |
| Auxiliary routing is distinct | `FallbackProvider.chat_stream_once` pins its captured primary. Main failover/circuit-open selection may choose another leaf. Leaf capability alone does not prove wrapper capability. |
| Summary selection is input-greedy | Governor caps output at 2,048, selects the largest input-fitting closed chunk, treats truncation as terminal, and forced recovery continues toward the entire eligible prefix. |
| Files exist but shell visibility differs | All eight advertised generated images were verified on the host. MCP writes bytes/JSON through the current media root; managed bubblewrap masks the entire instance and restores workspace/grants, without media. Empty raw `exec.sandbox` does not disable this wrapper. |
| Fresh image results differ from saved replay | Eight fresh `read_file` results, and seven later shared reads, contained native image blocks without errors. Runner appends live results; the source writer deliberately replaces inline images with a placeholder for later turns. |

Relevant code: `providers/{factory,fallback_provider,openai_compat_provider,request_budget}.py`,
`agent/{context_governance,memory,runner,loop}.py`, `agent/tools/{mcp,shell,sandbox,filesystem,path_utils}.py`,
`utils/artifacts.py`, `session/{turn_writer,source_projection}.py` and `providers/image_history.py`.
This is a confirmed budget/recovery contract mismatch and a confirmed media-access mismatch,
not missing host files, erased history or proof that fresh vision is broken.

The aggregate truncation result does not distinguish provider `finish_reason=length` from a
locally overlong summary with `stop`. Canonical summaries request reasoning effort `none`,
but the compatibility adapter omits that wire field unless supported native thinking controls
apply; the Q6 alias has no explicit style. Hidden reasoning exhaustion is **unproven**.
218 seconds is below the observed 900-second operation and 300-second idle deadlines.
Actual model-side image comprehension is unproven without wire/response evidence. A prior
60-second filesystem search timeout is a tool-choice issue, not a reason to increase timeouts.

## Reference architecture and accepted direction

The comparison concerns complete lifecycles. A shared last-resort failure guard is not evidence
that another harness encounters Nanobot's practical failure. No matched workload replay has yet
attributed how much each difference contributes to latency or reliability.

- DSH inspected baseline: `903a6dd97cb9fd5eff68e1d1a4a3799661ab1cf1`. Its
  [prepared call](https://github.com/deepseek-ai/deepseek-harness/blob/903a6dd97cb9fd5eff68e1d1a4a3799661ab1cf1/packages/llm/llm/src/index.ts) captures model metadata,
  controls and adapter together; its [compaction policy](https://github.com/deepseek-ai/deepseek-harness/blob/903a6dd97cb9fd5eff68e1d1a4a3799661ab1cf1/packages/compaction/compaction-basic/src/config.ts)
  reserves output plus compaction headroom before pressure qualifies. Its default summary/headroom
  allowance is 65,536, configurable per target; that number is evidence, not a Nanobot default.
  Local uncommitted pi-ai changes are not part of the committed comparative baseline.
- OpenCode V2 inspected baseline: `3ba597f7769e19cc3b5f67e3d6667e7b6318f6a3`. Its
  [request preparation](https://github.com/anomalyco/opencode/blob/3ba597f7769e19cc3b5f67e3d6667e7b6318f6a3/packages/core/src/session/model-request.ts)
  fits output to model capacity; its [compaction lifecycle](https://github.com/anomalyco/opencode/blob/3ba597f7769e19cc3b5f67e3d6667e7b6318f6a3/packages/core/src/session/compaction.ts)
  reserves space and preserves a cache-compatible prefix when possible. This is V2 source evidence,
  not qualification of an installed OpenCode build. Do not copy emergency omission of older
  exchanges into Nanobot: retained source and eligible-prefix coverage remain controlling.

The transferable boundary is a provider-neutral context governor fed by the actual route's
prepared request and declared capabilities. Reuse Nanobot's existing policy, capability,
generation, preview and checkpoint owners; do not create another provider registry, routing
service, token meter or persistence framework. No model-name branches belong in the governor.

## Contract 1: resolve, prepare and measure the actual route

Keep existing finalized-request accounting and its conservative margins. The invariant remains:
`finalized input + completion reserve + safety buffer <= effective route context window`.
Finalization includes converted messages, tools, schemas, media and normalization; every
concrete main, fallback, transformed retry and auxiliary leaf retains its final guard.
Do not disable fallback, remove the safety buffer or certify all routes with primary's preview.

Extend the existing preview/call-context seam with a small typed captured **route selection**:
purpose, opaque route identity, effective model/generation/window, request fingerprint and
measurement/provenance. Selection uses the same next-attempt decision as dispatch, including
the circuit-open primary case. Preview is non-networking and certifies only that selection.
Dispatch consumes that captured selection or reports a typed route change for re-preparation;
it cannot silently use its quote for a different leaf or changed request. This is an in-process
provider-edge contract, not a new routing service or durable provider handle.

Resolve model context/input/output limits, request generation defaults, reasoning-budget semantics,
modalities and optional counting/native-compaction support through existing provider capabilities
and explicit configuration. Distinguish model limits from a user's per-request output default.
Custom aliases need no hardcoded catalog entry: use declared route metadata/config overrides.
Unknown facts stay unknown; missing exact counting does not reject an otherwise supported model.
An unknown window cannot support a proactive capacity guarantee: report that limitation and the
existing configuration path rather than inventing a capacity or disabling all ordinary chat.

Main failover selects and rebuilds/recounts the next leaf's portable projection, clears
incompatible continuation state and re-enters governance if reduction is required. No automatic
failover after any delivered delta; context-length remains non-fallbackable. Unsupported preview
is explicit: use that leaf's existing conservative accounting/final guard, never a primary quote.
Prepared estimates remain estimates; do not claim tokenizer accuracy for an arbitrary local alias.
Use the adapter's supported counter/tokenizer when available, otherwise its existing conservative
estimate with explicit provenance. Anchor to reported usage only while route, model, generation
semantics and prepared envelope remain compatible; invalidate/reprice after a switch. Wrappers
must preserve the selected route's supported accounting, not globally disable it because fallbacks
exist. Preserve current strict-versus-compatible policy instead of silently weakening strict routes.

Auxiliary preparation has a separate **captured-primary-only** capability/preview contract,
matching existing once-only dispatch even when normal requests have fallbacks. Delegate through
the wrapper only with that route/purpose binding. Historical-prefix mode still requires exact
converted prefix/tool/envelope agreement; otherwise use canonical summary mode with a reason.
Adding `supports_source_summary=True` alone is insufficient. No auxiliary fallback or retries
inside the helper, and no foreign provider receipts/handles propagated through the wrapper.
Same-model summaries remain the default. A separately configured summary model must resolve its
own capabilities/budgets; never silently move historical content to a different provider.
Native compaction is an optional adapter capability; portable source remains available when a
model/provider changes and its opaque state becomes incompatible. Adding native compaction to
another adapter is not a prerequisite for these corrections.

Use route-finalized units for pressure, chunk feasibility, rebuilt requests and final admission.
Record content-free route/purpose/fingerprint, measurement source, input/limit/reserve/media
allowance and why preview declined. Distinguish requested reasoning setting, applied supported
wire controls and provider-reported usage/finish reason. Do not guess an alias's thinking API
or change live configuration as a diagnostic. Never log prompt/image bytes, secrets or endpoints.

## Contract 2: prevent pressure failures, then recover bounded exceptions

This direction supersedes the fixed 2,048-token summary assumption and retry-first approach in
the earlier draft, and intentionally revises the conflicting guidance in
[context-management-migration-design.md](context-management-migration-design.md). Do not replace
2,048 with another universal constant or make an 8:1 compression ratio a correctness rule.

### Ordinary long-session lifecycle

1. Before normal model dispatch, evaluate pressure on the prepared working context, including
   actual tools, schemas, media and fixed instructions. Confirm existing bounded tool/media
   projections run before measurement; preserve full source independently. Correct demonstrated
   wiring gaps only, without introducing a new pruner, storage engine or legacy reducer.
2. Resolve a separate summary-generation budget from the selected summary route and purpose.
   Default its requested output allowance to that runtime's resolved generation setting, with
   an explicit compaction override when configured; clamp to declared model output/context
   capacity and operation limits. A normal output default is not a model's maximum capability.
   Separately bound retained summary text by the rebuilt main request's available space.
   Account for reasoning according to adapter semantics, without double counting it. Apply
   disabled-reasoning controls only when supported; otherwise reserve generation room for it.
3. Reserve compaction capacity before the main context fills: summary instructions/envelope,
   useful summary generation, estimate uncertainty and anticipated next-step growth. Evaluate
   both main and summary envelopes under their own route limits. The current pressure ratio
   is an upper trigger bound, not permission to consume required summary room. Use existing
   output/media bounds to size growth conservatively; re-evaluate after a large tool result.
   An unbounded or irreducible new input remains an explicit exceptional case, not a guarantee
   that a headroom percentage can absorb every possible attachment or tool response.
4. Select an eligible closed historical prefix while preserving the previous checkpoint,
   newest user input, recent useful exchanges and current runtime instructions. Prefer the
   verified historical-prefix path when legal, retaining cache compatibility where supported.
   Otherwise prepare canonical evidence explicitly; measure the envelope actually sent.
   Fit summary input and generation together before dispatch, not just the largest input that
   can fit. The once-only summary helper transports; the governor plans and validates.
5. Require complete useful summary text within its retained-text budget. Fully remeasure the
   rebuilt main request, including its retained tail and fixed costs. Preserve artifact dependency
   proofs, contiguous coverage, request-ledger eligibility and execution ownership. Publish by
   captured source/checkpoint CAS before changing the live projection. A conflict discards the
   candidate; compaction never replays tools, removes source events or adopts stale authority.

Any new deployment-tunable budget fields belong in typed `SourceCompactionConfig` with validation
and existing aliases. Keep global policy defaults distinct from route/model facts; no hidden
environment switches or per-alias patches in the governor. Keep the existing deadlines and
four-physical-call ceiling for this correction, counting all failed auxiliary attempts and usage.

### Secondary emergency recovery

Normalize input overflow, summary output truncation, transport failure, cancellation and invalid
envelopes separately at the provider edge. After an exceptional rejection, use the captured actual
route to replan: a valid larger generation allowance within policy or smaller closed summary
chunks may help. These are bounded fallbacks after preventive budgeting, not the normal strategy.

Discard truncated text, including a `stop` result exceeding the retained allowance; retry only
under the same operation deadline/call ceiling at the same evidence cursor. Never treat it as a
checkpoint or advance coverage. Only complete folds extend disjoint contiguous evidence proofs.
Do not omit unsummarized exchanges to force a fit. Source preservation does not promise lossless
semantic compression, but every covered exchange must be considered by a successful summary.

Rebuild/recount after useful reduction and stop once the main request fits with the planned
headroom where feasible; emergency recovery need not consume the entire eligible prefix. A fully
valid fitting candidate may publish at the work ceiling even if the headroom goal is unmet.
No progress, irreducible fixed/current content, cancellation, deadline or provider failure leaves
the prior checkpoint/source intact and reports an actionable reason. Preserve root rejection,
recovery outcome and usage; never turn a disconnect into automatic replay of uncertain work.

## Contract 3: generated-media access without exposing the instance

Choose an additive optional `ExecTool.input_files` list for this repair. Advertise the usable
contract in the existing tool/MCP-result guidance: an ordinary copy declares the generated source
file in `input_files`, uses its returned original path, and writes only to an authorized workspace.
Omitting the field preserves current shell confinement; supplying a path is not an access grant.

Resolve each input through the existing `ReadFile`/current media resolver and its capability-specific
read policy. Open only approved regular files; reject escaping, symlink and private-control paths.
Bind the invocation snapshot to those actual opened bytes and current permissions, not merely a
checked pathname that can change before use. Preserve any source/chat scope already enforced by
that policy; do not add a separate per-chat allowlist, provenance ledger or producer handoff.
Earlier-turn files use the same fresh read-policy check; unavailable/denied inputs fail explicitly.

Create bounded server-owned per-invocation read snapshots and mount only those files read-only
at their **original advertised paths** inside the managed sandbox. Revalidate before launch;
fail closed on source identity/symlink or permission changes. Bound count/bytes with existing
media/file limits and clean staging after the owned command tree ends. Keep the whole instance,
private source/artifact storage and credentials masked. Exact mounts expose no additional parent
directory, sibling JSON or other-chat payload; individually approved files retain current read-policy
semantics. Do not mount a whole media directory or grant arbitrary read-only host inputs. Existing write,
native process supervision, cancellation, Once/Full and request-context boundaries still apply.

For cross-turn vision, preserve live native image results and the newest-image-batch window.
Clarify the persisted-placeholder/model-facing availability wording: saved replay contains no
image bytes and cannot substantiate visual claims; a known file may be explicitly reread under
current policy when needed. Missing/deleted/denied files remain honestly unavailable. Do not
automatically inject all historical images, rewrite old source events or claim images were seen.

Optional follow-up: small durable authorized media-reference/provenance metadata and bounded
version-aware reread, if later use cases require stronger cross-turn discovery. Existing generated media plus
text-only private source artifacts are distinct stores; the latter does not already support binary.
A canonical binary-store/schema/retention migration is **not** needed for the confirmed shell
blocker. This plan adds no temporary-chat retention or resurrected media after disposal.
Structured process exit codes may improve copy-error clarity later; do not turn every nonzero
exit into a tool error or change `fail_on_tool_error` behavior. It is not a compaction fix.

## Bounded implementation packages

One writer owns each overlapping seam. P1 and P3 may proceed independently; P2 follows P1.
Within P2, complete preventive lifecycle wiring before adding secondary recovery. Keep each delta
reviewable and hand shared files over sequentially. Do not rewrite unrelated provider adapters.

| Package / owner boundary | Outcome and focused check locations |
| --- | --- |
| P1: provider capabilities/policy, prepared preview/dispatch and auxiliary contract; existing `providers/` owners, factory only if required | Carry actual route facts/accounting through wrappers; preserve custom aliases and optional counting. Reuse `tests/providers/test_fallback_provider.py` and `tests/providers/test_final_request_budget.py`. |
| P2: `agent/{context_governance,memory}.py`, necessary `config/schema.py` fields and owning compaction contract documentation; runner wiring only if required | First: shared pressure measurement, reserved summary capacity, model-derived generation/text budgets and verified prefix preparation. Then: bounded exceptional recovery, preserving publication invariants. Reuse affected cases in `tests/agent/test_finalized_budget_recovery.py`, `tests/agent/test_source_pressure_compaction.py`, `tests/agent/test_source_dependency_compaction.py`, `tests/agent/test_source_compaction_publication.py` and `tests/session/test_source_compaction.py`. |
| P3: `agent/tools/{shell,sandbox,path_utils,mcp}.py`, reuse of the existing file resolver and narrowly scoped availability wording | Exact inputs under current read policy; no directory exposure, new media authority or binary retention. `.venv/bin/python -m pytest tests/tools/test_sandbox.py tests/tools/test_filesystem_tools.py tests/tools/test_mcp_tool.py tests/utils/test_artifacts.py tests/agent/test_recent_image_governance.py -q`; real managed-sandbox copy/read smoke on a disposable workspace |
| P4, optional closest UI edge: existing compaction event/types and `ThreadShell.tsx` | Show failed/recovered state and actionable sanitized reason if current delivery hides it. Reuse `tests/agent/test_context_error_delivery.py` and affected existing WebUI thread checks; coordinate shared files with mobile package F. |

## Acceptance outcomes and evidence

Use a small set of public use-case families, not a test for each branch or a Cartesian matrix:

- **Ordinary long-session continuity:** representative mixed text/tool/image conversation
  continues through proactive compaction while retaining the current request and useful recent
  context. Include materially different model windows/generation semantics and a custom alias
  with declared metadata. Adding fallbacks must not disable preventive accounting; model/route
  changes reprice the context and discard incompatible native state. Exact counting is optional.
- **Summary capacity:** the selected model has room for a complete useful summary before main
  admission fails. Summary output and retained-text allowances follow resolved capabilities and
  policy, including supported reasoning semantics. A valid historical prefix is preserved where
  compatible. Success must not depend on a universal 2,048 limit or fixed compression ratio.
- **Exceptional interruption/recovery:** a genuine overflow or truncated summary can recover
  within bounded work when reducible. Otherwise irreducible input, cancellation, deadline or
  conflict ends truthfully with prior source/checkpoint, newest input, tool closure, dependencies
  and authority preserved. No failed summary publication, over-budget dispatch or duplicate effect.
- **Artifact use:** an approved generated-file copy succeeds at its advertised path with declared
  inputs; denied paths remain denied and exact mounts expose no extra siblings/parent contents.
  Fresh image results stay native; saved replay describes unavailable visual evidence accurately
  and permits authorized explicit reread. Preserve temporary disposal and current read scope.

Prioritize the ordinary lifecycle over deliberately forced recovery. Use an existing harness and
sanitized representative workload to compare baseline and candidate under matching provider/model,
limits and tool/media inputs. Record trigger measurement, reserved capacity, summary allowance,
main/auxiliary usage, latency, reported cache reuse when available and task-continuation evidence.
Do not infer reliability from terminal guards, invent performance results or require new benchmark
infrastructure. Reuse existing integrity cases; add only a missing observable requirement above.

Prior focused checks already passed: the existing terminal-summary/publication and recent-image
governance selectors reported **9 passed**. They verify present contracts, not the proposed lifecycle;
do not rerun unchanged checks for planning. Necessary new/changed behavior cases must come from
a fresh requirements-only author given accepted outcomes/public interface only, no history,
repository, implementation, tests, diffs or logs. Extract that contract; do not send this whole
implementation/evidence plan. Workers may integrate, not redesign those cases. Run affected
selectors first; the package table identifies existing check locations, not a demand to run every
case after every edit. Reuse passing checks until inputs change; broaden for a concrete risk/gate.

For qualifying Python source repairs retain frozen before/after full-source BasedPyright reports
under identical dependencies, command/source hashes and zero new diagnostic signatures using
`scripts/check_type_regressions.py`; follow `docs/migration/p6-error-attribution.md`. Run owned-file
`ruff check`, never `ruff format`. Strict-zero plus the required full suite closes P6; a bounded
repair does not close it or invent a P6-closure prerequisite for every correction. Report skips,
unavailable sandbox/runtime qualification and provider-wire uncertainty separately from passes.

## Integration, rollout and rollback

These fixes stay in existing provider/governor/tool owners. Mobile reconnect reattaches canonical
history/status; it neither retries unknown mutations nor changes permission/context authority.
Preserve [mobile-remote-request-plan.md](mobile-remote-request-plan.md), especially immutable
submitted snapshots, temporary disposal and the retained release/installation review gates.

Start with isolated route and ordinary long-session checks, then exceptional recovery and
disposable media/sandbox data. Run a bounded canary within the implementation task's selected
runtime/provider scope using sanitized inputs; preserve the live conversation. Planning makes no
provider calls or configuration changes. Keep deadlines/call ceilings and source retention while
replacing the fixed summary cap with resolved operation budgets. Revise the old compaction
guidance/cases intentionally with P2. Background precomputation remains optional later work.
Rollback stops new lifecycle/recovery/input mounting, invalidates affected ephemeral route/provider
state and uses retained source/checkpoints under a compatible reader. Quiesce newer writers before
downgrade; never delete source/artifacts, restore stale grants or publish a failed summary.

The preventive, provider-neutral direction is accepted for implementation planning. Actual wire
reasoning/vision behavior and canary performance remain unqualified; implementation must report
remaining agreed outcomes before claiming completion. Optional durable binary references/retention
need a separate concrete use case, not an additional gate on the confirmed corrections.
