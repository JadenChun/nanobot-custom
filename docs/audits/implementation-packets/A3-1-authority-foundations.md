# A3-1 implementation packet — Canonical truth and authority foundations

**Status.** Ready for packet review. Implementation depends on the accepted A3-0 baseline.

**Planning source.** `origin/staging` at `29612c3df01dc3dce3501fe349bcf555cd6c58b6`.
The frozen audit source is `9562e57f8333d7f14b6aaeae837660723a9b2650`.
The source delta is the bundled skill-path fix and its test. K01-STG-F01 tracks the remaining path collision.
[I05 reconciliation](../module-review-2026-11-06/integrated/I05-final-reconciliation.md) · [K01 addendum](../module-review-2026-11-06/reports/K01.md) · [A3-0 baseline packet](A3-0-qualification-baseline.md) · [A3 roadmap](https://chatgpt.com/c/6aef25bd-7db8-83ec-bc98-f7fbf21da9d0)

## Objective

Make each selected runtime fact come from the component that can prove it.
The stage covers instance roots, file targets, archive provenance, source outcomes, provider attempts, tool guidance, and input validation.
Use the existing runtime owners and connect each change to a live consumer.
Keep the existing SDK signatures, protocol fields, durable record formats, and feature set.
The changes correct specific wrong outcomes at those existing interfaces.
Keep fact types with the owners that produce them.
Reuse existing types when they carry the required fact.
Client data, model output, and provider response fields cannot supply server-owned authority or commit proof.

## Entry gate

- The coordinator reviews this packet before dispatch. Finish the A3 packet series before product implementation starts.
- The coding agent starts from the revision in the accepted A3-0 report, in a clean implementation worktree based on `staging`.
- The A3-0 report supplies the source and dependency identities, baseline check results, and exact pre-existing failure signatures.
- If `origin/staging` has advanced, the coordinator checks the changed owners and updates this packet where the changes affect it.
- If a baseline failure makes a required outcome impossible to verify, the coordinator resolves that evidence gap before the affected work starts.

The audit worktree contains the handoff documents. The coding agent uses the implementation worktree for product changes.
One agent owns the write set. Apply the slices below in order so shared files have one writer.

## 1. Instance roots and effective file targets — R01

### 1a. Use the bound media root for configured image generation

**Evidence.** `Config.bind_source_path` records the instance config path.
`RuntimeComposition` captures preflight paths, and `AgentLoop.from_config` binds the artifact reader to the config's media root.
The configured `generate_image` factory and its reload path can still write generated images under the ambient media root.
This is the media part of C00-F06. [C00 source review](../module-review-2026-11-06/reports/C00.md)

**Change.** Pass the reader's captured media root to the configured image tool through its existing construction inputs.
Pass that root to the generated-image writer.
Preserve it when the owner replaces the tool or enables it through reload.
Where the owning entry point supplies an explicit config path, use that path for the image settings lookup.
Retain legacy defaults for callers that supply no bound root.
Keep `save_dir` containment, artifact metadata, and explicit output-file policy checks.
Use the current preflight, config, and tool context for these instance facts.

**Write set.** `nanobot/agent/loop.py`, `nanobot/agent/tools/context.py`, `nanobot/agent/tools/image_generation.py`, and the generated-image writer in `nanobot/utils/artifacts.py`.
Use `nanobot/runtime_composition.py` only if the existing construction boundary needs to pass the captured root.
Keep the MCP media writer, network whitelist, and other instance credential stores outside this slice.

**AC1a.** Start the SDK with an explicit config and a different ambient config.
With a fixture provider, the built-in tool writes its image and sidecar under the explicit instance's media root.
The instance reader can read the artifact, and the ambient media root receives no artifact.
Verify the same result after the existing runtime-control reload.
Verify that the default caller and denied output paths retain their behavior.
Start with `tests/tools/test_image_generation_tool.py`, `tests/tools/test_migration_image_adapter.py`, and `tests/agent/test_configured_artifact_store.py`.

### 1b. Use the same target for risky-file checks and writes

**Evidence.** `RiskyActionPolicy` resolves relative paths from agent home.
The built-in write and edit tools resolve paths from the live server-owned scope.
Those paths can differ for an authorized selected project. This is H01-F4.
[H01 source review](../module-review-2026-11-06/reports/H01.md)

**Change.** Validate the built-in `write_file` or `edit_file` arguments through the existing tool boundary.
Use that tool's `resolve_write_path` result for risk assessment.
Use the same admitted scope and effective target for dispatch.
If the resolved target changes before dispatch, stop the write and use the existing refusal or approval path.
Keep live path, grant, and credential-file checks at execution.
If target resolution denies access, return the existing denial without reading the target for risk assessment.
Keep the current approval threshold, batch gate, and approval interface.
The target is a server-derived fact. Client fields, tool arguments, and display paths do not confer authority.

**Write set.** The file-risk path in `nanobot/agent/policy.py`, the policy wiring in `nanobot/agent/loop.py`, and the built-in file dispatch in `nanobot/agent/tools/execution.py` and `nanobot/agent/tools/filesystem.py`.
Keep general tool registration and shell policy outside this slice.
Keep cross-turn approval binding and the structured approval cutover for A3-4.

**AC1b.** A risky edit or overwrite in an authorized project requires approval when the same relative path under agent home is absent or small.
Neither file changes before approval. An admitted write applies to the assessed project target.
A revoked grant or changed target prevents the write.
WebUI messages still reject caller-supplied `workspace_scope` and `source_token`.
Start with `tests/agent/test_migration_engine.py`, `tests/agent/test_workspace_scope.py`, and the ingress cases in `nanobot/channels/websocket/tests/test_websocket_channel.py`.

## 2. Archive provenance — R05

**Evidence.** The legacy archive can fall back to bounded raw transcript text.
`ContextBuilder.build_system_prompt` currently includes the stored summary in the system message.
Old summaries have no trusted flag that distinguishes a generated summary from a raw fallback.
This is K01-F12. [K01 source review](../module-review-2026-11-06/reports/K01.md)

**Change.** Keep the legacy summary in the existing durable record.
Project it once as historical data below system and developer authority.
Use the assistant-role historical summary convention of the canonical `CheckpointNode`.
Apply the rule to both a new compaction and a summary loaded from an older session.
Keep the active user input and history order.
Include the historical data in budget calculations and reuse the existing request-binding rules for provider state.
If the changed prefix invalidates a provider continuation, rebuild the request through the existing portable path.
Determine provenance from the server-owned input and its construction path. Text and labels inside the summary carry no provenance.

**Write set.** Legacy transcript construction in `nanobot/agent/context.py` and its compaction caller in `nanobot/agent/context_governance.py` if needed.
Keep canonical checkpoint admission, summary durability, and Dream write custody outside this slice.

**AC2.** A failed archive and a loaded legacy checkpoint both supply their retained text once as historical context.
Neither places the text in a system or developer message, including after provider request conversion.
The next turn still receives its current user input, and a saved session retains its existing history.
Start with `tests/agent/test_context_builder.py`, `tests/agent/test_context_prompt_cache.py`, and `tests/providers/test_bound_conversation_receipts.py`.

## 3. Required source writes and terminal outcomes — R06

**Evidence.** `SourceTurnWriter` fences further writes after a mandatory recording failure.
`AgentRunner._drain_injections` can catch that failure as an optional callback error.
A late accepted continuation can also carry a prior error into a later successful result.
These are H01-F1 and H01-F2. [H01 source review](../module-review-2026-11-06/reports/H01.md)

**Change.** Keep the source writer's committed receipt as proof of a recorded follow-up.
Propagate a mandatory drain failure to the owning turn and prevent further dispatch or successful completion after the writer is fenced.
Preserve the existing retained input, republish path, and cancellation behavior.
Keep optional legacy callback failures recoverable.
When a continuation is admitted, recompute the terminal disposition for the new attempt.
Keep evidence of earlier attempts in history and usage where the current owner records it.
Return an ordinary recoverable tool error to the model under the current tool-error policy.

Separate the effect from its reply. A durable receipt can prove `APPLIED`.
A rejection before the effect can prove `NOT_APPLIED`.
If an effect may have occurred, keep it `UNCERTAIN` until the owner has proof.
Use `PARTIAL` only when the owner can prove that a subset applied.
Use the existing receipt and failure paths for this slice. Keep this distinction internal to the owning boundary.
An exception or timeout alone does not justify replay.

**Write set.** The injection and terminal-result paths in `nanobot/agent/runner.py` and `nanobot/agent/loop.py`.
Use `nanobot/session/turn_writer.py` only to propagate existing write proof or failure.
The durable WebUI mutation outcome projection in E02-F20 belongs to A3-4.

**AC3a.** A mandatory source-write failure during either drain prevents a subsequent model request and a success result.
The input remains recoverable through the existing source owner.
If the commit response is lost, recovery uses the original source identity and applies the input at most once.
An optional callback failure still follows the legacy recovery path.

**AC3b.** If the turn accepts a late follow-up after an earlier model error and then succeeds, the public turn result reports success.
A later real error or cancellation still reports its own terminal state.
An ordinary tool failure can still be followed by a successful model response.
Start with `tests/agent/test_runner_injections.py`, `tests/agent/test_runner_errors.py`, `tests/session/test_source_turn_writer.py`, and `tests/migration/test_local_runtime_activation.py`.

## 4. Prepared provider attempt — R08

**Evidence.** `PreparedRequestRoute` already binds the selected leaf to its request envelope and limits.
If the primary fails and fallback differs from the prepared route, `ProviderRouteChanged` triggers one reprepare.
The reprepare can start at the primary again. This is H03-F3.
[H03 source review](../module-review-2026-11-06/reports/H03.md)

**Change.** Carry the consumed candidate and next eligible candidate through the existing typed route-change and request context.
Only the fallback owner may advance the candidate after a fallback-eligible result.
If the route changes before dispatch, reprepare without marking a candidate consumed.
Apply the same rule to normal and no-tools dispatch.
Keep the retry bound, source ticket discard, provider-state binding, and single-attempt auxiliary behavior.
Calculate the fallback leaf's own limits before it receives the request.
Keep continuation after visible deltas subject to the existing retry rules.
Reset the candidate state for each logical request.
Keep the public usage record and credentials outside the route fact.

**Write set.** Route preparation and rejection in `nanobot/providers/request_budget.py`, `nanobot/providers/base.py`, and `nanobot/providers/fallback_provider.py`.
Keep runner changes to the two reprepare branches in `nanobot/agent/runner.py`.
Only touch the route preview path in `nanobot/agent/context_governance.py` if the existing context does not already carry the candidate.
Explicit stream terminal validation, physical-attempt metrics, and OAuth generations remain separate R08 work.

**AC4.** For a prepared primary that fails and admits fallback, the same logical request reaches the fallback without replaying the consumed primary.
The fallback receives a request that satisfies its own final limits, or the runtime rejects it before dispatch.
A pre-dispatch route change and a single-attempt auxiliary request retain their existing guarantees.
Start with `tests/agent/test_runner_fallback.py` and `tests/providers/test_final_request_budget.py`.

## 5. Core tool guidance — R15

**Evidence.** The default core registry omits `apply_patch`, `grep`, and `find_files`.
Shipped prompt guidance names them as available. T01-F10 and K01-F11 share the `apply_patch` fact.
[I03 capability review](../module-review-2026-11-06/integrated/I03-client-capabilities.md)

**Change.** Remove the unconditional availability claims for those three tools from active built-in guidance.
The existing tool definitions describe any of them that an operator has registered and admitted for the request.
Treat declared, configured, registered, ready, and authorized as separate facts.
The scoped registry and existing policy supply the executable surface. Prompt text supplies guidance.
Keep feature activation and user-owned prompt files as they are.

**Write set.** Built-in claims in `nanobot/templates/agent/tool_contract.md` and `nanobot/templates/agent/identity.md`.
If another active built-in template repeats one of the three claims, include only that reference.
The registry and factory manifest are reference inputs for this slice.
Dynamic readiness, complete prompt parity, and the bundled-skill origin collision belong to A3-6.

**AC5.** A default model request contains no built-in claim that the three absent tools are available.
Its tool definitions still match the scoped registry. Existing configured tools and filtered requests keep their current execution policy.
Start with `tests/agent/test_context_builder.py` and `tests/agent/test_tool_loader_scopes.py`.

## 6. Validate input before work — R22

**Evidence.** `TurnRequest` can compare a media-normalized prompt with the original `message` alias.
The bundled keyframe script can enter a gap loop with a non-progressing interval.
These are H01-F3 and K01-F10. [H01 source review](../module-review-2026-11-06/reports/H01.md) · [K01 source review](../module-review-2026-11-06/reports/K01.md)

**Change.** Check text alias conflicts before media adoption. Apply the existing input presence rules.
Adopt media once, then project the same normalized text to `prompt`, `content`, and `message`.
Keep the other constructor aliases and media behavior.
At CLI argument validation, reject a non-finite or nonpositive `--max-gap-seconds`.
At the gap calculation, reject a step that cannot advance the timestamp before it can loop.
Keep the current defaults and valid extraction path.

**Write set.** `TurnRequest` in `nanobot/agent/turn.py` and `nanobot/skills/video-understanding/scripts/extract_keyframes.py`.

**AC6a.** Equal legacy and current text aliases with a media marker produce the same normalized text and one attachment result.
An empty normalized result stays empty. Conflicting aliases fail before attachment adoption.

**AC6b.** An invalid gap exits with an argument error before extraction or output creation.
A value that cannot advance at the current timestamp exits without looping.
The normal default still produces the expected frames.
Start with `tests/agent/test_turn_contracts.py`, `tests/attachments/test_attachment_adoption.py`, and `tests/tools/test_video_understanding_extractor.py`.

## Validation and handoff

- Run the closest existing checks for each slice with `uv run --no-sync python -m pytest` and the listed test targets.
  Use a focused boundary check when existing tests do not verify the criterion.
- For new or behavior-changing test cases, request a fresh requirements-only author under `AGENTS.md`.
  Give the author the public interface and observable criterion. Keep implementation material with the coding agent.
- Run `uv run --no-sync ruff check nanobot tests conftest.py` against the accepted dependency environment.
- Capture the full-source strict report before and after each bounded repair with `uv run --no-sync basedpyright --outputjson`.
  Use the same checker, configuration, and dependencies for each pair.
  Retain source hashes, reports, exact commands, and environment evidence.
  Set the variables below to the two retained reports and their corresponding source roots.
  Require zero introduced diagnostic signatures across all files.
  [Repository checker contract](../../migration/p6-error-attribution.md)
- Compare observed failures with A3-0 by signature and scope. Preserve the original baseline and explain each new result.
- Revert a failed slice and its dependent changes as a unit on the implementation branch.
  This stage requires no conversion of existing session or replay records.
- Return the source revision, the exact changed paths, the result for each AC, the commands and outcomes, and the remaining gaps by finding ID.
  Include evidence links for the before and after reports.

```sh
uv run --no-sync python scripts/check_type_regressions.py \
  "$A3_CHECK_BEFORE_JSON" "$A3_CHECK_AFTER_JSON" \
  --baseline-root "$A3_CHECK_BEFORE_ROOT" --current-root "$A3_CHECK_AFTER_ROOT"
```

A3-1 passes when the coordinator can verify every criterion and the applicable repository checks.
The next packet to prepare is A3-2, runtime and process ownership.
Implementation of A3-2 and A3-3 depends on accepted A3-1 evidence.
