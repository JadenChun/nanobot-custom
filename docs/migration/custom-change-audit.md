# Custom-change audit against pinned upstream

Audit date: 2026-09-08. This is a static compatibility audit, not a passing runtime baseline or a merge-readiness claim. Preserve all custom behavior through retained code or verified equivalents, except existing defects explicitly resolved through their focused decisions. Context enhancements remain in scope; sustained goals are optional follow-on work, not a migration requirement.

The audit accounts for **all 158 rename-aware common-base → staging diff entries**: 84 additions, 72 modifications and 2 deletions; 57 test files and 101 other files. Source findings cover 37 CORE, 17 PROVIDER, 41 EDGE and 6 REPO non-test files. See the [158-row inventory](p0-compatibility-inventory.tsv), [coverage/method limits](p0-coverage.md), [baseline gate](p0-baseline.md), and [migration plan](../nanobot-upstream-migration-plan.md).

Pinned references: common base `63d646f731a505731e97909d114ce257c6de36bc`; staging `4aa562d92828189b3fcefad079773efbab1d715a`; upstream `f47b8f23155ddec23cfb80ec001b544d495df89c`. Local evidence paths below are relative to `nanobot/` unless they start with `deploy/`, `docs/`, `tests/`, `pyproject.toml` or a repository dotfile; cited line numbers belong to the respective pinned tree. Repository-root files such as `README.md` and `.gitignore` are exceptions to the `nanobot/` prefix. Brace lists expand within their explicit directory. An unqualified sibling filename in the same evidence bullet inherits the preceding explicit directory (for example, `agent/tools/crawler.py, delegation.py` means `nanobot/agent/tools/delegation.py`); symbols without a `.py` suffix are method/contract shorthand, not additional file paths. The TSV uses the same evidence shorthand. Use `git show <sha>:<path>` to reproduce evidence. Tests cited are existing source evidence or explicitly labeled proposed/upstream cases. This static audit did not execute them. The subsequent isolated [P0 mapped baseline](p0-baseline.md) executed 61 staging files (868 cases: 866 passed, 2 failed); it does not establish upstream replacement equivalence. Therefore every per-finding `NOT-RUN` label below means the named migration replacement/differential validation remains unrun, not that no staging baseline test was executed.

## Conclusions and gates

- The largest conflicts are execution ABI/whole-batch authorization/terminal stops (CORE-01–03, EDGE-01–02), session namespaces and memory migration (CORE-04–05), config/provider state/auth contracts (PROVIDER-01–10), and cron session binding plus destination fanout (EDGE-15–16). These require adapters before upstream constructors, normalizers or dispatch can run against fork data.
- Prefer upstream provider/image/network/filesystem/channel implementations, then port demonstrated gaps. Overlap is not proven equivalence. Current SDK opt-in traces and HTTP API `stream:true → 400` are distinct from CLI/provider streaming; importing upstream SSE must not silently expand the API.
- Existing behaviors needing focused decisions are insecure Codex TLS retry, whole-turn API empty-result retry, non-atomic auth refresh/writeback, potentially racing key-pool pacing, and nondurable fanout deduplication. Their existence is not an instruction to preserve vulnerabilities or permission to fix them silently in a documentation task.
- Startup must preserve headless/classic defaults, logging and one foreground gateway. WebSocket omission enables upstream defaults; explicit `enabled:false` is honored at startup. Loader discovery cannot re-enable background spawn or bypass role boundaries. Current VPN exposure is flag-enabled in source/deployment examples; live state is unknown, and preservation does not require a retirement decision.
- Recover local memory receipts before constructing the upstream migrator that renames `HISTORY.md`; migrate cron origins/destinations before upstream strips legacy delivery fields and disables unbound jobs. Never assume source import is data migration.

Severity describes the potential migration consequence, not proof of a live incident. **Confirmed conflict** means incompatible source contracts; **custom-only port** means no equivalent was established; **compatible-overlap** means a promising upstream implementation still needs differential validation; **existing-defect decision** marks a pre-existing risk or stale claim requiring a scoped decision; **runtime-unverified** explicitly limits deployment/auth/tool assertions.

## Audited behavior groups
<a id="core-01"></a>
### CORE-01: Runner ABI and policy preflight

**HIGH · confirmed conflict. Owner: integration owner.**

- Local evidence: agent/runner.py AgentRunner(provider), AgentRunSpec, whole-batch preflight at 436; delegation._run_role / DelegatedReadOnlyPolicy.
- Upstream evidence: agent/runner.py AgentRunner(), AgentRunSpec.runtime LLMRuntime; agent/tools/execution.py execute_tool_calls.
- Preserve/adapt: Adapt runtime/spec ABI and preflight every batch before any call, including approval/recovery paths; an allowed read followed by a forbidden mutation executes neither.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_capability_policy.py; tests/agent/test_runner.py.
- Gap or decision: Add mixed allowed/forbidden batch fixture through upstream executor; test parent and delegated routes.

<a id="core-02"></a>
### CORE-02: Terminal tool outcomes

**HIGH · confirmed conflict. Owner: integration owner.**

- Local evidence: agent/runner.py _execute_tools / ToolOutcome, terminal branch near 508.
- Upstream evidence: agent/tools/execution.py execute_tool_calls sequential loop at 53-94 has no matching terminal branch.
- Preserve/adapt: Keep first terminal status, emit legal skipped results and stop later tools; approval_required/policy_blocked/error must not trigger another model/finalization/continuation call.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_turn_execution.py::test_nested_policy_stop_skips_later_write_in_same_outer_batch; tests/agent/test_turn_execution.py::test_contextual_tool_stop_does_not_make_an_extra_model_call; tests/agent/test_turn_execution.py::test_contextual_tool_error_is_terminal_without_an_extra_model_call.
- Gap or decision: Exercise imported execution path and nested read-only stop without policy elevation.

<a id="core-03"></a>
### CORE-03: Canonical foreground roles, turn ownership and hooks

**HIGH · confirmed conflict. Owner: integration owner + policy owner.**

- Local evidence: agent/{capabilities,delegation,policy,turn,write_guard,loop,hook}.py; execute_turn locks/approval CAS; deleted agent/subagent.py.
- Upstream evidence: agent/subagent.py SubagentManager; agent/tools/loader.py ToolLoader; agent/{turn_hooks,turn_delivery,automation_turns,cron_turns}.py; agent/hook.py CompositeHook.wants_streaming.
- Preserve/adapt: Keep exact immutable foreground registries and optional routing, worker reads under resource policy and scoped writes, no shell/nested delegation. Preserve callbacks, run-local budgets, locks, cancellation and approval CAS. Isolate optional hook failures without swallowing enforcement exceptions; preserve deliberate absence of background agents.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_capability_policy.py; tests/agent/test_turn_execution.py; tests/agent/test_correction_round.py; tests/agent/test_hook_composite.py; tests/agent/test_task_cancel.py; tests/agent/test_orchestrator_harness.py.
- Gap or decision: Snapshot main/all role registries and test upstream factory paths; wants_streaming exception isolation differs. Test cancellation lock release and alias approval races.

<a id="core-04"></a>
### CORE-04: Session namespace, CAS and history projection

**HIGH · confirmed conflict. Owner: session owner.**

- Local evidence: session/manager.py Session revision/generation, save at 441-513, get_model_history; workspace/sessions sanitized keys.
- Upstream evidence: session/manager.py FileSessionStore workspace-ID runtime namespace/encoded keys/checkpoint sidecars.
- Preserve/adapt: Dual-read and collision-map old/new keys; retain append suffix merge, identical messages, stale generation rejection and scheduled ring. Keep model projection distinct from traces; no reset resurrection or silent empty overwrite.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_memory_budget_hardening.py; tests/agent/test_correction_round.py; tests/agent/test_session_manager_history.py.
- Gap or decision: Round-trip legacy storage and generation races across processes; interrupted migration, moved workspace and colliding sanitized keys.

<a id="core-05"></a>
### CORE-05: Memory migration and receipt recovery

**HIGH · confirmed conflict. Owner: session/memory owner.**

- Local evidence: agent/memory.py MemoryStore.read_long_term (line 157), MEMORY.md/HISTORY.md, pending receipts, cross-process locks/digest checks/offset commit.
- Upstream evidence: agent/memory.py MemoryStore._maybe_migrate_legacy_history converts history.jsonl, Dream cursor, renames HISTORY.md backup; upstream read_memory (line 227) vs local read_long_term (line 157).
- Preserve/adapt: Recover local receipts before invoking upstream constructor/migrator. One archival owner preserves source, offsets, idempotence and supported legacy reader behavior. Do not silently rename history on startup or enable Dream.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_memory_budget_hardening.py::test_receipt_recovery_is_idempotent_and_offset_monotonic; tests/agent/test_correction_round.py::test_memory_receipt_digest_conflict_preserves_new_memory; tests/agent/test_memory_consolidation_types.py; tests/agent/test_consolidate_offset.py.
- Gap or decision: Copied MEMORY/HISTORY plus pending receipts: initialization/rename/crash/digest conflict/reload; explicit archival format mapping before activation.

<a id="core-06"></a>
### CORE-06: Context bounds, message representation and governor

**HIGH · confirmed conflict. Owner: context owner + provider owner; integration owner wires shared files.**

- Local evidence: agent/context_budget.py; agent/message_content.py; utils/prompt_budget.py reduce_messages_to_budget/assert_prompt_fits; helpers; _image_calls.
- Upstream evidence: agent/context_governance.py ContextGovernor; providers/conversation_state.py; session/summary.py; agent/runner.py.
- Preserve/adapt: Adopt one governor with exact finalized request guards for normal/streaming/retry/finalize/delegated/auxiliary paths. Keep newest legal turn, media/signatures, zero transport on irreducible requests; follow context design for new persistence.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_memory_budget_hardening.py; tests/agent/test_context_budget.py; tests/agent/test_runner_context_trim.py; tests/agent/test_gemini_thought_signature.py; tests/agent/test_loop_save_turn.py; tests/agent/test_loop_consolidation_tokens.py.
- Gap or decision: Upstream controller/guard ABI, actual serialized media/tool schemas, pending delta boundaries and context crash matrix remain untested.

<a id="core-07"></a>
### CORE-07: Typed run records and trace isolation

**MODERATE · custom-only port. Owner: run-store owner.**

- Local evidence: agent/run_store.py RunStore; agent/turn.py typed sanitized terminal records/run links.
- Upstream evidence: utils/run_records.py write_run_record automation schema is not equivalent.
- Preserve/adapt: Preserve typed RunRecord schema, atomic storage, collision/traversal safeguards, status and scheduled links; keep private metadata out of model history and context sync.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_run_store.py; tests/agent/test_turn_contracts.py.
- Gap or decision: Round-trip both record schemas via explicit adapter; do not alias solely because both write JSON.

<a id="core-08"></a>
### CORE-08: Context repositories, policy and safe sync

**HIGH · custom-only port. Owner: policy/context-repo owner.**

- Local evidence: context_repo/{manager,policy}.py ContextRepoManager/ResourceAccessPolicy; utils/git_sync.py ahead path/blob/remote revalidation; secret_scan.py.
- Upstream evidence: utils/gitstore.py memory Git store; no equivalent external-context policy/sync contract.
- Preserve/adapt: Preserve manifests, ordered context/skills/memory/run summaries, protected/writable/proposal-required paths, ahead-commit secret/path checks and unrelated dirt. Recheck new tools against policy; private runs never sync. Shell checks are not an OS sandbox.
- Existing migration-replacement validation targets (NOT-RUN): tests/context_repo/test_manager_policy.py; tests/context_repo/test_git_sync.py; tests/context_repo/test_git_sync_safe_push.py; tests/context_repo/test_git_sync_runs_policy.py.
- Gap or decision: Synthetic local bare remotes for advanced remote/ahead secret/path races; imported tool dispatch cannot bypass resource policy.

<a id="core-09"></a>
### CORE-09: Bootstrap templates and external skill precedence

**MODERATE · confirmed conflict. Owner: context/skills owner.**

- Local evidence: agent/context.py BOOTSTRAP TOOLS/IDENTITY + external manifest fallback/memory/run summaries; agent/skills.py extra paths; templates.
- Upstream evidence: agent/context.py _load_bootstrap_files project AGENTS + agent SOUL/USER; SkillsLoader plugin/disabled metadata, no local extra_paths.
- Preserve/adapt: Explicitly load retained TOOLS/IDENTITY/context material; preserve workspace > ordered external > builtin skill precedence. Copying files alone does not preserve loading behavior.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_context_prompt_cache.py.
- Gap or decision: Duplicate names/disabled external skills/project-vs-agent workspace fixtures; private context must stay scoped and cache invalidation correct.

<a id="core-10"></a>
### CORE-10: Bundled custom skills and video assets

**MODERATE · custom-only port. Owner: skills/release owner.**

- Local evidence: skills/README.md; desktop-use/mobile-app-testing/mobile-dogfood/webapp-testing/video-understanding SKILL.md; frame-selection.md; extract_keyframes.py.
- Upstream evidence: agent/skills.py metadata validation/discovery and pyproject.toml package assets; custom skills absent.
- Preserve/adapt: Preserve five skills, video reference/extractor, discovery metadata and required tool schemas; install runnable assets independently of repository cwd.
- Existing migration-replacement validation targets (NOT-RUN): tests/tools/test_video_understanding_extractor.py.
- Gap or decision: Installed-wheel discovery and local synthetic-video/FFmpeg full-path exercise; OCR/storyboard tests do not prove packaging.

<a id="provider-01"></a>
### PROVIDER-01: Config and logical-budget schema

**HIGH · confirmed conflict. Owner: config owner; integration owner owns schema.**

- Local evidence: config/schema.py MaxTokensConfig at 31-93, context_paths/context_repos/fallback; providers/base.py 646-669.
- Upstream evidence: config/schema.py AgentDefaults scalar max_tokens/context_window_tokens/fallback_models; custom extras absent.
- Preserve/adapt: Versioned lossless config adapter preserves custom consumers, total logical input budget with output reserve subtracted once, unknown custom fields and save/reload. Do not silently parse object maxTokens as upstream scalar.
- Existing migration-replacement validation targets (NOT-RUN): tests/config/test_config_migration.py; tests/agent/test_memory_budget_hardening.py.
- Gap or decision: Validate old/new/legacy key shapes, zero/empty values and direct Config construction; root static AST shape check passed, behavior NOT-RUN.

<a id="provider-02"></a>
### PROVIDER-02: Loader migrations and runtime paths

**MODERATE · custom-only port. Owner: config owner.**

- Local evidence: config/loader.py BOM at 42, empty paths at 65-67, missing-only ltx-desktop timeout at 72-111; paths.py credential/locks/artifacts helpers.
- Upstream evidence: config/loader.py UTF-8 loading/environment resolution; config/paths.py new workspace/web roots; custom helpers absent.
- Preserve/adapt: Preserve BOM acceptance, empty-context omission, explicit timeouts and exported custom paths while adopting upstream environment/web handling. Config._migrate_config is a static helper, not an automatic Pydantic validator.
- Existing migration-replacement validation targets (NOT-RUN): tests/config/test_config_migration.py; tests/config/test_config_paths.py.
- Gap or decision: Direct model_validate versus load_config/save round-trips, environment precedence and absent/explicit migration fields.

<a id="provider-03"></a>
### PROVIDER-03: API key pool, cooldowns and pacing

**HIGH · custom-only port; existing-defect decision. Owner: provider owner.**

- Local evidence: providers/openai_compat_provider.py key pool at 148-197/258-347/1499-1597, pacing 447-463, timeout 368-406; schema api_keys.
- Upstream evidence: providers/openai_compat_provider.py constructor at 511; config/schema.py single api_key, no custom pool/rate/timeout.
- Preserve/adapt: Preserve nonempty api_keys precedence including filtered-empty list, preferred/ordered cooldown-ranked selection, quota/rate/overload timings, provider pacing and minimum timeout. Anthropic/Azure currently use first key only. Do not silently require strict exclusion while all keys cool.
- Existing migration-replacement validation targets (NOT-RUN): tests/providers/test_provider_retry.py; tests/providers/test_litellm_kwargs.py; tests/config/test_config_migration.py.
- Gap or decision: Potential pre-existing pacing race: concurrent sleepers append without rechecking. Decide intended serialized pacing, then fake-clock all-cooling/preference/cancellation/partial-stream/concurrency tests; not a preserved defect requirement.

<a id="provider-04"></a>
### PROVIDER-04: Fallback policy and state propagation

**HIGH · confirmed conflict. Owner: provider owner.**

- Local evidence: providers/fallback_provider.py tuple constructor, 5-minute cooldown, quota/rate-only, primary override, no replay after delta.
- Upstream evidence: providers/fallback_provider.py primary/presets/factory constructor 124-150, broader breaker and segmented timeout recovery; context 164-191.
- Preserve/adapt: Explicitly adapt legacy fallback triggers/generation/override/no-replay semantics or version the behavior. Forward upstream context and retry callbacks rather than dropping kwargs; invalidate/recount concrete provider state.
- Existing migration-replacement validation targets (NOT-RUN): tests/providers/test_fallback_provider.py.
- Gap or decision: Matrix quota/nonquota/cooldown/partial output plus smaller windows, state usage and segmented timeout recovery; no accidental side-effect replay.

<a id="provider-05"></a>
### PROVIDER-05: Provider base ABI and final transport guards

**HIGH · confirmed conflict. Owner: provider owner.**

- Local evidence: providers/base.py subclass-wrapped guards 166-207, generation/image/error helpers.
- Upstream evidence: providers/base.py ProviderConversationState, CallContext, richer response/usage/retry callbacks.
- Preserve/adapt: Retain upstream call context/state/usage APIs and add guards at every concrete transport, including retry and streaming. Preserve malformed tool arguments for rejection rather than executable repair.
- Existing migration-replacement validation targets (NOT-RUN): tests/providers/test_provider_retry.py; tests/agent/test_memory_budget_hardening.py; upstream provider_sdk_retry_defaults tests.
- Gap or decision: Exercise direct/native/streaming/tools/retry actual payload boundaries; constructor wrappers alone do not prove coverage.

<a id="provider-06"></a>
### PROVIDER-06: Media fallback and provider-specific payload details

**HIGH · compatible-overlap. Owner: provider owner.**

- Local evidence: providers/base.py image pruning/missing notices/vision description before strip 452-607; compat tool images 711-867; registry OpenCodeGo/MiMo/MiniMax routing.
- Upstream evidence: providers/base.py stripping 1200-1260; providers/openai_responses/converters.py; providers/registry.py.
- Preserve/adapt: Port demonstrated recent-image/notices/_meta/vision gaps onto upstream converters. Preserve provider session headers, request IDs and endpoint-specific exclusions without recursive fallback, input mutation or replay.
- Existing migration-replacement validation targets (NOT-RUN): tests/providers/test_provider_retry.py; tests/providers/test_responses_api.py; tests/providers/test_litellm_kwargs.py.
- Gap or decision: Fixture table for media/tool-image, raw prefixes, vision fallback and OpenCode MiniMax rejection; all parser branches not audited.

<a id="provider-07"></a>
### PROVIDER-07: Reasoning and Responses routing

**HIGH · confirmed conflict. Owner: provider owner.**

- Local evidence: providers/openai_compat_provider.py reasoning/max_completion_tokens/temp/DeepSeek/Copilot use_responses_api at 555-656.
- Upstream evidence: providers/openai_compat_provider.py richer api_type/responses_models/efforts/circuitbreaker/state at 890-1092.
- Preserve/adapt: Use an explicit routing/parameter table for prefixed aliases, none/max effort, tools and explicit override. Adopt upstream converters and state without losing local family handling or combining conflicting predicates.
- Existing migration-replacement validation targets (NOT-RUN): tests/providers/test_reasoning_model_kwargs.py; tests/providers/test_responses_api.py; tests/providers/test_litellm_kwargs.py.
- Gap or decision: Run upstream openai_responses/responses_circuit_breaker/github_copilot_routing tests plus local matrix.

<a id="provider-08"></a>
### PROVIDER-08: Codex auth import, refresh and storage

**HIGH · confirmed conflict; existing-defect decision; runtime-unverified. Owner: auth owner.**

- Local evidence: providers/codex_auth.py oauthkit/official precedence 27-59, env/XDG/HOME/CODEX_HOME nested/flat paths 168-227, writeback 229-265.
- Upstream evidence: providers/openai_codex_provider.py oauthkit.get_token; providers/openai_codex_oauth.py login.
- Preserve/adapt: Preserve path/format/account/expiry and refresh/writeback compatibility through an explicit adapter. Existing non-atomic write plus no refresh lock requires a focused durability/ownership decision; never inspect live credentials for fixtures.
- Existing migration-replacement validation targets (NOT-RUN): tests/providers/test_codex_auth.py.
- Gap or decision: Synthetic path precedence, omitted refresh token, old/new formats, concurrent refresh/writeback and permissions. Installed oauthkit storage contract not verified.

<a id="provider-09"></a>
### PROVIDER-09: Copilot credential importer and refresh owner

**HIGH · confirmed conflict; runtime-unverified. Owner: auth/provider owner.**

- Local evidence: providers/copilot_auth.py NANOBOT_COPILOT_AUTH_PATH/cache bearer/github_token/expiry/endpoints; local factory custom headers win.
- Upstream evidence: providers/github_copilot_provider.py FileTokenStorage/OAuthToken and async refresh lock/enterprise routing.
- Preserve/adapt: One-way importer must preserve newer target credentials and custom override path, with one refresh owner. Retain custom header precedence. Local cached endpoint values are not honored by the current factory; do not invent that contract.
- Existing migration-replacement validation targets (NOT-RUN): tests/providers/test_copilot_auth.py.
- Gap or decision: Synthetic legacy/current/both stores, endpoint/config/header precedence and upstream concurrent/enterprise/routing checks.

<a id="provider-10"></a>
### PROVIDER-10: Codex constructor, system messages and TLS exception

**HIGH · confirmed conflict; existing-defect decision. Owner: provider/auth owner.**

- Local evidence: providers/openai_codex_provider.py workspace constructor and cache; TLS certificate-failure retry verify=False at 70-79; anthropic_provider.py joins all system messages 63-106.
- Upstream evidence: providers/openai_codex_provider.py proxy/extra_body/identity constructor at 58; anthropic_provider.py last-system assignment 233-242.
- Preserve/adapt: Adapt Codex workspace/session/state interfaces and combine ordered system blocks without losing cache annotations. Insecure TLS retry is an existing defect requiring an explicit migration decision; recommend verified TLS with configurable trust roots, not silent preservation or removal.
- Existing migration-replacement validation targets (NOT-RUN): tests/providers/test_openai_codex_provider.py; tests/providers/test_litellm_kwargs.py; tests/providers/test_provider_retry.py.
- Gap or decision: Multiple system messages and mixed cached blocks; certificate failure/custom trust/no insecure retry test after policy resolved. No TLS change in this audit.

<a id="provider-11"></a>
### PROVIDER-11: SDK results and opt-in trace privacy

**HIGH · confirmed conflict. Owner: SDK owner.**

- Local evidence: nanobot.py RunResult run_id/stop_reason/include_messages=False; per-turn detached sanitized hooks; docs/PYTHON_SDK.md.
- Upstream evidence: nanobot.py expanded facade/process_direct/run_streamed; sdk/types.py captured messages default, no matching local run_id.
- Preserve/adapt: Add canonical facade adapter retaining IDs, stop status, callbacks and opt-in detached sanitized messages. Preserve established SDK behavior; new streaming/features cannot silently alter trace privacy.
- Existing migration-replacement validation targets (NOT-RUN): tests/test_nanobot_facade.py; tests/agent/test_turn_contracts.py.
- Gap or decision: Concurrent hooks and default/opt-in trace tests; upstream facade/streaming comparison; exclude system/raw/media/tool args.

<a id="provider-12"></a>
### PROVIDER-12: HTTP API contract, lock ownership and empty retry

**HIGH · confirmed conflict; existing-defect decision. Owner: API owner + integration owner.**

- Local evidence: api/server.py execute_turn without outer lock 109-148, api: session_id, whole-turn retry on empty output; stream:true returns 400.
- Upstream evidence: api/server.py outer session locks + process_direct 293-440; SSE/media and fallback text instead of rerun.
- Preserve/adapt: Keep one serialization owner and preserve actual HTTP stream:true rejection, single-user validation and optional session_id unless separately expanded. Empty-result whole-turn retry can duplicate side effects: decide safe migration semantics explicitly, recommend reconcile/fallback rather than rerun.
- Existing migration-replacement validation targets (NOT-RUN): tests/test_openai_api.py::test_stream_true_returns_400; tests/test_openai_api.py.
- Gap or decision: No second-session-lock regression, concurrent sessions, empty result after durable tool effect and interleaved retries. No API/SSE enablement in this audit.

<a id="provider-13"></a>
### PROVIDER-13: Provider factory identity and release packaging

**MODERATE · confirmed conflict. Owner: factory/release owner.**

- Local evidence: nanobot.py embedded factory/crawler wiring 209-352; pyproject.toml crawler-worker/desktop extras/nanobot-crawl-worker.
- Upstream evidence: providers/factory.py presets/signature cache at 150-240; pyproject.toml cli.entry/build hooks/web assets.
- Preserve/adapt: Custom pool/rate/timeout/crawler fields must participate in cache/reload identity. Preserve crawler entrypoint, optional Darwin dependencies and installed assets additively; freeze build/channel dependencies explicitly.
- Existing migration-replacement validation targets (NOT-RUN): tests/test_nanobot_facade.py; tests/config/test_config_migration.py; tests/deploy/test_deployment_bundle.py.
- Gap or decision: Wheel metadata/entrypoint, clean optional imports, Darwin markers and cache invalidation/keepalive lifecycle; no build/install performed.

<a id="provider-14"></a>
### PROVIDER-14: Azure newline-only change

**LOW · compatible-overlap. Owner: provider owner.**

- Local evidence: providers/azure_openai_provider.py custom diff is EOF newline only.
- Upstream evidence: providers/azure_openai_provider.py and factory.py support base-only Azure auth.
- Preserve/adapt: No semantic custom port required for this file; keep upstream implementation and do not reinstate a key-only factory requirement by accident.
- Existing migration-replacement validation targets (NOT-RUN): tests/providers/test_azure_openai_provider.py.
- Gap or decision: Static diff classified only; enabled Azure/auth runtime remains unverified.

<a id="edge-01"></a>
### EDGE-01: Tool result and concurrency ABI

**HIGH · confirmed conflict. Owner: tool ABI owner + integration owner.**

- Local evidence: agent/tools/base.py ToolOutcome export, supports_parallel_calls, execute_with_context; registry.py.
- Upstream evidence: agent/tools/base.py ToolResult/error, read_only/concurrency_safe/exclusive and plugin hooks; registry recognizes ToolResult errors/caches definitions.
- Preserve/adapt: Map contextual terminal/policy metadata at runner boundary and errors to ToolResult.error; implement effective concurrency/read-only hooks. String compatibility alone loses failure semantics.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_turn_execution.py; tests/tools/test_tool_validation.py.
- Gap or decision: Imported runner/registry integration, definition-cache refresh and ToolOutcome-to-ToolResult mapping.

<a id="edge-02"></a>
### EDGE-02: Tool discovery, factories and name collisions

**HIGH · confirmed conflict. Owner: tool registry owner + integration owner.**

- Local evidence: manager-dependent agent/tools/crawler.py, delegation.py, explore.py; image.py generate_image.
- Upstream evidence: agent/tools/loader.py public concrete class discovery/enabled/create, builtins overwrite with warning; image_generation.py generate_image.
- Preserve/adapt: Provide explicit create(ctx)/scopes or opt out of discovery with deliberate registration. Enforce unique names and one image backend adapter; no default constructors bypass role/resource policies.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_capability_policy.py; tests/agent/test_orchestrator_harness.py; tests/agent/test_image_tool.py.
- Gap or decision: Snapshot main/all role registries via actual loader, missing dependencies and duplicate builtins/plugins. New discovered mutations disabled by default.

<a id="edge-03"></a>
### EDGE-03: Browser, device and desktop tools

**HIGH · custom-only port. Owner: custom tools owner.**

- Local evidence: agent/tools/agent_browser.py read-only classifier 184-205; agent_device.py 167-188; desktop_use.py serialized native tool.
- Upstream evidence: No equivalent upstream tool implementations; agent/tools/base.py concurrency/factory contracts.
- Preserve/adapt: Retain CLI schemas, cwd/screenshot/mutation restrictions and native serialization; translate supports_parallel_calls=False to upstream exclusive/concurrency safety, preserve platform config/guards.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_agent_device_registration.py; tests/agent/test_capability_policy.py; tests/tools/test_agent_browser_tool.py; tests/tools/test_agent_device_tool.py; tests/tools/test_desktop_use_humanize.py.
- Gap or decision: Headless imports and policy tests on all roles; native macOS/device/browser runtime unverified.

<a id="edge-04"></a>
### EDGE-04: Filesystem policy and upstream safety overlap

**HIGH · compatible-overlap. Owner: filesystem/policy owner.**

- Local evidence: agent/tools/filesystem.py secret-name block 16-28, /workspace alias 38-67, resource/write scopes/locks 69-137, SearchFiles.
- Upstream evidence: agent/tools/filesystem.py richer workspace helpers, read_only flags and device-path rejection.
- Preserve/adapt: Adopt upstream filesystem implementation; port missing secret filename, alias, resource-policy, scoped-write and lock semantics. Preserve upstream device protections and ToolResult behavior.
- Existing migration-replacement validation targets (NOT-RUN): tests/tools/test_filesystem_tools.py; tests/tools/test_tool_validation.py; tests/agent/test_capability_policy.py.
- Gap or decision: Cross-path/symlink/alias/protected-resource and concurrent writes via all imported tool factories.

<a id="edge-05"></a>
### EDGE-05: Shell and owned-process lifecycle

**HIGH · compatible-overlap. Owner: execution tools owner.**

- Local evidence: agent/tools/shell.py process cleanup/noninteractive git/desktop mutation/internal URL/workspace checks; process.py.
- Upstream evidence: agent/tools/shell.py config/factory/sandbox/env/stronger guards/exclusive; execution and _windows_job lifecycle.
- Preserve/adapt: Keep upstream exec/session/sandbox lifecycle; adapt custom owned-process and desktop-active mutation prohibitions, noninteractive environment and policy gaps without parallel process-management stacks.
- Existing migration-replacement validation targets (NOT-RUN): tests/tools/test_process.py; tests/tools/test_tool_validation.py; tests/tools/test_web_fetch_security.py.
- Gap or decision: Cancellation/process-tree/restart cleanup, sandbox/platform and desktop guard matrix; source guards do not prove OS isolation.

<a id="edge-06"></a>
### EDGE-06: Foreground crawler and bounded socket worker

**HIGH · custom-only port. Owner: crawler owner.**

- Local evidence: agent/tools/{crawler,delegation,explore}.py; social_crawl.py schema/state/serialized defense 250-339; crawler/worker.py serialized operations/HTTPS DNS allowlist, rejects click 606-608.
- Upstream evidence: agent/tools/loader.py/context factories; no equivalent foreground crawler worker.
- Preserve/adapt: Preserve role wrappers, model override, one serialized browser session, domain/operation/credential boundary and socket protocol. Current worker rejects all click; do not infer support from README.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_crawler_subagent.py; tests/agent/test_orchestrator_harness.py; tests/agent/test_capability_policy.py; tests/tools/test_social_crawl.py.
- Gap or decision: Imported factories plus gateway/worker version pair, denied click and DNS/redirect operations; live socket/browser unverified.

<a id="edge-07"></a>
### EDGE-07: Image tool name and backend contract

**MODERATE · confirmed conflict. Owner: image/provider owner.**

- Local evidence: agent/tools/image.py generate_image explicit output_path, Codex CLI/OpenRouter backend and turn prompt.
- Upstream evidence: agent/tools/image_generation.py same name, plugin config/artifact/reference-image/multiple-count contract.
- Preserve/adapt: One upstream tool/backend adapter preserves local output/backend/turn compatibility and upstream edit/count/artifact support; never register both classes with the same name.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_image_tool.py; upstream image-generation tests.
- Gap or decision: Schema aliases, backend selection, explicit output paths and artifact/edit/count cases; provider media gap cross-reference PROVIDER-06.

<a id="edge-08"></a>
### EDGE-08: VPN broker tools and official profiles

**HIGH · custom-only port; runtime-unverified. Owner: custom agent/tools/release owner.**

- Local evidence: agent/tools/trend_vpn.py four serialized tools and flag 252-255; trend_vpngate.py official profile validation/ranking; unit/env enable flag.
- Upstream evidence: No equivalent upstream tool or profile implementation.
- Preserve/adapt: Preserve current flag-controlled broker boundary, validation/ranking and exclusive factories. Historical retirement contradicted by current enabled configuration; only changing exposure needs a decision, not preservation.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_trend_vpn_tools.py; tests/agent/test_trend_vpngate.py.
- Gap or decision: Effective deployed environment/broker/VPN unverified; synthetic profile/flag/factory tests. Infrastructure separation is proposed, not existing.

<a id="edge-09"></a>
### EDGE-09: MCP classification and modern wrapper overlap

**HIGH · compatible-overlap. Owner: MCP/policy owner.**

- Local evidence: agent/tools/mcp.py annotation/trusted-local classifier, delegated tool_filter, timeout/union normalization.
- Upstream evidence: agent/tools/mcp.py sanitized names/reconnect/retry/ToolResult/resources/prompts/enabledTools; ordinary wrapper lacks read_only override.
- Preserve/adapt: Port trusted-local-only fail-closed classification/filter to upstream wrapper while keeping sanitization/reconnect/resources/prompts/restrictions. Never classify arbitrary remote heuristic matches as safe.
- Existing migration-replacement validation targets (NOT-RUN): tests/tools/test_mcp_tool.py; tests/agent/test_capability_policy.py.
- Gap or decision: Malformed annotations, remote lookalikes, sanitization collisions, reconnect policy and delegated resource/prompt access; real servers unverified.

<a id="edge-10"></a>
### EDGE-10: Deleted spawn and upstream background routes

**HIGH · confirmed conflict. Owner: integration/tool registry owner.**

- Local evidence: agent/tools/spawn.py deleted from staging; bounded foreground delegation replaces old background path.
- Upstream evidence: agent/tools/spawn.py factory/wait schema/background-or-inline dispatch at 90-107.
- Preserve/adapt: Keep background spawning absent under existing migration defaults; imports alone must not register spawn in main or delegated scopes. Foreground tooling remains optional, not a mandatory pipeline.
- Existing migration-replacement validation targets (NOT-RUN): tests/agent/test_orchestrator_harness.py; tests/agent/test_capability_policy.py.
- Gap or decision: Startup loader registry proof across enabled plugins and role paths; no hidden subagent task creation.

<a id="edge-11"></a>
### EDGE-11: Web safety routing and explicit message ledger

**MODERATE · compatible-overlap. Owner: web/message adapter owner.**

- Local evidence: agent/tools/web.py login-gated host routing/SSRF-safe fetch 235-264; message.py contextual send ledger/suppression.
- Upstream evidence: agent/tools/web.py evolved network safety; message.py current_request_context/_CURRENT_MESSAGE_SENDS/media safeguards.
- Preserve/adapt: Port only missing blocked-host config and contextual delivery ledger/suppression onto upstream safety and messaging, retaining media protections.
- Existing migration-replacement validation targets (NOT-RUN): tests/tools/test_message_tool_suppress.py; tests/tools/test_web_fetch_security.py.
- Gap or decision: Explicit send versus final response suppression, media-only sends and blocked-host redirects; no duplicate final result.

<a id="edge-12"></a>
### EDGE-12: Telegram authorization, fast commands and quota (CHAN-01)

**HIGH · confirmed conflict; existing documentation defect. Owner: Telegram/config owner + deployment documentation owner.**

- Local evidence: channels/telegram.py sender+allowChats checks 172-189/252-271/1002+, fixed fastCommands argv/cwd/timeout, quota/mentions/media. Existing deploy/README.md:52-55 incorrectly says allowFrom OR allowChats is sufficient; channels/base.py:113-121 denies empty allow_from, and channels/telegram.py:262-271 checks sender authorization before chat authorization.
- Upstream evidence: channels/telegram/runtime.py sender/pairing/mention logic but missing allow_chats/fast_commands/quota; manifest.py setup fields.
- Preserve/adapt: Add custom fields to runtime plus manifest/settings serialization; sender/pairing then chat authorization then mention policy before fast commands. Preserve quota/menu/mentions while keeping upstream commands, ordered ingress/webhooks/polling/retry. Correct deployment guidance during implementation: allowFrom is required for sender authorization; allowChats only adds the group constraint and cannot grant sender access. No runtime change is requested by this audit.
- Existing migration-replacement validation targets (NOT-RUN): tests/channels/test_telegram_channel.py; tests/cli/test_commands.py.
- Gap or decision: Validate docs/config examples and an allowChats-only config with empty allowFrom: it must remain denied before message/fast-command work. Test allowed sender plus denied group separately; add quota routing/reset display and settings round-trip. The README defect is pre-existing; this is not evidence of a deployed exploit.

<a id="edge-13"></a>
### EDGE-13: Media delivery and result-mode manager (CHAN-02)

**HIGH · compatible-overlap. Owner: channel delivery owner.**

- Local evidence: channels/manager.py taskUpdateMode=result progress gate; Telegram caption/media-only expectations.
- Upstream evidence: channels/manager.py per-channel transport defaults/concurrent ordered outbound; channels/telegram/runtime.py existing video/video-note/animation/security/stream splitting.
- Preserve/adapt: Keep upstream transport/media ordering and retries; retain result-only scheduled delivery, short captions and media-only behavior without replacing richer manager.
- Existing migration-replacement validation targets (NOT-RUN): tests/channels/test_telegram_channel.py; tests/channels/test_channel_plugins.py; tests/tools/test_message_tool_suppress.py.
- Gap or decision: Result-mode across progress/final/explicit media/destination fanout and imported transport defaults.

<a id="edge-14"></a>
### EDGE-14: GUI activation and settings semantics (CHAN-03)

**HIGH · confirmed conflict. Owner: startup/config owner.**

- Local evidence: Legacy configs omit GUI fields; existing systemd foreground gateway.
- Upstream evidence: channels/websocket/manifest.py default_enabled/always_enabled; manager default section; channel_instance_specs explicit enabled:false; webui action refuses toggle.
- Preserve/adapt: Migrate omission to explicit GUI disablement and preserve it on save/reload. Explicit false is honored at startup; always_enabled restricts UI toggles, not the config false. No new listener/setup mutation from ordinary startup.
- Existing migration-replacement validation targets (NOT-RUN): tests/channels/test_channel_plugins.py; tests/cli/test_commands.py; tests/deploy/test_deployment_bundle.py.
- Gap or decision: Headless subprocess listener/config/template/browser/build checks remain required; no GUI runtime test performed.

<a id="edge-15"></a>
### EDGE-15: Cron binding and destructive legacy normalization (CRON-01)

**HIGH · confirmed conflict. Owner: cron/session owner.**

- Local evidence: cron/types.py CronDestination/additional_destinations/delivery_destinations and occurrence/run fields; service persistence/locks/TZ; agent/tools/cron.py.
- Upstream evidence: cron/types.py session_key/origin canonical payload; service.py 132-164 clears legacy channel/to/deliver, 198-224 disables unbound jobs.
- Preserve/adapt: Before upstream normalization, bind legacy jobs to valid session origins and migrate primary/additional destinations plus IDs/state/TZ. Preserve one execution, serialization and occurrence records; do not merely retain unused legacy fields.
- Existing migration-replacement validation targets (NOT-RUN): tests/cron/test_cron_service.py; tests/cron/test_cron_tool_list.py; tests/cron/test_cron_delivery.py.
- Gap or decision: Old bound/unbound/multidest jobs conversion twice, false deliver/next-fire/disabled states and source counts. Deploy README legacy execution-context claim must be revised during implementation.

<a id="edge-16"></a>
### EDGE-16: Cron fanout and existing receipt gap (CRON-02)

**HIGH · confirmed conflict; existing-defect decision. Owner: cron/delivery owner.**

- Local evidence: cron/delivery.py and cli _execute_cron_job transient (channel,to,content,media) dedupe, no durable per-destination receipt.
- Upstream evidence: cron/{bound_runner,session_delivery}.py and cli/gateway_runtime.py run/provenance/suppression.
- Preserve/adapt: Integrate one execution and fanout at upstream bound-delivery layer with run/occurrence metadata. Lack of durable destination receipts is an existing robustness gap, not introduced by upstream; choose reconciliation/idempotence behavior explicitly.
- Existing migration-replacement validation targets (NOT-RUN): tests/cron/test_cron_delivery.py; tests/cron/test_cron_service.py; tests/tools/test_message_tool_suppress.py.
- Gap or decision: Partial fanout/crash/restart/uncertain external send; tests currently prove in-memory dedupe, not exactly-once delivery.

<a id="edge-17"></a>
### EDGE-17: CLI module split and classic/headless defaults (CLI-01)

**HIGH · confirmed conflict. Owner: CLI/integration owner.**

- Local evidence: cli/commands.py classic entry/agent rendering and gateway --log-file; deployment ExecStart.
- Upstream evidence: cli/entry.py and cli/agent.py native TUI default; cli/gateway.py no --log-file.
- Preserve/adapt: Keep upstream module structure but preserve staging bare/interactive/one-shot/classic defaults and log option/rotation. Imported TUI is optional capability, not a default replacement; exact service command must parse.
- Existing migration-replacement validation targets (NOT-RUN): tests/cli/test_commands.py; tests/cli/test_cli_input.py; tests/deploy/test_deployment_bundle.py.
- Gap or decision: Real headless subprocess/help/ExecStart/rotation checks; no frontend download/build or second gateway.

<a id="edge-18"></a>
### EDGE-18: CLI custom commands, onboarding and rendering (CLI-02)

**MODERATE · compatible-overlap. Owner: CLI owner.**

- Local evidence: cli/onboard.py max-token/effective-key display; cli/stream.py duplicate-log suppression; command/builtin.py quota/stop/restart/new; commands provider/crawler/artifacts/cron.
- Upstream evidence: split CLI modules, providers/factory.py, upstream richer restart/notices/builtins.
- Preserve/adapt: Port quota, required aliases, rendering/log sink and onboarding semantics into upstream apps; preserve upstream optional capabilities without changing classic defaults or current API stream rejection.
- Existing migration-replacement validation targets (NOT-RUN): tests/cli/test_commands.py; tests/cli/test_cli_input.py; tests/cli/test_restart_command.py.
- Gap or decision: Quota/reset timezone, stop/new/restart, one-shot streaming logs and secret-redacted effective-key output.

<a id="edge-19"></a>
### EDGE-19: Gateway unit executable contract (DEPLOY-01)

**HIGH · confirmed conflict; runtime-unverified. Owner: release/CLI owner.**

- Local evidence: deploy/systemd/nanobot-gateway.service ExecStart --config/--log-file, hardening and writes.
- Upstream evidence: cli/gateway.py lacks --log-file option.
- Preserve/adapt: Keep existing foreground systemd lifecycle/permissions/paths and compatible log option before rollout; no GUI launcher or second gateway. Current service state is unverified.
- Existing migration-replacement validation targets (NOT-RUN): tests/deploy/test_deployment_bundle.py.
- Gap or decision: Current substring tests insufficient: actual help/argument parse and isolated ExecStart smoke with log rotation.

<a id="edge-20"></a>
### EDGE-20: Dependency install and deploy side effects (DEPLOY-02)

**HIGH · confirmed conflict. Owner: release owner.**

- Local evidence: deploy/scripts/deploy_nanobot_gateway.sh base-only install, pip upgrade, rsync --delete, forced autoSync and conditional restart.
- Upstream evidence: channel manifests own requirements; manager.ensure_enabled_channel_dependencies; protected unit venv not writable.
- Preserve/adapt: Preinstall frozen enabled-channel dependencies in artifact; separate install/config conversion/activation and preserve deliberate auto-sync after cutover. Never rely on service-time dependency mutation under ProtectSystem.
- Existing migration-replacement validation targets (NOT-RUN): tests/deploy/test_deployment_bundle.py.
- Gap or decision: Clean immutable install/import enabled channels under service permissions, no network install; script dry-run separation and copied config diff.

<a id="edge-21"></a>
### EDGE-21: Crawler deployment and reproducibility (DEPLOY-03)

**MODERATE · custom-only port; runtime-unverified. Owner: release/crawler owner.**

- Local evidence: deploy crawler env/script/unit range dependencies, mutable checkout/pip upgrade, dedicated user/socket0660/resource caps.
- Upstream evidence: No equivalent service; upstream project release dependency/build workflow.
- Preserve/adapt: Preserve worker isolation and protocol while freezing build/browser/OS inputs; test gateway/worker rollback pair. No live systemd/socket/browser verification claimed.
- Existing migration-replacement validation targets (NOT-RUN): tests/deploy/test_deployment_bundle.py; tests/tools/test_social_crawl.py.
- Gap or decision: Installed crawler entrypoint/health/socket permissions/limits and locked rebuild; README click contradiction documented in EDGE-06.

<a id="edge-22"></a>
### EDGE-22: Deployment examples, updater and mutable settings (DEPLOY-04)

**HIGH · confirmed conflict; runtime-unverified. Owner: config/release owner.**

- Local evidence: deploy/README.md, deploy/nanobot-config.example.json and deploy/nanobot.env.example custom context/crawler/Telegram/tool fields; deploy/scripts/update_nanobot_gateway.sh main default/fetch/switch/ff-only/deploy. deploy/README.md:52-55 has the existing allowFrom OR allowChats defect established by EDGE-12.
- Upstream evidence: upstream Config scalar tokens/new channels; WebUI settings rewrite config/templates.
- Preserve/adapt: Preserve all deployment extensions and environment precedence through schema/settings adapters; do not copy legacy config blindly. Pin accepted fork SHA/artifact and separate updater side effects. Correct stale cron/click documentation and the allowFrom OR allowChats claim during implementation; sender allowFrom is required and allowChats adds a group constraint.
- Existing migration-replacement validation targets (NOT-RUN): tests/deploy/test_deployment_bundle.py; tests/config/test_config_migration.py.
- Gap or decision: Validate sanitized example old/new/save/reload, no settings field loss and exact selected branch/artifact. Documentation/config validation must reject the claim that allowChats substitutes for allowFrom; see EDGE-12 sender-first negative cases. Actual live configuration remains unknown.

<a id="repo-01"></a>
### REPO-01: Ignore rules and lockfile assumption

**LOW · custom-only port. Owner: release owner.**

- Local evidence: .gitignore adds .tmp/; both pinned trees ignore uv.lock.
- Upstream evidence: .gitignore has tmp/ and *.tmp, but not the fork's .tmp/ rule.
- Preserve/adapt: Retain the custom temporary-directory exclusion. A committed lockfile is a future release requirement/decision, not an existing tracked artifact; explicitly resolve its ignore policy.
- Existing migration-replacement validation targets (NOT-RUN): Static git diff and git ls-tree/.gitignore inspection.
- Gap or decision: No runtime semantics; future release owner must validate frozen production/channel dependencies.

<a id="repo-02"></a>
### REPO-02: README and historical architecture documents

**MODERATE · existing-defect decision. Owner: documentation/integration owner.**

- Local evidence: README.md claims inherited behavior unchanged; docs/ORCHESTRATOR_HARNESS_IMPLEMENTATION_PLAN.md and UNIFIED_CONTEXT_REFACTOR_PLAN.md contain older pipeline/compression descriptions.
- Upstream evidence: README.md current upstream product; custom historical plans absent.
- Preserve/adapt: Keep historical documents as history, not authoritative current behavior. Migration evidence and current contracts supersede stale claims; update user documentation when implementation changes ship.
- Existing migration-replacement validation targets (NOT-RUN): Static document-to-code comparison in independent source audits.
- Gap or decision: No doc claim establishes compatibility; review installed user guidance against final accepted contracts.

<a id="repo-03"></a>
### REPO-03: Local development permissions

**LOW · custom-only port. Owner: repository owner.**

- Local evidence: .claude/settings.local.json contains 54 development permission entries.
- Upstream evidence: No corresponding Nanobot runtime or packaging contract.
- Preserve/adapt: Developer-local metadata only; do not import permissions into Nanobot authorization or expose raw entries in audit artifacts.
- Existing migration-replacement validation targets (NOT-RUN): Static scope and packaging reference review.
- Gap or decision: Explicit nonruntime disposition; no application validation required and no cleanup authorized.

<a id="repo-04"></a>
### REPO-04: Finder metadata

**LOW · compatible-overlap. Owner: repository owner.**

- Local evidence: .DS_Store binary metadata.
- Upstream evidence: No runtime behavior counterpart.
- Preserve/adapt: Explicitly exclude from behavioral preservation requirements while retaining this diff entry in accounting; do not delete or clean it during audit.
- Existing migration-replacement validation targets (NOT-RUN): Static diff status/type inspection.
- Gap or decision: Nonruntime metadata; no semantic test needed.

## Completion limits

Each changed file maps exactly once to one inventory row, with multiple finding IDs where it participates in several contracts. Group mappings cover named changed behaviors and symbols; they do not prove every parser branch or every execution path was examined. In particular, the large OpenAI-compatible provider, GUI settings mutations and distributed crash boundaries require the stated focused tests. Exact set accounting is not semantic completeness.

Independent source auditors supplied CORE, PROVIDER and EDGE evidence; the parent checked critical memory/cron/auth/API/startup claims and statically checked schema/runner AST shapes. This documentation planner was reused because of thread capacity; it was not a fresh independent reviewer. The source audit itself exercised no tests, build, imports, provider sessions, credentials, live state, Telegram/MCP/VPN/crawler/systemd or external deliveries. The subsequent [isolated mapped baseline](p0-baseline.md) ran 61 staged files and records 866 passes plus two unresolved P0 failures. Future synthetic fixtures remain affected-adapter gates once their contracts are defined. P0 remains incomplete pending disposition of its two baseline gaps and independent checkpoint review. No code, schema, deployment or data changes are made by these documents.
