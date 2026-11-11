# A3-6 implementation packet — Capability readiness and workbench

**Status.** Ready for packet review. Dispatch after A3-5 passes.
[Delegation cutover](A3-5-delegation-cutover.md) · [Tool findings](../module-review-2026-11-06/reports/T01.md) · [Provider findings](../module-review-2026-11-06/reports/H03.md) · [Skill findings](../module-review-2026-11-06/reports/K01.md)

## Objective and boundary

Advertise the capability that the current request can actually use.
Keep provider, MCP, skill, and UI state with their existing owners.
Publish replacement resources after validation and readiness. Preserve live authorization when a request crosses a generation change.
Use the A3-1 attempt and provenance rules and the A3-4 mutation outcome contract.

## 1. Publish tool and MCP generations at readiness — R15, R17

Prepare each server connection and its wrappers privately. Use a bounded readiness deadline.
Validate names and the supported schema subset before publishing definitions.
Reject schemas whose advertised constraints the local validator cannot enforce.
Share the initial readiness result with concurrent callers.
Publish all ready bindings for one server together. Close failed candidates and retain the old healthy generation when still authorized.
Keep one connection owner for each admitted generation. Bind the model request's schema to that generation at dispatch.
If a replacement occurs, admitted calls may use the retained binding while their live authority still permits it.
If the binding expired or access was revoked, refuse execution. Do not rebind the call by tool name.
Close an old generation after its admitted work settles. Require disposal for a plugin that acquires resources before publication.

**Edit boundary.** `nanobot/agent/tools/mcp.py`, `nanobot/agent/tools/registry.py`, and `nanobot/agent/tools/loader.py`.
Use the A3-2 transport owner and A3-5 role policy.
Keep the current provider-facing schema shape and plugin entry points where they meet the contract.

**AC6.1.** A failed or hanging MCP start publishes no tools. Concurrent initial callers observe one bounded result.
A failed replacement leaves the authorized old connection usable. An admitted call uses its original binding or receives a refusal after revocation.
The old resource closes once after its last call. A rejected schema never reaches the model.

## 2. Keep provider terminal, attempt, and credentials truthful — R01, R04, R08, R17

In the Responses stream adapter, require an explicit terminal event before accepting output as complete.
Treat clean EOF without a terminal event as transport failure.
Keep visible partial output separate from a usable terminal result. Do not dispatch incomplete tool calls or retry after visible delivery.
Count physical transport attempts at the existing transport observer. Keep the current logical usage record contract.
Preserve the A3-1 fallback cursor and final request limits. Refresh route-owned capability facts when model or window identity changes.

Give each OAuth login intent a provider and credential-profile generation.
On cancel, expiry, logout, or supersession, revoke publication before a late exchange can save credentials.
Use the A3-1 bound instance paths for MCP runtime, CLI Apps state, and usage reads and writes.
Bind network policy at the existing validation boundary. Loading another config cannot replace a live instance's allowlist.
Keep install arguments and executable metadata bound to the admitted CLI Apps catalog source.
Choose the supported provider credential-profile policy before changing custody. An explicit user-wide profile must have a named owner.
Keep existing credentials readable under the approved migration contract. Never copy a token into another instance based on an ambient path.
Use a credential input owned by the Bedrock client where the installed SDK supports it.
If the SDK only supports process-level identity, gate that mode explicitly. Do not swap process credentials around asynchronous calls.

**Edit boundary.** Responses parsing and dispatch in `nanobot/providers/openai_responses/parsing.py` and `nanobot/providers/openai_compat_provider.py`.
Use `nanobot/providers/base.py`, `nanobot/llm_usage/store.py`, and `nanobot/agent/model_runtime.py` for attempt, usage, and route facts.
Keep login intent with `nanobot/webui/settings_services.py`, `nanobot/providers/oauth_credentials.py`, `nanobot/providers/openai_codex_oauth.py`, and `nanobot/providers/xai_oauth.py`.
Use the owning paths in `nanobot/providers/bedrock_provider.py`, `nanobot/webui/mcp_oauth_api.py`, and `nanobot/apps/cli/service.py`.
Use the existing network guard in `nanobot/security/network.py` and its configuration wiring for the scoped policy.

**AC6.2.** EOF after a partial tool or text delta produces failure without a tool effect or silent retry.
Explicit completion and incomplete terminals retain their defined behavior.
A fallback and a transport retry have distinct attempt evidence without duplicate logical usage.
Model selection uses its own capabilities. A cancelled or older login cannot overwrite the current credentials.
Two explicit instance roots retain separate local state and network policy.
A lower-trust catalog source cannot supply executable metadata for a higher-trust entry.

## 3. Resolve skill and feature provenance — R01, R04, R15

Validate a skill candidate before it participates in precedence. Use the same identity and metadata rule as the supported packaging check.
Keep a valid admitted workspace override.
Resolve prompt, Dream, and evaluator overrides through the current resource policy.
Reject an escaping symlink or repo entrypoint before it contributes instructions.
Bind knowledge writes to the selected project's admitted repository association.
For a bundled skill, use the loader's selected source identity for both the summary and the read.
Keep the path readable through the existing read policy. A selected-project collision cannot supply the bundled skill body.
Derive visible guidance from the scoped registry and active runtime consumers.
Keep disabled or unready features out of active discovery. Activation of goals or local triggers requires its own supported runtime and recovery proof.
Correct the bundled cron, installer, mobile-testing, and browser guidance against the live contracts.
Keep video frame extraction local unless the admitted provider and media policy authorize the semantic request.

**Edit boundary.** `nanobot/agent/skills.py`, its prompt and read adapters in `nanobot/agent/context.py` and `nanobot/agent/tools/filesystem.py`, and the affected built-in guidance.
Use the prompt owners in `nanobot/agent/memory.py`, `nanobot/utils/evaluator.py`, and `nanobot/utils/workspace_prompts.py`.
Use the knowledge tool in `nanobot/agent/tools/repositories.py` for project association.
Use `nanobot/skills/video-understanding/scripts/extract_keyframes.py` for the semantic data exit.
Keep user-owned skills, existing path grants, and unrelated feature commands intact.

**AC6.3.** A malformed high-priority skill does not hide a valid candidate. An admitted valid override still wins.
Progressive read returns the selected bundled body when a different project file has the same display path.
A disabled or unready tool is absent from active guidance. A local video workflow makes no ambient-provider call.
An escaping prompt override and a knowledge write to an unrelated repo are refused.

## 4. Bound tool results and finish workbench projection — R13, R14

Set a separate download limit before buffering a `web_fetch` response. Preserve the existing URL and redirect guards.
Bound the tool-result projection before it reaches thread serialization.
If a cancelled VPN start can still acquire a lease, keep that attempt with the VPN owner until it records or closes the lease.
Use turn identity for reasoning and Changes Review. Keep both attached to the originating turn after reload.
Refresh an edited file preview only for the same authorized path and current selection.
Fence a late asset page by its session and cursor. A3-8 supplies global pagination and retained index semantics.

**Edit boundary.** Fetch and projection owners in `nanobot/agent/tools/web.py` and the existing tool result and WebUI event paths.
Keep VPN cancellation and lease cleanup in `nanobot/agent/tools/trend_vpn.py`.
Use `webui/src/components/thread/ThreadShell.tsx`, `webui/src/components/FilePreviewPanel.tsx`, `webui/src/hooks/useSessionGeneratedAssets.ts`, and existing activity rows.

**AC6.4.** Oversize or streaming input stays within the download limit. Large tool results produce a bounded thread payload.
Replay shows reasoning and file changes on the original turn.
A file refresh or asset response from an old selection cannot modify the current view.
Late VPN completion after cancellation closes or retains the lease with its owner.

## Verification and rollback

Use `tests/agent/test_mcp_connection.py`, `tests/agent/test_mcp_reconnect_crash.py`, `tests/tools/test_tool_loader.py`, and `tests/webui/test_mcp_presets_runtime.py`.
Use `tests/providers/test_openai_responses.py`, `tests/providers/test_llm_usage_observer.py`, `tests/providers/test_oauth_credential_coordinator.py`, and `tests/agent/test_model_runtime_resolver.py`.
Use `tests/agent/test_skills_loader.py`, `tests/agent/test_workspace_scope.py`, `tests/tools/test_web_fetch_security.py`, and `tests/webui/test_settings_capabilities.py`.
Use `tests/agent/test_trend_vpn_tools.py` for the optional VPN lease boundary.
Start UI checks with `webui/src/tests/file-preview-panel.test.tsx`, `webui/src/tests/generated-assets-sheet.test.tsx`, and `webui/src/tests/thinking-reasoning-shell.test.tsx`.
Exercise concurrency at the real registry, provider, and login boundaries using fixture transports.
Drain admitted generations before reverting their owner. Preserve the last valid credential and runtime binding.
If the previous release cannot read a changed credential profile, retain the supported export or re-authentication path.
