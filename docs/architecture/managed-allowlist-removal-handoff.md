# Work Plan & Handoff: Remove Managed Tool Allowlist (Phases 1–3)

Status: **Phases 1, 2, and 3 implemented and verified (2026-11-27), ready for PR review against `staging`.**
Read [`permission-model-review.md`](permission-model-review.md) for the full architectural rationale and locked-in decisions.

Branch: `codex/remove-managed-tool-allowlist` (merged up-to-date with `origin/staging`).

---

## 1. Already completed on this branch (`b84bfa23c`)

- Removed `MANAGED_WEBUI_TOOL_NAMES`, `managed_webui_tool_names()`, and `ManagedToolDispatchGuard` from `kohaku/agent/tools/registry.py`; replaced with neutral `ToolDispatchGuard` (used by Dream in `kohaku/agent/memory.py`) and `HostProcessAdmissionGuard`.
- Removed allowlist name-filtering from `kohaku/agent/loop.py` and the 4 guard sites in `kohaku/agent/delegation.py`.
- Added `Tool.launches_host_process` (`kohaku/agent/tools/base.py`, default `False`) and enabled it on `agent_browser`, `agent_device`, `desktop_use`, `run_cli_app`, and `trend_vpn_*`.
- Added `ResourceAccessPolicy.sandbox_active()` and `ResourceAccessPolicy.automation_working_dir()` (`kohaku/context_repo/policy.py`), replacing `allows_unconfined_automation()` in `agent_browser.py` and `agent_device.py`.
- Added `RiskyActionPolicy.sandbox_active` and `_sandbox_crossing_reason()` (`kohaku/agent/policy.py`), wired from `kohaku/agent/loop.py`.
- Added initial temp-screenshot and `file:`/`javascript:` handling to `classify_browser_action()` (`kohaku/agent/tools/agent_browser.py`).
- Updated runner recovery note (`kohaku/agent/runner.py`) and initial tests (`test_managed_exec.py`, `test_managed_mcp_access.py`, `test_loop_runner_integration.py`, `test_p6_dream_compact_boundaries.py`, `test_tool_loader.py`).

---

## 2. Phase 1 Implementation Checklist (Ordered Tasks)

### Task 1.1 — Fix `explicit == "allow"` precedence in `RiskyActionPolicy`
- **File:** `kohaku/agent/policy.py` (`_risky_reason`, lines ~677–684)
- **Change:** Move `if explicit == "allow": return None` **above** `if (crossing := self._sandbox_crossing_reason(tool_call)) is not None: return crossing`.
- **Expected order for non-`exec` tools:**
  1. `explicit == "deny"` → `self._refuse_denied_rule(tool_call)`
  2. `explicit == "allow"` → `None` (allows the host tool even in Sandbox)
  3. `self._sandbox_crossing_reason(tool_call)` → ask in Sandbox for state-changing host-process actions
  4. Profile (`autonomous` / `balanced` / `strict`) & `explicit == "ask"`
- **Keep untouched:** `exec` hard-invariants (`launch_path` confinement, `private_network_exception`, and `sandbox_exception` ordering).

### Task 1.2 — Generalize `auto_review.py` to all permission escalations + fix `PendingTurnApproval` payload
- **Files:**
  - `kohaku/agent/auto_review.py`
  - `kohaku/agent/loop.py` (`_resolve_live_approval` / `PendingTurnApproval` construction around lines ~3987–4036)
  - Check compatibility with `kohaku/harness/direct.py` and `kohaku/harness/runner.py` (harness callers pass `ExternalToolCall` converted to `ToolCallRequest`).
- **Changes in `kohaku/agent/auto_review.py`:**
  1. **Remove the `exec`-only gate** (`if call.name != "exec": ...`) in `check_auto_review_eligibility()`.
  2. **Keep & expand Stage 1 deterministic hard-stops** (any match returns an ineligibility reason → forces `ask_user`, never auto-approved):
     - Explicit `ask` or `deny` rule from `match_explicit_rule(...)` (for all tools, passing `command` for `exec` and `target_path` for `write_file`/`edit_file`).
     - **`exec`**: keep all existing checks (`sandbox_exec_risk_reason`, `_COMPILED_EXEC_DENY_PATTERNS`, `_CREDENTIAL_PATH_PATTERN`, `_REMOTE_PIPE_TO_SHELL_PATTERN`), **and add** a Kohaku self-configuration guard matching `~/.kohaku`, `~/.nanobot`, or Kohaku `config.json` paths so the agent can never auto-approve editing its own config/MCP servers.
     - **`write_file` / `edit_file`**: reject credential/system paths (`_CREDENTIAL_PATH_PATTERN`) and Kohaku config/runtime paths (`~/.kohaku`, `~/.nanobot`, `config.json`).
     - **`agent_browser` / `agent_device`**: reject dangerous URL schemes (`file:`, `javascript:`, `data:`, `view-source:file:`), file uploads (`upload`), arbitrary script execution (`eval`, `evaluate`), or explicit file-output options outside workspace/media. Normal user-requested UI interactions (click, fill, type, press, select, scroll, open, etc.) are eligible for Stage 2.
     - **`desktop_use` / `run_cli_app` / `trend_vpn_*` / `cron` / other tools**: reject empty/malformed calls or arguments referencing credential/Kohaku-config paths; otherwise eligible for Stage 2.
  3. **Generalize `_build_review_user_message()` and `_SYSTEM_PROMPT`:**
     - Serialize `tool_call.name` plus sanitized, size-bounded `tool_call.arguments` (keeping `command` + `working_dir` clearly formatted for `exec`, and passing structured args for all other tools).
     - Update `_SYSTEM_PROMPT` from "proposed local Sandbox escalation command" to a general conservative permission reviewer: approve ONLY when the tool call directly fulfills the user's explicit request, stays bounded, and does not exfiltrate secrets, modify Kohaku config/permissions, perform irreversible external side effects/payments/submissions without explicit instruction, or damage host state. Otherwise return `ask_user`.
- **Changes in `kohaku/agent/loop.py`:**
  - Around line 3991, `PendingTurnApproval` currently passes `json.dumps({"command": command} if isinstance(command, str) else {})`. Update it to pass sanitized, bounded `call.arguments` for non-`exec` tools (keeping `{"command": command}` shape for `exec` so existing UI/tests stay compatible).

### Task 1.3 — Tighten `agent_browser` classifier & return screenshots inline as images
- **File:** `kohaku/agent/tools/agent_browser.py`
- **Classifier updates:**
  1. Expand `_NON_NAVIGATION_URL_SCHEMES` to include `("file:", "javascript:", "data:", "view-source:")` so `data:` HTML/JS and `view-source:file://...` are treated as `state_changing`.
  2. For `open` / `goto` with a bare local filesystem path (e.g., `/etc/passwd`, `~/...`, `./...` without `http://` or `https://`), classify as `state_changing` rather than `read_navigation`.
  3. In `_is_temp_screenshot()`, allow leading global shaping flags (e.g. `--json screenshot` or `--full` before/after) cleanly, while still rejecting any output path or `--screenshot-dir`.
- **Inline screenshot return (unblocks the Figma flow in Sandbox):**
  - Follow the existing multimodal image pattern in `kohaku/agent/tools/desktop_use.py` (lines ~558–563: base64 `image_url` data URI, respecting size caps / downscaling or copying into Kohaku's media dir).
  - When a `screenshot` command succeeds (`returncode == 0`) and `agent-browser` outputs a PNG/JPEG file path (in stdout/JSON), read that generated image file directly inside `AgentBrowserTool.execute()` (with size cap / validation that it is a real image file) and return a `ToolOutcome` / multimodal content containing both the JSON text summary and the inline `image_url` block so the model can see the screenshot directly without calling `read_file` on a temp directory outside the workspace.
  - Ensure `agent_browser` still returns plain JSON cleanly if no image file was produced or if the file is missing/unreadable.

### Task 1.4 — Surface explicit `deny` rules in `tool_search` (transparent denials)
- **Files:** `kohaku/context/tool_surface.py`, `kohaku/agent/loop.py` (where `ToolSurface` is constructed), and `tool_search` tool implementation.
- **Change:**
  - When a user has an explicit `deny` rule for a tool in `ToolPermissionsConfig`, `tool_search` should NOT silently pretend `"No deferred tools matched"`.
  - Either return the match annotated as blocked by explicit deny rule (without activating its schema for calling) or append a clear note in `tool_search` results such as: `Blocked by explicit permission deny rule: <tool_name>` so the model can accurately explain the policy block to the user.
  - Keep `dispatch_error` filtering for stale Dream/MCP leases intact.

### Task 1.5 — Check legacy `kohaku/agent/tools/image.py`
- Verify whether legacy `kohaku/agent/tools/image.py` (which has a `codex` CLI subprocess fallback at line 248) can still be registered in WebUI. If reachable, set `launches_host_process = True` when the subprocess backend is configured (or unconditionally on that legacy class).

### Task 1.6 — Tests
- [ ] Update `tests/agent/test_fork_tool_factories.py::test_foreground_automation_requires_native_access` (Linux-only): default-restricted with an inside-workspace cwd is now allowed; a sibling/outside cwd is refused.
- [ ] Add new focused tests covering:
  1. Managed WebUI turn exposes `agent_browser` (and other previously hidden tools like `search_sessions`, `my`, `create_goal`) through `tool_search`.
  2. Sandbox host-tool policy:
     - `agent_browser` read-navigation and path-less `screenshot` run without asking;
     - `click`, `file://`, `javascript:`, `data:`, `view-source:file:`, bare local path, and screenshot with output path ask;
     - explicit `allow` rule allows state-changing host tool in Sandbox;
     - explicit `deny` rule refuses and is surfaced transparently by `tool_search`;
     - Full access keeps existing profile behavior.
  3. Generalized `auto_review.py`:
     - non-`exec` calls (e.g. `agent_browser click`, `desktop_use`, `write_file`) reach Stage 2 LLM review when clean;
     - Stage 1 hard-stops force `ask_user` for `agent_browser upload`/`eval`/`file://`, `exec`/`write_file` touching `~/.kohaku` / `~/.nanobot` / `config.json`, credential paths, and explicit `ask`/`deny` rules.
  4. `agent_browser` inline screenshot return: successful `screenshot` returns image content alongside JSON metadata without requiring `read_file` on `/tmp`.
  5. Dream `ToolDispatchGuard` still denies stale Dream operations.
  6. Delegated reviewer subagent (`REVIEWER_CAPABILITIES`) retains read-only `agent_browser` in a managed WebUI turn.

### Task 1.7 — Documentation updates
- Update `docs/sandbox-folders.md` to document the policy-only model (host-process tools stay available and are governed by `RiskyActionPolicy` + generalized auto-reviewer; MCP & installed plugins are trusted on config/install while `config.json` stays protected).
- Update `.agent/security.md` to reflect that host-process tools use `Tool.launches_host_process` + `RiskyActionPolicy` + `automation_working_dir` instead of a static allowlist.

---

## 3. Validation Requirements Before Merge (Per `AGENTS.md`)

1. **Lint:**
   ```bash
   ruff check kohaku/ tests/
   ```
2. **Affected tests (and comparison against unmodified `staging` for full suite):**
   ```bash
   pytest tests/agent/test_managed_exec.py \
          tests/agent/test_managed_mcp_access.py \
          tests/agent/test_loop_runner_integration.py \
          tests/agent/test_p6_dream_compact_boundaries.py \
          tests/agent/test_fork_tool_factories.py \
          tests/agent/test_sandbox_auto_review.py \
          tests/tools/test_tool_loader.py -v
   ```
   *(Plus any new test files added for browser screenshots, generalized auto-review, and `tool_search` deny reporting.)*
3. **Strict type checking (P6 regression protocol — `docs/migration/p6-error-attribution.md`):**
   - Capture a frozen before/after full-source `basedpyright` report under the exact same dependency environment.
   - Run `scripts/check_type_regressions.py` and verify `PASS_NO_NEW_DIAGNOSTICS` (zero newly introduced diagnostic signatures across all files).
   - Retain source hashes and command/environment evidence in the handoff/PR note.

---

## 4. Follow-Up Phases (Out of Scope for This PR — Do NOT Mix Into Phase 1)

Recorded here so Phase 1 stays small, clean, and zero-protocol-churn:

- **Phase 2 (Next PR — Workspace-Write Alignment):**
  - Make human-selected **Working in** (`project_path`) an implicit writable root (priority `0`) in Sandbox in both `kohaku/context_repo/policy.py::_validate_managed_path` and `kohaku/agent/tools/sandbox.py::_managed_mounts` (matching `kohaku/harness/claude_sdk.py` and Codex `workspace-write`).
  - Preserve explicit `read_only` folder grants (priority `1`) beating workspace (`0`), private runtime masking, and `_overlaps_system_control`.
  - **Enforce `SelectWorkspaceTool` provenance invariant:** model-called `select_workspace` (`kohaku/session/workspace_focus.py::select_from_tool`) must NOT confer or widen write authority to an ungranted directory.
  - Decouple WebUI from legacy `restrict_to_workspace` (`webui/src/App.tsx:1722`), keeping L5 only for non-OS-sandboxed channels (Telegram, Discord, CLI, API).
  - Update `docs/sandbox-folders.md` and WebUI copy.
- **Phase 3 (Cleanup PR — Retire `.folder-grants-active`):**
  - Retire the gateway-wide `grants.is_managed` / `.folder-grants-active` latch and `HostProcessAdmissionGuard` pre-activation barrier (since WebUI runs `sandbox_default=True`).
  - Preserve volatile Full access, Full → Sandbox `A3-2` process-tree retirement, `DesktopPermissionOwner` / `OwnerPermissionAdapter` for Android remote, and remote preview boundaries.
- **Separate parallel issue (not part of this branch):**
  - Model proxy secret masking (`{{SECRET_…}}`) prompt guidance in `tool_contract.md`.
