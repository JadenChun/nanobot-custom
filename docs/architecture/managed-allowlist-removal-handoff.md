# Handoff: remove the managed WebUI tool allowlist

Status: **design stage, work in progress.** This branch holds the first code pass
plus the design review. It is not ready to merge. Read
[`permission-model-review.md`](permission-model-review.md) first; this note
records the discussion, the decisions so far, and what is left.

Branch: `codex/remove-managed-tool-allowlist`, cut from `staging` at `6aaebd722`.
It contains only this work. Parallel chat-UI and subagent-display work, plus the
prompt edits in `tool_contract.md` and `format_hint.md`, were in the same
working tree but are **not** part of this branch.

## 1. What started this

- WebUI session `lolu` could not use the browser to open a Figma link. Codex,
  given the same prompt and model, opened the link in a browser tab and took
  screenshots.
- Root cause: in a managed (Sandbox) WebUI chat, `agent_browser` was hidden by a
  fixed tool-name allowlist (`MANAGED_WEBUI_TOOL_NAMES`). `tool_search` then
  reported "No deferred tools matched", so the model could not find the tool. A
  second check inside `agent_browser` / `agent_device`
  (`ResourceAccessPolicy.allows_unconfined_automation`) refused in every managed
  chat, **including Full access**.

## 2. The user's position (the design direction)

- The managed allowlist is a redundant, blocking layer. Safety belongs to the
  **policy and permission mode** (`RiskyActionPolicy`, profile, allow/ask/deny
  rules, Sandbox vs Full access), as in Codex.
- The workspace is the agent's **default working area**, not a limit on what it
  may do. Compare Codex: sandbox plus approval-on-escalation, not hidden tools.
- The user questioned what "managed folder access" is for; it felt like a
  self-imposed restriction on top of permissions and policy.

## 3. Decisions taken in this pass

1. **No tool-name allowlist.** Every registered tool stays visible and callable in
   a managed WebUI chat, and in delegated children (role profiles still decide
   which tools a role has).
2. **The policy layer decides.** `RiskyActionPolicy` has a new `sandbox_active`
   predicate, which means managed mode without a current Full access reference.
   In a Sandbox turn, a tool that declares `Tool.launches_host_process` asks
   before acting outside Sandbox. This mirrors the existing
   "run outside Sandbox" exec approval:
   - `agent_browser`: read-only navigation runs; state-changing actions ask.
   - `agent_device`: same rule.
   - `desktop_use`: actions in `READ_ONLY_DESKTOP_ACTIONS` run; others ask.
   - `run_cli_app`, `trend_vpn_*`: always ask in Sandbox.
   - The check runs after an explicit `deny` rule and before an explicit
     `allow` rule and the profile, the same ordering as exec Sandbox exceptions.
     An `autonomous` profile does not skip it in Sandbox.
   - Full access keeps the existing profile behaviour.
3. **Browser classifier.**
   - `screenshot` with no output path and no `--screenshot-dir` counts as
     read-navigation. Only shaping flags and one `@ref` are allowed.
     agent-browser writes such images to a temp directory.
   - Any `file:` or `javascript:` argument counts as state-changing.
4. **Kept the dispatch-guard mechanism, without the allowlist.**
   - `ToolDispatchGuard`: a neutral base. Dream's operation fence in
     `memory.py` now subclasses it.
   - `HostProcessAdmissionGuard`: never denies by name. While the grant store is
     not yet managed, it takes the native-process reservation for
     `launches_host_process` tools, so turning Sandbox on cannot race with an
     unconfined launch already in flight. Before, every non-allowlisted tool took
     this reservation. If Sandbox becomes active during admission, the call is
     refused with "retry", so the policy re-assesses it under the new mode.
   - The loop no longer installs a guard on an issued Dream registry. Before, the
     managed guard could replace Dream's guard.
5. **Browser and device working directory.** `allows_unconfined_automation` was
   replaced by `ResourceAccessPolicy.automation_working_dir`. In a restricted
   scope, the process cwd defaults to the workspace, and a caller
   `working_dir` outside the workspace is refused. Unrestricted scope is
   unchanged.
6. `ResourceAccessPolicy.sandbox_active()` was added, and the runner's
   recovery note was made generic ("not available in the current run").

## 4. Files changed

- `kohaku/agent/tools/registry.py`: allowlist removed; new guard classes.
- `kohaku/agent/tools/base.py`: `Tool.launches_host_process` (default False).
- `kohaku/agent/tools/{agent_browser,agent_device,desktop_use,cli_apps,trend_vpn}.py`:
  declare `launches_host_process`. The browser and device tools also drop the
  managed refusal and use the cwd confinement.
- `kohaku/agent/policy.py`: `sandbox_active` field and `_sandbox_crossing_reason`.
- `kohaku/context_repo/policy.py`: `sandbox_active()` and
  `automation_working_dir()`.
- `kohaku/agent/loop.py`: the managed branch no longer filters by name; the
  host admission guard is installed; `sandbox_active` is passed into
  `RiskyActionPolicy`.
- `kohaku/agent/delegation.py`: the four allowlist guard sites now use
  `_host_admission_guard`.
- `kohaku/agent/memory.py`, `kohaku/agent/runner.py`: as described above.
- Tests: `test_managed_exec.py`, `test_managed_mcp_access.py`,
  `test_loop_runner_integration.py`, `test_p6_dream_compact_boundaries.py`,
  `test_tool_loader.py`.

## 5. Checks done so far (informal; the design is not final)

- `ruff check kohaku/`: clean.
- basedpyright, before and after, under the same environment: 962 vs 962
  diagnostics. `scripts/check_type_regressions.py` reported
  `PASS_NO_NEW_DIAGNOSTICS`. This is not the formal P6 evidence; repeat it
  before merging.
- Affected test files against the exact branch content: 223 passed, 23 skipped.
- The full suite was **not** completed. A partial run showed failures that also
  occur on unmodified `staging`, for example
  `test_orchestrator_harness`, `test_migration_bus_admission`,
  `test_history_replay`, `test_sandbox_auto_review` and
  `test_remote_owner_gateway_lifecycle`. Compare the full suite against
  `staging` before merging.

## 6. Remaining work and open questions

- [ ] `tests/agent/test_fork_tool_factories.py::test_foreground_automation_requires_native_access`
      runs on Linux only and still expects the old refusal. Update it:
      default-restricted with an "inside" cwd is now allowed; a sibling cwd is refused.
- [ ] New tests:
      - a managed WebUI turn exposes `agent_browser` through `tool_search`;
      - in Sandbox, navigation and a path-less screenshot run, while a click or
        `file://` asks;
      - Full access is unchanged;
      - the Dream guard still denies.
- [ ] Return the screenshot to the model as an image. Today the browser output
      is only a temp path, and `read_file` on the temp dir may be refused in
      Sandbox. The Figma flow needs this (review doc, P2).
- [ ] Workspace as the default scope, not a limit. In managed mode the chat's
      `project_path` is not writable by default; the writable roots are agent
      home, context-repo roots and grants (`context_repo/policy.py`). This is
      review-doc phase P3.
- [ ] Decide whether host-tool Sandbox crossings should use the auto-reviewer
      (`auto_review.py` handles `exec` only) or keep asking the user.
- [ ] Plugin tools (entry points) are no longer withheld in managed chats. They
      are treated like MCP: opt-in configuration. A plugin that starts host
      processes should declare `launches_host_process`. Confirm this is wanted.
- [ ] Update `docs/sandbox-folders.md` and `.agent/security.md` to describe the
      policy-only model.
- [ ] Separate issue from the same session: the model proxy masks secrets as
      `{{SECRET_…}}` and restores them in model output. The prompt
      (`tool_contract.md`, edited in the parallel work) should tell the model to
      pass such values through unchanged.

