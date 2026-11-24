# Claude Code Harness Implementation Work Plan

**Branch** — `codex/coding-harness-session`, based on `staging` at `1bdf7920`.
**Design** — [Claude Code harness design](./coding-harness-session-design.md).
**Repository rules** — [AGENTS.md](../../AGENTS.md), [design](../../.agent/design.md),
[security](../../.agent/security.md), and [gotchas](../../.agent/gotchas.md).

The first release includes direct Claude Code sessions and Claude-backed subagents in Kohaku chats.
Both use the same SDK adapter and model bridge.

## Completed Prerequisites

- Stage 1A (`8d936351`) registered `find_files` and `grep`.
  It also added safe parallel reads and recovery from ordinary tool errors.
- Stage 1B (`4d011735`) registered `list_sessions`, `search_sessions`, and `read_session`.
  `ListSessionsTool` already reads the harness and locked workspace from `harness_binding`.
- Stage 1B (`1bdf7920`) added sandbox exclusions, local binding, and Unix socket settings to `ExecToolConfig`.
  `permission_rules.py` provides the command risk and save-eligibility checks.
  The gateway persists eligible exclusions from `approve_and_save`.
  The existing approval card renders **Allow once**, **Allow & save**, and **Deny**.

## Verified SDK Feasibility Findings (`2026-11-24` Live Spike)

Verified in an isolated Python 3.14 Linux scratch venv (repo `.venv` untouched) with **`claude-agent-sdk 0.2.188`** (bundled CLI **`2.1.331`**), **`bwrap 0.12.0`**, and **`socat 1.8.1.3`**:

- **Packaging & runtime:** Self-contained PyPI wheel bundles `_bundled/claude` (`2.1.331`) with no separate Node.js/npm required. `pyproject.toml` optional extra should target `claude-agent-sdk>=0.2.188` (lazy-imported only in `kohaku/harness/claude_sdk.py`).
- **Loopback wire traffic (`127.0.0.1`):**
  - Launch subprocess with `DISABLE_TELEMETRY=1`, `DISABLE_ERROR_REPORTING=1`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, `setting_sources=[]`, and `strict_mcp_config=True` — eliminates non-essential traffic and prevents ambient `~/.claude` or MCP leaks.
  - CLI posts to `POST /v1/messages?beta=true` (accept query params on `/v1/messages`) with custom model strings passed verbatim (harmless `[claude-code:unrecognized_model]` stderr line should not be treated as a failure).
  - Real CLI `2.1.331` sends Anthropic-specific top-level fields (`output_config: {'effort': ...}`, `thinking: {'type': 'adaptive', 'display': 'updates'}`, `context_management: {'edits': [...]}`), `anthropic-beta` headers, and tool key `defer_loading`. Phase 1's bridge must tolerate/strip these cleanly for non-Anthropic providers (and check tool-search/deferred-loading compatibility during the non-Claude live smoke test).
- **Pre-tool security & live approvals:**
  - `HookMatcher(matcher=None, hooks=[...], on_failure="block")` is supported and fires `PreToolUse` before `autoAllowBashIfSandboxed` runs.
  - `PreToolUse` returning `permissionDecision="deny"` blocked out-of-scope `Write` reliably; returning `permissionDecision="ask"` escalated sandboxed `Bash` straight into async `can_use_tool` (`PermissionResultAllow` / `PermissionResultDeny`).
- **Linux OS sandbox:**
  - `SandboxSettings` TypedDict supports `enabled`, `autoAllowBashIfSandboxed`, `excludedCommands`, `allowUnsandboxedCommands`, `network` (`allowUnixSockets`, `allowLocalBinding` [macOS-only], `allowedDomains`, `deniedDomains`), `ignoreViolations`, `enableWeakerNestedSandbox`. It does **not** include a `failIfUnavailable` key — Kohaku must pre-check `managed_bwrap_available()` + `shutil.which("socat")` on Linux before launch to fail closed.
  - Live `bwrap` check passed: workspace write succeeded; out-of-workspace shell write (`touch` outside `cwd`) was blocked by the OS sandbox.
- **Resume & fork:** `resume=session_id` + `fork_session=True` preserved full prior turn/tool history and returned a distinct forked `session_id`. `ClaudeSDKClient` also exposes `interrupt()`, `set_model()`, `set_permission_mode()`, `get_context_usage()`, and `effort` option.

## Execution and Verification

1. Make one verified commit per phase on `codex/coding-harness-session`.
   Keep the phase commits local for review.
2. Before Phase 1 source edits, capture a frozen full-source `basedpyright --outputjson` report.
   Keep the source hashes, checker command, dependency environment, and exit status.
   For each phase, capture the after report with the same command and environment.
   Use `scripts/check_type_regressions.py` to require `PASS_NO_NEW_DIAGNOSTICS` and `introduced: 0` across all files.
   Follow the [migration validation procedure](../migration/p6-error-attribution.md).
3. Run `ruff check kohaku/`, the affected tests, and `git diff --check` for each phase.
   Use the [repository lint guidance](../../.agent/gotchas.md).
4. If a requirement needs new behavior cases, follow the requirements-only test authoring procedure in [AGENTS.md](../../AGENTS.md).
5. Keep third-party SDK types and imports at the adapter boundary.
   Verify the base installation with the optional SDK absent.

## Phase 1 — Loopback Anthropic Messages Bridge

**Goal** — Route Claude Code model requests through the selected Kohaku preset and its usage accounting.

**Code** — `kohaku/harness/anthropic_bridge.py` and focused bridge tests.

- Bind the bridge to `127.0.0.1:0`. Use a random token scoped to the owning execution context.
- Check the bearer token or `x-api-key` with constant-time comparison. Return `401` for invalid credentials.
- Support `POST /v1/messages` with SSE and non-streaming JSON. Support `POST /v1/messages/count_tokens`.
- Translate system prompts, messages, images, tools, tool choice, generation limits, and supported thinking fields.
- Route SDK compaction and model aliases through the same bound `LLMRuntime`.
- Preserve supported Anthropic cache and thinking fields on the Anthropic path.
  Translate provider-specific fields and errors at the bridge boundary.
- Map text, thinking, tool input, usage, and stop reasons to the Anthropic response format.
- Record usage through `bind_llm_usage_source`. Propagate disconnects and cancellation to the provider.

### Phase 1 Acceptance

- Scripted provider checks cover streaming and non-streaming responses, tool round trips, authentication, errors, and cancellation.
- Use an isolated SDK smoke environment so the type checker environment stays fixed.
  Run the real SDK through the bridge on a Claude preset and a representative non-Claude preset.
  Check tool calls, streaming, compaction, and usage attribution.
  Record the SDK and bundled CLI versions, model presets, and observed results.
  Present only presets that pass the relevant compatibility checks as supported.
  [Claude Code gateway support](https://code.claude.com/docs/en/llm-gateway)

## Phase 2 — Shared SDK Adapter

**Goal** — Provide a common execution path for direct sessions and parent-owned Claude subagents.

**Code** — Optional `coding-sdk` extra in `pyproject.toml`.
Use `kohaku/harness/backend.py`, `runner.py`, and `claude_sdk.py` for the typed boundary, runner, and SDK client.

- Define `SessionHarnessBinding` and a typed turn request and outcome.
  Pass the execution owner, workspace, model, and admitted capability scope to the adapter.
  Use the existing writer and lifecycle owner for the direct session or delegated child.
- Load the SDK on admission. Start it with controlled settings and a restricted process environment.
  Pass the loopback URL and token to the subprocess. Keep provider credentials in Kohaku.
- Start or resume the SDK by its stored ID. Store the ID as soon as the SDK reports it.
  Map live text and reasoning through `AgentHook`.
  Persist text, tool intents, results, edits, and settlement through `SourceTurnWriter`.
  Normalize SDK tool payloads for the existing file edit and activity views.
- Reuse Stage 1B settings and permission checks.
  Apply scope and private-path checks through SDK tool restrictions and pre-tool checks.
  Use `can_use_tool` to wait for a live approval when the operation needs one.
  Store the pending decision in `PendingExternalApproval` with its owner, run, generation, and exact effect.
  Resolve it once through the existing approval control path after a fresh authority check.
  Persist a saved exclusion only for an eligible executable selected by the gateway.
- Enforce file scope strictly on SDK file tools (`Write`, `Edit`, `NotebookEdit`) via `PreToolUse` hooks.
  For direct sessions and workspace-scoped tasks, use Claude Code's native `Bash` under its OS sandbox (`bwrap` / Seatbelt).
  For narrow `write_scope` delegated workers, keep file edits strictly inside `write_scope` via `PreToolUse` hooks while allowing sandboxed `Bash` within the workspace root for test execution (still gated by `permission_rules.py` and `can_use_tool`).
  Keep routine tests callable by Claude Code. Use the existing 3-choice exception card for eligible unsandboxed requests.
- Support both normal `resume=external_session_id` (same session/child follow-up) and `resume=prior_external_session_id` with `fork_session=True` (carrying investigation context into a fresh implementation child admission).
- Bind Stop, idle release, and shutdown to the owned SDK process.
  Cancel pending approvals when the owner stops work.
  Fence old permissions on revocation and reapply current policy before resume.

### Phase 2 Acceptance

- Focused adapter checks cover streaming, normalized edits, scoped reads and writes, the three approval actions, Stop, and resume.
  Check approved exclusions and refusal for commands that are ineligible to save.
- Prove shell confinement with the real SDK.
  An authorized test completes and can write to its approved output directory.
  A command that attempts a write beyond the assignment scope is blocked.
  Check the Sandbox and Full access configurations. Check the narrower child scope under parent Full access.
- Confirm that Stop cancels the process and pending approvals.
  Confirm that the SDK absence produces an actionable capability error.

## Phase 3 — Direct Claude Code Sessions (Gateway + WebUI)

**Goal** — Let the user select Claude Code as the harness for a new persisted chat in WebUI, while keeping wire payloads backward-compatible for TUI and Android until Phase 5.

- Use the normal new-chat flow. Place a **Harness** dropdown above the composer input container.
  Default to **Kohaku**. Offer **Claude Code** according to backend availability.
  Keep the **Model** control separate.
- When **Temporary chat** (`onCreateTemporaryChat`) is active, disable/hide **Claude Code** in the Harness dropdown and reject `harness="claude_sdk"` for temporary-chat creation at the gateway.
- Pass the draft selection through `ThreadShell`, `App.onCreateChat`, `useSessions`, and the client's `new_chat` request.
  Apply the selection through all normal create paths.
  Keep the draft if session creation fails.
- Validate the choice in `kohaku/webui/inbound_commands.py` before the gateway creates the chat.
  Save `harness_binding` with the session. Map a missing harness field to the native backend.
  Publish the saved harness in session attach, controls, and list payloads (additive fields only so TUI and Android remain compatible).
- Show the **Claude** badge in the WebUI direct chat header and sidebar.
  Use the saved harness binding to render it after reload.
- Dispatch direct Claude Code turns at the existing `AgentLoop` runner boundary (`kohaku/agent/loop.py`).
  Bind the working folder on the first turn. Keep **Working in** available before that turn.
  Reject a harness change on a created chat and a folder change on a locked Claude Code chat.
- Route direct approvals and Stop through the owning chat.
  Exclude the direct Claude Code chat from Dream memory. Use the existing read-only tools to inspect it.
  Reconcile unfinished work through the session recovery owner on gateway restart.

### Phase 3 Acceptance

- Choose each harness from above the new-chat composer. Confirm the first turn uses the selected harness.
  Open an established chat and confirm that its saved harness and Model are shown independently.
- Confirm that the direct **Claude** badge appears in the header and sidebar after reconnect.
  Confirm that changing a selection or receiving a create error preserves the draft.
- Verify workspace selection before Turn 1, its lock after Turn 1, and SDK resume on a follow-up.
- Verify live approval, Stop, interrupted or uncertain restart recovery, and authorized historical lookup.
  Confirm that an interrupted run requires a fresh request before another command runs.
- Run the affected session and WebUI tests.
  In `webui/`, run `bun run lint`, `bun run build`, and `bun run test`.

## Phase 4 — Orthogonal Claude Code Delegation in the Original Chat

**Goal** — Let Kohaku compose `harness="claude_sdk"` independently with any existing delegation role (`explore`, `plan_task`, `delegate_task`, `review_work`) while preserving each role's capability contract.

- Route Claude-backed children through `ChildBackend` **and** `AgentLoop.execute_child_turn` into the **same** `ExternalHarnessRunner` seam at `kohaku/agent/loop.py` used by direct sessions (one `SourceTurnWriter`, one `AgentHook`, one `PendingExternalApproval` bridge, and one Stop path).
- Add an optional `harness: "native" | "claude_sdk"` parameter (default `"native"`) to `explore`, `plan_task`, `delegate_task`, and `review_work`.
  Strictly preserve each role's original capability boundary: `explore`, `plan_task`, and `review_work` remain read-only; `delegate_task` enforces `write_scope` on file tools and enables workspace-sandboxed test execution when backed by Claude Code.
- Add concise tool guidance to prefer `harness="claude_sdk"` (when available) for codebase investigation, technical planning, implementation, and testing.
- Support carrying Claude investigation context into implementation **without widening an existing read-only child**:
  Add an optional `continue_from_child_id` parameter on `delegate_task` (validated as owned by the same parent chat and backed by Claude Code).
  Kohaku re-checks current user, chat, and workspace authority, creates a **fresh implementation child admission**, and launches the SDK with `resume=<prior_external_session_id>` and `fork_session=True`.
  Regular `followup_child` continues the same child under its original immutable scope.
- Pass the relevant PRD references, findings, and agreed plan to the child.
  Include the authorized workspace, write scope, and acceptance checks.
- Project the child's harness identity through the existing child activity and history payloads.
  Update the live and retained child projections and WebUI client types.
  Show a **Claude** badge on the WebUI activity item and in the child detail header.
  Use the normal child activity, approval, follow-up, and Stop controls in the original chat.
- Return the result with changed files, commands, test outcomes, and remaining failures.
  Use the same owner when the main chat receives later work status or completion.
  Reconcile interruption through the existing child recovery path.

### Phase 4 Acceptance

- Start in Kohaku with a PRD and call `explore(harness="claude_sdk")` / `plan_task(harness="claude_sdk")`.
  Claude returns findings under read-only scope (cannot edit files). Kohaku discusses the plan in the original chat.
  When the user asks for implementation, Kohaku calls `delegate_task(harness="claude_sdk", write_scope=..., continue_from_child_id=...)`: verify a fresh child admission is created under current authority, Claude retains prior investigation context via `fork_session=True`, edits within `write_scope`, and runs tests.
  Verify the original read-only child record remains immutable and read-only.
- Give Kohaku an existing plan and ask it to implement it via `delegate_task(harness="claude_sdk", ...)`.
  Verify that Claude can inspect, edit within `write_scope`, and test in one assignment.
- Confirm that the parent chat shows the **Claude** badge on the correct child through reload and reconnect.
  Verify badge behavior for both native and Claude Code children when their models differ from their harnesses.
- Verify approvals and Stop in the parent chat, including when a child is running a test.
  Verify that interruption retains evidence and that continuation checks current authority.
- Run affected delegation, sandbox, recovery, and WebUI tests (`bun run lint`, `bun run build`, `bun run test`).

## Phase 5 — TUI & Android Client UI Parity

**Goal** — Surface the exact same **Harness** selector (new chat) and **Claude** badge (direct sessions and delegated children) in the TUI and Android remote clients using the gateway payloads established in Phases 3–4.

- **TUI:** Add the **Harness** choice on new persisted chats (disabled for temporary chats), render the **Claude** badge on direct Claude Code sessions and Claude-backed child activity/detail views, and preserve workspace-lock feedback after Turn 1.
- **Android (`mobile/`):** Parse `harness_binding` and child harness identity from existing session/child payloads, add the **Harness** selector on new persisted chats (excluding temporary chats), render the **Claude** badge in the session list, chat header, child activity row, and child detail header, and display the locked workspace state after Turn 1.

### Phase 5 Acceptance

- Verify TUI and Android both render the **Claude** badge on direct sessions and delegated children across reconnect/reload, independent of the selected model.
- Verify creating a Claude Code chat and observing workspace lock + live approvals + Stop from TUI and Android.
- Run applicable TUI and Android build/unit checks.

## Release Review

- [ ] Direct harness selection (excluding temporary chats) and orthogonal `harness="claude_sdk"` role delegation both work end to end.
- [ ] Direct sessions and delegated children both execute through the shared `AgentLoop` -> `ExternalHarnessRunner` seam.
- [ ] A direct chat and a Claude-backed child show the **Claude** badge from their saved harness identity across WebUI, TUI, and Android.
- [ ] Read-only delegated children remain immutable; `continue_from_child_id` creates a fresh implementation admission via `fork_session=True`.
- [ ] Claude Code runs routine tests within the workspace sandbox while enforcing `write_scope` strictly on file tools.
- [ ] Supported model presets have live SDK evidence, and provider credentials stay in Kohaku.
- [ ] Direct and child history preserve accurate ownership and restart outcomes.
- [ ] The base install and full required local test suite pass with the optional SDK absent.
- [ ] The branch passes the applicable lint, tests, and full-source type regression gates.
