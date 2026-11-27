# Permission model review: sandbox, policy and workspace scope

Status: **implemented (Phases 1–3 complete on `codex/remove-managed-tool-allowlist`, 2026-11-27).** See [`managed-allowlist-removal-handoff.md`](managed-allowlist-removal-handoff.md) for the implementation record and verification summary.

## 1. Why this review

A WebUI chat in Sandbox mode was asked to read a Figma link. The same prompt on the same model in Codex opened the link in a browser, took screenshots, and read the design. Kohaku could not:

- `tool_search("browser screenshot navigate")` and `tool_search("desktop")` returned `"No deferred tools matched"`, although `agent_browser` and `desktop_use` are enabled and `agent-browser` 0.25.3 is installed.
- Cause: in a managed WebUI turn, the registry was narrowed to `MANAGED_WEBUI_TOOL_NAMES` (`kohaku/agent/tools/registry.py`, applied in `kohaku/agent/loop.py`). Browser, desktop, device, CLI-app, VPN, and plugin tools were not on that list, and `ToolSurface.search` drops tools with a `dispatch_error`, so the denial looked like *"tool does not exist"*.
- A second wall inside `agent_browser` and `agent_device` (`ResourceAccessPolicy.allows_unconfined_automation()`) refused in every managed chat, **including Full access**.

Auditing `origin/staging` against Kohaku's own product goals, its A3 decisions (`docs/architecture/a3/decisions.md`), its external coding harness (`kohaku/harness/claude_sdk.py`), and Codex's permission model showed that the incident is a symptom of overlapping layers rather than a one-off bug.

## 2. Current layers on `staging` (~10.8k LOC)

| # | Layer | What it decides | Where |
|---|---|---|---|
| L1 | Permission profile, rules & auto-reviewer (`autonomous` / `balanced` / `strict`, `allow`/`ask`/`deny` rules, `sandbox_exclusions`, 2-stage LLM `auto_review.py`) | Whether a specific tool call may run, be auto-reviewed, or prompt the user | `kohaku/agent/policy.py`, `kohaku/agent/permission_rules.py`, `kohaku/agent/auto_review.py` |
| L2 | OS execution sandbox (`restricted` = Sandbox, `full` = Full access) | Write boundary and private-runtime masking for commands and file tools | `kohaku/security/workspace_access.py`, `kohaku/agent/tools/sandbox.py` (Linux `bwrap` / macOS Seatbelt) |
| L3 | Folder grants (`Settings → Folders`, `folder-grants.json`) | Which folders are writable inside Sandbox | `kohaku/security/folder_grants.py`, `kohaku/context_repo/policy.py` |
| L4 | Managed tool allowlist | Which tools exist at all in a WebUI turn once grants are active | `MANAGED_WEBUI_TOOL_NAMES`, `ManagedToolDispatchGuard`, `allows_unconfined_automation()` |
| L5 | Legacy workspace restriction (`restrict_to_workspace`) | Project folder as a hard read/write limit for file tools and an app-level exec guard | `ToolWorkspace.allowed_root`, `kohaku/agent/tools/filesystem.py`, `webui/src/App.tsx` |
| L6 | Authority references, native-process admission ledger & folder-access recovery | Proof tokens (`WorkspaceFullAccessReference`, `WorkspaceTaskAccessReference`), process-tree retirement, gateway-wide `.folder-grants-active` barrier, remote owner adapter | `kohaku/security/desktop_permissions.py`, `kohaku/security/native_admission.py`, `kohaku/session/workspace_access_recovery.py`, `kohaku/remote/permissions.py` |

### Observed defects on `staging`

1. **L4 conflates filesystem sandboxing with tool existence and rots silently.** Beyond intentionally blocking host-process tools (`agent_browser`, `agent_device`, `desktop_use`, `run_cli_app`, `trend_vpn_*`), the hardcoded 27-name list accidentally hid newer built-in tools (`search_sessions`, `create_goal`, `update_goal`, `my`, `crawl_research`, `social_crawl`), all entry-point plugins (`nanobot.tools`), and delegated reviewer subagents' read-only browser/device tools (`REVIEWER_CAPABILITIES` in `kohaku/agent/delegation.py`).
2. **Gateway-wide `.folder-grants-active` (`grants.is_managed`) leaks across chat modes.** Full access chats still hit L4 narrowing because `grants.is_managed` is gateway-wide rather than per-chat. Now that WebUI starts with `sandbox_default=True`, the "unmanaged → managed" activation latch is vestigial.
3. **Internal divergence on what "Sandbox" means for the active workspace.** Native Kohaku WebUI keeps the chat's selected **Working in** (`project_path`) read-only until a saved grant or `Once` grant is added, while Kohaku's own `kohaku/harness/claude_sdk.py` already treats `self.workspace` as implicitly writable in Sandbox (matching Codex `workspace-write`).
4. **Legacy L5 (`restrict_to_workspace`) still overlaps WebUI Sandbox.** `webui/src/App.tsx` still sets `restrict_to_workspace: access_mode === "restricted"` even though `docs/sandbox-folders.md` states Sandbox allows broad reads under the OS account and enforces writes via `bwrap`/Seatbelt.
5. **Silent discovery denials.** `ToolSurface.search()` drops blocked tools silently instead of reporting why a tool is blocked.
6. **`auto_review.py` is artificially limited to `exec`.** `AgentLoop` already routes every live `RiskyActionPolicy` approval through `review_sandbox_escalation()`, but Stage 1 rejects every non-`exec` tool, and `PendingTurnApproval` sends `{}` instead of tool arguments for non-`exec` calls.

## 3. What we keep vs what we change (Kohaku requirements, not blind Codex copying)

Kohaku is a persistent personal assistant (WebUI/desktop, multi-folder **Working in**, Agent Map, browser/desktop/device automation, channels, Android remote client) rather than a single-repo terminal CLI. We adopt Codex's clean **three-layer separation** while preserving Kohaku's own A3 safety guarantees:

### Deliberately KEPT (do not weaken)
- **OS Sandbox confinement (`bwrap` / macOS Seatbelt)** with fail-loud behavior (never fall back silently to Full access).
- **Private runtime masking** (`~/.kohaku` credentials, sessions, `config.json` hidden inside Sandbox and protected from agent writes).
- **Nested read-only overrides** (an explicit `read_only` folder grant beats a writable parent or workspace).
- **Real process-tree ownership & settlement (`A3-2`)**: volatile Full access ends on restart; returning from Full access → Sandbox fences new work and tracks descendant settlement.
- **Exact single-use effect approvals** bound to real tool preflight (`RiskyActionPolicy`), plus Android remote `OwnerPermissionAdapter` and remote preview confinement.
- **Legacy L5 (`restrict_to_workspace`) for non-WebUI channels** (Telegram, Discord, CLI, API) that do not run the WebUI OS sandbox.

### Accepted design changes (locked in 2026-11-27)
1. **Delete L4 (Managed tool allowlist) completely.** Every enabled tool stays discoverable and callable in WebUI chats (Sandbox and Full access) and delegated children (still bounded by role profiles).
2. **Rule precedence in `RiskyActionPolicy` (L1):**
   1. Explicit `deny` → refuse (and surface transparently in `tool_search`).
   2. Explicit `allow` → **allow, including in Sandbox mode for host-process tools** (lets users opt a host tool in without disabling the filesystem Sandbox). *(Hard non-bypassable invariants on `exec` — nested-agent launch confinement and private-network/SSRF exceptions — remain above `allow`.)*
   3. Sandbox crossing (`_sandbox_crossing_reason`) → ask before a `launches_host_process` tool performs a state-changing host action in Sandbox.
   4. Profile (`autonomous` / `balanced` / `strict`) and explicit `ask` rules.
3. **Generalize `auto_review.py` into a cross-tool 2-stage auto-reviewer:**
   - Not limited to `exec`. Evaluates all live permission escalations so the agent stays as autonomous as possible while preserving deterministic guardrails against genuinely destructive actions.
   - **Stage 1 (deterministic hard-stops → always `ask_user`, never auto-approve):**
     - Explicit user `ask` or `deny` permission rules.
     - Private-network / SSRF exceptions (already bypassed before auto-review in `loop.py`).
     - `exec`: existing `sandbox_exec_risk_reason`, `DEFAULT_EXEC_DENY_PATTERNS`, credential paths (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.kube`, `~/.netrc`, `/etc/shadow`, `/etc/sudoers`), pipe-to-shell (`curl|sh`), and **commands touching Kohaku's own config/runtime paths** (`~/.kohaku`, `~/.nanobot`, `config.json`) or installing into Kohaku's runtime environment.
     - `agent_browser` / `agent_device`: `file:`, `javascript:`, `data:`, `view-source:file:` URLs; `upload` (exfiltrating host files); `eval`/`evaluate`; or file output paths outside workspace/media.
     - `write_file` / `edit_file`: credential, system-auth, or Kohaku config/runtime paths.
     - `desktop_use`, `run_cli_app`, `trend_vpn_*`, `cron`: deterministic checks for sensitive/destructive targets; bounded user-requested actions proceed to Stage 2.
   - **Stage 2 (conservative LLM reviewer):** receives `user_request`, `workspace`, `approval_summary`, `tool_call.name`, and sanitized `tool_call.arguments`. Approves only when the action directly fulfills the user's explicit request, is bounded, and does not exfiltrate secrets, trigger irreversible external side effects/payments/submissions without explicit instruction, or damage host state. Otherwise falls back to the human live approval card.
4. **Trust model for MCP servers and `nanobot.tools` entry-point plugins:**
   - User-configured MCP servers and user-installed `nanobot.tools` plugins are **trusted on configuration/install** (`launches_host_process = False` by default; plugins that want Sandbox prompts can opt in with `launches_host_process = True`).
   - **Invariant against agent self-trust:** the agent must never be able to self-edit `~/.kohaku/config.json` / register MCP servers without explicit manual approval. In Sandbox, `sandbox_private_roots` already blocks access to `config.json`; Stage 1 of `auto_review.py` additionally forbids auto-approving any escalation that touches `~/.kohaku` / `~/.nanobot` / `config.json`. Any future agent tool that proposes adding an MCP server must require explicit manual confirmation before persisting.
5. **User-selected "Working in" (`project_path`) = implicit `workspace-write` root in Sandbox (Phase 2):**
   - Aligns native Kohaku WebUI with `kohaku/harness/claude_sdk.py` and Codex `workspace-write`.
   - Writable roots in Sandbox = **user-selected `project_path` (priority 0)** + agent home (priority 0) + context-repo write roots (priority 0) + private temp/cache + saved RW entries in Settings → Folders (priority 1) + one-shot `Once` task grants (priority 2).
   - **Overrides preserved:** explicit `read_only` grants in Settings → Folders (priority 1) still beat the implicit workspace write (priority 0); private `~/.kohaku` runtime paths, system-control directories (`_overlaps_system_control`), and context-repo protected-file policies remain enforced.
   - **Critical provenance invariant for `select_workspace`:** `SelectWorkspaceTool` (`kohaku/agent/tools/workspace.py`) is model-callable. Model-called `select_workspace` **must NOT widen write authority** to an ungranted directory (it may only focus inside already-authorized roots, or change relative-path focus without conferring write access). Only human-selected **Working in** (WebUI picker / session scope) confers implicit workspace-write authority.
6. **Retire gateway-wide `.folder-grants-active` (`grants.is_managed`) & WebUI L5 (Phases 2–3):**
   - Stop setting `restrict_to_workspace: access_mode === "restricted"` in WebUI (`App.tsx`); keep L5 strictly for non-WebUI channels.
   - Retire the gateway-wide `grants.is_managed` latch and `HostProcessAdmissionGuard` pre-activation barrier while keeping per-chat Full-access retirement and remote owner verification intact.

## 4. Target three-layer model

1. **Execution Sandbox (commands & file tools only, per-chat mode: `Sandbox` | `Full access`):**
   - Broad reads (excluding `~/.kohaku` private runtime and explicit deny rules); writes confined to user-selected `project_path`, agent home, private temp/cache, and Settings → Folders extra RW roots, with nested `read_only` overrides and protected subpaths (`.git` internals, private runtime).
   - Enforced by `bwrap` / Seatbelt; fail loud if unavailable.
2. **Approval Policy (every tool call — single source of truth):**
   - `RiskyActionPolicy` (`deny` → `allow` → Sandbox crossing → profile/`ask`) + exact single-use `effect_id` + live in-turn approval + generalized 2-stage `auto_review.py` + shared `sandbox_exclusions`.
3. **Per-Capability Controls (browser, desktop, device, CLI apps, VPN, MCP, plugins):**
   - Enabled by own config; never hidden by Sandbox mode.
   - Declare `launches_host_process` and action classifiers (`classify_browser_action`, `classify_device_action`, `READ_ONLY_DESKTOP_ACTIONS`).
   - Browser returns screenshots inline as images (or via Kohaku's media dir) so visual flows (e.g. Figma) work end-to-end in Sandbox without weakening filesystem boundaries.
   - `tool_search` reports policy-blocked tools (`blocked by deny rule`) instead of `"No deferred tools matched"`.

## 5. Coupled subsystems to keep in sync

| Subsystem | Files | Coupling to respect |
|---|---|---|
| OS Sandbox mount builder | `kohaku/agent/tools/sandbox.py::_managed_mounts()`, `shell.py`, `seatbelt_profile.py` | Phase 2 must add user-selected `project_path` here in lockstep with `context_repo/policy.py`. |
| External coding harness | `kohaku/harness/backend.py`, `claude_sdk.py`, `direct.py`, `runner.py` | Shares `auto_review.py` (`ExternalToolPolicy` already uses workspace-write). Keep `auto_review` compatible with both native and harness callers. |
| Live approval payload | `kohaku/agent/loop.py` (`PendingTurnApproval`) | Must include sanitized tool arguments for non-`exec` tools so both auto-review and human approval cards show real call details. |
| Delegation & subagents | `kohaku/agent/delegation.py`, `loop.py` | Children already inherit `RiskyActionPolicy(sandbox_active=...)` via `CompositeToolPolicy`; `_ChildAuthorityPolicy` uses `dispatch_error` for Dream/MCP/revocation. |
| Remote / Mobile | `kohaku/remote/permissions.py`, `webui/file_preview.py`, `webui/media_gateway.py` | Phase 3 must preserve `OwnerPermissionAdapter`, `DesktopPermissionOwner` tokens, and remote preview scope checks. |
| WebUI frontend | `webui/src/App.tsx`, `ThreadShell.tsx`, `FolderAccessRequestCard.tsx`, `SecuritySettings.tsx` | Phase 1 needs zero WebUI changes; Phase 2 drops `restrict_to_workspace` binding in `App.tsx` and updates copy; Phase 3 touches permission projection only if needed. |

## 6. Phased refactor plan

| Phase | Scope | Branch / PR | Blast radius |
|---|---|---|---|
| **Phase 1** | Finish removing L4 allowlist; fix `allow` precedence above `_sandbox_crossing_reason`; generalize `auto_review.py` to all tools + fix `PendingTurnApproval` non-`exec` args; tighten `agent_browser` classifier & return screenshots inline as images; surface explicit `deny` rules in `tool_search`; update tests & docs. | Current branch: `codex/remove-managed-tool-allowlist` | **Low** — backend-only, zero WebUI/remote protocol changes. |
| **Phase 2** | Make user-selected **Working in** (`project_path`) an implicit writable root in Sandbox in both `context_repo/policy.py` and `sandbox.py::_managed_mounts()`; enforce `select_workspace` provenance guard (model cannot widen writes); decouple WebUI from legacy `restrict_to_workspace` (`App.tsx`); update `docs/sandbox-folders.md`. | Follow-up PR | **Medium** — touches `context_repo/policy.py`, `sandbox.py`, `workspace_focus.py`, `webui/src/App.tsx`. |
| **Phase 3** | Retire gateway-wide `.folder-grants-active` (`grants.is_managed`) latch and `HostProcessAdmissionGuard` pre-activation barrier; keep per-chat Full-access retirement and remote owner verification; trim obsolete L6 branches. | Follow-up PR | **Medium–High** — touches `security/`, `session/`, `remote/`, `webui/`. |
