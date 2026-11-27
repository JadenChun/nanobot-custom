# Permission model review: sandbox, policy and workspace scope

Status: proposal, not implemented. Date: 2026-11-27.

## 1. Why this review

A WebUI chat in Sandbox mode was asked to read a Figma link. The same prompt
on the same model in Codex opened the link in a browser, took screenshots and
read the design. Kohaku could not:

- `tool_search("browser screenshot navigate")` and `tool_search("desktop")`
  returned "No deferred tools matched", although `agent_browser` and
  `desktop_use` are enabled and `agent-browser` 0.25.3 is installed.
- Cause: in a managed WebUI turn the registry is narrowed to
  `MANAGED_WEBUI_TOOL_NAMES` (`kohaku/agent/tools/registry.py:35`, applied at
  `kohaku/agent/loop.py:7455`). Browser, desktop and device tools are not on
  that list, and `ToolSurface.search` drops tools with a dispatch error, so
  the denial looks like "tool does not exist".

The incident is a symptom. The underlying question is whether the layering is
right. We compare it with Codex's model below.

## 2. Current layers (as implemented)

| # | Layer | What it decides | Where |
|---|---|---|---|
| L1 | Permission profile and rules (`autonomous` / `balanced` / `strict`, allow/ask/deny rules) | Whether a specific tool call may run or must be approved | `kohaku/agent/policy.py`, `kohaku/agent/permission_rules.py`, `ToolPermissionsConfig` |
| L2 | Sandbox mode per chat (`restricted` = Sandbox, `full` = Full access) | Write boundary for commands and file tools | `kohaku/security/workspace_access.py`, `kohaku/agent/tools/sandbox.py` (Seatbelt / bwrap) |
| L3 | Folder grants (Settings → Folders, `folder-grants.json`) | Which folders are writable inside Sandbox | `kohaku/security/folder_grants.py`, `kohaku/context_repo/policy.py:316` |
| L4 | Managed tool allowlist | Which tools exist at all in a WebUI turn once grants are active | `MANAGED_WEBUI_TOOL_NAMES`, `ManagedToolDispatchGuard` |
| L5 | Legacy workspace restriction (`restrict_to_workspace`) | Project folder as a hard limit for file tools and an app-level exec guard | `ToolWorkspace.allowed_root`, `kohaku/agent/tools/filesystem.py:272` |
| L6 | Full access / task references, native-process admission ledger | Proof objects that authorize unconfined work and barriers for grant activation | `kohaku/security/desktop_permissions.py` (1.7k lines), `kohaku/security/native_admission.py` (1.2k lines) |

L1 already handles the browser well (`kohaku/agent/policy.py:678`): read-only
navigation runs, state-changing actions (click, type, upload, download,
eval, screenshot) ask, explicit rules override, `autonomous` skips prompts.
`desktop_use` and `agent_device` have matching rules. L4 removes these
tools before L1 is consulted.

Observed behaviour on a real install (`~/.nanobot`):

- `.folder-grants-active` exists, so `grants.is_managed` is true for the
  whole gateway. It is not a per-chat setting.
- The L4 condition is `grants.is_managed or local_private_workspace_authority`
  with no Full access exemption, so Full access chats appear to be narrowed
  too. (From reading the code; needs a regression test to confirm.)
- Subagents with a read-only browser (`REVIEWER_CAPABILITIES`,
  `kohaku/agent/delegation.py:1292`) get the same guard
  (`kohaku/agent/delegation.py:1316`), so they lose the browser too.
- In managed mode the writable set is the agent home, context repo write roots,
  explicit grants and task grants (`kohaku/context_repo/policy.py:357-376`).
  The chat's selected project folder is not writable by default; writing to it
  needs a saved grant or a `request_folder_access` turn stop.

## 3. Codex reference model

Source: developers.openai.com/codex/concepts/sandboxing,
/codex/permissions, /codex/browser, /codex/computer-use.

1. **Two controls, kept separate.** "The sandbox defines technical boundaries.
   The approval policy decides when the agent must stop and ask before crossing
   them." Modes: `read-only`, `workspace-write`, `danger-full-access`.
   Approvals: `on-request`, `never`, with an optional auto-reviewer.
2. **The sandbox covers spawned commands (plus file edits) only.** "Permission
   profiles govern sandboxed commands that run on your machine. Connectors, MCP
   servers, browser or computer-use surfaces ... use their own controls."
   "To limit these surfaces, configure each capability directly."
3. **The workspace is the default working area, not a fence on reading.**
   `:workspace` "allows writes inside the active workspace roots and system temp
   directories"; reads are broad. The session's workspace root is writable
   without a separate grant. `writable_roots` / `--add-dir` add more roots.
   Protected paths (`.git`, `.agents`, `.codex`, `.aws`) stay read-only, and
   deny rules (for example `**/*.env`) win over broader access.
4. **Per-capability controls for non-sandboxable surfaces.**
   - Browser: site permissions for browsing, downloads and uploads, with
     per-site exceptions and confirmation for consequential actions.
   - Computer Use: an always-allowed app list plus OS permissions.
   - MCP: server config and allowlists.
   - Choosing a sandbox mode never removes these tools.
5. **Fail loud.** If the platform sandbox cannot enforce a policy, the command
   is refused with a reason. Tools are not silently removed.

## 4. Gaps

| ID | Gap | Effect |
|---|---|---|
| G1 | L4 ties non-shell capabilities to the filesystem sandbox. Codex governs them per capability. | The browser, desktop and device tools are lost in every managed WebUI chat, although L1 already has rules for them. |
| G2 | Managed activation is gateway-wide and sticky (`.folder-grants-active`), not part of the chat's mode. | Full access chats and subagents inherit Sandbox restrictions. |
| G3 | Two workspace models coexist: legacy L5 (project folder = hard limit) and L3 grants. `access_mode = restricted` also sets `restrict_to_workspace=True`. | Behaviour depends on which path a tool takes; the docs say "reads anywhere" but L5 still exists. |
| G4 | The project folder is not an implicit writable root. | The first edit in a newly opened project triggers a grant prompt or a turn stop. Codex treats it as the working area by default. |
| G5 | Denials are silent in discovery (`tool_search` says "no match"; the namespace index omits the tool). | The agent cannot explain the problem or suggest switching mode. |
| G6 | Authority machinery (L6) is large and serves L3/L4. | Every new capability needs special-casing; there is a high cost to change. |

## 5. Target model

Three layers, each with one job.

**A. Execution sandbox (commands and file tools only).**
- Per-chat mode: `read-only` | `workspace` (default) | `full`.
- In `workspace` mode, the writable roots are the chat's project folder, the
  agent home, private temp/cache, and any extra roots from Settings → Folders.
  Folder grants become additional roots, not the base.
- Reads are broad, except deny-listed paths (kohaku private runtime, user
  deny globs). Protected subpaths inside roots are read-only (`.git`
  internals, kohaku config).
- Enforced by Seatbelt or bwrap. If the sandbox is unavailable, refuse with a
  reason; never fall back silently.
- Retire L5 for WebUI chats. Keep it only for channels that have no OS
  sandbox (Telegram etc.), documented as an app-level guard.

**B. Approval policy (every tool call).**
- The existing L1 profiles and rules stay as they are.
- Crossing the sandbox (write outside the roots, unsandboxed exec, network
  exceptions) is an approval request under `balanced` / `strict`, not a turn
  stop. `autonomous` uses the auto-reviewer for these.

**C. Per-capability controls (browser, desktop, device, MCP).**
- A tool is present when its own config enables it. The sandbox mode never
  removes it.
- Browser: L1 action classes plus the following.
  - `file://` and local-path navigation count as state-changing (they ask).
  - Upload and download count as state-changing (already the case).
  - In `workspace` mode, screenshots and downloads go only to a
    kohaku-owned media directory.
  - Optional per-site allow/deny rules later.
- Desktop: action classes plus an optional always-allowed app list.
- MCP: `enabledTools` as today.
- An explicit `deny` rule is the only way to hide a tool in a mode, and
  `tool_search` reports "blocked by <rule>" instead of "no match".

What we give up: in `workspace` mode, an approved browser upload or desktop
action can reach outside the writable roots. This is the same trade Codex
makes, and approvals cover it.

## 6. Migration plan

| Phase | Change | Size | Tests |
|---|---|---|---|
| P1 | Remove the L4 name gate for `agent_browser`, `desktop_use`, `agent_device` (main agent and subagents); add the `file://` and screenshot-path rules to the browser classifier; make `tool_search` and the namespace index report policy-blocked tools. | Small | Reproduce the Figma case in a managed WebUI turn; update `tests/agent/test_managed_exec.py:535` and `tests/agent/test_managed_mcp_access.py:209` (they encode the current design on purpose). |
| P2 | Make managed narrowing a property of the chat mode: `full` never narrows; drop the gateway-wide activation flag as the switch. | Small–medium | Full access chat keeps every enabled tool; Sandbox chat keeps the write boundary. |
| P3 | Treat the chat's project folder as an implicit writable root in `workspace` mode; folder grants become extra roots; retire L5 for WebUI. | Medium | First edit in a new project needs no prompt; writes outside the roots still ask; private runtime stays hidden. |
| P4 | Reassess L6 (task/full references, native-admission ledger) once L4 is gone; keep only what A and B need. | Medium–large | Follow the P6 validation rules in `AGENTS.md` (type-regression comparator, full suite). |

P1 unblocks the browser on its own. P2 and P3 change behaviour users see and
should ship with updated docs (`docs/sandbox-folders.md`, `.agent/security.md`).

## 7. Open questions

1. Should `autonomous` auto-approve sandbox crossings through the auto-reviewer,
   or still ask for writes outside the roots?
2. Should browser site permissions (per-domain allow/deny) be in P1, or wait
   until there is a need?
3. Does anything still depend on the gateway-wide `.folder-grants-active` flag
   (for example the native-process barrier) after P2?

