# Security Boundaries

The agent operates with significant power (file system, shell, web). The following guards must not be bypassed when modifying related code.

## Workspace Restriction

Filesystem tools (`read_file`, `write_file`, `edit_file`, `list_dir`, `apply_patch`) resolve paths through the workspace path resolver (`agent/tools/filesystem.py` / `agent/tools/path_utils.py`), which enforces that the resolved path must lie under the active workspace when workspace restriction is enabled. The media upload directory is always an internal extra read root while restricted.

Additional filesystem roots must be capability-specific. `extra_allowed_dirs` is a legacy read-only alias. Use `extra_read_allowed_dirs` for read-only roots, `extra_write_allowed_dirs` only when a write-capable tool is intentionally allowed to modify an extra directory, and exact file allowlists when a tool may modify only specific files.

Shell execution (`ExecTool`, `agent/tools/shell.py`) also respects `restrict_to_workspace` as an application-level guard: if enabled and `working_dir` is outside the workspace, the command is rejected before execution, and command text is checked for obvious workspace escapes. This is not process-level isolation; use an exec sandbox backend for that.

**Rule**: Any new path-handling logic must go through the workspace path resolver or perform an equivalent containment check with explicit read/write capability semantics.

- **WebUI Sandbox workspace-write**: Human-selected **Working in** (`selection_source == "user"`) grants implicit write access (priority 0) in Sandbox, enforced in both `ResourceAccessPolicy` and `bwrap`/Seatbelt `_managed_mounts()`, with explicit `read_only` grants (priority 1) overriding workspace write.
- **Tool focus provenance**: Model-called `select_workspace` (`selection_source == "tool"`) changes relative-path focus only and must never widen write authority.
- **Non-WebUI channels**: Legacy `restrict_to_workspace` remains active for non-WebUI channels (Telegram, Discord, CLI, API) that do not run the WebUI OS sandbox.

## SSRF Protection

All outbound HTTP requests from agent tools must pass through the shared URL guards in `security/network.py` (`validate_url_target` or `resolve_url_target`). By default they block loopback, RFC1918 private addresses, CGNAT ranges, link-local ranges, and cloud metadata endpoints (including `169.254.169.254`).

For direct requests, the only escape hatch is `configure_ssrf_whitelist(cidrs)`, which reads from `config.tools.ssrf_whitelist` at load time. An explicitly configured `providers.<name>.proxy` is a separate user-authorized trust boundary for provider requests and provider-returned image URL downloads. Those downloads still reject malformed URLs and locally identifiable private/internal targets on every redirect, but hostnames unavailable to local DNS are delegated to the trusted proxy. The user-selected proxy owns final DNS resolution and network egress policy.

HTTP/SSE MCP transports are part of this boundary: validate configured MCP URLs before probing or constructing clients, and validate each outgoing HTTP request before redirects are followed. Local/private HTTP MCP endpoints are allowed only through the explicit SSRF whitelist. Stdio MCP servers are not part of the HTTP SSRF path.

**Rule**: Do not add direct `httpx.get` / `requests.get` calls in tools. Route through the existing web fetch utilities or replicate the `validate_url_target` check.

## Shell Sandbox

`tools/sandbox.py` provides optional command wrapping. The only backend currently shipped is `bwrap` (bubblewrap), intended for containerized deployments. On Windows and bare-metal Linux without `bwrap`, commands run in the native shell with workspace restriction as an application-level guard only.

**Rule**: If adding a new sandbox backend, implement `_wrap_<name>(command, workspace, cwd) -> str` and register it in `_BACKENDS`.

## Host-Process Tools and Policy-Only Access

Managed WebUI turns do not use a static tool allowlist; all enabled built-in tools, installed `nanobot.tools` plugins, and configured MCP servers remain discoverable in both Sandbox and Full access modes.

- **Host-process declaration**: Tools that launch or drive host-level processes outside the OS Sandbox (`agent_browser`, `agent_device`, `desktop_use`, `run_cli_app`, `trend_vpn_*`) declare `Tool.launches_host_process = True` and resolve their working directory through `ResourceAccessPolicy.automation_working_dir()`.
- **Policy & auto-review**: Every tool call is governed by `RiskyActionPolicy` (`deny` → `allow` → Sandbox host-process crossing → profile/`ask` rules) and the generalized 2-stage `auto_review.py` (which never auto-approves explicit `ask`/`deny` rules, credential/system paths, or Kohaku config/runtime paths such as `~/.kohaku`, `~/.nanobot`, and `config.json`).
- **Transparent discovery denials**: Explicit `deny` rules block schema activation in `ToolSurface` and are reported transparently by `tool_search` instead of hiding the tool.
- **Trusted MCP & plugins**: Configured MCP servers and installed `nanobot.tools` plugins are trusted on configuration/install (`launches_host_process = False` by default), while `config.json` and private runtime roots remain protected.
