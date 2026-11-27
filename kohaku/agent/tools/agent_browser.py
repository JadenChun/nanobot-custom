"""Tool wrapper for the agent-browser CLI."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx

from kohaku.agent.tools.base import Tool
from kohaku.agent.tools.process import run_owned_process
from kohaku.agent.turn import ToolOutcome

if TYPE_CHECKING:
    from kohaku.context_repo.policy import ResourceAccessPolicy

READ_NAVIGATION_ACTIONS = frozenset({
    "back",
    "console",
    "errors",
    "forward",
    "inspect",
    "navigate",
    "network",
    "open",
    "reload",
    "scroll",
    "snapshot",
    "url",
    "wait",
})
_FILE_OUTPUT_OPTIONS = frozenset({
    "--file",
    "--filename",
    "--output",
    "--out",
    "--path",
    "--save",
    "-o",
})
_CONFIG_OWNED_OPTIONS = frozenset({
    "--cdp",
    "--session",
    "--pin-tab",
    "--no-pin-tab",
    "--executable-path",
})
# Screenshot options that only shape the image. Without a path or
# ``--screenshot-dir``, agent-browser writes the image to a temporary directory.
_TEMP_SCREENSHOT_FLAGS = frozenset({"--full", "-f", "--annotate", "--json"})
_LEADING_TEMP_SCREENSHOT_FLAGS = _TEMP_SCREENSHOT_FLAGS | frozenset({"--headed"})
_TEMP_SCREENSHOT_VALUE_OPTIONS = frozenset({"--screenshot-format", "--screenshot-quality"})
# Local or script URLs read host files or run code; they are not read-navigation.
_NON_NAVIGATION_URL_SCHEMES = ("file:", "javascript:", "data:", "view-source:")
_IMAGE_PATH_KEYS = ("path", "file", "screenshot", "output")
_IMAGE_PATH_PATTERN = re.compile(
    r"""(["']?)([^\s"'<>|*?\r\n]+\.(?:png|jpe?g))\1""",
    re.IGNORECASE,
)
_DEFAULT_MAX_INLINE_SCREENSHOT_BYTES = 900 * 1024
try:
    _max_inline_screenshot_bytes = int(
        os.environ.get(
            "KOHAKU_AGENT_BROWSER_MAX_INLINE_BYTES",
            _DEFAULT_MAX_INLINE_SCREENSHOT_BYTES,
        )
    )
except (TypeError, ValueError):
    _max_inline_screenshot_bytes = _DEFAULT_MAX_INLINE_SCREENSHOT_BYTES
_MAX_INLINE_SCREENSHOT_BYTES = max(1, _max_inline_screenshot_bytes)


def _is_temp_screenshot(args: Sequence[str]) -> bool:
    """Return whether args are ``[global shaping flags] screenshot [@ref] [shaping options]``.

    Any other positional argument could be an output path, so it is not
    accepted here.
    """
    if not args:
        return False
    seen_screenshot = False
    refs = 0
    index = 0
    while index < len(args):
        arg = args[index]
        option, has_value, _ = arg.partition("=")
        if not seen_screenshot:
            if arg == "screenshot":
                seen_screenshot = True
                index += 1
                continue
            if option in _LEADING_TEMP_SCREENSHOT_FLAGS and not has_value:
                index += 1
                continue
            if option in _TEMP_SCREENSHOT_VALUE_OPTIONS:
                if not has_value and index + 1 >= len(args):
                    return False
                index += 1 if has_value else 2
                continue
            return False
        if option in _TEMP_SCREENSHOT_FLAGS and not has_value:
            index += 1
            continue
        if option in _TEMP_SCREENSHOT_VALUE_OPTIONS:
            if not has_value and index + 1 >= len(args):
                return False
            index += 1 if has_value else 2
            continue
        if arg.startswith("@") and refs == 0:
            refs += 1
            index += 1
            continue
        return False
    return seen_screenshot and index == len(args)


def _is_local_filesystem_path(arg: str) -> bool:
    """Return True if *arg* looks like a bare local filesystem path."""
    if arg.startswith(("http://", "https://")):
        return False
    if arg in {"~", ".", ".."} or arg.startswith(("/", "~/", "./", "../", "\\")):
        return True
    if len(arg) >= 3 and arg[0].isalpha() and arg[1] == ":" and arg[2] in ("/", "\\"):
        return True
    return False


def _is_screenshot_invocation(args: Sequence[str]) -> bool:
    normalized = [arg.strip().lower() for arg in args]
    return "screenshot" in normalized or any(
        arg == "--screenshot" or arg.startswith("--screenshot=") for arg in normalized
    )


def _find_image_path_in_json(obj: object) -> str | None:
    if isinstance(obj, dict):
        mapping = cast(dict[str, Any], obj)
        for key in _IMAGE_PATH_KEYS:
            value = mapping.get(key)
            if isinstance(value, str) and value.strip().lower().endswith((".png", ".jpg", ".jpeg")):
                return value.strip()
        for value in mapping.values():
            found = _find_image_path_in_json(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in cast(list[Any], obj):
            found = _find_image_path_in_json(item)
            if found is not None:
                return found
    return None


def _extract_screenshot_path(stdout_text: str, cwd: str) -> Path | None:
    stripped = stdout_text.strip()
    if not stripped:
        return None
    candidate: str | None = None
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            candidate = _find_image_path_in_json(parsed)
    if candidate is None:
        for line in stdout_text.splitlines():
            lower_line = line.lower()
            marker = "saved to "
            idx = lower_line.find(marker)
            if idx != -1:
                after = line[idx + len(marker) :].strip().strip("\"'")
                if after.lower().endswith((".png", ".jpg", ".jpeg")):
                    candidate = after
                    break
    if candidate is None:
        match = _IMAGE_PATH_PATTERN.search(stdout_text)
        if match is not None:
            candidate = match.group(2)
    if not candidate:
        return None
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = Path(cwd) / path
    return path


def _inline_screenshot_image(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        size = path.stat().st_size
    except OSError:
        return None
    if size <= 0 or size > _MAX_INLINE_SCREENSHOT_BYTES:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    else:
        return None
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
    }


def _config_owned_option(args: Sequence[str]) -> str | None:
    for arg in args:
        option = arg.split("=", 1)[0].strip().lower()
        if option in _CONFIG_OWNED_OPTIONS:
            return option
    return None


def _reports_stale_pinned_tab(stdout: str, stderr: str) -> bool:
    # agent-browser emits tab_gone in both human-readable stderr and --json
    # stdout error objects. Checking the serialized text handles either shape.
    return "tab_gone" in stdout.lower() or "tab_gone" in stderr.lower()


def classify_browser_action(args: object) -> str:
    """Classify agent-browser args for a later policy layer.

    Navigation and inspection primitives are read-navigation operations.  A
    missing/unknown command is conservatively state-changing, as are generic
    click/type/submit/evaluate commands.
    """

    if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
        return "state_changing"
    if any(not isinstance(arg, str) for arg in cast(Sequence[object], args)):
        return "state_changing"
    normalized_args = [arg.strip().lower() for arg in cast(Sequence[object], args) if isinstance(arg, str)]
    option_names = [arg.split("=", 1)[0] for arg in normalized_args]
    if any(arg.startswith(_NON_NAVIGATION_URL_SCHEMES) for arg in normalized_args):
        return "state_changing"
    if _is_temp_screenshot(normalized_args):
        # Viewing the page: the image goes to agent-browser's temporary directory.
        return "read_navigation"
    # Screenshots are potentially file-producing even when no output path is
    # present in the invocation. Every other screenshot form, including
    # option-based forms such as ``open --screenshot=page.png``, is a write.
    if "screenshot" in normalized_args or any(
        arg.startswith("--screenshot") for arg in normalized_args
    ):
        return "state_changing"
    if any(
        option in _FILE_OUTPUT_OPTIONS
        or option.startswith(("--output", "--out", "--path", "--file", "--save"))
        for option in option_names
    ):
        return "state_changing"
    if any(arg in {"--help", "-h", "--version"} for arg in normalized_args):
        # Help/version are safe only when they are the command itself. A
        # state-changing command mixed into the same invocation is unknown.
        command_tokens = [arg for arg in normalized_args if not arg.startswith("-")]
        if len(command_tokens) > 1:
            return "state_changing"
        return "read_navigation"
    action = next((arg for arg in normalized_args if arg and not arg.startswith("-")), "")
    if action not in READ_NAVIGATION_ACTIONS:
        return "state_changing"
    trailing_args = normalized_args[normalized_args.index(action) + 1 :]
    if action in {"open", "goto", "navigate"} and any(
        not arg.startswith("-") and _is_local_filesystem_path(arg)
        for arg in trailing_args
    ):
        return "state_changing"
    # A second CLI action indicates a mixed/ambiguous request. Safe action
    # arguments (URLs, selectors, offsets) are not action names and remain
    # valid for the first command.
    action_names = READ_NAVIGATION_ACTIONS | frozenset({
        "click", "type", "fill", "submit", "evaluate", "input", "press", "app",
        "settings", "close", "record", "upload", "download",
    })
    if any(arg in action_names for arg in trailing_args):
        return "state_changing"
    return "read_navigation"


def browser_action_metadata(args: Sequence[str]) -> dict[str, Any]:
    """Return machine-readable browser action classification metadata."""

    action_class = classify_browser_action(args)
    return {
        "action_class": action_class,
        "read_navigation": action_class == "read_navigation",
        "state_changing": action_class == "state_changing",
    }


# Explicit alias for policy consumers that use a tool-prefixed name.
classify_agent_browser_action = classify_browser_action


def _resolve_npx() -> str | None:
    """Resolve the runnable npx command, including Windows .cmd launchers."""

    return shutil.which("npx") or shutil.which("npx.cmd")


class AgentBrowserTool(Tool):
    exposure = "deferred"
    namespace = "ui"
    search_keywords = ("browser", "screenshot", "web app", "click", "navigate")
    """Execute browser automation tasks via agent-browser."""

    def __init__(
        self,
        package: str = "agent-browser",
        timeout: int = 180,
        max_output_chars: int = 12000,
        working_dir: str | None = None,
        read_only: bool = False,
        cdp_url: str = "",
        pin_tab: bool = True,
        session: str = "default",
        launch_command: str = "",
        launch_wait_seconds: int = 20,
        executable_path: str = "",
        *,
        resource_policy: ResourceAccessPolicy | None = None,
    ):
        self.package = package
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.working_dir = working_dir
        self._resource_policy = resource_policy
        self._read_only_mode = read_only
        self.cdp_url = cdp_url.strip()
        self.pin_tab = pin_tab
        self.session = session.strip()
        self.launch_command = launch_command.strip()
        self.launch_wait_seconds = max(1, int(launch_wait_seconds))
        self.executable_path = executable_path.strip()
        self._last_launch_attempt = 0.0

    @property
    def read_only(self) -> bool:
        return self._read_only_mode

    @property
    def exclusive(self) -> bool:
        return True

    @property
    def launches_host_process(self) -> bool:
        return True

    @property
    def supports_parallel_calls(self) -> bool:
        """Browser sessions are shared state and must be serialized."""
        return False

    async def _cdp_ready(self) -> bool:
        if not self.cdp_url:
            return False
        try:
            async with httpx.AsyncClient(timeout=0.6) as client:
                response = await client.get(f"{self.cdp_url.rstrip('/')}/json/version")
            return response.status_code == 200
        except Exception:
            return False

    async def _launch_shared_browser(self) -> bool:
        """Start the shared browser and wait for its CDP endpoint."""
        if not self.launch_command:
            return False
        try:
            argv = shlex.split(self.launch_command)
        except ValueError:
            return False
        if not argv:
            return False
        try:
            subprocess.Popen(  # noqa: S603 - operator-configured launch command
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            return False
        deadline = time.monotonic() + self.launch_wait_seconds
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            if await self._cdp_ready():
                return True
        return False

    async def _ensure_browser_session(self) -> bool:
        if not self.cdp_url:
            return False
        if await self._cdp_ready():
            return True
        # Do not pay the launch wait on every call when the browser cannot start
        # (for example, it is already running on the profile without the port).
        if not self.launch_command or time.monotonic() - self._last_launch_attempt < 60:
            return False
        self._last_launch_attempt = time.monotonic()
        return await self._launch_shared_browser()

    _META_COMMANDS = frozenset({
        "dashboard",
        "doctor",
        "install",
        "mcp",
        "plugin",
        "session",
        "skills",
        "upgrade",
    })

    def _needs_browser_session(self, args: Sequence[str]) -> bool:
        if any(arg in {"--help", "-h", "--version", "-V"} for arg in args):
            return False
        first_positional = next((arg for arg in args if not arg.startswith("-")), "")
        return first_positional not in self._META_COMMANDS

    async def _reset_pinned_tab(self, npx_command: str, cwd: str) -> bool:
        command = [
            npx_command,
            "--yes",
            self.package,
            *self._browser_flags(True),
            "tab",
            "new",
            "about:blank",
        ]
        try:
            result = await run_owned_process(
                command,
                timeout=min(self.timeout, 60),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except Exception:
            return False
        return result.returncode == 0 and not result.timed_out

    def _browser_flags(self, attached: bool) -> list[str]:
        if attached:
            flags = ["--cdp", self.cdp_url]
            session = self.session
        else:
            flags = ["--executable-path", self.executable_path] if self.executable_path else []
            session = f"nanobot-isolated-{self.session}" if self.session else "nanobot-isolated"
        if session:
            flags.extend(["--session", session])
        flags.append("--pin-tab" if self.pin_tab else "--no-pin-tab")
        return flags

    @property
    def name(self) -> str:
        return "agent_browser"

    @property
    def description(self) -> str:
        session_hint = (
            " A shared Chrome session is attached automatically: it runs on the user's real "
            "Chrome profile, so their cookies, sign-ins, and open tabs are available. Use `tab` "
            "to switch tabs and prefer inspecting the existing page before navigating."
            if self.cdp_url
            else ""
        )
        return (
            "Run agent-browser CLI for browser/electron automation, web-app testing, visual QA, "
            "screenshots, and session recording. Pass CLI args as a string array (for example "
            "['--help'], ['--headed', 'open', 'https://example.com'], ['dashboard', 'start'], "
            "['screenshot', 'page.png', '--annotate'], or ['record', 'start', 'session.webm'])."
            + session_hint
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "description": "CLI arguments passed to agent-browser.",
                    "items": {"type": "string"},
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (5-900).",
                    "minimum": 5,
                    "maximum": 900,
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command.",
                },
            },
            "required": ["args"],
        }

    async def execute(
        self,
        args: list[str],
        timeout: int | None = None,
        working_dir: str | None = None,
        **kwargs: Any,
    ) -> str | list[dict[str, Any]] | ToolOutcome:
        if not args:
            return "Error: args must include at least one CLI argument"
        if len(args) > 80:
            return "Error: too many CLI arguments (max 80)"
        if any(len(arg) > 2000 for arg in args):
            return "Error: one or more CLI arguments exceed max length (2000 chars)"
        if option := _config_owned_option(args):
            return ToolOutcome(
                content=f"Error: {option} is controlled by browser configuration",
                stop_reason="policy_blocked",
                policy_metadata={
                    "policy": "browser_configuration",
                    "reason": "caller-supplied configuration override",
                    "option": option,
                    "requires_approval": False,
                },
            )
        if self.read_only and working_dir is not None:
            return ToolOutcome(
                content="Error: caller-supplied working_dir is blocked for delegated read-only browser actions",
                stop_reason="policy_blocked",
                policy_metadata={
                    "policy": "delegated_read_only",
                    "reason": "caller working_dir override",
                    "requires_approval": False,
                },
            )
        if self.read_only and classify_browser_action(args) != "read_navigation":
            return ToolOutcome(
                content="Error: state-changing or unknown browser action is blocked",
                stop_reason="policy_blocked",
                policy_metadata={
                    "policy": "delegated_read_only",
                    "action_class": "state_changing",
                    "requires_approval": False,
                },
            )

        run_timeout = timeout if timeout is not None else self.timeout
        cwd = working_dir or self.working_dir or os.getcwd()
        if self._resource_policy is not None:
            confined = self._resource_policy.automation_working_dir(working_dir, self.working_dir)
            if confined is None:
                return ToolOutcome(
                    content="Error: working_dir is outside the workspace for this chat",
                    stop_reason="policy_blocked",
                    policy_metadata={"policy": "workspace_access", "requires_approval": False},
                )
            cwd = str(confined)

        npx_command = _resolve_npx()
        if not npx_command:
            return (
                "Error: 'npx' was not found in PATH. "
                "Install Node.js/npm first, then retry."
            )

        needs_session = self._needs_browser_session(args)
        attached = False
        if needs_session and self.cdp_url:
            attached = await self._ensure_browser_session()
            if not attached:
                return json.dumps(
                    {
                        "error": "Configured shared browser is unavailable; browser action was not run",
                        "browser_session": {
                            "attached": False,
                            "cdp_url": self.cdp_url,
                        },
                    },
                    ensure_ascii=False,
                )
        command = [
            npx_command,
            "--yes",
            self.package,
            *(self._browser_flags(attached) if needs_session else []),
            *args,
        ]

        async def run() -> Any:
            return await run_owned_process(
                command,
                timeout=run_timeout,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

        try:
            result = await run()
        except Exception as e:
            return f"Error: failed to start agent-browser: {e}"

        stdout_text = result.stdout.decode("utf-8", errors="replace")
        stderr_text = result.stderr.decode("utf-8", errors="replace")

        # A pinned tab can disappear (the user closed it). Recreate a bound tab
        # once and retry so the session does not stay wedged.
        stale_pinned_tab = (
            attached
            and self.pin_tab
            and result.returncode != 0
            and not result.timed_out
            and _reports_stale_pinned_tab(stdout_text, stderr_text)
        )
        if stale_pinned_tab and self.read_only:
            return "Error: pinned browser tab is stale; delegated read-only action cannot reset a tab"
        if stale_pinned_tab and await self._reset_pinned_tab(npx_command, cwd):
            try:
                result = await run()
            except Exception as e:
                return f"Error: failed to start agent-browser: {e}"
            stdout_text = result.stdout.decode("utf-8", errors="replace")
            stderr_text = result.stderr.decode("utf-8", errors="replace")

        if result.timed_out:
            return json.dumps(
                {
                    "error": f"agent-browser timed out after {run_timeout} seconds",
                    "command": command,
                    "cwd": cwd,
                    "action_class": classify_browser_action(args),
                    "browser_session": {
                        "attached": attached,
                        "cdp_url": self.cdp_url or None,
                    },
                },
                ensure_ascii=False,
            )

        stdout_truncated = len(stdout_text) > self.max_output_chars
        stderr_truncated = len(stderr_text) > self.max_output_chars
        if stdout_truncated:
            stdout_text = stdout_text[: self.max_output_chars]
        if stderr_truncated:
            stderr_text = stderr_text[: self.max_output_chars]

        payload_text = json.dumps(
            {
                "command": command,
                "cwd": cwd,
                "exitCode": result.returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "stdoutTruncated": stdout_truncated,
                "stderrTruncated": stderr_truncated,
                "action_class": classify_browser_action(args),
                "browser_session": {
                    "attached": attached,
                    "cdp_url": self.cdp_url or None,
                },
            },
            ensure_ascii=False,
        )
        if result.returncode == 0 and _is_screenshot_invocation(args):
            screenshot_path = _extract_screenshot_path(stdout_text, cwd)
            grants = (
                self._resource_policy.folder_grants
                if self._resource_policy is not None
                else None
            )
            is_private = (
                screenshot_path is not None
                and grants is not None
                and grants.is_private_path(screenshot_path)
            )
            image_block = (
                _inline_screenshot_image(screenshot_path)
                if screenshot_path is not None and not is_private
                else None
            )
            if image_block is not None:
                return [{"type": "text", "text": payload_text}, image_block]
        return payload_text


__all__ = [
    "AgentBrowserTool",
    "READ_NAVIGATION_ACTIONS",
    "browser_action_metadata",
    "classify_agent_browser_action",
    "classify_browser_action",
]
