"""Tool wrapper for the agent-browser CLI."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import time
from collections.abc import Sequence
from typing import Any, cast

import httpx

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.process import run_owned_process
from nanobot.agent.turn import ToolOutcome

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
    # Screenshots are potentially file-producing even when no output path is
    # present in the invocation.  Treat every screenshot form, including
    # option-based forms such as ``open --screenshot=page.png``, as a write.
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
    # A second CLI action indicates a mixed/ambiguous request. Safe action
    # arguments (URLs, selectors, offsets) are not action names and remain
    # valid for the first command.
    action_names = READ_NAVIGATION_ACTIONS | frozenset({
        "click", "type", "fill", "submit", "evaluate", "input", "press", "app",
        "settings", "close", "record", "upload", "download",
    })
    if any(arg in action_names for arg in normalized_args[normalized_args.index(action) + 1:]):
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
    ):
        self.package = package
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.working_dir = working_dir
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
    ) -> str | ToolOutcome:
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

        return json.dumps(
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


__all__ = [
    "AgentBrowserTool",
    "READ_NAVIGATION_ACTIONS",
    "browser_action_metadata",
    "classify_agent_browser_action",
    "classify_browser_action",
]
