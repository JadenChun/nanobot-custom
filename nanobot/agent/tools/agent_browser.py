"""Tool wrapper for the @agentic/agent-browser CLI."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from nanobot.agent.tools.base import Tool


class AgentBrowserTool(Tool):
    """Execute browser automation tasks via @agentic/agent-browser."""

    def __init__(
        self,
        package: str = "@agentic/agent-browser@latest",
        timeout: int = 180,
        max_output_chars: int = 12000,
        working_dir: str | None = None,
    ):
        self.package = package
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.working_dir = working_dir

    @property
    def name(self) -> str:
        return "agent_browser"

    @property
    def description(self) -> str:
        return (
            "Run @agentic/agent-browser CLI for browser/electron automation. "
            "Pass CLI args as a string array (for example ['--help'] or ['run', 'Find X', '--json'])."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "description": "CLI arguments passed to @agentic/agent-browser.",
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
    ) -> str:
        if not args:
            return "Error: args must include at least one CLI argument"
        if len(args) > 80:
            return "Error: too many CLI arguments (max 80)"
        if any(len(arg) > 2000 for arg in args):
            return "Error: one or more CLI arguments exceed max length (2000 chars)"

        run_timeout = timeout if timeout is not None else self.timeout
        cwd = working_dir or self.working_dir or os.getcwd()

        command = ["npx", "--yes", self.package, *args]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except FileNotFoundError:
            return (
                "Error: 'npx' was not found in PATH. "
                "Install Node.js/npm first, then retry."
            )
        except Exception as e:
            return f"Error: failed to start @agentic/agent-browser: {e}"

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=run_timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            return json.dumps(
                {
                    "error": f"agent-browser timed out after {run_timeout} seconds",
                    "command": command,
                    "cwd": cwd,
                },
                ensure_ascii=False,
            )

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

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
                "exitCode": process.returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "stdoutTruncated": stdout_truncated,
                "stderrTruncated": stderr_truncated,
            },
            ensure_ascii=False,
        )
