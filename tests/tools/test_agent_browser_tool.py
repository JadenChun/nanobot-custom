from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from nanobot.agent.tools.agent_browser import AgentBrowserTool


class _FakeProcess:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self):
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_agent_browser_requires_args() -> None:
    tool = AgentBrowserTool()
    result = await tool.execute(args=[])
    assert result == "Error: args must include at least one CLI argument"


@pytest.mark.asyncio
async def test_agent_browser_reports_missing_npx() -> None:
    tool = AgentBrowserTool()
    with patch("nanobot.agent.tools.agent_browser._resolve_npx", return_value=None):
        result = await tool.execute(args=["--version"])

    assert "npx" in result
    assert "Install Node.js/npm first" in result


@pytest.mark.asyncio
async def test_agent_browser_returns_json_payload_for_success() -> None:
    tool = AgentBrowserTool(working_dir="/tmp/work")
    fake_process = _FakeProcess(stdout=b"agent-browser 0.25.5\n", stderr=b"", returncode=0)
    npx_path = r"C:\Program Files\nodejs\npx.cmd"

    with patch(
        "nanobot.agent.tools.agent_browser._resolve_npx",
        return_value=npx_path,
    ), patch(
        "nanobot.agent.tools.agent_browser.asyncio.create_subprocess_exec",
        return_value=fake_process,
    ) as mock_exec:
        result = await tool.execute(args=["--version"])

    payload = json.loads(result)
    assert payload["command"] == [npx_path, "--yes", "agent-browser", "--version"]
    assert payload["cwd"] == "/tmp/work"
    assert payload["exitCode"] == 0
    assert payload["stdout"] == "agent-browser 0.25.5\n"
    assert payload["stderr"] == ""
    mock_exec.assert_called_once()
