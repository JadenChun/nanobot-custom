from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from nanobot.agent.tools.agent_device import AgentDeviceTool


class _FakeProcess:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self):
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_agent_device_requires_args() -> None:
    tool = AgentDeviceTool()
    result = await tool.execute(args=[])
    assert result == "Error: args must include at least one CLI argument"


@pytest.mark.asyncio
async def test_agent_device_reports_missing_npx() -> None:
    tool = AgentDeviceTool()
    with patch(
        "nanobot.agent.tools.agent_device.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError,
    ):
        result = await tool.execute(args=["devices", "--platform", "ios"])

    assert "npx" in result
    assert "Install Node.js/npm first" in result


@pytest.mark.asyncio
async def test_agent_device_returns_json_payload_for_success() -> None:
    tool = AgentDeviceTool(working_dir="/tmp/work")
    fake_process = _FakeProcess(stdout=b"booted\n", stderr=b"", returncode=0)

    with patch(
        "nanobot.agent.tools.agent_device.asyncio.create_subprocess_exec",
        return_value=fake_process,
    ) as mock_exec:
        result = await tool.execute(args=["devices", "--platform", "ios"])

    payload = json.loads(result)
    assert payload["command"] == ["npx", "--yes", "agent-device", "devices", "--platform", "ios"]
    assert payload["cwd"] == "/tmp/work"
    assert payload["exitCode"] == 0
    assert payload["stdout"] == "booted\n"
    assert payload["stderr"] == ""
    mock_exec.assert_called_once()
