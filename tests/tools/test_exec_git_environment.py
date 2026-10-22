"""Git defaults reach both execution modes without importing unrelated secrets."""

from unittest.mock import AsyncMock

import pytest

from nanobot.agent.tools import shell
from nanobot.agent.tools.shell import ExecTool


@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("ssh_command", [None, "ssh -i /configured/key", ""])
async def test_session_preparation_preserves_git_defaults(monkeypatch, windows, ssh_command):
    monkeypatch.setattr(shell, "_IS_WINDOWS", windows)
    monkeypatch.setenv("NANOBOT_SECRET_TOKEN", "do-not-copy")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "1")
    if ssh_command is None:
        monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    else:
        monkeypatch.setenv("GIT_SSH_COMMAND", ssh_command)
    launch = AsyncMock(return_value="session-ready")
    monkeypatch.setattr(ExecTool, "_execute_session", launch)
    tool = ExecTool()
    assert await tool.execute(command="git fetch", yield_time_ms=0) == "session-ready"
    launch.assert_awaited_once()
    prepared = launch.call_args.args[0]
    assert prepared.env["GIT_TERMINAL_PROMPT"] == "0"
    assert prepared.env["GIT_SSH_COMMAND"] == (
        "ssh -o BatchMode=yes -o ConnectTimeout=10" if ssh_command is None else ssh_command
    )
    assert "NANOBOT_SECRET_TOKEN" not in prepared.env


async def test_denied_git_command_does_not_reach_execution(monkeypatch):
    launch = AsyncMock(side_effect=AssertionError("denied command launched"))
    monkeypatch.setattr(ExecTool, "_execute_session", launch)
    result = await ExecTool(deny_patterns=[r"git\s+push"]).execute(
        command="git push", yield_time_ms=0,
    )
    assert "blocked by deny pattern" in result
    launch.assert_not_awaited()
