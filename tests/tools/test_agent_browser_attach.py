"""Tests for the shared-browser attach behaviour of the agent-browser tool."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.agent_browser import AgentBrowserTool


def _process() -> AsyncMock:
    return AsyncMock(
        return_value=SimpleNamespace(
            timed_out=False,
            stdout=b"ok\n",
            stderr=b"",
            returncode=0,
        )
    )


def _prepare(monkeypatch, process: AsyncMock) -> None:
    monkeypatch.setattr("nanobot.agent.tools.agent_browser._resolve_npx", lambda: "npx")
    monkeypatch.setattr("nanobot.agent.tools.agent_browser.run_owned_process", process)
    monkeypatch.setattr(
        "nanobot.agent.tools.agent_browser.asyncio.sleep", AsyncMock(return_value=None)
    )


@pytest.mark.asyncio
async def test_attached_session_flags_are_prepended(monkeypatch):
    process = _process()
    _prepare(monkeypatch, process)
    tool = AgentBrowserTool(cdp_url="http://127.0.0.1:9222", session="agent")
    monkeypatch.setattr(tool, "_cdp_ready", AsyncMock(return_value=True))

    payload = json.loads(await tool.execute(["snapshot"]))

    assert payload["command"][:4] == ["npx", "--yes", "agent-browser", "--cdp"]
    assert payload["command"][4] == "http://127.0.0.1:9222"
    assert "--pin-tab" in payload["command"]
    session_flag = payload["command"].index("--session")
    assert payload["command"][session_flag + 1] == "agent"
    assert payload["command"][-1] == "snapshot"
    assert payload["browser_session"] == {
        "attached": True,
        "cdp_url": "http://127.0.0.1:9222",
    }


@pytest.mark.asyncio
async def test_pin_tab_can_be_disabled(monkeypatch):
    process = _process()
    _prepare(monkeypatch, process)
    tool = AgentBrowserTool(cdp_url="http://127.0.0.1:9222", pin_tab=False, session="")
    monkeypatch.setattr(tool, "_cdp_ready", AsyncMock(return_value=True))

    payload = json.loads(await tool.execute(["snapshot"]))

    assert "--pin-tab" not in payload["command"]
    assert "--no-pin-tab" in payload["command"]
    assert "--session" not in payload["command"]


@pytest.mark.asyncio
async def test_launch_command_starts_shared_browser(monkeypatch):
    process = _process()
    _prepare(monkeypatch, process)
    tool = AgentBrowserTool(
        cdp_url="http://127.0.0.1:9222",
        launch_command="/home/jaden/.local/bin/chrome-agent",
    )
    ready = AsyncMock(side_effect=[False, False, True])
    monkeypatch.setattr(tool, "_cdp_ready", ready)
    popen = MagicMock()
    monkeypatch.setattr("nanobot.agent.tools.agent_browser.subprocess.Popen", popen)

    payload = json.loads(await tool.execute(["get", "url"]))

    popen.assert_called_once()
    assert popen.call_args.args[0] == ["/home/jaden/.local/bin/chrome-agent"]
    assert payload["browser_session"]["attached"] is True


@pytest.mark.asyncio
async def test_unreachable_configured_session_fails_before_cli_action(monkeypatch):
    process = _process()
    _prepare(monkeypatch, process)
    tool = AgentBrowserTool(
        cdp_url="http://127.0.0.1:9222",
        executable_path="/opt/chrome",
    )
    monkeypatch.setattr(tool, "_cdp_ready", AsyncMock(return_value=False))

    payload = json.loads(await tool.execute(["snapshot"]))

    assert "shared browser" in payload["error"].lower()
    process.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_session_config_keeps_plain_command(monkeypatch):
    process = _process()
    _prepare(monkeypatch, process)
    tool = AgentBrowserTool()

    payload = json.loads(await tool.execute(["snapshot"]))

    assert payload["command"] == [
        "npx", "--yes", "agent-browser", "--session", "nanobot-isolated-default",
        "--pin-tab", "snapshot",
    ]
    assert payload["browser_session"]["attached"] is False


@pytest.mark.asyncio
async def test_meta_commands_do_not_touch_the_browser_session(monkeypatch):
    process = _process()
    _prepare(monkeypatch, process)
    tool = AgentBrowserTool(cdp_url="http://127.0.0.1:9222")
    probe = AsyncMock(return_value=False)
    launch = AsyncMock(return_value=False)
    monkeypatch.setattr(tool, "_cdp_ready", probe)
    monkeypatch.setattr(tool, "_launch_shared_browser", launch)

    payload = json.loads(await tool.execute(["--help"]))

    probe.assert_not_awaited()
    launch.assert_not_awaited()
    assert payload["command"] == ["npx", "--yes", "agent-browser", "--help"]
    assert payload["browser_session"]["attached"] is False


@pytest.mark.asyncio
async def test_launch_attempt_is_not_repeated_on_every_call(monkeypatch):
    process = _process()
    _prepare(monkeypatch, process)
    tool = AgentBrowserTool(
        cdp_url="http://127.0.0.1:9222",
        launch_command="/home/jaden/.local/bin/chrome-agent",
    )
    monkeypatch.setattr(tool, "_cdp_ready", AsyncMock(return_value=False))
    launch = AsyncMock(return_value=False)
    monkeypatch.setattr(tool, "_launch_shared_browser", launch)

    await tool.execute(["snapshot"])
    await tool.execute(["snapshot"])

    launch.assert_awaited_once()


@pytest.mark.asyncio
async def test_stale_pinned_tab_is_reset_and_retried(monkeypatch):
    calls: list[list[str]] = []

    async def process(command, **kwargs):
        calls.append(list(command))
        failed = len(calls) == 1
        return SimpleNamespace(
            timed_out=False,
            stdout=b"" if failed else b"ok\n",
            stderr=b"tab_gone: bound tab is gone" if failed else b"",
            returncode=1 if failed else 0,
        )

    _prepare(monkeypatch, process)
    tool = AgentBrowserTool(cdp_url="http://127.0.0.1:9222", session="nanobot")
    monkeypatch.setattr(tool, "_cdp_ready", AsyncMock(return_value=True))

    payload = json.loads(await tool.execute(["get", "url"]))

    assert payload["exitCode"] == 0
    assert calls[0][-2:] == ["get", "url"]
    assert calls[1][-3:] == ["tab", "new", "about:blank"]
    assert calls[2][-2:] == ["get", "url"]


@pytest.mark.asyncio
async def test_read_only_stale_tab_never_resets_or_retries(monkeypatch):
    process = _process()
    process.return_value = SimpleNamespace(
        timed_out=False,
        stdout=b'{"error":{"code":"tab_gone","message":"bound tab missing"}}',
        stderr=b"",
        returncode=1,
    )
    _prepare(monkeypatch, process)
    tool = AgentBrowserTool(read_only=True, cdp_url="http://127.0.0.1:9222", session="nanobot")
    monkeypatch.setattr(tool, "_cdp_ready", AsyncMock(return_value=True))

    result = await tool.execute(["--json", "snapshot"])

    assert isinstance(result, str)
    assert "cannot reset" in result.lower()
    process.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        ["--cdp", "http://attacker"],
        ["--cdp=http://attacker"],
        ["--session", "attacker"],
        ["--session=attacker"],
        ["--pin-tab"],
        ["--pin-tab=false"],
        ["--no-pin-tab"],
        ["snapshot", "--no-pin-tab=true"],
        ["--executable-path", "/tmp/chrome"],
        ["--executable-path=/tmp/chrome"],
        ["--cdp", "http://attacker", "snapshot"],
    ],
)
async def test_config_owned_browser_flags_are_rejected_before_cli(monkeypatch, override):
    process = _process()
    _prepare(monkeypatch, process)

    result = await AgentBrowserTool().execute([*override, "snapshot"])

    assert "configuration" in str(result).lower()
    process.assert_not_awaited()


@pytest.mark.asyncio
async def test_isolated_and_attached_modes_use_distinct_named_sessions(monkeypatch):
    calls: list[list[str]] = []

    async def process(command, **kwargs):
        calls.append(list(command))
        return SimpleNamespace(timed_out=False, stdout=b"ok", stderr=b"", returncode=0)

    _prepare(monkeypatch, process)
    tool = AgentBrowserTool(session="team")
    monkeypatch.setattr(tool, "_cdp_ready", AsyncMock(return_value=True))

    await tool.execute(["snapshot"])
    tool.cdp_url = "http://127.0.0.1:9222"
    await tool.execute(["snapshot"])

    isolated_session = calls[0][calls[0].index("--session") + 1]
    attached_session = calls[1][calls[1].index("--session") + 1]
    assert isolated_session == "nanobot-isolated-team"
    assert attached_session == "team"
    assert isolated_session != attached_session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reset_timeout,reset_exit_code",
    [(False, 1), (True, 0)],
)
async def test_json_tab_gone_retries_only_after_successful_reset(
    monkeypatch,
    reset_timeout,
    reset_exit_code,
):
    calls: list[list[str]] = []

    async def process(command, **kwargs):
        calls.append(list(command))
        index = len(calls)
        if index == 1:
            return SimpleNamespace(
                timed_out=False,
                stdout=b'{"error":{"code":"tab_gone"}}',
                stderr=b"",
                returncode=1,
            )
        if index == 2:
            return SimpleNamespace(
                timed_out=reset_timeout,
                stdout=b"reset failed",
                stderr=b"",
                returncode=reset_exit_code,
            )
        return SimpleNamespace(timed_out=False, stdout=b"ok", stderr=b"", returncode=0)

    _prepare(monkeypatch, process)
    tool = AgentBrowserTool(cdp_url="http://127.0.0.1:9222")
    monkeypatch.setattr(tool, "_cdp_ready", AsyncMock(return_value=True))

    payload = json.loads(await tool.execute(["--json", "snapshot"]))

    assert payload["exitCode"] == 1
    assert len(calls) == 2
