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
    assert "--session" in payload["command"]
    assert payload["command"][-2:] == ["agent", "snapshot"]
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
async def test_unreachable_session_without_launch_uses_isolated_profile(monkeypatch):
    process = _process()
    _prepare(monkeypatch, process)
    tool = AgentBrowserTool(
        cdp_url="http://127.0.0.1:9222",
        executable_path="/opt/chrome",
    )
    monkeypatch.setattr(tool, "_cdp_ready", AsyncMock(return_value=False))

    payload = json.loads(await tool.execute(["snapshot"]))

    assert "--cdp" not in payload["command"]
    assert "--executable-path" in payload["command"]
    assert "/opt/chrome" in payload["command"]
    assert payload["browser_session"]["attached"] is False


@pytest.mark.asyncio
async def test_no_session_config_keeps_plain_command(monkeypatch):
    process = _process()
    _prepare(monkeypatch, process)
    tool = AgentBrowserTool()

    payload = json.loads(await tool.execute(["snapshot"]))

    assert payload["command"] == ["npx", "--yes", "agent-browser", "snapshot"]
    assert payload["browser_session"]["attached"] is False
