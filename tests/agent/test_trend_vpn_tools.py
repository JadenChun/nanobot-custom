from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import nanobot.agent.tools.trend_vpn as trend_vpn_module
from nanobot.agent.tools.trend_vpn import (
    TrendVpnBrowserFetchTool,
    TrendVpnFetchTool,
    TrendVpnSessionCloseTool,
    TrendVpnSessionStartTool,
)


def test_trend_vpn_tools_are_serialized_and_have_narrow_schemas():
    tools = [
        TrendVpnSessionStartTool(),
        TrendVpnFetchTool(),
        TrendVpnBrowserFetchTool(),
        TrendVpnSessionCloseTool(),
    ]
    assert [tool.name for tool in tools] == [
        "trend_vpn_session_start",
        "trend_vpn_fetch",
        "trend_vpn_browser_fetch",
        "trend_vpn_session_close",
    ]
    assert all(tool.supports_parallel_calls is False for tool in tools)
    assert "session_id" in TrendVpnFetchTool.parameters["required"]
    assert "url" in TrendVpnFetchTool.parameters["required"]
    assert TrendVpnFetchTool.parameters["properties"]["url"]["type"] == "string"
    assert TrendVpnFetchTool.parameters["properties"]["max_chars"]["maximum"] == 120000
    assert "session_id" in TrendVpnBrowserFetchTool.parameters["required"]
    assert "url" in TrendVpnBrowserFetchTool.parameters["required"]


@pytest.mark.asyncio
async def test_session_start_cancelled_turn_closes_late_session(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()
    close_calls: list[str] = []

    class FakeClient:
        async def request_async(self, op, **params):
            if op == "start_session":
                started.set()
                await release.wait()
                return {"ok": True, "session_id": "late-session"}
            if op == "close_session":
                close_calls.append(params["session_id"])
                closed.set()
                return {"ok": True}
            raise AssertionError(f"unexpected op: {op}")

    candidate = SimpleNamespace(
        profile_text="profile",
        label="late",
        hostname="vpn.example.com",
    )
    monkeypatch.setattr(trend_vpn_module, "fetch_candidates", lambda *args, **kwargs: [candidate])
    client = FakeClient()
    monkeypatch.setattr(TrendVpnSessionStartTool, "_client", staticmethod(lambda: client))

    tool = TrendVpnSessionStartTool()
    task = asyncio.create_task(tool.execute(ttl_seconds=60))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    release.set()
    await asyncio.wait_for(closed.wait(), timeout=1)
    assert close_calls == ["late-session"]
