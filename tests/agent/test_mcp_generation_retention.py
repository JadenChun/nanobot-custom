"""Regression coverage for MCP generation retention (A3-6 AC6.1)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from mcp.types import TextContent

from nanobot.agent.tools.mcp import MCPProvider, MCPToolWrapper
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import MCPServerConfig

TOOL_NAME = "mcp_probe_probe"

TOOL_DEF = SimpleNamespace(
    name="probe",
    description="probe",
    inputSchema={"type": "object", "properties": {}},
)


class FakeSession:
    def __init__(self, result: object | None = None) -> None:
        if result is None:
            result = SimpleNamespace(content=[], isError=False)
        self.result = result
        self.calls: list[tuple[str, object]] = []

    async def call_tool(self, name: str, arguments: object = None):
        self.calls.append((name, arguments))
        return self.result


class GatedSession:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def call_tool(self, name: str, arguments: object = None):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return SimpleNamespace(content=[], isError=False)


class FakeConnection:
    def __init__(
        self,
        label: str,
        closed: list[str] | None = None,
        closed_event: asyncio.Event | None = None,
    ) -> None:
        self.label = label
        self.closed = closed if closed is not None else []
        self.closed_event = closed_event
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        self.closed.append(self.label)
        if self.closed_event is not None:
            self.closed_event.set()


def _config(command: str) -> dict[str, MCPServerConfig]:
    return {"probe": MCPServerConfig(type="stdio", command=command)}


def _text_session(text: str) -> FakeSession:
    return FakeSession(
        SimpleNamespace(content=[TextContent(type="text", text=text)], isError=False)
    )


@pytest.mark.asyncio
async def test_failed_replacement_retains_old_generation() -> None:
    registry = ToolRegistry()
    config = _config("one")
    provider = MCPProvider(config, registry, server_loader=lambda: dict(config))
    closed: list[str] = []
    closed_event = asyncio.Event()
    connection = FakeConnection("probe", closed, closed_event)
    session_ok = FakeSession()
    attempts = 0

    async def fake_connect(servers, registry, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            registry.register(MCPToolWrapper(session_ok, "probe", TOOL_DEF))
            return {"probe": connection}
        return {}

    try:
        with patch("nanobot.agent.tools.mcp.connect_mcp_servers", new=fake_connect):
            await provider.connect()
            old_tool = registry.get(TOOL_NAME)
            assert old_tool is not None

            config["probe"] = MCPServerConfig(type="stdio", command="two")
            result = await provider.reload()

            assert registry.get(TOOL_NAME) is old_tool
            assert result["ok"] is False
            assert result["failed"] == ["probe"]
            assert provider.connected_server_names == {"probe"}
            await asyncio.sleep(0.05)
            assert closed == []
            assert closed_event.is_set() is False

            await old_tool.execute()
            assert [name for name, _ in session_ok.calls] == ["probe"]
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_old_generation_closes_once_after_admitted_call_settles() -> None:
    registry = ToolRegistry()
    config = _config("one")
    provider = MCPProvider(config, registry, server_loader=lambda: dict(config))
    closed: list[str] = []
    closed_event = asyncio.Event()
    connection = FakeConnection("probe", closed, closed_event)
    session = GatedSession()

    async def fake_connect(servers, registry, **kwargs):
        registry.register(MCPToolWrapper(session, "probe", TOOL_DEF))
        return {"probe": connection}

    try:
        with patch("nanobot.agent.tools.mcp.connect_mcp_servers", new=fake_connect):
            await provider.connect()
            wrapper = registry.get(TOOL_NAME)
            assert wrapper is not None

            task = asyncio.create_task(wrapper.execute())
            await asyncio.wait_for(session.started.wait(), timeout=1)

            await provider._close_server("probe")
            assert closed == []

            session.release.set()
            await task
            await asyncio.wait_for(closed_event.wait(), timeout=1)
            assert closed == ["probe"]

            await provider._close_server("probe")
            assert closed == ["probe"]
    finally:
        session.release.set()
        await provider.aclose()


@pytest.mark.asyncio
async def test_ready_replacement_swaps_tools_and_retires_old_generation() -> None:
    registry = ToolRegistry()
    config = _config("one")
    provider = MCPProvider(config, registry, server_loader=lambda: dict(config))
    closed: list[str] = []
    old_closed_event = asyncio.Event()
    old_connection = FakeConnection("probe-old", closed, old_closed_event)
    new_connection = FakeConnection("probe-new", closed)
    attempts = 0
    replacement: dict[str, MCPToolWrapper] = {}

    async def fake_connect(servers, registry, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            registry.register(MCPToolWrapper(_text_session("old"), "probe", TOOL_DEF))
            return {"probe": old_connection}
        new_tool = MCPToolWrapper(_text_session("new"), "probe", TOOL_DEF)
        replacement["tool"] = new_tool
        registry.register(new_tool)
        return {"probe": new_connection}

    try:
        with patch("nanobot.agent.tools.mcp.connect_mcp_servers", new=fake_connect):
            await provider.connect()
            old_tool = registry.get(TOOL_NAME)
            assert old_tool is not None

            config["probe"] = MCPServerConfig(type="stdio", command="two")
            result = await provider.reload()

            assert result["ok"] is True
            assert result["changed"] == ["probe"]
            new_tool = registry.get(TOOL_NAME)
            assert new_tool is not old_tool
            assert new_tool is replacement["tool"]
            assert await new_tool.execute() == "new"
            await asyncio.wait_for(old_closed_event.wait(), timeout=1)
            assert closed == ["probe-old"]
    finally:
        await provider.aclose()
