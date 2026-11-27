"""Tests for transparent explicit-deny reporting and managed WebUI tool_search visibility."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from kohaku.agent.loop import AgentLoop
from kohaku.agent.tools.base import Tool
from kohaku.agent.tools.registry import ToolDispatchGuard, ToolRegistry
from kohaku.agent.tools.sessions import SearchSessionsTool
from kohaku.agent.turn import TurnRequest
from kohaku.config.schema import Config
from kohaku.context.tool_search import ToolSearchTool
from kohaku.context.tool_surface import ToolSurface, ToolSurfaceState
from kohaku.providers.base import LLMResponse, ToolCallRequest
from tests.agent.p6_admitted_fixtures import AgentInputs, AsyncAdmittedFactory
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async
from tests.agent.test_migration_engine import ScriptedProvider


class _StubDeferredTool(Tool):
    def __init__(self, name: str, description: str, *, namespace: str = "browser") -> None:
        self._name = name
        self._description = description
        self.namespace = namespace
        self.exposure = "deferred"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        _ = kwargs
        return "ok"


class _StaleLeaseGuard(ToolDispatchGuard):
    def __init__(self, blocked_name: str) -> None:
        self._blocked_name = blocked_name

    def error(self, name: str) -> str | None:
        if name == self._blocked_name:
            return f"Tool '{name}' is unavailable after lease expiry"
        return None


def _tool_names(definitions: list[dict[str, Any]]) -> list[str]:
    return [ToolRegistry.schema_name(defn) for defn in definitions]


def _call(name: str, arguments: dict[str, Any], call_id: str) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)],
    )


def _tool_messages(messages: list[dict[str, Any]]) -> list[str]:
    return [
        str(msg.get("content", ""))
        for msg in messages
        if msg.get("role") == "tool"
    ]


def test_tool_search_reports_explicit_deny_without_activating() -> None:
    registry = ToolRegistry()
    registry.register(_StubDeferredTool("read_file", "Read a file", namespace="core"))
    registry.get("read_file").exposure = "direct"  # type: ignore[union-attr]
    registry.register(_StubDeferredTool("agent_browser", "Browse web pages", namespace="browser"))
    registry.register(_StubDeferredTool("desktop_use", "Control desktop", namespace="desktop"))

    surface = ToolSurface(
        registry,
        state=ToolSurfaceState(activated=("agent_browser",)),
        deny_checker=lambda name: name == "agent_browser",
    )
    search_tool = ToolSearchTool(surface)

    result = asyncio.run(search_tool.execute(query="agent_browser"))

    text = str(result)
    assert "No deferred tools matched" not in text
    assert "Blocked by explicit permission deny rule: agent_browser" in text
    assert "Activated 0 tool(s)" in text
    assert "agent_browser" not in surface.visible_names()
    assert "agent_browser" not in _tool_names(surface.definitions_for_provider())

    # Empty-query namespace listing also surfaces the deny note without activating.
    ns_result = asyncio.run(search_tool.execute(query="", namespace="browser"))
    assert "Blocked by explicit permission deny rule: agent_browser" in str(ns_result)
    assert "agent_browser" not in _tool_names(surface.definitions_for_provider())


def test_dispatch_error_filtering_still_hides_stale_lease_tools() -> None:
    registry = ToolRegistry()
    registry.register(_StubDeferredTool("stale_mcp_tool", "Stale MCP operation", namespace="mcp:s"))
    registry.set_dispatch_guard(_StaleLeaseGuard("stale_mcp_tool"))

    surface = ToolSurface(registry, deny_checker=lambda _name: False)
    search_tool = ToolSearchTool(surface)

    result = asyncio.run(search_tool.execute(query="stale_mcp_tool"))

    assert "No deferred tools matched" in str(result)
    assert "Blocked by explicit permission deny rule" not in str(result)


async def _build_managed_webui_loop(
    tmp_path: Path,
    admitted_async: AsyncAdmittedFactory,
    provider: ScriptedProvider,
    *,
    permission_rules: dict[str, Any] | None = None,
) -> AgentLoop:
    config = Config()
    config.tools.agent_browser.enabled = True
    config.tools.image_generation.enabled = False
    if permission_rules:
        config.tools.permissions.rules = permission_rules

    def build(inputs: AgentInputs) -> AgentLoop:
        loop = AgentLoop(
            bus=inputs.bus,
            provider=provider,
            workspace=inputs.config.workspace_path,
            session_manager=inputs.sessions,
            tool_registry=inputs.tools,
            cron_service=inputs.cron,
            turn_delivery_factory=inputs.delivery,
            tools_config=config.tools,
            restrict_to_workspace=True,
        )
        if not loop.tools.has("search_sessions"):
            loop.tools.register(SearchSessionsTool(sessions=loop.sessions))
        return loop

    return await admitted_async(tmp_path, build, config=config)


async def test_managed_webui_turn_exposes_deferred_tools_via_tool_search(
    tmp_path: Path,
    admitted_async: AsyncAdmittedFactory,
) -> None:
    provider = ScriptedProvider([
        _call("tool_search", {"query": "agent_browser", "limit": 1}, "s1"),
        _call("tool_search", {"query": "search_sessions", "limit": 1}, "s2"),
        _call("tool_search", {"query": "my", "limit": 1}, "s3"),
        _call("tool_search", {"query": "create_goal", "limit": 1}, "s4"),
        LLMResponse(content="All tools discovered."),
    ])
    loop = await _build_managed_webui_loop(tmp_path, admitted_async, provider)
    try:
        assert loop.sessions.has_managed_authority
        res = await loop.execute_turn(
            TurnRequest(content="Find tools", session_key="webui:discover-all"),
        )
        assert res.error is None, res.error

        final_tools = set(_tool_names(provider.calls[-1][1]))
        for expected in ("agent_browser", "search_sessions", "my", "create_goal"):
            assert expected in final_tools
    finally:
        await loop.aclose()


async def test_managed_webui_turn_reports_explicit_deny_in_tool_search(
    tmp_path: Path,
    admitted_async: AsyncAdmittedFactory,
) -> None:
    provider = ScriptedProvider([
        _call("tool_search", {"query": "agent_browser", "limit": 1}, "s1"),
        LLMResponse(content="agent_browser is blocked by policy."),
    ])
    loop = await _build_managed_webui_loop(
        tmp_path,
        admitted_async,
        provider,
        permission_rules={"agent_browser": {"*": "deny"}},
    )
    try:
        res = await loop.execute_turn(
            TurnRequest(content="Open browser", session_key="webui:deny-browser"),
        )
        assert res.error is None, res.error

        tool_outputs = _tool_messages(provider.calls[1][0])
        assert any(
            "Blocked by explicit permission deny rule: agent_browser" in out
            and "No deferred tools matched" not in out
            for out in tool_outputs
        ), tool_outputs
        assert "agent_browser" not in _tool_names(provider.calls[1][1])
    finally:
        await loop.aclose()
