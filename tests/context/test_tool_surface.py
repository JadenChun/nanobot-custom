"""WP-4 acceptance at the agent, model, and MCP transport boundaries."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp import types as mcp_types

from kohaku.agent.loop import AgentLoop
from kohaku.agent.tools.base import Tool
from kohaku.agent.tools.cli_apps import CliAppsTool
from kohaku.agent.tools.mcp import MCPProvider, connect_mcp_servers
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.agent.turn import TurnRequest
from kohaku.config.schema import MCPServerConfig
from kohaku.context.budget import TokenMeter
from kohaku.providers.base import LLMResponse, ToolCallRequest
from kohaku.providers.openai_compat_provider import OpenAICompatProvider
from kohaku.providers.registry import find_by_name
from tests.agent.test_migration_engine import ScriptedProvider, engine_config
from tests.agent.test_migration_source_engine import upgrade


class TestCapability(Tool):
    __test__ = False

    def __init__(self, name, exposure="deferred", namespace="bulk"):
        self._name = name
        self.exposure = exposure
        self.namespace = namespace

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self.name

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return self.name


def _call(name, arguments, call_id):
    return LLMResponse(
        content=None, tool_calls=[ToolCallRequest(call_id, name, arguments)],
    )


def _names(definitions):
    return [ToolRegistry.schema_name(tool) for tool in definitions]


def _portable(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("canonical", [False, True])
async def test_discovery_order_prefix_and_reload(engine_config, canonical):
    engine_config.agents.defaults.max_tool_iterations = 10
    registry = ToolRegistry()
    for name in ("direct_a", "direct_b"):
        registry.register(TestCapability(name, "direct"))
    deferred = [f"deferred_{index:02}" for index in range(1, 21)]
    for name in deferred + ["blocked_tool"]:
        registry.register(TestCapability(name))
    admitted = ("direct_a", "direct_b", *deferred)
    provider = ScriptedProvider([
        _call("tool_search", {"query": "deferred_17"}, "exact"),
        _call("tool_search", {"query": "namespace:bulk", "limit": 0}, "low"),
        _call("tool_search", {"query": "namespace:bulk", "limit": 17}, "high"),
        _call("tool_search", {"query": "namespace:bulk"}, "default"),
        _call("tool_search", {"query": "blocked_tool"}, "blocked"),
        LLMResponse(content="Discovery complete."),
        _call("tool_search", {"query": "deferred_17"}, "repeat"),
        LLMResponse(content="Discovery retained."),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=registry, provider=provider)
    key = "cli:surface"
    if canonical:
        upgrade(loop, key)
    try:
        result = await loop.execute_turn(TurnRequest(
            content="Discover tools.", session_key=key, tool_names=admitted,
        ))
        assert result.error is None, result.error
        first = provider.calls[0][1]
        assert _names(first) == ["direct_a", "direct_b", "tool_search"]
        chosen = ["deferred_17", *deferred[:16]]
        assert _names(provider.calls[5][1]) == [*_names(first), *chosen]
        assert all(_portable(call[1][:len(first)]) == _portable(first) for call in provider.calls)
        body = {row["tool_call_id"]: row["content"] for row in provider.calls[5][0]
                if row["role"] == "tool"}
        for label, expected in (("low", 1), ("high", 16), ("default", 8)):
            assert sum(line.startswith("- deferred_") for line in body[label].splitlines()) == expected
        assert "No deferred tools matched" in body["blocked"]
        loop.sessions.invalidate(key)
        followup = await loop.execute_turn(TurnRequest(
            content="Keep these tools.", session_key=key, tool_names=admitted,
        ))
        assert followup.error is None, followup.error
        assert all(_names(call[1]) == [*_names(first), *chosen] for call in provider.calls[6:])
    finally:
        await loop.aclose()


async def test_goal_tools_are_offered_on_creation_and_active_turn(engine_config):
    provider = ScriptedProvider([
        _call("create_goal", {"objective": "Prove discovery.", "ui_summary": "Verification"}, "goal"),
        LLMResponse(content="Goal saved.", finish_reason="refusal"),
        _call("update_goal", {"action": "complete", "recap": "Verified."}, "done"),
        LLMResponse(content="Goal complete."),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    key = "cli:goal"
    upgrade(loop, key)
    try:
        assert (await loop.execute_turn(TurnRequest(
            content="/goal Prove discovery.", session_key=key,
        ))).error is None
        assert (await loop.execute_turn(TurnRequest(
            content="Keep going.", session_key=key,
        ))).error is None
        assert {"create_goal", "update_goal"} <= set(_names(provider.calls[0][1]))
        assert {"create_goal", "update_goal"} <= set(_names(provider.calls[2][1]))
    finally:
        await loop.aclose()


async def test_attached_cli_app_and_mcp_presets(engine_config):
    engine_config.tools.cli_apps.enable = True
    registry = ToolRegistry()
    registry.register(CliAppsTool(workspace=engine_config.workspace_path))
    small = [f"mcp_small_{index}" for index in range(16)]
    large = [f"mcp_large_{index}" for index in range(17)]
    for group, names in (("small", small), ("large", large)):
        for name in names:
            registry.register(TestCapability(name, namespace=f"mcp:{group}"))
    provider = ScriptedProvider([LLMResponse(content="OK")] * 3)
    loop = AgentLoop.from_config(engine_config, tool_registry=registry, provider=provider)
    key = "cli:presets"
    try:
        attached = await loop.execute_turn(TurnRequest(
            content="Use these attachments.", session_key=key,
            metadata={
                "cli_apps": [{"name": "example", "entry_point": "example"}],
                "mcp_presets": [{"name": "small"}, {"name": "large"}],
            },
        ))
        assert attached.error is None, attached.error
        assert {"run_cli_app", *small} <= set(_names(provider.calls[0][1]))
        assert set(large).isdisjoint(_names(provider.calls[0][1]))
        hint = 'MCP preset attached. Use tool_search with namespace="mcp:large".'
        assert hint in str(provider.calls[0][0])
        assert (await loop.execute_turn(TurnRequest(
            content="Continue without an attachment.", session_key=key,
        ))).error is None
        assert hint not in str(provider.calls[1][0])
    finally:
        await loop.aclose()


async def test_responses_hosted_search_requires_explicit_opt_in(engine_config):
    engine_config.agents.defaults.model = "gpt-5.4"
    spec = find_by_name("openai")
    assert spec is not None
    for opted_in in (False, True):
        provider = OpenAICompatProvider(
            api_key="test-key", spec=spec, api_type="responses", default_model="gpt-5.4",
            supports_native_tool_search=opted_in,
        )
        create = AsyncMock(return_value={
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Done."}]}],
        })
        provider._client = SimpleNamespace(responses=SimpleNamespace(create=create))
        registry = ToolRegistry()
        registry.register(TestCapability("direct_a", "direct"))
        registry.register(TestCapability("deferred_a"))
        loop = AgentLoop.from_config(engine_config, tool_registry=registry, provider=provider)
        try:
            response = await loop.execute_turn(TurnRequest(
                content="Hello", session_key=f"cli:hosted-{opted_in}",
                tool_names=("direct_a", "deferred_a"),
            ))
            assert response.error is None, response.error
            tools = create.await_args.kwargs["tools"]
            if opted_in:
                assert {"type": "tool_search"} in tools
                assert next(tool for tool in tools if tool.get("name") == "deferred_a")[
                    "defer_loading"
                ] is True
            else:
                assert _names(tools) == ["direct_a", "tool_search"]
                assert all("defer_loading" not in tool for tool in tools)
        finally:
            await loop.aclose()


def _fake_stdio(monkeypatch, session_factory):
    @asynccontextmanager
    async def stdio_client(params):
        yield params, object()

    @asynccontextmanager
    async def client_session(params, _write):
        yield session_factory(params)

    monkeypatch.setattr("mcp.client.stdio.stdio_client", stdio_client)
    monkeypatch.setattr("mcp.ClientSession", client_session)


class Catalogue:
    def __init__(self):
        self.acquisitions = []
        self.failures = {}

    async def initialize(self):
        return SimpleNamespace(instructions="Use stable catalogue identifiers.")

    @staticmethod
    def page(rows, key, cursor):
        half = len(rows) // 2
        assert cursor in (None, "opaque next page")
        return SimpleNamespace(**{
            key: rows[:half] if cursor is None else rows[half:],
            "nextCursor": "opaque next page" if cursor is None else None,
        })

    async def list_tools(self, cursor=None):
        schema = {
            "type": "object",
            "$defs": {"label": {"type": "string", "pattern": "^label:"}},
            "properties": {
                "key": {"oneOf": [{"$ref": "#/$defs/label"}, {"type": "integer"}]},
                "email": {"type": "string", "format": "email"},
            },
        }
        names = [f"read_entry_{index}" for index in range(78)] + ["read collision", "read_collision"]
        return self.page([mcp_types.Tool(name=name, inputSchema=schema) for name in names],
                         "tools", cursor)

    async def list_resources(self, cursor=None):
        return self.page([
            mcp_types.Resource(name=f"Resource {index}", uri=f"file:///entry-{index}")
            for index in range(50)
        ], "resources", cursor)

    async def list_prompts(self, cursor=None):
        return self.page([
            mcp_types.Prompt(
                name=f"prompt_{index}",
                arguments=[mcp_types.PromptArgument(name="key", required=True)],
            )
            for index in range(20)
        ], "prompts", cursor)

    async def read_resource(self, uri):
        if str(uri) in self.failures:
            raise self.failures[str(uri)]
        return mcp_types.ReadResourceResult(contents=[
            mcp_types.TextResourceContents(uri=uri, text=f"Read {uri}"),
        ])

    async def get_prompt(self, name, arguments):
        return mcp_types.GetPromptResult(messages=[mcp_types.PromptMessage(
            role="user", content=mcp_types.TextContent(
                type="text", text=f"Got {name} with {arguments['key']}",
            ),
        )])

    async def call_tool(self, name, arguments, **kwargs):
        self.acquisitions.append((name, arguments))
        return mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text=name)])


async def test_paginated_mcp_catalogue_and_collision_budget(engine_config, monkeypatch, record_property):
    engine_config.agents.defaults.max_tool_iterations = 12
    catalogue = Catalogue()
    _fake_stdio(monkeypatch, lambda _params: catalogue)
    provider = ScriptedProvider([LLMResponse(content="OK")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    owner = MCPProvider({"catalogue": MCPServerConfig(command="mock")}, loop.tools)
    try:
        assert (await loop.execute_turn(TurnRequest(content="Hello", session_key="cli:no-mcp"))).error is None
        await owner.connect()
        assert owner.connected_server_names == {"catalogue"}
        round_trips = [
            _call("tool_search", {"query": "read_entry_77", "limit": 1}, "second-page"),
            _call("tool_search", {"query": "collision", "namespace": "mcp:catalogue", "limit": 2}, "collisions"),
            _call("tool_search", {"query": "namespace:mcp", "limit": 2}, "extras"),
            _call("mcp_resources", {"action": "list", "server": "catalogue"}, "resources"),
            _call("mcp_prompts", {"action": "list", "server": "catalogue"}, "prompts"),
            _call("mcp_resources", {
                "action": "read", "server": "catalogue", "uri": "file:///entry-49",
            }, "resource"),
            _call("mcp_prompts", {
                "action": "get", "server": "catalogue", "name": "prompt_19", "arguments": {"key": "label"},
            }, "prompt"),
            LLMResponse(content="Finished."),
        ]
        provider.responses.extend(round_trips)
        key = "cli:large-mcp"
        result = await loop.execute_turn(TurnRequest(content="Inspect the catalogue.", session_key=key))
        assert result.error is None, result.error
        empty_prompt, empty_tools, _ = provider.calls[0]
        mcp_prompt, mcp_tools, _ = provider.calls[1]
        increase = TokenMeter.count_text(_portable(mcp_tools)) - TokenMeter.count_text(_portable(empty_tools))
        increase += TokenMeter.count_text(mcp_prompt[0]["content"]) - TokenMeter.count_text(empty_prompt[0]["content"])
        record_property("mcp_turn_1_added_tokens", increase)
        assert 0 < increase <= 300
        assert _portable(empty_tools) == _portable(mcp_tools)
        assert "Use stable catalogue identifiers." in mcp_prompt[0]["content"]
        assert "mcp_catalogue_read_entry_77" in _names(provider.calls[2][1])
        eventual = provider.calls[-1][1]
        collisions = [tool["function"] for tool in eventual if "collision" in ToolRegistry.schema_name(tool)]
        assert len(collisions) == 2 and len({tool["name"] for tool in collisions}) == 2
        for tool in collisions:
            assert len(tool["name"]) <= 64
            assert tool["parameters"]["$defs"]["label"]["pattern"] == "^label:"
            assert tool["parameters"]["properties"]["email"]["format"] == "email"
            assert len(tool["parameters"]["properties"]["key"]["oneOf"]) == 2
        assert {"mcp_resources", "mcp_prompts"} <= set(_names(eventual))
        results = {row["tool_call_id"]: row["content"] for row in provider.calls[-1][0]
                   if row["role"] == "tool"}
        assert "file:///entry-49" in results["resources"] and "file:///entry-0" in results["resources"]
        assert "prompt_19" in results["prompts"] and "prompt_0" in results["prompts"]
        assert results["resource"] == "Read file:///entry-49"
        assert results["prompt"] == "Got prompt_19 with label"
        for tool in collisions:
            await loop.tools.execute(tool["name"], {"key": "label:value"})
        assert {original for original, args in catalogue.acquisitions} == {"read collision", "read_collision"}
        for index, failure in enumerate((TimeoutError(), RuntimeError("MCP server error"))):
            uri = f"file:///entry-{index}"
            catalogue.failures[uri] = failure
            response = await loop.tools.execute("mcp_resources", {
                "action": "read", "server": "catalogue", "uri": uri,
            })
            assert response.is_error
    finally:
        await owner.aclose()
        await loop.aclose()


async def test_three_mcp_startup_attempts_overlap(monkeypatch):
    attempted = set()
    all_started = asyncio.Event()
    release = asyncio.Event()

    @asynccontextmanager
    async def unavailable(params):
        attempted.add(params.command)
        if len(attempted) == 3:
            all_started.set()
        await release.wait()
        raise ConnectionError("Unavailable MCP test server.")
        yield

    monkeypatch.setattr("mcp.client.stdio.stdio_client", unavailable)
    task = asyncio.create_task(connect_mcp_servers(
        {name: MCPServerConfig(command=name) for name in ("a", "b", "c")}, ToolRegistry(),
    ))
    try:
        await asyncio.wait_for(all_started.wait(), timeout=1)
        assert attempted == {"a", "b", "c"}
    finally:
        release.set()
        assert await asyncio.wait_for(task, timeout=2) == {}


async def test_sampled_mcp_call_uses_equivalent_same_turn_reconnect(engine_config, monkeypatch):
    names = {"read_health", "read_status"}

    class Session:
        def __init__(self, replacement):
            self.replacement = replacement

        async def initialize(self):
            return None

        async def list_tools(self):
            return SimpleNamespace(tools=[
                mcp_types.Tool(name=name, inputSchema={"type": "object", "properties": {}})
                for name in sorted(names)
            ])

        async def call_tool(self, name, arguments, **kwargs):
            if not self.replacement and name == "read_health":
                raise ConnectionError("connection closed")
            label = "replacement" if self.replacement else "original"
            return mcp_types.CallToolResult(content=[mcp_types.TextContent(
                type="text", text=f"{label} {name}",
            )])

    sessions = iter([Session(False), Session(True)])
    _fake_stdio(monkeypatch, lambda _params: next(sessions))
    offered = asyncio.Event()
    release = asyncio.Event()

    class BlockingProvider(ScriptedProvider):
        async def chat(self, messages, tools=None, model=None, **kwargs):
            if tools and "mcp_health_read_status" in _names(tools) and not offered.is_set():
                offered.set()
                await release.wait()
            return await super().chat(messages, deepcopy(tools), model, **kwargs)

    provider = BlockingProvider([
        _call("tool_search", {"query": "mcp_health_read_status", "limit": 1}, "discover"),
        _call("mcp_health_read_status", {}, "status"),
        LLMResponse(content="Status checked."),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    owner = MCPProvider({"health": MCPServerConfig(command="mock")}, loop.tools)
    await owner.connect()
    task = asyncio.create_task(loop.execute_turn(TurnRequest(content="Check status.", session_key="cli:health")))
    try:
        await asyncio.wait_for(offered.wait(), timeout=2)
        failure = await loop.tools.execute("mcp_health_read_health", {})
        assert failure.is_error
        release.set()
        assert (await asyncio.wait_for(task, timeout=2)).content == "Status checked."
        assert "replacement read_status" in str(provider.calls[-1][0])
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await owner.aclose()
        await loop.aclose()


async def test_exec_session_and_message_defer_and_auto_activate(engine_config):
    from kohaku.agent.tools.base import ToolResult

    class _YieldingExec(Tool):
        exposure = "direct"
        namespace = "core"
        @property
        def name(self): return "exec"
        @property
        def description(self): return "Run shell."
        @property
        def parameters(self): return {"type": "object", "properties": {}}
        async def execute(self, **kwargs): return ToolResult("Process running. session_id: s1\nElapsed: 0.1s")

    class _FakeGenImage(Tool):
        exposure = "deferred"
        namespace = "media"
        @property
        def name(self): return "generate_image"
        @property
        def description(self): return "Gen image."
        @property
        def parameters(self): return {"type": "object", "properties": {}}
        async def execute(self, **kwargs): return ToolResult("Saved image.png")

    provider = ScriptedProvider([
        _call("exec", {}, "e1"),
        _call("generate_image", {}, "g1"),
        LLMResponse(content="Done."),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        assert "search_sessions" not in loop.tools.tool_names
        edit_tool = loop.tools.get("edit_file")
        assert edit_tool is not None and "apply_patch" not in edit_tool.description
        loop.tools.replace(_YieldingExec())
        loop.tools.replace(_FakeGenImage())
        res = await loop.execute_turn(TurnRequest(content="Run and generate", session_key="cli:auto-act"))
        assert res.error is None, res.error
        turn0 = _names(provider.calls[0][1])
        turn1 = _names(provider.calls[1][1])
        turn2 = _names(provider.calls[2][1])
        assert "exec_session" not in turn0 and "message" not in turn0
        assert "exec_session" in turn1 and "list_exec_sessions" in turn1
        assert "message" in turn2
    finally:
        await loop.aclose()
