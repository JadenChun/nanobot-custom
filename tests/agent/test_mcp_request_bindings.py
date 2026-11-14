"""Request-owned MCP definitions, dispatch identity and connection lifetime."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from mcp.types import TextContent

from nanobot.agent.hook import AgentHook
from nanobot.agent.policy import RiskyActionPolicy
from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.filesystem import WriteFileTool
from nanobot.agent.tools.mcp import MCPProvider, MCPToolWrapper
from nanobot.agent.tools.registry import CapturedToolBindings, ToolRegistry
from nanobot.agent.turn import TurnContext
from nanobot.config.schema import MCPServerConfig
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.request_budget import ProviderRouteChanged
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import JsonlSessionStore
from nanobot.session.manager import Session as CanonicalSession
from nanobot.session.turn_writer import SourceTurnWriter
from tests.agent.runner_helpers import make_run_spec

TOOL_NAME = "mcp_probe_probe"


class Session:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[dict[str, object]] = []
        self.block = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def call_tool(self, name: str, arguments: dict[str, object]):
        assert name == "probe"
        self.calls.append(arguments)
        self.started.set()
        if self.block:
            await self.release.wait()
        return SimpleNamespace(
            content=[TextContent(type="text", text=self.label)], isError=False,
        )


class Connection:
    def __init__(self) -> None:
        self.close_calls = 0
        self.closed = asyncio.Event()

    async def aclose(self) -> None:
        self.close_calls += 1
        self.closed.set()


class OfflineProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(provider_name="offline")
        self.responses = iter(responses)
        self.definitions: list[object] = []
        self.sampled = asyncio.Event()
        self.release = asyncio.Event()

    def get_default_model(self) -> str:
        return "offline"

    async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        self.definitions.append(deepcopy(tools))
        if len(self.definitions) == 1:
            self.sampled.set()
            await self.release.wait()
        return next(self.responses)


def tool_response() -> LLMResponse:
    return LLMResponse(
        content=None, finish_reason="tool_calls",
        tool_calls=[ToolCallRequest(id="probe-call", name=TOOL_NAME, arguments={"value": "7"})],
    )


def run_spec(provider: LLMProvider, registry: ToolRegistry, **kwargs):
    return make_run_spec(
        provider, initial_messages=[{"role": "user", "content": "probe"}],
        tools=registry, model="offline", max_iterations=2,
        max_tool_result_chars=1000, finalize_on_max_iterations=False, **kwargs,
    )


class MCPFixture:
    def __init__(self) -> None:
        self.registry = ToolRegistry()
        self.config = {"probe": MCPServerConfig(type="stdio", command="old")}
        self.owner = MCPProvider(self.config, self.registry, server_loader=lambda: dict(self.config))
        self.sessions = [Session("old"), Session("replacement")]
        self.connections = [Connection(), Connection()]
        self.tools: list[MCPToolWrapper] = []
        self.fail_replacement = False

    async def connect(self, servers, registry, **kwargs):
        index = len(self.tools)
        if index and self.fail_replacement:
            return {}
        schema = {"type": "object", "properties": {
            "value": {"type": "integer" if index == 0 else "string"},
        }, "required": ["value"]}
        tool = MCPToolWrapper(self.sessions[index], "probe", SimpleNamespace(
            name="probe", description="old" if index == 0 else "replacement", inputSchema=schema,
        ))
        self.tools.append(tool)
        if servers["probe"].enabled_tools:
            registry.register(tool)
        return {"probe": self.connections[index]}

    async def reload(self) -> dict[str, object]:
        self.config["probe"] = MCPServerConfig(type="stdio", command="replacement")
        return await self.owner.reload()


@pytest.fixture
async def connected_mcp():
    fixture = MCPFixture()
    with patch("nanobot.agent.tools.mcp.connect_mcp_servers", new=fixture.connect):
        await fixture.owner.connect()
        try:
            yield fixture
        finally:
            await fixture.owner.aclose()


@pytest.mark.parametrize("view", ["root", "filtered", "artifact_isolated"])
async def test_runner_never_dispatches_replacement_for_sampled_schema(view: str) -> None:
    fixture = MCPFixture()
    provider = OfflineProvider([tool_response(), LLMResponse(content="done")])
    task = None
    with patch("nanobot.agent.tools.mcp.connect_mcp_servers", new=fixture.connect):
        try:
            await fixture.owner.connect()
            registry = (fixture.registry.filtered([TOOL_NAME]) if view == "filtered"
                        else fixture.registry)
            options = {"artifact_reader_factory": lambda *_args: None} if view == "artifact_isolated" else {}
            task = asyncio.create_task(AgentRunner().run(run_spec(provider, registry, **options)))
            await asyncio.wait_for(provider.sampled.wait(), 1)
            assert provider.definitions[0][0]["function"]["description"] == "old"
            lease = fixture.tools[0]._generation_lease
            assert lease._sampled == 1 and lease._inflight == 0
            await fixture.reload()
            assert fixture.connections[0].close_calls == 0
            assert lease in fixture.owner._retired_generations
            provider.release.set()
            result = await task
            assert fixture.sessions[1].calls == []
            assert fixture.sessions[0].calls == []
            assert any("refused" in str(message.get("content")) for message in result.messages)
            await asyncio.wait_for(fixture.connections[0].closed.wait(), 1)
            assert lease._sampled == lease._inflight == 0
            assert fixture.connections[0].close_calls == 1
            later = OfflineProvider([tool_response(), LLMResponse(content="later")])
            later.release.set()
            await AgentRunner().run(run_spec(later, fixture.registry))
            assert later.definitions[0][0]["function"]["description"] == "replacement"
            assert fixture.sessions[1].calls == [{"value": "7"}]
        finally:
            provider.release.set()
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await fixture.owner.aclose()


async def test_failed_reload_keeps_sampled_registration_usable(connected_mcp) -> None:
    fixture = connected_mcp
    fixture.fail_replacement = True
    provider = OfflineProvider([tool_response(), LLMResponse(content="done")])
    task = asyncio.create_task(AgentRunner().run(run_spec(provider, fixture.registry)))
    try:
        await asyncio.wait_for(provider.sampled.wait(), 1)
        lease = fixture.tools[0]._generation_lease
        assert (await fixture.reload())["ok"] is False
        assert fixture.registry.get(TOOL_NAME) is fixture.tools[0]
        assert fixture.connections[0].close_calls == 0
        provider.release.set()
        await task
        assert fixture.sessions[0].calls == [{"value": 7}]
        assert fixture.sessions[1].calls == []
        assert lease._sampled == lease._inflight == 0
        assert fixture.connections[0].close_calls == 0
    finally:
        provider.release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_multi_server_reload_owns_each_visible_replacement_until_settlement() -> None:
    registry = ToolRegistry()
    config = {name: MCPServerConfig(type="stdio", command="old") for name in ("alpha", "beta")}
    owner = MCPProvider(config, registry, server_loader=lambda: dict(config))
    retirement_entered, retirement_release = asyncio.Event(), asyncio.Event()
    sessions: dict[tuple[str, str], Session] = {}
    connections: dict[tuple[str, str], Connection] = {}
    wrappers: dict[tuple[str, str], MCPToolWrapper] = {}

    class ReloadConnection(Connection):
        def __init__(self, phase: str) -> None:
            super().__init__()
            self.phase = phase

        async def aclose(self) -> None:
            self.close_calls += 1
            if self.phase == "old" and any(
                tool.description == "new" for tool in registry.iter_tools()
            ):
                retirement_entered.set()
                await retirement_release.wait()
            self.closed.set()

    async def connect(servers, candidates, **kwargs):
        result = {}
        for name, cfg in servers.items():
            key = (cfg.command, name)
            session = sessions[key] = Session(cfg.command)
            tool = wrappers[key] = MCPToolWrapper(session, name, SimpleNamespace(
                name="probe", description=cfg.command,
                inputSchema={"type": "object", "properties": {"value": {"type": "string"}}},
            ))
            candidates.register(tool)
            result[name] = connections[key] = ReloadConnection(cfg.command)
        return result

    capture = None
    provider = None
    reload_task = runner_task = active_call = closing = None
    with patch("nanobot.agent.tools.mcp.connect_mcp_servers", new=connect):
        try:
            await owner.connect()
            config.update({name: MCPServerConfig(type="stdio", command="new") for name in config})
            reload_task = asyncio.create_task(owner.reload())
            await asyncio.wait_for(retirement_entered.wait(), 1)
            capture = registry.capture_request_tools()
            sampled, = capture.iter_tools()
            first = next(name for (phase, name), tool in wrappers.items()
                         if phase == "new" and tool is sampled)
            second, = set(config) - {first}
            provider = OfflineProvider([
                LLMResponse(content=None, finish_reason="tool_calls", tool_calls=[
                    ToolCallRequest(id="sampled", name=sampled.name, arguments={"value": "7"}),
                ]),
                LLMResponse(content="done"),
            ])
            runner_task = asyncio.create_task(AgentRunner().run(run_spec(provider, registry)))
            await asyncio.wait_for(provider.sampled.wait(), 1)
            assert provider.definitions[0][0]["function"]["description"] == "new"
            lease = sampled.generation_lease
            assert lease is not None, "a visible replacement must already own its generation"
            assert capture._leases == (lease,)
            assert lease._sampled == 2 and lease._inflight == 0
            sessions["new", first].block = True
            active_call = asyncio.create_task(capture.execute(sampled.name, {"value": "7"}))
            await asyncio.wait_for(sessions["new", first].started.wait(), 1)
            assert lease._sampled == 2 and lease._inflight == 1

            retirement_release.set()
            assert (await reload_task)["ok"] is True
            assert set(registry.tool_names) == {f"mcp_{name}_probe" for name in config}
            assert sampled.generation_lease is lease
            assert lease._sampled == 2 and lease._inflight == 1
            assert all(wrappers["new", name].generation_lease is not None for name in config)
            assert all(connections["old", name].close_calls == 1 for name in config)

            config[first] = MCPServerConfig(type="stdio", command="later")
            assert (await owner.reload())["ok"] is True
            assert lease in owner._retired_generations
            assert connections["new", first].close_calls == 0
            assert "refused" in str(await capture.execute(sampled.name, {"value": "8"}))
            provider.release.set()
            result = await runner_task
            assert any("refused" in str(message.get("content")) for message in result.messages)
            assert lease._sampled == 1 and lease._inflight == 1
            assert sessions["new", first].calls == [{"value": "7"}]
            assert sessions["later", first].calls == []

            later = OfflineProvider([
                LLMResponse(content=None, finish_reason="tool_calls", tool_calls=[
                    ToolCallRequest(id="later", name=sampled.name, arguments={"value": "9"}),
                ]),
                LLMResponse(content="later done"),
            ])
            later.release.set()
            await AgentRunner().run(run_spec(later, registry))
            assert sessions["later", first].calls == [{"value": "9"}]
            assert {definition["function"]["description"]
                    for definition in later.definitions[0]} == {"new", "later"}

            closing = asyncio.create_task(owner.aclose())
            await asyncio.wait_for(connections["new", second].closed.wait(), 1)
            await asyncio.wait_for(connections["later", first].closed.wait(), 1)
            assert not closing.done()
            assert connections["new", first].close_calls == 0
            assert "refused" in str(await capture.execute(sampled.name, {"value": "10"}))
            capture.release()
            capture.release()
            assert lease._sampled == 0 and lease._inflight == 1
            assert connections["new", first].close_calls == 0
            sessions["new", first].release.set()
            assert await active_call == "new"
            await asyncio.wait_for(closing, 1)
            assert lease._sampled == lease._inflight == 0
            assert owner._retired_generations == set()
            assert all(connection.close_calls == 1 for connection in connections.values())
            assert all(sessions["old", name].calls == [] for name in config)
            assert sessions["new", second].calls == []
        finally:
            retirement_release.set()
            if provider is not None:
                provider.release.set()
            for session in sessions.values():
                session.release.set()
            for task in (reload_task, runner_task, active_call):
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)
            if capture is not None:
                capture.release()
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            await owner.aclose()


class PreparedBarrier(AgentHook):
    def __init__(self) -> None:
        self.prepared = asyncio.Event()
        self.release = asyncio.Event()
        self.tools: list[object] = []
        self.arguments: list[object] = []

    async def before_execute_tool(self, context, call, tool, params) -> None:
        self.tools.append(tool)
        self.arguments.append(deepcopy(params))
        self.prepared.set()
        await self.release.wait()


class StreamingHook(AgentHook):
    def wants_streaming(self) -> bool:
        return True


@pytest.mark.parametrize("mode", ["prepared_route", "stream_retry", "malformed"])
async def test_logical_request_retries_keep_the_sampled_definitions(
    connected_mcp, mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = connected_mcp
    first_response = (
        LLMResponse(content=None, finish_reason="error", error_status_code=429,
                    error_kind="rate_limit")
        if mode == "stream_retry" else
        LLMResponse(content=None, finish_reason="tool_calls", tool_calls=[
            ToolCallRequest(id="bad", name="", arguments={}),
        ])
    )
    provider = OfflineProvider([first_response, tool_response(), LLMResponse(content="done")])
    if mode == "stream_retry":
        monkeypatch.setattr(provider, "_CHAT_RETRY_DELAYS", (0,))
    elif mode == "prepared_route":
        async def route_request(*, messages, tools, **kwargs):
            provider.definitions.append(deepcopy(tools))
            if len(provider.definitions) == 1:
                provider.sampled.set()
                await provider.release.wait()
                tools[0]["function"]["description"] = "caller mutation"
                raise ProviderRouteChanged(
                    expected="primary", actual="fallback", actual_fallback_index=1,
                    consumed_through=0,
                )
            return tool_response() if len(provider.definitions) == 2 else LLMResponse(content="done")

        monkeypatch.setattr(provider, "chat_with_retry", route_request)
    task = asyncio.create_task(AgentRunner().run(run_spec(
        provider, fixture.registry, hook=StreamingHook() if mode == "stream_retry" else None,
    )))
    try:
        await asyncio.wait_for(provider.sampled.wait(), 1)
        lease = fixture.tools[0]._generation_lease
        assert lease._sampled == 1
        original = deepcopy(provider.definitions[0])
        await fixture.reload()
        provider.release.set()
        await task
        assert provider.definitions[0] == provider.definitions[1] == original
        assert provider.definitions[2][0]["function"]["description"] == "replacement"
        assert fixture.sessions[0].calls == fixture.sessions[1].calls == []
        assert lease._sampled == lease._inflight == 0
        await asyncio.wait_for(fixture.connections[0].closed.wait(), 1)
        assert fixture.connections[0].close_calls == 1
    finally:
        provider.release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("when", ["provider", "prepared"])
@pytest.mark.parametrize("revocation", ["remove", "disable", "oauth"])
async def test_live_revocation_refuses_sampled_and_prepared_calls(
    when: str, revocation: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = MCPFixture()
    authorized = True
    if revocation == "oauth":
        fixture.config["probe"] = MCPServerConfig(
            type="streamableHttp", auth="oauth", url="https://probe.example/mcp",
        )
        fixture.owner._servers = dict(fixture.config)
        monkeypatch.setattr(
            "nanobot.agent.tools.mcp_oauth.mcp_oauth_has_credentials",
            lambda *_args: authorized,
        )
    provider = OfflineProvider([tool_response(), LLMResponse(content="done")])
    hook = PreparedBarrier()
    if when == "prepared":
        provider.release.set()
    task = None
    with patch("nanobot.agent.tools.mcp.connect_mcp_servers", new=fixture.connect):
        try:
            await fixture.owner.connect()
            task = asyncio.create_task(AgentRunner().run(run_spec(
                provider, fixture.registry, hook=hook if when == "prepared" else None,
            )))
            barrier = hook.prepared if when == "prepared" else provider.sampled
            await asyncio.wait_for(barrier.wait(), 1)
            lease = fixture.tools[0]._generation_lease
            assert lease._sampled == 1
            if when == "prepared":
                assert hook.tools == [fixture.tools[0]]
                assert hook.arguments == [{"value": 7}]
            if revocation == "remove":
                fixture.config.clear()
                await fixture.owner.reload()
            elif revocation == "disable":
                fixture.config["probe"] = MCPServerConfig(type="stdio", command="old", enabled_tools=[])
                await fixture.owner.reload()
            else:
                # No reload is needed: lost authorization is checked live.
                authorized = False
            assert fixture.connections[0].close_calls == 0
            hook.release.set()
            provider.release.set()
            result = await task
            assert fixture.sessions[0].calls == fixture.sessions[1].calls == []
            assert any("refused" in str(message.get("content")) for message in result.messages)
            assert lease._sampled == lease._inflight == 0
            if revocation == "oauth":
                authorized = True
                assert "refused" in str(await fixture.tools[0].execute(value=7))
                assert fixture.sessions[0].calls == []
                authorized = False
                await fixture.owner.reload()
                assert fixture.registry.get(TOOL_NAME) is None
            await asyncio.wait_for(fixture.connections[0].closed.wait(), 1)
            assert fixture.connections[0].close_calls == 1
        finally:
            hook.release.set()
            provider.release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            await fixture.owner.aclose()


async def test_capture_detaches_definitions_and_parameter_preparation(connected_mcp) -> None:
    fixture = connected_mcp
    capture = fixture.registry.capture_request_tools()
    try:
        assert isinstance(capture, CapturedToolBindings)
        original = capture.get_definitions()
        caller_copy = capture.get_definitions()
        caller_copy[0]["function"]["parameters"]["properties"]["value"]["type"] = "string"
        registry_copy = fixture.registry.get_definitions()
        registry_copy[0]["function"]["description"] = "changed"
        fixture.tools[0].parameters["properties"]["value"]["type"] = "string"
        assert capture.get_definitions() == original
        tool, arguments, error = capture.prepare_call(TOOL_NAME, '{"arguments":{"value":"7"}}')
        assert tool is fixture.tools[0]
        assert arguments == {"value": 7}
        assert error is None
        assert fixture.tools[0]._generation_lease._sampled == 1
    finally:
        capture.release()
        capture.release()
    assert fixture.tools[0]._generation_lease._sampled == 0


async def test_borrow_has_independent_narrowed_lifetime_and_live_revocation(connected_mcp) -> None:
    fixture = connected_mcp
    outside = MCPToolWrapper(fixture.sessions[0], "probe", SimpleNamespace(
        name="outside", description="outside", inputSchema={"type": "object", "properties": {}},
    ))
    lease = fixture.tools[0]._generation_lease
    outside.bind_generation(lease)
    fixture.registry.register(outside)
    parent = fixture.registry.capture_request_tools()
    child = parent.borrow_filtered([TOOL_NAME, "missing"])
    try:
        assert lease._sampled == 2
        assert child.tool_names == [TOOL_NAME]
        parent.release()
        parent.release()
        assert lease._sampled == 1
        assert await child.execute(TOOL_NAME, {"value": "7"}) == "old"
        assert fixture.sessions[0].calls == [{"value": 7}]
        assert child.get(outside.name) is None
        assert "not found" in str(await child.execute(outside.name, {}))
        excluded = child.borrow_filtered([outside.name])
        assert excluded.tool_names == []
        excluded.release()
        fixture.registry.unregister(TOOL_NAME)
        fixture.registry.register(fixture.tools[0])
        assert "refused" in str(await child.execute(TOOL_NAME, {"value": 7}))
        fresh = fixture.registry.borrow_filtered([TOOL_NAME])
        try:
            assert await fresh.execute(TOOL_NAME, {"value": 7}) == "old"
        finally:
            fresh.release()
        with pytest.raises(RuntimeError, match="released"):
            parent.borrow_filtered([TOOL_NAME])
        closing = asyncio.create_task(fixture.owner.aclose())
        await asyncio.sleep(0)
        assert not closing.done()
        assert fixture.connections[0].close_calls == 0
        assert "refused" in str(await child.execute(TOOL_NAME, {"value": 7}))
        child.release()
        child.release()
        await asyncio.wait_for(closing, 1)
        assert lease._sampled == lease._inflight == 0
        assert fixture.connections[0].close_calls == 1
    finally:
        parent.release()
        child.release()


@pytest.mark.parametrize("artifact_isolated", [False, True])
async def test_borrowed_child_runs_after_parent_release(
    connected_mcp, artifact_isolated: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = connected_mcp
    parent = fixture.registry.capture_request_tools()
    child = parent.borrow_filtered([TOOL_NAME])
    parent.release()
    lease = fixture.tools[0]._generation_lease
    provider = OfflineProvider([tool_response(), LLMResponse(content="done")])
    provider.release.set()
    admitted = []
    original_admission = fixture.registry.dispatch_admission

    @contextmanager
    def admission(name):
        admitted.append(name)
        with original_admission(name) as error:
            yield error

    monkeypatch.setattr(fixture.registry, "dispatch_admission", admission)
    options = {"artifact_reader_factory": lambda *_args: None} if artifact_isolated else {}
    try:
        assert lease._sampled == 1
        await AgentRunner().run(run_spec(provider, child, **options))
        assert fixture.sessions[0].calls == [{"value": 7}]
        assert fixture.sessions[1].calls == []
        assert admitted == [TOOL_NAME]
        assert lease._sampled == 1
    finally:
        child.release()
    assert lease._sampled == lease._inflight == 0


@pytest.mark.parametrize("phase", ["preparation", "provider", "summary"])
async def test_request_failures_release_sampled_holds(
    connected_mcp, phase: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = connected_mcp
    runner = AgentRunner()
    provider = OfflineProvider([LLMResponse(content="done")])
    entered = asyncio.Event()
    release = asyncio.Event()

    async def fail(*_args, **_kwargs):
        entered.set()
        await release.wait()
        raise RuntimeError(f"{phase} failed")

    if phase == "preparation":
        monkeypatch.setattr(runner.context_governor, "prepare_request", fail)
    elif phase == "provider":
        monkeypatch.setattr(provider, "chat_with_retry", fail)
    else:
        provider.release.set()
        monkeypatch.setattr(runner.context_governor, "summarize_provider_compaction", fail)
    task = asyncio.create_task(runner.run(run_spec(provider, fixture.registry)))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        lease = fixture.tools[0]._generation_lease
        assert lease._sampled == 1
        await fixture.reload()
        assert fixture.connections[0].close_calls == 0
        release.set()
        with pytest.raises(RuntimeError, match=f"{phase} failed"):
            await task
        await asyncio.wait_for(fixture.connections[0].closed.wait(), 1)
        assert lease._sampled == lease._inflight == 0
        assert fixture.connections[0].close_calls == 1
        assert fixture.sessions[0].calls == fixture.sessions[1].calls == []
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("phase", ["preparation", "provider", "execution", "result"])
async def test_cancellation_releases_only_after_owned_work_settles(
    connected_mcp, phase: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = connected_mcp
    runner = AgentRunner()
    provider = OfflineProvider([tool_response(), LLMResponse(content="done")])
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked(*_args, **_kwargs):
        entered.set()
        await release.wait()

    options = {}
    if phase == "preparation":
        monkeypatch.setattr(runner.context_governor, "prepare_request", blocked)
    elif phase == "provider":
        entered = provider.sampled
    elif phase == "execution":
        fixture.sessions[0].block = True
        entered = fixture.sessions[0].started
        provider.release.set()
    else:
        provider.release.set()
        options["tool_result_callback"] = blocked
    task = asyncio.create_task(runner.run(run_spec(provider, fixture.registry, **options)))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        lease = fixture.tools[0]._generation_lease
        assert lease._sampled == 1
        assert lease._inflight == (1 if phase == "execution" else 0)
        await fixture.reload()
        assert fixture.connections[0].close_calls == 0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(fixture.connections[0].closed.wait(), 1)
        assert lease._sampled == lease._inflight == 0
        assert fixture.connections[0].close_calls == 1
        assert fixture.sessions[1].calls == []
        assert len(fixture.sessions[0].calls) == (1 if phase in {"execution", "result"} else 0)
    finally:
        release.set()
        provider.release.set()
        fixture.sessions[0].release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_non_tool_response_and_shutdown_drain_retired_capture(connected_mcp) -> None:
    fixture = connected_mcp
    provider = OfflineProvider([LLMResponse(content="done")])
    task = asyncio.create_task(AgentRunner().run(run_spec(provider, fixture.registry)))
    closing = None
    try:
        await asyncio.wait_for(provider.sampled.wait(), 1)
        lease = fixture.tools[0]._generation_lease
        await fixture.reload()
        assert lease in fixture.owner._retired_generations
        closing = asyncio.create_task(fixture.owner.aclose())
        await asyncio.wait_for(fixture.connections[1].closed.wait(), 1)
        assert not closing.done()
        assert fixture.connections[0].close_calls == 0
        assert lease._sampled == 1
        provider.release.set()
        assert (await task).final_content == "done"
        await asyncio.wait_for(closing, 1)
        await fixture.owner.aclose()
        assert fixture.owner._retired_generations == set()
        assert [connection.close_calls for connection in fixture.connections] == [1, 1]
        assert lease._sampled == lease._inflight == 0
    finally:
        provider.release.set()
        await asyncio.gather(task, return_exceptions=True)
        if closing is not None:
            await closing


async def test_settled_tool_round_releases_before_next_provider_wait(connected_mcp) -> None:
    fixture = connected_mcp
    next_sampled = asyncio.Event()
    next_release = asyncio.Event()
    result_entered = asyncio.Event()
    result_release = asyncio.Event()

    class NextRoundProvider(OfflineProvider):
        async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
            response = await super().chat(messages, tools, **kwargs)
            if len(self.definitions) == 2:
                next_sampled.set()
                await next_release.wait()
            return response

    async def record_result(_result):
        result_entered.set()
        await result_release.wait()

    provider = NextRoundProvider([tool_response(), LLMResponse(content="done")])
    provider.release.set()
    task = asyncio.create_task(AgentRunner().run(run_spec(
        provider, fixture.registry, tool_result_callback=record_result,
    )))
    try:
        await asyncio.wait_for(result_entered.wait(), 1)
        old_lease = fixture.tools[0]._generation_lease
        assert old_lease._sampled == 1 and old_lease._inflight == 0
        await fixture.reload()
        assert fixture.connections[0].close_calls == 0
        result_release.set()
        await asyncio.wait_for(next_sampled.wait(), 1)
        await asyncio.wait_for(fixture.connections[0].closed.wait(), 1)
        assert not task.done()
        assert old_lease._sampled == old_lease._inflight == 0
        assert fixture.connections[0].close_calls == 1
        assert fixture.tools[1]._generation_lease._sampled == 1
        assert provider.definitions[1][0]["function"]["description"] == "replacement"
        assert fixture.sessions[0].calls == [{"value": 7}]
        assert fixture.sessions[1].calls == []
        next_release.set()
        await task
        assert fixture.tools[1]._generation_lease._sampled == 0
    finally:
        result_release.set()
        next_release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_cancelled_shutdown_keeps_retired_borrower_tracked(connected_mcp) -> None:
    fixture = connected_mcp
    capture = fixture.registry.borrow_filtered([TOOL_NAME])
    closing = None
    try:
        lease = fixture.tools[0]._generation_lease
        await fixture.reload()
        closing = asyncio.create_task(fixture.owner.aclose())
        await asyncio.wait_for(fixture.connections[1].closed.wait(), 1)
        assert fixture.connections[0].close_calls == 0
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert lease in fixture.owner._retired_generations
        assert lease._sampled == 1
        assert "refused" in str(await capture.execute(TOOL_NAME, {"value": 7}))
        capture.release()
        capture.release()
        await asyncio.wait_for(fixture.owner.aclose(), 1)
        assert fixture.owner._retired_generations == set()
        assert [connection.close_calls for connection in fixture.connections] == [1, 1]
        assert lease._sampled == lease._inflight == 0
        assert fixture.sessions[0].calls == fixture.sessions[1].calls == []
    finally:
        capture.release()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)


async def test_shutdown_cancellation_does_not_abandon_generation_close(
    connected_mcp, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = connected_mcp
    started = asyncio.Event()
    release = asyncio.Event()
    connection = fixture.connections[0]
    lease = fixture.tools[0]._generation_lease

    async def blocked_close():
        connection.close_calls += 1
        started.set()
        await release.wait()
        connection.closed.set()

    monkeypatch.setattr(connection, "aclose", blocked_close)
    capture = fixture.registry.capture_request_tools()
    capture.release()
    closing = asyncio.create_task(fixture.owner.aclose())
    try:
        await asyncio.wait_for(started.wait(), 1)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert lease in fixture.owner._retired_generations
        assert not connection.closed.is_set()
        retry = asyncio.create_task(fixture.owner.aclose())
        await asyncio.sleep(0)
        assert not retry.done()
        release.set()
        await asyncio.wait_for(retry, 1)
        assert connection.close_calls == 1
        assert connection.closed.is_set()
        assert fixture.owner._retired_generations == set()
    finally:
        release.set()
        await asyncio.gather(closing, return_exceptions=True)


async def test_initial_readiness_never_publishes_unpinned_wrappers() -> None:
    fixture = MCPFixture()
    ready = asyncio.Event()
    release = asyncio.Event()

    async def connect(servers, registry, **kwargs):
        connections = await fixture.connect(servers, registry, **kwargs)
        ready.set()
        await release.wait()
        return connections

    with patch("nanobot.agent.tools.mcp.connect_mcp_servers", new=connect):
        task = asyncio.create_task(fixture.owner.connect())
        try:
            await asyncio.wait_for(ready.wait(), 1)
            assert fixture.registry.get(TOOL_NAME) is None
            capture = fixture.registry.capture_request_tools()
            assert capture.tool_names == []
            capture.release()
            release.set()
            await task
            assert fixture.registry.get(TOOL_NAME) is fixture.tools[0]
            capture = fixture.registry.capture_request_tools()
            assert fixture.tools[0]._generation_lease._sampled == 1
            capture.release()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await fixture.owner.aclose()


async def test_readiness_publication_collision_closes_unpublished_connection() -> None:
    fixture = MCPFixture()
    existing = MCPToolWrapper(Session("existing"), "probe", SimpleNamespace(
        name="probe", description="existing", inputSchema={"type": "object", "properties": {}},
    ))
    fixture.registry.register(existing)
    with patch("nanobot.agent.tools.mcp.connect_mcp_servers", new=fixture.connect):
        try:
            await fixture.owner.connect()
            assert fixture.registry.get(TOOL_NAME) is existing
            assert fixture.owner.connected_server_names == set()
            assert fixture.connections[0].close_calls == 1
            assert fixture.sessions[0].calls == []
        finally:
            await fixture.owner.aclose()


@pytest.mark.parametrize("phase", ["response", "result"])
async def test_mandatory_source_failure_releases_owned_binding(
    connected_mcp, phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = connected_mcp
    store = JsonlSessionStore(tmp_path)
    session = CanonicalSession("test:mcp")
    store.save(session)
    store.source.upgrade_source_session(
        session.key, expected_revision=session.revision, expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    provider = OfflineProvider([tool_response(), LLMResponse(content="done")])
    entered = asyncio.Event()
    release = asyncio.Event()
    async with acquire_execution(store, session.key, run_id="mcp-run") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="mcp-run")
        writer.append_message({"role": "user", "content": "probe"})
        if phase == "response":
            entered = provider.sampled

            def fail_response(*_args, **_kwargs):
                raise OSError("source response failed")

            monkeypatch.setattr(writer, "begin_tool_batch", fail_response)
        else:
            provider.release.set()

            async def fail_result(*_args, **_kwargs):
                entered.set()
                await release.wait()
                raise OSError("source result failed")

            monkeypatch.setattr(writer, "record_tool_result_event", fail_result)
        task = asyncio.create_task(AgentRunner().run(run_spec(
            provider, fixture.registry, source_writer=writer,
            turn_context=TurnContext(execution_owner=owner),
        )))
        owner.track_task(task)
        try:
            try:
                await asyncio.wait_for(entered.wait(), 1)
            except TimeoutError:
                if task.done():
                    task.result()
                raise
            lease = fixture.tools[0]._generation_lease
            assert lease._sampled == 1 and lease._inflight == 0
            await fixture.reload()
            assert fixture.connections[0].close_calls == 0
            release.set()
            provider.release.set()
            with pytest.raises(OSError, match=f"source {phase} failed"):
                await task
            await asyncio.wait_for(fixture.connections[0].closed.wait(), 1)
            assert writer.failed is True
            assert lease._sampled == lease._inflight == 0
            assert fixture.sessions[1].calls == []
            assert len(fixture.sessions[0].calls) == (1 if phase == "result" else 0)
            assert fixture.connections[0].close_calls == 1
        finally:
            release.set()
            provider.release.set()
            await asyncio.gather(task, return_exceptions=True)


async def test_risk_assessment_resolves_the_captured_concrete_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    old = WriteFileTool(workspace=old_root, allowed_dir=old_root)
    replacement = WriteFileTool(workspace=new_root, allowed_dir=new_root)
    registry = ToolRegistry()
    registry.register(old)
    current_resolver_calls = []

    def current_resolver(name, path):
        current_resolver_calls.append((name, path))
        return registry.get(name).resolve_write_path(path)

    policy = RiskyActionPolicy(workspace=old_root, write_target_resolver=current_resolver)
    original_evaluate = RiskyActionPolicy.evaluate
    assessed = []

    async def evaluate(self, **kwargs):
        assessed.append(self.write_target_resolver("write_file", "probe.txt"))
        return await original_evaluate(self, **kwargs)

    monkeypatch.setattr(RiskyActionPolicy, "evaluate", evaluate)
    provider = OfflineProvider([LLMResponse(content=None, tool_calls=[ToolCallRequest(
        id="write", name="write_file", arguments={"path": "probe.txt", "content": "safe"},
    )]), LLMResponse(content="done")])
    task = asyncio.create_task(AgentRunner().run(run_spec(provider, registry, tool_policy=policy)))
    try:
        await asyncio.wait_for(provider.sampled.wait(), 1)
        registry.replace(replacement)
        provider.release.set()
        await task
        assert assessed == [old_root / "probe.txt"]
        assert current_resolver_calls == []
        assert not (old_root / "probe.txt").exists()
        assert not (new_root / "probe.txt").exists()
    finally:
        provider.release.set()
        await asyncio.gather(task, return_exceptions=True)
