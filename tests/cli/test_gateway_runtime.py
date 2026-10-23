"""Gateway teardown tests use the real acquired, effect-fenced composition.

Legacy timeout-and-detach cleanup is not the gateway lifecycle contract:
gateway tasks must settle before their composition releases shared resources.
"""

import asyncio

import pytest

from nanobot.agent.hook import AgentRunHookContext
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.mcp import MCPProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.cli.gateway_runtime import (
    _close_gateway_runtime,
    _gateway_readiness_payload,
    _MCPReadinessHook,
)
from nanobot.providers.unconfigured_provider import UnconfiguredProvider
from nanobot.runtime_composition import RuntimeState
from tests.agent.p6_admitted_fixtures import AdmittedGraphs


class _TrackingMCPProvider(MCPProvider):
    def __init__(self) -> None:
        super().__init__({}, ToolRegistry())
        self.connect_calls = 0

    async def connect(self) -> None:
        self.connect_calls += 1


@pytest.fixture
async def gateway_graph(tmp_path):
    events = []
    with pytest.MonkeyPatch.context() as patches:
        graphs = AdmittedGraphs(tmp_path, patches)
        mcp_close = MCPProvider.aclose

        async def close_mcp(provider):
            events.append("mcp-close")
            await mcp_close(provider)

        patches.setattr(MCPProvider, "aclose", close_mcp)

        def build(inputs):
            agent = AgentLoop.from_config(
                inputs.config, inputs.bus,
                provider=UnconfiguredProvider("test/model"), model="test/model",
                context_window_tokens=4096, session_manager=inputs.sessions,
                tool_registry=inputs.tools, cron_service=inputs.cron,
                turn_delivery_factory=inputs.delivery,
            )
            agent_close = agent.aclose

            async def close_agent():
                events.append("agent-close")
                await agent_close()

            patches.setattr(agent, "aclose", close_agent)
            return agent

        try:
            await graphs.acquire(tmp_path, build)
            composition = graphs.graphs[0][0]
            handle = composition.require_ready()
            composition_close = composition.aclose

            async def close_composition():
                events.append("composition-close")
                await composition_close()

            patches.setattr(composition, "aclose", close_composition)
            yield handle, composition, events
        finally:
            await graphs.close()


async def close_gateway(graph, tasks=()):
    handle, composition, events = graph
    await _close_gateway_runtime(
        handle, composition, tasks,
        disconnect_webui=lambda: events.append("disconnect"),
        restore=lambda: events.append("restore"),
    )


async def test_runtime_tasks_settle_before_composition_resources_close(gateway_graph):
    handle, composition, events = gateway_graph
    started = asyncio.Event()

    async def run():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            events.append("task-settled")

    task = asyncio.create_task(run())
    await started.wait()
    await close_gateway(gateway_graph, [task])

    assert task.cancelled()
    assert composition.state is RuntimeState.CLOSED
    assert handle.agent._closed and handle.mcp._closing
    expected = ["task-settled", "disconnect", "composition-close",
                "agent-close", "mcp-close", "restore"]
    assert events == expected
    assert not handle.bus._pending


async def test_cancellation_resistant_cleanup_is_joined_before_resource_release(gateway_graph):
    _, composition, events = gateway_graph
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def run():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            events.append("task-settled")

    task = asyncio.create_task(run())
    await started.wait()
    shutdown = asyncio.create_task(close_gateway(gateway_graph, [task]))
    try:
        async with asyncio.timeout(1):
            await cleaning.wait()
        assert not shutdown.done()
        assert composition.state is RuntimeState.READY
        assert "composition-close" not in events
        release.set()
        await shutdown
        assert task.done()
        assert events.index("task-settled") < events.index("composition-close")
        assert composition.state is RuntimeState.CLOSED
    finally:
        release.set()
        await asyncio.gather(task, shutdown, return_exceptions=True)


async def test_agent_background_work_is_settled_before_gateway_close_returns(gateway_graph):
    handle, _, events = gateway_graph
    started = asyncio.Event()

    async def background():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            events.append("background-settled")

    handle.agent.schedule_background(background())
    await started.wait()
    await close_gateway(gateway_graph)
    assert "background-settled" in events
    assert events.index("background-settled") < events.index("restore")
    assert not handle.agent._background_tasks


async def test_composition_error_is_propagated_not_silently_logged(gateway_graph, monkeypatch):
    handle, composition, events = gateway_graph
    original_close = composition.aclose
    error = RuntimeError("composition cleanup failed")

    async def fail_after_close():
        await original_close()
        raise error

    monkeypatch.setattr(composition, "aclose", fail_after_close)
    try:
        with pytest.raises(RuntimeError) as caught:
            await close_gateway(gateway_graph)
        assert caught.value is error
        assert handle.agent._closed and handle.mcp._closing
        assert events.count("agent-close") == events.count("mcp-close") == 1
    finally:
        monkeypatch.setattr(composition, "aclose", original_close)


async def test_repeated_gateway_cleanup_does_not_double_close_owned_resources(gateway_graph):
    _, composition, events = gateway_graph
    await close_gateway(gateway_graph)
    await close_gateway(gateway_graph)
    assert composition.state is RuntimeState.CLOSED
    assert events.count("agent-close") == events.count("mcp-close") == 1


@pytest.mark.parametrize("outcome", ["finished", "failed", "cancelled"])
async def test_settled_runtime_task_results_are_retrieved(gateway_graph, outcome):
    _, composition, _ = gateway_graph

    async def run():
        if outcome == "failed":
            raise ValueError("runtime task failed")
        if outcome == "cancelled":
            raise asyncio.CancelledError()

    task = asyncio.create_task(run())
    await asyncio.sleep(0)
    await close_gateway(gateway_graph, [task])
    assert task.done()
    assert composition.state is RuntimeState.CLOSED


def test_gateway_readiness_is_degraded_when_required_websocket_is_unavailable() -> None:
    channels = type(
        "Channels",
        (),
        {
            "enabled_channels": ["websocket"],
            "get_status": lambda self: {
                "websocket": {
                    "enabled": True,
                    "running": False,
                    "state": "starting",
                }
            },
        },
    )()

    ready, payload = _gateway_readiness_payload(channels)

    assert ready is False
    assert payload == {
        "status": "degraded",
        "process": "alive",
        "ready": False,
        "websocket": "starting",
    }


async def test_health_server_reports_its_actual_bound_management_port() -> None:
    from nanobot.cli.gateway_runtime import _health_server

    ports: list[int] = []
    bound = asyncio.Event()

    def capture(port: int) -> None:
        ports.append(port)
        bound.set()

    server = asyncio.create_task(
        _health_server("127.0.0.1", 0, object(), capture),
    )
    try:
        await asyncio.wait_for(bound.wait(), timeout=2)
        assert len(ports) == 1
        assert 1 <= ports[0] <= 65535
    finally:
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)


async def test_mcp_readiness_hook_delegates_to_application_provider() -> None:
    provider = _TrackingMCPProvider()
    hook = _MCPReadinessHook(provider)

    await hook.before_run(AgentRunHookContext(messages=[]))

    assert provider.connect_calls == 1


@pytest.mark.parametrize("calls", [1, 3])
def test_webui_refresh_notification_is_lazy_and_discards_runtime(calls) -> None:
    from nanobot.cli.gateway_runtime import _webui_refresh_notification
    from nanobot.providers.unconfigured_provider import UnconfiguredProvider
    from nanobot.utils.llm_runtime import LLMRuntime

    class UnreadRuntime(LLMRuntime):
        def __bool__(self):
            raise AssertionError("refresh result must not be inspected")

        def __str__(self):
            raise AssertionError("refresh result must not be formatted")

    runtime = UnreadRuntime.capture(
        UnconfiguredProvider("test/model"), "test/model", context_window_tokens=4096,
    )
    seen: list[str] = []

    def refresh() -> LLMRuntime:
        seen.append("refresh")
        return runtime

    notify = _webui_refresh_notification(refresh)
    assert seen == []
    for count in range(1, calls + 1):
        assert notify() is None
        assert seen == ["refresh"] * count
    assert runtime.model == "test/model"


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, asyncio.CancelledError])
def test_webui_refresh_notification_preserves_error_identity(error_type) -> None:
    from nanobot.cli.gateway_runtime import _webui_refresh_notification
    from nanobot.utils.llm_runtime import LLMRuntime

    error = error_type("synthetic refresh error")
    seen: list[str] = []

    def refresh() -> LLMRuntime:
        seen.append("refresh")
        raise error

    notify = _webui_refresh_notification(refresh)
    assert seen == []
    with pytest.raises(error_type) as caught:
        notify()
    assert caught.value is error
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert seen == ["refresh"]


def test_webui_refresh_notification_keeps_captured_method_binding() -> None:
    from nanobot.cli.gateway_runtime import _webui_refresh_notification
    from nanobot.providers.unconfigured_provider import UnconfiguredProvider
    from nanobot.utils.llm_runtime import LLMRuntime

    runtime = LLMRuntime.capture(
        UnconfiguredProvider("test/model"), "test/model", context_window_tokens=4096,
    )
    seen: list[str] = []

    class Owner:
        def refresh_runtime_config(self) -> LLMRuntime:
            seen.append("captured")
            return runtime

    def replacement() -> LLMRuntime:
        raise AssertionError("notification must retain the original bound method")

    owner = Owner()
    notify = _webui_refresh_notification(owner.refresh_runtime_config)
    owner.refresh_runtime_config = replacement
    assert seen == []
    assert notify() is None
    assert seen == ["captured"]
