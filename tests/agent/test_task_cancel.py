"""Tests for /stop task cancellation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.outbound_events import StreamDeltaEvent, StreamEndEvent
from nanobot.providers.base import LLMResponse, ToolCallRequest
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade

engine_config = engine_fixtures.engine_config


def _make_loop(tmp_path, *, tools_config=None, provider=None):
    """Create an AgentLoop with the real canonical dependency graph."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.config.schema import ToolsConfig

    bus = MessageBus()
    provider = provider or MagicMock()
    if isinstance(provider, MagicMock):
        provider.get_default_model.return_value = "test-model"
    config = tools_config or ToolsConfig()

    # Post-guard tests open only the constructor seam; turn execution runs
    # with the real context/session/subagent implementations below.
    with patch("nanobot.migration_guard.require_runtime_ready"):
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=tmp_path,
            tools_config=config,
            timezone="Asia/Kuala_Lumpur",
        )
    return loop, bus


class TestActiveTaskTracking:
    @pytest.mark.asyncio
    async def test_completed_task_removes_empty_session_group(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)
        release = asyncio.Event()
        task = asyncio.create_task(release.wait())

        loop._track_active_task("test:c1", task)
        release.set()
        await task
        await asyncio.sleep(0)

        assert "test:c1" not in loop._active_tasks

    @pytest.mark.asyncio
    async def test_session_group_remains_until_last_task_completes(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)
        releases = [asyncio.Event(), asyncio.Event()]
        tasks = [asyncio.create_task(release.wait()) for release in releases]
        for task in tasks:
            loop._track_active_task("test:c1", task)

        releases[0].set()
        await tasks[0]
        await asyncio.sleep(0)

        assert loop._active_tasks["test:c1"] == {tasks[1]}

        releases[1].set()
        await tasks[1]
        await asyncio.sleep(0)

        assert "test:c1" not in loop._active_tasks

    @pytest.mark.asyncio
    async def test_old_callback_preserves_replacement_session_group(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)
        old_release = asyncio.Event()
        new_release = asyncio.Event()
        old_task = asyncio.create_task(old_release.wait())
        new_task = asyncio.create_task(new_release.wait())

        loop._track_active_task("test:c1", old_task)
        loop._active_tasks.pop("test:c1")
        loop._track_active_task("test:c1", new_task)

        old_release.set()
        await old_task
        await asyncio.sleep(0)

        assert loop._active_tasks["test:c1"] == {new_task}

        new_release.set()
        await new_task
        await asyncio.sleep(0)

        assert "test:c1" not in loop._active_tasks


class TestHandleStop:
    @pytest.mark.asyncio
    async def test_stop_no_active_task(self, tmp_path):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, bus = _make_loop(tmp_path)
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)
        assert "No active task" in out.content

    @pytest.mark.asyncio
    async def test_aclose_cancels_active_turn_before_resources(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)
        events: list[str] = []

        async def active_turn():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                events.append("turn_cancelled")
                raise

        task = asyncio.create_task(active_turn())
        await asyncio.sleep(0)
        loop._active_tasks["test:c1"] = {task}

        async def close_subagents():
            events.append("resources_closed")

        loop.subagents.close = close_subagents
        loop._exec_session_manager.close_all = AsyncMock()
        await loop.aclose()

        assert events == ["turn_cancelled", "resources_closed"]
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_aclose_serializes_duplicate_cleanup(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)
        entered = asyncio.Event()
        release = asyncio.Event()
        concurrent = 0
        max_concurrent = 0

        async def close_subagents():
            nonlocal concurrent, max_concurrent
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            entered.set()
            await release.wait()
            concurrent -= 1

        loop.subagents.close = close_subagents
        loop._exec_session_manager.close_all = AsyncMock()
        first = asyncio.create_task(loop.aclose())
        await entered.wait()
        second = asyncio.create_task(loop.aclose())
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        await asyncio.gather(first, second)

        assert max_concurrent == 1

    @pytest.mark.asyncio
    async def test_stop_cancels_active_task(self, tmp_path):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop(tmp_path)
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocked_task():
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        loop.subagents.cancel_by_session = AsyncMock(return_value=0)
        loop._exec_session_manager.terminate_by_owner = AsyncMock(return_value=0)
        task = asyncio.create_task(blocked_task())
        await asyncio.wait_for(started.wait(), timeout=1.0)
        loop._track_active_task("test:c1", task)

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)

        assert cancelled.is_set()
        assert "stopped" in out.content.lower()
        assert loop._active_tasks == {}
        loop.subagents.cancel_by_session.assert_awaited_once_with("test:c1")
        loop._exec_session_manager.terminate_by_owner.assert_awaited_once_with("test:c1")

    @pytest.mark.asyncio
    async def test_stop_cancels_multiple_tasks(self, tmp_path):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, bus = _make_loop(tmp_path)
        events = [asyncio.Event(), asyncio.Event()]

        async def slow(idx):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                events[idx].set()
                raise

        tasks = [asyncio.create_task(slow(i)) for i in range(2)]
        await asyncio.sleep(0)
        loop._active_tasks["test:c1"] = set(tasks)

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)

        assert all(e.is_set() for e in events)
        assert "2 task" in out.content


class TestDispatch:
    @pytest.mark.asyncio
    async def test_run_logs_and_continues_after_leaked_cancelled_error(self, monkeypatch, tmp_path):
        loop, bus = _make_loop(tmp_path)
        loop.aclose = AsyncMock()
        loop.auto_compact.check_expired = MagicMock()
        warnings: list[str] = []
        calls = 0

        async def consume_once_then_stop():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise asyncio.CancelledError()
            loop.stop()
            raise asyncio.TimeoutError()

        monkeypatch.setattr(bus, "consume_inbound", consume_once_then_stop)
        monkeypatch.setattr(
            "nanobot.agent.loop.logger.warning",
            lambda message, *args, **kwargs: warnings.append(message),
        )

        await loop.run()

        assert calls == 2
        assert any("Ignoring leaked CancelledError" in warning for warning in warnings)

    def test_exec_tool_not_registered_when_disabled(self, tmp_path):
        from nanobot.agent.tools.shell import ExecToolConfig
        from nanobot.config.schema import ToolsConfig

        loop, _bus = _make_loop(tmp_path, tools_config=ToolsConfig(exec=ExecToolConfig(enable=False)))

        assert loop.tools.get("exec") is None

    @pytest.mark.asyncio
    async def test_dispatch_processes_and_publishes(self, tmp_path):
        from nanobot.agent.turn import TurnResult
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop(tmp_path)
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="hello")

        async def fake_execute(request, *, _options):
            async with loop._admit_after_bus_cleanup(request.session_key):
                async with _options.admission_scope(request.session_key) as io:
                    io.result = TurnResult(
                        run_id="run-1",
                        outbound=OutboundMessage(channel="test", chat_id="c1", content="hi"),
                    )
                    return io.result

        loop.execute_turn = fake_execute
        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert out.content == "hi"

    @pytest.mark.asyncio
    async def test_dispatch_streaming_preserves_message_metadata(self, tmp_path):
        from nanobot.agent.turn import TurnResult
        from nanobot.bus.events import InboundMessage

        loop, bus = _make_loop(tmp_path)
        msg = InboundMessage(
            channel="matrix",
            sender_id="u1",
            chat_id="!room:matrix.org",
            content="hello",
            metadata={
                "_wants_stream": True,
                "thread_root_event_id": "$root1",
                "thread_reply_to_event_id": "$reply1",
            },
        )

        async def fake_execute(request, *, _options):
            async with loop._admit_after_bus_cleanup(request.session_key):
                async with _options.admission_scope(request.session_key) as io:
                    assert io.delivery is not None and io.delivery.streaming
                    await io.delivery.events.emit(StreamDeltaEvent(content="hi"))
                    await io.delivery.events.emit(StreamEndEvent())
                    io.result = TurnResult(run_id="run-2")
                    return io.result

        loop.execute_turn = fake_execute

        await loop._dispatch(msg)
        first = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        second = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        assert first.metadata["thread_root_event_id"] == "$root1"
        assert first.metadata["thread_reply_to_event_id"] == "$reply1"
        assert isinstance(first.event, StreamDeltaEvent)
        assert second.metadata["thread_root_event_id"] == "$root1"
        assert second.metadata["thread_reply_to_event_id"] == "$reply1"
        assert isinstance(second.event, StreamEndEvent)

    @pytest.mark.asyncio
    async def test_same_session_dispatches_serialize(self, tmp_path):
        from nanobot.agent.turn import TurnResult
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop(tmp_path)
        order = []
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def fake_execute(request, *, _options):
            async with loop._admit_after_bus_cleanup(request.session_key):
                async with _options.admission_scope(request.session_key) as io:
                    order.append(f"start-{request.content}")
                    if request.content == "a":
                        first_started.set()
                        await release_first.wait()
                    order.append(f"end-{request.content}")
                    io.result = TurnResult(
                        run_id=f"run-{request.content}",
                        outbound=OutboundMessage(
                            channel="test", chat_id="c1", content=request.content
                        ),
                    )
                    return io.result

        loop.execute_turn = fake_execute
        msg1 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="a")
        msg2 = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="b")

        t1 = asyncio.create_task(loop._dispatch(msg1))
        await asyncio.wait_for(first_started.wait(), timeout=1.0)
        t2 = asyncio.create_task(loop._dispatch(msg2))
        await asyncio.sleep(0)
        assert order == ["start-a"]

        release_first.set()
        await asyncio.gather(t1, t2)
        assert order == ["start-a", "end-a", "start-b", "end-b"]


class TestRetiredBackgroundRoutes:
    """The imported background implementation remains inert under the fork contract."""

    def test_main_registry_excludes_background_routes(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)

        assert loop.tools.get("spawn") is None
        assert loop.tools.get("spawn_pipeline") is None
        assert loop.tools.get("delegate_task") is not None

    def test_subagent_scope_is_rejected_before_tool_construction(self):
        from nanobot.agent.tools.loader import ToolLoader
        from nanobot.agent.tools.registry import ToolRegistry

        registry = ToolRegistry()
        with patch("nanobot.migration_guard.require_runtime_ready"):
            with pytest.raises(ValueError, match="Unsupported runtime tool scope"):
                ToolLoader().load(object(), registry, scope="subagent")
        assert registry.tool_names == []

    def test_foreground_worker_has_no_shell_or_nested_delegation(self, tmp_path):
        from nanobot.agent.delegation import ForegroundAgentManager

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        manager = ForegroundAgentManager(provider=provider, workspace=tmp_path)
        scopes = manager._normalize_scopes(tmp_path, ["reports/"])
        tools = manager._worker_tools("foreground", scopes)

        assert tools.get("exec") is None
        assert tools.get("delegate_task") is None
        assert tools.get("spawn") is None

    def test_disabled_exec_does_not_reopen_background_routes(self, tmp_path):
        from nanobot.agent.tools.shell import ExecToolConfig
        from nanobot.config.schema import ToolsConfig

        loop, _bus = _make_loop(tmp_path, tools_config=ToolsConfig(exec=ExecToolConfig(enable=False)))

        assert loop.tools.get("exec") is None
        assert loop.tools.get("spawn") is None
        assert loop.tools.get("spawn_pipeline") is None

    def test_factory_registry_keeps_foreground_roles_available(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)

        assert {"plan_task", "delegate_task", "review_work", "explore"} <= set(
            loop.tools.tool_names
        )
        assert {"spawn", "spawn_pipeline"}.isdisjoint(loop.tools.tool_names)

    # `test_foreground_delegation_returns_direct_result` is retired. It injected the
    # retired inline `manager.runner` and called `run_plan` outside a canonical turn.
    # The planner-result routing is asserted by
    # `tests/agent/test_migration_engine.py::test_real_main_runner_delegates_to_real_planner_runner`.

    # `test_foreground_parent_cancellation_reaches_runner` is retired. It injected the
    # retired inline `manager.runner` and cancelled `run_plan` directly. Canonical Stop
    # cancellation is asserted by
    # `tests/agent/test_delegation_child_authority.py::test_original_stop_keeps_root_fence_until_resistant_root_settles`.


class TestForegroundResultRouting:
    """Foreground roles return to the parent turn instead of publishing background messages."""

    @pytest.mark.asyncio
    async def test_foreground_plan_uses_parent_result(self, engine_config):
        from nanobot.agent.loop import AgentLoop
        from nanobot.agent.tools.registry import ToolRegistry
        from nanobot.agent.turn import TurnRequest

        provider = ScriptedProvider([
            LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id="plan", name="plan_task", arguments={"objective": "Inspect the report"},
            )]),
            LLMResponse(content="Planner result"),
            LLMResponse(content="planned"),
        ])
        loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
        upgrade(loop, "cli:source")
        try:
            result = await loop.execute_turn(TurnRequest(
                content="inspect the report", session_key="cli:source",
            ))
            assert result.content == "planned"
            assert result.tools_used == ["plan_task"]
            assert "# Planning Role" in provider.calls[1][0][0]["content"]
            assert "Planner result" in str(provider.calls[2][0])
            outbound = []
            while loop.bus.outbound_size:
                outbound.append(await loop.bus.consume_outbound())
            assert "Planner result" not in {message.content for message in outbound}
        finally:
            await loop.aclose()

    def test_read_only_foreground_roles_have_no_background_tools(self, tmp_path):
        from nanobot.agent.delegation import ForegroundAgentManager

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        manager = ForegroundAgentManager(provider=provider, workspace=tmp_path)

        for role in ("planner", "reviewer", "explorer", "crawler"):
            tools = manager._tools_for_role(role)
            assert tools.get("spawn") is None
            assert tools.get("spawn_pipeline") is None
            assert tools.get("delegate_task") is None

    def test_loader_does_not_register_background_tools_from_imports(self):
        from nanobot.agent.tools.loader import ToolLoader
        from nanobot.agent.tools.registry import ToolRegistry

        registry = ToolRegistry()
        with patch("nanobot.migration_guard.require_runtime_ready"):
            with pytest.raises(ValueError, match="Unsupported runtime tool scope"):
                ToolLoader().load(object(), registry, scope="subagent")
        assert registry.tool_names == []

    @pytest.mark.asyncio
    async def test_foreground_worker_result_stays_in_parent_turn(self, engine_config):
        from nanobot.agent.loop import AgentLoop
        from nanobot.agent.tools.registry import ToolRegistry
        from nanobot.agent.turn import TurnRequest

        provider = ScriptedProvider([
            LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id="delegate", name="delegate_task", arguments={
                    "contract": "Update the report and validate it.",
                    "write_scope": ["reports/"],
                },
            )]),
            LLMResponse(content="worker result"),
            LLMResponse(content="main complete"),
        ])
        loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
        upgrade(loop, "cli:source")
        try:
            result = await loop.execute_turn(TurnRequest(
                content="delegate the report work", session_key="cli:source",
            ))
            assert result.content == "main complete"
            assert result.tools_used == ["delegate_task"]
            assert "worker result" in str(provider.calls[2][0])
            outbound = []
            while loop.bus.outbound_size:
                outbound.append(await loop.bus.consume_outbound())
            assert "worker result" not in {message.content for message in outbound}
        finally:
            await loop.aclose()
