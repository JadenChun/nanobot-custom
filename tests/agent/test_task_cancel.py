"""Tests for /stop task cancellation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.bus.outbound_events import StreamDeltaEvent, StreamEndEvent
from nanobot.providers.base import LLMProvider, LLMResponse


class _ScriptedProvider(LLMProvider):
    """Deterministic provider used to exercise the complete foreground route."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(provider_name="scripted-test")
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, object] | None = None,
    ) -> LLMResponse:
        from nanobot.agent.tools.context import current_request_context

        self.calls.append({
            "messages": messages,
            "tools": tools or [],
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "tool_choice": tool_choice,
            "request_context": current_request_context(),
        })
        if not self._responses:
            raise AssertionError("scripted provider received an unexpected call")
        return self._responses.pop(0)

    def get_default_model(self) -> str:
        return "scripted-model"


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
        manager = loop._fork_dependencies.foreground_manager
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocked_run(_spec):
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        manager.runner.run = blocked_run
        loop.subagents.cancel_by_session = AsyncMock(return_value=0)
        loop._exec_session_manager.terminate_by_owner = AsyncMock(return_value=0)
        task = asyncio.create_task(manager.run_plan(objective="Inspect the report"))
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

    @pytest.mark.asyncio
    async def test_foreground_delegation_returns_direct_result(self, tmp_path):
        from nanobot.agent.delegation import ForegroundAgentManager
        from nanobot.agent.runner import AgentRunResult

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        manager = ForegroundAgentManager(provider=provider, workspace=tmp_path)
        manager.runner.run = AsyncMock(return_value=AgentRunResult(
            final_content="complete",
            messages=[],
        ))

        result = await manager.run_plan(objective="Inspect the report")

        assert result == "complete"
        manager.runner.run.assert_awaited_once()
        spec = manager.runner.run.await_args.args[0]
        assert spec.tools.get("spawn") is None
        assert spec.tools.get("delegate_task") is None

    @pytest.mark.asyncio
    async def test_foreground_parent_cancellation_reaches_runner(self, tmp_path):
        from nanobot.agent.delegation import ForegroundAgentManager

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        manager = ForegroundAgentManager(provider=provider, workspace=tmp_path)
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocked_run(_spec):
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        manager.runner.run = blocked_run
        task = asyncio.create_task(manager.run_plan(objective="Inspect the report"))
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()


class TestForegroundResultRouting:
    """Foreground roles return to the parent turn instead of publishing background messages."""

    @pytest.mark.asyncio
    async def test_foreground_plan_uses_parent_result(self, tmp_path):
        from nanobot.agent.delegation import ForegroundAgentManager
        from nanobot.agent.runner import AgentRunner
        from nanobot.agent.tools.context import RequestContext, request_context
        from nanobot.agent.tools.delegation import PlanTaskTool
        from nanobot.bus.queue import MessageBus
        from nanobot.utils.llm_runtime import LLMRuntime

        provider = _ScriptedProvider([
            LLMResponse(content="Planner result", finish_reason="stop"),
        ])
        runtime = LLMRuntime.capture(provider, "scripted-model", context_window_tokens=128_000)
        manager = ForegroundAgentManager(
            provider=provider,
            workspace=tmp_path,
            runtime=runtime,
        )
        bus = MessageBus()
        with request_context(RequestContext(
            channel="cli",
            chat_id="direct",
            session_key="cli:foreground-e2e",
            runtime=runtime,
            workspace=tmp_path,
        )):
            result = await PlanTaskTool(manager).execute(objective="Inspect the report")

        assert result == "Planner result"
        assert bus.outbound_size == 0
        assert getattr(manager, "_background_tasks", set()) == set()
        assert isinstance(manager.runner, AgentRunner)
        assert manager._baseline_runtime is runtime
        assert len(provider.calls) == 1
        context = provider.calls[0]["request_context"]
        assert context is not None
        assert context.runtime is runtime
        assert context.runtime.provider is provider
        assert context.workspace == tmp_path
        assert "# Planning Role" in provider.calls[0]["messages"][0]["content"]

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
    async def test_foreground_worker_result_stays_in_parent_turn(self, tmp_path):
        from nanobot.agent.delegation import ForegroundAgentManager
        from nanobot.agent.runner import AgentRunResult

        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        manager = ForegroundAgentManager(provider=provider, workspace=tmp_path)
        manager.runner.run = AsyncMock(return_value=AgentRunResult(
            final_content="worker result",
            messages=[],
        ))

        result = await manager.run_worker(
            contract="# Contract\nUpdate the report and validate it.",
            write_scope=["reports/"],
        )

        assert result == "worker result"
        manager.runner.run.assert_awaited_once()
