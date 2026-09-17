"""Tests for /stop task cancellation."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from nanobot.bus.outbound_events import StreamDeltaEvent, StreamEndEvent


def _make_loop(tmp_path, *, tools_config=None):
    """Create an AgentLoop with the real canonical dependency graph."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.config.schema import ToolsConfig

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    config = tools_config or ToolsConfig()

    # Open the migration guard only for construction.  Turn and dispatch
    # assertions below run with real context/session/dependency implementations.
    with patch("nanobot.migration_guard.require_runtime_ready"):
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=tmp_path,
            tools_config=config,
            timezone="Asia/Kuala_Lumpur",
        )
    return loop, bus


class TestHandleStop:
    @pytest.mark.asyncio
    async def test_stop_no_active_task(self, tmp_path):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop(tmp_path)
        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)
        assert "No active task" in out.content

    @pytest.mark.asyncio
    async def test_stop_cancels_active_task(self, tmp_path):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop(tmp_path)
        cancelled = asyncio.Event()

        async def slow_task():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow_task())
        await asyncio.sleep(0)
        loop._track_active_task("test:c1", task)

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)

        assert cancelled.is_set()
        assert "stopped" in out.content.lower()

    @pytest.mark.asyncio
    async def test_stop_cancels_multiple_tasks(self, tmp_path):
        from nanobot.bus.events import InboundMessage
        from nanobot.command.builtin import cmd_stop
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop(tmp_path)
        events = [asyncio.Event(), asyncio.Event()]

        async def slow(idx):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                events[idx].set()
                raise

        tasks = [asyncio.create_task(slow(i)) for i in range(2)]
        await asyncio.sleep(0)
        for task in tasks:
            loop._track_active_task("test:c1", task)

        msg = InboundMessage(channel="test", sender_id="u1", chat_id="c1", content="/stop")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/stop", loop=loop)
        out = await cmd_stop(ctx)

        assert all(e.is_set() for e in events)
        assert "2 task" in out.content


class TestDispatch:
    def test_exec_tool_not_registered_when_disabled(self, tmp_path):
        from nanobot.agent.tools.shell import ExecToolConfig
        from nanobot.config.schema import ToolsConfig

        loop, _bus = _make_loop(
            tmp_path,
            tools_config=ToolsConfig(exec=ExecToolConfig(enable=False)),
        )

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
    async def test_cli_dispatch_marks_only_the_final_response_as_turn_complete(self, tmp_path):
        from nanobot.agent.turn import TurnResult
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop(tmp_path)
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="marketing", content="hello")

        async def fake_execute(request, *, _options):
            async with loop._admit_after_bus_cleanup(request.session_key):
                async with _options.admission_scope(request.session_key) as io:
                    io.result = TurnResult(
                        run_id="run-cli",
                        outbound=OutboundMessage(channel="cli", chat_id="marketing", content="hi"),
                    )
                    return io.result

        loop.execute_turn = fake_execute

        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        assert out.content == "hi"
        assert out.channel == "cli"
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bus.consume_outbound(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_cli_dispatch_publishes_empty_completion_after_message_tool_delivery(self, tmp_path):
        from nanobot.agent.turn import TurnResult
        from nanobot.bus.events import InboundMessage

        loop, bus = _make_loop(tmp_path)
        msg = InboundMessage(channel="cli", sender_id="u1", chat_id="marketing", content="hello")

        async def fake_execute(request, *, _options):
            async with loop._admit_after_bus_cleanup(request.session_key):
                async with _options.admission_scope(request.session_key) as io:
                    io.result = TurnResult(run_id="run-cli-empty")
                    return io.result

        loop.execute_turn = fake_execute

        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        assert out.content == ""
        assert out.channel == "cli"
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bus.consume_outbound(), timeout=0.05)

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
                    io.result = TurnResult(run_id="run-stream")
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
    async def test_dispatch_preserves_stream_completion_before_final_response(self, tmp_path):
        from nanobot.agent.turn import TurnResult
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop(tmp_path)
        msg = InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="123",
            content="hello",
            metadata={"_wants_stream": True, "message_id": 10},
        )

        async def fake_execute(request, *, _options):
            async with loop._admit_after_bus_cleanup(request.session_key):
                async with _options.admission_scope(request.session_key) as io:
                    assert io.delivery is not None and io.delivery.streaming
                    await io.delivery.events.emit(
                        StreamEndEvent(resuming=True, merge_next=True)
                    )
                    io.result = TurnResult(
                        run_id="run-result",
                        outbound=OutboundMessage(
                            channel="telegram",
                            chat_id="123",
                            content="final answer",
                            metadata={"message_id": 10},
                        ),
                    )
                    return io.result

        loop.execute_turn = fake_execute

        await loop._dispatch(msg)
        stream_end = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        assert isinstance(stream_end.event, StreamEndEvent)
        assert stream_end.event.resuming is True
        assert stream_end.event.merge_next is True
        assert out.content == "final answer"
        assert out.metadata["message_id"] == 10

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bus.consume_outbound(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_dispatch_preserves_stream_merge_boundary_and_final_response(self, tmp_path):
        from nanobot.agent.turn import TurnResult
        from nanobot.bus.events import InboundMessage, OutboundMessage

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
                    await io.delivery.events.emit(StreamDeltaEvent(content="final answer"))
                    await io.delivery.events.emit(
                        StreamEndEvent(resuming=False, merge_next=False)
                    )
                    io.result = TurnResult(
                        run_id="run-merge",
                        outbound=OutboundMessage(
                            channel="matrix",
                            chat_id="!room:matrix.org",
                            content="final answer",
                            metadata={"message_id": 10},
                        ),
                    )
                    return io.result

        loop.execute_turn = fake_execute

        await loop._dispatch(msg)
        stream_delta = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        stream_end = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        assert isinstance(stream_delta.event, StreamDeltaEvent)
        assert stream_delta.content == "final answer"
        assert isinstance(stream_end.event, StreamEndEvent)
        assert stream_end.event.merge_next is False
        assert out.content == "final answer"
        assert out.metadata["message_id"] == 10

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bus.consume_outbound(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_dispatch_delivers_approval_response_from_turn_result(self, tmp_path):
        from nanobot.agent.turn import TurnResult
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop(tmp_path)
        msg = InboundMessage(
            channel="telegram",
            sender_id="u1",
            chat_id="123",
            content="delete files",
            metadata={"_wants_stream": True, "message_id": 10},
        )

        async def fake_execute(request, *, _options):
            async with loop._admit_after_bus_cleanup(request.session_key):
                async with _options.admission_scope(request.session_key) as io:
                    io.result = TurnResult(
                        run_id="run-approval",
                        status="approval_required",
                        outbound=OutboundMessage(
                            channel="telegram",
                            chat_id="123",
                            content="Reply `yes` to continue.",
                            metadata={"message_id": 10},
                        ),
                    )
                    return io.result

        loop.execute_turn = fake_execute

        await loop._dispatch(msg)
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        assert out.content == "Reply `yes` to continue."
        assert out.metadata["message_id"] == 10

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bus.consume_outbound(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_processing_lock_serializes(self, tmp_path):
        from nanobot.agent.turn import TurnResult
        from nanobot.bus.events import InboundMessage, OutboundMessage

        loop, bus = _make_loop(tmp_path)
        order = []

        async def fake_execute(request, *, _options):
            async with loop._admit_after_bus_cleanup(request.session_key):
                async with _options.admission_scope(request.session_key) as io:
                    order.append(f"start-{request.content}")
                    await asyncio.sleep(0.05)
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
        t2 = asyncio.create_task(loop._dispatch(msg2))
        await asyncio.gather(t1, t2)
        assert order == ["start-a", "end-a", "start-b", "end-b"]
