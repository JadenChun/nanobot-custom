"""Test cmd_stop drains pending queue to prevent mid-turn injection deadlock."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.command.builtin import cmd_stop
from nanobot.command.router import CommandContext
from nanobot.webui.metadata import WEBUI_TARGET_TURN_METADATA_KEY


@pytest.mark.asyncio
async def test_cmd_stop_drains_pending_queue():
    """cmd_stop should drain pending queue in addition to cancelling active tasks."""
    mock_loop = MagicMock()
    mock_loop._cancel_active_tasks = AsyncMock(return_value=1)
    mock_loop._pending_queues = {}

    pending = asyncio.Queue()
    await pending.put("msg1")
    await pending.put("msg2")
    mock_loop._pending_queues["test-session"] = pending

    ctx = CommandContext(
        msg=MagicMock(channel="websocket", chat_id="test-chat", metadata={}),
        session=None,
        key="test-session",
        raw="/stop",
        loop=mock_loop,
    )

    result = await cmd_stop(ctx)

    assert isinstance(result, OutboundMessage)
    assert "Stopped 3 task(s)" in result.content  # 1 cancelled + 2 drained
    assert "test-session" not in mock_loop._pending_queues


@pytest.mark.asyncio
async def test_cmd_stop_with_empty_pending_queue():
    """cmd_stop should work correctly when pending queue is empty."""
    mock_loop = MagicMock()
    mock_loop._cancel_active_tasks = AsyncMock(return_value=2)
    mock_loop._pending_queues = {}

    pending = asyncio.Queue()
    mock_loop._pending_queues["test-session"] = pending

    ctx = CommandContext(
        msg=MagicMock(channel="websocket", chat_id="test-chat", metadata={}),
        session=None,
        key="test-session",
        raw="/stop",
        loop=mock_loop,
    )

    result = await cmd_stop(ctx)

    assert "Stopped 2 task(s)" in result.content
    assert "test-session" not in mock_loop._pending_queues


@pytest.mark.asyncio
async def test_cmd_stop_no_pending_queue():
    """cmd_stop should work when no pending queue exists."""
    mock_loop = MagicMock()
    mock_loop._cancel_active_tasks = AsyncMock(return_value=0)
    mock_loop._pending_queues = {}

    ctx = CommandContext(
        msg=MagicMock(channel="websocket", chat_id="test-chat", metadata={}),
        session=None,
        key="test-session",
        raw="/stop",
        loop=mock_loop,
    )

    result = await cmd_stop(ctx)

    assert "No active task to stop" in result.content


def _make_loop_with_publisher():
    mock_loop = MagicMock()
    mock_loop._cancel_active_tasks = AsyncMock(return_value=1)
    mock_loop._pending_queues = {}
    mock_loop.runtime_event_publisher = MagicMock(run_status_changed=AsyncMock())
    return mock_loop


@pytest.mark.asyncio
async def test_cmd_stop_rejects_stale_target_without_cancelling(monkeypatch):
    """cmd_stop should ignore a target turn that is no longer active."""
    mock_loop = _make_loop_with_publisher()
    monkeypatch.setattr(
        "nanobot.session.webui_turns.websocket_turn_id",
        lambda chat_id: "turn-new",
    )

    ctx = CommandContext(
        msg=MagicMock(
            channel="websocket",
            chat_id="test-chat",
            metadata={WEBUI_TARGET_TURN_METADATA_KEY: "turn-old"},
        ),
        session=None,
        key="test-session",
        raw="/stop",
        loop=mock_loop,
    )

    result = await cmd_stop(ctx)

    assert "No active task" in result.content
    mock_loop._cancel_active_tasks.assert_not_awaited()
    mock_loop.runtime_event_publisher.run_status_changed.assert_not_awaited()


@pytest.mark.asyncio
async def test_cmd_stop_projects_stopping_and_interrupted_for_matching_target(monkeypatch):
    """cmd_stop should project stopping then interrupted for the active target turn."""
    mock_loop = _make_loop_with_publisher()
    monkeypatch.setattr(
        "nanobot.session.webui_turns.websocket_turn_id",
        lambda chat_id: "turn-a",
    )

    ctx = CommandContext(
        msg=MagicMock(
            channel="websocket",
            chat_id="test-chat",
            metadata={WEBUI_TARGET_TURN_METADATA_KEY: "turn-a"},
        ),
        session=None,
        key="test-session",
        raw="/stop",
        loop=mock_loop,
    )

    await cmd_stop(ctx)

    mock_loop._cancel_active_tasks.assert_awaited_once()
    statuses = [
        call.args[2] if len(call.args) > 2 else call.kwargs["status"]
        for call in mock_loop.runtime_event_publisher.run_status_changed.await_args_list
    ]
    assert statuses == ["stopping", "interrupted"]


@pytest.mark.asyncio
async def test_cmd_stop_without_target_keeps_existing_behavior():
    """cmd_stop without a target should keep cancelling active tasks."""
    mock_loop = _make_loop_with_publisher()

    ctx = CommandContext(
        msg=MagicMock(channel="websocket", chat_id="test-chat", metadata={}),
        session=None,
        key="test-session",
        raw="/stop",
        loop=mock_loop,
    )

    await cmd_stop(ctx)

    mock_loop._cancel_active_tasks.assert_awaited_once()
