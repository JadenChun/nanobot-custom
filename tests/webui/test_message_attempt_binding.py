"""Regression tests for WebUI message attempt (turn_id) payload binding."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.channels.websocket.tests.test_websocket_channel import (
    _authenticate_local_webui_connection,
    _ch,
)
from nanobot.session import webui_turns as wth


@pytest.fixture(autouse=True)
def _isolate_runtime_state(tmp_path, monkeypatch) -> None:
    """Keep turn state and transcript writes isolated to this test."""
    wth._WEBSOCKET_ACTIVE_TURNS.clear()
    wth._WEBSOCKET_TURN_WALL_STARTED_AT.clear()
    wth._WEBSOCKET_TURN_IDS.clear()
    wth._WEBSOCKET_TURN_OWNERS.clear()
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)
    yield
    wth._WEBSOCKET_ACTIVE_TURNS.clear()
    wth._WEBSOCKET_TURN_WALL_STARTED_AT.clear()
    wth._WEBSOCKET_TURN_IDS.clear()
    wth._WEBSOCKET_TURN_OWNERS.clear()


@pytest.fixture()
def bus() -> MagicMock:
    instance = MagicMock()
    instance.publish_inbound = AsyncMock()
    return instance


def _sent_payloads(connection: AsyncMock) -> list[dict[str, Any]]:
    return [json.loads(call.args[0]) for call in connection.send.await_args_list]


def _conflict_errors(
    connection: AsyncMock,
    *,
    chat_id: str,
    turn_id: str,
) -> list[dict[str, Any]]:
    return [
        payload
        for payload in _sent_payloads(connection)
        if payload.get("event") == "error"
        and payload.get("detail") == "not_admitted"
        and payload.get("reason") == "attempt_payload_changed"
        and payload.get("chat_id") == chat_id
        and payload.get("turn_id") == turn_id
    ]


def _message_envelope(*, chat_id: str, content: str, turn_id: str) -> dict[str, Any]:
    return {
        "type": "message",
        "chat_id": chat_id,
        "content": content,
        "turn_id": turn_id,
        "webui": True,
        "request_settings_version": 1,
    }


@pytest.mark.asyncio
async def test_repeated_turn_id_with_changed_payload_is_rejected_before_publish(
    bus: MagicMock,
) -> None:
    channel = _ch(bus)
    connection = AsyncMock()
    _authenticate_local_webui_connection(channel, connection)

    await channel._dispatch_envelope(
        connection,
        "webui-client",
        _message_envelope(chat_id="chat-a", content="hello", turn_id="turn-1"),
    )
    publishes_after_first = bus.publish_inbound.await_count
    assert _conflict_errors(connection, chat_id="chat-a", turn_id="turn-1") == []

    await channel._dispatch_envelope(
        connection,
        "webui-client",
        _message_envelope(chat_id="chat-a", content="different", turn_id="turn-1"),
    )

    conflicts = _conflict_errors(connection, chat_id="chat-a", turn_id="turn-1")
    assert len(conflicts) == 1
    payload = conflicts[0]
    assert payload["event"] == "error"
    assert payload["detail"] == "not_admitted"
    assert payload["reason"] == "attempt_payload_changed"
    assert payload["chat_id"] == "chat-a"
    assert payload["turn_id"] == "turn-1"

    assert bus.publish_inbound.await_count == publishes_after_first


@pytest.mark.asyncio
async def test_repeated_turn_id_with_identical_payload_is_not_a_payload_conflict(
    bus: MagicMock,
) -> None:
    channel = _ch(bus)
    connection = AsyncMock()
    _authenticate_local_webui_connection(channel, connection)
    envelope = _message_envelope(
        chat_id="chat-a",
        content="hello",
        turn_id="turn-same",
    )

    await channel._dispatch_envelope(connection, "webui-client", envelope)
    await channel._dispatch_envelope(connection, "webui-client", dict(envelope))

    assert _conflict_errors(connection, chat_id="chat-a", turn_id="turn-same") == []


@pytest.mark.asyncio
async def test_same_turn_id_in_another_chat_is_not_a_payload_conflict(
    bus: MagicMock,
) -> None:
    channel = _ch(bus)
    connection = AsyncMock()
    _authenticate_local_webui_connection(channel, connection)

    await channel._dispatch_envelope(
        connection,
        "webui-client",
        _message_envelope(chat_id="chat-a", content="hello", turn_id="turn-shared"),
    )

    await channel._dispatch_envelope(
        connection,
        "webui-client",
        _message_envelope(chat_id="chat-b", content="different", turn_id="turn-shared"),
    )

    assert _conflict_errors(connection, chat_id="chat-b", turn_id="turn-shared") == []
    assert all(
        payload.get("reason") != "attempt_payload_changed"
        for payload in _sent_payloads(connection)
        if payload.get("chat_id") == "chat-b"
    )
