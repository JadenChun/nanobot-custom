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
from nanobot.webui.metadata import WEBUI_TARGET_TURN_METADATA_KEY


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    for name in (
        "_WEBSOCKET_ACTIVE_TURNS",
        "_WEBSOCKET_TURN_WALL_STARTED_AT",
        "_WEBSOCKET_TURN_IDS",
        "_WEBSOCKET_TURN_OWNERS",
    ):
        getattr(wth, name).clear()
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)
    yield
    for name in (
        "_WEBSOCKET_ACTIVE_TURNS",
        "_WEBSOCKET_TURN_WALL_STARTED_AT",
        "_WEBSOCKET_TURN_IDS",
        "_WEBSOCKET_TURN_OWNERS",
    ):
        getattr(wth, name).clear()


@pytest.fixture()
def bus() -> MagicMock:
    instance = MagicMock()
    instance.publish_inbound = AsyncMock()
    return instance


def _payloads(connection: AsyncMock) -> list[dict[str, Any]]:
    return [json.loads(call.args[0]) for call in connection.send.await_args_list]


async def _dispatch(channel, connection: AsyncMock, payload: dict[str, Any]) -> None:
    await channel._dispatch_envelope(connection, "client", payload)


async def test_targeted_stop_reaches_the_owner(bus: MagicMock) -> None:
    channel = _ch(bus)
    connection = AsyncMock()
    _authenticate_local_webui_connection(channel, connection)
    wth.register_queued_websocket_turn_if_idle("chat-stop", "turn-active")

    await _dispatch(
        channel,
        connection,
        {
            "type": "message",
            "chat_id": "chat-stop",
            "content": "/stop",
            "webui": True,
            "request_settings_version": 1,
            "request_intent": "new",
            "target_turn_id": "turn-active",
        },
    )

    assert bus.publish_inbound.await_count == 1
    assert not [
        payload for payload in _payloads(connection) if payload.get("event") == "error"
    ]

    msg = bus.publish_inbound.await_args.args[0]
    assert msg.metadata[WEBUI_TARGET_TURN_METADATA_KEY] == "turn-active"


async def test_targeted_stop_rejects_malformed_target(bus: MagicMock) -> None:
    channel = _ch(bus)
    connection = AsyncMock()
    _authenticate_local_webui_connection(channel, connection)
    wth.register_queued_websocket_turn_if_idle("chat-stop", "turn-active")

    await _dispatch(
        channel,
        connection,
        {
            "type": "message",
            "chat_id": "chat-stop",
            "content": "/stop",
            "webui": True,
            "request_settings_version": 1,
            "request_intent": "new",
            "target_turn_id": 123,
        },
    )

    assert bus.publish_inbound.await_count == 0
    errors = [payload for payload in _payloads(connection) if payload.get("event") == "error"]
    assert len(errors) == 1
    assert errors[0]["detail"] == "invalid_request_settings"
    assert errors[0]["reason"] == "target_turn_id"


async def test_non_stop_new_request_cannot_carry_a_target(bus: MagicMock) -> None:
    channel = _ch(bus)
    connection = AsyncMock()
    _authenticate_local_webui_connection(channel, connection)
    wth.register_queued_websocket_turn_if_idle("chat-stop", "turn-active")

    await _dispatch(
        channel,
        connection,
        {
            "type": "message",
            "chat_id": "chat-stop",
            "content": "hello",
            "webui": True,
            "request_settings_version": 1,
            "request_intent": "new",
            "target_turn_id": "turn-active",
        },
    )

    assert bus.publish_inbound.await_count == 0
    errors = [payload for payload in _payloads(connection) if payload.get("event") == "error"]
    assert len(errors) == 1
    assert errors[0]["detail"] == "invalid_request_settings"
    assert errors[0]["reason"] == "target_turn_id"
