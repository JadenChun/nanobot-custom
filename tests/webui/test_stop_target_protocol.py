"""Live Stop expectations cross the existing authenticated gateway seams."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.stop_target import StopExpectation
from nanobot.bus.events import (
    INBOUND_META_STOP_CLIENT,
    INBOUND_META_STOP_EXPECTATION,
    OutboundMessage,
)
from nanobot.bus.outbound_events import StopTargetEvent, WebUIAdmissionRejectedEvent
from nanobot.bus.queue import MessageBus
from nanobot.channels.manager import ChannelManager
from nanobot.channels.websocket.runtime import _parse_envelope
from nanobot.channels.websocket.tests.test_websocket_channel import (
    _authenticate_local_webui_connection,
    _ch,
)
from nanobot.channels.websocket.tests.ws_test_client import InProcessHttpChannel, http_get
from nanobot.config.schema import Config
from nanobot.session import webui_turns as turns
from nanobot.session.manager import SessionManager
from nanobot.webui.outbound_projection import WebUIOutboundProjector
from nanobot.webui.session_projection import WebUISessionProjection
from tests.agent.test_stop_expectation import target_value
from tests.webui.test_child_thread_route import _auth_headers, _thread_url


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)
    for name in ("_WEBSOCKET_ACTIVE_TURNS", "_WEBSOCKET_TURN_WALL_STARTED_AT", "_WEBSOCKET_TURN_IDS", "_WEBSOCKET_TURN_OWNERS"):
        getattr(turns, name).clear()
    yield
    for name in ("_WEBSOCKET_ACTIVE_TURNS", "_WEBSOCKET_TURN_WALL_STARTED_AT", "_WEBSOCKET_TURN_IDS", "_WEBSOCKET_TURN_OWNERS"):
        getattr(turns, name).clear()


def envelope():
    return {
        "type": "message", "chat_id": "exact-stop", "content": "/stop", "webui": True,
        "turn_id": "attempt-stop", "request_settings_version": 1, "request_intent": "new",
        "stop_expectation": target_value(),
    }


def payloads(connection):
    return [json.loads(call.args[0]) for call in connection.send.await_args_list]


async def test_busy_root_with_null_turn_id_admits_typed_stop_at_real_ingress():
    bus = MagicMock(publish_inbound=AsyncMock())
    channel = _ch(bus)
    connection = AsyncMock()
    _authenticate_local_webui_connection(channel, connection)
    turns.register_queued_websocket_turn_if_idle("exact-stop", "current-root-turn")
    raw = envelope()
    await channel._dispatch_envelope(connection, "client", raw)
    assert not [value for value in payloads(connection) if value.get("event") == "error"]
    bus.publish_inbound.assert_awaited_once()
    message = bus.publish_inbound.await_args.args[0]
    assert message.metadata[INBOUND_META_STOP_CLIENT] is True
    assert type(message.metadata[INBOUND_META_STOP_EXPECTATION]) is StopExpectation
    assert message.metadata[INBOUND_META_STOP_EXPECTATION].to_dict() == raw["stop_expectation"]
    assert message.metadata["webui_turn_id"] == "attempt-stop"


@pytest.mark.parametrize("fault", ["null", "unknown", "bool", "nested-duplicate", "duplicate", "non-stop"])
async def test_malformed_expectation_is_refused_before_publish_with_exact_error(fault):
    bus = MagicMock(publish_inbound=AsyncMock())
    channel = _ch(bus)
    connection = AsyncMock()
    _authenticate_local_webui_connection(channel, connection)
    raw = envelope()
    if fault == "null":
        raw["stop_expectation"] = None
    elif fault == "unknown":
        raw["stop_expectation"]["unexpected"] = True
    elif fault == "bool":
        raw["stop_expectation"]["control_epoch"] = True
    elif fault == "non-stop":
        raw["content"] = "ordinary input"
    else:
        text = json.dumps(raw)
        text = (
            text.replace('"generation": 0', '"generation": 0, "generation": 0')
            if fault == "nested-duplicate" else text.replace('"stop_expectation":', '"stop_expectation": null, "stop_expectation":')
        )
        raw = _parse_envelope(text)
        assert raw is not None
    await channel._dispatch_envelope(connection, "client", raw)
    bus.publish_inbound.assert_not_awaited()
    errors = [value for value in payloads(connection) if value.get("event") == "error"]
    assert errors == [{
        "event": "error", "chat_id": "exact-stop", "turn_id": "attempt-stop",
        "detail": "not_admitted", "reason": "invalid_stop_expectation",
    }]


async def test_same_attempt_echo_reconciles_and_changed_expectation_conflicts():
    bus = MagicMock(publish_inbound=AsyncMock())
    channel = _ch(bus)
    connection = AsyncMock()
    _authenticate_local_webui_connection(channel, connection)
    raw = envelope()
    await channel._dispatch_envelope(connection, "client", raw)
    await channel._dispatch_envelope(connection, "client", deepcopy(raw))
    assert not [value for value in payloads(connection) if value.get("reason") == "attempt_payload_changed"]
    published = bus.publish_inbound.await_count
    changed = deepcopy(raw)
    changed["stop_expectation"]["pending_work_digest"] = "1" * 64
    await channel._dispatch_envelope(connection, "client", changed)
    assert bus.publish_inbound.await_count == published
    errors = [value for value in payloads(connection) if value.get("event") == "error"]
    assert errors[-1] == {
        "event": "error", "chat_id": "exact-stop", "turn_id": "attempt-stop",
        "detail": "not_admitted", "reason": "attempt_payload_changed",
    }


@pytest.mark.parametrize("target", [None, target_value()])
async def test_live_target_hydrates_and_updates_transiently_without_goal_or_history(target):
    sessions = MagicMock()
    sessions.read_session_metadata.return_value = {"metadata": {"stop_target": {"historical": True}}}
    projection = WebUISessionProjection(sessions, stop_target_for_key=lambda key: target)
    assert projection.attach_fields("websocket:exact-stop")["stop_target"] == target
    assert projection.hydration_events("websocket:exact-stop", "exact-stop") == ({
        "event": "stop_target", "chat_id": "exact-stop", "stop_target": target,
    },)
    transport = SimpleNamespace(webui_subscribers=lambda chat_id: (), send_payload=AsyncMock())
    projector = WebUIOutboundProjector(transport, projection)
    await projector.hydrate("exact-stop")
    await projector.send(OutboundMessage(
        channel="websocket", chat_id="exact-stop", content="", event=StopTargetEvent(target),
    ))
    assert transport.send_payload.await_count == 2
    for call in transport.send_payload.await_args_list:
        assert call.args == ("exact-stop", {"event": "stop_target", "chat_id": "exact-stop", "stop_target": target})
        assert call.kwargs == {"persistence": "transient"}
    assert WebUISessionProjection(sessions).attach_fields("websocket:exact-stop")["stop_target"] is None


def test_live_target_hydration_without_session_manager_requires_configured_owner():
    unsupported = WebUISessionProjection(None)
    modern_idle = WebUISessionProjection(None, stop_target_for_key=lambda _key: None)
    assert unsupported.hydration_events("websocket:exact-stop", "exact-stop") == ()
    assert modern_idle.hydration_events("websocket:exact-stop", "exact-stop") == ({
        "event": "stop_target", "chat_id": "exact-stop", "stop_target": None,
    },)
    assert modern_idle.attach_fields("websocket:exact-stop")["stop_target"] is None


async def test_runtime_typed_refusal_uses_existing_error_event_and_turn_identity():
    bus = MagicMock(publish_inbound=AsyncMock())
    channel = _ch(bus)
    connection = AsyncMock()
    _authenticate_local_webui_connection(channel, connection)
    channel.webui_attach(connection, "exact-stop")
    await channel.send(OutboundMessage(
        channel="websocket", chat_id="exact-stop", content="",
        event=WebUIAdmissionRejectedEvent("not_admitted", "stale_stop_target", "attempt-stop"),
    ))
    assert payloads(connection)[-1] == {
        "event": "error", "chat_id": "exact-stop", "turn_id": "attempt-stop",
        "detail": "not_admitted", "reason": "stale_stop_target",
    }


async def test_channel_manager_wires_live_callback_to_attach_hydration_and_thread(tmp_path):
    target = target_value()
    keys = []

    def current(key):
        keys.append(key)
        return target

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = SessionManager(workspace, sessions_root=tmp_path / "sessions")
    sessions.upsert_session_metadata("websocket:exact-stop", {"webui": True}, create_if_absent=True)
    config = Config.model_validate({
        "agents": {"defaults": {"workspace": str(workspace)}},
        "channels": {"websocket": {"enabled": True, "websocketRequiresToken": False}},
    })
    manager = await ChannelManager(
        config, MessageBus(), session_manager=sessions, webui_static_dist=False,
        webui_stop_target_for_key=current,
    ).acquire()
    channel = manager.channels["websocket"]
    projection = channel.gateway.session_projection
    assert projection.attach_fields("websocket:exact-stop")["stop_target"] == target
    assert projection.hydration_events("websocket:exact-stop", "exact-stop")[-1]["stop_target"] == target
    http_channel = InProcessHttpChannel(channel.config, manager.bus, gateway=channel.gateway)
    try:
        response = await http_get(_thread_url(http_channel, "websocket:exact-stop"), headers=_auth_headers(http_channel))
        assert response.status_code == 200, response.text
        assert response.json()["stop_target"] == target
        assert keys == ["websocket:exact-stop"] * 3
    finally:
        await http_channel.stop()
        await manager.stop_all()
