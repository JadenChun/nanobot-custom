"""Live Stop expectations cross the existing authenticated gateway seams."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kohaku.agent.stop_target import StopExpectation
from kohaku.bus.events import (
    INBOUND_META_STOP_CLIENT,
    INBOUND_META_STOP_EXPECTATION,
    OutboundMessage,
)
from kohaku.bus.outbound_events import StopTargetEvent, WebUIAdmissionRejectedEvent
from kohaku.bus.queue import MessageBus
from kohaku.channels.manager import ChannelManager
from kohaku.channels.websocket.runtime import _parse_envelope
from kohaku.channels.websocket.tests.test_websocket_channel import (
    _authenticate_local_webui_connection,
    _ch,
)
from kohaku.channels.websocket.tests.ws_test_client import InProcessHttpChannel, http_get
from kohaku.config.schema import Config
from kohaku.session import webui_turns as turns
from kohaku.session.manager import SessionManager
from kohaku.webui.outbound_projection import WebUIOutboundProjector
from kohaku.webui.session_projection import WebUISessionProjection
from tests.agent.test_stop_expectation import target_value
from tests.webui.test_child_thread_route import _auth_headers, _thread_url


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr("kohaku.config.paths.get_data_dir", lambda: tmp_path)
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
    run = {
        "state": "waiting_approval",
        "turn_id": "live-turn",
        "run_id": "live-run",
        "source_incarnation": "live-source",
        "started_at": 100,
        "can_stop": True,
        "can_steer": True,
    }

    def current(key, *, source_snapshot=None):
        if source_snapshot is not None:
            assert source_snapshot.state.session_key == key
        keys.append(key)
        return target

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = SessionManager(workspace, sessions_root=tmp_path / "sessions")
    sessions.upsert_session_metadata("websocket:exact-stop", {"webui": True}, create_if_absent=True)
    # A chat carries modern Stop authority only when its source can prove it.
    seeded = sessions.get_or_create("websocket:exact-stop")
    sessions.save(seeded)
    sessions._jsonl_store.source.upgrade_source_session(
        "websocket:exact-stop",
        expected_revision=seeded.revision,
        expected_generation=seeded.generation,
        expected_incarnation=seeded.incarnation,
    )
    config = Config.model_validate({
        "agents": {"defaults": {"workspace": str(workspace)}},
        "channels": {"websocket": {"enabled": True, "websocketRequiresToken": False}},
    })
    manager = await ChannelManager(
        config, MessageBus(), session_manager=sessions, webui_static_dist=False,
        webui_stop_target_for_key=current,
        webui_run_state_for_key=lambda _key: run,
    ).acquire()
    channel = manager.channels["websocket"]
    projection = channel.gateway.session_projection
    attached = projection.attach_fields("websocket:exact-stop")
    assert attached["stop_target"] == target
    assert attached["run_state"] == run
    hydration = projection.hydration_events("websocket:exact-stop", "exact-stop")
    assert next(frame["stop_target"] for frame in hydration if "stop_target" in frame) == target
    assert next(frame["run_state"] for frame in hydration if "run_state" in frame) == run
    http_channel = InProcessHttpChannel(channel.config, manager.bus, gateway=channel.gateway)
    try:
        response = await http_get(_thread_url(http_channel, "websocket:exact-stop"), headers=_auth_headers(http_channel))
        assert response.status_code == 200, response.text
        assert response.json()["stop_target"] == target
        assert response.json()["run_state"] == run
        assert keys == ["websocket:exact-stop"] * 3
    finally:
        await http_channel.stop()
        await manager.stop_all()


class _StopCaseLoop:
    """Minimal owner surface for the repeated-Stop guard."""

    def __init__(self, current_target):
        self.current_target = current_target
        self.cancelled: list[tuple[object, dict[str, Any]]] = []

    def stop_target_for_key(self, key: str):
        return self.current_target

    async def _cancel_active_tasks(self, key: str, **kwargs: Any) -> int:
        self.cancelled.append((key, kwargs))
        return 1


@pytest.mark.parametrize("with_expectation", [True, False])
async def test_repeated_stop_after_settlement_does_not_stop_a_later_turn(with_expectation):
    from kohaku.agent.stop_target import parse_stop_expectation
    from kohaku.bus.events import InboundMessage
    from kohaku.command.builtin import cmd_stop
    from kohaku.command.router import CommandContext

    settled = target_value()
    later = {**target_value(), "active_turn_id": "turn-later"}
    turns.register_queued_websocket_turn_if_idle("exact-stop", "turn-later")
    loop = _StopCaseLoop(later)
    metadata: dict[str, Any] = {
        INBOUND_META_STOP_CLIENT: True,
        "webui_turn_id": "stop-attempt",
        "webui_target_turn_id": "turn-earlier",
    }
    if with_expectation:
        metadata[INBOUND_META_STOP_EXPECTATION] = parse_stop_expectation(settled)
    message = InboundMessage(
        channel="websocket",
        sender_id="webui",
        chat_id="exact-stop",
        content="/stop",
        metadata=metadata,
    )

    reply = await cmd_stop(CommandContext(
        msg=message, session=None, key=message.session_key, raw="/stop", loop=loop,
    ))

    assert isinstance(reply.event, WebUIAdmissionRejectedEvent)
    assert reply.event.reason == "stale_stop_target"
    assert loop.cancelled == []
    assert turns.websocket_turn_id("exact-stop") == "turn-later"
