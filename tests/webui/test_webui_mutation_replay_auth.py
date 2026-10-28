"""Per-caller authorization for WebUI mutation cache replay."""

from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace

import pytest
from websockets.datastructures import Headers

from nanobot.session.manager import Session, SessionManager
from nanobot.webui.inbound_commands import WebUICommandRouter
from nanobot.webui.ws_http import GatewayHTTPHandler


class _Connection:
    def __init__(self, name: str, *, peer: str = "127.0.0.1", forwarded: bool = False) -> None:
        self.name = name
        self.remote_address = (peer, 1234)
        values = [("Host", "127.0.0.1:8765")]
        if forwarded:
            values.append(("X-Forwarded-For", "203.0.113.12"))
        self.request = SimpleNamespace(headers=Headers(values))


@pytest.mark.asyncio
async def test_local_operation_replay_authorizes_each_current_connection() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class RecoveryOwner:
        calls = 0

        async def continue_request(self, _payload):
            self.calls += 1
            started.set()
            await release.wait()
            return {
                "workspace_access_request": {
                    "status": "continued",
                    "target": {"path": "/private/project"},
                    "task": {"excerpt": "private original task"},
                },
                "workspace_scope": {"project_path": "/private/project"},
            }

        async def publish(self, *_args):
            return None

    handler = object.__new__(GatewayHTTPHandler)
    handler.config = SimpleNamespace(host="127.0.0.1")
    handler.workspace_access_recovery = RecoveryOwner()
    handler.check_api_token = lambda _request: True
    handler._log = SimpleNamespace(exception=lambda *_args: None)

    async def resolved(self, connection, request, path):
        return await self._handle_workspace_access_action(connection, request, path)

    handler._dispatch_resolved = MethodType(resolved, handler)
    local = _Connection("local")
    reconnect = _Connection("reconnect")
    revoked = _Connection("revoked")
    forwarded = _Connection("forwarded", forwarded=True)
    remote = _Connection("remote", peer="203.0.113.12")
    token_only = _Connection("token-only")
    webui_connections = {local, reconnect, revoked, forwarded, remote}
    received: list[dict[str, object]] = []

    async def send(connection, event, **payload):
        received.append({"connection": connection.name, "event": event, **payload})

    router = WebUICommandRouter(
        SimpleNamespace(webui_send_event=send),
        SimpleNamespace(
            http=handler,
            endpoint=SimpleNamespace(webui_connections=webui_connections),
            media=None,
            ingress=None,
            transcripts=None,
            workspaces=None,
            temporary_chats=None,
            session_projection=None,
            session_manager=None,
        ),
    )
    envelope = {
        "request_id": "local-continue",
        "action": "workspace.access.continue",
        "payload": {
            "chat_id": "chat-a",
            "access_request_id": "0" * 32,
            "expected_revision": 2,
        },
    }

    await router.start_webui_request(local, envelope)
    await asyncio.wait_for(started.wait(), timeout=2)

    # An authenticated forwarded WebUI connection cannot join the in-flight
    # local result, even when it knows the exact request ID and payload.
    await router.start_webui_request(forwarded, envelope)
    # Non-WebUI token-only and remote WebUI connections also cannot receive it.
    await router.start_webui_request(token_only, envelope)
    await router.start_webui_request(remote, {**envelope, "request_id": "remote-fresh"})

    release.set()
    await asyncio.gather(*tuple(router.request_tasks.values()))

    # A valid local reconnect may replay the completed result without repeating
    # the recovery action.
    await router.start_webui_request(reconnect, envelope)
    await asyncio.gather(*tuple(router.request_tasks.values()))

    # Membership is checked again after an operation completes but before a
    # queued delivery sends its cached private result.
    await router.start_webui_request(revoked, envelope)
    webui_connections.remove(revoked)
    await asyncio.gather(*tuple(router.request_tasks.values()))

    local_responses = [item for item in received if item.get("event") == "webui_response"]
    local_result = next(item for item in local_responses if item.get("connection") == "local")
    reconnect_result = next(
        item for item in local_responses if item.get("connection") == "reconnect"
    )
    revoked_result = next(
        item for item in local_responses if item.get("connection") == "revoked"
    )
    forwarded_result = next(
        item for item in local_responses if item.get("connection") == "forwarded"
    )
    token_result = next(
        item for item in local_responses if item.get("connection") == "token-only"
    )
    remote_result = next(
        item for item in local_responses
        if item.get("connection") == "remote" and item.get("request_id") == "remote-fresh"
    )

    assert local_result["ok"] is True
    assert reconnect_result["ok"] is True
    assert revoked_result["ok"] is False
    assert revoked_result["error"]["message"] == "access_denied"
    assert "private original task" in repr(local_result["result"])
    assert "private original task" in repr(reconnect_result["result"])
    assert forwarded_result["ok"] is False
    assert forwarded_result["error"]["message"] == "local_ui_required"
    assert token_result["ok"] is False
    assert token_result["error"]["message"] == "access_denied"
    assert remote_result["ok"] is False
    assert remote_result["error"]["message"] == "local_ui_required"
    assert "private original task" not in repr(forwarded_result)
    assert "private original task" not in repr(revoked_result)
    assert handler.workspace_access_recovery.calls == 1


@pytest.mark.asyncio
async def test_permission_operation_replay_is_bound_to_canonical_source_lifetime(
    tmp_path,
) -> None:
    home = tmp_path / "agent"
    home.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    chat_id = "source-bound-chat"
    session_key = f"websocket:{chat_id}"

    def create_canonical_session() -> None:
        session = Session(session_key, metadata={"webui": True})
        sessions._jsonl_store.save(session)
        sessions._jsonl_store.source.upgrade_source_session(
            session_key,
            expected_revision=session.revision,
            expected_generation=session.generation,
            expected_incarnation=session.incarnation,
        )

    create_canonical_session()
    handler = SimpleNamespace(
        authorize_webui_mutation=lambda *_args: None,
    )
    dispatches = 0

    async def dispatch(_connection, _action, _payload, *, expected_source_binding=None):
        nonlocal dispatches
        dispatches += 1
        return SimpleNamespace(
            status_code=200,
            body=b'{"ack":"original"}',
            reason_phrase="OK",
        )

    handler.dispatch_webui_mutation = dispatch
    connected = _Connection("local")
    webui_connections = {connected}
    frames: list[dict[str, object]] = []

    async def send(_connection, event, **payload):
        frames.append({"event": event, **payload})

    router = WebUICommandRouter(
        SimpleNamespace(webui_send_event=send),
        SimpleNamespace(
            http=handler,
            endpoint=SimpleNamespace(webui_connections=webui_connections),
            media=None,
            ingress=None,
            transcripts=None,
            workspaces=None,
            temporary_chats=None,
            session_projection=None,
            session_manager=sessions,
        ),
    )
    envelope = {
        "request_id": "source-bound-once",
        "action": "workspace.access.allow_once",
        "payload": {"chat_id": chat_id, "access_request_id": "0" * 32},
    }

    async def send_and_wait() -> dict[str, object]:
        before = len(frames)
        await router.start_webui_request(connected, envelope)
        await asyncio.gather(*tuple(router.request_tasks.values()))
        return frames[before]

    first = await send_and_wait()
    same_source_replay = await send_and_wait()
    assert first["ok"] is True
    assert same_source_replay["result"] == first["result"]
    assert dispatches == 1

    sessions.reset_session(session_key)
    assert router._permission_source_binding(envelope["payload"]) != (
        router.request_operations["source-bound-once"].source_binding
    )
    reset_replay = await send_and_wait()
    assert reset_replay["ok"] is False, (reset_replay, router.request_operations["source-bound-once"].source_binding)
    assert reset_replay["error"]["status"] == 409
    assert reset_replay["error"]["message"] == "source_changed"
    assert dispatches == 1

    assert sessions.delete_session(session_key)
    create_canonical_session()
    recreated_replay = await send_and_wait()
    assert recreated_replay["ok"] is False
    assert recreated_replay["error"]["message"] == "source_changed"
    assert "original" not in repr(recreated_replay)
    assert dispatches == 1


@pytest.mark.asyncio
async def test_permission_result_is_not_delivered_after_source_changes_in_flight(
    tmp_path,
) -> None:
    home = tmp_path / "agent"
    home.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    chat_id = "source-pending-delivery"
    session_key = f"websocket:{chat_id}"
    session = Session(session_key, metadata={"webui": True})
    sessions._jsonl_store.save(session)
    sessions._jsonl_store.source.upgrade_source_session(
        session_key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    started = asyncio.Event()
    release = asyncio.Event()
    dispatches = 0

    async def dispatch(_connection, _action, _payload, *, expected_source_binding=None):
        nonlocal dispatches
        dispatches += 1
        started.set()
        await release.wait()
        return SimpleNamespace(status_code=200, body=b'{"ack":"private-original"}', reason_phrase="OK")

    handler = SimpleNamespace(
        authorize_webui_mutation=lambda *_args: None,
        dispatch_webui_mutation=dispatch,
    )
    connection = _Connection("local")
    frames: list[dict[str, object]] = []

    async def send(_connection, event, **payload):
        frames.append({"event": event, **payload})

    router = WebUICommandRouter(
        SimpleNamespace(webui_send_event=send),
        SimpleNamespace(
            http=handler,
            endpoint=SimpleNamespace(webui_connections={connection}),
            media=None,
            ingress=None,
            transcripts=None,
            workspaces=None,
            temporary_chats=None,
            session_projection=None,
            session_manager=sessions,
        ),
    )
    envelope = {
        "request_id": "source-pending",
        "action": "workspace.access.allow_once",
        "payload": {"chat_id": chat_id, "access_request_id": "0" * 32},
    }
    await router.start_webui_request(connection, envelope)
    await asyncio.wait_for(started.wait(), timeout=2)
    assert sessions.reset_session(session_key) is not None
    release.set()
    await asyncio.gather(*tuple(router.request_tasks.values()))

    assert dispatches == 1
    assert len(frames) == 1
    assert frames[0]["ok"] is False
    assert frames[0]["error"]["status"] == 409
    assert frames[0]["error"]["message"] == "source_changed"
    assert "private-original" not in repr(frames[0])
