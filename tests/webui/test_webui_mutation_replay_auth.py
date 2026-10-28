"""Per-caller authorization for WebUI mutation cache replay."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
from websockets.datastructures import Headers

from nanobot.security.folder_grants import FolderGrantStore
from nanobot.security.workspace_access import default_workspace_scope
from nanobot.session.manager import Session, SessionManager
from nanobot.session.runtime_authority import DurableSessionAuthority, SessionAuthorityLease
from nanobot.webui.inbound_commands import WebUICommandRouter
from nanobot.webui.workspaces import (
    WebUIWorkspaceController,
    WorkspacePermissionActionError,
)
from nanobot.webui.ws_http import GatewayHTTPHandler


class _Connection:
    def __init__(self, name: str, *, peer: str = "127.0.0.1", forwarded: bool = False) -> None:
        self.name = name
        self.remote_address = (peer, 1234)
        values = [("Host", "127.0.0.1:8765")]
        if forwarded:
            values.append(("X-Forwarded-For", "203.0.113.12"))
        self.request = SimpleNamespace(headers=Headers(values))


async def _managed_prepare_router(
    tmp_path: Path,
    chat_id: str,
    *,
    publish_event: Callable[..., Awaitable[None]] | None = None,
) -> tuple[
    SessionAuthorityLease,
    SessionManager,
    WebUIWorkspaceController,
    WebUICommandRouter,
    _Connection,
    _Connection,
    list[dict[str, object]],
]:
    home = tmp_path / "agent"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    lease = await DurableSessionAuthority(
        home,
        sessions_root=tmp_path / "sessions",
    ).acquire()
    sessions = lease.manager
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=FolderGrantStore(
            agent_home=home,
            instance_root=instance,
            path=instance / "webui" / "folder-grants.json",
        ),
        native_supervisor_available=True,
    )
    controller.stage_scope(chat_id, default_workspace_scope(project, True))

    async def idle_publish(*_args: object, **_kwargs: object) -> None:
        return None

    handler = object.__new__(GatewayHTTPHandler)
    handler.config = SimpleNamespace(host="127.0.0.1")
    handler.session_manager = sessions
    handler.workspaces = controller
    handler.check_api_token = lambda _request: True
    handler._log = SimpleNamespace(exception=lambda *_args: None)
    handler.bus = SimpleNamespace(publish_event=publish_event or idle_publish)

    async def dispatch(self, connection, request, path):
        return await self._handle_workspace_permission_action(connection, request, path)

    handler._dispatch_resolved = MethodType(dispatch, handler)
    connection = _Connection("local")
    joined = _Connection("joined")
    connections = {connection, joined}
    frames: list[dict[str, object]] = []

    async def send(target, event, **payload: object) -> None:
        frames.append({"connection": target.name, "event": event, **payload})

    router = WebUICommandRouter(
        SimpleNamespace(webui_send_event=send),
        SimpleNamespace(
            http=handler,
            endpoint=SimpleNamespace(webui_connections=connections),
            media=None,
            ingress=None,
            transcripts=None,
            workspaces=controller,
            temporary_chats=None,
            session_projection=None,
            session_manager=sessions,
        ),
    )
    return lease, sessions, controller, router, connection, joined, frames


async def _send_mutation(
    router: WebUICommandRouter,
    connection: _Connection,
    frames: list[dict[str, object]],
    envelope: dict[str, object],
) -> dict[str, object]:
    before = len(frames)
    await router.start_webui_request(connection, envelope)
    pending = tuple(router.request_tasks.values())
    if pending:
        await asyncio.gather(*pending)
    assert len(frames) > before
    return frames[before]


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


@pytest.mark.asyncio
async def test_staged_prepare_replay_stays_cached_and_full_can_be_enabled_before_first_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease, sessions, controller, router, connection, reconnect, frames = await _managed_prepare_router(
        tmp_path,
        "staged-prepare-chat",
    )
    chat_id = "staged-prepare-chat"
    key = f"websocket:{chat_id}"
    prepare_calls = 0
    prepare = controller.prepare_full

    def counted_prepare(payload: object, *, local_ui: bool):
        nonlocal prepare_calls
        prepare_calls += 1
        return prepare(payload, local_ui=local_ui)

    controller.prepare_full = counted_prepare
    prepare_envelope = {
        "request_id": "staged-prepare",
        "action": "workspace.access.prepare_full",
        "payload": {"chat_id": chat_id},
    }

    try:
        prepared = await _send_mutation(router, connection, frames, prepare_envelope)
        replayed = await _send_mutation(router, connection, frames, prepare_envelope)
        reconnect_replay = await _send_mutation(router, reconnect, frames, prepare_envelope)
        assert prepared["ok"] is True
        assert replayed["ok"] is True
        assert reconnect_replay["ok"] is True
        assert replayed["result"] == prepared["result"]
        assert reconnect_replay["result"] == prepared["result"]
        assert prepare_calls == 1

        result = prepared.get("result")
        assert isinstance(result, dict)
        permissions = result.get("workspace_permissions")
        assert isinstance(permissions, dict)
        runtime_id = permissions.get("runtime_id")
        revision = permissions.get("revision")
        assert isinstance(runtime_id, str)
        assert type(revision) is int
        assert permissions.get("mode") == "restricted"
        backend = sessions.source_backend(key)
        assert backend is not None
        assert backend.source.load_source_snapshot(key).events == []
        assert controller.ingress_full_reference(chat_id, local_ui=True) is None

        enabled = await _send_mutation(router, connection, frames, {
            "request_id": "staged-prepare-enable",
            "action": "workspace.access.mode.set",
            "payload": {
                "chat_id": chat_id,
                "mode": "full",
                "expected_runtime_id": runtime_id,
                "expected_permission_revision": revision,
            },
        })
        assert enabled["ok"] is True
        assert controller.ingress_full_reference(chat_id, local_ui=True) is not None
        assert backend.source.load_source_snapshot(key).events == []
        assert prepare_calls == 1

        monkeypatch.setattr(controller, "_source_binding", lambda _key: None)
        failed_read_replay = await _send_mutation(router, connection, frames, prepare_envelope)
        assert failed_read_replay.get("ok") is False
        failed_read_error = failed_read_replay.get("error")
        assert isinstance(failed_read_error, dict)
        assert failed_read_error.get("message") == "source_changed"
        assert prepare_calls == 1
    finally:
        if lease.close is not None:
            await lease.close()


@pytest.mark.asyncio
async def test_prepare_original_join_and_cached_replay_fail_after_reset(
    tmp_path: Path,
) -> None:
    published = asyncio.Event()
    release = asyncio.Event()

    async def gated_publish(*_args: object, **_kwargs: object) -> None:
        published.set()
        await release.wait()

    lease, sessions, controller, router, connection, joined, frames = await _managed_prepare_router(
        tmp_path,
        "prepare-reset-chat",
        publish_event=gated_publish,
    )
    chat_id = "prepare-reset-chat"
    key = f"websocket:{chat_id}"
    prepare_calls = 0
    prepare = controller.prepare_full

    def counted_prepare(payload: object, *, local_ui: bool):
        nonlocal prepare_calls
        prepare_calls += 1
        return prepare(payload, local_ui=local_ui)

    controller.prepare_full = counted_prepare
    envelope = {
        "request_id": "prepare-before-reset",
        "action": "workspace.access.prepare_full",
        "payload": {"chat_id": chat_id},
    }

    try:
        await router.start_webui_request(connection, envelope)
        await asyncio.wait_for(published.wait(), timeout=3)
        # A second local caller joins the already running request before its
        # event publication await is released.
        await router.start_webui_request(joined, envelope)
        old_source = controller._source_binding(key)
        assert old_source is not None
        assert sessions.reset_session(key) is not None
        new_source = controller._source_binding(key)
        assert new_source is not None and new_source != old_source
        release.set()
        await asyncio.gather(*tuple(router.request_tasks.values()))

        joined_results = [
            frame for frame in frames
            if frame.get("event") == "webui_response"
            and frame.get("request_id") == "prepare-before-reset"
        ]
        assert len(joined_results) == 2
        for frame in joined_results:
            assert frame.get("ok") is False
            error = frame.get("error")
            assert isinstance(error, dict)
            assert error.get("message") in {"permission_changed", "source_changed"}
            assert "workspace_permissions" not in repr(frame)
        assert prepare_calls == 1

        # The exact request ID replay uses its completed cached operation and
        # cannot refresh or repeat the original prepare projection.
        replayed = await _send_mutation(router, connection, frames, envelope)
        assert replayed.get("ok") is False
        replay_error = replayed.get("error")
        assert isinstance(replay_error, dict)
        assert replay_error.get("message") in {"permission_changed", "source_changed"}
        assert prepare_calls == 1

        # The cached operation still holds the original projection privately;
        # it cannot confirm Full after the source lifetime has changed.
        original_result = router.request_operations["prepare-before-reset"].task.result().result
        assert isinstance(original_result, dict)
        original_permissions = original_result.get("workspace_permissions")
        assert isinstance(original_permissions, dict)
        with pytest.raises(WorkspacePermissionActionError, match="permission_changed"):
            controller.set_permission_mode({
                "chat_id": chat_id,
                "mode": "full",
                "expected_runtime_id": original_permissions.get("runtime_id"),
                "expected_permission_revision": original_permissions.get("revision"),
            }, local_ui=True)
        assert controller.ingress_full_reference(chat_id, local_ui=True) is None
        new_prepare = await _send_mutation(router, connection, frames, {
            "request_id": "prepare-after-reset",
            "action": "workspace.access.prepare_full",
            "payload": {"chat_id": chat_id},
        })
        assert new_prepare.get("ok") is True
        new_result = new_prepare.get("result")
        assert isinstance(new_result, dict)
        new_permissions = new_result.get("workspace_permissions")
        assert isinstance(new_permissions, dict)
        enabled = await _send_mutation(router, connection, frames, {
            "request_id": "enable-after-reset",
            "action": "workspace.access.mode.set",
            "payload": {
                "chat_id": chat_id,
                "mode": "full",
                "expected_runtime_id": new_permissions.get("runtime_id"),
                "expected_permission_revision": new_permissions.get("revision"),
            },
        })
        assert enabled.get("ok") is True
        assert controller.ingress_full_reference(chat_id, local_ui=True) is not None
        assert prepare_calls == 2

        assert sessions.delete_session(key)
        deleted_replay = await _send_mutation(router, connection, frames, envelope)
        assert deleted_replay.get("ok") is False
        deleted_error = deleted_replay.get("error")
        assert isinstance(deleted_error, dict)
        assert deleted_error.get("message") == "source_changed"
        assert prepare_calls == 2
    finally:
        release.set()
        if lease.close is not None:
            await lease.close()


@pytest.mark.asyncio
async def test_prepare_admission_binding_is_checked_before_controller_execution(
    tmp_path: Path,
) -> None:
    lease, sessions, controller, router, connection, _, frames = await _managed_prepare_router(
        tmp_path,
        "prepare-existing-chat",
    )
    existing_chat = "prepare-existing-chat"
    existing_key = f"websocket:{existing_chat}"
    scope = controller.scope_for_new_chat({}, controls_available=True)
    sessions.admit_persistent_write(existing_key)
    controller.persist_scope(existing_chat, scope, create_if_absent=True)
    prepare_calls = 0
    prepare = controller.prepare_full

    def counted_prepare(payload: object, *, local_ui: bool):
        nonlocal prepare_calls
        prepare_calls += 1
        return prepare(payload, local_ui=local_ui)

    controller.prepare_full = counted_prepare
    lock = router.request_locks.setdefault(connection, asyncio.Lock())

    try:
        await lock.acquire()
        try:
            await router.start_webui_request(connection, {
                "request_id": "prepare-existing-queued",
                "action": "workspace.access.prepare_full",
                "payload": {"chat_id": existing_chat},
            })
            assert sessions.reset_session(existing_key) is not None
        finally:
            lock.release()
        await asyncio.gather(*tuple(router.request_tasks.values()))
        existing_result = next(
            frame for frame in frames
            if frame.get("request_id") == "prepare-existing-queued"
        )
        assert existing_result.get("ok") is False
        assert prepare_calls == 0

        absent_chat = "prepare-absent-queued"
        controller.stage_scope(absent_chat, scope)
        await lock.acquire()
        try:
            await router.start_webui_request(connection, {
                "request_id": "prepare-absent-queued",
                "action": "workspace.access.prepare_full",
                "payload": {"chat_id": absent_chat},
            })
            # Simulate a competing creator winning while this operation waits
            # behind the connection's queued request lock.
            counted_prepare({"chat_id": absent_chat}, local_ui=True)
        finally:
            lock.release()
        await asyncio.gather(*tuple(router.request_tasks.values()))
        absent_result = next(
            frame for frame in frames
            if frame.get("request_id") == "prepare-absent-queued"
        )
        assert absent_result.get("ok") is False
        assert prepare_calls == 1
    finally:
        if lease.close is not None:
            await lease.close()
