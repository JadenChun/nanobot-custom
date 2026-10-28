"""Permission-v2 routes keep the existing local authenticated mutation boundary."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from nanobot.security.desktop_permissions import WorkspacePermissionConflictError
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.security.workspace_access import default_workspace_scope
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import Session, SessionManager, SessionWriteConflict
from nanobot.session.pending_followups import PendingFollowupRecord
from nanobot.session.webui_turns import (
    clear_websocket_turn_if_current,
    register_queued_websocket_turn_if_idle,
)
from nanobot.session.workspace_access_recovery import (
    WorkspaceAccessActionError,
    WorkspaceAccessRecoveryOwner,
)
from nanobot.webui import ws_http
from nanobot.webui.inbound_commands import WebUICommandRouter
from nanobot.webui.workspaces import WebUIWorkspaceController, WorkspacePermissionActionError
from nanobot.webui.ws_http import WEBSOCKET_MUTATION_ROUTES, GatewayHTTPHandler


@pytest.mark.parametrize(("action", "path"), [
    ("workspace.access.allow_once", "/api/workspaces/access/allow-once"),
    ("workspace.access.mode.set", "/api/workspaces/access/mode"),
    ("workspace.access.prepare_full", "/api/workspaces/access/prepare-full"),
])
def test_permission_v2_actions_have_frozen_local_mutation_routes(action: str, path: str) -> None:
    assert WEBSOCKET_MUTATION_ROUTES[action] == path


def test_prepare_full_fails_closed_for_unknown_or_unmanaged_draft(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agent"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
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
    chat = str(uuid.uuid4())

    with pytest.raises(WorkspacePermissionActionError) as local_only:
        controller.prepare_full({"chat_id": chat}, local_ui=False)
    assert local_only.value.status == 403

    with pytest.raises(WorkspacePermissionActionError) as unknown:
        controller.prepare_full({"chat_id": chat}, local_ui=True)
    assert unknown.value.status == 404

    controller.stage_scope(chat, default_workspace_scope(project, True))
    with pytest.raises(WorkspacePermissionActionError) as unmanaged:
        controller.prepare_full({"chat_id": chat}, local_ui=True)
    assert unmanaged.value.status == 503


def test_real_ordinary_chat_permission_projection_has_no_recovery_record(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agent"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    key = "websocket:ordinary-chat"
    session = Session(key, metadata={
        "webui": True,
        "workspace_scope": default_workspace_scope(project, True).metadata(),
    })
    sessions._jsonl_store.save(session)
    sessions._jsonl_store.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
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
    recovery = WorkspaceAccessRecoveryOwner(sessions, controller)

    projection = recovery.workspace_permissions("ordinary-chat", controls_available=True)

    assert projection["mode"] == "restricted"
    assert projection["state"] == "active"
    assert projection["task_access"] is None
    assert projection["retirement"] is None


def test_real_ordinary_owning_thread_read_projects_permissions_without_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "agent"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    key = "websocket:ordinary-owning-read"
    session = Session(
        key,
        metadata={
            "webui": True,
            "workspace_scope": default_workspace_scope(project, True).metadata(),
        },
    )
    sessions._jsonl_store.save(session)
    sessions._jsonl_store.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    workspaces = WebUIWorkspaceController(
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
    recovery = WorkspaceAccessRecoveryOwner(sessions, workspaces)

    handler = object.__new__(GatewayHTTPHandler)
    handler.config = SimpleNamespace(host="127.0.0.1")
    handler.workspace_controls_available = lambda _connection: True
    handler.check_api_token = lambda _request: True
    handler.session_manager = sessions
    handler.workspaces = workspaces
    handler.workspace_access_recovery = recovery
    handler.media = SimpleNamespace(
        augment_transcript_media=lambda paths, **_kwargs: paths,
        rewrite_local_markdown_images=lambda text, **_kwargs: text,
    )
    handler._log = SimpleNamespace(exception=lambda *_args: None)
    monkeypatch.setattr(ws_http, "build_webui_thread_response", lambda *_args, **_kwargs: {
        "chat_id": "ordinary-owning-read",
        "messages": [],
    })

    request = Request(
        "/api/webui/threads/websocket:ordinary-owning-read",
        Headers([("Host", "127.0.0.1:8765")]),
    )
    connection = SimpleNamespace(
        remote_address=("127.0.0.1", 1234),
        request=SimpleNamespace(headers=request.headers),
    )

    response = handler._handle_webui_thread_get(
        request,
        key,
        connection=connection,
    )

    assert response.status_code == 200, (response.status_code, response.body)
    body = json.loads(response.body)
    assert body["workspace_access_request"] is None
    assert body["workspace_permissions"]["mode"] == "restricted"
    assert body["workspace_permissions"]["state"] == "active"
    assert body["workspace_permissions"]["task_access"] is None


def test_old_generation_once_residual_projects_without_current_request(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agent"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    key = "websocket:old-once-residual"
    session = Session(key, metadata={"webui": True})
    sessions._jsonl_store.save(session)
    sessions._jsonl_store.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    grants = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=grants,
        native_supervisor_available=True,
    )
    owner = sessions.desktop_permissions
    backend = sessions.source_backend(key)
    assert backend is not None
    initial = backend.source.load_source_snapshot(key)
    old_source = {
        "store_identity": initial.state.store_identity,
        "incarnation": initial.state.incarnation,
        "generation": initial.state.generation,
    }
    reference = owner.activate_once(
        session_key=key,
        source_binding=old_source,
        task_turn_id="old-turn",
        task_input_event_id="0" * 32,
        path=project,
        file_access="read_only",
        commands=False,
    )
    reservation = grants.view.try_begin_owned_native_process(owner, reference)
    assert reservation is not None

    # Model a source generation change while an old owner-tagged native tree is
    # still in the durable ledger. The new source has no matching Once marker.
    backend.source.reset_source_session(key, initial.state, [])
    owner.restart()
    recovery = WorkspaceAccessRecoveryOwner(sessions, controller)
    try:
        projection = recovery.workspace_permissions("old-once-residual", controls_available=True)
        assert projection["mode"] == "restricted"
        assert projection["state"] == "retiring"
        assert projection["task_access"] is None
        assert projection["retirement"]["reason"] == "restart"
        assert projection["retirement"]["active_processes"] == 1
        assert projection["retirement"]["unknown_processes"] == 0
    finally:
        assert reservation.cancel_if_unlaunched()


@pytest.mark.parametrize("status", ["cancelled", "invalidated"])
def test_terminal_once_request_keeps_duration_and_expired_state(
    tmp_path: Path,
    status: str,
) -> None:
    home = tmp_path / "agent"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    key = f"websocket:terminal-once-{status}"
    chat_id = key.removeprefix("websocket:")
    session = Session(
        key,
        messages=[{"role": "user", "content": "Read the selected project"}],
        metadata={
            "webui": True,
            "workspace_scope": default_workspace_scope(home, True).metadata(),
        },
    )
    sessions._jsonl_store.save(session)
    sessions._jsonl_store.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    backend = sessions.source_backend(key)
    assert backend is not None
    snapshot = backend.source.load_source_snapshot(key)
    task_input = next(event for event in snapshot.events if event.get("role") == "user")
    binding = {
        "store_identity": snapshot.state.store_identity,
        "incarnation": snapshot.state.incarnation,
        "generation": snapshot.state.generation,
    }
    owner = sessions.desktop_permissions
    task_input_event_id = task_input["_event_id"]
    record: dict[str, Any] = {
        "version": 1,
        "access_request_id": "a" * 32,
        "revision": 2,
        "task": {"turn_id": "terminal-once-turn"},
        "blocked_turn_id": "terminal-once-turn",
        "tool_call_id": None,
        "created_at_ms": 1,
        "status": status,
        "reason": "folder_selection_required",
        "operation": "locate",
        "target": {"kind": "unknown", "path": None},
        "purpose": "Locate the requested project",
        "selected_path": str(project),
        "continuation_turn_id": None,
        "source_binding": binding,
        "task_input_event_id": task_input_event_id,
        "latest_input_event_id": task_input_event_id,
        "blocked_run_id": "b" * 32,
        "expected_scope": {"project_path": str(home), "access_mode": "restricted"},
        "resolved_grants_revision": 0,
        "selection_authority": {
            "kind": "once",
            "runtime_id": owner.runtime_id,
            "authority_epoch": 1,
        },
    }
    backend.source.commit_workspace_access_request(
        key,
        expected_binding=binding,
        expected_record=None,
        replacement_record=record,
        expected_latest_input_event_id=task_input_event_id,
    )
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
    request = WorkspaceAccessRecoveryOwner(sessions, controller).get(chat_id)
    assert request is not None
    assert request["status"] == status
    assert request["selected_duration"] == "once"
    assert request["task_access_state"] == "expired"


def test_allow_once_does_not_reauthorize_while_prior_once_tree_is_active(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agent"
    old_project = tmp_path / "old-project"
    selected_project = tmp_path / "selected-project"
    instance = tmp_path / "instance"
    for directory in (home, old_project, selected_project):
        directory.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    key = "websocket:once-retirement-action"
    chat_id = "once-retirement-action"
    session = Session(
        key,
        messages=[{"role": "user", "content": "Read the selected project"}],
        metadata={
            "webui": True,
            "workspace_scope": default_workspace_scope(home, True).metadata(),
        },
    )
    sessions._jsonl_store.save(session)
    sessions._jsonl_store.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    backend = sessions.source_backend(key)
    assert backend is not None
    snapshot = backend.source.load_source_snapshot(key)
    task_input = next(event for event in snapshot.events if event.get("role") == "user")
    binding = {
        "store_identity": snapshot.state.store_identity,
        "incarnation": snapshot.state.incarnation,
        "generation": snapshot.state.generation,
    }
    turn_id = "once-retirement-turn"
    task_input_event_id = task_input["_event_id"]
    record: dict[str, Any] = {
        "version": 1,
        "access_request_id": "c" * 32,
        "revision": 1,
        "task": {"turn_id": turn_id},
        "blocked_turn_id": turn_id,
        "tool_call_id": None,
        "created_at_ms": 1,
        "status": "pending",
        "reason": "folder_selection_required",
        "operation": "locate",
        "target": {"kind": "unknown", "path": None},
        "purpose": "Locate the requested project",
        "selected_path": None,
        "continuation_turn_id": None,
        "source_binding": binding,
        "task_input_event_id": task_input_event_id,
        "latest_input_event_id": task_input_event_id,
        "blocked_run_id": "d" * 32,
        "expected_scope": {"project_path": str(home), "access_mode": "restricted"},
        "resolved_grants_revision": None,
    }
    backend.source.commit_workspace_access_request(
        key,
        expected_binding=binding,
        expected_record=None,
        replacement_record=record,
        expected_latest_input_event_id=task_input_event_id,
    )
    grants = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=grants,
        native_supervisor_available=True,
    )
    owner = sessions.desktop_permissions
    reference = owner.activate_once(
        session_key=key,
        source_binding=binding,
        task_turn_id=turn_id,
        task_input_event_id=task_input_event_id,
        path=old_project,
        file_access="read_only",
        commands=False,
    )
    owner_tag = owner.native_owner_tag(reference)
    assert owner_tag is not None
    reservation = grants.view.try_begin_owned_native_process(owner, reference)
    assert reservation is not None
    owner.retire_once(
        session_key=key,
        source_binding=binding,
        task_turn_id=turn_id,
        task_input_event_id=task_input_event_id,
        reason="task_cancelled",
        active_turn=False,
        active_processes=0,
        unknown_processes=0,
        process_counts=lambda: grants.view.tagged_native_process_counts(owner_tag),
    )
    recovery = WorkspaceAccessRecoveryOwner(sessions, controller)
    try:
        with pytest.raises(WorkspaceAccessActionError) as rejected:
            recovery.allow_once({
                "chat_id": chat_id,
                "access_request_id": record["access_request_id"],
                "expected_revision": record["revision"],
                "path": str(selected_project),
                "file_access": "read_only",
                "commands": False,
                "expected_grants_revision": controller.grant_snapshot().revision,
                "expected_runtime_id": owner.runtime_id,
                "expected_permission_revision": owner.revision,
            })
        assert rejected.value.status == 409
        assert str(rejected.value) == "native_processes_active"
        latest = backend.source.load_source_snapshot(key)
        assert latest.header["metadata"]["workspace_access_request"] == record
        projection = owner.projection(
            session_key=key,
            source_binding=binding,
            task_turn_id=turn_id,
            task_input_event_id=task_input_event_id,
        )
        assert projection["task_access"]["state"] == "retiring"
        assert projection["retirement"]["active_processes"] == 1
    finally:
        assert reservation.cancel_if_unlaunched()


@pytest.mark.asyncio
async def test_real_mode_conflict_is_serialized_and_cached_as_typed_websocket_frame(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agent"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    chat_id = "mode-frame"
    key = f"websocket:{chat_id}"
    session = Session(
        key,
        messages=[{"role": "user", "content": "Read the project"}],
        metadata={
            "webui": True,
            "workspace_scope": default_workspace_scope(project, True).metadata(),
        },
    )
    sessions._jsonl_store.save(session)
    sessions._jsonl_store.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    grants = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=grants,
        native_supervisor_available=True,
    )
    owner = sessions.desktop_permissions
    source = controller._source_binding(key)
    assert source is not None
    initial = owner.projection(session_key=key, source_binding=source)
    enabled = controller.set_permission_mode({
        "chat_id": chat_id,
        "mode": "full",
        "expected_runtime_id": owner.runtime_id,
        "expected_permission_revision": initial["revision"],
    }, local_ui=True)
    assert enabled["workspace_permissions"]["mode"] == "full"
    reference = owner.full_reference(session_key=key, source_binding=source)
    assert reference is not None
    backend = sessions.source_backend(key)
    assert backend is not None
    source_snapshot = backend.source.load_source_snapshot(key)
    user_event = next(event for event in source_snapshot.events if event.get("role") == "user")
    task_event_id = user_event["_event_id"]
    current_scope = controller.scope_for_session_key(key)
    recovery_record = {
        "version": 1,
        "access_request_id": "e" * 32,
        "revision": 1,
        "task": {"turn_id": "retiring-turn"},
        "blocked_turn_id": "retiring-turn",
        "tool_call_id": None,
        "created_at_ms": 1,
        "status": "pending",
        "reason": "folder_selection_required",
        "operation": "locate",
        "target": {"kind": "unknown", "path": None},
        "purpose": "Locate the requested project",
        "selected_path": None,
        "continuation_turn_id": None,
        "source_binding": source,
        "task_input_event_id": task_event_id,
        "latest_input_event_id": task_event_id,
        "blocked_run_id": "f" * 32,
        "expected_scope": {
            "project_path": str(current_scope.project_path),
            "access_mode": current_scope.access_mode,
        },
        "resolved_grants_revision": None,
    }
    backend.source.commit_workspace_access_request(
        key,
        expected_binding=source,
        expected_record=None,
        replacement_record=recovery_record,
        expected_latest_input_event_id=task_event_id,
    )
    recovery = WorkspaceAccessRecoveryOwner(sessions, controller)
    owner_tag = owner.native_owner_tag(reference)
    assert owner_tag is not None
    reservation = grants.view.try_begin_owned_native_process(owner, reference)
    assert reservation is not None
    closed: list[bool] = []
    assert owner.register_native_stdin(
        reference,
        object(),
        lambda: closed.append(True) or True,
        lambda: grants.view.tagged_native_process_counts(owner_tag),
    )

    class Connection:
        remote_address = ("127.0.0.1", 1234)
        request = SimpleNamespace(headers=Headers([("Host", "127.0.0.1:8765")]))

    connection = Connection()
    connections = {connection}
    frames: list[dict[str, Any]] = []
    dispatches = 0

    async def send(_connection, event: str, **payload: Any) -> None:
        frames.append({"event": event, **payload})

    async def publish_event(*_args: object, **_kwargs: object) -> None:
        return None

    handler = object.__new__(GatewayHTTPHandler)
    handler.config = SimpleNamespace(host="127.0.0.1")
    handler.check_api_token = lambda _request: True
    handler.authorize_webui_mutation = lambda *_args: None
    handler.session_manager = sessions
    handler.workspaces = controller
    handler.bus = SimpleNamespace(publish_event=publish_event)
    handler._log = SimpleNamespace(exception=lambda *_args: None)

    async def dispatch(_connection, request, path):
        nonlocal dispatches
        dispatches += 1
        return await handler._handle_workspace_permission_action(
            _connection, request, path,
        )

    handler._dispatch_resolved = dispatch
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
    current = controller._permission_projection(
        key,
        source,
        can_enable_full=True,
        unavailable_reason=None,
    )
    envelope = {
        "request_id": "full-conflict-frame",
        "action": "workspace.access.mode.set",
        "payload": {
            "chat_id": chat_id,
            "mode": "restricted",
            "expected_runtime_id": owner.runtime_id,
            "expected_permission_revision": current["revision"],
        },
    }
    turn_owner = register_queued_websocket_turn_if_idle(chat_id, "retiring-turn")
    assert turn_owner is not None
    async with acquire_execution(
        sessions._jsonl_store,
        key,
        run_id="full-retirement-turn",
    ):
        await router.start_webui_request(connection, envelope)
        await asyncio.gather(*tuple(router.request_tasks.values()))
    assert clear_websocket_turn_if_current(chat_id, turn_owner)
    first = frames[-1]
    assert first["error"]["workspace_permissions"]["retirement"] == {
        "reason": "mode_changed",
        "active_turn": True,
        "active_processes": 1,
        "unknown_processes": 0,
    }

    retained_projection = recovery.workspace_permissions(
        chat_id,
        controls_available=True,
    )
    assert retained_projection["retirement"] == {
        "reason": "mode_changed",
        "active_turn": False,
        "active_processes": 1,
        "unknown_processes": 0,
    }
    controller_projection = controller._permission_projection(
        key,
        source,
        can_enable_full=True,
        unavailable_reason=None,
    )
    assert retained_projection["retirement"] == controller_projection["retirement"]

    assert reservation.cancel_if_unlaunched()
    quiescent = recovery.workspace_permissions(chat_id, controls_available=True)
    assert quiescent["mode"] == "full"
    assert quiescent["state"] == "retiring"
    assert quiescent["retirement"] == {
        "reason": "mode_changed",
        "active_turn": False,
        "active_processes": 0,
        "unknown_processes": 0,
    }
    assert not owner.authorizes_full(reference, session_key=key, source_binding=source)
    stable_revision = quiescent["revision"]
    assert recovery.workspace_permissions(chat_id, controls_available=True)["revision"] == stable_revision
    assert owner.revision == stable_revision

    await router.start_webui_request(connection, envelope)
    await asyncio.gather(*tuple(router.request_tasks.values()))
    replay = frames[-1]

    for frame in (first, replay):
        assert frame["event"] == "webui_response"
        assert frame["ok"] is False
        error = frame["error"]
        assert error["status"] == 409
        assert error["message"] == "native_processes_active"
        assert error["workspace_permissions"]["state"] == "retiring"
        assert error["workspace_permissions"]["retirement"]["active_processes"] == 1
        assert error["workspace_scope"]["effective_access"]["boundary"] == "folder"
        assert error["workspace_scope"]["effective_access"]["source"] != "full"
    assert replay == first
    assert dispatches == 1
    assert closed == [True]

    latest = controller._permission_projection(
        key,
        source,
        can_enable_full=True,
        unavailable_reason=None,
    )
    retry = {
        "request_id": "full-conflict-retry",
        "action": "workspace.access.mode.set",
        "payload": {
            "chat_id": chat_id,
            "mode": "restricted",
            "expected_runtime_id": owner.runtime_id,
            "expected_permission_revision": latest["revision"],
        },
    }
    await router.start_webui_request(connection, retry)
    await asyncio.gather(*tuple(router.request_tasks.values()))
    assert frames[-1]["ok"] is True
    assert frames[-1]["result"]["workspace_permissions"]["mode"] == "restricted"
    assert dispatches == 2


@pytest.mark.asyncio
async def test_full_retirement_activity_keeps_canonical_pending_followup_busy(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agent"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    key = "websocket:retirement-followup"
    session = Session(key, metadata={"webui": True})
    sessions._jsonl_store.save(session)
    sessions._jsonl_store.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    backend = sessions.source_backend(key)
    assert backend is not None
    workspaces = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=home,
        default_restrict_to_workspace=True,
        native_supervisor_available=True,
    )
    owner = sessions.desktop_permissions
    source = workspaces._source_binding(key)
    assert source is not None
    reference = owner.enable_full(
        session_key=key,
        source_binding=source,
        persistent=True,
        existing=True,
        local=True,
        idle=True,
        configured_sandbox=False,
        supervisor_available=True,
        stores_healthy=True,
    )
    with pytest.raises(WorkspacePermissionConflictError, match="chat_running"):
        owner.disable_full(
            session_key=key,
            source_binding=source,
            active_turn=True,
            active_processes=0,
            unknown_processes=0,
        )
    assert not owner.authorizes_full(reference, session_key=key, source_binding=source)

    followup = PendingFollowupRecord(
        identifier="a" * 32,
        sender_id="local-ui",
        chat_id="retirement-followup",
        content="Please also inspect the changelog",
        media=(),
        metadata={},
    )
    async with acquire_execution(backend, key, run_id="pending-followup") as execution:
        assert backend.source.admit_pending_followup(key, execution, followup)

    projection = workspaces._permission_projection(
        key,
        source,
        can_enable_full=True,
        unavailable_reason=None,
    )
    assert projection["state"] == "retiring"
    assert projection["retirement"]["active_turn"] is True


def test_full_retirement_activity_retains_busy_when_source_cannot_be_rechecked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "agent"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    key = "websocket:retirement-source-unavailable"
    session = Session(key, metadata={"webui": True})
    sessions._jsonl_store.save(session)
    sessions._jsonl_store.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    workspaces = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=home,
        default_restrict_to_workspace=True,
        native_supervisor_available=True,
    )
    owner = sessions.desktop_permissions
    source = workspaces._source_binding(key)
    assert source is not None
    reference = owner.enable_full(
        session_key=key,
        source_binding=source,
        persistent=True,
        existing=True,
        local=True,
        idle=True,
        configured_sandbox=False,
        supervisor_available=True,
        stores_healthy=True,
    )
    with pytest.raises(WorkspacePermissionConflictError, match="chat_running"):
        owner.disable_full(
            session_key=key,
            source_binding=source,
            active_turn=True,
            active_processes=0,
            unknown_processes=0,
        )
    assert not owner.authorizes_full(reference, session_key=key, source_binding=source)
    monkeypatch.setattr(workspaces, "_source_binding", lambda _key: None)

    projection = workspaces._permission_projection(
        key,
        source,
        can_enable_full=True,
        unavailable_reason=None,
    )

    assert projection["state"] == "retiring"
    assert projection["retirement"]["active_turn"] is True


@pytest.mark.parametrize("path", [
    "/api/workspaces/access/allow-once",
    "/api/workspaces/access/mode",
    "/api/workspaces/access/prepare-full",
])
@pytest.mark.asyncio
async def test_permission_v2_routes_reject_direct_http_writes(path: str) -> None:
    handler = object.__new__(GatewayHTTPHandler)
    handler.config = SimpleNamespace(trusted_proxy_auth=None, token_issue_path=None)
    handler.settings_routes = SimpleNamespace(is_mutation_path=lambda _path: False)
    handler._log = SimpleNamespace()
    handler._log_slow_http = lambda *_args: None
    response = await handler.dispatch(
        SimpleNamespace(remote_address=("127.0.0.1", 1234)),
        Request(path, Headers([("Host", "127.0.0.1:8765")])),
    )
    assert response.status_code == 405


@pytest.mark.asyncio
async def test_permission_projection_is_attached_only_to_local_owning_thread_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = SimpleNamespace(project_path="/tmp/project")
    permission_projection = {
        "version": 1,
        "mode": "restricted",
        "state": "active",
        "revision": 7,
    }

    class Recovery:
        def __init__(self) -> None:
            self.permission_calls: list[tuple[str, bool]] = []

        def get(self, chat_id: str) -> dict[str, object]:
            return {"chat_id": chat_id, "status": "ready"}

        def workspace_permissions(
            self,
            chat_id: str,
            *,
            controls_available: bool,
        ) -> dict[str, object]:
            self.permission_calls.append((chat_id, controls_available))
            return permission_projection

        def preview_permission_context(self, _chat_id: str) -> tuple[None, None]:
            return None, None

    recovery = Recovery()
    handler = object.__new__(GatewayHTTPHandler)
    handler.config = SimpleNamespace(host="127.0.0.1")
    handler.workspace_controls_available = lambda _connection: True
    handler.check_api_token = lambda _request: True
    handler.session_manager = SimpleNamespace(
        source_backend=lambda _key: object(),
        read_session_file=lambda _key: {"messages": []},
    )
    handler.workspaces = SimpleNamespace(
        scope_for_session_key=lambda _key: scope,
        scope_payload=lambda _scope, **_kwargs: {"project_path": "/tmp/project"},
        folder_grants_allowed_for_session=lambda _key: False,
    )
    handler.workspace_access_recovery = recovery
    handler.media = SimpleNamespace(
        augment_transcript_media=lambda paths, **_kwargs: paths,
        rewrite_local_markdown_images=lambda text, **_kwargs: text,
    )
    handler._log = SimpleNamespace(exception=lambda *_args: None)
    monkeypatch.setattr(ws_http, "build_webui_thread_response", lambda *_args, **_kwargs: {
        "chat_id": "chat-a",
        "messages": [],
    })

    request = Request(
        "/api/webui/threads/websocket:chat-a",
        Headers([("Host", "127.0.0.1:8765")]),
    )
    local = SimpleNamespace(
        remote_address=("127.0.0.1", 1234),
        request=SimpleNamespace(headers=request.headers),
    )
    local_response = handler._handle_webui_thread_get(
        request,
        "websocket:chat-a",
        connection=local,
    )
    assert local_response.status_code == 200, (local_response.status_code, local_response.body)
    local_body = json.loads(local_response.body)
    assert local_body["workspace_permissions"] == permission_projection
    assert local_body["workspace_access_request"] == {"chat_id": "chat-a", "status": "ready"}
    assert recovery.permission_calls == [("chat-a", True)]

    remote = SimpleNamespace(
        remote_address=("203.0.113.1", 1234),
        request=SimpleNamespace(headers=request.headers),
    )
    remote_response = handler._handle_webui_thread_get(
        request,
        "websocket:chat-a",
        connection=remote,
    )
    remote_body = json.loads(remote_response.body)
    assert "workspace_permissions" not in remote_body
    assert "workspace_access_request" not in remote_body
    assert recovery.permission_calls == [("chat-a", True)]


@pytest.mark.asyncio
async def test_local_thread_request_permissions_and_scope_share_live_once_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "agent"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    sessions_root = tmp_path / "sessions"
    home.mkdir()
    project.mkdir()
    session_manager = SessionManager(home, sessions_root=sessions_root)
    key = "websocket:chat-once"
    scope = default_workspace_scope(project, True)
    session = Session(
        key,
        metadata={"webui": True, "workspace_scope": scope.metadata()},
    )
    backend = session_manager._jsonl_store
    backend.save(session)
    backend.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    workspaces = WebUIWorkspaceController(
        session_manager=session_manager,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=FolderGrantStore(
            agent_home=home,
            instance_root=instance,
            path=instance / "webui" / "folder-grants.json",
        ),
        native_supervisor_available=False,
    )
    owner = session_manager.desktop_permissions
    source = workspaces._source_binding(key)
    assert source is not None
    reference = owner.activate_once(
        session_key=key,
        source_binding=source,
        task_turn_id="turn-once",
        task_input_event_id="event-once",
        path=project,
        file_access="read_only",
        commands=False,
    )
    selected_access = workspaces.focus_owner.effective_access(
        project,
        permission_reference=reference,
    )
    permissions = owner.projection(
        session_key=key,
        source_binding=source,
        task_turn_id="turn-once",
        task_input_event_id="event-once",
    )

    class Recovery:
        def get(self, _chat_id: str) -> dict[str, object]:
            return {"selected_access": selected_access, "task_access_state": "active"}

        def workspace_permissions(
            self,
            _chat_id: str,
            *,
            controls_available: bool,
        ) -> dict[str, object]:
            assert controls_available
            return permissions

        def preview_permission_context(self, _chat_id: str) -> tuple[object, object]:
            return owner, reference

    handler = object.__new__(GatewayHTTPHandler)
    handler.config = SimpleNamespace(host="127.0.0.1")
    handler.workspace_controls_available = lambda _connection: True
    handler.check_api_token = lambda _request: True
    session_manager.read_session_file = lambda _key: {"messages": []}
    handler.session_manager = session_manager
    handler.workspaces = workspaces
    handler.workspace_access_recovery = Recovery()
    handler.media = SimpleNamespace(
        augment_transcript_media=lambda paths, **_kwargs: paths,
        rewrite_local_markdown_images=lambda text, **_kwargs: text,
    )
    handler._log = SimpleNamespace(exception=lambda *_args: None)
    monkeypatch.setattr(ws_http, "build_webui_thread_response", lambda *_args, **_kwargs: {
        "chat_id": "chat-once",
        "messages": [],
    })

    request = Request(
        "/api/webui/threads/websocket:chat-once",
        Headers([("Host", "127.0.0.1:8765")]),
    )
    connection = SimpleNamespace(
        remote_address=("127.0.0.1", 1234),
        request=SimpleNamespace(headers=request.headers),
    )
    response = handler._handle_webui_thread_get(
        request,
        "websocket:chat-once",
        connection=connection,
    )
    assert response.status_code == 200
    body = json.loads(response.body)
    thread_access = body["workspace_scope"]["effective_access"]
    assert thread_access["source"] == "task"
    assert thread_access["read"] is True
    assert thread_access["write"] is False
    assert body["workspace_access_request"]["selected_access"] == thread_access
    assert body["workspace_permissions"]["task_access"]["state"] == "active"
    assert body["workspace_permissions"]["task_access"]["folders"] == [
        {"path": str(project.resolve()), "file_access": "read_only", "commands": False},
    ]


def test_restricted_mode_is_idempotent_while_once_is_live(tmp_path: Path) -> None:
    home = tmp_path / "agent"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    manager = SessionManager(home, sessions_root=tmp_path / "sessions")
    key = "websocket:mode-once"
    session = Session(key, metadata={"webui": True})
    manager._jsonl_store.save(session)
    manager._jsonl_store.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    grants = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    controller = WebUIWorkspaceController(
        session_manager=manager,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=grants,
        native_supervisor_available=False,
    )
    source = controller._source_binding(key)
    assert source is not None
    owner = manager.desktop_permissions
    reference = owner.activate_once(
        session_key=key,
        source_binding=source,
        task_turn_id="turn-mode-once",
        task_input_event_id="event-mode-once",
        path=project,
        file_access="read_only",
        commands=False,
    )
    before = manager._jsonl_store.source.load_source_snapshot(key).state
    projection = owner.projection(session_key=key, source_binding=source)

    result = controller.set_permission_mode({
        "chat_id": "mode-once",
        "mode": "restricted",
        "expected_runtime_id": owner.runtime_id,
        "expected_permission_revision": projection["revision"],
    }, local_ui=True)

    after = manager._jsonl_store.source.load_source_snapshot(key).state
    assert owner.is_current_task_reference(reference)
    assert after == before
    assert result["workspace_permissions"]["task_access"] is None


def test_full_mode_action_projects_the_live_full_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from nanobot.agent.tools import native_process

    home = tmp_path / "agent"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    monkeypatch.setattr(native_process, "native_supervisor_available", lambda: True)
    manager = SessionManager(home, sessions_root=tmp_path / "sessions")
    key = "websocket:mode-full"
    scope = default_workspace_scope(project, True)
    session = Session(key, metadata={"webui": True, "workspace_scope": scope.metadata()})
    manager._jsonl_store.save(session)
    manager._jsonl_store.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    grants = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    controller = WebUIWorkspaceController(
        session_manager=manager,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=grants,
        native_supervisor_available=True,
    )
    owner = manager.desktop_permissions
    source = controller._source_binding(key)
    assert source is not None
    current = controller._permission_projection(
        key,
        source,
        can_enable_full=True,
        unavailable_reason=None,
    )

    result = controller.set_permission_mode({
        "chat_id": "mode-full",
        "mode": "full",
        "expected_runtime_id": owner.runtime_id,
        "expected_permission_revision": current["revision"],
    }, local_ui=True)

    assert "error" not in result
    assert result["workspace_permissions"]["mode"] == "full"
    assert result["workspace_permissions"]["state"] == "active"
    assert result["workspace_scope"]["access_mode"] == "full"
    effective = result["workspace_scope"]["effective_access"]
    assert effective["source"] == "full"
    assert effective["boundary"] == "unrestricted"
    assert effective["read"] is True
    assert effective["write"] is True
    assert effective["commands"] == {
        "requested": True,
        "available": True,
        "backend": "native",
        "reason": None,
    }

    reference = owner.full_reference(session_key=key, source_binding=source)
    assert reference is not None
    reservation = grants.view.try_begin_owned_native_process(owner, reference)
    assert reservation is not None
    owner.restart()
    residual = controller._permission_projection(
        key,
        source,
        can_enable_full=True,
        unavailable_reason=None,
    )
    residual_scope = controller.scope_for_session_key(key)
    assert residual["mode"] == "full"
    assert residual["state"] == "retiring"
    assert residual_scope.access_mode == "full"
    future_access = controller.scope_payload(
        residual_scope,
        permission_retiring=True,
    )["effective_access"]
    assert future_access["source"] != "full"
    assert future_access["boundary"] == "folder"
    with pytest.raises(WorkspacePermissionActionError) as unavailable:
        controller.set_permission_mode({
            "chat_id": "mode-full",
            "mode": "full",
            "expected_runtime_id": owner.runtime_id,
            "expected_permission_revision": residual["revision"],
        }, local_ui=True)
    assert unavailable.value.message == "permission_retiring"


def test_reenabling_full_during_retirement_reports_permission_retiring(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agent"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    manager = SessionManager(home, sessions_root=tmp_path / "sessions")
    key = "websocket:mode-retiring"
    session = Session(key, metadata={
        "webui": True,
        "workspace_scope": default_workspace_scope(project, True).metadata(),
    })
    manager._jsonl_store.save(session)
    manager._jsonl_store.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    controller = WebUIWorkspaceController(
        session_manager=manager,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=FolderGrantStore(
            agent_home=home,
            instance_root=instance,
            path=instance / "webui" / "folder-grants.json",
        ),
        native_supervisor_available=True,
    )
    owner = manager.desktop_permissions
    source = controller._source_binding(key)
    assert source is not None
    initial = owner.projection(session_key=key, source_binding=source)
    enabled = controller.set_permission_mode({
        "chat_id": "mode-retiring",
        "mode": "full",
        "expected_runtime_id": owner.runtime_id,
        "expected_permission_revision": initial["revision"],
    }, local_ui=True)
    reference = owner.full_reference(session_key=key, source_binding=source)
    assert reference is not None
    active = [1]
    assert owner.register_native_stdin(
        reference, object(), lambda: True, lambda: (active[0], 0),
    )

    retirement = controller.set_permission_mode({
        "chat_id": "mode-retiring",
        "mode": "restricted",
        "expected_runtime_id": owner.runtime_id,
        "expected_permission_revision": enabled["workspace_permissions"]["revision"],
    }, local_ui=True)
    assert retirement["error"] == {
        "status": 409,
        "message": "native_processes_active",
    }
    assert retirement["workspace_permissions"]["state"] == "retiring"
    assert retirement["workspace_scope"]["effective_access"]["source"] != "full"
    assert retirement["workspace_scope"]["effective_access"]["boundary"] == "folder"

    projection = retirement["workspace_permissions"]
    with pytest.raises(WorkspacePermissionActionError) as caught:
        controller.set_permission_mode({
            "chat_id": "mode-retiring",
            "mode": "full",
            "expected_runtime_id": owner.runtime_id,
            "expected_permission_revision": projection["revision"],
        }, local_ui=True)
    assert caught.value.status == 409
    assert caught.value.message == "permission_retiring"


def test_reset_and_delete_wait_for_full_native_tree_and_preserve_delete_observer(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agent"
    instance = tmp_path / "instance"
    home.mkdir()
    manager = SessionManager(home, sessions_root=tmp_path / "sessions")
    key = "websocket:lifecycle"
    session = Session(key, metadata={"webui": True})
    manager._jsonl_store.save(session)
    manager._jsonl_store.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    grants = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    WebUIWorkspaceController(
        session_manager=manager,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=grants,
        native_supervisor_available=True,
    )
    source = manager.source_backend(key).source.load_source_snapshot(key).state
    binding = {
        "store_identity": source.store_identity,
        "incarnation": source.incarnation,
        "generation": source.generation,
    }
    owner = manager.desktop_permissions
    full = owner.enable_full(
        session_key=key,
        source_binding=binding,
        persistent=True,
        existing=True,
        local=True,
        idle=True,
        configured_sandbox=False,
        supervisor_available=True,
        stores_healthy=True,
    )
    active = [1]
    handle = object()
    assert owner.register_native_stdin(
        full, handle, lambda: True, lambda: (active[0], 0),
    )
    deleted: list[str] = []
    manager.set_delete_observer(deleted.append)

    with pytest.raises(SessionWriteConflict):
        manager.reset_session(key)
    retiring = owner.projection(session_key=key, source_binding=binding)
    assert retiring["mode"] == "full"
    assert retiring["state"] == "retiring"
    assert retiring["retirement"]["active_processes"] == 1
    assert manager._jsonl_store.source.load_source_snapshot(key).state.generation == source.generation

    active[0] = 0
    manager.reset_session(key)
    reset = manager._jsonl_store.source.load_source_snapshot(key)
    assert reset.state.generation == source.generation + 1
    assert owner.full_reference(session_key=key, source_binding=binding) is None
    assert deleted == []

    assert manager.delete_session(key)
    assert deleted == [key]

