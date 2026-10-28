from unittest.mock import MagicMock

import pytest

from nanobot.security.folder_grants import FolderGrantStore
from nanobot.security.workspace_access import (
    WORKSPACE_SCOPE_METADATA_KEY,
    WorkspaceScopeError,
    default_workspace_scope,
)
from nanobot.session.manager import SessionManager, SessionStore
from nanobot.webui.workspaces import (
    WebUIWorkspaceController,
    read_webui_default_access_mode,
    read_webui_workspace_state,
    webui_workspace_state_path,
    workspaces_payload,
    write_webui_default_access_mode,
)


def test_workspace_state_defaults_when_file_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")

    state = read_webui_workspace_state()

    assert state["default_access_mode"] == "default"
    assert webui_workspace_state_path() == tmp_path / "webui" / "workspace-state.json"


def test_workspace_payload_is_config_data_dir_scoped(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    default = tmp_path / "default"
    default.mkdir()

    payload = workspaces_payload(
        default_workspace=default,
        default_restrict_to_workspace=False,
        controls_available=True,
    )

    assert payload["default_scope"]["project_path"] == str(default.resolve())
    assert payload["default_scope"]["access_mode"] == "restricted"
    assert payload["controls"]["can_use_full_access"] is True
    assert payload["default_access_mode"] == "default"
    assert payload["controls"]["can_change_project"] is True
    assert payload["controls"]["can_pick_folder"] is False


def test_workspace_payload_hides_mutable_state_when_controls_unavailable(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    default = tmp_path / "default"
    default.mkdir()

    payload = workspaces_payload(
        default_workspace=default,
        default_restrict_to_workspace=False,
        controls_available=False,
    )

    assert payload["default_scope"]["project_path"] == str(default.resolve())
    assert payload["controls"]["can_change_project"] is False
    assert payload["controls"]["can_use_full_access"] is False
    assert payload["controls"]["can_pick_folder"] is False


def test_workspace_payload_advertises_native_folder_picker(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    default = tmp_path / "default"
    default.mkdir()

    payload = workspaces_payload(
        default_workspace=default,
        default_restrict_to_workspace=False,
        controls_available=True,
        folder_picker_available=True,
    )

    assert payload["controls"]["can_pick_folder"] is True


def test_workspace_payload_uses_webui_default_access_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    default = tmp_path / "default"
    default.mkdir()

    assert write_webui_default_access_mode("full") is True
    assert write_webui_default_access_mode("full") is False

    payload = workspaces_payload(
        default_workspace=default,
        default_restrict_to_workspace=True,
        controls_available=True,
    )

    assert payload["default_access_mode"] == "full"
    assert payload["default_scope"]["project_path"] == str(default.resolve())
    assert payload["default_scope"]["access_mode"] == "restricted"
    assert payload["controls"]["can_use_full_access"] is True
    assert payload["controls"]["full_access_unavailable_reason"] is None

    unrestricted = workspaces_payload(
        default_workspace=default,
        default_restrict_to_workspace=False,
        controls_available=True,
    )
    assert unrestricted["default_access_mode"] == "full"
    assert unrestricted["default_scope"]["access_mode"] == "restricted"


def test_legacy_restricted_webui_default_access_mode_maps_to_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")

    assert write_webui_default_access_mode("restricted") is False
    assert read_webui_default_access_mode() == "default"


@pytest.mark.parametrize(
    ("restrict_to_workspace", "expected_mode"),
    [(True, "restricted"), (False, "restricted")],
)
def test_webui_default_access_applies_to_unscoped_old_sessions(
    tmp_path,
    monkeypatch,
    restrict_to_workspace: bool,
    expected_mode: str,
) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    default = tmp_path / "default"
    default.mkdir()
    sessions = SessionManager(tmp_path / "sessions")
    key = "websocket:old-chat"
    sessions.save(sessions.get_or_create(key))
    source_path = sessions._get_session_path(key)
    source_bytes = source_path.read_bytes()
    write_webui_default_access_mode("full")
    preference_path = webui_workspace_state_path()
    preference_bytes = preference_path.read_bytes()
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=default,
        default_restrict_to_workspace=restrict_to_workspace,
    )

    scope = controller.scope_for_session_key(key)
    new_scope = controller.scope_for_new_chat({}, controls_available=True)

    assert scope.project_path == default.resolve()
    assert scope.access_mode == expected_mode
    assert new_scope.access_mode == expected_mode
    assert source_path.read_bytes() == source_bytes
    assert preference_path.read_bytes() == preference_bytes
    assert not (tmp_path / "instance" / "webui" / "folder-grants.json").exists()


def test_indexed_scope_preserves_missing_and_explicit_null_semantics(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    default = tmp_path / "default"
    default.mkdir()
    write_webui_default_access_mode("full")
    controller = WebUIWorkspaceController(
        session_manager=None,
        default_workspace=default,
        default_restrict_to_workspace=True,
    )
    supplied_fallback = tmp_path / "supplied-fallback"
    supplied_fallback.mkdir()
    fallback = default_workspace_scope(supplied_fallback, restrict_to_workspace=True)

    missing = controller.scope_for_indexed_metadata(
        None,
        scope_present=False,
        default_scope=fallback,
    )
    explicit_null = controller.scope_for_indexed_metadata(
        None,
        scope_present=True,
        default_scope=fallback,
    )

    assert missing.project_path == supplied_fallback.resolve()
    assert missing.access_mode == "restricted"
    assert explicit_null.project_path == default.resolve()
    assert explicit_null.access_mode == "restricted"


def test_webui_default_access_does_not_override_explicit_session_scope(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    default = tmp_path / "default"
    project = tmp_path / "project"
    default.mkdir()
    project.mkdir()
    sessions = SessionManager(tmp_path / "sessions")
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=default,
        default_restrict_to_workspace=True,
    )
    explicit = default_workspace_scope(project, restrict_to_workspace=False)
    controller.persist_scope("explicit-chat", explicit)

    scope = controller.scope_for_session_key("websocket:explicit-chat")

    assert scope.project_path == project.resolve()
    assert scope.access_mode == "restricted"

    unrestricted = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=default,
        default_restrict_to_workspace=False,
    )
    full_scope = unrestricted.scope_for_session_key("websocket:explicit-chat")
    assert full_scope.project_path == project.resolve()
    assert full_scope.access_mode == "restricted"
    unrestricted.persist_scope(
        "narrower-chat",
        default_workspace_scope(project, restrict_to_workspace=True),
    )
    assert unrestricted.scope_for_session_key("websocket:narrower-chat").access_mode == "restricted"


def test_scope_for_session_key_reads_metadata_without_full_history(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    default = tmp_path / "default"
    project = tmp_path / "project"
    default.mkdir()
    project.mkdir()
    sessions = SessionManager(tmp_path / "sessions")
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=default,
        default_restrict_to_workspace=True,
    )
    explicit = default_workspace_scope(project, restrict_to_workspace=False)
    controller.persist_scope("metadata-only", explicit)

    def fail_full_read(_key: str) -> None:
        raise AssertionError("scope lookup should not read full session history")

    monkeypatch.setattr(sessions, "read_session_file", fail_full_read)

    scope = controller.scope_for_session_key("websocket:metadata-only")

    assert scope.project_path == project.resolve()
    assert scope.access_mode == "restricted"

    unrestricted = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=default,
        default_restrict_to_workspace=False,
    )
    unrestricted_scope = unrestricted.scope_for_session_key("websocket:metadata-only")
    assert unrestricted_scope.project_path == project.resolve()
    assert unrestricted_scope.access_mode == "restricted"


def test_managed_session_keeps_removed_project_focus_with_unavailable_access(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    default = tmp_path / "default"
    acknowledged = tmp_path / "approved"
    removed = tmp_path / "removed-project"
    for directory in (default, acknowledged, removed):
        directory.mkdir()
    sessions = SessionManager(tmp_path / "sessions")
    store = FolderGrantStore(
        agent_home=default,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
        sessions_root=tmp_path / "sessions",
    )
    store.upsert(
        {"path": str(acknowledged), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=default,
        default_restrict_to_workspace=True,
        grant_store=store,
    )
    controller.persist_scope(
        "removed-project",
        default_workspace_scope(removed, restrict_to_workspace=True),
    )
    removed.rmdir()

    scope = controller.scope_for_session_key("websocket:removed-project")
    payload = controller.scope_payload(scope)

    assert scope.project_path == removed.resolve()
    assert scope.access_mode == "restricted"
    assert payload["effective_access"]["exists"] is False
    assert payload["effective_access"]["read"] is False


def test_managed_session_keeps_replaced_project_focus_unavailable(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    default = tmp_path / "default"
    project = tmp_path / "project"
    default.mkdir()
    project.mkdir()
    sessions = SessionManager(tmp_path / "sessions")
    store = FolderGrantStore(
        agent_home=default,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
    )
    store.upsert(
        {"path": str(project), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=default,
        default_restrict_to_workspace=True,
        grant_store=store,
    )
    controller.persist_scope(
        "replaced-project",
        default_workspace_scope(project, restrict_to_workspace=True),
    )
    project.rmdir()
    project.symlink_to(default, target_is_directory=True)

    scope = controller.scope_for_session_key("websocket:replaced-project")
    payload = controller.scope_payload(scope)

    assert scope.project_path == project
    assert payload["project_path"] == str(project)
    assert payload["effective_access"]["exists"] is False
    assert payload["effective_access"]["read"] is False
    assert payload["effective_access"]["commands"]["reason"] == "path_changed"


def test_new_chat_scope_is_persisted_only_after_first_message(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    default = tmp_path / "default"
    project = tmp_path / "project"
    default.mkdir()
    project.mkdir()
    sessions = SessionManager(tmp_path / "sessions")
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=default,
        default_restrict_to_workspace=True,
    )
    scope = default_workspace_scope(project, restrict_to_workspace=False)

    controller.stage_scope("draft-chat", scope)

    assert sessions.list_sessions() == []
    assert controller.scope_for_session_key("websocket:draft-chat") == scope

    controller.persist_scope("draft-chat", scope)

    assert [item["key"] for item in sessions.list_sessions()] == ["websocket:draft-chat"]


def test_scope_for_session_key_always_reads_the_active_store(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    default = tmp_path / "default"
    project = tmp_path / "project"
    default.mkdir()
    project.mkdir()
    workspace = tmp_path / "session-data"
    full_scope = default_workspace_scope(project, restrict_to_workspace=False)
    restricted_scope = default_workspace_scope(project, restrict_to_workspace=True)

    residual_sessions = SessionManager(workspace)
    residual = residual_sessions.get_or_create("websocket:cached")
    residual.metadata[WORKSPACE_SCOPE_METADATA_KEY] = full_scope.metadata()
    residual_sessions.save(residual)

    store = MagicMock(spec=SessionStore)
    store.read_metadata.side_effect = [
        {
            "key": "websocket:cached",
            "created_at": None,
            "updated_at": None,
            "metadata": {WORKSPACE_SCOPE_METADATA_KEY: full_scope.metadata()},
        },
        {
            "key": "websocket:cached",
            "created_at": None,
            "updated_at": None,
            "metadata": {WORKSPACE_SCOPE_METADATA_KEY: restricted_scope.metadata()},
        },
        {
            "key": "websocket:cached",
            "created_at": None,
            "updated_at": None,
            "metadata": {WORKSPACE_SCOPE_METADATA_KEY: full_scope.metadata()},
        },
    ]
    sessions = SessionManager(workspace, store=store)
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=default,
        default_restrict_to_workspace=False,
    )

    first = controller.scope_for_session_key("websocket:cached")
    second = controller.scope_for_session_key("websocket:cached")
    restricted_controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=default,
        default_restrict_to_workspace=True,
    )
    clamped = restricted_controller.scope_for_session_key("websocket:cached")

    assert first.project_path == project.resolve()
    assert first.access_mode == "restricted"
    assert second.project_path == project.resolve()
    assert second.access_mode == "restricted"
    assert clamped.project_path == project.resolve()
    assert clamped.access_mode == "restricted"
    assert store.read_metadata.call_count == 3
    store.update_metadata.assert_not_called()


def test_remote_existing_chat_can_reduce_its_workspace_access(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    default = tmp_path / "default"
    project = tmp_path / "project"
    default.mkdir()
    project.mkdir()
    sessions = SessionManager(tmp_path / "sessions")
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=default,
        default_restrict_to_workspace=True,
    )
    controller.persist_scope(
        "remote-chat",
        default_workspace_scope(project, restrict_to_workspace=False),
    )

    scope = controller.scope_for_set_request(
        {
            "workspace_scope": {
                "project_path": str(project),
                "access_mode": "restricted",
            }
        },
        chat_id="remote-chat",
        chat_running=False,
        controls_available=False,
    )

    assert scope.project_path == project.resolve()
    assert scope.access_mode == "restricted"


@pytest.mark.parametrize(
    ("default_restricted", "project_name", "access_mode", "allowed"),
    [
        (False, "default", "restricted", True),
        (True, "default", "full", False),
        (False, "other", "restricted", False),
    ],
)
def test_remote_new_chat_only_allows_non_escalating_scope_change(
    tmp_path,
    monkeypatch,
    default_restricted: bool,
    project_name: str,
    access_mode: str,
    allowed: bool,
) -> None:
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")
    default = tmp_path / "default"
    other = tmp_path / "other"
    default.mkdir()
    other.mkdir()
    controller = WebUIWorkspaceController(
        session_manager=None,
        default_workspace=default,
        default_restrict_to_workspace=default_restricted,
    )
    requested_path = tmp_path / project_name

    def resolve():
        return controller.scope_for_new_chat(
            {
                "workspace_scope": {
                    "project_path": str(requested_path),
                    "access_mode": access_mode,
                }
            },
            controls_available=False,
        )

    if allowed:
        scope = resolve()
        assert scope.project_path == requested_path.resolve()
        assert scope.access_mode == access_mode
    elif default_restricted and access_mode == "full":
        with pytest.raises(WorkspaceScopeError) as rejected:
            resolve()
        assert rejected.value.message == "full_access_unavailable"
        assert rejected.value.status == 409
    else:
        with pytest.raises(WorkspaceScopeError) as rejected:
            resolve()
        assert rejected.value.message == "workspace controls are localhost-only"
        assert rejected.value.status == 403
