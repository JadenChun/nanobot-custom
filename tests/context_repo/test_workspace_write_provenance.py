"""Tests for Phase 2 workspace-write provenance and Sandbox mount alignment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kohaku.agent.tools.context import RequestContext, request_context
from kohaku.agent.tools.sandbox import (
    _managed_mounts,
    managed_bwrap_available,
    wrap_managed_command,
)
from kohaku.agent.turn import TurnContext, TurnRequest, TurnSource
from kohaku.context_repo import ContextRepoManager, ResourceAccessPolicy
from kohaku.context_repo.catalog import RepositoryCatalog
from kohaku.security.folder_grants import FolderGrantStore
from kohaku.security.workspace_access import (
    WORKSPACE_SCOPE_METADATA_KEY,
    bind_workspace_scope,
    build_workspace_scope,
    current_workspace_scope,
    reset_workspace_scope,
    workspace_scope_from_metadata,
)
from kohaku.session.execution_owner import acquire_execution
from kohaku.session.manager import JsonlSessionStore, Session, SessionManager
from kohaku.session.workspace_focus import WorkspaceFocusOwner


def _make_store(tmp_path: Path, home: Path) -> FolderGrantStore:
    instance = tmp_path / "instance"
    return FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )


def _overlaps_none(_path: Path) -> bool:
    return False


def test_user_selected_websocket_workspace_is_implicitly_writable_without_grant(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agent-home"
    project = tmp_path / "selected-project"
    outside = tmp_path / "outside"
    for directory in (home, project, outside):
        directory.mkdir()

    store = _make_store(tmp_path, home)
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=store.view,
        managed_webui_only=True,
        sandbox_default=True,
    )
    scope = build_workspace_scope(
        project,
        "restricted",
        source_channel="websocket",
    )
    assert scope.selection_source == "user"
    assert "selection_source" not in scope.metadata()

    token = bind_workspace_scope(scope)
    try:
        assert policy.implicit_writable_workspace() == project.resolve()
        policy.validate_path(project / "src" / "main.py", "write")
        with pytest.raises(PermissionError):
            policy.validate_path(outside / "notes.txt", "write")

        roots, _ = policy.readable_roots_summary()
        assert str(project.resolve()) in roots

        _home, _working, mounts = _managed_mounts(
            cwd=project,
            agent_home=home,
            grants=store.view,
            workspace_root=policy.implicit_writable_workspace(),
            include_folder_grants=True,
            grant_snapshot=None,
            task_roots=(),
            overlaps_system_control=_overlaps_none,
            system_control_error="system control",
        )
        assert (project.resolve(), True, 0) in mounts

        if managed_bwrap_available():
            wrapped = wrap_managed_command(
                "true",
                cwd=project,
                agent_home=home,
                grants=store.view,
                workspace_root=policy.implicit_writable_workspace(),
            )
            assert f"--bind {project.resolve()} {project.resolve()}" in wrapped
    finally:
        reset_workspace_scope(token)


def test_explicit_read_only_grant_overrides_implicit_workspace_write(tmp_path: Path) -> None:
    home = tmp_path / "agent-home"
    project = tmp_path / "selected-project"
    readonly_sub = project / "vendor"
    for directory in (home, project, readonly_sub):
        directory.mkdir(parents=True)

    store = _make_store(tmp_path, home)
    store.upsert(
        {"path": str(readonly_sub), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=store.view,
        managed_webui_only=True,
        sandbox_default=True,
    )
    scope = build_workspace_scope(
        project,
        "restricted",
        source_channel="websocket",
    )
    request = RequestContext(
        channel="websocket",
        chat_id="ws-ro-sub",
        session_key="websocket:ws-ro-sub",
        workspace=home,
        workspace_scope=scope,
        persistent_session=True,
    )
    token = bind_workspace_scope(scope)
    try:
        with request_context(request):
            policy.validate_path(project / "app.py", "write")
            with pytest.raises(PermissionError):
                policy.validate_path(readonly_sub / "locked.py", "write")

            _home, _working, mounts = _managed_mounts(
                cwd=project,
                agent_home=home,
                grants=store.view,
                workspace_root=policy.implicit_writable_workspace(),
                include_folder_grants=True,
                grant_snapshot=None,
                task_roots=(),
                overlaps_system_control=_overlaps_none,
                system_control_error="system control",
            )
            ordered = sorted(mounts, key=lambda row: (len(row[0].parts), str(row[0]), row[2]))
            assert ordered.index((project.resolve(), True, 0)) < ordered.index(
                (readonly_sub.resolve(), False, 1)
            )
            if managed_bwrap_available():
                wrapped = wrap_managed_command(
                    "true",
                    cwd=project,
                    agent_home=home,
                    grants=store.view,
                    workspace_root=policy.implicit_writable_workspace(),
                )
                assert f"--bind {project.resolve()} {project.resolve()}" in wrapped
                assert f"--ro-bind {readonly_sub.resolve()} {readonly_sub.resolve()}" in wrapped
                assert wrapped.index(f"--bind {project.resolve()}") < wrapped.index(
                    f"--ro-bind {readonly_sub.resolve()}"
                )
    finally:
        reset_workspace_scope(token)

    # Explicit read_only grant on the workspace root itself also overrides priority 0.
    store_root = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance-root-ro",
        path=tmp_path / "instance-root-ro" / "webui" / "folder-grants.json",
    )
    store_root.upsert(
        {"path": str(project), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    policy_root = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=store_root.view,
        managed_webui_only=True,
        sandbox_default=True,
    )
    token = bind_workspace_scope(scope)
    try:
        with request_context(request):
            with pytest.raises(PermissionError):
                policy_root.validate_path(project / "app.py", "write")
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_select_workspace_tool_does_not_grant_implicit_write_across_turns(
    tmp_path: Path,
) -> None:
    home = tmp_path / "agent-home"
    ungranted = tmp_path / "ungranted-project"
    instance = tmp_path / "instance"
    sessions_root = tmp_path / "sessions"
    for directory in (home, ungranted):
        directory.mkdir()

    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=store.view,
        managed_webui_only=True,
        sandbox_default=True,
    )
    sessions = SessionManager(home, sessions_root=sessions_root)
    backend = JsonlSessionStore(home, sessions_root=sessions_root)
    key = "websocket:tool-focus"
    session = Session(key)
    session.metadata["webui"] = True
    session.metadata[WORKSPACE_SCOPE_METADATA_KEY] = build_workspace_scope(
        home,
        "restricted",
        source_channel="websocket",
    ).metadata()
    backend.save(session)
    backend.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    owner = WorkspaceFocusOwner(
        sessions=sessions,
        agent_home=home,
        grants=store.view,
        policy=policy,
        catalog=RepositoryCatalog(instance / "repositories", runtime_root=instance),
    )
    initial = build_workspace_scope(home, "restricted", source_channel="websocket")
    token = bind_workspace_scope(initial)
    request = RequestContext(
        channel="websocket",
        chat_id="tool-focus",
        session_key=key,
        workspace=home,
        workspace_scope=initial,
        persistent_session=True,
    )
    try:
        with request_context(request):
            async with acquire_execution(backend, key, run_id="run-1") as execution:
                context = TurnContext(
                    request=TurnRequest(source=TurnSource.GATEWAY, session_key=key),
                    session_key=key,
                    execution_owner=execution,
                )
                result = await owner.select_from_tool(
                    context,
                    action="select",
                    path=str(ungranted),
                )
        payload = json.loads(result)
        assert payload["workspace_scope"]["project_path"] == str(ungranted.resolve())
        active_scope = current_workspace_scope()
        assert active_scope is not None
        assert active_scope.selection_source == "tool"
        assert policy.implicit_writable_workspace() is None
        with pytest.raises(PermissionError):
            policy.validate_path(ungranted / "file.txt", "write")

        # Next turn: restoring from persisted session metadata preserves selection_source="tool".
        persisted = sessions.read_session_metadata(key)
        assert persisted is not None
        assert persisted["metadata"][WORKSPACE_SCOPE_METADATA_KEY]["selection_source"] == "tool"
        restored = workspace_scope_from_metadata(
            persisted["metadata"],
            default_workspace=home,
            default_restrict_to_workspace=True,
            source_channel="websocket",
        )
        assert restored.selection_source == "tool"
        token2 = bind_workspace_scope(restored)
        try:
            assert policy.implicit_writable_workspace() is None
            with pytest.raises(PermissionError):
                policy.validate_path(ungranted / "next-turn.txt", "write")
        finally:
            reset_workspace_scope(token2)

        # Clearing back to agent_home restores selection_source="user".
        cleared = owner.resolve_scope(current=restored, action="clear", from_tool=True)
        assert cleared.project_path == home.resolve()
        assert cleared.selection_source == "user"
    finally:
        reset_workspace_scope(token)


def test_broad_and_protected_roots_are_never_implicitly_writable(tmp_path: Path) -> None:
    home = tmp_path / "agent-home"
    context_repo = tmp_path / "context-repo"
    home.mkdir()
    context_repo.mkdir()
    (context_repo / "nanobot.context.json").write_text(
        json.dumps({"name": "ctx", "read_only": True}),
        encoding="utf-8",
    )
    store = _make_store(tmp_path, home)
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager.from_config(context_repos=[str(context_repo)]),
        folder_grants=store.view,
        managed_webui_only=True,
        sandbox_default=True,
    )

    protected_candidates = [
        Path.home().expanduser().resolve(strict=False),
        Path("/"),
        store.view.sandbox_instance_root,
        store.path.parent,
        context_repo,
        tmp_path / "does-not-exist",
    ]
    for candidate in protected_candidates:
        scope = build_workspace_scope(
            candidate,
            "restricted",
            source_channel="websocket",
        )
        token = bind_workspace_scope(scope)
        try:
            assert policy.implicit_writable_workspace() is None, f"unexpectedly writable: {candidate}"
        finally:
            reset_workspace_scope(token)

        if candidate != context_repo:
            _home, _working, mounts = _managed_mounts(
                cwd=home,
                agent_home=home,
                grants=store.view,
                workspace_root=candidate,
                include_folder_grants=True,
                grant_snapshot=None,
                task_roots=(),
                overlaps_system_control=_overlaps_none,
                system_control_error="system control",
            )
            assert mounts == [(home.resolve(), True, 0)]


@pytest.mark.parametrize("channel", ["cli", "telegram"])
def test_non_websocket_channels_do_not_get_implicit_workspace_write(
    tmp_path: Path,
    channel: str,
) -> None:
    home = tmp_path / "agent-home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()

    store = _make_store(tmp_path, home)
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=store.view,
        managed_webui_only=True,
        sandbox_default=True,
    )
    scope = build_workspace_scope(
        project,
        "restricted",
        source_channel=channel,
    )
    token = bind_workspace_scope(scope)
    try:
        assert policy.implicit_writable_workspace() is None
        with pytest.raises(PermissionError):
            policy.validate_path(project / "file.txt", "write")
    finally:
        reset_workspace_scope(token)
