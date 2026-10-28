import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.turn import ToolOutcome, TurnContext, TurnRequest, TurnSource
from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.context_repo.catalog import RepositoryCatalog
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.security.workspace_access import (
    WORKSPACE_SCOPE_METADATA_KEY,
    WorkspaceFocusCell,
    WorkspaceScopeResolver,
    bind_workspace_focus,
    bind_workspace_scope,
    current_workspace_focus,
    current_workspace_scope,
    default_workspace_scope,
    reset_workspace_scope,
)
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import JsonlSessionStore, Session, SessionManager
from nanobot.session.workspace_focus import WorkspaceFocusOwner


def test_child_task_observes_parent_focus_cell_updates_without_cross_run_leakage(tmp_path):
    home = tmp_path / "home"
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    home.mkdir()
    project_a.mkdir()
    project_b.mkdir()
    scope_a = default_workspace_scope(project_a, True)
    scope_b = default_workspace_scope(project_b, True)
    other_scope = default_workspace_scope(home, True)

    async def run() -> None:
        parent_token = bind_workspace_scope(scope_a)
        parent_cell = current_workspace_focus()
        assert parent_cell is not None
        ready = asyncio.Event()

        async def observe_parent():
            await ready.wait()
            return current_workspace_scope()

        child = asyncio.create_task(observe_parent())
        other_cell = WorkspaceFocusCell(other_scope)
        other_token = bind_workspace_focus(other_cell)
        other = current_workspace_scope()
        reset_workspace_scope(other_token)
        parent_cell.replace(scope_b)
        ready.set()
        assert await child == scope_b
        assert other == other_scope
        assert current_workspace_scope() == scope_b
        reset_workspace_scope(parent_token)

    asyncio.run(run())


def test_legacy_scope_binding_still_creates_an_updatable_run_cell(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    token = bind_workspace_scope(default_workspace_scope(home, True))
    try:
        cell = current_workspace_focus()
        assert cell is not None
        cell.replace(default_workspace_scope(project, True))
        assert current_workspace_scope() == default_workspace_scope(project, True)
    finally:
        reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_concurrent_chat_selection_updates_only_its_owned_focus(tmp_path: Path) -> None:
    home = tmp_path / "agent-home"
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    instance = tmp_path / "instance"
    sessions_root = tmp_path / "sessions"
    for directory in (home, project_a, project_b):
        directory.mkdir()

    grants = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    grants.upsert(
        {"path": str(project_a), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    grants.upsert(
        {"path": str(project_b), "file_access": "read_only", "commands": False},
        expected_revision=1,
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=grants.view,
        managed_webui_only=True,
    )
    catalog = RepositoryCatalog(instance / "repositories", runtime_root=instance)
    sessions = SessionManager(home, sessions_root=sessions_root)
    backend = JsonlSessionStore(home, sessions_root=sessions_root)
    keys = ("websocket:chat-a", "websocket:chat-b")
    for key in keys:
        session = Session(key)
        backend.save(session)
        backend.source.upgrade_source_session(
            key,
            expected_revision=session.revision,
            expected_generation=session.generation,
            expected_incarnation=session.incarnation,
        )
    focus_owner = WorkspaceFocusOwner(
        sessions=sessions,
        agent_home=home,
        grants=grants.view,
        policy=policy,
        catalog=catalog,
    )
    ready = 0
    both_ready = asyncio.Event()

    async def select_for_chat(key: str, project: Path) -> Path:
        nonlocal ready
        initial = default_workspace_scope(home, True)
        focus_token = bind_workspace_scope(initial)
        request = RequestContext(
            channel="websocket",
            chat_id=key.split(":", 1)[1],
            session_key=key,
            workspace=home,
            workspace_scope=initial,
            persistent_session=True,
        )
        try:
            with request_context(request):
                async with acquire_execution(backend, key, run_id=f"run-{key}") as execution:
                    context = TurnContext(
                        request=TurnRequest(source=TurnSource.GATEWAY, session_key=key),
                        session_key=key,
                        execution_owner=execution,
                    )
                    ready += 1
                    if ready == len(keys):
                        both_ready.set()
                    await both_ready.wait()
                    result = await focus_owner.select_from_tool(
                        context,
                        action="select",
                        path=str(project),
                    )
                    assert not result.startswith("Error:")
                    current = current_workspace_scope()
                    assert current is not None
                    return current.project_path
        finally:
            reset_workspace_scope(focus_token)

    selected_a, selected_b = await asyncio.gather(
        select_for_chat(keys[0], project_a),
        select_for_chat(keys[1], project_b),
    )

    assert selected_a == project_a.resolve()
    assert selected_b == project_b.resolve()
    metadata_a = sessions.read_session_metadata(keys[0])
    metadata_b = sessions.read_session_metadata(keys[1])
    assert metadata_a is not None and metadata_b is not None
    assert metadata_a["metadata"]["workspace_scope"]["project_path"] == str(
        project_a.resolve()
    )
    assert metadata_b["metadata"]["workspace_scope"]["project_path"] == str(
        project_b.resolve()
    )


@pytest.mark.asyncio
async def test_system_turn_cannot_commit_workspace_focus(tmp_path: Path) -> None:
    home = tmp_path / "agent-home"
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    instance = tmp_path / "instance"
    sessions_root = tmp_path / "sessions"
    for directory in (home, project_a, project_b):
        directory.mkdir()

    grants = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    grants.upsert(
        {"path": str(project_a), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    grants.upsert(
        {"path": str(project_b), "file_access": "read_only", "commands": False},
        expected_revision=1,
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=grants.view,
        managed_webui_only=True,
    )
    sessions = SessionManager(home, sessions_root=sessions_root)
    backend = JsonlSessionStore(home, sessions_root=sessions_root)
    key = "websocket:background-focus"
    session = Session(key)
    session.metadata[WORKSPACE_SCOPE_METADATA_KEY] = default_workspace_scope(
        project_a,
        True,
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
        grants=grants.view,
        policy=policy,
        catalog=RepositoryCatalog(instance / "repositories", runtime_root=instance),
    )
    initial = default_workspace_scope(project_a, True)
    focus_token = bind_workspace_scope(initial)
    request = RequestContext(
        channel="websocket",
        chat_id="background-focus",
        session_key=key,
        workspace=home,
        workspace_scope=initial,
        persistent_session=True,
    )
    try:
        with request_context(request):
            async with acquire_execution(backend, key, run_id="system-run") as execution:
                context = TurnContext(
                    request=TurnRequest(source=TurnSource.SYSTEM_COMPAT, session_key=key),
                    session_key=key,
                    execution_owner=execution,
                )
                result = await owner.select_from_tool(
                    context,
                    action="select",
                    path=str(project_b),
                )

        assert isinstance(result, ToolResult)
        assert result.is_error
        assert current_workspace_scope() == initial
        metadata = sessions.read_session_metadata(key)
        assert metadata is not None
        assert metadata["metadata"][WORKSPACE_SCOPE_METADATA_KEY]["project_path"] == str(
            project_a.resolve()
        )
    finally:
        reset_workspace_scope(focus_token)


def test_declared_grant_metadata_does_not_follow_replaced_symlink(tmp_path: Path) -> None:
    home = tmp_path / "agent-home"
    granted = tmp_path / "granted"
    outside = tmp_path / "outside"
    instance = tmp_path / "instance"
    sessions_root = tmp_path / "sessions"
    for directory in (home, granted, outside):
        directory.mkdir()
    grants = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    snapshot = grants.upsert(
        {"path": str(granted), "file_access": "read_write", "commands": True},
        expected_revision=0,
    )
    declared = snapshot.grants[0]
    granted.rmdir()
    try:
        granted.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=grants.view,
        managed_webui_only=True,
    )
    owner = WorkspaceFocusOwner(
        sessions=SessionManager(home, sessions_root=sessions_root),
        agent_home=home,
        grants=grants.view,
        policy=policy,
        catalog=RepositoryCatalog(instance / "repositories", runtime_root=instance),
    )

    access = owner.effective_access(granted, declared_grant=declared)

    assert access["exists"] is False
    assert access["read"] is False
    assert access["write"] is False
    assert access["commands"]["available"] is False
    assert access["commands"]["reason"] == "path_changed"


@pytest.mark.parametrize("home_overlaps_repo", [False, True])
def test_effective_access_matches_exact_protected_and_readable_directories(
    tmp_path: Path,
    home_overlaps_repo: bool,
) -> None:
    repo = tmp_path / "configured"
    protected = repo / "config"
    readable = repo / "docs"
    protected.mkdir(parents=True)
    readable.mkdir()
    (repo / "nanobot.context.json").write_text(json.dumps({
        "name": "configured",
        "protected": ["config/**"],
    }), encoding="utf-8")
    home = repo if home_overlaps_repo else tmp_path / "agent-home"
    home.mkdir(exist_ok=True)
    instance = tmp_path / "instance"
    grants = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    other = tmp_path / "other-grant"
    other.mkdir()
    grants.upsert(
        {"path": str(other), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager.from_config(context_repos=[str(repo)]),
        folder_grants=grants.view,
    )
    owner = WorkspaceFocusOwner(
        sessions=SessionManager(home, sessions_root=tmp_path / "sessions"),
        agent_home=home,
        grants=grants.view,
        policy=policy,
        catalog=RepositoryCatalog(instance / "repositories", runtime_root=instance),
    )

    protected_access = owner.effective_access(protected)
    readable_access = owner.effective_access(readable)

    assert protected_access["read"] is False
    assert readable_access["read"] is True


@pytest.mark.parametrize("replacement", ["home", "another_grant", "ungranted", "missing"])
@pytest.mark.asyncio
async def test_replaced_persisted_focus_fails_before_file_and_command_effects(
    tmp_path: Path,
    replacement: str,
) -> None:
    home = tmp_path / "agent-home"
    project = tmp_path / "project"
    other = tmp_path / "other-grant"
    ungranted = tmp_path / "ungranted"
    for directory in (home, project, other, ungranted):
        directory.mkdir()
    home_marker = home / "marker.txt"
    other_marker = other / "marker.txt"
    ungranted_marker = ungranted / "marker.txt"
    home_marker.write_text("home marker", encoding="utf-8")
    other_marker.write_text("other marker", encoding="utf-8")
    ungranted_marker.write_text("ungranted marker", encoding="utf-8")
    instance = tmp_path / "instance"
    grants = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    grants.upsert(
        {"path": str(project), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    grants.upsert(
        {"path": str(other), "file_access": "read_write", "commands": True},
        expected_revision=1,
    )
    resolver = WorkspaceScopeResolver(
        default_workspace=home,
        default_restrict_to_workspace=True,
        managed_state=lambda: grants.view.is_managed,
    )
    metadata = {
        WORKSPACE_SCOPE_METADATA_KEY: {
            "project_path": str(project),
            "access_mode": "restricted",
        }
    }
    project.rmdir()
    if replacement != "missing":
        destination = {
            "home": home,
            "another_grant": other,
            "ungranted": ungranted,
        }[replacement]
        try:
            project.symlink_to(destination, target_is_directory=True)
        except (NotImplementedError, OSError) as exc:
            pytest.skip(f"directory symlink unavailable: {exc}")

    scope = resolver.for_turn(
        channel="websocket", message_metadata=None, session_metadata=metadata,
    )
    assert scope.project_path == project
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=grants.view,
        managed_webui_only=True,
    )
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    owner = WorkspaceFocusOwner(
        sessions=sessions,
        agent_home=home,
        grants=grants.view,
        policy=policy,
        catalog=RepositoryCatalog(instance / "repositories", runtime_root=instance),
    )
    focus = owner.scope_payload(scope)
    assert focus["project_path"] == str(project)
    assert focus["effective_access"]["exists"] is False
    assert focus["effective_access"]["commands"]["reason"] == "path_changed"
    grant_rows = owner.management_payload()["grants"]
    row = next(item for item in grant_rows if item["path"] == str(project))
    assert row["exists"] is False
    root_rows = owner.repository_roots()["roots"]
    assert [item["path"] for item in root_rows] == [str(other)]

    focus_token = bind_workspace_scope(scope)
    request = RequestContext(
        channel="websocket",
        chat_id="replaced-focus",
        session_key="websocket:replaced-focus",
        workspace=home,
        workspace_scope=scope,
        persistent_session=True,
    )
    read_tool = ReadFileTool(
        workspace=home,
        allowed_dir=home,
        restrict_to_workspace=True,
        resource_policy=policy,
    )
    write_tool = WriteFileTool(
        workspace=home,
        allowed_dir=home,
        restrict_to_workspace=True,
        resource_policy=policy,
    )
    exec_tool = ExecTool(
        working_dir=str(home),
        restrict_to_workspace=True,
        resource_policy=policy,
        timeout=2,
    )
    try:
        with request_context(request):
            read_result = await read_tool.execute(path="marker.txt")
            write_result = await write_tool.execute(path="effect.txt", content="changed")
            exec_result = await exec_tool.execute(command="touch exec-effect.txt")
    finally:
        reset_workspace_scope(focus_token)

    assert isinstance(read_result, ToolResult) and read_result.is_error
    assert isinstance(write_result, ToolResult) and write_result.is_error
    assert isinstance(exec_result, ToolResult) and exec_result.is_error
    assert "unavailable or changed" in str(read_result)
    assert "unavailable or changed" in str(write_result)
    assert "unavailable" in str(exec_result)
    assert not isinstance(read_result, ToolOutcome)
    assert not isinstance(write_result, ToolOutcome)
    assert not (home / "effect.txt").exists()
    assert not (home / "exec-effect.txt").exists()
    assert not (other / "effect.txt").exists()
    assert not (other / "exec-effect.txt").exists()
    assert not (ungranted / "effect.txt").exists()
    assert not (ungranted / "exec-effect.txt").exists()


def test_current_once_root_is_discoverable_only_in_owning_local_task(tmp_path: Path) -> None:
    home = tmp_path / "agent-home"
    project = tmp_path / "ungranted-project"
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    home.mkdir()
    instance = tmp_path / "instance"
    sessions_root = tmp_path / "sessions"
    grants = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    sessions = SessionManager(home, sessions_root=sessions_root)
    backend = JsonlSessionStore(home, sessions_root=sessions_root)
    session_key = "websocket:once-discovery"
    session = Session(
        session_key,
        metadata={"webui": True},
    )
    backend.save(session)
    backend.source.upgrade_source_session(
        session_key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    source = backend.source.load_source_snapshot(session_key).state
    source_binding = {
        "store_identity": source.store_identity,
        "incarnation": source.incarnation,
        "generation": source.generation,
    }
    permission_owner = sessions.desktop_permissions
    task_reference = permission_owner.activate_once(
        session_key=session_key,
        source_binding=source_binding,
        task_turn_id="original-task-turn",
        task_input_event_id="original-input-event",
        path=project,
        file_access="read_only",
        commands=False,
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=grants.view,
        managed_webui_only=True,
    )
    focus_owner = WorkspaceFocusOwner(
        sessions=sessions,
        agent_home=home,
        grants=grants.view,
        policy=policy,
        catalog=RepositoryCatalog(instance / "repositories", runtime_root=instance),
    )
    scope = default_workspace_scope(home, True)
    focus_token = bind_workspace_focus(WorkspaceFocusCell(
        scope,
        permission_owner=permission_owner,
        task_reference=task_reference,
    ))
    request = RequestContext(
        channel="websocket",
        chat_id="once-discovery",
        session_key=session_key,
        workspace=home,
        workspace_scope=scope,
        persistent_session=True,
    )
    try:
        with request_context(request):
            roots = focus_owner.repository_roots()["roots"]
            discovered = focus_owner.discover(str(project))

        assert [row["path"] for row in roots] == [str(project.resolve())]
        assert roots[0]["effective_access"]["source"] == "task"
        assert [candidate["path"] for candidate in discovered["candidates"]] == [
            str(project.resolve()),
        ]
        assert grants.snapshot().revision == 0
        assert grants.snapshot().grants == ()
        assert focus_owner.management_payload()["grants"] == []
        assert focus_owner.management_payload()["repositories"] == []
        with request_context(RequestContext(
            channel="telegram",
            chat_id="once-discovery",
            session_key=session_key,
            workspace=home,
            workspace_scope=scope,
            persistent_session=True,
        )):
            assert focus_owner.repository_roots()["roots"] == []
    finally:
        reset_workspace_scope(focus_token)

    assert focus_owner.repository_roots()["roots"] == []
