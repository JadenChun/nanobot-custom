"""The shared resource policy composes live folder grants with repo rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.context_repo.policy import FolderAccessNeededError
from nanobot.security.desktop_permissions import DesktopPermissionOwner
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
)


def _grant(
    store: FolderGrantStore,
    path: Path,
    *,
    access: str = "read_write",
    commands: bool = True,
    revision: int,
) -> None:
    store.upsert(
        {"path": str(path), "file_access": access, "commands": commands},
        expected_revision=revision,
    )


def test_managed_policy_uses_live_read_write_and_command_grants(tmp_path, monkeypatch):
    home = tmp_path / "agent"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    target = project / "file.txt"
    target.write_text("fixture")
    store = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
    )
    _grant(store, project, revision=0)
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=store.view,
    )

    policy.validate_path(target, "read")
    policy.validate_path(target, "write")
    policy.validate_path(project, "exec")

    _grant(store, project, access="read_only", commands=False, revision=1)
    policy.validate_path(target, "read")
    with pytest.raises(PermissionError):
        policy.validate_path(target, "write")
    with pytest.raises(PermissionError):
        policy.validate_path(project, "exec")


def test_explicit_once_reference_authorizes_only_its_live_root_outside_turn_context(tmp_path):
    home = tmp_path / "agent"
    selected = tmp_path / "selected"
    home.mkdir()
    selected.mkdir()
    target = selected / "README.md"
    target.write_text("task fixture")
    store = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
    )
    owner = DesktopPermissionOwner()
    source = {
        "store_identity": "store-a",
        "incarnation": "incarnation-a",
        "generation": 0,
    }
    reference = owner.activate_once(
        session_key="websocket:chat-a",
        source_binding=source,
        task_turn_id="turn-a",
        task_input_event_id="event-a",
        path=selected,
        file_access="read_only",
        commands=False,
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=store.view,
    )

    policy.validate_path(
        target,
        "read",
        managed=True,
        permission_owner=owner,
        permission_reference=reference,
    )
    with pytest.raises(PermissionError):
        policy.validate_path(
            target,
            "write",
            managed=True,
            permission_owner=owner,
            permission_reference=reference,
        )
    owner.retire_once(
        session_key="websocket:chat-a",
        source_binding=source,
        task_turn_id="turn-a",
        task_input_event_id="event-a",
        reason="task_cancelled",
        active_turn=False,
        active_processes=0,
        unknown_processes=0,
    )
    with pytest.raises(PermissionError):
        policy.validate_path(
            target,
            "read",
            managed=True,
            permission_owner=owner,
            permission_reference=reference,
        )


def test_most_specific_child_grant_narrows_parent_and_revoke_reveals_parent(tmp_path):
    home = tmp_path / "agent"
    parent = tmp_path / "projects"
    child = parent / "locked"
    home.mkdir()
    child.mkdir(parents=True)
    target = child / "file.txt"
    target.write_text("fixture")
    store = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
    )
    _grant(store, parent, revision=0)
    _grant(store, child, access="read_only", commands=False, revision=1)
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=store.view,
    )

    with pytest.raises(PermissionError):
        policy.validate_path(target, "edit")
    _ = store.revoke(str(child), expected_revision=2)
    policy.validate_path(target, "edit")
    policy.validate_path(child, "exec")


def test_final_symlink_target_uses_its_own_grant_and_private_paths_stay_hidden(tmp_path):
    home = tmp_path / "agent"
    external = tmp_path / "external"
    instance = tmp_path / "instance"
    home.mkdir()
    external.mkdir()
    instance.mkdir()
    secret = instance / "credentials" / "key.txt"
    secret.parent.mkdir()
    secret.write_text("private")
    (home / "link").symlink_to(external, target_is_directory=True)
    (external / "file.txt").write_text("project")
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    _grant(store, external, access="read_only", commands=False, revision=0)
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=store.view,
    )

    policy.validate_path(home / "link" / "file.txt", "read")
    with pytest.raises(PermissionError):
        policy.validate_path(home / "link" / "file.txt", "write")
    with pytest.raises(PermissionError):
        policy.validate_path(secret, "read")


def test_unexpected_store_replacement_keeps_managed_access_fail_closed(tmp_path):
    home = tmp_path / "agent"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
    )
    _grant(store, project, revision=0)
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=store.view,
    )
    store.path.write_text("{}", encoding="utf-8")

    with pytest.raises(PermissionError):
        policy.validate_path(project / "file.txt", "read")


def test_agent_home_capabilities_survive_instance_privacy_exclusions(tmp_path):
    instance = tmp_path / "instance"
    home = instance / "workspace"
    sessions = instance / "sessions"
    credentials = instance / "credentials"
    home.mkdir(parents=True)
    (home / "skills").mkdir()
    (home / "skills" / "custom.md").write_text("skill")
    (home / "memory").mkdir()
    history = home / "memory" / "history.jsonl"
    history.write_text("history")
    sessions.mkdir()
    credentials.mkdir()
    (instance / "config.json").write_text("{}")
    (sessions / "session.json").write_text("{}")
    (credentials / "key.json").write_text("{}")
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        sessions_root=sessions,
        path=instance / "webui" / "folder-grants.json",
    )
    project = tmp_path / "project"
    project.mkdir()
    _grant(store, project, revision=0)
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=store.view,
    )

    policy.validate_path(home / "skills" / "custom.md", "read")
    policy.validate_path(history, "read")
    policy.validate_path(home / "new.md", "write")
    for private_path in (
        instance / "config.json",
        sessions / "session.json",
        credentials / "key.json",
        store.path,
        store.lock_path,
    ):
        with pytest.raises(PermissionError):
            policy.validate_path(private_path, "read")


@pytest.mark.parametrize("repo_grant", [None, "read_only"])
def test_managed_configured_repository_denials_precede_missing_folder_grants(
    tmp_path: Path,
    repo_grant: str | None,
) -> None:
    home = tmp_path / "agent-home"
    home.mkdir()
    context = tmp_path / "context"
    (context / "config").mkdir(parents=True)
    (context / "src").mkdir()
    (context / "data").mkdir()
    (context / "outputs").mkdir()
    (context / "nanobot.context.json").write_text(json.dumps({
        "name": "context",
        "writable": ["outputs/**"],
        "protected": ["config/**"],
        "proposalRequired": ["src/**"],
        "stores": {"state": {"path": "data/state.json", "directEdit": False}},
    }), encoding="utf-8")
    read_only = tmp_path / "read-only"
    (read_only / "outputs").mkdir(parents=True)
    (read_only / "nanobot.context.json").write_text(json.dumps({
        "name": "read-only",
        "writable": ["outputs/**"],
    }), encoding="utf-8")
    target_context = tmp_path / "target-context"
    target_context.mkdir()
    target = tmp_path / "target"
    (target / "config").mkdir(parents=True)
    (target / "src").mkdir()
    (target / "outputs").mkdir()
    (target_context / "nanobot.context.json").write_text(json.dumps({
        "name": "target-context",
        "targetRepos": {
            "target": {
                "path": str(target),
                "read": ["**"],
                "write": ["outputs/**"],
                "protected": ["config/**"],
                "proposalRequired": ["src/**"],
            }
        },
    }), encoding="utf-8")
    manager = ContextRepoManager.from_config(context_repos=[
        {"path": str(context)},
        {"path": str(read_only), "read_only": True},
        {"path": str(target_context)},
    ])
    grants = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
    )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    grants.upsert(
        {"path": str(unrelated), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    if repo_grant is not None:
        grants.upsert(
            {"path": str(context), "file_access": repo_grant, "commands": False},
            expected_revision=1,
        )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=manager,
        folder_grants=grants.view,
    )
    terminal_paths = [
        (context / "config" / "private.txt", "read", "protected"),
        (context / "src" / "change.py", "write", "proposal"),
        (context / "data" / "state.json", "write", "managed store"),
        (context / "random.txt", "write", "writable area"),
        (read_only / "outputs" / "draft.md", "write", "read-only"),
        (target / "config" / "private.txt", "read", "protected"),
        (target / "src" / "change.py", "write", "proposal"),
    ]

    for path, action, _label in terminal_paths:
        with pytest.raises(PermissionError) as caught:
            policy.validate_path(path, action)
        assert not isinstance(caught.value, FolderAccessNeededError)


@pytest.mark.parametrize("repo_kind", ["context", "target"])
def test_managed_private_runtime_precedes_configured_repository_allow(
    tmp_path: Path,
    repo_kind: str,
) -> None:
    instance = tmp_path / "instance"
    home = instance / "workspace"
    private_repo = instance / "sessions" / f"configured-{repo_kind}"
    home.mkdir(parents=True)
    private_repo.mkdir(parents=True)
    sentinel = private_repo / "runtime-note.txt"
    sentinel.write_text("PRIVATE_RUNTIME_SENTINEL", encoding="utf-8")

    if repo_kind == "context":
        (private_repo / "nanobot.context.json").write_text(
            json.dumps({"name": "private-context", "writable": ["**"]}),
            encoding="utf-8",
        )
        manager = ContextRepoManager.from_config(context_repos=[{"path": str(private_repo)}])
    else:
        manifest_root = tmp_path / "target-manifest"
        manifest_root.mkdir()
        (manifest_root / "nanobot.context.json").write_text(
            json.dumps({
                "name": "target-config",
                "targetRepos": {
                    "private-target": {
                        "path": str(private_repo),
                        "read": ["**"],
                        "write": ["**"],
                    },
                },
            }),
            encoding="utf-8",
        )
        manager = ContextRepoManager.from_config(context_repos=[{"path": str(manifest_root)}])

    unrelated = tmp_path / "unrelated-public-project"
    unrelated.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        sessions_root=instance / "sessions",
        path=instance / "webui" / "folder-grants.json",
    )
    store.upsert(
        {"path": str(unrelated), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=manager,
        restrict_to_workspace=True,
        folder_grants=store.view,
    )

    assert policy.managed_mode
    assert store.view.is_private_path(sentinel)
    for action in ("read", "list", "write", "edit"):
        with pytest.raises(PermissionError) as denied:
            policy.validate_path(sentinel, action)
        assert type(denied.value) is PermissionError
        assert "private nanobot runtime storage" in str(denied.value)


def test_managed_prompt_tolerates_missing_memory_file_and_rejects_existing_escape(tmp_path):
    home = tmp_path / "agent-home"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    _grant(store, project, revision=0)
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        folder_grants=store.view,
        managed_webui_only=True,
    )
    builder = ContextBuilder(home, resource_policy=policy, folder_grants=store.view)
    scope = build_workspace_scope(project, "full", source_channel="websocket")
    token = bind_workspace_scope(scope)
    request = RequestContext(
        channel="websocket",
        chat_id="fixture-chat",
        session_key="websocket:fixture-chat",
        workspace=home,
        workspace_scope=scope,
    )
    try:
        with request_context(request):
            prompt = builder.build_system_prompt(workspace=project)
        assert "# Current Project" in prompt
        assert f"Working directory: {project.resolve()}" in prompt
        assert not builder.memory.memory_file.exists()

        outside = tmp_path / "outside.md"
        outside.write_text("PRIVATE_OUTSIDE_AGENT_HOME", encoding="utf-8")
        (home / "SOUL.md").symlink_to(outside)
        with request_context(request):
            with pytest.raises(PermissionError, match="outside agent home"):
                builder.build_system_prompt(workspace=project)
    finally:
        reset_workspace_scope(token)
