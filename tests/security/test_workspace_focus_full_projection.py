"""Requirements test: protected context-repo directories stay denied under Full access.

A protected directory must not be advertised as readable or writable by the
workspace focus projection, even while Full access is live, because the path
policy already denies it.
"""

from __future__ import annotations

import json

import pytest

from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.context_repo.catalog import RepositoryCatalog
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.session.manager import SessionManager
from nanobot.session.workspace_focus import WorkspaceFocusOwner

CHAT = "websocket:chat-a"
BINDING = {"store_identity": "store-req", "incarnation": "inc-b", "generation": 1}


def _build_owner(tmp_path):
    repo = tmp_path / "configured"
    (repo / "config").mkdir(parents=True)
    (repo / "nanobot.context.json").write_text(
        json.dumps({"name": "configured", "protected": ["config/**"]}), encoding="utf-8"
    )
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir()
    instance = tmp_path / "instance"
    instance.mkdir()

    manager = ContextRepoManager.from_config(context_repos=[str(repo)])
    grants = FolderGrantStore(
        agent_home=agent_home, instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    ).view
    policy = ResourceAccessPolicy(
        workspace=tmp_path, context_manager=manager, folder_grants=grants
    )
    catalog = RepositoryCatalog(instance, runtime_root=instance)
    sessions = SessionManager(agent_home, sessions_root=tmp_path / "sessions")
    owner = WorkspaceFocusOwner(
        sessions=sessions, agent_home=agent_home, grants=grants,
        policy=policy, catalog=catalog,
    )
    return owner, policy, sessions, repo


def _live_full(sessions):
    reference = sessions.desktop_permissions.enable_full(
        session_key=CHAT, source_binding=dict(BINDING), persistent=True, existing=True,
        local=True, idle=True, configured_sandbox=False, supervisor_available=True,
        stores_healthy=True,
    )
    assert reference is not None
    request_ref = sessions.desktop_permissions.full_reference(
        session_key=CHAT, source_binding=dict(BINDING), request_turn_id="turn-1"
    )
    assert request_ref is not None
    return request_ref


def test_projection_denies_protected_directory_without_full(tmp_path) -> None:
    owner, _policy, _sessions, repo = _build_owner(tmp_path)

    access = owner.effective_access(repo / "config")
    assert access["read"] is False


def test_projection_denies_protected_directory_under_live_full(tmp_path) -> None:
    owner, _policy, sessions, repo = _build_owner(tmp_path)
    reference = _live_full(sessions)

    access = owner.effective_access(repo / "config", permission_reference=reference)
    assert access["read"] is False
    assert access["write"] is False


def test_projection_allows_an_ordinary_directory_under_live_full(tmp_path) -> None:
    owner, _policy, sessions, _repo = _build_owner(tmp_path)
    reference = _live_full(sessions)
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()

    access = owner.effective_access(ordinary, permission_reference=reference)
    assert access["read"] is True


def test_policy_still_denies_the_protected_path_under_full(tmp_path) -> None:
    owner, policy, sessions, repo = _build_owner(tmp_path)
    reference = _live_full(sessions)
    secret_file = repo / "config" / "secret.txt"
    secret_file.write_text("secret", encoding="utf-8")

    with pytest.raises(PermissionError):
        policy.validate_path(
            secret_file, "read", managed=True, allow_folder_grants=True,
            permission_owner=sessions.desktop_permissions,
            permission_reference=reference,
        )


def _build_scoped_owner(tmp_path):
    """Owner whose configured repository has protected, writable, and proposal areas."""
    repo = tmp_path / "configured"
    for name in ("config", "ordinary/sub", "docs/sub"):
        (repo / name).mkdir(parents=True)
    (repo / "nanobot.context.json").write_text(
        json.dumps({
            "name": "configured",
            "protected": ["config/**"],
            "writable": ["ordinary/**"],
            "proposalRequired": ["docs/**"],
        }),
        encoding="utf-8",
    )
    agent_home = tmp_path / "agent-home"
    agent_home.mkdir()
    instance = tmp_path / "instance"
    instance.mkdir()

    manager = ContextRepoManager.from_config(context_repos=[str(repo)])
    grants = FolderGrantStore(
        agent_home=agent_home, instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    ).view
    policy = ResourceAccessPolicy(
        workspace=tmp_path, context_manager=manager, folder_grants=grants
    )
    catalog = RepositoryCatalog(instance, runtime_root=instance)
    sessions = SessionManager(agent_home, sessions_root=tmp_path / "sessions")
    owner = WorkspaceFocusOwner(
        sessions=sessions, agent_home=agent_home, grants=grants,
        policy=policy, catalog=catalog,
    )
    return owner, policy, sessions, repo


def _live_once(sessions, path):
    reference = sessions.desktop_permissions.activate_once(
        session_key=CHAT,
        source_binding=dict(BINDING),
        task_turn_id="turn-once",
        task_input_event_id="input-once",
        path=path,
        file_access="read_write",
        commands=True,
    )
    assert sessions.desktop_permissions.is_current_task_reference(reference)
    return reference


def _policy_allows(policy, sessions, reference, path, action) -> bool:
    try:
        policy.validate_path(
            path,
            action,
            managed=True,
            allow_folder_grants=True,
            target_kind="directory",
            permission_owner=sessions.desktop_permissions,
            permission_reference=reference,
        )
    except PermissionError:
        return False
    return True


def test_c1_once_projection_denies_a_configured_protected_directory(tmp_path) -> None:
    owner, policy, sessions, repo = _build_scoped_owner(tmp_path)
    reference = _live_once(sessions, repo)
    protected = repo / "config"

    access = owner.effective_access(protected, permission_reference=reference)

    assert access["source"] == "task"
    assert access["read"] is False
    assert access["write"] is False
    with pytest.raises(PermissionError):
        policy.validate_path(
            protected, "read", managed=True, allow_folder_grants=True,
            target_kind="directory",
            permission_owner=sessions.desktop_permissions,
            permission_reference=reference,
        )


def test_c2_once_projection_allows_an_ordinary_directory(tmp_path) -> None:
    owner, policy, sessions, repo = _build_scoped_owner(tmp_path)
    reference = _live_once(sessions, repo)
    ordinary = repo / "ordinary" / "sub"

    access = owner.effective_access(ordinary, permission_reference=reference)

    assert access["source"] == "task"
    assert access["read"] is True
    assert access["write"] is True
    assert _policy_allows(policy, sessions, reference, ordinary, "read") is True
    assert _policy_allows(policy, sessions, reference, ordinary, "write") is True


def test_c3_full_projection_denies_the_configured_protected_directory(tmp_path) -> None:
    owner, policy, sessions, repo = _build_scoped_owner(tmp_path)
    reference = _live_full(sessions)
    protected = repo / "config"

    access = owner.effective_access(protected, permission_reference=reference)

    assert access["source"] == "full"
    assert access["read"] is False
    assert access["write"] is False
    with pytest.raises(PermissionError):
        policy.validate_path(
            protected, "read", managed=True, allow_folder_grants=True,
            target_kind="directory",
            permission_owner=sessions.desktop_permissions,
            permission_reference=reference,
        )


def test_c4_normal_projection_distinguishes_ordinary_and_protected(tmp_path) -> None:
    owner, policy, sessions, repo = _build_scoped_owner(tmp_path)

    ordinary = owner.effective_access(repo / "ordinary" / "sub")
    protected = owner.effective_access(repo / "config")

    assert ordinary["read"] is True
    assert protected["read"] is False
    assert protected["write"] is False
    assert _policy_allows(policy, sessions, None, repo / "config", "read") is False


def test_c5_once_projection_denies_write_to_a_proposal_path(tmp_path) -> None:
    owner, _policy, sessions, repo = _build_scoped_owner(tmp_path)
    reference = _live_once(sessions, repo)
    proposal = repo / "docs" / "sub"

    access = owner.effective_access(proposal, permission_reference=reference)

    assert access["write"] is False

