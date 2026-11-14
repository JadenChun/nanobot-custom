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
