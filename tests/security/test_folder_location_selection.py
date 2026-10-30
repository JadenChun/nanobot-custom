"""Saved catalog selections reject a changed canonical folder identity."""
from pathlib import Path

import pytest

from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.context_repo.catalog import RepositoryCatalog, RepositoryEntry
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.security.workspace_access import (
    WorkspaceScopeError,
    default_workspace_scope,
)
from nanobot.session.manager import SessionManager
from nanobot.session.workspace_focus import WorkspaceFocusOwner


def _owner(tmp_path: Path, home: Path, instance: Path) -> WorkspaceFocusOwner:
    grants = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    policy = ResourceAccessPolicy(
        workspace=home,
        context_manager=ContextRepoManager(),
        restrict_to_workspace=True,
        folder_grants=grants.view,
        managed_webui_only=True,
    )
    return WorkspaceFocusOwner(
        sessions=SessionManager(home, sessions_root=tmp_path / "sessions"),
        agent_home=home,
        grants=grants.view,
        policy=policy,
        catalog=RepositoryCatalog(instance / "catalog"),
    )


def _saved_row(owner: WorkspaceFocusOwner, name: str) -> dict[str, object]:
    management = owner.management_payload()
    repositories = management["repositories"]
    assert isinstance(repositories, list)
    return next(row for row in repositories if row["name"] == name)


def test_replaced_saved_repository_symlink_is_rejected_before_focus(tmp_path: Path):
    home = tmp_path / "home"
    outside = tmp_path / "outside"
    saved = tmp_path / "saved"
    home.mkdir()
    outside.mkdir()
    saved.mkdir()
    owner = _owner(tmp_path, home, tmp_path / "instance")
    owner.catalog.update(RepositoryEntry(name="saved", path=str(saved)))
    saved.rmdir()
    try:
        saved.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    row = _saved_row(owner, "saved")
    assert row["exists"] is False
    assert row["effective_access"]["exists"] is False
    assert row["effective_access"]["commands"]["reason"] == "path_changed"

    with pytest.raises(WorkspaceScopeError, match="changed"):
        owner.resolve_scope(
            current=default_workspace_scope(home, True),
            action="select",
            repository="saved",
        )


def test_replaced_saved_repository_ancestor_is_rejected_before_focus(tmp_path: Path):
    home = tmp_path / "home"
    parent = tmp_path / "parent"
    saved = parent / "saved"
    outside = tmp_path / "outside"
    for directory in (home, saved, outside / "saved"):
        directory.mkdir(parents=True)
    owner = _owner(tmp_path, home, tmp_path / "instance")
    owner.catalog.update(RepositoryEntry(name="saved", path=str(saved)))
    parent.rename(tmp_path / "original")
    try:
        parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    row = _saved_row(owner, "saved")
    assert row["exists"] is False
    assert row["effective_access"]["exists"] is False
    assert row["effective_access"]["commands"]["reason"] == "path_changed"

    with pytest.raises(WorkspaceScopeError, match="changed"):
        owner.resolve_scope(
            current=default_workspace_scope(home, True),
            action="select",
            repository="saved",
        )


def test_new_explicit_symlink_path_is_still_canonicalized(tmp_path: Path):
    home = tmp_path / "home"
    target = home / "target"
    link = home / "link"
    home.mkdir()
    target.mkdir()
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    owner = _owner(tmp_path, home, tmp_path / "instance")

    scope = owner.resolve_scope(
        current=default_workspace_scope(home, True),
        action="select",
        path=str(link),
    )

    assert scope.project_path == target.resolve()


def test_unchanged_saved_repository_remains_selectable(tmp_path: Path):
    home = tmp_path / "home"
    saved = home / "saved"
    home.mkdir()
    saved.mkdir()
    owner = _owner(tmp_path, home, tmp_path / "instance")
    owner.catalog.update(RepositoryEntry(name="saved", path=str(saved)))

    scope = owner.resolve_scope(
        current=default_workspace_scope(home, True),
        action="select",
        repository="saved",
    )

    assert scope.project_path == saved.resolve()
