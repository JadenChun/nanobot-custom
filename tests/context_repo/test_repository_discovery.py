import subprocess
from pathlib import Path

from nanobot.context_repo.catalog import RepositoryCatalog
from nanobot.context_repo.discovery import discover_repositories
from nanobot.context_repo.policy import ResourceAccessPolicy
from nanobot.security.folder_grants import FolderGrantStore


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    return path


def _managed_policy(tmp_path: Path, root: Path) -> tuple[FolderGrantStore, ResourceAccessPolicy]:
    home = tmp_path / "agent-home"
    home.mkdir(exist_ok=True)
    store = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        sessions_root=tmp_path / "instance" / "sessions",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
    )
    store.upsert(
        {"path": str(root), "file_access": "read_only", "commands": True},
        expected_revision=0,
    )
    return store, ResourceAccessPolicy(workspace=home, folder_grants=store.view)


def test_discovery_finds_bounded_git_roots_and_does_not_follow_symlinks(tmp_path):
    parent = tmp_path / "projects"
    parent.mkdir()
    first = _git_repo(parent / "alpha")
    second = _git_repo(parent / "nested" / "beta")
    outside = _git_repo(tmp_path / "outside")
    (parent / "escape").symlink_to(outside, target_is_directory=True)
    (parent / ".hidden").mkdir()
    _git_repo(parent / ".hidden" / "hidden-repo")
    store, policy = _managed_policy(tmp_path, parent)
    catalog = RepositoryCatalog(tmp_path / "instance" / "repositories", runtime_root=tmp_path / "instance")

    result = discover_repositories(
        parent,
        policy=policy,
        catalog=catalog,
        access_for_path=lambda path: {
            "policy": "managed",
            "source": "grant",
            "grant_root": str(parent),
            "exists": path.is_dir(),
            "read": True,
            "write": False,
            "commands": {"requested": True, "available": True, "backend": "bwrap", "reason": None},
        },
    )

    assert {row["path"] for row in result["candidates"]} == {str(first), str(second)}
    assert result["visited_directories"] <= 512
    assert result["truncated"] is False
    assert store.view.is_managed


def test_discovery_rejects_unreadable_start_and_caps_traversal(tmp_path, monkeypatch):
    parent = tmp_path / "projects"
    parent.mkdir()
    for index in range(520):
        (parent / f"dir-{index:04}").mkdir()
    home = tmp_path / "agent-home"
    home.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        sessions_root=tmp_path / "instance" / "sessions",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
    )
    store.revoke(str(tmp_path / "missing"), expected_revision=0)
    policy = ResourceAccessPolicy(workspace=home, folder_grants=store.view)
    catalog = RepositoryCatalog(tmp_path / "instance" / "repositories", runtime_root=tmp_path / "instance")

    def access(path: Path):
        return {
            "policy": "managed",
            "source": "none",
            "grant_root": None,
            "exists": path.is_dir(),
            "read": False,
            "write": False,
            "commands": {
                "requested": False,
                "available": False,
                "backend": "none",
                "reason": "not_granted",
            },
        }

    unlisted = discover_repositories(parent, policy=policy, catalog=catalog, access_for_path=access)
    assert unlisted["visited_directories"] == 512
    assert unlisted["truncated"] is True

    store, policy = _managed_policy(tmp_path, parent)
    catalog = RepositoryCatalog(tmp_path / "instance" / "repositories", runtime_root=tmp_path / "instance")
    result = discover_repositories(parent, policy=policy, catalog=catalog, access_for_path=access)
    assert result["visited_directories"] == 512
    assert result["truncated"] is True

    original = policy.validate_path
    visited: list[Path] = []

    def validate_path(path, action, **kwargs):
        visited.append(path)
        if path != parent.resolve():
            raise PermissionError("not readable")
        return original(path, action, **kwargs)

    monkeypatch.setattr(policy, "validate_path", validate_path)
    result = discover_repositories(parent, policy=policy, catalog=catalog, access_for_path=access)

    assert result["visited_directories"] == 1
    assert result["candidates"] == []
    assert len(visited) > 1
