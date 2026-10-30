"""Sandbox confines writes, while ordinary reads and commands remain usable."""
from pathlib import Path

import pytest

from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.security.folder_grants import FolderGrantStore


def test_sandbox_reads_unlisted_paths_and_ignores_legacy_command_checkbox(tmp_path: Path):
    home, project, outside, readonly = (tmp_path / name for name in ("home", "project", "outside", "project/reference"))
    for path in (home, project, outside, readonly):
        path.mkdir(parents=True, exist_ok=True)
    store = FolderGrantStore(agent_home=home, instance_root=tmp_path / "instance", path=tmp_path / "instance/webui/folder-grants.json")
    store.upsert({"path": str(project), "file_access": "read_write", "commands": False}, expected_revision=0)
    store.upsert({"path": str(readonly), "file_access": "read_only", "commands": False}, expected_revision=1)
    policy = ResourceAccessPolicy(workspace=home, context_manager=ContextRepoManager(), folder_grants=store.view, restrict_to_workspace=True)
    policy.validate_path(outside / "readme", "read")
    policy.validate_path(outside, "list")
    policy.validate_path(outside, "exec")
    assert policy.validate_exec(f"cat {outside}/readme", project) is None
    policy.validate_path(project / "unmapped/file", "write")
    policy.validate_path(readonly, "exec")
    for path in (outside / "file", readonly / "file"):
        with pytest.raises(PermissionError):
            policy.validate_path(path, "write")
    with pytest.raises(PermissionError, match="private"):
        policy.validate_path(store.path, "read")


def test_folder_map_is_metadata_only_and_never_an_exclusive_write_list(tmp_path):
    from nanobot.agent.context import ContextBuilder
    from nanobot.context_repo.catalog import RepositoryCatalog, RepositoryEntry
    home, documents, reference = (tmp_path / name for name in ("home", "documents", "documents/reference"))
    for path in (home, documents, reference):
        path.mkdir(parents=True, exist_ok=True)
    (reference / "unread.txt").write_text("CONTENTS_MUST_NOT_BE_PRELOADED")
    store = FolderGrantStore(agent_home=home, instance_root=tmp_path / "instance", path=tmp_path / "instance/webui/folder-grants.json")
    store.upsert({"path": str(documents), "file_access": "read_write", "commands": False}, expected_revision=0)
    policy = ResourceAccessPolicy(workspace=home, folder_grants=store.view)
    catalog = RepositoryCatalog(tmp_path / "instance/catalog")
    catalog.update(RepositoryEntry(name="documents", path=str(documents), purpose="OMIT_THIS_MAP_DESCRIPTION", agent_map=False))
    catalog.update(RepositoryEntry(name="reference", path=str(reference), purpose="Only navigation metadata"))
    builder = ContextBuilder(home, resource_policy=policy, folder_grants=store.view)
    builder.repositories = catalog
    prompt = builder.build_system_prompt(include_memory=False)
    assert str(documents) in prompt
    assert '"effective_access": "read_write"' in prompt
    assert "Only navigation metadata" in prompt
    assert "OMIT_THIS_MAP_DESCRIPTION" not in prompt
    assert "CONTENTS_MUST_NOT_BE_PRELOADED" not in prompt
    policy.validate_path(documents / "unmapped/new.txt", "write")


def test_sandbox_write_boundary_holds_before_any_folder_is_saved(tmp_path):
    home, outside = tmp_path / "home", tmp_path / "outside"
    home.mkdir()
    outside.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance/webui/folder-grants.json",
    )
    assert not store.view.is_managed
    policy = ResourceAccessPolicy(workspace=home, folder_grants=store.view, sandbox_default=True)
    assert policy.managed_mode
    policy.validate_path(outside / "readme", "read")
    policy.validate_path(home / "notes.txt", "write")
    with pytest.raises(PermissionError):
        policy.validate_path(outside / "notes.txt", "write")


def test_cli_channel_policy_keeps_legacy_access_until_grants_activate(tmp_path):
    home, outside = tmp_path / "home", tmp_path / "outside"
    home.mkdir()
    outside.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance/webui/folder-grants.json",
    )
    policy = ResourceAccessPolicy(workspace=home, folder_grants=store.view)
    assert not policy.managed_mode
    policy.validate_path(outside / "notes.txt", "write")


def test_sandbox_file_reads_cannot_reach_gateway_process_environment(tmp_path):
    import os
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(agent_home=home, instance_root=tmp_path / "instance", path=tmp_path / "instance/webui/folder-grants.json")
    store.upsert({"path": str(project), "file_access": "read_write", "commands": False}, expected_revision=0)
    policy = ResourceAccessPolicy(workspace=home, folder_grants=store.view)
    for path in (Path('/proc/self/environ'), Path(f'/proc/{os.getpid()}/environ'), Path(f'/proc/{os.getpid()}/task')):
        with pytest.raises(PermissionError, match="Host process state"):
            policy.validate_path(path, "read")
    policy.validate_path(Path('/proc/cpuinfo'), "read")
