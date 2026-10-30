from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.agent.tools.filesystem import WriteFileTool
from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.context_repo.manager import ManagedContextRepo


@pytest.mark.parametrize("field", ["stores", "modules", "entrypoints", "sync", "tools"])
def test_manifest_objects_reject_non_string_keys(tmp_path: Path, field: str) -> None:
    repo = ManagedContextRepo(
        path=tmp_path, name="test", managed=True, manifest={field: {1: {}}},
    )
    accessor = "sync_config" if field == "sync" else field
    with pytest.raises(ValueError, match="keys must be strings"):
        getattr(repo, accessor)


def test_malformed_store_cannot_discard_direct_edit_restriction(tmp_path: Path) -> None:
    repo = ManagedContextRepo(
        path=tmp_path, name="test", managed=True,
        manifest={"stores": {"state": {
            "path": "outputs/state.json", "directEdit": False, 1: "invalid",
        }}},
    )
    with pytest.raises(ValueError, match="keys must be strings"):
        repo.blocks_direct_store_edit("outputs/state.json")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.mark.parametrize("content", ["{", "[]", "null", '"invalid"'])
def test_invalid_manifest_cannot_become_an_unrestricted_plain_repo(
    tmp_path: Path, content: str,
) -> None:
    (tmp_path / "nanobot.context.json").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        ContextRepoManager.from_config(context_repos=[str(tmp_path)])


def test_plain_repository_without_manifest_remains_supported(tmp_path: Path) -> None:
    manager = ContextRepoManager.from_config(context_repos=[str(tmp_path)])
    assert manager.repos[0].managed is False
    ResourceAccessPolicy(workspace=tmp_path, context_manager=manager).validate_path(
        tmp_path / "ordinary-file.md", "write",
    )


@pytest.mark.parametrize("managed", [False, True])
def test_read_only_no_exec_repo_prompt_explains_available_read_tools(tmp_path: Path, managed: bool):
    if managed:
        _write_json(tmp_path / "nanobot.context.json", {"name": "shared-agents"})
    manager = ContextRepoManager.from_config(context_repos=[{
        "path": str(tmp_path), "readOnly": True, "allowExec": False,
    }])

    summary = manager.prompt_summary()

    assert "read-only" in summary
    assert "Shell commands cannot access" in summary
    assert "read_file" in summary and "list_dir" in summary
    assert "Normal writes to managed paths are autonomous" not in summary


@pytest.mark.parametrize("manifest_kind", ["relative", "relative-nested", "absolute", "directory"])
def test_missing_custom_manifest_is_not_downgraded_to_plain_repo(
    tmp_path: Path, manifest_kind: str,
) -> None:
    repo = tmp_path / "shared"
    repo.mkdir()
    if manifest_kind == "relative":
        manifest = "shared.json"
    elif manifest_kind == "relative-nested":
        manifest = "nested/shared.json"
    elif manifest_kind == "directory":
        manifest_dir = repo / "manifest-dir"
        manifest_dir.mkdir()
        manifest = str(manifest_dir)
    else:
        manifest = str(tmp_path / "shared.json")

    with pytest.raises(ValueError, match="manifest") as caught:
        ContextRepoManager.from_config(
            context_repos=[{"path": str(repo), "manifest": manifest}],
        )

    assert manifest in str(caught.value)


def test_existing_custom_manifest_override_loads(tmp_path: Path) -> None:
    repo = tmp_path / "shared"
    repo.mkdir()
    manifest = repo / "private.json"
    _write_json(manifest, {"name": "private-snapshot"})

    manager = ContextRepoManager.from_config(
        context_repos=[{"path": str(repo), "manifest": str(manifest)}],
    )

    assert manager.repos[0].managed is True
    assert manager.repos[0].name == "private-snapshot"
    assert manager.repos[0].manifest_path == manifest


@pytest.mark.parametrize("allow_exec", [None, True])
def test_default_and_true_allow_exec_preserve_read_only_exec_behavior(
    tmp_path: Path, allow_exec: bool | None,
) -> None:
    repo = tmp_path / "shared"
    repo.mkdir()
    (repo / "script.py").write_text("pass", encoding="utf-8")
    config = {"path": str(repo), "readOnly": True}
    if allow_exec is not None:
        config["allowExec"] = allow_exec
    manager = ContextRepoManager.from_config(context_repos=[config])
    policy = ResourceAccessPolicy(workspace=tmp_path / "workspace", context_manager=manager)

    assert manager.repos[0].allow_exec is True
    policy.validate_path(repo / "script.py", "exec")
    assert policy.validate_exec("python3 script.py", repo) is None


@pytest.mark.parametrize("order", ["parent-first", "child-first"])
def test_parent_exec_denial_cannot_be_bypassed_by_nested_context_or_target(
    tmp_path: Path, order: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    parent = tmp_path / "parent"
    nested = parent / "nested"
    nested.mkdir(parents=True)
    _write_json(parent / "nanobot.context.json", {
        "targetRepos": {
            "overlap": {"path": str(nested), "read": ["**"]},
        },
    })
    parent_config = {"path": str(parent), "allowExec": False}
    child_config = {"path": str(nested), "allowExec": True}
    configs = (
        [parent_config, child_config]
        if order == "parent-first"
        else [child_config, parent_config]
    )
    policy = ResourceAccessPolicy(
        workspace=workspace,
        context_manager=ContextRepoManager.from_config(context_repos=configs),
    )

    for candidate in (nested, nested / "missing.txt"):
        with pytest.raises(PermissionError, match="execution is disabled") as caught:
            policy.validate_path(candidate, "exec")
        assert type(caught.value) is PermissionError


@pytest.mark.parametrize("flags", [(True, False), (False, True), (False, False), (True, True)])
@pytest.mark.parametrize("use_alias", [False, True])
def test_duplicate_context_attachments_preserve_execution_denial(
    tmp_path: Path, flags: tuple[bool, bool], use_alias: bool,
) -> None:
    repo = tmp_path / "shared"
    repo.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    _write_json(repo / "nanobot.context.json", {})
    manager = ContextRepoManager.from_config(context_repos=[
        {"path": str(repo), "readOnly": True, "autoSync": False, "allowExec": flags[0]},
        {"path": str(alias if use_alias else repo), "allowExec": flags[1]},
    ])
    assert len(manager.repos) == 1
    assert manager.repos[0].read_only is True
    assert manager.repos[0].auto_sync is False
    policy = ResourceAccessPolicy(workspace=workspace, context_manager=manager)
    candidate = repo / "tool.py"
    if all(flags):
        policy.validate_path(candidate, "exec")
        assert policy.validate_exec(f"python3 {candidate}", workspace) is None
        assert policy.validate_exec("python3", repo) is None
    else:
        with pytest.raises(PermissionError, match="execution is disabled"):
            policy.validate_path(candidate, "exec")
        assert "execution is disabled" in str(
            policy.validate_exec(f"python3 {candidate}", workspace),
        )
        assert "execution is disabled" in str(policy.validate_exec("python3", repo))


def test_context_repo_manager_loads_manifest_and_contracts(tmp_path: Path) -> None:
    repo = tmp_path / "pota"
    (repo / "skills" / "seo-writer").mkdir(parents=True)
    (repo / "skills" / "seo-writer" / "SKILL.md").write_text("---\nname: seo-writer\n---\n", encoding="utf-8")
    (repo / "CONTEXT.md").write_text("# Pota\n", encoding="utf-8")
    _write_json(repo / "nanobot.context.json", {
        "name": "pota",
        "entrypoints": {
            "context": "CONTEXT.md",
            "skills": ["skills/**/SKILL.md"],
        },
        "writable": ["outputs/**"],
        "protected": ["config/**"],
        "stores": {
            "seo_pipeline": {
                "path": "data/seo-pipeline.json",
                "directEdit": False,
                "syncPaths": ["data/seo-pipeline.json"],
            }
        },
    })

    manager = ContextRepoManager.from_config(context_repos=[str(repo)])

    assert len(manager.repos) == 1
    loaded = manager.repos[0]
    assert loaded.managed is True
    assert loaded.name == "pota"
    assert manager.skill_roots() == [(repo / "skills").resolve()]
    assert manager.context_files() == [(repo / "CONTEXT.md").resolve()]
    assert "data/seo-pipeline.json" in (loaded.sync_include_patterns() or [])


def test_context_repo_manager_resolves_target_repos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "pota"
    target = tmp_path / "website"
    (repo / "outputs").mkdir(parents=True)
    (target / "content" / "blog").mkdir(parents=True)
    monkeypatch.setenv("POTA_WEBSITE_REPO", str(target))
    _write_json(repo / "nanobot.context.json", {
        "name": "pota",
        "writable": ["outputs/**"],
        "targetRepos": {
            "pota_website": {
                "type": "website",
                "pathEnv": "POTA_WEBSITE_REPO",
                "read": ["**"],
                "write": ["content/blog/**"],
                "proposalRequired": ["src/**"],
                "commands": {"build": "npm run build"},
            }
        },
    })

    manager = ContextRepoManager.from_config(context_repos=[str(repo)])
    targets = manager.target_repos()

    assert len(targets) == 1
    assert targets[0].name == "pota_website"
    assert targets[0].path == target.resolve()
    assert targets[0].resolved_from == "env:POTA_WEBSITE_REPO"
    assert target.resolve() in manager.read_roots()
    assert (target / "content" / "blog").resolve() in manager.write_roots()
    assert "pota_website" in manager.prompt_summary()


def test_context_repo_manager_resolves_target_repo_config_override(tmp_path: Path) -> None:
    repo = tmp_path / "pota"
    target = tmp_path / "website"
    repo.mkdir()
    target.mkdir()
    _write_json(repo / "nanobot.context.json", {
        "name": "pota",
        "targetRepos": {
            "pota_website": {
                "type": "website",
                "pathEnv": "POTA_WEBSITE_REPO",
                "write": ["content/blog/**"],
            }
        },
    })

    manager = ContextRepoManager.from_config(context_repos=[{
        "path": str(repo),
        "targetRepoPaths": {"pota_website": str(target)},
    }])

    targets = manager.target_repos()
    assert len(targets) == 1
    assert targets[0].path == target.resolve()
    assert targets[0].resolved_from == "path"


def test_resource_policy_enforces_managed_boundaries(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo = tmp_path / "pota"
    (repo / "outputs").mkdir(parents=True)
    (repo / "config").mkdir(parents=True)
    (repo / "data").mkdir(parents=True)
    _write_json(repo / "nanobot.context.json", {
        "name": "pota",
        "writable": ["outputs/**"],
        "protected": ["config/**"],
        "proposalRequired": ["outputs/strategy/**"],
        "stores": {
            "seo_pipeline": {
                "path": "data/seo-pipeline.json",
                "directEdit": False,
            }
        },
    })
    manager = ContextRepoManager.from_config(context_repos=[str(repo)])
    policy = ResourceAccessPolicy(workspace=workspace, context_manager=manager, restrict_to_workspace=True)

    policy.validate_path(repo / "outputs" / "draft.md", "write")
    policy.validate_path(repo / "CONTEXT.md", "read")

    with pytest.raises(PermissionError, match="requires a proposal"):
        policy.validate_path(repo / "outputs" / "strategy" / "launch.md", "write")
    with pytest.raises(PermissionError):
        policy.validate_path(repo / "config" / "gsc_oauth.json", "read")
    with pytest.raises(PermissionError):
        policy.validate_path(repo / "data" / "seo-pipeline.json", "write")
    with pytest.raises(PermissionError):
        policy.validate_path(repo / "random.md", "write")
    with pytest.raises(PermissionError):
        policy.validate_path(tmp_path / "outside.txt", "read")


@pytest.mark.asyncio
async def test_write_tool_can_write_managed_output_when_restricted(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo = tmp_path / "pota"
    (repo / "outputs").mkdir(parents=True)
    _write_json(repo / "nanobot.context.json", {
        "name": "pota",
        "writable": ["outputs/**"],
    })
    manager = ContextRepoManager.from_config(context_repos=[str(repo)])
    policy = ResourceAccessPolicy(workspace=workspace, context_manager=manager, restrict_to_workspace=True)
    tool = WriteFileTool(
        workspace=workspace,
        allowed_dir=workspace,
        extra_allowed_dirs=policy.extra_write_dirs(),
        resource_policy=policy,
    )

    result = await tool.execute(path=str(repo / "outputs" / "draft.md"), content="hello")

    assert "Successfully wrote" in result
    assert (repo / "outputs" / "draft.md").read_text(encoding="utf-8") == "hello"
    assert (repo / "outputs" / "draft.md").resolve() in policy.touched_paths


@pytest.mark.asyncio
async def test_write_tool_can_write_allowed_target_repo_path_when_restricted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo = tmp_path / "pota"
    target = tmp_path / "website"
    (repo / "outputs").mkdir(parents=True)
    (target / "content" / "blog").mkdir(parents=True)
    (target / "src").mkdir(parents=True)
    monkeypatch.setenv("POTA_WEBSITE_REPO", str(target))
    _write_json(repo / "nanobot.context.json", {
        "name": "pota",
        "targetRepos": {
            "pota_website": {
                "type": "website",
                "pathEnv": "POTA_WEBSITE_REPO",
                "read": ["**"],
                "write": ["content/blog/**", "public/images/blog/**"],
                "protected": [".env", ".env.*"],
                "proposalRequired": ["src/**", "package.json"],
            }
        },
    })
    manager = ContextRepoManager.from_config(context_repos=[str(repo)])
    policy = ResourceAccessPolicy(workspace=workspace, context_manager=manager, restrict_to_workspace=True)
    tool = WriteFileTool(
        workspace=workspace,
        allowed_dir=workspace,
        extra_allowed_dirs=policy.extra_write_dirs(),
        resource_policy=policy,
    )

    result = await tool.execute(path=str(target / "content" / "blog" / "draft.md"), content="hello")

    assert "Successfully wrote" in result
    assert (target / "content" / "blog" / "draft.md").read_text(encoding="utf-8") == "hello"
    with pytest.raises(PermissionError, match="protected"):
        policy.validate_path(target / ".env", "read")
    with pytest.raises(PermissionError, match="requires a proposal"):
        policy.validate_path(target / "src" / "layout.tsx", "write")
    with pytest.raises(PermissionError, match="not in a writable target repo area"):
        policy.validate_path(target / "README.md", "write")
