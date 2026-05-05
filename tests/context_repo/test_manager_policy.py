from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.agent.tools.filesystem import WriteFileTool
from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
