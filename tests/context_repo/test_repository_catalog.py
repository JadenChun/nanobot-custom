"""Existing paths, on-demand context and automatic knowledge save boundaries."""

import asyncio
import json
from pathlib import Path

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool
from nanobot.agent.tools.repositories import RepositoryCatalogTool, SaveKnowledgeTool
from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.context_repo.catalog import RepositoryCatalog, RepositoryEntry
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
)
from tests.context_repo.test_git_sync_safe_push import (
    _bootstrap_remote_with_initial,
    _configure_user,
    _git,
    _git_out,
)


def _repo(path: Path):
    path.mkdir()
    _git(path, "init")
    _configure_user(path)
    return path


@pytest.mark.asyncio
async def test_agent_remembers_existing_repos_without_manifest_or_duplicate_clone(tmp_path):
    workspace = tmp_path / "agent"
    workspace.mkdir()
    knowledge = _repo(tmp_path / "knowledge")
    project = _repo(tmp_path / "project")
    catalog = RepositoryCatalog(tmp_path / "runtime")
    tool = RepositoryCatalogTool(catalog, ReadFileTool(workspace=workspace))
    assert not getattr(await tool.execute("remember", name="knowledge", path=str(knowledge), kind="knowledge"), "is_error", False)
    assert not getattr(await tool.execute("remember", name="project", path=str(project), knowledge_repos=["knowledge"]), "is_error", False)
    reloaded = RepositoryCatalog(catalog.root)
    assert reloaded.get("project").knowledge_repos == ["knowledge"]
    assert reloaded.get("project").path == str(project)
    assert not (project / "nanobot.context.json").exists()
    assert list(workspace.iterdir()) == []
    failure = await tool.execute("forget", name="knowledge")
    assert failure.is_error  # associations must not silently dangle
    assert len(catalog.entries()) == 2


@pytest.mark.asyncio
async def test_catalog_does_not_grant_access_outside_current_scope(tmp_path):
    workspace = _repo(tmp_path / "agent")
    outside = _repo(tmp_path / "outside")
    catalog = RepositoryCatalog(tmp_path / "runtime")
    catalog.update(RepositoryEntry(name="outside", path=str(outside), kind="knowledge"))
    reader, writer = ReadFileTool(workspace=workspace), WriteFileTool(workspace=workspace)
    token = bind_workspace_scope(build_workspace_scope(workspace, "restricted"))
    try:
        index = RepositoryCatalogTool(catalog, reader)
        assert json.loads(await index.execute("list")) == []
        assert (await index.execute("remember", name="outside", path=str(outside))).is_error
        assert (await SaveKnowledgeTool(catalog, writer).execute("outside", "notes.md", "decision")).is_error
    finally:
        reset_workspace_scope(token)
    assert not (outside / "notes.md").exists()


@pytest.mark.asyncio
async def test_knowledge_append_is_idempotent_and_sync_is_knowledge_only(tmp_path):
    remote, _seed, knowledge = _bootstrap_remote_with_initial(tmp_path)
    project = _repo(tmp_path / "project")
    workspace = _repo(tmp_path / "agent")
    catalog = RepositoryCatalog(tmp_path / "runtime")
    catalog.update(RepositoryEntry(name="notes", path=str(knowledge), kind="knowledge",
                                   auto_sync=True, sync_paths=["decisions/**"]))
    catalog.update(RepositoryEntry(name="project", path=str(project), knowledge_repos=["notes"]))
    tool = SaveKnowledgeTool(catalog, WriteFileTool(workspace=workspace))
    result = json.loads(await tool.execute("notes", "decisions/project.md", "Use the catalog."))
    assert result["saved"] is True and result["synced"] is True
    commit = _git_out(knowledge, "rev-parse", "HEAD")
    await tool.execute("notes", "decisions/project.md", "Use the catalog.")
    assert _git_out(knowledge, "rev-parse", "HEAD") == commit
    assert _git_out(remote, "show", "main:decisions/project.md") == "Use the catalog."
    assert (await tool.execute("project", "notes.md", "wrong destination")).is_error
    assert not (project / "notes.md").exists()
    assert _git_out(project, "status", "--porcelain") == ""


@pytest.mark.asyncio
async def test_offline_sync_still_saves_locally_and_can_be_retried(tmp_path):
    _remote, _seed, knowledge = _bootstrap_remote_with_initial(tmp_path)
    workspace = _repo(tmp_path / "agent")
    _git(knowledge, "remote", "set-url", "origin", str(tmp_path / "missing.git"))
    catalog = RepositoryCatalog(tmp_path / "runtime")
    catalog.update(RepositoryEntry(name="notes", path=str(knowledge), kind="knowledge", auto_sync=True, sync_paths=["*.md"]))
    result = json.loads(await SaveKnowledgeTool(catalog, WriteFileTool(workspace=workspace)).execute("notes", "note.md", "Keep this decision."))
    assert result["saved"] is True and result["synced"] is False and result["sync_state"] == "pending"
    assert (knowledge / "note.md").read_text() == "Keep this decision.\n"
    assert _git_out(knowledge, "show", "HEAD:note.md") == "Keep this decision."


@pytest.mark.asyncio
async def test_concurrent_notes_do_not_lose_each_other(tmp_path):
    workspace = _repo(tmp_path / "agent")
    knowledge = _repo(tmp_path / "knowledge")
    catalog = RepositoryCatalog(tmp_path / "runtime")
    catalog.update(RepositoryEntry(name="notes", path=str(knowledge), kind="knowledge"))
    tool = SaveKnowledgeTool(catalog, WriteFileTool(workspace=workspace))
    outcomes = await asyncio.gather(*(tool.execute("notes", "notes.md", f"Decision {i}") for i in range(5)))
    assert all(json.loads(result)["saved"] for result in outcomes)
    assert sorted(line for line in (knowledge / "notes.md").read_text().splitlines() if line) == [f"Decision {i}" for i in range(5)]


@pytest.mark.asyncio
async def test_knowledge_does_not_escape_or_write_runtime_artifacts(tmp_path):
    workspace = _repo(tmp_path / "agent")
    knowledge = _repo(tmp_path / "knowledge")
    catalog = RepositoryCatalog(tmp_path / "runtime")
    catalog.update(RepositoryEntry(name="notes", path=str(knowledge), kind="knowledge"))
    tool = SaveKnowledgeTool(catalog, WriteFileTool(workspace=workspace))
    assert (await tool.execute("notes", "../escape.md", "decision")).is_error
    (knowledge / "outside").symlink_to(tmp_path, target_is_directory=True)
    assert (await tool.execute("notes", "outside/escape.md", "decision")).is_error
    catalog.update(RepositoryEntry(name="runtime", path=str(catalog.root), kind="knowledge"))
    assert (await tool.execute("runtime", "transcript.md", "private runtime data")).is_error
    assert not (tmp_path / "escape.md").exists()


@pytest.mark.parametrize("path", ["config/private.md", "outputs/strategy/decision.md",
                                 "outputs/state.json", "scripts/helper.py"])
async def test_automatic_knowledge_save_retains_managed_repository_policy(tmp_path, path):
    workspace, knowledge = _repo(tmp_path / "agent"), _repo(tmp_path / "knowledge")
    (knowledge / "nanobot.context.json").write_text(json.dumps({
        "writable": ["outputs/**"], "protected": ["config/**", "scripts/**"],
        "proposalRequired": ["outputs/strategy/**"],
        "stores": {"state": {"path": "outputs/state.json", "directEdit": False}},
    }))
    policy = ResourceAccessPolicy(workspace=workspace,
        context_manager=ContextRepoManager.from_config(context_repos=[str(knowledge)]))
    catalog = RepositoryCatalog(tmp_path / "runtime")
    catalog.update(RepositoryEntry(name="notes", path=str(knowledge), kind="knowledge"))
    tool = SaveKnowledgeTool(catalog, WriteFileTool(workspace=workspace, resource_policy=policy))
    assert (await tool.execute("notes", path, "Selected decision")).is_error
    assert not (knowledge / path).exists()
    result = json.loads(await tool.execute("notes", "outputs/notes.md", "Selected decision"))
    assert result["saved"] is True


def test_prompt_catalog_stays_small_and_context_is_read_on_demand_after_switch(tmp_path):
    workspace = _repo(tmp_path / "agent")
    project_a, project_b = _repo(tmp_path / "a"), _repo(tmp_path / "b")
    knowledge = _repo(tmp_path / "knowledge")
    sessions_root = tmp_path / "instance" / "sessions"
    catalog = RepositoryCatalog.for_workspace(workspace, sessions_root=sessions_root)
    catalog.update(RepositoryEntry(name="notes", path=str(knowledge), kind="knowledge"))
    for name, path in (("a", project_a), ("b", project_b)):
        (path / "AGENTS.md").write_text(f"Instructions for {name} only")
        catalog.update(RepositoryEntry(name=name, path=str(path), knowledge_repos=["notes"]))
    (knowledge / "notes.md").write_text("Durable fact loaded only when requested")
    builder = ContextBuilder(workspace, sessions_root=sessions_root)
    prompts = [builder.build_system_prompt(workspace=path) for path in (project_a, project_b, workspace, project_a)]
    assert "Instructions for a only" in prompts[0] and "Instructions for b only" not in prompts[0]
    assert "Instructions for b only" in prompts[1] and "Instructions for a only" not in prompts[1]
    assert "Instructions for a only" not in prompts[2]
    assert "Instructions for a only" in prompts[3]
    assert all("Durable fact loaded only when requested" not in prompt for prompt in prompts)
    assert all(str(knowledge) in prompt for prompt in prompts)
    assert len(catalog.prompt_summary()) < 1500
