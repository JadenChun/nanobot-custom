"""Regression coverage for A3-6 AC6.3 skill provenance and scope boundaries."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from nanobot.agent.skills import BUILTIN_SKILLS_DIR, SkillsLoader
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool
from nanobot.agent.tools.repositories import SaveKnowledgeTool
from nanobot.context_repo.catalog import RepositoryCatalog, RepositoryEntry
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
)
from nanobot.utils.workspace_prompts import load_workspace_prompt_override


def _skill_entry(entries, name: str):
    matches = [
        entry
        for entry in entries
        if (entry["name"] if isinstance(entry, dict) else entry.name) == name
    ]
    assert len(matches) == 1, f"expected exactly one {name!r} entry, got {matches!r}"
    return matches[0]


def _entry_field(entry, field: str):
    return entry[field] if isinstance(entry, dict) else getattr(entry, field)


def _load_video_script():
    script = (
        Path(__file__).resolve().parents[2]
        / "nanobot"
        / "skills"
        / "video-understanding"
        / "scripts"
        / "extract_keyframes.py"
    )
    assert script.is_file()
    spec = importlib.util.spec_from_file_location("extract_keyframes_a3_6", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_malformed_high_priority_skill_does_not_hide_valid_candidate(tmp_path: Path):
    workspace = tmp_path / "workspace"
    builtin = tmp_path / "builtin"
    for root in (workspace / "skills" / "dup", builtin / "dup"):
        root.mkdir(parents=True)

    (builtin / "dup" / "SKILL.md").write_text(
        "---\nname: dup\ndescription: valid builtin\n---\n# Builtin",
        encoding="utf-8",
    )
    (workspace / "skills" / "dup" / "SKILL.md").write_text(
        "---\nname: [unterminated\n---\n# Broken",
        encoding="utf-8",
    )

    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)
    entries = loader.list_skills(filter_unavailable=False)
    assert _entry_field(_skill_entry(entries, "dup"), "source") == "builtin"

    (workspace / "skills" / "dup" / "SKILL.md").write_text(
        "# Workspace override",
        encoding="utf-8",
    )
    entries = loader.list_skills(filter_unavailable=False)
    assert _entry_field(_skill_entry(entries, "dup"), "source") == "workspace"


@pytest.mark.asyncio
async def test_selected_project_collision_cannot_replace_bundled_body(tmp_path: Path):
    assert (BUILTIN_SKILLS_DIR / "memory" / "SKILL.md").is_file()

    agent = tmp_path / "agent"
    agent.mkdir()
    project = tmp_path / "project"
    (project / "skills" / "memory").mkdir(parents=True)
    (project / "skills" / "memory" / "SKILL.md").write_text(
        "# Memory\n\nPROJECT-COLLISION\n",
        encoding="utf-8",
    )

    token = bind_workspace_scope(build_workspace_scope(project, "restricted"))
    try:
        text = await ReadFileTool(workspace=agent).execute("skills/memory/SKILL.md")
    finally:
        reset_workspace_scope(token)

    assert "PROJECT-COLLISION" not in str(text)
    assert "Memory" in str(text)


def test_video_storyboard_is_local_only(tmp_path: Path):
    module = _load_video_script()
    frame = {
        "index": 1,
        "timestamp": "00:00:01",
        "reason": "scene",
        "path": "frames/frame.jpg",
        "ocr_excerpt": "hello",
    }

    storyboard = module._generate_semantic_storyboard(
        [frame],
        tmp_path,
        semantic_enabled=True,
        semantic_model="ambient-model",
        semantic_max_frames=12,
    )

    assert storyboard["generator"] == "fallback"
    assert "ambient-model" not in json.dumps(storyboard)


@pytest.mark.asyncio
async def test_knowledge_write_requires_selected_project_association(tmp_path: Path):
    agent = tmp_path / "agent"
    agent.mkdir()
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    project_a.mkdir()
    project_b.mkdir()
    knowledge = project_a / "knowledge"
    knowledge.mkdir()

    catalog = RepositoryCatalog(tmp_path / "runtime")
    catalog.update(
        RepositoryEntry(name="notes", path=str(knowledge), kind="knowledge")
    )
    catalog.update(
        RepositoryEntry(
            name="a",
            path=str(project_a),
            kind="project",
            knowledge_repos=["notes"],
        )
    )
    catalog.update(
        RepositoryEntry(name="b", path=str(project_b), kind="project")
    )

    token = bind_workspace_scope(build_workspace_scope(project_a, "restricted"))
    try:
        result = await SaveKnowledgeTool(
            catalog, WriteFileTool(workspace=agent)
        ).execute("notes", "note.md", "Decision")
        assert json.loads(result)["saved"] is True
    finally:
        reset_workspace_scope(token)

    token = bind_workspace_scope(build_workspace_scope(project_b, "restricted"))
    try:
        result = await SaveKnowledgeTool(
            catalog, WriteFileTool(workspace=agent)
        ).execute("notes", "note-b.md", "Decision")
        assert getattr(result, "is_error", False) is True
        assert not (knowledge / "note-b.md").exists()
    finally:
        reset_workspace_scope(token)


def test_escaping_prompt_override_symlink_is_refused(tmp_path: Path):
    workspace = tmp_path / "workspace"
    (workspace / "prompts").mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("ESCAPED INSTRUCTION", encoding="utf-8")
    (workspace / "prompts" / "dream.md").symlink_to(outside)

    text, chars = load_workspace_prompt_override(
        workspace / "prompts" / "dream.md", workspace=workspace
    )

    assert text is None
    assert chars == 0
