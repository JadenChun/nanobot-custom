"""Requirement case U5: repo entrypoints cannot escape the admitted repo root."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.context_repo import ContextRepoManager


def _skill(path: Path, marker: str) -> None:
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(marker, encoding="utf-8")


def _manifest(repo: Path, skills: str, context: str) -> None:
    (repo / "nanobot.context.json").write_text(
        json.dumps(
            {
                "name": "entrypoint-case",
                "entrypoints": {
                    "skills": [skills, "skills/**/SKILL.md"],
                    "context": context,
                },
            }
        ),
        encoding="utf-8",
    )


def _assembled_text(manager: ContextRepoManager) -> str:
    parts = [path.read_text(encoding="utf-8") for path in manager.context_files()]
    for root in manager.skill_roots():
        parts.extend(skill.read_text(encoding="utf-8") for skill in sorted(root.glob("**/SKILL.md")))
    return "\n".join(parts)


@pytest.mark.parametrize(
    ("case", "skills_entry", "context_entry", "use_symlink"),
    [
        ("C6a-dotdot", "../outside/skills/**/SKILL.md", "../outside/CONTEXT-escaped.md", False),
        ("C6b-symlink", "./link/skills/**/SKILL.md", "./link/CONTEXT-escaped.md", True),
    ],
)
def test_repo_entrypoint_cannot_escape_admitted_root(
    tmp_path: Path,
    case: str,
    skills_entry: str,
    context_entry: str,
    use_symlink: bool,
) -> None:
    repo = tmp_path / "repoR"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    _skill(outside / "skills" / "evil", "OUTSIDE_SKILL_MARKER")
    (outside / "CONTEXT-escaped.md").write_text("OUTSIDE_PROMPT_TEXT", encoding="utf-8")
    _skill(repo / "skills" / "inside", "INSIDE_SKILL_MARKER")
    if use_symlink:
        (repo / "link").symlink_to(outside, target_is_directory=True)
    _manifest(repo, skills_entry, context_entry)

    manager = ContextRepoManager.from_config(context_repos=[str(repo)])
    roots = manager.skill_roots()
    files = manager.context_files()

    outside_root = outside.resolve()
    assert not any(root.resolve().is_relative_to(outside_root) for root in roots)
    assert all(root.resolve().is_relative_to(repo.resolve()) for root in roots)
    assert all(path.resolve().is_relative_to(repo.resolve()) for path in files)
    text = _assembled_text(manager)
    assert "OUTSIDE_PROMPT_TEXT" not in text
    assert "OUTSIDE_SKILL_MARKER" not in text

    assert (repo / "skills").resolve() in roots
    assert "INSIDE_SKILL_MARKER" in text
