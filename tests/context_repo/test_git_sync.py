from __future__ import annotations

import subprocess
from pathlib import Path

from nanobot.utils.git_sync import _select_changed_paths


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def test_select_changed_paths_filters_include_and_exclude(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "outputs").mkdir()
    (repo / "config").mkdir()
    (repo / "data").mkdir()
    (repo / "outputs" / "draft.md").write_text("draft", encoding="utf-8")
    (repo / "config" / "secret.json").write_text("secret", encoding="utf-8")
    (repo / "data" / "seo-pipeline.json").write_text("{}", encoding="utf-8")

    selected = _select_changed_paths(
        repo,
        include_paths=["outputs/**", "data/seo-pipeline.json"],
        exclude_paths=["config/**", "**/*secret*.json"],
    )

    assert sorted(selected) == ["data/seo-pipeline.json", "outputs/draft.md"]
