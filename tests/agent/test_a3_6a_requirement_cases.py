"""Requirements-only cases C1-C4 (R1 identity precedence, R2 alternatives).

Provenance: a3-evidence/a3-reconciliation-2026-11-14/req-a36a/CASES.md.
Scenarios, inputs and expected outcomes are preserved from that file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.skills import SkillsLoader


def _write_skill(root: Path, name: str, frontmatter: list[str]) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    path = skill_dir / "SKILL.md"
    path.write_text(
        "\n".join(["---", *frontmatter, "---", "", "# Skill"]),
        encoding="utf-8",
    )
    return path


def _valid_frontmatter(name: str, description: str) -> list[str]:
    return [f"name: {name}", f"description: {description}"]


# C1 - Invalid workspace candidate does not shadow a valid bundled skill.
@pytest.mark.parametrize(
    ("directory_name", "workspace_frontmatter"),
    [
        pytest.param("dup", ["name: dup"], id="V1-missing-description"),
        pytest.param("dup", ["name: dup", 'description: ""'], id="V2-empty-description"),
        pytest.param(
            "dup",
            ["name: dup", "description: " + ("a" * 1025)],
            id="V3-oversize-description",
        ),
        pytest.param(
            "a" * 65,
            ["name: " + ("a" * 65), "description: Workspace skill."],
            id="V4-name-over-64",
        ),
        pytest.param(
            "dup",
            ["name: other", "description: Workspace skill."],
            id="V5-name-differs-from-directory",
        ),
    ],
)
def test_c1_invalid_workspace_candidate_does_not_shadow_valid_bundled_skill(
    tmp_path: Path,
    directory_name: str,
    workspace_frontmatter: list[str],
) -> None:
    workspace = tmp_path / "workspace"
    builtin = tmp_path / "builtin"
    (workspace / "skills").mkdir(parents=True)
    builtin.mkdir(parents=True)
    _write_skill(
        builtin,
        directory_name,
        _valid_frontmatter(directory_name, "Bundled skill."),
    )
    _write_skill(workspace / "skills", directory_name, workspace_frontmatter)
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    listing = loader.list_skills(filter_unavailable=False)

    matches = [entry for entry in listing if entry["name"] == directory_name]
    assert len(matches) == 1
    assert matches[0]["source"] == "builtin"
    assert matches[0]["name"] == directory_name


# C2 - Valid workspace override is selected.
def test_c2_valid_workspace_override_is_selected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    builtin = tmp_path / "builtin"
    (workspace / "skills").mkdir(parents=True)
    builtin.mkdir(parents=True)
    _write_skill(builtin, "dup", _valid_frontmatter("dup", "Bundled skill."))
    _write_skill(
        workspace / "skills",
        "dup",
        _valid_frontmatter("dup", "Workspace skill."),
    )
    loader = SkillsLoader(workspace, builtin_skills_dir=builtin)

    listing = loader.list_skills(filter_unavailable=False)

    matches = [entry for entry in listing if entry["name"] == "dup"]
    assert len(matches) == 1
    assert matches[0]["source"] == "workspace"
    assert loader.get_skill_description("dup") == "Workspace skill."
    assert all(entry["source"] != "builtin" for entry in matches)


def _alternatives_loader(tmp_path: Path) -> SkillsLoader:
    workspace = tmp_path / "workspace"
    skills_root = workspace / "skills"
    skills_root.mkdir(parents=True)
    _write_skill(
        skills_root,
        "platform-skill",
        [
            "name: platform-skill",
            "description: Platform skill.",
            'metadata: {"nanobot":{"requires":{"anyBins":["npx","adb"]}}}',
        ],
    )
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    return SkillsLoader(workspace, builtin_skills_dir=builtin)


# C3 - One satisfied alternative makes the skill available.
@pytest.mark.parametrize("present", ["npx", "adb"])
def test_c3_one_satisfied_alternative_makes_skill_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, present: str
) -> None:
    monkeypatch.setattr(
        "nanobot.agent.skills.shutil.which",
        lambda command: f"/fake/{command}" if command == present else None,
    )
    loader = _alternatives_loader(tmp_path)

    available, _reason = loader.get_skill_availability("platform-skill")

    listed = {entry["name"] for entry in loader.list_skills(filter_unavailable=True)}
    assert "platform-skill" in listed
    assert available is True


# C4 - No satisfied alternative makes the skill unavailable.
def test_c4_no_satisfied_alternative_makes_skill_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nanobot.agent.skills.shutil.which", lambda _command: None)
    loader = _alternatives_loader(tmp_path)

    available, reason = loader.get_skill_availability("platform-skill")

    assert available is False
    assert reason


def test_mobile_skills_declare_platform_alternatives() -> None:
    """The two mobile skills require npx plus one platform CLI, not all three."""
    builtin = Path(__file__).resolve().parents[2] / "nanobot" / "skills"
    loader = SkillsLoader(builtin.parent, builtin_skills_dir=builtin)
    for name in ("mobile-app-testing", "mobile-dogfood"):
        requirements = loader.get_skill_requirements(name)
        assert requirements["bins"] == ["npx"]
        assert requirements["any_bins"] == ["xcrun", "adb"]
