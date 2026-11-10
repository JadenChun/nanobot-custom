from __future__ import annotations

import subprocess
from pathlib import Path


def _classify(changed: str) -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(repo / "scripts/check_ci_paths.sh")],
        input=changed,
        text=True,
        capture_output=True,
        check=True,
    )
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def test_migration_only_change_runs_python_jobs() -> None:
    result = _classify("docs/migration/run-p0-baseline.sh\n")

    assert result == {
        "python_required": "true",
        "webui_required": "false",
        "tui_required": "false",
        "docker_required": "false",
    }


def test_documentation_only_change_runs_no_build_jobs() -> None:
    result = _classify("docs/implementation-progress.md\n")

    assert result == {
        "python_required": "false",
        "webui_required": "false",
        "tui_required": "false",
        "docker_required": "false",
    }


def test_workflow_change_runs_every_job() -> None:
    result = _classify(".github/workflows/ci.yml\n")

    assert result == {
        "python_required": "true",
        "webui_required": "true",
        "tui_required": "true",
        "docker_required": "true",
    }
