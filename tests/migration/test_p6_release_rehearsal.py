from __future__ import annotations

import hashlib
import io
import os
import stat
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml

from scripts.p6_release_rehearsal import (
    ReleaseArtifactError,
    _validate_report,
    inspect_archive,
    inspect_dependency_inputs,
    inspect_service_unit,
    main,
    run_state_rehearsal,
)

_PACKAGE_MEMBERS = (
    "nanobot/__init__.py",
    "nanobot/templates/IDENTITY.md",
    "nanobot/templates/TOOLS.md",
    "nanobot/skills/README.md",
    "nanobot/skills/desktop-use/SKILL.md",
    "nanobot/skills/mobile-app-testing/SKILL.md",
    "nanobot/skills/mobile-dogfood/SKILL.md",
    "nanobot/skills/webapp-testing/SKILL.md",
    "nanobot/skills/video-understanding/SKILL.md",
    "nanobot/skills/video-understanding/scripts/extract_keyframes.py",
    "nanobot/crawler/worker.py",
    "nanobot/channels/telegram/webui/index.ts",
    "nanobot/channels/websocket/webui/index.ts",
    "nanobot/web/dist/index.html",
)


def test_candidate_ci_trigger_preserves_native_matrix_and_strict_policy() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.load((root / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    events = workflow["on"]
    assert "workflow_dispatch" in events
    assert events["push"]["branches"] == [
        "main", "staging", "codex/upstream-migration", "codex/harness-migration-candidate",
    ]
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]["include"]
    assert [(row["os"], row["python-version"]) for row in matrix] == [
        ("ubuntu-latest", "3.11"), ("ubuntu-latest", "3.14"), ("windows-latest", "3.14"),
    ]
    assert workflow["jobs"]["windows_process"]["runs-on"] == "windows-latest"
    steps = workflow["jobs"]["test"]["steps"]
    assert any(step.get("run") == "uv run --no-sync basedpyright" for step in steps)
    assert any(step.get("run") == "uv sync --all-extras --dev" for step in steps)
    assert any(step.get("run") == "uv run --no-sync python -m scripts.install_channel_dependencies --all-channels" for step in steps)
    assert all(job["runs-on"] in {"ubuntu-latest", "windows-latest", "${{ matrix.os }}"}
               for job in workflow["jobs"].values())


def _write_zip(
    path: Path,
    members: tuple[str, ...],
    *,
    metadata_root: str = "nanobot_ai-0.3.0.dist-info",
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, member)
        archive.writestr(f"{metadata_root}/METADATA", "Name: nanobot-ai\n")
        archive.writestr(f"{metadata_root}/RECORD", "")


def _write_sdist(path: Path, members: tuple[str, ...]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for member in members + (
            "nanobot-ai-0.3.0/README.md",
            "nanobot-ai-0.3.0/LICENSE",
            "nanobot-ai-0.3.0/THIRD_PARTY_NOTICES.md",
            "nanobot-ai-0.3.0/pyproject.toml",
            "nanobot-ai-0.3.0/hatch_build.py",
        ):
            data = member.encode()
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_sdist_include_entries_are_exact_project_roots() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"] == [
        "/nanobot/",
        "/nanobot/web/dist/",
        "/hatch_build.py",
        "/README.md",
        "/LICENSE",
        "/THIRD_PARTY_NOTICES.md",
        "/pyproject.toml",
    ]


def test_archive_inventory_covers_wheel_and_sdist_assets(tmp_path: Path) -> None:
    wheel = tmp_path / "nanobot_ai-0.3.0-py3-none-any.whl"
    sdist = tmp_path / "nanobot-ai-0.3.0.tar.gz"
    wheel_members = _PACKAGE_MEMBERS
    sdist_members = tuple(f"nanobot-ai-0.3.0/{member}" for member in _PACKAGE_MEMBERS)
    _write_zip(wheel, wheel_members)
    _write_sdist(sdist, sdist_members)

    wheel_report = inspect_archive(wheel)
    sdist_report = inspect_archive(sdist)

    assert wheel_report.format == "wheel"
    assert sdist_report.format == "sdist"
    assert wheel_report.missing == ()
    assert sdist_report.missing == ()
    assert wheel_report.sha256 == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert sdist_report.sha256 == hashlib.sha256(sdist.read_bytes()).hexdigest()
    assert wheel_report.sha256 != sdist_report.sha256
    assert wheel_report.forbidden == ()
    assert sdist_report.forbidden == ()


def test_source_manifest_retains_required_release_assets() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    include = pyproject["tool"]["hatch"]["build"]["include"]

    for pattern in (
        "nanobot/**/*.py",
        "nanobot/templates/**/*.md",
        "nanobot/skills/**/*.md",
        "nanobot/skills/**/*.sh",
        "nanobot/skills/**/*.py",
        "nanobot/channels/*/webui/**/*",
        "nanobot/web/dist/**/*",
    ):
        assert pattern in include
    for relative in _PACKAGE_MEMBERS:
        if relative == "nanobot/web/dist/index.html":
            # The generated bundle is intentionally absent in this offline
            # checkout; the Hatch include/artifact rules are the gate here.
            continue
        assert (root / relative).is_file(), relative


def test_archive_inventory_rejects_local_state_and_traversal(tmp_path: Path) -> None:
    wheel = tmp_path / "nanobot_ai-0.3.0-py3-none-any.whl"
    _write_zip(
        wheel,
        tuple(f"nanobot_ai-0.3.0/{member}" for member in _PACKAGE_MEMBERS)
        + ("nanobot_ai-0.3.0/.env", "../secrets.txt"),
    )

    with pytest.raises(ReleaseArtifactError, match="forbidden|unsafe"):
        inspect_archive(wheel)


def test_wheel_inventory_requires_canonical_package_paths(tmp_path: Path) -> None:
    wheel = tmp_path / "nanobot_ai-0.3.0-py3-none-any.whl"
    _write_zip(wheel, tuple(f"nanobot_ai-0.3.0/{member}" for member in _PACKAGE_MEMBERS))

    report = inspect_archive(wheel)

    assert "nanobot/__init__.py" in report.missing
    assert "nanobot_ai-0.3.0.dist-info/METADATA" not in report.missing


def test_wheel_inventory_requires_filename_derived_metadata_root(tmp_path: Path) -> None:
    wheel = tmp_path / "nanobot_ai-0.3.0-py3-none-any.whl"
    _write_zip(wheel, _PACKAGE_MEMBERS, metadata_root="other-0.3.0.dist-info")

    report = inspect_archive(wheel)

    assert report.missing == (
        "nanobot_ai-0.3.0.dist-info/METADATA",
        "nanobot_ai-0.3.0.dist-info/RECORD",
    )


def test_sdist_inventory_requires_one_filename_derived_root(tmp_path: Path) -> None:
    sdist = tmp_path / "nanobot-ai-0.3.0.tar.gz"
    _write_sdist(sdist, tuple(f"other-root/{member}" for member in _PACKAGE_MEMBERS))

    with pytest.raises(ReleaseArtifactError, match="outside distribution root"):
        inspect_archive(sdist)


def test_archive_inventory_rejects_zip_symlinks_and_required_directories(tmp_path: Path) -> None:
    symlink = tmp_path / "nanobot_ai-0.3.0-py3-none-any.whl"
    with zipfile.ZipFile(symlink, "w") as archive:
        info = zipfile.ZipInfo("nanobot/__init__.py")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "elsewhere")
    with pytest.raises(ReleaseArtifactError, match="symlink"):
        inspect_archive(symlink)

    directory = tmp_path / "nanobot_ai-0.3.0-py3-none-any.whl"
    _write_zip(directory, tuple(member for member in _PACKAGE_MEMBERS if member != "nanobot/web/dist/index.html"))
    with zipfile.ZipFile(directory, "a") as archive:
        archive.writestr("nanobot/web/dist/index.html", "")
        info = archive.getinfo("nanobot/web/dist/index.html")
        info.external_attr = (stat.S_IFDIR | 0o755) << 16
    report = inspect_archive(directory)
    assert report.missing == ("nanobot/web/dist/index.html",)


def test_archive_inventory_rejects_concurrent_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "nanobot_ai-0.3.0-py3-none-any.whl"
    _write_zip(wheel, _PACKAGE_MEMBERS)
    replacement = tmp_path / "replacement.whl"
    replacement.write_bytes(b"replacement")
    original_open = Path.open

    class _ReplacingHandle:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def fileno(self):
            return self._handle.fileno()

        def read(self, *args):
            data = self._handle.read(*args)
            os.replace(replacement, wheel)
            return data

    def open_and_replace(self, *args, **kwargs):
        handle = original_open(self, *args, **kwargs)
        if self == wheel and args and args[0] == "rb":
            return _ReplacingHandle(handle)
        return handle

    monkeypatch.setattr(Path, "open", open_and_replace)
    with pytest.raises(ReleaseArtifactError, match="replaced|changed"):
        inspect_archive(wheel)


def test_dependency_inventory_records_unlocked_release_inputs() -> None:
    report = inspect_dependency_inputs(Path(__file__).resolve().parents[2])

    assert report.project == "nanobot-ai"
    assert report.dependencies
    assert "crawler-worker" in report.optional_dependencies
    assert report.lockfile_present == (Path(__file__).resolve().parents[2] / "uv.lock").is_file()
    assert report.lockfile_ignored is True
    assert report.lockfile_path == "uv.lock"


def test_service_inventory_requires_foreground_gateway_and_file_logging() -> None:
    root = Path(__file__).resolve().parents[2]
    report = inspect_service_unit(root / "deploy/systemd/nanobot-gateway.service")

    assert report.valid is True
    assert report.missing == ()
    assert report.forbidden == ()


def test_service_inventory_ignores_comments_and_directives_outside_service(tmp_path: Path) -> None:
    unit = tmp_path / "commented.service"
    unit.write_text(
        """
[Unit]
ExecStart=/opt/nanobot gateway --config /ignored --log-file /ignored
# ExecStart=/opt/nanobot gateway --config /comment --log-file /comment

[Service]
# EnvironmentFile=/comment
WorkingDirectory=
ExecStart=/opt/nanobot gateway --config /etc/nanobot.json
EnvironmentFile=/etc/nanobot.env
""",
        encoding="utf-8",
    )

    report = inspect_service_unit(unit)

    assert report.valid is False
    assert "--log-file " in report.missing


def test_service_inventory_honors_execstart_reset_and_rejects_duplicates(tmp_path: Path) -> None:
    reset = tmp_path / "reset.service"
    reset.write_text(
        """
[Service]
ExecStart=/opt/old/nanobot gateway --config /old --log-file /old
ExecStart=
ExecStart=/opt/nanobot gateway --config /etc/nanobot.json --log-file /var/log/nanobot.log
EnvironmentFile=/etc/nanobot.env
WorkingDirectory=/opt/nanobot
""",
        encoding="utf-8",
    )
    duplicate = tmp_path / "duplicate.service"
    duplicate.write_text(
        """
[Service]
ExecStart=/opt/one/nanobot gateway --config /one --log-file /one
ExecStart=/opt/two/nanobot gateway --config /two --log-file /two
EnvironmentFile=/etc/nanobot.env
WorkingDirectory=/opt/nanobot
""",
        encoding="utf-8",
    )

    assert inspect_service_unit(reset).valid is True
    duplicate_report = inspect_service_unit(duplicate)
    assert duplicate_report.valid is False
    assert "duplicate ExecStart" in duplicate_report.forbidden


def test_service_inventory_rejects_wrapper_with_nanobot_arguments(tmp_path: Path) -> None:
    unit = tmp_path / "wrapper.service"
    unit.write_text(
        """
[Service]
ExecStart=/bin/echo nanobot gateway --config /etc/nanobot.json --log-file /var/log/nanobot.log
EnvironmentFile=/etc/nanobot.env
WorkingDirectory=/opt/nanobot
""",
        encoding="utf-8",
    )

    report = inspect_service_unit(unit)

    assert report.valid is False
    assert "nanobot gateway" in report.missing


def test_service_inventory_applies_environment_file_reset(tmp_path: Path) -> None:
    unit = tmp_path / "environment-reset.service"
    unit.write_text(
        """
[Service]
ExecStart=/opt/nanobot/bin/nanobot gateway --config /etc/nanobot.json --log-file /var/log/nanobot.log
EnvironmentFile=/etc/nanobot.env
EnvironmentFile=
WorkingDirectory=/opt/nanobot
""",
        encoding="utf-8",
    )

    report = inspect_service_unit(unit)

    assert report.valid is False
    assert "EnvironmentFile=" in report.missing


def test_state_rehearsal_is_idempotent_and_retains_new_writes() -> None:
    report = run_state_rehearsal()

    assert report["session"]["conversion_runs_equal"] is True
    assert report["session"]["rollback_retained_new_write"] is True
    assert report["session"]["canonical_copy_retained"] is True
    assert report["session"]["rollback_restored"] is True
    assert report["session"]["backup_unchanged"] is True
    assert report["cron"]["conversion_runs_equal"] is True
    assert report["cron"]["legacy_backup_matches_source"] is True
    assert report["cron"]["source_jobs_retained"] is True
    assert report["external_effects"] == ()


def test_state_rehearsal_blocks_and_reports_external_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    def attempt_network_effect() -> dict[str, object]:
        import socket

        socket.socket()
        return {}

    monkeypatch.setattr(
        "scripts.p6_release_rehearsal._run_state_rehearsal_body", attempt_network_effect
    )

    report = run_state_rehearsal()

    assert report["external_effects"] == ("network.socket",)
    with pytest.raises(ReleaseArtifactError, match="external_effects"):
        _validate_report({"state_rehearsal": report})


def _valid_state_report() -> dict[str, object]:
    return {
        "session": {
            "conversion_runs_equal": True,
            "rollback_retained_new_write": True,
            "canonical_copy_retained": True,
            "rollback_restored": True,
            "backup_unchanged": True,
        },
        "cron": {
            "conversion_runs_equal": True,
            "legacy_backup_matches_source": True,
            "source_jobs_retained": True,
        },
        "external_effects": [],
    }


@pytest.mark.parametrize(
    "mutate",
    (
        lambda report: report["session"].pop("backup_unchanged"),
        lambda report: report["cron"].update(conversion_runs_equal=False),
        lambda report: report.update(external_effects=["network.socket"]),
        lambda report: report.update(external_effects=""),
    ),
)
def test_cli_rejects_invalid_state_rehearsal_invariants(monkeypatch, capsys, mutate) -> None:
    report = _valid_state_report()
    mutate(report)
    monkeypatch.setattr("scripts.p6_release_rehearsal.run_state_rehearsal", lambda: report)

    assert main(["--state-rehearsal", "--json"]) == 2
    assert "state_rehearsal" in capsys.readouterr().err
