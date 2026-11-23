"""A3-9 release gate regression tests.

These tests exercise the public WebUI build gates, the desktop parity
checker, the CI workflow contract, and the two-APK update helper.
"""

import json  # noqa: F401 - contract import surface
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import kohaku.webui.build as build_module  # noqa: F401 - contract import surface
from kohaku.webui.build import (
    WebUIBuildError,
    ensure_webui_bundle,
    inspect_webui_bundle,
    require_fresh_webui_bundle,
)


def _touch(path: Path, ns: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(path.name, encoding="utf-8")
    os.utime(path, ns=(ns, ns))


def test_stale_bundle_is_rejected_for_release(tmp_path: Path) -> None:
    source = tmp_path / "webui"
    dist = tmp_path / "kohaku/web/dist"
    _touch(source / "package.json", 10)
    _touch(source / "src/App.tsx", 30)
    _touch(dist / "index.html", 20)

    status = inspect_webui_bundle(source_dir=source, dist_dir=dist)

    with pytest.raises(WebUIBuildError, match="fresh WebUI bundle"):
        require_fresh_webui_bundle(status)


def test_missing_bundle_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "webui"
    dist = tmp_path / "kohaku/web/dist"
    _touch(source / "package.json", 10)

    with pytest.raises(WebUIBuildError, match="built WebUI bundle"):
        require_fresh_webui_bundle(
            inspect_webui_bundle(source_dir=source, dist_dir=dist)
        )


def test_fresh_bundle_is_accepted(tmp_path: Path) -> None:
    source = tmp_path / "webui"
    dist = tmp_path / "kohaku/web/dist"
    _touch(source / "package.json", 10)
    _touch(dist / "index.html", 20)

    status = inspect_webui_bundle(source_dir=source, dist_dir=dist)

    assert require_fresh_webui_bundle(status) is status


def test_release_mode_cannot_skip_stale_bundle(tmp_path: Path) -> None:
    source = tmp_path / "webui"
    dist = tmp_path / "kohaku/web/dist"
    _touch(source / "package.json", 10)
    _touch(source / "src/App.tsx", 30)
    _touch(dist / "index.html", 20)

    with pytest.raises(WebUIBuildError, match="skipping is disabled"):
        ensure_webui_bundle(
            mode="skip",
            source_dir=source,
            dist_dir=dist,
            environ={
                "KOHAKU_REQUIRE_FRESH_WEBUI": "1",
                "KOHAKU_SKIP_WEBUI_BUILD": "1",
            },
        )


def test_zero_exit_stale_build_fails_the_release_switch(tmp_path: Path) -> None:
    source = tmp_path / "webui"
    dist = tmp_path / "kohaku/web/dist"
    _touch(source / "package.json", 10)
    _touch(source / "src/App.tsx", 30)
    _touch(dist / "index.html", 20)

    def fake_run(command, *, cwd: Path, check: bool) -> None:
        # A build command that exits zero without refreshing the bundle.
        assert check is True

    with pytest.raises(WebUIBuildError, match="fresh WebUI bundle"):
        ensure_webui_bundle(
            mode="auto",
            source_dir=source,
            dist_dir=dist,
            runner="bun",
            environ={"KOHAKU_REQUIRE_FRESH_WEBUI": "1"},
            subprocess_run=fake_run,
        )


def test_fresh_build_passes_the_release_switch(tmp_path: Path) -> None:
    source = tmp_path / "webui"
    dist = tmp_path / "kohaku/web/dist"
    _touch(source / "package.json", 10)
    _touch(source / "src/App.tsx", 20)

    def fake_run(command, *, cwd: Path, check: bool) -> None:
        assert check is True
        if command == ["bun", "run", "build"]:
            _touch(dist / "index.html", 30)

    status = ensure_webui_bundle(
        mode="auto",
        source_dir=source,
        dist_dir=dist,
        runner="bun",
        environ={"KOHAKU_REQUIRE_FRESH_WEBUI": "1"},
        subprocess_run=fake_run,
    )

    assert status.reason == "fresh"


def _ci_fresh_webui_step() -> tuple[str, dict[str, str]]:
    repo = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (repo / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    for step in workflow["jobs"]["webui_bundle"]["steps"]:
        if step.get("name") == "Verify fresh WebUI bundle for release":
            env = {str(key): str(value) for key, value in (step.get("env") or {}).items()}
            return step["run"], env
    raise AssertionError("CI fresh WebUI verification step is missing")


def _run_ci_fresh_webui_step(
    tmp_path: Path, body: str, step_env: dict[str, str], variant: str
) -> subprocess.CompletedProcess[str]:
    repo = Path(__file__).resolve().parents[2]
    root = tmp_path / variant
    (root / "kohaku" / "webui").mkdir(parents=True)
    shutil.copy2(repo / "kohaku" / "webui" / "build.py", root / "kohaku" / "webui" / "build.py")
    _touch(root / "webui" / "package.json", 10)
    _touch(root / "webui" / "src" / "App.tsx", 30 if variant == "stale" else 20)
    if variant != "missing":
        _touch(root / "kohaku" / "web" / "dist" / "index.html", 20 if variant == "stale" else 40)

    bin_dir = tmp_path / f"{variant}-bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)
    env = {
        **os.environ,
        **step_env,
        "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
    }
    return subprocess.run(
        ["bash", "-c", body],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.skipif(os.name == "nt", reason="the workflow step is a POSIX shell heredoc")
def test_ci_fresh_webui_step_executes_for_fresh_stale_and_missing(tmp_path: Path) -> None:
    body, step_env = _ci_fresh_webui_step()
    assert step_env.get("KOHAKU_REQUIRE_FRESH_WEBUI") == "1"

    fresh = _run_ci_fresh_webui_step(tmp_path, body, step_env, "fresh")
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    assert "up to date" in fresh.stdout
    assert "AttributeError" not in fresh.stdout + fresh.stderr

    stale = _run_ci_fresh_webui_step(tmp_path, body, step_env, "stale")
    stale_output = stale.stdout + stale.stderr
    assert stale.returncode != 0, stale_output
    assert "fresh WebUI bundle" in stale_output
    assert "source is newer" in stale_output
    assert "AttributeError" not in stale_output

    missing = _run_ci_fresh_webui_step(tmp_path, body, step_env, "missing")
    missing_output = missing.stdout + missing.stderr
    assert missing.returncode != 0, missing_output
    assert "built WebUI bundle" in missing_output
    assert "missing" in missing_output
    assert "AttributeError" not in missing_output


def test_parity_checker_self_test_passes() -> None:
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "mobile/scripts/check-desktop-parity.py"),
            "--self-test",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "async route" in result.stdout
    assert "missing-async-route" not in result.stderr


def test_update_helper_rejects_identical_artifacts(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    script = repo / "mobile/scripts/test-update-persistence.sh"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    adb = bin_dir / "adb"
    adb.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "devices" ]; then\n'
        "  printf 'List of devices attached\\n'\n"
        '  printf "fake-1\\tdevice\\n"\n'
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    adb.chmod(0o755)

    apk = tmp_path / "same.apk"
    apk.write_bytes(b"same")

    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "ANDROID_SERIAL": "fake-1",
        "KOHAKU_MOBILE_UPDATE_SEED_APK": str(apk),
        "KOHAKU_MOBILE_UPDATE_CANDIDATE_APK": str(apk),
    }

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert "identical" in result.stdout + result.stderr
