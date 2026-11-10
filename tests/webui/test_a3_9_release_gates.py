"""A3-9 release gate regression tests.

These tests exercise the public WebUI build gates, the desktop parity
checker, the CI workflow contract, and the two-APK update helper.
"""

import json  # noqa: F401 - contract import surface
import os
import subprocess
import sys
from pathlib import Path

import pytest

import nanobot.webui.build as build_module  # noqa: F401 - contract import surface
from nanobot.webui.build import (
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
    dist = tmp_path / "nanobot/web/dist"
    _touch(source / "package.json", 10)
    _touch(source / "src/App.tsx", 30)
    _touch(dist / "index.html", 20)

    status = inspect_webui_bundle(source_dir=source, dist_dir=dist)

    with pytest.raises(WebUIBuildError, match="fresh WebUI bundle"):
        require_fresh_webui_bundle(status)


def test_missing_bundle_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "webui"
    dist = tmp_path / "nanobot/web/dist"
    _touch(source / "package.json", 10)

    with pytest.raises(WebUIBuildError, match="built WebUI bundle"):
        require_fresh_webui_bundle(
            inspect_webui_bundle(source_dir=source, dist_dir=dist)
        )


def test_fresh_bundle_is_accepted(tmp_path: Path) -> None:
    source = tmp_path / "webui"
    dist = tmp_path / "nanobot/web/dist"
    _touch(source / "package.json", 10)
    _touch(dist / "index.html", 20)

    status = inspect_webui_bundle(source_dir=source, dist_dir=dist)

    assert require_fresh_webui_bundle(status) is status


def test_release_mode_cannot_skip_stale_bundle(tmp_path: Path) -> None:
    source = tmp_path / "webui"
    dist = tmp_path / "nanobot/web/dist"
    _touch(source / "package.json", 10)
    _touch(source / "src/App.tsx", 30)
    _touch(dist / "index.html", 20)

    with pytest.raises(WebUIBuildError, match="skipping is disabled"):
        ensure_webui_bundle(
            mode="skip",
            source_dir=source,
            dist_dir=dist,
            environ={
                "NANOBOT_REQUIRE_FRESH_WEBUI": "1",
                "NANOBOT_SKIP_WEBUI_BUILD": "1",
            },
        )


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
        "NANOBOT_UPDATE_SEED_APK": str(apk),
        "NANOBOT_UPDATE_CANDIDATE_APK": str(apk),
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
