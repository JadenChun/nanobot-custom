"""R01-F17 / C2: mobile toolchain identity is checked against the lock.

The cases use a disposable fixture toolchain tree. No Android SDK, Go, JDK or
network resource is needed, and the real host toolchain is not read.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MOBILE_ENV = REPO / "mobile" / "scripts" / "mobile-env.sh"
LOCK = REPO / "mobile" / "toolchain.lock.json"


def _write_exec(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _make_fixture(tmp_path: Path) -> Path:
    mobile = tmp_path / "mobile"
    scripts = mobile / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(MOBILE_ENV, scripts / "mobile-env.sh")
    shutil.copy2(LOCK, mobile / "toolchain.lock.json")

    toolchain = mobile / "toolchain"
    _write_exec(toolchain / "jdk21" / "bin" / "javac", 'printf "javac 21.0.13\\n"\n')
    _write_exec(
        toolchain / "go" / "bin" / "go",
        'printf "go version go1.27.3 linux/amd64\\n"\n',
    )
    _write_exec(toolchain / "uv" / "uv", 'printf "uv 0.13.6\\n"\n')
    _write_exec(
        toolchain / "android-sdk" / "platform-tools" / "adb",
        """
if [ "$1" = "version" ]; then
  printf 'Android Debug Bridge version 1.0.41\\nVersion 37.0.1\\n'
  exit 0
fi
exit 0
""",
    )
    _write_exec(
        toolchain / "android-sdk" / "emulator" / "emulator",
        'printf "Android emulator version 37.2.13.0 (build_id 123)\\n"\n',
    )
    _write_exec(
        toolchain / "android-sdk" / "cmdline-tools" / "latest" / "bin" / "sdkmanager",
        "exit 0\n",
    )
    (toolchain / "android-sdk" / "cmdline-tools" / "latest" / "source.properties").write_text(
        "Pkg.Revision=23.0\n", encoding="utf-8"
    )
    for directory in (
        "build-tools/36.0.0",
        "platforms/android-36",
        "ndk/29.0.14206865",
        "cmake/3.31.6",
        "system-images/android-36/google_apis/x86_64",
    ):
        (toolchain / "android-sdk" / directory).mkdir(parents=True)

    _write_exec(toolchain / "bun" / "bin" / "bun", 'printf "1.4.2\\n"\n')
    (tmp_path / "home").mkdir()
    return mobile


def _run(mobile: Path, *, path_prefix: str = "") -> subprocess.CompletedProcess[str]:
    path = (path_prefix + ":" if path_prefix else "") + "/usr/bin:/bin"
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1/scripts/mobile-env.sh"; mobile_require_toolchain',
            "_",
            str(mobile),
        ],
        env={**os.environ, "HOME": str(mobile.parent / "home"), "PATH": path},
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_missing_bun_names_pinned_version(tmp_path: Path) -> None:
    mobile = _make_fixture(tmp_path)
    (mobile / "toolchain" / "bun" / "bin" / "bun").unlink()

    result = _run(mobile)

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "bun" in output
    assert "1.4.2" in output


def test_path_bun_wrong_version_does_not_satisfy_the_pin(tmp_path: Path) -> None:
    mobile = _make_fixture(tmp_path)
    (mobile / "toolchain" / "bun" / "bin" / "bun").unlink()
    path_bin = tmp_path / "path-bin"
    _write_exec(path_bin / "bun", 'printf "1.3.9\\n"\n')

    result = _run(mobile, path_prefix=str(path_bin))

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "1.4.2" in output
    assert "1.3.9" in output


def test_matching_toolchain_reports_every_lock_entry(tmp_path: Path) -> None:
    mobile = _make_fixture(tmp_path)

    result = _run(mobile)

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    for name in ("bun", "jdk", "go", "uv", "android", "gradle"):
        assert name in output.lower()
    assert "1.4.2" in output
    assert "21.0.13" in output
    assert "1.27.3" in output
    assert "unavailable" in output.lower()  # no Gradle archive in the fixture
    expected_image_path = (
        mobile
        / "toolchain"
        / "android-sdk"
        / "system-images"
        / "android-36"
        / "google_apis"
        / "x86_64"
    )
    assert f"({expected_image_path})" in output


def test_wrong_jdk_version_is_refused(tmp_path: Path) -> None:
    mobile = _make_fixture(tmp_path)
    _write_exec(
        mobile / "toolchain" / "jdk21" / "bin" / "javac",
        'printf "javac 17.0.12\\n"\n',
    )

    result = _run(mobile)

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "jdk" in output.lower()
    assert "21.0.13" in output
    assert "17.0.12" in output


def test_missing_adb_after_sdk_directory_exists_is_refused(tmp_path: Path) -> None:
    mobile = _make_fixture(tmp_path)
    (mobile / "toolchain" / "android-sdk" / "platform-tools" / "adb").unlink()

    result = _run(mobile)

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "adb" in output.lower()


def test_lock_still_pins_the_linux_release_tuple() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))

    assert lock["host"] == "linux-x86_64"
    assert lock["app"]["bun"] == "1.4.2"
    assert lock["jdk"]["version"] == "21.0.13+9"
    assert lock["go"]["version"] == "1.27.3"
