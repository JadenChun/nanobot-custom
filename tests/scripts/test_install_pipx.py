"""Q01-F03 / C1: the pipx installer path verifies the installed executable.

The cases use a disposable HOME, a fake ``pipx`` and a fake installed
executable. The live installer never runs against the user's environment.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO / "scripts" / "install.sh"
INSTALL_PS1 = REPO / "scripts" / "install.ps1"

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="the POSIX installer is exercised through /bin/sh"
)


def _system_python() -> str:
    candidate = Path(sys.base_prefix) / "bin" / "python3"
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("python3")
    if found:
        return found
    raise AssertionError("a system python3 is needed for the installer fixture")


def _write_exec(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _make_installer_fixture(tmp_path: Path, *, create_executable: bool) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    pipx_bin = tmp_path / "pipx-bin"
    home = tmp_path / "home"
    home.mkdir()
    bin_dir.mkdir()

    pipx_log = tmp_path / "pipx.log"
    installed_log = tmp_path / "installed.log"

    # A sentinel proves the temporary `pipx run` environment is never used.
    sentinel = tmp_path / "temporary-env" / "kohaku"
    _write_exec(
        sentinel,
        'printf "TEMPORARY-ENV-SENTINEL\\n"\nexit 99\n',
    )

    create_line = ""
    if create_executable:
        create_line = """
/bin/mkdir -p "$PIPX_BIN_DIR"
/bin/cat > "$PIPX_BIN_DIR/kohaku" <<'INSTALLED'
#!/bin/sh
printf 'installed-kohaku %s\\n' "$*" >> "$INSTALLED_LOG"
printf 'kohaku 9.9.9\\n'
INSTALLED
/bin/chmod +x "$PIPX_BIN_DIR/kohaku"
"""
    else:
        create_line = '/bin/rm -f "$PIPX_BIN_DIR/kohaku"\n'

    _write_exec(
        bin_dir / "pipx",
        f"""set -eu
printf '%s\\n' "$*" >> "$PIPX_LOG"
case "$1" in
  environment)
    if [ "$2" = "--value" ] && [ "$3" = "PIPX_BIN_DIR" ]; then
      printf '%s\\n' "$PIPX_BIN_DIR"
      exit 0
    fi
    ;;
  install)
{create_line}
    exit 0
    ;;
  run)
    printf 'TEMPORARY-ENV-SENTINEL\\n'
    exit 99
    ;;
esac
exit 0
""",
    )

    return {
        **os.environ,
        "HOME": str(home),
        "PATH": str(bin_dir),
        "PYTHON": _system_python(),
        "PIPX_HOME": str(tmp_path / "pipx-home"),
        "PIPX_BIN_DIR": str(pipx_bin),
        "PIPX_LOG": str(pipx_log),
        "INSTALLED_LOG": str(installed_log),
        "KOHAKU_SKIP_WIZARD": "1",
    }


def test_pipx_version_check_uses_installed_executable(tmp_path: Path) -> None:
    env = _make_installer_fixture(tmp_path, create_executable=True)

    result = _run_installer(env)

    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert "kohaku 9.9.9" in output
    assert str(Path(env["PIPX_BIN_DIR"]) / "kohaku") in output
    assert "TEMPORARY-ENV-SENTINEL" not in output

    pipx_log = Path(env["PIPX_LOG"]).read_text(encoding="utf-8")
    assert "install --python" in pipx_log
    assert " run " not in pipx_log
    installed_log = Path(env["INSTALLED_LOG"]).read_text(encoding="utf-8")
    assert "--version" in installed_log


def test_pipx_missing_installed_executable_is_an_explicit_failure(tmp_path: Path) -> None:
    env = _make_installer_fixture(tmp_path, create_executable=False)

    result = _run_installer(env)

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "not found" in output.lower()
    assert "kohaku" in output
    assert "TEMPORARY-ENV-SENTINEL" not in output
    assert not Path(env["INSTALLED_LOG"]).exists()


def test_pipx_dry_run_reports_installed_executable_not_temporary_environment(
    tmp_path: Path,
) -> None:
    env = _make_installer_fixture(tmp_path, create_executable=True)

    result = _run_installer(env, "--dry-run")

    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert "pipx run" not in output
    assert str(tmp_path / "pipx-bin" / "kohaku") in output


def test_powershell_installer_never_uses_pipx_run() -> None:
    script = INSTALL_PS1.read_text(encoding="utf-8")

    assert "pipx run --spec" not in script
    assert "PIPX_BIN_DIR" in script
    assert "Resolve-PipxKohaku" in script


def _run_installer(env: dict[str, str], *args: str):
    import subprocess

    return subprocess.run(
        [str(INSTALL_SH), *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
