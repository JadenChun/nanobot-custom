import sys
import time
from pathlib import Path

import pytest

from nanobot.apps.cli.service import CliAppManager

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="The native whole-tree supervisor is Linux-only.",
)


def _write_child(tmp_path: Path) -> None:
    (tmp_path / "child.py").write_text(
        "import pathlib\n"
        "import time\n"
        "\n"
        "time.sleep(3)\n"
        "pathlib.Path('late.marker').write_text('late', encoding='utf-8')\n",
        encoding="utf-8",
    )


def test_root_exit_settles_descendant(tmp_path):
    _write_child(tmp_path)

    root_code = (
        "import subprocess\n"
        "import sys\n"
        "\n"
        "print('root', flush=True)\n"
        "subprocess.Popen([sys.executable, 'child.py'])\n"
    )

    manager = CliAppManager(workspace=tmp_path, data_dir=tmp_path / "data")

    started = time.monotonic()
    result = manager._run_owned_argv(
        [sys.executable, "-c", root_code],
        timeout=10,
        cwd=tmp_path,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 2.5
    assert "root" in result.stdout

    marker = tmp_path / "late.marker"
    assert not marker.exists()

    time.sleep(3.5)
    assert not marker.exists()
