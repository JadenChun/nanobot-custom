import asyncio
import os
import shlex
import sys
from pathlib import Path

import pytest

from nanobot.agent.tools.exec_session import ExecSessionManager
from nanobot.security.native_admission import NativeAdmissionLedger

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="The native whole-tree supervisor is Linux-only.",
)


def _write_child(tmp_path: Path) -> None:
    (tmp_path / "child.py").write_text(
        "import pathlib\n"
        "import time\n"
        "\n"
        "time.sleep(1.5)\n"
        "pathlib.Path('late.marker').write_text('late', encoding='utf-8')\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_deadline_settles_descendant_without_poll(tmp_path):
    _write_child(tmp_path)

    root_code = (
        "import subprocess\n"
        "import sys\n"
        "\n"
        "subprocess.Popen([sys.executable, 'child.py'])\n"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(root_code)}"

    ledger = NativeAdmissionLedger(tmp_path / "native-admissions.json")
    reservation = ledger.begin(managed=lambda: False)
    assert reservation is not None

    manager = ExecSessionManager()
    try:
        session_id, _initial = await manager.start(
            command=command,
            cwd=str(tmp_path),
            env=dict(os.environ),
            timeout=1,
            shell_program="/bin/bash",
            login=False,
            yield_time_ms=0,
            max_output_chars=1000,
            native_admission=reservation,
        )
        assert session_id

        await asyncio.sleep(2.2)

        assert not (tmp_path / "late.marker").exists()

        poll = await manager.write(
            session_id=session_id,
            chars=None,
            close_stdin=False,
            terminate=False,
            yield_time_ms=0,
            max_output_chars=1000,
        )
        assert poll.done is True
    finally:
        await manager.close_all()
