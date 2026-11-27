"""The Darwin supervisor on a real host, without any signal (A3-2M, WP4).

Every command here exits by itself and nothing asks for a kill. A failing test
still ends its launch through the real signaller, so these follow the darwin_host
opt-in like the other real-host tests. Kill behaviour is in
test_seatbelt_supervisor_host.py.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from kohaku.agent.tools.native_process import NativeProcessHandle, spawn_native_process
from kohaku.agent.tools.native_process_supervisor_darwin import profile_directory
from kohaku.security.native_admission import NativeAdmissionLedger

pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS Seatbelt"),
    pytest.mark.darwin_host,
]


@pytest.fixture(autouse=True)
def _enabled(native_supervisor_host: None) -> Iterator[None]:
    yield


async def _spawn(
    tmp_path: Path, command: str,
) -> tuple[NativeProcessHandle, NativeAdmissionLedger]:
    ledger = NativeAdmissionLedger(tmp_path / "native-admissions.json")
    reservation = ledger.begin(managed=lambda: False)
    assert reservation is not None
    handle = await spawn_native_process(
        command=command, cwd=str(tmp_path), env=dict(os.environ),
        shell_program="/bin/bash", login=False, reservation=reservation,
    )
    return handle, ledger


@pytest.mark.asyncio
async def test_command_relays_streams_and_publishes_a_receipt(tmp_path: Path) -> None:
    handle, ledger = await _spawn(
        tmp_path, 'read line; echo "$line $KOHAKU_SANDBOX"; echo problem >&2; exit 3',
    )

    stdout, stderr = await asyncio.wait_for(handle.communicate(b"typed\n"), timeout=20)

    assert (stdout, stderr, handle.returncode) == (b"typed seatbelt\n", b"problem\n", 3)
    assert await handle.wait_quiescent(timeout=10)
    assert ledger.reconcile() is False
    assert list(profile_directory(ledger.receipt_dir).iterdir()) == []


@pytest.mark.asyncio
async def test_escaped_daemon_holds_back_the_receipt_until_it_exits(tmp_path: Path) -> None:
    daemon = (
        "import os, signal, time\n"
        "signal.alarm(20)\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"
        "    if os.fork() == 0:\n"
        "        null = os.open(os.devnull, os.O_RDWR)\n"
        "        for fd in (0, 1, 2): os.dup2(null, fd)\n"
        "        time.sleep(2)\n"
        "        open('daemon.done', 'w').close()\n"
        "    os._exit(0)\n"
    )
    handle, ledger = await _spawn(
        tmp_path, f"{shlex.quote(sys.executable)} -c {shlex.quote(daemon)}",
    )

    assert await asyncio.wait_for(handle.wait(), timeout=20) == 0
    assert await handle.wait_quiescent(timeout=0.5) is False
    assert ledger.reconcile() is True
    assert await handle.wait_quiescent(timeout=15)
    assert (tmp_path / "daemon.done").exists()
    assert ledger.reconcile() is False


@pytest.mark.asyncio
async def test_nested_sandbox_is_refused_with_an_explanation(tmp_path: Path) -> None:
    handle, _ledger = await _spawn(
        tmp_path,
        "/usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)' /usr/bin/true",
    )

    _stdout, stderr = await asyncio.wait_for(handle.communicate(), timeout=20)

    assert handle.returncode == 71
    assert b"sandbox_apply: Operation not permitted" in stderr
    assert b"[kohaku] This command tried to apply its own macOS sandbox" in stderr
    assert await handle.wait_quiescent(timeout=10)
