"""Stopping a real macOS command tree through the supervisor (A3-2M, AC4.2).

These kill real processes, so they run only on a disposable macOS host. Every
process started here also exits by itself.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from kohaku.agent.tools.native_process import spawn_native_process
from kohaku.security import darwin_proc
from kohaku.security.native_admission import NativeAdmissionLedger

pytestmark = pytest.mark.darwin_host

_LIFETIME = 30

# A double-forked daemon and a SIGTERM-ignoring sibling, both outside the
# command's process group. The root then either exits or lingers.
_TREE = f"""
import os, signal, sys, time
signal.alarm({_LIFETIME})
def linger(name):
    signal.alarm({_LIFETIME})
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2): os.dup2(null, fd)
    with open(name + ".tmp", "w") as stream: stream.write(str(os.getpid()))
    os.replace(name + ".tmp", name)
    time.sleep({_LIFETIME})
    os._exit(0)
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        linger("daemon")
    os._exit(0)
if os.fork() == 0:
    os.setpgid(0, 0)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    linger("stubborn")
while not (os.path.exists("daemon") and os.path.exists("stubborn")):
    time.sleep(0.02)
if sys.argv[1] == "linger":
    time.sleep({_LIFETIME})
"""


@pytest.fixture(autouse=True)
def _enabled(native_supervisor_host: None) -> Iterator[None]:
    yield


def _alive(pid: int) -> bool:
    return darwin_proc.process_info(pid) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["exit", "linger"])
async def test_stop_kills_escaped_descendants_and_publishes_a_receipt(
    root: str, tmp_path: Path,
) -> None:
    ledger = NativeAdmissionLedger(tmp_path / "native-admissions.json")
    reservation = ledger.begin(managed=lambda: False)
    assert reservation is not None
    handle = await spawn_native_process(
        command=f"{shlex.quote(sys.executable)} -c {shlex.quote(_TREE)} {root}",
        cwd=str(tmp_path), env=dict(os.environ),
        shell_program="/bin/bash", login=False, reservation=reservation,
    )
    deadline = time.monotonic() + 15
    while not ((tmp_path / "daemon").exists() and (tmp_path / "stubborn").exists()):
        assert time.monotonic() < deadline, "the launch never started its processes"
        await asyncio.sleep(0.02)
    escapees = [int((tmp_path / name).read_text()) for name in ("daemon", "stubborn")]
    if root == "exit":
        assert await asyncio.wait_for(handle.wait(), timeout=10) == 0
    assert all(_alive(pid) for pid in escapees)
    assert ledger.reconcile() is True

    await asyncio.wait_for(handle.kill_process_tree(), timeout=15)

    assert await handle.wait_quiescent(timeout=15)
    assert not any(_alive(pid) for pid in escapees)
    assert ledger.reconcile() is False
