"""Real-signal tests of the launch domain (A3-2M, WP3, AC3.2).

These kill real processes, so they run only on a disposable macOS host (see
tests/darwin/gate.py). Every process they start also exits by itself, so a
broken engine cannot leave anything running. A second guard, independent of the
engine, refuses to signal any process older than the test.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kohaku.agent.tools import seatbelt_tracking as tracking
from kohaku.security import darwin_proc as darwin

pytestmark = pytest.mark.darwin_host

_LIFETIME = 30


class _GuardedSignaller:
    def __init__(self, floor_us: int) -> None:
        self.floor_us = floor_us
        self.killed: list[int] = []
        self.violations: list[int] = []

    def kill(self, pid: int) -> None:
        info = darwin.process_info(pid)
        if info is not None and info.start_us < self.floor_us:
            self.violations.append(pid)
            raise PermissionError(pid)
        tracking.HostSignaller().kill(pid)
        self.killed.append(pid)


_ESCAPEES = f"""
import os, signal, sys, time
signal.alarm({_LIFETIME})
ready = sys.argv[1]
def linger(name):
    signal.alarm({_LIFETIME})
    open(os.path.join(ready, name), "w").close()
    time.sleep({_LIFETIME})
    os._exit(0)
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        linger("daemon")
    os._exit(0)
if os.fork() == 0:
    os.setpgid(0, 0)
    linger("group")
if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    linger("stubborn")
"""

# At most 2 ** 6 processes, each bounded by its own alarm.
_FORK_STORM = f"""
import os, signal, sys, time
signal.alarm({_LIFETIME})
for _ in range(6):
    if os.fork() == 0:
        signal.alarm({_LIFETIME})
        os.setsid()
open(os.path.join(sys.argv[1], str(os.getpid())), "w").close()
time.sleep({_LIFETIME})
"""


@pytest.mark.parametrize(("script", "expected"), [(_ESCAPEES, 3), (_FORK_STORM, 8)])
def test_kill_until_quiescent_reaps_the_whole_tree(script, expected, tmp_path: Path) -> None:
    owner = darwin.process_info(os.getpid())
    assert owner is not None
    launch_id = darwin.new_launch_id()
    profile = tmp_path / "launch.sb"
    profile.write_text(
        "(version 1)\n(allow default)\n" + darwin.marker_rules(launch_id), encoding="ascii",
    )
    ready = tmp_path / "ready"
    ready.mkdir()
    signaller = _GuardedSignaller(owner.start_us)
    root = subprocess.Popen(
        [darwin.SANDBOX_EXEC, "-f", str(profile), sys.executable, "-c", script, str(ready)],
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    domain = tracking.LaunchDomain(
        launch_id, table=tracking.HostProcTable(), signaller=signaller, owner=owner,
        root_exited=lambda: root.poll() is not None,
    )
    deadline = time.monotonic() + 15
    while len(list(ready.iterdir())) < expected:
        assert time.monotonic() < deadline, "the launch never started its processes"
        time.sleep(0.02)

    result = domain.kill_until_quiescent(time.monotonic() + 15)

    assert isinstance(result, tracking.Quiescent), result
    assert signaller.violations == []
    assert signaller.killed
    assert darwin.scan_members(launch_id) == []
    root.wait(timeout=5)
