"""One select batch may hold events that an earlier event in it made stale.

The relay runs in a subprocess because it owns descriptors 0, 1 and 2. Signals
and the receipt are replaced there, so nothing is killed and nothing is stored.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import kohaku

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX descriptors")

_RELAY = """
import json, os, selectors, sys, time, types

sys.path.insert(0, sys.argv[1])
from kohaku.agent.tools import native_process_supervisor as sup

owner_action = sys.argv[2]
report_fd = os.dup(1)
signals, receipts, batches = [], [], []
os.kill = os.killpg = lambda pid, number: signals.append(number)
sup.current_host_identity = lambda: types.SimpleNamespace(
    trusted=True, host_id="host", boot_id="boot", pid_namespace="namespace",
)
sup.process_start_time = lambda pid: 1
sup.write_quiescence_receipt = lambda directory, receipt: receipts.append(receipt["result"])

gate_r, gate_w = os.pipe()
status_r, status_w = os.pipe()
control_r, control_w = os.pipe()
in_r, in_w = os.pipe()
out_r, out_w = os.pipe()
err_r, err_w = os.pipe()
os.dup2(in_r, 0)
os.dup2(out_w, 1)
os.dup2(err_w, 2)
os.close(out_w)
os.close(err_w)
sys.stdin = os.fdopen(0, "rb", closefd=False)


class KillWithPendingWrite(selectors.DefaultSelector):
    # Once, as soon as an output write is pending: the owner asks for a kill and
    # stops reading, and both events are delivered together, the kill first.
    def select(self, timeout=None):
        kinds = {key.data[0] for key in self.get_map().values()}
        if batches or "supervisor_output" not in kinds:
            return super().select(timeout)
        if owner_action == "kill":
            os.write(control_w, b"KILL")
        os.close(control_w)
        os.close(out_r)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            events = super().select(0.02)
            ready = {key.data[0] for key, _mask in events}
            if {"control", "supervisor_output"} <= ready:
                events.sort(key=lambda event: event[0].data[0] != "control")
                batches.append([key.data[0] for key, _mask in events])
                return events
        raise AssertionError("the two events never arrived together")


sup.selectors.DefaultSelector = KillWithPendingWrite


class Backend:
    kill_requested = False

    def prepare(self, payload):
        pass

    def spawn_root(self, payload):
        # Exits by itself once its output has been read.
        self.child = sup.spawn_command_root(
            ["/bin/sh", "-c", "head -c 4000000 /dev/zero; sleep 0.2"], cwd="/", env={},
        )
        return self.child

    def filter_output(self, name, chunk):
        return chunk

    def request_kill(self, child):
        self.kill_requested = True

    def kill_step(self, child, *, root_reported):
        pass

    def tree_quiescent(self, child):
        return self.kill_requested

    def kill_on_control_eof(self):
        return True

    def close(self):
        pass


os.write(gate_w, b"GO" + bytes([10]))
backend = Backend()
returncode = sup._supervise(
    {"launch_id": "0" * 32, "receipt_dir": "unused"}, backend,
    gate_fd=gate_r, status_fd=status_w, control_fd=control_r,
)
status = os.read(status_r, 65536).decode()
backend.child.stdout.close()
backend.child.stderr.close()
backend.child.wait(timeout=10)
os.write(report_fd, json.dumps({
    "returncode": returncode, "status": status, "batches": batches,
    "signals": signals, "receipts": receipts,
}).encode())
"""


@pytest.mark.parametrize("owner_action", ["kill", "eof"])
def test_a_kill_batched_with_a_pending_output_write_ends_normally(
    tmp_path: Path, owner_action: str,
) -> None:
    script = tmp_path / "relay.py"
    script.write_text(_RELAY, encoding="utf-8")
    package_root = Path(kohaku.__file__).resolve().parent.parent

    completed = subprocess.run(
        [sys.executable, "-I", str(script), str(package_root), owner_action],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=40,
        close_fds=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["batches"] and result["batches"][0][0] == "control"
    assert "supervisor_output" in result["batches"][0]
    assert "is not registered" not in result["status"]
    statuses = [json.loads(line)["type"] for line in result["status"].splitlines()]
    assert statuses == ["ready", "root", "root_exit"]
    assert result["returncode"] == 0
    assert result["receipts"] == ["quiescent"]
    assert result["signals"] == []
