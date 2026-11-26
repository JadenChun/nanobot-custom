"""A supervisor that cannot prove its tree publishes nothing (A3-2M, AC4.3).

A stand-in backend runs the real relay and ledger. Its command exits by itself
and nothing is signalled.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
import types
from pathlib import Path

import pytest

from kohaku.agent.tools import native_process
from kohaku.security.native_admission import NativeAdmissionLedger

pytestmark = pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"},
    reason="requires durable POSIX process identities",
)

_BOOTSTRAP = """
import sys
from kohaku.agent.tools import native_process_supervisor as supervisor


class GivesUp:
    def prepare(self, payload): pass
    def spawn_root(self, payload):
        return supervisor.spawn_command_root(
            supervisor.shell_argv(payload), cwd=payload["cwd"], env=payload["env"])
    def filter_output(self, name, chunk): return chunk
    def request_kill(self, child): pass
    def kill_step(self, child, *, root_reported):
        if root_reported:
            raise RuntimeError("command tree could not be proven quiescent")
    def tree_quiescent(self, child): return False
    def close(self): pass


sys.argv = [sys.argv[3]] + sys.argv[4:]
raise SystemExit(supervisor.run_supervisor(GivesUp))
"""


@pytest.mark.asyncio
async def test_giving_up_leaves_the_launch_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = tmp_path / "bootstrap.py"
    bootstrap.write_text(_BOOTSTRAP, encoding="utf-8")
    launcher = tmp_path / "helper"
    launcher.write_text(
        "#!/bin/sh\n"
        f'exec {shlex.quote(sys.executable)} -I -B {shlex.quote(str(bootstrap))} "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o700)
    monkeypatch.setattr(native_process, "native_supervisor_available", lambda: True)
    monkeypatch.setattr(
        native_process, "sys",
        types.SimpleNamespace(platform=sys.platform, executable=str(launcher)),
    )
    ledger = NativeAdmissionLedger(tmp_path / "native-admissions.json")
    reservation = ledger.begin(managed=lambda: False)
    assert reservation is not None

    handle = await native_process.spawn_native_process(
        command="echo done", cwd=str(tmp_path), env=dict(os.environ),
        shell_program="/bin/bash", login=False, reservation=reservation,
    )
    assert await asyncio.wait_for(handle.wait(), timeout=10) == 0
    assert handle.stdout is not None
    assert await asyncio.wait_for(handle.stdout.read(), timeout=10) == b"done\n"
    assert ledger.reconcile() is True

    with pytest.raises(RuntimeError, match="quiescence proof is missing"):
        await asyncio.wait_for(handle.kill_process_tree(), timeout=10)

    assert handle.quiescence_proven() is False
    assert ledger.reconcile() is True
    assert not list(ledger.receipt_dir.glob("*.receipt"))
    record = ledger._read()["reservations"]
    assert [item["state"] for item in record] == ["unknown"]
