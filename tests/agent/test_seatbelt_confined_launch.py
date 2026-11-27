"""Confined launches through the Darwin backend and the shared relay (A3-2M, WP5).

Domains and backends are fakes, so nothing is ever signalled. The relay tests
start commands that exit by themselves.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from kohaku.agent.tools import native_process
from kohaku.agent.tools import native_process_supervisor as supervisor
from kohaku.agent.tools import native_process_supervisor_darwin as darwin_supervisor
from kohaku.agent.tools.native_process_supervisor import NativeLaunchConfig
from kohaku.agent.tools.seatbelt_profile import SeatbeltConfinement
from kohaku.agent.tools.seatbelt_tracking import Quiescent
from tests.agent.test_native_process_supervisor_darwin import (
    _LAUNCH_ID,
    _backend,
    _child,
    _Domain,
    _payload,
    _profile,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX file modes")


def _confined(tmp_path: Path, *, die_with_parent: bool = True) -> NativeLaunchConfig:
    payload = _payload(tmp_path)
    payload["confinement"] = {
        "rules": "(version 1)\n(deny default)\n",
        "params": [("P0", "/work")],
        "argv": ["/bin/sh", "-c", "exit 0"],
        "die_with_parent": die_with_parent,
    }
    return payload


@pytest.fixture
def temp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "launch-tmp"
    root.mkdir()
    monkeypatch.setattr(darwin_supervisor, "LAUNCH_TMP_ROOT", root)
    monkeypatch.setattr(
        darwin_supervisor, "canonical_path", lambda path: path.resolve(strict=False),
    )
    return root


def _read(payload: object) -> NativeLaunchConfig:
    read, write = os.pipe()
    try:
        os.write(write, json.dumps(payload).encode())
        os.close(write)
        return supervisor._read_exact_json(read)
    finally:
        os.close(read)


def test_launch_data_accepts_an_exact_confinement(tmp_path: Path) -> None:
    payload = _confined(tmp_path)

    assert _read(payload) == payload
    assert "confinement" not in _read(_payload(tmp_path))


@pytest.mark.parametrize(
    "change",
    [
        {"rules": 1}, {"params": [["P0"]]}, {"params": [["A=B", "/x"]]},
        {"params": [["", "/x"]]}, {"params": "P0"}, {"argv": []}, {"argv": [1]},
        {"die_with_parent": 1}, {"extra": True},
    ],
)
def test_launch_data_rejects_a_malformed_confinement(
    tmp_path: Path, change: dict[str, object],
) -> None:
    payload: dict[str, Any] = dict(_confined(tmp_path))
    payload["confinement"] = {**payload["confinement"], **change}

    with pytest.raises(ValueError, match="confinement is invalid"):
        _read(payload)


def test_the_linux_backend_refuses_a_confinement(tmp_path: Path) -> None:
    backend = supervisor._SubreaperBackend()

    with pytest.raises(ValueError, match="cannot apply a Seatbelt confinement"):
        backend.prepare(_confined(tmp_path))
    assert backend.kill_on_control_eof() is False


def test_confined_profile_hides_runtime_files_and_ends_with_the_markers(
    tmp_path: Path, temp_root: Path,
) -> None:
    domain = _Domain()
    backend = _backend(domain)

    backend.prepare(_confined(tmp_path))

    text = _profile(tmp_path).read_text(encoding="utf-8")
    assert "(allow default)" not in text
    assert text.index("(deny default)") < text.index('(param "R0")')
    assert text.endswith(domain.profile_markers())
    assert (temp_root / f"kh-{_LAUNCH_ID}").stat().st_mode & 0o777 == 0o700


def test_confined_root_receives_parameters_argv_and_private_caches(
    tmp_path: Path, temp_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: dict[str, Any] = {}

    def spawn(args: list[str], *, cwd: str, env: dict[str, str]) -> Any:
        launched.update(args=args, env=env)
        return _child()

    monkeypatch.setattr(darwin_supervisor, "spawn_command_root", spawn)
    backend = _backend(_Domain())
    payload = _confined(tmp_path)
    backend.prepare(payload)
    backend.spawn_root(payload)

    args = launched["args"]
    private = str(temp_root / f"kh-{_LAUNCH_ID}")
    defined = [args[index + 1] for index, value in enumerate(args) if value == "-D"]
    receipts = str((tmp_path / ".ledger-receipts").resolve())
    assert "P0=/work" in defined
    assert f"KOHAKU_LAUNCH_TMP={private}" in defined
    assert any(item.startswith("R") and item.endswith(f"={receipts}") for item in defined)
    assert args[args.index("--") + 1:] == ["/bin/sh", "-c", "exit 0"]
    assert launched["env"]["TMPDIR"] == private
    assert launched["env"]["KOHAKU_SANDBOX"] == "seatbelt"
    assert "TMPDIR" not in payload["env"]


def test_a_confined_launch_ends_with_its_root(tmp_path: Path, temp_root: Path) -> None:
    now = [100.0]
    domain = _Domain()
    domain.results = [None, Quiescent()]
    backend = _backend(domain, lambda: now[0])
    backend.prepare(_confined(tmp_path))

    # No stop was requested, yet leftovers are ended once the root has exited.
    assert backend.tree_quiescent(_child(0)) is False
    now[0] += 1.0
    assert backend.tree_quiescent(_child(0)) is True
    assert len(domain.steps) == 2
    assert domain.polls == 0


@pytest.mark.parametrize("die_with_parent", [True, False])
def test_only_unledgered_launches_die_with_their_parent(
    tmp_path: Path, temp_root: Path, die_with_parent: bool,
) -> None:
    backend = _backend(_Domain())
    backend.prepare(_confined(tmp_path, die_with_parent=die_with_parent))

    assert backend.kill_on_control_eof() is die_with_parent


def test_full_access_launches_do_not_die_with_their_parent(tmp_path: Path) -> None:
    backend = _backend(_Domain())
    backend.prepare(_payload(tmp_path))

    assert backend.kill_on_control_eof() is False


@pytest.mark.parametrize("proven", [True, False])
def test_the_temporary_folder_is_removed_only_after_proof(
    tmp_path: Path, temp_root: Path, monkeypatch: pytest.MonkeyPatch, proven: bool,
) -> None:
    monkeypatch.setattr(darwin_supervisor, "spawn_command_root", lambda *a, **k: _child())
    domain = _Domain()
    domain.results = [Quiescent() if proven else None]
    backend = _backend(domain)
    payload = _confined(tmp_path)
    backend.prepare(payload)
    backend.spawn_root(payload)
    backend.tree_quiescent(_child(0))

    backend.close()

    assert (temp_root / f"kh-{_LAUNCH_ID}").exists() is not proven


def test_an_unused_temporary_folder_is_removed(tmp_path: Path, temp_root: Path) -> None:
    backend = _backend(_Domain())
    backend.prepare(_confined(tmp_path))

    backend.close()

    assert list(temp_root.iterdir()) == []


def test_an_existing_temporary_folder_is_never_reused(tmp_path: Path, temp_root: Path) -> None:
    (temp_root / f"kh-{_LAUNCH_ID}").mkdir()

    with pytest.raises(FileExistsError):
        _backend(_Domain()).prepare(_confined(tmp_path))


def test_removal_handles_read_only_entries_without_following_links(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep", encoding="utf-8")
    outside.chmod(0o500)
    root = tmp_path / "private"
    locked = root / "cache" / "locked"
    locked.mkdir(parents=True)
    (locked / "module").write_text("x", encoding="utf-8")
    (root / "link-to-folder").symlink_to(outside, target_is_directory=True)
    os.link(outside / "keep", root / "hard-link")
    locked.chmod(0o500)
    locked.parent.chmod(0o000)
    try:
        darwin_supervisor.remove_launch_temp(root)

        assert not root.exists()
        assert (outside / "keep").read_text(encoding="utf-8") == "keep"
        assert outside.stat().st_mode & 0o777 == 0o500
    finally:
        outside.chmod(0o700)


def test_removal_refuses_a_link(tmp_path: Path) -> None:
    (tmp_path / "target").mkdir()
    (tmp_path / "target" / "file").write_text("x", encoding="utf-8")
    (tmp_path / "link").symlink_to(tmp_path / "target", target_is_directory=True)

    darwin_supervisor.remove_launch_temp(tmp_path / "link")

    assert (tmp_path / "target" / "file").exists()


# The real relay with a stand-in backend. A stop is only recorded: the command
# waits for that record and then exits by itself.
_BOOTSTRAP = """
import sys, time
from pathlib import Path
from kohaku.agent.tools import native_process_supervisor as supervisor


class Records:
    def prepare(self, payload):
        self.log = Path(payload["cwd"]) / "events"
        self.dies = payload["confinement"]["die_with_parent"]
    def record(self, event):
        with self.log.open("a") as stream:
            stream.write(event + "\\n")
    def spawn_root(self, payload):
        return supervisor.spawn_command_root(
            payload["confinement"]["argv"], cwd=payload["cwd"], env=payload["env"])
    def filter_output(self, name, chunk): return chunk
    def request_kill(self, child): self.record("kill")
    def kill_step(self, child, *, root_reported): pass
    def tree_quiescent(self, child): return True
    def kill_on_control_eof(self): return self.dies
    def close(self): self.record("closed")


sys.argv = [sys.argv[3]] + sys.argv[4:]
raise SystemExit(supervisor.run_supervisor(Records))
"""


@pytest.fixture
def stand_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Run the real relay with the recording backend; returns ephemeral folders."""
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
    created: list[Path] = []
    real = native_process.tempfile.mkdtemp

    def mkdtemp(prefix: str) -> str:
        path = real(prefix=prefix, dir=tmp_path)
        created.append(Path(path))
        return path

    monkeypatch.setattr(native_process, "tempfile", types.SimpleNamespace(mkdtemp=mkdtemp))
    return created


_ready = pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="requires durable POSIX process identities",
)


async def _spawn(tmp_path: Path, command: str) -> native_process.NativeProcessHandle:
    confinement = SeatbeltConfinement(
        rules="(version 1)", params=(), argv=("/bin/sh", "-c", command), cwd=str(tmp_path),
    )
    return await native_process.spawn_native_process(
        command=command, cwd=str(tmp_path), env=dict(os.environ),
        shell_program=None, login=False, confinement=confinement,
    )


@_ready
@pytest.mark.asyncio
async def test_an_unledgered_launch_keeps_its_proof_after_removing_its_folder(
    tmp_path: Path, stand_in: list[Path],
) -> None:
    handle = await _spawn(tmp_path, "echo done")

    stdout, _stderr = await asyncio.wait_for(handle.communicate(), timeout=15)

    assert stdout == b"done\n"
    assert await handle.wait_quiescent(timeout=10)
    assert handle.quiescence_proven()
    assert [path.exists() for path in stand_in] == [False]
    await handle.kill_process_tree()
    assert (tmp_path / "events").read_text() == "closed\n"


@_ready
@pytest.mark.asyncio
async def test_losing_the_owner_requests_a_stop_and_still_finishes(
    tmp_path: Path, stand_in: list[Path],
) -> None:
    events = tmp_path / "events"
    command = (
        "i=0; while [ ! -f events ] && [ $i -lt 200 ]; do sleep 0.05; i=$((i+1)); done; "
        "echo stopped > outcome"
    )
    handle = await _spawn(tmp_path, command)

    # What the supervisor sees when the owning process is gone.
    await handle.close_control()

    assert await asyncio.wait_for(handle.wait(), timeout=15) == 0
    assert await handle.wait_quiescent(timeout=10)
    assert events.read_text() == "kill\nclosed\n"


@_ready
@pytest.mark.asyncio
async def test_an_unconfined_launch_still_needs_a_reservation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs an admission reservation"):
        await native_process.spawn_native_process(
            command="true", cwd=str(tmp_path), env={}, shell_program=None, login=False,
        )
