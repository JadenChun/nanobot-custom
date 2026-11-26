"""Seatbelt supervisor backend against a fake launch domain (A3-2M, WP4).

No process is started or signalled here, so these run on any POSIX host.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kohaku.agent.tools import native_process_supervisor_darwin as darwin_supervisor
from kohaku.agent.tools.native_process_supervisor import NativeLaunchConfig
from kohaku.agent.tools.seatbelt_tracking import NotQuiescent, Quiescent
from kohaku.security import darwin_proc

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX file modes")

_LAUNCH_ID = "0123456789abcdef" * 2


class _Domain:
    def __init__(self) -> None:
        self.polls = 0
        self.proven = False
        self.steps: list[float] = []
        self.results: list[Quiescent | NotQuiescent | None] = []
        self.root_exited: Callable[[], bool] = lambda: False
        self.protected: tuple[int, ...] = ()

    def profile_markers(self) -> str:
        return darwin_proc.marker_rules(_LAUNCH_ID)

    def poll(self) -> bool:
        self.polls += 1
        return self.proven

    def kill_step(self, deadline: float) -> Quiescent | NotQuiescent | None:
        self.steps.append(deadline)
        return self.results.pop(0) if self.results else None


def _backend(domain: _Domain, clock: Callable[[], float] = lambda: 100.0):
    def factory(
        launch_id: str, *, root_exited: Callable[[], bool], protected: Iterable[int],
    ) -> _Domain:
        assert launch_id == _LAUNCH_ID
        domain.root_exited = root_exited
        domain.protected = tuple(protected)
        return domain

    return darwin_supervisor.SeatbeltBackend(domain_factory=factory, clock=clock)


def _payload(tmp_path: Path, *, login: bool = False) -> NativeLaunchConfig:
    return {
        "command": "echo hello", "cwd": str(tmp_path),
        "env": {"HOME": str(tmp_path), "KOHAKU_SANDBOX": "caller"},
        "shell_program": "/bin/bash", "login": login,
        "launch_id": _LAUNCH_ID, "receipt_dir": str(tmp_path / ".ledger-receipts"),
    }


def _profile(tmp_path: Path) -> Path:
    return tmp_path / ".ledger-receipts-profiles" / f"{_LAUNCH_ID}.sb"


def _child(returncode: int | None = None) -> Any:
    return SimpleNamespace(pid=4321, returncode=returncode)


def test_prepare_writes_an_owner_only_profile_with_the_markers_last(tmp_path: Path) -> None:
    domain = _Domain()
    backend = _backend(domain)

    backend.prepare(_payload(tmp_path))

    profile = _profile(tmp_path)
    text = profile.read_text(encoding="utf-8")
    assert stat.S_IMODE(profile.stat().st_mode) == 0o600
    assert stat.S_IMODE(profile.parent.stat().st_mode) == 0o700
    assert text.index("(allow default)") < text.index(darwin_proc.member_name(_LAUNCH_ID))
    assert text.endswith(domain.profile_markers())
    assert domain.protected == (os.getppid(),)
    backend.close()
    assert not profile.exists()


def test_profile_is_never_written_over_an_existing_path(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    profile.parent.mkdir()
    profile.symlink_to(tmp_path / "elsewhere")

    with pytest.raises(FileExistsError):
        _backend(_Domain()).prepare(_payload(tmp_path))

    assert not (tmp_path / "elsewhere").exists()


def test_profile_directory_must_not_be_a_symlink(tmp_path: Path) -> None:
    (tmp_path / "target").mkdir()
    _profile(tmp_path).parent.symlink_to(tmp_path / "target", target_is_directory=True)

    with pytest.raises(RuntimeError, match="not private"):
        _backend(_Domain()).prepare(_payload(tmp_path))


def test_root_runs_under_sandbox_exec_with_the_marker_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: dict[str, Any] = {}
    child = _child()

    def spawn(args: list[str], *, cwd: str, env: dict[str, str]) -> Any:
        launched.update(args=args, cwd=cwd, env=env)
        return child

    monkeypatch.setattr(darwin_supervisor, "spawn_command_root", spawn)
    domain = _Domain()
    backend = _backend(domain)
    payload = _payload(tmp_path, login=True)
    backend.prepare(payload)

    assert domain.root_exited() is False
    assert backend.spawn_root(payload) is child

    assert launched["args"] == [
        "/usr/bin/sandbox-exec", "-f", str(_profile(tmp_path)), "--",
        "/bin/bash", "-l", "-c", "echo hello",
    ]
    assert launched["env"]["KOHAKU_SANDBOX"] == "seatbelt"
    assert launched["env"]["HOME"] == str(tmp_path)
    assert payload["env"]["KOHAKU_SANDBOX"] == "caller"
    assert domain.root_exited() is False
    child.returncode = 0
    assert domain.root_exited() is True


def test_spawn_refuses_without_a_prepared_profile(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not prepared"):
        _backend(_Domain()).spawn_root(_payload(tmp_path))


def test_root_group_is_signalled_only_while_the_root_is_unreaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signalled: list[int] = []
    monkeypatch.setattr(
        darwin_supervisor.os, "killpg", lambda pid, _signal: signalled.append(pid),
    )
    backend = _backend(_Domain())

    backend.request_kill(_child(returncode=0))
    assert signalled == []
    backend.request_kill(_child())
    assert signalled == [4321]


def test_without_a_kill_request_the_tree_is_only_observed(tmp_path: Path) -> None:
    domain = _Domain()
    backend = _backend(domain)
    backend.prepare(_payload(tmp_path))

    assert backend.tree_quiescent(_child(0)) is False
    assert not _profile(tmp_path).exists()
    domain.proven = True
    assert backend.tree_quiescent(_child(0)) is True
    assert backend.tree_quiescent(_child(0)) is True
    assert domain.polls == 2
    assert domain.steps == []


def test_kill_proof_keeps_one_deadline_and_paces_its_passes(tmp_path: Path) -> None:
    now = [100.0]
    domain = _Domain()
    domain.results = [None, Quiescent()]
    backend = _backend(domain, lambda: now[0])
    backend.prepare(_payload(tmp_path))

    backend.kill_step(_child(), root_reported=False)
    backend.kill_step(_child(), root_reported=False)
    assert len(domain.steps) == 1
    assert backend.tree_quiescent(_child(0)) is False
    now[0] += 1.0
    backend.kill_step(_child(0), root_reported=True)

    deadline = 100.0 + darwin_supervisor.KILL_DEADLINE_SECONDS
    assert domain.steps == [deadline, deadline]
    assert backend.tree_quiescent(_child(0)) is True
    backend.kill_step(_child(0), root_reported=True)
    assert len(domain.steps) == 2
    assert domain.polls == 0


def test_an_unprovable_kill_gives_up_instead_of_claiming_quiescence(tmp_path: Path) -> None:
    domain = _Domain()
    domain.results = [NotQuiescent("foreign_uid_member", (77,))]
    backend = _backend(domain)
    backend.prepare(_payload(tmp_path))

    with pytest.raises(RuntimeError, match="foreign_uid_member; pids: 77"):
        backend.kill_step(_child(0), root_reported=True)

    assert backend.tree_quiescent(_child(0)) is False


@pytest.mark.parametrize("split", [1, 10, 30])
def test_nested_sandbox_failure_is_explained_once_across_chunks(split: int) -> None:
    backend = _backend(_Domain())
    message = b"sandbox-exec: " + darwin_supervisor.NESTED_SANDBOX_SIGNATURE + b"\n"
    cut = len(b"sandbox-exec: ") + split
    note = darwin_supervisor.NESTED_SANDBOX_NOTE

    first = backend.filter_output("stderr", message[:cut])
    second = backend.filter_output("stderr", message[cut:])

    assert first + second == message + note
    assert backend.filter_output("stderr", message) == message


def test_nested_sandbox_text_on_stdout_is_left_alone() -> None:
    backend = _backend(_Domain())
    chunk = darwin_supervisor.NESTED_SANDBOX_SIGNATURE

    assert backend.filter_output("stdout", chunk) == chunk
