"""How the shell tool routes managed commands to macOS confinement (A3-2M, AC5.2).

Processes, preparation and the supervisor are fakes: nothing is launched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kohaku.agent.tools import exec_session, shell
from kohaku.agent.tools.native_process import NativeProcessHandle
from kohaku.agent.tools.seatbelt_profile import SeatbeltConfinement
from kohaku.agent.tools.shell import ExecTool
from kohaku.security.workspace_access import reset_workspace_scope
from tests.agent.test_managed_exec import _bind_managed_turn, _managed_tools

_HINT = "Sandbox may have blocked this operation"
_UNSETTLED = "could not confirm that every process"
_COMMAND = "cat marker.txt"


class _Process(NativeProcessHandle):
    """A finished command. Its pid can never name a real process."""

    def __init__(self, *, output: bytes, code: int, settled: bool) -> None:
        self._root_pid = 2**30
        self._returncode = code
        self._output = output
        self._settled = settled
        self.settle_waits: list[float] = []

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        return self._output, b""

    async def wait_quiescent(self, timeout: float) -> bool:
        self.settle_waits.append(timeout)
        return self._settled

    async def kill_process_tree(self) -> None:
        pytest.fail("a finished command must not be stopped")


class _Host:
    """One managed turn on a host whose confinement is chosen by the test."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
        self.home, self.project, _store, policy = _managed_tools(tmp_path)
        self.tool = ExecTool(
            working_dir=str(self.home), resource_policy=policy, restrict_to_workspace=True,
        )
        self.prepared: list[tuple[str, dict[str, Any]]] = []
        self.spawned: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.process = _Process(output=b"", code=0, settled=True)
        mac = platform == "darwin"
        monkeypatch.setattr(shell, "managed_seatbelt_available", lambda: mac)
        monkeypatch.setattr(shell, "managed_bwrap_available", lambda: not mac)
        monkeypatch.setattr(shell, "native_supervisor_available", lambda: True)
        monkeypatch.setattr(shell, "prepare_managed_seatbelt", self._seatbelt)
        monkeypatch.setattr(shell, "wrap_managed_command", self._bwrap)

        async def spawn(*args: Any, **kwargs: Any) -> _Process:
            self.spawned.append((args, kwargs))
            return self.process

        monkeypatch.setattr(ExecTool, "_spawn", staticmethod(spawn))

    def _seatbelt(self, command: str, **kwargs: Any) -> SeatbeltConfinement:
        self.prepared.append(("seatbelt", {"command": command, **kwargs}))
        return SeatbeltConfinement(
            "(version 1)", (), ("/bin/sh", "-c", command), str(kwargs["cwd"]),
        )

    def _bwrap(self, command: str, **kwargs: Any) -> str:
        self.prepared.append(("bwrap", {"command": command, **kwargs}))
        return f"bwrap -- {command}"

    async def run(self, **kwargs: Any) -> Any:
        token, request = _bind_managed_turn(self.home, self.project)
        try:
            with request:
                return await self.tool.execute(command=_COMMAND, **kwargs)
        finally:
            reset_workspace_scope(token)


@pytest.mark.asyncio
async def test_macos_runs_the_command_unchanged_under_a_confinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _Host(tmp_path, monkeypatch, "darwin")
    host.process = _Process(output=b"MARKER", code=0, settled=True)

    result = await host.run()

    assert "MARKER" in result and _HINT not in result and _UNSETTLED not in result
    assert [name for name, _ in host.prepared] == ["seatbelt"]
    (args, kwargs), = host.spawned
    command, cwd, _env, shell_program, login = args
    assert command == _COMMAND
    # The command keeps its real folder instead of starting in the agent home.
    assert cwd == str(host.project.resolve())
    assert shell_program is None and login is False
    assert kwargs["confinement"].argv == ("/bin/sh", "-c", _COMMAND)
    assert kwargs["native_admission"] is None
    assert host.process.settle_waits == [shell._CONFINED_SETTLE_SECONDS]


@pytest.mark.asyncio
async def test_linux_keeps_the_wrapped_command_and_spawns_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _Host(tmp_path, monkeypatch, "linux")
    host.process = _Process(output=b"cat: x: Operation not permitted", code=1, settled=False)

    result = await host.run()

    assert [name for name, _ in host.prepared] == ["bwrap"]
    (args, kwargs), = host.spawned
    assert args[0] == f"bwrap -- {_COMMAND}"
    assert args[1] == str(host.home.resolve())
    # No new keyword reaches the Linux launch, and none of the macOS notes appear.
    assert set(kwargs) == {"process_tree", "native_admission"}
    assert host.process.settle_waits == []
    assert _HINT not in result and _UNSETTLED not in result


def test_both_platforms_receive_the_same_confinement_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[set[str]] = []
    for platform in ("darwin", "linux"):
        (tmp_path / platform).mkdir()
        host = _Host(tmp_path / platform, monkeypatch, platform)
        token, request = _bind_managed_turn(host.home, host.project)
        try:
            with request:
                host.tool._prepare_command(_COMMAND, None, None, None, None)
        finally:
            reset_workspace_scope(token)
        received.append(set(host.prepared[0][1]))

    assert received[0] == received[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output", "code", "hinted"),
    [
        (b"sh: x: Operation not permitted", 1, True),
        (b"No such file or directory", 1, False),
        (b"Operation not permitted", 0, False),
    ],
)
async def test_a_denied_confined_command_explains_the_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: bytes, code: int, hinted: bool,
) -> None:
    host = _Host(tmp_path, monkeypatch, "darwin")
    host.process = _Process(output=output, code=code, settled=True)

    result = await host.run()

    assert (_HINT in result) is hinted
    assert getattr(result, "is_error", False) is hinted


@pytest.mark.asyncio
async def test_an_unconfirmed_exit_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _Host(tmp_path, monkeypatch, "darwin")
    host.process = _Process(output=b"done", code=0, settled=False)

    result = await host.run()

    assert _UNSETTLED in result


@pytest.mark.asyncio
async def test_a_session_receives_the_confinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _Host(tmp_path, monkeypatch, "darwin")
    started: list[dict[str, Any]] = []

    async def start(**kwargs: Any) -> Any:
        started.append(kwargs)
        raise RuntimeError("stop after recording")

    monkeypatch.setattr(host.tool._session_manager, "start", start)

    await host.run(yield_time_ms=0)

    assert started[0]["confinement"].argv == ("/bin/sh", "-c", _COMMAND)
    assert started[0]["command"] == _COMMAND
    assert host.spawned == []


@pytest.mark.asyncio
@pytest.mark.parametrize("confined", [True, False])
async def test_the_session_launch_forwards_only_a_real_confinement(
    monkeypatch: pytest.MonkeyPatch, confined: bool,
) -> None:
    calls: list[dict[str, Any]] = []

    async def spawn(*args: Any, **kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(ExecTool, "_spawn", staticmethod(spawn))
    confinement = SeatbeltConfinement("(version 1)", (), ("/bin/sh",), "/") if confined else None

    await exec_session.ExecSessionManager()._spawn(
        "true", "/", {}, None, False, confinement=confinement,
    )

    assert calls[0].get("confinement") is confinement
    assert ("confinement" in calls[0]) is confined


@pytest.mark.asyncio
async def test_a_confined_launch_goes_to_the_supervisor_without_a_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def spawn_native_process(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(shell, "native_supervisor_available", lambda: True)
    monkeypatch.setattr(shell, "spawn_native_process", spawn_native_process)
    confinement = SeatbeltConfinement("(version 1)", (), ("/bin/sh",), "/")

    await ExecTool._spawn("true", "/", {}, confinement=confinement)

    assert calls[0]["confinement"] is confinement
    assert calls[0]["reservation"] is None


@pytest.mark.asyncio
async def test_a_confinement_has_no_unsupervised_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shell, "native_supervisor_available", lambda: False)
    confinement = SeatbeltConfinement("(version 1)", (), ("/bin/sh", "-c", "touch ran"), "/")

    with pytest.raises(RuntimeError, match="supervisor is unavailable"):
        await ExecTool._spawn("touch ran", str(tmp_path), {}, confinement=confinement)

    assert not (tmp_path / "ran").exists()
