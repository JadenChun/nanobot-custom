"""Native process streams remain live under backpressure."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import signal
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.tools import native_process
from nanobot.agent.tools.exec_session import (
    ExecSessionManager,
    _ExecSession,  # pyright: ignore[reportPrivateUsage]
)
from nanobot.agent.tools.native_process import NativeProcessHandle, spawn_native_process
from nanobot.agent.tools.shell import ExecTool
from nanobot.security import native_admission
from nanobot.security.desktop_permissions import (
    DesktopPermissionOwner,
    WorkspacePermissionConflictError,
    WorkspacePermissionUnavailableError,
)
from nanobot.security.native_admission import NativeAdmissionLedger

_SLOW_INPUT_BYTES = 256 * 1024
_DUPLEX_BYTES = 512 * 1024


def _python_command(source: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"


async def _spawn(tmp_path: Path, source: str) -> tuple[NativeProcessHandle, NativeAdmissionLedger]:
    ledger = NativeAdmissionLedger(tmp_path / "native-admissions.json")
    reservation = ledger.begin(managed=lambda: False)
    assert reservation is not None
    handle = await spawn_native_process(
        command=_python_command(source),
        cwd=str(tmp_path),
        env=dict(os.environ),
        shell_program="/bin/bash",
        login=False,
        reservation=reservation,
    )
    return handle, ledger


async def _finish_after_failure(handle: NativeProcessHandle) -> None:
    """Release test-owned children if an assertion times out on a broken facade."""
    if handle.returncode is not None and handle._supervisor.returncode is not None:
        return
    if handle._supervisor.returncode is None:
        try:
            await asyncio.wait_for(handle.kill_process_tree(), timeout=1)
        except asyncio.TimeoutError:
            pass
    readers = [
        asyncio.create_task(stream.read())
        for stream in (handle.stdout, handle.stderr)
        if stream is not None
    ]
    await asyncio.wait_for(asyncio.gather(*readers), timeout=8)
    await asyncio.wait_for(handle.wait(), timeout=8)
    await asyncio.wait_for(handle._supervisor.wait(), timeout=8)


async def _wait_for_quiescence(ledger: NativeAdmissionLedger) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        blocked = ledger.reconcile()
        state = json.loads(ledger.path.read_text(encoding="utf-8"))
        if not state["reservations"]:
            assert blocked is False
            assert state["outcomes"][-1]["result"] == "quiescent"
            return
        assert not any(item["state"] == "unknown" for item in state["reservations"])
        await asyncio.sleep(0.01)
    raise AssertionError("native supervisor did not publish quiescence proof")


async def _wait_for_marker(path: Path, *, timeout: float = 5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        await asyncio.sleep(0.01)
    return path.exists()


@pytest.mark.asyncio
async def test_handle_delivers_all_input_after_child_delays_reading(tmp_path: Path) -> None:
    source = (
        "import sys,time; time.sleep(0.2); "
        "data=sys.stdin.buffer.read(); "
        "sys.stdout.write(f'READ {len(data)}\\n')"
    )
    handle, ledger = await _spawn(tmp_path, source)
    try:
        stdout, stderr = await asyncio.wait_for(
            handle.communicate(b"x" * _SLOW_INPUT_BYTES),
            timeout=8,
        )
        assert stderr == b""
        assert stdout == f"READ {_SLOW_INPUT_BYTES}\n".encode()
        assert handle.returncode == 0
        assert await asyncio.wait_for(handle._supervisor.wait(), timeout=8) == 0
        assert ledger.reconcile() is False
    finally:
        if handle.returncode is None:
            await _finish_after_failure(handle)


@pytest.mark.asyncio
async def test_session_write_delivers_all_input_after_child_delays_reading(tmp_path: Path) -> None:
    source = (
        "import sys,time; time.sleep(0.2); "
        "data=sys.stdin.buffer.read(); "
        "sys.stdout.write(f'READ {len(data)}\\n')"
    )
    manager = ExecSessionManager()
    ledger = NativeAdmissionLedger(tmp_path / "native-admissions.json")
    reservation = ledger.begin(managed=lambda: False)
    assert reservation is not None
    session_id: str | None = None
    try:
        session_id, initial = await manager.start(
            command=_python_command(source),
            cwd=str(tmp_path),
            env=dict(os.environ),
            timeout=10,
            shell_program="/bin/bash",
            login=False,
            yield_time_ms=0,
            max_output_chars=1000,
            native_admission=reservation,
        )
        assert initial.done is False
        result = await asyncio.wait_for(
            manager.write(
                session_id=session_id,
                chars="x" * _SLOW_INPUT_BYTES,
                close_stdin=True,
                terminate=False,
                yield_time_ms=5000,
                max_output_chars=1000,
            ),
            timeout=8,
        )
        assert result.done is True
        assert result.exit_code == 0
        assert f"READ {_SLOW_INPUT_BYTES}" in result.output
        await _wait_for_quiescence(ledger)
    finally:
        await asyncio.wait_for(manager.close_all(), timeout=8)
        if session_id is not None:
            await _wait_for_quiescence(ledger)


@pytest.mark.asyncio
async def test_communicate_sends_and_receives_large_duplex_streams(tmp_path: Path) -> None:
    source = (
        "import os,sys; total=524288; block=4096; received=0; "
        "exec('for _ in range(total//block):\\n'"
        "' data=sys.stdin.buffer.read(block)\\n'"
        "' if not data: break\\n'"
        "' received+=len(data)\\n'"
        "' os.write(1,b\"o\"*len(data))\\n'"
        "' os.write(2,b\"e\"*len(data))'); "
        "sys.stdout.buffer.write(f'READ {received}\\n'.encode())"
    )
    handle, ledger = await _spawn(tmp_path, source)
    try:
        stdout, stderr = await asyncio.wait_for(
            handle.communicate(b"i" * _DUPLEX_BYTES),
            timeout=12,
        )
        assert stdout.startswith(b"o" * _DUPLEX_BYTES)
        assert stdout.endswith(f"READ {_DUPLEX_BYTES}\n".encode())
        assert len(stderr) == _DUPLEX_BYTES
        assert stderr == b"e" * _DUPLEX_BYTES
        assert handle.returncode == 0
        assert await asyncio.wait_for(handle._supervisor.wait(), timeout=8) == 0
        assert ledger.reconcile() is False
    finally:
        if handle.returncode is None:
            await _finish_after_failure(handle)


@pytest.mark.asyncio
async def test_kill_services_control_while_output_pipe_is_full(tmp_path: Path) -> None:
    source = (
        "import os,time,threading; "
        "threading.Timer(4, lambda: os._exit(0)).start(); "
        "chunk=b'x'*65536; "
        "exec('while True: os.write(1,chunk); os.write(2,chunk)')"
    )
    handle, ledger = await _spawn(tmp_path, source)
    try:
        await handle.wait_root_started()
        await asyncio.sleep(0.2)
        try:
            await asyncio.wait_for(handle.kill_process_tree(), timeout=2)
        except asyncio.TimeoutError:
            # Drain only after recording the failure so teardown can unblock the
            # pre-correction supervisor's blocking stdout relay.
            readers = [
                asyncio.create_task(stream.read())
                for stream in (handle.stdout, handle.stderr)
                if stream is not None
            ]
            await asyncio.wait_for(asyncio.gather(*readers), timeout=8)
            await asyncio.wait_for(handle.wait(), timeout=8)
            await asyncio.wait_for(handle._supervisor.wait(), timeout=8)
            assert False, "KILL was blocked behind a full output pipe"
        assert handle.returncode == -signal.SIGKILL
        assert await asyncio.wait_for(handle._supervisor.wait(), timeout=8) == 256 - signal.SIGKILL
        assert ledger.reconcile() is False
        if handle.stdout is not None:
            await asyncio.wait_for(handle.stdout.read(), timeout=2)
        if handle.stderr is not None:
            await asyncio.wait_for(handle.stderr.read(), timeout=2)
    finally:
        if handle.returncode is None:
            await _finish_after_failure(handle)


@pytest.mark.asyncio
async def test_once_retirement_closes_owned_interactive_stdin(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    source_binding = {
        "store_identity": "store-a",
        "incarnation": "incarnation-a",
        "generation": 0,
    }
    owner = DesktopPermissionOwner()
    reference = owner.activate_once(
        session_key="websocket:chat-a",
        source_binding=source_binding,
        task_turn_id="turn-a",
        task_input_event_id="event-a",
        path=root,
        file_access="read_only",
        commands=True,
    )
    owner_tag = owner.native_owner_tag(reference)
    assert owner_tag is not None
    ledger = NativeAdmissionLedger(tmp_path / "native-admissions.json")
    reservation = ledger.begin_owned(
        owner_tag,
        permission_owner=owner,
        permission_reference=reference,
    )
    started = tmp_path / "stdin-reader-started"
    source = (
        "from pathlib import Path; import sys; "
        f"Path({str(started)!r}).write_text('ready'); "
        "data=sys.stdin.buffer.read(); sys.stdout.write(f'CLOSED {len(data)}\\n')"
    )
    handle = await spawn_native_process(
        command=_python_command(source),
        cwd=str(root),
        env=dict(os.environ),
        shell_program="/bin/bash",
        login=False,
        reservation=reservation,
    )
    try:
        assert await _wait_for_marker(started)
        owner.retire_once(
            session_key="websocket:chat-a",
            source_binding=source_binding,
            task_turn_id="turn-a",
            task_input_event_id="event-a",
            reason="task_cancelled",
            active_turn=False,
            active_processes=0,
            unknown_processes=0,
            process_counts=lambda: ledger.owner_counts(owner_tag),
        )
        projection = owner.projection(
            session_key="websocket:chat-a",
            source_binding=source_binding,
            task_turn_id="turn-a",
            task_input_event_id="event-a",
        )
        assert projection["task_access"]["state"] == "retiring"
        assert handle.stdin is not None and handle.stdin.is_closing()
        stdout, stderr = await asyncio.wait_for(handle.communicate(), timeout=8)
        assert stderr == b""
        assert stdout == b"CLOSED 0\n"
        assert await asyncio.wait_for(handle._supervisor.wait(), timeout=8) == 0
        assert ledger.reconcile() is False
        owner.retire_once(
            session_key="websocket:chat-a",
            source_binding=source_binding,
            task_turn_id="turn-a",
            task_input_event_id="event-a",
            reason="task_cancelled",
            active_turn=False,
            active_processes=0,
            unknown_processes=0,
            process_counts=lambda: ledger.owner_counts(owner_tag),
        )
        assert owner.projection(
            session_key="websocket:chat-a",
            source_binding=source_binding,
            task_turn_id="turn-a",
            task_input_event_id="event-a",
        )["task_access"]["state"] == "expired"
    finally:
        if handle.returncode is None:
            await _finish_after_failure(handle)


@pytest.mark.parametrize("permission_kind", ["once", "full"])
@pytest.mark.asyncio
async def test_fenced_permission_blocks_interactive_input_from_home_cwd(
    tmp_path: Path,
    permission_kind: str,
) -> None:
    home = tmp_path / "agent"
    project = tmp_path / "private-project"
    home.mkdir()
    project.mkdir()
    source_binding = {
        "store_identity": "store-a",
        "incarnation": "incarnation-a",
        "generation": 0,
    }
    owner = DesktopPermissionOwner()
    session_key = "websocket:chat-input-fence"
    if permission_kind == "once":
        reference = owner.activate_once(
            session_key=session_key,
            source_binding=source_binding,
            task_turn_id="turn-input-fence",
            task_input_event_id="event-input-fence",
            path=project,
            file_access="read_only",
            commands=True,
        )
    else:
        reference = owner.enable_full(
            session_key=session_key,
            source_binding=source_binding,
            persistent=True,
            existing=True,
            local=True,
            idle=True,
            configured_sandbox=False,
            supervisor_available=True,
            stores_healthy=True,
        )
    owner_tag = owner.native_owner_tag(reference)
    assert owner_tag is not None
    ledger = NativeAdmissionLedger(tmp_path / "native-admissions.json")
    reservation = ledger.begin_owned(
        owner_tag,
        permission_owner=owner,
        permission_reference=reference,
    )
    started = tmp_path / "input-fence-started"
    source = (
        "from pathlib import Path; import sys; "
        f"Path({str(started)!r}).write_text('ready'); "
        "data=sys.stdin.buffer.read(); sys.stdout.write(f'CLOSED {len(data)}\\n')"
    )
    manager = ExecSessionManager()
    session_id: str | None = None
    try:
        session_id, initial = await manager.start(
            command=_python_command(source),
            cwd=str(home),
            env=dict(os.environ),
            timeout=10,
            shell_program="/bin/bash",
            login=False,
            yield_time_ms=0,
            max_output_chars=1000,
            native_admission=reservation,
        )
        assert initial.done is False
        assert await _wait_for_marker(started)
        session = manager._sessions[session_id]
        assert session.process.stdin is not None
        assert not session.process.stdin.is_closing()
        before_fence = await manager.write(
            session_id=session_id,
            chars="before",
            close_stdin=False,
            terminate=False,
            yield_time_ms=0,
            max_output_chars=1000,
        )
        assert before_fence.done is False

        def counts() -> tuple[int, int]:
            return ledger.owner_counts(owner_tag)

        if permission_kind == "once":
            owner.retire_once(
                session_key=session_key,
                source_binding=source_binding,
                task_turn_id="turn-input-fence",
                task_input_event_id="event-input-fence",
                reason="task_cancelled",
                active_turn=False,
                active_processes=0,
                unknown_processes=0,
                process_counts=counts,
            )
        else:
            with pytest.raises(WorkspacePermissionConflictError, match="native_processes_active"):
                owner.disable_full(
                    session_key=session_key,
                    source_binding=source_binding,
                    active_turn=False,
                    active_processes=0,
                    unknown_processes=0,
                    process_counts=counts,
                )
        assert session.process.stdin.is_closing()
        if permission_kind == "once":
            replacement = owner.activate_once(
                session_key=session_key,
                source_binding=source_binding,
                task_turn_id="replacement-turn",
                task_input_event_id="replacement-event",
                path=project,
                file_access="read_only",
                commands=True,
            )
            assert owner.is_current_task_reference(replacement)
        else:
            with pytest.raises(WorkspacePermissionUnavailableError, match="permission_retiring"):
                owner.enable_full(
                    session_key=session_key,
                    source_binding=source_binding,
                    persistent=True,
                    existing=True,
                    local=True,
                    idle=True,
                    configured_sandbox=False,
                    supervisor_available=True,
                    stores_healthy=True,
                    process_counts=counts,
                )
        with pytest.raises(RuntimeError, match="process input is blocked"):
            await manager.write(
                session_id=session_id,
                chars="must not reach fenced process",
                close_stdin=False,
                terminate=False,
                yield_time_ms=0,
                max_output_chars=1000,
            )

        result = await manager.write(
            session_id=session_id,
            chars=None,
            close_stdin=False,
            terminate=False,
            yield_time_ms=5000,
            max_output_chars=1000,
        )
        assert result.done is True
        assert result.output == "CLOSED 6\n"
        await _wait_for_quiescence(ledger)
        if permission_kind == "once":
            owner.retire_once(
                session_key=session_key,
                source_binding=source_binding,
                task_turn_id="turn-input-fence",
                task_input_event_id="event-input-fence",
                reason="task_cancelled",
                active_turn=False,
                active_processes=0,
                unknown_processes=0,
                process_counts=counts,
            )
        else:
            owner.disable_full(
                session_key=session_key,
                source_binding=source_binding,
                active_turn=False,
                active_processes=0,
                unknown_processes=0,
                process_counts=counts,
            )
    finally:
        await asyncio.wait_for(manager.close_all(), timeout=8)


@pytest.mark.asyncio
async def test_stdin_write_admitted_before_fence_finishes_before_retirement(tmp_path: Path) -> None:
    home = tmp_path / "agent"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    owner = DesktopPermissionOwner()
    source_binding = {
        "store_identity": "store-a",
        "incarnation": "incarnation-a",
        "generation": 0,
    }
    reference = owner.activate_once(
        session_key="websocket:chat-write-race",
        source_binding=source_binding,
        task_turn_id="turn-write-race",
        task_input_event_id="event-write-race",
        path=project,
        file_access="read_only",
        commands=True,
    )
    entered_write = threading.Event()
    release_write = threading.Event()
    retirement_finished = threading.Event()
    writes: list[bytes] = []
    write_results: list[str | None] = []
    thread_errors: list[BaseException] = []

    class BlockingStdin:
        def write(self, payload: bytes) -> None:
            writes.append(payload)
            entered_write.set()
            if not release_write.wait(timeout=3):
                raise TimeoutError("test write release was not signaled")

        async def drain(self) -> None:
            return None

    process: Any = types.SimpleNamespace(
        returncode=None,
        stdin=BlockingStdin(),
        stdout=None,
        stderr=None,
    )
    session = _ExecSession(
        session_id="write-race",
        process=process,
        command="blocked stdin writer",
        cwd=str(home),
        timeout=10,
        permission_owner=owner,
        permission_reference=reference,
    )

    def write_before_fence() -> None:
        try:
            write_results.append(asyncio.run(session.write("before")))
        except BaseException as exc:
            thread_errors.append(exc)

    def retire() -> None:
        try:
            owner.retire_once(
                session_key="websocket:chat-write-race",
                source_binding=source_binding,
                task_turn_id="turn-write-race",
                task_input_event_id="event-write-race",
                reason="task_cancelled",
                active_turn=False,
                active_processes=0,
                unknown_processes=0,
            )
            retirement_finished.set()
        except BaseException as exc:
            thread_errors.append(exc)

    writer = threading.Thread(target=write_before_fence)
    fence = threading.Thread(target=retire)
    writer.start()
    assert await asyncio.to_thread(entered_write.wait, 2)
    fence.start()
    assert not await asyncio.to_thread(retirement_finished.wait, 0.05)
    release_write.set()
    await asyncio.to_thread(writer.join, 2)
    await asyncio.to_thread(fence.join, 2)

    assert thread_errors == []
    assert write_results == [None]
    assert retirement_finished.is_set()
    assert writes == [b"before"]
    assert await session.write("after") == "workspace permission changed after this process started"
    assert writes == [b"before"]


@pytest.mark.asyncio
async def test_reserved_shell_launch_never_falls_back_without_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nanobot.agent.tools.shell as shell_module

    ledger = NativeAdmissionLedger(tmp_path / "admissions.json")
    reservation = ledger.begin(managed=lambda: False)
    assert reservation is not None
    spawned = False

    async def forbidden_spawn(*_args, **_kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("reserved work reached ordinary subprocess launch")

    monkeypatch.setattr(shell_module, "native_supervisor_available", lambda: False)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_spawn)
    with pytest.raises(RuntimeError, match="supervisor"):
        await ExecTool._spawn(
            "true", str(tmp_path), dict(os.environ), native_admission=reservation,
        )
    assert not spawned
    state = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert state["reservations"] == []
    assert state["outcomes"][-1]["result"] == "not_started"


@pytest.mark.asyncio
async def test_root_stdio_drains_while_another_receipt_publisher_holds_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform != "linux":
        pytest.skip("native subreaper receipt publication is Linux-only")

    output_size = 393_216
    ledger = NativeAdmissionLedger(tmp_path / "native-admissions.json")
    reservation_a = ledger.begin(managed=lambda: False)
    reservation_b = ledger.begin(managed=lambda: False)
    assert reservation_a is not None and reservation_b is not None
    monkeypatch.setattr(
        native_admission,
        "_RECEIPT_PUBLICATION_LOCK_TIMEOUT_SECONDS",
        0.05,
    )

    publisher_paused = tmp_path / "publisher-a-paused-in-fsync"
    release_publisher = tmp_path / "release-publisher-a"
    publisher_b_started = tmp_path / "publisher-b-started"
    publisher_b_controlled = tmp_path / "publisher-b-serviced-kill"

    publisher_a_bootstrap = tmp_path / "publisher-a-bootstrap.py"
    publisher_a_bootstrap.write_text(
        "import os,runpy,sys,time\n"
        "from pathlib import Path\n"
        "from nanobot.security import native_admission as admission\n"
        f"receipt_dir=Path({str(ledger.receipt_dir)!r})\n"
        f"paused=Path({str(publisher_paused)!r})\n"
        f"release=Path({str(release_publisher)!r})\n"
        "real_fsync=admission._fsync_directory\n"
        "def pause_receipt_fsync(path):\n"
        " if path==receipt_dir:\n"
        "  paused.write_text(str(os.getpid()),encoding='ascii')\n"
        "  deadline=time.monotonic()+20\n"
        "  while not release.exists():\n"
        "   if time.monotonic()>deadline: raise TimeoutError('test release missing')\n"
        "   time.sleep(.01)\n"
        " real_fsync(path)\n"
        "admission._fsync_directory=pause_receipt_fsync\n"
        "sys.argv=[sys.argv[3]]+sys.argv[4:]\n"
        "module=runpy.run_module('nanobot.agent.tools.native_process_supervisor',"
        "run_name='native_supervisor_test')\n"
        "raise SystemExit(module['main']())\n",
        encoding="utf-8",
    )
    publisher_b_bootstrap = tmp_path / "publisher-b-bootstrap.py"
    publisher_b_bootstrap.write_text(
        "import os,runpy,sys\n"
        "from pathlib import Path\n"
        f"started=Path({str(publisher_b_started)!r})\n"
        f"controlled=Path({str(publisher_b_controlled)!r})\n"
        "sys.argv=[sys.argv[3]]+sys.argv[4:]\n"
        "module=runpy.run_module('nanobot.agent.tools.native_process_supervisor',"
        "run_name='native_supervisor_test')\n"
        "real_publish=module['write_quiescence_receipt']\n"
        "def note_publish(receipt_dir,payload):\n"
        " started.write_text(str(os.getpid()),encoding='ascii')\n"
        " return real_publish(receipt_dir,payload)\n"
        "module['write_quiescence_receipt']=note_publish\n"
        "real_kill=module['_kill_adopted_children']\n"
        "def note_kill(root_pid):\n"
        " controlled.write_text(str(os.getpid()),encoding='ascii')\n"
        " return real_kill(root_pid)\n"
        "module['_kill_adopted_children']=note_kill\n"
        "raise SystemExit(module['main']())\n",
        encoding="utf-8",
    )

    def launcher(name: str, bootstrap: Path) -> Path:
        path = tmp_path / name
        path.write_text(
            "#!/bin/sh\n"
            f"exec {shlex.quote(sys.executable)} -I -B {shlex.quote(str(bootstrap))} \"$@\"\n",
            encoding="utf-8",
        )
        path.chmod(0o700)
        return path

    async def spawn_using_launcher(
        helper_launcher: Path,
        reservation,
        command: str,
    ) -> NativeProcessHandle:
        monkeypatch.setattr(
            native_process,
            "sys",
            types.SimpleNamespace(platform=sys.platform, executable=str(helper_launcher)),
        )
        try:
            return await spawn_native_process(
                command=command,
                cwd=str(tmp_path),
                env=dict(os.environ),
                shell_program="/bin/bash",
                login=False,
                reservation=reservation,
            )
        finally:
            monkeypatch.setattr(native_process, "sys", sys)

    publisher_a: NativeProcessHandle | None = None
    publisher_b: NativeProcessHandle | None = None
    communication: asyncio.Task[tuple[bytes, bytes]] | None = None
    communicated_while_publication_blocked = False
    control_serviced_while_publication_blocked = False
    root_result: int | None = None
    output: tuple[bytes, bytes] | None = None
    try:
        publisher_a = await spawn_using_launcher(
            launcher("publisher-a-helper", publisher_a_bootstrap),
            reservation_a,
            ":",
        )
        assert await asyncio.wait_for(publisher_a.wait(), timeout=3) == 0
        assert await _wait_for_marker(publisher_paused)

        source = (
            f"import os; os.write(1,b'o'*{output_size}); "
            f"os.write(2,b'e'*{output_size})"
        )
        publisher_b = await spawn_using_launcher(
            launcher("publisher-b-helper", publisher_b_bootstrap),
            reservation_b,
            _python_command(source),
        )
        root_result = await asyncio.wait_for(publisher_b.wait(), timeout=5)
        assert await _wait_for_marker(publisher_b_started)

        communication = asyncio.create_task(publisher_b.communicate())
        try:
            output = await asyncio.wait_for(
                asyncio.shield(communication),
                timeout=3,
            )
            communicated_while_publication_blocked = True
        except asyncio.TimeoutError:
            communicated_while_publication_blocked = False

        await asyncio.wait_for(publisher_b.kill_process_tree(), timeout=2)
        control_serviced_while_publication_blocked = await _wait_for_marker(
            publisher_b_controlled,
            timeout=2,
        )

        with pytest.raises(native_admission.NativeAdmissionActiveError):
            ledger.reconcile()
        b_record = next(
            item
            for item in json.loads(ledger.path.read_text(encoding="utf-8"))["reservations"]
            if item["launch_id"] == reservation_b.launch_id
        )
        assert b_record["state"] == "active"
        assert (ledger.receipt_dir / f"{reservation_b.launch_id}.receipt").exists() is False
        assert publisher_b._supervisor.returncode is None
    finally:
        release_publisher.touch()
        if publisher_a is not None:
            await asyncio.wait_for(publisher_a.communicate(), timeout=8)
            await asyncio.wait_for(publisher_a._supervisor.wait(), timeout=8)
        if publisher_b is not None:
            if communication is not None:
                output = await asyncio.wait_for(asyncio.shield(communication), timeout=8)
            else:
                await _finish_after_failure(publisher_b)
            await asyncio.wait_for(publisher_b._supervisor.wait(), timeout=8)

    assert root_result == 0
    assert communicated_while_publication_blocked is True, (
        "finite root stdout/stderr waited for another helper's receipt fsync"
    )
    assert control_serviced_while_publication_blocked is True, (
        "the supervisor control loop stopped during receipt publication"
    )
    assert output == (b"o" * output_size, b"e" * output_size)
    assert ledger.reconcile() is False
