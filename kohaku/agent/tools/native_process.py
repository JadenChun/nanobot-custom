"""Root-process facade backed by a whole-tree supervisor for this platform."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import sys
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, cast

from kohaku.agent.tools.executor_capabilities import (
    tree_ownership,
    tree_ownership_available,
)
from kohaku.agent.tools.seatbelt_profile import SeatbeltConfinement
from kohaku.security.native_admission import (
    NativeProcessReservation,
    process_start_time,
)


class NativeProcessHandle:
    """Expose command-root wait and streams without waiting for tree admission."""

    def __init__(
        self,
        supervisor: asyncio.subprocess.Process,
        status_reader: asyncio.StreamReader,
        control_fd: int,
        on_supervisor_exit: Callable[[], None] | None = None,
    ) -> None:
        self._supervisor = supervisor
        self._status_reader = status_reader
        self._control_fd = control_fd
        self._on_supervisor_exit = on_supervisor_exit
        self._root_pid: int | None = None
        self._returncode: int | None = None
        loop = asyncio.get_running_loop()
        self._ready: asyncio.Future[None] = loop.create_future()
        self._root_started: asyncio.Future[int] = loop.create_future()
        self._root_done = asyncio.Event()
        self._receipt_path: Path | None = None
        # Set for a launch with no admission ledger: its private receipt folder,
        # and the proof read from it just before the folder is removed.
        self._ephemeral_directory: Path | None = None
        self._ephemeral_proof = False
        self._status_task = asyncio.create_task(self._read_status())
        self._reap_task = asyncio.create_task(self._reap_supervisor())
        self.stdin = supervisor.stdin
        self.stdout = supervisor.stdout
        self.stderr = supervisor.stderr

    @property
    def pid(self) -> int:
        if self._root_pid is None:
            raise RuntimeError("native command root has not started")
        return self._root_pid

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def set_on_supervisor_exit(self, callback: Callable[[], None] | None) -> None:
        """Replace the internal owner cleanup hook after handle registration."""
        self._on_supervisor_exit = callback

    def set_receipt_path(self, path: Path, *, ephemeral_directory: Path | None = None) -> None:
        """Bind the supervisor's quiescence receipt for this launch.

        An ephemeral directory holds a receipt nothing else will read; it is
        removed when the supervisor exits.
        """
        self._receipt_path = path
        self._ephemeral_directory = ephemeral_directory

    def _receipt_published(self) -> bool:
        if self._ephemeral_directory is not None:
            return self._ephemeral_proof
        return self._receipt_path is not None and self._receipt_path.is_file()

    def quiescence_proven(self) -> bool:
        """Return true only after the supervisor published its tree proof."""
        task = self._reap_task
        if not task.done() or task.cancelled() or task.exception() is not None:
            return False
        return self._receipt_published()

    async def wait_quiescent(self, timeout: float | None = None) -> bool:
        """Wait for supervisor exit and the published tree-quiescence receipt."""
        task = self._reap_task
        if not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except asyncio.TimeoutError:
                return False
        if task.cancelled() or task.exception() is not None:
            return False
        return self._receipt_published()

    async def _read_status(self) -> None:
        try:
            while line := await self._status_reader.readline():
                status = _parse_status(line)
                if status.kind == "ready":
                    if not self._ready.done():
                        self._ready.set_result(None)
                elif status.kind == "root":
                    if status.pid is None or status.pid <= 1:
                        raise ValueError("native supervisor sent an invalid root pid")
                    self._root_pid = status.pid
                    if not self._root_started.done():
                        self._root_started.set_result(status.pid)
                elif status.kind == "root_exit":
                    if status.returncode is None:
                        raise ValueError("native supervisor sent an invalid command result")
                    self._returncode = status.returncode
                    self._root_done.set()
                elif status.kind == "error":
                    error = RuntimeError(
                        status.message or "native command supervisor failed"
                    )
                    if not self._ready.done():
                        self._ready.set_exception(error)
                    if not self._root_started.done():
                        self._root_started.set_exception(error)
                    if not self._root_done.is_set():
                        self._returncode = -1
                        self._root_done.set()
                    return
                else:
                    raise ValueError("native supervisor sent an unknown status record")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            error = RuntimeError("native supervisor status is unavailable")
            error.__cause__ = exc
            if not self._ready.done():
                self._ready.set_exception(error)
            if not self._root_started.done():
                self._root_started.set_exception(error)
        finally:
            if not self._root_done.is_set():
                self._returncode = -1
                self._root_done.set()
            for future in (self._ready, self._root_started):
                if not future.done():
                    future.set_exception(RuntimeError("native supervisor exited before command start"))

    async def wait(self) -> int:
        await self._root_done.wait()
        return self._returncode if self._returncode is not None else -1

    async def wait_ready(self) -> None:
        await self._ready

    async def wait_root_started(self) -> int:
        return await self._root_started

    async def _reap_supervisor(self) -> None:
        try:
            await self._supervisor.wait()
            if self._ephemeral_directory is not None:
                self._ephemeral_proof = (
                    self._receipt_path is not None and self._receipt_path.is_file()
                )
                shutil.rmtree(self._ephemeral_directory, ignore_errors=True)
        finally:
            await self.close_control()
            if self._on_supervisor_exit is not None:
                self._on_supervisor_exit()
                self._on_supervisor_exit = None

    def close_stdin_now(self) -> bool:
        """Close interactive input synchronously when its permission is fenced."""
        if self.stdin is None or self.stdin.is_closing():
            return True
        try:
            self.stdin.close()
        except Exception:
            return False
        return self.stdin.is_closing()

    async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
        stdout_task = asyncio.create_task(self.stdout.read() if self.stdout is not None else _empty_bytes())
        stderr_task = asyncio.create_task(self.stderr.read() if self.stderr is not None else _empty_bytes())
        stdin_task = asyncio.create_task(self._write_stdin(input))
        try:
            await self.wait()
            await stdin_task
            stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
            return stdout, stderr
        finally:
            tasks = (stdin_task, stdout_task, stderr_task)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _write_stdin(self, input: bytes | None) -> None:
        if self.stdin is None:
            return
        if input is not None:
            self.stdin.write(input)
            with suppress(BrokenPipeError, ConnectionResetError):
                await self.stdin.drain()
        if not self.stdin.is_closing():
            self.stdin.close()
            with suppress(BrokenPipeError, ConnectionResetError):
                await self.stdin.wait_closed()

    async def kill_process_tree(self) -> None:
        try:
            os.write(self._control_fd, b"KILL\n")
        except OSError:
            with suppress(ProcessLookupError):
                self._supervisor.kill()
        await self.wait()
        # Retire the supervisor before returning so callers can observe its
        # exit without racing asyncio's process-exit wakeup.
        reap_task = self._reap_task
        if not reap_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(reap_task), timeout=1.0)
            except asyncio.TimeoutError:
                # The control request was serviced, but the supervisor has not
                # published quiescence proof yet. Keep the handle owned; the
                # caller can retry after the receipt publisher advances.
                return
        if reap_task.cancelled():
            raise RuntimeError("native process tree settlement was cancelled")
        if reap_task.exception() is not None:
            raise RuntimeError("native process tree settlement failed") from reap_task.exception()
        if not self._receipt_published():
            raise RuntimeError("native process tree quiescence proof is missing")

    async def close_control(self) -> None:
        if self._control_fd >= 0:
            fd, self._control_fd = self._control_fd, -1
            with suppress(OSError):
                os.close(fd)


async def _empty_bytes() -> bytes:
    return b""


@dataclass(frozen=True, slots=True)
class _SupervisorStatus:
    kind: Literal["ready", "root", "root_exit", "error"]
    pid: int | None = None
    returncode: int | None = None
    message: str | None = None


def _parse_status(line: bytes) -> _SupervisorStatus:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("native supervisor sent an invalid status record")
    raw = cast(dict[object, object], payload)
    data: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError("native supervisor sent an invalid status record")
        data[key] = value
    kind = data.get("type")
    if kind == "ready" and set(data) == {"type"}:
        return _SupervisorStatus("ready")
    if kind == "root" and set(data) == {"type", "pid"}:
        pid = data["pid"]
        if type(pid) is not int or pid <= 1:
            raise ValueError("native supervisor sent an invalid root pid")
        return _SupervisorStatus("root", pid=pid)
    if kind == "root_exit" and set(data) == {"type", "returncode"}:
        returncode = data["returncode"]
        if type(returncode) is not int:
            raise ValueError("native supervisor sent an invalid command result")
        return _SupervisorStatus("root_exit", returncode=returncode)
    if kind == "error" and set(data) == {"type", "message"}:
        message = data["message"]
        if not isinstance(message, str):
            raise ValueError("native supervisor sent an invalid error status")
        return _SupervisorStatus("error", message=message)
    raise ValueError("native supervisor sent an invalid status record")


def native_supervisor_available() -> bool:
    return tree_ownership_available()


def _supervisor_module() -> str:
    module = "kohaku.agent.tools.native_process_supervisor"
    return f"{module}_darwin" if tree_ownership() == "darwin_seatbelt" else module


async def spawn_native_process(
    *,
    command: str,
    cwd: str,
    env: dict[str, str],
    shell_program: str | None,
    login: bool,
    reservation: NativeProcessReservation | None = None,
    confinement: SeatbeltConfinement | None = None,
) -> NativeProcessHandle:
    """Start a supervisor, persist its identity, then release its launch gate.

    A launch without a reservation must be a confined one. It has no admission
    ledger: it ends when this process stops listening, and its proof lives only
    as long as the handle.
    """
    if reservation is None and confinement is None:
        raise ValueError("an unconfined native launch needs an admission reservation")
    if not native_supervisor_available():
        if reservation is not None:
            reservation.mark_unproven()
        raise RuntimeError("whole-tree native process proof is unavailable on this platform")

    gate_read, gate_write = os.pipe()
    status_read, status_write = os.pipe()
    control_read, control_write = os.pipe()
    config_read, config_write = os.pipe()
    helper: asyncio.subprocess.Process | None = None
    handle: NativeProcessHandle | None = None
    gate_open = False
    transferred = False
    status_stream = None
    permission_owner = reservation.permission_owner if reservation is not None else None
    permission_reference = (
        reservation.permission_reference if reservation is not None else None
    )
    registered_permission_handle = False
    ephemeral_directory: Path | None = None
    try:
        if reservation is None:
            ephemeral_directory = Path(tempfile.mkdtemp(prefix="kohaku-launch-"))
            launch_id = secrets.token_hex(16)
            receipt_dir = ephemeral_directory / "receipts"
        else:
            launch_id = reservation.launch_id
            receipt_dir = reservation.ledger.receipt_dir
        helper = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-m",
            _supervisor_module(),
            "--gate-fd",
            str(gate_read),
            "--status-fd",
            str(status_write),
            "--control-fd",
            str(control_read),
            "--config-fd",
            str(config_read),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
            pass_fds=(gate_read, status_write, control_read, config_read),
            # A terminal interrupt must not end the supervisor of a launch that
            # it has to clean up once this process is gone.
            start_new_session=reservation is None,
        )
        for fd in (gate_read, status_write, control_read, config_read):
            os.close(fd)
        gate_read = status_write = control_read = config_read = -1
        status_stream = os.fdopen(status_read, "rb", buffering=0)
        status_read = -1
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, status_stream)
        handle = NativeProcessHandle(helper, reader, control_write)
        handle.set_receipt_path(
            receipt_dir / f"{launch_id}.receipt",
            ephemeral_directory=ephemeral_directory,
        )
        control_write = -1
        config: dict[str, object] = {
            "command": command,
            "cwd": cwd,
            "env": env,
            "shell_program": shell_program or "/bin/bash",
            "login": login,
            "launch_id": launch_id,
            "receipt_dir": str(receipt_dir),
        }
        if confinement is not None:
            config["confinement"] = {
                "rules": confinement.rules,
                "params": confinement.params,
                "argv": confinement.argv,
                "die_with_parent": reservation is None,
            }
        payload = json.dumps(config, separators=(",", ":")).encode()
        await asyncio.to_thread(_write_all, config_write, payload)
        os.close(config_write)
        config_write = -1
        await asyncio.wait_for(handle.wait_ready(), timeout=5)
        if (
            reservation is not None
            and permission_owner is not None
            and permission_reference is not None
        ):
            ledger = reservation.ledger
            owner_tag = reservation.owner
            if owner_tag is None:
                raise RuntimeError("workspace native reservation is missing its owner tag")
            if not permission_owner.register_native_stdin(
                permission_reference,
                handle,
                handle.close_stdin_now,
                lambda: ledger.owner_counts(owner_tag),
            ):
                raise PermissionError("workspace permission changed before native launch")
            registered_permission_handle = True
            handle.set_on_supervisor_exit(
                lambda: permission_owner.unregister_native_stdin(permission_reference, handle)
            )
        def open_gate() -> None:
            if os.write(gate_write, b"GO\n") != 3:
                raise OSError("native command launch gate did not open")

        if reservation is None:
            open_gate()
        else:
            starttime = process_start_time(helper.pid)
            if starttime is None:
                raise RuntimeError("native supervisor identity could not be read")
            reservation.admit_launch(helper.pid, starttime, open_gate)
        gate_open = True
        os.close(gate_write)
        gate_write = -1
        await asyncio.wait_for(handle.wait_root_started(), timeout=10)
        transferred = True
        return handle
    except BaseException:
        if gate_open:
            if reservation is not None:
                reservation.mark_unproven()
            if handle is not None:
                await handle.kill_process_tree()
        else:
            if reservation is not None:
                reservation.cancel_if_unlaunched()
            if helper is not None and helper.returncode is None:
                with suppress(ProcessLookupError):
                    helper.kill()
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(helper.wait(), timeout=5)
        raise
    finally:
        for fd in (gate_read, gate_write, status_read, status_write, control_read, control_write, config_read, config_write):
            if fd >= 0:
                with suppress(OSError):
                    os.close(fd)
        if status_stream is not None and not transferred:
            with suppress(OSError):
                status_stream.close()
        if handle is not None and not transferred:
            await handle.close_control()
        # Once a handle exists, it removes the folder when the supervisor exits.
        if handle is None and ephemeral_directory is not None:
            shutil.rmtree(ephemeral_directory, ignore_errors=True)
        if registered_permission_handle and handle is not None and not transferred:
            assert permission_owner is not None and permission_reference is not None
            permission_owner.unregister_native_stdin(permission_reference, handle)


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]
