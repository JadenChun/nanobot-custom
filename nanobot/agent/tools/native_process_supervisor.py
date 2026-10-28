"""Linux subreaper that proves an unrestricted command tree is quiescent."""

from __future__ import annotations

import ctypes
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import TypedDict, cast

from nanobot.security.native_admission import (
    QuiescenceReceipt,
    current_host_identity,
    process_start_time,
    write_quiescence_receipt,
)

_PR_SET_CHILD_SUBREAPER = 36
_CHUNK = 64 * 1024
_INPUT_BUFFER_LIMIT = 256 * 1024
_OUTPUT_BUFFER_LIMIT = 256 * 1024
_LAUNCH_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class NativeLaunchConfig(TypedDict):
    command: str
    cwd: str
    env: dict[str, str]
    shell_program: str
    login: bool
    launch_id: str
    receipt_dir: str


def _send(status_fd: int, payload: dict[str, object]) -> None:
    data = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
    os.write(status_fd, data)


def _read_exact_json(fd: int) -> NativeLaunchConfig:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = os.read(fd, _CHUNK)
        if not chunk:
            break
        size += len(chunk)
        if size > 1024 * 1024:
            raise ValueError("native command launch data is too large")
        chunks.append(chunk)
    payload: object = json.loads(b"".join(chunks))
    if not isinstance(payload, dict):
        raise ValueError("native command launch data is invalid")
    raw = cast(dict[object, object], payload)
    data: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError("native command launch data is invalid")
        data[key] = value
    required = {
        "command", "cwd", "env", "shell_program", "login", "launch_id", "receipt_dir",
    }
    if set(data) != required:
        raise ValueError("native command launch data has an invalid shape")
    command = data["command"]
    cwd = data["cwd"]
    shell_program = data["shell_program"]
    login = data["login"]
    launch_id = data["launch_id"]
    receipt_dir = data["receipt_dir"]
    env_value = data["env"]
    if (
        not isinstance(command, str)
        or not isinstance(cwd, str)
        or not isinstance(shell_program, str)
        or not shell_program
        or not isinstance(login, bool)
        or not isinstance(launch_id, str)
        or not _LAUNCH_ID_RE.fullmatch(launch_id)
        or not isinstance(receipt_dir, str)
        or not receipt_dir
        or not isinstance(env_value, dict)
    ):
        raise ValueError("native command launch data is invalid")
    raw_env = cast(dict[object, object], env_value)
    env: dict[str, str] = {}
    for key, value in raw_env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("native command launch data environment is invalid")
        env[key] = value
    return {
        "command": command,
        "cwd": cwd,
        "env": env,
        "shell_program": shell_program,
        "login": login,
        "launch_id": launch_id,
        "receipt_dir": receipt_dir,
    }


def _subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _spawn_root(payload: NativeLaunchConfig) -> subprocess.Popen[bytes]:
    shell = payload["shell_program"]
    command = payload["command"]
    args = [shell]
    if payload["login"] and Path(shell).name.lower() in {"bash", "zsh"}:
        args.append("-l")
    args.extend(("-c", command))
    cwd = payload["cwd"]
    env = payload["env"]
    return subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        close_fds=True,
        start_new_session=True,
        bufsize=0,
    )


def _supervise(
    payload: NativeLaunchConfig,
    *,
    gate_fd: int,
    status_fd: int,
    control_fd: int,
) -> int:
    child: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    output_open: set[str] = {"stdout", "stderr"}
    root_returncode: int | None = None
    root_reported = False
    kill_requested = False
    discard_output = False
    tree_quiescent = False
    output_fds_closed: set[str] = set()
    receipt_thread: threading.Thread | None = None
    receipt_finished = threading.Event()
    receipt_errors: list[BaseException] = []
    receipt_published = False

    try:
        _subreaper()
        _send(status_fd, {"type": "ready"})
        gate = bytearray()
        while not gate.endswith(b"\n"):
            chunk = os.read(gate_fd, 64)
            if not chunk:
                raise RuntimeError("native command launch gate closed")
            gate.extend(chunk)
            if len(gate) > 64:
                raise ValueError("native command launch gate is invalid")
        if bytes(gate) != b"GO\n":
            raise ValueError("native command launch gate is invalid")
        os.close(gate_fd)
        gate_fd = -1

        child = _spawn_root(payload)
        stdin_stream = child.stdin
        stdout_stream = child.stdout
        stderr_stream = child.stderr
        assert stdin_stream is not None and stdout_stream is not None and stderr_stream is not None
        _send(status_fd, {"type": "root", "pid": child.pid})

        output_streams = {"stdout": stdout_stream, "stderr": stderr_stream}
        output_fds = {"stdout": 1, "stderr": 2}
        output_buffers = {"stdout": bytearray(), "stderr": bytearray()}
        output_read_registered: set[str] = set()
        output_write_registered: set[str] = set()
        output_sinks_closed: set[str] = set()
        for name, stream in output_streams.items():
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, ("child_output", name))
            output_read_registered.add(name)
            os.set_blocking(output_fds[name], False)
        input_fd = sys.stdin.fileno()
        os.set_blocking(input_fd, False)
        os.set_blocking(stdin_stream.fileno(), False)
        selector.register(input_fd, selectors.EVENT_READ, ("input", ""))
        selector.register(control_fd, selectors.EVENT_READ, ("control", ""))

        pending_input = bytearray()
        input_read_registered = True
        input_read_paused = False
        child_stdin_write_registered = False
        input_eof = False
        input_closed = False
        while True:
            if not root_reported:
                polled = child.poll()
                if polled is not None:
                    root_returncode = polled
                    root_reported = True
                    _send(status_fd, {"type": "root_exit", "returncode": polled})

            for key, _mask in selector.select(timeout=0.02):
                kind, name = key.data
                if kind == "input":
                    capacity = _INPUT_BUFFER_LIMIT - len(pending_input)
                    if capacity <= 0:
                        selector.unregister(input_fd)
                        input_read_registered = False
                        input_read_paused = True
                        continue
                    try:
                        chunk = os.read(input_fd, min(_CHUNK, capacity))
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(input_fd)
                        input_read_registered = False
                        input_eof = True
                        if not pending_input and not input_closed:
                            with suppress(OSError):
                                stdin_stream.close()
                            input_closed = True
                    else:
                        pending_input.extend(chunk)
                        if not child_stdin_write_registered:
                            selector.register(
                                stdin_stream,
                                selectors.EVENT_WRITE,
                                ("child_stdin", ""),
                            )
                            child_stdin_write_registered = True
                        if len(pending_input) >= _INPUT_BUFFER_LIMIT:
                            selector.unregister(input_fd)
                            input_read_registered = False
                            input_read_paused = True
                elif kind == "control":
                    try:
                        control = os.read(control_fd, 128)
                    except BlockingIOError:
                        continue
                    if not control:
                        selector.unregister(control_fd)
                    elif b"KILL" in control:
                        kill_requested = True
                        discard_output = True
                        for output_name in ("stdout", "stderr"):
                            output_buffers[output_name].clear()
                            if output_name in output_write_registered:
                                selector.unregister(output_fds[output_name])
                                output_write_registered.discard(output_name)
                            if (
                                output_name in output_open
                                and output_name not in output_read_registered
                            ):
                                selector.register(
                                    output_streams[output_name],
                                    selectors.EVENT_READ,
                                    ("child_output", output_name),
                                )
                                output_read_registered.add(output_name)
                        with suppress(OSError):
                            os.killpg(child.pid, signal.SIGKILL)
                elif kind == "child_stdin":
                    try:
                        written = os.write(stdin_stream.fileno(), pending_input)
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        pending_input.clear()
                        if child_stdin_write_registered:
                            selector.unregister(stdin_stream.fileno())
                            child_stdin_write_registered = False
                        if input_read_registered:
                            selector.unregister(input_fd)
                            input_read_registered = False
                        with suppress(OSError):
                            os.close(input_fd)
                        input_eof = True
                        input_closed = True
                        with suppress(OSError):
                            stdin_stream.close()
                    else:
                        del pending_input[:written]
                        if not pending_input:
                            selector.unregister(stdin_stream.fileno())
                            child_stdin_write_registered = False
                            if input_eof and not input_closed:
                                with suppress(OSError):
                                    stdin_stream.close()
                                input_closed = True
                        if (
                            input_read_paused
                            and len(pending_input) < _INPUT_BUFFER_LIMIT
                            and not input_eof
                        ):
                            selector.register(input_fd, selectors.EVENT_READ, ("input", ""))
                            input_read_registered = True
                            input_read_paused = False
                elif kind == "child_output":
                    capacity = _OUTPUT_BUFFER_LIMIT - len(output_buffers[name])
                    if not discard_output and name not in output_sinks_closed and capacity <= 0:
                        selector.unregister(key.fileobj)
                        output_read_registered.discard(name)
                        continue
                    try:
                        read_size = (
                            _CHUNK
                            if discard_output or name in output_sinks_closed
                            else min(_CHUNK, capacity)
                        )
                        chunk = os.read(key.fd, read_size)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        output_read_registered.discard(name)
                        output_open.discard(name)
                        output_streams[name].close()
                    elif not discard_output and name not in output_sinks_closed:
                        output_buffers[name].extend(chunk)
                        if name not in output_write_registered:
                            selector.register(
                                output_fds[name],
                                selectors.EVENT_WRITE,
                                ("supervisor_output", name),
                            )
                            output_write_registered.add(name)
                        if len(output_buffers[name]) >= _OUTPUT_BUFFER_LIMIT:
                            selector.unregister(key.fileobj)
                            output_read_registered.discard(name)
                elif kind == "supervisor_output":
                    buffer = output_buffers[name]
                    try:
                        written = os.write(key.fd, buffer)
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        selector.unregister(key.fd)
                        output_write_registered.discard(name)
                        output_sinks_closed.add(name)
                        buffer.clear()
                    else:
                        del buffer[:written]
                        if not buffer:
                            selector.unregister(key.fd)
                            output_write_registered.discard(name)
                    if (
                        name in output_open
                        and name not in output_read_registered
                        and len(buffer) < _OUTPUT_BUFFER_LIMIT
                    ):
                        selector.register(
                            output_streams[name],
                            selectors.EVENT_READ,
                            ("child_output", name),
                        )
                        output_read_registered.add(name)

            for name, output_fd in output_fds.items():
                if name in output_fds_closed:
                    continue
                if name in output_sinks_closed or (
                    name not in output_open and not output_buffers[name]
                ):
                    with suppress(OSError):
                        os.close(output_fd)
                    output_fds_closed.add(name)

            if not root_reported:
                continue
            if kill_requested:
                _kill_adopted_children(child.pid)
            if not tree_quiescent and _wait_for_all_children(child.pid):
                tree_quiescent = True
                identity = current_host_identity()
                supervisor_starttime = process_start_time(os.getpid())
                if not identity.trusted or supervisor_starttime is None:
                    raise RuntimeError("native supervisor identity is unavailable")
                host_id = identity.host_id
                boot_id = identity.boot_id
                pid_namespace = identity.pid_namespace
                if host_id is None or boot_id is None or pid_namespace is None:
                    raise RuntimeError("native supervisor identity is incomplete")
                receipt: QuiescenceReceipt = {
                    "version": 1,
                    "launch_id": payload["launch_id"],
                    "host_id": host_id,
                    "boot_id": boot_id,
                    "pid_namespace": pid_namespace,
                    "supervisor_pid": os.getpid(),
                    "supervisor_starttime": supervisor_starttime,
                    "result": "quiescent",
                    "completed_ns": time.time_ns(),
                }

                def publish_receipt() -> None:
                    try:
                        write_quiescence_receipt(Path(payload["receipt_dir"]), receipt)
                    except BaseException as exc:
                        receipt_errors.append(exc)
                    finally:
                        receipt_finished.set()

                receipt_thread = threading.Thread(
                    target=publish_receipt,
                    name="native-quiescence-receipt",
                    daemon=True,
                )
                receipt_thread.start()

            if receipt_thread is not None and receipt_finished.is_set() and not receipt_published:
                receipt_thread.join()
                if receipt_errors:
                    raise receipt_errors[0]
                receipt_published = True

            if (
                not output_open
                and not any(output_buffers.values())
                and tree_quiescent
                and receipt_published
            ):
                break

        if root_returncode is None:
            root_returncode = child.wait()
        return root_returncode
    except BaseException as exc:
        try:
            _send(status_fd, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        except OSError:
            pass
        if child is not None and child.poll() is None:
            with suppress(OSError):
                os.killpg(child.pid, signal.SIGKILL)
        if child is not None:
            with suppress(Exception):
                child.wait(timeout=5)
        return 125
    finally:
        selector.close()
        for fd in (gate_fd, status_fd, control_fd):
            if fd >= 0:
                with suppress(OSError):
                    os.close(fd)


def _kill_adopted_children(root_pid: int) -> None:
    task_children = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    while True:
        try:
            child_ids = [int(value) for value in task_children.read_text().split()]
        except (OSError, ValueError):
            child_ids = []
        for pid in child_ids:
            if pid != root_pid:
                with suppress(OSError):
                    os.kill(pid, signal.SIGKILL)
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG | getattr(os, "__WALL", 0x40000000))
        except ChildProcessError:
            return
        except OSError:
            return
        if pid == 0:
            break


def _wait_for_all_children(root_pid: int) -> bool:
    flags = os.WNOHANG | getattr(os, "__WALL", 0x40000000)
    try:
        while True:
            pid, _status = os.waitpid(-1, flags)
            if pid == 0:
                return False
            if pid != root_pid:
                continue
    except ChildProcessError:
        return True
    except OSError:
        return False


def main() -> int:
    args = sys.argv[1:]
    gate_fd = -1
    status_fd = -1
    control_fd = -1
    config_fd = -1
    payload: NativeLaunchConfig | None = None
    try:
        gate_fd = int(args[args.index("--gate-fd") + 1])
        status_fd = int(args[args.index("--status-fd") + 1])
        control_fd = int(args[args.index("--control-fd") + 1])
        config_fd = int(args[args.index("--config-fd") + 1])
        payload = _read_exact_json(config_fd)
    except (ValueError, IndexError, OSError, json.JSONDecodeError) as exc:
        print(f"native command supervisor setup failed: {exc}", file=sys.stderr)
        return 125
    finally:
        if config_fd >= 0:
            with suppress(OSError):
                os.close(config_fd)
    return _supervise(payload, gate_fd=gate_fd, status_fd=status_fd, control_fd=control_fd)


if __name__ == "__main__":
    raise SystemExit(main())
