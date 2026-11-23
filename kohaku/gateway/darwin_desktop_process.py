"""macOS process proof and instance-bound signaling for the desktop gateway."""

from __future__ import annotations

import ctypes
import os
import select
import signal
import struct
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_CTL_KERN = 1
_KERN_PROCARGS2 = 49
_TASK_AUDIT_TOKEN = 15
_AUDIT_TOKEN = ctypes.c_uint32 * 8


@lru_cache(maxsize=1)
def _libsystem() -> ctypes.CDLL:
    lib = ctypes.CDLL(None, use_errno=True)
    lib.task_name_for_pid.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p]
    lib.task_info.argtypes = [
        ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.mach_port_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    lib.sysctl.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_size_t,
    ]
    return lib


@lru_cache(maxsize=1)
def _libproc() -> ctypes.CDLL:
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    lib.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    lib.proc_signal_with_audittoken.argtypes = [ctypes.c_void_p, ctypes.c_int]
    return lib


def darwin_desktop_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        _libsystem()
        _libproc()
    except (AttributeError, OSError):
        return False
    return True


def _audit_token(pid: int) -> ctypes.Array[ctypes.c_uint32]:
    lib = _libsystem()
    current_task = ctypes.c_uint32.in_dll(lib, "mach_task_self_")
    target_task = ctypes.c_uint32()
    if lib.task_name_for_pid(current_task, pid, ctypes.byref(target_task)) != 0:
        raise OSError("could not obtain gateway task name")
    try:
        token = _AUDIT_TOKEN()
        count = ctypes.c_uint32(len(token))
        result = lib.task_info(target_task, _TASK_AUDIT_TOKEN, token, ctypes.byref(count))
        # audit_token_to_euid() and audit_token_to_pid() use words 1 and 5.
        if (
            result != 0 or count.value != len(token)
            or token[1] != os.geteuid() or token[5] != pid
        ):
            raise OSError("could not verify gateway task identity")
        return token
    finally:
        lib.mach_port_deallocate(current_task, target_task)


def _process_invocation(pid: int, launcher: str) -> tuple[list[str], bool]:
    lib = _libsystem()
    mib = (ctypes.c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, pid)
    size = ctypes.c_size_t()
    size_ptr = ctypes.byref(size)
    if lib.sysctl(mib, len(mib), None, size_ptr, None, 0) != 0:
        raise OSError("could not size gateway arguments")
    if not 4 <= size.value <= os.sysconf("SC_ARG_MAX"):
        raise OSError("invalid gateway argument size")
    data = ctypes.create_string_buffer(size.value)
    if lib.sysctl(mib, len(mib), data, size_ptr, None, 0) != 0:
        raise OSError("could not read gateway arguments")
    # Layout: argc, executable path, NUL padding, argv, environment.
    argc = struct.unpack_from("=i", data)[0]
    if argc <= 0:
        raise ValueError("invalid gateway argument count")
    _, _, arguments = data.raw[4:size.value].partition(b"\0")
    parts = arguments.lstrip(b"\0").split(b"\0", argc)
    if len(parts) != argc + 1:
        raise ValueError("incomplete gateway argument vector")
    # Framework CPython re-execs Python.app, carrying the original interpreter
    # selector in its exec environment. Inspect only the launch metadata, never
    # expose the remaining environment to callers or logs.
    entries = parts[argc].split(b"\0")
    selected = os.fsencode(launcher)
    launchers = [entry for entry in entries if entry.startswith(b"__PYVENV_LAUNCHER__=")]
    overrides = [entry for entry in entries if entry.startswith(b"PYTHONEXECUTABLE=")]
    framework_launch = launchers == [b"__PYVENV_LAUNCHER__=" + selected] and all(
        entry in (b"PYTHONEXECUTABLE=", b"PYTHONEXECUTABLE=" + selected) for entry in overrides
    )
    return [os.fsdecode(part) for part in parts[:argc]], framework_launch


def _framework_image(launcher: str) -> Path | None:
    resolved = Path(os.path.realpath(launcher, strict=True))
    version = resolved.parent.parent
    if (
        resolved.parent.name != "bin"
        or resolved.name != f"python{version.name}"
        or version.parent.name != "Versions"
        or version.parent.parent.name != "Python.framework"
    ):
        return None
    return version / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"


def darwin_gateway_matches(pid: int, command: list[str], python_executable: str) -> bool:
    """Require same-user execution of the exact selected gateway command."""
    try:
        if not command or command[0] != python_executable:
            return False
        _audit_token(pid)
        executable = ctypes.create_string_buffer(4096)  # PROC_PIDPATHINFO_MAXSIZE
        if _libproc().proc_pidpath(pid, executable, len(executable)) <= 0:
            return False
        argv, framework_launch = _process_invocation(pid, python_executable)
        if not (argv == command and os.path.samefile(executable.value, python_executable)):
            if not framework_launch or argv[1:] != command[1:]:
                return False
            image = _framework_image(python_executable)
            if (
                image is None
                or not Path(argv[0]).is_absolute()
                or not os.path.samefile(executable.value, image)
                or not os.path.samefile(argv[0], image)
            ):
                return False
        return os.getpgid(pid) == pid and os.getsid(pid) == pid
    except (OSError, ValueError):
        return False


@dataclass(frozen=True)
class DarwinProcessWatch:
    """An exit watch plus an audit token containing the kernel PID generation."""

    _watch: select.kqueue
    _token: ctypes.Array[ctypes.c_uint32]

    @classmethod
    def capture(cls, pid: int) -> DarwinProcessWatch:
        # Register before the caller revalidates birth identity. A watch or
        # token captured for a replacement process cannot authorize a stop.
        watch = select.kqueue()
        try:
            event = select.kevent(
                pid, filter=select.KQ_FILTER_PROC,
                flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                fflags=select.KQ_NOTE_EXIT,
            )
            watch.control([event], 0, 0)
            return cls(watch, _audit_token(pid))
        except BaseException:
            watch.close()
            raise

    def send_signal(self, signum: signal.Signals) -> None:
        # Do not substitute kill(pid): the kernel compares pidversion when
        # delivering a signal by audit token, even if the PID has been reused.
        if _libproc().proc_signal_with_audittoken(self._token, signum) != 0:
            raise OSError(ctypes.get_errno(), "could not signal captured gateway")

    def wait_for_exit(self, timeout_s: float) -> bool:
        events = self._watch.control(None, 1, timeout_s)
        return any(event.fflags & select.KQ_NOTE_EXIT for event in events)

    def close(self) -> None:
        self._watch.close()
