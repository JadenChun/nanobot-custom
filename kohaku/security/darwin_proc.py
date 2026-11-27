"""Darwin process primitives for the Seatbelt executor (A3-2M, WP2).

Everything here is read-only except seatbelt_self_test, which launches three
short-lived children that exit by themselves. The only signal this module can
send is a last-resort kill of one of those direct children if it overruns its
timeout. Signalling a command's process tree belongs to the launch-domain engine
and is gated by the K1-K4 invariants in the A3-2M packet.

The membership predicate matches a process only when all three hold:
sandbox_check(pid, NULL, 0) reports it as sandboxed, a per-launch "member"
Mach global name is allowed, and a per-launch "fence" name is denied. A single
deny-marker is unsafe: App-sandboxed apps also deny unknown names.

sandbox_check is private Apple API (gate G1, accepted behind this self-test).
Only the Apple Silicon calling convention is qualified (gate G5); any other ABI
is refused, so the executor fails closed.
"""

from __future__ import annotations

import ctypes
import errno
import functools
import os
import platform
import secrets
import struct
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

_LIBSYSTEM = "/usr/lib/libSystem.B.dylib"
_PROC_ALL_PIDS = 1
_PROC_PIDTBSDINFO = 3
_PROC_PIDT_SHORTBSDINFO = 13
_BSDINFO_FORMAT = "<12I16s32s5IiQQ"
_BSDINFO_SIZE = struct.calcsize(_BSDINFO_FORMAT)
_SHORT_BSDINFO_FORMAT = "<4I16s8I"
_SHORT_BSDINFO_SIZE = struct.calcsize(_SHORT_BSDINFO_FORMAT)
_RENAME_EXCL = 0x4
_FILTER_NONE = 0
_FILTER_GLOBAL_NAME = 2
_ZOMBIE = 5
_LAUNCH_ID_LENGTH = 32
# sys/fcntl.h and unistd.h values that Python does not expose on every platform.
_F_GETPATH = 50
_O_EVTONLY = 0x8000
_PATH_BUFFER = 1024
_CS_DARWIN_USER_TEMP_DIR = 65537


class DarwinUnavailableError(RuntimeError):
    """The primitive cannot run here; callers must fail closed."""


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    uid: int
    status: int
    start_us: int

    @property
    def zombie(self) -> bool:
        return self.status == _ZOMBIE


def new_launch_id() -> str:
    return secrets.token_hex(_LAUNCH_ID_LENGTH // 2)


def _check_launch_id(launch_id: str) -> str:
    if len(launch_id) != _LAUNCH_ID_LENGTH or any(c not in "0123456789abcdef" for c in launch_id):
        raise ValueError("launch id must be 32 lowercase hex characters")
    return launch_id


def member_name(launch_id: str) -> str:
    return f"dev.kohaku.launch.{_check_launch_id(launch_id)}.member"


def fence_name(launch_id: str) -> str:
    return f"dev.kohaku.launch.{_check_launch_id(launch_id)}.fence"


def marker_rules(launch_id: str) -> str:
    """Return the two SBPL rules that tag every process of one launch."""
    return (
        f'(allow mach-lookup (global-name "{member_name(launch_id)}"))\n'
        f'(deny mach-lookup (global-name "{fence_name(launch_id)}"))\n'
    )


def abi_supported() -> bool:
    """Only the arm64 variadic convention was qualified (gate G5)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@functools.lru_cache(maxsize=1)
def _libc() -> ctypes.CDLL:
    try:
        return ctypes.CDLL(_LIBSYSTEM, use_errno=True)
    except OSError as exc:
        raise DarwinUnavailableError("libSystem is unavailable") from exc


def _symbol(name: str) -> ctypes._NamedFuncPointer:  # pyright: ignore[reportPrivateUsage]
    try:
        return getattr(_libc(), name)
    except AttributeError as exc:
        raise DarwinUnavailableError(f"{name} is unavailable") from exc


def list_pids() -> list[int]:
    """Return every pid that proc_listallpids reports at this moment."""
    function = _symbol("proc_listallpids")
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_void_p, ctypes.c_int]
    count = int(function(None, 0))
    if count <= 0:
        raise DarwinUnavailableError("proc_listallpids failed")
    capacity = count * 2 + 64
    for _ in range(6):
        buffer = (ctypes.c_int * capacity)()
        returned = int(function(buffer, ctypes.sizeof(buffer)))
        if returned <= 0:
            raise DarwinUnavailableError("proc_listallpids failed")
        if returned < capacity:
            return [pid for pid in buffer[:returned] if pid > 0]
        capacity *= 2
    raise DarwinUnavailableError("process table kept growing during a scan")


def _pid_info(pid: int, flavor: int, size: int) -> bytes | None:
    function = _symbol("proc_pidinfo")
    function.restype = ctypes.c_int
    function.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int,
    ]
    buffer = ctypes.create_string_buffer(size)
    if int(function(pid, flavor, 0, buffer, size)) != size:
        return None
    return buffer.raw


def process_info(pid: int) -> ProcessInfo | None:
    """Return identity fields for one pid.

    None means the pid does not exist, is a zombie, or belongs to another user.
    """
    raw = _pid_info(pid, _PROC_PIDTBSDINFO, _BSDINFO_SIZE)
    if raw is None:
        return None
    fields = struct.unpack(_BSDINFO_FORMAT, raw)
    if int(fields[3]) != pid:
        return None
    return ProcessInfo(
        pid=pid,
        ppid=int(fields[4]),
        uid=int(fields[5]),
        status=int(fields[1]),
        start_us=int(fields[20]) * 1_000_000 + int(fields[21]),
    )


def process_start_time(pid: int) -> int | None:
    """Return a stable per-process start stamp in microseconds, if the pid exists."""
    info = process_info(pid)
    return None if info is None else info.start_us


def parent_pid(pid: int) -> int | None:
    """Return the parent of any live process, including another user's."""
    raw = _pid_info(pid, _PROC_PIDT_SHORTBSDINFO, _SHORT_BSDINFO_SIZE)
    if raw is None:
        return None
    fields = struct.unpack(_SHORT_BSDINFO_FORMAT, raw)
    return int(fields[1]) if int(fields[0]) == pid else None


@functools.lru_cache(maxsize=1)
def _sandbox_check_function() -> Callable[..., int]:
    if not abi_supported():
        raise DarwinUnavailableError("sandbox_check is qualified only on Apple Silicon")
    # sandbox_check is variadic. The arm64 Apple ABI passes variadic arguments
    # on the stack, so five padding longs fill x3-x7 and push the filter
    # argument onto the stack where the callee reads it.
    prototype = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_long, ctypes.c_long,
        ctypes.c_char_p,
    )
    try:
        return prototype(("sandbox_check", _libc()))
    except AttributeError as exc:
        raise DarwinUnavailableError("sandbox_check is unavailable") from exc


def is_sandboxed(pid: int) -> bool | None:
    """True when the process runs under any Seatbelt profile; None when unknown."""
    result = int(_sandbox_check_function()(pid, None, _FILTER_NONE, 0, 0, 0, 0, 0, None))
    if result == 1:
        return True
    if result == 0:
        return False
    return None


def denies_global_name(pid: int, name: str) -> bool | None:
    """True when the target's own profile denies mach-lookup of name; None when unknown."""
    result = int(
        _sandbox_check_function()(
            pid, b"mach-lookup", _FILTER_GLOBAL_NAME, 0, 0, 0, 0, 0, name.encode("ascii"),
        )
    )
    if result == 1:
        return True
    if result == 0:
        return False
    return None


def membership(pid: int, launch_id: str) -> bool | None:
    """Whether pid belongs to one launch; None when the kernel gives no answer.

    A pid that no longer exists, or is a zombie, reports as sandboxed with every
    name denied, so it is correctly not a member.
    """
    sandboxed = is_sandboxed(pid)
    if sandboxed is not True:
        return sandboxed
    member_denied = denies_global_name(pid, member_name(launch_id))
    if member_denied is not False:
        return None if member_denied is None else False
    return denies_global_name(pid, fence_name(launch_id))


def is_member(pid: int, launch_id: str) -> bool:
    """Exact membership of one launch. Anything unknown is not a member."""
    return membership(pid, launch_id) is True


def scan_members(launch_id: str, pids: Iterable[int] | None = None) -> list[int]:
    """Read-only scan. Returns the pids that currently satisfy is_member."""
    candidates = list_pids() if pids is None else list(pids)
    return [pid for pid in candidates if is_member(pid, launch_id)]


def host_uuid() -> str:
    function = _symbol("gethostuuid")
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    buffer = ctypes.create_string_buffer(16)
    wait = (ctypes.c_long * 2)(0, 0)
    if int(function(buffer, wait)) != 0:
        raise DarwinUnavailableError("gethostuuid failed")
    return str(uuid.UUID(bytes=buffer.raw))


def boot_session_uuid() -> str:
    function = _symbol("sysctlbyname")
    function.restype = ctypes.c_int
    function.argtypes = [
        ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p, ctypes.c_size_t,
    ]
    buffer = ctypes.create_string_buffer(64)
    size = ctypes.c_size_t(ctypes.sizeof(buffer))
    if int(function(b"kern.bootsessionuuid", buffer, ctypes.byref(size), None, 0)) != 0:
        raise DarwinUnavailableError("kern.bootsessionuuid is unavailable")
    value = buffer.value.decode("ascii").strip()
    if not value:
        raise DarwinUnavailableError("kern.bootsessionuuid is empty")
    return value


def rename_excl(source: Path, target: Path) -> None:
    """Publish source at target without replacing an existing target."""
    function = _symbol("renamex_np")
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    if int(function(os.fsencode(source), os.fsencode(target), _RENAME_EXCL)) != 0:
        code = ctypes.get_errno()
        raise OSError(code or errno.EIO, os.strerror(code or errno.EIO), str(target))


def canonical_path(path: Path) -> Path:
    """Spell an absolute path the way the kernel does, as Seatbelt rules require.

    Symlinks, firmlinks and letter case are resolved for the part that exists;
    the rest is kept as given. Raises if an existing part cannot be spelled,
    because a rule with another spelling would not match.
    """
    current = path.resolve(strict=False)
    missing: list[str] = []
    while True:
        try:
            spelled = _kernel_path(current)
            break
        except (FileNotFoundError, NotADirectoryError):
            pass
        except OSError:
            # Sockets cannot be opened; find their on-disk name in the parent.
            parent = canonical_path(current.parent)
            spelled = parent / _entry_name(parent, current)
            break
        if current == current.parent:
            raise DarwinUnavailableError("the filesystem root cannot be spelled")
        missing.append(current.name)
        current = current.parent
    return spelled.joinpath(*reversed(missing))


def _kernel_path(path: Path) -> Path:
    if platform.system() != "Darwin":
        raise DarwinUnavailableError("kernel path spelling needs macOS")
    import fcntl

    flags = _O_EVTONLY | os.O_NONBLOCK | os.O_CLOEXEC
    descriptor = os.open(path, flags)
    try:
        raw = fcntl.fcntl(descriptor, _F_GETPATH, bytes(_PATH_BUFFER))
    finally:
        os.close(descriptor)
    return Path(os.fsdecode(raw.split(b"\0", 1)[0]))


def _entry_name(parent: Path, path: Path) -> str:
    wanted = path.lstat()
    with os.scandir(parent) as entries:
        matches = [
            entry.name for entry in entries
            if os.path.samestat(entry.stat(follow_symlinks=False), wanted)
        ]
    if path.name in matches:
        return path.name
    if len(matches) != 1:
        raise DarwinUnavailableError("a path has no single on-disk name")
    return matches[0]


def user_temp_dir() -> str | None:
    """The per-user temporary folder macOS assigns, if this host reports one."""
    if platform.system() != "Darwin":
        return None
    try:
        return os.confstr(_CS_DARWIN_USER_TEMP_DIR)
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class SelfTestResult:
    ok: bool
    failures: tuple[str, ...]


_CHILD_SECONDS = 3
# EX_OSERR: sandbox-exec's exit status when sandbox_apply is refused.
_APPLY_REFUSED_STATUS = 71
_ESTABLISH_SECONDS = 2.5

_DENY_DEFAULT_PROFILE = (
    "(version 1)\n(deny default)\n(allow process-exec)\n(allow file-read*)\n"
    "(allow sysctl-read)\n"
)


def _spawn_sandboxed(profile: Path, *wrapper: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [SANDBOX_EXEC, "-f", str(profile), *wrapper, "/bin/sleep", str(_CHILD_SECONDS)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True, start_new_session=True,
    )


def seatbelt_self_test() -> SelfTestResult:
    """Prove the membership predicate on this host before anything relies on it.

    Launches one marked process, one deny-default control, and one marked process
    that tries to replace its profile. All exit by themselves and no signal is
    sent. Any failure means tree ownership must stay unavailable.
    """
    if not abi_supported():
        return SelfTestResult(False, ("unsupported platform or ABI",))
    failures: list[str] = []
    launch_id = new_launch_id()
    foreign_id = new_launch_id()
    children: list[subprocess.Popen[bytes]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="kohaku-selftest-") as directory:
            root = Path(directory)
            os.chmod(root, 0o700)
            marked = root / "marked.sb"
            marked.write_text(
                "(version 1)\n(allow default)\n" + marker_rules(launch_id), encoding="ascii",
            )
            control = root / "control.sb"
            control.write_text(_DENY_DEFAULT_PROFILE, encoding="ascii")
            for path in (marked, control):
                os.chmod(path, 0o600)
            member = _spawn_sandboxed(marked)
            children.append(member)
            probe = _spawn_sandboxed(control)
            children.append(probe)

            deadline = time.monotonic() + _ESTABLISH_SECONDS
            while time.monotonic() < deadline and not is_member(member.pid, launch_id):
                time.sleep(0.02)
            if not is_member(member.pid, launch_id):
                failures.append("the marked launch never became a member")
            if is_member(member.pid, foreign_id):
                failures.append("a foreign launch id matched the marked process")
            if is_member(os.getpid(), launch_id):
                failures.append("the unsandboxed supervisor matched as a member")
            while time.monotonic() < deadline and is_sandboxed(probe.pid) is not True:
                time.sleep(0.02)
            if is_sandboxed(probe.pid) is not True:
                failures.append("the deny-default control never became sandboxed")
            elif is_member(probe.pid, launch_id):
                failures.append("a deny-default sandbox without the member name matched")
            matched = scan_members(launch_id)
            if matched != [member.pid]:
                failures.append(f"scan matched {len(matched)} processes, expected exactly 1")
            # The proof assumes a member can never shed its markers.
            other = root / "other.sb"
            other.write_text(
                "(version 1)\n(allow default)\n(deny network*)\n", encoding="ascii",
            )
            os.chmod(other, 0o600)
            replacer = _spawn_sandboxed(marked, SANDBOX_EXEC, "-f", str(other))
            children.append(replacer)
            if replacer.wait(timeout=_CHILD_SECONDS + 5) != _APPLY_REFUSED_STATUS:
                failures.append("a marked process was not refused a different profile")
    except (OSError, DarwinUnavailableError, ValueError, subprocess.SubprocessError) as exc:
        failures.append(f"self-test could not run: {type(exc).__name__}")
    finally:
        for child in children:
            try:
                child.wait(timeout=_CHILD_SECONDS + 5)
            except subprocess.TimeoutExpired:
                # Our own direct child that did not exit on its own.
                child.kill()
                child.wait()
    return SelfTestResult(not failures, tuple(failures))
