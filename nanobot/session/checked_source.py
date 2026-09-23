"""Checked lock-only authority for one current JSONL session namespace.

This dormant migration capability pins the configured directory ancestry and
holds the repository's migration, execution, and session-file flock locks.
It deliberately exposes no source-file or arbitrary lock operations.
"""

from __future__ import annotations

import errno
import os
import stat
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

try:  # pragma: no cover - platform qualification covers unavailable fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from nanobot.session.manager import JsonlSessionStore


ErrorCode = Literal[
    "invalid", "unavailable", "unsafe", "changed", "busy", "missing", "closed", "io",
    "limit",
]
_ALLOWED_LOCK_MODES = frozenset({0o600, 0o604, 0o640, 0o644})
_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_SOURCE_READ_CHUNK_SIZE = 64 * 1024
_MIGRATION_LOCK = ".workspace-migration.lock"
_SESSION_FILES_LOCK = ".session-files.lock"
_EXECUTION_DIRECTORY = ".execution-locks"
_CONSTRUCTOR_TOKEN = object()


class CheckedSessionAuthorityError(RuntimeError):
    """Content-free failure raised by the checked lock capability."""

    __slots__ = ("code",)

    def __init__(self, code: ErrorCode) -> None:
        if code not in {
            "invalid", "unavailable", "unsafe", "changed", "busy", "missing", "closed", "io",
            "limit",
        }:
            code = "io"
        self.code: ErrorCode = code
        super().__init__(f"checked session authority: {code}")


class _FailureError(Exception):
    __slots__ = ("code",)

    def __init__(self, code: ErrorCode) -> None:
        self.code = code


def _fail(code: ErrorCode) -> None:
    raise _FailureError(code) from None


def _error_code(exc: OSError) -> ErrorCode:
    if exc.errno == errno.ENOENT:
        return "missing"
    if exc.errno in {errno.ELOOP, errno.ENOTDIR, errno.EACCES, errno.EPERM}:
        return "unsafe"
    if exc.errno in {errno.ENOSYS, getattr(errno, "ENOTSUP", -1), getattr(errno, "EOPNOTSUPP", -1)}:
        return "unavailable"
    return "io"


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_uid, stat.S_IMODE(value.st_mode)


def _same_directory(value: os.stat_result, expected: tuple[int, int, int, int]) -> bool:
    return stat.S_ISDIR(value.st_mode) and _stat_identity(value) == expected


def _same_edge(value: os.stat_result, expected: tuple[int, int, int, int]) -> bool:
    return stat.S_ISDIR(value.st_mode) and _stat_identity(value) == expected


def _directory_stat(fd: int) -> os.stat_result:
    try:
        value = os.fstat(fd)
    except OSError as exc:
        _fail(_error_code(exc))
    if not stat.S_ISDIR(value.st_mode):
        _fail("unsafe")
    return value


def _stat_at(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except (TypeError, NotImplementedError):
        _fail("unavailable")
    except OSError as exc:
        _fail(_error_code(exc))


def _open_directory_at(parent_fd: int, name: str) -> int:
    try:
        return os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except (TypeError, NotImplementedError):
        _fail("unavailable")
    except OSError as exc:
        _fail(_error_code(exc))


def _require_platform() -> None:
    if os.name != "posix" or fcntl is None:
        _fail("unavailable")
    if any(not isinstance(getattr(os, flag, None), int) for flag in ("O_DIRECTORY", "O_NOFOLLOW")):
        _fail("unavailable")
    if not all(
        isinstance(getattr(fcntl, flag, None), int)
        for flag in ("LOCK_EX", "LOCK_NB", "LOCK_UN")
    ) or not callable(getattr(fcntl, "flock", None)):
        _fail("unavailable")
    for function in (os.open, os.stat, os.mkdir):
        if function not in getattr(os, "supports_dir_fd", set()):
            _fail("unavailable")
    if not callable(getattr(os, "fchmod", None)):
        _fail("unavailable")


def _configured_components(path: object) -> tuple[str, ...]:
    try:
        raw = os.fspath(path)
    except (TypeError, ValueError):
        _fail("invalid")
    if not isinstance(raw, str) or not raw.startswith("/") or "\x00" in raw:
        _fail("invalid")
    components = tuple(raw.split("/")[1:])
    if not components or any(not part or part in {".", ".."} for part in components):
        _fail("invalid")
    if "/" + "/".join(components) != raw:
        _fail("invalid")
    return components


def _is_safe_component(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and "\x00" not in value
    )


@dataclass(frozen=True, slots=True)
class _DirectoryBaseline:
    fd: int
    name: str | None
    parent_fd: int | None
    identity: tuple[int, int, int, int]
    edge: tuple[int, int, int, int] | None


@dataclass(frozen=True, slots=True)
class _LockBaseline:
    fd: int
    parent_fd: int
    name: str
    identity: tuple[int, int, int, int, int, int]


def _lock_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
        value.st_size,
    )


def _validate_lock_stat(value: os.stat_result) -> None:
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.geteuid()
        or value.st_nlink != 1
        or value.st_size != 0
        or stat.S_IMODE(value.st_mode) not in _ALLOWED_LOCK_MODES
    ):
        _fail("unsafe")


def _source_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _validate_source_stat(value: os.stat_result) -> None:
    mode = value.st_mode
    if (
        not stat.S_ISREG(mode)
        or value.st_uid != os.geteuid()
        or value.st_nlink != 1
        or not mode & 0o400
        or mode & (0o022 | 0o7000)
    ):
        _fail("unsafe")


def _require_source_read_platform() -> None:
    flags = ("O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC")
    if os.name != "posix" or any(
        not isinstance(getattr(os, flag, None), int) for flag in flags
    ):
        _fail("unavailable")
    if not callable(os.read):
        _fail("unavailable")


class CheckedSessionAuthority:
    """Factory-only lock capability bound to one backend, key, process, and thread."""

    __slots__ = (
        "_backend", "_key", "_storage_key", "_path", "_components", "_pid", "_thread",
        "_mutex", "_state", "_owned_fds", "_directories", "_execution_directory",
        "_locks",
    )

    def __init__(
        self,
        token: object,
        backend: JsonlSessionStore,
        key: str,
        storage_key: str,
        path: str,
        components: tuple[str, ...],
    ) -> None:
        if token is not _CONSTRUCTOR_TOKEN:
            raise TypeError("CheckedSessionAuthority instances are issued by open_checked_session")
        self._backend = backend
        self._key = key
        self._storage_key = storage_key
        self._path = path
        self._components = components
        self._pid = os.getpid()
        self._thread = threading.current_thread()
        self._mutex = threading.RLock()
        self._state: Literal["new", "active", "closed"] = "new"
        self._owned_fds: list[int] = []
        self._directories: list[_DirectoryBaseline] = []
        self._execution_directory: _DirectoryBaseline | None = None
        self._locks: list[_LockBaseline] = []

    def __copy__(self) -> CheckedSessionAuthority:
        raise TypeError("CheckedSessionAuthority cannot be copied")

    def __deepcopy__(self, memo: object) -> CheckedSessionAuthority:
        raise TypeError("CheckedSessionAuthority cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("CheckedSessionAuthority cannot be serialized")

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("CheckedSessionAuthority cannot be serialized")

    def _close_after_fork(self) -> None:
        # Only one thread survives fork; never touch the inherited mutex or LOCK_UN.
        descriptors = self._owned_fds
        self._owned_fds = []
        self._locks = []
        self._directories = []
        self._execution_directory = None
        self._state = "closed"
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _check_process(self) -> None:
        if os.getpid() != self._pid:
            self._close_after_fork()
            self._raise_public("invalid")

    def _check_thread(self) -> None:
        if threading.current_thread() is not self._thread:
            self._raise_public("invalid")

    def _capture_directories(self) -> None:
        try:
            descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except (TypeError, NotImplementedError):
            _fail("unavailable")
        except OSError as exc:
            _fail(_error_code(exc))
        self._owned_fds.append(descriptor)
        root_stat = _directory_stat(descriptor)
        self._directories.append(
            _DirectoryBaseline(descriptor, None, None, _stat_identity(root_stat), None)
        )
        parent_fd = descriptor
        for component in self._components:
            before = _stat_at(parent_fd, component)
            if before is None:
                _fail("missing")
            if not stat.S_ISDIR(before.st_mode):
                _fail("unsafe")
            child_fd = _open_directory_at(parent_fd, component)
            self._owned_fds.append(child_fd)
            child_stat = _directory_stat(child_fd)
            edge_identity = _stat_identity(before)
            if _stat_identity(child_stat) != edge_identity:
                _fail("changed")
            after = _stat_at(parent_fd, component)
            if after is None or not _same_edge(after, edge_identity):
                _fail("changed")
            self._directories.append(
                _DirectoryBaseline(
                    child_fd,
                    component,
                    parent_fd,
                    _stat_identity(child_stat),
                    edge_identity,
                )
            )
            parent_fd = child_fd
        if len(self._directories) < 2:
            _fail("invalid")

    def _validate_initial_directory_policy(self) -> None:
        if len(self._directories) < 3:
            _fail("unsafe")
        effective_uid = os.geteuid()
        namespace = self._directories[-1].identity
        migration_parent = self._directories[-2].identity
        for identity in (namespace, migration_parent):
            owner, mode = identity[2], identity[3]
            if (
                owner != effective_uid
                or mode & 0o500 != 0o500
                or mode & (0o022 | 0o7000)
            ):
                _fail("unsafe")
        for baseline in self._directories[:-2]:
            owner, mode = baseline.identity[2], baseline.identity[3]
            sticky_root = owner == 0 and bool(mode & stat.S_ISVTX)
            if owner not in {effective_uid, 0}:
                _fail("unsafe")
            if mode & 0o022 and not sticky_root:
                _fail("unsafe")

    def _validate_directories(self) -> None:
        if not self._directories:
            _fail("closed")
        try:
            current_path = os.fspath(self._backend.sessions_dir)
        except (AttributeError, TypeError, ValueError):
            _fail("changed")
        if current_path != self._path:
            _fail("changed")
        for baseline in self._directories:
            current = _directory_stat(baseline.fd)
            if not _same_directory(current, baseline.identity):
                _fail("changed")
            if baseline.parent_fd is not None and baseline.name is not None:
                edge = _stat_at(baseline.parent_fd, baseline.name)
                if edge is None or baseline.edge is None or not _same_edge(edge, baseline.edge):
                    _fail("changed")
        root_fd = self._directories[-1].fd
        try:
            backend_path_stat = os.stat(self._path, follow_symlinks=False)
        except FileNotFoundError:
            _fail("changed")
        except (TypeError, NotImplementedError):
            _fail("unavailable")
        except OSError as exc:
            _fail(_error_code(exc))
        if not _same_directory(backend_path_stat, self._directories[-1].identity):
            _fail("changed")
        if not _same_directory(_directory_stat(root_fd), self._directories[-1].identity):
            _fail("changed")

    def _open_execution_directory(self) -> None:
        namespace_fd = self._directories[-1].fd
        before = _stat_at(namespace_fd, _EXECUTION_DIRECTORY)
        if before is None:
            try:
                os.mkdir(_EXECUTION_DIRECTORY, 0o700, dir_fd=namespace_fd)
            except FileExistsError:
                # A concurrent creator is adopted only through a fresh no-follow baseline.
                pass
            except (TypeError, NotImplementedError):
                _fail("unavailable")
            except OSError as exc:
                _fail(_error_code(exc))
        else:
            if not stat.S_ISDIR(before.st_mode):
                _fail("unsafe")
        fresh = _stat_at(namespace_fd, _EXECUTION_DIRECTORY)
        if fresh is None:
            _fail("missing")
        if not stat.S_ISDIR(fresh.st_mode):
            _fail("unsafe")
        descriptor = _open_directory_at(namespace_fd, _EXECUTION_DIRECTORY)
        self._owned_fds.append(descriptor)
        opened = _directory_stat(descriptor)
        identity = _stat_identity(opened)
        after = _stat_at(namespace_fd, _EXECUTION_DIRECTORY)
        if after is None or not _same_edge(after, identity) or identity != _stat_identity(fresh):
            _fail("changed")
        if opened.st_uid != os.geteuid() or stat.S_IMODE(opened.st_mode) != 0o700:
            _fail("unsafe")
        self._execution_directory = _DirectoryBaseline(
            descriptor,
            _EXECUTION_DIRECTORY,
            namespace_fd,
            identity,
            identity,
        )

    def _validate_lock(self, baseline: _LockBaseline) -> None:
        try:
            opened = os.fstat(baseline.fd)
        except OSError as exc:
            _fail(_error_code(exc))
        _validate_lock_stat(opened)
        if _lock_identity(opened) != baseline.identity:
            _fail("changed")
        current = _stat_at(baseline.parent_fd, baseline.name)
        if current is None:
            _fail("changed")
        _validate_lock_stat(current)
        if _lock_identity(current) != baseline.identity:
            _fail("changed")

    def _validate_all(self) -> None:
        self._validate_directories()
        if self._execution_directory is not None:
            baseline = self._execution_directory
            current = _directory_stat(baseline.fd)
            if not _same_directory(current, baseline.identity):
                _fail("changed")
            edge = _stat_at(baseline.parent_fd, baseline.name or "")
            if edge is None or baseline.edge is None or not _same_edge(edge, baseline.edge):
                _fail("changed")
            if current.st_uid != os.geteuid() or stat.S_IMODE(current.st_mode) != 0o700:
                _fail("unsafe")
        for lock in self._locks:
            self._validate_lock(lock)

    def _open_lock(self, parent_fd: int, name: str) -> tuple[int, bool, os.stat_result]:
        for _attempt in range(3):
            before = _stat_at(parent_fd, name)
            if before is None:
                try:
                    descriptor = os.open(
                        name,
                        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=parent_fd,
                    )
                except FileExistsError:
                    continue
                except (TypeError, NotImplementedError):
                    _fail("unavailable")
                except OSError as exc:
                    _fail(_error_code(exc))
                self._owned_fds.append(descriptor)
                try:
                    created = os.fstat(descriptor)
                except OSError as exc:
                    _fail(_error_code(exc))
                if (
                    not stat.S_ISREG(created.st_mode)
                    or created.st_uid != os.geteuid()
                    or created.st_nlink != 1
                    or created.st_size != 0
                    or stat.S_IMODE(created.st_mode) & ~0o600
                ):
                    _fail("unsafe")
                try:
                    os.fchmod(descriptor, 0o600)
                except (TypeError, NotImplementedError):
                    _fail("unavailable")
                except OSError as exc:
                    _fail(_error_code(exc))
                try:
                    created = os.fstat(descriptor)
                except OSError as exc:
                    _fail(_error_code(exc))
                if (
                    not stat.S_ISREG(created.st_mode)
                    or created.st_uid != os.geteuid()
                    or created.st_nlink != 1
                    or created.st_size != 0
                    or stat.S_IMODE(created.st_mode) != 0o600
                ):
                    _fail("unsafe")
                path_value = _stat_at(parent_fd, name)
                if path_value is None or _lock_identity(path_value) != _lock_identity(created):
                    _fail("changed")
                return descriptor, True, created
            _validate_lock_stat(before)
            try:
                descriptor = os.open(
                    name,
                    os.O_RDWR | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
            except (TypeError, NotImplementedError):
                _fail("unavailable")
            except FileNotFoundError:
                _fail("changed")
            except OSError as exc:
                _fail(_error_code(exc))
            self._owned_fds.append(descriptor)
            try:
                opened = os.fstat(descriptor)
            except OSError as exc:
                _fail(_error_code(exc))
            _validate_lock_stat(opened)
            if _lock_identity(opened) != _lock_identity(before):
                _fail("changed")
            after = _stat_at(parent_fd, name)
            if after is None or _lock_identity(after) != _lock_identity(opened):
                _fail("changed")
            return descriptor, False, opened
        _fail("changed")

    def _acquire_lock(self, parent_fd: int, name: str) -> None:
        self._validate_all()
        descriptor, _created, opened = self._open_lock(parent_fd, name)
        _validate_lock_stat(opened)
        before_lock_path = _stat_at(parent_fd, name)
        if before_lock_path is None or _lock_identity(before_lock_path) != _lock_identity(opened):
            _fail("changed")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {
                errno.EACCES,
                errno.EAGAIN,
                getattr(errno, "EWOULDBLOCK", errno.EAGAIN),
            }:
                lock_error: ErrorCode | None = "busy"
            else:
                lock_error = _error_code(exc)
        else:
            lock_error = None
        try:
            post_lock = os.fstat(descriptor)
        except OSError as exc:
            _fail(_error_code(exc))
        _validate_lock_stat(post_lock)
        after_lock_path = _stat_at(parent_fd, name)
        if (
            after_lock_path is None
            or _lock_identity(after_lock_path) != _lock_identity(post_lock)
            or _lock_identity(post_lock) != _lock_identity(opened)
        ):
            _fail("changed")
        if lock_error is not None:
            _fail(lock_error)
        baseline = _LockBaseline(descriptor, parent_fd, name, _lock_identity(post_lock))
        self._locks.append(baseline)
        self._validate_all()

    def _acquire(self) -> None:
        _require_platform()
        self._capture_directories()
        self._validate_initial_directory_policy()
        self._validate_directories()
        namespace_fd = self._directories[-1].fd
        namespace_parent_fd = self._directories[-2].fd
        self._open_execution_directory()
        self._validate_all()
        execution_fd = self._execution_directory.fd if self._execution_directory else -1
        self._acquire_lock(namespace_parent_fd, _MIGRATION_LOCK)
        self._acquire_lock(execution_fd, f"{self._storage_key}.lock")
        self._acquire_lock(namespace_fd, _SESSION_FILES_LOCK)

    def _release_resources(self, *, unlock: bool) -> ErrorCode | None:
        descriptors = self._owned_fds
        locks = self._locks
        self._owned_fds = []
        self._locks = []
        self._directories = []
        self._execution_directory = None
        close_error = False
        if unlock and fcntl is not None:
            for baseline in reversed(locks):
                try:
                    fcntl.flock(baseline.fd, fcntl.LOCK_UN)
                except OSError:
                    close_error = True
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                close_error = True
        return "io" if close_error else None

    def _raise_public(self, code: ErrorCode) -> None:
        # Called only after internal exception handling has ended, so public
        # failures carry neither a cause nor a live exception context.
        raise CheckedSessionAuthorityError(code) from None

    def __enter__(self) -> CheckedSessionAuthority:
        self._check_process()
        self._check_thread()
        failure: ErrorCode | None = None
        try:
            with self._mutex:
                if self._state != "new":
                    failure = "closed" if self._state == "closed" else "invalid"
                else:
                    try:
                        self._acquire()
                    except _FailureError as exc:
                        failure = exc.code
                    except OSError as exc:
                        failure = _error_code(exc)
                    if failure is None:
                        self._state = "active"
                    else:
                        self._release_resources(unlock=True)
                        self._state = "closed"
        except _FailureError as exc:
            failure = exc.code
            if self._state == "new":
                self._release_resources(unlock=True)
                self._state = "closed"
        if failure is not None:
            self._raise_public(failure)
        return self

    def require_current(self) -> None:
        self._check_process()
        self._check_thread()
        failure: ErrorCode | None = None
        try:
            with self._mutex:
                if self._state != "active":
                    _fail("closed" if self._state == "closed" else "invalid")
                self._validate_all()
        except _FailureError as exc:
            failure = exc.code
        if failure is not None:
            self._raise_public(failure)

    def read_source_bytes(self) -> bytes:
        """Return a bounded, exact byte snapshot of this authority's JSONL source."""
        self._check_process()
        self._check_thread()
        if self._state != "active":
            self._raise_public("closed")
        self.require_current()

        failure: ErrorCode | None = None
        result: bytes | None = None
        descriptor: int | None = None
        try:
            with self._mutex:
                if self._state != "active":
                    _fail("closed")
                self._validate_all()
                _require_source_read_platform()

                source_name = f"{self._storage_key}.jsonl"
                namespace_fd = self._directories[-1].fd
                before = _stat_at(namespace_fd, source_name)
                if before is None:
                    _fail("missing")
                _validate_source_stat(before)
                baseline = _source_signature(before)
                if before.st_size > _MAX_SOURCE_BYTES:
                    _fail("limit")

                try:
                    descriptor = os.open(
                        source_name,
                        os.O_RDONLY
                        | os.O_NOFOLLOW
                        | os.O_NONBLOCK
                        | os.O_CLOEXEC,
                        dir_fd=namespace_fd,
                    )
                except FileNotFoundError:
                    _fail("changed")
                except (TypeError, NotImplementedError):
                    _fail("unavailable")
                except OSError as exc:
                    _fail(_error_code(exc))

                try:
                    opened = os.fstat(descriptor)
                except OSError as exc:
                    _fail(_error_code(exc))
                if _source_signature(opened) != baseline:
                    _fail("changed")
                current = _stat_at(namespace_fd, source_name)
                if current is None or _source_signature(current) != baseline:
                    _fail("changed")

                self._validate_all()
                data = bytearray()
                while True:
                    self._validate_all()
                    try:
                        opened = os.fstat(descriptor)
                    except OSError as exc:
                        _fail(_error_code(exc))
                    current = _stat_at(namespace_fd, source_name)
                    if (
                        _source_signature(opened) != baseline
                        or current is None
                        or _source_signature(current) != baseline
                    ):
                        _fail("changed")

                    request_size = min(
                        _SOURCE_READ_CHUNK_SIZE,
                        _MAX_SOURCE_BYTES + 1 - len(data),
                    )
                    try:
                        chunk = os.read(descriptor, request_size)
                    except OSError as exc:
                        if exc.errno == errno.EINTR:
                            continue
                        _fail(_error_code(exc))
                    if not chunk:
                        if len(data) != before.st_size:
                            _fail("changed")
                        break
                    if len(chunk) > request_size:
                        _fail("io")
                    data.extend(chunk)
                    if len(data) > _MAX_SOURCE_BYTES:
                        _fail("limit")

                    self._validate_all()
                    try:
                        opened = os.fstat(descriptor)
                    except OSError as exc:
                        _fail(_error_code(exc))
                    current = _stat_at(namespace_fd, source_name)
                    if (
                        _source_signature(opened) != baseline
                        or current is None
                        or _source_signature(current) != baseline
                    ):
                        _fail("changed")

                self._validate_all()
                try:
                    opened = os.fstat(descriptor)
                except OSError as exc:
                    _fail(_error_code(exc))
                current = _stat_at(namespace_fd, source_name)
                if (
                    _source_signature(opened) != baseline
                    or current is None
                    or _source_signature(current) != baseline
                ):
                    _fail("changed")
                self._validate_all()
                result = bytes(data)
        except _FailureError as exc:
            failure = exc.code
        except OSError as exc:
            failure = _error_code(exc)
        finally:
            if descriptor is not None:
                owned_descriptor = descriptor
                descriptor = None
                try:
                    os.close(owned_descriptor)
                except OSError as exc:
                    if failure is None:
                        failure = _error_code(exc)

        if failure is not None:
            self._raise_public(failure)
        if result is None:
            self._raise_public("io")
        return result

    def close(self) -> None:
        if os.getpid() != self._pid:
            self._close_after_fork()
            return
        self._check_thread()
        failure: ErrorCode | None = None
        try:
            with self._mutex:
                if self._state == "closed":
                    return
                close_error = self._release_resources(unlock=True)
                self._state = "closed"
                failure = close_error
        except _FailureError as exc:
            failure = exc.code
        if failure is not None:
            self._raise_public(failure)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if os.getpid() != self._pid:
            self.close()
            return False
        self._check_thread()
        failure: ErrorCode | None = None
        try:
            with self._mutex:
                if self._state == "active" and exc_type is None:
                    self._validate_all()
        except _FailureError as validation_error:
            failure = validation_error.code
        close_error = self._release_resources(unlock=True)
        self._state = "closed"
        if failure is None:
            failure = close_error
        if failure is not None and exc_type is None:
            self._raise_public(failure)
        return False


def open_checked_session(backend: JsonlSessionStore, key: str) -> CheckedSessionAuthority:
    """Issue a context-managed, lock-only authority for ``backend`` and ``key``.

    The caller must already hold the memory authority.  The returned object
    acquires migration, execution, then session-file flock locks on entry.
    """
    from nanobot.session.manager import JsonlSessionStore as StoreType

    if type(backend) is not StoreType or not isinstance(key, str) or not key:
        raise CheckedSessionAuthorityError("invalid")
    failure: ErrorCode | None = None
    storage_key = ""
    path = ""
    components: tuple[str, ...] = ()
    try:
        storage_key = backend.storage_key(key)
        decoded = backend.decode_storage_key(storage_key)
        path = os.fspath(backend.sessions_dir)
        if (
            not _is_safe_component(storage_key)
            or decoded != key
            or backend.storage_key(decoded) != storage_key
        ):
            failure = "invalid"
        else:
            components = _configured_components(path)
    except _FailureError as exc:
        failure = exc.code
    except (AttributeError, TypeError, ValueError, UnicodeError):
        failure = "invalid"
    if failure is not None:
        raise CheckedSessionAuthorityError(failure)
    return CheckedSessionAuthority(
        _CONSTRUCTOR_TOKEN, backend, key, storage_key, path, components
    )
