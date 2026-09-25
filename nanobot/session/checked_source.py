"""Checked lock-only authority for one current JSONL session namespace.

This dormant migration capability pins the configured directory ancestry and
holds the repository's migration, execution, and session-file flock locks.
Its recovery variant reads one detached candidate under an active workspace-memory
lease; an explicitly issued replacement variant may replace it once.
"""

from __future__ import annotations

import errno
import os
import re
import secrets
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

try:  # pragma: no cover - platform qualification covers unavailable fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from nanobot.agent.memory_recovery_storage import WorkspaceMemoryOperation
    from nanobot.session.manager import JsonlSessionStore
    from nanobot.session.recovery_locator import RecoveryCandidate


ErrorCode = Literal[
    "invalid", "unavailable", "unsafe", "changed", "busy", "missing", "closed", "io",
    "limit", "uncertain",
]
_ALLOWED_LOCK_MODES = frozenset({0o600, 0o604, 0o640, 0o644})
_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_SOURCE_READ_CHUNK_SIZE = 64 * 1024
_TEMP_CREATE_ATTEMPTS = 8
_MIGRATION_LOCK = ".workspace-migration.lock"
_SESSION_FILES_LOCK = ".session-files.lock"
_EXECUTION_DIRECTORY = ".execution-locks"
_RECOVERY_NAMESPACE_RE = re.compile(r"[0-9a-f]{32}\Z")
_RECOVERY_MARKER_MAX_BYTES = 4096
_CONSTRUCTOR_TOKEN = object()


class CheckedSessionAuthorityError(RuntimeError):
    """Content-free failure raised by the checked lock capability."""

    __slots__ = ("code",)

    def __init__(self, code: ErrorCode) -> None:
        if code not in {
            "invalid", "unavailable", "unsafe", "changed", "busy", "missing", "closed", "io",
            "limit", "uncertain",
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


def _temporary_name() -> str:
    return f".checked-source-{secrets.token_hex(16)}.tmp"


def _require_source_read_platform() -> None:
    flags = ("O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC")
    if os.name != "posix" or any(
        not isinstance(getattr(os, flag, None), int) for flag in flags
    ):
        _fail("unavailable")
    if not callable(os.read):
        _fail("unavailable")


def _recovery_directory_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid


def _recovery_file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_uid,
        value.st_gid,
    )


def _require_recovery_directory(value: os.stat_result) -> None:
    permissions = stat.S_IMODE(value.st_mode)
    if (
        not stat.S_ISDIR(value.st_mode)
        or value.st_ino <= 0
        or value.st_uid != os.geteuid()
        or permissions & 0o700 != 0o700
        or permissions & (0o022 | 0o7000)
    ):
        _fail("unsafe")


def _require_recovery_file(value: os.stat_result) -> None:
    _validate_source_stat(value)


def _open_recovery_directory(parent_fd: int, name: str) -> tuple[int, os.stat_result]:
    before = _stat_at(parent_fd, name)
    if before is None:
        _fail("changed")
    _require_recovery_directory(before)
    descriptor = _open_directory_at(parent_fd, name)
    try:
        opened = _directory_stat(descriptor)
        _require_recovery_directory(opened)
        after = _stat_at(parent_fd, name)
        if (
            after is None
            or _recovery_directory_identity(before) != _recovery_directory_identity(opened)
            or _recovery_directory_identity(after) != _recovery_directory_identity(opened)
        ):
            _fail("changed")
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _read_recovery_marker(
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, int, int, int, int, int, int, int, int] | None,
) -> bytes:
    before = _stat_at(parent_fd, name)
    if before is None:
        _fail("changed")
    _require_recovery_file(before)
    baseline = _recovery_file_identity(before)
    if expected_identity is not None and baseline != expected_identity:
        _fail("changed")
    if before.st_size > _RECOVERY_MARKER_MAX_BYTES:
        _fail("changed")

    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            _fail("changed")
        except (TypeError, NotImplementedError):
            _fail("unavailable")
        except OSError as exc:
            _fail(_error_code(exc))

        opened = os.fstat(descriptor)
        _require_recovery_file(opened)
        if _recovery_file_identity(opened) != baseline:
            _fail("changed")
        chunks: list[bytes] = []
        remaining = _RECOVERY_MARKER_MAX_BYTES + 1
        while remaining:
            try:
                chunk = os.read(descriptor, min(remaining, _SOURCE_READ_CHUNK_SIZE))
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                _fail(_error_code(exc))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        current = _stat_at(parent_fd, name)
        if (
            len(payload) > _RECOVERY_MARKER_MAX_BYTES
            or len(payload) != opened.st_size
            or _recovery_file_identity(after) != baseline
            or current is None
            or _recovery_file_identity(current) != baseline
        ):
            _fail("changed")
        return payload
    except OSError as exc:
        _fail(_error_code(exc))
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _valid_recovery_binding(payload: bytes) -> bool:
    if not payload or len(payload) > _RECOVERY_MARKER_MAX_BYTES or not payload.endswith(b"\n"):
        return False
    try:
        text = payload[:-1].decode("utf-8")
        path = os.path.normpath(text)
    except (UnicodeDecodeError, TypeError, ValueError):
        return False
    return text.startswith("/") and path == text and ".." not in text.split("/")


def _open_absolute_recovery_directory(
    path: str,
) -> tuple[list[_DirectoryBaseline], list[int]]:
    components = _configured_components(path)
    owned: list[int] = []
    baselines: list[_DirectoryBaseline] = []
    try:
        try:
            descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except (TypeError, NotImplementedError):
            _fail("unavailable")
        except OSError as exc:
            _fail(_error_code(exc))
        owned.append(descriptor)
        info = _directory_stat(descriptor)
        baselines.append(_DirectoryBaseline(descriptor, None, None, _stat_identity(info), None))
        parent_fd = descriptor
        for component in components:
            before = _stat_at(parent_fd, component)
            if before is None or not stat.S_ISDIR(before.st_mode):
                _fail("changed")
            child_fd = _open_directory_at(parent_fd, component)
            owned.append(child_fd)
            opened = _directory_stat(child_fd)
            edge = _stat_identity(before)
            after = _stat_at(parent_fd, component)
            if after is None or _stat_identity(opened) != edge or not _same_edge(after, edge):
                _fail("changed")
            baselines.append(
                _DirectoryBaseline(child_fd, component, parent_fd, _stat_identity(opened), edge)
            )
            parent_fd = child_fd
        return baselines, owned
    except BaseException:
        for descriptor in reversed(owned):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _revalidate_recovery_path(baselines: list[_DirectoryBaseline]) -> None:
    for baseline in baselines:
        if not _same_directory(_directory_stat(baseline.fd), baseline.identity):
            _fail("changed")
        if baseline.parent_fd is not None and baseline.name is not None:
            edge = _stat_at(baseline.parent_fd, baseline.name)
            if edge is None or baseline.edge is None or not _same_edge(edge, baseline.edge):
                _fail("changed")


def _require_source_replace_platform() -> None:
    flags = ("O_CREAT", "O_EXCL", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
    if os.name != "posix" or any(
        not isinstance(getattr(os, flag, None), int) for flag in flags
    ):
        _fail("unavailable")
    if not all(
        callable(getattr(os, function, None))
        for function in ("fchmod", "fsync", "lseek", "read", "replace", "write")
    ):
        _fail("unavailable")


class CheckedSessionAuthority:
    """Factory-only checked lock and source capability bound to one caller."""

    __slots__ = (
        "_backend", "_key", "_storage_key", "_path", "_components", "_pid", "_thread",
        "_mutex", "_state", "_owned_fds", "_directories", "_execution_directory",
        "_locks", "_source_evidence", "_replacement_used", "_memory_operation",
        "_recovery_candidate", "_recovery_only", "_recovery_namespace",
        "_recovery_replace", "_expected_recovery_source_identity",
    )

    def __init__(
        self,
        token: object,
        backend: JsonlSessionStore | None,
        key: str,
        storage_key: str,
        path: str,
        components: tuple[str, ...],
        *,
        memory_operation: WorkspaceMemoryOperation | None = None,
        recovery_candidate: RecoveryCandidate | None = None,
        recovery_replace: bool = False,
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
        self._source_evidence: tuple[
            bytes, tuple[int, int, int, int, int, int, int, int, int]
        ] | None = None
        self._replacement_used = False
        self._memory_operation = memory_operation
        self._recovery_candidate = recovery_candidate
        self._recovery_only = recovery_candidate is not None
        self._recovery_namespace: _DirectoryBaseline | None = None
        self._recovery_replace = recovery_replace
        self._expected_recovery_source_identity = (
            recovery_candidate.session_file_identity if recovery_candidate is not None else None
        )

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
        self._recovery_namespace = None
        self._state = "closed"
        self._source_evidence = None
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
        if self._recovery_only:
            current_path = self._path
        else:
            try:
                if self._backend is None:
                    _fail("changed")
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
        if self._recovery_only:
            if self._recovery_namespace is None:
                _fail("closed")
            namespace_fd = self._recovery_namespace.fd
        else:
            namespace_fd = self._directories[-1].fd
        before = _stat_at(namespace_fd, _EXECUTION_DIRECTORY)
        if before is None:
            if self._recovery_only:
                self._validate_recovery_candidate()
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

    def _require_memory_operation_current(self) -> tuple[int, int]:
        operation = self._memory_operation
        candidate = self._recovery_candidate
        if operation is None or candidate is None:
            _fail("invalid")
        try:
            operation.require_current()
            identity = operation.workspace_identity
        except Exception:
            _fail("changed")
        expected = candidate.workspace_identity
        if len(expected) != 5 or identity != expected[:2]:
            _fail("changed")
        return identity

    def _validate_recovery_candidate(self) -> None:
        candidate = self._recovery_candidate
        if candidate is None:
            return
        lease_identity = self._require_memory_operation_current()
        _require_source_read_platform()
        if os.listdir not in getattr(os, "supports_fd", set()):
            _fail("unavailable")
        workspace_baselines: list[_DirectoryBaseline] = []
        workspace_fds: list[int] = []
        state_fd: int | None = None
        target_namespace_fd: int | None = None
        target_namespace_identity: tuple[int, int, int, int] | None = None
        try:
            workspace_baselines, workspace_fds = _open_absolute_recovery_directory(
                os.fspath(candidate.workspace_path)
            )
            workspace_fd = workspace_baselines[-1].fd
            workspace_stat = _directory_stat(workspace_fd)
            _require_recovery_directory(workspace_stat)
            if (
                _recovery_directory_identity(workspace_stat) != candidate.workspace_identity
                or (workspace_stat.st_dev, workspace_stat.st_ino) != lease_identity
            ):
                _fail("changed")

            state_fd, state_stat = _open_recovery_directory(workspace_fd, ".nanobot")
            state_payload = _read_recovery_marker(
                state_fd, "workspace-id", candidate.workspace_marker_identity
            )
            expected_id = candidate.workspace_id.encode("ascii")
            if state_payload not in (expected_id, expected_id + b"\n"):
                _fail("changed")

            root_fd = self._directories[-1].fd
            root_stat = _directory_stat(root_fd)
            _require_recovery_directory(root_stat)
            if _recovery_directory_identity(root_stat) != candidate.sessions_root_identity:
                _fail("changed")
            expected_binding = f"{candidate.workspace_path}\n".encode("utf-8")
            root_scan_identity = _recovery_file_identity(root_stat)
            try:
                namespace_names = os.listdir(root_fd)
            except (TypeError, NotImplementedError):
                _fail("unavailable")
            except OSError as exc:
                _fail(_error_code(exc))
            matching_bindings = 0
            target_name = candidate.workspace_id
            source_name = f"{self._storage_key}.jsonl"
            for name in sorted(namespace_names):
                if _RECOVERY_NAMESPACE_RE.fullmatch(name) is None:
                    continue
                namespace_fd, namespace_stat = _open_recovery_directory(root_fd, name)
                try:
                    marker_entry = _stat_at(namespace_fd, ".workspace")
                    if marker_entry is None:
                        if name == target_name:
                            _fail("changed")
                        continue
                    expected_marker = (
                        candidate.namespace_marker_identity if name == target_name else None
                    )
                    payload = _read_recovery_marker(
                        namespace_fd, ".workspace", expected_marker
                    )
                    if not _valid_recovery_binding(payload):
                        _fail("changed")
                    if payload == expected_binding:
                        matching_bindings += 1
                        if name != target_name:
                            _fail("changed")
                    if name == target_name:
                        if (
                            _recovery_directory_identity(namespace_stat)
                            != candidate.namespace_identity
                            or payload != expected_binding
                        ):
                            _fail("changed")
                        target_namespace_fd = namespace_fd
                        target_namespace_identity = _stat_identity(namespace_stat)
                        namespace_fd = -1
                        source_entry = _stat_at(target_namespace_fd, source_name)
                        if source_entry is None:
                            _fail("changed")
                        _validate_source_stat(source_entry)
                        if (
                            _recovery_file_identity(source_entry)
                            != self._expected_recovery_source_identity
                        ):
                            _fail("changed")
                finally:
                    if namespace_fd >= 0:
                        try:
                            os.close(namespace_fd)
                        except OSError:
                            pass
            root_after_scan = _directory_stat(root_fd)
            root_path_after_scan = _stat_at(
                self._directories[-2].fd, self._components[-1]
            )
            if (
                _recovery_file_identity(root_after_scan) != root_scan_identity
                or root_path_after_scan is None
                or _recovery_file_identity(root_path_after_scan) != root_scan_identity
            ):
                _fail("changed")
            if matching_bindings != 1 or target_namespace_fd is None:
                _fail("changed")

            _revalidate_recovery_path(workspace_baselines)
            workspace_final = _directory_stat(workspace_fd)
            state_current = _stat_at(workspace_fd, ".nanobot")
            state_opened = _directory_stat(state_fd)
            if (
                _recovery_directory_identity(workspace_final) != candidate.workspace_identity
                or state_current is None
                or _recovery_directory_identity(state_current)
                != _recovery_directory_identity(state_stat)
                or _recovery_directory_identity(state_opened)
                != _recovery_directory_identity(state_stat)
            ):
                _fail("changed")
            root_final = _directory_stat(root_fd)
            root_current = _stat_at(self._directories[-2].fd, self._components[-1])
            target_current = _stat_at(root_fd, target_name)
            target_opened = (
                _directory_stat(target_namespace_fd)
                if target_namespace_fd is not None
                else None
            )
            if (
                _recovery_directory_identity(root_final) != candidate.sessions_root_identity
                or root_current is None
                or _recovery_directory_identity(root_current) != candidate.sessions_root_identity
                or target_current is None
                or _recovery_directory_identity(target_current) != candidate.namespace_identity
                or target_namespace_identity is None
                or target_opened is None
                or _recovery_directory_identity(target_opened) != candidate.namespace_identity
            ):
                _fail("changed")
            pinned = self._recovery_namespace
            if pinned is None:
                self._owned_fds.append(target_namespace_fd)
                self._recovery_namespace = _DirectoryBaseline(
                    target_namespace_fd,
                    target_name,
                    root_fd,
                    target_namespace_identity,
                    target_namespace_identity,
                )
                target_namespace_fd = None
            else:
                pinned_edge = _stat_at(pinned.parent_fd or root_fd, pinned.name or "")
                if (
                    not _same_directory(_directory_stat(pinned.fd), pinned.identity)
                    or _recovery_directory_identity(_directory_stat(pinned.fd))
                    != candidate.namespace_identity
                    or pinned.identity != target_namespace_identity
                    or pinned_edge is None
                    or pinned.edge is None
                    or not _same_edge(pinned_edge, pinned.edge)
                    or _recovery_directory_identity(pinned_edge)
                    != candidate.namespace_identity
                ):
                    _fail("changed")
            namespace_fd = (
                self._recovery_namespace.fd
                if self._recovery_namespace is not None
                else target_namespace_fd
            )
            if namespace_fd is None:
                _fail("changed")
            final_workspace_marker = _read_recovery_marker(
                state_fd, "workspace-id", candidate.workspace_marker_identity
            )
            final_namespace_marker = _read_recovery_marker(
                namespace_fd, ".workspace", candidate.namespace_marker_identity
            )
            final_source = _stat_at(namespace_fd, source_name)
            if (
                final_workspace_marker not in (expected_id, expected_id + b"\n")
                or final_namespace_marker != expected_binding
                or final_source is None
                or _recovery_file_identity(final_source) != self._expected_recovery_source_identity
            ):
                _fail("changed")
            self._require_memory_operation_current()
        except OSError as exc:
            _fail(_error_code(exc))
        finally:
            if target_namespace_fd is not None:
                try:
                    os.close(target_namespace_fd)
                except OSError:
                    pass
            if state_fd is not None:
                try:
                    os.close(state_fd)
                except OSError:
                    pass
            for descriptor in reversed(workspace_fds):
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def _validate_all(self) -> None:
        self._validate_directories()
        if self._recovery_only:
            self._validate_recovery_candidate()
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
            if self._recovery_only:
                self._validate_recovery_candidate()
            _fail(lock_error)
        baseline = _LockBaseline(descriptor, parent_fd, name, _lock_identity(post_lock))
        self._locks.append(baseline)
        self._validate_all()

    def _acquire(self) -> None:
        _require_platform()
        self._capture_directories()
        self._validate_initial_directory_policy()
        self._validate_all()
        namespace_fd = self._directories[-1].fd
        namespace_parent_fd = self._directories[-2].fd
        self._open_execution_directory()
        self._validate_all()
        execution_fd = self._execution_directory.fd if self._execution_directory else -1
        self._acquire_lock(namespace_parent_fd, _MIGRATION_LOCK)
        self._acquire_lock(execution_fd, f"{self._storage_key}.lock")
        session_files_fd = (
            self._recovery_namespace.fd
            if self._recovery_only and self._recovery_namespace is not None
            else namespace_fd
        )
        self._acquire_lock(session_files_fd, _SESSION_FILES_LOCK)

    def _release_resources(self, *, unlock: bool) -> ErrorCode | None:
        descriptors = self._owned_fds
        locks = self._locks
        self._owned_fds = []
        self._locks = []
        self._directories = []
        self._execution_directory = None
        self._recovery_namespace = None
        self._source_evidence = None
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

        with self._mutex:
            self._source_evidence = None
        failure: ErrorCode | None = None
        result: bytes | None = None
        result_signature: tuple[int, int, int, int, int, int, int, int, int] | None = None
        descriptor: int | None = None
        try:
            with self._mutex:
                if self._state != "active":
                    _fail("closed")
                self._validate_all()
                _require_source_read_platform()

                source_name = f"{self._storage_key}.jsonl"
                if self._recovery_only:
                    if self._recovery_namespace is None:
                        _fail("closed")
                    namespace_fd = self._recovery_namespace.fd
                else:
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
                result_signature = baseline
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
            self._source_evidence = None
            self._raise_public(failure)
        if result is None or result_signature is None:
            self._raise_public("io")
        self._source_evidence = (result, result_signature)
        return result

    def verify_recovery_source_bytes(self, expected_bytes: bytes) -> None:
        """Durably verify exact source bytes against this recovery candidate."""
        self._check_process()
        self._check_thread()
        if not self._recovery_only:
            self._raise_public("invalid")
        if self._state != "active":
            self._raise_public("closed")
        if type(expected_bytes) is not bytes:
            self._raise_public("invalid")

        failure: ErrorCode | None = None
        descriptor: int | None = None
        try:
            with self._mutex:
                if self._state != "active":
                    _fail("closed")
                if self._replacement_used:
                    _fail("invalid")
                self._validate_all()
                _require_source_read_platform()
                if not callable(getattr(os, "fsync", None)):
                    _fail("unavailable")
                if len(expected_bytes) > _MAX_SOURCE_BYTES:
                    _fail("limit")

                namespace = self._recovery_namespace
                if namespace is None:
                    _fail("closed")
                namespace_fd = namespace.fd
                source_name = f"{self._storage_key}.jsonl"
                before = _stat_at(namespace_fd, source_name)
                if before is None:
                    _fail("missing")
                _require_recovery_file(before)
                baseline = _recovery_file_identity(before)
                if baseline != self._expected_recovery_source_identity:
                    _fail("changed")
                if before.st_size > _MAX_SOURCE_BYTES:
                    _fail("limit")
                if before.st_size != len(expected_bytes):
                    _fail("changed")

                try:
                    descriptor = os.open(
                        source_name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                        dir_fd=namespace_fd,
                    )
                except FileNotFoundError:
                    _fail("changed")
                except (TypeError, NotImplementedError):
                    _fail("unavailable")
                except OSError as exc:
                    _fail(_error_code(exc))

                opened = os.fstat(descriptor)
                _require_recovery_file(opened)
                current = _stat_at(namespace_fd, source_name)
                if (
                    _recovery_file_identity(opened) != baseline
                    or current is None
                    or _recovery_file_identity(current) != baseline
                ):
                    _fail("changed")

                self._validate_all()
                self._verify_temp_bytes(descriptor, expected_bytes)
                self._validate_all()
                opened = os.fstat(descriptor)
                current = _stat_at(namespace_fd, source_name)
                if (
                    _recovery_file_identity(opened) != baseline
                    or current is None
                    or _recovery_file_identity(current) != baseline
                ):
                    _fail("changed")

                try:
                    os.fsync(descriptor)
                except (TypeError, NotImplementedError):
                    _fail("unavailable")
                except OSError as exc:
                    _fail(_error_code(exc))
                try:
                    os.fsync(namespace_fd)
                except (TypeError, NotImplementedError):
                    _fail("unavailable")
                except OSError as exc:
                    if exc.errno == errno.EINVAL:
                        _fail("unavailable")
                    _fail(_error_code(exc))

                self._validate_all()
                opened = os.fstat(descriptor)
                current = _stat_at(namespace_fd, source_name)
                if (
                    _recovery_file_identity(opened) != baseline
                    or current is None
                    or _recovery_file_identity(current) != baseline
                ):
                    _fail("changed")
                self._verify_temp_bytes(descriptor, expected_bytes)
                opened = os.fstat(descriptor)
                current = _stat_at(namespace_fd, source_name)
                if (
                    _recovery_file_identity(opened) != baseline
                    or current is None
                    or _recovery_file_identity(current) != baseline
                ):
                    _fail("changed")
                self._validate_all()
        except _FailureError as exc:
            failure = exc.code
        except OSError as exc:
            failure = _error_code(exc)
        except Exception:
            failure = "io"
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    if failure is None:
                        failure = _error_code(exc)

        if failure is not None:
            self._raise_public(failure)

    def _detach_and_close(self, descriptor: int) -> None:
        try:
            self._owned_fds.remove(descriptor)
        except ValueError:
            return
        os.close(descriptor)

    def _check_temp_entry(self, descriptor: int, name: str, parent_fd: int) -> os.stat_result:
        try:
            opened = os.fstat(descriptor)
        except OSError as exc:
            _fail(_error_code(exc))
        _validate_source_stat(opened)
        if stat.S_IMODE(opened.st_mode) != 0o600:
            _fail("unsafe")
        current = _stat_at(parent_fd, name)
        if current is None or _source_signature(current) != _source_signature(opened):
            _fail("changed")
        return opened

    def _verify_temp_bytes(self, descriptor: int, expected: bytes) -> None:
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
        except OSError as exc:
            _fail(_error_code(exc))
        offset = 0
        while offset < len(expected):
            try:
                chunk = os.read(descriptor, min(_SOURCE_READ_CHUNK_SIZE, len(expected) - offset))
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                _fail(_error_code(exc))
            if not chunk or len(chunk) > len(expected) - offset:
                _fail("changed")
            if chunk != expected[offset : offset + len(chunk)]:
                _fail("changed")
            offset += len(chunk)
        while True:
            try:
                extra = os.read(descriptor, 1)
                break
            except OSError as exc:
                if exc.errno != errno.EINTR:
                    _fail(_error_code(exc))
        if extra:
            _fail("changed")

    def replace_source_bytes(self, new_bytes: bytes) -> None:
        """Replace a regular checked source, when this authority permits it."""
        self._replace_source_bytes(new_bytes, recovery=False)

    def replace_recovery_source_bytes(self, new_bytes: bytes) -> None:
        """Replace one previously read recovery source on explicit write authority."""
        self._replace_source_bytes(new_bytes, recovery=True)

    def _replace_source_bytes(self, new_bytes: bytes, *, recovery: bool) -> None:
        self._check_process()
        self._check_thread()
        if self._recovery_only != recovery or (recovery and not self._recovery_replace):
            self._raise_public("invalid")
        if self._state != "active":
            self._raise_public("closed")
        if type(new_bytes) is not bytes:
            self._raise_public("invalid")
        failure: ErrorCode | None = None
        descriptor: int | None = None
        replacement_consumed = False
        replace_attempted = False
        try:
            with self._mutex:
                if self._state != "active":
                    _fail("closed")
                evidence = self._source_evidence
                if evidence is None or self._replacement_used:
                    _fail("invalid")
                if len(new_bytes) > _MAX_SOURCE_BYTES:
                    _fail("limit")
                prior_bytes, prior_signature = evidence
                replacement_consumed = True
                self._source_evidence = None
                self._replacement_used = True
                self._validate_all()
                _require_source_replace_platform()
                source_name = f"{self._storage_key}.jsonl"
                if recovery:
                    if self._recovery_namespace is None:
                        _fail("closed")
                    namespace_fd = self._recovery_namespace.fd
                else:
                    namespace_fd = self._directories[-1].fd

                current_bytes = self.read_source_bytes()
                current_evidence = self._source_evidence
                self._source_evidence = None
                if (
                    current_evidence is None
                    or current_evidence[1] != prior_signature
                    or current_bytes != prior_bytes
                ):
                    _fail("changed")

                temp_name = ""
                for _attempt in range(_TEMP_CREATE_ATTEMPTS):
                    temp_name = _temporary_name()
                    try:
                        descriptor = os.open(
                            temp_name,
                            os.O_RDWR
                            | os.O_CREAT
                            | os.O_EXCL
                            | os.O_NOFOLLOW
                            | os.O_CLOEXEC
                            | os.O_NONBLOCK,
                            0o600,
                            dir_fd=namespace_fd,
                        )
                    except FileExistsError:
                        continue
                    except (TypeError, NotImplementedError):
                        _fail("unavailable")
                    except OSError as exc:
                        _fail(_error_code(exc))
                    self._owned_fds.append(descriptor)
                    break
                if descriptor is None:
                    _fail("changed")

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
                if stat.S_IMODE(created.st_mode) != 0o600:
                    try:
                        os.fchmod(descriptor, 0o600)
                    except (TypeError, NotImplementedError):
                        _fail("unavailable")
                    except OSError as exc:
                        _fail(_error_code(exc))
                self._check_temp_entry(descriptor, temp_name, namespace_fd)

                offset = 0
                while offset < len(new_bytes):
                    try:
                        written = os.write(descriptor, memoryview(new_bytes)[offset:])
                    except OSError as exc:
                        if exc.errno == errno.EINTR:
                            continue
                        _fail(_error_code(exc))
                    if written <= 0 or written > len(new_bytes) - offset:
                        _fail("io")
                    offset += written

                self._check_temp_entry(descriptor, temp_name, namespace_fd)
                self._verify_temp_bytes(descriptor, new_bytes)
                self._check_temp_entry(descriptor, temp_name, namespace_fd)
                try:
                    os.fsync(descriptor)
                except OSError as exc:
                    _fail(_error_code(exc))
                self._check_temp_entry(descriptor, temp_name, namespace_fd)

                current_bytes = self.read_source_bytes()
                current_evidence = self._source_evidence
                self._source_evidence = None
                if (
                    current_evidence is None
                    or current_evidence[1] != prior_signature
                    or current_bytes != prior_bytes
                ):
                    _fail("changed")
                self._validate_all()
                self._check_temp_entry(descriptor, temp_name, namespace_fd)

                replace_attempted = True
                try:
                    os.replace(
                        temp_name,
                        source_name,
                        src_dir_fd=namespace_fd,
                        dst_dir_fd=namespace_fd,
                    )
                except (TypeError, NotImplementedError):
                    _fail("uncertain")
                except OSError:
                    _fail("uncertain")

                if not recovery:
                    self._validate_all()
                replaced = self._check_temp_entry(descriptor, source_name, namespace_fd)
                if replaced.st_size != len(new_bytes):
                    _fail("changed")
                self._verify_temp_bytes(descriptor, new_bytes)
                replaced = self._check_temp_entry(descriptor, source_name, namespace_fd)
                if recovery:
                    self._expected_recovery_source_identity = _recovery_file_identity(replaced)
                self._validate_all()
                try:
                    os.fsync(namespace_fd)
                except OSError:
                    _fail("uncertain")
                self._validate_all()
                self._check_temp_entry(descriptor, source_name, namespace_fd)
        except _FailureError as exc:
            failure = "uncertain" if replace_attempted else exc.code
        except CheckedSessionAuthorityError as exc:
            if replace_attempted:
                failure = "uncertain"
            else:
                failure = "changed" if exc.code == "missing" else exc.code
        except OSError as exc:
            if replace_attempted:
                failure = "uncertain"
            else:
                failure = _error_code(exc)
                if failure == "missing":
                    failure = "changed"
        except Exception:
            failure = "uncertain" if replace_attempted else "io"
        finally:
            if replacement_consumed:
                self._source_evidence = None
            if descriptor is not None:
                owned_descriptor = descriptor
                descriptor = None
                try:
                    self._detach_and_close(owned_descriptor)
                except OSError as exc:
                    if failure is None:
                        failure = "uncertain" if replace_attempted else _error_code(exc)

        if failure is not None:
            self._raise_public(failure)

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


def open_checked_recovery_session_for_replace(
    operation: WorkspaceMemoryOperation,
    candidate: RecoveryCandidate,
) -> CheckedSessionAuthority:
    """Issue a one-use replacement authority for a checked recovery candidate."""
    return _open_checked_recovery_session(operation, candidate, replace=True)


def open_checked_recovery_session(
    operation: WorkspaceMemoryOperation,
    candidate: RecoveryCandidate,
) -> CheckedSessionAuthority:
    """Issue a read-only authority for one located candidate and active memory lease."""
    return _open_checked_recovery_session(operation, candidate, replace=False)


def _open_checked_recovery_session(
    operation: WorkspaceMemoryOperation,
    candidate: RecoveryCandidate,
    *,
    replace: bool,
) -> CheckedSessionAuthority:
    """Validate shared candidate issuance before any lock side effects.

    The candidate is validated before any lock directory or lock file is created.
    The caller must keep ``operation`` active for the entire checked-session scope.
    """
    from nanobot.agent.memory_recovery_storage import WorkspaceMemoryOperation as OperationType
    from nanobot.session.manager import JsonlSessionStore as StoreType
    from nanobot.session.recovery_locator import RecoveryCandidate as CandidateType

    failure = False
    storage_key = ""
    path = ""
    components: tuple[str, ...] = ()
    try:
        if type(operation) is not OperationType or type(candidate) is not CandidateType:
            raise ValueError
        if (
            not isinstance(candidate.workspace_path, Path)
            or not isinstance(candidate.sessions_root, Path)
            or not isinstance(candidate.namespace_path, Path)
            or not isinstance(candidate.session_path, Path)
            or not isinstance(candidate.session_key, str)
            or not candidate.session_key.strip()
            or not isinstance(candidate.workspace_id, str)
            or _RECOVERY_NAMESPACE_RE.fullmatch(candidate.workspace_id) is None
        ):
            raise ValueError
        identity_values = (
            candidate.workspace_identity,
            candidate.sessions_root_identity,
            candidate.namespace_identity,
            candidate.workspace_marker_identity,
            candidate.namespace_marker_identity,
            candidate.session_file_identity,
        )
        for expected_length, identity in zip((5, 5, 5, 9, 9, 9), identity_values, strict=True):
            if (
                type(identity) is not tuple
                or len(identity) != expected_length
                or any(type(value) is not int for value in identity)
            ):
                raise ValueError
        workspace_path = os.fspath(candidate.workspace_path)
        path = os.fspath(candidate.sessions_root)
        workspace_components = _configured_components(workspace_path)
        components = _configured_components(path)
        del workspace_components
        operation.require_current()
        lease_identity = operation.workspace_identity
        lease_workspace = operation._authority.workspace
        if (
            candidate.workspace_path != lease_workspace
            or candidate.workspace_identity[:2] != lease_identity
            or candidate.namespace_path != candidate.sessions_root / candidate.workspace_id
        ):
            raise ValueError
        storage_key = StoreType.storage_key(candidate.session_key)
        decoded = StoreType.decode_storage_key(storage_key)
        if (
            not _is_safe_component(storage_key)
            or decoded != candidate.session_key
            or StoreType.storage_key(decoded) != storage_key
            or candidate.session_path
            != candidate.namespace_path / f"{storage_key}.jsonl"
        ):
            raise ValueError
        if not stat.S_ISDIR(candidate.workspace_identity[2]):
            raise ValueError
        if not stat.S_ISDIR(candidate.sessions_root_identity[2]):
            raise ValueError
        if not stat.S_ISDIR(candidate.namespace_identity[2]):
            raise ValueError
        for identity in (
            candidate.workspace_marker_identity,
            candidate.namespace_marker_identity,
            candidate.session_file_identity,
        ):
            if not stat.S_ISREG(identity[2]):
                raise ValueError
    except Exception:
        failure = True
    if failure:
        raise CheckedSessionAuthorityError("invalid") from None
    return CheckedSessionAuthority(
        _CONSTRUCTOR_TOKEN,
        None,
        candidate.session_key,
        storage_key,
        path,
        components,
        memory_operation=operation,
        recovery_candidate=candidate,
        recovery_replace=replace,
    )


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
