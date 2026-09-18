"""Descriptor-bound private storage for memory-recovery bytes.

This is the small B1a storage boundary.  It owns only the workspace binding,
the persistent ``.memory-recovery.lock``, the private recovery blob layout, and
verified immutable byte publication/read operations.  It deliberately does not
know about manifests, phases, receipts, pointers, cleanup, or runtime memory
policy.

The authority is process-local and must be used as a context manager.  Its
lock is the first lock in the future order: memory authority, namespace
migration (when that lock is introduced), execution, then session-files.  B1a
acquires no later lock.  Content digests provide portable identity when a tree
is copied; open descriptors and inode identities bind one operation to one
tree.  This is not authenticity or rollback protection against an arbitrary
same-UID owner that can rewrite or delete the whole store after validation.

Only the caller-supplied workspace alias is resolved.  All storage traversal
after that point is descriptor-relative, no-follow, and directory-qualified.
Unsupported platforms fail closed when this feature is invoked; importing
this module has no platform side effects.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Self

try:  # pragma: no cover - platform qualification is tested through the gate.
    import fcntl as _fcntl
except ImportError:  # pragma: no cover
    _fcntl = None  # type: ignore[assignment]


_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_LOCK_NAME = ".memory-recovery.lock"
_MEMORY_NAME = "memory"
_RECOVERY_NAME = ".recovery"
_BLOBS_NAME = "blobs"
_DIGEST_DIR_NAME = "sha256"
_COMPANION_PREFIX = ".install-"
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_READ_CHUNK = 64 * 1024
_MISSING = object()


class MemoryStorageError(RuntimeError):
    """Fixed, content-free error raised by the memory storage boundary."""

    _MESSAGES = {
        "unavailable": "memory recovery storage is unavailable",
        "invalid": "memory recovery storage input is invalid",
        "closed": "memory recovery authority is closed",
        "binding": "memory recovery storage binding changed",
        "corrupt": "memory recovery storage content is unsafe",
        "conflict": "memory recovery storage content conflicts",
        "missing": "memory recovery blob is unavailable",
        "durability": "memory recovery storage durability failed",
        "io": "memory recovery storage operation failed",
    }

    def __init__(self, code: str) -> None:
        if code not in self._MESSAGES:
            code = "io"
        self.code = code
        super().__init__(self._MESSAGES[code])


# Names used by callers that prefer the package/boundary vocabulary.
MemoryRecoveryStorageError = MemoryStorageError
WorkspaceMemoryAuthorityError = MemoryStorageError


class MemoryStorageUnavailableError(MemoryStorageError):
    """The descriptor, no-follow, fsync, or lock capability is unavailable."""

    def __init__(self) -> None:
        super().__init__("unavailable")


class MemoryStorageConflictError(MemoryStorageError):
    """A caller-owned digest destination has different or unsafe content."""

    def __init__(self) -> None:
        super().__init__("conflict")


class MemoryStorageCorruptError(MemoryStorageError):
    """A known private object no longer satisfies its storage invariant."""

    def __init__(self) -> None:
        super().__init__("corrupt")


class MemoryStorageBindingError(MemoryStorageError):
    """An observed workspace, directory, lock, or file binding changed."""

    def __init__(self) -> None:
        super().__init__("binding")


class MemoryStorageDurabilityError(MemoryStorageError):
    """A required file or directory durability boundary failed."""

    def __init__(self) -> None:
        super().__init__("durability")


class MemoryStorageMissingError(MemoryStorageError):
    """A requested target blob is not present."""

    def __init__(self) -> None:
        super().__init__("missing")


@dataclass(frozen=True, slots=True)
class _Entry:
    name: str
    identity: tuple[int, int]
    info: os.stat_result
    data: bytes | None = None


def _error(code: str) -> MemoryStorageError:
    return MemoryStorageError(code)


def _close(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except (OSError, TypeError, ValueError):
        pass


def _fstat(fd: int, *, code: str = "io") -> os.stat_result:
    try:
        return os.fstat(fd)
    except (OSError, TypeError, ValueError):
        raise _error(code) from None


def _fsync(fd: int) -> None:
    try:
        os.fsync(fd)
    except (OSError, TypeError, ValueError):
        raise _error("durability") from None


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _same_binding(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        _identity(first) == _identity(second)
        and stat.S_IFMT(first.st_mode) == stat.S_IFMT(second.st_mode)
    )


def _same_snapshot(first: os.stat_result, second: os.stat_result) -> bool:
    """Compare content/binding fields while allowing our own hard-link count change."""

    return (
        _same_binding(first, second)
        and first.st_uid == second.st_uid
        and first.st_gid == second.st_gid
        and stat.S_IMODE(first.st_mode) == stat.S_IMODE(second.st_mode)
        and first.st_size == second.st_size
        and getattr(first, "st_mtime_ns", None) == getattr(second, "st_mtime_ns", None)
        and getattr(first, "st_ctime_ns", None) == getattr(second, "st_ctime_ns", None)
    )


def _same_identity_shape(first: os.stat_result, second: os.stat_result) -> bool:
    """Compare a retained inode while allowing link-count/time updates."""

    return (
        _same_binding(first, second)
        and first.st_uid == second.st_uid
        and first.st_gid == second.st_gid
        and stat.S_IMODE(first.st_mode) == stat.S_IMODE(second.st_mode)
        and first.st_size == second.st_size
    )


def _uid() -> int:
    get_effective = getattr(os, "geteuid", None)
    if callable(get_effective):
        return int(get_effective())
    return int(os.getuid())


def _supports_dir_fd(function: object) -> bool:
    return function in getattr(os, "supports_dir_fd", set())


def _require_capabilities() -> None:
    """Fail only on explicit use; importing this module stays portable."""

    if os.name != "posix":
        raise MemoryStorageUnavailableError()
    for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"):
        value = getattr(os, name, None)
        if not isinstance(value, int) or value <= 0:
            raise MemoryStorageUnavailableError()
    if not callable(getattr(os, "fsync", None)):
        raise MemoryStorageUnavailableError()
    if _fcntl is None or not callable(getattr(_fcntl, "flock", None)):
        raise MemoryStorageUnavailableError()
    if not all(isinstance(getattr(_fcntl, name, None), int) for name in ("LOCK_EX", "LOCK_UN")):
        raise MemoryStorageUnavailableError()
    required = {os.open, os.stat, os.mkdir, os.link}
    if not required <= getattr(os, "supports_dir_fd", set()):
        raise MemoryStorageUnavailableError()
    if os.link not in getattr(os, "supports_follow_symlinks", set()):
        raise MemoryStorageUnavailableError()


def _dir_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if isinstance(cloexec, int):
        flags |= cloexec
    return flags


def _leaf_flags() -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if isinstance(cloexec, int):
        flags |= cloexec
    return flags


def _write_flags() -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if isinstance(cloexec, int):
        flags |= cloexec
    return flags


def _canonical_workspace(value: Path | str) -> Path:
    try:
        candidate = Path(value).expanduser()
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise _error("invalid") from None
    if not resolved.is_absolute() or resolved == Path(resolved.anchor) or ".." in resolved.parts:
        raise _error("invalid") from None
    return resolved


def _private_directory(info: os.stat_result, *, workspace: bool = False) -> None:
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != _uid():
        raise _error("corrupt") from None
    if not workspace and stat.S_IMODE(info.st_mode) != _PRIVATE_DIRECTORY_MODE:
        raise _error("corrupt") from None


def _private_file(info: os.stat_result, *, links: set[int] | frozenset[int]) -> None:
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != _uid()
        or stat.S_IMODE(info.st_mode) != _PRIVATE_FILE_MODE
        or info.st_nlink not in links
    ):
        raise _error("corrupt") from None


def _open_directory_path(path: Path) -> int:
    """Open every component from the anchor with no-follow directory flags."""

    fd: int | None = None
    try:
        fd = os.open(path.anchor, _dir_flags())
        for component in path.parts[1:]:
            try:
                expected = os.stat(component, dir_fd=fd, follow_symlinks=False)
            except (OSError, TypeError, ValueError):
                raise _error("io") from None
            if not stat.S_ISDIR(expected.st_mode):
                raise _error("corrupt") from None
            try:
                child = os.open(component, _dir_flags(), dir_fd=fd)
            except (OSError, TypeError, ValueError):
                raise _error("io") from None
            try:
                opened = _fstat(child, code="binding")
                if not _same_binding(expected, opened) or not stat.S_ISDIR(opened.st_mode):
                    raise _error("binding") from None
            except BaseException:
                _close(child)
                raise
            _close(fd)
            fd = child
        if fd is None:  # pragma: no cover - absolute non-root paths always have components.
            raise _error("invalid") from None
        return fd
    except MemoryStorageError:
        _close(fd)
        raise
    except (OSError, TypeError, ValueError):
        _close(fd)
        raise _error("io") from None


def _stat_entry(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except (OSError, TypeError, ValueError):
        raise _error("io") from None


def _check_directory_binding(parent_fd: int, name: str, child_fd: int) -> os.stat_result:
    """Prove a directory descriptor and its parent name still refer together."""

    parent_info = _fstat(parent_fd, code="binding")
    if not stat.S_ISDIR(parent_info.st_mode):
        raise _error("binding") from None
    child_info = _fstat(child_fd, code="binding")
    _private_directory(child_info)
    entry = _stat_entry(parent_fd, name)
    if entry is None or not _same_binding(entry, child_info):
        raise _error("binding") from None
    return child_info


def _sync_directory_boundary(parent_fd: int, name: str, child_fd: int) -> None:
    """Fsync a child and parent, proving bindings around each durability call."""

    _check_directory_binding(parent_fd, name, child_fd)
    _fsync(child_fd)
    _check_directory_binding(parent_fd, name, child_fd)
    _fsync(parent_fd)
    _check_directory_binding(parent_fd, name, child_fd)


def _open_child_directory(parent_fd: int, name: str, *, create: bool) -> int:
    """Open one private directory and bind its name to its descriptor."""

    observed = _stat_entry(parent_fd, name)
    if observed is None:
        if not create:
            raise _error("missing") from None
        try:
            os.mkdir(name, _PRIVATE_DIRECTORY_MODE, dir_fd=parent_fd)
        except FileExistsError:
            pass
        except (OSError, TypeError, ValueError):
            raise _error("io") from None
        observed = _stat_entry(parent_fd, name)
        if observed is None:
            raise _error("binding") from None
    if not stat.S_ISDIR(observed.st_mode):
        raise _error("corrupt") from None
    try:
        child = os.open(name, _dir_flags(), dir_fd=parent_fd)
    except (OSError, TypeError, ValueError):
        raise _error("io") from None
    try:
        opened = _fstat(child, code="binding")
        _private_directory(opened)
        if not _same_binding(observed, opened):
            raise _error("binding") from None
        # Existing directories use the same repair path as newly-created
        # ones.  A prior failed parent fsync is therefore repaired on retry.
        _sync_directory_boundary(parent_fd, name, child)
        return child
    except BaseException:
        _close(child)
        raise


def _validate_digest(value: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise _error("invalid") from None
    return value


def _write_all(fd: int, content: bytes) -> None:
    view = memoryview(content)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(fd, view[offset : offset + _READ_CHUNK])
        except (OSError, TypeError, ValueError):
            raise _error("io") from None
        if written <= 0:
            raise _error("io") from None
        offset += written


class WorkspaceMemoryAuthority(AbstractContextManager[Self]):
    """A process-local authority for one private workspace memory namespace.

    Constructing this object does not create files.  Entering the context
    acquires the persistent workspace lock, creates or validates the private
    layout, and pins descriptors for the operation.  ``publish_blob`` and
    ``read_blob`` require the context to be active.
    """

    def __init__(self, workspace: Path | str) -> None:
        _require_capabilities()
        self._workspace_path = _canonical_workspace(workspace)
        self._workspace_fd: int | None = None
        self._lock_fd: int | None = None
        self._memory_fd: int | None = None
        self._recovery_fd: int | None = None
        self._blobs_fd: int | None = None
        self._digest_fd: int | None = None
        self._workspace_identity: tuple[int, int] | None = None
        self._lock_identity: tuple[int, int] | None = None
        self._pid = os.getpid()
        self._active = False
        self._lock_held = False

    @property
    def workspace(self) -> Path:
        """The resolved caller workspace alias, for diagnostics and tests."""

        return self._workspace_path

    @property
    def blob_directory(self) -> Path:
        """The conventional layout path; callers cannot use it for I/O here."""

        return self._workspace_path / _MEMORY_NAME / _RECOVERY_NAME / _BLOBS_NAME / _DIGEST_DIR_NAME

    @property
    def lock_path(self) -> Path:
        return self._workspace_path / _LOCK_NAME

    def __reduce__(self) -> object:
        raise TypeError("memory recovery authority cannot be serialized")

    def __copy__(self) -> Self:
        raise TypeError("memory recovery authority cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        raise TypeError("memory recovery authority cannot be copied")

    def __enter__(self) -> Self:
        if self._active:
            raise _error("closed") from None
        if self._pid != os.getpid():
            raise _error("closed") from None
        _require_capabilities()
        workspace_fd: int | None = None
        lock_fd: int | None = None
        try:
            workspace_fd = _open_directory_path(self._workspace_path)
            workspace_info = _fstat(workspace_fd)
            _private_directory(workspace_info, workspace=True)
            self._workspace_identity = _identity(workspace_info)
            self._workspace_fd = workspace_fd
            workspace_fd = None

            lock_fd = self._open_or_create_lock()
            self._lock_fd = lock_fd
            lock_fd = None
            try:
                _fcntl.flock(self._lock_fd, _fcntl.LOCK_EX)  # type: ignore[arg-type]
            except (OSError, TypeError, ValueError):
                raise _error("io") from None
            self._lock_held = True
            self._lock_identity = _identity(_fstat(self._lock_fd))  # type: ignore[arg-type]
            # This is deliberately before the first mkdir: a lock path swapped
            # while this process was blocked must never authorize mutation.
            self._assert_binding(layout=False)

            assert self._workspace_fd is not None
            self._memory_fd = _open_child_directory(self._workspace_fd, _MEMORY_NAME, create=True)
            self._recovery_fd = _open_child_directory(self._memory_fd, _RECOVERY_NAME, create=True)
            self._blobs_fd = _open_child_directory(self._recovery_fd, _BLOBS_NAME, create=True)
            self._digest_fd = _open_child_directory(self._blobs_fd, _DIGEST_DIR_NAME, create=True)
            self._assert_binding()
            self._active = True
            return self
        except MemoryStorageError:
            self._cleanup()
            _close(workspace_fd)
            _close(lock_fd)
            raise
        except (OSError, TypeError, ValueError):
            self._cleanup()
            _close(workspace_fd)
            _close(lock_fd)
            raise _error("io") from None
        except BaseException:
            self._cleanup()
            _close(workspace_fd)
            _close(lock_fd)
            raise

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._cleanup()

    def close(self) -> None:
        """Release the authority and all descriptors; never remove storage."""

        self._cleanup()

    def _cleanup(self) -> None:
        self._active = False
        if self._lock_held and self._lock_fd is not None and _fcntl is not None:
            try:
                _fcntl.flock(self._lock_fd, _fcntl.LOCK_UN)
            except (OSError, TypeError, ValueError):
                pass
        self._lock_held = False
        for attr in ("_digest_fd", "_blobs_fd", "_recovery_fd", "_memory_fd", "_lock_fd", "_workspace_fd"):
            fd = getattr(self, attr)
            setattr(self, attr, None)
            _close(fd)

    def _open_or_create_lock(self) -> int:
        assert self._workspace_fd is not None
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
        cloexec = getattr(os, "O_CLOEXEC", 0)
        if isinstance(cloexec, int):
            flags |= cloexec
        lock_fd: int | None = None
        for _ in range(8):
            try:
                lock_fd = os.open(_LOCK_NAME, flags, dir_fd=self._workspace_fd)
                break
            except FileNotFoundError:
                try:
                    lock_fd = os.open(
                        _LOCK_NAME,
                        flags | os.O_CREAT | os.O_EXCL,
                        _PRIVATE_FILE_MODE,
                        dir_fd=self._workspace_fd,
                    )
                    break
                except FileExistsError:
                    continue
            except (OSError, TypeError, ValueError):
                raise _error("io") from None
        if lock_fd is None:
            raise _error("io") from None
        try:
            self._check_lock_binding(lock_fd)
            _fsync(lock_fd)
            self._check_lock_binding(lock_fd)
            _fsync(self._workspace_fd)
            self._check_lock_binding(lock_fd)
            return lock_fd
        except BaseException:
            _close(lock_fd)
            raise

    def _check_lock_binding(self, lock_fd: int) -> None:
        assert self._workspace_fd is not None
        workspace_info = _fstat(self._workspace_fd, code="binding")
        if not stat.S_ISDIR(workspace_info.st_mode) or workspace_info.st_uid != _uid():
            raise _error("binding") from None
        lock_info = _fstat(lock_fd, code="binding")
        _private_file(lock_info, links={1})
        path_lock = _stat_entry(self._workspace_fd, _LOCK_NAME)
        if path_lock is None or not _same_binding(path_lock, lock_info):
            raise _error("binding") from None

    def _assert_process(self) -> None:
        if not self._active and self._workspace_fd is None:
            raise _error("closed") from None
        if self._pid != os.getpid():
            raise _error("closed") from None

    def _assert_binding(self, *, layout: bool = True) -> None:
        """Rewalk the configured path and every observed descriptor binding."""

        self._assert_process()
        if (
            self._workspace_fd is None
            or self._lock_fd is None
            or self._workspace_identity is None
            or self._lock_identity is None
        ):
            raise _error("closed") from None
        try:
            workspace_info = _fstat(self._workspace_fd, code="binding")
        except MemoryStorageError:
            raise
        _private_directory(workspace_info, workspace=True)
        if _identity(workspace_info) != self._workspace_identity:
            raise _error("binding") from None
        rebound = _open_directory_path(self._workspace_path)
        try:
            rebound_info = _fstat(rebound, code="binding")
            if not _same_binding(workspace_info, rebound_info):
                raise _error("binding") from None
        finally:
            _close(rebound)

        try:
            lock_info = _fstat(self._lock_fd, code="binding")
        except MemoryStorageError:
            raise
        _private_file(lock_info, links={1})
        if _identity(lock_info) != self._lock_identity:
            raise _error("binding") from None
        path_lock = _stat_entry(self._workspace_fd, _LOCK_NAME)
        if path_lock is None:
            raise _error("binding") from None
        _private_file(path_lock, links={1})
        if not _same_binding(path_lock, lock_info):
            raise _error("binding") from None
        if not layout:
            return

        parents = (
            (self._workspace_fd, _MEMORY_NAME, self._memory_fd),
            (self._memory_fd, _RECOVERY_NAME, self._recovery_fd),
            (self._recovery_fd, _BLOBS_NAME, self._blobs_fd),
            (self._blobs_fd, _DIGEST_DIR_NAME, self._digest_fd),
        )
        for parent, name, child in parents:
            if parent is None or child is None:
                raise _error("closed") from None
            try:
                child_info = _fstat(child, code="binding")
            except MemoryStorageError:
                raise
            _private_directory(child_info)
            entry = _stat_entry(parent, name)
            if entry is None or not _same_binding(entry, child_info):
                raise _error("binding") from None

    def _entry_binding(self, entry: _Entry) -> None:
        if self._digest_fd is None:
            raise _error("closed") from None
        current = _stat_entry(self._digest_fd, entry.name)
        if current is None or not _same_snapshot(current, entry.info):
            raise _error("binding") from None

    def _open_verified_file(
        self,
        name: str,
        digest: str,
        *,
        expected: bytes | None,
        missing_code: str,
        conflict_code: str,
        allowed_links: set[int] | frozenset[int],
        fsync: bool = False,
    ) -> _Entry:
        assert self._digest_fd is not None
        observed = _stat_entry(self._digest_fd, name)
        if observed is None:
            raise _error(missing_code) from None
        fd: int | None = None
        try:
            fd = os.open(name, _leaf_flags(), dir_fd=self._digest_fd)
        except (OSError, TypeError, ValueError):
            raise _error("corrupt") from None
        try:
            before = _fstat(fd, code="binding")
            if not _same_binding(observed, before):
                raise _error("binding") from None
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != _uid()
                or stat.S_IMODE(before.st_mode) != _PRIVATE_FILE_MODE
                or before.st_nlink not in allowed_links
            ):
                raise _error("corrupt") from None
            digest_state = hashlib.sha256()
            result = bytearray()
            position = 0
            while True:
                try:
                    chunk = os.read(fd, _READ_CHUNK)
                except (OSError, TypeError, ValueError):
                    raise _error("io") from None
                if not chunk:
                    break
                digest_state.update(chunk)
                if expected is not None:
                    end = position + len(chunk)
                    if end > len(expected) or chunk != expected[position:end]:
                        raise _error(conflict_code) from None
                else:
                    result.extend(chunk)
                position += len(chunk)
            after = _fstat(fd, code="binding")
            if not _same_snapshot(before, after) or position != after.st_size:
                raise _error("binding") from None
            if digest_state.hexdigest() != digest:
                raise _error(conflict_code) from None
            if expected is not None and position != len(expected):
                raise _error(conflict_code) from None
            current = _stat_entry(self._digest_fd, name)
            if current is None or not _same_snapshot(current, after):
                raise _error("binding") from None
            if fsync:
                _fsync(fd)
                final = _fstat(fd, code="binding")
                current = _stat_entry(self._digest_fd, name)
                if current is None or not _same_snapshot(final, after) or not _same_snapshot(current, final):
                    raise _error("binding") from None
            return _Entry(name, _identity(after), after, bytes(result) if expected is None else expected)
        finally:
            _close(fd)

    def _validate_relationship(self, target: _Entry, companion: _Entry | None) -> None:
        """Accept exact original hardlinks or copied independent private files."""

        if self._digest_fd is None:
            raise _error("closed") from None
        if companion is None:
            if target.info.st_nlink != 1:
                raise _error("corrupt") from None
            return
        same = target.identity == companion.identity
        if same:
            if target.info.st_nlink != 2 or companion.info.st_nlink != 2:
                raise _error("corrupt") from None
            names: set[str] = set()
            try:
                entries = os.listdir(self._digest_fd)
            except (OSError, TypeError, ValueError):
                raise _error("io") from None
            for name in entries:
                info = _stat_entry(self._digest_fd, name)
                if info is not None and _identity(info) == target.identity:
                    names.add(name)
            if names != {target.name, companion.name}:
                raise _error("corrupt") from None
        else:
            if target.info.st_nlink != 1 or companion.info.st_nlink != 1:
                raise _error("corrupt") from None

    def _state(self, digest: str, *, expected: bytes | None) -> tuple[_Entry | None, _Entry | None]:
        target_name = digest
        companion_name = _COMPANION_PREFIX + digest
        target_info = _stat_entry(self._digest_fd_or_closed(), target_name)
        companion_info = _stat_entry(self._digest_fd_or_closed(), companion_name)
        target: _Entry | None = None
        companion: _Entry | None = None
        if target_info is not None:
            target = self._open_verified_file(
                target_name,
                digest,
                expected=expected,
                missing_code="missing",
                conflict_code="conflict" if expected is not None else "corrupt",
                allowed_links={1, 2},
            )
        if companion_info is not None:
            companion = self._open_verified_file(
                companion_name,
                digest,
                expected=expected,
                missing_code="missing",
                conflict_code="conflict" if expected is not None else "corrupt",
                allowed_links={1, 2},
            )
        if target is not None and companion is not None:
            self._validate_relationship(target, companion)
        elif target is not None:
            self._validate_relationship(target, None)
        elif companion is not None and companion.info.st_nlink != 1:
            # A companion with no target cannot have an unexplained extra
            # hardlink.  Recovery links it only from this nlink=1 state.
            raise _error("corrupt") from None
        return target, companion

    def _digest_fd_or_closed(self) -> int:
        if self._digest_fd is None:
            raise _error("closed") from None
        return self._digest_fd

    def _sync_digest_directory(self) -> None:
        digest_fd = self._digest_fd_or_closed()
        self._assert_binding()
        _fsync(digest_fd)
        self._assert_binding()

    def _create_companion(self, digest: str, content: bytes) -> _Entry:
        assert self._digest_fd is not None
        name = _COMPANION_PREFIX + digest
        fd: int | None = None
        try:
            try:
                fd = os.open(name, _write_flags(), _PRIVATE_FILE_MODE, dir_fd=self._digest_fd)
            except FileExistsError:
                return self._open_verified_file(
                    name,
                    digest,
                    expected=content,
                    missing_code="missing",
                    conflict_code="conflict",
                    allowed_links={1, 2},
                )
            except (OSError, TypeError, ValueError):
                raise _error("io") from None
            before = _fstat(fd)
            _private_file(before, links={1})
            _write_all(fd, content)
            current = _stat_entry(self._digest_fd, name)
            if current is None or not _same_binding(current, before):
                raise _error("binding") from None
            _fsync(fd)
            after = _fstat(fd, code="binding")
            if (
                not _same_binding(before, after)
                or after.st_uid != before.st_uid
                or after.st_gid != before.st_gid
                or stat.S_IMODE(after.st_mode) != _PRIVATE_FILE_MODE
                or after.st_nlink != 1
                or after.st_size != len(content)
            ):
                raise _error("binding") from None
            current = _stat_entry(self._digest_fd, name)
            if current is None or not _same_identity_shape(current, after):
                raise _error("binding") from None
            verified = self._open_verified_file(
                name,
                digest,
                expected=content,
                missing_code="missing",
                conflict_code="conflict",
                allowed_links={1},
            )
            self._entry_binding(verified)
            self._sync_digest_directory()
            self._assert_binding()
            return verified
        finally:
            _close(fd)

    def _compare_final_state(
        self,
        before_target: _Entry,
        before_companion: _Entry | None,
        after_target: _Entry | None,
        after_companion: _Entry | None,
    ) -> None:
        """Reject a replacement that happened during the final dir fsync."""

        if after_target is None or (before_companion is None) != (after_companion is None):
            raise _error("binding") from None
        if after_target.identity != before_target.identity or not _same_snapshot(
            before_target.info, after_target.info
        ):
            raise _error("binding") from None
        if after_target.data != before_target.data:
            raise _error("binding") from None
        if before_companion is not None and after_companion is not None:
            if after_companion.identity != before_companion.identity or not _same_snapshot(
                before_companion.info, after_companion.info
            ):
                raise _error("binding") from None
            if after_companion.data != before_companion.data:
                raise _error("binding") from None
        self._validate_relationship(after_target, after_companion)

    def _durable_success(
        self,
        digest: str,
        content: bytes | None,
        *,
        retained_companion_identity: tuple[int, int] | None = None,
    ) -> bytes:
        target, companion = self._state(digest, expected=content)
        if target is None:
            raise _error("missing") from None
        self._validate_relationship(target, companion)
        if retained_companion_identity is not None and (
            companion is None or companion.identity != retained_companion_identity
        ):
            raise _error("binding") from None
        durable_target = self._open_verified_file(
            digest,
            digest,
            expected=content,
            missing_code="missing",
            conflict_code="conflict" if content is not None else "corrupt",
            allowed_links={1, 2},
            fsync=True,
        )
        before_directory_target, before_directory_companion = self._state(digest, expected=content)
        if before_directory_target is None:
            raise _error("missing") from None
        # Keep the first verified state bound through the target-file fsync.
        # A copied tree legitimately has independent target/companion inodes,
        # but replacing either path with another exact file during this window
        # is still a binding change and must fail before directory-fsync rebasing.
        self._compare_final_state(
            target,
            companion,
            before_directory_target,
            before_directory_companion,
        )
        self._validate_relationship(before_directory_target, before_directory_companion)
        if retained_companion_identity is not None and (
            before_directory_companion is None
            or before_directory_companion.identity != retained_companion_identity
        ):
            raise _error("binding") from None
        self._entry_binding(durable_target)
        self._assert_binding()
        self._sync_digest_directory()
        after_directory_target, after_directory_companion = self._state(digest, expected=content)
        self._compare_final_state(
            before_directory_target,
            before_directory_companion,
            after_directory_target,
            after_directory_companion,
        )
        if retained_companion_identity is not None and (
            after_directory_companion is None
            or after_directory_companion.identity != retained_companion_identity
        ):
            raise _error("binding") from None
        self._assert_binding()
        return after_directory_target.data or b""

    def publish_blob(self, content: bytes) -> str:
        """Publish bytes under their exact SHA-256 name without clobbering."""

        self._assert_process()
        if type(content) is not bytes:
            raise _error("invalid") from None
        digest = hashlib.sha256(content).hexdigest()
        self._assert_binding()
        target, companion = self._state(digest, expected=content)
        if target is not None:
            # A copied tree may contain only a safe target.  Restore the
            # permanent companion by exclusive creation, never replacement.
            if companion is None:
                self._create_companion(digest, content)
            self._durable_success(digest, content)
            return digest

        if companion is None:
            companion = self._create_companion(digest, content)
        else:
            if companion.info.st_nlink != 1:
                raise _error("corrupt") from None
        self._entry_binding(companion)
        self._assert_binding()
        linked = False
        try:
            assert self._digest_fd is not None
            os.link(
                _COMPANION_PREFIX + digest,
                digest,
                src_dir_fd=self._digest_fd,
                dst_dir_fd=self._digest_fd,
                follow_symlinks=False,
            )
            linked = True
        except FileExistsError:
            # The destination is the sole arbitration point.  Verify the
            # winner; this path never replaces or removes it.
            target, current_companion = self._state(digest, expected=content)
            if target is None:
                raise _error("conflict") from None
            self._entry_binding(companion)
            if current_companion is None:
                raise _error("binding") from None
        except (OSError, TypeError, ValueError):
            raise _error("io") from None
        self._assert_binding()
        self._durable_success(
            digest,
            content,
            retained_companion_identity=companion.identity if linked else None,
        )
        return digest

    # Small names make the typed boundary convenient without adding another
    # storage abstraction.  They all retain the same context/lock semantics.
    publish = publish_blob
    put = publish_blob

    def read_blob(self, digest: str) -> bytes:
        """Read and fully verify one immutable target blob."""

        digest = _validate_digest(digest)
        self._assert_process()
        self._assert_binding()
        return self._durable_success(digest, None)

    read = read_blob
    get = read_blob


# A descriptive alias for callers that prefer a storage noun.
MemoryRecoveryStorage = WorkspaceMemoryAuthority


def open_workspace_memory(workspace: Path | str) -> WorkspaceMemoryAuthority:
    """Return the explicit context-managed workspace authority."""

    return WorkspaceMemoryAuthority(workspace)


__all__ = [
    "MemoryRecoveryStorage",
    "MemoryRecoveryStorageError",
    "MemoryStorageBindingError",
    "MemoryStorageConflictError",
    "MemoryStorageCorruptError",
    "MemoryStorageDurabilityError",
    "MemoryStorageError",
    "MemoryStorageMissingError",
    "MemoryStorageUnavailableError",
    "WorkspaceMemoryAuthority",
    "WorkspaceMemoryAuthorityError",
    "open_workspace_memory",
]
