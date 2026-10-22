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
The existing workspace ``memory`` directory may retain owner-controlled
permissions without group/other write access; the recovery subtree remains
exactly ``0700``.
Unsupported platforms fail closed when this feature is invoked; importing
this module has no platform side effects. Legacy target snapshots are bounded
to 64 MiB, so larger targets require a separately qualified reader.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import stat
import threading
import weakref
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NoReturn, Protocol, Self, cast

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
_EVIDENCE_KINDS = frozenset(("transactions", "conversions"))
_RECORD_SLOT_RE = re.compile(r"(?:genesis\.json|next-[0-9a-f]{64}\.json)\Z")
_NAMESPACE_CAPABILITY = object()
_OPERATION_CAPABILITY = object()
_BOUND_TARGET_CAPABILITY = object()
_BOUND_RECEIPT_TARGET_CAPABILITY = object()
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_READ_CHUNK = 64 * 1024
_MEMORY_TARGET_MAX_BYTES = 64 * 1024 * 1024
MEMORY_TARGET_MAX_BYTES = _MEMORY_TARGET_MAX_BYTES
_MEMORY_TARGET_NAMES = frozenset(
    (
        "MEMORY.md",
        "history.jsonl",
        "HISTORY.md",
        ".cursor",
        ".dream_cursor",
        ".consolidation.pending.json",
    )
)
_RECEIPT_TARGET_NAME = ".consolidation.pending.json"
_SNAPSHOT_CAPABILITY = object()
_MISSING = object()


class MemoryStorageError(RuntimeError):
    """Fixed, content-free error raised by the memory storage boundary."""

    _MESSAGES = {
        "unavailable": "memory recovery storage is unavailable",
        "invalid": "memory recovery storage input is invalid",
        "closed": "memory recovery authority is closed",
        "borrowed": "memory recovery authority is borrowed",
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


@dataclass(frozen=True, slots=True)
class MemoryTargetStatIdentity:
    """Stable stat fields used to qualify one detached legacy-target read."""

    device: int
    inode: int
    mode: int
    uid: int
    gid: int
    links: int
    size: int
    mtime_ns: int | None
    ctime_ns: int | None


_TargetStatSeal = tuple[int, int, int, int, int, int, int, int | None, int | None]
_TargetSnapshotSeal = tuple[str, bytes | None, _TargetStatSeal | None, int]


@dataclass(frozen=True, slots=True, init=False, repr=False, weakref_slot=True)
class MemoryTargetSnapshot:
    """Opaque immutable bytes captured under one active memory authority."""

    _name: str
    _content: bytes | None
    _stat_identity: MemoryTargetStatIdentity | None
    _authority: WorkspaceMemoryAuthority
    _epoch: int

    def __init__(
        self,
        name: str,
        content: bytes | None,
        stat_identity: MemoryTargetStatIdentity | None,
        authority: WorkspaceMemoryAuthority,
        epoch: int,
        *,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _SNAPSHOT_CAPABILITY:
            raise _error("invalid") from None
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_content", content)
        object.__setattr__(self, "_stat_identity", stat_identity)
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_epoch", epoch)

    @property
    def name(self) -> str:
        """The exact allowlisted workspace-memory basename."""

        return self._name

    @property
    def content(self) -> bytes | None:
        """Exact bytes, or ``None`` when the target was absent."""

        return self._content

    @property
    def exists(self) -> bool:
        """Whether the target existed at the qualified snapshot point."""

        return self._content is not None

    @property
    def stat_identity(self) -> MemoryTargetStatIdentity | None:
        """Stable target identity, or ``None`` for an absent target."""

        return self._stat_identity

    def issuance_comparison(
        self, *, _capability: object | None = None,
    ) -> _TargetSnapshotSeal:
        if _capability is not _SNAPSHOT_CAPABILITY:
            raise _error("invalid") from None
        identity = self._stat_identity
        stat_seal = (
            None
            if identity is None
            else (
                identity.device,
                identity.inode,
                identity.mode,
                identity.uid,
                identity.gid,
                identity.links,
                identity.size,
                identity.mtime_ns,
                identity.ctime_ns,
            )
        )
        return self._name, self._content, stat_seal, self._epoch

    def qualify_for_authority(
        self, authority: WorkspaceMemoryAuthority, epoch: int, allow_receipt: bool,
        *, _capability: object | None = None,
    ) -> None:
        if _capability is not _SNAPSHOT_CAPABILITY:
            raise _error("invalid") from None
        if self._authority is not authority:
            raise _error("invalid") from None
        if self._epoch != epoch:
            raise _error("binding") from None
        if (
            type(self._name) is not str
            or self._name not in _MEMORY_TARGET_NAMES
            or (self._name == _RECEIPT_TARGET_NAME and not allow_receipt)
            or (self._content is None) != (self._stat_identity is None)
            or (
                self._stat_identity is not None
                and type(self._stat_identity) is not MemoryTargetStatIdentity
            )
            or (
                self._content is not None
                and (
                    type(self._content) is not bytes
                    or len(self._content) > _MEMORY_TARGET_MAX_BYTES
                    or self._stat_identity is None
                    or self._stat_identity.size != len(self._content)
                )
            )
        ):
            raise _error("invalid") from None
        WorkspaceMemoryAuthority.require_issued_snapshot(
            authority, self, _capability=_SNAPSHOT_CAPABILITY,
        )
        if allow_receipt:
            identity = self._stat_identity
            if self._name != _RECEIPT_TARGET_NAME:
                raise _error("invalid") from None
            if self._content is None or identity is None:
                raise _error("invalid") from None
            if (
                not stat.S_ISREG(identity.mode)
                or identity.uid != _uid()
                or identity.links != 1
                or stat.S_IMODE(identity.mode) != _PRIVATE_FILE_MODE
            ):
                raise _error("corrupt") from None

    def __repr__(self) -> str:
        return "<MemoryTargetSnapshot>"

    def __reduce__(self) -> NoReturn:
        raise TypeError("memory target snapshot cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("memory target snapshot cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("memory target snapshot cannot be copied")


def _target_snapshot_seal(snapshot: MemoryTargetSnapshot) -> _TargetSnapshotSeal:
    return MemoryTargetSnapshot.issuance_comparison(
        snapshot, _capability=_SNAPSHOT_CAPABILITY,
    )


def _target_stat_identity(info: os.stat_result) -> MemoryTargetStatIdentity:
    return MemoryTargetStatIdentity(
        device=info.st_dev,
        inode=info.st_ino,
        mode=info.st_mode,
        uid=info.st_uid,
        gid=info.st_gid,
        links=info.st_nlink,
        size=info.st_size,
        mtime_ns=getattr(info, "st_mtime_ns", None),
        ctime_ns=getattr(info, "st_ctime_ns", None),
    )


def _valid_memory_target(name: str, info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode) or info.st_uid != _uid() or info.st_nlink != 1:
        raise _error("corrupt") from None
    mode = stat.S_IMODE(info.st_mode)
    if name == _RECEIPT_TARGET_NAME:
        if mode != _PRIVATE_FILE_MODE:
            raise _error("corrupt") from None
    elif not (mode & 0o400) or mode & (0o022 | 0o7000):
        raise _error("corrupt") from None


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
    # The construction source is os.geteuid, whose stdlib contract returns int.
    # Patched callables remain called and converted at the original point.
    get_effective: Callable[[], int] | None = getattr(os, "geteuid", None)
    if callable(get_effective):
        return int(get_effective())
    return int(os.getuid())


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
    if not required <= getattr(os, "supports_dir_fd", set[object]()):
        raise MemoryStorageUnavailableError()
    if os.link not in getattr(os, "supports_follow_symlinks", set[object]()):
        raise MemoryStorageUnavailableError()


def _require_lock_gate_capabilities() -> None:
    """Require the duplicate-descriptor flock probe used only by the lock gate."""

    _require_capabilities()
    if (
        not callable(getattr(os, "dup", None))
        or _fcntl is None
        or not isinstance(getattr(_fcntl, "LOCK_NB", None), int)
    ):
        raise MemoryStorageUnavailableError()


def require_memory_lock_gate_capabilities() -> None:
    """Require the supported descriptor and flock capabilities for a lock-only gate."""

    _require_lock_gate_capabilities()


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


class RecoveryDescriptorOpener(Protocol):
    """The descriptor syscall collaborator captured by a recovery consumer."""

    def __call__(
        self, path: str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None,
    ) -> int: ...


def open_recovery_absolute_directory(opener: RecoveryDescriptorOpener, path: str) -> int:
    """Open an absolute directory with the recovery directory flags."""
    return opener(path, _dir_flags())


def open_recovery_relative_directory(
    opener: RecoveryDescriptorOpener, name: str, parent_fd: int,
) -> int:
    """Open a directory relative to the caller-owned parent descriptor."""
    return opener(name, _dir_flags(), dir_fd=parent_fd)


def open_recovery_read_leaf(opener: RecoveryDescriptorOpener, name: str, parent_fd: int) -> int:
    """Open a read leaf relative to the caller-owned parent descriptor."""
    return opener(name, _leaf_flags(), dir_fd=parent_fd)


def create_recovery_exclusive_leaf(
    opener: RecoveryDescriptorOpener, name: str, parent_fd: int,
) -> int:
    """Create a private exclusive leaf relative to the caller-owned parent."""
    return opener(name, _write_flags(), 0o600, dir_fd=parent_fd)


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


def _workspace_memory_directory(info: os.stat_result) -> None:
    """Validate the existing workspace ``memory`` directory without changing it."""

    _private_directory(info, workspace=True)
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o700 != 0o700 or mode & 0o022 or info.st_mode & 0o7000:
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
        if cast(object, fd) is None:  # pragma: no cover - absolute non-root paths always have components.
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


def memory_target_stat_identity(info: os.stat_result) -> MemoryTargetStatIdentity:
    """Describe observed target metadata without issuing a snapshot or authority."""

    return _target_stat_identity(info)


def require_safe_memory_target(name: str, info: os.stat_result) -> None:
    """Validate ordinary-memory or private-receipt target metadata."""

    _valid_memory_target(name, info)


def stat_recovery_descriptor(fd: int, *, code: str = "io") -> os.stat_result:
    """Read owned descriptor metadata with the storage boundary's fixed error mapping."""

    return _fstat(fd, code=code)


def stat_recovery_entry(parent_fd: int, name: str) -> os.stat_result | None:
    """Read one descriptor-relative entry without following its symlink."""

    return _stat_entry(parent_fd, name)


def close_recovery_descriptor(fd: int | None) -> None:
    """Close an owned optional descriptor with the existing cleanup policy."""

    _close(fd)


def _check_directory_binding(
    parent_fd: int,
    name: str,
    child_fd: int,
    *,
    workspace_memory: bool = False,
    expected_mode: int | None = None,
) -> os.stat_result:
    """Prove a directory descriptor and its parent name still refer together."""

    parent_info = _fstat(parent_fd, code="binding")
    if not stat.S_ISDIR(parent_info.st_mode):
        raise _error("binding") from None
    child_info = _fstat(child_fd, code="binding")
    if workspace_memory:
        _workspace_memory_directory(child_info)
    else:
        _private_directory(child_info)
    if expected_mode is not None and stat.S_IMODE(child_info.st_mode) != expected_mode:
        raise _error("binding") from None
    entry = _stat_entry(parent_fd, name)
    if entry is None or not _same_binding(entry, child_info):
        raise _error("binding") from None
    return child_info


def _sync_directory_boundary(
    parent_fd: int,
    name: str,
    child_fd: int,
    *,
    workspace_memory: bool = False,
    expected_mode: int | None = None,
) -> None:
    """Fsync a child and parent, proving bindings around each durability call."""

    _check_directory_binding(
        parent_fd,
        name,
        child_fd,
        workspace_memory=workspace_memory,
        expected_mode=expected_mode,
    )
    _fsync(child_fd)
    _check_directory_binding(
        parent_fd,
        name,
        child_fd,
        workspace_memory=workspace_memory,
        expected_mode=expected_mode,
    )
    _fsync(parent_fd)
    _check_directory_binding(
        parent_fd,
        name,
        child_fd,
        workspace_memory=workspace_memory,
        expected_mode=expected_mode,
    )


def _open_child_directory(
    parent_fd: int,
    name: str,
    *,
    create: bool,
    workspace_memory: bool = False,
    sync_existing: bool = True,
) -> int:
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
        if workspace_memory:
            _workspace_memory_directory(opened)
            expected_mode = stat.S_IMODE(opened.st_mode)
        else:
            _private_directory(opened)
            expected_mode = None
        if not _same_binding(observed, opened):
            raise _error("binding") from None
        # Existing directories use the same repair path as newly-created
        # ones.  A prior failed parent fsync is therefore repaired on retry.
        if sync_existing:
            _sync_directory_boundary(
                parent_fd,
                name,
                child,
                workspace_memory=workspace_memory,
                expected_mode=expected_mode,
            )
        return child
    except BaseException:
        _close(child)
        raise


def open_recovery_private_directory(
    parent_fd: int,
    name: str,
    *,
    create: bool,
    sync_existing: bool = True,
) -> int:
    """Open a checked private recovery directory, retaining binding and durability checks."""

    return _open_child_directory(
        parent_fd, name, create=create, sync_existing=sync_existing,
    )


def _validate_digest(value: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise _error("invalid") from None
    return value


def _validate_evidence_kind(value: str) -> str:
    if type(value) is not str or value not in _EVIDENCE_KINDS:
        raise _error("invalid") from None
    return value


def _validate_record_slot(value: str) -> str:
    if type(value) is not str or _RECORD_SLOT_RE.fullmatch(value) is None:
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


class _ImmutableSlotStore:
    """Shared no-clobber publication for digest and named immutable slots."""

    def __init__(
        self,
        directory_fd: Callable[[], int],
        assert_binding: Callable[[], None],
        sync_directory: Callable[[], None],
    ) -> None:
        self._directory_fd = directory_fd
        self._assert_binding = assert_binding
        self._sync_directory = sync_directory

    def _fd(self) -> int:
        return self._directory_fd()

    def require_entry_binding(self, entry: _Entry) -> None:
        self._entry_binding(entry)

    def read_verified_entry(
        self, name: str, *, content_digest: str | None, expected: bytes | None,
        missing_code: str, conflict_code: str,
        allowed_links: set[int] | frozenset[int], fsync: bool = False,
    ) -> _Entry:
        return self._open_verified_file(
            name, content_digest=content_digest, expected=expected,
            missing_code=missing_code, conflict_code=conflict_code,
            allowed_links=allowed_links, fsync=fsync,
        )

    def require_entry_relationship(self, target: _Entry, companion: _Entry | None) -> None:
        self._validate_relationship(target, companion)

    def create_durable_companion(self, target_name: str, content: bytes) -> _Entry:
        return self._create_companion(target_name, content)

    def require_final_state(
        self, before_target: _Entry, before_companion: _Entry | None,
        after_target: _Entry | None, after_companion: _Entry | None,
    ) -> None:
        self._compare_final_state(
            before_target, before_companion, after_target, after_companion,
        )

    def finish_durable_publication(
        self, target_name: str, *, content_digest: str | None, expected: bytes | None,
        retained_companion_identity: tuple[int, int] | None = None,
    ) -> bytes:
        return self._durable_success(
            target_name, content_digest=content_digest, expected=expected,
            retained_companion_identity=retained_companion_identity,
        )

    def _entry_binding(self, entry: _Entry) -> None:
        current = _stat_entry(self._fd(), entry.name)
        if current is None or not _same_snapshot(current, entry.info):
            raise _error("binding") from None

    def _open_verified_file(
        self,
        name: str,
        *,
        content_digest: str | None,
        expected: bytes | None,
        missing_code: str,
        conflict_code: str,
        allowed_links: set[int] | frozenset[int],
        fsync: bool = False,
    ) -> _Entry:
        directory_fd = self._fd()
        observed = _stat_entry(directory_fd, name)
        if observed is None:
            raise _error(missing_code) from None
        fd: int | None = None
        try:
            fd = os.open(name, _leaf_flags(), dir_fd=directory_fd)
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
            if content_digest is not None and digest_state.hexdigest() != content_digest:
                raise _error(conflict_code) from None
            if expected is not None and position != len(expected):
                raise _error(conflict_code) from None
            current = _stat_entry(directory_fd, name)
            if current is None or not _same_snapshot(current, after):
                raise _error("binding") from None
            if fsync:
                _fsync(fd)
                final = _fstat(fd, code="binding")
                current = _stat_entry(directory_fd, name)
                if current is None or not _same_snapshot(final, after) or not _same_snapshot(
                    current, final
                ):
                    raise _error("binding") from None
            return _Entry(name, _identity(after), after, bytes(result) if expected is None else expected)
        finally:
            _close(fd)

    def _validate_relationship(self, target: _Entry, companion: _Entry | None) -> None:
        """Accept exact original hardlinks or copied independent private files."""

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
                entries = os.listdir(self._fd())
            except (OSError, TypeError, ValueError):
                raise _error("io") from None
            for name in entries:
                info = _stat_entry(self._fd(), name)
                if info is not None and _identity(info) == target.identity:
                    names.add(name)
            if names != {target.name, companion.name}:
                raise _error("corrupt") from None
        elif (
            target.info.st_nlink != 1
            or companion.info.st_nlink != 1
            or target.data != companion.data
        ):
            raise _error("corrupt") from None

    def _compare_presence_state(
        self,
        before_target: _Entry | None,
        before_companion: _Entry | None,
        after_target: _Entry | None,
        after_companion: _Entry | None,
    ) -> None:
        """Keep a missing/read state bound while checking both record names."""

        for before, after in (
            (before_target, after_target),
            (before_companion, after_companion),
        ):
            if (before is None) != (after is None):
                raise _error("binding") from None
            if before is not None and after is not None:
                if (
                    before.identity != after.identity
                    or not _same_snapshot(before.info, after.info)
                    or before.data != after.data
                ):
                    raise _error("binding") from None

    def state(
        self,
        target_name: str,
        *,
        content_digest: str | None,
        expected: bytes | None,
    ) -> tuple[_Entry | None, _Entry | None]:
        directory_fd = self._fd()
        companion_name = _COMPANION_PREFIX + target_name
        target_info = _stat_entry(directory_fd, target_name)
        companion_info = _stat_entry(directory_fd, companion_name)
        target: _Entry | None = None
        companion: _Entry | None = None
        conflict_code = "conflict" if expected is not None else "corrupt"
        if target_info is not None:
            target = self._open_verified_file(
                target_name,
                content_digest=content_digest,
                expected=expected,
                missing_code="missing",
                conflict_code=conflict_code,
                allowed_links={1, 2},
            )
        if companion_info is not None:
            companion = self._open_verified_file(
                companion_name,
                content_digest=content_digest,
                expected=expected,
                missing_code="missing",
                conflict_code=conflict_code,
                allowed_links={1, 2},
            )
        if target is not None and companion is not None:
            self._validate_relationship(target, companion)
        elif target is not None:
            self._validate_relationship(target, None)
        elif companion is not None and companion.info.st_nlink != 1:
            raise _error("corrupt") from None
        return target, companion

    def _create_companion(self, target_name: str, content: bytes) -> _Entry:
        companion_name = _COMPANION_PREFIX + target_name
        content_digest = hashlib.sha256(content).hexdigest()
        fd: int | None = None
        try:
            try:
                fd = os.open(
                    companion_name,
                    _write_flags(),
                    _PRIVATE_FILE_MODE,
                    dir_fd=self._fd(),
                )
            except FileExistsError:
                return self._open_verified_file(
                    companion_name,
                    content_digest=content_digest,
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
            current = _stat_entry(self._fd(), companion_name)
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
            current = _stat_entry(self._fd(), companion_name)
            if current is None or not _same_identity_shape(current, after):
                raise _error("binding") from None
            verified = self._open_verified_file(
                companion_name,
                content_digest=content_digest,
                expected=content,
                missing_code="missing",
                conflict_code="conflict",
                allowed_links={1},
            )
            self._entry_binding(verified)
            self._sync_directory()
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
        target_name: str,
        *,
        content_digest: str | None,
        expected: bytes | None,
        initial_state: tuple[_Entry, _Entry | None] | None = None,
        retained_companion_identity: tuple[int, int] | None = None,
    ) -> bytes:
        if initial_state is None:
            target, companion = self.state(
                target_name, content_digest=content_digest, expected=expected
            )
        else:
            target, companion = initial_state
        if target is None:
            raise _error("missing") from None
        self._validate_relationship(target, companion)
        if retained_companion_identity is not None and (
            companion is None or companion.identity != retained_companion_identity
        ):
            raise _error("binding") from None
        durable_target = self._open_verified_file(
            target_name,
            content_digest=content_digest,
            expected=expected,
            missing_code="missing",
            conflict_code="conflict" if expected is not None else "corrupt",
            allowed_links={1, 2},
            fsync=True,
        )
        before_directory_target, before_directory_companion = self.state(
            target_name, content_digest=content_digest, expected=expected
        )
        if before_directory_target is None:
            raise _error("missing") from None
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
        self._sync_directory()
        after_directory_target, after_directory_companion = self.state(
            target_name, content_digest=content_digest, expected=expected
        )
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
        if after_directory_target is None:
            raise _error("missing") from None
        return after_directory_target.data or b""

    def publish(self, target_name: str, content: bytes, *, content_digest: str | None) -> None:
        self._assert_binding()
        target, companion = self.state(
            target_name, content_digest=content_digest, expected=content
        )
        if target is not None:
            if companion is None:
                self._create_companion(target_name, content)
            self._durable_success(
                target_name,
                content_digest=content_digest,
                expected=content,
            )
            return

        if companion is None:
            companion = self._create_companion(target_name, content)
        elif companion.info.st_nlink != 1:
            raise _error("corrupt") from None
        self._entry_binding(companion)
        self._assert_binding()
        linked = False
        companion_name = _COMPANION_PREFIX + target_name
        try:
            os.link(
                companion_name,
                target_name,
                src_dir_fd=self._fd(),
                dst_dir_fd=self._fd(),
                follow_symlinks=False,
            )
            linked = True
        except FileExistsError:
            target, current_companion = self.state(
                target_name, content_digest=content_digest, expected=content
            )
            if target is None:
                raise _error("conflict") from None
            self._entry_binding(companion)
            if current_companion is None:
                raise _error("binding") from None
        except (OSError, TypeError, ValueError):
            raise _error("io") from None
        self._assert_binding()
        self._durable_success(
            target_name,
            content_digest=content_digest,
            expected=content,
            retained_companion_identity=companion.identity if linked else None,
        )

    def read(
        self,
        target_name: str,
        *,
        content_digest: str | None,
        missing_is_none: bool,
    ) -> bytes | None:
        self._assert_binding()
        target, companion = self.state(
            target_name, content_digest=content_digest, expected=None
        )
        if target is None:
            self._assert_binding()
            confirmed_target, confirmed_companion = self.state(
                target_name, content_digest=content_digest, expected=None
            )
            self._compare_presence_state(
                target,
                companion,
                confirmed_target,
                confirmed_companion,
            )
            self._assert_binding()
            if missing_is_none and companion is None:
                return None
            raise _error("missing") from None
        return self._durable_success(
            target_name,
            content_digest=content_digest,
            expected=None,
            initial_state=(target, companion),
        )


def _check_workspace_memory_lock_binding(workspace_fd: int, lock_fd: int) -> None:
    """Validate the persistent lock and its workspace-relative name."""

    workspace_info = _fstat(workspace_fd, code="binding")
    if not stat.S_ISDIR(workspace_info.st_mode) or workspace_info.st_uid != _uid():
        raise _error("binding") from None
    lock_info = _fstat(lock_fd, code="binding")
    _private_file(lock_info, links={1})
    path_lock = _stat_entry(workspace_fd, _LOCK_NAME)
    if path_lock is None or not _same_binding(path_lock, lock_info):
        raise _error("binding") from None


def _open_or_create_workspace_memory_lock(
    workspace_fd: int, check_binding: Callable[[int], None] | None = None,
    *, create: bool = True,
) -> int:
    """Open or exclusively create the persistent private workspace lock."""

    def validate_binding(lock_fd: int) -> None:
        if check_binding is None:
            _check_workspace_memory_lock_binding(workspace_fd, lock_fd)
        else:
            check_binding(lock_fd)

    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if isinstance(cloexec, int):
        flags |= cloexec
    lock_fd: int | None = None
    for _ in range(8):
        try:
            lock_fd = os.open(_LOCK_NAME, flags, dir_fd=workspace_fd)
            break
        except FileNotFoundError:
            if not create:
                raise _error("missing") from None
            try:
                lock_fd = os.open(
                    _LOCK_NAME,
                    flags | os.O_CREAT | os.O_EXCL,
                    _PRIVATE_FILE_MODE,
                    dir_fd=workspace_fd,
                )
                break
            except FileExistsError:
                continue
            except (OSError, TypeError, ValueError):
                raise _error("io") from None
        except (OSError, TypeError, ValueError):
            raise _error("io") from None
    if lock_fd is None:
        raise _error("io") from None
    try:
        validate_binding(lock_fd)
        _fsync(lock_fd)
        validate_binding(lock_fd)
        _fsync(workspace_fd)
        validate_binding(lock_fd)
        return lock_fd
    except BaseException:
        _close(lock_fd)
        raise


class _WorkspaceMemoryLockGate(AbstractContextManager["_WorkspaceMemoryLockGate"]):
    """A process- and thread-bound lock gate without a recovery-layout policy."""

    def __init__(self, workspace: Path | str, *, existing_only: bool = False) -> None:
        _require_lock_gate_capabilities()
        self._workspace_path = _canonical_workspace(workspace)
        self._existing_only = existing_only
        self._workspace_fd: int | None = None
        self._lock_fd: int | None = None
        self._lock_anchor_fd: int | None = None
        self._workspace_binding: os.stat_result | None = None
        self._lock_binding: os.stat_result | None = None
        self._lock_anchor_binding: os.stat_result | None = None
        self._pid = os.getpid()
        self._thread = threading.current_thread()
        self._used = False
        self._active = False
        self._lock_held = False

    def uses_existing_layout_only(self) -> bool:
        """Observe constructor policy only; this does not establish a live lock."""
        return self._existing_only

    @property
    def workspace_identity(self) -> tuple[int, int]:
        """Identify the locked directory without lending its descriptor.

        Consumers must still require the current lock before using their own
        directory pins; the returned inode identity is not an admission token.
        """
        self.require_current()
        binding = self._workspace_binding
        if binding is None:
            raise _error("closed") from None
        return binding.st_dev, binding.st_ino

    @property
    def workspace(self) -> Path:
        """The resolved workspace alias pinned by this gate."""

        return self._workspace_path

    @property
    def lock_path(self) -> Path:
        """The persistent lock's path in the pinned workspace."""

        return self._workspace_path / _LOCK_NAME

    def __reduce__(self) -> NoReturn:
        raise TypeError("workspace memory lock gate cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("workspace memory lock gate cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("workspace memory lock gate cannot be copied")

    def __enter__(self) -> Self:
        self._require_owner()
        if self._used:
            raise _error("closed") from None
        self._used = True
        workspace_fd: int | None = None
        lock_fd: int | None = None
        lock_anchor_fd: int | None = None
        try:
            _require_lock_gate_capabilities()
            workspace_fd = _open_directory_path(self._workspace_path)
            workspace_info = _fstat(workspace_fd)
            _private_directory(workspace_info, workspace=True)
            self._workspace_binding = workspace_info
            self._workspace_fd = workspace_fd
            workspace_fd = None

            assert self._workspace_fd is not None
            lock_fd = _open_or_create_workspace_memory_lock(
                self._workspace_fd, create=not self._existing_only
            )
            lock_binding = _fstat(lock_fd, code="binding")
            try:
                lock_anchor_fd = os.dup(lock_fd)
            except (OSError, TypeError, ValueError):
                raise _error("io") from None
            lock_anchor_binding = _fstat(lock_anchor_fd, code="binding")
            if not _same_binding(lock_binding, lock_anchor_binding):
                raise _error("binding") from None

            self._lock_fd = lock_fd
            lock_fd = None
            self._lock_binding = lock_binding
            self._lock_anchor_fd = lock_anchor_fd
            lock_anchor_fd = None
            self._lock_anchor_binding = lock_anchor_binding
            try:
                assert _fcntl is not None
                _fcntl.flock(self._lock_anchor_fd, _fcntl.LOCK_EX)
            except (OSError, TypeError, ValueError):
                raise _error("io") from None
            self._lock_held = True
            self._active = True
            # Revalidate the path and original descriptor after blocking flock.
            self.require_current()
            return self
        except MemoryStorageError:
            self._cleanup()
            _close(workspace_fd)
            _close(lock_fd)
            _close(lock_anchor_fd)
            raise
        except (OSError, TypeError, ValueError):
            self._cleanup()
            _close(workspace_fd)
            _close(lock_fd)
            _close(lock_anchor_fd)
            raise _error("io") from None
        except BaseException:
            self._cleanup()
            _close(workspace_fd)
            _close(lock_fd)
            _close(lock_anchor_fd)
            raise

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._require_owner()
        validation_code: str | None = None
        try:
            self.require_current()
        except MemoryStorageError as error:
            validation_code = error.code
        except BaseException:
            validation_code = "binding"

        descriptors_current = self._cleanup()
        if exc_type is None:
            if validation_code is not None:
                raise _error(validation_code) from None
            if not descriptors_current:
                raise _error("binding") from None

    def require_current(self) -> None:
        """Prove that this active gate still refers to its original workspace lock."""

        self._require_owner()
        if not self._active:
            raise _error("closed") from None
        if (
            self._workspace_fd is None
            or self._lock_fd is None
            or self._lock_anchor_fd is None
            or self._workspace_binding is None
            or self._lock_binding is None
            or self._lock_anchor_binding is None
        ):
            raise _error("closed") from None

        workspace_info = _fstat(self._workspace_fd, code="binding")
        if not _same_binding(self._workspace_binding, workspace_info):
            raise _error("binding") from None
        _private_directory(workspace_info, workspace=True)
        rebound = _open_directory_path(self._workspace_path)
        try:
            rebound_info = _fstat(rebound, code="binding")
            if not _same_binding(workspace_info, rebound_info):
                raise _error("binding") from None
        finally:
            _close(rebound)

        anchor_info = _fstat(self._lock_anchor_fd, code="binding")
        if not _same_binding(self._lock_anchor_binding, anchor_info):
            raise _error("binding") from None
        lock_info = _fstat(self._lock_fd, code="binding")
        if not _same_binding(self._lock_binding, lock_info):
            raise _error("binding") from None
        if not self._original_lock_fd_shares_anchor():
            raise _error("binding") from None
        _check_workspace_memory_lock_binding(self._workspace_fd, self._lock_fd)

    def close(self) -> None:
        """Release this gate and detach its descriptors without removing storage."""

        self._require_owner()
        if not self._cleanup():
            raise _error("binding") from None

    def _require_owner(self) -> None:
        if self._pid != os.getpid() or self._thread is not threading.current_thread():
            raise _error("closed") from None

    def _cleanup(self) -> bool:
        """Detach state, then release only descriptors still bound to this gate.

        Device, inode, and file type detect closed or reused descriptor numbers.
        The anchor probe detects a sequential same-inode reopen of the
        original descriptor. An uncooperative thread can still race raw
        ``os.close`` after that probe, or replace the anchor itself; those
        actions cannot be made atomic within this cooperating-process boundary.
        """

        lock_fd = self._lock_fd
        lock_anchor_fd = self._lock_anchor_fd
        workspace_fd = self._workspace_fd
        lock_binding = self._lock_binding
        lock_anchor_binding = self._lock_anchor_binding
        workspace_binding = self._workspace_binding
        lock_held = self._lock_held
        self._lock_fd = None
        self._lock_anchor_fd = None
        self._workspace_fd = None
        self._lock_binding = None
        self._lock_anchor_binding = None
        self._workspace_binding = None
        self._lock_held = False
        self._active = False
        self._used = True

        lock_current = self._descriptor_matches(lock_fd, lock_binding)
        anchor_current = self._descriptor_matches(lock_anchor_fd, lock_anchor_binding)
        workspace_current = self._descriptor_matches(workspace_fd, workspace_binding)
        pair_absent = (
            lock_fd is None
            and lock_anchor_fd is None
            and lock_binding is None
            and lock_anchor_binding is None
            and not lock_held
        )
        lock_pair_current = pair_absent
        if (
            lock_held
            and lock_fd is not None
            and lock_anchor_fd is not None
            and lock_current
            and anchor_current
        ):
            lock_pair_current = self._original_fd_shares_anchor(lock_fd, lock_anchor_fd)

        # Validate both saved descriptors before any unlock or close. The
        # anchor owns flock; the original fd is closed only after its open-file
        # description is proved to share the anchor's lock.
        if lock_anchor_fd is not None and anchor_current:
            if lock_held and _fcntl is not None:
                try:
                    _fcntl.flock(lock_anchor_fd, _fcntl.LOCK_UN)
                except (OSError, TypeError, ValueError):
                    pass
            _close(lock_anchor_fd)
        if lock_fd is not None and lock_pair_current:
            _close(lock_fd)
        if workspace_fd is not None and workspace_current:
            _close(workspace_fd)
        return lock_pair_current and anchor_current and workspace_current

    def _original_lock_fd_shares_anchor(self) -> bool:
        if self._lock_fd is None or self._lock_anchor_fd is None:
            return False
        return self._original_fd_shares_anchor(self._lock_fd, self._lock_anchor_fd)

    @staticmethod
    def _original_fd_shares_anchor(lock_fd: int, lock_anchor_fd: int) -> bool:
        """Use the held anchor lock to test whether the original fd is its dup."""

        if _fcntl is None:
            return False
        try:
            lock_info = os.fstat(lock_fd)
            anchor_info = os.fstat(lock_anchor_fd)
        except (OSError, TypeError, ValueError):
            return False
        if not _same_binding(lock_info, anchor_info):
            return False
        try:
            _fcntl.flock(lock_fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except (OSError, TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _descriptor_matches(
        fd: int | None, binding: os.stat_result | None
    ) -> bool:
        if fd is None or binding is None:
            return fd is None and binding is None
        try:
            current = os.fstat(fd)
        except (OSError, TypeError, ValueError):
            return False
        return _same_binding(binding, current)


WorkspaceMemoryLockGate = _WorkspaceMemoryLockGate


def open_workspace_memory_lock(workspace: Path | str) -> _WorkspaceMemoryLockGate:
    """Return a single-use lock gate without requiring memory recovery storage."""

    return _WorkspaceMemoryLockGate(workspace)


def open_existing_workspace_memory_lock(workspace: Path | str) -> _WorkspaceMemoryLockGate:
    """Acquire an existing lock without creating a workspace or recovery layout."""

    return _WorkspaceMemoryLockGate(workspace, existing_only=True)


@dataclass(frozen=True, slots=True)
class _OperationHooks:
    operation: WorkspaceMemoryOperation
    check_owner: Callable[[], None]
    check_new: Callable[[], None]
    check_issued: Callable[[int], None]
    check_current: Callable[[int], None]
    check_bound_current: Callable[[int], None]
    activate: Callable[[int, tuple[int, int]], None]
    rollback: Callable[[], None]
    identity: Callable[[], tuple[int, int]]
    check_exit: Callable[[], bool]
    finish: Callable[[], None]
    add_parent: Callable[[BoundMemoryTarget], None]
    remove_parent: Callable[[BoundMemoryTarget], None]
    check_parent: Callable[[BoundMemoryTarget], None]


@dataclass(frozen=True, slots=True)
class _OperationActions:
    enter: Callable[[], None]
    require_current: Callable[[], None]
    workspace_identity: Callable[[], tuple[int, int]]
    exit: Callable[[], None]
    borrow_target: Callable[[MemoryTargetSnapshot], BoundMemoryTarget]
    borrow_receipt: Callable[[MemoryTargetSnapshot], BoundMemoryTarget]


@dataclass(frozen=True, slots=True)
class _BoundHooks:
    bound: BoundMemoryTarget
    check_owner: Callable[[], None]
    check_new: Callable[[], None]
    check_epoch: Callable[[int], None]
    check_snapshot: Callable[[int], None]
    activate: Callable[[int, tuple[int, int]], None]
    rollback: Callable[[], None]
    check_registered: Callable[[], None]
    check_parent_present: Callable[[], None]
    check_local_current: Callable[[], None]
    duplicate_parent: Callable[[], int]
    check_exit: Callable[[], bool]
    finish: Callable[[], None]
    close_parent: Callable[[], None]


@dataclass(frozen=True, slots=True)
class _BoundActions:
    enter: Callable[[], None]
    require_current: Callable[[], None]
    duplicate_dir_fd: Callable[[], int]
    snapshot_target: Callable[[], MemoryTargetSnapshot]
    exit: Callable[[], None]


@dataclass(frozen=True, slots=True)
class _NamespaceActions:
    check_process: Callable[[], None]
    check_binding: Callable[[], None]
    open_children: Callable[[str, str], tuple[int, int]]
    check_kind_binding: Callable[[str, os.stat_result], None]
    register: Callable[[ImmutableEvidenceNamespace], None]
    unregister: Callable[[ImmutableEvidenceNamespace], None]


_OperationActionBinder = Callable[[_OperationHooks], _OperationActions]
_AuthorityBoundActionBinder = Callable[
    [_OperationHooks, _BoundHooks, int, bool, str], _BoundActions
]
_OperationBoundActionBinder = Callable[[_BoundHooks, bool, str], _BoundActions]


@dataclass(frozen=True, slots=True, init=False)
class _BoundInitialization:
    retain_owner: Callable[[WorkspaceMemoryAuthority], None]
    capture_validator: Callable[[_BoundInitialization], None]
    retain_validator: Callable[[Callable[[MemoryTargetSnapshot, int, bool], None]], None]
    retain_thread: Callable[[threading.Thread], None]
    retain_epoch: Callable[[int | None], None]


@dataclass(frozen=True, slots=True, init=False)
class _OperationBindingPush:
    apply: Callable[[_OperationActionBinder], None]


@dataclass(frozen=True, slots=True, init=False)
class _OperationTargetBindingPush:
    apply: Callable[[_AuthorityBoundActionBinder], _BoundActions]


@dataclass(frozen=True, slots=True, init=False)
class _BoundActionBindingPush:
    apply: Callable[[_OperationBoundActionBinder], None]


class WorkspaceMemoryAuthority(AbstractContextManager["WorkspaceMemoryAuthority"]):
    """A process-local authority for one private workspace memory namespace.

    Constructing this object does not create files.  Entering the context
    acquires the persistent workspace lock, creates or validates the private
    layout, and pins descriptors for the operation.  ``publish_blob`` and
    ``read_blob`` require the context to be active.
    """

    def __init__(self, workspace: Path | str, *, existing_only: bool = False) -> None:
        _require_capabilities()
        self._workspace_path = _canonical_workspace(workspace)
        self._existing_only = existing_only
        self._workspace_fd: int | None = None
        self._lock_fd: int | None = None
        self._memory_fd: int | None = None
        self._recovery_fd: int | None = None
        self._blobs_fd: int | None = None
        self._digest_fd: int | None = None
        self._workspace_identity: tuple[int, int] | None = None
        self._lock_identity: tuple[int, int] | None = None
        self._memory_mode: int | None = None
        self._pid = os.getpid()
        self._active = False
        self._lock_held = False
        # Weakref callbacks may run synchronously during cyclic GC while a
        # lifecycle operation holds this lock, so they must be able to reenter.
        self._lifecycle_lock = threading.RLock()
        self._entering = False
        self._activation_thread: threading.Thread | None = None
        self._activation_thread_id: int | None = None
        self._activation_epoch = 0
        self._borrowers: set[WorkspaceMemoryOperation] = set()
        self._namespaces: list[ImmutableEvidenceNamespace] = []
        self._target_snapshot_issuance: dict[
            int,
            tuple[
                weakref.ReferenceType[MemoryTargetSnapshot],
                _TargetSnapshotSeal,
            ],
        ] = {}
        self._blob_store = _ImmutableSlotStore(
            self._digest_fd_or_closed,
            self._assert_binding,
            self._sync_digest_directory,
        )

    def uses_existing_layout_only(self) -> bool:
        """Observe constructor policy only; this does not issue or admit a borrow."""
        return self._existing_only

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

    def operation(self) -> WorkspaceMemoryOperation:
        """Issue one opaque lease for the current authority activation."""

        self._require_process_owner()
        with self._lifecycle_lock:
            self._require_process_owner()
            return WorkspaceMemoryOperation(
                self,
                _capability=_OPERATION_CAPABILITY,
                _issued_epoch=self._activation_epoch,
                _thread=threading.current_thread(),
            )

    def _check_operation_locked(self, hooks: _OperationHooks, *, bound: bool = False) -> None:
        hooks.check_owner()
        if threading.current_thread() is not self._activation_thread:
            raise _error("closed") from None
        if bound:
            hooks.check_bound_current(self._activation_epoch)
        else:
            hooks.check_current(self._activation_epoch)
        if not self._active or not self._lock_held or hooks.operation not in self._borrowers:
            raise _error("closed") from None

    def _bind_operation_actions(self, hooks: _OperationHooks) -> _OperationActions:
        # The constructor invokes this binder only after capability qualification.
        def enter() -> None:
            self._require_process_owner()
            with self._lifecycle_lock:
                self._require_process_owner()
                hooks.check_new()
                hooks.check_owner()
                if threading.current_thread() is not self._activation_thread:
                    raise _error("closed") from None
                if (
                    not self._active or not self._lock_held
                    or self._workspace_identity is None
                ):
                    raise _error("closed") from None
                epoch = self._activation_epoch
                hooks.check_issued(epoch)
                self._assert_binding()
                if not self._active or not self._lock_held or self._activation_epoch != epoch:
                    raise _error("closed") from None
                hooks.check_issued(epoch)
                identity = self._workspace_identity
                try:
                    self._register_operation(hooks.operation)
                    hooks.activate(epoch, identity)
                except BaseException:
                    self._unregister_operation(hooks.operation)
                    hooks.rollback()
                    raise

        def current() -> None:
            self._require_process_owner()
            with self._lifecycle_lock:
                self._require_process_owner()
                self._check_operation_locked(hooks)
                self._assert_binding()

        def identity() -> tuple[int, int]:
            self._require_process_owner()
            with self._lifecycle_lock:
                self._require_process_owner()
                hooks.check_owner()
                return hooks.identity()

        def exit_operation() -> None:
            self._require_process_owner()
            with self._lifecycle_lock:
                self._require_process_owner()
                if not hooks.check_exit():
                    return
                hooks.finish()
                self._unregister_operation(hooks.operation)

        def borrow(snapshot: MemoryTargetSnapshot, receipt: bool) -> BoundMemoryTarget:
            self._require_process_owner()
            with self._lifecycle_lock:
                self._require_process_owner()
                self._check_operation_locked(hooks)
                epoch = self._activation_epoch
                self._validate_snapshot(snapshot, epoch, receipt)
                if receipt and (snapshot.name != _RECEIPT_TARGET_NAME or not snapshot.exists):
                    raise _error("invalid") from None
                self._assert_binding()

                return BoundMemoryTarget(
                    hooks.operation, snapshot,
                    _capability=_BOUND_RECEIPT_TARGET_CAPABILITY if receipt else _BOUND_TARGET_CAPABILITY,
                )

        def borrow_target(snapshot: MemoryTargetSnapshot) -> BoundMemoryTarget:
            return borrow(snapshot, False)

        def borrow_receipt(snapshot: MemoryTargetSnapshot) -> BoundMemoryTarget:
            return borrow(snapshot, True)

        return _OperationActions(enter, current, identity, exit_operation, borrow_target, borrow_receipt)

    def _bind_bound_actions(
        self, operation: _OperationHooks, hooks: _BoundHooks, epoch: int, receipt: bool,
        name: str,
    ) -> _BoundActions:
        def operation_current() -> None:
            hooks.check_owner()
            self._check_operation_locked(operation, bound=True)
            hooks.check_epoch(epoch)
            # The bound scope retains the epoch from its issuance, not reentry.
            if epoch != self._activation_epoch:
                raise _error("closed") from None

        def current_locked() -> None:
            operation_current()
            hooks.check_registered()
            operation.check_parent(hooks.bound)
            hooks.check_snapshot(epoch)
            hooks.check_parent_present()
            self._assert_binding()
            hooks.check_local_current()

        def enter() -> None:
            self._require_process_owner()
            with self._lifecycle_lock:
                self._require_process_owner()
                hooks.check_new()
                operation_current()
                hooks.check_snapshot(epoch)
                self._assert_binding()
                source_fd = self._memory_fd
                if source_fd is None:
                    raise _error("closed") from None
                try:
                    memory_fd = os.dup(source_fd)
                except (OSError, TypeError, ValueError):
                    raise _error("io") from None
                try:
                    source_info = _fstat(source_fd, code="binding")
                    duplicate_info = _fstat(memory_fd, code="binding")
                    _workspace_memory_directory(duplicate_info)
                    if not _same_binding(source_info, duplicate_info):
                        raise _error("binding") from None
                    self._assert_binding()
                    try:
                        operation.add_parent(hooks.bound)
                        hooks.activate(memory_fd, _identity(duplicate_info))
                    except BaseException:
                        operation.remove_parent(hooks.bound)
                        hooks.rollback()
                        raise
                except BaseException:
                    _close(memory_fd)
                    raise

        def current() -> None:
            self._require_process_owner()
            with self._lifecycle_lock:
                self._require_process_owner()
                current_locked()

        def duplicate() -> int:
            self._require_process_owner()
            with self._lifecycle_lock:
                self._require_process_owner()
                current_locked()
                return hooks.duplicate_parent()

        def snapshot() -> MemoryTargetSnapshot:
            current()
            # Preserve the original reader's two separately observable validations.
            issued = self.snapshot_target(name)
            self._require_process_owner()
            with self._lifecycle_lock:
                self._require_process_owner()
                current_locked()
                self._validate_snapshot(issued, epoch, receipt)
                if issued.name != name:
                    raise _error("binding") from None
                return issued

        def exit_bound() -> None:
            self._require_process_owner()
            with self._lifecycle_lock:
                self._require_process_owner()
                operation.check_owner()
                if not hooks.check_exit():
                    return
                hooks.finish()
                operation.remove_parent(hooks.bound)
                hooks.close_parent()

        return _BoundActions(enter, current, duplicate, snapshot, exit_bound)

    def _namespace_actions(self) -> _NamespaceActions:
        return _NamespaceActions(
            self._assert_process,
            self._assert_binding,
            self._open_namespace_children,
            self._check_namespace_kind_binding,
            self._register_namespace,
            self._unregister_namespace,
        )

    def _open_namespace_children(self, kind: str, identity: str) -> tuple[int, int]:
        kind_fd: int | None = None
        record_fd: int | None = None
        try:
            kind_fd = _open_child_directory(
                self._recovery_fd_or_closed(), kind,
                create=not self._existing_only, sync_existing=not self._existing_only,
            )
            record_fd = _open_child_directory(
                kind_fd, identity,
                create=not self._existing_only, sync_existing=not self._existing_only,
            )
            return kind_fd, record_fd
        except BaseException:
            _close(record_fd)
            _close(kind_fd)
            raise

    def _check_namespace_kind_binding(self, kind: str, info: os.stat_result) -> None:
        entry = _stat_entry(self._recovery_fd_or_closed(), kind)
        if entry is None or not _same_binding(entry, info):
            raise _error("binding") from None

    def open_evidence_namespace(
        self, kind: str, identity: str
    ) -> ImmutableEvidenceNamespace:
        """Open the private immutable record namespace for one chain identity."""

        kind = _validate_evidence_kind(kind)
        identity = _validate_digest(identity)
        self._assert_process()
        self._assert_binding()
        return ImmutableEvidenceNamespace(
            self,
            kind,
            identity,
            _capability=_NAMESPACE_CAPABILITY,
        )

    open_namespace = open_evidence_namespace

    def __reduce__(self) -> NoReturn:
        raise TypeError("memory recovery authority cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("memory recovery authority cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("memory recovery authority cannot be copied")

    def __enter__(self) -> Self:
        self._require_process_owner()
        activating_thread = threading.current_thread()
        with self._lifecycle_lock:
            self._require_process_owner()
            if self._active or self._entering:
                raise _error("closed") from None
            self._entering = True
        workspace_fd: int | None = None
        lock_fd: int | None = None
        try:
            _require_capabilities()
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
            self._memory_fd = _open_child_directory(
                self._workspace_fd,
                _MEMORY_NAME,
                create=not self._existing_only,
                workspace_memory=True,
                sync_existing=not self._existing_only,
            )
            self._memory_mode = stat.S_IMODE(_fstat(self._memory_fd).st_mode)
            self._recovery_fd = _open_child_directory(
                self._memory_fd, _RECOVERY_NAME, create=not self._existing_only,
                sync_existing=not self._existing_only,
            )
            self._blobs_fd = _open_child_directory(
                self._recovery_fd, _BLOBS_NAME, create=not self._existing_only,
                sync_existing=not self._existing_only,
            )
            self._digest_fd = _open_child_directory(
                self._blobs_fd, _DIGEST_DIR_NAME, create=not self._existing_only,
                sync_existing=not self._existing_only,
            )
            self._assert_binding()
            self._require_process_owner()
            with self._lifecycle_lock:
                if not self._entering or self._active:
                    raise _error("closed") from None
                self._activation_thread = activating_thread
                # Diagnostic only; authorization uses the retained thread object.
                self._activation_thread_id = threading.get_ident()
                self._activation_epoch += 1
                self._active = True
                self._entering = False
                return self
        except MemoryStorageError:
            self._require_process_owner()
            with self._lifecycle_lock:
                self._cleanup()
                self._entering = False
            _close(workspace_fd)
            _close(lock_fd)
            raise
        except (OSError, TypeError, ValueError):
            self._require_process_owner()
            with self._lifecycle_lock:
                self._cleanup()
                self._entering = False
            _close(workspace_fd)
            _close(lock_fd)
            raise _error("io") from None
        except BaseException:
            self._require_process_owner()
            with self._lifecycle_lock:
                self._cleanup()
                self._entering = False
            _close(workspace_fd)
            _close(lock_fd)
            raise

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._require_process_owner()
        with self._lifecycle_lock:
            self._require_process_owner()
            if self._borrowers:
                raise _error("borrowed") from None
            if self._entering:
                raise _error("closed") from None
            self._cleanup()

    def close(self) -> None:
        """Release the authority and all descriptors; never remove storage."""

        self._require_process_owner()
        with self._lifecycle_lock:
            self._require_process_owner()
            if self._borrowers:
                raise _error("borrowed") from None
            if self._entering:
                raise _error("closed") from None
            self._cleanup()

    def _cleanup(self) -> None:
        """Release authority state; callers hold ``_lifecycle_lock``."""

        self._active = False
        self._activation_thread = None
        self._activation_thread_id = None
        self._memory_mode = None
        self._target_snapshot_issuance.clear()
        for namespace in tuple(self._namespaces):
            namespace.close()
        self._namespaces.clear()
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

    def _require_process_owner(self) -> None:
        if self._pid != os.getpid():
            raise _error("closed") from None

    def _register_operation(self, operation: WorkspaceMemoryOperation) -> None:
        self._borrowers.add(operation)

    def _unregister_operation(self, operation: WorkspaceMemoryOperation) -> None:
        self._borrowers.discard(operation)

    def _open_or_create_lock(self) -> int:
        assert self._workspace_fd is not None
        return _open_or_create_workspace_memory_lock(
            self._workspace_fd, self._check_lock_binding, create=not self._existing_only,
        )

    def _check_lock_binding(self, lock_fd: int) -> None:
        assert self._workspace_fd is not None
        _check_workspace_memory_lock_binding(self._workspace_fd, lock_fd)

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
            (self._workspace_fd, _MEMORY_NAME, self._memory_fd, True, self._memory_mode),
            (self._memory_fd, _RECOVERY_NAME, self._recovery_fd, False, None),
            (self._recovery_fd, _BLOBS_NAME, self._blobs_fd, False, None),
            (self._blobs_fd, _DIGEST_DIR_NAME, self._digest_fd, False, None),
        )
        for parent, name, child, workspace_memory, expected_mode in parents:
            if parent is None or child is None:
                raise _error("closed") from None
            try:
                child_info = _fstat(child, code="binding")
            except MemoryStorageError:
                raise
            if workspace_memory:
                _workspace_memory_directory(child_info)
            else:
                _private_directory(child_info)
            if expected_mode is not None and stat.S_IMODE(child_info.st_mode) != expected_mode:
                raise _error("binding") from None
            entry = _stat_entry(parent, name)
            if entry is None or not _same_binding(entry, child_info):
                raise _error("binding") from None

    def _entry_binding(self, entry: _Entry) -> None:
        self._blob_store.require_entry_binding(entry)

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
        return self._blob_store.read_verified_entry(
            name,
            content_digest=digest,
            expected=expected,
            missing_code=missing_code,
            conflict_code=conflict_code,
            allowed_links=allowed_links,
            fsync=fsync,
        )

    def _validate_relationship(self, target: _Entry, companion: _Entry | None) -> None:
        self._blob_store.require_entry_relationship(target, companion)

    def _state(self, digest: str, *, expected: bytes | None) -> tuple[_Entry | None, _Entry | None]:
        return self._blob_store.state(digest, content_digest=digest, expected=expected)

    def _digest_fd_or_closed(self) -> int:
        if self._digest_fd is None:
            raise _error("closed") from None
        return self._digest_fd

    def _recovery_fd_or_closed(self) -> int:
        if self._recovery_fd is None:
            raise _error("closed") from None
        return self._recovery_fd

    def _unregister_namespace(self, namespace: ImmutableEvidenceNamespace) -> None:
        try:
            self._namespaces.remove(namespace)
        except ValueError:
            pass

    def _register_namespace(self, namespace: ImmutableEvidenceNamespace) -> None:
        self._namespaces.append(namespace)

    def _sync_digest_directory(self) -> None:
        digest_fd = self._digest_fd_or_closed()
        self._assert_binding()
        _fsync(digest_fd)
        self._assert_binding()

    def _create_companion(self, digest: str, content: bytes) -> _Entry:
        return self._blob_store.create_durable_companion(digest, content)

    def _compare_final_state(
        self,
        before_target: _Entry,
        before_companion: _Entry | None,
        after_target: _Entry | None,
        after_companion: _Entry | None,
    ) -> None:
        self._blob_store.require_final_state(
            before_target,
            before_companion,
            after_target,
            after_companion,
        )

    def _durable_success(
        self,
        digest: str,
        content: bytes | None,
        *,
        retained_companion_identity: tuple[int, int] | None = None,
    ) -> bytes:
        return self._blob_store.finish_durable_publication(
            digest,
            content_digest=digest,
            expected=content,
            retained_companion_identity=retained_companion_identity,
        )

    def _validate_snapshot(
        self,
        snapshot: MemoryTargetSnapshot,
        epoch: int,
        allow_receipt: bool,
    ) -> None:
        if type(snapshot) is not MemoryTargetSnapshot:
            raise _error("invalid") from None
        MemoryTargetSnapshot.qualify_for_authority(
            snapshot, self, epoch, allow_receipt, _capability=_SNAPSHOT_CAPABILITY,
        )

    def require_issued_snapshot(
        self, snapshot: MemoryTargetSnapshot, *, _capability: object | None = None,
    ) -> None:
        if _capability is not _SNAPSHOT_CAPABILITY:
            raise _error("invalid") from None
        self._require_issued_target_snapshot_locked(snapshot)

    def capture_bound_validator(
        self, hooks: _BoundInitialization, *, _capability: object | None = None,
    ) -> None:
        if (_capability is not _BOUND_TARGET_CAPABILITY
                and _capability is not _BOUND_RECEIPT_TARGET_CAPABILITY):
            raise _error("invalid") from None
        if type(hooks) is not _BoundInitialization:
            raise _error("invalid") from None
        validator = self._validate_snapshot
        hooks.retain_validator(validator)

    def initialize_operation_actions(
        self, push: _OperationBindingPush, *, _capability: object | None = None,
    ) -> None:
        if _capability is not _OPERATION_CAPABILITY:
            raise _error("invalid") from None
        if type(push) is not _OperationBindingPush:
            raise _error("invalid") from None
        binder = self._bind_operation_actions
        push.apply(binder)

    def bind_operation_target_actions(
        self, push: _OperationTargetBindingPush, *, _capability: object | None = None,
    ) -> _BoundActions:
        if _capability is not _BOUND_TARGET_CAPABILITY:
            raise _error("invalid") from None
        if type(push) is not _OperationTargetBindingPush:
            raise _error("invalid") from None
        binder = self._bind_bound_actions
        return push.apply(binder)

    def bind_namespace_actions(
        self, *, _capability: object | None = None,
    ) -> _NamespaceActions:
        if _capability is not _NAMESPACE_CAPABILITY:
            raise _error("invalid") from None
        return self._namespace_actions()


    def _register_target_snapshot_locked(self, snapshot: MemoryTargetSnapshot) -> None:
        """Seal an issued snapshot without retaining it after callers release it."""

        snapshot_id = id(snapshot)
        authority_ref = weakref.ref(self)

        def forget(reference: weakref.ReferenceType[MemoryTargetSnapshot]) -> None:
            authority = authority_ref()
            if authority is None or authority._pid != os.getpid():
                return
            with authority._lifecycle_lock:
                if authority._pid != os.getpid():
                    return
                issued = authority._target_snapshot_issuance.get(snapshot_id)
                if issued is not None and issued[0] is reference:
                    del authority._target_snapshot_issuance[snapshot_id]

        reference = weakref.ref(snapshot, forget)
        self._target_snapshot_issuance[snapshot_id] = (reference, _target_snapshot_seal(snapshot))

    def _require_issued_target_snapshot_locked(
        self, snapshot: MemoryTargetSnapshot
    ) -> None:
        """Authenticate object identity and unchanged fields for one issued snapshot."""

        issued = self._target_snapshot_issuance.get(id(snapshot))
        if issued is None or issued[0]() is not snapshot:
            raise _error("invalid") from None
        if issued[1] != _target_snapshot_seal(snapshot):
            raise _error("invalid") from None

    def snapshot_target(self, name: str) -> MemoryTargetSnapshot:
        """Read an allowlisted legacy target without modifying workspace state.

        Reads are capped at 64 MiB. Larger targets fail closed; the bound is a
        conservative qualification limit, not a claim that every legacy
        history size is supported.
        """

        self._require_process_owner()
        with self._lifecycle_lock:
            self._require_process_owner()
            epoch = self._activation_epoch
            if (
                not self._active
                or not self._lock_held
                or epoch <= 0
                or threading.current_thread() is not self._activation_thread
                or self._memory_fd is None
            ):
                raise _error("closed") from None
            if type(name) is not str or name not in _MEMORY_TARGET_NAMES:
                raise _error("invalid") from None
            self._assert_binding()
            memory_fd = self._memory_fd
            observed = _stat_entry(memory_fd, name)
            if observed is None:
                # Confirm absence after proving the pinned parent binding. This
                # distinguishes a stable absence from a create-at-the-boundary.
                self._assert_binding()
                if _stat_entry(memory_fd, name) is not None:
                    raise _error("binding") from None
                content = None
                identity = None
            else:
                _valid_memory_target(name, observed)
                if observed.st_size > _MEMORY_TARGET_MAX_BYTES:
                    raise _error("corrupt") from None
                try:
                    target_fd = os.open(name, _leaf_flags(), dir_fd=memory_fd)
                except OSError as error:
                    if error.errno == errno.ENOENT:
                        code = "binding"
                    elif error.errno in (
                        errno.EACCES,
                        errno.EISDIR,
                        errno.ELOOP,
                        errno.ENOTDIR,
                        errno.EPERM,
                    ):
                        code = "corrupt"
                    else:
                        code = "io"
                    raise _error(code) from None
                except (TypeError, ValueError):
                    raise _error("io") from None
                try:
                    before = _fstat(target_fd, code="binding")
                    before_identity = _target_stat_identity(before)
                    if before_identity != _target_stat_identity(observed):
                        raise _error("binding") from None
                    _valid_memory_target(name, before)
                    content_buffer = bytearray()
                    while True:
                        remaining = _MEMORY_TARGET_MAX_BYTES - len(content_buffer)
                        try:
                            chunk = os.read(target_fd, min(_READ_CHUNK, remaining + 1))
                        except (OSError, TypeError, ValueError):
                            raise _error("io") from None
                        if not chunk:
                            break
                        if len(chunk) > remaining:
                            raise _error("corrupt") from None
                        content_buffer.extend(chunk)
                    after = _fstat(target_fd, code="binding")
                    after_identity = _target_stat_identity(after)
                    if before_identity != after_identity or len(content_buffer) != after.st_size:
                        raise _error("binding") from None
                    _valid_memory_target(name, after)
                    current = _stat_entry(memory_fd, name)
                    if current is None or _target_stat_identity(current) != after_identity:
                        raise _error("binding") from None
                    _valid_memory_target(name, current)
                    content = bytes(content_buffer)
                    identity = after_identity
                finally:
                    _close(target_fd)
            self._assert_binding()
            if (
                not self._active
                or not self._lock_held
                or self._activation_epoch != epoch
                or threading.current_thread() is not self._activation_thread
            ):
                raise _error("closed") from None
            snapshot = MemoryTargetSnapshot(
                name,
                content,
                identity,
                self,
                epoch,
                _capability=_SNAPSHOT_CAPABILITY,
            )
            self._register_target_snapshot_locked(snapshot)
            return snapshot

    def publish_blob(self, content: bytes) -> str:
        """Publish bytes under their exact SHA-256 name without clobbering."""

        self._assert_process()
        if type(content) is not bytes:
            raise _error("invalid") from None
        digest = hashlib.sha256(content).hexdigest()
        self._blob_store.publish(digest, content, content_digest=digest)
        return digest

    # Small names make the typed boundary convenient without adding another
    # storage abstraction.  They all retain the same context/lock semantics.
    publish = publish_blob
    put = publish_blob

    def read_blob(self, digest: str) -> bytes:
        """Read and fully verify one immutable target blob."""

        digest = _validate_digest(digest)
        self._assert_process()
        content = self._blob_store.read(digest, content_digest=digest, missing_is_none=False)
        assert content is not None
        return content

    read = read_blob
    get = read_blob


class WorkspaceMemoryOperation(AbstractContextManager["WorkspaceMemoryOperation"]):
    """An opaque, one-shot lease on one authority activation."""

    _NEW = "new"
    _ACTIVE = "active"
    _CLOSED = "closed"

    _actions: _OperationActions

    def __init__(
        self,
        authority: WorkspaceMemoryAuthority,
        *,
        _capability: object | None = None,
        _issued_epoch: int | None = None,
        _thread: threading.Thread | None = None,
    ) -> None:
        if _capability is not _OPERATION_CAPABILITY:
            raise _error("invalid") from None
        if _issued_epoch is None or _thread is None:
            raise _error("invalid") from None
        self._authority = authority
        self._pid = os.getpid()
        self._thread = _thread
        self._issued_epoch = _issued_epoch
        self._state = self._NEW
        self._registered = False
        self._epoch: int | None = None
        self._workspace_identity: tuple[int, int] | None = None
        self._target_parent_borrowers: set[BoundMemoryTarget] = set()
        self._hooks = _OperationHooks(
            self, self._require_owner, self._check_new, self._check_issued, self._check_current,
            self._check_bound_current, self._activate, self._rollback,
            self._identity, self._check_exit, self._finish,
            self._add_parent, self._remove_parent, self._check_parent,
        )
        # Derive actions from the exact retained owner, never a selectable binder.
        binding_push = WorkspaceMemoryOperation._operation_binding_push(
            self, _capability=_OPERATION_CAPABILITY,
        )
        WorkspaceMemoryAuthority.initialize_operation_actions(
            authority, binding_push, _capability=_OPERATION_CAPABILITY,
        )

    def _operation_binding_push(
        self, *, _capability: object | None = None,
    ) -> _OperationBindingPush:
        if _capability is not _OPERATION_CAPABILITY:
            raise _error("invalid") from None

        def apply(binder: _OperationActionBinder) -> None:
            self._actions = binder(self._hooks)

        push = object.__new__(_OperationBindingPush)
        object.__setattr__(push, "apply", apply)
        return push

    def _target_binding_push(
        self, hooks: _BoundHooks, receipt: bool, name: str,
        *, _capability: object | None = None,
    ) -> _OperationTargetBindingPush:
        if _capability is not _BOUND_TARGET_CAPABILITY:
            raise _error("invalid") from None

        def apply(binder: _AuthorityBoundActionBinder) -> _BoundActions:
            return binder(self._hooks, hooks, self._epoch or 0, receipt, name)

        push = object.__new__(_OperationTargetBindingPush)
        object.__setattr__(push, "apply", apply)
        return push

    def initialize_bound_owner(
        self, hooks: _BoundInitialization, *, _capability: object | None = None,
    ) -> None:
        if (_capability is not _BOUND_TARGET_CAPABILITY
                and _capability is not _BOUND_RECEIPT_TARGET_CAPABILITY):
            raise _error("invalid") from None
        if type(hooks) is not _BoundInitialization:
            raise _error("invalid") from None
        hooks.retain_owner(self._authority)
        hooks.capture_validator(hooks)

    def initialize_bound_context(
        self, hooks: _BoundInitialization, *, _capability: object | None = None,
    ) -> None:
        if (_capability is not _BOUND_TARGET_CAPABILITY
                and _capability is not _BOUND_RECEIPT_TARGET_CAPABILITY):
            raise _error("invalid") from None
        if type(hooks) is not _BoundInitialization:
            raise _error("invalid") from None
        hooks.retain_thread(self._thread)
        hooks.retain_epoch(self._epoch)

    def initialize_bound_actions(
        self, push: _BoundActionBindingPush, *, _capability: object | None = None,
    ) -> None:
        if (_capability is not _BOUND_TARGET_CAPABILITY
                and _capability is not _BOUND_RECEIPT_TARGET_CAPABILITY):
            raise _error("invalid") from None
        if type(push) is not _BoundActionBindingPush:
            raise _error("invalid") from None
        binder = self._bind_bound_actions
        push.apply(binder)

    def is_owned_by(self, authority: object) -> bool:
        return self._authority is authority

    def matches_authority_observation(self, observe: Callable[[], object]) -> bool:
        """Compare issuer identity before a caller's live observation, without admission."""
        return self._authority is observe()

    def _bind_bound_actions(
        self, hooks: _BoundHooks, receipt: bool, name: str,
    ) -> _BoundActions:
        binding_push = WorkspaceMemoryOperation._target_binding_push(
            self, hooks, receipt, name, _capability=_BOUND_TARGET_CAPABILITY,
        )
        return WorkspaceMemoryAuthority.bind_operation_target_actions(
            self._authority, binding_push, _capability=_BOUND_TARGET_CAPABILITY,
        )

    @property
    def workspace(self) -> Path:
        """Return the bound path without exposing the owning authority."""
        # The existing identity accessor owns the active, process/thread-bound
        # lease check under the lifecycle lock. A registered lease prevents the
        # authority from closing, and its informational path is immutable.
        _ = self.workspace_identity
        return self._authority.workspace

    @property
    def workspace_identity(self) -> tuple[int, int]:
        """Return the bound workspace inode identity for this active lease."""

        return self._actions.workspace_identity()

    def borrow_target_parent(self, snapshot: MemoryTargetSnapshot) -> BoundMemoryTarget:
        """Issue scoped parent access anchored to an issued target snapshot."""

        return self._actions.borrow_target(snapshot)

    def borrow_pending_receipt_parent(
        self, snapshot: MemoryTargetSnapshot
    ) -> BoundMemoryTarget:
        """Issue scoped parent access for one present, verified receipt snapshot."""

        return self._actions.borrow_receipt(snapshot)

    def __reduce__(self) -> NoReturn:
        raise TypeError("memory recovery operation cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("memory recovery operation cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("memory recovery operation cannot be copied")

    def _require_owner(self) -> None:
        if self._pid != os.getpid() or self._thread is not threading.current_thread():
            raise _error("closed") from None

    def _check_new(self) -> None:
        if self._state != self._NEW:
            raise _error("closed") from None

    def _check_issued(self, epoch: int) -> None:
        if self._issued_epoch != epoch:
            raise _error("closed") from None

    def _check_current(self, epoch: int) -> None:
        if self._state != self._ACTIVE or not self._registered or self._epoch != epoch:
            raise _error("closed") from None

    def _check_bound_current(self, epoch: int) -> None:
        self._check_current(epoch)
        if self._issued_epoch != epoch:
            raise _error("closed") from None

    def _activate(self, epoch: int, identity: tuple[int, int]) -> None:
        self._epoch = epoch
        self._workspace_identity = identity
        self._registered = True
        self._state = self._ACTIVE

    def _rollback(self) -> None:
        self._epoch = None
        self._workspace_identity = None
        self._registered = False
        self._state = self._NEW

    def _identity(self) -> tuple[int, int]:
        if self._state != self._ACTIVE or self._workspace_identity is None:
            raise _error("closed") from None
        return self._workspace_identity

    def _check_exit(self) -> bool:
        self._require_owner()
        if self._state == self._CLOSED:
            return False
        if self._state != self._ACTIVE or not self._registered:
            raise _error("closed") from None
        if self._target_parent_borrowers:
            raise _error("borrowed") from None
        return True

    def _finish(self) -> None:
        self._registered = False
        self._state = self._CLOSED

    def _add_parent(self, bound: BoundMemoryTarget) -> None:
        self._target_parent_borrowers.add(bound)

    def _remove_parent(self, bound: BoundMemoryTarget) -> None:
        self._target_parent_borrowers.discard(bound)

    def _check_parent(self, bound: BoundMemoryTarget) -> None:
        if bound not in self._target_parent_borrowers:
            raise _error("closed") from None

    def __enter__(self) -> Self:
        self._actions.enter()
        return self

    def require_current(self) -> None:
        """Require that this lease still belongs to its active authority epoch."""

        self._actions.require_current()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._actions.exit()


class BoundMemoryTarget(AbstractContextManager["BoundMemoryTarget"]):
    """A scoped parent-directory borrow anchored to one issued target snapshot.

    The scope retains a private duplicate of the authority's pinned ``memory``
    descriptor. Its snapshot operations do not mutate workspace state. The
    explicit caller-owned descriptor duplicate is a trusted in-process handoff:
    the caller may use descriptor-relative APIs, and the scope cannot revoke
    that duplicate after it is returned.
    """

    _NEW = "new"
    _ACTIVE = "active"
    _CLOSED = "closed"

    _authority: WorkspaceMemoryAuthority
    _validate_snapshot: Callable[[MemoryTargetSnapshot, int, bool], None]
    _thread: threading.Thread
    _epoch: int | None
    _actions: _BoundActions

    def __init__(
        self,
        operation: WorkspaceMemoryOperation,
        snapshot: MemoryTargetSnapshot,
        *,
        _capability: object | None = None,
    ) -> None:
        if _capability is _BOUND_TARGET_CAPABILITY:
            self._receipt_scope = False
        elif _capability is _BOUND_RECEIPT_TARGET_CAPABILITY:
            self._receipt_scope = True
        else:
            raise _error("invalid") from None
        self._operation = operation
        initialization = BoundMemoryTarget._initialization_hooks(self, _capability)
        WorkspaceMemoryOperation.initialize_bound_owner(
            operation, initialization, _capability=_capability,
        )
        self._snapshot = snapshot
        self._pid = os.getpid()
        WorkspaceMemoryOperation.initialize_bound_context(
            operation, initialization, _capability=_capability,
        )
        self._snapshot_name = snapshot.name
        self._snapshot_content = snapshot.content
        self._snapshot_stat_identity = snapshot.stat_identity
        self._state = self._NEW
        self._registered = False
        self._memory_fd: int | None = None
        self._memory_identity: tuple[int, int] | None = None
        hooks = _BoundHooks(
            self, self._require_owner, self._check_new, self._check_epoch,
            self._require_snapshot_unchanged,
            self._activate, self._rollback, self._check_registered,
            self._check_parent_present, self._check_local_current,
            self._duplicate_parent, self._check_exit, self._finish, self._close_parent,
        )
        binding_push = BoundMemoryTarget._action_binding_push(
            self, hooks, _capability=_capability,
        )
        WorkspaceMemoryOperation.initialize_bound_actions(
            operation, binding_push, _capability=_capability,
        )

    def _initialization_hooks(self, capability: object) -> _BoundInitialization:
        if (capability is not _BOUND_TARGET_CAPABILITY
                and capability is not _BOUND_RECEIPT_TARGET_CAPABILITY):
            raise _error("invalid") from None

        def retain_owner(owner: WorkspaceMemoryAuthority) -> None:
            self._authority = owner

        def retain_validator(
            validator: Callable[[MemoryTargetSnapshot, int, bool], None],
        ) -> None:
            self._validate_snapshot = validator

        def capture_validator(initialization: _BoundInitialization) -> None:
            WorkspaceMemoryAuthority.capture_bound_validator(
                self._authority, initialization, _capability=capability,
            )

        def retain_thread(thread: threading.Thread) -> None:
            self._thread = thread

        def retain_epoch(epoch: int | None) -> None:
            self._epoch = epoch

        hooks = object.__new__(_BoundInitialization)
        object.__setattr__(hooks, "retain_owner", retain_owner)
        object.__setattr__(hooks, "capture_validator", capture_validator)
        object.__setattr__(hooks, "retain_validator", retain_validator)
        object.__setattr__(hooks, "retain_thread", retain_thread)
        object.__setattr__(hooks, "retain_epoch", retain_epoch)
        return hooks

    def _action_binding_push(
        self, hooks: _BoundHooks, *, _capability: object | None = None,
    ) -> _BoundActionBindingPush:
        if (_capability is not _BOUND_TARGET_CAPABILITY
                and _capability is not _BOUND_RECEIPT_TARGET_CAPABILITY):
            raise _error("invalid") from None

        def apply(binder: _OperationBoundActionBinder) -> None:
            self._actions = binder(hooks, self._receipt_scope, self._snapshot_name)

        push = object.__new__(_BoundActionBindingPush)
        object.__setattr__(push, "apply", apply)
        return push

    @property
    def name(self) -> str:
        """The allowlisted target basename bound to this active scope."""

        self.require_current()
        return self._snapshot_name

    def duplicate_dir_fd(self) -> int:
        """Return a caller-owned duplicate; exiting this scope never revokes it."""

        return self._actions.duplicate_dir_fd()

    def __reduce__(self) -> NoReturn:
        raise TypeError("bound memory target cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("bound memory target cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("bound memory target cannot be copied")

    def __enter__(self) -> Self:
        self._actions.enter()
        return self

    def require_current(self) -> None:
        """Verify this parent borrow still belongs to its live operation epoch."""

        self._actions.require_current()

    def _require_owner(self) -> None:
        if self._pid != os.getpid() or threading.current_thread() is not self._thread:
            raise _error("closed") from None

    def _check_new(self) -> None:
        if self._state != self._NEW:
            raise _error("closed") from None

    def _check_epoch(self, epoch: int) -> None:
        if self._epoch != epoch:
            raise _error("closed") from None

    def _require_snapshot_unchanged(self, epoch: int) -> None:
        self._validate_snapshot(self._snapshot, epoch, self._receipt_scope)
        if (
            self._snapshot.name != self._snapshot_name
            or self._snapshot.content != self._snapshot_content
            or self._snapshot.stat_identity != self._snapshot_stat_identity
        ):
            raise _error("invalid") from None

    def _activate(self, memory_fd: int, identity: tuple[int, int]) -> None:
        self._memory_fd = memory_fd
        self._memory_identity = identity
        self._registered = True
        self._state = self._ACTIVE

    def _rollback(self) -> None:
        # The action still owns its local duplicate and closes it on failure.
        self._memory_fd = None
        self._memory_identity = None
        self._registered = False
        self._state = self._NEW

    def _check_registered(self) -> None:
        if self._state != self._ACTIVE or not self._registered:
            raise _error("closed") from None

    def _check_parent_present(self) -> None:
        if self._memory_fd is None or self._memory_identity is None:
            raise _error("closed") from None

    def _check_local_current(self) -> None:
        memory_fd = self._memory_fd
        memory_identity = self._memory_identity
        if memory_fd is None or memory_identity is None:
            raise _error("closed") from None
        info = _fstat(memory_fd, code="binding")
        _workspace_memory_directory(info)
        if _identity(info) != memory_identity:
            raise _error("binding") from None

    def _duplicate_parent(self) -> int:
        memory_fd = self._memory_fd
        if memory_fd is None:
            raise _error("closed") from None
        try:
            return os.dup(memory_fd)
        except (OSError, TypeError, ValueError):
            raise _error("io") from None

    def _check_exit(self) -> bool:
        self._require_owner()
        if self._state == self._CLOSED:
            return False
        self._check_registered()
        return True

    def _finish(self) -> None:
        self._registered = False
        self._state = self._CLOSED

    def _close_parent(self) -> None:
        memory_fd = self._memory_fd
        self._memory_fd = None
        self._memory_identity = None
        _close(memory_fd)

    def snapshot_target(self) -> MemoryTargetSnapshot:
        """Read the bound target through the authority's verified no-follow reader."""

        return self._actions.snapshot_target()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._actions.exit()


class ImmutableEvidenceNamespace(AbstractContextManager["ImmutableEvidenceNamespace"]):
    """A descriptor-bound immutable record namespace under one chain identity."""

    def __init__(
        self,
        authority: WorkspaceMemoryAuthority,
        kind: str,
        identity: str,
        *,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _NAMESPACE_CAPABILITY:
            raise _error("invalid") from None
        self._authority = authority
        self.kind = _validate_evidence_kind(kind)
        self.identity = _validate_digest(identity)
        self._actions = WorkspaceMemoryAuthority.bind_namespace_actions(
            authority, _capability=_NAMESPACE_CAPABILITY,
        )
        self._kind_fd: int | None = None
        self._record_fd: int | None = None
        self._kind_identity: tuple[int, int] | None = None
        self._record_identity: tuple[int, int] | None = None
        self._pid = os.getpid()
        self._active = False
        kind_fd: int | None = None
        record_fd: int | None = None
        try:
            self._actions.check_process()
            self._actions.check_binding()
            kind_fd, record_fd = self._actions.open_children(self.kind, self.identity)
            self._kind_fd = kind_fd
            self._record_fd = record_fd
            kind_fd = None
            record_fd = None
            self._kind_identity = _identity(_fstat(self._kind_fd, code="binding"))
            self._record_identity = _identity(_fstat(self._record_fd, code="binding"))
            self._record_store = _ImmutableSlotStore(
                self._record_fd_or_closed,
                self._assert_binding,
                self._sync_record_directory,
            )
            self._active = True
            self._assert_binding()
            self._actions.register(self)
        except BaseException:
            self._active = False
            _close(record_fd)
            _close(kind_fd)
            _close(self._record_fd)
            _close(self._kind_fd)
            self._record_fd = None
            self._kind_fd = None
            # Registration may append before failing. Remove only this exact
            # scope; previous namespaces and their owned descriptors survive.
            self._actions.unregister(self)
            raise

    @property
    def directory(self) -> Path:
        return (
            self._authority.workspace
            / _MEMORY_NAME
            / _RECOVERY_NAME
            / self.kind
            / self.identity
        )

    record_directory = directory

    def __reduce__(self) -> NoReturn:
        raise TypeError("memory recovery namespace cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("memory recovery namespace cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        raise TypeError("memory recovery namespace cannot be copied")

    def __enter__(self) -> Self:
        self._assert_process()
        self._assert_binding()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        self._cleanup()

    def _close_from_authority(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        was_active = self._active
        self._active = False
        record_fd = self._record_fd
        kind_fd = self._kind_fd
        self._record_fd = None
        self._kind_fd = None
        _close(record_fd)
        _close(kind_fd)
        if was_active:
            self._actions.unregister(self)

    def _assert_process(self) -> None:
        if not self._active or self._pid != os.getpid():
            raise _error("closed") from None
        self._actions.check_process()

    def _assert_binding(self) -> None:
        self._assert_process()
        self._actions.check_binding()
        kind_fd = self._kind_fd
        record_fd = self._record_fd
        kind_identity = self._kind_identity
        record_identity = self._record_identity
        if kind_fd is None or record_fd is None or kind_identity is None or record_identity is None:
            raise _error("closed") from None
        kind_info = _fstat(kind_fd, code="binding")
        _private_directory(kind_info)
        if _identity(kind_info) != kind_identity:
            raise _error("binding") from None
        self._actions.check_kind_binding(self.kind, kind_info)
        record_info = _fstat(record_fd, code="binding")
        _private_directory(record_info)
        if _identity(record_info) != record_identity:
            raise _error("binding") from None
        record_entry = _stat_entry(kind_fd, self.identity)
        if record_entry is None or not _same_binding(record_entry, record_info):
            raise _error("binding") from None

    def _record_fd_or_closed(self) -> int:
        if self._record_fd is None:
            raise _error("closed") from None
        return self._record_fd

    def _sync_record_directory(self) -> None:
        record_fd = self._record_fd_or_closed()
        self._assert_binding()
        _fsync(record_fd)
        self._assert_binding()

    def publish_record(self, slot: str, content: bytes) -> str:
        """Publish one immutable named record without replacing an existing entry."""

        slot = _validate_record_slot(slot)
        self._assert_process()
        if type(content) is not bytes:
            raise _error("invalid") from None
        self._record_store.publish(slot, content, content_digest=None)
        return slot

    publish = publish_record
    put = publish_record

    def read_record(self, slot: str) -> bytes | None:
        """Read and verify one named record, returning ``None`` only when absent."""

        slot = _validate_record_slot(slot)
        self._assert_process()
        return self._record_store.read(slot, content_digest=None, missing_is_none=True)

    read = read_record
    get = read_record


# A descriptive alias for callers that prefer a storage noun.
MemoryRecoveryStorage = WorkspaceMemoryAuthority


def open_workspace_memory(workspace: Path | str) -> WorkspaceMemoryAuthority:
    """Return the explicit context-managed workspace authority."""

    return WorkspaceMemoryAuthority(workspace)


def open_existing_workspace_memory(workspace: Path | str) -> WorkspaceMemoryAuthority:
    """Open a complete existing authority layout without creating or repairing it."""

    return WorkspaceMemoryAuthority(workspace, existing_only=True)


__all__ = [
    "MEMORY_TARGET_MAX_BYTES",
    "MemoryRecoveryStorage",
    "MemoryRecoveryStorageError",
    "MemoryStorageBindingError",
    "MemoryStorageConflictError",
    "MemoryStorageCorruptError",
    "MemoryStorageDurabilityError",
    "MemoryStorageError",
    "MemoryStorageMissingError",
    "MemoryStorageUnavailableError",
    "BoundMemoryTarget",
    "MemoryTargetSnapshot",
    "MemoryTargetStatIdentity",
    "ImmutableEvidenceNamespace",
    "WorkspaceMemoryAuthority",
    "WorkspaceMemoryAuthorityError",
    "WorkspaceMemoryLockGate",
    "WorkspaceMemoryOperation",
    "close_recovery_descriptor",
    "memory_target_stat_identity",
    "open_existing_workspace_memory",
    "open_recovery_private_directory",
    "open_workspace_memory",
    "require_memory_lock_gate_capabilities",
    "require_safe_memory_target",
    "stat_recovery_descriptor",
    "stat_recovery_entry",
]
