"""Offline bootstrap replacement for selected legacy memory targets.

This module deliberately provides a narrow, descriptor-relative mutation for
one-time workspace-memory bootstrap. It is not a general filesystem writer.
The authority lock coordinates participating nanobot processes; it cannot
prevent an uncooperative process with the same UID from racing these checks.
"""

from __future__ import annotations

import errno
import os
import secrets
import stat
from typing import NoReturn, cast

from nanobot.agent import memory_recovery_storage as storage

_MAX_TARGET_BYTES = 64 * 1024 * 1024
_TARGET_NAMES = frozenset(("MEMORY.md", "history.jsonl", "HISTORY.md", ".cursor", ".dream_cursor"))
_RECEIPT_NAME = ".consolidation.pending.json"
_TEMP_PREFIX = ".memory-recovery-replace-"
_TEMP_ATTEMPTS = 32
_READ_SIZE = 64 * 1024


class MemoryTargetReplacementUncertainError(storage.MemoryStorageError):
    """A rename was attempted, so the target's final state is uncertain."""

    def __init__(self) -> None:
        self.code = "uncertain"
        RuntimeError.__init__(self, "memory target replacement outcome is uncertain")


def _fail(code: str) -> NoReturn:
    raise storage.MemoryStorageError(code) from None


def _check_capabilities() -> None:
    flags = ("O_NOFOLLOW", "O_NONBLOCK")
    if os.name != "posix" or any(not isinstance(getattr(os, name, None), int) for name in flags):
        _fail("unavailable")
    dir_fd = getattr(os, "supports_dir_fd", set[object]())
    follow_symlinks = getattr(os, "supports_follow_symlinks", set[object]())
    if not {os.open, os.stat, os.rename} <= dir_fd or os.stat not in follow_symlinks:
        _fail("unavailable")
    if not callable(getattr(os, "fchmod", None)):
        _fail("unavailable")


def _stable_stat(info: os.stat_result) -> tuple[int | None, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        getattr(info, "st_mtime_ns", None),
        getattr(info, "st_ctime_ns", None),
    )


def _snapshot_matches(
    first: storage.MemoryTargetSnapshot, second: storage.MemoryTargetSnapshot
) -> bool:
    return (
        first.name == second.name
        and first.content == second.content
        and first.stat_identity == second.stat_identity
    )


def _same_snapshot_as_source(
    expected: storage.MemoryTargetSnapshot,
    current: storage.MemoryTargetSnapshot,
) -> None:
    if not _snapshot_matches(expected, current):
        _fail("binding")


def _named_stat(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except (OSError, TypeError, ValueError):
        _fail("io")


def _safe_stage(info: os.stat_result, content_size: int) -> None:
    get_uid = getattr(os, "geteuid", os.getuid)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != get_uid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_size != content_size
    ):
        _fail("corrupt")


def _same_stage_binding(first: os.stat_result, second: os.stat_result) -> bool:
    return (
        (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)
        and stat.S_ISREG(first.st_mode)
        and stat.S_ISREG(second.st_mode)
        and first.st_nlink == second.st_nlink == 1
    )


def _write_all(fd: int, content: bytes) -> None:
    view = memoryview(content)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(fd, view[offset:])
        except OSError as error:
            if error.errno == errno.EINTR:
                continue
            _fail("io")
        except (TypeError, ValueError):
            _fail("io")
        if type(written) is not int or written <= 0:
            _fail("io")
        offset += written


def _read_all(fd: int) -> bytes:
    content = bytearray()
    while True:
        try:
            chunk = os.read(fd, min(_READ_SIZE, _MAX_TARGET_BYTES + 1 - len(content)))
        except OSError as error:
            if error.errno == errno.EINTR:
                continue
            _fail("io")
        except (TypeError, ValueError):
            _fail("io")
        if not chunk:
            return bytes(content)
        content.extend(chunk)
        if len(content) > _MAX_TARGET_BYTES:
            _fail("corrupt")


def _verify_stage(
    parent_fd: int,
    stage_fd: int,
    stage_name: str,
    content: bytes,
    *,
    expected_stat: os.stat_result | None = None,
) -> os.stat_result:
    try:
        before = os.fstat(stage_fd)
    except (OSError, TypeError, ValueError):
        _fail("binding")
    _safe_stage(before, len(content))
    if expected_stat is not None and _stable_stat(before) != _stable_stat(expected_stat):
        _fail("corrupt")
    named = _named_stat(parent_fd, stage_name)
    if named is None or not _same_stage_binding(before, named):
        _fail("binding")
    if _stable_stat(before) != _stable_stat(named):
        _fail("binding")

    try:
        os.lseek(stage_fd, 0, os.SEEK_SET)
    except (OSError, TypeError, ValueError):
        _fail("io")
    observed = _read_all(stage_fd)
    try:
        after = os.fstat(stage_fd)
    except (OSError, TypeError, ValueError):
        _fail("binding")
    after_named = _named_stat(parent_fd, stage_name)
    if (
        after_named is None
        or not _same_stage_binding(after, after_named)
        or _stable_stat(before) != _stable_stat(after)
        or _stable_stat(after) != _stable_stat(after_named)
    ):
        _fail("binding")
    _safe_stage(after, len(content))
    if observed != content:
        _fail("corrupt")
    return after


def _fsync(fd: int) -> None:
    while True:
        try:
            os.fsync(fd)
            return
        except OSError as error:
            if error.errno == errno.EINTR:
                continue
            _fail("durability")
        except (TypeError, ValueError):
            _fail("durability")


def _create_stage(parent_fd: int) -> tuple[int, str]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    for _ in range(_TEMP_ATTEMPTS):
        name = _TEMP_PREFIX + secrets.token_hex(16)
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError:
            continue
        except (OSError, TypeError, ValueError):
            _fail("io")
        return descriptor, name
    _fail("io")


def _cleanup_stage_if_bound(
    scope: storage.BoundMemoryTarget,
    parent_fd: int,
    stage_fd: int | None,
    stage_name: str | None,
) -> None:
    if stage_fd is None or stage_name is None:
        return
    try:
        scope.require_current()
        descriptor_info = os.fstat(stage_fd)
        named_info = _named_stat(parent_fd, stage_name)
        if (
            named_info is None
            or not _same_stage_binding(descriptor_info, named_info)
            or _stable_stat(descriptor_info) != _stable_stat(named_info)
        ):
            return
        scope.require_current()
        descriptor_info = os.fstat(stage_fd)
        named_info = _named_stat(parent_fd, stage_name)
        if (
            named_info is not None
            and _same_stage_binding(descriptor_info, named_info)
            and _stable_stat(descriptor_info) == _stable_stat(named_info)
        ):
            os.unlink(stage_name, dir_fd=parent_fd)
    except BaseException:
        # Cleanup must not replace the content-free primary error. If the
        # scope is no longer bound, leave the private stage in place.
        return


def _verify_replaced_snapshot(
    snapshot: storage.MemoryTargetSnapshot,
    content: bytes,
    stage_info: os.stat_result,
) -> None:
    identity = snapshot.stat_identity
    if (
        snapshot.content != content
        or identity is None
        or (identity.device, identity.inode) != (stage_info.st_dev, stage_info.st_ino)
        or identity.uid != stage_info.st_uid
        or identity.gid != stage_info.st_gid
        or identity.links != 1
        or identity.size != len(content)
        or stat.S_IMODE(identity.mode) != 0o600
    ):
        _fail("binding")


def _close_all(descriptors: list[int | None]) -> None:
    first_error: OSError | None = None
    for descriptor in descriptors:
        if descriptor is None:
            continue
        try:
            os.close(descriptor)
        except OSError as error:
            if first_error is None:
                first_error = error
        except (TypeError, ValueError):
            if first_error is None:
                first_error = OSError(errno.EBADF, "descriptor close failed")
    if first_error is not None:
        raise first_error


def replace_target_if_unchanged(
    operation: storage.WorkspaceMemoryOperation,
    snapshot: storage.MemoryTargetSnapshot,
    new_bytes: bytes,
) -> storage.MemoryTargetSnapshot:
    """Atomically replace one unchanged allowlisted target with private bytes."""

    if (
        type(operation) is not storage.WorkspaceMemoryOperation
        or type(snapshot) is not storage.MemoryTargetSnapshot
        or type(new_bytes) is not bytes
        or len(new_bytes) > _MAX_TARGET_BYTES
        or snapshot.name not in _TARGET_NAMES
        or cast(object, snapshot.name) == _RECEIPT_NAME
    ):
        _fail("invalid")
    _check_capabilities()

    rename_attempted = False
    try:
        with operation.borrow_target_parent(snapshot) as scope:
            scope.require_current()
            current = scope.snapshot_target()
            _same_snapshot_as_source(snapshot, current)
            parent_fd = scope.duplicate_dir_fd()
            stage_fd: int | None = None
            stage_name: str | None = None
            try:
                scope.require_current()
                stage_fd, stage_name = _create_stage(parent_fd)
                try:
                    os.fchmod(stage_fd, 0o600)
                except (OSError, TypeError, ValueError):
                    _fail("io")
                _write_all(stage_fd, new_bytes)
                initial_stage_info = _verify_stage(parent_fd, stage_fd, stage_name, new_bytes)
                _fsync(stage_fd)
                stage_info = _verify_stage(
                    parent_fd,
                    stage_fd,
                    stage_name,
                    new_bytes,
                    expected_stat=initial_stage_info,
                )
                scope.require_current()
                current = scope.snapshot_target()
                _same_snapshot_as_source(snapshot, current)
                scope.require_current()
                stage_info = _verify_stage(
                    parent_fd,
                    stage_fd,
                    stage_name,
                    new_bytes,
                    expected_stat=stage_info,
                )
                rename_attempted = True
                os.rename(
                    stage_name,
                    snapshot.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                scope.require_current()
                replaced = scope.snapshot_target()
                _verify_replaced_snapshot(replaced, new_bytes, stage_info)
                _fsync(parent_fd)
                scope.require_current()
                final_snapshot = scope.snapshot_target()
                _verify_replaced_snapshot(final_snapshot, new_bytes, stage_info)
                return final_snapshot
            except BaseException:
                if not rename_attempted:
                    _cleanup_stage_if_bound(scope, parent_fd, stage_fd, stage_name)
                raise
            finally:
                _close_all([stage_fd, parent_fd])
    except BaseException as error:
        if rename_attempted:
            raise MemoryTargetReplacementUncertainError() from None
        if isinstance(error, storage.MemoryStorageError):
            raise
        if isinstance(error, Exception):
            raise storage.MemoryStorageError("io") from None
        raise


__all__ = ["MemoryTargetReplacementUncertainError", "replace_target_if_unchanged"]
