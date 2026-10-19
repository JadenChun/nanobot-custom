"""Descriptor-bound durability verification for selected legacy memory targets."""

from __future__ import annotations

import errno
import os
from typing import NoReturn, cast

from nanobot.agent import memory_recovery_storage as storage

_TARGET_NAMES = frozenset(("HISTORY.md", "history.jsonl", ".cursor", ".dream_cursor"))
_MAX_TARGET_BYTES = storage.MEMORY_TARGET_MAX_BYTES
_READ_SIZE = 64 * 1024


def _fail(code: str) -> NoReturn:
    raise storage.MemoryStorageError(code) from None


def _check_capabilities() -> None:
    if os.name != "posix":
        _fail("unavailable")
    for name in ("O_NOFOLLOW", "O_NONBLOCK"):
        value = getattr(os, name, None)
        if not isinstance(value, int) or value <= 0:
            _fail("unavailable")
    if not all(
        callable(getattr(os, name, None))
        for name in ("open", "read", "fstat", "close", "fsync")
    ):
        _fail("unavailable")
    if os.open not in getattr(os, "supports_dir_fd", set[object]()):
        _fail("unavailable")
    if os.stat not in getattr(os, "supports_dir_fd", set[object]()):
        _fail("unavailable")
    if os.stat not in getattr(os, "supports_follow_symlinks", set[object]()):
        _fail("unavailable")


def _same_snapshot(
    expected: storage.MemoryTargetSnapshot, current: storage.MemoryTargetSnapshot
) -> bool:
    return (
        expected.name == current.name
        and expected.content == current.content
        and expected.stat_identity == current.stat_identity
    )


def _require_snapshot_match(
    expected: storage.MemoryTargetSnapshot, current: storage.MemoryTargetSnapshot
) -> None:
    if not _same_snapshot(expected, current):
        _fail("binding")


def _fsync(fd: int) -> None:
    try:
        os.fsync(fd)
    except (OSError, TypeError, ValueError):
        _fail("durability")


def _read_target(fd: int) -> bytes:
    content = bytearray()
    while True:
        remaining = _MAX_TARGET_BYTES - len(content)
        try:
            chunk = os.read(fd, min(_READ_SIZE, remaining + 1))
        except OSError as error:
            if error.errno == errno.EINTR:
                continue
            _fail("io")
        except (TypeError, ValueError):
            _fail("io")
        if not chunk:
            return bytes(content)
        if len(chunk) > remaining:
            _fail("corrupt")
        content.extend(chunk)


def _open_target(parent_fd: int, name: str) -> int:
    try:
        return storage.open_recovery_read_leaf(os.open, name, parent_fd)
    except OSError as error:
        if error.errno == errno.ENOENT:
            _fail("binding")
        if error.errno in (
            errno.EACCES,
            errno.EISDIR,
            errno.ELOOP,
            errno.ENOTDIR,
            errno.EPERM,
        ):
            _fail("corrupt")
        _fail("io")
    except (TypeError, ValueError):
        _fail("io")


def _require_descriptor_binding(
    parent_fd: int,
    target_fd: int,
    name: str,
    expected: storage.MemoryTargetStatIdentity,
) -> None:
    descriptor_info = storage.stat_recovery_descriptor(target_fd, code="binding")
    storage.require_safe_memory_target(name, descriptor_info)
    if storage.memory_target_stat_identity(descriptor_info) != expected:
        _fail("binding")

    named_info = storage.stat_recovery_entry(parent_fd, name)
    if named_info is None:
        _fail("binding")
    storage.require_safe_memory_target(name, named_info)
    if storage.memory_target_stat_identity(named_info) != expected:
        _fail("binding")


def durably_verify_target(
    operation: storage.WorkspaceMemoryOperation,
    snapshot: storage.MemoryTargetSnapshot,
    expected_bytes: object,
) -> storage.MemoryTargetSnapshot:
    """Fsync an unchanged target and its bound parent, then return a fresh snapshot."""

    if (
        type(operation) is not storage.WorkspaceMemoryOperation
        or type(snapshot) is not storage.MemoryTargetSnapshot
        or type(expected_bytes) is not bytes
        or snapshot.name not in _TARGET_NAMES
        or snapshot.content is None
        or snapshot.stat_identity is None
        or len(expected_bytes) > _MAX_TARGET_BYTES
        or snapshot.content != expected_bytes
        or snapshot.stat_identity.size != len(expected_bytes)
    ):
        _fail("invalid")
    _check_capabilities()

    parent_fd: int | None = None
    target_fd: int | None = None
    try:
        with operation.borrow_target_parent(snapshot) as scope:
            scope.require_current()
            current = scope.snapshot_target()
            _require_snapshot_match(snapshot, current)
            parent_fd = scope.duplicate_dir_fd()
            scope.require_current()
            target_fd = _open_target(parent_fd, snapshot.name)
            expected_identity = snapshot.stat_identity
            if cast(object, expected_identity) is None:  # Narrow the issued-snapshot invariant for type checkers.
                _fail("invalid")
            _require_descriptor_binding(parent_fd, target_fd, snapshot.name, expected_identity)
            content = _read_target(target_fd)
            if content != expected_bytes:
                _fail("binding")
            _require_descriptor_binding(parent_fd, target_fd, snapshot.name, expected_identity)

            scope.require_current()
            _fsync(target_fd)
            _require_descriptor_binding(parent_fd, target_fd, snapshot.name, expected_identity)
            scope.require_current()
            _fsync(parent_fd)
            scope.require_current()

            verified = scope.snapshot_target()
            _require_snapshot_match(snapshot, verified)
            return verified
    except storage.MemoryStorageError:
        raise
    except (OSError, TypeError, ValueError):
        _fail("io")
    finally:
        storage.close_recovery_descriptor(target_fd)
        storage.close_recovery_descriptor(parent_fd)


__all__ = ["durably_verify_target"]
