"""Descriptor-relative archive of legacy HISTORY.md with a frozen backup name.

A present issued history.jsonl snapshot, or a privately issued pending receipt
with explicit validated digest and size, anchors the pinned memory directory during
archive and replay. The legacy source identity is frozen separately in PREPARED
evidence. Participating writers hold the workspace lock; an uncooperative same-UID
pathname race around conditional unlink cannot be eliminated with portable POSIX APIs.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from typing import NoReturn

from nanobot.agent import memory_recovery_storage as storage

_SOURCE = "HISTORY.md"
_ANCHOR = "history.jsonl"
_RECEIPT = ".consolidation.pending.json"
_BACKUP = re.compile(r"HISTORY\.md\.bak(?:\.([2-9][0-9]*))?\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CHUNK = 64 * 1024
_MAX_BYTES = 64 * 1024 * 1024


class LegacyHistoryArchiveUncertainError(storage.MemoryStorageError):
    """A link or unlink was attempted; inspect the frozen backup before replay."""

    def __init__(self) -> None:
        self.code = "uncertain"
        RuntimeError.__init__(self, "legacy history archive outcome is uncertain")


def _fail(code: str) -> NoReturn:
    raise storage.MemoryStorageError(code) from None


def _capabilities() -> None:
    required = {os.open, os.stat, os.link, os.unlink}
    if (
        os.name != "posix"
        or any(not isinstance(getattr(os, flag, None), int) for flag in ("O_NOFOLLOW", "O_NONBLOCK"))
        or not required <= getattr(os, "supports_dir_fd", set[object]())
        or os.link not in getattr(os, "supports_follow_symlinks", set[object]())
        or os.stat not in getattr(os, "supports_follow_symlinks", set[object]())
        or os.listdir not in getattr(os, "supports_fd", set[object]())
    ):
        _fail("unavailable")


def _read_capabilities() -> None:
    if (
        os.name != "posix"
        or any(not isinstance(getattr(os, flag, None), int) for flag in ("O_NOFOLLOW", "O_NONBLOCK"))
        or not {os.open, os.stat} <= getattr(os, "supports_dir_fd", set[object]())
        or os.stat not in getattr(os, "supports_follow_symlinks", set[object]())
    ):
        _fail("unavailable")


def _scope(
    operation: storage.WorkspaceMemoryOperation,
    snapshot: storage.MemoryTargetSnapshot,
    *,
    for_archive: bool,
) -> storage.BoundMemoryTarget:
    if (
        type(operation) is not storage.WorkspaceMemoryOperation
        or type(snapshot) is not storage.MemoryTargetSnapshot
        or snapshot.name not in ((_ANCHOR,) if for_archive else (_SOURCE, _ANCHOR))
        or snapshot.content is None
        or snapshot.stat_identity is None
    ):
        _fail("invalid")
    return operation.borrow_target_parent(snapshot)


def _stat(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
        info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )


def _expected(identity: storage.MemoryTargetStatIdentity) -> tuple[int | None, ...]:
    return (
        identity.device, identity.inode, identity.mode, identity.uid, identity.gid,
        identity.links, identity.size, identity.mtime_ns, identity.ctime_ns,
    )


def _validate_evidence(
    expected_source_stat: storage.MemoryTargetStatIdentity,
    expected_digest: str,
    expected_size: int,
) -> None:
    if (
        type(expected_source_stat) is not storage.MemoryTargetStatIdentity
        or type(expected_digest) is not str
        or _DIGEST.fullmatch(expected_digest) is None
        or type(expected_size) is not int
        or not 0 <= expected_size <= _MAX_BYTES
    ):
        _fail("invalid")
    identity = expected_source_stat
    mode = identity.mode
    if (
        type(identity.device) is not int or type(identity.inode) is not int
        or type(mode) is not int or type(identity.uid) is not int
        or type(identity.gid) is not int or type(identity.links) is not int
        or type(identity.size) is not int
        or type(identity.mtime_ns) is not int
        or type(identity.ctime_ns) is not int
        or identity.inode <= 0
        or not stat.S_ISREG(mode) or identity.uid != os.geteuid()
        or identity.links != 1 or identity.size != expected_size
        or not stat.S_IMODE(mode) & 0o400
        or stat.S_IMODE(mode) & (0o022 | 0o7000)
    ):
        _fail("invalid")


def _check_entry(
    info: os.stat_result,
    identity: storage.MemoryTargetStatIdentity,
    links: int,
    *,
    initial: bool,
) -> None:
    expected = _expected(identity)
    observed = _signature(info)
    if (
        not stat.S_ISREG(info.st_mode)
        or observed[:5] != expected[:5]
        or observed[5] != links
        or observed[6:8] != expected[6:8]
        or (initial and observed[8] != expected[8])
    ):
        _fail("conflict")


def _read_entry(
    parent_fd: int,
    name: str,
    identity: storage.MemoryTargetStatIdentity,
    digest: str,
    size: int,
    links: int,
    *,
    initial: bool,
    sync_source: bool = False,
) -> os.stat_result:
    before = _stat(parent_fd, name)
    if before is None:
        _fail("conflict")
    _check_entry(before, identity, links, initial=initial)
    descriptor = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    try:
        if _signature(os.fstat(descriptor)) != _signature(before):
            _fail("conflict")
        if sync_source:
            try:
                os.fsync(descriptor)
            except OSError:
                _fail("durability")
        observed_digest = hashlib.sha256()
        length = 0
        while True:
            chunk = os.read(descriptor, min(_CHUNK, size + 1 - length))
            if not chunk:
                break
            observed_digest.update(chunk)
            length += len(chunk)
            if length > size:
                _fail("conflict")
        if (
            length != size
            or observed_digest.hexdigest() != digest
            or _signature(os.fstat(descriptor)) != _signature(before)
            or (after := _stat(parent_fd, name)) is None
            or _signature(after) != _signature(before)
        ):
            _fail("conflict")
        return after
    finally:
        os.close(descriptor)


def choose_legacy_backup_name(
    operation: storage.WorkspaceMemoryOperation,
    anchor_snapshot: storage.MemoryTargetSnapshot,
) -> str:
    """Choose the first absent conventional slot, counting dangling symlinks."""
    _capabilities()
    with _scope(operation, anchor_snapshot, for_archive=False) as scope:
        scope.require_current()
        parent_fd = scope.duplicate_dir_fd()
        try:
            current = scope.snapshot_target()
            if (
                current.content != anchor_snapshot.content
                or current.stat_identity != anchor_snapshot.stat_identity
            ):
                _fail("conflict")
            names = set(os.listdir(parent_fd))
            index = 1
            while True:
                name = f"{_SOURCE}.bak" if index == 1 else f"{_SOURCE}.bak.{index}"
                if name not in names:
                    scope.require_current()
                    if _stat(parent_fd, name) is None:
                        return name
                index += 1
        except storage.MemoryStorageError:
            raise
        except (OSError, TypeError, ValueError):
            _fail("io")
        finally:
            os.close(parent_fd)


def verify_legacy_backup_only(
    operation: storage.WorkspaceMemoryOperation,
    history_anchor_snapshot: storage.MemoryTargetSnapshot,
    *,
    expected_source_stat: storage.MemoryTargetStatIdentity,
    expected_digest: str,
    expected_size: int,
    backup_name: str,
) -> str:
    """Prove the archived one-link backup without archive application or replay."""
    if type(backup_name) is not str or _BACKUP.fullmatch(backup_name) is None:
        _fail("invalid")
    _validate_evidence(expected_source_stat, expected_digest, expected_size)
    _read_capabilities()
    try:
        with _scope(operation, history_anchor_snapshot, for_archive=True) as scope:
            scope.require_current()
            parent_fd = scope.duplicate_dir_fd()
            try:
                def check_anchor() -> None:
                    scope.require_current()
                    current = scope.snapshot_target()
                    if (current.content != history_anchor_snapshot.content
                            or current.stat_identity != history_anchor_snapshot.stat_identity):
                        _fail("conflict")

                check_anchor()
                if _stat(parent_fd, _SOURCE) is not None:
                    _fail("conflict")
                _read_entry(parent_fd, backup_name, expected_source_stat, expected_digest,
                            expected_size, 1, initial=False)
                check_anchor()
                try:
                    os.fsync(parent_fd)
                except OSError:
                    _fail("durability")
                check_anchor()
                if _stat(parent_fd, _SOURCE) is not None:
                    _fail("conflict")
                _read_entry(parent_fd, backup_name, expected_source_stat, expected_digest,
                            expected_size, 1, initial=False)
                check_anchor()
                return backup_name
            finally:
                os.close(parent_fd)
    except storage.MemoryStorageError:
        raise
    except (OSError, TypeError, ValueError):
        _fail("io")


def archive_legacy_history_if_unchanged(
    operation: storage.WorkspaceMemoryOperation,
    history_anchor_snapshot: storage.MemoryTargetSnapshot,
    *,
    expected_source_stat: storage.MemoryTargetStatIdentity,
    expected_digest: str,
    expected_size: int,
    backup_name: str,
    expected_receipt_digest: str | None = None,
    expected_receipt_size: int | None = None,
) -> str:
    """Archive verified HISTORY.md; replay only matching two- or one-link states."""
    if type(backup_name) is not str or _BACKUP.fullmatch(backup_name) is None:
        _fail("invalid")
    _validate_evidence(expected_source_stat, expected_digest, expected_size)
    receipt_anchor = (
        type(history_anchor_snapshot) is storage.MemoryTargetSnapshot
        and history_anchor_snapshot.name == _RECEIPT
    )
    if receipt_anchor:
        if (
            type(expected_receipt_digest) is not str
            or _DIGEST.fullmatch(expected_receipt_digest) is None
            or type(expected_receipt_size) is not int
            or not 0 <= expected_receipt_size <= _MAX_BYTES
        ):
            _fail("invalid")
        if (
            history_anchor_snapshot.content is None
            or history_anchor_snapshot.stat_identity is None
            or len(history_anchor_snapshot.content) != expected_receipt_size
            or hashlib.sha256(history_anchor_snapshot.content).hexdigest()
            != expected_receipt_digest
        ):
            _fail("conflict")
    elif expected_receipt_digest is not None or expected_receipt_size is not None:
        _fail("invalid")
    _capabilities()
    mutation_attempted = False
    try:
        if receipt_anchor:
            scope_context = operation.borrow_pending_receipt_parent(history_anchor_snapshot)
        else:
            scope_context = _scope(operation, history_anchor_snapshot, for_archive=True)
        with scope_context as scope:
            scope.require_current()
            parent_fd = scope.duplicate_dir_fd()
            backup_fd: int | None = None
            try:
                def require_anchor_current() -> None:
                    scope.require_current()
                    if receipt_anchor:
                        anchor = scope.snapshot_target()
                        if (
                            anchor.content != history_anchor_snapshot.content
                            or anchor.stat_identity != history_anchor_snapshot.stat_identity
                        ):
                            _fail("conflict")
                        assert expected_receipt_digest is not None
                        assert expected_receipt_size is not None
                        assert history_anchor_snapshot.stat_identity is not None
                        _read_entry(
                            parent_fd, _RECEIPT, history_anchor_snapshot.stat_identity,
                            expected_receipt_digest, expected_receipt_size, 1, initial=True,
                        )

                anchor = scope.snapshot_target()
                if (
                    anchor.content != history_anchor_snapshot.content
                    or anchor.stat_identity != history_anchor_snapshot.stat_identity
                ):
                    _fail("conflict")
                require_anchor_current()
                source = _stat(parent_fd, _SOURCE)
                backup = _stat(parent_fd, backup_name)
                if backup is None:
                    if source is None:
                        _fail("conflict")
                    _read_entry(
                        parent_fd, _SOURCE, expected_source_stat, expected_digest,
                        expected_size, 1, initial=True, sync_source=True,
                    )
                    require_anchor_current()
                    _read_entry(
                        parent_fd, _SOURCE, expected_source_stat, expected_digest,
                        expected_size, 1, initial=True,
                    )
                    if _stat(parent_fd, backup_name) is not None:
                        _fail("conflict")
                    mutation_attempted = True
                    os.link(
                        _SOURCE, backup_name, src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd, follow_symlinks=False,
                    )
                    backup = _stat(parent_fd, backup_name)
                if backup is None:
                    _fail("conflict")
                if source is None:
                    _read_entry(
                        parent_fd, backup_name, expected_source_stat, expected_digest,
                        expected_size, 1, initial=False,
                    )
                    require_anchor_current()
                    os.fsync(parent_fd)
                    require_anchor_current()
                    _read_entry(
                        parent_fd, backup_name, expected_source_stat, expected_digest,
                        expected_size, 1, initial=False,
                    )
                    return backup_name
                backup_info: os.stat_result | None = None
                for _ in range(2):
                    source_info = _read_entry(
                        parent_fd, _SOURCE, expected_source_stat, expected_digest,
                        expected_size, 2, initial=False,
                    )
                    backup_info = _read_entry(
                        parent_fd, backup_name, expected_source_stat, expected_digest,
                        expected_size, 2, initial=False,
                    )
                    if _signature(source_info) != _signature(backup_info):
                        _fail("conflict")
                    if _ == 0:
                        require_anchor_current()
                        os.fsync(parent_fd)
                if backup_info is None:
                    _fail("conflict")
                backup_fd = os.open(
                    backup_name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_fd,
                )
                if _signature(os.fstat(backup_fd)) != _signature(backup_info):
                    _fail("conflict")
                require_anchor_current()
                # Recheck the named source immediately before unlink. This is
                # conditional only with respect to cooperating locked writers.
                _read_entry(
                    parent_fd, _SOURCE, expected_source_stat, expected_digest,
                    expected_size, 2, initial=False,
                )
                if (
                    _signature(os.fstat(backup_fd)) != _signature(backup_info)
                    or (named_backup := _stat(parent_fd, backup_name)) is None
                    or _signature(named_backup) != _signature(backup_info)
                ):
                    _fail("conflict")
                mutation_attempted = True
                os.unlink(_SOURCE, dir_fd=parent_fd)
                if _stat(parent_fd, _SOURCE) is not None:
                    _fail("conflict")
                held_backup = os.fstat(backup_fd)
                named_backup = _stat(parent_fd, backup_name)
                if (
                    named_backup is None
                    or _signature(held_backup) != _signature(named_backup)
                    or held_backup.st_nlink != 1
                    or _signature(held_backup)[:5] != _signature(backup_info)[:5]
                    or _signature(held_backup)[6:8] != _signature(backup_info)[6:8]
                ):
                    _fail("conflict")
                _read_entry(
                    parent_fd, backup_name, expected_source_stat, expected_digest,
                    expected_size, 1, initial=False,
                )
                os.fsync(parent_fd)
                require_anchor_current()
                _read_entry(
                    parent_fd, backup_name, expected_source_stat, expected_digest,
                    expected_size, 1, initial=False,
                )
                return backup_name
            finally:
                if backup_fd is not None:
                    os.close(backup_fd)
                os.close(parent_fd)
    except BaseException as error:
        if mutation_attempted:
            raise LegacyHistoryArchiveUncertainError() from None
        if isinstance(error, storage.MemoryStorageError):
            raise
        if isinstance(error, Exception):
            raise storage.MemoryStorageError("io") from None
        raise


__all__ = [
    "LegacyHistoryArchiveUncertainError",
    "choose_legacy_backup_name",
    "archive_legacy_history_if_unchanged",
    "verify_legacy_backup_only",
]
