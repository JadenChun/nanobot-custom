"""Focused durability checks for descriptor-bound legacy memory targets."""

from __future__ import annotations

import errno
import os
import stat
import types
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_target_verify as verifier

_TARGET_NAMES = ("HISTORY.md", "history.jsonl", ".cursor", ".dream_cursor")


@pytest.mark.parametrize("expected", [None, bytearray(b"target"), "target", 1])
def test_nonbytes_refused_before_capabilities_or_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expected: object,
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory, "HISTORY.md", b"target")
    calls = []
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("HISTORY.md")
        with authority.operation() as operation:
            monkeypatch.setattr(verifier, "_check_capabilities", lambda: calls.append("capability"))
            monkeypatch.setattr(storage.WorkspaceMemoryOperation, "borrow_target_parent",
                                lambda *args: calls.append("filesystem"))
            _patch_verifier_os(monkeypatch, fsync=lambda fd: calls.append("fsync"))
            with pytest.raises(storage.MemoryStorageError):
                verifier.durably_verify_target(operation, snapshot, expected)
    assert calls == []


def _verify_capable() -> bool:
    if os.name != "posix":
        return False
    if any(
        not isinstance(getattr(os, name, None), int) or getattr(os, name) <= 0
        for name in ("O_NOFOLLOW", "O_NONBLOCK")
    ):
        return False
    return (
        os.open in getattr(os, "supports_dir_fd", set())
        and os.stat in getattr(os, "supports_dir_fd", set())
        and os.stat in getattr(os, "supports_follow_symlinks", set())
        and callable(getattr(os, "fsync", None))
    )


pytestmark = pytest.mark.skipif(
    not _verify_capable(), reason="durable target verification requires POSIX descriptor support"
)


def _target(memory: Path, name: str, content: bytes = b"legacy bytes") -> Path:
    target = memory / name
    target.write_bytes(content)
    target.chmod(0o644)
    return target


def _stat_signature(info: os.stat_result) -> tuple[int | None, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _patch_verifier_os(
    monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> types.ModuleType:
    proxy = types.ModuleType("memory_recovery_target_verify_os")
    proxy.__dict__.update(vars(os))
    proxy.__dict__.update(overrides)
    if "open" in overrides:
        proxy.supports_dir_fd = set(proxy.supports_dir_fd) | {proxy.open}
    monkeypatch.setattr(verifier, "os", proxy)
    return proxy


def _verify(
    authority: storage.WorkspaceMemoryAuthority,
    snapshot: storage.MemoryTargetSnapshot,
    expected_bytes: bytes,
) -> storage.MemoryTargetSnapshot:
    with authority.operation() as operation:
        return verifier.durably_verify_target(operation, snapshot, expected_bytes)


def _assert_fixed_storage_error(action: Any, sensitive: bytes = b"secret target bytes") -> None:
    with pytest.raises(storage.MemoryStorageError) as failure:
        action()
    assert str(failure.value) in storage.MemoryStorageError._MESSAGES.values()
    assert sensitive.decode("ascii", errors="ignore") not in str(failure.value)


@pytest.mark.parametrize("name", _TARGET_NAMES)
def test_verify_fsyncs_target_then_bound_parent_and_returns_fresh_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    content = b"legacy target\x00bytes"
    target = _target(memory, name, content)
    before = target.stat()
    original_fsync = os.fsync
    synced: list[os.stat_result] = []

    def record_fsync(fd: int) -> None:
        synced.append(os.fstat(fd))
        original_fsync(fd)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target(name)
        _patch_verifier_os(monkeypatch, fsync=record_fsync)
        verified = _verify(authority, snapshot, content)

    assert type(verified) is storage.MemoryTargetSnapshot
    assert verified is not snapshot
    assert verified.name == snapshot.name == name
    assert verified.content == snapshot.content == content
    assert verified.stat_identity == snapshot.stat_identity
    assert len(synced) == 2
    assert stat.S_ISREG(synced[0].st_mode)
    assert stat.S_ISDIR(synced[1].st_mode)
    assert synced[0].st_ino == before.st_ino
    assert synced[1].st_ino == memory.stat().st_ino
    assert target.read_bytes() == content
    assert _stat_signature(target.stat()) == _stat_signature(before)


def test_verify_rejects_changed_bytes_and_copied_inode_without_syncing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _target(memory, "HISTORY.md", b"original")
    before = target.stat()
    synced: list[int] = []
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        changed = authority.snapshot_target("HISTORY.md")
        _patch_verifier_os(monkeypatch, fsync=synced.append)
        target.write_bytes(b"mutated!")
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        _assert_fixed_storage_error(lambda: _verify(authority, changed, b"original"))
        assert synced == []
        assert target.read_bytes() == b"mutated!"

        replacement = authority.snapshot_target("HISTORY.md")
        target.unlink()
        _target(memory, "HISTORY.md", b"mutated!")
        assert target.stat().st_ino != replacement.stat_identity.inode  # type: ignore[union-attr]
        _assert_fixed_storage_error(lambda: _verify(authority, replacement, b"mutated!"))
        assert synced == []
        assert target.read_bytes() == b"mutated!"


def test_verify_rejects_wrong_authority_and_symlink_without_following(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_memory = tmp_path / "source" / "memory"
    other_memory = tmp_path / "other" / "memory"
    source_memory.mkdir(parents=True)
    other_memory.mkdir(parents=True)
    source = _target(source_memory, ".cursor", b"secret target bytes")
    other = _target(other_memory, ".cursor", b"secret target bytes")
    followed: list[int] = []
    with storage.WorkspaceMemoryAuthority(tmp_path / "source") as source_authority:
        snapshot = source_authority.snapshot_target(".cursor")
        _patch_verifier_os(monkeypatch, fsync=followed.append)
        with storage.WorkspaceMemoryAuthority(tmp_path / "other") as other_authority:
            _assert_fixed_storage_error(lambda: _verify(other_authority, snapshot, b"secret target bytes"))
        assert followed == []

        source.unlink()
        source.symlink_to(other)
        _assert_fixed_storage_error(lambda: _verify(source_authority, snapshot, b"secret target bytes"))

    assert source.is_symlink()
    assert other.read_bytes() == b"secret target bytes"
    assert followed == []


@pytest.mark.parametrize("fail_fsync", [False, True])
def test_verify_closes_descriptors_on_success_and_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_fsync: bool
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _target(memory, "history.jsonl", b"unchanged")
    before = _stat_signature(target.stat())
    original_open = os.open
    original_duplicate_parent = storage.BoundMemoryTarget.duplicate_dir_fd
    original_close = storage._close
    allocated: list[int] = []
    closed: list[int] = []

    def track_open(*args: Any, **kwargs: Any) -> int:
        fd = original_open(*args, **kwargs)
        allocated.append(fd)
        return fd

    def track_duplicate_parent(scope: storage.BoundMemoryTarget) -> int:
        fd = original_duplicate_parent(scope)
        allocated.append(fd)
        return fd

    def track_close(fd: int | None) -> None:
        if fd in allocated:
            closed.append(fd)
        original_close(fd)

    def run_fsync(fd: int) -> None:
        if fail_fsync:
            raise OSError(errno.EIO, "private failure detail")
        os.fsync(fd)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("history.jsonl")
        _patch_verifier_os(monkeypatch, fsync=run_fsync, open=track_open)
        monkeypatch.setattr(storage.BoundMemoryTarget, "duplicate_dir_fd", track_duplicate_parent)
        monkeypatch.setattr(storage, "_close", track_close)
        if fail_fsync:
            _assert_fixed_storage_error(lambda: _verify(authority, snapshot, b"unchanged"))
        else:
            verified = _verify(authority, snapshot, b"unchanged")
            assert verified.content == b"unchanged"

    assert Counter(allocated) == Counter(closed)
    assert target.read_bytes() == b"unchanged"
    assert _stat_signature(target.stat()) == before


def test_verify_rechecks_after_parent_fsync_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _target(memory, ".dream_cursor", b"before fsync")
    original_fsync = os.fsync
    synced: list[os.stat_result] = []

    def mutate_after_parent_sync(fd: int) -> None:
        info = os.fstat(fd)
        synced.append(info)
        original_fsync(fd)
        if stat.S_ISDIR(info.st_mode):
            target.write_bytes(b"after fsync")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target(".dream_cursor")
        _patch_verifier_os(monkeypatch, fsync=mutate_after_parent_sync)
        _assert_fixed_storage_error(
            lambda: _verify(authority, snapshot, b"before fsync"), b"before fsync"
        )

    assert len(synced) == 2
    assert target.read_bytes() == b"after fsync"


def test_verify_rejects_missing_target_and_mismatched_expected_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _target(memory, "HISTORY.md", b"issued bytes")
    synced: list[int] = []

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("HISTORY.md")
        _patch_verifier_os(monkeypatch, fsync=synced.append)
        _assert_fixed_storage_error(lambda: _verify(authority, snapshot, b"other bytes"))
        assert target.read_bytes() == b"issued bytes"
        assert synced == []

        target.unlink()
        _assert_fixed_storage_error(lambda: _verify(authority, snapshot, b"issued bytes"))

    assert not target.exists()
    assert synced == []


def test_verify_fails_closed_when_descriptor_capability_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _target(memory, ".cursor", b"unchanged")
    before = _stat_signature(target.stat())
    synced: list[int] = []

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target(".cursor")
        _patch_verifier_os(monkeypatch, O_NOFOLLOW=None, fsync=synced.append)
        with authority.operation() as operation:
            with pytest.raises(storage.MemoryStorageError) as failure:
                verifier.durably_verify_target(operation, snapshot, b"unchanged")
        assert failure.value.code == "unavailable"
        assert str(failure.value) in storage.MemoryStorageError._MESSAGES.values()

    assert target.read_bytes() == b"unchanged"
    assert _stat_signature(target.stat()) == before
    assert synced == []
