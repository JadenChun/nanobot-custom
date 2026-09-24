"""Conditional replacement of selected legacy memory targets."""

from __future__ import annotations

import errno
import os
import shutil
import stat
import threading
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_target_mutation as mutation


def _replace_capable() -> bool:
    if os.name != "posix":
        return False
    required_flags = ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")
    if any(not isinstance(getattr(os, name, None), int) for name in required_flags):
        return False
    required = {os.open, os.stat, os.mkdir, os.link, os.rename}
    return required <= getattr(os, "supports_dir_fd", set()) and os.link in getattr(
        os, "supports_follow_symlinks", set()
    )


pytestmark = pytest.mark.skipif(
    not _replace_capable(), reason="target replacement requires POSIX no-follow dir-fd support"
)

_TEMP_PREFIX = ".memory-recovery-replace-"


def _temps(memory: Path) -> list[Path]:
    return list(memory.glob(f"{_TEMP_PREFIX}*"))


def _write_target(memory: Path, name: str, content: bytes, mode: int = 0o644) -> Path:
    target = memory / name
    target.write_bytes(content)
    target.chmod(mode)
    return target


def _assert_no_temporary_files(memory: Path) -> None:
    assert _temps(memory) == []


def _assert_private_temporary_files(memory: Path) -> None:
    for stage in _temps(memory):
        info = stage.stat()
        assert stat.S_IMODE(info.st_mode) == 0o600
        assert stat.S_ISREG(info.st_mode)
        assert info.st_nlink == 1


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


def _replace(
    authority: storage.WorkspaceMemoryAuthority,
    snapshot: storage.MemoryTargetSnapshot,
    content: bytes,
) -> storage.MemoryTargetSnapshot:
    with authority.operation() as operation:
        return mutation.replace_target_if_unchanged(operation, snapshot, content)


def test_replace_absent_and_empty_targets_and_consumes_prior_snapshot(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir(mode=0o755)
    absent_path = memory / "history.jsonl"
    empty_path = _write_target(memory, "MEMORY.md", b"")
    empty_before = empty_path.stat()

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        absent = authority.snapshot_target("history.jsonl")
        absent_after = _replace(authority, absent, b"new history")
        empty = authority.snapshot_target("MEMORY.md")
        empty_after = _replace(authority, empty, b"new memory")

        assert absent_after.name == "history.jsonl"
        assert absent_after.content == b"new history"
        assert absent_after.stat_identity is not None
        assert empty_after.name == "MEMORY.md"
        assert empty_after.content == b"new memory"
        assert empty_after.stat_identity is not None
        with pytest.raises(storage.MemoryStorageError) as stale:
            _replace(authority, empty, b"must not replace")
        assert stale.value.code == "binding"

    assert absent_path.read_bytes() == b"new history"
    assert stat.S_IMODE(absent_path.stat().st_mode) == 0o600
    assert empty_path.read_bytes() == b"new memory"
    assert stat.S_IMODE(empty_path.stat().st_mode) == 0o600
    assert empty_path.stat().st_ino != empty_before.st_ino
    assert stat.S_IMODE(memory.stat().st_mode) == 0o755
    _assert_no_temporary_files(memory)


def test_replace_writes_exact_bytes_with_private_mode(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "HISTORY.md", b"old\x00bytes", 0o644)
    before = target.stat()
    content = b"replacement\nbytes\x00"

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        old = authority.snapshot_target("HISTORY.md")
        after = _replace(authority, old, content)

    observed = target.stat()
    assert target.read_bytes() == content
    assert stat.S_IMODE(observed.st_mode) == 0o600
    assert observed.st_ino != before.st_ino
    assert after.content == content
    assert after.stat_identity is not None
    assert after.stat_identity.inode == observed.st_ino
    assert after.stat_identity.links == 1
    _assert_no_temporary_files(memory)


def test_replace_rejects_receipt_and_invalid_content_without_mutation(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    receipt = _write_target(memory, ".consolidation.pending.json", b"receipt", 0o600)
    target = _write_target(memory, "MEMORY.md", b"old")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        receipt_snapshot = authority.snapshot_target(".consolidation.pending.json")
        with pytest.raises(storage.MemoryStorageError) as rejected_receipt:
            _replace(authority, receipt_snapshot, b"no")
        assert rejected_receipt.value.code == "invalid"

        snapshot = authority.snapshot_target("MEMORY.md")
        for invalid in (bytearray(b"no"), b"x" * (64 * 1024 * 1024 + 1)):
            with pytest.raises(storage.MemoryStorageError) as rejected_content:
                _replace(authority, snapshot, invalid)  # type: ignore[arg-type]
            assert rejected_content.value.code == "invalid"

    assert receipt.read_bytes() == b"receipt"
    assert target.read_bytes() == b"old"
    _assert_no_temporary_files(memory)


def test_replace_rejects_snapshot_from_other_authority(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_memory = first / "memory"
    second_memory = second / "memory"
    first_memory.mkdir(parents=True)
    second_memory.mkdir(parents=True)
    _write_target(first_memory, "MEMORY.md", b"first")
    second_target = _write_target(second_memory, "MEMORY.md", b"second")

    with storage.WorkspaceMemoryAuthority(first) as source:
        snapshot = source.snapshot_target("MEMORY.md")
        with storage.WorkspaceMemoryAuthority(second) as other:
            with pytest.raises(storage.MemoryStorageError) as rejected:
                _replace(other, snapshot, b"must not replace")
            assert rejected.value.code == "invalid"

    assert second_target.read_bytes() == b"second"
    _assert_no_temporary_files(second_memory)


def test_replace_rejects_snapshot_from_previous_authority_epoch(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"same bytes")
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    with authority:
        old = authority.snapshot_target("MEMORY.md")
    with authority:
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _replace(authority, old, b"must not replace")
        assert rejected.value.code == "binding"
    assert target.read_bytes() == b"same bytes"
    _assert_no_temporary_files(memory)


def test_replace_rejects_call_from_nonactivation_thread(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"same bytes")
    result: list[tuple[str, str] | tuple[str, Any]] = []

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")

        def replace_on_worker() -> None:
            try:
                _replace(authority, snapshot, b"must not replace")
            except storage.MemoryStorageError as error:
                result.append(("error", error.code))
            else:
                result.append(("ok", b"unexpected"))

        worker = threading.Thread(target=replace_on_worker)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()

    assert result == [("error", "closed")]
    assert target.read_bytes() == b"same bytes"
    _assert_no_temporary_files(memory)


def test_replace_rejects_copied_workspace_snapshot_without_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source_memory = source / "memory"
    source_memory.mkdir(parents=True)
    source_target = _write_target(source_memory, "MEMORY.md", b"source bytes")
    source_authority = storage.WorkspaceMemoryAuthority(source)
    with source_authority:
        snapshot = source_authority.snapshot_target("MEMORY.md")

    copied = tmp_path / "copied"
    shutil.copytree(source, copied)
    copied_memory = copied / "memory"
    copied_target = copied_memory / "MEMORY.md"
    with storage.WorkspaceMemoryAuthority(copied) as copied_authority:
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _replace(copied_authority, snapshot, b"copied replacement")
        assert rejected.value.code == "invalid"

    assert source_target.read_bytes() == b"source bytes"
    assert copied_target.read_bytes() == b"source bytes"
    _assert_no_temporary_files(source_memory)
    _assert_no_temporary_files(copied_memory)


def test_replace_detects_same_inode_bytes_change_when_mtime_is_restored(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"original")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        assert snapshot.stat_identity is not None
        os.utime(target, ns=(snapshot.stat_identity.mtime_ns, snapshot.stat_identity.mtime_ns))
        target.write_bytes(b"mutated!")
        os.utime(target, ns=(snapshot.stat_identity.mtime_ns, snapshot.stat_identity.mtime_ns))
        assert target.stat().st_ino == snapshot.stat_identity.inode
        assert target.stat().st_mtime_ns == snapshot.stat_identity.mtime_ns

        with pytest.raises(storage.MemoryStorageError) as rejected:
            _replace(authority, snapshot, b"replacement")
        assert rejected.value.code == "binding"

    assert target.read_bytes() == b"mutated!"
    _assert_no_temporary_files(memory)


def test_replace_detects_same_bytes_on_swapped_inode(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"same bytes")
    original_inode = target.stat().st_ino

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        target.unlink()
        _write_target(memory, "MEMORY.md", b"same bytes")
        assert target.stat().st_ino != original_inode
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _replace(authority, snapshot, b"replacement")
        assert rejected.value.code in {"binding", "corrupt"}

    assert target.read_bytes() == b"same bytes"
    _assert_no_temporary_files(memory)


def test_replace_rechecks_target_after_staging_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"original")
    original_fsync = mutation._fsync
    changed = False

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        assert snapshot.stat_identity is not None

        def change_target_after_stage_sync(fd: int) -> None:
            nonlocal changed
            original_fsync(fd)
            if not changed and stat.S_ISREG(os.fstat(fd).st_mode):
                changed = True
                target.write_bytes(b"mutated!")
                os.utime(
                    target,
                    ns=(snapshot.stat_identity.mtime_ns, snapshot.stat_identity.mtime_ns),
                )

        monkeypatch.setattr(mutation, "_fsync", change_target_after_stage_sync)
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _replace(authority, snapshot, b"replacement")
        assert rejected.value.code == "binding"

    assert changed
    assert target.read_bytes() == b"mutated!"
    _assert_no_temporary_files(memory)


def test_replace_rejects_symlink_target_without_touching_referent(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"legacy")
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        target.unlink()
        target.symlink_to(outside)
        with pytest.raises(storage.MemoryStorageError):
            _replace(authority, snapshot, b"replacement")

    assert target.is_symlink()
    assert outside.read_bytes() == b"outside"
    _assert_no_temporary_files(memory)


@pytest.mark.parametrize("rebound", ["parent", "lock"])
def test_replace_revalidates_parent_and_lock_after_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rebound: str
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"legacy")
    moved_memory = tmp_path / "memory-moved"
    lock_path = tmp_path / ".memory-recovery.lock"
    original_fsync = mutation._fsync
    rebound_once = False

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")

        def rebound_fsync(fd: int) -> None:
            nonlocal rebound_once
            original_fsync(fd)
            if rebound_once or not stat.S_ISREG(os.fstat(fd).st_mode):
                return
            rebound_once = True
            if rebound == "parent":
                memory.rename(moved_memory)
                memory.mkdir()
                _write_target(memory, "MEMORY.md", b"new parent")
            else:
                lock_path.unlink()
                lock_path.write_bytes(b"new lock")
                lock_path.chmod(0o600)

        monkeypatch.setattr(mutation, "_fsync", rebound_fsync)
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _replace(authority, snapshot, b"replacement")
        assert rejected.value.code in {"binding", "corrupt"}
        monkeypatch.setattr(mutation, "_fsync", original_fsync)

    assert rebound_once
    if rebound == "parent":
        assert (moved_memory / "MEMORY.md").read_bytes() == b"legacy"
        assert (memory / "MEMORY.md").read_bytes() == b"new parent"
        _assert_private_temporary_files(moved_memory)
        _assert_no_temporary_files(memory)
    else:
        assert target.read_bytes() == b"legacy"
        _assert_private_temporary_files(memory)


def test_partial_write_failure_cleans_stage_and_preserves_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"legacy")
    original_write = os.write
    calls = 0

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")

        def partial_then_fail(fd: int, content: bytes) -> int:
            nonlocal calls
            calls += 1
            if calls == 1:
                return original_write(fd, content[:2])
            raise OSError("injected staged write failure")

        monkeypatch.setattr(os, "write", partial_then_fail)
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _replace(authority, snapshot, b"replacement")
        assert rejected.value.code == "io"

    assert calls == 2
    assert target.read_bytes() == b"legacy"
    _assert_no_temporary_files(memory)


def test_stage_fsync_failure_is_pre_mutation_and_cleans_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"legacy")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        monkeypatch.setattr(
            mutation,
            "_fsync",
            lambda fd: (_ for _ in ()).throw(storage.MemoryStorageError("durability")),
        )
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _replace(authority, snapshot, b"replacement")
        assert rejected.value.code == "durability"

    assert target.read_bytes() == b"legacy"
    _assert_no_temporary_files(memory)


def test_rename_failure_is_uncertain_and_leaves_private_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"legacy")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")

        def fail_rename(*args: Any, **kwargs: Any) -> None:
            raise OSError("injected rename failure")

        monkeypatch.setattr(os, "rename", fail_rename)
        monkeypatch.setattr(
            os,
            "supports_dir_fd",
            set(os.supports_dir_fd) | {fail_rename},
        )
        with pytest.raises(mutation.MemoryTargetReplacementUncertainError) as rejected:
            _replace(authority, snapshot, b"replacement")
        assert rejected.value.code == "uncertain"
        assert str(rejected.value) == "memory target replacement outcome is uncertain"

    assert target.read_bytes() == b"legacy"
    stages = _temps(memory)
    assert len(stages) == 1
    assert stat.S_IMODE(stages[0].stat().st_mode) == 0o600


def test_post_rename_directory_fsync_failure_is_uncertain_and_resnapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"legacy")
    original_fsync = mutation._fsync

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")

        def fail_directory_sync(fd: int) -> None:
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise storage.MemoryStorageError("durability")
            original_fsync(fd)

        monkeypatch.setattr(mutation, "_fsync", fail_directory_sync)
        secret_content = b"PRIVATE-CONTENT-9b107f"
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _replace(authority, snapshot, secret_content)
        assert rejected.value.code == "uncertain"
        assert str(rejected.value) == "memory target replacement outcome is uncertain"
        assert secret_content.decode() not in str(rejected.value)
        monkeypatch.setattr(mutation, "_fsync", original_fsync)
        fresh = authority.snapshot_target("MEMORY.md")
        assert fresh.content == secret_content
        assert fresh.stat_identity is not None
        assert fresh.stat_identity.inode == target.stat().st_ino

    assert target.read_bytes() == secret_content
    _assert_no_temporary_files(memory)


def test_replace_handles_interrupted_and_short_stage_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"legacy")
    original_create = mutation._create_stage
    original_write = os.write
    original_read = os.read
    stage_fd: int | None = None
    write_calls = 0
    read_calls = 0

    def remember_stage(parent_fd: int) -> tuple[int, str]:
        nonlocal stage_fd
        stage_fd, name = original_create(parent_fd)
        return stage_fd, name

    def short_write(fd: int, content: bytes) -> int:
        nonlocal write_calls
        if fd != stage_fd:
            return original_write(fd, content)
        write_calls += 1
        if write_calls == 1:
            raise InterruptedError(errno.EINTR, "interrupted staged write")
        return original_write(fd, content[:3])

    def short_read(fd: int, size: int) -> bytes:
        nonlocal read_calls
        if fd != stage_fd:
            return original_read(fd, size)
        read_calls += 1
        if read_calls == 1:
            raise InterruptedError(errno.EINTR, "interrupted staged read")
        return original_read(fd, min(size, 5))

    monkeypatch.setattr(mutation, "_create_stage", remember_stage)
    monkeypatch.setattr(os, "write", short_write)
    monkeypatch.setattr(os, "read", short_read)
    content = b"multiple short write and read chunks"

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        result = _replace(authority, snapshot, content)

    assert result.content == content
    assert target.read_bytes() == content
    assert write_calls > 2
    assert read_calls > 2
    _assert_no_temporary_files(memory)


def test_corrupted_staged_bytes_fail_before_target_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"legacy")
    before = target.stat()
    original_fsync = mutation._fsync

    def corrupt_stage(fd: int) -> None:
        if stat.S_ISREG(os.fstat(fd).st_mode):
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, b"tampered!!!")
        original_fsync(fd)

    monkeypatch.setattr(mutation, "_fsync", corrupt_stage)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _replace(authority, snapshot, b"replacement")
        assert rejected.value.code == "corrupt"

    after = target.stat()
    assert target.read_bytes() == b"legacy"
    assert _stat_signature(after) == _stat_signature(before)
    _assert_no_temporary_files(memory)


def test_rename_that_succeeds_then_raises_is_uncertain_without_private_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"legacy")
    original_rename = os.rename

    def rename_then_raise(*args: Any, **kwargs: Any) -> None:
        original_rename(*args, **kwargs)
        raise OSError("secret rename diagnostic")

    monkeypatch.setattr(os, "rename", rename_then_raise)
    monkeypatch.setattr(
        os,
        "supports_dir_fd",
        set(os.supports_dir_fd) | {rename_then_raise},
    )
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        with pytest.raises(mutation.MemoryTargetReplacementUncertainError) as rejected:
            _replace(authority, snapshot, b"private replacement bytes")
        assert rejected.value.code == "uncertain"
        assert str(rejected.value) == "memory target replacement outcome is uncertain"
        assert "secret" not in str(rejected.value)

    assert target.read_bytes() == b"private replacement bytes"
    _assert_no_temporary_files(memory)


def test_parent_descriptor_is_duplicated_once_closed_and_safe_to_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"legacy")
    original_duplicate = storage.BoundMemoryTarget.duplicate_dir_fd
    duplicate_fds: list[int] = []

    def track_duplicate(scope: storage.BoundMemoryTarget) -> int:
        descriptor = original_duplicate(scope)
        duplicate_fds.append(descriptor)
        return descriptor

    monkeypatch.setattr(storage.BoundMemoryTarget, "duplicate_dir_fd", track_duplicate)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        with authority.operation() as operation:
            result = mutation.replace_target_if_unchanged(operation, snapshot, b"replacement")
            assert result.content == b"replacement"
            assert len(duplicate_fds) == 1
            duplicate_fd = duplicate_fds[0]
            with pytest.raises(OSError):
                os.fstat(duplicate_fd)
            keeper_fd = os.open(os.devnull, os.O_RDONLY)
            try:
                os.dup2(keeper_fd, duplicate_fd)
            finally:
                if keeper_fd != duplicate_fd:
                    os.close(keeper_fd)

        try:
            os.fstat(duplicate_fd)
        finally:
            os.close(duplicate_fd)

    assert target.read_bytes() == b"replacement"
    _assert_no_temporary_files(memory)


def test_missing_descriptor_relative_rename_capability_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write_target(memory, "MEMORY.md", b"legacy")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        monkeypatch.setattr(os, "rename", lambda *args, **kwargs: None)
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _replace(authority, snapshot, b"replacement")
        assert rejected.value.code == "unavailable"

    assert target.read_bytes() == b"legacy"
    _assert_no_temporary_files(memory)
