"""Scoped parent-directory borrowing for verified memory-target snapshots."""

from __future__ import annotations

import gc
import multiprocessing
import os
import stat
import subprocess
import sys
import textwrap
import threading
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent import memory_recovery_storage as storage


def _scope_capable() -> bool:
    if os.name != "posix":
        return False
    flags = ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")
    if any(not isinstance(getattr(os, name, None), int) for name in flags):
        return False
    required = {os.open, os.stat, os.mkdir, os.link}
    return required <= getattr(os, "supports_dir_fd", set()) and os.link in getattr(
        os, "supports_follow_symlinks", set()
    )


pytestmark = pytest.mark.skipif(
    not _scope_capable(), reason="target parent scopes require POSIX no-follow dir-fd support"
)


def _scope_child(bound: storage.BoundMemoryTarget, result: Any) -> None:
    try:
        duplicate = bound.duplicate_dir_fd()
    except storage.MemoryStorageError as error:
        result.put(("error", error.code))
    else:
        os.close(duplicate)
        result.put(("ok", "unexpected"))


def _target(memory: Path, name: str = "MEMORY.md", content: bytes = b"legacy") -> Path:
    target = memory / name
    target.write_bytes(content)
    if name == ".consolidation.pending.json":
        target.chmod(0o600)
    else:
        target.chmod(0o644)
    return target


def test_scoped_parent_reads_snapshot_and_leaves_target_unchanged(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir(mode=0o755)
    target = _target(memory, content=b"legacy\x00bytes")
    before = target.stat()
    before_bytes = target.read_bytes()

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        initial = authority.snapshot_target("MEMORY.md")
        assert initial.content == before_bytes
        with authority.operation() as operation:
            with operation.borrow_target_parent(initial) as bound:
                assert bound.name == "MEMORY.md"
                duplicate = bound.duplicate_dir_fd()
                try:
                    descriptor_info = os.fstat(duplicate)
                finally:
                    os.close(duplicate)
                memory_info = memory.stat()
                assert (descriptor_info.st_dev, descriptor_info.st_ino) == (
                    memory_info.st_dev,
                    memory_info.st_ino,
                )
                bound.require_current()
                current = bound.snapshot_target()
                assert current.name == initial.name
                assert current.content == before_bytes
                assert current.stat_identity is not None
                assert current.stat_identity.inode == before.st_ino

    after = target.stat()
    assert target.read_bytes() == before_bytes
    assert (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    assert stat.S_IMODE(memory.stat().st_mode) == 0o755


def test_parent_scope_blocks_operation_and_authority_close_until_exit(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        with authority.operation() as operation:
            bound = operation.borrow_target_parent(snapshot)
            with bound:
                with pytest.raises(storage.MemoryStorageError) as operation_close:
                    operation.__exit__(None, None, None)
                assert operation_close.value.code == "borrowed"
                with pytest.raises(storage.MemoryStorageError) as authority_close:
                    authority.close()
                assert authority_close.value.code == "borrowed"
                bound.require_current()

            operation.require_current()


def test_caller_duplicate_reuse_does_not_change_scope_owned_descriptor(
    tmp_path: Path,
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory)
    unrelated_fd: int | None = None

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        with authority.operation() as operation:
            with operation.borrow_target_parent(snapshot) as bound:
                caller_fd = bound.duplicate_dir_fd()
                os.close(caller_fd)
                unrelated_fd = os.open(os.devnull, os.O_RDONLY)
                assert unrelated_fd == caller_fd
                os.fstat(unrelated_fd)

    assert unrelated_fd is not None
    try:
        os.fstat(unrelated_fd)
    finally:
        os.close(unrelated_fd)


def test_parent_descriptor_closes_and_scope_is_invalid_after_exit(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        with authority.operation() as operation:
            bound = operation.borrow_target_parent(snapshot)
            with bound:
                borrowed_fd = bound._memory_fd
                assert borrowed_fd is not None
                caller_duplicate = bound.duplicate_dir_fd()
                os.fstat(borrowed_fd)
            with pytest.raises(OSError):
                os.fstat(borrowed_fd)
            os.fstat(caller_duplicate)
            os.close(caller_duplicate)
            for action in (
                bound.require_current,
                lambda: bound.name,
                bound.duplicate_dir_fd,
                bound.snapshot_target,
            ):
                with pytest.raises(storage.MemoryStorageError) as closed:
                    action()
                assert closed.value.code == "closed"
            with pytest.raises(storage.MemoryStorageError) as reentry:
                bound.__enter__()
            assert reentry.value.code == "closed"


def test_parent_scope_rejects_wrong_authority_snapshot(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_memory = first / "memory"
    second_memory = second / "memory"
    first_memory.mkdir(parents=True)
    second_memory.mkdir(parents=True)
    _target(first_memory, content=b"first")
    _target(second_memory, content=b"second")

    with storage.WorkspaceMemoryAuthority(first) as source:
        source_snapshot = source.snapshot_target("MEMORY.md")
        with storage.WorkspaceMemoryAuthority(second) as authority:
            with authority.operation() as operation:
                with pytest.raises(storage.MemoryStorageError) as rejected:
                    operation.borrow_target_parent(source_snapshot)
                assert rejected.value.code == "invalid"


def test_parent_scope_rejects_previous_authority_epoch(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory)
    authority = storage.WorkspaceMemoryAuthority(tmp_path)

    with authority:
        old_snapshot = authority.snapshot_target("MEMORY.md")
    with authority:
        with authority.operation() as operation:
            with pytest.raises(storage.MemoryStorageError) as rejected:
                operation.borrow_target_parent(old_snapshot)
            assert rejected.value.code == "binding"


def test_parent_scope_rejects_receipt_and_non_snapshot_values(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory, ".consolidation.pending.json", b"receipt")
    _target(memory)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        receipt = authority.snapshot_target(".consolidation.pending.json")
        with authority.operation() as operation:
            for invalid in (receipt, object()):
                with pytest.raises(storage.MemoryStorageError) as rejected:
                    operation.borrow_target_parent(invalid)  # type: ignore[arg-type]
                assert rejected.value.code == "invalid"


@pytest.mark.parametrize(
    ("target_name", "mutation"),
    [
        ("MEMORY.md", "name"),
        ("MEMORY.md", "content"),
        ("MEMORY.md", "stat"),
        ("history.jsonl", "absent-name"),
        ("history.jsonl", "absent-content"),
        ("MEMORY.md", "epoch"),
    ],
)
def test_parent_scope_rejects_preborrow_snapshot_field_changes(
    tmp_path: Path, target_name: str, mutation: str
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory, content=b"original")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target(target_name)
        empty_snapshot = authority.snapshot_target("MEMORY.md")
        with authority.operation() as operation:
            if mutation == "name":
                object.__setattr__(snapshot, "_name", "history.jsonl")
            elif mutation == "content":
                object.__setattr__(snapshot, "_content", b"tampered")
            elif mutation == "stat":
                assert snapshot.stat_identity is not None
                object.__setattr__(
                    snapshot,
                    "_stat_identity",
                    replace(snapshot.stat_identity, inode=snapshot.stat_identity.inode + 1),
                )
            elif mutation == "absent-name":
                object.__setattr__(snapshot, "_name", "HISTORY.md")
            elif mutation == "absent-content":
                object.__setattr__(snapshot, "_content", empty_snapshot._content)
                object.__setattr__(snapshot, "_stat_identity", empty_snapshot._stat_identity)
            else:
                object.__setattr__(snapshot, "_epoch", snapshot._epoch + 1)

            with pytest.raises(storage.MemoryStorageError) as rejected:
                operation.borrow_target_parent(snapshot)
            assert rejected.value.code in {"invalid", "binding"}


def test_parent_scope_rejects_preborrow_in_place_stat_identity_changes(
    tmp_path: Path,
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory, content=b"original")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        identity = snapshot.stat_identity
        assert identity is not None
        original_inode = identity.inode

        with authority.operation() as operation:
            try:
                object.__setattr__(identity, "inode", original_inode + 1)
                with pytest.raises(storage.MemoryStorageError) as rejected:
                    operation.borrow_target_parent(snapshot)
                assert rejected.value.code == "invalid"
            finally:
                object.__setattr__(identity, "inode", original_inode)

            with operation.borrow_target_parent(snapshot) as bound:
                bound.require_current()


def test_snapshot_issuance_entries_are_weak_and_cleaned_on_release(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        snapshot_id = id(snapshot)
        issuance = authority._target_snapshot_issuance[snapshot_id]
        assert isinstance(issuance[0], weakref.ReferenceType)
        assert issuance[0]() is snapshot
        del snapshot
        # Release occurs outside the lifecycle mutex; the weakref cleanup
        # removes the identity-keyed seal without retaining the snapshot.
        gc.collect()
        assert snapshot_id not in authority._target_snapshot_issuance


def test_snapshot_issuance_entries_clear_when_authority_closes(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory)
    authority = storage.WorkspaceMemoryAuthority(tmp_path)

    with authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        snapshot_id = id(snapshot)
        assert snapshot_id in authority._target_snapshot_issuance

    assert snapshot.content == b"legacy"
    assert authority._target_snapshot_issuance == {}


def test_snapshot_weakref_cleanup_does_not_deadlock_during_cycle_collection(
    tmp_path: Path,
) -> None:
    script = textwrap.dedent(
        """
        import gc
        import sys
        from pathlib import Path

        from nanobot.agent import memory_recovery_storage as storage

        workspace = Path(sys.argv[1])
        memory = workspace / "memory"
        memory.mkdir()
        (memory / "MEMORY.md").write_bytes(b"legacy")

        with storage.WorkspaceMemoryAuthority(workspace) as authority:
            snapshot = authority.snapshot_target("MEMORY.md")
            object.__setattr__(snapshot, "_content", snapshot)
            del snapshot
            with authority._lifecycle_lock:
                gc.collect()
            assert not authority._target_snapshot_issuance
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr


def test_parent_scope_rejects_unissued_snapshot_clone_with_matching_fields(
    tmp_path: Path,
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory, content=b"original")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        forged = storage.MemoryTargetSnapshot(
            snapshot.name,
            snapshot.content,
            snapshot.stat_identity,
            authority,
            snapshot._epoch,
            _capability=storage._SNAPSHOT_CAPABILITY,
        )
        with authority.operation() as operation:
            with pytest.raises(storage.MemoryStorageError) as rejected:
                operation.borrow_target_parent(forged)
            assert rejected.value.code == "invalid"


@pytest.mark.parametrize("field", ["_name", "_content", "_stat_identity", "_epoch"])
def test_parent_scope_rechecks_original_snapshot_seal_on_each_use(
    tmp_path: Path, field: str
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory, content=b"original")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        original = getattr(snapshot, field)
        if field == "_name":
            changed: object = "HISTORY.md"
        elif field == "_content":
            changed = b"tampered"
        elif field == "_stat_identity":
            assert snapshot.stat_identity is not None
            changed = replace(snapshot.stat_identity, inode=snapshot.stat_identity.inode + 1)
        else:
            changed = snapshot._epoch + 1

        with authority.operation() as operation:
            with operation.borrow_target_parent(snapshot) as bound:
                object.__setattr__(snapshot, field, changed)
                with pytest.raises(storage.MemoryStorageError):
                    bound.duplicate_dir_fd()
                object.__setattr__(snapshot, field, original)
                bound.require_current()


def test_parent_scope_rechecks_in_place_stat_identity_changes_on_each_use(
    tmp_path: Path,
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory, content=b"original")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        identity = snapshot.stat_identity
        assert identity is not None
        original_inode = identity.inode

        with authority.operation() as operation:
            with operation.borrow_target_parent(snapshot) as bound:
                try:
                    object.__setattr__(identity, "inode", original_inode + 1)
                    with pytest.raises(storage.MemoryStorageError) as rejected:
                        bound.duplicate_dir_fd()
                    assert rejected.value.code == "invalid"
                finally:
                    object.__setattr__(identity, "inode", original_inode)
                bound.require_current()


def test_parent_scope_rejects_cross_thread_access_without_opening_on_worker(
    tmp_path: Path,
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory)
    results: list[tuple[str, str]] = []

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        with authority.operation() as operation:
            bound = operation.borrow_target_parent(snapshot)
            with bound:
                def access_from_worker() -> None:
                    try:
                        duplicate = bound.duplicate_dir_fd()
                    except storage.MemoryStorageError as error:
                        results.append(("error", error.code))
                    else:
                        os.close(duplicate)
                        results.append(("ok", "unexpected"))

                worker = threading.Thread(target=access_from_worker)
                worker.start()
                worker.join(timeout=5)
                assert not worker.is_alive()
                assert results == [("error", "closed")]
                bound.require_current()


def test_parent_scope_rejects_use_after_fork(tmp_path: Path) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork is unavailable")
    memory = tmp_path / "memory"
    memory.mkdir()
    _target(memory)
    context = multiprocessing.get_context("fork")
    result = context.Queue()

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        with authority.operation() as operation:
            with operation.borrow_target_parent(snapshot) as bound:
                child = context.Process(target=_scope_child, args=(bound, result))
                child.start()
                child.join(timeout=10)
                assert not child.is_alive()
                assert child.exitcode == 0
                assert result.get(timeout=2) == ("error", "closed")
                bound.require_current()
    result.close()
    result.join_thread()


@pytest.mark.parametrize("rebound", ["memory", "lock"])
def test_parent_scope_rejects_parent_or_lock_binding_changes(
    tmp_path: Path, rebound: str
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _target(memory)
    moved_memory = tmp_path / "memory-moved"
    lock = tmp_path / ".memory-recovery.lock"

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        with authority.operation() as operation:
            bound = operation.borrow_target_parent(snapshot)
            with bound:
                borrowed_fd = bound._memory_fd
                assert borrowed_fd is not None
                if rebound == "memory":
                    memory.rename(moved_memory)
                    memory.mkdir()
                    _target(memory, content=b"replacement parent")
                else:
                    lock.unlink()
                    lock.write_bytes(b"replacement lock")
                    lock.chmod(0o600)
                with pytest.raises(storage.MemoryStorageError) as rejected:
                    bound.require_current()
                assert rejected.value.code in {"binding", "corrupt"}
            with pytest.raises(OSError):
                os.fstat(borrowed_fd)

    if rebound == "memory":
        assert (moved_memory / "MEMORY.md").read_bytes() == b"legacy"
        assert (memory / "MEMORY.md").read_bytes() == b"replacement parent"
    else:
        assert target.read_bytes() == b"legacy"


def test_scope_is_bound_to_its_captured_snapshot_name(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _target(memory, content=b"legacy")
    _target(memory, "history.jsonl", b"history")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        with authority.operation() as operation:
            with operation.borrow_target_parent(snapshot) as bound:
                object.__setattr__(snapshot, "_name", "history.jsonl")
                with pytest.raises(storage.MemoryStorageError) as rejected:
                    bound.snapshot_target()
                assert rejected.value.code == "invalid"
                object.__setattr__(snapshot, "_name", "MEMORY.md")
                assert bound.snapshot_target().content == b"legacy"

    assert target.read_bytes() == b"legacy"
