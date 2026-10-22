"""Nonmutating parent borrowing for an authentic consolidation receipt snapshot."""

from __future__ import annotations

import multiprocessing
import os
import stat
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent import memory_recovery_storage as storage

_RECEIPT = ".consolidation.pending.json"


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
    not _scope_capable(), reason="receipt parent scopes require POSIX no-follow dir-fd support"
)


def _receipt(memory: Path, content: bytes = b"pending receipt") -> Path:
    target = memory / _RECEIPT
    target.write_bytes(content)
    target.chmod(0o600)
    return target


def _child_require_current(scope: storage.BoundMemoryTarget, result: Any) -> None:
    try:
        scope.require_current()
    except storage.MemoryStorageError as error:
        result.put(("error", error.code))
    else:
        result.put(("ok", "unexpected"))


def test_receipt_scope_is_present_private_and_nonmutating(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _receipt(memory, b"private pending bytes\x00")
    before = os.stat(target, follow_symlinks=False)
    before_bytes = target.read_bytes()

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target(_RECEIPT)
        with authority.operation() as operation:
            with pytest.raises(storage.MemoryStorageError) as ordinary_scope:
                operation.borrow_target_parent(snapshot)
            assert ordinary_scope.value.code == "invalid"

            with operation.borrow_pending_receipt_parent(snapshot) as scope:
                assert scope.name == _RECEIPT
                scope.require_current()
                current = scope.snapshot_target()
                assert current.name == _RECEIPT
                assert current.content == before_bytes
                assert current.stat_identity is not None
                assert stat.S_IMODE(current.stat_identity.mode) == 0o600

                caller_fd = scope.duplicate_dir_fd()
                try:
                    parent = os.fstat(caller_fd)
                    bound_parent = os.stat(".", dir_fd=caller_fd)
                    assert (parent.st_dev, parent.st_ino) == (
                        bound_parent.st_dev,
                        bound_parent.st_ino,
                    )
                    scope_fd = scope._memory_fd
                    assert scope_fd is not None
                    os.close(caller_fd)
                    replacement_fd = os.open(os.devnull, os.O_RDONLY)
                    try:
                        assert replacement_fd == caller_fd
                        os.fstat(replacement_fd)
                        scope.require_current()
                        os.fstat(scope_fd)
                    finally:
                        os.close(replacement_fd)
                finally:
                    # The descriptor may already have been closed and reused above.
                    try:
                        os.fstat(caller_fd)
                    except OSError:
                        pass
                    else:
                        os.close(caller_fd)

    after = os.stat(target, follow_symlinks=False)
    assert target.read_bytes() == before_bytes
    assert (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )


def test_receipt_scope_rejects_absent_swapped_and_forged_snapshots(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _receipt(memory)
    (memory / "MEMORY.md").write_bytes(b"ordinary")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        absent = authority.snapshot_target("HISTORY.md")
        receipt = authority.snapshot_target(_RECEIPT)
        ordinary = authority.snapshot_target("MEMORY.md")
        forged = storage.MemoryTargetSnapshot(
            receipt.name,
            receipt.content,
            receipt.stat_identity,
            authority,
            receipt._epoch,
            _capability=storage._SNAPSHOT_CAPABILITY,
        )
        with authority.operation() as operation:
            for invalid in (absent, ordinary, forged, object()):
                with pytest.raises(storage.MemoryStorageError):
                    operation.borrow_pending_receipt_parent(invalid)  # type: ignore[arg-type]

            original = receipt._content
            object.__setattr__(receipt, "_content", b"swapped")
            with pytest.raises(storage.MemoryStorageError) as changed:
                operation.borrow_pending_receipt_parent(receipt)
            assert changed.value.code == "invalid"
            object.__setattr__(receipt, "_content", original)
            with operation.borrow_pending_receipt_parent(receipt) as scope:
                scope.require_current()


def test_receipt_scope_rejects_wrong_authority_and_previous_epoch(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_memory = first / "memory"
    second_memory = second / "memory"
    first_memory.mkdir(parents=True)
    second_memory.mkdir(parents=True)
    _receipt(first_memory)
    _receipt(second_memory)

    authority = storage.WorkspaceMemoryAuthority(first)
    with authority:
        stale = authority.snapshot_target(_RECEIPT)
    with authority:
        current = authority.snapshot_target(_RECEIPT)
        with storage.WorkspaceMemoryAuthority(second) as other:
            foreign = other.snapshot_target(_RECEIPT)
            with authority.operation() as operation:
                with pytest.raises(storage.MemoryStorageError) as old:
                    operation.borrow_pending_receipt_parent(stale)
                assert old.value.code == "binding"
                with pytest.raises(storage.MemoryStorageError) as wrong_authority:
                    operation.borrow_pending_receipt_parent(foreign)
                assert wrong_authority.value.code == "invalid"
                with operation.borrow_pending_receipt_parent(current) as scope:
                    scope.require_current()


@pytest.mark.parametrize("entry", ["mode", "hardlink", "symlink"])
def test_receipt_snapshot_rejects_unsafe_file_entries_without_mutation(
    tmp_path: Path, entry: str
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _receipt(memory)
    if entry == "mode":
        target.chmod(0o644)
    elif entry == "hardlink":
        os.link(target, memory / "linked-receipt")
    else:
        target.unlink()
        target.symlink_to(memory / "missing-target")

    before = os.lstat(target)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError):
            authority.snapshot_target(_RECEIPT)
    after = os.lstat(target)
    assert (after.st_dev, after.st_ino, after.st_mode, after.st_nlink) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
    )


def test_receipt_scope_enforces_borrower_lifetime_and_closed_state(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _receipt(memory)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target(_RECEIPT)
        with authority.operation() as operation:
            scope = operation.borrow_pending_receipt_parent(snapshot)
            with scope:
                with pytest.raises(storage.MemoryStorageError) as operation_close:
                    operation.__exit__(None, None, None)
                assert operation_close.value.code == "borrowed"
                with pytest.raises(storage.MemoryStorageError) as authority_close:
                    authority.close()
                assert authority_close.value.code == "borrowed"
                scope.require_current()
            for action in (
                scope.require_current,
                lambda: scope.name,
                scope.duplicate_dir_fd,
                scope.snapshot_target,
            ):
                with pytest.raises(storage.MemoryStorageError) as closed:
                    action()
                assert closed.value.code == "closed"


def test_receipt_scope_rejects_cross_thread_access(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _receipt(memory)
    outcomes: list[tuple[str, str]] = []

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target(_RECEIPT)
        with authority.operation() as operation:
            with operation.borrow_pending_receipt_parent(snapshot) as scope:
                def use_scope() -> None:
                    try:
                        scope.require_current()
                    except storage.MemoryStorageError as error:
                        outcomes.append(("error", error.code))
                    else:
                        outcomes.append(("ok", "unexpected"))

                worker = threading.Thread(target=use_scope)
                worker.start()
                worker.join(timeout=3)
                assert not worker.is_alive()

    assert outcomes == [("error", "closed")]


def test_receipt_scope_rejects_access_from_forked_process(tmp_path: Path) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork is unavailable")
    memory = tmp_path / "memory"
    memory.mkdir()
    _receipt(memory)
    context = multiprocessing.get_context("fork")
    result = context.Queue()
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target(_RECEIPT)
        with authority.operation() as operation:
            with operation.borrow_pending_receipt_parent(snapshot) as scope:
                child = context.Process(target=_child_require_current, args=(scope, result))
                child.start()
                child.join(timeout=3)
                if child.is_alive():
                    child.terminate()
                    child.join()
                    pytest.fail("forked receipt-scope check timed out")
                assert child.exitcode == 0
                assert result.get(timeout=1) == ("error", "closed")
    result.close()
    result.join_thread()


@pytest.mark.parametrize("rebound", ["memory", "lock"])
def test_receipt_scope_detects_parent_or_lock_rebinding(
    tmp_path: Path, rebound: str
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    _receipt(memory)
    lock_path = tmp_path / ".memory-recovery.lock"

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target(_RECEIPT)
        with authority.operation() as operation:
            with operation.borrow_pending_receipt_parent(snapshot) as scope:
                if rebound == "memory":
                    moved = tmp_path / "moved-memory"
                    memory.rename(moved)
                    memory.mkdir()
                else:
                    moved = tmp_path / "moved-lock"
                    lock_path.rename(moved)
                    lock_path.write_bytes(b"replacement lock")
                    lock_path.chmod(0o600)
                with pytest.raises(storage.MemoryStorageError) as changed:
                    scope.require_current()
                assert changed.value.code == "binding"


def test_receipt_snapshot_weakref_cleanup_is_safe_during_cycle_collection(
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
        target = memory / ".consolidation.pending.json"
        target.write_bytes(b"receipt")
        target.chmod(0o600)

        with storage.WorkspaceMemoryAuthority(workspace) as authority:
            snapshot = authority.snapshot_target(".consolidation.pending.json")
            with authority.operation() as operation:
                scope = operation.borrow_pending_receipt_parent(snapshot)
                scope.__enter__()
                scope.__exit__(None, None, None)
                object.__setattr__(snapshot, "_content", snapshot)
                del scope, snapshot
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


def test_receipt_scope_storage_boundary_remains_nonmutating() -> None:
    source = Path(storage.__file__).read_text(encoding="utf-8")
    for forbidden in ("os.replace(", "os.rename(", "os.chmod(", "os.unlink("):
        assert forbidden not in source
