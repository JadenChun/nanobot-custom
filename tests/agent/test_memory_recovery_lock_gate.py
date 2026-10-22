"""Focused tests for the lock-only workspace memory gate."""

from __future__ import annotations

import multiprocessing
import os
import stat
import threading
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent import memory_recovery_storage as storage


def _storage_capable() -> bool:
    if os.name != "posix":
        return False
    if any(
        not isinstance(getattr(os, name, None), int)
        for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")
    ):
        return False
    if not all(
        function in getattr(os, "supports_dir_fd", set())
        for function in (os.open, os.stat, os.mkdir, os.link)
    ):
        return False
    if os.link not in getattr(os, "supports_follow_symlinks", set()):
        return False
    return (
        getattr(storage, "_fcntl", None) is not None
        and isinstance(getattr(storage._fcntl, "LOCK_NB", None), int)
    )


def _fork_capable() -> bool:
    if not _storage_capable():
        return False
    try:
        multiprocessing.get_context("fork")
    except (AttributeError, ValueError):
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _storage_capable(), reason="workspace memory locks require POSIX no-follow dir-fd/flock"
)


def _snapshot(path: Path) -> tuple[int, int, int, int, int]:
    info = path.lstat()
    return info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid


def test_gate_requires_nonblocking_flock_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert storage._fcntl is not None
    monkeypatch.setattr(storage._fcntl, "LOCK_NB", None)
    with pytest.raises(storage.MemoryStorageUnavailableError):
        storage.open_workspace_memory_lock(tmp_path)

    # The strict authority does not depend on the gate-only probe capability.
    with storage.WorkspaceMemoryAuthority(tmp_path):
        pass


def test_gate_requires_an_existing_workspace(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(storage.MemoryStorageError) as failure:
        storage.open_workspace_memory_lock(missing).__enter__()
    assert failure.value.code in {"io", "corrupt"}
    assert not missing.exists()


def test_existing_only_gate_never_creates_missing_workspace_or_lock(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(storage.MemoryStorageError):
        with storage.open_existing_workspace_memory_lock(missing):
            pytest.fail("missing workspace accepted")
    assert not missing.exists()
    with pytest.raises(storage.MemoryStorageError) as failure:
        with storage.open_existing_workspace_memory_lock(tmp_path):
            pytest.fail("missing lock accepted")
    assert failure.value.code == "missing"
    assert list(tmp_path.iterdir()) == []
    with storage.open_workspace_memory_lock(tmp_path):
        pass
    with storage.open_existing_workspace_memory_lock(tmp_path) as gate:
        gate.require_current()
    assert not (tmp_path / "memory").exists()


def test_existing_only_authority_requires_complete_layout_without_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with storage.open_workspace_memory_lock(tmp_path):
        pass
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    with pytest.raises(storage.MemoryStorageError):
        with storage.open_existing_workspace_memory(tmp_path):
            pytest.fail("incomplete layout accepted")
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    with storage.open_workspace_memory(tmp_path):
        pass

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("existing-only authority attempted a create or repair")

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_sync_directory_boundary", forbidden)
        with storage.open_existing_workspace_memory(tmp_path) as authority:
            with authority.operation() as operation:
                operation.require_current()
                with pytest.raises(storage.MemoryStorageError):
                    authority.open_evidence_namespace("transactions", "a" * 64)


def test_existing_only_gate_rejects_unsafe_lock(tmp_path: Path) -> None:
    lock = tmp_path / ".memory-recovery.lock"
    lock.write_bytes(b"private")
    lock.chmod(0o644)
    before = _snapshot(lock)
    with pytest.raises(storage.MemoryStorageError):
        with storage.open_existing_workspace_memory_lock(tmp_path):
            pytest.fail("unsafe lock accepted")
    assert _snapshot(lock) == before


def test_gate_leaves_absent_and_ordinary_memory_layout_untouched(tmp_path: Path) -> None:
    tmp_path.chmod(0o755)
    lock_path = tmp_path / ".memory-recovery.lock"

    with storage.open_workspace_memory_lock(tmp_path) as gate:
        assert gate.workspace == tmp_path.resolve()
        assert gate.lock_path == lock_path
        gate.require_current()
        assert not (tmp_path / "memory").exists()

    lock_inode = lock_path.stat().st_ino
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
    assert not (tmp_path / "memory").exists()

    memory = tmp_path / "memory"
    memory.mkdir(mode=0o755)
    memory.chmod(0o755)
    marker = memory / "existing.txt"
    marker.write_bytes(b"ordinary workspace memory")
    before_memory = _snapshot(memory)
    before_marker = _snapshot(marker)

    with storage.open_workspace_memory_lock(tmp_path) as gate:
        gate.require_current()
        assert gate.workspace == tmp_path.resolve()

    assert _snapshot(memory) == before_memory
    assert _snapshot(marker) == before_marker
    assert marker.read_bytes() == b"ordinary workspace memory"
    assert not (memory / ".recovery").exists()
    assert lock_path.stat().st_ino == lock_inode


def test_fresh_lock_is_private_persistent_and_durably_synced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fsync = storage.os.fsync
    synced: set[tuple[int, int]] = set()

    def recording_fsync(fd: int) -> None:
        info = os.fstat(fd)
        synced.add((info.st_dev, info.st_ino))
        real_fsync(fd)

    monkeypatch.setattr(storage.os, "fsync", recording_fsync)
    lock_path = tmp_path / ".memory-recovery.lock"

    with storage.open_workspace_memory_lock(tmp_path) as gate:
        assert gate.require_current() is None

    info = lock_path.stat()
    assert stat.S_IMODE(info.st_mode) == 0o600
    assert info.st_nlink == 1
    assert (info.st_dev, info.st_ino) in synced
    workspace_info = tmp_path.stat()
    assert (workspace_info.st_dev, workspace_info.st_ino) in synced
    inode = info.st_ino

    with storage.open_workspace_memory_lock(tmp_path):
        pass
    assert lock_path.stat().st_ino == inode
    assert not (tmp_path / "memory").exists()


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "mode", "directory"])
def test_unsafe_existing_lock_is_rejected_without_mutation(
    tmp_path: Path, unsafe: str
) -> None:
    lock_path = tmp_path / ".memory-recovery.lock"
    outside = tmp_path / "outside"
    outside.write_bytes(b"sentinel")
    outside.chmod(0o600)

    if unsafe == "symlink":
        lock_path.symlink_to(outside)
    elif unsafe == "hardlink":
        os.link(outside, lock_path)
    elif unsafe == "mode":
        lock_path.write_bytes(b"sentinel")
        lock_path.chmod(0o644)
    else:
        lock_path.mkdir(mode=0o700)

    before_lock = _snapshot(lock_path)
    before_outside = _snapshot(outside)
    with pytest.raises(storage.MemoryStorageError) as failure:
        with storage.open_workspace_memory_lock(tmp_path):
            pytest.fail("unsafe lock was accepted")

    assert failure.value.__cause__ is None
    assert failure.value.__suppress_context__
    assert "sentinel" not in str(failure.value)
    assert str(tmp_path) not in str(failure.value)
    assert _snapshot(lock_path) == before_lock
    assert _snapshot(outside) == before_outside
    assert outside.read_bytes() == b"sentinel"
    assert not (tmp_path / "memory").exists()


def test_lock_owner_mismatch_is_rejected_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / ".memory-recovery.lock"
    lock_path.touch(mode=0o600)
    lock_path.chmod(0o600)
    before = _snapshot(lock_path)
    real_uid = os.geteuid() if hasattr(os, "geteuid") else os.getuid()
    calls = 0

    def uid() -> int:
        nonlocal calls
        calls += 1
        # Workspace validation runs twice before the lock's owner check.
        return real_uid if calls <= 2 else real_uid + 1

    monkeypatch.setattr(storage, "_uid", uid)
    with pytest.raises(storage.MemoryStorageError) as failure:
        with storage.open_workspace_memory_lock(tmp_path):
            pytest.fail("wrong-owner lock was accepted")

    assert failure.value.code == "corrupt"
    assert _snapshot(lock_path) == before
    assert not (tmp_path / "memory").exists()


@pytest.mark.parametrize("check", ["require_current", "clean_exit"])
def test_replacement_is_rejected_by_current_check_and_clean_exit(
    tmp_path: Path, check: str
) -> None:
    lock_path = tmp_path / ".memory-recovery.lock"
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"keep replacement")
    replacement.chmod(0o600)

    gate = storage.open_workspace_memory_lock(tmp_path)
    gate.__enter__()
    original = tmp_path / "original-lock"
    lock_path.rename(original)
    replacement.rename(lock_path)
    before_rejection = _snapshot(lock_path)
    try:
        with pytest.raises(storage.MemoryStorageError) as failure:
            if check == "require_current":
                gate.require_current()
            else:
                gate.__exit__(None, None, None)
        assert failure.value.code == "binding"
        assert _snapshot(lock_path) == before_rejection
        assert lock_path.read_bytes() == b"keep replacement"
        assert (tmp_path / "original-lock").exists()
        assert not (tmp_path / "memory").exists()
    finally:
        gate.close()


def test_gate_is_single_use_and_close_detaches_descriptors(tmp_path: Path) -> None:
    gate = storage.open_workspace_memory_lock(tmp_path)
    gate.__enter__()
    with pytest.raises(storage.MemoryStorageError) as active_reentry:
        gate.__enter__()
    assert active_reentry.value.code == "closed"
    gate.require_current()
    gate.close()
    assert gate._workspace_fd is None
    assert gate._lock_fd is None

    unrelated_fd = os.open(os.devnull, os.O_RDONLY)
    try:
        gate.close()
        os.fstat(unrelated_fd)
        with pytest.raises(storage.MemoryStorageError) as reentry:
            gate.__enter__()
        assert reentry.value.code == "closed"
    finally:
        os.close(unrelated_fd)


def test_close_skips_reused_lock_descriptor_without_unlocking_it(tmp_path: Path) -> None:
    assert storage._fcntl is not None
    gate = storage.open_workspace_memory_lock(tmp_path)
    gate.__enter__()
    lock_fd = gate._lock_fd
    assert lock_fd is not None

    unrelated = tmp_path / "unrelated.lock"
    unrelated.write_bytes(b"unrelated lock")
    unrelated_fd: int | None = None
    probe_fd: int | None = None
    try:
        os.close(lock_fd)
        unrelated_fd = os.open(unrelated, os.O_RDWR)
        assert unrelated_fd == lock_fd
        storage._fcntl.flock(unrelated_fd, storage._fcntl.LOCK_EX | storage._fcntl.LOCK_NB)

        with pytest.raises(storage.MemoryStorageError) as failure:
            gate.close()
        assert failure.value.code == "binding"
        assert os.fstat(unrelated_fd).st_ino == unrelated.stat().st_ino

        probe_fd = os.open(unrelated, os.O_RDWR)
        with pytest.raises(BlockingIOError):
            storage._fcntl.flock(probe_fd, storage._fcntl.LOCK_EX | storage._fcntl.LOCK_NB)
    finally:
        gate.close()
        if probe_fd is not None:
            os.close(probe_fd)
        if unrelated_fd is not None:
            storage._fcntl.flock(unrelated_fd, storage._fcntl.LOCK_UN)
            os.close(unrelated_fd)


def test_close_skips_same_inode_reopened_lock_descriptor(tmp_path: Path) -> None:
    assert storage._fcntl is not None
    gate = storage.open_workspace_memory_lock(tmp_path)
    gate.__enter__()
    lock_fd = gate._lock_fd
    assert lock_fd is not None
    original_info = os.fstat(lock_fd)
    suspect_fd: int | None = None

    try:
        os.close(lock_fd)
        suspect_fd = os.open(gate.lock_path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
        assert suspect_fd == lock_fd
        suspect_info = os.fstat(suspect_fd)
        assert (suspect_info.st_dev, suspect_info.st_ino) == (
            original_info.st_dev,
            original_info.st_ino,
        )

        with pytest.raises(BlockingIOError):
            storage._fcntl.flock(
                suspect_fd, storage._fcntl.LOCK_EX | storage._fcntl.LOCK_NB
            )
        with pytest.raises(storage.MemoryStorageError) as current_failure:
            gate.require_current()
        assert current_failure.value.code == "binding"

        with pytest.raises(storage.MemoryStorageError) as close_failure:
            gate.close()
        assert close_failure.value.code == "binding"
        assert os.fstat(suspect_fd).st_ino == suspect_info.st_ino
        # The anchor released the gate lock; the still-open suspect fd can now lock it.
        storage._fcntl.flock(suspect_fd, storage._fcntl.LOCK_EX | storage._fcntl.LOCK_NB)
        storage._fcntl.flock(suspect_fd, storage._fcntl.LOCK_UN)
    finally:
        try:
            gate.close()
        except storage.MemoryStorageError:
            pass
        if suspect_fd is not None:
            try:
                storage._fcntl.flock(suspect_fd, storage._fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(suspect_fd)
            except OSError:
                pass


def test_close_skips_reused_workspace_descriptor(tmp_path: Path) -> None:
    gate = storage.open_workspace_memory_lock(tmp_path)
    gate.__enter__()
    workspace_fd = gate._workspace_fd
    assert workspace_fd is not None

    unrelated_directory = tmp_path / "unrelated-directory"
    unrelated_directory.mkdir()
    unrelated_fd: int | None = None
    try:
        os.close(workspace_fd)
        unrelated_fd = os.open(unrelated_directory, os.O_RDONLY | os.O_DIRECTORY)
        assert unrelated_fd == workspace_fd

        with pytest.raises(storage.MemoryStorageError) as failure:
            gate.close()
        assert failure.value.code == "binding"
        assert stat.S_ISDIR(os.fstat(unrelated_fd).st_mode)
        assert unrelated_directory.is_dir()
    finally:
        gate.close()
        if unrelated_fd is not None:
            os.close(unrelated_fd)


def test_exceptional_exit_validates_replacement_and_preserves_body_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert storage._fcntl is not None
    gate = storage.open_workspace_memory_lock(tmp_path)
    original_require_current = gate.require_current
    validation_errors: list[str] = []

    def recording_require_current() -> None:
        try:
            original_require_current()
        except storage.MemoryStorageError as error:
            validation_errors.append(error.code)
            raise

    monkeypatch.setattr(gate, "require_current", recording_require_current)
    lock_path = tmp_path / ".memory-recovery.lock"
    original_lock = tmp_path / "original-lock"
    replacement = tmp_path / "replacement-lock"
    unrelated = tmp_path / "unrelated.lock"
    unrelated.write_bytes(b"unrelated lock")
    unrelated_fd: int | None = None
    probe_fd: int | None = None
    body_error = RuntimeError("preserve the body failure")

    try:
        with pytest.raises(RuntimeError) as failure:
            with gate:
                lock_fd = gate._lock_fd
                assert lock_fd is not None
                os.close(lock_fd)
                unrelated_fd = os.open(unrelated, os.O_RDWR)
                assert unrelated_fd == lock_fd
                storage._fcntl.flock(
                    unrelated_fd, storage._fcntl.LOCK_EX | storage._fcntl.LOCK_NB
                )

                lock_path.rename(original_lock)
                replacement.write_bytes(b"replacement")
                replacement.chmod(0o600)
                replacement.rename(lock_path)
                raise body_error

        assert failure.value is body_error
        assert validation_errors == ["binding"]
        assert os.fstat(unrelated_fd).st_ino == unrelated.stat().st_ino
        assert lock_path.read_bytes() == b"replacement"
        assert original_lock.exists()
        assert not (tmp_path / "memory").exists()

        probe_fd = os.open(unrelated, os.O_RDWR)
        with pytest.raises(BlockingIOError):
            storage._fcntl.flock(probe_fd, storage._fcntl.LOCK_EX | storage._fcntl.LOCK_NB)
    finally:
        gate.close()
        if probe_fd is not None:
            os.close(probe_fd)
        if unrelated_fd is not None:
            storage._fcntl.flock(unrelated_fd, storage._fcntl.LOCK_UN)
            os.close(unrelated_fd)


def test_gate_is_bound_to_its_creating_thread(tmp_path: Path) -> None:
    gate = storage.open_workspace_memory_lock(tmp_path)
    gate.__enter__()
    result: list[tuple[str, str | None]] = []

    def use_from_other_thread() -> None:
        try:
            gate.close()
        except storage.MemoryStorageError as error:
            result.append((error.code, str(error)))
        else:
            result.append(("closed", None))

    worker = threading.Thread(target=use_from_other_thread)
    worker.start()
    worker.join(timeout=3)
    try:
        assert not worker.is_alive()
        assert result == [("closed", "memory recovery authority is closed")]
        gate.require_current()
    finally:
        gate.close()


def _contending_child(
    workspace: str,
    kind: str,
    started: Any,
    acquired: Any,
    result: Any,
) -> None:
    assert storage._fcntl is not None
    real_flock = storage._fcntl.flock

    def signaling_flock(fd: int, operation: int, *args: object) -> None:
        if operation == storage._fcntl.LOCK_EX:
            started.set()
        real_flock(fd, operation, *args)

    storage._fcntl.flock = signaling_flock
    try:
        manager = (
            storage.open_workspace_memory_lock(Path(workspace))
            if kind == "gate"
            else storage.WorkspaceMemoryAuthority(Path(workspace))
        )
        with manager:
            acquired.set()
        result.put(("ok",))
    except BaseException as error:
        result.put(("error", type(error).__name__, getattr(error, "code", None), str(error)))


def _assert_contends_with(
    workspace: Path, parent_kind: str, child_kind: str
) -> None:
    workspace.mkdir()
    context = multiprocessing.get_context("fork")
    started = context.Event()
    acquired = context.Event()
    result = context.Queue()
    manager = (
        storage.open_workspace_memory_lock(workspace)
        if parent_kind == "gate"
        else storage.WorkspaceMemoryAuthority(workspace)
    )
    process = context.Process(
        target=_contending_child,
        args=(str(workspace), child_kind, started, acquired, result),
    )

    try:
        with manager:
            process.start()
            assert started.wait(timeout=4)
            assert not acquired.wait(timeout=0.15)

        assert acquired.wait(timeout=4)
        process.join(timeout=4)
        assert not process.is_alive()
        assert result.get(timeout=2) == ("ok",)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)


def test_gate_contends_with_strict_authority_in_both_directions(tmp_path: Path) -> None:
    if not _fork_capable():
        pytest.skip("cross-authority contention requires multiprocessing fork")
    _assert_contends_with(tmp_path / "gate-parent", "gate", "authority")
    _assert_contends_with(tmp_path / "authority-parent", "authority", "gate")


def test_lock_replacement_while_blocked_is_rejected_after_flock(tmp_path: Path) -> None:
    if not _fork_capable():
        pytest.skip("blocked replacement check requires multiprocessing fork")
    context = multiprocessing.get_context("fork")
    started = context.Event()
    acquired = context.Event()
    result = context.Queue()
    lock_path = tmp_path / ".memory-recovery.lock"
    gate = storage.open_workspace_memory_lock(tmp_path)
    gate.__enter__()
    process = context.Process(
        target=_contending_child,
        args=(str(tmp_path), "gate", started, acquired, result),
    )

    try:
        process.start()
        assert started.wait(timeout=4)
        assert not acquired.wait(timeout=0.15)
        original = tmp_path / "original-lock"
        lock_path.rename(original)
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"keep replacement")
        replacement.chmod(0o600)
        replacement.rename(lock_path)
        replacement_snapshot = _snapshot(lock_path)
        gate.close()

        assert result.get(timeout=4) == (
            "error",
            "MemoryStorageError",
            "binding",
            "memory recovery storage binding changed",
        )
        assert not acquired.is_set()
        assert _snapshot(lock_path) == replacement_snapshot
        assert lock_path.read_bytes() == b"keep replacement"
        assert not (tmp_path / "memory").exists()
        process.join(timeout=4)
        assert not process.is_alive()
    finally:
        gate.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)


def _forked_close_child(gate: Any, started: Any, acquired: Any, result: Any) -> None:
    inherited_fd = gate._lock_fd
    assert inherited_fd is not None
    os.close(inherited_fd)
    unrelated_fd = os.open(os.devnull, os.O_RDONLY)
    if unrelated_fd != inherited_fd:
        os.dup2(unrelated_fd, inherited_fd)
        os.close(unrelated_fd)
        unrelated_fd = inherited_fd

    try:
        try:
            gate.close()
        except storage.MemoryStorageError as error:
            close_result = (error.code, os.fstat(unrelated_fd).st_ino >= 0)
        else:
            close_result = ("closed", False)
        assert storage._fcntl is not None
        real_flock = storage._fcntl.flock

        def signaling_flock(fd: int, operation: int, *args: object) -> None:
            if operation == storage._fcntl.LOCK_EX:
                started.set()
            real_flock(fd, operation, *args)

        storage._fcntl.flock = signaling_flock
        with storage.open_workspace_memory_lock(gate.workspace):
            acquired.set()
        result.put((close_result, "ok"))
    except BaseException as error:
        result.put((("error", False), type(error).__name__, getattr(error, "code", None), str(error)))
    finally:
        try:
            os.close(unrelated_fd)
        except OSError:
            pass


def test_fork_child_cannot_unlock_or_close_inherited_gate_fd(tmp_path: Path) -> None:
    if not _fork_capable():
        pytest.skip("fork lifecycle check requires multiprocessing fork")
    context = multiprocessing.get_context("fork")
    started = context.Event()
    acquired = context.Event()
    result = context.Queue()
    gate = storage.open_workspace_memory_lock(tmp_path)
    gate.__enter__()
    process = context.Process(target=_forked_close_child, args=(gate, started, acquired, result))
    try:
        process.start()
        assert started.wait(timeout=4)
        assert not acquired.wait(timeout=0.15)
        gate.close()
        assert acquired.wait(timeout=4)
        process.join(timeout=4)
        assert not process.is_alive()
        assert result.get(timeout=2) == (("closed", True), "ok")
    finally:
        gate.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
