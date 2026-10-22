"""Supported storage interfaces preserve the lock-only and private-directory policies."""

from __future__ import annotations

import errno
import inspect
import os
import stat
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_storage as storage


def test_lock_gate_alias_retains_the_original_distinct_type() -> None:
    gate_type = getattr(storage, "WorkspaceMemoryLockGate")
    assert gate_type is storage._WorkspaceMemoryLockGate
    assert gate_type is not storage.WorkspaceMemoryAuthority
    assert gate_type.__name__ == "_WorkspaceMemoryLockGate"
    assert gate_type.__qualname__ == "_WorkspaceMemoryLockGate"
    for name in (
        "WorkspaceMemoryLockGate", "require_memory_lock_gate_capabilities",
        "open_recovery_private_directory",
    ):
        assert name in storage.__all__


@pytest.mark.parametrize("existing_only", [False, True])
@pytest.mark.parametrize("ordinary_memory", [False, True])
def test_real_factories_keep_lock_only_behavior(
    tmp_path: Path, existing_only: bool, ordinary_memory: bool,
) -> None:
    gate_type = getattr(storage, "WorkspaceMemoryLockGate")
    memory = tmp_path / "memory"
    if ordinary_memory:
        memory.mkdir(mode=0o755)
        memory.chmod(0o755)
        (memory / "sentinel").write_bytes(b"ordinary memory")
    if existing_only:
        with storage.open_workspace_memory_lock(tmp_path):
            pass
    factory = (storage.open_existing_workspace_memory_lock if existing_only
               else storage.open_workspace_memory_lock)
    with factory(tmp_path) as gate:
        assert type(gate) is gate_type
        assert not isinstance(gate, storage.WorkspaceMemoryAuthority)
        assert gate.workspace == tmp_path.resolve()
        assert gate.require_current() is None
        assert stat.S_IMODE(gate.lock_path.stat().st_mode) == 0o600
        assert not (memory / ".recovery").exists()
    with pytest.raises(storage.MemoryStorageError) as raised:
        gate.require_current()
    assert raised.value.code == "closed"
    if ordinary_memory:
        assert stat.S_IMODE(memory.stat().st_mode) == 0o755
        assert (memory / "sentinel").read_bytes() == b"ordinary memory"
        assert sorted(p.name for p in memory.iterdir()) == ["sentinel"]
    else:
        assert not memory.exists()


def test_existing_lock_factory_does_not_create_missing_layout(tmp_path: Path) -> None:
    gate_type = getattr(storage, "WorkspaceMemoryLockGate")
    gate = storage.open_existing_workspace_memory_lock(tmp_path)
    assert type(gate) is gate_type
    with pytest.raises(storage.MemoryStorageError) as raised:
        with gate:
            pytest.fail("missing lock admitted")
    assert raised.value.code == "missing"
    assert list(tmp_path.iterdir()) == []


def test_capability_probe_uses_the_current_private_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = getattr(storage, "require_memory_lock_gate_capabilities")
    calls: list[str] = []

    def original() -> None:
        calls.append("original")

    def replacement() -> None:
        calls.append("replacement")

    monkeypatch.setattr(storage, "_require_lock_gate_capabilities", original)
    assert probe() is None
    monkeypatch.setattr(storage, "_require_lock_gate_capabilities", replacement)
    assert probe() is None
    assert calls == ["original", "replacement"]


@pytest.mark.parametrize("fault", [
    OSError("probe"), ValueError("probe"), storage.MemoryStorageUnavailableError(),
    KeyboardInterrupt("probe"),
])
def test_capability_probe_preserves_fault_identity(
    monkeypatch: pytest.MonkeyPatch, fault: BaseException,
) -> None:
    probe = getattr(storage, "require_memory_lock_gate_capabilities")

    def fail() -> None:
        raise fault

    monkeypatch.setattr(storage, "_require_lock_gate_capabilities", fail)
    with pytest.raises(type(fault)) as raised:
        probe()
    assert raised.value is fault


def test_capability_probe_still_requires_duplicate_descriptors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = getattr(storage, "require_memory_lock_gate_capabilities")
    assert probe() is None
    monkeypatch.setattr(storage.os, "dup", None)
    with pytest.raises(storage.MemoryStorageUnavailableError):
        probe()


@pytest.mark.parametrize("create", [False, True])
@pytest.mark.parametrize("sync", [None, False, True])
def test_directory_interface_forwards_exact_arguments(
    monkeypatch: pytest.MonkeyPatch, create: bool, sync: bool | None,
) -> None:
    opener = getattr(storage, "open_recovery_private_directory")
    parameters = inspect.signature(opener).parameters
    assert tuple(parameters) == ("parent_fd", "name", "create", "sync_existing")
    assert parameters["create"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["sync_existing"].default is True
    calls: list[tuple[int, str, dict[str, object]]] = []

    def replacement(parent_fd: int, name: str, **kwargs: object) -> int:
        calls.append((parent_fd, name, kwargs))
        return 173

    monkeypatch.setattr(storage, "_open_child_directory", replacement)
    arguments = {} if sync is None else {"sync_existing": sync}
    assert opener(37, "receipt-quarantine", create=create, **arguments) == 173
    assert calls == [(37, "receipt-quarantine", {
        "create": create, "sync_existing": True if sync is None else sync,
    })]


@pytest.mark.parametrize("kind", [OSError, TypeError, ValueError, RuntimeError, KeyboardInterrupt])
def test_directory_interface_preserves_fault_identity(
    monkeypatch: pytest.MonkeyPatch, kind: type[BaseException],
) -> None:
    opener = getattr(storage, "open_recovery_private_directory")
    fault = kind("directory")

    def fail(*args: object, **kwargs: object) -> int:
        raise fault

    monkeypatch.setattr(storage, "_open_child_directory", fail)
    with pytest.raises(kind) as raised:
        opener(37, "receipt-quarantine", create=False, sync_existing=False)
    assert raised.value is fault


@pytest.mark.parametrize("already_exists,create", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("sync", [False, True])
def test_directory_interface_real_private_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    already_exists: bool, create: bool, sync: bool,
) -> None:
    opener = getattr(storage, "open_recovery_private_directory")
    storage.require_memory_lock_gate_capabilities()
    child = tmp_path / "receipt-quarantine"
    if already_exists:
        child.mkdir(mode=0o700)
        child.chmod(0o700)
    real_sync = storage._sync_directory_boundary
    calls: list[tuple[int, str, int, dict[str, object]]] = []

    def record(parent_fd: int, name: str, child_fd: int, **kwargs: object) -> None:
        calls.append((parent_fd, name, child_fd, kwargs))
        real_sync(parent_fd, name, child_fd, **kwargs)

    monkeypatch.setattr(storage, "_sync_directory_boundary", record)
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = opener(parent_fd, child.name, create=create, sync_existing=sync)
        try:
            observed, opened = child.lstat(), os.fstat(fd)
            assert (opened.st_dev, opened.st_ino) == (observed.st_dev, observed.st_ino)
            assert stat.S_ISDIR(opened.st_mode)
            assert stat.S_IMODE(opened.st_mode) == 0o700
            assert calls == ([(parent_fd, child.name, fd, {
                "workspace_memory": False, "expected_mode": None,
            })] if sync else [])
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


@pytest.mark.parametrize("unsafe", ["missing", "symlink", "file", "mode"])
def test_directory_interface_refuses_unsafe_entries_and_closes_opened_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe: str,
) -> None:
    opener = getattr(storage, "open_recovery_private_directory")
    child = tmp_path / "receipt-quarantine"
    if unsafe == "symlink":
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        child.symlink_to(target, target_is_directory=True)
    elif unsafe == "file":
        child.write_bytes(b"not a directory")
    elif unsafe == "mode":
        child.mkdir(mode=0o755)
        child.chmod(0o755)
    before = sorted((p.name, p.lstat()) for p in tmp_path.iterdir())
    real_close = storage._close
    closed: list[int] = []

    def record(fd: int | None) -> None:
        if fd is not None:
            closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(storage, "_close", record)
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(storage.MemoryStorageError) as raised:
            opener(parent_fd, child.name, create=False)
        assert raised.value.code == ("missing" if unsafe == "missing" else "corrupt")
        assert len(closed) == (1 if unsafe == "mode" else 0)
        for fd in closed:
            with pytest.raises(OSError) as bad_fd:
                os.fstat(fd)
            assert bad_fd.value.errno == errno.EBADF
        assert sorted((p.name, p.lstat()) for p in tmp_path.iterdir()) == before
    finally:
        os.close(parent_fd)


@pytest.mark.parametrize("kind", [OSError, TypeError, RuntimeError, KeyboardInterrupt])
def test_directory_interface_closes_on_real_post_open_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: type[BaseException],
) -> None:
    opener = getattr(storage, "open_recovery_private_directory")
    fault = kind("sync")
    opened: list[int] = []

    def fail(parent_fd: int, name: str, fd: int, **kwargs: object) -> None:
        assert stat.S_ISDIR(os.fstat(fd).st_mode)
        opened.append(fd)
        raise fault

    monkeypatch.setattr(storage, "_sync_directory_boundary", fail)
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        with pytest.raises(kind) as raised:
            opener(parent_fd, "receipt-quarantine", create=True)
        assert raised.value is fault
        assert len(opened) == 1
        with pytest.raises(OSError) as bad_fd:
            os.fstat(opened[0])
        assert bad_fd.value.errno == errno.EBADF
        assert stat.S_IMODE((tmp_path / "receipt-quarantine").stat().st_mode) == 0o700
    finally:
        os.close(parent_fd)
