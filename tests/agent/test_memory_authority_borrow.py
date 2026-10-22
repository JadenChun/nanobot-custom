"""Focused tests for the B2a1 workspace-memory authority lease."""

from __future__ import annotations

import copy
import multiprocessing
import os
import pickle
import select
import shutil
import signal
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
    return hasattr(storage, "_fcntl") and storage._fcntl is not None


def _fork_capable() -> bool:
    if not _storage_capable():
        return False
    try:
        multiprocessing.get_context("fork")
    except (AttributeError, ValueError):
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _storage_capable(), reason="B1a storage requires POSIX descriptor capabilities"
)


def _error_result(action: Any) -> tuple[str, str, str]:
    try:
        action()
    except BaseException as exc:  # child/thread reports only fixed public details
        return type(exc).__name__, getattr(exc, "code", ""), str(exc)
    return "ok", "", ""


def _active_operation_child(operation: storage.WorkspaceMemoryOperation, result: Any) -> None:
    result.put(
        (
            _error_result(operation.require_current),
            _error_result(lambda: operation.__exit__(None, None, None)),
        )
    )


def _new_operation_child(operation: storage.WorkspaceMemoryOperation, result: Any) -> None:
    result.put(_error_result(operation.__enter__))


def _fork_action_with_locked_lifecycle(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    action: str,
) -> bytes:
    read_fd, write_fd = os.pipe()
    authority._lifecycle_lock.acquire()
    child_pid: int | None = None
    try:
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            try:
                os.write(write_fd, b"S")
                try:
                    if action == "authority-operation":
                        authority.operation()
                    elif action == "authority-close":
                        authority.close()
                    elif action == "authority-exit":
                        authority.__exit__(None, None, None)
                    elif action == "authority-enter":
                        authority.__enter__()
                    elif action == "operation-enter":
                        operation.__enter__()
                    elif action == "operation-identity":
                        _ = operation.workspace_identity
                    elif action == "operation-require":
                        operation.require_current()
                    elif action == "operation-exit":
                        operation.__exit__(None, None, None)
                    else:
                        raise AssertionError(action)
                except MemoryError:
                    raise
                except BaseException as exc:
                    code = getattr(exc, "code", "")
                    payload = b"E" + str(code).encode("ascii", errors="replace")
                else:
                    payload = b"O"
                os.write(write_fd, b"R" + payload)
            finally:
                os.close(write_fd)
                os._exit(0)

        os.close(write_fd)
        ready, _, _ = select.select([read_fd], [], [], 5)
        if not ready or os.read(read_fd, 1) != b"S":
            if child_pid is not None:
                os.kill(child_pid, signal.SIGKILL)
            return b"Xstart"
        ready, _, _ = select.select([read_fd], [], [], 1)
        if ready:
            return os.read(read_fd, 64)
        os.kill(child_pid, signal.SIGKILL)
        return b"Xtimeout"
    finally:
        authority._lifecycle_lock.release()
        os.close(read_fd)
        if child_pid is not None:
            os.waitpid(child_pid, 0)


def test_inactive_operation_is_opaque_and_does_not_touch_filesystem(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    before = sorted(tmp_path.iterdir())
    operation = authority.operation()
    assert sorted(tmp_path.iterdir()) == before
    assert not hasattr(operation, "close")
    assert not hasattr(operation, "release")
    assert not hasattr(operation, "borrow_count")
    with pytest.raises(storage.MemoryStorageError) as raised:
        operation.__enter__()
    assert raised.value.code == "closed"
    assert sorted(tmp_path.iterdir()) == before


def test_operation_constructor_is_factory_only_and_non_serializable(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    with pytest.raises(storage.MemoryStorageError) as raised:
        storage.WorkspaceMemoryOperation(authority)
    assert raised.value.code == "invalid"

    operation = authority.operation()
    for copier in (
        copy.copy,
        copy.deepcopy,
    ):
        with pytest.raises(TypeError):
            copier(operation)
    with pytest.raises(TypeError):
        pickle.dumps(operation)


def test_operation_scope_exposes_only_workspace_identity(tmp_path: Path) -> None:
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation = authority.operation()
        with pytest.raises(storage.MemoryStorageError) as raised:
            _ = operation.workspace_identity
        assert raised.value.code == "closed"
        with operation as current:
            expected = os.stat(tmp_path, follow_symlinks=False)
            assert current is operation
            assert operation.workspace_identity == (expected.st_dev, expected.st_ino)
            operation.require_current()
            assert isinstance(operation.workspace_identity, tuple)
            assert len(operation.workspace_identity) == 2
        authority.close()


def test_nested_leases_are_supported_and_release_in_lifo_order(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    first = authority.operation()
    second = authority.operation()
    with first:
        with second:
            with pytest.raises(storage.MemoryStorageError) as raised:
                authority.close()
            assert raised.value.code == "borrowed"
        with pytest.raises(storage.MemoryStorageError) as raised:
            authority.close()
        assert raised.value.code == "borrowed"
    authority.close()


def test_nested_and_multiple_operations_reject_close_until_all_exit(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    first = authority.operation()
    second = authority.operation()
    first.__enter__()
    second.__enter__()
    with pytest.raises(storage.MemoryStorageError) as raised:
        authority.close()
    assert raised.value.code == "borrowed"
    with pytest.raises(storage.MemoryStorageError) as raised:
        first.__enter__()
    assert raised.value.code == "closed"
    first.__exit__(None, None, None)
    with pytest.raises(storage.MemoryStorageError) as raised:
        authority.close()
    assert raised.value.code == "borrowed"
    second.__exit__(None, None, None)
    authority.close()


def test_body_exception_releases_operation_exactly_once(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    operation = authority.operation()
    with pytest.raises(RuntimeError, match="body failure"):
        with operation:
            raise RuntimeError("body failure")
    operation.__exit__(None, None, None)
    authority.close()


def test_repeated_exit_is_harmless_and_repeated_enter_is_rejected(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    operation = authority.operation()
    operation.__enter__()
    with pytest.raises(storage.MemoryStorageError) as raised:
        operation.__enter__()
    assert raised.value.code == "closed"
    with pytest.raises(storage.MemoryStorageError) as raised:
        authority.close()
    assert raised.value.code == "borrowed"
    operation.__exit__(None, None, None)
    operation.__exit__(None, None, None)
    authority.close()


def test_explicit_close_rejection_preserves_authority_and_namespace(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    namespace = authority.open_evidence_namespace("transactions", "a" * 64)
    operation = authority.operation()
    operation.__enter__()
    with pytest.raises(storage.MemoryStorageError) as raised:
        authority.close()
    assert raised.value.code == "borrowed"
    assert authority._active is True
    assert authority._lock_held is True
    assert authority._workspace_fd is not None
    assert authority._lock_fd is not None
    assert namespace.publish_record("genesis.json", b"still usable") == "genesis.json"
    assert namespace.read_record("genesis.json") == b"still usable"
    operation.__exit__(None, None, None)
    authority.close()


def test_context_exit_rejection_has_no_partial_cleanup_and_recovers(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    operation = authority.operation()
    operation.__enter__()
    with pytest.raises(storage.MemoryStorageError) as raised:
        authority.__exit__(None, None, None)
    assert raised.value.code == "borrowed"
    assert authority._active is True
    assert authority._lock_held is True
    operation.__exit__(None, None, None)
    authority.__exit__(None, None, None)
    with storage.WorkspaceMemoryAuthority(tmp_path):
        pass


def test_failed_authority_entry_does_not_register_epoch_or_borrow(tmp_path: Path) -> None:
    workspace = tmp_path / "created-after-entry-failure"
    authority = storage.WorkspaceMemoryAuthority(workspace)
    operation = authority.operation()
    initial_epoch = authority._activation_epoch
    with pytest.raises(storage.MemoryStorageError):
        authority.__enter__()
    assert authority._activation_epoch == initial_epoch
    workspace.mkdir(mode=0o700)
    with authority:
        with pytest.raises(storage.MemoryStorageError) as raised:
            operation.__enter__()
        assert raised.value.code == "closed"
        fresh_operation = authority.operation()
        with fresh_operation:
            fresh_operation.require_current()
        authority.close()


def test_failed_operation_entry_does_not_borrow_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    operation = authority.operation()
    original = authority._assert_binding
    failed = True

    def fail_once(*, layout: bool = True) -> None:
        nonlocal failed
        if failed:
            failed = False
            raise storage.MemoryStorageError("binding")
        original(layout=layout)

    monkeypatch.setattr(authority, "_assert_binding", fail_once)
    with pytest.raises(storage.MemoryStorageError) as raised:
        operation.__enter__()
    assert raised.value.code == "binding"
    authority.close()


def test_wrong_thread_cannot_enter_use_or_exit(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    operation = authority.operation()
    results: list[tuple[str, str, str]] = []

    def foreign_thread() -> None:
        results.extend(
            (
                _error_result(operation.__enter__),
                _error_result(operation.require_current),
                _error_result(lambda: operation.__exit__(None, None, None)),
            )
        )

    thread = threading.Thread(target=foreign_thread)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert results and all(item[0] == "MemoryStorageError" for item in results)
    assert all(item[1] == "closed" for item in results)
    with operation:
        operation.require_current()
    authority.close()


def test_wrong_thread_cannot_exit_active_operation(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    operation = authority.operation()
    operation.__enter__()
    result: list[tuple[str, str, str]] = []

    def foreign_thread() -> None:
        result.append(_error_result(operation.require_current))
        result.append(_error_result(lambda: operation.__exit__(None, None, None)))

    thread = threading.Thread(target=foreign_thread)
    thread.start()
    thread.join(timeout=5)
    assert result and all(item[1] == "closed" for item in result)
    with pytest.raises(storage.MemoryStorageError) as raised:
        authority.close()
    assert raised.value.code == "borrowed"
    operation.__exit__(None, None, None)
    authority.close()


def test_distinct_thread_object_rejected_when_integer_ids_collide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    operation = authority.operation()
    operation.__enter__()
    owner_ident = authority._activation_thread_id
    assert owner_ident is not None
    results: list[tuple[str, str, str]] = []
    finished = threading.Event()
    real_get_ident = threading.get_ident
    owner_thread = threading.current_thread()
    foreign_holder: dict[str, threading.Thread] = {}
    owner_real_ident = real_get_ident()

    monkeypatch.setattr(storage.threading, "get_ident", lambda: owner_ident)

    def exact_current_thread() -> threading.Thread:
        if real_get_ident() == owner_real_ident:
            return owner_thread
        return foreign_holder["thread"]

    monkeypatch.setattr(storage.threading, "current_thread", exact_current_thread)

    def foreign_thread() -> None:
        try:
            results.extend(
                (
                    _error_result(operation.require_current),
                    _error_result(lambda: operation.__exit__(None, None, None)),
                    _error_result(authority.close),
                )
            )
        finally:
            finished.set()

    thread = threading.Thread(target=foreign_thread)
    foreign_holder["thread"] = thread
    thread.start()
    assert finished.wait(timeout=5)
    monkeypatch.undo()
    threading._active[owner_real_ident] = owner_thread
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert results[0][1] == "closed"
    assert results[1][1] == "closed"
    assert results[2][1] == "borrowed"
    operation.require_current()
    operation.__exit__(None, None, None)
    authority.close()


def test_entry_and_close_interleaving_never_leaves_closed_borrowed_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    ready = threading.Event()
    register_paused = threading.Event()
    allow_register = threading.Event()
    operation_active = threading.Event()
    allow_owner_exit = threading.Event()
    cleanup_started = threading.Event()
    allow_cleanup = threading.Event()
    close_started = threading.Event()
    close_done = threading.Event()
    owner_result: dict[str, Any] = {}
    close_result: dict[str, Any] = {}
    operation_holder: dict[str, storage.WorkspaceMemoryOperation] = {}
    original_register = authority._register_operation
    original_cleanup = authority._cleanup

    def paused_register(operation: storage.WorkspaceMemoryOperation) -> None:
        register_paused.set()
        assert allow_register.wait(timeout=5)
        original_register(operation)

    def paused_cleanup() -> None:
        cleanup_started.set()
        assert allow_cleanup.wait(timeout=5)
        original_cleanup()

    monkeypatch.setattr(authority, "_register_operation", paused_register)
    monkeypatch.setattr(authority, "_cleanup", paused_cleanup)

    def owner() -> None:
        try:
            authority.__enter__()
            operation = authority.operation()
            operation_holder["operation"] = operation
            ready.set()
            operation.__enter__()
            operation_active.set()
            assert allow_owner_exit.wait(timeout=5)
            operation.__exit__(None, None, None)
            owner_result["status"] = "ok"
        except BaseException as exc:
            owner_result["error"] = exc

    def close() -> None:
        close_started.set()
        try:
            authority.close()
            close_result["status"] = "ok"
        except BaseException as exc:
            close_result["error"] = exc
        finally:
            close_done.set()

    owner_thread = threading.Thread(target=owner)
    owner_thread.start()
    assert ready.wait(timeout=5)
    assert register_paused.wait(timeout=5)
    close_thread = threading.Thread(target=close)
    close_thread.start()
    assert close_started.wait(timeout=5)
    cleanup_started.wait(timeout=1)
    allow_register.set()
    assert operation_active.wait(timeout=5)

    # A lock-protected close waits here; the legacy implementation can already
    # have completed cleanup while registration is paused.
    close_done.wait(timeout=1)
    if cleanup_started.is_set():
        assert authority._active is True or not authority._borrowers
    else:
        assert authority._active is True
        assert operation_holder["operation"] in authority._borrowers
    allow_cleanup.set()
    allow_owner_exit.set()
    owner_thread.join(timeout=5)
    close_thread.join(timeout=5)
    assert not owner_thread.is_alive()
    assert not close_thread.is_alive()
    assert "error" not in owner_result
    operation = operation_holder["operation"]
    if "error" in close_result:
        assert isinstance(close_result["error"], storage.MemoryStorageError)
        assert close_result["error"].code == "borrowed"
    assert not (not authority._active and operation in authority._borrowers)
    if authority._active:
        authority.close()


def test_close_and_entry_interleaving_rejects_entry_without_borrower(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    ready = threading.Event()
    start_entry = threading.Event()
    entry_attempted = threading.Event()
    entry_done = threading.Event()
    allow_cleanup = threading.Event()
    cleanup_started = threading.Event()
    close_done = threading.Event()
    owner_result: dict[str, Any] = {}
    entry_result: dict[str, Any] = {}
    operation_holder: dict[str, storage.WorkspaceMemoryOperation] = {}
    original_cleanup = authority._cleanup

    def paused_cleanup() -> None:
        cleanup_started.set()
        assert allow_cleanup.wait(timeout=5)
        original_cleanup()

    monkeypatch.setattr(authority, "_cleanup", paused_cleanup)

    def owner() -> None:
        try:
            authority.__enter__()
            operation = authority.operation()
            operation_holder["operation"] = operation
            ready.set()
            assert start_entry.wait(timeout=5)
            entry_attempted.set()
            try:
                operation.__enter__()
            except BaseException as exc:
                entry_result["error"] = exc
            else:
                entry_result["status"] = "ok"
            entry_done.set()
            if operation._state == operation._ACTIVE:
                operation.__exit__(None, None, None)
            owner_result["status"] = "ok"
        except BaseException as exc:
            owner_result["error"] = exc

    def close() -> None:
        try:
            authority.close()
            owner_result["close"] = "ok"
        except BaseException as exc:
            owner_result["close_error"] = exc
        finally:
            close_done.set()

    owner_thread = threading.Thread(target=owner)
    owner_thread.start()
    assert ready.wait(timeout=5)
    close_thread = threading.Thread(target=close)
    close_thread.start()
    assert cleanup_started.wait(timeout=5)
    start_entry.set()
    assert entry_attempted.wait(timeout=5)
    entry_done.wait(timeout=1)
    allow_cleanup.set()
    assert close_done.wait(timeout=5)
    owner_thread.join(timeout=5)
    close_thread.join(timeout=5)
    assert not owner_thread.is_alive()
    assert not close_thread.is_alive()
    assert "error" not in owner_result
    operation = operation_holder["operation"]
    if entry_result.get("status") == "ok":
        # The lock must have serialized close before registration.
        pytest.fail("entry registered while authority cleanup was paused")
    assert isinstance(entry_result.get("error"), storage.MemoryStorageError)
    assert entry_result["error"].code == "closed"
    assert not authority._borrowers
    assert not authority._active
    assert operation._state == operation._NEW


@pytest.mark.skipif(not _fork_capable(), reason="lifecycle mutex fork test requires fork")
@pytest.mark.parametrize(
    "action",
    [
        "authority-operation",
        "authority-close",
        "authority-exit",
        "authority-enter",
        "operation-enter",
        "operation-identity",
        "operation-require",
        "operation-exit",
    ],
)
def test_forked_child_cannot_use_inherited_locked_lifecycle_mutex(
    tmp_path: Path, action: str
) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    active_operation = authority.operation()
    active_operation.__enter__()
    pending_operation = authority.operation()
    target = pending_operation if action == "operation-enter" else active_operation
    result = _fork_action_with_locked_lifecycle(authority, target, action)
    assert result == b"REclosed"
    active_operation.require_current()
    with pytest.raises(storage.MemoryStorageError) as raised:
        authority.close()
    assert raised.value.code == "borrowed"
    active_operation.__exit__(None, None, None)
    authority.close()


@pytest.mark.skipif(not _fork_capable(), reason="lifecycle mutex fork test requires fork")
@pytest.mark.parametrize("action", ["authority-operation", "authority-close"])
def test_unborrowed_active_authority_rejects_forked_lifecycle_calls(
    tmp_path: Path, action: str
) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    result = _fork_action_with_locked_lifecycle(authority, authority.operation(), action)
    assert result == b"REclosed"
    assert authority._active is True
    assert authority._lock_held is True
    authority.close()


@pytest.mark.skipif(not _fork_capable(), reason="file-lock fork test requires fork")
def test_parent_file_lock_still_excludes_competitor_after_fork_rejection(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    operation = authority.operation()
    operation.__enter__()
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(read_fd)
        try:
            os.write(write_fd, b"S")
            competitor = storage.WorkspaceMemoryAuthority(tmp_path)
            competitor.__enter__()
            os.write(write_fd, b"R")
            competitor.close()
        except BaseException:
            os.write(write_fd, b"E")
        finally:
            os.close(write_fd)
            os._exit(0)

    os.close(write_fd)
    try:
        ready, _, _ = select.select([read_fd], [], [], 5)
        assert ready and os.read(read_fd, 1) == b"S"
        ready, _, _ = select.select([read_fd], [], [], 0.2)
        assert not ready
        operation.__exit__(None, None, None)
        authority.close()
        ready, _, _ = select.select([read_fd], [], [], 5)
        assert ready and os.read(read_fd, 1) == b"R"
    finally:
        os.close(read_fd)
        _, status = os.waitpid(child_pid, 0)
        assert os.WIFEXITED(status)


def test_operation_issued_in_old_epoch_cannot_enter_after_reentry(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    operation = authority.operation()
    issued_epoch = authority._activation_epoch
    authority.close()
    authority.__enter__()
    assert authority._activation_epoch > issued_epoch
    with pytest.raises(storage.MemoryStorageError) as raised:
        operation.__enter__()
    assert raised.value.code == "closed"
    authority.close()


@pytest.mark.skipif(not _fork_capable(), reason="operation fork test requires fork")
def test_forked_process_cannot_use_or_exit_inherited_active_operation(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    result = context.Queue()
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    operation = authority.operation()
    operation.__enter__()
    process = context.Process(target=_active_operation_child, args=(operation, result))
    process.start()
    process.join(timeout=5)
    try:
        assert not process.is_alive()
        inherited_results = result.get(timeout=2)
        assert all(item[0] == "MemoryStorageError" for item in inherited_results)
        assert all(item[1] == "closed" for item in inherited_results)
        operation.require_current()
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        operation.__exit__(None, None, None)
        authority.close()


@pytest.mark.skipif(not _fork_capable(), reason="operation fork test requires fork")
def test_forked_process_cannot_enter_inherited_new_operation(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    result = context.Queue()
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    operation = authority.operation()
    process = context.Process(target=_new_operation_child, args=(operation, result))
    process.start()
    process.join(timeout=5)
    try:
        assert not process.is_alive()
        inherited_result = result.get(timeout=2)
        assert inherited_result[0] == "MemoryStorageError"
        assert inherited_result[1] == "closed"
        with operation:
            operation.require_current()
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        authority.close()


def test_stale_operation_cannot_cross_authority_reentry_epoch(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    operation = authority.operation()
    operation.__enter__()
    operation.__exit__(None, None, None)
    authority.close()
    authority.__enter__()
    with pytest.raises(storage.MemoryStorageError) as raised:
        operation.require_current()
    assert raised.value.code == "closed"
    with pytest.raises(storage.MemoryStorageError) as raised:
        operation.__enter__()
    assert raised.value.code == "closed"
    authority.close()


def test_binding_loss_blocks_require_current_but_owner_exit_still_releases(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    authority = storage.WorkspaceMemoryAuthority(workspace)
    authority.__enter__()
    operation = authority.operation()
    operation.__enter__()
    moved = tmp_path / "workspace-moved"
    workspace.rename(moved)
    workspace.mkdir(mode=0o700)
    with pytest.raises(storage.MemoryStorageError) as raised:
        operation.require_current()
    assert raised.value.code == "binding"
    operation.__exit__(None, None, None)
    authority.close()
    shutil.rmtree(moved)


def test_authority_namespace_remains_usable_after_rejected_close(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    operation = authority.operation()
    operation.__enter__()
    namespace = authority.open_evidence_namespace("conversions", "b" * 64)
    with pytest.raises(storage.MemoryStorageError) as raised:
        authority.close()
    assert raised.value.code == "borrowed"
    assert namespace.publish_record("genesis.json", b"available") == "genesis.json"
    operation.__exit__(None, None, None)
    authority.close()


def test_borrowed_error_is_fixed_and_content_free(tmp_path: Path) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    operation = authority.operation()
    operation.__enter__()
    with pytest.raises(storage.MemoryStorageError) as raised:
        authority.close()
    assert raised.value.code == "borrowed"
    assert str(raised.value) == "memory recovery authority is borrowed"
    operation.__exit__(None, None, None)
    authority.close()


def test_operation_failed_registration_restores_new_unregistered_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation = authority.operation()
        original_register = authority._register_operation

        def failed_register(current: storage.WorkspaceMemoryOperation) -> None:
            original_register(current)
            raise RuntimeError("registration failure")

        with monkeypatch.context() as failure:
            failure.setattr(authority, "_register_operation", failed_register)
            with pytest.raises(RuntimeError, match="registration failure"):
                operation.__enter__()
        assert operation not in authority._borrowers
        assert operation._state == operation._NEW
        assert operation._registered is False
        assert operation._epoch is None
        assert operation._workspace_identity is None
        with operation:
            operation.require_current()


def test_operation_constructor_cannot_select_another_authority_binder(tmp_path: Path) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    with storage.WorkspaceMemoryAuthority(first_path) as first:
        with storage.WorkspaceMemoryAuthority(second_path) as second:
            with pytest.raises(TypeError):
                storage.WorkspaceMemoryOperation(
                    first, _capability=storage._OPERATION_CAPABILITY,
                    _issued_epoch=first._activation_epoch, _thread=threading.current_thread(),
                    _bind_actions=second._bind_operation_actions,
                )
            assert not first._borrowers
            assert not second._borrowers
            with first.operation() as operation:
                assert operation._authority is first
                assert operation in first._borrowers
                assert operation not in second._borrowers
                operation.require_current()
