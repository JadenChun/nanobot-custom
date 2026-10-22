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
from types import SimpleNamespace
from typing import Any

import pytest

from nanobot.agent import memory_recovery_storage as storage


@pytest.mark.parametrize("fault", [RuntimeError("binder"), KeyboardInterrupt("binder")])
def test_r2_operation_constructor_lookup_precedes_armed_hooks_fault(fault):
    trace = []

    class Authority(storage.WorkspaceMemoryAuthority):
        def __getattribute__(self, name):
            if name == "_bind_operation_actions":
                trace.append("binder")
                raise fault
            return super().__getattribute__(name)

    class Operation(storage.WorkspaceMemoryOperation):
        def __getattribute__(self, name):
            if name == "_hooks" and "_hooks" in vars(self):
                trace.append("hooks")
                raise AssertionError("later hooks fault")
            return super().__getattribute__(name)

    operation = object.__new__(Operation)
    with pytest.raises(type(fault)) as raised:
        Operation.__init__(operation, object.__new__(Authority),
                           _capability=storage._OPERATION_CAPABILITY,
                           _issued_epoch=1, _thread=threading.current_thread())
    assert raised.value is fault
    assert trace == ["binder"]
    assert "_hooks" in vars(operation)
    assert "_actions" not in vars(operation)


def test_r2_operation_constructor_binder_changes_later_hooks():
    changed, actions = object(), object()
    trace = []
    operation = object.__new__(storage.WorkspaceMemoryOperation)

    class Authority(storage.WorkspaceMemoryAuthority):
        @property
        def _bind_operation_actions(self):
            trace.append("lookup")
            operation._hooks = changed

            def bind(hooks):
                trace.append("call")
                assert hooks is changed
                return actions

            return bind

    storage.WorkspaceMemoryOperation.__init__(
        operation, object.__new__(Authority), _capability=storage._OPERATION_CAPABILITY,
        _issued_epoch=1, _thread=threading.current_thread(),
    )
    assert operation._actions is actions
    assert trace == ["lookup", "call"]


@pytest.mark.parametrize("fault", [RuntimeError("binder"), KeyboardInterrupt("binder")])
def test_r2_operation_bound_binder_lookup_precedes_missing_hooks(fault):
    class Authority(storage.WorkspaceMemoryAuthority):
        @property
        def _bind_bound_actions(self):
            raise fault

    operation = object.__new__(storage.WorkspaceMemoryOperation)
    operation._authority = object.__new__(Authority)
    with pytest.raises(type(fault)) as raised:
        operation._bind_bound_actions(object(), object(), object())
    assert raised.value is fault
    assert "_hooks" not in vars(operation)


def test_r2_operation_bound_binder_side_effects_are_live_without_receipt_truthiness():
    operation = object.__new__(storage.WorkspaceMemoryOperation)
    hooks, bound_hooks, actions, name = object(), object(), object(), object()
    trace = []

    class Receipt:
        def __bool__(self):
            raise AssertionError("receipt must not be tested")

    receipt = Receipt()

    class Authority(storage.WorkspaceMemoryAuthority):
        @property
        def _bind_bound_actions(self):
            trace.append("lookup")
            operation._hooks = hooks
            operation._epoch = 9

            def bind(*args):
                trace.append("call")
                assert args == (hooks, bound_hooks, 9, receipt, name)
                return actions

            return bind

    operation._authority = object.__new__(Authority)
    assert operation._bind_bound_actions(bound_hooks, receipt, name) is actions
    assert trace == ["lookup", "call"]


@pytest.mark.parametrize("capability_name", ["_BOUND_TARGET_CAPABILITY", "_BOUND_RECEIPT_TARGET_CAPABILITY"])
@pytest.mark.parametrize("fault", [RuntimeError("partial"), KeyboardInterrupt("partial")])
@pytest.mark.parametrize("point", ["owner", "reread", "validator", "pid", "thread", "epoch",
                                   "name", "content", "stat_identity", "binder"])
def test_r2_bound_initialization_exact_failure_partial_state(
    capability_name, fault, point, monkeypatch,
):
    trace = []
    validator = object()
    target = None

    class Authority(storage.WorkspaceMemoryAuthority):
        @property
        def _validate_snapshot(self):
            trace.append("validator")
            if point == "validator":
                raise fault
            return validator

    authority = object.__new__(Authority)

    class Operation(storage.WorkspaceMemoryOperation):
        def __getattribute__(self, name):
            label = {"_authority": "owner", "_thread": "thread", "_epoch": "epoch",
                     "_bind_bound_actions": "binder"}.get(name)
            if label is not None:
                trace.append(label)
                if point == label:
                    raise fault
            return super().__getattribute__(name)

    class Bound(storage.BoundMemoryTarget):
        def __getattribute__(self, name):
            if name == "_authority":
                trace.append("reread")
                if point == "reread":
                    raise fault
            return super().__getattribute__(name)

    class Snapshot:
        def __getattribute__(self, name):
            if name in ("name", "content", "stat_identity"):
                trace.append(name)
                if point == name:
                    raise fault
            return super().__getattribute__(name)

        name = "MEMORY.md"
        content = None
        stat_identity = None

    def pid():
        trace.append("pid")
        if point == "pid":
            raise fault
        return 123

    operation = object.__new__(Operation)
    operation._authority = authority
    operation._thread = threading.current_thread()
    operation._epoch = 7
    target = object.__new__(Bound)
    monkeypatch.setattr(storage.os, "getpid", pid)
    with pytest.raises(type(fault)) as raised:
        Bound.__init__(target, operation, Snapshot(),
                       _capability=getattr(storage, capability_name))
    assert raised.value is fault
    order = ["owner", "reread", "validator", "pid", "thread", "epoch", "name",
             "content", "stat_identity", "binder"]
    assert trace == order[:order.index(point) + 1]
    assignments = ["_receipt_scope", "_operation", "_authority", "_validate_snapshot",
                   "_snapshot", "_pid", "_thread", "_epoch", "_snapshot_name",
                   "_snapshot_content", "_snapshot_stat_identity", "_state", "_registered",
                   "_memory_fd", "_memory_identity"]
    counts = {"owner": 2, "reread": 3, "validator": 3, "pid": 5, "thread": 6,
              "epoch": 7, "name": 8, "content": 9, "stat_identity": 10, "binder": 15}
    assert set(vars(target)) == set(assignments[:counts[point]])


@pytest.mark.parametrize("capability_name", ["_BOUND_TARGET_CAPABILITY", "_BOUND_RECEIPT_TARGET_CAPABILITY"])
def test_r2_bound_owner_setter_reread_and_cached_validator(capability_name, monkeypatch):
    first = object.__new__(storage.WorkspaceMemoryAuthority)
    second = object.__new__(storage.WorkspaceMemoryAuthority)
    before, after, actions = object(), object(), object()
    first._validate_snapshot = object()
    second._validate_snapshot = before
    operation = object.__new__(storage.WorkspaceMemoryOperation)
    operation._authority = first
    operation._thread = threading.current_thread()
    operation._epoch = 7
    operation._bind_bound_actions = lambda *args: actions

    class Bound(storage.BoundMemoryTarget):
        def __setattr__(self, name, value):
            super().__setattr__(name, second if name == "_authority" else value)

    target = Bound(operation, SimpleNamespace(name="MEMORY.md", content=None, stat_identity=None),
                   _capability=getattr(storage, capability_name))
    assert target._authority is second
    assert target._validate_snapshot is before
    monkeypatch.setattr(second, "_validate_snapshot", after)
    assert target._validate_snapshot is before
    assert target._actions is actions


@pytest.mark.parametrize("fault", [RuntimeError("binder"), KeyboardInterrupt("binder")])
def test_r2_bound_binder_lookup_precedes_receipt_and_name_fault(fault):
    trace = []

    class Operation(storage.WorkspaceMemoryOperation):
        @property
        def _bind_bound_actions(self):
            trace.append("binder")
            raise fault

    class Bound(storage.BoundMemoryTarget):
        def __getattribute__(self, name):
            if name == "_receipt_scope" and "_snapshot_name" in vars(self):
                trace.append("receipt")
                raise AssertionError("later receipt fault")
            return super().__getattribute__(name)

    operation = object.__new__(Operation)
    operation._authority = object.__new__(storage.WorkspaceMemoryAuthority)
    operation._thread = threading.current_thread()
    operation._epoch = 7
    target = object.__new__(Bound)
    with pytest.raises(type(fault)) as raised:
        Bound.__init__(target, operation, SimpleNamespace(name="MEMORY.md", content=None,
                                                        stat_identity=None),
                       _capability=storage._BOUND_TARGET_CAPABILITY)
    assert raised.value is fault
    assert trace == ["binder"]
    assert "_memory_identity" in vars(target) and "_actions" not in vars(target)


def test_r2_bound_binder_changes_later_receipt_and_name_reads():
    target = object.__new__(storage.BoundMemoryTarget)
    actions, name = object(), object()
    trace = []

    class Operation(storage.WorkspaceMemoryOperation):
        @property
        def _bind_bound_actions(self):
            trace.append("binder")
            target._receipt_scope = "changed"
            target._snapshot_name = name

            def bind(hooks, receipt, actual_name):
                trace.append("call")
                assert hooks.bound is target
                assert receipt == "changed" and actual_name is name
                return actions

            return bind

    operation = object.__new__(Operation)
    operation._authority = object.__new__(storage.WorkspaceMemoryAuthority)
    operation._thread = threading.current_thread()
    operation._epoch = 7
    storage.BoundMemoryTarget.__init__(
        target, operation, SimpleNamespace(name="MEMORY.md", content=None, stat_identity=None),
        _capability=storage._BOUND_TARGET_CAPABILITY,
    )
    assert target._actions is actions
    assert trace == ["binder", "call"]


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


def test_operation_workspace_is_lease_scoped_and_does_not_expose_authority(tmp_path: Path) -> None:
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation = authority.operation()
        with pytest.raises(storage.MemoryStorageError) as inactive:
            _ = operation.workspace
        assert inactive.value.code == "closed"
        with operation:
            assert operation.workspace == authority.workspace == tmp_path.resolve()
            assert isinstance(operation.workspace, Path)
            assert not hasattr(operation, "authority")
            failures = []

            def read_from_other_thread() -> None:
                try:
                    _ = operation.workspace
                except storage.MemoryStorageError as error:
                    failures.append(error.code)

            thread = threading.Thread(target=read_from_other_thread)
            thread.start()
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert failures == ["closed"]
        with pytest.raises(storage.MemoryStorageError) as closed:
            _ = operation.workspace
        assert closed.value.code == "closed"


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


def test_parent_failed_registration_closes_duplicate_and_restores_new_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedParents(set[storage.BoundMemoryTarget]):
        def add(self, bound: storage.BoundMemoryTarget) -> None:
            super().add(bound)
            raise RuntimeError("parent registration failure")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        with authority.operation() as operation:
            bound = operation.borrow_target_parent(snapshot)
            parents = operation._target_parent_borrowers
            operation._target_parent_borrowers = FailedParents()
            duplicates: list[int] = []
            original_dup = storage.os.dup

            def track_dup(fd: int) -> int:
                duplicate = original_dup(fd)
                duplicates.append(duplicate)
                return duplicate

            with monkeypatch.context() as failure:
                failure.setattr(storage.os, "dup", track_dup)
                with pytest.raises(RuntimeError, match="parent registration failure"):
                    bound.__enter__()
            assert len(duplicates) == 1
            with pytest.raises(OSError):
                os.fstat(duplicates[0])
            assert not operation._target_parent_borrowers
            assert bound._state == bound._NEW
            assert bound._registered is False
            assert bound._memory_fd is None
            assert bound._memory_identity is None
            operation._target_parent_borrowers = parents
            with bound:
                bound.require_current()


def test_parent_current_preserves_snapshot_before_missing_descriptor_error_order(
    tmp_path: Path,
) -> None:
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        with authority.operation() as operation:
            with operation.borrow_target_parent(snapshot) as bound:
                name = snapshot.name
                fd = bound._memory_fd
                try:
                    object.__setattr__(snapshot, "_name", "not-allowlisted")
                    bound._memory_fd = None
                    with pytest.raises(storage.MemoryStorageError) as rejected:
                        bound.require_current()
                    assert rejected.value.code == "invalid"
                finally:
                    object.__setattr__(snapshot, "_name", name)
                    bound._memory_fd = fd
                bound.require_current()


def test_bound_constructor_cannot_select_another_operation_owner_or_actions(tmp_path: Path) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    with storage.WorkspaceMemoryAuthority(first_path) as first:
        with storage.WorkspaceMemoryAuthority(second_path) as second:
            with first.operation() as first_operation, second.operation() as second_operation:
                snapshot = first.snapshot_target("MEMORY.md")
                other_snapshot = second.snapshot_target("MEMORY.md")
                other = second_operation.borrow_target_parent(other_snapshot)
                with pytest.raises(TypeError):
                    storage.BoundMemoryTarget(
                        first_operation, snapshot, _capability=storage._BOUND_TARGET_CAPABILITY,
                        _authority_owner=second, _thread=threading.current_thread(),
                        _epoch=second._activation_epoch,
                        _validate_snapshot=second._validate_snapshot,
                        _bind_actions=lambda hooks: other._actions,
                    )
                assert not first_operation._target_parent_borrowers
                assert not second_operation._target_parent_borrowers
                with first_operation.borrow_target_parent(snapshot) as bound:
                    assert bound._authority is first
                    assert bound._operation is first_operation
                    assert bound in first_operation._target_parent_borrowers
                    assert bound not in second_operation._target_parent_borrowers
                    bound.require_current()
