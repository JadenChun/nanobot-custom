"""Close settlement owns no generic writer and derives data from real owners."""

import asyncio
import inspect
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace

import pytest

from nanobot.session import execution_owner as owner_module
from nanobot.session.execution_owner import (
    ExecutionOwner,
    ExecutionOwnershipError,
    acquire_execution,
)
from nanobot.session.manager import JsonlSessionStore
from nanobot.session.source_store import SourceSessionStore, SourceSnapshot
from tests.session.test_execution_owner import seed


@pytest.fixture
def backend(tmp_path):
    return JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")


def settle_operation():
    operation = getattr(SourceSessionStore, "settle_execution_close", None)
    assert callable(operation), "source-owned guarded execution close operation is missing"
    return operation


def namespace_bytes(backend):
    return {path.relative_to(backend.sessions_dir): path.read_bytes()
            for path in backend.sessions_dir.rglob("*") if path.is_file()}


def test_command_accepts_only_backend_and_owner_not_unlocked_record_payloads():
    operation = settle_operation()
    assert list(inspect.signature(operation).parameters) == ["backend", "owner"]
    assert isinstance(SourceSessionStore.__dict__["settle_execution_close"], staticmethod)
    assert callable(getattr(ExecutionOwner, "closing_transaction", None))
    assert callable(getattr(ExecutionOwner, "matches_current_execution_lifetime", None))
    assert callable(getattr(ExecutionOwner, "copy_execution_lease", None))


def test_alleged_owner_refuses_before_any_attribute_lookup(backend):
    operation = settle_operation()
    touched = []

    class AllegedOwner:
        def __getattribute__(self, name):
            touched.append(name)
            raise AssertionError("invalid owner was inspected")

    before = namespace_bytes(backend)
    with pytest.raises(ExecutionOwnershipError):
        operation(backend, AllegedOwner())
    assert touched == [] and namespace_bytes(backend) == before


async def test_running_owner_cannot_settle_before_real_quiescence(backend):
    operation = settle_operation()
    original = seed(backend)
    async with acquire_execution(backend, original.state.session_key, run_id="live") as owner:
        before = namespace_bytes(backend)
        with pytest.raises(ExecutionOwnershipError):
            operation(backend, owner)
        assert namespace_bytes(backend) == before
        owner.validate_dispatch()


@pytest.mark.parametrize("registered", [False, True])
async def test_child_cannot_settle_even_during_real_root_quiescence(backend, registered):
    operation = settle_operation()
    original = seed(backend)
    admitted = asyncio.Event()
    observed = []
    async with acquire_execution(backend, original.state.session_key, run_id="tasks") as owner:
        before = namespace_bytes(backend)

        async def child():
            await admitted.wait()
            assert owner._closing
            with pytest.raises(ExecutionOwnershipError):
                operation(backend, owner)
            observed.append("refused")

        task = asyncio.create_task(child())
        if registered:
            owner.track_task(task)
        admitted.set()
        await owner.quiesce()
        await task
        assert observed == ["refused"] and namespace_bytes(backend) == before


async def test_other_backend_cannot_reuse_real_closing_owner(backend, tmp_path):
    operation = settle_operation()
    original = seed(backend)
    other = JsonlSessionStore(tmp_path / "other", sessions_root=tmp_path / "other-private")
    seed(other, original.state.session_key)
    async with acquire_execution(backend, original.state.session_key, run_id="store") as owner:
        await owner.quiesce()
        before, other_before = namespace_bytes(backend), namespace_bytes(other)
        with pytest.raises(ExecutionOwnershipError):
            operation(other, owner)
        assert namespace_bytes(backend) == before and namespace_bytes(other) == other_before


async def test_expired_real_owner_cannot_settle_again(backend):
    operation = settle_operation()
    original = seed(backend)
    async with acquire_execution(backend, original.state.session_key, run_id="closed") as owner:
        pass
    before = namespace_bytes(backend)
    with pytest.raises(ExecutionOwnershipError):
        operation(backend, owner)
    assert namespace_bytes(backend) == before


async def test_close_transaction_keeps_original_locks_read_validate_then_drain(
    backend, monkeypatch,
):
    settle_operation()
    original = seed(backend)
    async with acquire_execution(backend, original.state.session_key, run_id="order") as owner:
        await owner.quiesce()
        trace = []
        mutex, files = owner._mutex, backend.locked_session_files
        read, validate, drain = (
            SourceSessionStore.load_source_snapshot, owner._validate_unlocked, owner._require_drained,
        )

        @contextmanager
        def observed_mutex():
            with mutex:
                trace.append("mutex-enter")
                try:
                    yield
                finally:
                    trace.append("mutex-exit")

        @contextmanager
        def observed_files():
            with files():
                trace.append("files-enter")
                try:
                    yield
                finally:
                    trace.append("files-exit")

        def observed_read(source, *args, **kwargs):
            trace.append("read")
            return read(source, *args, **kwargs)

        def observed_validate(*args, **kwargs):
            trace.append("validate")
            return validate(*args, **kwargs)

        def observed_drain():
            trace.append("drain")
            return drain()

        before = namespace_bytes(backend)
        with monkeypatch.context() as patch:
            patch.setattr(owner, "_mutex", observed_mutex())
            patch.setattr(backend, "locked_session_files", observed_files)
            patch.setattr(SourceSessionStore, "load_source_snapshot", observed_read)
            patch.setattr(owner, "_validate_unlocked", observed_validate)
            patch.setattr(owner, "_require_drained", observed_drain)
            with ExecutionOwner.closing_transaction(owner, backend) as snapshot:
                assert type(snapshot) is SourceSnapshot
                assert snapshot.state.session_key == owner.key
                trace.append("body")
        assert trace == [
            "mutex-enter", "files-enter", "read", "files-enter", "files-exit",
            "validate", "drain", "body", "files-exit", "mutex-exit",
        ]
        assert namespace_bytes(backend) == before


async def test_normal_close_dispatches_source_command_and_owner_context_nonvirtually(
    backend, monkeypatch,
):
    operation = settle_operation()
    original = seed(backend)
    calls = []
    async with acquire_execution(backend, original.state.session_key, run_id="dispatch") as owner:
        def observing(selected_backend, selected_owner):
            assert selected_backend is backend and selected_owner is owner
            calls.append("settle")
            return operation(selected_backend, selected_owner)

        def forbidden_override(*_args):
            pytest.fail("settlement dispatched an owner instance context override")

        with monkeypatch.context() as patch:
            patch.setattr(SourceSessionStore, "settle_execution_close", staticmethod(observing))
            patch.setattr(owner, "closing_transaction", forbidden_override)
            await owner.aclose()
        assert calls == ["settle"]
    loaded = backend.source.load_source_snapshot(original.state.session_key)
    assert "execution_lease" not in loaded.header["metadata"]


async def test_success_preserves_late_source_selections_copy_and_validation_hooks(
    backend, monkeypatch,
):
    settle_operation()
    original = seed(backend)
    select_source = JsonlSessionStore.source.fget
    read, write = SourceSessionStore.load_source_snapshot, SourceSessionStore._write
    copy_lease, links = owner_module.deepcopy, owner_module.validate_execution_links
    selected, trace = [], []
    async with acquire_execution(backend, original.state.session_key, run_id="late") as owner:
        before = owner.read_snapshot()

        def selecting(value):
            assert value is backend
            source = select_source(value)
            selected.append(source)
            trace.append("select")
            return source

        def reading(source, *args, **kwargs):
            assert source is selected[-1]
            trace.append("read")
            return read(source, *args, **kwargs)

        def copying(value):
            trace.append("copy")
            return copy_lease(value)

        def validating(value):
            trace.append("links")
            return links(value)

        def writing(source, header, events, **kwargs):
            assert source is selected[-1]
            trace.append("write")
            assert events == before.events
            assert header["revision"] == before.state.revision + 1
            assert header["updated_at"] == before.header["updated_at"]
            assert header["metadata"] == {
                key: value for key, value in before.header["metadata"].items()
                if key != "execution_lease"
            }
            return write(source, header, events, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(JsonlSessionStore, "source", property(selecting))
            patch.setattr(SourceSessionStore, "load_source_snapshot", reading)
            patch.setattr(SourceSessionStore, "_write", writing)
            patch.setattr(owner_module, "deepcopy", copying)
            patch.setattr(owner_module, "validate_execution_links", validating)
            await owner.aclose()
        assert trace == ["select", "read", "copy", "links", "select", "write"]
        assert len(selected) == 2 and selected[0] is not selected[1]
    loaded = backend.source.load_source_snapshot(original.state.session_key)
    assert loaded.events == original.events and loaded.state.revision == before.state.revision + 1


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize("restore_fails", [False, True])
async def test_write_failure_keeps_first_error_exact_recovery_and_cleanup(
    backend, monkeypatch, phase, restore_fails,
):
    settle_operation()
    original = seed(backend)
    write = SourceSessionStore._write
    primary, secondary = OSError("primary close write"), OSError("secondary recovery write")
    writes, stores = [], []
    async with acquire_execution(backend, original.state.session_key, run_id="fault") as owner:
        before = owner.read_snapshot()
        saved_lease = deepcopy(before.header["metadata"]["execution_lease"])

        def failing(source, header, events, **kwargs):
            writes.append((deepcopy(header), deepcopy(events)))
            stores.append(source)
            if len(writes) == 1:
                if phase == "after":
                    write(source, header, events, **kwargs)
                raise primary
            if restore_fails:
                raise secondary
            return write(source, header, events, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(SourceSessionStore, "_write", failing)
            with pytest.raises(OSError) as caught:
                await owner.aclose()
        assert caught.value is primary
        assert getattr(primary, "__notes__", []) == (
            ["Persisting execution recovery metadata also failed."] if restore_fails else []
        )
        assert len(writes) == 2 and stores[0] is not stores[1]
        assert writes[0][1] == writes[1][1] == before.events
        recovered_header = writes[1][0]
        expected_lease = saved_lease | {"recovery_required": True}
        assert recovered_header["metadata"]["execution_lease"] == expected_lease
        assert recovered_header["metadata"]["execution_recovery_required"] is True
        assert recovered_header["revision"] == before.state.revision + (2 if phase == "after" else 1)
        assert not owner._active and not owner._lock.is_locked
        with pytest.raises(ExecutionOwnershipError):
            owner.validate_dispatch()
    loaded = backend.source.load_source_snapshot(original.state.session_key)
    if not restore_fails:
        assert loaded.header["metadata"]["execution_lease"] == expected_lease
        assert loaded.header["metadata"]["execution_recovery_required"] is True


@pytest.mark.parametrize("change", [None, "incarnation", "generation"])
async def test_lifetime_predicate_is_data_only_and_matches_original_identity_fields(backend, change):
    settle_operation()
    original = seed(backend)
    async with acquire_execution(backend, original.state.session_key, run_id="identity") as owner:
        snapshot = owner.read_snapshot()
        if change == "incarnation":
            snapshot = replace(snapshot, state=replace(snapshot.state, incarnation="different"))
        elif change == "generation":
            snapshot = replace(snapshot, state=replace(snapshot.state, generation=999))
        before = namespace_bytes(backend)
        assert ExecutionOwner.matches_current_execution_lifetime(owner, snapshot) is (change is None)
        assert namespace_bytes(backend) == before


@pytest.mark.parametrize("change", ["incarnation", "generation"])
async def test_failed_close_does_not_restore_lease_for_different_reloaded_lifetime(
    backend, monkeypatch, change,
):
    settle_operation()
    original = seed(backend)
    read = SourceSessionStore.load_source_snapshot
    primary = OSError("primary close write")
    reads, writes = [], []
    async with acquire_execution(backend, original.state.session_key, run_id="mismatch") as owner:
        def reading(source, *args, **kwargs):
            snapshot = read(source, *args, **kwargs)
            reads.append(snapshot)
            if len(reads) == 2:
                state = replace(snapshot.state, **{
                    change: "different" if change == "incarnation" else 999,
                })
                return replace(snapshot, state=state)
            return snapshot

        def refusing(source, *_args, **_kwargs):
            writes.append(source)
            raise primary

        with monkeypatch.context() as patch:
            patch.setattr(SourceSessionStore, "load_source_snapshot", reading)
            patch.setattr(SourceSessionStore, "_write", refusing)
            with pytest.raises(OSError) as caught:
                await owner.aclose()
        assert caught.value is primary and len(reads) == 2 and len(writes) == 1
        assert getattr(primary, "__notes__", []) == []
        assert not owner._active and not owner._lock.is_locked


@pytest.mark.parametrize("seam", ["deepcopy", "validate_execution_links"])
async def test_original_prewrite_helper_failure_keeps_identity_and_no_recovery_write(
    backend, monkeypatch, seam,
):
    settle_operation()
    original = seed(backend)
    primary = OSError("original helper fault")
    trace = []
    async with acquire_execution(backend, original.state.session_key, run_id="helper") as owner:
        before = namespace_bytes(backend)

        def refusing(*_args):
            trace.append(seam)
            raise primary

        def forbidden_write(*_args, **_kwargs):
            pytest.fail("prewrite helper failure reached source persistence")

        with monkeypatch.context() as patch:
            patch.setattr(owner_module, seam, refusing)
            patch.setattr(SourceSessionStore, "_write", forbidden_write)
            with pytest.raises(OSError) as caught:
                await owner.aclose()
        assert caught.value is primary and trace == [seam]
        assert namespace_bytes(backend) == before
        assert not owner._active and not owner._lock.is_locked
