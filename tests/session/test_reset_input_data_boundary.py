"""Reset input is data, with original late checks and no execution authority."""

import inspect
import json
from dataclasses import FrozenInstanceError, fields, replace

import pytest

from nanobot.session import source as source_module
from nanobot.session.execution_owner import ExecutionOwnershipError, acquire_execution
from nanobot.session.manager import JsonlSessionStore, SessionStorageError
from nanobot.session.runtime_authority import DurableSessionAuthority
from nanobot.session.source import new_source_event
from nanobot.session.source_store import SourceSessionStore
from tests.session.test_execution_owner import seed


@pytest.fixture
def backend(tmp_path):
    return JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")


def read_operation():
    operation = getattr(SourceSessionStore, "read_reset_input", None)
    assert callable(operation), "owner-local reset input data operation is missing"
    return operation


def namespace_bytes(backend):
    return {path.relative_to(backend.sessions_dir): path.read_bytes()
            for path in backend.sessions_dir.rglob("*") if path.is_file()}


def test_handoff_retains_data_only_and_does_not_eagerly_query_or_snapshot(backend, monkeypatch):
    operation = read_operation()
    initial = seed(backend)
    before = namespace_bytes(backend)
    read = SourceSessionStore._read_unlocked
    snapshot = SourceSessionStore._snapshot
    trace = []
    captured = {}

    class Metadata(dict):
        def __contains__(self, key):
            if key == "runtime_checkpoint":
                trace.append("runtime-presence")
            return super().__contains__(key)

    def reading(source, *args, **kwargs):
        header, events = read(source, *args, **kwargs)
        header["metadata"] = Metadata(header["metadata"])
        captured.update(header=header, events=events)
        trace.append("read")
        return header, events

    def building(source, header, events):
        trace.append("snapshot")
        assert header is captured["header"] and events is captured["events"]
        return snapshot(source, header, events)

    monkeypatch.setattr(SourceSessionStore, "_read_unlocked", reading)
    monkeypatch.setattr(SourceSessionStore, "_snapshot", building)
    data = operation(backend.source, initial.state.session_key)
    assert trace == ["read"]
    assert data.events is captured["events"]
    assert [field.name for field in fields(data)] == ["events", "has_runtime_checkpoint", "snapshot"]
    assert not any(hasattr(data, name) for name in ("header", "backend", "source", "owner", "fd"))
    for callback, names in ((data.has_runtime_checkpoint, {"header"}),
                            (data.snapshot, {"header", "events"})):
        assert set(callback.__code__.co_freevars) == names
        for name, cell in zip(callback.__code__.co_freevars, callback.__closure__):
            assert cell.cell_contents is captured[name]
    assert data.has_runtime_checkpoint() is False
    rebuilt = data.snapshot(backend.source)
    assert trace == ["read", "runtime-presence", "snapshot"]
    assert rebuilt.state == initial.state and rebuilt.events == initial.events
    assert namespace_bytes(backend) == before


@pytest.mark.parametrize("kind", ["storage", "missing", "io"])
def test_read_fault_keeps_exact_first_error_without_snapshot(backend, monkeypatch, kind):
    operation = read_operation()
    initial = seed(backend)
    before = namespace_bytes(backend)
    fault = {"storage": SessionStorageError(), "missing": FileNotFoundError("missing"),
             "io": OSError("read failed")}[kind]
    trace = []

    def fail_read(_source, key):
        trace.append(key)
        raise fault

    def forbidden_snapshot(*_args):
        pytest.fail("a failed input read reached snapshot assembly")

    monkeypatch.setattr(SourceSessionStore, "_read_unlocked", fail_read)
    monkeypatch.setattr(SourceSessionStore, "_snapshot", forbidden_snapshot)
    with pytest.raises(type(fault)) as caught:
        operation(backend.source, initial.state.session_key)
    assert caught.value is fault and trace == [initial.state.session_key]
    assert namespace_bytes(backend) == before


@pytest.mark.parametrize("fail", [False, True])
def test_snapshot_hook_and_source_selection_are_late(backend, monkeypatch, fail):
    operation = read_operation()
    initial = seed(backend)
    before = namespace_bytes(backend)
    reader = backend.source
    data = operation(reader, initial.state.session_key)
    selected = backend.source
    assert selected is not reader
    snapshot = SourceSessionStore._snapshot
    fault = OSError("selected snapshot failed")
    observed = []

    def selected_snapshot(source, header, events):
        observed.append(source)
        if fail:
            raise fault
        return snapshot(source, header, events)

    monkeypatch.setattr(SourceSessionStore, "_snapshot", selected_snapshot)
    if fail:
        with pytest.raises(OSError) as caught:
            data.snapshot(selected)
        assert caught.value is fault
    else:
        assert data.snapshot(selected).state == initial.state
    assert observed == [selected]
    assert namespace_bytes(backend) == before


def test_runtime_predicate_fault_is_lazy_and_retains_identity(backend, monkeypatch):
    operation = read_operation()
    initial = seed(backend)
    before = namespace_bytes(backend)
    read = SourceSessionStore._read_unlocked
    fault = RuntimeError("runtime metadata read failed")
    touched = []

    class Metadata(dict):
        def __contains__(self, key):
            if key == "runtime_checkpoint":
                touched.append(key)
                raise fault
            return super().__contains__(key)

    def reading(source, *args, **kwargs):
        header, events = read(source, *args, **kwargs)
        header["metadata"] = Metadata(header["metadata"])
        return header, events

    monkeypatch.setattr(SourceSessionStore, "_read_unlocked", reading)
    data = operation(backend.source, initial.state.session_key)
    assert touched == []
    with pytest.raises(RuntimeError) as caught:
        data.has_runtime_checkpoint()
    assert caught.value is fault and touched == ["runtime_checkpoint"]
    assert namespace_bytes(backend) == before


def test_handoff_is_detached_editable_data_not_an_execution_capability(backend):
    operation = read_operation()
    initial = seed(backend)
    before = namespace_bytes(backend)
    data = operation(backend.source, initial.state.session_key)
    with pytest.raises(FrozenInstanceError):
        data.events = []
    data.events[0]["content"] = "local-only edit"
    rebuilt = data.snapshot(backend.source)
    assert rebuilt.events[0]["content"] == "local-only edit"
    rebuilt.events[0]["content"] = "snapshot-only edit"
    assert data.events[0]["content"] == "local-only edit"
    with pytest.raises(ExecutionOwnershipError):
        backend.source.reset_source_session(initial.state.session_key, initial.state, [], owner=data)
    assert backend.source.load_source_snapshot(initial.state.session_key).events == initial.events
    assert namespace_bytes(backend) == before


@pytest.mark.parametrize("owned", [False, True])
async def test_mode_reset_keeps_import_guard_and_second_source_lookup_order(
    tmp_path, monkeypatch, owned,
):
    operation = read_operation()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "private").acquire()
    try:
        manager = lease.manager
        key = "cli:reset-order"
        manager.admit_persistent_write(key)
        backend = manager._jsonl_store
        resolver = manager._mode_resolver
        trace = []
        awaiting_snapshot = False
        source_property = JsonlSessionStore.source
        inventory = resolver._inventory_for_key
        links = source_module.validate_execution_links

        def validator(events):
            trace.append("guard")
            return links(events)

        def forbidden_validator(_events):
            pytest.fail("mode validator was reselected after inventory")

        def inventory_read(actual_key):
            frame = inspect.currentframe()
            assert frame is not None and frame.f_back is not None
            explicit_reset_read = frame.f_back.f_code.co_name == "_reset_unlocked"
            result = inventory(actual_key)
            if explicit_reset_read:
                trace.append("inventory")
                patch.setattr(source_module, "validate_execution_links", forbidden_validator)
            return result

        def read_input(source, actual_key):
            nonlocal awaiting_snapshot
            trace.append("read-input")
            data = operation(source, actual_key)
            has_runtime, build = data.has_runtime_checkpoint, data.snapshot
            awaiting_snapshot = True

            def runtime_presence():
                trace.append("runtime")
                return has_runtime()

            def snapshot(selected_source):
                nonlocal awaiting_snapshot
                trace.append("snapshot")
                awaiting_snapshot = False
                return build(selected_source)

            return replace(data, has_runtime_checkpoint=runtime_presence, snapshot=snapshot)

        def source_getter(store):
            if awaiting_snapshot:
                trace.append("snapshot-source")
            source = source_property.fget(store)

            def forbidden_override(_key):
                pytest.fail("an instance override redirected the input reader")

            source.read_reset_input = forbidden_override
            return source

        with monkeypatch.context() as patch:
            patch.setattr(source_module, "validate_execution_links", validator)
            patch.setattr(resolver, "_inventory_for_key", inventory_read)
            patch.setattr(SourceSessionStore, "read_reset_input", read_input)
            patch.setattr(JsonlSessionStore, "source", property(source_getter))
            if owned:
                async with acquire_execution(backend, key, run_id="reset") as owner:
                    await owner.quiesce()
                    resolved = resolver.reset(key, [], owner=owner)
                    assert owner._active is False and owner._lock.is_locked
            else:
                resolved = resolver.reset(key, [], owner=None)
        assert trace == ["inventory", "read-input", *([] if owned else ["guard", "runtime"]),
                         "snapshot-source", "snapshot"]
        assert resolved.state == "active" and resolved.mode == "canonical"
        assert backend.source.load_source_snapshot(key).events == []
    finally:
        await lease.close()


@pytest.mark.parametrize("evidence", ["unresolved", "runtime_data_fault", "both_data_fault"])
async def test_mode_refusal_precedes_snapshot_and_short_circuits_runtime_read(
    tmp_path, monkeypatch, evidence,
):
    operation = read_operation()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "private").acquire()
    try:
        manager = lease.manager
        key = "cli:reset-refusal"
        manager.admit_persistent_write(key)
        backend, resolver = manager._jsonl_store, manager._mode_resolver
        if evidence != "runtime_data_fault":
            initial = backend.source.load_source_snapshot(key)
            event = new_source_event({"role": "assistant", "content": "",
                                      "tool_calls": [{"id": "call"}]}, run_id="unresolved")
            event["_execution_ids"] = ["pending-execution"]
            backend.source.append_source_events(key, initial.state, [event])
        before = namespace_bytes(backend)
        read = SourceSessionStore._read_unlocked
        in_reset_read = False
        runtime_queries = []

        class Metadata(dict):
            def __contains__(self, name):
                if name == "runtime_checkpoint":
                    runtime_queries.append(name)
                return super().__contains__(name)

        def reading(source, *args, **kwargs):
            header, events = read(source, *args, **kwargs)
            if in_reset_read:
                header["metadata"] = Metadata(header["metadata"])
                if evidence != "unresolved":
                    # Parsed-data fault only. Strict on-disk grammar forbids this key.
                    header["metadata"]["runtime_checkpoint"] = {"pending": "data-fault"}
            return header, events

        def read_input(source, actual_key):
            nonlocal in_reset_read
            in_reset_read = True
            try:
                return operation(source, actual_key)
            finally:
                in_reset_read = False

        def forbidden_snapshot(*_args):
            pytest.fail("ownerless refusal reached snapshot assembly")

        with monkeypatch.context() as patch:
            patch.setattr(SourceSessionStore, "_read_unlocked", reading)
            patch.setattr(SourceSessionStore, "read_reset_input", read_input)
            patch.setattr(SourceSessionStore, "_snapshot", forbidden_snapshot)
            with pytest.raises(ExecutionOwnershipError):
                resolver.reset(key, [], owner=None)
        assert runtime_queries == (["runtime_checkpoint"] if evidence == "runtime_data_fault" else [])
        assert namespace_bytes(backend) == before
    finally:
        await lease.close()


@pytest.mark.parametrize("seam", ["read", "snapshot"])
async def test_mode_private_read_faults_keep_identity_and_no_journal_write(tmp_path, monkeypatch, seam):
    operation = read_operation()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "private").acquire()
    try:
        manager = lease.manager
        key = "cli:reset-fault"
        manager.admit_persistent_write(key)
        backend, resolver = manager._jsonl_store, manager._mode_resolver
        before = namespace_bytes(backend)
        fault = OSError("original reset input fault")
        read = SourceSessionStore._read_unlocked
        in_reset_read = False
        touched = []

        def reading(source, *args, **kwargs):
            if in_reset_read and seam == "read":
                touched.append("read")
                raise fault
            return read(source, *args, **kwargs)

        def read_input(source, actual_key):
            nonlocal in_reset_read
            in_reset_read = True
            try:
                return operation(source, actual_key)
            finally:
                in_reset_read = False

        def snapshot_fault(*_args):
            touched.append("snapshot")
            raise fault

        with monkeypatch.context() as patch:
            patch.setattr(SourceSessionStore, "read_reset_input", read_input)
            patch.setattr(SourceSessionStore, "_read_unlocked", reading)
            if seam == "snapshot":
                patch.setattr(SourceSessionStore, "_snapshot", snapshot_fault)
            with pytest.raises(OSError) as caught:
                resolver.reset(key, [], owner=None)
        assert caught.value is fault and touched == [seam]
        assert namespace_bytes(backend) == before
    finally:
        await lease.close()


async def test_real_durable_runtime_checkpoint_is_rejected_before_input_handoff(tmp_path, monkeypatch):
    read_operation()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "private").acquire()
    try:
        manager = lease.manager
        key = "cli:invalid-runtime"
        manager.admit_persistent_write(key)
        backend, resolver = manager._jsonl_store, manager._mode_resolver
        name = f"{backend.storage_key(key)}.jsonl"
        rows = [json.loads(row) for row in resolver.storage.read(name).splitlines()]
        rows[0]["metadata"]["runtime_checkpoint"] = {"pending": "invalid-source-state"}
        rows[0]["revision"] += 1
        resolver.storage.write(name, b"".join((json.dumps(row) + "\n").encode() for row in rows))
        before = namespace_bytes(backend)

        def forbidden_read_input(*_args):
            pytest.fail("invalid durable source passed strict transaction admission")

        monkeypatch.setattr(SourceSessionStore, "read_reset_input", forbidden_read_input)
        with pytest.raises(SessionStorageError):
            resolver.reset(key, [], owner=None)
        assert namespace_bytes(backend) == before
    finally:
        await lease.close()
