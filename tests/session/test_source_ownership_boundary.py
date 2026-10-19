"""Source-domain operations retain admitted parent locks and late original hooks."""

from datetime import timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.session import execution_owner
from nanobot.session.execution_owner import (
    ExecutionBusyError,
    ExecutionOwnershipError,
    acquire_execution,
    execution_lock_guard,
)
from nanobot.session.manager import JsonlSessionStore, Session, SessionStorageError
from nanobot.session.source_store import SourceSessionStore, SourceSnapshot
from tests.session.test_execution_owner import seed

TIMESTAMP = "2026-10-13T00:00:00+00:00"


@pytest.fixture
def seeded(tmp_path):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    return backend, seed(backend)


def command(name):
    value = getattr(SourceSessionStore, name, None)
    assert callable(value), f"source-owned {name} operation is missing"
    return value


def require_execution_lock(backend, key):
    with pytest.raises(ExecutionBusyError), execution_lock_guard(backend, key):
        pytest.fail("source operation released the admitted execution lock")


@pytest.mark.parametrize("name", ["publish_execution_lease", "delete_unowned_source"])
def test_concrete_source_domain_operations_exist(name):
    command(name)


async def test_acquisition_uses_concrete_command_and_original_late_clock_write(seeded, monkeypatch):
    publish = command("publish_execution_lease")
    backend, initial = seeded
    key = initial.state.session_key
    trace = []
    original_write = SourceSessionStore._write

    class SelectedSource(SourceSessionStore):
        def publish_execution_lease(self, *args, **kwargs):
            pytest.fail("instance override must not redirect lease publication")

    def source(value):
        return SelectedSource(value)

    def write(value, header, events, **kwargs):
        require_execution_lock(backend, key)
        trace.append("write")
        lease = header["metadata"].get("execution_lease")
        if lease is not None:
            assert lease["acquired_at"] == TIMESTAMP
            assert lease["run_id"] == "run"
            assert header["revision"] == initial.state.revision + 1
            assert events == initial.events
        return original_write(value, header, events, **kwargs)

    def now(tz):
        assert tz is timezone.utc
        require_execution_lock(backend, key)
        trace.append("clock")
        # Select the old write hook only after the acquisition's original clock.
        patch.setattr(SourceSessionStore, "_write", write)
        return SimpleNamespace(isoformat=lambda: TIMESTAMP)

    def publish_domain(*args, **kwargs):
        trace.append("domain")
        return publish(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(JsonlSessionStore, "source", property(source))
        patch.setattr(SourceSessionStore, "publish_execution_lease", staticmethod(publish_domain))
        patch.setattr(execution_owner, "datetime", SimpleNamespace(now=now))
        async with acquire_execution(backend, key, run_id="run") as owner:
            assert trace == ["domain", "clock", "write"]
            owner.validate_dispatch()
            assert owner.read_snapshot().header["updated_at"] == initial.header["updated_at"]
    assert trace == ["domain", "clock", "write", "write"]
    assert "execution_lease" not in backend.source.load_source_snapshot(key).header["metadata"]


@pytest.mark.parametrize("phase", ["clock", "source", "lookup", "before", "after"])
async def test_lease_first_fault_effects_and_recovery_are_unchanged(seeded, monkeypatch, phase):
    publish = command("publish_execution_lease")
    backend, snapshot = seeded
    key = snapshot.state.session_key
    metadata = snapshot.header["metadata"]
    before = backend.get_session_path(key).read_bytes()
    failure = RuntimeError(f"controlled {phase} lease failure")
    trace = []
    original_write = SourceSessionStore._write

    class SelectedSource(SourceSessionStore):
        def __getattribute__(self, name):
            if name == "_write":
                trace.append("lookup")
                require_execution_lock(backend, key)
                if phase == "lookup":
                    raise failure
            return super().__getattribute__(name)

    def source(value):
        trace.append("source")
        require_execution_lock(backend, key)
        if phase == "source":
            raise failure
        return SelectedSource(value)

    def clock():
        trace.append("clock")
        require_execution_lock(backend, key)
        if phase == "clock":
            raise failure
        return TIMESTAMP

    def write(value, header, events, **kwargs):
        trace.append("write")
        require_execution_lock(backend, key)
        if phase == "before":
            raise failure
        original_write(value, header, events, **kwargs)
        raise failure

    with execution_lock_guard(backend, key), backend.locked_session_files():
        with monkeypatch.context() as patch:
            patch.setattr(JsonlSessionStore, "source", property(source))
            patch.setattr(SourceSessionStore, "_write", write)
            with pytest.raises(RuntimeError) as caught:
                publish(backend, snapshot, metadata, token="1" * 32, run_id="failed",
                        recovery_required=False, select_acquired_at=clock)
            assert caught.value is failure
            require_execution_lock(backend, key)
    expected = ["clock", "source", "lookup", "write"]
    stop = {"clock": 1, "source": 2, "lookup": 3, "before": 4, "after": 4}[phase]
    assert trace == expected[:stop]
    assert snapshot.header["revision"] == snapshot.state.revision + (phase != "clock")
    assert ("execution_lease" in metadata) is (phase != "clock")
    assert (backend.get_session_path(key).read_bytes() != before) is (phase == "after")
    async with acquire_execution(backend, key, run_id="next") as owner:
        assert owner.recovery_required is (phase == "after")
        if phase == "after":
            with pytest.raises(ExecutionOwnershipError):
                owner.validate_dispatch()
        else:
            owner.validate_dispatch()


def test_lease_preserves_snapshot_operand_order_and_captured_metadata(seeded):
    publish = command("publish_execution_lease")
    backend, initial = seeded
    key = initial.state.session_key
    trace = []

    class ObservedSnapshot(SourceSnapshot):
        def __getattribute__(self, name):
            if name in {"state", "header", "events"}:
                trace.append(name)
            return super().__getattribute__(name)

    snapshot = ObservedSnapshot(initial.state, initial.header, initial.events, initial.checkpoint)
    metadata = initial.header["metadata"]
    replacement_metadata = dict(metadata, title="clock replaced metadata")

    def clock():
        trace.append("clock")
        # The original metadata reference was selected before clock evaluation.
        initial.header["metadata"] = replacement_metadata
        return TIMESTAMP

    with execution_lock_guard(backend, key), backend.locked_session_files():
        committed = publish(backend, snapshot, metadata, token="2" * 32, run_id="run",
                            recovery_required=False, select_acquired_at=clock)
    assert trace == ["state", "state", "clock", "header", "header", "events"]
    assert metadata["execution_lease"]["acquired_at"] == TIMESTAMP
    assert committed.header["metadata"] == replacement_metadata
    assert "execution_lease" not in backend.source.load_source_snapshot(key).header["metadata"]


def test_lease_clock_corruption_does_not_repair_required_source_metadata(seeded):
    publish = command("publish_execution_lease")
    backend, snapshot = seeded
    key = snapshot.state.session_key
    metadata = snapshot.header["metadata"]

    def clock():
        snapshot.header["metadata"] = {"title": "clock removed source schema"}
        return TIMESTAMP

    with execution_lock_guard(backend, key), backend.locked_session_files():
        committed = publish(backend, snapshot, metadata, token="3" * 32, run_id="run",
                            recovery_required=False, select_acquired_at=clock)
    assert metadata["execution_lease"]["acquired_at"] == TIMESTAMP
    assert committed.header["metadata"] == {"title": "clock removed source schema"}
    with pytest.raises(SessionStorageError):
        backend.source.load_source_snapshot(key)


async def test_acquisition_clock_failure_never_issues_owner_and_releases_lock(seeded, monkeypatch):
    command("publish_execution_lease")
    backend, initial = seeded
    key = initial.state.session_key
    before = backend.get_session_path(key).read_bytes()
    failure = RuntimeError("controlled acquisition clock failure")

    def now(tz):
        assert tz is timezone.utc
        require_execution_lock(backend, key)
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(execution_owner, "datetime", SimpleNamespace(now=now))
        with pytest.raises(RuntimeError) as caught:
            async with acquire_execution(backend, key, run_id="failed"):
                pytest.fail("a failed clock issued an execution owner")
        assert caught.value is failure
    assert backend.get_session_path(key).read_bytes() == before
    async with acquire_execution(backend, key, run_id="next") as owner:
        assert not owner.recovery_required
        owner.validate_dispatch()


def test_backend_delete_uses_concrete_domain_with_two_original_source_selections(seeded, monkeypatch):
    delete = command("delete_unowned_source")
    backend, initial = seeded
    key = initial.state.session_key
    trace = []
    original_delete = SourceSessionStore._delete_unlocked
    original_fsync = JsonlSessionStore._fsync_directory

    class SelectedSource(SourceSessionStore):
        def delete_unowned_source(self, *args, **kwargs):
            pytest.fail("instance override must not redirect source deletion")

    def source(value):
        trace.append("source")
        return SelectedSource(value)

    def remove(value, actual_key, expected, **kwargs):
        require_execution_lock(backend, key)
        assert actual_key == key and expected == initial.state
        trace.append("delete")
        return original_delete(value, actual_key, expected, **kwargs)

    def fsync(path):
        require_execution_lock(backend, key)
        trace.append("fsync")
        return original_fsync(path)

    def delete_domain(*args, **kwargs):
        trace.append("domain")
        return delete(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(JsonlSessionStore, "source", property(source))
        patch.setattr(SourceSessionStore, "delete_unowned_source", staticmethod(delete_domain))
        patch.setattr(SourceSessionStore, "_delete_unlocked", remove)
        patch.setattr(JsonlSessionStore, "_fsync_directory", staticmethod(fsync))
        assert backend.delete(key) is True
    assert trace == ["domain", "source", "source", "delete", "fsync"]
    assert not backend.get_session_path(key).exists()
    assert (backend.sessions_dir / ".execution-locks" / f"{backend.storage_key(key)}.lock").exists()


@pytest.mark.parametrize("marker", ["execution_lease", "execution_recovery_required"])
def test_unowned_delete_refuses_persisted_markers_before_delete_selection(seeded, monkeypatch, marker):
    command("delete_unowned_source")
    backend, initial = seeded
    key = initial.state.session_key
    initial.header["metadata"][marker] = True
    with backend.locked_session_files():
        backend.source._write(initial.header, initial.events)
    before = backend.get_session_path(key).read_bytes()
    selections = []

    class SelectedSource(SourceSessionStore):
        def __getattribute__(self, name):
            if name == "_delete_unlocked":
                selections.append(name)
                pytest.fail("persisted execution marker selected a delete hook")
            return super().__getattribute__(name)

    with monkeypatch.context() as patch:
        patch.setattr(JsonlSessionStore, "source", property(SelectedSource))
        with pytest.raises(ExecutionOwnershipError):
            backend.delete(key)
    assert selections == []
    assert backend.get_session_path(key).read_bytes() == before
    with execution_lock_guard(backend, key):
        pass


@pytest.mark.parametrize("phase", ["load", "lookup", "current", "unlink", "fsync"])
def test_delete_first_fault_locks_and_irreversible_effect_are_unchanged(seeded, monkeypatch, phase):
    command("delete_unowned_source")
    backend, initial = seeded
    key = initial.state.session_key
    path = backend.get_session_path(key)
    before = path.read_bytes()
    failure = RuntimeError(f"controlled {phase} delete failure")
    trace = []
    original_load = SourceSessionStore.load_source_snapshot
    original_current = SourceSessionStore._current_unlocked
    original_unlink = Path.unlink
    original_fsync = JsonlSessionStore._fsync_directory

    def observe(name):
        trace.append(name)
        require_execution_lock(backend, key)
        if phase == name:
            raise failure

    class SelectedSource(SourceSessionStore):
        def __getattribute__(self, name):
            if name == "_delete_unlocked":
                observe("lookup")
            return super().__getattribute__(name)

    def load(value, actual_key):
        observe("load")
        return original_load(value, actual_key)

    def current(value, actual_key, expected):
        observe("current")
        return original_current(value, actual_key, expected)

    def unlink(value, *args, **kwargs):
        if value == path:
            observe("unlink")
        return original_unlink(value, *args, **kwargs)

    def fsync(value):
        observe("fsync")
        return original_fsync(value)

    with monkeypatch.context() as patch:
        patch.setattr(JsonlSessionStore, "source", property(SelectedSource))
        patch.setattr(SourceSessionStore, "load_source_snapshot", load)
        patch.setattr(SourceSessionStore, "_current_unlocked", current)
        patch.setattr(Path, "unlink", unlink)
        patch.setattr(JsonlSessionStore, "_fsync_directory", staticmethod(fsync))
        with pytest.raises(RuntimeError) as caught:
            backend.delete(key)
        assert caught.value is failure
    expected = ["load", "lookup", "current", "unlink", "fsync"]
    assert trace == expected[:expected.index(phase) + 1]
    assert path.exists() is (phase != "fsync")
    if path.exists():
        assert path.read_bytes() == before
    with execution_lock_guard(backend, key):
        pass


def test_legacy_delete_never_selects_source_domain(tmp_path, monkeypatch):
    command("delete_unowned_source")
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")
    session = Session("test:legacy", messages=[{"role": "user", "content": "input"}])
    backend.save(session)

    def forbidden(*args, **kwargs):
        pytest.fail("legacy deletion selected source-domain deletion")

    with monkeypatch.context() as patch:
        patch.setattr(SourceSessionStore, "delete_unowned_source", staticmethod(forbidden))
        assert backend.delete(session.key) is True
        assert backend.delete(session.key) is False
    assert not backend.get_session_path(session.key).exists()
