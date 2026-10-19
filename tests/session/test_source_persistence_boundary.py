"""Source persistence validates records before late backend-owned write selection."""

import json
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from nanobot.session import mode_storage, source_store
from nanobot.session.manager import JsonlSessionStore
from nanobot.session.source import SourceValidationError
from nanobot.session.source_store import SourceSessionStore, SourceSnapshot, SourceState
from tests.session.test_source_store import seed, upgrade

KEY = "test:source-persistence"
PREPARE = [
    "clone_header", "clone_events", "ids", "links", "canonical", "snapshot", "serialize", "decode",
]
ORDINARY = [*PREPARE, "select_writer", "select_path", "key", "path", "write"]
ROOT = [
    *PREPARE, "namespace", "select_current", "current", "select_root_write",
    "select_storage_key", "key", "storage_key", "root_write",
]


@pytest.fixture(autouse=True)
def operation():
    selected = getattr(JsonlSessionStore, "persist_source_records", None)
    assert callable(selected), "validated source persistence operation is missing"
    return selected


def probe(monkeypatch, *, fault=None):
    trace, paths, writes = [], [], []
    failure = OSError("injected source persistence failure")

    def step(name):
        trace.append(name)
        if name == fault:
            raise failure

    class Header(dict):
        def __getitem__(self, name):
            if name == "key":
                step("key")
            return super().__getitem__(name)

    header = Header(key=KEY, metadata={"custom": {"keep": True}})
    events = [{"role": "user", "content": "retained 中文 🌟", "_event_id": "event"}]
    path = object()

    class Backend:
        def __init__(self):
            self.sessions_dir = Path(__file__).parent
            self.on_select = None

        @property
        def _write_text_atomic(self):
            step("select_writer")
            selected = self.write
            if self.on_select is not None:
                self.on_select()
            return selected

        @property
        def get_session_path(self):
            step("select_path")

            def resolve(key):
                step("path")
                assert key == KEY
                paths.append(self)
                return path

            return resolve

        def write(self, selected_path, text):
            step("write")
            assert selected_path is path
            writes.append((self, text))

        @property
        def storage_key(self):
            step("select_storage_key")

            def storage_key(key):
                step("storage_key")
                assert key == KEY
                return "stored-key"

            return storage_key

    backend = Backend()
    source = SourceSessionStore(backend)
    snapshot = SourceSnapshot(SourceState("store", KEY, "incarnation", 1, 0, None, ""), {}, [], None)
    original_copy = source_store.deepcopy
    original_ids = source_store.validate_event_ids
    original_links = source_store.validate_execution_links
    original_canonical = source_store.canonical_events
    original_bytes = source_store._source_bytes

    def cloning(value):
        if value is header:
            step("clone_header")
        elif value is events:
            step("clone_events")
        return original_copy(value)

    def ids(value):
        step("ids")
        return original_ids(value)

    def links(value):
        step("links")
        return original_links(value)

    def canonical(value):
        step("canonical")
        return original_canonical(value)

    def snapshotting(cloned_header, cloned_events):
        step("snapshot")
        assert cloned_header is not header and cloned_events is not events
        return snapshot

    class Serialized(bytes):
        def decode(self, encoding="utf-8"):
            step("decode")
            assert encoding == "utf-8"
            return super().decode(encoding)

    def serialize(cloned_header, cloned_events):
        step("serialize")
        assert cloned_header is not header and cloned_events is not events
        return Serialized(original_bytes(cloned_header, cloned_events))

    class RootAuthority:
        @property
        def require_current(self):
            step("select_current")

            def current(selected_backend):
                step("current")
                assert selected_backend is source._backend

            return current

        @property
        def write_text_atomic(self):
            step("select_root_write")

            def write(name, text):
                step("root_write")
                assert name == "stored-key.jsonl"
                writes.append((self, text))

            return write

    root = RootAuthority()

    def namespace(selected_backend):
        step("namespace")
        assert selected_backend is source._backend

    monkeypatch.setattr(source_store, "deepcopy", cloning)
    monkeypatch.setattr(source_store, "validate_event_ids", ids)
    monkeypatch.setattr(source_store, "validate_execution_links", links)
    monkeypatch.setattr(source_store, "canonical_events", canonical)
    monkeypatch.setattr(source_store, "_source_bytes", serialize)
    monkeypatch.setattr(source, "_snapshot", snapshotting)
    monkeypatch.setattr(mode_storage, "require_namespace_write_access", namespace)
    return source, header, events, snapshot, backend, root, trace, paths, writes, failure


def invoke(operation, setup, *, rooted=False):
    source, header, events, _snapshot, _backend, root, *_rest = setup
    return operation(
        source, header, events, root_authority=root if rooted else None,
        observe_backend=lambda: source._backend,
    )


def test_ordinary_commit_preserves_complete_operand_and_hook_order(monkeypatch, operation):
    setup = probe(monkeypatch)
    result = invoke(operation, setup)
    source, _header, _events, snapshot, backend, _root, trace, paths, writes, _failure = setup
    assert result is snapshot and trace == ORDINARY
    assert paths == [backend] and writes[0][0] is backend
    assert source._backend is backend
    assert json.loads(writes[0][1].splitlines()[0])["key"] == KEY


@pytest.mark.parametrize("fault", ORDINARY)
def test_ordinary_first_failure_stops_every_later_operation(monkeypatch, operation, fault):
    setup = probe(monkeypatch, fault=fault)
    with pytest.raises(OSError) as caught:
        invoke(operation, setup)
    assert caught.value is setup[-1] and setup[6] == ORDINARY[:ORDINARY.index(fault) + 1]


def test_root_commit_retains_namespace_current_key_and_write_order(monkeypatch, operation):
    setup = probe(monkeypatch)
    assert invoke(operation, setup, rooted=True) is setup[3]
    assert setup[6] == ROOT and setup[7] == [] and setup[8][0][0] is setup[5]


@pytest.mark.parametrize("fault", ROOT[len(PREPARE):])
def test_root_first_failure_never_selects_ordinary_writer(monkeypatch, operation, fault):
    setup = probe(monkeypatch, fault=fault)
    with pytest.raises(OSError) as caught:
        invoke(operation, setup, rooted=True)
    assert caught.value is setup[-1] and setup[6] == ROOT[:ROOT.index(fault) + 1]
    assert "select_writer" not in setup[6]


@pytest.mark.parametrize("rows", [
    [{"role": "user", "content": "missing identity"}],
    [{"role": "invalid", "_event_id": "event"}],
    [{"role": "user", "_event_id": "event", "content": float("nan")}],
    [{"role": "user", "_event_id": "event"}, {"role": "user", "_event_id": "event"}],
])
def test_invalid_records_cannot_reach_snapshot_or_any_writer(monkeypatch, operation, rows):
    setup = probe(monkeypatch)
    source, header, *_rest = setup
    with pytest.raises(SourceValidationError):
        operation(source, header, rows, root_authority=None, observe_backend=lambda: source._backend)
    assert "snapshot" not in setup[6] and "select_writer" not in setup[6] and setup[8] == []


def test_prepared_handoff_is_detached_data_not_write_authority(monkeypatch):
    setup = probe(monkeypatch)
    source, header, events, snapshot, *_rest = setup
    prepared = SourceSessionStore.prepare_source_persistence(source, header, events)
    assert setup[6] == PREPARE and setup[8] == []
    assert [field.name for field in fields(prepared)] == ["snapshot", "text", "read_key"]
    assert not any(hasattr(prepared, name) for name in ("__dict__", "backend", "owner", "fd", "write"))
    with pytest.raises(FrozenInstanceError):
        prepared.text = "replacement"
    header["key"] = "outside mutation"
    events[0]["content"] = "outside mutation"
    assert prepared.snapshot is snapshot and prepared.read_key() == KEY
    assert json.loads(prepared.text.splitlines()[1])["content"] == "retained 中文 🌟"


def test_backend_prepares_nonvirtually_instead_of_accepting_forged_data(monkeypatch, operation):
    setup = probe(monkeypatch)
    monkeypatch.setattr(
        setup[0], "prepare_source_persistence",
        lambda *_args: pytest.fail("instance preparation override bypassed source validation"),
    )
    assert invoke(operation, setup) is setup[3] and setup[6] == ORDINARY


def test_writer_selection_and_path_use_two_late_backend_observations(monkeypatch, operation):
    setup = probe(monkeypatch)
    source, _header, _events, _snapshot, first, _root, trace, paths, writes, _failure = setup
    second = type(first)()
    first.on_select = lambda: setattr(source, "_backend", second)
    assert invoke(operation, setup) is setup[3]
    assert trace == ORDINARY and paths == [second] and writes[0][0] is first


def test_real_compatibility_commit_retains_detached_snapshot_and_atomic_bytes(tmp_path):
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    snapshot = upgrade(backend, seed(backend))
    header, events = snapshot.header, snapshot.events
    header["revision"] += 1
    events[0]["content"] = "committed 中文 🌟"
    with backend.locked_session_files():
        result = backend.source._write(header, events)
    durable = backend.get_session_path(snapshot.state.session_key).read_bytes()
    header["metadata"]["custom"]["keep"] = False
    events[0]["content"] = "caller mutation"
    assert result.events[0]["content"] == "committed 中文 🌟"
    assert result.header["metadata"]["custom"]["keep"] is True
    assert backend.source.load_source_snapshot(snapshot.state.session_key).events == result.events
    assert backend.get_session_path(snapshot.state.session_key).read_bytes() == durable
    assert not list(backend.sessions_dir.glob("*.tmp"))
