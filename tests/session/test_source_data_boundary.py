"""Source conversion exposes data operations, not storage or write authority."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy

import pytest

from nanobot.session import session_mode, source_store
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source import SourceValidationError
from tests.session.test_source_store import seed, upgrade


@pytest.fixture
def storage(tmp_path):
    return JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")


def test_snapshot_records_committed_base_through_existing_operation(storage, monkeypatch):
    snapshot = upgrade(storage, seed(storage))
    path = storage.get_session_path(snapshot.state.session_key)
    before = path.read_bytes()
    trace = []
    record = Session.record_committed_base
    remember = Session._remember_committed_base

    def recording(self):
        trace.append(("public", self))
        record(self)

    def remembering(self):
        trace.append(("original", self))
        remember(self)

    monkeypatch.setattr(Session, "record_committed_base", recording)
    monkeypatch.setattr(Session, "_remember_committed_base", remembering)
    session = snapshot.to_session()
    assert trace == [("public", session), ("original", session)]
    assert session.key == snapshot.state.session_key
    assert session.revision == session._base_revision == snapshot.state.revision
    assert session.generation == session._base_generation == snapshot.state.generation
    assert session.incarnation == session._base_incarnation == snapshot.state.incarnation
    assert session.created_at.isoformat() == snapshot.header["created_at"]
    assert session.updated_at.isoformat() == snapshot.header["updated_at"]
    assert session.last_archived == snapshot.header["last_archived"]
    assert session.provider_state is session._base_provider_state is None
    assert session.messages == session._base_messages == snapshot.events
    assert session.metadata == session._base_metadata == snapshot.header["metadata"]
    assert session._replacement_pending is session._projection_edit_pending is False
    assert session._append_ids == {}
    session.messages[0]["content"] = "detached edit"
    session.metadata["custom"]["keep"] = False
    assert session._base_messages == snapshot.events
    assert session._base_metadata == snapshot.header["metadata"]
    assert storage.source.load_source_snapshot(session.key).events == snapshot.events
    assert path.read_bytes() == before


@pytest.mark.parametrize("seam", ["record_committed_base", "_remember_committed_base"])
def test_snapshot_retains_recorder_fault_identity_and_durable_bytes(storage, monkeypatch, seam):
    snapshot = upgrade(storage, seed(storage))
    path = storage.get_session_path(snapshot.state.session_key)
    before = path.read_bytes()
    fault = RuntimeError("committed recorder fault")
    observed = []

    def fail(self):
        observed.append(self)
        raise fault

    monkeypatch.setattr(Session, seam, fail)
    with pytest.raises(RuntimeError) as failure:
        snapshot.to_session()
    assert failure.value is fault
    assert len(observed) == 1 and observed[0].messages == snapshot.events
    assert path.read_bytes() == before


def test_selected_upgrader_preserves_structured_linkage_and_strips_legacy_claims(storage):
    session = seed(storage, messages=[
        {"role": "assistant", "tool_calls": [{"id": "same"}]},
        {"role": "tool", "tool_call_id": "same", "content": "first",
         "_source_origin": {"claim": "untrusted"}, "_artifact_read": {"claim": "untrusted"}},
        {"role": "assistant", "tool_calls": [{"id": "same"}]},
        {"role": "tool", "tool_call_id": "same", "content": "second"},
    ])
    original = deepcopy(session.messages)
    path = storage.get_session_path(session.key)
    before = path.read_bytes()
    selected = source_store.select_legacy_source_event_upgrader()
    assert selected is source_store._upgrade_events
    events = selected(session.messages)
    assert session.messages == original
    assert len({event["_event_id"] for event in events}) == 4
    assert events[0]["_execution_ids"] != events[2]["_execution_ids"]
    assert events[1]["_execution_id"] == events[0]["_execution_ids"][0]
    assert events[3]["_execution_id"] == events[2]["_execution_ids"][0]
    assert all("_artifact_read" not in event and "_source_origin" not in event for event in events)
    events[1]["content"] = "detached"
    assert session.messages == original and path.read_bytes() == before


@pytest.mark.parametrize("events,message", [
    ([{"role": "user", "tool_calls": []}], "Invalid legacy execution linkage"),
    ([{"role": "assistant", "tool_calls": "bad"}], "Invalid legacy execution linkage"),
    ([{"role": "assistant", "tool_calls": [{"id": "p"}], "_execution_ids": []}],
     "Invalid legacy execution linkage"),
    ([{"role": "assistant", "tool_calls": [None]}], "Invalid legacy provider call"),
    ([{"role": "assistant", "tool_calls": [{"id": ""}]}], "Invalid legacy provider call"),
    ([{"role": "assistant", "tool_calls": [{"id": "same"}, {"id": "same"}]}],
     "Ambiguous legacy execution linkage"),
    ([{"role": "tool", "tool_call_id": "orphan"}], "Unmatched legacy execution result"),
    ([{"role": "assistant", "tool_calls": [{"id": "p"}], "_execution_ids": ["one"]},
      {"role": "tool", "tool_call_id": "p", "_execution_id": "other"}],
     "Conflicting legacy execution linkage"),
])
def test_selected_upgrader_retains_original_refusals_without_mutation(storage, events, message):
    session = seed(storage, messages=events)
    original = deepcopy(session.messages)
    path = storage.get_session_path(session.key)
    before = path.read_bytes()
    selected = source_store.select_legacy_source_event_upgrader()
    for operation in (selected, source_store._upgrade_events):
        with pytest.raises(SourceValidationError, match=f"^{message}$"):
            operation(session.messages)
        assert session.messages == original and path.read_bytes() == before


def test_selector_returns_current_callable_and_preserves_fault_identity(storage, monkeypatch):
    session = seed(storage)
    fault = SourceValidationError("original event fault")
    observed = []

    def fail(events):
        observed.append(events)
        raise fault

    monkeypatch.setattr(source_store, "_upgrade_events", fail)
    selected = source_store.select_legacy_source_event_upgrader()
    assert selected is fail
    with pytest.raises(SourceValidationError) as failure:
        selected(session.messages)
    assert failure.value is fault and observed == [session.messages]
    assert observed[0] is session.messages


def test_mode_selects_once_at_import_and_does_not_follow_later_source_rebind(storage, monkeypatch):
    session = seed(storage)
    original = source_store._upgrade_events
    selected_events = original(session.messages)
    trace = []

    def selected(events):
        assert events is session.messages
        trace.append("convert")
        return selected_events

    def select():
        trace.append("select")
        return selected

    def late(events):
        raise AssertionError("late source rebind changed import-time binding")

    monkeypatch.setattr(source_store, "select_legacy_source_event_upgrader", select)
    name = "nanobot.session._source_data_import_probe"
    spec = importlib.util.spec_from_file_location(name, session_mode.__file__)
    assert spec is not None and spec.loader is not None
    isolated = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, isolated)
    spec.loader.exec_module(isolated)
    assert trace == ["select"] and isolated._upgrade_events is selected
    monkeypatch.setattr(source_store, "_upgrade_events", late)
    data, revision = isolated.SessionModeResolver._render_canonical(session)
    assert trace == ["select", "convert"]
    assert revision == session.revision + 1
    assert [json.loads(line) for line in data.splitlines()][1:] == selected_events


def test_mode_keeps_module_local_upgrader_fault_seam(storage, monkeypatch):
    session = seed(storage)
    path = storage.get_session_path(session.key)
    before = path.read_bytes()
    fault = SourceValidationError("consumer event fault")

    def fail(events):
        assert events is session.messages
        raise fault

    monkeypatch.setattr(session_mode, "_upgrade_events", fail)
    with pytest.raises(SourceValidationError) as failure:
        session_mode.SessionModeResolver._render_canonical(session)
    assert failure.value is fault and path.read_bytes() == before
