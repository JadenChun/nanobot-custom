"""Strict source CAS, exact evidence, and atomic JSONL failure behavior."""

import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from nanobot.session.manager import (
    JsonlSessionStore,
    Session,
    SessionStorageError,
    SessionWriteConflict,
)
from nanobot.session.source import SourceValidationError, new_source_event, prefix_digest


@pytest.fixture
def backend(tmp_path):
    return JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")


def seed(backend, *, messages=None):
    session = Session("test:source")
    session.messages = messages if messages is not None else [{"role": "user", "content": "original"}]
    session.metadata["custom"] = {"keep": True}
    backend.save(session)
    return session


def upgrade(backend, session):
    return backend.source.upgrade_source_session(
        session.key, expected_revision=session.revision, expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )


def candidate(snapshot, *, boundary=None, identity="checkpoint"):
    boundary = len(snapshot.events) if boundary is None else boundary
    return {
        "schema_version": 1, "digest_version": 1, "summary_format_version": 1,
        "session_generation": snapshot.state.generation, "source_revision": snapshot.state.revision,
        "checkpoint_id": identity, "parent_checkpoint_id": snapshot.state.checkpoint_id,
        "producing_model": "model", "producing_provider": "provider",
        "digest_algorithm": "sha256", "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": "Retained facts", "artifact_references": [], "boundary": boundary,
        "boundary_event_id": snapshot.events[boundary - 1]["_event_id"] if boundary else None,
        "prefix_digest": prefix_digest(snapshot.events, boundary), "auxiliary_usage": None,
    }


def test_upgrade_preserves_content_identity_metadata_and_no_read_time_writes(backend):
    session = seed(backend, messages=[{"role": "user", "content": "same"}] * 2)
    before = deepcopy(session.messages)
    snapshot = upgrade(backend, session)
    assert snapshot.events == before
    assert snapshot.events[0]["_event_id"] != snapshot.events[1]["_event_id"]
    assert snapshot.header["metadata"]["custom"] == {"keep": True}
    path = backend.get_session_path(session.key)
    durable = path.read_bytes()
    snapshot.events[0]["content"] = "mutated snapshot"
    assert backend.source.load_source_snapshot(session.key).events == before
    assert path.read_bytes() == durable
    with pytest.raises(SessionWriteConflict):
        upgrade(backend, session)


def test_explicit_upgrade_assigns_missing_legacy_ids_without_value_deduplication(backend):
    key = "test:legacy"
    path = backend.get_session_path(key)
    rows = [{"role": "user", "content": "same"}] * 2
    header = {"_type": "metadata", "key": key, "metadata": {"other": 1}}
    path.write_text("\n".join(json.dumps(row) for row in [header, *rows]) + "\n")
    snapshot = backend.source.upgrade_source_session(
        key, expected_revision=0, expected_generation=0, expected_incarnation="legacy",
    )
    assert len({row["_event_id"] for row in snapshot.events}) == 2
    assert all("_run_id" not in row for row in snapshot.events)
    assert [{k: v for k, v in row.items() if k != "_event_id"} for row in snapshot.events] == rows


def test_append_requires_exact_state_before_idempotent_retry(backend):
    snapshot = upgrade(backend, seed(backend))
    event = new_source_event({"role": "assistant", "content": "result"}, run_id="run")
    committed = backend.source.append_source_events(snapshot.state.session_key, snapshot.state, [event])
    with pytest.raises(SessionWriteConflict):
        backend.source.append_source_events(snapshot.state.session_key, snapshot.state, [event])
    retry = backend.source.append_source_events(committed.state.session_key, committed.state, [event])
    assert retry.state == committed.state and len(retry.events) == 2
    with pytest.raises(SourceValidationError):
        backend.source.append_source_events(committed.state.session_key, committed.state, [dict(event, content="changed")])


def test_checkpoint_atomic_commit_does_not_edit_source_cursor_or_activity(backend):
    snapshot = upgrade(backend, seed(backend))
    committed = backend.source.commit_context_checkpoint(snapshot.state.session_key, snapshot.state, candidate(snapshot))
    assert committed.events == snapshot.events
    assert committed.state.generation == snapshot.state.generation
    assert committed.state.revision == snapshot.state.revision + 1
    for name in ("last_consolidated", "last_archived", "updated_at"):
        assert committed.header.get(name) == snapshot.header.get(name)
    appended = backend.source.append_source_events(committed.state.session_key, committed.state, [
        new_source_event({"role": "user", "content": "new input"}, run_id="next"),
    ])
    reloaded = backend.source.load_source_snapshot(appended.state.session_key)
    assert reloaded.checkpoint == committed.checkpoint
    assert reloaded.checkpoint["source_revision"] < reloaded.state.revision - 1


def test_stale_checkpoint_after_append_or_metadata_cannot_commit(backend):
    snapshot = upgrade(backend, seed(backend))
    proposal = candidate(snapshot)
    backend.source.update_metadata(snapshot.state.session_key, {"title": "new"})
    with pytest.raises(SessionWriteConflict):
        backend.source.commit_context_checkpoint(snapshot.state.session_key, snapshot.state, proposal)
    latest = backend.source.load_source_snapshot(snapshot.state.session_key)
    backend.source.append_source_events(latest.state.session_key, latest.state, [
        new_source_event({"role": "user", "content": "append"}, run_id="next"),
    ])
    with pytest.raises(SessionWriteConflict):
        backend.source.commit_context_checkpoint(latest.state.session_key, latest.state, candidate(latest))


def test_reset_invalidates_checkpoint_and_stale_ownership(backend):
    snapshot = upgrade(backend, seed(backend))
    committed = backend.source.commit_context_checkpoint(snapshot.state.session_key, snapshot.state, candidate(snapshot))
    reset = backend.source.reset_source_session(committed.state.session_key, committed.state, [])
    assert reset.events == [] and reset.checkpoint is None
    assert reset.state.generation == committed.state.generation + 1
    with pytest.raises(SessionWriteConflict):
        backend.source.commit_context_checkpoint(committed.state.session_key, committed.state, candidate(committed, identity="second"))


def test_corrupt_source_read_and_mutations_never_repair_or_overwrite(backend):
    snapshot = upgrade(backend, seed(backend))
    path = backend.get_session_path(snapshot.state.session_key)
    path.write_bytes(path.read_bytes() + b'{private corrupt evidence\n')
    before = path.read_bytes()
    for action in (
        lambda: backend.source.load_source_snapshot(snapshot.state.session_key),
        lambda: backend.source.append_source_events(snapshot.state.session_key, snapshot.state, []),
        lambda: backend.source.reset_source_session(snapshot.state.session_key, snapshot.state, []),
    ):
        with pytest.raises(SessionStorageError, match="^Session storage is invalid; recovery is required.$"):
            action()
        assert path.read_bytes() == before


def test_legacy_tool_link_upgrade_keeps_duplicate_provider_ids_distinct(backend):
    rows = []
    for content in ("first", "second"):
        rows.extend([
            {"role": "assistant", "tool_calls": [{"id": "same", "function": {"name": "read_file"}}]},
            {"role": "tool", "tool_call_id": "same", "content": content},
        ])
    snapshot = upgrade(backend, seed(backend, messages=rows))
    assert snapshot.events[0]["_execution_ids"] != snapshot.events[2]["_execution_ids"]
    assert snapshot.events[1]["_execution_id"] == snapshot.events[0]["_execution_ids"][0]
    assert snapshot.events[3]["_execution_id"] == snapshot.events[2]["_execution_ids"][0]
    assert backend.source.commit_context_checkpoint(snapshot.state.session_key, snapshot.state, candidate(snapshot)).checkpoint


@pytest.mark.parametrize("rows", [
    [{"role": "tool", "tool_call_id": "orphan", "content": "keep"}],
    [{"role": "assistant", "tool_calls": [{"id": "same"}, {"id": "same"}]}],
])
def test_ambiguous_upgrade_leaves_original_bytes_unchanged(backend, rows):
    session = seed(backend, messages=rows)
    before = backend.get_session_path(session.key).read_bytes()
    with pytest.raises(SourceValidationError):
        upgrade(backend, session)
    assert backend.get_session_path(session.key).read_bytes() == before


def test_unresolved_call_is_retained_but_cannot_be_covered(backend):
    snapshot = upgrade(backend, seed(backend, messages=[
        {"role": "user", "content": "request"},
        {"role": "assistant", "tool_calls": [{"id": "pending"}]},
    ]))
    with pytest.raises(SourceValidationError):
        backend.source.commit_context_checkpoint(snapshot.state.session_key, snapshot.state, candidate(snapshot))
    assert backend.source.commit_context_checkpoint(snapshot.state.session_key, snapshot.state, candidate(snapshot, boundary=1)).checkpoint


def test_upgrade_rejects_recovery_without_touching_sidecar(backend):
    session = seed(backend)
    session.metadata["runtime_checkpoint"] = {"phase": "awaiting_tools"}
    backend.save_runtime_checkpoint(session)
    main_before = backend.get_session_path(session.key).read_bytes()
    sidecar = backend.get_runtime_checkpoint_path(session.key)
    before = sidecar.read_bytes()
    with pytest.raises(SessionWriteConflict):
        upgrade(backend, session)
    assert backend.get_session_path(session.key).read_bytes() == main_before
    assert sidecar.read_bytes() == before


def test_invalid_stored_checkpoint_is_ignored_but_retained_for_cas(backend):
    snapshot = upgrade(backend, seed(backend))
    path = backend.get_session_path(snapshot.state.session_key)
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    header["metadata"]["context_checkpoint"] = {"checkpoint_id": "bad", "schema_version": 99}
    path.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    loaded = backend.source.load_source_snapshot(snapshot.state.session_key)
    assert loaded.checkpoint is None and loaded.state.checkpoint_id == "bad"
    assert loaded.header["metadata"]["context_checkpoint"]["schema_version"] == 99
    with pytest.raises(SessionWriteConflict):
        backend.source.append_source_events(snapshot.state.session_key, snapshot.state, [])
    committed = backend.source.commit_context_checkpoint(loaded.state.session_key, loaded.state, candidate(loaded))
    assert committed.checkpoint["parent_checkpoint_id"] == "bad"


def test_failure_after_replace_exposes_only_complete_new_snapshot(backend, monkeypatch):
    snapshot = upgrade(backend, seed(backend))
    before = backend.get_session_path(snapshot.state.session_key).read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(type(backend), "_fsync_directory", lambda *_: (_ for _ in ()).throw(OSError("fsync")))
        with pytest.raises(OSError):
            backend.source.commit_context_checkpoint(snapshot.state.session_key, snapshot.state, candidate(snapshot))
    assert snapshot.checkpoint is None
    assert backend.get_session_path(snapshot.state.session_key).read_bytes() != before
    loaded = backend.source.load_source_snapshot(snapshot.state.session_key)
    assert loaded.checkpoint and loaded.events == snapshot.events


def test_compatibility_readers_keep_source_exact_and_ignore_sidecar(backend):
    session = seed(backend, messages=[
        {"role": "assistant", "tool_calls": [{"id": "call", "function": {
            "name": "write_stdin", "arguments": '{"session_id": "old", "yield_time_ms": 1}',
        }}]},
        {"role": "tool", "tool_call_id": "call", "name": "write_stdin", "content": "original"},
    ])
    snapshot = upgrade(backend, session)
    sidecar = backend.get_runtime_checkpoint_path(session.key)
    sidecar.write_text('{"version":1,"private":"untouched"}')
    before = sidecar.read_bytes()
    assert backend.load(session.key).messages == snapshot.events
    assert backend.read(session.key)["messages"] == snapshot.events
    assert backend.read_metadata(session.key)["metadata"]["source_schema_version"] == 1
    assert backend.repair(session.key).messages == snapshot.events
    assert sidecar.read_bytes() == before
    with pytest.raises(SessionWriteConflict):
        backend.save(backend.load(session.key))
    with pytest.raises(SessionWriteConflict):
        backend.save_runtime_checkpoint(backend.load(session.key))
    assert sidecar.read_bytes() == before


def test_compatibility_metadata_update_uses_strict_source_transaction(backend):
    snapshot = upgrade(backend, seed(backend))
    committed = backend.source.commit_context_checkpoint(snapshot.state.session_key, snapshot.state, candidate(snapshot))
    assert backend.update_metadata(committed.state.session_key, {
        "title": "updated", "context_checkpoint": None, "source_schema_version": 99,
        "runtime_checkpoint": {"phase": "fake"},
    })
    loaded = backend.source.load_source_snapshot(committed.state.session_key)
    assert loaded.checkpoint == committed.checkpoint
    assert loaded.events == committed.events
    assert loaded.state.revision == committed.state.revision + 1
    assert loaded.header["metadata"]["title"] == "updated"
    assert "runtime_checkpoint" not in loaded.header["metadata"]


def test_known_source_corruption_does_not_enter_compatibility_repair(backend):
    snapshot = upgrade(backend, seed(backend))
    path = backend.get_session_path(snapshot.state.session_key)
    path.write_bytes(path.read_bytes() + b'{broken\n')
    before = path.read_bytes()
    for reader in (backend.load, backend.read, backend.read_metadata, backend.repair):
        with pytest.raises(SessionStorageError):
            reader(snapshot.state.session_key)
        assert path.read_bytes() == before


def test_consolidation_cursor_is_independent_of_context_checkpoint(backend):
    snapshot = upgrade(backend, seed(backend))
    checkpoint = backend.source.commit_context_checkpoint(snapshot.state.session_key, snapshot.state, candidate(snapshot))
    advanced = backend.source.advance_consolidated(checkpoint.state.session_key, checkpoint.state, 1)
    assert advanced.header["last_consolidated"] == 1
    assert advanced.header["last_archived"] == advanced.header["last_consolidated"] == 1
    assert advanced.checkpoint == checkpoint.checkpoint and advanced.events == checkpoint.events
    with pytest.raises(SessionWriteConflict):
        backend.source.advance_consolidated(checkpoint.state.session_key, checkpoint.state, 1)
    with pytest.raises(SourceValidationError):
        backend.source.advance_consolidated(advanced.state.session_key, advanced.state, 0)


@pytest.mark.parametrize("header", ["{broken", "", '{"_type":"metadata","metadata":null}'])
def test_unidentifiable_header_cannot_fall_back_to_legacy_load(backend, header):
    snapshot = upgrade(backend, seed(backend))
    path = backend.get_session_path(snapshot.state.session_key)
    lines = path.read_text().splitlines()
    path.write_text("\n".join([header, *lines[1:]]) + "\n")
    before = path.read_bytes()
    for reader in (backend.load, backend.read, backend.read_metadata, backend.source.load_source_snapshot):
        with pytest.raises(SessionStorageError):
            reader(snapshot.state.session_key)
        assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["last_archived", "last_consolidated"])
def test_legacy_single_cursor_upgrade_preserves_both_reader_aliases(backend, field):
    session = seed(backend)
    path = backend.get_session_path(session.key)
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    header.pop("last_archived")
    header.pop("last_consolidated")
    header[field] = 1
    path.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    snapshot = upgrade(backend, session)
    assert snapshot.header["last_archived"] == snapshot.header["last_consolidated"] == 1
    assert snapshot.to_session().last_consolidated == 1


def test_manager_cache_uses_same_source_metadata_filter(tmp_path):
    from nanobot.session.manager import SessionManager
    manager = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "private")
    original = manager.get_or_create("test:cache")
    manager.save(original)
    upgrade(manager._jsonl_store, original)
    # A cached legacy object from before upgrade is invalidated on metadata update.
    manager.update_session_metadata(original.key, {"runtime_checkpoint": {"phase": "fake"}})
    assert manager.get_cached(original.key) is None
    source = manager.get_or_create(original.key)
    manager.update_session_metadata(source.key, {
        "runtime_checkpoint": {"phase": "fake"}, "pending_user_turn": True,
        "webui_recovery": {"status": "resuming"}, "title": "new",
    })
    assert not any(name in source.metadata for name in (
        "runtime_checkpoint", "pending_user_turn", "webui_recovery",
    ))
    assert source.metadata["title"] == "new"


def _checkpoint_process(workspace, private, key, ready, go, outcomes, identity):
    try:
        backend = JsonlSessionStore(workspace, sessions_root=private)
        snapshot = backend.source.load_source_snapshot(key)
        proposal = candidate(snapshot, identity=identity)
        ready.put(True)
        if not go.wait(10):
            raise TimeoutError
        backend.source.commit_context_checkpoint(key, snapshot.state, proposal)
        outcomes.put("committed")
    except SessionWriteConflict:
        outcomes.put("conflict")
    except Exception as exc:
        outcomes.put(type(exc).__name__)


def test_two_processes_cannot_both_commit_same_checkpoint_base(backend, tmp_path):
    import multiprocessing
    snapshot = upgrade(backend, seed(backend))
    context = multiprocessing.get_context("spawn")
    ready, outcomes, go = context.Queue(), context.Queue(), context.Event()
    processes = [context.Process(target=_checkpoint_process, args=(
        tmp_path / "workspace", tmp_path / "private", snapshot.state.session_key,
        ready, go, outcomes, str(index),
    )) for index in range(2)]
    try:
        for process in processes:
            process.start()
        assert ready.get(timeout=10) and ready.get(timeout=10)
        go.set()
        assert sorted(outcomes.get(timeout=10) for _ in processes) == ["committed", "conflict"]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        ready.close()
        outcomes.close()
    loaded = backend.source.load_source_snapshot(snapshot.state.session_key)
    assert loaded.state.revision == snapshot.state.revision + 1
    assert loaded.events == snapshot.events


def test_store_binding_rejects_snapshot_from_copied_other_workspace(backend, tmp_path):
    snapshot = upgrade(backend, seed(backend))
    other = JsonlSessionStore(tmp_path / "other-workspace", sessions_root=tmp_path / "private")
    other.get_session_path(snapshot.state.session_key).write_bytes(
        backend.get_session_path(snapshot.state.session_key).read_bytes()
    )
    with pytest.raises(SessionWriteConflict):
        other.source.append_source_events(snapshot.state.session_key, snapshot.state, [])


def test_deleted_source_and_recreated_session_reject_old_state(backend):
    session = seed(backend)
    snapshot = upgrade(backend, session)
    backend.delete(session.key)
    with pytest.raises(SessionWriteConflict):
        backend.source.reset_source_session(session.key, snapshot.state, [])
    fresh = seed(backend)
    new = upgrade(backend, fresh)
    assert new.state.revision == snapshot.state.revision
    assert new.state.incarnation != snapshot.state.incarnation
    with pytest.raises(SessionWriteConflict):
        backend.source.append_source_events(session.key, snapshot.state, [])


def test_before_replace_failure_leaves_source_and_candidate_unchanged(backend, monkeypatch):
    snapshot = upgrade(backend, seed(backend))
    proposal = candidate(snapshot)
    before_candidate = deepcopy(proposal)
    before = backend.get_session_path(snapshot.state.session_key).read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr("nanobot.session.manager.os.replace", lambda *_: (_ for _ in ()).throw(OSError("replace")))
        with pytest.raises(OSError):
            backend.source.commit_context_checkpoint(snapshot.state.session_key, snapshot.state, proposal)
    assert backend.get_session_path(snapshot.state.session_key).read_bytes() == before
    assert proposal == before_candidate and snapshot.checkpoint is None


def test_file_fsync_precedes_replace_and_directory_fsync_follows(backend, monkeypatch):
    import os
    import stat
    snapshot = upgrade(backend, seed(backend))
    original_fsync, original_replace = os.fsync, os.replace
    order = []
    def fsync(fd):
        order.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        return original_fsync(fd)
    def replace(source, target):
        order.append("replace")
        return original_replace(source, target)
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    backend.source.commit_context_checkpoint(snapshot.state.session_key, snapshot.state, candidate(snapshot))
    assert order == ["file", "replace", "directory"]


@pytest.mark.parametrize("field", ["revision", "generation", "incarnation", "last_archived"])
def test_missing_governed_header_fields_are_not_defaulted(backend, field):
    snapshot = upgrade(backend, seed(backend))
    path = backend.get_session_path(snapshot.state.session_key)
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    del header[field]
    path.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    with pytest.raises(SessionStorageError):
        backend.source.load_source_snapshot(snapshot.state.session_key)


def test_boolean_expected_counter_is_not_equal_to_integer_counter(backend):
    from dataclasses import replace
    snapshot = upgrade(backend, seed(backend))
    malformed = replace(snapshot.state, generation=False)
    with pytest.raises(SessionWriteConflict):
        backend.source.append_source_events(snapshot.state.session_key, malformed, [])


@pytest.mark.parametrize("payload", [
    {"role": [], "_event_id": "event"},
    {"role": "user", "_event_id": "event", "_run_id": True},
])
def test_invalid_source_input_raises_typed_validation_error(backend, payload):
    snapshot = upgrade(backend, seed(backend))
    with pytest.raises(SourceValidationError):
        backend.source.append_source_events(snapshot.state.session_key, snapshot.state, [payload])
