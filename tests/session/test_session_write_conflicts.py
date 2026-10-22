"""Real JSONL stores must preserve independent writers and reject stale resets."""

import json
from copy import deepcopy

import pytest

from nanobot.session.manager import (
    JsonlSessionStore,
    Session,
    SessionStorageError,
    SessionWriteConflict,
)


@pytest.fixture
def store(tmp_path):
    return JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")


def seed(store):
    session = Session("test:revision")
    session.add_message("user", "base")
    store.save(session)
    return session


def test_independent_identical_appends_are_not_content_deduplicated(store):
    original = seed(store)
    first, second = store.load(original.key), store.load(original.key)
    first.add_message("assistant", "same")
    second.add_message("assistant", "same")
    store.save(first)
    store.save(second)
    loaded = store.load(original.key)
    assert [row["content"] for row in loaded.messages] == ["base", "same", "same"]
    assert len({row["_event_id"] for row in loaded.messages}) == 3
    assert loaded.revision == 3 and loaded.generation == 0


def test_stale_reset_and_append_after_reset_cannot_overwrite_history(store):
    original = seed(store)
    reset, append = store.load(original.key), store.load(original.key)
    reset.clear()
    append.add_message("user", "new")
    store.save(append)
    with pytest.raises(SessionWriteConflict):
        store.save(reset)
    assert reset.revision == 1 and reset.generation == 0
    reset = store.load(original.key)
    reset.clear()
    store.save(reset)
    assert reset.generation == 1 and reset.revision == 3
    append.add_message("assistant", "stale")
    with pytest.raises(SessionWriteConflict):
        store.save(append)
    assert store.load(original.key).messages == []


def test_delete_and_recreate_cannot_be_undone_by_old_writer(store):
    old = seed(store)
    assert store.delete(old.key)
    with pytest.raises(SessionWriteConflict):
        store.save(old)
    fresh = Session(old.key)
    fresh.add_message("user", "fresh")
    store.save(fresh)
    assert fresh.revision == old.revision
    assert fresh.incarnation != old.incarnation
    with pytest.raises(SessionWriteConflict):
        store.save(old)
    assert store.load(old.key).messages == fresh.messages


def test_stale_metadata_preserves_new_fields_and_reserved_state(store):
    old = seed(store)
    old.metadata["title"] = "old"
    store.save(old)
    stale = store.load(old.key)
    store.update_metadata(old.key, {"title": "new", "new_field": 1})
    stale.metadata["other"] = 2
    stale.metadata["context_checkpoint"] = {"forged": True}
    stale.revision = 999
    store.save(stale)
    loaded = store.load(old.key)
    assert loaded.metadata == {"title": "new", "new_field": 1, "other": 2}
    assert loaded.revision == 4


def test_existing_prefix_edits_need_explicit_replacement(store):
    session = seed(store)
    session.messages[0]["content"] = "edited"
    with pytest.raises(SessionWriteConflict):
        store.save(session)
    session.replace_messages(session.messages)
    store.save(session)
    assert session.generation == 1
    assert store.load(session.key).messages[0]["content"] == "edited"


def test_before_replace_failure_keeps_disk_and_caller_bookkeeping(store, monkeypatch):
    session = seed(store)
    session.add_message("assistant", "new")
    before = store.get_session_path(session.key).read_bytes()
    messages = deepcopy(session.messages)
    with monkeypatch.context() as patch:
        patch.setattr("nanobot.session.manager.os.replace", lambda *_: (_ for _ in ()).throw(OSError()))
        with pytest.raises(OSError):
            store.save(session)
    assert store.get_session_path(session.key).read_bytes() == before
    assert session.revision == 1 and session.messages == messages
    store.save(session)
    assert len(store.load(session.key).messages) == 2


def test_after_replace_failure_retry_does_not_duplicate_append(store, monkeypatch):
    session = seed(store)
    session.add_message("assistant", "new")
    from nanobot.session import manager
    original_replace = manager.os.replace
    def replace_then_fail(source, target):
        original_replace(source, target)
        raise OSError("after replacement")
    with monkeypatch.context() as patch:
        patch.setattr(manager.os, "replace", replace_then_fail)
        with pytest.raises(OSError):
            store.save(session)
    assert session.revision == 1
    assert len(store.load(session.key).messages) == 2
    store.save(session)
    assert len(store.load(session.key).messages) == 2


def test_corrupt_existing_evidence_is_not_repaired_during_save(store):
    session = seed(store)
    path = store.get_session_path(session.key)
    path.write_text(path.read_text() + '{secret corrupted evidence\n')
    before = path.read_bytes()
    with pytest.raises(SessionStorageError):
        store.save(session)
    assert path.read_bytes() == before


@pytest.mark.parametrize("field,value", [("revision", True), ("generation", -1), ("revision", "1")])
def test_malformed_counters_fail_closed(store, field, value):
    session = seed(store)
    path = store.get_session_path(session.key)
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    header[field] = value
    lines[0] = json.dumps(header)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(SessionStorageError):
        store.load(session.key)


def _append_in_process(workspace, private, key, ready, go, outcome):
    try:
        store = JsonlSessionStore(workspace, sessions_root=private)
        session = store.load(key)
        session.add_message("assistant", "same")
        ready.put(True)
        if not go.wait(10):
            raise TimeoutError
        store.save(session, fsync=True)
        outcome.put("saved")
    except Exception as exc:
        outcome.put(type(exc).__name__)


def test_two_real_processes_merge_under_file_lock(store, tmp_path):
    import multiprocessing
    original = seed(store)
    context = multiprocessing.get_context("spawn")
    ready, outcome, go = context.Queue(), context.Queue(), context.Event()
    processes = [context.Process(target=_append_in_process, args=(
        tmp_path / "workspace", tmp_path / "private", original.key, ready, go, outcome,
    )) for _ in range(2)]
    try:
        for process in processes:
            process.start()
        assert ready.get(timeout=10) and ready.get(timeout=10)
        go.set()
        assert [outcome.get(timeout=10) for _ in processes] == ["saved", "saved"]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        ready.close()
        outcome.close()
    loaded = store.load(original.key)
    assert len(loaded.messages) == 3 and loaded.revision == 3
    assert len({row["_event_id"] for row in loaded.messages}) == 3


def test_legacy_identical_rows_keep_distinct_positions_on_append(store):
    key = "legacy:session"
    path = store.get_session_path(key)
    row = {"role": "user", "content": "same"}
    path.write_text("\n".join(map(json.dumps, [
        {"_type": "metadata", "key": key}, row, row,
    ])) + "\n")
    first, second = store.load(key), store.load(key)
    assert first.revision == 0 and first.generation == 0
    assert path.read_text().count("_event_id") == 0
    first.add_message("assistant", "same")
    second.add_message("assistant", "same")
    store.save(first)
    store.save(second)
    loaded = store.load(key)
    assert len(loaded.messages) == 4
    assert loaded.messages[:2] == [row, row]
    assert loaded.revision == 2


def test_summary_insertion_is_revision_checked_without_reset_generation(store):
    original = seed(store)
    summary, append = store.load(original.key), store.load(original.key)
    summary.commit_summary_checkpoint("summary", insert_at=0)
    append.add_message("assistant", "tail")
    store.save(append)
    with pytest.raises(SessionWriteConflict):
        store.save(summary)
    latest = store.load(original.key)
    latest.commit_summary_checkpoint("summary", insert_at=1)
    store.save(latest)
    assert latest.generation == 0 and latest.revision == 3
    assert latest.messages[-1]["content"] == "tail"


def test_cached_metadata_update_is_not_replayed_over_later_update(tmp_path):
    from nanobot.session.manager import SessionManager
    manager = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "private")
    session = manager.get_or_create("test:metadata")
    manager.save(session)
    manager.update_session_metadata(session.key, {"title": "first", "context_checkpoint": "fake"})
    assert "context_checkpoint" not in session.metadata
    manager._store.update_metadata(session.key, {"title": "second"})
    session.add_message("user", "new")
    manager.save(session)
    assert session.metadata["title"] == "second"


def test_stale_runtime_sidecar_cannot_replace_newer_revision(store):
    original = seed(store)
    stale = store.load(original.key)
    store.update_metadata(original.key, {"title": "new"})
    stale.metadata["runtime_checkpoint"] = {"phase": "awaiting_tools"}
    with pytest.raises(SessionWriteConflict):
        store.save_runtime_checkpoint(stale)
    assert not store.get_runtime_checkpoint_path(original.key).exists()


def provider_state(identifier):
    from nanobot.providers.base import ProviderConversationState
    return ProviderConversationState(
        kind="openai_responses", provider="test", model="test", version=1,
        payload={"response_id": identifier},
    )


def test_unrelated_save_preserves_loaded_recovery_sidecar(store):
    session = seed(store)
    session.metadata["runtime_checkpoint"] = {"phase": "awaiting_tools"}
    session.provider_state = provider_state("recovered")
    store.save_runtime_checkpoint(session)
    loaded = store.load(session.key)
    loaded.metadata["title"] = "changed"
    store.save(loaded)
    restored = store.load(session.key)
    assert restored.metadata["runtime_checkpoint"] == {"phase": "awaiting_tools"}
    assert restored.provider_state == provider_state("recovered")


def test_metadata_primitive_materializes_recovery_before_advancing_revision(store):
    session = seed(store)
    session.metadata["runtime_checkpoint"] = {"phase": "awaiting_tools"}
    session.provider_state = provider_state("recovered")
    store.save_runtime_checkpoint(session)
    store.update_metadata(session.key, {"title": "new"})
    restored = store.load(session.key)
    assert restored.metadata["runtime_checkpoint"] == {"phase": "awaiting_tools"}
    assert restored.provider_state == provider_state("recovered")
    assert restored.revision == 2


def test_stale_metadata_save_preserves_installed_or_invalidated_provider_state(store):
    original = seed(store)
    stale = store.load(original.key)
    original.provider_state = provider_state("new")
    store.save(original)
    stale.metadata["title"] = "title"
    store.save(stale)
    assert store.load(original.key).provider_state == provider_state("new")
    original.provider_state = None
    store.save(original)
    stale.metadata["title"] = "other"
    store.save(stale)
    assert store.load(original.key).provider_state is None


def test_conflicting_provider_changes_fail_and_merged_appends_invalidate_handles(store):
    original = seed(store)
    first, second = store.load(original.key), store.load(original.key)
    first.provider_state = provider_state("first")
    second.provider_state = provider_state("second")
    store.save(first)
    with pytest.raises(SessionWriteConflict):
        store.save(second)
    first, second = store.load(original.key), store.load(original.key)
    first.add_message("assistant", "branch one")
    second.add_message("assistant", "branch two")
    store.save(first)
    store.save(second)
    assert store.load(original.key).provider_state is None


def test_stale_save_does_not_regress_activity_timestamp(store):
    from datetime import timedelta
    original = seed(store)
    stale = store.load(original.key)
    original.updated_at += timedelta(hours=1)
    store.save(original)
    stale.metadata["title"] = "new title"
    store.save(stale)
    assert store.load(original.key).updated_at == original.updated_at


def test_provider_change_based_on_old_transcript_is_rejected(store):
    session = seed(store)
    stale = store.load(session.key)
    session.add_message("user", "new transcript")
    store.save(session)
    stale.provider_state = provider_state("based-on-old-transcript")
    with pytest.raises(SessionWriteConflict):
        store.save(stale)
    assert store.load(session.key).provider_state is None


def test_scheduled_ring_merges_by_run_id_and_stays_bounded(store):
    session = seed(store)
    first, second = store.load(session.key), store.load(session.key)
    first.metadata[Session.SCHEDULED_RING_KEY] = [{"run_id": str(i)} for i in range(20)]
    second.metadata[Session.SCHEDULED_RING_KEY] = [{"run_id": "19"}, {"run_id": "20"}]
    store.save(first)
    store.save(second)
    assert store.load(session.key).recent_scheduled_runs() == [
        {"run_id": str(i)} for i in range(1, 21)
    ]


def test_stale_recovery_change_cannot_replace_newer_transcript_outcome(store):
    original = seed(store)
    first, stale = store.load(original.key), store.load(original.key)
    first.add_message("assistant", "completed work")
    first.metadata["runtime_checkpoint"] = {"phase": "tools_completed"}
    store.save(first)
    stale.metadata["runtime_checkpoint"] = {"phase": "awaiting_tools"}
    with pytest.raises(SessionWriteConflict):
        store.save(stale)
    assert store.load(original.key).metadata["runtime_checkpoint"] == {"phase": "tools_completed"}


def test_independent_append_with_recovery_evidence_must_reload(store):
    original = seed(store)
    first, stale = store.load(original.key), store.load(original.key)
    first.add_message("assistant", "completed work")
    first.metadata["runtime_checkpoint"] = {"phase": "tools_completed"}
    store.save(first)
    stale.add_message("user", "independent input")
    with pytest.raises(SessionWriteConflict):
        store.save(stale)
    assert store.load(original.key).metadata["runtime_checkpoint"] == {"phase": "tools_completed"}


def test_concurrent_sidecar_changes_conflict_even_without_main_revision_change(store):
    original = seed(store)
    first, stale = store.load(original.key), store.load(original.key)
    first.metadata["runtime_checkpoint"] = {"phase": "tools_completed"}
    store.save_runtime_checkpoint(first)
    stale.metadata["runtime_checkpoint"] = {"phase": "awaiting_tools"}
    with pytest.raises(SessionWriteConflict):
        store.save_runtime_checkpoint(stale)
    assert store.load(original.key).metadata["runtime_checkpoint"] == {"phase": "tools_completed"}


def test_explicit_checkpoint_removal_clears_materialized_main_state(store):
    session = seed(store)
    session.metadata["runtime_checkpoint"] = {"phase": "tools_completed"}
    session.provider_state = provider_state("latest")
    store.save_runtime_checkpoint(session)
    store.update_metadata(session.key, {"title": "materialized"})
    session = store.load(session.key)
    session.metadata.pop("runtime_checkpoint")
    store.save_runtime_checkpoint(session)
    restored = store.load(session.key)
    assert "runtime_checkpoint" not in restored.metadata
    assert restored.provider_state == provider_state("latest")


def test_new_append_cannot_absorb_unobserved_active_sidecar(store):
    session = seed(store)
    stale = store.load(session.key)
    session.metadata["runtime_checkpoint"] = {"phase": "awaiting_tools"}
    store.save_runtime_checkpoint(session)
    stale.add_message("user", "independent input")
    with pytest.raises(SessionWriteConflict):
        store.save(stale)
