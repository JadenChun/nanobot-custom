"""Preserve commit preparation/publication through existing persistence routes."""

import errno
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest

from nanobot.providers.base import ProviderConversationState
from nanobot.session import manager as module
from nanobot.session.manager import JsonlSessionStore, Session, SessionManager, SessionWriteConflict


@pytest.fixture
def store(tmp_path):
    return JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")


def seed(store):
    session = Session("test:commit-owner")
    session.add_message("user", "base")
    store.save(session)
    return session


def state(session):
    return deepcopy({name: value for name, value in vars(session).items() if name != "_append_ids"})


@pytest.mark.parametrize("fault", ["before_replace", "after_replace", "unlink", "file_fsync", "dir_fsync"])
def test_full_write_failure_retains_caller_and_retry_identity(store, monkeypatch, fault):
    session = seed(store)
    session.add_message("assistant", "tail")
    before = state(session)
    path = store.get_session_path(session.key)
    original_bytes = path.read_bytes()
    replace = module.os.replace
    unlink = Path.unlink
    calls = []

    def replacement(source, target):
        if fault == "before_replace":
            raise OSError("injected")
        replace(source, target)
        if fault == "after_replace":
            raise OSError("injected")

    def removal(target, *args, **kwargs):
        if target == store.get_runtime_checkpoint_path(session.key):
            raise OSError("injected")
        return unlink(target, *args, **kwargs)

    def sync(fd):
        calls.append(fd)
        if fault == "file_fsync" or len(calls) == 2:
            raise OSError(errno.EIO, "injected")

    with monkeypatch.context() as patch:
        if "replace" in fault:
            patch.setattr(module.os, "replace", replacement)
        elif fault == "unlink":
            patch.setattr(Path, "unlink", removal)
        else:
            patch.setattr(module.os, "fsync", sync)
        with pytest.raises(OSError, match="injected"):
            store.save(session, fsync="fsync" in fault)
    assert state(session) == before
    row = session.messages[-1]
    identity = session._append_ids[(1, id(row))][1]
    assert "_event_id" not in row
    disk_changed = fault in {"after_replace", "unlink", "dir_fsync"}
    assert (path.read_bytes() != original_bytes) is disk_changed
    if disk_changed:
        assert store.load(session.key).messages[-1]["_event_id"] == identity
    store.save(session)
    assert len(store.load(session.key).messages) == 2
    assert session.messages[-1]["_event_id"] == identity
    assert session._append_ids == {}


def test_replacing_pending_row_at_same_position_allocates_new_identity(store, monkeypatch):
    session = seed(store)
    session.add_message("assistant", "first")
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "replace", Mock(side_effect=OSError("injected")))
        with pytest.raises(OSError):
            store.save(session)
    first = session.messages[-1]
    identity = session._append_ids[(1, id(first))][1]
    session.messages[-1] = {"role": "assistant", "content": "second"}
    store.save(session)
    assert session.messages[-1]["_event_id"] != identity
    assert len(session.messages) == 2


@pytest.mark.parametrize("equal", [True, False])
def test_duplicate_append_identity_equality_is_content_free(store, equal):
    session = seed(store)
    session.add_message("assistant", "tail")
    store.save(session)
    duplicate = deepcopy(session.messages[-1])
    session.messages.append(duplicate)
    session._append_ids[(2, id(duplicate))] = (duplicate, duplicate["_event_id"])
    if equal:
        store.save(session)
        assert len(session.messages) == 2
    else:
        duplicate["content"] = "secret-unequal"
        with pytest.raises(SessionWriteConflict) as caught:
            store.save(session)
        assert str(caught.value) == SessionWriteConflict.DEFAULT_MESSAGE
        assert len(store.load(session.key).messages) == 2


def provider():
    return ProviderConversationState(kind="openai_responses", provider="test", model="test", version=1,
                                     payload={"response_id": "private-handle"})


@pytest.mark.parametrize("conflict", ["base", "checkpoint", "provider"])
def test_checkpoint_validation_phase_precedes_tempfile(store, monkeypatch, conflict):
    session = seed(store)
    session.metadata["runtime_checkpoint"] = {"phase": "local"}
    overlay = Mock()
    if conflict == "base":
        session._base_revision -= 1
    else:
        def overlay_conflict(disk, path):
            if conflict == "checkpoint":
                disk.metadata["runtime_checkpoint"] = {"phase": "secret-disk"}
            else:
                disk.provider_state = provider()
        overlay.side_effect = overlay_conflict
    monkeypatch.setattr(store, "_overlay_checkpoint_unlocked", overlay)
    before = store.get_session_path(session.key).read_bytes()
    with pytest.raises(SessionWriteConflict) as caught:
        store.save_runtime_checkpoint(session)
    assert str(caught.value) == SessionWriteConflict.DEFAULT_MESSAGE
    assert overlay.call_count == (0 if conflict == "base" else 1)
    assert not store.get_runtime_checkpoint_path(session.key).exists()
    assert not list(store.sessions_dir.glob(".*.tmp"))
    assert store.get_session_path(session.key).read_bytes() == before


@pytest.mark.parametrize("fault", ["serialize", "replace"])
def test_checkpoint_failure_does_not_publish_baselines(store, monkeypatch, fault):
    session = seed(store)
    session.metadata["runtime_checkpoint"] = {"phase": "local"}
    session.provider_state = provider()
    before = state(session)
    with monkeypatch.context() as patch:
        patch.setattr(module.json if fault == "serialize" else module.os,
                      "dump" if fault == "serialize" else "replace", Mock(side_effect=OSError("injected")))
        with pytest.raises(OSError, match="injected"):
            store.save_runtime_checkpoint(session)
    assert state(session) == before
    assert not store.get_runtime_checkpoint_path(session.key).exists()
    assert not list(store.sessions_dir.glob(".*.tmp"))


def test_checkpoint_success_publishes_only_checkpoint_and_provider_baselines(store):
    session = seed(store)
    session.add_message("assistant", "pending")
    row = session.messages[-1]
    session._append_ids[(1, id(row))] = (row, "pending-id")
    session._replacement_pending = session._projection_edit_pending = True
    session.metadata["runtime_checkpoint"] = {"phase": ["local"]}
    session.provider_state = provider()
    before = state(session)
    append_ids = session._append_ids.copy()
    main = store.get_session_path(session.key).read_bytes()
    store.save_runtime_checkpoint(session)
    expected = deepcopy(before)
    expected["_base_metadata"]["runtime_checkpoint"] = deepcopy(session.metadata["runtime_checkpoint"])
    expected["_base_provider_state"] = deepcopy(session.provider_state)
    assert state(session) == expected
    assert session._append_ids == append_ids
    assert session._append_ids[(1, id(row))][0] is row
    assert session._base_metadata["runtime_checkpoint"] is not session.metadata["runtime_checkpoint"]
    assert session._base_provider_state is not session.provider_state
    assert store.get_session_path(session.key).read_bytes() == main


@pytest.mark.parametrize("intent", ["replacement", "projection", "append"])
@pytest.mark.parametrize("failure", [False, True])
def test_fresh_cached_upsert_preserves_intent_and_failed_creation_retry(tmp_path, monkeypatch, intent, failure):
    manager = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "private")
    session = manager.get_or_create("test:fresh-intent")
    session.add_message("user", "unsaved")
    session.metadata["local"] = ["draft"]
    session.provider_state = provider()
    session._replacement_pending = intent == "replacement"
    session._projection_edit_pending = intent == "projection"
    row = session.messages[0]
    session._append_ids[(0, id(row))] = (row, "preallocated")
    before = state(session)
    if failure:
        with monkeypatch.context() as patch:
            patch.setattr(manager._jsonl_store, "update_metadata", Mock(side_effect=OSError("injected")))
            with pytest.raises(OSError):
                manager.upsert_session_metadata(session.key, {"title": "first"}, create_if_absent=True)
        assert state(session) == before
        assert session._base_messages is None
        assert manager.read_session_snapshot(session.key).messages == []
    assert manager.upsert_session_metadata(session.key, {"title": "accepted"}, create_if_absent=True)
    assert session.messages == before["messages"]
    assert session.provider_state == before["provider_state"]
    assert session.metadata["local"] == ["draft"]
    assert session.metadata["title"] == "accepted"
    assert session._replacement_pending == before["_replacement_pending"]
    assert session._projection_edit_pending == before["_projection_edit_pending"]
    assert session._append_ids[(0, id(row))] == (row, "preallocated")
    assert session._append_ids[(0, id(row))][0] is row
    assert session._base_messages == []
    assert "local" not in session._base_metadata
    manager.save(session)
    assert manager.read_session_snapshot(session.key).messages[0]["content"] == "unsaved"


@pytest.mark.parametrize("baseline,outcome", [(True, "unused"), (False, "loaded"), (False, "none"), (False, "raise")])
def test_upsert_loader_is_conditional_once_and_failure_publication(tmp_path, monkeypatch, baseline, outcome):
    manager = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "private")
    session = manager.get_or_create("test:loader")
    if baseline:
        manager.save(session)
    else:
        manager._jsonl_store.save(Session(session.key))
    committed = manager._jsonl_store.load(session.key)
    before = state(session)
    load = Mock(return_value=None if outcome == "none" else committed,
                side_effect=OSError("loader injected") if outcome == "raise" else None)
    monkeypatch.setattr(manager._jsonl_store, "load", load)
    if outcome == "raise":
        with pytest.raises(OSError, match="loader injected"):
            manager.upsert_session_metadata(session.key, {"title": "accepted"})
        assert state(session) == before
    else:
        assert manager.upsert_session_metadata(session.key, {"title": "accepted"})
        if outcome == "none":
            assert manager.get_cached(session.key) is None
            assert state(session) == before
        else:
            assert session.metadata["title"] == "accepted"
    assert load.call_args_list == ([] if baseline else [((session.key,), {})])


@pytest.mark.parametrize("upsert", [False, True])
def test_metadata_publication_retains_alias_difference_and_removals(tmp_path, monkeypatch, upsert):
    manager = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "private")
    session = manager.get_or_create("test:copies")
    session.metadata["obsolete"] = "old"
    manager.save(session)
    nested = {"items": ["accepted"]}
    accepted_values = []
    accept = module.accepted_metadata_updates

    def capture_accepted(*args, **kwargs):
        accepted = accept(*args, **kwargs)
        accepted_values.append(accepted)
        return accepted

    monkeypatch.setattr(module, "accepted_metadata_updates", capture_accepted)
    if upsert:
        assert manager.upsert_session_metadata(session.key, {"title": nested}, removals=("obsolete",))
        assert "obsolete" not in session.metadata and "obsolete" not in session._base_metadata
    else:
        assert manager.update_session_metadata(session.key, {"title": nested})
    accepted_nested = accepted_values[-1]["title"]
    assert (session.metadata["title"] is accepted_nested) is (not upsert)
    assert session._base_metadata["title"] is not nested
    assert session._base_metadata["title"] is not session.metadata["title"]
    nested["items"].append("later")
    accepted_nested["items"].append("accepted-later")
    assert session.metadata["title"]["items"] == (["accepted"] if upsert else ["accepted", "accepted-later"])
    assert session._base_metadata["title"] == {"items": ["accepted"]}
    assert manager.read_session_metadata(session.key)["metadata"]["title"] == {"items": ["accepted"]}


def test_merge_removal_preserves_concurrent_change_and_ring_order(store):
    session = seed(store)
    session.metadata["obsolete"] = "old"
    store.save(session)
    stale = store.load(session.key)
    store.update_metadata(session.key, {"obsolete": "new"})
    stale.metadata.pop("obsolete")
    stale.metadata[Session.SCHEDULED_RING_KEY] = [{"run_id": "a"}, {"run_id": "b"}]
    store.save(stale)
    assert stale.metadata["obsolete"] == "new"
    assert stale.recent_scheduled_runs() == [{"run_id": "a"}, {"run_id": "b"}]
    assert json.loads(store.get_session_path(session.key).read_text().splitlines()[0])["metadata"]["obsolete"] == "new"


@pytest.mark.parametrize("checkpoint", [False, True])
def test_source_schema_refusal_never_writes_or_overlays_checkpoint(store, monkeypatch, checkpoint):
    session = seed(store)
    disk = deepcopy(session)
    disk.metadata["source_schema_version"] = 1
    monkeypatch.setattr(store, "_load_unlocked", Mock(return_value=disk))
    overlay = Mock()
    monkeypatch.setattr(store, "_overlay_checkpoint_unlocked", overlay)
    before = store.get_session_path(session.key).read_bytes()
    with pytest.raises(SessionWriteConflict) as caught:
        (store.save_runtime_checkpoint if checkpoint else store.save)(session)
    assert str(caught.value) == SessionWriteConflict.DEFAULT_MESSAGE
    assert overlay.call_count == (0 if checkpoint else 1)
    assert store.get_session_path(session.key).read_bytes() == before
    assert not store.get_runtime_checkpoint_path(session.key).exists()


def test_provider_handle_stays_private_after_commit_and_checkpoint(tmp_path):
    manager = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "private")
    session = manager.get_or_create("test:privacy")
    session.add_message("user", "public")
    session.provider_state = provider()
    manager.save(session)
    session.metadata["runtime_checkpoint"] = {"phase": "awaiting_tools"}
    manager.save_runtime_checkpoint(session)
    assert "private-handle" not in repr(session)
    assert "private-handle" not in str(manager.read_session_file(session.key))
    assert "private-handle" not in str(manager.read_session_metadata(session.key))
    assert "private-handle" not in str(manager.list_sessions())
