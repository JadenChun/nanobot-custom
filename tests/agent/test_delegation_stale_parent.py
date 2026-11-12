"""Regression test for A3-3 AC3.1: delegated admission rejects stale parent."""

from __future__ import annotations

import pytest

from nanobot.agent.child_sessions import ChildAdmission, DelegationTree
from nanobot.session.manager import SessionManager
from nanobot.session.runtime_authority import read_runtime_session


def _source_parent(manager: SessionManager, key: str):
    parent = manager.get_or_create(key)
    manager.save(parent)
    backend = manager._jsonl_store
    backend.source.upgrade_source_session(
        key,
        expected_revision=parent.revision,
        expected_generation=parent.generation,
        expected_incarnation=parent.incarnation,
    )
    manager.invalidate(key)
    return manager.get_or_create(key)


@pytest.mark.asyncio
async def test_delegation_rejects_stale_parent(tmp_path, monkeypatch):
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)

    manager = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    parent_key = "websocket:chat-a"

    old_parent = _source_parent(manager, parent_key)
    assert old_parent is not None

    tree = DelegationTree(manager)
    old = await tree.reserve(
        parent=old_parent,
        parent_key=parent_key,
        assignment="same assignment",
        attempt_id="attempt-1",
    )
    assert isinstance(old, ChildAdmission)

    # Canonical child history lives in the child source snapshot, not in
    # legacy Session.messages. Record one exchange so the data-preservation
    # check below observes the durable child source.
    assert tree.record_exchange(
        old.child_id,
        assignment="same assignment",
        result_text="old child history",
    ) is True

    manager.delete_session(parent_key)
    new_parent = _source_parent(manager, parent_key)
    assert new_parent is not None
    assert new_parent.incarnation != old_parent.incarnation

    old_state = read_runtime_session(manager, old.session_key)
    assert old_state.executable is False
    assert old_state.issue == "stale_parent"

    snapshot = manager.read_session_snapshot(old.session_key)
    assert snapshot is not None
    child_source = manager.source_backend(old.session_key).source.load_source_snapshot(
        old.session_key,
    )
    assert any(
        "old child history" in str(event.get("content", ""))
        for event in child_source.events
    )

    children = tree.children_of(parent_key, parent_incarnation=new_parent.incarnation)
    assert old.child_id not in {child.child_id for child in children}

    stopped = await tree.stop_for_parent(
        parent_key, parent_incarnation=new_parent.incarnation
    )
    assert old.child_id not in stopped

    old_child = tree.child(old.child_id)
    assert old_child is not None
    assert old_child.live is True

    new = await tree.reserve(
        parent=new_parent,
        parent_key=parent_key,
        assignment="same assignment",
        attempt_id="attempt-1",
    )
    assert isinstance(new, ChildAdmission)
    assert new.child_id != old.child_id
    assert new.parent_incarnation == new_parent.incarnation

    await tree.stop_for_parent(parent_key, parent_incarnation=new_parent.incarnation)
