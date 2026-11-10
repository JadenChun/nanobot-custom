"""Regression test for A3-3 AC3.1: delegated admission rejects stale parent."""

from __future__ import annotations

import pytest

from nanobot.agent.child_sessions import ChildAdmission, DelegationTree
from nanobot.session.manager import Session, SessionManager
from nanobot.session.runtime_authority import read_runtime_session


@pytest.mark.asyncio
async def test_delegation_rejects_stale_parent(tmp_path, monkeypatch):
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)

    manager = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    parent_key = "websocket:chat-a"

    manager.save(Session(key=parent_key))
    old_parent = manager.read_session_snapshot(parent_key)
    assert old_parent is not None

    tree = DelegationTree(manager)
    old = await tree.reserve(
        parent=old_parent,
        parent_key=parent_key,
        assignment="same assignment",
        attempt_id="attempt-1",
    )
    assert isinstance(old, ChildAdmission)

    child_session = manager.get_or_create(old.session_key)
    child_session.add_message("user", "old child history")
    manager.save(child_session)

    manager.delete_session(parent_key)
    manager.save(Session(key=parent_key))
    new_parent = manager.read_session_snapshot(parent_key)
    assert new_parent is not None
    assert new_parent.incarnation != old_parent.incarnation

    old_state = read_runtime_session(manager, old.session_key)
    assert old_state.executable is False
    assert old_state.issue == "stale_parent"

    snapshot = manager.read_session_snapshot(old.session_key)
    assert snapshot is not None
    assert any("old child history" in str(m.get("content", "")) for m in snapshot.messages)

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
