"""A malformed child record must fence one chat, not every chat's Stop.

Reproduces the observed staging shape: one completed delegated child whose
durable record lost its source-state binding. Child admission stays fenced, but
an unrelated chat keeps a real Stop target.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from nanobot.agent.child_sessions import DELEGATION_CHILD_METADATA_KEY
from tests.agent.test_delegation_child_sessions import (
    _new_parent,
    _new_tree,
    _reserve,
)


@pytest.mark.asyncio
async def test_malformed_child_fences_only_its_own_root(tmp_path) -> None:
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)
    child = await _reserve(
        tree,
        parent,
        assignment="report one heading",
        attempt_id="attempt-attribution",
    )
    unrelated = sessions.get_or_create("websocket:unrelated-chat")
    sessions.save(unrelated)

    corrupted = child.metadata()
    corrupted["source_state"] = None
    corrupted["parent_source_state"] = None
    corrupted["root_source_state"] = None
    assert sessions.upsert_session_metadata(
        child.session_key,
        {DELEGATION_CHILD_METADATA_KEY: corrupted},
    )

    restored_sessions, restored = _new_tree(tmp_path)

    assert restored.recovery_error is not None
    # Child admission stays fenced across the tree.
    parent_now = restored_sessions.get_or_create(parent.key)
    refused = await _reserve(
        restored,
        parent_now,
        assignment="must wait",
        attempt_id="attempt-after-corruption",
    )
    assert not hasattr(refused, "child_id")

    # The failure is attributed to the chat that owns the record, so another
    # chat is not fenced by it.
    assert restored.recovery_error_child_keys() == frozenset({parent.key})
