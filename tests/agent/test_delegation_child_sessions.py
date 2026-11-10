"""Regression tests for durable delegated child sessions (A3-5 AC5.1)."""

import asyncio

import pytest

from nanobot.agent.child_sessions import (
    CHILD_SESSION_PREFIX,
    DEFAULT_CHILD_PROFILE,
    DELEGATION_CHILD_METADATA_KEY,
    SUPPORTED_CHILD_PROFILES,
    ChildAdmission,
    ChildRefusal,
    DelegationTree,
    DelegationTreePolicy,
)
from nanobot.session.manager import SessionManager


def _new_tree(tmp_path, *, policy=None):
    sessions = SessionManager(
        tmp_path / "workspace",
        sessions_root=tmp_path / "sessions",
    )
    return sessions, DelegationTree(sessions, policy=policy or DelegationTreePolicy())


def _new_parent(sessions, key="websocket:parent"):
    parent = sessions.get_or_create(key)
    sessions.save(parent)
    return parent


def _reserve(tree, parent, *, assignment, attempt_id, profile=DEFAULT_CHILD_PROFILE):
    return tree.reserve(
        parent=parent,
        parent_key=parent.key,
        assignment=assignment,
        attempt_id=attempt_id,
        profile=profile,
        policy_facts={"scope": assignment},
        runtime_generation=0,
    )


@pytest.mark.asyncio
async def test_siblings_have_distinct_identities_and_grandchild_is_admitted(
    tmp_path,
):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)

    first = await _reserve(
        tree,
        parent,
        assignment="first sibling assignment",
        attempt_id="attempt-first",
    )
    second = await _reserve(
        tree,
        parent,
        assignment="second sibling assignment",
        attempt_id="attempt-second",
    )
    assert isinstance(first, ChildAdmission)
    assert isinstance(second, ChildAdmission)

    first_session_key = tree.child(first.child_id).session_key
    assert first_session_key == first.session_key

    assert first.child_id != second.child_id
    assert first.session_key != second.session_key
    assert all(
        admission.session_key.startswith(CHILD_SESSION_PREFIX)
        for admission in (first, second)
    )

    assert first.depth == 1
    assert second.depth == 1
    assert first.parent_key == parent.key
    assert second.parent_key == parent.key
    assert first.parent_incarnation == second.parent_incarnation
    assert first.root_key == second.root_key == parent.key

    for admission in (first, second):
        child_session = sessions.get_or_create(admission.session_key)
        runtime = child_session.metadata["_runtime_session"]
        assert runtime["profile"] == "delegated"
        assert runtime["parent_key"] == admission.parent_key
        assert runtime["parent_incarnation"] == admission.parent_incarnation
        assert runtime["delegation_depth"] == admission.depth
        assert DELEGATION_CHILD_METADATA_KEY in child_session.metadata

    first_as_parent = sessions.get_or_create(first_session_key)
    grandchild = await _reserve(
        tree,
        first_as_parent,
        assignment="grandchild assignment",
        attempt_id="attempt-grandchild",
    )
    assert isinstance(grandchild, ChildAdmission)
    assert grandchild.depth == 2
    assert grandchild.parent_key == first.session_key
    assert grandchild.parent_incarnation == first_as_parent.incarnation
    assert {child.child_id for child in tree.children_of(first.session_key)} == {
        grandchild.child_id,
    }
    grandchild_session = sessions.get_or_create(grandchild.session_key)
    assert grandchild_session.metadata["_runtime_session"]["delegation_depth"] == 2

    great_grandchild = await _reserve(
        tree,
        grandchild_session,
        assignment="great-grandchild assignment",
        attempt_id="attempt-great-grandchild",
    )
    assert isinstance(great_grandchild, ChildRefusal)
    assert great_grandchild.reason == "depth_limit"
    assert great_grandchild.detail
    assert list(tree.children_of(grandchild.session_key)) == []
    assert {child.child_id for child in tree.children_of(parent.key)} == {
        first.child_id,
        second.child_id,
    }


@pytest.mark.asyncio
async def test_repeated_attempt_returns_same_child(tmp_path):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)

    first = await _reserve(
        tree,
        parent,
        assignment="review the budget",
        attempt_id="attempt-repeat",
    )
    assert isinstance(first, ChildAdmission)

    repeated = await _reserve(
        tree,
        parent,
        assignment="review the budget",
        attempt_id="attempt-repeat",
    )
    assert isinstance(repeated, ChildAdmission)
    assert repeated.child_id == first.child_id
    assert repeated.session_key == first.session_key

    changed = await _reserve(
        tree,
        parent,
        assignment="review the schedule",
        attempt_id="attempt-repeat",
    )
    assert isinstance(changed, ChildAdmission)
    assert changed.child_id != first.child_id


@pytest.mark.asyncio
async def test_depth_and_capacity_refusals_leave_no_live_child(tmp_path):
    sessions, tree = _new_tree(
        tmp_path,
        policy=DelegationTreePolicy(
            max_depth=2,
            max_live_children_per_parent=2,
            max_live_descendants=4,
        ),
    )
    parent = _new_parent(sessions)

    first = await _reserve(
        tree,
        parent,
        assignment="capacity child one",
        attempt_id="attempt-capacity-one",
    )
    second = await _reserve(
        tree,
        parent,
        assignment="capacity child two",
        attempt_id="attempt-capacity-two",
    )
    refused = await _reserve(
        tree,
        parent,
        assignment="capacity child three",
        attempt_id="attempt-capacity-three",
    )
    assert isinstance(first, ChildAdmission)
    assert isinstance(second, ChildAdmission)
    assert isinstance(refused, ChildRefusal)
    assert refused.reason == "child_limit"
    assert refused.detail

    live = [child for child in tree.children_of(parent.key) if child.live]
    assert len(live) == 2
    assert {child.child_id for child in live} == {first.child_id, second.child_id}
    live_descendants = list(tree.live_descendants(parent.key))
    assert len(live_descendants) == 2
    assert {child.child_id for child in live_descendants} == {
        first.child_id,
        second.child_id,
    }

    deep_sessions, deep_tree = _new_tree(
        tmp_path / "deep",
        policy=DelegationTreePolicy(max_depth=1),
    )
    deep_parent = _new_parent(deep_sessions, key="websocket:deep-parent")

    shallow = await _reserve(
        deep_tree,
        deep_parent,
        assignment="shallow child",
        attempt_id="attempt-shallow",
    )
    assert isinstance(shallow, ChildAdmission)

    shallow_session_key = deep_tree.child(shallow.child_id).session_key
    shallow_as_parent = deep_sessions.get_or_create(shallow_session_key)
    too_deep = await _reserve(
        deep_tree,
        shallow_as_parent,
        assignment="too deep child",
        attempt_id="attempt-too-deep",
    )
    assert isinstance(too_deep, ChildRefusal)
    assert too_deep.reason == "depth_limit"
    assert too_deep.detail

    deep_live = [
        child for child in deep_tree.children_of(deep_parent.key) if child.live
    ]
    assert [child.child_id for child in deep_live] == [shallow.child_id]
    assert {
        child.child_id for child in deep_tree.live_descendants(deep_parent.key)
    } == {shallow.child_id}


@pytest.mark.asyncio
async def test_prelaunch_release_and_uncertain_retention(tmp_path):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)

    released_child = await _reserve(
        tree,
        parent,
        assignment="release before launch",
        attempt_id="attempt-release",
    )
    assert isinstance(released_child, ChildAdmission)

    await tree.release(released_child.child_id, reason="prelaunch_failed")
    released = tree.child(released_child.child_id)
    assert released is not None
    assert released.state == "refused"
    assert released.live is False
    assert list(tree.live_descendants(parent.key)) == []

    uncertain_child = await _reserve(
        tree,
        parent,
        assignment="uncertain terminal work",
        attempt_id="attempt-uncertain",
    )
    assert isinstance(uncertain_child, ChildAdmission)

    await tree.complete(
        uncertain_child.child_id,
        outcome="ambiguous",
        state="uncertain",
    )
    uncertain = tree.child(uncertain_child.child_id)
    assert uncertain is not None
    assert uncertain.state == "uncertain"
    assert uncertain.outcome == "ambiguous"
    assert uncertain.terminal is True
    assert uncertain.live is False
    assert uncertain_child.child_id in {
        child.child_id for child in tree.children_of(parent.key)
    }

    persisted = sessions.get_or_create(uncertain_child.session_key)
    assert persisted.metadata["_runtime_session"]["profile"] == "delegated"
    assert DELEGATION_CHILD_METADATA_KEY in persisted.metadata
    assert list(tree.live_descendants(parent.key)) == []


@pytest.mark.asyncio
async def test_stop_subtree_cancels_descendants_and_keeps_sibling(tmp_path):
    # The default policy is depth one; opt into a deeper tree to exercise
    # descendant cancellation without changing the production default.
    sessions, tree = _new_tree(
        tmp_path,
        policy=DelegationTreePolicy(max_depth=2),
    )
    parent = _new_parent(sessions)

    sibling_a = await _reserve(
        tree,
        parent,
        assignment="sibling A",
        attempt_id="attempt-sibling-a",
    )
    sibling_b = await _reserve(
        tree,
        parent,
        assignment="sibling B",
        attempt_id="attempt-sibling-b",
    )
    assert isinstance(sibling_a, ChildAdmission)
    assert isinstance(sibling_b, ChildAdmission)

    sibling_a_session_key = tree.child(sibling_a.child_id).session_key
    sibling_a_as_parent = sessions.get_or_create(sibling_a_session_key)
    nested = await _reserve(
        tree,
        sibling_a_as_parent,
        assignment="nested under A",
        attempt_id="attempt-nested-under-a",
    )
    assert isinstance(nested, ChildAdmission)

    await tree.stop_subtree(sibling_a.child_id)

    stopped_a = tree.child(sibling_a.child_id)
    stopped_nested = tree.child(nested.child_id)
    live_b = tree.child(sibling_b.child_id)
    assert stopped_a is not None
    assert stopped_nested is not None
    assert live_b is not None
    assert stopped_a.state == "stopped"
    assert stopped_nested.state == "stopped"
    assert stopped_a.live is False
    assert stopped_nested.live is False
    assert live_b.state != "stopped"
    assert live_b.live is True
    remaining = list(tree.live_descendants(parent.key))
    assert len(remaining) == 1
    assert remaining[0].child_id == sibling_b.child_id


@pytest.mark.asyncio
async def test_unsupported_role_profile_is_refused(tmp_path):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)

    assert DEFAULT_CHILD_PROFILE == "delegated"
    assert SUPPORTED_CHILD_PROFILES == frozenset({"delegated"})

    for profile in ("crawler", "planner"):
        assert profile not in SUPPORTED_CHILD_PROFILES
        refusal = await _reserve(
            tree,
            parent,
            assignment=f"{profile} assignment",
            attempt_id=f"attempt-{profile}",
            profile=profile,
        )
        assert isinstance(refusal, ChildRefusal)
        assert refusal.reason == "unsupported_profile"
        assert refusal.detail

    assert list(tree.children_of(parent.key)) == []
    assert list(tree.live_descendants(parent.key)) == []


@pytest.mark.asyncio
async def test_restart_fences_live_child_and_restores_attempt(tmp_path):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)

    child = await _reserve(
        tree,
        parent,
        assignment="survive a restart",
        attempt_id="attempt-restart",
    )
    assert isinstance(child, ChildAdmission)
    activated = await tree.activate(child.child_id)
    assert activated is not None
    assert activated.state == "active"

    restored_sessions, restored = _new_tree(tmp_path)
    restored_parent = restored_sessions.get_or_create(parent.key)

    restored_child = restored.child(child.child_id)
    assert restored_child is not None
    assert restored_child.state == "uncertain"
    assert restored_child.outcome == "restart_unresolved"

    repeated = await _reserve(
        restored,
        restored_parent,
        assignment="survive a restart",
        attempt_id="attempt-restart",
    )
    assert isinstance(repeated, ChildAdmission)
    assert repeated.child_id == child.child_id
    assert repeated.session_key == child.session_key


@pytest.mark.asyncio
async def test_stop_for_parent_stops_subtree_and_leaves_other_parents(tmp_path):
    sessions, tree = _new_tree(tmp_path)
    parent_a = _new_parent(sessions, key="websocket:parent-a")
    parent_b = _new_parent(sessions, key="websocket:parent-b")

    child_a1 = await _reserve(
        tree,
        parent_a,
        assignment="parent A child one",
        attempt_id="attempt-a-one",
    )
    child_a2 = await _reserve(
        tree,
        parent_a,
        assignment="parent A child two",
        attempt_id="attempt-a-two",
    )
    child_b1 = await _reserve(
        tree,
        parent_b,
        assignment="parent B child one",
        attempt_id="attempt-b-one",
    )
    assert isinstance(child_a1, ChildAdmission)
    assert isinstance(child_a2, ChildAdmission)
    assert isinstance(child_b1, ChildAdmission)

    await tree.stop_for_parent(parent_a.key)

    for stopped_child_id in (child_a1.child_id, child_a2.child_id):
        record = tree.child(stopped_child_id)
        assert record is not None
        assert record.state == "stopped"
        assert record.live is False

    survivor = tree.child(child_b1.child_id)
    assert survivor is not None
    assert survivor.state != "stopped"
    assert survivor.live is True

    assert list(tree.live_descendants(parent_a.key)) == []
    assert {record.child_id for record in tree.live_descendants(parent_b.key)} == {
        child_b1.child_id
    }


@pytest.mark.asyncio
async def test_stop_subtree_cancels_attached_task_and_keeps_sibling(tmp_path):
    sessions, tree = _new_tree(tmp_path)
    parent = _new_parent(sessions)

    sibling_a = await _reserve(
        tree,
        parent,
        assignment="attached task sibling A",
        attempt_id="attempt-attached-a",
    )
    sibling_b = await _reserve(
        tree,
        parent,
        assignment="attached task sibling B",
        attempt_id="attempt-attached-b",
    )
    assert isinstance(sibling_a, ChildAdmission)
    assert isinstance(sibling_b, ChildAdmission)

    task = asyncio.create_task(asyncio.sleep(60))
    await tree.attach_task(sibling_a.child_id, task)

    await tree.stop_subtree(sibling_a.child_id)

    # stop_subtree requests cancellation; yield so the task observes it.
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()

    stopped_a = tree.child(sibling_a.child_id)
    live_b = tree.child(sibling_b.child_id)
    assert stopped_a is not None
    assert live_b is not None
    assert stopped_a.state == "stopped"
    assert stopped_a.live is False
    assert live_b.state != "stopped"
    assert live_b.live is True

    if not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_or_refused_child_releases_capacity(tmp_path):
    sessions, tree = _new_tree(
        tmp_path,
        policy=DelegationTreePolicy(max_live_children_per_parent=1),
    )
    parent = _new_parent(sessions)

    refused = await _reserve(
        tree,
        parent,
        assignment="unsupported profile assignment",
        attempt_id="attempt-unsupported",
        profile="crawler",
    )
    assert isinstance(refused, ChildRefusal)
    assert refused.reason == "unsupported_profile"
    assert list(tree.children_of(parent.key)) == []

    first = await _reserve(
        tree,
        parent,
        assignment="fails and frees the slot",
        attempt_id="attempt-fails",
    )
    assert isinstance(first, ChildAdmission)

    await tree.complete(first.child_id, outcome="failed", state="failed")
    failed = tree.child(first.child_id)
    assert failed is not None
    assert failed.state == "failed"
    assert failed.live is False

    second = await _reserve(
        tree,
        parent,
        assignment="fits after the failure",
        attempt_id="attempt-after-failure",
    )
    assert isinstance(second, ChildAdmission)
    assert second.child_id != first.child_id
    live = [record for record in tree.children_of(parent.key) if record.live]
    assert [record.child_id for record in live] == [second.child_id]
