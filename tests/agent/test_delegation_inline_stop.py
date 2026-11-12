"""Regression coverage for inline child stop and control-generation checks.

These cases now drive the canonical DelegationTree and child-control path. The
retired inline runner path (``ForegroundAgentManager.runner``) no longer exists,
so each case reserves a source-backed child and attaches a real asyncio task to
observe cancellation instead of injecting a fake runner.
"""

import asyncio

from tests.agent.test_delegation_child_controls import _live_child, _manager, _parent


async def test_foreground_stop_cancels_the_runner_and_settles_the_child(tmp_path):
    manager = _manager(tmp_path)
    parent = _parent(manager)
    tree = manager._delegation_tree

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def worker():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    admission = await _live_child(
        manager,
        parent,
        assignment="Do the bounded work with acceptance evidence and validation.",
        attempt_id="turn-1",
    )
    task = asyncio.create_task(worker())
    await tree.attach_task(admission.child_id, task)
    await started.wait()

    stopped = await manager.control_child(
        parent_key=parent.key,
        child_id=admission.child_id,
        action="stop",
    )

    assert stopped["ok"] is True
    assert stopped["state"] == "stopped"
    assert cancelled.is_set()
    assert task.done()

    final = tree.child(admission.child_id)
    assert final is not None
    assert final.state == "stopped"


async def test_stopping_one_sibling_leaves_the_other_running(tmp_path):
    manager = _manager(tmp_path)
    parent = _parent(manager)
    tree = manager._delegation_tree

    child_a = await _live_child(
        manager,
        parent,
        assignment="first sibling contract with acceptance evidence.",
        attempt_id="turn-a",
    )
    child_b = await _live_child(
        manager,
        parent,
        assignment="second sibling contract with acceptance evidence.",
        attempt_id="turn-b",
    )

    a_started = asyncio.Event()
    a_cancelled = asyncio.Event()
    b_started = asyncio.Event()
    b_release = asyncio.Event()
    b_finished = asyncio.Event()

    async def worker_a():
        a_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            a_cancelled.set()
            raise

    async def worker_b():
        b_started.set()
        await b_release.wait()
        b_finished.set()

    task_a = asyncio.create_task(worker_a())
    task_b = asyncio.create_task(worker_b())
    await tree.attach_task(child_a.child_id, task_a)
    await tree.attach_task(child_b.child_id, task_b)
    await a_started.wait()
    await b_started.wait()

    stopped = await manager.control_child(
        parent_key=parent.key,
        child_id=child_a.child_id,
        action="stop",
    )

    assert stopped["ok"] is True
    assert a_cancelled.is_set()
    assert task_a.done()
    assert tree.child(child_a.child_id).state == "stopped"
    assert not task_b.cancelled()
    assert tree.child(child_b.child_id).state == "active"

    b_release.set()
    await b_finished.wait()
    await tree.complete(child_b.child_id, outcome="completed")
    assert tree.child(child_b.child_id).state == "completed"


async def test_control_generation_is_checked(tmp_path):
    # The runtime-generation guard is not asserted by the canonical control
    # suite, so this case is preserved here.
    manager = _manager(tmp_path)
    parent = _parent(manager)

    admission = await _live_child(
        manager,
        parent,
        assignment="Do the bounded work with acceptance evidence and validation.",
        attempt_id="turn-1",
    )
    assert admission.runtime_generation == 0

    stale = await manager.control_child(
        parent_key=parent.key,
        child_id=admission.child_id,
        action="wait",
        runtime_generation=99,
    )
    assert stale["ok"] is False
    assert stale["reason"] == "stale_generation"

    matching = await manager.control_child(
        parent_key=parent.key,
        child_id=admission.child_id,
        action="wait",
        runtime_generation=admission.runtime_generation,
    )
    assert matching["ok"] is True
    assert matching["state"] == "active"
