"""Execution lock lifetime, persistent fences, and crash recovery requirements."""

import asyncio
import pickle
from dataclasses import replace

import pytest

from nanobot.session.execution_owner import (
    ExecutionBusyError,
    ExecutionOwnershipError,
    acquire_execution,
)
from nanobot.session.manager import JsonlSessionStore, Session, SessionWriteConflict
from nanobot.session.source import new_source_event


@pytest.fixture
def backend(tmp_path):
    return JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "private")


def seed(backend, key="test:owned", *, messages=None):
    session = Session(key, messages=messages or [{"role": "user", "content": "input"}])
    backend.save(session)
    return backend.source.upgrade_source_session(
        key, expected_revision=session.revision, expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )


async def test_claim_release_preserve_source_activity_and_fence_old_capability(backend):
    original = seed(backend)
    async with acquire_execution(backend, original.state.session_key, run_id="run") as owner:
        snapshot = owner.read_snapshot()
        owner.validate_dispatch()
        assert snapshot.state.revision == original.state.revision + 1
        assert snapshot.events == original.events
        assert snapshot.header["updated_at"] == original.header["updated_at"]
        assert snapshot.header["metadata"]["execution_lease"]["run_id"] == "run"
        with pytest.raises(TypeError):
            pickle.dumps(owner)
    loaded = backend.source.load_source_snapshot(original.state.session_key)
    assert "execution_lease" not in loaded.header["metadata"]
    assert loaded.state.revision == original.state.revision + 2
    with pytest.raises(ExecutionOwnershipError):
        owner.validate_dispatch()


async def test_same_process_waiter_cannot_reenter_and_cancel_never_acquires_later(backend):
    original = seed(backend)
    acquired = []
    async def waiter():
        async with acquire_execution(backend, original.state.session_key, run_id="waiter"):
            acquired.append(True)
    async with acquire_execution(backend, original.state.session_key, run_id="owner"):
        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.08)
        assert not task.done() and not acquired
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    await asyncio.sleep(0.08)
    assert not acquired
    async with acquire_execution(backend, original.state.session_key, run_id="later") as owner:
        owner.validate_dispatch()


async def test_different_sessions_can_be_owned_together(backend):
    first, second = seed(backend), seed(backend, "test:other")
    async with acquire_execution(backend, first.state.session_key, run_id="one") as one:
        async with acquire_execution(backend, second.state.session_key, run_id="two") as two:
            one.validate_dispatch()
            two.validate_dispatch()


async def test_offline_mutations_reset_and_public_delete_cannot_bypass_owner(backend):
    original = seed(backend)
    key = original.state.session_key
    async with acquire_execution(backend, key, run_id="run") as owner:
        snapshot = owner.read_snapshot()
        owner.validate_dispatch()  # OS lock protects the interval after this check.
        for action in (
            lambda: backend.source.append_source_events(key, snapshot.state, []),
            lambda: backend.source.reset_source_session(key, snapshot.state, []),
            lambda: backend.source.delete_source_session(key, snapshot.state),
            lambda: backend.delete(key),
        ):
            with pytest.raises(ExecutionBusyError):
                action()
        assert backend.source.load_source_snapshot(key).events == original.events


async def test_owned_writes_require_cas_and_metadata_cannot_replace_lease(backend):
    original = seed(backend)
    key = original.state.session_key
    async with acquire_execution(backend, key, run_id="run") as owner:
        snapshot = owner.read_snapshot()
        lease = snapshot.header["metadata"]["execution_lease"]
        backend.update_metadata(key, {"title": "updated", "execution_lease": None})
        with pytest.raises(SessionWriteConflict):
            backend.source.append_source_events(key, snapshot.state, [], owner=owner)
        current = owner.read_snapshot()
        assert current.header["metadata"]["execution_lease"] == lease
        appended = backend.source.append_source_events(key, current.state, [
            new_source_event({"role": "assistant", "content": "result"}, run_id="run"),
        ], owner=owner)
        assert len(appended.events) == 2
        owner.validate_dispatch()


async def test_reset_requires_root_quiescence_and_invalidates_owner(backend):
    original = seed(backend)
    key = original.state.session_key
    async with acquire_execution(backend, key, run_id="run") as owner:
        snapshot = owner.read_snapshot()
        with pytest.raises(ExecutionOwnershipError):
            backend.source.reset_source_session(key, snapshot.state, [], owner=owner)
        await owner.quiesce()
        reset = backend.source.reset_source_session(key, snapshot.state, [], owner=owner)
        assert reset.state.generation == snapshot.state.generation + 1
        assert "execution_lease" not in reset.header["metadata"]
        with pytest.raises(ExecutionOwnershipError):
            owner.validate_dispatch()
    async with acquire_execution(backend, key, run_id="new") as owner:
        owner.validate_dispatch()


async def test_child_shares_capability_but_cannot_release_or_reset_it(backend):
    original = seed(backend)
    async with acquire_execution(backend, original.state.session_key, run_id="run") as owner:
        async def child():
            owner.validate_dispatch()
            with pytest.raises(ExecutionOwnershipError):
                await owner.aclose()
            with pytest.raises(ExecutionOwnershipError):
                await owner.quiesce()
            snapshot = owner.read_snapshot()
            backend.source.append_source_events(owner.key, snapshot.state, [
                new_source_event({"role": "assistant", "content": "child"}, run_id="child"),
            ], owner=owner)
        task = asyncio.create_task(child())
        owner.track_task(task)
        await task
        owner.validate_dispatch()


async def test_cleanup_holds_exclusivity_until_registered_effect_finishes(backend):
    original = seed(backend)
    effect_started, finish_effect, leaving, released = (asyncio.Event() for _ in range(4))
    async def effect():
        effect_started.set()
        await finish_effect.wait()
    async def run():
        async with acquire_execution(backend, original.state.session_key, run_id="run") as owner:
            owner.track_task(asyncio.create_task(effect()))
            await effect_started.wait()
            leaving.set()
        released.set()
    root = asyncio.create_task(run())
    await leaving.wait()
    await asyncio.sleep(0)
    assert not released.is_set()
    with pytest.raises(ExecutionBusyError):
        backend.delete(original.state.session_key)
    finish_effect.set()
    await root
    assert released.is_set()


async def test_unresolved_intent_requires_recovery_across_clean_releases(backend):
    original = seed(backend, messages=[
        {"role": "assistant", "tool_calls": [{"id": "unknown-outcome"}]},
    ])
    for _ in range(2):
        async with acquire_execution(backend, original.state.session_key, run_id="recovery") as owner:
            assert owner.recovery_required and owner.unresolved_executions
            with pytest.raises(ExecutionOwnershipError):
                owner.validate_dispatch()
    snapshot = backend.source.load_source_snapshot(original.state.session_key)
    assert snapshot.header["metadata"]["execution_recovery_required"] is True


async def test_wrong_store_or_state_cannot_borrow_owner(backend, tmp_path):
    original = seed(backend)
    other = JsonlSessionStore(tmp_path / "other", sessions_root=tmp_path / "private")
    async with acquire_execution(backend, original.state.session_key, run_id="run") as owner:
        snapshot = owner.read_snapshot()
        other.get_session_path(owner.key).write_bytes(backend.get_session_path(owner.key).read_bytes())
        with pytest.raises(ExecutionOwnershipError):
            other.source.append_source_events(owner.key, snapshot.state, [], owner=owner)
        with pytest.raises(SessionWriteConflict):
            backend.source.append_source_events(owner.key, replace(snapshot.state, generation=999), [], owner=owner)


async def test_source_delete_retains_lock_inode_and_fences_recreation(backend):
    original = seed(backend)
    key = original.state.session_key
    async with acquire_execution(backend, key, run_id="run") as owner:
        lock_path = backend.sessions_dir / ".execution-locks" / f"{backend.storage_key(key)}.lock"
        inode = lock_path.stat().st_ino
        snapshot = owner.read_snapshot()
        await owner.quiesce()
        backend.source.delete_source_session(key, snapshot.state, owner=owner)
        assert lock_path.exists() and lock_path.stat().st_ino == inode
    fresh = seed(backend)
    assert fresh.state.incarnation != original.state.incarnation
    async with acquire_execution(backend, key, run_id="new"):
        assert lock_path.stat().st_ino == inode
    with pytest.raises(ExecutionOwnershipError):
        owner.validate_dispatch()


def _crash_owner(workspace, private, key, ready, exit_now):
    import os
    async def run():
        backend = JsonlSessionStore(workspace, sessions_root=private)
        async with acquire_execution(backend, key, run_id="crashed"):
            ready.put("owned")
            while not exit_now.is_set():
                await asyncio.sleep(0.01)
            os._exit(0)
    asyncio.run(run())


async def test_process_death_releases_os_lock_but_requires_recovery(backend, tmp_path):
    import multiprocessing
    original = seed(backend)
    context = multiprocessing.get_context("spawn")
    ready, exit_now = context.Queue(), context.Event()
    process = context.Process(target=_crash_owner, args=(
        tmp_path / "workspace", tmp_path / "private", original.state.session_key, ready, exit_now,
    ))
    try:
        process.start()
        assert await asyncio.to_thread(ready.get, timeout=10) == "owned"
        with pytest.raises(ExecutionBusyError):
            backend.delete(original.state.session_key)
        exit_now.set()
        await asyncio.to_thread(process.join, 10)
        assert process.exitcode == 0
        with pytest.raises(ExecutionOwnershipError):
            backend.source.append_source_events(original.state.session_key,
                                                backend.source.load_source_snapshot(original.state.session_key).state, [])
        async with acquire_execution(backend, original.state.session_key, run_id="after-crash") as owner:
            assert owner.recovery_required
            with pytest.raises(ExecutionOwnershipError):
                owner.validate_dispatch()
        async with acquire_execution(backend, original.state.session_key, run_id="still-unreconciled") as owner:
            assert owner.recovery_required
    finally:
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 5)
        ready.close()


@pytest.mark.parametrize("phase", ["before", "after"])
async def test_failed_acquisition_cannot_dispatch_and_leaves_recoverable_state(backend, monkeypatch, phase):
    original = seed(backend)
    entered = False
    with monkeypatch.context() as patch:
        if phase == "before":
            patch.setattr("nanobot.session.manager.os.replace", lambda *_: (_ for _ in ()).throw(OSError("replace")))
        else:
            patch.setattr(type(backend), "_fsync_directory", lambda *_: (_ for _ in ()).throw(OSError("fsync")))
        with pytest.raises(OSError):
            async with acquire_execution(backend, original.state.session_key, run_id="failed"):
                entered = True
    assert not entered
    async with acquire_execution(backend, original.state.session_key, run_id="next") as owner:
        assert owner.recovery_required is (phase == "after")


@pytest.mark.parametrize("phase", ["before", "after"])
async def test_failed_release_retains_abandoned_lease_and_invalidates_capability(backend, monkeypatch, phase):
    original = seed(backend)
    with pytest.raises(OSError):
        async with acquire_execution(backend, original.state.session_key, run_id="failed-close") as owner:
            if phase == "before":
                monkeypatch.setattr("nanobot.session.manager.os.replace", lambda *_: (_ for _ in ()).throw(OSError("replace")))
            else:
                monkeypatch.setattr(type(backend), "_fsync_directory", lambda *_: (_ for _ in ()).throw(OSError("fsync")))
    monkeypatch.undo()
    with pytest.raises(ExecutionOwnershipError):
        owner.validate_dispatch()
    async with acquire_execution(backend, original.state.session_key, run_id="next") as replacement:
        assert replacement.recovery_required
        with pytest.raises(ExecutionOwnershipError):
            replacement.validate_dispatch()


async def test_cancelled_cleanup_still_drains_effect_before_unlocking(backend):
    original = seed(backend)
    started, finish, closing = (asyncio.Event() for _ in range(3))
    async def effect():
        started.set()
        await finish.wait()
    async def run():
        async with acquire_execution(backend, original.state.session_key, run_id="run") as owner:
            owner.track_task(asyncio.create_task(effect()))
            await started.wait()
            closing.set()
    task = asyncio.create_task(run())
    await closing.wait()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    with pytest.raises(ExecutionBusyError):
        backend.delete(original.state.session_key)
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with acquire_execution(backend, original.state.session_key, run_id="next") as owner:
        owner.validate_dispatch()


async def test_unregistered_child_cannot_dispatch_mutate_or_register_itself(backend):
    original = seed(backend)
    async with acquire_execution(backend, original.state.session_key, run_id="run") as owner:
        snapshot = owner.read_snapshot()
        async def unregistered():
            with pytest.raises(ExecutionOwnershipError):
                owner.validate_dispatch()
            with pytest.raises(ExecutionOwnershipError):
                backend.source.append_source_events(owner.key, snapshot.state, [], owner=owner)
            with pytest.raises(ExecutionOwnershipError):
                owner.track_task(asyncio.current_task())
        await asyncio.create_task(unregistered())
        owner.validate_dispatch()


async def test_offline_writes_cannot_bypass_recovery_marker(backend):
    original = seed(backend, messages=[
        {"role": "assistant", "tool_calls": [{"id": "uncertain"}]},
    ])
    async with acquire_execution(backend, original.state.session_key, run_id="recover"):
        pass
    snapshot = backend.source.load_source_snapshot(original.state.session_key)
    assert "execution_lease" not in snapshot.header["metadata"]
    assert snapshot.header["metadata"]["execution_recovery_required"] is True
    for action in (
        lambda: backend.source.append_source_events(original.state.session_key, snapshot.state, []),
        lambda: backend.source.advance_consolidated(original.state.session_key, snapshot.state, 0),
        lambda: backend.source.reset_source_session(original.state.session_key, snapshot.state, []),
        lambda: backend.delete(original.state.session_key),
    ):
        with pytest.raises(ExecutionOwnershipError):
            action()
    async with acquire_execution(backend, original.state.session_key, run_id="explicit-reset") as owner:
        await owner.quiesce()
        backend.source.reset_source_session(owner.key, owner.read_snapshot().state, [], owner=owner)
    async with acquire_execution(backend, original.state.session_key, run_id="new-run") as owner:
        owner.validate_dispatch()


@pytest.mark.parametrize("field,value", [("token", "other"), ("incarnation", "other"), ("generation", 999)])
async def test_dispatch_rejects_changed_persisted_lease(backend, field, value):
    original = seed(backend)
    async with acquire_execution(backend, original.state.session_key, run_id="run") as owner:
        snapshot = owner.read_snapshot()
        saved = dict(snapshot.header["metadata"]["execution_lease"])
        snapshot.header["metadata"]["execution_lease"][field] = value
        backend.source._write(snapshot.header, snapshot.events)
        try:
            with pytest.raises(ExecutionOwnershipError):
                owner.validate_dispatch()
        finally:
            snapshot.header["metadata"]["execution_lease"] = saved
            backend.source._write(snapshot.header, snapshot.events)


async def test_acquisition_waits_for_other_process_and_observes_abandonment(backend, tmp_path):
    import multiprocessing
    original = seed(backend)
    context = multiprocessing.get_context("spawn")
    ready, exit_now = context.Queue(), context.Event()
    process = context.Process(target=_crash_owner, args=(
        tmp_path / "workspace", tmp_path / "private", original.state.session_key, ready, exit_now,
    ))
    entered = asyncio.Event()
    async def contender():
        async with acquire_execution(backend, original.state.session_key, run_id="contender") as owner:
            entered.set()
            assert owner.recovery_required
            with pytest.raises(ExecutionOwnershipError):
                owner.validate_dispatch()
    task = None
    try:
        process.start()
        assert await asyncio.to_thread(ready.get, timeout=10) == "owned"
        task = asyncio.create_task(contender())
        await asyncio.sleep(0.1)
        assert not entered.is_set() and not task.done()
        exit_now.set()
        await asyncio.wait_for(task, 10)
        await asyncio.to_thread(process.join, 10)
        assert entered.is_set() and process.exitcode == 0
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 5)
        ready.close()
