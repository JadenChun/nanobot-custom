"""Checked fence owners: preserved transaction phases and admitted defensive checks."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

import nanobot.cron.delivery_store as delivery
import nanobot.cron.occurrence_store as occurrence
from tests.cron.test_cron_delivery_store import _prepare
from tests.cron.test_occurrence_store import _claim, _store


@pytest.fixture(params=["occurrence", "delivery"])
def owner(request, tmp_path):
    if request.param == "occurrence":
        store = _store(tmp_path)
        claim = _claim(store)
        return SimpleNamespace(
            module=occurrence, store=store, claim=claim,
            cls=occurrence.OccurrenceExecutionFence, acquire=store.acquire_execution_fence,
            release=store.release_execution_fence, start=store.begin_execution,
            validate=store._assert_execution_fence, registry=store._issued_fences,
            error=occurrence.OccurrenceOwnershipError,
            proof="_assert_execution_lock_binding", guard="_execution_guard",
            live=occurrence._LIVE_FENCES, callback=occurrence._close_inherited_execution_fences,
            wrapper="assert_execution_fence_valid", held="assert_ledger_binding",
        )
    _service, _occurrence_claim, store, record = _prepare(tmp_path)
    claim = store.claim_destination(
        record.outbox_id, record.destinations[0].destination_id, 0, owner="sender"
    )
    return SimpleNamespace(
        module=delivery, store=store, claim=claim,
        cls=delivery.DeliverySendFence, acquire=store.acquire_send_fence,
        release=store.release_send_fence, start=store.mark_sending,
        validate=store._assert_send_fence, registry=store._issued_send_fences,
        error=delivery.DeliveryOutboxOwnershipError,
        proof="_assert_send_lock_binding", guard="_send_fence_guard",
        live=delivery._LIVE_SEND_FENCES, callback=delivery._close_inherited_send_fences,
        wrapper="assert_send_fence_valid", held="assert_outbox_binding",
    )


def _flag(fence, name, value):
    # Preservation cases run unchanged against both the original and new private layout.
    if hasattr(fence, "_lifecycle"):
        setattr(fence._lifecycle, name, value)
    else:
        setattr(fence, "_" + name, value)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", [
    "_active", "_store", "_claim", "_pid", "_task", "_root_binding", "_lock_fd",
    "_lock_inode", "_frozen", "_started", "claim_argument", "claim_content",
])
async def test_preserved_suffix_short_circuit(owner, monkeypatch, field):
    fence = owner.acquire(owner.claim)
    issued = owner.registry[id(fence)]
    suffix = ["_active", "_store", "_claim", "_pid", "_task", "_root_binding",
              "_lock_fd", "_lock_inode"]
    if owner.module is occurrence:
        suffix += ["_occurrence_id", "_owner", "_generation", "_frozen_snapshot",
                   "_frozen_policy", "_started"]
        frozen = "_frozen_snapshot"
    else:
        suffix += ["_frozen_claim", "claim_fields", "_started"]
        frozen = "_frozen_claim"
    target = frozen if field == "_frozen" else field
    old = None
    if target in {"_active", "_started"}:
        old = getattr(fence, target[1:])
        _flag(fence, target[1:], not old)
    elif target not in {"claim_argument", "claim_content"}:
        old = getattr(fence, target)
        setattr(fence, target, object())
    original_get = owner.cls.__getattribute__
    events = []
    lifecycle_reads = 0

    def observed(self, name):
        nonlocal lifecycle_reads
        if self is fence:
            if name == "_lifecycle":
                name_for_event = ["_active", "_started", "alias"][lifecycle_reads]
                lifecycle_reads += 1
                events.append(name_for_event)
            elif name in suffix:
                events.append(name)
        return original_get(self, name)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(owner.cls, "__getattribute__", observed)
            patch.setattr(owner.store, owner.proof, lambda *a: pytest.fail("late lock proof"))
            if owner.module is delivery:
                original_fields = delivery._claim_fields

                def fields(claim):
                    events.append("claim_fields")
                    return original_fields(claim)

                patch.setattr(delivery, "_claim_fields", fields)
            if target == "claim_content":
                if owner.module is occurrence:
                    owner.claim.job_snapshot["changed"] = True
                else:
                    object.__setattr__(owner.claim, "request_digest", "changed")
            argument = replace(owner.claim) if target == "claim_argument" else owner.claim
            with pytest.raises(owner.error, match="foreign, stale, inactive, or transferred"):
                owner.validate(fence, claim=argument, require_started=False)
        if target in suffix:
            assert events == suffix[:suffix.index(target) + 1]
        elif target == "claim_content" and owner.module is delivery:
            assert events == suffix[:suffix.index("claim_fields") + 1]
        else:
            assert events == suffix
    finally:
        if target in {"_active", "_started"}:
            _flag(fence, target[1:], old)
        elif target == "claim_content":
            if owner.module is occurrence:
                owner.claim.job_snapshot.pop("changed")
            else:
                object.__setattr__(owner.claim, "request_digest", fence._frozen_claim[-1])
        elif target != "claim_argument":
            setattr(fence, target, old)
        assert owner.registry[id(fence)] is issued
        owner.release(fence)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["_occurrence_id", "_owner", "_generation", "_frozen_policy"])
async def test_preserved_occurrence_extra_suffix(tmp_path, monkeypatch, field):
    store = _store(tmp_path)
    claim = _claim(store)
    fence = store.acquire_execution_fence(claim)
    old = getattr(fence, field)
    setattr(fence, field, object())
    with monkeypatch.context() as patch:
        patch.setattr(store, "_assert_execution_lock_binding", lambda *a: pytest.fail("lock proof"))
        with pytest.raises(occurrence.OccurrenceOwnershipError):
            store._assert_execution_fence(fence, require_started=False)
    setattr(fence, field, old)
    store.release_execution_fence(fence)


@pytest.mark.asyncio
async def test_preserved_exact_type_task_and_started_gates(owner, monkeypatch):
    class Hostile:
        def __getattribute__(self, name):
            raise AssertionError("field read before exact type")

    with pytest.raises(owner.error, match="exact issued"):
        owner.validate(Hostile(), require_started=False)
    fence = owner.acquire(owner.claim)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(owner.store, owner.proof, lambda *a: pytest.fail("lock proof"))
            with pytest.raises(owner.error, match="has not started"):
                owner.validate(fence, require_started=True)
            owner.registry[id(fence)].started = True
            with pytest.raises(owner.error, match="foreign, stale"):
                owner.validate(fence, require_started=False)
            owner.registry[id(fence)].started = False

            def no_task():
                raise RuntimeError("no event loop")

            patch.setattr(asyncio, "current_task", no_task)
            with pytest.raises(owner.error, match="foreign, stale"):
                owner.validate(fence, require_started=False)
        async def other_task():
            with pytest.raises(owner.error):
                owner.validate(fence, require_started=False)
        await asyncio.create_task(other_task())
    finally:
        owner.release(fence)
    with pytest.raises(owner.error):
        owner.validate(fence, require_started=False)


@pytest.mark.asyncio
async def test_new_content_free_validation_and_binding_delegates(owner, monkeypatch):
    fence = owner.acquire(owner.claim)
    try:
        assert getattr(owner.store, owner.wrapper)(fence, require_started=False) is None
        calls = []
        with monkeypatch.context() as patch:
            private_name = "_assert_execution_fence" if owner.module is occurrence else "_assert_send_fence"
            patch.setattr(owner.store, private_name, lambda *a, **kw: calls.append((a, kw)) or object())
            assert fence.__enter__() is fence
            assert calls == [((fence,), {"require_started": False})]
        with monkeypatch.context() as patch:
            patch.setattr(owner.store, "_assert_path_binding", lambda binding: calls.append(binding))
            patch.setattr(owner.store, "_assert_binding", lambda held: calls.append(held))
            with owner.store._held_lock() as held:
                # Held-lock entry/exit may invoke the retained private seams as well.
                calls.clear()
                assert owner.store.assert_root_binding(held.root_binding) is None
                assert getattr(owner.store, owner.held)(held) is None
                assert calls == [held.root_binding, held]
                calls.clear()
                held.root_binding.assert_bound()
                held.assert_bound()
                assert calls == [held.root_binding, held]
        assert fence.issued_state_mismatch(store=owner.store, issued=owner.registry[id(fence)],
                                           claim=owner.claim) is False
    finally:
        owner.release(fence)


@pytest.mark.parametrize("triggered", [False, True])
def test_new_persisted_parser_keeps_actual_identity_type(monkeypatch, triggered):
    value = {"store_namespace": "test", "job_id": "job", "job_revision": 1,
             "scheduled_fire_time_ms": 100}
    if triggered:
        value.update(trigger_version=2, source="manual", sequence=1, controller=None)
    parsed = occurrence.CronOccurrenceStore.parse_persisted_identity(value)
    assert type(parsed) is (occurrence.TriggeredOccurrenceIdentity if triggered
                            else occurrence.ScheduledOccurrenceIdentity)
    sentinel = object()
    monkeypatch.setattr(occurrence.CronOccurrenceStore, "_identity_from_wire", lambda v: sentinel)
    assert occurrence.CronOccurrenceStore.parse_persisted_identity(value) is sentinel


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["fence", "issued"])
@pytest.mark.parametrize("operation", ["start", "release"])
async def test_new_equal_cell_alias_refused_before_effects(owner, monkeypatch, side, operation):
    fence = owner.acquire(owner.claim)
    issued = owner.registry[id(fence)]
    cell = fence._lifecycle
    assert issued.lifecycle is cell
    replacement = owner.module._FenceLifecycle(active=cell.active, started=cell.started)
    if side == "fence":
        fence._lifecycle = replacement
    else:
        issued.lifecycle = replacement
    try:
        with monkeypatch.context() as patch:
            patch.setattr(owner.store, owner.proof, lambda *a: pytest.fail("alias reached lock proof"))
            patch.setattr(owner.store, "_held_lock", lambda: pytest.fail("alias reached durable IO"))
            with pytest.raises(owner.error, match="foreign, stale"):
                if operation == "start":
                    owner.start(owner.claim, fence)
                else:
                    owner.release(fence)
        assert owner.registry[id(fence)] is issued
        assert fence in owner.live
        assert not issued.started and not cell.started and cell.active
        if owner.module is occurrence:
            assert id(owner.claim) not in owner.store._started_executions
        else:
            assert owner.store._issued_claims[id(owner.claim)].phase == "claimed"
    finally:
        fence._lifecycle = cell
        issued.lifecycle = cell
        owner.release(fence)


@pytest.mark.asyncio
async def test_new_alias_check_follows_original_suffix(owner, monkeypatch):
    fence = owner.acquire(owner.claim)
    issued = owner.registry[id(fence)]
    original_cell = issued.lifecycle
    class HostileCell:
        def __getattribute__(self, name):
            raise AssertionError("alias read before original suffix")
    issued.lifecycle = HostileCell()
    try:
        # frozen-content refusal must short circuit before reading issued.lifecycle.
        with monkeypatch.context() as patch:
            if owner.module is occurrence:
                patch.setattr(occurrence, "_canonical_json", lambda value: "mismatch")
            else:
                patch.setattr(delivery, "_claim_fields", lambda value: ())
            assert fence.issued_state_mismatch(store=owner.store, issued=issued,
                                               claim=owner.claim) is True
    finally:
        issued.lifecycle = original_cell
        owner.release(fence)


@pytest.mark.asyncio
async def test_new_constructor_layout_and_readonly_flags(owner):
    fence = owner.acquire(owner.claim)
    try:
        assert "_lifecycle" in owner.cls.__slots__
        assert "_active" not in owner.cls.__slots__ and "_started" not in owner.cls.__slots__
        assert fence.active and not fence.started
        for name in ("active", "started"):
            with pytest.raises(AttributeError):
                setattr(fence, name, False)
        public = {name for name in dir(owner.cls) if not name.startswith("_")}
        assert public == {"active", "started", "issued_state_mismatch", "close_inherited_after_fork"}
        args = dict(store=owner.store, claim=owner.claim, task=asyncio.current_task(),
                    root_binding=fence._root_binding, lock_fd=fence._lock_fd,
                    lock_inode=fence._lock_inode)
        with pytest.raises(TypeError, match="issued only"):
            owner.cls(object(), **args)
        fallback = owner.cls(owner.module._FENCE_CONSTRUCTION_TOKEN, **args)
        assert fallback._lifecycle is not fence._lifecycle
        assert fallback.active and not fallback.started
        owner.start(owner.claim, fence)
        assert fence.started and owner.registry[id(fence)].started
    finally:
        owner.release(fence)
    assert not fence.active


@pytest.mark.asyncio
async def test_new_parent_checked_cleanup_and_callback_are_inert(owner, monkeypatch):
    fence = owner.acquire(owner.claim)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(owner.cls, "_after_fork_child", lambda self: pytest.fail("parent cleanup"))
            assert fence.close_inherited_after_fork() is None
            owner.callback()
        assert fence.active and id(fence) in owner.registry
    finally:
        owner.release(fence)


@pytest.mark.asyncio
async def test_new_parent_free_callback_does_not_enter_child_seam(owner, monkeypatch):
    fence = owner.acquire(owner.claim)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(owner.cls, "_after_fork_child", lambda self: pytest.fail("parent child seam"))
            owner.callback()
        assert fence.active and id(fence) in owner.registry
    finally:
        owner.release(fence)


@pytest.mark.asyncio
async def test_new_child_cleanup_resolves_private_seam_without_unlock(owner, monkeypatch):
    fence = owner.acquire(owner.claim)
    calls = []
    try:
        with monkeypatch.context() as patch:
            patch.setattr(owner.module.os, "getpid", lambda: fence._pid + 1)
            patch.setattr(owner.cls, "_after_fork_child", lambda self: calls.append(self))
            assert fence.close_inherited_after_fork() is None
            owner.callback()
        assert calls == [fence, fence]
        with monkeypatch.context() as patch:
            patch.setattr(owner.module.os, "close", lambda fd: calls.append(("close", fd)))
            patch.setattr(type(fence._root_binding), "close", lambda self: calls.append("root-close"))
            patch.setattr(owner.store._flock, "flock", lambda *a: pytest.fail("child unlock"))
            fence._after_fork_child()
            fence._after_fork_child()
        assert calls[2:] == [("close", fence._lock_fd), "root-close"]
        assert not fence.active and id(fence) in owner.registry
    finally:
        _flag(fence, "active", True)
        owner.release(fence)


@pytest.mark.asyncio
async def test_preserved_durable_start_failure_has_no_started_bookkeeping(owner, monkeypatch):
    fence = owner.acquire(owner.claim)
    issued = owner.registry[id(fence)]
    failure = RuntimeError("durable failure")
    try:
        with monkeypatch.context() as patch:
            def fail(*args, **kwargs):
                raise failure
            name = "_read_record" if owner.module is occurrence else "_replace_record"
            patch.setattr(owner.store, name, fail)
            with pytest.raises(RuntimeError) as caught:
                owner.start(owner.claim, fence)
            assert caught.value is failure
        assert not fence.started and not issued.started
        if owner.module is occurrence:
            assert id(owner.claim) not in owner.store._started_executions
            owner.start(owner.claim, fence)
            assert id(owner.claim) in owner.store._started_executions
        else:
            assert id(owner.claim) not in owner.store._issued_claims
    finally:
        owner.release(fence)


@pytest.mark.asyncio
@pytest.mark.parametrize("already_sending", [False, True])
async def test_preserved_delivery_branches_revision_and_reread(tmp_path, monkeypatch, already_sending):
    _service, _claim_obj, store, record = _prepare(tmp_path)
    claim = store.claim_destination(record.outbox_id, record.destinations[0].destination_id,
                                    0, owner="sender")
    if already_sending:
        first = store.acquire_send_fence(claim)
        store.mark_sending(claim, first)
        store.release_send_fence(first)
    issued_claim = store._issued_claims[id(claim)]
    previous_revision = issued_claim.current_revision
    fence = store.acquire_send_fence(claim)
    events = []
    read = store._read_bytes
    replace_record = store._replace_record
    try:
        with monkeypatch.context() as patch:
            def observed_read(*args, **kwargs):
                events.append(("read", kwargs))
                return read(*args, **kwargs)
            def observed_replace(*args, **kwargs):
                assert not fence.started and not store._issued_send_fences[id(fence)].started
                events.append(("replace", kwargs))
                return replace_record(*args, **kwargs)
            patch.setattr(store, "_read_bytes", observed_read)
            patch.setattr(store, "_replace_record", observed_replace)
            result = store.mark_sending(claim, fence)
        assert issued_claim.phase == "sending"
        assert fence.started and store._issued_send_fences[id(fence)].started
        if already_sending:
            assert [event[0] for event in events] == ["read"]
            assert issued_claim.current_revision == previous_revision
        else:
            assert [event[0] for event in events] == ["read", "replace", "read"]
            assert issued_claim.current_revision == result.record_revision == previous_revision + 1
            assert events[-1][1] == {"durable": False}
        with pytest.raises(delivery.DeliveryOutboxOwnershipError, match="already started"):
            store.mark_sending(claim, fence)
        assert id(claim) in store._issued_claims  # Outside the consuming catch.
    finally:
        store.release_send_fence(fence)


@pytest.mark.asyncio
async def test_preserved_delivery_post_replace_read_failure(tmp_path, monkeypatch):
    _service, _claim_obj, store, record = _prepare(tmp_path)
    claim = store.claim_destination(record.outbox_id, record.destinations[0].destination_id,
                                    0, owner="sender")
    fence = store.acquire_send_fence(claim)
    issued = store._issued_send_fences[id(fence)]
    issued_claim = store._issued_claims[id(claim)]
    old_revision = issued_claim.current_revision
    read = store._read_bytes
    failure = RuntimeError("post-replace reread")
    try:
        with monkeypatch.context() as patch:
            def observed_read(*args, **kwargs):
                if not kwargs.get("with_proof"):
                    assert issued.started and fence.started
                    assert issued_claim.phase == "sending"
                    assert issued_claim.current_revision == old_revision + 1
                    raise failure
                return read(*args, **kwargs)
            patch.setattr(store, "_read_bytes", observed_read)
            with pytest.raises(RuntimeError) as caught:
                store.mark_sending(claim, fence)
            assert caught.value is failure
        assert issued.started and fence.started
        assert id(claim) not in store._issued_claims
    finally:
        store.release_send_fence(fence)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_proof", [False, True])
async def test_preserved_release_order_and_first_error(owner, monkeypatch, fail_proof):
    fence = owner.acquire(owner.claim)
    issued = owner.registry[id(fence)]
    events = []
    proof = getattr(owner.store, owner.proof)
    close = owner.module.os.close
    root_cls = type(issued.root_binding)
    root_close = root_cls.close
    flock = owner.store._flock.flock
    failure = RuntimeError("release proof")
    proof_count = 0
    class Guard:
        def __enter__(self):
            events.append("guard-enter")
        def __exit__(self, *args):
            assert id(fence) not in owner.registry and fence not in owner.live
            assert not fence.active
            events.append("guard-exit")
    def observed_proof(*args):
        nonlocal proof_count
        proof_count += 1
        events.append("proof")
        if proof_count == 2 and fail_proof:
            raise failure
        return proof(*args)
    def observed_flock(fd, operation):
        events.append("unlock")
        flock(fd, operation)
        if fail_proof:
            raise OSError("secondary unlock")
    def observed_close(fd):
        if fd == issued.lock_fd:
            events.append("close")
        return close(fd)
    def observed_root_close(self):
        events.append("root-close")
        return root_close(self)
    with monkeypatch.context() as patch:
        patch.setattr(owner.store, owner.guard, Guard())
        patch.setattr(owner.store, owner.proof, observed_proof)
        patch.setattr(owner.store._flock, "flock", observed_flock)
        patch.setattr(owner.module.os, "close", observed_close)
        patch.setattr(root_cls, "close", observed_root_close)
        if fail_proof:
            with pytest.raises(RuntimeError) as caught:
                owner.release(fence)
            assert caught.value is failure
        else:
            owner.release(fence)
    assert events == ["guard-enter", "proof", "guard-exit", "proof", "unlock", "close", "root-close"]
    assert not fence.active
