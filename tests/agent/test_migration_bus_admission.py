import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nanobot.agent.runner import AgentRunResult
from nanobot.bus.events import InboundMessage

from .test_migration_canonical_admission import host


def bus_host(tmp_path):
    loop = host(tmp_path)
    loop._unified_session = False
    loop._recovery_admission = None
    loop._pending_queues = {}
    loop._pending_republish = {}
    loop._busy_consumers = {}
    loop._stopped_pending_followup_ids = {}
    loop._active_tasks = {}
    loop._discarding_sessions = set()
    loop._preserve_inflight_turns_on_shutdown = True
    loop.bus = SimpleNamespace(publish_inbound_nowait=Mock())
    loop.runtime_event_publisher = SimpleNamespace(clear_turn=Mock(), run_status_changed=AsyncMock())
    delivery = SimpleNamespace(complete=AsyncMock(), idle=AsyncMock(), fail=AsyncMock(), abort_stream=AsyncMock())
    loop.turn_delivery_factory = SimpleNamespace(unrouted=Mock(return_value=delivery), create=Mock(return_value=delivery))
    loop._publish_next_deferred_automation_turn = AsyncMock()
    coordinator = SimpleNamespace(complete=Mock())
    loop._automation_turn_coordinators = [('fixture', coordinator)]
    return loop, delivery, coordinator


def message(content='task'):
    return InboundMessage(channel='telegram', chat_id='42', sender_id='user', content=content)


async def test_bus_scope_is_locked_preserves_envelope_and_drains_once(tmp_path):
    loop, delivery, coordinator = bus_host(tmp_path)
    loop.sessions.get_or_create('telegram:42')
    msg = message()
    msg.require_existing_session = True
    msg.input_role = 'system'
    leftover = message('leftover')
    async def stage(inbound, **kwargs):
        assert inbound.require_existing_session and inbound.input_role == 'system'
        assert inbound.timestamp == msg.timestamp
        assert loop._get_session_lock('telegram:42').locked()
        assert kwargs['pending_queue'] is loop._pending_queues['telegram:42']
        kwargs['pending_queue'].put_nowait(leftover)
        return SimpleNamespace(run_result=AgentRunResult(final_content='answer', messages=[]), outbound=None)
    loop._process_turn.side_effect = stage
    await loop._dispatch(msg)
    assert not loop._pending_queues
    loop.bus.publish_inbound_nowait.assert_called_once_with(leftover)
    delivery.complete.assert_awaited_once()
    delivery.idle.assert_awaited_once()
    coordinator.complete.assert_called_once()


@pytest.mark.parametrize('gate', [False, True])
async def test_waiting_bus_cancellation_has_no_lifecycle_effects(tmp_path, gate):
    loop, delivery, coordinator = bus_host(tmp_path)
    lock = loop._get_session_lock('telegram:42')
    if gate:
        loop._concurrency_gate = asyncio.Semaphore(0)
    else:
        await lock.acquire()
    task = asyncio.create_task(loop._dispatch(message()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not loop._pending_queues
    delivery.idle.assert_not_awaited()
    delivery.complete.assert_not_awaited()
    coordinator.complete.assert_not_called()
    loop._publish_next_deferred_automation_turn.assert_not_awaited()
    if not gate:
        lock.release()


async def test_delivery_failure_does_not_repeat_stage_or_change_computation(tmp_path):
    loop, delivery, coordinator = bus_host(tmp_path)
    delivery.complete.side_effect = RuntimeError('delivery')
    await loop._dispatch(message())
    loop._process_turn.assert_awaited_once()
    delivery.complete.assert_awaited_once()
    delivery.fail.assert_not_awaited()
    coordinator.complete.assert_called_once()
    records = list(loop.run_store.runs_dir.glob('*.json'))
    record = loop.run_store.load(records[0].stem)
    assert record.status.value == 'completed'
    assert record.metadata['delivery_error_type'] == 'RuntimeError'


@pytest.mark.parametrize('factory', [False, True])
async def test_stage_or_factory_failure_releases_queue_and_notifies_once(tmp_path, factory):
    loop, delivery, coordinator = bus_host(tmp_path)
    (loop.turn_delivery_factory.create if factory else loop._process_turn).side_effect = RuntimeError('fixture')
    await loop._dispatch(message())
    assert not loop._pending_queues
    delivery.fail.assert_awaited_once()
    delivery.idle.assert_awaited_once()
    coordinator.complete.assert_called_once()


async def test_bus_and_direct_share_admission_lock(tmp_path):
    loop, delivery, _ = bus_host(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def stage(msg, **kwargs):
        calls.append(msg.content)
        if msg.content == 'bus':
            entered.set()
            await release.wait()
        return SimpleNamespace(run_result=AgentRunResult(final_content='answer', messages=[]), outbound=None)
    loop._process_turn.side_effect = stage
    bus = asyncio.create_task(loop._dispatch(message('bus')))
    await entered.wait()
    direct = asyncio.create_task(loop.process_direct('direct', session_key='telegram:42'))
    await asyncio.sleep(0)
    assert calls == ['bus']
    release.set()
    await asyncio.gather(bus, direct)
    assert calls == ['bus', 'direct']
    delivery.complete.assert_awaited_once()


async def test_admitted_cancellation_cleans_owned_queue_and_preserves_shutdown_checkpoint(tmp_path):
    loop, delivery, coordinator = bus_host(tmp_path)
    entered = asyncio.Event()
    async def stage(msg, **kwargs):
        entered.set()
        await asyncio.Event().wait()
    loop._process_turn.side_effect = stage
    task = asyncio.create_task(loop._dispatch(message()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not loop._pending_queues
    delivery.abort_stream.assert_awaited_once()
    delivery.idle.assert_awaited_once()
    coordinator.complete.assert_called_once()


async def test_cleanup_does_not_remove_replacement_queue(tmp_path):
    loop, _, _ = bus_host(tmp_path)
    newer = asyncio.Queue()
    async def stage(msg, **kwargs):
        loop._pending_queues['telegram:42'] = newer
        return SimpleNamespace(run_result=AgentRunResult(final_content='answer', messages=[]), outbound=None)
    loop._process_turn.side_effect = stage
    await loop._dispatch(message())
    assert loop._pending_queues['telegram:42'] is newer


@pytest.mark.parametrize('outcome', ['reject', 'error', 'cancel'])
async def test_recovery_registration_always_unregistered(tmp_path, outcome):
    from nanobot.session.recovery import RECOVERY_INBOUND_METADATA_KEY
    loop, delivery, _ = bus_host(tmp_path)
    recovery = SimpleNamespace(register_recovery_task=Mock(), unregister_recovery_task=Mock(), admit=AsyncMock(return_value=False))
    if outcome == 'error':
        recovery.admit.side_effect = RuntimeError('admit')
    if outcome == 'cancel':
        recovery.admit.side_effect = asyncio.CancelledError()
    loop._recovery_admission = recovery
    msg = message()
    msg.metadata[RECOVERY_INBOUND_METADATA_KEY] = 'recovery'
    if outcome == 'cancel':
        with pytest.raises(asyncio.CancelledError):
            await loop._dispatch(msg)
    else:
        await loop._dispatch(msg)
    recovery.register_recovery_task.assert_called_once()
    recovery.unregister_recovery_task.assert_called_once_with('telegram:42', asyncio.current_task())
    delivery.idle.assert_not_awaited()


async def test_completion_observes_stage_continuation_metadata(tmp_path):
    from nanobot.session.turn_continuation import INTERNAL_CONTINUATION_PENDING_META
    loop, delivery, _ = bus_host(tmp_path)
    async def stage(msg, **kwargs):
        msg.metadata[INTERNAL_CONTINUATION_PENDING_META] = True
        return SimpleNamespace(run_result=AgentRunResult(final_content='answer', messages=[]), outbound=None)
    loop._process_turn.side_effect = stage
    await loop._dispatch(message())
    assert delivery.complete.await_args.kwargs['publish_completion'] is False
    delivery.idle.assert_not_awaited()
    loop.runtime_event_publisher.clear_turn.assert_not_called()


async def test_failed_leftovers_retained_in_order_and_retried_once(tmp_path):
    loop, _, _ = bus_host(tmp_path)
    one, two = message('one'), message('two')
    async def stage(msg, **kwargs):
        kwargs['pending_queue'].put_nowait(one)
        kwargs['pending_queue'].put_nowait(two)
        return SimpleNamespace(run_result=AgentRunResult(final_content='answer', messages=[]), outbound=None)
    loop._process_turn.side_effect = stage
    loop.bus.publish_inbound_nowait.side_effect = RuntimeError('queue')
    await loop._dispatch(message())
    assert loop._pending_republish['telegram:42'] == [one, two]
    assert not loop._pending_queues
    loop._process_turn.side_effect = None
    loop.bus.publish_inbound_nowait.reset_mock(side_effect=True)
    await loop._dispatch(message())
    assert [call.args[0] for call in loop.bus.publish_inbound_nowait.call_args_list] == [one, two]
    assert not loop._pending_republish


async def test_completion_cancellation_preserves_computation_and_aborts_stream(tmp_path):
    loop, delivery, coordinator = bus_host(tmp_path)
    delivery.complete.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await loop._dispatch(message())
    delivery.complete.assert_awaited_once()
    delivery.abort_stream.assert_awaited_once()
    coordinator.complete.assert_called_once()
    record = loop.run_store.load(next(loop.run_store.runs_dir.glob('*.json')).stem)
    assert record.status.value == 'completed'
    assert record.metadata['delivery_error_type'] == 'CancelledError'
    assert not loop._pending_queues


async def test_hung_cleanup_is_bounded_retains_input_and_releases_lock(tmp_path, monkeypatch):
    import nanobot.agent.loop as module
    monkeypatch.setattr(module, '_BUS_CLEANUP_TIMEOUT_SECONDS', 0.01)
    loop, delivery, _ = bus_host(tmp_path)
    entered = asyncio.Event()
    leftover = message('leftover')
    async def stage(msg, **kwargs):
        kwargs['pending_queue'].put_nowait(leftover)
        return SimpleNamespace(run_result=AgentRunResult(final_content='answer', messages=[]), outbound=None)
    async def hang(*args):
        entered.set()
        await asyncio.Event().wait()
    loop._process_turn.side_effect = stage
    delivery.idle.side_effect = hang
    loop.runtime_event_publisher.clear_turn.side_effect = RuntimeError('clear')
    task = asyncio.create_task(loop._dispatch(message()))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert not loop._get_session_lock('telegram:42').locked()
    loop.bus.publish_inbound_nowait.assert_called_once_with(leftover)
    loop._publish_next_deferred_automation_turn.assert_awaited_once()


async def test_redirected_cancel_uses_effective_key_and_compact_completes(tmp_path, monkeypatch):
    import nanobot.agent.loop as module
    loop, delivery, _ = bus_host(tmp_path)
    await loop.approval_admission.store(session_key='cron:job', channel='telegram', chat_id='42', summary='task')
    loop._preserve_inflight_turns_on_shutdown = False
    keys = []
    get_or_create = loop.sessions.get_or_create

    def observed_get_or_create(key):
        keys.append(key)
        return get_or_create(key)

    monkeypatch.setattr(loop.sessions, 'get_or_create', observed_get_or_create)
    loop._clear_pending_user_turn = Mock()
    monkeypatch.setattr(module, 'restore_runtime_checkpoint', lambda session: True)
    loop._process_turn.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await loop._dispatch(message('yes'))
    assert keys == ['cron:job']
    assert loop.turn_delivery_factory.create.call_args.args[1] == 'cron:job'
    assert not loop._pending_queues
    assert loop.runtime_event_publisher.clear_turn.call_args.args == ('cron:job',)
    with pytest.raises(asyncio.CancelledError):
        await loop._dispatch(message('/compact'))
    delivery.complete.assert_awaited_once_with(None, publish_completion=True)


@pytest.mark.parametrize('compact', [False, True])
async def test_failure_notices_have_deadlines_before_cleanup(tmp_path, monkeypatch, compact):
    import nanobot.agent.loop as module
    monkeypatch.setattr(module, '_BUS_CLEANUP_TIMEOUT_SECONDS', 0.01)
    loop, delivery, _ = bus_host(tmp_path)
    async def hang(*args, **kwargs):
        await asyncio.Event().wait()
    if compact:
        loop._process_turn.side_effect = asyncio.CancelledError()
        delivery.complete.side_effect = hang
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(loop._dispatch(message('/compact')), 1)
    else:
        loop._process_turn.side_effect = RuntimeError('stage')
        delivery.fail.side_effect = hang
        await asyncio.wait_for(loop._dispatch(message()), 1)
    assert not loop._pending_queues
    assert not loop._get_session_lock('telegram:42').locked()


async def test_held_input_wakes_when_late_idle_finishes_without_another_turn(tmp_path, monkeypatch):
    import nanobot.agent.loop as module
    from nanobot.bus.queue import MessageBus

    monkeypatch.setattr(module, '_BUS_CLEANUP_TIMEOUT_SECONDS', 0.01)
    loop, delivery, _ = bus_host(tmp_path)
    loop.bus = MessageBus()
    release = asyncio.Event()

    async def late_idle():
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    delivery.idle.side_effect = late_idle
    await loop._dispatch(message('first'))
    one, two = message('held one'), message('held two')
    waiting = [asyncio.create_task(loop._dispatch(item)) for item in (one, two)]
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not loop._get_session_lock('telegram:42').locked()
    assert loop._process_turn.await_count == 1
    assert loop.bus.inbound_size == 0
    release.set()
    await asyncio.wait_for(asyncio.gather(*waiting), 1)
    assert loop._process_turn.await_count == 3
    assert [call.args[0].content for call in loop._process_turn.await_args_list] == ['first', 'held one', 'held two']
    assert not loop._pending_republish


@pytest.mark.parametrize('original_pending', [False, True])
async def test_fenced_approval_retains_original_snapshot(tmp_path, original_pending):
    loop, _, _ = bus_host(tmp_path)
    key = 'telegram:42'
    if original_pending:
        await loop.approval_admission.store(session_key=key, channel='telegram', chat_id='42', summary='old')
    fence = asyncio.get_running_loop().create_future()
    loop._bus_cleanup_tasks[key] = {fence}
    loop._concurrency_gate = asyncio.Semaphore(1)
    waiting = asyncio.create_task(loop._dispatch(message('yes')))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not loop._get_session_lock(key).locked()
    assert not loop._concurrency_gate.locked()
    await loop.approval_admission.store(session_key=key, channel='telegram', chat_id='42', summary='new')
    fence.set_result(None)
    await asyncio.wait_for(waiting, 1)
    if original_pending:
        loop._process_turn.assert_not_awaited()
    else:
        assert not loop._process_turn.call_args.kwargs['run_context'].policy.approval_granted
    snapshot = await loop.approval_admission.snapshot(key, 'yes')
    assert snapshot.pending.summary == 'new'


async def test_cancelled_fence_waiter_leaves_approval_and_lifecycle_untouched(tmp_path):
    loop, delivery, coordinator = bus_host(tmp_path)
    key = 'telegram:42'
    await loop.approval_admission.store(session_key=key, channel='telegram', chat_id='42', summary='old')
    fence = asyncio.get_running_loop().create_future()
    loop._bus_cleanup_tasks[key] = {fence}
    waiting = asyncio.create_task(loop._dispatch(message('yes')))
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert not fence.cancelled()
    delivery.idle.assert_not_awaited()
    coordinator.complete.assert_not_called()
    loop._process_turn.assert_not_awaited()
    snapshot = await loop.approval_admission.snapshot(key, 'yes')
    assert snapshot.pending.summary == 'old'
    fence.set_result(None)


async def test_fence_requires_all_operations_to_settle(tmp_path):
    loop, _, _ = bus_host(tmp_path)
    one, two = (asyncio.get_running_loop().create_future() for _ in range(2))
    loop._bus_cleanup_tasks['telegram:42'] = {one, two}
    waiting = asyncio.create_task(loop.process_direct('next', session_key='telegram:42'))
    await asyncio.sleep(0)
    one.set_result(None)
    await asyncio.sleep(0)
    loop._process_turn.assert_not_awaited()
    two.set_result(None)
    await asyncio.wait_for(waiting, 1)
    loop._process_turn.assert_awaited_once()
    assert not loop._bus_cleanup_tasks


async def test_real_bus_leftovers_are_enqueued_once_in_fifo_order(tmp_path):
    from nanobot.bus.queue import MessageBus

    loop, _, _ = bus_host(tmp_path)
    loop.bus = MessageBus()
    one, two = message('one'), message('two')
    async def stage(msg, **kwargs):
        kwargs['pending_queue'].put_nowait(one)
        kwargs['pending_queue'].put_nowait(two)
        return SimpleNamespace(run_result=AgentRunResult(final_content='answer', messages=[]), outbound=None)
    loop._process_turn.side_effect = stage
    await loop._dispatch(message())
    assert loop.bus.inbound_size == 2
    assert await loop.bus.consume_inbound() is one
    assert await loop.bus.consume_inbound() is two
    assert not loop._pending_republish


async def test_repeated_cancellation_fences_unfinished_stream_abort(tmp_path):
    loop, delivery, _ = bus_host(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    async def abort():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
    delivery.abort_stream.side_effect = abort
    loop._process_turn.side_effect = asyncio.CancelledError()
    owner = asyncio.create_task(loop._dispatch(message()))
    await entered.wait()
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    loop._process_turn.side_effect = None
    successor = asyncio.create_task(loop.process_direct('next', session_key='telegram:42'))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not successor.done()
    assert loop._process_turn.await_count == 1
    release.set()
    await asyncio.wait_for(successor, 1)
    assert loop._process_turn.await_count == 2
