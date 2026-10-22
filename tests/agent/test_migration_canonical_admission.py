import asyncio
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.run_store import RunStore
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import DelegationBudget, DeliveryTarget, RunStatus, TurnRequest
from nanobot.agent.turn_admission import ApprovalAdmission
from nanobot.session.manager import SessionManager


def host(tmp_path):
    loop = object.__new__(AgentLoop)
    loop._fork_dependencies = SimpleNamespace(
        foreground_manager=SimpleNamespace(new_budget=DelegationBudget),
        resource_policy=SimpleNamespace(workspace=tmp_path, context_manager=object()))
    loop.approval_admission = ApprovalAdmission()
    loop.sessions = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    loop.run_store = RunStore(tmp_path, session_manager=loop.sessions)
    loop._session_locks = weakref.WeakValueDictionary()
    loop._concurrency_gate = None
    loop._bus_cleanup_tasks = {}
    loop._pending_republish = {}
    loop.tools = ToolRegistry()
    loop._process_turn = AsyncMock(return_value=SimpleNamespace(
        run_result=AgentRunResult(final_content='answer', messages=[]), outbound=None))
    return loop


async def test_canonical_result_comes_from_runner_not_outbound(tmp_path):
    loop = host(tmp_path)
    loop._process_turn.return_value.run_result = AgentRunResult(
        final_content='blocked', messages=[], stop_reason='policy_blocked', policy_metadata={'why': 'test'})
    result = await loop.execute_turn(TurnRequest(content='task', session_key='cli:test'))
    assert result.status is RunStatus.POLICY_BLOCKED and result.content == 'blocked'
    assert result.policy_metadata == {'why': 'test'}
    assert loop.run_store.load(result.run_id).status is RunStatus.POLICY_BLOCKED
    call = loop._process_turn.await_args
    assert call.kwargs['run_context'].session_key == 'cli:test'


async def test_same_session_serializes_and_other_session_progresses(tmp_path):
    loop = host(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def stage(msg, **kwargs):
        calls.append(msg.content)
        if msg.content == 'first':
            entered.set()
            await release.wait()
        return SimpleNamespace(run_result=AgentRunResult(final_content=msg.content, messages=[]), outbound=None)
    loop._process_turn.side_effect = stage
    one = asyncio.create_task(loop.execute_turn(TurnRequest(content='first', session_key='cli:same')))
    await entered.wait()
    two = asyncio.create_task(loop.execute_turn(TurnRequest(content='second', session_key='cli:same')))
    other = await loop.execute_turn(TurnRequest(content='other', session_key='cli:other'))
    assert other.content == 'other' and 'second' not in calls
    release.set()
    await asyncio.gather(one, two)
    assert calls == ['first', 'other', 'second']


@pytest.mark.parametrize('reply,status,calls', [('yes', RunStatus.COMPLETED, 1), ('no', RunStatus.CANCELLED, 0)])
async def test_approval_reply_uses_task_lock_or_cancels_without_execution(tmp_path, reply, status, calls):
    loop = host(tmp_path)
    await loop.approval_admission.store(session_key='cron:job', channel='telegram', chat_id='42', summary='risky')
    result = await loop.execute_turn(TurnRequest(content=reply, route=DeliveryTarget(channel='telegram', chat_id='42')))
    assert result.status is status and loop._process_turn.await_count == calls
    if calls:
        assert loop._process_turn.await_args.kwargs['session_key'] == 'cron:job'
        assert loop._process_turn.await_args.kwargs['run_context'].policy.approval_granted


async def test_cancelled_waiter_finalizes_without_stage_execution(tmp_path):
    loop = host(tmp_path)
    lock = loop._get_session_lock('cli:same')
    await lock.acquire()
    task = asyncio.create_task(loop.execute_turn(TurnRequest(content='task', session_key='cli:same', run_id='waiting')))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    lock.release()
    assert loop.run_store.load('waiting').status is RunStatus.CANCELLED
    loop._process_turn.assert_not_awaited()
    assert (await loop.execute_turn(TurnRequest(content='next', session_key='cli:same'))).status is RunStatus.COMPLETED


async def test_usage_and_tool_error_callback_survive_conversion(tmp_path):
    from nanobot.agent.turn import TurnCallbacks
    from nanobot.providers.base import LLMUsage
    loop = host(tmp_path)
    loop._process_turn.return_value.run_result = AgentRunResult(final_content='failed', messages=[],
        stop_reason='tool_error', error='fixture error', usage=LLMUsage.estimated(input_tokens=10, output_tokens=3))
    error = AsyncMock(side_effect=RuntimeError('observer failed'))
    complete = AsyncMock()
    result = await loop.execute_turn(TurnRequest(content='task', callbacks=TurnCallbacks(on_error=error, on_complete=complete)))
    assert result.status is RunStatus.TOOL_ERROR
    assert result.usage['input_tokens'] == 10 and result.usage['output_tokens'] == 3
    record = loop.run_store.load(result.run_id)
    assert record.usage == result.usage and record.error_type == 'tool_error'
    assert record.error is None
    error.assert_awaited_once()
    complete.assert_not_awaited()
    loop._process_turn.assert_awaited_once()


async def test_trusted_reply_and_scheduled_provenance(tmp_path):
    from nanobot.agent.turn import ApprovalGrant, ScheduledTurnLink
    loop = host(tmp_path)
    request = TurnRequest(content='task', session_key='cron:job',
        delivery_target=DeliveryTarget(channel='telegram', chat_id='42', message_id='trusted'),
        metadata={'message_id': 'untrusted', 'purpose': 'fixture'},
        approval_grant=ApprovalGrant(approved=True, resumed_run_id='previous'),
        scheduled_link=ScheduledTurnLink(job_id='job', visible_session_key='telegram:42:topic:7'))
    result = await loop.execute_turn(request)
    msg = loop._process_turn.await_args.args[0]
    assert msg.metadata['message_id'] == 'trusted'
    assert request.metadata['message_id'] == 'untrusted'
    record = loop.run_store.load(result.run_id)
    assert record.resumed_run_id == 'previous'
    assert record.visible_session_key == 'telegram:42:topic:7'
    assert record.metadata['purpose'] == 'fixture'
    assert record.scheduled_link.job_id == 'job'


async def test_direct_adapter_uses_canonical_admission_and_preserves_options(tmp_path):
    from nanobot.bus.events import OutboundMessage
    loop = host(tmp_path)
    loop.runtime_event_publisher = SimpleNamespace(run_status_changed=AsyncMock(), clear_turn=lambda key: None)
    outbound = OutboundMessage(channel='cli', chat_id='direct', content='answer')
    loop._process_turn.return_value.outbound = outbound
    runtime, hook, factory = object(), object(), object()
    admitted, progress = AsyncMock(), AsyncMock()
    result = await asyncio.wait_for(loop.process_direct('task', session_key='cli:test',
        runtime=runtime, hooks=[hook], hook_factories=[factory], on_runtime_admitted=admitted,
        on_progress=progress, ephemeral=True, _run_extra_hooks_for_ephemeral=True,
        persist_user_message=False, attributes={'fixture': 1}), timeout=1)
    assert result is outbound
    call = loop._process_turn.await_args
    assert call.kwargs['run_context'].session_key == 'cli:test'
    assert call.kwargs['runtime'] is runtime
    assert call.kwargs['hooks'] == [hook] and call.kwargs['hook_factories'] == [factory]
    assert call.kwargs['on_runtime_admitted'] is admitted
    assert call.kwargs['on_progress'] is progress
    assert call.kwargs['ephemeral'] and call.kwargs['run_extra_hooks_for_ephemeral']
    assert call.kwargs['attributes'] == {'fixture': 1}
    from nanobot.session.turn_continuation import SKIP_USER_PERSIST_META
    assert call.args[0].metadata[SKIP_USER_PERSIST_META]
    loop.runtime_event_publisher.run_status_changed.assert_awaited_once()


async def test_direct_exception_propagates_after_terminal_record(tmp_path):
    loop = host(tmp_path)
    loop.runtime_event_publisher = SimpleNamespace(run_status_changed=AsyncMock(), clear_turn=lambda key: None)
    loop._process_turn.side_effect = ValueError('fixture failure')
    with pytest.raises(ValueError, match='fixture failure'):
        await loop.process_direct('task')
    records = list(loop.run_store.runs_dir.glob('*.json'))
    assert len(records) == 1
    assert loop.run_store.load(records[0].stem).status is RunStatus.ERROR
    loop._process_turn.assert_awaited_once()


async def test_cancelled_direct_waiter_does_not_publish_idle_for_active_turn(tmp_path):
    loop = host(tmp_path)
    idle = AsyncMock()
    loop.runtime_event_publisher = SimpleNamespace(run_status_changed=idle, clear_turn=lambda key: None)
    lock = loop._get_session_lock('cli:same')
    await lock.acquire()
    task = asyncio.create_task(loop.process_direct('waiting', session_key='cli:same'))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    idle.assert_not_awaited()
    lock.release()


async def test_direct_approval_cleanup_uses_effective_key_while_lock_owned(tmp_path):
    loop = host(tmp_path)
    await loop.approval_admission.store(session_key='cron:job', channel='telegram', chat_id='42', summary='task')
    observed = []
    async def idle(msg, key, status):
        assert loop._get_session_lock(key).locked()
        observed.append((key, status))
    loop.runtime_event_publisher = SimpleNamespace(run_status_changed=idle, clear_turn=lambda key: None)
    await loop.process_direct('yes', session_key='telegram:42', channel='telegram', chat_id='42')
    assert observed == [('cron:job', 'idle')]


async def test_cancelled_gate_waiter_does_not_publish_idle(tmp_path):
    loop = host(tmp_path)
    loop._concurrency_gate = asyncio.Semaphore(0)
    idle = AsyncMock()
    loop.runtime_event_publisher = SimpleNamespace(run_status_changed=idle, clear_turn=lambda key: None)
    task = asyncio.create_task(loop.process_direct('waiting', session_key='cli:same'))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    idle.assert_not_awaited()
    assert not loop._get_session_lock('cli:same').locked()


async def test_idle_publisher_failure_still_clears_and_unlocks(tmp_path):
    loop = host(tmp_path)
    cleared = []
    loop.runtime_event_publisher = SimpleNamespace(
        run_status_changed=AsyncMock(side_effect=RuntimeError('publisher')),
        clear_turn=cleared.append)
    await loop.process_direct('task', session_key='cli:same')
    assert cleared == ['cli:same']
    assert not loop._get_session_lock('cli:same').locked()


@pytest.mark.parametrize('fence', ['session', 'gate', 'cleanup'])
async def test_unadmitted_cancel_terminal_has_no_admission_effects(tmp_path, fence):
    from nanobot.agent.turn import TurnCallbacks

    loop = host(tmp_path)
    reached = asyncio.Event()
    lock = loop._get_session_lock('cli:queued')
    future = asyncio.get_running_loop().create_future()
    if fence == 'session':
        await lock.acquire()
        original = lock.acquire

        async def acquire():
            reached.set()
            return await original()

        lock.acquire = acquire
    elif fence == 'gate':
        class Gate(asyncio.Semaphore):
            async def acquire(self):
                reached.set()
                return await super().acquire()

        loop._concurrency_gate = Gate(0)
    else:
        loop._bus_cleanup_tasks['cli:queued'] = {future}
        original_wait = asyncio.wait

        async def wait(pending):
            reached.set()
            return await original_wait(pending)

        # This local admission spy never owns or cancels the cleanup future.
        from unittest.mock import patch
        patcher = patch('nanobot.agent.loop.asyncio.wait', wait)
        patcher.start()
    complete = AsyncMock()
    loop.approval_admission.consume = AsyncMock(side_effect=AssertionError('consume'))
    loop.sessions._load = lambda key: pytest.fail('history load')
    loop.sessions._repair = lambda *args, **kwargs: pytest.fail('repair')
    task = asyncio.create_task(loop.execute_turn(TurnRequest(
        content='private task', session_key='cli:queued', run_id='queue-record',
        callbacks=TurnCallbacks(on_complete=complete))))
    try:
        await reached.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        record = loop.run_store.load('queue-record')
        assert record.status is RunStatus.CANCELLED
        assert record.completed_at and record.stop_reason == 'cancelled'
        loop._process_turn.assert_not_awaited()
        loop.approval_admission.consume.assert_not_awaited()
        complete.assert_not_awaited()
        assert not future.cancelled()
        assert not loop.sessions._cancellation_receipts
    finally:
        if fence == 'session':
            lock.release()
        elif fence == 'cleanup':
            patcher.stop()
        future.set_result(None)


@pytest.mark.parametrize('transition', ['active', 'dispose', 'evict'])
async def test_cancelled_waiter_privacy_revocation_survives_cache_loss(tmp_path, transition):
    loop = host(tmp_path)
    key = 'cli:private'
    if transition == 'active':
        loop.sessions.get_or_create_transient(key, retain=True)
    reached = asyncio.Event()
    lock = loop._get_session_lock(key)
    await lock.acquire()
    original = lock.acquire

    async def acquire():
        reached.set()
        return await original()

    lock.acquire = acquire
    task = asyncio.create_task(loop.execute_turn(TurnRequest(
        content='private', session_key=key, run_id='private-waiter')))
    await reached.wait()
    if transition != 'active':
        session = loop.sessions.get_or_create_transient(key, retain=True)
        if transition == 'evict':
            loop.sessions._max_cached_sessions = 0
            loop.sessions.get_or_create('cli:pressure')
        loop.sessions.invalidate(key)
        del session
        assert loop.sessions.get_cached(key) is None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    lock.release()
    assert not loop.run_store.path_for('private-waiter').exists()
    loop._process_turn.assert_not_awaited()
    assert not loop.sessions._cancellation_receipts


@pytest.mark.parametrize('fault', ['render', 'fsync', 'link', 'cleanup'])
async def test_cancelled_waiter_bookkeeping_fault_preserves_cancellation(tmp_path, monkeypatch, fault):
    import os

    loop = host(tmp_path)
    reached = asyncio.Event()
    lock = loop._get_session_lock('cli:fault')
    await lock.acquire()
    original = lock.acquire

    async def acquire():
        reached.set()
        return await original()

    lock.acquire = acquire

    def fail(*args, **kwargs):
        raise OSError('private error body')

    if fault == 'cleanup':
        unlink = os.unlink

        def cleanup(path):
            unlink(path)
            fail()

        monkeypatch.setattr('nanobot.agent.run_store.os.unlink', cleanup)
    else:
        monkeypatch.setattr('nanobot.agent.run_store.' + {
            'render': 'json.dump', 'fsync': 'os.fsync', 'link': 'os.link',
        }[fault], fail)
    task = asyncio.create_task(loop.execute_turn(TurnRequest(
        content='private', session_key='cli:fault', run_id='fault')))
    await reached.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    lock.release()
    assert not list(loop.run_store.runs_dir.glob('*.tmp'))
    assert not loop.sessions._cancellation_receipts
    loop._process_turn.assert_not_awaited()


@pytest.mark.parametrize('exclusion', ['ephemeral', 'required', 'custom', 'managed', 'claimed', 'source', 'alias'])
async def test_cancelled_waiter_exclusions_do_not_load_or_admit(tmp_path, monkeypatch, exclusion):
    import json

    from nanobot.agent.loop import _TurnExecutionOptions
    from nanobot.bus.events import InboundMessage
    from nanobot.session.runtime_authority import OfflineSessionOperator

    loop = host(tmp_path)
    key = 'cli:exclude'
    options = _TurnExecutionOptions()
    request = TurnRequest(content='task', session_key=key, run_id='excluded')
    lease = None
    if exclusion == 'ephemeral':
        options = _TurnExecutionOptions(ephemeral=True)
    elif exclusion == 'required':
        options = _TurnExecutionOptions(inbound=InboundMessage(
            channel='cli', chat_id='exclude', sender_id='user',
            content='task', require_existing_session=True))
    elif exclusion == 'custom':
        from unittest.mock import Mock
        loop.sessions._custom_store = Mock()
    elif exclusion in {'managed', 'claimed'}:
        lease = await OfflineSessionOperator(
            loop.sessions.workspace, copied_sessions_root=loop.sessions.sessions_root,
        ).acquire()
        if exclusion == 'managed':
            loop.sessions = lease.manager
        else:
            lease.manager.admit_persistent_write('cli:claim')
    elif exclusion == 'source':
        loop.sessions._get_session_path(key).write_text(json.dumps({
            '_type': 'metadata', 'key': key, 'metadata': {'source_schema_version': 1},
        }) + '\nmalformed history must not be read\n')
    else:
        await loop.approval_admission.store(session_key='cron:actual', channel='cli',
                                           chat_id='exclude', summary='private task')
        request = TurnRequest(content='yes', session_key=key, run_id='excluded')
        key = 'cron:actual'
    reached = asyncio.Event()
    lock = loop._get_session_lock(key)
    await lock.acquire()
    original = lock.acquire

    async def acquire():
        reached.set()
        return await original()

    lock.acquire = acquire
    monkeypatch.setattr(loop.sessions, '_load', lambda key: pytest.fail('history load'))
    monkeypatch.setattr(loop.sessions, '_repair', lambda *a, **k: pytest.fail('repair'))
    monkeypatch.setattr(loop.sessions, 'source_backend', lambda key: pytest.fail('source load'))
    loop.approval_admission.consume = AsyncMock(side_effect=AssertionError('consume'))
    before = {p.name: p.read_bytes() for p in loop.sessions.sessions_dir.iterdir() if p.is_file()}
    task = asyncio.create_task(loop.execute_turn(request, _options=options))
    try:
        await reached.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not loop.run_store.path_for('excluded').exists()
        assert not loop.sessions._cancellation_receipts
        assert before == {p.name: p.read_bytes() for p in loop.sessions.sessions_dir.iterdir() if p.is_file()}
        loop._process_turn.assert_not_awaited()
        loop.approval_admission.consume.assert_not_awaited()
        if exclusion == 'alias':
            assert (await loop.approval_admission.snapshot('cli:exclude', 'yes')).pending is not None
    finally:
        lock.release()
        if lease is not None:
            await lease.close()


async def test_cancellation_before_record_construction_releases_receipt(tmp_path):
    loop = host(tmp_path)
    reached = asyncio.Event()

    async def snapshot(*args):
        reached.set()
        await asyncio.Future()

    loop.approval_admission.snapshot = snapshot
    task = asyncio.create_task(loop.execute_turn(TurnRequest(content='private', run_id='too-early')))
    await reached.wait()
    assert len(loop.sessions._cancellation_receipts) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not loop.sessions._cancellation_receipts
    assert not loop.run_store.path_for('too-early').exists()
    loop._process_turn.assert_not_awaited()


@pytest.mark.parametrize('outcome', ['success', 'stage-error', 'rejection', 'construction-error'])
async def test_cancellation_receipt_released_on_non_cancel_outcomes(tmp_path, outcome):
    from nanobot.agent.loop import _TurnExecutionOptions
    from nanobot.bus.events import InboundMessage

    loop = host(tmp_path)
    request = TurnRequest(content='task', session_key='cli:release', run_id='release')
    options = _TurnExecutionOptions()
    if outcome == 'construction-error':
        def fail():
            raise ValueError('construction')
        loop._fork_dependencies.foreground_manager.new_budget = fail
        with pytest.raises(ValueError, match='construction'):
            await loop.execute_turn(request)
    else:
        if outcome == 'stage-error':
            loop._process_turn.side_effect = ValueError('stage')
        elif outcome == 'rejection':
            options = _TurnExecutionOptions(inbound=InboundMessage(
                channel='cli', chat_id='release', sender_id='user',
                content='task', require_existing_session=True))
        result = await loop.execute_turn(request, _options=options)
        assert result.status is (RunStatus.COMPLETED if outcome == 'success' else RunStatus.ERROR)
    assert not loop.sessions._cancellation_receipts


async def test_admitted_cancellation_keeps_existing_save_path(tmp_path):
    loop = host(tmp_path)
    reached = asyncio.Event()

    async def stage(*args, **kwargs):
        reached.set()
        await asyncio.Future()

    loop._process_turn.side_effect = stage
    loop.run_store.insert_cancelled_terminal = lambda record: pytest.fail('fallback after admission')
    task = asyncio.create_task(loop.execute_turn(TurnRequest(
        content='task', session_key='cli:admitted', run_id='admitted-cancel')))
    await reached.wait()
    assert loop.run_store.load('admitted-cancel').status is RunStatus.RUNNING
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert loop.run_store.load('admitted-cancel').status is RunStatus.CANCELLED
    assert not loop.sessions._cancellation_receipts
