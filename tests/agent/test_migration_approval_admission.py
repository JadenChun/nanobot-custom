import asyncio

import pytest

from nanobot.agent.turn import ApprovalDecision
from nanobot.agent.turn_admission import ApprovalAdmission


async def store(admission, summary='action'):
    return await admission.store(session_key='cron:job', channel='telegram', chat_id='42', summary=summary)


@pytest.mark.parametrize('reply,action,key', [
    (' YES! ', 'affirmative', 'cron:job'), ('go ahead.', 'affirmative', 'cron:job'),
    ('no', 'negative', 'telegram:42'), ('/status', 'command', 'telegram:42'),
    ('yes but delete everything', 'other', 'telegram:42'), ('hello', 'other', 'telegram:42')])
async def test_staging_reply_classification_and_consumption(reply, action, key):
    admission = ApprovalAdmission()
    pending = await store(admission)
    snapshot = await admission.snapshot('telegram:42', reply)
    assert snapshot.pending is pending
    assert snapshot.action == action and snapshot.effective_key == key
    assert await admission.consume(snapshot) == action
    remaining = await admission.snapshot('cron:job', 'yes')
    assert (remaining.pending is pending) == (action == 'command')


async def test_replacement_while_waiting_for_session_lock_cannot_be_consumed():
    admission = ApprovalAdmission()
    await store(admission, 'old')
    lock = asyncio.Lock()
    await lock.acquire()
    observed = asyncio.Event()
    async def waiter():
        snapshot = await admission.snapshot('telegram:42', 'yes')
        observed.set()
        async with lock:
            return await admission.consume(snapshot)
    task = asyncio.create_task(waiter())
    await observed.wait()
    newer = await store(admission, 'new')
    lock.release()
    assert await task == 'stale'
    assert (await admission.snapshot('telegram:42', 'yes')).pending is newer


async def test_only_one_alias_reply_can_consume_same_request():
    admission = ApprovalAdmission()
    await store(admission)
    first = await admission.snapshot('telegram:42', 'yes')
    second = await admission.snapshot('cron:job', 'yes')
    results = await asyncio.gather(admission.consume(first), admission.consume(second))
    assert sorted(results) == ['affirmative', 'stale']


async def test_clear_old_identity_preserves_new_visible_alias():
    admission = ApprovalAdmission()
    old = await store(admission)
    newer = await admission.store(session_key='cron:other', channel='telegram', chat_id='42', summary='new')
    assert not await admission.clear('telegram:42', old)
    assert await admission.clear('cron:job', old)
    assert (await admission.snapshot('telegram:42', 'yes')).pending is newer


async def test_cancelled_waiter_leaves_approval_pending_and_lock_reusable():
    admission = ApprovalAdmission()
    pending = await store(admission)
    lock = asyncio.Lock()
    await lock.acquire()
    observed = asyncio.Event()
    async def waiter():
        snapshot = await admission.snapshot('telegram:42', 'yes')
        observed.set()
        async with lock:
            return await admission.consume(snapshot)
    task = asyncio.create_task(waiter())
    await observed.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    lock.release()
    async with lock:
        snapshot = await admission.snapshot('telegram:42', 'yes')
        assert snapshot.pending is pending
        assert await admission.consume(snapshot) == 'affirmative'


async def test_no_pending_snapshot_does_not_consume_new_approval():
    admission = ApprovalAdmission()
    snapshot = await admission.snapshot('telegram:42', 'yes')
    pending = await store(admission)
    assert await admission.consume(snapshot) == 'none'
    assert (await admission.snapshot('telegram:42', 'yes')).pending is pending


async def test_command_snapshot_leaves_replacement_approval_pending():
    admission = ApprovalAdmission()
    await store(admission, 'old')
    snapshot = await admission.snapshot('telegram:42', '/status')
    newer = await store(admission, 'new')
    assert await admission.consume(snapshot) == 'command'
    assert (await admission.snapshot('telegram:42', 'yes')).pending is newer


@pytest.mark.asyncio
async def test_typed_approval_matches_exact_id_and_is_single_use():
    admission = ApprovalAdmission()
    pending = await store(admission)
    decision = ApprovalDecision(approval_id=pending.approval_id, decision='approve')
    snapshot = await admission.snapshot_decision('cron:job', decision)
    assert snapshot.pending is pending
    assert snapshot.action == 'affirmative' and snapshot.typed
    assert await admission.consume(snapshot) == 'affirmative'

    again = await admission.snapshot_decision(
        'cron:job', ApprovalDecision(approval_id=pending.approval_id, decision='approve'))
    assert again.pending is None
    assert await admission.consume(again) in {'stale', 'none'}


@pytest.mark.asyncio
@pytest.mark.parametrize('mismatch', ['approval_id', 'generation'])
async def test_wrong_typed_id_or_stale_generation_is_stale(mismatch):
    admission = ApprovalAdmission()
    pending = await admission.store(
        session_key='cron:job', channel='telegram', chat_id='42', summary='action',
        generation=1 if mismatch == 'generation' else None,
    )
    fields = {'approval_id': pending.approval_id, 'decision': 'approve'}
    if mismatch == 'approval_id':
        fields['approval_id'] = 'not-the-pending-id'
    else:
        fields['generation'] = 2
    snapshot = await admission.snapshot_decision('cron:job', ApprovalDecision(**fields))
    assert snapshot.pending is pending
    assert snapshot.action == 'stale'
    assert await admission.consume(snapshot) == 'stale'
    assert (await admission.snapshot_decision('cron:job', None)).pending is pending


@pytest.mark.asyncio
async def test_typed_deny_consumes_without_grant():
    admission = ApprovalAdmission()
    pending = await store(admission)
    decision = ApprovalDecision(approval_id=pending.approval_id, decision='deny')
    snapshot = await admission.snapshot_decision('cron:job', decision)
    assert snapshot.pending is pending
    assert snapshot.action == 'negative'
    assert await admission.consume(snapshot) == 'negative'
    assert (await admission.snapshot_decision('cron:job', None)).pending is None


@pytest.mark.asyncio
async def test_two_typed_clients_race_one_decision():
    admission = ApprovalAdmission()
    pending = await store(admission)
    def matching():
        return ApprovalDecision(approval_id=pending.approval_id, decision='approve')
    first = await admission.snapshot_decision('cron:job', matching())
    second = await admission.snapshot_decision('cron:job', matching())
    results = await asyncio.gather(admission.consume(first), admission.consume(second))
    assert sorted(results) == ['affirmative', 'stale']


@pytest.mark.asyncio
async def test_text_does_not_consume_typed_request():
    admission = ApprovalAdmission()
    pending = await store(admission)
    snapshot = await admission.snapshot_decision('cron:job', None)
    assert snapshot.pending is pending
    assert snapshot.action == 'other' and not snapshot.typed
    assert await admission.consume(snapshot) == 'other'
    assert (await admission.snapshot_decision('cron:job', None)).pending is pending


@pytest.mark.asyncio
async def test_expired_approval_is_removed_before_decision(monkeypatch):
    admission = ApprovalAdmission()
    now = [1_000_000.0]
    monkeypatch.setattr('nanobot.agent.turn_admission.time.time', lambda: now[0])
    pending = await store(admission)

    now[0] += 899
    assert (await admission.snapshot('telegram:42', 'yes')).pending is pending
    now[0] += 2
    assert (await admission.snapshot('telegram:42', 'yes')).pending is None
    now[0] -= 901
    assert (await admission.snapshot('telegram:42', 'yes')).pending is None

    typed = await store(admission, 'typed')
    now[0] += 901
    decision = ApprovalDecision(approval_id=typed.approval_id, decision='approve')
    assert (await admission.snapshot_decision('cron:job', decision)).pending is None


@pytest.mark.asyncio
async def test_revoke_removes_session_and_alias_keys():
    admission = ApprovalAdmission()
    pending = await store(admission)
    other = await admission.store(
        session_key='cron:other', channel='telegram', chat_id='99', summary='other')
    assert (await admission.snapshot('cron:job', 'yes')).pending is pending

    await admission.revoke('cron:job')

    assert (await admission.snapshot('cron:job', 'yes')).pending is None
    assert (await admission.snapshot('telegram:42', 'yes')).pending is None
    assert (await admission.snapshot('cron:other', 'yes')).pending is other
