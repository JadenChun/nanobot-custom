import asyncio

import pytest

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
