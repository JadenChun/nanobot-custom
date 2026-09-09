import asyncio

import pytest

from nanobot.agent.tools.message import MessageTool, capture_message_deliveries
from nanobot.agent.turn import DeliveryState, DeliveryTarget, TurnContext


def context(chat):
    return TurnContext(delivery=DeliveryState(primary=DeliveryTarget(channel='telegram', chat_id=chat)))


async def test_contextual_send_keeps_detached_full_ledger_and_target_capture(tmp_path):
    media = tmp_path / 'image.png'
    media.write_bytes(b'fixture')
    async def send(msg):
        msg.content = 'callback mutation'
        msg.media.clear()
    tool = MessageTool(send_callback=send, workspace=tmp_path)
    ctx = context('42')
    with capture_message_deliveries() as targets:
        await tool.execute_with_context(ctx, content='final', media=[str(media)])
    assert targets == {('telegram', '42')}
    assert len(ctx.delivery.sent_messages) == 1
    record = ctx.delivery.sent_messages[0]
    assert record.content == 'final' and record.media == [str(media)]
    assert ctx.delivery.delivered


@pytest.mark.parametrize('failure', [RuntimeError('failed'), asyncio.CancelledError()])
async def test_failed_or_cancelled_send_does_not_claim_delivery(failure):
    async def send(msg):
        raise failure
    tool = MessageTool(send_callback=send)
    ctx = context('42')
    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await tool.execute_with_context(ctx, content='final')
    else:
        await tool.execute_with_context(ctx, content='final')
    assert not ctx.delivery.sent_messages and not ctx.delivery.delivered


async def test_shared_tool_concurrent_contexts_have_separate_ledgers():
    async def send(msg):
        await asyncio.sleep(0)
    tool = MessageTool(send_callback=send)
    one, two = context('1'), context('2')
    await asyncio.gather(tool.execute_with_context(one, content='one'),
                         tool.execute_with_context(two, content='two'))
    assert [(m.chat_id, m.content) for m in one.delivery.sent_messages] == [('1', 'one')]
    assert [(m.chat_id, m.content) for m in two.delivery.sent_messages] == [('2', 'two')]


async def test_reply_id_inherited_only_for_primary_and_suppression_has_no_receipt():
    async def send(msg):
        pass
    tool = MessageTool(send_callback=send)
    ctx = TurnContext(delivery=DeliveryState(primary=DeliveryTarget(
        channel='telegram', chat_id='42', message_id='reply')))
    await tool.execute_with_context(ctx, content='primary')
    await tool.execute_with_context(ctx, content='elsewhere', chat_id='43')
    assert ctx.delivery.sent_messages[0].metadata['message_id'] == 'reply'
    assert 'message_id' not in ctx.delivery.sent_messages[1].metadata
    token = tool.set_suppress_delivery(True)
    try:
        await tool.execute_with_context(ctx, content='suppressed')
    finally:
        tool.reset_suppress_delivery(token)
    assert len(ctx.delivery.sent_messages) == 2


@pytest.mark.parametrize("contextual", [False, True])
async def test_actual_run_stage_retains_runner_outcome_and_plain_tool_ledger(contextual):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nanobot.agent.loop import AgentLoop, TurnKind
    from nanobot.agent.loop import TurnContext as StageContext
    from nanobot.agent.runner import AgentRunResult
    from nanobot.bus.events import InboundMessage

    async def send(msg):
        pass
    tool = MessageTool(send_callback=send)
    result = AgentRunResult(final_content='blocked', messages=[], stop_reason='policy_blocked',
                            policy_metadata={'reason': 'fixture'}, tools_used=['message'])
    async def run(*args, **kwargs):
        if contextual:
            await tool.execute_with_context(kwargs['run_context'], content='explicit')
        else:
            await tool.execute(content='explicit', channel='telegram', chat_id='42')
        return result
    # Exercise the real stage with injected collaborators; never construct a runtime.
    loop = object.__new__(AgentLoop)
    loop._run_agent_loop = run
    delivery = SimpleNamespace(running=AsyncMock(), record_usage=lambda usage: None)
    ctx = StageContext(msg=InboundMessage(channel='telegram', chat_id='42', sender_id='cron', content='task'),
                       session_key='cron:test', turn_id='test', runtime=object(),
                       kind=TurnKind.SYSTEM, delivery=delivery)
    ctx.transcript_input = object()
    if contextual:
        ctx.run_context = context('42')
    await loop._run_turn(ctx)
    if contextual:
        assert ctx.sent_messages is ctx.run_context.delivery.sent_messages
    assert ctx.run_result is result
    assert ctx.stop_reason == 'policy_blocked'
    assert [(m.chat_id, m.content) for m in ctx.sent_messages] == [('42', 'explicit')]


async def test_contextual_capture_requires_one_ledger_and_restores_after_cancel():
    async def send(msg):
        if msg.content == 'cancel':
            raise asyncio.CancelledError()
    tool = MessageTool(send_callback=send)
    ctx = context('42')
    with capture_message_deliveries([]):
        with pytest.raises(ValueError, match='share one list'):
            await tool.execute_with_context(ctx, content='rejected')
    assert not ctx.delivery.sent_messages
    with capture_message_deliveries(ctx.delivery.sent_messages) as targets:
        with pytest.raises(asyncio.CancelledError):
            await tool.execute_with_context(ctx, content='cancel')
        await tool.execute(content='plain', channel='telegram', chat_id='43')
        await tool.execute_with_context(ctx, content='contextual')
    assert targets == {('telegram', '42'), ('telegram', '43')}
    assert [m.content for m in ctx.delivery.sent_messages] == ['plain', 'contextual']
    assert ctx.delivery.message_id is None


async def test_trusted_route_does_not_inherit_other_request_metadata():
    from nanobot.agent.tools.context import RequestContext, current_request_context, request_context
    async def send(msg):
        pass
    tool = MessageTool(send_callback=send)
    ctx = context('42')
    previous = RequestContext(channel='slack', chat_id='old', message_id='old',
                              metadata={'topic': 'private'})
    with request_context(previous):
        await tool.execute_with_context(ctx, content='new')
        assert current_request_context() is previous
    assert ctx.delivery.sent_messages[0].metadata == {}


async def test_denied_media_has_no_submission_or_ledger(tmp_path):
    from unittest.mock import AsyncMock
    send = AsyncMock()
    tool = MessageTool(send_callback=send, workspace=tmp_path, restrict_to_workspace=True)
    ctx = context('42')
    with capture_message_deliveries(ctx.delivery.sent_messages) as targets:
        result = await tool.execute_with_context(ctx, content='no', media=[str(tmp_path.parent / 'outside')])
    assert 'not allowed' in str(result)
    assert not ctx.delivery.sent_messages and not targets
    send.assert_not_awaited()


async def test_ledger_nested_metadata_and_buttons_are_detached():
    from nanobot.agent.tools.context import RequestContext, request_context
    async def send(msg):
        msg.metadata['nested']['value'] = 2
        msg.buttons[0][0] = 'changed'
    tool = MessageTool(send_callback=send)
    ctx = context('42')
    with request_context(RequestContext(channel='telegram', chat_id='42',
                                        metadata={'nested': {'value': 1}})):
        await tool.execute_with_context(ctx, content='yes', buttons=[['original']])
    assert ctx.delivery.sent_messages[0].metadata['nested']['value'] == 1
    assert ctx.delivery.sent_messages[0].buttons == [['original']]
