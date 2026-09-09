from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nanobot.agent.loop import AgentLoop, TurnKind
from nanobot.agent.loop import TurnContext as StageContext
from nanobot.agent.turn import TurnContext, TurnRequest
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.session.manager import Session
from nanobot.session.summary import SessionSummaryCheckpoint


def host():
    loop = object.__new__(AgentLoop)
    loop.sessions = SimpleNamespace(save=Mock())
    loop.runtime_event_publisher = SimpleNamespace(session_turn_persisted=AsyncMock())
    return loop


def context(session, content='task', run_id='current'):
    return StageContext(
        msg=InboundMessage(channel='cli', chat_id='test', sender_id='user', content=content),
        session_key=session.key, session=session, turn_id='upstream-id', kind=TurnKind.USER,
        runtime=None, delivery=SimpleNamespace(),
        run_context=TurnContext(request=TurnRequest(content=content), run_id=run_id),
    )


def test_early_input_and_final_rows_have_trusted_run_id_once():
    loop = host()
    session = Session(key='cli:test')
    session.add_run_message('old', 'user', 'old input')
    msg = context(session).msg
    msg.metadata['_run_id'] = 'forged'
    assert loop._persist_user_message_early(msg, session, run_id='current', _run_id='forged')
    loop._save_turn(session, [
        {'role': 'system', 'content': 'prompt'},
        {'role': 'user', 'content': 'task'},
        {'role': 'assistant', 'content': 'answer', '_run_id': 'forged'},
    ], 2, input_persisted_early=True, run_id='current')
    assert [m['content'] for m in session.get_run_messages('current')] == ['task', 'answer']
    assert [m['content'] for m in session.get_run_messages('old')] == ['old input']
    assert not session.get_run_messages('forged')
    assert '_run_id' not in repr(session.get_history())


@pytest.mark.parametrize('boundary', [1, 2, 3])
def test_summary_boundaries_do_not_relabel_old_rows_or_tag_hidden_markers(boundary):
    loop = host()
    session = Session(key='cli:test')
    session.add_run_message('old', 'user', 'old input')
    messages = [{'role': 'system', 'content': 'prompt'},
                {'role': 'user', 'content': 'new input'},
                {'role': 'assistant', 'content': 'new answer'}]
    loop._save_turn(session, messages, 1, run_id='current',
                    summary_checkpoint=SessionSummaryCheckpoint(summary='summary', transcript_boundary=boundary))
    assert [m['content'] for m in session.get_run_messages('old')] == ['old input']
    assert [m['content'] for m in session.get_run_messages('current')] == ['new input', 'new answer']
    assert len([m for m in session.messages if not m.get('_run_id')]) == 1


def test_detail_selection_is_deeply_detached_and_tool_attribution_is_trusted():
    loop = host()
    session = Session(key='cli:test')
    loop._save_turn(session, [
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'call', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'call', 'content': [{'type': 'text', 'text': 'result'}], '_run_id': 'forged'},
    ], 0, run_id='current')
    selected = session.get_run_messages('current')
    selected[0]['tool_calls'][0]['function']['arguments'] = 'changed'
    selected[1]['content'][0]['text'] = 'changed'
    assert session.messages[0]['tool_calls'][0]['function']['arguments'] == '{}'
    assert session.messages[1]['content'][0]['text'] == 'result'
    assert not session.get_run_messages('forged')


async def test_command_shortcut_tags_input_and_reply():
    loop = host()
    session = Session(key='cli:test')
    ctx = context(session, '/help')
    loop.commands = SimpleNamespace(dispatch=AsyncMock(return_value=OutboundMessage(
        channel='cli', chat_id='test', content='help text')))
    assert await loop._dispatch_command(ctx)
    assert [m['content'] for m in session.get_run_messages('current')] == ['/help', 'help text']
    assert all(m['_command'] for m in session.get_run_messages('current'))


async def test_persist_stage_forwards_canonical_run_id():
    from nanobot.agent.context import TranscriptInput

    loop = host()
    session = Session(key='cli:test')
    ctx = context(session)
    ctx.transcript_input = TranscriptInput(history=[], current_message='task')
    ctx.all_messages = [{'role': 'system', 'content': 'prompt'},
                        {'role': 'user', 'content': 'task'},
                        {'role': 'assistant', 'content': 'answer'}]
    ctx.final_content = 'answer'
    ctx.delivery = SimpleNamespace(record_latency=Mock())
    await loop._persist_turn(ctx)
    assert [m['content'] for m in session.get_run_messages('current')] == ['task', 'answer']


def test_internal_followup_tagged_without_reassigning_deduplicated_input():
    loop = host()
    session = Session(key='cli:test')
    msg = InboundMessage(channel='system', chat_id='cli:test', sender_id='subagent',
                         content='result', metadata={'subagent_task_id': 'task'})
    assert loop._persist_subagent_followup(session, msg, run_id='one')
    assert not loop._persist_subagent_followup(session, msg, run_id='two')
    assert len(session.get_run_messages('one')) == 1
    assert session.get_run_messages('two') == []


async def test_fresh_build_tags_current_input_without_relabeling_retained_history():
    from nanobot.agent.turn import HistoryMode

    loop = host()
    session = Session(key='cli:test')
    session.add_run_message('old', 'user', 'old input')
    session.add_run_message('old', 'assistant', 'old answer')
    ctx = context(session)
    ctx.run_context.request = TurnRequest(content='task', history_mode=HistoryMode.FRESH)
    ctx.runtime = SimpleNamespace()
    ctx.delivery = SimpleNamespace(runtime_admitted=AsyncMock())
    loop._request_context_for_turn = lambda ctx: None
    loop._resolve_runtime_context_for_turn = AsyncMock(return_value=[])
    await loop._build_turn(ctx)
    assert ctx.history == []
    assert ctx.input_persisted_early
    assert [m['content'] for m in session.get_run_messages('current')] == ['task']
    assert [m['content'] for m in session.get_run_messages('old')] == ['old input', 'old answer']


def test_summary_before_early_input_preserves_input_run_tag_once():
    loop = host()
    session = Session(key='cli:test')
    session.add_run_message('old', 'user', 'old input')
    loop._persist_user_message_early(context(session).msg, session, run_id='current')
    loop._save_turn(session, [
        {'role': 'system', 'content': 'prompt'},
        {'role': 'user', 'content': 'task'},
        {'role': 'assistant', 'content': 'answer'},
    ], 2, run_id='current', input_persisted_early=True,
        summary_checkpoint=SessionSummaryCheckpoint(summary='summary', transcript_boundary=1))
    assert [m['content'] for m in session.get_run_messages('current')] == ['task', 'answer']
    assert session.messages[1].get('_run_id') is None
    assert session.last_archived == 1
