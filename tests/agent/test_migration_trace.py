from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.trace import sanitize_trace
from nanobot.agent.turn import TraceMode, TurnRequest
from nanobot.events import NO_EVENTS
from nanobot.runtime_context import RUNTIME_CONTEXT_HISTORY_META
from nanobot.session.manager import Session

from .test_migration_canonical_admission import host


def test_trace_is_detached_bounded_and_structurally_private():
    raw = [
        {'role': 'system', 'content': 'system secret'},
        {'role': 'developer', 'content': 'developer secret'},
        {'role': 'user', 'content': 'hidden', '_hidden_history': True},
        {'role': 'user', 'content': 'input\n\nruntime secret',
         RUNTIME_CONTEXT_HISTORY_META: {'version': 1, 'suffix': 'runtime secret'}},
        {'role': 'assistant', 'content': 'answer', 'reasoning_content': 'private reasoning',
         '_run_id': 'private', 'tool_calls': [{'id': 'call', 'function': {'name': 'read_file', 'arguments': 'secret'}}]},
        {'role': 'tool', 'content': 'long text ' * 1000, 'tool_call_id': 'call', 'name': 'read_file'},
        {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,SECRET'}}]},
    ]
    before = deepcopy(raw)
    result = sanitize_trace(raw)
    rendered = repr(result)
    for omitted in ('system secret', 'developer secret', 'hidden', 'runtime secret',
                    'arguments', 'private reasoning', '_run_id', 'SECRET'):
        assert omitted not in rendered
    assert len(result[2]['content']) <= 4003
    assert result[0]['content'] == 'input'
    assert sanitize_trace(result) == result
    result[1]['tool_calls'][0]['name'] = 'changed'
    assert raw == before


def stage_host(tmp_path, *, fail=False):
    loop = host(tmp_path)
    session = Session(key='cli:test')
    session.add_run_message('previous', 'user', 'previous secret')
    loop.sessions = SimpleNamespace(save=Mock())
    loop.runtime_event_publisher = SimpleNamespace(session_turn_persisted=AsyncMock())
    delivery = SimpleNamespace(session_key=session.key, streaming=False, events=NO_EVENTS,
                               record_latency=Mock())
    loop.turn_delivery_factory = SimpleNamespace(create=lambda *args: delivery)
    loop._process_turn = AgentLoop._process_turn.__get__(loop)
    async def restore(ctx):
        ctx.session = session
    async def build(ctx):
        from nanobot.agent.context import TranscriptInput
        # Exercise expanded approval/runtime input while preserving original public text.
        ctx.msg.content = 'expanded private input'
        ctx.input_persisted_early = loop._persist_user_message_early(ctx.msg, session, run_id=ctx.run_context.run_id, trace_owner=ctx)
        ctx.transcript_input = TranscriptInput(history=[], current_message=ctx.msg.content)
        if fail:
            raise RuntimeError('build failed')
    async def run(ctx):
        ctx.all_messages = [{'role': 'system', 'content': 'system secret'},
                            {'role': 'user', 'content': 'expanded private input'},
                            {'role': 'assistant', 'content': 'answer'}]
        ctx.final_content = 'answer'
        ctx.run_result = AgentRunResult(messages=ctx.all_messages, final_content='answer')
    loop._restore_turn = restore
    loop._compact_session = AsyncMock()
    loop._dispatch_command = AsyncMock(return_value=False)
    loop._build_turn = build
    loop._run_turn = run
    loop._prepare_outbound = AsyncMock()
    return loop, session


@pytest.mark.parametrize('fail', [False, True])
async def test_real_stage_pipeline_returns_only_this_runs_sanitized_detail(tmp_path, fail):
    loop, session = stage_host(tmp_path, fail=fail)
    result = await loop.execute_turn(TurnRequest(content='original input', session_key='cli:test',
                                                trace_mode=TraceMode.SANITIZED))
    assert result.messages == ([{'role': 'user', 'content': 'original input'}] if fail else [
        {'role': 'user', 'content': 'original input'}, {'role': 'assistant', 'content': 'answer'}])
    assert session.messages[0]['content'] == 'previous secret'
    assert session.messages[1]['content'] == 'expanded private input'
    assert result.error == ('build failed' if fail else None)


async def test_default_trace_mode_does_not_read_or_sanitize(tmp_path, monkeypatch):
    loop, session = stage_host(tmp_path)
    monkeypatch.setattr('nanobot.agent.loop.sanitize_trace', Mock(side_effect=AssertionError('sanitized')))
    loop.run_store.load_trace = Mock(side_effect=AssertionError('loaded'))
    result = await loop.execute_turn(TurnRequest(content='input', session_key='cli:test'))
    assert result.messages is None
    assert len(session.get_run_messages(result.run_id)) == 2


def test_reasoning_content_blocks_are_not_public_trace_text():
    assert sanitize_trace([{"role": "assistant", "content": [
        {"type": "reasoning", "text": "private thought"},
        {"type": "thinking", "text": "another private thought"},
        {"type": "text", "text": "public answer"},
    ]}]) == [{"role": "assistant", "content": "public answer"}]
    assert sanitize_trace([{"role": [], "content": "malformed"}]) == []


async def test_save_failure_still_replaces_exact_expanded_input(tmp_path):
    loop, _ = stage_host(tmp_path)
    loop.sessions.save.side_effect = OSError('write failed')
    result = await loop.execute_turn(TurnRequest(content='yes', session_key='cli:test', trace_mode=TraceMode.SANITIZED))
    assert result.messages == [{'role': 'user', 'content': 'yes'}]
    assert result.error == 'write failed'


async def test_reused_run_id_does_not_select_previous_invocation(tmp_path):
    loop, session = stage_host(tmp_path)
    session.add_run_message('same', 'user', 'older invocation private')
    for content in ('one', 'two'):
        result = await loop.execute_turn(TurnRequest(content=content, session_key='cli:test',
                                                    run_id='same', trace_mode=TraceMode.SANITIZED))
        assert result.messages == [{'role': 'user', 'content': content}, {'role': 'assistant', 'content': 'answer'}]
    assert len(session.get_run_messages('same')) == 5


async def test_transient_session_trace_does_not_write_detail(tmp_path):
    from nanobot.session.manager import SessionManager, SessionPolicy

    loop, session = stage_host(tmp_path)
    session.policy = SessionPolicy(persist=False)
    store = SimpleNamespace(save=Mock(side_effect=AssertionError('persistent detail write')))
    loop.sessions = SessionManager(tmp_path / 'workspace', sessions_root=tmp_path / 'sessions', store=store)
    result = await loop.execute_turn(TurnRequest(content='input', session_key='cli:test', trace_mode=TraceMode.SANITIZED))
    assert result.messages == [{'role': 'user', 'content': 'input'}, {'role': 'assistant', 'content': 'answer'}]
    store.save.assert_not_called()


async def test_injected_input_runtime_suffix_is_removed_without_rewriting_its_text(tmp_path):
    loop, _ = stage_host(tmp_path)
    run = loop._run_turn
    async def injecting(ctx):
        await run(ctx)
        from nanobot.runtime_context import RUNTIME_CONTEXT_MESSAGE_META
        ctx.all_messages.insert(2, {'role': 'user', 'content': 'Original approval reply: real text\n\nprivate runtime',
            '_meta': {RUNTIME_CONTEXT_MESSAGE_META: {'version': 1, 'suffix': 'private runtime'}}})
    loop._run_turn = injecting
    result = await loop.execute_turn(TurnRequest(content='input', session_key='cli:test', trace_mode=TraceMode.SANITIZED))
    assert result.messages[1] == {'role': 'user', 'content': 'Original approval reply: real text'}


async def test_internal_input_trace_uses_original_request_without_prior_detail(tmp_path):
    loop, session = stage_host(tmp_path)
    async def build(ctx):
        from nanobot.agent.context import TranscriptInput
        ctx.msg.sender_id = 'subagent'
        ctx.msg.metadata['subagent_task_id'] = 'internal'
        ctx.msg.content = 'internal expanded input'
        loop._persist_subagent_followup(session, ctx.msg, run_id=ctx.run_context.run_id, trace_owner=ctx)
        ctx.input_persisted_early = True
        ctx.transcript_input = TranscriptInput(history=[], current_message=ctx.msg.content)
    loop._build_turn = build
    result = await loop.execute_turn(TurnRequest(content='internal result', session_key='cli:test', trace_mode=TraceMode.SANITIZED))
    assert result.messages == [{'role': 'assistant', 'content': 'internal result'}, {'role': 'assistant', 'content': 'answer'}]
