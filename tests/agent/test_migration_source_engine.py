"""Real canonical source turns with temporary state and synthetic providers."""

from unittest.mock import Mock

import pytest

from agent import test_migration_engine as engine_fixtures
from agent.test_migration_engine import ScriptedProvider
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import HistoryMode, TraceMode, TurnRequest
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.source import validate_execution_links

engine_config = engine_fixtures.engine_config

def upgrade(loop, key='cli:source'):
    session = loop.sessions.get_or_create(key)
    loop.sessions.save(session)
    backend = loop.sessions._jsonl_store
    backend.source.upgrade_source_session(key, expected_revision=session.revision,
        expected_generation=session.generation, expected_incarnation=session.incarnation)
    loop.sessions.invalidate(key)
    return backend


def forbid_legacy(loop, monkeypatch):
    for method in ('_save_turn', '_set_runtime_checkpoint'):
        monkeypatch.setattr(loop, method, Mock(side_effect=AssertionError('legacy source path')))
    monkeypatch.setattr(loop.sessions, 'save', Mock(side_effect=AssertionError('legacy save')))
    monkeypatch.setattr(loop.auto_compact, 'prepare_session', Mock(side_effect=AssertionError('legacy compaction')))


async def test_canonical_source_turn_persists_each_event_once(engine_config, monkeypatch):
    (engine_config.workspace_path / 'sample.txt').write_text('fixture contents')
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='r', name='read_file', arguments={'path': 'sample.txt'})]),
        LLMResponse(content='done'),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    forbid_legacy(loop, monkeypatch)
    try:
        result = await loop.execute_turn(TurnRequest(content='read the fixture', session_key='cli:source', trace_mode=TraceMode.SANITIZED))
        assert result.error is None, result.error
        assert result.content == 'done'
        events = backend.source.load_source_snapshot('cli:source').events
        assert [e['role'] for e in events] == ['user', 'assistant', 'tool', 'assistant']
        assert not validate_execution_links(events)
        assert len({e['_event_id'] for e in events}) == 4
        assert len(provider.calls) == 2
        assert [e['role'] for e in result.messages] == ['user', 'assistant', 'tool', 'assistant']
    finally:
        await loop.aclose()


async def test_source_reload_ignores_archive_cursor_and_fresh_excludes_old_input(engine_config, monkeypatch):
    provider = ScriptedProvider([LLMResponse(content='first'), LLMResponse(content='second'), LLMResponse(content='fresh')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    forbid_legacy(loop, monkeypatch)
    try:
        for content, mode in [('old instruction', HistoryMode.SESSION), ('new instruction', HistoryMode.SESSION), ('fresh instruction', HistoryMode.FRESH)]:
            result = await loop.execute_turn(TurnRequest(content=content, session_key='cli:source', history_mode=mode))
            assert result.error is None, result.error
            if content == 'old instruction':
                snapshot = backend.source.load_source_snapshot('cli:source')
                backend.source.advance_consolidated('cli:source', snapshot.state, len(snapshot.events))
        assert 'old instruction' in str(provider.calls[1][0])
        assert 'old instruction' not in str(provider.calls[2][0])
        assert sum('fresh instruction' in str(m.get('content')) for m in provider.calls[2][0]) == 1
        assert len(backend.source.load_source_snapshot('cli:source').events) == 6
    finally:
        await loop.aclose()


async def test_source_nested_planner_stream_does_not_leak_into_root_history(engine_config, monkeypatch):
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='plan', name='plan_task', arguments={'objective': 'fixture plan'})]),
        LLMResponse(content='child plan'), LLMResponse(content='parent final'), LLMResponse(content='next'),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    forbid_legacy(loop, monkeypatch)
    try:
        result = await loop.execute_turn(TurnRequest(content='plan fixture', session_key='cli:source'))
        assert result.error is None, result.error
        assert result.content == 'parent final'
        events = backend.source.load_source_snapshot('cli:source').events
        assert sum('_parent_stream_id' in e for e in events) == 2
        assert not validate_execution_links(events)
        result = await loop.execute_turn(TurnRequest(content='next input', session_key='cli:source'))
        assert result.error is None, result.error
        assert not any(m.get('role') == 'assistant' and m.get('content') == 'child plan' for m in provider.calls[-1][0])
    finally:
        await loop.aclose()


@pytest.mark.parametrize('command', ['/new', '/model default', '/help', '/compact'])
async def test_source_commands_never_full_save(engine_config, monkeypatch, command):
    provider = ScriptedProvider([])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    forbid_legacy(loop, monkeypatch)
    try:
        before = backend.source.load_source_snapshot('cli:source')
        result = await loop.execute_turn(TurnRequest(content=command, session_key='cli:source'))
        assert result.error is None, result.error
        after = backend.source.load_source_snapshot('cli:source')
        assert len(after.events) == (0 if command == '/new' else 2)
        assert not provider.calls
        if command == '/new':
            assert after.state.generation == before.state.generation + 1
    finally:
        await loop.aclose()


@pytest.mark.parametrize('failed_role', ['user', 'assistant', 'tool'])
async def test_source_write_failure_stops_before_next_effect(engine_config, monkeypatch, failed_role):
    (engine_config.workspace_path / 'sample.txt').write_text('evidence')
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='r', name='read_file', arguments={'path': 'sample.txt'})]),
        LLMResponse(content='must not reach final'),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    source_type = type(backend.source)
    original_write = source_type.append_source_events
    def write(store, key, expected, events, **kwargs):
        if any(e['role'] == failed_role for e in events):
            raise OSError('synthetic storage failure')
        return original_write(store, key, expected, events, **kwargs)
    monkeypatch.setattr(source_type, 'append_source_events', write)
    tool = loop.tools.get('read_file')
    original_execute = tool.execute_with_context
    from unittest.mock import AsyncMock
    observed = AsyncMock(wraps=original_execute)
    monkeypatch.setattr(tool, 'execute_with_context', observed)
    try:
        result = await loop.execute_turn(TurnRequest(content='read fixture', session_key='cli:source'))
        assert result.error and 'storage failure' in result.error
        assert len(provider.calls) == (0 if failed_role == 'user' else 1)
        assert observed.await_count == (1 if failed_role == 'tool' else 0)
        events = backend.source.load_source_snapshot('cli:source').events
        assert [e['role'] for e in events] == {'user': [], 'assistant': ['user'], 'tool': ['user', 'assistant']}[failed_role]
    finally:
        await loop.aclose()


async def test_source_large_result_is_retained_without_legacy_offload(engine_config, monkeypatch):
    from nanobot.agent.tools.base import Tool
    class Large(Tool):
        name = 'large_fixture'
        description = 'fixture'
        parameters = {'type': 'object', 'properties': {}}
        async def execute(self, **kwargs):
            return 'large-result-' * 5000
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='large', name='large_fixture', arguments={})]),
        LLMResponse(content='done'),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    loop.tools.register(Large())
    backend = upgrade(loop)
    monkeypatch.setattr('nanobot.agent.context_governance.maybe_persist_tool_result', Mock(side_effect=AssertionError('legacy offload')))
    try:
        result = await loop.execute_turn(TurnRequest(content='get fixture', session_key='cli:source'))
        assert result.error is None, result.error
        events = backend.source.load_source_snapshot('cli:source').events
        assert events[2]['content'] == 'large-result-' * 5000
        assert any(m.get('content') == events[2]['content'] for m in provider.calls[1][0])
    finally:
        await loop.aclose()


async def test_reused_run_id_has_distinct_stream_events_and_current_trace(engine_config):
    provider = ScriptedProvider([LLMResponse(content='first'), LLMResponse(content='second')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    try:
        for content in ['first input', 'second input']:
            result = await loop.execute_turn(TurnRequest(content=content, session_key='cli:source', run_id='reused'))
            assert result.error is None, result.error
        events = backend.source.load_source_snapshot('cli:source').events
        assert len({e['_event_id'] for e in events}) == 4
        assert len({e['_stream_id'] for e in events}) == 2
        trace = loop.run_store.load_trace('reused')
        assert [e['content'] for e in trace] == ['second input', 'second']
    finally:
        await loop.aclose()


async def test_history_command_uses_root_source_before_archive_cursor(engine_config):
    provider = ScriptedProvider([LLMResponse(content='old answer')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    try:
        result = await loop.execute_turn(TurnRequest(content='old input', session_key='cli:source'))
        assert result.error is None
        snapshot = backend.source.load_source_snapshot('cli:source')
        backend.source.advance_consolidated('cli:source', snapshot.state, len(snapshot.events))
        result = await loop.execute_turn(TurnRequest(content='/history', session_key='cli:source'))
        assert result.error is None, result.error
        assert 'old input' in result.content and 'old answer' in result.content
    finally:
        await loop.aclose()


async def test_source_scoped_worker_records_child_effects_under_parent_owner(engine_config):
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='delegate', name='delegate_task', arguments={
            'contract': 'Write fixture content.', 'write_scope': ['allowed.txt']})]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='write', name='write_file', arguments={'path': 'allowed.txt', 'content': 'worker output'})]),
        LLMResponse(content='worker complete'), LLMResponse(content='main complete'),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    try:
        result = await loop.execute_turn(TurnRequest(content='write fixture', session_key='cli:source'))
        assert result.error is None, result.error
        assert result.content == 'main complete'
        assert (engine_config.workspace_path / 'allowed.txt').read_text() == 'worker output'
        events = backend.source.load_source_snapshot('cli:source').events
        assert not validate_execution_links(events)
        assert len([e for e in events if e.get('_parent_stream_id')]) == 4
        assert len([e for e in events if e['role'] == 'tool' and e['_dispatched']]) == 2
    finally:
        await loop.aclose()


async def test_source_policy_terminal_persists_final_without_effect(engine_config):
    provider = ScriptedProvider([LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id='danger', name='exec', arguments={'command': 'rm -rf .'})])])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    try:
        result = await loop.execute_turn(TurnRequest(content='fixture', session_key='cli:source'))
        assert result.stop_reason == 'approval_required'
        events = backend.source.load_source_snapshot('cli:source').events
        assert [e['role'] for e in events] == ['user', 'assistant', 'tool', 'assistant']
        assert events[2]['_dispatched'] is False
        assert events[-1]['content'] == result.content
        assert not validate_execution_links(events)
    finally:
        await loop.aclose()


async def test_source_irreducible_context_makes_zero_provider_calls(engine_config):
    provider = ScriptedProvider([])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider,
                               context_window_tokens=32)
    backend = upgrade(loop)
    try:
        result = await loop.execute_turn(TurnRequest(content='too much context', session_key='cli:source'))
        assert result.error and 'context budget' in result.error
        assert not provider.calls
        assert [e['content'] for e in backend.source.load_source_snapshot('cli:source').events] == ['too much context']
    finally:
        await loop.aclose()


async def test_source_waiter_does_not_consume_approval_before_owner_releases(engine_config):
    import asyncio
    from unittest.mock import AsyncMock

    from nanobot.session.execution_owner import acquire_execution

    provider = ScriptedProvider([LLMResponse(content='done')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    consume = AsyncMock(wraps=loop.approval_admission.consume)
    loop.approval_admission.consume = consume
    task = None
    try:
        async with acquire_execution(backend, 'cli:source', run_id='other-process-equivalent'):
            task = asyncio.create_task(loop.execute_turn(TurnRequest(content='input', session_key='cli:source')))
            await asyncio.sleep(0.08)
            assert not task.done() and not provider.calls
            consume.assert_not_awaited()
        result = await asyncio.wait_for(task, 5)
        assert result.error is None, result.error
        assert consume.await_count == 1
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await loop.aclose()


async def test_source_metadata_finalization_preserves_concurrent_title(engine_config):
    provider = ScriptedProvider([LLMResponse(content='done')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    backend.update_metadata('cli:source', {'title': 'before'})
    original_chat = provider.chat
    async def chat(*args, **kwargs):
        backend.update_metadata('cli:source', {'title': 'concurrent title'})
        return await original_chat(*args, **kwargs)
    provider.chat = chat
    try:
        result = await loop.execute_turn(TurnRequest(content='input', session_key='cli:source'))
        assert result.error is None, result.error
        assert backend.source.load_source_snapshot('cli:source').header['metadata']['title'] == 'concurrent title'
    finally:
        await loop.aclose()


async def test_source_initial_transcript_binds_actual_events_and_fresh_scope(engine_config, monkeypatch):
    from nanobot.session.source_projection import RuntimeNode, SourceNode

    provider = ScriptedProvider([LLMResponse(content='first'), LLMResponse(content='second'),
                                 LLMResponse(content='fresh')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    captured = []
    run = loop.runner.run

    async def observe(spec):
        captured.append(spec.source_transcript)
        return await run(spec)

    monkeypatch.setattr(loop.runner, 'run', observe)
    try:
        for mode in (HistoryMode.SESSION, HistoryMode.SESSION, HistoryMode.FRESH):
            result = await loop.execute_turn(TurnRequest(content='same', session_key='cli:source', history_mode=mode))
            assert result.error is None
        events = backend.source.load_source_snapshot('cli:source').events
        first, second, fresh = captured
        assert isinstance(first.nodes[0], RuntimeNode)
        assert [node.event_id for node in first.nodes if isinstance(node, SourceNode)] == [events[0]['_event_id']]
        assert [node.event_id for node in second.nodes if isinstance(node, SourceNode)] == [row['_event_id'] for row in events[:3]]
        assert [node.event_id for node in fresh.nodes if isinstance(node, SourceNode)] == [events[4]['_event_id']]
        assert first.nodes[-1].event_id != second.nodes[-1].event_id
    finally:
        await loop.aclose()


async def test_malformed_retry_chain_returns_final_recorded_identity(engine_config, monkeypatch):
    def malformed(text):
        return LLMResponse(content=text, finish_reason='tool_calls',
            tool_calls=[ToolCallRequest(id='same', name='', arguments={})])
    provider = ScriptedProvider([malformed('first attempt'), malformed('second attempt'),
                                 LLMResponse(content='final answer')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    captured = []
    request = loop.runner._request_model

    async def observe(*args, **kwargs):
        recorded, usage = await request(*args, **kwargs)
        captured.append(recorded)
        return recorded, usage

    monkeypatch.setattr(loop.runner, '_request_model', observe)
    try:
        result = await loop.execute_turn(TurnRequest(content='request', session_key='cli:source'))
        assert result.error is None and result.content == 'final answer'
        events = backend.source.load_source_snapshot('cli:source').events
        assert [event['content'] for event in events] == ['request', 'first attempt', 'second attempt', 'final answer']
        assert len(captured) == 2
        assert all(recorded.source_event_id == events[-1]['_event_id'] for recorded in captured)
        assert all(recorded.response.content == 'final answer' for recorded in captured)
        assert not validate_execution_links(events)
    finally:
        await loop.aclose()
