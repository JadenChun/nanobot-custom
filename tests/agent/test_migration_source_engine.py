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
        from nanobot.agent.tools.read_provenance import parse_read_receipt, root_binding
        receipt = parse_read_receipt(events[2]['_source_origin'])
        assert receipt.matches(events[2]['content'])
        assert receipt.source_access.relative_path == 'sample.txt'
        assert receipt.root_binding == root_binding(engine_config.workspace_path)
        assert events[2]['_execution_id'] == events[1]['_execution_ids'][0]
        assert events[2]['_run_id'] == events[1]['_run_id']
        assert events[2]['_stream_id'] == events[1]['_stream_id']
        assert '_source_origin' not in str(provider.calls[-1][0])
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


async def test_repeated_provider_call_ids_keep_exact_tool_replay_bindings(engine_config, monkeypatch):
    from nanobot.session.source_projection import SourceNode

    (engine_config.workspace_path / 'same.txt').write_text('same result')
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='same', name='read_file',
            arguments={'path': 'same.txt'}) for _ in range(2)]),
        LLMResponse(content='done'),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    captured = []
    prepare = loop.runner.context_governor.prepare_request

    async def observe(state, messages, **kwargs):
        if state.source is not None:
            captured.append((state.source.transcript, state.source.eligible_ids))
        return await prepare(state, messages, **kwargs)

    monkeypatch.setattr(loop.runner.context_governor, 'prepare_request', observe)
    try:
        result = await loop.execute_turn(TurnRequest(content='read twice', session_key='cli:source'))
        assert result.error is None
        events = backend.source.load_source_snapshot('cli:source').events
        assert len(captured) == 2
        nodes = [node for node in captured[1][0].nodes if isinstance(node, SourceNode)]
        assert [node.event_id for node in nodes] == [row['_event_id'] for row in events[:-1]]
        tool_nodes = [node for node in nodes if node.message['role'] == 'tool']
        assert len(tool_nodes) == 2 and tool_nodes[0].event_id != tool_nodes[1].event_id
        assert [node.message['tool_call_id'] for node in tool_nodes] == ['same', 'same']
        assert captured[1][1] == {events[0]['_event_id']}  # New tool evidence is still pending.
    finally:
        await loop.aclose()


@pytest.mark.parametrize('mode', ['length', 'continuation'])
async def test_source_runtime_continuations_preserve_replay_provenance(engine_config, monkeypatch, mode):
    from nanobot.session.source_projection import RuntimeNode, SourceNode

    provider = ScriptedProvider([LLMResponse(content='segment', finish_reason='length' if mode == 'length' else 'stop'),
                                 LLMResponse(content='complete')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    captured = []
    prepare = loop.runner.context_governor.prepare_request
    run = loop.runner.run

    async def observe(state, messages, **kwargs):
        assert state.source.transcript.messages() == messages
        captured.append(state.source.transcript)
        return await prepare(state, messages, **kwargs)

    async def configure(spec):
        if mode == 'continuation':
            notes = iter(['finish the objective', None])
            spec.continuation_callback = lambda: next(notes)
        return await run(spec)

    monkeypatch.setattr(loop.runner.context_governor, 'prepare_request', observe)
    monkeypatch.setattr(loop.runner, 'run', configure)
    try:
        result = await loop.execute_turn(TurnRequest(content='objective', session_key='cli:source'))
        assert result.error is None
        events = backend.source.load_source_snapshot('cli:source').events
        assert len(captured) == 2
        nodes = captured[1].nodes
        assert [node.event_id for node in nodes if isinstance(node, SourceNode)] == [row['_event_id'] for row in events[:2]]
        assert isinstance(nodes[-1], RuntimeNode)
        assert [row['content'] for row in events] == ['objective', 'segment', 'complete']
    finally:
        await loop.aclose()


@pytest.mark.parametrize('mode', ['error', 'max_iterations'])
async def test_legacy_terminal_replay_keeps_existing_replacement_behavior(mode):
    from agent.runner_helpers import make_run_spec
    from nanobot.agent.runner import AgentRunner

    provider = ScriptedProvider([LLMResponse(content='failure', finish_reason='error')])
    result = await AgentRunner().run(make_run_spec(provider,
        initial_messages=[{'role': 'user', 'content': 'input'}, {'role': 'assistant', 'content': 'existing'}],
        tools=ToolRegistry(), model='fixture', max_iterations=1 if mode == 'error' else 0,
        max_tool_result_chars=100, finalize_on_max_iterations=False))
    assert len(result.messages) == 2
    if mode == 'error':
        assert result.messages[-1]['content'] == 'existing'
    else:
        assert result.messages[-1]['content'] == result.final_content


@pytest.mark.parametrize('failure', ['error', 'malformed'])
async def test_only_validated_source_request_input_is_acknowledged(engine_config, monkeypatch, failure):
    responses = [LLMResponse(content='failed', finish_reason='error')] if failure == 'error' else [
        LLMResponse(content='invalid attempt', finish_reason='tool_calls',
                    tool_calls=[ToolCallRequest(id='bad', name='', arguments={})]),
        LLMResponse(content='recovered'),
    ]
    provider = ScriptedProvider(responses)
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    snapshots, ledgers = [], []
    prepare = loop.runner.context_governor.prepare_request

    async def observe(state, messages, **kwargs):
        snapshots.append(state.source.eligible_ids)
        ledgers.append(state.source)
        return await prepare(state, messages, **kwargs)

    monkeypatch.setattr(loop.runner.context_governor, 'prepare_request', observe)
    try:
        await loop.execute_turn(TurnRequest(content='request', session_key='cli:source'))
        events = backend.source.load_source_snapshot('cli:source').events
        assert all(not eligible for eligible in snapshots)
        expected = frozenset() if failure == 'error' else {events[0]['_event_id']}
        assert ledgers[-1].eligible_ids == expected
        assert not ledgers[-1]._pending
    finally:
        await loop.aclose()


async def test_unbound_source_hook_mutation_stops_before_transport(engine_config, monkeypatch):
    from nanobot.agent.hook import AgentHook

    class MutatingHook(AgentHook):
        async def before_iteration(self, context):
            context.messages.append({'role': 'user', 'content': 'unbound instruction'})

    provider = ScriptedProvider([])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop)
    run = loop.runner.run

    async def configure(spec):
        spec.hook = MutatingHook()
        return await run(spec)

    monkeypatch.setattr(loop.runner, 'run', configure)
    try:
        result = await loop.execute_turn(TurnRequest(content='request', session_key='cli:source'))
        assert result.error and 'bound source projection' in result.error
        assert provider.calls == []
    finally:
        await loop.aclose()


async def test_cancelled_source_request_discards_ticket(engine_config, monkeypatch):
    import asyncio

    class CancelledProvider(ScriptedProvider):
        async def chat(self, *args, **kwargs):
            raise asyncio.CancelledError()

    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=CancelledProvider([]))
    upgrade(loop)
    ledgers = []
    prepare = loop.runner.context_governor.prepare_request

    async def observe(state, messages, **kwargs):
        ledgers.append(state.source)
        return await prepare(state, messages, **kwargs)

    monkeypatch.setattr(loop.runner.context_governor, 'prepare_request', observe)
    try:
        with pytest.raises(asyncio.CancelledError):
            await loop.execute_turn(TurnRequest(content='request', session_key='cli:source'))
        assert not ledgers[0].eligible_ids and not ledgers[0]._pending
    finally:
        await loop.aclose()


@pytest.mark.parametrize('scope', ['root', 'incompatible'])
async def test_real_source_turn_reuses_scoped_checkpoint_after_restart(engine_config, scope):
    from datetime import datetime, timezone

    from nanobot.session.source import prefix_digest

    first_provider = ScriptedProvider([LLMResponse(content='old evidence')])
    first = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=first_provider)
    backend = upgrade(first)
    try:
        result = await first.execute_turn(TurnRequest(content='old private detail', session_key='cli:source'))
        assert result.error is None
    finally:
        await first.aclose()
    snapshot = backend.source.load_source_snapshot('cli:source')
    proposal = {
        'schema_version': 1, 'digest_version': 1, 'summary_format_version': 1,
        'session_generation': snapshot.state.generation, 'source_revision': snapshot.state.revision,
        'checkpoint_id': 'fixture-checkpoint', 'parent_checkpoint_id': snapshot.state.checkpoint_id,
        'producing_model': 'fixture', 'producing_provider': 'fixture',
        'digest_algorithm': 'sha256', 'created_at': datetime.now(timezone.utc).isoformat(),
        'summary': 'retained historical facts', 'artifact_references': [], 'boundary': len(snapshot.events),
        'boundary_event_id': snapshot.events[-1]['_event_id'],
        'prefix_digest': prefix_digest(snapshot.events, len(snapshot.events)), 'auxiliary_usage': None,
        'projection_scope': scope, 'projection_version': 1,
    }
    backend.source.commit_context_checkpoint('cli:source', snapshot.state, proposal)
    provider = ScriptedProvider([LLMResponse(content='new answer'), LLMResponse(content='fresh answer')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        result = await loop.execute_turn(TurnRequest(content='new instruction', session_key='cli:source'))
        assert result.error is None
        rows = provider.calls[0][0]
        if scope == 'root':
            assert 'retained historical facts' in str(rows)
            assert 'old private detail' not in str(rows)
            assert 'retained historical facts' not in rows[0]['content']
            assert [row['content'] for row in rows if row['role'] == 'user'] == ['new instruction']
        else:
            assert 'retained historical facts' not in str(rows)
            assert 'old private detail' in str(rows)
        result = await loop.execute_turn(TurnRequest(content='fresh instruction', session_key='cli:source',
                                                    history_mode=HistoryMode.FRESH))
        assert result.error is None
        assert 'retained historical facts' not in str(provider.calls[1][0])
        assert 'old private detail' not in str(provider.calls[1][0])
        assert backend.source.load_source_snapshot('cli:source').events[:len(snapshot.events)] == snapshot.events
        assert len(provider.calls) == 2  # Reuse performs no auxiliary model request.
    finally:
        await loop.aclose()


@pytest.mark.parametrize('replace_at', [0, 1, 2])
async def test_source_tickets_follow_governor_resulting_view(engine_config, monkeypatch, replace_at):
    from nanobot.session.source_projection import CheckpointNode, RuntimeNode, SourceNode
    from nanobot.session.source_request import SourceRequestLedger, SourceTranscript

    def malformed(text):
        return LLMResponse(content=text, finish_reason='tool_calls',
            tool_calls=[ToolCallRequest(id='bad', name='', arguments={})])

    provider = ScriptedProvider([malformed('first'), malformed('second'), LLMResponse(content='done')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    prepare = loop.runner.context_governor.prepare_request
    issued = []
    issue = SourceRequestLedger.issue
    calls = 0
    observed_state = []
    resulting_views = []

    def observe_ticket(ledger, view):
        ticket = issue(ledger, view)
        issued.append(ticket)
        return ticket

    async def replace_projection(state, messages, **kwargs):
        nonlocal calls
        view = kwargs['source_view']
        notes = state.source.validate_view(view)
        # A synthetic publication isolates the runner/governor view handoff.
        # Durable checkpoint generation/CAS is tested by its own integration.
        if calls == replace_at:
            active = state.source.transcript
            nodes = list(active.nodes)
            insertion = next(i for i, node in enumerate(nodes) if isinstance(node, SourceNode))
            nodes.insert(insertion, CheckpointNode('replacement', 'historical facts'))
            state.source.replace_projection(SourceTranscript(tuple(nodes)))
            if kwargs['transcript'] is not None:
                kwargs['transcript'][:] = state.source.transcript.messages()
            view = state.source.transcript.with_runtime(*notes)
            messages = view.messages()
            kwargs['source_view'] = view
        calls += 1
        observed_state.append(state)
        result = await prepare(state, messages, **kwargs)
        assert state.source_view == view
        resulting_views.append(view)
        assert all(isinstance(node, RuntimeNode) for node in notes)
        return result

    monkeypatch.setattr(SourceRequestLedger, 'issue', observe_ticket)
    monkeypatch.setattr(loop.runner.context_governor, 'prepare_request', replace_projection)
    try:
        result = await loop.execute_turn(TurnRequest(content='request', session_key='cli:source'))
        assert result.error is None, result.error
        assert result.content == 'done'
        assert calls == len(issued) == len(provider.calls) == 3
        assert [ticket.checkpoint_id for ticket in issued] == [
            None if index < replace_at else 'replacement' for index in range(3)]
        state = observed_state[-1]
        assert not state.source._pending
        assert sum(isinstance(node, RuntimeNode) for node in state.source.transcript.nodes) == 1
        assert sum(isinstance(node, RuntimeNode) for node in state.source_view.nodes) == 3
        assert [view.messages() for view in resulting_views] == [call[0] for call in provider.calls]
        events = backend.source.load_source_snapshot('cli:source').events
        assert [event['content'] for event in events] == ['request', 'first', 'second', 'done']
        assert state.source.eligible_ids == {events[0]['_event_id']}
    finally:
        await loop.aclose()


async def test_source_compaction_scope_and_live_transcript_follow_admission(engine_config, monkeypatch):
    provider = ScriptedProvider([LLMResponse(content='root'), LLMResponse(content='fresh')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop)
    prepare = loop.runner.context_governor.prepare_request
    scopes = []

    async def observe(state, messages, **kwargs):
        compaction = state.source_compaction
        assert compaction.raw_messages is kwargs['transcript']
        assert compaction.raw_messages == state.source.transcript.messages()
        scopes.append((compaction.stream_id, compaction.writer.stream_id))
        return await prepare(state, messages, **kwargs)

    monkeypatch.setattr(loop.runner.context_governor, 'prepare_request', observe)
    try:
        for mode in (HistoryMode.SESSION, HistoryMode.FRESH):
            result = await loop.execute_turn(TurnRequest(content='input', session_key='cli:source', history_mode=mode))
            assert result.error is None, result.error
        assert scopes[0][0] is None
        assert scopes[1][0] == scopes[1][1]
        assert scopes[0][1] != scopes[1][1]
    finally:
        await loop.aclose()


async def test_real_pressure_compaction_publishes_checkpoint_and_reuses_after_restart(engine_config, monkeypatch):
    from nanobot.providers.base import LLMUsage

    class PressureProvider(ScriptedProvider):
        def estimate_prompt_tokens(self, messages, tools, model):
            auxiliary = messages[0]['content'].startswith('Summarize historical evidence')
            return ((100 if auxiliary else 5000) + str(messages).count('old_fact'), 'fixture')

    engine_config.agents.defaults.context_window_tokens = 12000
    engine_config.agents.defaults.max_tokens = 256
    provider = PressureProvider([
        LLMResponse(content='old_fact ' * 5000),
        LLMResponse(content='retained historical facts', usage=LLMUsage(
            input_tokens=50, output_tokens=4, total_tokens=54, reported_tokens=54)),
        LLMResponse(content='new answer'),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    states = []
    prepare = loop.runner.context_governor.prepare_request

    async def observe(state, messages, **kwargs):
        states.append(state)
        result = await prepare(state, messages, **kwargs)
        assert state.source_compaction.raw_messages == state.source.transcript.messages()
        return result

    monkeypatch.setattr(loop.runner.context_governor, 'prepare_request', observe)
    try:
        first = await loop.execute_turn(TurnRequest(content='old objective', session_key='cli:source'))
        assert first.error is None, first.error
        prefix = backend.source.load_source_snapshot('cli:source').events
        second = await loop.execute_turn(TurnRequest(content='current objective', session_key='cli:source'))
        assert second.error is None, second.error
        assert second.content == 'new answer'
        assert len(provider.calls) == 3
        assert 'old_fact' in str(provider.calls[1][0])  # Auxiliary canonical evidence.
        assert 'old_fact' not in str(provider.calls[2][0])
        assert [row['content'] for row in provider.calls[2][0] if row['role'] == 'user'] == ['current objective']
        assert 'retained historical facts' not in provider.calls[2][0][0]['content']
        saved = backend.source.load_source_snapshot('cli:source')
        assert saved.events[:len(prefix)] == prefix and len(saved.events) == 4
        assert saved.checkpoint['summary'] == 'retained historical facts'
        assert saved.checkpoint['auxiliary_usage']['total_tokens'] == 54
        assert states[-1].source_compaction.attempts[-1].calls == 1
        assert not states[-1].source.has_pending_requests
    finally:
        await loop.aclose()
    restarted_provider = PressureProvider([LLMResponse(content='resumed')])
    restarted = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=restarted_provider)
    try:
        result = await restarted.execute_turn(TurnRequest(content='next objective', session_key='cli:source'))
        assert result.error is None, result.error
        assert result.content == 'resumed' and len(restarted_provider.calls) == 1
        rows = restarted_provider.calls[0][0]
        assert 'retained historical facts' in str(rows) and 'old_fact' not in str(rows)
    finally:
        await restarted.aclose()


@pytest.mark.parametrize('mode', ['fresh', 'child'])
async def test_real_local_pressure_compaction_preserves_root_and_continues_length_recovery(engine_config, monkeypatch, mode):
    from copy import deepcopy

    from nanobot.agent.context_governance import ContextGovernor
    from nanobot.utils.runtime import build_length_recovery_message

    class LocalPressureProvider(ScriptedProvider):
        async def chat(self, messages, **kwargs):
            return await super().chat(deepcopy(messages), **kwargs)

        def estimate_prompt_tokens(self, messages, tools, model):
            auxiliary = messages[0]['content'].startswith('Summarize historical evidence')
            text = str(messages) if auxiliary else str([
                row for row in messages if row['role'] == 'assistant'])
            return ((100 if auxiliary else 1000) + 8000 * text.count('first_segment')
                    + 1000 * text.count('second_segment'), 'fixture')

    engine_config.agents.defaults.context_window_tokens = 12000
    engine_config.agents.defaults.max_tokens = 256
    responses = [LLMResponse(content='root original')]
    if mode == 'child':
        responses.append(LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id='plan', name='plan_task', arguments={'objective': 'study locally'})]))
    responses.extend([
        LLMResponse(content='first_segment', finish_reason='length'),
        LLMResponse(content='second_segment', finish_reason='length'),
        LLMResponse(content='local historical facts'),
        LLMResponse(content='local done'),
    ])
    if mode == 'child':
        responses.append(LLMResponse(content='parent done'))
    provider = LocalPressureProvider(responses)
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    local_states = []
    local_requests = []
    prepare = ContextGovernor.prepare_request

    async def observe(governor, state, messages, **kwargs):
        result = await prepare(governor, state, messages, **kwargs)
        if state.source_compaction.stream_id is not None:
            local_states.append(state)
            local_requests.append(deepcopy(result[0]))
            assert state.source_compaction.raw_messages == state.source.transcript.messages()
        return result

    monkeypatch.setattr(ContextGovernor, 'prepare_request', observe)
    try:
        seed = await loop.execute_turn(TurnRequest(content='root private', session_key='cli:source'))
        assert seed.error is None, seed.error
        before = backend.source.load_source_snapshot('cli:source')
        result = await loop.execute_turn(TurnRequest(content='study locally', session_key='cli:source',
            history_mode=HistoryMode.FRESH if mode == 'fresh' else HistoryMode.SESSION))
        assert result.error is None, result.error
        assert result.content.endswith('local done' if mode == 'fresh' else 'parent done')
        assert not provider.responses
        assert len(local_states) == 3
        local_indices = (1, 2, 4) if mode == 'fresh' else (2, 3, 5)
        assert local_requests == [provider.calls[index][0] for index in local_indices]
        assert all('root private' not in str(rows) and 'root original' not in str(rows)
                   for rows in local_requests)
        assert local_requests[-1].count(build_length_recovery_message('first_segment')) == 1
        assert local_requests[-1].count(build_length_recovery_message('second_segment')) == 1
        state = local_states[-1]
        assert len(state.source_compaction.attempts) == 1
        assert state.source_compaction.attempts[0].calls == 1
        assert state.source_compaction.local_checkpoint.summary == 'local historical facts'
        assert not state.source.has_pending_requests
        assert not any(row.get('role') == 'assistant' and row.get('content') == 'first_segment'
                       for row in state.messages)
        assert 'second_segment' in str(state.messages)
        assert 'local historical facts' in str(state.messages)
        assert 'root private' not in str(state.messages)
        after = backend.source.load_source_snapshot('cli:source')
        assert after.checkpoint == before.checkpoint
        assert after.events[:len(before.events)] == before.events
        for segment in ('first_segment', 'second_segment'):
            persisted = [row for row in after.events if row.get('content') == segment]
            assert len(persisted) == 1
            assert persisted[0]['_stream_id'] == state.source_compaction.stream_id
        if mode == 'child':
            intents = [row for row in after.events if any(
                call.get('id') == 'plan' for call in row.get('tool_calls', []))]
            results = [row for row in after.events if row.get('tool_call_id') == 'plan']
            assert len(intents) == len(results) == 1
            assert results[0]['_execution_id'] == intents[0]['_execution_ids'][0]
        assert not validate_execution_links(after.events)
        summaries = [call[0] for call in provider.calls if call[0][0]['content'].startswith('Summarize historical evidence')]
        assert len(summaries) == 1
        assert 'root private' not in str(summaries[0])
        assert 'first_segment' in str(summaries[0])
        assert 'second_segment' not in str(summaries[0])
    finally:
        await loop.aclose()


async def test_read_receipts_distinguish_repeated_provider_call_ids(engine_config, monkeypatch):
    from nanobot.agent.tools.read_provenance import parse_read_receipt

    for name in ('first', 'second'):
        (engine_config.workspace_path / name).write_text(name)
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(id='same', name='read_file',
            arguments={'path': name}) for name in ('first', 'second')]),
        LLMResponse(content='done'),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop)
    forbid_legacy(loop, monkeypatch)
    try:
        result = await loop.execute_turn(TurnRequest(content='read both', session_key='cli:source'))
        assert result.error is None, result.error
        events = backend.source.load_source_snapshot('cli:source').events
        results = [event for event in events if event['role'] == 'tool']
        assert len(results) == 2
        assert results[0]['_execution_id'] != results[1]['_execution_id']
        for name, event in zip(('first', 'second'), results):
            receipt = parse_read_receipt(event['_source_origin'])
            assert receipt.source_access.relative_path == name and receipt.matches(event['content'])
    finally:
        await loop.aclose()
