"""Trusted source turn identities survive runner batches and persistence failures."""

import pytest

from nanobot.agent.tools.execution import KnownToolResult
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source import SourceValidationError, validate_execution_links
from nanobot.session.turn_writer import SourceTurnWriter


@pytest.fixture
def store(tmp_path):
    backend = JsonlSessionStore(tmp_path)
    session = Session('test:writer')
    backend.save(session)
    backend.source.upgrade_source_session(session.key, expected_revision=session.revision,
        expected_generation=session.generation, expected_incarnation=session.incarnation)
    return backend


def assistant():
    return {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'same', 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}
        for name in ('one', 'two')], '_event_id': 'forged', '_execution_ids': ['bad', 'bad']}


def result(ordinal=0, content='raw'):
    return KnownToolResult(ordinal, 'same', ('one', 'two')[ordinal], content, content,
                           {'name': ('one', 'two')[ordinal], 'status': 'ok'}, True)


async def test_writer_allocates_trusted_events_and_distinct_execution_links(store):
    async with acquire_execution(store, 'test:writer', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        user = writer.append_message({'role': 'user', 'content': 'input', '_run_id': 'forged'})
        intent = writer.begin_tool_batch(assistant())
        first_id = await writer.record_tool_result_event(result())
        second_id = await writer.record_tool_result_event(result(1, 'second'))
        final = writer.append_message({'role': 'assistant', 'content': 'done'})
        snapshot = owner.read_snapshot()
        assert [e['role'] for e in snapshot.events] == ['user', 'assistant', 'tool', 'tool', 'assistant']
        assert all(e['_run_id'] == 'run' for e in snapshot.events)
        assert user['_event_id'] != intent['_event_id'] != final['_event_id']
        assert len(set(intent['_execution_ids'])) == 2 and 'bad' not in intent['_execution_ids']
        assert not validate_execution_links(snapshot.events)
        assert snapshot.events[2]['_execution_id'] == intent['_execution_ids'][0]
        assert snapshot.events[3]['_execution_id'] == intent['_execution_ids'][1]
        assert first_id == snapshot.events[2]['_event_id']
        assert second_id == snapshot.events[3]['_event_id'] and first_id != second_id


async def test_result_binding_and_duplicate_callback_fail_without_writes(store):
    async with acquire_execution(store, 'test:writer', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        writer.begin_tool_batch(assistant())
        wrong = KnownToolResult(0, 'other', 'one', 'raw', 'raw', {}, True)
        before = owner.read_snapshot()
        with pytest.raises(SourceValidationError):
            await writer.record_tool_result(wrong)
        assert owner.read_snapshot().events == before.events


async def test_failed_write_poisoned_writer_cannot_append_or_dispatch_more(store, monkeypatch):
    async with acquire_execution(store, 'test:writer', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        writer.begin_tool_batch(assistant())
        with monkeypatch.context() as patch:
            patch.setattr(type(store.source), 'append_source_events', lambda *a, **kw: (_ for _ in ()).throw(OSError('disk')))
            with pytest.raises(OSError):
                await writer.record_tool_result(result())
        with pytest.raises(SourceValidationError):
            writer.append_message({'role': 'assistant', 'content': 'must not continue'})
        assert len(owner.read_snapshot().events) == 1


async def test_child_serialization_failure_poisoning_stops_parent(store):
    async with acquire_execution(store, 'test:writer', run_id='run') as owner:
        parent = SourceTurnWriter(store.source, owner, run_id='run')
        parent.begin_tool_batch(assistant())
        child = parent.child_stream()
        child.begin_tool_batch(assistant())
        with pytest.raises(SourceValidationError):
            await child.record_tool_result(result(content=object()))
        with pytest.raises(SourceValidationError):
            await parent.record_tool_result(result())
        assert len(owner.read_snapshot().events) == 2


@pytest.mark.parametrize('observer', ['finalize', 'stream_end', 'reasoning', 'length'])
async def test_known_assistant_is_durable_before_observer_failure(store, observer):
    from unittest.mock import AsyncMock, MagicMock

    from agent.runner_helpers import make_run_spec

    from nanobot.agent.hook import AgentHook
    from nanobot.agent.runner import AgentRunner
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.agent.turn import TurnContext
    from nanobot.providers.base import LLMResponse

    class Broken(AgentHook):
        def finalize_content(self, context, content):
            if observer in {'finalize', 'length'}:
                raise RuntimeError('observer')
            return content
        def wants_streaming(self):
            return observer == 'stream_end'
        async def on_stream_end(self, *args, **kwargs):
            raise RuntimeError('observer')
        async def emit_reasoning(self, *args):
            raise RuntimeError('observer')
    response = LLMResponse(content='known response', finish_reason='length' if observer == 'length' else 'stop',
                           reasoning_content='private reasoning' if observer == 'reasoning' else None)
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=response)
    provider.chat_stream_with_retry = AsyncMock(return_value=response)
    async with acquire_execution(store, 'test:writer', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        writer.append_message({'role': 'user', 'content': 'input'})
        with pytest.raises(RuntimeError, match='observer'):
            await AgentRunner().run(make_run_spec(provider, initial_messages=[{'role': 'user', 'content': 'input'}],
                tools=ToolRegistry(), model='fixture', max_iterations=1, max_tool_result_chars=100,
                source_writer=writer, turn_context=TurnContext(execution_owner=owner), hook=Broken()))
        events = owner.read_snapshot().events
        assert [e['content'] for e in events] == ['input', 'known response']
        assert 'private reasoning' not in str(events)


def test_projection_preserves_media_and_removes_only_marked_runtime_context():
    from nanobot.runtime_context import RUNTIME_CONTEXT_HISTORY_META
    events = [{'_event_id': 'one', 'role': 'user', 'content': 'input\n\nold runtime',
               'media': ['/fixture/image.png'], RUNTIME_CONTEXT_HISTORY_META: {'version': 1, 'suffix': 'old runtime'}},
              {'_event_id': 'two', '_parent_stream_id': 'root', 'role': 'assistant', 'content': 'private child'}]
    from copy import deepcopy
    original = deepcopy(events)
    messages, identities = SourceTurnWriter.project_events(events)
    assert identities == ['one']
    assert 'old runtime' not in str(messages) and 'private child' not in str(messages)
    assert '/fixture/image.png' in str(messages)
    assert events == original


async def test_multimodal_source_retains_text_without_inline_image_or_reasoning(store):
    async with acquire_execution(store, 'test:writer', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        writer.begin_tool_batch(assistant())
        await writer.record_tool_result(result(content=[
            {'type': 'text', 'text': 'visible evidence'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,private-image'}},
            {'type': 'thinking', 'thinking': 'private reasoning'},
        ]))
        content = owner.read_snapshot().events[-1]['content']
        assert content[0]['text'] == 'visible evidence'
        assert 'unavailable' in content[1]['text']
        assert 'private-image' not in str(content) and 'private reasoning' not in str(content)


async def test_source_runner_requires_writer_dispatch_owner_match(store):
    from unittest.mock import AsyncMock, MagicMock

    from agent.runner_helpers import make_run_spec

    from nanobot.agent.runner import AgentRunner
    from nanobot.agent.tools.registry import ToolRegistry

    provider = MagicMock()
    provider.chat_with_retry = AsyncMock()
    async with acquire_execution(store, 'test:writer', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        with pytest.raises(ValueError, match='same dispatch owner'):
            await AgentRunner().run(make_run_spec(provider, initial_messages=[], tools=ToolRegistry(),
                model='fixture', max_iterations=1, max_tool_result_chars=100, source_writer=writer))
        provider.chat_with_retry.assert_not_awaited()


async def test_writer_strips_forged_origin_and_preserves_changed_inline_result(store):
    from dataclasses import replace

    from nanobot.agent.tools.read_provenance import FilesystemReadReceipt
    from nanobot.session.artifact import parse_artifact_descriptor
    from session.test_artifact import fixture

    raw, _ = fixture()
    descriptor = parse_artifact_descriptor(raw)
    receipt = FilesystemReadReceipt(descriptor.source_access, 'a' * 64,
                                    descriptor.byte_length, descriptor.sha256)
    async with acquire_execution(store, 'test:writer', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        user = writer.append_message({'role': 'user', 'content': 'input',
                                      '_source_origin': receipt.to_dict()})
        assert '_source_origin' not in user
        payload = assistant()
        payload['_source_origin'] = receipt.to_dict()
        intent = writer.begin_tool_batch(payload)
        assert '_source_origin' not in intent
        await writer.record_tool_result(replace(result(content='changed'), producer_receipt=receipt))
        event = owner.read_snapshot().events[-1]
        assert event['content'] == 'changed' and '_source_origin' not in event
        await writer.record_tool_result(result(1))


@pytest.mark.parametrize(
    ('tool_name', 'producer'),
    [
        ('read_file', 'builtin_grep_file_text'),
        ('grep', 'builtin_read_file_text'),
    ],
)
async def test_writer_rejects_receipts_copied_between_canonical_producers(
    store, tool_name, producer,
):
    import hashlib

    from nanobot.agent.tools.read_provenance import FilesystemReadReceipt
    from nanobot.session.artifact import parse_artifact_descriptor
    from session.test_artifact import fixture

    raw, _ = fixture()
    descriptor = parse_artifact_descriptor(raw)
    content = 'canonical tool output'
    receipt = FilesystemReadReceipt(
        descriptor.source_access,
        'a' * 64,
        len(content.encode()),
        hashlib.sha256(content.encode()).hexdigest(),
        producer=producer,
    )
    payload = {
        'role': 'assistant',
        'content': '',
        'tool_calls': [{
            'id': 'call',
            'type': 'function',
            'function': {'name': tool_name, 'arguments': '{}'},
        }],
    }
    async with acquire_execution(store, 'test:writer', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        writer.begin_tool_batch(payload)
        await writer.record_tool_result(KnownToolResult(
            0, 'call', tool_name, content, content, {'status': 'ok'}, True, receipt,
        ))
        event = owner.read_snapshot().events[-1]
        assert event['content'] == content
        assert '_source_origin' not in event
