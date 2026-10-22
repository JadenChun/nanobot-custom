"""Pressure orchestration: bounded calls, canonical chunks, and failure isolation."""

import asyncio
from dataclasses import replace

import pytest

from nanobot.agent.context_governance import ContextGovernor, ContextWindowExceededError
from nanobot.providers.base import LLMResponse, LLMUsage
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import SessionWriteConflict
from nanobot.session.source import SourceValidationError
from nanobot.session.source_projection import RuntimeNode, source_nodes
from nanobot.session.source_request import SourceRequestLedger, SourceTranscript
from nanobot.session.turn_writer import SourceTurnWriter
from tests.agent.test_source_compaction_publication import state_for
from tests.agent.test_source_compaction_publication import store as store


def pressure_state(writer, *, chunks=3, local=False):
    state = state_for(writer, local=local)
    for index in range(chunks):
        writer.append_message({'role': 'assistant', 'content': f'chunk-{index}'})
    writer.append_message({'role': 'user', 'content': 'current instruction'})
    nodes = source_nodes(writer.owner.read_snapshot().events, stream_id=writer.stream_id if local else None)
    active = SourceTranscript((RuntimeNode('policy', {'role': 'system', 'content': 'policy'}), *nodes))
    state.source = SourceRequestLedger(active, initial_eligible_ids=frozenset(n.event_id for n in nodes[:-1]))
    state.source_compaction.raw_messages[:] = active.messages()
    state.config.context_window_tokens = 12000
    state.source_compaction.runtime = replace(state.source_compaction.runtime, context_window_tokens=12000)
    provider = state.config.provider
    provider.response = LLMResponse(content='retained facts', usage=LLMUsage(
        input_tokens=10, output_tokens=2, total_tokens=12, reported_tokens=12))

    def count(messages, tools, model):
        auxiliary = messages[0]['content'].startswith('Summarize historical evidence')
        text = str(messages)
        return ((100 if auxiliary else 5200) + text.count('chunk-') * (6000 if auxiliary else 1500), 'fixture')

    provider.estimate_prompt_tokens = count
    return state


async def prepare(state):
    return await ContextGovernor().prepare_request(state, state.source.transcript.messages(),
        source_view=state.source.transcript, transcript=state.source_compaction.raw_messages,
        tool_definitions=None)


@pytest.mark.parametrize('local', [False, True])
async def test_pressure_summarizes_disjoint_chunks_and_accounts_every_call(store, local):
    import json

    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        state = pressure_state(writer, local=local)
        before = owner.read_snapshot()
        messages, context = await prepare(state)
        assert context is not None
        metrics = state.source_compaction.attempts[-1]
        assert metrics.calls == 3 and metrics.status == 'succeeded'
        assert metrics.usage.total_tokens == 36 and not metrics.usage_unknown
        assert metrics.after_tokens <= ContextGovernor.input_budget(state.config) * 0.6
        assert metrics.latency_ms >= 0
        assert 'chunk-' not in str(messages) and messages[-1]['content'] == 'current instruction'
        assert state.source_compaction.raw_messages == state.source.transcript.messages()
        assert state.messages == state.source_view.messages() == messages
        payloads = [json.loads(call[0][1]['content']) for call in state.config.provider.calls]
        evidence = [row['content'] for payload in payloads for row in payload['events']]
        assert [text for text in evidence if text.startswith('chunk-')] == ['chunk-0', 'chunk-1', 'chunk-2']
        assert 'current instruction' not in evidence
        assert [payload['previous_summary'] for payload in payloads] == [None, 'retained facts', 'retained facts']
        after = owner.read_snapshot()
        assert after.events == before.events
        if local:
            assert after == before and state.source_compaction.local_checkpoint is not None
        else:
            assert after.checkpoint['auxiliary_usage']['total_tokens'] == 36
            assert after.checkpoint['boundary'] == len(before.events) - 1
        assert not state.source.has_pending_requests  # Main transport issues the ticket later.


async def test_four_call_cap_does_not_publish_oversized_partial_projection(store):
    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        state = pressure_state(SourceTurnWriter(store.source, owner, run_id='run'), chunks=8)
        before, active = owner.read_snapshot(), state.source.transcript
        with pytest.raises(ContextWindowExceededError):
            await prepare(state)
        assert len(state.config.provider.calls) == 4
        metrics = state.source_compaction.attempts[-1]
        assert metrics.calls == 4 and metrics.usage.total_tokens == 48 and metrics.status == 'failed'
        assert owner.read_snapshot() == before and state.source.transcript == active


@pytest.mark.parametrize('failure', ['invalid', 'cancelled', 'metadata'])
async def test_failed_attempt_retains_old_checkpoint_and_reports_cost(store, failure):
    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        state = pressure_state(SourceTurnWriter(store.source, owner, run_id='run'))
        before, active = owner.read_snapshot(), state.source.transcript
        if failure == 'invalid':
            state.config.provider.response = LLMResponse(content='', usage=LLMUsage.empty_request())
        elif failure == 'cancelled':
            state.config.provider.response = asyncio.CancelledError()
        else:
            chat = state.config.provider.chat

            async def mutate(*args, **kwargs):
                store.source.update_metadata('test:compact', {'title': 'new title'})
                return await chat(*args, **kwargs)

            state.config.provider.chat = mutate
        with pytest.raises((SourceValidationError, SessionWriteConflict, asyncio.CancelledError)):
            await prepare(state)
        after = owner.read_snapshot()
        assert after.events == before.events and after.checkpoint == before.checkpoint
        assert state.source.transcript == active and not state.source.has_pending_requests
        metrics = state.source_compaction.attempts[-1]
        assert metrics.calls == (3 if failure == 'metadata' else 1)
        assert metrics.status == ('cancelled' if failure == 'cancelled' else 'failed')
        assert metrics.usage_unknown == (failure == 'cancelled')


async def test_oversized_fixed_request_never_starts_auxiliary_transport(store):
    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        state = pressure_state(SourceTurnWriter(store.source, owner, run_id='run'))
        state.config.provider.estimate_prompt_tokens = lambda *args: (15000, 'fixture')
        before = owner.read_snapshot()
        with pytest.raises(ContextWindowExceededError):
            await prepare(state)
        assert state.config.provider.calls == [] and owner.read_snapshot() == before


async def test_fitting_early_group_does_not_spend_calls_before_impossible_later_group(store):
    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        state = pressure_state(SourceTurnWriter(store.source, owner, run_id='run'))
        count = state.config.provider.estimate_prompt_tokens

        def giant_group(messages, tools, model):
            auxiliary = messages[0]['content'].startswith('Summarize historical evidence')
            if auxiliary and 'chunk-2' in str(messages):
                return 30000, 'fixture indivisible oversized event'
            return count(messages, tools, model)

        state.config.provider.estimate_prompt_tokens = giant_group
        before, active = owner.read_snapshot(), state.source.transcript
        with pytest.raises(SourceValidationError, match='Indivisible source group'):
            await prepare(state)
        assert state.config.provider.calls == []
        assert owner.read_snapshot() == before and state.source.transcript == active


async def test_unknown_auxiliary_usage_preserves_known_counts_without_fabricating_total(store):
    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        state = pressure_state(SourceTurnWriter(store.source, owner, run_id='run'))
        chat = state.config.provider.chat

        async def mixed_usage(*args, **kwargs):
            response = await chat(*args, **kwargs)
            return replace(response, usage=None) if len(state.config.provider.calls) == 2 else response

        state.config.provider.chat = mixed_usage
        await prepare(state)
        metrics = state.source_compaction.attempts[-1]
        assert metrics.calls == 3 and metrics.usage_unknown
        assert metrics.usage.total_tokens == 24
        assert owner.read_snapshot().checkpoint['auxiliary_usage'] is None


async def test_repeated_pressure_compaction_keeps_live_instruction_and_advances_parent(store):
    import json

    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        state = pressure_state(writer)
        await prepare(state)
        first = owner.read_snapshot()
        first_checkpoint = first.checkpoint['checkpoint_id']
        state.source.acknowledge(state.source.issue(state.source_view))
        for index in range(3, 6):
            event = writer.append_message({'role': 'assistant', 'content': f'chunk-{index}'})
            state.source.append_source(event['_event_id'], {'role': 'assistant', 'content': f'chunk-{index}'})
        state.source_compaction.raw_messages[:] = state.source.transcript.messages()
        state.source.acknowledge(state.source.issue(state.source.transcript))
        messages, _ = await prepare(state)
        second = owner.read_snapshot()
        assert second.checkpoint['parent_checkpoint_id'] == first_checkpoint
        assert second.checkpoint['boundary'] > first.checkpoint['boundary']
        assert second.events[:len(first.events)] == first.events
        assert [row['content'] for row in messages if row['role'] == 'user'] == ['current instruction']
        payloads = [json.loads(call[0][1]['content']) for call in state.config.provider.calls[3:]]
        assert all('objective' not in str(payload['events']) for payload in payloads)
        assert payloads[0]['previous_summary'] == 'retained facts'
        assert [m.calls for m in state.source_compaction.attempts] == [3, 3]
        assert state.source.projection_epoch == 2
