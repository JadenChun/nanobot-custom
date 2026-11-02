"""Pressure orchestration: bounded calls, canonical chunks, and failure isolation."""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest
from openai import AsyncOpenAI

from nanobot.agent.context_governance import (
    ContextGovernanceConfig,
    ContextGovernor,
    ContextWindowExceededError,
    ModelRequestState,
    SourceCompactionState,
    SourceCompactionUnavailableError,
)
from nanobot.agent.tools.execution import KnownToolResult
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import ToolOutcome
from nanobot.providers.base import (
    GenerationSettings,
    LLMResponse,
    LLMUsage,
    ProviderCallContext,
)
from nanobot.providers.conversation_state import ProviderConversationStateController
from nanobot.providers.key_pool import KeyPoolCoordinator
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import find_by_name
from nanobot.providers.request_budget import derive_provider_request_limits
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import SessionWriteConflict
from nanobot.session.source import SourceValidationError, prefix_digest
from nanobot.session.source_projection import (
    CheckpointNode,
    RuntimeNode,
    SourceNode,
    project_source,
    source_nodes,
)
from nanobot.session.source_request import SourceRequestLedger, SourceTranscript
from nanobot.session.turn_writer import SourceTurnWriter
from nanobot.utils.llm_runtime import LLMRuntime
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


def install_test_stream_dispatch(provider, *, after_response=None, observe_context=None):
    stream_with_context = provider.chat_stream_with_context

    async def dispatch(*, provider_context, **kwargs):
        if observe_context is not None:
            observe_context(provider_context)
        if provider_context.on_transport_attempt is not None:
            provider_context.on_transport_attempt('test-stream')
        response = await stream_with_context(provider_context=provider_context, **kwargs)
        return after_response(response) if after_response is not None else response

    provider.chat_stream_with_context = dispatch


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
            install_test_stream_dispatch(state.config.provider)
        else:
            def mutate(_response):
                store.source.update_metadata('test:compact', {'title': 'new title'})
                return _response

            install_test_stream_dispatch(state.config.provider, after_response=mutate)
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

        def mixed_usage(response):
            return replace(response, usage=None) if len(state.config.provider.calls) == 2 else response

        install_test_stream_dispatch(state.config.provider, after_response=mixed_usage)
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


@pytest.mark.parametrize(('tail_first_role', 'expect_prefix'), [('user', True), ('assistant', False)])
async def test_local_manager_prefix_then_disjoint_canonical_fold_uses_real_source_and_sdk(
    store, tail_first_role, expect_prefix,
):
    from nanobot.providers.base import PreparedRequestBudgetPreview

    model = 'Qwen3.8-27B-UD-Q6_K_XL'
    context_window = 12000
    generation = GenerationSettings(temperature=0.2, max_tokens=256, reasoning_effort='medium')
    provider = OpenAICompatProvider(
        api_key='fixture-credential', api_base='http://manager.test/v1',
        default_model=model, provider_name='local_manager',
        spec=find_by_name('local_manager'),
    )
    tools = [{
        'type': 'function',
        'function': {
            'name': 'read_image',
            'description': 'Read a historical image.',
            'parameters': {
                'type': 'object',
                'properties': {'path': {'type': 'string'}},
                'required': ['path'],
                'additionalProperties': False,
            },
        },
    }]

    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        eligible: set[str] = set()
        first_user = writer.append_message({
            'role': 'user', 'content': 'Compare the two historical images.',
            'media': ['/historical/a.png', '/historical/b.png'],
        })
        eligible.add(first_user['_event_id'])
        first_calls = [
            {'id': 'image-a', 'type': 'function',
             'function': {'name': 'read_image', 'arguments': '{"path":"a.png"}'}},
            {'id': 'image-b', 'type': 'function',
             'function': {'name': 'read_image', 'arguments': '{"path":"b.png"}'}},
        ]
        intent = writer.begin_tool_batch({
            'role': 'assistant', 'content': '', 'tool_calls': first_calls,
            'reasoning_content': 'PRIVATE_REASONING_SENTINEL',
        })
        eligible.add(intent['_event_id'])
        for ordinal, (call_id, content) in enumerate((
            ('image-a', 'Image A contains a blue vehicle.'),
            ('image-b', 'Image B contains a red vehicle.'),
        )):
            outcome = ToolOutcome(content=content)
            result_id = await writer.record_tool_result_event(KnownToolResult(
                ordinal, call_id, 'read_image', outcome, content, {'status': 'ok'}, True,
            ))
            eligible.add(result_id)
        historical_answer = writer.append_message({
            'role': 'assistant', 'content': 'The first images show different vehicles.',
            'reasoning_content': 'PRIVATE_REASONING_SENTINEL',
        })
        eligible.add(historical_answer['_event_id'])

        second_user = writer.append_message({
            'role': 'user', 'content': 'Compare the next two historical images.',
            'media': ['/historical/c.png', '/historical/d.png'],
        })
        eligible.add(second_user['_event_id'])
        second_calls = [
            {'id': 'image-c', 'type': 'function',
             'function': {'name': 'read_image', 'arguments': '{"path":"c.png"}'}},
            {'id': 'image-d', 'type': 'function',
             'function': {'name': 'read_image', 'arguments': '{"path":"d.png"}'}},
        ]
        second_intent = writer.begin_tool_batch({
            'role': 'assistant', 'content': '', 'tool_calls': second_calls,
            'reasoning_content': 'PRIVATE_REASONING_SENTINEL',
        })
        eligible.add(second_intent['_event_id'])
        for ordinal, (call_id, content) in enumerate((
            ('image-c', 'Image C contains a green vehicle.'),
            ('image-d', 'Image D contains a yellow vehicle.'),
        )):
            outcome = ToolOutcome(content=content)
            result_id = await writer.record_tool_result_event(KnownToolResult(
                ordinal, call_id, 'read_image', outcome, content, {'status': 'ok'}, True,
            ))
            eligible.add(result_id)
        second_historical_answer = writer.append_message({
            'role': 'assistant', 'content': 'The second images also show different vehicles.',
            'reasoning_content': 'PRIVATE_REASONING_SENTINEL',
        })
        eligible.add(second_historical_answer['_event_id'])
        chunk_ids = {}
        for content, role in (('chunk-0', tail_first_role), ('chunk-1', 'assistant')):
            event = writer.append_message({'role': role, 'content': content})
            eligible.add(event['_event_id'])
            chunk_ids[content] = event['_event_id']
        current = writer.append_message({
            'role': 'user', 'content': 'What changed in the current image?',
            'media': ['/current/image.png'],
        })
        writer.append_message({'role': 'assistant', 'content': 'PENDING_TAIL_SENTINEL'})

        active_nodes = list(source_nodes(owner.read_snapshot().events))
        for event_id, expansion in (
            (chunk_ids['chunk-0'], ' CHUNK0_ACTIVE_EXPANSION_SENTINEL'),
            (current['_event_id'], ' CURRENT_IMAGE_EXPANSION_SENTINEL'),
        ):
            node_index = next(i for i, node in enumerate(active_nodes) if node.event_id == event_id)
            node = active_nodes[node_index]
            active_nodes[node_index] = SourceNode(
                node.event_id,
                {**node.message, 'content': node.message['content'] + expansion},
                node.artifact,
            )
        system = RuntimeNode('actual-system-head', {'role': 'system', 'content': 'System policy.'})
        active = SourceTranscript((system, *active_nodes))
        raw_messages = active.messages()
        ledger = SourceRequestLedger(active, initial_eligible_ids=frozenset(eligible))
        runtime = LLMRuntime(provider, model, generation, context_window)
        state = ModelRequestState(
            ContextGovernanceConfig(
                provider=provider, model=model, tools=ToolRegistry(), workspace=None,
                session_key=owner.key, max_tool_result_chars=16000,
                context_window_tokens=context_window, max_tokens=generation.max_tokens,
                generation=generation, preserve_source=True,
            ),
            ProviderConversationStateController(
                provider=provider, model=model, messages=raw_messages, session_id=owner.key,
            ),
            source=ledger,
            source_compaction=SourceCompactionState(
                writer, runtime, raw_messages, None, overall_timeout_seconds=5.0,
            ),
        )

        view = state.source.transcript
        governor = ContextGovernor()
        attempt = governor.capture_source_compaction(state, view, tool_definitions=tools)
        assert attempt.candidate.boundaries == (1, 4, 5, 6, 9, 10, 11, 12)
        selected_prefix = attempt.candidate.select_historical_prefix(max_boundary=10)
        assert selected_prefix.boundary == 10
        assert selected_prefix.source_node_count == 10
        assert selected_prefix.transcript is not None

        limits = derive_provider_request_limits(context_window, generation.max_tokens)
        main_context = ProviderCallContext(
            request_limits=limits, context_window_tokens=context_window,
        )
        main_preview = provider.preview_request_budget(
            messages=view.messages(), tools=tools, model=model, generation=generation,
            provider_context=main_context, stream=True, tool_choice='auto',
        )
        from nanobot.agent.memory import Consolidator

        prefix_payload = Consolidator.source_prefix_summary_messages(
            selected_prefix.transcript.messages(),
        )
        fixed_view = attempt.candidate.rebuild(
            attempt.candidate.boundaries[-1],
            CheckpointNode('fixed-preview', 'x'),
        ).with_runtime(*attempt.request_notes)
        fixed_preview = provider.preview_request_budget(
            messages=fixed_view.messages(), tools=tools, model=model, generation=generation,
            provider_context=main_context, stream=True, tool_choice='auto',
        )
        fixed_tokens = fixed_preview.measurement.input_tokens
        budget = fixed_preview.measurement.input_limit_tokens or 0
        target = int(budget * 0.6)
        allowance = min(2048, max(1, (target if fixed_tokens < target else budget) - fixed_tokens))
        prefix_generation = GenerationSettings(
            temperature=generation.temperature,
            max_tokens=allowance,
            reasoning_effort=generation.reasoning_effort,
        )
        prefix_context = ProviderCallContext(
            request_limits=derive_provider_request_limits(context_window, allowance),
            context_window_tokens=context_window,
            source_summary_mode='prefix',
        )
        prefix_preview = provider.preview_request_budget(
            messages=prefix_payload, tools=tools, model=model, generation=prefix_generation,
            provider_context=prefix_context,
            stream=True, tool_choice='auto',
        )
        assert isinstance(main_preview, PreparedRequestBudgetPreview)
        assert isinstance(prefix_preview, PreparedRequestBudgetPreview)
        assert prefix_preview.converted_messages[-1] == prefix_payload[-1]
        prefix_rows = prefix_preview.converted_messages[:-1]
        converted_prefix_match = (
            main_preview.converted_messages[:len(prefix_rows)] == prefix_rows
        )
        assert prefix_preview.converted_tools == main_preview.converted_tools
        for field in ('model', 'tools', 'temperature', 'tool_choice'):
            assert prefix_preview.effective_body.get(field) == main_preview.effective_body.get(field)
        assert converted_prefix_match is expect_prefix

        captured: list[dict[str, object]] = []

        def sse_chunk(content, finish_reason):
            return {
                'id': 'chatcmpl-prefix-fixture', 'object': 'chat.completion.chunk',
                'created': 1, 'model': model,
                'choices': [{'index': 0, 'delta': {'content': content},
                             'finish_reason': finish_reason}],
            }

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            captured.append(body)
            summary = f'Summary fold {len(captured)}'
            chunks = [sse_chunk(summary, None), sse_chunk(None, 'stop')]
            payload = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks)
            return httpx.Response(
                200, headers={'content-type': 'text/event-stream'},
                content=(payload + 'data: [DONE]\n\n').encode(), request=request,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            provider._client = AsyncOpenAI(
                api_key='fixture-credential', base_url='http://manager.test/v1',
                http_client=http_client,
            )
            before = owner.read_snapshot()
            await governor.compact_source_request(
                state, view, tool_definitions=tools, force=True, stream=True,
                tool_choice='auto',
            )

        metrics = state.source_compaction.attempts[-1]
        assert metrics.status == 'succeeded'
        assert metrics.calls == len(captured)
        assert metrics.boundary == 12
        assert len(captured) == (2 if expect_prefix else 1)
        if expect_prefix:
            assert metrics.prefix_mode == 'historical_prefix'
            assert metrics.prefix_fallback_reason is None
            assert metrics.prefix_source_nodes == 10
            assert metrics.prefix_estimated_tokens == prefix_preview.measurement.input_tokens
            assert metrics.converted_prefix_match is True
            first_body, second_body = captured
            assert first_body['model'] == main_preview.effective_body['model'] == model
            assert first_body['tools'] == list(main_preview.converted_tools or ())
            assert first_body.get('temperature') == main_preview.effective_body.get('temperature')
            assert first_body == prefix_preview.effective_body
            tail_evidence = attempt.candidate.summary_evidence(12, after=10)
            tail_payload = Consolidator.source_summary_messages(
                [node.message for node in tail_evidence.nodes], 'Summary fold 1',
            )
            canonical_generation = replace(prefix_generation, reasoning_effort='none')
            canonical_context = ProviderCallContext(
                request_limits=derive_provider_request_limits(context_window, allowance),
                context_window_tokens=context_window,
                source_summary_mode='canonical',
            )
            tail_preview = provider.preview_request_budget(
                messages=tail_payload, tools=None, model=model,
                generation=canonical_generation, provider_context=canonical_context,
                stream=True, tool_choice=None,
            )
            assert second_body == tail_preview.effective_body
            assert second_body.get('tools') is None
            fold_payload = json.loads(tail_payload[1]['content'])
            assert fold_payload['previous_summary'] == 'Summary fold 1'
            assert fold_payload['events'] == [
                {'role': tail_first_role, 'content': 'chunk-0'},
                {'role': 'assistant', 'content': 'chunk-1'},
            ]
        else:
            assert metrics.prefix_mode == 'canonical'
            assert metrics.prefix_fallback_reason == 'converted_prefix_mismatch'
            assert metrics.prefix_source_nodes == 0
            assert metrics.prefix_estimated_tokens is None
            assert metrics.converted_prefix_match is False
            first_body = captured[0]
            canonical_evidence = attempt.candidate.summary_evidence(12)
            canonical_payload = Consolidator.source_summary_messages(
                [node.message for node in canonical_evidence.nodes], None,
            )
            canonical_generation = replace(prefix_generation, reasoning_effort='none')
            canonical_context = ProviderCallContext(
                request_limits=derive_provider_request_limits(context_window, allowance),
                context_window_tokens=context_window,
                source_summary_mode='canonical',
            )
            canonical_preview = provider.preview_request_budget(
                messages=canonical_payload, tools=None, model=model,
                generation=canonical_generation, provider_context=canonical_context,
                stream=True, tool_choice=None,
            )
            assert first_body == canonical_preview.effective_body
            assert first_body.get('tools') is None
            fold_payload = json.loads(canonical_payload[1]['content'])
            assert fold_payload['previous_summary'] is None
            assert fold_payload['events'][-2:] == [
                {'role': tail_first_role, 'content': 'chunk-0'},
                {'role': 'assistant', 'content': 'chunk-1'},
            ]
        all_requests = str(captured)
        assert 'PRIVATE_REASONING_SENTINEL' not in all_requests
        assert 'CHUNK0_ACTIVE_EXPANSION_SENTINEL' not in all_requests
        assert 'CURRENT_IMAGE_EXPANSION_SENTINEL' not in all_requests
        assert 'PENDING_TAIL_SENTINEL' not in all_requests
        after = owner.read_snapshot()
        assert after.events == before.events
        assert after.checkpoint['boundary'] == 12
        assert after.checkpoint['summary'] == f'Summary fold {len(captured)}'
        active_after = state.source.transcript.messages()
        assert 'CURRENT_IMAGE_EXPANSION_SENTINEL' in str(active_after)
        assert 'PENDING_TAIL_SENTINEL' in str(active_after)
        assert state.source_compaction.raw_messages == active_after


async def test_local_manager_prefix_uses_actual_body_for_quote_heavy_source(store):
    """Price the native prefix actually dispatched, not its larger canonical surrogate."""

    from nanobot.agent.memory import Consolidator
    from nanobot.providers.base import PreparedRequestBudgetPreview
    from nanobot.session.checkpoint_dependencies import SummaryDependencyProof

    model = 'Qwen3.8-27B-UD-Q6_K_XL'
    context_window = 32768
    generation = GenerationSettings(temperature=0.2, max_tokens=256, reasoning_effort='none')
    provider = OpenAICompatProvider(
        api_key='fixture-credential', api_base='http://manager.test/v1',
        default_model=model, provider_name='local_manager',
        spec=find_by_name('local_manager'),
    )
    historical_content = '\\' * 32000

    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        historical_user = writer.append_message({'role': 'user', 'content': historical_content})
        historical_answer = writer.append_message({
            'role': 'assistant', 'content': 'Observed synthetic quoted-code evidence.',
        })
        current = writer.append_message({'role': 'user', 'content': 'CURRENT_TAIL_SENTINEL'})
        writer.append_message({'role': 'assistant', 'content': 'PENDING_TAIL_SENTINEL'})

        persisted_events = owner.read_snapshot().events
        persisted_user = next(
            event for event in persisted_events
            if event['_event_id'] == historical_user['_event_id']
        )
        assert persisted_user['content'] == historical_content
        nodes = list(source_nodes(persisted_events))
        active = SourceTranscript((
            RuntimeNode('actual-system-head', {'role': 'system', 'content': 'System policy.'}),
            *nodes,
        ))
        raw_messages = active.messages()
        eligible = frozenset((historical_user['_event_id'], historical_answer['_event_id']))
        ledger = SourceRequestLedger(active, initial_eligible_ids=eligible)
        runtime = LLMRuntime(provider, model, generation, context_window)
        state = ModelRequestState(
            ContextGovernanceConfig(
                provider=provider, model=model, tools=ToolRegistry(), workspace=None,
                session_key=owner.key, max_tool_result_chars=16000,
                context_window_tokens=context_window, max_tokens=generation.max_tokens,
                generation=generation, preserve_source=True,
            ),
            ProviderConversationStateController(
                provider=provider, model=model, messages=raw_messages, session_id=owner.key,
            ),
            source=ledger,
            source_compaction=SourceCompactionState(
                writer, runtime, raw_messages, None, overall_timeout_seconds=5.0,
            ),
        )

        view = state.source.transcript
        governor = ContextGovernor()
        attempt = governor.capture_source_compaction(state, view, tool_definitions=None)
        selected_prefix = attempt.candidate.select_historical_prefix(max_boundary=2)
        assert selected_prefix.boundary == 2
        assert selected_prefix.source_node_count == 2
        assert selected_prefix.transcript is not None

        limits = derive_provider_request_limits(context_window, generation.max_tokens)
        main_context = ProviderCallContext(
            request_limits=limits, context_window_tokens=context_window,
        )
        main_preview = provider.preview_request_budget(
            messages=view.messages(), tools=None, model=model, generation=generation,
            provider_context=main_context, stream=True, tool_choice=None,
        )
        fixed_view = attempt.candidate.rebuild(
            attempt.candidate.boundaries[-1], CheckpointNode('fixed-preview', 'x'),
        ).with_runtime(*attempt.request_notes)
        fixed_preview = provider.preview_request_budget(
            messages=fixed_view.messages(), tools=None, model=model, generation=generation,
            provider_context=main_context, stream=True, tool_choice=None,
        )
        assert isinstance(main_preview, PreparedRequestBudgetPreview)
        assert isinstance(fixed_preview, PreparedRequestBudgetPreview)
        budget = fixed_preview.measurement.input_limit_tokens or 0
        fixed_tokens = fixed_preview.measurement.input_tokens
        target = int(budget * 0.6)
        allowance = min(2048, max(1, (target if fixed_tokens < target else budget) - fixed_tokens))

        prefix_payload = Consolidator.source_prefix_summary_messages(
            selected_prefix.transcript.messages(),
        )
        prefix_generation = GenerationSettings(
            temperature=generation.temperature,
            max_tokens=allowance,
            reasoning_effort=generation.reasoning_effort,
        )
        prefix_context = ProviderCallContext(
            request_limits=derive_provider_request_limits(context_window, allowance),
            context_window_tokens=context_window,
            source_summary_mode='prefix',
        )
        prefix_preview = provider.preview_request_budget(
            messages=prefix_payload, tools=None, model=model, generation=prefix_generation,
            provider_context=prefix_context, stream=True, tool_choice=None,
        )
        canonical_evidence = attempt.candidate.summary_evidence(selected_prefix.boundary)
        canonical_payload = Consolidator.source_summary_messages(
            [node.message for node in canonical_evidence.nodes], None,
        )
        canonical_generation = replace(prefix_generation, reasoning_effort='none')
        canonical_context = ProviderCallContext(
            request_limits=derive_provider_request_limits(context_window, allowance),
            context_window_tokens=context_window,
            source_summary_mode='canonical',
        )
        canonical_preview = provider.preview_request_budget(
            messages=canonical_payload, tools=None, model=model,
            generation=canonical_generation, provider_context=canonical_context,
            stream=True, tool_choice=None,
        )
        assert isinstance(prefix_preview, PreparedRequestBudgetPreview)
        assert isinstance(canonical_preview, PreparedRequestBudgetPreview)
        assert main_preview.measurement.fits
        assert prefix_preview.measurement.fits
        assert not canonical_preview.measurement.fits
        assert prefix_preview.measurement.input_tokens < canonical_preview.measurement.input_tokens
        assert canonical_evidence.nodes[0].message['content'] == historical_content

        captured: list[dict[str, object]] = []

        def sse_chunk(content, finish_reason):
            return {
                'id': 'chatcmpl-prefix-media-surrogate', 'object': 'chat.completion.chunk',
                'created': 1, 'model': model,
                'choices': [{'index': 0, 'delta': {'content': content},
                             'finish_reason': finish_reason}],
            }

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            captured.append(body)
            chunks = [sse_chunk('Retained quoted-code fact.', None), sse_chunk(None, 'stop')]
            payload = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks)
            return httpx.Response(
                200, headers={'content-type': 'text/event-stream'},
                content=(payload + 'data: [DONE]\n\n').encode(), request=request,
            )

        original_publish = governor.publish_source_compaction
        published_proofs = []

        def capture_publish(*args, proof=None, **kwargs):
            published_proofs.append(proof)
            return original_publish(*args, proof=proof, **kwargs)

        governor.publish_source_compaction = capture_publish
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            provider._client = AsyncOpenAI(
                api_key='fixture-credential', base_url='http://manager.test/v1',
                http_client=http_client,
            )
            before = owner.read_snapshot()
            await governor.compact_source_request(
                state, view, tool_definitions=None, force=True, stream=True,
            )

        metrics = state.source_compaction.attempts[-1]
        assert metrics.status == 'succeeded'
        assert metrics.calls == 1 == len(captured)
        assert metrics.boundary == 2
        assert metrics.prefix_mode == 'historical_prefix'
        assert metrics.prefix_fallback_reason is None
        assert metrics.prefix_source_nodes == 2
        assert metrics.prefix_estimated_tokens == prefix_preview.measurement.input_tokens
        assert metrics.converted_prefix_match is True
        assert captured[0] == prefix_preview.effective_body
        assert 'CURRENT_TAIL_SENTINEL' not in str(captured)
        assert 'PENDING_TAIL_SENTINEL' not in str(captured)

        assert len(published_proofs) == 1
        proof = published_proofs[0]
        assert type(proof) is SummaryDependencyProof
        assert proof.candidate == attempt.candidate
        assert len(proof.chunks) == 1
        chunk = proof.chunks[0]
        assert chunk.evidence.after == 0
        assert chunk.evidence.boundary == 2
        assert [node.event_id for node in chunk.evidence.nodes] == [
            historical_user['_event_id'], historical_answer['_event_id'],
        ]
        assert chunk.evidence.nodes[0].message['content'] == historical_content
        assert chunk.previous_summary is None
        assert chunk.summary == 'Retained quoted-code fact.'

        after = owner.read_snapshot()
        assert after.events == before.events
        assert after.checkpoint['boundary'] == 2
        assert after.checkpoint['prefix_digest'] == prefix_digest(before.events, 2)
        assert after.checkpoint['summary'] == 'Retained quoted-code fact.'
        active_after = state.source.transcript.messages()
        assert active_after[-2]['content'] == 'CURRENT_TAIL_SENTINEL'
        assert active_after[-1]['content'] == 'PENDING_TAIL_SENTINEL'
        assert state.source_compaction.raw_messages == active_after
        assert state.source.projection_epoch == 1
        assert not state.source.has_pending_requests
        assert current['_event_id'] not in {node.event_id for node in proof.chunks[0].evidence.nodes}


async def test_shared_overall_deadline_expires_across_progressing_summary_folds(store):
    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        state = pressure_state(SourceTurnWriter(store.source, owner, run_id='run'), chunks=3)
        state.source_compaction.overall_timeout_seconds = 0.08
        provider = state.config.provider
        started_calls = 0
        stream_with_context = provider.chat_stream_with_context

        async def slow_progress(*, provider_context, **kwargs):
            nonlocal started_calls
            started_calls += 1
            if provider_context.on_transport_attempt is not None:
                provider_context.on_transport_attempt('test-stream')
            await asyncio.sleep(0.05)
            return await stream_with_context(provider_context=provider_context, **kwargs)

        provider.chat_stream_with_context = slow_progress
        before, active = owner.read_snapshot(), state.source.transcript

        with pytest.raises(
            SourceCompactionUnavailableError,
            match='operation deadline expired',
        ) as timeout:
            await prepare(state)

        assert timeout.value.reason == 'summary_overall_timeout'
        assert started_calls == 2
        assert len(provider.calls) == 1  # One fold progressed before the shared deadline expired.
        after = owner.read_snapshot()
        assert after.events == before.events and after.checkpoint == before.checkpoint
        assert state.source.transcript == active
        metrics = state.source_compaction.attempts[-1]
        assert metrics.status == 'failed'
        assert metrics.failure_reason == 'summary_overall_timeout'


async def test_four_actual_auxiliary_folds_share_deadline_across_provider_pacing(
    store, monkeypatch,
):
    from nanobot.agent.memory import Consolidator
    from nanobot.providers.base import PreparedRequestBudgetPreview
    from nanobot.session.checkpoint_dependencies import SummaryDependencyProof

    model = 'Qwen3.8-27B-UD-Q6_K_XL'
    context_window = 32768
    generation = GenerationSettings(temperature=0.2, max_tokens=256, reasoning_effort='none')
    provider = OpenAICompatProvider(
        api_key='fixture-credential', api_base='http://manager.test/v1',
        default_model=model, provider_name='local_manager',
        spec=find_by_name('local_manager'),
    )

    class CoherentClock:
        def __init__(self, initial: float) -> None:
            self.current = initial
            self.sleeps: list[float] = []

        def __call__(self) -> float:
            return self.current

        def advance(self, seconds: float) -> None:
            self.current += seconds

        async def sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            self.advance(seconds)
            await asyncio.sleep(0)

    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        old_user = writer.append_message({'role': 'user', 'content': 'Prior objective.'})
        old_answer = writer.append_message({'role': 'assistant', 'content': 'Prior evidence.'})
        old_nodes = source_nodes(owner.read_snapshot().events)
        old_transcript = SourceTranscript((
            RuntimeNode('pacing-system-head', {'role': 'system', 'content': 'System policy.'}),
            *old_nodes,
        ))
        old_messages = old_transcript.messages()
        state = ModelRequestState(
            ContextGovernanceConfig(
                provider=provider, model=model, tools=ToolRegistry(), workspace=None,
                session_key=owner.key, max_tool_result_chars=16000,
                context_window_tokens=context_window, max_tokens=generation.max_tokens,
                generation=generation, preserve_source=True,
            ),
            ProviderConversationStateController(
                provider=provider, model=model, messages=old_messages, session_id=owner.key,
            ),
            source=SourceRequestLedger(
                old_transcript,
                initial_eligible_ids=frozenset((old_user['_event_id'], old_answer['_event_id'])),
            ),
            source_compaction=SourceCompactionState(
                writer, LLMRuntime(provider, model, generation, context_window),
                old_messages, None, overall_timeout_seconds=5.0,
            ),
        )
        governor = ContextGovernor()
        old_attempt = governor.capture_source_compaction(
            state, state.source.transcript, tool_definitions=None,
        )
        governor.publish_source_compaction(
            state, old_attempt, boundary=2, summary='Previously retained checkpoint facts.',
            auxiliary_usage=None,
        )
        prior_snapshot = owner.read_snapshot()
        assert prior_snapshot.checkpoint is not None
        assert prior_snapshot.checkpoint['boundary'] == 2

        historical_content = '\\' * 20000
        eligible_ids: list[str] = []
        for index in range(4):
            user = writer.append_message({'role': 'user', 'content': historical_content})
            assistant = writer.append_message({
                'role': 'assistant', 'content': f'Observed evidence group {index}.',
            })
            eligible_ids.extend((user['_event_id'], assistant['_event_id']))
        current = writer.append_message({'role': 'user', 'content': 'CURRENT_TAIL_SENTINEL'})
        writer.append_message({'role': 'assistant', 'content': 'PENDING_TAIL_SENTINEL'})
        assert current['_event_id'] not in eligible_ids

        projected = project_source(owner.read_snapshot())
        request_nodes = [RuntimeNode(
            'pacing-system-head', {'role': 'system', 'content': 'System policy.'},
        )]
        if projected.checkpoint is not None:
            request_nodes.append(projected.checkpoint)
        if projected.pinned is not None:
            request_nodes.append(projected.pinned)
        request_nodes.extend(projected.nodes)
        active = SourceTranscript(tuple(request_nodes))
        raw_messages = active.messages()
        state.source = SourceRequestLedger(
            active, initial_eligible_ids=frozenset(eligible_ids),
        )
        state.source_compaction.raw_messages[:] = raw_messages
        state.conversation.replace_transcript(raw_messages)
        state.source_compaction.overall_timeout_seconds = 187.0

        attempt = governor.capture_source_compaction(state, active, tool_definitions=None)
        candidate = attempt.candidate
        assert candidate.previous_boundary == 2
        assert candidate.boundaries == tuple(range(3, 11))
        before = owner.read_snapshot()
        active_before = state.source.transcript

        limits = derive_provider_request_limits(context_window, generation.max_tokens)
        provider_context = ProviderCallContext(
            request_limits=limits, context_window_tokens=context_window,
        )
        main_preview = provider.preview_request_budget(
            messages=active.messages(), tools=None, model=model, generation=generation,
            provider_context=provider_context, stream=True, tool_choice=None,
        )
        fixed_view = candidate.rebuild(
            candidate.boundaries[-1], CheckpointNode('fixed-preview', 'x'),
        ).with_runtime(*attempt.request_notes)
        fixed_preview = provider.preview_request_budget(
            messages=fixed_view.messages(), tools=None, model=model, generation=generation,
            provider_context=provider_context, stream=True, tool_choice=None,
        )
        assert isinstance(main_preview, PreparedRequestBudgetPreview)
        assert isinstance(fixed_preview, PreparedRequestBudgetPreview)
        assert not main_preview.measurement.fits
        assert fixed_preview.measurement.fits
        input_limit = fixed_preview.measurement.input_limit_tokens or 0
        target = int(input_limit * 0.6)
        allowance = min(
            2048,
            max(
                1,
                (target if fixed_preview.measurement.input_tokens < target else input_limit)
                - fixed_preview.measurement.input_tokens,
            ),
        )
        auxiliary_generation = GenerationSettings(
            temperature=generation.temperature, max_tokens=allowance, reasoning_effort='none',
        )
        auxiliary_context = ProviderCallContext(
            request_limits=derive_provider_request_limits(context_window, allowance),
            context_window_tokens=context_window,
            source_summary_mode='canonical',
        )
        expected_intervals = ((2, 4), (4, 6), (6, 8), (8, 10))
        expected_bodies: list[dict[str, object]] = []
        for index, (after, boundary) in enumerate(expected_intervals):
            evidence = candidate.summary_evidence(boundary, after=after)
            assert [node.event_id for node in evidence.nodes] == eligible_ids[index * 2:index * 2 + 2]
            assert [node.message['content'] for node in evidence.nodes] == [
                historical_content, f'Observed evidence group {index}.',
            ]
            payload = Consolidator.source_summary_messages(
                [node.message for node in evidence.nodes],
                'Previously retained checkpoint facts.'
                if index == 0 else f'Fold {index}',
            )
            preview = provider.preview_request_budget(
                messages=payload, tools=None, model=model, generation=auxiliary_generation,
                provider_context=auxiliary_context, stream=True, tool_choice=None,
            )
            assert isinstance(preview, PreparedRequestBudgetPreview)
            assert preview.measurement.fits
            expected_bodies.append(preview.effective_body)
        combined = candidate.summary_evidence(6, after=2)
        oversized_payload = Consolidator.source_summary_messages(
            [node.message for node in combined.nodes], None,
        )
        oversized_preview = provider.preview_request_budget(
            messages=oversized_payload, tools=None, model=model,
            generation=auxiliary_generation, provider_context=auxiliary_context,
            stream=True, tool_choice=None,
        )
        assert isinstance(oversized_preview, PreparedRequestBudgetPreview)
        assert not oversized_preview.measurement.fits

        loop = asyncio.get_running_loop()
        clock = CoherentClock(loop.time())
        monkeypatch.setattr(loop, 'time', clock)
        start = clock.current
        deadline = start + state.source_compaction.overall_timeout_seconds
        pacer = KeyPoolCoordinator(1, rate_limit=1, clock=clock, sleep=clock.sleep)
        leaf_stream = provider.chat_stream_with_context
        paced_calls: list[tuple[float, ProviderCallContext, dict[str, object]]] = []

        async def paced_leaf_dispatch(*, provider_context, **kwargs):
            await pacer.admit()
            paced_calls.append((clock.current, provider_context, {
                key: kwargs.get(key) for key in (
                    'model', 'max_tokens', 'temperature', 'reasoning_effort', 'tools', 'tool_choice',
                )
            }))
            return await leaf_stream(provider_context=provider_context, **kwargs)

        provider.chat_stream_with_context = paced_leaf_dispatch

        def sse_event(content, finish_reason=None):
            chunk = {
                'id': 'chatcmpl-fourfold-deadline', 'object': 'chat.completion.chunk',
                'created': 1, 'model': model,
                'choices': [{'index': 0,
                             'delta': {'content': content} if content is not None else {},
                             'finish_reason': finish_reason}],
            }
            return ('data: ' + json.dumps(chunk) + '\n\n').encode()

        class CompletedStream(httpx.AsyncByteStream):
            def __init__(self, index: int) -> None:
                self.index = index
                self.closed = False

            async def __aiter__(self):
                yield sse_event(f'Fold {self.index}')
                yield sse_event(None, 'stop')
                yield b'data: [DONE]\n\n'

            async def aclose(self) -> None:
                self.closed = True

        class FourthDeadlineStream(httpx.AsyncByteStream):
            def __init__(self) -> None:
                self.closed = False

            async def __aiter__(self):
                yield sse_event('Incomplete fourth summary.')
                clock.advance(5.0)
                await asyncio.sleep(0)
                yield sse_event(None, 'stop')
                yield b'data: [DONE]\n\n'

            async def aclose(self) -> None:
                self.closed = True

        streams: list[httpx.AsyncByteStream] = []
        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            bodies.append(body)
            if len(bodies) == 4:
                stream = FourthDeadlineStream()
            else:
                stream = CompletedStream(len(bodies))
            streams.append(stream)
            return httpx.Response(
                200, headers={'content-type': 'text/event-stream'}, stream=stream, request=request,
            )

        original_preflight = governor._preflight_summary_dependencies
        observed_proofs: list[SummaryDependencyProof] = []

        def observe_preflight(state_arg, attempt_arg, proof):
            observed_proofs.append(proof)
            return original_preflight(state_arg, attempt_arg, proof)

        governor._preflight_summary_dependencies = observe_preflight
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            provider._client = AsyncOpenAI(
                api_key='fixture-credential', base_url='http://manager.test/v1',
                http_client=http_client, max_retries=0,
            )
            with pytest.raises(
                SourceCompactionUnavailableError,
                match='operation deadline expired',
            ) as timeout:
                await governor.compact_source_request(
                    state, active, tool_definitions=None, force=True, stream=True,
                )

        assert timeout.value.reason == 'summary_overall_timeout'
        assert len(paced_calls) == 4
        assert len(bodies) == 4  # Four physical auxiliary SDK/HTTP dispatches; no fifth.
        assert len(streams) == 4 and all(stream.closed for stream in streams)
        assert clock.sleeps == [61.0, 61.0, 61.0]
        assert [entry[0] for entry in paced_calls] == pytest.approx(
            [start, start + 61.0, start + 122.0, start + 183.0],
        )
        assert clock.current >= deadline
        assert len(observed_proofs) == 4
        for index, proof in enumerate(observed_proofs):
            assert type(proof) is SummaryDependencyProof
            assert proof.candidate == candidate
            assert len(proof.chunks) == index + 1
            for previous_index, chunk in enumerate(proof.chunks[:-1]):
                assert (
                    chunk.evidence.after, chunk.evidence.boundary,
                ) == expected_intervals[previous_index]
                assert chunk.summary == f'Fold {previous_index + 1}'
            current_chunk = proof.chunks[-1]
            assert (
                current_chunk.evidence.after, current_chunk.evidence.boundary,
            ) == expected_intervals[index]
            assert current_chunk.summary == 'Preflight dependency validation.'
            assert current_chunk.previous_summary == (
                'Previously retained checkpoint facts.'
                if index == 0 else f'Fold {index}'
            )

        assert [body for body in bodies] == expected_bodies
        assert all(
            body['messages'][0]['content'].startswith('Summarize historical evidence')
            and body.get('tools') is None
            for body in bodies
        )
        assert len(paced_calls) == len(bodies) == state.source_compaction.attempts[-1].calls == 4
        assert all(context.source_summary_mode == 'canonical' for _, context, _ in paced_calls)
        assert all(context.single_attempt for _, context, _ in paced_calls)
        assert all(
            context.request_deadline_monotonic == pytest.approx(deadline)
            and context.context_window_tokens == context_window
            and context.request_limits == derive_provider_request_limits(context_window, allowance)
            for _, context, _ in paced_calls
        )
        assert all(
            details == {
                'model': model, 'max_tokens': allowance, 'temperature': generation.temperature,
                'reasoning_effort': 'none', 'tools': None, 'tool_choice': None,
            }
            for _, _, details in paced_calls
        )
        metrics = state.source_compaction.attempts[-1]
        assert metrics.status == 'failed'
        assert metrics.calls == 4
        assert metrics.failure_reason == 'summary_overall_timeout'
        after = owner.read_snapshot()
        assert after.events == before.events
        assert after.checkpoint == before.checkpoint
        assert after.checkpoint['boundary'] == 2
        assert after.checkpoint['summary'] == 'Previously retained checkpoint facts.'
        assert state.source.transcript == active_before
        assert state.source_compaction.raw_messages == active_before.messages()
        assert state.source.transcript.messages()[-2]['content'] == 'CURRENT_TAIL_SENTINEL'
        assert state.source.transcript.messages()[-1]['content'] == 'PENDING_TAIL_SENTINEL'
        assert not state.source.has_pending_requests
