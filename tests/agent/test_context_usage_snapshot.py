"""Completion-owned context measurements, independent of additive spending."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kohaku.agent.runner import AgentRunner
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.bus.queue import MessageBus
from kohaku.bus.runtime_events import RuntimeEventPublisher, TurnCompleted
from kohaku.providers.base import LLMResponse, LLMUsage
from tests.agent.runner_helpers import make_run_spec


async def test_runner_exposes_request_bound_context_snapshot() -> None:
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="answer", usage=LLMUsage.reported(input_tokens=2000, output_tokens=25),
    ))
    result = await AgentRunner().run(make_run_spec(
        provider, initial_messages=[{"role": "user", "content": "hello"}],
        tools=ToolRegistry(), model="test-model", context_window_tokens=12000, max_tokens=256,
        max_iterations=2, max_tool_result_chars=10000,
    ))
    assert result.context_usage.context_tokens == 2000
    assert result.context_usage.context_window_tokens == 12000
    assert result.usage.input_tokens == 2000


async def test_publisher_snapshot_truth_table_independent_of_rounds_and_reset() -> None:
    from kohaku.providers.base import ContextUsageSnapshot

    bus = MessageBus()
    publisher = RuntimeEventPublisher(bus)
    seen = []
    bus.subscribe(seen.append, TurnCompleted)
    measured = ContextUsageSnapshot(2000, 12000)
    invalid = ContextUsageSnapshot(None, None)
    for updates, expected in [
        ([None], None), ([measured, None], measured),
        ([measured, invalid, None], invalid), ([invalid, measured], measured),
    ]:
        for value in updates:
            publisher.record_turn_context_usage("test", value)
            publisher.record_turn_usage("test", [])
        await publisher.turn_completed(channel="websocket", chat_id="test", session_key="test", metadata={})
        assert seen[-1].context_usage == expected
    await publisher.turn_completed(channel="websocket", chat_id="test", session_key="test", metadata={})
    assert seen[-1].context_usage is None
    publisher.record_turn_context_usage("test", measured)
    publisher.clear_turn("test")
    await publisher.turn_completed(channel="websocket", chat_id="test", session_key="test", metadata={})
    assert seen[-1].context_usage is None


@pytest.mark.parametrize("payload", [None, {}, {"context_tokens": True, "context_window_tokens": 12000},
    {"context_tokens": -1, "context_window_tokens": 12000},
    {"context_tokens": 2000, "context_window_tokens": 0}])
def test_snapshot_malformed_present_is_unavailable(payload) -> None:
    from kohaku.providers.base import ContextUsageSnapshot

    assert ContextUsageSnapshot.from_value(payload) == ContextUsageSnapshot(None, None)


@pytest.mark.parametrize("invalidate", [False, True])
async def test_unmeasured_preserves_own_window_but_cannot_resurrect_after_invalidation(invalidate) -> None:
    from kohaku.providers.base import ContextUsageSnapshot, ProviderCallContext

    provider = MagicMock()
    runner = AgentRunner()
    spec = make_run_spec(provider, initial_messages=[], tools=ToolRegistry(), model="test-model",
                        max_iterations=2, max_tool_result_chars=10000,
                        context_window_tokens=12000, max_tokens=256)
    state = runner._make_request_state(spec, [], None)
    state.messages = [{"role": "user", "content": "hello"}]
    state.capture_prepared_context_usage(ProviderCallContext())
    first = LLMResponse(content="answer", usage=LLMUsage.reported(input_tokens=5000, output_tokens=10))
    recorded = runner._record_and_settle(spec, state, None, first)
    historical = runner._record_request_usage(spec, state, first, recorded_response=recorded)
    assert state.context_usage == ContextUsageSnapshot(5000, 12000)
    if invalidate:
        state.invalidate_context_usage()
    state.config.context_window_tokens = 24000
    state.capture_prepared_context_usage(ProviderCallContext())
    empty = LLMResponse(content="error", finish_reason="error", usage=LLMUsage.empty_request())
    recorded = runner._record_and_settle(spec, state, None, empty)
    empty_usage = runner._record_request_usage(spec, state, empty, recorded_response=recorded)
    assert (historical + empty_usage).context_tokens == 5000
    assert state.context_usage == (ContextUsageSnapshot(None, None) if invalidate else ContextUsageSnapshot(5000, 12000))
    zero = LLMResponse(content="zero", usage=LLMUsage.reported(input_tokens=0, output_tokens=1))
    recorded = runner._record_and_settle(spec, state, None, zero)
    runner._record_request_usage(spec, state, zero, recorded_response=recorded)
    assert state.context_usage == ContextUsageSnapshot(0, 24000)


@pytest.mark.parametrize("later_request", [False, True])
async def test_native_success_cannot_authorize_its_own_old_dispatch_but_later_request_can(monkeypatch, later_request) -> None:
    from kohaku.agent.context import TranscriptInput
    from kohaku.providers.base import (
        ContextUsageSnapshot,
        ProviderConversationState,
        ToolCallRequest,
    )
    from tests.agent.test_runner_governance import _build_transcript

    provider = MagicMock()
    provider.can_resume_conversation_state.return_value = False
    native = LLMResponse(
        content="before native" if not later_request else None,
        tool_calls=[ToolCallRequest("tool-1", "inspect", {})] if later_request else [],
        finish_reason="tool_calls" if later_request else "stop",
        usage=LLMUsage.reported(input_tokens=9000, output_tokens=10),
        provider_compaction_applied=True,
        provider_compaction_state=ProviderConversationState("native", "fixture", "test-model", 1),
        provider_compaction_scope="current_request",
    )
    provider.chat_with_retry = AsyncMock(side_effect=[native, LLMResponse(
        content="after native", usage=LLMUsage.reported(input_tokens=2000, output_tokens=10),
    )])
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")
    monkeypatch.setattr("kohaku.agent.context_governance.estimate_prompt_tokens_chain", lambda *_a: (100, "test"))
    result = await AgentRunner().run(make_run_spec(
        provider, initial_messages=None,
        transcript_input=TranscriptInput(history=[], current_message="inspect"),
        transcript_builder=_build_transcript, consolidate_history=AsyncMock(return_value="summary"),
        consolidate_provider_compaction=AsyncMock(return_value="portable checkpoint"),
        tools=tools, model="test-model", max_iterations=2, max_tool_result_chars=10000,
        context_window_tokens=12000, max_tokens=256,
    ))
    assert result.context_usage == (ContextUsageSnapshot(2000, 12000) if later_request else ContextUsageSnapshot(None, None))
    assert result.usage.input_tokens == (11000 if later_request else 9000)


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("replacement", [False, True])
async def test_real_silent_image_replacement_relay_survives_runner_error_and_cancellation(cancelled, replacement) -> None:
    import asyncio

    from kohaku.agent.turn_delivery import TurnDeliveryFactory
    from kohaku.agent.turn_hooks import AgentTurnHookSpec, build_agent_turn_hook
    from kohaku.bus.events import InboundMessage
    from kohaku.bus.outbound_events import StreamEndEvent, TurnEndEvent
    from kohaku.providers.base import ContextUsageSnapshot
    from kohaku.session.source_projection import SourceNode
    from kohaku.session.source_request import SourceRequestLedger, SourceTranscript
    from kohaku.webui.outbound_wire import encode_turn_end
    from kohaku.webui.transcript import replay_transcript_to_ui_messages
    from tests.agent.test_migration_engine import ScriptedProvider

    def image(label):
        return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/" + label + ".png"}}]}
    initial = [image("old"), image("new")] if replacement else [image("only")]
    source = SourceRequestLedger(SourceTranscript(tuple(SourceNode(str(i), m) for i, m in enumerate(initial))), initial_eligible_ids=frozenset(str(i) for i in range(len(initial))))

    class SourceRunner(AgentRunner):
        def _make_request_state(self, spec, messages, compaction):
            state = super()._make_request_state(spec, messages, compaction)
            state.source = source
            state.config.preserve_source = True
            return state

    provider = ScriptedProvider([])
    provider.estimate_prompt_tokens = lambda *_a: (100, "test")
    failure = asyncio.CancelledError() if cancelled else RuntimeError("after silent replacement")
    provider.chat_stream_with_retry = AsyncMock(side_effect=failure)
    bus = MessageBus()
    factory = TurnDeliveryFactory(bus)
    delivery = factory.create(InboundMessage(channel="websocket", sender_id="u", chat_id="t", content="new"), "websocket:t")
    factory.runtime_event_publisher.record_turn_context_usage("websocket:t", ContextUsageSnapshot(9000, 12000))
    seen = []
    bus.subscribe(seen.append, TurnCompleted)
    hook = build_agent_turn_hook(AgentTurnHookSpec(
        context_usage_hook=delivery.context_usage_hook(), streaming=True, events=delivery.events,
        ephemeral=True,  # Built-in must survive skip-extras.
    ))
    with pytest.raises(type(failure)) as caught:
        await SourceRunner().run(make_run_spec(
            provider, initial_messages=initial, tools=ToolRegistry(), model="test-model",
            max_iterations=1, max_tool_result_chars=10000, context_window_tokens=12000, max_tokens=256,
            hook=hook, session_key="websocket:t", events=delivery.events,
        ))
    assert caught.value is failure
    assert source.projection_epoch == (1 if replacement else 0)
    assert ("Earlier image omitted" in str(source.transcript.messages()[0])) is replacement
    await delivery.publish_cancelled_terminal()
    assert seen[-1].context_usage == (ContextUsageSnapshot(None, None) if replacement else ContextUsageSnapshot(9000, 12000))
    wire = dict(encode_turn_end("t", TurnEndEvent(context_usage=seen[-1].context_usage)))
    wire.update(turn_id="t", turn_seq=2)
    rows = replay_transcript_to_ui_messages([{"event": "user", "text": "new", "turn_id": "t"}, wire])
    assert rows[0]["contextUsage"]["contextTokens"] == (None if replacement else 9000)
    assert all(not isinstance(m.event, StreamEndEvent) for m in list(bus.outbound._queue))


async def test_state_creation_failure_relays_no_update_and_preserves_original(monkeypatch) -> None:
    from kohaku.agent.hook import AgentHook

    observed = []
    class Observer(AgentHook):
        async def on_finally(self, context):
            observed.append(context.context_usage)
    runner = AgentRunner()
    error = RuntimeError("no state")
    monkeypatch.setattr(runner, "_make_request_state", MagicMock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        await runner.run(make_run_spec(MagicMock(), initial_messages=[], tools=ToolRegistry(),
            model="test-model", max_iterations=1, max_tool_result_chars=10000, hook=Observer()))
    assert caught.value is error
    assert observed == [None]
