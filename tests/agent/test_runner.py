"""Focused companion coverage for current AgentRunner terminal and stream contracts."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.runner_helpers import make_run_spec
from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.policy import ToolPolicy, ToolPolicyDecision
from nanobot.agent.progress_hook import AgentProgressHook
from nanobot.bus.outbound_events import ProgressEvent, StreamDeltaEvent, StreamEndEvent
from nanobot.config.schema import AgentDefaults
from nanobot.events import EventSink
from nanobot.providers.base import LLMResponse, ToolCallRequest

_MAX_TOOL_RESULT_CHARS = AgentDefaults().max_tool_result_chars


@pytest.mark.asyncio
async def test_runner_policy_block_closes_resumed_stream_segment() -> None:
    provider = MagicMock()
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(
        content="I can remove those files.",
        tool_calls=[ToolCallRequest(id="call_1", name="exec", arguments={"command": "rm -rf x"})],
        usage=None,
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    endings: list[bool] = []

    class StreamingHook(AgentHook):
        def wants_streaming(self) -> bool:
            return True

        async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
            endings.append(resuming)

    class BlockingPolicy(ToolPolicy):
        async def evaluate(self, *, messages, tool_calls):
            return ToolPolicyDecision(
                action="respond",
                stop_reason="approval_required",
                response="Approval required before continuing.",
            )

    from nanobot.agent.runner import AgentRunner

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        hook=StreamingHook(),
        tool_policy=BlockingPolicy(),
    ))

    assert result.stop_reason == "approval_required"
    assert result.final_content.startswith("Approval required before continuing.")
    # A policy response is a resumed/finalized segment after the provider's
    # tool-call segment has been closed.
    assert endings == [True]


@pytest.mark.asyncio
async def test_runner_provider_error_does_not_emit_stream_end() -> None:
    provider = MagicMock()
    provider.chat_stream_with_retry = AsyncMock(side_effect=RuntimeError("provider down"))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    endings: list[bool] = []

    class StreamingHook(AgentHook):
        def wants_streaming(self) -> bool:
            return True

        async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
            endings.append(resuming)

    from nanobot.agent.runner import AgentRunner

    with pytest.raises(RuntimeError, match="provider down"):
        await AgentRunner().run(make_run_spec(
            provider,
            initial_messages=[],
            tools=tools,
            model="test-model",
            max_iterations=1,
            max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
            hook=StreamingHook(),
        ))

    assert endings == []


@pytest.mark.asyncio
async def test_runner_cancellation_reraises_without_stream_end() -> None:
    provider = MagicMock()
    started = asyncio.Event()

    async def stream(*, on_content_delta, **kwargs):
        started.set()
        await asyncio.Future()

    provider.chat_stream_with_retry = stream
    tools = MagicMock()
    tools.get_definitions.return_value = []
    endings: list[bool] = []

    class StreamingHook(AgentHook):
        def wants_streaming(self) -> bool:
            return True

        async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
            endings.append(resuming)

    from nanobot.agent.runner import AgentRunner

    task = asyncio.create_task(AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        hook=StreamingHook(),
    )))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert endings == []


@pytest.mark.asyncio
async def test_runner_streaming_max_iterations_closes_only_intermediate_segment() -> None:
    provider = MagicMock()
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={"path": "."})],
        usage=None,
    ))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="tool result")
    endings: list[bool] = []

    class StreamingHook(AgentHook):
        def wants_streaming(self) -> bool:
            return True

        async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
            endings.append(resuming)

    from nanobot.agent.runner import AgentRunner

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        hook=StreamingHook(),
    ))

    assert result.stop_reason == "max_iterations"
    assert endings == [True]


@pytest.mark.asyncio
async def test_runner_stream_filters_think_prefix_before_visible_delta() -> None:
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock()

    async def chat_stream_with_retry(*, on_content_delta, **kwargs):
        await on_content_delta("<think>hidden")
        await on_content_delta("</think>Hello")
        return LLMResponse(content="<think>hidden</think>Hello", tool_calls=[], usage=None)

    provider.chat_stream_with_retry = chat_stream_with_retry
    provider.chat_with_retry = AsyncMock()
    tools = MagicMock()
    tools.get_definitions.return_value = []
    events = []

    async def publish(event):
        events.append(event)

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        hook=AgentProgressHook(EventSink(publish), streaming=True),
    ))

    assert result.final_content == "Hello"
    assert [event.content for event in events if isinstance(event, StreamDeltaEvent)] == ["Hello"]
    assert [event.resuming for event in events if isinstance(event, StreamEndEvent)] == [False]


@pytest.mark.asyncio
async def test_runner_does_not_repeat_streamed_tool_preamble_in_progress() -> None:
    from nanobot.agent.runner import AgentRunner

    provider = MagicMock()
    call_count = 0

    async def chat_stream_with_retry(*, on_content_delta, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await on_content_delta("Checking the repo now.")
            return LLMResponse(
                content="Checking the repo now.",
                tool_calls=[ToolCallRequest(id="call_1", name="list_dir", arguments={})],
                usage=None,
            )
        await on_content_delta("All set.")
        return LLMResponse(content="All set.", tool_calls=[], usage=None)

    provider.chat_stream_with_retry = chat_stream_with_retry
    provider.chat_with_retry = AsyncMock()
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="ok")
    events = []

    async def publish(event):
        events.append(event)

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        hook=AgentProgressHook(EventSink(publish), streaming=True),
    ))

    assert result.final_content == "All set."
    assert result.tools_used == ["list_dir"]
    assert [event.content for event in events if isinstance(event, StreamDeltaEvent)] == [
        "Checking the repo now.",
        "All set.",
    ]
    assert [event.resuming for event in events if isinstance(event, StreamEndEvent)] == [True, False]
    progress = [event for event in events if isinstance(event, ProgressEvent)]
    assert all("Checking the repo now." not in event.content for event in progress)
    assert any(event.tool_hint and "list_dir" in event.content for event in progress)
