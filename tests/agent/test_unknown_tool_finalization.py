"""Narrow recovery for a genuinely unknown tool after fail-fast tool execution."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import pytest

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.base import (
    Tool,
    ToolPreparationError,
    ToolPreparationErrorKind,
    ToolResult,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import TurnContext, TurnRequest
from nanobot.providers.base import LLMProvider, LLMResponse, LLMUsage, ToolCallRequest
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.session.source_projection import project_events
from tests.agent.runner_helpers import make_run_spec
from tests.agent.test_migration_engine import engine_config as _engine_config_factory


@pytest.fixture
def engine_config(tmp_path, monkeypatch):
    return _engine_config_factory.__wrapped__(tmp_path, monkeypatch)


class _EffectTool(Tool):
    def __init__(self, name: str, effects: list[str]) -> None:
        self._name = name
        self.effects = effects

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Record {self._name}."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        self.effects.append(self.name)
        return f"{self.name} completed"


class _FailureTool(Tool):
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls = 0

    @property
    def name(self) -> str:
        return "known_tool"

    @property
    def description(self) -> str:
        return "Return a fixture failure."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.mode == "exception":
            raise RuntimeError("Error: Tool 'reply' not found")
        if self.mode == "raw_error_text":
            return "Error: Tool 'reply' not found"
        if self.mode == "metadata_spoof":
            from nanobot.agent.turn import ToolOutcome

            return ToolOutcome(
                ToolResult.error("Error: Tool 'reply' not found"),
                stop_reason="tool_error",
                policy_metadata={"_prepare_error_kind": "unknown_tool"},
            )
        return ToolResult.error("Error: Tool 'reply' not found")


class _CaptureProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(provider_name="unknown-tool-test")
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "model": model, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected extra provider request")
        return self.responses.popleft()

    def get_default_model(self) -> str:
        return "test-model"


class _StreamingHook(AgentHook):
    def __init__(self) -> None:
        super().__init__()
        self.deltas: list[str] = []
        self.stream_ends: list[bool] = []

    def wants_streaming(self) -> bool:
        return True

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        self.deltas.append(delta)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        self.stream_ends.append(resuming)


class _BlockingOutputHook(_StreamingHook):
    def __init__(self) -> None:
        super().__init__()
        self.final_output_started = asyncio.Event()

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        self.deltas.append(delta)
        if delta == "ROTATION_OK":
            self.final_output_started.set()
            await asyncio.Event().wait()


class _BlockingCloseHook(_StreamingHook):
    def __init__(self) -> None:
        super().__init__()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        self.close_completed = asyncio.Event()

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        self.stream_ends.append(resuming)
        if not resuming:
            self.close_started.set()
            await self.close_release.wait()
            self.close_completed.set()


async def test_unknown_tool_finalizes_once_after_success_and_skips_later_tools() -> None:
    responses = [
        LLMResponse(content="work completed", tool_calls=[
            ToolCallRequest(id="effect", name="append_record", arguments={}),
            ToolCallRequest(id="invented", name="reply", arguments={"text": "ROTATION_OK"}),
            ToolCallRequest(id="later", name="later_effect", arguments={}),
        ], usage=LLMUsage(5, 2, 7, reported_tokens=7)),
        LLMResponse(content="ROTATION_OK", usage=LLMUsage(3, 1, 4, reported_tokens=4)),
    ]
    requests: list[dict[str, Any]] = []
    provider = MagicMock()

    async def chat_with_retry(**kwargs: Any) -> LLMResponse:
        requests.append(kwargs)
        return responses.pop(0)

    provider.chat_with_retry = chat_with_retry
    provider.chat_stream_with_retry = chat_with_retry
    effects: list[str] = []
    tools = ToolRegistry()
    tools.register(_EffectTool("append_record", effects))
    tools.register(_EffectTool("later_effect", effects))
    hook = _StreamingHook()

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "return ROTATION_OK after recording"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=1000,
        fail_on_tool_error=True,
        turn_context=TurnContext(run_id="unknown-finalizer-run"),
        hook=hook,
    ))

    assert result.stop_reason == "completed"
    assert result.final_content == "ROTATION_OK"
    assert effects == ["append_record"]
    assert len(requests) == 2
    assert requests[1]["tools"] is None
    assert requests[1]["tool_choice"] == "none"
    assert result.usage is not None and result.usage.total_tokens == 11
    assert [item.total_tokens for item in result.round_usages] == [7, 4]
    assert result.tools_used == ["append_record"]
    assert [event["status"] for event in result.tool_events] == ["ok", "error", "skipped"]
    assert all(not any(key.startswith("_") for key in event) for event in result.tool_events)
    assert hook.deltas == ["ROTATION_OK"]
    assert hook.stream_ends == [True, False]
    tool_messages = {
        message.get("tool_call_id"): message
        for message in result.messages
        if message.get("role") == "tool"
    }
    assert "Tool 'reply' not found." in tool_messages["invented"]["content"]
    assert "Skipped" in tool_messages["later"]["content"]
    assert [message["content"] for message in result.messages if message.get("role") == "assistant"][-1] == "ROTATION_OK"


@pytest.mark.parametrize("mode", ["tool_error", "metadata_spoof", "exception"])
async def test_dispatched_failures_never_receive_unknown_tool_finalization(mode: str) -> None:
    provider = MagicMock()
    requests: list[dict[str, Any]] = []
    response = LLMResponse(
        content="try known tool",
        tool_calls=[ToolCallRequest(id="known", name="known_tool", arguments={})],
    )

    async def chat_with_retry(**kwargs: Any) -> LLMResponse:
        requests.append(kwargs)
        if len(requests) > 1:
            raise AssertionError("dispatched tool failure must not request recovery")
        return response

    provider.chat_with_retry = chat_with_retry
    tool = _FailureTool(mode)
    tools = ToolRegistry()
    tools.register(tool)

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "run the tool"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=1000,
        fail_on_tool_error=True,
    ))

    assert result.stop_reason == "tool_error"
    assert tool.calls == 1
    assert len(requests) == 1
    assert result.tools_used == []


async def test_dispatched_error_text_does_not_use_unknown_tool_finalizer() -> None:
    provider = MagicMock()
    responses = [
        LLMResponse(content=None, tool_calls=[
            ToolCallRequest(id="known", name="known_tool", arguments={}),
        ]),
        LLMResponse(content="the tool reported that text"),
    ]
    calls: list[dict[str, Any]] = []

    async def chat_with_retry(**kwargs: Any) -> LLMResponse:
        calls.append(kwargs)
        return responses.pop(0)

    provider.chat_with_retry = chat_with_retry
    tool = _FailureTool("raw_error_text")
    tools = ToolRegistry()
    tools.register(tool)

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "report the tool result"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=1000,
        fail_on_tool_error=True,
    ))

    assert result.stop_reason == "completed"
    assert result.final_content == "the tool reported that text"
    assert tool.calls == 1
    assert len(calls) == 2
    assert calls[1]["tools"] is not None
    assert "tool_choice" not in calls[1]


async def test_unknown_tool_keeps_normal_recovery_when_fail_fast_is_disabled() -> None:
    responses = [
        LLMResponse(content=None, tool_calls=[
            ToolCallRequest(id="unknown", name="reply", arguments={}),
        ]),
        LLMResponse(content="normal recovery"),
    ]
    requests: list[dict[str, Any]] = []
    provider = MagicMock()

    async def chat_with_retry(**kwargs: Any) -> LLMResponse:
        requests.append(kwargs)
        return responses.pop(0)

    provider.chat_with_retry = chat_with_retry

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "finish this"}],
        tools=ToolRegistry(),
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=1000,
        fail_on_tool_error=False,
    ))

    assert result.stop_reason == "completed"
    assert result.final_content == "normal recovery"
    assert len(requests) == 2
    assert requests[1]["tools"] == []
    assert "tool_choice" not in requests[1]


async def test_registry_filters_and_unregister_preserve_withheld_provenance() -> None:
    tools = ToolRegistry()
    tools.register(_EffectTool("hidden", []))
    tools.register(_EffectTool("visible", []))

    filtered = tools.filtered(["visible"])
    nested = filtered.filtered([])
    _, _, hidden_error = nested.prepare_call("hidden", {})
    _, _, absent_error = nested.prepare_call("never_registered", {})
    assert isinstance(hidden_error, ToolPreparationError)
    assert hidden_error.kind is ToolPreparationErrorKind.WITHHELD_TOOL
    assert isinstance(absent_error, ToolPreparationError)
    assert absent_error.kind is ToolPreparationErrorKind.UNKNOWN_TOOL

    filtered.unregister("visible")
    _, _, removed_error = filtered.prepare_call("visible", {})
    assert isinstance(removed_error, ToolPreparationError)
    assert removed_error.kind is ToolPreparationErrorKind.WITHHELD_TOOL

    effects: list[str] = []
    nested.register(_EffectTool("hidden", effects))
    restored, _, restored_error = nested.prepare_call("hidden", {})
    assert isinstance(restored, _EffectTool)
    assert restored_error is None
    assert await nested.execute("hidden", {}) == "hidden completed"
    assert effects == ["hidden"]


@pytest.mark.parametrize(
    "final_response",
    [
        LLMResponse(content="  "),
        LLMResponse(content="partial", finish_reason="error"),
        LLMResponse(content="blocked", finish_reason="refusal"),
        LLMResponse(content="filtered", finish_reason="content_filter"),
        LLMResponse(content="partial", finish_reason="length"),
        LLMResponse(content=None, finish_reason="tool_calls", tool_calls=[
            ToolCallRequest(id="finalizer-tool", name="later_effect", arguments={}),
        ]),
    ],
)
async def test_invalid_finalizer_falls_back_without_dispatch(final_response: LLMResponse) -> None:
    responses = [
        LLMResponse(content=None, tool_calls=[
            ToolCallRequest(id="unknown", name="reply", arguments={"text": "ROTATION_OK"}),
            ToolCallRequest(id="later", name="later_effect", arguments={}),
        ]),
        final_response,
    ]
    requests: list[dict[str, Any]] = []
    provider = MagicMock()

    async def chat_with_retry(**kwargs: Any) -> LLMResponse:
        requests.append(kwargs)
        return responses.pop(0)

    provider.chat_with_retry = chat_with_retry
    effects: list[str] = []
    tools = ToolRegistry()
    tools.register(_EffectTool("later_effect", effects))

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "return ROTATION_OK"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=1000,
        fail_on_tool_error=True,
    ))

    assert result.stop_reason == "tool_error"
    assert "Tool 'reply' not found." in (result.final_content or "")
    assert len(requests) == 2
    assert requests[1]["tools"] is None
    assert requests[1]["tool_choice"] == "none"
    assert effects == []
    assert result.tools_used == []


@pytest.mark.parametrize("failure", [RuntimeError("provider failed"), asyncio.CancelledError()])
async def test_finalizer_exceptions_fallback_or_propagate(failure: BaseException) -> None:
    requests: list[dict[str, Any]] = []
    provider = MagicMock()

    async def chat_with_retry(**kwargs: Any) -> LLMResponse:
        requests.append(kwargs)
        if len(requests) == 1:
            return LLMResponse(content=None, tool_calls=[
                ToolCallRequest(id="unknown", name="reply", arguments={}),
            ])
        raise failure

    provider.chat_with_retry = chat_with_retry
    spec = make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "finish the task"}],
        tools=ToolRegistry(),
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=1000,
        fail_on_tool_error=True,
    )
    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await AgentRunner().run(spec)
    else:
        result = await AgentRunner().run(spec)
        assert result.stop_reason == "tool_error"
        assert "Tool 'reply' not found." in (result.final_content or "")
    assert len(requests) == 2


async def test_finalizer_real_timeout_keeps_tool_error_without_retrying() -> None:
    requests: list[dict[str, Any]] = []
    provider = MagicMock()

    async def chat_with_retry(**kwargs: Any) -> LLMResponse:
        requests.append(kwargs)
        if len(requests) == 1:
            return LLMResponse(content=None, tool_calls=[
                ToolCallRequest(id="unknown", name="reply", arguments={}),
            ])
        await asyncio.Event().wait()
        raise AssertionError("timed-out finalizer unexpectedly returned")

    provider.chat_with_retry = chat_with_retry
    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "finish the task"}],
        tools=ToolRegistry(),
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=1000,
        fail_on_tool_error=True,
        llm_timeout_s=0.01,
    ))

    assert result.stop_reason == "tool_error"
    assert "Tool 'reply' not found." in str(result.final_content)
    assert len(requests) == 2
    assert requests[1]["tools"] is None
    assert requests[1]["tool_choice"] == "none"


async def test_session_disabled_tools_are_withheld_and_do_not_finalize(engine_config) -> None:
    provider = _CaptureProvider([LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id="disabled", name="disabled_tool", arguments={}),
    ])])
    tools = ToolRegistry()
    tools.register(_EffectTool("disabled_tool", []))
    loop = AgentLoop.from_config(engine_config, tool_registry=tools, provider=provider)
    try:
        session = loop.sessions.get_or_create("cli:disabled-tool")
        session.policy = replace(session.policy, disabled_tools=frozenset({"disabled_tool"}))
        loop.sessions.save(session)

        result = await loop.execute_turn(TurnRequest(
            content="use the disabled tool",
            session_key="cli:disabled-tool",
        ))

        assert result.stop_reason == "tool_error"
        assert len(provider.calls) == 1
        disabled_names = {
            item["function"]["name"] for item in provider.calls[0]["tools"]
        }
        assert "disabled_tool" not in disabled_names
    finally:
        await loop.aclose()


@pytest.mark.parametrize("managed", [False, True])
async def test_websocket_registry_keeps_unregistered_tools_withheld(
    engine_config, managed: bool,
) -> None:
    provider = _CaptureProvider([LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id="withheld-read", name="read_file", arguments={"path": "sample.txt"}),
    ])])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        assert loop.tools.get("read_file") is not None
        loop.tools.unregister("read_file")
        grants = FolderGrantStore.for_workspace(
            loop.workspace, sessions_root=loop.sessions.sessions_root,
        )
        assert grants.view is loop._fork_dependencies.resource_policy.folder_grants
        if managed:
            grants.revoke(str(loop.workspace / "missing-grant"), expected_revision=0)
        assert grants.view.is_managed is managed

        result = await loop.execute_turn(TurnRequest(
            content="read the withheld file",
            session_key="websocket:withheld-tool",
        ))

        assert result.stop_reason == "tool_error"
        assert len(provider.calls) == 1
        assert "read_file" not in {
            item["function"]["name"] for item in provider.calls[0]["tools"]
        }
    finally:
        await loop.aclose()


async def test_dream_filter_keeps_disabled_name_withheld(engine_config) -> None:
    provider = _CaptureProvider([LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id="dream-disabled", name="read_file", arguments={}),
    ])])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        dream_tools = loop.context.memory.build_dream_tools()
        filtered = loop.context.memory.filter_dream_tools(
            dream_tools,
            frozenset({"read_file"}),
            engine_config.workspace_path,
        )
        assert filtered is not None
        _, _, error = filtered.prepare_call("read_file", {})
        assert isinstance(error, ToolPreparationError)
        assert error.kind is ToolPreparationErrorKind.WITHHELD_TOOL
        result = await AgentRunner().run(make_run_spec(
            provider,
            initial_messages=[{"role": "user", "content": "read the disabled file"}],
            tools=filtered,
            model="test-model",
            max_iterations=2,
            max_tool_result_chars=1000,
            fail_on_tool_error=True,
        ))
        assert result.stop_reason == "tool_error"
        assert result.tools_used == []
        assert len(provider.calls) == 1
        assert "read_file" not in {
            tool["function"]["name"] for tool in provider.calls[0]["tools"]
        }
    finally:
        await loop.aclose()


def _upgrade_source_session(loop: AgentLoop, key: str) -> Any:
    session = loop.sessions.get_or_create(key)
    loop.sessions.save(session)
    backend = loop.sessions._jsonl_store
    backend.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    loop.sessions.invalidate(key)
    return backend


async def test_source_turn_records_rejection_and_binds_one_final_response(engine_config) -> None:
    provider = _CaptureProvider([
        LLMResponse(content=None, tool_calls=[
            ToolCallRequest(id="effect", name="append_record", arguments={}),
            ToolCallRequest(id="unknown", name="reply", arguments={"text": "ROTATION_OK"}),
            ToolCallRequest(id="later", name="later_effect", arguments={}),
        ]),
        LLMResponse(content="ROTATION_OK"),
    ])
    effects: list[str] = []
    tools = ToolRegistry()
    tools.register(_EffectTool("append_record", effects))
    tools.register(_EffectTool("later_effect", effects))
    loop = AgentLoop.from_config(engine_config, tool_registry=tools, provider=provider)
    key = "cli:unknown-tool-source"
    try:
        backend = _upgrade_source_session(loop, key)

        result = await loop.execute_turn(TurnRequest(content="return ROTATION_OK", session_key=key))

        assert result.error is None, result.error
        events = backend.source.load_source_snapshot(key).events
        rejected = next(event for event in events if event.get("tool_call_id") == "unknown")
        assert result.content == "ROTATION_OK"
        assert effects == ["append_record"]
        assert rejected["_dispatched"] is False
        assert rejected["_outcome_status"] == "error"
        assert rejected["_prepare_error_kind"] == "unknown_tool"
        final_positions = [
            index for index, event in enumerate(events)
            if event.get("role") == "assistant" and event.get("content") == "ROTATION_OK"
        ]
        assert final_positions == [len(events) - 1]
        assert events.index(rejected) < final_positions[0]
        assert result.tools_used == ["append_record"]
        assert len(provider.calls) == 2
        assert provider.calls[1]["tools"] is None
        assert provider.calls[1]["tool_choice"] == "none"
        public, _ = project_events(events)
        assert "_prepare_error_kind" not in str(public)
        assert not any(message.get("_prepare_error_kind") for message in result.messages or [])
    finally:
        await loop.aclose()


async def test_source_finalizer_provider_exception_keeps_tool_error_fallback(engine_config) -> None:
    provider = _CaptureProvider([])
    calls: list[dict[str, Any]] = []

    async def chat_with_retry(**kwargs: Any) -> LLMResponse:
        calls.append(kwargs)
        if len(calls) == 1:
            return LLMResponse(content=None, tool_calls=[
                ToolCallRequest(id="unknown", name="reply", arguments={}),
            ])
        raise RuntimeError("finalizer unavailable")

    provider.chat_with_retry = chat_with_retry
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    key = "cli:unknown-tool-provider-error"
    try:
        backend = _upgrade_source_session(loop, key)
        result = await loop.execute_turn(TurnRequest(content="finish the task", session_key=key))

        events = backend.source.load_source_snapshot(key).events
        assert result.error is None, result.error
        assert result.stop_reason == "tool_error"
        assert "Tool 'reply' not found." in str(result.final_content)
        assert len(calls) == 2
        assert calls[1]["tools"] is None
        assert calls[1]["tool_choice"] == "none"
        assert events[-1]["role"] == "assistant"
        assert events[-1]["content"] == result.final_content
        rejected = next(event for event in events if event.get("tool_call_id") == "unknown")
        assert rejected["_prepare_error_kind"] == "unknown_tool"
    finally:
        await loop.aclose()


async def test_streaming_unknown_tool_finalizer_cancellation_closes_stream() -> None:
    provider = MagicMock()
    requests: list[dict[str, Any]] = []
    finalizer_started = asyncio.Event()
    effects: list[str] = []

    async def chat_with_retry(**kwargs: Any) -> LLMResponse:
        requests.append(kwargs)
        if len(requests) == 1:
            return LLMResponse(content=None, tool_calls=[
                ToolCallRequest(id="effect", name="append_record", arguments={}),
                ToolCallRequest(id="unknown", name="reply", arguments={}),
            ])
        finalizer_started.set()
        await asyncio.Event().wait()
        raise AssertionError("cancelled finalizer unexpectedly returned")

    provider.chat_with_retry = chat_with_retry
    provider.chat_stream_with_retry = chat_with_retry
    tools = ToolRegistry()
    tools.register(_EffectTool("append_record", effects))
    hook = _StreamingHook()
    spec = make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "record then reply"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=1000,
        fail_on_tool_error=True,
        hook=hook,
    )

    task = asyncio.create_task(AgentRunner().run(spec))
    await asyncio.wait_for(finalizer_started.wait(), timeout=2)
    task.cancel("provider cancellation")
    with pytest.raises(asyncio.CancelledError) as cancellation:
        await task

    assert cancellation.value.args == ("provider cancellation",)
    assert effects == ["append_record"]
    assert len(requests) == 2
    assert requests[1]["tools"] is None
    assert requests[1]["tool_choice"] == "none"
    assert hook.stream_ends == [True, False]


async def test_streaming_final_output_cancellation_closes_stream_once() -> None:
    provider = MagicMock()
    requests: list[dict[str, Any]] = []
    effects: list[str] = []

    async def chat_with_retry(**kwargs: Any) -> LLMResponse:
        requests.append(kwargs)
        if len(requests) == 1:
            return LLMResponse(content=None, tool_calls=[
                ToolCallRequest(id="effect", name="append_record", arguments={}),
                ToolCallRequest(id="unknown", name="reply", arguments={}),
            ])
        return LLMResponse(content="ROTATION_OK")

    provider.chat_with_retry = chat_with_retry
    provider.chat_stream_with_retry = chat_with_retry
    tools = ToolRegistry()
    tools.register(_EffectTool("append_record", effects))
    hook = _BlockingOutputHook()
    task = asyncio.create_task(AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "record then reply"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=1000,
        fail_on_tool_error=True,
        hook=hook,
    )))

    await asyncio.wait_for(hook.final_output_started.wait(), timeout=2)
    task.cancel("final output cancellation")
    with pytest.raises(asyncio.CancelledError) as cancellation:
        await task

    assert cancellation.value.args == ("final output cancellation",)
    assert effects == ["append_record"]
    assert len(requests) == 2
    assert requests[1]["tools"] is None
    assert requests[1]["tool_choice"] == "none"
    assert hook.deltas == ["ROTATION_OK"]
    assert hook.stream_ends == [True, False]


async def test_streaming_close_cancellation_finishes_single_close_callback() -> None:
    provider = MagicMock()
    requests: list[dict[str, Any]] = []
    effects: list[str] = []

    async def chat_with_retry(**kwargs: Any) -> LLMResponse:
        requests.append(kwargs)
        if len(requests) == 1:
            return LLMResponse(content=None, tool_calls=[
                ToolCallRequest(id="effect", name="append_record", arguments={}),
                ToolCallRequest(id="unknown", name="reply", arguments={}),
            ])
        return LLMResponse(content="ROTATION_OK")

    provider.chat_with_retry = chat_with_retry
    provider.chat_stream_with_retry = chat_with_retry
    tools = ToolRegistry()
    tools.register(_EffectTool("append_record", effects))
    hook = _BlockingCloseHook()
    task = asyncio.create_task(AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "record then reply"}],
        tools=tools,
        model="test-model",
        max_iterations=1,
        max_tool_result_chars=1000,
        fail_on_tool_error=True,
        hook=hook,
    )))

    await asyncio.wait_for(hook.close_started.wait(), timeout=2)
    task.cancel("stream close cancellation")
    await asyncio.sleep(0)
    hook.close_release.set()
    with pytest.raises(asyncio.CancelledError) as cancellation:
        await task

    assert cancellation.value.args == ("stream close cancellation",)
    assert hook.close_completed.is_set()
    assert hook.stream_ends == [True, False]
    assert effects == ["append_record"]
    assert len(requests) == 2
