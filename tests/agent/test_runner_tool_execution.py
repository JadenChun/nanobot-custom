"""Tests for AgentRunner tool execution: batching, concurrency, exclusive tools."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kohaku.agent.hook import AgentHook, AgentHookContext
from kohaku.agent.runner import AgentRunner
from kohaku.agent.tools.base import Tool, ToolResult
from kohaku.agent.tools.context import ToolContext
from kohaku.agent.tools.execution import execute_tool_calls
from kohaku.agent.tools.loader import ToolLoader
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.agent.turn import ToolOutcome, TurnContext
from kohaku.config.schema import AgentDefaults, ToolsConfig
from kohaku.providers.base import LLMResponse, ToolCallRequest
from kohaku.providers.openai_compat_provider import OpenAICompatProvider
from kohaku.providers.openai_responses.parsing import parse_response_output
from tests.agent.runner_helpers import install_test_tool_dispatch, make_run_spec

_MAX_TOOL_RESULT_CHARS = AgentDefaults().max_tool_result_chars


class _DelayTool(Tool):
    def __init__(
        self,
        name: str,
        *,
        delay: float,
        read_only: bool,
        shared_events: list[str],
        exclusive: bool = False,
    ):
        self._name = name
        self._delay = delay
        self._read_only = read_only
        self._shared_events = shared_events
        self._exclusive = exclusive

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def exclusive(self) -> bool:
        return self._exclusive

    async def execute(self, **kwargs):
        self._shared_events.append(f"start:{self._name}")
        await asyncio.sleep(self._delay)
        self._shared_events.append(f"end:{self._name}")
        return self._name


class _LegacyErrorPluginTool(Tool):
    @property
    def name(self) -> str:
        return "legacy_plugin"

    @property
    def description(self) -> str:
        return "legacy entry-point plugin"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        return "Error: legacy plugin failed"


class _StructuredSuccessPluginTool(Tool):
    @property
    def name(self) -> str:
        return "structured_success_plugin"

    @property
    def description(self) -> str:
        return "structured entry-point plugin"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        return ToolResult("Error: generated report successfully")


class _RawErrorPrefixTool(Tool):
    @property
    def name(self) -> str:
        return "raw_error_prefix"

    @property
    def description(self) -> str:
        return "successful unstructured output beginning with Error"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        return "Error: generated report successfully"


class _StructuredErrorTool(Tool):
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    @property
    def name(self) -> str:
        return "structured_error"

    @property
    def description(self) -> str:
        return "structured terminal tool failure"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        self.calls.append(self.name)
        return ToolOutcome(
            ToolResult.error("Error: structured failure"),
            policy_metadata={"source": "fixture"},
        )


async def _run_optional_tool_response(response: LLMResponse):
    provider = MagicMock()
    calls = {"n": 0}

    async def chat_with_retry(*, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return response
        return LLMResponse(content="done", tool_calls=[], usage=None)

    provider.chat_with_retry = chat_with_retry
    tools = ToolRegistry()
    shared_events: list[str] = []
    tools.register(_DelayTool(
        "optional_tool",
        delay=0,
        read_only=True,
        shared_events=shared_events,
    ))

    result = await AgentRunner().run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "try optional"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))
    return result, shared_events


def _load_entry_point_plugin(tool_cls: type[Tool], tmp_path) -> ToolRegistry:
    mock_ep = MagicMock()
    mock_ep.name = tool_cls.__name__
    mock_ep.load.return_value = tool_cls

    registry = ToolRegistry()
    with (
        patch("kohaku.migration_guard.require_runtime_ready"),
        patch("kohaku.agent.tools.fork_factories.construct_core_tools", return_value=[]),
        patch("kohaku.agent.tools.loader.entry_points", return_value=[mock_ep]),
    ):
        ToolLoader(test_classes=[]).load(ToolContext(config=ToolsConfig(), workspace=str(tmp_path)), registry)
    return registry


def _tool_message(result, tool_call_id: str) -> dict:
    return [
        msg for msg in result.messages
        if msg.get("role") == "tool" and msg.get("tool_call_id") == tool_call_id
    ][0]


@pytest.mark.asyncio
async def test_tool_execution_propagates_preparation_failure():
    tools = MagicMock()
    tools.prepare_call.side_effect = RuntimeError("tool preparation failed")
    tools.execute = AsyncMock()

    with pytest.raises(RuntimeError, match="tool preparation failed"):
        await execute_tool_calls(
            tools,
            [ToolCallRequest(id="call-1", name="demo", arguments={})],
            concurrent=False,
            external_lookup_counts={},
            workspace_violation_counts={},
            hook=AgentHook(),
            context=AgentHookContext(iteration=0, messages=[]),
        )

    tools.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_tool_execution_propagates_cancellation_without_error_hook():
    tools = MagicMock()
    tools.prepare_call.return_value = (None, {}, None)
    tools.execute = AsyncMock(side_effect=asyncio.CancelledError)
    install_test_tool_dispatch(tools)

    events: list[str] = []

    class RecordingHook(AgentHook):
        async def before_execute_tool(
            self,
            context: AgentHookContext,
            tool_call: ToolCallRequest,
            tool: Any,
            params: Any,
        ) -> None:
            events.append("before")

        async def on_execute_tool_error(
            self,
            context: AgentHookContext,
            tool_call: ToolCallRequest,
            tool: Any,
            params: Any,
            error: Any,
        ) -> None:
            events.append("error")

    with pytest.raises(asyncio.CancelledError):
        await execute_tool_calls(
            tools,
            [ToolCallRequest(id="call-1", name="demo", arguments={})],
            concurrent=False,
            external_lookup_counts={},
            workspace_violation_counts={},
            hook=RecordingHook(),
            context=AgentHookContext(iteration=0, messages=[]),
        )

    assert events == ["before"]


@pytest.mark.asyncio
async def test_tool_execution_batches_read_only_tools_before_exclusive_work():
    tools = ToolRegistry()
    shared_events: list[str] = []
    read_a = _DelayTool("read_a", delay=0.05, read_only=True, shared_events=shared_events)
    read_b = _DelayTool("read_b", delay=0.05, read_only=True, shared_events=shared_events)
    write_a = _DelayTool("write_a", delay=0.01, read_only=False, shared_events=shared_events)
    tools.register(read_a)
    tools.register(read_b)
    tools.register(write_a)

    await execute_tool_calls(
        tools,
        [
            ToolCallRequest(id="ro1", name="read_a", arguments={}),
            ToolCallRequest(id="ro2", name="read_b", arguments={}),
            ToolCallRequest(id="rw1", name="write_a", arguments={}),
        ],
        concurrent=True,
        external_lookup_counts={},
        workspace_violation_counts={},
        hook=AgentHook(),
        context=AgentHookContext(iteration=0, messages=[]),
    )

    assert shared_events[0:2] == ["start:read_a", "start:read_b"]
    assert "end:read_a" in shared_events and "end:read_b" in shared_events
    assert shared_events.index("end:read_a") < shared_events.index("start:write_a")
    assert shared_events.index("end:read_b") < shared_events.index("start:write_a")
    assert shared_events[-2:] == ["start:write_a", "end:write_a"]


@pytest.mark.asyncio
async def test_tool_execution_does_not_batch_exclusive_read_only_tools():
    tools = ToolRegistry()
    shared_events: list[str] = []
    read_a = _DelayTool("read_a", delay=0.03, read_only=True, shared_events=shared_events)
    read_b = _DelayTool("read_b", delay=0.03, read_only=True, shared_events=shared_events)
    ddg_like = _DelayTool(
        "ddg_like",
        delay=0.01,
        read_only=True,
        shared_events=shared_events,
        exclusive=True,
    )
    tools.register(read_a)
    tools.register(ddg_like)
    tools.register(read_b)

    await execute_tool_calls(
        tools,
        [
            ToolCallRequest(id="ro1", name="read_a", arguments={}),
            ToolCallRequest(id="ddg1", name="ddg_like", arguments={}),
            ToolCallRequest(id="ro2", name="read_b", arguments={}),
        ],
        concurrent=True,
        external_lookup_counts={},
        workspace_violation_counts={},
        hook=AgentHook(),
        context=AgentHookContext(iteration=0, messages=[]),
    )

    assert shared_events[0] == "start:read_a"
    assert shared_events.index("end:read_a") < shared_events.index("start:ddg_like")
    assert shared_events.index("end:ddg_like") < shared_events.index("start:read_b")


@pytest.mark.asyncio
async def test_runner_rejects_near_miss_tool_name_without_executing():
    provider = MagicMock()
    call_count = {"n": 0}
    captured_second_call: list[dict] = []

    async def chat_with_retry(*, messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="readFile",
                        arguments={"path": "notes.txt"},
                    )
                ],
                finish_reason="tool_calls",
                usage=None,
            )
        captured_second_call[:] = messages
        return LLMResponse(content="done", tool_calls=[], usage=None)

    provider.chat_with_retry = chat_with_retry
    tools = ToolRegistry()
    shared_events: list[str] = []
    tools.register(_DelayTool(
        "read_file",
        delay=0,
        read_only=True,
        shared_events=shared_events,
    ))

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "read notes"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.final_content == "done"
    assert result.tools_used == []
    assert shared_events == []
    assistant_message = [
        msg for msg in result.messages
        if msg.get("role") == "assistant" and msg.get("tool_calls")
    ][0]
    assert assistant_message["tool_calls"][0]["function"]["name"] == "readFile"
    tool_message = [
        msg for msg in result.messages
        if msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1"
    ][0]
    assert tool_message["name"] == "readFile"
    assert "Tool 'readFile' not found" in tool_message["content"]
    assert "Did you mean 'read_file'?" in tool_message["content"]
    replayed_assistant = [
        msg for msg in captured_second_call
        if msg.get("role") == "assistant" and msg.get("tool_calls")
    ][0]
    assert replayed_assistant["tool_calls"][0]["function"]["name"] == "readFile"


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ['{path:"notes.txt"}', "null"])
async def test_runner_rejects_openai_compat_invalid_arguments_without_executing(arguments):
    with patch("kohaku.providers.openai_compat_provider.AsyncOpenAI"):
        parsed = OpenAICompatProvider()._parse({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "optional_tool",
                            "arguments": arguments,
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        })

    result, shared_events = await _run_optional_tool_response(parsed)

    assert result.final_content == "done"
    assert parsed.tool_calls[0].arguments == arguments
    assert result.tools_used == []
    assert shared_events == []
    tool_message = _tool_message(result, "call_1")
    assert "parameters must be a JSON object" in tool_message["content"]


@pytest.mark.asyncio
async def test_runner_rejects_openai_responses_malformed_arguments_without_executing():
    parsed = parse_response_output({
        "output": [{
            "type": "function_call",
            "call_id": "call_1",
            "id": "fc_1",
            "name": "optional_tool",
            "arguments": "{bad",
        }],
        "status": "completed",
        "usage": {},
    })

    result, shared_events = await _run_optional_tool_response(parsed)

    assert result.final_content == "done"
    assert parsed.tool_calls[0].arguments == "{bad"
    assert result.tools_used == []
    assert shared_events == []
    tool_message = _tool_message(result, "call_1|fc_1")
    assert "parameters must be a JSON object" in tool_message["content"]


@pytest.mark.asyncio
async def test_runner_rejects_openai_responses_array_arguments_without_executing():
    parsed = parse_response_output({
        "output": [{
            "type": "function_call",
            "call_id": "call_1",
            "id": "fc_1",
            "name": "optional_tool",
            "arguments": [],
        }],
        "status": "completed",
        "usage": {},
    })

    result, shared_events = await _run_optional_tool_response(parsed)

    assert result.final_content == "done"
    assert parsed.tool_calls[0].arguments == []
    assert result.tools_used == []
    assert shared_events == []
    tool_message = _tool_message(result, "call_1|fc_1")
    assert "parameters must be a JSON object" in tool_message["content"]


@pytest.mark.asyncio
async def test_runner_returns_legacy_entry_point_error_to_model(tmp_path):
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="working",
            tool_calls=[ToolCallRequest(id="call_1", name="legacy_plugin", arguments={})],
            usage=None,
        ),
        LLMResponse(content="reported plugin failure", tool_calls=[], usage=None),
    ])

    result = await AgentRunner().run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "run plugin"}],
        tools=_load_entry_point_plugin(_LegacyErrorPluginTool, tmp_path),
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.stop_reason == "completed"
    assert result.final_content == "reported plugin failure"
    assert result.tool_events == [
        {"name": "legacy_plugin", "status": "error", "detail": "Error: legacy plugin failed"}
    ]


@pytest.mark.asyncio
async def test_runner_preserves_structured_plugin_success_that_starts_with_error(tmp_path):
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="working",
            tool_calls=[
                ToolCallRequest(id="call_1", name="structured_success_plugin", arguments={})
            ],
            usage=None,
        ),
        LLMResponse(content="done", tool_calls=[], usage=None),
    ])

    result = await AgentRunner().run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "run plugin"}],
        tools=_load_entry_point_plugin(_StructuredSuccessPluginTool, tmp_path),
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.stop_reason == "completed"
    assert result.tool_events == [
        {
            "name": "structured_success_plugin",
            "status": "ok",
            "detail": "Error: generated report successfully",
        }
    ]


@pytest.mark.asyncio
async def test_runner_continues_after_unstructured_success_that_starts_with_error():
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(
            content="working",
            tool_calls=[ToolCallRequest(id="call_1", name="raw_error_prefix", arguments={})],
            usage=None,
        ),
        LLMResponse(content="final response", tool_calls=[], usage=None),
    ])
    tools = ToolRegistry()
    tools.register(_RawErrorPrefixTool())

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "produce a report"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        fail_on_tool_error=True,
        turn_context=TurnContext(run_id="raw-success-run"),
    ))

    assert result.stop_reason == "completed"
    assert result.final_content == "final response"
    assert provider.chat_with_retry.await_count == 2
    assert result.tool_events == [
        {
            "name": "raw_error_prefix",
            "status": "ok",
            "detail": "Error: generated report successfully",
        }
    ]
    tool_message = _tool_message(result, "call_1")
    assert tool_message["content"] == "Error: generated report successfully"
    assert "Analyze the error above" not in tool_message["content"]


@pytest.mark.asyncio
async def test_runner_stops_after_structured_tool_error_before_later_tool_or_model_call():
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="working",
        tool_calls=[
            ToolCallRequest(id="call_1", name="structured_error", arguments={}),
            ToolCallRequest(id="call_2", name="later_tool", arguments={}),
        ],
        usage=None,
    ))
    calls: list[str] = []
    tools = ToolRegistry()
    tools.register(_StructuredErrorTool(calls))
    tools.register(_DelayTool(
        "later_tool",
        delay=0,
        read_only=True,
        shared_events=calls,
    ))

    result = await AgentRunner().run(make_run_spec(
        provider,
        initial_messages=[{"role": "user", "content": "do work"}],
        tools=tools,
        model="test-model",
        max_iterations=2,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
        fail_on_tool_error=True,
        turn_context=TurnContext(run_id="structured-error-run"),
    ))

    assert result.stop_reason == "tool_error"
    assert result.policy_metadata == {"source": "fixture"}
    assert provider.chat_with_retry.await_count == 1
    assert calls == ["structured_error"]
    assert result.tool_events == [
        {"name": "structured_error", "status": "error", "detail": "Error: structured failure"},
        {"name": "later_tool", "status": "skipped", "detail": "tool batch stopped"},
    ]
    assert "Analyze the error above" in _tool_message(result, "call_1")["content"]
    assert "Skipped" in _tool_message(result, "call_2")["content"]


@pytest.mark.asyncio
async def test_runner_blocks_repeated_external_fetches():
    provider = MagicMock()
    captured_final_call: list[dict] = []
    call_count = {"n": 0}

    async def chat_with_retry(*, messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 3:
            return LLMResponse(
                content="working",
                tool_calls=[ToolCallRequest(id=f"call_{call_count['n']}", name="web_fetch", arguments={"url": "https://example.com"})],
                usage=None,
            )
        captured_final_call[:] = messages
        return LLMResponse(content="done", tool_calls=[], usage=None)

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value="page content")

    runner = AgentRunner()
    result = await runner.run(make_run_spec(provider,
        initial_messages=[{"role": "user", "content": "research task"}],
        tools=tools,
        model="test-model",
        max_iterations=4,
        max_tool_result_chars=_MAX_TOOL_RESULT_CHARS,
    ))

    assert result.final_content == "done"
    assert tools.execute.await_count == 2
    blocked_tool_message = [
        msg for msg in captured_final_call
        if msg.get("role") == "tool" and msg.get("tool_call_id") == "call_3"
    ][0]
    assert "repeated external lookup blocked" in blocked_tool_message["content"]


@pytest.mark.asyncio
async def test_concurrent_read_only_batch_with_turn_context_policy_and_ordered_recording():
    from kohaku.agent.policy import ToolPolicy
    from kohaku.agent.tools.execution import KnownToolResult

    read_b_started = asyncio.Event()
    read_b_finished = asyncio.Event()
    observed_intents: dict[str, object] = {}
    mutating_ran: list[str] = []

    class _BarrierReadTool(Tool):
        def __init__(self, tool_name: str, *, wait_for_sibling: bool, fail: bool = False) -> None:
            self._name = tool_name
            self._wait_for_sibling = wait_for_sibling
            self._fail = fail

        @property
        def name(self) -> str:
            return self._name

        @property
        def description(self) -> str:
            return self._name

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}, "required": []}

        @property
        def read_only(self) -> bool:
            return True

        async def execute(self, **kwargs: Any) -> Any:
            return self._name

        async def execute_with_context(self, context: Any, **kwargs: Any) -> Any:
            if self._wait_for_sibling:
                # Proves genuine concurrency: read_a cannot finish until read_b starts and finishes.
                await asyncio.wait_for(read_b_started.wait(), timeout=1.0)
                await asyncio.wait_for(read_b_finished.wait(), timeout=1.0)
                observed_intents[self._name] = context.active_tool_intent_binding
                return "read_a_ok"
            read_b_started.set()
            observed_intents[self._name] = context.active_tool_intent_binding
            read_b_finished.set()
            if self._fail:
                return ToolResult.error("Error: read_b failed")
            return "read_b_ok"

    class _WriteTool(Tool):
        @property
        def name(self) -> str:
            return "write_a"

        @property
        def description(self) -> str:
            return "write_a"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}, "required": []}

        async def execute(self, **kwargs: Any) -> str:
            mutating_ran.append("write_a")
            return "written"

    tools = ToolRegistry()
    tools.register(_BarrierReadTool("read_a", wait_for_sibling=True))
    tools.register(_BarrierReadTool("read_b", wait_for_sibling=False, fail=True))
    tools.register(_WriteTool())

    recorded: list[KnownToolResult] = []
    hook_order: list[tuple[str, str]] = []

    class _OrderHook(AgentHook):
        async def after_execute_tool(self, context, tool_call, tool, params, result) -> None:
            hook_order.append(("after", tool_call.id))

        async def on_execute_tool_error(self, context, tool_call, tool, params, error) -> None:
            hook_order.append(("error", tool_call.id))

    async def on_result(record: KnownToolResult) -> None:
        recorded.append(record)

    turn_context = TurnContext(run_id="concurrent-run")
    intent_0 = object()
    intent_1 = object()
    intent_2 = object()

    results, events = await execute_tool_calls(
        tools,
        [
            ToolCallRequest(id="c0", name="read_a", arguments={}),
            ToolCallRequest(id="c1", name="read_b", arguments={}),
            ToolCallRequest(id="c2", name="write_a", arguments={}),
        ],
        concurrent=True,
        external_lookup_counts={},
        workspace_violation_counts={},
        hook=_OrderHook(),
        context=AgentHookContext(iteration=0, messages=[]),
        tool_policy=ToolPolicy(),
        turn_context=turn_context,
        fail_on_tool_error=True,
        tool_result_callback=on_result,
        tool_intent_bindings=(intent_0, intent_1, intent_2),  # type: ignore[arg-type]
    )

    # Both read-only calls ran concurrently and saw isolated task-local intent bindings.
    assert observed_intents == {"read_a": intent_0, "read_b": intent_1}
    assert turn_context.active_tool_intent_binding is None
    # Recording and post-execution hooks flushed in strict ordinal order despite read_b finishing first.
    assert [(r.ordinal, r.tool_call_id, r.dispatched) for r in recorded] == [
        (0, "c0", True),
        (1, "c1", True),
        (2, "c2", False),
    ]
    assert hook_order == [("after", "c0"), ("error", "c1")]
    # Error in concurrent read-only batch 1 skipped mutating batch 2.
    assert mutating_ran == []
    assert results[0] == "read_a_ok"
    assert isinstance(results[1], ToolOutcome)
    assert results[1].stop_reason == "tool_error"
    assert "Skipped" in str(results[2])
    assert [e["status"] for e in events] == ["ok", "error", "skipped"]
