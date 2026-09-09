"""Policy and terminal outcomes must survive the upstream execution path."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.runner_helpers import make_run_spec
from nanobot.agent.policy import DelegatedReadOnlyPolicy
from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import ToolOutcome, TurnContext
from nanobot.providers.base import LLMResponse, ToolCallRequest


class RecordingTool(Tool):
    def __init__(self, name, calls, outcome):
        self._name, self.calls, self.outcome = name, calls, outcome

    name = property(lambda self: self._name)
    description = property(lambda self: self._name)
    parameters = property(lambda self: {"type": "object", "properties": {}})

    async def execute(self, context=None):
        self.calls.append((self.name, context))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


@pytest.mark.parametrize("mode", ["policy", "terminal", "cancel"])
async def test_real_runner_stops_before_later_tools_and_model_calls(mode):
    calls = []
    context = TurnContext(run_id="test-run")
    tools = ToolRegistry()
    outcome = {"policy": "read", "terminal": ToolOutcome("blocked", "policy_blocked", {"role": "worker"}),
               "cancel": asyncio.CancelledError()}[mode]
    tools.register(RecordingTool("read_file", calls, outcome))
    tools.register(RecordingTool("write_file", calls, "written"))
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id="r", name="read_file", arguments={}),
        ToolCallRequest(id="w", name="write_file", arguments={}),
    ]))
    inject = AsyncMock(return_value=[])
    spec = make_run_spec(provider, initial_messages=[{"role": "user", "content": "inspect"}],
                         tools=tools, model="test", max_iterations=1, max_tool_result_chars=10000,
                         concurrent_tools=True, turn_context=context,
                         tool_policy=DelegatedReadOnlyPolicy(role="reviewer") if mode == "policy" else None,
                         injection_callback=inject)
    if mode == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await AgentRunner().run(spec)
    else:
        result = await AgentRunner().run(spec)
        assert result.stop_reason == "policy_blocked"
        assert result.policy_metadata["role"] == ("reviewer" if mode == "policy" else "worker")
        assert [m["tool_call_id"] for m in result.messages if m["role"] == "tool"] == ["r", "w"]
        assert "Skipped" in result.messages[-1]["content"]
    assert calls == ([] if mode == "policy" else [("read_file", context)])
    assert provider.chat_with_retry.await_count == 1
    inject.assert_not_awaited()


@pytest.mark.parametrize("outcome", [ToolOutcome("failed", "tool_error", {"reason": "fixture"}),
                                    "Error: failed", ToolResult.error("failed")])
async def test_tool_errors_preserve_fail_fast_semantics(outcome):
    calls = []
    tools = ToolRegistry()
    tools.register(RecordingTool("read_file", calls, outcome))
    tools.register(RecordingTool("write_file", calls, "written"))
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id="r", name="read_file", arguments={}),
        ToolCallRequest(id="w", name="write_file", arguments={}),
    ]))
    result = await AgentRunner().run(make_run_spec(
        provider, initial_messages=[], tools=tools, model="test", max_iterations=1,
        max_tool_result_chars=10000, fail_on_tool_error=True, concurrent_tools=True))
    assert result.stop_reason == "tool_error"
    assert calls == [("read_file", None)]
    assert provider.chat_with_retry.await_count == 1
    assert "Skipped" in result.messages[-1]["content"]
    if isinstance(outcome, ToolOutcome):
        assert result.policy_metadata == {"reason": "fixture"}


async def test_delegation_captures_crawler_capacity_without_mutating_provider(tmp_path):
    from nanobot.agent.delegation import ForegroundAgentManager
    from nanobot.agent.runner import AgentRunResult
    from nanobot.providers.base import GenerationSettings

    provider = MagicMock()
    provider.generation = GenerationSettings(max_tokens=4000, temperature=0.7)
    crawler = MagicMock()
    crawler.generation = GenerationSettings(max_tokens=3000, temperature=0.2)
    manager = ForegroundAgentManager(provider=provider, workspace=tmp_path, model="main", crawler_provider=crawler)
    manager.crawler_runner.run = AsyncMock(return_value=AgentRunResult("done", []))
    await manager._run_role(role="crawler", prompt="test", assignment="test", tools=ToolRegistry(),
                            max_iterations=2, runner=manager.crawler_runner, model="crawler",
                            max_input_tokens=20000, max_output_tokens=2000)
    spec = manager.crawler_runner.run.call_args.args[0]
    assert spec.runtime.provider is crawler
    assert spec.runtime.context_window_tokens == 20000
    assert spec.runtime.generation.max_tokens == 2000
    assert spec.runtime.generation.temperature == 0.2
    assert crawler.generation.max_tokens == 3000
    assert provider.generation.max_tokens == 4000
