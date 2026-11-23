"""A successful tool call that repeats unchanged must not spin forever."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kohaku.agent.runner import AgentRunner, AgentRunSpec
from kohaku.agent.tools.base import Tool
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.agent.turn import RunStatus
from kohaku.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from kohaku.utils.llm_runtime import LLMRuntime


class _FixedTool(Tool):
    """Return a fixed result so a repeated call is genuinely a no-op cycle."""

    def __init__(self, results: list[str]) -> None:
        self._results = results
        self.calls = 0

    @property
    def name(self) -> str:
        return "noop_tool"

    @property
    def description(self) -> str:
        return "returns a fixed value"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"value": {"type": "string"}}}

    async def execute(self, **kwargs: Any) -> str:
        index = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[index]


def _provider() -> MagicMock:
    """A provider that always requests the same tool call and no answer."""
    provider = MagicMock(spec=LLMProvider)
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="",
        tool_calls=[ToolCallRequest(
            id="repeat",
            name="noop_tool",
            arguments={"value": "same"},
        )],
        usage=None,
    ))
    provider.get_default_model.return_value = "test-model"
    return provider


@pytest.mark.asyncio
async def test_identical_successful_tool_result_stops_the_turn() -> None:
    """The turn must end instead of running to the iteration ceiling."""
    provider = _provider()
    tool = _FixedTool(["unchanged output"] * 50)
    registry = ToolRegistry()
    registry.register(tool)

    result = await AgentRunner().run(AgentRunSpec(
        initial_messages=[],
        tools=registry,
        runtime=LLMRuntime.capture(provider, "test-model", context_window_tokens=128_000),
        max_iterations=50,
        max_tool_result_chars=16_000,
    ))

    assert result.stop_reason == RunStatus.REPEATED_NO_PROGRESS.value
    # It must stop well before the iteration ceiling.
    assert tool.calls < 10


@pytest.mark.asyncio
async def test_changing_tool_result_does_not_stop_the_turn() -> None:
    """Real progress must never be mistaken for a stall."""
    provider = _provider()
    tool = _FixedTool([f"value {index}" for index in range(50)])
    registry = ToolRegistry()
    registry.register(tool)

    result = await AgentRunner().run(AgentRunSpec(
        initial_messages=[],
        tools=registry,
        runtime=LLMRuntime.capture(provider, "test-model", context_window_tokens=128_000),
        max_iterations=8,
        max_tool_result_chars=16_000,
    ))

    assert result.stop_reason != RunStatus.REPEATED_NO_PROGRESS.value
