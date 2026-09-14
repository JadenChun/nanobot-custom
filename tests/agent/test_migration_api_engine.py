"""Real-engine evidence for the canonical non-streaming HTTP API adapter."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agent import test_migration_engine as engine_fixtures
from agent.test_migration_engine import ScriptedProvider
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.api.server import handle_chat_completions
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

engine_config = engine_fixtures.engine_config


class _ApiRequest:
    """Minimal aiohttp request boundary for an offline handler invocation."""

    content_type = "application/json"

    def __init__(self, loop: AgentLoop, body: dict[str, Any]) -> None:
        self.app = {
            "agent_loop": loop,
            "model_name": "synthetic-model",
            "request_timeout": 1.0,
        }
        self._body = body

    async def json(self) -> dict[str, Any]:
        return self._body


async def _post(loop: AgentLoop, content: str, session_id: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"messages": [{"role": "user", "content": content}]}
    if session_id is not None:
        body["session_id"] = session_id
    response = await handle_chat_completions(_ApiRequest(loop, body))
    assert response.status == 200
    return json.loads(response.body)


class _BlockingProvider(ScriptedProvider):
    """A deterministic provider that exposes when real runner calls overlap."""

    def __init__(self, names: tuple[str, ...]) -> None:
        super().__init__([])
        self.started = {name: asyncio.Event() for name in names}
        self.release = {name: asyncio.Event() for name in names}
        self.calls: list[str] = []

    async def chat(self, messages, tools=None, model=None, **kwargs):
        content = next(
            str(message.get("content", ""))
            for message in reversed(messages)
            if message.get("role") == "user"
        )
        name = next(name for name in self.started if content.rstrip().endswith(name))
        self.calls.append(name)
        self.started[name].set()
        await self.release[name].wait()
        return LLMResponse(content=f"done: {name}")


@pytest.mark.asyncio
async def test_real_api_engine_serializes_a_session_and_overlaps_distinct_sessions(engine_config) -> None:
    provider = _BlockingProvider(("first", "second", "left", "right"))
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        first = asyncio.create_task(_post(loop, "first"))
        await provider.started["first"].wait()
        second = asyncio.create_task(_post(loop, "second"))
        await asyncio.sleep(0)
        assert provider.calls == ["first"]

        provider.release["first"].set()
        await provider.started["second"].wait()
        provider.release["second"].set()
        first_body, second_body = await asyncio.gather(first, second)
        assert [
            first_body["choices"][0]["message"]["content"],
            second_body["choices"][0]["message"]["content"],
        ] == ["done: first", "done: second"]

        left = asyncio.create_task(_post(loop, "left", session_id="left"))
        right = asyncio.create_task(_post(loop, "right", session_id="right"))
        await asyncio.gather(provider.started["left"].wait(), provider.started["right"].wait())
        assert provider.calls[-2:] == ["left", "right"]

        provider.release["left"].set()
        provider.release["right"].set()
        left_body, right_body = await asyncio.gather(left, right)
        assert left_body["choices"][0]["message"]["content"] == "done: left"
        assert right_body["choices"][0]["message"]["content"] == "done: right"
    finally:
        await loop.aclose()


class _CountingSideEffectTool(Tool):
    def __init__(self) -> None:
        self.executions = 0

    @property
    def name(self) -> str:
        return "api_counting_effect"

    @property
    def description(self) -> str:
        return "Perform one test-only durable side effect."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        self.executions += 1
        return "effect recorded"


@pytest.mark.asyncio
async def test_real_api_runner_does_not_replay_a_tool_after_blank_finalization(engine_config) -> None:
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="effect-1",
            name="api_counting_effect",
            arguments={},
        )]),
        LLMResponse(content=""),
        LLMResponse(content=""),
        LLMResponse(content=""),
        LLMResponse(content="lock remains usable"),
    ])
    tools = ToolRegistry()
    effect = _CountingSideEffectTool()
    tools.register(effect)
    loop = AgentLoop.from_config(engine_config, tool_registry=tools, provider=provider)
    try:
        fallback = await _post(loop, "perform the effect", session_id="effects")
        assert fallback["choices"][0]["message"]["content"] == EMPTY_FINAL_RESPONSE_MESSAGE
        assert effect.executions == 1
        assert len(provider.calls) == 4

        reusable = await _post(loop, "continue", session_id="effects")
        assert reusable["choices"][0]["message"]["content"] == "lock remains usable"
        assert effect.executions == 1
        assert len(provider.calls) == 5
    finally:
        await loop.aclose()
