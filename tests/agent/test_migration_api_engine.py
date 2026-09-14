"""Real-engine evidence for the canonical non-streaming HTTP API adapter."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

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


class _ActiveProviderCall(ScriptedProvider):
    """Block one real provider request until the HTTP timeout cancels it."""

    def __init__(self, *, cleanup_error: bool = False, wait_for_cleanup: bool = False) -> None:
        super().__init__([])
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.release_cleanup = asyncio.Event()
        self.finished = asyncio.Event()
        self.calls: list[str] = []
        self.cleanup_error = cleanup_error
        self.wait_for_cleanup = wait_for_cleanup

    async def chat(self, messages, tools=None, model=None, **kwargs):
        content = next(
            str(message.get("content", ""))
            for message in reversed(messages)
            if message.get("role") == "user"
        )
        self.calls.append(content)
        if content == "provider timeout":
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                self.cleanup_started.set()
                if self.wait_for_cleanup:
                    await self.release_cleanup.wait()
                raise
            finally:
                self.finished.set()
                if self.cleanup_error:
                    raise RuntimeError("provider cleanup failed")
        return LLMResponse(content=f"completed: {content}")


class _ActiveToolCall(Tool):
    def __init__(self, *, cleanup_error: bool = False, wait_for_cleanup: bool = False) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.cleanup_started = asyncio.Event()
        self.release_cleanup = asyncio.Event()
        self.finished = asyncio.Event()
        self.executions = 0
        self.completed_effects = 0
        self.cleanup_error = cleanup_error
        self.wait_for_cleanup = wait_for_cleanup

    @property
    def name(self) -> str:
        return "api_active_tool"

    @property
    def description(self) -> str:
        return "Block until cancellation without committing its side effect."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        self.executions += 1
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            self.cleanup_started.set()
            if self.wait_for_cleanup:
                await self.release_cleanup.wait()
            raise
        else:
            self.completed_effects += 1
            return "effect committed"
        finally:
            self.finished.set()
            if self.cleanup_error:
                raise RuntimeError("tool cleanup failed")


class _ActiveToolProvider(ScriptedProvider):
    def __init__(self) -> None:
        super().__init__([])
        self.calls: list[str] = []

    async def chat(self, messages, tools=None, model=None, **kwargs):
        content = next(
            str(message.get("content", ""))
            for message in reversed(messages)
            if message.get("role") == "user"
        )
        self.calls.append(content)
        if content == "tool timeout":
            return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id="active-tool",
                name="api_active_tool",
                arguments={},
            )])
        return LLMResponse(content=f"completed: {content}")


async def _post_with_timeout(
    loop: AgentLoop,
    content: str,
    *,
    session_id: str,
    timeout_s: float,
) -> tuple[int, dict[str, Any]]:
    request = _ApiRequest(loop, {
        "messages": [{"role": "user", "content": content}],
        "session_id": session_id,
    })
    request.app["request_timeout"] = timeout_s
    response = await handle_chat_completions(request)
    return response.status, json.loads(response.body)


async def _post_with_readiness_cleanup_error(
    prepare_agent: Any,
    *,
    timeout_s: float,
) -> tuple[Any, AsyncMock]:
    execute_direct = AsyncMock()
    request = _ApiRequest(SimpleNamespace(_execute_direct=execute_direct), {
        "messages": [{"role": "user", "content": "readiness"}],
    })
    request.app["request_timeout"] = timeout_s
    request.app["prepare_agent"] = prepare_agent
    return await handle_chat_completions(request), execute_direct


@pytest.mark.asyncio
async def test_api_timeout_keeps_504_when_readiness_cleanup_raises() -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def prepare_agent() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
            raise RuntimeError("readiness cleanup failed")

    request = asyncio.create_task(_post_with_readiness_cleanup_error(
        prepare_agent,
        timeout_s=0.02,
    ))
    await started.wait()
    response, execute_direct = await asyncio.wait_for(request, timeout=0.5)

    assert response.status == 504
    assert json.loads(response.body)["error"]["message"] == "Request timed out after 0.02s"
    assert cleaned.is_set()
    execute_direct.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_cancellation_reraises_when_readiness_cleanup_raises() -> None:
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def prepare_agent() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
            raise RuntimeError("readiness cleanup failed")

    request = asyncio.create_task(_post_with_readiness_cleanup_error(
        prepare_agent,
        timeout_s=5.0,
    ))
    await started.wait()
    request.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_real_api_timeout_cancels_active_provider_and_releases_its_session(engine_config) -> None:
    provider = _ActiveProviderCall()
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        timed_out = asyncio.create_task(_post_with_timeout(
            loop,
            "provider timeout",
            session_id="active-provider",
            timeout_s=0.02,
        ))
        await provider.started.wait()
        status, body = await asyncio.wait_for(timed_out, timeout=0.5)

        assert status == 504
        assert body["error"]["message"] == "Request timed out after 0.02s"
        await asyncio.wait_for(provider.finished.wait(), timeout=0.1)
        assert provider.cancelled.is_set()

        successor = await asyncio.wait_for(
            _post(loop, "provider successor", session_id="active-provider"),
            timeout=0.2,
        )
        assert successor["choices"][0]["message"]["content"] == "completed: provider successor"
        assert provider.calls == ["provider timeout", "provider successor"]
        assert not loop._get_session_lock("api:active-provider").locked()
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_real_api_timeout_keeps_504_when_provider_cleanup_raises(engine_config) -> None:
    provider = _ActiveProviderCall(cleanup_error=True)
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        status, body = await _post_with_timeout(
            loop,
            "provider timeout",
            session_id="provider-cleanup-error",
            timeout_s=0.02,
        )

        assert status == 504
        assert body["error"]["message"] == "Request timed out after 0.02s"
        await asyncio.wait_for(provider.finished.wait(), timeout=0.1)
        assert provider.cancelled.is_set()

        successor = await asyncio.wait_for(
            _post(loop, "provider successor", session_id="provider-cleanup-error"),
            timeout=0.2,
        )
        assert successor["choices"][0]["message"]["content"] == "completed: provider successor"
        assert provider.calls == ["provider timeout", "provider successor"]
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_real_api_cancellation_rejoins_provider_before_successor(engine_config) -> None:
    provider = _ActiveProviderCall(wait_for_cleanup=True)
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        request = asyncio.create_task(_post_with_timeout(
            loop,
            "provider timeout",
            session_id="provider-cancelled",
            timeout_s=5.0,
        ))
        await provider.started.wait()
        request.cancel()
        await provider.cleanup_started.wait()
        request.cancel()
        provider.release_cleanup.set()

        with pytest.raises(asyncio.CancelledError):
            await request
        await asyncio.wait_for(provider.finished.wait(), timeout=0.1)
        assert provider.cancelled.is_set()

        successor = await asyncio.wait_for(
            _post(loop, "provider successor", session_id="provider-cancelled"),
            timeout=0.2,
        )
        assert successor["choices"][0]["message"]["content"] == "completed: provider successor"
        assert provider.calls == ["provider timeout", "provider successor"]
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_real_api_timeout_cancels_active_tool_without_replay_and_releases_session(engine_config) -> None:
    provider = _ActiveToolProvider()
    tools = ToolRegistry()
    tool = _ActiveToolCall()
    tools.register(tool)
    loop = AgentLoop.from_config(engine_config, tool_registry=tools, provider=provider)
    try:
        timed_out = asyncio.create_task(_post_with_timeout(
            loop,
            "tool timeout",
            session_id="active-tool",
            timeout_s=0.02,
        ))
        await tool.started.wait()
        status, body = await asyncio.wait_for(timed_out, timeout=0.5)

        assert status == 504
        assert body["error"]["message"] == "Request timed out after 0.02s"
        await asyncio.wait_for(tool.finished.wait(), timeout=0.1)
        assert tool.cancelled.is_set()
        assert tool.executions == 1
        assert tool.completed_effects == 0

        successor = await asyncio.wait_for(
            _post(loop, "tool successor", session_id="active-tool"),
            timeout=0.2,
        )
        assert successor["choices"][0]["message"]["content"] == "completed: tool successor"
        assert provider.calls == ["tool timeout", "tool successor"]
        assert tool.executions == 1
        assert not loop._get_session_lock("api:active-tool").locked()
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_real_api_cancellation_rejoins_tool_before_successor(engine_config) -> None:
    provider = _ActiveToolProvider()
    tools = ToolRegistry()
    tool = _ActiveToolCall(wait_for_cleanup=True)
    tools.register(tool)
    loop = AgentLoop.from_config(engine_config, tool_registry=tools, provider=provider)
    try:
        request = asyncio.create_task(_post_with_timeout(
            loop,
            "tool timeout",
            session_id="tool-cancelled",
            timeout_s=5.0,
        ))
        await tool.started.wait()
        request.cancel()
        await tool.cleanup_started.wait()
        request.cancel()
        tool.release_cleanup.set()

        with pytest.raises(asyncio.CancelledError):
            await request
        await asyncio.wait_for(tool.finished.wait(), timeout=0.1)
        assert tool.cancelled.is_set()
        assert tool.executions == 1
        assert tool.completed_effects == 0

        successor = await asyncio.wait_for(
            _post(loop, "tool successor", session_id="tool-cancelled"),
            timeout=0.2,
        )
        assert successor["choices"][0]["message"]["content"] == "completed: tool successor"
        assert provider.calls == ["tool timeout", "tool successor"]
        assert tool.executions == 1
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_real_api_timeout_keeps_504_when_tool_cleanup_raises(engine_config) -> None:
    provider = _ActiveToolProvider()
    tools = ToolRegistry()
    tool = _ActiveToolCall(cleanup_error=True)
    tools.register(tool)
    loop = AgentLoop.from_config(engine_config, tool_registry=tools, provider=provider)
    try:
        status, body = await _post_with_timeout(
            loop,
            "tool timeout",
            session_id="tool-cleanup-error",
            timeout_s=0.02,
        )

        assert status == 504
        assert body["error"]["message"] == "Request timed out after 0.02s"
        await asyncio.wait_for(tool.finished.wait(), timeout=0.1)
        assert tool.cancelled.is_set()
        assert tool.executions == 1
        assert tool.completed_effects == 0

        successor = await asyncio.wait_for(
            _post(loop, "tool successor", session_id="tool-cleanup-error"),
            timeout=0.2,
        )
        assert successor["choices"][0]["message"]["content"] == "completed: tool successor"
        assert provider.calls == ["tool timeout", "tool successor"]
        assert tool.executions == 1
    finally:
        await loop.aclose()
