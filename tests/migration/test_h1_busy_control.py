"""Offline H1 routing tests: no task execution or channel activation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from nanobot.agent.busy_control import (
    BusyControlRouter,
    ControlDecision,
    ControlRequest,
    ControlSnapshot,
    ProviderControlModel,
    RoutingProposal,
)
from nanobot.config.schema import BusyControlConfig
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


@dataclass
class ScriptedModel:
    proposal: RoutingProposal = field(
        default_factory=lambda: RoutingProposal(intent="steer", confidence=0.95)
    )
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event | None = None
    calls: list[ControlRequest] = field(default_factory=list)
    error: bool = False
    resist_cancellation: bool = False

    async def classify(self, request: ControlRequest) -> RoutingProposal:
        self.calls.append(request)
        self.started.set()
        if self.release is not None:
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                if not self.resist_cancellation:
                    raise
                await self.release.wait()
        if self.error:
            raise ValueError("sensitive provider failure that must not be returned")
        return self.proposal


def snapshot(run_id: str = "run-a", generation: int = 1) -> ControlSnapshot:
    return ControlSnapshot(run_id=run_id, generation=generation, state="running", queued=2)


@pytest.mark.parametrize("intent", ["steer", "next-turn", "stop", "stop-and-replace", "status", "clarify"])
async def test_router_only_proposes_a_generation_bound_route(intent: str) -> None:
    model = ScriptedModel(RoutingProposal.model_validate({"intent": intent, "confidence": 0.95}))
    router = BusyControlRouter(model, BusyControlConfig())
    view = snapshot()
    result = await router.propose("完整消息 — keep all these details", view)
    assert result.intent == intent
    assert result.observed_run == view
    assert result.reason == "classified"
    assert model.calls[0].content == "完整消息 — keep all these details"
    assert not hasattr(model.calls[0], "tools")
    assert not hasattr(model.calls[0], "history")
    await router.aclose()


async def test_explicit_direct_stop_bypasses_inference() -> None:
    model = ScriptedModel()
    router = BusyControlRouter(model, BusyControlConfig())
    result = await router.propose(" /STOP ", snapshot())
    assert result.intent == "stop" and result.reason == "explicit-stop"
    assert not model.calls
    await router.aclose()


@pytest.mark.parametrize("text", ['He said "/stop"', "> /stop", "转发：/stop", "someone said stop"])
async def test_quoted_forwarded_control_is_not_an_explicit_stop(text: str) -> None:
    model = ScriptedModel(RoutingProposal(intent="stop", confidence=0.99))
    router = BusyControlRouter(model, BusyControlConfig())
    result = await router.propose(text, snapshot(), direct=False)
    assert result.intent == "clarify" and result.reason == "ambiguous"
    assert len(model.calls) == 1
    await router.aclose()


async def test_idle_message_is_next_turn_without_auxiliary_cost() -> None:
    model = ScriptedModel()
    router = BusyControlRouter(model, BusyControlConfig())
    result = await router.propose("a new task", None)
    assert result == ControlDecision(intent="next-turn", observed_run=None, reason="idle")
    assert not model.calls
    await router.aclose()


async def test_low_confidence_does_not_guess_stop_or_steer() -> None:
    model = ScriptedModel(RoutingProposal(intent="steer", confidence=0.2))
    router = BusyControlRouter(model, BusyControlConfig())
    assert (await router.propose("perhaps something else", snapshot())).intent == "clarify"
    await router.aclose()


async def test_provider_error_is_honest_queue_fallback() -> None:
    router = BusyControlRouter(ScriptedModel(error=True), BusyControlConfig())
    result = await router.propose("new input", snapshot())
    assert result.intent == "next-turn" and result.reason == "error"
    assert "sensitive" not in repr(result)
    await router.aclose()


async def test_timeout_returns_while_resistant_provider_retains_capacity() -> None:
    release = asyncio.Event()
    model = ScriptedModel(release=release, resist_cancellation=True)
    router = BusyControlRouter(model, BusyControlConfig(max_concurrent=1, timeout_seconds=0.02))
    result = await router.propose("first", snapshot())
    assert result.intent == "next-turn" and result.reason == "timeout"
    assert router.active_calls == 1
    second = await router.propose("second", snapshot())
    assert second.reason == "overloaded" and second.intent == "next-turn"
    assert len(model.calls) == 1
    release.set()
    await router.aclose()
    assert router.active_calls == 0


async def test_task_provider_block_does_not_block_control_inference() -> None:
    task_block = asyncio.Event()
    task = asyncio.create_task(task_block.wait())
    router = BusyControlRouter(ScriptedModel(), BusyControlConfig())
    try:
        result = await router.propose("add this to the current task", snapshot())
        assert result.intent == "steer"
        assert not task.done()
    finally:
        task_block.set()
        await task
        await router.aclose()


async def test_out_of_order_results_keep_their_original_run_generation() -> None:
    model = ScriptedModel(release=asyncio.Event())
    router = BusyControlRouter(model, BusyControlConfig())
    older = asyncio.create_task(router.propose("older", snapshot("old", 7)))
    await model.started.wait()
    model.release.set()
    result = await older
    assert result.observed_run == snapshot("old", 7)
    assert result.observed_run != snapshot("new", 8)
    await router.aclose()


async def test_rate_limit_does_not_create_an_unbounded_waiter_queue() -> None:
    clock = [0.0]
    model = ScriptedModel()
    router = BusyControlRouter(model, BusyControlConfig(max_calls_per_window=1), clock=lambda: clock[0])
    assert (await router.propose("one", snapshot())).reason == "classified"
    assert (await router.propose("two", snapshot())).reason == "rate-limited"
    clock[0] = 60.0
    assert (await router.propose("three", snapshot())).reason == "classified"
    assert len(model.calls) == 2
    await router.aclose()


async def test_oversized_input_is_preserved_for_task_lane_not_truncated_for_control() -> None:
    model = ScriptedModel()
    router = BusyControlRouter(model, BusyControlConfig(max_input_bytes=8))
    result = await router.propose("这条消息有更多细节", snapshot())
    assert result.reason == "input-limit" and result.intent == "next-turn"
    assert not model.calls
    await router.aclose()


async def test_caller_cancellation_is_not_hidden_as_queue_fallback() -> None:
    model = ScriptedModel(release=asyncio.Event())
    router = BusyControlRouter(model, BusyControlConfig())
    task = asyncio.create_task(router.propose("cancel classification", snapshot()))
    await model.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await router.aclose()
    assert router.active_calls == 0


@pytest.mark.parametrize("settings", [
    {"maxConcurrent": 0}, {"maxConcurrent": True}, {"timeoutSeconds": 0},
    {"timeoutSeconds": float("nan")}, {"timeoutSeconds": float("inf")},
    {"maxCallsPerWindow": -1}, {"maxOutputTokens": 0}, {"maxInputBytes": 0},
    {"minimumConfidence": -0.1}, {"minimumConfidence": 1.1},
])
def test_limits_reject_invalid_values(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BusyControlConfig.model_validate(settings)


class ScriptedProvider(LLMProvider):
    def __init__(self, response: LLMResponse) -> None:
        super().__init__(provider_name="h1-offline")
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None) -> LLMResponse:
        self.calls.append(dict(messages=messages, tools=tools, model=model, max_tokens=max_tokens,
                               temperature=temperature, tool_choice=tool_choice))
        return self.response

    def get_default_model(self) -> str:
        return "h1-offline"


async def test_provider_adapter_is_stateless_tool_free_and_bounded() -> None:
    provider = ScriptedProvider(LLMResponse(content='{"intent":"steer","confidence":0.95}'))
    adapter = ProviderControlModel(provider, "routing-model")
    result = await adapter.classify(ControlRequest("user text", snapshot(), True, 128, 4096))
    assert result.intent == "steer"
    call = provider.calls[0]
    assert call["tools"] == [] and call["tool_choice"] == "none"
    assert call["max_tokens"] == 128 and call["model"] == "routing-model"
    assert call["temperature"] == 0
    assert "provider_state" not in call and "conversation_binding" not in call
    assert "user text" in json.dumps(call["messages"])
    assert "history" not in json.dumps(call["messages"])


@pytest.mark.parametrize("response", [
    LLMResponse(content='{"intent":"stop","confidence":0.9,"tools":["exec"]}'),
    LLMResponse(content='{"intent":"stop","intent":"steer","confidence":0.9}'),
    LLMResponse(content='{"intent":"unknown","confidence":0.9}'),
    LLMResponse(content='{"intent":"stop","confidence":"0.9"}'),
    LLMResponse(content='{"intent":"stop","confidence":NaN}'),
    LLMResponse(content="x" * 4097), LLMResponse(content=None),
    LLMResponse(content='{"intent":"stop","confidence":0.9}', finish_reason="error"),
    LLMResponse(content='{"intent":"stop","confidence":0.9}', tool_calls=[ToolCallRequest("id", "exec", {})]),
])
async def test_provider_adapter_rejects_malformed_or_effectful_responses(response: LLMResponse) -> None:
    router = BusyControlRouter(ProviderControlModel(ScriptedProvider(response), "routing-model"), BusyControlConfig())
    result = await router.propose("some input", snapshot())
    assert result.intent == "next-turn" and result.reason == "error"
    await router.aclose()
