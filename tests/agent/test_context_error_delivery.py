"""Local request-budget failures retain their cause through real source turns."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.loop import AgentLoop, _TurnExecutionOptions
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import TurnCallbacks, TurnRequest
from nanobot.bus.events import InboundMessage
from nanobot.bus.outbound_events import TurnEndEvent
from nanobot.providers.base import GenerationSettings
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.session.webui_turns import WebuiTurnCoordinator
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_migration_source_engine import upgrade

engine_config = engine_fixtures.engine_config


@pytest.mark.parametrize("streaming", [False, True])
async def test_final_budget_failure_releases_source_ticket_and_reports_context(
    engine_config, streaming,
):
    async def chunks():
        yield SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Recovered."), finish_reason="stop",
        )], usage=None)

    async def create(**kwargs):
        if kwargs.get("stream"):
            return chunks()
        return {"choices": [{"message": {"content": "Recovered."}, "finish_reason": "stop"}]}

    provider = OpenAICompatProvider(
        api_key="fixture", provider_name="budget-fixture", api_type="chat_completions",
        extra_body={"metadata": {"fixture": "private-payload-" + "x" * 200_000}},
    )
    transport = AsyncMock(side_effect=create)
    provider._ensure_client = AsyncMock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=transport)),
    ))
    provider.generation = GenerationSettings(max_tokens=256)
    loop = AgentLoop.from_config(
        engine_config, tool_registry=ToolRegistry(), provider=provider,
        context_window_tokens=100_000,
    )
    key = "websocket:context-budget-source"
    backend = upgrade(loop, key)
    coordinator = WebuiTurnCoordinator(
        bus=loop.bus, sessions=loop.sessions, schedule_background=lambda coro: coro.close(),
    )
    try:
        msg = InboundMessage(
            channel="websocket", sender_id="user", chat_id="context-budget-source",
            content="Check this request",
            metadata={"webui": True, "webui_turn_id": "budget-turn"},
        )
        with coordinator.connected():
            failed = await loop.execute_turn(TurnRequest(
                content="Check this request", session_key=key,
                metadata={"webui": True, "webui_turn_id": "budget-turn"},
                callbacks=TurnCallbacks(on_stream=AsyncMock()) if streaming else TurnCallbacks(),
            ), _options=_TurnExecutionOptions(
                inbound=msg, admission_scope=partial(loop._bus_turn_scope, msg),
            ))

        transport.assert_not_awaited()
        assert failed.stop_reason == "error"
        assert failed.content is not None and "context budget" in failed.content
        assert "Stale or foreign source request ticket" not in str(failed.error)
        outbounds = [loop.bus.outbound.get_nowait() for _ in range(loop.bus.outbound_size)]
        ended = next(out.event for out in outbounds if isinstance(out.event, TurnEndEvent))
        assert ended.failure_kind == "model"
        assert ended.failure_error_kind == "context_length"
        assert "context budget" in ended.failure_message
        assert ended.failure_attempts is None
        assert "private-payload" not in str(outbounds)
        assert "private-payload" not in str(backend.source.load_source_snapshot(key).events)

        provider._extra_body = None
        resumed = await loop.execute_turn(TurnRequest(content="Try a small request", session_key=key))
        assert resumed.error is None, resumed.error
        assert resumed.content == "Recovered."
        transport.assert_awaited_once()
    finally:
        await loop.aclose()


async def test_early_context_guard_is_reported_as_context_failure(engine_config):
    provider = OpenAICompatProvider(api_key="fixture", provider_name="budget-fixture")
    transport = AsyncMock()
    provider._ensure_client = AsyncMock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=transport)),
    ))
    loop = AgentLoop.from_config(
        engine_config, tool_registry=ToolRegistry(), provider=provider, context_window_tokens=32,
    )
    key = "websocket:early-context-budget"
    upgrade(loop, key)
    coordinator = WebuiTurnCoordinator(
        bus=loop.bus, sessions=loop.sessions, schedule_background=lambda coro: coro.close(),
    )
    try:
        msg = InboundMessage(
            channel="websocket", sender_id="user", chat_id="early-context-budget",
            content="Too much context", metadata={"webui": True},
        )
        with coordinator.connected():
            failed = await loop.execute_turn(
                TurnRequest(content="Too much context", session_key=key),
                _options=_TurnExecutionOptions(
                    inbound=msg, admission_scope=partial(loop._bus_turn_scope, msg),
                ),
            )
        assert failed.error is not None and "context budget" in failed.error
        transport.assert_not_awaited()
        outbounds = [loop.bus.outbound.get_nowait() for _ in range(loop.bus.outbound_size)]
        ended = next(out.event for out in outbounds if isinstance(out.event, TurnEndEvent))
        assert ended.failure_kind == "model"
        assert ended.failure_error_kind == "context_length"
        assert "context budget" in ended.failure_message
        assert not any(out.content == "Sorry, I encountered an error." for out in outbounds)
    finally:
        await loop.aclose()
