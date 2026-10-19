"""Offline real-engine evidence for the dormant silent scheduled turn."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nanobot.agent.loop import AgentLoop, _TurnExecutionOptions
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import (
    ApprovalGrant,
    DeliveryTarget,
    HistoryMode,
    ScheduledTurnLink,
    TurnCallbacks,
    TurnRequest,
    TurnSource,
)
from nanobot.bus.events import InboundMessage
from nanobot.bus.outbound_events import StreamDeltaEvent
from nanobot.providers.base import LLMResponse, ToolCallRequest
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_migration_engine import ScriptedProvider

engine_config = engine_fixtures.engine_config


def _request(**changes):
    values = {
        "content": "compute report",
        "source": TurnSource.CRON,
        "session_key": "cron:job-1",
        "route": DeliveryTarget(channel="telegram", chat_id="primary"),
        "run_id": "cron-occurrence-fixture-1",
        "history_mode": HistoryMode.FRESH,
        "scheduled_link": ScheduledTurnLink(
            job_id="job-1", instruction="compute report",
            visible_session_key="telegram:primary", occurrence_id="fixture",
        ),
    }
    return TurnRequest(**(values | changes))


async def test_silent_delivery_rejects_channel_stream_events(engine_config) -> None:
    loop = AgentLoop.from_config(
        engine_config, tool_registry=ToolRegistry(), provider=ScriptedProvider([]),
    )
    try:
        loop.bus.publish_event = AsyncMock(side_effect=AssertionError("stream routed"))
        delivery = loop.turn_delivery_factory.silent(
            InboundMessage(channel="telegram", sender_id="cron", chat_id="primary", content="report"),
            "cron:job-1",
        )
        assert not delivery.events.accepts(StreamDeltaEvent)
        assert delivery.events.publish is None
        loop.bus.publish_event.assert_not_awaited()
    finally:
        await loop.aclose()


async def test_real_fresh_turn_retains_detail_without_any_channel_publication(engine_config) -> None:
    provider = ScriptedProvider([LLMResponse(content="new report")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        detail = loop.sessions.get_or_create("cron:job-1")
        detail.add_run_message("old-run", "user", "old private instruction")
        detail.add_run_message("old-run", "assistant", "old private result")
        loop.sessions.save(detail)
        loop.bus.publish_outbound = AsyncMock(side_effect=AssertionError("ordinary output"))
        loop.bus.publish_event = AsyncMock(side_effect=AssertionError("routed event"))

        result = await loop.execute_silent_cron_turn(_request())

        assert result.run_id == "cron-occurrence-fixture-1"
        assert result.content == "new report"
        assert result.sent_messages == ()
        assert loop.run_store.load(result.run_id).scheduled_link is not None
        assert "old private instruction" not in str(provider.calls[0][0])
        assert "old private result" not in str(provider.calls[0][0])
        assert "message" not in {tool["function"]["name"] for tool in provider.calls[0][1]}
        saved = loop.sessions.get_or_create("cron:job-1")
        assert "_compaction_route" not in saved.metadata
        assert [msg["content"] for msg in saved.get_run_messages("old-run")] == [
            "old private instruction", "old private result",
        ]
        assert [msg["content"] for msg in saved.get_run_messages(result.run_id)] == [
            "compute report", "new report",
        ]
        loop.bus.publish_outbound.assert_not_awaited()
        loop.bus.publish_event.assert_not_awaited()
    finally:
        await loop.aclose()


@pytest.mark.parametrize("changes", [
    {"source": TurnSource.DIRECT},
    {"history_mode": HistoryMode.SESSION},
    {"session_key": "telegram:primary"},
    {"run_id": None},
    {"run_id": "invalid run id"},
    {"scheduled_link": None},
    {"scheduled_link": ScheduledTurnLink(
        job_id="other-job", instruction="compute report",
        visible_session_key="telegram:primary", occurrence_id="fixture",
    )},
    {"approval_grant": ApprovalGrant(approved=True, source="cron")},
    {"callbacks": TurnCallbacks(on_progress=AsyncMock())},
    {"tool_names": ("message",)},
])
async def test_invalid_silent_admission_rejected_before_model(engine_config, changes) -> None:
    provider = ScriptedProvider([])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        with pytest.raises(ValueError):
            await loop.execute_silent_cron_turn(_request(**changes))
        assert provider.calls == []
    finally:
        await loop.aclose()


async def test_scheduled_instruction_cannot_consume_pending_human_approval(engine_config) -> None:
    provider = ScriptedProvider([])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        pending = await loop.approval_admission.store(
            session_key="cron:job-1", channel="telegram", chat_id="primary", summary="risky",
        )
        with pytest.raises(ValueError, match="pending approval"):
            await loop.execute_silent_cron_turn(_request(
                content="yes",
                scheduled_link=ScheduledTurnLink(
                    job_id="job-1", instruction="yes",
                    visible_session_key="telegram:primary", occurrence_id="fixture",
                ),
            ))
        assert (await loop.approval_admission.snapshot("cron:job-1", "yes")).pending is pending
        assert provider.calls == []
    finally:
        await loop.aclose()


async def test_internal_silent_option_refuses_unfiltered_message_tool(engine_config) -> None:
    provider = ScriptedProvider([])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        with pytest.raises(ValueError, match="non-publishing"):
            await loop.execute_turn(_request(), _options=_TurnExecutionOptions(silent_output=True))
        assert provider.calls == []
    finally:
        await loop.aclose()


async def test_provider_cannot_call_message_tool_outside_silent_allowlist(engine_config) -> None:
    provider = ScriptedProvider([LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id="send-1", name="message", arguments={"content": "bypass"}),
    ])])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        loop.bus.publish_outbound = AsyncMock(side_effect=AssertionError("message delivered"))
        result = await loop.execute_silent_cron_turn(_request())
        assert result.status.value != "completed"
        assert result.sent_messages == ()
        loop.bus.publish_outbound.assert_not_awaited()
    finally:
        await loop.aclose()


async def test_silent_risky_action_requires_approval_without_persisting_request(
    engine_config,
) -> None:
    provider = ScriptedProvider([LLMResponse(content=None, tool_calls=[
        ToolCallRequest(id="danger", name="exec", arguments={"command": "rm -rf ."}),
    ])])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    try:
        result = await loop.execute_silent_cron_turn(_request())
        assert result.status.value == "approval_required"
        assert result.tools_used == []
        assert len(provider.calls) == 1
        snapshot = await loop.approval_admission.snapshot("cron:job-1", "yes")
        assert snapshot.pending is None
    finally:
        await loop.aclose()
