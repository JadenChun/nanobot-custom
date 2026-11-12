"""Regression coverage for A3-5 AC5.4 typed child events."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.child_sessions import ChildAdmission
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.context import (
    RequestContext,
    bind_request_context,
    reset_request_context,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import DeliveryTarget, TurnRequest
from nanobot.bus.outbound_events import DelegatedChildEvent
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.source_store import SourceState
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade

pytest_plugins = ("tests.agent.test_migration_source_engine",)


def _state(key: str, incarnation: str, generation: int) -> SourceState:
    return SourceState(
        store_identity="store-a",
        session_key=key,
        incarnation=incarnation,
        revision=3,
        generation=generation,
        checkpoint_id="checkpoint-a",
        checkpoint_fingerprint="fingerprint-a",
    )


def _make_admission() -> ChildAdmission:
    return ChildAdmission(
        child_id="child-a",
        session_key="delegated:child-a",
        parent_key="websocket:parent",
        parent_incarnation="inc-1",
        root_key="websocket:parent",
        profile="delegated",
        depth=1,
        attempt_id="parent-run-a",
        assignment_digest="digest",
        policy_facts={},
        runtime_generation=0,
        source_state=_state("delegated:child-a", "inc-child", 2),
        parent_source_state=_state("websocket:parent", "inc-1", 1),
        root_source_state=_state("websocket:parent", "inc-1", 1),
        run_sequence=2,
        state_sequence=4,
        current_run_id="child-run-a",
        holds_capacity=False,
        pending_approval_id="approval-a",
        blocked_run_id="child-run-a",
        state="completed",
        outcome="completed",
    )


@pytest.mark.asyncio
async def test_child_admission_emits_active_and_completed_events(engine_config):
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="plan-child",
            name="plan_task",
            arguments={"objective": "fixture plan"},
        )]),
        LLMResponse(content="done"),
        LLMResponse(content="parent completed"),
    ])
    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=provider,
    )
    parent_key = "websocket:parent"
    upgrade(loop, parent_key)
    manager = loop._fork_dependencies.foreground_manager
    sink = AsyncMock()
    manager.set_child_event_sink(sink)
    try:
        result = await loop.execute_turn(TurnRequest(
            content="Plan the fixture work.",
            session_key=parent_key,
            route=DeliveryTarget(channel="websocket", chat_id="parent"),
        ))

        assert result.error is None, result.error
        assert sink.await_count >= 2
        admissions = [call.args[0] for call in sink.await_args_list]
        assert admissions[0].state == "active"
        assert admissions[-1].state == "completed"
        for admission in admissions:
            assert admission.session_key.startswith("delegated:")
            assert admission.profile == "delegated"
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_parent_publisher_projects_child_reference():
    admission = _make_admission()
    loop = AgentLoop.__new__(AgentLoop)
    loop.bus = SimpleNamespace(publish_outbound=AsyncMock())

    request_context = RequestContext(
        channel="websocket",
        chat_id="chat-a",
        session_key="websocket:chat-a",
        metadata={"webui_turn_id": "turn-a"},
    )
    token = bind_request_context(request_context)
    try:
        await loop._publish_delegated_child(admission)
    finally:
        reset_request_context(token)

    loop.bus.publish_outbound.assert_awaited_once()
    outbound = loop.bus.publish_outbound.await_args.args[0]
    assert outbound.channel == "websocket"
    assert outbound.chat_id == "chat-a"
    assert isinstance(outbound.event, DelegatedChildEvent)

    event = outbound.event
    assert event.parent_session_key == admission.parent_key
    assert event.child_session_key == admission.session_key
    assert event.child_id == admission.child_id
    assert event.profile == admission.profile
    assert event.state == admission.state
    assert event.outcome == admission.outcome
    assert event.runtime_generation == admission.runtime_generation
    assert event.turn_id == "turn-a"
    assert event.parent_run_id == "parent-run-a"
    assert event.child_run_id == "child-run-a"
    assert event.parent_store_identity == "store-a"
    assert event.parent_incarnation == "inc-1"
    assert event.parent_generation == 1
    assert event.root_key == "websocket:parent"
    assert event.root_incarnation == "inc-1"
    assert event.child_incarnation == "inc-child"
    assert event.child_generation == 2
    assert event.run_sequence == 2
    assert event.state_sequence == 4
    assert event.control_epoch == 0
    assert event.holds_capacity is False
    assert event.can_stop is False
    assert event.owned_work is False
    assert event.pending_approval_id == "approval-a"
    assert event.blocked_run_id == "child-run-a"
