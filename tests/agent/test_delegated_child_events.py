"""Regression coverage for A3-5 AC5.4 typed child events."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.child_sessions import ChildAdmission
from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.tools.context import (
    RequestContext,
    bind_request_context,
    reset_request_context,
)
from nanobot.agent.turn import TurnContext
from nanobot.bus.outbound_events import DelegatedChildEvent
from nanobot.session.manager import SessionManager


class _Provider:
    """Minimal provider surface required to construct the manager."""

    provider_name = "test"
    default_model = "test-model"
    model = "test-model"
    name = "test"

    def get_default_model(self) -> str:
        return self.default_model


def _make_admission() -> ChildAdmission:
    return ChildAdmission(
        child_id="child-a",
        session_key="delegated:child-a",
        parent_key="websocket:parent",
        parent_incarnation="inc-1",
        root_key="websocket:parent",
        profile="delegated",
        depth=1,
        attempt_id="turn-1",
        assignment_digest="digest",
        policy_facts={},
        runtime_generation=0,
    )


@pytest.mark.asyncio
async def test_child_admission_emits_active_and_completed_events(tmp_path):
    manager = ForegroundAgentManager(
        provider=_Provider(),
        workspace=tmp_path,
        sessions=SessionManager(
            tmp_path / "workspace",
            sessions_root=tmp_path / "sessions",
        ),
    )
    manager.runner.run = AsyncMock(
        return_value=AgentRunResult(final_content="done", messages=[])
    )
    sink = AsyncMock()
    manager.set_child_event_sink(sink)

    context = TurnContext(session_key="websocket:parent", run_id="turn-1")
    await manager.run_plan(objective="fixture plan", turn_context=context)

    assert sink.await_count >= 2
    admissions = [call.args[0] for call in sink.await_args_list]
    assert admissions[0].state == "active"
    assert admissions[-1].state == "completed"
    for admission in admissions:
        assert admission.session_key.startswith("delegated:")
        assert admission.profile == "delegated"


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
