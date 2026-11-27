"""The child record keeps the tool call that spawned it, for inline display anchoring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kohaku.agent.child_sessions import (
    ChildAdmission,
    DelegationTree,
    _parse_child_admission,
)
from kohaku.agent.loop import AgentLoop
from kohaku.agent.tools.context import (
    RequestContext,
    bind_request_context,
    reset_request_context,
)
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.agent.turn import DeliveryTarget, TurnCallbacks, TurnRequest
from kohaku.providers.base import LLMResponse, ToolCallRequest
from kohaku.session.manager import SessionManager
from kohaku.webui.outbound_wire import encode_delegated_child
from kohaku.webui.ws_http import _hydrated_child_payload
from tests.agent.test_delegated_child_events import _make_admission
from tests.agent.test_delegation_stale_parent import _source_parent
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade

pytest_plugins = ("tests.agent.test_migration_source_engine",)


@pytest.mark.parametrize("spawn_call_id", ["call-spawn", None])
async def test_reserve_persists_optional_spawn_call_id(tmp_path, monkeypatch, spawn_call_id):
    monkeypatch.setattr("kohaku.config.paths.get_data_dir", lambda: tmp_path)
    manager = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    parent = _source_parent(manager, "websocket:chat-a")
    tree = DelegationTree(manager)

    admission = await tree.reserve(
        parent=parent,
        parent_key="websocket:chat-a",
        assignment="inspect the fixture",
        attempt_id="attempt-1",
        spawn_call_id=spawn_call_id,
    )
    assert isinstance(admission, ChildAdmission)
    assert admission.spawn_call_id == spawn_call_id

    saved = manager.read_session_metadata(admission.session_key)["metadata"]["delegation_child"]
    assert saved.get("spawn_call_id") == spawn_call_id
    assert _parse_child_admission(admission.session_key, saved).spawn_call_id == spawn_call_id
    # Records written before the field existed read as None.
    legacy = {key: value for key, value in saved.items() if key != "spawn_call_id"}
    assert _parse_child_admission(admission.session_key, legacy).spawn_call_id is None


@pytest.mark.parametrize(
    ("tool", "arguments", "streaming", "expected"),
    [
        # delegate_task has a durable source intent binding, streaming or not.
        (
            "delegate_task",
            {"contract": "Implement the fixture and run checks.", "write_scope": ["out.txt"]},
            False,
            "spawn-call",
        ),
        # Read-only role tools only have the live progress sink.
        ("explore", {"task": "Read the fixture source."}, True, "spawn-call"),
        ("explore", {"task": "Read the fixture source."}, False, None),
    ],
)
async def test_role_tools_record_the_spawning_call(
    engine_config, tool, arguments, streaming, expected,
):
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="spawn-call", name=tool, arguments=arguments,
        )]),
        LLMResponse(content="child report"),
        LLMResponse(content="parent completed"),
    ])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    parent_key = "websocket:parent"
    upgrade(loop, parent_key)
    callbacks = (
        TurnCallbacks(on_progress=AsyncMock(), on_stream=AsyncMock(), on_stream_end=AsyncMock())
        if streaming
        else TurnCallbacks()
    )
    try:
        result = await loop.execute_turn(TurnRequest(
            content="Work on the fixture.",
            session_key=parent_key,
            route=DeliveryTarget(channel="websocket", chat_id="parent"),
            callbacks=callbacks,
        ))
        assert result.error is None, result.error
        children = [
            loop.sessions.read_session_metadata(row["key"])["metadata"]["delegation_child"]
            for row in loop.sessions.list_sessions()
            if row["key"].startswith("delegated:")
        ]
        [child] = children
        assert child.get("spawn_call_id") == expected
    finally:
        await loop.aclose()


async def test_child_event_wire_and_hydration_carry_spawn_call_id():
    admission = _make_admission()
    admission = type(admission)(**{
        **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
        "spawn_call_id": "call-spawn",
    })
    loop = AgentLoop.__new__(AgentLoop)
    loop.bus = SimpleNamespace(publish_outbound=AsyncMock())
    token = bind_request_context(RequestContext(
        channel="websocket", chat_id="chat-a", session_key="websocket:chat-a",
    ))
    try:
        await loop._publish_delegated_child(admission)
    finally:
        reset_request_context(token)

    event = loop.bus.publish_outbound.await_args.args[0].event
    assert event.spawn_call_id == "call-spawn"
    assert encode_delegated_child("chat-a", event)["spawn_call_id"] == "call-spawn"

    def hydrate(payload):
        return _hydrated_child_payload(
            admission.session_key,
            payload,
            parent_key=admission.parent_key,
            root_key=admission.root_key,
            parent_source=None,
            root_source=None,
            child_source=None,
            runtime_executable=False,
        )

    assert hydrate(admission.metadata())["spawn_call_id"] == "call-spawn"
    legacy = {k: v for k, v in admission.metadata().items() if k != "spawn_call_id"}
    assert hydrate(legacy)["spawn_call_id"] is None
