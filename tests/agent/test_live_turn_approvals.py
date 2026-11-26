"""Verify decisions and cancellation on a live native approval."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from kohaku.agent.turn import ApprovalDecision
from kohaku.bus.events import InboundMessage
from kohaku.bus.outbound_events import ExternalApprovalEvent
from kohaku.bus.runtime_events import TurnCompleted
from kohaku.providers.base import LLMResponse
from tests.agent.test_exact_effect_approval import CountExec, risky_call
from tests.migration.test_local_runtime_activation import (
    _config,
    _running_composition,
    _ScriptedProvider,
)


@pytest.mark.parametrize(("action", "effects"), [
    ("approve", 1), ("deny", 0), ("stop", 0),
])
async def test_live_approval_decision_or_stop_stays_in_the_original_turn(
    tmp_path, monkeypatch, action, effects,
):
    config = _config(tmp_path)
    config.tools.permissions.profile = "balanced"
    provider = _ScriptedProvider([
        LLMResponse(content=None, tool_calls=[risky_call("prompted")]),
        LLMResponse(content="The original turn continued."),
    ])
    async with _running_composition(config, provider, monkeypatch) as (handle, _):
        chat_id = "live-native-approval"
        key = f"websocket:{chat_id}"
        turn_id = "original-turn"
        counter = CountExec()
        handle.tools.replace(counter)
        completions: list[TurnCompleted] = []
        handle.bus.subscribe(completions.append, TurnCompleted)
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket",
            sender_id="webui",
            chat_id=chat_id,
            content="Perform the fixture effect.",
            metadata={
                "webui": True,
                "webui_turn_id": turn_id,
                "live_turn_controls": True,
            },
        ))
        async with asyncio.timeout(5):
            while True:
                outbound = await handle.bus.consume_outbound()
                if isinstance(outbound.event, ExternalApprovalEvent) and outbound.event.approval:
                    approval = outbound.event.approval
                    break
        assert counter.count == 0 and completions == []
        exact = ApprovalDecision(
            approval_id=approval.approval_id,
            decision=action if action != "stop" else "approve",
            session_key=key,
            session_incarnation=approval.session_incarnation,
            run_id=approval.run_id,
            effect_id=approval.tool_call_id,
            generation=approval.runtime_generation,
        )
        assert not handle.agent.resolve_run_approval(key, replace(exact, effect_id="wrong"))
        if action == "stop":
            from tests.agent.test_original_chat_stop_target import stop_direct

            target = handle.agent.stop_target_for_key(key)
            assert target is not None
            await stop_direct(handle.agent, target, chat_id=chat_id)
        else:
            assert handle.agent.resolve_run_approval(key, exact)
            assert not handle.agent.resolve_run_approval(key, exact)
        async with asyncio.timeout(5):
            while not completions:
                await asyncio.sleep(0)
        assert counter.count == effects
        assert len(completions) == 1
        assert completions[0].context.metadata["webui_turn_id"] == turn_id
        assert completions[0].outcome == ("cancelled" if action == "stop" else "completed")
        backend = handle.sessions.source_backend(key)
        assert backend is not None
        events = backend.source.load_source_snapshot(key).events
        user_messages = [event["content"] for event in events if event.get("role") == "user"]
        assert user_messages == ["Perform the fixture effect."]
        assert not any("I'm about to take a risky action" in str(event.get("content")) for event in events)


async def test_live_folder_access_continues_the_original_turn(tmp_path, monkeypatch):
    from kohaku.bus.outbound_events import SessionUpdatedEvent
    from kohaku.channels.websocket.runtime import WebSocketConfig
    from kohaku.providers.base import ToolCallRequest
    from kohaku.webui.gateway_services import build_gateway_services

    config = _config(tmp_path)
    provider = _ScriptedProvider([
        LLMResponse(content=None, tool_calls=[
            ToolCallRequest(id="folder", name="request_folder_access", arguments={"purpose": "Read the chosen folder"}),
        ]),
        LLMResponse(content="The folder is available to the original turn."),
    ])
    async with _running_composition(config, provider, monkeypatch) as (handle, _):
        chat_id = "live-folder-approval"
        key = f"websocket:{chat_id}"
        turn_id = "folder-turn"
        gateway = build_gateway_services(
            config=WebSocketConfig(enabled=True),
            bus=handle.bus,
            session_manager=handle.sessions,
            workspace_path=config.workspace_path,
            default_restrict_to_workspace=True,
            config_path=tmp_path / "runtime" / "config.json",
            runtime_surface="browser",
            runtime_model_name=None,
            runtime_capabilities_overrides=None,
        )
        recovery = gateway.http.workspace_access_recovery
        assert recovery is not None
        completions: list[TurnCompleted] = []
        handle.bus.subscribe(completions.append, TurnCompleted)
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket", sender_id="webui", chat_id=chat_id,
            content="Use a folder I will select.",
            metadata={"webui": True, "webui_turn_id": turn_id, "live_turn_controls": True},
        ))
        async with asyncio.timeout(5):
            while True:
                outbound = await handle.bus.consume_outbound()
                if isinstance(outbound.event, SessionUpdatedEvent) and outbound.event.workspace_access_request:
                    request = outbound.event.workspace_access_request
                    break
        assert not completions
        assert request["blocked_turn_id"] == turn_id
        ready = recovery.resolve({
            "chat_id": chat_id,
            "access_request_id": request["access_request_id"],
            "expected_revision": request["revision"],
            "expected_grants_revision": gateway.workspaces.grant_snapshot().revision,
            "path": str(config.workspace_path),
        })["workspace_access_request"]
        result = await recovery.continue_request({
            "chat_id": chat_id,
            "access_request_id": ready["access_request_id"],
            "expected_revision": ready["revision"],
        })
        assert result["workspace_access_request"]["continuation_turn_id"] == turn_id
        async with asyncio.timeout(5):
            while not completions:
                await asyncio.sleep(0)
        assert len(completions) == 1
        assert completions[0].context.metadata["webui_turn_id"] == turn_id
        assert completions[0].outcome == "completed"
        backend = handle.sessions.source_backend(key)
        assert backend is not None
        events = backend.source.load_source_snapshot(key).events
        assert [e["content"] for e in events if e.get("role") == "user"] == ["Use a folder I will select."]
        assert not any("Folder access is needed to continue this task" in str(e.get("content")) for e in events)


async def test_live_folder_access_updates_active_run_scope_for_subsequent_steer(tmp_path, monkeypatch):
    from kohaku.bus.outbound_events import SessionUpdatedEvent
    from kohaku.channels.websocket.runtime import WebSocketConfig
    from kohaku.providers.base import ToolCallRequest
    from kohaku.security.workspace_access import WORKSPACE_SCOPE_METADATA_KEY
    from kohaku.webui.gateway_services import build_gateway_services

    config = _config(tmp_path)
    chosen = config.workspace_path / "chosen-subdir"
    chosen.mkdir(parents=True, exist_ok=True)
    second_started = asyncio.Event()
    release_second = asyncio.Event()
    provider = _ScriptedProvider([
        LLMResponse(content=None, tool_calls=[
            ToolCallRequest(id="folder", name="request_folder_access", arguments={"purpose": "Inspect subdir"}),
        ]),
        LLMResponse(content="Opened subdir."),
        LLMResponse(content="Done after steer."),
    ])
    original_chat = provider.chat

    async def gated_chat(*args, **kwargs):
        if len(provider.requests) == 1:
            second_started.set()
            await release_second.wait()
        return await original_chat(*args, **kwargs)

    monkeypatch.setattr(provider, "chat", gated_chat)
    async with _running_composition(config, provider, monkeypatch) as (handle, _):
        chat_id = "live-folder-steer-scope"
        key = f"websocket:{chat_id}"
        turn_id = "folder-steer-turn"
        gateway = build_gateway_services(
            config=WebSocketConfig(enabled=True),
            bus=handle.bus,
            session_manager=handle.sessions,
            workspace_path=config.workspace_path,
            default_restrict_to_workspace=True,
            config_path=tmp_path / "runtime" / "config.json",
            runtime_surface="browser",
            runtime_model_name=None,
            runtime_capabilities_overrides=None,
        )
        recovery = gateway.http.workspace_access_recovery
        assert recovery is not None
        completions: list[TurnCompleted] = []
        handle.bus.subscribe(completions.append, TurnCompleted)
        initial_scope = {
            "project_path": str(config.workspace_path.resolve()),
            "access_mode": "read_write",
        }
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket", sender_id="webui", chat_id=chat_id,
            content="Open the chosen subdir.",
            metadata={
                "webui": True,
                "webui_turn_id": turn_id,
                "live_turn_controls": True,
                WORKSPACE_SCOPE_METADATA_KEY: initial_scope,
            },
        ))
        async with asyncio.timeout(5):
            while True:
                outbound = await handle.bus.consume_outbound()
                if isinstance(outbound.event, SessionUpdatedEvent) and outbound.event.workspace_access_request:
                    request = outbound.event.workspace_access_request
                    break
        ready = recovery.resolve({
            "chat_id": chat_id,
            "access_request_id": request["access_request_id"],
            "expected_revision": request["revision"],
            "expected_grants_revision": gateway.workspaces.grant_snapshot().revision,
            "path": str(chosen),
        })["workspace_access_request"]
        await recovery.continue_request({
            "chat_id": chat_id,
            "access_request_id": ready["access_request_id"],
            "expected_revision": ready["revision"],
        })
        await asyncio.wait_for(second_started.wait(), timeout=5)
        try:
            active_run = handle.agent.run_control.active_run(key)
            assert active_run is not None
            assert active_run.workspace_scope is not None
            assert active_run.workspace_scope["project_path"] == str(chosen.resolve())
            updated_scope = dict(active_run.workspace_scope)
            steer_msg = InboundMessage(
                channel="websocket",
                sender_id="webui",
                chat_id=chat_id,
                content="Also summarize the files.",
                metadata={
                    "webui": True,
                    WORKSPACE_SCOPE_METADATA_KEY: updated_scope,
                },
            )
            admitted = await handle.agent.run_control.enqueue_steer(
                key, steer_msg, turn_id=turn_id,
            )
            assert admitted is True
        finally:
            release_second.set()
        async with asyncio.timeout(5):
            while not completions:
                await asyncio.sleep(0)
        assert completions[0].outcome == "completed"


async def test_live_folder_access_continue_tolerates_concurrent_stop_cancellation(tmp_path, monkeypatch):
    from kohaku.bus.outbound_events import SessionUpdatedEvent
    from kohaku.channels.websocket.runtime import WebSocketConfig
    from kohaku.providers.base import ToolCallRequest
    from kohaku.webui.gateway_services import build_gateway_services

    config = _config(tmp_path)
    provider = _ScriptedProvider([
        LLMResponse(content=None, tool_calls=[
            ToolCallRequest(id="folder", name="request_folder_access", arguments={"purpose": "Read folder"}),
        ]),
        LLMResponse(content="Should not run after cancellation."),
    ])
    async with _running_composition(config, provider, monkeypatch) as (handle, _):
        chat_id = "live-folder-concurrent-stop"
        key = f"websocket:{chat_id}"
        turn_id = "folder-race-turn"
        gateway = build_gateway_services(
            config=WebSocketConfig(enabled=True),
            bus=handle.bus,
            session_manager=handle.sessions,
            workspace_path=config.workspace_path,
            default_restrict_to_workspace=True,
            config_path=tmp_path / "runtime" / "config.json",
            runtime_surface="browser",
            runtime_model_name=None,
            runtime_capabilities_overrides=None,
        )
        recovery = gateway.http.workspace_access_recovery
        assert recovery is not None
        completions: list[TurnCompleted] = []
        handle.bus.subscribe(completions.append, TurnCompleted)
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket", sender_id="webui", chat_id=chat_id,
            content="Pick a folder.",
            metadata={"webui": True, "webui_turn_id": turn_id, "live_turn_controls": True},
        ))
        async with asyncio.timeout(5):
            while True:
                outbound = await handle.bus.consume_outbound()
                if isinstance(outbound.event, SessionUpdatedEvent) and outbound.event.workspace_access_request:
                    request = outbound.event.workspace_access_request
                    break
        ready = recovery.resolve({
            "chat_id": chat_id,
            "access_request_id": request["access_request_id"],
            "expected_revision": request["revision"],
            "expected_grants_revision": gateway.workspaces.grant_snapshot().revision,
            "path": str(config.workspace_path),
        })["workspace_access_request"]
        live_waiter = handle.sessions.live_folder_access[key]
        original_commit = recovery._commit_action

        def cancel_during_commit(*args, **kwargs):
            result = original_commit(*args, **kwargs)
            live_waiter.decision.cancel()
            return result

        monkeypatch.setattr(recovery, "_commit_action", cancel_during_commit)
        result = await recovery.continue_request({
            "chat_id": chat_id,
            "access_request_id": ready["access_request_id"],
            "expected_revision": ready["revision"],
        })
        assert result["workspace_access_request"]["continuation_turn_id"] == turn_id
        async with asyncio.timeout(5):
            while not completions:
                await asyncio.sleep(0)
        assert completions[0].outcome == "cancelled"
