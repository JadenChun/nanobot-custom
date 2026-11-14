"""Typed original-chat decisions cross real ingress, bus and child admission."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.outbound_events import WebUIAdmissionRejectedEvent
from nanobot.bus.runtime_events import TurnCompleted, TurnRuntimeAdmitted
from nanobot.channels.websocket.runtime import WebSocketChannel, WebSocketConfig, _parse_envelope
from nanobot.channels.websocket.tests.test_websocket_channel import (
    _authenticate_local_webui_connection,
)
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session import webui_turns as turns
from nanobot.session.persistence_metadata import SOURCE_PENDING_FOLLOWUPS_METADATA_KEY
from nanobot.webui.gateway_services import build_gateway_services
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade

pytest_plugins = ("tests.agent.test_migration_source_engine",)
KEY = "websocket:approval-siblings"
CHAT = "approval-siblings"


class SiblingProvider(ScriptedProvider):
    def __init__(self):
        super().__init__([])
        self.parent_started = asyncio.Event()
        self.parent_release = asyncio.Event()
        self.alpha_held = asyncio.Event()
        self.alpha_release = asyncio.Event()
        self.parent_calls = 0
        self.child_calls = {"alpha.txt": 0, "beta.txt": 0}

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append((deepcopy(messages), tools, model))
        if not any("Worker Role" in str(row.get("content")) for row in messages):
            self.parent_calls += 1
            if self.parent_calls == 1:
                return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                    id=f"delegate-{name}", name="delegate_task",
                    arguments={"contract": f"Replace {name} with the approved content.",
                               "write_scope": [name], "background": True},
                ) for name in self.child_calls])
            self.parent_started.set()
            await self.parent_release.wait()
            return LLMResponse(content="parent finished")
        text = str(messages)
        name = "alpha.txt" if "Replace alpha.txt" in text else "beta.txt"
        self.child_calls[name] += 1
        count = self.child_calls[name]
        if count <= 2:
            return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id=f"write-{name}-{count}", name="write_file",
                arguments={"path": name, "content": f"approved {name}\n"},
            )])
        if name == "alpha.txt":
            self.alpha_held.set()
            await self.alpha_release.wait()
        return LLMResponse(content=f"finished {name}")


def frames(connection):
    return [json.loads(call.args[0]) for call in connection.send.await_args_list]


def decision_envelope(pending, decision="approve", *, turn_id="decide-alpha"):
    return {"type": "message", "chat_id": CHAT, "content": "approval_decision",
            "webui": True, "turn_id": turn_id, "request_settings_version": 1,
            "request_intent": "new", "approval_decision": {
                "approval_id": pending.approval_id, "decision": decision,
                "session_key": pending.session_key,
                "session_incarnation": pending.session_incarnation,
                "run_id": pending.run_id,
                "generation": pending.generation,
            }}


@asynccontextmanager
async def siblings(engine_config, monkeypatch, *, pending_turn_id=None):
    for name in ("_WEBSOCKET_ACTIVE_TURNS", "_WEBSOCKET_TURN_WALL_STARTED_AT", "_WEBSOCKET_TURN_IDS", "_WEBSOCKET_TURN_OWNERS"):
        getattr(turns, name).clear()
    provider = SiblingProvider()
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop, KEY)
    loop.sessions.upsert_session_metadata(KEY, {"title": "Sibling approval fixture"})
    disconnect = turns.WebuiTurnCoordinator(
        bus=loop.bus, sessions=loop.sessions, schedule_background=loop.schedule_background,
    ).subscribe()
    for name in provider.child_calls:
        (engine_config.workspace_path / name).write_text("protected line\n" * 20)
    config = WebSocketConfig.model_validate({"enabled": True, "allowFrom": ["*"],
                                           "websocketRequiresToken": False})
    gateway = build_gateway_services(
        config=config, bus=loop.bus, session_manager=loop.sessions, static_dist_path=None,
        workspace_path=engine_config.workspace_path, default_restrict_to_workspace=True,
        runtime_model_name=None, runtime_surface="browser", runtime_capabilities_overrides=None,
        stop_target_for_key=loop.stop_target_for_key,
    )
    channel = WebSocketChannel(config, loop.bus, gateway=gateway)
    connection = AsyncMock()
    _authenticate_local_webui_connection(channel, connection)
    ready = asyncio.Event()
    original_store = loop.approval_admission.store

    async def store(**kwargs):
        if pending_turn_id is not None:
            kwargs["turn_id"] = pending_turn_id
        result = await original_store(**kwargs)
        if len(loop.approval_admission._pending) == 2:
            ready.set()
        return result

    monkeypatch.setattr(loop.approval_admission, "store", store)
    run = asyncio.create_task(loop.run())
    try:
        await channel._dispatch_envelope(connection, "client", {
            "type": "message", "chat_id": CHAT, "content": "Start both bounded workers.",
            "webui": True, "turn_id": "parent-turn", "request_settings_version": 1,
            "request_intent": "new",
        })
        await asyncio.wait_for(provider.parent_started.wait(), timeout=5)
        await asyncio.wait_for(ready.wait(), timeout=5)
        tree = loop._fork_dependencies.foreground_manager._delegation_tree
        await asyncio.gather(*(tree.wait_child(child.child_id, timeout=5) for child in tree.children_of(KEY)))
        children = sorted(tree.children_of(KEY), key=lambda child: child.task_label)
        assert len(children) == 2 and all(child.state == "waiting_approval" for child in children)
        pending = {"alpha" if "alpha.txt" in child.task_label else "beta":
                   loop.approval_admission._pending[child.pending_approval_id] for child in children}
        yield loop, provider, channel, connection, pending
    finally:
        provider.parent_release.set()
        provider.alpha_release.set()
        loop.stop()
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        await loop.aclose()
        await channel.stop()
        disconnect()
        turns.clear_websocket_turns(CHAT)


async def wait_refusal(loop, *, channel=None):
    while True:
        message = await asyncio.wait_for(loop.bus.consume_outbound(), timeout=5)
        if isinstance(message.event, WebUIAdmissionRejectedEvent):
            return message.event
        if channel is not None:
            await channel.send(message)


async def wait_child_approval(loop, held, *, channel=None):
    reached = asyncio.create_task(held.wait())
    refused = asyncio.create_task(wait_refusal(loop, channel=channel))
    try:
        done, _ = await asyncio.wait(
            {reached, refused}, timeout=5, return_when=asyncio.FIRST_COMPLETED,
        )
        assert reached in done, refused.result() if refused in done else "child did not resume"
    finally:
        reached.cancel()
        refused.cancel()
        await asyncio.gather(reached, refused, return_exceptions=True)


def replace_pending(loop, pending, **changes):
    replacement = replace(pending, **changes)
    for index in [loop.approval_admission._pending, *loop.approval_admission._aliases.values()]:
        if index.get(pending.approval_id) is pending:
            index[pending.approval_id] = replacement
    return replacement


@pytest.mark.parametrize("identities", ["public", "omitted"])
async def test_root_approval_keeps_canonical_effect_path_through_real_ingress(engine_config, monkeypatch, identities):
    turns.clear_websocket_turns(CHAT)
    target = engine_config.workspace_path / "root-approval.txt"
    target.write_text("protected line\n" * 20)
    provider = ScriptedProvider([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id=identifier, name="write_file",
            arguments={"path": target.name, "content": "approved root write\n"},
        )]) for identifier in ("root-blocked", "root-approved")
    ] + [LLMResponse(content="root resumed")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop, KEY)
    loop.sessions.upsert_session_metadata(KEY, {"title": "Root approval fixture"})
    disconnect = turns.WebuiTurnCoordinator(
        bus=loop.bus, sessions=loop.sessions, schedule_background=loop.schedule_background,
    ).subscribe()
    config = WebSocketConfig.model_validate({"enabled": True, "allowFrom": ["*"],
                                           "websocketRequiresToken": False})
    gateway = build_gateway_services(
        config=config, bus=loop.bus, session_manager=loop.sessions, static_dist_path=None,
        workspace_path=engine_config.workspace_path, default_restrict_to_workspace=True,
        runtime_model_name=None, runtime_surface="browser", runtime_capabilities_overrides=None,
        stop_target_for_key=loop.stop_target_for_key,
    )
    channel = WebSocketChannel(config, loop.bus, gateway=gateway)
    connection = AsyncMock()
    _authenticate_local_webui_connection(channel, connection)
    pending_ready = asyncio.Event()
    original_store = loop.approval_admission.store

    async def store(**kwargs):
        pending = await original_store(**kwargs)
        pending_ready.set()
        return pending

    monkeypatch.setattr(loop.approval_admission, "store", store)
    write_tool = loop.tools.get("write_file")
    write = AsyncMock(wraps=write_tool.execute)
    monkeypatch.setattr(write_tool, "execute", write)
    run = asyncio.create_task(loop.run())
    try:
        await channel._dispatch_envelope(connection, "client", {
            "type": "message", "chat_id": CHAT, "content": "Replace the protected root file.",
            "webui": True, "turn_id": "root-turn", "request_settings_version": 1,
            "request_intent": "new",
        })
        await asyncio.wait_for(pending_ready.wait(), timeout=5)
        await asyncio.wait_for(asyncio.gather(*loop._active_root_task_identity(KEY)), timeout=5)
        while loop.bus.outbound_size:
            await channel.send(loop.bus.outbound.get_nowait())
        assert not turns.websocket_turn_reserved(CHAT)
        write.assert_not_awaited()
        # A settled canonical source remains valid after its ephemeral cache
        # entry is discarded; approval ownership lives in the source/admission.
        loop.sessions.invalidate(KEY)
        assert loop.sessions.get_cached(KEY) is None
        pending = next(iter(loop.approval_admission._pending.values()))
        assert pending.session_key == KEY
        resumed = asyncio.Event()
        unsubscribe = loop.bus.subscribe(lambda event: resumed.set(), TurnCompleted)
        try:
            raw = decision_envelope(pending, turn_id="approve-root")
            if identities == "omitted":
                raw["approval_decision"] = {"approval_id": pending.approval_id, "decision": "approve"}
            await channel._dispatch_envelope(
                connection, "client", raw,
            )
            assert not [frame for frame in frames(connection) if frame.get("event") == "error"]
            await asyncio.wait_for(resumed.wait(), timeout=5)
        finally:
            unsubscribe()
        write.assert_awaited_once()
        assert target.read_text() == "approved root write\n"
        assert len(provider.calls) == 3
        assert pending.approval_id not in loop.approval_admission._pending
        accepted = [frame for frame in frames(connection) if frame.get("event") == "message_accepted"]
        assert accepted[-1]["turn_id"] == "approve-root"
        assert accepted[-1]["starts_turn"] is False
        assert not [frame for frame in frames(connection) if frame.get("event") == "error"]
    finally:
        loop.stop()
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        await loop.aclose()
        await channel.stop()
        disconnect()
        turns.clear_websocket_turns(CHAT)


def parent_facts(loop, channel):
    backend = loop.sessions.source_backend(KEY)
    snapshot = backend.source.load_source_snapshot(KEY)
    return {
        "tasks": loop._active_root_task_identity(KEY),
        "bindings": {task: binding for task, binding in loop._active_root_bindings.items()
                     if binding.session_key == KEY},
        "queue": loop._pending_queues.get(KEY),
        "queue_size": loop._pending_queues[KEY].qsize() if KEY in loop._pending_queues else 0,
        "events": deepcopy(snapshot.events),
        "pending": deepcopy(snapshot.header["metadata"].get(SOURCE_PENDING_FOLLOWUPS_METADATA_KEY)),
        "turn": turns.websocket_turn_id(CHAT),
        "owner": turns._WEBSOCKET_TURN_OWNERS.get(CHAT),
        "scope": channel.gateway.workspaces.scope_for_session_key(KEY),
        "permissions": {identity: reference for identity, reference in
                        loop.sessions.desktop_permissions._tasks.items() if identity.session_key == KEY},
        "retirements": {identity: deepcopy(retirement) for identity, retirement in
                        loop.sessions.desktop_permissions._task_retirements.items() if identity.session_key == KEY},
    }


async def test_streaming_sibling_controls_do_not_touch_parent_and_stop_remains_responsive(engine_config, monkeypatch):
    async with siblings(engine_config, monkeypatch) as (loop, provider, channel, connection, pending):
        facts = parent_facts(loop, channel)
        parent_inputs = deepcopy([messages for messages, _, _ in provider.calls
                                  if not any("Worker Role" in str(row.get("content")) for row in messages)])
        lifecycle = []
        loop.bus.subscribe(lifecycle.append, TurnRuntimeAdmitted)
        loop.bus.subscribe(lifecycle.append, TurnCompleted)
        retire = Mock(wraps=loop._retire_workspace_task_before_input)
        invalidate = Mock(wraps=loop.sessions.invalidate_workspace_access_request)
        transcript = Mock(wraps=channel.gateway.transcripts.append_user_message)
        defaults = Mock(wraps=channel.gateway.workspaces.remember_request_defaults)
        persist_scope = Mock(wraps=channel.gateway.workspaces.persist_scope)
        attachments = Mock(wraps=channel.gateway.media.attachment_ingress.accept_batch)
        manager = loop._fork_dependencies.foreground_manager
        alpha = manager._delegation_tree.child_for_session(pending["alpha"].session_key)
        write_tool = manager._suspended_children[alpha.child_id].request.tools.get("write_file")
        write = AsyncMock(wraps=write_tool.execute)
        monkeypatch.setattr(loop, "_retire_workspace_task_before_input", retire)
        monkeypatch.setattr(loop.sessions, "invalidate_workspace_access_request", invalidate)
        monkeypatch.setattr(channel.gateway.transcripts, "append_user_message", transcript)
        monkeypatch.setattr(channel.gateway.workspaces, "remember_request_defaults", defaults)
        monkeypatch.setattr(channel.gateway.workspaces, "persist_scope", persist_scope)
        monkeypatch.setattr(channel.gateway.media.attachment_ingress, "accept_batch", attachments)
        monkeypatch.setattr(write_tool, "execute", write)
        raw = decision_envelope(pending["alpha"])
        assert set(raw["approval_decision"]) == {
            "approval_id", "decision", "session_key", "session_incarnation", "run_id", "generation",
        }
        assert raw["turn_id"] not in {pending["alpha"].turn_id, pending["alpha"].run_id, "parent-turn"}
        raw["media"] = [{"name": "ignored.txt", "data": "not control authority"}]
        raw["quoted_context"] = "not model input"
        await channel._dispatch_envelope(connection, "client", raw)
        errors = [frame for frame in frames(connection) if frame.get("event") == "error"]
        assert not errors, errors
        await wait_child_approval(loop, provider.alpha_held, channel=channel)
        assert parent_facts(loop, channel) == facts
        assert not provider.parent_release.is_set() and not provider.alpha_release.is_set()
        accepted = [frame for frame in frames(connection) if frame.get("event") == "message_accepted"]
        assert accepted[-1]["turn_id"] == "decide-alpha" and accepted[-1]["starts_turn"] is False
        beta = loop._fork_dependencies.foreground_manager._delegation_tree.child_for_session(pending["beta"].session_key)
        manager = loop._fork_dependencies.foreground_manager
        resumed = manager.resume_child_approval_decision
        beta_decided = asyncio.Event()

        async def observe_resume(**kwargs):
            result = await resumed(**kwargs)
            if kwargs["child_id"] == beta.child_id:
                beta_decided.set()
            return result

        monkeypatch.setattr(manager, "resume_child_approval_decision", observe_resume)
        await channel._dispatch_envelope(connection, "client", decision_envelope(pending["beta"], "deny", turn_id="decide-beta"))
        await asyncio.wait_for(beta_decided.wait(), timeout=5)
        tree = loop._fork_dependencies.foreground_manager._delegation_tree
        settled = await tree.wait_child(beta.child_id, timeout=5)
        assert settled.state == "stopped" and not settled.holds_capacity
        assert provider.child_calls["beta.txt"] == 1
        assert (engine_config.workspace_path / "beta.txt").read_text() == "protected line\n" * 20
        assert (engine_config.workspace_path / "alpha.txt").read_text() == "approved alpha.txt\n"
        assert parent_facts(loop, channel) == facts
        assert provider.parent_calls == 2
        assert [messages for messages, _, _ in provider.calls
                if not any("Worker Role" in str(row.get("content")) for row in messages)] == parent_inputs
        assert not [event for event in lifecycle if event.context.session_key == KEY]
        while loop.bus.outbound_size:
            await channel.send(loop.bus.outbound.get_nowait())
        live_updates = {
            (frame["child_session_key"], frame["state"])
            for frame in frames(connection)
            if frame.get("event") == "delegated_child"
            and frame.get("chat_id") == CHAT
            and frame.get("pending_approval_id") is None
        }
        assert (pending["alpha"].session_key, "active") in live_updates
        assert (pending["beta"].session_key, "stopped") in live_updates
        decisions = [
            frame for frame in frames(connection)
            if frame.get("event") == "message_accepted"
            and frame.get("turn_id") in {"decide-alpha", "decide-beta"}
        ]
        assert len(decisions) == 2 and all(frame["starts_turn"] is False for frame in decisions)
        retire.assert_not_called()
        invalidate.assert_not_called()
        transcript.assert_not_called()
        defaults.assert_not_called()
        persist_scope.assert_not_called()
        attachments.assert_not_called()
        write.assert_awaited_once()
        assert not any(frame.get("event") == "user" and frame.get("text") == "approval_decision" for frame in frames(connection))
        target = loop.stop_target_for_key(KEY)
        assert target is not None
        await channel._dispatch_envelope(connection, "client", {
            "type": "message", "chat_id": CHAT, "content": "/stop", "webui": True,
            "turn_id": "stop-siblings", "request_settings_version": 1, "request_intent": "new",
            "stop_expectation": target,
        })
        await asyncio.wait_for(asyncio.gather(*facts["tasks"], return_exceptions=True), timeout=5)
        assert not provider.parent_release.is_set() and not provider.alpha_release.is_set()


@pytest.mark.parametrize("pending_turn_id", [None, "child-inner-turn"])
@pytest.mark.parametrize("echo", ["omitted", "null", "matching"])
async def test_optional_echoes_corroborate_without_becoming_required(engine_config, monkeypatch, pending_turn_id, echo):
    async with siblings(engine_config, monkeypatch, pending_turn_id=pending_turn_id) as (loop, provider, channel, connection, pending):
        before = parent_facts(loop, channel)
        manager = loop._fork_dependencies.foreground_manager
        tree = manager._delegation_tree
        alpha = tree.child_for_session(pending["alpha"].session_key)
        beta = tree.child_for_session(pending["beta"].session_key)
        write_tool = manager._suspended_children[alpha.child_id].request.tools.get("write_file")
        write = AsyncMock(wraps=write_tool.execute)
        resume = AsyncMock(wraps=manager.resume_child_approval_decision)
        monkeypatch.setattr(write_tool, "execute", write)
        monkeypatch.setattr(manager, "resume_child_approval_decision", resume)
        raw = decision_envelope(pending["alpha"], turn_id="optional-echo-attempt")
        if echo != "omitted":
            raw["approval_decision"].update(
                effect_id=pending["alpha"].effect_id if echo == "matching" else None,
                turn_id=pending_turn_id if echo == "matching" else None,
            )
        sent = deepcopy(raw["approval_decision"])
        await channel._dispatch_envelope(connection, "client", raw)
        await wait_child_approval(loop, provider.alpha_held)
        resume.assert_awaited_once()
        decision = resume.await_args.kwargs["decision"]
        assert decision.effect_id == sent.get("effect_id")
        assert decision.turn_id == sent.get("turn_id")
        assert raw["approval_decision"] == sent
        assert parent_facts(loop, channel) == before
        assert tree.child(beta.child_id) == beta
        assert pending["beta"].approval_id in loop.approval_admission._pending
        assert pending["alpha"].approval_id not in loop.approval_admission._pending
        write.assert_awaited_once()
        assert (engine_config.workspace_path / "alpha.txt").read_text() == "approved alpha.txt\n"


@pytest.mark.parametrize("fault", ["absent", "empty", "malformed", "short", "uppercase", "mismatched-child", "missing-child"])
async def test_invalid_canonical_effect_refuses_before_child_scheduling(engine_config, monkeypatch, fault):
    async with siblings(engine_config, monkeypatch) as (loop, provider, channel, connection, pending):
        manager = loop._fork_dependencies.foreground_manager
        tree = manager._delegation_tree
        alpha = tree.child_for_session(pending["alpha"].session_key)
        beta = tree.child_for_session(pending["beta"].session_key)
        if fault in {"mismatched-child", "missing-child"}:
            effect = None if fault == "missing-child" else "0" * 64
        else:
            effect = {"absent": None, "empty": "", "malformed": "not-a-fingerprint",
                      "short": "a" * 63, "uppercase": "A" * 64}[fault]
            pending["alpha"] = replace_pending(loop, pending["alpha"], effect_id=effect)
        tree._children[alpha.child_id] = replace(alpha, pending_effect_id=effect)
        schedule = Mock(wraps=loop.schedule_background)
        monkeypatch.setattr(loop, "schedule_background", schedule)
        raw = decision_envelope(pending["alpha"])
        # Matching optional echoes cannot make an invalid canonical effect valid.
        if pending["alpha"].effect_id:
            raw["approval_decision"]["effect_id"] = pending["alpha"].effect_id
        before = parent_facts(loop, channel)
        calls = dict(provider.child_calls)
        await channel._dispatch_envelope(connection, "client", raw)
        refused = await wait_refusal(loop)
        assert refused.turn_id == "decide-alpha" and refused.reason == "approval_stale"
        assert not [call for call in schedule.call_args_list
                    if call.args[0].__qualname__.endswith(".resume_child")]
        assert parent_facts(loop, channel) == before
        assert tree.child(beta.child_id) == beta
        assert provider.child_calls == calls
        assert all(item.approval_id in loop.approval_admission._pending for item in pending.values())
        assert (engine_config.workspace_path / "alpha.txt").read_text() == "protected line\n" * 20


async def test_retargeted_child_effect_cannot_spend_original_consent(engine_config, monkeypatch):
    async with siblings(engine_config, monkeypatch) as (loop, provider, channel, connection, pending):
        manager = loop._fork_dependencies.foreground_manager
        tree = manager._delegation_tree
        alpha = tree.child_for_session(pending["alpha"].session_key)
        beta = tree.child_for_session(pending["beta"].session_key)
        write_tool = manager._suspended_children[alpha.child_id].request.tools.get("write_file")
        write = AsyncMock(wraps=write_tool.execute)
        chat = provider.chat
        resume = manager.resume_child_approval_decision
        finished = asyncio.Event()

        async def retarget(*args, **kwargs):
            response = await chat(*args, **kwargs)
            for call in response.tool_calls:
                if call.id == "write-alpha.txt-2":
                    call.arguments["content"] = "changed prepared effect\n"
            return response

        async def observe_resume(**kwargs):
            result = await resume(**kwargs)
            finished.set()
            return result

        monkeypatch.setattr(write_tool, "execute", write)
        monkeypatch.setattr(provider, "chat", retarget)
        monkeypatch.setattr(manager, "resume_child_approval_decision", observe_resume)
        before = parent_facts(loop, channel)
        await channel._dispatch_envelope(connection, "client", decision_envelope(pending["alpha"]))
        await asyncio.wait_for(finished.wait(), timeout=5)
        current = tree.child(alpha.child_id)
        assert current.state == "waiting_approval"
        assert current.pending_approval_id != pending["alpha"].approval_id
        replacement = loop.approval_admission._pending[current.pending_approval_id]
        assert replacement.effect_id != pending["alpha"].effect_id
        assert current.pending_effect_id == replacement.effect_id
        assert pending["alpha"].approval_id not in loop.approval_admission._pending
        assert parent_facts(loop, channel) == before
        assert tree.child(beta.child_id) == beta
        assert pending["beta"].approval_id in loop.approval_admission._pending
        write.assert_not_awaited()
        assert provider.child_calls == {"alpha.txt": 2, "beta.txt": 1}
        assert (engine_config.workspace_path / "alpha.txt").read_text() == "protected line\n" * 20


@pytest.mark.parametrize("fault", ["missing-session", "missing-run", "missing-incarnation", "missing-generation", "null-session", "null-run", "null-incarnation", "null-generation", "wrong-run", "wrong-incarnation", "wrong-generation", "wrong-effect", "wrong-turn", "history-turn", "parent-turn", "delivery-turn", "wrong-sibling", "wrong-parent", "wrong-source", "unauthorized", "untrusted", "stale", "malformed", "unknown-field", "duplicate-field"])
async def test_invalid_typed_controls_never_fall_back_to_parent(engine_config, monkeypatch, fault):
    async with siblings(engine_config, monkeypatch) as (loop, provider, channel, connection, pending):
        raw = decision_envelope(pending["alpha"])
        if fault.startswith(("missing-", "null-")):
            prefix, name = fault.split("-", 1)
            field = {"session": "session_key", "run": "run_id", "incarnation": "session_incarnation", "generation": "generation"}[name]
            if prefix == "missing":
                raw["approval_decision"].pop(field)
            else:
                raw["approval_decision"][field] = None
        elif fault == "wrong-run":
            raw["approval_decision"]["run_id"] = "wrong"
        elif fault in {"wrong-incarnation", "wrong-generation", "wrong-effect", "wrong-turn"}:
            field = {"wrong-incarnation": "session_incarnation", "wrong-generation": "generation",
                     "wrong-effect": "effect_id", "wrong-turn": "turn_id"}[fault]
            raw["approval_decision"][field] = 99 if field == "generation" else "wrong"
        elif fault in {"history-turn", "parent-turn", "delivery-turn"}:
            raw["approval_decision"]["turn_id"] = {
                "history-turn": pending["alpha"].run_id,
                "parent-turn": turns.websocket_turn_id(CHAT),
                "delivery-turn": raw["turn_id"],
            }[fault]
        elif fault == "wrong-sibling":
            raw["approval_decision"]["session_key"] = pending["beta"].session_key
        elif fault == "wrong-parent":
            raw["chat_id"] = "another-parent"
        elif fault == "stale":
            raw["approval_decision"]["approval_id"] = "stale"
        elif fault == "wrong-source":
            monkeypatch.setattr(channel.gateway.workspaces, "source_token", lambda _chat: "wrong")
        elif fault == "unauthorized":
            monkeypatch.setattr(channel, "is_allowed", lambda _client: False)
        elif fault == "untrusted":
            channel._webui_connections.discard(connection)
        elif fault == "malformed":
            raw["approval_decision"] = None
        elif fault == "unknown-field":
            raw["approval_decision"]["other"] = True
        else:
            encoded = json.dumps(raw).replace('"decision": "approve"', '"decision": "deny", "decision": "approve"')
            raw = _parse_envelope(encoded)
        before = parent_facts(loop, channel)
        calls = dict(provider.child_calls)
        await channel._dispatch_envelope(connection, "client", raw)
        errors = [frame for frame in frames(connection) if frame.get("event") == "error"]
        if not errors:
            refusal = await wait_refusal(loop)
            assert refusal.turn_id == "decide-alpha"
        else:
            assert errors[-1]["turn_id"] == "decide-alpha"
        assert parent_facts(loop, channel) == before
        assert provider.child_calls == calls
        assert all(item.approval_id in loop.approval_admission._pending for item in pending.values())


async def test_idle_child_approval_and_duplicate_leave_no_parent_reservation(engine_config, monkeypatch):
    async with siblings(engine_config, monkeypatch) as (loop, provider, channel, connection, pending):
        roots = loop._active_root_task_identity(KEY)
        provider.parent_release.set()
        await asyncio.wait_for(asyncio.gather(*roots), timeout=5)
        while loop.bus.outbound_size:
            await channel.send(loop.bus.outbound.get_nowait())
        assert not turns.websocket_turn_reserved(CHAT)
        assert not loop._active_root_task_identity(KEY)
        before = parent_facts(loop, channel)
        manager = loop._fork_dependencies.foreground_manager
        alpha = manager._delegation_tree.child_for_session(pending["alpha"].session_key)
        write_tool = manager._suspended_children[alpha.child_id].request.tools.get("write_file")
        write = AsyncMock(wraps=write_tool.execute)
        monkeypatch.setattr(write_tool, "execute", write)
        raw = decision_envelope(pending["alpha"])
        await channel._dispatch_envelope(connection, "client", raw)
        await asyncio.wait_for(provider.alpha_held.wait(), timeout=5)
        assert parent_facts(loop, channel) == before
        assert pending["beta"].approval_id in loop.approval_admission._pending
        calls = dict(provider.child_calls)
        await channel._dispatch_envelope(connection, "client", deepcopy(raw))
        refused = await wait_refusal(loop)
        assert refused.turn_id == "decide-alpha" and refused.reason == "approval_stale"
        assert provider.child_calls == calls
        assert parent_facts(loop, channel) == before
        backend = loop.sessions.source_backend(pending["alpha"].session_key)
        events = backend.source.load_source_snapshot(pending["alpha"].session_key).events
        assert sum(event.get("tool_call_id") == "write-alpha.txt-2" for event in events) == 1
        write.assert_awaited_once()
        assert (engine_config.workspace_path / "alpha.txt").read_text() == "approved alpha.txt\n"
        changed = deepcopy(raw)
        changed["approval_decision"]["decision"] = "deny"
        await channel._dispatch_envelope(connection, "client", changed)
        assert frames(connection)[-1]["reason"] == "attempt_payload_changed"
        assert provider.child_calls == calls


@pytest.mark.parametrize("change", ["revoked", "expired", "replaced-approval", "replaced-pending-object", "malformed-pending-effect", "absent-pending-effect", "changed-child-effect", "missing-child-effect", "replaced-source", "cancelled", "original-stop", "capability-revoked"])
async def test_child_controls_revalidate_after_scheduling_without_touching_sibling_or_root(engine_config, monkeypatch, change):
    async with siblings(engine_config, monkeypatch) as (loop, provider, channel, connection, pending):
        manager = loop._fork_dependencies.foreground_manager
        tree = manager._delegation_tree
        alpha = tree.child_for_session(pending["alpha"].session_key)
        beta = tree.child_for_session(pending["beta"].session_key)
        scheduled = asyncio.Event()
        captured = []
        schedule = loop.schedule_background

        def hold(coro):
            if coro.__qualname__.endswith(".resume_child"):
                captured.append(coro)
                scheduled.set()
            else:
                schedule(coro)

        monkeypatch.setattr(loop, "schedule_background", hold)
        await channel._dispatch_envelope(connection, "client", decision_envelope(pending["alpha"]))
        await asyncio.wait_for(scheduled.wait(), timeout=5)
        assert len(captured) == 1
        if change == "revoked":
            await loop.approval_admission.revoke(alpha.session_key)
        elif change in {"expired", "replaced-pending-object", "malformed-pending-effect", "absent-pending-effect"}:
            changes = {
                "expired": {"expires_at": 1.0},
                "replaced-pending-object": {},
                "malformed-pending-effect": {"effect_id": "not-a-fingerprint"},
                "absent-pending-effect": {"effect_id": None},
            }[change]
            replace_pending(loop, pending["alpha"], **changes)
        elif change in {"changed-child-effect", "missing-child-effect"}:
            tree._children[alpha.child_id] = replace(
                alpha, pending_effect_id=None if change == "missing-child-effect" else "0" * 64,
            )
        elif change == "replaced-approval":
            previous = pending["alpha"]
            await loop.approval_admission.store(
                session_key=alpha.session_key, channel="websocket", chat_id=CHAT,
                summary="replacement", session_incarnation=previous.session_incarnation,
                run_id=previous.run_id, effect_id=previous.effect_id, generation=previous.generation,
                source_state=previous.source_state,
            )
        elif change == "replaced-source":
            backend = loop.sessions.source_backend(alpha.session_key)
            snapshot = backend.source.load_source_snapshot(alpha.session_key)
            backend.source.reset_source_session(alpha.session_key, snapshot.state, [])
            loop.sessions.invalidate(alpha.session_key)
        elif change == "cancelled":
            await tree.stop_subtree(alpha.child_id)
        elif change == "original-stop":
            roots = loop._active_root_task_identity(KEY)
            target = loop.stop_target_for_key(KEY)
            assert target is not None
            await channel._dispatch_envelope(connection, "client", {
                "type": "message", "chat_id": CHAT, "content": "/stop", "webui": True,
                "turn_id": "stop-before-resume", "request_settings_version": 1, "request_intent": "new",
                "stop_expectation": target,
            })
            await asyncio.wait_for(asyncio.gather(*roots, return_exceptions=True), timeout=5)
            assert not provider.parent_release.is_set()
        else:
            suspended = manager._suspended_children[alpha.child_id]
            suspended.parent_bindings.release()
        before = parent_facts(loop, channel)
        sibling = tree.child(beta.child_id)
        sibling_pending = pending["beta"].approval_id in loop.approval_admission._pending
        if change != "original-stop":
            assert sibling_pending
        calls = dict(provider.child_calls)
        schedule(captured.pop())
        refused = await wait_refusal(loop)
        assert refused.turn_id == "decide-alpha" and refused.reason == "approval_stale"
        assert parent_facts(loop, channel) == before
        assert tree.child(beta.child_id) == sibling
        assert (pending["beta"].approval_id in loop.approval_admission._pending) == sibling_pending
        assert provider.child_calls == calls
        assert (engine_config.workspace_path / "alpha.txt").read_text() == "protected line\n" * 20
