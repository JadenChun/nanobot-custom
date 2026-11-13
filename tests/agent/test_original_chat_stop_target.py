"""Exact observed Stop control through canonical owners and the real bus."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nanobot.agent.stop_target import parse_stop_expectation
from nanobot.agent.turn import TurnRequest
from nanobot.bus.events import (
    INBOUND_META_STOP_CLIENT,
    INBOUND_META_STOP_EXPECTATION,
    InboundMessage,
)
from nanobot.bus.outbound_events import StopTargetEvent, WebUIAdmissionRejectedEvent
from nanobot.bus.runtime_events import TurnRunStatusChanged
from nanobot.command.builtin import cmd_stop
from nanobot.command.router import CommandContext
from nanobot.providers.base import LLMResponse
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.pending_followups import (
    PendingChildFollowup,
    mint_pending_followup_id,
    record_pending_followup,
)
from nanobot.session.persistence_metadata import SOURCE_PENDING_FOLLOWUPS_METADATA_KEY
from nanobot.session.source_store import SourceSessionStore
from tests.migration.test_local_runtime_activation import (
    _config,
    _running_composition,
    _ScriptedProvider,
)


@pytest.fixture(autouse=True)
def isolate_exact_stop_projection():
    from nanobot.session.webui_turns import clear_websocket_turns

    clear_websocket_turns("exact-stop")
    yield
    clear_websocket_turns("exact-stop")


def stop_message(target, *, chat_id="exact-stop"):
    metadata = {INBOUND_META_STOP_CLIENT: True, "webui_turn_id": "stop-attempt"}
    if target is not None:
        metadata[INBOUND_META_STOP_EXPECTATION] = parse_stop_expectation(target)
    return InboundMessage(
        channel="websocket", sender_id="webui", chat_id=chat_id,
        content="/stop", metadata=metadata,
    )


async def stop_direct(agent, target, *, chat_id="exact-stop"):
    message = stop_message(target, chat_id=chat_id)
    return await cmd_stop(CommandContext(
        msg=message, session=None, key=message.session_key, raw="/stop", loop=agent,
    ))


async def pending_record(handle, key, text="saved input"):
    message = InboundMessage(
        channel="websocket", sender_id="webui", chat_id=key.split(":", 1)[1],
        content=text,
    )
    mint_pending_followup_id(message)
    record = record_pending_followup(message)
    backend = handle.sessions.source_backend(key)
    assert backend is not None
    async with acquire_execution(backend, key, run_id="test-journal") as owner:
        assert backend.source.admit_pending_followup(key, owner, record)
    return record


async def rejection_from_bus(handle):
    while True:
        reply = await asyncio.wait_for(handle.bus.consume_outbound(), timeout=5)
        if isinstance(reply.event, WebUIAdmissionRejectedEvent):
            return reply


@pytest.mark.parametrize("change", ["held-child", "root-pending", "live-queue", "idle-child-pending", "nested-pending", "malformed", "orphan", "child-budget", "source-mismatch"])
async def test_legacy_stop_proves_root_only_before_any_mutation(monkeypatch, tmp_path, change):
    import nanobot.agent.loop as loop_module
    from nanobot.session.webui_turns import (
        activate_queued_websocket_turn,
        register_queued_websocket_turn_if_idle,
        websocket_turn_id,
    )

    provider = _ScriptedProvider([LLMResponse(content="unused")], block_first=True)
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        agent = handle.agent
        key = "websocket:exact-stop"
        owner = register_queued_websocket_turn_if_idle("exact-stop", "legacy-root", publish=False)
        activate_queued_websocket_turn("exact-stop", owner)
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket", sender_id="webui", chat_id="exact-stop", content="first",
            metadata={"webui_turn_id": "legacy-root", "_websocket_turn_owner": owner},
        ))
        await asyncio.wait_for(provider.first_started.wait(), timeout=5)
        assert websocket_turn_id("exact-stop") == "legacy-root"
        tree = agent._fork_dependencies.foreground_manager._delegation_tree
        if change == "root-pending":
            admitted = asyncio.Event()
            original_admit = agent._admit_pending_activity

            async def observe(*args, **kwargs):
                result = await original_admit(*args, **kwargs)
                admitted.set()
                return result

            monkeypatch.setattr(agent, "_admit_pending_activity", observe)
            await handle.bus.publish_inbound(InboundMessage(
                channel="websocket", sender_id="webui", chat_id="exact-stop", content="queued",
            ))
            await asyncio.wait_for(admitted.wait(), timeout=5)
        elif change == "live-queue":
            agent._pending_queues[key].put_nowait(InboundMessage(
                channel="websocket", sender_id="webui", chat_id="exact-stop", content="live queued",
            ))
        elif change in {"held-child", "idle-child-pending", "nested-pending", "orphan", "child-budget", "source-mismatch"}:
            child = await tree.reserve(parent=handle.sessions.get_or_create(key), parent_key=key,
                                       assignment="delayed work", attempt_id="delayed-child")
            if change == "nested-pending":
                nested = await tree.reserve(
                    parent=handle.sessions.get_or_create(child.session_key), parent_key=child.session_key,
                    assignment="nested delayed work", attempt_id="nested-delayed-child",
                )
                await tree.complete(nested.child_id, outcome="completed")
                await pending_record(handle, nested.session_key)
            if change != "held-child":
                await tree.complete(child.child_id, outcome="completed")
                child = tree.child(child.child_id)
            if change == "idle-child-pending":
                await pending_record(handle, child.session_key)
            elif change == "orphan":
                tree._children[child.child_id] = replace(child, parent_key="delegated:absent")
            elif change == "source-mismatch":
                tree._children[child.child_id] = replace(
                    child, source_state=replace(child.source_state, generation=99),
                )
            elif change == "child-budget":
                monkeypatch.setattr(loop_module, "MAX_STOP_TARGET_CHILDREN", 0)
        else:
            original_load = SourceSessionStore.load_source_snapshot

            def load(self, session_key, *args, **kwargs):
                snapshot = original_load(self, session_key, *args, **kwargs)
                if session_key == key:
                    snapshot = replace(snapshot, header={**snapshot.header, "metadata": {
                        **snapshot.header["metadata"], SOURCE_PENDING_FOLLOWUPS_METADATA_KEY: [{"bad": True}],
                    }})
                return snapshot

            monkeypatch.setattr(SourceSessionStore, "load_source_snapshot", load)
        backend = handle.sessions.source_backend(key)
        before = backend.get_session_path(key).read_bytes()
        children = dict(tree._children)
        epoch = tree.original_chat_control_epoch(key)
        activity = agent._source_activity_admissions[key]
        pending = await agent.approval_admission.store(
            session_key=key, channel="websocket", chat_id="exact-stop", summary="keep approval",
        )
        retire = Mock(wraps=agent._retire_workspace_task_before_input)
        invalidate = Mock(wraps=handle.sessions.invalidate_workspace_access_request)
        monkeypatch.setattr(agent, "_retire_workspace_task_before_input", retire)
        monkeypatch.setattr(handle.sessions, "invalidate_workspace_access_request", invalidate)
        message = stop_message(None)
        message.metadata["webui_target_turn_id"] = "legacy-root"
        reply = await cmd_stop(CommandContext(msg=message, session=None, key=key,
                                            raw="/stop", loop=agent))
        assert isinstance(reply.event, WebUIAdmissionRejectedEvent)
        assert reply.event.reason == "stale_stop_target"
        assert tree.original_chat_control_epoch(key) == epoch
        assert tree._children == children and not tree._stopping_sessions
        assert activity.followup_stop_admission is None
        assert not activity.root_task.done() and not activity.root_task.cancelling()
        assert backend.get_session_path(key).read_bytes() == before
        assert pending.approval_id in agent.approval_admission._pending
        retire.assert_not_called()
        invalidate.assert_not_called()


async def test_legacy_stop_accepts_real_root_only(monkeypatch, tmp_path):
    from nanobot.session.webui_turns import (
        activate_queued_websocket_turn,
        register_queued_websocket_turn_if_idle,
    )

    provider = _ScriptedProvider([LLMResponse(content="unused")], block_first=True)
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        owner = register_queued_websocket_turn_if_idle("exact-stop", "legacy-root", publish=False)
        activate_queued_websocket_turn("exact-stop", owner)
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket", sender_id="webui", chat_id="exact-stop", content="first",
            metadata={"webui_turn_id": "legacy-root", "_websocket_turn_owner": owner},
        ))
        await asyncio.wait_for(provider.first_started.wait(), timeout=5)
        message = stop_message(None)
        message.metadata["webui_target_turn_id"] = "legacy-root"
        reply = await cmd_stop(CommandContext(msg=message, session=None, key=message.session_key,
                                            raw="/stop", loop=handle.agent))
        assert reply.event is None and reply.content == "Stopped 1 task(s)."
        assert provider.first_cancelled.is_set()


@pytest.mark.parametrize("state", ["root-only", "pending-only", "unavailable"])
async def test_real_children_get_derives_live_stop_target_without_event(monkeypatch, tmp_path, state):
    from urllib.parse import quote

    from nanobot.channels.websocket.tests.ws_test_client import http_get
    from tests.webui.test_child_thread_route import _auth_headers, _make_channel

    provider = _ScriptedProvider([LLMResponse(content="parent answered")], block_first=state == "root-only")
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        key = "websocket:exact-stop"
        if state == "root-only":
            await handle.bus.publish_inbound(InboundMessage(
                channel="websocket", sender_id="webui", chat_id="exact-stop", content="first",
            ))
            await asyncio.wait_for(provider.first_started.wait(), timeout=5)
        else:
            await handle.agent.execute_turn(TurnRequest(content="first", session_key=key))
            await pending_record(handle, key)
            if state == "unavailable":
                monkeypatch.setattr(handle.agent, "_stop_pending_inventory", Mock(side_effect=ValueError("unavailable")))
        channel = _make_channel(tmp_path, handle.sessions)
        channel.gateway.http.stop_target_for_key = handle.agent.stop_target_for_key
        try:
            response = await http_get(
                f"http://127.0.0.1:{channel.config.port}/api/sessions/{quote(key, safe='')}/delegated-children",
                headers=_auth_headers(channel),
            )
            assert response.status_code == 200, response.text
            target = response.json()["stop_target"]
            assert target == handle.agent.stop_target_for_key(key)
            if state == "unavailable":
                assert target is None
            else:
                assert target is not None and target["children"] == []
                assert target["active_root_present"] is (state == "root-only")
        finally:
            await channel.stop()


@pytest.mark.parametrize("entry", ["bus", "sdk"])
async def test_canonical_root_publishes_live_target_after_queue_binding(monkeypatch, tmp_path, entry):
    provider = _ScriptedProvider([LLMResponse(content="done")], block_first=True)
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        agent = handle.agent
        key = "websocket:exact-stop"
        if entry == "bus":
            await handle.bus.publish_inbound(InboundMessage(
                channel="websocket", sender_id="webui", chat_id="exact-stop", content="first",
            ))
        else:
            asyncio.create_task(agent.execute_turn(TurnRequest(content="first", session_key=key)))
        await asyncio.wait_for(provider.first_started.wait(), timeout=5)
        target = agent.stop_target_for_key(key)
        assert target is not None
        roots = agent._active_root_task_identity(key)
        messages = [handle.bus.outbound.get_nowait() for _ in range(handle.bus.outbound_size)]
        targets = [message.event.stop_target for message in messages if isinstance(message.event, StopTargetEvent)]
        assert target in targets
        provider.release_first.set()
        await asyncio.wait_for(asyncio.gather(*roots), timeout=5)
        messages = [handle.bus.outbound.get_nowait() for _ in range(handle.bus.outbound_size)]
        targets = [message.event.stop_target for message in messages if isinstance(message.event, StopTargetEvent)]
        assert targets[-1] is None


async def test_matching_bus_root_uses_dispatch_task_and_discards_before_cancel(monkeypatch, tmp_path):
    provider = _ScriptedProvider([LLMResponse(content="unused")], block_first=True)
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, run):
        key = "websocket:exact-stop"
        first_saves = []
        original_save = handle.agent.run_store.save

        def save(record):
            if not first_saves:
                assert not handle.agent.run_store.path_for(record.run_id).exists()
                backend = handle.sessions.source_backend(key)
                assert not any(
                    event.get("_run_id") == record.run_id
                    for event in backend.source.load_source_snapshot(key).events
                )
                first_saves.append(record.run_id)
            return original_save(record)

        monkeypatch.setattr(handle.agent.run_store, "save", save)
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket", sender_id="webui", chat_id="exact-stop", content="first",
        ))
        await asyncio.wait_for(provider.first_started.wait(), timeout=5)
        agent = handle.agent
        admission = agent._source_activity_admissions[key]
        roots = agent._active_root_task_identity(key)
        assert roots == (admission.root_task,)
        assert admission.task not in roots
        assert admission.root_queue is agent._pending_queues[key]
        binding = agent._active_root_bindings[admission.root_task]
        assert binding.unique_run_id
        assert first_saves == [binding.run_id]
        assert binding.source_state.same_origin(admission.source_state)
        target = agent.stop_target_for_key(key)
        assert target is not None and target["active_root_present"] is True
        assert target["active_root_run_ids"] == [binding.run_id]
        assert target["children"] == []
        await handle.bus.publish_inbound(stop_message(target))
        while True:
            reply = await asyncio.wait_for(handle.bus.consume_outbound(), timeout=5)
            if reply.content.startswith("Stopped"):
                break
        assert provider.first_cancelled.is_set() and not run.done()
        assert len(provider.requests) == 1
        assert agent.stop_target_for_key(key) is None


@pytest.mark.parametrize("change", ["queue", "activity-source", "held-add", "held-remove", "child-run"])
async def test_real_bus_stale_stop_has_no_owner_mutation(monkeypatch, tmp_path, change):
    provider = _ScriptedProvider([LLMResponse(content="unused")], block_first=True)
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, run):
        key = "websocket:exact-stop"
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket", sender_id="webui", chat_id="exact-stop", content="first",
        ))
        await asyncio.wait_for(provider.first_started.wait(), timeout=5)
        agent = handle.agent
        tree = agent._fork_dependencies.foreground_manager._delegation_tree
        pending_admitted = asyncio.Event()
        original_admit = agent._admit_pending_activity

        async def observe_pending(*args, **kwargs):
            outcome = await original_admit(*args, **kwargs)
            pending_admitted.set()
            return outcome

        monkeypatch.setattr(agent, "_admit_pending_activity", observe_pending)
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket", sender_id="webui", chat_id="exact-stop", content="saved pending input",
        ))
        await asyncio.wait_for(pending_admitted.wait(), timeout=5)
        child = None
        if change in {"held-remove", "child-run"}:
            child = await tree.reserve(
                parent=handle.sessions.get_or_create(key), parent_key=key,
                assignment="owned work", attempt_id="child-original",
            )
            if change == "child-run":
                child = await tree.register_run(child.child_id, "original-child-run")
                assert child is not None
        target = agent.stop_target_for_key(key)
        assert target is not None
        original_queue = agent._pending_queues[key]
        activity = agent._source_activity_admissions[key]
        original_state = activity.source_state
        if change == "queue":
            agent._pending_queues[key] = asyncio.Queue()
        elif change == "activity-source":
            activity.source_state = replace(original_state, generation=original_state.generation + 1)
        elif change == "held-add":
            child = await tree.reserve(
                parent=handle.sessions.get_or_create(key), parent_key=key,
                assignment="new owned work", attempt_id="child-added",
            )
        elif change == "held-remove":
            await tree.complete(child.child_id, outcome="completed")
        else:
            replacement = await tree.register_run(child.child_id, "replacement-child-run")
            assert replacement is not None and replacement.run_sequence == child.run_sequence + 1
        backend = handle.sessions.source_backend(key)
        assert backend.source.load_source_snapshot(key).header["metadata"][SOURCE_PENDING_FOLLOWUPS_METADATA_KEY]
        source_before = backend.get_session_path(key).read_bytes()
        epoch = tree.original_chat_control_epoch(key)
        stopping = set(tree._stopping_sessions)
        pending = agent._pending_queues[key]
        child_states = dict(tree._children)
        pending_approval = await agent.approval_admission.store(
            session_key=key, channel="websocket", chat_id="exact-stop", summary="pending effect",
        )
        retire = Mock(wraps=agent._retire_workspace_task_before_input)
        invalidate = Mock(wraps=handle.sessions.invalidate_workspace_access_request)
        revoke = Mock(wraps=agent.approval_admission.revoke)
        discard = Mock(wraps=SourceSessionStore.discard_pending_followups)
        monkeypatch.setattr(agent, "_retire_workspace_task_before_input", retire)
        monkeypatch.setattr(handle.sessions, "invalidate_workspace_access_request", invalidate)
        monkeypatch.setattr(agent.approval_admission, "revoke", revoke)
        monkeypatch.setattr(SourceSessionStore, "discard_pending_followups", discard)
        statuses = []
        handle.bus.subscribe(statuses.append, TurnRunStatusChanged)
        try:
            await handle.bus.publish_inbound(stop_message(target))
            reply = await rejection_from_bus(handle)
            assert reply.event.detail == "not_admitted"
            assert reply.event.reason == "stale_stop_target"
            assert reply.event.turn_id == "stop-attempt"
            assert tree.original_chat_control_epoch(key) == epoch
            assert tree._stopping_sessions == stopping
            assert tree._children == child_states
            assert activity.followup_stop_admission is None
            assert agent._pending_queues[key] is pending
            assert backend.get_session_path(key).read_bytes() == source_before
            assert not activity.root_task.done() and not activity.root_task.cancelling()
            assert not provider.first_cancelled.is_set() and not run.done()
            assert not [event for event in statuses if event.status == "stopping"]
            retire.assert_not_called()
            invalidate.assert_not_called()
            revoke.assert_not_called()
            discard.assert_not_called()
            assert (await agent.approval_admission.snapshot_decision(key, None)).pending is pending_approval
        finally:
            activity.source_state = original_state
            agent._pending_queues[key] = original_queue


async def test_idle_pending_only_stop_discards_exact_durable_input(monkeypatch, tmp_path):
    provider = _ScriptedProvider([LLMResponse(content="initial")])
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        key = "websocket:exact-stop"
        await handle.agent.execute_turn(TurnRequest(content="first", session_key=key))
        assert handle.agent.stop_target_for_key(key) is None
        record = await pending_record(handle, key)
        target = handle.agent.stop_target_for_key(key)
        assert target is not None
        assert target["active_root_present"] is False
        assert target["active_turn_id"] is None
        assert target["active_root_run_ids"] == target["children"] == []
        result = await stop_direct(handle.agent, target)
        assert result.event is None and "stop" in result.content.lower()
        backend = handle.sessions.source_backend(key)
        metadata = backend.source.load_source_snapshot(key).header["metadata"]
        assert SOURCE_PENDING_FOLLOWUPS_METADATA_KEY not in metadata, record.identifier
        assert handle.agent.stop_target_for_key(key) is None
        assert len(provider.requests) == 1


async def test_captured_queue_replacement_and_settled_removal_keep_exact_ownership(monkeypatch, tmp_path):
    provider = _ScriptedProvider([LLMResponse(content="finished")], block_first=True)
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        agent = handle.agent
        key = "websocket:exact-stop"
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket", sender_id="webui", chat_id="exact-stop", content="first",
        ))
        await asyncio.wait_for(provider.first_started.wait(), timeout=5)
        root = agent._source_activity_admissions[key].root_task
        original = agent._pending_queues[key]
        admission = agent._capture_original_chat_stop(key, None)
        assert admission.pending_queue is original
        assert admission.active_root_identity == (root,)
        assert agent._validate_original_chat_stop(key, admission)
        agent._pending_queues.pop(key)
        assert not agent._validate_original_chat_stop(key, admission), "live queue removal must refuse"
        replacement = asyncio.Queue()
        agent._pending_queues[key] = replacement
        assert not agent._validate_original_chat_stop(key, admission)
        assert await agent._cancel_active_tasks(key, _stop_admission=admission) == 0
        assert not root.done() and not root.cancelling()
        provider.release_first.set()
        await asyncio.wait_for(root, timeout=5)
        assert root.done() and not root.cancelled()
        assert agent._pending_queues[key] is replacement
        assert not agent._validate_original_chat_stop(key, admission), "even settled roots cannot adopt a new queue"
        agent._pending_queues.pop(key)
        assert agent._validate_original_chat_stop(key, admission), "settled original queue removal is normal cleanup"


@pytest.mark.parametrize("replacement_kind", ["generation", "incarnation"])
async def test_real_source_replacement_refuses_without_mutation(monkeypatch, tmp_path, replacement_kind):
    provider = _ScriptedProvider([LLMResponse(content="unused")], block_first=True)
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        agent = handle.agent
        key = "websocket:exact-stop"
        await handle.bus.publish_inbound(InboundMessage(
            channel="websocket", sender_id="webui", chat_id="exact-stop", content="first",
        ))
        await asyncio.wait_for(provider.first_started.wait(), timeout=5)
        target = agent.stop_target_for_key(key)
        assert target is not None
        queue = agent._pending_queues[key]
        backend = handle.sessions.source_backend(key)
        snapshot = backend.source.load_source_snapshot(key)
        plan = backend.source.plan_source_reset(snapshot, [], updated_at="2026-11-13T00:00:00+00:00")
        replacement_bytes = plan.target_bytes
        if replacement_kind == "incarnation":
            replacement_bytes = replacement_bytes.replace(snapshot.state.incarnation.encode(), b"f" * 32)
        # Simulate the external replacement the paused request must refuse.
        # The old owner cannot authorize this write, and is deliberately stale.
        backend.get_session_path(key).write_bytes(replacement_bytes)
        replacement = backend.source.load_source_snapshot(key)
        assert replacement.state.generation != snapshot.state.generation
        if replacement_kind == "incarnation":
            assert replacement.state.incarnation != snapshot.state.incarnation
        before = backend.get_session_path(key).read_bytes()
        tree = agent._fork_dependencies.foreground_manager._delegation_tree
        epoch = tree.original_chat_control_epoch(key)
        retire = Mock(wraps=agent._retire_workspace_task_before_input)
        invalidate = Mock(wraps=handle.sessions.invalidate_workspace_access_request)
        monkeypatch.setattr(agent, "_retire_workspace_task_before_input", retire)
        monkeypatch.setattr(handle.sessions, "invalidate_workspace_access_request", invalidate)
        await handle.bus.publish_inbound(stop_message(target))
        reply = await rejection_from_bus(handle)
        assert reply.event.reason == "stale_stop_target"
        assert agent._pending_queues[key] is queue
        assert tree.original_chat_control_epoch(key) == epoch
        assert not tree._stopping_sessions
        assert backend.get_session_path(key).read_bytes() == before
        assert not provider.first_cancelled.is_set()
        retire.assert_not_called()
        invalidate.assert_not_called()


async def test_sdk_live_root_outside_active_tasks_is_stoppable(monkeypatch, tmp_path):
    provider = _ScriptedProvider([LLMResponse(content="unused")], block_first=True)
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        key = "websocket:exact-stop"
        task = asyncio.create_task(handle.agent.execute_turn(TurnRequest(content="sdk", session_key=key)))
        try:
            await asyncio.wait_for(provider.first_started.wait(), timeout=5)
            assert key not in handle.agent._active_tasks
            target = handle.agent.stop_target_for_key(key)
            assert target is not None and target["active_root_present"] is True
            await stop_direct(handle.agent, target)
            assert task.done() and provider.first_cancelled.is_set()
        finally:
            provider.release_first.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)


async def test_pending_replacement_refuses_before_mutation(monkeypatch, tmp_path):
    provider = _ScriptedProvider([LLMResponse(content="initial")])
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        key = "websocket:exact-stop"
        await handle.agent.execute_turn(TurnRequest(content="first", session_key=key))
        original = await pending_record(handle, key)
        target = deepcopy(handle.agent.stop_target_for_key(key))
        backend = handle.sessions.source_backend(key)
        async with acquire_execution(backend, key, run_id="replace-pending") as owner:
            assert backend.source.discard_pending_followups(key, owner) == (original.identifier,)
        await pending_record(handle, key, "replacement")
        before = backend.get_session_path(key).read_bytes()
        tree = handle.agent._fork_dependencies.foreground_manager._delegation_tree
        epoch = tree.original_chat_control_epoch(key)
        reply = await stop_direct(handle.agent, target)
        assert reply.event.reason == "stale_stop_target"
        assert backend.get_session_path(key).read_bytes() == before
        assert tree.original_chat_control_epoch(key) == epoch


@pytest.mark.parametrize("first_explicit,reuse", [(False, True), (True, True), (False, False)])
async def test_reused_canonical_run_id_executes_but_old_target_never_retargets(
    monkeypatch, tmp_path, first_explicit, reuse,
):
    provider = _ScriptedProvider([LLMResponse(content="first"), LLMResponse(content="second")], block_first=True)
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        agent = handle.agent
        key = "websocket:exact-stop"
        first = asyncio.create_task(agent.execute_turn(TurnRequest(
            content="one", session_key=key, run_id="caller-reused" if first_explicit else None,
        )))
        await asyncio.wait_for(provider.first_started.wait(), timeout=5)
        first_binding = agent._active_root_bindings[first]
        old_target = agent.stop_target_for_key(key)
        assert (old_target is None) is first_explicit
        messages = [handle.bus.outbound.get_nowait() for _ in range(handle.bus.outbound_size)]
        targets = [message.event.stop_target for message in messages if isinstance(message.event, StopTargetEvent)]
        assert targets[-1] == old_target
        if old_target is not None:
            assert old_target["active_turn_id"] is None
        provider.release_first.set()
        first_result = await asyncio.wait_for(first, timeout=5)
        while not handle.bus.outbound.empty():
            handle.bus.outbound.get_nowait()
        entered, release = asyncio.Event(), asyncio.Event()
        original_chat = provider.chat

        async def second_chat(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original_chat(*args, **kwargs)

        monkeypatch.setattr(provider, "chat", second_chat)
        second = asyncio.create_task(agent.execute_turn(TurnRequest(
            content="two", session_key=key, run_id=first_result.run_id if reuse else None,
        )))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            binding = agent._active_root_bindings[second]
            assert binding.source_state.same_origin(first_binding.source_state)
            assert binding.stream_id != first_binding.stream_id
            assert (binding.run_id == first_result.run_id) is reuse
            assert binding.unique_run_id is not reuse
            assert (agent.stop_target_for_key(key) is None) is reuse
            messages = [handle.bus.outbound.get_nowait() for _ in range(handle.bus.outbound_size)]
            targets = [message.event.stop_target for message in messages if isinstance(message.event, StopTargetEvent)]
            assert targets[-1] == agent.stop_target_for_key(key)
            tree = agent._fork_dependencies.foreground_manager._delegation_tree
            epoch = tree.original_chat_control_epoch(key)
            backend = handle.sessions.source_backend(key)
            before = backend.get_session_path(key).read_bytes()
            retirement = Mock(wraps=agent._retire_workspace_task_before_input)
            monkeypatch.setattr(agent, "_retire_workspace_task_before_input", retirement)
            reply = await stop_direct(agent, old_target)
            assert reply.event.reason == "stale_stop_target"
            assert tree.original_chat_control_epoch(key) == epoch
            assert not second.done() and not second.cancelling()
            assert backend.get_session_path(key).read_bytes() == before
            retirement.assert_not_called()
        finally:
            release.set()
            result = await asyncio.wait_for(second, timeout=5)
        assert (result.run_id == first_result.run_id) is reuse
        assert result.content == "second"


@pytest.mark.parametrize("existing", ["malformed", "unreadable", "source-attribution"])
async def test_generated_run_requires_positive_pre_save_absence_and_no_attribution(
    monkeypatch, tmp_path, existing,
):
    import nanobot.agent.loop as loop_module

    provider = _ScriptedProvider([LLMResponse(content="first"), LLMResponse(content="second")], block_first=True)
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        agent = handle.agent
        key = "websocket:exact-stop"
        run_id = "generated-collision"
        if existing == "source-attribution":
            provider.block_first = False
            prior = await agent.execute_turn(TurnRequest(content="prior", session_key=key, run_id=run_id))
            agent.run_store.path_for(prior.run_id).unlink()
            provider.requests.clear()
            provider.block_first = True
        path = agent.run_store.path_for(run_id)
        if existing == "malformed":
            path.write_text("not a run record", encoding="utf-8")
            assert agent.run_store.load(run_id) is None
        elif existing == "unreadable":
            original_lstat = type(path).lstat

            def lstat(self, *args, **kwargs):
                if self == path:
                    raise PermissionError("run identity cannot be inspected")
                return original_lstat(self, *args, **kwargs)

            monkeypatch.setattr(type(path), "lstat", lstat)
        monkeypatch.setattr(loop_module, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=run_id)))
        task = asyncio.create_task(agent.execute_turn(TurnRequest(content="current", session_key=key)))
        try:
            await asyncio.wait_for(provider.first_started.wait(), timeout=5)
            assert not agent._active_root_bindings[task].unique_run_id
            assert agent.stop_target_for_key(key) is None
            assert not task.done()
        finally:
            provider.release_first.set()
            await asyncio.wait_for(task, timeout=5)


async def test_eligibility_is_checked_again_when_echoing_same_public_target(monkeypatch, tmp_path):
    provider = _ScriptedProvider([LLMResponse(content="unused")], block_first=True)
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        agent = handle.agent
        key = "websocket:exact-stop"
        task = asyncio.create_task(agent.execute_turn(TurnRequest(content="one", session_key=key)))
        try:
            await asyncio.wait_for(provider.first_started.wait(), timeout=5)
            target = agent.stop_target_for_key(key)
            assert target is not None
            agent._active_root_bindings[task] = replace(agent._active_root_bindings[task], unique_run_id=False)
            tree = agent._fork_dependencies.foreground_manager._delegation_tree
            epoch = tree.original_chat_control_epoch(key)
            reply = await stop_direct(agent, target)
            assert reply.event.reason == "stale_stop_target"
            assert tree.original_chat_control_epoch(key) == epoch
            assert not task.done() and not task.cancelling()
        finally:
            provider.release_first.set()
            await asyncio.wait_for(task, timeout=5)


@pytest.mark.parametrize("nested", [False, True])
async def test_matching_children_only_stop_reaches_owned_subtree_without_model(monkeypatch, tmp_path, nested):
    provider = _ScriptedProvider([LLMResponse(content="parent answered")])
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        agent = handle.agent
        key = "websocket:exact-stop"
        await agent.execute_turn(TurnRequest(content="first", session_key=key))
        tree = agent._fork_dependencies.foreground_manager._delegation_tree
        child = await tree.reserve(
            parent=handle.sessions.get_or_create(key), parent_key=key,
            assignment="background child", attempt_id="child-background",
        )
        children = [child]
        if nested:
            children.append(await tree.reserve(
                parent=handle.sessions.get_or_create(child.session_key), parent_key=child.session_key,
                assignment="nested work", attempt_id="child-nested",
            ))
        tasks = []
        for item in children:
            task = asyncio.create_task(asyncio.Event().wait())
            tasks.append(task)
            await tree.attach_task(item.child_id, task)
        await asyncio.sleep(0)
        target = agent.stop_target_for_key(key)
        assert target is not None and target["active_root_present"] is False
        assert target["active_root_run_ids"] == []
        assert [row["child_id"] for row in target["children"]] == sorted(item.child_id for item in children)
        assert all(row["child_run_id"] is None for row in target["children"])
        await stop_direct(agent, target)
        assert all(task.cancelled() for task in tasks)
        assert all(not tree.child(item.child_id).holds_capacity for item in children)
        assert len(provider.requests) == 1
        assert agent.stop_target_for_key(key) is None


async def test_idle_pending_only_descendant_is_in_digest_and_stopped(monkeypatch, tmp_path):
    provider = _ScriptedProvider([LLMResponse(content="parent answered")])
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        agent = handle.agent
        key = "websocket:exact-stop"
        await agent.execute_turn(TurnRequest(content="first", session_key=key))
        tree = agent._fork_dependencies.foreground_manager._delegation_tree
        child = await tree.reserve(
            parent=handle.sessions.get_or_create(key), parent_key=key,
            assignment="idle child", attempt_id="idle-child",
        )
        await tree.complete(child.child_id, outcome="completed")
        assert agent.stop_target_for_key(key) is None
        await pending_record(handle, child.session_key)
        target = agent.stop_target_for_key(key)
        assert target is not None and target["children"] == []
        assert target["active_root_present"] is False
        reply = await stop_direct(agent, target)
        assert reply.event is None
        backend = handle.sessions.source_backend(child.session_key)
        assert SOURCE_PENDING_FOLLOWUPS_METADATA_KEY not in backend.source.load_source_snapshot(child.session_key).header["metadata"]
        assert agent.stop_target_for_key(key) is None
        assert len(provider.requests) == 1


async def test_terminal_typed_child_pending_stop_retains_fenced_evidence(monkeypatch, tmp_path):
    provider = _ScriptedProvider([LLMResponse(content="parent answered")])
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        agent = handle.agent
        key = "websocket:exact-stop"
        parent_result = await agent.execute_turn(TurnRequest(content="first", session_key=key))
        tree = agent._fork_dependencies.foreground_manager._delegation_tree
        child = await tree.reserve(
            parent=handle.sessions.get_or_create(key), parent_key=key,
            assignment="retained typed input", attempt_id="typed-idle-child",
        )
        child = await tree.register_run(child.child_id, "child-original-run")
        child = await tree.complete(child.child_id, outcome="completed")
        parent_state = tree.source_state(key)
        state = child.source_state
        intent = PendingChildFollowup(
            child_id=child.child_id, parent_key=key,
            parent_store_identity=parent_state.store_identity,
            parent_incarnation=parent_state.incarnation,
            parent_generation=parent_state.generation,
            parent_run_id=parent_result.run_id, parent_tool_event_id="tool-intent",
            parent_execution_id="tool-execution", parent_dispatch_ordinal=1,
            child_key=child.session_key, child_store_identity=state.store_identity,
            child_incarnation=state.incarnation, child_generation=state.generation,
            child_run_id=child.current_run_id, child_state_sequence=child.state_sequence,
            child_control_epoch=child.control_epoch,
        )
        generic = await pending_record(handle, child.session_key)
        record = replace(generic, child_followup=intent)
        backend = handle.sessions.source_backend(child.session_key)
        async with acquire_execution(backend, child.session_key, run_id="typed-journal") as owner:
            backend.source.discard_pending_followups(child.session_key, owner)
            backend.source.admit_pending_followup(child.session_key, owner, record)
        before = backend.source.load_source_snapshot(child.session_key)
        target = agent.stop_target_for_key(key)
        assert target is not None and target["children"] == []
        await stop_direct(agent, target)
        stopped = tree.child(child.child_id)
        assert stopped.control_epoch > intent.child_control_epoch
        after = backend.source.load_source_snapshot(child.session_key)
        assert after.events == before.events
        assert after.header["metadata"][SOURCE_PENDING_FOLLOWUPS_METADATA_KEY] == [record.to_dict()]
        assert agent.stop_target_for_key(key) is None

        # An older epoch is evidence only when the complete typed origin still
        # matches. A malformed or replaced owner must remain unavailable.
        for changed in (
            replace(intent, child_control_epoch=stopped.control_epoch + 1),
            replace(intent, child_incarnation="replaced-child"),
            replace(intent, parent_generation=intent.parent_generation + 1),
        ):
            async with acquire_execution(backend, child.session_key, run_id="replace-journal") as owner:
                backend.source.discard_pending_followups(child.session_key, owner)
                backend.source.admit_pending_followup(
                    child.session_key, owner, replace(record, child_followup=changed),
                )
            assert agent.stop_target_for_key(key) is None
            refused = await stop_direct(agent, target)
            assert isinstance(refused.event, WebUIAdmissionRejectedEvent)
            assert tree.child(child.child_id).control_epoch == stopped.control_epoch


@pytest.mark.parametrize("entry", ["internal", "legacy"])
async def test_canonical_pre_admission_compact_receipt_has_no_typed_target(monkeypatch, tmp_path, entry):
    provider = _ScriptedProvider([LLMResponse(content="parent answered")])
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        agent = handle.agent
        key = "websocket:exact-stop"
        await agent.execute_turn(TurnRequest(content="first", session_key=key))
        assert handle.sessions.source_backend(key) is not None
        async with agent._get_session_lock(key):
            compact = InboundMessage(
                channel="websocket", sender_id="webui", chat_id="exact-stop", content="/compact",
            )
            queued = agent._enqueue_bus_dispatch(compact, key)
            state = agent._compact_dispatches[queued]
            await asyncio.sleep(0)
            assert not queued.done()
            assert queued in agent._active_root_task_identity(key)
            assert queued not in agent._active_root_bindings
            assert agent.stop_target_for_key(key) is None
            queued.cancel()
            await asyncio.gather(queued, return_exceptions=True)
            assert queued.done() and queued.cancelled()
            assert agent._compact_dispatches.get(queued) is state
            assert agent.stop_target_for_key(key) is None
            refused = await stop_direct(agent, None)
            assert refused.event.reason == "stale_stop_target"
            assert agent._compact_dispatches.get(queued) is state
            if entry == "legacy":
                from nanobot.session.webui_turns import (
                    activate_queued_websocket_turn,
                    register_queued_websocket_turn_if_idle,
                )

                owner = register_queued_websocket_turn_if_idle("exact-stop", "compact-receipt", publish=False)
                activate_queued_websocket_turn("exact-stop", owner)
            internal_stop = InboundMessage(
                channel="websocket", sender_id="webui", chat_id="exact-stop", content="/stop",
                metadata=({INBOUND_META_STOP_CLIENT: True, "webui_target_turn_id": "compact-receipt"}
                          if entry == "legacy" else {}),
            )
            await cmd_stop(CommandContext(
                msg=internal_stop, session=None, key=key, raw="/stop", loop=agent,
            ))
            assert queued not in agent._compact_dispatches
            assert agent.stop_target_for_key(key) is None
        assert len(provider.requests) == 1


@pytest.mark.parametrize("fault", ["malformed", "pending-budget", "source-budget", "child-budget", "child-source", "orphan"])
async def test_incomplete_or_exhausted_inventory_has_no_partial_target_or_control(monkeypatch, tmp_path, fault):
    import nanobot.agent.loop as loop_module

    provider = _ScriptedProvider([LLMResponse(content="parent answered")])
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        agent = handle.agent
        key = "websocket:exact-stop"
        await agent.execute_turn(TurnRequest(content="first", session_key=key))
        await pending_record(handle, key)
        tree = agent._fork_dependencies.foreground_manager._delegation_tree
        target = agent.stop_target_for_key(key)
        assert target is not None
        if fault == "malformed":
            original_load = SourceSessionStore.load_source_snapshot

            def load(self, session_key, *args, **kwargs):
                snapshot = original_load(self, session_key, *args, **kwargs)
                if session_key == key:
                    snapshot = replace(snapshot, header={**snapshot.header, "metadata": {
                        **snapshot.header["metadata"], SOURCE_PENDING_FOLLOWUPS_METADATA_KEY: [{"bad": True}],
                    }})
                return snapshot

            monkeypatch.setattr(SourceSessionStore, "load_source_snapshot", load)
        elif fault == "pending-budget":
            monkeypatch.setattr(loop_module, "MAX_STOP_TARGET_PENDING_RECORDS", 0)
        elif fault == "source-budget":
            monkeypatch.setattr(loop_module, "MAX_STOP_TARGET_SOURCE_EVENTS", 0)
        else:
            child = await tree.reserve(
                parent=handle.sessions.get_or_create(key), parent_key=key,
                assignment="child inventory", attempt_id="child-inventory",
            )
            if fault == "child-budget":
                await tree.complete(child.child_id, outcome="completed")
                monkeypatch.setattr(loop_module, "MAX_STOP_TARGET_CHILDREN", 0)
            elif fault == "child-source":
                tree._children[child.child_id] = replace(child, source_state=None)
            else:
                tree._children[child.child_id] = replace(child, parent_key="delegated:missing-parent")
        assert agent.stop_target_for_key(key) is None
        epoch = tree.original_chat_control_epoch(key)
        children = dict(tree._children)
        reply = await stop_direct(agent, target)
        assert reply.event.reason == "stale_stop_target"
        assert tree.original_chat_control_epoch(key) == epoch
        assert tree._children == children
        assert not tree._stopping_sessions


async def test_sdk_closing_owner_cannot_appear_as_children_only_work(monkeypatch, tmp_path):
    from nanobot.agent.loop import _SourceActivityAdmission

    provider = _ScriptedProvider([LLMResponse(content="answered")], block_first=True)
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        agent = handle.agent
        key = "websocket:exact-stop"
        task = asyncio.create_task(agent.execute_turn(TurnRequest(content="one", session_key=key)))
        await asyncio.wait_for(provider.first_started.wait(), timeout=5)
        tree = agent._fork_dependencies.foreground_manager._delegation_tree
        await tree.reserve(
            parent=handle.sessions.get_or_create(key), parent_key=key,
            assignment="background child", attempt_id="closing-child",
        )
        entered, release = asyncio.Event(), asyncio.Event()
        close = _SourceActivityAdmission.close

        async def blocked_close(self):
            entered.set()
            await release.wait()
            await close(self)

        monkeypatch.setattr(_SourceActivityAdmission, "close", blocked_close)
        provider.release_first.set()
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            assert not task.done() and key not in agent._active_tasks
            assert agent.stop_target_for_key(key) is None
            assert task in agent._active_root_task_identity(key)
        finally:
            release.set()
            await asyncio.wait_for(task, timeout=5)
        target = agent.stop_target_for_key(key)
        assert target is not None and target["active_root_present"] is False
        assert len(target["children"]) == 1


async def test_followup_submitted_before_fence_cannot_commit_after_stop(monkeypatch, tmp_path):
    provider = _ScriptedProvider([LLMResponse(content="unused")], block_first=True)
    async with _running_composition(_config(tmp_path), provider, monkeypatch) as (handle, _run):
        agent = handle.agent
        key = "websocket:exact-stop"
        root = asyncio.create_task(agent.execute_turn(TurnRequest(content="first", session_key=key)))
        await asyncio.wait_for(provider.first_started.wait(), timeout=5)
        admission = agent._source_activity_admissions[key]
        entered, release = asyncio.Event(), asyncio.Event()
        record_activity = agent._record_foreground_activity

        async def blocked_activity(*args, **kwargs):
            entered.set()
            await release.wait()
            return await record_activity(*args, **kwargs)

        monkeypatch.setattr(agent, "_record_foreground_activity", blocked_activity)
        message = InboundMessage(channel="websocket", sender_id="webui", chat_id="exact-stop", content="activity")
        first = asyncio.create_task(admission.submit(message))
        await asyncio.wait_for(entered.wait(), timeout=5)
        queued_message = replace(message, content="must not become durable after Stop", metadata={})
        followup_id = mint_pending_followup_id(queued_message)
        queued = asyncio.create_task(admission.submit(queued_message, followup_id=followup_id))
        await asyncio.sleep(0)
        assert admission.queue.qsize() == 1
        target = agent.stop_target_for_key(key)
        assert target is not None
        stopper = asyncio.create_task(stop_direct(agent, target))
        try:
            async with asyncio.timeout(5):
                while admission.followup_stop_admission is None:
                    await asyncio.sleep(0)
            release.set()
            await asyncio.wait_for(first, timeout=5)
            outcome = await asyncio.wait_for(queued, timeout=5)
            assert outcome.followup_pending is False
            await asyncio.wait_for(stopper, timeout=5)
            backend = handle.sessions.source_backend(key)
            snapshot = backend.source.load_source_snapshot(key)
            assert SOURCE_PENDING_FOLLOWUPS_METADATA_KEY not in snapshot.header["metadata"]
            assert not any(event.get("content") == queued_message.content for event in snapshot.events)
            assert provider.first_cancelled.is_set()
        finally:
            release.set()
            provider.release_first.set()
            await asyncio.wait_for(asyncio.gather(first, queued, stopper, root, return_exceptions=True), timeout=5)
