"""Offline owning-inbox integration with the actual AgentLoop and session stores."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.automation_turns import AutomationTurnError
from nanobot.agent.busy_control import ControlDecision
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.turn import RunStatus
from nanobot.bus.events import InboundMessage
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.runtime_context import RuntimeContextBlock
from nanobot.session.busy_inbox import InboxPayload, Origin, SessionBusyInbox
from tests.agent.p6_admitted_fixtures import AgentInputs
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async


@pytest.fixture(params=["canonical", "legacy"])
async def owned(tmp_path, request, admitted_async):
    provider = MagicMock()
    provider.get_default_model.return_value = "offline-model"
    provider.generation = GenerationSettings()
    provider.supports_bound_conversation_state.return_value = False
    provider.can_resume_conversation_state.return_value = False
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done"))

    def build(inputs: AgentInputs):
        return AgentLoop(bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
                         model="offline-model", max_iterations=4, session_manager=inputs.sessions,
                         tool_registry=inputs.tools, cron_service=inputs.cron,
                         turn_delivery_factory=inputs.delivery, tools_config=inputs.config.tools,
                         restrict_to_workspace=True, enable_legacy_auto_compact=False)

    keys = ("assistant",) if request.param == "legacy" else ()
    loop = await admitted_async(tmp_path, build, keys=keys)
    inbox = SessionBusyInbox(loop.sessions, owner_id="owner", session_key="assistant")
    consumer = loop._own_busy_inbox(inbox, automation_scope=())
    return loop, inbox, consumer, provider


def payload(number, *, content=None, media=(), channel="telegram", scope=()):
    return InboxPayload(owner_id="owner", session_key="assistant", content=content or f"task {number}",
                        media=media, scope=scope,
                        origin=Origin(channel=channel, account_id="account", chat_id=f"chat-{number}",
                                      sender_id="user", transport_id=str(number), thread_id="thread"))


def route(inbox, consumer, number, *, intent="next-turn", **kwargs):
    row = inbox.admit(payload(number, **kwargs)).entry
    return inbox.route(row.id, row.revision, ControlDecision(intent, consumer.current_run, "offline fixture"),
                       current_run=consumer.current_run, active_scope=consumer.active_scope)


def automation(number=1):
    return InboundMessage(channel="telegram", chat_id="scheduled-chat", sender_id="cron",
                          content=f"scheduled {number}", session_key_override="assistant",
                          metadata={"_cron_trigger": {"run_id": f"occurrence-{number}", "job_id": "job"},
                                    "_cron_defer_until_session_idle": True})


async def test_burst_uses_one_real_runner_fifo_without_overflow_or_republish(owned):
    loop, inbox, consumer, provider = owned
    started, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def chat(**kwargs):
        seen.append(deepcopy(kwargs["messages"]))
        if len(seen) == 1:
            started.set()
            await release.wait()
        return LLMResponse(content="done")

    provider.chat_with_retry.side_effect = chat
    route(inbox, consumer, 0)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    owning_task = consumer.task
    try:
        for number in range(1, 26):
            route(inbox, consumer, number)
            consumer.wake()
            assert consumer.task is owning_task
        assert len(seen) == 1
        assert not loop._pending_queues and not loop._pending_republish
        assert not loop._deferred_automation_turns
        with pytest.raises(RuntimeError, match="inbox"):
            loop._enqueue_bus_dispatch(InboundMessage("telegram", "user", "x", "raw"), "assistant")
    finally:
        release.set()
    await asyncio.wait_for(consumer.wait_idle(), 180)
    assert consumer.last_error is None
    assert len(seen) == 26
    assert all(row.state == "completed" for row in inbox.entries())
    assert [next(m["content"] for m in reversed(messages) if m["role"] == "user")
            for messages in seen] == [f"task {number}" for number in range(26)]
    assert loop.bus.inbound_size == 0
    snapshot = loop.sessions.read_session_snapshot("assistant")
    assert snapshot is not None
    assert [row["content"] for row in snapshot.messages if row["role"] == "user"] == [f"task {n}" for n in range(26)]


async def test_initial_clarifications_and_media_have_one_owned_source_manifest(owned, tmp_path):
    loop, inbox, consumer, provider = owned
    image = tmp_path / "input.png"
    image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="))
    row = route(inbox, consumer, 1, intent="clarify", content="原始指令\ncomplete original", media=(str(image),))
    row = inbox.resume_clarification(row.id, row.revision, payload(2, content="first clarification"))
    row = inbox.route(row.id, row.revision, ControlDecision("clarify", None, "fixture"), current_run=None, active_scope=())
    row = inbox.resume_clarification(row.id, row.revision, payload(3, content="second clarification"))
    inbox.route(row.id, row.revision, ControlDecision("next-turn", None, "fixture"), current_run=None, active_scope=())
    requests = []

    async def chat(**kwargs):
        requests.append(deepcopy(kwargs["messages"]))
        claimed = inbox.entries()[0]
        assert claimed.state == "claimed" and claimed.dispatch_started
        if loop.sessions.source_backend("assistant") is not None:
            assert len(claimed.source_event_ids) == 3
        assert not any(n.stage == "completed" for n in inbox.notices())
        return LLMResponse(content="done")

    provider.chat_with_retry.side_effect = chat
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert len(requests) == 1
    text = str(requests[0])
    assert text.count("complete original") == text.count("first clarification") == text.count("second clarification") == 1
    assert "data:image/png;base64," in text and "_busy_input_id" not in text
    snapshot = loop.sessions.read_session_snapshot("assistant")
    assert snapshot is not None
    rows = [r for r in snapshot.messages if r["role"] == "user"]
    assert [r["content"] for r in rows] == ["原始指令\ncomplete original", "first clarification", "second clarification"]
    assert rows[0]["media"] == [str(image)]
    assert inbox.entries()[0].state == "completed"
    if loop.sessions.source_backend("assistant") is not None:
        assert inbox.entries()[0].source_event_ids == tuple(r["_event_id"] for r in rows)


class GateTool(Tool):
    name = "gate"
    description = "Offline test gate"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, started, release):
        self.started, self.release = started, release

    async def execute(self):
        self.started.set()
        await self.release.wait()
        return "gate result"


async def test_steer_waits_for_tool_and_independent_work_keeps_its_destination(owned):
    loop, inbox, consumer, provider = owned
    started, release = asyncio.Event(), asyncio.Event()
    loop.tools.register(GateTool(started, release))
    requests = []

    async def chat(**kwargs):
        requests.append(deepcopy(kwargs["messages"]))
        if len(requests) == 1:
            return LLMResponse(content="", tool_calls=[ToolCallRequest(id="gate-call", name="gate", arguments={})])
        return LLMResponse(content="original result" if len(requests) == 2 else "independent result")

    provider.chat_with_retry.side_effect = chat
    route(inbox, consumer, 1, scope=("gate",))
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    try:
        route(inbox, consumer, 2, intent="steer", channel="slack", content="steer this task", scope=("gate",))
        route(inbox, consumer, 3, content="independent task")
        consumer.wake()
        assert inbox.entries()[1].state == "next-step" and len(requests) == 1
    finally:
        release.set()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert consumer.last_error is None
    assert len(requests) == 3
    assert str(requests[1]).count("steer this task") == 1
    assert "independent task" not in str(requests[1])
    assert all(row.state == "completed" for row in inbox.entries())
    originals = [loop.run_store.load(row.claimed_run_id) for row in inbox.entries()]
    assert originals[0].delivery_target.chat_id == "chat-1"
    assert originals[1].run_id == originals[0].run_id
    assert originals[2].delivery_target.chat_id == "chat-3"


async def test_claim_is_not_completed_until_actual_terminal_record_returns(owned, monkeypatch):
    loop, inbox, consumer, _provider = owned
    saved, release = asyncio.Event(), asyncio.Event()
    execute = loop._execute_busy_turn

    async def hold(turn):
        result = await execute(turn)
        persisted = loop.run_store.load(result.run_id)
        assert persisted is not None and persisted.status is RunStatus.COMPLETED
        saved.set()
        await release.wait()
        return result

    monkeypatch.setattr(loop, "_execute_busy_turn", hold)
    route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(saved.wait(), 60)
    try:
        assert inbox.entries()[0].state == "claimed"
        assert not any(n.stage == "completed" for n in inbox.notices())
    finally:
        release.set()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert inbox.entries()[0].state == "completed"


@pytest.mark.parametrize("failure", ["provider", "terminal-save"])
async def test_failure_keeps_dispatched_work_uncertain_and_fences_younger_work(owned, monkeypatch, failure):
    loop, inbox, consumer, provider = owned
    if failure == "provider":
        provider.chat_with_retry.side_effect = RuntimeError("offline provider failure")
    else:
        save = loop.run_store.save

        def fail_terminal(record):
            if record.status is not RunStatus.RUNNING:
                raise OSError("terminal record unavailable")
            return save(record)

        monkeypatch.setattr(loop.run_store, "save", fail_terminal)
    route(inbox, consumer, 1)
    route(inbox, consumer, 2)
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    first, second = inbox.entries()
    assert first.state == "uncertain" and first.payload is not None and first.dispatch_started
    assert second.state == "next-turn"
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert provider.chat_with_retry.call_count == 1
    assert not any(n.stage == "completed" for n in inbox.notices())


async def test_automation_has_bounded_fairness_and_settles_after_terminal_delivery(owned, monkeypatch):
    loop, inbox, consumer, provider = owned
    started, release = asyncio.Event(), asyncio.Event()
    saved, settle = asyncio.Event(), asyncio.Event()
    seen = []
    execute = loop._execute_busy_turn

    async def hold_automation(turn):
        result = await execute(turn)
        if turn.automation is not None:
            assert result.record is not None and loop.run_store.load(result.run_id).status is RunStatus.COMPLETED
            saved.set()
            await settle.wait()
        return result

    monkeypatch.setattr(loop, "_execute_busy_turn", hold_automation)

    async def chat(**kwargs):
        latest = next(m["content"] for m in reversed(kwargs["messages"]) if m["role"] == "user")
        seen.append(latest)
        if len(seen) == 1:
            started.set()
            await release.wait()
        return LLMResponse(content="done")

    provider.chat_with_retry.side_effect = chat
    for n in range(7):
        route(inbox, consumer, n)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    pending = asyncio.create_task(loop.submit_cron_turn(automation()))
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(saved.wait(), 60)
    try:
        assert seen[:4] == ["task 0", "task 1", "task 2", "scheduled 1"]
        assert not pending.done()
        assert all(row.state != "claimed" for row in inbox.entries())
    finally:
        settle.set()
    await asyncio.wait_for(pending, 60)
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert seen == ["task 0", "task 1", "task 2", "scheduled 1", "task 3", "task 4", "task 5", "task 6"]
    assert not loop._deferred_automation_turns and loop.bus.inbound_size == 0


async def test_shutdown_terminates_active_and_queued_automation_waiters(owned):
    loop, inbox, consumer, provider = owned
    inbox.limits = inbox.limits.model_copy(update={"max_pending": 1})
    started = asyncio.Event()

    async def blocked(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    provider.chat_with_retry.side_effect = blocked
    active = asyncio.create_task(loop.submit_cron_turn(automation(1)))
    await asyncio.wait_for(started.wait(), 60)
    queued = asyncio.create_task(loop.submit_cron_turn(automation(2)))
    await asyncio.sleep(0)
    with pytest.raises(AutomationTurnError, match="capacity"):
        await loop.submit_cron_turn(automation(3))
    await asyncio.wait_for(loop.aclose(), 60)
    results = await asyncio.wait_for(asyncio.gather(active, queued, return_exceptions=True), 60)
    assert all(isinstance(error, AutomationTurnError) for error in results)
    assert consumer.task is None or consumer.task.done()
    assert inbox.entries() == () and not loop._cron_turns.pending_job_ids_for_session("assistant")


async def test_cancellation_retains_inputs_and_obsolete_steers_for_owning_recovery(owned):
    loop, inbox, consumer, provider = owned
    started = asyncio.Event()

    async def blocked(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    provider.chat_with_retry.side_effect = blocked
    route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    route(inbox, consumer, 2, intent="steer")
    await asyncio.wait_for(loop.aclose(), 60)
    assert [r.state for r in inbox.entries()] == ["uncertain", "next-turn"]
    assert inbox.entries()[0].payload.content == "task 1"
    assert provider.chat_with_retry.call_count == 1 and loop.bus.inbound_size == 0


async def test_owner_registration_is_unique_and_undispatched_restart_claim_requeues(owned):
    loop, inbox, consumer, _provider = owned
    with pytest.raises(RuntimeError, match="owner"):
        loop._own_busy_inbox(inbox, automation_scope=())
    route(inbox, consumer, 1)
    claimed = inbox.claim_next_turn("dead-before-dispatch", current_run=None)
    assert claimed is not None
    consumer.recover()
    assert inbox.entries()[0].state == "next-turn"
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert inbox.entries()[0].state == "completed"


@pytest.mark.parametrize("kind", ["human", "automation"])
async def test_delivery_failure_never_replays_computed_work(owned, monkeypatch, kind):
    loop, inbox, consumer, provider = owned
    original_scope = loop._busy_turn_scope

    @asynccontextmanager
    async def failing_delivery(msg, key):
        async with original_scope(msg, key) as io:
            async def fail(*_args, **_kwargs):
                raise OSError("offline delivery unavailable")
            monkeypatch.setattr(io.delivery, "complete", fail)
            yield io

    monkeypatch.setattr(loop, "_busy_turn_scope", failing_delivery)
    if kind == "automation":
        with pytest.raises(AutomationTurnError, match="delivery"):
            await asyncio.wait_for(loop.submit_cron_turn(automation()), 60)
    else:
        route(inbox, consumer, 1)
        consumer.wake()
        await asyncio.wait_for(consumer.wait_idle(), 60)
        assert inbox.entries()[0].state == "completed"
        record = loop.run_store.load(inbox.entries()[0].claimed_run_id)
        assert record.status is RunStatus.COMPLETED
        assert record.metadata["delivery_error_type"] == "OSError"
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert provider.chat_with_retry.call_count == 1 and loop.bus.inbound_size == 0


async def test_automation_execution_failure_terminates_waiters_and_fences_following_work(owned):
    loop, inbox, consumer, provider = owned
    provider.chat_with_retry.side_effect = RuntimeError("offline automation failure")
    with pytest.raises(AutomationTurnError, match="did not complete"):
        await asyncio.wait_for(loop.submit_cron_turn(automation()), 60)
    route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert inbox.entries()[0].state == "next-turn" and provider.chat_with_retry.call_count == 1
    with pytest.raises(AutomationTurnError, match="recovery"):
        await asyncio.wait_for(loop.submit_cron_turn(automation(2)), 60)


async def test_pending_routing_and_clarification_keep_fifo_without_blocking_ready_unrelated_work(owned):
    _loop, inbox, consumer, provider = owned
    first = inbox.admit(payload(1)).entry
    route(inbox, consumer, 2)
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert provider.chat_with_retry.call_count == 0
    inbox.route(first.id, first.revision, ControlDecision("clarify", None, "ambiguous"),
                current_run=None, active_scope=())
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert [r.state for r in inbox.entries()] == ["awaiting-clarification", "completed"]


async def test_initial_context_and_scope_ceiling_are_preserved(owned):
    loop, inbox, consumer, provider = owned

    async def host_context(_context):
        return RuntimeContextBlock("trusted-context", "host fixture context")

    loop._runtime_context_providers.append(host_context)
    seen = []

    async def chat(**kwargs):
        seen.append(deepcopy(kwargs))
        return LLMResponse(content="done")

    provider.chat_with_retry.side_effect = chat
    route(inbox, consumer, 1, scope=("read_file",))
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert "host fixture context" in str(seen[0]["messages"])
    assert [tool["function"]["name"] for tool in seen[0]["tools"]] == ["read_file"]


async def test_turn_settlement_preserves_a_live_notice_send_until_restart_recovery(owned):
    _loop, inbox, consumer, provider = owned
    row = route(inbox, consumer, 1)
    notice = inbox.claim_notice(next(n.id for n in inbox.notices() if n.ingress_id == row.id))
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert provider.chat_with_retry.call_count == 1 and inbox.entries()[0].state == "completed"
    assert next(n for n in inbox.notices() if n.id == notice.id).state == "sending"
    consumer.recover()
    assert next(n for n in inbox.notices() if n.id == notice.id).state == "uncertain"


async def test_partial_initial_source_failure_retains_full_input_and_never_calls_provider(owned, monkeypatch):
    loop, inbox, consumer, provider = owned
    from nanobot.agent.busy_steering import BusySteering, reconcile_busy_source
    from nanobot.session.execution_owner import acquire_execution

    row = route(inbox, consumer, 1, intent="clarify")
    row = inbox.resume_clarification(row.id, row.revision, payload(2, content="complete clarification"))
    inbox.route(row.id, row.revision, ControlDecision("next-turn", None, "fixture"),
                current_run=None, active_scope=())
    original = BusySteering._message
    calls = 0

    def fail_second(adapter, message):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("offline interrupted input rendering")
        return original(adapter, message)

    monkeypatch.setattr(BusySteering, "_message", fail_second)
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    saved = inbox.entries()[0]
    assert saved.state == "uncertain" and saved.payload.content == "task 1"
    assert saved.clarifications[0].content == "complete clarification"
    assert saved.source_event_ids == () and provider.chat_with_retry.call_count == 0
    backend = loop.sessions.source_backend("assistant")
    if backend is not None:
        async with acquire_execution(backend, "assistant", run_id="source-recovery") as owner:
            reconcile_busy_source(inbox, owner)
        recovered = inbox.entries()[0]
        assert len(recovered.source_event_ids) == 1 and recovered.state == "uncertain"
        assert recovered.claimed_run_id == saved.claimed_run_id
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert provider.chat_with_retry.call_count == 0
