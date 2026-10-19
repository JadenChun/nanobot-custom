"""Offline controls exercise the real owning loop, persistence and effect barriers."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace

import pytest

from nanobot.agent.automation_turns import AutomationTurnError
from nanobot.agent.busy_control import ControlDecision, ControlSnapshot
from nanobot.agent.tools.base import Tool
from nanobot.agent.turn import RunStatus
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.busy_inbox import InboxConflictError, SessionBusyInbox
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async
from tests.migration.test_h1_busy_dispatch import automation, payload, route
from tests.migration.test_h1_busy_dispatch import owned as owned


def control(inbox, consumer, number, intent, *, observed=None, **kwargs):
    row = inbox.admit(payload(number, **kwargs)).entry
    return inbox.route(row.id, row.revision,
                       ControlDecision(intent, observed or consumer.current_run, "classified"),
                       current_run=consumer.current_run, active_scope=consumer.active_scope)


def entry(inbox, identity):
    return next(row for row in inbox.entries() if row.id == identity)


class CountTool(Tool):
    name = "count"
    description = "Offline effect counter"
    parameters = {"type": "object", "properties": {}}

    def __init__(self):
        self.calls = 0

    async def execute(self):
        self.calls += 1
        return "counted"


async def test_status_is_factual_content_free_and_not_a_task_input(owned):
    loop, inbox, consumer, provider = owned
    started, release = asyncio.Event(), asyncio.Event()
    requests = []

    async def chat(**kwargs):
        requests.append(deepcopy(kwargs["messages"]))
        if len(requests) == 1:
            started.set()
            await release.wait()
        return LLMResponse(content="done")

    provider.chat_with_retry.side_effect = chat
    route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    route(inbox, consumer, 2)
    scheduled = asyncio.create_task(loop.submit_cron_turn(automation()))
    await asyncio.sleep(0)
    try:
        row = control(inbox, consumer, 3, "status", content="private status enquiry")
        result = await consumer.apply_control(row.id, row.revision)
        report = result.control_report
        assert result.state == "completed" and result.payload is None
        assert report.active_run.run_id == consumer.current_run.run_id
        assert report.active_run.state == "running" and report.active_run.queued == 1
        assert report.queued == report.automation_queued == 1
        assert not report.recovery_required and report.target_status is None
        assert "private" not in repr(report) and len(requests) == 1
    finally:
        release.set()
    await asyncio.wait_for(consumer.wait_idle(), 90)
    await asyncio.wait_for(scheduled, 60)
    snapshot = loop.sessions.read_session_snapshot("assistant")
    assert snapshot is not None
    assert "private status enquiry" not in str(snapshot.messages)
    assert "private status enquiry" not in str(requests)
    assert all(row.state == "completed" for row in inbox.entries())


async def test_stale_stop_generation_and_late_apply_cannot_cancel_younger_turn(owned):
    _loop, inbox, consumer, provider = owned
    started = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    calls = 0

    async def chat(**_kwargs):
        nonlocal calls
        index = calls
        calls += 1
        started[index].set()
        await release[index].wait()
        return LLMResponse(content="done")

    provider.chat_with_retry.side_effect = chat
    route(inbox, consumer, 1)
    route(inbox, consumer, 2)
    consumer.wake()
    await asyncio.wait_for(started[0].wait(), 60)
    first = consumer.current_run
    late = control(inbox, consumer, 3, "stop")
    release[0].set()
    await asyncio.wait_for(started[1].wait(), 60)
    second = consumer.current_run
    try:
        assert second.generation > first.generation and second.run_id != first.run_id
        with pytest.raises(InboxConflictError, match="revision"):
            await consumer.apply_control(late.id, late.revision)
        settled = entry(inbox, late.id)
        assert settled.state == "completed" and settled.control_report.target_status is RunStatus.COMPLETED
        for index, observed in enumerate((first, replace(second, generation=second.generation + 1)), 4):
            row = control(inbox, consumer, index, "stop", observed=observed)
            assert row.intent == "status"
            result = await consumer.apply_control(row.id, row.revision)
            assert result.control_report.active_run.run_id == second.run_id
            assert result.control_report.active_run.state == "running"
        assert consumer.current_run == second and not release[1].is_set()
    finally:
        release[1].set()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert consumer.last_error is None and calls == 2


async def test_stop_provider_waits_for_actual_terminal_and_fences_dispatched_input(owned):
    loop, inbox, consumer, provider = owned
    started = asyncio.Event()

    async def blocked(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    provider.chat_with_retry.side_effect = blocked
    initial = route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    observed = consumer.current_run
    queued = route(inbox, consumer, 2)
    row = control(inbox, consumer, 3, "stop", content="stop this task")
    result = await consumer.apply_control(row.id, row.revision, wait_seconds=30)
    await asyncio.wait_for(consumer.wait_idle(), 60)
    record = loop.run_store.load(observed.run_id)
    assert record.status is RunStatus.CANCELLED and record.completed_at is not None
    assert result.state == "uncertain" and result.control_report.target_status is RunStatus.CANCELLED
    assert result.control_report.active_run is None and result.control_report.recovery_required
    assert entry(inbox, initial.id).state == "uncertain"
    assert entry(inbox, queued.id).state == "next-turn"
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert provider.chat_with_retry.call_count == 1 and consumer.status().recovery_required
    snapshot = loop.sessions.read_session_snapshot("assistant")
    assert "stop this task" not in str(snapshot.messages)


async def test_resistant_provider_reports_blocked_and_cannot_start_a_tool_after_stop(owned):
    _loop, inbox, consumer, provider = owned
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    tool = CountTool()
    _loop.tools.register(tool)
    cancellations = 0

    async def chat(**_kwargs):
        nonlocal cancellations
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellations += 1
            cancelled.set()
            await release.wait()
        return LLMResponse(content="late tool", finish_reason="tool_calls",
                           tool_calls=[ToolCallRequest(id="late", name="count", arguments={})])

    provider.chat_with_retry.side_effect = chat
    route(inbox, consumer, 1, scope=("count",))
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    row = control(inbox, consumer, 2, "stop-and-replace", content="complete replacement", scope=("count",))
    try:
        result = await consumer.apply_control(row.id, row.revision, wait_seconds=0.01)
        await asyncio.wait_for(cancelled.wait(), 60)
        assert result.state == "control" and result.control_started
        assert result.control_report.active_run.state == "blocked"
        assert result.control_report.target_status is None and consumer.current_run.state == "blocked"
        again = await consumer.apply_control(result.id, result.revision, wait_seconds=0)
        assert again.revision == result.revision and cancellations == 1
        status = control(inbox, consumer, 3, "status")
        status = await consumer.apply_control(status.id, status.revision)
        assert status.control_report.active_run.state == "blocked" and status.control_report.queued == 1
        assert tool.calls == 0 and provider.chat_with_retry.call_count == 1
    finally:
        release.set()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    result = entry(inbox, row.id)
    assert result.state == "uncertain" and result.payload.content == "complete replacement"
    assert result.control_report.target_status is RunStatus.CANCELLED and result.control_report.recovery_required
    assert result.control_report.active_run is None and tool.calls == 0
    assert provider.chat_with_retry.call_count == 1 and cancellations == 1


async def test_resistant_tool_cannot_continue_the_remaining_tool_batch(owned):
    loop, inbox, consumer, provider = owned
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class ResistantTool(CountTool):
        async def execute(self):
            self.calls += 1
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            return "effect may have happened"

    tool = ResistantTool()
    loop.tools.register(tool)
    provider.chat_with_retry.return_value = LLMResponse(content=None, finish_reason="tool_calls", tool_calls=[
        ToolCallRequest(id="one", name="count", arguments={}),
        ToolCallRequest(id="two", name="count", arguments={}),
    ])
    route(inbox, consumer, 1, scope=("count",))
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    row = control(inbox, consumer, 2, "stop")
    try:
        result = await consumer.apply_control(row.id, row.revision, wait_seconds=0.01)
        await asyncio.wait_for(cancelled.wait(), 60)
        assert result.control_report.active_run.state == "blocked" and tool.calls == 1
    finally:
        release.set()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert tool.calls == 1 and provider.chat_with_retry.call_count == 1
    assert all(row.state == "uncertain" for row in inbox.entries())


async def test_safe_undispatched_stop_replacement_preserves_full_input_and_fifo(owned, tmp_path, monkeypatch):
    loop, inbox, consumer, provider = owned
    entered = asyncio.Event()
    execute = loop._execute_busy_turn
    requests = []
    image = tmp_path / "replacement.png"
    image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="))

    async def admit(turn):
        entered.set()
        return await execute(turn)

    async def chat(**kwargs):
        requests.append(deepcopy(kwargs["messages"]))
        assert not consumer.task.cancelling()
        return LLMResponse(content="replacement done")

    monkeypatch.setattr(loop, "_execute_busy_turn", admit)
    provider.chat_with_retry.side_effect = chat
    async with loop._get_session_lock("assistant"):
        initial = route(inbox, consumer, 1)
        consumer.wake()
        await asyncio.wait_for(entered.wait(), 60)
        observed = consumer.current_run
        older = route(inbox, consumer, 2)
        row = route(inbox, consumer, 3, intent="clarify", content="原始 replacement text",
                    media=(str(image),), scope=("count", "read_file"), channel="discord")
        row = inbox.resume_clarification(row.id, row.revision,
                                        payload(4, content="complete clarification", scope=("count",)))
        row = inbox.route(row.id, row.revision, ControlDecision("stop-and-replace", observed, "classified"),
                          current_run=consumer.current_run, active_scope=consumer.active_scope)
        result = await consumer.apply_control(row.id, row.revision, wait_seconds=30)
        assert entry(inbox, initial.id).state == "cancelled"
        # Managed namespaces do not write unadmitted run records. The durable
        # cancelled inbox claim proves disposition without inventing a terminal.
        assert result.control_report.target_status is None
        assert not result.control_report.recovery_required
        assert result.payload == row.payload and result.clarifications == row.clarifications
        assert result.origin == row.origin and result.effective_scope == frozenset({"count"})
        assert provider.chat_with_retry.call_count == 0
    await asyncio.wait_for(consumer.wait_idle(), 90)
    assert consumer.last_error is None and len(requests) == 2
    assert entry(inbox, older.id).state == entry(inbox, row.id).state == "completed"
    text = str(requests[-1])
    assert text.count("原始 replacement text") == text.count("complete clarification") == 1
    assert "data:image/png;base64," in text
    record = loop.run_store.load(entry(inbox, row.id).claimed_run_id)
    assert record.delivery_target.channel == "discord" and record.delivery_target.chat_id == "chat-3"
    assert record.delivery_target.thread_id == "thread" and record.delivery_target.message_id == "3"


async def test_stop_racing_completed_computation_does_not_replay_effects(owned, monkeypatch):
    loop, inbox, consumer, provider = owned
    computed, release = asyncio.Event(), asyncio.Event()
    original = loop._busy_turn_scope
    first = True

    @asynccontextmanager
    async def hold_delivery(msg, key):
        nonlocal first
        async with original(msg, key) as io:
            yield io
            if first:
                first = False
                computed.set()
                await release.wait()

    monkeypatch.setattr(loop, "_busy_turn_scope", hold_delivery)
    initial = route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(computed.wait(), 60)
    row = control(inbox, consumer, 2, "stop-and-replace")
    try:
        result = await consumer.apply_control(row.id, row.revision, wait_seconds=30)
    finally:
        release.set()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert entry(inbox, initial.id).state == entry(inbox, row.id).state == "completed"
    assert result.control_report.target_status is RunStatus.COMPLETED
    assert not result.control_report.recovery_required
    record = loop.run_store.load(entry(inbox, initial.id).claimed_run_id)
    assert record.status is RunStatus.COMPLETED and record.metadata["delivery_error_type"] == "CancelledError"
    assert provider.chat_with_retry.call_count == 2


async def test_stop_cleanup_retains_the_admission_lock_until_collaborators_quiesce(owned, monkeypatch):
    loop, inbox, consumer, provider = owned
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cleanup_keys = []
    original_cancel = loop.subagents.cancel_by_session
    original_terminate = loop._exec_session_manager.terminate_by_owner

    async def chat(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    async def cancel_children(key):
        assert loop._get_session_lock(key).locked()
        cleanup_keys.append(("children", key))
        cleaning.set()
        await release.wait()
        return await original_cancel(key)

    async def terminate_exec(key):
        assert loop._get_session_lock(key).locked()
        cleanup_keys.append(("exec", key))
        return await original_terminate(key)

    monkeypatch.setattr(loop.subagents, "cancel_by_session", cancel_children)
    monkeypatch.setattr(loop._exec_session_manager, "terminate_by_owner", terminate_exec)
    provider.chat_with_retry.side_effect = chat
    route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    observed = consumer.current_run
    row = control(inbox, consumer, 2, "stop-and-replace")
    try:
        result = await consumer.apply_control(row.id, row.revision, wait_seconds=0.01)
        await asyncio.wait_for(cleaning.wait(), 60)
        assert result.control_report.active_run.state == "blocked"
        assert loop._get_session_lock("assistant").locked()
        assert consumer.current_run.run_id == observed.run_id
        assert provider.chat_with_retry.call_count == 1
    finally:
        release.set()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert cleanup_keys == [("children", "assistant"), ("exec", "assistant")]
    assert not loop._get_session_lock("assistant").locked()
    assert entry(inbox, row.id).state == "uncertain"


async def test_terminal_save_failure_never_claims_a_successful_stop(owned, monkeypatch):
    loop, inbox, consumer, provider = owned
    started = asyncio.Event()

    async def chat(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    provider.chat_with_retry.side_effect = chat
    initial = route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    save = loop.run_store.save

    def fail_terminal(record):
        if record.status is RunStatus.CANCELLED:
            raise OSError("offline terminal persistence failure")
        return save(record)

    row = control(inbox, consumer, 2, "stop-and-replace")
    with monkeypatch.context() as patch:
        patch.setattr(loop.run_store, "save", fail_terminal)
        result = await consumer.apply_control(row.id, row.revision, wait_seconds=30)
        await asyncio.wait_for(consumer.wait_idle(), 60)
    assert result.state == entry(inbox, initial.id).state == "uncertain"
    assert result.control_report.target_status is None and result.control_report.recovery_required
    assert isinstance(consumer.last_error, OSError)
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert provider.chat_with_retry.call_count == 1


async def test_control_settlement_failure_retains_intent_and_cannot_repeat_execution(owned, monkeypatch):
    _loop, inbox, consumer, provider = owned
    started = asyncio.Event()

    async def chat(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    provider.chat_with_retry.side_effect = chat
    route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    row = control(inbox, consumer, 2, "stop-and-replace")
    persist = inbox._persist
    failures = 0

    def fail_once(state, expected, **kwargs):
        nonlocal failures
        if not failures and any(value.id == row.id and value.state == "uncertain" for value in state.entries):
            failures += 1
            raise OSError("offline control settlement failure")
        return persist(state, expected, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(inbox, "_persist", fail_once)
        result = await consumer.apply_control(row.id, row.revision, wait_seconds=30)
        await asyncio.wait_for(consumer.wait_idle(), 60)
    assert failures == 1 and isinstance(consumer.last_error, OSError)
    assert result.state == "uncertain" and result.payload.content == "task 2"
    assert result.control_report.recovery_required
    consumer.recover()
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert provider.chat_with_retry.call_count == 1


async def test_cancelled_control_waiter_does_not_abandon_stopping_ownership(owned):
    _loop, inbox, consumer, provider = owned
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def chat(**_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return LLMResponse(content="late answer")

    provider.chat_with_retry.side_effect = chat
    route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    row = control(inbox, consumer, 2, "stop")
    waiter = asyncio.create_task(consumer.apply_control(row.id, row.revision, wait_seconds=30))
    try:
        await asyncio.wait_for(cancelled.wait(), 60)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        pending = entry(inbox, row.id)
        assert pending.state == "control" and pending.control_started
        assert consumer.current_run.state == "stopping"
    finally:
        release.set()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert entry(inbox, row.id).state == "uncertain"
    assert entry(inbox, row.id).control_report.target_status is RunStatus.CANCELLED


@pytest.mark.parametrize("started_control", [False, True])
async def test_restart_does_not_reissue_controls_or_replay_uncertain_replacement(owned, started_control):
    loop, inbox, consumer, provider = owned
    initial = route(inbox, consumer, 1)
    claimed = inbox.claim_next_turn("lost-run", current_run=None)
    inbox.begin_dispatch(claimed.id, claimed.revision)
    observed = ControlSnapshot("lost-run", 11, "running")
    row = inbox.admit(payload(2, content="retained replacement")).entry
    row = inbox.route(row.id, row.revision, ControlDecision("stop-and-replace", observed, "classified"),
                      current_run=observed, active_scope=())
    if started_control:
        inbox.begin_control(row.id, row.revision, current_run=observed,
                            report=consumer.status())
    restarted = SessionBusyInbox(loop.sessions, owner_id="owner", session_key="assistant")
    restarted.reconcile_orphans(live_run_ids=frozenset())
    consumer.recover()
    assert entry(restarted, initial.id).state == "uncertain"
    saved = entry(restarted, row.id)
    assert saved.state == "uncertain" and saved.payload.content == "retained replacement"
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert provider.chat_with_retry.call_count == 0 and consumer.status().recovery_required


async def test_control_write_failure_cannot_issue_cancellation(owned, monkeypatch):
    _loop, inbox, consumer, provider = owned
    started, release = asyncio.Event(), asyncio.Event()
    cancelled = False

    async def chat(**_kwargs):
        nonlocal cancelled
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        return LLMResponse(content="done")

    provider.chat_with_retry.side_effect = chat
    route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    row = control(inbox, consumer, 2, "stop")
    persist = inbox._persist

    def fail_start(state, expected, **kwargs):
        if any(value.id == row.id and value.control_started for value in state.entries):
            raise OSError("offline control start failure")
        return persist(state, expected, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(inbox, "_persist", fail_start)
            with pytest.raises(OSError, match="control start"):
                await consumer.apply_control(row.id, row.revision)
        assert not cancelled and consumer.current_run.state == "running"
        assert not entry(inbox, row.id).control_started
    finally:
        release.set()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert entry(inbox, row.id).control_report.target_status is RunStatus.COMPLETED


async def test_automation_stop_settles_futures_and_persists_recovery_barrier(owned):
    loop, inbox, consumer, provider = owned
    started = asyncio.Event()

    async def chat(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    provider.chat_with_retry.side_effect = chat
    active = asyncio.create_task(loop.submit_cron_turn(automation(1)))
    await asyncio.wait_for(started.wait(), 60)
    queued = asyncio.create_task(loop.submit_cron_turn(automation(2)))
    await asyncio.sleep(0)
    row = control(inbox, consumer, 1, "stop-and-replace")
    result = await consumer.apply_control(row.id, row.revision, wait_seconds=30)
    await asyncio.wait_for(consumer.wait_idle(), 60)
    errors = await asyncio.wait_for(asyncio.gather(active, queued, return_exceptions=True), 60)
    assert all(isinstance(error, AutomationTurnError) for error in errors)
    assert result.state == "uncertain" and result.control_report.recovery_required
    reopened = SessionBusyInbox(loop.sessions, owner_id="owner", session_key="assistant")
    assert reopened.entries()[0].state == "uncertain"
    route(inbox, consumer, 2)
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert provider.chat_with_retry.call_count == 1


async def test_shutdown_joins_resistant_stop_without_repeated_cancellation(owned):
    loop, inbox, consumer, provider = owned
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancellations = 0

    async def chat(**_kwargs):
        nonlocal cancellations
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellations += 1
            cancelled.set()
            await release.wait()
        return LLMResponse(content="late")

    provider.chat_with_retry.side_effect = chat
    route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    row = control(inbox, consumer, 2, "stop")
    await consumer.apply_control(row.id, row.revision, wait_seconds=0.01)
    await asyncio.wait_for(cancelled.wait(), 60)
    closing = asyncio.create_task(loop.aclose())
    try:
        await asyncio.sleep(0)
        assert not closing.done() and consumer.current_run is not None
        assert cancellations == 1
    finally:
        release.set()
    await asyncio.wait_for(closing, 60)
    assert consumer.task.done() and consumer.current_run is None
    assert entry(inbox, row.id).state == "uncertain" and cancellations == 1


async def test_two_controls_share_one_stop_request_and_settle_their_own_ingress(owned):
    _loop, inbox, consumer, provider = owned
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancellations = 0

    async def chat(**_kwargs):
        nonlocal cancellations
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellations += 1
            cancelled.set()
            await release.wait()
        return LLMResponse(content="late")

    provider.chat_with_retry.side_effect = chat
    route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    first = control(inbox, consumer, 2, "stop")
    second = control(inbox, consumer, 3, "stop-and-replace")
    try:
        results = await asyncio.gather(*(consumer.apply_control(row.id, row.revision, wait_seconds=0.01)
                                         for row in (first, second)))
        await asyncio.wait_for(cancelled.wait(), 60)
        assert cancellations == 1 and all(row.control_started for row in results)
        assert all(row.control_report.active_run.state == "blocked" for row in results)
    finally:
        release.set()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert all(entry(inbox, row.id).state == "uncertain" for row in (first, second))
    assert entry(inbox, second.id).control_report.queued == consumer.status().queued == 0
    assert cancellations == 1 and provider.chat_with_retry.call_count == 1


async def test_unapplied_replacement_queues_after_natural_completion(owned):
    loop, inbox, consumer, provider = owned
    started, release = asyncio.Event(), asyncio.Event()
    requests = []

    async def chat(**kwargs):
        requests.append(deepcopy(kwargs["messages"]))
        if len(requests) == 1:
            started.set()
            await release.wait()
        return LLMResponse(content="done")

    provider.chat_with_retry.side_effect = chat
    initial = route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    row = control(inbox, consumer, 2, "stop-and-replace", channel="discord", content="new independent task")
    release.set()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    replacement = entry(inbox, row.id)
    assert entry(inbox, initial.id).state == replacement.state == "completed"
    assert not replacement.control_started and replacement.control_report.target_status is RunStatus.COMPLETED
    assert not replacement.control_report.recovery_required and len(requests) == 2
    assert "new independent task" not in str(requests[0])
    assert str(requests[1]).count("new independent task") == 1
    record = loop.run_store.load(replacement.claimed_run_id)
    assert record.delivery_target.channel == "discord" and record.delivery_target.chat_id == "chat-2"


@pytest.mark.parametrize("failure", ["children", "exec", "both"])
async def test_cleanup_fault_keeps_replacement_and_actual_terminal_distinct(owned, monkeypatch, failure):
    loop, inbox, consumer, provider = owned
    started = asyncio.Event()
    tool = CountTool()
    loop.tools.register(tool)
    cleanup_calls = []
    cleanup_errors = {
        "children": OSError("child cleanup failure"),
        "exec": OSError("exec cleanup failure"),
    }
    failing = {failure} if failure != "both" else {"children", "exec"}
    original_cancel = loop.subagents.cancel_by_session
    original_terminate = loop._exec_session_manager.terminate_by_owner
    calls = 0

    async def chat(**_kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            return LLMResponse(content=None, finish_reason="tool_calls", tool_calls=[
                ToolCallRequest(id="late", name="count", arguments={}),
            ])
        started.set()
        await asyncio.Event().wait()

    async def cleanup(component, key, operation):
        assert key == "assistant" and loop._get_session_lock(key).locked()
        cleanup_calls.append((component, key))
        if component in failing:
            raise cleanup_errors[component]
        return await operation(key)

    async def cancel_children(key):
        return await cleanup("children", key, original_cancel)

    async def terminate_exec(key):
        return await cleanup("exec", key, original_terminate)

    provider.chat_with_retry.side_effect = chat
    monkeypatch.setattr(loop.subagents, "cancel_by_session", cancel_children)
    monkeypatch.setattr(loop._exec_session_manager, "terminate_by_owner", terminate_exec)
    route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    row = control(inbox, consumer, 2, "stop-and-replace")
    result = await consumer.apply_control(row.id, row.revision, wait_seconds=30)
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert cleanup_calls == [("children", "assistant"), ("exec", "assistant")]
    assert result.state == "uncertain" and result.payload.content == "task 2"
    assert result.control_report.target_status is RunStatus.CANCELLED
    assert result.control_report.recovery_required
    if failure == "both":
        assert isinstance(consumer.last_error, BaseExceptionGroup)
        assert consumer.last_error.exceptions == (cleanup_errors["children"], cleanup_errors["exec"])
    else:
        assert consumer.last_error is cleanup_errors[failure]
    consumer.wake()
    await asyncio.wait_for(consumer.wait_idle(), 60)
    assert provider.chat_with_retry.call_count == 1 and tool.calls == 0


@pytest.mark.parametrize("wait", [-1, 31, float("inf"), float("nan"), True])
async def test_invalid_wait_cannot_commit_or_issue_a_stop(owned, wait):
    _loop, inbox, consumer, provider = owned
    started, release = asyncio.Event(), asyncio.Event()

    async def chat(**_kwargs):
        started.set()
        await release.wait()
        return LLMResponse(content="done")

    provider.chat_with_retry.side_effect = chat
    route(inbox, consumer, 1)
    consumer.wake()
    await asyncio.wait_for(started.wait(), 60)
    observed = consumer.current_run
    row = control(inbox, consumer, 2, "stop")
    try:
        with pytest.raises(ValueError, match="control wait"):
            await consumer.apply_control(row.id, row.revision, wait_seconds=wait)
        assert entry(inbox, row.id) == row and consumer.current_run == observed
        assert not consumer.task.cancelling() and provider.chat_with_retry.call_count == 1
    finally:
        release.set()
    await asyncio.wait_for(consumer.wait_idle(), 60)
