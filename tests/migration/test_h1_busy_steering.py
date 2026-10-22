"""H1 steering through the real runner and private session/source authorities."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent import busy_steering as steering_module
from nanobot.agent.busy_control import ControlDecision, ControlSnapshot
from nanobot.agent.busy_steering import BusySteering
from nanobot.agent.context import ContextBuilder
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import DeliveryState, DeliveryTarget, TurnContext
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.busy_inbox import InboxPayload, Origin, SessionBusyInbox
from nanobot.session.execution_owner import ExecutionOwnershipError, acquire_execution
from nanobot.session.manager import JsonlSessionStore, Session, SessionManager
from nanobot.session.runtime_authority import DurableSessionAuthority
from nanobot.session.source import SourceValidationError, new_source_event
from nanobot.session.source_projection import source_nodes
from nanobot.session.source_request import SourceTranscript
from nanobot.session.turn_writer import SourceTurnWriter
from tests.agent.runner_helpers import make_run_spec


def active(generation=1):
    return ControlSnapshot(run_id="run", generation=generation, state="running", queued=0)


def payload(number=1, *, content="完整的指令\nkeep the entire message", media=(), scope=("read",)):
    return InboxPayload(owner_id="owner", session_key="assistant", content=content, media=media,
                        scope=scope, origin=Origin(channel="slack", account_id="account", chat_id="other-chat",
                                                  sender_id="sender", transport_id=str(number), thread_id="thread"))


def admit_steer(inbox, number=1, **kwargs):
    row = inbox.admit(payload(number, **kwargs)).entry
    return inbox.route(row.id, row.revision, ControlDecision("steer", active(), "classified"),
                       current_run=active(), active_scope=("read",))


@pytest.fixture(params=["legacy", "canonical"])
async def sessions(tmp_path, request):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "runtime" / "sessions"
    if request.param == "legacy":
        yield SessionManager(workspace, sessions_root=root)
    else:
        lease = await DurableSessionAuthority(workspace, sessions_root=root).acquire()
        try:
            yield lease.manager
        finally:
            await lease.close()


@asynccontextmanager
async def invocation(sessions, tmp_path, *, scope=("read",), current_run=None):
    inbox = SessionBusyInbox(sessions, owner_id="owner", session_key="assistant")
    sessions.prepare_busy_inbox_metadata("assistant")
    backend = sessions.source_backend("assistant")
    builder = ContextBuilder(tmp_path)
    target = DeliveryTarget(channel="telegram", chat_id="original-chat")

    def steering_for(context):
        return BusySteering(inbox, context=context, observed_run=active(), active_scope=scope,
                            context_builder=builder, current_run=current_run or active)

    if backend is None:
        context = TurnContext(run_id="run", session_key="assistant", delivery=DeliveryState(primary=target))
        yield inbox, context, steering_for(context)
    else:
        async with acquire_execution(backend, "assistant", run_id="run") as owner:
            writer = SourceTurnWriter(backend.source, owner, run_id="run")
            writer.append_message({"role": "user", "content": "initial task"})
            context = TurnContext(run_id="run", session_key="assistant", execution_owner=owner,
                                  source_writer=writer, delivery=DeliveryState(primary=target))
            yield inbox, context, steering_for(context)


def spec_for(provider, context, steering, *, tools=None, iterations=4, **kwargs):
    writer = context.source_writer
    return make_run_spec(provider, model="fixture", tools=tools if tools is not None else ToolRegistry(), max_iterations=iterations,
                         max_tool_result_chars=1000, turn_context=context, source_writer=writer,
                         initial_messages=[{"role": "user", "content": "initial task"}] if writer is None else None,
                         source_transcript=SourceTranscript(source_nodes(writer.stream_events())) if writer else None,
                         busy_steering=steering, **kwargs)


class GateTool(Tool):
    name = "gate"
    description = "Private offline blocked tool"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, started, release):
        self.started, self.release = started, release

    async def execute(self):
        self.started.set()
        await self.release.wait()
        return "known tool result"


@pytest.mark.parametrize("blocked", ["provider", "tool"])
async def test_full_steer_reaches_next_model_once_and_never_mid_effect(sessions, tmp_path, blocked):
    started, release = asyncio.Event(), asyncio.Event()
    provider = MagicMock()
    requests = []
    tools = ToolRegistry()
    if blocked == "tool":
        tools.register(GateTool(started, release))

    async with invocation(sessions, tmp_path) as (inbox, context, steering):
        async def chat(**kwargs):
            requests.append(deepcopy(kwargs["messages"]))
            if len(requests) == 1:
                if blocked == "provider":
                    started.set()
                    await release.wait()
                    return LLMResponse(content="first response")
                return LLMResponse(content="", tool_calls=[ToolCallRequest(id="tool", name="gate", arguments={})])
            if context.source_writer is not None:
                row = inbox.entries()[0]
                assert row.source_event_ids
                assert row.source_event_ids[0] in {event["_event_id"] for event in context.source_writer.stream_events()}
            return LLMResponse(content="final response")

        provider.chat_with_retry = AsyncMock(side_effect=chat)
        spec = spec_for(provider, context, steering, tools=tools)
        policy, destination = context.policy, context.delivery.primary
        task = asyncio.create_task(AgentRunner().run(spec))
        if context.execution_owner is not None:
            context.execution_owner.track_task(task)
        try:
            await asyncio.wait_for(started.wait(), 2)
            original = payload()
            admit_steer(inbox)
            assert len(requests) == 1 and "完整" not in str(requests[0])
            assert inbox.entries()[0].state == "next-step"
            if context.source_writer is not None:
                assert not any(event.get("_busy_input_id") for event in context.source_writer.stream_events())
            release.set()
            result = await asyncio.wait_for(task, 3)
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        assert result.final_content == "final response" and result.had_injections
        assert len(requests) == 2
        assert sum(row.get("content") == original.content for row in requests[1]) == 1
        assert sum(row.get("content") == original.content for row in result.messages) == 1
        assert spec.tools is tools and context.policy is policy and context.delivery.primary is destination
        assert inbox.entries()[0].state == "claimed" and inbox.entries()[0].dispatch_started
        assert "completed" not in [notice.stage for notice in inbox.notices()]
        if context.source_writer is not None:
            events = context.source_writer.stream_events()
            assert sum(event.get("content") == original.content for event in events) == 1
            assert "_busy_input_id" not in str(requests)


async def test_pre_model_drain_keeps_every_clarification_and_image(sessions, tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="))
    async with invocation(sessions, tmp_path) as (inbox, context, steering):
        original = inbox.admit(payload(content="original instruction", media=(str(image),))).entry
        row = inbox.route(original.id, 0, ControlDecision("clarify", active(), "ambiguous"),
                          current_run=active(), active_scope=("read",))
        for number in range(2, 6):
            row = inbox.resume_clarification(row.id, row.revision, payload(number, content=f"clarification {number}"))
            row = inbox.route(row.id, row.revision,
                              ControlDecision("steer" if number == 5 else "clarify", active(), "classified"),
                              current_run=active(), active_scope=("read",))
        requests = []
        async def chat(**kwargs):
            requests.append(deepcopy(kwargs["messages"]))
            return LLMResponse(content="done")
        provider = MagicMock(chat_with_retry=AsyncMock(side_effect=chat))
        await AgentRunner().run(spec_for(provider, context, steering))
        assert len(requests) == 1
        assert all(f"clarification {number}" in str(requests[0]) for number in range(2, 6))
        assert "data:image/png;base64," in str(requests[0])
        assert str(requests[0]).count("original instruction") == 1
        if context.source_writer is not None:
            events = [event for event in context.source_writer.stream_events() if event.get("_busy_input_id")]
            assert len(events) == 5 and len(inbox.entries()[0].source_event_ids) == 5
            assert events[0]["content"] == "original instruction" and events[0]["media"] == [str(image)]
            assert "data:image" not in str(events)


async def test_bounded_drain_preserves_overflow_and_never_claims_twice(sessions, tmp_path):
    async with invocation(sessions, tmp_path) as (inbox, context, steering):
        for number in range(45):
            admit_steer(inbox, number, content=f"message {number}")
        assert len(steering.drain(limit=3)) == 3
        assert len(steering.drain(limit=3)) == 3
        rows = inbox.entries()
        assert sum(row.state == "claimed" for row in rows) == 6
        assert sum(row.state == "next-step" for row in rows) == 39
        assert [row.seq for row in steering.claims] == list(range(1, 7))
        if context.source_writer is not None:
            assert len([event for event in context.source_writer.stream_events() if event.get("_busy_input_id")]) == 6


@pytest.mark.parametrize("response_kind", ["final", "tool"])
async def test_no_steer_claim_when_no_model_round_remains(sessions, tmp_path, response_kind):
    async with invocation(sessions, tmp_path) as (inbox, context, steering):
        async def chat(**_kwargs):
            admit_steer(inbox)
            if response_kind == "tool":
                return LLMResponse(content="", tool_calls=[ToolCallRequest(id="tool", name="unknown", arguments={})])
            return LLMResponse(content="done")
        provider = MagicMock(chat_with_retry=AsyncMock(side_effect=chat))
        await AgentRunner().run(spec_for(provider, context, steering, iterations=1, finalize_on_max_iterations=False))
        assert inbox.entries()[0].state == "next-step" and not inbox.entries()[0].dispatch_started
        assert steering.claims == ()
        inbox.reconcile_orphans(live_run_ids=frozenset())
        assert inbox.entries()[0].state == "next-turn"


async def test_changed_generation_cannot_claim_a_late_steer(sessions, tmp_path):
    now = [active()]
    async with invocation(sessions, tmp_path, current_run=lambda: now[0]) as (inbox, _context, steering):
        admit_steer(inbox)
        now[0] = active(2)
        assert steering.drain(limit=3) == ()
        assert inbox.entries()[0].state == "next-step"


async def test_scope_is_checked_again_before_claim(sessions, tmp_path):
    async with invocation(sessions, tmp_path, scope=("read", "write")) as (inbox, _context, steering):
        admit_steer(inbox, scope=("read",))
        assert steering.drain(limit=3) == ()
        assert inbox.entries()[0].state == "next-turn"


async def test_existing_positional_runner_options_keep_their_meaning():
    provider = MagicMock(chat_with_retry=AsyncMock(return_value=LLMResponse(content="done")))
    base = make_run_spec(provider, model="fixture", tools=ToolRegistry(), max_iterations=1,
                         max_tool_result_chars=1000,
                         initial_messages=[{"role": "user", "content": "initial task"}])

    def continuation():
        return None

    # These arguments use the public constructor order from before H1 steering.
    spec = AgentRunSpec(
        base.initial_messages, base.tools, base.runtime, base.max_iterations,
        base.max_tool_result_chars, None, None, False, None, None, None, None,
        None, False, None, None, "standard", None, None, None, None, None,
        7.0, continuation,
    )
    assert spec.llm_timeout_s == 7.0
    assert spec.continuation_callback is continuation
    assert spec.busy_steering is None
    result = await AgentRunner().run(spec)
    assert result.final_content == "done"
    provider.chat_with_retry.assert_awaited_once()


async def test_steering_requires_its_canonical_source_transcript(sessions, tmp_path):
    async with invocation(sessions, tmp_path) as (inbox, context, steering):
        admit_steer(inbox)
        provider = MagicMock(chat_with_retry=AsyncMock(return_value=LLMResponse(content="done")))
        spec = spec_for(provider, context, steering)
        if context.source_writer is None:
            result = await AgentRunner().run(spec)
            assert result.final_content == "done"
            assert inbox.entries()[0].state == "claimed"
            provider.chat_with_retry.assert_awaited_once()
            return

        before = context.source_writer.stream_events()
        spec.initial_messages = [{"role": "user", "content": "initial task"}]
        spec.source_transcript = None
        with pytest.raises(ValueError, match="source transcript"):
            await AgentRunner().run(spec)
        provider.chat_with_retry.assert_not_awaited()
        row = inbox.entries()[0]
        assert row.state == "next-step" and not row.dispatch_started
        assert row.source_event_ids == () and steering.claims == ()
        assert context.source_writer.stream_events() == before


async def test_durable_steering_cannot_mix_with_lossy_callbacks(sessions, tmp_path):
    async with invocation(sessions, tmp_path) as (_inbox, context, steering):
        provider = MagicMock(chat_with_retry=AsyncMock())
        with pytest.raises(ValueError, match="exclusive"):
            await AgentRunner().run(spec_for(provider, context, steering, injection_callback=AsyncMock(return_value=[])))
        provider.chat_with_retry.assert_not_awaited()


async def test_steering_cannot_be_attached_to_a_different_invocation(sessions, tmp_path):
    async with invocation(sessions, tmp_path) as (inbox, context, steering):
        admit_steer(inbox)
        provider = MagicMock(chat_with_retry=AsyncMock())
        spec = spec_for(provider, context, steering)
        spec.turn_context = TurnContext(run_id="run", session_key="assistant")
        with pytest.raises(ValueError, match="same admitted run context"):
            await AgentRunner().run(spec)
        provider.chat_with_retry.assert_not_awaited()
        assert inbox.entries()[0].state == "next-step"


async def test_same_key_in_another_backend_does_not_grant_steering_ownership(sessions, tmp_path):
    inbox = SessionBusyInbox(sessions, owner_id="owner", session_key="assistant")
    admit_steer(inbox)
    backend = JsonlSessionStore(tmp_path / "other-runtime")
    session = Session("assistant")
    backend.save(session)
    backend.source.upgrade_source_session(session.key, expected_revision=session.revision,
        expected_generation=session.generation, expected_incarnation=session.incarnation)
    async with acquire_execution(backend, "assistant", run_id="run") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="run")
        context = TurnContext(run_id="run", session_key="assistant", execution_owner=owner, source_writer=writer)
        with pytest.raises((ValueError, ExecutionOwnershipError)):
            BusySteering(inbox, context=context, observed_run=active(), active_scope=("read",),
                         context_builder=ContextBuilder(tmp_path), current_run=active)
        assert inbox.entries()[0].state == "next-step"
        assert owner.read_snapshot().events == []


async def test_cancellation_after_binding_is_visible_uncertainty_not_completion(sessions, tmp_path):
    started, release = asyncio.Event(), asyncio.Event()
    async with invocation(sessions, tmp_path) as (inbox, context, steering):
        calls = 0
        async def chat(**_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                admit_steer(inbox)
                return LLMResponse(content="first response")
            started.set()
            await release.wait()
            return LLMResponse(content="must not report success")
        provider = MagicMock(chat_with_retry=AsyncMock(side_effect=chat))
        task = asyncio.create_task(AgentRunner().run(spec_for(provider, context, steering)))
        if context.execution_owner is not None:
            context.execution_owner.track_task(task)
        try:
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        row = inbox.entries()[0]
        assert len(steering.claims) == 1 and row.dispatch_started
        inbox.reconcile_orphans(live_run_ids=frozenset())
        recovered = inbox.entries()[0]
        assert recovered.state == "uncertain" and recovered.source_event_ids == row.source_event_ids
        assert "completed" not in [notice.stage for notice in inbox.notices()]
        assert inbox.claim_next_turn("new", current_run=None) is None


@pytest.mark.parametrize("failure", ["source", "manifest", "forged"])
async def test_source_binding_failure_stops_before_next_dispatch_without_replay(sessions, tmp_path, monkeypatch, failure):
    async with invocation(sessions, tmp_path) as (inbox, context, steering):
        writer = context.source_writer
        if writer is None:
            # Legacy dispatch markers provide conservative recovery without source evidence.
            admit_steer(inbox)
            steering.drain(limit=1)
            inbox.reconcile_orphans(live_run_ids=frozenset())
            assert inbox.entries()[0].state == "uncertain"
            return
        async def chat(**_kwargs):
            admit_steer(inbox)
            if failure == "source":
                monkeypatch.setattr(writer, "append_busy_input", MagicMock(side_effect=OSError("disk")))
            elif failure == "manifest":
                monkeypatch.setattr(inbox, "bind_source", MagicMock(side_effect=OSError("disk")))
            else:
                monkeypatch.setattr(writer, "append_busy_input", MagicMock(return_value={"_event_id": "not-persisted"}))
            return LLMResponse(content="first response")
        provider = MagicMock(chat_with_retry=AsyncMock(side_effect=chat))
        with pytest.raises((OSError, SourceValidationError)):
            await AgentRunner().run(spec_for(provider, context, steering))
        assert provider.chat_with_retry.await_count == 1
        assert inbox.entries()[0].dispatch_started
        events = [event for event in writer.stream_events() if event.get("_busy_input_id")]
        assert len(events) == (1 if failure == "manifest" else 0)
        inbox.reconcile_orphans(live_run_ids=frozenset())
        assert inbox.entries()[0].state == "uncertain"
        steering_module.reconcile_busy_source(inbox, context.execution_owner)
        recovered = inbox.entries()[0]
        assert recovered.state == "uncertain"
        assert recovered.source_event_ids == tuple(event["_event_id"] for event in events)
        assert inbox.claim_next_turn("new", current_run=None) is None


async def test_partial_source_append_is_reconciled_as_evidence_not_replayed(sessions, tmp_path, monkeypatch):
    async with invocation(sessions, tmp_path) as (inbox, context, steering):
        original = inbox.admit(payload(content="original")).entry
        row = inbox.route(original.id, 0, ControlDecision("clarify", active(), "ambiguous"),
                          current_run=active(), active_scope=("read",))
        row = inbox.resume_clarification(row.id, row.revision, payload(2, content="clarification"))
        inbox.route(row.id, row.revision, ControlDecision("steer", active(), "classified"),
                    current_run=active(), active_scope=("read",))
        writer = context.source_writer
        if writer is None:
            assert len(steering.drain(limit=1)) == 2
            inbox.reconcile_orphans(live_run_ids=frozenset())
            assert inbox.entries()[0].state == "uncertain"
            return
        append = writer.append_busy_input
        calls = 0
        def broken(content, media, *, binding_id):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("disk")
            return append(content, media, binding_id=binding_id)
        monkeypatch.setattr(writer, "append_busy_input", broken)
        provider = MagicMock(chat_with_retry=AsyncMock())
        with pytest.raises(OSError):
            await AgentRunner().run(spec_for(provider, context, steering))
        provider.chat_with_retry.assert_not_awaited()
        assert inbox.entries()[0].source_event_ids == ()
        inbox.reconcile_orphans(live_run_ids=frozenset())
        steering_module.reconcile_busy_source(inbox, context.execution_owner)
        recovered = inbox.entries()[0]
        assert recovered.state == "uncertain" and len(recovered.source_event_ids) == 1
        before = inbox.entries()
        steering_module.reconcile_busy_source(inbox, context.execution_owner)
        assert inbox.entries() == before
        assert len([event for event in writer.stream_events() if event.get("_busy_input_id")]) == 1
        assert inbox.claim_next_turn("new", current_run=None) is None


async def test_writer_binding_is_idempotent_only_for_same_owned_stream(sessions, tmp_path):
    async with invocation(sessions, tmp_path) as (_inbox, context, _steering):
        writer = context.source_writer
        if writer is None:
            assert "_busy_input_id" not in new_source_event({"role": "user", "content": "input", "_busy_input_id": "a" * 64}, run_id="run")
            return
        first = writer.append_busy_input("input", (), binding_id="a" * 64)
        again = writer.append_busy_input("input", (), binding_id="a" * 64)
        assert first == again
        assert len([event for event in writer.stream_events() if event.get("_busy_input_id")]) == 1
        with pytest.raises(SourceValidationError):
            writer.append_busy_input("changed", (), binding_id="a" * 64)
        other = SourceTurnWriter(writer.store, writer.owner, run_id="run")
        with pytest.raises(SourceValidationError):
            other.append_busy_input("input", (), binding_id="a" * 64)
        ordinary = writer.append_message({"role": "user", "content": "ordinary", "_busy_input_id": "a" * 64})
        assert "_busy_input_id" not in ordinary

