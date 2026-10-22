"""Known results reach mandatory storage before fallible observers or later effects."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.policy import DelegatedReadOnlyPolicy
from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.execution import execute_tool_calls
from nanobot.agent.turn import ToolOutcome, TurnContext
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.execution_owner import ExecutionOwnershipError, acquire_execution
from nanobot.session.manager import JsonlSessionStore, Session
from tests.agent.runner_helpers import make_run_spec


def calls():
    return [ToolCallRequest(id="same", name=name, arguments={}) for name in ("read_file", "write_file")]


async def execute(tools, *, hook=None, callback=None, **kwargs):
    return await execute_tool_calls(
        tools, calls(), concurrent=True, external_lookup_counts={}, workspace_violation_counts={},
        hook=hook or AgentHook(), context=AgentHookContext(iteration=0, messages=[]),
        tool_result_callback=callback, **kwargs,
    )


def registry(result="raw"):
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.prepare_call.return_value = (None, {}, None)
    tools.execute = AsyncMock(return_value=result)
    return tools


@pytest.mark.parametrize("kind", ["success", "empty", "structured", "terminal", "error", "exception", "preparation", "policy"])
async def test_records_each_known_outcome_once_and_detaches_evidence(kind):
    raw = {"success": "x" * 100000, "empty": None, "structured": {"items": [1]},
           "terminal": ToolOutcome("stop", "policy_blocked"), "error": "Error: broken"}.get(kind, "raw")
    tools = registry(raw)
    if kind == "exception":
        tools.execute.side_effect = RuntimeError("broken")
    if kind == "preparation":
        tools.prepare_call.return_value = (None, {}, "Error: bad arguments")
    records = []
    async def record(value):
        records.append(value)
    await execute(tools, callback=record, tool_policy=DelegatedReadOnlyPolicy(role="reviewer") if kind == "policy" else None)
    assert len(records) == 2
    assert [r.ordinal for r in records] == [0, 1]
    assert [r.tool_call_id for r in records] == ["same", "same"]
    assert records[0].dispatched is (kind not in {"preparation", "policy"})
    if kind in {"terminal", "policy"}:
        assert records[1].dispatched is False
    if kind == "exception":
        assert records[0].raw_result == {"exception_type": "RuntimeError", "message": "broken"}
    if kind == "success":
        assert records[0].raw_result == raw
    if kind == "structured":
        raw["items"].append(2)
        assert records[0].raw_result == {"items": [1]}


@pytest.mark.parametrize("failure", [OSError("storage"), asyncio.CancelledError()])
async def test_recording_failure_stops_batch_without_observer_or_retry(failure):
    tools = registry()
    hook = AgentHook()
    hook.after_execute_tool = AsyncMock()
    hook.on_execute_tool_error = AsyncMock()
    callback = AsyncMock(side_effect=failure)
    with pytest.raises(type(failure)):
        await execute(tools, hook=hook, callback=callback)
    assert tools.execute.await_count == 1
    assert callback.await_count == 1
    hook.after_execute_tool.assert_not_awaited()
    hook.on_execute_tool_error.assert_not_awaited()


async def test_post_hook_failure_cannot_erase_recorded_result():
    tools = registry()
    hook = AgentHook()
    hook.after_execute_tool = AsyncMock(side_effect=RuntimeError("observer"))
    records = []
    async def record(value):
        records.append(value)
    with pytest.raises(RuntimeError, match="observer"):
        await execute(tools, hook=hook, callback=record)
    assert len(records) == 1 and records[0].raw_result == "raw"
    assert tools.execute.await_count == 1


async def test_pre_hook_failure_and_tool_cancellation_have_no_fabricated_result():
    for before in (True, False):
        tools = registry()
        hook = AgentHook()
        if before:
            hook.before_execute_tool = AsyncMock(side_effect=RuntimeError("before"))
        else:
            tools.execute.side_effect = asyncio.CancelledError()
        callback = AsyncMock()
        with pytest.raises(RuntimeError if before else asyncio.CancelledError):
            await execute(tools, hook=hook, callback=callback)
        callback.assert_not_awaited()
        assert tools.execute.await_count == (0 if before else 1)


async def test_real_owner_fences_before_effect_and_missing_callback_before_provider(tmp_path):
    backend = JsonlSessionStore(tmp_path)
    session = Session("test:recording")
    backend.save(session)
    backend.source.upgrade_source_session(session.key, expected_revision=session.revision,
                                         expected_generation=session.generation,
                                         expected_incarnation=session.incarnation)
    tools = registry()
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done"))
    async with acquire_execution(backend, session.key, run_id="run") as owner:
        context = TurnContext(execution_owner=owner)
        with pytest.raises(ValueError, match="recording"):
            await execute(tools, turn_context=context)
        with pytest.raises(ValueError, match="recording"):
            await AgentRunner().run(make_run_spec(provider, initial_messages=[], tools=tools,
                model="test", max_iterations=1, max_tool_result_chars=100, turn_context=context))
        provider.chat_with_retry.assert_not_awaited()
        async def child():
            with pytest.raises(ExecutionOwnershipError):
                await execute(tools, turn_context=context, callback=AsyncMock())
        await asyncio.create_task(child())
        tools.execute.assert_not_awaited()
    with pytest.raises(ExecutionOwnershipError):
        await execute(tools, turn_context=context, callback=AsyncMock())
    tools.execute.assert_not_awaited()


async def test_runner_stops_before_second_provider_call_on_persistence_failure():
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="", tool_calls=calls()))
    tools = registry()
    callback = AsyncMock(side_effect=OSError("storage"))
    with pytest.raises(OSError, match="storage"):
        await AgentRunner().run(make_run_spec(provider, initial_messages=[], tools=tools,
            model="test", max_iterations=2, max_tool_result_chars=100,
            tool_result_callback=callback))
    assert provider.chat_with_retry.await_count == 1
    assert tools.execute.await_count == 1
    assert callback.await_count == 1


async def test_fence_and_recording_order_with_real_owner(tmp_path):
    backend = JsonlSessionStore(tmp_path)
    session = Session("test:order")
    backend.save(session)
    backend.source.upgrade_source_session(session.key, expected_revision=session.revision,
                                         expected_generation=session.generation,
                                         expected_incarnation=session.incarnation)
    events = []
    tools = registry()
    def prepare(*args):
        events.append("prepare")
        return None, {}, None
    async def effect(*args):
        events.append("effect")
        return {"items": [1]}
    tools.prepare_call.side_effect = prepare
    tools.execute.side_effect = effect
    class Hook(AgentHook):
        async def before_execute_tool(self, *args):
            events.append("before")
        async def after_execute_tool(self, context, call, tool, params, result):
            events.append("after")
            result["items"].append(2)
    records = []
    async def record(value):
        events.append("record")
        records.append(value)
    async with acquire_execution(backend, session.key, run_id="run") as owner:
        original_validate = owner.validate_dispatch
        def validate():
            events.append("fence")
            original_validate()
        owner.validate_dispatch = validate
        await execute(tools, hook=Hook(), callback=record,
                      turn_context=TurnContext(execution_owner=owner))
    assert events == ["prepare", "before", "fence", "effect", "record", "after"] * 2
    assert all(r.raw_result == {"items": [1]} for r in records)


async def test_repeated_lookup_rejection_is_recorded_without_dispatch():
    records = []
    async def record(value):
        records.append(value)
    tools = registry()
    counts = {}
    for _ in range(3):
        await execute_tool_calls(tools, [ToolCallRequest(id="r", name="web_fetch", arguments={"url": "https://example.com"})],
            concurrent=False, external_lookup_counts=counts, workspace_violation_counts={},
            hook=AgentHook(), context=AgentHookContext(iteration=0, messages=[]), tool_result_callback=record)
    assert len(records) == 3 and tools.execute.await_count == 2
    assert records[-1].dispatched is False
    assert "repeated external lookup" in records[-1].raw_result


async def test_completed_result_survives_later_cancellation_in_real_source_store(tmp_path):
    from nanobot.session.source import new_source_event

    backend = JsonlSessionStore(tmp_path)
    session = Session("test:durable")
    backend.save(session)
    backend.source.upgrade_source_session(session.key, expected_revision=session.revision,
                                         expected_generation=session.generation,
                                         expected_incarnation=session.incarnation)
    tools = registry()
    tools.execute.side_effect = ["first result", asyncio.CancelledError()]
    async with acquire_execution(backend, session.key, run_id="run") as owner:
        intent = new_source_event({"role": "assistant", "content": "",
                                   "tool_calls": [call.to_openai_tool_call() for call in calls()]}, run_id="run")
        intent["_execution_ids"] = ["execution-first", "execution-second"]
        backend.source.append_source_events(owner.key, owner.read_snapshot().state, [intent], owner=owner)
        async def record(value):
            event = new_source_event({"role": "tool", "tool_call_id": value.tool_call_id,
                                      "name": value.name, "content": value.raw_result}, run_id="run")
            event["_execution_id"] = intent["_execution_ids"][value.ordinal]
            backend.source.append_source_events(owner.key, owner.read_snapshot().state, [event], owner=owner)
        with pytest.raises(asyncio.CancelledError):
            await execute(tools, callback=record, turn_context=TurnContext(execution_owner=owner))
    snapshot = backend.source.load_source_snapshot(session.key)
    assert len(snapshot.events) == 2
    assert snapshot.events[-1]["content"] == "first result"
    assert snapshot.events[-1]["_execution_id"] == "execution-first"
    assert snapshot.header["metadata"]["execution_recovery_required"] is True
    async with acquire_execution(backend, session.key, run_id="recover") as owner:
        assert owner.recovery_required
        with pytest.raises(ExecutionOwnershipError):
            owner.validate_dispatch()
    assert tools.execute.await_count == 2
