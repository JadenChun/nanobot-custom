"""Delayed work must match the source at the final execution-lock boundary."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from nanobot.agent.automation_turns import AutomationTurnError
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.exec_session import ExecCompletion
from nanobot.agent.turn import DeliveryTarget, RunStatus, TurnRequest
from nanobot.bus.events import InboundMessage
from nanobot.process_completion import ProcessCompletionRuntime
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.security.workspace_access import build_workspace_scope
from nanobot.session.source_store import SourceOriginMismatchError
from tests.agent import p6_admitted_fixtures

admitted_async = p6_admitted_fixtures.admitted_async


@pytest.fixture
async def origin_loop(tmp_path, admitted_async):
    provider = MagicMock()
    provider.get_default_model.return_value = "offline"
    provider.generation = GenerationSettings()
    provider.supports_bound_conversation_state.return_value = False
    provider.can_resume_conversation_state.return_value = False
    provider.estimate_prompt_tokens.return_value = (100, "fixture")
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="continued once"))

    def build(inputs):
        return AgentLoop(bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
            model="offline", session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True,
            enable_legacy_auto_compact=False)

    loop = await admitted_async(tmp_path, build)
    loop.sessions.admit_persistent_write("websocket:origin")
    return loop


@pytest.mark.parametrize("change", ["reset", "recreate", "store", "key"])
async def test_stale_origin_at_final_admission_writes_nothing_and_never_calls_provider(
    origin_loop, monkeypatch, change,
):
    import nanobot.agent.loop as loop_module

    loop = origin_loop
    key = "websocket:origin"
    backend, snapshot = loop.sessions.read_existing_canonical_source(key)
    expected = snapshot.state
    if change == "store":
        expected = replace(expected, store_identity="another-store")
    elif change == "key":
        expected = replace(expected, session_key="websocket:another")
    entered, release = asyncio.Event(), asyncio.Event()
    original = loop_module.acquire_execution

    @asynccontextmanager
    async def barrier(*args, **kwargs):
        entered.set()
        await release.wait()
        async with original(*args, **kwargs) as owner:
            yield owner

    monkeypatch.setattr(loop_module, "acquire_execution", barrier)
    task = asyncio.create_task(loop.execute_turn(TurnRequest(content="continue old work",
        session_key=key, route=DeliveryTarget("websocket", "origin"),
        expected_source_state=expected)))
    try:
        if change in {"reset", "recreate"}:
            await asyncio.wait_for(entered.wait(), 2)
            if change == "recreate":
                assert loop.sessions.delete_session(key)
                loop.sessions.admit_persistent_write(key)
            else:
                loop.sessions.reset_session(key)
        before = backend.source.load_source_snapshot(key)
        release.set()
        result = await asyncio.wait_for(task, 2)
        after = backend.source.load_source_snapshot(key)
        assert result.status is RunStatus.ERROR
        assert before == after
        loop.provider.chat_with_retry.assert_not_awaited()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_origin_allows_revision_changes_and_is_captured_from_the_admitted_owner(origin_loop):
    loop = origin_loop
    key = "websocket:origin"
    _backend, snapshot = loop.sessions.read_existing_canonical_source(key)
    loop.sessions.update_session_metadata(key, {"title": "new title"})
    observed = []

    async def context(request):
        observed.append(request.source_state)
        return []

    loop.register_runtime_context_provider(context)
    result = await loop.execute_turn(TurnRequest(content="continue once", session_key=key,
        route=DeliveryTarget("websocket", "origin"), expected_source_state=snapshot.state))
    assert result.status is RunStatus.COMPLETED
    loop.provider.chat_with_retry.assert_awaited_once()
    assert observed and observed[0].incarnation == snapshot.state.incarnation
    assert observed[0].generation == snapshot.state.generation
    assert observed[0].revision > snapshot.state.revision


async def test_missing_source_condition_never_creates_a_replacement(origin_loop):
    loop = origin_loop
    key = "websocket:origin"
    _backend, snapshot = loop.sessions.read_existing_canonical_source(key)
    assert loop.sessions.delete_session(key)
    result = await loop.execute_turn(TurnRequest(content="continue deleted work", session_key=key,
        expected_source_state=snapshot.state))
    assert result.status is RunStatus.ERROR
    assert loop.sessions.read_existing_canonical_source(key) is None
    loop.provider.chat_with_retry.assert_not_awaited()


@pytest.mark.parametrize("failure_at", ["body", "cleanup"])
async def test_admitted_missing_file_preserves_the_exact_error(origin_loop, monkeypatch, failure_at):
    import nanobot.agent.loop as loop_module

    loop = origin_loop
    key = "websocket:origin"
    backend, snapshot = loop.sessions.read_existing_canonical_source(key)
    missing = FileNotFoundError("an admitted operation's file is missing")
    if failure_at == "cleanup":
        original = loop_module.acquire_execution

        @asynccontextmanager
        async def cleanup_failure(*args, **kwargs):
            async with original(*args, **kwargs) as owner:
                yield owner
            raise missing

        monkeypatch.setattr(loop_module, "acquire_execution", cleanup_failure)

    with pytest.raises(FileNotFoundError) as raised:
        async with loop._acquire_source_execution_owner(
            backend, key, run_id="missing-file", workspace_access_continuation=None,
            workspace_full_reference=None, expected_source_state=snapshot.state,
        ):
            if failure_at == "body":
                raise missing

    assert raised.value is missing
    assert loop.sessions.read_existing_canonical_source(key)[1].state.same_origin(snapshot.state)


async def test_process_admitted_missing_file_is_interrupted_without_source_change(
    origin_loop, tmp_path, monkeypatch,
):
    loop = origin_loop
    key = "websocket:origin"
    _backend, snapshot = loop.sessions.read_existing_canonical_source(key)
    missing = FileNotFoundError("provider input file is missing")
    loop.provider.chat_with_retry.side_effect = missing
    completion = Mock(wraps=loop._local_trigger_turns.complete)
    monkeypatch.setattr(loop._local_trigger_turns, "complete", completion)
    runtime = ProcessCompletionRuntime(tmp_path / "process-runtime", loop)
    runtime.start()
    runtime.started("job", "offline command", str(loop.workspace), RequestContext(
        channel="websocket", chat_id="origin", session_key=key, source_state=snapshot.state,
        workspace_scope=build_workspace_scope(loop.workspace, "restricted")))
    try:
        runtime.finished("job", ExecCompletion("private result", 0))
        task = runtime._tasks["job"]
        with pytest.raises(AutomationTurnError) as raised:
            await asyncio.wait_for(task, 2)
        assert raised.value.__cause__ is missing
        assert runtime.get("job").state == "unknown"
        assert runtime.get("job").reason == "continuation_interrupted"
        assert loop.sessions.read_existing_canonical_source(key)[1].state.same_origin(snapshot.state)
        assert not loop._local_trigger_turns._waiters
        completion.assert_called_once()
        assert completion.call_args.kwargs["error"] is missing
        loop.provider.chat_with_retry.assert_awaited_once()
    finally:
        await runtime.aclose()


@pytest.mark.parametrize("cached", [False, True])
async def test_typed_origin_satisfies_existing_session_without_replacement(
    origin_loop, monkeypatch, cached,
):
    loop = origin_loop
    key = "websocket:origin"
    _backend, snapshot = loop.sessions.read_existing_canonical_source(key)
    if cached:
        loop.sessions.get_or_create(key)
    else:
        assert loop.sessions.get_cached(key) is None
    load_session = Mock(wraps=loop.sessions.get_or_create)
    monkeypatch.setattr(loop.sessions, "get_or_create", load_session)
    monkeypatch.setattr(loop.sessions, "admit_persistent_write", Mock(side_effect=AssertionError("create")))

    cached_before = loop.sessions.get_cached(key)
    async with loop._admit_after_bus_cleanup(
        key, require_existing_session=True, expected_source_state=snapshot.state,
    ) as session:
        assert session is None
        assert loop.sessions.get_cached(key) is cached_before
        load_session.assert_not_called()

    await loop._dispatch(InboundMessage(
        channel="websocket", chat_id="origin", sender_id="user", content="continue existing source",
        require_existing_session=True, source_state=snapshot.state,
    ))

    loop.provider.chat_with_retry.assert_awaited_once()
    current = loop.sessions.read_existing_canonical_source(key)[1]
    assert current.state.same_origin(snapshot.state)
    assert sum(event.get("content") == "continue existing source" for event in current.events) == 1
    assert loop.sessions.get_cached(key) is None


@pytest.mark.parametrize("change", ["missing", "stale"])
async def test_typed_origin_with_existing_session_requirement_refuses_changed_source(
    origin_loop, monkeypatch, change,
):
    loop = origin_loop
    key = "websocket:origin"
    _backend, snapshot = loop.sessions.read_existing_canonical_source(key)
    if change == "missing":
        assert loop.sessions.delete_session(key)
    else:
        loop.sessions.reset_session(key)
    before = loop.sessions.read_existing_canonical_source(key)
    monkeypatch.setattr(loop.sessions, "get_or_create", Mock(side_effect=AssertionError("replacement")))
    monkeypatch.setattr(loop.sessions, "admit_persistent_write", Mock(side_effect=AssertionError("create")))
    completion = Mock(wraps=loop._local_trigger_turns.complete)
    monkeypatch.setattr(loop._local_trigger_turns, "complete", completion)

    await loop._dispatch(InboundMessage(
        channel="websocket", chat_id="origin", sender_id="user", content="continue changed source",
        require_existing_session=True, source_state=snapshot.state,
    ))

    assert loop.sessions.read_existing_canonical_source(key) == before
    assert loop.sessions.get_cached(key) is None
    loop.provider.chat_with_retry.assert_not_awaited()
    completion.assert_called_once()
    assert isinstance(completion.call_args.kwargs["error"], SourceOriginMismatchError)


async def test_process_final_admission_rejection_settles_waiter_without_success(
    origin_loop, tmp_path, monkeypatch,
):
    import nanobot.agent.loop as loop_module

    loop = origin_loop
    key = "websocket:origin"
    _backend, snapshot = loop.sessions.read_existing_canonical_source(key)
    runtime = ProcessCompletionRuntime(tmp_path / "process-runtime", loop)
    runtime.start()
    runtime.started("job", "offline command", str(loop.workspace), RequestContext(
        channel="websocket", chat_id="origin", session_key=key, source_state=snapshot.state,
        workspace_scope=build_workspace_scope(loop.workspace, "restricted")))
    completion = Mock(wraps=loop._local_trigger_turns.complete)
    monkeypatch.setattr(loop._local_trigger_turns, "complete", completion)
    entered, release = asyncio.Event(), asyncio.Event()
    original = loop_module.acquire_execution

    @asynccontextmanager
    async def barrier(*args, **kwargs):
        entered.set()
        await release.wait()
        async with original(*args, **kwargs) as owner:
            yield owner

    monkeypatch.setattr(loop_module, "acquire_execution", barrier)
    try:
        runtime.finished("job", ExecCompletion("private result", 0))
        task = runtime._tasks["job"]
        await asyncio.wait_for(entered.wait(), 2)
        loop.sessions.reset_session(key)
        before = loop.sessions.read_existing_canonical_source(key)[1]
        release.set()
        await asyncio.wait_for(task, 2)
        assert runtime.get("job").state == "cancelled"
        assert runtime.get("job").reason == "source_changed"
        assert loop.sessions.read_existing_canonical_source(key)[1] == before
        assert not loop._local_trigger_turns._waiters
        completion.assert_called_once()
        assert isinstance(completion.call_args.kwargs["error"], SourceOriginMismatchError)
        loop.provider.chat_with_retry.assert_not_awaited()
    finally:
        release.set()
        await runtime.aclose()
