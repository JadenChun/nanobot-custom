"""Real Dream issuance and trusted queued compact settlement boundaries."""

import asyncio
import gc
import socket
import subprocess
import weakref
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory import MemoryStore
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import InboundMessage
from nanobot.bus.runtime_events import TurnCompleted
from nanobot.command.builtin import cmd_stop
from nanobot.command.router import CommandContext
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.runtime_composition import RuntimeState
from nanobot.session.runtime_authority import DurableSessionAuthority
from tests.agent.p6_admitted_fixtures import AdmittedGraphs, AgentInputs
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async


class NamedTool(Tool):
    def __init__(self, name: str):
        self.label = name

    @property
    def name(self):
        return self.label

    @property
    def description(self):
        return "Test identity only."

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return "unused"


@pytest.fixture
async def loop(tmp_path, admitted_async):
    def build(inputs: AgentInputs) -> AgentLoop:
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.generation = GenerationSettings(max_tokens=100)
        provider.supports_tools = True
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done"))
        return AgentLoop(
            bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
            session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True,
            context_window_tokens=32_000, enable_legacy_auto_compact=True,
            session_ttl_minutes=0,
        )
    result = await admitted_async(tmp_path, build, keys=("dream:test", "websocket:test"))
    result.context.memory.write_memory("PRIVATE_DREAM_MEMORY")
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["reconstruct", "equality", "add", "replace", "rename", "foreign-store", "workspace"])
async def test_dream_registry_rejects_unissued_or_changed_identities(loop, mutation, monkeypatch):
    store = loop.context.memory
    tools = store.build_dream_tools()
    assert store.owns_dream_tools(tools, store.workspace)
    target = store
    workspace = store.workspace
    if mutation in {"reconstruct", "equality"}:
        issued_hash = hash(tools)

        class EqualRegistry(ToolRegistry):
            def __hash__(self):
                return issued_hash

            def __eq__(self, other):
                return True

        reconstructed = EqualRegistry() if mutation == "equality" else ToolRegistry()
        for name in tools.tool_names:
            reconstructed.register(tools.get(name))
        tools = reconstructed
    elif mutation == "add":
        tools.register(NamedTool("extra"))
    elif mutation == "replace":
        tools.replace(NamedTool("read_file"))
    elif mutation == "rename":
        tool = tools.get("read_file")
        assert tool is not None
        monkeypatch.setattr(type(tool), "name", property(lambda _: "renamed"))
    elif mutation == "foreign-store":
        target = MemoryStore(store.workspace, sessions_root=loop.sessions.sessions_dir.parent)
    elif mutation == "workspace":
        workspace = store.workspace / "foreign"
    assert not target.owns_dream_tools(tools, workspace)
    assert target.filter_dream_tools(tools, frozenset(), workspace) is None


@pytest.mark.asyncio
async def test_dream_owner_subsets_filter_exclusions_and_weak_lifetime(loop):
    store = loop.context.memory
    issued = store.build_dream_tools()
    read = issued.get("read_file")
    assert read is not None
    derived = store.filter_dream_tools(issued, frozenset({"read_file"}), store.workspace)
    assert derived is not None
    assert store.owns_dream_tools(derived, store.workspace)
    assert "read_file" not in derived.tool_names
    for name in derived.tool_names:
        assert derived.get(name) is issued.get(name)
    derived.register(read)
    assert not store.owns_dream_tools(derived, store.workspace)
    issued.unregister("read_file")
    assert store.owns_dream_tools(issued, store.workspace)
    for name in list(issued.tool_names):
        issued.unregister(name)
    assert store.owns_dream_tools(issued, store.workspace)
    reference = weakref.ref(issued)
    del issued
    gc.collect()
    assert reference() is None
    assert len(store._dream_tool_bindings) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["issued", "removed", "filtered", "empty", "ordinary", "foreign", "canonical", "absent", "deleted", "temporary", "ephemeral-temporary", "corrupt"])
async def test_primary_memory_exception_requires_real_legacy_issued_dream(loop, kind):
    store = loop.context.memory
    tools = store.build_dream_tools()
    key = "dream:test"
    if kind == "removed":
        tools.unregister("write_file")
    elif kind == "filtered":
        tools = store.filter_dream_tools(tools, frozenset({"write_file"}), store.workspace)
    elif kind == "empty":
        for name in list(tools.tool_names):
            tools.unregister(name)
    elif kind == "ordinary":
        tools = ToolRegistry()
    elif kind == "foreign":
        tools = MemoryStore(store.workspace, sessions_root=loop.sessions.sessions_dir.parent).build_dream_tools()
    elif kind == "canonical":
        key = "dream:canonical"
        loop.sessions.admit_persistent_write(key)
        assert loop.sessions.resolve_session_mode(key).mode == "canonical"
    elif kind == "absent":
        key = "dream:absent"
        assert loop.sessions.resolve_session_mode(key).state == "absent"
    elif kind == "deleted":
        assert loop.sessions.delete_session(key)
        assert loop.sessions.resolve_session_mode(key).state == "deleted"
    elif kind in {"temporary", "ephemeral-temporary"}:
        key = "dream:temporary"
        retained = loop.sessions.get_or_create_transient(key, retain=True)
        assert not retained.policy.persist
    elif kind == "corrupt":
        source_path = loop.sessions._get_session_path(key)
        original = source_path.read_bytes()
        source_path.write_text("broken header\n", encoding="utf-8")
        loop.sessions.invalidate(key)
    before_cursor = store.get_last_dream_cursor()
    if kind == "corrupt":
        try:
            with pytest.raises(RuntimeError):
                await loop.process_direct("test", session_key=key, ephemeral=True, tools=tools)
            loop.provider.chat_with_retry.assert_not_awaited()
            assert source_path.read_bytes() == b"broken header\n"
        finally:
            source_path.write_bytes(original)
    else:
        result = await loop.process_direct(
            "test", session_key=key, ephemeral=kind != "temporary", tools=tools,
        )
        assert result is not None and result.content == "done"
        messages = loop.provider.chat_with_retry.await_args.kwargs["messages"]
        system = messages[0]["content"]
        permitted = kind in {"issued", "removed", "filtered", "empty"}
        assert ("PRIVATE_DREAM_MEMORY" in system) is permitted
        assert sum(str(m.get("content", "")).count("PRIVATE_DREAM_MEMORY") for m in messages) == int(permitted)
    assert store.get_last_dream_cursor() == before_cursor
    assert store.read_memory() == "PRIVATE_DREAM_MEMORY"


def message():
    return InboundMessage(
        channel="websocket", sender_id="user", chat_id="route", content="/compact",
        session_key_override="websocket:test",
        metadata={"webui_turn_id": "queued-exact", "nested": {"value": "original"}},
    )


async def stop(loop):
    msg = message()
    return await cmd_stop(CommandContext(msg=msg, session=None, key="websocket:test", raw="/stop", loop=loop))


@pytest.mark.asyncio
@pytest.mark.parametrize("wait", ["pre-start", "lock", "concurrency"])
async def test_queued_compact_settles_once_without_admitted_effects(loop, wait):
    key = "websocket:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "incumbent")
    loop.sessions.save(session)
    path = loop.sessions._get_session_path(key)
    before = path.read_bytes()
    metadata = deepcopy(message().metadata)
    completions = []
    loop.bus.subscribe(completions.append, TurnCompleted)
    lock = loop._get_session_lock(key)
    if wait == "lock":
        await lock.acquire()
    elif wait == "concurrency":
        loop._concurrency_gate = asyncio.Semaphore(0)
    msg = message()
    task = loop._enqueue_bus_dispatch(msg, key)
    msg.metadata["nested"]["value"] = "mutated"
    if wait != "pre-start":
        await asyncio.sleep(0)
    try:
        reply = await stop(loop)
        assert reply.content == "Stopped 1 task(s)."
        assert task.cancelled()
        assert len(completions) == 1
        context = completions[0].context
        assert (context.channel, context.chat_id, context.session_key) == ("websocket", "route", key)
        assert context.metadata == metadata
        assert completions[0].outcome == "completed"
        assert loop.bus.outbound_size == 0
        loop.provider.chat_with_retry.assert_not_awaited()
        assert not loop._pending_queues
        assert not loop._active_tasks
        assert not loop._compact_dispatches
        assert path.read_bytes() == before
        assert loop.sessions.get_or_create(key).last_archived == 0
        assert (await stop(loop)).content == "No active task to stop."
        assert len(completions) == 1
    finally:
        if wait == "lock":
            lock.release()
        elif wait == "concurrency":
            loop._concurrency_gate = None
    result = await loop.process_direct("successor", session_key=key)
    assert result is not None and result.content == "done"


@pytest.mark.asyncio
async def test_callback_during_join_and_concurrent_stop_retain_one_receipt(loop):
    completed = []
    loop.bus.subscribe(completed.append, TurnCompleted)
    callback = asyncio.Event()
    observed = []
    async with loop._get_session_lock("websocket:test"):
        task = loop._enqueue_bus_dispatch(message(), "websocket:test")
        state = loop._compact_dispatches[task]

        def done(task):
            observed.append(loop._compact_dispatches.get(task) is state)
            callback.set()

        task.add_done_callback(done)
        await asyncio.sleep(0)
        replies = await asyncio.gather(stop(loop), stop(loop))
    assert callback.is_set()
    assert observed == [True]
    assert sorted(r.content for r in replies) == ["No active task to stop.", "Stopped 1 task(s)."]
    assert len(completed) == 1
    assert not loop._compact_dispatches and not loop._active_tasks


@pytest.mark.asyncio
async def test_fallback_publisher_failure_is_observed_without_replay(loop, monkeypatch):
    original = loop.bus.publish
    attempts = []

    async def publish(event):
        if isinstance(event, TurnCompleted):
            attempts.append(event)
            raise RuntimeError("completion publisher refused")
        await original(event)

    monkeypatch.setattr(loop.bus, "publish", publish)
    queued = loop._enqueue_bus_dispatch(message(), "websocket:test")
    with pytest.raises(RuntimeError, match="completion publisher refused"):
        await stop(loop)
    assert queued.cancelled()
    assert len(attempts) == 1
    assert not loop._compact_dispatches and not loop._active_tasks
    assert (await stop(loop)).content == "No active task to stop."
    assert len(attempts) == 1
    loop.provider.chat_with_retry.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancellation_during_awaited_settlement_waits_and_preserves_cancellation(loop):
    entered, release = asyncio.Event(), asyncio.Event()
    completions = []

    async def subscriber(event):
        entered.set()
        await release.wait()
        completions.append(event)

    loop.bus.subscribe(subscriber, TurnCompleted)
    queued = loop._enqueue_bus_dispatch(message(), "websocket:test")
    queued.cancel()
    stopper = asyncio.create_task(stop(loop))
    await entered.wait()
    stopper.cancel()
    await asyncio.sleep(0)
    assert not stopper.done()
    assert loop._compact_dispatches
    completion = loop._compact_dispatches[queued].completion
    assert completion is not None
    for _ in range(2):
        stopper.cancel()
        await asyncio.sleep(0)
        assert not stopper.done()
        assert not completion.cancelling() and not completion.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await stopper
    assert len(completions) == 1
    assert not loop._compact_dispatches and not loop._active_tasks


@pytest.mark.asyncio
async def test_stop_attempts_all_receipts_and_original_cleanup_after_publisher_failure(loop, monkeypatch):
    key = "websocket:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "incumbent")
    loop.sessions.save(session)
    path = loop.sessions._get_session_path(key)
    before = path.read_bytes()
    original = loop.bus.publish
    failure = RuntimeError("first completion refused")
    attempts = []

    async def publish(event):
        if isinstance(event, TurnCompleted):
            attempts.append(event)
            if len(attempts) == 1:
                raise failure
        await original(event)

    monkeypatch.setattr(loop.bus, "publish", publish)
    subagents = AsyncMock(wraps=loop.subagents.cancel_by_session)
    executions = AsyncMock(wraps=loop._exec_session_manager.terminate_by_owner)
    monkeypatch.setattr(loop.subagents, "cancel_by_session", subagents)
    monkeypatch.setattr(loop._exec_session_manager, "terminate_by_owner", executions)
    tasks = [loop._enqueue_bus_dispatch(message(), key) for _ in range(2)]
    with pytest.raises(RuntimeError) as error:
        await stop(loop)
    assert error.value is failure
    assert len(attempts) == 2
    assert all(task.cancelled() for task in tasks)
    assert not loop._compact_dispatches and not loop._active_tasks
    subagents.assert_awaited_once_with(key)
    executions.assert_awaited_once_with(key)
    loop.provider.chat_with_retry.assert_not_awaited()
    assert path.read_bytes() == before
    assert not loop._pending_queues and loop.bus.outbound_size == 0
    assert (await stop(loop)).content == "No active task to stop."
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_shutdown_failure_still_settles_batch_and_joins_background_before_resources(tmp_path):
    home, connect, popen = Path.home.__func__, socket.socket.connect, subprocess.Popen
    background = None
    with pytest.MonkeyPatch.context() as patches:
        graphs = AdmittedGraphs(tmp_path, patches)

        def build(inputs: AgentInputs) -> AgentLoop:
            provider = MagicMock()
            provider.get_default_model.return_value = "test-model"
            provider.generation = GenerationSettings(max_tokens=100)
            provider.supports_tools = True
            provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done"))
            return AgentLoop(
                bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
                session_manager=inputs.sessions, tool_registry=inputs.tools,
                cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
                tools_config=inputs.config.tools, restrict_to_workspace=True,
                session_ttl_minutes=0,
            )

        loop = await graphs.acquire(tmp_path, build, keys=("websocket:test",))
        graph, borrowed = graphs.graphs[0]
        assert not borrowed
        handle = graph.require_ready()
        entered, finalized = asyncio.Event(), asyncio.Event()
        order = []

        async def work():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                order.append("background-finally")
                finalized.set()

        loop.schedule_background(work())
        background = next(iter(loop._background_tasks))
        await entered.wait()
        original_close = loop.subagents.close

        async def close_resources():
            await original_close()
            assert finalized.is_set() and background.done()
            order.append("resources")

        resources = AsyncMock(side_effect=close_resources)
        patches.setattr(loop.subagents, "close", resources)
        failure = RuntimeError("shutdown completion refused")
        attempts = []
        original = loop.bus.publish

        async def publish(event):
            if isinstance(event, TurnCompleted):
                attempts.append(event)
                if len(attempts) == 1:
                    raise failure
            await original(event)

        patches.setattr(loop.bus, "publish", publish)
        tasks = [loop._enqueue_bus_dispatch(message(), "websocket:test") for _ in range(2)]
        for task in tasks:
            task.cancel()
        try:
            with pytest.raises(ExceptionGroup) as error:
                await graph.aclose()
            assert error.value.exceptions == (failure,)
            assert len(attempts) == 2
            assert all(task.cancelled() for task in tasks)
            assert background.done() and finalized.is_set()
            assert order == ["background-finally", "resources"]
            resources.assert_awaited_once()
            assert not loop._background_tasks and not loop._compact_dispatches and not loop._active_tasks
            assert loop._closed and graph.state is RuntimeState.CLOSED
            assert graph._owner_loop is asyncio.get_running_loop()
            assert handle.mcp._closing and not handle.mcp.connected_server_names
            assert not handle.bus._pending
            loop.provider.chat_with_retry.assert_not_awaited()
            close_task = loop._close_task
            with pytest.raises(RuntimeError) as repeated:
                await loop.aclose()
            assert repeated.value is failure and loop._close_task is close_task
            await graph.aclose()
            resources.assert_awaited_once()
            assert len(attempts) == 2
        finally:
            # Red-run containment only: never leave the demonstrated orphan alive.
            background.cancel()
            await asyncio.gather(background, *tasks, return_exceptions=True)
            await graph.aclose()
            lease = await DurableSessionAuthority(
                graph.preflight.workspace, sessions_root=graph.preflight.sessions_root,
            ).acquire()
            assert lease.close is not None
            await lease.close()
    assert Path.home.__func__ is home
    assert socket.socket.connect is connect and subprocess.Popen is popen


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_claimed_compact_scope_remains_only_completion_publisher(loop, cancel):
    key = "websocket:test"
    session = loop.sessions.get_or_create(key)
    session.add_message("user", "question")
    session.add_message("assistant", "answer")
    loop.sessions.save(session)
    completed = []
    loop.bus.subscribe(completed.append, TurnCompleted)
    entered = asyncio.Event()
    if cancel:
        async def wait(**kwargs):
            entered.set()
            await asyncio.Event().wait()
        loop.provider.chat_with_retry.side_effect = wait
    task = loop._enqueue_bus_dispatch(message(), key)
    if cancel:
        await entered.wait()
        assert (await stop(loop)).content == "Stopped 1 task(s)."
        assert task.cancelled()
    else:
        await task
    assert len(completed) == 1
    events = [loop.bus.outbound.get_nowait().event for _ in range(loop.bus.outbound_size)]
    assert [event.phase for event in events] == ["started", "cancelled" if cancel else "succeeded"]
    assert events[0].compaction_id == events[1].compaction_id
    assert not loop._compact_dispatches and not loop._active_tasks
