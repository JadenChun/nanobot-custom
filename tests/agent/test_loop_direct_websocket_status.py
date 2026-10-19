import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop, TurnContext, TurnKind
from nanobot.agent.runner import AgentRunResult
from nanobot.agent.turn import TurnContext as RunTurnContext
from nanobot.agent.turn_delivery import TurnDelivery
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.outbound_events import GoalStatusEvent
from nanobot.bus.runtime_events import TurnRunStatusChanged
from nanobot.channels.websocket.runtime import WebSocketChannel
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.session import webui_turns as wth
from nanobot.session.manager import Session
from nanobot.session.webui_turns import WebuiTurnCoordinator, WebuiTurnRoutePolicy
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.webui.metadata import WEBSOCKET_TURN_OWNER_METADATA_KEY
from tests.agent.p6_admitted_fixtures import AgentInputs, AsyncAdmittedFactory
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async


async def _make_loop(
    tmp_path: Path, acquire: AsyncAdmittedFactory, key: str,
    *, native: bool = False,
) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    provider.estimate_prompt_tokens.return_value = (0, "test-counter")
    response = LLMResponse(content="done", tool_calls=[])
    provider.chat_with_retry = AsyncMock(return_value=response)
    provider.chat_stream_with_retry = AsyncMock(return_value=response)

    def build(inputs: AgentInputs) -> AgentLoop:
        return AgentLoop(
            bus=inputs.bus,
            provider=provider,
            workspace=inputs.config.workspace_path,
            model="test-model",
            session_manager=inputs.sessions,
            tool_registry=inputs.tools,
            cron_service=inputs.cron,
            turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools,
            restrict_to_workspace=True,
            enable_legacy_auto_compact=False,
        )

    loop = await acquire(tmp_path, build, keys=() if native else (key,))
    WebuiTurnCoordinator(
        bus=loop.bus,
        sessions=loop.sessions,
        schedule_background=lambda coro: loop.schedule_background(coro),
    ).subscribe()
    loop.turn_delivery_factory.route_policy = WebuiTurnRoutePolicy(loop.sessions)
    loop.tools.get_definitions = MagicMock(return_value=[])
    return loop


@pytest.mark.asyncio
async def test_process_direct_websocket_clears_run_status(tmp_path: Path, admitted_async) -> None:
    loop = await _make_loop(tmp_path, admitted_async, "cron:reminder-1")
    gateway = MagicMock()
    channel = WebSocketChannel(
        {"enabled": True, "allowFrom": ["*"]},
        loop.bus,
        gateway=gateway,
    )

    try:
        response = await loop.process_direct(
            "deliver reminder",
            session_key="cron:reminder-1",
            channel="websocket",
            chat_id="chat-1",
        )

        assert response is not None
        assert response.content == "done"

        events = []
        while loop.bus.outbound_size:
            event = await loop.bus.consume_outbound()
            events.append(event)
            await channel.send(event)

        status_messages = [
            event
            for event in events
            if isinstance(event.event, GoalStatusEvent)
        ]
        statuses = [event.event for event in status_messages]
        assert [status.status for status in statuses] == ["running", "idle"]
        assert isinstance(statuses[0].started_at, float)
        assert statuses[1].started_at is None
        owners = {
            event.metadata[WEBSOCKET_TURN_OWNER_METADATA_KEY]
            for event in status_messages
        }
        assert len(owners) == 1
        assert wth.websocket_turn_wall_started_at("chat-1") is None
        assert "chat-1" not in wth._WEBSOCKET_ACTIVE_TURNS
    finally:
        wth._WEBSOCKET_ACTIVE_TURNS.clear()
        wth._WEBSOCKET_TURN_WALL_STARTED_AT.clear()
        wth._WEBSOCKET_TURN_IDS.clear()
        wth._WEBSOCKET_TURN_OWNERS.clear()


@pytest.mark.asyncio
async def test_process_direct_reuses_existing_session_lock(
    tmp_path: Path, admitted_async, monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = await _make_loop(tmp_path, admitted_async, "api:fixed")
    session_key = "api:fixed"
    lock = loop._session_locks.setdefault(session_key, asyncio.Lock())
    await lock.acquire()
    entered = asyncio.Event()

    async def _process_turn(
        msg: InboundMessage,
        *,
        session_key: str,
        run_context: RunTurnContext,
        session: Session | None,
        runtime: LLMRuntime | None,
        delivery: TurnDelivery | None,
        **_kwargs: object,
    ) -> TurnContext:
        entered.set()
        assert run_context.session_key == session_key
        assert lock.locked()
        return TurnContext(
            msg=msg,
            session_key=session_key,
            turn_id=run_context.run_id,
            runtime=runtime,
            kind=TurnKind.USER,
            delivery=(delivery if delivery is not None else loop.turn_delivery_factory.create(msg, session_key)),
            session=session,
            run_context=run_context,
            run_result=AgentRunResult(final_content=msg.content, messages=[]),
            outbound=OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=msg.content),
        )

    monkeypatch.setattr(loop, "_process_turn", _process_turn)
    task = asyncio.create_task(loop.process_direct("direct", session_key=session_key))
    try:
        await asyncio.sleep(0)
        assert not entered.is_set()

        lock.release()
        response = await asyncio.wait_for(task, timeout=1.0)

        assert entered.is_set()
        assert response is not None
        assert response.content == "direct"
    finally:
        if lock.locked():
            lock.release()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_process_direct_applies_per_run_hooks(tmp_path: Path, admitted_async) -> None:
    from nanobot.agent.hook import AgentHook, AgentRunHookContext

    loop = await _make_loop(tmp_path, admitted_async, "api:per-run-hook")
    events: list[tuple[str, str | None]] = []

    class RecordingHook(AgentHook):
        async def before_run(self, context: AgentRunHookContext) -> None:
            events.append(("before", None))

        async def after_run(self, context: AgentRunHookContext) -> None:
            events.append(("after", context.final_content))

    response = await loop.process_direct(
        "hello",
        session_key="api:per-run-hook",
        hooks=[RecordingHook()],
    )

    assert response is not None
    assert response.content == "done"
    assert events == [("before", None), ("after", "done")]


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("terminal", ["completed", "cancelled"])
async def test_direct_admitted_lifecycle_retains_owner(
    tmp_path: Path,
    admitted_async,
    monkeypatch: pytest.MonkeyPatch,
    native: bool,
    terminal: str,
) -> None:
    key = "websocket:foreground-control"
    loop = await _make_loop(tmp_path, admitted_async, key, native=native)
    channel = WebSocketChannel(
        {"enabled": True, "allowFrom": ["*"]}, loop.bus, gateway=MagicMock(),
    )
    statuses: list[TurnRunStatusChanged] = []
    contexts: list[RunTurnContext] = []
    loop.bus.subscribe(statuses.append, TurnRunStatusChanged)
    process = loop._process_turn

    async def observe(msg: InboundMessage, *, run_context: RunTurnContext, **kwargs):
        contexts.append(run_context)
        return await process(msg, run_context=run_context, **kwargs)

    monkeypatch.setattr(loop, "_process_turn", observe)
    cancellation = asyncio.CancelledError("foreground cancellation")
    if terminal == "cancelled":
        loop.provider.chat_with_retry = AsyncMock(side_effect=cancellation)
        loop.provider.chat_stream_with_retry = AsyncMock(side_effect=cancellation)

    try:
        if terminal == "cancelled":
            with pytest.raises(asyncio.CancelledError) as caught:
                await loop.process_direct(
                    "hello", session_key=key, channel="websocket", chat_id="foreground-control",
                )
            assert caught.value is cancellation
        else:
            response = await loop.process_direct(
                "hello", session_key=key, channel="websocket", chat_id="foreground-control",
            )
            assert response is not None and response.content == "done"

        assert [event.status for event in statuses] == ["running", "idle"]
        owner = statuses[0].context.metadata[WEBSOCKET_TURN_OWNER_METADATA_KEY]
        assert isinstance(owner, str) and owner
        assert statuses[1].context.metadata[WEBSOCKET_TURN_OWNER_METADATA_KEY] == owner
        assert isinstance(statuses[0].started_at, float)
        assert statuses[1].started_at is None
        while loop.bus.outbound_size:
            await channel.send(await loop.bus.consume_outbound())
        assert wth.websocket_turn_wall_started_at("foreground-control") is None
        assert "foreground-control" not in wth._WEBSOCKET_ACTIVE_TURNS
        assert len(contexts) == 1
        context = contexts[0]
        assert context.request is not None
        assert context.request.content == "hello" and context.request.metadata == {}
        assert (context.source_writer is not None) is native
        record = loop.run_store.load(context.run_id)
        assert record is not None and record.status.value == terminal
        loop.sessions.invalidate(key)
        reloaded = loop.sessions.get_or_create(key)
        assert [message["role"] for message in reloaded.messages] == (
            ["user", "assistant"] if terminal == "completed" else ["user"]
        )
        assert reloaded.messages[0]["content"] == "hello"
        assert not loop._source_activity_admissions
    finally:
        wth.clear_websocket_turns("foreground-control")
