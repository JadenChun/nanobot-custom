import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop, _TurnExecutionOptions
from nanobot.agent.tools.sessions import ReadSessionTool
from nanobot.agent.turn import TurnRequest
from nanobot.bus.events import (
    INBOUND_META_RUNTIME_CONTROL,
    RUNTIME_CONTROL_SESSION_DISCARD,
    InboundMessage,
)
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.session.keys import UNIFIED_SESSION_KEY
from tests.agent import p6_admitted_fixtures

admitted_async = p6_admitted_fixtures.admitted_async
# Exercise the real consumer only in the discard test. The shared fixture still
# fences class-level run, processes, network, channels, MCP and cron startup.
_run_agent = AgentLoop.run


def _message(key: str, content: str) -> InboundMessage:
    return InboundMessage(
        channel="websocket",
        sender_id="user",
        chat_id=key.removeprefix("websocket:"),
        content=content,
        session_key_override=key,
        require_existing_session=True,
    )


async def _loop(tmp_path, admitted_async, responses: list[str], **kwargs) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings()
    provider.chat_with_retry = AsyncMock(
        side_effect=[LLMResponse(content=response, usage=None) for response in responses]
    )
    def build(inputs):
        return AgentLoop(
            bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
            model="test-model", session_manager=inputs.sessions,
            tool_registry=inputs.tools, cron_service=inputs.cron,
            turn_delivery_factory=inputs.delivery, enable_legacy_auto_compact=False,
            **kwargs,
        )

    return await admitted_async(tmp_path, build, keys=(UNIFIED_SESSION_KEY,))


@pytest.mark.asyncio
@pytest.mark.parametrize("session_reader", [False, True])
async def test_transient_session_keeps_history_without_persisting_or_durable_tools(
    tmp_path, admitted_async, session_reader,
) -> None:
    loop = await _loop(tmp_path, admitted_async, ["first answer", "second answer"])
    # P4's foreground manifest excludes persisted-session readers. Preserve the
    # policy check for an explicitly supplied read-only tool as a separate case.
    assert not loop.tools.has("read_session")
    if session_reader:
        loop.tools.register(ReadSessionTool(loop.sessions))
    loop.context.memory.write_memory("private durable memory")
    key = "websocket:transient-test"
    loop.sessions.get_or_create_transient(
        key,
        retain=True,
        disabled_tools={"create_goal", "update_goal", "spawn", "cron"},
    )

    await loop._dispatch(_message(key, "first question"))
    await loop._dispatch(_message(key, "second question"))

    calls = loop.provider.chat_with_retry.await_args_list
    assert "private durable memory" not in str(calls[0].kwargs["messages"])
    tool_names = {item["function"]["name"] for item in calls[0].kwargs["tools"]}
    assert "read_file" in tool_names
    assert ("read_session" in tool_names) is session_reader
    assert {"create_goal", "update_goal", "spawn", "cron"}.isdisjoint(tool_names)
    assert "first answer" in str(calls[1].kwargs["messages"])
    session = loop.sessions.get_cached(key)
    assert session is not None
    assert [message["role"] for message in session.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert loop.sessions.read_session_file(key) is None


@pytest.mark.asyncio
async def test_transient_session_stays_outside_unified_session(tmp_path, admitted_async) -> None:
    loop = await _loop(tmp_path, admitted_async, ["private answer"], unified_session=True)
    durable = loop.sessions.get_or_create(UNIFIED_SESSION_KEY)
    durable.add_message("user", "durable question")
    loop.sessions.save(durable)
    key = "websocket:transient-unified"
    transient = loop.sessions.get_or_create_transient(key, retain=True)

    await loop._dispatch(_message(key, "private question"))

    assert [message["content"] for message in transient.messages] == [
        "private question",
        "private answer",
    ]
    assert [message["content"] for message in durable.messages] == ["durable question"]
    assert loop.sessions.read_session_file(key) is None


@pytest.mark.asyncio
async def test_missing_required_session_cannot_fall_back_to_disk(tmp_path, admitted_async) -> None:
    loop = await _loop(tmp_path, admitted_async, [])
    key = "websocket:transient-stale"
    loop.sessions.get_or_create_transient(key, retain=True)
    loop.sessions.invalidate(key)

    message = _message(key, "stale private message")
    with pytest.raises(RuntimeError, match="required session is not active"):
        await loop.execute_turn(
            TurnRequest(content=message.content, session_key=key),
            _options=_TurnExecutionOptions(inbound=message, raise_errors=True),
        )
    # The bus adapter handles the same rejection without crashing its consumer.
    await loop._dispatch(message)

    loop.provider.chat_with_retry.assert_not_awaited()
    assert loop.sessions.get_cached(key) is None
    assert loop.sessions.read_session_file(key) is None


@pytest.mark.asyncio
async def test_session_discard_control_cancels_active_turn(
    tmp_path, monkeypatch, admitted_async,
) -> None:
    provider_started = asyncio.Event()

    async def block_provider(**_kwargs: object) -> LLMResponse:
        provider_started.set()
        await asyncio.Event().wait()
        raise AssertionError("provider blocker unexpectedly released")

    loop = await _loop(tmp_path, admitted_async, [])

    async def wait_for_discard(key: str) -> None:
        while loop.sessions.get_cached(key) is not None or key in loop._discarding_sessions:
            await asyncio.sleep(0)

    loop.provider.chat_with_retry = AsyncMock(side_effect=block_provider)
    terminate_exec_sessions = AsyncMock(return_value=1)
    monkeypatch.setattr(
        loop._exec_session_manager,
        "terminate_by_owner",
        terminate_exec_sessions,
    )
    key = "websocket:transient-cancelled"
    previous_file_state = loop._file_state_store.for_session(key)
    loop.sessions.get_or_create_transient(
        key,
        retain=True,
        disabled_tools={"create_goal", "update_goal", "spawn", "cron"},
    )
    run_task = asyncio.create_task(_run_agent(loop))
    try:
        await loop.bus.publish_inbound(_message(key, "private"))
        await asyncio.wait_for(provider_started.wait(), timeout=2)
        active_task = next(iter(loop._active_tasks[key]))

        await loop.bus.publish_inbound(
            InboundMessage(
                channel="websocket",
                sender_id="webui",
                chat_id="transient-cancelled",
                content="",
                metadata={
                    INBOUND_META_RUNTIME_CONTROL: RUNTIME_CONTROL_SESSION_DISCARD,
                },
                session_key_override=key,
            )
        )

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(active_task, timeout=2)
        await asyncio.wait_for(wait_for_discard(key), timeout=2)
        assert loop.sessions.get_cached(key) is None
        assert loop.sessions.read_session_file(key) is None
        assert loop._file_state_store.for_session(key) is not previous_file_state
        terminate_exec_sessions.assert_awaited_once_with(key)

    finally:
        loop.stop()
        await loop.bus.publish_inbound(_message(key, "wake"))
        await asyncio.wait_for(run_task, timeout=2)
    assert loop._closed
