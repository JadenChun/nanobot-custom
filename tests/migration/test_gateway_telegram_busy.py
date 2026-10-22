"""Gateway startup registers authenticated busy input before provider polling."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.channels.telegram.gateway_busy import GatewayTelegramBusy
from nanobot.channels.telegram.runtime import TelegramChannel, TelegramConfig
from nanobot.channels.telegram.tests.test_telegram_channel import (
    _FakeApp,
    _FakeBuilder,
    _FakeHTTPXRequest,
)
from nanobot.config.schema import Config, TelegramBusyConfig
from nanobot.providers.base import GenerationSettings, LLMResponse
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async
from tests.migration.test_h1_telegram_inbox import _receipt_for, _telegram_message


async def _setup(tmp_path, admitted_async, monkeypatch):
    config = Config()
    config.gateway.telegram_busy = TelegramBusyConfig(enabled=True,
        session_key="telegram:-100123", owner_sender_ids=["42"], owner_chat_ids=["-100123"],
        control_provider="openai", control_model="test/control", tool_scope=[], automation_scope=[])
    provider = MagicMock()
    provider.get_default_model.return_value = "test/task"
    provider.generation = GenerationSettings()
    provider.supports_bound_conversation_state.return_value = False
    provider.can_resume_conversation_state.return_value = False
    provider.estimate_prompt_tokens.return_value = (100, "fixture")

    def build(inputs):
        return AgentLoop(bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
            model="test/task", max_iterations=4, session_manager=inputs.sessions,
            tool_registry=inputs.tools, cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True, enable_legacy_auto_compact=False)

    agent = await admitted_async(tmp_path, build, config=config, keys=("telegram:-100123",))
    channel = TelegramChannel(TelegramConfig(allow_from=["42"], allow_chats=["-100123"]), agent.bus)
    control = MagicMock()

    async def classify(**kwargs):
        text = json.loads(kwargs["messages"][-1]["content"])["message"]
        intent = "status" if text == "status please" else "next-turn"
        return LLMResponse(content=json.dumps({"intent": intent, "confidence": 1.0}))

    control.chat = AsyncMock(side_effect=classify)
    gateway = GatewayTelegramBusy(config, agent, channel, control_provider=control)
    polled = []

    def polling():
        assert gateway.edge is not None and channel.owns_busy_inbox(gateway.edge)
        assert channel.authenticated_account_id == "999"
        polled.append(True)

    app = _FakeApp(polling)
    monkeypatch.setattr("nanobot.channels.telegram.runtime.HTTPXRequest", _FakeHTTPXRequest)
    monkeypatch.setattr("nanobot.channels.telegram.runtime.Application",
                        SimpleNamespace(builder=lambda: _FakeBuilder(app)))
    sent = []

    async def receipt(request):
        sent.append(request)
        return _receipt_for(request, 1000 + len(sent) * 10)

    monkeypatch.setattr(channel, "send_routed_with_receipt", receipt)
    await channel._start_app()
    assert polled == [True]
    return gateway, agent, provider, control, sent


async def _until(condition):
    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0.001)


async def test_authenticated_startup_keeps_status_responsive_and_queues_same_owner_work(
    tmp_path, admitted_async, monkeypatch,
):
    gateway, agent, provider, control, sent = await _setup(tmp_path, admitted_async, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            entered.set()
            await release.wait()
        return LLMResponse(content="done")

    provider.chat_with_retry = AsyncMock(side_effect=chat)
    edge = gateway.edge
    user = SimpleNamespace(id=42, username="alice", first_name="Alice")
    try:
        await edge.handle_update(_telegram_message(1, -100123, "first"), user, content="first")
        await asyncio.wait_for(entered.wait(), 5)
        await edge.handle_update(_telegram_message(2, -100123, "status please"), user, content="status please")
        assert edge.consumer.current_run is not None
        assert len(calls) == 1
        await edge.handle_update(_telegram_message(3, -100123, "next task"), user, content="next task")
        assert len(calls) == 1
        assert control.chat.await_count == 2
        assert all(call.kwargs["tools"] == [] and call.kwargs["tool_choice"] == "none"
                   for call in control.chat.await_args_list)
        release.set()
        await _until(lambda: len(calls) == 2 and edge.consumer.current_run is None)
        assert all(request.address.chat_id == "-100123" and request.address.thread_id == "77" for request in sent)
        assert agent.sessions.read_session_metadata("telegram:-100123") is not None
    finally:
        release.set()
        await gateway.aclose()


async def test_direct_stop_cancels_current_turn_without_a_control_model_call(tmp_path, admitted_async, monkeypatch):
    gateway, _agent, provider, control, _sent = await _setup(tmp_path, admitted_async, monkeypatch)
    entered = asyncio.Event()

    async def chat(**_kwargs):
        entered.set()
        await asyncio.Event().wait()

    provider.chat_with_retry = AsyncMock(side_effect=chat)
    user = SimpleNamespace(id=42, username="alice", first_name="Alice")
    edge = gateway.edge
    try:
        await edge.handle_update(_telegram_message(1, -100123, "work"), user, content="work")
        await asyncio.wait_for(entered.wait(), 5)
        await edge.handle_update(_telegram_message(2, -100123, "/stop"), user, content="/stop")
        await _until(lambda: edge.consumer.current_run is None)
        control.chat.assert_not_awaited()
    finally:
        await gateway.aclose()


@pytest.mark.parametrize("field,value", [("control_provider", "auto"), ("control_model", ""),
                                        ("owner_sender_ids", ["*"]), ("owner_chat_ids", ["0"])])
def test_busy_gateway_config_requires_explicit_bounded_owner(field, value):
    data = {"enabled": True, "session_key": "telegram:42", "owner_sender_ids": ["42"],
            "owner_chat_ids": ["42"], "control_provider": "openai", "control_model": "test/control"}
    data[field] = value
    with pytest.raises(ValueError):
        TelegramBusyConfig.model_validate(data)
