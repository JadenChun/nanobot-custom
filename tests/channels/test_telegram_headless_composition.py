"""Headless Telegram-to-canonical-turn composition checks."""

from __future__ import annotations

import asyncio
from collections import deque
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import RunStatus, ToolOutcome
from nanobot.bus.events import OutboundMessage
from nanobot.bus.outbound_events import ProgressEvent
from nanobot.bus.queue import MessageBus
from nanobot.channels.manager import ChannelManager
from nanobot.channels.telegram.runtime import TelegramChannel
from nanobot.config.schema import Config
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse, ToolCallRequest


class _ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse], *, expected_calls: int | None = None) -> None:
        super().__init__(provider_name="synthetic")
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []
        self.generation = GenerationSettings(max_tokens=256, temperature=0.1)
        self.called = asyncio.Event()
        self.finished = asyncio.Event()
        self.expected_calls = expected_calls

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append({
            "messages": deepcopy(messages),
            "tools": tools,
            "model": model,
            "kwargs": kwargs,
        })
        self.called.set()
        if not self.responses:
            raise AssertionError("unexpected extra provider request")
        response = self.responses.popleft()
        if self.expected_calls is not None and len(self.calls) >= self.expected_calls:
            self.finished.set()
        return response

    def get_default_model(self) -> str:
        return "synthetic-model"


class _BlockingProvider(_ScriptedProvider):
    def __init__(self) -> None:
        super().__init__([])
        self.cancelled = asyncio.Event()

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append({
            "messages": deepcopy(messages),
            "tools": tools,
            "model": model,
            "kwargs": kwargs,
        })
        self.called.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _TerminalToolError(Tool):
    def __init__(self) -> None:
        self.executions = 0

    @property
    def name(self) -> str:
        return "composition_tool_error"

    @property
    def description(self) -> str:
        return "Return a structured terminal tool error for the composition test."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **_: Any) -> str:
        return "legacy result"

    async def execute_with_context(self, context, **_: Any) -> ToolOutcome:
        self.executions += 1
        return ToolOutcome(
            content="synthetic tool failure",
            stop_reason=RunStatus.TOOL_ERROR.value,
            policy_metadata={"source": "composition-test"},
        )


class _FakeBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []
        self.sent_media: list[dict[str, Any]] = []
        self.sent = asyncio.Event()
        self.shutdown_calls = 0
        self.commands: list[Any] = []

    async def get_me(self) -> SimpleNamespace:
        return SimpleNamespace(id=999, username="nanobot_test")

    async def set_my_commands(self, commands: list[Any]) -> None:
        self.commands = list(commands)

    async def send_message(self, **kwargs: Any) -> SimpleNamespace:
        self.sent_messages.append(kwargs)
        self.sent.set()
        return SimpleNamespace(message_id=len(self.sent_messages))

    async def send_photo(self, **kwargs: Any) -> SimpleNamespace:
        self.sent_media.append({"kind": "photo", **kwargs})
        self.sent.set()
        return SimpleNamespace(message_id=len(self.sent_media))

    async def send_video(self, **kwargs: Any) -> SimpleNamespace:
        self.sent_media.append({"kind": "video", **kwargs})
        self.sent.set()
        return SimpleNamespace(message_id=len(self.sent_media))

    async def send_voice(self, **kwargs: Any) -> SimpleNamespace:
        self.sent_media.append({"kind": "voice", **kwargs})
        self.sent.set()
        return SimpleNamespace(message_id=len(self.sent_media))

    async def send_audio(self, **kwargs: Any) -> SimpleNamespace:
        self.sent_media.append({"kind": "audio", **kwargs})
        self.sent.set()
        return SimpleNamespace(message_id=len(self.sent_media))

    async def send_document(self, **kwargs: Any) -> SimpleNamespace:
        self.sent_media.append({"kind": "document", **kwargs})
        self.sent.set()
        return SimpleNamespace(message_id=len(self.sent_media))

    async def set_message_reaction(self, **_: Any) -> None:
        return None

    async def send_chat_action(self, **_: Any) -> None:
        return None

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeUpdater:
    def __init__(self, started: asyncio.Event) -> None:
        self.started = started
        self.stop_calls = 0
        self.polling_kwargs: dict[str, Any] = {}

    async def start_polling(self, **kwargs: Any) -> None:
        self.polling_kwargs = kwargs
        self.started.set()

    async def stop(self) -> None:
        self.stop_calls += 1


class _FakeTelegramApplication:
    def __init__(self, started: asyncio.Event) -> None:
        self.bot = _FakeBot()
        self.updater = _FakeUpdater(started)
        self.handlers: list[Any] = []
        self.error_handlers: list[Any] = []
        self.initialize_calls = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.shutdown_calls = 0

    def add_error_handler(self, handler: Any) -> None:
        self.error_handlers.append(handler)

    def add_handler(self, handler: Any) -> None:
        self.handlers.append(handler)

    async def initialize(self) -> None:
        self.initialize_calls += 1

    async def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeApplicationBuilder:
    def __init__(self, started: asyncio.Event) -> None:
        self.started = started
        self.app = _FakeTelegramApplication(started)
        self.token_value: str | None = None
        self.request_value: Any = None
        self.poll_request_value: Any = None

    def token(self, token: str) -> "_FakeApplicationBuilder":
        self.token_value = token
        return self

    def request(self, request: Any) -> "_FakeApplicationBuilder":
        self.request_value = request
        return self

    def get_updates_request(self, request: Any) -> "_FakeApplicationBuilder":
        self.poll_request_value = request
        return self

    def build(self) -> _FakeTelegramApplication:
        return self.app


class _FakeHTTPXRequest:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeApplicationFactory:
    """Small PTB Application.builder() seam for real Telegram lifecycle tests."""

    started: asyncio.Event
    builders: list[_FakeApplicationBuilder]

    @classmethod
    def builder(cls) -> _FakeApplicationBuilder:
        builder = _FakeApplicationBuilder(cls.started)
        cls.builders.append(builder)
        return builder


@pytest.fixture
async def offline_composition(tmp_path, monkeypatch):
    """Build deterministic production objects while denying network/process escape."""
    monkeypatch.setattr("nanobot.migration_guard.require_runtime_ready", lambda: None)
    # The fake Telegram transport never uses optional PTB Webhooks. Production
    # dependency preflight remains covered separately and must stay enabled.
    monkeypatch.setattr(
        "nanobot.optional_features.check_enabled_channel_dependencies",
        lambda _names, _plugins: {},
    )
    lifecycle_started = asyncio.Event()
    _FakeApplicationFactory.started = lifecycle_started
    _FakeApplicationFactory.builders = []
    monkeypatch.setattr(
        "nanobot.channels.telegram.runtime.Application", _FakeApplicationFactory
    )
    monkeypatch.setattr(
        "nanobot.channels.telegram.runtime.HTTPXRequest", _FakeHTTPXRequest
    )
    for target in (
        "socket.socket.connect",
        "socket.socket.connect_ex",
        "socket.create_connection",
        "socket.getaddrinfo",
    ):
        monkeypatch.setattr(target, Mock(side_effect=AssertionError("outbound network access")))
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec",
        AsyncMock(side_effect=AssertionError("process execution")),
    )
    monkeypatch.setattr(
        "asyncio.create_subprocess_shell",
        AsyncMock(side_effect=AssertionError("shell execution")),
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = Config.model_validate({
        "agents": {"defaults": {"workspace": str(workspace), "model": "synthetic-model"}},
        "channels": {
            "taskUpdateMode": "result",
            "websocket": {"enabled": False},
            "telegram": {
                "enabled": True,
                "token": "synthetic-token",
                "allowFrom": ["42"],
                "replyToMessage": True,
                "streaming": False,
            },
        },
    })
    config.bind_source_path(tmp_path / "instance" / "config.json")

    webui_dist = Mock(name="webui_dist")
    gateway_factory = Mock(name="gateway_factory")
    monkeypatch.setattr("nanobot.channels.manager._default_webui_dist", webui_dist)
    monkeypatch.setattr("nanobot.webui.gateway_services.build_gateway_services", gateway_factory)

    bus = MessageBus()
    provider = _ScriptedProvider([])
    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=workspace,
        model="synthetic-model",
        channels_config=config.channels,
        restrict_to_workspace=True,
        tools_config=config.tools,
    )
    manager = ChannelManager(config, bus, webui_static_dist=True)
    await manager.acquire()
    channel = manager.channels["telegram"]
    assert isinstance(channel, TelegramChannel)
    bot = _FakeBot()
    return SimpleNamespace(
        bus=bus,
        bot=bot,
        channel=channel,
        config=config,
        gateway_factory=gateway_factory,
        loop=loop,
        manager=manager,
        provider=provider,
        lifecycle_started=lifecycle_started,
        application_factory=_FakeApplicationFactory,
        webui_dist=webui_dist,
        workspace=workspace,
    )


async def test_missing_telegram_extra_still_fences_headless_activation(tmp_path, monkeypatch) -> None:
    """The offline composition seam must not weaken production dependency admission."""
    monkeypatch.setattr("nanobot.optional_features.extra_installed", lambda *_args: False)
    config = Config.model_validate({
        "channels": {
            "websocket": {"enabled": False},
            "telegram": {"enabled": True, "token": "synthetic-token", "allowFrom": ["42"]},
        },
    })
    config.bind_source_path(tmp_path / "instance" / "config.json")

    manager = ChannelManager(config, MessageBus(), webui_static_dist=False)
    await manager.acquire()

    assert "telegram" not in manager.channels
    assert "preinstall" in manager._channel_errors["telegram"].lower()


def _telegram_update(*, sender_id: int = 42, chat_id: int = 123, message_id: int = 101):
    reply = SimpleNamespace(
        message_id=77,
        text="previous answer",
        caption=None,
        from_user=SimpleNamespace(id=7, username="bob", first_name="Bob"),
    )
    message = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id, type="private", is_forum=True),
        chat_id=chat_id,
        text="inspect the attached image",
        caption=None,
        entities=[],
        caption_entities=[],
        reply_to_message=reply,
        photo=None,
        voice=None,
        audio=None,
        document=None,
        video=None,
        video_note=None,
        animation=None,
        location=None,
        media_group_id=None,
        message_thread_id=9,
        message_id=message_id,
    )
    return SimpleNamespace(
        message=message,
        effective_user=SimpleNamespace(id=sender_id, username="alice", first_name="Alice"),
    )


async def _stop_composition(env, manager_task, loop_task) -> None:
    env.loop.stop()
    await asyncio.wait_for(loop_task, timeout=3)
    await env.manager.stop_all()
    await asyncio.wait_for(manager_task, timeout=3)
    await env.loop.aclose()


async def _start_composition(env):
    """Start the real channel supervisor and manager dispatcher with fake PTB."""
    manager_task = asyncio.create_task(env.manager.start_all())
    await asyncio.wait_for(env.lifecycle_started.wait(), timeout=3)
    await asyncio.wait_for(env.channel._app_ready.wait(), timeout=3)
    assert env.channel._app is not None
    env.bot = env.channel._app.bot
    loop_task = asyncio.create_task(env.loop.run())
    await asyncio.sleep(0)
    return manager_task, loop_task


async def _replace_loop(env, provider, *, tool_registry=None) -> None:
    await env.loop.aclose()
    env.loop = AgentLoop(
        bus=env.bus,
        provider=provider,
        workspace=env.workspace,
        model="synthetic-model",
        channels_config=env.config.channels,
        restrict_to_workspace=True,
        tools_config=env.config.tools,
        tool_registry=tool_registry,
    )


@pytest.mark.asyncio
async def test_authorized_update_reaches_canonical_turn_and_preserves_telegram_route(
    offline_composition, monkeypatch, tmp_path
) -> None:
    env = offline_composition
    media_path = env.workspace / "telegram.png"
    media_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    update = _telegram_update()

    async def download_media(message, *, add_failure_content=False):
        if message is update.message:
            return [str(media_path)], [f"[image: {media_path}]"]
        return [], []

    env.channel._download_message_media = AsyncMock(side_effect=download_media)
    env.provider.responses.extend([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            id="send-1",
            name="message",
            arguments={
                "content": "final answer",
                "channel": "telegram",
                "chat_id": "123",
                "media": [str(media_path)],
            },
        )]),
        LLMResponse(content="final answer", tool_calls=[]),
    ])
    env.provider.expected_calls = 2
    canonical_calls = []
    canonical_results = []
    original_execute_turn = env.loop.execute_turn

    async def observe_execute_turn(*args, **kwargs):
        canonical_calls.append(args[0])
        result = await original_execute_turn(*args, **kwargs)
        canonical_results.append(result)
        return result

    monkeypatch.setattr(env.loop, "execute_turn", observe_execute_turn)
    authorization_decisions = Mock(wraps=env.channel._is_update_allowed)
    sender_decisions = Mock(wraps=env.channel.is_allowed)
    monkeypatch.setattr(env.channel, "_is_update_allowed", authorization_decisions)
    monkeypatch.setattr(env.channel, "is_allowed", sender_decisions)
    manager_task, loop_task = await _start_composition(env)

    try:
        await env.bus.publish_outbound(OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="internal progress",
            event=ProgressEvent(content="internal progress"),
        ))
        await env.channel._on_message(update, None)
        await asyncio.wait_for(env.provider.finished.wait(), timeout=3)
        await asyncio.wait_for(env.bot.sent.wait(), timeout=3)
        await asyncio.wait_for(_wait_for(lambda: len(canonical_results) == 1), timeout=3)

        assert len(canonical_calls) == 1
        request = canonical_calls[0]
        assert request.content.startswith(f"[Reply to @bob: previous answer]\n{update.message.text}")
        assert f"[image: {media_path}]" in request.content
        assert request.session_key == "telegram:123:topic:9"
        assert request.route is not None
        assert request.route.message_id == 101
        assert request.media == (str(media_path),)
        assert len(env.provider.calls) == 2
        # The queued Telegram path deliberately performs ingress admission and
        # dequeue-time revalidation; BaseChannel then performs its sender-only
        # defense-in-depth check before bus publication.
        assert authorization_decisions.call_count == 2
        assert sender_decisions.call_count == 3
        current = [message for message in env.provider.calls[0]["messages"] if message["role"] == "user"][-1]
        assert isinstance(current["content"], list)
        assert current["content"][0]["_meta"]["path"] == str(media_path)

        result = canonical_results[0]
        assert result.status is RunStatus.COMPLETED
        assert result.outbound is None  # matching explicit send suppresses the duplicate final reply
        assert len(result.sent_messages) == 1
        assert result.sent_messages[0].metadata == {
            "message_id": 101,
            "user_id": 42,
            "username": "alice",
            "first_name": "Alice",
            "is_group": False,
            "message_thread_id": 9,
            "is_forum": True,
            "reply_to_message_id": 77,
            "_record_channel_delivery": True,
        }

        assert env.bot.sent_messages == []  # result-only mode drops ordinary ProgressEvent output
        assert len(env.bot.sent_media) == 1
        media_send = env.bot.sent_media[0]
        assert media_send["chat_id"] == 123
        assert media_send["message_thread_id"] == 9
        assert media_send["reply_parameters"].message_id == 101
        assert media_send["caption"] == "final answer"
        assert env.webui_dist.call_count == 0
        assert env.gateway_factory.call_count == 0
    finally:
        await _stop_composition(env, manager_task, loop_task)

    assert env.manager._dispatch_task is not None
    assert env.manager._dispatch_task.done()
    assert not env.manager._channel_tasks
    assert not env.manager._outbound_tasks
    assert not env.manager._outbound_tails


@pytest.mark.asyncio
async def test_queued_update_revalidates_revoked_telegram_policy(
    offline_composition, monkeypatch
) -> None:
    env = offline_composition
    env.channel._running = True
    authorization_decisions = Mock(wraps=env.channel._is_update_allowed)
    sender_decisions = Mock(wraps=env.channel.is_allowed)
    monkeypatch.setattr(env.channel, "_is_update_allowed", authorization_decisions)
    monkeypatch.setattr(env.channel, "is_allowed", sender_decisions)

    try:
        await env.channel._on_message(_telegram_update(), None)
        env.channel.config.allow_from = []
        await asyncio.wait_for(
            _wait_for(lambda: not env.channel._inbound_workers), timeout=3
        )

        assert authorization_decisions.call_count == 2
        assert sender_decisions.call_count == 2
        assert env.bus.inbound_size == 0
        assert env.provider.calls == []
    finally:
        env.channel._running = False
        await env.loop.aclose()


@pytest.mark.asyncio
async def test_unauthorized_update_never_enters_bus_or_transport(
    offline_composition, monkeypatch
) -> None:
    env = offline_composition
    publish_inbound = AsyncMock(wraps=env.bus.publish_inbound)
    env.bus.publish_inbound = publish_inbound
    env.channel._handle_message = AsyncMock()
    authorization_decisions = Mock(wraps=env.channel._is_update_allowed)
    sender_decisions = Mock(wraps=env.channel.is_allowed)
    monkeypatch.setattr(env.channel, "_is_update_allowed", authorization_decisions)
    monkeypatch.setattr(env.channel, "is_allowed", sender_decisions)

    await env.channel._on_message(_telegram_update(sender_id=99), None)

    assert authorization_decisions.call_count == 1
    assert sender_decisions.call_count == 1
    publish_inbound.assert_not_awaited()
    env.channel._handle_message.assert_not_awaited()
    assert env.bus.inbound_size == 0
    assert env.provider.calls == []
    assert env.bot.sent_messages == []
    assert env.bot.sent_media == []

    await env.loop.aclose()


@pytest.mark.asyncio
async def test_cancelled_turn_has_no_delivery_and_structured_tool_error_is_terminal(
    offline_composition, monkeypatch
) -> None:
    env = offline_composition
    env.provider = _BlockingProvider()
    await _replace_loop(env, env.provider)
    manager_task, loop_task = await _start_composition(env)
    try:
        await env.channel._on_message(_telegram_update(), None)
        await asyncio.wait_for(env.provider.called.wait(), timeout=3)
        await _stop_composition(env, manager_task, loop_task)
    finally:
        if not manager_task.done() or not loop_task.done():
            await _stop_composition(env, manager_task, loop_task)

    assert env.provider.cancelled.is_set()
    assert env.bot.sent_messages == []
    assert env.bot.sent_media == []
    assert env.bus.inbound_size == 0
    assert env.bus.outbound_size == 0

    tool_registry = ToolRegistry()
    terminal_error = _TerminalToolError()
    tool_registry.register(terminal_error)
    tool_provider = _ScriptedProvider([LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id="tool-error-1", name="composition_tool_error", arguments={})],
    )], expected_calls=1)
    env.bus = MessageBus()
    env.manager = ChannelManager(env.config, env.bus, webui_static_dist=True)
    await env.manager.acquire()
    env.channel = env.manager.channels["telegram"]
    assert isinstance(env.channel, TelegramChannel)
    await _replace_loop(env, tool_provider, tool_registry=tool_registry)
    canonical_results = []
    original_execute_turn = env.loop.execute_turn

    async def observe_execute_turn(*args, **kwargs):
        result = await original_execute_turn(*args, **kwargs)
        canonical_results.append(result)
        return result

    monkeypatch.setattr(env.loop, "execute_turn", observe_execute_turn)
    manager_task, loop_task = await _start_composition(env)
    try:
        await env.channel._on_message(_telegram_update(), None)
        await asyncio.wait_for(env.bot.sent.wait(), timeout=3)
        await asyncio.wait_for(_wait_for(lambda: len(canonical_results) == 1), timeout=3)
        result = canonical_results[0]
        assert result.status is RunStatus.TOOL_ERROR
        assert result.stop_reason == RunStatus.TOOL_ERROR.value
        assert result.error is None
        assert terminal_error.executions == 1
        assert result.tools_used == []
        assert len(tool_provider.calls) == 1
        assert result.outbound is not None
        assert result.outbound.content.startswith("synthetic tool failure")
        assert len(env.bot.sent_messages) == 1
        assert env.bot.sent_messages[0]["chat_id"] == 123
        assert env.bot.sent_messages[0]["text"].startswith("synthetic tool failure")
    finally:
        await _stop_composition(env, manager_task, loop_task)

    assert env.bus.inbound_size == 0
    assert env.bus.outbound_size == 0


@pytest.mark.asyncio
async def test_headless_manager_lifecycle_does_not_start_websocket_or_leak_tasks(
    offline_composition,
) -> None:
    env = offline_composition
    manager_task = asyncio.create_task(env.manager.start_all())
    await asyncio.wait_for(env.lifecycle_started.wait(), timeout=3)
    await asyncio.wait_for(env.channel._app_ready.wait(), timeout=3)
    assert env.channel._app is not None
    app = env.channel._app
    await env.manager.stop_all()
    await asyncio.wait_for(manager_task, timeout=3)
    await env.loop.aclose()

    assert set(env.manager.channels) == {"telegram"}
    assert env.config.channels.websocket["enabled"] is False
    assert env.webui_dist.call_count == 0
    assert env.gateway_factory.call_count == 0
    assert env.manager._dispatch_task is not None
    assert env.manager._dispatch_task.done()
    assert not env.manager._channel_tasks
    assert not env.manager._outbound_tasks
    assert not env.manager._outbound_tails
    assert not env.channel._inbound_workers
    assert not env.channel._media_group_tasks
    assert not env.channel._typing_tasks
    assert app.initialize_calls == 1
    assert app.start_calls == 1
    assert app.stop_calls == 1
    assert app.shutdown_calls == 1
    assert app.updater.stop_calls == 1
    assert app.bot.shutdown_calls == 1


async def _wait_for(predicate) -> None:
    while not predicate():
        await asyncio.sleep(0)
