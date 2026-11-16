"""Post-append WebUI admission receipts for the live turn owners."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.loop import AgentLoop, TurnContext, TurnKind
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import DeliveryTarget, TurnRequest
from nanobot.agent.turn_delivery import TurnDeliveryFactory
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.outbound_events import MessageAdmittedEvent
from nanobot.bus.queue import MessageBus
from nanobot.bus.runtime_events import UserInputCommitted
from nanobot.command.builtin import USER_SHELL_COMMAND
from nanobot.providers.base import LLMResponse
from nanobot.session.manager import SessionManager, SessionPolicy
from nanobot.session.webui_turns import WebuiTurnCoordinator
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade

engine_config = engine_fixtures.engine_config


@pytest.mark.parametrize("source_mode", [False, True], ids=["legacy", "source"])
@pytest.mark.parametrize("attachment_only", [False, True], ids=["text", "attachment-only"])
async def test_browser_receipt_follows_real_user_append_before_model(
    engine_config, source_mode: bool, attachment_only: bool,
):
    key = "websocket:receipt-source" if source_mode else "websocket:receipt-legacy"
    chat_id = key.split(":", 1)[1]

    class ObservingProvider(ScriptedProvider):
        admissions: list[bool]
        admissions_at_model: list[tuple[bool, ...]]

        async def chat(self, messages, tools=None, model=None, **kwargs):
            self.admissions_at_model.append(tuple(self.admissions))
            return await super().chat(messages, tools, model, **kwargs)

    provider = ObservingProvider([LLMResponse(content="done")])
    provider.admissions = []
    provider.admissions_at_model = []
    content = "" if attachment_only else "hello"
    media: list[str] = []
    if attachment_only:
        image_path = engine_config.workspace_path / "receipt-only.png"
        image_path.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x00\x00\x02\x00\x01"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        media = [str(image_path)]
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    if source_mode:
        source_backend = upgrade(loop, key)
    else:
        source_backend = None
    coordinator = WebuiTurnCoordinator(
        bus=loop.bus,
        sessions=loop.sessions,
        schedule_background=lambda coro: coro.close(),
    )
    handled_commits: list[UserInputCommitted] = []
    original_commit_handler = coordinator._handle_user_input_committed

    async def observe_commit(event: UserInputCommitted) -> None:
        handled_commits.append(event)
        await original_commit_handler(event)

    coordinator._handle_user_input_committed = observe_commit
    disconnect = coordinator.subscribe()

    original_publish_outbound = loop.bus.publish_outbound

    async def observe_outbound(message):
        if isinstance(message.event, MessageAdmittedEvent):
            if source_backend is None:
                session = loop.sessions.get_or_create(key)
                was_committed = any(
                    row.get("role") == "user" and row.get("content") == content
                    for row in session.messages
                )
            else:
                snapshot = source_backend.source.load_source_snapshot(key)
                was_committed = any(
                    row.get("role") == "user" and row.get("content") == content
                    for row in snapshot.events
                )
            provider.admissions.append(was_committed)
        await original_publish_outbound(message)

    loop.bus.publish_outbound = observe_outbound
    committed_events: list[UserInputCommitted] = []
    loop.bus.subscribe(committed_events.append, UserInputCommitted)
    try:
        result = await loop.execute_turn(TurnRequest(
            content=content,
            media=media,
            session_key=key,
            source="gateway",
            route=DeliveryTarget(channel="websocket", chat_id=chat_id),
            metadata={"webui": True, "webui_turn_id": "attempt-1"},
        ))

        assert result.content == "done"
        assert len(committed_events) == 1
        assert committed_events[0].context.channel == "websocket"
        assert committed_events[0].context.metadata.get("webui") is True
        assert committed_events[0].context.metadata["webui_turn_id"] == "attempt-1"
        assert len(handled_commits) == 1
        assert provider.admissions == [True]
        assert provider.admissions_at_model == [(True,)]
        assert loop.bus.outbound_size >= 1
    finally:
        disconnect()
        await loop.aclose()


@pytest.mark.parametrize("source_mode", [False, True], ids=["legacy", "source"])
async def test_shortcut_command_receipt_precedes_assistant_reply(
    engine_config, source_mode: bool,
):
    key = "websocket:command-source" if source_mode else "websocket:command-legacy"
    chat_id = key.split(":", 1)[1]
    provider = ScriptedProvider([])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    source_backend = upgrade(loop, key) if source_mode else None
    coordinator = WebuiTurnCoordinator(
        bus=loop.bus,
        sessions=loop.sessions,
        schedule_background=lambda coro: coro.close(),
    )
    handled_commits: list[UserInputCommitted] = []
    original_commit_handler = coordinator._handle_user_input_committed

    async def observe_commit(event: UserInputCommitted) -> None:
        handled_commits.append(event)
        await original_commit_handler(event)

    coordinator._handle_user_input_committed = observe_commit
    disconnect = coordinator.subscribe()
    observed_rows: list[list[str]] = []
    original_publish_outbound = loop.bus.publish_outbound

    async def observe_outbound(message):
        if isinstance(message.event, MessageAdmittedEvent):
            if source_backend is None:
                rows = loop.sessions.get_or_create(key).messages
            else:
                rows = source_backend.source.load_source_snapshot(key).events
            observed_rows.append([str(row.get("role")) for row in rows])
        await original_publish_outbound(message)

    loop.bus.publish_outbound = observe_outbound
    try:
        result = await loop.execute_turn(TurnRequest(
            content="/help",
            session_key=key,
            source="gateway",
            route=DeliveryTarget(channel="websocket", chat_id=chat_id),
            metadata={"webui": True, "webui_turn_id": "attempt-help"},
        ))

        assert result.stop_reason == "command"
        assert len(handled_commits) == 1
        assert observed_rows == [["user"]]
        assert not provider.calls
    finally:
        disconnect()
        await loop.aclose()


@pytest.mark.parametrize("source_mode", [False, True], ids=["legacy", "source"])
async def test_append_failure_does_not_publish_commit_receipt(
    engine_config, monkeypatch, source_mode: bool,
):
    key = "websocket:receipt-failure"
    provider = ScriptedProvider([LLMResponse(content="unreachable")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    source_backend = upgrade(loop, key) if source_mode else None
    coordinator = WebuiTurnCoordinator(
        bus=loop.bus,
        sessions=loop.sessions,
        schedule_background=lambda coro: coro.close(),
    )
    handled_commits: list[UserInputCommitted] = []
    original_commit_handler = coordinator._handle_user_input_committed

    async def observe_commit(event: UserInputCommitted) -> None:
        handled_commits.append(event)
        await original_commit_handler(event)

    coordinator._handle_user_input_committed = observe_commit
    disconnect = coordinator.subscribe()
    seen: list[MessageAdmittedEvent] = []
    original_publish_outbound = loop.bus.publish_outbound

    async def observe_outbound(message):
        if isinstance(message.event, MessageAdmittedEvent):
            seen.append(message.event)
        await original_publish_outbound(message)

    loop.bus.publish_outbound = observe_outbound
    original_append = loop.sessions.save
    committed_events: list[UserInputCommitted] = []
    loop.bus.subscribe(committed_events.append, UserInputCommitted)

    def fail_save(session):
        if any(row.get("role") == "user" and row.get("content") == "hello" for row in session.messages):
            raise OSError("fixture append failure")
        original_append(session)

    if source_backend is None:
        monkeypatch.setattr(loop.sessions, "save", fail_save)
    else:
        source_type = type(source_backend.source)
        original_source_append = source_type.append_source_events

        def fail_source_append(store, source_key, expected, events, **kwargs):
            if any(
                event.get("role") == "user" and event.get("content") == "hello"
                for event in events
            ):
                raise OSError("fixture append failure")
            return original_source_append(store, source_key, expected, events, **kwargs)

        monkeypatch.setattr(source_type, "append_source_events", fail_source_append)
    try:
        result = await loop.execute_turn(TurnRequest(
                content="hello",
                session_key=key,
                source="gateway",
                route=DeliveryTarget(channel="websocket", chat_id="receipt-failure"),
                metadata={"webui": True, "webui_turn_id": "attempt-fail"},
        ))
        assert result.error == "fixture append failure"
        assert seen == []
        assert committed_events == []
        assert not handled_commits
        assert not provider.calls
    finally:
        disconnect()
        await loop.aclose()


async def test_source_shell_receipt_precedes_shell_dispatch(engine_config):
    key = "websocket:source-shell-receipt"
    chat_id = key.split(":", 1)[1]
    provider = ScriptedProvider([])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    source_backend = upgrade(loop, key)
    coordinator = WebuiTurnCoordinator(
        bus=loop.bus,
        sessions=loop.sessions,
        schedule_background=lambda coro: coro.close(),
    )
    disconnect = coordinator.subscribe()
    observed_rows: list[list[str]] = []
    original_publish_outbound = loop.bus.publish_outbound

    async def observe_outbound(message):
        if isinstance(message.event, MessageAdmittedEvent):
            rows = source_backend.source.load_source_snapshot(key).events
            observed_rows.append([str(row.get("role")) for row in rows])
        await original_publish_outbound(message)

    loop.bus.publish_outbound = observe_outbound

    async def shell_dispatch(_context):
        assert observed_rows == [["user"]]
        return OutboundMessage(channel="websocket", chat_id=chat_id, content="shell complete")

    loop.commands.dispatch = AsyncMock(side_effect=shell_dispatch)
    try:
        result = await loop.execute_turn(TurnRequest(
            content=f"{USER_SHELL_COMMAND} echo harmless",
            session_key=key,
            source="gateway",
            route=DeliveryTarget(channel="websocket", chat_id=chat_id),
            metadata={"webui": True, "webui_turn_id": "attempt-shell"},
        ))

        assert result.stop_reason == "command"
        assert observed_rows == [["user"]]
        loop.commands.dispatch.assert_awaited_once()
        assert [row["role"] for row in source_backend.source.load_source_snapshot(key).events] == [
            "user", "assistant",
        ]
        assert not provider.calls
    finally:
        disconnect()
        await loop.aclose()


async def test_temporary_session_receives_receipt_after_lifetime_append(engine_config):
    key = "websocket:temporary-receipt"
    chat_id = key.split(":", 1)[1]
    provider = ScriptedProvider([LLMResponse(content="done")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    session = loop.sessions.get_or_create(key)
    session.policy = SessionPolicy(persist=False)
    coordinator = WebuiTurnCoordinator(
        bus=loop.bus,
        sessions=loop.sessions,
        schedule_background=lambda coro: coro.close(),
    )
    disconnect = coordinator.subscribe()
    admission_state: list[bool] = []
    original_publish_outbound = loop.bus.publish_outbound

    async def observe_outbound(message):
        if isinstance(message.event, MessageAdmittedEvent):
            admission_state.append(any(
                row.get("role") == "user" and row.get("content") == "temporary input"
                for row in session.messages
            ))
        await original_publish_outbound(message)

    loop.bus.publish_outbound = observe_outbound
    try:
        result = await loop.execute_turn(TurnRequest(
            content="temporary input",
            session_key=key,
            source="gateway",
            route=DeliveryTarget(channel="websocket", chat_id=chat_id),
            metadata={"webui": True, "webui_turn_id": "attempt-temporary"},
        ))

        assert result.content == "done"
        assert admission_state == [True]
        assert session.policy.persist is False
    finally:
        disconnect()
        await loop.aclose()


@pytest.mark.parametrize(
    ("source_mode", "command", "expect_receipt"),
    [
        (False, "/new", False),
        (False, "/compact", False),
        (True, "/new", False),
        (True, "/compact", True),
    ],
    ids=["legacy-new", "legacy-compact", "source-new", "source-compact"],
)
async def test_new_and_compact_ack_only_an_input_their_owner_appends(
    engine_config, source_mode: bool, command: str, expect_receipt: bool,
):
    key = "websocket:new-compact-source" if source_mode else "websocket:new-compact-legacy"
    chat_id = key.split(":", 1)[1]
    provider = ScriptedProvider([])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    source_backend = upgrade(loop, key) if source_mode else None
    coordinator = WebuiTurnCoordinator(
        bus=loop.bus,
        sessions=loop.sessions,
        schedule_background=lambda coro: coro.close(),
    )
    disconnect = coordinator.subscribe()
    receipts: list[MessageAdmittedEvent] = []
    original_publish_outbound = loop.bus.publish_outbound

    async def observe_outbound(message):
        if isinstance(message.event, MessageAdmittedEvent):
            receipts.append(message.event)
        await original_publish_outbound(message)

    loop.bus.publish_outbound = observe_outbound
    if not source_mode:
        loop.commands.dispatch = AsyncMock(return_value=OutboundMessage(
            channel="websocket", chat_id=chat_id, content="command result",
        ))
    try:
        result = await loop.execute_turn(TurnRequest(
            content=command,
            session_key=key,
            source="gateway",
            route=DeliveryTarget(channel="websocket", chat_id=chat_id),
            metadata={"webui": True, "webui_turn_id": "attempt-command"},
        ))

        assert result.stop_reason == "command"
        assert len(receipts) == int(expect_receipt)
        if source_mode:
            rows = source_backend.source.load_source_snapshot(key).events
            assert [row["role"] for row in rows] == (
                ["user", "assistant"] if command == "/compact" else []
            )
        else:
            rows = loop.sessions.get_or_create(key).messages
            assert not any(row.get("role") == "user" for row in rows)
        assert not provider.calls
    finally:
        disconnect()
        await loop.aclose()


async def test_false_legacy_input_append_result_does_not_publish_receipt(tmp_path):
    key = "websocket:empty-input"
    bus = MessageBus()
    seen: list[UserInputCommitted] = []
    bus.subscribe(seen.append, UserInputCommitted)
    session_manager = SessionManager(tmp_path / "sessions")
    session = session_manager.get_or_create(key)
    loop = object.__new__(AgentLoop)
    loop.auto_compact = None
    loop.sessions = session_manager
    loop._request_context_for_turn = lambda _ctx: None
    loop._resolve_runtime_context_for_turn = AsyncMock(return_value=[])
    delivery = TurnDeliveryFactory(bus).create(
        InboundMessage(
            sender_id="user",
            channel="websocket",
            chat_id="empty-input",
            content="",
            metadata={"webui": True, "webui_turn_id": "attempt-empty"},
        ),
        key,
    )
    ctx = TurnContext(
        msg=delivery.input_message,
        session_key=key,
        turn_id="run-empty",
        runtime=SimpleNamespace(),
        kind=TurnKind.USER,
        delivery=delivery,
        session=session,
    )

    await loop._build_turn(ctx)

    assert ctx.input_persisted_early is False
    assert seen == []
    assert session.messages == []


@pytest.mark.parametrize("source_mode", [False, True], ids=["legacy", "source"])
async def test_runtime_context_failure_precedes_any_input_commit(
    engine_config, source_mode: bool,
):
    key = "websocket:context-failure-source" if source_mode else "websocket:context-failure-legacy"
    chat_id = key.split(":", 1)[1]
    provider = ScriptedProvider([LLMResponse(content="unreachable")])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    source_backend = upgrade(loop, key) if source_mode else None
    coordinator = WebuiTurnCoordinator(
        bus=loop.bus,
        sessions=loop.sessions,
        schedule_background=lambda coro: coro.close(),
    )
    disconnect = coordinator.subscribe()
    committed: list[UserInputCommitted] = []
    receipts: list[MessageAdmittedEvent] = []
    loop.bus.subscribe(committed.append, UserInputCommitted)
    original_publish_outbound = loop.bus.publish_outbound

    async def observe_outbound(message):
        if isinstance(message.event, MessageAdmittedEvent):
            receipts.append(message.event)
        await original_publish_outbound(message)

    loop.bus.publish_outbound = observe_outbound
    loop._resolve_runtime_context_for_turn = AsyncMock(
        side_effect=RuntimeError("fixture context failure"),
    )
    try:
        result = await loop.execute_turn(TurnRequest(
            content="hello",
            session_key=key,
            source="gateway",
            route=DeliveryTarget(channel="websocket", chat_id=chat_id),
            metadata={"webui": True, "webui_turn_id": "attempt-context-failure"},
        ))

        assert result.error == "fixture context failure"
        assert committed == []
        assert receipts == []
        if source_backend is None:
            assert not any(
                row.get("role") == "user" and row.get("content") == "hello"
                for row in loop.sessions.get_or_create(key).messages
            )
        else:
            assert not source_backend.source.load_source_snapshot(key).events
        assert not provider.calls
    finally:
        disconnect()
        await loop.aclose()
