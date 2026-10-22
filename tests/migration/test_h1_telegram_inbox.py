"""Authenticated Telegram facts and stable album admission for H1."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.busy_control import (
    BusyControlRouter,
    ControlDecision,
    ControlRequest,
    ControlSnapshot,
    RoutingProposal,
)
from nanobot.agent.busy_dispatch import BusyInboxConsumer, BusyTurn
from nanobot.agent.loop import AgentLoop
from nanobot.agent.turn import RunStatus
from nanobot.bus.events import OutboundMessage
from nanobot.channels.routed_delivery import (
    RoutedDeliveryError,
    RoutedDeliveryReceipt,
    RoutedDeliveryRequest,
    RoutedPartReceipt,
)
from nanobot.channels.telegram.runtime import TelegramChannel, TelegramConfig
from nanobot.config.schema import BusyControlConfig
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.session.busy_inbox import (
    InboxCapacityError,
    InboxConflictError,
    InboxPayload,
    InboxResult,
    Origin,
    SessionBusyInbox,
    TelegramAlbumMember,
)
from nanobot.session.manager import SessionManager
from tests.agent.p6_admitted_fixtures import AgentInputs, AsyncAdmittedFactory
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async


def _telegram_message(
    message_id: int,
    chat_id: int,
    content: str,
    *,
    chat_type: str = "supergroup",
    forward_origin: object | None = None,
    reply_to_message: object | None = None,
    quote: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        message_id=message_id,
        chat_id=chat_id,
        chat=SimpleNamespace(id=chat_id, type=chat_type),
        message_thread_id=77,
        reply_to_message=reply_to_message,
        forward_origin=forward_origin,
        quote=quote,
        text=content,
        caption=None,
        location=None,
    )


async def _flush_telegram_album(
    channel: TelegramChannel,
    user: SimpleNamespace,
    group_id: str,
    members: tuple[tuple[int, str], ...],
) -> None:
    key = f"-100123:{group_id}"
    for message_id, content in members:
        await _buffer_telegram_album_member(channel, user, group_id, message_id, content)
    await channel._flush_media_group(key)


async def _buffer_telegram_album_member(
    channel: TelegramChannel,
    user: SimpleNamespace,
    group_id: str,
    message_id: int,
    content: str,
) -> None:
    key = f"-100123:{group_id}"
    message = _telegram_message(message_id, -100123, content)
    message.media_group_id = group_id
    await channel._process_message_update(
        SimpleNamespace(message=message, effective_user=user), MagicMock(),
    )
    timer = channel._media_group_tasks.pop(key, None)
    if timer is not None:
        timer.cancel()
        await asyncio.gather(timer, return_exceptions=True)


def _isolate_telegram_album_runtime(channel: TelegramChannel, consumer, monkeypatch) -> None:
    monkeypatch.setattr(channel, "_is_group_message_for_bot", AsyncMock(return_value=True))
    monkeypatch.setattr(channel, "_download_message_media", AsyncMock(return_value=([], [])))
    monkeypatch.setattr(channel, "_start_typing", MagicMock())
    monkeypatch.setattr(channel, "_add_reaction", AsyncMock())
    monkeypatch.setattr(channel, "_remember_thread_context", MagicMock())
    monkeypatch.setattr(channel, "_build_message_metadata", MagicMock(return_value={}))
    monkeypatch.setattr(channel, "_derive_topic_session_key", MagicMock(return_value="assistant"))
    monkeypatch.setattr(channel, "_handle_message", AsyncMock())
    monkeypatch.setattr(consumer, "wake", MagicMock())
    edge = channel._busy_inbox
    assert edge is not None
    monkeypatch.setattr(edge, "wake_notices", MagicMock())


def _receipt_for(
    request: RoutedDeliveryRequest,
    first_message_id: int,
) -> RoutedDeliveryReceipt:
    address = request.address
    thread_id = int(address.thread_id) if address.thread_id is not None else None
    reply_id = int(address.reply_to_message_id) if address.reply_to_message_id is not None else None
    parts: list[RoutedPartReceipt] = []
    for message_index, _attachment in enumerate(request.bundle.attachments):
        parts.append(RoutedPartReceipt(
            index=len(parts),
            message_index=message_index,
            part_index=0,
            kind="media",
            provider_message_id=first_message_id + len(parts),
            provider_chat_id=int(address.chat_id),
            provider_thread_id=thread_id,
            provider_reply_to_message_id=reply_id,
        ))
    for part_index, _text in enumerate(request.bundle.emitted_text_parts):
        parts.append(RoutedPartReceipt(
            index=len(parts),
            message_index=len(request.bundle.attachments),
            part_index=part_index,
            kind="text",
            provider_message_id=first_message_id + len(parts),
            provider_chat_id=int(address.chat_id),
            provider_thread_id=thread_id,
            provider_reply_to_message_id=reply_id,
        ))
    return RoutedDeliveryReceipt(
        address=address,
        action_id=request.action_id,
        bundle_digest=request.bundle_digest,
        parts=tuple(parts),
    )


def _claimed_result_turn(inbox: SessionBusyInbox, consumer: BusyInboxConsumer) -> BusyTurn:
    row = inbox.admit(InboxPayload(
        owner_id="owner",
        session_key="assistant",
        origin=Origin(
            channel="telegram",
            account_id="999",
            chat_id="-100123",
            chat_type="supergroup",
            sender_id="42",
            transport_id="result-input",
            thread_id="77",
            output_reply_to_message_id="501",
        ),
        content="task",
    )).entry
    queued = inbox.route(
        row.id,
        row.revision,
        ControlDecision("next-turn", None, "idle"),
        current_run=None,
        active_scope=(),
    )
    claimed = inbox.claim_next_turn("output-only-run", current_run=None)
    assert claimed is not None and claimed.id == queued.id
    return BusyTurn(consumer=consumer, run_id="output-only-run", initial=claimed)


def _payload(
    album_members: tuple[TelegramAlbumMember, ...],
    *,
    transport_id: str = "album-1",
) -> InboxPayload:
    ordered = tuple(sorted(album_members, key=lambda member: member.message_id))
    return InboxPayload(
        owner_id="owner",
        session_key="telegram:chat:topic:42",
        origin=Origin(
            channel="telegram",
            account_id="bot-1",
            chat_id="chat",
            chat_type="supergroup",
            sender_id="user-1",
            transport_id=transport_id,
            thread_id="42",
            reply_to_message_id="80",
            forwarded=False,
        ),
        content="\n".join(member.content for member in ordered),
        media=tuple(path for member in ordered for path in member.media),
        album_id="media-group-7",
        album_members=album_members,
    )


def test_origin_persists_authenticated_chat_type_reply_and_forward_provenance() -> None:
    origin = Origin(
        channel="telegram",
        account_id="bot-1",
        chat_id="-100123",
        chat_type="supergroup",
        sender_id="42|alice",
        transport_id="501",
        thread_id="77",
        reply_to_message_id="500",
        forwarded=True,
    )

    assert origin.chat_type == "supergroup"
    assert origin.reply_to_message_id == "500"
    assert origin.forwarded is True


def test_album_reordering_and_duplicate_members_do_not_create_new_work(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    media_12 = tmp_path / "album-12.png"
    media_12.write_bytes(b"member twelve")
    media_11 = tmp_path / "album-11.png"
    media_11.write_bytes(b"member eleven")
    first = TelegramAlbumMember.create("12", "caption", (str(media_12),))
    second = TelegramAlbumMember.create("11", "", (str(media_11),))
    store = SessionBusyInbox(sessions, owner_id="owner", session_key="telegram:chat:topic:42")

    admission = store.admit(_payload((first, second)))
    restored = SessionBusyInbox(sessions, owner_id="owner", session_key="telegram:chat:topic:42")
    duplicate = restored.admit(_payload((second, first)))

    assert admission.duplicate is False
    assert duplicate.duplicate is True
    assert duplicate.entry.id == admission.entry.id
    assert len(restored.entries()) == 1
    assert tuple(member.message_id for member in duplicate.entry.album_members) == ("11", "12")


def test_album_member_identity_binds_local_media_bytes_across_reload(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    media_path = tmp_path / "album.png"
    media_path.write_bytes(b"original album bytes")
    store = SessionBusyInbox(sessions, owner_id="owner", session_key="telegram:chat:topic:42")
    first = TelegramAlbumMember.create("11", "caption", (str(media_path),))
    admitted = store.admit(_payload((first,)))

    restored = SessionBusyInbox(sessions, owner_id="owner", session_key="telegram:chat:topic:42")
    duplicate = restored.admit(_payload((TelegramAlbumMember.create("11", "caption", (str(media_path),)),)))
    assert duplicate.duplicate is True
    assert duplicate.entry.id == admitted.entry.id

    media_path.write_bytes(b"changed album bytes")
    restored = SessionBusyInbox(sessions, owner_id="owner", session_key="telegram:chat:topic:42")
    changed = _payload((TelegramAlbumMember.create("11", "caption", (str(media_path),)),))
    with pytest.raises(InboxConflictError, match="album member"):
        restored.admit(changed)

    assert len(restored.entries()) == 1


def test_late_album_member_becomes_separate_supplemental_ingress(tmp_path) -> None:
    sessions = SessionManager(tmp_path)
    store = SessionBusyInbox(sessions, owner_id="owner", session_key="telegram:chat:topic:42")
    first = TelegramAlbumMember.create("11", "first")
    original = store.admit(_payload((first,))).entry
    queued = store.route(
        original.id,
        original.revision,
        ControlDecision("next-turn", None, "idle"),
        current_run=None,
        active_scope=(),
    )
    claimed = store.claim_next_turn("run-original", current_run=None)
    assert claimed is not None and claimed.id == queued.id
    started = store.begin_dispatch(claimed.id, claimed.revision)
    store.finish(started.id, started.revision, "completed")

    late_media = tmp_path / "late-member.png"
    late_media.write_bytes(b"late album member")
    late = TelegramAlbumMember.create("12", "late", (str(late_media),))
    supplement = store.admit(_payload((first, late)))

    assert supplement.duplicate is False
    assert supplement.entry.id != original.id
    assert supplement.entry.payload is not None
    assert supplement.entry.payload.content == "late"
    assert supplement.entry.payload.media == (str(late_media),)
    assert tuple(member.message_id for member in supplement.entry.album_members) == ("12",)


class _OfflineControlModel:
    async def classify(self, request: ControlRequest) -> RoutingProposal:
        return RoutingProposal(intent="next-turn", confidence=1.0)


@pytest.fixture
async def registered_telegram_owner(tmp_path, admitted_async: AsyncAdmittedFactory):
    provider = MagicMock()
    provider.get_default_model.return_value = "offline-model"
    provider.generation = GenerationSettings()
    provider.supports_bound_conversation_state.return_value = False
    provider.can_resume_conversation_state.return_value = False
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done"))

    def build(inputs: AgentInputs) -> AgentLoop:
        return AgentLoop(
            bus=inputs.bus,
            provider=provider,
            workspace=inputs.config.workspace_path,
            model="offline-model",
            max_iterations=4,
            session_manager=inputs.sessions,
            tool_registry=inputs.tools,
            cron_service=inputs.cron,
            turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools,
            restrict_to_workspace=True,
            enable_legacy_auto_compact=False,
        )

    loop = await admitted_async(tmp_path, build, keys=("assistant",))
    inbox = SessionBusyInbox(loop.sessions, owner_id="owner", session_key="assistant")
    channel = TelegramChannel(
        TelegramConfig(allow_from=["42"], allow_chats=["-100123"]),
        loop.bus,
    )
    channel._bot_user_id = 999
    consumer = loop.register_telegram_busy_inbox(
        channel,
        inbox,
        BusyControlRouter(_OfflineControlModel(), BusyControlConfig()),
        account_id="999",
        owner_sender_ids=frozenset({"42"}),
        owner_chat_ids=frozenset({"-100123"}),
        tool_scope=(),
        automation_scope=(),
    ).consumer
    return loop, inbox, consumer, channel, provider


async def test_runtime_album_buffer_deduplicates_and_flushes_in_message_order(
    registered_telegram_owner, monkeypatch
) -> None:
    _loop, inbox, consumer, channel, _provider = registered_telegram_owner
    _isolate_telegram_album_runtime(channel, consumer, monkeypatch)
    user = SimpleNamespace(id=42, username="alice", first_name="Alice")

    await _flush_telegram_album(
        channel, user, "album-order-a", ((12, "second"), (11, "first"), (12, "second")),
    )
    await _flush_telegram_album(
        channel, user, "album-order-b", ((21, "first"), (22, "second")),
    )

    admitted = tuple(entry for entry in inbox.entries() if entry.payload is not None)
    assert len(admitted) == 2
    assert admitted[0].payload.content == admitted[1].payload.content == "first\nsecond"
    assert tuple(member.message_id for member in admitted[0].album_members) == ("11", "12")
    assert tuple(member.message_id for member in admitted[1].album_members) == ("21", "22")


async def test_runtime_album_buffer_rejects_changed_duplicate_member(
    registered_telegram_owner, monkeypatch
) -> None:
    _loop, inbox, consumer, channel, _provider = registered_telegram_owner
    _isolate_telegram_album_runtime(channel, consumer, monkeypatch)
    user = SimpleNamespace(id=42, username="alice", first_name="Alice")

    await _buffer_telegram_album_member(channel, user, "album-conflict", 31, "original")
    with pytest.raises(InboxConflictError, match="album member"):
        await _buffer_telegram_album_member(channel, user, "album-conflict", 31, "changed")

    assert inbox.entries() == ()


async def test_runtime_late_album_member_keeps_its_own_reply_route(
    registered_telegram_owner, monkeypatch
) -> None:
    _loop, inbox, consumer, channel, _provider = registered_telegram_owner
    _isolate_telegram_album_runtime(channel, consumer, monkeypatch)
    user = SimpleNamespace(id=42, username="alice", first_name="Alice")

    await _flush_telegram_album(
        channel, user, "album-supplement", ((11, "old"), (12, "old member")),
    )
    await _flush_telegram_album(
        channel, user, "album-supplement",
        ((11, "old"), (12, "old member"), (13, "late")),
    )

    admitted = tuple(entry for entry in inbox.entries() if entry.payload is not None)
    assert len(admitted) == 2
    assert admitted[1].payload is not None
    assert admitted[1].payload.content == "late"
    assert admitted[1].origin.output_reply_to_message_id == "13"


@pytest.mark.parametrize("excluded_binding", ["sender", "chat", "account"])
async def test_registered_edge_consumes_updates_outside_narrow_owner_binding(
    registered_telegram_owner, monkeypatch, excluded_binding
) -> None:
    loop, inbox, _consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    channel.config = TelegramConfig(allow_from=["42"], allow_chats=[])
    chat_id = -100123
    if excluded_binding == "sender":
        edge.owner_sender_ids = frozenset({"43"})
    elif excluded_binding == "chat":
        edge.owner_chat_ids = frozenset({"-100321"})
        chat_id = -100456
    else:
        channel._bot_user_id = 998

    user = SimpleNamespace(id=42, username="alice", first_name="Alice")
    message = _telegram_message(501, chat_id, "hello")
    update = SimpleNamespace(message=message, effective_user=user)
    legacy_dispatch = AsyncMock()
    monkeypatch.setattr(channel, "_handle_message", legacy_dispatch)
    monkeypatch.setattr(channel, "_is_group_message_for_bot", AsyncMock(return_value=True))
    monkeypatch.setattr(channel, "_download_message_media", AsyncMock(return_value=([], [])))
    monkeypatch.setattr(channel, "_start_typing", MagicMock())
    monkeypatch.setattr(channel, "_add_reaction", AsyncMock())
    monkeypatch.setattr(channel, "_remember_thread_context", MagicMock())
    monkeypatch.setattr(channel, "_build_message_metadata", MagicMock(return_value={}))
    monkeypatch.setattr(channel, "_derive_topic_session_key", MagicMock(return_value="rogue-session"))

    await channel._process_message_update(update, MagicMock())

    assert inbox.entries() == ()
    assert loop.bus.inbound_size == 0
    legacy_dispatch.assert_not_awaited()


async def test_registered_edge_keeps_direct_start_and_help_handlers(
    registered_telegram_owner,
) -> None:
    _loop, _inbox, _consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    edge.owner_sender_ids = frozenset({"43"})
    channel.config = TelegramConfig(allow_from=["42"], allow_chats=[])
    user = SimpleNamespace(id=42, username="alice", first_name="Alice")
    message = _telegram_message(501, -100123, "/start")
    message.reply_text = AsyncMock()
    update = SimpleNamespace(message=message, effective_user=user)

    await channel._on_start(update, MagicMock())
    await channel._on_help(update, MagicMock())

    assert message.reply_text.await_count == 2


async def test_registered_edge_rejects_update_without_chat_type(
    registered_telegram_owner, monkeypatch
) -> None:
    _loop, inbox, _consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    router = AsyncMock(return_value=ControlDecision("next-turn", None, "idle"))
    monkeypatch.setattr(edge.router, "propose", router)
    wake_notices = MagicMock()
    monkeypatch.setattr(edge, "wake_notices", wake_notices)

    user = SimpleNamespace(id=42, username="alice", first_name="Alice")
    message = _telegram_message(501, -100123, "missing authenticated chat type")
    message.chat = SimpleNamespace(id=-100123, type=None)

    consumed = await edge.handle_update(message, user, content="missing authenticated chat type")

    assert consumed is True
    assert inbox.entries() == ()
    router.assert_not_awaited()
    wake_notices.assert_not_called()


async def test_full_update_persists_assembled_input_before_acceptance_receipt(
    registered_telegram_owner, monkeypatch, tmp_path
) -> None:
    _loop, inbox, _consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    user = SimpleNamespace(id=42, username="alice", first_name="Alice")
    message = _telegram_message(701, -100123, "message body")
    message.caption = "caption body"
    message.reply_to_message = SimpleNamespace(message_id=700)
    message.quote = SimpleNamespace(text="quoted source")
    current_media = tmp_path / "current.png"
    reply_media = tmp_path / "reply.png"
    current_media.write_bytes(b"complete current attachment")
    reply_media.write_bytes(b"complete replied attachment")

    async def propose(
        _content: str, observed: ControlSnapshot | None, *, direct: bool,
    ) -> ControlDecision:
        return ControlDecision("clarify", observed, "ambiguous")

    monkeypatch.setattr(edge.router, "propose", AsyncMock(side_effect=propose))
    monkeypatch.setattr(channel, "_is_group_message_for_bot", AsyncMock(return_value=True))
    monkeypatch.setattr(channel, "_download_message_media", AsyncMock(side_effect=[
        ([str(current_media)], ["[current photo]"]),
        ([str(reply_media)], ["[replied photo]"]),
    ]))
    monkeypatch.setattr(
        channel, "_extract_reply_context", AsyncMock(return_value="[quoted source text]"),
    )
    monkeypatch.setattr(channel, "_start_typing", MagicMock())
    monkeypatch.setattr(channel, "_add_reaction", AsyncMock())
    monkeypatch.setattr(channel, "_remember_thread_context", MagicMock())

    requests: list[RoutedDeliveryRequest] = []
    payloads_at_receipt: list[InboxPayload] = []
    receipt_sent = asyncio.Event()

    async def send_receipt(request: RoutedDeliveryRequest) -> RoutedDeliveryReceipt:
        requests.append(request)
        payloads_at_receipt.extend(
            entry.payload for entry in inbox.entries() if entry.payload is not None
        )
        receipt_sent.set()
        return _receipt_for(request, 1200 + len(requests) * 10)

    monkeypatch.setattr(channel, "send_routed_with_receipt", send_receipt)

    try:
        await channel._process_message_update(
            SimpleNamespace(message=message, effective_user=user), MagicMock(),
        )
        await asyncio.wait_for(receipt_sent.wait(), timeout=5)
    finally:
        await edge.aclose()

    assert requests[0].bundle.content == "Received and saved for routing."
    assert len(payloads_at_receipt) >= 1
    payload = payloads_at_receipt[0]
    assert payload.content == "[quoted source text]\nmessage body\ncaption body\n[current photo]"
    assert payload.media == (str(reply_media), str(current_media))
    assert reply_media.read_bytes() == b"complete replied attachment"
    assert current_media.read_bytes() == b"complete current attachment"
    assert payload.origin.channel == "telegram"
    assert payload.origin.account_id == "999"
    assert payload.origin.sender_id == "42|alice"
    assert payload.origin.chat_type == "supergroup"
    assert payload.origin.chat_id == "-100123"
    assert payload.origin.thread_id == "77"
    assert payload.origin.reply_to_message_id == "700"
    assert payload.origin.quoted_message_id == "700"
    assert payload.origin.output_reply_to_message_id == "701"
    assert payload.origin.forwarded is False


@pytest.mark.parametrize(
    ("provenance", "expected_direct"),
    [("direct", True), ("forwarded", False), ("quoted", False)],
)
async def test_stop_routes_before_fast_commands_and_rejects_indirect_authority(
    registered_telegram_owner, monkeypatch, provenance, expected_direct
) -> None:
    _loop, _inbox, consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    calls: list[tuple[str, ControlSnapshot | None, bool]] = []

    async def propose(
        content: str, observed: ControlSnapshot | None, *, direct: bool,
    ) -> ControlDecision:
        calls.append((content, observed, direct))
        if direct:
            return ControlDecision("stop", observed, "explicit-stop")
        return ControlDecision("next-turn", observed, "classified")

    monkeypatch.setattr(edge.router, "propose", AsyncMock(side_effect=propose))
    apply_control = AsyncMock()
    monkeypatch.setattr(consumer, "apply_control", apply_control)
    monkeypatch.setattr(consumer, "wake", MagicMock())
    fast_command = AsyncMock(return_value=True)
    legacy_dispatch = AsyncMock()
    monkeypatch.setattr(channel, "_try_fast_command", fast_command)
    monkeypatch.setattr(channel, "_handle_message", legacy_dispatch)
    monkeypatch.setattr(channel, "_remember_thread_context", MagicMock())

    reply = SimpleNamespace(message_id=500) if provenance == "quoted" else None
    channel._bot_username = "nanobot"
    message = _telegram_message(
        501,
        -100123,
        "/stop@nanobot",
        forward_origin=SimpleNamespace() if provenance == "forwarded" else None,
        reply_to_message=reply,
        quote=SimpleNamespace() if provenance == "quoted" else None,
    )
    user = SimpleNamespace(id=42, username="alice", first_name="Alice")
    await channel._process_forward_command(
        SimpleNamespace(message=message, effective_user=user), MagicMock(),
    )

    assert calls == [("/stop", None, expected_direct)]
    fast_command.assert_not_awaited()
    legacy_dispatch.assert_not_awaited()
    assert apply_control.await_count == (1 if expected_direct else 0)


async def test_bound_unmentioned_group_stop_is_consumed_before_routing_or_legacy_fallback(
    registered_telegram_owner, monkeypatch
) -> None:
    _loop, inbox, consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    monkeypatch.setattr(channel, "_ensure_bot_identity", AsyncMock(return_value=(999, "nanobot")))
    propose = AsyncMock()
    monkeypatch.setattr(edge.router, "propose", propose)
    apply_control = AsyncMock()
    monkeypatch.setattr(consumer, "apply_control", apply_control)
    wake = MagicMock()
    monkeypatch.setattr(consumer, "wake", wake)
    fast_command = AsyncMock(return_value=True)
    legacy_dispatch = AsyncMock()
    receipt_send = AsyncMock()
    monkeypatch.setattr(channel, "_try_fast_command", fast_command)
    monkeypatch.setattr(channel, "_handle_message", legacy_dispatch)
    monkeypatch.setattr(channel, "send_routed_with_receipt", receipt_send)
    monkeypatch.setattr(channel, "_remember_thread_context", MagicMock())

    message = _telegram_message(502, -100123, "/stop")
    user = SimpleNamespace(id=42, username="alice", first_name="Alice")
    await channel._process_forward_command(
        SimpleNamespace(message=message, effective_user=user), MagicMock(),
    )

    assert inbox.entries() == ()
    propose.assert_not_awaited()
    apply_control.assert_not_awaited()
    wake.assert_not_called()
    fast_command.assert_not_awaited()
    legacy_dispatch.assert_not_awaited()
    receipt_send.assert_not_awaited()


@pytest.mark.parametrize("failure", [InboxCapacityError("full"), OSError("commit failed")])
async def test_transport_admission_failure_does_not_send_success_receipt(
    registered_telegram_owner, monkeypatch, failure
) -> None:
    _loop, inbox, _consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    admit = MagicMock(side_effect=failure)
    monkeypatch.setattr(inbox, "admit", admit)
    receipt_send = AsyncMock()
    monkeypatch.setattr(channel, "send_routed_with_receipt", receipt_send)
    wake_notices = MagicMock()
    monkeypatch.setattr(edge, "wake_notices", wake_notices)
    legacy_dispatch = AsyncMock()
    monkeypatch.setattr(channel, "_handle_message", legacy_dispatch)
    monkeypatch.setattr(channel, "_is_group_message_for_bot", AsyncMock(return_value=True))
    monkeypatch.setattr(channel, "_download_message_media", AsyncMock(return_value=([], [])))
    monkeypatch.setattr(channel, "_start_typing", MagicMock())
    monkeypatch.setattr(channel, "_add_reaction", AsyncMock())
    monkeypatch.setattr(channel, "_remember_thread_context", MagicMock())
    monkeypatch.setattr(channel, "_build_message_metadata", MagicMock(return_value={}))
    monkeypatch.setattr(channel, "_derive_topic_session_key", MagicMock(return_value="assistant"))
    message = _telegram_message(503, -100123, "save this")
    user = SimpleNamespace(id=42, username="alice", first_name="Alice")

    with pytest.raises(type(failure)):
        await channel._process_message_update(
            SimpleNamespace(message=message, effective_user=user), MagicMock(),
        )

    admit.assert_called_once()
    wake_notices.assert_not_called()
    receipt_send.assert_not_awaited()
    legacy_dispatch.assert_not_awaited()


async def test_clarification_reply_uses_exact_notice_and_preserves_origin_scope(
    registered_telegram_owner, monkeypatch
) -> None:
    _loop, inbox, consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    edge.tool_scope = ("read", "write")
    monkeypatch.setattr(consumer, "wake", MagicMock())
    monkeypatch.setattr(edge, "wake_notices", MagicMock())
    proposals: list[str] = []

    async def propose(
        content: str, observed: ControlSnapshot | None, *, direct: bool,
    ) -> ControlDecision:
        proposals.append(content)
        if content in {"question one", "question two", "unmatched reply"}:
            return ControlDecision("clarify", observed, "ambiguous")
        return ControlDecision("next-turn", observed, "classified")

    monkeypatch.setattr(edge.router, "propose", AsyncMock(side_effect=propose))
    user = SimpleNamespace(id=42, username="alice", first_name="Alice")
    requests: list[RoutedDeliveryRequest] = []

    async def send_notice(request: RoutedDeliveryRequest) -> RoutedDeliveryReceipt:
        requests.append(request)
        return _receipt_for(request, 1300 + len(requests) * 10)

    monkeypatch.setattr(channel, "send_routed_with_receipt", send_notice)

    await edge.handle_update(
        _telegram_message(601, -100123, "question one"), user, content="question one",
    )
    await edge._flush_notices_once()
    first = next(row for row in inbox.entries() if row.payload is not None
                 and row.payload.content == "question one")
    assert first.state == "awaiting-clarification"
    assert first.clarification_receipt_message_id is not None

    await edge.handle_update(
        _telegram_message(602, -100123, "question two"), user, content="question two",
    )
    await edge._flush_notices_once()
    second = next(row for row in inbox.entries() if row.payload is not None
                  and row.payload.content == "question two")
    assert second.state == "awaiting-clarification"
    assert second.clarification_receipt_message_id is not None
    assert second.clarification_receipt_message_id != first.clarification_receipt_message_id

    unmatched_message = _telegram_message(
        603, -100123, "unmatched reply", reply_to_message=SimpleNamespace(message_id=9999),
    )
    await edge.handle_update(unmatched_message, user, content="unmatched reply")
    await edge._flush_notices_once()
    unmatched = next(row for row in inbox.entries() if row.payload is not None
                     and row.payload.origin.transport_id == "603")
    assert unmatched.id not in {first.id, second.id}
    assert unmatched.state == "awaiting-clarification"
    assert unmatched.origin.output_reply_to_message_id == "603"

    edge.tool_scope = ("read",)
    exact_reply = _telegram_message(
        604,
        -100123,
        "answer for question one",
        reply_to_message=SimpleNamespace(message_id=first.clarification_receipt_message_id),
    )
    await edge.handle_update(exact_reply, user, content="answer for question one")
    await edge._flush_notices_once()

    resumed = next(row for row in inbox.entries() if row.id == first.id)
    still_awaiting = next(row for row in inbox.entries() if row.id == second.id)
    assert resumed.state == "next-turn"
    assert resumed.origin == first.origin
    assert resumed.payload == first.payload
    assert resumed.clarifications[-1].content == "answer for question one"
    assert resumed.clarifications[-1].origin.transport_id == "604"
    assert resumed.effective_scope == frozenset({"read"})
    assert still_awaiting.state == "awaiting-clarification"
    assert still_awaiting.clarifications == ()
    assert still_awaiting.origin == second.origin
    assert "unmatched reply" in proposals


async def test_busy_task_routes_and_delivers_new_notice_while_provider_is_blocked(
    registered_telegram_owner, monkeypatch
) -> None:
    _loop, inbox, consumer, channel, provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()
    queued_notice_sent = asyncio.Event()
    attempts: list[RoutedDeliveryRequest] = []

    async def blocked_chat(*_args: object, **_kwargs: object) -> LLMResponse:
        provider_started.set()
        await release_provider.wait()
        return LLMResponse(content="task answer")

    async def send_routed(request: RoutedDeliveryRequest) -> RoutedDeliveryReceipt:
        attempts.append(request)
        if request.bundle.content == "Queued for the next task turn.":
            queued_notice_sent.set()
        return _receipt_for(request, 1100 + len(attempts) * 10)

    monkeypatch.setattr(provider, "chat_with_retry", AsyncMock(side_effect=blocked_chat))
    monkeypatch.setattr(channel, "send_routed_with_receipt", send_routed)
    original = inbox.admit(InboxPayload(
        owner_id="owner",
        session_key="assistant",
        origin=Origin(
            channel="telegram",
            account_id="999",
            chat_id="-100123",
            chat_type="supergroup",
            sender_id="42",
            transport_id="blocked-task",
            thread_id="77",
            output_reply_to_message_id="501",
        ),
        content="long task",
    )).entry
    queued = inbox.route(
        original.id,
        original.revision,
        ControlDecision("next-turn", None, "idle"),
        current_run=None,
        active_scope=(),
    )
    consumer.wake()

    try:
        await asyncio.wait_for(provider_started.wait(), timeout=5)
        user = SimpleNamespace(id=42, username="alice", first_name="Alice")
        accepted = await edge.handle_update(
            _telegram_message(502, -100123, "another task"),
            user,
            content="another task",
            direct=True,
        )
        await asyncio.wait_for(queued_notice_sent.wait(), timeout=5)

        additional = next(entry for entry in inbox.entries() if entry.payload is not None
                          and entry.payload.content == "another task")
        assert accepted is True
        assert additional.state == "next-turn"
        assert additional.origin.output_reply_to_message_id == "502"
        assert provider.chat_with_retry.await_count == 1
        assert not release_provider.is_set()
        assert any("another task" not in request.bundle.content for request in attempts)
        assert queued in inbox.entries() or any(entry.id == queued.id for entry in inbox.entries())
    finally:
        release_provider.set()
        await consumer.wait_idle()
        await edge.aclose()

    assert provider.chat_with_retry.await_count == 2
    result_requests = [request for request in attempts if request.bundle.content == "task answer"]
    assert len(result_requests) == 2
    assert all(request.address.chat_id == "-100123" for request in result_requests)
    assert all(request.address.thread_id == "77" for request in result_requests)
    assert {request.address.reply_to_message_id for request in result_requests} == {"501", "502"}


async def test_active_status_notice_reports_running_task_before_provider_release(
    registered_telegram_owner, monkeypatch
) -> None:
    _loop, inbox, consumer, channel, provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()
    status_notice_sent = asyncio.Event()
    attempts: list[RoutedDeliveryRequest] = []
    expected_status = "The control request finished; the task is still running."

    async def blocked_chat(*_args: object, **_kwargs: object) -> LLMResponse:
        provider_started.set()
        await release_provider.wait()
        return LLMResponse(content="task answer")

    async def propose(
        content: str, observed: ControlSnapshot | None, *, direct: bool,
    ) -> ControlDecision:
        if content == "status question":
            assert observed is not None and observed.state == "running"
            return ControlDecision("status", observed, "requested-status")
        return ControlDecision("next-turn", observed, "classified")

    async def send_routed(request: RoutedDeliveryRequest) -> RoutedDeliveryReceipt:
        attempts.append(request)
        if (request.address.reply_to_message_id == "702"
                and request.bundle.content in {expected_status, "The task completed."}):
            status_notice_sent.set()
        return _receipt_for(request, 1600 + len(attempts) * 10)

    monkeypatch.setattr(provider, "chat_with_retry", AsyncMock(side_effect=blocked_chat))
    monkeypatch.setattr(edge.router, "propose", AsyncMock(side_effect=propose))
    monkeypatch.setattr(channel, "send_routed_with_receipt", send_routed)

    try:
        await edge.handle_update(
            _telegram_message(701, -100123, "long task"),
            SimpleNamespace(id=42, username="alice", first_name="Alice"),
            content="long task",
        )
        await asyncio.wait_for(provider_started.wait(), timeout=5)
        await edge.handle_update(
            _telegram_message(702, -100123, "status question"),
            SimpleNamespace(id=42, username="alice", first_name="Alice"),
            content="status question",
        )
        await asyncio.wait_for(status_notice_sent.wait(), timeout=5)

        status_requests = [request for request in attempts
                           if request.address.reply_to_message_id == "702"]
        assert any(request.bundle.content == expected_status for request in status_requests)
        assert all(request.bundle.content != "The task completed." for request in status_requests)
        status_entry = next(row for row in inbox.entries()
                            if row.origin.transport_id == "702")
        assert status_entry.intent == "status"
        assert status_entry.control_report is not None
        assert status_entry.control_report.active_run is not None
        assert status_entry.control_report.active_run.state == "running"
        assert status_entry.state == "completed"
        assert not release_provider.is_set()
        assert consumer.current_run is not None and consumer.current_run.state == "running"
    finally:
        release_provider.set()
        await consumer.wait_idle()
        await edge.aclose()


async def test_cross_route_steer_gets_notice_and_result_stays_on_initial_origin(
    registered_telegram_owner, monkeypatch
) -> None:
    loop, inbox, consumer, channel, provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    edge.owner_chat_ids = frozenset({"-100123", "-100456"})
    channel.config = TelegramConfig(allow_from=["42"], allow_chats=["-100123", "-100456"])
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()
    steer_notice_sent = asyncio.Event()
    steer_completion_notice_sent = asyncio.Event()
    attempts: list[RoutedDeliveryRequest] = []

    async def blocked_chat(*_args: object, **_kwargs: object) -> LLMResponse:
        provider_started.set()
        await release_provider.wait()
        return LLMResponse(content="original result")

    async def propose(
        content: str, observed: ControlSnapshot | None, *, direct: bool,
    ) -> ControlDecision:
        if content == "cross-route steer" and observed is not None:
            return ControlDecision("steer", observed, "classified")
        return ControlDecision("next-turn", observed, "idle")

    async def send_routed(request: RoutedDeliveryRequest) -> RoutedDeliveryReceipt:
        attempts.append(request)
        steer_notice = (
            "Added as guidance for the current task; it may not have been consumed yet."
        )
        if request.address.chat_id == "-100456" and request.bundle.content == steer_notice:
            steer_notice_sent.set()
        if request.address.chat_id == "-100456" and request.bundle.content == "The task completed.":
            steer_completion_notice_sent.set()
        return _receipt_for(request, 1500 + len(attempts) * 10)

    monkeypatch.setattr(provider, "chat_with_retry", AsyncMock(side_effect=blocked_chat))
    monkeypatch.setattr(edge.router, "propose", AsyncMock(side_effect=propose))
    monkeypatch.setattr(channel, "send_routed_with_receipt", send_routed)

    try:
        await edge.handle_update(
            _telegram_message(701, -100123, "initial task"),
            SimpleNamespace(id=42, username="alice", first_name="Alice"),
            content="initial task",
        )
        await asyncio.wait_for(provider_started.wait(), timeout=5)
        await edge.handle_update(
            _telegram_message(702, -100456, "cross-route steer"),
            SimpleNamespace(id=42, username="alice", first_name="Alice"),
            content="cross-route steer",
        )
        await asyncio.wait_for(steer_notice_sent.wait(), timeout=5)
        release_provider.set()
        await consumer.wait_idle()
        await asyncio.wait_for(steer_completion_notice_sent.wait(), timeout=5)
    finally:
        release_provider.set()
        await consumer.wait_idle()
        await edge.aclose()

    results = [request for request in attempts if request.bundle.content == "original result"]
    assert len(results) == 1
    assert results[0].address.chat_id == "-100123"
    assert results[0].address.thread_id == "77"
    assert results[0].address.reply_to_message_id == "701"
    cross_route = [request for request in attempts if request.address.chat_id == "-100456"]
    assert cross_route
    assert any(request.bundle.content.startswith("Added as guidance") for request in cross_route)
    assert any(request.bundle.content == "The task completed." for request in cross_route)
    assert all(request.bundle.content != "original result" for request in cross_route)
    assert all(request.address.reply_to_message_id == "702" for request in cross_route)
    assert loop.bus.outbound_size == 0

    initial = next(row for row in inbox.entries() if row.origin.transport_id == "701")
    assert initial.state == "completed"
    assert initial.result is not None and initial.result.disposition == "delivered"
    assert initial.result.origin == initial.origin
    assert initial.claimed_run_id is not None
    record = loop.run_store.load(initial.claimed_run_id)
    assert record is not None and record.status is RunStatus.COMPLETED


async def test_notice_ack_write_fault_is_recovered_uncertain_without_replay(
    registered_telegram_owner, monkeypatch
) -> None:
    loop, inbox, _consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    inbox.admit(InboxPayload(
        owner_id="owner",
        session_key="assistant",
        origin=Origin(
            channel="telegram",
            account_id="999",
            chat_id="-100123",
            chat_type="supergroup",
            sender_id="42",
            transport_id="notice-fault",
            thread_id="77",
            output_reply_to_message_id="501",
        ),
        content="private input must not appear in notice text",
    ))
    attempts: list[RoutedDeliveryRequest] = []

    async def send_notice(request: RoutedDeliveryRequest) -> RoutedDeliveryReceipt:
        attempts.append(request)
        return _receipt_for(request, 900)

    def fail_acknowledgement(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("notice acknowledgement persistence failed")

    def fail_disposition(*_args: object, **_kwargs: object) -> None:
        raise OSError("notice disposition persistence failed")

    monkeypatch.setattr(channel, "send_routed_with_receipt", send_notice)
    monkeypatch.setattr(inbox, "acknowledge_notice", fail_acknowledgement)
    monkeypatch.setattr(inbox, "fail_notice", fail_disposition)

    with pytest.raises(OSError, match="disposition persistence"):
        await edge._flush_notices_once()

    assert len(attempts) == 1
    assert "private input" not in attempts[0].bundle.content
    assert inbox.notices()[0].state == "sending"
    inbox.reconcile_orphans(live_run_ids=frozenset())
    recovered = SessionBusyInbox(loop.sessions, owner_id="owner", session_key="assistant")
    assert recovered.notices()[0].state == "uncertain"

    await edge._flush_notices_once()
    assert len(attempts) == 1


async def test_notice_cancelled_while_waiting_for_target_lock_is_retryable(
    registered_telegram_owner, monkeypatch
) -> None:
    _loop, inbox, _consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    inbox.admit(InboxPayload(
        owner_id="owner",
        session_key="assistant",
        origin=Origin(
            channel="telegram",
            account_id="999",
            chat_id="-100123",
            chat_type="supergroup",
            sender_id="42",
            transport_id="notice-lock",
            thread_id="77",
            output_reply_to_message_id="501",
        ),
        content="input",
    ))
    send_message = AsyncMock()
    channel._app = SimpleNamespace(bot=SimpleNamespace(send_message=send_message))
    channel._app_ready.set()
    target_lock = asyncio.Lock()
    await target_lock.acquire()
    channel._receipt_target_locks[(-100123, 77)] = target_lock

    task = asyncio.create_task(edge._flush_notices_once())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert inbox.notices()[0].state == "pending"
    send_message.assert_not_awaited()
    target_lock.release()


async def test_result_cancelled_while_waiting_for_target_lock_is_retryable(
    registered_telegram_owner
) -> None:
    _loop, inbox, consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    turn = _claimed_result_turn(inbox, consumer)
    send_message = AsyncMock()
    channel._app = SimpleNamespace(bot=SimpleNamespace(send_message=send_message))
    channel._app_ready.set()
    target_lock = asyncio.Lock()
    await target_lock.acquire()
    channel._receipt_target_locks[(-100123, 77)] = target_lock

    task = asyncio.create_task(edge.deliver_result(turn, OutboundMessage(
        channel="telegram", chat_id="-100123", content="result",
    )))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    result = next(entry for entry in inbox.entries() if entry.id == turn.initial.id).result
    assert result is not None and result.disposition == "definitely-unsent"
    assert result.bundle_digest is not None
    send_message.assert_not_awaited()
    target_lock.release()


async def test_result_ack_receipt_envelope_is_saved_before_delivery_and_reloads(
    registered_telegram_owner, monkeypatch
) -> None:
    loop, inbox, consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    turn = _claimed_result_turn(inbox, consumer)
    requests: list[RoutedDeliveryRequest] = []

    async def send_result(request: RoutedDeliveryRequest) -> RoutedDeliveryReceipt:
        requests.append(request)
        current = next(entry for entry in inbox.entries() if entry.id == turn.initial.id)
        assert current.result is not None
        assert current.result.output_digest
        assert current.result.bundle_digest == request.bundle_digest
        assert current.result.request_address is not None
        assert current.result.request_address.chat_id == request.address.chat_id
        return _receipt_for(request, 951)

    monkeypatch.setattr(channel, "send_routed_with_receipt", send_result)
    await edge.deliver_result(turn, OutboundMessage(
        channel="telegram", chat_id="-100123", content="result",
    ))

    restored = SessionBusyInbox(loop.sessions, owner_id="owner", session_key="assistant")
    result = next(entry for entry in restored.entries() if entry.id == turn.initial.id).result
    assert result is not None and result.disposition == "delivered"
    assert result.receipt_envelope is not None
    assert result.receipt_envelope.address.chat_id == requests[0].address.chat_id
    assert result.receipt_envelope.address.account_id == requests[0].address.account_id
    assert result.receipt_envelope.address.sender_id == requests[0].address.sender_id
    assert result.receipt_envelope.address.thread_id == requests[0].address.thread_id
    assert result.receipt_envelope.address.reply_to_message_id == requests[0].address.reply_to_message_id
    assert result.receipt_envelope.action_id == requests[0].action_id
    assert result.receipt_envelope.bundle_digest == requests[0].bundle_digest
    assert len(result.receipt_envelope.parts) == len(_receipt_for(requests[0], 951).parts)
    assert result.output_digest != result.bundle_digest


async def test_result_ack_before_receipt_save_recovers_uncertain_without_replay(
    registered_telegram_owner, monkeypatch
) -> None:
    loop, inbox, consumer, channel, provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    turn = _claimed_result_turn(inbox, consumer)
    requests: list[RoutedDeliveryRequest] = []

    async def send_result(request: RoutedDeliveryRequest) -> RoutedDeliveryReceipt:
        requests.append(request)
        return _receipt_for(request, 952)

    settle_result = inbox.settle_result

    def fail_delivered_receipt(*args: object, **kwargs: object):
        if kwargs.get("disposition") == "delivered":
            raise RuntimeError("result receipt persistence failed")
        return settle_result(*args, **kwargs)

    monkeypatch.setattr(channel, "send_routed_with_receipt", send_result)
    monkeypatch.setattr(inbox, "settle_result", fail_delivered_receipt)

    with pytest.raises(RuntimeError, match="receipt persistence"):
        await edge.deliver_result(turn, OutboundMessage(
            channel="telegram", chat_id="-100123", content="result",
        ))

    restored = SessionBusyInbox(loop.sessions, owner_id="owner", session_key="assistant")
    result = next(entry for entry in restored.entries() if entry.id == turn.initial.id).result
    assert result is not None and result.disposition == "uncertain"
    assert result.receipt_envelope is not None
    receipt_address = result.receipt_envelope.address
    request_address = requests[0].address
    assert (
        receipt_address.channel,
        receipt_address.account_id,
        receipt_address.sender_id,
        receipt_address.chat_type,
        receipt_address.chat_id,
        receipt_address.thread_id,
        receipt_address.reply_to_message_id,
    ) == (
        request_address.channel,
        request_address.account_id,
        request_address.sender_id,
        request_address.chat_type,
        request_address.chat_id,
        request_address.thread_id,
        request_address.reply_to_message_id,
    )
    assert result.receipt_envelope.action_id == requests[0].action_id
    assert result.receipt_envelope.bundle_digest == requests[0].bundle_digest
    assert result.receipt_envelope.parts
    assert len(requests) == 1
    assert provider.chat_with_retry.await_count == 0
    with pytest.raises(InboxConflictError, match="not eligible"):
        await edge.retry_result(turn.initial.id)


async def test_remote_reference_retry_reuses_exact_url_action_route_and_bundle_digest(
    registered_telegram_owner, monkeypatch
) -> None:
    _loop, inbox, consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    monkeypatch.setattr("nanobot.channels.telegram.busy_inbox.validate_url_target", lambda _url: (True, None))
    turn = _claimed_result_turn(inbox, consumer)
    url = "https://media.example.test/image.png"
    requests: list[RoutedDeliveryRequest] = []

    async def send_result(request: RoutedDeliveryRequest) -> RoutedDeliveryReceipt:
        requests.append(request)
        if len(requests) == 1:
            raise RoutedDeliveryError("definitely-unsent", "offline before send")
        return _receipt_for(request, 953)

    monkeypatch.setattr(channel, "send_routed_with_receipt", send_result)
    with pytest.raises(RoutedDeliveryError):
        await edge.deliver_result(turn, OutboundMessage(
            channel="telegram", chat_id="-100123", content="image", media=[url],
        ))
    row = next(entry for entry in inbox.entries() if entry.id == turn.initial.id)
    assert row.result is not None and row.result.disposition == "definitely-unsent"

    await edge.retry_result(row.id)

    assert len(requests) == 2
    assert requests[0] == requests[1]
    attachment = requests[0].bundle.attachments[0]
    assert attachment.source == url
    assert attachment.reference_digest
    assert attachment.content_digest is None
    assert requests[0].action_id == requests[1].action_id
    assert requests[0].address == requests[1].address
    assert requests[0].bundle_digest == requests[1].bundle_digest


async def test_changed_remote_reference_is_rejected_without_fetching(
    registered_telegram_owner, monkeypatch
) -> None:
    _loop, inbox, consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    monkeypatch.setattr("nanobot.channels.telegram.busy_inbox.validate_url_target", lambda _url: (True, None))
    turn = _claimed_result_turn(inbox, consumer)
    original_media = ("https://media.example.test/first.png",)
    media_digests = edge._media_digests(original_media)
    output_digest = edge._result_digest("image", original_media, media_digests, ())
    original = InboxResult(
        run_id=turn.run_id,
        origin=turn.initial.origin,
        action_id="a" * 64,
        output_digest=output_digest,
        bundle_digest=None,
        request_address=None,
        content="image",
        media=original_media,
        media_digests=media_digests,
        buttons=(),
        disposition="definitely-unsent",
    )
    changed = InboxResult(
        run_id=original.run_id,
        origin=original.origin,
        action_id=original.action_id,
        output_digest=original.output_digest,
        bundle_digest=original.bundle_digest,
        request_address=original.request_address,
        content=original.content,
        media=("https://media.example.test/changed.png",),
        media_digests=original.media_digests,
        buttons=original.buttons,
        disposition=original.disposition,
    )

    with pytest.raises(RoutedDeliveryError, match="stored_output_digest_changed"):
        edge._request(turn.initial.origin, changed)


async def test_definitely_unsent_result_retries_output_without_repeating_task(
    registered_telegram_owner, monkeypatch, tmp_path
) -> None:
    _loop, inbox, consumer, channel, provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    image_path = tmp_path / "result.png"
    image_path.write_bytes(b"stable output image")
    turn = _claimed_result_turn(inbox, consumer)
    response = OutboundMessage(
        channel="telegram",
        chat_id="-100123",
        content="result",
        media=[str(image_path)],
    )
    attempts: list[RoutedDeliveryRequest] = []

    async def send_result(request: RoutedDeliveryRequest) -> RoutedDeliveryReceipt:
        attempts.append(request)
        if len(attempts) == 1:
            raise RoutedDeliveryError("definitely-unsent", "offline before send")
        return _receipt_for(request, 950)

    monkeypatch.setattr(channel, "send_routed_with_receipt", send_result)

    with pytest.raises(RoutedDeliveryError) as first_failure:
        await edge.deliver_result(turn, response)
    assert first_failure.value.disposition == "definitely-unsent"
    row = next(entry for entry in inbox.entries() if entry.id == turn.initial.id)
    assert row.result is not None and row.result.disposition == "definitely-unsent"

    await edge.retry_result(row.id)

    settled = next(entry for entry in inbox.entries() if entry.id == row.id)
    assert settled.result is not None and settled.result.disposition == "delivered"
    assert len(attempts) == 2
    assert provider.chat_with_retry.await_count == 0


async def test_changed_attachment_bytes_block_output_only_retry(
    registered_telegram_owner, monkeypatch, tmp_path
) -> None:
    _loop, inbox, consumer, channel, _provider = registered_telegram_owner
    edge = channel._busy_inbox
    assert edge is not None
    image_path = tmp_path / "result.png"
    image_path.write_bytes(b"original output image")
    turn = _claimed_result_turn(inbox, consumer)
    response = OutboundMessage(
        channel="telegram",
        chat_id="-100123",
        content="result",
        media=[str(image_path)],
    )
    attempts: list[RoutedDeliveryRequest] = []

    async def send_result(request: RoutedDeliveryRequest) -> RoutedDeliveryReceipt:
        attempts.append(request)
        raise RoutedDeliveryError("definitely-unsent", "offline before send")

    monkeypatch.setattr(channel, "send_routed_with_receipt", send_result)

    with pytest.raises(RoutedDeliveryError):
        await edge.deliver_result(turn, response)
    row = next(entry for entry in inbox.entries() if entry.id == turn.initial.id)
    assert row.result is not None and row.result.disposition == "definitely-unsent"
    image_path.write_bytes(b"changed output image")

    with pytest.raises(RoutedDeliveryError) as changed:
        await edge.retry_result(row.id)

    assert changed.value.disposition == "definitely-unsent"
    assert changed.value.reason == "stored_output_digest_changed"
    assert len(attempts) == 1


async def test_unbound_telegram_busy_turn_keeps_legacy_reply_target_and_scope_call(
    registered_telegram_owner, monkeypatch
) -> None:
    loop, inbox, consumer, _channel, _provider = registered_telegram_owner
    entry = inbox.admit(InboxPayload(
        owner_id="owner",
        session_key="assistant",
        origin=Origin(
            channel="telegram",
            account_id="999",
            chat_id="-100123",
            chat_type="supergroup",
            sender_id="42",
            transport_id="502",
            thread_id="77",
        ),
        content="legacy task",
    )).entry
    turn = BusyTurn(consumer=consumer, run_id="legacy-run", initial=entry)
    monkeypatch.delitem(loop._telegram_busy_edges, inbox.session_key)
    legacy_scope = MagicMock()
    monkeypatch.setattr(loop, "_busy_turn_scope", legacy_scope)
    execute_turn = AsyncMock(return_value=None)
    monkeypatch.setattr(loop, "execute_turn", execute_turn)

    await loop._execute_busy_turn(turn)

    call = execute_turn.await_args
    assert call is not None
    request = call.args[0]
    assert request.route.channel == "telegram"
    assert request.route.message_id == "502"
    assert request.route.thread_id == "77"
    options = call.kwargs["_options"]
    options.admission_scope(inbox.session_key)
    legacy_scope.assert_called_once_with(options.inbound, inbox.session_key)


@pytest.mark.parametrize("fail_result_delivery", [False, True])
async def test_execute_turn_result_uses_origin_reply_hook_before_terminal_save(
    registered_telegram_owner, monkeypatch, fail_result_delivery
) -> None:
    loop, inbox, consumer, channel, provider = registered_telegram_owner
    attempts: list[RoutedDeliveryRequest] = []

    async def send_routed(request: RoutedDeliveryRequest) -> RoutedDeliveryReceipt:
        attempts.append(request)
        if request.bundle.content == "done":
            current = inbox.entries()[0]
            assert current.claimed_run_id is not None
            record = loop.run_store.load(current.claimed_run_id)
            assert record is not None and record.status is RunStatus.RUNNING
            if fail_result_delivery:
                raise RuntimeError("offline provider outcome unavailable")
        base_message_id = 100 + (len(attempts) * 10)
        message_parts: list[RoutedPartReceipt] = []
        for index in range(len(request.bundle.attachments)):
            message_parts.append(RoutedPartReceipt(
                index=len(message_parts),
                message_index=index,
                part_index=0,
                kind="media",
                provider_message_id=base_message_id + len(message_parts),
                provider_chat_id=int(request.address.chat_id),
                provider_thread_id=(int(request.address.thread_id)
                                    if request.address.thread_id is not None else None),
                provider_reply_to_message_id=(int(request.address.reply_to_message_id)
                                              if request.address.reply_to_message_id is not None else None),
            ))
        for part_index, _part in enumerate(request.bundle.emitted_text_parts):
            message_parts.append(RoutedPartReceipt(
                index=len(message_parts),
                message_index=len(request.bundle.attachments),
                part_index=part_index,
                kind="text",
                provider_message_id=base_message_id + len(message_parts),
                provider_chat_id=int(request.address.chat_id),
                provider_thread_id=(int(request.address.thread_id)
                                    if request.address.thread_id is not None else None),
                provider_reply_to_message_id=(int(request.address.reply_to_message_id)
                                              if request.address.reply_to_message_id is not None else None),
            ))
        return RoutedDeliveryReceipt(
            address=request.address,
            action_id=request.action_id,
            bundle_digest=request.bundle_digest,
            parts=tuple(message_parts),
        )

    monkeypatch.setattr(channel, "send_routed_with_receipt", send_routed)
    row = inbox.admit(InboxPayload(
        owner_id="owner",
        session_key="assistant",
        origin=Origin(
            channel="telegram",
            account_id="999",
            chat_id="-100123",
            chat_type="supergroup",
            sender_id="42",
            transport_id="album:-100123:media-group-7",
            thread_id="77",
            output_reply_to_message_id="501",
        ),
        content="task",
    )).entry
    routed = inbox.route(
        row.id,
        row.revision,
        ControlDecision("next-turn", None, "idle"),
        current_run=None,
        active_scope=(),
    )
    consumer.wake()
    await consumer.wait_idle()

    output_attempts = [request for request in attempts if request.bundle.content == "done"]
    assert len(output_attempts) == 1
    assert output_attempts[0].address.chat_id == "-100123"
    assert output_attempts[0].address.thread_id == "77"
    assert output_attempts[0].address.reply_to_message_id == "501"
    assert provider.chat_with_retry.await_count == 1
    assert loop.bus.outbound_size == 0
    settled = next(entry for entry in inbox.entries() if entry.id == routed.id)
    assert settled.state == "completed"
    assert settled.result is not None
    record = loop.run_store.load(settled.claimed_run_id)
    assert record is not None and record.status is RunStatus.COMPLETED
    if fail_result_delivery:
        assert settled.result.disposition == "uncertain"
        assert record.metadata["delivery_error_type"] == "RuntimeError"
    else:
        assert settled.result.disposition == "delivered"
        assert "delivery_error_type" not in record.metadata

    consumer.wake()
    await consumer.wait_idle()
    assert provider.chat_with_retry.await_count == 1
