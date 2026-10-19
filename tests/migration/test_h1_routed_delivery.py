"""Exact Telegram route and complete-bundle receipt contracts for H1."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram.error import BadRequest

from nanobot.agent.turn_delivery import TurnDeliveryFactory
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.routed_delivery import (
    RoutedAddress,
    RoutedAttachment,
    RoutedBundle,
    RoutedDeliveryError,
    RoutedDeliveryReceipt,
    RoutedDeliveryRequest,
    RoutedPartReceipt,
    validate_routed_receipt,
)
from nanobot.channels.telegram.runtime import TelegramChannel, TelegramConfig


def _request() -> RoutedDeliveryRequest:
    address = RoutedAddress(
        channel="telegram",
        account_id="bot-1",
        sender_id="42|alice",
        chat_type="supergroup",
        chat_id="-100123",
        thread_id="77",
        reply_to_message_id="500",
    )
    bundle = RoutedBundle(
        content="answer",
        attachments=(RoutedAttachment(
            source="/tmp/answer.png",
            filename="answer.png",
            media_type="photo",
            content_digest=hashlib.sha256(b"image-bytes").hexdigest(),
        ),),
        buttons=(("Open", "Close"),),
    )
    return RoutedDeliveryRequest(address=address, action_id="a" * 64, bundle=bundle)


def _text_request() -> RoutedDeliveryRequest:
    address = RoutedAddress(
        channel="telegram",
        account_id="1",
        sender_id="42",
        chat_type="supergroup",
        chat_id="-100123",
        thread_id="77",
        reply_to_message_id="500",
    )
    return RoutedDeliveryRequest(
        address=address,
        action_id="b" * 64,
        bundle=RoutedBundle(content="answer"),
    )


def _provider_ack(request: RoutedDeliveryRequest, *, message_id: int, chat_id: int) -> SimpleNamespace:
    address = request.address
    return SimpleNamespace(
        message_id=message_id,
        chat=SimpleNamespace(id=chat_id),
        message_thread_id=int(address.thread_id) if address.thread_id is not None else None,
        reply_to_message=SimpleNamespace(
            message_id=int(address.reply_to_message_id)
        ) if address.reply_to_message_id is not None else None,
    )


def _telegram_channel(bot: SimpleNamespace, edge: SimpleNamespace) -> TelegramChannel:
    channel = TelegramChannel(
        TelegramConfig(allow_from=["42"], allow_chats=["-100123"]), MessageBus(),
    )
    channel._bot_user_id = 1
    channel._app = SimpleNamespace(bot=bot)
    channel._app_ready.set()
    channel._busy_inbox = edge
    return channel


def _receipt(request: RoutedDeliveryRequest) -> RoutedDeliveryReceipt:
    return RoutedDeliveryReceipt(
        address=request.address,
        action_id=request.action_id,
        bundle_digest=request.bundle_digest,
        parts=(
            RoutedPartReceipt(
                index=0,
                message_index=0,
                part_index=0,
                kind="media",
                provider_message_id=100,
                provider_chat_id=-100123,
                provider_thread_id=77,
                provider_reply_to_message_id=500,
            ),
            RoutedPartReceipt(
                index=1,
                message_index=1,
                part_index=0,
                kind="text",
                provider_message_id=101,
                provider_chat_id=-100123,
                provider_thread_id=77,
                provider_reply_to_message_id=500,
            ),
        ),
    )


def test_bundle_digest_covers_text_media_bytes_and_button_fallback() -> None:
    request = _request()
    changed_media = RoutedBundle(
        content=request.bundle.content,
        attachments=(RoutedAttachment(
            source="/tmp/answer.png",
            filename="answer.png",
            media_type="photo",
            content_digest=hashlib.sha256(b"different-image").hexdigest(),
        ),),
        buttons=request.bundle.buttons,
    )
    changed_buttons = RoutedBundle(content="answer", attachments=request.bundle.attachments)

    assert request.bundle_digest != changed_media.digest
    assert request.bundle_digest != changed_buttons.digest


def test_complete_route_receipt_validates_and_wrong_route_fails_closed() -> None:
    request = _request()
    receipt = _receipt(request)

    validate_routed_receipt(request, receipt)

    wrong_address = RoutedAddress(
        channel="telegram",
        account_id="bot-1",
        sender_id="42|alice",
        chat_type="supergroup",
        chat_id="-100123",
        thread_id="78",
        reply_to_message_id="500",
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_routed_receipt(request, RoutedDeliveryReceipt(
            address=wrong_address,
            action_id=receipt.action_id,
            bundle_digest=receipt.bundle_digest,
            parts=receipt.parts,
        ))


def test_incomplete_or_malformed_provider_route_evidence_is_rejected() -> None:
    request = _request()
    receipt = _receipt(request)
    incomplete = RoutedDeliveryReceipt(
        address=receipt.address,
        action_id=receipt.action_id,
        bundle_digest=receipt.bundle_digest,
        parts=receipt.parts[:1],
    )

    with pytest.raises(ValueError, match="complete bundle order"):
        validate_routed_receipt(request, incomplete)
    missing_thread = RoutedDeliveryReceipt(
        address=receipt.address,
        action_id=receipt.action_id,
        bundle_digest=receipt.bundle_digest,
        parts=(replace(receipt.parts[0], provider_thread_id=None), receipt.parts[1]),
    )
    with pytest.raises(ValueError, match="route evidence"):
        validate_routed_receipt(request, missing_thread)


async def test_telegram_rechecks_authorization_before_each_bundle_send(tmp_path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first image")
    second.write_bytes(b"second image")
    first_attachment = RoutedAttachment(
        source=str(first),
        filename=first.name,
        media_type="photo",
        content_digest=hashlib.sha256(first.read_bytes()).hexdigest(),
    )
    second_attachment = RoutedAttachment(
        source=str(second),
        filename=second.name,
        media_type="photo",
        content_digest=hashlib.sha256(second.read_bytes()).hexdigest(),
    )
    base_request = _text_request()
    request = RoutedDeliveryRequest(
        address=base_request.address,
        action_id=base_request.action_id,
        bundle=RoutedBundle(content="", attachments=(first_attachment, second_attachment)),
    )
    allowed = True
    edge = SimpleNamespace(authorizes_address=lambda _address: allowed)

    async def send_first(**_kwargs: object) -> SimpleNamespace:
        nonlocal allowed
        allowed = False
        return _provider_ack(request, message_id=101, chat_id=-100123)

    bot = SimpleNamespace(send_photo=AsyncMock(side_effect=send_first))
    channel = _telegram_channel(bot, edge)

    with pytest.raises(RoutedDeliveryError) as raised:
        await channel.send_routed_with_receipt(request)

    assert raised.value.disposition == "uncertain"
    assert raised.value.receipt is not None
    assert len(raised.value.receipt.parts) == 1
    bot.send_photo.assert_awaited_once()


async def test_malformed_telegram_acknowledgement_is_uncertain() -> None:
    request = _text_request()
    bot = SimpleNamespace(send_message=AsyncMock(return_value=_provider_ack(
        request, message_id=101, chat_id=-100999,
    )))
    channel = _telegram_channel(bot, SimpleNamespace(authorizes_address=lambda _address: True))

    with pytest.raises(RoutedDeliveryError) as raised:
        await channel.send_routed_with_receipt(request)

    assert raised.value.disposition == "uncertain"
    bot.send_message.assert_awaited_once()


async def test_telegram_routed_text_success_uses_exact_reply_and_thread() -> None:
    request = _text_request()
    authorizations: list[RoutedAddress] = []

    def authorize(address: RoutedAddress) -> bool:
        authorizations.append(address)
        return True

    bot = SimpleNamespace(send_message=AsyncMock(return_value=_provider_ack(
        request, message_id=101, chat_id=-100123,
    )))
    channel = _telegram_channel(bot, SimpleNamespace(authorizes_address=authorize))

    receipt = await channel.send_routed_with_receipt(request)

    bot.send_message.assert_awaited_once()
    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["chat_id"] == -100123
    assert kwargs["message_thread_id"] == 77
    assert kwargs["reply_parameters"].message_id == 500
    assert kwargs["parse_mode"] == "HTML"
    assert receipt.parts[0].provider_thread_id == 77
    assert receipt.parts[0].provider_reply_to_message_id == 500
    assert authorizations == [request.address, request.address]


async def test_telegram_routed_text_html_fallback_rechecks_authorization() -> None:
    request = _text_request()
    authorizations: list[RoutedAddress] = []
    calls: list[dict[str, object]] = []

    def authorize(address: RoutedAddress) -> bool:
        authorizations.append(address)
        return True

    async def send_message(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        if len(calls) == 1:
            raise BadRequest("Can't parse entities: unmatched tag")
        return _provider_ack(request, message_id=102, chat_id=-100123)

    bot = SimpleNamespace(send_message=AsyncMock(side_effect=send_message))
    channel = _telegram_channel(bot, SimpleNamespace(authorizes_address=authorize))

    receipt = await channel.send_routed_with_receipt(request)

    assert len(calls) == 2
    assert calls[0]["parse_mode"] == "HTML"
    assert "parse_mode" not in calls[1]
    assert all(call["chat_id"] == -100123 for call in calls)
    assert all(call["message_thread_id"] == 77 for call in calls)
    assert all(call["reply_parameters"].message_id == 500 for call in calls)
    assert receipt.parts[0].provider_thread_id == 77
    assert receipt.parts[0].provider_reply_to_message_id == 500
    assert authorizations == [request.address] * 3


async def test_trusted_output_hook_delivers_completion_once_without_legacy_publish() -> None:
    bus = MessageBus()
    message = InboundMessage(channel="telegram", sender_id="42", chat_id="-100123", content="hello")
    response = OutboundMessage(channel="telegram", chat_id="-100123", content="answer")
    delivered: list[OutboundMessage] = []

    async def output_hook(value: OutboundMessage) -> None:
        delivered.append(value)

    delivery = TurnDeliveryFactory(bus).create(
        message,
        message.session_key,
        suppress_channel_output=True,
        trusted_output_hook=output_hook,
    )

    await delivery.complete(response, publish_completion=False)

    assert delivered == [response]
    assert bus.outbound_size == 0


async def test_trusted_output_hook_delivers_error_without_legacy_publish() -> None:
    bus = MessageBus()
    message = InboundMessage(channel="telegram", sender_id="42", chat_id="-100123", content="hello")
    delivered: list[OutboundMessage] = []

    async def output_hook(value: OutboundMessage) -> None:
        delivered.append(value)

    delivery = TurnDeliveryFactory(bus).create(
        message,
        message.session_key,
        suppress_channel_output=True,
        trusted_output_hook=output_hook,
    )

    await delivery.fail(publish_completion=False)

    assert len(delivered) == 1
    assert delivered[0].content == "Sorry, I encountered an error."
    assert bus.outbound_size == 0
