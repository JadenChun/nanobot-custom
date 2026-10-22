"""Dormant E3 sender that calls an explicitly resolved receipt-capable channel."""

from __future__ import annotations

from typing import Protocol

from nanobot.channels.base import BaseChannel
from nanobot.channels.delivery_contract import (
    ChannelDeliveryPreflightError,
    ChannelDeliveryRejectedError,
    PartialChannelTextDeliveryError,
    ReceiptCapableTextChannel,
    ThreadReceiptCapableTextChannel,
    validate_receipt_binding,
)
from nanobot.cron.delivery_runtime import (
    ConfirmedDelivered,
    ConfirmedNotSent,
    DispatchRequest,
    PartialBundleOutcome,
    PermanentNotSent,
    SenderResult,
)


class ExactChannelResolver(Protocol):
    def __call__(self, name: str) -> BaseChannel | object | None: ...


class ExactChannelAuthority(Protocol):
    def get_channel(self, name: str) -> BaseChannel | None: ...


class CronChannelSender:
    """Resolve exactly one frozen channel name and invoke only its receipt API."""

    def __init__(self, authority: ExactChannelResolver | ExactChannelAuthority) -> None:
        if not callable(authority) and not callable(getattr(authority, "get_channel", None)):
            raise TypeError("authority must be an exact channel resolver")
        self._authority = authority

    def _resolve(self, name: str) -> object | None:
        if callable(self._authority):
            return self._authority(name)
        return self._authority.get_channel(name)

    async def send(self, request: DispatchRequest) -> SenderResult | PartialBundleOutcome:
        channel = self._resolve(request.channel)
        if channel is None or not isinstance(channel, ReceiptCapableTextChannel):
            return PermanentNotSent("unsupported_receipt_transport")
        messages = tuple(message.content for message in request.messages)
        try:
            if request.message_thread_id is not None:
                if not isinstance(channel, ThreadReceiptCapableTextChannel):
                    return PermanentNotSent("unsupported_topic_transport")
                receipt = await channel.send_thread_text_with_receipt(
                    request.to, messages, message_thread_id=request.message_thread_id,
                    action_id=request.action_id, request_digest=request.request_digest,
                )
            else:
                receipt = await channel.send_text_with_receipt(
                    request.to, messages, action_id=request.action_id, request_digest=request.request_digest,
                )
        except ChannelDeliveryPreflightError as exc:
            return ConfirmedNotSent(exc.reason_code)
        except ChannelDeliveryRejectedError as exc:
            return PermanentNotSent(exc.reason_code)
        except PartialChannelTextDeliveryError as exc:
            validate_receipt_binding(
                exc.receipt,
                channel=request.channel,
                target=request.to,
                action_id=request.action_id,
                request_digest=request.request_digest,
                message_count=len(messages),
                require_complete=False,
            )
            return PartialBundleOutcome()
        validate_receipt_binding(
            receipt,
            channel=request.channel,
            target=request.to,
            action_id=request.action_id,
            request_digest=request.request_digest,
            message_count=len(messages),
        )
        return ConfirmedDelivered(receipt.to_json())


__all__ = ["CronChannelSender", "ExactChannelAuthority", "ExactChannelResolver"]
