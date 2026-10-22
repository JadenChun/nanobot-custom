"""Exact, opt-in destination-bound delivery contracts for authenticated transports."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal

RoutedPartKind = Literal["media", "text"]
DeliveryDisposition = Literal["definitely-unsent", "uncertain"]
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MAX_TEXT_BYTES = 16 * 1024 * 1024
_MAX_PARTS = 4096


def _nonempty(value: object, name: str, *, maximum: int = 1024) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} must be a non-empty bounded string")
    return value


def _sha256(value: object, name: str) -> str:
    digest = _nonempty(value, name, maximum=64)
    if _DIGEST.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _is_string(value: object) -> bool:
    return isinstance(value, str)


@dataclass(frozen=True, slots=True)
class RoutedAddress:
    """The authenticated sender and exact Telegram output route."""

    channel: str
    account_id: str
    sender_id: str
    chat_type: str
    chat_id: str
    thread_id: str | None = None
    reply_to_message_id: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.channel, "channel", maximum=64)
        _nonempty(self.account_id, "account_id", maximum=128)
        _nonempty(self.sender_id, "sender_id", maximum=256)
        _nonempty(self.chat_id, "chat_id", maximum=256)
        try:
            numeric_chat_id = int(self.chat_id)
        except ValueError as exc:
            raise ValueError("chat_id must be a canonical Telegram identifier") from exc
        if numeric_chat_id == 0 or str(numeric_chat_id) != self.chat_id:
            raise ValueError("chat_id must be a canonical Telegram identifier")
        if self.chat_type not in {"private", "group", "supergroup", "channel"}:
            raise ValueError("chat_type is not a Telegram chat type")
        for name in ("thread_id", "reply_to_message_id"):
            value = getattr(self, name)
            if value is not None:
                _nonempty(value, name, maximum=64)
                if not value.isdecimal() or int(value) <= 0 or str(int(value)) != value:
                    raise ValueError(f"{name} must be a canonical positive Telegram identifier")


@dataclass(frozen=True, slots=True)
class RoutedAttachment:
    """A preflighted attachment reference bound to its immutable content digest."""

    source: str
    filename: str
    media_type: Literal["photo", "video", "voice", "audio", "document"]
    content_digest: str | None = None
    reference_digest: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.source, "attachment source", maximum=4096)
        _nonempty(self.filename, "attachment filename", maximum=512)
        if self.media_type not in {"photo", "video", "voice", "audio", "document"}:
            raise ValueError("unsupported routed attachment type")
        if (self.content_digest is None) == (self.reference_digest is None):
            raise ValueError("attachment requires exactly one local-content or reference digest")
        if self.content_digest is not None:
            _sha256(self.content_digest, "attachment content digest")
        if self.reference_digest is not None:
            _sha256(self.reference_digest, "attachment reference digest")


@dataclass(frozen=True, slots=True)
class RoutedBundle:
    """Complete text, media, and button-label fallback to be emitted as one action."""

    content: str
    attachments: tuple[RoutedAttachment, ...] = ()
    buttons: tuple[tuple[str, ...], ...] = ()
    text_parts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _is_string(self.content) or len(self.content.encode("utf-8")) > _MAX_TEXT_BYTES:
            raise ValueError("routed content exceeds its byte limit")
        if type(self.attachments) is not tuple or len(self.attachments) > _MAX_PARTS:
            raise ValueError("routed attachments must be a bounded tuple")
        if any(type(value) is not RoutedAttachment for value in self.attachments):
            raise ValueError("routed attachments must be validated values")
        if type(self.buttons) is not tuple or len(self.buttons) > 64:
            raise ValueError("routed buttons must be a bounded tuple")
        for row in self.buttons:
            if type(row) is not tuple or not row or len(row) > 16:
                raise ValueError("routed button rows must be non-empty bounded tuples")
            for label in row:
                _nonempty(label, "button label", maximum=256)
        if type(self.text_parts) is not tuple or len(self.text_parts) > _MAX_PARTS:
            raise ValueError("routed text parts must be a bounded tuple")
        if any(not _is_string(value) or not value or len(value.encode("utf-8")) > 16 * 1024
               for value in self.text_parts):
            raise ValueError("routed text parts must be non-empty bounded strings")
        if self.text_parts and not self.rendered_text:
            raise ValueError("empty routed text cannot contain emitted text parts")

    @property
    def rendered_text(self) -> str:
        if not self.buttons:
            return self.content
        labels = "\n".join(" · ".join(row) for row in self.buttons)
        return f"{self.content}\n\n{labels}" if self.content else labels

    @property
    def emitted_text_parts(self) -> tuple[str, ...]:
        if self.text_parts:
            return self.text_parts
        return (self.rendered_text,) if self.rendered_text else ()

    @property
    def digest(self) -> str:
        canonical = json.dumps(
            {
                "content": self.content,
                "attachments": [
                    {
                        "source": attachment.source,
                        "filename": attachment.filename,
                        "media_type": attachment.media_type,
                        "content_digest": attachment.content_digest,
                        "reference_digest": attachment.reference_digest,
                    }
                    for attachment in self.attachments
                ],
                "buttons": [list(row) for row in self.buttons],
                "text_parts": list(self.emitted_text_parts),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class RoutedDeliveryRequest:
    address: RoutedAddress
    action_id: str
    bundle: RoutedBundle

    def __post_init__(self) -> None:
        if type(self.address) is not RoutedAddress or type(self.bundle) is not RoutedBundle:
            raise ValueError("routed request requires validated address and bundle values")
        _sha256(self.action_id, "action_id")

    @property
    def bundle_digest(self) -> str:
        return self.bundle.digest


@dataclass(frozen=True, slots=True)
class RoutedPartReceipt:
    """Provider acknowledgement for one text chunk or attachment."""

    index: int
    message_index: int
    part_index: int
    kind: RoutedPartKind
    provider_message_id: int
    provider_chat_id: int
    provider_thread_id: int | None
    provider_reply_to_message_id: int | None

    def __post_init__(self) -> None:
        for name in ("index", "message_index", "part_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.kind not in {"media", "text"}:
            raise ValueError("unsupported routed receipt part")
        if type(self.provider_message_id) is not int or self.provider_message_id <= 0:
            raise ValueError("provider message ID must be a positive integer")
        if type(self.provider_chat_id) is not int or self.provider_chat_id == 0:
            raise ValueError("provider chat ID must be a non-zero integer")
        for name in ("provider_thread_id", "provider_reply_to_message_id"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer when present")


@dataclass(frozen=True, slots=True)
class RoutedDeliveryReceipt:
    address: RoutedAddress
    action_id: str
    bundle_digest: str
    parts: tuple[RoutedPartReceipt, ...]

    def __post_init__(self) -> None:
        if type(self.address) is not RoutedAddress:
            raise ValueError("routed receipt requires a validated address")
        _sha256(self.action_id, "action_id")
        _sha256(self.bundle_digest, "bundle digest")
        if type(self.parts) is not tuple or not 1 <= len(self.parts) <= _MAX_PARTS:
            raise ValueError("routed receipt requires a bounded non-empty part tuple")
        if any(type(part) is not RoutedPartReceipt for part in self.parts):
            raise ValueError("routed receipt contains an invalid provider part")
        if tuple(part.index for part in self.parts) != tuple(range(len(self.parts))):
            raise ValueError("routed receipt part indices must be contiguous")


class RoutedDeliveryError(RuntimeError):
    """Classified failure from an exact-route provider attempt."""

    disposition: DeliveryDisposition

    def __init__(
        self,
        disposition: DeliveryDisposition,
        reason: str,
        *,
        receipt: RoutedDeliveryReceipt | None = None,
    ) -> None:
        if disposition not in {"definitely-unsent", "uncertain"}:
            raise ValueError("invalid routed delivery disposition")
        if receipt is not None and type(receipt) is not RoutedDeliveryReceipt:
            raise ValueError("partial delivery evidence must be a routed receipt")
        self.disposition = disposition
        self.reason = _nonempty(reason, "delivery reason", maximum=128)
        self.receipt = receipt
        super().__init__(self.reason)


def validate_routed_receipt(
    request: RoutedDeliveryRequest,
    receipt: RoutedDeliveryReceipt,
) -> None:
    """Require exact request binding and complete media/text acknowledgement evidence."""

    if type(request) is not RoutedDeliveryRequest or type(receipt) is not RoutedDeliveryReceipt:
        raise ValueError("routed delivery returned malformed receipt evidence")
    if (receipt.address != request.address or receipt.action_id != request.action_id
            or receipt.bundle_digest != request.bundle_digest):
        raise ValueError("routed receipt does not match the requested destination and action")

    expected_media = len(request.bundle.attachments)
    text_index = expected_media
    expected_last = text_index if request.bundle.rendered_text else expected_media - 1
    if not request.bundle.rendered_text and not expected_media:
        raise ValueError("empty routed bundle has no deliverable parts")
    text_parts = [part for part in receipt.parts if part.kind == "text"]
    expected_parts = tuple(
        [("media", index, 0) for index in range(expected_media)]
        + ([
            ("text", text_index, part_index)
            for part_index in range(len(request.bundle.emitted_text_parts))
        ] if request.bundle.rendered_text else [])
    )
    if tuple((part.kind, part.message_index, part.part_index) for part in receipt.parts) != expected_parts:
        raise ValueError("receipt parts are not in complete bundle order")
    if receipt.parts[-1].message_index != expected_last:
        raise ValueError("receipt does not cover every bundle part")
    media_parts = [part for part in receipt.parts if part.kind == "media"]
    if len(media_parts) != expected_media:
        raise ValueError("receipt does not cover every bundle part")
    if any((part.message_index, part.part_index) != (index, 0)
           for index, part in enumerate(media_parts)):
        raise ValueError("receipt media sequence is incomplete")
    if request.bundle.rendered_text:
        if len(text_parts) != len(request.bundle.emitted_text_parts) or any(
            part.message_index != text_index for part in text_parts
        ):
            raise ValueError("receipt does not cover every bundle part")
        if tuple(part.part_index for part in text_parts) != tuple(range(len(text_parts))):
            raise ValueError("receipt text chunks are incomplete")
    if any(part.provider_chat_id != int(request.address.chat_id) for part in receipt.parts):
        raise ValueError("provider receipt contains a different chat destination")
    acknowledgements = tuple((part.provider_chat_id, part.provider_message_id) for part in receipt.parts)
    if len(set(acknowledgements)) != len(acknowledgements):
        raise ValueError("provider receipt contains duplicate acknowledgements")
    expected_thread = int(request.address.thread_id) if request.address.thread_id else None
    expected_reply = int(request.address.reply_to_message_id) if request.address.reply_to_message_id else None
    if any(part.provider_thread_id != expected_thread
           or part.provider_reply_to_message_id != expected_reply for part in receipt.parts):
        raise ValueError("provider receipt contains incomplete route evidence")


__all__ = [
    "DeliveryDisposition",
    "RoutedAddress",
    "RoutedAttachment",
    "RoutedBundle",
    "RoutedDeliveryError",
    "RoutedDeliveryReceipt",
    "RoutedDeliveryRequest",
    "RoutedPartReceipt",
    "validate_routed_receipt",
]
