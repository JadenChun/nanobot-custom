"""Opt-in, receipt-bearing text delivery contract for channel-owned transports.

This contract is deliberately separate from :class:`BaseChannel`: ordinary
channel delivery keeps its existing retry and best-effort semantics.  Callers
must explicitly resolve a capable channel and opt in to this whole-bundle API.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

_MAX_TEXT_BYTES = 512
_MAX_TIMESTAMP_BYTES = 64
_MAX_PARTS = 4096
_MAX_RECEIPT_BYTES = 64 * 1024
_DIGEST_LENGTH = 64


def _text(value: object, *, label: str, maximum: int = _MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{label} exceeds the {maximum}-byte limit")
    return value


def _digest(value: object, *, label: str) -> str:
    value = _text(value, label=label)
    if len(value) != _DIGEST_LENGTH or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


class ChannelDeliveryError(RuntimeError):
    """Base class for definite channel delivery classifications."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = _text(reason_code, label="reason_code")
        super().__init__(self.reason_code)


class ChannelDeliveryPreflightError(ChannelDeliveryError):
    """No provider operation began, so the complete bundle is confirmed unsent."""


class ChannelDeliveryRejectedError(ChannelDeliveryError):
    """The provider definitively rejected the bundle before acknowledging it."""


@dataclass(frozen=True, slots=True)
class ChannelTextPartReceipt:
    """Content-free evidence for one provider-acknowledged emitted text part."""

    index: int
    message_index: int
    part_index: int
    provider_message_id: int
    provider_chat_id: int
    provider_timestamp: str | None = None

    def __post_init__(self) -> None:
        for name in ("index", "message_index", "part_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.provider_message_id) is not int or self.provider_message_id <= 0:
            raise ValueError("provider_message_id must be a positive integer")
        if type(self.provider_chat_id) is not int or self.provider_chat_id == 0:
            raise ValueError("provider_chat_id must be a non-zero integer")
        if self.provider_timestamp is not None:
            _text(
                self.provider_timestamp,
                label="provider_timestamp",
                maximum=_MAX_TIMESTAMP_BYTES,
            )

    def to_json(self) -> dict[str, object]:
        value: dict[str, object] = {
            "index": self.index,
            "message_index": self.message_index,
            "part_index": self.part_index,
            "provider_message_id": self.provider_message_id,
            "provider_chat_id": self.provider_chat_id,
        }
        if self.provider_timestamp is not None:
            value["provider_timestamp"] = self.provider_timestamp
        return value


@dataclass(frozen=True, slots=True)
class ChannelTextReceipt:
    """Strict request-bound evidence for a complete text bundle."""

    channel: str
    target: str
    action_id: str
    request_digest: str
    parts: tuple[ChannelTextPartReceipt, ...]

    def __post_init__(self) -> None:
        _text(self.channel, label="channel")
        _text(self.target, label="target")
        _digest(self.action_id, label="action_id")
        _digest(self.request_digest, label="request_digest")
        if isinstance(self.parts, (str, bytes)) or not isinstance(cast(object, self.parts), Sequence):
            raise ValueError("parts must be an ordered sequence")
        parts = tuple(self.parts)
        if not parts or len(parts) > _MAX_PARTS:
            raise ValueError(f"parts must contain between 1 and {_MAX_PARTS} receipts")
        if any(type(part) is not ChannelTextPartReceipt for part in parts):
            raise ValueError("parts must contain only ChannelTextPartReceipt values")
        first = parts[0]
        if (first.index, first.message_index, first.part_index) != (0, 0, 0):
            raise ValueError("receipt evidence must begin at message 0 part 0")
        expected_message = 0
        expected_part = 0
        provider_acknowledgements: set[tuple[int, int]] = set()
        for expected_index, part in enumerate(parts):
            if part.index != expected_index:
                raise ValueError("receipt part indices must be unique and contiguous")
            acknowledgement = (part.provider_chat_id, part.provider_message_id)
            if acknowledgement in provider_acknowledgements:
                raise ValueError("provider acknowledgement IDs must be unique within a chat")
            provider_acknowledgements.add(acknowledgement)
            if part.message_index == expected_message:
                if part.part_index != expected_part:
                    raise ValueError("receipt part indices must be contiguous within each message")
                expected_part += 1
                continue
            if part.message_index != expected_message + 1 or part.part_index != 0:
                raise ValueError("receipt message indices must be unique and contiguous")
            expected_message += 1
            expected_part = 1
        object.__setattr__(self, "parts", parts)
        encoded = json.dumps(self.to_json(), separators=(",", ":"), sort_keys=True).encode()
        if len(encoded) > _MAX_RECEIPT_BYTES:
            raise ValueError(f"receipt exceeds the {_MAX_RECEIPT_BYTES}-byte limit")

    def to_json(self) -> dict[str, object]:
        return {
            "schema": "nanobot.channel_text_receipt.v1",
            "channel": self.channel,
            "target": self.target,
            "action_id": self.action_id,
            "request_digest": self.request_digest,
            "parts": [part.to_json() for part in self.parts],
        }


class PartialChannelTextDeliveryError(ChannelDeliveryError):
    """At least one part was acknowledged before the bundle became uncertain."""

    def __init__(self, receipt: ChannelTextReceipt) -> None:
        if type(receipt) is not ChannelTextReceipt:
            raise ValueError("partial delivery requires a strict prefix receipt")
        self.receipt = receipt
        super().__init__("partial_bundle")


@runtime_checkable
class ReceiptCapableTextChannel(Protocol):
    """Explicit opt-in capability for a no-hidden-retry whole text bundle."""

    async def send_text_with_receipt(
        self,
        target: str,
        messages: Sequence[str],
        *,
        action_id: str,
        request_digest: str,
    ) -> ChannelTextReceipt:
        """Attempt one ordered bundle and return strict provider evidence."""
        ...


def validate_receipt_binding(
    receipt: ChannelTextReceipt,
    *,
    channel: str,
    target: str,
    action_id: str,
    request_digest: str,
    message_count: int,
    require_complete: bool = True,
) -> None:
    """Fail closed unless receipt evidence is bound to the exact request."""

    if type(receipt) is not ChannelTextReceipt:
        raise ValueError("channel returned a malformed receipt")
    strict_receipt = ChannelTextReceipt(
        channel=receipt.channel,
        target=receipt.target,
        action_id=receipt.action_id,
        request_digest=receipt.request_digest,
        parts=receipt.parts,
    )
    if (
        strict_receipt.channel != channel
        or strict_receipt.target != target
        or strict_receipt.action_id != action_id
        or strict_receipt.request_digest != request_digest
    ):
        raise ValueError("channel receipt does not match the dispatch request")
    if type(message_count) is not int or message_count < 1:
        raise ValueError("message_count must be a positive integer")
    message_indices = {part.message_index for part in strict_receipt.parts}
    last_message_index = strict_receipt.parts[-1].message_index
    if last_message_index >= message_count:
        raise ValueError("channel receipt refers to an unknown dispatch message")
    if require_complete and message_indices != set(range(message_count)):
        raise ValueError("channel receipt does not cover every dispatch message")


__all__ = [
    "ChannelDeliveryError",
    "ChannelDeliveryPreflightError",
    "ChannelDeliveryRejectedError",
    "ChannelTextPartReceipt",
    "ChannelTextReceipt",
    "PartialChannelTextDeliveryError",
    "ReceiptCapableTextChannel",
    "validate_receipt_binding",
]
