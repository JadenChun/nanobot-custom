"""Private identities and normalized records for canonical WebUI follow-ups."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from pydantic import TypeAdapter

from nanobot.bus.events import InboundMessage
from nanobot.session.persistence_metadata import SOURCE_PENDING_FOLLOWUPS_METADATA_KEY

_FOLLOWUP_PROOF_ATTRIBUTE = "_nanobot_pending_followup_proof"
_FOLLOWUP_PROOF_TOKEN = object()
_METADATA_ADAPTER = TypeAdapter(dict[str, object])
_JOURNAL_ADAPTER = TypeAdapter(list[dict[str, object]])
_MEDIA_ADAPTER = TypeAdapter(list[str])
_PRIVATE_METADATA_KEYS = frozenset({
    "remote",
    "_websocket_turn_owner",
    "_recovery_followup_id",
})


class PendingFollowupStopError(RuntimeError):
    """Pending input could not be durably discarded before stopping its owner."""


@dataclass(frozen=True, slots=True)
class _PendingFollowupProof:
    identifier: str
    token: object


@dataclass(frozen=True, slots=True)
class PendingFollowupRecord:
    """A detached, JSON-safe follow-up admitted by the local WebUI runtime."""

    identifier: str
    sender_id: str
    chat_id: str
    content: str
    media: tuple[str, ...]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "sender_id": self.sender_id,
            "chat_id": self.chat_id,
            "content": self.content,
            "media": list(self.media),
            "metadata": deepcopy(self.metadata),
        }


def mint_pending_followup_id(message: InboundMessage) -> str:
    """Attach a server-minted, object-bound ID, ignoring every wire metadata ID."""
    proof = getattr(message, _FOLLOWUP_PROOF_ATTRIBUTE, None)
    if type(proof) is _PendingFollowupProof and proof.token is _FOLLOWUP_PROOF_TOKEN:
        return proof.identifier
    identifier = uuid4().hex
    setattr(message, _FOLLOWUP_PROOF_ATTRIBUTE, _PendingFollowupProof(
        identifier=identifier,
        token=_FOLLOWUP_PROOF_TOKEN,
    ))
    return identifier


def pending_followup_id(message: InboundMessage) -> str | None:
    """Read only the in-process proof attached by this module."""
    proof = getattr(message, _FOLLOWUP_PROOF_ATTRIBUTE, None)
    if type(proof) is not _PendingFollowupProof or proof.token is not _FOLLOWUP_PROOF_TOKEN:
        return None
    return proof.identifier


def restore_pending_followup_id(message: InboundMessage, identifier: str) -> None:
    """Restore a journal identity while decoding a record during recovery."""
    if len(identifier) != 32 or any(character not in "0123456789abcdef" for character in identifier):
        raise ValueError("Invalid pending follow-up identity")
    setattr(message, _FOLLOWUP_PROOF_ATTRIBUTE, _PendingFollowupProof(
        identifier=identifier,
        token=_FOLLOWUP_PROOF_TOKEN,
    ))


def copy_pending_followup_proof(
    source: InboundMessage,
    destination: InboundMessage,
) -> None:
    """Carry the server proof across trusted dataclass copies, never from metadata."""
    proof = getattr(source, _FOLLOWUP_PROOF_ATTRIBUTE, None)
    if type(proof) is _PendingFollowupProof and proof.token is _FOLLOWUP_PROOF_TOKEN:
        setattr(destination, _FOLLOWUP_PROOF_ATTRIBUTE, proof)


def record_pending_followup(message: InboundMessage) -> PendingFollowupRecord:
    """Normalize authenticated WebUI input before it enters durable source."""
    identifier = pending_followup_id(message)
    if message.channel != "websocket" or identifier is None:
        raise ValueError("WebUI follow-up has no server identity")
    try:
        metadata_value = _METADATA_ADAPTER.validate_json(json.dumps(message.metadata), strict=True)
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ValueError("WebUI follow-up metadata is not serializable") from None
    metadata: dict[str, object] = {}
    for name, value in metadata_value.items():
        if name not in _PRIVATE_METADATA_KEYS and name != SOURCE_PENDING_FOLLOWUPS_METADATA_KEY:
            metadata[name] = value
    return PendingFollowupRecord(
        identifier=identifier,
        sender_id=message.sender_id,
        chat_id=message.chat_id,
        content=message.content,
        media=tuple(message.media or ()),
        metadata=metadata,
    )


def parse_pending_followup_records(value: object) -> list[PendingFollowupRecord]:
    """Decode the reserved source value, failing closed on any malformed present row."""
    try:
        rows = _JOURNAL_ADAPTER.validate_python(value, strict=True)
    except ValueError:
        raise ValueError("Invalid source follow-up journal") from None
    records: list[PendingFollowupRecord] = []
    for raw in rows:
        identifier = raw.get("id")
        sender_id = raw.get("sender_id")
        chat_id = raw.get("chat_id")
        content = raw.get("content")
        media_value = raw.get("media")
        metadata_value = raw.get("metadata")
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(sender_id, str)
            or not sender_id
            or not isinstance(chat_id, str)
            or not chat_id
            or not isinstance(content, str)
            or len(identifier) != 32
            or any(character not in "0123456789abcdef" for character in identifier)
        ):
            raise ValueError("Invalid source follow-up journal row")
        try:
            media = _MEDIA_ADAPTER.validate_python(media_value, strict=True)
            metadata = _METADATA_ADAPTER.validate_python(metadata_value, strict=True)
        except ValueError:
            raise ValueError("Invalid source follow-up journal media or metadata") from None
        records.append(PendingFollowupRecord(
            identifier=identifier,
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            media=tuple(media),
            metadata=deepcopy(metadata),
        ))
    identities = [record.identifier for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate source follow-up identity")
    return records


def decode_pending_followups(value: object, session_key: str) -> list[InboundMessage]:
    """Validate persisted rows and restore internal proofs for canonical recovery."""
    messages: list[InboundMessage] = []
    for record in parse_pending_followup_records(value):
        message = InboundMessage(
            channel="websocket",
            sender_id=record.sender_id,
            chat_id=record.chat_id,
            content=record.content,
            media=list(record.media),
            metadata=deepcopy(record.metadata),
            session_key_override=session_key,
            require_existing_session=True,
        )
        restore_pending_followup_id(message, record.identifier)
        messages.append(message)
    return messages
