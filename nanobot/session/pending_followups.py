"""Private identities and normalized records for canonical WebUI follow-ups."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from pydantic import TypeAdapter

from nanobot.bus.events import InboundMessage
from nanobot.session.persistence_metadata import SOURCE_PENDING_FOLLOWUPS_METADATA_KEY

_FOLLOWUP_PROOF_ATTRIBUTE = "_nanobot_pending_followup_proof"
_CHILD_FOLLOWUP_ATTRIBUTE = "_nanobot_child_followup_proof"
_FOLLOWUP_PROOF_TOKEN = object()
_CHILD_FOLLOWUP_PROOF_TOKEN = object()
_METADATA_ADAPTER = TypeAdapter(dict[str, object])
_CHILD_FOLLOWUP_ADAPTER = TypeAdapter(dict[str, object])
_JOURNAL_ADAPTER = TypeAdapter(list[dict[str, object]])
_MEDIA_ADAPTER = TypeAdapter(list[str])
_PRIVATE_METADATA_KEYS = frozenset({
    "remote",
    "_websocket_turn_owner",
    "_recovery_followup_id",
    "child_followup",
    "_child_followup",
})


class PendingFollowupStopError(RuntimeError):
    """Pending input could not be durably discarded before stopping its owner."""


@dataclass(frozen=True, slots=True)
class _PendingFollowupProof:
    identifier: str
    token: object


@dataclass(frozen=True, slots=True)
class PendingChildFollowup:
    """Exact parent intent and canonical child lifetime for one queued input."""

    child_id: str
    parent_key: str
    parent_store_identity: str
    parent_incarnation: str
    parent_generation: int
    parent_run_id: str
    parent_tool_event_id: str
    parent_execution_id: str
    parent_dispatch_ordinal: int
    child_key: str
    child_store_identity: str
    child_incarnation: str
    child_generation: int
    child_run_id: str
    child_state_sequence: int
    child_control_epoch: int

    def to_dict(self) -> dict[str, object]:
        return {
            "child_id": self.child_id,
            "parent_key": self.parent_key,
            "parent_store_identity": self.parent_store_identity,
            "parent_incarnation": self.parent_incarnation,
            "parent_generation": self.parent_generation,
            "parent_run_id": self.parent_run_id,
            "parent_tool_event_id": self.parent_tool_event_id,
            "parent_execution_id": self.parent_execution_id,
            "parent_dispatch_ordinal": self.parent_dispatch_ordinal,
            "child_key": self.child_key,
            "child_store_identity": self.child_store_identity,
            "child_incarnation": self.child_incarnation,
            "child_generation": self.child_generation,
            "child_run_id": self.child_run_id,
            "child_state_sequence": self.child_state_sequence,
            "child_control_epoch": self.child_control_epoch,
        }

    def same_intent(self, other: object) -> bool:
        """Compare stable identity while allowing the child run to advance."""
        return (
            type(other) is PendingChildFollowup
            and self.child_id == other.child_id
            and self.parent_key == other.parent_key
            and self.parent_store_identity == other.parent_store_identity
            and self.parent_incarnation == other.parent_incarnation
            and self.parent_generation == other.parent_generation
            and self.parent_run_id == other.parent_run_id
            and self.parent_tool_event_id == other.parent_tool_event_id
            and self.parent_execution_id == other.parent_execution_id
            and self.parent_dispatch_ordinal == other.parent_dispatch_ordinal
            and self.child_key == other.child_key
            and self.child_store_identity == other.child_store_identity
            and self.child_incarnation == other.child_incarnation
            and self.child_generation == other.child_generation
        )

    @classmethod
    def from_value(cls, value: object) -> PendingChildFollowup:
        try:
            raw = _CHILD_FOLLOWUP_ADAPTER.validate_python(value, strict=True)
        except ValueError:
            raise ValueError("Invalid canonical child follow-up record") from None
        string_fields = (
            "child_id", "parent_key", "parent_store_identity", "parent_incarnation",
            "parent_run_id", "parent_tool_event_id", "parent_execution_id", "child_key",
            "child_store_identity", "child_incarnation", "child_run_id",
        )
        integer_fields = (
            "parent_generation", "parent_dispatch_ordinal", "child_generation",
            "child_state_sequence", "child_control_epoch",
        )
        if set(raw) != set(string_fields) | set(integer_fields):
            raise ValueError("Invalid canonical child follow-up record")

        strings: dict[str, str] = {}
        for name in string_fields:
            field_value = raw.get(name)
            if not isinstance(field_value, str) or not field_value:
                raise ValueError("Invalid canonical child follow-up record")
            strings[name] = field_value

        integers: dict[str, int] = {}
        for name in integer_fields:
            field_value = raw.get(name)
            if type(field_value) is not int or field_value < 0:
                raise ValueError("Invalid canonical child follow-up record")
            integers[name] = field_value

        return cls(
            child_id=strings["child_id"],
            parent_key=strings["parent_key"],
            parent_store_identity=strings["parent_store_identity"],
            parent_incarnation=strings["parent_incarnation"],
            parent_generation=integers["parent_generation"],
            parent_run_id=strings["parent_run_id"],
            parent_tool_event_id=strings["parent_tool_event_id"],
            parent_execution_id=strings["parent_execution_id"],
            parent_dispatch_ordinal=integers["parent_dispatch_ordinal"],
            child_key=strings["child_key"],
            child_store_identity=strings["child_store_identity"],
            child_incarnation=strings["child_incarnation"],
            child_generation=integers["child_generation"],
            child_run_id=strings["child_run_id"],
            child_state_sequence=integers["child_state_sequence"],
            child_control_epoch=integers["child_control_epoch"],
        )


@dataclass(frozen=True, slots=True)
class PendingFollowupRecord:
    """A detached, JSON-safe follow-up admitted by the local WebUI runtime."""

    identifier: str
    sender_id: str
    chat_id: str
    content: str
    media: tuple[str, ...]
    metadata: dict[str, object]
    child_followup: PendingChildFollowup | None = None

    def to_dict(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": self.identifier,
            "sender_id": self.sender_id,
            "chat_id": self.chat_id,
            "content": self.content,
            "media": list(self.media),
            "metadata": deepcopy(self.metadata),
        }
        if self.child_followup is not None:
            record["child_followup"] = self.child_followup.to_dict()
        return record


@dataclass(frozen=True, slots=True)
class _PendingChildFollowupProof:
    identifier: str
    intent: PendingChildFollowup
    token: object


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


def restore_pending_child_followup(
    message: InboundMessage,
    identifier: str,
    intent: PendingChildFollowup,
) -> None:
    """Restore one typed canonical child follow-up from its durable journal row."""
    restore_pending_followup_id(message, identifier)
    setattr(message, _CHILD_FOLLOWUP_ATTRIBUTE, _PendingChildFollowupProof(
        identifier=identifier,
        intent=intent,
        token=_CHILD_FOLLOWUP_PROOF_TOKEN,
    ))


def mint_child_followup_id(
    message: InboundMessage,
    intent: PendingChildFollowup,
) -> str:
    """Bind a local child message to one durable parent dispatch identity."""
    if type(intent) is not PendingChildFollowup or message.channel != "system":
        raise ValueError("Canonical child follow-up requires a trusted local input")
    identity = "\x1f".join((
        intent.child_id,
        intent.parent_store_identity,
        intent.parent_key,
        intent.parent_incarnation,
        str(intent.parent_generation),
        intent.parent_tool_event_id,
        intent.parent_execution_id,
        str(intent.parent_dispatch_ordinal),
    ))
    identifier = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    setattr(message, _FOLLOWUP_PROOF_ATTRIBUTE, _PendingFollowupProof(
        identifier=identifier,
        token=_FOLLOWUP_PROOF_TOKEN,
    ))
    setattr(message, _CHILD_FOLLOWUP_ATTRIBUTE, _PendingChildFollowupProof(
        identifier=identifier,
        intent=intent,
        token=_CHILD_FOLLOWUP_PROOF_TOKEN,
    ))
    return identifier


def copy_pending_followup_proof(
    source: InboundMessage,
    destination: InboundMessage,
) -> None:
    """Carry the server proof across trusted dataclass copies, never from metadata."""
    proof = getattr(source, _FOLLOWUP_PROOF_ATTRIBUTE, None)
    if type(proof) is _PendingFollowupProof and proof.token is _FOLLOWUP_PROOF_TOKEN:
        setattr(destination, _FOLLOWUP_PROOF_ATTRIBUTE, proof)
    child_proof = getattr(source, _CHILD_FOLLOWUP_ATTRIBUTE, None)
    if (
        type(child_proof) is _PendingChildFollowupProof
        and child_proof.token is _CHILD_FOLLOWUP_PROOF_TOKEN
    ):
        setattr(destination, _CHILD_FOLLOWUP_ATTRIBUTE, child_proof)


def pending_child_followup(message: InboundMessage) -> PendingChildFollowup | None:
    """Read only the trusted typed child intent attached by this module."""
    proof = getattr(message, _CHILD_FOLLOWUP_ATTRIBUTE, None)
    if (
        type(proof) is not _PendingChildFollowupProof
        or proof.token is not _CHILD_FOLLOWUP_PROOF_TOKEN
        or pending_followup_id(message) != proof.identifier
    ):
        return None
    return proof.intent


def record_pending_followup(message: InboundMessage) -> PendingFollowupRecord:
    """Normalize authenticated WebUI input before it enters durable source."""
    identifier = pending_followup_id(message)
    child_followup = pending_child_followup(message)
    if (message.channel != "websocket" and child_followup is None) or identifier is None:
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
        child_followup=child_followup,
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
        child_followup_value = raw.get("child_followup")
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
            child_followup = (
                PendingChildFollowup.from_value(child_followup_value)
                if child_followup_value is not None
                else None
            )
        except ValueError:
            raise ValueError("Invalid source follow-up journal media or metadata") from None
        records.append(PendingFollowupRecord(
            identifier=identifier,
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            media=tuple(media),
            metadata=deepcopy(metadata),
            child_followup=child_followup,
        ))
    identities = [record.identifier for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate source follow-up identity")
    return records


def decode_pending_followups(value: object, session_key: str) -> list[InboundMessage]:
    """Restore WebUI rows for recovery without replaying child tool intents."""
    messages: list[InboundMessage] = []
    for record in parse_pending_followup_records(value):
        if record.child_followup is not None:
            continue
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
