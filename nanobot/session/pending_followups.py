"""Private identities and normalized records for canonical WebUI follow-ups."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast
from uuid import uuid4

from pydantic import TypeAdapter

from nanobot.bus.events import InboundMessage
from nanobot.session.persistence_metadata import SOURCE_PENDING_FOLLOWUPS_METADATA_KEY

if TYPE_CHECKING:
    from nanobot.session.source_store import SourceState

BACKGROUND_RESULT_INBOUND_METADATA_KEY = "background_child_result"
MAX_BACKGROUND_RESULT_CHARS = 16000

_FOLLOWUP_PROOF_ATTRIBUTE = "_nanobot_pending_followup_proof"
_CHILD_FOLLOWUP_ATTRIBUTE = "_nanobot_child_followup_proof"
_BACKGROUND_RESULT_ATTRIBUTE = "_nanobot_background_result_proof"
_FOLLOWUP_PROOF_TOKEN = object()
_CHILD_FOLLOWUP_PROOF_TOKEN = object()
_BACKGROUND_RESULT_PROOF_TOKEN = object()
_METADATA_ADAPTER = TypeAdapter(dict[str, object])
_CHILD_FOLLOWUP_ADAPTER = TypeAdapter(dict[str, object])
_BACKGROUND_RESULT_ADAPTER = TypeAdapter(dict[str, object])
_JOURNAL_ADAPTER = TypeAdapter(list[dict[str, object]])
_MEDIA_ADAPTER = TypeAdapter(list[str])
_PRIVATE_METADATA_KEYS = frozenset({
    "remote",
    "_websocket_turn_owner",
    "_recovery_followup_id",
    "child_followup",
    "_child_followup",
    "background_result",
    "_background_result",
    "_stop_client",
    "_stop_expectation",
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
class PendingBackgroundResult:
    """One bounded child result bound to an exact parent lifetime and route."""

    outbox_id: str
    parent_key: str
    parent_store_identity: str
    parent_incarnation: str
    parent_generation: int
    parent_run_id: str
    parent_tool_event_id: str
    parent_execution_id: str
    parent_dispatch_ordinal: int
    child_id: str
    child_key: str
    child_run_id: str
    child_state_sequence: int
    child_control_epoch: int
    route_channel: str
    route_chat_id: str
    result: str
    stop_reason: str
    outcome: str
    created_at: float
    state: Literal["pending", "admitted", "stale"] = "pending"
    journal_id: str | None = None
    stale_reason: str | None = None
    published: bool = False

    def targets(self, state: SourceState) -> bool:
        return (
            self.parent_key == state.session_key
            and self.parent_store_identity == state.store_identity
            and self.parent_incarnation == state.incarnation
            and self.parent_generation == state.generation
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "outbox_id": self.outbox_id,
            "parent_key": self.parent_key,
            "parent_store_identity": self.parent_store_identity,
            "parent_incarnation": self.parent_incarnation,
            "parent_generation": self.parent_generation,
            "parent_run_id": self.parent_run_id,
            "parent_tool_event_id": self.parent_tool_event_id,
            "parent_execution_id": self.parent_execution_id,
            "parent_dispatch_ordinal": self.parent_dispatch_ordinal,
            "child_id": self.child_id,
            "child_key": self.child_key,
            "child_run_id": self.child_run_id,
            "child_state_sequence": self.child_state_sequence,
            "child_control_epoch": self.child_control_epoch,
            "route_channel": self.route_channel,
            "route_chat_id": self.route_chat_id,
            "result": self.result,
            "stop_reason": self.stop_reason,
            "outcome": self.outcome,
            "created_at": self.created_at,
            "state": self.state,
            "journal_id": self.journal_id,
            "stale_reason": self.stale_reason,
            "published": self.published,
        }

    def same_result(self, other: object) -> bool:
        """Compare the exact delivery intent without its reconciliation state."""
        if type(other) is not PendingBackgroundResult:
            return False
        ignored = {"state", "journal_id", "stale_reason", "created_at", "published"}
        return {
            key: value for key, value in self.to_dict().items() if key not in ignored
        } == {
            key: value for key, value in other.to_dict().items() if key not in ignored
        }

    @classmethod
    def from_value(cls, value: object) -> PendingBackgroundResult:
        try:
            raw = _BACKGROUND_RESULT_ADAPTER.validate_python(value, strict=True)
        except ValueError:
            raise ValueError("Invalid background child result record") from None
        required = {
            "outbox_id",
            "parent_key",
            "parent_store_identity",
            "parent_incarnation",
            "parent_generation",
            "parent_run_id",
            "parent_tool_event_id",
            "parent_execution_id",
            "parent_dispatch_ordinal",
            "child_id",
            "child_key",
            "child_run_id",
            "child_state_sequence",
            "child_control_epoch",
            "route_channel",
            "route_chat_id",
            "result",
            "stop_reason",
            "outcome",
            "created_at",
            "state",
        }
        optional = {"journal_id", "stale_reason", "published"}
        if not required <= set(raw) or set(raw) - required - optional:
            raise ValueError("Invalid background child result record")
        string_fields = (
            "outbox_id",
            "parent_key",
            "parent_store_identity",
            "parent_incarnation",
            "parent_run_id",
            "parent_tool_event_id",
            "parent_execution_id",
            "child_id",
            "child_key",
            "child_run_id",
            "route_channel",
            "route_chat_id",
            "stop_reason",
            "outcome",
        )
        strings: dict[str, str] = {}
        for name in string_fields:
            field_value = raw.get(name)
            if not isinstance(field_value, str) or not field_value:
                raise ValueError("Invalid background child result record")
            if name in {"outbox_id", "child_id"} and (
                len(field_value) != 32
                or any(character not in "0123456789abcdef" for character in field_value)
            ):
                raise ValueError("Invalid background child result record")
            strings[name] = field_value
        integer_fields = (
            "parent_generation",
            "parent_dispatch_ordinal",
            "child_state_sequence",
            "child_control_epoch",
        )
        integers: dict[str, int] = {}
        for name in integer_fields:
            field_value = raw.get(name)
            if type(field_value) is not int or field_value < 0:
                raise ValueError("Invalid background child result record")
            integers[name] = field_value
        result = raw.get("result")
        if not isinstance(result, str) or len(result) > MAX_BACKGROUND_RESULT_CHARS:
            raise ValueError("Invalid background child result record")
        created_at = raw.get("created_at")
        if (
            isinstance(created_at, bool)
            or not isinstance(created_at, (int, float))
            or not math.isfinite(created_at)
            or created_at < 0
        ):
            raise ValueError("Invalid background child result record")
        state = raw.get("state")
        if state not in {"pending", "admitted", "stale"}:
            raise ValueError("Invalid background child result record")
        journal_id = raw.get("journal_id")
        if journal_id is not None and (
            not isinstance(journal_id, str)
            or len(journal_id) != 32
            or any(character not in "0123456789abcdef" for character in journal_id)
        ):
            raise ValueError("Invalid background child result record")
        stale_reason = raw.get("stale_reason")
        if stale_reason is not None and (
            not isinstance(stale_reason, str) or not stale_reason
        ):
            raise ValueError("Invalid background child result record")
        published = raw.get("published", False)
        if type(published) is not bool:
            raise ValueError("Invalid background child result record")
        return cls(
            outbox_id=strings["outbox_id"],
            parent_key=strings["parent_key"],
            parent_store_identity=strings["parent_store_identity"],
            parent_incarnation=strings["parent_incarnation"],
            parent_generation=integers["parent_generation"],
            parent_run_id=strings["parent_run_id"],
            parent_tool_event_id=strings["parent_tool_event_id"],
            parent_execution_id=strings["parent_execution_id"],
            parent_dispatch_ordinal=integers["parent_dispatch_ordinal"],
            child_id=strings["child_id"],
            child_key=strings["child_key"],
            child_run_id=strings["child_run_id"],
            child_state_sequence=integers["child_state_sequence"],
            child_control_epoch=integers["child_control_epoch"],
            route_channel=strings["route_channel"],
            route_chat_id=strings["route_chat_id"],
            result=result,
            stop_reason=strings["stop_reason"],
            outcome=strings["outcome"],
            created_at=float(created_at),
            state=cast(Literal["pending", "admitted", "stale"], state),
            journal_id=journal_id,
            stale_reason=stale_reason,
            published=published,
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
    background_result: PendingBackgroundResult | None = None

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
        if self.background_result is not None:
            record["background_result"] = self.background_result.to_dict()
        return record


@dataclass(frozen=True, slots=True)
class _PendingChildFollowupProof:
    identifier: str
    intent: PendingChildFollowup
    token: object


@dataclass(frozen=True, slots=True)
class _PendingBackgroundResultProof:
    identifier: str
    intent: PendingBackgroundResult
    token: object


def background_result_id(
    *,
    child_id: str,
    child_run_id: str,
    parent_key: str,
    parent_store_identity: str,
    parent_incarnation: str,
    parent_generation: int,
    parent_tool_event_id: str,
    parent_execution_id: str,
    parent_dispatch_ordinal: int,
) -> str:
    """Return one stable outbox identity for an exact child result delivery."""
    identity = "\x1f".join((
        child_id,
        child_run_id,
        parent_key,
        parent_store_identity,
        parent_incarnation,
        str(parent_generation),
        parent_tool_event_id,
        parent_execution_id,
        str(parent_dispatch_ordinal),
    ))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


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


def mint_background_result_id(
    message: InboundMessage,
    intent: PendingBackgroundResult,
) -> str:
    """Bind one local child result to its stable durable outbox identity."""
    if type(intent) is not PendingBackgroundResult or message.channel != intent.route_channel:
        raise ValueError("Background child result requires its captured route")
    setattr(message, _FOLLOWUP_PROOF_ATTRIBUTE, _PendingFollowupProof(
        identifier=intent.outbox_id,
        token=_FOLLOWUP_PROOF_TOKEN,
    ))
    setattr(message, _BACKGROUND_RESULT_ATTRIBUTE, _PendingBackgroundResultProof(
        identifier=intent.outbox_id,
        intent=intent,
        token=_BACKGROUND_RESULT_PROOF_TOKEN,
    ))
    return intent.outbox_id


def restore_pending_background_result(
    message: InboundMessage,
    identifier: str,
    intent: PendingBackgroundResult,
) -> None:
    """Restore one durable background result row during recovery."""
    restore_pending_followup_id(message, identifier)
    setattr(message, _BACKGROUND_RESULT_ATTRIBUTE, _PendingBackgroundResultProof(
        identifier=identifier,
        intent=intent,
        token=_BACKGROUND_RESULT_PROOF_TOKEN,
    ))


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
    background_proof = getattr(source, _BACKGROUND_RESULT_ATTRIBUTE, None)
    if (
        type(background_proof) is _PendingBackgroundResultProof
        and background_proof.token is _BACKGROUND_RESULT_PROOF_TOKEN
    ):
        setattr(destination, _BACKGROUND_RESULT_ATTRIBUTE, background_proof)


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


def pending_background_result(message: InboundMessage) -> PendingBackgroundResult | None:
    """Read only the trusted typed background result attached by this module."""
    proof = getattr(message, _BACKGROUND_RESULT_ATTRIBUTE, None)
    if (
        type(proof) is not _PendingBackgroundResultProof
        or proof.token is not _BACKGROUND_RESULT_PROOF_TOKEN
        or pending_followup_id(message) != proof.identifier
    ):
        return None
    return proof.intent


def record_pending_followup(message: InboundMessage) -> PendingFollowupRecord:
    """Normalize authenticated WebUI input before it enters durable source."""
    identifier = pending_followup_id(message)
    child_followup = pending_child_followup(message)
    background_result = pending_background_result(message)
    if (
        message.channel != "websocket"
        and child_followup is None
        and background_result is None
    ) or identifier is None:
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
        background_result=background_result,
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
        background_result_value = raw.get("background_result")
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
            background_result = (
                PendingBackgroundResult.from_value(background_result_value)
                if background_result_value is not None
                else None
            )
            if child_followup is not None and background_result is not None:
                raise ValueError("Ambiguous source follow-up intent")
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
            background_result=background_result,
        ))
    identities = [record.identifier for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate source follow-up identity")
    return records


def background_result_content(intent: PendingBackgroundResult) -> str:
    """Return the bounded user-visible result line for one background child."""
    label = (
        "completed"
        if intent.outcome == "completed"
        else f"ended with {intent.stop_reason}"
    )
    text = intent.result.strip()
    if not text:
        return f"Background child {intent.child_id} {label}."
    return f"Background child {intent.child_id} {label}.\n\n{text}"


def decode_pending_followups(
    value: object, session_key: str, *, source_state: SourceState | None = None,
) -> list[InboundMessage]:
    """Restore WebUI rows for recovery without replaying child tool intents."""
    messages: list[InboundMessage] = []
    for record in parse_pending_followup_records(value):
        if record.child_followup is not None:
            continue
        if record.background_result is not None:
            intent = record.background_result
            if source_state is not None and not intent.targets(source_state):
                continue
            message = InboundMessage(
                channel=intent.route_channel,
                sender_id="subagent",
                chat_id=intent.route_chat_id,
                content=background_result_content(intent),
                metadata={
                    BACKGROUND_RESULT_INBOUND_METADATA_KEY: intent.outbox_id,
                    "background_child_id": intent.child_id,
                },
                session_key_override=session_key,
                require_existing_session=True,
                source_state=source_state,
            )
            restore_pending_background_result(message, record.identifier, intent)
            messages.append(message)
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
