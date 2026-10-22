"""Dormant H1 admission ledger and reply intents on the existing session owner.

No model, transport or task is dispatched here. A claim is not a dispatch;
the execution adapter must commit begin_dispatch before source/effect work.
Source-event binding and owning recovery remain separate integration gates.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nanobot.agent.busy_control import (
    ControlDecision,
    ControlIntent,
    ControlReport,
    ControlSnapshot,
)
from nanobot.config.schema import BusyInboxConfig
from nanobot.session.manager import SessionManager
from nanobot.session.persistence_metadata import BUSY_INBOX_METADATA_KEY, BUSY_RECOVERY_KEY_PREFIX

EntryState = Literal[
    "pending-routing", "awaiting-clarification", "next-step", "next-turn", "control",
    "claimed", "completed", "cancelled", "uncertain",
]
NoticeStage = Literal["accepted", "queued", "steered", "clarify", "control", "stopping", "blocked", "completed", "cancelled", "uncertain"]


class InboxConflictError(RuntimeError):
    """Wrong owner/revision/state, or work is not safely erasable."""


class InboxCapacityError(RuntimeError):
    """Admission cannot commit within its declared finite capacity."""


class InboxStateError(RuntimeError):
    """Malformed durable state requires owning recovery, never default fallback."""


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Origin(_Record):
    channel: str = Field(min_length=1, max_length=64)
    account_id: str = Field(min_length=1, max_length=128)
    chat_id: str = Field(min_length=1, max_length=256)
    chat_type: Literal["private", "group", "supergroup", "channel"] | None = None
    sender_id: str = Field(min_length=1, max_length=256)
    transport_id: str = Field(min_length=1, max_length=256)
    thread_id: str | None = Field(default=None, max_length=256)
    reply_to_message_id: str | None = Field(default=None, max_length=256)
    output_reply_to_message_id: str | None = Field(default=None, max_length=256)
    quoted_message_id: str | None = Field(default=None, max_length=256)
    forwarded: bool = False

    def transport_key(self) -> tuple[str, str, str, str | None, str]:
        return self.channel, self.account_id, self.chat_id, self.thread_id, self.transport_id


class TelegramAlbumMember(_Record):
    """One stable Telegram media-group member and its complete assembled payload."""

    message_id: str = Field(min_length=1, max_length=64)
    digest: str = Field(min_length=64, max_length=64)
    content: str = Field(repr=False)
    media: tuple[str, ...] = Field(default=(), repr=False, max_length=4)
    media_digests: tuple[str, ...] | None = Field(default=None, repr=False, max_length=4)

    @model_validator(mode="after")
    def validate_identity_and_digest(self) -> TelegramAlbumMember:
        if not self.message_id.isdecimal() or not re.fullmatch(r"[0-9a-f]{64}", self.digest):
            raise ValueError("invalid Telegram album member identity")
        if (self.media_digests is not None
                and (len(self.media_digests) != len(self.media)
                     or any(not re.fullmatch(r"[0-9a-f]{64}", digest)
                            for digest in self.media_digests))):
            raise ValueError("invalid Telegram album media digest evidence")
        if self.digest != self.payload_digest(self.content, self.media, self.media_digests):
            raise ValueError("Telegram album member digest does not match its payload")
        return self

    @staticmethod
    def payload_digest(
        content: str, media: tuple[str, ...], media_digests: tuple[str, ...] | None = None,
    ) -> str:
        payload: list[object] = [content, list(media)]
        if media_digests is not None:
            payload.append(list(media_digests))
        encoded = json.dumps(payload, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def create(cls, message_id: str, content: str, media: tuple[str, ...] = ()) -> TelegramAlbumMember:
        media_digests = tuple(hashlib.sha256(Path(source).read_bytes()).hexdigest() for source in media)
        return cls(
            message_id=message_id,
            digest=cls.payload_digest(content, media, media_digests),
            content=content,
            media=media,
            media_digests=media_digests,
        )


class InboxPayload(_Record):
    owner_id: str = Field(min_length=1, max_length=128)
    session_key: str = Field(min_length=1, max_length=1024)
    origin: Origin
    content: str = Field(repr=False)
    media: tuple[str, ...] = Field(default=(), repr=False)
    scope: tuple[str, ...] = ()
    album_id: str | None = Field(default=None, max_length=256)
    album_members: tuple[TelegramAlbumMember, ...] = Field(default=(), repr=False, max_length=10)

    @model_validator(mode="after")
    def validate_album(self) -> InboxPayload:
        if (self.album_id is None) != (not self.album_members):
            raise ValueError("Telegram album identity and members must be supplied together")
        if len({member.message_id for member in self.album_members}) != len(self.album_members):
            raise ValueError("Telegram album member identities must be unique")
        return self


class InboxResultReceiptPart(_Record):
    index: int = Field(ge=0)
    message_index: int = Field(ge=0)
    part_index: int = Field(ge=0)
    kind: Literal["media", "text"]
    provider_message_id: int = Field(gt=0)
    provider_chat_id: int
    provider_thread_id: int | None = Field(default=None, gt=0)
    provider_reply_to_message_id: int | None = Field(default=None, gt=0)


class InboxResultReceiptAddress(_Record):
    channel: str = Field(min_length=1, max_length=64)
    account_id: str = Field(min_length=1, max_length=128)
    sender_id: str = Field(min_length=1, max_length=256)
    chat_type: Literal["private", "group", "supergroup", "channel"]
    chat_id: str = Field(min_length=1, max_length=256)
    thread_id: str | None = Field(default=None, max_length=64)
    reply_to_message_id: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def validate_route(self) -> InboxResultReceiptAddress:
        try:
            chat_id = int(self.chat_id)
        except ValueError as exc:
            raise ValueError("receipt chat ID must be canonical") from exc
        if chat_id == 0 or str(chat_id) != self.chat_id:
            raise ValueError("receipt chat ID must be canonical")
        if self.channel != "telegram" or not self.account_id.isdecimal() or int(self.account_id) <= 0:
            raise ValueError("receipt route must bind a Telegram bot account")
        for value in (self.thread_id, self.reply_to_message_id):
            if value is not None and (not value.isdecimal() or int(value) <= 0 or str(int(value)) != value):
                raise ValueError("receipt thread and reply IDs must be canonical")
        return self


class InboxResultReceiptEnvelope(_Record):
    address: InboxResultReceiptAddress
    action_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    parts: tuple[InboxResultReceiptPart, ...] = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_parts(self) -> InboxResultReceiptEnvelope:
        if tuple(part.index for part in self.parts) != tuple(range(len(self.parts))):
            raise ValueError("result receipt part indices must be contiguous")
        expected_chat = int(self.address.chat_id)
        expected_thread = int(self.address.thread_id) if self.address.thread_id is not None else None
        expected_reply = (
            int(self.address.reply_to_message_id)
            if self.address.reply_to_message_id is not None else None
        )
        if any(part.provider_chat_id != expected_chat
               or part.provider_thread_id != expected_thread
               or part.provider_reply_to_message_id != expected_reply for part in self.parts):
            raise ValueError("result receipt parts do not match the exact Telegram route")
        return self


class InboxNoticeReceipt(_Record):
    notice_id: str = Field(min_length=1, max_length=256)
    action_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_message_id: int = Field(gt=0)
    provider_chat_id: int
    provider_thread_id: int | None = Field(default=None, gt=0)
    provider_reply_to_message_id: int | None = Field(default=None, gt=0)


class InboxResult(_Record):
    """Immutable full output retained separately from task and notice facts."""

    run_id: str = Field(min_length=1, max_length=128)
    origin: Origin
    action_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    request_address: InboxResultReceiptAddress | None = None
    content: str = Field(repr=False)
    media: tuple[str, ...] = Field(default=(), repr=False, max_length=64)
    media_digests: tuple[str | None, ...] = Field(default=(), repr=False, max_length=64)
    buttons: tuple[tuple[str, ...], ...] = ()
    disposition: Literal["sending", "delivered", "definitely-unsent", "uncertain"]
    receipt_parts: tuple[InboxResultReceiptPart, ...] = ()
    receipt_envelope: InboxResultReceiptEnvelope | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> InboxResult:
        if (self.media_digests and len(self.media_digests) != len(self.media)
                or any(digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest)
                       for digest in self.media_digests)
                or self.disposition == "delivered" and (
                    self.bundle_digest is None or self.request_address is None
                    or self.receipt_envelope is None or not self.receipt_parts
                )
                or tuple(part.index for part in self.receipt_parts) != tuple(range(len(self.receipt_parts)))):
            raise ValueError("result receipt evidence does not match its delivery disposition")
        if self.receipt_envelope is not None and (
            self.receipt_envelope.address != self.request_address
            or self.receipt_envelope.action_id != self.action_id
            or self.receipt_envelope.bundle_digest != self.bundle_digest
            or self.receipt_envelope.parts != self.receipt_parts
        ):
            raise ValueError("result receipt envelope is not bound to its persisted request")
        if (self.bundle_digest is None) != (self.request_address is None):
            raise ValueError("result request digest and address must be bound together")
        return self


class InboxEntry(_Record):
    id: str
    seq: int = Field(ge=1)
    eligible_seq: int = Field(ge=1)
    revision: int = Field(default=0, ge=0)
    state: EntryState = "pending-routing"
    origin: Origin
    fingerprint: str
    payload: InboxPayload | None = Field(repr=False)
    clarifications: tuple[InboxPayload, ...] = Field(default=(), repr=False)
    clarification_fingerprints: tuple[tuple[Origin, str], ...] = ()
    intent: ControlIntent | None = None
    observed_run: ControlSnapshot | None = None
    claimed_run_id: str | None = None
    dispatch_started: bool = False
    source_event_ids: tuple[str, ...] = Field(default=(), max_length=33)
    control_started: bool = False
    control_report: ControlReport | None = None
    album_id: str | None = Field(default=None, max_length=256)
    album_members: tuple[TelegramAlbumMember, ...] = Field(default=(), repr=False, max_length=10)
    clarification_receipt_message_id: int | None = Field(default=None, gt=0)
    clarification_receipt_origin: Origin | None = None
    result: InboxResult | None = Field(default=None, repr=False)

    @property
    def effective_scope(self) -> frozenset[str]:
        if self.payload is None:
            return frozenset()
        scope = set(self.payload.scope)
        for reply in self.clarifications:
            scope.intersection_update(reply.scope)
        return frozenset(scope)


class InboxNotice(_Record):
    id: str
    ingress_id: str
    origin: Origin
    stage: NoticeStage
    state: Literal["pending", "sending", "uncertain"] = "pending"
    intent: ControlIntent | None = None
    reason: str | None = Field(default=None, max_length=64)
    report: ControlReport | None = None
    acknowledged_message_id: int | None = None


class _State(_Record):
    schema_version: Literal[1] = 1
    owner_id: str
    session_key: str
    temporary_lifetime_id: str | None = None
    closed: bool = False
    revision: int = Field(default=0, ge=0)
    next_seq: int = Field(default=1, ge=1)
    entries: tuple[InboxEntry, ...] = Field(default=(), repr=False)
    notices: tuple[InboxNotice, ...] = ()
    emitted_notice_ids: tuple[str, ...] = ()
    notice_receipts: tuple[InboxNoticeReceipt, ...] = ()


class Admission(_Record):
    entry: InboxEntry
    duplicate: bool


class _LifecycleMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    busy_inbox: object = None


def _same_run(first: ControlSnapshot | None, second: ControlSnapshot | None) -> bool:
    return (first is not None and second is not None and first.run_id == second.run_id
            and first.generation == second.generation
            and first.state == second.state == "running")


def _parse_state(raw: object) -> _State:
    try:
        state = _State.model_validate_json(json.dumps(raw, ensure_ascii=False, allow_nan=False))
    except (ValidationError, ValueError, TypeError, UnicodeError):
        raise InboxStateError("busy inbox state is malformed") from None
    if (len({row.id for row in state.entries}) != len(state.entries)
            or len({row.seq for row in state.entries}) != len(state.entries)
            or len({row.eligible_seq for row in state.entries}) != len(state.entries)
            or any(not row.seq <= row.eligible_seq < state.next_seq for row in state.entries)
            or len({notice.id for notice in state.notices}) != len(state.notices)
            or len(set(state.emitted_notice_ids)) != len(state.emitted_notice_ids)
            or len({receipt.notice_id for receipt in state.notice_receipts}) != len(state.notice_receipts)
            or any(notice.id not in state.emitted_notice_ids for notice in state.notices)
            or any(receipt.notice_id not in state.emitted_notice_ids for receipt in state.notice_receipts)
            or any(notice.ingress_id not in {row.id for row in state.entries} for notice in state.notices)
            or (state.closed and (state.temporary_lifetime_id is None or state.entries or state.notices or state.emitted_notice_ids))):
        raise InboxStateError("busy inbox identity ordering is malformed")
    for row in state.entries:
        terminal = row.state in {"completed", "cancelled"}
        if (terminal != (row.payload is None)
                or (terminal and row.clarifications)
                or (row.state == "claimed" and not row.claimed_run_id)
                or (row.control_started and row.observed_run is None)
                or (row.source_event_ids and not row.dispatch_started)
                or len(set(row.source_event_ids)) != len(row.source_event_ids)
                or any(not identity or len(identity) > 64 for identity in row.source_event_ids)):
            raise InboxStateError("busy inbox entry state is malformed")
        if ((row.album_id is None) != (not row.album_members)
                or len({member.message_id for member in row.album_members}) != len(row.album_members)
                or (row.payload is not None and (
                    row.payload.album_id != row.album_id
                    or row.payload.album_members != row.album_members
                ))):
            raise InboxStateError("busy inbox album identity is malformed")
        if (row.result is not None
                and (row.result.origin != row.origin or row.result.run_id != row.claimed_run_id)):
            raise InboxStateError("busy inbox result delivery identity is malformed")
        if ((row.clarification_receipt_message_id is None)
                != (row.clarification_receipt_origin is None)):
            raise InboxStateError("busy inbox clarification receipt binding is malformed")
        if (row.clarification_receipt_origin is not None and (
                row.clarification_receipt_origin.channel != row.origin.channel
                or row.clarification_receipt_origin.account_id != row.origin.account_id
                or row.clarification_receipt_origin.chat_id != row.origin.chat_id
                or row.clarification_receipt_origin.sender_id != row.origin.sender_id
                or row.clarification_receipt_origin.thread_id != row.origin.thread_id
        )):
            raise InboxStateError("busy inbox clarification receipt route is malformed")
        if row.payload is not None and (
            row.payload.owner_id != state.owner_id or row.payload.session_key != state.session_key
            or row.payload.origin != row.origin
            or hashlib.sha256(row.payload.model_dump_json().encode()).hexdigest() != row.fingerprint
        ):
            raise InboxStateError("busy inbox entry identity is malformed")
        if any(reply.owner_id != state.owner_id or reply.session_key != state.session_key
               for reply in row.clarifications):
            raise InboxStateError("busy inbox clarification identity is malformed")
    return state


def require_busy_inbox_disposition(metadata: object) -> None:
    """Lifecycle writers call this under their existing mutation transaction.

    Temporary lifetime markers must survive generic reset/delete, otherwise an
    old transport or recovery wrapper could recreate an erased lifetime.
    """
    try:
        raw = _LifecycleMetadata.model_validate(metadata).busy_inbox
    except ValidationError:
        raise InboxStateError("busy inbox lifecycle metadata is malformed") from None
    if raw is None:
        return
    state = _parse_state(raw)
    if (state.temporary_lifetime_id is not None or state.notices
            or any(row.state not in {"completed", "cancelled"} for row in state.entries)):
        raise InboxConflictError("busy inbox needs a visible disposition before session lifecycle changes")


class SessionBusyInbox:
    """One locked metadata ledger per owner-session, no competing writer/store."""

    def __init__(self, sessions: SessionManager, *, owner_id: str, session_key: str,
                 limits: BusyInboxConfig | None = None, temporary_lifetime_id: str | None = None):
        if not owner_id or not session_key or temporary_lifetime_id == "":
            raise ValueError("inbox identity must be explicit")
        self.sessions = sessions
        self.owner_id = owner_id
        self.session_key = session_key
        self.limits = limits if limits is not None else BusyInboxConfig()
        self.temporary_lifetime_id = temporary_lifetime_id
        # Temporary recovery has no transcript and never writes the regular key.
        identity = json.dumps([owner_id, session_key, temporary_lifetime_id]).encode()
        self.journal_key = (BUSY_RECOVERY_KEY_PREFIX + hashlib.sha256(identity).hexdigest()
                            if temporary_lifetime_id is not None else session_key)
        self._closed = False

    def _read(self) -> tuple[_State, object]:
        if self._closed:
            raise InboxConflictError("temporary inbox is closed")
        metadata = self.sessions.read_session_metadata(self.journal_key)
        raw: object = metadata["metadata"].get(BUSY_INBOX_METADATA_KEY) if metadata is not None else None
        if raw is None:
            return _State(owner_id=self.owner_id, session_key=self.session_key,
                          temporary_lifetime_id=self.temporary_lifetime_id), None
        state = _parse_state(raw)
        if (state.owner_id != self.owner_id or state.session_key != self.session_key
                or state.temporary_lifetime_id != self.temporary_lifetime_id):
            raise InboxConflictError("busy inbox belongs to another owner-session")
        if state.closed:
            raise InboxConflictError("temporary inbox is closed")
        return state, raw

    def _persist(self, state: _State, expected: object, *, admission: bool = False) -> None:
        state = state.model_copy(update={"revision": state.revision + 1})
        data = state.model_dump_json().encode()
        # Reserve bounded transition/notice space before accepting each payload.
        reserved = 0
        if admission:
            for row in state.entries:
                if row.state in {"completed", "cancelled"}:
                    continue
                origins = (row.origin, *(reply.origin for reply in row.clarifications))
                notice_bytes = max(len(InboxNotice(
                    id=f"{row.id}:32:uncertain", ingress_id=row.id,
                    origin=origin, stage="uncertain",
                ).model_dump_json().encode()) for origin in origins)
                # Six possible route/control/recovery/outcome intents plus bounded
                # run/source identifiers. Repeated orphan recovery coalesces
                # the same stage instead of spending an unbounded new intent.
                reserved += 4096 + 6 * notice_bytes
        if len(data) + reserved > self.limits.max_bytes:
            raise InboxCapacityError("busy inbox storage capacity reached")
        self.sessions.commit_busy_inbox_metadata(self.journal_key, expected, state.model_dump(mode="json"))

    @staticmethod
    def _replace(
        state: _State,
        row: InboxEntry,
        *,
        stage: NoticeStage,
        origin: Origin | None = None,
        intent: ControlIntent | None = None,
        reason: str | None = None,
        report: ControlReport | None = None,
    ) -> _State:
        entries = tuple(row if entry.id == row.id else entry for entry in state.entries)
        identity = f"{row.id}:{len(row.clarification_fingerprints)}:{stage}"
        if identity in state.emitted_notice_ids:
            return state.model_copy(update={"entries": entries})
        notice = InboxNotice(id=identity, ingress_id=row.id,
                             origin=origin or row.origin, stage=stage,
                             intent=intent, reason=reason, report=report)
        return state.model_copy(update={"entries": entries, "notices": (*state.notices, notice),
                                        "emitted_notice_ids": (*state.emitted_notice_ids, identity)})

    @staticmethod
    def _album_key(payload: InboxPayload) -> tuple[str, str, str, str | None, str] | None:
        if payload.album_id is None:
            return None
        origin = payload.origin
        return (origin.account_id, origin.chat_id, origin.sender_id, origin.thread_id, payload.album_id)

    @staticmethod
    def _normalize_album(payload: InboxPayload) -> InboxPayload:
        if payload.album_id is None:
            if payload.album_members:
                raise InboxConflictError("album members require an album identity")
            return payload
        if not payload.album_members:
            raise InboxConflictError("album identity requires stable members")
        members = tuple(sorted(payload.album_members, key=lambda member: int(member.message_id)))
        if len({member.message_id for member in members}) != len(members):
            raise InboxConflictError("album member identities must be unique")
        content = "\n".join(member.content for member in members if member.content) or "[empty message]"
        media = tuple(path for member in members for path in member.media)
        return payload.model_copy(update={"album_members": members, "content": content, "media": media})

    @staticmethod
    def _find(state: _State, identity: str, revision: int) -> InboxEntry:
        row = next((entry for entry in state.entries if entry.id == identity), None)
        if row is None or type(revision) is not int or row.revision != revision:
            raise InboxConflictError("busy inbox revision changed")
        return row

    def entries(self) -> tuple[InboxEntry, ...]:
        with self.sessions.locked_session_files():
            return self._read()[0].entries

    def notices(self) -> tuple[InboxNotice, ...]:
        with self.sessions.locked_session_files():
            return self._read()[0].notices

    def admit(self, payload: InboxPayload) -> Admission:
        if payload.owner_id != self.owner_id or payload.session_key != self.session_key:
            raise InboxConflictError("input does not belong to this owner-session")
        payload = self._normalize_album(payload)
        if self._closed:
            raise InboxConflictError("temporary inbox is closed")
        self.sessions.prepare_busy_inbox_metadata(self.journal_key)
        with self.sessions.locked_session_files():
            state, expected = self._read()
            album_key = self._album_key(payload)
            if album_key is not None:
                related = tuple(row for row in state.entries
                                if row.album_id is not None and self._album_key_from_entry(row) == album_key)
                known: dict[str, str] = {}
                for row in related:
                    for member in row.album_members:
                        previous = known.get(member.message_id)
                        if previous is not None and previous != member.digest:
                            raise InboxStateError("busy inbox album history is conflicting")
                        known[member.message_id] = member.digest
                for member in payload.album_members:
                    previous = known.get(member.message_id)
                    if previous is not None and previous != member.digest:
                        raise InboxConflictError("album member identity payload changed")
                fresh = tuple(member for member in payload.album_members if member.message_id not in known)
                if not fresh:
                    if not related:
                        raise InboxStateError("album member history is missing its ingress")
                    return Admission(entry=related[-1], duplicate=True)
                if related:
                    member_identity = ",".join(member.message_id for member in fresh)
                    supplemental_id = hashlib.sha256(member_identity.encode()).hexdigest()[:32]
                    payload = payload.model_copy(update={
                        "content": "\n".join(member.content for member in fresh if member.content) or "[empty message]",
                        "media": tuple(path for member in fresh for path in member.media),
                        "album_members": fresh,
                        "origin": payload.origin.model_copy(update={
                            "transport_id": f"album:{payload.album_id}:{supplemental_id}",
                            "output_reply_to_message_id": fresh[0].message_id,
                        }),
                    })
            fingerprint = hashlib.sha256(payload.model_dump_json().encode()).hexdigest()
            duplicate = next((row for row in state.entries if row.origin.transport_key() == payload.origin.transport_key()), None)
            if duplicate is not None:
                if duplicate.fingerprint != fingerprint:
                    raise InboxConflictError("transport identity payload changed")
                return Admission(entry=duplicate, duplicate=True)
            if any(origin.transport_key() == payload.origin.transport_key()
                   for row in state.entries for origin, _fingerprint in row.clarification_fingerprints):
                raise InboxConflictError("transport identity already belongs to a clarification")
            if (len(state.entries) >= self.limits.max_records
                    or sum(row.state not in {"completed", "cancelled"} for row in state.entries) >= self.limits.max_pending):
                raise InboxCapacityError("busy inbox admission capacity reached")
            row = InboxEntry(id=uuid.uuid4().hex, seq=state.next_seq, eligible_seq=state.next_seq, origin=payload.origin,
                             fingerprint=fingerprint, payload=payload, album_id=payload.album_id,
                             album_members=payload.album_members)
            state = state.model_copy(update={"entries": (*state.entries, row), "next_seq": state.next_seq + 1})
            state = self._replace(state, row, stage="accepted")
            self._persist(state, expected, admission=True)
            return Admission(entry=row, duplicate=False)

    @staticmethod
    def _album_key_from_entry(row: InboxEntry) -> tuple[str, str, str, str | None, str] | None:
        if row.album_id is None:
            return None
        origin = row.origin
        return (origin.account_id, origin.chat_id, origin.sender_id, origin.thread_id, row.album_id)

    def route(self, identity: str, revision: int, decision: ControlDecision, *,
              current_run: ControlSnapshot | None, active_scope: tuple[str, ...]) -> InboxEntry:
        with self.sessions.locked_session_files():
            state, expected = self._read()
            row = self._find(state, identity, revision)
            if row.state != "pending-routing" or row.payload is None:
                raise InboxConflictError("input is not awaiting routing")
            intent = decision.intent
            if intent in {"steer", "stop", "stop-and-replace"} and not _same_run(decision.observed_run, current_run):
                # An obsolete stop is a status enquiry, never an instruction
                # that could execute against the next run's newer generation.
                intent = "status" if intent == "stop" else "next-turn"
            if intent == "steer" and not set(active_scope).issubset(row.effective_scope):
                intent = "next-turn"
            status: EntryState
            stage: NoticeStage
            if intent == "steer":
                status, stage = "next-step", "steered"
            elif intent == "next-turn":
                status, stage = "next-turn", "queued"
            elif intent == "clarify":
                status, stage = "awaiting-clarification", "clarify"
            else:
                status, stage = "control", "control"
            row = row.model_copy(update={"revision": revision + 1, "state": status,
                                         "intent": intent, "observed_run": decision.observed_run})
            self._persist(self._replace(state, row, stage=stage, intent=intent), expected)
            return row

    def begin_control(self, identity: str, revision: int, *, current_run: ControlSnapshot,
                      report: ControlReport) -> InboxEntry:
        """Persist stop intent before its owner requests cooperative cancellation."""
        with self.sessions.locked_session_files():
            state, expected = self._read()
            row = self._find(state, identity, revision)
            observed = row.observed_run
            if (row.state != "control" or row.intent not in {"stop", "stop-and-replace"}
                    or observed is None or observed.run_id != current_run.run_id
                    or observed.generation != current_run.generation):
                raise InboxConflictError("control does not target this run generation")
            if row.control_started:
                return row
            row = row.model_copy(update={"revision": revision + 1, "control_started": True,
                                         "control_report": report})
            self._persist(self._replace(state, row, stage="stopping",
                                        intent=row.intent, report=report), expected)
            return row

    def block_control(self, identity: str, revision: int, *, report: ControlReport) -> InboxEntry:
        with self.sessions.locked_session_files():
            state, expected = self._read()
            row = self._find(state, identity, revision)
            if (row.state != "control" or not row.control_started or report.active_run is None
                    or report.active_run.state != "blocked" or row.observed_run is None
                    or report.active_run.run_id != row.observed_run.run_id
                    or report.active_run.generation != row.observed_run.generation):
                raise InboxConflictError("control has no blocked owning run")
            if row.control_report == report:
                return row
            row = row.model_copy(update={"revision": revision + 1, "control_report": report})
            self._persist(self._replace(state, row, stage="blocked",
                                        intent=row.intent, report=report), expected)
            return row

    def finish_control(self, identity: str, revision: int, *, report: ControlReport) -> InboxEntry:
        """Settle control bookkeeping; task completion remains a separate fact.

        An undispatched replacement retains the original ingress and authority.
        Unknown old effects retain payloads and fence execution, including when
        the old occurrence was automation without a human claim in this ledger.
        """
        with self.sessions.locked_session_files():
            state, expected = self._read()
            row = self._find(state, identity, revision)
            if row.state != "control":
                raise InboxConflictError("input is not an unsettled control")
            status: EntryState
            if row.intent == "stop-and-replace":
                status = "uncertain" if report.recovery_required else "next-turn"
            else:
                status = "uncertain" if row.control_started and report.recovery_required else "completed"
            stage: NoticeStage = "queued" if status == "next-turn" else (
                "uncertain" if status == "uncertain" else "completed")
            row = row.model_copy(update={"revision": revision + 1, "state": status,
                                         "payload": None if status == "completed" else row.payload,
                                         "clarifications": () if status == "completed" else row.clarifications})
            entries = tuple(row if entry.id == row.id else entry for entry in state.entries)
            queued = sum(entry.state == "next-turn" or (
                entry.state == "control" and entry.intent == "stop-and-replace") for entry in entries)
            report = replace(report, queued=queued, active_run=(
                replace(report.active_run, queued=queued) if report.active_run else None))
            row = row.model_copy(update={"control_report": report})
            self._persist(self._replace(state, row, stage=stage,
                                        intent=row.intent, report=report), expected)
            return row

    def resume_clarification(self, identity: str, revision: int, reply: InboxPayload) -> InboxEntry:
        if reply.owner_id != self.owner_id or reply.session_key != self.session_key:
            raise InboxConflictError("clarification does not belong to this owner-session")
        fingerprint = hashlib.sha256(reply.model_dump_json().encode()).hexdigest()
        with self.sessions.locked_session_files():
            state, expected = self._read()
            for entry in state.entries:
                for origin, digest in entry.clarification_fingerprints:
                    if origin.transport_key() == reply.origin.transport_key():
                        if entry.id != identity or digest != fingerprint:
                            raise InboxConflictError("clarification transport identity changed")
                        return entry
                if entry.origin.transport_key() == reply.origin.transport_key():
                    raise InboxConflictError("clarification transport identity is already admitted")
            row = self._find(state, identity, revision)
            if row.state != "awaiting-clarification" or row.payload is None:
                raise InboxConflictError("input is not awaiting clarification")
            if len(row.clarification_fingerprints) >= self.limits.max_clarifications:
                raise InboxCapacityError("busy inbox clarification capacity reached")
            row = row.model_copy(update={
                "revision": row.revision + 1, "state": "pending-routing",
                "eligible_seq": state.next_seq, "intent": None, "observed_run": None,
                "clarifications": (*row.clarifications, reply),
                "clarification_fingerprints": (*row.clarification_fingerprints, (reply.origin, fingerprint)),
            })
            state = state.model_copy(update={"next_seq": state.next_seq + 1})
            state = self._replace(state, row, stage="accepted")
            notice = state.notices[-1].model_copy(update={"origin": reply.origin})
            state = state.model_copy(update={"notices": (*state.notices[:-1], notice)})
            self._persist(state, expected, admission=True)
            return row

    def _claim(self, state: _State, expected: object, row: InboxEntry, run_id: str) -> InboxEntry:
        if not run_id.strip() or len(run_id) > 128:
            raise ValueError("claim run identity must be explicit")
        row = row.model_copy(update={"revision": row.revision + 1, "state": "claimed", "claimed_run_id": run_id})
        self._persist(state.model_copy(update={"entries": tuple(row if entry.id == row.id else entry for entry in state.entries)}), expected)
        return row

    def claim_next_turn(self, run_id: str, *, current_run: ControlSnapshot | None) -> InboxEntry | None:
        with self.sessions.locked_session_files():
            state, expected = self._read()
            if current_run is not None or any(row.state in {"claimed", "uncertain"} for row in state.entries):
                return None
            row = min((entry for entry in state.entries if entry.state == "next-turn"),
                      key=lambda entry: entry.eligible_seq, default=None)
            if row is None or any(entry.state == "pending-routing" and entry.eligible_seq < row.eligible_seq for entry in state.entries):
                return None
            return self._claim(state, expected, row, run_id)

    def claim_steer(self, current_run: ControlSnapshot, *,
                    active_scope: tuple[str, ...] | None = None) -> InboxEntry | None:
        with self.sessions.locked_session_files():
            state, expected = self._read()
            if active_scope is not None:
                updated = state
                for entry in state.entries:
                    if (entry.state == "next-step" and _same_run(entry.observed_run, current_run)
                            and not set(active_scope).issubset(entry.effective_scope)):
                        entry = entry.model_copy(update={"revision": entry.revision + 1,
                                                         "state": "next-turn", "intent": "next-turn"})
                        updated = self._replace(updated, entry, stage="queued")
                if updated != state:
                    self._persist(updated, expected)
                    state, expected = self._read()
            row = min((entry for entry in state.entries if entry.state == "next-step" and _same_run(entry.observed_run, current_run)),
                      key=lambda entry: entry.eligible_seq, default=None)
            return self._claim(state, expected, row, current_run.run_id) if row is not None else None

    def begin_dispatch(self, identity: str, revision: int) -> InboxEntry:
        with self.sessions.locked_session_files():
            state, expected = self._read()
            row = self._find(state, identity, revision)
            if row.state != "claimed" or row.dispatch_started:
                raise InboxConflictError("input is not an undispatched claim")
            row = row.model_copy(update={"revision": revision + 1, "dispatch_started": True})
            self._persist(state.model_copy(update={"entries": tuple(row if entry.id == identity else entry for entry in state.entries)}), expected)
            return row

    def bind_source(self, identity: str, revision: int, source_event_ids: tuple[str, ...]) -> InboxEntry:
        """Record the owning adapter's verified durable source manifest.

        This metadata is not provider-dispatch authority. The adapter must first
        verify each event against its actual SourceTurnWriter's owned stream.
        """
        if (type(source_event_ids) is not tuple or not source_event_ids or len(source_event_ids) > 33
                or len(set(source_event_ids)) != len(source_event_ids)
                or any(not value or len(value) > 64 for value in source_event_ids)):
            raise InboxConflictError("invalid busy input source manifest")
        with self.sessions.locked_session_files():
            state, expected = self._read()
            row = self._find(state, identity, revision)
            if (row.state != "claimed" or not row.dispatch_started or row.source_event_ids
                    or len(source_event_ids) != 1 + len(row.clarifications)):
                raise InboxConflictError("input has no unbound source dispatch")
            row = row.model_copy(update={"revision": revision + 1, "source_event_ids": source_event_ids})
            self._persist(state.model_copy(update={"entries": tuple(row if entry.id == identity else entry for entry in state.entries)}), expected)
            return row

    def recover_source_binding(self, identity: str, revision: int, source_event_ids: tuple[str, ...]) -> InboxEntry:
        """Annotate verified recovery evidence without completing/requeueing work."""
        if (not source_event_ids or len(source_event_ids) > 33
                or len(set(source_event_ids)) != len(source_event_ids)
                or any(not value or len(value) > 64 for value in source_event_ids)):
            raise InboxConflictError("invalid busy input recovery manifest")
        with self.sessions.locked_session_files():
            state, expected = self._read()
            row = self._find(state, identity, revision)
            if (row.state not in {"claimed", "uncertain"} or not row.dispatch_started
                    or len(source_event_ids) > 1 + len(row.clarifications)
                    or source_event_ids[:len(row.source_event_ids)] != row.source_event_ids):
                raise InboxConflictError("input has no reconcilable source dispatch")
            if source_event_ids == row.source_event_ids:
                return row
            row = row.model_copy(update={"revision": revision + 1, "source_event_ids": source_event_ids})
            self._persist(state.model_copy(update={"entries": tuple(row if entry.id == identity else entry for entry in state.entries)}), expected)
            return row

    def finish(self, identity: str, revision: int, outcome: Literal["completed", "cancelled", "uncertain"]) -> InboxEntry:
        if outcome not in {"completed", "cancelled", "uncertain"}:
            raise ValueError("invalid inbox outcome")
        with self.sessions.locked_session_files():
            state, expected = self._read()
            row = self._find(state, identity, revision)
            if row.state != "claimed":
                raise InboxConflictError("input is not claimed")
            row = row.model_copy(update={"revision": revision + 1, "state": outcome,
                                         "payload": row.payload if outcome == "uncertain" else None,
                                         "clarifications": row.clarifications if outcome == "uncertain" else ()})
            self._persist(self._replace(state, row, stage=outcome), expected)
            return row

    def begin_result(
        self,
        identity: str,
        revision: int,
        *,
        run_id: str,
        origin: Origin,
        action_id: str,
        output_digest: str,
        content: str,
        media: tuple[str, ...],
        buttons: tuple[tuple[str, ...], ...],
        media_digests: tuple[str | None, ...] = (),
    ) -> InboxEntry:
        """Durably retain an immutable output before an exact-route send begins."""
        if not run_id or len(run_id) > 128:
            raise InboxConflictError("result run identity is invalid")
        candidate = InboxResult(
            run_id=run_id,
            origin=origin,
            action_id=action_id,
            output_digest=output_digest,
            content=content,
            media=media,
            media_digests=media_digests,
            buttons=buttons,
            disposition="sending",
        )
        with self.sessions.locked_session_files():
            state, expected = self._read()
            row = self._find(state, identity, revision)
            if row.claimed_run_id != run_id or row.origin != origin:
                raise InboxConflictError("result does not belong to the claimed ingress and run")
            previous = row.result
            if previous is None:
                if row.state != "claimed":
                    raise InboxConflictError("result has no live owning claim")
                result = candidate
            else:
                same_output = (
                    previous.run_id == candidate.run_id
                    and previous.origin == candidate.origin
                    and previous.action_id == candidate.action_id
                    and previous.output_digest == candidate.output_digest
                    and previous.content == candidate.content
                    and previous.media == candidate.media
                    and previous.media_digests == candidate.media_digests
                    and previous.buttons == candidate.buttons
                )
                if row.state not in {"claimed", "completed"} or not same_output:
                    raise InboxConflictError("persisted result is immutable")
                if previous.disposition != "definitely-unsent":
                    raise InboxConflictError("result is not eligible for output-only retry")
                result = InboxResult.model_validate(previous.model_dump() | {
                    "disposition": "sending",
                    "receipt_parts": (),
                    "receipt_envelope": None,
                })
            updated = row.model_copy(update={"revision": row.revision + 1, "result": result})
            self._persist(state.model_copy(update={
                "entries": tuple(updated if value.id == identity else value for value in state.entries)
            }), expected)
            return updated

    def bind_result_request(
        self,
        identity: str,
        revision: int,
        *,
        action_id: str,
        address: InboxResultReceiptAddress,
        bundle_digest: str,
    ) -> InboxEntry:
        """Persist the exact emitted request before a provider call can begin."""
        with self.sessions.locked_session_files():
            state, expected = self._read()
            row = self._find(state, identity, revision)
            previous = row.result
            if previous is None or previous.disposition != "sending":
                raise InboxConflictError("result has no active delivery attempt")
            if action_id != previous.action_id:
                raise InboxConflictError("result request action changed")
            if previous.bundle_digest is not None or previous.request_address is not None:
                if previous.bundle_digest != bundle_digest or previous.request_address != address:
                    raise InboxConflictError("result request is immutable across retries")
                return row
            result = InboxResult.model_validate(previous.model_dump() | {
                "bundle_digest": bundle_digest,
                "request_address": address.model_dump(),
            })
            row = row.model_copy(update={"revision": revision + 1, "result": result})
            self._persist(state.model_copy(update={
                "entries": tuple(row if value.id == identity else value for value in state.entries)
            }), expected)
            return row

    def settle_result(
        self,
        identity: str,
        revision: int,
        *,
        disposition: Literal["delivered", "definitely-unsent", "uncertain"],
        receipt_parts: tuple[InboxResultReceiptPart, ...] = (),
        receipt_envelope: InboxResultReceiptEnvelope | None = None,
    ) -> InboxEntry:
        """Persist the provider disposition separately from task terminal state."""
        with self.sessions.locked_session_files():
            state, expected = self._read()
            row = self._find(state, identity, revision)
            if row.result is None or row.result.disposition != "sending":
                raise InboxConflictError("result has no active delivery attempt")
            if receipt_envelope is not None and (
                receipt_envelope.address != row.result.request_address
                or receipt_envelope.action_id != row.result.action_id
                or receipt_envelope.bundle_digest != row.result.bundle_digest
            ):
                raise InboxConflictError("result receipt is not bound to the persisted request")
            if receipt_envelope is not None:
                if receipt_parts and receipt_parts != receipt_envelope.parts:
                    raise InboxConflictError("result receipt parts do not match its envelope")
                receipt_parts = receipt_envelope.parts
            if disposition == "delivered" and receipt_envelope is None:
                raise InboxConflictError("delivered result requires its complete receipt envelope")
            if disposition == "definitely-unsent" and receipt_envelope is not None:
                raise InboxConflictError("definitely-unsent result cannot contain provider receipts")
            result = InboxResult.model_validate(row.result.model_dump() | {
                "disposition": disposition,
                "receipt_parts": receipt_parts,
                "receipt_envelope": receipt_envelope,
            })
            row = row.model_copy(update={"revision": revision + 1, "result": result})
            self._persist(state.model_copy(update={
                "entries": tuple(row if value.id == identity else value for value in state.entries)
            }), expected)
            return row

    def claim_notice(self, identity: str) -> InboxNotice:
        with self.sessions.locked_session_files():
            state, expected = self._read()
            notice = next((value for value in state.notices if value.id == identity), None)
            if notice is None or notice.state != "pending":
                raise InboxConflictError("notice is not safely claimable")
            notice = notice.model_copy(update={"state": "sending"})
            self._persist(state.model_copy(update={"notices": tuple(notice if value.id == identity else value for value in state.notices)}), expected)
            return notice

    def acknowledge_notice(
        self,
        identity: str,
        *,
        receipt: InboxNoticeReceipt | None = None,
    ) -> None:
        with self.sessions.locked_session_files():
            state, expected = self._read()
            notice = next((value for value in state.notices if value.id == identity), None)
            if notice is None or notice.state != "sending":
                raise InboxConflictError("notice has no completed send claim")
            if receipt is not None:
                if type(receipt) is not InboxNoticeReceipt or receipt.notice_id != identity:
                    raise InboxConflictError("notice receipt is not bound to this notice")
                if any(value.notice_id == identity for value in state.notice_receipts):
                    raise InboxConflictError("notice receipt was already recorded")
            entries = state.entries
            if notice.stage == "clarify" and receipt is not None:
                row = next((value for value in entries if value.id == notice.ingress_id), None)
                if row is None or row.state != "awaiting-clarification":
                    raise InboxConflictError("clarification notice has no pending ingress")
                row = row.model_copy(update={
                    "revision": row.revision + 1,
                    "clarification_receipt_message_id": receipt.provider_message_id,
                    "clarification_receipt_origin": notice.origin,
                })
                entries = tuple(row if value.id == row.id else value for value in entries)
            self._persist(state.model_copy(update={
                "entries": entries,
                "notices": tuple(value for value in state.notices if value.id != identity),
                "notice_receipts": (*state.notice_receipts, receipt) if receipt is not None else state.notice_receipts,
            }), expected)

    def fail_notice(self, identity: str, *, definitely_unsent: bool) -> None:
        if type(definitely_unsent) is not bool:
            raise ValueError("notice failure disposition must be explicit")
        with self.sessions.locked_session_files():
            state, expected = self._read()
            notice = next((value for value in state.notices if value.id == identity), None)
            if notice is None or notice.state != "sending":
                raise InboxConflictError("notice has no active send claim")
            notice = notice.model_copy(update={"state": "pending" if definitely_unsent else "uncertain"})
            self._persist(state.model_copy(update={"notices": tuple(notice if value.id == identity else value for value in state.notices)}), expected)

    def reconcile_orphans(self, *, live_run_ids: frozenset[str], recover_notices: bool = True) -> None:
        """Reconcile owning run loss; normal turn endings leave live sends alone."""
        with self.sessions.locked_session_files():
            state, expected = self._read()
            updated = state
            for row in state.entries:
                if (row.result is not None and row.result.disposition == "sending"
                        and row.result.run_id not in live_run_ids):
                    row = row.model_copy(update={
                        "revision": row.revision + 1,
                        "result": row.result.model_copy(update={"disposition": "uncertain"}),
                    })
                    updated = self._replace(updated, row, stage="uncertain")
                if row.state == "claimed" and row.claimed_run_id not in live_run_ids:
                    status: EntryState = "uncertain" if row.dispatch_started else "next-turn"
                    stage: NoticeStage = "uncertain" if row.dispatch_started else "queued"
                    row = row.model_copy(update={"revision": row.revision + 1, "state": status,
                                                 "claimed_run_id": row.claimed_run_id if row.dispatch_started else None})
                    updated = self._replace(updated, row, stage=stage)
                elif row.state == "next-step" and (row.observed_run is None or row.observed_run.run_id not in live_run_ids):
                    row = row.model_copy(update={"revision": row.revision + 1, "state": "next-turn", "intent": "next-turn"})
                    updated = self._replace(updated, row, stage="queued")
                elif (row.state == "control" and row.control_started and row.observed_run is not None
                      and row.observed_run.run_id not in live_run_ids):
                    report = ControlReport(active_run=None,
                        queued=sum(entry.state == "next-turn" for entry in state.entries),
                        automation_queued=0, recovery_required=True)
                    row = row.model_copy(update={"revision": row.revision + 1, "state": "uncertain",
                                                 "control_report": report})
                    updated = self._replace(updated, row, stage="uncertain")
            if recover_notices:
                updated = updated.model_copy(update={"notices": tuple(
                    value.model_copy(update={"state": "uncertain"}) if value.state == "sending" else value
                    for value in updated.notices)})
            if updated != state:
                self._persist(updated, expected)

    def erase_temporary(self) -> None:
        if self.temporary_lifetime_id is None:
            raise InboxConflictError("regular inbox is not temporary")
        with self.sessions.locked_session_files():
            state, expected = self._read()
            if state.notices or any(row.state not in {"completed", "cancelled"} for row in state.entries):
                raise InboxConflictError("temporary inbox has pending or uncertain work")
            # Content-free retirement is durable; deleting the key would allow
            # stale wrappers to recreate the same temporary lifetime on restart.
            self._persist(state.model_copy(update={"entries": (), "notices": (), "emitted_notice_ids": (), "closed": True}), expected)
            self._closed = True
