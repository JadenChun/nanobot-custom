"""Pure typed payloads for dormant legacy-memory receipt transactions.

This module only constructs and validates evidence payloads. It does not read,
write, replay, or advance any recovery state. Blob descriptors are claims;
checking their bytes remains the recovery runtime's responsibility.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent.memory_recovery import MemoryReceipt
from nanobot.agent.memory_recovery_manifest import (
    MAX_INT,
    MAX_RECORD_BYTES,
    MemoryManifestError,
    derive_manifest_identity,
)

_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_WORKSPACE_ID = re.compile(r"[0-9a-f]{32}\Z")
_SOURCE_FORMAT = "legacy_jsonl"
_TARGET_NAMES = ("history.jsonl", ".cursor", "MEMORY.md")
_STAT_FIELDS = frozenset(
    ("device", "inode", "mode", "uid", "gid", "links", "size", "mtime_ns", "ctime_ns")
)
_CAPTURED_FIELDS = frozenset(
    (
        "schema_version",
        "source_format",
        "conversion_identity",
        "receipt_digest",
        "receipt_size",
        "session_key_digest",
        "start_offset",
        "end_offset",
        "session_generation",
        "chunk_digest",
        "base_memory_digest",
        "output_memory_digest",
        "output_memory_size",
        "reference_only",
        "history_marker_digest",
        "history_entry_digest",
        "created_at_digest",
    )
)
_PREPARED_FIELDS = frozenset(
    (
        "schema_version",
        "receipt_digest",
        "receipt_stat",
        "targets",
        "source",
        "history_marker_digest",
        "history_entry_digest",
        "history_row_digest",
        "memory_action",
    )
)
_SOURCE_FIELDS = frozenset(
    (
        "old_digest",
        "old_size",
        "old_generation",
        "old_revision",
        "old_cursor",
        "new_digest",
        "new_size",
        "new_generation",
        "new_revision",
        "new_cursor",
        "action",
    )
)
_OLD_TARGET_FIELDS = frozenset(("exists", "size", "digest", "stat"))
_NEW_TARGET_FIELDS = frozenset(("exists", "size", "blob_digest"))
_HISTORY_APPLIED_FIELDS = frozenset(
    ("schema_version", "receipt_digest", "history_row_digest", "history_target", "cursor_target")
)
_MEMORY_APPLIED_FIELDS = frozenset(
    ("schema_version", "receipt_digest", "memory_action", "memory_target")
)
_SOURCE_APPLIED_FIELDS = frozenset(("digest", "size", "generation", "revision", "cursor"))
_OFFSET_APPLIED_FIELDS = frozenset(("schema_version", "receipt_digest", "source"))
_RECEIPT_COMMITTED_FIELDS = frozenset(
    (
        "schema_version",
        "receipt_digest",
        "prepared_record_digest",
        "history_applied_record_digest",
        "memory_applied_record_digest",
        "offset_applied_record_digest",
        "receipt_size",
        "receipt_present",
    )
)
_COMPLETE_FIELDS = frozenset(
    (
        "schema_version",
        "receipt_digest",
        "commit_record_digest",
        "receipt",
        "history_target",
        "cursor_target",
        "memory_target",
        "source",
        "history_row_digest",
        "history_marker_digest",
        "marker_occurrences",
    )
)
_TRANSACTION_PHASE_SEQUENCE = {
    "CAPTURED": 0,
    "PREPARED": 1,
    "HISTORY_APPLIED": 2,
    "MEMORY_APPLIED": 3,
    "OFFSET_APPLIED": 4,
    "RECEIPT_COMMITTED": 5,
    "COMPLETE": 6,
}


class MemoryRecoveryTransactionError(ValueError):
    """A content-free validation error for receipt transaction payloads."""

    def __init__(self) -> None:
        super().__init__("memory recovery transaction payload is invalid")


def _fail() -> None:
    raise MemoryRecoveryTransactionError() from None


def _digest(value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail()
    return value


def _counter(value: object, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0) or value > MAX_INT:
        _fail()
    return value


def _text(value: object) -> str:
    if type(value) is not str or not value.strip():
        _fail()
    return value


def _workspace_id(value: object) -> str:
    if type(value) is not str or _WORKSPACE_ID.fullmatch(value) is None:
        _fail()
    return value


def _hash_text(value: str) -> str:
    try:
        raw = value.encode("utf-8")
    except UnicodeEncodeError:
        _fail()
    return hashlib.sha256(raw).hexdigest()


def _receipt_from_exact_bytes(receipt: MemoryReceipt) -> MemoryReceipt:
    """Reparse and compare the detached object so bytes and fields cannot diverge."""

    if type(receipt) is not MemoryReceipt or type(receipt.raw_bytes) is not bytes or not receipt.raw_bytes:
        _fail()
    try:
        payload = receipt_codec._decode(receipt.raw_bytes)
        parsed = receipt_codec._validate_payload(payload, receipt.raw_bytes)
    except Exception:
        _fail()
    if parsed != receipt:
        _fail()
    return parsed


def _stat_payload(value: object, *, receipt: bool) -> "MemoryFileStat":
    if type(value) is not dict or set(value) != _STAT_FIELDS:
        _fail()
    device = _counter(value["device"])
    inode = _counter(value["inode"])
    mode = _counter(value["mode"])
    uid = _counter(value["uid"])
    gid = _counter(value["gid"])
    links = _counter(value["links"], positive=True)
    size = _counter(value["size"])
    mtime_ns = value["mtime_ns"]
    ctime_ns = value["ctime_ns"]
    for timestamp in (mtime_ns, ctime_ns):
        if timestamp is not None and (type(timestamp) is not int or not -MAX_INT <= timestamp <= MAX_INT):
            _fail()
    if receipt and (not stat.S_ISREG(mode) or links != 1):
        _fail()
    return MemoryFileStat(device, inode, mode, uid, gid, links, size, mtime_ns, ctime_ns)


def _stat_to_payload(value: "MemoryFileStat") -> dict[str, int | None]:
    return {
        "device": value.device,
        "inode": value.inode,
        "mode": value.mode,
        "uid": value.uid,
        "gid": value.gid,
        "links": value.links,
        "size": value.size,
        "mtime_ns": value.mtime_ns,
        "ctime_ns": value.ctime_ns,
    }


@dataclass(frozen=True, slots=True)
class MemoryFileStat:
    device: int
    inode: int
    mode: int
    uid: int
    gid: int
    links: int
    size: int
    mtime_ns: int | None
    ctime_ns: int | None

    def to_payload(self) -> dict[str, int | None]:
        return _stat_to_payload(self)


@dataclass(frozen=True, slots=True)
class CapturedTransactionPayload:
    receipt_digest: str
    receipt_size: int
    session_key_digest: str
    start_offset: int
    end_offset: int
    session_generation: int
    chunk_digest: str
    base_memory_digest: str
    output_memory_digest: str
    output_memory_size: int
    reference_only: bool
    history_marker_digest: str
    history_entry_digest: str
    created_at_digest: str

    @property
    def identity(self) -> str:
        return derive_manifest_identity("transactions", self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "source_format": _SOURCE_FORMAT,
            "conversion_identity": None,
            "receipt_digest": self.receipt_digest,
            "receipt_size": self.receipt_size,
            "session_key_digest": self.session_key_digest,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "session_generation": self.session_generation,
            "chunk_digest": self.chunk_digest,
            "base_memory_digest": self.base_memory_digest,
            "output_memory_digest": self.output_memory_digest,
            "output_memory_size": self.output_memory_size,
            "reference_only": self.reference_only,
            "history_marker_digest": self.history_marker_digest,
            "history_entry_digest": self.history_entry_digest,
            "created_at_digest": self.created_at_digest,
        }


def capture_transaction_payload(
    receipt: MemoryReceipt,
    *,
    workspace_id: str,
    workspace_identity: tuple[int, int],
) -> CapturedTransactionPayload:
    """Build deterministic, content-free CAPTURED evidence from one receipt."""

    parsed = _receipt_from_exact_bytes(receipt)
    _workspace_id(workspace_id)
    if type(workspace_identity) is not tuple or len(workspace_identity) != 2:
        _fail()
    _counter(workspace_identity[0])
    _counter(workspace_identity[1])
    for value in (parsed.start_offset, parsed.end_offset, parsed.session_generation):
        _counter(value)
    if parsed.start_offset >= parsed.end_offset:
        _fail()
    _text(parsed.session_key)
    for value in (parsed.chunk_digest, parsed.base_memory_digest, parsed.output_memory_digest):
        _digest(value)
    if type(parsed.reference_only) is not bool:
        _fail()
    try:
        output_memory_size = len(parsed.memory_output.encode("utf-8"))
    except UnicodeEncodeError:
        _fail()
    captured = CapturedTransactionPayload(
        receipt_digest=hashlib.sha256(parsed.raw_bytes).hexdigest(),
        receipt_size=len(parsed.raw_bytes),
        session_key_digest=_hash_text(parsed.session_key),
        start_offset=parsed.start_offset,
        end_offset=parsed.end_offset,
        session_generation=parsed.session_generation,
        chunk_digest=parsed.chunk_digest,
        base_memory_digest=parsed.base_memory_digest,
        output_memory_digest=parsed.output_memory_digest,
        output_memory_size=output_memory_size,
        reference_only=parsed.reference_only,
        history_marker_digest=_hash_text(parsed.history_marker_id),
        history_entry_digest=_hash_text(parsed.history_entry),
        created_at_digest=_hash_text(parsed.created_at),
    )
    try:
        identity = derive_manifest_identity("transactions", captured.to_payload())
        _validate_record_bounds(captured.to_payload(), "CAPTURED", identity)
    except Exception:
        _fail()
    return captured


@dataclass(frozen=True, slots=True)
class OldTargetDescriptor:
    exists: bool
    size: int
    digest: str | None
    stat: MemoryFileStat | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "size": self.size,
            "digest": self.digest,
            "stat": None if self.stat is None else self.stat.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class NewTargetDescriptor:
    exists: bool
    size: int
    blob_digest: str | None

    def to_payload(self) -> dict[str, Any]:
        return {"exists": self.exists, "size": self.size, "blob_digest": self.blob_digest}


@dataclass(frozen=True, slots=True)
class TargetTransition:
    old: OldTargetDescriptor
    new: NewTargetDescriptor

    def to_payload(self) -> dict[str, Any]:
        return {"old": self.old.to_payload(), "new": self.new.to_payload()}


@dataclass(frozen=True, slots=True)
class SourceTransition:
    old_digest: str
    old_size: int
    old_generation: int
    old_revision: int
    old_cursor: int
    new_digest: str
    new_size: int
    new_generation: int
    new_revision: int
    new_cursor: int
    action: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "old_digest": self.old_digest,
            "old_size": self.old_size,
            "old_generation": self.old_generation,
            "old_revision": self.old_revision,
            "old_cursor": self.old_cursor,
            "new_digest": self.new_digest,
            "new_size": self.new_size,
            "new_generation": self.new_generation,
            "new_revision": self.new_revision,
            "new_cursor": self.new_cursor,
            "action": self.action,
        }


@dataclass(frozen=True, slots=True)
class PreparedTransactionPayload:
    captured: CapturedTransactionPayload
    receipt_stat: MemoryFileStat
    targets: Mapping[str, TargetTransition]
    source: SourceTransition
    history_row_digest: str
    memory_action: str

    @property
    def receipt_digest(self) -> str:
        return self.captured.receipt_digest

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "receipt_digest": self.receipt_digest,
            "receipt_stat": self.receipt_stat.to_payload(),
            "targets": {name: self.targets[name].to_payload() for name in _TARGET_NAMES},
            "source": self.source.to_payload(),
            "history_marker_digest": self.captured.history_marker_digest,
            "history_entry_digest": self.captured.history_entry_digest,
            "history_row_digest": self.history_row_digest,
            "memory_action": self.memory_action,
        }


@dataclass(frozen=True, slots=True)
class HistoryAppliedTransactionPayload:
    receipt_digest: str
    history_row_digest: str
    history_target: NewTargetDescriptor
    cursor_target: NewTargetDescriptor

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "receipt_digest": self.receipt_digest,
            "history_row_digest": self.history_row_digest,
            "history_target": self.history_target.to_payload(),
            "cursor_target": self.cursor_target.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class MemoryAppliedTransactionPayload:
    receipt_digest: str
    memory_action: str
    memory_target: NewTargetDescriptor

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "receipt_digest": self.receipt_digest,
            "memory_action": self.memory_action,
            "memory_target": self.memory_target.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class SourceAppliedState:
    digest: str
    size: int
    generation: int
    revision: int
    cursor: int

    def to_payload(self) -> dict[str, int | str]:
        return {
            "digest": self.digest,
            "size": self.size,
            "generation": self.generation,
            "revision": self.revision,
            "cursor": self.cursor,
        }


@dataclass(frozen=True, slots=True)
class OffsetAppliedTransactionPayload:
    receipt_digest: str
    source: SourceAppliedState

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "receipt_digest": self.receipt_digest,
            "source": self.source.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class ReceiptCommittedTransactionPayload:
    receipt_digest: str
    prepared_record_digest: str
    history_applied_record_digest: str
    memory_applied_record_digest: str
    offset_applied_record_digest: str
    receipt_size: int
    receipt_present: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "receipt_digest": self.receipt_digest,
            "prepared_record_digest": self.prepared_record_digest,
            "history_applied_record_digest": self.history_applied_record_digest,
            "memory_applied_record_digest": self.memory_applied_record_digest,
            "offset_applied_record_digest": self.offset_applied_record_digest,
            "receipt_size": self.receipt_size,
            "receipt_present": self.receipt_present,
        }


@dataclass(frozen=True, slots=True)
class CompleteTransactionPayload:
    receipt_digest: str
    commit_record_digest: str
    receipt: OldTargetDescriptor
    history_target: NewTargetDescriptor
    cursor_target: NewTargetDescriptor
    memory_target: NewTargetDescriptor
    source: SourceAppliedState
    history_row_digest: str
    history_marker_digest: str
    marker_occurrences: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "receipt_digest": self.receipt_digest,
            "commit_record_digest": self.commit_record_digest,
            "receipt": self.receipt.to_payload(),
            "history_target": self.history_target.to_payload(),
            "cursor_target": self.cursor_target.to_payload(),
            "memory_target": self.memory_target.to_payload(),
            "source": self.source.to_payload(),
            "history_row_digest": self.history_row_digest,
            "history_marker_digest": self.history_marker_digest,
            "marker_occurrences": self.marker_occurrences,
        }


def _old_target(value: object) -> OldTargetDescriptor:
    if type(value) is not dict or set(value) != _OLD_TARGET_FIELDS:
        _fail()
    exists = value["exists"]
    if type(exists) is not bool:
        _fail()
    size = _counter(value["size"])
    digest = value["digest"]
    stat_value = value["stat"]
    if exists:
        digest = _digest(digest)
        parsed_stat = _stat_payload(stat_value, receipt=False)
        if parsed_stat.size != size:
            _fail()
        return OldTargetDescriptor(True, size, digest, parsed_stat)
    if size != 0 or digest is not None or stat_value is not None:
        _fail()
    return OldTargetDescriptor(False, 0, None, None)


def _new_target(value: object) -> NewTargetDescriptor:
    if type(value) is not dict or set(value) != _NEW_TARGET_FIELDS:
        _fail()
    exists = value["exists"]
    if type(exists) is not bool:
        _fail()
    size = _counter(value["size"])
    digest = value["blob_digest"]
    if exists:
        digest = _digest(digest)
        return NewTargetDescriptor(True, size, digest)
    if size != 0 or digest is not None:
        _fail()
    return NewTargetDescriptor(False, 0, None)


def _source_transition(value: object, captured: CapturedTransactionPayload) -> SourceTransition:
    if type(value) is not dict or set(value) != _SOURCE_FIELDS:
        _fail()
    old_digest = _digest(value["old_digest"])
    new_digest = _digest(value["new_digest"])
    old_size = _counter(value["old_size"])
    new_size = _counter(value["new_size"])
    old_generation = _counter(value["old_generation"])
    new_generation = _counter(value["new_generation"])
    old_revision = _counter(value["old_revision"])
    new_revision = _counter(value["new_revision"])
    old_cursor = _counter(value["old_cursor"])
    new_cursor = _counter(value["new_cursor"])
    action = value["action"]
    if type(action) is not str or action != "advance":
        _fail()
    if (
        old_generation != captured.session_generation
        or new_generation != captured.session_generation
        or old_cursor != captured.start_offset
        or new_cursor != captured.end_offset
        or old_revision == MAX_INT
        or new_revision != old_revision + 1
        or old_digest == new_digest
    ):
        _fail()
    return SourceTransition(
        old_digest,
        old_size,
        old_generation,
        old_revision,
        old_cursor,
        new_digest,
        new_size,
        new_generation,
        new_revision,
        new_cursor,
        action,
    )


def _prepare(
    captured: CapturedTransactionPayload,
    *,
    receipt_digest: object,
    receipt_stat: object,
    targets: object,
    source: object,
    history_marker_digest: object,
    history_entry_digest: object,
    history_row_digest: object,
    memory_action: object,
) -> PreparedTransactionPayload:
    if type(captured) is not CapturedTransactionPayload:
        _fail()
    captured = _validated_captured(captured)
    if _digest(receipt_digest) != captured.receipt_digest:
        _fail()
    parsed_stat = _stat_payload(receipt_stat, receipt=True)
    if parsed_stat.size != captured.receipt_size:
        _fail()
    if type(targets) is not dict or set(targets) != set(_TARGET_NAMES):
        _fail()
    parsed_targets: dict[str, TargetTransition] = {}
    for name in _TARGET_NAMES:
        raw = targets[name]
        if type(raw) is not dict or set(raw) != {"old", "new"}:
            _fail()
        parsed_targets[name] = TargetTransition(_old_target(raw["old"]), _new_target(raw["new"]))
    parsed_source = _source_transition(source, captured)
    marker_digest = _digest(history_marker_digest)
    entry_digest = _digest(history_entry_digest)
    if (
        marker_digest != captured.history_marker_digest
        or entry_digest != captured.history_entry_digest
    ):
        _fail()
    row_digest = _digest(history_row_digest)
    if type(memory_action) is not str:
        _fail()
    expected_action = "preserve" if captured.reference_only else "replace"
    if memory_action != expected_action:
        _fail()
    memory_target = parsed_targets["MEMORY.md"]
    if memory_target.old.exists:
        if memory_target.old.digest != captured.base_memory_digest:
            _fail()
    elif captured.base_memory_digest != hashlib.sha256(b"").hexdigest():
        _fail()
    if memory_action == "replace":
        if (
            not memory_target.new.exists
            or memory_target.new.blob_digest != captured.output_memory_digest
            or memory_target.new.size != captured.output_memory_size
        ):
            _fail()
    else:
        if (
            memory_target.new.exists != memory_target.old.exists
            or memory_target.new.size != memory_target.old.size
            or memory_target.new.blob_digest != memory_target.old.digest
        ):
            _fail()
    return PreparedTransactionPayload(
        captured=captured,
        receipt_stat=parsed_stat,
        targets=MappingProxyType(parsed_targets),
        source=parsed_source,
        history_row_digest=row_digest,
        memory_action=memory_action,
    )


def prepare_transaction_payload(
    captured: CapturedTransactionPayload,
    *,
    receipt_stat: Mapping[str, object],
    targets: Mapping[str, object],
    source: Mapping[str, object],
    history_marker_digest: str,
    history_entry_digest: str,
    history_row_digest: str,
    memory_action: str,
) -> PreparedTransactionPayload:
    """Build PREPARED evidence and bind it to the CAPTURED receipt."""

    try:
        prepared = _prepare(
            captured,
            receipt_digest=captured.receipt_digest,
            receipt_stat=receipt_stat,
            targets=targets,
            source=source,
            history_marker_digest=history_marker_digest,
            history_entry_digest=history_entry_digest,
            history_row_digest=history_row_digest,
            memory_action=memory_action,
        )
        payload = prepared.to_payload()
        _validate_record_bounds(payload, "PREPARED", captured.identity)
        derive_manifest_identity("transactions", payload)
        return prepared
    except MemoryRecoveryTransactionError:
        raise
    except Exception:
        _fail()


def _validated_captured(value: CapturedTransactionPayload) -> CapturedTransactionPayload:
    try:
        payload = value.to_payload()
        decoded = _captured_from_payload(payload)
        if decoded != value:
            _fail()
        identity = derive_manifest_identity("transactions", payload)
        _validate_record_bounds(payload, "CAPTURED", identity)
        return decoded
    except MemoryRecoveryTransactionError:
        raise
    except Exception:
        _fail()


def _captured_from_payload(payload: object) -> CapturedTransactionPayload:
    if type(payload) is not dict or set(payload) != _CAPTURED_FIELDS:
        _fail()
    if type(payload["schema_version"]) is not int or payload["schema_version"] != _SCHEMA_VERSION:
        _fail()
    if payload["source_format"] != _SOURCE_FORMAT or payload["conversion_identity"] is not None:
        _fail()
    receipt_digest = _digest(payload["receipt_digest"])
    receipt_size = _counter(payload["receipt_size"], positive=True)
    session_key_digest = _digest(payload["session_key_digest"])
    start_offset = _counter(payload["start_offset"])
    end_offset = _counter(payload["end_offset"])
    session_generation = _counter(payload["session_generation"])
    if start_offset >= end_offset:
        _fail()
    chunk_digest = _digest(payload["chunk_digest"])
    base_memory_digest = _digest(payload["base_memory_digest"])
    output_memory_digest = _digest(payload["output_memory_digest"])
    output_memory_size = _counter(payload["output_memory_size"])
    reference_only = payload["reference_only"]
    if type(reference_only) is not bool:
        _fail()
    marker_digest = _digest(payload["history_marker_digest"])
    entry_digest = _digest(payload["history_entry_digest"])
    created_at_digest = _digest(payload["created_at_digest"])
    return CapturedTransactionPayload(
        receipt_digest,
        receipt_size,
        session_key_digest,
        start_offset,
        end_offset,
        session_generation,
        chunk_digest,
        base_memory_digest,
        output_memory_digest,
        output_memory_size,
        reference_only,
        marker_digest,
        entry_digest,
        created_at_digest,
    )


def _prepared_from_payload(
    payload: object, captured: CapturedTransactionPayload
) -> PreparedTransactionPayload:
    if type(payload) is not dict or set(payload) != _PREPARED_FIELDS:
        _fail()
    if type(payload["schema_version"]) is not int or payload["schema_version"] != _SCHEMA_VERSION:
        _fail()
    return _prepare(
        captured,
        receipt_digest=payload["receipt_digest"],
        receipt_stat=payload["receipt_stat"],
        targets=payload["targets"],
        source=payload["source"],
        history_marker_digest=payload["history_marker_digest"],
        history_entry_digest=payload["history_entry_digest"],
        history_row_digest=payload["history_row_digest"],
        memory_action=payload["memory_action"],
    )


def _validate_record_bounds(payload: dict[str, Any], phase: str, identity: str) -> None:
    sequence = _TRANSACTION_PHASE_SEQUENCE.get(phase)
    if sequence is None:
        _fail()
    try:
        record = json.dumps(
            {
                "version": 1,
                "kind": "transactions",
                "identity": identity,
                "sequence": sequence,
                "predecessor_digest": None if sequence == 0 else "0" * 64,
                "phase": phase,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError):
        _fail()
    if len(record) > MAX_RECORD_BYTES:
        _fail()


def _parse_payload(value: object) -> dict[str, Any]:
    if type(value) is bytes:
        if len(value) > MAX_RECORD_BYTES:
            _fail()
        try:
            parsed = json.loads(
                value.decode("utf-8"),
                object_pairs_hook=receipt_codec._reject_duplicate_keys,
                parse_constant=receipt_codec._reject_non_json_number,
            )
        except Exception:
            _fail()
        if type(parsed) is not dict:
            _fail()
        return parsed
    if type(value) is not dict:
        _fail()
    return value


def _phase_fields(payload: object, fields: frozenset[str]) -> dict[str, Any]:
    if type(payload) is not dict or set(payload) != fields:
        _fail()
    if type(payload["schema_version"]) is not int or payload["schema_version"] != _SCHEMA_VERSION:
        _fail()
    return payload


def _transaction_context(
    captured: object, prepared: object
) -> tuple[CapturedTransactionPayload, PreparedTransactionPayload]:
    if type(captured) is not CapturedTransactionPayload:
        _fail()
    valid_captured = _validated_captured(captured)
    if type(prepared) is not PreparedTransactionPayload or prepared.captured != valid_captured:
        _fail()
    valid_prepared = _prepared_from_payload(prepared.to_payload(), valid_captured)
    if valid_prepared != prepared:
        _fail()
    return valid_captured, valid_prepared


def _source_applied(value: object) -> SourceAppliedState:
    if type(value) is not dict or set(value) != _SOURCE_APPLIED_FIELDS:
        _fail()
    return SourceAppliedState(
        _digest(value["digest"]),
        _counter(value["size"]),
        _counter(value["generation"]),
        _counter(value["revision"]),
        _counter(value["cursor"]),
    )


def _prepared_source_state(prepared: PreparedTransactionPayload) -> SourceAppliedState:
    source = prepared.source
    return SourceAppliedState(
        source.new_digest,
        source.new_size,
        source.new_generation,
        source.new_revision,
        source.new_cursor,
    )


def _history_applied_from_payload(
    payload: object,
    captured: CapturedTransactionPayload,
    prepared: PreparedTransactionPayload,
) -> HistoryAppliedTransactionPayload:
    captured, prepared = _transaction_context(captured, prepared)
    raw = _phase_fields(payload, _HISTORY_APPLIED_FIELDS)
    receipt_digest = _digest(raw["receipt_digest"])
    row_digest = _digest(raw["history_row_digest"])
    history_target = _new_target(raw["history_target"])
    cursor_target = _new_target(raw["cursor_target"])
    if (
        receipt_digest != captured.receipt_digest
        or not history_target.exists
        or not cursor_target.exists
        or row_digest != prepared.history_row_digest
        or history_target != prepared.targets["history.jsonl"].new
        or cursor_target != prepared.targets[".cursor"].new
    ):
        _fail()
    return HistoryAppliedTransactionPayload(
        receipt_digest, row_digest, history_target, cursor_target
    )


def _validated_history_applied(
    value: object,
    captured: CapturedTransactionPayload,
    prepared: PreparedTransactionPayload,
) -> HistoryAppliedTransactionPayload:
    if type(value) is not HistoryAppliedTransactionPayload:
        _fail()
    decoded = _history_applied_from_payload(value.to_payload(), captured, prepared)
    if decoded != value:
        _fail()
    return decoded


def _memory_applied_from_payload(
    payload: object,
    captured: CapturedTransactionPayload,
    prepared: PreparedTransactionPayload,
    history_applied: object,
) -> MemoryAppliedTransactionPayload:
    captured, prepared = _transaction_context(captured, prepared)
    _validated_history_applied(history_applied, captured, prepared)
    raw = _phase_fields(payload, _MEMORY_APPLIED_FIELDS)
    receipt_digest = _digest(raw["receipt_digest"])
    action = raw["memory_action"]
    target = _new_target(raw["memory_target"])
    if (
        receipt_digest != captured.receipt_digest
        or type(action) is not str
        or action != prepared.memory_action
        or target != prepared.targets["MEMORY.md"].new
    ):
        _fail()
    return MemoryAppliedTransactionPayload(receipt_digest, action, target)


def _validated_memory_applied(
    value: object,
    captured: CapturedTransactionPayload,
    prepared: PreparedTransactionPayload,
    history_applied: object,
) -> MemoryAppliedTransactionPayload:
    if type(value) is not MemoryAppliedTransactionPayload:
        _fail()
    decoded = _memory_applied_from_payload(
        value.to_payload(), captured, prepared, history_applied
    )
    if decoded != value:
        _fail()
    return decoded


def _offset_applied_from_payload(
    payload: object,
    captured: CapturedTransactionPayload,
    prepared: PreparedTransactionPayload,
    history_applied: object,
    memory_applied: object,
) -> OffsetAppliedTransactionPayload:
    captured, prepared = _transaction_context(captured, prepared)
    _validated_memory_applied(
        memory_applied, captured, prepared, history_applied
    )
    raw = _phase_fields(payload, _OFFSET_APPLIED_FIELDS)
    receipt_digest = _digest(raw["receipt_digest"])
    source = _source_applied(raw["source"])
    if receipt_digest != captured.receipt_digest or source != _prepared_source_state(prepared):
        _fail()
    return OffsetAppliedTransactionPayload(receipt_digest, source)


def _validated_offset_applied(
    value: object,
    captured: CapturedTransactionPayload,
    prepared: PreparedTransactionPayload,
    history_applied: object,
    memory_applied: object,
) -> OffsetAppliedTransactionPayload:
    if type(value) is not OffsetAppliedTransactionPayload:
        _fail()
    decoded = _offset_applied_from_payload(
        value.to_payload(), captured, prepared, history_applied, memory_applied
    )
    if decoded != value:
        _fail()
    return decoded


def _record_digest_inputs(
    prepared_record_digest: object,
    history_applied_record_digest: object,
    memory_applied_record_digest: object,
    offset_applied_record_digest: object,
) -> tuple[str, str, str, str]:
    return (
        _digest(prepared_record_digest),
        _digest(history_applied_record_digest),
        _digest(memory_applied_record_digest),
        _digest(offset_applied_record_digest),
    )


def _receipt_committed_from_payload(
    payload: object,
    captured: CapturedTransactionPayload,
    prepared: PreparedTransactionPayload,
    history_applied: object,
    memory_applied: object,
    offset_applied: object,
    *,
    prepared_record_digest: object,
    history_applied_record_digest: object,
    memory_applied_record_digest: object,
    offset_applied_record_digest: object,
) -> ReceiptCommittedTransactionPayload:
    captured, prepared = _transaction_context(captured, prepared)
    _validated_offset_applied(
        offset_applied, captured, prepared, history_applied, memory_applied
    )
    expected_digests = _record_digest_inputs(
        prepared_record_digest,
        history_applied_record_digest,
        memory_applied_record_digest,
        offset_applied_record_digest,
    )
    raw = _phase_fields(payload, _RECEIPT_COMMITTED_FIELDS)
    receipt_digest = _digest(raw["receipt_digest"])
    record_digests = _record_digest_inputs(
        raw["prepared_record_digest"],
        raw["history_applied_record_digest"],
        raw["memory_applied_record_digest"],
        raw["offset_applied_record_digest"],
    )
    receipt_size = _counter(raw["receipt_size"], positive=True)
    receipt_present = raw["receipt_present"]
    if (
        receipt_digest != captured.receipt_digest
        or record_digests != expected_digests
        or receipt_size != captured.receipt_size
        or type(receipt_present) is not bool
        or receipt_present is not True
    ):
        _fail()
    return ReceiptCommittedTransactionPayload(
        receipt_digest, *record_digests, receipt_size, receipt_present
    )


def _validated_receipt_committed(
    value: object,
    captured: CapturedTransactionPayload,
    prepared: PreparedTransactionPayload,
    history_applied: object,
    memory_applied: object,
    offset_applied: object,
    *,
    prepared_record_digest: object,
    history_applied_record_digest: object,
    memory_applied_record_digest: object,
    offset_applied_record_digest: object,
) -> ReceiptCommittedTransactionPayload:
    if type(value) is not ReceiptCommittedTransactionPayload:
        _fail()
    decoded = _receipt_committed_from_payload(
        value.to_payload(),
        captured,
        prepared,
        history_applied,
        memory_applied,
        offset_applied,
        prepared_record_digest=prepared_record_digest,
        history_applied_record_digest=history_applied_record_digest,
        memory_applied_record_digest=memory_applied_record_digest,
        offset_applied_record_digest=offset_applied_record_digest,
    )
    if decoded != value:
        _fail()
    return decoded


def _complete_from_payload(
    payload: object,
    captured: CapturedTransactionPayload,
    prepared: PreparedTransactionPayload,
    history_applied: object,
    memory_applied: object,
    offset_applied: object,
    receipt_committed: object,
    *,
    commit_record_digest: object,
    prepared_record_digest: object,
    history_applied_record_digest: object,
    memory_applied_record_digest: object,
    offset_applied_record_digest: object,
) -> CompleteTransactionPayload:
    captured, prepared = _transaction_context(captured, prepared)
    _validated_receipt_committed(
        receipt_committed,
        captured,
        prepared,
        history_applied,
        memory_applied,
        offset_applied,
        prepared_record_digest=prepared_record_digest,
        history_applied_record_digest=history_applied_record_digest,
        memory_applied_record_digest=memory_applied_record_digest,
        offset_applied_record_digest=offset_applied_record_digest,
    )
    raw = _phase_fields(payload, _COMPLETE_FIELDS)
    receipt_digest = _digest(raw["receipt_digest"])
    parsed_commit_digest = _digest(raw["commit_record_digest"])
    expected_commit_digest = _digest(commit_record_digest)
    receipt = _old_target(raw["receipt"])
    history_target = _new_target(raw["history_target"])
    cursor_target = _new_target(raw["cursor_target"])
    memory_target = _new_target(raw["memory_target"])
    source = _source_applied(raw["source"])
    row_digest = _digest(raw["history_row_digest"])
    marker_digest = _digest(raw["history_marker_digest"])
    marker_occurrences = _counter(raw["marker_occurrences"], positive=True)
    expected_receipt = OldTargetDescriptor(
        True, captured.receipt_size, captured.receipt_digest, prepared.receipt_stat
    )
    if (
        receipt_digest != captured.receipt_digest
        or parsed_commit_digest != expected_commit_digest
        or receipt != expected_receipt
        or history_target != prepared.targets["history.jsonl"].new
        or cursor_target != prepared.targets[".cursor"].new
        or memory_target != prepared.targets["MEMORY.md"].new
        or source != _prepared_source_state(prepared)
        or row_digest != prepared.history_row_digest
        or marker_digest != captured.history_marker_digest
        or marker_occurrences != 1
    ):
        _fail()
    return CompleteTransactionPayload(
        receipt_digest,
        parsed_commit_digest,
        receipt,
        history_target,
        cursor_target,
        memory_target,
        source,
        row_digest,
        marker_digest,
        marker_occurrences,
    )


def _validate_built_payload(
    phase: str, payload: dict[str, Any], captured: CapturedTransactionPayload
) -> None:
    try:
        derive_manifest_identity("transactions", payload)
    except Exception:
        _fail()
    _validate_record_bounds(payload, phase, captured.identity)


def history_applied_transaction_payload(
    captured: CapturedTransactionPayload, prepared: PreparedTransactionPayload
) -> HistoryAppliedTransactionPayload:
    """Build HISTORY_APPLIED evidence from validated PREPARED descriptors."""

    captured, prepared = _transaction_context(captured, prepared)
    payload = HistoryAppliedTransactionPayload(
        captured.receipt_digest,
        prepared.history_row_digest,
        prepared.targets["history.jsonl"].new,
        prepared.targets[".cursor"].new,
    ).to_payload()
    _validate_built_payload("HISTORY_APPLIED", payload, captured)
    return _history_applied_from_payload(payload, captured, prepared)


def memory_applied_transaction_payload(
    captured: CapturedTransactionPayload,
    prepared: PreparedTransactionPayload,
    history_applied: HistoryAppliedTransactionPayload,
) -> MemoryAppliedTransactionPayload:
    """Build MEMORY_APPLIED evidence after typed history proof."""

    captured, prepared = _transaction_context(captured, prepared)
    history = _validated_history_applied(history_applied, captured, prepared)
    payload = MemoryAppliedTransactionPayload(
        captured.receipt_digest,
        prepared.memory_action,
        prepared.targets["MEMORY.md"].new,
    ).to_payload()
    _validate_built_payload("MEMORY_APPLIED", payload, captured)
    return _memory_applied_from_payload(payload, captured, prepared, history)


def offset_applied_transaction_payload(
    captured: CapturedTransactionPayload,
    prepared: PreparedTransactionPayload,
    history_applied: HistoryAppliedTransactionPayload,
    memory_applied: MemoryAppliedTransactionPayload,
) -> OffsetAppliedTransactionPayload:
    """Build OFFSET_APPLIED evidence after typed history and memory proofs."""

    captured, prepared = _transaction_context(captured, prepared)
    history = _validated_history_applied(history_applied, captured, prepared)
    memory = _validated_memory_applied(memory_applied, captured, prepared, history)
    payload = OffsetAppliedTransactionPayload(
        captured.receipt_digest, _prepared_source_state(prepared)
    ).to_payload()
    _validate_built_payload("OFFSET_APPLIED", payload, captured)
    return _offset_applied_from_payload(payload, captured, prepared, history, memory)


def receipt_committed_transaction_payload(
    captured: CapturedTransactionPayload,
    prepared: PreparedTransactionPayload,
    history_applied: HistoryAppliedTransactionPayload,
    memory_applied: MemoryAppliedTransactionPayload,
    offset_applied: OffsetAppliedTransactionPayload,
    *,
    prepared_record_digest: str,
    history_applied_record_digest: str,
    memory_applied_record_digest: str,
    offset_applied_record_digest: str,
) -> ReceiptCommittedTransactionPayload:
    """Build commit evidence using manifest-supplied predecessor record digests."""

    captured, prepared = _transaction_context(captured, prepared)
    history = _validated_history_applied(history_applied, captured, prepared)
    memory = _validated_memory_applied(memory_applied, captured, prepared, history)
    offset = _validated_offset_applied(
        offset_applied, captured, prepared, history, memory
    )
    payload = ReceiptCommittedTransactionPayload(
        captured.receipt_digest,
        *_record_digest_inputs(
            prepared_record_digest,
            history_applied_record_digest,
            memory_applied_record_digest,
            offset_applied_record_digest,
        ),
        captured.receipt_size,
        True,
    ).to_payload()
    _validate_built_payload("RECEIPT_COMMITTED", payload, captured)
    return _receipt_committed_from_payload(
        payload,
        captured,
        prepared,
        history,
        memory,
        offset,
        prepared_record_digest=prepared_record_digest,
        history_applied_record_digest=history_applied_record_digest,
        memory_applied_record_digest=memory_applied_record_digest,
        offset_applied_record_digest=offset_applied_record_digest,
    )


def complete_transaction_payload(
    captured: CapturedTransactionPayload,
    prepared: PreparedTransactionPayload,
    history_applied: HistoryAppliedTransactionPayload,
    memory_applied: MemoryAppliedTransactionPayload,
    offset_applied: OffsetAppliedTransactionPayload,
    receipt_committed: ReceiptCommittedTransactionPayload,
    *,
    commit_record_digest: str,
    prepared_record_digest: str,
    history_applied_record_digest: str,
    memory_applied_record_digest: str,
    offset_applied_record_digest: str,
) -> CompleteTransactionPayload:
    """Build final proof from typed phase records and explicit manifest digests."""

    captured, prepared = _transaction_context(captured, prepared)
    history = _validated_history_applied(history_applied, captured, prepared)
    memory = _validated_memory_applied(memory_applied, captured, prepared, history)
    offset = _validated_offset_applied(
        offset_applied, captured, prepared, history, memory
    )
    committed = _validated_receipt_committed(
        receipt_committed,
        captured,
        prepared,
        history,
        memory,
        offset,
        prepared_record_digest=prepared_record_digest,
        history_applied_record_digest=history_applied_record_digest,
        memory_applied_record_digest=memory_applied_record_digest,
        offset_applied_record_digest=offset_applied_record_digest,
    )
    payload = CompleteTransactionPayload(
        captured.receipt_digest,
        _digest(commit_record_digest),
        OldTargetDescriptor(
            True, captured.receipt_size, captured.receipt_digest, prepared.receipt_stat
        ),
        prepared.targets["history.jsonl"].new,
        prepared.targets[".cursor"].new,
        prepared.targets["MEMORY.md"].new,
        _prepared_source_state(prepared),
        prepared.history_row_digest,
        captured.history_marker_digest,
        1,
    ).to_payload()
    _validate_built_payload("COMPLETE", payload, captured)
    return _complete_from_payload(
        payload,
        captured,
        prepared,
        history,
        memory,
        offset,
        committed,
        commit_record_digest=commit_record_digest,
        prepared_record_digest=prepared_record_digest,
        history_applied_record_digest=history_applied_record_digest,
        memory_applied_record_digest=memory_applied_record_digest,
        offset_applied_record_digest=offset_applied_record_digest,
    )


def decode_transaction_payload(
    phase: str,
    payload: object,
    *,
    captured: CapturedTransactionPayload | None = None,
    prepared: PreparedTransactionPayload | None = None,
    history_applied: HistoryAppliedTransactionPayload | None = None,
    memory_applied: MemoryAppliedTransactionPayload | None = None,
    offset_applied: OffsetAppliedTransactionPayload | None = None,
    receipt_committed: ReceiptCommittedTransactionPayload | None = None,
    commit_record_digest: str | None = None,
    prepared_record_digest: str | None = None,
    history_applied_record_digest: str | None = None,
    memory_applied_record_digest: str | None = None,
    offset_applied_record_digest: str | None = None,
) -> CapturedTransactionPayload | PreparedTransactionPayload | HistoryAppliedTransactionPayload | MemoryAppliedTransactionPayload | OffsetAppliedTransactionPayload | ReceiptCommittedTransactionPayload | CompleteTransactionPayload:
    """Strictly decode typed phase evidence without state or filesystem I/O.

    Every phase after CAPTURED requires its typed predecessors. Record digests
    that come from the manifest are explicit inputs and are never reconstructed
    from the phase payloads.
    """

    try:
        raw = _parse_payload(payload)
        # The manifest API is authoritative for JSON-compatible types, UTF-8
        # string/key limits, depth, nodes, signed integers, and canonical bytes.
        derive_manifest_identity("transactions", raw)
        if phase == "CAPTURED":
            decoded = _captured_from_payload(raw)
            if captured is not None:
                _fail()
            _validate_record_bounds(raw, phase, decoded.identity)
            return decoded
        if phase == "PREPARED":
            if captured is None:
                _fail()
            valid_captured = _validated_captured(captured)
            decoded = _prepared_from_payload(raw, valid_captured)
            _validate_record_bounds(raw, phase, valid_captured.identity)
            return decoded
        if captured is None or prepared is None:
            _fail()
        valid_captured, valid_prepared = _transaction_context(captured, prepared)
        if phase == "HISTORY_APPLIED":
            decoded = _history_applied_from_payload(raw, valid_captured, valid_prepared)
        elif phase == "MEMORY_APPLIED":
            if history_applied is None:
                _fail()
            decoded = _memory_applied_from_payload(
                raw, valid_captured, valid_prepared, history_applied
            )
        elif phase == "OFFSET_APPLIED":
            if history_applied is None or memory_applied is None:
                _fail()
            decoded = _offset_applied_from_payload(
                raw, valid_captured, valid_prepared, history_applied, memory_applied
            )
        elif phase == "RECEIPT_COMMITTED":
            if (
                history_applied is None
                or memory_applied is None
                or offset_applied is None
                or prepared_record_digest is None
                or history_applied_record_digest is None
                or memory_applied_record_digest is None
                or offset_applied_record_digest is None
            ):
                _fail()
            decoded = _receipt_committed_from_payload(
                raw,
                valid_captured,
                valid_prepared,
                history_applied,
                memory_applied,
                offset_applied,
                prepared_record_digest=prepared_record_digest,
                history_applied_record_digest=history_applied_record_digest,
                memory_applied_record_digest=memory_applied_record_digest,
                offset_applied_record_digest=offset_applied_record_digest,
            )
        elif phase == "COMPLETE":
            if (
                history_applied is None
                or memory_applied is None
                or offset_applied is None
                or receipt_committed is None
                or commit_record_digest is None
                or prepared_record_digest is None
                or history_applied_record_digest is None
                or memory_applied_record_digest is None
                or offset_applied_record_digest is None
            ):
                _fail()
            decoded = _complete_from_payload(
                raw,
                valid_captured,
                valid_prepared,
                history_applied,
                memory_applied,
                offset_applied,
                receipt_committed,
                commit_record_digest=commit_record_digest,
                prepared_record_digest=prepared_record_digest,
                history_applied_record_digest=history_applied_record_digest,
                memory_applied_record_digest=memory_applied_record_digest,
                offset_applied_record_digest=offset_applied_record_digest,
            )
        else:
            _fail()
        _validate_record_bounds(raw, phase, valid_captured.identity)
        return decoded
    except MemoryRecoveryTransactionError:
        raise
    except (MemoryManifestError, TypeError, ValueError, UnicodeError, OverflowError):
        _fail()
    except Exception:
        _fail()


Payload: TypeAlias = (
    CapturedTransactionPayload
    | PreparedTransactionPayload
    | HistoryAppliedTransactionPayload
    | MemoryAppliedTransactionPayload
    | OffsetAppliedTransactionPayload
    | ReceiptCommittedTransactionPayload
    | CompleteTransactionPayload
)

__all__ = [
    "CapturedTransactionPayload",
    "CompleteTransactionPayload",
    "HistoryAppliedTransactionPayload",
    "MemoryAppliedTransactionPayload",
    "MemoryFileStat",
    "MemoryRecoveryTransactionError",
    "NewTargetDescriptor",
    "OffsetAppliedTransactionPayload",
    "OldTargetDescriptor",
    "Payload",
    "PreparedTransactionPayload",
    "ReceiptCommittedTransactionPayload",
    "SourceAppliedState",
    "SourceTransition",
    "TargetTransition",
    "capture_transaction_payload",
    "complete_transaction_payload",
    "decode_transaction_payload",
    "history_applied_transaction_payload",
    "memory_applied_transaction_payload",
    "offset_applied_transaction_payload",
    "prepare_transaction_payload",
    "receipt_committed_transaction_payload",
]
