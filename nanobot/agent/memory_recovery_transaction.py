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
    try:
        record = json.dumps(
            {
                "version": 1,
                "kind": "transactions",
                "identity": identity,
                "sequence": 0 if phase == "CAPTURED" else 1,
                "predecessor_digest": None if phase == "CAPTURED" else "0" * 64,
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


def decode_transaction_payload(
    phase: str,
    payload: object,
    *,
    captured: CapturedTransactionPayload | None = None,
) -> CapturedTransactionPayload | PreparedTransactionPayload:
    """Strictly decode CAPTURED/PREPARED payloads without state or filesystem I/O.

    PREPARED is a successor to CAPTURED, so its decoder requires the typed
    captured payload to check receipt, offset, generation, marker, and entry
    digest parity.
    """

    try:
        raw = _parse_payload(payload)
        # The manifest API is authoritative for JSON-compatible types, UTF-8
        # string/key limits, depth, nodes, signed integers, and canonical bytes.
        identity = derive_manifest_identity("transactions", raw)
        if phase == "CAPTURED":
            decoded = _captured_from_payload(raw)
            if captured is not None:
                _fail()
            _validate_record_bounds(raw, phase, identity)
            return decoded
        if phase == "PREPARED":
            if type(captured) is not CapturedTransactionPayload:
                _fail()
            decoded = _prepared_from_payload(raw, captured)
            _validate_record_bounds(raw, phase, captured.identity)
            return decoded
        _fail()
    except MemoryRecoveryTransactionError:
        raise
    except (MemoryManifestError, TypeError, ValueError, UnicodeError, OverflowError):
        _fail()
    except Exception:
        _fail()


Payload: TypeAlias = CapturedTransactionPayload | PreparedTransactionPayload

__all__ = [
    "CapturedTransactionPayload",
    "MemoryFileStat",
    "MemoryRecoveryTransactionError",
    "NewTargetDescriptor",
    "OldTargetDescriptor",
    "Payload",
    "PreparedTransactionPayload",
    "SourceTransition",
    "TargetTransition",
    "capture_transaction_payload",
    "decode_transaction_payload",
    "prepare_transaction_payload",
]
