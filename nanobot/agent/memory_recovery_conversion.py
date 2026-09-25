"""Pure typed evidence codec for legacy HISTORY.md conversion recovery.

This module describes captured input and deterministic output claims only. It
never reads the workspace, publishes blobs, applies mutations, or treats parsed
history rows as proof that a receipt was archived.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent.legacy_history_codec import (
    MAX_LEGACY_HISTORY_BYTES,
    LegacyHistoryEntry,
    parse_legacy_history,
)
from nanobot.agent.memory_recovery import MemoryReceipt
from nanobot.agent.memory_recovery_manifest import (
    MAX_INT,
    MAX_NODES,
    MAX_RECORD_BYTES,
    MAX_STRING_BYTES,
    ManifestRecord,
    MemoryManifestError,
    derive_manifest_identity,
)
from nanobot.agent.memory_recovery_marker import classify_history_marker_frame
from nanobot.agent.memory_recovery_transaction import (
    CapturedTransactionPayload,
    decode_transaction_payload,
)

_SCHEMA_VERSION = 1
_CONVERSION_POLICY = "legacy_history_v1"
_RECEIPT_TAIL_POLICY = "legacy_history_receipt_tail_v2"
_TARGET_NAMES = ("history.jsonl", ".cursor", ".dream_cursor")
_STAT_FIELDS = frozenset(
    ("device", "inode", "mode", "uid", "gid", "links", "size", "mtime_ns", "ctime_ns")
)
_CAPTURED_FIELDS = frozenset(
    ("schema_version", "conversion_policy", "receipt_transaction_identity")
)
_PREPARED_FIELDS = frozenset(
    (
        "schema_version",
        "conversion_policy",
        "conversion_identity",
        "disposition",
        "source",
        "targets",
        "backup",
        "fallback_timestamp",
        "row_count",
        "final_legacy_cursor",
        "sidecar_counters",
        "outputs",
    )
)
_V2_EVIDENCE_FIELDS = frozenset(
    ("receipt", "transaction_captured_record_digest", "tail_frame", "canonical_row")
)
_PHASE_BASE_FIELDS = frozenset(("schema_version", "conversion_identity", "disposition"))
_JSONL_APPLIED_FIELDS = _PHASE_BASE_FIELDS | frozenset(("history_target",))
_CURSORS_APPLIED_FIELDS = _PHASE_BASE_FIELDS | frozenset(
    ("history_target", "cursor_target", "dream_cursor_target")
)
_LEGACY_FINALIZED_FIELDS = _PHASE_BASE_FIELDS | frozenset(
    (
        "prepared_record_digest",
        "jsonl_applied_record_digest",
        "cursors_applied_record_digest",
        "targets",
        "source",
        "backup_basename",
        "backup",
    )
)
_COMPLETE_FIELDS = _PHASE_BASE_FIELDS | frozenset(
    ("finalized_record_digest", "targets", "source", "backup_basename", "backup")
)
_FILE_FIELDS = frozenset(("exists", "size", "digest", "stat", "blob_digest"))
_STAT_COUNTER_FIELDS = frozenset(("device", "inode", "mode", "uid", "gid", "links", "size"))
_NEW_FILE_FIELDS = frozenset(("exists", "size", "blob_digest"))
_BACKUP_FIELDS = frozenset(("basename", "candidates"))
_BACKUP_CANDIDATE_FIELDS = frozenset(("basename", "exists"))
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_FALLBACK_TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\Z")
_BACKUP_RE = re.compile(r"HISTORY\.md\.bak(?:\.([2-9][0-9]*))?\Z")
_CONVERSION_PHASE_SEQUENCE = {
    "CAPTURED": 0,
    "CONVERSION_PREPARED": 1,
    "JSONL_APPLIED": 2,
    "CURSORS_APPLIED": 3,
    "LEGACY_FINALIZED": 4,
    "COMPLETE": 5,
}


class ConversionError(ValueError):
    """Content-free validation error for legacy conversion evidence."""

    def __init__(self) -> None:
        super().__init__("memory recovery conversion payload is invalid")


def _fail() -> None:
    raise ConversionError() from None


def _digest(value: object) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        _fail()
    return value


def _counter(value: object, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0) or value > MAX_INT:
        _fail()
    return value


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _check_record_bounds(payload: dict[str, Any], phase: str, identity: str) -> None:
    sequence = _CONVERSION_PHASE_SEQUENCE.get(phase)
    if sequence is None:
        _fail()
    try:
        encoded = json.dumps(
            {
                "version": 1,
                "kind": "conversions",
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
    except (TypeError, ValueError, OverflowError, UnicodeError):
        _fail()
    if len(encoded) > MAX_RECORD_BYTES:
        _fail()


def _validate_built_payload(
    phase: str,
    payload: dict[str, Any],
    captured: CapturedConversionPayload,
) -> None:
    try:
        derive_manifest_identity("conversions", payload)
        _check_record_bounds(payload, phase, captured.identity)
    except Exception:
        _fail()


def _stat_from_payload(value: object) -> "MemoryFileStat":
    if type(value) is not dict or set(value) != _STAT_FIELDS:
        _fail()
    counters = {key: _counter(value[key]) for key in _STAT_COUNTER_FIELDS}
    if counters["links"] == 0:
        _fail()
    timestamps: dict[str, int | None] = {}
    for name in ("mtime_ns", "ctime_ns"):
        timestamp = value[name]
        if timestamp is not None and (
            type(timestamp) is not int or timestamp < -MAX_INT or timestamp > MAX_INT
        ):
            _fail()
        timestamps[name] = timestamp
    return MemoryFileStat(
        device=counters["device"],
        inode=counters["inode"],
        mode=counters["mode"],
        uid=counters["uid"],
        gid=counters["gid"],
        links=counters["links"],
        size=counters["size"],
        mtime_ns=timestamps["mtime_ns"],
        ctime_ns=timestamps["ctime_ns"],
    )


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
        return {
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "uid": self.uid,
            "gid": self.gid,
            "links": self.links,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True, slots=True)
class OldFileDescriptor:
    exists: bool
    size: int
    digest: str | None
    stat: MemoryFileStat | None
    blob_digest: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "exists": self.exists,
            "size": self.size,
            "digest": self.digest,
            "stat": None if self.stat is None else self.stat.to_payload(),
            "blob_digest": self.blob_digest,
        }


@dataclass(frozen=True, slots=True)
class NewFileDescriptor:
    exists: bool
    size: int
    blob_digest: str | None

    def to_payload(self) -> dict[str, Any]:
        return {"exists": self.exists, "size": self.size, "blob_digest": self.blob_digest}


@dataclass(frozen=True, slots=True)
class BackupCandidate:
    basename: str
    exists: bool

    def to_payload(self) -> dict[str, Any]:
        return {"basename": self.basename, "exists": self.exists}


@dataclass(frozen=True, slots=True)
class BackupPlan:
    basename: str
    candidates: tuple[BackupCandidate, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "basename": self.basename,
            "candidates": [candidate.to_payload() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class CapturedConversionPayload:
    receipt_transaction_identity: str
    schema_version: int = _SCHEMA_VERSION

    @property
    def identity(self) -> str:
        return derive_manifest_identity("conversions", self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "conversion_policy": _CONVERSION_POLICY if self.schema_version == 1 else _RECEIPT_TAIL_POLICY,
            "receipt_transaction_identity": self.receipt_transaction_identity,
        }


@dataclass(frozen=True, slots=True)
class JsonlAppliedConversionPayload:
    conversion_identity: str
    disposition: str
    history_target: NewFileDescriptor

    schema_version: int = _SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "conversion_identity": self.conversion_identity,
            "disposition": self.disposition,
            "history_target": self.history_target.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class CursorsAppliedConversionPayload:
    conversion_identity: str
    disposition: str
    history_target: NewFileDescriptor
    cursor_target: NewFileDescriptor
    dream_cursor_target: NewFileDescriptor

    schema_version: int = _SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "conversion_identity": self.conversion_identity,
            "disposition": self.disposition,
            "history_target": self.history_target.to_payload(),
            "cursor_target": self.cursor_target.to_payload(),
            "dream_cursor_target": self.dream_cursor_target.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class LegacyFinalizedConversionPayload:
    conversion_identity: str
    disposition: str
    prepared_record_digest: str
    jsonl_applied_record_digest: str
    cursors_applied_record_digest: str
    targets: Mapping[str, NewFileDescriptor]
    source: NewFileDescriptor
    backup_basename: str | None
    backup: NewFileDescriptor | None

    schema_version: int = _SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "conversion_identity": self.conversion_identity,
            "disposition": self.disposition,
            "prepared_record_digest": self.prepared_record_digest,
            "jsonl_applied_record_digest": self.jsonl_applied_record_digest,
            "cursors_applied_record_digest": self.cursors_applied_record_digest,
            "targets": {name: self.targets[name].to_payload() for name in _TARGET_NAMES},
            "source": self.source.to_payload(),
            "backup_basename": self.backup_basename,
            "backup": None if self.backup is None else self.backup.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class CompleteConversionPayload:
    conversion_identity: str
    disposition: str
    finalized_record_digest: str
    targets: Mapping[str, NewFileDescriptor]
    source: NewFileDescriptor
    backup_basename: str | None
    backup: NewFileDescriptor | None

    schema_version: int = _SCHEMA_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "conversion_identity": self.conversion_identity,
            "disposition": self.disposition,
            "finalized_record_digest": self.finalized_record_digest,
            "targets": {name: self.targets[name].to_payload() for name in _TARGET_NAMES},
            "source": self.source.to_payload(),
            "backup_basename": self.backup_basename,
            "backup": None if self.backup is None else self.backup.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class PreparedConversionPayload:
    captured: CapturedConversionPayload
    disposition: str
    source: OldFileDescriptor
    targets: Mapping[str, OldFileDescriptor]
    backup: BackupPlan | None
    fallback_timestamp: str | None
    row_count: int | None
    final_legacy_cursor: int | None
    sidecar_counters: Mapping[str, int] | None
    outputs: Mapping[str, NewFileDescriptor] | None

    receipt: Mapping[str, Any] | None = None
    transaction_captured_record_digest: str | None = None
    tail_frame: Mapping[str, Any] | None = None
    canonical_row: Mapping[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.captured.schema_version,
            "conversion_policy": _CONVERSION_POLICY if self.captured.schema_version == 1 else _RECEIPT_TAIL_POLICY,
            "conversion_identity": self.captured.identity,
            "disposition": self.disposition,
            "source": self.source.to_payload(),
            "targets": {name: self.targets[name].to_payload() for name in _TARGET_NAMES},
            "backup": None if self.backup is None else self.backup.to_payload(),
            "fallback_timestamp": self.fallback_timestamp,
            "row_count": self.row_count,
            "final_legacy_cursor": self.final_legacy_cursor,
            "sidecar_counters": (
                None if self.sidecar_counters is None else dict(self.sidecar_counters)
            ),
            "outputs": (
                None
                if self.outputs is None
                else {name: self.outputs[name].to_payload() for name in _TARGET_NAMES}
            ),
        }
        if self.captured.schema_version == 2 and self.disposition == "convert":
            payload.update({
                "receipt": None if self.receipt is None else dict(self.receipt),
                "transaction_captured_record_digest": self.transaction_captured_record_digest,
                "tail_frame": None if self.tail_frame is None else dict(self.tail_frame),
                "canonical_row": None if self.canonical_row is None else dict(self.canonical_row),
            })
        return payload


def capture_conversion_payload(receipt_transaction_identity: object) -> CapturedConversionPayload:
    """Construct exact v1 CAPTURED evidence for one receipt transaction."""

    try:
        captured = CapturedConversionPayload(_digest(receipt_transaction_identity))
        payload = captured.to_payload()
        identity = derive_manifest_identity("conversions", payload)
        _check_record_bounds(payload, "CAPTURED", identity)
        return captured
    except ConversionError:
        raise
    except Exception:
        _fail()


def capture_receipt_tail_conversion_payload(
    receipt_transaction_identity: object,
) -> CapturedConversionPayload:
    """Capture a receipt-tail conversion in a distinct v2 manifest namespace."""
    try:
        captured = CapturedConversionPayload(_digest(receipt_transaction_identity), 2)
        _validate_built_payload("CAPTURED", captured.to_payload(), captured)
        return captured
    except ConversionError:
        raise
    except Exception:
        _fail()


def _old_file(value: object) -> OldFileDescriptor:
    if type(value) is not dict or set(value) != _FILE_FIELDS:
        _fail()
    exists = value["exists"]
    if type(exists) is not bool:
        _fail()
    size = _counter(value["size"])
    if not exists:
        if any(value[key] is not None for key in ("digest", "stat", "blob_digest")) or size != 0:
            _fail()
        return OldFileDescriptor(False, 0, None, None, None)
    digest = _digest(value["digest"])
    blob_digest = _digest(value["blob_digest"])
    stat_value = _stat_from_payload(value["stat"])
    if stat_value.size != size or digest != blob_digest:
        _fail()
    return OldFileDescriptor(True, size, digest, stat_value, blob_digest)


def _new_file(value: object) -> NewFileDescriptor:
    if type(value) is not dict or set(value) != _NEW_FILE_FIELDS:
        _fail()
    exists = value["exists"]
    if type(exists) is not bool:
        _fail()
    size = _counter(value["size"])
    blob_digest = value["blob_digest"]
    if not exists:
        if size != 0 or blob_digest is not None:
            _fail()
        return NewFileDescriptor(False, 0, None)
    return NewFileDescriptor(True, size, _digest(blob_digest))


def _fallback_timestamp(value: object) -> str:
    if type(value) is not str or _FALLBACK_TIMESTAMP_RE.fullmatch(value) is None:
        _fail()
    try:
        datetime.strptime(value, "%Y-%m-%d %H:%M")
    except ValueError:
        _fail()
    return value


def _backup_index(basename: str) -> int:
    match = _BACKUP_RE.fullmatch(basename)
    if match is None:
        _fail()
    suffix = match.group(1)
    try:
        index = 1 if suffix is None else int(suffix)
    except ValueError:
        _fail()
    if index < 1 or index > MAX_NODES:
        _fail()
    return index


def _backup_plan(value: object) -> BackupPlan:
    if type(value) is not dict or set(value) != _BACKUP_FIELDS:
        _fail()
    basename = value["basename"]
    if type(basename) is not str:
        _fail()
    selected_index = _backup_index(basename)
    candidates_value = value["candidates"]
    if type(candidates_value) is not list or not candidates_value:
        _fail()
    if len(candidates_value) > MAX_NODES:
        _fail()
    if len(candidates_value) != selected_index:
        _fail()

    candidates: list[BackupCandidate] = []
    for index, candidate_value in enumerate(candidates_value, start=1):
        if type(candidate_value) is not dict or set(candidate_value) != _BACKUP_CANDIDATE_FIELDS:
            _fail()
        candidate_basename = candidate_value["basename"]
        candidate_exists = candidate_value["exists"]
        if type(candidate_basename) is not str or type(candidate_exists) is not bool:
            _fail()
        expected_name = "HISTORY.md.bak" if index == 1 else f"HISTORY.md.bak.{index}"
        if candidate_basename != expected_name:
            _fail()
        if candidate_exists != (index < selected_index):
            _fail()
        candidates.append(BackupCandidate(candidate_basename, candidate_exists))
    if candidates[-1].basename != basename or candidates[-1].exists:
        _fail()
    return BackupPlan(basename, tuple(candidates))


def _target_map(value: object) -> Mapping[str, OldFileDescriptor]:
    if type(value) is not dict or set(value) != set(_TARGET_NAMES):
        _fail()
    return MappingProxyType({name: _old_file(value[name]) for name in _TARGET_NAMES})


def _verify_target_contents(
    targets: Mapping[str, OldFileDescriptor],
    contents_value: object,
) -> dict[str, bytes | None]:
    if type(contents_value) is not dict or set(contents_value) != set(_TARGET_NAMES):
        _fail()
    verified: dict[str, bytes | None] = {}
    for name in _TARGET_NAMES:
        descriptor = targets[name]
        contents = contents_value[name]
        if not descriptor.exists:
            if contents is not None:
                _fail()
            verified[name] = None
            continue
        if type(contents) is not bytes:
            _fail()
        if len(contents) > MAX_STRING_BYTES:
            _fail()
        if len(contents) != descriptor.size or _sha256(contents) != descriptor.digest:
            _fail()
        verified[name] = contents
    return verified


def _counter_contents(
    targets: Mapping[str, OldFileDescriptor],
    contents_value: object,
) -> dict[str, int]:
    verified = _verify_target_contents(targets, contents_value)
    parsed: dict[str, int] = {}
    for name in _TARGET_NAMES:
        descriptor = targets[name]
        contents = verified[name]
        if not descriptor.exists:
            parsed[name] = 0
            continue
        if contents is None:
            _fail()
        if name == "history.jsonl":
            if contents:
                _fail()
            continue
        try:
            value = int(contents.decode("utf-8").strip())
        except (UnicodeError, ValueError):
            _fail()
        parsed[name] = _counter(value)
    if parsed[".dream_cursor"] > parsed[".cursor"]:
        _fail()
    return parsed


def _verify_source_bytes(source: OldFileDescriptor, source_bytes: object) -> bytes:
    if not source.exists or type(source_bytes) is not bytes:
        _fail()
    if len(source_bytes) != source.size or _sha256(source_bytes) != source.digest:
        _fail()
    if len(source_bytes) > MAX_LEGACY_HISTORY_BYTES:
        _fail()
    return source_bytes


def _preview_outputs(
    rows: tuple[LegacyHistoryEntry, ...],
    targets: Mapping[str, OldFileDescriptor],
) -> Mapping[str, NewFileDescriptor]:
    if not rows:
        return MappingProxyType(
            {
                name: NewFileDescriptor(
                    targets[name].exists,
                    targets[name].size,
                    targets[name].blob_digest,
                )
                for name in _TARGET_NAMES
            }
        )
    final_cursor = rows[-1].cursor
    jsonl = bytearray()
    for row in rows:
        record = {"cursor": row.cursor, "timestamp": row.timestamp, "content": row.content}
        try:
            line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        except (TypeError, ValueError, UnicodeError, OverflowError):
            _fail()
        if len(jsonl) + len(line) > MAX_LEGACY_HISTORY_BYTES:
            _fail()
        jsonl.extend(line)
    history_bytes = bytes(jsonl)
    cursor_bytes = str(final_cursor).encode("utf-8")
    return MappingProxyType(
        {
            "history.jsonl": NewFileDescriptor(True, len(history_bytes), _sha256(history_bytes)),
            ".cursor": NewFileDescriptor(True, len(cursor_bytes), _sha256(cursor_bytes)),
            ".dream_cursor": NewFileDescriptor(True, len(cursor_bytes), _sha256(cursor_bytes)),
        }
    )



def _receipt_tail_evidence(
    receipt: object,
    transaction_captured: object,
    transaction_captured_record: object,
    conversion_identity: str,
    source_bytes: bytes,
    fallback_timestamp: str,
) -> tuple[tuple[LegacyHistoryEntry, ...], Mapping[str, Any], str, Mapping[str, Any], Mapping[str, Any]]:
    if type(receipt) is not MemoryReceipt or type(receipt.raw_bytes) is not bytes:
        _fail()
    try:
        parsed = receipt_codec._validate_payload(
            receipt_codec._decode(receipt.raw_bytes), receipt.raw_bytes
        )
    except Exception:
        _fail()
    if parsed != receipt or type(transaction_captured) is not CapturedTransactionPayload:
        _fail()
    if transaction_captured.identity != conversion_identity:
        _fail()
    if (transaction_captured.receipt_digest != _sha256(parsed.raw_bytes)
            or transaction_captured.receipt_size != len(parsed.raw_bytes)):
        _fail()
    try:
        marker = parsed.history_marker_id.encode("utf-8")
        entry = parsed.history_entry.encode("utf-8")
    except UnicodeError:
        _fail()
    try:
        session_key = parsed.session_key.encode("utf-8")
        created_at = parsed.created_at.encode("utf-8")
        output_size = 0 if parsed.memory_output is None else len(parsed.memory_output.encode("utf-8"))
    except (AttributeError, UnicodeError):
        _fail()
    if (
        transaction_captured.history_marker_digest != _sha256(marker)
        or transaction_captured.history_entry_digest != _sha256(entry)
        or transaction_captured.created_at_digest != _sha256(created_at)
        or transaction_captured.session_key_digest != _sha256(session_key)
        or transaction_captured.chunk_digest != parsed.chunk_digest
        or transaction_captured.base_memory_digest != parsed.base_memory_digest
        or transaction_captured.output_memory_digest != parsed.output_memory_digest
        or transaction_captured.output_memory_size != output_size
        or transaction_captured.reference_only != parsed.reference_only
        or transaction_captured.start_offset != parsed.start_offset
        or transaction_captured.end_offset != parsed.end_offset
        or transaction_captured.session_generation != parsed.session_generation
    ):
        _fail()
    record = transaction_captured_record
    if (type(record) is not ManifestRecord or record.version != 1
            or record.kind != "transactions" or record.identity != conversion_identity
            or record.sequence != 0 or record.predecessor_digest is not None
            or record.phase != "CAPTURED" or type(record.raw_bytes) is not bytes
            or record.digest != _sha256(record.raw_bytes)):
        _fail()
    try:
        raw_record = json.loads(record.raw_bytes.decode("utf-8"), object_pairs_hook=_pairs_without_duplicates)
        loaded = decode_transaction_payload("CAPTURED", record.payload)
    except Exception:
        _fail()
    if (type(raw_record) is not dict or raw_record.get("payload") != transaction_captured.to_payload()
            or loaded != transaction_captured
            or raw_record.get("payload") != record.payload
            or record.raw_bytes != json.dumps(raw_record, ensure_ascii=False, sort_keys=True,
                                              separators=(",", ":"), allow_nan=False).encode("utf-8")
            or raw_record != {"version": 1, "kind": "transactions", "identity": conversion_identity,
                              "sequence": 0, "predecessor_digest": None, "phase": "CAPTURED",
                              "payload": transaction_captured.to_payload()}):
        _fail()
    try:
        source_bytes.decode("utf-8")
        status, _ = classify_history_marker_frame(source_bytes, parsed.history_marker_id, parsed.history_entry)
        if status != "present":
            _fail()
        frame = (b"<!-- nanobot-consolidation:" + marker + b" -->\n" + entry).rstrip() + b"\n\n"
        offset = source_bytes.rfind(frame)
        if offset < 0 or offset + len(frame) != len(source_bytes):
            _fail()
        # The marker must be an isolated, unique final frame. The prefix parser is untouched.
        if (source_bytes.count(b"<!-- nanobot-consolidation:" + marker) != 1
                or (offset != 0 and not source_bytes[:offset].endswith(b"\n\n"))):
            _fail()
        prefix_rows = parse_legacy_history(source_bytes[:offset], fallback_timestamp=fallback_timestamp)
        content = frame[:-2].decode("utf-8")
    except ConversionError:
        raise
    except Exception:
        _fail()
    row = LegacyHistoryEntry(len(prefix_rows) + 1, fallback_timestamp, content)
    try:
        canonical = (json.dumps({"cursor": row.cursor, "timestamp": row.timestamp,
                                 "content": row.content}, ensure_ascii=False) + "\n").encode("utf-8")
    except (UnicodeError, ValueError, TypeError):
        _fail()
    return (
        (*prefix_rows, row),
        MappingProxyType({"size": len(parsed.raw_bytes), "blob_digest": _sha256(parsed.raw_bytes)}),
        record.digest,
        MappingProxyType({"offset": offset, "size": len(frame), "digest": _sha256(frame)}),
        MappingProxyType({"cursor": row.cursor, "size": len(canonical), "blob_digest": _sha256(canonical)}),
    )


def _prepare(
    captured: CapturedConversionPayload,
    *,
    disposition: object,
    source: object,
    targets: object,
    backup: object,
    fallback_timestamp: object,
    source_bytes: object,
    target_contents: object,
    receipt: object,
    transaction_captured: object,
    transaction_captured_record: object,
) -> PreparedConversionPayload:
    captured = _validated_captured(captured)
    if type(disposition) is not str or disposition not in (
        "convert",
        "preserve_nonempty_jsonl",
    ):
        _fail()
    parsed_source = _old_file(source)
    parsed_targets = _target_map(targets)
    if not parsed_source.exists:
        _fail()

    if disposition == "preserve_nonempty_jsonl":
        if any(value is not None for value in (receipt, transaction_captured, transaction_captured_record)):
            _fail()
        if not parsed_targets["history.jsonl"].exists or parsed_targets["history.jsonl"].size == 0:
            _fail()
        if (
            backup is not None
            or fallback_timestamp is not None
            or source_bytes is not None
            or target_contents is not None
        ):
            _fail()
        prepared = PreparedConversionPayload(
            captured,
            disposition,
            parsed_source,
            parsed_targets,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    else:
        if parsed_targets["history.jsonl"].size != 0:
            _fail()
        if backup is None:
            _fail()
        parsed_backup = _backup_plan(backup)
        parsed_fallback = _fallback_timestamp(fallback_timestamp)
        raw_source = _verify_source_bytes(parsed_source, source_bytes)
        verified_contents = _verify_target_contents(parsed_targets, target_contents)
        if verified_contents["history.jsonl"] not in (None, b""):
            _fail()
        evidence = None
        if captured.schema_version == 2:
            rows, *evidence = _receipt_tail_evidence(
                receipt, transaction_captured, transaction_captured_record,
                captured.receipt_transaction_identity, raw_source, parsed_fallback,
            )
        else:
            if any(value is not None for value in (receipt, transaction_captured, transaction_captured_record)):
                _fail()
            try:
                rows = parse_legacy_history(
                    raw_source,
                    fallback_timestamp=parsed_fallback,
                    base_cursor=0,
                )
            except Exception:
                _fail()
        counters = _counter_contents(parsed_targets, target_contents)
        final_cursor = rows[-1].cursor if rows else 0
        old_cursor = counters[".cursor"]
        old_dream_cursor = counters[".dream_cursor"]
        if old_cursor > final_cursor or old_dream_cursor > final_cursor:
            _fail()
        sidecar_counters: Mapping[str, int] | None = MappingProxyType(
            {".cursor": old_cursor, ".dream_cursor": old_dream_cursor}
        )
        outputs = _preview_outputs(rows, parsed_targets)
        prepared = PreparedConversionPayload(
            captured,
            disposition,
            parsed_source,
            parsed_targets,
            parsed_backup,
            parsed_fallback,
            len(rows),
            final_cursor,
            sidecar_counters,
            outputs,
            *(evidence or ()),
        )
    payload = prepared.to_payload()
    try:
        derive_manifest_identity("conversions", payload)
        _check_record_bounds(payload, "CONVERSION_PREPARED", captured.identity)
    except (MemoryManifestError, ConversionError):
        _fail()
    return prepared


def prepare_conversion_payload(
    captured: CapturedConversionPayload,
    *,
    disposition: object,
    source: object,
    targets: object,
    source_bytes: object,
    target_contents: object = None,
    backup: object = None,
    fallback_timestamp: object = None,
    receipt: object = None,
    transaction_captured: object = None,
    transaction_captured_record: object = None,
) -> PreparedConversionPayload:
    """Build immutable PREPARED evidence from explicit captured claims.

    ``source_bytes`` is required for ``convert`` so the codec can verify the
    old source claim and preview exact JSONL/cursor bytes.  The preserve
    disposition records only old descriptors and does not consume file text.
    """

    try:
        return _prepare(
            captured,
            disposition=disposition,
            source=source,
            targets=targets,
            backup=backup,
            fallback_timestamp=fallback_timestamp,
            source_bytes=source_bytes,
            target_contents=target_contents,
            receipt=receipt,
            transaction_captured=transaction_captured,
            transaction_captured_record=transaction_captured_record,
        )
    except ConversionError:
        raise
    except Exception:
        _fail()


def _validated_captured(value: object) -> CapturedConversionPayload:
    if type(value) is not CapturedConversionPayload:
        _fail()
    payload = value.to_payload()
    parsed = _captured_from_payload(payload)
    try:
        identity = derive_manifest_identity("conversions", payload)
        _check_record_bounds(payload, "CAPTURED", identity)
    except Exception:
        _fail()
    if parsed != value:
        _fail()
    return parsed


def _captured_from_payload(value: object) -> CapturedConversionPayload:
    if type(value) is not dict or set(value) != _CAPTURED_FIELDS:
        _fail()
    version = value["schema_version"]
    if type(version) is not int or version not in (1, 2):
        _fail()
    if value["conversion_policy"] != (_CONVERSION_POLICY if version == 1 else _RECEIPT_TAIL_POLICY):
        _fail()
    return CapturedConversionPayload(_digest(value["receipt_transaction_identity"]), version)


def _prepared_from_payload(
    value: object,
    captured: CapturedConversionPayload,
) -> PreparedConversionPayload:
    fields = _PREPARED_FIELDS | (_V2_EVIDENCE_FIELDS if captured.schema_version == 2 and type(value) is dict and value.get("disposition") == "convert" else frozenset())
    if type(value) is not dict or set(value) != fields:
        _fail()
    if type(value["schema_version"]) is not int or value["schema_version"] != captured.schema_version:
        _fail()
    if value["conversion_policy"] != (_CONVERSION_POLICY if captured.schema_version == 1 else _RECEIPT_TAIL_POLICY):
        _fail()
    if _digest(value["conversion_identity"]) != captured.identity:
        _fail()

    disposition = value["disposition"]
    source = _old_file(value["source"])
    targets = _target_map(value["targets"])
    if disposition == "preserve_nonempty_jsonl":
        if not source.exists:
            _fail()
        if not targets["history.jsonl"].exists or targets["history.jsonl"].size == 0:
            _fail()
        if any(
            value[name] is not None
            for name in (
                "backup",
                "fallback_timestamp",
                "row_count",
                "final_legacy_cursor",
                "sidecar_counters",
                "outputs",
            )
        ):
            _fail()
        return PreparedConversionPayload(
            captured, disposition, source, targets, None, None, None, None, None, None
        )
    if disposition != "convert":
        _fail()
    if not source.exists or targets["history.jsonl"].size != 0:
        _fail()
    backup = _backup_plan(value["backup"])
    fallback = _fallback_timestamp(value["fallback_timestamp"])
    row_count = _counter(value["row_count"])
    final_cursor = _counter(value["final_legacy_cursor"])
    if final_cursor != row_count:
        _fail()
    outputs_value = value["outputs"]
    if type(outputs_value) is not dict or set(outputs_value) != set(_TARGET_NAMES):
        _fail()
    outputs = MappingProxyType({name: _new_file(outputs_value[name]) for name in _TARGET_NAMES})
    counters_value = value["sidecar_counters"]
    if type(counters_value) is not dict or set(counters_value) != {".cursor", ".dream_cursor"}:
        _fail()
    counters = {name: _counter(counters_value[name]) for name in (".cursor", ".dream_cursor")}
    if counters[".dream_cursor"] > counters[".cursor"]:
        _fail()
    for name in (".cursor", ".dream_cursor"):
        old = targets[name]
        if (not old.exists and counters[name] != 0) or (old.exists and old.size == 0):
            _fail()
    if counters[".cursor"] > final_cursor or counters[".dream_cursor"] > final_cursor:
        _fail()
    if row_count == 0:
        if counters[".cursor"] != 0 or counters[".dream_cursor"] != 0:
            _fail()
        for name in _TARGET_NAMES:
            old = targets[name]
            new = outputs[name]
            if (new.exists, new.size, new.blob_digest) != (
                old.exists,
                old.size,
                old.blob_digest,
            ):
                _fail()
    else:
        cursor_bytes = str(final_cursor).encode("utf-8")
        expected_cursor = NewFileDescriptor(True, len(cursor_bytes), _sha256(cursor_bytes))
        if outputs["history.jsonl"].exists is not True or outputs["history.jsonl"].size == 0:
            _fail()
        if outputs[".cursor"] != expected_cursor or outputs[".dream_cursor"] != expected_cursor:
            _fail()
    evidence: tuple[Mapping[str, Any], str, Mapping[str, Any], Mapping[str, Any]] | tuple[()] = ()
    if captured.schema_version == 2:
        receipt_value = value["receipt"]
        frame_value = value["tail_frame"]
        row_value = value["canonical_row"]
        if (type(receipt_value) is not dict or set(receipt_value) != {"size", "blob_digest"}
                or type(frame_value) is not dict or set(frame_value) != {"offset", "size", "digest"}
                or type(row_value) is not dict or set(row_value) != {"cursor", "size", "blob_digest"}):
            _fail()
        receipt_size = _counter(receipt_value["size"], positive=True)
        receipt_digest = _digest(receipt_value["blob_digest"])
        offset = _counter(frame_value["offset"])
        frame_size = _counter(frame_value["size"], positive=True)
        frame_digest = _digest(frame_value["digest"])
        row_cursor = _counter(row_value["cursor"], positive=True)
        row_size = _counter(row_value["size"], positive=True)
        row_digest = _digest(row_value["blob_digest"])
        if (offset + frame_size != source.size or row_cursor != final_cursor
                or row_size > outputs["history.jsonl"].size):
            _fail()
        evidence = (
            MappingProxyType({"size": receipt_size, "blob_digest": receipt_digest}),
            _digest(value["transaction_captured_record_digest"]),
            MappingProxyType({"offset": offset, "size": frame_size, "digest": frame_digest}),
            MappingProxyType({"cursor": row_cursor, "size": row_size, "blob_digest": row_digest}),
        )
    return PreparedConversionPayload(
        captured,
        disposition,
        source,
        targets,
        backup,
        fallback,
        row_count,
        final_cursor,
        MappingProxyType(counters),
        outputs,
        *evidence,
    )


def _validated_prepared(
    value: object,
    captured: CapturedConversionPayload,
) -> PreparedConversionPayload:
    if type(value) is not PreparedConversionPayload:
        _fail()
    if value.captured != captured:
        _fail()
    payload = value.to_payload()
    try:
        derive_manifest_identity("conversions", payload)
        _check_record_bounds(payload, "CONVERSION_PREPARED", captured.identity)
    except Exception:
        _fail()
    decoded = _prepared_from_payload(payload, captured)
    if decoded != value:
        _fail()
    return decoded


def _conversion_context(
    captured: object,
    prepared: object,
) -> tuple[CapturedConversionPayload, PreparedConversionPayload]:
    valid_captured = _validated_captured(captured)
    valid_prepared = _validated_prepared(prepared, valid_captured)
    return valid_captured, valid_prepared


def _old_projection(value: OldFileDescriptor) -> NewFileDescriptor:
    return NewFileDescriptor(value.exists, value.size, value.blob_digest)


def _prepared_target_projection(
    prepared: PreparedConversionPayload,
) -> Mapping[str, NewFileDescriptor]:
    if prepared.disposition == "convert":
        if prepared.outputs is None:
            _fail()
        return MappingProxyType({name: prepared.outputs[name] for name in _TARGET_NAMES})
    if prepared.disposition == "preserve_nonempty_jsonl":
        if prepared.outputs is not None:
            _fail()
        return MappingProxyType(
            {name: _old_projection(prepared.targets[name]) for name in _TARGET_NAMES}
        )
    _fail()


def _finalized_projection(
    prepared: PreparedConversionPayload,
    cursors_applied: CursorsAppliedConversionPayload,
) -> tuple[
    Mapping[str, NewFileDescriptor],
    NewFileDescriptor,
    str | None,
    NewFileDescriptor | None,
]:
    targets = MappingProxyType(
        {
            "history.jsonl": cursors_applied.history_target,
            ".cursor": cursors_applied.cursor_target,
            ".dream_cursor": cursors_applied.dream_cursor_target,
        }
    )
    if prepared.disposition == "convert":
        if prepared.backup is None:
            _fail()
        source = NewFileDescriptor(False, 0, None)
        backup_basename: str | None = prepared.backup.basename
        backup = NewFileDescriptor(
            True,
            prepared.source.size,
            prepared.source.blob_digest,
        )
        return targets, source, backup_basename, backup
    if prepared.disposition == "preserve_nonempty_jsonl":
        return targets, _old_projection(prepared.source), None, None
    _fail()



def _phase_payload(
    value: object,
    fields: frozenset[str],
    captured: CapturedConversionPayload,
    prepared: PreparedConversionPayload,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        _fail()
    if type(value["schema_version"]) is not int or value["schema_version"] != captured.schema_version:
        _fail()
    if _digest(value["conversion_identity"]) != captured.identity:
        _fail()
    if type(value["disposition"]) is not str or value["disposition"] != prepared.disposition:
        _fail()
    return value


def _new_target_map(value: object) -> Mapping[str, NewFileDescriptor]:
    if type(value) is not dict or set(value) != set(_TARGET_NAMES):
        _fail()
    return MappingProxyType({name: _new_file(value[name]) for name in _TARGET_NAMES})


def _jsonl_applied_from_payload(
    value: object,
    captured: CapturedConversionPayload,
    prepared: PreparedConversionPayload,
) -> JsonlAppliedConversionPayload:
    captured, prepared = _conversion_context(captured, prepared)
    raw = _phase_payload(value, _JSONL_APPLIED_FIELDS, captured, prepared)
    history_target = _new_file(raw["history_target"])
    expected = _prepared_target_projection(prepared)["history.jsonl"]
    if history_target != expected:
        _fail()
    return JsonlAppliedConversionPayload(captured.identity, prepared.disposition, history_target, captured.schema_version)


def _validated_jsonl_applied(
    value: object,
    captured: CapturedConversionPayload,
    prepared: PreparedConversionPayload,
) -> JsonlAppliedConversionPayload:
    if type(value) is not JsonlAppliedConversionPayload:
        _fail()
    decoded = _jsonl_applied_from_payload(value.to_payload(), captured, prepared)
    if decoded != value:
        _fail()
    return decoded


def _cursors_applied_from_payload(
    value: object,
    captured: CapturedConversionPayload,
    prepared: PreparedConversionPayload,
    jsonl_applied: object,
) -> CursorsAppliedConversionPayload:
    captured, prepared = _conversion_context(captured, prepared)
    jsonl = _validated_jsonl_applied(jsonl_applied, captured, prepared)
    raw = _phase_payload(value, _CURSORS_APPLIED_FIELDS, captured, prepared)
    history_target = _new_file(raw["history_target"])
    cursor_target = _new_file(raw["cursor_target"])
    dream_cursor_target = _new_file(raw["dream_cursor_target"])
    expected = _prepared_target_projection(prepared)
    if (
        history_target != jsonl.history_target
        or cursor_target != expected[".cursor"]
        or dream_cursor_target != expected[".dream_cursor"]
    ):
        _fail()
    return CursorsAppliedConversionPayload(
        captured.identity,
        prepared.disposition,
        history_target,
        cursor_target,
        dream_cursor_target,
        captured.schema_version,
    )


def _validated_cursors_applied(
    value: object,
    captured: CapturedConversionPayload,
    prepared: PreparedConversionPayload,
    jsonl_applied: object,
) -> CursorsAppliedConversionPayload:
    if type(value) is not CursorsAppliedConversionPayload:
        _fail()
    decoded = _cursors_applied_from_payload(value.to_payload(), captured, prepared, jsonl_applied)
    if decoded != value:
        _fail()
    return decoded


def _phase_record_digests(
    prepared_record_digest: object,
    jsonl_applied_record_digest: object,
    cursors_applied_record_digest: object,
) -> tuple[str, str, str]:
    return (
        _digest(prepared_record_digest),
        _digest(jsonl_applied_record_digest),
        _digest(cursors_applied_record_digest),
    )


def _legacy_finalized_from_payload(
    value: object,
    captured: CapturedConversionPayload,
    prepared: PreparedConversionPayload,
    jsonl_applied: object,
    cursors_applied: object,
    *,
    prepared_record_digest: object,
    jsonl_applied_record_digest: object,
    cursors_applied_record_digest: object,
) -> LegacyFinalizedConversionPayload:
    captured, prepared = _conversion_context(captured, prepared)
    _validated_jsonl_applied(jsonl_applied, captured, prepared)
    cursors = _validated_cursors_applied(
        cursors_applied, captured, prepared, jsonl_applied
    )
    expected_digests = _phase_record_digests(
        prepared_record_digest,
        jsonl_applied_record_digest,
        cursors_applied_record_digest,
    )
    raw = _phase_payload(value, _LEGACY_FINALIZED_FIELDS, captured, prepared)
    actual_digests = _phase_record_digests(
        raw["prepared_record_digest"],
        raw["jsonl_applied_record_digest"],
        raw["cursors_applied_record_digest"],
    )
    if actual_digests != expected_digests:
        _fail()
    targets = _new_target_map(raw["targets"])
    source = _new_file(raw["source"])
    backup_basename = raw["backup_basename"]
    backup_value = raw["backup"]
    backup = None if backup_value is None else _new_file(backup_value)
    expected_targets, expected_source, expected_basename, expected_backup = _finalized_projection(
        prepared, cursors
    )
    if (
        targets != expected_targets
        or source != expected_source
        or backup_basename != expected_basename
        or backup != expected_backup
    ):
        _fail()
    return LegacyFinalizedConversionPayload(
        captured.identity,
        prepared.disposition,
        *actual_digests,
        targets,
        source,
        backup_basename,
        backup,
        captured.schema_version,
    )


def _validated_legacy_finalized(
    value: object,
    captured: CapturedConversionPayload,
    prepared: PreparedConversionPayload,
    jsonl_applied: object,
    cursors_applied: object,
    *,
    prepared_record_digest: object,
    jsonl_applied_record_digest: object,
    cursors_applied_record_digest: object,
) -> LegacyFinalizedConversionPayload:
    if type(value) is not LegacyFinalizedConversionPayload:
        _fail()
    decoded = _legacy_finalized_from_payload(
        value.to_payload(),
        captured,
        prepared,
        jsonl_applied,
        cursors_applied,
        prepared_record_digest=prepared_record_digest,
        jsonl_applied_record_digest=jsonl_applied_record_digest,
        cursors_applied_record_digest=cursors_applied_record_digest,
    )
    if decoded != value:
        _fail()
    return decoded


def _complete_from_payload(
    value: object,
    captured: CapturedConversionPayload,
    prepared: PreparedConversionPayload,
    jsonl_applied: object,
    cursors_applied: object,
    legacy_finalized: object,
    *,
    finalized_record_digest: object,
) -> CompleteConversionPayload:
    captured, prepared = _conversion_context(captured, prepared)
    if type(legacy_finalized) is not LegacyFinalizedConversionPayload:
        _fail()
    if type(legacy_finalized) is not LegacyFinalizedConversionPayload:
        _fail()
    finalized = _validated_legacy_finalized(
        legacy_finalized,
        captured,
        prepared,
        jsonl_applied,
        cursors_applied,
        prepared_record_digest=legacy_finalized.prepared_record_digest,
        jsonl_applied_record_digest=legacy_finalized.jsonl_applied_record_digest,
        cursors_applied_record_digest=legacy_finalized.cursors_applied_record_digest,
    )
    expected_finalized_digest = _digest(finalized_record_digest)
    raw = _phase_payload(value, _COMPLETE_FIELDS, captured, prepared)
    if _digest(raw["finalized_record_digest"]) != expected_finalized_digest:
        _fail()
    targets = _new_target_map(raw["targets"])
    source = _new_file(raw["source"])
    backup_basename = raw["backup_basename"]
    backup_value = raw["backup"]
    backup = None if backup_value is None else _new_file(backup_value)
    if (
        targets != finalized.targets
        or source != finalized.source
        or backup_basename != finalized.backup_basename
        or backup != finalized.backup
    ):
        _fail()
    return CompleteConversionPayload(
        captured.identity,
        prepared.disposition,
        expected_finalized_digest,
        targets,
        source,
        backup_basename,
        backup,
        captured.schema_version,
    )


def _validated_complete(
    value: object,
    captured: CapturedConversionPayload,
    prepared: PreparedConversionPayload,
    jsonl_applied: object,
    cursors_applied: object,
    legacy_finalized: object,
    *,
    finalized_record_digest: object,
) -> CompleteConversionPayload:
    if type(value) is not CompleteConversionPayload:
        _fail()
    decoded = _complete_from_payload(
        value.to_payload(),
        captured,
        prepared,
        jsonl_applied,
        cursors_applied,
        legacy_finalized,
        finalized_record_digest=finalized_record_digest,
    )
    if decoded != value:
        _fail()
    return decoded


def jsonl_applied_conversion_payload(
    captured: CapturedConversionPayload,
    prepared: PreparedConversionPayload,
) -> JsonlAppliedConversionPayload:
    """Build JSONL_APPLIED from the prepared target projection."""

    captured, prepared = _conversion_context(captured, prepared)
    payload = JsonlAppliedConversionPayload(
        captured.identity,
        prepared.disposition,
        _prepared_target_projection(prepared)["history.jsonl"],
        captured.schema_version,
    ).to_payload()
    _validate_built_payload("JSONL_APPLIED", payload, captured)
    return _jsonl_applied_from_payload(payload, captured, prepared)


def cursors_applied_conversion_payload(
    captured: CapturedConversionPayload,
    prepared: PreparedConversionPayload,
    jsonl_applied: JsonlAppliedConversionPayload,
) -> CursorsAppliedConversionPayload:
    """Build CURSORS_APPLIED after typed JSONL evidence."""

    captured, prepared = _conversion_context(captured, prepared)
    jsonl = _validated_jsonl_applied(jsonl_applied, captured, prepared)
    expected = _prepared_target_projection(prepared)
    payload = CursorsAppliedConversionPayload(
        captured.identity,
        prepared.disposition,
        jsonl.history_target,
        expected[".cursor"],
        expected[".dream_cursor"],
        captured.schema_version,
    ).to_payload()
    _validate_built_payload("CURSORS_APPLIED", payload, captured)
    return _cursors_applied_from_payload(payload, captured, prepared, jsonl)


def legacy_finalized_conversion_payload(
    captured: CapturedConversionPayload,
    prepared: PreparedConversionPayload,
    jsonl_applied: JsonlAppliedConversionPayload,
    cursors_applied: CursorsAppliedConversionPayload,
    *,
    prepared_record_digest: str,
    jsonl_applied_record_digest: str,
    cursors_applied_record_digest: str,
) -> LegacyFinalizedConversionPayload:
    """Build LEGACY_FINALIZED from typed target evidence and manifest digests."""

    captured, prepared = _conversion_context(captured, prepared)
    jsonl = _validated_jsonl_applied(jsonl_applied, captured, prepared)
    cursors = _validated_cursors_applied(cursors_applied, captured, prepared, jsonl)
    digests = _phase_record_digests(
        prepared_record_digest,
        jsonl_applied_record_digest,
        cursors_applied_record_digest,
    )
    targets, source, backup_basename, backup = _finalized_projection(prepared, cursors)
    payload = LegacyFinalizedConversionPayload(
        captured.identity,
        prepared.disposition,
        *digests,
        targets,
        source,
        backup_basename,
        backup,
        captured.schema_version,
    ).to_payload()
    _validate_built_payload("LEGACY_FINALIZED", payload, captured)
    return _legacy_finalized_from_payload(
        payload,
        captured,
        prepared,
        jsonl,
        cursors,
        prepared_record_digest=prepared_record_digest,
        jsonl_applied_record_digest=jsonl_applied_record_digest,
        cursors_applied_record_digest=cursors_applied_record_digest,
    )


def complete_conversion_payload(
    captured: CapturedConversionPayload,
    prepared: PreparedConversionPayload,
    jsonl_applied: JsonlAppliedConversionPayload,
    cursors_applied: CursorsAppliedConversionPayload,
    legacy_finalized: LegacyFinalizedConversionPayload,
    *,
    finalized_record_digest: str,
) -> CompleteConversionPayload:
    """Build COMPLETE from typed finalized evidence and its manifest digest."""

    captured, prepared = _conversion_context(captured, prepared)
    jsonl = _validated_jsonl_applied(jsonl_applied, captured, prepared)
    cursors = _validated_cursors_applied(cursors_applied, captured, prepared, jsonl)
    finalized = _validated_legacy_finalized(
        legacy_finalized,
        captured,
        prepared,
        jsonl,
        cursors,
        prepared_record_digest=legacy_finalized.prepared_record_digest,
        jsonl_applied_record_digest=legacy_finalized.jsonl_applied_record_digest,
        cursors_applied_record_digest=legacy_finalized.cursors_applied_record_digest,
    )
    digest = _digest(finalized_record_digest)
    payload = CompleteConversionPayload(
        captured.identity,
        prepared.disposition,
        digest,
        finalized.targets,
        finalized.source,
        finalized.backup_basename,
        finalized.backup,
        captured.schema_version,
    ).to_payload()
    _validate_built_payload("COMPLETE", payload, captured)
    return _complete_from_payload(
        payload,
        captured,
        prepared,
        jsonl,
        cursors,
        finalized,
        finalized_record_digest=finalized_record_digest,
    )


class _DuplicateKeyError(ValueError):
    pass


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise _DuplicateKeyError
        parsed[key] = value
    return parsed


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _parse_payload(value: object) -> dict[str, Any]:
    if type(value) is bytes:
        if len(value) > MAX_RECORD_BYTES:
            _fail()
        try:
            parsed = json.loads(
                value.decode("utf-8"),
                object_pairs_hook=_pairs_without_duplicates,
                parse_constant=_reject_non_json_constant,
            )
        except Exception:
            _fail()
        if type(parsed) is not dict:
            _fail()
        return parsed
    if type(value) is not dict:
        _fail()
    return value


def decode_conversion_payload(
    payload: object,
    phase: object,
    *,
    captured: CapturedConversionPayload | None = None,
    prepared: PreparedConversionPayload | None = None,
    jsonl_applied: JsonlAppliedConversionPayload | None = None,
    cursors_applied: CursorsAppliedConversionPayload | None = None,
    legacy_finalized: LegacyFinalizedConversionPayload | None = None,
    prepared_record_digest: str | None = None,
    jsonl_applied_record_digest: str | None = None,
    cursors_applied_record_digest: str | None = None,
    finalized_record_digest: str | None = None,
) -> (
    CapturedConversionPayload
    | PreparedConversionPayload
    | JsonlAppliedConversionPayload
    | CursorsAppliedConversionPayload
    | LegacyFinalizedConversionPayload
    | CompleteConversionPayload
):
    """Strictly decode a conversion phase and its typed predecessor evidence."""

    try:
        if type(phase) is not str:
            _fail()
        raw = _parse_payload(payload)
        derive_manifest_identity("conversions", raw)
        if phase == "CAPTURED":
            if any(
                item is not None
                for item in (
                    captured,
                    prepared,
                    jsonl_applied,
                    cursors_applied,
                    legacy_finalized,
                    prepared_record_digest,
                    jsonl_applied_record_digest,
                    cursors_applied_record_digest,
                    finalized_record_digest,
                )
            ):
                _fail()
            decoded = _captured_from_payload(raw)
            _check_record_bounds(raw, phase, decoded.identity)
            return decoded
        if phase == "CONVERSION_PREPARED":
            if captured is None or any(
                item is not None
                for item in (
                    prepared,
                    jsonl_applied,
                    cursors_applied,
                    legacy_finalized,
                    prepared_record_digest,
                    jsonl_applied_record_digest,
                    cursors_applied_record_digest,
                    finalized_record_digest,
                )
            ):
                _fail()
            valid_captured = _validated_captured(captured)
            decoded_prepared = _prepared_from_payload(raw, valid_captured)
            _check_record_bounds(raw, phase, valid_captured.identity)
            return decoded_prepared

        if captured is None or prepared is None:
            _fail()
        valid_captured, valid_prepared = _conversion_context(captured, prepared)
        if phase == "JSONL_APPLIED":
            if any(
                item is not None
                for item in (
                    jsonl_applied,
                    cursors_applied,
                    legacy_finalized,
                    prepared_record_digest,
                    jsonl_applied_record_digest,
                    cursors_applied_record_digest,
                    finalized_record_digest,
                )
            ):
                _fail()
            decoded_jsonl = _jsonl_applied_from_payload(raw, valid_captured, valid_prepared)
            _check_record_bounds(raw, phase, valid_captured.identity)
            return decoded_jsonl
        if phase == "CURSORS_APPLIED":
            if jsonl_applied is None or any(
                item is not None
                for item in (
                    cursors_applied,
                    legacy_finalized,
                    prepared_record_digest,
                    jsonl_applied_record_digest,
                    cursors_applied_record_digest,
                    finalized_record_digest,
                )
            ):
                _fail()
            decoded_cursors = _cursors_applied_from_payload(
                raw, valid_captured, valid_prepared, jsonl_applied
            )
            _check_record_bounds(raw, phase, valid_captured.identity)
            return decoded_cursors
        if phase == "LEGACY_FINALIZED":
            if (
                jsonl_applied is None
                or cursors_applied is None
                or any(
                    item is not None
                    for item in (legacy_finalized, finalized_record_digest)
                )
                or prepared_record_digest is None
                or jsonl_applied_record_digest is None
                or cursors_applied_record_digest is None
            ):
                _fail()
            decoded_finalized = _legacy_finalized_from_payload(
                raw,
                valid_captured,
                valid_prepared,
                jsonl_applied,
                cursors_applied,
                prepared_record_digest=prepared_record_digest,
                jsonl_applied_record_digest=jsonl_applied_record_digest,
                cursors_applied_record_digest=cursors_applied_record_digest,
            )
            _check_record_bounds(raw, phase, valid_captured.identity)
            return decoded_finalized
        if phase == "COMPLETE":
            if (
                jsonl_applied is None
                or cursors_applied is None
                or legacy_finalized is None
                or finalized_record_digest is None
                or any(
                    item is not None
                    for item in (
                        prepared_record_digest,
                        jsonl_applied_record_digest,
                        cursors_applied_record_digest,
                    )
                )
            ):
                _fail()
            decoded_complete = _complete_from_payload(
                raw,
                valid_captured,
                valid_prepared,
                jsonl_applied,
                cursors_applied,
                legacy_finalized,
                finalized_record_digest=finalized_record_digest,
            )
            _check_record_bounds(raw, phase, valid_captured.identity)
            return decoded_complete
        _fail()
    except ConversionError:
        raise
    except (MemoryManifestError, TypeError, ValueError, UnicodeError, OverflowError):
        _fail()
    except Exception:
        _fail()
