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

from nanobot.agent.legacy_history_codec import (
    MAX_LEGACY_HISTORY_BYTES,
    LegacyHistoryEntry,
    parse_legacy_history,
)
from nanobot.agent.memory_recovery_manifest import (
    MAX_INT,
    MAX_NODES,
    MAX_RECORD_BYTES,
    MAX_STRING_BYTES,
    MemoryManifestError,
    derive_manifest_identity,
)

_SCHEMA_VERSION = 1
_CONVERSION_POLICY = "legacy_history_v1"
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
_FILE_FIELDS = frozenset(("exists", "size", "digest", "stat", "blob_digest"))
_STAT_COUNTER_FIELDS = frozenset(("device", "inode", "mode", "uid", "gid", "links", "size"))
_NEW_FILE_FIELDS = frozenset(("exists", "size", "blob_digest"))
_BACKUP_FIELDS = frozenset(("basename", "candidates"))
_BACKUP_CANDIDATE_FIELDS = frozenset(("basename", "exists"))
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_FALLBACK_TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\Z")
_BACKUP_RE = re.compile(r"HISTORY\.md\.bak(?:\.([2-9][0-9]*))?\Z")
_CONVERSION_PHASE_SEQUENCE = {"CAPTURED": 0, "CONVERSION_PREPARED": 1}


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

    @property
    def identity(self) -> str:
        return derive_manifest_identity("conversions", self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "conversion_policy": _CONVERSION_POLICY,
            "receipt_transaction_identity": self.receipt_transaction_identity,
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

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "conversion_policy": _CONVERSION_POLICY,
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
    if type(value["schema_version"]) is not int or value["schema_version"] != _SCHEMA_VERSION:
        _fail()
    if value["conversion_policy"] != _CONVERSION_POLICY:
        _fail()
    return CapturedConversionPayload(_digest(value["receipt_transaction_identity"]))


def _prepared_from_payload(
    value: object,
    captured: CapturedConversionPayload,
) -> PreparedConversionPayload:
    if type(value) is not dict or set(value) != _PREPARED_FIELDS:
        _fail()
    if type(value["schema_version"]) is not int or value["schema_version"] != _SCHEMA_VERSION:
        _fail()
    if value["conversion_policy"] != _CONVERSION_POLICY:
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
) -> CapturedConversionPayload | PreparedConversionPayload:
    """Strictly decode CAPTURED or CONVERSION_PREPARED evidence without I/O."""

    try:
        if type(phase) is not str:
            _fail()
        raw = _parse_payload(payload)
        derive_manifest_identity("conversions", raw)
        if phase == "CAPTURED":
            if captured is not None:
                _fail()
            decoded = _captured_from_payload(raw)
            _check_record_bounds(raw, "CAPTURED", decoded.identity)
            return decoded
        if phase == "CONVERSION_PREPARED":
            if captured is None:
                _fail()
            valid_captured = _validated_captured(captured)
            decoded_prepared = _prepared_from_payload(raw, valid_captured)
            _check_record_bounds(raw, "CONVERSION_PREPARED", valid_captured.identity)
            return decoded_prepared
        _fail()
    except ConversionError:
        raise
    except (MemoryManifestError, TypeError, ValueError, UnicodeError, OverflowError):
        _fail()
    except Exception:
        _fail()
