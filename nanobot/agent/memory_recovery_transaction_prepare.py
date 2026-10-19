"""Build immutable PREPARED proposals for one captured memory receipt.

Preparation verifies the completed v2 legacy-history conversion, reads the
canonical session through the checked-session authority, and publishes only
content-addressed proposal blobs. It never replaces live memory or session
state and never changes the receipt.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, cast

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_conversion as conversion
from nanobot.agent import memory_recovery_conversion_finalize as conversion_finalize
from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent.memory_recovery_manifest import MAX_INT, ManifestRecord, MemoryRecoveryManifest
from nanobot.session.checked_source import open_checked_recovery_session
from nanobot.session.legacy_receipt_source import capture_legacy_receipt_chunk_digest
from nanobot.session.recovery_locator import locate_recovery_candidate
from nanobot.session.source import (
    canonical_events,
    validate_event_ids,
    validate_execution_links,
)

_legacy_chunk_digest = capture_legacy_receipt_chunk_digest()

_RECEIPT = ".consolidation.pending.json"
_TARGETS = ("history.jsonl", ".cursor", "MEMORY.md")
_CONVERSION_PHASES = (
    "CAPTURED",
    "CONVERSION_PREPARED",
    "JSONL_APPLIED",
    "CURSORS_APPLIED",
    "LEGACY_FINALIZED",
    "COMPLETE",
)


class MemoryRecoveryTransactionPreparationError(RuntimeError):
    """Fixed, content-free failure while producing a PREPARED proposal."""

    def __init__(self) -> None:
        super().__init__("memory recovery transaction preparation is unavailable")


def _fail() -> NoReturn:
    raise MemoryRecoveryTransactionPreparationError() from None


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _counter(value: object) -> int:
    if type(value) is not int or not 0 <= value <= MAX_INT:
        _fail()
    return value


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(value)


def _decode_json(raw: bytes) -> Any:
    return json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_constant,
    )


def _receipt_stat(snapshot: storage.MemoryTargetSnapshot) -> dict[str, int | None]:
    info = snapshot.stat_identity
    if snapshot.name != _RECEIPT or snapshot.content is None or info is None:
        _fail()
    return {
        "device": info.device,
        "inode": info.inode,
        "mode": info.mode,
        "uid": info.uid,
        "gid": info.gid,
        "links": info.links,
        "size": info.size,
        "mtime_ns": info.mtime_ns,
        "ctime_ns": info.ctime_ns,
    }


def receipt_snapshot_stat(snapshot: storage.MemoryTargetSnapshot) -> dict[str, int | None]:
    """Project an existing receipt snapshot's stat; acquire no new authority."""
    return _receipt_stat(snapshot)


def require_transaction_receipt_binding(
    authority: storage.WorkspaceMemoryAuthority,
    issued: storage.MemoryTargetSnapshot,
) -> None:
    """Check the direct transaction receipt's bytes and complete stat binding."""

    _receipt_bound(authority, issued)


def _receipt_bound(
    authority: storage.WorkspaceMemoryAuthority,
    issued: storage.MemoryTargetSnapshot,
) -> None:
    current = authority.snapshot_target(_RECEIPT)
    if (
        current.content != issued.content
        or current.stat_identity != issued.stat_identity
        or _receipt_stat(current) != _receipt_stat(issued)
    ):
        _fail()


def _manifest(
    authority: storage.WorkspaceMemoryAuthority,
    kind: str,
    identity: str,
) -> MemoryRecoveryManifest:
    return MemoryRecoveryManifest(authority.open_evidence_namespace(kind, identity))


def _conversion_evidence(
    authority: storage.WorkspaceMemoryAuthority,
    captured: transaction.CapturedTransactionPayload,
    tx_record: ManifestRecord,
    receipt_snapshot: storage.MemoryTargetSnapshot | quarantine.ValidatedQuarantineCopy,
    receipt: receipt_codec.MemoryReceipt,
    operation: storage.WorkspaceMemoryOperation,
    *, read_only: bool = False,
) -> conversion.PreparedConversionPayload:
    v1 = conversion.capture_conversion_payload(captured.identity)
    if read_only and not (
        authority.workspace / "memory" / ".recovery" / "conversions" / v1.identity
    ).is_dir():
        _fail()
    if _manifest(authority, "conversions", v1.identity).load():
        _fail()

    v2 = conversion.capture_receipt_tail_conversion_payload(captured.identity)
    if read_only and not (
        authority.workspace / "memory" / ".recovery" / "conversions" / v2.identity
    ).is_dir():
        _fail()
    records = _manifest(authority, "conversions", v2.identity).load()
    if len(records) != len(_CONVERSION_PHASES) or tuple(
        item.phase for item in records
    ) != _CONVERSION_PHASES:
        _fail()

    decoded = conversion_finalize.decode_conversion_finalization_evidence(records, v2)
    prepared = decoded[0]
    if (
        prepared.disposition != "convert"
        or prepared.captured != v2
        or prepared.receipt is None
        or prepared.transaction_captured_record_digest != tx_record.digest
        or prepared.receipt.get("size") != captured.receipt_size
        or prepared.receipt.get("blob_digest") != captured.receipt_digest
    ):
        _fail()
    conversion_finalize.require_conversion_finalization_binding(
        authority,
        operation,
        receipt_snapshot,
        receipt,
        tx_record,
        captured,
        v2,
        prepared,
        finalized=True,
        read_only=read_only,
    )
    return prepared


def _parse_history_rows(raw: bytes) -> list[dict[str, Any]]:
    if not raw or not raw.endswith(b"\n"):
        _fail()
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line:
            _fail()
        record = _decode_json(line)
        if type(record) is not dict:
            _fail()
        cursor = cast(dict[object, object], record).get("cursor")
        timestamp = cast(dict[object, object], record).get("timestamp")
        content = cast(dict[object, object], record).get("content")
        if type(cursor) is not int or cursor < 0 or type(timestamp) is not str or type(content) is not str:
            _fail()
        # _decode_json/_strict_pairs constructs only ordinary JSON string keys.
        rows.append(cast(dict[str, object], record))
    if not rows:
        _fail()
    return rows


def _verified_marker_row(
    authority: storage.WorkspaceMemoryAuthority,
    prepared: conversion.PreparedConversionPayload,
    receipt: receipt_codec.MemoryReceipt,
    history_bytes: bytes,
    cursor_bytes: bytes,
) -> bytes:
    row_claim = prepared.canonical_row
    if (
        row_claim is None
        or prepared.fallback_timestamp is None
        or prepared.final_legacy_cursor is None
        or row_claim.get("cursor") != prepared.final_legacy_cursor
    ):
        _fail()
    row = authority.read_blob(row_claim["blob_digest"])
    expected_content = (
        f"<!-- nanobot-consolidation:{receipt.history_marker_id} -->\n"
        f"{receipt.history_entry}"
    ).rstrip()
    expected_row = (
        json.dumps(
            {
                "cursor": row_claim["cursor"],
                "timestamp": prepared.fallback_timestamp,
                "content": expected_content,
            },
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    marker_frame = f"<!-- nanobot-consolidation:{receipt.history_marker_id} -->".encode("utf-8")
    if (
        row != expected_row
        or _hash(row) != row_claim["blob_digest"]
        or len(row) != row_claim["size"]
        or history_bytes.count(marker_frame) != 1
        or not history_bytes.endswith(row)
        or cursor_bytes != str(row_claim["cursor"]).encode("ascii")
    ):
        _fail()
    rows = _parse_history_rows(history_bytes)
    if rows[-1] != _decode_json(row[:-1]):
        _fail()
    return row


def _session_records(raw: bytes, session_key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    text = raw.decode("utf-8", errors="strict")
    records = [
        _decode_json(line.encode("utf-8"))
        for line in text.splitlines()
        if line.strip()
    ]
    if not records or type(records[0]) is not dict:
        _fail()
    # Exact-dict checks plus _strict_pairs establish JSON string keys, not a
    # primitive field schema; those fields retain their ordered checks below.
    header = cast(dict[str, object], records[0])
    if (
        header.get("_type") != "metadata"
        or header.get("key") != session_key
        or type(header.get("metadata")) is not dict
    ):
        _fail()
    metadata = cast(dict[str, object], header["metadata"])
    if (
        type(metadata.get("source_schema_version")) is not int
        or metadata["source_schema_version"] != 1
        or "runtime_checkpoint" in metadata
        or not {
            "revision", "generation", "incarnation", "last_archived", "last_consolidated",
        }.issubset(header)
        or header["last_archived"] != header["last_consolidated"]
        or type(header.get("incarnation")) is not str
        or not header["incarnation"]
    ):
        _fail()
    for field in ("created_at", "updated_at"):
        if type(header.get(field)) is not str:
            _fail()
        datetime.fromisoformat(cast(str, header[field]))
    events: list[dict[str, Any]] = []
    for event in records[1:]:
        if (
            type(event) is not dict
            or "_type" in event
            or type(cast(dict[object, object], event).get("role")) is not str
            or event["role"] not in {"user", "assistant", "tool", "system", "developer"}
        ):
            _fail()
        events.append(cast(dict[str, object], event))
    canonical_events(records)
    validate_event_ids(events)
    validate_execution_links(events)
    return header, events


def _source_proposal(
    raw: bytes,
    receipt: receipt_codec.MemoryReceipt,
) -> tuple[dict[str, int | str], bytes]:
    header, events = _session_records(raw, receipt.session_key or "")
    revision = _counter(header["revision"])
    generation = _counter(header["generation"])
    old_cursor = _counter(header["last_archived"])
    start_offset = _counter(receipt.start_offset)
    end_offset = _counter(receipt.end_offset)
    # The legacy patcher rejects converted sessions; keep its receipt digest contract here.
    if (
        generation != receipt.session_generation
        or old_cursor != start_offset
        or not start_offset < end_offset <= len(events)
        or revision >= MAX_INT
        or _legacy_chunk_digest(events[start_offset:end_offset]) != receipt.chunk_digest
    ):
        _fail()
    old_digest = _hash(raw)
    proposed_header = deepcopy(header)
    proposed_header["last_archived"] = end_offset
    proposed_header["last_consolidated"] = end_offset
    proposed_header["revision"] = revision + 1
    proposed_records = [proposed_header, *events]
    canonical_events(proposed_records)
    proposed = b"".join(
        (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        for record in proposed_records
    )
    if len(proposed) > 64 * 1024 * 1024:
        _fail()
    return (
        {
            "old_digest": old_digest,
            "old_size": len(raw),
            "old_generation": generation,
            "old_revision": revision,
            "old_cursor": old_cursor,
            "new_digest": _hash(proposed),
            "new_size": len(proposed),
            "new_generation": generation,
            "new_revision": revision + 1,
            "new_cursor": end_offset,
            "action": "advance",
        },
        proposed,
    )


def _target_descriptor(
    snapshot: storage.MemoryTargetSnapshot,
    new_content: bytes | None,
) -> dict[str, object]:
    content = snapshot.content
    info = snapshot.stat_identity
    if content is None:
        if info is not None:
            _fail()
        old: dict[str, object] = {"exists": False, "size": 0, "digest": None, "stat": None}
    else:
        if info is None:
            _fail()
        digest = _hash(content)
        old = {
            "exists": True,
            "size": len(content),
            "digest": digest,
            "stat": {
                "device": info.device,
                "inode": info.inode,
                "mode": info.mode,
                "uid": info.uid,
                "gid": info.gid,
                "links": info.links,
                "size": info.size,
                "mtime_ns": info.mtime_ns,
                "ctime_ns": info.ctime_ns,
            },
        }
    if new_content is None:
        new: dict[str, object] = {"exists": False, "size": 0, "blob_digest": None}
    else:
        new = {
            "exists": True,
            "size": len(new_content),
            "blob_digest": _hash(new_content),
        }
    return {"old": old, "new": new}


def _same_snapshot(
    authority: storage.WorkspaceMemoryAuthority,
    snapshot: storage.MemoryTargetSnapshot,
) -> None:
    current = authority.snapshot_target(snapshot.name)
    if current.content != snapshot.content or current.stat_identity != snapshot.stat_identity:
        _fail()


def _record_proposal_blobs(
    authority: storage.WorkspaceMemoryAuthority,
    targets: dict[str, dict[str, object]],
    source_bytes: bytes,
    target_bytes: dict[str, bytes | None],
) -> None:
    contents = [content for content in target_bytes.values() if content is not None]
    contents.append(source_bytes)
    for content in contents:
        digest = authority.publish_blob(content)
        if digest != _hash(content) or authority.read_blob(digest) != content:
            _fail()
    for name in _TARGETS:
        new = targets[name]["new"]
        if type(new) is not dict:
            _fail()
        digest = cast(dict[object, object], new)["blob_digest"]
        content = target_bytes[name]
        if (content is None) != (digest is None):
            _fail()
        if content is not None and (digest != _hash(content) or authority.read_blob(str(digest)) != content):
            _fail()
    if authority.read_blob(_hash(source_bytes)) != source_bytes:
        _fail()


def _prepare_locked(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    receipt_snapshot: storage.MemoryTargetSnapshot,
    workspace_id: str,
    sessions_root: Path,
) -> ManifestRecord:
    raw = receipt_snapshot.content
    if raw is None:
        _fail()
    receipt = receipt_codec.decode_memory_receipt_bytes(raw)
    captured = transaction.capture_transaction_payload(
        receipt,
        workspace_id=workspace_id,
        workspace_identity=operation.workspace_identity,
    )
    tx_manifest = _manifest(authority, "transactions", captured.identity)
    tx_records = tx_manifest.load()
    if (
        not 1 <= len(tx_records) <= 2
        or tx_records[0].phase != "CAPTURED"
        or (len(tx_records) == 2 and tx_records[1].phase != "PREPARED")
    ):
        _fail()
    tx_record = tx_records[0]
    loaded_captured = transaction.decode_transaction_payload("CAPTURED", tx_record.payload)
    if loaded_captured != captured:
        _fail()
    recorded_prepared = None
    if len(tx_records) == 2:
        recorded_prepared = transaction.decode_transaction_payload(
            "PREPARED", tx_records[1].payload, captured=captured,
        )
        if _receipt_stat(receipt_snapshot) != recorded_prepared.receipt_stat.to_payload():
            _fail()

    conversion_prepared = _conversion_evidence(
        authority, captured, tx_record, receipt_snapshot, receipt, operation,
    )
    if conversion_prepared.outputs is None or conversion_prepared.canonical_row is None:
        _fail()

    candidate = locate_recovery_candidate(
        authority.workspace, sessions_root, receipt.session_key or "",
    )
    with open_checked_recovery_session(operation, candidate) as checked_session:
        source_bytes = checked_session.read_source_bytes()
        source_payload, proposed_source_bytes = _source_proposal(source_bytes, receipt)

        snapshots = {name: authority.snapshot_target(name) for name in _TARGETS}
        history = snapshots["history.jsonl"].content
        cursor = snapshots[".cursor"].content
        memory = snapshots["MEMORY.md"].content
        if history is None or cursor is None:
            _fail()
        marker_row = _verified_marker_row(
            authority, conversion_prepared, receipt, history, cursor,
        )
        row_digest = _hash(marker_row)

        memory_proposal = (
            receipt.memory_output.encode("utf-8")
            if receipt.memory_output is not None and not receipt.reference_only
            else memory
        )
        target_proposals: dict[str, dict[str, object]] = {
            "history.jsonl": _target_descriptor(snapshots["history.jsonl"], history),
            ".cursor": _target_descriptor(snapshots[".cursor"], cursor),
            "MEMORY.md": _target_descriptor(snapshots["MEMORY.md"], memory_proposal),
        }
        prepared = transaction.prepare_transaction_payload(
            captured,
            receipt_stat=_receipt_stat(receipt_snapshot),
            targets=target_proposals,
            source=source_payload,
            history_marker_digest=captured.history_marker_digest,
            history_entry_digest=captured.history_entry_digest,
            history_row_digest=row_digest,
            memory_action="preserve" if receipt.reference_only else "replace",
        )

        target_bytes = {
            "history.jsonl": history,
            ".cursor": cursor,
            "MEMORY.md": memory_proposal,
        }
        _receipt_bound(authority, receipt_snapshot)
        if checked_session.read_source_bytes() != source_bytes:
            _fail()
        for snapshot in snapshots.values():
            _same_snapshot(authority, snapshot)

        if recorded_prepared is None:
            _record_proposal_blobs(authority, target_proposals, proposed_source_bytes, target_bytes)
        else:
            if prepared != recorded_prepared:
                _fail()
            for content in (*target_bytes.values(), proposed_source_bytes):
                if content is not None and authority.read_blob(_hash(content)) != content:
                    _fail()

        _receipt_bound(authority, receipt_snapshot)
        if checked_session.read_source_bytes() != source_bytes:
            _fail()
        for snapshot in snapshots.values():
            _same_snapshot(authority, snapshot)

        if recorded_prepared is not None:
            refreshed = tx_manifest.load()
            if len(refreshed) != 2 or refreshed[-1] != tx_records[-1]:
                _fail()
            return refreshed[-1]

        record = tx_manifest.append(
            predecessor_digest=tx_record.digest,
            phase="PREPARED",
            payload=prepared.to_payload(),
        )
        refreshed = tx_manifest.load()
        if len(refreshed) != 2 or refreshed[-1] != record:
            _fail()
        decoded = transaction.decode_transaction_payload(
            "PREPARED", refreshed[-1].payload, captured=captured,
        )
        if decoded != prepared:
            _fail()
        _receipt_bound(authority, receipt_snapshot)
        if checked_session.read_source_bytes() != source_bytes:
            _fail()
        for snapshot in snapshots.values():
            _same_snapshot(authority, snapshot)
        return refreshed[-1]


def prepare_memory_recovery_transaction(
    workspace: Path | str,
    *,
    workspace_id: str,
    sessions_root: Path | str,
) -> ManifestRecord:
    """Append or verify one immutable transaction PREPARED record.

    The workspace-memory lock is always acquired before checked-session locks.
    A receipt copied byte-for-byte before PREPARED is accepted as the same
    capture after fresh validation. Replays after PREPARED require its complete
    recorded receipt stat identity as well as the exact receipt bytes.
    """
    try:
        root = Path(sessions_root)
        if not root.is_absolute():
            _fail()
        with (
            storage.open_workspace_memory(workspace) as authority,
            authority.operation() as operation,
        ):
            receipt_snapshot = authority.snapshot_target(_RECEIPT)
            if receipt_snapshot.content is None:
                _fail()
            with operation.borrow_pending_receipt_parent(receipt_snapshot) as scope:
                current = scope.snapshot_target()
                if (
                    current.content != receipt_snapshot.content
                    or current.stat_identity != receipt_snapshot.stat_identity
                ):
                    _fail()
            result = _prepare_locked(
                authority,
                operation,
                receipt_snapshot,
                workspace_id,
                root,
            )
            return result
    except MemoryRecoveryTransactionPreparationError:
        raise
    except Exception:
        _fail()


def require_transaction_conversion_evidence(
    authority: storage.WorkspaceMemoryAuthority,
    captured: transaction.CapturedTransactionPayload,
    tx_record: ManifestRecord,
    receipt_snapshot: storage.MemoryTargetSnapshot | quarantine.ValidatedQuarantineCopy,
    receipt: receipt_codec.MemoryReceipt,
    operation: storage.WorkspaceMemoryOperation,
    *, read_only: bool = False,
) -> conversion.PreparedConversionPayload:
    """Require and return the transaction completed-conversion evidence."""
    return _conversion_evidence(authority, captured, tx_record, receipt_snapshot, receipt, operation, read_only=read_only)


def read_verified_transaction_marker_row(
    authority: storage.WorkspaceMemoryAuthority,
    prepared: conversion.PreparedConversionPayload,
    receipt: receipt_codec.MemoryReceipt,
    history_bytes: bytes,
    cursor_bytes: bytes,
) -> bytes:
    """Read the marker row after existing history, cursor and blob checks."""
    return _verified_marker_row(authority, prepared, receipt, history_bytes, cursor_bytes)


def parse_recovery_session_records(raw: bytes, session_key: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Parse session records through the ordered recovery-source checks."""
    return _session_records(raw, session_key)


def derive_recovery_source_proposal(
    raw: bytes,
    receipt: receipt_codec.MemoryReceipt,
) -> tuple[dict[str, int | str], bytes]:
    """Derive the checked source transition without applying it."""
    return _source_proposal(raw, receipt)


def transaction_target_descriptor(
    snapshot: storage.MemoryTargetSnapshot,
    new_content: bytes | None,
) -> dict[str, object]:
    """Project the original snapshot and proposed bytes to a target transition."""
    return _target_descriptor(snapshot, new_content)


def require_current_transaction_snapshot(
    authority: storage.WorkspaceMemoryAuthority,
    snapshot: storage.MemoryTargetSnapshot,
) -> None:
    """Require current bytes and stat identity for the issued snapshot."""
    return _same_snapshot(authority, snapshot)


def legacy_transaction_chunk_digest(messages: list[dict[str, object]]) -> str:
    """Digest source events with the existing import-bound legacy encoder callback."""
    return _legacy_chunk_digest(messages)


__all__ = [
    "MemoryRecoveryTransactionPreparationError",
    "prepare_memory_recovery_transaction",
    "require_transaction_receipt_binding",
    "require_transaction_conversion_evidence",
    "read_verified_transaction_marker_row",
    "parse_recovery_session_records",
    "derive_recovery_source_proposal",
    "transaction_target_descriptor",
    "require_current_transaction_snapshot",
    "legacy_transaction_chunk_digest",
]
