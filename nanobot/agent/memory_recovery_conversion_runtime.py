"""Explicit offline preparation of one receipt-tail history conversion.

Only the authenticated tail frame and immutable evidence are written here. The
legacy migration target files are handled by later phases, not this coordinator.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_conversion as conversion
from nanobot.agent import memory_recovery_legacy_archive as archive
from nanobot.agent import memory_recovery_marker_stage as marker_stage
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent.legacy_history_codec import parse_legacy_history
from nanobot.agent.memory_recovery_manifest import JsonValue, ManifestRecord, MemoryRecoveryManifest

_TARGETS = ("history.jsonl", ".cursor", ".dream_cursor")
_RECEIPT = ".consolidation.pending.json"


class ConversionPreparationError(RuntimeError):
    """Fixed, content-free failure at the offline preparation boundary."""

    def __init__(self) -> None:
        super().__init__("memory recovery conversion preparation is unavailable")


def _fail() -> NoReturn:
    raise ConversionPreparationError() from None


def _descriptor(snapshot: storage.MemoryTargetSnapshot, blob: str | None) -> dict[str, object]:
    info = snapshot.stat_identity
    content = snapshot.content
    if content is None:
        if info is not None or blob is not None:
            _fail()
        return {"exists": False, "size": 0, "digest": None, "stat": None, "blob_digest": None}
    if info is None or blob != hashlib.sha256(content).hexdigest():
        _fail()
    return {
        "exists": True,
        "size": len(content),
        "digest": blob,
        "blob_digest": blob,
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


def _published(authority: storage.WorkspaceMemoryAuthority, content: bytes | None) -> str | None:
    if content is None:
        return None
    digest = authority.publish_blob(content)
    if authority.read_blob(digest) != content:
        _fail()
    return digest


def _chain(
    authority: storage.WorkspaceMemoryAuthority, kind: str, identity: str
) -> MemoryRecoveryManifest:
    return MemoryRecoveryManifest(authority.open_evidence_namespace(kind, identity))


def _loaded_captured(
    manifest: MemoryRecoveryManifest,
    payload: dict[str, JsonValue],
) -> ManifestRecord:
    records = manifest.load()
    if len(records) > 1 or (records and records[0].phase != "CAPTURED"):
        _fail()
    record = manifest.create(payload)
    if manifest.load() != (record,):
        _fail()
    return record


def _same_disk(
    authority: storage.WorkspaceMemoryAuthority,
    name: str,
    descriptor: conversion.OldFileDescriptor,
) -> bool:
    snapshot = authority.snapshot_target(name)
    info = snapshot.stat_identity
    old = descriptor.stat
    return (
        snapshot.exists == descriptor.exists
        and (
            snapshot.content is None
            or (
                len(snapshot.content) == descriptor.size
                and hashlib.sha256(snapshot.content).hexdigest() == descriptor.digest
                and authority.read_blob(descriptor.blob_digest or "") == snapshot.content
            )
        )
        and (
            (info is None and old is None)
            or (
                info is not None
                and old is not None
                and (
                    info.device,
                    info.inode,
                    info.mode,
                    info.uid,
                    info.gid,
                    info.links,
                    info.size,
                    info.mtime_ns,
                    info.ctime_ns,
                )
                == (
                    old.device,
                    old.inode,
                    old.mode,
                    old.uid,
                    old.gid,
                    old.links,
                    old.size,
                    old.mtime_ns,
                    old.ctime_ns,
                )
            )
        )
    )


def _output_bytes(
    prepared: conversion.PreparedConversionPayload,
    source: bytes,
) -> dict[str, bytes]:
    frame = prepared.tail_frame
    if frame is None or prepared.fallback_timestamp is None or prepared.final_legacy_cursor is None:
        _fail()
    offset = frame["offset"]
    prefix = parse_legacy_history(source[:offset], fallback_timestamp=prepared.fallback_timestamp)
    tail = source[offset:]
    rows = [(row.cursor, row.timestamp, row.content) for row in prefix]
    rows.append(
        (prepared.final_legacy_cursor, prepared.fallback_timestamp, tail[:-2].decode("utf-8"))
    )
    jsonl = b"".join(
        (
            json.dumps(
                {"cursor": cursor, "timestamp": timestamp, "content": content}, ensure_ascii=False
            )
            + "\n"
        ).encode("utf-8")
        for cursor, timestamp, content in rows
    )
    counter = str(prepared.final_legacy_cursor).encode("utf-8")
    return {"history.jsonl": jsonl, ".cursor": counter, ".dream_cursor": counter}


def _fallback_timestamp(snapshot: storage.MemoryTargetSnapshot) -> str:
    info = snapshot.stat_identity
    if info is None or info.mtime_ns is None:
        _fail()
    return datetime.fromtimestamp(info.mtime_ns // 1_000_000_000, timezone.utc).strftime(
        "%Y-%m-%d %H:%M"
    )


def _backup_plan(name: str) -> dict[str, object]:
    suffix = 1 if name == "HISTORY.md.bak" else int(name.rsplit(".", 1)[1])
    return {
        "basename": name,
        "candidates": [
            {
                "basename": "HISTORY.md.bak" if n == 1 else f"HISTORY.md.bak.{n}",
                "exists": n < suffix,
            }
            for n in range(1, suffix + 1)
        ],
    }


def _verify_outputs(
    authority: storage.WorkspaceMemoryAuthority,
    prepared: conversion.PreparedConversionPayload,
    source: bytes,
    *,
    publish: bool,
) -> None:
    if prepared.outputs is None or prepared.canonical_row is None:
        _fail()
    contents = _output_bytes(prepared, source)
    for name, content in contents.items():
        expected = prepared.outputs[name]
        if publish:
            _published(authority, content)
        observed = authority.read_blob(expected.blob_digest or "")
        if (
            expected.blob_digest != hashlib.sha256(content).hexdigest()
            or expected.size != len(content)
            or observed != content
        ):
            _fail()
    row = contents["history.jsonl"].splitlines(keepends=True)[-1]
    claim = prepared.canonical_row
    if publish:
        _published(authority, row)
    observed_row = authority.read_blob(claim["blob_digest"])
    if (
        claim["blob_digest"] != hashlib.sha256(row).hexdigest()
        or claim["size"] != len(row)
        or observed_row != row
        or claim["cursor"] != prepared.final_legacy_cursor
    ):
        _fail()


def prepare_receipt_tail_conversion(workspace: Path | str, *, workspace_id: str) -> ManifestRecord:
    """Stage and record a v2 PREPARED conversion, under one workspace lock.

    Replays stop at PREPARED. A later transaction/conversion phase requires the
    dedicated application coordinator and is deliberately outside this API.
    """
    try:
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
            raw_receipt = receipt_snapshot.content

            def receipt_still_bound() -> bool:
                fresh = authority.snapshot_target(_RECEIPT)
                return (
                    fresh.content == receipt_snapshot.content
                    and fresh.stat_identity == receipt_snapshot.stat_identity
                )

            receipt = receipt_codec.decode_memory_receipt_bytes(raw_receipt)
            captured_transaction = transaction.capture_transaction_payload(
                receipt,
                workspace_id=workspace_id,
                workspace_identity=operation.workspace_identity,
            )
            tx_manifest = _chain(authority, "transactions", captured_transaction.identity)
            if len(tx_manifest.load()) > 1:
                _fail()
            v1 = conversion.capture_conversion_payload(captured_transaction.identity)
            v2 = conversion.capture_receipt_tail_conversion_payload(captured_transaction.identity)
            v1_records = _chain(authority, "conversions", v1.identity).load()
            v2_manifest = _chain(authority, "conversions", v2.identity)
            v2_records = v2_manifest.load()
            if v1_records or len(v2_records) > 2:
                _fail()
            if len(v2_records) == 2 and v2_records[1].phase != "CONVERSION_PREPARED":
                _fail()
            if v2_records and v2_records[0].phase != "CAPTURED":
                _fail()
            tx_record = _loaded_captured(tx_manifest, captured_transaction.to_payload())
            loaded_transaction = transaction.decode_transaction_payload(
                "CAPTURED", tx_record.payload
            )
            if loaded_transaction != captured_transaction:
                _fail()
            _published(authority, raw_receipt)
            cv_record = (
                _loaded_captured(v2_manifest, v2.to_payload())
                if len(v2_records) < 2
                else v2_records[0]
            )
            loaded_conversion = conversion.decode_conversion_payload(cv_record.payload, "CAPTURED")
            if loaded_conversion != v2:
                _fail()
            if len(v2_records) == 2:
                prepared_record = v2_records[1]
                prepared = conversion.decode_conversion_payload(
                    prepared_record.payload,
                    "CONVERSION_PREPARED",
                    captured=v2,
                )
                if not _same_disk(authority, "HISTORY.md", prepared.source):
                    _fail()
                if any(
                    not _same_disk(authority, name, prepared.targets[name]) for name in _TARGETS
                ):
                    _fail()
                source_snapshot = authority.snapshot_target("HISTORY.md")
                target_snapshots = {name: authority.snapshot_target(name) for name in _TARGETS}
                source_bytes = source_snapshot.content
                if source_bytes is None:
                    _fail()
                source_claim = _descriptor(source_snapshot, prepared.source.blob_digest)
                target_claims = {
                    name: _descriptor(snapshot, prepared.targets[name].blob_digest)
                    for name, snapshot in target_snapshots.items()
                }
                if prepared.disposition == "convert":
                    backup_name = archive.choose_legacy_backup_name(operation, source_snapshot)
                    rebuilt = conversion.prepare_conversion_payload(
                        v2,
                        disposition="convert",
                        source=source_claim,
                        targets=target_claims,
                        source_bytes=source_bytes,
                        target_contents={name: target_snapshots[name].content for name in _TARGETS},
                        backup=_backup_plan(backup_name),
                        fallback_timestamp=_fallback_timestamp(source_snapshot),
                        receipt=receipt,
                        transaction_captured=loaded_transaction,
                        transaction_captured_record=tx_record,
                    )
                    _verify_outputs(authority, rebuilt, source_bytes, publish=False)
                elif prepared.disposition == "preserve_nonempty_jsonl":
                    rebuilt = conversion.prepare_conversion_payload(
                        v2,
                        disposition="preserve_nonempty_jsonl",
                        source=source_claim,
                        targets=target_claims,
                        source_bytes=None,
                    )
                else:
                    _fail()
                if rebuilt.to_payload() != prepared_record.payload:
                    _fail()
                if not receipt_still_bound():
                    _fail()
                return prepared_record

            initial_targets = {name: authority.snapshot_target(name) for name in _TARGETS}
            convert = initial_targets["history.jsonl"].content in (None, b"")
            if convert:
                staged = marker_stage.stage_history_marker(
                    authority,
                    operation,
                    receipt.history_marker_id,
                    receipt.history_entry,
                )
                source_snapshot = staged.snapshot
            else:
                source_snapshot = authority.snapshot_target("HISTORY.md")
            if source_snapshot.content is None:
                _fail()
            targets = {name: authority.snapshot_target(name) for name in _TARGETS}
            source_bytes = source_snapshot.content
            source = _descriptor(source_snapshot, _published(authority, source_bytes))
            target_descriptors = {
                name: _descriptor(snapshot, _published(authority, snapshot.content))
                for name, snapshot in targets.items()
            }
            backup_name: str | None = None
            if convert:
                backup_name = archive.choose_legacy_backup_name(operation, source_snapshot)
                backup = _backup_plan(backup_name)
                fallback = _fallback_timestamp(source_snapshot)
                prepared = conversion.prepare_conversion_payload(
                    v2,
                    disposition="convert",
                    source=source,
                    targets=target_descriptors,
                    source_bytes=source_bytes,
                    target_contents={name: targets[name].content for name in _TARGETS},
                    backup=backup,
                    fallback_timestamp=fallback,
                    receipt=receipt,
                    transaction_captured=loaded_transaction,
                    transaction_captured_record=tx_record,
                )
                _verify_outputs(authority, prepared, source_bytes, publish=True)
            else:
                prepared = conversion.prepare_conversion_payload(
                    v2,
                    disposition="preserve_nonempty_jsonl",
                    source=source,
                    targets=target_descriptors,
                    source_bytes=None,
                )
            if not receipt_still_bound() or not _same_disk(
                authority, "HISTORY.md", prepared.source
            ):
                _fail()
            if any(not _same_disk(authority, name, prepared.targets[name]) for name in _TARGETS):
                _fail()
            if (
                convert
                and archive.choose_legacy_backup_name(operation, source_snapshot) != backup_name
            ):
                _fail()
            return v2_manifest.append(
                predecessor_digest=cv_record.digest,
                phase="CONVERSION_PREPARED",
                payload=prepared.to_payload(),
            )
    except ConversionPreparationError:
        raise
    except Exception:
        _fail()


def derive_conversion_output_bytes(
    prepared: conversion.PreparedConversionPayload,
    source: bytes,
) -> dict[str, bytes]:
    """Derive conversion output from the checked frozen preparation evidence."""
    return _output_bytes(prepared, source)


__all__ = [
    "ConversionPreparationError",
    "prepare_receipt_tail_conversion",
    "derive_conversion_output_bytes",
]
