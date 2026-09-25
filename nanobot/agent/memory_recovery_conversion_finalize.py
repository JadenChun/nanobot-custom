"""Offline, replayable finalization of prepared receipt-tail conversion evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_conversion as conversion
from nanobot.agent import memory_recovery_conversion_apply as application
from nanobot.agent import memory_recovery_conversion_runtime as preparation
from nanobot.agent import memory_recovery_legacy_archive as archive
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_target_verify as verification
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent.memory_recovery_manifest import ManifestRecord

_TARGETS = ("history.jsonl", ".cursor", ".dream_cursor")
_RECEIPT = ".consolidation.pending.json"
_PHASES = (
    "CAPTURED", "CONVERSION_PREPARED", "JSONL_APPLIED", "CURSORS_APPLIED",
    "LEGACY_FINALIZED", "COMPLETE",
)


class ConversionFinalizationError(RuntimeError):
    """Content-free offline finalization failure."""

    def __init__(self) -> None:
        super().__init__("memory recovery conversion finalization is unavailable")


def _fail() -> None:
    raise ConversionFinalizationError() from None


def _decode(records: tuple[ManifestRecord, ...], captured: conversion.CapturedConversionPayload):
    if not 4 <= len(records) <= 6 or tuple(r.phase for r in records) != _PHASES[:len(records)]:
        _fail()
    if conversion.decode_conversion_payload(records[0].payload, "CAPTURED") != captured:
        _fail()
    prepared = conversion.decode_conversion_payload(
        records[1].payload, "CONVERSION_PREPARED", captured=captured,
    )
    jsonl = conversion.decode_conversion_payload(
        records[2].payload, "JSONL_APPLIED", captured=captured, prepared=prepared,
    )
    cursors = conversion.decode_conversion_payload(
        records[3].payload, "CURSORS_APPLIED", captured=captured,
        prepared=prepared, jsonl_applied=jsonl,
    )
    finalized = None
    if len(records) >= 5:
        finalized = conversion.decode_conversion_payload(
            records[4].payload, "LEGACY_FINALIZED", captured=captured,
            prepared=prepared, jsonl_applied=jsonl, cursors_applied=cursors,
            prepared_record_digest=records[1].digest,
            jsonl_applied_record_digest=records[2].digest,
            cursors_applied_record_digest=records[3].digest,
        )
    if len(records) == 6:
        conversion.decode_conversion_payload(
            records[5].payload, "COMPLETE", captured=captured, prepared=prepared,
            jsonl_applied=jsonl, cursors_applied=cursors, legacy_finalized=finalized,
            finalized_record_digest=records[4].digest,
        )
    return prepared, jsonl, cursors, finalized


def _verify_archive(
    operation: storage.WorkspaceMemoryOperation,
    receipt_snapshot: storage.MemoryTargetSnapshot,
    prepared: conversion.PreparedConversionPayload,
) -> None:
    if prepared.source.stat is None or prepared.backup is None or prepared.receipt is None:
        _fail()
    old = prepared.source
    info = old.stat
    source_stat = storage.MemoryTargetStatIdentity(
        info.device, info.inode, info.mode, info.uid, info.gid, info.links,
        info.size, info.mtime_ns, info.ctime_ns,
    )
    archive.archive_legacy_history_if_unchanged(
        operation, receipt_snapshot, expected_source_stat=source_stat,
        expected_digest=old.digest, expected_size=old.size,
        backup_name=prepared.backup.basename,
        expected_receipt_digest=prepared.receipt["blob_digest"],
        expected_receipt_size=prepared.receipt["size"],
    )


def _verify(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    receipt_snapshot: storage.MemoryTargetSnapshot,
    receipt: dict,
    tx_record: ManifestRecord,
    loaded_tx: transaction.CapturedTransactionPayload,
    captured: conversion.CapturedConversionPayload,
    prepared: conversion.PreparedConversionPayload,
    *,
    finalized: bool,
) -> None:
    application._bound_receipt(authority, receipt_snapshot)
    frozen_source = application._old_bytes(authority, prepared.source)
    if frozen_source is None:
        _fail()
    frozen = {name: application._old_bytes(authority, prepared.targets[name]) for name in _TARGETS}
    source_claim = prepared.source.to_payload()
    target_claims = {name: prepared.targets[name].to_payload() for name in _TARGETS}
    if prepared.disposition == "convert":
        claim = prepared.receipt
        raw = receipt_snapshot.content
        if (
            raw is None or claim is None or prepared.backup is None
            or prepared.transaction_captured_record_digest != tx_record.digest
            or claim["size"] != len(raw)
            or claim["blob_digest"] != hashlib.sha256(raw).hexdigest()
            or authority.read_blob(claim["blob_digest"]) != raw
        ):
            _fail()
        rebuilt = conversion.prepare_conversion_payload(
            captured, disposition="convert", source=source_claim, targets=target_claims,
            source_bytes=frozen_source, target_contents=frozen,
            backup=prepared.backup.to_payload(), fallback_timestamp=prepared.fallback_timestamp,
            receipt=receipt, transaction_captured=loaded_tx,
            transaction_captured_record=tx_record,
        )
        if prepared.outputs is None or prepared.canonical_row is None:
            _fail()
        output = {name: application._output_bytes(authority, prepared.outputs[name]) for name in _TARGETS}
        derived = preparation._output_bytes(prepared, frozen_source)
        if any(output[name] != derived[name] for name in _TARGETS):
            _fail()
        row = derived["history.jsonl"].splitlines(keepends=True)[-1]
        canonical = prepared.canonical_row
        if (
            canonical["size"] != len(row)
            or canonical["blob_digest"] != hashlib.sha256(row).hexdigest()
            or authority.read_blob(canonical["blob_digest"]) != row
        ):
            _fail()
    elif prepared.disposition == "preserve_nonempty_jsonl":
        rebuilt = conversion.prepare_conversion_payload(
            captured, disposition="preserve_nonempty_jsonl", source=source_claim,
            targets=target_claims, source_bytes=None,
        )
        output = frozen
    else:
        _fail()
    if rebuilt.to_payload() != prepared.to_payload():
        _fail()
    for name in _TARGETS:
        application._bound_receipt(authority, receipt_snapshot)
        snapshot = authority.snapshot_target(name)
        application._status(snapshot, prepared.targets[name], frozen[name], output[name])
        if snapshot.content is not None:
            verification.durably_verify_target(operation, snapshot, output[name])
        elif output[name] is not None:
            _fail()
    application._bound_receipt(authority, receipt_snapshot)
    if prepared.disposition == "preserve_nonempty_jsonl":
        if not application._descriptor_matches(
            authority.snapshot_target("HISTORY.md"), prepared.source, frozen_source,
        ):
            _fail()
        verification.durably_verify_target(
            operation, authority.snapshot_target("HISTORY.md"), frozen_source,
        )
    elif finalized:
        if authority.snapshot_target("HISTORY.md").exists:
            _fail()
        # A terminal phase requires the helper's descriptor-bound, durable
        # backup-only proof, not merely absence of the old source name.
        _verify_archive(operation, receipt_snapshot, prepared)
        application._bound_receipt(authority, receipt_snapshot)
        if authority.snapshot_target("HISTORY.md").exists:
            _fail()
    # Before archive, the source may have two links after a prior interrupted
    # link. The normal snapshot boundary rejects linked files; the archive
    # helper alone verifies the frozen source and both names descriptor-relative.


def finalize_prepared_conversion(workspace: Path | str, *, workspace_id: str) -> ManifestRecord:
    """Verify durable targets, archive legacy history, and claim terminal conversion phases."""
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
                if (current.content, current.stat_identity) != (
                    receipt_snapshot.content, receipt_snapshot.stat_identity,
                ):
                    _fail()
            raw = receipt_snapshot.content
            receipt = receipt_codec._validate_payload(receipt_codec._decode(raw), raw)
            captured_tx = transaction.capture_transaction_payload(
                receipt, workspace_id=workspace_id,
                workspace_identity=operation.workspace_identity,
            )
            tx_records = application._chain(authority, "transactions", captured_tx.identity).load()
            if len(tx_records) != 1 or tx_records[0].phase != "CAPTURED":
                _fail()
            tx_record = tx_records[0]
            loaded_tx = transaction.decode_transaction_payload("CAPTURED", tx_record.payload)
            if loaded_tx != captured_tx:
                _fail()
            v1 = conversion.capture_conversion_payload(captured_tx.identity)
            captured = conversion.capture_receipt_tail_conversion_payload(captured_tx.identity)
            if application._chain(authority, "conversions", v1.identity).load():
                _fail()
            manifest = application._chain(authority, "conversions", captured.identity)
            records = manifest.load()
            prepared, jsonl, cursors, finalized = _decode(records, captured)
            _verify(
                authority, operation, receipt_snapshot, receipt, tx_record, loaded_tx,
                captured, prepared, finalized=finalized is not None,
            )
            if prepared.disposition == "convert":
                _verify_archive(operation, receipt_snapshot, prepared)
                if authority.snapshot_target("HISTORY.md").exists:
                    _fail()
            _verify(
                authority, operation, receipt_snapshot, receipt, tx_record, loaded_tx,
                captured, prepared, finalized=True,
            )
            if finalized is None:
                payload = conversion.legacy_finalized_conversion_payload(
                    captured, prepared, jsonl, cursors,
                    prepared_record_digest=records[1].digest,
                    jsonl_applied_record_digest=records[2].digest,
                    cursors_applied_record_digest=records[3].digest,
                )
                application._bound_receipt(authority, receipt_snapshot)
                manifest.append(
                    predecessor_digest=records[3].digest, phase="LEGACY_FINALIZED",
                    payload=payload.to_payload(),
                )
                records = manifest.load()
                prepared, jsonl, cursors, finalized = _decode(records, captured)
                _verify(
                    authority, operation, receipt_snapshot, receipt, tx_record, loaded_tx,
                    captured, prepared, finalized=True,
                )
            if finalized is None:
                _fail()
            if len(records) == 5:
                payload = conversion.complete_conversion_payload(
                    captured, prepared, jsonl, cursors, finalized,
                    finalized_record_digest=records[4].digest,
                )
                application._bound_receipt(authority, receipt_snapshot)
                manifest.append(
                    predecessor_digest=records[4].digest, phase="COMPLETE",
                    payload=payload.to_payload(),
                )
            records = manifest.load()
            prepared, _, _, _ = _decode(records, captured)
            if len(records) != 6:
                _fail()
            _verify(
                authority, operation, receipt_snapshot, receipt, tx_record, loaded_tx,
                captured, prepared, finalized=True,
            )
            return records[-1]
    except ConversionFinalizationError:
        raise
    except Exception:
        _fail()


__all__ = ["ConversionFinalizationError", "finalize_prepared_conversion"]
