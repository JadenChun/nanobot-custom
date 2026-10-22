"""Reconstruct a sealed historical claim solely from retained immutable evidence.

The finalizer's live proof establishes the event. This verifier checks the
frozen projection after normal runtime may have changed current targets; it
does not claim to replay the unavailable old checked-session source.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_conversion as conversion
from nanobot.agent import memory_recovery_conversion_finalize as conversion_final
from nanobot.agent import memory_recovery_conversion_runtime as conversion_runtime
from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent import memory_recovery_transaction_prepare as preparation
from nanobot.agent.memory_recovery_admission_reader import ExistingEvidenceReader
from nanobot.agent.memory_recovery_manifest import ManifestRecord

_TARGETS = ("history.jsonl", ".cursor", "MEMORY.md")
_CONVERSION_TARGETS = ("history.jsonl", ".cursor", ".dream_cursor")
_STAT_KEYS = ("device", "inode", "mode", "uid", "gid", "links", "size", "mtime_ns", "ctime_ns")


class HistoricalSealError(RuntimeError):
    """Fixed content-free semantic admission failure."""

    def __init__(self) -> None:
        super().__init__("historical memory recovery seal is unavailable")


def _fail() -> NoReturn:
    raise HistoricalSealError() from None


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class HistoricalSealEvidence:
    """Digest-only validation result, not a startup-admission capability."""

    sealed_identities: tuple[str, ...]
    semantic_seal_digests: tuple[str, ...]


def _frozen_old(reader: ExistingEvidenceReader, recovery_fd: int,
                descriptor: conversion.OldFileDescriptor) -> bytes | None:
    if not descriptor.exists:
        return None
    if descriptor.blob_digest is None or descriptor.digest is None:
        _fail()
    raw = reader.blob(recovery_fd, descriptor.blob_digest, descriptor.size)
    if _digest(raw) != descriptor.digest:
        _fail()
    return raw


def _frozen_new(reader: ExistingEvidenceReader, recovery_fd: int,
                descriptor: conversion.NewFileDescriptor | transaction.NewTargetDescriptor) -> bytes | None:
    if not descriptor.exists:
        return None
    if descriptor.blob_digest is None:
        _fail()
    return reader.blob(recovery_fd, descriptor.blob_digest, descriptor.size)


def _transaction_phases(records: tuple[ManifestRecord, ...]):
    captured = transaction.decode_transaction_payload("CAPTURED", records[0].payload)
    assert isinstance(captured, transaction.CapturedTransactionPayload)
    prepared = transaction.decode_transaction_payload("PREPARED", records[1].payload,
                                                      captured=captured)
    assert isinstance(prepared, transaction.PreparedTransactionPayload)
    history = transaction.decode_transaction_payload("HISTORY_APPLIED", records[2].payload,
                                                     captured=captured, prepared=prepared)
    assert isinstance(history, transaction.HistoryAppliedTransactionPayload)
    if history != transaction.history_applied_transaction_payload(captured, prepared):
        _fail()
    memory = transaction.decode_transaction_payload("MEMORY_APPLIED", records[3].payload,
                                                    captured=captured, prepared=prepared,
                                                    history_applied=history)
    assert isinstance(memory, transaction.MemoryAppliedTransactionPayload)
    if memory != transaction.memory_applied_transaction_payload(captured, prepared, history):
        _fail()
    offset = transaction.decode_transaction_payload("OFFSET_APPLIED", records[4].payload,
                                                    captured=captured, prepared=prepared,
                                                    history_applied=history, memory_applied=memory)
    assert isinstance(offset, transaction.OffsetAppliedTransactionPayload)
    if offset != transaction.offset_applied_transaction_payload(captured, prepared, history, memory):
        _fail()
    committed = transaction.decode_transaction_payload(
        "RECEIPT_COMMITTED", records[5].payload, captured=captured, prepared=prepared,
        history_applied=history, memory_applied=memory, offset_applied=offset,
        prepared_record_digest=records[1].digest,
        history_applied_record_digest=records[2].digest,
        memory_applied_record_digest=records[3].digest,
        offset_applied_record_digest=records[4].digest,
    )
    assert isinstance(committed, transaction.ReceiptCommittedTransactionPayload)
    if committed != transaction.receipt_committed_transaction_payload(
        captured, prepared, history, memory, offset,
        prepared_record_digest=records[1].digest,
        history_applied_record_digest=records[2].digest,
        memory_applied_record_digest=records[3].digest,
        offset_applied_record_digest=records[4].digest,
    ):
        _fail()
    complete = transaction.decode_transaction_payload(
        "COMPLETE", records[6].payload, captured=captured, prepared=prepared,
        history_applied=history, memory_applied=memory, offset_applied=offset,
        receipt_committed=committed, commit_record_digest=records[5].digest,
        prepared_record_digest=records[1].digest,
        history_applied_record_digest=records[2].digest,
        memory_applied_record_digest=records[3].digest,
        offset_applied_record_digest=records[4].digest,
    )
    assert isinstance(complete, transaction.CompleteTransactionPayload)
    if complete != transaction.complete_transaction_payload(
        captured, prepared, history, memory, offset, committed,
        commit_record_digest=records[5].digest,
        prepared_record_digest=records[1].digest,
        history_applied_record_digest=records[2].digest,
        memory_applied_record_digest=records[3].digest,
        offset_applied_record_digest=records[4].digest,
    ):
        _fail()
    return captured, prepared


def _projection(reader: ExistingEvidenceReader, recovery_fd: int,
                receipt: receipt_codec.MemoryReceipt, records: tuple[ManifestRecord, ...],
                original_stat: list[int], workspace_id: str) -> dict[str, object]:
    captured, prepared = _transaction_phases(records)
    expected = transaction.capture_transaction_payload(
        receipt, workspace_id=workspace_id,
        workspace_identity=(reader.workspace_stat[0], reader.workspace_stat[1]),
    )
    if (captured != expected or captured.identity != records[0].identity
            or prepared.receipt_stat.to_payload() != dict(zip(_STAT_KEYS, original_stat, strict=True))):
        _fail()
    if reader.blob(recovery_fd, captured.receipt_digest, captured.receipt_size) != receipt.raw_bytes:
        _fail()

    v1 = conversion.capture_conversion_payload(captured.identity)
    reader.empty_conversion(recovery_fd, v1.identity)
    v2 = conversion.capture_receipt_tail_conversion_payload(captured.identity)
    converted_records = reader.chain(recovery_fd, "conversions", v2.identity)
    converted, _, _, _ = conversion_final._decode(converted_records, v2)
    if not isinstance(cast(object, converted), conversion.PreparedConversionPayload):
        _fail()
    if (converted.disposition != "convert" or converted.captured != v2
            or converted.receipt is None or converted.receipt.get("size") != captured.receipt_size
            or converted.receipt.get("blob_digest") != captured.receipt_digest
            or converted.transaction_captured_record_digest != records[0].digest
            or converted.outputs is None or converted.canonical_row is None
            or converted.backup is None or converted.source.digest is None):
        _fail()

    frozen_source = _frozen_old(reader, recovery_fd, converted.source)
    if frozen_source is None:
        _fail()
    old = {name: _frozen_old(reader, recovery_fd, converted.targets[name])
           for name in _CONVERSION_TARGETS}
    rebuilt = conversion.prepare_conversion_payload(
        v2, disposition="convert", source=converted.source.to_payload(),
        targets={name: converted.targets[name].to_payload() for name in _CONVERSION_TARGETS},
        source_bytes=frozen_source, target_contents=old,
        backup=converted.backup.to_payload(), fallback_timestamp=converted.fallback_timestamp,
        receipt=receipt, transaction_captured=captured, transaction_captured_record=records[0],
    )
    if rebuilt != converted:
        _fail()
    outputs = conversion_runtime._output_bytes(converted, frozen_source)
    for name in _CONVERSION_TARGETS:
        if _frozen_new(reader, recovery_fd, converted.outputs[name]) != outputs[name]:
            _fail()
    row = outputs["history.jsonl"].splitlines(keepends=True)[-1]
    row_claim = converted.canonical_row
    expected_content = (
        f"<!-- nanobot-consolidation:{receipt.history_marker_id} -->\n"
        f"{receipt.history_entry}"
    ).rstrip()
    expected_row = (json.dumps({
        "cursor": row_claim["cursor"],
        "timestamp": converted.fallback_timestamp,
        "content": expected_content,
    }, ensure_ascii=False) + "\n").encode("utf-8")
    if (row_claim["size"] != len(row) or row_claim["blob_digest"] != _digest(row)
            or row != expected_row or row_claim["cursor"] != converted.final_legacy_cursor
            or reader.blob(recovery_fd, row_claim["blob_digest"], len(row)) != row
            or outputs["history.jsonl"].count(
                f"<!-- nanobot-consolidation:{receipt.history_marker_id} -->".encode()) != 1
            or outputs[".cursor"] != str(row_claim["cursor"]).encode("ascii")
            or _digest(row) != prepared.history_row_digest):
        _fail()
    preparation._parse_history_rows(outputs["history.jsonl"])

    for name in ("history.jsonl", ".cursor"):
        previous = prepared.targets[name].old
        if (not previous.exists or previous.size != len(outputs[name])
                or previous.digest != _digest(outputs[name])):
            _fail()
        if _frozen_new(reader, recovery_fd, prepared.targets[name].new) != outputs[name]:
            _fail()
    memory_new = _frozen_new(reader, recovery_fd, prepared.targets["MEMORY.md"].new)
    if prepared.memory_action == "replace":
        if receipt.reference_only or receipt.memory_output is None or memory_new != receipt.memory_output.encode("utf-8"):
            _fail()
    elif prepared.memory_action == "preserve":
        old_memory = prepared.targets["MEMORY.md"].old
        if not receipt.reference_only or (memory_new is None) != (not old_memory.exists):
            _fail()
        if memory_new is not None and (_digest(memory_new) != old_memory.digest
                                       or len(memory_new) != old_memory.size):
            _fail()
    else:
        _fail()

    source = prepared.source
    new_source = reader.blob(recovery_fd, source.new_digest, source.new_size)
    header, events = preparation._session_records(new_source, receipt.session_key or "")
    if (type(header.get("generation")) is not int or header["generation"] != source.new_generation
            or type(header.get("revision")) is not int or header["revision"] != source.new_revision
            or type(header.get("last_archived")) is not int or header["last_archived"] != source.new_cursor
            or header.get("last_consolidated") != source.new_cursor
            or source.new_generation != captured.session_generation
            or source.new_revision != source.old_revision + 1
            or source.old_cursor != captured.start_offset or source.new_cursor != captured.end_offset
            or not source.old_cursor < source.new_cursor <= len(events)
            or preparation._legacy_chunk_digest(events[source.old_cursor:source.new_cursor])
            != receipt.chunk_digest):
        _fail()
    return {
        "schema_version": 1,
        "transaction_identity": captured.identity,
        "transaction_complete_digest": records[-1].digest,
        "conversion_complete_digest": converted_records[-1].digest,
        "receipt_digest": captured.receipt_digest,
        "receipt_size": captured.receipt_size,
        "receipt_original_stat": original_stat,
        "prepared_target_new_digests": {
            name: prepared.targets[name].new.blob_digest for name in _TARGETS
        },
        "session_source_new_digest": source.new_digest,
        "archive_backup_digest": converted.source.digest,
        "archive_backup_name": converted.backup.basename,
    }


def _one(reader: ExistingEvidenceReader, recovery_fd: int, index_fd: int,
         identity: str, seal: str, sessions_root: Path) -> None:
    with reader.directory(index_fd, identity) as directory:
        if quarantine._names(directory, 4) != {
            "prepared.json", "copy-ready.json", "committed.json", "receipt.bin",
        }:
            _fail()
        previous = None
        records: list[dict[str, object]] = []
        for phase in ("prepared", "copy-ready", "committed"):
            raw = reader.file(directory, f"{phase}.json", 4096)
            record = quarantine._parse(raw, phase, previous)
            previous = _digest(raw)
            records.append(dict(record))
        first, second, final = records
        copy = reader.file(directory, "receipt.bin", 64 * 1024 * 1024)
        copied_stat = quarantine._entry(directory, "receipt.bin")
        if copied_stat is None:
            _fail()
        if (first["transaction_identity"] != identity or final["version"] != 2
                or final["semantic_seal_digest"] != seal
                or any(record[key] != first[key] for record in records[1:] for key in (
                    "workspace", "workspace_stat", "workspace_id", "sessions_root",
                    "sessions_root_stat", "transaction_identity", "complete_digest",
                    "receipt_digest", "receipt_size", "receipt_stat",
                ))
                or first["workspace"] != str(reader.gate.workspace)
                or first["workspace_stat"] != list(reader.workspace_stat[:5])
                or first["sessions_root"] != str(sessions_root)
                or first["copy_stat"] is not None
                or second["copy_stat"] != final["copy_stat"]
                or second["copy_stat"] != list(quarantine._stat_tuple(copied_stat))
                or len(copy) != first["receipt_size"] or _digest(copy) != first["receipt_digest"]):
            _fail()
        # Q2's strict parser has already checked these dynamic field types.
        workspace_id = cast(str, first["workspace_id"])
        original_stat = cast(list[int], first["receipt_stat"])
        reader.workspace_id(workspace_id)
        reader.sessions(sessions_root, workspace_id)
        receipt = receipt_codec.decode_memory_receipt_bytes(copy)
        records_tx = reader.chain(recovery_fd, "transactions", identity)
        if records_tx[-1].digest != first["complete_digest"]:
            _fail()
        claims = _projection(reader, recovery_fd, receipt, records_tx,
                             original_stat, workspace_id)
        canonical = (json.dumps(claims, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=True) + "\n").encode("utf-8")
        if _digest(canonical) != seal:
            _fail()
        os.fsync(directory)
        reader.check()
        if (reader.file(directory, "committed.json", 4096) != quarantine._canonical(final)
                or reader.file(directory, "receipt.bin", 64 * 1024 * 1024) != copy):
            _fail()


def validate_historical_seals_under_gate(
    gate: storage._WorkspaceMemoryLockGate, sessions_root: Path | str,
) -> HistoricalSealEvidence:
    """Check every durable seal under a caller-held existing workspace lock.

    Startup admission calls this through memory_recovery_startup_admission.
    """
    try:
        reader = ExistingEvidenceReader(gate)
        root = quarantine._absolute(sessions_root)
        durable = quarantine.establish_durable_quarantine_seals(gate)
        if quarantine._entry(reader.workspace_fd, "memory") is None:
            if durable.sealed_identities:
                _fail()
            reader.check()
            return HistoricalSealEvidence((), ())
        with reader.directory(reader.workspace_fd, "memory", memory=True) as memory:
            if quarantine._entry(memory, ".recovery") is None:
                if durable.sealed_identities:
                    _fail()
                reader.check()
                return HistoricalSealEvidence((), ())
            with reader.directory(memory, ".recovery") as recovery:
                if quarantine._entry(recovery, "receipt-quarantine") is None:
                    if durable.sealed_identities:
                        _fail()
                    reader.check()
                    return HistoricalSealEvidence((), ())
                with reader.directory(recovery, "receipt-quarantine") as index:
                    before = quarantine._sealed_fingerprints(gate, index, sync=False)
                    if tuple((name, digest) for name, digest, _ in before) != tuple(zip(
                        durable.sealed_identities, durable.semantic_seal_digests, strict=True,
                    )):
                        _fail()
                    for identity, seal in zip(durable.sealed_identities,
                                              durable.semantic_seal_digests, strict=True):
                        _one(reader, recovery, index, identity, seal, root)
                    reader.check()
                    after = quarantine._sealed_fingerprints(gate, index, sync=False)
                    if after != before:
                        _fail()
        reader.check()
        return HistoricalSealEvidence(durable.sealed_identities, durable.semantic_seal_digests)
    except HistoricalSealError:
        raise
    except Exception:
        _fail()
