"""Offline, replayable application of a prepared session cursor proposal."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NoReturn

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_target_verify as target_verify
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent import memory_recovery_transaction_history as history
from nanobot.agent import memory_recovery_transaction_memory as memory_stage
from nanobot.agent import memory_recovery_transaction_prepare as preparation
from nanobot.agent.memory_recovery_manifest import ManifestRecord, MemoryRecoveryManifest
from nanobot.session.checked_source import (
    open_checked_recovery_session,
    open_checked_recovery_session_for_replace,
)
from nanobot.session.recovery_locator import locate_recovery_candidate

_PREFIX = ("CAPTURED", "PREPARED", "HISTORY_APPLIED", "MEMORY_APPLIED")


class MemoryRecoveryTransactionOffsetError(RuntimeError):
    """A content-free failure to prove or publish the cursor advance."""

    def __init__(self) -> None:
        super().__init__("memory recovery transaction offset proof is unavailable")


def _fail() -> NoReturn:
    raise MemoryRecoveryTransactionOffsetError() from None


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_state(raw: bytes, receipt: receipt_codec.MemoryReceipt, source: transaction.SourceTransition,
                  proposed: bytes) -> str:
    if len(raw) == source.old_size and _hash(raw) == source.old_digest:
        payload, derived = preparation._source_proposal(raw, receipt)
        if payload != source.to_payload() or derived != proposed:
            _fail()
        return "old"
    if len(raw) != source.new_size or _hash(raw) != source.new_digest or raw != proposed:
        _fail()
    header, events = preparation._session_records(raw, receipt.session_key or "")
    if (
        type(header["generation"]) is not int
        or type(header["revision"]) is not int
        or type(header["last_archived"]) is not int
        or header["generation"] != source.new_generation
        or header["revision"] != source.new_revision
        or header["last_archived"] != source.new_cursor
        or source.new_generation != source.old_generation
        or source.new_revision != source.old_revision + 1
        or source.old_cursor != receipt.start_offset
        or source.new_cursor != receipt.end_offset
        or not source.old_cursor < source.new_cursor <= len(events)
        or preparation._legacy_chunk_digest(events[source.old_cursor:source.new_cursor])
        != receipt.chunk_digest
    ):
        _fail()
    return "new"


def _proof(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    receipt_snapshot: storage.MemoryTargetSnapshot,
    workspace_id: str,
    records: tuple[ManifestRecord, ...],
) -> tuple[receipt_codec.MemoryReceipt, transaction.CapturedTransactionPayload,
           transaction.PreparedTransactionPayload, transaction.HistoryAppliedTransactionPayload,
           transaction.MemoryAppliedTransactionPayload, bytes]:
    if tuple(record.phase for record in records) not in {_PREFIX, (*_PREFIX, "OFFSET_APPLIED")}:
        _fail()
    raw = receipt_snapshot.content
    if raw is None:
        _fail()
    receipt = receipt_codec._validate_payload(receipt_codec._decode(raw), raw)
    captured = transaction.capture_transaction_payload(
        receipt, workspace_id=workspace_id, workspace_identity=operation.workspace_identity,
    )
    if transaction.decode_transaction_payload("CAPTURED", records[0].payload) != captured:
        _fail()
    prepared = transaction.decode_transaction_payload(
        "PREPARED", records[1].payload, captured=captured,
    )
    if preparation._receipt_stat(receipt_snapshot) != prepared.receipt_stat.to_payload():
        _fail()
    conversion = preparation._conversion_evidence(
        authority, captured, records[0], receipt_snapshot, receipt, operation,
    )
    if conversion.disposition != "convert" or conversion.outputs is None or conversion.canonical_row is None:
        _fail()
    applied_history = transaction.decode_transaction_payload(
        "HISTORY_APPLIED", records[2].payload, captured=captured, prepared=prepared,
    )
    if applied_history != transaction.history_applied_transaction_payload(captured, prepared):
        _fail()
    applied_memory = transaction.decode_transaction_payload(
        "MEMORY_APPLIED", records[3].payload, captured=captured, prepared=prepared,
        history_applied=applied_history,
    )
    if applied_memory != transaction.memory_applied_transaction_payload(captured, prepared, applied_history):
        _fail()
    snapshots = {name: authority.snapshot_target(name) for name in ("history.jsonl", ".cursor", "MEMORY.md")}
    history_bytes = snapshots["history.jsonl"].content
    cursor_bytes = snapshots[".cursor"].content
    if history_bytes is None or cursor_bytes is None:
        _fail()
    marker = preparation._verified_marker_row(
        authority, conversion, receipt, history_bytes, cursor_bytes,
    )
    if _hash(marker) != prepared.history_row_digest:
        _fail()
    for name, data in (("history.jsonl", history_bytes), (".cursor", cursor_bytes)):
        history._check_new_target_blob(authority, prepared.targets[name].new, data)
        if not memory_stage._same_snapshot(
            snapshots[name], target_verify.durably_verify_target(operation, snapshots[name], data)
        ):
            _fail()
    memory_snapshot = snapshots["MEMORY.md"]
    transition = prepared.targets["MEMORY.md"]
    expected_memory = (
        receipt.memory_output.encode("utf-8") if prepared.memory_action == "replace"
        else memory_snapshot.content
    )
    if prepared.memory_action == "replace":
        if receipt.memory_output is None or receipt.reference_only:
            _fail()
        if not (
            memory_stage._authorized_new_snapshot(memory_snapshot, transition.new, expected_memory, transition.old)
            or (memory_stage._snapshot_matches_old(memory_snapshot, transition.old)
                and memory_snapshot.content == expected_memory)
        ):
            _fail()
    elif prepared.memory_action == "preserve":
        if not receipt.reference_only or not memory_stage._snapshot_matches_old(memory_snapshot, transition.old):
            _fail()
    else:
        _fail()
    memory_stage._check_memory_blob(authority, transition.new, expected_memory)
    if not memory_stage._same_snapshot(
        memory_snapshot, memory_stage._durably_verify_memory(operation, memory_snapshot, expected_memory)
    ):
        _fail()
    source = prepared.source
    proposed = authority.read_blob(source.new_digest)
    if proposed is None or len(proposed) != source.new_size or _hash(proposed) != source.new_digest:
        _fail()
    preparation._receipt_bound(authority, receipt_snapshot)
    return receipt, captured, prepared, applied_history, applied_memory, proposed


def reconcile_memory_recovery_transaction_offset(
    workspace: Path | str, *, workspace_id: str, sessions_root: Path | str,
) -> ManifestRecord:
    """Durably advance the checked source, then publish exactly one OFFSET_APPLIED."""
    try:
        root = Path(sessions_root)
        if not root.is_absolute():
            _fail()
        with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
            receipt_snapshot = authority.snapshot_target(".consolidation.pending.json")
            if receipt_snapshot.content is None:
                _fail()
            with operation.borrow_pending_receipt_parent(receipt_snapshot) as scope:
                if not memory_stage._same_snapshot(receipt_snapshot, scope.snapshot_target()):
                    _fail()
            raw = receipt_snapshot.content
            receipt = receipt_codec._validate_payload(receipt_codec._decode(raw), raw)
            captured = transaction.capture_transaction_payload(
                receipt, workspace_id=workspace_id, workspace_identity=operation.workspace_identity,
            )
            manifest = MemoryRecoveryManifest(authority.open_evidence_namespace("transactions", captured.identity))
            records = manifest.load()
            receipt, captured, prepared, applied_history, applied_memory, proposed = _proof(
                authority, operation, receipt_snapshot, workspace_id, records,
            )
            source = prepared.source
            candidate = locate_recovery_candidate(authority.workspace, root, receipt.session_key or "")
            # The first authority is read-only. Replacement requires its own fresh, one-use authority.
            with open_checked_recovery_session(operation, candidate) as checked:
                state = _source_state(checked.read_source_bytes(), receipt, source, proposed)
            if len(records) == 5 and state != "new":
                _fail()
            if state == "old":
                candidate = locate_recovery_candidate(authority.workspace, root, receipt.session_key or "")
                with open_checked_recovery_session_for_replace(operation, candidate) as checked:
                    if _source_state(checked.read_source_bytes(), receipt, source, proposed) != "old":
                        _fail()
                    if manifest.load() != records or _proof(
                        authority, operation, receipt_snapshot, workspace_id, records,
                    )[-1] != proposed:
                        _fail()
                    # A failure may be after rename: never reuse this consumed authority.
                    try:
                        checked.replace_recovery_source_bytes(proposed)
                    except Exception:
                        pass
            # Fresh locator and authority classify old/new/divergent after any rename uncertainty.
            candidate = locate_recovery_candidate(authority.workspace, root, receipt.session_key or "")
            with open_checked_recovery_session(operation, candidate) as checked:
                if _source_state(checked.read_source_bytes(), receipt, source, proposed) != "new":
                    _fail()
                checked.verify_recovery_source_bytes(proposed)
                if _proof(authority, operation, receipt_snapshot, workspace_id, manifest.load())[-1] != proposed:
                    _fail()
                preparation._receipt_bound(authority, receipt_snapshot)
                if len(records) == 5:
                    current = manifest.load()
                    if current != records:
                        _fail()
                    decoded = transaction.decode_transaction_payload(
                        "OFFSET_APPLIED", current[-1].payload, captured=captured, prepared=prepared,
                        history_applied=applied_history, memory_applied=applied_memory,
                    )
                    if decoded != transaction.offset_applied_transaction_payload(
                        captured, prepared, applied_history, applied_memory,
                    ):
                        _fail()
                    return current[-1]
                if manifest.load() != records:
                    _fail()
                payload = transaction.offset_applied_transaction_payload(
                    captured, prepared, applied_history, applied_memory,
                )
                record = manifest.append(
                    predecessor_digest=records[-1].digest, phase="OFFSET_APPLIED",
                    payload=payload.to_payload(),
                )
                refreshed = manifest.load()
                if len(refreshed) != 5 or refreshed[:4] != records or refreshed[-1] != record:
                    _fail()
                if transaction.decode_transaction_payload(
                    "OFFSET_APPLIED", refreshed[-1].payload, captured=captured, prepared=prepared,
                    history_applied=applied_history, memory_applied=applied_memory,
                ) != payload:
                    _fail()
                _proof(authority, operation, receipt_snapshot, workspace_id, refreshed)
                checked.verify_recovery_source_bytes(proposed)
                return refreshed[-1]
    except MemoryRecoveryTransactionOffsetError:
        raise
    except Exception:
        _fail()


__all__ = ["MemoryRecoveryTransactionOffsetError", "reconcile_memory_recovery_transaction_offset"]
