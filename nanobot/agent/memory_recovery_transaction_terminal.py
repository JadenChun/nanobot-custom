"""Publish terminal receipt-transaction evidence after complete offline reproof.

This coordinator only appends the typed terminal manifest phases. It does not
remove the receipt or change the already-applied memory and session targets.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent import memory_recovery_transaction_offset as offset_stage
from nanobot.agent import memory_recovery_transaction_prepare as preparation
from nanobot.agent.memory_recovery_manifest import ManifestRecord, MemoryRecoveryManifest
from nanobot.session.checked_source import CheckedSessionAuthority, open_checked_recovery_session
from nanobot.session.recovery_locator import locate_recovery_candidate

_RECEIPT = ".consolidation.pending.json"
_PREFIX = (
    "CAPTURED",
    "PREPARED",
    "HISTORY_APPLIED",
    "MEMORY_APPLIED",
    "OFFSET_APPLIED",
)
_PHASES = (
    _PREFIX,
    (*_PREFIX, "RECEIPT_COMMITTED"),
    (*_PREFIX, "RECEIPT_COMMITTED", "COMPLETE"),
)


class MemoryRecoveryTransactionTerminalError(RuntimeError):
    """A fixed, content-free failure while proving or publishing terminal state."""

    def __init__(self) -> None:
        super().__init__("memory recovery transaction terminal proof is unavailable")


def _fail() -> NoReturn:
    raise MemoryRecoveryTransactionTerminalError() from None


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True, slots=True)
class _TerminalProof:
    receipt: receipt_codec.MemoryReceipt
    captured: transaction.CapturedTransactionPayload
    prepared: transaction.PreparedTransactionPayload
    history_applied: transaction.HistoryAppliedTransactionPayload
    memory_applied: transaction.MemoryAppliedTransactionPayload
    offset_applied: transaction.OffsetAppliedTransactionPayload
    receipt_committed: transaction.ReceiptCommittedTransactionPayload | None
    complete: transaction.CompleteTransactionPayload | None
    proposed_source_bytes: bytes


def _check_record_chain(
    records: tuple[ManifestRecord, ...],
    captured: transaction.CapturedTransactionPayload,
) -> None:
    phases = tuple(record.phase for record in records)
    if phases not in _PHASES:
        _fail()
    predecessor: str | None = None
    for index, record in enumerate(records):
        if (
            record.kind != "transactions"
            or record.identity != captured.identity
            or record.sequence != index
            or record.predecessor_digest != predecessor
            or _hash(record.raw_bytes) != record.digest
        ):
            _fail()
        predecessor = record.digest


def _decode_proof(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    receipt_snapshot: storage.MemoryTargetSnapshot,
    workspace_id: str,
    records: tuple[ManifestRecord, ...],
) -> _TerminalProof:
    if tuple(record.phase for record in records) not in _PHASES:
        _fail()

    # Reuse the established converted-target, receipt, marker, and blob proof.
    base = offset_stage._proof(  # pyright: ignore[reportPrivateUsage]
        authority, operation, receipt_snapshot, workspace_id, records[:5],
    )
    receipt, captured, prepared, history_applied, memory_applied, proposed_source_bytes = base
    _check_record_chain(records, captured)

    offset_recorded = transaction.decode_transaction_payload(
        "OFFSET_APPLIED",
        records[4].payload,
        captured=captured,
        prepared=prepared,
        history_applied=history_applied,
        memory_applied=memory_applied,
    )
    if not isinstance(offset_recorded, transaction.OffsetAppliedTransactionPayload):
        _fail()
    offset_applied = transaction.offset_applied_transaction_payload(
        captured,
        prepared,
        history_applied,
        memory_applied,
    )
    if offset_recorded != offset_applied:
        _fail()

    receipt_committed: transaction.ReceiptCommittedTransactionPayload | None = None
    complete: transaction.CompleteTransactionPayload | None = None
    if len(records) >= 6:
        committed_record = records[5]
        decoded_committed = transaction.decode_transaction_payload(
            "RECEIPT_COMMITTED",
            committed_record.payload,
            captured=captured,
            prepared=prepared,
            history_applied=history_applied,
            memory_applied=memory_applied,
            offset_applied=offset_applied,
            prepared_record_digest=records[1].digest,
            history_applied_record_digest=records[2].digest,
            memory_applied_record_digest=records[3].digest,
            offset_applied_record_digest=records[4].digest,
        )
        if not isinstance(decoded_committed, transaction.ReceiptCommittedTransactionPayload):
            _fail()
        expected_committed = transaction.receipt_committed_transaction_payload(
            captured,
            prepared,
            history_applied,
            memory_applied,
            offset_applied,
            prepared_record_digest=records[1].digest,
            history_applied_record_digest=records[2].digest,
            memory_applied_record_digest=records[3].digest,
            offset_applied_record_digest=records[4].digest,
        )
        if decoded_committed != expected_committed:
            _fail()
        receipt_committed = decoded_committed

    if len(records) == 7:
        if receipt_committed is None:
            _fail()
        decoded_complete = transaction.decode_transaction_payload(
            "COMPLETE",
            records[6].payload,
            captured=captured,
            prepared=prepared,
            history_applied=history_applied,
            memory_applied=memory_applied,
            offset_applied=offset_applied,
            receipt_committed=receipt_committed,
            commit_record_digest=records[5].digest,
            prepared_record_digest=records[1].digest,
            history_applied_record_digest=records[2].digest,
            memory_applied_record_digest=records[3].digest,
            offset_applied_record_digest=records[4].digest,
        )
        if not isinstance(decoded_complete, transaction.CompleteTransactionPayload):
            _fail()
        expected_complete = transaction.complete_transaction_payload(
            captured,
            prepared,
            history_applied,
            memory_applied,
            offset_applied,
            receipt_committed,
            commit_record_digest=records[5].digest,
            prepared_record_digest=records[1].digest,
            history_applied_record_digest=records[2].digest,
            memory_applied_record_digest=records[3].digest,
            offset_applied_record_digest=records[4].digest,
        )
        if decoded_complete != expected_complete:
            _fail()
        complete = decoded_complete

    return _TerminalProof(
        receipt,
        captured,
        prepared,
        history_applied,
        memory_applied,
        offset_applied,
        receipt_committed,
        complete,
        proposed_source_bytes,
    )


def _verify_new_source(
    checked_session: CheckedSessionAuthority,
    proof: _TerminalProof,
) -> None:
    """Bind terminal evidence to the exact, durably verified new session bytes."""

    source = proof.prepared.source
    raw = checked_session.read_source_bytes()
    if offset_stage._source_state(  # pyright: ignore[reportPrivateUsage]
        raw, proof.receipt, source, proof.proposed_source_bytes,
    ) != "new":
        _fail()
    if (
        len(raw) != source.new_size
        or _hash(raw) != source.new_digest
        or raw != proof.proposed_source_bytes
    ):
        _fail()

    header, events = preparation._session_records(  # pyright: ignore[reportPrivateUsage]
        raw, proof.receipt.session_key or "",
    )
    start = proof.captured.start_offset
    end = proof.captured.end_offset
    if (
        type(header.get("generation")) is not int
        or type(header.get("revision")) is not int
        or type(header.get("last_archived")) is not int
        or type(header.get("last_consolidated")) is not int
        or header["generation"] != source.new_generation
        or header["revision"] != source.new_revision
        or header["last_archived"] != source.new_cursor
        or header["last_consolidated"] != source.new_cursor
        or source.new_generation != proof.captured.session_generation
        or source.old_cursor != start
        or source.new_cursor != end
        or not start < end <= len(events)
        or preparation._legacy_chunk_digest(  # pyright: ignore[reportPrivateUsage]
            events[start:end],
        ) != proof.receipt.chunk_digest
    ):
        _fail()
    checked_session.verify_recovery_source_bytes(proof.proposed_source_bytes)


def _same_receipt_snapshot(
    scope: storage.BoundMemoryTarget,
    issued: storage.MemoryTargetSnapshot,
) -> None:
    current = scope.snapshot_target()
    if (
        current.name != issued.name
        or current.content != issued.content
        or current.stat_identity != issued.stat_identity
    ):
        _fail()


def _verify_receipt_current(
    authority: storage.WorkspaceMemoryAuthority,
    scope: storage.BoundMemoryTarget,
    issued: storage.MemoryTargetSnapshot,
) -> None:
    current = authority.snapshot_target(_RECEIPT)
    if (
        current.content != issued.content
        or current.stat_identity != issued.stat_identity
    ):
        _fail()
    _same_receipt_snapshot(scope, issued)


def _prove_current(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    receipt_scope: storage.BoundMemoryTarget,
    receipt_snapshot: storage.MemoryTargetSnapshot,
    workspace_id: str,
    records: tuple[ManifestRecord, ...],
    checked_session: CheckedSessionAuthority,
) -> _TerminalProof:
    proof = _decode_proof(
        authority,
        operation,
        receipt_snapshot,
        workspace_id,
        records,
    )
    _verify_new_source(checked_session, proof)
    _verify_receipt_current(authority, receipt_scope, receipt_snapshot)
    return proof


def _same_chain(
    manifest: MemoryRecoveryManifest,
    expected: tuple[ManifestRecord, ...],
) -> None:
    if manifest.load() != expected:
        _fail()


def reconcile_memory_recovery_transaction_terminal(
    workspace: Path | str,
    *,
    workspace_id: str,
    sessions_root: Path | str,
) -> ManifestRecord:
    """Append/replay RECEIPT_COMMITTED and COMPLETE without applying live state."""

    try:
        root = Path(sessions_root)
        if not root.is_absolute():
            _fail()
        with (
            storage.open_workspace_memory(workspace) as authority,
            authority.operation() as operation,
        ):
            receipt_snapshot = authority.snapshot_target(_RECEIPT)
            raw_receipt = receipt_snapshot.content
            if raw_receipt is None:
                _fail()
            with operation.borrow_pending_receipt_parent(receipt_snapshot) as receipt_scope:
                _same_receipt_snapshot(receipt_scope, receipt_snapshot)
                receipt = receipt_codec._validate_payload(  # pyright: ignore[reportPrivateUsage]
                    receipt_codec._decode(raw_receipt), raw_receipt,  # pyright: ignore[reportPrivateUsage]
                )
                captured = transaction.capture_transaction_payload(
                    receipt,
                    workspace_id=workspace_id,
                    workspace_identity=operation.workspace_identity,
                )
                manifest = MemoryRecoveryManifest(
                    authority.open_evidence_namespace("transactions", captured.identity)
                )
                candidate = locate_recovery_candidate(
                    authority.workspace,
                    root,
                    receipt.session_key or "",
                )
                with open_checked_recovery_session(operation, candidate) as checked_session:
                    records = manifest.load()
                    proof = _prove_current(
                        authority,
                        operation,
                        receipt_scope,
                        receipt_snapshot,
                        workspace_id,
                        records,
                        checked_session,
                    )

                    if len(records) == 5:
                        committed = transaction.receipt_committed_transaction_payload(
                            proof.captured,
                            proof.prepared,
                            proof.history_applied,
                            proof.memory_applied,
                            proof.offset_applied,
                            prepared_record_digest=records[1].digest,
                            history_applied_record_digest=records[2].digest,
                            memory_applied_record_digest=records[3].digest,
                            offset_applied_record_digest=records[4].digest,
                        )
                        _same_chain(manifest, records)
                        _verify_receipt_current(authority, receipt_scope, receipt_snapshot)
                        committed_record = manifest.append(
                            predecessor_digest=records[4].digest,
                            phase="RECEIPT_COMMITTED",
                            payload=committed.to_payload(),
                        )
                        refreshed = manifest.load()
                        if (
                            len(refreshed) != 6
                            or refreshed[:5] != records
                            or refreshed[5] != committed_record
                        ):
                            _fail()
                        records = refreshed
                        proof = _prove_current(
                            authority,
                            operation,
                            receipt_scope,
                            receipt_snapshot,
                            workspace_id,
                            records,
                            checked_session,
                        )

                    if len(records) == 6:
                        if proof.receipt_committed is None:
                            _fail()
                        # Reprove every live and durable anchor immediately before
                        # COMPLETE, including the just-published commit record.
                        current = manifest.load()
                        if current != records:
                            _fail()
                        proof = _prove_current(
                            authority,
                            operation,
                            receipt_scope,
                            receipt_snapshot,
                            workspace_id,
                            current,
                            checked_session,
                        )
                        committed = proof.receipt_committed
                        if committed is None:
                            _fail()
                        commit_record = current[5]
                        complete = transaction.complete_transaction_payload(
                            proof.captured,
                            proof.prepared,
                            proof.history_applied,
                            proof.memory_applied,
                            proof.offset_applied,
                            committed,
                            commit_record_digest=commit_record.digest,
                            prepared_record_digest=current[1].digest,
                            history_applied_record_digest=current[2].digest,
                            memory_applied_record_digest=current[3].digest,
                            offset_applied_record_digest=current[4].digest,
                        )
                        _same_chain(manifest, current)
                        _verify_receipt_current(authority, receipt_scope, receipt_snapshot)
                        complete_record = manifest.append(
                            predecessor_digest=commit_record.digest,
                            phase="COMPLETE",
                            payload=complete.to_payload(),
                        )
                        refreshed = manifest.load()
                        if (
                            len(refreshed) != 7
                            or refreshed[:6] != current
                            or refreshed[6] != complete_record
                        ):
                            _fail()
                        _prove_current(
                            authority,
                            operation,
                            receipt_scope,
                            receipt_snapshot,
                            workspace_id,
                            refreshed,
                            checked_session,
                        )
                        return refreshed[6]

                    if len(records) == 7 and proof.complete is not None:
                        return records[6]
                    _fail()
    except MemoryRecoveryTransactionTerminalError:
        raise
    except Exception:
        _fail()
    _fail()


__all__ = [
    "MemoryRecoveryTransactionTerminalError",
    "reconcile_memory_recovery_transaction_terminal",
]
