"""Prove and record the already-applied history stage of a receipt transaction.

The v2 conversion has already installed the marker-bearing history row and
cursor. This stage re-verifies that durable state and the immutable PREPARED
proposal, then appends HISTORY_APPLIED evidence without changing live targets.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_target_verify as target_verify
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent import memory_recovery_transaction_prepare as preparation
from nanobot.agent.memory_recovery_manifest import ManifestRecord, MemoryRecoveryManifest
from nanobot.session.checked_source import (
    CheckedSessionAuthority,
    open_checked_recovery_session,
)
from nanobot.session.recovery_locator import locate_recovery_candidate

_RECEIPT = ".consolidation.pending.json"
_TARGETS = ("history.jsonl", ".cursor", "MEMORY.md")
_PREFIX = ("CAPTURED", "PREPARED")
_HISTORY_PHASES = (*_PREFIX, "HISTORY_APPLIED")


class MemoryRecoveryTransactionHistoryError(RuntimeError):
    """Fixed, content-free failure while proving applied history state."""

    def __init__(self) -> None:
        super().__init__("memory recovery transaction history proof is unavailable")


def _fail() -> NoReturn:
    raise MemoryRecoveryTransactionHistoryError() from None


def _transaction_manifest(
    authority: storage.WorkspaceMemoryAuthority,
    captured: transaction.CapturedTransactionPayload,
) -> MemoryRecoveryManifest:
    return MemoryRecoveryManifest(
        authority.open_evidence_namespace("transactions", captured.identity)
    )


@dataclass(frozen=True, slots=True)
class _HistoryProof:
    captured: transaction.CapturedTransactionPayload
    prepared: transaction.PreparedTransactionPayload
    prepared_record: ManifestRecord
    applied: transaction.HistoryAppliedTransactionPayload
    source_bytes: bytes
    proposed_source_bytes: bytes
    snapshots: dict[str, storage.MemoryTargetSnapshot]
    target_bytes: dict[str, bytes | None]


def _check_new_target_blob(
    authority: storage.WorkspaceMemoryAuthority,
    descriptor: transaction.NewTargetDescriptor,
    expected: bytes | None,
) -> None:
    if expected is None:
        if descriptor.exists or descriptor.blob_digest is not None or descriptor.size != 0:
            _fail()
        return
    if not descriptor.exists or descriptor.blob_digest is None or descriptor.size != len(expected):
        _fail()
    if authority.read_blob(descriptor.blob_digest) != expected:
        _fail()


def _prove_locked(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    receipt_snapshot: storage.MemoryTargetSnapshot,
    workspace_id: str,
    captured: transaction.CapturedTransactionPayload,
    records: tuple[ManifestRecord, ...],
    checked_session: CheckedSessionAuthority,
) -> _HistoryProof:
    if tuple(record.phase for record in records) not in {_PREFIX, _HISTORY_PHASES}:
        _fail()
    raw_receipt = receipt_snapshot.content
    if raw_receipt is None:
        _fail()
    receipt = receipt_codec.decode_memory_receipt_bytes(raw_receipt)
    current_captured = transaction.capture_transaction_payload(
        receipt,
        workspace_id=workspace_id,
        workspace_identity=operation.workspace_identity,
    )
    if current_captured != captured:
        _fail()
    if records[0].phase != "CAPTURED":
        _fail()
    recorded_captured = transaction.decode_transaction_payload(
        "CAPTURED", records[0].payload,
    )
    if recorded_captured != captured:
        _fail()
    prepared = transaction.decode_transaction_payload(
        "PREPARED", records[1].payload, captured=captured,
    )
    if preparation.receipt_snapshot_stat(receipt_snapshot) != prepared.receipt_stat.to_payload():
        _fail()

    conversion_prepared = preparation.require_transaction_conversion_evidence(
        authority,
        captured,
        records[0],
        receipt_snapshot,
        receipt,
        operation,
    )
    if (
        conversion_prepared.disposition != "convert"
        or conversion_prepared.outputs is None
        or conversion_prepared.canonical_row is None
    ):
        _fail()

    source_bytes = checked_session.read_source_bytes()
    source_payload, proposed_source_bytes = preparation.derive_recovery_source_proposal(
        source_bytes, receipt,
    )
    if source_payload != prepared.source.to_payload():
        _fail()
    if authority.read_blob(prepared.source.new_digest) != proposed_source_bytes:
        _fail()

    snapshots = {name: authority.snapshot_target(name) for name in _TARGETS}
    history_bytes = snapshots["history.jsonl"].content
    cursor_bytes = snapshots[".cursor"].content
    memory_bytes = snapshots["MEMORY.md"].content
    if history_bytes is None or cursor_bytes is None:
        _fail()
    marker_row = preparation.read_verified_transaction_marker_row(
        authority,
        conversion_prepared,
        receipt,
        history_bytes,
        cursor_bytes,
    )
    expected_memory = (
        receipt.memory_output.encode("utf-8")
        if receipt.memory_output is not None and not receipt.reference_only
        else memory_bytes
    )
    target_bytes: dict[str, bytes | None] = {
        "history.jsonl": history_bytes,
        ".cursor": cursor_bytes,
        "MEMORY.md": expected_memory,
    }
    target_proposals = {
        name: preparation.transaction_target_descriptor(snapshots[name], target_bytes[name])
        for name in _TARGETS
    }
    reconstructed = transaction.prepare_transaction_payload(
        captured,
        receipt_stat=preparation.receipt_snapshot_stat(receipt_snapshot),
        targets=target_proposals,
        source=source_payload,
        history_marker_digest=captured.history_marker_digest,
        history_entry_digest=captured.history_entry_digest,
        history_row_digest=hashlib.sha256(marker_row).hexdigest(),
        memory_action="preserve" if receipt.reference_only else "replace",
    )
    if reconstructed != prepared:
        _fail()

    for name in _TARGETS:
        _check_new_target_blob(
            authority,
            prepared.targets[name].new,
            target_bytes[name],
        )

    for name, content in (("history.jsonl", history_bytes), (".cursor", cursor_bytes)):
        verified = target_verify.durably_verify_target(
            operation,
            snapshots[name],
            content,
        )
        if (
            verified.content != content
            or verified.stat_identity != snapshots[name].stat_identity
        ):
            _fail()

    proof = _HistoryProof(
        captured=captured,
        prepared=prepared,
        prepared_record=records[1],
        applied=transaction.history_applied_transaction_payload(captured, prepared),
        source_bytes=source_bytes,
        proposed_source_bytes=proposed_source_bytes,
        snapshots=snapshots,
        target_bytes=target_bytes,
    )
    _verify_still_current(
        authority,
        operation,
        receipt_snapshot,
        checked_session,
        proof,
    )
    return proof


def _verify_still_current(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    receipt_snapshot: storage.MemoryTargetSnapshot,
    checked_session: CheckedSessionAuthority,
    proof: _HistoryProof,
) -> None:
    preparation.require_transaction_receipt_binding(authority, receipt_snapshot)
    if checked_session.read_source_bytes() != proof.source_bytes:
        _fail()
    for name in _TARGETS:
        preparation.require_current_transaction_snapshot(authority, proof.snapshots[name])
    for name, content in (("history.jsonl", proof.target_bytes["history.jsonl"]), (".cursor", proof.target_bytes[".cursor"])):
        if content is None:
            _fail()
        verified = target_verify.durably_verify_target(
            operation,
            proof.snapshots[name],
            content,
        )
        if (
            verified.content != content
            or verified.stat_identity != proof.snapshots[name].stat_identity
        ):
            _fail()
    for name in _TARGETS:
        _check_new_target_blob(
            authority,
            proof.prepared.targets[name].new,
            proof.target_bytes[name],
        )
    if authority.read_blob(proof.prepared.source.new_digest) != proof.proposed_source_bytes:
        _fail()



def reconcile_memory_recovery_transaction_history(
    workspace: Path | str,
    *,
    workspace_id: str,
    sessions_root: Path | str,
) -> ManifestRecord:
    """Verify the converted history row and append/replay one HISTORY_APPLIED phase."""
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
            with operation.borrow_pending_receipt_parent(receipt_snapshot) as scope:
                current = scope.snapshot_target()
                if (
                    current.content != receipt_snapshot.content
                    or current.stat_identity != receipt_snapshot.stat_identity
                ):
                    _fail()
            receipt = receipt_codec.decode_memory_receipt_bytes(raw_receipt)
            captured = transaction.capture_transaction_payload(
                receipt,
                workspace_id=workspace_id,
                workspace_identity=operation.workspace_identity,
            )
            manifest = _transaction_manifest(authority, captured)
            records = manifest.load()
            candidate = locate_recovery_candidate(
                authority.workspace, root, receipt.session_key or "",
            )
            with open_checked_recovery_session(operation, candidate) as checked_session:
                proof = _prove_locked(
                    authority,
                    operation,
                    receipt_snapshot,
                    workspace_id,
                    captured,
                    records,
                    checked_session,
                )
                if len(records) == 3:
                    recorded = transaction.decode_transaction_payload(
                        "HISTORY_APPLIED",
                        records[2].payload,
                        captured=proof.captured,
                        prepared=proof.prepared,
                    )
                    if recorded != proof.applied:
                        _fail()
                    refreshed = manifest.load()
                    if len(refreshed) != 3 or refreshed[-1] != records[-1]:
                        _fail()
                    _verify_still_current(
                        authority,
                        operation,
                        receipt_snapshot,
                        checked_session,
                        proof,
                    )
                    return refreshed[-1]

                if len(records) != 2 or records[-1] != proof.prepared_record:
                    _fail()
                record = manifest.append(
                    predecessor_digest=proof.prepared_record.digest,
                    phase="HISTORY_APPLIED",
                    payload=proof.applied.to_payload(),
                )
                refreshed = manifest.load()
                if len(refreshed) != 3 or refreshed[-1] != record:
                    _fail()
                recorded = transaction.decode_transaction_payload(
                    "HISTORY_APPLIED",
                    refreshed[-1].payload,
                    captured=proof.captured,
                    prepared=proof.prepared,
                )
                if recorded != proof.applied:
                    _fail()
                _verify_still_current(
                    authority,
                    operation,
                    receipt_snapshot,
                    checked_session,
                    proof,
                )
                return refreshed[-1]
    except MemoryRecoveryTransactionHistoryError:
        raise
    except Exception:
        _fail()


def require_transaction_target_blob(
    authority: storage.WorkspaceMemoryAuthority,
    descriptor: transaction.NewTargetDescriptor,
    expected: bytes | None,
) -> None:
    """Require the proposed target blob, size and existence binding."""
    return _check_new_target_blob(authority, descriptor, expected)


__all__ = [
    "MemoryRecoveryTransactionHistoryError",
    "reconcile_memory_recovery_transaction_history",
    "require_transaction_target_blob",
]
