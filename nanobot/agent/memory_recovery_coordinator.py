"""Explicit offline reconciliation of an existing pending memory receipt.

This is never a startup hook. Each stage owns its own workspace/checked-session
locks; inspection leases are released before invoking a stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_conversion as conversion
from nanobot.agent import memory_recovery_conversion_apply as conversion_apply
from nanobot.agent import memory_recovery_conversion_finalize as conversion_finalize
from nanobot.agent import memory_recovery_conversion_runtime as conversion_prepare
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent import memory_recovery_transaction_history as history
from nanobot.agent import memory_recovery_transaction_memory as memory
from nanobot.agent import memory_recovery_transaction_offset as offset
from nanobot.agent import memory_recovery_transaction_prepare as transaction_prepare
from nanobot.agent import memory_recovery_transaction_terminal as terminal
from nanobot.agent.memory_recovery_manifest import ManifestRecord, MemoryRecoveryManifest
from nanobot.session.checked_source import open_checked_recovery_session
from nanobot.session.recovery_locator import locate_recovery_candidate

_CONVERSION_PHASES = (
    "CAPTURED", "CONVERSION_PREPARED", "JSONL_APPLIED", "CURSORS_APPLIED",
    "LEGACY_FINALIZED", "COMPLETE",
)
_TRANSACTION_PHASES = (
    "CAPTURED", "PREPARED", "HISTORY_APPLIED", "MEMORY_APPLIED", "OFFSET_APPLIED",
    "RECEIPT_COMMITTED", "COMPLETE",
)


class OfflineMemoryRecoveryError(RuntimeError):
    """A content-free failure requiring operator reconciliation."""

    def __init__(self) -> None:
        super().__init__("memory recovery unavailable")


def _fail() -> NoReturn:
    raise OfflineMemoryRecoveryError() from None


def _inspect(
    workspace: Path, sessions_root: Path, session_key: str, workspace_id: str,
    receipt_bytes: bytes,
) -> tuple[tuple[ManifestRecord, ...], tuple[ManifestRecord, ...]]:
    # No lock survives this function. The stage will re-open, revalidate, and
    # acquire its own workspace-first lock hierarchy.
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        snapshot = authority.snapshot_target(".consolidation.pending.json")
        if snapshot.content != receipt_bytes:
            _fail()
        receipt = receipt_codec.decode_memory_receipt_bytes(receipt_bytes)
        captured = transaction.capture_transaction_payload(
            receipt, workspace_id=workspace_id, workspace_identity=operation.workspace_identity,
        )
        candidate = locate_recovery_candidate(workspace, sessions_root, session_key)
        if candidate.workspace_id != workspace_id or candidate.session_key != session_key:
            _fail()
        with open_checked_recovery_session(operation, candidate) as checked:
            source_bytes = checked.read_source_bytes()
        tx = MemoryRecoveryManifest(
            authority.open_evidence_namespace("transactions", captured.identity)
        ).load()
        v1 = conversion.capture_conversion_payload(captured.identity)
        if MemoryRecoveryManifest(
            authority.open_evidence_namespace("conversions", v1.identity)
        ).load():
            _fail()
        v2 = conversion.capture_receipt_tail_conversion_payload(captured.identity)
        cv = MemoryRecoveryManifest(
            authority.open_evidence_namespace("conversions", v2.identity)
        ).load()
        if tx and transaction.decode_transaction_payload("CAPTURED", tx[0].payload) != captured:
            _fail()
        if cv and conversion.decode_conversion_payload(cv[0].payload, "CAPTURED") != v2:
            _fail()
        if cv and len(cv) >= 2:
            prepared = conversion.decode_conversion_payload(
                cv[1].payload, "CONVERSION_PREPARED", captured=v2,
            )
            if prepared.disposition != "convert":
                _fail()
        # The locator authenticates path/marker identity, not the JSONL header
        # or the pending receipt's event range. Reject a mismatched source
        # before conversion can touch the live history targets.
        transaction_prepare.parse_recovery_session_records(source_bytes, session_key)
        if len(tx) <= 1:
            transaction_prepare.derive_recovery_source_proposal(source_bytes, receipt)
        return tx, cv


def reconcile_pending_memory_receipt_offline(
    workspace: Path | str, *, sessions_root: Path | str,
) -> ManifestRecord:
    """Resume a checked receipt through transaction COMPLETE without removing it.

    This must be called by an offline operator; neither constructor nor startup
    invokes it. An unsupported preserve disposition requires manual handling.
    """
    try:
        root = Path(workspace)
        sessions = Path(sessions_root)
        if not root.is_absolute() or not sessions.is_absolute():
            _fail()
        receipt = receipt_codec.parse_memory_receipt(root / "memory" / ".consolidation.pending.json")
        if receipt is None or not receipt.session_key:
            _fail()
        candidate = locate_recovery_candidate(root, sessions, receipt.session_key)
        workspace_id = candidate.workspace_id

        # Each call makes a durable phase advance (some stages append two
        # records). The finite bound is a guard against unexpected no-progress.
        for _ in range(len(_CONVERSION_PHASES) + len(_TRANSACTION_PHASES) + 1):
            tx, cv = _inspect(root, sessions, receipt.session_key, workspace_id, receipt.raw_bytes)
            tx_phase = tx[-1].phase if tx else None
            cv_phase = cv[-1].phase if cv else None
            if tx_phase not in (None, *_TRANSACTION_PHASES) or cv_phase not in (None, *_CONVERSION_PHASES):
                _fail()
            if tx_phase in (None, "CAPTURED"):
                if cv_phase in (None, "CAPTURED"):
                    conversion_prepare.prepare_receipt_tail_conversion(root, workspace_id=workspace_id)
                elif cv_phase == "CONVERSION_PREPARED" or cv_phase == "JSONL_APPLIED":
                    conversion_apply.apply_prepared_conversion_targets(root, workspace_id=workspace_id)
                elif cv_phase in ("CURSORS_APPLIED", "LEGACY_FINALIZED"):
                    conversion_finalize.finalize_prepared_conversion(root, workspace_id=workspace_id)
                elif cv_phase == "COMPLETE":
                    transaction_prepare.prepare_memory_recovery_transaction(
                        root, workspace_id=workspace_id, sessions_root=sessions,
                    )
                else:
                    _fail()
            elif cv_phase != "COMPLETE":
                _fail()
            elif tx_phase == "PREPARED":
                history.reconcile_memory_recovery_transaction_history(
                    root, workspace_id=workspace_id, sessions_root=sessions,
                )
            elif tx_phase == "HISTORY_APPLIED":
                memory.reconcile_memory_recovery_transaction_memory(
                    root, workspace_id=workspace_id, sessions_root=sessions,
                )
            elif tx_phase == "MEMORY_APPLIED":
                offset.reconcile_memory_recovery_transaction_offset(
                    root, workspace_id=workspace_id, sessions_root=sessions,
                )
            elif tx_phase in ("OFFSET_APPLIED", "RECEIPT_COMMITTED", "COMPLETE"):
                return terminal.reconcile_memory_recovery_transaction_terminal(
                    root, workspace_id=workspace_id, sessions_root=sessions,
                )
            else:
                _fail()
    except OfflineMemoryRecoveryError:
        raise
    except Exception:
        _fail()
    _fail()


__all__ = ["OfflineMemoryRecoveryError", "reconcile_pending_memory_receipt_offline"]
