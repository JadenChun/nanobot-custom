"""Publish terminal receipt-transaction evidence after complete offline reproof.

This coordinator only appends the typed terminal manifest phases. It does not
remove the receipt or change the already-applied memory and session targets.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_conversion as conversion
from nanobot.agent import memory_recovery_conversion_apply as application
from nanobot.agent import memory_recovery_quarantine_storage as quarantine
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
    receipt_snapshot: storage.MemoryTargetSnapshot | quarantine.ValidatedQuarantineCopy,
    workspace_id: str,
    records: tuple[ManifestRecord, ...],
    *, read_only: bool = False,
) -> _TerminalProof:
    if tuple(record.phase for record in records) not in _PHASES:
        _fail()

    # Reuse the established converted-target, receipt, marker, and blob proof.
    base = offset_stage._proof(  # pyright: ignore[reportPrivateUsage]
        authority, operation, receipt_snapshot, workspace_id, records[:5], read_only=read_only,
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
    if not isinstance(cast(object, offset_recorded), transaction.OffsetAppliedTransactionPayload):
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
        if not isinstance(cast(object, decoded_committed), transaction.ReceiptCommittedTransactionPayload):
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
        if not isinstance(cast(object, decoded_complete), transaction.CompleteTransactionPayload):
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


_PROOF_TOKEN = object()


class TerminalCompleteProof:
    """Operation-scoped, recheckable seven-record terminal semantic evidence."""

    __slots__ = (
        "_sealed",
        "_authority", "_operation", "_checked", "_workspace_id", "_evidence", "_claims",
        "transaction_identity", "complete_digest", "receipt_digest", "receipt_size",
        "receipt_stat", "checked_session_source_digest", "conversion_complete_digest",
        "semantic_seal_digest",
    )

    def __init__(
        self, token: object, authority: storage.WorkspaceMemoryAuthority,
        operation: storage.WorkspaceMemoryOperation, checked: CheckedSessionAuthority,
        workspace_id: str,
        evidence: storage.MemoryTargetSnapshot | quarantine.ValidatedQuarantineCopy,
        claims: bytes,
    ) -> None:
        if token is not _PROOF_TOKEN:
            raise TypeError("terminal proof is issued by semantic verification")
        parsed = json.loads(claims)
        self._authority, self._operation, self._checked = authority, operation, checked
        self._workspace_id, self._evidence, self._claims = workspace_id, evidence, claims
        self.transaction_identity = parsed["transaction_identity"]
        self.complete_digest = parsed["transaction_complete_digest"]
        self.receipt_digest = parsed["receipt_digest"]
        self.receipt_size = parsed["receipt_size"]
        self.receipt_stat = tuple(parsed["receipt_original_stat"])
        self.checked_session_source_digest = parsed["session_source_new_digest"]
        self.conversion_complete_digest = parsed["conversion_complete_digest"]
        self.semantic_seal_digest = _hash(claims)
        self._sealed = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("terminal proof is immutable")
        object.__setattr__(self, name, value)

    @property
    def semantic_seal_claims(self) -> dict[str, object]:
        """A detached copy of the canonical, digest-only versioned seal claims."""
        # The guarded proof constructor retains the immutable canonical JSON
        # object produced by _terminal_seal_claims; every object key is a string.
        return cast(dict[str, object], json.loads(self._claims))

    def is_sealed_for(
        self, authority: storage.WorkspaceMemoryAuthority,
        operation: storage.WorkspaceMemoryOperation,
    ) -> bool:
        """Compare immutable issuance scope without granting or exposing it."""
        return (self._sealed and self._authority is authority
                and self._operation is operation)

    def require_quarantine_commit_match(
        self, authority: storage.WorkspaceMemoryAuthority,
        operation: storage.WorkspaceMemoryOperation, binding: quarantine.QuarantineBinding,
        *, copy_bytes: bytes, observe_copy_stat: Callable[[], tuple[int, ...]],
        original_stat: list[int], digest_claims: Callable[[bytes], str],
    ) -> None:
        """Check owned copy/claims; the store must still perform terminal reproof.

        Delayed observations retain the caller's real stat/digest fault seams.
        This exposes no proof fields and never issues publication authority.
        """
        evidence = self._evidence
        try:
            if (type(evidence) is not quarantine.ValidatedQuarantineCopy
                    or not quarantine.ValidatedQuarantineCopy.is_entered_for(
                        evidence, authority, operation, binding,
                    )
                    or evidence.phase != "copy-ready" or evidence.raw_bytes != copy_bytes
                    or evidence.copy_stat != observe_copy_stat()
                    or self.transaction_identity != binding.transaction_identity
                    or self.complete_digest != binding.complete_digest
                    or self.receipt_digest != binding.receipt_digest
                    or self.receipt_size != binding.receipt_size
                    or tuple(self.receipt_stat) != tuple(original_stat)
                    or self.semantic_seal_digest != digest_claims(self._claims)):
                raise quarantine.QuarantineError() from None
        except (AttributeError, OSError, TypeError, ValueError):
            raise quarantine.QuarantineError() from None

    def require_current(self) -> None:
        try:
            if _terminal_seal_claims(
                self._authority, self._operation, self._checked,
                self._workspace_id, self._evidence,
            ) != self._claims:
                _fail()
        except MemoryRecoveryTransactionTerminalError:
            raise
        except Exception:
            _fail()


def _terminal_seal_claims(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    checked_session: CheckedSessionAuthority,
    workspace_id: str,
    evidence: storage.MemoryTargetSnapshot | quarantine.ValidatedQuarantineCopy,
) -> bytes:
    if type(evidence) not in (storage.MemoryTargetSnapshot, quarantine.ValidatedQuarantineCopy):
        _fail()
    application.require_conversion_receipt_binding(authority, evidence)
    raw = application.receipt_evidence_bytes(evidence)
    receipt = receipt_codec.decode_memory_receipt_bytes(raw)
    captured = transaction.capture_transaction_payload(
        receipt, workspace_id=workspace_id, workspace_identity=operation.workspace_identity,
    )
    candidate = checked_session._recovery_candidate  # pyright: ignore[reportPrivateUsage]
    if (
        candidate is None or candidate.workspace_path != authority.workspace
        or candidate.workspace_id != workspace_id or candidate.session_key != receipt.session_key
    ):
        _fail()
    if not (
        authority.workspace / "memory" / ".recovery" / "transactions" / captured.identity
    ).is_dir():
        _fail()
    manifest = MemoryRecoveryManifest(authority.open_evidence_namespace("transactions", captured.identity))
    records = manifest.load()
    if len(records) != 7 or tuple(record.phase for record in records) != _PHASES[2]:
        _fail()
    proof = _decode_proof(authority, operation, evidence, workspace_id, records, read_only=True)
    if proof.complete is None:
        _fail()
    _verify_new_source(checked_session, proof)
    application.require_conversion_receipt_binding(authority, evidence)
    conversion_identity = conversion.capture_receipt_tail_conversion_payload(captured.identity).identity
    conversion_manifest = MemoryRecoveryManifest(
        authority.open_evidence_namespace("conversions", conversion_identity)
    )
    conversion_records = conversion_manifest.load()
    if len(conversion_records) != 6 or conversion_records[-1].phase != "COMPLETE":
        _fail()
    converted = preparation.require_transaction_conversion_evidence(
        authority, captured, records[0], evidence, receipt, operation, read_only=True,
    )
    if converted.backup is None or converted.source.digest is None:
        _fail()
    _same_chain(conversion_manifest, conversion_records)
    if type(evidence) is quarantine.ValidatedQuarantineCopy:
        binding = evidence.binding
        if (
            binding.transaction_identity != captured.identity
            or binding.complete_digest != records[-1].digest
            or binding.receipt_digest != captured.receipt_digest
            or binding.receipt_size != captured.receipt_size
            or binding.workspace_id != workspace_id
            or binding.workspace != candidate.workspace_path
            or binding.sessions_root != candidate.sessions_root
        ):
            _fail()
    stat_claim = application.receipt_evidence_stat(evidence)
    if any(
        descriptor.new.exists and descriptor.new.blob_digest is None
        for descriptor in proof.prepared.targets.values()
    ):
        _fail()
    keys = ("device", "inode", "mode", "uid", "gid", "links", "size", "mtime_ns", "ctime_ns")
    claims = {
        "schema_version": 1,
        "transaction_identity": captured.identity,
        "transaction_complete_digest": records[-1].digest,
        "conversion_complete_digest": conversion_records[-1].digest,
        "receipt_digest": captured.receipt_digest,
        "receipt_size": captured.receipt_size,
        "receipt_original_stat": [stat_claim[key] for key in keys],
        "prepared_target_new_digests": {
            name: proof.prepared.targets[name].new.blob_digest
            for name in ("history.jsonl", ".cursor", "MEMORY.md")
        },
        "session_source_new_digest": proof.prepared.source.new_digest,
        "archive_backup_digest": converted.source.digest,
        "archive_backup_name": converted.backup.basename,
    }
    _same_chain(manifest, records)
    application.require_conversion_receipt_binding(authority, evidence)
    checked_session.verify_recovery_source_bytes(proof.proposed_source_bytes)
    return (json.dumps(claims, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def prove_terminal_complete_under_locks(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    checked_session: CheckedSessionAuthority,
    workspace_id: str,
    receipt_evidence: storage.MemoryTargetSnapshot | quarantine.ValidatedQuarantineCopy,
) -> TerminalCompleteProof:
    """Read-only semantic proof with an issued live receipt or absent-pending Q1 copy."""
    try:
        claims = _terminal_seal_claims(
            authority, operation, checked_session, workspace_id, receipt_evidence,
        )
        return TerminalCompleteProof(
            _PROOF_TOKEN, authority, operation, checked_session, workspace_id,
            receipt_evidence, claims,
        )
    except MemoryRecoveryTransactionTerminalError:
        raise
    except Exception:
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
                receipt = receipt_codec.decode_memory_receipt_bytes(raw_receipt)
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
    "TerminalCompleteProof",
    "prove_terminal_complete_under_locks",
    "reconcile_memory_recovery_transaction_terminal",
]
