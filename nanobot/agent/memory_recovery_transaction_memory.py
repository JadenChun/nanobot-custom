"""Apply or prove the MEMORY.md target for one prepared recovery transaction.

This offline coordinator holds the workspace-memory authority before acquiring
checked session locks. It changes only MEMORY.md and publishes one typed
MEMORY_APPLIED manifest phase after the target is durably verified.
"""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_target_mutation as target_mutation
from nanobot.agent import memory_recovery_target_verify as target_verify
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent import memory_recovery_transaction_history as history
from nanobot.agent import memory_recovery_transaction_prepare as preparation
from nanobot.agent.memory_recovery_manifest import ManifestRecord, MemoryRecoveryManifest
from nanobot.session.checked_source import (
    CheckedSessionAuthority,
    open_checked_recovery_session,
)
from nanobot.session.recovery_locator import locate_recovery_candidate

_RECEIPT = ".consolidation.pending.json"
_TARGETS = ("history.jsonl", ".cursor", "MEMORY.md")
_PREFIX = ("CAPTURED", "PREPARED", "HISTORY_APPLIED")
_MEMORY_PHASES = (*_PREFIX, "MEMORY_APPLIED")


class MemoryRecoveryTransactionMemoryError(RuntimeError):
    """Fixed, content-free failure while proving or applying memory state."""

    def __init__(self) -> None:
        super().__init__("memory recovery transaction memory proof is unavailable")


def _fail() -> NoReturn:
    raise MemoryRecoveryTransactionMemoryError() from None


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _same_snapshot(
    expected: storage.MemoryTargetSnapshot,
    current: storage.MemoryTargetSnapshot,
) -> bool:
    return (
        expected.name == current.name
        and expected.content == current.content
        and expected.stat_identity == current.stat_identity
    )


def _receipt_manifest(
    authority: storage.WorkspaceMemoryAuthority,
    captured: transaction.CapturedTransactionPayload,
) -> MemoryRecoveryManifest:
    return MemoryRecoveryManifest(
        authority.open_evidence_namespace("transactions", captured.identity)
    )


def _snapshot_matches_old(
    snapshot: storage.MemoryTargetSnapshot,
    old: transaction.OldTargetDescriptor,
) -> bool:
    if not old.exists:
        return snapshot.content is None and snapshot.stat_identity is None
    info = snapshot.stat_identity
    if snapshot.content is None or info is None or old.stat is None or old.digest is None:
        return False
    return (
        len(snapshot.content) == old.size
        and _hash(snapshot.content) == old.digest
        and info.device == old.stat.device
        and info.inode == old.stat.inode
        and info.mode == old.stat.mode
        and info.uid == old.stat.uid
        and info.gid == old.stat.gid
        and info.links == old.stat.links
        and info.size == old.stat.size
        and info.mtime_ns == old.stat.mtime_ns
        and info.ctime_ns == old.stat.ctime_ns
    )


def _authorized_new_snapshot(
    snapshot: storage.MemoryTargetSnapshot,
    descriptor: transaction.NewTargetDescriptor,
    expected: bytes,
    old: transaction.OldTargetDescriptor,
) -> bool:
    info = snapshot.stat_identity
    if (
        not descriptor.exists
        or descriptor.blob_digest != _hash(expected)
        or descriptor.size != len(expected)
        or snapshot.content != expected
        or info is None
        or info.size != len(expected)
        or info.uid != os.getuid()
        or info.links != 1
        or not stat.S_ISREG(info.mode)
        or stat.S_IMODE(info.mode) != 0o600
    ):
        return False
    if old.exists and old.stat is not None and (
        info.device == old.stat.device and info.inode == old.stat.inode
    ):
        return False
    return True


def _fsync(fd: int) -> None:
    try:
        os.fsync(fd)
    except (OSError, TypeError, ValueError):
        _fail()


def _stat_at(parent_fd: int, name: str) -> storage.MemoryTargetStatIdentity | None:
    info = storage.stat_recovery_entry(parent_fd, name)
    if info is None:
        return None
    storage.require_safe_memory_target(name, info)
    return storage.memory_target_stat_identity(info)


def _durably_verify_memory(
    operation: storage.WorkspaceMemoryOperation,
    snapshot: storage.MemoryTargetSnapshot,
    expected: bytes | None,
) -> storage.MemoryTargetSnapshot:
    """Fsync and recheck MEMORY.md or its stable absence through a bound parent."""

    if (
        type(operation) is not storage.WorkspaceMemoryOperation
        or type(snapshot) is not storage.MemoryTargetSnapshot
        or snapshot.name != "MEMORY.md"
        or snapshot.content != expected
        or (expected is None) != (snapshot.stat_identity is None)
    ):
        _fail()
    parent_fd: int | None = None
    target_fd: int | None = None
    try:
        with operation.borrow_target_parent(snapshot) as scope:
            scope.require_current()
            current = scope.snapshot_target()
            if not _same_snapshot(snapshot, current):
                _fail()
            parent_fd = scope.duplicate_dir_fd()
            if expected is None:
                if _stat_at(parent_fd, "MEMORY.md") is not None:
                    _fail()
                _fsync(parent_fd)
                scope.require_current()
                current = scope.snapshot_target()
                if (
                    not _same_snapshot(snapshot, current)
                    or _stat_at(parent_fd, "MEMORY.md") is not None
                ):
                    _fail()
                _fsync(parent_fd)
                scope.require_current()
                verified = scope.snapshot_target()
                if not _same_snapshot(snapshot, verified):
                    _fail()
                return verified

            identity = snapshot.stat_identity
            if identity is None:
                _fail()
            try:
                target_fd = storage.open_recovery_read_leaf(
                    os.open, "MEMORY.md", parent_fd,
                )
                opened = os.fstat(target_fd)
            except (OSError, TypeError, ValueError):
                _fail()
            storage.require_safe_memory_target("MEMORY.md", opened)
            if (
                storage.memory_target_stat_identity(opened) != identity
                or _stat_at(parent_fd, "MEMORY.md") != identity
            ):
                _fail()
            _fsync(target_fd)
            _fsync(parent_fd)
            scope.require_current()
            current = scope.snapshot_target()
            try:
                opened_after = os.fstat(target_fd)
            except (OSError, TypeError, ValueError):
                _fail()
            storage.require_safe_memory_target("MEMORY.md", opened_after)
            if (
                not _same_snapshot(snapshot, current)
                or storage.memory_target_stat_identity(opened_after) != identity
                or _stat_at(parent_fd, "MEMORY.md") != identity
            ):
                _fail()
            return current
    except MemoryRecoveryTransactionMemoryError:
        raise
    except Exception:
        _fail()
    finally:
        storage.close_recovery_descriptor(target_fd)
        storage.close_recovery_descriptor(parent_fd)


@dataclass(frozen=True, slots=True)
class _MemoryProof:
    captured: transaction.CapturedTransactionPayload
    prepared: transaction.PreparedTransactionPayload
    prepared_record: ManifestRecord
    history_record: ManifestRecord
    history_applied: transaction.HistoryAppliedTransactionPayload
    memory_applied: transaction.MemoryAppliedTransactionPayload
    source_bytes: bytes
    proposed_source_bytes: bytes
    snapshots: dict[str, storage.MemoryTargetSnapshot]
    target_bytes: dict[str, bytes]
    output_memory_bytes: bytes | None
    memory_state: str


def _check_memory_blob(
    authority: storage.WorkspaceMemoryAuthority,
    descriptor: transaction.NewTargetDescriptor,
    expected: bytes | None,
) -> None:
    if expected is None:
        if descriptor.exists or descriptor.blob_digest is not None or descriptor.size != 0:
            _fail()
        return
    # The decoded descriptor is frozen. The preceding digest equality proves
    # a string before read_blob is reached; preserve the second digest read.
    if (
        not descriptor.exists
        or descriptor.blob_digest != _hash(expected)
        or descriptor.size != len(expected)
        or authority.read_blob(cast(str, descriptor.blob_digest)) != expected
    ):
        _fail()


def _proof_locked(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    receipt_snapshot: storage.MemoryTargetSnapshot,
    workspace_id: str,
    captured: transaction.CapturedTransactionPayload,
    records: tuple[ManifestRecord, ...],
    checked_session: CheckedSessionAuthority,
) -> _MemoryProof:
    phases = tuple(record.phase for record in records)
    if phases not in {_PREFIX, _MEMORY_PHASES}:
        _fail()
    raw_receipt = receipt_snapshot.content
    if raw_receipt is None:
        _fail()
    receipt = receipt_codec.decode_memory_receipt_bytes(raw_receipt)
    if transaction.capture_transaction_payload(
        receipt,
        workspace_id=workspace_id,
        workspace_identity=operation.workspace_identity,
    ) != captured:
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

    history_applied = transaction.decode_transaction_payload(
        "HISTORY_APPLIED",
        records[2].payload,
        captured=captured,
        prepared=prepared,
    )
    expected_history_applied = transaction.history_applied_transaction_payload(
        captured, prepared,
    )
    if history_applied != expected_history_applied:
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
    if (
        source_payload != prepared.source.to_payload()
        or authority.read_blob(prepared.source.new_digest) != proposed_source_bytes
    ):
        _fail()

    snapshots = {name: authority.snapshot_target(name) for name in _TARGETS}
    history_bytes = snapshots["history.jsonl"].content
    cursor_bytes = snapshots[".cursor"].content
    if history_bytes is None or cursor_bytes is None:
        _fail()
    marker_row = preparation.read_verified_transaction_marker_row(
        authority,
        conversion_prepared,
        receipt,
        history_bytes,
        cursor_bytes,
    )
    if _hash(marker_row) != prepared.history_row_digest:
        _fail()

    for name, content in (("history.jsonl", history_bytes), (".cursor", cursor_bytes)):
        descriptor = prepared.targets[name].new
        history.require_transaction_target_blob(authority, descriptor, content)
        verified = target_verify.durably_verify_target(operation, snapshots[name], content)
        if not _same_snapshot(snapshots[name], verified):
            _fail()

    memory_snapshot = snapshots["MEMORY.md"]
    memory_transition = prepared.targets["MEMORY.md"]
    if prepared.memory_action == "replace":
        if receipt.memory_output is None or captured.output_memory_digest is None:
            _fail()
        output_memory_bytes = receipt.memory_output.encode("utf-8")
        if (
            _hash(output_memory_bytes) != captured.output_memory_digest
            or len(output_memory_bytes) != captured.output_memory_size
        ):
            _fail()
    elif prepared.memory_action == "preserve":
        output_memory_bytes = memory_snapshot.content
        if not _snapshot_matches_old(memory_snapshot, memory_transition.old):
            _fail()
    else:
        _fail()
    _check_memory_blob(authority, memory_transition.new, output_memory_bytes)

    old_matches = _snapshot_matches_old(memory_snapshot, memory_transition.old)
    if prepared.memory_action == "preserve":
        if not old_matches:
            _fail()
        memory_state = "old"
        memory_bytes: bytes | None = memory_snapshot.content
    elif old_matches:
        memory_state = "old"
        memory_bytes = memory_snapshot.content
    elif output_memory_bytes is not None and _authorized_new_snapshot(
        memory_snapshot,
        memory_transition.new,
        output_memory_bytes,
        memory_transition.old,
    ) and (
        memory_transition.old.exists is False
        or memory_transition.old.digest != _hash(output_memory_bytes)
    ):
        memory_state = "new"
        memory_bytes = output_memory_bytes
    else:
        _fail()

    if len(records) == 4:
        memory_applied = transaction.decode_transaction_payload(
            "MEMORY_APPLIED",
            records[3].payload,
            captured=captured,
            prepared=prepared,
            history_applied=history_applied,
        )
    else:
        memory_applied = transaction.memory_applied_transaction_payload(
            captured, prepared, history_applied,
        )
    if memory_applied != transaction.memory_applied_transaction_payload(
        captured, prepared, history_applied,
    ):
        _fail()

    target_bytes = {"history.jsonl": history_bytes, ".cursor": cursor_bytes}
    proof = _MemoryProof(
        captured=captured,
        prepared=prepared,
        prepared_record=records[1],
        history_record=records[2],
        history_applied=history_applied,
        memory_applied=memory_applied,
        source_bytes=source_bytes,
        proposed_source_bytes=proposed_source_bytes,
        snapshots=snapshots,
        target_bytes=target_bytes,
        output_memory_bytes=output_memory_bytes,
        memory_state=memory_state,
    )
    _verify_still_current(
        authority,
        operation,
        receipt_snapshot,
        checked_session,
        proof,
        expected_memory_snapshot=memory_snapshot,
        expected_memory_bytes=memory_bytes,
    )
    return proof


def _verify_still_current(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    receipt_snapshot: storage.MemoryTargetSnapshot,
    checked_session: CheckedSessionAuthority,
    proof: _MemoryProof,
    *,
    expected_memory_snapshot: storage.MemoryTargetSnapshot,
    expected_memory_bytes: bytes | None,
) -> None:
    preparation.require_transaction_receipt_binding(authority, receipt_snapshot)
    if checked_session.read_source_bytes() != proof.source_bytes:
        _fail()
    for name in ("history.jsonl", ".cursor"):
        expected_snapshot = proof.snapshots[name]
        current = authority.snapshot_target(name)
        content = proof.target_bytes[name]
        if not _same_snapshot(expected_snapshot, current):
            _fail()
        verified = target_verify.durably_verify_target(operation, current, content)
        if not _same_snapshot(current, verified):
            _fail()
        history.require_transaction_target_blob(
            authority, proof.prepared.targets[name].new, content,
        )
    current_memory = authority.snapshot_target("MEMORY.md")
    if not _same_snapshot(expected_memory_snapshot, current_memory):
        _fail()
    if not _same_snapshot(
        current_memory,
        _durably_verify_memory(operation, current_memory, expected_memory_bytes),
    ):
        _fail()
    _check_memory_blob(
        authority,
        proof.prepared.targets["MEMORY.md"].new,
        proof.output_memory_bytes if proof.prepared.memory_action == "replace"
        else expected_memory_bytes,
    )
    if authority.read_blob(proof.prepared.source.new_digest) != proof.proposed_source_bytes:
        _fail()


def reconcile_memory_recovery_transaction_memory(
    workspace: Path | str,
    *,
    workspace_id: str,
    sessions_root: Path | str,
) -> ManifestRecord:
    """Apply or replay the MEMORY_APPLIED phase without touching source state."""

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
                if not _same_snapshot(receipt_snapshot, current):
                    _fail()
            receipt = receipt_codec.decode_memory_receipt_bytes(raw_receipt)
            captured = transaction.capture_transaction_payload(
                receipt,
                workspace_id=workspace_id,
                workspace_identity=operation.workspace_identity,
            )
            manifest = _receipt_manifest(authority, captured)
            records = manifest.load()
            candidate = locate_recovery_candidate(
                authority.workspace, root, receipt.session_key or "",
            )
            with open_checked_recovery_session(operation, candidate) as checked_session:
                proof = _proof_locked(
                    authority,
                    operation,
                    receipt_snapshot,
                    workspace_id,
                    captured,
                    records,
                    checked_session,
                )
                if len(records) == 4:
                    refreshed = manifest.load()
                    if len(refreshed) != 4 or refreshed != records:
                        _fail()
                    _verify_still_current(
                        authority,
                        operation,
                        receipt_snapshot,
                        checked_session,
                        proof,
                        expected_memory_snapshot=proof.snapshots["MEMORY.md"],
                        expected_memory_bytes=proof.output_memory_bytes,
                    )
                    return refreshed[-1]

                if len(records) != 3 or records[-1] != proof.history_record:
                    _fail()
                current_memory = proof.snapshots["MEMORY.md"]
                if proof.memory_state == "old":
                    before_bytes = current_memory.content
                    if (
                        proof.prepared.memory_action == "replace"
                        and proof.output_memory_bytes is not None
                        and before_bytes != proof.output_memory_bytes
                    ):
                        _verify_still_current(
                            authority,
                            operation,
                            receipt_snapshot,
                            checked_session,
                            proof,
                            expected_memory_snapshot=current_memory,
                            expected_memory_bytes=before_bytes,
                        )
                        try:
                            applied_snapshot = target_mutation.replace_target_if_unchanged(
                                operation,
                                current_memory,
                                proof.output_memory_bytes,
                            )
                        except target_mutation.MemoryTargetReplacementUncertainError:
                            applied_snapshot = authority.snapshot_target("MEMORY.md")
                            if _snapshot_matches_old(
                                applied_snapshot,
                                proof.prepared.targets["MEMORY.md"].old,
                            ):
                                _fail()
                            if not _authorized_new_snapshot(
                                applied_snapshot,
                                proof.prepared.targets["MEMORY.md"].new,
                                proof.output_memory_bytes,
                                proof.prepared.targets["MEMORY.md"].old,
                            ):
                                _fail()
                        if not _authorized_new_snapshot(
                            applied_snapshot,
                            proof.prepared.targets["MEMORY.md"].new,
                            proof.output_memory_bytes,
                            proof.prepared.targets["MEMORY.md"].old,
                        ):
                            _fail()
                        current_memory = _durably_verify_memory(
                            operation, applied_snapshot, proof.output_memory_bytes,
                        )
                    else:
                        current_memory = _durably_verify_memory(
                            operation, current_memory, before_bytes,
                        )
                else:
                    current_memory = _durably_verify_memory(
                        operation, current_memory, proof.output_memory_bytes,
                    )

                _verify_still_current(
                    authority,
                    operation,
                    receipt_snapshot,
                    checked_session,
                    proof,
                    expected_memory_snapshot=current_memory,
                    expected_memory_bytes=proof.output_memory_bytes,
                )
                record = manifest.append(
                    predecessor_digest=proof.history_record.digest,
                    phase="MEMORY_APPLIED",
                    payload=proof.memory_applied.to_payload(),
                )
                refreshed = manifest.load()
                if len(refreshed) != 4 or refreshed[:3] != records[:3] or refreshed[-1] != record:
                    _fail()
                decoded = transaction.decode_transaction_payload(
                    "MEMORY_APPLIED",
                    refreshed[-1].payload,
                    captured=proof.captured,
                    prepared=proof.prepared,
                    history_applied=proof.history_applied,
                )
                if decoded != proof.memory_applied:
                    _fail()
                _verify_still_current(
                    authority,
                    operation,
                    receipt_snapshot,
                    checked_session,
                    proof,
                    expected_memory_snapshot=current_memory,
                    expected_memory_bytes=proof.output_memory_bytes,
                )
                return refreshed[-1]
    except MemoryRecoveryTransactionMemoryError:
        raise
    except Exception:
        _fail()


def transaction_snapshots_match(
    expected: storage.MemoryTargetSnapshot,
    current: storage.MemoryTargetSnapshot,
) -> bool:
    """Compare transaction target names, bytes and complete stat identities."""
    return _same_snapshot(expected, current)


def transaction_snapshot_matches_old(
    snapshot: storage.MemoryTargetSnapshot,
    old: transaction.OldTargetDescriptor,
) -> bool:
    """Compare a target with the existing original transaction descriptor."""
    return _snapshot_matches_old(snapshot, old)


def transaction_snapshot_is_authorized_new(
    snapshot: storage.MemoryTargetSnapshot,
    descriptor: transaction.NewTargetDescriptor,
    expected: bytes,
    old: transaction.OldTargetDescriptor,
) -> bool:
    """Check the existing output policy including distinct inode binding."""
    return _authorized_new_snapshot(snapshot, descriptor, expected, old)


def durably_verify_transaction_memory(
    operation: storage.WorkspaceMemoryOperation,
    snapshot: storage.MemoryTargetSnapshot,
    expected: bytes | None,
) -> storage.MemoryTargetSnapshot:
    """Fsync and recheck MEMORY.md through its bound parent."""
    return _durably_verify_memory(operation, snapshot, expected)


def require_transaction_memory_blob(
    authority: storage.WorkspaceMemoryAuthority,
    descriptor: transaction.NewTargetDescriptor,
    expected: bytes | None,
) -> None:
    """Require the memory-specific proposed blob and digest checks."""
    return _check_memory_blob(authority, descriptor, expected)


__all__ = [
    "MemoryRecoveryTransactionMemoryError",
    "reconcile_memory_recovery_transaction_memory",
    "transaction_snapshots_match",
    "transaction_snapshot_matches_old",
    "transaction_snapshot_is_authorized_new",
    "durably_verify_transaction_memory",
    "require_transaction_memory_blob",
]
