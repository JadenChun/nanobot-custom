"""Read-only terminal semantics from live and quarantined receipt evidence."""

from __future__ import annotations

import hashlib
import os
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_conversion as conversion
from nanobot.agent import memory_recovery_legacy_archive as archive
from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent import memory_recovery_transaction_prepare as preparation
from nanobot.agent import memory_recovery_transaction_terminal as terminal
from nanobot.agent.memory_recovery_manifest import MemoryRecoveryManifest
from nanobot.session.checked_source import (
    CheckedSessionAuthorityError,
    open_checked_recovery_session,
)
from nanobot.session.recovery_locator import locate_recovery_candidate
from tests.agent.test_memory_recovery_transaction_prepare import _SESSION_KEY, _WORKSPACE_ID
from tests.agent.test_memory_recovery_transaction_terminal import (
    _reconcile,
    _setup,
    _transaction_chain,
)


def _binding(workspace: Path, sessions: Path, raw: bytes) -> quarantine.QuarantineBinding:
    _, records = _transaction_chain(workspace)
    return quarantine.QuarantineBinding(
        workspace, sessions, _WORKSPACE_ID, records[0].identity, records[-1].digest,
        hashlib.sha256(raw).hexdigest(), len(raw),
    )


@pytest.mark.parametrize('phase', ['copy-ready', 'committed'])
def test_live_and_copy_terminal_proof_match_without_mutation(
    tmp_path: Path, phase: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, sessions, memory, receipt, _ = _setup(tmp_path)
    _reconcile(workspace, sessions)
    raw = receipt.read_bytes()
    binding = _binding(workspace, sessions, raw)
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        candidate = locate_recovery_candidate(workspace, sessions, _SESSION_KEY)
        with open_checked_recovery_session(operation, candidate) as checked:
            snapshot = authority.snapshot_target('.consolidation.pending.json')
            before = sorted((memory / '.recovery').rglob('*'))
            live = terminal.prove_terminal_complete_under_locks(
                authority, operation, checked, _WORKSPACE_ID, snapshot,
            )
            live.require_current()
            assert live.complete_digest == binding.complete_digest
            assert live.receipt_digest == binding.receipt_digest
            assert len(live.semantic_seal_digest) == 64
            assert sorted((memory / '.recovery').rglob('*')) == before
        with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as store:
            store.prepare(raw)
            store.copy_receipt()
            receipt.unlink()
            if phase == 'committed':
                absence = store.prove_durable_absence()
                with (
                    quarantine.open_validated_quarantine_copy(
                        authority, operation, binding,
                        require_phase='copy-ready', require_pending_absent=True,
                    ) as copy,
                    open_checked_recovery_session(operation, candidate) as checked,
                ):
                    proof = terminal.prove_terminal_complete_under_locks(
                        authority, operation, checked, _WORKSPACE_ID, copy,
                    )
                    store.commit(absence, proof)
        def forbidden(*args, **kwargs):
            raise AssertionError('terminal proof attempted archive mutation')
        monkeypatch.setattr(os, 'link', forbidden)
        monkeypatch.setattr(os, 'unlink', forbidden)
        with quarantine.open_validated_quarantine_copy(
            authority, operation, binding, require_phase=phase, require_pending_absent=True,
        ) as copy, open_checked_recovery_session(operation, candidate) as checked:
            proof = terminal.prove_terminal_complete_under_locks(
                authority, operation, checked, _WORKSPACE_ID, copy,
            )
            proof.require_current()
            assert proof.semantic_seal_digest == live.semantic_seal_digest
            assert proof.receipt_stat == live.receipt_stat
            assert not receipt.exists()
            receipt.write_bytes(raw)
            receipt.chmod(0o600)
            with pytest.raises(terminal.MemoryRecoveryTransactionTerminalError):
                proof.require_current()
    # A fresh authority/session pair is required after a later startup.
    monkeypatch.undo()
    receipt.unlink()
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        candidate = locate_recovery_candidate(workspace, sessions, _SESSION_KEY)
        with quarantine.open_validated_quarantine_copy(
            authority, operation, binding, require_phase=phase, require_pending_absent=True,
        ) as copy, open_checked_recovery_session(operation, candidate) as checked:
            assert terminal.prove_terminal_complete_under_locks(
                authority, operation, checked, _WORKSPACE_ID, copy,
            ).semantic_seal_digest == live.semantic_seal_digest


@pytest.mark.parametrize('target', ['history.jsonl', '.cursor', 'MEMORY.md', 'HISTORY.md.bak', 'session'])
def test_live_proof_rechecks_target_session_and_archive_drift(tmp_path: Path, target: str) -> None:
    workspace, sessions, memory, _, source = _setup(tmp_path)
    _reconcile(workspace, sessions)
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        candidate = locate_recovery_candidate(workspace, sessions, _SESSION_KEY)
        exit_error = pytest.raises(CheckedSessionAuthorityError) if target == 'session' else nullcontext()
        with exit_error, open_checked_recovery_session(operation, candidate) as checked:
            snapshot = authority.snapshot_target('.consolidation.pending.json')
            proof = terminal.prove_terminal_complete_under_locks(
                authority, operation, checked, _WORKSPACE_ID, snapshot,
            )
            path = source if target == 'session' else memory / target
            path.write_bytes(path.read_bytes() + b'\n')
            with pytest.raises(
                terminal.MemoryRecoveryTransactionTerminalError,
                match='^memory recovery transaction terminal proof is unavailable$',
            ):
                proof.require_current()


def test_forged_complete_and_structural_marker_alone_do_not_prove_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, sessions, _, _, _ = _setup(tmp_path)
    _reconcile(workspace, sessions)
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        candidate = locate_recovery_candidate(workspace, sessions, _SESSION_KEY)
        with open_checked_recovery_session(operation, candidate) as checked:
            snapshot = authority.snapshot_target('.consolidation.pending.json')
            with pytest.raises(terminal.MemoryRecoveryTransactionTerminalError):
                terminal.prove_terminal_complete_under_locks(
                    authority, operation, checked, _WORKSPACE_ID, True,  # type: ignore[arg-type]
                )
            original = MemoryRecoveryManifest.load
            def forged(self):
                records = original(self)
                if len(records) == 7 and records[-1].phase == 'COMPLETE':
                    return (*records[:-1], replace(records[-1], digest='0' * 64))
                return records
            monkeypatch.setattr(MemoryRecoveryManifest, 'load', forged)
            with pytest.raises(terminal.MemoryRecoveryTransactionTerminalError):
                terminal.prove_terminal_complete_under_locks(
                    authority, operation, checked, _WORKSPACE_ID, snapshot,
                )


@pytest.mark.parametrize('kind', ['transactions', 'conversions'])
def test_missing_evidence_identity_fails_without_creating_directory(
    tmp_path: Path, kind: str,
) -> None:
    workspace, sessions, _, _, _ = _setup(tmp_path)
    _reconcile(workspace, sessions)
    _, records = _transaction_chain(workspace)
    identity = records[0].identity
    if kind == 'conversions':
        identity = conversion.capture_receipt_tail_conversion_payload(identity).identity
    parent = workspace / 'memory' / '.recovery' / kind
    target = parent / identity
    parked = parent / f'{identity}.unavailable'
    assert target.is_dir() and not parked.exists()
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        candidate = locate_recovery_candidate(workspace, sessions, _SESSION_KEY)
        with open_checked_recovery_session(operation, candidate) as checked:
            snapshot = authority.snapshot_target('.consolidation.pending.json')
            target.rename(parked)
            names_before = tuple(sorted(entry.name for entry in parent.iterdir()))
            with pytest.raises(
                terminal.MemoryRecoveryTransactionTerminalError,
                match='^memory recovery transaction terminal proof is unavailable$',
            ):
                terminal.prove_terminal_complete_under_locks(
                    authority, operation, checked, _WORKSPACE_ID, snapshot,
                )
            assert tuple(sorted(entry.name for entry in parent.iterdir())) == names_before
            assert not target.exists() and parked.is_dir()


def test_copy_binding_rejects_wrong_sessions_root_and_workspace(tmp_path: Path) -> None:
    workspace, sessions, _, receipt, _ = _setup(tmp_path)
    _reconcile(workspace, sessions)
    binding = _binding(workspace, sessions, receipt.read_bytes())
    other_root = tmp_path / 'other-sessions'
    other_root.mkdir(mode=0o700)
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as store:
            store.prepare(receipt.read_bytes())
            store.copy_receipt()
            receipt.unlink()
        for bad in (replace(binding, sessions_root=other_root),
                    replace(binding, workspace_id='f' * 32)):
            with pytest.raises(quarantine.QuarantineError):
                with quarantine.open_validated_quarantine_copy(
                    authority, operation, bad, require_phase='copy-ready', require_pending_absent=True,
                ):
                    pass


def test_copy_original_stat_must_match_prepared_and_copy_bytes_stay_fixed(tmp_path: Path) -> None:
    workspace, sessions, memory, receipt, _ = _setup(tmp_path)
    _reconcile(workspace, sessions)
    raw = receipt.read_bytes()
    replacement = memory / 'different-pending-inode'
    replacement.write_bytes(raw)
    replacement.chmod(0o600)
    os.replace(replacement, receipt)
    binding = _binding(workspace, sessions, raw)
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        candidate = locate_recovery_candidate(workspace, sessions, _SESSION_KEY)
        with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as store:
            store.prepare(raw)
            store.copy_receipt()
            receipt.unlink()
        with quarantine.open_validated_quarantine_copy(
            authority, operation, binding, require_phase='copy-ready', require_pending_absent=True,
        ) as copy, open_checked_recovery_session(operation, candidate) as checked:
            with pytest.raises(terminal.MemoryRecoveryTransactionTerminalError):
                terminal.prove_terminal_complete_under_locks(
                    authority, operation, checked, _WORKSPACE_ID, copy,
                )
            private_copy = (
                memory / '.recovery' / 'receipt-quarantine' / binding.transaction_identity / 'receipt.bin'
            )
            private_copy.write_bytes(raw + b'x')
            with pytest.raises(terminal.MemoryRecoveryTransactionTerminalError):
                terminal.prove_terminal_complete_under_locks(
                    authority, operation, checked, _WORKSPACE_ID, copy,
                )


def test_backup_only_verifier_rejects_source_reappearance(tmp_path: Path) -> None:
    workspace, sessions, memory, receipt, _ = _setup(tmp_path)
    _reconcile(workspace, sessions)
    _, records = _transaction_chain(workspace)
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        anchor = authority.snapshot_target('history.jsonl')
        backup = memory / 'HISTORY.md.bak'
        captured = transaction.decode_transaction_payload('CAPTURED', records[0].payload)
        receipt_snapshot = authority.snapshot_target('.consolidation.pending.json')
        prepared = preparation._conversion_evidence(
            authority, captured, records[0], receipt_snapshot,
            receipt_codec._validate_payload(receipt_codec._decode(receipt.read_bytes()), receipt.read_bytes()),
            operation,
        )
        assert prepared.source.stat is not None
        old = prepared.source
        info = old.stat
        stat_id = storage.MemoryTargetStatIdentity(
            info.device, info.inode, info.mode, info.uid, info.gid, info.links,
            info.size, info.mtime_ns, info.ctime_ns,
        )
        archive.verify_legacy_backup_only(
            operation, anchor, expected_source_stat=stat_id,
            expected_digest=old.digest, expected_size=old.size, backup_name=prepared.backup.basename,
        )
        assert backup.exists()
        (memory / 'HISTORY.md').write_bytes(b'forged')
        with pytest.raises(storage.MemoryStorageError):
            archive.verify_legacy_backup_only(
                operation, anchor, expected_source_stat=stat_id,
                expected_digest=old.digest, expected_size=old.size,
                backup_name=prepared.backup.basename,
            )
