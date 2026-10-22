"""Operator-only terminal receipt disposition and targeted crash replay."""

from __future__ import annotations

import multiprocessing
import os
import stat
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_receipt_finalizer as finalizer
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_transaction_terminal as terminal
from nanobot.session.recovery_locator import locate_recovery_candidate
from tests.agent.test_memory_recovery_quarantine_storage import _real_layout
from tests.agent.test_memory_recovery_transaction_prepare import _SESSION_KEY

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor protocol")


def _crash_after_boundary(workspace: Path, sessions: Path, boundary: str) -> None:
    if boundary == "unlink":
        original = os.unlink

        def unlink(name: str, *, dir_fd: int | None = None) -> None:
            original(name, dir_fd=dir_fd)
            if name == ".consolidation.pending.json":
                os._exit(81)

        os.unlink = unlink  # type: ignore[assignment]
    else:
        original_absence = quarantine.ReceiptQuarantineStore.prove_durable_absence

        def absence(self: quarantine.ReceiptQuarantineStore) -> quarantine.DurableAbsenceProof:
            original_absence(self)
            os._exit(82)

        quarantine.ReceiptQuarantineStore.prove_durable_absence = absence
    finalizer.finalize_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    os._exit(83)


def test_live_finalization_seals_and_preserves_private_copy(tmp_path: Path) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    raw = receipt.read_bytes()
    finalizer.finalize_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    assert not receipt.exists()
    identity_dir = (workspace / "memory" / ".recovery" / "receipt-quarantine"
                    / binding.transaction_identity)
    assert (identity_dir / "receipt.bin").read_bytes() == raw
    assert quarantine.inspect_receipt_quarantine(workspace).sealed_identities == (
        binding.transaction_identity,
    )
    with pytest.raises(finalizer.OfflineReceiptFinalizationError):
        finalizer.finalize_pending_memory_receipt_offline(workspace, sessions_root=sessions)


def test_replay_only_explicit_absent_copy_ready(tmp_path: Path) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as store:
            store.prepare(receipt.read_bytes())
            store.copy_receipt()
    with pytest.raises(finalizer.OfflineReceiptFinalizationError):
        finalizer.replay_unlinked_memory_receipt_offline(
            workspace, sessions_root=sessions, transaction_identity=binding.transaction_identity,
        )
    receipt.unlink()
    finalizer.replay_unlinked_memory_receipt_offline(
        workspace, sessions_root=sessions, transaction_identity=binding.transaction_identity,
    )
    assert quarantine.inspect_receipt_quarantine(workspace).sealed_identities == (
        binding.transaction_identity,
    )


def test_missing_replay_does_not_create_layout(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    with pytest.raises(finalizer.OfflineReceiptFinalizationError):
        finalizer.replay_unlinked_memory_receipt_offline(
            workspace, sessions_root=sessions, transaction_identity="a" * 64,
        )
    assert list(workspace.iterdir()) == []


@pytest.mark.parametrize("stage,after", [
    ("prepared.json", False), ("prepared.json", True),
    ("receipt.bin", False), ("receipt.bin", True),
    ("copy-ready.json", False), ("copy-ready.json", True),
    ("committed.json", False), ("committed.json", True),
])
def test_publication_fault_retains_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, after: bool,
) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    raw = receipt.read_bytes()
    original = quarantine.ReceiptQuarantineStore._publish

    def fault(self: quarantine.ReceiptQuarantineStore, name: str, content: bytes) -> None:
        if name == stage and not after:
            raise OSError("sensitive-do-not-echo")
        original(self, name, content)
        if name == stage:
            raise OSError("sensitive-do-not-echo")

    with monkeypatch.context() as patch:
        patch.setattr(quarantine.ReceiptQuarantineStore, "_publish", fault)
        with pytest.raises(finalizer.OfflineReceiptFinalizationError) as caught:
            finalizer.finalize_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    assert str(caught.value) == "offline receipt finalization unavailable"
    assert "sensitive" not in repr(caught.value)
    assert "sensitive" not in str(caught.value)
    identity_dir = workspace / "memory" / ".recovery" / "receipt-quarantine" / binding.transaction_identity
    assert identity_dir.exists()
    if stage == "committed.json":
        assert not receipt.exists()
        assert (identity_dir / "receipt.bin").read_bytes() == raw
        if not after:
            finalizer.replay_unlinked_memory_receipt_offline(
                workspace, sessions_root=sessions, transaction_identity=binding.transaction_identity,
            )
    else:
        assert receipt.read_bytes() == raw
        with pytest.raises(finalizer.OfflineReceiptFinalizationError):
            finalizer.replay_unlinked_memory_receipt_offline(
                workspace, sessions_root=sessions, transaction_identity=binding.transaction_identity,
            )


@pytest.mark.parametrize("stage", ["unlink-before", "unlink-after", "absence", "copy-proof"])
def test_unlink_and_post_unlink_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    raw = receipt.read_bytes()
    original_unlink = os.unlink
    original_absence = quarantine.ReceiptQuarantineStore.prove_durable_absence
    original_proof = terminal.prove_terminal_complete_under_locks

    def unlink(name: str, *, dir_fd: int | None = None) -> None:
        if name == ".consolidation.pending.json":
            if stage == "unlink-before":
                raise OSError("private receipt")
            original_unlink(name, dir_fd=dir_fd)
            if stage == "unlink-after":
                raise OSError("private receipt")
        else:
            original_unlink(name, dir_fd=dir_fd)

    def absence(self: quarantine.ReceiptQuarantineStore) -> quarantine.DurableAbsenceProof:
        if stage == "absence":
            raise OSError("private receipt")
        return original_absence(self)

    def proof(*args: object, **kwargs: object) -> terminal.TerminalCompleteProof:
        if stage == "copy-proof" and isinstance(args[-1], quarantine.ValidatedQuarantineCopy):
            raise OSError("private receipt")
        return original_proof(*args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as patch:
        patch.setattr(os, "unlink", unlink)
        patch.setattr(quarantine.ReceiptQuarantineStore, "prove_durable_absence", absence)
        patch.setattr(terminal, "prove_terminal_complete_under_locks", proof)
        if stage == "unlink-after":
            finalizer.finalize_pending_memory_receipt_offline(workspace, sessions_root=sessions)
        else:
            with pytest.raises(finalizer.OfflineReceiptFinalizationError) as caught:
                finalizer.finalize_pending_memory_receipt_offline(workspace, sessions_root=sessions)
            assert "private" not in str(caught.value)
    if stage == "unlink-before":
        assert receipt.read_bytes() == raw
    else:
        assert not receipt.exists()
        if stage != "unlink-after":
            finalizer.replay_unlinked_memory_receipt_offline(
                workspace, sessions_root=sessions, transaction_identity=binding.transaction_identity,
            )
            assert quarantine.inspect_receipt_quarantine(workspace).sealed_identities == (
                binding.transaction_identity,
            )


@pytest.mark.parametrize("drift", ["same-byte-inode", "bytes", "mode", "symlink", "hardlink"])
def test_pending_drift_before_unlink_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str,
) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    original = quarantine.ReceiptQuarantineStore.copy_receipt

    def changed(self: quarantine.ReceiptQuarantineStore) -> None:
        original(self)
        if drift == "bytes":
            receipt.write_bytes(b"altered")
        elif drift == "mode":
            receipt.chmod(0o644)
        elif drift == "hardlink":
            os.link(receipt, tmp_path / "other-link")
        else:
            moved = tmp_path / "original"
            receipt.rename(moved)
            if drift == "same-byte-inode":
                receipt.write_bytes(moved.read_bytes())
                receipt.chmod(0o600)
            else:
                receipt.symlink_to(moved)

    with monkeypatch.context() as patch:
        patch.setattr(quarantine.ReceiptQuarantineStore, "copy_receipt", changed)
        with pytest.raises(finalizer.OfflineReceiptFinalizationError):
            finalizer.finalize_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    assert receipt.exists()
    assert not (workspace / "memory" / ".recovery" / "receipt-quarantine"
                / binding.transaction_identity / "committed.json").exists()


@pytest.mark.parametrize("drift", ["session", "marker", "blob", "lock", "parent"])
def test_authority_and_semantic_drift_before_unlink_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str,
) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    raw = receipt.read_bytes()
    original = quarantine.ReceiptQuarantineStore.copy_receipt

    def changed(self: quarantine.ReceiptQuarantineStore) -> None:
        original(self)
        if drift == "session":
            session = locate_recovery_candidate(workspace, sessions, _SESSION_KEY).session_path
            session.write_bytes(session.read_bytes() + b" ")
        elif drift == "marker":
            (workspace / ".nanobot" / "workspace-id").write_text("f" * 32 + "\n")
        elif drift == "blob":
            blob = next((workspace / "memory" / ".recovery" / "blobs" / "sha256").iterdir())
            blob.write_bytes(b"corrupt")
        else:
            parent = workspace if drift == "lock" else workspace / "memory"
            name = ".memory-recovery.lock" if drift == "lock" else ".recovery"
            (parent / name).rename(parent / f"{name}.moved")

    with monkeypatch.context() as patch:
        patch.setattr(quarantine.ReceiptQuarantineStore, "copy_receipt", changed)
        with pytest.raises(finalizer.OfflineReceiptFinalizationError):
            finalizer.finalize_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    assert receipt.read_bytes() == raw
    assert not (workspace / "memory" / ".recovery" / "receipt-quarantine"
                / binding.transaction_identity / "committed.json").exists()


def test_replay_rejects_committed_and_noncanonical_identity(tmp_path: Path) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    finalizer.finalize_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    copy = (workspace / "memory" / ".recovery" / "receipt-quarantine"
            / binding.transaction_identity / "receipt.bin")
    assert stat.S_IMODE(copy.stat().st_mode) == 0o600
    for identity in (binding.transaction_identity, binding.transaction_identity.upper(), "z" * 64):
        with pytest.raises(finalizer.OfflineReceiptFinalizationError):
            finalizer.replay_unlinked_memory_receipt_offline(
                workspace, sessions_root=sessions, transaction_identity=identity,
            )
    assert not receipt.exists() and copy.exists()


@pytest.mark.parametrize("boundary,exit_code", [("unlink", 81), ("absence-fsync", 82)])
def test_process_crash_replays_only_explicit_copy_ready_identity(
    tmp_path: Path, boundary: str, exit_code: int,
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("subprocess crash qualification requires fork")
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    process = multiprocessing.get_context("fork").Process(
        target=_crash_after_boundary, args=(workspace, sessions, boundary),
    )
    process.start()
    process.join(timeout=25)
    assert process.exitcode == exit_code
    assert not receipt.exists()
    with pytest.raises(finalizer.OfflineReceiptFinalizationError):
        finalizer.finalize_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    finalizer.replay_unlinked_memory_receipt_offline(
        workspace, sessions_root=sessions, transaction_identity=binding.transaction_identity,
    )
    assert quarantine.inspect_receipt_quarantine(workspace).sealed_identities == (
        binding.transaction_identity,
    )
