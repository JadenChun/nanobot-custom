"""Real quarantine commits retain proof rejection and one-shot fault ordering."""

import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_transaction_terminal as terminal
from nanobot.session.checked_source import open_checked_recovery_session
from nanobot.session.recovery_locator import locate_recovery_candidate
from tests.agent.test_memory_recovery_quarantine_storage import _real_layout
from tests.agent.test_memory_recovery_transaction_prepare import _SESSION_KEY, _WORKSPACE_ID

ReadyCommit = tuple[
    storage.WorkspaceMemoryAuthority, storage.WorkspaceMemoryOperation,
    quarantine.QuarantineBinding, quarantine.ReceiptQuarantineStore,
    quarantine.DurableAbsenceProof, terminal.TerminalCompleteProof,
]


@contextmanager
def _ready_commit(tmp_path: Path) -> Iterator[ReadyCommit]:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        candidate = locate_recovery_candidate(workspace, sessions, _SESSION_KEY)
        with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as store:
            store.prepare(receipt.read_bytes())
            store.copy_receipt()
            receipt.unlink()
            absence = store.prove_durable_absence()
            with (
                quarantine.open_validated_quarantine_copy(
                    authority, operation, binding, require_phase="copy-ready",
                    require_pending_absent=True,
                ) as copy,
                open_checked_recovery_session(operation, candidate) as checked,
            ):
                proof = terminal.prove_terminal_complete_under_locks(
                    authority, operation, checked, _WORKSPACE_ID, copy,
                )
                yield authority, operation, binding, store, absence, proof


def _marker(binding: quarantine.QuarantineBinding) -> Path:
    return (binding.workspace / "memory" / ".recovery" / "receipt-quarantine"
            / binding.transaction_identity / "committed.json")


def test_foreign_store_absence_rejected_before_descriptor_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _ready_commit(tmp_path) as (authority, operation, binding, _, absence, proof):
        with quarantine.ReceiptQuarantineStore.from_operation(authority, operation, binding) as other:
            observed: list[int] = []

            def forbidden_fstat(fd: int) -> os.stat_result:
                observed.append(fd)
                raise AssertionError("foreign-store proof must be rejected before fstat")

            with monkeypatch.context() as patch:
                patch.setattr(os, "fstat", forbidden_fstat)
                with pytest.raises(quarantine.QuarantineError):
                    other.commit(absence, proof)
                assert not observed
        assert not _marker(binding).exists()


def test_terminal_reproof_failure_does_not_consume_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = terminal.TerminalCompleteProof.require_current
    with _ready_commit(tmp_path) as (_, _, binding, store, absence, proof):
        rechecks: list[str] = []

        def reprove_then_fail(value: terminal.TerminalCompleteProof) -> None:
            original(value)
            rechecks.append("reproved")
            raise OSError("injected private reproof fault")

        with monkeypatch.context() as patch:
            patch.setattr(terminal.TerminalCompleteProof, "require_current", reprove_then_fail)
            with pytest.raises(quarantine.QuarantineError) as raised:
                store.commit(absence, proof)
            assert str(raised.value) == str(quarantine.QuarantineError())
        assert rechecks == ["reproved"]
        assert not _marker(binding).exists()
        store.commit(absence, proof)
        assert _marker(binding).exists()


def test_publication_failure_keeps_absence_consumed_and_original_digest_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_current = terminal.TerminalCompleteProof.require_current
    original_open = os.open
    original_digest = getattr(quarantine, "_digest")
    with _ready_commit(tmp_path) as (_, _, binding, store, absence, proof):
        trace: list[str] = []
        original_publish = getattr(store, "_publish")

        def observed_digest(content: bytes) -> str:
            if hashlib.sha256(content).hexdigest() == proof.semantic_seal_digest:
                trace.append("claims")
            return original_digest(content)

        def observed_current(value: terminal.TerminalCompleteProof) -> None:
            trace.append("terminal")
            original_current(value)
            # Capability checks inspect os.open identity during real reproof;
            # inject only at the later publication boundary, as owning tests do.
            patch.setattr(os, "open", failing_open)

        def observed_publish(name: str, content: bytes) -> None:
            if name == "committed.json":
                trace.append("publish")
            original_publish(name, content)

        def failing_open(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes], flags: int,
            mode: int = 0o777, *, dir_fd: int | None = None,
        ) -> int:
            if path == "committed.json" and flags & os.O_CREAT:
                raise OSError("injected private publication fault")
            return original_open(path, flags, mode, dir_fd=dir_fd)

        with monkeypatch.context() as patch:
            patch.setattr(quarantine, "_digest", observed_digest)
            patch.setattr(terminal.TerminalCompleteProof, "require_current", observed_current)
            patch.setattr(store, "_publish", observed_publish)
            with pytest.raises(quarantine.QuarantineError) as raised:
                store.commit(absence, proof)
            assert str(raised.value) == str(quarantine.QuarantineError())
            assert trace == ["claims", "terminal", "publish"]
            assert not _marker(binding).exists()
            assert _marker(binding).with_name("receipt.bin").exists()
            trace.clear()
            with pytest.raises(quarantine.QuarantineError):
                store.commit(absence, proof)
            assert not trace
            assert not _marker(binding).exists()
