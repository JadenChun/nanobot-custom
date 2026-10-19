"""Explicit operator-only offline disposition of a terminal memory receipt.

No startup or runtime path imports or invokes these entrypoints. The private
raw copy is retained indefinitely; failed operations leave all residue intact.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import NoReturn

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_transaction_terminal as terminal
from nanobot.session.checked_source import CheckedSessionAuthority, open_checked_recovery_session
from nanobot.session.recovery_locator import locate_recovery_candidate

_PENDING = ".consolidation.pending.json"


class OfflineReceiptFinalizationError(RuntimeError):
    """Content-free failure requiring manual operator reconciliation."""

    def __init__(self) -> None:
        super().__init__("offline receipt finalization unavailable")


def _fail() -> NoReturn:
    raise OfflineReceiptFinalizationError() from None


def _paths(workspace: Path | str, sessions_root: Path | str) -> tuple[Path, Path]:
    root, sessions = Path(workspace), Path(sessions_root)
    if not root.is_absolute() or not sessions.is_absolute():
        _fail()
    return root, sessions


def _receipt(raw: bytes) -> receipt_codec.MemoryReceipt:
    parsed = receipt_codec.decode_memory_receipt_bytes(raw)
    if not parsed.session_key:
        _fail()
    return parsed


def _commit_copy(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    store: quarantine.ReceiptQuarantineStore,
    binding: quarantine.QuarantineBinding,
    checked: CheckedSessionAuthority,
    historical: terminal.TerminalCompleteProof | None = None,
) -> None:
    # The original checked session remains held through sealed publication.
    absence = store.prove_durable_absence()
    with quarantine.open_validated_quarantine_copy(
        authority, operation, binding, require_phase="copy-ready",
        require_pending_absent=True,
    ) as copy:
        fresh = terminal.prove_terminal_complete_under_locks(
            authority, operation, checked, binding.workspace_id, copy,
        )
        fresh.require_current()
        if (historical is not None and
                (fresh.semantic_seal_digest != historical.semantic_seal_digest
                 or fresh.receipt_stat != historical.receipt_stat)):
            _fail()
        store.commit(absence, fresh)


def finalize_pending_memory_receipt_offline(
    workspace: Path | str, *, sessions_root: Path | str,
) -> None:
    """Finalize one live terminal receipt while the assistant is stopped."""
    try:
        root, sessions = _paths(workspace, sessions_root)
        with storage.open_existing_workspace_memory(root) as authority:
            with authority.operation() as operation:
                snapshot = authority.snapshot_target(_PENDING)
                raw = snapshot.content
                if raw is None or snapshot.stat_identity is None:
                    _fail()
                parsed = _receipt(raw)
                assert parsed.session_key is not None
                candidate = locate_recovery_candidate(root, sessions, parsed.session_key)
                with open_checked_recovery_session(operation, candidate) as checked:
                    historical = terminal.prove_terminal_complete_under_locks(
                        authority, operation, checked, candidate.workspace_id, snapshot,
                    )
                    historical.require_current()
                    binding = quarantine.QuarantineBinding(
                        authority.workspace, sessions, candidate.workspace_id,
                        historical.transaction_identity, historical.complete_digest,
                        historical.receipt_digest, historical.receipt_size,
                    )
                    with quarantine.ReceiptQuarantineStore.from_operation(
                        authority, operation, binding,
                    ) as store:
                        store.prepare(raw)
                        store.copy_receipt()
                        historical.require_current()
                        with operation.borrow_pending_receipt_parent(snapshot) as parent:
                            parent.require_current()
                            current = parent.snapshot_target()
                            if (current.content != snapshot.content
                                    or current.stat_identity != snapshot.stat_identity):
                                _fail()
                            fd = parent.duplicate_dir_fd()
                            try:
                                # This comparison binds both exact bytes and all nine
                                # original stat fields to the pinned parent name.
                                named = quarantine._read(  # pyright: ignore[reportPrivateUsage]
                                    fd, _PENDING, quarantine._MAX_RECEIPT,  # pyright: ignore[reportPrivateUsage]
                                )
                                if (named is None or named[0] != snapshot.content
                                        or quarantine._stat_tuple(  # pyright: ignore[reportPrivateUsage]
                                            named[1]
                                        ) != historical.receipt_stat):
                                    _fail()
                                historical.require_current()
                                checked.require_current()
                                store._current()  # pyright: ignore[reportPrivateUsage]
                                operation.require_current()
                                try:
                                    os.unlink(_PENDING, dir_fd=fd)
                                except OSError:
                                    # An after-effect failure is uncertain until
                                    # the absent name and full proof are rechecked.
                                    if quarantine._entry(  # pyright: ignore[reportPrivateUsage]
                                        fd, _PENDING,
                                    ) is not None:
                                        _fail()
                                if quarantine._entry(  # pyright: ignore[reportPrivateUsage]
                                    fd, _PENDING,
                                ) is not None:
                                    _fail()
                            finally:
                                os.close(fd)
                        # Keep the original checked session alive through commit.
                        _commit_copy(authority, operation, store, binding, checked, historical)
    except OfflineReceiptFinalizationError:
        raise
    except Exception:
        _fail()


def replay_unlinked_memory_receipt_offline(
    workspace: Path | str, *, sessions_root: Path | str, transaction_identity: str,
) -> None:
    """Resume one explicitly named copy-ready/absent crash residue offline."""
    try:
        root, sessions = _paths(workspace, sessions_root)
        if (type(transaction_identity) is not str or len(transaction_identity) != 64
                or any(c not in "0123456789abcdef" for c in transaction_identity)):
            _fail()
        with storage.open_existing_workspace_memory(root) as authority:
            with authority.operation() as operation:
                binding = quarantine.read_existing_copy_ready_binding(
                    authority, operation, sessions, transaction_identity,
                )
                with quarantine.open_validated_quarantine_copy(
                    authority, operation, binding, require_phase="copy-ready",
                    require_pending_absent=True,
                ) as copy:
                    parsed = _receipt(copy.raw_bytes)
                    assert parsed.session_key is not None
                    candidate = locate_recovery_candidate(root, sessions, parsed.session_key)
                    if candidate.workspace_id != binding.workspace_id:
                        _fail()
                    with open_checked_recovery_session(operation, candidate) as checked:
                        proof = terminal.prove_terminal_complete_under_locks(
                            authority, operation, checked, binding.workspace_id, copy,
                        )
                        proof.require_current()
                        if (proof.transaction_identity != transaction_identity
                                or proof.receipt_digest != hashlib.sha256(copy.raw_bytes).hexdigest()):
                            _fail()
                        with quarantine.ReceiptQuarantineStore.from_existing_operation(
                            authority, operation, binding,
                        ) as store:
                            _commit_copy(authority, operation, store, binding, checked, proof)
    except OfflineReceiptFinalizationError:
        raise
    except Exception:
        _fail()


__all__ = ["OfflineReceiptFinalizationError", "finalize_pending_memory_receipt_offline",
           "replay_unlinked_memory_receipt_offline"]
