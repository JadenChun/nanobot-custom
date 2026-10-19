"""Fail-closed startup policy for retained memory-recovery receipt evidence."""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Generator, NoReturn

from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent.memory_recovery import (
    MemoryReceiptRecoveryRequiredError,
    require_memory_receipt_recovery,
)
from nanobot.agent.memory_recovery_historical_seal import validate_historical_seals_under_gate


class StartupReceiptAdmissionError(MemoryReceiptRecoveryRequiredError):
    """Fixed, content-free denial compatible with the pending-receipt boundary."""


def _deny() -> NoReturn:
    raise StartupReceiptAdmissionError() from None


def _paths(workspace: Path | str, sessions_root: Path | str) -> tuple[Path, Path]:
    return quarantine.normalize_recovery_root(workspace), quarantine.normalize_recovery_root(sessions_root)


def _inspect(workspace: Path) -> quarantine.QuarantineInspection:
    pending = workspace / "memory" / ".consolidation.pending.json"
    require_memory_receipt_recovery(pending)
    inspection = quarantine.inspect_receipt_quarantine(workspace)
    require_memory_receipt_recovery(pending)
    return inspection


def _complete(inspection: quarantine.QuarantineInspection) -> tuple[str, ...]:
    if (inspection.clean or inspection.incomplete or not inspection.committed_identities
            or inspection.committed_identities != inspection.sealed_identities):
        _deny()
    return inspection.sealed_identities


def _clean(inspection: quarantine.QuarantineInspection) -> bool:
    return (inspection.clean and not inspection.incomplete
            and not inspection.committed_identities and not inspection.sealed_identities)


def _verify_seals(
    gate: storage.WorkspaceMemoryLockGate, sessions_root: Path,
    identities: tuple[str, ...], workspace: Path,
) -> None:
    observed = _inspect(workspace)
    if _complete(observed) != identities:
        _deny()
    evidence = validate_historical_seals_under_gate(gate, sessions_root)
    if evidence.sealed_identities != identities:
        _deny()
    if _complete(_inspect(workspace)) != identities:
        _deny()
    gate.require_current()


def require_startup_receipt_admission(
    workspace: Path | str, *, sessions_root: Path | str,
) -> None:
    """Read-only clean preflight, or full historical proof under an existing lock.

    Semantic validation may fsync already existing immutable evidence.
    """
    try:
        path, root = _paths(workspace, sessions_root)
        inspection = _inspect(path)
        if _clean(inspection):
            return
        identities = _complete(inspection)
        with storage.open_existing_workspace_memory_lock(path) as gate:
            _verify_seals(gate, root, identities, path)
    except StartupReceiptAdmissionError:
        raise StartupReceiptAdmissionError() from None
    except Exception:
        _deny()


@contextmanager
def admitted_startup_mutation(
    workspace: Path | str, *, sessions_root: Path | str,
) -> Generator[storage.WorkspaceMemoryLockGate | None, None, None]:
    """Admit one bounded writer after a second inspection under its single lock.

    On descriptor-unsupported platforms only clean absence is admitted and
    there is no POSIX lock to yield. Never use this context around child launch.
    """
    try:
        path, root = _paths(workspace, sessions_root)
        inspection = _inspect(path)
        identities = None if _clean(inspection) else _complete(inspection)
    except Exception:
        _deny()

    if identities is not None:
        stack = ExitStack()
        try:
            gate = stack.enter_context(storage.open_existing_workspace_memory_lock(path))
            _verify_seals(gate, root, identities, path)
        except Exception:
            stack.close()
            _deny()
        try:
            yield gate
        finally:
            stack.close()
        return

    if os.name != "posix":
        try:
            if not _clean(_inspect(path)):
                _deny()
        except Exception:
            _deny()
        yield None
        return

    stack = ExitStack()
    try:
        path.mkdir(parents=True, exist_ok=True)
        gate = stack.enter_context(storage.open_workspace_memory_lock(path))
        if not _clean(_inspect(path)):
            _deny()
        gate.require_current()
    except Exception:
        stack.close()
        _deny()
    try:
        yield gate
    finally:
        stack.close()


__all__ = ["StartupReceiptAdmissionError", "admitted_startup_mutation",
           "require_startup_receipt_admission"]
