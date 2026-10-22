"""Locked, idempotent staging of authenticated history recovery markers."""

from __future__ import annotations

import errno
import os
from dataclasses import dataclass
from typing import Literal

from nanobot.agent import memory_recovery_marker as marker
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_target_mutation as target_mutation


@dataclass(frozen=True, slots=True)
class HistoryMarkerStageResult:
    """Disposition and authority-issued snapshot produced by marker staging."""

    disposition: Literal["already_present", "staged"]
    snapshot: storage.MemoryTargetSnapshot


def _same_snapshot(
    first: storage.MemoryTargetSnapshot,
    second: storage.MemoryTargetSnapshot,
) -> bool:
    return (
        first.name == second.name
        and first.content == second.content
        and first.stat_identity == second.stat_identity
    )


def _fsync_parent_and_reverify(
    operation: storage.WorkspaceMemoryOperation,
    snapshot: storage.MemoryTargetSnapshot,
) -> storage.MemoryTargetSnapshot:
    """Durably bind one issued target snapshot to its parent directory."""

    with operation.borrow_target_parent(snapshot) as scope:
        scope.require_current()
        parent_fd = scope.duplicate_dir_fd()
        try:
            while True:
                try:
                    os.fsync(parent_fd)
                    break
                except OSError as error:
                    if error.errno == errno.EINTR:
                        continue
                    raise storage.MemoryStorageError("durability") from None
                except (TypeError, ValueError):
                    raise storage.MemoryStorageError("durability") from None

            scope.require_current()
            current_snapshot = scope.snapshot_target()
            if not _same_snapshot(snapshot, current_snapshot):
                raise storage.MemoryStorageError("binding") from None
            return current_snapshot
        finally:
            try:
                os.close(parent_fd)
            except (OSError, TypeError, ValueError):
                raise storage.MemoryStorageError("io") from None


def stage_history_marker(
    authority: storage.WorkspaceMemoryAuthority,
    operation: storage.WorkspaceMemoryOperation,
    marker_id: str,
    history_entry: str,
) -> HistoryMarkerStageResult:
    """Append one exact marker frame under the active workspace authority.

    The operation borrow verifies that the supplied operation belongs to this
    authority and that its initial HISTORY.md snapshot remains bound before the
    frame is classified. An uncertain rename is reconciled by reading the
    target once: only the exact proposal is accepted, and the replacement is
    never retried.
    """

    original_snapshot = authority.snapshot_target("HISTORY.md")
    with operation.borrow_target_parent(original_snapshot) as scope:
        current_snapshot = scope.snapshot_target()
        if not _same_snapshot(original_snapshot, current_snapshot):
            raise storage.MemoryStorageError("binding") from None

    existing_history = (
        current_snapshot.content if current_snapshot.content is not None else b""
    )
    disposition, proposal = marker.classify_history_marker_frame(
        existing_history,
        marker_id,
        history_entry,
    )
    if disposition == "present":
        if proposal is not None:
            raise storage.MemoryStorageError("invalid") from None
        durable_snapshot = _fsync_parent_and_reverify(operation, current_snapshot)
        return HistoryMarkerStageResult("already_present", durable_snapshot)
    if disposition != "absent" or proposal is None:
        raise storage.MemoryStorageError("invalid") from None

    try:
        staged_snapshot = target_mutation.replace_target_if_unchanged(
            operation,
            original_snapshot,
            proposal,
        )
    except target_mutation.MemoryTargetReplacementUncertainError:
        reconciled_snapshot = authority.snapshot_target("HISTORY.md")
        if reconciled_snapshot.content == proposal:
            durable_snapshot = _fsync_parent_and_reverify(operation, reconciled_snapshot)
            return HistoryMarkerStageResult("staged", durable_snapshot)
        if _same_snapshot(original_snapshot, reconciled_snapshot):
            raise target_mutation.MemoryTargetReplacementUncertainError() from None
        raise storage.MemoryStorageError("binding") from None

    return HistoryMarkerStageResult("staged", staged_snapshot)


__all__ = ["HistoryMarkerStageResult", "stage_history_marker"]
