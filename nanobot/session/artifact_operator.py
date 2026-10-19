"""Explicit copied-state artifact inspection and exact rematerialization."""

from __future__ import annotations

import os
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.session.artifact import (
    ArtifactDescriptor,
    ArtifactValidationError,
    parse_artifact_reference,
    published_artifacts,
)
from nanobot.session.artifact_factory import ConfiguredArtifactReaderFactory
from nanobot.session.artifact_rematerialization import (
    ArtifactRematerializationError,
    ArtifactRematerializationResult,
    ArtifactRematerializer,
)
from nanobot.session.artifact_store import ArtifactStorageError
from nanobot.session.execution_owner import ExecutionOwnershipError, acquire_execution
from nanobot.session.manager import JsonlSessionStore, SessionManager
from nanobot.session.runtime_authority import OfflineSessionOperator
from nanobot.session.session_mode import LockedSessionSnapshot, SessionModeError
from nanobot.session.source import SourceValidationError
from nanobot.session.source_store import (
    SourceSessionStore,
    SourceSnapshot,
    SourceState,
    validate_source_session_bytes,
)
from nanobot.session.turn_writer import SourceTurnWriter

InspectionStatus = Literal["disabled", "complete", "incomplete"]


class ArtifactOperatorError(RuntimeError):
    """Fixed, content-free copied-artifact operator failure."""

    def __init__(self) -> None:
        super().__init__("Copied artifact operation is unavailable")


@dataclass(frozen=True, slots=True)
class ArtifactInspectionResult:
    status: InspectionStatus
    reachable_blobs: int = 0
    reachable_bytes: int = 0
    unreachable_blobs: int = 0
    unreachable_bytes: int = 0
    missing_blobs: int = 0
    missing_bytes: int = 0


@dataclass(frozen=True, slots=True)
class CopiedArtifactReadOperations:
    """Late read operations, not a resolver or a lease/admission proof."""

    locked_snapshot: Callable[[str], AbstractContextManager[LockedSessionSnapshot]]
    read_workspace_id: Callable[[], str]
    read_workspace: Callable[[], Path]
    read_namespace: Callable[[], Path]


def _absolute_nonroot(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate == Path(candidate.anchor) or ".." in candidate.parts:
        raise ArtifactOperatorError()
    return Path(os.path.abspath(os.fspath(candidate)))


def _same_identity(first: Path, second: Path) -> bool:
    if first == second:
        return True
    try:
        return first.exists() and second.exists() and first.samefile(second)
    except OSError:
        raise ArtifactOperatorError() from None


def _snapshot_from_bytes(data: bytes, *, key: str) -> SourceSnapshot:
    header, events = validate_source_session_bytes(data, expected_key=key)
    state = SourceState(
        "copied-artifact-snapshot",
        key,
        header["incarnation"],
        header["revision"],
        header["generation"],
        None,
        "",
    )
    return SourceSnapshot(state, header, events, None)


def _canonical_snapshot(
    locked: LockedSessionSnapshot,
    *,
    key: str,
    incarnation: str | None = None,
    generation: int | None = None,
) -> SourceSnapshot:
    _registry_id, _record_revision, mode, bound_incarnation, bound_generation = locked.binding
    if (
        mode != "canonical"
        or (incarnation is not None and incarnation != bound_incarnation)
        or (generation is not None and generation != bound_generation)
    ):
        raise ArtifactOperatorError()
    snapshot = _snapshot_from_bytes(locked.source_bytes, key=key)
    if (
        snapshot.state.incarnation != bound_incarnation
        or snapshot.state.generation != bound_generation
    ):
        raise ArtifactOperatorError()
    return snapshot


def _descriptor(snapshot: SourceSnapshot, workspace_id: str, reference: str) -> ArtifactDescriptor:
    identity = parse_artifact_reference(reference)
    matches = [
        descriptor
        for descriptor in published_artifacts(snapshot, workspace_id=workspace_id)
        if descriptor.artifact_id == identity
    ]
    if len(matches) != 1:
        raise ArtifactOperatorError()
    return matches[0]


def _install_scratch_source(
    root: Path,
    *,
    workspace_id: str,
    key: str,
    source_bytes: bytes,
) -> JsonlSessionStore:
    from nanobot.session.manager import JsonlSessionStore as ScratchSourceOwner

    return ScratchSourceOwner.create_copied_artifact_scratch(
        root, workspace_id=workspace_id, key=key, source_bytes=source_bytes,
        construct_backend=lambda workspace, sessions_root: JsonlSessionStore(
            workspace, sessions_root=sessions_root,
        ),
        mismatch_error=lambda: ArtifactOperatorError(),
    )


class CopiedArtifactOperator:
    """One-shot service retaining the copied namespace lease for each operation."""

    def __init__(
        self,
        copied_workspace: Path | str,
        *,
        copied_sessions_root: Path | str,
        copied_artifact_root: Path | str,
        artifact_factory: ConfiguredArtifactReaderFactory | None,
        read_file: ReadFileTool,
        live_workspace: Path | str | None = None,
        live_sessions_root: Path | str | None = None,
    ) -> None:
        if type(read_file) is not ReadFileTool:
            raise ArtifactOperatorError()
        self._read_file = read_file
        self._artifact_root = _absolute_nonroot(copied_artifact_root)
        self._factory = artifact_factory
        try:
            self._sessions = OfflineSessionOperator(
                Path(copied_workspace),
                copied_sessions_root=Path(copied_sessions_root),
                live_workspace=Path(live_workspace) if live_workspace is not None else None,
                live_sessions_root=(
                    Path(live_sessions_root) if live_sessions_root is not None else None
                ),
            )
            if artifact_factory is not None:
                if _same_identity(self._artifact_root, artifact_factory.settings.root):
                    raise ArtifactOperatorError()
                self._factory = artifact_factory.with_root(self._artifact_root)
        except ArtifactOperatorError:
            raise
        except (ArtifactValidationError, SessionModeError, OSError, RuntimeError, ValueError):
            raise ArtifactOperatorError() from None

    async def inspect(self, key: str) -> ArtifactInspectionResult:
        """Report content-free aggregate reachability for one exact canonical session."""
        if self._factory is None:
            return ArtifactInspectionResult("disabled")
        lease = None
        try:
            lease = await self._sessions.acquire()
            operations = SessionManager.copied_artifact_read_operations(lease.manager)
            if operations is None:
                raise ArtifactOperatorError()
            with operations.locked_snapshot(key) as locked:
                snapshot = _canonical_snapshot(locked, key=key)
            descriptors = published_artifacts(snapshot, workspace_id=operations.read_workspace_id())
            store = self._factory.create_store(
                workspace_id=operations.read_workspace_id(),
                session_key=key,
                incarnation=snapshot.state.incarnation,
                generation=snapshot.state.generation,
                forbidden_roots=(
                    operations.read_workspace(),
                    operations.read_namespace(),
                    *self._factory.read_policy_roots(self._read_file),
                ),
            )
            with store.transaction(blocking=False) as blobs:
                report = blobs.inspect(descriptors)
            status: InspectionStatus = (
                "complete" if report.missing_blobs == 0 else "incomplete"
            )
            return ArtifactInspectionResult(
                status,
                report.reachable_blobs,
                report.reachable_bytes,
                report.unreachable_blobs,
                report.unreachable_bytes,
                report.missing_blobs,
                report.missing_bytes,
            )
        except ArtifactOperatorError:
            raise
        except (
            ArtifactRematerializationError,
            ArtifactStorageError,
            ArtifactValidationError,
            ExecutionOwnershipError,
            SessionModeError,
            SourceValidationError,
            OSError,
            RuntimeError,
            UnicodeError,
            ValueError,
        ):
            raise ArtifactOperatorError() from None
        finally:
            if lease is not None and lease.close is not None:
                try:
                    close_result = lease.close()
                    if close_result is not None:
                        await close_result
                except Exception:
                    raise ArtifactOperatorError() from None

    async def rematerialize(
        self,
        key: str,
        *,
        incarnation: str,
        generation: int,
        reference: str,
    ) -> ArtifactRematerializationResult:
        """Recover one exact blob through a disposable real execution owner."""
        if self._factory is None:
            raise ArtifactOperatorError()
        lease = None
        try:
            lease = await self._sessions.acquire()
            operations = SessionManager.copied_artifact_read_operations(lease.manager)
            if operations is None:
                raise ArtifactOperatorError()
            with operations.locked_snapshot(key) as initial:
                snapshot = _canonical_snapshot(
                    initial,
                    key=key,
                    incarnation=incarnation,
                    generation=generation,
                )
                descriptor = _descriptor(snapshot, operations.read_workspace_id(), reference)

            store = self._factory.create_store(
                workspace_id=operations.read_workspace_id(),
                session_key=key,
                incarnation=incarnation,
                generation=generation,
                forbidden_roots=(
                    operations.read_workspace(),
                    operations.read_namespace(),
                    *self._factory.read_policy_roots(self._read_file),
                ),
            )
            with store.transaction(blocking=False) as blobs:
                with tempfile.TemporaryDirectory(prefix="nanobot-artifact-scratch-") as temporary:
                    backend = _install_scratch_source(
                        Path(temporary),
                        workspace_id=operations.read_workspace_id(),
                        key=key,
                        source_bytes=initial.source_bytes,
                    )
                    async with acquire_execution(
                        backend, key, run_id="artifact-rematerialization"
                    ) as owner:
                        writer = SourceTurnWriter(
                            SourceSessionStore(backend),
                            owner,
                            run_id="artifact-rematerialization",
                        )
                        reader = self._factory.create_reader(
                            writer,
                            False,
                            self._read_file,
                            store=store,
                        )
                        if reader is None:
                            raise ArtifactOperatorError()
                        result = ArtifactRematerializer(reader).rematerialize(
                            reference,
                            expected=owner.read_snapshot().state,
                            blobs=blobs,
                        )
                    blobs.verify_durable(descriptor, store=store)
                    with operations.locked_snapshot(key) as final:
                        if (
                            final.binding != initial.binding
                            or final.mode_record_bytes != initial.mode_record_bytes
                            or final.source_bytes != initial.source_bytes
                        ):
                            raise ArtifactOperatorError()
            return result
        except ArtifactOperatorError:
            raise
        except (
            ArtifactRematerializationError,
            ArtifactStorageError,
            ArtifactValidationError,
            ExecutionOwnershipError,
            SessionModeError,
            SourceValidationError,
            OSError,
            RuntimeError,
            UnicodeError,
            ValueError,
        ):
            raise ArtifactOperatorError() from None
        finally:
            if lease is not None and lease.close is not None:
                try:
                    close_result = lease.close()
                    if close_result is not None:
                        await close_result
                except Exception:
                    raise ArtifactOperatorError() from None


__all__ = [
    "ArtifactInspectionResult",
    "ArtifactOperatorError",
    "CopiedArtifactOperator",
]
