"""Dormant exact-source rematerialization of one published private artifact."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

from nanobot.session.artifact import (
    ArtifactDescriptor,
    ArtifactValidationError,
    bound_artifact_text,
    parse_artifact_reference,
    published_artifacts,
)
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.artifact_store import (
    ArtifactBlobTransaction,
    ArtifactInstallCorruptionError,
    ArtifactInstallIOError,
    ArtifactInstallLimitError,
    ArtifactInstallUncertainError,
    ArtifactStorageError,
)
from nanobot.session.execution_owner import ExecutionOwnershipError
from nanobot.session.manager import SessionWriteConflict
from nanobot.session.source import SourceValidationError
from nanobot.session.source_store import SourceSnapshot, SourceState

RematerializationStatus = Literal['installed', 'already_present']
RematerializationFailure = Literal[
    'unavailable', 'corrupt', 'limit', 'io', 'uncertain'
]


class ArtifactRematerializationError(RuntimeError):
    """Fixed, content-free rematerialization failure."""

    _MESSAGES = {
        'unavailable': 'Artifact rematerialization is unavailable',
        'corrupt': 'Artifact rematerialization found corrupt stored evidence',
        'limit': 'Artifact rematerialization exceeds configured limits',
        'io': 'Artifact rematerialization storage failed',
        'uncertain': 'Artifact rematerialization durability is uncertain',
    }

    def __init__(self, reason: RematerializationFailure):
        self.reason = reason
        super().__init__(self._MESSAGES[reason])


@dataclass(frozen=True)
class ArtifactRematerializationResult:
    status: RematerializationStatus
    reference: str
    byte_length: int


class ArtifactRematerializer:
    """Explicit no-replay recovery bound to one admitted artifact reader."""

    def __init__(self, reader: ArtifactReader):
        if type(reader) is not ArtifactReader:
            raise ArtifactRematerializationError('unavailable')
        self._reader = reader

    def _proof(
        self,
        identity: str,
        snapshot: SourceSnapshot,
        *,
        locked: bool,
    ) -> tuple[ArtifactDescriptor, bytes]:
        reader = self._reader
        descriptors = published_artifacts(snapshot, workspace_id=reader.store.workspace_id)
        descriptor = next((item for item in descriptors if item.artifact_id == identity), None)
        if descriptor is None:
            raise ArtifactValidationError('Artifact is unavailable')
        if locked:
            reader.verify_native_source_in_snapshot(descriptor, snapshot)
        else:
            reader.authorize_source(descriptor, snapshot)
        content = bound_artifact_text(
            descriptor, snapshot, workspace_id=reader.store.workspace_id
        )
        return descriptor, content

    def rematerialize(
        self,
        reference: str,
        *,
        expected: SourceState,
        blobs: ArtifactBlobTransaction | None = None,
    ) -> ArtifactRematerializationResult:
        """Install absent bytes, optionally using the caller's retained store transaction."""
        reader = self._reader
        writer, owner = reader.writer, reader.writer.owner
        try:
            if type(expected) is not SourceState:
                raise ArtifactValidationError('Invalid expected source state')
            if blobs is not None and (
                type(blobs) is not ArtifactBlobTransaction or not blobs.matches_store(reader.store)
            ):
                raise ArtifactStorageError('Artifact transaction uses another configured store')
            identity = parse_artifact_reference(reference)
            reader.revalidate_private_location()
            owner.validate_dispatch()
            preflight = owner.read_snapshot()
            if preflight.state != expected:
                raise SessionWriteConflict()
            descriptor, _ = self._proof(identity, preflight, locked=False)

            source = writer.store.with_artifact_store(reader.store)
            transaction = reader.store.transaction() if blobs is None else nullcontext(blobs)
            with transaction as active_blobs:
                active_blobs.assert_reachable()
                with source.rematerialization_snapshot(
                    expected.session_key, expected, owner=owner
                ) as snapshot:
                    current, content = self._proof(identity, snapshot, locked=True)
                    if current != descriptor:
                        raise ArtifactValidationError('Artifact descriptor changed')
                active_blobs.assert_reachable()

                installed = active_blobs.install_missing(descriptor, content)

                with source.rematerialization_snapshot(
                    expected.session_key, expected, owner=owner
                ) as snapshot:
                    current, final_content = self._proof(identity, snapshot, locked=True)
                    if current != descriptor or final_content != content:
                        raise ArtifactValidationError('Artifact source changed')
                    active_blobs.verify_durable(descriptor, store=reader.store)
                    active_blobs.assert_reachable()
                active_blobs.assert_reachable()
                owner.validate_dispatch()
                return ArtifactRematerializationResult(
                    installed.status, descriptor.reference, descriptor.byte_length
                )
        except ArtifactInstallCorruptionError:
            raise ArtifactRematerializationError('corrupt') from None
        except ArtifactInstallLimitError:
            raise ArtifactRematerializationError('limit') from None
        except ArtifactInstallUncertainError:
            raise ArtifactRematerializationError('uncertain') from None
        except ArtifactInstallIOError:
            raise ArtifactRematerializationError('io') from None
        except (
            ArtifactValidationError,
            ArtifactStorageError,
            ExecutionOwnershipError,
            SessionWriteConflict,
            SourceValidationError,
            FileNotFoundError,
            OSError,
        ):
            raise ArtifactRematerializationError('unavailable') from None


__all__ = [
    'ArtifactRematerializationError',
    'ArtifactRematerializationResult',
    'ArtifactRematerializer',
]
