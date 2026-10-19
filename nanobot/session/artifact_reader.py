"""Invocation-bound access to attested immutable historical text."""

from __future__ import annotations

import codecs
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.read_provenance import FilesystemReadReceipt
from nanobot.config.paths import get_media_dir
from nanobot.security.workspace_access import current_tool_workspace
from nanobot.session.artifact import (
    ArtifactDescriptor,
    ArtifactValidationError,
    attested_artifact_origin,
    parse_artifact_reference,
    published_artifacts,
)
from nanobot.session.artifact_grants import ArtifactGrantSet
from nanobot.session.artifact_store import (
    MAX_RANGE_BYTES,
    ArtifactBlobTransaction,
    PrivateArtifactStore,
)

if TYPE_CHECKING:
    from nanobot.session.source_store import SourceSnapshot
    from nanobot.session.turn_writer import SourceTurnWriter


@dataclass(frozen=True)
class ArtifactReadResult:
    reference: str
    text: str
    start_offset: int
    next_offset: int
    eof: bool


@dataclass(frozen=True)
class ArtifactReader:
    writer: SourceTurnWriter
    mode: Literal['root_history', 'isolated_stream']
    store: PrivateArtifactStore
    read_file: ReadFileTool
    grants: ArtifactGrantSet | None = None
    location_validator: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (type(self.read_file) is not ReadFileTool
                or self.mode not in {'root_history', 'isolated_stream'}
                or (self.mode == 'root_history' and self.writer.parent_stream_id is not None)):
            raise ArtifactValidationError('Invalid artifact read admission')
        location = self.writer.store.select_artifact_location_metadata()
        self._validate_private_root()
        if (self.store.workspace_id != location.workspace_id
                or self.store.session_key != self.writer.owner.key):
            raise ArtifactValidationError('Artifact store does not match read admission')
        if self.grants is not None:
            self.grants.bind_reader(self)

    def _validate_private_root(self) -> None:
        if self.location_validator is not None:
            self.location_validator()
        location = self.writer.store.select_artifact_location_metadata()
        ReadFileTool.validate_artifact_private_location(
            reader=self,
            source_location=location,
            select_workspace_access=lambda: current_tool_workspace,
            media_root=lambda: get_media_dir(),
            validation_error=lambda message: ArtifactValidationError(message),
        )

    def revalidate_private_location(self) -> None:
        """Retain the original live private-root policy and its late-bound hook."""
        self._validate_private_root()

    def _native_permitted(self, descriptor: ArtifactDescriptor, snapshot: SourceSnapshot) -> bool:
        event = next((row for row in snapshot.events if row.get('_event_id') == descriptor.event_id), None)
        if event is None:
            return False
        if self.mode == 'root_history':
            return event.get('_parent_stream_id') is None
        return (
            descriptor.stream_id == self.writer.stream_id
            and event.get('_parent_stream_id') == self.writer.parent_stream_id
        )

    def _validate_grant_issuer(self) -> None:
        self._validate_private_root()
        self.writer.owner.validate_dispatch()
        if self.writer.failed:
            raise ArtifactValidationError('Artifact grant issuer is unavailable')

    def require_grant_issuer(self) -> None:
        """Recheck this invocation through the existing late-bound issuer hook."""
        self._validate_grant_issuer()

    def _descriptor_for_id(self, identity: str, snapshot: SourceSnapshot) -> ArtifactDescriptor:
        descriptor = next((item for item in published_artifacts(
            snapshot, workspace_id=self.store.workspace_id
        ) if item.artifact_id == identity), None)
        if descriptor is None:
            raise ArtifactValidationError('Artifact is unavailable')
        return descriptor

    def _verify_descriptor(
        self,
        descriptor: ArtifactDescriptor,
        snapshot: SourceSnapshot,
        blobs: ArtifactBlobTransaction,
        *,
        require_native: bool,
    ) -> FilesystemReadReceipt:
        """Recheck all current evidence and policy without inferring a grant."""
        self._validate_private_root()
        writer = self.writer
        writer.owner.validate_dispatch()
        current = writer.owner.read_snapshot().state
        if ((current.store_identity, current.session_key, current.incarnation, current.generation)
                != (snapshot.state.store_identity, snapshot.state.session_key,
                    snapshot.state.incarnation, snapshot.state.generation)
                or descriptor.incarnation != self.store.incarnation
                or descriptor.generation != self.store.generation
                or descriptor.session_key != self.store.session_key):
            raise ArtifactValidationError('Artifact source admission changed')
        if require_native and not self._native_permitted(descriptor, snapshot):
            raise ArtifactValidationError('Artifact is outside the admitted stream')
        # Owner, invocation stream, and private-root checks above remain
        # invocation-local. The shared helper only attests source receipts,
        # current policy, manifest membership, and durable blob integrity.
        return self.read_file.verify_artifact_evidence(
            descriptor,
            snapshot,
            blobs,
            store=self.store,
        )

    def authorize_source(self, descriptor: ArtifactDescriptor, snapshot: SourceSnapshot) -> FilesystemReadReceipt:
        """Validate producer/scope/policy before publication or inline fallback."""
        return self._verify_native_source_locked(descriptor, snapshot)

    def _verify_native_source_locked(
        self, descriptor: ArtifactDescriptor, snapshot: SourceSnapshot
    ) -> FilesystemReadReceipt:
        """Prove native canonical source using an already-locked exact snapshot."""
        self._validate_grant_issuer()
        if not self._native_permitted(descriptor, snapshot):
            raise ArtifactValidationError('Artifact is outside the admitted stream')
        receipt = attested_artifact_origin(descriptor, snapshot, workspace_id=self.store.workspace_id)
        self.read_file.revalidate_artifact_source(receipt)
        return receipt

    def verify_native_source_in_snapshot(
        self, descriptor: ArtifactDescriptor, snapshot: SourceSnapshot,
    ) -> FilesystemReadReceipt:
        """Reprove real native source under the caller's retained source transaction."""
        return self._verify_native_source_locked(descriptor, snapshot)

    def verify_published(self, descriptor: ArtifactDescriptor, snapshot: SourceSnapshot,
                          blobs: ArtifactBlobTransaction) -> None:
        self._verify_descriptor(descriptor, snapshot, blobs, require_native=True)

    def verify_received_descriptor(
        self, descriptor: ArtifactDescriptor, snapshot: SourceSnapshot,
        blobs: ArtifactBlobTransaction,
    ) -> None:
        """Check received evidence only; this neither grants access nor returns text."""
        self._verify_descriptor(descriptor, snapshot, blobs, require_native=False)

    def verify_native_artifact(self, identity: str, snapshot: SourceSnapshot,
                               blobs: ArtifactBlobTransaction) -> ArtifactDescriptor:
        """Resolve and verify native evidence; this never issues a grant or reads text."""
        descriptor = self._descriptor_for_id(identity, snapshot)
        self._verify_descriptor(descriptor, snapshot, blobs, require_native=True)
        return descriptor

    def read(self, reference: str, *, offset: int = 0, max_bytes: int = 8192) -> ArtifactReadResult:
        self._validate_private_root()
        identity = parse_artifact_reference(reference)
        if (type(offset) is not int or offset < 0 or type(max_bytes) is not int
                or not 4 <= max_bytes <= MAX_RANGE_BYTES):
            raise ArtifactValidationError('Invalid artifact text range')
        writer, owner = self.writer, self.writer.owner
        owner.validate_dispatch()
        with self.store.transaction() as blobs:
            with writer.store.execution_transaction(owner.key, owner):
                owner.validate_dispatch()
                snapshot = owner.read_snapshot()
                descriptor = self._descriptor_for_id(identity, snapshot)
                if self._native_permitted(descriptor, snapshot):
                    self._verify_descriptor(descriptor, snapshot, blobs, require_native=True)
                elif self.grants is not None:
                    self.grants.authorize(self, descriptor, snapshot, blobs)
                else:
                    raise ArtifactValidationError('Artifact is outside the admitted stream')
                raw = blobs.read_range(descriptor, offset=offset, limit=max_bytes)
                # Skip an initial partial character, retain the trailing partial
                # character for the next page. A four-byte minimum ensures progress.
                leading = 0
                while leading < len(raw) and raw[leading] & 0xC0 == 0x80:
                    leading += 1
                decoder = codecs.getincrementaldecoder('utf-8')()
                text = decoder.decode(raw[leading:], final=False)
                trailing = len(decoder.getstate()[0])
                start = min(offset + leading, descriptor.byte_length)
                end = min(offset + len(raw) - trailing, descriptor.byte_length)
                blobs.assert_reachable()
                owner.validate_dispatch()
                return ArtifactReadResult(reference, text, start, end, end == descriptor.byte_length)
