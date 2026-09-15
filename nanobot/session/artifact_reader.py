"""Invocation-bound access to attested immutable historical text."""

from __future__ import annotations

import codecs
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.read_provenance import FilesystemReadReceipt, revalidate_read_receipt
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
from nanobot.session.artifact_verification import verify_published_artifact
from nanobot.session.execution_owner import execution_transaction

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
        backend = self.writer.store._backend
        self._validate_private_root()
        if (self.store.workspace_id != backend.sessions_dir.name
                or self.store.session_key != self.writer.owner.key):
            raise ArtifactValidationError('Artifact store does not match read admission')
        if self.grants is not None:
            self.grants.bind_reader(self)

    def _validate_private_root(self) -> None:
        if self.location_validator is not None:
            self.location_validator()
        backend = self.writer.store._backend
        access = current_tool_workspace(self.read_file._workspace,
            restrict_to_workspace=self.read_file._restrict_to_workspace,
            sandbox_restricts_workspace=self.read_file._sandbox_restricts_workspace)
        roots = [backend.workspace.resolve(), get_media_dir().resolve(),
                 *[root.resolve() for root in self.read_file._extra_read_allowed_dirs]]
        if self.read_file._workspace is not None:
            roots.append(self.read_file._workspace.resolve())
        if self.read_file._resource_policy is not None:
            roots.extend(root.resolve() for root in self.read_file._resource_policy.extra_read_dirs())
        if any(self.store.root == root or self.store.root.is_relative_to(root) for root in roots):
            raise ArtifactValidationError('Artifact store must be outside readable roots')
        if access.project_path is not None:
            root = access.project_path.resolve()
            if self.store.root == root or self.store.root.is_relative_to(root):
                raise ArtifactValidationError('Artifact store is inside the active project')
        if any(path.resolve().is_relative_to(self.store.root)
               for path in self.read_file._extra_read_allowed_files):
            raise ArtifactValidationError('Artifact store has a public file allowlist')

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
        access = current_tool_workspace(
            self.read_file._workspace,
            restrict_to_workspace=self.read_file._restrict_to_workspace,
            sandbox_restricts_workspace=self.read_file._sandbox_restricts_workspace,
        )
        # Owner, invocation stream, and private-root checks above remain
        # invocation-local. The shared helper only attests source receipts,
        # current policy, manifest membership, and durable blob integrity.
        return verify_published_artifact(
            descriptor,
            snapshot,
            blobs,
            store=self.store,
            workspace=access.project_path,
            policy=self.read_file._resource_policy,
            resolve_read=self.read_file._resolve_read,
        )

    def authorize_source(self, descriptor: ArtifactDescriptor, snapshot: SourceSnapshot) -> FilesystemReadReceipt:
        """Validate producer/scope/policy before publication or inline fallback."""
        self._validate_grant_issuer()
        if not self._native_permitted(descriptor, snapshot):
            raise ArtifactValidationError('Artifact is outside the admitted stream')
        receipt = attested_artifact_origin(descriptor, snapshot, workspace_id=self.store.workspace_id)
        access = current_tool_workspace(
            self.read_file._workspace,
            restrict_to_workspace=self.read_file._restrict_to_workspace,
            sandbox_restricts_workspace=self.read_file._sandbox_restricts_workspace,
        )
        revalidate_read_receipt(
            receipt,
            workspace=access.project_path,
            policy=self.read_file._resource_policy,
            resolve_read=self.read_file._resolve_read,
        )
        return receipt

    def verify_published(self, descriptor: ArtifactDescriptor, snapshot: SourceSnapshot,
                         blobs: ArtifactBlobTransaction) -> None:
        self._verify_descriptor(descriptor, snapshot, blobs, require_native=True)

    def read(self, reference: str, *, offset: int = 0, max_bytes: int = 8192) -> ArtifactReadResult:
        self._validate_private_root()
        identity = parse_artifact_reference(reference)
        if (type(offset) is not int or offset < 0 or type(max_bytes) is not int
                or not 4 <= max_bytes <= MAX_RANGE_BYTES):
            raise ArtifactValidationError('Invalid artifact text range')
        writer, owner = self.writer, self.writer.owner
        owner.validate_dispatch()
        with self.store.transaction() as blobs:
            with execution_transaction(writer.store._backend, owner.key, owner):
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
