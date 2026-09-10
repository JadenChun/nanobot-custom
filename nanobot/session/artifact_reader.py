"""Invocation-bound access to attested immutable historical text."""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.agent.tools.read_provenance import revalidate_read_receipt
from nanobot.config.paths import get_media_dir
from nanobot.security.workspace_access import current_tool_workspace
from nanobot.session.artifact import (
    ArtifactValidationError,
    attested_artifact_origin,
    parse_artifact_reference,
    published_artifacts,
)
from nanobot.session.artifact_store import MAX_RANGE_BYTES, PrivateArtifactStore
from nanobot.session.execution_owner import execution_transaction

if TYPE_CHECKING:
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

    def _validate_private_root(self) -> None:
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
                descriptors = published_artifacts(snapshot, workspace_id=self.store.workspace_id)
                descriptor = next((item for item in descriptors if item.artifact_id == identity), None)
                if descriptor is None:
                    raise ArtifactValidationError('Artifact is unavailable')
                receipt = attested_artifact_origin(descriptor, snapshot,
                                                   workspace_id=self.store.workspace_id)
                event = next(row for row in snapshot.events if row['_event_id'] == descriptor.event_id)
                if self.mode == 'root_history':
                    permitted = event.get('_parent_stream_id') is None
                else:
                    permitted = (descriptor.stream_id == writer.stream_id
                                 and event.get('_parent_stream_id') == writer.parent_stream_id)
                if not permitted:
                    raise ArtifactValidationError('Artifact is outside the admitted stream')
                access = current_tool_workspace(self.read_file._workspace,
                    restrict_to_workspace=self.read_file._restrict_to_workspace,
                    sandbox_restricts_workspace=self.read_file._sandbox_restricts_workspace)
                revalidate_read_receipt(receipt, workspace=access.project_path,
                    policy=self.read_file._resource_policy, resolve_read=self.read_file._resolve_read)
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
