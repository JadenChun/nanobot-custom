"""Invocation checkpoint admission and reversible dependency validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nanobot.session.artifact import ArtifactValidationError, published_artifacts
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.source_projection import CheckpointNode, SourceNode
from nanobot.session.source_request import SourceTranscript
from nanobot.session.source_store import SourceSnapshot

if TYPE_CHECKING:
    from nanobot.session.artifact import ArtifactDescriptor
    from nanobot.session.source_request import RequestNode


@dataclass
class CheckpointAdmission:
    initialized: bool = False
    boundary: int = 0
    checkpoint: CheckpointNode | None = None
    fingerprint: str | None = None
    rejected: set[str] = field(default_factory=set)
    reader: ArtifactReader | None = None

    def validate_dependencies(self, snapshot: SourceSnapshot, active: SourceTranscript, *,
                              enabled: bool, checkpoint: CheckpointNode | None = None) -> bool:
        """Authorize all references before allowing any storage fallback."""
        checkpoint = checkpoint or self.checkpoint
        references = set(checkpoint.artifact_references if checkpoint else ())
        bindings = [node.artifact for node in active.nodes
                    if isinstance(node, SourceNode) and node.artifact is not None]
        references.update('artifact:' + binding.artifact_id for binding in bindings)
        if not references:
            return True
        reader = self.reader
        if reader is None:
            raise ArtifactValidationError('Checkpoint dependencies lack an admitted verifier')
        manifests = {item.reference: item for item in
                     published_artifacts(snapshot, workspace_id=reader.store.workspace_id)}
        descriptors: list[ArtifactDescriptor] = []
        for reference in sorted(references):
            descriptor = manifests.get(reference)
            if descriptor is None:
                raise ArtifactValidationError('Checkpoint dependency is no longer published')
            reader.authorize_source(descriptor, snapshot)
            descriptors.append(descriptor)
        for binding in bindings:
            descriptor = manifests['artifact:' + binding.artifact_id]
            if (binding.event_id, binding.execution_id, binding.content_sha256, binding.block_index) != (
                    descriptor.event_id, descriptor.execution_id, descriptor.sha256, descriptor.block_index):
                raise ArtifactValidationError('Active artifact binding changed')
        if not enabled:
            return False
        try:
            with reader.store.transaction() as blobs:
                for descriptor in descriptors:
                    reader.verify_published(descriptor, snapshot, blobs)
        except OSError:
            return False
        return True

    def select(self, snapshot: SourceSnapshot, full: SourceTranscript) -> SourceTranscript:
        """Select once, after reader registration, preserving request expansions."""
        self.initialized = True
        stored = snapshot.checkpoint
        if (stored is None or snapshot.state.checkpoint_fingerprint in self.rejected
                or stored.get('projection_scope') != 'root'
                or type(stored.get('projection_version')) is not int
                or stored['projection_version'] != 1):
            return full
        references = tuple(stored['artifact_references'])
        if references and self.reader is None:
            self.rejected.add(snapshot.state.checkpoint_fingerprint)
            return full
        checkpoint = CheckpointNode(stored['checkpoint_id'], stored['summary'], references)
        self.checkpoint = checkpoint
        self.boundary = stored['boundary']
        self.fingerprint = snapshot.state.checkpoint_fingerprint
        if not self.validate_dependencies(snapshot, full, enabled=True):
            self.reject()
            return full
        covered = {event['_event_id'] for event in snapshot.events[:self.boundary]}
        newest = next((event['_event_id'] for event in reversed(snapshot.events)
                       if event.get('role') == 'user' and event.get('_parent_stream_id') is None
                       and not event.get('_command')), None)
        nodes: list[RequestNode] = []
        inserted = False
        for node in full.nodes:
            if isinstance(node, SourceNode) and node.event_id in covered:
                if not inserted:
                    nodes.append(checkpoint)
                    inserted = True
                if node.event_id != newest:
                    continue
            nodes.append(deepcopy(node))
        if not inserted:
            self.reject()
            return full
        return SourceTranscript(tuple(nodes))

    def reject(self) -> None:
        if self.fingerprint is not None:
            self.rejected.add(self.fingerprint)
        self.boundary, self.checkpoint, self.fingerprint = 0, None, None
