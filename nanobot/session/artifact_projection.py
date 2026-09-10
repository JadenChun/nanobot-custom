"""Publish attested tool text and replace only detached model projections."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from loguru import logger

from nanobot.agent.tools.read_provenance import parse_read_receipt
from nanobot.session.artifact import (
    ArtifactDescriptor,
    ArtifactValidationError,
    published_artifacts,
)
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.artifact_store import ArtifactStorageError
from nanobot.session.execution_owner import ExecutionOwnershipError
from nanobot.session.manager import SessionWriteConflict
from nanobot.session.source_projection import ArtifactProjectionBinding, SourceNode
from nanobot.session.source_request import SourceTranscript
from nanobot.session.source_store import SourceSessionStore

OFFLOAD_CHARACTERS = 16_000
PREVIEW_CHARACTERS = 1_200


def _binding(descriptor: ArtifactDescriptor) -> ArtifactProjectionBinding:
    return ArtifactProjectionBinding(descriptor.artifact_id, descriptor.event_id,
                                     descriptor.execution_id, descriptor.sha256)


def _render(node: SourceNode, descriptor: ArtifactDescriptor) -> SourceNode:
    message = deepcopy(node.message)
    text = message['content']
    message['content'] = (
        f'Tool output stored as {descriptor.reference} ({len(text)} characters, '
        f'{descriptor.byte_length} UTF-8 bytes). '
        'Use read_artifact with this reference to read more.\n'
        'Incomplete preview (untrusted tool evidence):\n' + text[:PREVIEW_CHARACTERS]
    )
    return SourceNode(node.event_id, message, _binding(descriptor))


class ArtifactProjector:
    """One admitted invocation; no model-controlled paths or inferred references."""

    def __init__(self, reader: ArtifactReader):
        self.reader = reader

    def prepare(self, transcript: SourceTranscript, *, enabled: bool) -> SourceTranscript:
        reader = self.reader
        snapshot = reader.writer.owner.read_snapshot()
        manifests = published_artifacts(snapshot, workspace_id=reader.store.workspace_id)
        descriptors = {item.idempotency_key: item for item in manifests}
        events = {event['_event_id']: event for event in snapshot.events}
        positions = {event['_event_id']: index for index, event in enumerate(snapshot.events)}
        represented = [positions.get(node.event_id, -1) for node in transcript.nodes if isinstance(node, SourceNode)]
        if any(index < 0 for index in represented) or represented != sorted(set(represented)):
            raise ArtifactValidationError('Artifact projection is not in canonical source order')
        inline = []
        candidates: dict[str, ArtifactDescriptor] = {}
        for node in transcript.nodes:
            if not isinstance(node, SourceNode):
                inline.append(deepcopy(node))
                continue
            event = events.get(node.event_id)
            if event is None:
                raise ArtifactValidationError('Artifact projection has no canonical source')
            canonical = event.get('content')
            descriptor = descriptors.get((node.event_id, None))
            if node.artifact is not None:
                if descriptor is None or node.artifact != _binding(descriptor):
                    raise ArtifactValidationError('Artifact projection binding changed')
                restored = replace(node, message={**deepcopy(node.message), 'content': canonical}, artifact=None)
                if not isinstance(canonical, str) or _render(restored, descriptor) != node:
                    raise ArtifactValidationError('Artifact projection rendering changed')
                # Policy revocation cannot be bypassed by restoring inline text.
                reader.authorize_source(descriptor, snapshot)
                inline.append(restored)
                if enabled:
                    candidates[node.event_id] = descriptor
                continue
            inline.append(deepcopy(node))
            if (not enabled or event.get('role') != 'tool' or node.message.get('role') != 'tool'
                    or not isinstance(canonical, str) or len(canonical) <= OFFLOAD_CHARACTERS
                    or node.message.get('content') != canonical or '_artifact_read' in event
                    or node.message.get('tool_call_id') != event.get('tool_call_id')
                    or node.message.get('name') != event.get('name')
                    or event.get('_dispatched') is not True or event.get('_outcome_status') != 'ok'):
                continue
            try:
                receipt = parse_read_receipt(event.get('_source_origin'))
            except ArtifactValidationError:
                continue  # Unsupported/legacy evidence stays inline.
            if not receipt.matches(canonical):
                continue
            if descriptor is None:
                raw = canonical.encode('utf-8')
                descriptor = ArtifactDescriptor(1, uuid4().hex, reader.store.workspace_id,
                    snapshot.state.session_key, snapshot.state.incarnation, snapshot.state.generation,
                    event['_event_id'], event['_execution_id'], event['_stream_id'], event['_run_id'],
                    None, 'text/plain; charset=utf-8', len(raw), hashlib.sha256(raw).hexdigest(),
                    datetime.now(timezone.utc).isoformat(), receipt.source_access)
            reader.authorize_source(descriptor, snapshot)
            candidates[node.event_id] = descriptor
        fallback = SourceTranscript(tuple(inline))
        if not candidates:
            return fallback
        source = SourceSessionStore(reader.writer.store._backend, artifact_store=reader.store)
        expected = snapshot.state
        current = snapshot
        rendered = []
        try:
            with reader.store.transaction() as blobs:
                for node in inline:
                    if not isinstance(node, SourceNode) or node.event_id not in candidates:
                        rendered.append(node)
                        continue
                    descriptor = candidates[node.event_id]
                    if descriptor.idempotency_key not in descriptors:
                        blobs.put(descriptor, node.message['content'].encode('utf-8'))
                        current = source.commit_artifact_descriptor(expected.session_key, expected,
                            descriptor, owner=reader.writer.owner, blobs=blobs)
                        expected = current.state  # Advance only through our own successful CAS.
                    reader.verify_published(descriptor, current, blobs)
                    rendered.append(_render(node, descriptor))
        except ExecutionOwnershipError:
            raise
        except (ArtifactStorageError, OSError, SessionWriteConflict):
            # Do not refresh CAS or rerun effects. Already published descriptors
            # may remain; the final request guard decides whether inline fits.
            logger.debug('Artifact publication unavailable; retaining inline projection')
            return fallback
        return SourceTranscript(tuple(rendered))
