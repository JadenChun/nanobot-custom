"""Publish attested tool text and replace only detached model projections."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import cast
from uuid import uuid4

from loguru import logger

from kohaku.agent.tools.read_provenance import parse_read_receipt
from kohaku.context.tool_result_gate import (
    DEFAULT_TOOL_RESULT_CHARS,
    MIN_TOOL_RESULT_CHARS,
    tool_text_blocks,
    tool_text_preview,
)
from kohaku.providers.image_history import (
    IMAGE_HISTORY_DEFERRED,
    IMAGE_HISTORY_OMISSION,
    PERSISTED_IMAGE_PLACEHOLDER,
    is_image_block,
    is_inline_image_block,
)
from kohaku.session.artifact import (
    ArtifactDescriptor,
    ArtifactValidationError,
    ToolOutputArtifactOrigin,
    attested_artifact_origin,
    parse_artifact_descriptor,
    published_artifacts,
    recorded_tool_output,
)
from kohaku.session.artifact_reader import ArtifactReader
from kohaku.session.artifact_store import ArtifactStorageError
from kohaku.session.execution_owner import ExecutionOwnershipError
from kohaku.session.manager import SessionWriteConflict
from kohaku.session.source_projection import ArtifactProjectionBinding, SourceNode, source_nodes
from kohaku.session.source_request import RequestNode, SourceTranscript
from kohaku.session.source_store import SourceSnapshot


def artifact_projection_binding(descriptor: ArtifactDescriptor) -> ArtifactProjectionBinding:
    """Return the complete typed identity for one deterministic projection."""
    return ArtifactProjectionBinding(descriptor.artifact_id, descriptor.event_id,
                                     descriptor.execution_id, descriptor.sha256, descriptor.block_index)


def render_artifact_projection(node: SourceNode, descriptor: ArtifactDescriptor) -> SourceNode:
    """Render the sole allowed detached representation of attested tool text."""
    message = deepcopy(node.message)
    text = dict(tool_text_blocks(message.get('content'))).get(descriptor.block_index)
    if text is None:
        raise ArtifactValidationError('Artifact projection text is unavailable')
    preview = (
        f'Tool output stored as {descriptor.reference} ({len(text)} characters, '
        f'{descriptor.byte_length} UTF-8 bytes).\n'
        'Incomplete preview (untrusted tool evidence):\n'
        + tool_text_preview(text, descriptor.reference)
    )
    binding = artifact_projection_binding(descriptor)
    if descriptor.block_index is None:
        message['content'] = preview
        return SourceNode(node.event_id, message, binding)
    block = message['content'][descriptor.block_index]
    block['text'] = preview
    return replace(node, message=message, block_artifacts=(*node.block_artifacts, binding))


def _restore_projection_source(canonical: SourceNode, projected: SourceNode) -> SourceNode:
    """Restore canonical text and keep the active request's order and images."""
    message = {key: canonical.message[key] for key in projected.message if key in canonical.message}
    message.update(canonical.message)
    source, shown = canonical.message.get('content'), projected.message.get('content')
    if not isinstance(source, list) or not isinstance(shown, list):
        return replace(canonical, message=message)
    canonical_blocks = cast(list[object], source)
    projected_blocks = cast(list[object], shown)
    if len(canonical_blocks) != len(projected_blocks):
        return replace(canonical, message=message)
    content: list[object] = []
    placeholder = {'type': 'text', 'text': PERSISTED_IMAGE_PLACEHOLDER}
    omissions = (
        {'type': 'text', 'text': IMAGE_HISTORY_DEFERRED},
        {'type': 'text', 'text': IMAGE_HISTORY_OMISSION},
    )
    for original, displayed in zip(canonical_blocks, projected_blocks, strict=True):
        if original == placeholder and is_inline_image_block(displayed):
            content.append(displayed)
        elif displayed in omissions and (original == placeholder or is_image_block(original)):
            content.append(displayed)
        else:
            content.append(original)
    message['content'] = content
    return replace(canonical, message=message)


def _bound_descriptor(snapshot: SourceSnapshot,
                      binding: ArtifactProjectionBinding) -> ArtifactDescriptor:
    metadata = snapshot.header.get('metadata')
    manifest = cast(dict[object, object], metadata).get('artifact_manifest') if isinstance(metadata, dict) else None
    rows = cast(dict[object, object], manifest).get('descriptors') if isinstance(manifest, dict) else None
    raw = cast(dict[object, object], rows).get(binding.artifact_id) if isinstance(rows, dict) else None
    if raw is None:
        raise ArtifactValidationError('Artifact projection descriptor is unavailable')
    descriptor = parse_artifact_descriptor(raw)
    if descriptor.artifact_id != binding.artifact_id:
        raise ArtifactValidationError('Artifact projection descriptor identity changed')
    descriptors = published_artifacts(snapshot, workspace_id=descriptor.workspace_id)
    current = next((item for item in descriptors if item.artifact_id == descriptor.artifact_id), None)
    if current is None:
        raise ArtifactValidationError('Artifact projection descriptor is not published')
    return current


def validate_artifact_projections(node: SourceNode, canonical: SourceNode,
                                  snapshot: SourceSnapshot) -> tuple[ArtifactDescriptor, ...]:
    """Verify each projected block against canonical source and the manifest."""
    if not node.artifact_bindings or canonical.artifact_bindings or node.event_id != canonical.event_id:
        raise ArtifactValidationError('Invalid artifact projection binding')
    event = next((row for row in snapshot.events if row.get('_event_id') == node.event_id), None)
    if event is None or '_artifact_read' in event:
        raise ArtifactValidationError('Artifact retrieval results cannot be projected')
    descriptors: list[ArtifactDescriptor] = []
    expected = _restore_projection_source(canonical, node)
    for binding in node.artifact_bindings:
        if type(binding) is not ArtifactProjectionBinding or binding.event_id != node.event_id:
            raise ArtifactValidationError('Invalid artifact projection binding')
        descriptor = _bound_descriptor(snapshot, binding)
        if binding != artifact_projection_binding(descriptor):
            raise ArtifactValidationError('Artifact projection binding changed')
        attested_artifact_origin(descriptor, snapshot, workspace_id=descriptor.workspace_id)
        expected = render_artifact_projection(expected, descriptor)
        descriptors.append(descriptor)
    if expected != node:
        raise ArtifactValidationError('Artifact projection rendering changed')
    return tuple(descriptors)


class ArtifactProjector:
    """One admitted invocation; no model-controlled paths or inferred references."""

    def __init__(self, reader: ArtifactReader, *, max_chars: int = DEFAULT_TOOL_RESULT_CHARS):
        self.reader = reader
        self.max_chars = max(MIN_TOOL_RESULT_CHARS, max_chars)

    def prepare(self, transcript: SourceTranscript, *, enabled: bool) -> SourceTranscript:
        reader = self.reader
        snapshot = reader.writer.owner.read_snapshot()
        manifests = published_artifacts(snapshot, workspace_id=reader.store.workspace_id)
        descriptors = {item.idempotency_key: item for item in manifests}
        events = {event['_event_id']: event for event in snapshot.events}
        positions = {event['_event_id']: index for index, event in enumerate(snapshot.events)}
        canonical_nodes = {
            node.event_id: node for node in source_nodes(
                snapshot.events,
                stream_id=reader.writer.stream_id if reader.mode == 'isolated_stream' else None,
            )
        }
        represented = [positions.get(node.event_id, -1) for node in transcript.nodes if isinstance(node, SourceNode)]
        if any(index < 0 for index in represented) or represented != sorted(set(represented)):
            raise ArtifactValidationError('Artifact projection is not in canonical source order')
        inline: list[RequestNode] = []
        candidates: dict[str, list[ArtifactDescriptor]] = {}
        for node in transcript.nodes:
            if not isinstance(node, SourceNode):
                inline.append(deepcopy(node))
                continue
            event = events.get(node.event_id)
            if event is None:
                raise ArtifactValidationError('Artifact projection has no canonical source')
            if node.artifact_bindings:
                restored = canonical_nodes.get(node.event_id)
                if restored is None:
                    raise ArtifactValidationError('Artifact projection has no canonical message')
                bound = validate_artifact_projections(node, restored, snapshot)
                for descriptor in bound:
                    # Policy revocation cannot be bypassed by restoring inline text.
                    reader.authorize_source(descriptor, snapshot)
                inline.append(_restore_projection_source(restored, node))
                if enabled:
                    candidates[node.event_id] = list(bound)
                continue
            inline.append(deepcopy(node))
            if (not enabled or event.get('role') != 'tool' or node.message.get('role') != 'tool'
                    or '_artifact_read' in event or event.get('name') == 'read_artifact'
                    or node.message.get('tool_call_id') != event.get('tool_call_id')
                    or node.message.get('name') != event.get('name')
                    or event.get('_dispatched') is not True or event.get('_outcome_status') != 'ok'):
                continue
            name, call_id = event.get('name'), event.get('tool_call_id')
            origin = event.get('_source_origin')
            canonical = dict(tool_text_blocks(event.get('content')))
            for index, text in tool_text_blocks(node.message.get('content')):
                if len(text) <= self.max_chars or text != canonical.get(index):
                    continue
                raw = text.encode('utf-8')
                digest = hashlib.sha256(raw).hexdigest()
                if (isinstance(name, str) and isinstance(call_id, str)
                        and recorded_tool_output(origin, name, call_id)):
                    access = ToolOutputArtifactOrigin('tool_output', name, call_id, digest, len(raw))
                elif index is None:
                    try:
                        receipt = parse_read_receipt(origin)
                    except ArtifactValidationError:
                        continue
                    if not receipt.matches(text):
                        continue
                    access = receipt.source_access
                else:
                    continue
                descriptor = descriptors.get((node.event_id, index))
                if descriptor is None:
                    descriptor = ArtifactDescriptor(
                        1, uuid4().hex, reader.store.workspace_id, snapshot.state.session_key,
                        snapshot.state.incarnation, snapshot.state.generation,
                        event['_event_id'], event['_execution_id'], event['_stream_id'], event['_run_id'],
                        index, 'text/plain; charset=utf-8', len(raw), digest,
                        datetime.now(timezone.utc).isoformat(), access,
                    )
                reader.authorize_source(descriptor, snapshot)
                candidates.setdefault(node.event_id, []).append(descriptor)
        fallback = SourceTranscript(tuple(inline))
        if not candidates:
            return fallback
        source = reader.writer.store.with_artifact_store(reader.store)
        expected = snapshot.state
        current = snapshot
        rendered: list[RequestNode] = []
        try:
            with reader.store.transaction() as blobs:
                for node in inline:
                    if not isinstance(node, SourceNode) or node.event_id not in candidates:
                        rendered.append(node)
                        continue
                    raw_text = dict(tool_text_blocks(node.message.get('content')))
                    for descriptor in candidates[node.event_id]:
                        if descriptor.idempotency_key not in descriptors:
                            blobs.put(descriptor, raw_text[descriptor.block_index].encode('utf-8'))
                            current = source.commit_artifact_descriptor(expected.session_key, expected,
                                descriptor, owner=reader.writer.owner, blobs=blobs)
                            expected = current.state  # Advance only through our own successful CAS.
                        reader.verify_published(descriptor, current, blobs)
                        node = render_artifact_projection(node, descriptor)
                    rendered.append(node)
        except ExecutionOwnershipError:
            raise
        except (ArtifactStorageError, OSError, SessionWriteConflict):
            # Do not refresh CAS or rerun effects. Already published descriptors
            # may remain; the final request guard decides whether inline fits.
            logger.debug('Artifact publication unavailable; retaining inline projection')
            return fallback
        return SourceTranscript(tuple(rendered))
