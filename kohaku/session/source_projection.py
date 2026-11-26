"""Detached, identity-bearing source context for the context governor.

This module selects evidence only. It cannot summarize, write a checkpoint, or
acknowledge a provider response on the governor's behalf.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from kohaku.attachments.context import media_paths_for_message
from kohaku.runtime_context import model_history_message, public_history_message
from kohaku.session.source import (
    SourceValidationError,
    validate_closed_boundary,
    validate_event_ids,
    validate_execution_links,
)
from kohaku.session.source_store import SourceSnapshot
from kohaku.session.summary import conversation_summary_block, is_summary_checkpoint
from kohaku.utils.helpers import content_with_media_breadcrumbs


@dataclass(frozen=True)
class ArtifactProjectionBinding:
    artifact_id: str
    event_id: str
    execution_id: str
    content_sha256: str
    block_index: int | None = None


@dataclass(frozen=True)
class SourceNode:
    event_id: str
    message: dict[str, Any]
    artifact: ArtifactProjectionBinding | None = None
    block_artifacts: tuple[ArtifactProjectionBinding, ...] = ()

    @property
    def artifact_bindings(self) -> tuple[ArtifactProjectionBinding, ...]:
        return ((self.artifact,) if self.artifact is not None else ()) + self.block_artifacts


@dataclass(frozen=True)
class RuntimeNode:
    node_id: str
    message: dict[str, Any]


@dataclass(frozen=True)
class CheckpointNode:
    checkpoint_id: str
    summary: str
    artifact_references: tuple[str, ...] = ()

    def message(self) -> dict[str, Any]:
        return {
            "role": "user",
            "content": conversation_summary_block(self.summary, self.artifact_references),
        }


@dataclass(frozen=True)
class SourceProjection:
    """A root checkpoint and detached suffix, or one isolated invocation stream."""

    nodes: tuple[SourceNode, ...]
    checkpoint: CheckpointNode | None
    boundary: int
    pinned: SourceNode | None

    def messages(self) -> list[dict[str, Any]]:
        rows = [self.checkpoint.message()] if self.checkpoint else []
        if self.pinned is not None:
            rows.append(deepcopy(self.pinned.message))
        rows.extend(deepcopy(node.message) for node in self.nodes)
        return rows


def source_nodes(events: list[dict[str, Any]], *, stream_id: str | None = None) -> tuple[SourceNode, ...]:
    """Project roots by default; an explicit stream never includes its children."""
    selected = events
    if stream_id is not None:
        selected = [deepcopy(event) for event in events if event.get('_stream_id') == stream_id]
        # Root visibility is applied by the common public-message sanitizer.
        # An explicitly selected child stream is its own projection root.
        for event in selected:
            event.pop('_parent_stream_id', None)
    messages, identities = project_events(selected)
    return tuple(SourceNode(identity, message) for identity, message in zip(identities, messages))


def project_source(snapshot: SourceSnapshot, *, stream_id: str | None = None,
                   use_checkpoint: bool = True) -> SourceProjection:
    """Use only validated root-scoped checkpoints; never apply them to a stream.

    The newest user is replayed once if covered, including in a long active turn.
    Callers bind any authorized model-facing expansion separately by its event ID.
    """
    nodes = source_nodes(snapshot.events, stream_id=stream_id)
    checkpoint = snapshot.checkpoint if use_checkpoint and stream_id is None else None
    if (checkpoint is None or checkpoint.get('artifact_references', [])
            or checkpoint.get('projection_scope') != 'root'
            or type(checkpoint.get('projection_version')) is not int
            or checkpoint['projection_version'] != 1):
        return SourceProjection(nodes, None, 0, None)
    boundary = checkpoint['boundary']
    covered = {event['_event_id'] for event in snapshot.events[:boundary]}
    newest_user = next((node for node in reversed(nodes) if node.message['role'] == 'user'), None)
    pinned = newest_user if newest_user is not None and newest_user.event_id in covered else None
    return SourceProjection(
        tuple(node for node in nodes if node.event_id not in covered),
        CheckpointNode(checkpoint['checkpoint_id'], checkpoint['summary']), boundary, pinned,
    )


def eligible_boundaries(events: list[dict[str, Any]], *, eligible_ids: frozenset[str],
                        previous_boundary: int = 0, stream_id: str | None = None) -> tuple[int, ...]:
    """Return advancing closed prefixes; never cross an unacknowledged visible event.

    Root boundaries count every canonical event (including hidden child streams).
    Local boundaries count only the explicitly selected stream and cannot be used
    as durable checkpoint boundaries.
    """
    selected = events if stream_id is None else [
        event for event in events if event.get('_stream_id') == stream_id
    ]
    validate_event_ids(selected)
    validate_execution_links(selected)
    if type(previous_boundary) is not int or not 0 <= previous_boundary <= len(selected):
        raise SourceValidationError('Invalid previous projection boundary')
    visible = {node.event_id for node in source_nodes(selected, stream_id=stream_id)}
    boundaries: list[int] = []
    for boundary in range(previous_boundary + 1, len(selected) + 1):
        identity = selected[boundary - 1]['_event_id']
        if identity in visible and identity not in eligible_ids:
            break
        try:
            validate_closed_boundary(selected, boundary)
        except SourceValidationError:
            continue
        boundaries.append(boundary)
    return tuple(boundaries)


def project_events(
    events: list[dict[str, Any]], *, include_runtime_context: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Model replay for the root stream and its exact event identities."""
    messages: list[dict[str, Any]] = []
    identities: list[str] = []
    for event in events:
        if event.get('_parent_stream_id') is not None or event.get('_command'):
            continue
        if is_summary_checkpoint(event):
            continue
        projected = (
            model_history_message(event) if include_runtime_context else public_history_message(event)
        )
        row = {key: deepcopy(projected[key]) for key in
               ('role', 'content', 'tool_calls', 'tool_call_id', 'name') if key in projected}
        if row.get('role') not in {'user', 'assistant', 'tool'}:
            continue
        content = row.get('content')
        if isinstance(content, dict):
            row['content'] = json.dumps(content, ensure_ascii=False)
        row['content'] = content_with_media_breadcrumbs(
            row['role'], row.get('content'), media_paths_for_message(event) or None,
        )
        messages.append(row)
        identities.append(event['_event_id'])
    return messages, identities
