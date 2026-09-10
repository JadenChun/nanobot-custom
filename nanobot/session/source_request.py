"""Explicit source request manifests and validated-input acknowledgment.

The governor issues a ticket only after final budget validation. The runner
acknowledges that exact ticket only after response validation and mandatory
recording; issuing or rendering a request never grants eligibility by itself.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from nanobot.session.source import SourceValidationError
from nanobot.session.source_projection import (
    ArtifactProjectionBinding,
    CheckpointNode,
    RuntimeNode,
    SourceNode,
)

RequestNode = SourceNode | RuntimeNode | CheckpointNode


@dataclass(frozen=True)
class SourceTranscript:
    nodes: tuple[RequestNode, ...]

    def messages(self) -> list[dict[str, Any]]:
        return [node.message() if isinstance(node, CheckpointNode) else deepcopy(node.message)
                for node in self.nodes]

    def with_runtime(self, *nodes: RuntimeNode) -> SourceTranscript:
        """Request-only instructions do not mutate the underlying transcript."""
        return SourceTranscript(deepcopy((*self.nodes, *nodes)))


@dataclass(frozen=True)
class SourceRequestTicket:
    ticket_id: str
    source_ids: tuple[str, ...]
    checkpoint_id: str | None
    projection_epoch: int


class SourceRequestLedger:
    """Invocation-local provenance, isolated from other runs and child streams."""

    def __init__(self, transcript: SourceTranscript, *, initial_eligible_ids: frozenset[str]):
        self.validate_transcript(transcript)
        represented = {node.event_id for node in transcript.nodes if isinstance(node, SourceNode)}
        if not initial_eligible_ids <= represented:
            raise SourceValidationError('Initial eligibility requires represented source nodes')
        self._transcript = deepcopy(transcript)
        self._initial = frozenset(initial_eligible_ids)
        self._acknowledged: set[str] = set()
        self._epoch = 0
        self._pending: dict[str, SourceRequestTicket] = {}

    @staticmethod
    def validate_transcript(transcript: SourceTranscript) -> None:
        source_ids, runtime_ids = set(), set()
        checkpoints = 0
        for node in transcript.nodes:
            if isinstance(node, SourceNode):
                if not node.event_id or node.event_id in source_ids:
                    raise SourceValidationError('Duplicate or missing source request identity')
                if node.artifact is not None:
                    from nanobot.session.artifact import parse_artifact_reference

                    binding = node.artifact
                    if (type(binding) is not ArtifactProjectionBinding or binding.event_id != node.event_id
                            or not isinstance(binding.artifact_id, str)
                            or not isinstance(binding.execution_id, str) or not binding.execution_id
                            or binding.block_index is not None or not isinstance(binding.content_sha256, str)
                            or len(binding.content_sha256) != 64
                            or any(c not in '0123456789abcdef' for c in binding.content_sha256)):
                        raise SourceValidationError('Invalid source artifact projection binding')
                    parse_artifact_reference('artifact:' + binding.artifact_id)
                source_ids.add(node.event_id)
            elif isinstance(node, RuntimeNode):
                if not node.node_id or node.node_id in runtime_ids:
                    raise SourceValidationError('Duplicate or missing runtime request identity')
                runtime_ids.add(node.node_id)
            elif isinstance(node, CheckpointNode):
                checkpoints += 1
                if not node.checkpoint_id or not node.summary.strip() or checkpoints > 1:
                    raise SourceValidationError('Invalid request checkpoint')
            else:
                raise SourceValidationError('Unbound source request node')

    @property
    def transcript(self) -> SourceTranscript:
        return deepcopy(self._transcript)

    @property
    def projection_epoch(self) -> int:
        return self._epoch

    @property
    def has_pending_requests(self) -> bool:
        return bool(self._pending)

    @property
    def eligible_ids(self) -> frozenset[str]:
        return self._initial | frozenset(self._acknowledged)

    def append_source(self, event_id: str, message: dict[str, Any]) -> None:
        candidate = SourceTranscript((*self._transcript.nodes, SourceNode(event_id, deepcopy(message))))
        self.validate_transcript(candidate)
        self._transcript = candidate

    def append_runtime(self, message: dict[str, Any]) -> RuntimeNode:
        node = RuntimeNode(uuid.uuid4().hex, deepcopy(message))
        self._transcript = SourceTranscript((*self._transcript.nodes, node))
        return deepcopy(node)

    def validate_view(self, view: SourceTranscript) -> tuple[RuntimeNode, ...]:
        """Validate without issuing/acknowledging; return detached request-only notes."""
        self.validate_transcript(view)
        # Equality detects unauthorized edits; identities come from explicit
        # nodes, never from matching message content or transcript positions.
        if view.nodes[:len(self._transcript.nodes)] != self._transcript.nodes:
            raise SourceValidationError('Request view changed the active projection')
        extensions = []
        for node in view.nodes[len(self._transcript.nodes):]:
            if not isinstance(node, RuntimeNode):
                raise SourceValidationError('Request extensions must be runtime instructions')
            extensions.append(deepcopy(node))
        return tuple(extensions)

    def issue(self, view: SourceTranscript) -> SourceRequestTicket:
        """Capture the exact validated view; extensions may contain runtime nodes only."""
        self.validate_view(view)
        ticket = SourceRequestTicket(
            uuid.uuid4().hex,
            tuple(node.event_id for node in view.nodes if isinstance(node, SourceNode)),
            next((node.checkpoint_id for node in view.nodes if isinstance(node, CheckpointNode)), None),
            self._epoch,
        )
        self._pending[ticket.ticket_id] = ticket
        return ticket

    def acknowledge(self, ticket: SourceRequestTicket) -> None:
        """Consume one local ticket after validated success; output stays pending."""
        if (ticket.projection_epoch != self._epoch
                or self._pending.get(ticket.ticket_id) != ticket):
            raise SourceValidationError('Stale or foreign source request ticket')
        del self._pending[ticket.ticket_id]
        self._acknowledged.update(ticket.source_ids)

    def discard(self, ticket: SourceRequestTicket) -> None:
        """Failed/malformed requests cannot subsequently acknowledge their input."""
        if self._pending.get(ticket.ticket_id) != ticket:
            raise SourceValidationError('Stale or foreign source request ticket')
        del self._pending[ticket.ticket_id]

    def replace_projection(self, transcript: SourceTranscript) -> None:
        """Publish an already committed projection and invalidate outstanding views."""
        self.validate_transcript(transcript)
        self._transcript = deepcopy(transcript)
        self._epoch += 1
        self._pending.clear()
