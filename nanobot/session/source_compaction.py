"""Captured source evidence and request rebuilding for the context governor.

These pure helpers neither reduce by token count nor call providers or storage.
The governor selects a legal boundary, validates the rebuilt payload, and commits
against the captured state before publishing a replacement ledger projection.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nanobot.session.source import SourceValidationError
from nanobot.session.source_projection import (
    CheckpointNode,
    SourceNode,
    eligible_boundaries,
    source_nodes,
)
from nanobot.session.source_request import SourceTranscript
from nanobot.session.source_store import SourceSnapshot

if TYPE_CHECKING:
    from nanobot.session.checkpoint_dependencies import SummaryEvidence


@dataclass(frozen=True)
class SourceCompactionCandidate:
    snapshot: SourceSnapshot
    transcript: SourceTranscript
    stream_id: str | None
    previous_boundary: int
    previous_checkpoint: CheckpointNode | None
    boundaries: tuple[int, ...]
    pinned_event_id: str | None

    @classmethod
    def capture(cls, snapshot: SourceSnapshot, transcript: SourceTranscript, *,
                eligible_ids: frozenset[str], stream_id: str | None = None,
                local_boundary: int = 0,
                local_checkpoint: CheckpointNode | None = None,
                accepted_boundary: int = 0,
                accepted_checkpoint: CheckpointNode | None = None) -> SourceCompactionCandidate:
        """Capture one exact source revision, with global or stream-local closure.

        Local state is invocation-owned and must never become a durable root
        boundary. Root state is explicitly admitted by the current invocation.
        """
        snapshot, transcript = deepcopy(snapshot), deepcopy(transcript)
        if stream_id is None:
            if local_boundary != 0 or local_checkpoint is not None:
                raise SourceValidationError('Root compaction cannot use local checkpoint state')
            previous_boundary, previous_checkpoint = accepted_boundary, deepcopy(accepted_checkpoint)
        else:
            if not stream_id:
                raise SourceValidationError('Local compaction requires a stream identity')
            previous_boundary, previous_checkpoint = local_boundary, deepcopy(local_checkpoint)
            if bool(previous_boundary) != (previous_checkpoint is not None):
                raise SourceValidationError('Local boundary and checkpoint must advance together')
        if bool(previous_boundary) != (previous_checkpoint is not None):
            raise SourceValidationError('Boundary and checkpoint must advance together')
        if type(previous_boundary) is not int or previous_boundary < 0:
            raise SourceValidationError('Invalid accepted compaction boundary')
        represented_checkpoints = tuple(node for node in transcript.nodes if isinstance(node, CheckpointNode))
        expected = (previous_checkpoint,) if previous_checkpoint is not None else ()
        if represented_checkpoints != expected:
            raise SourceValidationError('Active projection does not match captured checkpoint')
        canonical_nodes = source_nodes(snapshot.events, stream_id=stream_id)
        visible = {node.event_id for node in canonical_nodes}
        represented = [node.event_id for node in transcript.nodes if isinstance(node, SourceNode)]
        if len(represented) != len(set(represented)) or not set(represented) <= visible:
            raise SourceValidationError('Active projection contains foreign or duplicate source identities')
        selected = snapshot.events if stream_id is None else [
            event for event in snapshot.events if event.get('_stream_id') == stream_id
        ]
        suffix_ids = {node.event_id for node in source_nodes(selected[previous_boundary:], stream_id=stream_id)}
        newest_user = next((node for node in reversed(canonical_nodes)
                            if node.message.get('role') == 'user'), None)
        pinned_event_id = newest_user.event_id if newest_user is not None else None
        expected_ids = [node.event_id for node in canonical_nodes
                        if node.event_id in suffix_ids or node.event_id == pinned_event_id]
        if represented != expected_ids:
            raise SourceValidationError('Active projection omits or reorders canonical source evidence')
        boundaries = eligible_boundaries(snapshot.events, eligible_ids=eligible_ids,
            previous_boundary=previous_boundary, stream_id=stream_id)
        return cls(snapshot, transcript, stream_id, previous_boundary, previous_checkpoint, boundaries, pinned_event_id)

    def _events(self) -> list[dict[str, Any]]:
        return self.snapshot.events if self.stream_id is None else [
            event for event in self.snapshot.events if event.get('_stream_id') == self.stream_id
        ]

    def evidence(self, boundary: int, *, after: int | None = None) -> tuple[SourceNode, ...]:
        """Return canonical disjoint evidence; never use expanded request text."""
        self._require_boundary(boundary)
        if self.previous_checkpoint is not None and self.previous_checkpoint.artifact_references:
            raise SourceValidationError('Artifact checkpoint summary dependencies are not implemented')
        start = self.previous_boundary if after is None else after
        if start != self.previous_boundary and start not in self.boundaries:
            raise SourceValidationError('Evidence must start at a closed eligible boundary')
        if start >= boundary:
            raise SourceValidationError('Evidence chunk must advance its boundary')
        evidence = self._events()[start:boundary]
        if any('_artifact_read' in event for event in evidence):
            raise SourceValidationError('Artifact retrieval summary dependencies are not implemented')
        return source_nodes(evidence, stream_id=self.stream_id)

    def summary_evidence(self, boundary: int, *, after: int | None = None) -> SummaryEvidence:
        """Capture canonical or verified projected evidence for one summary fold."""
        from nanobot.agent.tools.artifact import parse_retrieval_receipt
        from nanobot.session.artifact_projection import validate_artifact_projection
        from nanobot.session.checkpoint_dependencies import (
            ProjectedArtifactDependency,
            RetrievalDependency,
            SummaryEvidence,
        )

        self._require_boundary(boundary)
        start = self.previous_boundary if after is None else after
        if start != self.previous_boundary and start not in self.boundaries:
            raise SourceValidationError('Evidence must start at a closed eligible boundary')
        if start >= boundary:
            raise SourceValidationError('Evidence chunk must advance its boundary')
        events = self._events()[start:boundary]
        canonical_nodes = source_nodes(events, stream_id=self.stream_id)
        active = {node.event_id: node for node in self.transcript.nodes if isinstance(node, SourceNode)}
        nodes: list[SourceNode] = []
        projections: list[ProjectedArtifactDependency] = []
        for canonical in canonical_nodes:
            node = active.get(canonical.event_id)
            if node is None:
                raise SourceValidationError('Summary evidence omits canonical source')
            if node.artifact is None:
                # Request-only expansions never enter the proof. The legacy
                # canonical node remains the only summary input in this case.
                nodes.append(canonical)
                continue
            validate_artifact_projection(node, canonical, self.snapshot)
            projections.append(ProjectedArtifactDependency(node.event_id, node.artifact))
            nodes.append(node)
        # Visibility is established before inspecting reserved retrieval metadata.
        visible = {node.event_id for node in canonical_nodes}
        retrievals: list[RetrievalDependency] = []
        for event in events:
            if event['_event_id'] not in visible or '_artifact_read' not in event:
                continue
            try:
                receipt = parse_retrieval_receipt(event['_artifact_read'])
            except (TypeError, ValueError, AttributeError):
                raise SourceValidationError('Invalid canonical artifact retrieval receipt') from None
            execution_id = event.get('_execution_id')
            if not isinstance(execution_id, str) or not execution_id:
                raise SourceValidationError('Invalid canonical artifact retrieval execution')
            retrievals.append(RetrievalDependency(event['_event_id'], execution_id, receipt))
        return SummaryEvidence(start, boundary, tuple(nodes), tuple(retrievals), tuple(projections))

    def _require_boundary(self, boundary: int) -> None:
        if type(boundary) is not int or boundary not in self.boundaries:
            raise SourceValidationError('Compaction requires an advancing eligible closed boundary')

    def rebuild(self, boundary: int, checkpoint: CheckpointNode) -> SourceTranscript:
        """Replace covered evidence while retaining runtime nodes and newest user.

        Retained source nodes keep their authorized request representation. Their
        expanded content is never substituted for canonical summary evidence.
        """
        self._require_boundary(boundary)
        # ``evidence`` deliberately retains the legacy dependency-free API.
        # Compaction itself consumes the typed canonical form so a dependent
        # checkpoint can retain its exact receipt proof through rebuilding.
        if not self.summary_evidence(boundary).nodes:
            raise SourceValidationError('Compaction requires new visible evidence')
        if not checkpoint.checkpoint_id or not checkpoint.summary.strip():
            raise SourceValidationError('Replacement checkpoint must contain a summary')
        covered = {event['_event_id'] for event in self._events()[:boundary]}
        rebuilt = []
        inserted = False
        for node in self.transcript.nodes:
            replace_node = isinstance(node, CheckpointNode) or (
                isinstance(node, SourceNode) and node.event_id in covered)
            if replace_node and not inserted:
                rebuilt.append(deepcopy(checkpoint))
                inserted = True
            if isinstance(node, CheckpointNode):
                continue
            if isinstance(node, SourceNode) and node.event_id in covered and node.event_id != self.pinned_event_id:
                continue
            rebuilt.append(deepcopy(node))
        if not inserted:
            raise SourceValidationError('Compaction boundary covers no represented evidence')
        return SourceTranscript(tuple(rebuilt))
