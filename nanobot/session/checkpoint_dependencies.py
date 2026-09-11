"""Trusted, runtime-only dependency proofs for schema-v2 summaries.

The records in this module are claims made by the summary caller.  They never
authorize an artifact by themselves: every consumer reconstructs the canonical
source evidence and then the store rechecks current read authorization and blob
durability while holding its normal transactions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nanobot.agent.tools.artifact import ArtifactRetrievalReceipt, parse_retrieval_receipt
from nanobot.session.artifact import (
    ArtifactDescriptor,
    ArtifactValidationError,
    parse_artifact_descriptor,
    published_artifacts,
)
from nanobot.session.source import (
    SourceValidationError,
    canonical_events,
    validate_event_ids,
    validate_execution_links,
)
from nanobot.session.source_projection import CheckpointNode, SourceNode
from nanobot.session.source_store import SourceSnapshot

if TYPE_CHECKING:
    from nanobot.session.artifact_reader import ArtifactReader
    from nanobot.session.artifact_store import ArtifactBlobTransaction
    from nanobot.session.source_compaction import SourceCompactionCandidate


@dataclass(frozen=True)
class RetrievalDependency:
    """One consumed, canonical `read_artifact` result and its trusted receipt."""

    event_id: str
    execution_id: str
    receipt: ArtifactRetrievalReceipt


@dataclass(frozen=True)
class SummaryEvidence:
    """One closed source interval supplied to an auxiliary summary call."""

    after: int
    boundary: int
    nodes: tuple[SourceNode, ...]
    retrievals: tuple[RetrievalDependency, ...]


@dataclass(frozen=True)
class ConsumedSummaryChunk:
    """A summary call's exact input linkage and returned summary text."""

    evidence: SummaryEvidence
    previous_summary: str | None
    summary: str


@dataclass(frozen=True)
class SummaryDependencyProof:
    """The complete contiguous fold used to create one checkpoint proposal."""

    candidate: SourceCompactionCandidate
    chunks: tuple[ConsumedSummaryChunk, ...]


@dataclass(frozen=True)
class _ValidatedProof:
    references: tuple[str, ...]
    boundary: int
    summary: str


@dataclass(frozen=True)
class _CanonicalRetrievalPayload:
    reference: str
    start_offset: int
    next_offset: int
    eof: bool
    text_bytes: int


def _require_candidate(proof: SummaryDependencyProof, snapshot: SourceSnapshot) -> SourceCompactionCandidate:
    # Import lazily: SourceCompactionCandidate constructs SummaryEvidence.
    from nanobot.session.source_compaction import SourceCompactionCandidate

    if type(proof) is not SummaryDependencyProof or type(snapshot) is not SourceSnapshot:
        raise SourceValidationError('Invalid summary dependency proof')
    candidate = proof.candidate
    if type(candidate) is not SourceCompactionCandidate or candidate.snapshot != snapshot:
        raise SourceValidationError('Summary dependency proof source changed')
    if type(candidate.previous_boundary) is not int or candidate.previous_boundary < 0:
        raise SourceValidationError('Invalid summary dependency proof boundary')
    previous = candidate.previous_checkpoint
    if (candidate.previous_boundary == 0) != (previous is None):
        raise SourceValidationError('Invalid accepted summary checkpoint state')
    if previous is not None and candidate.stream_id is None:
        if type(previous) is not CheckpointNode or not isinstance(snapshot.checkpoint, dict):
            raise SourceValidationError('Accepted summary checkpoint is unavailable')
        stored = snapshot.checkpoint
        expected = CheckpointNode(
            stored.get('checkpoint_id'), stored.get('summary'),
            tuple(stored.get('artifact_references', ())),
        )
        if stored.get('boundary') != candidate.previous_boundary or previous != expected:
            raise SourceValidationError('Accepted summary checkpoint changed')
    validate_event_ids(snapshot.events)
    validate_execution_links(snapshot.events)
    canonical_events(snapshot.events)
    return candidate


def _canonical_retrieval_payload(content: object,
                                 receipt: ArtifactRetrievalReceipt) -> _CanonicalRetrievalPayload:
    if not isinstance(content, str):
        raise SourceValidationError('Artifact retrieval result is not a canonical JSON payload')

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        names = [name for name, _ in pairs]
        if names != ['reference', 'text', 'start_offset', 'next_offset', 'eof']:
            raise ValueError
        return dict(pairs)

    try:
        payload = json.loads(content, object_pairs_hook=object_pairs)
        if (not isinstance(payload, dict) or payload.get('reference') != receipt.reference
                or not isinstance(payload.get('text'), str)
                or type(payload.get('start_offset')) is not int
                or type(payload.get('next_offset')) is not int
                or type(payload.get('eof')) is not bool
                or payload['start_offset'] != receipt.start_offset
                or payload['next_offset'] != receipt.next_offset
                or json.dumps(payload, ensure_ascii=False, separators=(',', ':')) != content):
            raise ValueError
        text_bytes = len(payload['text'].encode('utf-8'))
    except (TypeError, ValueError, UnicodeError):
        raise SourceValidationError('Artifact retrieval result is not a canonical JSON payload') from None
    return _CanonicalRetrievalPayload(receipt.reference, receipt.start_offset,
                                      receipt.next_offset, payload['eof'], text_bytes)


def _validate_retrieval(snapshot: SourceSnapshot,
                        dependency: RetrievalDependency) -> _CanonicalRetrievalPayload:
    if (type(dependency) is not RetrievalDependency or not isinstance(dependency.event_id, str)
            or not dependency.event_id or not isinstance(dependency.execution_id, str)
            or not dependency.execution_id or type(dependency.receipt) is not ArtifactRetrievalReceipt):
        raise SourceValidationError('Invalid consumed artifact retrieval proof')
    event = next((row for row in snapshot.events if row['_event_id'] == dependency.event_id), None)
    if (event is None or event.get('role') != 'tool' or event.get('_execution_id') != dependency.execution_id
            or event.get('_dispatched') is not True or event.get('_outcome_status') != 'ok'
            or event.get('name') != 'read_artifact'):
        raise SourceValidationError('Artifact retrieval execution was not successfully dispatched')
    raw_receipt = event.get('_artifact_read')
    try:
        receipt = parse_retrieval_receipt(raw_receipt)
    except (TypeError, ValueError, AttributeError):
        raise SourceValidationError('Invalid canonical artifact retrieval receipt') from None
    if receipt != dependency.receipt or not receipt.matches(event.get('content')):
        raise SourceValidationError('Artifact retrieval payload receipt does not match canonical result')
    payload = _canonical_retrieval_payload(event.get('content'), receipt)
    intent = None
    for row in snapshot.events:
        identities = row.get('_execution_ids')
        if isinstance(identities, list) and dependency.execution_id in identities:
            intent = row
            index = identities.index(dependency.execution_id)
            break
    if intent is None or intent.get('role') != 'assistant' or not isinstance(intent.get('tool_calls'), list):
        raise SourceValidationError('Artifact retrieval intent is absent')
    call = intent['tool_calls'][index]
    function = call.get('function') if isinstance(call, dict) else None
    if (not isinstance(function, dict) or function.get('name') != 'read_artifact'
            or event.get('tool_call_id') != call.get('id')
            or intent.get('_stream_id') != event.get('_stream_id')
            or intent.get('_parent_stream_id') != event.get('_parent_stream_id')):
        raise SourceValidationError('Artifact retrieval intent does not match canonical result')
    return payload


def _manifest_descriptors(snapshot: SourceSnapshot) -> dict[str, ArtifactDescriptor]:
    metadata = snapshot.header.get('metadata')
    manifest = metadata.get('artifact_manifest') if isinstance(metadata, dict) else None
    if (not isinstance(manifest, dict) or set(manifest) != {'schema_version', 'descriptors'}
            or not isinstance(manifest.get('descriptors'), dict) or not manifest['descriptors']):
        raise ArtifactValidationError('Summary dependency manifest is unavailable')
    first = next(iter(manifest['descriptors'].values()))
    workspace_id = parse_artifact_descriptor(first).workspace_id
    return {item.reference: item for item in published_artifacts(snapshot, workspace_id=workspace_id)}


def _validated_proof(proof: SummaryDependencyProof, snapshot: SourceSnapshot) -> _ValidatedProof:
    candidate = _require_candidate(proof, snapshot)
    if not isinstance(proof.chunks, tuple) or not proof.chunks:
        raise SourceValidationError('Summary dependency proof has no consumed chunks')
    cursor = candidate.previous_boundary
    previous_summary = candidate.previous_checkpoint.summary if candidate.previous_checkpoint else None
    references = set(candidate.previous_checkpoint.artifact_references if candidate.previous_checkpoint else ())
    retrievals: list[_CanonicalRetrievalPayload] = []
    for chunk in proof.chunks:
        if type(chunk) is not ConsumedSummaryChunk or type(chunk.evidence) is not SummaryEvidence:
            raise SourceValidationError('Invalid consumed summary proof record')
        evidence = chunk.evidence
        if (type(evidence.after) is not int or type(evidence.boundary) is not int
                or evidence.after != cursor or evidence.boundary <= cursor):
            raise SourceValidationError('Summary dependency evidence is not contiguous')
        if chunk.previous_summary != previous_summary:
            raise SourceValidationError('Summary dependency fold chain changed')
        if not isinstance(chunk.summary, str) or not chunk.summary.strip():
            raise SourceValidationError('Invalid consumed summary output')
        expected = candidate.summary_evidence(evidence.boundary, after=cursor)
        if evidence != expected:
            raise SourceValidationError('Summary dependency evidence differs from canonical source')
        for retrieval in evidence.retrievals:
            payload = _validate_retrieval(snapshot, retrieval)
            references.add(payload.reference)
            retrievals.append(payload)
        cursor = evidence.boundary
        previous_summary = chunk.summary
    if references:
        descriptors = _manifest_descriptors(snapshot)
        for reference in references:
            if reference not in descriptors:
                raise ArtifactValidationError('Summary dependency is not published')
        for payload in retrievals:
            descriptor = descriptors[payload.reference]
            if (not 0 <= payload.start_offset <= payload.next_offset <= descriptor.byte_length
                    or payload.next_offset - payload.start_offset != payload.text_bytes
                    or payload.eof != (payload.next_offset == descriptor.byte_length)):
                raise SourceValidationError('Artifact retrieval payload is outside descriptor bounds')
    return _ValidatedProof(tuple(sorted(references)), cursor, previous_summary)


def derive_summary_references(proof: SummaryDependencyProof, snapshot: SourceSnapshot) -> tuple[str, ...]:
    """Reconstruct sorted dependency references from consumed canonical evidence."""

    return _validated_proof(proof, snapshot).references


def _validate_current_scope(candidate: SourceCompactionCandidate,
                            snapshot: SourceSnapshot) -> None:
    """Require exact root state or exact admitted-stream history before use."""
    captured = candidate.snapshot
    if candidate.stream_id is None:
        if snapshot != captured:
            raise ArtifactValidationError('Dependent checkpoint source changed during validation')
        return
    before, after = captured.state, snapshot.state
    if ((before.store_identity, before.session_key, before.incarnation, before.generation)
            != (after.store_identity, after.session_key, after.incarnation, after.generation)
            or [event for event in captured.events if event.get('_stream_id') == candidate.stream_id]
            != [event for event in snapshot.events if event.get('_stream_id') == candidate.stream_id]):
        raise ArtifactValidationError('Dependent local source changed during validation')


def _validate_reader_binding(reader: ArtifactReader, snapshot: SourceSnapshot,
                             blobs: ArtifactBlobTransaction,
                             candidate: SourceCompactionCandidate) -> None:
    from nanobot.session.artifact_reader import ArtifactReader
    from nanobot.session.artifact_store import ArtifactBlobTransaction

    if type(reader) is not ArtifactReader or type(blobs) is not ArtifactBlobTransaction:
        raise ArtifactValidationError('Dependent checkpoint requires trusted artifact access')
    writer = reader.writer
    store = reader.store
    if (blobs._store is not store
            or store.workspace_id != writer.store._backend.sessions_dir.name
            or store.session_key != snapshot.state.session_key
            or store.incarnation != snapshot.state.incarnation
            or store.generation != snapshot.state.generation
            or writer.owner.key != snapshot.state.session_key
            or writer.store._store_identity != snapshot.state.store_identity):
        raise ArtifactValidationError('Dependent checkpoint artifact access binding changed')
    if candidate.stream_id is None:
        if reader.mode != 'root_history' or writer.parent_stream_id is not None:
            raise ArtifactValidationError('Dependent root checkpoint requires root reader admission')
    elif (reader.mode != 'isolated_stream' or writer.stream_id != candidate.stream_id):
        raise ArtifactValidationError('Dependent local checkpoint requires isolated reader admission')
    writer.owner.validate_dispatch()
    _validate_current_scope(candidate, writer.owner.read_snapshot())


def validate_summary_dependencies(proof: SummaryDependencyProof, snapshot: SourceSnapshot,
                                  reader: ArtifactReader, blobs: ArtifactBlobTransaction) -> tuple[str, ...]:
    """Recheck proof claims against current admission, manifests, and durable bytes."""

    candidate = _require_candidate(proof, proof.candidate.snapshot)
    references = derive_summary_references(proof, candidate.snapshot)
    _validate_current_scope(candidate, snapshot)
    _validate_reader_binding(reader, snapshot, blobs, candidate)
    manifests = {item.reference: item for item in
                 published_artifacts(snapshot, workspace_id=reader.store.workspace_id)}
    for reference in references:
        descriptor = manifests.get(reference)
        if descriptor is None:
            raise ArtifactValidationError('Summary dependency is not currently published')
        reader.verify_published(descriptor, snapshot, blobs)
    blobs.assert_reachable()
    reader.writer.owner.validate_dispatch()
    return references
