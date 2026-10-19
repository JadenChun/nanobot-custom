"""Consumed canonical summary evidence is the only dependency proof input."""

import hashlib
import json
from dataclasses import replace

import pytest

from nanobot.agent.tools.artifact import ReadArtifactTool
from nanobot.agent.tools.execution import KnownToolResult
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.session.artifact_projection import ArtifactProjector
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.artifact_store import PrivateArtifactStore
from nanobot.session.checkpoint_dependencies import (
    ConsumedSummaryChunk,
    RetrievalDependency,
    SummaryDependencyProof,
    derive_summary_references,
    validate_summary_dependencies,
)
from nanobot.session.source import SourceValidationError, new_source_event
from nanobot.session.source_compaction import SourceCompactionCandidate
from nanobot.session.source_projection import CheckpointNode, SourceNode, source_nodes
from nanobot.session.source_request import SourceTranscript
from nanobot.session.source_store import SourceSnapshot, SourceState
from tests.session.test_artifact_reader import setup_read


async def retrieval_candidate(tmp_path):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        artifact_tool = ReadArtifactTool(reader)
        outcome = await ToolRegistry.execute_prepared(
            artifact_tool, {'reference': descriptor.reference}, object()
        )
        writer.begin_tool_batch({'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'artifact', 'type': 'function',
            'function': {'name': 'read_artifact', 'arguments': '{}'},
        }]})
        await writer.record_tool_result(KnownToolResult(
            0, 'artifact', 'read_artifact', outcome, outcome.content, {'status': 'ok'}, True,
            retrieval_receipt=outcome.retrieval_receipt,
        ))
        snapshot = writer.owner.read_snapshot()
        transcript = SourceTranscript(source_nodes(snapshot.events))
        candidate = SourceCompactionCandidate.capture(
            snapshot, transcript, eligible_ids=frozenset(node.event_id for node in transcript.nodes),
        )
        yield candidate, descriptor


async def test_derives_only_consumed_validated_retrieval_receipts(tmp_path):
    async for candidate, descriptor in retrieval_candidate(tmp_path):
        evidence = candidate.summary_evidence(candidate.boundaries[-1])
        assert len(evidence.retrievals) == 1
        proof = SummaryDependencyProof(candidate, (
            ConsumedSummaryChunk(evidence, None, 'Summary of the canonical read.'),
        ))
        assert derive_summary_references(proof, candidate.snapshot) == (descriptor.reference,)
        # The legacy evidence accessor remains blocked while B3 is disabled.
        with pytest.raises(SourceValidationError, match='dependencies'):
            candidate.evidence(candidate.boundaries[-1])


async def test_derives_only_consumed_verified_projected_bindings(tmp_path):
    async with setup_read(tmp_path, 'evidence' * 6000) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        projected = ArtifactProjector(reader).prepare(
            SourceTranscript(source_nodes(writer.owner.read_snapshot().events)), enabled=True,
        )
        snapshot = writer.owner.read_snapshot()
        candidate = SourceCompactionCandidate.capture(
            snapshot, projected, eligible_ids=frozenset(node.event_id for node in projected.nodes),
        )
        evidence = candidate.summary_evidence(candidate.boundaries[-1])
        assert len(evidence.projections) == 1
        proof = SummaryDependencyProof(candidate, (
            ConsumedSummaryChunk(evidence, None, 'Summary of the projected canonical read.'),
        ))
        assert derive_summary_references(proof, snapshot) == (descriptor.reference,)


async def test_lowered_blob_cap_keeps_published_evidence_eligible_for_checkpoint_admission(tmp_path):
    async with setup_read(tmp_path, 'evidence' * 6000) as (writer, tool, store, descriptor, _):
        state = writer.owner.read_snapshot().state
        lowered_store = PrivateArtifactStore(
            store.root,
            workspace_id=store.workspace_id,
            session_key=state.session_key,
            incarnation=state.incarnation,
            generation=state.generation,
            max_blob_bytes=4,
            max_namespace_bytes=4,
        )
        reader = ArtifactReader(writer, 'root_history', lowered_store, tool)
        projected = ArtifactProjector(reader).prepare(
            SourceTranscript(source_nodes(writer.owner.read_snapshot().events)), enabled=True,
        )
        snapshot = writer.owner.read_snapshot()
        candidate = SourceCompactionCandidate.capture(
            snapshot, projected, eligible_ids=frozenset(node.event_id for node in projected.nodes),
        )
        evidence = candidate.summary_evidence(candidate.boundaries[-1])
        proof = SummaryDependencyProof(candidate, (
            ConsumedSummaryChunk(evidence, None, 'Summary of the retained evidence.'),
        ))

        with lowered_store.transaction() as blobs:
            assert validate_summary_dependencies(proof, snapshot, reader, blobs) == (descriptor.reference,)


async def test_unconsumed_suffix_cannot_affect_proof_reconstruction(tmp_path):
    async with setup_read(tmp_path, 'evidence' * 6000) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        projected = ArtifactProjector(reader).prepare(
            SourceTranscript(source_nodes(writer.owner.read_snapshot().events)), enabled=True,
        )
        suffix = new_source_event({'role': 'user', 'content': 'unconsumed suffix'}, run_id='next')
        suffix['_stream_id'] = 'root'
        snapshot = replace(writer.owner.read_snapshot(), events=[
            *writer.owner.read_snapshot().events, suffix,
        ])
        suffix_node = source_nodes(snapshot.events)[-1]
        transcript = SourceTranscript((*projected.nodes, SourceNode(
            suffix_node.event_id, suffix_node.message, object(),
        )))
        candidate = SourceCompactionCandidate.capture(
            snapshot, transcript, eligible_ids=frozenset(node.event_id for node in transcript.nodes),
        )
        evidence = candidate.summary_evidence(candidate.boundaries[0])
        proof = SummaryDependencyProof(candidate, (
            ConsumedSummaryChunk(evidence, None, 'Summary before the unconsumed suffix.'),
        ))
        assert derive_summary_references(proof, snapshot) == (descriptor.reference,)


async def test_forged_or_unconsumed_receipt_cannot_establish_a_dependency(tmp_path):
    async for candidate, _ in retrieval_candidate(tmp_path):
        evidence = candidate.summary_evidence(candidate.boundaries[-1])
        retrieval = evidence.retrievals[0]
        forged = RetrievalDependency(retrieval.event_id, retrieval.execution_id,
            replace(retrieval.receipt, sha256='0' * 64))
        proof = SummaryDependencyProof(candidate, (
            ConsumedSummaryChunk(replace(evidence, retrievals=(forged,)), None, 'Summary.'),
        ))
        with pytest.raises(SourceValidationError):
            derive_summary_references(proof, candidate.snapshot)


async def test_retrieval_payload_must_be_canonical_and_within_descriptor_bounds(tmp_path):
    async for candidate, _ in retrieval_candidate(tmp_path):
        event = next(row for row in candidate.snapshot.events if '_artifact_read' in row)
        payload = json.loads(event['content'])
        payload['eof'] = False
        event['content'] = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        event['_artifact_read']['sha256'] = hashlib.sha256(event['content'].encode()).hexdigest()
        evidence = candidate.summary_evidence(candidate.boundaries[-1])
        proof = SummaryDependencyProof(candidate, (
            ConsumedSummaryChunk(evidence, None, 'Summary of the canonical read.'),
        ))
        with pytest.raises(SourceValidationError, match='descriptor bounds'):
            derive_summary_references(proof, candidate.snapshot)


@pytest.mark.parametrize('field,value', [
    ('reference', None), ('reference', 7), ('text', None), ('text', []),
    ('start_offset', True), ('start_offset', '0'), ('next_offset', False),
    ('next_offset', None), ('eof', 1), ('eof', 'true'),
])
async def test_retrieval_proof_retains_canonical_field_refusals(tmp_path, field, value):
    async for candidate, _ in retrieval_candidate(tmp_path):
        event = next(row for row in candidate.snapshot.events if '_artifact_read' in row)
        payload = json.loads(event['content'])
        payload[field] = value
        event['content'] = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        event['_artifact_read']['sha256'] = hashlib.sha256(event['content'].encode()).hexdigest()
        evidence = candidate.summary_evidence(candidate.boundaries[-1])
        proof = SummaryDependencyProof(candidate, (
            ConsumedSummaryChunk(evidence, None, 'Summary of a malformed canonical read.'),
        ))

        with pytest.raises(SourceValidationError, match='canonical JSON payload'):
            derive_summary_references(proof, candidate.snapshot)


@pytest.mark.parametrize('summary', [None, False, '', '   '])
def test_summary_fold_retains_nonstring_and_empty_output_refusals(summary):
    event = new_source_event({'role': 'user', 'content': 'visible'}, run_id='run')
    event['_stream_id'] = 'root'
    snapshot = SourceSnapshot(
        SourceState('store', 'key', 'incarnation', 1, 0, None, 'fingerprint'), {}, [event], None,
    )
    transcript = SourceTranscript(source_nodes(snapshot.events))
    candidate = SourceCompactionCandidate.capture(
        snapshot, transcript, eligible_ids=frozenset(node.event_id for node in transcript.nodes),
    )
    evidence = candidate.summary_evidence(candidate.boundaries[-1])
    proof = SummaryDependencyProof(candidate, (ConsumedSummaryChunk(evidence, None, summary),))

    with pytest.raises(SourceValidationError, match='Invalid consumed summary output'):
        derive_summary_references(proof, snapshot)


async def test_prior_references_require_the_exact_accepted_summary_fold(tmp_path):
    async for original, descriptor in retrieval_candidate(tmp_path):
        prior = CheckpointNode('prior-checkpoint', 'Previously accepted summary.', (descriptor.reference,))
        followup = new_source_event({'role': 'user', 'content': 'New visible input.'}, run_id='next')
        followup['_stream_id'] = 'root'
        events = [*original.snapshot.events, followup]
        stored = {
            'checkpoint_id': prior.checkpoint_id, 'summary': prior.summary,
            'artifact_references': list(prior.artifact_references), 'boundary': len(original.snapshot.events),
        }
        snapshot = replace(
            original.snapshot,
            state=replace(original.snapshot.state, checkpoint_id=prior.checkpoint_id),
            events=events,
            checkpoint=stored,
        )
        transcript = SourceTranscript((prior, *source_nodes(events[len(original.snapshot.events):])))
        candidate = SourceCompactionCandidate.capture(
            snapshot, transcript, eligible_ids=frozenset({followup['_event_id']}),
            accepted_boundary=stored['boundary'], accepted_checkpoint=prior,
        )
        evidence = candidate.summary_evidence(candidate.boundaries[-1])
        proof = SummaryDependencyProof(candidate, (
            ConsumedSummaryChunk(evidence, prior.summary, 'Folded follow-up summary.'),
        ))
        assert derive_summary_references(proof, snapshot) == (descriptor.reference,)
        with pytest.raises(SourceValidationError, match='fold chain'):
            derive_summary_references(SummaryDependencyProof(candidate, (
                ConsumedSummaryChunk(evidence, None, 'Folded follow-up summary.'),
            )), snapshot)


def test_hidden_child_marker_is_filtered_before_receipt_parsing():
    root = new_source_event({'role': 'user', 'content': 'visible'}, run_id='run')
    root['_stream_id'] = 'root'
    call = {'id': 'artifact', 'type': 'function',
            'function': {'name': 'read_artifact', 'arguments': '{}'}}
    intent = new_source_event({'role': 'assistant', 'content': '', 'tool_calls': [call]}, run_id='run')
    intent.update(_stream_id='child', _parent_stream_id='root', _execution_ids=['execution'])
    child = new_source_event({'role': 'tool', 'tool_call_id': 'artifact', 'name': 'read_artifact',
                              'content': 'hidden', '_dispatched': True, '_outcome_status': 'ok'}, run_id='run')
    child.update(_stream_id='child', _parent_stream_id='root', _execution_id='execution',
                 _artifact_read={'forged': True})
    snapshot = SourceSnapshot(
        SourceState('store', 'key', 'incarnation', 1, 0, None, 'fingerprint'), {}, [root, intent, child], None,
    )
    transcript = SourceTranscript(source_nodes(snapshot.events))
    candidate = SourceCompactionCandidate.capture(
        snapshot, transcript, eligible_ids=frozenset(node.event_id for node in transcript.nodes),
    )
    evidence = candidate.summary_evidence(candidate.boundaries[-1])
    assert evidence.retrievals == ()
    assert tuple(node.event_id for node in evidence.nodes) == (root['_event_id'],)


def test_untyped_projected_nodes_are_not_summary_evidence():
    event = new_source_event({'role': 'user', 'content': 'visible'}, run_id='run')
    event['_stream_id'] = 'root'
    snapshot = SourceSnapshot(
        SourceState('store', 'key', 'incarnation', 1, 0, None, 'fingerprint'), {}, [event], None,
    )
    original = source_nodes(snapshot.events)[0]
    projected = SourceNode(original.event_id, original.message, object())
    candidate = SourceCompactionCandidate.capture(
        snapshot, SourceTranscript((projected,)), eligible_ids=frozenset({original.event_id}),
    )
    with pytest.raises(SourceValidationError, match='projection binding'):
        candidate.summary_evidence(candidate.boundaries[-1])


def _stored_checkpoint_fields():
    return {
        'checkpoint_id': 'prior-checkpoint', 'summary': 'Previously accepted summary.',
        'artifact_references': [], 'boundary': 2,
    }


def _checkpoint_comparison_proof(checkpoint):
    prior = CheckpointNode('prior-checkpoint', 'Previously accepted summary.')
    events = [new_source_event({'role': role, 'content': text}, run_id=run) for role, text, run in (
        ('user', 'Previous input.', 'previous'), ('assistant', 'Previous answer.', 'previous'),
        ('user', 'New input.', 'next'), ('assistant', 'New answer.', 'next'),
    )]
    for event in events:
        event['_stream_id'] = 'root'
    snapshot = SourceSnapshot(
        SourceState('store', 'key', 'incarnation', 1, 1, prior.checkpoint_id, 'fingerprint'),
        {}, events, checkpoint,
    )
    candidate = SourceCompactionCandidate.capture(
        snapshot, SourceTranscript((prior, *source_nodes(events[2:]))),
        eligible_ids=frozenset(event['_event_id'] for event in events[2:]),
        accepted_boundary=2, accepted_checkpoint=prior,
    )
    evidence = candidate.summary_evidence(candidate.boundaries[-1])
    return SummaryDependencyProof(candidate, (
        ConsumedSummaryChunk(evidence, prior.summary, 'Folded follow-up summary.'),
    ))


@pytest.mark.parametrize('overrides', [
    {}, {'artifact_references': ()}, {'artifact_references': ''}, {'boundary': 2.0},
    {'extra': 'ignored stored metadata'},
])
def test_checkpoint_field_comparison_preserves_accepted_values(overrides):
    proof = _checkpoint_comparison_proof({**_stored_checkpoint_fields(), **overrides})

    assert derive_summary_references(proof, proof.candidate.snapshot) == ()


def test_checkpoint_field_comparison_preserves_missing_reference_default():
    checkpoint = _stored_checkpoint_fields()
    del checkpoint['artifact_references']
    proof = _checkpoint_comparison_proof(checkpoint)

    assert derive_summary_references(proof, proof.candidate.snapshot) == ()


@pytest.mark.parametrize('field,value', [
    ('checkpoint_id', None), ('checkpoint_id', 7), ('checkpoint_id', 'different'),
    ('summary', None), ('summary', []), ('summary', ''), ('summary', 'different'),
    ('artifact_references', ['artifact:unaccepted']), ('artifact_references', 'unexpected'),
    ('boundary', None), ('boundary', 0), ('boundary', '2'),
])
def test_checkpoint_field_comparison_preserves_changed_field_refusals(field, value):
    proof = _checkpoint_comparison_proof({**_stored_checkpoint_fields(), field: value})

    with pytest.raises(SourceValidationError, match='^Accepted summary checkpoint changed$'):
        derive_summary_references(proof, proof.candidate.snapshot)


@pytest.mark.parametrize('field', ['checkpoint_id', 'summary', 'boundary'])
def test_checkpoint_field_comparison_preserves_missing_field_refusals(field):
    checkpoint = _stored_checkpoint_fields()
    del checkpoint[field]
    proof = _checkpoint_comparison_proof(checkpoint)

    with pytest.raises(SourceValidationError, match='^Accepted summary checkpoint changed$'):
        derive_summary_references(proof, proof.candidate.snapshot)


@pytest.mark.parametrize('checkpoint', [None, [], (), 'not a record'])
def test_checkpoint_field_comparison_preserves_record_type_refusals(checkpoint):
    proof = _checkpoint_comparison_proof(checkpoint)

    with pytest.raises(SourceValidationError, match='^Accepted summary checkpoint is unavailable$'):
        derive_summary_references(proof, proof.candidate.snapshot)


@pytest.mark.parametrize('references', [None, 7, False])
def test_checkpoint_field_comparison_preserves_noniterable_reference_faults(references):
    proof = _checkpoint_comparison_proof({**_stored_checkpoint_fields(), 'artifact_references': references})

    with pytest.raises(TypeError):
        derive_summary_references(proof, proof.candidate.snapshot)


@pytest.mark.parametrize('failing_field', [None, 'checkpoint_id', 'summary', 'artifact_references', 'boundary'])
def test_checkpoint_field_comparison_preserves_lookup_order_and_fault_identity(failing_field):
    lookups = []
    failure = LookupError('stored checkpoint lookup failed')

    class TracedCheckpoint(dict):
        def get(self, key, default=None):
            lookups.append(key)
            if key == failing_field:
                raise failure
            return super().get(key, default)

    proof = _checkpoint_comparison_proof(TracedCheckpoint(_stored_checkpoint_fields()))
    lookups.clear()
    expected = ['checkpoint_id', 'summary', 'artifact_references', 'boundary']

    if failing_field is None:
        assert derive_summary_references(proof, proof.candidate.snapshot) == ()
        assert lookups == expected
    else:
        with pytest.raises(LookupError) as exc:
            derive_summary_references(proof, proof.candidate.snapshot)
        assert exc.value is failure
        assert lookups == expected[:expected.index(failing_field) + 1]
