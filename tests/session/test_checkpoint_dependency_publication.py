"""Trusted schema-v2 checkpoint publication is a separate, locked store path."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from nanobot.agent.tools.artifact import ReadArtifactTool
from nanobot.agent.tools.execution import KnownToolResult
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.session.artifact import ArtifactValidationError
from nanobot.session.artifact_projection import ArtifactProjector
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.artifact_store import ArtifactStorageError, PrivateArtifactStore
from nanobot.session.checkpoint_dependencies import ConsumedSummaryChunk, SummaryDependencyProof
from nanobot.session.execution_owner import ExecutionOwnershipError
from nanobot.session.manager import SessionWriteConflict
from nanobot.session.source import SourceValidationError, prefix_digest
from nanobot.session.source_compaction import SourceCompactionCandidate
from nanobot.session.source_projection import source_nodes
from nanobot.session.source_request import SourceTranscript
from nanobot.session.source_store import SourceSessionStore
from tests.session.test_artifact_reader import setup_read


async def publication_inputs(tmp_path):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        # The dependent publisher uses one source store explicitly configured
        # with the same private artifact namespace as its reader.
        writer.store = SourceSessionStore(writer.store._backend, artifact_store=store)
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
        evidence = candidate.summary_evidence(candidate.boundaries[-1])
        summary = 'Summary which consumed the trusted artifact retrieval.'
        proof = SummaryDependencyProof(candidate, (ConsumedSummaryChunk(evidence, None, summary),))
        proposal = {
            'schema_version': 2, 'digest_version': 1, 'summary_format_version': 1,
            'session_generation': snapshot.state.generation, 'source_revision': snapshot.state.revision,
            'checkpoint_id': 'dependent-checkpoint', 'parent_checkpoint_id': snapshot.state.checkpoint_id,
            'producing_model': 'model', 'producing_provider': 'provider', 'digest_algorithm': 'sha256',
            'created_at': datetime.now(timezone.utc).isoformat(), 'summary': summary,
            'artifact_references': [descriptor.reference], 'boundary': evidence.boundary,
            'boundary_event_id': snapshot.events[evidence.boundary - 1]['_event_id'],
            'prefix_digest': prefix_digest(snapshot.events, evidence.boundary), 'auxiliary_usage': None,
            'projection_scope': 'root', 'projection_version': 1,
        }
        yield writer, reader, store, candidate, proof, proposal


async def projected_publication_inputs(tmp_path):
    async with setup_read(tmp_path, 'projected evidence ' * 2000) as (writer, tool, store, descriptor, _):
        writer.store = SourceSessionStore(writer.store._backend, artifact_store=store)
        reader = ArtifactReader(writer, 'root_history', store, tool)
        projected = ArtifactProjector(reader).prepare(
            SourceTranscript(source_nodes(writer.owner.read_snapshot().events)), enabled=True,
        )
        snapshot = writer.owner.read_snapshot()
        candidate = SourceCompactionCandidate.capture(
            snapshot, projected, eligible_ids=frozenset(node.event_id for node in projected.nodes),
        )
        evidence = candidate.summary_evidence(candidate.boundaries[-1])
        summary = 'Summary which consumed the trusted projected source output.'
        proof = SummaryDependencyProof(candidate, (ConsumedSummaryChunk(evidence, None, summary),))
        proposal = {
            'schema_version': 2, 'digest_version': 1, 'summary_format_version': 1,
            'session_generation': snapshot.state.generation, 'source_revision': snapshot.state.revision,
            'checkpoint_id': 'projected-dependent-checkpoint',
            'parent_checkpoint_id': snapshot.state.checkpoint_id,
            'producing_model': 'model', 'producing_provider': 'provider', 'digest_algorithm': 'sha256',
            'created_at': datetime.now(timezone.utc).isoformat(), 'summary': summary,
            'artifact_references': [descriptor.reference], 'boundary': evidence.boundary,
            'boundary_event_id': snapshot.events[evidence.boundary - 1]['_event_id'],
            'prefix_digest': prefix_digest(snapshot.events, evidence.boundary), 'auxiliary_usage': None,
            'projection_scope': 'root', 'projection_version': 1,
        }
        yield writer, reader, store, candidate, proof, proposal


async def test_commits_exact_schema_v2_dependency_proof_once(tmp_path):
    async for writer, reader, store, candidate, proof, proposal in publication_inputs(tmp_path):
        with store.transaction() as blobs:
            committed = writer.store.commit_dependent_context_checkpoint(
                writer.owner.key, candidate.snapshot.state, proposal, proof,
                owner=writer.owner, reader=reader, blobs=blobs,
            )
        assert committed.state.revision == candidate.snapshot.state.revision + 1
        assert committed.events == candidate.snapshot.events
        assert committed.checkpoint == proposal


async def test_commits_exact_schema_v2_projected_dependency_proof_once(tmp_path):
    async for writer, reader, store, candidate, proof, proposal in projected_publication_inputs(tmp_path):
        with store.transaction() as blobs:
            committed = writer.store.commit_dependent_context_checkpoint(
                writer.owner.key, candidate.snapshot.state, proposal, proof,
                owner=writer.owner, reader=reader, blobs=blobs,
            )
        assert committed.state.revision == candidate.snapshot.state.revision + 1
        assert committed.events == candidate.snapshot.events
        assert committed.checkpoint == proposal


@pytest.mark.parametrize('failure', ['forged', 'revoked', 'corrupt'])
async def test_projected_dependency_failure_never_writes(tmp_path, failure):
    async for writer, reader, store, candidate, proof, proposal in projected_publication_inputs(tmp_path):
        before = writer.store._backend.get_session_path(writer.owner.key).read_bytes()
        if failure == 'forged':
            chunk = proof.chunks[0]
            proof = SummaryDependencyProof(candidate, (
                replace(chunk, evidence=replace(chunk.evidence, projections=())),
            ))
        elif failure == 'revoked':
            reader.read_file._workspace = tmp_path / 'revoked'
        else:
            next(store.root.rglob('*.blob')).write_bytes(b'corrupt')
        with store.transaction() as blobs:
            with pytest.raises((ArtifactValidationError, ArtifactStorageError, SourceValidationError)):
                writer.store.commit_dependent_context_checkpoint(
                    writer.owner.key, candidate.snapshot.state, proposal, proof,
                    owner=writer.owner, reader=reader, blobs=blobs,
                )
        assert writer.store._backend.get_session_path(writer.owner.key).read_bytes() == before


async def test_rejects_nonexact_references_without_writing(tmp_path):
    async for writer, reader, store, candidate, proof, proposal in publication_inputs(tmp_path):
        before = writer.store._backend.get_session_path(writer.owner.key).read_bytes()
        proposal['artifact_references'] = []
        with store.transaction() as blobs:
            with pytest.raises(SourceValidationError):
                writer.store.commit_dependent_context_checkpoint(
                    writer.owner.key, candidate.snapshot.state, proposal, proof,
                    owner=writer.owner, reader=reader, blobs=blobs,
                )
        assert writer.store._backend.get_session_path(writer.owner.key).read_bytes() == before


async def test_stale_source_state_rejects_without_rebasing_or_writing(tmp_path):
    async for writer, reader, store, candidate, proof, proposal in publication_inputs(tmp_path):
        writer.store.update_metadata(writer.owner.key, {'title': 'concurrent metadata update'})
        before = writer.store._backend.get_session_path(writer.owner.key).read_bytes()
        with store.transaction() as blobs:
            with pytest.raises(SessionWriteConflict):
                writer.store.commit_dependent_context_checkpoint(
                    writer.owner.key, candidate.snapshot.state, proposal, proof,
                    owner=writer.owner, reader=reader, blobs=blobs,
                )
        assert writer.store._backend.get_session_path(writer.owner.key).read_bytes() == before


@pytest.mark.parametrize('failure', ['foreign', 'closed', 'revoked', 'corrupt'])
async def test_untrusted_transaction_or_current_dependency_failure_never_writes(tmp_path, failure):
    async for writer, reader, store, candidate, proof, proposal in publication_inputs(tmp_path):
        before = writer.store._backend.get_session_path(writer.owner.key).read_bytes()
        if failure == 'foreign':
            foreign = PrivateArtifactStore(
                (tmp_path / 'foreign-artifacts').resolve(), workspace_id=store.workspace_id,
                session_key=store.session_key, incarnation=store.incarnation, generation=store.generation,
            )
            with foreign.transaction() as blobs:
                with pytest.raises(ExecutionOwnershipError):
                    writer.store.commit_dependent_context_checkpoint(
                        writer.owner.key, candidate.snapshot.state, proposal, proof,
                        owner=writer.owner, reader=reader, blobs=blobs,
                    )
        elif failure == 'closed':
            with store.transaction() as blobs:
                pass
            with pytest.raises(ArtifactStorageError):
                writer.store.commit_dependent_context_checkpoint(
                    writer.owner.key, candidate.snapshot.state, proposal, proof,
                    owner=writer.owner, reader=reader, blobs=blobs,
                )
        else:
            if failure == 'revoked':
                reader.read_file._workspace = tmp_path / 'revoked'
            else:
                next(store.root.rglob('*.blob')).write_bytes(b'corrupt')
            with store.transaction() as blobs:
                with pytest.raises((ArtifactValidationError, ArtifactStorageError)):
                    writer.store.commit_dependent_context_checkpoint(
                        writer.owner.key, candidate.snapshot.state, proposal, proof,
                        owner=writer.owner, reader=reader, blobs=blobs,
                    )
        assert writer.store._backend.get_session_path(writer.owner.key).read_bytes() == before
