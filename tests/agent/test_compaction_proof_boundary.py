"""The reference-bearing branches retain their validated proof and refusal phases."""

from dataclasses import replace

import pytest

import nanobot.session.checkpoint_dependencies as dependencies
from nanobot.agent.context_governance import ContextGovernor
from nanobot.session.artifact_projection import ArtifactProjector
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.checkpoint_dependencies import ConsumedSummaryChunk, SummaryDependencyProof
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.source import SourceValidationError
from nanobot.session.source_store import SourceSessionStore
from nanobot.session.turn_writer import SourceTurnWriter
from tests.agent.test_source_compaction_publication import capture, state_for
from tests.agent.test_source_compaction_publication import store as store
from tests.agent.test_source_dependency_compaction import _prepare, _record_retrieval, _state
from tests.session.test_artifact_reader import setup_read


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("evidence", ["retrieval", "projection"])
async def test_reference_publication_dispatches_the_exact_proof(
    tmp_path, monkeypatch, local, evidence,
) -> None:
    async with setup_read(tmp_path, "bounded evidence " * 2000, child=local) as (
        writer, tool, blobs, descriptor, _,
    ):
        reader = ArtifactReader(writer, "isolated_stream" if local else "root_history", blobs, tool)
        if evidence == "retrieval":
            await _record_retrieval(writer, reader, descriptor.reference)
        state = _state(writer, reader, local=local)
        if evidence == "projection":
            projected = ArtifactProjector(reader).prepare(state.source.transcript, enabled=True)
            state.source.replace_projection(projected)
            state.source_compaction.raw_messages[:] = projected.messages()
            state.conversation.replace_transcript(state.source_compaction.raw_messages)
        before = writer.owner.read_snapshot()
        captured, dispatched = [], []
        publishing = False
        original_publish = ContextGovernor.publish_source_compaction
        original_commit = SourceSessionStore.commit_dependent_context_checkpoint
        original_validate = dependencies.validate_summary_dependencies

        def publish(self, request_state, attempt, **kwargs):
            nonlocal publishing
            proof = kwargs["proof"]
            assert type(proof) is SummaryDependencyProof
            captured.append(proof)
            publishing = True
            try:
                return original_publish(self, request_state, attempt, **kwargs)
            finally:
                publishing = False

        def commit(self, key, expected, proposal, proof, **kwargs):
            assert publishing and not local and proof is captured[0]
            assert proposal["artifact_references"] == [descriptor.reference]
            dispatched.append(proof)
            return original_commit(self, key, expected, proposal, proof, **kwargs)

        def validate(proof, snapshot, admitted_reader, transaction):
            if publishing and local:
                assert proof is captured[0]
                assert admitted_reader is reader
                dispatched.append(proof)
            return original_validate(proof, snapshot, admitted_reader, transaction)

        monkeypatch.setattr(ContextGovernor, "publish_source_compaction", publish)
        monkeypatch.setattr(SourceSessionStore, "commit_dependent_context_checkpoint", commit)
        monkeypatch.setattr(dependencies, "validate_summary_dependencies", validate)
        await _prepare(state, state.config.tools.get_definitions())
        after = writer.owner.read_snapshot()
        assert len(captured) == 1 and len(dispatched) == 1
        assert dispatched[0] is captured[0]
        assert state.source.transcript.messages() == state.source_compaction.raw_messages
        if local:
            assert after == before
            assert state.source_compaction.local_checkpoint.artifact_references == (descriptor.reference,)
        else:
            assert after.events == before.events
            assert after.state.revision == before.state.revision + 1
            assert after.checkpoint["schema_version"] == 2
            assert after.checkpoint["artifact_references"] == [descriptor.reference]


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("failure", ["missing", "wrong_type", "empty", "candidate", "summary", "boundary"])
async def test_invalid_proof_is_refused_before_dependent_publication(
    tmp_path, monkeypatch, local, failure,
) -> None:
    async with setup_read(tmp_path, child=local) as (writer, tool, blobs, descriptor, _):
        reader = ArtifactReader(writer, "isolated_stream" if local else "root_history", blobs, tool)
        await _record_retrieval(writer, reader, descriptor.reference)
        state = _state(writer, reader, local=local)
        governor = ContextGovernor()
        attempt = governor.capture_source_compaction(
            state, state.source.transcript, tool_definitions=state.config.tools.get_definitions(),
        )
        candidate = attempt.candidate
        boundary = candidate.boundaries[-1]
        summary = "retained dependency"
        chunk = ConsumedSummaryChunk(candidate.summary_evidence(boundary), None, summary)
        proof = SummaryDependencyProof(candidate, (chunk,))
        if failure == "missing":
            proof = None
        elif failure == "wrong_type":
            proof = object()
        elif failure == "empty":
            proof = replace(proof, chunks=())
        elif failure == "candidate":
            proof = replace(proof, candidate=replace(candidate, previous_boundary=1))
        elif failure == "summary":
            proof = replace(proof, chunks=(replace(chunk, summary="different fold"),))
        else:
            proof = replace(proof, chunks=(replace(
                chunk, evidence=replace(chunk.evidence, boundary=boundary - 1),
            ),))

        def unexpected(*args, **kwargs):
            pytest.fail("invalid proof reached publication or local dependency validation")

        monkeypatch.setattr(SourceSessionStore, "commit_dependent_context_checkpoint", unexpected)
        monkeypatch.setattr(dependencies, "validate_summary_dependencies", unexpected)
        before = writer.owner.read_snapshot()
        epoch, active = state.source.projection_epoch, state.source.transcript
        # The public view is a deepcopy; pin the actual owned projection too.
        owned_projection = state.source._transcript
        raw = state.source_compaction.raw_messages
        with pytest.raises(SourceValidationError):
            governor.publish_source_compaction(
                state, attempt, boundary=boundary, summary=summary, auxiliary_usage=None, proof=proof,
            )
        assert writer.owner.read_snapshot() == before
        assert state.source.transcript == active and state.source.projection_epoch == epoch
        assert state.source._transcript is owned_projection
        assert state.source_compaction.raw_messages is raw and raw == active.messages()
        assert not state.source.has_pending_requests


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("with_proof", [False, True])
async def test_empty_references_need_no_dependent_dispatch(store, monkeypatch, local, with_proof) -> None:
    async with acquire_execution(store, "test:compact", run_id="run") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="run")
        if local:
            writer = writer.child_stream()
        state = state_for(writer, local=local)
        governor = ContextGovernor()
        attempt = capture(governor, state)
        summary = "independent historical facts"
        proof = None
        if with_proof:
            proof = SummaryDependencyProof(attempt.candidate, (ConsumedSummaryChunk(
                attempt.candidate.summary_evidence(2), None, summary,
            ),))

        def unexpected(*args, **kwargs):
            pytest.fail("empty references must not enter a dependent branch")

        monkeypatch.setattr(SourceSessionStore, "commit_dependent_context_checkpoint", unexpected)
        monkeypatch.setattr(dependencies, "validate_summary_dependencies", unexpected)
        before = owner.read_snapshot()
        view = governor.publish_source_compaction(
            state, attempt, boundary=2, summary=summary, auxiliary_usage=None, proof=proof,
        )
        after = owner.read_snapshot()
        assert state.source_view == view
        assert state.source.transcript.messages() == state.source_compaction.raw_messages
        if local:
            assert after == before
            assert state.source_compaction.local_checkpoint.artifact_references == ()
        else:
            assert after.checkpoint["schema_version"] == 1
            assert after.checkpoint["artifact_references"] == []
            assert after.state.revision == before.state.revision + 1
