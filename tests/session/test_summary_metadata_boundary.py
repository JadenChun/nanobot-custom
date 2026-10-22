"""Typed summary metadata is observation, never publication authority."""

from __future__ import annotations

from contextlib import aclosing
from dataclasses import replace

import pytest

from nanobot.session import checkpoint_dependencies as dependencies
from nanobot.session.artifact import ArtifactValidationError
from nanobot.session.artifact_store import ArtifactStorageError
from nanobot.session.source import SourceValidationError
from nanobot.session.source_store import SourceSessionStore, SourceSnapshot
from tests.session.test_checkpoint_dependency_publication import publication_inputs


async def test_selected_validator_reconstructs_real_metadata_without_writing(tmp_path):
    selected = dependencies.select_summary_dependency_metadata_validator()
    assert selected is dependencies._validated_proof
    async with aclosing(publication_inputs(tmp_path)) as inputs:
        async for writer, reader, store, candidate, proof, proposal in inputs:
            path = writer.store._backend.get_session_path(writer.owner.key)
            before = path.read_bytes()
            metadata = selected(proof, candidate.snapshot)
            assert metadata.references == tuple(proposal['artifact_references'])
            assert metadata.boundary == proposal['boundary']
            assert metadata.summary == proposal['summary']
            assert metadata == dependencies._validated_proof(proof, candidate.snapshot)
            assert path.read_bytes() == before


async def test_selector_retains_the_current_original_callable_and_result(tmp_path, monkeypatch):
    async with aclosing(publication_inputs(tmp_path)) as inputs:
        async for writer, reader, store, candidate, proof, proposal in inputs:
            expected = dependencies._validated_proof(proof, candidate.snapshot)
            calls = []

            def validate(claim, snapshot):
                calls.append((claim, snapshot))
                return expected

            monkeypatch.setattr(dependencies, '_validated_proof', validate)
            selected = dependencies.select_summary_dependency_metadata_validator()
            assert selected is validate
            assert selected(proof, candidate.snapshot) is expected
            assert calls == [(proof, candidate.snapshot)]


async def test_selector_retains_original_fault_identity(tmp_path, monkeypatch):
    async with aclosing(publication_inputs(tmp_path)) as inputs:
        async for writer, reader, store, candidate, proof, proposal in inputs:
            fault = SourceValidationError('original proof seam')

            def validate(claim, snapshot):
                assert claim is proof and snapshot is candidate.snapshot
                raise fault

            monkeypatch.setattr(dependencies, '_validated_proof', validate)
            selected = dependencies.select_summary_dependency_metadata_validator()
            with pytest.raises(SourceValidationError) as failure:
                selected(proof, candidate.snapshot)
            assert failure.value is fault


@pytest.mark.parametrize('failure,message', [
    ('no_chunks', 'no consumed chunks'),
    ('fold_chain', 'fold chain changed'),
    ('not_contiguous', 'not contiguous'),
    ('empty_summary', 'Invalid consumed summary output'),
    ('whitespace_summary', 'Invalid consumed summary output'),
    ('missing_retrieval', 'differs from canonical source'),
    ('different_snapshot', 'source changed'),
    ('object_claim', 'Invalid summary dependency proof'),
])
async def test_selected_validator_retains_real_fold_refusals(tmp_path, failure, message):
    selected = dependencies.select_summary_dependency_metadata_validator()
    async with aclosing(publication_inputs(tmp_path)) as inputs:
        async for writer, reader, store, candidate, proof, proposal in inputs:
            snapshot = candidate.snapshot
            chunk = proof.chunks[0]
            if failure == 'no_chunks':
                proof = replace(proof, chunks=())
            elif failure == 'fold_chain':
                proof = replace(proof, chunks=(replace(chunk, previous_summary='foreign'),))
            elif failure == 'not_contiguous':
                proof = replace(proof, chunks=(replace(chunk,
                    evidence=replace(chunk.evidence, after=chunk.evidence.after + 1)),))
            elif failure in {'empty_summary', 'whitespace_summary'}:
                proof = replace(proof, chunks=(replace(chunk,
                    summary='' if failure == 'empty_summary' else '   '),))
            elif failure == 'missing_retrieval':
                proof = replace(proof, chunks=(replace(chunk,
                    evidence=replace(chunk.evidence, retrievals=())),))
            elif failure == 'different_snapshot':
                snapshot = replace(snapshot, state=replace(snapshot.state,
                    revision=snapshot.state.revision + 1))
            else:
                proof = object()
            path = writer.store._backend.get_session_path(writer.owner.key)
            before = path.read_bytes()
            with pytest.raises(SourceValidationError, match=message):
                selected(proof, snapshot)
            assert path.read_bytes() == before


async def test_selected_metadata_does_not_admit_a_closed_blob_transaction(tmp_path):
    selected = dependencies.select_summary_dependency_metadata_validator()
    async with aclosing(publication_inputs(tmp_path)) as inputs:
        async for writer, reader, store, candidate, proof, proposal in inputs:
            assert selected(proof, candidate.snapshot).references
            path = writer.store._backend.get_session_path(writer.owner.key)
            before = path.read_bytes()
            with store.transaction() as blobs:
                pass
            with pytest.raises(ArtifactStorageError):
                writer.store.commit_dependent_context_checkpoint(
                    writer.owner.key, candidate.snapshot.state, proposal, proof,
                    owner=writer.owner, reader=reader, blobs=blobs,
                )
            assert path.read_bytes() == before


async def test_workspace_metadata_observes_the_live_owned_backend(tmp_path, monkeypatch):
    async with aclosing(publication_inputs(tmp_path)) as inputs:
        async for writer, reader, store, candidate, proof, proposal in inputs:
            source = writer.store
            assert source.artifact_workspace_id == store.workspace_id
            changed = source._backend.sessions_dir / 'live-metadata-only'
            assert not changed.exists()
            with monkeypatch.context() as scoped:
                scoped.setattr(source._backend, 'sessions_dir', changed)
                assert source.artifact_workspace_id == changed.name
                assert not changed.exists()
            assert source.artifact_workspace_id == store.workspace_id


@pytest.mark.parametrize('different', [False, True])
async def test_source_identity_query_keeps_exact_snapshot_comparison(tmp_path, different):
    async with aclosing(publication_inputs(tmp_path)) as inputs:
        async for writer, reader, store, candidate, proof, proposal in inputs:
            snapshot = candidate.snapshot
            if different:
                snapshot = replace(snapshot, state=replace(snapshot.state,
                    store_identity=snapshot.state.store_identity + '-different'))
            assert writer.store.source_snapshot_identity_differs(snapshot) is different


@pytest.mark.parametrize('fault_at', [None, 'store', 'snapshot'])
async def test_identity_query_reads_store_before_snapshot_and_keeps_faults(
    tmp_path, monkeypatch, fault_at,
):
    async with aclosing(publication_inputs(tmp_path)) as inputs:
        async for writer, reader, store, candidate, proof, proposal in inputs:
            trace = []
            fault = RuntimeError('identity observation fault')
            original_store = SourceSessionStore.__getattribute__
            original_snapshot = SourceSnapshot.__getattribute__

            def source_attribute(self, name):
                if name == '_store_identity':
                    trace.append('store')
                    if fault_at == 'store':
                        raise fault
                return original_store(self, name)

            def snapshot_attribute(self, name):
                if name == 'state':
                    trace.append('snapshot')
                    if fault_at == 'snapshot':
                        raise fault
                return original_snapshot(self, name)

            with monkeypatch.context() as scoped:
                scoped.setattr(SourceSessionStore, '__getattribute__', source_attribute)
                scoped.setattr(SourceSnapshot, '__getattribute__', snapshot_attribute)
                if fault_at:
                    with pytest.raises(RuntimeError) as failure:
                        writer.store.source_snapshot_identity_differs(candidate.snapshot)
                    assert failure.value is fault
                else:
                    assert writer.store.source_snapshot_identity_differs(candidate.snapshot) is False
                assert trace == (['store'] if fault_at == 'store' else ['store', 'snapshot'])


@pytest.mark.parametrize('failure', [None, 'blobs', 'workspace', 'identity'])
async def test_reader_binding_retains_short_circuit_and_observation_order(
    tmp_path, monkeypatch, failure,
):
    async with aclosing(publication_inputs(tmp_path)) as inputs:
        async for writer, reader, store, candidate, proof, proposal in inputs:
            trace = []
            workspace_property = SourceSessionStore.artifact_workspace_id
            identity_query = SourceSessionStore.source_snapshot_identity_differs
            with store.transaction() as blobs:
                matches = blobs.matches_store

                def match(configured):
                    trace.append('blobs')
                    return False if failure == 'blobs' else matches(configured)

                def workspace(source):
                    trace.append('workspace')
                    if failure == 'workspace':
                        return 'deliberately-mismatched-workspace'
                    return workspace_property.__get__(source, SourceSessionStore)

                def identity(source, snapshot):
                    trace.append('identity')
                    return True if failure == 'identity' else identity_query(source, snapshot)

                with monkeypatch.context() as scoped:
                    scoped.setattr(blobs, 'matches_store', match)
                    scoped.setattr(SourceSessionStore, 'artifact_workspace_id', property(workspace))
                    scoped.setattr(SourceSessionStore, 'source_snapshot_identity_differs', identity)
                    if failure:
                        with pytest.raises(ArtifactValidationError, match='binding changed'):
                            dependencies._validate_reader_binding(reader, candidate.snapshot, blobs, candidate)
                    else:
                        dependencies._validate_reader_binding(reader, candidate.snapshot, blobs, candidate)
                    expected = ['blobs'] if failure == 'blobs' else (
                        ['blobs', 'workspace'] if failure == 'workspace' else
                        ['blobs', 'workspace', 'identity'])
                    assert trace == expected


async def test_publisher_selects_validator_before_real_source_snapshot_reads(tmp_path, monkeypatch):
    async with aclosing(publication_inputs(tmp_path)) as inputs:
        async for writer, reader, store, candidate, proof, proposal in inputs:
            original_selector = dependencies.select_summary_dependency_metadata_validator
            original_validate = dependencies._validated_proof
            original_snapshot = SourceSessionStore._snapshot
            selection = []
            rebound_calls = []
            rebound = False

            def select():
                selection.append('selected')
                return original_selector()

            def later_validator(claim, snapshot):
                rebound_calls.append((claim, snapshot))
                return original_validate(claim, snapshot)

            def snapshot_read(source, header, events):
                nonlocal rebound
                assert selection == ['selected']
                if not rebound:
                    monkeypatch.setattr(dependencies, '_validated_proof', later_validator)
                    rebound = True
                return original_snapshot(source, header, events)

            with monkeypatch.context() as scoped:
                scoped.setattr(dependencies, 'select_summary_dependency_metadata_validator', select)
                scoped.setattr(SourceSessionStore, '_snapshot', snapshot_read)
                with store.transaction() as blobs:
                    committed = writer.store.commit_dependent_context_checkpoint(
                        writer.owner.key, candidate.snapshot.state, proposal, proof,
                        owner=writer.owner, reader=reader, blobs=blobs,
                    )
            assert committed.checkpoint == proposal and rebound
            # Subsequent independent dependency reconstruction intentionally binds
            # late. Only it, not the publisher's first preselected validation, sees
            # this rebind during the original source snapshot fault seam.
            assert len(rebound_calls) == 1
            assert rebound_calls[0][0] is proof and rebound_calls[0][1] is candidate.snapshot
