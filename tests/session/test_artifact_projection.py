"""Artifact projection retains source evidence and synchronizes request ownership."""

from copy import deepcopy

import pytest

from nanobot.agent.tools.artifact import ReadArtifactTool
from nanobot.session.artifact import ArtifactValidationError
from nanobot.session.artifact_projection import ArtifactProjector
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.source_compaction import SourceCompactionCandidate
from nanobot.session.source_projection import RuntimeNode, SourceNode, source_nodes
from nanobot.session.source_request import SourceRequestLedger, SourceTranscript
from nanobot.session.source_store import SourceSessionStore
from tests.session.test_artifact_reader import setup_read


def active(writer):
    return SourceTranscript(source_nodes(writer.owner.read_snapshot().events))


async def test_projection_reuses_publication_and_restores_inline_without_writes(tmp_path):
    async with setup_read(tmp_path, 'evidence' * 6000) as (writer, tool, store, descriptor, _):
        projector = ArtifactProjector(ArtifactReader(writer, 'root_history', store, tool))
        original = active(writer)
        before = writer.owner.read_snapshot()
        projected = projector.prepare(original, enabled=True)
        assert projected.nodes[-1].artifact.artifact_id == descriptor.artifact_id
        assert len(projected.nodes[-1].message['content']) < 1600
        assert projected.nodes[-1].event_id == original.nodes[-1].event_id
        assert projector.prepare(projected, enabled=True) == projected
        assert ArtifactProjector(projector.reader).prepare(original, enabled=True) == projected
        assert projector.prepare(projected, enabled=False) == original
        assert writer.owner.read_snapshot() == before


@pytest.mark.parametrize('failure', ['missing', 'corrupt'])
async def test_storage_failure_restores_authorized_inline_text(tmp_path, failure):
    async with setup_read(tmp_path, 'evidence' * 6000) as (writer, tool, store, _, _):
        projector = ArtifactProjector(ArtifactReader(writer, 'root_history', store, tool))
        original = active(writer)
        projected = projector.prepare(original, enabled=True)
        path = next(store.root.rglob('*.blob'))
        if failure == 'missing':
            path.unlink()
        else:
            path.write_bytes(b'corrupt')
        assert projector.prepare(projected, enabled=True) == original


async def test_namespace_quota_failure_retains_authorized_inline_text(tmp_path):
    import hashlib
    from dataclasses import replace

    from nanobot.session.artifact_store import PrivateArtifactStore

    async with setup_read(tmp_path, 'evidence' * 6000) as (writer, tool, _store, descriptor, content):
        backend = writer.store._backend
        with backend.locked_session_files():
            header, events = writer.store._read_unlocked(writer.owner.key)
            header['metadata'].pop('artifact_manifest', None)
            writer.store._write(header, events)
        quota_store = PrivateArtifactStore(
            tmp_path.resolve() / 'quota-artifacts',
            workspace_id=backend.sessions_dir.name,
            session_key=descriptor.session_key,
            incarnation=descriptor.incarnation,
            generation=descriptor.generation,
            max_blob_bytes=len(content),
            max_namespace_bytes=len(content) + 4,
        )
        filler = replace(
            descriptor,
            artifact_id='c' * 32,
            byte_length=5,
            sha256=hashlib.sha256(b'usage').hexdigest(),
        )
        with quota_store.transaction() as blobs:
            blobs.put(filler, b'usage')
        projector = ArtifactProjector(ArtifactReader(writer, 'root_history', quota_store, tool))
        original = active(writer)

        assert projector.prepare(original, enabled=True) == original
        scope = quota_store.root.joinpath(*quota_store._parts)
        assert {entry.name for entry in scope.iterdir()} == {'.lock', filler.artifact_id + '.blob'}


async def test_revocation_cannot_restore_inline_as_an_access_bypass(tmp_path):
    async with setup_read(tmp_path, 'evidence' * 6000) as (writer, tool, store, _, _):
        projector = ArtifactProjector(ArtifactReader(writer, 'root_history', store, tool))
        projected = projector.prepare(active(writer), enabled=True)
        tool._workspace = tmp_path / 'different'
        for enabled in (True, False):
            with pytest.raises(ArtifactValidationError):
                projector.prepare(projected, enabled=enabled)


async def test_cas_conflict_keeps_inline_and_never_rebases_or_retries(tmp_path, monkeypatch):
    async with setup_read(tmp_path, 'evidence' * 6000) as (writer, tool, store, _, _):
        backend = writer.store._backend
        with backend.locked_session_files():
            header, events = writer.store._read_unlocked(writer.owner.key)
            header['metadata'].pop('artifact_manifest')
            writer.store._write(header, events)
        original = active(writer)
        commit = SourceSessionStore.commit_artifact_descriptor
        calls = []
        def conflict(source, key, expected, descriptor, **kwargs):
            calls.append(expected)
            source.update_metadata(key, {'other': 'writer'})
            return commit(source, key, expected, descriptor, **kwargs)
        monkeypatch.setattr(SourceSessionStore, 'commit_artifact_descriptor', conflict)
        projector = ArtifactProjector(ArtifactReader(writer, 'root_history', store, tool))
        assert projector.prepare(original, enabled=True) == original
        assert len(calls) == 1
        assert 'artifact_manifest' not in writer.owner.read_snapshot().header['metadata']


async def test_governor_replacement_keeps_runtime_notes_and_no_tools_obeys_hard_budget(tmp_path):
    from nanobot.agent.context_governance import (
        ContextGovernanceConfig,
        ContextGovernor,
        ContextWindowExceededError,
        ModelRequestState,
        SourceCompactionState,
    )
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.providers.base import LLMResponse, LLMUsage
    from nanobot.providers.conversation_state import ProviderConversationStateController
    from tests.agent.test_source_summary import SummaryProvider, runtime

    async with setup_read(tmp_path, 'evidence ' * 12000) as (writer, tool, store, _, _):
        original = active(writer)
        source = SourceRequestLedger(original, initial_eligible_ids=frozenset())
        raw = original.messages()
        registry = ToolRegistry()
        registry.register(ReadArtifactTool(ArtifactReader(writer, 'root_history', store, tool)))
        provider = SummaryProvider(LLMResponse(content='unused'))
        state = ModelRequestState(
            ContextGovernanceConfig(provider, 'fixture', registry, None, writer.owner.key, 16000,
                context_window_tokens=8192, max_tokens=256, preserve_source=True),
            ProviderConversationStateController(provider=provider, model='fixture', messages=raw),
            source=source, source_compaction=SourceCompactionState(writer, runtime(provider), raw, None))
        state.usage = LLMUsage.empty_request()
        note = RuntimeNode('retry', {'role': 'user', 'content': 'Retry instruction exactly once'})
        view = source.transcript.with_runtime(note)
        governor = ContextGovernor()
        prepared, _ = await governor.prepare_request(state, view.messages(),
            tool_definitions=registry.get_definitions(), source_view=view)
        assert prepared[-1] == note.message
        assert 'Retry instruction' not in str(raw)
        assert raw == source.transcript.messages() and state.usage is None
        assert source.projection_epoch == 1
        repeated = source.transcript.with_runtime(note)
        await governor.prepare_request(state, repeated.messages(),
            tool_definitions=registry.get_definitions(), source_view=repeated)
        assert source.projection_epoch == 1
        with pytest.raises(ContextWindowExceededError):
            await governor.prepare_request(state, source.transcript.messages(),
                                          tool_definitions=None, source_view=source.transcript)
        assert raw == original.messages() and source.projection_epoch == 2


async def test_summary_evidence_is_canonical_not_generated_reference(tmp_path):
    async with setup_read(tmp_path, 'evidence' * 6000) as (writer, tool, store, _, _):
        original = active(writer)
        projected = ArtifactProjector(ArtifactReader(writer, 'root_history', store, tool)).prepare(original, enabled=True)
        snapshot = writer.owner.read_snapshot()
        candidate = SourceCompactionCandidate.capture(snapshot, projected,
            eligible_ids=frozenset(node.event_id for node in projected.nodes))
        evidence = candidate.evidence(candidate.boundaries[-1])
        assert evidence == original.nodes
        assert 'Incomplete preview' not in str(evidence)
        summary_evidence = candidate.summary_evidence(candidate.boundaries[-1])
        assert summary_evidence.nodes == projected.nodes
        assert len(summary_evidence.projections) == 1
        assert 'Incomplete preview' in str(summary_evidence.nodes)
        modified = deepcopy(snapshot)
        modified.events[-1]['_artifact_read'] = {'reference': 'artifact:' + 'b' * 32}
        candidate = SourceCompactionCandidate.capture(modified, projected,
            eligible_ids=frozenset(node.event_id for node in projected.nodes))
        from nanobot.session.source import SourceValidationError
        with pytest.raises(SourceValidationError, match='dependencies'):
            candidate.evidence(candidate.boundaries[-1])


@pytest.mark.parametrize('tamper', ['binding', 'rendering', 'manifest', 'recursive', 'unsupported'])
async def test_summary_evidence_rejects_forged_projected_source(tmp_path, tamper):
    async with setup_read(tmp_path, 'evidence' * 6000) as (writer, tool, store, _, _):
        projected = ArtifactProjector(ArtifactReader(writer, 'root_history', store, tool)).prepare(
            active(writer), enabled=True,
        )
        snapshot = deepcopy(writer.owner.read_snapshot())
        nodes = list(projected.nodes)
        if tamper == 'binding':
            binding = nodes[-1].artifact
            nodes[-1] = SourceNode(nodes[-1].event_id, nodes[-1].message,
                                   type(binding)(binding.artifact_id, binding.event_id,
                                                 binding.execution_id, '0' * 64))
        elif tamper == 'rendering':
            nodes[-1] = SourceNode(nodes[-1].event_id,
                                   {**nodes[-1].message, 'content': 'forged preview'}, nodes[-1].artifact)
        elif tamper == 'manifest':
            snapshot.header['metadata'].pop('artifact_manifest')
        elif tamper == 'recursive':
            snapshot.events[-1]['_artifact_read'] = {'reference': 'artifact:' + 'b' * 32}
        else:
            snapshot.events[-1].pop('_source_origin')
        candidate = SourceCompactionCandidate.capture(
            snapshot, SourceTranscript(tuple(nodes)),
            eligible_ids=frozenset(node.event_id for node in nodes),
        )
        with pytest.raises(ArtifactValidationError):
            candidate.summary_evidence(candidate.boundaries[-1])


async def test_hidden_child_projection_is_excluded_from_root_summary_evidence(tmp_path):
    async with setup_read(tmp_path, 'child projected evidence ' * 2000, child=True) as (
            writer, tool, store, _, _):
        reader = ArtifactReader(writer, 'isolated_stream', store, tool)
        child = ArtifactProjector(reader).prepare(
            SourceTranscript(source_nodes(writer.owner.read_snapshot().events, stream_id=writer.stream_id)),
            enabled=True,
        )
        assert any(node.artifact is not None for node in child.nodes)
        snapshot = writer.owner.read_snapshot()
        root = SourceCompactionCandidate.capture(
            snapshot, SourceTranscript(source_nodes(snapshot.events)),
            eligible_ids=frozenset(event['_event_id'] for event in snapshot.events),
        )
        evidence = root.summary_evidence(root.boundaries[-1])
        assert evidence.nodes == () and evidence.projections == ()


def test_literal_artifact_text_does_not_create_typed_binding():
    node = SourceNode('event', {'role': 'user', 'content': 'artifact:' + 'b' * 32})
    assert node.artifact is None


async def test_owned_publication_loss_does_not_fall_back_to_transportable_inline(tmp_path, monkeypatch):
    from nanobot.session.execution_owner import ExecutionOwnershipError

    async with setup_read(tmp_path, 'evidence' * 6000) as (writer, tool, store, _, _):
        projector = ArtifactProjector(ArtifactReader(writer, 'root_history', store, tool))
        original = active(writer)
        def lost(*args, **kwargs):
            raise ExecutionOwnershipError()
        monkeypatch.setattr(ArtifactReader, 'verify_published', lost)
        with pytest.raises(ExecutionOwnershipError):
            projector.prepare(original, enabled=True)


@pytest.mark.parametrize('field,value', [('artifact_id', '../forged'), ('artifact_id', None),
    ('event_id', 'other'), ('content_sha256', 'bad'), ('block_index', 0)])
def test_ledger_rejects_malformed_artifact_binding(field, value):
    from dataclasses import replace

    from nanobot.session.source import SourceValidationError
    from nanobot.session.source_projection import ArtifactProjectionBinding

    binding = ArtifactProjectionBinding('b' * 32, 'event', 'execution', 'a' * 64)
    node = SourceNode('event', {'role': 'tool', 'content': 'reference'}, replace(binding, **{field: value}))
    with pytest.raises(SourceValidationError):
        SourceRequestLedger(SourceTranscript((node,)), initial_eligible_ids=frozenset())


async def test_offload_rejects_reordered_source_before_publication(tmp_path):
    async with setup_read(tmp_path, 'evidence' * 6000) as (writer, tool, store, _, _):
        original = active(writer)
        before = writer.owner.read_snapshot()
        with pytest.raises(ArtifactValidationError, match='order'):
            ArtifactProjector(ArtifactReader(writer, 'root_history', store, tool)).prepare(
                SourceTranscript(tuple(reversed(original.nodes))), enabled=True)
        assert writer.owner.read_snapshot() == before
