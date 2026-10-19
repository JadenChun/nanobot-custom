"""Dependent checkpoints recover canonical evidence without widening authorization."""

from copy import deepcopy
from dataclasses import replace

import pytest

from nanobot.session.artifact import ArtifactValidationError
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.artifact_rematerialization import ArtifactRematerializer
from nanobot.session.checkpoint_admission import CheckpointAdmission
from nanobot.session.source import SourceValidationError
from nanobot.session.source_compaction import SourceCompactionCandidate
from nanobot.session.source_projection import CheckpointNode, RuntimeNode, SourceNode, source_nodes
from nanobot.session.source_request import SourceRequestLedger, SourceTranscript
from tests.session.test_artifact_reader import setup_read


def checkpoint_snapshot(writer, descriptor):
    snapshot = writer.owner.read_snapshot()
    checkpoint = {'schema_version': 2, 'projection_scope': 'root', 'projection_version': 1,
                  'checkpoint_id': 'dependent', 'summary': 'Historical evidence',
                  'boundary': len(snapshot.events), 'artifact_references': [descriptor.reference]}
    return replace(snapshot, checkpoint=checkpoint)


@pytest.mark.parametrize('failure', ['missing', 'corrupt', 'no_tools'])
async def test_admission_and_authorized_recovery(tmp_path, failure):
    async with setup_read(tmp_path, 'evidence' * 6000) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        snapshot = checkpoint_snapshot(writer, descriptor)
        full = SourceTranscript((RuntimeNode('policy', {'role': 'system', 'content': 'Policy'}),
                                 *source_nodes(snapshot.events)))
        admission = CheckpointAdmission(reader=reader)
        reduced = admission.select(snapshot, full)
        assert reduced != full
        assert admission.checkpoint.artifact_references == (descriptor.reference,)
        assert descriptor.reference in str(reduced.messages())
        if failure != 'no_tools':
            path = next(store.root.rglob('*.blob'))
            if failure == 'missing':
                path.unlink()
            else:
                path.write_bytes(b'corrupt')
        assert not admission.validate_dependencies(snapshot, reduced, enabled=failure != 'no_tools')
        admission.reject()
        assert admission.boundary == 0 and admission.checkpoint is None
        assert admission.select(snapshot, full) == full
        # Canonical source and physical checkpoint metadata never change.
        assert writer.owner.read_snapshot().events == snapshot.events


async def test_absent_reader_skips_dependent_but_accepts_legacy_checkpoint(tmp_path):
    async with setup_read(tmp_path, 'evidence') as (writer, _, _, descriptor, _):
        snapshot = checkpoint_snapshot(writer, descriptor)
        full = SourceTranscript(source_nodes(snapshot.events))
        admission = CheckpointAdmission()
        assert admission.select(snapshot, full) == full
        legacy = replace(snapshot, checkpoint={**snapshot.checkpoint,
                         'schema_version': 1, 'artifact_references': []})
        assert CheckpointAdmission().select(legacy, full) != full


async def test_dependent_checkpoint_admits_only_after_explicit_rematerialization(tmp_path):
    async with setup_read(tmp_path, 'evidence' * 6000) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        snapshot = checkpoint_snapshot(writer, descriptor)
        full = SourceTranscript(source_nodes(snapshot.events))
        missing = CheckpointAdmission(reader=reader)
        reduced = missing.select(snapshot, full)
        source_path = writer.store._backend.get_session_path(writer.owner.key)
        before = source_path.read_bytes()
        next(store.root.rglob('*.blob')).unlink()

        assert not missing.validate_dependencies(snapshot, reduced, enabled=True)
        result = ArtifactRematerializer(reader).rematerialize(
            descriptor.reference, expected=snapshot.state
        )
        admitted = CheckpointAdmission(reader=reader)
        retried = admitted.select(snapshot, full)
        assert result.status == 'installed'
        assert admitted.validate_dependencies(snapshot, retried, enabled=True)
        assert source_path.read_bytes() == before


@pytest.mark.parametrize('failure', ['revoked', 'manifest', 'scope'])
async def test_authorization_failure_cannot_expand_inline(tmp_path, failure):
    async with setup_read(tmp_path, 'evidence') as (writer, tool, store, descriptor, _):
        snapshot = checkpoint_snapshot(writer, descriptor)
        full = SourceTranscript(source_nodes(snapshot.events))
        admission = CheckpointAdmission(reader=ArtifactReader(writer, 'root_history', store, tool))
        reduced = admission.select(snapshot, full)
        if failure == 'revoked':
            tool._workspace = tmp_path / 'revoked'
        elif failure == 'manifest':
            snapshot = deepcopy(snapshot)
            snapshot.header['metadata'].pop('artifact_manifest')
        else:
            admission.reader = ArtifactReader(writer, 'isolated_stream', store, tool)
            # Exact stream authorization still applies to the retained verifier.
            original = writer.stream_id
            writer.stream_id = 'other-stream'
        try:
            with pytest.raises(ArtifactValidationError):
                admission.validate_dependencies(snapshot, reduced, enabled=False)
        finally:
            if failure == 'scope':
                writer.stream_id = original


def test_recovery_preserves_expansion_runtime_and_eligibility():
    expanded = SourceNode('user', {'role': 'user', 'content': 'User + authorized expansion'})
    full = SourceTranscript((RuntimeNode('policy', {'role': 'system', 'content': 'Policy'}),
                             expanded, SourceNode('old', {'role': 'assistant', 'content': 'Old'})))
    ledger = SourceRequestLedger(full, initial_eligible_ids=frozenset({'user'}))
    ledger.replace_projection(SourceTranscript((full.nodes[0], CheckpointNode('summary', 'Facts'), expanded)))
    ledger.append_source('pending', {'role': 'assistant', 'content': 'Pending'})
    runtime = ledger.append_runtime({'role': 'user', 'content': 'Continue'})
    note = RuntimeNode('retry', {'role': 'user', 'content': 'Retry once'})
    ticket = ledger.issue(ledger.transcript.with_runtime(note))
    with pytest.raises(SourceValidationError, match='pending'):
        ledger.restore_recovery()
    ledger.discard(ticket)
    ledger.restore_recovery()
    assert ledger.transcript.nodes == (*full.nodes, SourceNode('pending', {'role': 'assistant', 'content': 'Pending'}), runtime)
    assert ledger.eligible_ids == frozenset({'user'})
    assert 'Retry once' not in str(ledger.transcript)
    detached = ledger.recovery_transcript
    detached.nodes[1].message['content'] = 'mutation'
    assert ledger.transcript.nodes[1] == expanded


async def test_capture_never_rediscovers_rejected_disk_checkpoint(tmp_path):
    async with setup_read(tmp_path, 'evidence') as (writer, _, _, descriptor, _):
        snapshot = checkpoint_snapshot(writer, descriptor)
        full = SourceTranscript(source_nodes(snapshot.events))
        candidate = SourceCompactionCandidate.capture(snapshot, full,
            eligible_ids=frozenset(node.event_id for node in full.nodes))
        assert candidate.previous_boundary == 0 and candidate.previous_checkpoint is None
        assert candidate.snapshot.state == snapshot.state


@pytest.mark.parametrize('failure', ['missing', 'no_tools', 'revoked'])
async def test_governor_recovery_updates_raw_payload_and_retains_retry_once(tmp_path, failure):
    from nanobot.agent.context_governance import (
        ContextGovernanceConfig,
        ContextGovernor,
        ModelRequestState,
        SourceCompactionState,
    )
    from nanobot.agent.tools.artifact import ReadArtifactTool
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.providers.base import LLMResponse, LLMUsage
    from nanobot.providers.conversation_state import ProviderConversationStateController
    from tests.agent.test_source_summary import SummaryProvider, runtime

    async with setup_read(tmp_path, 'evidence') as (writer, tool, store, descriptor, _):
        snapshot = checkpoint_snapshot(writer, descriptor)
        full = SourceTranscript(source_nodes(snapshot.events))
        source = SourceRequestLedger(full, initial_eligible_ids=frozenset())
        reader = ArtifactReader(writer, 'root_history', store, tool)
        admission = CheckpointAdmission(reader=reader)
        reduced = admission.select(snapshot, full)
        source.replace_projection(reduced)
        raw = reduced.messages()
        registry = ToolRegistry()
        registry.register(ReadArtifactTool(reader))
        provider = SummaryProvider(LLMResponse(content='unused'))
        state = ModelRequestState(
            ContextGovernanceConfig(provider, 'fixture', registry, None, writer.owner.key, 16000,
                context_window_tokens=8192, max_tokens=256, preserve_source=True),
            ProviderConversationStateController(provider=provider, model='fixture', messages=raw),
            source=source, source_compaction=SourceCompactionState(writer, runtime(provider), raw, None,
                                                                 admission=admission))
        state.usage = LLMUsage.empty_request()
        state.messages = reduced.messages()
        if failure == 'missing':
            next(store.root.rglob('*.blob')).unlink()
        elif failure == 'revoked':
            tool._workspace = tmp_path / 'revoked'
        note = RuntimeNode('retry', {'role': 'user', 'content': 'Retry exactly once'})
        view = source.transcript.with_runtime(note)
        governor = ContextGovernor()
        if failure == 'revoked':
            with pytest.raises(ArtifactValidationError):
                await governor.prepare_request(state, view.messages(), source_view=view, tool_definitions=None)
            assert source.transcript == reduced and raw == reduced.messages()
            return
        prepared, _ = await governor.prepare_request(state, view.messages(), source_view=view,
            tool_definitions=None if failure == 'no_tools' else registry.get_definitions())
        assert raw == full.messages() and source.transcript == full
        assert prepared == [*full.messages(), note.message]
        assert source.eligible_ids == frozenset() and state.usage is None
        assert admission.checkpoint is None and admission.boundary == 0
