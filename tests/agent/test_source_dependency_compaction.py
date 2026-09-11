"""Dependency-aware pressure compaction uses only trusted canonical receipts."""

from datetime import datetime, timezone

import pytest
from session.test_artifact_reader import setup_read

from agent.test_source_summary import SummaryProvider, runtime
from nanobot.agent.context_governance import (
    ContextGovernanceConfig,
    ContextGovernor,
    ContextWindowExceededError,
    ModelRequestState,
    SourceCompactionState,
)
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.tools.artifact import ReadArtifactTool
from nanobot.agent.tools.execution import KnownToolResult
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import TurnContext
from nanobot.providers.base import LLMResponse, LLMUsage
from nanobot.providers.conversation_state import ProviderConversationStateController
from nanobot.session.artifact import ArtifactValidationError
from nanobot.session.artifact_projection import ArtifactProjector
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.artifact_store import PrivateArtifactStore
from nanobot.session.checkpoint_admission import CheckpointAdmission
from nanobot.session.source import SourceValidationError, prefix_digest
from nanobot.session.source_projection import CheckpointNode, RuntimeNode, source_nodes
from nanobot.session.source_request import SourceRequestLedger, SourceTranscript
from nanobot.session.turn_writer import SourceTurnWriter


async def _record_retrieval(writer, reader, reference):
    outcome = await ToolRegistry.execute_prepared(ReadArtifactTool(reader), {'reference': reference}, object())
    writer.begin_tool_batch({'role': 'assistant', 'content': '', 'tool_calls': [{
        'id': 'artifact', 'type': 'function',
        'function': {'name': 'read_artifact', 'arguments': '{}'},
    }]})
    await writer.record_tool_result(KnownToolResult(
        0, 'artifact', 'read_artifact', outcome, outcome.content, {'status': 'ok'}, True,
        retrieval_receipt=outcome.retrieval_receipt,
    ))


def _state(writer, reader, *, local=False):
    writer.append_message({'role': 'user', 'content': 'current instruction'})
    stream = writer.stream_id if local else None
    nodes = source_nodes(writer.owner.read_snapshot().events, stream_id=stream)
    active = SourceTranscript((RuntimeNode('policy', {'role': 'system', 'content': 'policy'}), *nodes))
    registry = ToolRegistry()
    registry.register(ReadArtifactTool(reader))
    provider = SummaryProvider(LLMResponse(content='trusted retrieval summary', usage=LLMUsage(
        input_tokens=10, output_tokens=2, total_tokens=12, reported_tokens=12,
    )))
    state = ModelRequestState(
        ContextGovernanceConfig(provider, 'fixture', registry, None, writer.owner.key, 16000,
                                context_window_tokens=12000, max_tokens=256, preserve_source=True),
        ProviderConversationStateController(provider=provider, model='fixture', messages=active.messages()),
        source=SourceRequestLedger(active, initial_eligible_ids=frozenset(
            node.event_id for node in nodes[:-1]
        )),
        source_compaction=SourceCompactionState(writer, runtime(provider, 12000), active.messages(), stream),
    )

    def count(messages, tools, model):
        if messages[0]['content'].startswith('Summarize historical evidence'):
            return 100, 'fixture'
        return (6000 if 'Historical context summary' in str(messages) else 11000), 'fixture'

    provider.estimate_prompt_tokens = count
    return state


async def _prepare(state, definitions):
    return await ContextGovernor().prepare_request(
        state, state.source.transcript.messages(), source_view=state.source.transcript,
        transcript=state.source_compaction.raw_messages, tool_definitions=definitions,
    )


async def test_trusted_retrieval_pressure_compaction_publishes_v2_and_restart_admits(tmp_path):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        await _record_retrieval(writer, reader, descriptor.reference)
        state = _state(writer, reader)
        before = writer.owner.read_snapshot()
        definitions = state.config.tools.get_definitions()
        messages, _ = await _prepare(state, definitions)
        after = writer.owner.read_snapshot()

        assert len(state.config.provider.calls) == 1
        assert after.state.revision == before.state.revision + 1
        assert after.checkpoint['schema_version'] == 2
        assert after.checkpoint['artifact_references'] == [descriptor.reference]
        assert state.source_compaction.admission.fingerprint == after.state.checkpoint_fingerprint
        assert descriptor.reference in str(messages)

        restarted_writer = SourceTurnWriter(writer.store, writer.owner, run_id='restart')
        full = SourceTranscript(source_nodes(after.events))
        admitted = CheckpointAdmission(reader=ArtifactReader(restarted_writer, 'root_history', store, tool))
        selected = admitted.select(after, full)
        assert selected != full and admitted.checkpoint is not None
        assert admitted.checkpoint.artifact_references == (descriptor.reference,)


async def test_later_fold_carries_and_deduplicates_prior_dependency_without_model_reference(tmp_path):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        await _record_retrieval(writer, reader, descriptor.reference)
        state = _state(writer, reader)
        definitions = state.config.tools.get_definitions()
        await _prepare(state, definitions)
        first = writer.owner.read_snapshot()
        state.source.acknowledge(state.source.issue(state.source_view))
        event = writer.append_message({'role': 'assistant', 'content': 'later evidence'})
        state.source.append_source(event['_event_id'], {'role': 'assistant', 'content': 'later evidence'})
        state.source_compaction.raw_messages[:] = state.source.transcript.messages()
        state.source.acknowledge(state.source.issue(state.source.transcript))

        def count(messages, tools, model):
            if messages[0]['content'].startswith('Summarize historical evidence'):
                return 100, 'fixture'
            return (11_000 if 'later evidence' in str(messages) else 6_000), 'fixture'

        state.config.provider.estimate_prompt_tokens = count
        await _prepare(state, definitions)
        second = writer.owner.read_snapshot()
        assert second.checkpoint['parent_checkpoint_id'] == first.checkpoint['checkpoint_id']
        assert second.checkpoint['artifact_references'] == [descriptor.reference]
        assert state.source_compaction.local_checkpoint is None
        assert len(second.checkpoint['artifact_references']) == 1


async def test_new_runner_invocation_admits_dependent_checkpoint_before_transport(tmp_path):
    covered_result = 'covered retrieval result content for direct runner'
    checkpoint_summary = 'checkpoint summary for direct runner admission'
    async with setup_read(tmp_path, text=covered_result) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        await _record_retrieval(writer, reader, descriptor.reference)
        state = _state(writer, reader)
        state.config.provider.response = LLMResponse(content=checkpoint_summary)
        await _prepare(state, state.config.tools.get_definitions())
        restarted_writer = SourceTurnWriter(writer.store, writer.owner, run_id='restart')
        restarted_writer.append_message({'role': 'user', 'content': 'restart instruction'})
        snapshot = writer.owner.read_snapshot()
        registry = ToolRegistry()
        registry.register(tool)
        provider = SummaryProvider(LLMResponse(content='continued'))

        def factory(invocation_writer, isolated, read_file):
            return ArtifactReader(invocation_writer,
                                  'isolated_stream' if isolated else 'root_history', store, read_file)

        runner = AgentRunner(artifact_reader_factory=factory)
        result = await runner.run(AgentRunSpec(
            initial_messages=None, tools=registry, runtime=runtime(provider, 12000), max_iterations=1,
            max_tool_result_chars=16000, workspace=tool._workspace, session_key=writer.owner.key,
            turn_context=TurnContext(run_id='restart', session_key=writer.owner.key,
                                     execution_owner=writer.owner),
            source_writer=restarted_writer, source_transcript=SourceTranscript(source_nodes(snapshot.events)),
            source_initial_eligible_ids=frozenset(), artifact_read_allowed=True,
        ))
        assert result.error is None and result.final_content == 'continued'
        payload = str(provider.calls[0][0])
        assert checkpoint_summary in payload
        assert covered_result not in payload
        assert descriptor.reference in payload


@pytest.mark.parametrize('missing_blob', [False, True])
async def test_no_retrieval_schema_retains_canonical_source_without_dependent_publication(
        tmp_path, missing_blob):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        await _record_retrieval(writer, reader, descriptor.reference)
        state = _state(writer, reader)
        if missing_blob:
            next(store.root.rglob('*.blob')).unlink()
        before = writer.owner.read_snapshot()
        with pytest.raises(ContextWindowExceededError):
            await _prepare(state, None)
        assert state.config.provider.calls == []
        assert writer.owner.read_snapshot() == before
        assert descriptor.reference in str(state.source.transcript.messages())
        assert state.source.transcript.messages() == state.source_compaction.raw_messages
        assert not state.source.has_pending_requests


async def test_projected_source_summary_requires_retrieval_schema_without_compaction_writes(tmp_path):
    async with setup_read(tmp_path, 'projected evidence ' * 2000) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        state = _state(writer, reader)
        projected = ArtifactProjector(reader).prepare(state.source.transcript, enabled=True)
        state.source.replace_projection(projected)
        state.source_compaction.raw_messages[:] = projected.messages()
        state.conversation.replace_transcript(state.source_compaction.raw_messages)
        before = writer.owner.read_snapshot()

        retained = await ContextGovernor().compact_source_request(
            state, projected, tool_definitions=None,
        )

        assert retained == projected
        assert state.config.provider.calls == []
        assert writer.owner.read_snapshot() == before
        assert state.source.transcript == projected


@pytest.mark.parametrize('failure', ['no_schema', 'corrupt'])
async def test_projected_dependency_loss_restores_inline_without_auxiliary_compaction(tmp_path, failure):
    async with setup_read(tmp_path, 'projected evidence ' * 2000) as (writer, tool, store, _, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        state = _state(writer, reader)
        projected = ArtifactProjector(reader).prepare(state.source.transcript, enabled=True)
        state.source.replace_projection(projected)
        state.source_compaction.raw_messages[:] = projected.messages()
        state.conversation.replace_transcript(state.source_compaction.raw_messages)
        before = writer.owner.read_snapshot()
        definitions = None
        if failure == 'corrupt':
            next(store.root.rglob('*.blob')).write_bytes(b'corrupt')
            definitions = state.config.tools.get_definitions()

        with pytest.raises(ContextWindowExceededError):
            await _prepare(state, definitions)

        assert state.config.provider.calls == []
        assert writer.owner.read_snapshot().checkpoint == before.checkpoint
        assert state.source_compaction.artifact_projection_unavailable
        assert 'Tool output stored as' not in str(state.source.transcript.messages())


async def test_projected_source_summary_publishes_v2_without_artifact_publication(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from nanobot.session.source_store import SourceSessionStore

    async with setup_read(tmp_path, 'projected evidence ' * 2000) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        state = _state(writer, reader)
        projected = ArtifactProjector(reader).prepare(state.source.transcript, enabled=True)
        state.source.replace_projection(projected)
        state.source_compaction.raw_messages[:] = projected.messages()
        state.conversation.replace_transcript(state.source_compaction.raw_messages)
        publish_descriptor = Mock(side_effect=AssertionError('auxiliary compaction must not publish artifacts'))
        monkeypatch.setattr(SourceSessionStore, 'commit_artifact_descriptor', publish_descriptor)

        await _prepare(state, state.config.tools.get_definitions())

        checkpoint = writer.owner.read_snapshot().checkpoint
        assert checkpoint['schema_version'] == 2
        assert checkpoint['artifact_references'] == [descriptor.reference]
        assert publish_descriptor.call_count == 0
        auxiliary_payload = str(state.config.provider.calls[0][0])
        assert 'Tool output stored as ' + descriptor.reference in auxiliary_payload
        assert len(auxiliary_payload) < 4_000


async def test_proofless_publication_cannot_downgrade_projected_evidence_to_schema_v1(tmp_path):
    async with setup_read(tmp_path, 'projected evidence ' * 2000) as (writer, tool, store, _, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        state = _state(writer, reader)
        projected = ArtifactProjector(reader).prepare(state.source.transcript, enabled=True)
        state.source.replace_projection(projected)
        state.source_compaction.raw_messages[:] = projected.messages()
        attempt = ContextGovernor().capture_source_compaction(
            state, projected, tool_definitions=state.config.tools.get_definitions(),
        )
        before = writer.owner.read_snapshot()
        with pytest.raises(SourceValidationError, match='complete proof'):
            ContextGovernor().publish_source_compaction(
                state, attempt, boundary=attempt.candidate.boundaries[-1],
                summary='forged projected v1 summary', auxiliary_usage=None,
            )
        assert writer.owner.read_snapshot() == before


async def test_projected_dependency_cas_change_after_summary_never_publishes_projection(tmp_path):
    from nanobot.session.manager import SessionWriteConflict

    async with setup_read(tmp_path, 'projected evidence ' * 2000) as (writer, tool, store, _, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        state = _state(writer, reader)
        projected = ArtifactProjector(reader).prepare(state.source.transcript, enabled=True)
        state.source.replace_projection(projected)
        state.source_compaction.raw_messages[:] = projected.messages()
        state.conversation.replace_transcript(state.source_compaction.raw_messages)
        original_chat = state.config.provider.chat

        async def mutate_after_transport(*args, **kwargs):
            response = await original_chat(*args, **kwargs)
            writer.store.update_metadata(writer.owner.key, {'title': 'concurrent title'})
            return response

        state.config.provider.chat = mutate_after_transport
        before = writer.owner.read_snapshot()
        with pytest.raises(SessionWriteConflict):
            await _prepare(state, state.config.tools.get_definitions())
        after = writer.owner.read_snapshot()
        assert len(state.config.provider.calls) == 1
        assert after.events == before.events and after.checkpoint == before.checkpoint
        assert state.source.transcript == projected


async def test_isolated_projected_summary_keeps_root_checkpoint_and_stream_scope(tmp_path):
    async with setup_read(tmp_path, 'projected evidence ' * 2000, child=True) as (
            writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'isolated_stream', store, tool)
        state = _state(writer, reader, local=True)
        projected = ArtifactProjector(reader).prepare(state.source.transcript, enabled=True)
        state.source.replace_projection(projected)
        state.source_compaction.raw_messages[:] = projected.messages()
        state.conversation.replace_transcript(state.source_compaction.raw_messages)
        before = writer.owner.read_snapshot()

        await _prepare(state, state.config.tools.get_definitions())

        after = writer.owner.read_snapshot()
        assert after == before
        assert state.source_compaction.local_checkpoint is not None
        assert state.source_compaction.local_checkpoint.artifact_references == (descriptor.reference,)


async def test_revoked_dependency_preflight_never_transports_or_publishes(tmp_path):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        await _record_retrieval(writer, reader, descriptor.reference)
        state = _state(writer, reader)
        next(store.root.rglob('*.blob')).write_bytes(b'corrupt')
        before = writer.owner.read_snapshot()
        with pytest.raises((SourceValidationError, OSError)):
            await _prepare(state, state.config.tools.get_definitions())
        assert state.config.provider.calls == []
        assert writer.owner.read_snapshot() == before
        assert state.source.transcript.messages() == state.source_compaction.raw_messages
        assert not state.source.has_pending_requests


async def test_proofless_publication_cannot_downgrade_retrieval_evidence_to_schema_v1(tmp_path):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        await _record_retrieval(writer, reader, descriptor.reference)
        state = _state(writer, reader)
        governor = ContextGovernor()
        attempt = governor.capture_source_compaction(
            state, state.source.transcript, tool_definitions=state.config.tools.get_definitions(),
        )
        before = writer.owner.read_snapshot()
        with pytest.raises(SourceValidationError, match='complete proof'):
            governor.publish_source_compaction(
                state, attempt, boundary=attempt.candidate.boundaries[-1], summary='forged v1 summary',
                auxiliary_usage=None,
            )
        assert writer.owner.read_snapshot() == before
        assert state.source.transcript == attempt.candidate.transcript


async def test_dependent_validation_enters_source_transaction_only_under_blob_lock(tmp_path, monkeypatch):
    from contextlib import contextmanager

    import nanobot.session.execution_owner as ownership

    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        await _record_retrieval(writer, reader, descriptor.reference)
        state = _state(writer, reader)
        blob_active = False
        entered = 0
        original_transaction = PrivateArtifactStore.transaction
        original_execution = ownership.execution_transaction

        @contextmanager
        def marked_blob(self):
            nonlocal blob_active
            with original_transaction(self) as blobs:
                blob_active = True
                try:
                    yield blobs
                finally:
                    blob_active = False

        @contextmanager
        def checked_execution(*args, **kwargs):
            nonlocal entered
            assert blob_active
            entered += 1
            with original_execution(*args, **kwargs):
                yield

        monkeypatch.setattr(PrivateArtifactStore, 'transaction', marked_blob)
        monkeypatch.setattr(ownership, 'execution_transaction', checked_execution)
        await _prepare(state, state.config.tools.get_definitions())
        # One fence is immediately before summary transport and one wraps the
        # final dependent root commit.
        assert entered >= 2


async def test_root_dependency_cas_change_after_summary_never_publishes_projection(tmp_path):
    from nanobot.session.manager import SessionWriteConflict

    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'root_history', store, tool)
        await _record_retrieval(writer, reader, descriptor.reference)
        state = _state(writer, reader)
        original_chat = state.config.provider.chat

        async def mutate_after_transport(*args, **kwargs):
            response = await original_chat(*args, **kwargs)
            writer.store.update_metadata(writer.owner.key, {'title': 'concurrent title'})
            return response

        state.config.provider.chat = mutate_after_transport
        before = writer.owner.read_snapshot()
        active = state.source.transcript
        with pytest.raises(SessionWriteConflict):
            await _prepare(state, state.config.tools.get_definitions())
        after = writer.owner.read_snapshot()
        assert len(state.config.provider.calls) == 1
        assert after.events == before.events and after.checkpoint == before.checkpoint
        assert state.source.transcript == active


@pytest.mark.parametrize('revocation', ['corrupt_blob', 'filesystem_policy'])
async def test_dependency_change_after_first_fold_blocks_second_transport_and_publication(
        tmp_path, revocation):
    async with setup_read(tmp_path) as (writer, tool, store, descriptor, _):
        initial = writer.owner.read_snapshot()
        proposal = {
            'schema_version': 1, 'digest_version': 1, 'summary_format_version': 1,
            'session_generation': initial.state.generation, 'source_revision': initial.state.revision,
            'checkpoint_id': 'prior-checkpoint', 'parent_checkpoint_id': initial.state.checkpoint_id,
            'producing_model': 'fixture', 'producing_provider': 'fixture', 'digest_algorithm': 'sha256',
            'created_at': datetime.now(timezone.utc).isoformat(), 'summary': 'prior summary',
            'artifact_references': [], 'boundary': len(initial.events),
            'boundary_event_id': initial.events[-1]['_event_id'],
            'prefix_digest': prefix_digest(initial.events, len(initial.events)), 'auxiliary_usage': None,
            'projection_scope': 'root', 'projection_version': 1,
        }
        committed = writer.store.commit_context_checkpoint(
            writer.owner.key, initial.state, proposal, owner=writer.owner,
        )
        prior = CheckpointNode('prior-checkpoint', 'prior summary')
        reader = ArtifactReader(writer, 'root_history', store, tool)
        await _record_retrieval(writer, reader, descriptor.reference)
        writer.append_message({'role': 'assistant', 'content': 'later chunk'})
        writer.append_message({'role': 'user', 'content': 'current instruction'})
        snapshot = writer.owner.read_snapshot()
        suffix = source_nodes(snapshot.events[len(initial.events):])
        active = SourceTranscript((RuntimeNode('policy', {'role': 'system', 'content': 'policy'}),
                                   prior, *suffix))
        registry = ToolRegistry()
        registry.register(ReadArtifactTool(reader))
        provider = SummaryProvider(LLMResponse(content='trusted retrieval summary'))
        state = ModelRequestState(
            ContextGovernanceConfig(provider, 'fixture', registry, None, writer.owner.key, 16000,
                                    context_window_tokens=12000, max_tokens=256, preserve_source=True),
            ProviderConversationStateController(provider=provider, model='fixture', messages=active.messages()),
            source=SourceRequestLedger(active, initial_eligible_ids=frozenset(
                node.event_id for node in suffix[:-1]
            )),
            source_compaction=SourceCompactionState(
                writer, runtime(provider, 12000), active.messages(), None,
                admission=CheckpointAdmission(True, len(initial.events), prior,
                                              committed.state.checkpoint_fingerprint),
            ),
        )

        def count(messages, tools, model):
            if messages[0]['content'].startswith('Summarize historical evidence'):
                payload = messages[1]['content']
                return (20_000 if 'later chunk' in payload and 'prior summary' in payload else 100), 'fixture'
            text = str(messages)
            if 'Historical context summary' not in text or 'prior summary' in text:
                return 11_000, 'fixture'
            return (8_000 if 'later chunk' in text else 5_000), 'fixture'

        provider.estimate_prompt_tokens = count
        original_chat = provider.chat

        async def revoke_after_first_fold(*args, **kwargs):
            response = await original_chat(*args, **kwargs)
            if len(provider.calls) == 1:
                if revocation == 'corrupt_blob':
                    next(store.root.rglob('*.blob')).write_bytes(b'corrupt')
                else:
                    tool._workspace = tmp_path / 'revoked-workspace'
            return response

        provider.chat = revoke_after_first_fold
        with pytest.raises((OSError, ArtifactValidationError)):
            await _prepare(state, registry.get_definitions())
        assert len(provider.calls) == 1
        assert writer.owner.read_snapshot().checkpoint == proposal
        assert state.source.transcript == active


async def test_isolated_dependent_compaction_keeps_root_checkpoint_and_allows_root_append(tmp_path):
    async with setup_read(tmp_path, child=True) as (writer, tool, store, descriptor, _):
        reader = ArtifactReader(writer, 'isolated_stream', store, tool)
        await _record_retrieval(writer, reader, descriptor.reference)
        state = _state(writer, reader, local=True)
        parent = SourceTurnWriter(writer.store, writer.owner, run_id='root')
        original_chat = state.config.provider.chat

        async def append_unrelated(*args, **kwargs):
            response = await original_chat(*args, **kwargs)
            parent.append_message({'role': 'assistant', 'content': 'unrelated root append'})
            return response

        state.config.provider.chat = append_unrelated
        before = writer.owner.read_snapshot()
        await _prepare(state, state.config.tools.get_definitions())
        after = writer.owner.read_snapshot()
        assert after.checkpoint == before.checkpoint is None
        assert after.state.revision == before.state.revision + 1
        assert state.source_compaction.local_checkpoint is not None
        assert state.source_compaction.local_checkpoint.artifact_references == (descriptor.reference,)
