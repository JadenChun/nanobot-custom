"""Governor publication uses actual owned source storage and exact request budgets."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from agent.test_source_summary import SummaryProvider, runtime
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
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import JsonlSessionStore, Session, SessionWriteConflict
from nanobot.session.source import SourceValidationError
from nanobot.session.source_projection import RuntimeNode, project_source, source_nodes
from nanobot.session.source_request import SourceRequestLedger, SourceTranscript
from nanobot.session.turn_writer import SourceTurnWriter


@pytest.fixture
def store(tmp_path):
    backend = JsonlSessionStore(tmp_path)
    session = Session('test:compact')
    backend.save(session)
    backend.source.upgrade_source_session(session.key, expected_revision=session.revision,
        expected_generation=session.generation, expected_incarnation=session.incarnation)
    return backend


def state_for(writer, *, local=False):
    for role, content in [('user', 'objective'), ('assistant', 'old evidence'), ('assistant', 'pending')]:
        writer.append_message({'role': role, 'content': content})
    provider = SummaryProvider(LLMResponse(content='unused'))
    captured_runtime = runtime(provider)
    stream = writer.stream_id if local else None
    nodes = source_nodes(writer.owner.read_snapshot().events, stream_id=stream)
    active = SourceTranscript((RuntimeNode('policy', {'role': 'system', 'content': 'policy'}), *nodes))
    raw = active.messages()
    ledger = SourceRequestLedger(active, initial_eligible_ids=frozenset(n.event_id for n in nodes[:2]))
    state = ModelRequestState(
        ContextGovernanceConfig(provider, 'fixture', ToolRegistry(), None, 'test:compact', 16000,
                                context_window_tokens=8192, max_tokens=256, preserve_source=True),
        ProviderConversationStateController(provider=provider, model='fixture', messages=raw,
                                            session_id='test:compact'),
        source=ledger, source_compaction=SourceCompactionState(writer, captured_runtime, raw, stream))
    return state


def capture(governor, state, *, note='retry only', tools=None):
    view = state.source.transcript.with_runtime(RuntimeNode('retry', {'role': 'user', 'content': note}))
    return governor.capture_source_compaction(state, view, tool_definitions=tools)


def publish(governor, state, attempt, summary='retained facts'):
    return governor.publish_source_compaction(state, attempt, boundary=2, summary=summary,
                                              auxiliary_usage={'input_tokens': 10, 'output_tokens': 2})


async def test_root_publication_commits_before_replacing_exact_live_projection(store):
    governor = ContextGovernor()
    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        state = state_for(writer)
        raw = state.source_compaction.raw_messages
        alias = raw
        before = owner.read_snapshot()
        state.usage = LLMUsage.empty_request()
        attempt = capture(governor, state)
        view = publish(governor, state, attempt)
        after = owner.read_snapshot()
        assert after.events == before.events and after.state.revision == before.state.revision + 1
        assert after.checkpoint['source_revision'] == before.state.revision
        assert after.checkpoint['auxiliary_usage'] == {'input_tokens': 10, 'output_tokens': 2}
        assert after.checkpoint['boundary'] == 2
        assert alias is raw and raw == state.source.transcript.messages()
        assert state.source_view == view and state.messages == view.messages()
        assert raw[-1]['content'] == 'pending' and view.messages()[-1]['content'] == 'retry only'
        assert state.source.projection_epoch == 1 and not state.source.has_pending_requests
        assert state.usage is None and state.source_compaction.local_checkpoint is None
        assert 'old evidence' not in str(raw)
        assert project_source(after).checkpoint.summary == 'retained facts'
        ticket = state.source.issue(view)
        assert ticket.checkpoint_id == after.checkpoint['checkpoint_id']
        state.source.discard(ticket)
        writer.append_message({'role': 'assistant', 'content': 'after publication'})
        assert owner.read_snapshot().events[:len(before.events)] == before.events


@pytest.mark.parametrize('mode', ['fresh', 'child'])
async def test_local_publication_retains_root_metadata_and_allows_unrelated_append(store, mode):
    governor = ContextGovernor()
    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        parent = SourceTurnWriter(store.source, owner, run_id='run')
        parent.append_message({'role': 'user', 'content': 'root private'})
        root_state = state_for(parent)
        publish(governor, root_state, capture(governor, root_state), summary='root historical facts')
        writer = parent.child_stream() if mode == 'child' else SourceTurnWriter(store.source, owner, run_id='fresh')
        state = state_for(writer, local=True)
        attempt = capture(governor, state)
        # Child writers share their coordinator; an independent fresh root has
        # no sibling capable of appending during its invocation.
        if mode == 'child':
            parent.append_message({'role': 'assistant', 'content': 'unrelated root append'})
        before = owner.read_snapshot()
        publish(governor, state, attempt)
        after = owner.read_snapshot()
        assert after == before and after.checkpoint['summary'] == 'root historical facts'
        assert state.source_compaction.local_boundary == 2
        assert state.source_compaction.local_checkpoint.summary == 'retained facts'
        assert 'root private' not in str(state.messages)
        second = capture(governor, state)
        assert second.candidate.previous_boundary == 2
        assert second.candidate.previous_checkpoint == state.source_compaction.local_checkpoint
        assert second.candidate.boundaries == ()  # Pending assistant is still ineligible.


@pytest.mark.parametrize('drift', ['metadata', 'source', 'ledger', 'epoch', 'pending', 'raw'])
async def test_drift_rejects_publication_without_replacing_projection(store, drift):
    governor = ContextGovernor()
    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        state = state_for(writer)
        attempt = capture(governor, state)
        if drift == 'metadata':
            store.source.update_metadata('test:compact', {'title': 'concurrent title'})
        elif drift == 'source':
            writer.append_message({'role': 'assistant', 'content': 'concurrent evidence'})
        elif drift == 'ledger':
            state.source.append_runtime({'role': 'user', 'content': 'new instruction'})
        elif drift == 'epoch':
            state.source.replace_projection(state.source.transcript)
        elif drift == 'pending':
            state.source.issue(state.source.transcript)
        else:
            state.source_compaction.raw_messages.append({'role': 'user', 'content': 'unbound'})
        before = owner.read_snapshot()
        active = state.source.transcript
        with pytest.raises((SourceValidationError, SessionWriteConflict)):
            publish(governor, state, attempt)
        assert owner.read_snapshot() == before
        assert state.source.transcript == active


@pytest.mark.parametrize('oversized', ['retry', 'tools', 'summary'])
async def test_complete_request_budget_is_checked_before_checkpoint_write(store, oversized):
    governor = ContextGovernor()
    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        state = state_for(writer)
        attempt = capture(governor, state, note='large ' * 20000 if oversized == 'retry' else 'retry',
            tools=[{'type': 'function', 'function': {'name': 'x', 'description': 'large ' * 20000}}]
                  if oversized == 'tools' else None)
        before = owner.read_snapshot()
        active = state.source.transcript
        with pytest.raises((ContextWindowExceededError, SourceValidationError)):
            publish(governor, state, attempt, summary='large ' * 5000 if oversized == 'summary' else 'facts')
        assert owner.read_snapshot() == before and state.source.transcript == active


async def test_checkpoint_write_failure_keeps_old_memory_and_source(store, monkeypatch):
    governor = ContextGovernor()
    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        state = state_for(writer)
        attempt = capture(governor, state)
        before = owner.read_snapshot()
        raw = deepcopy(state.source_compaction.raw_messages)
        monkeypatch.setattr(writer.store, 'commit_context_checkpoint', Mock(side_effect=OSError('disk')))
        with pytest.raises(OSError):
            publish(governor, state, attempt)
        assert owner.read_snapshot() == before
        assert state.source.transcript == attempt.candidate.transcript
        assert state.source_compaction.raw_messages == raw


async def test_local_scoped_append_rejects_candidate_even_without_ledger_update(store):
    governor = ContextGovernor()
    async with acquire_execution(store, 'test:compact', run_id='run') as owner:
        writer = SourceTurnWriter(store.source, owner, run_id='run')
        state = state_for(writer, local=True)
        attempt = capture(governor, state)
        writer.append_message({'role': 'assistant', 'content': 'same stream changed'})
        before = owner.read_snapshot()
        with pytest.raises(SourceValidationError, match='Local source changed'):
            publish(governor, state, attempt)
        assert owner.read_snapshot() == before
        assert state.source.transcript == attempt.candidate.transcript
        assert state.source_compaction.local_boundary == 0
