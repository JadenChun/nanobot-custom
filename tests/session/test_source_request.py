"""Request acknowledgment is explicit, ticket-scoped, and independent of output."""

from dataclasses import replace

import pytest

from nanobot.session.source import SourceValidationError
from nanobot.session.source_projection import CheckpointNode, RuntimeNode, SourceNode
from nanobot.session.source_request import SourceRequestLedger, SourceTranscript


def source(identity, text='same'):
    return SourceNode(identity, {'role': 'user', 'content': text})


def ledger():
    return SourceRequestLedger(SourceTranscript((source('history'), source('input'))),
                               initial_eligible_ids=frozenset({'history'}))


def test_request_issue_does_not_acknowledge_input_or_newly_recorded_output():
    state = ledger()
    ticket = state.issue(state.transcript)
    state.append_source('response', {'role': 'assistant', 'content': 'output'})
    assert state.eligible_ids == {'history'}
    state.acknowledge(ticket)
    assert state.eligible_ids == {'history', 'input'}
    assert 'response' not in state.eligible_ids


def test_malformed_outer_ticket_cannot_acknowledge_after_valid_retry():
    state = ledger()
    original = state.issue(state.transcript)
    state.discard(original)
    retry_view = state.transcript.with_runtime(RuntimeNode('correction', {'role': 'user', 'content': 'retry'}))
    retry = state.issue(retry_view)
    state.acknowledge(retry)
    assert state.eligible_ids == {'history', 'input'}
    with pytest.raises(SourceValidationError):
        state.acknowledge(original)
    assert len(state.transcript.nodes) == 2


def test_tickets_cannot_be_reused_forged_or_transferred_between_streams():
    state, child = ledger(), ledger()
    ticket = state.issue(state.transcript)
    with pytest.raises(SourceValidationError):
        child.acknowledge(ticket)
    with pytest.raises(SourceValidationError):
        state.acknowledge(replace(ticket, source_ids=('unseen',)))
    state.acknowledge(ticket)
    with pytest.raises(SourceValidationError):
        state.acknowledge(ticket)


def test_projection_replacement_invalidates_old_tickets_and_preserves_pending_runtime():
    state = ledger()
    runtime = state.append_runtime({'role': 'user', 'content': 'continue objective'})
    ticket = state.issue(state.transcript)
    projected = SourceTranscript((CheckpointNode('checkpoint', 'facts'), source('input'), runtime))
    state.replace_projection(projected)
    with pytest.raises(SourceValidationError):
        state.acknowledge(ticket)
    assert state.transcript.messages()[-1]['content'] == 'continue objective'
    assert state.eligible_ids == {'history'}
    assert state.issue(state.transcript).checkpoint_id == 'checkpoint'


def test_detached_transcripts_and_runtime_views_cannot_mutate_ledger():
    state = ledger()
    detached = state.transcript
    detached.nodes[0].message['content'] = 'forged'
    with pytest.raises(SourceValidationError):
        state.issue(detached)
    assert state.transcript.nodes[0].message['content'] == 'same'
    rendered = state.transcript.messages()
    rendered[0]['content'] = 'also forged'
    assert state.transcript.nodes[0].message['content'] == 'same'


def test_equal_content_does_not_grant_eligibility_to_other_identity():
    state = ledger()
    state.acknowledge(state.issue(state.transcript))
    state.append_source('new', {'role': 'user', 'content': 'same'})
    assert 'new' not in state.eligible_ids


def test_request_cannot_silently_drop_source_or_active_runtime():
    state = ledger()
    state.append_runtime({'role': 'user', 'content': 'active instruction'})
    with pytest.raises(SourceValidationError):
        state.issue(SourceTranscript(state.transcript.nodes[:-1]))
    with pytest.raises(SourceValidationError):
        state.issue(SourceTranscript(state.transcript.nodes[1:]))


def test_initial_eligibility_requires_explicit_source_node():
    with pytest.raises(SourceValidationError):
        SourceRequestLedger(SourceTranscript((source('a'),)), initial_eligible_ids=frozenset({'b'}))
    with pytest.raises(SourceValidationError):
        SourceRequestLedger(SourceTranscript((source('a'), source('a'))), initial_eligible_ids=frozenset())
