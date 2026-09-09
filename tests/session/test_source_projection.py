"""Projection scope and exact source boundaries for checkpoint governance."""

from copy import deepcopy

import pytest

from nanobot.session.source import SourceValidationError, new_source_event
from nanobot.session.source_projection import eligible_boundaries, project_source, source_nodes
from nanobot.session.source_store import SourceSnapshot, SourceState


def event(role, content, *, stream='root', parent=None, **extra):
    row = new_source_event({'role': role, 'content': content, **extra}, run_id='run')
    row.update(extra)
    row['_stream_id'] = stream
    if parent is not None:
        row['_parent_stream_id'] = parent
    return row


def snapshot(events, checkpoint=None):
    # Projection consumes only detached events and an already validated checkpoint.
    return SourceSnapshot(SourceState('fixture', 'test:projection', 'incarnation', 1, 0,
        None, 'fingerprint'), {}, events, checkpoint)


def checkpoint(boundary, **extra):
    return {'checkpoint_id': 'summary', 'summary': 'facts', 'boundary': boundary,
            'projection_scope': 'root', 'projection_version': 1, **extra}


def test_checkpoint_replays_newest_covered_user_once_and_keeps_source_unchanged():
    events = [event('user', 'old'), event('assistant', 'old answer'),
              event('user', 'keep this instruction'), event('assistant', 'progress'),
              event('assistant', 'latest')]
    original = deepcopy(events)
    projected = project_source(snapshot(events, checkpoint(4)))
    assert [row['content'] for row in projected.messages()][1:] == ['keep this instruction', 'latest']
    assert projected.pinned.event_id == events[2]['_event_id']
    assert projected.nodes[0].event_id == events[4]['_event_id']
    projected.messages()[-1]['content'] = 'changed'
    assert events == original


def test_uncovered_newest_user_is_not_duplicated():
    events = [event('user', 'old'), event('user', 'new')]
    projected = project_source(snapshot(events, checkpoint(1)))
    assert projected.pinned is None
    assert [row['content'] for row in projected.messages()][1:] == ['new']


@pytest.mark.parametrize('extra', [
    {'projection_scope': None}, {'projection_scope': 'child'},
    {'projection_version': 2}, {'projection_version': True},
])
def test_incompatible_scope_falls_back_to_full_source(extra):
    events = [event('user', 'original')]
    projection = project_source(snapshot(events, checkpoint(1, **extra)))
    assert projection.checkpoint is None
    assert projection.messages() == [{'role': 'user', 'content': 'original'}]


def test_unscoped_checkpoint_and_rollback_use_full_source():
    events = [event('user', 'original')]
    old = checkpoint(1)
    del old['projection_scope']
    assert project_source(snapshot(events, old)).checkpoint is None
    assert project_source(snapshot(events, checkpoint(1)), use_checkpoint=False).checkpoint is None


def test_child_and_fresh_streams_never_inherit_root_or_sibling_evidence():
    events = [event('user', 'root private'),
              event('user', 'assignment', stream='child', parent='root'),
              event('assistant', 'sibling private', stream='sibling', parent='root'),
              event('assistant', 'nested private', stream='nested', parent='child'),
              event('user', 'fresh', stream='fresh')]
    source = snapshot(events, checkpoint(5))
    child = project_source(source, stream_id='child')
    assert child.checkpoint is None and child.boundary == 0
    assert child.messages() == [{'role': 'user', 'content': 'assignment'}]
    fresh = project_source(source, stream_id='fresh')
    assert fresh.messages() == [{'role': 'user', 'content': 'fresh'}]
    assert [node.message['content'] for node in source_nodes(events)] == ['root private', 'fresh']


def test_global_prefix_closure_includes_hidden_child_executions():
    call = {'id': 'same', 'type': 'function', 'function': {'name': 'delegate', 'arguments': '{}'}}
    events = [event('user', 'request'),
              event('assistant', '', tool_calls=[call], _execution_ids=['parent-call']),
              event('user', 'assignment', stream='child', parent='root'),
              event('assistant', 'child done', stream='child', parent='root'),
              event('tool', 'result', tool_call_id='same', _execution_id='parent-call')]
    eligible = frozenset(node.event_id for node in source_nodes(events))
    assert eligible_boundaries(events, eligible_ids=eligible) == (1, 5)
    child_ids = frozenset(node.event_id for node in source_nodes(events, stream_id='child'))
    assert eligible_boundaries(events, eligible_ids=child_ids, stream_id='child') == (1, 2)


def test_equal_content_cannot_acknowledge_a_different_event():
    events = [event('user', 'same'), event('user', 'same'), event('assistant', 'end')]
    assert eligible_boundaries(events, eligible_ids=frozenset([events[0]['_event_id'],
        events[2]['_event_id']])) == (1,)


def test_boundary_only_advances_after_previous_prefix():
    events = [event('user', 'old'), event('assistant', 'new')]
    assert eligible_boundaries(events, previous_boundary=1,
        eligible_ids=frozenset([events[1]['_event_id']])) == (2,)


def test_summary_is_never_a_system_message():
    projected = project_source(snapshot([event('user', 'input')], checkpoint(1)))
    assert projected.messages()[0]['role'] == 'assistant'
    assert 'untrusted evidence' in projected.messages()[0]['content']


def test_invalid_execution_evidence_is_not_treated_as_an_open_prefix():
    events = [event('user', 'input'), event('tool', 'orphan',
              tool_call_id='missing', _execution_id='missing')]
    with pytest.raises(SourceValidationError, match='execution result'):
        eligible_boundaries(events, eligible_ids=frozenset(row['_event_id'] for row in events))


def test_checkpoint_projection_survives_real_store_reload(tmp_path):
    from datetime import datetime, timezone

    from nanobot.session.manager import JsonlSessionStore, Session
    from nanobot.session.source import prefix_digest

    backend = JsonlSessionStore(tmp_path / 'workspace', sessions_root=tmp_path / 'private')
    session = Session('test:projection')
    session.messages = [{'role': 'user', 'content': 'retain objective'},
                        {'role': 'assistant', 'content': 'evidence'}]
    backend.save(session)
    source = backend.source.upgrade_source_session(session.key, expected_revision=session.revision,
        expected_generation=session.generation, expected_incarnation=session.incarnation)
    proposal = checkpoint(2,
        schema_version=1, digest_version=1, summary_format_version=1,
        session_generation=source.state.generation, source_revision=source.state.revision,
        parent_checkpoint_id=None, producing_model='fixture', producing_provider='fixture',
        digest_algorithm='sha256', created_at=datetime.now(timezone.utc).isoformat(),
        artifact_references=[], boundary_event_id=source.events[-1]['_event_id'],
        prefix_digest=prefix_digest(source.events, 2), auxiliary_usage=None)
    backend.source.commit_context_checkpoint(session.key, source.state, proposal)
    reopened = JsonlSessionStore(tmp_path / 'workspace', sessions_root=tmp_path / 'private')
    reloaded = reopened.source.load_source_snapshot(session.key)
    assert reloaded.events == source.events
    projection = project_source(reloaded)
    assert projection.checkpoint.summary == 'facts'
    assert projection.pinned.event_id == source.events[0]['_event_id']
    assert projection.nodes == ()
