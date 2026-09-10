"""Compaction evidence is canonical; retained request nodes keep their provenance."""

from copy import deepcopy

import pytest

from nanobot.session.source import SourceValidationError, new_source_event
from nanobot.session.source_compaction import SourceCompactionCandidate
from nanobot.session.source_projection import CheckpointNode, RuntimeNode, SourceNode, source_nodes
from nanobot.session.source_request import SourceTranscript
from nanobot.session.source_store import SourceSnapshot, SourceState


def event(role, content, *, stream='root', parent=None, **extra):
    row = new_source_event({'role': role, 'content': content, **extra}, run_id='run')
    row.update(extra, _stream_id=stream)
    if parent is not None:
        row['_parent_stream_id'] = parent
    return row


def snapshot(events, checkpoint=None):
    return SourceSnapshot(SourceState('store', 'test:key', 'incarnation', 1, 0,
        None, 'fingerprint'), {}, events, checkpoint)


def capture(events, *, nodes=None, eligible=None, checkpoint=None, **kwargs):
    if nodes is None:
        nodes = source_nodes(events, stream_id=kwargs.get('stream_id'))
    return SourceCompactionCandidate.capture(snapshot(events, checkpoint), SourceTranscript(tuple(nodes)),
        eligible_ids=frozenset(eligible if eligible is not None else (e['_event_id'] for e in events)), **kwargs)


def test_rebuild_retains_runtime_and_expanded_pin_but_summarizes_canonical_evidence():
    events = [event('user', 'old'), event('assistant', 'facts'), event('user', 'current'),
              event('assistant', 'progress'), event('assistant', 'pending')]
    nodes = list(source_nodes(events))
    nodes[2] = SourceNode(nodes[2].event_id, {'role': 'user', 'content': 'current + authorized context'})
    policy = RuntimeNode('policy', {'role': 'system', 'content': 'policy'})
    note = RuntimeNode('note', {'role': 'user', 'content': 'continue'})
    nodes = [policy, *nodes[:4], note, nodes[4]]
    original = deepcopy(nodes)
    candidate = capture(events, nodes=nodes, eligible=[e['_event_id'] for e in events[:4]])
    assert candidate.boundaries == (1, 2, 3, 4)
    assert [n.message['content'] for n in candidate.evidence(4)] == ['old', 'facts', 'current', 'progress']
    rebuilt = candidate.rebuild(4, CheckpointNode('next', 'summary'))
    assert rebuilt.nodes == (policy, CheckpointNode('next', 'summary'), nodes[3], note, nodes[-1])
    assert nodes == original
    events[0]['content'] = 'changed after capture'
    assert candidate.evidence(4)[0].message['content'] == 'old'


def test_repeated_compaction_excludes_covered_pin_from_new_evidence():
    events = [event('user', 'objective'), event('assistant', 'old'),
              event('assistant', 'new'), event('assistant', 'newer')]
    old = CheckpointNode('old', 'prior facts')
    checkpoint = {'checkpoint_id': old.checkpoint_id, 'summary': old.summary,
                  'boundary': 2, 'projection_scope': 'root', 'projection_version': 1}
    nodes = (old, source_nodes(events)[0], *source_nodes(events)[2:])
    candidate = capture(events, nodes=nodes, checkpoint=checkpoint)
    assert candidate.previous_checkpoint == old
    assert [n.message['content'] for n in candidate.evidence(3)] == ['new']
    assert [n.message['content'] for n in candidate.evidence(4, after=3)] == ['newer']
    assert candidate.rebuild(4, CheckpointNode('new', 'folded')).nodes == (
        CheckpointNode('new', 'folded'), nodes[1])


def test_global_closure_keeps_child_evidence_out_of_summary():
    call = {'id': 'call', 'type': 'function', 'function': {'name': 'delegate', 'arguments': '{}'}}
    events = [event('user', 'root'),
              event('assistant', '', tool_calls=[call], _execution_ids=['execution']),
              event('user', 'child assignment', stream='child', parent='root'),
              event('assistant', 'child secret', stream='child', parent='root'),
              event('tool', 'returned', tool_call_id='call', _execution_id='execution')]
    root = capture(events)
    assert root.boundaries == (1, 5)
    assert 'child secret' not in str(root.evidence(5))
    with pytest.raises(SourceValidationError, match='closed boundary'):
        root.evidence(4)
    child = capture(events, stream_id='child')
    assert child.boundaries == (1, 2)
    assert [n.message['content'] for n in child.evidence(2)] == ['child assignment', 'child secret']
    assert child.rebuild(2, CheckpointNode('local', 'child summary')).nodes[-1].message['content'] == 'child assignment'


def test_unacknowledged_identical_event_is_a_barrier():
    events = [event('user', 'same'), event('assistant', 'same'), event('assistant', 'same')]
    candidate = capture(events, eligible=[events[0]['_event_id'], events[2]['_event_id']])
    assert candidate.boundaries == (1,)
    with pytest.raises(SourceValidationError):
        candidate.rebuild(3, CheckpointNode('bad', 'facts'))


@pytest.mark.parametrize('kind', ['foreign', 'missing', 'checkpoint', 'duplicate'])
def test_capture_rejects_projection_identity_drift(kind):
    events = [event('user', 'input'), event('assistant', 'result')]
    nodes = list(source_nodes(events))
    if kind == 'foreign':
        nodes[0] = SourceNode('foreign', nodes[0].message)
    elif kind == 'missing':
        nodes.pop()
    elif kind == 'checkpoint':
        nodes.insert(0, CheckpointNode('foreign', 'summary'))
    else:
        nodes.append(nodes[0])
    with pytest.raises(SourceValidationError):
        capture(events, nodes=nodes)


def test_local_checkpoint_ignores_root_checkpoint_and_rejects_cross_scope_nodes():
    events = [event('user', 'root secret'), event('user', 'assignment', stream='child', parent='root'),
              event('assistant', 'old child', stream='child', parent='root'),
              event('assistant', 'new child', stream='child', parent='root')]
    child = source_nodes(events, stream_id='child')
    old = CheckpointNode('local', 'child facts')
    candidate = capture(events, nodes=(old, child[0], child[2]), stream_id='child',
        local_boundary=2, local_checkpoint=old,
        checkpoint={'projection_scope': 'root', 'projection_version': 1,
                    'checkpoint_id': 'root', 'summary': 'root secret', 'boundary': 1})
    assert candidate.boundaries == (3,)
    assert candidate.evidence(3) == (child[2],)
    with pytest.raises(SourceValidationError, match='foreign'):
        capture(events, nodes=(*child, source_nodes(events)[0]), stream_id='child')


def test_hidden_only_boundary_cannot_publish_empty_summary_evidence():
    events = [event('user', 'hidden', stream='child', parent='root'), event('user', 'input')]
    candidate = capture(events, eligible=[])
    assert candidate.boundaries == (1,)
    assert candidate.evidence(1) == ()
    with pytest.raises(SourceValidationError, match='new visible evidence'):
        candidate.rebuild(1, CheckpointNode('empty', 'invented'))


def test_reordered_source_nodes_cannot_replace_newest_instruction_with_old_pin():
    events = [event('user', 'old'), event('user', 'newest'), event('assistant', 'progress')]
    nodes = source_nodes(events)
    with pytest.raises(SourceValidationError, match='reorders'):
        capture(events, nodes=(nodes[1], nodes[0], nodes[2]))


def test_rendered_role_cannot_change_canonical_pin_selection():
    events = [event('user', 'old'), event('user', 'newest'), event('assistant', 'progress')]
    nodes = list(source_nodes(events))
    nodes[1] = SourceNode(nodes[1].event_id, {'role': 'assistant', 'content': 'authorized expansion'})
    nodes[2] = SourceNode(nodes[2].event_id, {'role': 'user', 'content': 'rendered user'})
    candidate = capture(events, nodes=nodes)
    assert candidate.pinned_event_id == events[1]['_event_id']
    assert candidate.rebuild(3, CheckpointNode('next', 'facts')).nodes == (
        CheckpointNode('next', 'facts'), nodes[1])


def test_covered_non_pin_cannot_reappear_beside_checkpoint():
    events = [event('user', 'old'), event('assistant', 'old answer'), event('user', 'new')]
    old = CheckpointNode('old', 'facts')
    with pytest.raises(SourceValidationError, match='canonical source'):
        capture(events, nodes=(old, *source_nodes(events)), checkpoint={
            'checkpoint_id': 'old', 'summary': 'facts', 'boundary': 2,
            'projection_scope': 'root', 'projection_version': 1})
