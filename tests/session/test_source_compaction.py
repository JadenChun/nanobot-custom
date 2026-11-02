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
    if checkpoint is not None and kwargs.get('stream_id') is None:
        kwargs.setdefault('accepted_boundary', checkpoint['boundary'])
        kwargs.setdefault('accepted_checkpoint', CheckpointNode(checkpoint['checkpoint_id'], checkpoint['summary']))
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
    summary_evidence = candidate.summary_evidence(4)
    assert [node.message['content'] for node in summary_evidence.nodes] == [
        'old', 'facts', 'current', 'progress',
    ]
    assert summary_evidence.projections == ()
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


def test_selects_historical_parallel_tool_batches_before_expanded_current_user():
    first_batch = [
        {'id': 'image-a', 'type': 'function',
         'function': {'name': 'read_image', 'arguments': '{"path":"a.png"}'}},
        {'id': 'image-b', 'type': 'function',
         'function': {'name': 'read_image', 'arguments': '{"path":"b.png"}'}},
    ]
    second_batch = [
        {'id': 'image-c', 'type': 'function',
         'function': {'name': 'read_image', 'arguments': '{"path":"c.png"}'}},
        {'id': 'image-d', 'type': 'function',
         'function': {'name': 'read_image', 'arguments': '{"path":"d.png"}'}},
    ]
    events = [
        event('user', 'Compare these images.'),
        event('assistant', '', tool_calls=first_batch,
              _execution_ids=['image-exec-a', 'image-exec-b'],
              reasoning_content='private reasoning from the first image batch'),
        event('tool', 'Image A contains a blue car.', tool_call_id='image-a',
              name='read_image', _execution_id='image-exec-a'),
        event('tool', 'Image B contains a red car.', tool_call_id='image-b',
              name='read_image', _execution_id='image-exec-b'),
        event('assistant', '', tool_calls=second_batch,
              _execution_ids=['image-exec-c', 'image-exec-d'],
              reasoning_content='private reasoning from the second image batch'),
        event('tool', 'Image C is a street scene.', tool_call_id='image-c',
              name='read_image', _execution_id='image-exec-c'),
        event('tool', 'Image D is a parking lot.', tool_call_id='image-d',
              name='read_image', _execution_id='image-exec-d'),
        event('assistant', 'The images show different settings.',
              reasoning_content='private final reasoning'),
        event('user', 'What changed?', media=['/current/private.png']),
        event('assistant', 'pending latest response'),
    ]
    active = list(source_nodes(events))
    current_index = 8
    active[current_index] = SourceNode(
        active[current_index].event_id,
        {'role': 'user', 'content': 'What changed? EXPANDED_CURRENT_IMAGE_SECRET'},
    )
    system = RuntimeNode('actual-system-head', {'role': 'system', 'content': 'System policy.'})
    candidate = capture(
        events, nodes=(system, *active),
        eligible=[row['_event_id'] for row in events[:8]],
    )

    selection = candidate.select_historical_prefix(max_boundary=current_index)

    assert selection.boundary == 8
    assert selection.source_node_count == 8
    assert selection.fallback_reason is None
    assert selection.transcript is not None
    assert selection.transcript.nodes == (system, *active[:8])
    assert 'System policy.' not in repr(selection)
    assert tuple(node.event_id for node in selection.transcript.nodes if isinstance(node, SourceNode)) == tuple(
        node.event_id for node in source_nodes(events[:8])
    )
    evidence = candidate.summary_evidence(selection.boundary)
    assert tuple(node.event_id for node in evidence.nodes) == tuple(
        node.event_id for node in source_nodes(events[:8])
    )
    rendered_evidence = repr(evidence)
    assert 'EXPANDED_CURRENT_IMAGE_SECRET' not in rendered_evidence
    assert 'private reasoning' not in rendered_evidence


def test_select_historical_prefix_returns_safe_fallback_without_a_closed_group():
    events = [event('user', 'current request')]
    candidate = capture(events, eligible=[events[0]['_event_id']])

    selection = candidate.select_historical_prefix(max_boundary=1)

    assert selection.boundary is None
    assert selection.transcript is None
    assert selection.source_node_count == 0
    assert selection.fallback_reason == 'no_substantive_historical_group'


def test_select_historical_prefix_cuts_at_intermediate_runtime_node():
    events = [event('user', 'old question'), event('assistant', 'old answer'),
              event('user', 'later question'), event('assistant', 'later response')]
    source = source_nodes(events)
    system = RuntimeNode('system', {'role': 'system', 'content': 'System policy.'})
    note = RuntimeNode('request-note', {'role': 'user', 'content': 'current instruction'})
    candidate = capture(events, nodes=(system, source[0], source[1], note, source[2], source[3]))

    selection = candidate.select_historical_prefix(max_boundary=4)

    assert selection.boundary == 2
    assert selection.source_node_count == 2
    assert selection.fallback_reason is None
    assert selection.transcript is not None
    assert selection.transcript.nodes == (system, source[0], source[1])


def test_select_historical_prefix_rounds_down_before_first_representation_mismatch():
    events = [event('user', 'old question'), event('assistant', 'old answer'),
              event('assistant', 'later answer'), event('assistant', 'latest response')]
    source = list(source_nodes(events))
    source[3] = SourceNode(source[3].event_id,
                           {'role': 'assistant', 'content': 'expanded latest response'})
    candidate = capture(events, nodes=source)

    selection = candidate.select_historical_prefix(max_boundary=4)

    assert selection.boundary == 3
    assert selection.source_node_count == 3
    assert selection.fallback_reason is None
    assert selection.transcript is not None
    assert selection.transcript.nodes == tuple(source[:3])


def test_select_historical_prefix_rejects_an_expanded_covered_pin_after_checkpoint():
    events = [event('user', 'covered user'), event('assistant', 'new historical answer')]
    old = CheckpointNode('prior', 'prior summary')
    source = list(source_nodes(events))
    source[0] = SourceNode(source[0].event_id,
                           {'role': 'user', 'content': 'covered user plus raw expansion'})
    candidate = capture(events, nodes=(old, *source), checkpoint={
        'checkpoint_id': old.checkpoint_id, 'summary': old.summary, 'boundary': 1,
        'projection_scope': 'root', 'projection_version': 1,
    })

    selection = candidate.select_historical_prefix(max_boundary=2)

    assert selection.boundary is None
    assert selection.transcript is None
    assert selection.source_node_count == 0
    assert selection.fallback_reason == 'covered_pinned_user'


def test_select_historical_prefix_retains_exact_previous_checkpoint():
    events = [event('assistant', 'already summarized'), event('assistant', 'new facts'),
              event('assistant', 'more new facts')]
    old = CheckpointNode('prior', 'prior summary')
    checkpoint = {'checkpoint_id': old.checkpoint_id, 'summary': old.summary, 'boundary': 1,
                  'projection_scope': 'root', 'projection_version': 1}
    source = source_nodes(events)
    system = RuntimeNode('system', {'role': 'system', 'content': 'System policy.'})
    candidate = capture(events, nodes=(system, old, *source[1:]), checkpoint=checkpoint)

    selection = candidate.select_historical_prefix(max_boundary=3)

    assert selection.boundary == 3
    assert selection.source_node_count == 2
    assert selection.fallback_reason is None
    assert selection.transcript is not None
    assert selection.transcript.nodes == (system, old, *source[1:])


def test_capture_rejects_nonexact_previous_checkpoint_for_prefix_selection():
    events = [event('assistant', 'already summarized'), event('assistant', 'new facts')]
    actual = CheckpointNode('prior', 'prior summary')
    wrong = CheckpointNode('prior', 'edited summary')
    with pytest.raises(SourceValidationError, match='captured checkpoint'):
        capture(events, nodes=(wrong, *source_nodes(events)[1:]), checkpoint={
            'checkpoint_id': actual.checkpoint_id, 'summary': actual.summary, 'boundary': 1,
            'projection_scope': 'root', 'projection_version': 1,
        })


@pytest.mark.parametrize('max_boundary', [True, -1, 3, '2'])
def test_select_historical_prefix_reports_invalid_requested_boundary(max_boundary):
    events = [event('user', 'old'), event('assistant', 'answer')]
    candidate = capture(events)

    selection = candidate.select_historical_prefix(max_boundary=max_boundary)

    assert selection.boundary is None
    assert selection.transcript is None
    assert selection.source_node_count == 0
    assert selection.fallback_reason == 'invalid_requested_boundary'


def test_select_historical_prefix_propagates_malformed_dependency_evidence():
    call = {'id': 'artifact-read', 'type': 'function',
            'function': {'name': 'read_artifact', 'arguments': '{}'}}
    events = [
        event('assistant', '', tool_calls=[call], _execution_ids=['artifact-execution']),
        event('tool', 'malformed retrieval result', tool_call_id='artifact-read',
              name='read_artifact', _execution_id='artifact-execution',
              _artifact_read={'forged': True}),
        event('assistant', 'follow-up answer'),
    ]
    candidate = capture(events)

    with pytest.raises(SourceValidationError, match='artifact retrieval receipt'):
        candidate.select_historical_prefix(max_boundary=len(events))


async def test_select_historical_prefix_accepts_only_verified_artifact_projection(tmp_path):
    from nanobot.session.artifact import ArtifactValidationError
    from nanobot.session.artifact_projection import ArtifactProjector
    from nanobot.session.artifact_reader import ArtifactReader
    from tests.session.test_artifact_reader import setup_read

    async with setup_read(tmp_path, 'evidence' * 6_000) as (writer, tool, store, _, _):
        snapshot = writer.owner.read_snapshot()
        projector = ArtifactProjector(ArtifactReader(writer, 'root_history', store, tool))
        projected = projector.prepare(SourceTranscript(source_nodes(snapshot.events)), enabled=True)
        eligible = frozenset(node.event_id for node in source_nodes(snapshot.events))
        candidate = SourceCompactionCandidate.capture(
            snapshot, projected, eligible_ids=eligible,
        )

        selection = candidate.select_historical_prefix(max_boundary=len(snapshot.events))

        assert selection.boundary == len(snapshot.events)
        assert selection.transcript is not None
        evidence = candidate.summary_evidence(selection.boundary)
        assert len(evidence.projections) == 1
        assert evidence.nodes[-1].artifact is not None
        assert len(evidence.nodes[-1].message['content']) < 1_600
        tampered_nodes = list(projected.nodes)
        projected_tool = tampered_nodes[-1]
        assert isinstance(projected_tool, SourceNode)
        tampered_nodes[-1] = SourceNode(
            projected_tool.event_id,
            {**projected_tool.message, 'content': projected_tool.message['content'] + ' forged'},
            projected_tool.artifact,
        )
        tampered = SourceCompactionCandidate.capture(
            snapshot, SourceTranscript(tuple(tampered_nodes)), eligible_ids=eligible,
        )
        with pytest.raises(ArtifactValidationError):
            tampered.select_historical_prefix(max_boundary=len(snapshot.events))
