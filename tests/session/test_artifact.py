"""Artifact metadata cannot reinterpret source identity, text selectors or paths."""

import hashlib
from copy import deepcopy
from dataclasses import replace

import pytest

from nanobot.session.artifact import (
    ArtifactValidationError,
    bound_artifact_text,
    parse_artifact_descriptor,
    parse_artifact_reference,
)
from nanobot.session.source import new_source_event, prefix_digest
from nanobot.session.source_store import SourceSnapshot, SourceState

WORKSPACE = 'a' * 32
ARTIFACT = 'b' * 32


def fixture(content='evidence', *, block=None):
    call = {'id': 'same', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}
    intent = new_source_event({'role': 'assistant', 'content': '', 'tool_calls': [call, deepcopy(call)]}, run_id='run')
    intent.update(_execution_ids=['execution-one', 'execution-two'], _stream_id='stream')
    events = [intent]
    for execution in ('execution-one', 'execution-two'):
        event = new_source_event({'role': 'tool', 'name': 'read_file', 'tool_call_id': 'same',
                                  'content': content}, run_id='run')
        event.update(_execution_id=execution, _stream_id='stream')
        events.append(event)
    text = content if block is None else content[block]['text']
    encoded = text.encode('utf-8')
    raw = {
        'schema_version': 1, 'artifact_id': ARTIFACT, 'workspace_id': WORKSPACE,
        'session_key': 'test:artifact', 'incarnation': 'incarnation', 'generation': 2,
        'event_id': events[1]['_event_id'], 'execution_id': 'execution-one', 'stream_id': 'stream',
        'run_id': 'run', 'block_index': block, 'mime_type': 'text/plain; charset=utf-8',
        'byte_length': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest(),
        'created_at': '2026-09-10T00:00:00+00:00',
        'source_access': {'kind': 'filesystem', 'access_class': 'workspace',
                          'repository': None, 'relative_path': 'notes/evidence.txt'},
    }
    snapshot = SourceSnapshot(SourceState('store', 'test:artifact', 'incarnation', 7, 2, None, 'fingerprint'),
                              {}, events, None)
    return raw, snapshot


def test_descriptor_roundtrip_binds_unicode_bytes_and_keeps_source_digest():
    raw, source = fixture('evidence 中文 🌟')
    before = deepcopy(source)
    digest = prefix_digest(source.events, len(source.events))
    descriptor = parse_artifact_descriptor(raw)
    assert descriptor.to_dict() == raw
    assert descriptor.reference == 'artifact:' + ARTIFACT
    assert parse_artifact_reference(descriptor.reference) == ARTIFACT
    assert bound_artifact_text(descriptor, source, workspace_id=WORKSPACE) == 'evidence 中文 🌟'.encode()
    raw['source_access']['relative_path'] = 'mutated'
    assert descriptor.source_access.relative_path == 'notes/evidence.txt'
    assert source == before and prefix_digest(source.events, len(source.events)) == digest


def test_repeated_provider_call_ids_have_distinct_execution_and_idempotency_bindings():
    raw, source = fixture()
    first = parse_artifact_descriptor(raw)
    raw.update(artifact_id='c' * 32, event_id=source.events[2]['_event_id'], execution_id='execution-two')
    second = parse_artifact_descriptor(raw)
    assert first.idempotency_key != second.idempotency_key
    assert bound_artifact_text(first, source, workspace_id=WORKSPACE) == b'evidence'
    assert bound_artifact_text(second, source, workspace_id=WORKSPACE) == b'evidence'
    with pytest.raises(ArtifactValidationError, match='execution'):
        bound_artifact_text(replace(second, execution_id=first.execution_id), source, workspace_id=WORKSPACE)


def test_mixed_blocks_select_exact_text_and_never_inline_images_or_reasoning():
    content = [{'type': 'text', 'text': 'first'}, {'type': 'image_url', 'image_url': {'url': 'private'}},
               {'type': 'text', 'text': 'second'}, {'type': 'thinking', 'text': 'private reasoning'}]
    raw, source = fixture(content, block=2)
    descriptor = parse_artifact_descriptor(raw)
    assert bound_artifact_text(descriptor, source, workspace_id=WORKSPACE) == b'second'
    for index in (0, 1, 3, 4, None):
        with pytest.raises(ArtifactValidationError):
            bound_artifact_text(replace(descriptor, block_index=index), source, workspace_id=WORKSPACE)


@pytest.mark.parametrize('field,value', [
    ('schema_version', True), ('schema_version', 2), ('generation', True), ('generation', -1),
    ('byte_length', True), ('byte_length', -1), ('block_index', False), ('block_index', -1),
    ('block_index', '0'), ('artifact_id', '../escape'), ('artifact_id', 'A' * 32),
    ('workspace_id', 'a' * 31), ('sha256', 'not a digest'), ('mime_type', 'image/png'),
    ('created_at', '2026-09-10T00:00:00'), ('created_at', 'invalid'), ('event_id', ''),
    ('stream_id', None), ('execution_id', '\x00'), ('run_id', '  '),
])
def test_invalid_fields_fail_with_content_free_errors(field, value):
    raw, _ = fixture()
    raw[field] = value
    with pytest.raises(ArtifactValidationError):
        parse_artifact_descriptor(raw)


@pytest.mark.parametrize('field', ['extra', 'missing'])
def test_descriptor_requires_exact_schema(field):
    raw, _ = fixture()
    if field == 'extra':
        raw['path'] = '/private/forged'
    else:
        del raw['source_access']
    with pytest.raises(ArtifactValidationError, match='schema'):
        parse_artifact_descriptor(raw)


@pytest.mark.parametrize('path', ['/absolute', '../escape', 'a/../escape', 'a//b', 'a/./b',
                                   'C:/private', 'a\\b', 'a\x00b', 'a\nb', 'file:stream', ''])
def test_source_locator_rejects_noncanonical_paths(path):
    raw, _ = fixture()
    raw['source_access']['relative_path'] = path
    with pytest.raises(ArtifactValidationError, match='locator'):
        parse_artifact_descriptor(raw)


@pytest.mark.parametrize('access,repository', [('workspace', 'repo'), ('context_repository', None),
    ('target_repository', ''), ('unknown', None)])
def test_access_class_cannot_silently_fall_back_to_workspace(access, repository):
    raw, _ = fixture()
    raw['source_access'].update(access_class=access, repository=repository)
    with pytest.raises(ArtifactValidationError):
        parse_artifact_descriptor(raw)


@pytest.mark.parametrize('field,value', [('workspace_id', 'c' * 32), ('session_key', 'another'),
    ('incarnation', 'recreated'), ('generation', 3), ('event_id', 'missing'),
    ('execution_id', 'execution-two'), ('stream_id', 'sibling'), ('run_id', 'another'),
    ('byte_length', 1), ('sha256', 'd' * 64)])
def test_descriptor_cannot_rebind_a_source_event(field, value):
    raw, source = fixture()
    raw[field] = value
    with pytest.raises(ArtifactValidationError):
        bound_artifact_text(parse_artifact_descriptor(raw), source, workspace_id=WORKSPACE)


def test_matching_result_cannot_conceal_cross_stream_call_provenance():
    raw, source = fixture()
    source.events[0]['_stream_id'] = 'other-stream'
    with pytest.raises(ArtifactValidationError, match='stream provenance'):
        bound_artifact_text(parse_artifact_descriptor(raw), source, workspace_id=WORKSPACE)


@pytest.mark.parametrize('value', [ARTIFACT, 'artifact:../x', 'artifact:' + ARTIFACT + '/x',
    'artifact:' + ARTIFACT + '\n', 'https://host/' + ARTIFACT, 'artifact:' + ARTIFACT.upper(), None])
def test_reference_is_never_a_path_url_or_access_token(value):
    with pytest.raises(ArtifactValidationError):
        parse_artifact_reference(value)


@pytest.mark.parametrize('extra_ids', [None, ['execution-one']])
def test_non_call_execution_fields_cannot_spoof_intent_provenance(extra_ids):
    raw, source = fixture()
    unrelated = new_source_event({'role': 'user', 'content': 'unrelated'}, run_id='run')
    unrelated.update(_stream_id='stream', _execution_ids=extra_ids)
    source.events.insert(0, unrelated)
    descriptor = parse_artifact_descriptor(raw)
    # Harmless non-call fields neither interfere with lookup nor become intent.
    assert bound_artifact_text(descriptor, source, workspace_id=WORKSPACE) == b'evidence'
    source.events[1]['_stream_id'] = 'wrong actual call stream'
    with pytest.raises(ArtifactValidationError, match='stream provenance'):
        bound_artifact_text(descriptor, source, workspace_id=WORKSPACE)


def test_manifest_membership_preserves_exact_immutable_descriptor_bindings():
    from nanobot.session.artifact import published_artifacts

    raw, source = fixture()
    assert published_artifacts(source, workspace_id=WORKSPACE) == ()
    source.header['metadata'] = {'artifact_manifest': {'schema_version': 1,
        'descriptors': {ARTIFACT: raw}}}
    assert published_artifacts(source, workspace_id=WORKSPACE) == (parse_artifact_descriptor(raw),)
    raw['sha256'] = '0' * 64
    with pytest.raises(ArtifactValidationError, match='integrity'):
        published_artifacts(source, workspace_id=WORKSPACE)


@pytest.mark.parametrize('failure', ['null', 'version', 'fields', 'wrong_id', 'duplicate_binding'])
def test_invalid_manifest_is_not_treated_as_empty(failure):
    from nanobot.session.artifact import published_artifacts

    raw, source = fixture()
    manifest = {'schema_version': 1, 'descriptors': {ARTIFACT: raw}}
    if failure == 'null':
        manifest = None
    elif failure == 'version':
        manifest['schema_version'] = True
    elif failure == 'fields':
        manifest['grant_all'] = True
    elif failure == 'wrong_id':
        manifest['descriptors']['c' * 32] = manifest['descriptors'].pop(ARTIFACT)
    else:
        duplicate = {**raw, 'artifact_id': 'c' * 32}
        manifest['descriptors']['c' * 32] = duplicate
    source.header['metadata'] = {'artifact_manifest': manifest}
    before = deepcopy(source)
    with pytest.raises(ArtifactValidationError):
        published_artifacts(source, workspace_id=WORKSPACE)
    assert source == before


def test_attestation_requires_receipt_and_cannot_relabel_origin():
    from nanobot.agent.tools.read_provenance import FilesystemReadReceipt
    from nanobot.session.artifact import attested_artifact_origin

    raw, source = fixture()
    descriptor = parse_artifact_descriptor(raw)
    with pytest.raises(ArtifactValidationError):
        attested_artifact_origin(descriptor, source, workspace_id=WORKSPACE)
    receipt = FilesystemReadReceipt(descriptor.source_access, 'a' * 64,
                                    descriptor.byte_length, descriptor.sha256)
    source.events[1].update(_source_origin=receipt.to_dict(), _dispatched=True, _outcome_status='ok')
    assert attested_artifact_origin(descriptor, source, workspace_id=WORKSPACE) == receipt
    changed = replace(descriptor, source_access=replace(descriptor.source_access,
        access_class='context_repository', repository='forged'))
    with pytest.raises(ArtifactValidationError):
        attested_artifact_origin(changed, source, workspace_id=WORKSPACE)
    source.events[1]['_source_origin']['sha256'] = '0' * 64
    with pytest.raises(ArtifactValidationError):
        attested_artifact_origin(descriptor, source, workspace_id=WORKSPACE)
