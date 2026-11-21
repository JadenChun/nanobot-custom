"""Raw canonical evidence, atomic append/CAS and cross-process owner controls."""
from __future__ import annotations

import asyncio
import json
import multiprocessing
from dataclasses import replace
from pathlib import Path

import pytest

from nanobot.runtime_context import public_history_message
from nanobot.session.execution_owner import ExecutionBusyError, acquire_execution
from nanobot.session.input_attempt import (
    InputAttempt,
    InputAttemptConflictError,
    InputAttemptSource,
    find_input_attempt,
    fingerprint,
    principal_tag,
)
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.source import new_source_event
from nanobot.session.turn_writer import SourceTurnWriter
from tests.session.test_source_store import candidate


@pytest.fixture
def seeded(tmp_path):
    backend = JsonlSessionStore(tmp_path / 'workspace', sessions_root=tmp_path / 'private')
    session = Session('websocket:attempt-chat')
    backend.save(session)
    snapshot = backend.source.upgrade_source_session(session.key, expected_revision=session.revision,
        expected_generation=session.generation, expected_incarnation=session.incarnation)
    attempt = InputAttempt(InputAttemptSource.from_state(snapshot.state), 'attempt-1',
                           fingerprint('exact submitted text'), principal_tag('local'))
    return backend, attempt


def test_frozen_cross_language_vectors_and_surrogates():
    path = Path(__file__).resolve().parents[2] / 'webui/src/tests/fixtures/input-attempt-vectors.json'
    for vector in json.loads(path.read_text()):
        assert vector['text'].encode().hex() == vector['utf8_hex']
        assert fingerprint(vector['text']) == vector['fingerprint']
    for text in ('\ud800', '\udc00', 'x\ud800y', 'x\udc00y'):
        with pytest.raises(InputAttemptConflictError):
            fingerprint(text)


async def test_atomic_commit_join_conflict_and_distinct_ids(seeded):
    backend, attempt = seeded
    key = attempt.source.session_key
    async with acquire_execution(backend, key, run_id='original-run') as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id='original-run')
        committed = writer.admit_input({'role': 'user', 'content': 'exact submitted text',
                                       '_input_attempt': {'forged': True}}, attempt)
        first = owner.read_snapshot()
        assert committed.disposition == 'committed'
        assert first.events[0]['_input_attempt'] == attempt.evidence()
        again = writer.admit_input({'role': 'user', 'content': 'exact submitted text'}, attempt)
        assert again.disposition == 'joined' and again.proof == committed.proof
        assert owner.read_snapshot() == first
        for changed in (replace(attempt, principal=principal_tag('mobile:other')),
                        replace(attempt, fingerprint=fingerprint('changed'))):
            with pytest.raises(InputAttemptConflictError):
                writer.admit_input({'role': 'user', 'content': 'changed'}, changed)
            assert owner.read_snapshot() == first
        # An exact event-ID retry remains compatible; a newly minted event is not a join.
        assert backend.source.append_source_events(key, first.state, [first.events[0]], owner=owner) == first
        forged = new_source_event({'role': 'user', 'content': 'exact submitted text'}, run_id='other-run')
        forged['_input_attempt'] = attempt.evidence()
        with pytest.raises(InputAttemptConflictError):
            backend.source.append_source_events(key, first.state, [forged], owner=owner)
        second = writer.admit_input({'role': 'user', 'content': 'exact submitted text'},
                                    replace(attempt, turn_id='attempt-2'))
        assert second.disposition == 'committed' and second.proof.event_id != committed.proof.event_id
        assert len(owner.read_snapshot().events) == 2


@pytest.mark.parametrize('conflict', ['same', 'payload', 'principal'])
async def test_atomic_append_rejects_duplicate_attempts_within_one_batch(seeded, conflict):
    backend, attempt = seeded
    key = attempt.source.session_key
    other = attempt
    content = 'exact submitted text'
    if conflict == 'payload':
        content = 'different submitted text'
        other = replace(attempt, fingerprint=fingerprint(content))
    elif conflict == 'principal':
        other = replace(attempt, principal=principal_tag('mobile:other'))
    first = new_source_event({'role': 'user', 'content': 'exact submitted text'}, run_id='first-run')
    first['_input_attempt'] = attempt.evidence()
    second = new_source_event({'role': 'user', 'content': content}, run_id='second-run')
    second['_input_attempt'] = other.evidence()
    async with acquire_execution(backend, key, run_id='batch-run') as owner:
        before = owner.read_snapshot()
        with pytest.raises(InputAttemptConflictError):
            backend.source.append_source_events(key, before.state, [first, second], owner=owner)
        assert owner.read_snapshot() == before
        # Repeating the identical event is still the source event-ID idempotent retry.
        saved = backend.source.append_source_events(key, before.state, [first], owner=owner)
        assert saved.events == [first]
        assert backend.source.append_source_events(key, saved.state, [first], owner=owner) == saved
        proof = find_input_attempt(saved.events, attempt)
        assert proof is not None and proof.event_id == first['_event_id']


async def test_raw_checkpoint_revision_advance_retains_proof_without_backfill(seeded):
    backend, attempt = seeded
    key = attempt.source.session_key
    async with acquire_execution(backend, key, run_id='original-run') as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id='original-run')
        admitted = writer.admit_input({'role': 'user', 'content': 'exact submitted text'}, attempt)
        original = owner.read_snapshot()
        checkpoint = backend.source.commit_context_checkpoint(key, original.state,
            candidate(original), owner=owner)
        assert checkpoint.events == original.events and checkpoint.state.same_origin(original.state)
        assert checkpoint.state.revision > original.state.revision
        assert find_input_attempt(checkpoint.events, attempt) == admitted.proof
    async with acquire_execution(backend, key, run_id='later-run') as owner:
        joined = SourceTurnWriter(backend.source, owner, run_id='later-run').admit_input(
            {'role': 'user', 'content': 'exact submitted text'}, attempt)
        assert joined.disposition == 'joined' and joined.proof == admitted.proof
        assert owner.read_snapshot().events == original.events


def test_untrusted_reserved_evidence_is_stripped_from_new_legacy_and_public_paths(seeded):
    backend, attempt = seeded
    raw = {'role': 'user', 'content': 'text', '_input_attempt': attempt.evidence()}
    assert '_input_attempt' not in new_source_event(raw, run_id='trusted')
    assert '_input_attempt' not in public_history_message(raw)
    legacy = Session('websocket:legacy', messages=[raw])
    backend.save(legacy)
    upgraded = backend.source.upgrade_source_session(legacy.key, expected_revision=legacy.revision,
        expected_generation=legacy.generation, expected_incarnation=legacy.incarnation)
    assert '_input_attempt' not in upgraded.events[0]
    assert raw['_input_attempt'] == attempt.evidence()


async def test_reset_does_not_authenticate_copied_evidence(seeded):
    backend, attempt = seeded
    key = attempt.source.session_key
    async with acquire_execution(backend, key, run_id='original-run') as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id='original-run')
        writer.admit_input({'role': 'user', 'content': 'exact submitted text'}, attempt)
        snapshot = owner.read_snapshot()
        await owner.quiesce()
        reset = backend.source.reset_source_session(key, snapshot.state, snapshot.events, owner=owner)
        assert reset.state.generation == snapshot.state.generation + 1
        assert '_input_attempt' not in reset.events[0]
        assert find_input_attempt(reset.events, attempt) is None


def _second_process(workspace, private, key, query, principal, outcomes, proceed):
    """No listener/provider; prove the real OS lock and non-owning raw read."""
    backend = JsonlSessionStore(Path(workspace), sessions_root=Path(private))
    attempt = InputAttempt.parse_query(query, principal=principal, session_key=key)

    async def run():
        proof = find_input_attempt(backend.source.load_source_snapshot(key).events, attempt)
        try:
            async with acquire_execution(backend, key, run_id='second-process', wait_for_lock=False):
                raise AssertionError('parent OS lock was not retained')
        except ExecutionBusyError:
            outcomes.put(('locked-read', proof.to_dict()))
        assert await asyncio.to_thread(proceed.wait, 20)
        async with acquire_execution(backend, key, run_id='second-process', wait_for_lock=False) as owner:
            joined = SourceTurnWriter(backend.source, owner, run_id='second-process').admit_input(
                {'role': 'user', 'content': 'exact submitted text'}, attempt)
            outcomes.put((joined.disposition, joined.proof.to_dict(), owner.read_snapshot().events))
    asyncio.run(run())


async def test_isolated_second_process_lock_read_and_original_join(seeded):
    backend, attempt = seeded
    context = multiprocessing.get_context('spawn')
    outcomes = context.Queue()
    proceed = context.Event()
    process = context.Process(target=_second_process, args=(
        str(backend.workspace), str(backend.sessions_root), attempt.source.session_key,
        attempt.query(), attempt.principal, outcomes, proceed,
    ))
    try:
        async with acquire_execution(backend, attempt.source.session_key, run_id='original-run') as owner:
            committed = SourceTurnWriter(backend.source, owner, run_id='original-run').admit_input(
                {'role': 'user', 'content': 'exact submitted text'}, attempt)
            events = owner.read_snapshot().events
            process.start()
            observed = await asyncio.to_thread(outcomes.get, True, 20)
            assert observed == ('locked-read', committed.proof.to_dict())
        proceed.set()
        joined = await asyncio.to_thread(outcomes.get, True, 20)
        assert joined == ('joined', committed.proof.to_dict(), events)
        await asyncio.to_thread(process.join, 20)
        assert process.exitcode == 0
    finally:
        proceed.set()
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(20)
        outcomes.close()


@pytest.mark.parametrize('evidence', [{'version': 2}, {'invalid': True}])
async def test_malformed_additive_evidence_is_readable_but_blocks_fresh_admission(seeded, evidence):
    backend, attempt = seeded
    key = attempt.source.session_key
    snapshot = backend.source.load_source_snapshot(key)
    raw = new_source_event({'role': 'user', 'content': 'ordinary readable history'}, run_id='old-run')
    raw['_input_attempt'] = evidence
    backend.source._write(snapshot.header, [raw])
    assert backend.source.load_source_snapshot(key).events[0]['content'] == 'ordinary readable history'
    async with acquire_execution(backend, key, run_id='new-run') as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id='new-run')
        with pytest.raises(InputAttemptConflictError):
            writer.admit_input({'role': 'user', 'content': 'exact submitted text'}, attempt)
        assert len(owner.read_snapshot().events) == 1
