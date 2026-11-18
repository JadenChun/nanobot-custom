"""Authenticated attach is positive-only, noncreating canonical readback."""
from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.channels.websocket.tests.test_websocket_channel import _sent_ws_payloads
from nanobot.remote.edge import AccessIdentity
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.input_attempt import (
    InputAttempt,
    InputAttemptSource,
    fingerprint,
    principal_tag,
)
from nanobot.session.manager import JsonlSessionStore, Session
from nanobot.session.turn_writer import SourceTurnWriter
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_input_attempt_reconciliation import (
    channel_for,
    frame_for,
    isolate_projection,
)
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade
from tests.session.test_source_store import candidate

engine_config = engine_fixtures.engine_config
isolate_projection = isolate_projection


async def seeded_channel(config, *, principal=None):
    loop = AgentLoop.from_config(config, tool_registry=ToolRegistry(), provider=ScriptedProvider([]))
    backend = upgrade(loop, 'websocket:attempt-chat')
    snapshot = backend.source.load_source_snapshot('websocket:attempt-chat')
    frame = frame_for(snapshot)
    attempt = InputAttempt.parse_message(frame['input_attempt'], principal=principal or principal_tag('local'),
        session_key=snapshot.state.session_key, turn_id=frame['turn_id'], text=frame['content'])
    async with acquire_execution(backend, snapshot.state.session_key, run_id='original-run') as owner:
        admission = SourceTurnWriter(backend.source, owner, run_id='original-run').admit_input(
            {'role': 'user', 'content': frame['content']}, attempt)
    channel, conn = channel_for(loop, config)
    return loop, backend, channel, conn, frame, admission.proof


async def lookup(channel, conn, query, *, chat_id='attempt-chat', client_id='untrusted-query-label'):
    conn.send.reset_mock()
    await channel._dispatch_envelope(conn, client_id, {
        'type': 'attach', 'chat_id': chat_id, 'input_attempt_query': query,
    })
    attached = [frame for frame in _sent_ws_payloads(conn) if frame['event'] == 'attached']
    return attached[-1] if attached else None


async def test_actual_attach_is_pure_and_supplies_exact_canonical_proof(engine_config, monkeypatch):
    loop, backend, channel, conn, frame, proof = await seeded_channel(engine_config)
    source_path = backend.get_session_path('websocket:attempt-chat')
    try:
        await channel._dispatch_envelope(conn, 'spoofed-label', {'type': 'attach', 'chat_id': 'attempt-chat'})
        capability = [item for item in _sent_ws_payloads(conn) if item['event'] == 'attached'][-1]
        assert capability['input_attempt_source']['principal'] == principal_tag('local')
        query = {'version': 1, 'turn_id': frame['turn_id'], **frame['input_attempt']}
        before = (source_path.read_bytes(), source_path.stat(), deepcopy(loop.sessions.read_session_metadata('websocket:attempt-chat')))
        spies = []
        for owner, name in ((loop.sessions, 'get_or_create'), (loop.sessions, 'save'),
                            (type(backend.source), 'upgrade_source_session'), (type(backend.source), 'append_source_events'),
                            (type(backend.source), '_write'), (channel._commands._workspaces, 'persist_scope'),
                            (channel._commands, 'start_webui_request')):
            spy = Mock(side_effect=AssertionError('lookup attempted mutation'))
            monkeypatch.setattr(owner, name, spy)
            spies.append(spy)
        import nanobot.agent.loop as loop_module
        import nanobot.session.execution_owner as owner_module
        for owner in (owner_module, loop_module):
            spy = Mock(side_effect=AssertionError('lookup acquired execution'))
            monkeypatch.setattr(owner, 'acquire_execution', spy)
            spies.append(spy)
        replay = channel._commands._durable_replay
        if replay is not None:
            for name in ('lookup', 'begin', 'complete', 'uncertain'):
                spy = Mock(side_effect=AssertionError('lookup entered mutation replay'))
                monkeypatch.setattr(replay, name, spy)
                spies.append(spy)
        inbound = AsyncMock(side_effect=AssertionError('lookup published input'))
        monkeypatch.setattr(loop.bus, 'publish_inbound', inbound)
        attached = await lookup(channel, conn, query)
        assert attached['input_attempt_result'] == {
            'version': 1, 'turn_id': 'attempt-1', 'disposition': 'admitted', 'proof': proof.to_dict(),
        }
        assert source_path.read_bytes() == before[0]
        after_stat = source_path.stat()
        assert (after_stat.st_ino, after_stat.st_size, after_stat.st_mtime_ns, after_stat.st_ctime_ns) == (
            before[1].st_ino, before[1].st_size, before[1].st_mtime_ns, before[1].st_ctime_ns)
        assert loop.sessions.read_session_metadata('websocket:attempt-chat') == before[2]
        for spy in spies:
            spy.assert_not_called()
        inbound.assert_not_awaited()
        bridge = os.environ.get('NANOBOT_INPUT_ATTEMPT_PURE_BRIDGE')
        if bridge:
            Path(bridge).write_text(json.dumps({'capability': capability, 'query': {
                'type': 'attach', 'chat_id': 'attempt-chat', 'input_attempt_query': query,
            }, 'attached': attached}, indent=2) + '\n')
    finally:
        await channel.stop()
        await loop.aclose()


@pytest.mark.parametrize('case', ['missing', 'malformed', 'duplicate', 'unsupported', 'bad_event', 'bad_payload',
    'wrong_origin', 'wrong_principal', 'wrong_fingerprint', 'storage_error', 'reset', 'fork', 'store_replacement'])
async def test_lookup_miss_or_invalid_evidence_is_only_unknown(engine_config, monkeypatch, case):
    loop, backend, channel, conn, frame, proof = await seeded_channel(engine_config)
    query = {'version': 1, 'turn_id': frame['turn_id'], **frame['input_attempt']}
    key = 'websocket:attempt-chat'
    try:
        snapshot = backend.source.load_source_snapshot(key)
        events = deepcopy(snapshot.events)
        chat_id = 'attempt-chat'
        if case == 'missing':
            backend.get_session_path(key).unlink()
        elif case == 'storage_error':
            monkeypatch.setattr(loop.sessions, 'read_existing_canonical_source', Mock(side_effect=OSError('disk')))
        elif case == 'wrong_origin':
            query['source'] = {**query['source'], 'generation': 999}
        elif case == 'wrong_principal':
            channel.gateway.endpoint.webui_connections.discard(conn)
            conn.mobile_identity = AccessIdentity('different-installation', ('chat.read',), time.time() + 60, 1)
            monkeypatch.setattr(channel, 'revalidate_mobile_connection', AsyncMock(return_value=True))
        elif case == 'wrong_fingerprint':
            query['fingerprint'] = fingerprint('not the original')
        elif case == 'reset':
            async with acquire_execution(backend, key, run_id='reset') as owner:
                current = owner.read_snapshot()
                await owner.quiesce()
                backend.source.reset_source_session(key, current.state, [], owner=owner)
        elif case == 'store_replacement':
            replacement = JsonlSessionStore(engine_config.workspace_path,
                sessions_root=backend.sessions_root.parent / 'replacement-sessions')
            replacement.get_session_path(key).write_bytes(backend.get_session_path(key).read_bytes())
            monkeypatch.setattr(loop.sessions, '_jsonl_store', replacement)
        elif case == 'fork':
            fork = Session('websocket:fork-chat', messages=events)
            backend.save(fork)
            forked = backend.source.upgrade_source_session(fork.key, expected_revision=fork.revision,
                expected_generation=fork.generation, expected_incarnation=fork.incarnation)
            assert all('_input_attempt' not in event for event in forked.events)
            chat_id = 'fork-chat'
            query['source'] = InputAttemptSource.from_state(forked.state).to_dict()
            query['source_token'] = InputAttemptSource.from_state(forked.state).token
        else:
            if case == 'malformed':
                events[0]['_input_attempt'] = {'invalid': True}
            elif case == 'unsupported':
                events[0]['_input_attempt']['version'] = 2
            elif case == 'bad_event':
                events[0].pop('_run_id')
            elif case == 'bad_payload':
                events[0]['content'] = 'corrupted payload'
            elif case == 'duplicate':
                events.append({**events[0], '_event_id': 'another-event'})
            backend.source._write(snapshot.header, events)
        attached = await lookup(channel, conn, query, chat_id=chat_id)
        assert attached['input_attempt_result'] == {'version': 1, 'turn_id': 'attempt-1', 'disposition': 'unknown'}
        assert loop.bus.inbound_size == 0
    finally:
        await channel.stop()
        await loop.aclose()


async def test_covering_checkpoint_is_not_the_query_authority(engine_config):
    loop, backend, channel, conn, frame, proof = await seeded_channel(engine_config)
    try:
        key = 'websocket:attempt-chat'
        snapshot = backend.source.load_source_snapshot(key)
        checkpoint = backend.source.commit_context_checkpoint(key, snapshot.state, candidate(snapshot))
        assert checkpoint.state.revision > snapshot.state.revision
        attached = await lookup(channel, conn, {'version': 1, 'turn_id': frame['turn_id'], **frame['input_attempt']})
        assert attached['input_attempt_result']['proof'] == proof.to_dict()
        assert checkpoint.events == snapshot.events
    finally:
        await channel.stop()
        await loop.aclose()


@pytest.mark.parametrize('stage', ['before_read', 'read_denied', 'before_delivery'])
async def test_mobile_live_read_authorization_is_revalidated(engine_config, monkeypatch, stage):
    loop, backend, channel, conn, frame, proof = await seeded_channel(
        engine_config, principal=principal_tag('mobile:installation-one'))
    channel.gateway.endpoint.webui_connections.discard(conn)
    conn.mobile_identity = AccessIdentity('installation-one', ('chat.read',), time.time() + 60, 1)
    read = Mock(wraps=loop.sessions.read_existing_canonical_source)
    monkeypatch.setattr(loop.sessions, 'read_existing_canonical_source', read)
    checks = AsyncMock(return_value=True)
    if stage == 'before_read':
        checks.return_value = False
    if stage == 'read_denied':
        conn.mobile_identity = replace(conn.mobile_identity, scopes=('chat.write',))
    if stage == 'before_delivery':
        checks.side_effect = [True, False]
    monkeypatch.setattr(channel, 'revalidate_mobile_connection', checks)
    try:
        attached = await lookup(channel, conn, proof.attempt.query())
        assert attached is None
        if stage != 'before_delivery':
            read.assert_not_called()
        else:
            assert read.call_count == 1
    finally:
        await channel.stop()
        await loop.aclose()


async def test_generic_socket_and_spoofed_principal_cannot_obtain_capability(engine_config, monkeypatch):
    loop, backend, channel, conn, frame, proof = await seeded_channel(engine_config)
    channel.gateway.endpoint.webui_connections.discard(conn)
    conn.mobile_identity = None
    read = Mock(wraps=loop.sessions.read_existing_canonical_source)
    monkeypatch.setattr(loop.sessions, 'read_existing_canonical_source', read)
    try:
        attached = await lookup(channel, conn, proof.attempt.query(), client_id='installation-one')
        assert attached is not None and 'input_attempt_source' not in attached and 'input_attempt_result' not in attached
        read.assert_not_called()
        raw = {**proof.attempt.query(), 'principal': proof.attempt.principal}
        attached = await lookup(channel, conn, raw)
        assert attached is not None and 'input_attempt_result' not in attached
        read.assert_not_called()
    finally:
        await channel.stop()
        await loop.aclose()


@pytest.mark.parametrize('change', ['principal', 'scopes', 'subscription'])
async def test_identity_and_subscription_changes_during_read_cannot_receive_proof(engine_config, monkeypatch, change):
    loop, backend, channel, conn, frame, proof = await seeded_channel(
        engine_config, principal=principal_tag('mobile:installation-one'))
    channel.gateway.endpoint.webui_connections.discard(conn)
    conn.mobile_identity = AccessIdentity('installation-one', ('chat.read',), time.time() + 60, 1)
    monkeypatch.setattr(channel, 'revalidate_mobile_connection', AsyncMock(return_value=True))
    original_read = loop.sessions.read_existing_canonical_source
    def read_and_replace(*args, **kwargs):
        result = original_read(*args, **kwargs)
        if change == 'principal':
            conn.mobile_identity = replace(conn.mobile_identity, installation_id='replacement-installation')
        elif change == 'scopes':
            conn.mobile_identity = replace(conn.mobile_identity, scopes=('chat.write',))
        else:
            channel.webui_detach(conn, 'attempt-chat')
        return result
    monkeypatch.setattr(loop.sessions, 'read_existing_canonical_source', read_and_replace)
    try:
        assert await lookup(channel, conn, proof.attempt.query()) is None
    finally:
        await channel.stop()
        await loop.aclose()


@pytest.mark.parametrize('case', ['missing', 'legacy', 'transient', 'temporary'])
async def test_advertisement_never_creates_or_upgrades_an_ineligible_source(engine_config, case):
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=ScriptedProvider([]))
    channel, conn = channel_for(loop, engine_config)
    key = 'websocket:ineligible'
    try:
        if case == 'temporary':
            await channel._dispatch_envelope(conn, 'local', {'type': 'new_temporary_chat'})
            attached = [item for item in _sent_ws_payloads(conn) if item['event'] == 'attached'][-1]
            chat_id = attached['chat_id']
        else:
            chat_id = 'ineligible'
            if case == 'legacy':
                loop.sessions.save(loop.sessions.get_or_create(key))
            elif case == 'transient':
                loop.sessions.get_or_create_transient(key, retain=True)
            source = loop.sessions._jsonl_store.get_session_path(key)
            before = source.read_bytes() if source.exists() else None
            await channel._dispatch_envelope(conn, 'local', {'type': 'attach', 'chat_id': chat_id})
            replies = _sent_ws_payloads(conn)
            attachments = [item for item in replies if item['event'] == 'attached']
            if attachments:
                attached = attachments[-1]
            else:
                # Retained unowned transient sessions already refuse ordinary attach.
                assert case == 'transient' and any(item['event'] == 'error' for item in replies)
                attached = replies[-1]
            assert (source.read_bytes() if source.exists() else None) == before
        assert 'input_attempt_source' not in attached and 'input_attempt_result' not in attached
    finally:
        await channel.stop()
        await loop.aclose()


async def test_local_raw_principal_and_unknown_query_keys_are_never_proof(engine_config, monkeypatch):
    loop, backend, channel, conn, frame, proof = await seeded_channel(engine_config)
    read = Mock(wraps=loop.sessions.read_existing_canonical_source)
    monkeypatch.setattr(loop.sessions, 'read_existing_canonical_source', read)
    try:
        for extra in ({'principal': proof.attempt.principal}, {'extra': True}, {'version': 2}):
            attached = await lookup(channel, conn, {**proof.attempt.query(), **extra})
            assert attached['input_attempt_result']['disposition'] == 'unknown'
        read.assert_not_called()
    finally:
        await channel.stop()
        await loop.aclose()
