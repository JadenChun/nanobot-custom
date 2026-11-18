"""Negotiated canonical input admission, using the real channel/bus/loop."""
from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.channels.websocket.runtime import WebSocketChannel
from nanobot.channels.websocket.tests.test_websocket_channel import (
    _authenticate_local_webui_connection,
    _basic_handler,
    _sent_ws_payloads,
)
from nanobot.providers.base import LLMResponse
from nanobot.security.desktop_permissions import permission_source_token
from nanobot.session import webui_turns as wth
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade
from tests.webui.test_message_attempt_binding import _message_envelope

engine_config = engine_fixtures.engine_config


@pytest.fixture(autouse=True)
def isolate_projection(tmp_path, monkeypatch):
    monkeypatch.setattr('nanobot.config.paths.get_data_dir', lambda: tmp_path)
    names = ('_WEBSOCKET_ACTIVE_TURNS', '_WEBSOCKET_TURN_WALL_STARTED_AT',
             '_WEBSOCKET_TURN_IDS', '_WEBSOCKET_TURN_OWNERS', '_WEBSOCKET_TURN_STOPPING')
    for name in names:
        getattr(wth, name).clear()
    yield
    for name in names:
        getattr(wth, name).clear()


def channel_for(loop, config):
    channel = WebSocketChannel(
        {'enabled': True, 'allowFrom': ['*'], 'host': '127.0.0.1'}, loop.bus,
        gateway=_basic_handler(loop.bus, session_manager=loop.sessions,
                               workspace_path=config.workspace_path,
                               default_restrict_to_workspace=True),
    )
    connection = AsyncMock()
    _authenticate_local_webui_connection(channel, connection)
    return channel, connection


def frame_for(snapshot, *, chat_id='attempt-chat', text='exact submitted text', turn_id='attempt-1'):
    state = snapshot.state
    source = {'store_identity': state.store_identity, 'session_key': state.session_key,
              'incarnation': state.incarnation, 'generation': state.generation}
    binding = {key: source[key] for key in ('store_identity', 'incarnation', 'generation')}
    return {
        **_message_envelope(chat_id=chat_id, content=text, turn_id=turn_id),
        'input_attempt': {
            'version': 1, 'source': source, 'source_token': permission_source_token(binding),
            'fingerprint': hashlib.sha256(b'nanobot.input-attempt.v1\0new\0' + text.encode()).hexdigest(),
        },
    }


async def drain(loop, channel):
    while loop.bus.outbound_size:
        await channel.send(await loop.bus.consume_outbound())


async def dispatch_one(loop, channel):
    assert loop.bus.inbound_size == 1, 'valid first publication is required'
    await loop._dispatch(await loop.bus.consume_inbound())
    await drain(loop, channel)


async def test_negotiated_exact_duplicate_joins_original_after_completion(engine_config):
    provider = ScriptedProvider([LLMResponse(content='first'), LLMResponse(content='duplicate')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop, 'websocket:attempt-chat')
    channel, conn = channel_for(loop, engine_config)
    coordinator = wth.WebuiTurnCoordinator(bus=loop.bus, sessions=loop.sessions,
                                         schedule_background=lambda coro: coro.close())
    disconnect = coordinator.subscribe()
    frame = frame_for(backend.source.load_source_snapshot('websocket:attempt-chat'))
    try:
        await channel._dispatch_envelope(conn, 'client-label', frame)
        await dispatch_one(loop, channel)
        first = backend.source.load_source_snapshot('websocket:attempt-chat')
        assert len([e for e in first.events if e['role'] == 'user']) == 1
        assert len(provider.calls) == 1
        await channel._dispatch_envelope(conn, 'client-label', dict(frame))
        if loop.bus.inbound_size:
            await dispatch_one(loop, channel)
        second = backend.source.load_source_snapshot('websocket:attempt-chat')
        assert first.state.same_origin(second.state)
        assert len([e for e in second.events if e['role'] == 'user']) == 1, 'negotiated original attempt appended twice'
        assert len(provider.calls) == 1, 'joined admission executed another model call'
        assert second.events == first.events
        receipts = [e for e in _sent_ws_payloads(conn) if e['event'] == 'message_admitted']
        assert receipts and all(e.get('input_attempt_proof') for e in receipts)
    finally:
        disconnect()
        await channel.stop()
        await loop.aclose()


async def test_original_owner_blocks_precommit_duplicate_and_readback_crosses_receipt_and_completion(engine_config):
    import asyncio

    from nanobot.bus.runtime_events import UserInputCommitted
    from nanobot.session.input_attempt import principal_tag

    entered_receipt, release_receipt = asyncio.Event(), asyncio.Event()
    entered_model, release_model = asyncio.Event(), asyncio.Event()

    class BarrierProvider(ScriptedProvider):
        async def chat(self, *args, **kwargs):
            entered_model.set()
            await release_model.wait()
            return await super().chat(*args, **kwargs)

    provider = BarrierProvider([LLMResponse(content='one result')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop, 'websocket:attempt-chat')
    channel, conn = channel_for(loop, engine_config)
    coordinator = wth.WebuiTurnCoordinator(bus=loop.bus, sessions=loop.sessions,
                                         schedule_background=lambda coro: coro.close())
    original = coordinator._handle_user_input_committed

    async def pause_receipt(event: UserInputCommitted):
        entered_receipt.set()
        await release_receipt.wait()
        await original(event)

    coordinator._handle_user_input_committed = pause_receipt
    disconnect = coordinator.subscribe()
    frame = frame_for(backend.source.load_source_snapshot('websocket:attempt-chat'))
    dropped = []
    async def lose_receipts(raw):
        event = __import__('json').loads(raw)
        if event.get('event') in {'message_accepted', 'message_admitted'}:
            dropped.append(event)
            # Successful transport write whose reply never reaches the consumer.
    conn.send.side_effect = lose_receipts
    task = None
    try:
        await channel._dispatch_envelope(conn, 'local', {'type': 'attach', 'chat_id': 'attempt-chat'})
        capability = [e for e in _sent_ws_payloads(conn) if e['event'] == 'attached'][-1]
        await channel._dispatch_envelope(conn, 'local', frame)
        assert loop.bus.inbound_size == 1
        # Before append: the second route is unresolved, not negative proof.
        await channel._dispatch_envelope(conn, 'local', dict(frame))
        assert loop.bus.inbound_size == 1
        assert any(e.get('detail') == 'input_attempt_unknown' and e.get('reason') == 'busy_new_request'
                   for e in _sent_ws_payloads(conn))
        distinct = {**frame, 'turn_id': 'distinct-during-busy'}
        await channel._dispatch_envelope(conn, 'local', distinct)
        assert loop.bus.inbound_size == 1  # A distinct ID does not bypass busy admission.
        task = asyncio.create_task(loop._dispatch(await loop.bus.consume_inbound()))
        await entered_receipt.wait()
        assert provider.calls == []
        committed = backend.source.load_source_snapshot('websocket:attempt-chat')
        users = [e for e in committed.events if e['role'] == 'user']
        assert len(users) == 1
        assert not any(e['event'] == 'message_admitted' for e in _sent_ws_payloads(conn))
        await channel._dispatch_envelope(conn, 'local', dict(frame))
        assert loop.bus.inbound_size == 0
        query = {'version': 1, 'turn_id': frame['turn_id'], **frame['input_attempt']}
        async def readback():
            await channel._dispatch_envelope(conn, 'spoofed-label', {
                'type': 'attach', 'chat_id': 'attempt-chat', 'input_attempt_query': query,
            })
            return [e for e in _sent_ws_payloads(conn) if e['event'] == 'attached'][-1]
        before_receipt = await readback()
        proof = before_receipt['input_attempt_result']['proof']
        assert proof['event_id'] == users[0]['_event_id'] and proof['run_id'] == users[0]['_run_id']
        assert proof['principal'] == principal_tag('local')
        release_receipt.set()
        await entered_model.wait()
        # The route ACK and queued owner receipt are deliberately not delivered.
        await drain(loop, channel)
        running_attached = await readback()
        assert running_attached['input_attempt_result']['proof'] == proof
        release_model.set()
        await task
        task = None
        await drain(loop, channel)
        delivered_before_read = len(_sent_ws_payloads(conn))
        completed_attached = await readback()
        assert not any(event.get('event') == 'goal_status' and event.get('status') == 'running'
                       for event in _sent_ws_payloads(conn)[delivered_before_read:])
        assert completed_attached['input_attempt_result']['proof'] == proof
        assert {event['event'] for event in dropped} == {'message_accepted', 'message_admitted'}
        import json
        import os
        from pathlib import Path
        bridge = os.environ.get('NANOBOT_INPUT_ATTEMPT_BRIDGE')
        if bridge:
            Path(bridge).write_text(json.dumps({'capability': capability, 'query': {
                'type': 'attach', 'chat_id': 'attempt-chat', 'input_attempt_query': query,
            }, 'attached': completed_attached, 'attached_running': running_attached,
                'dropped_receipts': [event['event'] for event in dropped]}, indent=2) + '\n')
        assert len(provider.calls) == 1
        assert len([e for e in backend.source.load_source_snapshot('websocket:attempt-chat').events if e['role'] == 'user']) == 1
    finally:
        release_receipt.set()
        release_model.set()
        if task is not None:
            await task
        disconnect()
        await channel.stop()
        await loop.aclose()


async def test_recreated_router_and_loop_join_before_any_new_run_or_lifecycle(engine_config, monkeypatch):
    from unittest.mock import Mock

    from nanobot.bus.runtime_events import SessionTurnStarted, TurnRunStatusChanged
    from nanobot.session.input_attempt import fingerprint

    first_provider = ScriptedProvider([LLMResponse(content='one')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=first_provider)
    backend = upgrade(loop, 'websocket:attempt-chat')
    channel, conn = channel_for(loop, engine_config)
    coordinator = wth.WebuiTurnCoordinator(bus=loop.bus, sessions=loop.sessions,
                                         schedule_background=lambda coro: coro.close())
    disconnect = coordinator.subscribe()
    frame = frame_for(backend.source.load_source_snapshot('websocket:attempt-chat'))
    try:
        await channel._dispatch_envelope(conn, 'local', frame)
        await dispatch_one(loop, channel)
        original = backend.source.load_source_snapshot('websocket:attempt-chat')
    finally:
        disconnect()
        await channel.stop()
        await loop.aclose()
    # This is router/loop recreation, explicitly not a process-crash claim.
    provider = ScriptedProvider([LLMResponse(content='legitimate second')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    channel, conn = channel_for(loop, engine_config)
    coordinator = wth.WebuiTurnCoordinator(bus=loop.bus, sessions=loop.sessions,
                                         schedule_background=lambda coro: coro.close())
    disconnect = coordinator.subscribe()
    started, status = [], []
    loop.bus.subscribe(started.append, SessionTurnStarted)
    loop.bus.subscribe(status.append, TurnRunStatusChanged)
    try:
        with monkeypatch.context() as patch:
            save = Mock(side_effect=AssertionError('joined admission saved a new RunStore record'))
            patch.setattr(loop.run_store, 'save', save)
            await channel._dispatch_envelope(conn, 'different-label', dict(frame))
            await dispatch_one(loop, channel)
            assert not started and not status and not provider.calls
            save.assert_not_called()
            assert backend.source.load_source_snapshot('websocket:attempt-chat').events == original.events
            changed = {**frame, 'content': 'changed content', 'input_attempt': {
                **frame['input_attempt'], 'fingerprint': fingerprint('changed content'),
            }}
            await channel._dispatch_envelope(conn, 'different-label', changed)
            await dispatch_one(loop, channel)
            assert not started and not status and not provider.calls
            save.assert_not_called()
            assert backend.source.load_source_snapshot('websocket:attempt-chat').events == original.events
        distinct = {**frame, 'turn_id': 'attempt-2'}
        await channel._dispatch_envelope(conn, 'local', distinct)
        await dispatch_one(loop, channel)
        assert len(provider.calls) == 1 and len(first_provider.calls) == 1
        assert len([e for e in backend.source.load_source_snapshot('websocket:attempt-chat').events if e['role'] == 'user']) == 2
    finally:
        disconnect()
        await channel.stop()
        await loop.aclose()


@pytest.mark.parametrize('after_replace', [False, True], ids=['before-replace', 'after-replace-before-ack'])
async def test_mandatory_append_failure_never_receipts_or_executes_and_committed_proof_is_readable(
    engine_config, monkeypatch, after_replace,
):
    from nanobot.session.input_attempt import principal_tag
    from nanobot.session.source_store import SourceSessionStore

    provider = ScriptedProvider([])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop, 'websocket:attempt-chat')
    channel, conn = channel_for(loop, engine_config)
    coordinator = wth.WebuiTurnCoordinator(bus=loop.bus, sessions=loop.sessions,
                                         schedule_background=lambda coro: coro.close())
    disconnect = coordinator.subscribe()
    frame = frame_for(backend.source.load_source_snapshot('websocket:attempt-chat'))
    write = SourceSessionStore._write
    faulted = False

    def fail_input(store, header, events):
        nonlocal faulted
        if not faulted and any('_input_attempt' in event for event in events):
            faulted = True
            if after_replace:
                write(store, header, events)
            raise OSError('controlled atomic boundary failure')
        return write(store, header, events)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(SourceSessionStore, '_write', fail_input)
            await channel._dispatch_envelope(conn, 'local', frame)
            await dispatch_one(loop, channel)
        assert faulted and provider.calls == []
        assert not any(e['event'] == 'message_admitted' for e in _sent_ws_payloads(conn))
        await channel._dispatch_envelope(conn, 'local', {
            'type': 'attach', 'chat_id': 'attempt-chat',
            'input_attempt_query': {'version': 1, 'turn_id': frame['turn_id'], **frame['input_attempt']},
        })
        result = [e for e in _sent_ws_payloads(conn) if e['event'] == 'attached'][-1]['input_attempt_result']
        assert result['disposition'] == ('admitted' if after_replace else 'unknown')
        users = [e for e in backend.source.load_source_snapshot('websocket:attempt-chat').events if e['role'] == 'user']
        assert len(users) == int(after_replace)
        if after_replace:
            assert result['proof']['event_id'] == users[0]['_event_id']
            assert result['proof']['principal'] == principal_tag('local')
            # A repeated delivery joins proof; it does not resume unknown effects.
            await channel._dispatch_envelope(conn, 'local', dict(frame))
            if loop.bus.inbound_size:
                await dispatch_one(loop, channel)
            assert not provider.calls
    finally:
        disconnect()
        await channel.stop()
        await loop.aclose()


async def test_one_scripted_tool_effect_and_private_evidence_never_reaches_projections(engine_config):
    import json

    from loguru import logger

    from nanobot.agent.tools.base import Tool
    from nanobot.channels.notification_routes import notification_metadata
    from nanobot.providers.base import ToolCallRequest
    from nanobot.runtime_context import public_history_message

    effects = []
    class OnceTool(Tool):
        name = 'scripted_effect'
        description = 'Record a deterministic isolated effect.'
        parameters = {'type': 'object', 'properties': {}}
        async def execute(self, **kwargs):
            effects.append('one')
            return 'recorded'
    tools = ToolRegistry()
    tools.register(OnceTool())
    provider = ScriptedProvider([LLMResponse(content='', tool_calls=[
        ToolCallRequest(id='one-tool', name='scripted_effect', arguments={}),
    ]), LLMResponse(content='done')])
    loop = AgentLoop.from_config(engine_config, tool_registry=tools, provider=provider)
    backend = upgrade(loop, 'websocket:attempt-chat')
    channel, conn = channel_for(loop, engine_config)
    coordinator = wth.WebuiTurnCoordinator(bus=loop.bus, sessions=loop.sessions,
                                         schedule_background=lambda coro: coro.close())
    disconnect = coordinator.subscribe()
    frame = frame_for(backend.source.load_source_snapshot('websocket:attempt-chat'))
    logs = []
    log_id = logger.add(lambda message: logs.append(str(message)))
    try:
        for _ in range(2):
            await channel._dispatch_envelope(conn, 'local', dict(frame))
            await dispatch_one(loop, channel)
        events = backend.source.load_source_snapshot('websocket:attempt-chat').events
        users = [event for event in events if event['role'] == 'user']
        assert len(users) == len(effects) == 1
        assert len(provider.calls) == 2  # Original tool round + final answer, no replay rounds.
        assert '_input_attempt' in users[0]
        assert '_input_attempt' not in str(provider.calls)
        assert '_input_attempt' not in str([public_history_message(event) for event in events])
        run_bytes = loop.run_store.path_for(users[0]['_run_id']).read_text()
        assert '_input_attempt' not in run_bytes and 'fingerprint' not in run_bytes and 'principal' not in run_bytes
        assert '_input_attempt' not in ''.join(logs) and 'fingerprint' not in ''.join(logs)
        assert notification_metadata('websocket', {'_input_attempt': users[0]['_input_attempt']}) == {}
        receipt = next(e for e in _sent_ws_payloads(conn) if e['event'] == 'message_admitted')
        assert json.loads(json.dumps(receipt['input_attempt_proof']))['event_id'] == users[0]['_event_id']
    finally:
        logger.remove(log_id)
        disconnect()
        await channel.stop()
        await loop.aclose()


@pytest.mark.parametrize('extra', [
    {'media': ['meaningful']}, {'attachments': [{'id': 'meaningful'}]},
    {'cli_apps': [{'id': 'app'}]}, {'mcp_presets': [{'id': 'mcp'}]},
    {'session_mentions': [{'key': 'other'}]}, {'quoted_context': 'quoted'},
    {'target_turn_id': 'another'}, {'approval_decision': {'approval_id': 'a', 'decision': 'approve'}},
    {'user_shell': True}, {'content': '/stop'}, {'content': '!echo hello'},
    {'principal': 'a' * 64}, {'_input_attempt': {'forged': True}}, {'event_id': 'forged'},
    {'run_id': 'forged'}, {'unexpected': True}, {'turn_id': ' has-whitespace'},
])
async def test_explicit_negotiation_rejects_unsupported_forms_without_publication(engine_config, extra):
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=ScriptedProvider([]))
    backend = upgrade(loop, 'websocket:attempt-chat')
    channel, conn = channel_for(loop, engine_config)
    frame = {**frame_for(backend.source.load_source_snapshot('websocket:attempt-chat')), **extra}
    before = backend.source.load_source_snapshot('websocket:attempt-chat').events
    try:
        await channel._dispatch_envelope(conn, 'local', frame)
        assert loop.bus.inbound_size == 0
        assert any(e['event'] == 'error' for e in _sent_ws_payloads(conn))
        assert backend.source.load_source_snapshot('websocket:attempt-chat').events == before
    finally:
        await channel.stop()
        await loop.aclose()


async def test_another_authorized_installation_cannot_execute_same_source_turn_identity(engine_config, monkeypatch):
    import time

    from nanobot.remote.edge import AccessIdentity

    provider = ScriptedProvider([LLMResponse(content='one')])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    backend = upgrade(loop, 'websocket:attempt-chat')
    channel, conn = channel_for(loop, engine_config)
    coordinator = wth.WebuiTurnCoordinator(bus=loop.bus, sessions=loop.sessions,
                                         schedule_background=lambda coro: coro.close())
    disconnect = coordinator.subscribe()
    frame = frame_for(backend.source.load_source_snapshot('websocket:attempt-chat'))
    try:
        await channel._dispatch_envelope(conn, 'local', frame)
        await dispatch_one(loop, channel)
        original = backend.source.load_source_snapshot('websocket:attempt-chat').events
        channel.gateway.endpoint.webui_connections.discard(conn)
        conn.mobile_identity = AccessIdentity('other-installation', ('chat.read', 'chat.write'), time.time() + 60, 1)
        monkeypatch.setattr(channel, 'revalidate_mobile_connection', AsyncMock(return_value=True))
        await channel._dispatch_envelope(conn, 'local', dict(frame))
        await dispatch_one(loop, channel)
        assert len(provider.calls) == 1
        assert backend.source.load_source_snapshot('websocket:attempt-chat').events == original
        assert any(e.get('detail') == 'input_attempt_unknown' and e.get('reason') == 'input_attempt_conflict'
                   for e in _sent_ws_payloads(conn))
    finally:
        disconnect()
        await channel.stop()
        await loop.aclose()
