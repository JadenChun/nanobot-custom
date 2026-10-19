"""Real temporary manager/channel lifecycle proofs on isolated copied roots."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.bus.events import INBOUND_META_RUNTIME_CONTROL, RUNTIME_CONTROL_SESSION_DISCARD
from nanobot.channels.websocket.runtime import WebSocketChannel
from nanobot.channels.websocket.tests.test_websocket_channel import (
    _basic_handler,
    _new_temporary_chat,
)
from nanobot.session.manager import Session, SessionManager
from nanobot.session.runtime_authority import OfflineSessionOperator
from nanobot.webui.temporary_chats import TemporaryChatError
from nanobot.webui.transcript import read_transcript_lines
from tests.session.test_p5_stage2_consumers import state


@pytest.mark.parametrize('evidence', ['absent', 'legacy', 'source', 'corrupt', 'duplicate', 'symlink', 'wrong-key'])
def test_cancellation_receipt_source_check_is_header_only(tmp_path, monkeypatch, evidence):
    import json

    manager = SessionManager(tmp_path / 'workspace', sessions_root=tmp_path / 'sessions')
    key = 'cli:header'
    path = manager._get_session_path(key)
    header = {'_type': 'metadata', 'key': key, 'metadata': {}}
    if evidence == 'source':
        header['metadata'] = {'source_schema_version': 1}
    elif evidence == 'wrong-key':
        header['key'] = 'other'
    if evidence == 'symlink':
        target = tmp_path / 'target'
        target.write_text(json.dumps(header) + '\n')
        path.symlink_to(target)
    elif evidence == 'corrupt':
        path.write_text('not json\n')
    elif evidence == 'duplicate':
        path.write_text('{"_type":"metadata","key":"cli:header","metadata":{},"metadata":{}}\n')
    elif evidence != 'absent':
        # Invalid history is deliberately irrelevant to intact format evidence.
        path.write_text(json.dumps(header) + '\nnever read this history\n')
    monkeypatch.setattr(manager, '_load', lambda key: pytest.fail('history load'))
    monkeypatch.setattr(manager, '_repair', lambda *a, **k: pytest.fail('repair'))
    monkeypatch.setattr(manager, 'source_backend', lambda key: pytest.fail('source load'))
    calls = []
    with manager.track_unadmitted_cancellation(key) as receipt:
        inserted = manager.record_unadmitted_cancellation(receipt, lambda: calls.append('insert') or True)
    assert inserted == (evidence in {'absent', 'legacy'})
    assert calls == (['insert'] if inserted else [])
    assert not manager._cancellation_receipts


def test_cancellation_receipt_exact_identity_release_and_revocation(tmp_path):
    from copy import copy

    manager = SessionManager(tmp_path / 'workspace', sessions_root=tmp_path / 'sessions')
    other = SessionManager(tmp_path / 'other', sessions_root=tmp_path / 'sessions')
    calls = []

    def insert():
        calls.append('insert')
        return True
    with manager.track_unadmitted_cancellation('cli:private') as receipt:
        assert not other.record_unadmitted_cancellation(receipt, insert)
        assert not manager.record_unadmitted_cancellation(copy(receipt), insert)
        manager.get_or_create_transient('cli:private', retain=True)
        manager.invalidate('cli:private')
        assert not manager.record_unadmitted_cancellation(receipt, insert)
    assert not manager.record_unadmitted_cancellation(receipt, insert)
    assert not calls and not manager._cancellation_receipts


def test_cancellation_receipt_foreign_equality_and_resolver_install_are_denied(tmp_path):
    manager = SessionManager(tmp_path / 'workspace', sessions_root=tmp_path / 'sessions')
    calls = []

    class Foreign:
        def __eq__(self, other):
            return True

        def __hash__(self):
            return receipt.__hash__()

    with manager.track_unadmitted_cancellation('cli:install') as receipt:
        assert not manager.record_unadmitted_cancellation(Foreign(), lambda: calls.append('insert') or True)
        manager._install_mode_resolver(MagicMock(), operator=False)
        assert not manager.record_unadmitted_cancellation(receipt, lambda: calls.append('insert') or True)
    assert not calls and not manager._cancellation_receipts


def test_cancellation_privacy_publication_thread_order_is_storage_then_mutex(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    manager = SessionManager(tmp_path / 'workspace', sessions_root=tmp_path / 'sessions')
    key = 'cli:thread'
    storage_held, publish, published = Event(), Event(), Event()

    def publish_under_storage_lock():
        with manager.locked_session_files():
            storage_held.set()
            assert publish.wait(timeout=5)
            manager._remember(Session(key=key, policy=manager.get_or_create_transient('cli:detached').policy))
            manager.invalidate(key)
            published.set()

    calls = []
    with manager.track_unadmitted_cancellation(key) as receipt:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(publish_under_storage_lock)
            assert storage_held.wait(timeout=5)
            # Storage unavailable must fail closed, never wait while holding privacy mutex.
            from filelock import Timeout
            with pytest.raises(Timeout):
                manager.record_unadmitted_cancellation(receipt, lambda: calls.append('insert') or True)
            publish.set()
            assert published.wait(timeout=5)
            future.result(timeout=5)
        assert not manager.record_unadmitted_cancellation(receipt, lambda: calls.append('insert') or True)
    assert not calls and not manager._cancellation_receipts


def test_cancellation_insertion_holds_privacy_mutex_until_complete(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock, get_ident

    manager = SessionManager(tmp_path / 'workspace', sessions_root=tmp_path / 'sessions')
    key = 'cli:insert-order'
    inserting, finish_insert, publication_attempt, published = Event(), Event(), Event(), Event()
    publication_thread = None

    class ObservedMutex:
        def __init__(self):
            self.lock = Lock()

        def __enter__(self):
            if get_ident() == publication_thread:
                publication_attempt.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    manager._cancellation_privacy_lock = ObservedMutex()

    def insert():
        inserting.set()
        assert finish_insert.wait(timeout=5)
        assert not published.is_set()
        return True

    def publish():
        nonlocal publication_thread
        publication_thread = get_ident()
        from nanobot.session.manager import SessionPolicy
        manager._remember(Session(key=key, policy=SessionPolicy(persist=False)))
        manager.invalidate(key)
        published.set()

    with manager.track_unadmitted_cancellation(key) as receipt:
        with ThreadPoolExecutor(max_workers=2) as pool:
            insertion = pool.submit(manager.record_unadmitted_cancellation, receipt, insert)
            assert inserting.wait(timeout=5)
            publication = pool.submit(publish)
            assert publication_attempt.wait(timeout=5)
            assert not published.is_set()
            finish_insert.set()
            assert insertion.result(timeout=5)
            publication.result(timeout=5)
        assert not manager.record_unadmitted_cancellation(receipt, lambda: pytest.fail('revoked insert'))
    assert not manager._cancellation_receipts


@pytest.fixture(autouse=True)
def isolated_webui(tmp_path, monkeypatch):
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.webui.workspaces.get_webui_dir", lambda: tmp_path / "webui")


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("disposal", ["discard", "disconnect"])
async def test_real_channel_temporary_lifecycle_and_late_events(tmp_path, managed, disposal):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = (await OfflineSessionOperator(workspace, copied_sessions_root=tmp_path / "sessions").acquire()
             if managed else None)
    manager = lease.manager if lease else SessionManager(workspace, sessions_root=tmp_path / "sessions")
    bus = MagicMock()
    bus.publish_inbound = AsyncMock()
    channel = WebSocketChannel({"enabled": True, "allowFrom": ["*"]}, bus,
        gateway=_basic_handler(bus, session_manager=manager, workspace_path=workspace))
    owner, other = AsyncMock(), AsyncMock()
    try:
        before = state(manager)
        chat_id = await _new_temporary_chat(channel, owner)
        key = f"websocket:{chat_id}"
        session = manager.get_cached(key)
        assert session is channel._temporary_chats._active_sessions[chat_id]
        assert session is not None and not session.policy.persist and not session.policy.log_content
        assert {"create_goal", "update_goal", "spawn", "cron"} <= session.policy.disabled_tools
        manager._max_cached_sessions = 1
        manager.get_or_create("cli:cache-pressure")
        assert key not in manager._cache
        assert manager.get_cached(key) is session
        with pytest.raises(TemporaryChatError, match="^temporary_chat_unavailable$"):
            channel._temporary_chats.message_policy(other, chat_id, "hello")
        for command in ("/goal private", "/new", "/dream"):
            with pytest.raises(TemporaryChatError, match="^temporary_chat_command_rejected$"):
                channel._temporary_chats.message_policy(owner, chat_id, command)
        with pytest.raises(TemporaryChatError, match="^temporary_chat_workspace_rejected$"):
            channel._temporary_chats.validate_workspace_update(chat_id)
        with pytest.raises(TemporaryChatError, match="^temporary_chat_unavailable$"):
            channel._temporary_chats.validate_attach(chat_id)
        upload = tmp_path / "private-upload.txt"
        upload.write_text("attachment")
        channel.gateway.media.store_inbound_attachments = MagicMock(return_value=([str(upload)], None))
        await channel._dispatch_envelope(owner, "webui", {"type": "message", "chat_id": chat_id,
            "content": "private", "webui": True, "turn_id": "turn-1",
            "media": [{"data_url": "data:text/plain;base64,cHJpdmF0ZQ=="}],
            "workspace_scope": {"project_path": str(tmp_path), "access_mode": "full"}})
        assert manager.get_cached(key) is session
        assert state(manager) == before
        assert manager.read_session_metadata(key) is None
        assert read_transcript_lines(key) == []
        if disposal == "discard":
            await channel._dispatch_envelope(owner, "webui", {"type": "discard_temporary_chat", "chat_id": chat_id})
        else:
            await channel._cleanup_connection(owner)
        control = bus.publish_inbound.await_args.args[0]
        assert control.metadata[INBOUND_META_RUNTIME_CONTROL] == RUNTIME_CONTROL_SESSION_DISCARD
        assert manager.get_cached(key) is None
        assert not upload.exists()
        assert not channel._temporary_chats.owns(owner, chat_id)
        with pytest.raises(TemporaryChatError, match="^temporary_chat_unavailable$"):
            channel._temporary_chats.message_policy(owner, chat_id, "late")
        assert channel._persist_turn_transcript_event(chat_id, {"content": "late"}, phase="complete", metadata={"webui": True})
        assert channel._persist_turn_stream_event(chat_id, {"content": "late"}, completed_text="late", phase="answer", metadata={"webui": True})
        assert read_transcript_lines(key) == [] and state(manager) == before
    finally:
        channel._temporary_chats.close()
        if lease:
            await lease.close()


async def test_managed_transient_default_is_detached_and_activation_rejects_collisions(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = await OfflineSessionOperator(workspace, copied_sessions_root=tmp_path / "sessions").acquire()
    manager = lease.manager
    try:
        key = "websocket:private"
        before = state(manager)
        first = manager.get_or_create_transient(key)
        assert manager.get_or_create_transient(key) is not first
        assert manager.get_cached(key) is None and state(manager) == before
        retained = manager.get_or_create_transient(key, retain=True, disabled_tools=("cron",))
        assert manager.get_cached(key) is retained
        assert manager.get_or_create_transient(key) is not retained
        with pytest.raises(RuntimeError, match="^temporary session activation conflicts with session policy$"):
            manager.get_or_create_transient(key, retain=True, disabled_tools=("spawn",))
        assert manager.get_cached(key) is retained and state(manager) == before
        manager.get_or_create("cli:cached-persistent")
        with pytest.raises(RuntimeError, match="^temporary session activation conflicts with session policy$"):
            manager.get_or_create_transient("cli:cached-persistent", retain=True)
        manager.admit_persistent_write("cli:durable")
        before = state(manager)
        with pytest.raises(RuntimeError, match="^temporary session activation conflicts with session policy$"):
            manager.get_or_create_transient("cli:durable", retain=True)
        assert not manager.get_or_create_transient("cli:durable").policy.persist
        assert manager.get_cached("cli:durable") is None and state(manager) == before
    finally:
        await lease.close()


@pytest.mark.parametrize("evidence", ["record", "sidecar"])
async def test_managed_activation_rejects_invalid_evidence_without_cache_or_write(tmp_path, evidence):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = await OfflineSessionOperator(workspace, copied_sessions_root=tmp_path / "sessions").acquire()
    manager = lease.manager
    key = "websocket:private-key"
    try:
        manager.admit_persistent_write(key)
        if evidence == "record":
            name = manager._mode_resolver._record_name(key)
            manager._mode_resolver.storage.write(name, manager._mode_resolver.storage.read(name) + b" ")
        else:
            manager._mode_resolver.storage.write(f"{manager._storage_key(key)}.checkpoint.json", b"{}")
        before = state(manager)
        with pytest.raises(RuntimeError) as error:
            manager.get_or_create_transient(key, retain=True)
        assert key not in str(error.value) and str(manager.sessions_dir) not in str(error.value)
        assert state(manager) == before and key not in manager._cache
    finally:
        await lease.close()


def test_unmanaged_activation_rejects_durable_collision_even_with_live_transient_cache(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = SessionManager(workspace, sessions_root=tmp_path / "sessions")
    key = "websocket:collision"
    retained = manager.get_or_create_transient(key, retain=True)
    manager._jsonl_store.save(Session(key))
    before = state(manager)
    with pytest.raises(RuntimeError, match="^temporary session activation conflicts with session policy$"):
        manager.get_or_create_transient(key, retain=True)
    assert manager.get_cached(key) is retained and state(manager) == before
