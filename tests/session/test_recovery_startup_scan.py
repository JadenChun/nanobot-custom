"""Startup recovery classifies from bounded headers, not settled canonical bodies."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.session.manager import SessionManager
from nanobot.session.recovery import RecoveryCoordinator

SETTLED_KEY = "websocket:settled-startup"
UNRESOLVED_KEY = "websocket:unresolved-startup"


def _canonical_manager(tmp_path: Path, keys: tuple[str, ...]) -> SessionManager:
    manager = SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    backend = manager._jsonl_store
    for key in keys:
        session = manager.get_or_create(key)
        session.metadata["webui"] = True
        manager.save(session)
        backend.source.upgrade_source_session(
            key,
            expected_revision=session.revision,
            expected_generation=session.generation,
            expected_incarnation=session.incarnation,
        )
        manager.invalidate(key)
    return manager


def _count_body_loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from nanobot.session.source_store import SourceSessionStore

    loaded: list[str] = []
    original = SourceSessionStore.load_source_snapshot

    def tracked(self: SourceSessionStore, key: str, *args: object, **kwargs: object):
        loaded.append(key)
        return original(self, key, *args, **kwargs)

    monkeypatch.setattr(SourceSessionStore, "load_source_snapshot", tracked)
    return loaded


@pytest.mark.asyncio
async def test_settled_canonical_session_scan_reads_no_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = _canonical_manager(tmp_path, (SETTLED_KEY,))
    loaded = _count_body_loads(monkeypatch)
    bus = MessageBus()

    await RecoveryCoordinator(sessions, bus).scan()

    # A session with no pending input, checkpoint, or interrupted tail is
    # classified from its header alone. Its canonical events are never read.
    assert SETTLED_KEY not in loaded
    assert bus.inbound.empty()
    assert bus.outbound.empty()


@pytest.mark.asyncio
async def test_unresolved_canonical_session_still_loads_owner_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.session.pending_followups import (
        mint_pending_followup_id,
        record_pending_followup,
    )
    from nanobot.session.persistence_metadata import (
        SOURCE_PENDING_FOLLOWUPS_METADATA_KEY,
    )

    sessions = _canonical_manager(tmp_path, (UNRESOLVED_KEY,))
    backend = sessions.source_backend(UNRESOLVED_KEY)
    assert backend is not None
    from nanobot.bus.events import InboundMessage

    message = InboundMessage(
        channel="websocket",
        sender_id="user",
        chat_id="unresolved-startup",
        content="finish this turn",
        metadata={"webui": True},
    )
    mint_pending_followup_id(message)
    record = record_pending_followup(message)
    from nanobot.session.execution_owner import acquire_execution

    async with acquire_execution(backend, UNRESOLVED_KEY, run_id="startup") as owner:
        assert backend.source.admit_pending_followup(UNRESOLVED_KEY, owner, record)
    header = backend.source.load_source_snapshot(UNRESOLVED_KEY).header
    assert SOURCE_PENDING_FOLLOWUPS_METADATA_KEY in header["metadata"]

    loaded = _count_body_loads(monkeypatch)
    bus = MessageBus()
    await RecoveryCoordinator(sessions, bus).scan()

    # The journal marker is a header fact, so the session is still unresolved
    # and goes through the full owner-validated recovery path.
    assert UNRESOLVED_KEY in loaded
    restored = bus.inbound.get_nowait()
    assert restored.content == "finish this turn"
