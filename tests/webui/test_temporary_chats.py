"""Requirements cases for Temporary Chat tombstone retirement (U7 / E02-F10).

The author cases come from
``a3-evidence/a3-reconciliation-2026-11-14/req-a38/CASES.md`` C7. Turn
settlement is observed through the product's own runtime events, and the
bounded fallback is exercised with the injectable monotonic clock.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kohaku.bus.queue import MessageBus
from kohaku.bus.runtime_events import (
    RuntimeEventContext,
    SessionTurnStarted,
    TurnCompleted,
)
from kohaku.security.folder_grants import FolderGrantStore
from kohaku.session.manager import SessionManager
from kohaku.webui.temporary_chats import TemporaryChatError, WebUITemporaryChats
from kohaku.webui.workspaces import WebUIWorkspaceController

SETTLE_GRACE_SECONDS = 6.0


class _Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _build(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[WebUITemporaryChats, MessageBus, SessionManager, _Clock]:
    monkeypatch.setattr("kohaku.config.paths.get_data_dir", lambda: tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    instance = tmp_path / "instance"
    sessions = SessionManager(home, sessions_root=tmp_path / "sessions")
    controller = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=home,
        default_restrict_to_workspace=True,
        grant_store=FolderGrantStore(
            agent_home=home,
            instance_root=instance,
            path=instance / "webui" / "folder-grants.json",
        ),
        native_supervisor_available=True,
    )
    bus = MessageBus()
    clock = _Clock()
    chats = WebUITemporaryChats(
        bus=bus,
        session_manager=sessions,
        workspaces=controller,
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
        clock=clock,
    )
    controller.set_temporary_owners(
        binding=lambda chat_id: chats.binding(None, chat_id),
        set_mode=lambda chat_id, mode: chats.set_access_mode(chat_id, mode),
    )
    return chats, bus, sessions, clock


def _context(chat_id: str) -> RuntimeEventContext:
    return RuntimeEventContext(
        channel="websocket",
        chat_id=chat_id,
        session_key=f"websocket:{chat_id}",
    )


async def _start_turn(bus: MessageBus, chat_id: str) -> None:
    await bus.publish(SessionTurnStarted(context=_context(chat_id)))


async def _complete_turn(bus: MessageBus, chat_id: str) -> None:
    await bus.publish(TurnCompleted(context=_context(chat_id), outcome="completed"))


@pytest.mark.asyncio
async def test_tombstone_retires_after_turn_generation_settles(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chats, bus, sessions, clock = _build(tmp_path, monkeypatch)
    owner = object()
    chat_id = chats.create(owner, trusted_webui=True)
    await _start_turn(bus, chat_id)

    await chats.discard(owner, chat_id)
    assert chats.tombstone_exists(chat_id, 1)

    await _complete_turn(bus, chat_id)
    assert chats.tombstone_exists(chat_id, 1)
    clock.advance(SETTLE_GRACE_SECONDS)

    assert not chats.tombstone_exists(chat_id)
    assert chats.tombstone_retired(chat_id)
    assert chats.tombstone_count() == 0

    # The late event is stale: no turn, session, or transcript authority.
    assert chats.should_persist_transcript(chat_id) is False
    with pytest.raises(TemporaryChatError) as refused:
        chats.message_policy(owner, chat_id, "late event")
    assert refused.value.detail == "temporary_chat_unavailable"
    assert chats.binding(None, chat_id) is None
    assert sessions.get_cached(f"websocket:{chat_id}") is None
    assert all(session.key != f"websocket:{chat_id}" for session in sessions.list_sessions())


@pytest.mark.asyncio
async def test_settled_discard_without_active_turn_retires_after_grace(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chats, _bus, sessions, clock = _build(tmp_path, monkeypatch)
    owner = object()
    chat_id = chats.create(owner, trusted_webui=True)

    await chats.discard(owner, chat_id)
    assert chats.tombstone_exists(chat_id, 0)
    clock.advance(SETTLE_GRACE_SECONDS)

    assert not chats.tombstone_exists(chat_id)
    assert chats.tombstone_retired(chat_id)
    assert chats.should_persist_transcript(chat_id) is False
    with pytest.raises(TemporaryChatError):
        chats.message_policy(owner, chat_id, "late event")
    assert sessions.list_sessions() == []


@pytest.mark.asyncio
async def test_bounded_fallback_retires_an_unsettled_turn(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chats, bus, _sessions, clock = _build(tmp_path, monkeypatch)
    owner = object()
    chat_id = chats.create(owner, trusted_webui=True)
    await _start_turn(bus, chat_id)
    await chats.discard(owner, chat_id)

    assert chats.tombstone_exists(chat_id, 1)
    clock.advance(WebUITemporaryChats.TOMBSTONE_FALLBACK_SECONDS + 1.0)

    assert not chats.tombstone_exists(chat_id)
    assert chats.tombstone_retired(chat_id)
    assert chats.should_persist_transcript(chat_id) is False


@pytest.mark.asyncio
async def test_late_turn_start_after_discard_keeps_the_tombstone(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chats, bus, _sessions, clock = _build(tmp_path, monkeypatch)
    owner = object()
    chat_id = chats.create(owner, trusted_webui=True)
    await _start_turn(bus, chat_id)
    await chats.discard(owner, chat_id)
    await _complete_turn(bus, chat_id)
    clock.advance(SETTLE_GRACE_SECONDS)
    assert not chats.tombstone_exists(chat_id)

    # A queued turn that starts after retirement must not regain authority.
    await _start_turn(bus, chat_id)
    assert chats.should_persist_transcript(chat_id) is False
    with pytest.raises(TemporaryChatError):
        chats.message_policy(owner, chat_id, "queued turn")


@pytest.mark.asyncio
async def test_tombstone_count_stays_bounded_across_settled_cycles(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chats, bus, _sessions, clock = _build(tmp_path, monkeypatch)
    owner = object()
    chat_ids: list[str] = []
    for _ in range(200):
        chat_id = chats.create(owner, trusted_webui=True)
        chat_ids.append(chat_id)
        await _start_turn(bus, chat_id)
        await chats.discard(owner, chat_id)
        await _complete_turn(bus, chat_id)
        clock.advance(SETTLE_GRACE_SECONDS)
        assert chats.tombstone_count() == 0

    assert chats.retired_tombstone_count() <= WebUITemporaryChats.MAX_RETIRED_TOMBSTONES
    assert chats.should_persist_transcript(chat_ids[0]) is False


@pytest.mark.asyncio
async def test_active_tombstones_are_capacity_bounded(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chats, _bus, _sessions, _clock = _build(tmp_path, monkeypatch)
    owner = object()
    for _ in range(WebUITemporaryChats.MAX_ACTIVE_TOMBSTONES + 40):
        chat_id = chats.create(owner, trusted_webui=True)
        await chats.discard(owner, chat_id)

    assert chats.tombstone_count() <= WebUITemporaryChats.MAX_ACTIVE_TOMBSTONES
    assert chats.retired_tombstone_count() <= WebUITemporaryChats.MAX_RETIRED_TOMBSTONES
