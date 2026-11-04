from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.providers.base import LLMResponse
from nanobot.session.legacy_h1 import (
    LegacyH1RecoveryRequiredError,
    LegacyH1StateError,
    require_legacy_h1_disposition,
)
from nanobot.session.manager import SessionManager
from tests.agent.conftest import make_loop

KEY = "telegram:42"
OWNER = "owner"


def _state(*, entries=None, notices=None, temporary_lifetime_id=None, closed=False):
    return {
        "schema_version": 1,
        "owner_id": OWNER,
        "session_key": KEY,
        "temporary_lifetime_id": temporary_lifetime_id,
        "closed": closed,
        "revision": 1,
        "next_seq": 2,
        "entries": entries or [],
        "notices": notices or [],
        "emitted_notice_ids": [],
        "notice_receipts": [],
    }


def _entry(state: str, *, result: dict | None = None):
    return {"state": state, "result": result}


def _result(disposition: str):
    return {"disposition": disposition}


UNSETTLED = {
    "pending": _state(entries=[_entry("pending-routing")]),
    "accepted": _state(entries=[_entry("next-turn")]),
    "running": _state(entries=[_entry("claimed")]),
    "uncertain": _state(entries=[_entry("uncertain")]),
    "sending": _state(entries=[_entry("completed", result=_result("sending"))]),
    "uncertain-delivery": _state(entries=[_entry("completed", result=_result("uncertain"))]),
    "notice": _state(notices=[{"state": "pending"}]),
    "temporary": _state(temporary_lifetime_id="lifetime"),
}


def _manager(tmp_path: Path) -> SessionManager:
    return SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "sessions")


def _seed_legacy(manager: SessionManager, key: str, state: object) -> Path:
    session = manager.get_or_create(key)
    manager.save(session, fsync=True)
    path = manager._get_session_path(key)
    rows = path.read_text().splitlines()
    header = json.loads(rows[0])
    header["metadata"]["busy_inbox"] = state
    rows[0] = json.dumps(header, ensure_ascii=False, allow_nan=False)
    path.write_text("\n".join(rows) + "\n")
    manager.invalidate(key)
    return path


@pytest.mark.parametrize("name", sorted(UNSETTLED))
def test_unsettled_legacy_h1_blocks_lifecycle_changes_without_discarding_state(
    tmp_path: Path, name: str,
) -> None:
    manager = _manager(tmp_path)
    path = _seed_legacy(manager, KEY, UNSETTLED[name])

    with pytest.raises(LegacyH1RecoveryRequiredError, match="legacy H1"):
        manager.reset_session(KEY)
    with pytest.raises(LegacyH1RecoveryRequiredError, match="recover"):
        manager.reset_session(KEY, replacement_events=[{"role": "user", "content": "replacement"}])
    with pytest.raises(LegacyH1RecoveryRequiredError, match="legacy H1"):
        manager.delete_session(KEY)

    assert path.exists()
    metadata = manager.read_session_metadata(KEY)
    assert metadata is not None
    assert metadata["metadata"]["busy_inbox"] == UNSETTLED[name]


def test_malformed_legacy_h1_blocks_turns_and_lifecycle_without_echoing_payload(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    _seed_legacy(manager, KEY, {"content": "SECRET-LEGACY-PAYLOAD"})

    for operation in (
        lambda: manager.reset_session(KEY),
        lambda: manager.delete_session(KEY),
    ):
        with pytest.raises(LegacyH1StateError, match="malformed") as error:
            operation()
        assert "SECRET-LEGACY-PAYLOAD" not in str(error.value)


def test_fully_settled_legacy_h1_metadata_is_inert(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    settled = _state(entries=[_entry("completed", result=_result("delivered"))])
    _seed_legacy(manager, KEY, settled)

    require_legacy_h1_disposition({"busy_inbox": settled})
    manager.reset_session(KEY)
    assert manager.read_session_metadata(KEY) is not None
    assert manager.delete_session(KEY)
    assert manager.read_session_metadata(KEY) is None


def test_closed_temporary_lifetime_is_inert(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    closed = _state(temporary_lifetime_id="lifetime", closed=True)
    _seed_legacy(manager, KEY, closed)

    require_legacy_h1_disposition({"busy_inbox": closed})
    manager.reset_session(KEY)


@pytest.mark.asyncio
async def test_unsettled_legacy_h1_refuses_new_turn_before_provider_effect(tmp_path: Path) -> None:
    loop = make_loop(tmp_path)
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="should not run"))
    _seed_legacy(loop.sessions, KEY, UNSETTLED["pending"])

    with pytest.raises(LegacyH1RecoveryRequiredError, match="legacy H1"):
        await loop.process_direct("new question", session_key=KEY, channel="telegram", chat_id="42")

    loop.provider.chat_with_retry.assert_not_awaited()
    metadata = loop.sessions.read_session_metadata(KEY)
    assert metadata is not None
    assert metadata["metadata"]["busy_inbox"] == UNSETTLED["pending"]


@pytest.mark.asyncio
async def test_unrelated_session_stays_usable_and_recovery_records_stay_private(tmp_path: Path) -> None:
    loop = make_loop(tmp_path)
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="ordinary reply"))
    _seed_legacy(loop.sessions, KEY, UNSETTLED["pending"])
    recovery_key = "h1-recovery:" + "a" * 64
    _seed_legacy(loop.sessions, recovery_key, UNSETTLED["pending"])

    result = await loop.process_direct(
        "ordinary question", session_key="telegram:other", channel="telegram", chat_id="other",
    )

    assert result is not None and result.content == "ordinary reply"
    assert KEY in {item["key"] for item in loop.sessions.list_sessions()}
    assert recovery_key not in {item["key"] for item in loop.sessions.list_sessions()}
    assert loop.sessions.read_session_metadata(KEY) is not None
