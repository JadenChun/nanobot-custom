"""E02-F08 search/read contract cases U1-U4.

The cases come from ``req-a34p7/CASES.md``. This module supplies the literal
interface details the case file leaves open (D1-D10):

- D1: ``WebuiSessionAccess.search`` / ``WebuiSessionAccess.read``.
- D2: hit fields are ``session_key`` (identifier) and ``source_kind``. The
  canonical kind is ``CANONICAL_ROW_SOURCE`` (``session``); the display kind is
  ``TRANSCRIPT_ROW_SOURCE`` (``webui_transcript``).
- D3: a read payload has ``session_key``, ``source_kind``, ``title``,
  ``updated_at``, and ``messages`` with ``message_index``, ``role``,
  ``timestamp``, and text ``content``.
- D4: canonical results carry no non-authoritative marker.
- D5: absence is the typed ``None`` result of ``read``; ``search`` never raises.
- D6: canonical rows are saved through ``SessionManager``; display rows are
  WebUI transcript JSONL files under the patched WebUI directory. Each case
  builds fresh directories.
- D7: the identifier is the exact ``str`` session key, with no trimming or case
  folding.
- D8: an unreadable display row is absent from hits; ``read`` returns ``None``.
- D9: search takes an explicit limit and ranks exact, prefix, substring, then
  message matches.
- D10: this implementation selects U2 branch A (exclusion), so display rows
  never appear in hits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kohaku.webui.session_list_index as session_list_index
import kohaku.webui.transcript as transcript
from kohaku.session.manager import SessionManager
from kohaku.webui.session_access import WebuiSessionAccess
from kohaku.webui.session_list_index import (
    CANONICAL_ROW_SOURCE,
    TRANSCRIPT_ROW_SOURCE,
    list_webui_sessions,
)

EXCLUDED = "websocket:current"
SEARCH_LIMIT = 5
READ_LIMIT = 5


@pytest.fixture(autouse=True)
def _isolate_webui_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    webui_dir = tmp_path / "webui"
    webui_dir.mkdir()
    monkeypatch.setattr(session_list_index, "get_webui_dir", lambda: webui_dir)
    monkeypatch.setattr(transcript, "get_webui_dir", lambda: webui_dir)
    return webui_dir


def _manager(tmp_path: Path) -> SessionManager:
    return SessionManager(tmp_path / "sessions")


def _canonical_session(
    manager: SessionManager,
    key: str,
    title: str,
    token: str,
) -> None:
    session = manager.get_or_create(key)
    session.metadata.update({"title": title, "title_user_edited": True})
    session.add_message("user", f"question {token}")
    session.add_message("assistant", f"answer {token}")
    manager.save(session)


def _display_row(webui_dir: Path, key: str, token: str) -> Path:
    chat_id = key.removeprefix("websocket:")
    path = webui_dir / f"{SessionManager.safe_key(key)}.jsonl"
    path.write_text(
        json.dumps({"event": "user", "chat_id": chat_id, "text": f"question {token}"})
        + "\n"
        + json.dumps({"event": "message", "chat_id": chat_id, "text": f"answer {token}"})
        + "\n",
        encoding="utf-8",
    )
    return path


def _read(access: WebuiSessionAccess, key: str, *, query: str = ""):
    return access.read(
        key,
        query=query,
        limit=READ_LIMIT,
        exclude_session_key=EXCLUDED,
    )


def test_u1_canonical_hit_is_readable(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    access = WebuiSessionAccess(manager)
    key = "websocket:u1-canonical"
    token = "u1canonicaltoken"
    _canonical_session(manager, key, "Canonical title U1", token)

    hits = access.search(token, SEARCH_LIMIT, exclude_session_key=EXCLUDED)

    assert hits
    hit = next(item for item in hits if item["session_key"] == key)
    assert hit["source_kind"] == CANONICAL_ROW_SOURCE

    result = _read(access, key)

    assert result is not None
    assert result["source_kind"] == CANONICAL_ROW_SOURCE
    assert result["title"] == "Canonical title U1"
    assert any(token in message["content"] for message in result["messages"])
    assert "non_authoritative" not in result


def test_u2_display_only_hit_is_absent(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    access = WebuiSessionAccess(manager)
    key = "websocket:u2-display-only"
    token = "u2displaytoken"
    _display_row(tmp_path / "webui", key, token)

    # The sidebar index offers the display row; only the hit list must not.
    assert key in [row["key"] for row in list_webui_sessions(manager)]

    hits = access.search(token, SEARCH_LIMIT, exclude_session_key=EXCLUDED)

    assert key not in [item["session_key"] for item in hits]
    assert all(item["source_kind"] == CANONICAL_ROW_SOURCE for item in hits)
    for item in hits:
        assert _read(access, item["session_key"]) is not None
    assert _read(access, key) is None
    assert TRANSCRIPT_ROW_SOURCE != CANONICAL_ROW_SOURCE


def test_u3_corrupted_display_row_is_absent(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    access = WebuiSessionAccess(manager)
    key = "websocket:u3-corrupt"
    token = "u3corrupttoken"
    path = tmp_path / "webui" / f"{SessionManager.safe_key(key)}.jsonl"
    path.write_bytes(
        b'{"event":"user","chat_id":"u3-corrupt","text":"question '
        + token.encode()
        + b' \xff\xfe"}\n'
    )

    hits = access.search(token, SEARCH_LIMIT, exclude_session_key=EXCLUDED)

    assert key not in [item["session_key"] for item in hits]
    assert _read(access, key) is None
    for item in hits:
        result = _read(access, item["session_key"])
        assert result is not None
        assert result["messages"]


def test_u4_identifier_round_trips_exactly(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    access = WebuiSessionAccess(manager)
    key = "websocket:Case-Sensitive:RoundTrip_01"
    token = "u4roundtriptoken"
    _canonical_session(manager, key, "Round trip U4", token)

    hits = access.search(token, SEARCH_LIMIT, exclude_session_key=EXCLUDED)

    hit = next(item for item in hits if item["session_key"] == key)
    hit_identifier = hit["session_key"]

    assert isinstance(hit_identifier, str)
    assert hit_identifier == key

    via_hit = _read(access, hit_identifier)
    direct = _read(access, key)

    assert via_hit is not None
    assert direct is not None
    assert via_hit == direct
    assert any(token in message["content"] for message in via_hit["messages"])


def test_search_hits_never_resolve_to_no_content(tmp_path: Path) -> None:
    """Cross-case invariant: every hit reads back with content."""
    manager = _manager(tmp_path)
    access = WebuiSessionAccess(manager)
    token = "sharedinvarianttoken"
    canonical_key = "websocket:invariant-canonical"
    display_key = "websocket:invariant-display"
    _canonical_session(manager, canonical_key, "Invariant canonical", token)
    _display_row(tmp_path / "webui", display_key, token)

    hits = access.search(token, SEARCH_LIMIT, exclude_session_key=EXCLUDED)

    assert [item["session_key"] for item in hits] == [canonical_key]
    for item in hits:
        result = access.read(
            item["session_key"],
            query=token,
            limit=READ_LIMIT,
            exclude_session_key=EXCLUDED,
        )
        assert result is not None
        assert result["messages"]
