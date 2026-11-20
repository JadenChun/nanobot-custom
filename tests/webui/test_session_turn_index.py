"""Turn totals and the whole-session turn index.

A navigator that lists every turn in a session needs two things the page
projection did not report: how many turns exist, and a stable label for each.
Both are derived from transcript chunk refs and user rows only, so neither
requires replaying the whole history.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.webui.transcript import (
    append_transcript_object,
    build_webui_thread_response,
    session_turn_index,
)


def _seed_turns(key: str, chat_id: str, turns: int) -> None:
    for index in range(1, turns + 1):
        append_transcript_object(
            key,
            {"event": "user", "chat_id": chat_id, "text": f"question {index}"},
        )
        append_transcript_object(
            key,
            {"event": "message", "chat_id": chat_id, "text": f"answer {index}"},
        )
        append_transcript_object(key, {"event": "turn_end", "chat_id": chat_id})


def _seed(key: str, turns: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)
    _seed_turns(key, key.split(":", 1)[1], turns)


def test_page_reports_exact_turn_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "websocket:turn-totals"
    _seed(key, 24, monkeypatch, tmp_path)

    page = build_webui_thread_response(key, limit=4, direction="latest")
    assert page is not None
    details = page["page"]

    # Every seeded turn is counted even though the page shows a slice.
    assert details["total_turn_count"] == 24
    assert details["loaded_turn_count"] >= 1
    assert details["loaded_turn_count"] < 24
    assert details["first_turn_ordinal"] == 24 - details["loaded_turn_count"]
    assert details["has_more_before"] is True

    whole = build_webui_thread_response(key)
    assert whole is not None
    assert whole["page"]["total_turn_count"] == 24
    assert whole["page"]["first_turn_ordinal"] == 0
    assert whole["page"]["loaded_turn_count"] == 24
    assert whole["page"]["has_more_before"] is False


def test_turn_index_lists_every_turn_with_user_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "websocket:turn-index"
    _seed(key, 5, monkeypatch, tmp_path)

    index = session_turn_index(key)
    assert index is not None
    assert index["total_turn_count"] == 5
    assert [turn["ordinal"] for turn in index["turns"]] == [0, 1, 2, 3, 4]
    assert [turn["label"] for turn in index["turns"]] == [
        "question 1",
        "question 2",
        "question 3",
        "question 4",
        "question 5",
    ]
    assert index["page"] == {
        "before_cursor": None,
        "has_more_before": False,
        "loaded_turn_count": 5,
    }

    # Each entry carries a usable cursor: paging with it returns that turn last.
    last_cursor = index["turns"][-1]["cursor"]
    jumped = build_webui_thread_response(key, limit=4, direction="latest", before=last_cursor)
    assert jumped is not None
    assert jumped["page"]["first_turn_ordinal"] + jumped["page"]["loaded_turn_count"] == 5


def test_turn_index_pages_backwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "websocket:turn-index-pages"
    _seed(key, 10, monkeypatch, tmp_path)

    newest = session_turn_index(key, limit=4)
    assert newest is not None
    assert [turn["ordinal"] for turn in newest["turns"]] == [6, 7, 8, 9]
    assert newest["page"]["has_more_before"] is True
    cursor = newest["page"]["before_cursor"]
    assert cursor

    older = session_turn_index(key, limit=4, before=cursor)
    assert older is not None
    assert [turn["ordinal"] for turn in older["turns"]] == [2, 3, 4, 5]

    # The total is stable across pages, so a navigator can size the whole list
    # while it streams older pages in.
    assert newest["total_turn_count"] == older["total_turn_count"] == 10


def test_turn_index_is_absent_for_an_unknown_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)
    assert session_turn_index("websocket:missing") is None


def test_turns_route_serves_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The navigator reaches the index over HTTP, not only in-process."""
    import asyncio
    import json as jsonlib

    from websockets.datastructures import Headers
    from websockets.http11 import Request

    from nanobot.webui.ws_http import GatewayHTTPHandler

    key = "websocket:turn-route"
    _seed(key, 3, monkeypatch, tmp_path)

    handler = object.__new__(GatewayHTTPHandler)
    handler.check_api_token = lambda _request: True

    request = Request(
        f"/api/sessions/{key}/turns",
        Headers([("Host", "127.0.0.1:8765")]),
    )
    response = asyncio.run(handler._handle_session_turns_get(request, key))

    assert response.status_code == 200
    payload = jsonlib.loads(response.body)
    assert payload["total_turn_count"] == 3
    assert [turn["label"] for turn in payload["turns"]] == [
        "question 1",
        "question 2",
        "question 3",
    ]
    assert payload["turns"][0]["cursor"]


def test_turns_route_rejects_an_invalid_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from websockets.datastructures import Headers
    from websockets.http11 import Request

    from nanobot.webui.ws_http import GatewayHTTPHandler

    key = "websocket:turn-route-bad"
    _seed(key, 2, monkeypatch, tmp_path)

    handler = object.__new__(GatewayHTTPHandler)
    handler.check_api_token = lambda _request: True

    request = Request(
        f"/api/sessions/{key}/turns?limit=0",
        Headers([("Host", "127.0.0.1:8765")]),
    )
    response = asyncio.run(handler._handle_session_turns_get(request, key))
    assert response.status_code == 400

