"""Turn totals and the whole-session turn index.

A navigator that lists every turn in a session needs two things the page
projection did not report: how many turns exist, and a stable label for each.
Both are derived from transcript chunk refs and user rows only, so neither
requires replaying the whole history.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from kohaku.session.runtime_authority import DurableSessionAuthority
from kohaku.webui import transcript
from kohaku.webui.transcript import (
    append_transcript_object,
    build_webui_thread_response,
    session_turn_index,
)
from tests.webui.test_child_thread_route import _make_channel


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
    monkeypatch.setattr("kohaku.config.paths.get_data_dir", lambda: tmp_path)
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


@pytest.mark.parametrize("segmented", [False, True])
def test_thread_window_and_index_count_only_structural_turn_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, segmented: bool,
) -> None:
    key = "websocket:literal-turn-end"
    chat_id = "literal-turn-end"
    monkeypatch.setattr("kohaku.config.paths.get_data_dir", lambda: tmp_path)
    prefix_turns = 4 if segmented else 0
    if segmented:
        # Force real rotations, then keep the two probe turns in the active file.
        with monkeypatch.context() as patch:
            patch.setattr(transcript, "_ACTIVE_TRANSCRIPT_ROTATE_BYTES", 540)
            patch.setattr(transcript, "_TARGET_ACTIVE_TRANSCRIPT_BYTES", 300)
            _seed_turns(key, chat_id, prefix_turns)
        assert (transcript.webui_transcript_segments_dir(key) / "manifest.json").is_file()

    for prompt, reply in [("literal marker", "turn_end"), ("next prompt", "next reply")]:
        append_transcript_object(key, {"event": "user", "chat_id": chat_id, "text": prompt})
        append_transcript_object(key, {"event": "message", "chat_id": chat_id, "text": reply})
        append_transcript_object(key, {"event": "turn_end", "chat_id": chat_id})

    page = build_webui_thread_response(key, limit=2, direction="latest")
    index = session_turn_index(key, limit=2)
    assert page is not None and index is not None
    assert page["page"]["total_turn_count"] == index["total_turn_count"] == prefix_turns + 2
    assert page["page"]["first_turn_ordinal"] == prefix_turns + 1
    assert [row["content"] for row in page["messages"]] == ["next prompt", "next reply"]
    assert [turn["label"] for turn in index["turns"]] == ["literal marker", "next prompt"]
    older = build_webui_thread_response(key, limit=2, before=page["page"]["before_cursor"])
    assert older is not None
    assert [row["content"] for row in older["messages"]] == ["literal marker", "turn_end"]


def test_turn_index_is_absent_for_an_unknown_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("kohaku.config.paths.get_data_dir", lambda: tmp_path)
    assert session_turn_index("websocket:missing") is None


@pytest.mark.parametrize(("source_state", "status"), [
    ("active", 200),
    ("absent", 404),
    ("deleted", 404),
    ("not_webui", 404),
    ("source_missing", 503),
])
async def test_turns_route_requires_saved_history_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_state: str, status: int,
) -> None:
    """A readable display file cannot authorize prompt summaries on its own."""
    key = "websocket:turn-route"
    _seed(key, 3, monkeypatch, tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = await DurableSessionAuthority(workspace, sessions_root=tmp_path / "sessions").acquire()
    try:
        manager = lease.manager
        if source_state != "absent":
            assert manager.upsert_session_metadata(
                key, {"webui": source_state != "not_webui"}, create_if_absent=True,
            )
        if source_state == "deleted":
            assert manager.delete_session(key)
        elif source_state == "source_missing":
            manager._get_session_path(key).unlink()

        gateway = _make_channel(tmp_path, manager).gateway
        token = gateway.tokens.issue_api_token(300)
        headers = Headers([("Host", "127.0.0.1:8765"), ("Authorization", f"Bearer {token}")])
        connection = SimpleNamespace(
            remote_address=("127.0.0.1", 1234), request=SimpleNamespace(headers=headers),
        )
        path = f"/api/sessions/{key}/turns"
        response = await gateway.http.dispatch(connection, Request(path, headers))
        assert response is not None and response.status_code == status
        if status != 200:
            assert b"question" not in response.body
            return
        payload = json.loads(response.body)
        assert payload["total_turn_count"] == 3
        assert [turn["label"] for turn in payload["turns"]] == [
            "question 1", "question 2", "question 3",
        ]
        assert payload["turns"][0]["cursor"]

        refused = await gateway.http.dispatch(connection, Request(path, Headers()))
        assert refused is not None and refused.status_code == 401
    finally:
        assert lease.close is not None
        await lease.close()


def test_turns_route_rejects_an_invalid_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from websockets.datastructures import Headers
    from websockets.http11 import Request

    from kohaku.webui.ws_http import GatewayHTTPHandler

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

