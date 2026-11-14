"""Regression tests for the A3-8 no-query transcript read budget."""

from __future__ import annotations

from nanobot.webui.transcript import (
    append_transcript_object,
    build_webui_thread_response,
)


def _append_numbered_turn(key: str, chat_id: str, idx: int) -> None:
    append_transcript_object(
        key,
        {"event": "user", "chat_id": chat_id, "text": f"question {idx} " + ("x" * 24)},
    )
    append_transcript_object(
        key,
        {"event": "message", "chat_id": chat_id, "text": f"answer {idx} " + ("y" * 24)},
    )
    append_transcript_object(key, {"event": "turn_end", "chat_id": chat_id})


def _contents(payload: dict) -> list[str]:
    return [str(message.get("content") or "") for message in payload["messages"]]


def test_no_query_thread_response_uses_bounded_page(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.webui.transcript._NO_QUERY_TRANSCRIPT_PAGE_LIMIT", 4)
    key = "websocket:no-query-budget"
    chat_id = "no-query-budget"
    for idx in range(1, 7):
        _append_numbered_turn(key, chat_id, idx)

    payload = build_webui_thread_response(key)

    assert payload is not None
    assert payload["page"]["has_more_before"] is True
    assert payload["page"]["loaded_message_count"] <= 6
    contents = _contents(payload)
    assert any(text.startswith("question 6 ") for text in contents)
    assert not any(text.startswith("question 1 ") for text in contents)

    older = build_webui_thread_response(key, before=payload["page"]["before_cursor"])

    assert older is not None
    assert older["page"]["loaded_message_count"] <= 8
    older_contents = _contents(older)
    assert any(text.startswith("question 1 ") for text in older_contents)
    assert not any(text.startswith("question 6 ") for text in older_contents)
