"""Requirements-only case C5 (R3 no-query thread projection ceiling).

Provenance: a3-evidence/a3-reconciliation-2026-11-14/req-a36a/CASES.md.
Scenarios, inputs and expected outcomes are preserved from that file.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from nanobot.webui.transcript import (
    append_transcript_object,
    build_webui_thread_response,
)
from nanobot.webui.ws_http import THREAD_PROJECTION_BUDGET_BYTES, GatewayHTTPHandler


def _seed_turns(key: str, chat_id: str, turns: int) -> None:
    for index in range(1, turns + 1):
        append_transcript_object(
            key,
            {
                "event": "user",
                "chat_id": chat_id,
                "text": f"question {index} " + ("x" * 3000),
            },
        )
        append_transcript_object(
            key,
            {
                "event": "message",
                "chat_id": chat_id,
                "text": f"answer {index} " + ("y" * 3000),
            },
        )
        append_transcript_object(key, {"event": "turn_end", "chat_id": chat_id})


def _handler(tmp_path: Path) -> GatewayHTTPHandler:
    handler = object.__new__(GatewayHTTPHandler)
    handler.stop_target_for_key = None
    handler.config = SimpleNamespace(host="127.0.0.1")
    handler.check_api_token = lambda _request: True
    handler.session_manager = SimpleNamespace(
        read_session_file=lambda _key: {"messages": []},
    )
    handler.workspaces = SimpleNamespace(
        scope_for_session_key=lambda _key: SimpleNamespace(project_path=str(tmp_path)),
        scope_payload=lambda _scope, **_kwargs: {"project_path": str(tmp_path)},
        folder_grants_allowed_for_session=lambda _key: False,
    )
    handler.workspace_access_recovery = None
    handler.media = SimpleNamespace(
        augment_transcript_media=lambda paths, **_kwargs: paths,
        rewrite_local_markdown_images=lambda text, **_kwargs: text,
    )
    handler._log = SimpleNamespace(exception=lambda *_args: None)
    return handler


# C5 - No-query thread projection stays within the byte ceiling.
def test_c5_no_query_thread_projection_stays_within_byte_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)
    key = "websocket:no-query-ceiling"
    chat_id = "no-query-ceiling"
    _seed_turns(key, chat_id, 320)

    unconstrained = build_webui_thread_response(key)
    assert unconstrained is not None
    stored_message_count = len(unconstrained["messages"])
    unconstrained_bytes = len(
        json.dumps(unconstrained, ensure_ascii=False).encode("utf-8")
    )
    assert unconstrained_bytes > THREAD_PROJECTION_BUDGET_BYTES
    assert stored_message_count > 0

    request = Request(
        f"/api/webui/threads/{key}",
        Headers([("Host", "127.0.0.1:8765")]),
    )
    response = _handler(tmp_path)._handle_webui_thread_get(request, key)

    assert response.status_code == 200
    body = response.body
    payload = json.loads(body)
    assert len(body) <= THREAD_PROJECTION_BUDGET_BYTES
    assert len(payload["messages"]) < stored_message_count
    assert payload["page"]["has_more_before"] is True
    assert payload["page"]["loaded_message_count"] == len(payload["messages"])
