"""Requirements cases for session-delete completeness (U6 / E02-F06).

The author cases come from
``a3-evidence/a3-reconciliation-2026-11-14/req-a38/CASES.md`` C6. The fault is
injected on the transcript unlink, which is the deterministic cleanup seam the
case asks for.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from nanobot.session.manager import SessionManager
from nanobot.webui.transcript import (
    append_transcript_object,
    read_transcript_lines,
    webui_transcript_path,
)
from nanobot.webui.ws_http import GatewayHTTPHandler

KEY = "websocket:delete-case"


def _handler(sessions: SessionManager) -> SimpleNamespace:
    return SimpleNamespace(
        check_api_token=lambda _request: True,
        session_manager=sessions,
        cron_service=None,
        local_trigger_store=None,
        workspaces=SimpleNamespace(discard_draft_scope=lambda _key: False),
    )


def _delete(sessions: SessionManager) -> dict[str, object]:
    response = GatewayHTTPHandler._handle_session_delete(
        _handler(sessions),
        Request(f"/api/sessions/{KEY}/delete", Headers()),
        KEY,
    )
    assert response.status_code == 200
    return json.loads(response.body)


def _seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[SessionManager, Path]:
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)
    sessions = SessionManager(tmp_path / "workspace")
    session = sessions.get_or_create(KEY)
    session.add_message("user", "keep me readable")
    sessions.save(session, fsync=True)
    append_transcript_object(KEY, {"event": "user", "chat_id": "delete-case", "text": "hello"})
    append_transcript_object(
        KEY, {"event": "message", "chat_id": "delete-case", "text": "world"},
    )
    return sessions, webui_transcript_path(KEY)


def test_session_delete_reports_partial_then_completes_on_retry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nanobot.webui.transcript import pending_transcript_deletions

    sessions, transcript_path = _seed(tmp_path, monkeypatch)
    before = read_transcript_lines(KEY)
    assert len(before) == 2

    original_unlink = Path.unlink

    def failing_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self == transcript_path:
            raise OSError("injected transcript unlink failure")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    first = _delete(sessions)

    assert first["complete"] is False
    assert str(transcript_path) in first["remaining"]
    assert read_transcript_lines(KEY) == before
    assert pending_transcript_deletions(KEY) == (str(transcript_path),)

    monkeypatch.setattr(Path, "unlink", original_unlink)
    second = _delete(sessions)

    assert second["complete"] is True
    assert second["remaining"] == []
    assert not transcript_path.exists()
    assert pending_transcript_deletions(KEY) == ()

    third = _delete(sessions)
    assert third == {"deleted": False, "complete": True, "remaining": []}


def test_session_delete_reports_complete_when_cleanup_succeeds(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions, transcript_path = _seed(tmp_path, monkeypatch)

    result = _delete(sessions)

    assert result["deleted"] is True
    assert result["complete"] is True
    assert result["remaining"] == []
    assert not transcript_path.exists()
    assert not sessions._get_session_path(KEY).exists()
