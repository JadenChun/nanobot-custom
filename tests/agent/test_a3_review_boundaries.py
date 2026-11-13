"""Regression tests for A3 review findings.

These tests cover only public boundary contracts: denied write targets stay
denied, completed commands keep their success result, stale parents cannot
control children, mixed legacy transcripts are quarantined, journal open
failures are bounded, and compressed oversize fetches are refused.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import socket
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from nanobot.agent.child_sessions import ChildAdmission
from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.policy import RiskyActionPolicy
from nanobot.agent.runner import ToolCallRequest
from nanobot.agent.tools import web as web_module
from nanobot.agent.tools.exec_session import ExecSessionManager
from nanobot.agent.tools.web import _get_with_safe_redirects
from nanobot.process_completion import ProcessCompletionRuntime
from nanobot.session.manager import Session, SessionManager
from nanobot.webui import transcript


def test_denied_write_target_stays_denied() -> None:
    allowed = {"value": False}

    def resolver(tool_name, raw_path):
        return Path("/project/file.txt") if allowed["value"] else None

    policy = RiskyActionPolicy(
        workspace=Path("/project"),
        write_target_resolver=resolver,
    )
    call = ToolCallRequest(
        id="write-1",
        name="write_file",
        arguments={"path": "link.txt", "content": "x" * 32},
    )

    assert policy._risky_reason(call) is None

    allowed["value"] = True

    class Tool:
        def resolve_write_path(self, path):
            return Path("/project/file.txt")

    error, target = policy.validate_write_target_binding(
        call, Tool(), {"path": "link.txt"}
    )

    assert target is None
    assert "not assessable" in error


async def test_completed_command_keeps_success_when_polled_after_deadline() -> None:
    with tempfile.TemporaryDirectory() as d:
        manager = ExecSessionManager()
        session_id, initial = await manager.start(
            command="echo hi",
            cwd=d,
            env={"PATH": "/usr/bin:/bin"},
            timeout=1,
            shell_program="/bin/bash",
            login=False,
            yield_time_ms=0,
            max_output_chars=1000,
        )
        await asyncio.sleep(1.3)
        poll = await manager.write(
            session_id=session_id,
            chars=None,
            close_stdin=False,
            terminate=False,
            yield_time_ms=0,
            max_output_chars=1000,
        )
        await manager.close_all()

    assert poll.done is True
    assert poll.exit_code == 0
    assert poll.timed_out is False
    assert "hi" in poll.output


async def test_child_control_rejects_recreated_parent(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test"
    sessions = SessionManager(
        tmp_path / "workspace", sessions_root=tmp_path / "sessions"
    )
    manager = ForegroundAgentManager(
        provider=provider, workspace=tmp_path, sessions=sessions
    )
    key = "websocket:parent"
    parent = sessions.get_or_create(key)
    sessions.save(parent)
    sessions._jsonl_store.source.upgrade_source_session(
        key, expected_revision=parent.revision,
        expected_generation=parent.generation, expected_incarnation=parent.incarnation,
    )
    sessions.invalidate(key)
    parent = sessions.get_or_create(key)

    child = await manager._delegation_tree.reserve(
        parent=parent,
        parent_key=key,
        assignment="work",
        attempt_id="attempt-1",
    )
    assert isinstance(child, ChildAdmission)
    await manager._delegation_tree.activate(child.child_id)

    sessions.delete_session(key)
    sessions.save(Session(key=key))

    result = await manager.control_child(
        parent_key=key, child_id=child.child_id, action="stop"
    )

    assert result["ok"] is False
    assert result["reason"] == "stale_parent"


def test_mixed_legacy_history_is_quarantined(tmp_path: Path) -> None:
    legacy = Path(
        tmp_path / f"{SessionManager.safe_key('websocket:a:b')}.jsonl"
    )
    legacy.write_text(
        json.dumps({"event": "user", "chat_id": "a:b", "text": "first"})
        + "\n"
        + json.dumps({"event": "user", "chat_id": "a_b", "text": "second"})
        + "\n",
        encoding="utf-8",
    )

    with patch.object(transcript, "get_webui_dir", lambda: tmp_path):
        assert transcript.read_transcript_lines("websocket:a:b") == []
        assert transcript.read_transcript_lines("websocket:a_b") == []
        assert legacy.is_file()


def test_journal_open_failure_is_bounded(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "processes.sqlite3").mkdir()

    assistant = type("Assistant", (), {"sessions": None})()

    with pytest.raises(Exception) as exc:
        ProcessCompletionRuntime(root, assistant)

    assert (
        type(exc.value).__name__ == "OperationalError"
        or "unable to open database file" in str(exc.value)
    )


async def test_compressed_oversize_fetch_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(web_module, "MAX_FETCH_DOWNLOAD_BYTES", 64)
    monkeypatch.setattr(
        "nanobot.security.network.socket.getaddrinfo",
        lambda *args, **kwargs: [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                0,
                "",
                ("93.184.216.34", 0),
            )
        ],
    )
    body = gzip.compress(b"x" * 4096)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "gzip",
            },
            content=body,
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        response, error = await _get_with_safe_redirects(
            client, "https://example.com/compressed"
        )

    assert response is None
    assert "download limit" in error or "compression" in error
