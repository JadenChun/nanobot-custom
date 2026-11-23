"""Stop authority projection in the saved-history response.

A chat whose Stop target this gateway cannot derive must omit the field, so a
client keeps its legacy Stop path. A chat with canonical authority emits the
field, including an explicit null when nothing is live.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import quote

import pytest

from kohaku.channels.websocket.tests.ws_test_client import http_get
from kohaku.session.manager import Session
from tests.webui.test_child_thread_route import (
    _auth_headers,
    _make_channel,
    _make_session_manager,
)

PARENT_KEY = "websocket:chat-a"


def _thread_url(channel, key: str) -> str:
    return (
        f"http://127.0.0.1:{channel.config.port}"
        f"/api/sessions/{quote(key, safe='')}/webui-thread"
    )


@pytest.mark.asyncio
async def test_thread_response_omits_stop_target_when_it_cannot_be_derived(
    tmp_path: Path,
) -> None:
    """A plain chat keeps legacy Stop: the field must be absent, never null."""
    sessions = _make_session_manager(tmp_path)
    sessions.save(Session(key=PARENT_KEY, metadata={"webui": True}))
    channel = _make_channel(tmp_path, sessions)
    server_task = asyncio.create_task(channel.start())
    try:
        response = await http_get(_thread_url(channel, PARENT_KEY), headers=_auth_headers(channel))
        assert response.status_code == 200, response.text
        body = response.json()
        assert "stop_target" not in body
    finally:
        await channel.stop()
        await server_task


@pytest.mark.asyncio
async def test_thread_response_includes_stop_target_when_authority_exists(
    tmp_path: Path,
) -> None:
    """A canonical chat declares modern Stop, with null while nothing is live."""
    sessions = _make_session_manager(tmp_path)
    session = Session(key=PARENT_KEY, metadata={"webui": True})
    sessions.save(session)
    sessions._jsonl_store.source.upgrade_source_session(
        PARENT_KEY,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    assert sessions.source_backend(PARENT_KEY) is not None
    channel = _make_channel(tmp_path, sessions)
    channel.gateway.http.stop_target_for_key = lambda key, **kwargs: None
    server_task = asyncio.create_task(channel.start())
    try:
        response = await http_get(_thread_url(channel, PARENT_KEY), headers=_auth_headers(channel))
        assert response.status_code == 200, response.text
        body = response.json()
        assert "stop_target" in body
        assert body["stop_target"] is None
    finally:
        await channel.stop()
        await server_task
