"""Regression tests for the authorized child transcript read route (A3-5)."""

from __future__ import annotations

import asyncio
import itertools
from pathlib import Path
from urllib.parse import quote

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.channels.websocket.tests.ws_test_client import (
    InProcessHttpChannel,
    http_get,
)
from nanobot.session.manager import Session, SessionManager
from nanobot.webui.gateway_services import build_gateway_services

_PARENT_KEY = "websocket:chat-a"
_CHILD_KEY = "delegated:a3-5-child"
_TRANSCRIPT = [
    {"role": "user", "content": "assignment"},
    {"role": "assistant", "content": "result"},
]
_IN_PROCESS_PORTS = itertools.count(41_100)


@pytest.fixture(autouse=True)
def _isolate_runtime_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)


def _next_port() -> int:
    """Allocate a unique in-process route key, not a native listening socket."""
    return next(_IN_PROCESS_PORTS)


def _make_channel(tmp_path: Path, session_manager: SessionManager) -> InProcessHttpChannel:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    config = WebSocketConfig.model_validate({
        "enabled": True,
        "allowFrom": ["*"],
        "host": "127.0.0.1",
        "port": _next_port(),
        "path": "/",
        "websocketRequiresToken": False,
    })
    bus = MessageBus()
    gateway = build_gateway_services(
        config=config,
        bus=bus,
        session_manager=session_manager,
        static_dist_path=None,
        workspace_path=workspace,
        default_restrict_to_workspace=False,
        runtime_model_name=None,
        runtime_surface="browser",
        runtime_capabilities_overrides=None,
    )
    return InProcessHttpChannel(config, bus, gateway=gateway)


def _make_session_manager(tmp_path: Path) -> SessionManager:
    return SessionManager(tmp_path / "workspace", sessions_root=tmp_path / "sessions")


def _seed_child_session(
    session_manager: SessionManager,
    *,
    child_key: str = _CHILD_KEY,
    parent_key: str = _PARENT_KEY,
    transcript: list[dict[str, str]] | None = None,
) -> None:
    session_manager.save(Session(key=parent_key))
    session_manager.upsert_session_metadata(
        child_key,
        {
            "delegation_child": {
                "parent_key": parent_key,
                "session_key": child_key,
                "transcript": [
                    dict(message) for message in (transcript or _TRANSCRIPT)
                ],
            }
        },
        create_if_absent=True,
    )


def _thread_url(channel: InProcessHttpChannel, child_key: str) -> str:
    encoded = quote(child_key, safe="")
    return (
        f"http://127.0.0.1:{channel.config.port}"
        f"/api/sessions/{encoded}/webui-thread"
    )


def _auth_headers(channel: InProcessHttpChannel) -> dict[str, str]:
    token = channel.gateway.tokens.issue_api_token(300)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_child_thread_returns_bounded_transcript(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    _seed_child_session(session_manager)
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        response = await http_get(
            _thread_url(channel, _CHILD_KEY),
            headers=_auth_headers(channel),
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert [
            (message["role"], message["content"]) for message in body["messages"]
        ] == [
            ("user", "assignment"),
            ("assistant", "result"),
        ]
    finally:
        await channel.stop()
        await server_task


@pytest.mark.asyncio
async def test_child_thread_requires_known_websocket_parent(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    orphan_key = "delegated:a3-5-no-parent"
    non_websocket_key = "delegated:a3-5-non-websocket-parent"
    unknown_parent_key = "delegated:a3-5-unknown-parent"

    session_manager.save(Session(key=orphan_key))
    session_manager.save(Session(key="telegram:chat-a"))
    session_manager.upsert_session_metadata(
        non_websocket_key,
        {
            "delegation_child": {
                "parent_key": "telegram:chat-a",
                "session_key": non_websocket_key,
                "transcript": [dict(message) for message in _TRANSCRIPT],
            }
        },
        create_if_absent=True,
    )
    session_manager.upsert_session_metadata(
        unknown_parent_key,
        {
            "delegation_child": {
                "parent_key": "websocket:missing-parent",
                "session_key": unknown_parent_key,
                "transcript": [dict(message) for message in _TRANSCRIPT],
            }
        },
        create_if_absent=True,
    )
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        headers = _auth_headers(channel)
        for child_key in (orphan_key, non_websocket_key, unknown_parent_key):
            response = await http_get(_thread_url(channel, child_key), headers=headers)
            assert response.status_code == 404, (child_key, response.text)
    finally:
        await channel.stop()
        await server_task


@pytest.mark.asyncio
async def test_child_thread_requires_token(tmp_path: Path) -> None:
    session_manager = _make_session_manager(tmp_path)
    _seed_child_session(session_manager)
    channel = _make_channel(tmp_path, session_manager)
    server_task = asyncio.create_task(channel.start())
    try:
        url = _thread_url(channel, _CHILD_KEY)
        missing = await http_get(url)
        assert missing.status_code == 401, missing.text

        incorrect = await http_get(
            url,
            headers={"Authorization": "Bearer nbwt_not-a-valid-token"},
        )
        assert incorrect.status_code == 401, incorrect.text
    finally:
        await channel.stop()
        await server_task
