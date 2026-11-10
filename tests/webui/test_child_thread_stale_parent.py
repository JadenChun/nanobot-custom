"""Regression test for A3-3 AC3.1: child thread route rejects stale parent."""

from __future__ import annotations

import asyncio
from urllib.parse import quote

import pytest

from nanobot.bus.queue import MessageBus
from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.channels.websocket.tests.ws_test_client import InProcessHttpChannel, http_get
from nanobot.session.manager import Session, SessionManager
from nanobot.session.runtime_authority import RuntimeSessionMetadata, read_runtime_session
from nanobot.webui.gateway_services import build_gateway_services


@pytest.mark.asyncio
async def test_child_thread_route_rejects_stale_parent(tmp_path, monkeypatch):
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    manager = SessionManager(workspace, sessions_root=tmp_path / "sessions")
    config = WebSocketConfig.model_validate({
        "enabled": True,
        "allowFrom": ["*"],
        "host": "127.0.0.1",
        "port": 42200,
        "path": "/",
        "websocketRequiresToken": False,
    })
    bus = MessageBus()
    gateway = build_gateway_services(
        config=config,
        bus=bus,
        session_manager=manager,
        static_dist_path=None,
        workspace_path=workspace,
        default_restrict_to_workspace=False,
        runtime_model_name=None,
        runtime_surface="browser",
        runtime_capabilities_overrides=None,
    )
    channel = InProcessHttpChannel(config, bus, gateway=gateway)
    server_task = asyncio.create_task(channel.start())

    try:
        token = channel.gateway.tokens.issue_api_token(300)

        parent_key = "websocket:chat-a"
        child_key = "delegated:a3-3-stale"
        transcript = [
            {"role": "user", "content": "old assignment"},
            {"role": "assistant", "content": "old result"},
        ]

        manager.save(Session(key=parent_key))
        old_parent = manager.read_session_snapshot(parent_key)
        assert old_parent is not None

        manager.upsert_session_metadata(
            child_key,
            {
                "delegation_child": {
                    "parent_key": parent_key,
                    "session_key": child_key,
                    "parent_incarnation": old_parent.incarnation,
                    "transcript": transcript,
                },
                "_runtime_session": RuntimeSessionMetadata(
                    profile="delegated",
                    execution_kind="native",
                    delegation_depth=1,
                    parent_key=parent_key,
                    parent_incarnation=old_parent.incarnation,
                    backend_binding="native-runner",
                ).payload(),
            },
            create_if_absent=True,
        )

        manager.delete_session(parent_key)
        manager.save(Session(key=parent_key))
        new_parent = manager.read_session_snapshot(parent_key)
        assert new_parent is not None
        assert new_parent.incarnation != old_parent.incarnation

        state = read_runtime_session(manager, child_key)
        assert state.present is True
        assert state.executable is False
        assert state.issue == "stale_parent"

        url = (
            f"http://127.0.0.1:{channel.config.port}"
            f"/api/sessions/{quote(child_key, safe='')}/webui-thread"
        )
        response = await http_get(url, headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 404
        assert "old assignment" not in response.text
    finally:
        await channel.stop()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
