"""Operation-bound grant receipts survive concurrent management refreshes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import websockets

from nanobot.bus.queue import MessageBus
from nanobot.channels.websocket.runtime import WebSocketChannel, WebSocketConfig
from nanobot.security.folder_grants import FolderGrantStore, FolderGrantUpsertResult
from nanobot.session.manager import SessionManager
from nanobot.webui.gateway_services import build_gateway_services


async def _webui_mutation(
    connection: Any,
    request_id: str,
    payload: dict[str, object],
) -> dict[str, Any]:
    await connection.send(json.dumps({
        "type": "webui_request",
        "request_id": request_id,
        "action": "workspace.grant.upsert",
        "payload": payload,
    }))
    while True:
        response = json.loads(await asyncio.wait_for(connection.recv(), timeout=5))
        if (
            isinstance(response, dict)
            and response.get("event") == "webui_response"
            and response.get("request_id") == request_id
        ):
            return response


@pytest.mark.asyncio
async def test_noop_alias_receipt_stays_bound_across_concurrent_grant_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    instance = tmp_path / "instance"
    home = instance / "workspace"
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_c = tmp_path / "project-c"
    alias_a = tmp_path / "project-a-alias"
    for path in (home, project_a, project_b, project_c):
        path.mkdir(parents=True)
    alias_a.symlink_to(project_a, target_is_directory=True)

    config_path = instance / "config.json"
    monkeypatch.setattr("nanobot.config.loader._current_config_path", config_path)
    session_manager = SessionManager(home, sessions_root=instance / "sessions")
    config = WebSocketConfig.model_validate({
        "enabled": True,
        "allowFrom": ["*"],
        "host": "127.0.0.1",
        "port": 0,
        "path": "/ws",
        "websocketRequiresToken": True,
    })
    bus = MessageBus()
    gateway = build_gateway_services(
        config=config,
        bus=bus,
        session_manager=session_manager,
        static_dist_path=None,
        workspace_path=home,
        default_restrict_to_workspace=True,
        config_path=config_path,
        runtime_model_name=None,
        runtime_surface="browser",
        runtime_capabilities_overrides=None,
    )
    grant_store = gateway.workspaces.grant_store
    assert isinstance(grant_store, FolderGrantStore)
    upsert_operations: list[FolderGrantUpsertResult] = []
    original_upsert_with_result = grant_store.upsert_with_result

    def record_upsert(
        raw: object,
        *,
        expected_revision: int,
    ) -> FolderGrantUpsertResult:
        operation = original_upsert_with_result(raw, expected_revision=expected_revision)
        upsert_operations.append(operation)
        return operation

    monkeypatch.setattr(grant_store, "upsert_with_result", record_upsert)
    channel = WebSocketChannel(config.model_dump(), bus, gateway=gateway)

    original_broadcast = gateway.http._broadcast_workspace_metadata
    first_broadcast_entered = asyncio.Event()
    release_first_broadcast = asyncio.Event()
    pause_next_broadcast = False

    async def hold_first_broadcast() -> None:
        nonlocal pause_next_broadcast
        if pause_next_broadcast:
            pause_next_broadcast = False
            first_broadcast_entered.set()
            await release_first_broadcast.wait()
        await original_broadcast()

    gateway.http._broadcast_workspace_metadata = hold_first_broadcast
    server_task = asyncio.create_task(channel.start())
    first = second = forwarded = reconnect = None

    async def connect(client_id: str, *, forwarded_ip: str | None = None):
        token = gateway.tokens.issue_token(60, audience="webui")
        headers = {"X-Forwarded-For": forwarded_ip} if forwarded_ip else None
        port = channel._server.sockets[0].getsockname()[1]
        return await websockets.connect(
            f"ws://127.0.0.1:{port}/ws?client_id={client_id}&token={token}",
            origin=f"http://127.0.0.1:{port}",
            additional_headers=headers,
        )

    try:
        for _ in range(500):
            if channel._server is not None:
                break
            if server_task.done():
                await server_task
            await asyncio.sleep(0.01)
        assert channel._server is not None

        first = await connect("local-a")
        second = await connect("local-b")
        forwarded = await connect("forwarded", forwarded_ip="203.0.113.12")

        async def upsert(
            connection: Any,
            request_id: str,
            path: Path,
            expected_revision: int,
        ) -> dict[str, Any]:
            return await _webui_mutation(
                connection,
                request_id,
                {
                    "expected_revision": expected_revision,
                    "path": str(path),
                    "file_access": "read_only",
                    "commands": False,
                },
            )

        seed_a = await upsert(first, "seed-a", project_a, 0)
        add_c = await upsert(second, "add-c", project_c, 1)
        assert seed_a["ok"] is True
        assert add_c["ok"] is True
        assert add_c["result"]["management"]["grants_revision"] == 2

        stable_alias = await upsert(first, "stable-alias", alias_a, 2)
        assert stable_alias["ok"] is True

        pause_next_broadcast = True
        alias_request = {
            "expected_revision": 2,
            "path": str(alias_a),
            "file_access": "read_only",
            "commands": False,
        }
        alias_a_response = asyncio.create_task(
            _webui_mutation(first, "racing-alias-a", alias_request)
        )
        await asyncio.wait_for(first_broadcast_entered.wait(), timeout=5)
        committed_a = grant_store.snapshot()
        assert committed_a.revision == 2
        assert {grant.path for grant in committed_a.grants} == {
            str(project_a.resolve()),
            str(project_c.resolve()),
        }

        # A forwarded authenticated caller cannot join this private operation.
        forwarded_replay = await _webui_mutation(
            forwarded,
            "racing-alias-a",
            alias_request,
        )
        assert forwarded_replay["ok"] is False
        assert forwarded_replay["error"]["message"] == "local_ui_required"
        assert "grant_receipt" not in forwarded_replay.get("result", {})

        add_b = await upsert(second, "add-b", project_b, 2)
        assert add_b["ok"] is True
        assert add_b["result"]["management"]["grants_revision"] == 3
        release_first_broadcast.set()
        response_a = await asyncio.wait_for(alias_a_response, timeout=5)

        assert response_a["ok"] is True
        result_a = response_a["result"]
        receipt_a = result_a["grant_receipt"]
        assert set(receipt_a) == {
            "submitted_path",
            "canonical_path",
            "file_access",
            "commands",
            "grants_revision",
        }
        assert receipt_a == {
            "submitted_path": str(alias_a),
            "canonical_path": str(project_a.resolve()),
            "file_access": "read_only",
            "commands": False,
            "grants_revision": 2,
        }
        assert upsert_operations[3].submitted_path == str(alias_a)
        assert upsert_operations[3].grant.path == str(project_a.resolve())
        assert upsert_operations[3].snapshot.revision == receipt_a["grants_revision"]
        # The mutable global view reflects B, while the receipt still identifies
        # the exact no-op A operation and its unchanged committed revision.
        assert result_a["management"]["grants_revision"] == 3
        assert "grant_receipt" not in result_a["management"]
        management_grants = result_a["management"]["grants"]
        assert [row["path"] for row in management_grants] == [
            str(project_a.resolve()),
            str(project_b.resolve()),
            str(project_c.resolve()),
        ]
        assert all(
            row["file_access"] == "read_only" and row["commands"] is False
            for row in management_grants
        )
        assert add_b["result"]["grant_receipt"] == {
            "submitted_path": str(project_b),
            "canonical_path": str(project_b.resolve()),
            "file_access": "read_only",
            "commands": False,
            "grants_revision": 3,
        }

        reconnect = await connect("local-reconnect")
        replay = await _webui_mutation(reconnect, "racing-alias-a", alias_request)
        assert replay["ok"] is True
        assert replay["result"]["grant_receipt"] == receipt_a
        assert grant_store.snapshot().revision == 3
        assert len(upsert_operations) == 5
        generic_payload = gateway.workspaces.payload(
            controls_available=True,
            management_available=True,
        )
        assert "grant_receipt" not in generic_payload
    finally:
        release_first_broadcast.set()
        if reconnect is not None:
            await reconnect.close()
        if forwarded is not None:
            await forwarded.close()
        if first is not None:
            await first.close()
        if second is not None:
            await second.close()
        await channel.stop()
        await server_task
