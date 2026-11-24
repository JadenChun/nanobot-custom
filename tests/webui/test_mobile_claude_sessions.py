"""Exercise Claude Code through the authenticated Android gateway routes."""

from __future__ import annotations

import json
from contextlib import contextmanager
from urllib.parse import quote

import httpx
import pytest
import websockets

from kohaku.channels.websocket.tests.ws_test_client import WsTestClient
from kohaku.remote.owner import RemoteAccessOwner
from tests.webui.test_direct_claude_sessions import INITIAL_BINDING, _gateway_config
from tests.webui.test_gateway_webui_smoke import _start_gateway, _stop_gateway, _wait_for_bootstrap


@pytest.fixture
def mobile_gateway(tmp_path):
    config, base, workspace = _gateway_config(tmp_path)
    remote = RemoteAccessOwner(tmp_path / "owner")
    remote.start()
    remote.store.set_enabled(True)
    try:
        settings = json.loads(config.read_text())
        settings["remoteAccess"] = {
            "enabled": True, "stateDir": str(remote.store.root), "socketPath": str(remote.socket_path),
        }
        config.write_text(json.dumps(settings))
        origin = base.replace("http:", "https:", 1)

        def issue(name, scopes):
            pair = remote.tokens.issue_device(
                name, scopes, device_fingerprint="a" * 64, device_generation=1, origin=origin,
            )
            return {"Authorization": f"Bearer {pair.access_token}"}

        auth = issue("phone-owner", ("owner", "chat.read", "chat.write"))
        legacy = issue("phone-legacy", ("chat.read", "chat.write"))
        log = tmp_path / "gateway.log"

        @contextmanager
        def gateway():
            process = _start_gateway(config, log, claude_fixture="fake-claude")
            try:
                _wait_for_bootstrap(base, process, log)
                yield process
            finally:
                _stop_gateway(process)

        yield gateway, config, base, workspace, auth, legacy
    finally:
        remote.stop()


async def event(client, name, predicate=lambda _frame: True):
    while True:
        frame = await client.recv_raw()
        if frame["event"] == name and predicate(frame):
            return frame


def socket_url(base):
    return base.replace("http:", "ws:", 1) + "/mobile/v1/ws"


async def attach(client, chat_id):
    await client.send_json({"type": "attach", "chat_id": chat_id})
    return await event(client, "attached")


async def create(client):
    await client.send_json({"type": "new_chat", "harness": "claude_sdk"})
    return await event(client, "attached")


async def send(client, chat_id, content, turn_id, **fields):
    await client.send_json({
        "type": "message", "webui": True, "chat_id": chat_id, "content": content,
        "turn_id": turn_id, **fields,
    })


async def test_mobile_direct_chat_lock_and_recovery(mobile_gateway):
    gateway, _, base, workspace, auth, legacy = mobile_gateway
    project = workspace / "project"
    project.mkdir()
    url = socket_url(base)

    async with httpx.AsyncClient(base_url=base, headers=auth, trust_env=False) as http:
        async def read(path="read"):
            response = await http.get(f"/mobile/v1/{path}")
            response.raise_for_status()
            return response.json()

        with gateway() as process:
            with pytest.raises(websockets.exceptions.InvalidStatus) as missing:
                await websockets.connect(url)
            assert missing.value.response.status_code == 401
            async with WsTestClient(url, extra_headers=legacy) as client:
                await client.recv_ready()
                await client.send_json({"type": "new_chat", "harness": "claude_sdk"})
                assert (await event(client, "error"))["detail"] == "scope_denied"

            async with WsTestClient(url, extra_headers=auth) as client:
                ready = (await client.recv_ready()).raw
                capabilities = ready["harness_capabilities"]
                assert capabilities["claude_sdk"]["available"] is True
                before = await read()
                assert before["harness_capabilities"] == capabilities
                for rejected in (
                    {"type": "new_temporary_chat"},
                    {"type": "new_temporary_chat", "harness": "claude_sdk"},
                    {"type": "new_chat", "harness": "claude_sdk", "ephemeral": True},
                    {"type": "new_chat", "ephemeral": True},
                ):
                    await client.send_json(rejected)
                    assert (await event(client, "error"))["detail"] in {
                        "harness_requires_persistent_session", "access_denied",
                        "temporary_chat_unavailable",
                    }
                assert {row["key"] for row in (await read())["sessions"]} == {
                    row["key"] for row in before["sessions"]
                }

                attached = await create(client)
                chat_id = attached["chat_id"]
                key = f"websocket:{chat_id}"
                thread_path = f"sessions/{quote(key, safe='')}/thread"
                assert attached["harness_capabilities"] == capabilities
                assert attached["harness_binding"] == INITIAL_BINDING
                assert (await event(client, "session_updated"))["harness_binding"] == INITIAL_BINDING

                async def row():
                    return next(row for row in (await read())["sessions"] if row["key"] == key)

                assert (await row())["harness_binding"] == INITIAL_BINDING
                thread = await read(thread_path)
                assert thread["harness_binding"] == INITIAL_BINDING
                assert thread["harness_capabilities"] == capabilities

                async def select(path):
                    await client.send_json({
                        "type": "set_workspace_scope", "chat_id": chat_id,
                        "workspace_scope": {"project_path": str(path), "access_mode": "restricted"},
                    })

                await select(project)
                assert (await event(client, "session_updated"))["workspace_scope"]["project_path"] == str(project)
                await send(client, chat_id, "first request", "mobile-first")
                update = await event(client, "session_updated", lambda frame: bool(
                    frame.get("harness_binding", {}).get("locked_workspace")
                ))
                assert update["harness_binding"]["locked_workspace"] == str(project)
                assert "Claude reply to first request" in (await event(client, "delta"))["text"]
                await event(client, "turn_end")
                binding = (await row())["harness_binding"]
                assert binding["locked_workspace"] == str(project)
                assert binding["external_session_id"] == "test-sdk-session"
                assert (await read(thread_path))["harness_binding"] == binding
                assert (await read(f"{thread_path}?view=controls"))["harness_binding"] == binding
                assert (await read(f"{thread_path}?view=history"))["harness_binding"] == binding
                assert (await attach(client, chat_id))["harness_binding"] == binding
                await select(workspace)
                assert (await event(client, "error"))["reason"] == "harness_workspace_locked"
                assert (await read(thread_path))["harness_binding"] == binding

                await send(client, chat_id, "request approval before restart", "mobile-interrupted")
                assert (await event(client, "external_approval"))["approval"]["command"] == "printf phase-three"
                process.kill()
                process.wait(timeout=10)

        with gateway():
            async with WsTestClient(url, extra_headers=auth) as client:
                await client.recv_ready()
                recovered = await attach(client, chat_id)
                assert recovered["harness_binding"]["locked_workspace"] == str(project)
                assert recovered["recovery_state"]["can_continue"] is False
                assert (await row())["recovery_state"]["can_continue"] is False
                assert (await read(thread_path))["recovery_state"] == recovered["recovery_state"]


@pytest.mark.parametrize("decision", ["approve", "approve_and_save", "stop"])
async def test_mobile_direct_approval_replays_and_settles(mobile_gateway, decision):
    gateway, config, base, _, auth, legacy = mobile_gateway
    url = socket_url(base)
    with gateway():
        async with WsTestClient(url, extra_headers=auth) as client:
            await client.recv_ready()
            chat_id = (await create(client))["chat_id"]
            await send(client, chat_id, "request approval", "mobile-approval")
            pending = await event(client, "external_approval")
            assert "child_id" not in pending
            assert pending["approval"]["session_key"] == f"websocket:{chat_id}"
            assert pending["approval"]["command"] == "printf phase-three"
            assert pending["approval"]["save_exclusion_command"] == "printf"

        identity = {field: pending["approval"][field] for field in (
            "session_key", "session_incarnation", "run_id", "generation", "effect_id",
        )}
        approval = {"approval_id": pending["approval_id"], "decision": decision, **identity}
        if decision == "approve":
            async with WsTestClient(url, extra_headers=legacy) as client:
                await client.recv_ready()
                await attach(client, chat_id)
                await send(client, chat_id, "approval_decision", "legacy-decision", approval_decision=approval)
                while True:
                    frame = await client.recv_raw()
                    assert frame["event"] != "external_approval"
                    if frame["event"] == "error":
                        assert frame["detail"] == "scope_denied"
                        break

        async with WsTestClient(url, extra_headers=auth) as client:
            await client.recv_ready()
            attached = await attach(client, chat_id)
            assert await event(client, "external_approval") == pending
            if decision != "stop":
                await send(client, chat_id, "approval_decision", "mobile-decision", approval_decision=approval)
            else:
                key = quote(f"websocket:{chat_id}", safe="")
                async with httpx.AsyncClient(base_url=base, headers=auth, trust_env=False) as http:
                    controls = await http.get(f"/mobile/v1/sessions/{key}/thread?view=controls")
                    controls.raise_for_status()
                    target = controls.json()["stop_target"]
                assert target and target == attached["stop_target"]
                await send(client, chat_id, "/stop", "mobile-stop", stop_expectation=target)
            cleared = await event(client, "external_approval")
            assert cleared["approval_id"] == pending["approval_id"]
            assert cleared["approval"] is None
            if decision != "stop":
                assert (await event(client, "delta"))["text"] == f"Approval decision: {decision}"
            await event(client, "turn_end")
            if decision == "stop":
                assert (await attach(client, chat_id))["stop_target"] is None
                await send(client, chat_id, "fresh after stop", "mobile-after-stop")
                assert (await event(client, "delta"))["text"] == "Claude reply to fresh after stop"
                await event(client, "turn_end")
        excluded = json.loads(config.read_text()).get("tools", {}).get(
            "exec", {},
        ).get("sandboxExcludedCommands", [])
        assert ("printf" in excluded) == (decision == "approve_and_save")
