"""Phase 3 acceptance cases at the authenticated gateway boundary."""

from __future__ import annotations

import asyncio
import json
from urllib.parse import quote

import pytest
import websockets

from tests.webui.test_gateway_webui_smoke import (
    _free_port,
    _get_json,
    _recv_until,
    _start_gateway,
    _stop_gateway,
    _wait_for_bootstrap,
    _write_smoke_config,
)

INITIAL_BINDING = {
    "backend": "external",
    "harness": "claude_sdk",
    "external_session_id": None,
    "locked_workspace": None,
    "runtime_generation": 0,
}


def _gateway_config(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ws_port = _free_port()
    config = tmp_path / "config.json"
    _write_smoke_config(config, workspace=workspace, ws_port=ws_port, gateway_port=_free_port())
    return config, f"http://127.0.0.1:{ws_port}", workspace


def _session_rows(base, bootstrap):
    return _get_json(f"{base}/api/sessions", token=bootstrap["api_token"])["sessions"]


def _ws_url(bootstrap):
    return f'{bootstrap["ws_url"]}?token={bootstrap["token"]}&client_id=smoke'


async def _message(ws, chat_id, prompt, turn_id):
    await ws.send(json.dumps({
        "type": "message", "chat_id": chat_id, "content": prompt,
        "webui": True, "turn_id": turn_id,
    }))


@pytest.mark.parametrize("sdk_available", [False, True])
async def test_creation_reconnect_and_temporary_rejection(tmp_path, sdk_available):
    config, base, _workspace = _gateway_config(tmp_path)
    log = tmp_path / "gateway.log"
    process = _start_gateway(
        config, log, claude_fixture="fake-claude" if sdk_available else "no-claude",
    )
    try:
        bootstrap = _wait_for_bootstrap(base, process, log)
        async with websockets.connect(_ws_url(bootstrap)) as ws:
            ready = await _recv_until(ws, "ready")
            assert ready["harness_capabilities"]["claude_sdk"]["available"] is sdk_available
            await ws.send(json.dumps({"type": "new_chat", "harness": "claude_sdk"}))
            if sdk_available:
                attached = await _recv_until(ws, "attached")
                assert attached["harness_binding"] == INITIAL_BINDING
                key = f'websocket:{attached["chat_id"]}'
                row = next(row for row in _session_rows(base, bootstrap) if row["key"] == key)
                assert row["harness_binding"] == INITIAL_BINDING
                reconnect = _wait_for_bootstrap(base, process, log)
                async with websockets.connect(_ws_url(reconnect)) as second:
                    await _recv_until(second, "ready")
                    await second.send(json.dumps({"type": "attach", "chat_id": attached["chat_id"]}))
                    assert (await _recv_until(second, "attached"))["harness_binding"] == INITIAL_BINDING
            else:
                assert (await _recv_until(ws, "error"))["detail"] == "harness_unavailable"

            await ws.send(json.dumps({"type": "new_chat"}))
            native = await _recv_until(ws, "attached")
            assert native.get("harness_binding") is None

            before = {row["key"] for row in _session_rows(base, bootstrap)}
            for frame in (
                {"type": "new_temporary_chat", "harness": "claude_sdk"},
                {"type": "new_chat", "harness": "claude_sdk", "ephemeral": True},
            ):
                await ws.send(json.dumps(frame))
                error = await _recv_until(ws, "error")
                assert error["detail"] == "harness_requires_persistent_session"
            assert {row["key"] for row in _session_rows(base, bootstrap)} == before
    finally:
        _stop_gateway(process)


async def test_first_turn_locks_workspace_and_interruption_requires_fresh_input(tmp_path):
    config, base, workspace = _gateway_config(tmp_path)
    alternate = workspace / "checkout"
    alternate.mkdir()
    log = tmp_path / "gateway.log"
    calls_path = tmp_path / "claude-calls.jsonl"
    process = _start_gateway(config, log, claude_fixture="fake-claude")

    def calls():
        return [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []

    try:
        bootstrap = _wait_for_bootstrap(base, process, log)
        async with websockets.connect(_ws_url(bootstrap)) as ws:
            await _recv_until(ws, "ready")
            await ws.send(json.dumps({"type": "new_chat", "harness": "claude_sdk"}))
            chat_id = (await _recv_until(ws, "attached"))["chat_id"]
            await _recv_until(ws, "session_updated")

            async def select(path):
                await ws.send(json.dumps({
                    "type": "set_workspace_scope", "chat_id": chat_id,
                    "workspace_scope": {"project_path": str(path), "access_mode": "restricted"},
                }))

            await select(alternate)
            assert (await _recv_until(ws, "session_updated"))["workspace_scope"]["project_path"] == str(alternate)
            await _message(ws, chat_id, "first request", "test-first")
            edit = (await _recv_until(ws, "file_edit"))["edits"][0]
            assert edit["path"] == "hello.txt"
            assert (alternate / edit["path"]).read_text() == "hello"
            assert "Claude reply to first request" in (await _recv_until(ws, "delta"))["text"]
            assert (await _recv_until(ws, "turn_end")).get("outcome", "completed") == "completed"
            key = f"websocket:{chat_id}"
            assert calls() == [{"prompt": "first request", "workspace": str(alternate), "resume": None}]
            row = next(row for row in _session_rows(base, bootstrap) if row["key"] == key)
            binding = row["harness_binding"]
            assert binding["locked_workspace"] == str(alternate.resolve())
            assert binding["external_session_id"] == "test-sdk-session"
            controls = _get_json(
                f"{base}/api/sessions/{quote(key, safe='')}/webui-thread?view=controls",
                token=bootstrap["api_token"],
            )
            assert controls["harness_binding"] == binding
            history = _get_json(
                f"{base}/api/sessions/{quote(key, safe='')}/webui-thread",
                token=bootstrap["api_token"],
            )
            assert any(
                edit["path"] == "hello.txt"
                for message in history["messages"] for edit in message.get("fileEdits", [])
            )

            await select(workspace)
            error = await _recv_until(ws, "error")
            assert error["reason"] == "harness_workspace_locked"

            await _message(ws, chat_id, "interrupt this SDK turn", "test-interrupted")
            async with asyncio.timeout(8):
                while len(calls()) < 2:
                    await asyncio.sleep(0.05)
            assert calls()[1] == {
                "prompt": "interrupt this SDK turn", "workspace": str(alternate),
                "resume": "test-sdk-session",
            }
            # Simulate an unclean stop before the SDK reports its outcome.
            process.kill()
            process.wait(timeout=10)

        process = _start_gateway(config, log, claude_fixture="fake-claude")
        bootstrap = _wait_for_bootstrap(base, process, log)
        async with websockets.connect(_ws_url(bootstrap)) as ws:
            await _recv_until(ws, "ready")
            await ws.send(json.dumps({"type": "attach", "chat_id": chat_id}))
            attached = await _recv_until(ws, "attached")
            assert attached["harness_binding"]["locked_workspace"] == str(alternate)
            assert attached["harness_binding"]["external_session_id"] == "test-sdk-session"
            assert attached["recovery_state"]["status"] == "awaiting_user"
            assert attached["recovery_state"]["can_continue"] is False
            await asyncio.sleep(0.3)
            assert len(calls()) == 2
            await _message(ws, chat_id, "fresh request", "test-fresh")
            assert "Claude reply to fresh request" in (await _recv_until(ws, "delta"))["text"]
            assert (await _recv_until(ws, "turn_end")).get("outcome", "completed") == "completed"
            assert calls()[2] == {
                "prompt": "fresh request", "workspace": str(alternate), "resume": "test-sdk-session",
            }
    finally:
        _stop_gateway(process)


@pytest.mark.parametrize("decision", ["approve", "approve_and_save", "deny", "stop"])
async def test_live_approval_uses_the_original_claude_turn(tmp_path, decision):
    config, base, _workspace = _gateway_config(tmp_path)
    log = tmp_path / "gateway.log"
    process = _start_gateway(config, log, claude_fixture="fake-claude")
    try:
        bootstrap = _wait_for_bootstrap(base, process, log)
        async with websockets.connect(_ws_url(bootstrap)) as ws:
            await _recv_until(ws, "ready")
            await ws.send(json.dumps({"type": "new_chat", "harness": "claude_sdk"}))
            chat_id = (await _recv_until(ws, "attached"))["chat_id"]
            await _message(ws, chat_id, "request approval", "test-approval")
            async with asyncio.timeout(8):
                frames = []
                while not frames or frames[-1]["event"] != "external_approval":
                    frames.append(json.loads(await ws.recv()))
            pending = frames[-1]
            assert pending["approval"]["command"] == "printf phase-three"
            assert pending["approval"]["save_exclusion_command"] == "printf"
            assert any(frame["event"] == "reasoning_delta" for frame in frames)
            assert any(
                tool["name"] == "exec"
                for frame in frames for tool in frame.get("tool_events", [])
            )
            if decision == "stop":
                target = next(
                    frame["stop_target"] for frame in reversed(frames)
                    if frame["event"] == "stop_target" and frame["stop_target"]
                )
                await ws.send(json.dumps({
                    "type": "message", "chat_id": chat_id, "content": "/stop",
                    "webui": True, "turn_id": "test-stop", "stop_expectation": target,
                }))
                cleared = (await _recv_until(ws, "external_approval"))
                assert cleared["approval"] is None
                await _recv_until(ws, "turn_end")
                await _message(ws, chat_id, "fresh after stop", "test-after-stop")
                assert (await _recv_until(ws, "delta"))["text"] == "Claude reply to fresh after stop"
                await _recv_until(ws, "turn_end")
            else:
                identity = {
                    field: pending["approval"][field]
                    for field in ("session_key", "session_incarnation", "run_id", "generation", "effect_id")
                }
                await ws.send(json.dumps({
                    "type": "message", "chat_id": chat_id, "content": "approval_decision",
                    "webui": True, "turn_id": "test-decision",
                    "approval_decision": {
                        "approval_id": pending["approval_id"], "decision": decision, **identity,
                    },
                }))
                cleared = await _recv_until(ws, "external_approval")
                assert cleared["approval"] is None
                assert (await _recv_until(ws, "delta"))["text"] == f"Approval decision: {decision}"
                await _recv_until(ws, "turn_end")
            excluded = json.loads(config.read_text()).get("tools", {}).get("exec", {}).get(
                "sandboxExcludedCommands", [],
            )
            assert ("printf" in excluded) == (decision == "approve_and_save")
    finally:
        _stop_gateway(process)
