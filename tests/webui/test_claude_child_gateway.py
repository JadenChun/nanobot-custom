"""Verify the original chat owns a running Claude child's live controls."""

import asyncio
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from kohaku.channels.websocket.runtime import WebSocketChannel, WebSocketConfig
from kohaku.channels.websocket.tests.ws_test_client import WsTestClient, http_get
from kohaku.config import loader
from kohaku.config.schema import Config, RemoteAccessConfig
from kohaku.harness.backend import ExternalApprovalRegistry
from kohaku.harness.runner import ExternalHarnessRunner
from kohaku.providers.base import LLMResponse
from kohaku.remote.owner import RemoteAccessOwner
from kohaku.runtime_composition import RuntimeComposition
from kohaku.session.webui_turns import WebuiTurnCoordinator
from kohaku.webui.gateway_services import build_gateway_services
from tests.agent.test_delegation_child_followups import _tool_call
from tests.agent.test_migration_engine import ScriptedProvider
from tests.harness.test_claude_sdk_adapter import check_tool, sdk

pytest_plugins = ("tests.harness.test_claude_sdk_adapter",)


async def _event(client, name, predicate=lambda _value: True):
    while True:
        payload = await client.recv_raw()
        if payload.get("event") == name and predicate(payload):
            return payload


@pytest.mark.parametrize("mobile_decision", [None, "approve", "deny"])
async def test_claude_child_approval_and_stop_survive_parent_reconnect(
    tmp_path, monkeypatch, unused_tcp_port, sdk_boundary, request, mobile_decision,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    config_path = runtime_dir / "config.json"
    config = Config()
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = "test-model"
    config.agents.defaults.dream.enabled = False
    config.gateway.heartbeat.enabled = False
    config.artifacts.enabled = False
    config.tools.restrict_to_workspace = True
    config.bind_source_path(config_path)
    config_path.write_text(config.model_dump_json(by_alias=True))
    monkeypatch.setattr(loader, "_current_config_path", config_path)
    provider = ScriptedProvider([
        _tool_call("plan_task", {"objective": "Plan the native fixture."}, "native"),
        LLMResponse(content="Native fixture plan."),
        LLMResponse(content="Plan received."),
        _tool_call("delegate_task", {
            "contract": "Implement the fixture and run a scoped test.",
            "harness": "claude_sdk", "write_scope": ["out.txt"], "background": True,
        }, "claude"),
        LLMResponse(content="The worker is running."),
        *([LLMResponse(content="The worker received the decision.")] if mobile_decision == "deny" else []),
    ])
    monkeypatch.setattr("kohaku.providers.factory.make_provider", lambda _config: provider)
    monkeypatch.setattr("kohaku.agent.delegation.claude_sdk_available", lambda: True)

    async def skip_title(*_args, **_kwargs):
        return None

    monkeypatch.setattr("kohaku.session.webui_turns.maybe_generate_webui_title", skip_title)
    testing = asyncio.Event()
    stopped = asyncio.Event()

    async def script(options):
        yield sdk.SystemMessage(subtype="init", data={"session_id": "claude-in-original-chat"})
        cleanup = {"command": "git clean -fd"}
        yield sdk.AssistantMessage(content=[sdk.ToolUseBlock(id="cleanup", name="Bash", input=cleanup)])
        yield sdk.StreamEvent(event={"type": "message_stop"})
        assert (await check_tool(options, "Bash", cleanup, "cleanup"))["permissionDecision"] == "ask"
        allowed = await options.can_use_tool("Bash", cleanup, SimpleNamespace(tool_use_id="cleanup"))
        if mobile_decision == "deny":
            assert isinstance(allowed, sdk.PermissionResultDeny), allowed
            yield sdk.UserMessage(content=[
                sdk.ToolResultBlock(tool_use_id="cleanup", content="Denied.", is_error=True),
            ])
            yield sdk.ResultMessage(
                session_id="claude-in-original-chat", is_error=False, terminal_reason=None, result="Denied.",
            )
            return
        assert isinstance(allowed, sdk.PermissionResultAllow), allowed
        assert allowed.updated_input.get("dangerouslyDisableSandbox") is not True
        yield sdk.UserMessage(content=[sdk.ToolResultBlock(tool_use_id="cleanup", content="Done.")])
        command = cleanup if mobile_decision else {"command": "pytest -q"}
        yield sdk.AssistantMessage(content=[sdk.ToolUseBlock(id="tests", name="Bash", input=command)])
        yield sdk.StreamEvent(event={"type": "message_stop"})
        assert (await check_tool(options, "Bash", command, "tests"))["permissionDecision"] == (
            "ask" if mobile_decision else "allow"
        )
        testing.set()
        try:
            if mobile_decision:
                await options.can_use_tool("Bash", command, SimpleNamespace(tool_use_id="tests"))
            else:
                await asyncio.Event().wait()
        finally:
            stopped.set()

    sdk.script = script
    composition = RuntimeComposition.define(
        config, sessions_root=runtime_dir / "sessions",
    )
    handle = await composition.acquire()
    await handle.agent.external_harness.shutdown()
    handle.agent.external_harness = ExternalHarnessRunner(sdk_boundary[0], approvals=ExternalApprovalRegistry())
    ws_config = WebSocketConfig(enabled=True, allow_from=["*"], port=unused_tcp_port)
    remote_access = None
    mobile_auth = None
    if mobile_decision:
        owner = RemoteAccessOwner(tmp_path / "owner")
        owner.start()
        request.addfinalizer(owner.stop)
        owner.store.set_enabled(True)
        remote_access = RemoteAccessConfig(
            enabled=True, state_dir=str(owner.store.root), socket_path=str(owner.socket_path),
        )
        pair = owner.tokens.issue_device(
            "phone-owner", ("owner", "chat.read", "chat.write"),
            device_fingerprint="a" * 64, device_generation=1,
            origin=f"https://127.0.0.1:{unused_tcp_port}",
        )
        mobile_auth = {"Authorization": f"Bearer {pair.access_token}"}
    gateway = build_gateway_services(
        config=ws_config, bus=handle.bus, session_manager=handle.sessions,
        workspace_path=workspace, default_restrict_to_workspace=True, config_path=config_path,
        runtime_surface="browser", runtime_capabilities_overrides=None, runtime_model_name=None,
        stop_target_for_key=handle.agent.stop_target_for_key,
        child_control=handle.agent.control_delegated_child,
        remote_access=remote_access,
    )
    channel = WebSocketChannel(ws_config, handle.bus, gateway=gateway)
    coordinator = WebuiTurnCoordinator(
        bus=handle.bus, sessions=handle.sessions, schedule_background=handle.agent.schedule_background,
    )
    unsubscribe = coordinator.subscribe()

    async def deliver_outbound():
        while True:
            await channel.send(await handle.bus.consume_outbound())

    server = asyncio.create_task(channel.start())
    agent = asyncio.create_task(handle.agent.run())
    outbound = asyncio.create_task(deliver_outbound())
    ws_path = "/mobile/v1/ws" if mobile_auth else ws_config.path
    url = f"ws://127.0.0.1:{unused_tcp_port}{ws_path}"
    try:
        token = "" if mobile_auth else gateway.tokens.issue_token(60, audience="webui")
        api_token = gateway.tokens.issue_api_token(60)
        async with asyncio.timeout(12), WsTestClient(url, token=token, extra_headers=mobile_auth) as client:
            await client.recv_ready()
            await client.send_json({"type": "new_chat"})
            attached = await _event(client, "attached")
            assert attached.get("harness_binding") is None
            chat_id = attached["chat_id"]
            await client.send_json({
                "type": "message", "chat_id": chat_id, "content": "Plan the fixture.",
                "webui": True, "turn_id": "plan-native",
            })
            native = await _event(client, "delegated_child")
            assert native.get("harness", "native") == "native"
            await _event(client, "turn_end", lambda frame: frame.get("turn_id") == "plan-native")

            await client.send_json({
                "type": "message", "chat_id": chat_id, "content": "Implement the fixture.",
                "webui": True, "turn_id": "start-claude",
            })
            child = await _event(client, "delegated_child", lambda frame: frame.get("harness") == "claude_sdk")
            assert child["parent_session_key"] == f"websocket:{chat_id}"
            turn_ended = False
            pending = None
            while pending is None or not turn_ended:
                frame = await client.recv_raw()
                if frame.get("event") == "turn_end" and frame.get("turn_id") == "start-claude":
                    turn_ended = True
                if frame.get("event") == "external_approval":
                    pending = frame
            assert pending["child_id"] == child["child_id"]
            assert pending["approval"]["session_key"] == child["child_session_key"]
            assert pending["approval"]["command"] == "git clean -fd"
            assert pending["approval"]["save_exclusion_command"] is None

        token = "" if mobile_auth else gateway.tokens.issue_token(60, audience="webui")
        async with asyncio.timeout(12), WsTestClient(url, token=token, extra_headers=mobile_auth) as client:
            await client.recv_ready()
            await client.send_json({"type": "attach", "chat_id": chat_id})
            await _event(client, "attached")
            replayed = await _event(client, "external_approval")
            assert replayed == pending
            api_path = "/mobile/v1/sessions" if mobile_auth else "/api/sessions"
            base = f"http://127.0.0.1:{unused_tcp_port}{api_path}"
            auth = mobile_auth or {"Authorization": f"Bearer {api_token}"}
            parent_key = quote(f"websocket:{chat_id}", safe="")
            response = await http_get(f"{base}/{parent_key}/delegated-children", headers=auth)
            response.raise_for_status()
            activity = response.json()
            children = {row["child_id"]: row.get("harness", "native") for row in activity["children"]}
            assert children == {native["child_id"]: "native", child["child_id"]: "claude_sdk"}
            assert activity["stop_target"]["children"][0]["child_id"] == child["child_id"]
            if mobile_auth:
                controls = await http_get(f"{base}/{parent_key}/thread?view=controls", headers=auth)
                controls.raise_for_status()
                assert controls.json()["stop_target"] == activity["stop_target"]
            child_key = quote(child["child_session_key"], safe="")
            thread_route = "thread" if mobile_auth else "webui-thread"
            response = await http_get(f"{base}/{child_key}/{thread_route}", headers=auth)
            response.raise_for_status()
            assert response.json()["harness"] == "claude_sdk"

            identity = {
                field: replayed["approval"][field]
                for field in ("session_key", "session_incarnation", "run_id", "generation", "effect_id")
            }
            await client.send_json({
                "type": "message", "chat_id": chat_id, "webui": True, "turn_id": "allow-child",
                "content": "approval_decision", "approval_decision": {
                    "approval_id": replayed["approval_id"], "decision": mobile_decision or "approve", **identity,
                },
            })
            closed = await _event(client, "external_approval")
            assert closed["approval"] is None
            if mobile_decision == "deny":
                await _event(client, "delegated_child", lambda frame: (
                    frame["child_id"] == child["child_id"] and frame["state"] == "completed"
                ))
                assert not config.tools.exec.sandbox_excluded_commands
                return
            await testing.wait()
            if mobile_decision:
                next_approval = await _event(client, "external_approval")
                assert next_approval["approval"]["effect_id"] == "tests"
                assert next_approval["approval"]["save_exclusion_command"] is None

            await client.send_json({"type": "attach", "chat_id": chat_id})
            attached = await _event(client, "attached")
            assert attached["stop_target"]["children"]
            if mobile_decision:
                assert await _event(client, "external_approval") == next_approval
            await client.send_json({
                "type": "message", "chat_id": chat_id, "content": "/stop",
                "webui": True, "turn_id": "stop-original-chat", "stop_expectation": attached["stop_target"],
            })
            await stopped.wait()
            if mobile_decision:
                cleared = await _event(client, "external_approval")
                assert cleared["approval_id"] == next_approval["approval_id"]
                assert cleared["approval"] is None
            settled = await _event(client, "delegated_child", lambda frame: (
                frame["child_id"] == child["child_id"] and frame["state"] in {"stopped", "uncertain"}
            ))
            assert settled["harness"] == "claude_sdk"
            assert len(sdk.clients) == 1
            assert not Config.model_validate_json(config_path.read_text()).tools.exec.sandbox_excluded_commands
    finally:
        unsubscribe()
        for task in (agent, outbound):
            task.cancel()
        await asyncio.gather(agent, outbound, return_exceptions=True)
        await channel.stop()
        await asyncio.gather(server, return_exceptions=True)
        await composition.aclose()
