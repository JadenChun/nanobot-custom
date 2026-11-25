"""Exercise loopback exemption and exact-effect approval for private-network URLs in exec."""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from kohaku.agent.hook import AgentHook, AgentHookContext
from kohaku.agent.loop import AgentLoop
from kohaku.agent.policy import CompositeToolPolicy, RiskyActionPolicy, ToolPolicy
from kohaku.agent.tools.execution import execute_tool_calls
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.agent.tools.shell import ExecTool
from kohaku.agent.turn import (
    ApprovalDecision,
    DeliveryTarget,
    ToolOutcome,
    TurnContext,
    TurnRequest,
)
from kohaku.agent.turn_admission import ApprovalAdmission
from kohaku.config.loader import save_config
from kohaku.config.schema import ToolPermissionsConfig
from kohaku.providers.base import LLMResponse, ToolCallRequest
from kohaku.security.workspace_access import (
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
)
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade

engine_config = engine_fixtures.engine_config
_spawn = asyncio.create_subprocess_exec
KEY = "websocket:private-network-approval"
ROUTE = DeliveryTarget(channel="websocket", chat_id="private-network-approval")


def _fake_resolve_ip(ip: str):
    def _resolve(hostname, port, family=0, type_=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]

    return _resolve


async def _dispatch(
    root: Path,
    tool: ExecTool,
    call: ToolCallRequest,
    *,
    context: TurnContext | None = None,
    permissions: ToolPermissionsConfig | None = None,
) -> tuple[str | ToolOutcome, str]:
    registry = ToolRegistry()
    registry.register(tool)
    risky = RiskyActionPolicy(
        workspace=root,
        effect_authority=context.policy.effect_authority if context else None,
        permissions=permissions or ToolPermissionsConfig(profile="autonomous"),
    )
    policy = CompositeToolPolicy((ToolPolicy(), risky))
    captured = registry.capture_request_tools()
    try:
        results, events = await execute_tool_calls(
            captured,
            [call],
            concurrent=False,
            external_lookup_counts={},
            workspace_violation_counts={},
            hook=AgentHook(),
            context=AgentHookContext(iteration=0, messages=[]),
            tool_policy=policy,
            turn_context=context,
            fail_on_tool_error=True,
        )
    finally:
        captured.release()
    return results[0], events[0]["status"]


async def _issue_approval(result: ToolOutcome) -> tuple[ApprovalAdmission, TurnContext]:
    assert result.stop_reason == "approval_required"
    admission = ApprovalAdmission()
    pending = await admission.store(
        session_key="websocket:fixture",
        channel="websocket",
        chat_id="fixture",
        summary=str(result.policy_metadata.get("summary") or "private network"),
        effect_id=result.policy_metadata["effect_id"],
    )
    context = TurnContext(run_id="resume", session_key="websocket:fixture")
    snapshot = await admission.snapshot_decision(
        "websocket:fixture",
        ApprovalDecision(approval_id=pending.approval_id, decision="approve"),
    )
    assert await admission.consume(snapshot, context=context) == "affirmative"
    return admission, context


@pytest.mark.parametrize("access_mode", ["restricted", "full"])
async def test_webui_sandbox_and_full_access_allow_literal_loopback_without_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, access_mode: str,
) -> None:
    monkeypatch.setattr("kohaku.security.network.socket.getaddrinfo", _fake_resolve_ip("127.0.0.1"))
    tool = ExecTool(working_dir=str(tmp_path), restrict_to_workspace=True)
    call = ToolCallRequest(
        id="loopback",
        name="exec",
        arguments={"command": "printf 'http://127.0.0.1:8084/v1/models'"},
    )
    token = bind_workspace_scope(
        build_workspace_scope(tmp_path, access_mode, source_channel="websocket")
    )
    try:
        result, status = await _dispatch(tmp_path, tool, call)
    finally:
        reset_workspace_scope(token)
    assert status == "ok", result
    assert "http://127.0.0.1:8084/v1/models" in str(result)


async def test_private_lan_url_requests_exact_approval_and_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("kohaku.security.network.socket.getaddrinfo", _fake_resolve_ip("192.168.1.50"))
    tool = ExecTool(working_dir=str(tmp_path), restrict_to_workspace=True)
    call = ToolCallRequest(
        id="lan",
        name="exec",
        arguments={"command": "printf 'http://192.168.1.50:8080/v1/models'"},
    )
    token = bind_workspace_scope(
        build_workspace_scope(tmp_path, "restricted", source_channel="websocket")
    )
    try:
        ask, status = await _dispatch(tmp_path, tool, call)
        assert status == "skipped"
        assert isinstance(ask, ToolOutcome)
        assert ask.stop_reason == "approval_required"
        assert "private network URL" in str(ask.content)

        admission, context = await _issue_approval(ask)
        try:
            # A different command cannot reuse the approval.
            other_call = ToolCallRequest(
                id="lan-other",
                name="exec",
                arguments={"command": "printf 'http://192.168.1.51:8080/v1/models'"},
            )
            other_ask, _ = await _dispatch(tmp_path, tool, other_call, context=context)
            assert isinstance(other_ask, ToolOutcome)
            assert other_ask.stop_reason == "approval_required"
            assert other_ask.policy_metadata["effect_id"] != ask.policy_metadata["effect_id"]

            # The approved exact command runs and consumes the single-use grant.
            result, status = await _dispatch(tmp_path, tool, call, context=context)
            assert status == "ok", result
            assert "http://192.168.1.50:8080/v1/models" in str(result)

            # Repeating the same command requires fresh approval.
            repeated, _ = await _dispatch(tmp_path, tool, call, context=context)
            assert isinstance(repeated, ToolOutcome)
            assert repeated.stop_reason == "approval_required"
        finally:
            admission.retire(context)
    finally:
        reset_workspace_scope(token)


async def test_cloud_metadata_url_never_requests_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "kohaku.security.network.socket.getaddrinfo",
        _fake_resolve_ip("169.254.169.254"),
    )
    tool = ExecTool(working_dir=str(tmp_path), restrict_to_workspace=False)
    spawn = AsyncMock(side_effect=AssertionError("metadata URL reached spawn"))
    monkeypatch.setattr(tool, "_spawn", spawn)
    call = ToolCallRequest(
        id="metadata",
        name="exec",
        arguments={"command": "curl -s http://169.254.169.254/latest/meta-data/"},
    )
    token = bind_workspace_scope(
        build_workspace_scope(tmp_path, "full", source_channel="websocket")
    )
    try:
        result, status = await _dispatch(tmp_path, tool, call)
    finally:
        reset_workspace_scope(token)
    assert status == "error"
    assert isinstance(result, ToolOutcome)
    assert result.stop_reason == "tool_error"
    assert "internal/private URL detected" in str(result.content)
    spawn.assert_not_awaited()


async def test_hard_denied_command_with_private_url_refuses_without_asking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("kohaku.security.network.socket.getaddrinfo", _fake_resolve_ip("192.168.1.50"))
    tool = ExecTool(working_dir=str(tmp_path), restrict_to_workspace=False)
    spawn = AsyncMock(side_effect=AssertionError("denied command reached spawn"))
    monkeypatch.setattr(tool, "_spawn", spawn)
    call = ToolCallRequest(
        id="denied",
        name="exec",
        arguments={"command": "rm -f scratch.txt && curl http://192.168.1.50:8080/"},
    )
    result, status = await _dispatch(tmp_path, tool, call)
    assert status == "error"
    assert isinstance(result, ToolOutcome)
    assert result.stop_reason == "tool_error"
    assert "deny pattern filter" in str(result.content)
    spawn.assert_not_awaited()


async def test_private_network_turn_asks_and_resumes_through_agent_loop(
    engine_config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[str] = []
    cmd = "printf 'http://192.168.1.50:8080/v1/models'"

    async def launch_fixture(*args, **kwargs):
        assert args[-1] == cmd, args
        launched.append(args[-1])
        return await _spawn(*args, **kwargs)

    monkeypatch.setattr("asyncio.create_subprocess_exec", launch_fixture)
    monkeypatch.setattr("kohaku.agent.tools.sandbox.managed_bwrap_available", lambda: False)
    monkeypatch.setattr("kohaku.agent.tools.shell.managed_bwrap_available", lambda: False)
    monkeypatch.setattr("kohaku.security.network.socket.getaddrinfo", _fake_resolve_ip("192.168.1.50"))

    config_path = engine_config.runtime_data_dir / "config.json"
    engine_config.tools.permissions.profile = "autonomous"
    save_config(engine_config, config_path)
    responses = [
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(
                id="ask",
                name="exec",
                arguments={"command": cmd, "unsandboxed": True},
            )],
        ),
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(
                id="run",
                name="exec",
                arguments={"command": cmd, "unsandboxed": True},
            )],
        ),
        LLMResponse(content="finished private network check"),
    ]
    provider = ScriptedProvider(responses)
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    upgrade(loop, KEY)
    try:
        blocked = await loop.execute_turn(TurnRequest(
            content="Check LAN service",
            session_key=KEY,
            route=ROUTE,
        ))
        assert blocked.stop_reason == "approval_required"
        assert "private network URL" in str(blocked.content)
        # Combined sandbox + private-network approval must not offer saving a sandbox exclusion.
        assert not blocked.policy_metadata.get("save_exclusion_command")
        assert not launched

        resumed = await loop.execute_turn(TurnRequest(
            content="approval_decision",
            session_key=KEY,
            route=ROUTE,
            approval_decision=ApprovalDecision(
                approval_id=blocked.policy_metadata["approval_id"],
                decision="approve",
            ),
        ))
        assert resumed.stop_reason == "completed", resumed
        assert resumed.content == "finished private network check"
        assert launched == [cmd]
    finally:
        await loop.aclose()
