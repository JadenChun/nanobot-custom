"""Tests for standardized ToolPermissionsConfig profiles, rules, and non-expiring approvals."""

from __future__ import annotations

from pathlib import Path

import pytest

from kohaku.agent.policy import RiskyActionPolicy
from kohaku.agent.tools.filesystem import EditFileTool, WriteFileTool
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.agent.turn import ApprovalDecision, TurnContext
from kohaku.agent.turn_admission import APPROVAL_TTL_SECONDS, ApprovalAdmission
from kohaku.config.schema import Config, ToolPermissionsConfig
from kohaku.providers.base import ToolCallRequest


def test_default_config_uses_autonomous_permission_profile() -> None:
    cfg = Config()
    assert cfg.tools.permissions.profile == "autonomous"
    assert cfg.tools.permissions.rules == {}


async def test_autonomous_profile_allows_large_workspace_writes_and_edits(tmp_path: Path) -> None:
    doc = tmp_path / "storyboard_v3.md"
    doc.write_text("\n".join(f"old line {i}" for i in range(40)) + "\n", encoding="utf-8")
    tools = ToolRegistry()
    tools.register(WriteFileTool(workspace=tmp_path, allowed_dir=tmp_path))
    tools.register(EditFileTool(workspace=tmp_path, allowed_dir=tmp_path))

    policy = RiskyActionPolicy(
        workspace=tmp_path,
        permissions=ToolPermissionsConfig(profile="autonomous"),
    ).for_tools(tools)

    write_call = ToolCallRequest(
        id="w1",
        name="write_file",
        arguments={"path": "storyboard_v3.md", "content": "brand new storyboard\n"},
    )
    decision = await policy.evaluate(messages=[], tool_calls=[write_call])
    assert decision.action == "allow"
    assert decision.requires_approval is False

    # Write-target binding (TOCTOU guard) is still populated in autonomous mode.
    error, target = policy.validate_write_target_binding(write_call, tools.get("write_file"))
    assert error is None
    assert target == doc.resolve()

    edit_call = ToolCallRequest(
        id="e1",
        name="edit_file",
        arguments={
            "path": "storyboard_v3.md",
            "old_text": "old line",
            "new_text": "new line",
            "replace_all": True,
        },
    )
    edit_decision = await policy.evaluate(messages=[], tool_calls=[edit_call])
    assert edit_decision.action == "allow"
    assert edit_decision.requires_approval is False


async def test_autonomous_profile_still_refuses_unconfined_nested_agent_launch(
    tmp_path: Path,
) -> None:
    policy = RiskyActionPolicy(
        workspace=tmp_path,
        permissions=ToolPermissionsConfig(profile="autonomous", rules={"exec": "allow"}),
    )
    launch = ToolCallRequest(
        id="nested",
        name="exec",
        arguments={"command": "codex --yolo 'do work'"},
    )
    decision = await policy.evaluate(messages=[], tool_calls=[launch])
    assert decision.action == "respond"
    assert decision.stop_reason == "policy_blocked"
    assert decision.requires_approval is False


async def test_balanced_profile_allows_local_work_and_asks_on_external_mutations(
    tmp_path: Path,
) -> None:
    doc = tmp_path / "notes.md"
    doc.write_text("\n".join(f"line {i}" for i in range(30)) + "\n", encoding="utf-8")
    policy = RiskyActionPolicy(
        workspace=tmp_path,
        permissions=ToolPermissionsConfig(profile="balanced"),
    )

    # Local overwrite, local rm -rf, and cron are autonomous in balanced mode.
    for call in (
        ToolCallRequest(id="w1", name="write_file", arguments={"path": "notes.md", "content": "new\n"}),
        ToolCallRequest(id="rm1", name="exec", arguments={"command": "rm -rf build/"}),
        ToolCallRequest(id="c1", name="cron", arguments={"action": "add"}),
    ):
        dec = await policy.evaluate(messages=[], tool_calls=[call])
        assert dec.action == "allow", call.name

    # External push and desktop input still require approval in balanced mode.
    push = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(id="p1", name="exec", arguments={"command": "git push origin main"})],
    )
    assert push.stop_reason == "approval_required"

    desktop = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(id="d1", name="desktop_use", arguments={"action": "left_click"})],
    )
    assert desktop.stop_reason == "approval_required"


async def test_declarative_rules_support_allow_ask_deny_last_match_wins(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    secret_file = secrets / "prod.env"
    secret_file.write_text("API_KEY=1\n", encoding="utf-8")

    policy = RiskyActionPolicy(
        workspace=tmp_path,
        permissions=ToolPermissionsConfig(
            profile="autonomous",
            rules={
                "exec": {
                    "*": "allow",
                    "git push*": "ask",
                    "rm -rf /*": "deny",
                },
                "write_file": {
                    "*": "allow",
                    "secrets/*": "deny",
                },
            },
        ),
    )

    ok_exec = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(id="ok", name="exec", arguments={"command": "pytest -q"})],
    )
    assert ok_exec.action == "allow"

    ask_exec = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(id="push", name="exec", arguments={"command": "git push origin main"})],
    )
    assert ask_exec.stop_reason == "approval_required"

    deny_exec = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(id="wipe", name="exec", arguments={"command": "rm -rf /tmp/x"})],
    )
    assert deny_exec.stop_reason == "policy_blocked"
    assert deny_exec.requires_approval is False

    deny_write = await policy.evaluate(
        messages=[],
        tool_calls=[ToolCallRequest(
            id="sec",
            name="write_file",
            arguments={"path": "secrets/prod.env", "content": "overwritten"},
        )],
    )
    assert deny_write.stop_reason == "policy_blocked"
    assert deny_write.requires_approval is False


async def test_pending_approval_does_not_expire_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert APPROVAL_TTL_SECONDS == 0.0
    admission = ApprovalAdmission()
    pending = await admission.store(
        session_key="websocket:chat-1",
        channel="websocket",
        chat_id="chat-1",
        summary="push commits",
        effect_id="a" * 64,
        run_id="blocked-run",
    )
    assert pending.expires_at == 0.0

    # Simulate the user returning 24 hours later.
    later = pending.created_at + 24 * 3600
    monkeypatch.setattr("kohaku.agent.turn_admission.time.time", lambda: later)

    decision = ApprovalDecision(approval_id=pending.approval_id, decision="approve")
    snapshot = await admission.snapshot_decision("websocket:chat-1", decision)
    assert snapshot.action == "affirmative"
    assert snapshot.pending is pending

    context = TurnContext(run_id="resume-run", session_key="websocket:chat-1")
    consumed = await admission.consume(snapshot, context=context)
    assert consumed == "affirmative"
    assert context.policy.effect_authority is not None
    assert context.policy.effect_authority.permits("a" * 64) is True
