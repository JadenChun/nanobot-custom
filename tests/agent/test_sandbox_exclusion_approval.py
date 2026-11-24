"""Exercise Sandbox exception approval through canonical turn admission."""

import asyncio
import json

import pytest
from pydantic import ValidationError

from kohaku.agent.loop import AgentLoop
from kohaku.agent.permission_rules import (
    extract_saveable_sandbox_command,
    matches_sandbox_excluded_command,
)
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.agent.tools.shell import ExecToolConfig
from kohaku.agent.turn import ApprovalDecision, DeliveryTarget, TurnRequest
from kohaku.config.loader import save_config
from kohaku.providers.base import LLMResponse, ToolCallRequest
from tests.agent import test_migration_engine as engine_fixtures
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade

engine_config = engine_fixtures.engine_config
_spawn = asyncio.create_subprocess_exec
KEY = "websocket:sandbox-exclusion"
ROUTE = DeliveryTarget(channel="websocket", chat_id="sandbox-exclusion")


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("printf kohaku-fixture", "printf"),
        ("PrInTf kohaku-fixture", "printf"),
        ("printf one && printf two", None),
        ("docker ps || docker version", None),
        ("docker ps; docker version", None),
        ("docker ps | cat", None),
        ("docker ps &", None),
        ("docker ps\ndocker version", None),
        ("printf one > fixture.txt", None),
        ("docker load < image.tar", None),
        ("printf $(whoami)", None),
        ("printf `whoami`", None),
        ("bash -c 'printf fixture'", None),
        ("python3 -c 'print(\"fixture\")'", None),
        ("node -e 'console.log(\"fixture\")'", None),
        ("env printf fixture", None),
        ("sudo printf fixture", None),
        ("git push origin example", None),
        ("rm -rf fixture", None),
        ("npm publish", None),
    ],
)
def test_extract_saveable_sandbox_command(command, expected):
    assert extract_saveable_sandbox_command(command) == expected


@pytest.mark.parametrize(
    ("command", "excluded", "expected"),
    [
        ("printf kohaku-fixture", ["printf"], True),
        ("printf-extra kohaku-fixture", ["printf"], False),
        ("printf kohaku-fixture", ["printf-extra"], False),
        ("git push origin example", ["git"], False),
        ("docker ps && rm -rf /", ["docker"], False),
    ],
)
def test_match_sandbox_exclusion_requires_a_full_safe_command(command, excluded, expected):
    assert matches_sandbox_excluded_command(command, excluded) is expected


def test_sandbox_exclusions_config_normalizes_bare_names():
    assert ExecToolConfig().sandbox_excluded_commands == []
    config = ExecToolConfig.model_validate({"sandboxExcludedCommands": [" Docker ", "Git"]})
    assert config.sandbox_excluded_commands == ["docker", "git"]
    for invalid in ("/usr/bin/git", r"bin\git", "git push", "git;push"):
        with pytest.raises(ValidationError, match="bare executable"):
            ExecToolConfig.model_validate({"sandboxExcludedCommands": [invalid]})


@pytest.mark.parametrize("decision", ["approve", "approve_and_save", "deny"])
def test_parse_approval_decision(decision):
    assert ApprovalDecision.from_mapping({
        "approval_id": "fixture-approval", "decision": decision,
    }).decision == decision


def command_response(identifier, command, *, unsandboxed=False):
    return LLMResponse(content=None, tool_calls=[ToolCallRequest(
        id=identifier, name="exec", arguments={"command": command, "unsandboxed": unsandboxed},
    )])


async def run_turn(loop, decision=None):
    return await loop.execute_turn(TurnRequest(
        content="approval_decision" if decision else "Run the fixture command.",
        session_key=KEY, route=ROUTE, approval_decision=decision,
    ))


@pytest.fixture
def sandbox_loop(engine_config, monkeypatch):
    launched = []

    async def launch_fixture(*args, **kwargs):
        # Allow only the inert fixture commands to reach a real process.
        assert args[-1] in {"printf kohaku-fixture", "printf subsequent-fixture"}, args
        launched.append(args[-1])
        return await _spawn(*args, **kwargs)

    monkeypatch.setattr("asyncio.create_subprocess_exec", launch_fixture)
    monkeypatch.setattr("kohaku.agent.tools.sandbox.managed_bwrap_available", lambda: False)
    monkeypatch.setattr("kohaku.agent.tools.shell.managed_bwrap_available", lambda: False)
    config_path = engine_config.runtime_data_dir / "config.json"

    def create(responses, exclusions=()):
        engine_config.tools.permissions.profile = "autonomous"
        engine_config.tools.exec.sandbox_excluded_commands.extend(exclusions)
        save_config(engine_config, config_path)
        provider = ScriptedProvider(responses)
        loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
        upgrade(loop, KEY)
        return loop, provider, config_path, launched

    return create


@pytest.mark.parametrize("decision", ["approve", "approve_and_save"])
async def test_sandbox_exception_resumes_and_only_save_updates_live_config(sandbox_loop, decision):
    responses = [
        command_response("blocked", "printf kohaku-fixture", unsandboxed=True),
        command_response("resumed", "printf kohaku-fixture", unsandboxed=True),
        LLMResponse(content="done"),
        command_response("subsequent", "printf subsequent-fixture"),
        LLMResponse(content="done"),
    ]
    loop, provider, config_path, launched = sandbox_loop(responses)
    initial = config_path.read_bytes()
    try:
        blocked = await run_turn(loop)
        assert blocked.stop_reason == "approval_required"
        assert blocked.policy_metadata["save_exclusion_command"] == "printf"
        assert not launched

        resumed = await run_turn(loop, ApprovalDecision.from_mapping({
            "approval_id": blocked.policy_metadata["approval_id"],
            "decision": decision,
            "command": "printf client-fixture",
            "save_exclusion_command": "client-fixture",
        }))
        assert resumed.stop_reason == "completed", resumed
        assert launched == ["printf kohaku-fixture"]
        assert any(
            row.get("role") == "tool" and "kohaku-fixture" in str(row.get("content"))
            for row in provider.calls[2][0]
        )
        if decision == "approve_and_save":
            assert json.loads(config_path.read_bytes())["tools"]["exec"]["sandboxExcludedCommands"] == [
                "printf",
            ]
            assert (await run_turn(loop)).stop_reason == "completed"
            assert launched == ["printf kohaku-fixture", "printf subsequent-fixture"]
        else:
            assert config_path.read_bytes() == initial
            assert (await run_turn(loop)).stop_reason == "completed"
            assert launched == ["printf kohaku-fixture"]
            assert any(
                row.get("role") == "tool" and "Sandbox command confinement is unavailable" in str(row.get("content"))
                for row in provider.calls[4][0]
            )
    finally:
        await loop.aclose()


@pytest.mark.parametrize(
    ("command", "unsandboxed"),
    [("bash -c 'printf fixture'", True), ("git push origin example", False)],
)
async def test_ineligible_save_preserves_pending_approval_and_config(
    sandbox_loop, command, unsandboxed,
):
    loop, provider, config_path, launched = sandbox_loop(
        [command_response("blocked", command, unsandboxed=unsandboxed)], exclusions=["git"],
    )
    initial = config_path.read_bytes()
    try:
        blocked = await run_turn(loop)
        assert blocked.stop_reason == "approval_required"
        assert not blocked.policy_metadata.get("save_exclusion_command")
        approval_id = blocked.policy_metadata["approval_id"]
        refused = await run_turn(loop, ApprovalDecision(approval_id, "approve_and_save"))
        assert refused.stop_reason == "invalid_approval_decision"
        assert config_path.read_bytes() == initial
        assert not launched
        assert len(provider.calls) == 1
        assert (await run_turn(loop, ApprovalDecision(approval_id, "deny"))).stop_reason == "cancelled"
    finally:
        await loop.aclose()
