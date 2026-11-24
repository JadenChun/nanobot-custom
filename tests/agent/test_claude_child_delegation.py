"""Exercise public role tools through canonical turns with a simulated Claude SDK."""

import json
from copy import deepcopy

import pytest

from kohaku.agent.loop import AgentLoop
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.agent.turn import DeliveryTarget, TurnRequest
from kohaku.harness.runner import ExternalHarnessRunner
from kohaku.providers.base import LLMResponse
from tests.agent.test_delegation_child_followups import _tool_call
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_source_engine import upgrade
from tests.harness.test_claude_sdk_adapter import check_tool, sdk

pytest_plugins = (
    "tests.agent.test_migration_source_engine",
    "tests.harness.test_claude_sdk_adapter",
)

PARENT = "websocket:claude-role"
EXPLORE = (
    '---EXPLORE---\n{"summary":"Observed fixture","findings":["A file"],'
    '"references":[],"open_questions":[],"searched_areas":[],"partial":false}\n---END---'
)
ROLES = [
    ("explore", {"task": "Read the fixture source."}),
    ("plan_task", {"objective": "Plan the fixture implementation."}),
    ("review_work", {"goal": "Review the fixture", "acceptance_criteria": "Check fixture results."}),
    ("delegate_task", {"contract": "Implement the fixture and run the checks.", "write_scope": ["out.txt"]}),
]


def _saved_children(loop, parent=PARENT):
    result = {}
    for row in loop.sessions.list_sessions():
        key = row["key"]
        if not key.startswith("delegated:"):
            continue
        metadata = loop.sessions.read_session_metadata(key)["metadata"]
        child = metadata["delegation_child"]
        if child["parent_key"] == parent:
            result[child["child_id"]] = metadata
    return result


@pytest.fixture
async def role_gateway(engine_config, sdk_boundary, monkeypatch):
    provider = ScriptedProvider([])
    loop = AgentLoop.from_config(engine_config, tool_registry=ToolRegistry(), provider=provider)
    await loop.external_harness.shutdown()
    backend, _ = sdk_boundary
    loop.external_harness = ExternalHarnessRunner(backend)
    monkeypatch.setattr("kohaku.agent.delegation.claude_sdk_available", lambda: True)
    upgrade(loop, PARENT)

    async def invoke(name, args, *, parent=PARENT, workspace=None, native_result=None):
        call = f"public-call-{len(provider.calls)}"
        provider.responses.append(_tool_call(name, args, call))
        if native_result is not None:
            provider.responses.append(LLMResponse(content=native_result))
        provider.responses.append(LLMResponse(content="The original chat received the result."))
        result = await loop.execute_turn(TurnRequest(
            content="Work on the disposable fixture.", session_key=parent,
            route=DeliveryTarget(channel="websocket", chat_id=parent.split(":")[1]),
            metadata={"workspace_scope": {"project_path": str(workspace), "access_mode": "restricted"}}
            if workspace is not None else {},
        ))
        assert result.error is None, result.error
        source = loop.sessions.source_backend(parent).source.load_source_snapshot(parent)
        tool_result = next(row for row in source.events if row.get("tool_call_id") == call)
        return str(tool_result["content"])

    try:
        yield loop, invoke
    finally:
        await loop.aclose()


@pytest.mark.parametrize(("tool", "arguments"), ROLES)
@pytest.mark.parametrize("harness", ["native", "claude_sdk"])
async def test_harness_selection_keeps_role_authority(role_gateway, tool, arguments, harness):
    loop, invoke = role_gateway
    answer = EXPLORE if tool == "explore" else "Observed fixture."

    async def script(options):
        assert options.resume is None and not options.fork_session
        assert "Read" in options.tools
        if tool == "delegate_task":
            assert {"Bash", "Write", "Edit"} <= set(options.tools)
            assert json.loads(options.settings)["sandbox"]["enabled"] is True
        else:
            assert {"Bash", "Write", "Edit", "NotebookEdit"}.isdisjoint(options.tools)
        yield sdk.ResultMessage(session_id="role-context", is_error=False, terminal_reason=None, result=answer)

    sdk.script = script
    if harness == "claude_sdk":
        response = await invoke(tool, {**arguments, "harness": harness})
        assert len(sdk.clients) == 1
    else:
        response = await invoke(tool, arguments, native_result=answer)
        assert sdk.clients == []
    assert "Observed fixture" in response
    [saved] = _saved_children(loop).values()
    assert saved["delegation_child"]["state"] == "completed"
    assert saved["delegation_child"].get("harness", "native") == harness


async def test_scoped_worker_forks_read_only_context_and_followup_resumes(role_gateway):
    loop, invoke = role_gateway
    workspace = loop.workspace
    out = workspace / "out.txt"
    denied = workspace / "unchanged.txt"
    denied.write_text("original")

    async def script(options):
        call = len(sdk.clients)
        if call == 1:
            assert {"Bash", "Write", "Edit", "NotebookEdit"}.isdisjoint(options.tools)
            yield sdk.ResultMessage(
                session_id="read-only-context", is_error=False, terminal_reason=None, result=EXPLORE,
            )
            return
        assert options.resume == ("read-only-context" if call == 2 else "worker-context")
        assert options.fork_session is (call == 2)
        assert options.cwd == workspace
        sandbox = json.loads(options.settings)["sandbox"]
        assert sandbox["enabled"] and not sandbox["allowUnsandboxedCommands"]
        for path in ("unchanged.txt", "out.txt/nested"):
            assert (await check_tool(options, "Write", {"file_path": path}))["permissionDecision"] == "deny"
        yield sdk.SystemMessage(subtype="init", data={"session_id": "worker-context"})
        arguments = {"file_path": "out.txt", "content": f"change {call}"}
        yield sdk.AssistantMessage(content=[sdk.ToolUseBlock(id=f"out-{call}", name="Write", input=arguments)])
        assert (await check_tool(options, "Write", arguments, f"out-{call}"))["permissionDecision"] == "allow"
        out.write_text(arguments["content"])
        [post] = options.hooks["PostToolUse"][0].hooks
        await post({"tool_use_id": f"out-{call}"}, f"out-{call}", {})
        yield sdk.UserMessage(content=[sdk.ToolResultBlock(tool_use_id=f"out-{call}", content="written")])
        command = {"command": "pytest -q"}
        yield sdk.AssistantMessage(content=[sdk.ToolUseBlock(id=f"test-{call}", name="Bash", input=command)])
        assert (await check_tool(options, "Bash", command, f"test-{call}"))["permissionDecision"] == "allow"
        yield sdk.UserMessage(content=[sdk.ToolResultBlock(tool_use_id=f"test-{call}", content="1 passed")])
        yield sdk.ResultMessage(
            session_id="worker-context", is_error=False, terminal_reason=None, result="Implementation report.",
        )

    sdk.script = script
    await invoke("explore", {"task": "Read the plan.", "harness": "claude_sdk"})
    [(prior_id, original)] = _saved_children(loop).items()
    original = deepcopy(original)
    report = await invoke("delegate_task", {
        "contract": "Apply the agreed fixture plan and verify the file.",
        "harness": "claude_sdk", "write_scope": ["out.txt"], "continue_from_child_id": prior_id,
    })
    assert "Observed changed files" in report and "out.txt" in report and "ok: pytest -q" in report
    saved = _saved_children(loop)
    assert saved.pop(prior_id) == original
    [(worker_id, worker)] = saved.items()
    assert worker_id != prior_id and worker["harness_binding"]["external_session_id"] == "worker-context"
    assert denied.read_text() == "original"

    disposition = await invoke("followup_child", {"child_id": worker_id, "content": "Check the scope again."})
    assert "Follow-up applied" in disposition
    assert len(sdk.clients) == 3 and out.read_text() == "change 3"
    updated = _saved_children(loop)
    assert set(updated) == {prior_id, worker_id}
    assert updated[worker_id]["harness_binding"]["external_session_id"] == "worker-context"
    assert updated[worker_id]["delegation_child"]["policy_facts"] == worker["delegation_child"]["policy_facts"]
    assert updated[prior_id] == original


@pytest.mark.parametrize("mismatch", ["parent", "workspace", "native"])
async def test_context_carryover_refuses_stale_authority_before_launch(role_gateway, mismatch):
    loop, invoke = role_gateway

    async def script(_options):
        yield sdk.ResultMessage(session_id="prior", is_error=False, terminal_reason=None, result=EXPLORE)

    sdk.script = script
    alternate_parent = "websocket:different-parent"
    source_parent = alternate_parent if mismatch == "parent" else PARENT
    if source_parent != PARENT:
        upgrade(loop, source_parent)
    await invoke(
        "explore", {"task": "Investigate the fixture.", **({} if mismatch == "native" else {"harness": "claude_sdk"})},
        parent=source_parent, native_result=EXPLORE if mismatch == "native" else None,
    )
    [prior_id] = _saved_children(loop, source_parent)
    calls = len(sdk.clients)
    before = _saved_children(loop)
    alternate = loop.workspace / "other"
    alternate.mkdir()
    report = await invoke("delegate_task", {
        "contract": "Implement the bounded fixture task.",
        "harness": "claude_sdk", "write_scope": ["out.txt"], "continue_from_child_id": prior_id,
    }, workspace=alternate if mismatch == "workspace" else None)
    assert "Choose a completed Claude Code child from this chat and working folder." in report
    assert len(sdk.clients) == calls and _saved_children(loop) == before
