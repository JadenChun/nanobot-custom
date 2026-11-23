"""Exercise preparation, approval and mutation through the real tool dispatcher."""

import asyncio
import subprocess
from pathlib import Path

import pytest

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.policy import CompositeToolPolicy, RiskyActionPolicy, ToolPolicy
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.execution import execute_tool_calls
from nanobot.agent.tools.filesystem import EditFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.turn import ApprovalDecision, ToolOutcome, TurnContext
from nanobot.agent.turn_admission import ApprovalAdmission
from nanobot.agent.write_guard import FileLockRegistry
from nanobot.context_repo import ContextRepoManager, ResourceAccessPolicy
from nanobot.providers.base import ToolCallRequest
from nanobot.security.folder_grants import FolderGrantStore
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    build_workspace_scope,
    reset_workspace_scope,
)


async def dispatch(
    root: Path,
    tool: Tool,
    call: ToolCallRequest,
    *,
    hook: AgentHook | None = None,
    context: TurnContext | None = None,
    composite: bool = False,
) -> tuple[str | ToolOutcome, str]:
    registry = ToolRegistry()
    registry.register(tool)
    risky = RiskyActionPolicy(
        workspace=root, effect_authority=context.policy.effect_authority if context else None,
    )
    policy = CompositeToolPolicy((ToolPolicy(), risky)) if composite else risky
    captured = registry.capture_request_tools()
    try:
        results, events = await execute_tool_calls(
            captured, [call], concurrent=False, external_lookup_counts={},
            workspace_violation_counts={}, hook=hook or AgentHook(),
            context=AgentHookContext(iteration=0, messages=[]), tool_policy=policy,
            turn_context=context, fail_on_tool_error=True,
        )
    finally:
        captured.release()
    return results[0], events[0]["status"]


def edit_call(old: str, new: str, **params: object) -> ToolCallRequest:
    return ToolCallRequest(id="edit", name="edit_file", arguments={
        "path": "sample.txt", "old_text": old, "new_text": new, **params,
    })


async def issue_approval(result: ToolOutcome) -> tuple[ApprovalAdmission, TurnContext]:
    assert result.stop_reason == "approval_required"
    admission = ApprovalAdmission()
    pending = await admission.store(
        session_key="cli:fixture", channel="cli", chat_id="fixture", summary="test effect",
        effect_id=result.policy_metadata["effect_id"],
    )
    context = TurnContext(run_id="resume", session_key="cli:fixture")
    snapshot = await admission.snapshot_decision(
        "cli:fixture", ApprovalDecision(approval_id=pending.approval_id, decision="approve"),
    )
    assert await admission.consume(snapshot, context=context) == "affirmative"
    return admission, context


@pytest.mark.parametrize("selection", [{}, {"occurrence": "2"}, {"line_hint": 102}])
async def test_edit_policy_scores_the_selected_edit_against_the_file(tmp_path, selection):
    editor = EditFileTool(workspace=tmp_path)
    target = tmp_path / "sample.txt"
    # A common suffix must not look edited just because the lines repeat.
    lines = ["  setting=“old”\r\n", *(["same context\r\n"] * 100)]
    if selection:
        lines.append("  setting=“old”\r\n")
    lines.extend(["same context\r\n"] * 300)
    target.write_bytes("".join(lines).encode())
    result, status = await dispatch(
        tmp_path, editor, edit_call('setting="old"', 'setting="new" ', **selection),
        composite=True,
    )
    assert status == "ok", result
    lines[101 if selection else 0] = "  setting=“new”\r\n"
    assert target.read_bytes() == "".join(lines).encode()


async def test_unique_one_line_edit_is_routine(tmp_path):
    target = tmp_path / "sample.txt"
    before = "".join(f"setting_{index:03}=true\n" for index in range(100))
    target.write_text(before)
    result, status = await dispatch(
        tmp_path, EditFileTool(workspace=tmp_path), edit_call("setting_000=true", "setting_000=false"),
    )
    assert status == "ok", result
    assert target.read_text() == before.replace("setting_000=true", "setting_000=false")


@pytest.mark.parametrize(
    ("params", "feedback"),
    [({}, "old_text appears 2 times"), ({"replace_all": True, "expected_replacements": 1},
                                     "expected 1 replacements but would make 2")],
)
async def test_invalid_edit_reports_validation_not_approval(tmp_path, params, feedback):
    target = tmp_path / "sample.txt"
    before = "same\nother\nsame\n"
    target.write_text(before)
    result, _ = await dispatch(
        tmp_path, EditFileTool(workspace=tmp_path), edit_call("same", "new", **params),
    )
    assert feedback in (result.content if isinstance(result, ToolOutcome) else result)
    assert target.read_text() == before


@pytest.mark.parametrize("grant", ["read_only", None])
async def test_edit_preflight_cannot_replace_the_folder_access_decision(tmp_path, grant):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    target = project / "sample.txt"
    target.write_text("old\n")
    instance = tmp_path / "instance"
    store = FolderGrantStore(
        agent_home=home, instance_root=instance, path=instance / "folder-grants.json",
    )
    if grant is not None:
        store.upsert(
            {"path": str(project), "file_access": grant, "commands": False}, expected_revision=0,
        )
    editor = EditFileTool(workspace=home, resource_policy=ResourceAccessPolicy(
        workspace=home, context_manager=ContextRepoManager(), folder_grants=store.view,
        sandbox_default=True,
    ))
    call = edit_call("old", "new", replace_all=True)
    call.arguments["path"] = str(target)
    result, _ = await dispatch(home, editor, call, composite=True)
    assert result.stop_reason == "folder_access_required"
    assert result.policy_metadata["folder_access"]["reason"] == "write_not_granted"
    assert target.read_text() == "old\n"


@pytest.mark.parametrize("bulk", [False, True])
async def test_approved_edit_cannot_authorize_a_different_file_snapshot(tmp_path, bulk):
    target = tmp_path / "sample.txt"
    before = "".join(f"original {n}\n" for n in range(20))
    target.write_text(before)
    editor = EditFileTool(workspace=tmp_path)
    call = edit_call("original" if bulk else before, "replacement", replace_all=bulk)
    ask, _ = await dispatch(tmp_path, editor, call)
    admission, context = await issue_approval(ask)
    try:
        target.write_text(before + "different file version\n")
        # The callable replacement is still valid for this file, but its
        # preimage (and resulting content) is not what the user approved.
        changes, _ = await dispatch(tmp_path, editor, call, context=context)
        assert changes.stop_reason == "approval_required"
        assert changes.policy_metadata["effect_id"] != ask.policy_metadata["effect_id"]
        assert target.read_text() == before + "different file version\n"
        target.write_text(before)
        result, status = await dispatch(tmp_path, editor, call, context=context)
        assert status == "ok", result
        assert target.read_text() == (before.replace("original", "replacement") if bulk else "replacement")
    finally:
        admission.retire(context)


async def test_edit_rechecks_prepared_content_at_mutation(tmp_path, monkeypatch):
    target = tmp_path / "sample.txt"
    before = "old\n" + ("context\n" * 20)
    target.write_text(before)
    locks = FileLockRegistry()
    editor = EditFileTool(workspace=tmp_path, lock_registry=locks)
    acquire = locks.acquire

    async def change_during_acquire(path, owner):
        await asyncio.sleep(0)
        target.write_text(before + "changed while acquiring the lock\n")
        return await acquire(path, owner)

    monkeypatch.setattr(locks, "acquire", change_during_acquire)
    result, status = await dispatch(tmp_path, editor, edit_call("old", "new"))
    assert status == "error"
    assert "changed" in str(result.content)
    assert target.read_text() == before + "changed while acquiring the lock\n"


@pytest.mark.parametrize("argument", ["command", "cmd"])
async def test_hard_denied_command_never_asks_or_launches(tmp_path, monkeypatch, argument):
    tool = ExecTool(working_dir=str(tmp_path))

    async def unexpected_spawn(*args, **kwargs):
        pytest.fail("a denied command reached spawn")

    monkeypatch.setattr(tool, "_spawn", unexpected_spawn)
    monkeypatch.setattr(tool, "_stage_input_files", lambda _: pytest.fail("denied call was staged"))
    call = ToolCallRequest(id="denied", name="exec", arguments={argument: "rm -f scratch.txt"})
    result, status = await dispatch(tmp_path, tool, call)
    assert status == "error"
    assert result.stop_reason == "tool_error"
    assert "deny pattern filter" in str(result.content)


async def test_missing_managed_backend_is_not_an_approval_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("nanobot.agent.tools.sandbox.managed_bwrap_available", lambda: False)
    root = tmp_path / "workspace"
    root.mkdir()
    instance = tmp_path / "instance"
    grants = FolderGrantStore(
        agent_home=root, instance_root=instance, path=instance / "folder-grants.json",
    )
    grants.upsert(
        {"path": str(root), "file_access": "read_write", "commands": True}, expected_revision=0,
    )
    tool = ExecTool(
        working_dir=str(root),
        resource_policy=ResourceAccessPolicy(
            workspace=root, context_manager=ContextRepoManager(), folder_grants=grants.view,
        ),
    )

    async def unexpected_spawn(*args, **kwargs):
        pytest.fail("a command without confinement reached spawn")

    monkeypatch.setattr(tool, "_spawn", unexpected_spawn)
    call = ToolCallRequest(id="unsupported", name="exec", arguments={"command": "git clean -n"})
    result, status = await dispatch(root, tool, call)
    assert status == "error"
    assert result.stop_reason == "tool_error"
    assert "Sandbox command confinement is unavailable" in str(result.content)


async def test_admissible_shell_alias_requires_one_exact_approval(tmp_path):
    # A dry run in a disposable repository exercises real execution without deleting anything.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "untracked.txt").write_text("keep this\n")
    command = ToolCallRequest(id="git", name="exec", arguments={
        "arguments": {"cmd": "git clean -n", "workdir": str(tmp_path)},
    })
    tool = ExecTool(working_dir=str(tmp_path))
    ask, _ = await dispatch(tmp_path, tool, command)
    admission, context = await issue_approval(ask)
    try:
        class DenyAfterAssessment(AgentHook):
            async def before_execute_tool(self, context, tool_call, tool, params):
                tool.deny_patterns.append(r"\bgit\s+clean\b")

        refused, status = await dispatch(tmp_path, tool, command, hook=DenyAfterAssessment(),
                                         context=context)
        assert status == "error"
        assert "deny pattern filter" in str(refused.content)
        assert context.policy.effect_authority.permits(ask.policy_metadata["effect_id"])
        tool.deny_patterns.pop()

        result, status = await dispatch(tmp_path, tool, command, context=context)
        assert status == "ok", result
        assert "Would remove untracked.txt" in result
        assert (tmp_path / "untracked.txt").read_text() == "keep this\n"
        repeated, _ = await dispatch(tmp_path, tool, command, context=context)
        assert repeated.stop_reason == "approval_required"
    finally:
        admission.retire(context)


async def test_shell_approval_does_not_follow_a_new_project(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    tool = ExecTool(working_dir=str(first), restrict_to_workspace=True)
    call = ToolCallRequest(id="push", name="exec", arguments={"cmd": "git push"})

    async def unexpected_spawn(*args, **kwargs):
        pytest.fail("an approval for a different project reached spawn")

    monkeypatch.setattr(tool, "_spawn", unexpected_spawn)
    token = bind_workspace_scope(build_workspace_scope(first, "restricted"))
    try:
        ask, _ = await dispatch(first, tool, call)
        admission, context = await issue_approval(ask)
    finally:
        reset_workspace_scope(token)
    token = bind_workspace_scope(build_workspace_scope(second, "restricted"))
    try:
        new_ask, _ = await dispatch(first, tool, call, context=context)
        assert new_ask.stop_reason == "approval_required"
        assert new_ask.policy_metadata["effect_id"] != ask.policy_metadata["effect_id"]
    finally:
        reset_workspace_scope(token)
        admission.retire(context)


async def test_shell_launch_does_not_follow_a_changed_working_directory(
    tmp_path, monkeypatch,
):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    tool = ExecTool(working_dir=str(first))
    call = ToolCallRequest(id="cleanup", name="exec", arguments={"cmd": "git clean -n"})
    ask, _ = await dispatch(first, tool, call, composite=True)
    admission, context = await issue_approval(ask)
    launches = []
    stage = tool._stage_input_files

    def change_working_directory(values):
        tool.working_dir = str(second)
        return stage(values)

    async def record_launch(command, cwd, *args, **kwargs):
        launches.append(cwd)
        raise RuntimeError("unexpected launch")

    monkeypatch.setattr(tool, "_stage_input_files", change_working_directory)
    monkeypatch.setattr(tool, "_spawn", record_launch)
    try:
        result, status = await dispatch(first, tool, call, context=context, composite=True)
        assert status == "error"
        assert launches == []
        assert "working_dir changed after approval" in str(result.content)
    finally:
        admission.retire(context)


async def test_unavailable_command_never_requests_approval(tmp_path):
    call = ToolCallRequest(id="unavailable", name="exec", arguments={"cmd": "git push"})
    results, events = await execute_tool_calls(
        ToolRegistry(), [call], concurrent=False, external_lookup_counts={},
        workspace_violation_counts={}, hook=AgentHook(),
        context=AgentHookContext(iteration=0, messages=[]),
        tool_policy=RiskyActionPolicy(workspace=tmp_path),
    )
    assert events[0]["status"] == "error"
    assert "Tool 'exec' not found" in results[0]
