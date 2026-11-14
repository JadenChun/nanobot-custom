"""Regression test for A3-1 AC1b: risk assessment and execution must bind to the same target."""

from pathlib import Path

import pytest

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.policy import RiskyActionPolicy
from nanobot.agent.tools.execution import execute_tool_calls
from nanobot.agent.tools.filesystem import WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import ToolCallRequest
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    reset_workspace_scope,
    validate_workspace_scope_payload,
)

SMALL_CONTENT = "small file content\n"
REPLACEMENT = "replacement content\n"
LARGE_LINES = [f"large line {i}\n" for i in range(1, 301)]


class RetargetSymlinkHook(AgentHook):
    """Swap the approved symlink target just before tool execution."""

    def __init__(self, link: Path, new_target: Path) -> None:
        self._link = link
        self._new_target = new_target

    async def before_execute_tool(self, context, tool_call, tool, params):
        self._link.unlink()
        self._link.symlink_to(self._new_target)


@pytest.mark.asyncio
async def test_write_does_not_follow_retargeted_symlink(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()

    small = project / "small.txt"
    small.write_text(SMALL_CONTENT)

    large = project / "large.txt"
    large.write_text("".join(LARGE_LINES))

    link = project / "link.txt"
    link.symlink_to(small)

    scope = validate_workspace_scope_payload(
        {"project_path": str(project), "access_mode": "restricted"},
        default_workspace=home,
        default_restrict_to_workspace=True,
    )
    token = bind_workspace_scope(scope)
    try:
        tool = WriteFileTool(workspace=home, allowed_dir=home, restrict_to_workspace=True)
        policy = RiskyActionPolicy(
            workspace=home,
            write_target_resolver=lambda name, raw: tool.resolve_write_path(raw),
        )
        registry = ToolRegistry()
        registry.register(tool)
        hook = RetargetSymlinkHook(link, large)

        request = ToolCallRequest(
            id="retarget",
            name="write_file",
            arguments={"path": "link.txt", "content": REPLACEMENT},
        )

        results, events = await execute_tool_calls(
            registry,
            [request],
            concurrent=False,
            external_lookup_counts={},
            workspace_violation_counts={},
            hook=hook,
            context=AgentHookContext(iteration=0, messages=[]),
            tool_policy=policy,
        )

        assert large.read_text() == "".join(LARGE_LINES)
        assert small.read_text() == SMALL_CONTENT
        assert len(results) == 1
        result = results[0]
        assert not result.lower().startswith("success")
        if events:
            assert events[0].get("status") != "success"
        assert REPLACEMENT not in large.read_text()
    finally:
        reset_workspace_scope(token)
