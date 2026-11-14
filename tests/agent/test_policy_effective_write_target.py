from pathlib import Path

import pytest

from nanobot.agent.policy import RiskyActionPolicy
from nanobot.agent.tools.filesystem import WriteFileTool
from nanobot.providers.base import ToolCallRequest
from nanobot.security.workspace_access import (
    bind_workspace_scope,
    reset_workspace_scope,
    validate_workspace_scope_payload,
)


def _numbered_lines(prefix: str) -> str:
    return "\n".join(f"{prefix} {index}" for index in range(1, 301)) + "\n"


@pytest.mark.asyncio
async def test_risky_write_uses_effective_project_target(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    other = tmp_path / "other"
    for directory in (home, project, other):
        directory.mkdir()

    home_readme = home / "README.md"
    project_readme = project / "README.md"
    other_readme = other / "README.md"

    home_readme.write_text("small\n", encoding="utf-8")
    project_readme.write_text(_numbered_lines("project original"), encoding="utf-8")
    other_readme.write_text(_numbered_lines("other original"), encoding="utf-8")

    tool = WriteFileTool(workspace=home, allowed_dir=home, restrict_to_workspace=True)

    project_scope = validate_workspace_scope_payload(
        {"project_path": str(project), "access_mode": "restricted"},
        default_workspace=home,
        default_restrict_to_workspace=True,
    )
    project_binding = bind_workspace_scope(project_scope)
    try:
        assert tool.resolve_write_path("README.md") == project_readme.resolve()

        policy = RiskyActionPolicy(
            workspace=home,
            write_target_resolver=lambda name, raw: tool.resolve_write_path(raw),
        )

        replacement = _numbered_lines("replacement")
        request = ToolCallRequest(
            id="write-project-readme",
            name="write_file",
            arguments={"path": "README.md", "content": replacement},
        )

        decision = await policy.evaluate(messages=[], tool_calls=[request])
        assert decision.stop_reason == "approval_required"
        assert decision.requires_approval is True
        assert project_readme.read_text(encoding="utf-8").startswith("project original 1")
        assert home_readme.read_text(encoding="utf-8") == "small\n"

        assert policy.validate_write_target(request, tool) is None

        # Re-assess the same call so the scope-change check below still has a
        # current project target to compare against.
        reassessment = await policy.evaluate(messages=[], tool_calls=[request])
        assert reassessment.stop_reason == "approval_required"

        other_scope = validate_workspace_scope_payload(
            {"project_path": str(other), "access_mode": "restricted"},
            default_workspace=home,
            default_restrict_to_workspace=True,
        )
        other_binding = bind_workspace_scope(other_scope)
        try:
            error = policy.validate_write_target(request, tool)
            assert error is not None
            assert "changed before dispatch" in error
        finally:
            reset_workspace_scope(other_binding)

        denied_request = ToolCallRequest(
            id="write-denied-target",
            name="write_file",
            arguments={"path": "../outside.txt", "content": "x"},
        )
        denied_decision = await policy.evaluate(messages=[], tool_calls=[denied_request])
        assert denied_decision.action == "allow"

        denied_result = await tool.execute(path="../outside.txt", content="x")
        assert "outside" in denied_result.lower()
        assert not (tmp_path / "outside.txt").exists()

        await tool.execute(path="README.md", content=replacement)
        assert project_readme.read_text(encoding="utf-8").startswith("replacement 1")
        assert home_readme.read_text(encoding="utf-8") == "small\n"
    finally:
        reset_workspace_scope(project_binding)
