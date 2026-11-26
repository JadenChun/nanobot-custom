"""Verify the default guidance and its declared capability groups."""

from importlib.resources import files as pkg_files

from kohaku.agent.context import ContextBuilder
from kohaku.context.budget import TokenMeter


def test_universal_tool_contract_is_bounded():
    content = (pkg_files("kohaku") / "templates" / "agent" / "tool_contract.md").read_text(
        encoding="utf-8",
    )
    assert TokenMeter.count_text(content) <= 550
    assert "narrowest tool" in content
    assert "apply_patch" not in content


def test_workspace_and_delivery_sections_follow_available_tools(tmp_path):
    builder = ContextBuilder(tmp_path)
    names = frozenset({"repository_catalog", "message"})
    compiled = builder.compile_system_prompt(tool_names=names)
    assert "repository_discovery" in compiled.section_tokens
    assert "chat_delivery" in compiled.section_tokens
    assert "project_selection" not in compiled.section_tokens
    assert "folder_requests" not in compiled.section_tokens


def test_identity_guidance_does_not_advertise_grep():
    content = (pkg_files("kohaku") / "templates" / "agent" / "identity.md").read_text(
        encoding="utf-8",
    )
    assert "grep" not in content
