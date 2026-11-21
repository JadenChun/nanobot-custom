"""Regression tests for registry-derived tool guidance scoping."""

from pathlib import Path

from nanobot.agent.context import ContextBuilder


def test_disabled_tool_lines_are_absent_from_guidance(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builder = ContextBuilder(workspace)

    prompt = builder.build_system_prompt(
        tool_names=frozenset({"read_file", "write_file", "edit_file", "list_dir"})
    )

    assert "web_search" not in prompt
    assert "web_fetch" not in prompt
    assert "cron" not in prompt
    assert "run_cli_app" not in prompt
    assert "generate_image" not in prompt
    assert "read_file" in prompt


def test_full_tool_set_keeps_guidance_lines(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builder = ContextBuilder(workspace)

    prompt = builder.build_system_prompt(
        tool_names=frozenset(
            {
                "read_file",
                "write_file",
                "edit_file",
                "list_dir",
                "exec",
                "web_search",
                "web_fetch",
                "message",
                "cron",
                "generate_image",
                "run_cli_app",
                "repository_catalog",
                "save_knowledge",
            }
        )
    )

    assert "web_search" in prompt
    assert "cron" in prompt
    assert "generate_image" in prompt


def test_unspecified_tool_set_keeps_legacy_template(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    builder = ContextBuilder(workspace)

    prompt = builder.build_system_prompt()

    assert "web_search" in prompt
    assert "cron" in prompt
