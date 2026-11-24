"""Tests for default core tool guidance."""

from importlib.resources import files as pkg_files

from kohaku.agent.context import ContextBuilder


def _template_text(name: str) -> str:
    return (pkg_files("kohaku") / "templates" / "agent" / name).read_text(encoding="utf-8")


def test_tool_contract_uses_only_registered_core_tool_names():
    content = _template_text("tool_contract.md")

    assert "edit_file" in content
    assert "write_file" in content
    assert "find_files" in content
    assert "grep" in content
    assert "apply_patch" not in content


def test_default_prompt_has_no_unconditional_optional_tool_claim(tmp_path):
    prompt = ContextBuilder(tmp_path).build_system_prompt()

    assert "apply_patch" not in prompt
    assert "find_files" in prompt
    assert "grep" in prompt

    scoped = ContextBuilder(tmp_path).build_system_prompt(
        tool_names=frozenset({"read_file", "list_dir"}),
    )
    assert "find_files" not in scoped
    assert "grep" not in scoped
    assert "read_file" in scoped
    assert "list_dir" in scoped


def test_identity_guidance_does_not_advertise_grep():
    content = _template_text("identity.md")

    assert "grep" not in content
