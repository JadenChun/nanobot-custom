from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.agent.tools import image as image_tool
from nanobot.agent.tools.image import (
    ImageGenerationTool,
    _extract_image_paths,
    image_generation_available,
    reset_current_user_image_request,
    set_current_user_image_request,
)
from nanobot.config.schema import ImageConfig


def test_image_config_defaults_to_codex_cli() -> None:
    assert ImageConfig().provider == "codex_cli"


def test_image_generation_available_for_default_codex_cli() -> None:
    assert image_generation_available(ImageConfig(provider="codex_cli"))


def test_image_tool_prompt_schema_asks_for_plain_scene_description() -> None:
    tool_desc = ImageGenerationTool.description
    prompt_desc = ImageGenerationTool.parameters["properties"]["prompt"]["description"]

    assert "plain natural-language description" in tool_desc
    assert "placement labels" in tool_desc
    assert "Plain natural-language description" in prompt_desc
    assert "visible scene only" in prompt_desc
    assert "SEO" not in tool_desc
    assert "SEO" not in prompt_desc


def test_extract_image_paths_from_codex_output() -> None:
    paths = _extract_image_paths(
        "Generated image saved at C:\\Users\\chang\\.codex\\generated_images\\session\\ig_1.png"
    )

    assert paths == [Path("C:\\Users\\chang\\.codex\\generated_images\\session\\ig_1.png")]


def test_workspace_output_paths_go_to_generated_images(tmp_path) -> None:
    tool = ImageGenerationTool(ImageConfig(provider="codex_cli"), workspace=tmp_path)

    assert tool._resolve_output_path("avatar.png") == tmp_path / "generated_images" / "avatar.png"
    assert (
        tool._resolve_output_path(str(tmp_path / "avatar.png"))
        == tmp_path / "generated_images" / "avatar.png"
    )
    assert (
        tool._resolve_output_path(str(tmp_path / "generated_images" / "avatar.png"))
        == tmp_path / "generated_images" / "avatar.png"
    )


@pytest.mark.asyncio
async def test_codex_cli_backend_copies_new_generated_image(tmp_path, monkeypatch) -> None:
    codex_home = tmp_path / "codex-home"
    generated = codex_home / "generated_images" / "session-1" / "ig_1.png"
    output = tmp_path / "workspace" / "avatar.png"

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            generated.parent.mkdir(parents=True)
            generated.write_bytes(b"png-bytes")
            return (
                f"Generated image saved at {generated}".encode(),
                b"",
            )

    async def fake_create_subprocess_exec(*args, **kwargs):
        assert args[:7] == (
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--color",
            "never",
            "-m",
            "gpt-5.4-mini",
        )
        assert "cute robot" in args[7]
        return FakeProcess()

    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(image_tool.shutil, "which", lambda command: command)
    monkeypatch.setattr(image_tool.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    tool = ImageGenerationTool(
        ImageConfig(provider="codex_cli", codex_command="codex", codex_model="gpt-5.4-mini")
    )
    result = json.loads(await tool.execute("cute robot", str(output), "1:1"))

    assert result["provider"] == "codex_cli"
    assert Path(result["path"]) == output
    assert Path(result["source_path"]) == generated
    assert output.read_bytes() == b"png-bytes"


@pytest.mark.asyncio
async def test_codex_cli_backend_uses_tool_prompt_over_turn_context(tmp_path, monkeypatch) -> None:
    codex_home = tmp_path / "codex-home"
    generated = codex_home / "generated_images" / "session-1" / "ig_1.png"
    output = tmp_path / "workspace" / "generated_images" / "avatar.png"
    captured_prompt = ""

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            generated.parent.mkdir(parents=True)
            generated.write_bytes(b"png-bytes")
            return (f"Generated image saved at {generated}".encode(), b"")

    async def fake_create_subprocess_exec(*args, **kwargs):
        nonlocal captured_prompt
        captured_prompt = args[7]
        return FakeProcess()

    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(image_tool.shutil, "which", lambda command: command)
    monkeypatch.setattr(image_tool.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    token = set_current_user_image_request("help me generate a man eating salmon don image")
    try:
        tool = ImageGenerationTool(ImageConfig(provider="codex_cli", codex_command="codex"))
        result = json.loads(
            await tool.execute("A tasteful, high-quality rewritten prompt", str(output), "1:1")
        )
    finally:
        reset_current_user_image_request(token)

    assert result["prompt"] == "A tasteful, high-quality rewritten prompt"
    assert captured_prompt == "A tasteful, high-quality rewritten prompt"
    assert output.read_bytes() == b"png-bytes"


@pytest.mark.asyncio
async def test_codex_cli_backend_falls_back_to_generated_images_scan(tmp_path, monkeypatch) -> None:
    codex_home = tmp_path / "codex-home"
    generated = codex_home / "generated_images" / "session-1" / "ig_2.png"
    output = tmp_path / "workspace" / "avatar.png"

    class FakeProcess:
        returncode = 0

        async def communicate(self):
            generated.parent.mkdir(parents=True)
            generated.write_bytes(b"png-bytes")
            return (b"done", b"")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess()

    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(image_tool.shutil, "which", lambda command: command)
    monkeypatch.setattr(image_tool.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    tool = ImageGenerationTool(ImageConfig(provider="codex_cli", codex_command="codex"))
    result = json.loads(await tool.execute("cute robot", str(output), "1:1"))

    assert Path(result["source_path"]) == generated
    assert output.read_bytes() == b"png-bytes"
