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


def test_generated_image_scan_selects_newest_eligible_candidate(tmp_path) -> None:
    import os

    root = tmp_path / "generated_images"
    root.mkdir()
    paths = {}
    for name, stamp in (("old.png", 900), ("earlier.jpg", 999), ("latest.PNG", 1001),
                        ("known.webp", 2000), ("not-image.txt", 3000)):
        path = root / name
        path.write_bytes(b"synthetic-image")
        os.utime(path, (stamp, stamp))
        paths[name] = path.resolve()

    selected = image_tool._find_generated_image(
        "done", tmp_path, {paths["known.webp"]}, start_time=1000,
    )

    assert selected == paths["latest.PNG"]


@pytest.mark.parametrize("case", ["missing-root", "empty", "known", "too-old"])
def test_generated_image_scan_keeps_empty_and_exclusion_results(tmp_path, case) -> None:
    import os

    root = tmp_path / "generated_images"
    known = set()
    if case != "missing-root":
        root.mkdir()
    if case in {"known", "too-old"}:
        path = root / "candidate.png"
        path.write_bytes(b"synthetic-image")
        stamp = 1000 if case == "known" else 997
        os.utime(path, (stamp, stamp))
        if case == "known":
            known.add(path.resolve())

    assert image_tool._find_generated_image("done", tmp_path, known, start_time=1000) is None


def test_generated_image_scan_preserves_explicit_path_precedence(tmp_path) -> None:
    import os

    explicit = tmp_path / "explicit.png"
    explicit.write_bytes(b"explicit-image")
    os.utime(explicit, (1, 1))
    root = tmp_path / "generated_images"
    root.mkdir()
    candidate = root / "newer.png"
    candidate.write_bytes(b"newer-image")
    os.utime(candidate, (2000, 2000))

    assert image_tool._find_generated_image(
        str(explicit), tmp_path, {explicit.resolve()}, start_time=1000,
    ) == explicit.resolve()


def test_generated_image_scan_preserves_stable_timestamp_ties(tmp_path, monkeypatch) -> None:
    import os

    root = tmp_path / "generated_images"
    root.mkdir()
    paths = [root / "z-first.png", root / "a-second.png"]
    for path in paths:
        path.write_bytes(b"synthetic-image")
        os.utime(path, (1000, 1000))
    monkeypatch.setattr(Path, "rglob", lambda _path, _pattern: iter(paths))

    assert image_tool._find_generated_image("done", tmp_path, set(), start_time=1000) == paths[0]
