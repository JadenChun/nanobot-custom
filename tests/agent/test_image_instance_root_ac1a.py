from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.image_generation import (
    ImageGenerationTool,
    ImageGenerationToolConfig,
    reload_image_generation_tool,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.loader import load_config, save_config, set_config_path
from nanobot.config.schema import Config, ProviderConfig
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.providers.image_generation import GeneratedImageResponse

PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


class _UnusedProvider(LLMProvider):
    def __init__(self) -> None:
        pass

    def get_default_model(self) -> str:
        return "synthetic-model"

    async def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:
        raise AssertionError("The LLM provider must not be called")


class _FakeImageGenerationProvider:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def generate(self, **kwargs: Any) -> GeneratedImageResponse:
        return GeneratedImageResponse(images=[PNG_DATA_URL], content="", raw={})


def _make_config(
    *,
    config_path: Path,
    workspace: Path,
    model: str,
    image_model: str,
    save_dir: str,
) -> Config:
    config = Config()
    config.bind_source_path(config_path)
    config.agents.defaults.workspace = str(workspace)
    config.agents.defaults.model = model
    config.agents.defaults.model_preset = None
    config.tools.agent_browser.enabled = False
    config.tools.agent_device.enabled = False
    config.tools.desktop_use.enabled = False
    config.tools.restrict_to_workspace = True
    config.tools.image_generation = ImageGenerationToolConfig(
        configuration_source="upstream",
        enabled=True,
        provider="openrouter",
        model=image_model,
        save_dir=save_dir,
    )
    return config


def _walk_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def _artifact_path(payload: Any, instance_dir: Path, workspace_dir: Path) -> Path:
    media_dir = instance_dir / "media"
    for raw in _walk_strings(payload):
        if raw.startswith("data:") or "://" in raw:
            continue
        if Path(raw).suffix.lower() not in _IMAGE_SUFFIXES:
            continue

        candidate = Path(raw)
        candidates = [candidate] if candidate.is_absolute() else [
            instance_dir / candidate,
            workspace_dir / candidate,
            media_dir / candidate,
        ]
        for path in candidates:
            if path.is_file():
                resolved = path.resolve()
                assert resolved.is_relative_to(media_dir.resolve()), resolved
                return resolved

        matches = list(media_dir.rglob(candidate.name)) if media_dir.exists() else []
        if len(matches) == 1:
            resolved = matches[0].resolve()
            assert resolved.is_relative_to(media_dir.resolve()), resolved
            return resolved

    raise AssertionError(f"No generated image path found in tool result: {payload!r}")


def _image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]


def _assert_sidecar_exists(image_path: Path) -> None:
    sidecars = [
        image_path.with_suffix(".json"),
        Path(f"{image_path}.json"),
    ]
    assert any(sidecar.is_file() for sidecar in sidecars), image_path


def _result_ok(result: Any) -> bool:
    if isinstance(result, dict):
        return result.get("ok") is True
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (TypeError, ValueError):
            return False
        return isinstance(parsed, dict) and parsed.get("ok") is True
    return getattr(result, "ok", None) is True


@pytest.mark.asyncio
async def test_explicit_instance_media_root_survives_ambient_config_and_reload(
    tmp_path: Path,
) -> None:
    instance_dir = tmp_path / "instance"
    ambient_dir = tmp_path / "ambient"
    instance_workspace = instance_dir / "workspace"
    instance_dir.mkdir()
    ambient_dir.mkdir()
    instance_workspace.mkdir()

    explicit_path = instance_dir / "config.json"
    ambient_path = ambient_dir / "config.json"

    explicit_config = _make_config(
        config_path=explicit_path,
        workspace=instance_workspace,
        model="synthetic-model",
        image_model="explicit-model",
        save_dir="generated",
    )
    save_config(explicit_config, explicit_path)

    ambient_config = _make_config(
        config_path=ambient_path,
        workspace=ambient_dir / "workspace",
        model="ambient-model",
        image_model="ambient-model",
        save_dir="ambient-generated",
    )
    save_config(ambient_config, ambient_path)
    set_config_path(ambient_path)

    instance_media = instance_dir / "media"
    ambient_media = ambient_dir / "media"
    loop: AgentLoop | None = None

    try:
        with (
            patch("nanobot.migration_guard.require_runtime_ready", lambda *a, **k: None),
            patch(
                "nanobot.agent.tools.image_generation.get_image_gen_provider",
                return_value=_FakeImageGenerationProvider,
            ),
        ):
            loop = AgentLoop.from_config(
                load_config(explicit_path),
                tool_registry=ToolRegistry(),
                provider=_UnusedProvider(),
                image_generation_provider_configs={
                    "openrouter": ProviderConfig(api_key="test-only-key")
                },
            )

        tool = loop.tools.get("generate_image")
        assert isinstance(tool, ImageGenerationTool)

        with patch(
            "nanobot.agent.tools.image_generation.get_image_gen_provider",
            return_value=_FakeImageGenerationProvider,
        ):
            first_result = await tool.execute(prompt="explicit instance")

        first_payload = json.loads(first_result)
        first_image = _artifact_path(first_payload, instance_dir, instance_workspace)
        assert first_image.is_relative_to(instance_media.resolve())
        _assert_sidecar_exists(first_image)
        assert _image_files(ambient_media) == []

        reloaded_config = load_config(explicit_path)
        reloaded_config.tools.image_generation.model = "reloaded-model"
        save_config(reloaded_config, explicit_path)

        with patch(
            "nanobot.agent.tools.image_generation.get_image_gen_provider",
            return_value=_FakeImageGenerationProvider,
        ):
            reload_result = await reload_image_generation_tool(loop, loop.tools)
            assert _result_ok(reload_result)

            replacement = loop.tools.get("generate_image")
            assert isinstance(replacement, ImageGenerationTool)
            replacement_config = getattr(
                replacement, "config", getattr(replacement, "tool_config", None)
            )
            assert replacement_config is not None
            assert replacement_config.model == "reloaded-model"

            second_result = await replacement.execute(prompt="explicit instance after reload")

        second_payload = json.loads(second_result)
        second_image = _artifact_path(second_payload, instance_dir, instance_workspace)
        assert second_image.is_relative_to(instance_media.resolve())
        _assert_sidecar_exists(second_image)
        assert _image_files(ambient_media) == []
    finally:
        if loop is not None:
            await loop.aclose()
