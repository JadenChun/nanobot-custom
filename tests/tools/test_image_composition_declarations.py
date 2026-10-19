"""Native factories and optional image publication keep their constructed values."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.file_state import FileStates
from nanobot.agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from nanobot.agent.tools.image_generation import ImageGenerationTool, reload_image_generation_tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config import loader
from nanobot.config.schema import Config, ToolsConfig
from nanobot.providers.image_generation import ImageGenerationError
from tests.tools.test_image_generation_tool import PNG_DATA_URL, FakeImageClient


@pytest.mark.parametrize("tool_type", [ReadFileTool, WriteFileTool, EditFileTool, ListDirTool, SearchFilesTool])
@pytest.mark.parametrize("restrict,sandbox", [(False, ""), (True, ""), (False, "bwrap")])
def test_filesystem_factory_keeps_concrete_class_and_constructor_values(
    tmp_path, tool_type, restrict, sandbox,
) -> None:
    config = ToolsConfig(restrict_to_workspace=restrict)
    config.exec.sandbox = sandbox
    states = FileStates()
    tool = tool_type.create(ToolContext(config=config, workspace=str(tmp_path), file_state_store=states))
    assert type(tool) is tool_type
    assert tool._workspace == tmp_path
    assert tool._allowed_dir == (tmp_path if restrict or sandbox else None)
    assert tool._explicit_file_states is states
    assert tool._restrict_to_workspace is restrict
    assert tool._sandbox_restricts_workspace is bool(sandbox)
    assert tool._extra_read_allowed_dirs == [BUILTIN_SKILLS_DIR, tmp_path / "skills"]
    assert tool._extra_read_allowed_files == [tmp_path / "memory" / "history.jsonl"]
    assert tool._resource_policy is None and tool._lock_registry is None


def test_inherited_factory_retains_subclass_dispatch(tmp_path) -> None:
    class SpecializedWriter(WriteFileTool):
        pass

    tool = SpecializedWriter.create(ToolContext(config=ToolsConfig(), workspace=str(tmp_path)))
    assert type(tool) is SpecializedWriter
    assert tool._workspace == tmp_path


@pytest.mark.parametrize("restrict,sandbox", [(False, ""), (True, ""), (False, "bwrap")])
def test_image_factory_keeps_the_native_writer_and_file_states(tmp_path, restrict, sandbox) -> None:
    config = ToolsConfig(restrict_to_workspace=restrict)
    config.exec.sandbox = sandbox
    config.image_generation.configuration_source = "upstream"
    config.image_generation.enabled = True
    states = FileStates()
    tool = ImageGenerationTool.create(ToolContext(
        config=config, workspace=str(tmp_path), file_state_store=states,
    ))
    assert type(tool) is ImageGenerationTool and type(tool._write_tool) is WriteFileTool
    assert tool._write_tool._explicit_file_states is states
    assert tool._write_tool._allowed_dir == (tmp_path if restrict or sandbox else None)
    assert tool._write_tool._sandbox_restricts_workspace is bool(sandbox)


async def test_image_reload_reuses_the_original_writer_identity(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loader, "_current_config_path", tmp_path / "config.json")
    config = Config()
    config.tools.image_generation.configuration_source = "upstream"
    config.tools.image_generation.enabled = True
    states = FileStates()
    original = ImageGenerationTool.create(ToolContext(
        config=config.tools, workspace=str(tmp_path), file_state_store=states,
    ))
    writer = original._write_tool
    registry = ToolRegistry()
    registry.register(original)
    config.tools.image_generation.model = "reloaded-model"
    loader.save_config(config)
    state = SimpleNamespace(workspace=tmp_path, tools_config=config.tools)
    result = await reload_image_generation_tool(state, registry)
    replacement = registry.get("generate_image")
    assert result["ok"] and isinstance(replacement, ImageGenerationTool)
    assert replacement is not original and replacement.config.model == "reloaded-model"
    assert replacement._write_tool is writer and writer._explicit_file_states is states


@pytest.mark.parametrize("output", [None, "saved.png"])
@pytest.mark.parametrize("destination_mode", ["native", "none", "error"])
async def test_optional_destination_preserves_publication_and_early_fault_phases(
    tmp_path, monkeypatch, output, destination_mode,
) -> None:
    monkeypatch.setattr(loader, "_current_config_path", tmp_path / "config.json")
    config = ToolsConfig()
    config.image_generation.configuration_source = "upstream"
    config.image_generation.enabled = True
    tool = ImageGenerationTool.create(ToolContext(config=config, workspace=str(tmp_path)))
    client = FakeImageClient()
    monkeypatch.setattr(tool, "_provider_client", lambda: client)
    destination_reads = []
    original_destination = tool._output_destination

    def destination(value):
        destination_reads.append(value)
        if destination_mode == "none":
            return None
        if destination_mode == "error":
            raise ImageGenerationError("destination failure")
        return original_destination(value)

    monkeypatch.setattr(tool, "_output_destination", destination)
    publish = AsyncMock(wraps=tool._publish_output)
    monkeypatch.setattr(tool, "_publish_output", publish)
    response = await tool.execute("visible prompt", output_path=output)
    if output is not None and destination_mode == "error":
        assert response.is_error and "destination failure" in response
        assert destination_reads == [output] and client.calls == []
        publish.assert_not_awaited()
        return
    result = json.loads(response)
    assert len(result["artifacts"]) == 1 and len(client.calls) == 1
    if output is not None and destination_mode == "native":
        target = tmp_path / "generated_images" / output
        publish.assert_awaited_once_with(output, target, Path(result["artifacts"][0]["path"]))
        assert destination_reads == [output, output]
        assert result["path"] == str(target)
        assert target.read_bytes() == base64.b64decode(PNG_DATA_URL.split(",", 1)[1])
    else:
        publish.assert_not_awaited()
        assert "path" not in result
        assert destination_reads == ([] if output is None else [output])
