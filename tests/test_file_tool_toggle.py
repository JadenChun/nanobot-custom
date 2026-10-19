from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.filesystem import FileToolsConfig, ReadFileTool
from nanobot.config.schema import Config, ToolsConfig
from tests.agent.p6_admitted_fixtures import AgentInputs
from tests.agent.p6_admitted_fixtures import admitted_async as admitted_async

FILE_TOOL_NAMES = {
    "apply_patch",
    "edit_file",
    "find_files",
    "grep",
    "list_dir",
    "read_file",
    "write_file",
}


def test_file_tools_enabled_by_default():
    assert FileToolsConfig().enable is True
    assert Config().tools.file.enable is True


def test_file_tool_gate_follows_flag():
    cfg = ToolsConfig()
    cfg.file.enable = False
    assert ReadFileTool.enabled(SimpleNamespace(config=cfg)) is False
    assert ReadFileTool.enabled(SimpleNamespace(config=ToolsConfig())) is True


@pytest.mark.parametrize("enabled", [False, True])
async def test_file_tool_loader_obeys_flag_in_acquired_graph(tmp_path, admitted_async, enabled):
    config = Config()
    config.tools.file.enable = enabled
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    def build(inputs: AgentInputs) -> AgentLoop:
        return AgentLoop.from_config(
            inputs.config, inputs.bus, provider=provider,
            session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            enable_legacy_auto_compact=False,
        )

    loop = await admitted_async(tmp_path, build, config=config)
    names = set(loop.tools.tool_names)
    if enabled:
        assert FILE_TOOL_NAMES & names == {"read_file", "write_file", "edit_file", "list_dir"}
    else:
        assert FILE_TOOL_NAMES.isdisjoint(names)
    assert {"message", "delegate_task", "review_work", "explore"} <= names
