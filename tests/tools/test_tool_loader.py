"""Tests for tool plugin architecture: ToolLoader, ToolContext, metadata."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.loader import _SKIP_MODULES, ToolLoader
from nanobot.agent.tools.runtime_control import AgentRuntimeControl
from nanobot.config.schema import Config
from nanobot.providers.base import GenerationSettings, LLMResponse
from tests.agent.p6_admitted_fixtures import AgentInputs, SyncAdmittedFactory
from tests.agent.p6_admitted_fixtures import admitted_sync as admitted_sync


@pytest.fixture
def loader_context(tmp_path: Path, admitted_sync: SyncAdmittedFactory, monkeypatch) -> ToolContext:
    config = Config()
    config.tools.image_generation.configuration_source = "upstream"
    config.tools.image_generation.enabled = False
    config.tools.agent_browser.enabled = False
    config.tools.agent_device.enabled = False
    config.tools.desktop_use.enabled = False
    monkeypatch.setenv("AGENT_VPN_BROKER_ENABLED", "false")
    captured: list[AgentInputs] = []

    def build(inputs: AgentInputs) -> AgentLoop:
        captured.append(inputs)
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.supports_tools = True
        provider.generation = GenerationSettings(max_tokens=100)
        provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="unused"))
        return AgentLoop(
            bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
            session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            tools_config=inputs.config.tools, restrict_to_workspace=True,
            session_ttl_minutes=0,
        )

    loop = admitted_sync(tmp_path, build, config=config)
    inputs = captured[0]
    assert loop.bus is inputs.bus and loop.sessions is inputs.sessions
    assert loop.tools is inputs.tools and loop.cron_service is inputs.cron
    assert loop.turn_delivery_factory is inputs.delivery
    assert loop.sessions.has_managed_authority
    assert loop._fork_dependencies.crawler_enabled is False
    assert not admitted_sync.runner.get_loop().is_running()
    return ToolContext(
        config=inputs.config.tools, workspace=str(loop.workspace), bus=loop.bus,
        subagent_manager=loop.subagents, cron_service=loop.cron_service,
        exec_session_manager=loop._exec_session_manager, sessions=loop.sessions,
        image_generation_provider_configs={}, timezone="UTC",
        workspace_sandbox=loop.workspace_scopes.sandbox_status,
        runtime_control=AgentRuntimeControl(loop), fork_dependencies=loop._fork_dependencies,
    )


class _MinimalTool(Tool):
    @property
    def name(self) -> str:
        return "test_minimal"

    @property
    def description(self) -> str:
        return "A test tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> Any:
        return "ok"


def test_tool_default_config_cls_is_none():
    assert _MinimalTool.config_cls() is None


def test_tool_default_config_key_is_empty():
    assert _MinimalTool.config_key == ""


def test_tool_default_enabled_is_true():
    assert _MinimalTool.enabled(None) is True


def test_tool_default_create_returns_instance():
    tool = _MinimalTool.create(None)
    assert isinstance(tool, _MinimalTool)
    assert tool.name == "test_minimal"


def test_tool_plugin_discoverable_default_is_true():
    assert _MinimalTool._plugin_discoverable is True


# --- ToolContext tests ---


def test_tool_context_has_required_fields():
    field_names = {f.name for f in fields(ToolContext)}
    required = {
        "config", "workspace", "bus", "subagent_manager",
        "cron_service", "exec_session_manager", "file_state_store",
        "provider_snapshot_loader", "image_generation_provider_configs", "timezone",
        "runtime_control",
    }
    assert required <= field_names


def test_tool_context_defaults():
    ctx = ToolContext(config=None, workspace="/tmp")
    assert ctx.bus is None
    assert ctx.subagent_manager is None
    assert ctx.cron_service is None
    assert ctx.exec_session_manager is None
    assert ctx.provider_snapshot_loader is None
    assert ctx.image_generation_provider_configs is None
    assert ctx.runtime_control is None
    assert ctx.timezone == "UTC"


# --- ToolLoader tests ---


def test_skip_modules_excludes_infrastructure():
    infra = {"base", "schema", "registry", "context", "loader", "config",
             "file_state", "sandbox", "mcp", "__init__"}
    assert infra <= _SKIP_MODULES


def test_discover_finds_concrete_tools():
    loader = ToolLoader()
    discovered = loader.discover()
    class_names = {cls.__name__ for cls in discovered}
    assert "ApplyPatchTool" in class_names
    assert "ExecTool" in class_names
    assert "CliAppsTool" in class_names
    assert "MessageTool" in class_names
    assert "MyTool" in class_names
    assert "SpawnTool" in class_names
    assert "ExecSessionTool" in class_names


def test_discover_excludes_abstract_and_mcp():
    loader = ToolLoader()
    discovered = loader.discover()
    class_names = {cls.__name__ for cls in discovered}
    assert "_FsTool" not in class_names
    assert "_SearchTool" not in class_names
    assert "MCPToolWrapper" not in class_names
    assert "MCPResourceWrapper" not in class_names
    assert "MCPPromptWrapper" not in class_names


def test_discover_skips_private_classes():
    loader = ToolLoader()
    discovered = loader.discover()
    for cls in discovered:
        assert not cls.__name__.startswith("_")


def test_loader_registers_exec_with_real_tools_config(tmp_path, loader_context):
    """Real config objects catch bad ctx.config attribute paths that mocks hide."""
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.config.schema import ToolsConfig

    ctx = loader_context
    assert isinstance(ctx.config, ToolsConfig)
    registry = ToolRegistry()
    registered = ToolLoader().load(ctx, registry)

    assert "exec" in registered
    assert registry.has("exec")


def test_loader_wires_shared_exec_session_manager(tmp_path, loader_context):
    from nanobot.agent.tools.exec_session import (
        ExecSessionManager,
        ExecSessionTool,
        ListExecSessionsTool,
    )
    from nanobot.agent.tools.registry import ToolRegistry

    ctx = loader_context
    manager = ctx.exec_session_manager
    assert isinstance(manager, ExecSessionManager)
    registry = ToolRegistry()
    ToolLoader().load(ctx, registry)

    assert registry.get("exec")._session_manager is manager
    assert isinstance(registry.get("exec_session"), ExecSessionTool)
    assert registry.get("exec_session")._manager is manager
    assert registry.get("write_stdin") is None
    assert isinstance(registry.get("list_exec_sessions"), ListExecSessionsTool)
    assert registry.get("list_exec_sessions")._manager is manager
    session_tool = ExecSessionTool.create(ctx)
    list_tool = ListExecSessionsTool.create(ctx)
    assert isinstance(session_tool, ExecSessionTool) and session_tool._manager is manager
    assert isinstance(list_tool, ListExecSessionsTool) and list_tool._manager is manager
    definition_names = {
        definition["function"]["name"]
        for definition in registry.get_definitions()
    }
    assert "exec_session" in definition_names
    assert "list_exec_sessions" in definition_names
    assert "write_stdin" not in definition_names


# --- Task 4: _FsTool.create() ---


def test_fs_tool_create_builds_from_context():
    from nanobot.agent.tools.filesystem import ReadFileTool
    mock_config = MagicMock()
    mock_config.restrict_to_workspace = False
    mock_config.exec.sandbox = ""
    ctx = ToolContext(config=mock_config, workspace="/tmp/test")
    tool = ReadFileTool.create(ctx)
    assert isinstance(tool, ReadFileTool)
    assert tool._workspace == Path("/tmp/test")


def test_fs_tool_create_respects_restrict_to_workspace():
    from nanobot.agent.tools.filesystem import ReadFileTool
    mock_config = MagicMock()
    mock_config.restrict_to_workspace = True
    mock_config.exec.sandbox = ""
    ctx = ToolContext(config=mock_config, workspace="/tmp/test")
    tool = ReadFileTool.create(ctx)
    assert tool._allowed_dir == Path("/tmp/test")


def test_fs_tool_create_respects_sandbox():
    from nanobot.agent.tools.filesystem import ReadFileTool
    mock_config = MagicMock()
    mock_config.restrict_to_workspace = False
    mock_config.exec.sandbox = "bwrap"
    ctx = ToolContext(config=mock_config, workspace="/tmp/test")
    tool = ReadFileTool.create(ctx)
    assert tool._allowed_dir == Path("/tmp/test")


# --- Task 5: MessageTool, SpawnTool, CronTool ---


async def test_message_tool_create():
    from nanobot.agent.tools.message import MessageTool
    mock_bus = MagicMock()
    mock_config = MagicMock()
    ctx = ToolContext(config=mock_config, workspace="/tmp", bus=mock_bus)
    tool = MessageTool.create(ctx)
    assert isinstance(tool, MessageTool)


def test_spawn_tool_create():
    from nanobot.agent.tools.spawn import SpawnTool
    mock_mgr = MagicMock()
    mock_config = MagicMock()
    ctx = ToolContext(config=mock_config, workspace="/tmp", subagent_manager=mock_mgr)
    tool = SpawnTool.create(ctx)
    assert isinstance(tool, SpawnTool)


def test_cron_tool_enabled_without_service():
    from nanobot.agent.tools.cron import CronTool
    mock_config = MagicMock()
    ctx = ToolContext(config=mock_config, workspace="/tmp", cron_service=None)
    assert CronTool.enabled(ctx) is False


def test_cron_tool_enabled_with_service():
    from nanobot.agent.tools.cron import CronTool
    mock_service = MagicMock()
    mock_config = MagicMock()
    ctx = ToolContext(config=mock_config, workspace="/tmp", cron_service=mock_service)
    assert CronTool.enabled(ctx) is True


def test_cron_tool_create():
    from nanobot.agent.tools.cron import CronTool
    mock_service = MagicMock()
    mock_config = MagicMock()
    ctx = ToolContext(
        config=mock_config, workspace="/tmp",
        cron_service=mock_service, timezone="Asia/Shanghai",
    )
    tool = CronTool.create(ctx)
    assert isinstance(tool, CronTool)


# --- Task 6: ExecTool, WebTools, ImageGenerationTool ---


def test_exec_tool_config_cls():
    from nanobot.agent.tools.shell import ExecTool, ExecToolConfig
    assert ExecTool.config_cls() is ExecToolConfig
    assert ExecTool.config_key == "exec"


def test_exec_tool_enabled():
    from nanobot.agent.tools.shell import ExecTool
    mock_config = MagicMock()
    mock_config.exec.enable = True
    ctx = ToolContext(config=mock_config, workspace="/tmp")
    assert ExecTool.enabled(ctx) is True
    mock_config.exec.enable = False
    assert ExecTool.enabled(ctx) is False


def test_exec_tool_create():
    from nanobot.agent.tools.shell import ExecTool
    mock_config = MagicMock()
    mock_config.exec.enable = True
    mock_config.exec.timeout = 120
    mock_config.exec.sandbox = ""
    mock_config.exec.path_prepend = "/venv/bin"
    mock_config.exec.path_append = ""
    mock_config.exec.allowed_env_keys = []
    mock_config.exec.allow_patterns = []
    mock_config.exec.deny_patterns = []
    mock_config.restrict_to_workspace = False
    ctx = ToolContext(config=mock_config, workspace="/tmp")
    tool = ExecTool.create(ctx)
    assert isinstance(tool, ExecTool)
    assert tool.path_prepend == "/venv/bin"


def test_web_tools_config_cls():
    from nanobot.agent.tools.web import WebFetchTool, WebSearchTool, WebToolsConfig
    assert WebSearchTool.config_key == "web"
    assert WebSearchTool.config_cls() is WebToolsConfig
    assert WebFetchTool.config_key == "web"
    assert WebFetchTool.config_cls() is WebToolsConfig


def test_web_tools_enabled():
    from nanobot.agent.tools.web import WebSearchTool
    mock_config = MagicMock()
    mock_config.web.enable = True
    ctx = ToolContext(config=mock_config, workspace="/tmp")
    assert WebSearchTool.enabled(ctx) is True
    mock_config.web.enable = False
    assert WebSearchTool.enabled(ctx) is False


def test_web_search_tool_create():
    from nanobot.agent.tools.web import WebSearchTool
    mock_config = MagicMock()
    mock_config.web.enable = True
    mock_config.web.search = MagicMock()
    mock_config.web.proxy = None
    mock_config.web.user_agent = None
    ctx = ToolContext(config=mock_config, workspace="/tmp")
    tool = WebSearchTool.create(ctx)
    assert isinstance(tool, WebSearchTool)


def test_web_fetch_tool_create():
    from nanobot.agent.tools.web import WebFetchTool
    mock_config = MagicMock()
    mock_config.web.enable = True
    mock_config.web.fetch = MagicMock()
    mock_config.web.proxy = None
    mock_config.web.user_agent = None
    ctx = ToolContext(config=mock_config, workspace="/tmp")
    tool = WebFetchTool.create(ctx)
    assert isinstance(tool, WebFetchTool)


def test_image_gen_tool_config_cls():
    from nanobot.agent.tools.image_generation import ImageGenerationTool, ImageGenerationToolConfig
    assert ImageGenerationTool.config_key == "image_generation"
    assert ImageGenerationTool.config_cls() is ImageGenerationToolConfig


def test_image_gen_tool_enabled():
    from nanobot.agent.tools.image_generation import ImageGenerationTool, ImageGenerationToolConfig
    from nanobot.config.schema import ProviderConfig, ToolsConfig
    mock_config = ToolsConfig(image_generation=ImageGenerationToolConfig(
        configuration_source="upstream", enabled=True,
    ))
    ctx = ToolContext(
        config=mock_config, workspace="/tmp",
        image_generation_provider_configs={"openrouter": ProviderConfig(api_key="synthetic-offline-key")},
    )
    assert ImageGenerationTool.enabled(ctx) is True
    mock_config.image_generation.enabled = False
    assert ImageGenerationTool.enabled(ctx) is False


def test_image_gen_tool_create():
    from nanobot.agent.tools.image_generation import ImageGenerationTool, ImageGenerationToolConfig
    from nanobot.config.schema import ProviderConfig, ToolsConfig
    mock_config = ToolsConfig(image_generation=ImageGenerationToolConfig(
        configuration_source="upstream", enabled=True,
    ))
    provider_config = ProviderConfig(api_key="synthetic-offline-key", api_base="https://example.invalid")
    ctx = ToolContext(
        config=mock_config, workspace="/tmp",
        image_generation_provider_configs={"openrouter": provider_config},
    )
    tool = ImageGenerationTool.create(ctx)
    assert isinstance(tool, ImageGenerationTool)
    assert tool.config is mock_config.image_generation
    assert tool.legacy_config is None
    assert tool.provider_configs == {"openrouter": provider_config}


# --- Task 7: MyToolConfig + MCP wrappers ---


def test_my_tool_config_cls():
    from nanobot.agent.tools.self import MyTool, MyToolConfig
    assert MyTool.config_key == "my"
    assert MyTool.config_cls() is MyToolConfig


def test_my_tool_enabled():
    from nanobot.agent.tools.self import MyTool
    mock_config = MagicMock()
    mock_config.my.enable = True
    ctx = ToolContext(
        config=mock_config,
        workspace="/tmp",
        runtime_control=MagicMock(),
    )
    assert MyTool.enabled(ctx) is True
    mock_config.my.enable = False
    assert MyTool.enabled(ctx) is False


def test_my_tool_requires_runtime_control():
    from nanobot.agent.tools.self import MyTool

    mock_config = MagicMock()
    mock_config.my.enable = True
    ctx = ToolContext(config=mock_config, workspace="/tmp")

    assert MyTool.enabled(ctx) is False


def test_mcp_wrappers_not_discoverable():
    from nanobot.agent.tools.mcp import MCPPromptWrapper, MCPResourceWrapper, MCPToolWrapper
    assert MCPToolWrapper._plugin_discoverable is False
    assert MCPResourceWrapper._plugin_discoverable is False
    assert MCPPromptWrapper._plugin_discoverable is False


# --- Task 10: Integration test ---


def test_loader_registers_same_tools_as_old_hardcoded(loader_context):
    """Verify the loader produces the same tool set as the old _register_default_tools."""
    from nanobot.agent.tools.loader import ToolLoader
    from nanobot.agent.tools.registry import ToolRegistry

    ctx = loader_context
    registry = ToolRegistry()
    loader = ToolLoader()
    registered = loader.load(ctx, registry)

    expected = {
        "read_file", "write_file", "edit_file", "list_dir",
        "exec", "exec_session", "list_exec_sessions",
        "web_search", "web_fetch",
        "message", "cron", "plan_task", "delegate_task", "review_work", "explore",
    }
    actual = set(registered)
    assert expected <= actual, f"Missing tools: {expected - actual}"
    assert actual == expected
    assert set(registry.tool_names) == expected
    assert not actual & {"find_files", "grep", "spawn", "spawn_pipeline", "my"}


def test_non_core_loader_scope_stays_closed_without_registry_mutation():
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.config.schema import ToolsConfig
    registry = ToolRegistry()
    ctx = ToolContext(config=ToolsConfig(), workspace="/tmp")
    with pytest.raises(ValueError, match="scope"):
        ToolLoader().load(ctx, registry, scope="subagent")
    assert registry.tool_names == []
