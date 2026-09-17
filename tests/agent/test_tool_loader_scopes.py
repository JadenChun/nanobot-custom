from unittest.mock import patch

import pytest

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.loader import ToolLoader
from nanobot.config.schema import ToolsConfig


class _CoreOnlyTool(Tool):
    _scopes = {"core"}

    @property
    def name(self):
        return "core_only"

    @property
    def description(self):
        return "..."

    @property
    def parameters(self):
        return {"type": "object"}

    async def execute(self, **_):
        return "ok"


class _SubagentOnlyTool(Tool):
    _scopes = {"subagent"}

    @property
    def name(self):
        return "sub_only"

    @property
    def description(self):
        return "..."

    @property
    def parameters(self):
        return {"type": "object"}

    async def execute(self, **_):
        return "ok"


class _UniversalTool(Tool):
    _scopes = {"core", "subagent", "memory"}

    @property
    def name(self):
        return "universal"

    @property
    def description(self):
        return "..."

    @property
    def parameters(self):
        return {"type": "object"}

    async def execute(self, **_):
        return "ok"


@pytest.mark.asyncio
async def test_loader_filters_by_scope():
    from nanobot.agent.tools.registry import ToolRegistry

    loader = ToolLoader(test_classes=[_CoreOnlyTool, _SubagentOnlyTool, _UniversalTool])

    registry = ToolRegistry()
    loader._discover_plugins = lambda: {
        "core": _CoreOnlyTool,
        "subagent": _SubagentOnlyTool,
        "universal": _UniversalTool,
    }
    ctx = ToolContext(config=ToolsConfig(), workspace="/tmp")
    with (
        patch("nanobot.migration_guard.require_runtime_ready"),
        patch("nanobot.agent.tools.fork_factories.construct_core_tools", return_value=[]),
    ):
        loader.load(ctx, registry, scope="core")

    assert registry.has("core_only")
    assert not registry.has("sub_only")
    assert registry.has("universal")
