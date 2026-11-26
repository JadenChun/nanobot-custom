"""Deferred tool exposure keeps the turn-1 schema small and cache-stable."""

from __future__ import annotations

import asyncio
from typing import Any

from kohaku.agent.tools.base import Tool
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.context.tool_search import ToolSearchTool
from kohaku.context.tool_surface import ToolSurface, ToolSurfaceState


class _Tool(Tool):
    def __init__(self, name: str, description: str, *, namespace: str = "core",
                 exposure: str = "direct", keywords: tuple[str, ...] = ()) -> None:
        self._name = name
        self._description = description
        self.namespace = namespace
        self.exposure = exposure
        self.search_keywords = keywords

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"path": {"type": "string"}}}

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_Tool("read_file", "Read a file"))
    registry.register(_Tool("write_file", "Write a file"))
    for index in range(20):
        registry.register(_Tool(
            f"mcp_github_op{index}", f"GitHub operation {index}",
            namespace="mcp:github", exposure="deferred", keywords=("github", "issue"),
        ))
    return registry


def _names(surface: ToolSurface) -> list[str]:
    return [definition["function"]["name"] for definition in surface.definitions_for_provider()]


def test_deferred_tools_are_absent_from_the_initial_schema() -> None:
    surface = ToolSurface(_registry())

    assert surface.direct_names() == ("read_file", "write_file")
    assert _names(surface) == ["read_file", "write_file"]


def test_namespace_index_summarizes_withheld_tools() -> None:
    surface = ToolSurface(_registry())

    lines = surface.namespace_index()

    assert len(lines) == 1
    assert lines[0].startswith("- mcp:github: ")
    assert "20 tools" in lines[0]
    assert all(tool.name not in lines[0] for tool in surface.deferred_tools())


def test_search_activates_only_matching_tools_and_appends_them() -> None:
    surface = ToolSurface(_registry())

    activated = surface.search("github issue", limit=3)
    surface.state = surface.state.with_activated([tool.name for tool in activated])

    assert len(activated) == 3
    names = _names(surface)
    # The direct prefix is unchanged; activated tools join the tail only.
    assert names[:2] == ["read_file", "write_file"]
    assert set(names[2:]) == {tool.name for tool in activated}


def test_search_prefers_an_exact_tool_name() -> None:
    surface = ToolSurface(_registry())

    matches = surface.search("mcp_github_op7", limit=1)

    assert [tool.name for tool in matches] == ["mcp_github_op7"]


def test_search_supports_namespace_only_queries() -> None:
    surface = ToolSurface(_registry())

    matches = surface.search("", namespace="mcp:github", limit=4)

    assert len(matches) == 4
    assert {tool.namespace for tool in matches} == {"mcp:github"}


def test_search_returns_nothing_for_unknown_namespace() -> None:
    surface = ToolSurface(_registry())

    assert surface.search("query", namespace="mcp:absent") == []


def test_activation_state_is_append_only_and_deduplicated() -> None:
    state = ToolSurfaceState()

    state = state.with_activated(["a", "b"])
    state = state.with_activated(["b", "c"])
    state = state.with_activated(["a"])

    assert state.activated == ("a", "b", "c")


def test_tool_search_activates_through_the_tool_call_edge() -> None:
    surface = ToolSurface(_registry())
    tool = ToolSearchTool(surface)

    result = asyncio.run(tool.execute(query="github issue", limit=2))

    assert not str(result).startswith("No deferred tools")
    assert "Activated 2 tool(s)" in str(result)
    assert len(surface.state.activated) == 2


def test_tool_search_reports_no_match_without_activating() -> None:
    surface = ToolSurface(_registry())
    tool = ToolSearchTool(surface)

    result = asyncio.run(tool.execute(query="kubernetes cluster"))

    assert "No deferred tools matched" in str(result)
    assert surface.state.activated == ()


def test_tool_search_without_a_surface_is_an_error() -> None:
    result = asyncio.run(ToolSearchTool().execute(query="anything"))

    assert result.is_error
