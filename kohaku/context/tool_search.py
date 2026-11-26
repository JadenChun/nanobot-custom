"""Discover deferred tools so their schemas join the next model request."""

from __future__ import annotations

from typing import Any

from kohaku.agent.tools.base import Tool, ToolResult
from kohaku.context.tool_surface import ToolSurface

MIN_RESULTS = 1
MAX_RESULTS = 16


class ToolSearchTool(Tool):
    """Rank withheld tools and activate the best matches for this session."""

    exposure = "direct"
    namespace = "core"

    def __init__(self, surface: ToolSurface | None = None) -> None:
        self._surface = surface

    @property
    def name(self) -> str:
        return "tool_search"

    @property
    def description(self) -> str:
        return (
            "Find and activate deferred tools by keyword, exact name, or namespace."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Keywords, an exact tool name, or an empty string to list a namespace."
                    ),
                },
                "namespace": {
                    "type": "string",
                    "description": "Restrict the search to one tool namespace.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum tools to activate (1-16).",
                },
            },
            "required": ["query"],
        }

    @property
    def read_only(self) -> bool:
        return True

    def bind_surface(self, surface: ToolSurface) -> None:
        """Attach the live surface owned by the current session."""
        self._surface = surface

    def search(self, surface: ToolSurface, query: str, *,
               namespace: str | None, limit: int) -> ToolResult:
        """Return a compact activation report for the matched tools."""
        matches = surface.search(query, namespace=namespace, limit=limit)
        if not matches:
            hint = f' in namespace "{namespace}"' if namespace else ""
            return ToolResult(
                f"No deferred tools matched {query!r}{hint}. "
                "Try other keywords or an empty query with a namespace."
            )
        before = len(surface.state.activated)
        surface.activate([tool.name for tool in matches])
        added = len(surface.state.activated) - before
        lines = [f"Activated {added} tool(s). Matched schemas load on the next call:"]
        for tool in matches:
            summary = " ".join(tool.description.split())
            lines.append(f"- {tool.name} ({tool.namespace}): {summary[:200]}")
        return ToolResult("\n".join(lines))

    async def execute(self, query: str, namespace: str | None = None,
                      limit: int = 8, **kwargs: Any) -> ToolResult:
        surface = self._surface
        if surface is None:
            return ToolResult.error("Error: tool search is unavailable in this session")
        clamped = max(MIN_RESULTS, min(MAX_RESULTS, limit))
        return self.search(surface, query, namespace=namespace, limit=clamped)
