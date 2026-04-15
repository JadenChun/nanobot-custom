"""Planner-only explore tool backed by foreground read-only subagents."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool


class ExploreTool(Tool):
    """Run a focused read-only exploration pass in separate context."""

    name = "explore"
    description = (
        "Launch a foreground read-only explore subagent for broad or repeated investigation "
        "without filling the planner context with raw search history. Use this only when "
        "simple direct read/search tools are insufficient."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Specific research task for the explore subagent.",
                "minLength": 3,
            },
            "thoroughness": {
                "type": "string",
                "description": "How deep the explore pass should search before reporting back.",
                "enum": ["quick", "medium", "deep"],
                "default": "medium",
            },
        },
        "required": ["task"],
        "additionalProperties": False,
    }

    def __init__(self, manager: Any, *, max_iterations: int) -> None:
        self._manager = manager
        self._max_iterations = max_iterations

    async def execute(
        self,
        task: str,
        thoroughness: str = "medium",
        **_: Any,
    ) -> str:
        result = await self._manager.run_explore(
            task=task,
            thoroughness=thoroughness,
            max_iterations=self._max_iterations,
        )

        lines: list[str] = ["[Explore Findings]"]
        if result.get("partial"):
            reason = result.get("stop_reason") or "loop guard"
            lines.append(f"Status: partial ({reason})")
        if summary := result.get("summary"):
            lines.extend(["Summary:", summary])
        findings = result.get("findings") or []
        if findings:
            lines.append("Findings:")
            lines.extend(f"- {item}" for item in findings)
        references = result.get("references") or []
        if references:
            lines.append("References:")
            lines.extend(f"- {item}" for item in references)
        open_questions = result.get("open_questions") or []
        if open_questions:
            lines.append("Open Questions:")
            lines.extend(f"- {item}" for item in open_questions)
        searched_areas = result.get("searched_areas") or []
        if searched_areas:
            lines.append("Searched Areas:")
            lines.extend(f"- {item}" for item in searched_areas)
        return "\n".join(lines)
