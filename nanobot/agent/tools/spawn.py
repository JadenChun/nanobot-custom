"""Spawn tool for creating background subagents."""

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """Tool to spawn a subagent for background task execution."""

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "For important tasks (research, writing, coding), set review=true with a goal "
            "to enable quality validation - a separate review agent will evaluate the output "
            "and request revisions if needed. "
            "The subagent will complete the task and report back when done. "
            "For deliverables or existing projects, inspect the workspace first "
            "and use a dedicated subdirectory when helpful."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task for the subagent to complete",
                },
                "label": {
                    "type": "string",
                    "description": "Optional short label for the task (for display)",
                },
                "goal": {
                    "type": "string",
                    "description": (
                        "The high-level goal that defines what success looks like. "
                        "Used by the review agent to evaluate the generation output. "
                        "Required when review is true."
                    ),
                },
                "review": {
                    "type": "boolean",
                    "description": (
                        "Enable generation-review validation loop. "
                        "Use for research, content creation, complex multi-step work, "
                        "or any task where output quality matters. "
                        "Simple lookups and quick tasks don't need this. Default: false."
                    ),
                },
            },
            "required": ["task"],
        }

    async def execute(
        self, task: str, label: str | None = None,
        goal: str | None = None, review: bool = False,
        **kwargs: Any,
    ) -> str:
        """Spawn a subagent to execute the given task."""
        return await self._manager.spawn(
            task=task,
            label=label,
            goal=goal,
            review=review,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
            session_key=self._session_key,
        )
