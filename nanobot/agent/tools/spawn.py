"""Spawn tool for creating background subagents."""

from typing import Any, TYPE_CHECKING

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SpawnTool(Tool):
    """
    Tool to spawn a subagent for background task execution.

    The subagent runs asynchronously and announces its result back
    to the main agent when complete.

    Supports optional role injection (loads a predefined role prompt
    from skills/roles/{role}.md) and skill hints (tells the subagent
    which SKILL.md to follow).
    """

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn a subagent to handle a task in the background. "
            "Use this for complex or time-consuming tasks that can run independently. "
            "The subagent will complete the task and report back when done. "
            "Optionally specify a role (loads predefined persona from skills/roles/) "
            "and a skill (tells subagent which SKILL.md to follow)."
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
                "role": {
                    "type": "string",
                    "description": (
                        "Optional role name. Loads the role prompt from "
                        "skills/roles/{role}.md into the subagent's system prompt. "
                        "Examples: 'researcher', 'copywriter', 'analyst', 'scriptwriter'"
                    ),
                },
                "skill": {
                    "type": "string",
                    "description": (
                        "Optional skill name. Tells the subagent to read and follow "
                        "skills/{skill}/SKILL.md. Examples: 'seo-writer', 'tiktok-marketing'"
                    ),
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        label: str | None = None,
        role: str | None = None,
        skill: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Spawn a subagent to execute the given task."""
        return await self._manager.spawn(
            task=task,
            label=label,
            role=role,
            skill=skill,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
        )
