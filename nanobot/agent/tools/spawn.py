"""Spawn tool for creating background subagents."""

import contextvars
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
        self._origin_channel_var: contextvars.ContextVar[str] = contextvars.ContextVar(
            "spawn_tool_origin_channel",
            default=self._origin_channel,
        )
        self._origin_chat_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
            "spawn_tool_origin_chat_id",
            default=self._origin_chat_id,
        )
        self._session_key_var: contextvars.ContextVar[str] = contextvars.ContextVar(
            "spawn_tool_session_key",
            default=self._session_key,
        )

    def set_context(self, channel: str, chat_id: str, session_key: str | None = None) -> None:
        """Set the origin context for subagent announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id
        self._session_key = session_key or f"{channel}:{chat_id}"
        self._origin_channel_var.set(self._origin_channel)
        self._origin_chat_id_var.set(self._origin_chat_id)
        self._session_key_var.set(self._session_key)

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
            "If the background task needs to write workspace files, provide write_scope with "
            "workspace-relative paths or directory prefixes ending in '/'. "
            "If write_scope is omitted, the background subagent stays read-only. "
            "Set notify=true only when the user explicitly wants a later completion message. "
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
                "notify": {
                    "type": "boolean",
                    "description": (
                        "Whether to send a later completion message back to the user when the "
                        "background subagent finishes. Default: false. Only set this when the "
                        "user explicitly wants background completion updates."
                    ),
                },
                "write_scope": {
                    "type": "array",
                    "description": (
                        "Optional workspace-relative files or directory prefixes this background "
                        "task may write. Required for writable background work. Directory prefixes "
                        "must end with '/'. Leave unset for read-only background tasks."
                    ),
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["task"],
        }

    async def execute(
        self, task: str, label: str | None = None,
        goal: str | None = None, review: bool = False,
        notify: bool = False,
        write_scope: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Spawn a subagent to execute the given task."""
        return await self._manager.spawn(
            task=task,
            label=label,
            goal=goal,
            review=review,
            notify=notify,
            write_scope=write_scope,
            origin_channel=self._origin_channel_var.get(),
            origin_chat_id=self._origin_chat_id_var.get(),
            session_key=self._session_key_var.get(),
        )
