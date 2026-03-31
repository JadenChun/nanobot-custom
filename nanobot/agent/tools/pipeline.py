"""Pipeline tool for sequential subagent execution with fail-fast."""

from typing import Any, TYPE_CHECKING

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SpawnPipelineTool(Tool):
    """
    Spawn a pipeline of sequential subagent tasks.

    Tasks run one after another with code-level routing (no LLM calls
    between steps). If any step fails or produces no output file, the
    pipeline stops immediately and reports the failure with context
    about completed and remaining steps.

    This is more token-efficient than spawning individual subagents
    for sequential workflows, as the main agent only gets called once
    at the end (or on failure).
    """

    def __init__(self, manager: "SubagentManager"):
        self._manager = manager
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the origin context for pipeline announcements."""
        self._origin_channel = channel
        self._origin_chat_id = chat_id

    @property
    def name(self) -> str:
        return "spawn_pipeline"

    @property
    def description(self) -> str:
        return (
            "Spawn a pipeline of sequential subagent tasks. Tasks run one after "
            "another automatically — no LLM calls between steps. If any step fails "
            "or its expected output file is missing/empty, the pipeline stops and "
            "reports which steps completed, which failed, and which remain. "
            "Use for known-linear workflows like 'research → write → review'. "
            "Keep pipelines short (2-4 steps max)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "Sequential tasks. Each runs after the previous completes.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "Task instructions for this step",
                            },
                            "role": {
                                "type": "string",
                                "description": (
                                    "Role name — loads skills/roles/{role}.md into subagent prompt. "
                                    "Examples: 'researcher', 'copywriter', 'analyst', 'scriptwriter'"
                                ),
                            },
                            "skill": {
                                "type": "string",
                                "description": (
                                    "Skill name — tells subagent to follow skills/{skill}/SKILL.md. "
                                    "Examples: 'seo-writer', 'tiktok-marketing'"
                                ),
                            },
                            "output_path": {
                                "type": "string",
                                "description": (
                                    "Expected output file path (relative to workspace). "
                                    "Pipeline validates this file exists and is non-empty after the step."
                                ),
                            },
                            "label": {
                                "type": "string",
                                "description": "Human-readable label for this step",
                            },
                        },
                        "required": ["task"],
                    },
                    "minItems": 1,
                    "maxItems": 4,
                },
            },
            "required": ["tasks"],
        }

    async def execute(self, tasks: list[dict[str, Any]], **kwargs: Any) -> str:
        """Spawn the pipeline."""
        return await self._manager.spawn_pipeline(
            tasks=tasks,
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
        )
