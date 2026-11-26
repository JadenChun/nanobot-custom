"""Foreground role tools exposed to the main orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kohaku.agent.tools.base import Tool
from kohaku.agent.turn import ToolOutcome, TurnContext
from kohaku.harness.claude_sdk import child_harness_parameter
from kohaku.session.harness_binding import ChatHarness

if TYPE_CHECKING:
    from kohaku.agent.delegation import ForegroundAgentManager


class _ForegroundTool(Tool):
    exposure = "deferred"
    namespace = "delegation"

    _tool_name: str
    _tool_description: str
    _tool_parameters: dict[str, Any]
    _has_harness = True

    def __init__(self, manager: "ForegroundAgentManager") -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return self._tool_name

    @property
    def description(self) -> str:
        return self._tool_description

    @property
    def parameters(self) -> dict[str, Any]:
        if not self._has_harness:
            return self._tool_parameters
        return {
            **self._tool_parameters,
            "properties": {
                **self._tool_parameters["properties"], "harness": child_harness_parameter(),
            },
        }

    @property
    def supports_parallel_calls(self) -> bool:
        return False


_ARTIFACT_REFERENCES = {
    "type": "array",
    "items": {"type": "string", "pattern": "^artifact:[0-9a-f]{32}$"},
    "maxItems": 16,
    "uniqueItems": True,
}


class PlanTaskTool(_ForegroundTool):
    _tool_name = "plan_task"
    search_keywords = ("plan", "architecture", "design", "claude_sdk", "coding")
    _tool_description = (
        "Ask a foreground read-only planner for an implementation-ready plan when sequencing, "
        "dependencies, or ambiguity make direct work unreliable. Waits for the plan. Do not use "
        "for simple or already-clear tasks."
    )
    _tool_parameters = {
        "type": "object",
        "properties": {
            "objective": {"type": "string", "minLength": 3},
            "context": {
                "type": "string",
                "description": "Plain-text evidence already gathered; keep raw values and references.",
            },
            "artifact_references": _ARTIFACT_REFERENCES,
        },
        "required": ["objective"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        objective: str,
        context: str = "",
        turn_context: TurnContext | None = None,
        artifact_references: list[str] | None = None,
        harness: ChatHarness = "native",
        **_: Any,
    ) -> str | ToolOutcome:
        return await self._manager.run_plan(
            objective=objective,
            context=context,
            turn_context=turn_context,
            artifact_references=artifact_references,
            harness=harness,
        )

    async def execute_with_context(self, turn_context: TurnContext, /, **kwargs: Any) -> str | ToolOutcome:
        return await self.execute(turn_context=turn_context, **kwargs)


class DelegateTaskTool(_ForegroundTool):
    _tool_name = "delegate_task"
    search_keywords = ("delegate", "implement", "code", "coding", "claude_sdk", "claude", "worker")
    _tool_description = (
        "Run one bounded implementation or artifact task in a foreground worker and wait for its "
        "completion report. Provide a self-contained Markdown contract and explicit writable "
        "workspace paths. Use direct tools when isolated worker context adds no value."
    )
    _tool_parameters = {
        "type": "object",
        "properties": {
            "contract": {
                "type": "string",
                "description": "Markdown task contract with objective, evidence, acceptance criteria, non-goals, and validation.",
                "minLength": 20,
            },
            "write_scope": {
                "type": "array",
                "description": "Workspace-relative files or directory prefixes ending in a slash.",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
            "artifact_references": _ARTIFACT_REFERENCES,
            "continue_from_child_id": {
                "type": "string",
                "minLength": 1,
                "description": "Fork a completed Claude Code child from this chat into a new worker.",
            },
            "background": {
                "type": "boolean",
                "description": (
                    "Start the child and return immediately. The child continues after this "
                    "turn under its admitted scope. Defaults to false to wait for the result."
                ),
                "default": False,
            },
        },
        "required": ["contract", "write_scope"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        contract: str,
        write_scope: list[str],
        turn_context: TurnContext | None = None,
        artifact_references: list[str] | None = None,
        harness: ChatHarness = "native",
        background: bool = False,
        continue_from_child_id: str | None = None,
        **_: Any,
    ) -> str | ToolOutcome:
        return await self._manager.run_worker(
            contract=contract,
            write_scope=write_scope,
            turn_context=turn_context,
            artifact_references=artifact_references,
            harness=harness,
            background=background,
            continue_from_child_id=continue_from_child_id,
        )

    async def execute_with_context(self, context: TurnContext, /, **kwargs: Any) -> str | ToolOutcome:
        return await self.execute(turn_context=context, **kwargs)


class FollowupChildTool(_ForegroundTool):
    _has_harness = False
    _tool_name = "followup_child"
    _tool_description = (
        "Send an ordered follow-up to one canonical child owned by this chat. "
        "Use the child_id from its delegated result."
    )
    _tool_parameters = {
        "type": "object",
        "properties": {
            "child_id": {"type": "string", "minLength": 1},
            "content": {"type": "string", "minLength": 1},
        },
        "required": ["child_id", "content"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        child_id: str,
        content: str,
        turn_context: TurnContext | None = None,
        **_: Any,
    ) -> str | ToolOutcome:
        return await self._manager.followup_child(
            child_id=child_id,
            content=content,
            turn_context=turn_context,
        )

    async def execute_with_context(self, context: TurnContext, /, **kwargs: Any) -> str | ToolOutcome:
        return await self.execute(turn_context=context, **kwargs)


class ReviewWorkTool(_ForegroundTool):
    _tool_name = "review_work"
    search_keywords = ("review", "audit", "verify", "claude_sdk")
    _tool_description = (
        "Run a foreground independent read-only review and wait for PASS, CORRECT, or REJECT "
        "evidence. Use for important deliverables, risky changes, uncertain results, or explicit "
        "quality requests; do not review every routine task."
    )
    _tool_parameters = {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "minLength": 3},
            "acceptance_criteria": {"type": "string", "minLength": 3},
            "evidence": {"type": "string"},
            "relevant_paths": {
                "type": "array",
                "items": {"type": "string"},
            },
            "artifact_references": _ARTIFACT_REFERENCES,
        },
        "required": ["goal", "acceptance_criteria"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        goal: str,
        acceptance_criteria: str,
        evidence: str = "",
        relevant_paths: list[str] | None = None,
        turn_context: TurnContext | None = None,
        artifact_references: list[str] | None = None,
        harness: ChatHarness = "native",
        **_: Any,
    ) -> str | ToolOutcome:
        return await self._manager.run_review(
            goal=goal,
            acceptance_criteria=acceptance_criteria,
            evidence=evidence,
            relevant_paths=relevant_paths,
            turn_context=turn_context,
            artifact_references=artifact_references,
            harness=harness,
        )

    async def execute_with_context(self, context: TurnContext, /, **kwargs: Any) -> str | ToolOutcome:
        return await self.execute(turn_context=context, **kwargs)


__all__ = ["DelegateTaskTool", "FollowupChildTool", "PlanTaskTool", "ReviewWorkTool"]
