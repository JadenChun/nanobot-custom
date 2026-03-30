"""Subagent manager for background task execution."""

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ExecToolConfig
from nanobot.providers.base import LLMProvider
from nanobot.utils.helpers import build_assistant_message


class SubagentManager:
    """Manages background subagent execution."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        web_search_config: "WebSearchConfig | None" = None,
        web_proxy: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
    ):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._session_tasks: dict[str, set[str]] = {}  # session_key -> {task_id, ...}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        goal: str | None = None,
        review: bool = False,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin, goal=goal, review=review)
        )
        self._running_tasks[task_id] = bg_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        def _cleanup(_: asyncio.Task) -> None:
            self._running_tasks.pop(task_id, None)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

        bg_task.add_done_callback(_cleanup)

        mode = " with review" if review else ""
        logger.info("Spawned subagent [{}]{}: {}", task_id, mode, display_label)
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    # ------------------------------------------------------------------
    # Shared agent loop
    # ------------------------------------------------------------------

    async def _run_agent_loop(
        self,
        task_id: str,
        messages: list[dict[str, Any]],
        tools: ToolRegistry,
        max_iterations: int = 15,
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Run an agent loop. Returns (final_result, messages)."""
        iteration = 0
        final_result: str | None = None

        while iteration < max_iterations:
            iteration += 1

            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tools.get_definitions(),
                model=self.model,
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    tc.to_openai_tool_call()
                    for tc in response.tool_calls
                ]
                messages.append(build_assistant_message(
                    response.content or "",
                    tool_calls=tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                ))

                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Subagent [{}] executing: {}({})", task_id, tool_call.name, args_str[:200])
                    result = await tools.execute(tool_call.name, tool_call.arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result,
                    })
            else:
                final_result = response.content
                break

        return final_result, messages

    # ------------------------------------------------------------------
    # Generation tools
    # ------------------------------------------------------------------

    def _build_generation_tools(self) -> ToolRegistry:
        """Build the full tool set for the generation agent."""
        tools = ToolRegistry()
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(EditFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        tools.register(WebFetchTool(proxy=self.web_proxy))
        return tools

    # ------------------------------------------------------------------
    # Review agent
    # ------------------------------------------------------------------

    def _build_review_tools(self) -> ToolRegistry:
        """Build a read-only tool set for the review agent."""
        tools = ToolRegistry()
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        tools.register(WebFetchTool(proxy=self.web_proxy))
        return tools

    def _build_review_prompt(self, goal: str) -> str:
        """Build a skeptical system prompt for the review agent."""
        from nanobot.agent.context import ContextBuilder

        time_ctx = ContextBuilder._build_runtime_context(None, None)

        return f"""# Review Agent

{time_ctx}

You are a review agent. Your job is to critically evaluate whether a generation agent's output meets the stated goal.

## Goal
{goal}

## Your Role
- Be skeptical. Assume the output may be incomplete, incorrect, or low quality until proven otherwise.
- Verify claims by reading output files, running tests, or checking structure.
- Use web search to research evaluation criteria if you are unsure how to assess quality for this type of task.
- Run linters, tests, or validation commands via exec when applicable.
- Be specific in your feedback — cite file paths, exact issues, and concrete suggestions.

## Rules
1. You CANNOT modify any files. You have read-only access plus exec for running tests/linters.
2. You MUST read the actual output files or artifacts to evaluate them — do not trust summaries alone.
3. Check for: completeness against the goal, correctness, quality, missing requirements, and potential issues.
4. If the output is genuinely good and meets the goal, approve it. Do not reject good work unnecessarily.

## Workspace
{self.workspace}

## Output Format
You MUST end your final response with exactly this JSON block:

---REVIEW---
{{"approved": true_or_false, "confidence": 0_to_100, "issues": ["issue1", "issue2"], "feedback": "Detailed actionable feedback for the generation agent"}}
---END---

If approved, set feedback to a brief confirmation. If not approved, feedback MUST contain specific, actionable instructions for what to fix."""

    @staticmethod
    def _parse_review(result: str) -> dict[str, Any]:
        """Extract the structured review JSON from review agent output."""
        match = re.search(r"---REVIEW---\s*\n(.*?)\n\s*---END---", result, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                return {
                    "approved": bool(data.get("approved", False)),
                    "confidence": int(data.get("confidence", 0)),
                    "issues": data.get("issues", []),
                    "feedback": str(data.get("feedback", "")),
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        # Fallback: treat as rejection so generation agent gets another chance
        return {
            "approved": False,
            "confidence": 0,
            "issues": ["Failed to parse review output"],
            "feedback": result[-500:] if result else "Review agent produced no output.",
        }

    async def _run_review_loop(
        self,
        task_id: str,
        goal: str,
        generation_result: str,
        generation_messages: list[dict[str, Any]],
        generation_tools: ToolRegistry,
        max_rounds: int = 3,
    ) -> tuple[str, dict[str, Any]]:
        """Run the generation-review loop. Returns (final_result, review_meta)."""
        review_tools = self._build_review_tools()
        review_prompt = self._build_review_prompt(goal)
        best_result = generation_result
        best_confidence = 0
        last_review: dict[str, Any] = {}

        for round_num in range(1, max_rounds + 1):
            logger.info("Review round {}/{} for subagent [{}]", round_num, max_rounds, task_id)

            # Build review messages (fresh context each round — reviewer shouldn't accumulate)
            review_messages: list[dict[str, Any]] = [
                {"role": "system", "content": review_prompt},
                {"role": "user", "content": (
                    "The generation agent has completed its task. "
                    f"Its final response was:\n\n{generation_result}\n\n"
                    "Please evaluate whether the output meets the goal. "
                    "Read any output files mentioned and verify the work."
                )},
            ]

            review_result, _ = await self._run_agent_loop(
                task_id=f"{task_id}-review-r{round_num}",
                messages=review_messages,
                tools=review_tools,
                max_iterations=15,
            )

            review = self._parse_review(review_result or "")
            last_review = review
            logger.info(
                "Review [{}] round {}: approved={}, confidence={}",
                task_id, round_num, review["approved"], review["confidence"],
            )

            # Track best effort
            if review["confidence"] > best_confidence:
                best_confidence = review["confidence"]
                best_result = generation_result

            if review["approved"]:
                return generation_result, review

            if round_num >= max_rounds:
                logger.warning("Max review rounds reached for [{}], accepting best effort", task_id)
                break

            # Append feedback to generation context and re-run
            logger.info("Review [{}]: feedback sent to generation agent", task_id)
            feedback_msg = (
                f"[Review Agent Feedback — Round {round_num}]\n"
                f"Issues: {'; '.join(review['issues'])}\n"
                f"Feedback: {review['feedback']}\n\n"
                f"Please revise your work to address these issues."
            )
            generation_messages.append({"role": "user", "content": feedback_msg})

            generation_result, generation_messages = await self._run_agent_loop(
                task_id=task_id,
                messages=generation_messages,
                tools=generation_tools,
                max_iterations=15,
            )
            generation_result = generation_result or "Revision completed but no final response."

        return best_result, last_review

    # ------------------------------------------------------------------
    # Subagent execution
    # ------------------------------------------------------------------

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        goal: str | None = None,
        review: bool = False,
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info("Subagent [{}] starting task: {}", task_id, label)

        try:
            tools = self._build_generation_tools()
            system_prompt = self._build_subagent_prompt()
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            final_result, messages = await self._run_agent_loop(
                task_id=task_id,
                messages=messages,
                tools=tools,
                max_iterations=15,
            )

            if final_result is None:
                final_result = "Task completed but no final response was generated."

            # Run review loop if requested
            review_meta: dict[str, Any] | None = None
            if review:
                effective_goal = goal or task
                try:
                    final_result, review_meta = await self._run_review_loop(
                        task_id=task_id,
                        goal=effective_goal,
                        generation_result=final_result,
                        generation_messages=messages,
                        generation_tools=tools,
                    )
                except Exception as e:
                    # Fail-open: accept generation result if review crashes
                    logger.error("Review loop failed for [{}], accepting generation result: {}", task_id, e)

            logger.info("Subagent [{}] completed successfully", task_id)
            await self._announce_result(task_id, label, task, final_result, origin, "ok", review_meta)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error("Subagent [{}] failed: {}", task_id, e)
            await self._announce_result(task_id, label, task, error_msg, origin, "error")

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
        review_meta: dict[str, Any] | None = None,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        review_note = ""
        if review_meta:
            if review_meta.get("approved"):
                review_note = f"\nReview: approved (confidence: {review_meta.get('confidence', '?')}%)"
            else:
                review_note = (
                    f"\nReview: best-effort (confidence: {review_meta.get('confidence', '?')}%)" 
                    f"\nReview issues: {'; '.join(review_meta.get('issues', []))}"
                )

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}{review_note}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        # Inject as system message to trigger main agent
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.info("Subagent [{}] announced result to {}:{}", task_id, origin['channel'], origin['chat_id'])

    def _build_subagent_prompt(self) -> str:
        """Build a focused system prompt for the subagent."""
        from nanobot.agent.context import ContextBuilder
        from nanobot.agent.skills import SkillsLoader

        time_ctx = ContextBuilder._build_runtime_context(None, None)
        parts = [f"""# Subagent

{time_ctx}

You are a subagent spawned by the main agent to complete a specific task.
Stay focused on the assigned task. Your final response will be reported back to the main agent.

## Workspace
{self.workspace}"""]

        skills_summary = SkillsLoader(self.workspace).build_skills_summary()
        if skills_summary:
            parts.append(f"## Skills\n\nRead SKILL.md with read_file to use a skill.\n\n{skills_summary}")

        return "\n\n".join(parts)

    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for the given session. Returns count cancelled."""
        tasks = [self._running_tasks[tid] for tid in self._session_tasks.get(session_key, [])
                 if tid in self._running_tasks and not self._running_tasks[tid].done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
