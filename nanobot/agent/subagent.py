"""Subagent manager for background task execution.

Supports role injection, skill hints, pipelines with fail-fast,
and structured summary output for the hybrid supervisor pattern.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.agent_browser import AgentBrowserTool
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool


class SubagentManager:
    """
    Manages background subagent execution.

    Subagents are lightweight agent instances that run in the background
    to handle specific tasks. They share the same LLM provider but have
    isolated context and a focused system prompt.

    Supports:
    - Role injection: load a role prompt from skills/roles/{role}.md into the system prompt
    - Skill hints: tell the subagent which SKILL.md to read and follow
    - Pipelines: run a sequence of subagent tasks with code-level routing and fail-fast
    - Structured summaries: subagents output confidence scores and summaries
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: "MaxTokensConfig | None" = None,
        brave_api_key: str | None = None,
        agent_browser_config: AgentBrowserConfig | None = None,
        exec_config: ExecToolConfig | None = None,
        restrict_to_workspace: bool = False,
    ):
        from nanobot.config.schema import AgentBrowserConfig, ExecToolConfig, MaxTokensConfig
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens or MaxTokensConfig()
        self.brave_api_key = brave_api_key
        self.agent_browser_config = agent_browser_config or AgentBrowserConfig()
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        role: str | None = None,
        skill: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        """
        Spawn a subagent to execute a task in the background.

        Args:
            task: The task description for the subagent.
            label: Optional human-readable label for the task.
            role: Optional role name — loads skills/roles/{role}.md into system prompt.
            skill: Optional skill name — tells subagent to read and follow skills/{skill}/SKILL.md.
            origin_channel: The channel to announce results to.
            origin_chat_id: The chat ID to announce results to.

        Returns:
            Status message indicating the subagent was started.
        """
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")

        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
        }

        # Create background task
        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin, role=role, skill=skill)
        )
        self._running_tasks[task_id] = bg_task

        # Cleanup when done
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(task_id, None))

        logger.info("Spawned subagent [{}]: {}", task_id, display_label)
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def spawn_pipeline(
        self,
        tasks: list[dict[str, Any]],
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        """
        Spawn a pipeline of sequential subagent tasks in the background.

        Tasks run sequentially with code-level routing (no LLM between steps).
        If any step fails or produces no output, the pipeline stops immediately
        and reports the failure to the main agent.

        Args:
            tasks: List of task dicts, each with keys:
                - task (str, required): Task description
                - role (str, optional): Role name for system prompt injection
                - skill (str, optional): Skill name for SKILL.md hint
                - output_path (str, optional): Expected output file (relative to workspace)
                - label (str, optional): Human-readable step label

        Returns:
            Status message indicating the pipeline was started.
        """
        pipeline_id = str(uuid.uuid4())[:8]
        step_labels = [t.get("label", f"Step {i+1}") for i, t in enumerate(tasks)]
        display = f"Pipeline [{' → '.join(step_labels)}]"

        origin = {
            "channel": origin_channel,
            "chat_id": origin_chat_id,
        }

        bg_task = asyncio.create_task(
            self._run_pipeline(pipeline_id, tasks, origin)
        )
        self._running_tasks[pipeline_id] = bg_task
        bg_task.add_done_callback(lambda _: self._running_tasks.pop(pipeline_id, None))

        logger.info("Spawned pipeline [{}]: {}", pipeline_id, display)
        return f"{display} started (id: {pipeline_id}). I'll notify you when all steps complete."

    async def _run_pipeline(
        self,
        pipeline_id: str,
        tasks: list[dict[str, Any]],
        origin: dict[str, str],
    ) -> None:
        """Execute a pipeline of sequential subagent tasks with fail-fast."""
        completed: list[dict[str, Any]] = []
        total = len(tasks)

        for i, step in enumerate(tasks):
            step_label = step.get("label", f"Step {i+1}")
            step_id = f"{pipeline_id}-s{i+1}"

            logger.info("Pipeline [{}] starting step {}/{}: {}", pipeline_id, i+1, total, step_label)

            try:
                # Run the subagent synchronously (within the pipeline's async task)
                result = await self._run_subagent_inline(
                    task_id=step_id,
                    task=step["task"],
                    label=step_label,
                    role=step.get("role"),
                    skill=step.get("skill"),
                )

                # Validate output file if specified
                output_path = step.get("output_path")
                if output_path:
                    full_path = self.workspace / output_path
                    if not full_path.exists():
                        raise ValueError(f"Expected output file not created: {output_path}")
                    if full_path.stat().st_size == 0:
                        raise ValueError(f"Output file is empty: {output_path}")

                # Parse summary from result
                summary_data = self._parse_summary(result)

                completed.append({
                    "step": i + 1,
                    "label": step_label,
                    "status": "ok",
                    "summary": summary_data.get("summary", ""),
                    "confidence": summary_data.get("confidence"),
                    "output_path": output_path,
                })

                logger.info("Pipeline [{}] step {}/{} completed: {}", pipeline_id, i+1, total, step_label)

            except Exception as e:
                logger.error("Pipeline [{}] step {}/{} failed: {}", pipeline_id, i+1, total, e)

                # Fail-fast: announce failure with context
                remaining = [
                    {"step": j + i + 2, "label": tasks[j + i + 1].get("label", f"Step {j+i+2}")}
                    for j in range(total - i - 1)
                ]
                await self._announce_pipeline_result(
                    pipeline_id=pipeline_id,
                    completed=completed,
                    failed={"step": i + 1, "label": step_label, "error": str(e)},
                    remaining=remaining,
                    origin=origin,
                )
                return

        # All steps succeeded
        await self._announce_pipeline_result(
            pipeline_id=pipeline_id,
            completed=completed,
            failed=None,
            remaining=[],
            origin=origin,
        )

    async def _run_subagent_inline(
        self,
        task_id: str,
        task: str,
        label: str,
        role: str | None = None,
        skill: str | None = None,
    ) -> str:
        """Run a subagent synchronously (for pipeline use). Returns the final result text."""
        tools = self._build_tools()

        # Load role content if specified
        role_content = self._load_role(role) if role else None

        system_prompt = self._build_subagent_prompt(
            task, role_content=role_content, skill_name=skill,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        max_iterations = 25
        iteration = 0
        final_result: str | None = None

        while iteration < max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens.output,
            )

            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in response.tool_calls
                ]
                messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": tool_call_dicts,
                })

                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.debug("Subagent [{}] executing: {} with arguments: {}", task_id, tool_call.name, args_str)
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

        return final_result or "Task completed but no final response was generated."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
        role: str | None = None,
        skill: str | None = None,
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info("Subagent [{}] starting task: {}", task_id, label)

        try:
            final_result = await self._run_subagent_inline(
                task_id=task_id, task=task, label=label, role=role, skill=skill,
            )

            logger.info("Subagent [{}] completed successfully", task_id)
            await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error("Subagent [{}] failed: {}", task_id, e)
            await self._announce_result(task_id, label, task, error_msg, origin, "error")

    def _build_tools(self) -> ToolRegistry:
        """Build the tool registry for a subagent."""
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
        ))
        tools.register(WebSearchTool(api_key=self.brave_api_key))
        tools.register(WebFetchTool())
        if self.agent_browser_config.enabled:
            tools.register(AgentBrowserTool(
                package=self.agent_browser_config.package,
                timeout=self.agent_browser_config.timeout,
                max_output_chars=self.agent_browser_config.max_output_chars,
                working_dir=str(self.workspace),
            ))
        return tools

    def _load_role(self, role: str) -> str | None:
        """Load a role prompt from skills/roles/{role}.md."""
        role_file = self.workspace / "skills" / "roles" / f"{role}.md"
        if role_file.exists():
            content = role_file.read_text(encoding="utf-8")
            logger.debug("Loaded role prompt: {}", role)
            return content
        logger.warning("Role file not found: {}", role_file)
        return None

    def _parse_summary(self, result: str) -> dict[str, Any]:
        """Parse structured summary from subagent result.

        Looks for a block like:
            ---SUMMARY---
            confidence: 85
            summary: Did X and found Y.
            output_files: [file1.md, file2.md]
            ---END---
        """
        data: dict[str, Any] = {}
        match = re.search(
            r"---SUMMARY---\s*\n(.*?)\n\s*---END---",
            result,
            re.DOTALL,
        )
        if match:
            block = match.group(1)
            for line in block.strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "confidence":
                        try:
                            data["confidence"] = int(value)
                        except ValueError:
                            pass
                    elif key == "summary":
                        data["summary"] = value
                    elif key == "output_files":
                        data["output_files"] = [f.strip() for f in value.strip("[]").split(",")]
        else:
            # Fallback: use the last 200 chars as summary
            data["summary"] = result[-200:].strip() if len(result) > 200 else result.strip()

        return data

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        # Parse summary for compact announcement
        summary_data = self._parse_summary(result)
        summary_text = summary_data.get("summary", result[:500])
        confidence = summary_data.get("confidence")
        conf_str = f"\nConfidence: {confidence}%" if confidence is not None else ""

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Summary: {summary_text}{conf_str}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        # Inject as system message to trigger main agent
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug("Subagent [{}] announced result to {}:{}", task_id, origin['channel'], origin['chat_id'])

    async def _announce_pipeline_result(
        self,
        pipeline_id: str,
        completed: list[dict[str, Any]],
        failed: dict[str, Any] | None,
        remaining: list[dict[str, Any]],
        origin: dict[str, str],
    ) -> None:
        """Announce pipeline result — either all-success or failure with context."""
        if failed is None:
            # All steps succeeded
            status_text = "completed successfully"
            steps_summary = "\n".join(
                f"  - Step {s['step']} ({s['label']}): {s.get('summary', 'done')}"
                + (f" [confidence: {s['confidence']}%]" if s.get('confidence') is not None else "")
                for s in completed
            )
            content = f"""[Pipeline {status_text} — {len(completed)} steps]

{steps_summary}

Review the output files and proceed with the workflow (git commit, notify user, etc.)."""
        else:
            # Failure
            completed_text = "\n".join(
                f"  - Step {s['step']} ({s['label']}): completed"
                for s in completed
            ) or "  (none)"
            remaining_text = "\n".join(
                f"  - Step {s['step']} ({s['label']}): not started"
                for s in remaining
            ) or "  (none)"

            content = f"""[Pipeline FAILED at step {failed['step']}]

Failed step: {failed['label']}
Error: {failed['error']}

Completed steps:
{completed_text}

Remaining steps (not executed):
{remaining_text}

Decide how to proceed: retry the failed step with adjusted instructions, skip it, or abort the workflow."""

        msg = InboundMessage(
            channel="system",
            sender_id="pipeline",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=content,
        )
        await self.bus.publish_inbound(msg)
        logger.debug("Pipeline [{}] announced result", pipeline_id)

    def _build_subagent_prompt(
        self,
        task: str,
        role_content: str | None = None,
        skill_name: str | None = None,
    ) -> str:
        """Build a focused system prompt for the subagent.

        Args:
            task: The task description.
            role_content: Optional role prompt content to inject.
            skill_name: Optional skill name — adds instruction to read SKILL.md.
        """
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"

        prompt = f"""# Subagent

## Current Time
{now} ({tz})

You are a subagent spawned by the main agent to complete a specific task.

## Rules
1. Stay focused - complete only the assigned task, nothing else
2. Your final response will be reported back to the main agent
3. Do not initiate conversations or take on side tasks
4. Be concise but informative in your findings

## What You Can Do
- Read and write files in the workspace
- Execute shell commands
- Search the web and fetch web pages
- Use agent_browser for advanced browser/electron automation
- Complete the task thoroughly

## What You Cannot Do
- Send messages directly to users (no message tool available)
- Spawn other subagents
- Access the main agent's conversation history

## Workspace
Your workspace is at: {self.workspace}
Skills are available at: {self.workspace}/skills/ (read SKILL.md files as needed)"""

        # Inject role if provided
        if role_content:
            prompt += f"\n\n## Your Role\n{role_content}"

        # Inject skill hint if provided
        if skill_name:
            prompt += (
                f"\n\n## Primary Skill\n"
                f"Read and follow the workflow in: {self.workspace}/skills/{skill_name}/SKILL.md\n"
                f"Reference files are in: {self.workspace}/skills/{skill_name}/references/"
            )

        # Structured summary output format
        prompt += """

## Output Format

When you have completed the task, end your response with a structured summary block:

---SUMMARY---
confidence: [0-100, how confident you are in the quality of your output]
summary: [2-3 sentence summary of what you did and key findings/outputs]
output_files: [comma-separated list of files you created or modified]
---END---"""

        return prompt

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
