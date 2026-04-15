"""Workflow state management for agentic orchestration.

Manages workflow lifecycle via JSON plan files and markdown summaries
stored in the workspace. Designed for the hybrid supervisor pattern
where the main agent loads lightweight state (~500 tokens) per wake-up.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


class WorkflowState:
    """Manages workflow state via files in the workspace.

    State files:
        workspace/current/plan.json       - Active workflow plan + step statuses
        workspace/current/summary_index.md - Rolling phase summaries
        workspace/current/artifacts/       - Step-by-step detailed outputs
        workspace/output/                  - Final outputs for human review
        workspace/archive/                 - Completed workflow plans
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.current_dir = workspace / "workspace" / "current"
        self.artifacts_dir = self.current_dir / "artifacts"
        self.output_dir = workspace / "workspace" / "output"
        self.archive_dir = workspace / "workspace" / "archive"
        self.plan_file = self.current_dir / "plan.json"
        self.summary_file = self.current_dir / "summary_index.md"

    def _ensure_dirs(self) -> None:
        """Ensure all workspace directories exist."""
        for d in [self.current_dir, self.artifacts_dir, self.output_dir, self.archive_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_plan(self) -> dict[str, Any]:
        """Read current plan. Returns {"status": "idle"} if no plan exists."""
        if not self.plan_file.exists():
            return {"status": "idle"}
        try:
            return json.loads(self.plan_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read plan.json, returning idle state")
            return {"status": "idle"}

    def is_active(self) -> bool:
        """Check if there is an active workflow."""
        return self.get_plan().get("status") == "in_progress"

    def create_plan(
        self,
        workflow_id: str,
        goal: str,
        phases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a new workflow plan.

        Args:
            workflow_id: Short identifier (e.g., "wf-2026-03-12-seo-blog")
            goal: Human-readable goal description
            phases: List of phase dicts, each with "name" and "steps" keys.
                    Each step should have: role, skill, task, output_path, label.

        Returns:
            The created plan dict.
        """
        self._ensure_dirs()

        plan = {
            "id": workflow_id,
            "goal": goal,
            "status": "in_progress",
            "current_phase": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "phases": [],
        }

        for phase in phases:
            plan["phases"].append({
                "name": phase["name"],
                "status": "pending",
                "steps": phase.get("steps", []),
                "summary": None,
                "confidence": None,
            })

        self._write_plan(plan)

        # Reset summary index
        self.summary_file.write_text(
            "# Workflow Summaries\n\n"
            f"**Workflow**: {workflow_id}\n"
            f"**Goal**: {goal}\n\n---\n\n",
            encoding="utf-8",
        )

        logger.info("Created workflow plan: {} ({})", workflow_id, goal)
        return plan

    def update_phase(
        self,
        phase_idx: int,
        status: str,
        summary: str | None = None,
        confidence: int | None = None,
    ) -> dict[str, Any]:
        """Update a phase's status, summary, and confidence.

        Args:
            phase_idx: Index of the phase to update.
            status: New status ("pending", "in_progress", "completed", "failed").
            summary: Short summary text (2-3 sentences).
            confidence: Self-assessed confidence score (0-100).

        Returns:
            Updated plan dict.
        """
        plan = self.get_plan()
        if phase_idx >= len(plan.get("phases", [])):
            logger.error("Phase index {} out of range", phase_idx)
            return plan

        phase = plan["phases"][phase_idx]
        phase["status"] = status
        if summary is not None:
            phase["summary"] = summary
        if confidence is not None:
            phase["confidence"] = confidence

        plan["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Advance current_phase if this phase completed
        if status == "completed":
            plan["current_phase"] = phase_idx + 1
            # Append to summary index
            self._append_summary(phase["name"], summary, confidence)

        self._write_plan(plan)
        return plan

    def complete_workflow(self) -> dict[str, Any]:
        """Complete the active workflow: archive plan, clear artifacts, reset state.

        Returns:
            The archived plan dict.
        """
        plan = self.get_plan()
        if plan.get("status") == "idle":
            return plan

        plan["status"] = "completed"
        plan["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Archive the plan
        self._ensure_dirs()
        archive_name = f"{plan.get('id', 'unknown')}.json"
        archive_path = self.archive_dir / archive_name
        archive_path.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Archived workflow: {}", archive_name)

        # Clear artifacts
        if self.artifacts_dir.exists():
            for f in self.artifacts_dir.iterdir():
                if f.name != ".gitkeep":
                    if f.is_file():
                        f.unlink()
                    elif f.is_dir():
                        shutil.rmtree(f)

        # Reset plan to idle
        self._write_plan({"status": "idle"})

        # Reset summary index
        self.summary_file.write_text(
            "# Workflow Summaries\n\n"
            "Phase summaries for the active workflow appear here. Reset on new workflow.\n",
            encoding="utf-8",
        )

        return plan

    def get_compact_state(self) -> str:
        """Return a compact state string for supervisor context (~500 tokens).

        This is what the supervisor reads on every wake-up to understand
        the current workflow state without loading full artifacts.
        """
        plan = self.get_plan()

        if plan.get("status") == "idle":
            return "**Workflow Status**: idle — no active workflow.\n"

        lines = [
            f"**Workflow**: {plan.get('id', 'unknown')}",
            f"**Goal**: {plan.get('goal', 'unknown')}",
            f"**Status**: {plan.get('status', 'unknown')}",
            f"**Current Phase**: {plan.get('current_phase', 0)} / {len(plan.get('phases', []))}",
            "",
        ]

        for i, phase in enumerate(plan.get("phases", [])):
            marker = "→" if i == plan.get("current_phase", 0) else " "
            conf = f" (confidence: {phase['confidence']}%)" if phase.get("confidence") is not None else ""
            lines.append(f"{marker} Phase {i+1}: {phase['name']} — {phase['status']}{conf}")
            if phase.get("summary"):
                lines.append(f"  Summary: {phase['summary']}")

        # Append summary index content if it exists
        if self.summary_file.exists():
            summary_content = self.summary_file.read_text(encoding="utf-8").strip()
            # Only include if there's actual content beyond the header
            summary_lines = summary_content.split("\n")
            if len(summary_lines) > 4:  # More than just header
                lines.append("")
                lines.append("### Recent Summaries")
                lines.append(summary_content)

        return "\n".join(lines)

    def _write_plan(self, plan: dict[str, Any]) -> None:
        """Write plan dict to plan.json."""
        self._ensure_dirs()
        self.plan_file.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _append_summary(
        self,
        phase_name: str,
        summary: str | None,
        confidence: int | None,
    ) -> None:
        """Append a phase summary to summary_index.md."""
        if not summary:
            return
        self._ensure_dirs()

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        conf_str = f" | Confidence: {confidence}%" if confidence is not None else ""

        entry = (
            f"\n## {phase_name} — {timestamp}{conf_str}\n\n"
            f"{summary}\n\n---\n"
        )

        with open(self.summary_file, "a", encoding="utf-8") as f:
            f.write(entry)
