"""Internal execution policy for the main agent harness."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanobot.agent.tools.filesystem import _find_match
from nanobot.providers.base import ToolCallRequest


_APPROVAL_REQUIRED = "approval_required"


@dataclass(slots=True)
class ToolPolicyDecision:
    """Decision returned before executing a tool batch."""

    action: str = "allow"
    stop_reason: str | None = None
    response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolPolicy:
    """Base policy surface for tool-batch interception."""

    async def evaluate(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_calls: list[ToolCallRequest],
    ) -> ToolPolicyDecision:
        return ToolPolicyDecision()


@dataclass(slots=True)
class RiskyActionPolicy(ToolPolicy):
    """Require approval for risky or hard-to-undo tool batches."""

    workspace: Path
    approval_granted: bool = False

    _RISKY_EXEC_PATTERNS = (
        (re.compile(r"\brm\s+-[rf]{1,2}\b"), "delete files or directories"),
        (re.compile(r"\bgit\s+reset\b"), "reset git history"),
        (re.compile(r"\bgit\s+clean\b"), "remove untracked files"),
        (re.compile(r"\bgit\s+checkout\b[^|;&\n]*\s+-f\b"), "force checkout changes"),
        (re.compile(r"\bdrop\s+table\b"), "drop database tables"),
        (re.compile(r"\bdelete\s+from\b"), "delete database rows"),
        (re.compile(r"\btruncate\s+table\b"), "truncate database tables"),
        (re.compile(r"\b(kubectl|helm)\b[^|;&\n]*\b(apply|delete|upgrade|rollback)\b"), "change deployed infrastructure"),
        (re.compile(r"\b(npm|pnpm|yarn|bun)\b[^|;&\n]*\bpublish\b"), "publish a package"),
    )

    def mutating_tool_names(self) -> set[str]:
        return {"write_file", "edit_file", "exec", "cron"}

    async def evaluate(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_calls: list[ToolCallRequest],
    ) -> ToolPolicyDecision:
        if self.approval_granted or not tool_calls:
            return ToolPolicyDecision()

        reasons: list[str] = []
        for tool_call in tool_calls:
            reason = self._risky_reason(tool_call)
            if reason:
                reasons.append(reason)

        if not reasons:
            return ToolPolicyDecision()

        summary = "; ".join(dict.fromkeys(reasons))
        response = (
            "I'm about to take a risky action and want your approval first. "
            f"Planned action: {summary}. Reply `yes` to continue or `no` to cancel."
        )
        return ToolPolicyDecision(
            action="respond",
            stop_reason=_APPROVAL_REQUIRED,
            response=response,
            metadata={"reasons": reasons, "summary": summary},
        )

    def batch_has_mutation(self, tool_calls: list[ToolCallRequest]) -> bool:
        return any(tc.name in self.mutating_tool_names() for tc in tool_calls)

    def _risky_reason(self, tool_call: ToolCallRequest) -> str | None:
        if tool_call.name == "exec":
            return self._risky_exec_reason(str(tool_call.arguments.get("command") or ""))
        if tool_call.name == "write_file":
            return self._risky_write_reason(
                str(tool_call.arguments.get("path") or ""),
                str(tool_call.arguments.get("content") or ""),
            )
        if tool_call.name == "edit_file":
            return self._risky_edit_reason(
                str(tool_call.arguments.get("path") or ""),
                str(tool_call.arguments.get("old_text") or ""),
                str(tool_call.arguments.get("new_text") or ""),
                bool(tool_call.arguments.get("replace_all")),
            )
        if tool_call.name == "cron":
            return "schedule a recurring automation"
        return None

    def _risky_exec_reason(self, command: str) -> str | None:
        lower = command.strip().lower()
        if not lower:
            return None
        for pattern, label in self._RISKY_EXEC_PATTERNS:
            if pattern.search(lower):
                return label
        return None

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return candidate.resolve()

    def _changed_line_count(self, before: str, after: str) -> tuple[int, int]:
        before_lines = before.splitlines()
        after_lines = after.splitlines()
        matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines)
        changed = 0
        total = max(len(before_lines), len(after_lines), 1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                changed += max(i2 - i1, j2 - j1)
        return changed, total

    def _is_agent_state_file(self, raw_path: str) -> bool:
        """Markdown and memory files the agent manages freely — no size checks."""
        p = Path(raw_path)
        return p.suffix == ".md" or "memory" in p.parts

    def _risky_write_reason(self, raw_path: str, new_content: str) -> str | None:
        if not raw_path:
            return None
        if self._is_agent_state_file(raw_path):
            return None
        path = self._resolve_workspace_path(raw_path)
        if not path.exists() or not path.is_file():
            return None
        try:
            old_content = path.read_text(encoding="utf-8")
        except Exception:
            return f"overwrite existing file {path.name}"
        changed, total = self._changed_line_count(old_content, new_content)
        if changed > 200 or changed / total > 0.5:
            return f"overwrite a large portion of {path.name}"
        return None

    def _risky_edit_reason(
        self,
        raw_path: str,
        old_text: str,
        new_text: str,
        replace_all: bool,
    ) -> str | None:
        if not raw_path:
            return None
        if self._is_agent_state_file(raw_path):
            return None
        if replace_all:
            return f"apply a bulk replace in {Path(raw_path).name}"

        touched_lines = max(len(old_text.splitlines()), len(new_text.splitlines()))
        if touched_lines > 200:
            return f"edit more than 200 lines in {Path(raw_path).name}"

        path = self._resolve_workspace_path(raw_path)
        if not path.exists() or not path.is_file():
            return None

        try:
            content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        except Exception:
            return None

        match, count = _find_match(content, old_text.replace("\r\n", "\n"))
        if count > 1:
            return f"replace repeated content in {path.name}"
        if match is not None:
            changed, total = self._changed_line_count(match, new_text.replace("\r\n", "\n"))
            if changed > 200 or changed / total > 0.5:
                return f"rewrite most of the matched block in {path.name}"
        return None
