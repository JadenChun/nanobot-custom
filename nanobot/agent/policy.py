"""Internal execution policy for the main agent harness."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import shlex
import uuid
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from nanobot.agent.tools.filesystem import find_edit_match as _find_match
from nanobot.agent.turn import ToolOutcome
from nanobot.context_repo import ContextRepoManager
from nanobot.providers.base import ToolCallRequest

_APPROVAL_REQUIRED = "approval_required"


@dataclass(slots=True)
class ToolPolicyDecision:
    """Decision returned before executing a tool batch."""

    action: str = "allow"
    stop_reason: str | None = None
    response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # ``True`` is reserved for the main agent's run-wide approval flow.  A
    # delegated read-only policy explicitly returns ``False`` so the caller
    # cannot accidentally create a pending approval or ask the user to elevate
    # an immutable role.
    requires_approval: bool = False


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
class DelegatedReadOnlyPolicy(ToolPolicy):
    """Non-elevatable policy used by planner/reviewer/explorer/crawler roles.

    Role registries are the first boundary, but a model can still emit a
    forged or stale tool call.  This policy validates the complete batch before
    execution.  Any disallowed, unknown, malformed, or state-changing call
    blocks the entire batch; ``approval_granted`` is intentionally ignored so a
    main-agent approval cannot turn a read-only delegate into a writer.
    """

    allowed_tools: Collection[str] = field(
        default_factory=cast("Callable[[], frozenset[str]]", frozenset),
    )
    role: str = "read_only"
    # Accepted for construction compatibility with generic policy factories.
    # It must not affect evaluation.
    approval_granted: bool = False

    _NEVER_READ_ONLY = frozenset({
        "write_file",
        "edit_file",
        "exec",
        "cron",
        "message",
        "delegate_task",
        "plan_task",
        "review_work",
        "explore",
        "crawl_research",
        "desktop_use",
    })

    def __post_init__(self) -> None:
        if not self.allowed_tools:
            try:
                from nanobot.agent.capabilities import role_capabilities

                self.allowed_tools = role_capabilities(self.role).tools
            except ValueError:
                # An unregistered role with no explicit tools remains empty
                # and therefore fails closed for every call.
                self.allowed_tools = frozenset()
        self.allowed_tools = frozenset(
            name for name in self.allowed_tools if isinstance(cast(object, name), str) and name
        )

    async def evaluate(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_calls: list[ToolCallRequest],
    ) -> ToolPolicyDecision:
        _ = messages
        if not tool_calls:
            return ToolPolicyDecision()

        reasons: list[str] = []
        blocked_tools: list[str] = []
        for tool_call in tool_calls:
            reason = self._blocked_reason(tool_call)
            if reason:
                reasons.append(reason)
                if isinstance(cast(object, tool_call.name), str) and tool_call.name not in blocked_tools:
                    blocked_tools.append(tool_call.name)

        if not reasons:
            return ToolPolicyDecision()

        # Keep this response deliberately free of approval language.  The
        # outer loop treats policy_blocked as terminal and only creates pending
        # approval state for approval_required.
        summary = "; ".join(dict.fromkeys(reasons))
        return ToolPolicyDecision(
            action="respond",
            stop_reason="policy_blocked",
            response=f"Blocked by delegated read-only policy: {summary}.",
            metadata={
                "policy": "delegated_read_only",
                "role": self.role,
                "blocked_tools": blocked_tools,
                "reasons": list(dict.fromkeys(reasons)),
                "requires_approval": False,
            },
            requires_approval=False,
        )

    def _blocked_reason(self, tool_call: ToolCallRequest) -> str | None:
        name = tool_call.name if isinstance(cast(object, tool_call.name), str) else ""
        if name not in self.allowed_tools:
            return f"tool {name or '<unknown>'!r} is not permitted"
        if name in self._NEVER_READ_ONLY:
            return f"tool {name!r} is state-changing and not permitted"

        arguments = tool_call.arguments
        if not isinstance(arguments, dict):
            return f"tool {name!r} supplied malformed arguments"

        if name == "agent_browser":
            from nanobot.agent.tools.agent_browser import classify_browser_action

            if cast(dict[object, object], arguments).get("working_dir") is not None:
                return "agent_browser caller working_dir override is not permitted"
            args = cast(dict[object, object], arguments).get("args")
            if not isinstance(args, (list, tuple)):
                return "agent_browser action is unknown"
            if classify_browser_action(cast(list[object] | tuple[object, ...], args)) != "read_navigation":
                return "agent_browser action is state-changing or unknown"
        elif name == "agent_device":
            from nanobot.agent.tools.agent_device import classify_device_action

            if cast(dict[object, object], arguments).get("working_dir") is not None:
                return "agent_device caller working_dir override is not permitted"
            args = cast(dict[object, object], arguments).get("args")
            if not isinstance(args, (list, tuple)):
                return "agent_device action is unknown"
            if classify_device_action(cast(list[object] | tuple[object, ...], args)) != "read_navigation":
                return "agent_device action is state-changing or unknown"
        elif name == "social_crawl":
            from nanobot.agent.tools.social_crawl import classify_crawler_action

            action = cast(dict[object, object], arguments).get("action")
            if not isinstance(action, str):
                return "social_crawl action is unknown"
            if classify_crawler_action(action) != "read_navigation":
                return "social_crawl action is state-changing or unknown"
        return None


@dataclass(slots=True)
class RiskyActionPolicy(ToolPolicy):
    """Require approval for risky or hard-to-undo tool batches."""

    workspace: Path
    approval_granted: bool = False
    context_manager: ContextRepoManager = field(default_factory=ContextRepoManager)
    approved_effect_ids: frozenset[str] = field(default_factory=frozenset)
    write_target_resolver: Callable[[str, str], Path | None] | None = field(
        default=None, repr=False, compare=False,
    )
    _assessed_write_targets: dict[str, Path] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )
    _denied_write_targets: set[str] = field(
        default_factory=set, init=False, repr=False, compare=False,
    )

    _RISKY_EXEC_PATTERNS = (
        (re.compile(r"\brm\s+-[rf]{1,2}\b"), "delete files or directories"),
        (re.compile(r"\bgit\s+reset\b"), "reset git history"),
        (re.compile(r"\bgit\s+clean\b"), "remove untracked files"),
        (re.compile(r"\bgit\b[^|;&\n]*\bbranch\b[^|;&\n]*(?:\s-D\b|\s-d\b|--delete\b)"), "delete a git branch"),
        (re.compile(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*\s--delete\b"), "delete a git branch"),
        (re.compile(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*(?:^|\s)\+?:[^\s|;&]+"), "delete a git branch"),
        (re.compile(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*\s--mirror\b"), "delete git branches through a mirror push"),
        (re.compile(r"\bgit\s+push\b"), "push commits to a remote repository"),
        (re.compile(r"\bgit\s+checkout\b[^|;&\n]*\s+-f\b"), "force checkout changes"),
        (re.compile(r"\bdrop\s+table\b"), "drop database tables"),
        (re.compile(r"\bdelete\s+from\b"), "delete database rows"),
        (re.compile(r"\btruncate\s+table\b"), "truncate database tables"),
        (re.compile(r"\b(kubectl|helm)\b[^|;&\n]*\b(apply|delete|upgrade|rollback)\b"), "change deployed infrastructure"),
        (re.compile(r"\b(npm|pnpm|yarn|bun)\b[^|;&\n]*\bpublish\b"), "publish a package"),
    )

    def mutating_tool_names(self) -> set[str]:
        return {"write_file", "edit_file", "exec", "cron", "agent_browser", "agent_device"}

    async def evaluate(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_calls: list[ToolCallRequest],
    ) -> ToolPolicyDecision:
        if not tool_calls:
            return ToolPolicyDecision()

        self._assessed_write_targets.clear()
        self._denied_write_targets.clear()
        calls_to_assess: list[ToolCallRequest] = list(tool_calls)
        if self.approval_granted and self.approved_effect_ids:
            calls_to_assess = []
            for tool_call in tool_calls:
                if self._effect_fingerprint(tool_call) in self.approved_effect_ids:
                    if tool_call.name in {"write_file", "edit_file"}:
                        arguments = tool_call.arguments
                        raw_path = (
                            str(cast(dict[object, object], arguments).get("path") or "")
                            if isinstance(arguments, dict) else ""
                        )
                        resolved = self._resolve_write_target(tool_call.name, raw_path)
                        if resolved is not None:
                            self._assessed_write_targets[tool_call.id] = resolved
                        else:
                            self._denied_write_targets.add(tool_call.id)
                    continue
                calls_to_assess.append(tool_call)
            if not calls_to_assess:
                return ToolPolicyDecision()

        reasons: list[str] = []
        for tool_call in calls_to_assess:
            reason = self._risky_reason(tool_call)
            if reason:
                reasons.append(reason)

        if (self.approval_granted and not self.approved_effect_ids) or not reasons:
            return ToolPolicyDecision()

        summary = "; ".join(dict.fromkeys(reasons))
        approval_id = uuid.uuid4().hex
        effect_id = ",".join(
            self._effect_fingerprint(tool_call) for tool_call in calls_to_assess
        )
        response = (
            "I'm about to take a risky action and want your approval first. "
            f"Planned action: {summary}. Approval ID: {approval_id}. "
            "Reply with the approval control to continue or deny."
        )
        return ToolPolicyDecision(
            action="respond",
            stop_reason=_APPROVAL_REQUIRED,
            response=response,
            metadata={
                "reasons": reasons,
                "summary": summary,
                "requires_approval": True,
                "approval_id": approval_id,
                "effect_id": effect_id or None,
            },
            requires_approval=True,
        )

    def batch_has_mutation(self, tool_calls: list[ToolCallRequest]) -> bool:
        return any(tc.name in self.mutating_tool_names() for tc in tool_calls)

    def _risky_reason(self, tool_call: ToolCallRequest) -> str | None:
        if tool_call.name == "exec":
            return self._risky_exec_reason(
                str(tool_call.arguments.get("command") or ""),
                str(tool_call.arguments.get("working_dir") or ""),
            )
        if tool_call.name in {"write_file", "edit_file"}:
            raw_path = str(tool_call.arguments.get("path") or "")
            resolved = self._resolve_write_target(tool_call.name, raw_path)
            if resolved is None:
                # The target could not be assessed. Never let a later hook
                # change it into an unbound write.
                self._denied_write_targets.add(tool_call.id)
                return None
            self._assessed_write_targets[tool_call.id] = resolved
            if tool_call.name == "write_file":
                return self._risky_write_reason(
                    raw_path,
                    str(tool_call.arguments.get("content") or ""),
                    resolved,
                )
            return self._risky_edit_reason(
                raw_path,
                str(tool_call.arguments.get("old_text") or ""),
                str(tool_call.arguments.get("new_text") or ""),
                bool(tool_call.arguments.get("replace_all")),
                resolved,
            )
        if tool_call.name == "cron":
            return "schedule a recurring automation"
        if tool_call.name == "agent_browser":
            # Browser classification is deliberately conservative: only the
            # WP2 read/navigation primitives are autonomous.  Unknown and
            # generic actions (click/type/submit/evaluate/etc.) require the
            # same run-wide approval as other side effects.
            from nanobot.agent.tools.agent_browser import classify_browser_action

            args: object = tool_call.arguments.get("args") or []
            if classify_browser_action(args) != "read_navigation":
                return "perform a state-changing browser action"
        if tool_call.name == "agent_device":
            # Device inspection is safe; interaction, app/system controls, and
            # unknown commands require the main run-wide approval.
            from nanobot.agent.tools.agent_device import classify_device_action

            args = tool_call.arguments.get("args") or []
            if classify_device_action(args) != "read_navigation":
                return "perform a state-changing device action"
        return None

    def _risky_exec_reason(self, command: str, working_dir: str | None = None) -> str | None:
        lower = command.strip().lower()
        if not lower:
            return None
        if self._is_autonomous_context_git_command(command, working_dir):
            return None
        for pattern, label in self._RISKY_EXEC_PATTERNS:
            if pattern.search(lower):
                return label
        return None

    def _command_working_dir(self, command: str, working_dir: str | None) -> Path:
        base = self._resolve_workspace_path(working_dir or ".")
        git_c_path = self._extract_git_c_path(command)
        if git_c_path:
            candidate = Path(git_c_path).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            return candidate.resolve(strict=False)
        return base

    def _extract_git_c_path(self, command: str) -> str | None:
        try:
            tokens = shlex.split(command, posix=False)
        except ValueError:
            return None
        cleaned = [token.strip("'\"") for token in tokens]
        lowered = [token.lower() for token in cleaned]
        for index, token in enumerate(lowered):
            if token != "git":
                continue
            if index + 2 < len(tokens) and cleaned[index + 1] == "-C":
                return cleaned[index + 2]
        return None

    def _is_autonomous_context_git_command(self, command: str, working_dir: str | None) -> bool:
        lower = command.strip().lower()
        if not re.search(r"\bgit\b", lower):
            return False
        if self._deletes_git_branch(lower):
            return False

        cwd = self._command_working_dir(command, working_dir)
        target = self.context_manager.find_target_repo_for_path(cwd)
        repo = self.context_manager.find_repo_for_path(cwd)
        if target and (repo is None or len(str(target.path or "")) >= len(str(repo.path))):
            return False
        return bool(repo and repo.auto_push_enabled())

    def _deletes_git_branch(self, lower_command: str) -> bool:
        return bool(
            re.search(r"\bgit\b[^|;&\n]*\bbranch\b[^|;&\n]*(?:\s-D\b|\s-d\b|--delete\b)", lower_command)
            or re.search(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*\s--delete\b", lower_command)
            or re.search(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*(?:^|\s)\+?:[^\s|;&]+", lower_command)
            or re.search(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*\s--mirror\b", lower_command)
        )

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return candidate.resolve()

    def _effect_fingerprint(self, tool_call: ToolCallRequest) -> str:
        arguments = (
            cast(dict[str, Any], tool_call.arguments)
            if isinstance(tool_call.arguments, dict) else {}
        )
        payload: dict[str, Any] = {
            "name": tool_call.name,
            "arguments": arguments,
        }
        if tool_call.name in {"write_file", "edit_file"}:
            raw_path = str(arguments.get("path") or "")
            target = self._resolve_write_target(tool_call.name, raw_path)
            if target is not None:
                payload["target"] = str(target)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _resolve_write_target(self, tool_name: str, raw_path: str) -> Path | None:
        if not raw_path:
            return None
        if self.write_target_resolver is not None:
            try:
                return self.write_target_resolver(tool_name, raw_path)
            except (OSError, PermissionError, ValueError):
                return None
        return self._resolve_workspace_path(raw_path)

    def validate_write_target(self, tool_call: ToolCallRequest, tool: Any) -> str | None:
        """Recheck the assessed target through the tool immediately before dispatch."""
        error, _target = self.validate_write_target_binding(tool_call, tool)
        return str(error.content) if isinstance(error, ToolOutcome) else error

    def validate_write_target_binding(
        self,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any = None,
    ) -> tuple[str | ToolOutcome | None, Path | None]:
        """Recheck and return the assessed target for the mutation boundary."""
        if tool_call.id in self._denied_write_targets:
            self._denied_write_targets.discard(tool_call.id)
            raw_path = ""
            if isinstance(params, dict):
                raw_path = str(cast(dict[object, object], params).get("path") or "")
            if not raw_path and isinstance(tool_call.arguments, dict):
                raw_path = str(
                    cast(dict[object, object], tool_call.arguments).get("path") or ""
                )
            explain = getattr(tool, "write_refusal", None)
            if raw_path and callable(explain):
                refusal = cast(Callable[[str], str | ToolOutcome | None], explain)(raw_path)
                if refusal is not None:
                    return refusal, None
            return "write target was not assessable and remains unauthorized", None
        assessed = self._assessed_write_targets.pop(tool_call.id, None)
        if assessed is None:
            if tool_call.name in {"write_file", "edit_file"}:
                return "write target was not assessed", None
            return None, None
        raw_path = ""
        if isinstance(params, dict):
            raw_path = str(cast(dict[object, object], params).get("path") or "")
        if not raw_path:
            arguments = tool_call.arguments
            if isinstance(arguments, dict):
                raw_path = str(cast(dict[object, object], arguments).get("path") or "")
        if not raw_path:
            return None, None
        resolver = getattr(tool, "resolve_write_path", None)
        if not callable(resolver):
            return None, None
        try:
            resolved = cast(Callable[[str], str | Path], resolver)(raw_path)
            current = Path(resolved).expanduser().resolve(strict=False)
        except (OSError, PermissionError, ValueError):
            return f"Error: write target is no longer accessible: {raw_path}", None
        if current != assessed:
            return (
                f"Error: write target changed before dispatch: {raw_path}. "
                "Retry the request so the new target can be assessed."
            ), None
        return None, assessed

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

    # Files at or below this many lines are considered trivial state; a full
    # rewrite is not "risky" because there is nothing of substance to lose.
    # The absolute ``changed > 200`` guard still catches genuinely large edits.
    _SMALL_FILE_LINE_THRESHOLD = 10

    def _is_agent_state_file(self, path: Path) -> bool:
        """Memory files the agent manages freely — no size checks."""
        return "memory" in path.parts

    def _is_autonomous_managed_path(self, path: Path) -> bool:
        """Writable managed paths already have finer-grained repo policy checks."""
        target = self.context_manager.find_target_repo_for_path(path)
        repo = self.context_manager.find_repo_for_path(path)
        if target and (repo is None or len(str(target.path or "")) >= len(str(repo.path))):
            rel = target.rel_path(path)
            return bool(
                rel
                and not target.is_protected(rel)
                and not target.requires_proposal(rel)
                and target.is_writable(rel)
            )

        if repo:
            rel = repo.rel_path(path)
            return bool(
                rel
                and not repo.read_only
                and not repo.is_protected(rel)
                and not repo.requires_proposal(rel)
                and not repo.blocks_direct_store_edit(rel)
                and repo.is_writable(rel)
            )

        return False

    def _risky_write_reason(
        self,
        raw_path: str,
        new_content: str,
        resolved_path: Path | None = None,
    ) -> str | None:
        if not raw_path:
            return None
        path = resolved_path or self._resolve_workspace_path(raw_path)
        if self._is_agent_state_file(path) or self._is_autonomous_managed_path(path):
            return None
        if not path.exists() or not path.is_file():
            return None
        try:
            old_content = path.read_text(encoding="utf-8")
        except Exception:
            return f"overwrite existing file {path.name}"
        changed, total = self._changed_line_count(old_content, new_content)
        if changed > 200:
            return f"overwrite a large portion of {path.name}"
        if total <= self._SMALL_FILE_LINE_THRESHOLD:
            return None
        if changed / total > 0.5:
            return f"overwrite a large portion of {path.name}"
        return None

    def _risky_edit_reason(
        self,
        raw_path: str,
        old_text: str,
        new_text: str,
        replace_all: bool,
        resolved_path: Path | None = None,
    ) -> str | None:
        if not raw_path:
            return None
        path = resolved_path or self._resolve_workspace_path(raw_path)
        if self._is_agent_state_file(path) or self._is_autonomous_managed_path(path):
            return None
        if replace_all:
            return f"apply a bulk replace in {path.name}"

        touched_lines = max(len(old_text.splitlines()), len(new_text.splitlines()))
        if touched_lines > 200:
            return f"edit more than 200 lines in {path.name}"

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
