"""Declarative permission rules, risk classifiers, and write-target TOCTOU guard."""

from __future__ import annotations

import difflib
import fnmatch
import re
import shlex
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from kohaku.agent.turn import ToolOutcome
from kohaku.context_repo import ContextRepoManager
from kohaku.providers.base import ToolCallRequest

if TYPE_CHECKING:
    from kohaku.config.schema import PermissionAction, ToolPermissionsConfig

# Desktop control actions that only observe the local screen. Every other
# action emits global input events or is unknown.
READ_ONLY_DESKTOP_ACTIONS = frozenset({"info", "screenshot", "cursor_position", "wait"})

TOOL_RULE_ALIASES: dict[str, tuple[str, ...]] = {
    "exec": ("exec", "bash", "shell"),
    "edit_file": ("edit_file", "edit"),
    "write_file": ("write_file", "write"),
}

LOCAL_DESTRUCTIVE_EXEC_PATTERNS = (
    (re.compile(r"\brm\s+-[rf]{1,2}\b"), "delete files or directories"),
    (re.compile(r"\bgit\s+reset\b"), "reset git history"),
    (re.compile(r"\bgit\s+clean\b"), "remove untracked files"),
    (re.compile(r"\bgit\b[^|;&\n]*\bbranch\b[^|;&\n]*(?:\s-D\b|\s-d\b|--delete\b)"), "delete a git branch"),
    (re.compile(r"\bgit\s+checkout\b[^|;&\n]*\s+-f\b"), "force checkout changes"),
)

EXTERNAL_EXEC_PATTERNS = (
    (re.compile(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*\s--delete\b"), "delete a git branch"),
    (re.compile(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*(?:^|\s)\+?:[^\s|;&]+"), "delete a git branch"),
    (re.compile(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*\s--mirror\b"), "delete git branches through a mirror push"),
    (re.compile(r"\bgit\s+push\b"), "push commits to a remote repository"),
    (re.compile(r"\bdrop\s+table\b"), "drop database tables"),
    (re.compile(r"\bdelete\s+from\b"), "delete database rows"),
    (re.compile(r"\btruncate\s+table\b"), "truncate database tables"),
    (re.compile(r"\b(kubectl|helm)\b[^|;&\n]*\b(apply|delete|upgrade|rollback)\b"), "change deployed infrastructure"),
    (re.compile(r"\b(npm|pnpm|yarn|bun)\b[^|;&\n]*\bpublish\b"), "publish a package"),
)

RISKY_EXEC_PATTERNS = LOCAL_DESTRUCTIVE_EXEC_PATTERNS + EXTERNAL_EXEC_PATTERNS

_BARE_EXECUTABLE = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_.+-]*", re.ASCII)
_SHELL_SYNTAX = re.compile(r"[\x00-\x08\x0a-\x1f\x7f;&|<>`$(){}\\#*?\[\]%!^]")
_UNSAVEABLE_EXECUTABLES = frozenset({
    "sh", "bash", "zsh", "dash", "fish", "ksh", "csh", "tcsh", "ash",
    "pwsh", "powershell", "cmd", "python", "python3", "node", "bun", "deno",
    "ruby", "perl", "env", "sudo", "su", "doas", "xargs", "exec", "eval",
    "command", "builtin", "source", "time", "nohup", "busybox", "setsid",
    "nice", "timeout", "chroot", "runuser", "trap", "enable",
})


def normalize_sandbox_executable(value: str) -> str:
    """Validate a configured exclusion and return its lowercased name."""
    name = value.strip()
    if not _BARE_EXECUTABLE.fullmatch(name):
        raise ValueError("Sandbox exclusions must be bare executable names")
    return name.lower()


def _sandbox_command_tokens(command: str) -> list[str]:
    # Reject shell syntax before tokenization, including syntax inside quotes.
    # This keeps the exception independent of shell-specific expansion rules.
    if _SHELL_SYNTAX.search(command):
        return []
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    if not tokens or not _BARE_EXECUTABLE.fullmatch(tokens[0]):
        return []
    if command.lstrip().split(maxsplit=1)[0] != tokens[0]:
        return []
    return tokens


def sandbox_command_name(command: str) -> str | None:
    """Return the bare CLI name of a simple command, including risky commands."""
    tokens = _sandbox_command_tokens(command)
    if not tokens:
        return None
    name = tokens[0].lower()
    generic = name.removesuffix(".exe")
    if generic in _UNSAVEABLE_EXECUTABLES or re.fullmatch(r"python\d+(?:\.\d+)*", generic):
        return None
    return name


def sandbox_exec_risk_reason(command: str) -> str | None:
    """Check both shell text and literal tokens for risky actions."""
    return match_exec_risk_reason(command) or match_exec_risk_reason(
        " ".join(_sandbox_command_tokens(command)),
    )


def extract_saveable_sandbox_command(command: str) -> str | None:
    """Return the executable name only when its command can be saved."""
    name = sandbox_command_name(command)
    return name if name and sandbox_exec_risk_reason(command) is None else None


def matches_sandbox_excluded_command(command: str, excluded: Iterable[str]) -> bool:
    """Match a complete, save-eligible command against configured names."""
    name = extract_saveable_sandbox_command(command)
    return name is not None and name in excluded

# Files at or below this many lines are considered trivial state; a full
# rewrite is not "risky" because there is nothing of substance to lose.
# The absolute ``changed > 200`` guard still catches genuinely large edits.
SMALL_FILE_LINE_THRESHOLD = 10


def rule_candidates(
    tool_call: ToolCallRequest,
    *,
    workspace: Path,
    target_path: Path | None = None,
    command: str | None = None,
) -> tuple[str, ...]:
    """Return normalized match candidates for pattern-scoped permission rules."""
    if tool_call.name == "exec":
        cmd = (command if command is not None else str(tool_call.arguments.get("command") or "")).strip()
        return (cmd,) if cmd else ()
    if tool_call.name in {"write_file", "edit_file"}:
        candidates: list[str] = []
        raw_path = str(tool_call.arguments.get("path") or "").strip()
        if target_path is not None:
            try:
                rel = target_path.relative_to(workspace.resolve(strict=False)).as_posix()
                candidates.append(rel)
            except ValueError:
                pass
            candidates.append(target_path.as_posix())
            candidates.append(target_path.name)
        if raw_path and raw_path not in candidates:
            candidates.append(raw_path)
        return tuple(candidates)
    if tool_call.name == "desktop_use":
        action = tool_call.arguments.get("action")
        return (action.strip(),) if isinstance(action, str) and action.strip() else ()
    return ()


def match_explicit_rule(
    permissions: ToolPermissionsConfig | None,
    tool_call: ToolCallRequest,
    *,
    workspace: Path,
    target_path: Path | None = None,
    command: str | None = None,
) -> PermissionAction | None:
    """Evaluate declarative ``permissions.rules`` (last-match-wins)."""
    if permissions is None or not permissions.rules:
        return None
    aliases = TOOL_RULE_ALIASES.get(tool_call.name, (tool_call.name,))
    candidates = rule_candidates(
        tool_call, workspace=workspace, target_path=target_path, command=command,
    )
    matched: PermissionAction | None = None
    for key, rule in permissions.rules.items():
        if key != "*" and key not in aliases:
            continue
        if isinstance(rule, str):
            matched = rule
        else:
            for pattern, action in rule.items():
                if pattern == "*" or any(
                    fnmatch.fnmatchcase(candidate, pattern) for candidate in candidates
                ):
                    matched = action
    return matched


def match_exec_risk_reason(command: str, *, include_local: bool = True) -> str | None:
    """Classify a shell command against risky exec patterns."""
    lower = command.strip().lower()
    if not lower:
        return None
    patterns = RISKY_EXEC_PATTERNS if include_local else EXTERNAL_EXEC_PATTERNS
    for pattern, label in patterns:
        if pattern.search(lower):
            return label
    return None


def changed_line_count(before: str, after: str) -> tuple[int, int]:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    total = max(len(before_lines), len(after_lines), 1)
    start, before_end, after_end = 0, len(before_lines), len(after_lines)
    while start < min(before_end, after_end) and before_lines[start] == after_lines[start]:
        start += 1
    while (
        before_end > start
        and after_end > start
        and before_lines[before_end - 1] == after_lines[after_end - 1]
    ):
        before_end -= 1
        after_end -= 1
    matcher = difflib.SequenceMatcher(
        a=before_lines[start:before_end], b=after_lines[start:after_end],
    )
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed += max(i2 - i1, j2 - j1)
    return changed, total


def is_large_replacement(before: str, after: str) -> bool:
    changed, total = changed_line_count(before, after)
    return changed > 200 or (total > SMALL_FILE_LINE_THRESHOLD and changed / total > 0.5)


def is_agent_state_file(path: Path) -> bool:
    """Memory files the agent manages freely — no size checks."""
    return "memory" in path.parts


def is_autonomous_managed_path(context_manager: ContextRepoManager, path: Path) -> bool:
    """Writable managed paths already have finer-grained repo policy checks."""
    target = context_manager.find_target_repo_for_path(path)
    repo = context_manager.find_repo_for_path(path)
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


@dataclass(slots=True)
class WriteTargetGuard:
    """TOCTOU guard binding assessed write/edit targets to actual tool dispatch."""

    workspace: Path
    resolver: Callable[[str, str], Path | None] | None = None
    _assessed: dict[str, Path] = field(default_factory=dict, init=False, repr=False)
    _denied: set[str] = field(default_factory=set, init=False, repr=False)

    def clear(self) -> None:
        self._assessed.clear()
        self._denied.clear()

    def resolve_workspace_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return candidate.resolve()

    def resolve(self, tool_name: str, raw_path: str) -> Path | None:
        if not raw_path:
            return None
        if self.resolver is not None:
            try:
                return self.resolver(tool_name, raw_path)
            except (OSError, PermissionError, ValueError):
                return None
        return self.resolve_workspace_path(raw_path)

    def assess(self, call: ToolCallRequest, raw_path: str) -> Path | None:
        target = self.resolve(call.name, raw_path)
        if target is None:
            # Never let a later hook turn a denied target into an unbound write.
            self._denied.add(call.id)
        else:
            self._assessed[call.id] = target
        return target

    def validate_binding(
        self,
        tool_call: ToolCallRequest,
        tool: Any,
        params: Any = None,
    ) -> tuple[str | ToolOutcome | None, Path | None]:
        """Recheck and return the assessed target for the mutation boundary."""
        if tool_call.id in self._denied:
            self._denied.discard(tool_call.id)
            raw_path = self._extract_path(tool_call, params)
            explain = getattr(tool, "write_refusal", None)
            if raw_path and callable(explain):
                refusal = cast(Callable[[str], str | ToolOutcome | None], explain)(raw_path)
                if refusal is not None:
                    return refusal, None
            return "write target was not assessable and remains unauthorized", None
        assessed = self._assessed.pop(tool_call.id, None)
        if assessed is None:
            if tool_call.name in {"write_file", "edit_file"}:
                return "write target was not assessed", None
            return None, None
        raw_path = self._extract_path(tool_call, params)
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

    @staticmethod
    def _extract_path(tool_call: ToolCallRequest, params: Any) -> str:
        raw_path = ""
        if isinstance(params, dict):
            raw_path = str(cast(dict[object, object], params).get("path") or "")
        if not raw_path and isinstance(tool_call.arguments, dict):
            raw_path = str(cast(dict[object, object], tool_call.arguments).get("path") or "")
        return raw_path
