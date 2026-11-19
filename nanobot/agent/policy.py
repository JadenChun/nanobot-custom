"""Internal execution policy for the main agent harness."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import uuid
from collections.abc import Callable, Collection
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.filesystem import EditCandidate, EditContentBinding, EditFileTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.turn import ToolOutcome, TurnContext
from nanobot.agent.turn_admission import EffectApprovalAuthority
from nanobot.context_repo import ContextRepoManager
from nanobot.providers.base import ToolCallRequest

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry

_APPROVAL_REQUIRED = "approval_required"
_POLICY_BLOCKED = "policy_blocked"
# A reduced-protection nested agent launch is refused, not approved, when the
# launch path cannot prove outer confinement and process ownership.
_LAUNCH_CONFINEMENT_UNPROVEN = "launch_confinement_unproven"

# Desktop control actions that only observe the local screen. Every other
# action, and every action value that is not an exact match, emits global
# input events or is unknown, so it needs the same exact-effect approval as
# other state-changing tools.
_READ_ONLY_DESKTOP_ACTIONS = frozenset({"info", "screenshot", "cursor_position", "wait"})

# External agent launches that lower their own sandbox or approval protection.
# A launch that keeps the tool's default protection is not listed and stays an
# ordinary command. These patterns match the lowercased command, so every
# alternative is written in lowercase, including values such as
# ``bypassPermissions`` and ``danger-full-access``.
_REDUCED_PROTECTION_LAUNCHES = (
    re.compile(
        r"\bcodex\b[^|;&\n]*?"
        r"(?:--yolo\b|--dangerously-bypass-approvals-and-sandbox\b|--full-auto\b)"
    ),
    re.compile(
        r"\bcodex\b[^|;&\n]*?"
        r"(?:--sandbox[= ]+[\\'\"]*danger-full-access\b"
        r"|--ask-for-approval[= ]+[\\'\"]*never\b)"
    ),
    # Codex config overrides that select a full-access sandbox or disable approval.
    re.compile(
        r"\bcodex\b[^|;&\n]*?(?:-c\s+|--config\s+|--config=)[\\'\"]*"
        r"(?:sandbox_mode|approval_policy)\s*=\s*[\\'\"]*"
        r"(?:danger-full-access|never)\b"
    ),
    re.compile(
        r"\bclaude\b[^|;&\n]*?"
        r"(?:--dangerously-skip-permissions\b"
        r"|--permission-mode[= ]+[\\'\"]*bypasspermissions\b)"
    ),
    re.compile(r"\bgemini\b[^|;&\n]*?(?:--yolo\b|\s-y\b)"),
)

# A tmux command starts or types into a process the shell executor does not own.
_TMUX_LAUNCH_COMMANDS = re.compile(
    r"\btmux\b[^|;&\n]*\b(?:send-keys|new-session|new-window|respawn-pane|split-window)\b"
)


def nested_launch_path(command: str) -> str | None:
    """Classify a reduced-protection nested launch and its launch path.

    Returns ``"shell"`` for a direct launch the shell executor starts itself,
    ``"tmux"`` when the launch is typed into a tmux session, and ``None`` when
    the command does not lower a nested agent's protection.
    """
    lower = command.strip().lower()
    if not lower:
        return None
    if not any(pattern.search(lower) for pattern in _REDUCED_PROTECTION_LAUNCHES):
        return None
    return "tmux" if _TMUX_LAUNCH_COMMANDS.search(lower) else "shell"


def nested_launch_id(
    command: str, working_dir: str | None = None, workspace: Path | str | None = None,
) -> str:
    """Identity of one nested launch: the exact command and effective directory.

    The policy and the executor resolve the same directory from the same
    workspace, so one launch has one identity on both sides of the boundary.
    """
    base = Path(str(workspace if workspace is not None else Path.cwd())).expanduser()
    target = Path(working_dir).expanduser() if working_dir else base
    if not target.is_absolute():
        target = base / target
    payload = json.dumps(
        {"command": " ".join(command.split()), "cwd": str(target.resolve(strict=False))},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LaunchConfinementAttestation:
    """Executor statement that confinement and process ownership hold for one launch."""

    launch_id: str
    path: str
    sandbox: str
    process_owner: str

    def covers(self, launch_id: str, path: str) -> bool:
        """True only for the attested launch, path, and non-empty providers."""
        return (
            bool(launch_id)
            and launch_id == self.launch_id
            and path == self.path
            and bool(self.sandbox)
            and bool(self.process_owner)
        )


LaunchConfinementAttestor = Callable[[str, str | None], LaunchConfinementAttestation | None]

# Executors register the attestor they can honestly issue for their workspace.
_LAUNCH_CONFINEMENT_ATTESTORS: dict[str, LaunchConfinementAttestor] = {}


def _workspace_key(workspace: Path | str | None) -> str | None:
    if workspace is None:
        return None
    raw = str(workspace)
    if not raw:
        return None
    try:
        return str(Path(raw).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return raw


def register_launch_confinement_attestor(
    workspace: Path | str, attestor: LaunchConfinementAttestor,
) -> None:
    key = _workspace_key(workspace)
    if key is not None:
        _LAUNCH_CONFINEMENT_ATTESTORS[key] = attestor


def unregister_launch_confinement_attestor(workspace: Path | str) -> None:
    key = _workspace_key(workspace)
    if key is not None:
        _LAUNCH_CONFINEMENT_ATTESTORS.pop(key, None)


def resolve_launch_confinement_attestor(
    workspace: Path | None,
) -> LaunchConfinementAttestor | None:
    key = _workspace_key(workspace)
    return None if key is None else _LAUNCH_CONFINEMENT_ATTESTORS.get(key)


@dataclass(slots=True)
class ToolPolicyDecision:
    """Decision returned before executing a tool batch."""

    action: str = "allow"
    stop_reason: str | None = None
    response: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # ``True`` is reserved for the main agent's exact-effect approval flow. A
    # delegated read-only policy explicitly returns ``False`` so the caller
    # cannot accidentally create a pending approval or ask the user to elevate
    # an immutable role.
    requires_approval: bool = False
    # A concrete tool's validation result, not a consent decision.
    preflight_result: str | ToolOutcome | None = None
    edit_binding: EditContentBinding | None = None
    exec_cwd: Path | None = None


class ToolPolicy:
    """Base policy surface for tool-batch interception."""

    def for_tools(self, tools: ToolRegistry) -> ToolPolicy:
        """Bind assessment to the registrations used by this response."""
        return self

    async def evaluate(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_calls: list[ToolCallRequest],
    ) -> ToolPolicyDecision:
        return ToolPolicyDecision()

    def authorize_dispatch(
        self,
        tool_call: ToolCallRequest,
        arguments: dict[str, Any],
        turn_context: TurnContext | None,
    ) -> ToolPolicyDecision:
        return ToolPolicyDecision()

    def validate_write_target_binding(
        self,
        tool_call: ToolCallRequest,
        tool: Tool | None,
        params: object,
    ) -> tuple[str | ToolOutcome | None, Path | None]:
        return None, None


@dataclass(slots=True)
class CompositeToolPolicy(ToolPolicy):
    """Apply trusted child role policy before canonical risky-action policy."""

    policies: tuple[ToolPolicy, ...]

    def for_tools(self, tools: ToolRegistry) -> ToolPolicy:
        return CompositeToolPolicy(tuple(policy.for_tools(tools) for policy in self.policies))

    async def evaluate(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_calls: list[ToolCallRequest],
    ) -> ToolPolicyDecision:
        for policy in self.policies:
            decision = await policy.evaluate(messages=messages, tool_calls=tool_calls)
            if decision.action != "allow":
                return decision
        return ToolPolicyDecision()

    def authorize_dispatch(
        self,
        tool_call: ToolCallRequest,
        arguments: dict[str, Any],
        turn_context: TurnContext | None,
    ) -> ToolPolicyDecision:
        edit_binding = None
        exec_cwd = None
        for policy in self.policies:
            decision = policy.authorize_dispatch(tool_call, arguments, turn_context)
            if decision.action != "allow" or decision.preflight_result is not None:
                return decision
            if decision.edit_binding is not None:
                edit_binding = decision.edit_binding
            if decision.exec_cwd is not None:
                exec_cwd = decision.exec_cwd
        return ToolPolicyDecision(edit_binding=edit_binding, exec_cwd=exec_cwd)

    def validate_write_target_binding(
        self, tool_call: ToolCallRequest, tool: Tool | None, params: object,
    ) -> tuple[str | ToolOutcome | None, Path | None]:
        target = None
        for policy in self.policies:
            error, assessed = policy.validate_write_target_binding(tool_call, tool, params)
            if error is not None:
                return error, None
            if assessed is not None:
                target = assessed
        return None, target

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


@dataclass(frozen=True, slots=True)
class _EffectAssessment:
    reason: str | None = None
    edit_target: Path | None = None
    edit_binding: EditContentBinding | None = None
    exec_cwd: Path | None = None
    refusal: str | ToolOutcome | None = None


@dataclass(slots=True)
class RiskyActionPolicy(ToolPolicy):
    """Require approval for risky or hard-to-undo tool batches."""

    workspace: Path
    # Retained descriptive inputs for internal compatibility. Neither public
    # booleans nor fingerprint sets authorize dispatch.
    approval_granted: bool = False
    context_manager: ContextRepoManager = field(default_factory=ContextRepoManager)
    approved_effect_ids: frozenset[str] = field(default_factory=frozenset)
    # init=True deliberately preserves this SAME sealed owner through the
    # occurrence-local dataclasses.replace used by captured MCP registries.
    effect_authority: EffectApprovalAuthority | None = field(
        default=None, repr=False, compare=False,
    )
    write_target_resolver: Callable[[str, str], Path | None] | None = field(
        default=None, repr=False, compare=False,
    )
    # The executor that starts a nested launch issues its own confinement
    # attestation. ``None`` resolves the executor registered for this workspace.
    launch_confinement_attestor: LaunchConfinementAttestor | None = field(
        default=None, repr=False, compare=False,
    )
    _tools: ToolRegistry | None = field(default=None, repr=False, compare=False)
    _assessed_write_targets: dict[str, Path] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )
    _denied_write_targets: set[str] = field(
        default_factory=set, init=False, repr=False, compare=False,
    )
    _risky_calls: dict[str, tuple[str, str]] = field(
        default_factory=dict, init=False, repr=False, compare=False,
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

    def for_tools(self, tools: ToolRegistry) -> RiskyActionPolicy:
        def resolve_target(name: str, raw_path: str) -> Path | None:
            resolver = getattr(tools.get(name), "resolve_write_path", None)
            if not callable(resolver):
                return None
            resolved = cast(Callable[[str], str | Path], resolver)(raw_path)
            return Path(resolved).expanduser().resolve(strict=False)

        return replace(self, _tools=tools, write_target_resolver=resolve_target)

    def mutating_tool_names(self) -> set[str]:
        return {
            "write_file", "edit_file", "exec", "cron",
            "agent_browser", "agent_device", "desktop_use",
        }

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
        self._risky_calls.clear()
        blocked: list[tuple[str, str]] = []
        for tool_call in tool_calls:
            prepared = self._prepare_effect_call(tool_call)
            if prepared is None:
                continue  # Normal tool preparation reports the validation failure.
            assessment = self._assess(prepared)
            if isinstance(assessment, ToolPolicyDecision):
                return assessment
            if assessment.refusal is not None:
                continue  # Consent cannot make this operation admissible.
            reason = assessment.reason
            effect_id = self._effect_fingerprint(prepared, assessment)
            if self.effect_authority is not None and effect_id == self.effect_authority.effect_id:
                reason = reason or "repeat the previously approved risky action"
            if reason:
                self._risky_calls[tool_call.id] = (effect_id, reason)
                if self.effect_authority is None or not self.effect_authority.permits(effect_id):
                    blocked.append((effect_id, reason))
        if blocked:
            return self._request_approval(*blocked[0])
        return ToolPolicyDecision()

    def _request_approval(self, effect_id: str, summary: str) -> ToolPolicyDecision:
        """One decision identifies one concrete effect, never an entire batch."""
        approval_id = uuid.uuid4().hex
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
                "reasons": [summary],
                "summary": summary,
                "requires_approval": True,
                "approval_id": approval_id,
                "effect_id": effect_id or None,
            },
            requires_approval=True,
        )

    def authorize_dispatch(
        self, tool_call: ToolCallRequest, arguments: dict[str, Any],
        turn_context: TurnContext | None,
    ) -> ToolPolicyDecision:
        """Recheck prepared arguments and consume only at actual dispatch."""
        prepared = ToolCallRequest(id=tool_call.id, name=tool_call.name, arguments=arguments)
        assessed = self._risky_calls.get(tool_call.id)
        # Reprepare concrete edits and shell calls after hooks. Project
        # selection, grants and file contents can change after batch review.
        assessment = (
            self._assess(prepared, previous_reason=assessed[1] if assessed else None)
            if assessed is None or prepared.name in {"exec", "edit_file"}
            else _EffectAssessment(reason=assessed[1])
        )
        if isinstance(assessment, ToolPolicyDecision):
            return assessment
        if assessment.refusal is not None:
            return ToolPolicyDecision(preflight_result=assessment.refusal)
        effect_id = self._effect_fingerprint(prepared, assessment)
        reason = assessment.reason
        if self.effect_authority is not None and effect_id == self.effect_authority.effect_id:
            reason = reason or "repeat the previously approved risky action"
        if reason is None:
            return ToolPolicyDecision(edit_binding=assessment.edit_binding)
        if (
            (assessed is None or effect_id == assessed[0])
            and self.effect_authority is not None
            and self.effect_authority.consume(effect_id, turn_context)
        ):
            return ToolPolicyDecision(
                edit_binding=assessment.edit_binding, exec_cwd=assessment.exec_cwd,
            )
        return self._request_approval(effect_id, reason)

    def _prepare_effect_call(self, call: ToolCallRequest) -> ToolCallRequest | None:
        tools = self._tools
        if tools is None:
            return call
        tool = tools.get(call.name)
        if tool is None:
            return None  # The request cannot approve a withheld or missing binding.
        if type(tool) not in {EditFileTool, ExecTool}:
            return call
        _, params, error = tools.prepare_call(call.name, call.arguments)
        if error is not None or not isinstance(params, dict):
            return None
        return replace(call, arguments=cast(dict[str, Any], params))

    def _assess(
        self, call: ToolCallRequest, *, previous_reason: str | None = None,
    ) -> _EffectAssessment | ToolPolicyDecision:
        if call.name == "edit_file":
            raw_path = str(call.arguments.get("path") or "")
            path = self._assess_write_target(call, raw_path)
            if path is None:
                return _EffectAssessment()
            candidate = EditFileTool.prepare_edit(path, **call.arguments)
            if isinstance(candidate, str):
                return _EffectAssessment(refusal=candidate)
            reason = self._risky_edit_reason(path, candidate, bool(call.arguments.get("replace_all")))
            return _EffectAssessment(
                reason=reason or previous_reason,
                edit_target=path,
                edit_binding=candidate.content_binding(),
            )

        reason = self._risky_reason(call)
        if isinstance(reason, ToolPolicyDecision):
            return reason
        reason = reason or previous_reason
        if reason is not None and call.name == "exec" and self._tools is not None:
            tool = self._tools.get(call.name)
            if type(tool) is ExecTool:
                check = tool.approval_preflight(**call.arguments)
                if not isinstance(check, Path):
                    return _EffectAssessment(refusal=check)
                return _EffectAssessment(reason=reason, exec_cwd=check)
        return _EffectAssessment(reason=reason)

    def batch_has_mutation(self, tool_calls: list[ToolCallRequest]) -> bool:
        return any(tc.name in self.mutating_tool_names() for tc in tool_calls)

    def _risky_reason(self, tool_call: ToolCallRequest) -> str | ToolPolicyDecision | None:
        if tool_call.name == "exec":
            command, working_dir = ExecTool.request_location(**tool_call.arguments)
            if not command:
                return None
            launch_path = nested_launch_path(command)
            if launch_path is not None:
                if not self._launch_confinement_covers(command, working_dir, launch_path):
                    return self._refuse_launch(launch_path)
                # Exact approval is required in addition to the attestation.
                return "launch a nested agent with reduced sandbox or approval protections"
            return self._risky_exec_reason(command, working_dir)
        if tool_call.name == "write_file":
            raw_path = str(tool_call.arguments.get("path") or "")
            resolved = self._assess_write_target(tool_call, raw_path)
            if resolved is None:
                return None
            return self._risky_write_reason(
                raw_path,
                str(tool_call.arguments.get("content") or ""),
                resolved,
            )
        if tool_call.name == "cron":
            return "schedule a recurring automation"
        if tool_call.name == "agent_browser":
            # Browser classification is deliberately conservative: only the
            # WP2 read/navigation primitives are autonomous.  Unknown and
            # generic actions (click/type/submit/evaluate/etc.) require the
            # same exact-effect approval as other side effects.
            from nanobot.agent.tools.agent_browser import classify_browser_action

            args: object = tool_call.arguments.get("args") or []
            if classify_browser_action(args) != "read_navigation":
                return "perform a state-changing browser action"
        if tool_call.name == "agent_device":
            # Device inspection is safe; interaction, app/system controls, and
            # unknown commands require exact-effect approval.
            from nanobot.agent.tools.agent_device import classify_device_action

            args = tool_call.arguments.get("args") or []
            if classify_device_action(args) != "read_navigation":
                return "perform a state-changing device action"
        if tool_call.name == "desktop_use":
            # Desktop control emits global input events. Only the exact
            # read-only action names stay autonomous; any other value,
            # including a non-string or unknown action, needs approval.
            action = tool_call.arguments.get("action")
            if not isinstance(action, str) or action not in _READ_ONLY_DESKTOP_ACTIONS:
                return "perform a state-changing desktop input action"
        return None

    def _risky_exec_reason(self, command: str, working_dir: str | None = None) -> str | None:
        _ = working_dir
        lower = command.strip().lower()
        if not lower:
            return None
        # The ordinary shell policy applies to every shell call. A Git word,
        # an autoPush repository, or a benign segment never exempts an effect.
        for pattern, label in self._RISKY_EXEC_PATTERNS:
            if pattern.search(lower):
                return label
        return None

    def _launch_confinement_covers(
        self, command: str, working_dir: str | None, launch_path: str,
    ) -> bool:
        """True only when the executor attests this launch's confinement."""
        if launch_path != "shell":
            # A tmux command types into a process the shell executor neither
            # confines for that launch nor owns; it cannot attest either.
            return False
        attestation = self._launch_confinement_attestation(command, working_dir)
        if attestation is None:
            return False
        return attestation.covers(
            nested_launch_id(command, working_dir, self.workspace), launch_path,
        )

    def _launch_confinement_attestation(
        self, command: str, working_dir: str | None,
    ) -> LaunchConfinementAttestation | None:
        attestor = self.launch_confinement_attestor or resolve_launch_confinement_attestor(
            self.workspace,
        )
        if attestor is None:
            return None
        try:
            return attestor(command, working_dir)
        except (OSError, ValueError):
            return None

    def _refuse_launch(self, launch_path: str) -> ToolPolicyDecision:
        """Typed refusal that precedes every process start for this launch."""
        summary = (
            "launch a nested agent with reduced sandbox or approval protections without "
            "proven outer confinement and process ownership"
        )
        return ToolPolicyDecision(
            action="respond",
            stop_reason=_POLICY_BLOCKED,
            response=(
                "I cannot start that nested agent launch. Its settings reduce the sandbox "
                "or approval protection, and this launch path cannot prove outer "
                "confinement and process ownership. Use the agent's default protection, "
                "or run it inside a confined shell this executor owns."
            ),
            metadata={
                "reason_id": _LAUNCH_CONFINEMENT_UNPROVEN,
                "reasons": [summary],
                "summary": summary,
                "launch_path": launch_path,
                "requires_approval": False,
            },
            requires_approval=False,
        )

    def _resolve_workspace_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return candidate.resolve()

    def _effect_fingerprint(
        self, tool_call: ToolCallRequest, assessment: _EffectAssessment | None = None,
    ) -> str:
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
            target = (assessment.edit_target if assessment is not None and assessment.edit_target
                      else self._resolve_write_target(tool_call.name, raw_path))
            if target is not None:
                payload["target"] = str(target)
        if assessment is not None:
            if assessment.edit_binding is not None:
                payload["edit_content"] = assessment.edit_binding
            if assessment.exec_cwd is not None:
                payload["cwd"] = str(assessment.exec_cwd)
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

    def _assess_write_target(self, call: ToolCallRequest, raw_path: str) -> Path | None:
        target = self._resolve_write_target(call.name, raw_path)
        if target is None:
            # Never let a later hook turn a denied target into an unbound write.
            self._denied_write_targets.add(call.id)
        else:
            self._assessed_write_targets[call.id] = target
        return target

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
        total = max(len(before_lines), len(after_lines), 1)
        # Exclude unchanged context before the matcher can treat repeated
        # lines as noise (e.g. a one-line change above a generated section).
        start, before_end, after_end = 0, len(before_lines), len(after_lines)
        while start < min(before_end, after_end) and before_lines[start] == after_lines[start]:
            start += 1
        while before_end > start and after_end > start and before_lines[before_end - 1] == after_lines[after_end - 1]:
            before_end -= 1
            after_end -= 1
        matcher = difflib.SequenceMatcher(a=before_lines[start:before_end], b=after_lines[start:after_end])
        changed = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                changed += max(i2 - i1, j2 - j1)
        return changed, total

    # Files at or below this many lines are considered trivial state; a full
    # rewrite is not "risky" because there is nothing of substance to lose.
    # The absolute ``changed > 200`` guard still catches genuinely large edits.
    _SMALL_FILE_LINE_THRESHOLD = 10

    def _is_large_replacement(self, before: str, after: str) -> bool:
        changed, total = self._changed_line_count(before, after)
        return changed > 200 or (total > self._SMALL_FILE_LINE_THRESHOLD and changed / total > 0.5)

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
        if self._is_large_replacement(old_content, new_content):
            return f"overwrite a large portion of {path.name}"
        return None

    def _risky_edit_reason(
        self, path: Path, candidate: EditCandidate, replace_all: bool,
    ) -> str | None:
        if self._is_agent_state_file(path) or self._is_autonomous_managed_path(path):
            return None
        if replace_all:
            return f"apply a bulk replace in {path.name}"

        if candidate.before is None:
            return None
        if self._is_large_replacement(
            candidate.before.decode("utf-8"), candidate.after.decode("utf-8"),
        ):
            return f"edit a large portion of {path.name}"
        return None
