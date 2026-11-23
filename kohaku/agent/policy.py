"""Internal execution policy for the main agent harness."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Collection
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from kohaku.agent.launch_confinement import (
    LAUNCH_CONFINEMENT_UNPROVEN as _LAUNCH_CONFINEMENT_UNPROVEN,
)
from kohaku.agent.launch_confinement import (
    LaunchConfinementAttestation,
    LaunchConfinementAttestor,
    launch_confinement_covers,
    nested_launch_id,
    nested_launch_path,
    register_launch_confinement_attestor,
    resolve_launch_confinement_attestor,
    unregister_launch_confinement_attestor,
)
from kohaku.agent.permission_rules import (
    EXTERNAL_EXEC_PATTERNS,
    LOCAL_DESTRUCTIVE_EXEC_PATTERNS,
    READ_ONLY_DESKTOP_ACTIONS,
    RISKY_EXEC_PATTERNS,
    WriteTargetGuard,
    changed_line_count,
    is_agent_state_file,
    is_autonomous_managed_path,
    is_large_replacement,
    match_exec_risk_reason,
    match_explicit_rule,
)
from kohaku.agent.tools.base import Tool
from kohaku.agent.tools.filesystem import EditCandidate, EditContentBinding, EditFileTool
from kohaku.agent.tools.shell import ExecTool
from kohaku.agent.turn import ToolOutcome, TurnContext
from kohaku.agent.turn_admission import EffectApprovalAuthority
from kohaku.context_repo import ContextRepoManager
from kohaku.providers.base import ToolCallRequest

if TYPE_CHECKING:
    from kohaku.agent.tools.registry import ToolRegistry
    from kohaku.config.schema import PermissionProfile, ToolPermissionsConfig

__all__ = [
    "CompositeToolPolicy",
    "DelegatedReadOnlyPolicy",
    "LaunchConfinementAttestation",
    "LaunchConfinementAttestor",
    "RiskyActionPolicy",
    "ToolPolicy",
    "ToolPolicyDecision",
    "nested_launch_id",
    "nested_launch_path",
    "register_launch_confinement_attestor",
    "resolve_launch_confinement_attestor",
    "unregister_launch_confinement_attestor",
]

_APPROVAL_REQUIRED = "approval_required"
_POLICY_BLOCKED = "policy_blocked"


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
                from kohaku.agent.capabilities import role_capabilities

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
            from kohaku.agent.tools.agent_browser import classify_browser_action

            if cast(dict[object, object], arguments).get("working_dir") is not None:
                return "agent_browser caller working_dir override is not permitted"
            args = cast(dict[object, object], arguments).get("args")
            if not isinstance(args, (list, tuple)):
                return "agent_browser action is unknown"
            if classify_browser_action(cast(list[object] | tuple[object, ...], args)) != "read_navigation":
                return "agent_browser action is state-changing or unknown"
        elif name == "agent_device":
            from kohaku.agent.tools.agent_device import classify_device_action

            if cast(dict[object, object], arguments).get("working_dir") is not None:
                return "agent_device caller working_dir override is not permitted"
            args = cast(dict[object, object], arguments).get("args")
            if not isinstance(args, (list, tuple)):
                return "agent_device action is unknown"
            if classify_device_action(cast(list[object] | tuple[object, ...], args)) != "read_navigation":
                return "agent_device action is state-changing or unknown"
        elif name == "social_crawl":
            from kohaku.agent.tools.social_crawl import classify_crawler_action

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
    """Evaluate tool permissions, hard confinement guards, and exact-effect approvals."""

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
    # Declarative tool permissions (profile + allow/ask/deny rules).
    # ``None`` retains legacy ``strict`` evaluation for direct callers/tests;
    # AgentLoop passes ``tools_config.permissions`` (default ``autonomous``).
    permissions: ToolPermissionsConfig | None = None
    _tools: ToolRegistry | None = field(default=None, repr=False, compare=False)
    _write_guard: WriteTargetGuard = field(init=False, repr=False, compare=False)
    _risky_calls: dict[str, tuple[str, str]] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )

    _LOCAL_DESTRUCTIVE_EXEC_PATTERNS = LOCAL_DESTRUCTIVE_EXEC_PATTERNS
    _EXTERNAL_EXEC_PATTERNS = EXTERNAL_EXEC_PATTERNS
    _RISKY_EXEC_PATTERNS = RISKY_EXEC_PATTERNS

    def __post_init__(self) -> None:
        self._write_guard = WriteTargetGuard(
            workspace=self.workspace,
            resolver=self.write_target_resolver,
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

    def batch_has_mutation(self, tool_calls: list[ToolCallRequest]) -> bool:
        return any(tc.name in self.mutating_tool_names() for tc in tool_calls)

    async def evaluate(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_calls: list[ToolCallRequest],
    ) -> ToolPolicyDecision:
        _ = messages
        if not tool_calls:
            return ToolPolicyDecision()

        self._write_guard.clear()
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
        return self._write_guard.validate_binding(tool_call, tool, params)

    def _effective_profile(self) -> PermissionProfile:
        return "strict" if self.permissions is None else self.permissions.profile

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
            path = self._write_guard.assess(call, raw_path)
            if path is None:
                return _EffectAssessment()
            candidate = EditFileTool.prepare_edit(path, **call.arguments)
            if isinstance(candidate, str):
                return _EffectAssessment(refusal=candidate)
            explicit = match_explicit_rule(
                self.permissions, call, workspace=self.workspace, target_path=path,
            )
            if explicit == "deny":
                return self._refuse_denied_rule(call, target_path=path)
            if explicit == "allow":
                edit_reason: str | None = None
            elif explicit == "ask":
                edit_reason = (
                    self._risky_edit_reason(path, candidate, bool(call.arguments.get("replace_all")))
                    or f"edit {path.name}"
                )
            elif self._effective_profile() == "strict":
                edit_reason = self._risky_edit_reason(
                    path, candidate, bool(call.arguments.get("replace_all")),
                )
            else:
                edit_reason = None
            return _EffectAssessment(
                reason=edit_reason or previous_reason,
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

    def _risky_reason(self, tool_call: ToolCallRequest) -> str | ToolPolicyDecision | None:
        profile = self._effective_profile()
        if tool_call.name == "exec":
            command, working_dir = ExecTool.request_location(**tool_call.arguments)
            if not command:
                return None
            launch_path = nested_launch_path(command)
            if launch_path is not None and not launch_confinement_covers(
                workspace=self.workspace,
                command=command,
                working_dir=working_dir,
                launch_path=launch_path,
                attestor=self.launch_confinement_attestor,
            ):
                # Hard non-bypassable invariant regardless of profile or rules.
                return self._refuse_launch(launch_path)
            explicit = match_explicit_rule(
                self.permissions, tool_call, workspace=self.workspace, command=command,
            )
            if explicit == "deny":
                return self._refuse_denied_rule(tool_call, detail=command)
            if explicit == "allow":
                return None
            if explicit == "ask":
                return (
                    match_exec_risk_reason(command, include_local=True)
                    or f"run shell command: {command.strip()[:80]}"
                )
            if profile == "autonomous":
                return None
            if launch_path is not None:
                return "launch a nested agent with reduced sandbox or approval protections"
            return match_exec_risk_reason(command, include_local=(profile == "strict"))

        if tool_call.name == "write_file":
            raw_path = str(tool_call.arguments.get("path") or "")
            resolved = self._write_guard.assess(tool_call, raw_path)
            if resolved is None:
                return None
            explicit = match_explicit_rule(
                self.permissions, tool_call, workspace=self.workspace, target_path=resolved,
            )
            if explicit == "deny":
                return self._refuse_denied_rule(tool_call, target_path=resolved)
            if explicit == "allow":
                return None
            if explicit == "ask":
                return (
                    self._risky_write_reason(
                        raw_path, str(tool_call.arguments.get("content") or ""), resolved,
                    )
                    or f"write {resolved.name}"
                )
            if profile != "strict":
                return None
            return self._risky_write_reason(
                raw_path, str(tool_call.arguments.get("content") or ""), resolved,
            )

        explicit = match_explicit_rule(self.permissions, tool_call, workspace=self.workspace)
        if explicit == "deny":
            return self._refuse_denied_rule(tool_call)
        if explicit == "allow":
            return None

        if tool_call.name == "cron":
            return "schedule a recurring automation" if (explicit == "ask" or profile == "strict") else None
        if tool_call.name == "agent_browser":
            if explicit == "ask":
                return "perform a browser action"
            if profile == "autonomous":
                return None
            from kohaku.agent.tools.agent_browser import classify_browser_action

            args: object = tool_call.arguments.get("args") or []
            if classify_browser_action(args) != "read_navigation":
                return "perform a state-changing browser action"
            return None
        if tool_call.name == "agent_device":
            if explicit == "ask":
                return "perform a device action"
            if profile == "autonomous":
                return None
            from kohaku.agent.tools.agent_device import classify_device_action

            args = tool_call.arguments.get("args") or []
            if classify_device_action(args) != "read_navigation":
                return "perform a state-changing device action"
            return None
        if tool_call.name == "desktop_use":
            if explicit == "ask":
                return "perform a desktop input action"
            if profile == "autonomous":
                return None
            action = tool_call.arguments.get("action")
            if not isinstance(action, str) or action not in READ_ONLY_DESKTOP_ACTIONS:
                return "perform a state-changing desktop input action"
            return None
        return f"run tool {tool_call.name}" if explicit == "ask" else None

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

    def _refuse_denied_rule(
        self,
        tool_call: ToolCallRequest,
        *,
        target_path: Path | None = None,
        detail: str | None = None,
    ) -> ToolPolicyDecision:
        target_label = (
            target_path.name
            if target_path is not None
            else (detail.strip()[:80] if detail else tool_call.name)
        )
        summary = f"tool {tool_call.name!r} ({target_label}) is denied by permission rules"
        return ToolPolicyDecision(
            action="respond",
            stop_reason=_POLICY_BLOCKED,
            response=f"Blocked by permission policy: {summary}.",
            metadata={
                "reason_id": "permission_rule_denied",
                "reasons": [summary],
                "summary": summary,
                "tool": tool_call.name,
                "requires_approval": False,
            },
            requires_approval=False,
        )

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
            target = (
                assessment.edit_target
                if assessment is not None and assessment.edit_target
                else self._write_guard.resolve(tool_call.name, raw_path)
            )
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

    def _changed_line_count(self, before: str, after: str) -> tuple[int, int]:
        return changed_line_count(before, after)

    def _is_large_replacement(self, before: str, after: str) -> bool:
        return is_large_replacement(before, after)

    def _risky_write_reason(
        self,
        raw_path: str,
        new_content: str,
        resolved_path: Path | None = None,
    ) -> str | None:
        if not raw_path:
            return None
        path = resolved_path or self._write_guard.resolve_workspace_path(raw_path)
        if is_agent_state_file(path) or is_autonomous_managed_path(self.context_manager, path):
            return None
        if not path.exists() or not path.is_file():
            return None
        try:
            old_content = path.read_text(encoding="utf-8")
        except Exception:
            return f"overwrite existing file {path.name}"
        if is_large_replacement(old_content, new_content):
            return f"overwrite a large portion of {path.name}"
        return None

    def _risky_edit_reason(
        self, path: Path, candidate: EditCandidate, replace_all: bool,
    ) -> str | None:
        if is_agent_state_file(path) or is_autonomous_managed_path(self.context_manager, path):
            return None
        if replace_all:
            return f"apply a bulk replace in {path.name}"
        if candidate.before is None:
            return None
        if is_large_replacement(
            candidate.before.decode("utf-8"), candidate.after.decode("utf-8"),
        ):
            return f"edit a large portion of {path.name}"
        return None
