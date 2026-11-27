"""Conservative two-stage auto-reviewer for live Sandbox approval escalations."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from kohaku.agent.permission_rules import match_explicit_rule, sandbox_exec_risk_reason
from kohaku.agent.tools.shell import DEFAULT_EXEC_DENY_PATTERNS
from kohaku.providers.base import ToolCallRequest

if TYPE_CHECKING:
    from kohaku.config.schema import ToolPermissionsConfig
    from kohaku.utils.llm_runtime import LLMRuntime

AutoReviewDecision = Literal["approve", "ask_user"]

DEFAULT_AUTO_REVIEW_TIMEOUT_S = 20.0
DEFAULT_AUTO_REVIEW_MAX_TOKENS = 1024

_COMPILED_EXEC_DENY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern) for pattern in DEFAULT_EXEC_DENY_PATTERNS
)

_CREDENTIAL_PATH_PATTERN = re.compile(
    r"(?:~|\$HOME|\$\{HOME\}|/[^\s;|&'\"]*)/?\.(?:ssh|aws|gnupg|kube|netrc)\b"
    r"|(?<![A-Za-z0-9_.-])(?:/etc/shadow|/etc/sudoers)\b",
    re.IGNORECASE,
)

_REMOTE_PIPE_TO_SHELL_PATTERN = re.compile(
    r"\b(?:curl|wget)\b[^|\n]*\|\s*(?:sudo\s+|env\s+)?(?:ba|z|da|k|a)?sh\b",
    re.IGNORECASE,
)

_THINK_BLOCK_RE = re.compile(
    r"^(?:\s*<think>.*?</think>\s*)+",
    re.DOTALL | re.IGNORECASE,
)

_CODE_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)

_SYSTEM_PROMPT = (
    "You are a conservative security reviewer evaluating whether a proposed local "
    "Sandbox escalation command may be auto-approved without interrupting the user.\n\n"
    "Rules:\n"
    "- Return {\"decision\": \"approve\", \"reason\": \"<short reason>\"} ONLY when the "
    "proposed command directly fulfills the user's explicit request, stays bounded to "
    "the active workspace or expected local project tooling, and does not exfiltrate "
    "secrets or damage host state.\n"
    "- When in doubt, or if the command is broader than requested, accesses sensitive "
    "data, or could cause unexpected side effects, return "
    "{\"decision\": \"ask_user\", \"reason\": \"<short reason>\"}.\n"
    "- Respond ONLY with a single valid JSON object matching "
    "{\"decision\": \"approve\" | \"ask_user\", \"reason\": \"<short reason>\"}. "
    "Do not include any other text."
)


@dataclass(frozen=True, slots=True)
class AutoReviewResult:
    """Outcome of evaluating a live Sandbox escalation."""

    decision: AutoReviewDecision
    reason: str
    stage: Literal["filter", "llm"] = "llm"


def _strip_json_fences(raw: str) -> str:
    text = _THINK_BLOCK_RE.sub("", raw).strip()
    match = _CODE_FENCE_RE.match(text)
    if match is not None:
        return match.group(1).strip()
    return text


def check_auto_review_eligibility(
    call: ToolCallRequest,
    *,
    permissions: ToolPermissionsConfig | None,
    workspace: Path,
) -> str | None:
    """Return an ineligibility reason if Stage 1 requires human approval, else None."""
    if call.name != "exec":
        return f"non-exec tool '{call.name}' requires human approval"

    raw_command = call.arguments.get("command")
    command = raw_command.strip() if isinstance(raw_command, str) else ""
    if not command:
        return "empty exec command requires human approval"

    explicit = match_explicit_rule(
        permissions,
        call,
        workspace=workspace,
        command=command,
    )
    if explicit in {"ask", "deny"}:
        return f"explicit '{explicit}' permission rule requires human approval"

    risk_reason = sandbox_exec_risk_reason(command)
    if risk_reason is not None:
        return f"high-risk command ({risk_reason}) requires human approval"

    lowered = command.lower()
    for deny_pattern in _COMPILED_EXEC_DENY_PATTERNS:
        if deny_pattern.search(lowered):
            return "command matches built-in exec deny pattern"

    if _CREDENTIAL_PATH_PATTERN.search(command):
        return "command references user credential or system auth paths"

    if _REMOTE_PIPE_TO_SHELL_PATTERN.search(command):
        return "remote pipe-to-shell command requires human approval"

    return None


def _build_review_user_message(
    *,
    user_prompt: str,
    call: ToolCallRequest,
    workspace: Path,
    summary: str,
) -> str:
    raw_command = call.arguments.get("command")
    command = raw_command.strip() if isinstance(raw_command, str) else ""
    raw_working_dir = call.arguments.get("working_dir")
    working_dir = (
        raw_working_dir.strip()
        if isinstance(raw_working_dir, str) and raw_working_dir.strip()
        else str(workspace)
    )
    payload = {
        "user_request": user_prompt.strip() or "(no explicit user prompt provided)",
        "workspace": str(workspace),
        "approval_summary": summary,
        "tool_call": {
            "name": call.name,
            "command": command,
            "working_dir": working_dir,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_review_response(content: str | None) -> AutoReviewResult:
    if not isinstance(content, str) or not content.strip():
        return AutoReviewResult(
            decision="ask_user",
            reason="empty reviewer response",
            stage="llm",
        )
    try:
        parsed: object = json.loads(_strip_json_fences(content))
    except (json.JSONDecodeError, TypeError, ValueError):
        return AutoReviewResult(
            decision="ask_user",
            reason="malformed JSON from auto-reviewer",
            stage="llm",
        )
    if not isinstance(parsed, dict):
        return AutoReviewResult(
            decision="ask_user",
            reason="non-object JSON from auto-reviewer",
            stage="llm",
        )
    obj = cast(dict[str, object], parsed)
    decision = obj.get("decision")
    raw_reason = obj.get("reason")
    reason = (
        raw_reason.strip()[:200]
        if isinstance(raw_reason, str) and raw_reason.strip()
        else "no reason provided"
    )
    if decision != "approve":
        return AutoReviewResult(decision="ask_user", reason=reason, stage="llm")
    return AutoReviewResult(decision="approve", reason=reason, stage="llm")


async def review_sandbox_escalation(
    *,
    runtime: LLMRuntime | None,
    permissions: ToolPermissionsConfig | None,
    workspace: Path,
    call: ToolCallRequest,
    summary: str,
    user_prompt: str,
    timeout_s: float = DEFAULT_AUTO_REVIEW_TIMEOUT_S,
    max_tokens: int = DEFAULT_AUTO_REVIEW_MAX_TOKENS,
) -> AutoReviewResult:
    """Evaluate a live Sandbox escalation using Stage 1 filter + Stage 2 LLM review.

    Always fails closed to ``decision="ask_user"`` on any ineligible action,
    missing runtime, provider error, timeout, or invalid JSON.
    """
    ineligible_reason = check_auto_review_eligibility(
        call,
        permissions=permissions,
        workspace=workspace,
    )
    if ineligible_reason is not None:
        return AutoReviewResult(
            decision="ask_user",
            reason=ineligible_reason,
            stage="filter",
        )

    if runtime is None:
        return AutoReviewResult(
            decision="ask_user",
            reason="missing active LLM runtime",
            stage="filter",
        )

    user_message = _build_review_user_message(
        user_prompt=user_prompt,
        call=call,
        workspace=workspace,
        summary=summary,
    )
    messages: list[dict[str, object]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        response = await asyncio.wait_for(
            runtime.provider.chat(
                messages=messages,
                tools=None,
                model=runtime.model,
                max_tokens=max_tokens,
                temperature=0.0,
                reasoning_effort="none",
            ),
            timeout=timeout_s,
        )
    except TimeoutError:
        return AutoReviewResult(
            decision="ask_user",
            reason="auto-reviewer timed out",
            stage="llm",
        )
    except Exception as exc:
        return AutoReviewResult(
            decision="ask_user",
            reason=f"auto-reviewer error: {type(exc).__name__}",
            stage="llm",
        )

    if response.finish_reason == "error":
        return AutoReviewResult(
            decision="ask_user",
            reason="auto-reviewer provider returned error",
            stage="llm",
        )

    return _parse_review_response(response.content)
