"""Conservative two-stage auto-reviewer for live Sandbox approval escalations."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from kohaku.agent.permission_rules import match_explicit_rule, sandbox_exec_risk_reason
from kohaku.agent.tools.shell import DEFAULT_EXEC_DENY_PATTERNS
from kohaku.providers.base import ToolCallRequest

if TYPE_CHECKING:
    from kohaku.config.schema import ToolPermissionsConfig
    from kohaku.utils.llm_runtime import LLMRuntime

AutoReviewDecision = Literal["approve", "ask_user"]

DEFAULT_AUTO_REVIEW_TIMEOUT_S = 30.0
DEFAULT_AUTO_REVIEW_MAX_TOKENS = 1024
DEFAULT_RECENT_CONVERSATION_TURNS = 6
DEFAULT_RECENT_CONVERSATION_MAX_CHARS = 400

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
    "You are a security reviewer evaluating whether a proposed Sandbox escalation "
    "shell command may be auto-approved without interrupting the user.\n\n"
    "Context:\n"
    "- Every command sent to you is requesting to run outside the OS sandbox (for example, "
    "to talk to Docker/local daemons, run user-installed local CLI utilities, or use host "
    "dev/project tooling). Do NOT reject a command merely because it runs outside the "
    "sandbox or invokes a local CLI binary located outside the active workspace folder.\n"
    "- Use both `user_request` and `recent_conversation` (when provided) to understand "
    "short follow-up messages (e.g. 'ok it is sent now', 'go ahead', 'do the same for X').\n\n"
    "Rules:\n"
    "- Return {\"decision\": \"approve\", \"reason\": \"<short reason>\"} when the "
    "proposed command aligns with the user's request and recent conversation, performs "
    "routine local project/tooling/service work, and does not destroy data, exfiltrate "
    "secrets, or trigger unexpected external side effects.\n"
    "- Return {\"decision\": \"ask_user\", \"reason\": \"<short reason>\"} when the "
    "command is destructive, accesses sensitive credentials/private data, performs "
    "unrequested external actions (pushing code, publishing, sending external messages, "
    "modifying remote/production infrastructure), or is unrelated to the user's intent.\n"
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


def _extract_message_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in cast(list[object], content):
            if isinstance(item, Mapping):
                block = cast(Mapping[str, Any], item)
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    text = str(block["text"]).strip()
                    if text:
                        parts.append(text)
        return "\n".join(parts).strip()
    return ""


def extract_recent_conversation(
    messages: Sequence[Mapping[str, Any]] | None,
    *,
    current_user_prompt: str = "",
    max_turns: int = DEFAULT_RECENT_CONVERSATION_TURNS,
    max_chars_per_turn: int = DEFAULT_RECENT_CONVERSATION_MAX_CHARS,
) -> list[dict[str, str]]:
    """Extract recent trusted user/assistant text turns, excluding all tool output."""
    if not messages or max_turns <= 0:
        return []
    extracted: list[dict[str, str]] = []
    for raw in messages:
        role = raw.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = _extract_message_text(raw.get("content"))
        if not text:
            continue
        if len(text) > max_chars_per_turn:
            text = text[: max_chars_per_turn - 1].rstrip() + "…"
        extracted.append({"role": str(role), "text": text})
    normalized_current = current_user_prompt.strip()
    if (
        normalized_current
        and extracted
        and extracted[-1]["role"] == "user"
        and extracted[-1]["text"] == normalized_current[:max_chars_per_turn]
    ):
        extracted.pop()
    return extracted[-max_turns:]


def _build_review_user_message(
    *,
    user_prompt: str,
    call: ToolCallRequest,
    workspace: Path,
    summary: str,
    recent_conversation: Sequence[Mapping[str, str]] | None = None,
) -> str:
    raw_command = call.arguments.get("command")
    command = raw_command.strip() if isinstance(raw_command, str) else ""
    raw_working_dir = call.arguments.get("working_dir")
    working_dir = (
        raw_working_dir.strip()
        if isinstance(raw_working_dir, str) and raw_working_dir.strip()
        else str(workspace)
    )
    payload: dict[str, Any] = {
        "user_request": user_prompt.strip() or "(no explicit user prompt provided)",
        "workspace": str(workspace),
        "approval_summary": summary,
        "tool_call": {
            "name": call.name,
            "command": command,
            "working_dir": working_dir,
        },
    }
    if recent_conversation:
        payload["recent_conversation"] = [
            {"role": str(item["role"]), "text": str(item["text"])}
            for item in recent_conversation
            if "role" in item and "text" in item
        ]
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
    recent_conversation: Sequence[Mapping[str, str]] | None = None,
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
        recent_conversation=recent_conversation,
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
