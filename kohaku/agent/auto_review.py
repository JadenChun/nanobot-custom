"""Conservative two-stage auto-reviewer for live Sandbox approval escalations."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable, Mapping
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

_SYSTEM_AUTH_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:/etc/passwd|/etc/shadow|/etc/sudoers(?:\.d)?|/etc/pam\.d|/etc/ssh)(?:/|\b)",
    re.IGNORECASE,
)

_KOHAKU_CONFIG_PATTERN = re.compile(
    r"(?:(?:~|\$HOME|\$\{HOME\}|/[^\s;|&'\"]*)/)?\.(?:kohaku|nanobot)(?:/|\b)"
    r"|(?<![A-Za-z0-9_.-])config\.json\b",
    re.IGNORECASE,
)

_REMOTE_PIPE_TO_SHELL_PATTERN = re.compile(
    r"\b(?:curl|wget)\b[^|\n]*\|\s*(?:sudo\s+|env\s+)?(?:ba|z|da|k|a)?sh\b",
    re.IGNORECASE,
)

_DANGEROUS_URL_SCHEMES = ("file:", "javascript:", "data:", "view-source:")
_DANGEROUS_BROWSER_DEVICE_ACTIONS = frozenset({"upload", "eval", "evaluate"})
_FILE_OUTPUT_EXACT_OPTIONS = frozenset({
    "-o",
    "--file",
    "--filename",
    "--output",
    "--out",
    "--path",
    "--save",
    "--screenshot-dir",
})
_FILE_OUTPUT_OPTION_PREFIXES = (
    "--output",
    "--out",
    "--path",
    "--file",
    "--save",
    "--screenshot-dir",
)
_SENSITIVE_ARG_KEY_FRAGMENTS = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
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
    "You are a conservative security reviewer evaluating whether a proposed live "
    "permission escalation (shell command, browser/device/desktop action, file edit, "
    "or host tool call) may be auto-approved without interrupting the user.\n\n"
    "Rules:\n"
    "- Return {\"decision\": \"approve\", \"reason\": \"<short reason>\"} ONLY when the "
    "proposed tool call directly fulfills the user's explicit request, stays bounded to "
    "the requested target or active workspace, and does not exfiltrate secrets, modify "
    "Kohaku config/permissions, trigger irreversible external side effects/payments/"
    "submissions without explicit instruction, or damage host state.\n"
    "- When in doubt, or if the action is broader than requested, accesses sensitive "
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


def _resolve_target_path(raw_path: str, workspace: Path) -> Path:
    expanded = Path(raw_path).expanduser()
    if not expanded.is_absolute():
        return (workspace / expanded).resolve(strict=False)
    return expanded.resolve(strict=False)


def _is_within_workspace_or_media(raw_path: str, workspace: Path) -> bool:
    if not raw_path.strip():
        return False
    resolved = _resolve_target_path(raw_path, workspace)
    allowed_roots = (
        workspace.resolve(strict=False),
        (workspace / "media").resolve(strict=False),
        (Path.home() / ".kohaku" / "media").resolve(strict=False),
    )
    return any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots)


def _iter_string_values(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in cast(Mapping[object, object], value).values():
            yield from _iter_string_values(item)
    elif isinstance(value, (list, tuple)):
        for item in cast(list[object] | tuple[object, ...], value):
            yield from _iter_string_values(item)


def _check_sensitive_text(text: str, *, include_system_auth: bool = False) -> str | None:
    if _KOHAKU_CONFIG_PATTERN.search(text):
        return "references Kohaku configuration or runtime paths"
    if _CREDENTIAL_PATH_PATTERN.search(text):
        return "references user credential or system auth paths"
    if include_system_auth and _SYSTEM_AUTH_PATH_PATTERN.search(text):
        return "references system authentication paths"
    return None


def _is_file_output_option(option: str) -> bool:
    return option in _FILE_OUTPUT_EXACT_OPTIONS or option.startswith(_FILE_OUTPUT_OPTION_PREFIXES)


def _check_browser_or_device_args(
    tool_name: str,
    arguments: Mapping[str, object],
    workspace: Path,
) -> str | None:
    raw_args = arguments.get("args")
    if not isinstance(raw_args, (list, tuple)) or not raw_args:
        return f"empty or malformed {tool_name} args require human approval"
    args_list = list(cast(list[object] | tuple[object, ...], raw_args))
    if any(not isinstance(item, str) or not item.strip() for item in args_list):
        return f"malformed {tool_name} args require human approval"
    str_args = [cast(str, item).strip() for item in args_list]

    idx = 0
    while idx < len(str_args):
        raw = str_args[idx]
        lowered = raw.lower()
        if lowered.startswith(_DANGEROUS_URL_SCHEMES):
            return f"{tool_name} dangerous URL scheme requires human approval"
        if not lowered.startswith("-") and lowered in _DANGEROUS_BROWSER_DEVICE_ACTIONS:
            return f"{tool_name} action '{lowered}' requires human approval"
        if lowered in {"--eval", "--evaluate"}:
            return f"{tool_name} script evaluation requires human approval"
        if sensitive := _check_sensitive_text(raw):
            return f"{tool_name} {sensitive}"

        option, has_value, value = raw.partition("=")
        option_lower = option.lower()
        if _is_file_output_option(option_lower):
            out_path = value if has_value else (str_args[idx + 1] if idx + 1 < len(str_args) else "")
            if not out_path or not _is_within_workspace_or_media(out_path, workspace):
                return f"{tool_name} file output outside workspace/media requires human approval"
            if not has_value:
                idx += 2
                continue
        idx += 1
    return None


def _sanitize_value(
    value: object,
    *,
    max_str_chars: int,
    depth: int = 0,
) -> object:
    if depth > 4:
        return "<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= max_str_chars:
            return value
        return f"{value[:max_str_chars]}... (+{len(value) - max_str_chars} chars)"
    if isinstance(value, Mapping):
        sanitized: dict[str, object] = {}
        items = list(cast(Mapping[object, object], value).items())
        for key, item_val in items[:20]:
            key_str = str(key)
            key_lower = key_str.lower()
            if any(frag in key_lower for frag in _SENSITIVE_ARG_KEY_FRAGMENTS):
                sanitized[key_str] = "<redacted>"
            else:
                sanitized[key_str] = _sanitize_value(
                    item_val,
                    max_str_chars=max_str_chars,
                    depth=depth + 1,
                )
        if len(items) > 20:
            sanitized["_truncated_keys"] = len(items) - 20
        return sanitized
    if isinstance(value, (list, tuple)):
        seq = list(cast(list[object] | tuple[object, ...], value))
        out = [
            _sanitize_value(item, max_str_chars=max_str_chars, depth=depth + 1)
            for item in seq[:20]
        ]
        if len(seq) > 20:
            out.append(f"... (+{len(seq) - 20} items)")
        return out
    text = str(value)
    return text if len(text) <= max_str_chars else f"{text[:max_str_chars]}..."


def sanitize_tool_arguments(
    arguments: object,
    *,
    max_total_chars: int = 2000,
    max_str_chars: int = 500,
) -> dict[str, object]:
    """Return a sanitized, size-bounded copy of tool arguments for review/UI payloads."""
    if not isinstance(arguments, Mapping):
        return {}
    sanitized = cast(
        dict[str, object],
        _sanitize_value(arguments, max_str_chars=max_str_chars),
    )
    encoded = json.dumps(sanitized, ensure_ascii=False)
    if len(encoded) <= max_total_chars:
        return sanitized
    tighter = cast(
        dict[str, object],
        _sanitize_value(arguments, max_str_chars=120),
    )
    if len(json.dumps(tighter, ensure_ascii=False)) <= max_total_chars:
        return tighter
    return {
        "_truncated": True,
        "keys": list(sanitized.keys())[:20],
        "preview": encoded[:max_total_chars],
    }


def check_auto_review_eligibility(
    call: ToolCallRequest,
    *,
    permissions: ToolPermissionsConfig | None,
    workspace: Path,
) -> str | None:
    """Return an ineligibility reason if Stage 1 requires human approval, else None."""
    if not isinstance(call.arguments, Mapping):
        return "malformed tool arguments require human approval"

    if call.name == "exec":
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

        if sensitive := _check_sensitive_text(command):
            return f"command {sensitive}"

        if _REMOTE_PIPE_TO_SHELL_PATTERN.search(command):
            return "remote pipe-to-shell command requires human approval"

        return None

    if call.name in {"write_file", "edit_file"}:
        raw_path = call.arguments.get("path")
        path_str = raw_path.strip() if isinstance(raw_path, str) else ""
        if not path_str:
            return f"empty {call.name} path requires human approval"

        target_path = _resolve_target_path(path_str, workspace)
        explicit = match_explicit_rule(
            permissions,
            call,
            workspace=workspace,
            target_path=target_path,
        )
        if explicit in {"ask", "deny"}:
            return f"explicit '{explicit}' permission rule requires human approval"

        for candidate in (path_str, target_path.as_posix()):
            if sensitive := _check_sensitive_text(candidate, include_system_auth=True):
                return f"{call.name} target {sensitive}"

        return None

    explicit = match_explicit_rule(
        permissions,
        call,
        workspace=workspace,
    )
    if explicit in {"ask", "deny"}:
        return f"explicit '{explicit}' permission rule requires human approval"

    if call.name in {"agent_browser", "agent_device"}:
        return _check_browser_or_device_args(call.name, call.arguments, workspace)

    if call.name == "desktop_use":
        action = call.arguments.get("action")
        if not isinstance(action, str) or not action.strip():
            return "empty or malformed desktop_use action requires human approval"

    if call.name == "run_cli_app" and not call.arguments:
        return "empty run_cli_app arguments require human approval"

    for text in _iter_string_values(call.arguments):
        if sensitive := _check_sensitive_text(text):
            return f"{call.name} {sensitive}"

    return None


def _build_review_user_message(
    *,
    user_prompt: str,
    call: ToolCallRequest,
    workspace: Path,
    summary: str,
) -> str:
    if call.name == "exec":
        raw_command = call.arguments.get("command")
        command = raw_command.strip() if isinstance(raw_command, str) else ""
        raw_working_dir = call.arguments.get("working_dir")
        working_dir = (
            raw_working_dir.strip()
            if isinstance(raw_working_dir, str) and raw_working_dir.strip()
            else str(workspace)
        )
        tool_payload: dict[str, object] = {
            "name": call.name,
            "command": command,
            "working_dir": working_dir,
        }
    else:
        tool_payload = {
            "name": call.name,
            "arguments": sanitize_tool_arguments(call.arguments),
        }

    payload = {
        "user_request": user_prompt.strip() or "(no explicit user prompt provided)",
        "workspace": str(workspace),
        "approval_summary": summary,
        "tool_call": tool_payload,
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
