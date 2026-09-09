"""Detached opt-in public execution traces; persisted detail remains internal."""

from collections.abc import Mapping, Sequence
from typing import Any

from nanobot.agent.message_content import content_to_text
from nanobot.runtime_context import public_history_message


def sanitize_trace(messages: Any) -> list[dict[str, Any]]:
    """Return a detached, bounded SDK trace without internal/raw fields.

    The canonical loop retains the full provider exchange for session
    persistence.  SDK callers only receive this reduced view when explicitly
    opting in: system prompts, internal metadata, media payloads, and tool
    arguments are omitted while textual content remains useful for debugging.
    """
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return []

    detached: list[dict[str, Any]] = []
    for raw in messages:
        if not isinstance(raw, Mapping):
            continue
        raw = public_history_message(raw)
        role = raw.get("role")
        if not isinstance(role, str) or role not in {"user", "assistant", "tool"} or raw.get("_hidden_history"):
            continue

        entry: dict[str, Any] = {"role": str(role)}
        if "content" in raw:
            content = raw.get("content")
            if isinstance(content, list):
                content = [block for block in content if not (
                    isinstance(block, Mapping) and block.get("type") in
                    ("thinking", "reasoning", "redacted_thinking", "reasoning_content")
                )]
            elif isinstance(content, Mapping) and content.get("type") in (
                "thinking", "reasoning", "redacted_thinking", "reasoning_content"
            ):
                content = None
            text = content_to_text(content, max_chars=4_000)
            # Keep the ellipsis inside the limit so a defensive second pass is stable.
            entry["content"] = text if len(text) <= 4_000 else text[:3_997].rstrip() + "..."

        if role == "assistant" and isinstance(raw.get("tool_calls"), Sequence):
            calls: list[dict[str, str]] = []
            for raw_call in raw["tool_calls"]:
                if not isinstance(raw_call, Mapping):
                    continue
                function = raw_call.get("function")
                name = function.get("name") if isinstance(function, Mapping) else raw_call.get("name")
                call: dict[str, str] = {}
                if raw_call.get("id"):
                    call["id"] = str(raw_call["id"])
                if name:
                    call["name"] = str(name)
                if call:
                    calls.append(call)
            if calls:
                entry["tool_calls"] = calls
        elif role == "tool":
            if raw.get("tool_call_id"):
                entry["tool_call_id"] = str(raw["tool_call_id"])
            if raw.get("name"):
                entry["name"] = str(raw["name"])

        detached.append(entry)
    return detached

