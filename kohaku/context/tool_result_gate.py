"""Bound model-facing tool text and retain complete results when storage permits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from loguru import logger

from kohaku.utils.helpers import maybe_persist_tool_result
from kohaku.utils.runtime import ensure_nonempty_tool_result

DEFAULT_TOOL_RESULT_CHARS = 16_000
MIN_TOOL_RESULT_CHARS = 2_000
PROTECTED_FILESYSTEM_TOOLS = frozenset({"read_file", "grep"})


def tool_text_blocks(content: object) -> tuple[tuple[int | None, str], ...]:
    """Select scalar text or text blocks while leaving other blocks intact."""
    if isinstance(content, str):
        return ((None, content),)
    if not isinstance(content, list):
        return ()
    result: list[tuple[int | None, str]] = []
    for index, item in enumerate(cast(list[object], content)):
        if not isinstance(item, dict):
            return ()
        block = cast(dict[object, object], item)
        if not isinstance(block.get("type"), str):
            return ()
        text = block.get("text")
        if block["type"] == "text":
            if not isinstance(text, str):
                return ()
            result.append((index, text))
    return tuple(result)


def tool_text_preview(text: str, reference: str | None = None) -> str:
    """Keep the start and end of the output and describe the omitted section."""
    head, tail = text[:800], text[-400:]
    omitted = text[len(head):-len(tail)]
    action = f' Call read_artifact(reference="{reference}") to read more.' if reference else ""
    return (
        f"{head}\n[Omitted {len(omitted)} characters and {omitted.count(chr(10))} line breaks."
        f"{action}]\n{tail}"
    )


class ToolResultGate:
    """Apply a character bound to each complete or partial text result."""

    def __init__(self, workspace: Path | None, session_key: str | None, max_chars: int) -> None:
        self.workspace = workspace
        self.session_key = session_key
        self.max_chars = max(MIN_TOOL_RESULT_CHARS, max_chars)

    def normalize(
        self, tool_call_id: str, tool_name: str, result: Any, *, persist: bool = True,
    ) -> Any:
        content = ensure_nonempty_tool_result(tool_name, result)
        if tool_name == "read_artifact":
            return content
        oversized = {
            index: text for index, text in tool_text_blocks(content) if len(text) > self.max_chars
        }
        if not oversized:
            return content

        def bound(text: str, block_id: str) -> str:
            if (persist and self.workspace is not None and self.workspace.is_dir()
                    and tool_name not in PROTECTED_FILESYSTEM_TOOLS):
                try:
                    return maybe_persist_tool_result(
                        self.workspace, self.session_key, block_id, text, max_chars=self.max_chars,
                    )
                except (OSError, UnicodeError):
                    logger.warning("Cannot save complete output for tool {}", tool_name)
            return tool_text_preview(text)

        scalar = oversized.get(None)
        if scalar is not None:
            return bound(scalar, tool_call_id)
        # The text selection validated the type of every block in this list.
        blocks = cast(list[dict[str, Any]], content)
        return [
            {**item, "text": bound(oversized[index], f"{tool_call_id}_text_{index}")}
            if index in oversized else item
            for index, item in enumerate(blocks)
        ]
