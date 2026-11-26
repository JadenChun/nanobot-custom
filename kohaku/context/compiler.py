"""Compile complete prompt sections with declared capabilities and budgets."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

from kohaku.context.budget import TokenMeter
from kohaku.context.types import (
    SYSTEM_BLOCKS_META,
    SYSTEM_SEPARATOR,
    CompiledPrompt,
    PromptSection,
    SystemTextBlock,
    TurnContextEnvelope,
    TurnKind,
)
from kohaku.runtime_context import RuntimeContextBlock


def truncate_section_to_tokens(text: str, max_tokens: int, *, hint: str | None = None) -> str:
    """Include the source-reading hint within the section's token allowance."""
    suffix = hint or (
        f"\n\n[Truncated to fit {max_tokens}-token budget; "
        "use read_file on the source file if full text is needed.]"
    )
    return TokenMeter.truncate_text(text, max_tokens, suffix=suffix)


class PromptCompiler:
    """Assemble deterministic system tiers and resolve turn-only skill input."""

    @staticmethod
    def compile_system_prompt(
        sections: Iterable[PromptSection], *, tool_names: frozenset[str] | None = None,
    ) -> CompiledPrompt:
        static: list[str] = []
        snapshot: list[str] = []
        mcp: list[str] = []
        counts: dict[str, int] = {}
        for section in sorted(sections, key=lambda item: item.priority):
            if tool_names is not None and (
                not section.requires_all_tools.issubset(tool_names)
                or (section.requires_any_tools and section.requires_any_tools.isdisjoint(tool_names))
            ):
                continue
            text = section.content.strip()
            if not text or section.tier == "turn_delta":
                continue
            if section.max_tokens is not None:
                text = truncate_section_to_tokens(text, section.max_tokens)
            counts[section.id] = TokenMeter.count_text(text)
            if section.id.startswith("mcp:"):
                mcp.append(text)
            (static if section.tier == "static" else snapshot).append(text)
        if mcp:
            counts["mcp_instructions"] = TokenMeter.count_text(SYSTEM_SEPARATOR.join(mcp))
        return CompiledPrompt(SYSTEM_SEPARATOR.join(static), SYSTEM_SEPARATOR.join(snapshot), counts)

    @staticmethod
    def build_turn_envelope(
        blocks: Sequence[RuntimeContextBlock],
        *,
        turn_kind: TurnKind,
        include_skill_context: bool,
        user_text: str,
        explicit_skill: Callable[[str], RuntimeContextBlock | None],
    ) -> TurnContextEnvelope:
        if turn_kind is TurnKind.USER and include_skill_context:
            skill = explicit_skill(user_text)
            if skill is not None:
                return TurnContextEnvelope((*blocks, skill))
        return TurnContextEnvelope(tuple(blocks))


def split_system_blocks(message: dict[str, Any]) -> list[SystemTextBlock] | None:
    """Recover compiler block boundaries from a validated portable message."""
    if message.get("role") != "system":
        return None
    metadata: object = message.get("_meta")
    if not isinstance(metadata, dict):
        return None
    raw = cast(dict[object, object], metadata).get(SYSTEM_BLOCKS_META)
    if not isinstance(raw, list):
        return None
    rows = cast(list[object], raw)
    if not 1 <= len(rows) <= 2:
        return None
    blocks: list[SystemTextBlock] = []
    for item in rows:
        if not isinstance(item, dict):
            return None
        row = cast(dict[object, object], item)
        text = row.get("text")
        if row.get("type") != "text" or not isinstance(text, str) or not text:
            return None
        blocks.append({"type": "text", "text": text})
    return blocks if "".join(block["text"] for block in blocks) == message.get("content") else None


def use_native_system_blocks(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Expand a portable system message for a provider with split caches."""
    for index, message in enumerate(messages):
        blocks = split_system_blocks(message)
        if blocks and len(blocks) == 2:
            converted = {**message, "content": blocks}
            return [*messages[:index], converted, *messages[index + 1:]], True
    return messages, False
