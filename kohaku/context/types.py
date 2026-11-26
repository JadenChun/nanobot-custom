"""Stable system prompt tiers and turn-local context contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from kohaku.runtime_context import RuntimeContextBlock

PromptTier = Literal["static", "session_snapshot", "turn_delta"]
RetentionPolicy = Literal["turn_only", "persist_compact", "ephemeral_body"]
SYSTEM_BLOCKS_META = "system_prompt_blocks"
SYSTEM_SEPARATOR = "\n\n---\n\n"


class TurnKind(Enum):
    USER = "user"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class PromptSection:
    id: str
    tier: PromptTier
    content: str
    priority: int = 100
    max_tokens: int | None = None
    requires_all_tools: frozenset[str] = frozenset()
    requires_any_tools: frozenset[str] = frozenset()


class SystemTextBlock(TypedDict):
    type: Literal["text"]
    text: str


@dataclass(frozen=True, slots=True)
class CompiledPrompt:
    static_text: str
    session_snapshot_text: str
    section_tokens: dict[str, int]

    @property
    def full_system_text(self) -> str:
        return SYSTEM_SEPARATOR.join(
            text for text in (self.static_text, self.session_snapshot_text) if text
        )

    def system_blocks(self) -> list[SystemTextBlock]:
        """Return the two cacheable portions in exact concatenation order."""
        blocks: list[SystemTextBlock] = []
        for text in (self.static_text, self.session_snapshot_text):
            if text:
                blocks.append({"type": "text", "text": (SYSTEM_SEPARATOR if blocks else "") + text})
        return blocks


@dataclass(frozen=True, slots=True)
class PromptSourceFile:
    path: Path
    root: Path
    allow_folder_grants: bool


@dataclass(frozen=True, slots=True)
class SessionPromptSnapshot:
    """Frozen prompt inputs for the active session and its selected workspace."""

    workspace: Path
    generation: int
    incarnation: str
    sections: tuple[PromptSection, ...]
    source_files: tuple[PromptSourceFile, ...] = ()


@dataclass(frozen=True, slots=True)
class TurnContextEnvelope:
    """Context that accompanies a fresh turn and has explicit retention."""

    blocks: tuple[RuntimeContextBlock, ...] = ()
