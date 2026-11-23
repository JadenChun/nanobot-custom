"""Pure parser for captured legacy HISTORY.md contents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

MAX_LEGACY_HISTORY_BYTES = 64 * 1024 * 1024
_MAX_CURSOR = (1 << 63) - 1

_LEGACY_ENTRY_START_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*")
_LEGACY_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*")
_LEGACY_RAW_MESSAGE_RE = re.compile(
    r"^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s+[A-Z][A-Z0-9_]*(?:\s+\[tools:\s*[^\]]+\])?:"
)


@dataclass(frozen=True, slots=True)
class LegacyHistoryEntry:
    """One detached, immutable row parsed from a legacy history capture."""

    cursor: int
    timestamp: str
    content: str


def parse_legacy_history(
    captured_bytes: bytes,
    *,
    fallback_timestamp: str,
    base_cursor: int = 0,
) -> tuple[LegacyHistoryEntry, ...]:
    """Parse captured HISTORY.md bytes without filesystem or clock access.

    Rows begin at ``base_cursor + 1``. The fallback timestamp is supplied by
    the caller so parsing remains deterministic.
    """
    if not isinstance(cast(object, captured_bytes), bytes):
        raise TypeError("captured_bytes must be bytes")
    if len(captured_bytes) > MAX_LEGACY_HISTORY_BYTES:
        raise ValueError("captured HISTORY.md exceeds 64 MiB")
    if not isinstance(cast(object, fallback_timestamp), str):
        raise TypeError("fallback_timestamp must be a string")
    if isinstance(base_cursor, bool) or not isinstance(cast(object, base_cursor), int):
        raise TypeError("base_cursor must be an integer")
    if not 0 <= base_cursor <= _MAX_CURSOR:
        raise ValueError("base_cursor must be between 0 and the signed 63-bit maximum")

    normalized = (
        captured_bytes.decode("utf-8", errors="replace")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )
    if not normalized:
        return ()

    entries: list[LegacyHistoryEntry] = []
    cursor = base_cursor
    for chunk in _split_legacy_history_chunks(normalized):
        if cursor == _MAX_CURSOR:
            raise OverflowError("legacy history cursor exceeds the signed 63-bit maximum")
        cursor += 1

        timestamp = fallback_timestamp
        content = chunk
        match = _LEGACY_TIMESTAMP_RE.match(chunk)
        if match:
            timestamp = match.group(1)
            remainder = chunk[match.end() :].lstrip()
            if remainder:
                content = remainder

        entries.append(LegacyHistoryEntry(cursor, timestamp, content))

    return tuple(entries)


def _split_legacy_history_chunks(text: str) -> list[str]:
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    saw_blank_separator = False

    for line in lines:
        if saw_blank_separator and line.strip() and current:
            chunks.append("\n".join(current).strip())
            current = [line]
            saw_blank_separator = False
            continue
        if _should_start_new_legacy_chunk(line, current):
            chunks.append("\n".join(current).strip())
            current = [line]
            saw_blank_separator = False
            continue
        current.append(line)
        saw_blank_separator = not line.strip()

    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _should_start_new_legacy_chunk(line: str, current: list[str]) -> bool:
    if not current:
        return False
    if not _LEGACY_ENTRY_START_RE.match(line):
        return False
    if _is_raw_legacy_chunk(current) and _LEGACY_RAW_MESSAGE_RE.match(line):
        return False
    return True


def _is_raw_legacy_chunk(lines: list[str]) -> bool:
    first_nonempty = next((line for line in lines if line.strip()), "")
    match = _LEGACY_TIMESTAMP_RE.match(first_nonempty)
    if not match:
        return False
    return first_nonempty[match.end() :].lstrip().startswith("[RAW]")
