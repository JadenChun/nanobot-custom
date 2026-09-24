"""Pure parsing tests for captured legacy HISTORY.md bytes."""

from dataclasses import FrozenInstanceError

import pytest

from nanobot.agent.legacy_history_codec import (
    MAX_LEGACY_HISTORY_BYTES,
    LegacyHistoryEntry,
    parse_legacy_history,
)
from nanobot.agent.memory import MemoryStore

FALLBACK_TIMESTAMP = "2030-01-02 03:04"
MAX_CURSOR = (1 << 63) - 1


@pytest.mark.parametrize(
    "captured",
    [
        (
            b" \r\n[2026-04-01 10:00] First event. \r\n\r\n"
            b"[2026-04-01 10:01] Second event.\r"
        ),
        (
            b"[2026-04-01 10:05] [RAW] 2 messages\n"
            b"[2026-04-01 10:04] USER: hello\n"
            b"[2026-04-01 10:04] ASSISTANT: hi\n"
            b"[2026-04-01 10:06] Normal event after raw block."
        ),
        b"[2026-03-25\xe2\x80\x932026-04-02] Multi-day summary.\n"
        b"[2026-03-26/27] Cross-day summary.",
        b"[2026-04-01 10:00] Broken \xff data still needs migration.\n\n"
        b"Legacy chunk without timestamp.",
    ],
)
def test_matches_memory_store_legacy_parser(captured: bytes, monkeypatch):
    memory_store = object.__new__(MemoryStore)
    monkeypatch.setattr(
        memory_store,
        "_legacy_fallback_timestamp",
        lambda: FALLBACK_TIMESTAMP,
    )
    reference = memory_store._parse_legacy_history(captured.decode("utf-8", errors="replace"))

    actual = parse_legacy_history(captured, fallback_timestamp=FALLBACK_TIMESTAMP)

    assert tuple(
        {"cursor": row.cursor, "timestamp": row.timestamp, "content": row.content}
        for row in actual
    ) == tuple(reference)


def test_uses_fixed_fallback_and_increments_from_base_cursor():
    entries = parse_legacy_history(
        b"Unstamped first chunk.\n\n[2026-03-25/27] Unstamped date header.",
        fallback_timestamp=FALLBACK_TIMESTAMP,
        base_cursor=40,
    )

    assert entries == (
        LegacyHistoryEntry(41, FALLBACK_TIMESTAMP, "Unstamped first chunk."),
        LegacyHistoryEntry(
            42,
            FALLBACK_TIMESTAMP,
            "[2026-03-25/27] Unstamped date header.",
        ),
    )


def test_results_are_detached_immutable_records_in_an_immutable_collection():
    entries = parse_legacy_history(
        b"[2026-04-01 10:00] Content.",
        fallback_timestamp=FALLBACK_TIMESTAMP,
    )

    assert isinstance(entries, tuple)
    with pytest.raises(FrozenInstanceError):
        entries[0].content = "changed"


def test_rejects_input_larger_than_64_mib_before_decoding():
    assert MAX_LEGACY_HISTORY_BYTES == 64 * 1024 * 1024
    captured = b"x" * (MAX_LEGACY_HISTORY_BYTES + 1)

    with pytest.raises(ValueError, match="64 MiB"):
        parse_legacy_history(captured, fallback_timestamp=FALLBACK_TIMESTAMP)


def test_allows_empty_input_at_max_cursor_without_incrementing():
    assert parse_legacy_history(
        b" \r\n ",
        fallback_timestamp=FALLBACK_TIMESTAMP,
        base_cursor=MAX_CURSOR,
    ) == ()


def test_rejects_cursor_increment_overflow():
    with pytest.raises(OverflowError):
        parse_legacy_history(
            b"one entry",
            fallback_timestamp=FALLBACK_TIMESTAMP,
            base_cursor=MAX_CURSOR,
        )


@pytest.mark.parametrize("invalid", ["text", bytearray(b"bytes"), memoryview(b"bytes")])
def test_rejects_non_bytes_input(invalid):
    with pytest.raises(TypeError, match="bytes"):
        parse_legacy_history(invalid, fallback_timestamp=FALLBACK_TIMESTAMP)


@pytest.mark.parametrize("invalid", [None, 1, b"timestamp"])
def test_rejects_non_string_fallback_timestamp(invalid):
    with pytest.raises(TypeError, match="fallback_timestamp"):
        parse_legacy_history(b"content", fallback_timestamp=invalid)


@pytest.mark.parametrize("invalid", [-1, MAX_CURSOR + 1, 1.5, True])
def test_rejects_invalid_base_cursor(invalid):
    with pytest.raises((TypeError, ValueError), match="base_cursor"):
        parse_legacy_history(
            b"content",
            fallback_timestamp=FALLBACK_TIMESTAMP,
            base_cursor=invalid,
        )
