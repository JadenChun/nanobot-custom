"""Seekable, rebuildable read index and bounded tail reader for WebUI transcripts.

The index never owns conversation data or read authority. Every entry points to
an exact byte range in the existing active file or immutable rotated segments,
and every managed read verifies the producing ``SourceState`` origin before
returning any rows, cursors, or prompt summaries.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from loguru import logger

from kohaku.session.source_store import SourceState

INDEX_VERSION = 1
INDEX_PAGE_SIZE = 64
PREINDEX_TAIL_INITIAL_BYTES = 4096
PREINDEX_TAIL_MAX_BYTES = 256 * 1024
_ACTIVE_CHUNK_ID = "active"
_CURSOR_VERSION = 2

WindowStatus = Literal[
    "ok",
    "empty",
    "stale_cursor",
    "not_found",
    "oversized_tail",
    "unavailable",
]
WindowCoverage = Literal["indexed", "tail_only", "empty"]


@dataclass(frozen=True, slots=True)
class IndexReadMetrics:
    """Deterministic I/O and decode counters for bounded-read verification."""

    bytes_read: int = 0
    records_decoded: int = 0
    index_pages_read: int = 0
    manifest_entries_decoded: int = 0
    chunks_opened: int = 0


@dataclass(frozen=True, slots=True)
class IndexedPrompt:
    """One visible user prompt inside a turn."""

    prompt_id: str
    turn_ordinal: int
    record_index: int
    start_byte: int
    end_byte: int
    label: str
    created_at_ms: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "turn_ordinal": self.turn_ordinal,
            "record_index": self.record_index,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "label": self.label,
            "created_at_ms": self.created_at_ms,
        }

    @classmethod
    def from_dict(cls, raw: object) -> IndexedPrompt | None:
        if not isinstance(raw, dict):
            return None
        data = cast(dict[str, Any], raw)
        prompt_id = data.get("prompt_id")
        turn_ordinal = data.get("turn_ordinal")
        record_index = data.get("record_index")
        start_byte = data.get("start_byte")
        end_byte = data.get("end_byte")
        label = data.get("label")
        created_at_ms = data.get("created_at_ms")
        if (
            not isinstance(prompt_id, str)
            or not prompt_id
            or type(turn_ordinal) is not int
            or turn_ordinal < 0
            or type(record_index) is not int
            or record_index < 0
            or type(start_byte) is not int
            or start_byte < 0
            or type(end_byte) is not int
            or end_byte <= start_byte
            or not isinstance(label, str)
            or (created_at_ms is not None and type(created_at_ms) is not int)
        ):
            return None
        return cls(
            prompt_id=prompt_id,
            turn_ordinal=turn_ordinal,
            record_index=record_index,
            start_byte=start_byte,
            end_byte=end_byte,
            label=label,
            created_at_ms=created_at_ms,
        )


@dataclass(frozen=True, slots=True)
class IndexedTurn:
    """Physical byte range and structural metadata for one display turn."""

    ordinal: int
    chunk_id: str
    file_name: str
    start_byte: int
    end_byte: int
    ui_messages: int
    has_turn_end: bool
    incomplete: bool
    prompt_offset_before: int
    prompts: tuple[IndexedPrompt, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "chunk_id": self.chunk_id,
            "file_name": self.file_name,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "ui_messages": self.ui_messages,
            "has_turn_end": self.has_turn_end,
            "incomplete": self.incomplete,
            "prompt_offset_before": self.prompt_offset_before,
            "prompts": [prompt.to_dict() for prompt in self.prompts],
        }

    @classmethod
    def from_dict(cls, raw: object) -> IndexedTurn | None:
        if not isinstance(raw, dict):
            return None
        data = cast(dict[str, Any], raw)
        ordinal = data.get("ordinal")
        chunk_id = data.get("chunk_id")
        file_name = data.get("file_name")
        start_byte = data.get("start_byte")
        end_byte = data.get("end_byte")
        ui_messages = data.get("ui_messages")
        has_turn_end = data.get("has_turn_end")
        incomplete = data.get("incomplete")
        prompt_offset_before = data.get("prompt_offset_before")
        raw_prompts = data.get("prompts")
        if (
            type(ordinal) is not int
            or ordinal < 0
            or not isinstance(chunk_id, str)
            or not chunk_id
            or not isinstance(file_name, str)
            or not file_name
            or "/" in file_name
            or "\\" in file_name
            or type(start_byte) is not int
            or start_byte < 0
            or type(end_byte) is not int
            or end_byte <= start_byte
            or type(ui_messages) is not int
            or ui_messages < 0
            or type(has_turn_end) is not bool
            or type(incomplete) is not bool
            or type(prompt_offset_before) is not int
            or prompt_offset_before < 0
            or not isinstance(raw_prompts, list)
        ):
            return None
        prompts: list[IndexedPrompt] = []
        for item in cast(list[object], raw_prompts):
            parsed = IndexedPrompt.from_dict(item)
            if parsed is None:
                return None
            prompts.append(parsed)
        return cls(
            ordinal=ordinal,
            chunk_id=chunk_id,
            file_name=file_name,
            start_byte=start_byte,
            end_byte=end_byte,
            ui_messages=ui_messages,
            has_turn_end=has_turn_end,
            incomplete=incomplete,
            prompt_offset_before=prompt_offset_before,
            prompts=tuple(prompts),
        )


@dataclass(frozen=True, slots=True)
class TranscriptIndexHead:
    """Small committed head binding the paged directory to exact physical files."""

    version: int
    session_key: str
    origin: str | None
    active_file: str | None
    active_ino: int | None
    active_dev: int | None
    active_covered_bytes: int
    manifest_ino: int | None
    manifest_size: int | None
    manifest_mtime_ns: int | None
    segment_count: int
    last_segment_id: str | None
    last_segment_bytes: int
    total_turns: int
    total_prompts: int
    page_size: int
    page_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "session_key": self.session_key,
            "origin": self.origin,
            "active_file": self.active_file,
            "active_ino": self.active_ino,
            "active_dev": self.active_dev,
            "active_covered_bytes": self.active_covered_bytes,
            "manifest_ino": self.manifest_ino,
            "manifest_size": self.manifest_size,
            "manifest_mtime_ns": self.manifest_mtime_ns,
            "segment_count": self.segment_count,
            "last_segment_id": self.last_segment_id,
            "last_segment_bytes": self.last_segment_bytes,
            "total_turns": self.total_turns,
            "total_prompts": self.total_prompts,
            "page_size": self.page_size,
            "page_count": self.page_count,
        }

    @classmethod
    def from_dict(cls, raw: object) -> TranscriptIndexHead | None:
        if not isinstance(raw, dict):
            return None
        data = cast(dict[str, Any], raw)
        if data.get("version") != INDEX_VERSION:
            return None
        session_key = data.get("session_key")
        origin = data.get("origin")
        active_file = data.get("active_file")
        active_ino = data.get("active_ino")
        active_dev = data.get("active_dev")
        active_covered_bytes = data.get("active_covered_bytes")
        manifest_ino = data.get("manifest_ino")
        manifest_size = data.get("manifest_size")
        manifest_mtime_ns = data.get("manifest_mtime_ns")
        segment_count = data.get("segment_count")
        last_segment_id = data.get("last_segment_id")
        last_segment_bytes = data.get("last_segment_bytes")
        total_turns = data.get("total_turns")
        total_prompts = data.get("total_prompts")
        page_size = data.get("page_size")
        page_count = data.get("page_count")
        if (
            not isinstance(session_key, str)
            or not session_key
            or (origin is not None and (not isinstance(origin, str) or not origin))
            or (
                active_file is not None
                and (not isinstance(active_file, str) or "/" in active_file or "\\" in active_file)
            )
            or (active_ino is not None and type(active_ino) is not int)
            or (active_dev is not None and type(active_dev) is not int)
            or type(active_covered_bytes) is not int
            or active_covered_bytes < 0
            or (manifest_ino is not None and type(manifest_ino) is not int)
            or (manifest_size is not None and type(manifest_size) is not int)
            or (manifest_mtime_ns is not None and type(manifest_mtime_ns) is not int)
            or type(segment_count) is not int
            or segment_count < 0
            or (last_segment_id is not None and not isinstance(last_segment_id, str))
            or type(last_segment_bytes) is not int
            or last_segment_bytes < 0
            or type(total_turns) is not int
            or total_turns < 0
            or type(total_prompts) is not int
            or total_prompts < 0
            or type(page_size) is not int
            or page_size <= 0
            or type(page_count) is not int
            or page_count < 0
        ):
            return None
        return cls(
            version=INDEX_VERSION,
            session_key=session_key,
            origin=origin,
            active_file=active_file,
            active_ino=active_ino,
            active_dev=active_dev,
            active_covered_bytes=active_covered_bytes,
            manifest_ino=manifest_ino,
            manifest_size=manifest_size,
            manifest_mtime_ns=manifest_mtime_ns,
            segment_count=segment_count,
            last_segment_id=last_segment_id,
            last_segment_bytes=last_segment_bytes,
            total_turns=total_turns,
            total_prompts=total_prompts,
            page_size=page_size,
            page_count=page_count,
        )


@dataclass(frozen=True, slots=True)
class HistoryWindowResult:
    """Result of one bounded latest, before, after, or around history read."""

    status: WindowStatus
    coverage: WindowCoverage
    lines: list[dict[str, Any]]
    first_turn_ordinal: int | None
    last_turn_ordinal: int | None
    total_turn_count: int | None
    total_prompt_count: int | None
    user_message_offset: int | None
    before_cursor: str | None
    after_cursor: str | None
    has_more_before: bool
    has_more_after: bool
    target_prompt_id: str | None
    origin: str | None
    metrics: IndexReadMetrics


@dataclass(frozen=True, slots=True)
class PromptSummaryItem:
    """One prompt entry in a bounded prompt-summary range."""

    prompt_id: str
    turn_ordinal: int
    prompt_ordinal: int
    label: str
    created_at_ms: int | None
    locator: str


@dataclass(frozen=True, slots=True)
class PromptSummaryRangeResult:
    """Result of one bounded prompt-summary directory read."""

    status: WindowStatus
    entries: list[PromptSummaryItem]
    before_cursor: str | None
    has_more_before: bool
    total_prompt_count: int | None
    total_turn_count: int | None
    origin: str | None
    metrics: IndexReadMetrics


def _webui_dir() -> Path:
    from kohaku.webui import transcript as _transcript

    return _transcript.get_webui_dir()


def transcript_index_dir(session_key: str) -> Path:
    from kohaku.webui.transcript import transcript_stem

    return _webui_dir() / f"{transcript_stem(session_key)}.index"


def _head_path(session_key: str) -> Path:
    return transcript_index_dir(session_key) / "head.json"


def _page_path(session_key: str, page_index: int) -> Path:
    return transcript_index_dir(session_key) / "pages" / f"{page_index:06d}.json"


def delete_transcript_index(session_key: str) -> bool:
    """Remove all rebuildable index files for *session_key*."""
    directory = transcript_index_dir(session_key)
    if not directory.is_dir():
        return False
    try:
        shutil.rmtree(directory)
        return True
    except OSError as exc:
        logger.warning("Failed to delete transcript index {}: {}", directory, exc)
        return False


def encode_window_cursor(*, origin: str | None, ordinal: int) -> str:
    payload = {"v": _CURSOR_VERSION, "origin": origin or "", "ordinal": ordinal}
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def encode_tail_window_cursor(*, origin: str | None, tail_turns: int) -> str:
    """Origin-bound cursor for a pre-index ``tail_only`` window before index build."""
    payload = {"v": _CURSOR_VERSION, "origin": origin or "", "tail_turns": tail_turns}
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_window_cursor(
    value: str | None,
    *,
    expected_origin: str | None,
    total_turns: int | None = None,
) -> tuple[int | None, bool]:
    """Return ``(ordinal, valid)``. ``(None, True)`` means no cursor supplied."""
    if not value:
        return None, True
    try:
        padded = value + "=" * (-len(value) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None, False
    if not isinstance(data, dict):
        return None, False
    cursor = cast(dict[str, Any], data)
    if cursor.get("v") != _CURSOR_VERSION:
        return None, False
    origin = cursor.get("origin")
    if not isinstance(origin, str) or origin != (expected_origin or ""):
        return None, False
    ordinal = cursor.get("ordinal")
    if type(ordinal) is int and ordinal >= 0:
        return ordinal, True
    tail_turns = cursor.get("tail_turns")
    if (
        type(tail_turns) is int
        and tail_turns > 0
        and total_turns is not None
        and total_turns >= 0
    ):
        return max(0, total_turns - tail_turns), True
    return None, False


def encode_prompt_locator(
    *,
    origin: str | None,
    page: int,
    ordinal: int,
    prompt_id: str,
) -> str:
    payload = {
        "v": _CURSOR_VERSION,
        "origin": origin or "",
        "page": page,
        "ordinal": ordinal,
        "prompt_id": prompt_id,
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_prompt_locator(
    value: str | None,
    *,
    expected_origin: str | None,
) -> tuple[int, int, str] | None:
    if not value:
        return None
    try:
        padded = value + "=" * (-len(value) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    loc = cast(dict[str, Any], data)
    if loc.get("v") != _CURSOR_VERSION:
        return None
    origin = loc.get("origin")
    page = loc.get("page")
    ordinal = loc.get("ordinal")
    prompt_id = loc.get("prompt_id")
    if (
        not isinstance(origin, str)
        or origin != (expected_origin or "")
        or type(page) is not int
        or page < 0
        or type(ordinal) is not int
        or ordinal < 0
        or not isinstance(prompt_id, str)
        or not prompt_id
    ):
        return None
    return page, ordinal, prompt_id


def _atomic_write_json(path: Path, data: object) -> None:
    from kohaku.webui.transcript import transcript_fsync_directory

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        transcript_fsync_directory(path.parent)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _stat_file(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError:
        return None


def _manifest_stat(session_key: str) -> tuple[int | None, int | None, int | None]:
    from kohaku.webui.transcript import transcript_manifest_path

    info = _stat_file(transcript_manifest_path(session_key))
    if info is None:
        return None, None, None
    return info.st_ino, info.st_size, info.st_mtime_ns


def _expected_origin(expected_source: SourceState | None) -> str | None:
    if expected_source is None:
        return None
    from kohaku.webui.transcript import transcript_record_origin

    return transcript_record_origin(expected_source)


def _prompt_id_for_record(record: dict[str, Any], *, turn_ordinal: int, record_index: int) -> str:
    from kohaku.webui.transcript import TRANSCRIPT_PROMPT_SOURCE_EVENT_KEY

    committed = record.get(TRANSCRIPT_PROMPT_SOURCE_EVENT_KEY)
    if isinstance(committed, str) and committed:
        return committed
    return f"legacy:{turn_ordinal}:{record_index}"


def _build_turn_entry(
    *,
    ordinal: int,
    chunk_id: str,
    file_name: str,
    start_byte: int,
    end_byte: int,
    records: list[tuple[int, int, dict[str, Any]]],
    prompt_offset_before: int,
) -> IndexedTurn:
    from kohaku.webui.transcript import (
        WEBUI_TRANSCRIPT_INCOMPLETE_KEY,
        replay_transcript_to_ui_messages,
        transcript_is_user_row,
        transcript_turn_label,
    )

    raw_rows = [row for _, _, row in records]
    prompts: list[IndexedPrompt] = []
    for record_index, (rec_start, rec_end, row) in enumerate(records):
        if not transcript_is_user_row(row):
            continue
        created_raw = row.get("created_at_ms")
        created_at_ms = created_raw if type(created_raw) is int else None
        prompts.append(IndexedPrompt(
            prompt_id=_prompt_id_for_record(row, turn_ordinal=ordinal, record_index=record_index),
            turn_ordinal=ordinal,
            record_index=record_index,
            start_byte=rec_start,
            end_byte=rec_end,
            label=transcript_turn_label(row, ordinal),
            created_at_ms=created_at_ms,
        ))
    has_turn_end = bool(raw_rows and raw_rows[-1].get("event") == "turn_end")
    incomplete = any(row.get(WEBUI_TRANSCRIPT_INCOMPLETE_KEY) is True for row in raw_rows)
    return IndexedTurn(
        ordinal=ordinal,
        chunk_id=chunk_id,
        file_name=file_name,
        start_byte=start_byte,
        end_byte=end_byte,
        ui_messages=len(replay_transcript_to_ui_messages(raw_rows)),
        has_turn_end=has_turn_end,
        incomplete=incomplete,
        prompt_offset_before=prompt_offset_before,
        prompts=tuple(prompts),
    )


def _scan_file_records(
    path: Path,
    *,
    start_offset: int = 0,
    expected_origin: str | None,
) -> tuple[list[tuple[int, int, dict[str, Any]]], int, int]:
    """Scan JSONL records from *start_offset* with exact byte spans.

    Returns ``(records, end_offset, decoded_count)``.
    """
    from kohaku.webui.transcript import TRANSCRIPT_RECORD_ORIGIN_KEY

    records: list[tuple[int, int, dict[str, Any]]] = []
    decoded = 0
    offset = start_offset
    with open(path, "rb") as handle:
        if start_offset > 0:
            handle.seek(start_offset)
        while True:
            line_start = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                offset = line_start
                break
            line_end = handle.tell()
            offset = line_end
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(obj, dict):
                continue
            row = cast(dict[str, Any], obj)
            decoded += 1
            if expected_origin is not None and row.get(TRANSCRIPT_RECORD_ORIGIN_KEY) != expected_origin:
                continue
            records.append((line_start, line_end, row))
    return records, offset, decoded


def _turns_from_scanned_records(
    records: list[tuple[int, int, dict[str, Any]]],
    *,
    chunk_id: str,
    file_name: str,
    start_ordinal: int,
    start_prompt_offset: int,
) -> list[IndexedTurn]:
    turns: list[IndexedTurn] = []
    current: list[tuple[int, int, dict[str, Any]]] = []
    ordinal = start_ordinal
    prompt_offset = start_prompt_offset
    for item in records:
        current.append(item)
        if item[2].get("event") == "turn_end":
            turn = _build_turn_entry(
                ordinal=ordinal,
                chunk_id=chunk_id,
                file_name=file_name,
                start_byte=current[0][0],
                end_byte=current[-1][1],
                records=current,
                prompt_offset_before=prompt_offset,
            )
            turns.append(turn)
            ordinal += 1
            prompt_offset += len(turn.prompts)
            current = []
    if current:
        turn = _build_turn_entry(
            ordinal=ordinal,
            chunk_id=chunk_id,
            file_name=file_name,
            start_byte=current[0][0],
            end_byte=current[-1][1],
            records=current,
            prompt_offset_before=prompt_offset,
        )
        turns.append(turn)
    return turns


def _read_head(session_key: str) -> TranscriptIndexHead | None:
    path = _head_path(session_key)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    head = TranscriptIndexHead.from_dict(raw)
    if head is None or head.session_key != session_key:
        return None
    return head


def _read_page(session_key: str, page_index: int) -> list[IndexedTurn] | None:
    path = _page_path(session_key, page_index)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    data = cast(dict[str, Any], raw)
    if data.get("version") != INDEX_VERSION or data.get("page") != page_index:
        return None
    raw_turns = data.get("turns")
    if not isinstance(raw_turns, list):
        return None
    turns: list[IndexedTurn] = []
    for item in cast(list[object], raw_turns):
        parsed = IndexedTurn.from_dict(item)
        if parsed is None:
            return None
        turns.append(parsed)
    return turns


def _write_page(session_key: str, page_index: int, turns: list[IndexedTurn]) -> None:
    _atomic_write_json(
        _page_path(session_key, page_index),
        {
            "version": INDEX_VERSION,
            "page": page_index,
            "turns": [turn.to_dict() for turn in turns],
        },
    )


def _write_head(session_key: str, head: TranscriptIndexHead) -> None:
    _atomic_write_json(_head_path(session_key), head.to_dict())


def _chunk_path(session_key: str, chunk_id: str, file_name: str) -> Path:
    from kohaku.webui.transcript import webui_transcript_segments_dir

    if chunk_id == _ACTIVE_CHUNK_ID:
        return _webui_dir() / file_name
    return webui_transcript_segments_dir(session_key) / file_name


def _active_file_without_segment_scan(session_key: str) -> tuple[Path, bool, int]:
    """Resolve the active file path without stat-ing historical segments.

    Returns ``(active_path, has_segments, manifest_entries_decoded)``.
    """
    from kohaku.webui.transcript import (
        LEGACY_TRANSCRIPT_SEGMENT_MANIFEST_VERSION,
        TRANSCRIPT_SEGMENT_MANIFEST_VERSION,
        transcript_file_name,
        transcript_manifest_path,
        transcript_migrate_legacy,
        transcript_recover_rotation,
        transcript_rotation_intent_path,
        webui_transcript_path,
        webui_transcript_segments_dir,
    )

    transcript_migrate_legacy(session_key)
    if transcript_rotation_intent_path(session_key).exists():
        transcript_recover_rotation(session_key)
    manifest_path = transcript_manifest_path(session_key)
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = None
        if isinstance(data, dict):
            manifest = cast(dict[str, Any], data)
            version = manifest.get("version")
            raw_segments = manifest.get("segments")
            has_segments = isinstance(raw_segments, list) and len(cast(list[object], raw_segments)) > 0
            if version == TRANSCRIPT_SEGMENT_MANIFEST_VERSION:
                active_name = transcript_file_name(manifest.get("active"))
                if active_name is not None:
                    return _webui_dir() / active_name, has_segments, 0
                return webui_transcript_path(session_key), has_segments, 0
            if version == LEGACY_TRANSCRIPT_SEGMENT_MANIFEST_VERSION:
                return webui_transcript_path(session_key), has_segments, 0
    seg_dir = webui_transcript_segments_dir(session_key)
    has_segments = False
    if seg_dir.is_dir():
        with suppress(OSError):
            has_segments = any(
                child.name.endswith(".jsonl") for child in seg_dir.iterdir() if child.is_file()
            )
    return webui_transcript_path(session_key), has_segments, 0


def _read_byte_range_records(
    path: Path,
    *,
    start_byte: int,
    end_byte: int,
    expected_origin: str | None,
) -> tuple[list[dict[str, Any]] | None, int, int]:
    """Read one exact committed byte range, verifying line boundaries and origin.

    Returns ``(records_or_none_if_invalid, bytes_read, records_decoded)``.
    """
    from kohaku.webui.transcript import TRANSCRIPT_RECORD_ORIGIN_KEY

    info = _stat_file(path)
    if info is None or start_byte < 0 or end_byte <= start_byte or end_byte > info.st_size:
        return None, 0, 0
    read_start = start_byte - 1 if start_byte > 0 else 0
    read_len = end_byte - read_start
    try:
        with open(path, "rb") as handle:
            handle.seek(read_start)
            blob = handle.read(read_len)
    except OSError:
        return None, 0, 0
    if len(blob) != read_len:
        return None, len(blob), 0
    if start_byte > 0:
        if blob[:1] != b"\n":
            return None, len(blob), 0
        payload = blob[1:]
    else:
        payload = blob
    if not payload.endswith(b"\n"):
        return None, len(blob), 0

    records: list[dict[str, Any]] = []
    decoded = 0
    for raw_line in payload.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, len(blob), decoded
        if not isinstance(obj, dict):
            return None, len(blob), decoded
        row = cast(dict[str, Any], obj)
        decoded += 1
        if expected_origin is not None and row.get(TRANSCRIPT_RECORD_ORIGIN_KEY) != expected_origin:
            return None, len(blob), decoded
        records.append(row)
    return records, len(blob), decoded


def rebuild_transcript_index(
    session_key: str,
    *,
    expected_source: SourceState | None = None,
) -> TranscriptIndexHead | None:
    """Rebuild the paged read index from committed durable display files."""
    if expected_source is not None and expected_source.session_key != session_key:
        return None
    from kohaku.webui.transcript import (
        transcript_migrate_legacy,
        transcript_read_segment_manifest_state,
        transcript_segment_path,
        webui_transcript_path,
    )

    transcript_migrate_legacy(session_key)
    origin = _expected_origin(expected_source)
    entries, active_name = transcript_read_segment_manifest_state(session_key, recover_rotation=True)
    active_path = _webui_dir() / active_name if active_name else webui_transcript_path(session_key)

    all_turns: list[IndexedTurn] = []
    ordinal = 0
    prompt_offset = 0

    for entry in entries:
        segment_id = str(entry["id"])
        seg_path = transcript_segment_path(session_key, segment_id)
        if not seg_path.is_file():
            continue
        scanned, _, _ = _scan_file_records(seg_path, start_offset=0, expected_origin=origin)
        seg_turns = _turns_from_scanned_records(
            scanned,
            chunk_id=segment_id,
            file_name=seg_path.name,
            start_ordinal=ordinal,
            start_prompt_offset=prompt_offset,
        )
        all_turns.extend(seg_turns)
        ordinal += len(seg_turns)
        prompt_offset += sum(len(turn.prompts) for turn in seg_turns)

    active_info = _stat_file(active_path)
    active_covered_bytes = 0
    if active_info is not None and active_path.is_file():
        scanned_active, end_offset, _ = _scan_file_records(
            active_path, start_offset=0, expected_origin=origin,
        )
        active_covered_bytes = end_offset
        active_turns = _turns_from_scanned_records(
            scanned_active,
            chunk_id=_ACTIVE_CHUNK_ID,
            file_name=active_path.name,
            start_ordinal=ordinal,
            start_prompt_offset=prompt_offset,
        )
        all_turns.extend(active_turns)
        ordinal += len(active_turns)
        prompt_offset += sum(len(turn.prompts) for turn in active_turns)

    pages_dir = transcript_index_dir(session_key) / "pages"
    if pages_dir.is_dir():
        with suppress(OSError):
            shutil.rmtree(pages_dir)

    page_count = 0
    for idx in range(0, len(all_turns), INDEX_PAGE_SIZE):
        _write_page(session_key, page_count, all_turns[idx:idx + INDEX_PAGE_SIZE])
        page_count += 1

    manifest_ino, manifest_size, manifest_mtime_ns = _manifest_stat(session_key)
    last_segment_id = str(entries[-1]["id"]) if entries else None
    last_segment_bytes = int(entries[-1]["bytes"]) if entries else 0
    head = TranscriptIndexHead(
        version=INDEX_VERSION,
        session_key=session_key,
        origin=origin,
        active_file=active_path.name if active_info is not None else None,
        active_ino=active_info.st_ino if active_info is not None else None,
        active_dev=active_info.st_dev if active_info is not None else None,
        active_covered_bytes=active_covered_bytes,
        manifest_ino=manifest_ino,
        manifest_size=manifest_size,
        manifest_mtime_ns=manifest_mtime_ns,
        segment_count=len(entries),
        last_segment_id=last_segment_id,
        last_segment_bytes=last_segment_bytes,
        total_turns=ordinal,
        total_prompts=prompt_offset,
        page_size=INDEX_PAGE_SIZE,
        page_count=page_count,
    )
    _write_head(session_key, head)
    return head


def _append_records_to_head(
    session_key: str,
    head: TranscriptIndexHead,
    *,
    active_path: Path,
    active_info: os.stat_result,
    new_records: list[tuple[int, int, dict[str, Any]]],
    new_covered_bytes: int,
) -> TranscriptIndexHead | None:
    """Incrementally advance the last directory page and head by *new_records*."""
    from kohaku.webui.transcript import TRANSCRIPT_RECORD_ORIGIN_KEY

    if head.origin is not None:
        filtered = [
            item for item in new_records if item[2].get(TRANSCRIPT_RECORD_ORIGIN_KEY) == head.origin
        ]
    else:
        filtered = new_records

    if not filtered:
        updated = TranscriptIndexHead(
            version=head.version,
            session_key=head.session_key,
            origin=head.origin,
            active_file=active_path.name,
            active_ino=active_info.st_ino,
            active_dev=active_info.st_dev,
            active_covered_bytes=new_covered_bytes,
            manifest_ino=head.manifest_ino,
            manifest_size=head.manifest_size,
            manifest_mtime_ns=head.manifest_mtime_ns,
            segment_count=head.segment_count,
            last_segment_id=head.last_segment_id,
            last_segment_bytes=head.last_segment_bytes,
            total_turns=head.total_turns,
            total_prompts=head.total_prompts,
            page_size=head.page_size,
            page_count=head.page_count,
        )
        _write_head(session_key, updated)
        return updated

    page_size = head.page_size
    last_page_idx = head.page_count - 1 if head.page_count > 0 else 0
    last_page = _read_page(session_key, last_page_idx) if head.page_count > 0 else []
    if last_page is None:
        return None

    open_turn_records: list[tuple[int, int, dict[str, Any]]] = []
    if (
        last_page
        and not last_page[-1].has_turn_end
        and last_page[-1].chunk_id == _ACTIVE_CHUNK_ID
        and last_page[-1].file_name == active_path.name
    ):
        open_turn = last_page.pop()
        prior_records, _, _ = _scan_file_records(
            active_path,
            start_offset=open_turn.start_byte,
            expected_origin=head.origin,
        )
        # Keep only records up to open_turn.end_byte, then append filtered.
        open_turn_records = [
            item for item in prior_records if item[1] <= open_turn.end_byte
        ]
        next_ordinal = open_turn.ordinal
        next_prompt_offset = open_turn.prompt_offset_before
    else:
        next_ordinal = head.total_turns
        next_prompt_offset = head.total_prompts

    combined = [*open_turn_records, *filtered]
    new_turns = _turns_from_scanned_records(
        combined,
        chunk_id=_ACTIVE_CHUNK_ID,
        file_name=active_path.name,
        start_ordinal=next_ordinal,
        start_prompt_offset=next_prompt_offset,
    )

    page_turns = list(last_page)
    page_idx = last_page_idx
    pages_to_write: list[tuple[int, list[IndexedTurn]]] = []
    for turn in new_turns:
        if len(page_turns) >= page_size:
            pages_to_write.append((page_idx, page_turns))
            page_idx += 1
            page_turns = []
        page_turns.append(turn)
    if page_turns or head.page_count > 0:
        pages_to_write.append((page_idx, page_turns))

    for idx, turns_batch in pages_to_write:
        _write_page(session_key, idx, turns_batch)

    new_page_count = page_idx + 1 if (pages_to_write and pages_to_write[-1][1]) else head.page_count
    total_turns = new_turns[-1].ordinal + 1 if new_turns else head.total_turns
    total_prompts = (
        new_turns[-1].prompt_offset_before + len(new_turns[-1].prompts)
        if new_turns else head.total_prompts
    )
    updated_head = TranscriptIndexHead(
        version=INDEX_VERSION,
        session_key=session_key,
        origin=head.origin,
        active_file=active_path.name,
        active_ino=active_info.st_ino,
        active_dev=active_info.st_dev,
        active_covered_bytes=new_covered_bytes,
        manifest_ino=head.manifest_ino,
        manifest_size=head.manifest_size,
        manifest_mtime_ns=head.manifest_mtime_ns,
        segment_count=head.segment_count,
        last_segment_id=head.last_segment_id,
        last_segment_bytes=head.last_segment_bytes,
        total_turns=total_turns,
        total_prompts=total_prompts,
        page_size=page_size,
        page_count=new_page_count,
    )
    _write_head(session_key, updated_head)
    return updated_head


def on_transcript_record_appended(
    session_key: str,
    *,
    record: dict[str, Any],
    active_path: Path,
    start_byte: int,
    end_byte: int,
    active_info: os.stat_result,
) -> None:
    """Incrementally update the index after one durable active-file append."""
    from kohaku.webui.transcript import TRANSCRIPT_RECORD_ORIGIN_KEY

    raw_origin = record.get(TRANSCRIPT_RECORD_ORIGIN_KEY)
    record_origin = raw_origin if isinstance(raw_origin, str) and raw_origin else None
    head = _read_head(session_key)
    manifest_ino, manifest_size, manifest_mtime_ns = _manifest_stat(session_key)

    if head is None:
        # Only initialize O(1) directly when appending the very first line at byte 0
        # with no pre-existing rotated segments. Otherwise stay pre-index until read/rebuild.
        if start_byte != 0 or manifest_ino is not None:
            return
        empty_head = TranscriptIndexHead(
            version=INDEX_VERSION,
            session_key=session_key,
            origin=record_origin,
            active_file=active_path.name,
            active_ino=active_info.st_ino,
            active_dev=active_info.st_dev,
            active_covered_bytes=0,
            manifest_ino=None,
            manifest_size=None,
            manifest_mtime_ns=None,
            segment_count=0,
            last_segment_id=None,
            last_segment_bytes=0,
            total_turns=0,
            total_prompts=0,
            page_size=INDEX_PAGE_SIZE,
            page_count=0,
        )
        _append_records_to_head(
            session_key,
            empty_head,
            active_path=active_path,
            active_info=active_info,
            new_records=[(start_byte, end_byte, record)],
            new_covered_bytes=end_byte,
        )
        return

    if (
        head.origin != record_origin
        or head.active_file != active_path.name
        or head.active_ino != active_info.st_ino
        or head.active_dev != active_info.st_dev
        or head.active_covered_bytes != start_byte
        or head.manifest_ino != manifest_ino
        or head.manifest_size != manifest_size
        or head.manifest_mtime_ns != manifest_mtime_ns
    ):
        return

    _append_records_to_head(
        session_key,
        head,
        active_path=active_path,
        active_info=active_info,
        new_records=[(start_byte, end_byte, record)],
        new_covered_bytes=end_byte,
    )


def on_transcript_rotated(
    session_key: str,
    *,
    new_segment_ids: list[str],
    next_active_path: Path,
) -> None:
    """Update only the rotated tail of an existing index in O(active_file)."""
    head = _read_head(session_key)
    if head is None:
        return
    from kohaku.webui.transcript import transcript_segment_path

    # Find the earliest page containing any turn from the previous active chunk.
    # Older segments and their directory pages are immutable and untouched.
    first_active_page = head.page_count
    preserved_prefix: list[IndexedTurn] = []
    for page_idx in range(head.page_count - 1, -1, -1):
        turns = _read_page(session_key, page_idx)
        if turns is None:
            return
        if all(turn.chunk_id == _ACTIVE_CHUNK_ID for turn in turns):
            first_active_page = page_idx
            continue
        if any(turn.chunk_id == _ACTIVE_CHUNK_ID for turn in turns):
            first_active_page = page_idx
            preserved_prefix = [t for t in turns if t.chunk_id != _ACTIVE_CHUNK_ID]
        break

    if preserved_prefix:
        next_ordinal = preserved_prefix[-1].ordinal + 1
        next_prompt_offset = (
            preserved_prefix[-1].prompt_offset_before + len(preserved_prefix[-1].prompts)
        )
    elif first_active_page > 0:
        prev_page = _read_page(session_key, first_active_page - 1)
        if prev_page is None or not prev_page:
            return
        next_ordinal = prev_page[-1].ordinal + 1
        next_prompt_offset = prev_page[-1].prompt_offset_before + len(prev_page[-1].prompts)
    else:
        next_ordinal = 0
        next_prompt_offset = 0

    new_turns: list[IndexedTurn] = []
    last_seg_id = head.last_segment_id
    last_seg_bytes = head.last_segment_bytes
    for seg_id in new_segment_ids:
        seg_path = transcript_segment_path(session_key, seg_id)
        seg_info = _stat_file(seg_path)
        if seg_info is None:
            return
        scanned, _, _ = _scan_file_records(seg_path, start_offset=0, expected_origin=head.origin)
        seg_turns = _turns_from_scanned_records(
            scanned,
            chunk_id=seg_id,
            file_name=seg_path.name,
            start_ordinal=next_ordinal,
            start_prompt_offset=next_prompt_offset,
        )
        new_turns.extend(seg_turns)
        next_ordinal += len(seg_turns)
        next_prompt_offset += sum(len(turn.prompts) for turn in seg_turns)
        last_seg_id = seg_id
        last_seg_bytes = seg_info.st_size

    active_info = _stat_file(next_active_path)
    active_covered = 0
    if active_info is not None and next_active_path.is_file():
        scanned_active, end_offset, _ = _scan_file_records(
            next_active_path, start_offset=0, expected_origin=head.origin,
        )
        active_covered = end_offset
        active_turns = _turns_from_scanned_records(
            scanned_active,
            chunk_id=_ACTIVE_CHUNK_ID,
            file_name=next_active_path.name,
            start_ordinal=next_ordinal,
            start_prompt_offset=next_prompt_offset,
        )
        new_turns.extend(active_turns)
        next_ordinal += len(active_turns)
        next_prompt_offset += sum(len(turn.prompts) for turn in active_turns)

    page_turns = list(preserved_prefix)
    page_idx = first_active_page
    written_last_page = first_active_page - 1
    for turn in new_turns:
        if len(page_turns) >= head.page_size:
            _write_page(session_key, page_idx, page_turns)
            written_last_page = page_idx
            page_idx += 1
            page_turns = []
        page_turns.append(turn)
    if page_turns:
        _write_page(session_key, page_idx, page_turns)
        written_last_page = page_idx

    new_page_count = written_last_page + 1
    for stale_idx in range(new_page_count, head.page_count):
        with suppress(OSError):
            _page_path(session_key, stale_idx).unlink()

    manifest_ino, manifest_size, manifest_mtime_ns = _manifest_stat(session_key)
    _write_head(
        session_key,
        TranscriptIndexHead(
            version=INDEX_VERSION,
            session_key=session_key,
            origin=head.origin,
            active_file=next_active_path.name if active_info is not None else None,
            active_ino=active_info.st_ino if active_info is not None else None,
            active_dev=active_info.st_dev if active_info is not None else None,
            active_covered_bytes=active_covered,
            manifest_ino=manifest_ino,
            manifest_size=manifest_size,
            manifest_mtime_ns=manifest_mtime_ns,
            segment_count=head.segment_count + len(new_segment_ids),
            last_segment_id=last_seg_id,
            last_segment_bytes=last_seg_bytes,
            total_turns=next_ordinal,
            total_prompts=next_prompt_offset,
            page_size=head.page_size,
            page_count=new_page_count,
        ),
    )


def on_transcript_rewritten(
    session_key: str,
    *,
    expected_source: SourceState | None = None,
) -> None:
    """Publish a fresh index after a full display write (for example, a fork)."""
    rebuild_transcript_index(session_key, expected_source=expected_source)


def _validate_or_catchup_head(
    session_key: str,
    *,
    expected_origin: str | None,
) -> tuple[TranscriptIndexHead | None, int]:
    """Return a current, verified index head (catching up append deltas in O(delta)).

    Returns ``(head_or_none_if_missing_or_stale, delta_records_decoded)``.
    """
    from kohaku.webui.transcript import transcript_rotation_intent_path

    if transcript_rotation_intent_path(session_key).exists():
        return None, 0
    head = _read_head(session_key)
    if head is None or head.origin != expected_origin:
        return None, 0

    manifest_ino, manifest_size, manifest_mtime_ns = _manifest_stat(session_key)
    if (
        head.manifest_ino != manifest_ino
        or head.manifest_size != manifest_size
        or head.manifest_mtime_ns != manifest_mtime_ns
    ):
        return None, 0

    if head.active_file is None:
        active_path, _, _ = _active_file_without_segment_scan(session_key)
        if active_path.is_file():
            return None, 0
        return head, 0

    active_path = _webui_dir() / head.active_file
    active_info = _stat_file(active_path)
    if (
        active_info is None
        or active_info.st_ino != head.active_ino
        or active_info.st_dev != head.active_dev
        or active_info.st_size < head.active_covered_bytes
    ):
        return None, 0

    if active_info.st_size == head.active_covered_bytes:
        return head, 0

    # Active file grew via append after an interrupted index update. Verify the
    # covered boundary ends on a newline before catching up ONLY the delta.
    if head.active_covered_bytes > 0:
        try:
            with open(active_path, "rb") as handle:
                handle.seek(head.active_covered_bytes - 1)
                if handle.read(1) != b"\n":
                    return None, 0
        except OSError:
            return None, 0

    delta_records, end_offset, decoded = _scan_file_records(
        active_path,
        start_offset=head.active_covered_bytes,
        expected_origin=expected_origin,
    )
    caught_up = _append_records_to_head(
        session_key,
        head,
        active_path=active_path,
        active_info=active_info,
        new_records=delta_records,
        new_covered_bytes=end_offset,
    )
    return caught_up, decoded


def _load_turn_range(
    session_key: str,
    head: TranscriptIndexHead,
    *,
    turns: list[IndexedTurn],
    expected_origin: str | None,
    pages_read: int,
    extra_records_decoded: int = 0,
    target_prompt_id: str | None = None,
) -> HistoryWindowResult | None:
    """Materialize selected turns by reading ONLY their exact byte ranges."""
    from kohaku.webui.transcript import transcript_records_with_replay_identity

    if not turns:
        return HistoryWindowResult(
            status="empty",
            coverage="indexed",
            lines=[],
            first_turn_ordinal=None,
            last_turn_ordinal=None,
            total_turn_count=head.total_turns,
            total_prompt_count=head.total_prompts,
            user_message_offset=0,
            before_cursor=None,
            after_cursor=None,
            has_more_before=False,
            has_more_after=False,
            target_prompt_id=target_prompt_id,
            origin=head.origin,
            metrics=IndexReadMetrics(
                index_pages_read=pages_read,
                records_decoded=extra_records_decoded,
            ),
        )

    lines: list[dict[str, Any]] = []
    bytes_read = 0
    records_decoded = extra_records_decoded
    chunks_opened = 0

    # Group contiguous turns within the same chunk file into one bounded range read.
    groups: list[list[IndexedTurn]] = []
    for turn in turns:
        if (
            groups
            and groups[-1][-1].chunk_id == turn.chunk_id
            and groups[-1][-1].file_name == turn.file_name
            and groups[-1][-1].end_byte == turn.start_byte
        ):
            groups[-1].append(turn)
        else:
            groups.append([turn])

    for group in groups:
        chunk_path = _chunk_path(session_key, group[0].chunk_id, group[0].file_name)
        range_start = group[0].start_byte
        range_end = group[-1].end_byte
        raw_records, n_bytes, n_decoded = _read_byte_range_records(
            chunk_path,
            start_byte=range_start,
            end_byte=range_end,
            expected_origin=expected_origin,
        )
        chunks_opened += 1
        bytes_read += n_bytes
        records_decoded += n_decoded
        if raw_records is None:
            return None
        # Split the coalesced chunk records back onto their exact structural turns.
        turn_idx = 0
        current_turn_rows: list[dict[str, Any]] = []
        for row in raw_records:
            current_turn_rows.append(row)
            if row.get("event") == "turn_end":
                if turn_idx >= len(group):
                    return None
                lines.extend(transcript_records_with_replay_identity(
                    current_turn_rows, turn_ordinal=group[turn_idx].ordinal,
                ))
                turn_idx += 1
                current_turn_rows = []
        if current_turn_rows:
            if turn_idx >= len(group):
                return None
            lines.extend(transcript_records_with_replay_identity(
                current_turn_rows, turn_ordinal=group[turn_idx].ordinal,
            ))
            turn_idx += 1
        if turn_idx != len(group):
            return None

    first_turn = turns[0]
    last_turn = turns[-1]
    has_more_before = first_turn.ordinal > 0
    has_more_after = last_turn.ordinal + 1 < head.total_turns
    return HistoryWindowResult(
        status="ok",
        coverage="indexed",
        lines=lines,
        first_turn_ordinal=first_turn.ordinal,
        last_turn_ordinal=last_turn.ordinal,
        total_turn_count=head.total_turns,
        total_prompt_count=head.total_prompts,
        user_message_offset=first_turn.prompt_offset_before,
        before_cursor=(
            encode_window_cursor(origin=head.origin, ordinal=first_turn.ordinal)
            if has_more_before else None
        ),
        after_cursor=(
            encode_window_cursor(origin=head.origin, ordinal=last_turn.ordinal + 1)
            if has_more_after else None
        ),
        has_more_before=has_more_before,
        has_more_after=has_more_after,
        target_prompt_id=target_prompt_id,
        origin=head.origin,
        metrics=IndexReadMetrics(
            bytes_read=bytes_read,
            records_decoded=records_decoded,
            index_pages_read=pages_read,
            manifest_entries_decoded=0,
            chunks_opened=chunks_opened,
        ),
    )


def _select_indexed_turns_backward(
    session_key: str,
    head: TranscriptIndexHead,
    *,
    upper_ordinal: int,
    limit: int,
) -> tuple[list[IndexedTurn] | None, int]:
    """Select turns ending before *upper_ordinal* up to *limit* UI messages."""
    selected: list[IndexedTurn] = []
    message_budget = 0
    pages_read = 0
    if upper_ordinal <= 0 or head.page_count == 0:
        return [], 0

    end_ordinal = min(upper_ordinal, head.total_turns) - 1
    page_idx = end_ordinal // head.page_size
    while page_idx >= 0 and message_budget < limit:
        page_turns = _read_page(session_key, page_idx)
        pages_read += 1
        if page_turns is None:
            return None, pages_read
        for turn in reversed(page_turns):
            if turn.ordinal > end_ordinal:
                continue
            selected.append(turn)
            message_budget += max(1, turn.ui_messages)
            if message_budget >= limit:
                break
        page_idx -= 1
    selected.reverse()
    return selected, pages_read


def _select_indexed_turns_forward(
    session_key: str,
    head: TranscriptIndexHead,
    *,
    start_ordinal: int,
    limit: int,
) -> tuple[list[IndexedTurn] | None, int]:
    """Select turns starting at *start_ordinal* up to *limit* UI messages."""
    selected: list[IndexedTurn] = []
    message_budget = 0
    pages_read = 0
    if start_ordinal >= head.total_turns or head.page_count == 0:
        return [], 0

    page_idx = start_ordinal // head.page_size
    while page_idx < head.page_count and message_budget < limit:
        page_turns = _read_page(session_key, page_idx)
        pages_read += 1
        if page_turns is None:
            return None, pages_read
        for turn in page_turns:
            if turn.ordinal < start_ordinal:
                continue
            selected.append(turn)
            message_budget += max(1, turn.ui_messages)
            if message_budget >= limit:
                break
        page_idx += 1
    return selected, pages_read


def read_preindex_latest_tail(
    session_key: str,
    *,
    limit: int,
    expected_source: SourceState | None = None,
    max_tail_bytes: int = PREINDEX_TAIL_MAX_BYTES,
) -> HistoryWindowResult:
    """Bounded latest-window fallback for pre-index histories.

    Reads backwards from EOF of the active file in bounded chunks up to
    *max_tail_bytes* without scanning older segments or rewriting files. An
    oversized recent record/turn returns ``status="oversized_tail"``, never an
    empty chat.
    """
    from kohaku.webui.transcript import (
        TRANSCRIPT_RECORD_ORIGIN_KEY,
        replay_transcript_to_ui_messages,
        transcript_records_with_replay_identity,
    )

    expected_origin = _expected_origin(expected_source)
    active_path, has_segments, manifest_decoded = _active_file_without_segment_scan(session_key)
    info = _stat_file(active_path)
    if info is None or info.st_size == 0:
        if has_segments:
            return HistoryWindowResult(
                status="unavailable",
                coverage="tail_only",
                lines=[],
                first_turn_ordinal=None,
                last_turn_ordinal=None,
                total_turn_count=None,
                total_prompt_count=None,
                user_message_offset=None,
                before_cursor=None,
                after_cursor=None,
                has_more_before=True,
                has_more_after=False,
                target_prompt_id=None,
                origin=expected_origin,
                metrics=IndexReadMetrics(manifest_entries_decoded=manifest_decoded),
            )
        return HistoryWindowResult(
            status="empty",
            coverage="empty",
            lines=[],
            first_turn_ordinal=None,
            last_turn_ordinal=None,
            total_turn_count=0,
            total_prompt_count=0,
            user_message_offset=0,
            before_cursor=None,
            after_cursor=None,
            has_more_before=False,
            has_more_after=False,
            target_prompt_id=None,
            origin=expected_origin,
            metrics=IndexReadMetrics(manifest_entries_decoded=manifest_decoded),
        )

    file_size = info.st_size
    window_bytes = min(file_size, max(PREINDEX_TAIL_INITIAL_BYTES, min(max_tail_bytes, 4096)))
    total_bytes_read = 0
    total_decoded = 0
    chunks_opened = 0

    while True:
        read_start = max(0, file_size - window_bytes)
        read_len = file_size - read_start
        try:
            with open(active_path, "rb") as handle:
                handle.seek(read_start)
                blob = handle.read(read_len)
        except OSError:
            return HistoryWindowResult(
                status="unavailable",
                coverage="tail_only",
                lines=[],
                first_turn_ordinal=None,
                last_turn_ordinal=None,
                total_turn_count=None,
                total_prompt_count=None,
                user_message_offset=None,
                before_cursor=None,
                after_cursor=None,
                has_more_before=False,
                has_more_after=False,
                target_prompt_id=None,
                origin=expected_origin,
                metrics=IndexReadMetrics(
                    bytes_read=total_bytes_read,
                    records_decoded=total_decoded,
                    manifest_entries_decoded=manifest_decoded,
                    chunks_opened=chunks_opened,
                ),
            )
        chunks_opened += 1
        total_bytes_read = len(blob)

        # If we started mid-file, drop the leading partial line up to the first '\n'.
        raw_lines = blob.splitlines()
        if read_start > 0:
            if b"\n" not in blob:
                raw_lines = []
            else:
                raw_lines = raw_lines[1:]

        decoded_rows: list[dict[str, Any]] = []
        parse_count = 0
        for raw_line in raw_lines:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(obj, dict):
                decoded_rows.append(cast(dict[str, Any], obj))
                parse_count += 1
        total_decoded = parse_count

        if not decoded_rows:
            if read_start == 0:
                return HistoryWindowResult(
                    status="empty",
                    coverage="empty",
                    lines=[],
                    first_turn_ordinal=None,
                    last_turn_ordinal=None,
                    total_turn_count=0,
                    total_prompt_count=0,
                    user_message_offset=0,
                    before_cursor=None,
                    after_cursor=None,
                    has_more_before=has_segments,
                    has_more_after=False,
                    target_prompt_id=None,
                    origin=expected_origin,
                    metrics=IndexReadMetrics(
                        bytes_read=total_bytes_read,
                        records_decoded=total_decoded,
                        manifest_entries_decoded=manifest_decoded,
                        chunks_opened=chunks_opened,
                    ),
                )
            if window_bytes >= max_tail_bytes:
                return HistoryWindowResult(
                    status="oversized_tail",
                    coverage="tail_only",
                    lines=[],
                    first_turn_ordinal=None,
                    last_turn_ordinal=None,
                    total_turn_count=None,
                    total_prompt_count=None,
                    user_message_offset=None,
                    before_cursor=None,
                    after_cursor=None,
                    has_more_before=True,
                    has_more_after=False,
                    target_prompt_id=None,
                    origin=expected_origin,
                    metrics=IndexReadMetrics(
                        bytes_read=total_bytes_read,
                        records_decoded=total_decoded,
                        manifest_entries_decoded=manifest_decoded,
                        chunks_opened=chunks_opened,
                    ),
                )
            window_bytes = min(file_size, max_tail_bytes, window_bytes * 4)
            continue

        crossed_origin_boundary = False
        if expected_origin is not None:
            qualified_rows = [
                row for row in decoded_rows if row.get(TRANSCRIPT_RECORD_ORIGIN_KEY) == expected_origin
            ]
            if not qualified_rows:
                # Never adopt rows from a prior lifetime or unbound legacy rows.
                return HistoryWindowResult(
                    status="empty",
                    coverage="tail_only",
                    lines=[],
                    first_turn_ordinal=None,
                    last_turn_ordinal=None,
                    total_turn_count=None,
                    total_prompt_count=None,
                    user_message_offset=None,
                    before_cursor=None,
                    after_cursor=None,
                    has_more_before=False,
                    has_more_after=False,
                    target_prompt_id=None,
                    origin=expected_origin,
                    metrics=IndexReadMetrics(
                        bytes_read=total_bytes_read,
                        records_decoded=total_decoded,
                        manifest_entries_decoded=manifest_decoded,
                        chunks_opened=chunks_opened,
                    ),
                )
            crossed_origin_boundary = decoded_rows[0].get(TRANSCRIPT_RECORD_ORIGIN_KEY) != expected_origin
            rows = qualified_rows
        else:
            rows = decoded_rows

        # Partition rows into structural turns. When read_start > 0 and we have
        # not crossed a prior-lifetime origin boundary, the very first fragment
        # before the first turn_end may be truncated.
        turns: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for row in rows:
            current.append(row)
            if row.get("event") == "turn_end":
                turns.append(current)
                current = []
        if current:
            turns.append(current)

        at_origin_start = read_start == 0 or crossed_origin_boundary
        complete_turns = turns if at_origin_start else turns[1:]
        if not complete_turns:
            if at_origin_start:
                complete_turns = turns
            elif window_bytes >= max_tail_bytes:
                return HistoryWindowResult(
                    status="oversized_tail",
                    coverage="tail_only",
                    lines=[],
                    first_turn_ordinal=None,
                    last_turn_ordinal=None,
                    total_turn_count=None,
                    total_prompt_count=None,
                    user_message_offset=None,
                    before_cursor=None,
                    after_cursor=None,
                    has_more_before=True,
                    has_more_after=False,
                    target_prompt_id=None,
                    origin=expected_origin,
                    metrics=IndexReadMetrics(
                        bytes_read=total_bytes_read,
                        records_decoded=total_decoded,
                        manifest_entries_decoded=manifest_decoded,
                        chunks_opened=chunks_opened,
                    ),
                )
            else:
                window_bytes = min(file_size, max_tail_bytes, window_bytes * 4)
                continue

        selected: list[list[dict[str, Any]]] = []
        selected_messages = 0
        for turn in reversed(complete_turns):
            selected.append(turn)
            selected_messages += len(replay_transcript_to_ui_messages(turn))
            if selected_messages >= limit:
                break
        if selected_messages < limit and not at_origin_start and window_bytes < max_tail_bytes:
            window_bytes = min(file_size, max_tail_bytes, window_bytes * 4)
            continue

        selected.reverse()
        has_more_before = (
            (not crossed_origin_boundary and (has_segments or read_start > 0))
            or len(selected) < len(complete_turns)
        )
        annotated: list[dict[str, Any]] = []
        # When has_more_before is True we only read a tail window whose absolute
        # starting ordinal is unknown: use negative ordinals so their replay IDs
        # never collide with real indexed ordinals (0, 1, ...) when older pages
        # are prepended after the index is built. When has_more_before is False
        # we read from the origin start so idx is the exact canonical ordinal.
        for idx, turn in enumerate(selected):
            annotated.extend(
                transcript_records_with_replay_identity(
                    turn,
                    turn_ordinal=-(len(selected) - idx) if has_more_before else idx,
                ),
            )
        return HistoryWindowResult(
            status="ok",
            coverage="tail_only",
            lines=annotated,
            first_turn_ordinal=None,
            last_turn_ordinal=None,
            total_turn_count=None,
            total_prompt_count=None,
            user_message_offset=None,
            before_cursor=(
                encode_tail_window_cursor(origin=expected_origin, tail_turns=len(selected))
                if has_more_before and selected
                else None
            ),
            after_cursor=None,
            has_more_before=has_more_before,
            has_more_after=False,
            target_prompt_id=None,
            origin=expected_origin,
            metrics=IndexReadMetrics(
                bytes_read=total_bytes_read,
                records_decoded=total_decoded,
                manifest_entries_decoded=manifest_decoded,
                chunks_opened=chunks_opened,
            ),
        )


def read_history_window(
    session_key: str,
    *,
    expected_source: SourceState | None = None,
    limit: int = 40,
    direction: Literal["latest", "before", "after", "around"] = "latest",
    cursor: str | None = None,
    prompt_locator: str | None = None,
    prompt_id: str | None = None,
    allow_rebuild: bool = True,
) -> HistoryWindowResult:
    """Serve one bounded ``latest``, ``before``, ``after``, or ``around`` window."""
    if expected_source is not None and expected_source.session_key != session_key:
        return HistoryWindowResult(
            status="unavailable",
            coverage="empty",
            lines=[],
            first_turn_ordinal=None,
            last_turn_ordinal=None,
            total_turn_count=None,
            total_prompt_count=None,
            user_message_offset=None,
            before_cursor=None,
            after_cursor=None,
            has_more_before=False,
            has_more_after=False,
            target_prompt_id=None,
            origin=None,
            metrics=IndexReadMetrics(),
        )
    expected_origin = _expected_origin(expected_source)
    bounded_limit = max(1, min(1000, int(limit)))

    head, delta_decoded = _validate_or_catchup_head(
        session_key, expected_origin=expected_origin,
    )
    if head is None:
        if direction == "latest" and cursor is None:
            return read_preindex_latest_tail(
                session_key,
                limit=bounded_limit,
                expected_source=expected_source,
            )
        if not allow_rebuild:
            return HistoryWindowResult(
                status="unavailable",
                coverage="empty",
                lines=[],
                first_turn_ordinal=None,
                last_turn_ordinal=None,
                total_turn_count=None,
                total_prompt_count=None,
                user_message_offset=None,
                before_cursor=None,
                after_cursor=None,
                has_more_before=False,
                has_more_after=False,
                target_prompt_id=None,
                origin=expected_origin,
                metrics=IndexReadMetrics(),
            )
        head = rebuild_transcript_index(session_key, expected_source=expected_source)
        delta_decoded = 0
        if head is None:
            return HistoryWindowResult(
                status="unavailable",
                coverage="empty",
                lines=[],
                first_turn_ordinal=None,
                last_turn_ordinal=None,
                total_turn_count=None,
                total_prompt_count=None,
                user_message_offset=None,
                before_cursor=None,
                after_cursor=None,
                has_more_before=False,
                has_more_after=False,
                target_prompt_id=None,
                origin=expected_origin,
                metrics=IndexReadMetrics(),
            )

    def _stale(status: WindowStatus = "stale_cursor") -> HistoryWindowResult:
        return HistoryWindowResult(
            status=status,
            coverage="indexed",
            lines=[],
            first_turn_ordinal=None,
            last_turn_ordinal=None,
            total_turn_count=head.total_turns,
            total_prompt_count=head.total_prompts,
            user_message_offset=None,
            before_cursor=None,
            after_cursor=None,
            has_more_before=False,
            has_more_after=False,
            target_prompt_id=prompt_id,
            origin=head.origin,
            metrics=IndexReadMetrics(records_decoded=delta_decoded),
        )

    if direction == "latest":
        turns, pages_read = _select_indexed_turns_backward(
            session_key, head, upper_ordinal=head.total_turns, limit=bounded_limit,
        )
        if turns is None:
            return _stale("unavailable")
        result = _load_turn_range(
            session_key, head, turns=turns, expected_origin=expected_origin,
            pages_read=pages_read, extra_records_decoded=delta_decoded,
        )
        return result if result is not None else _stale("unavailable")

    if direction == "before":
        ordinal, valid = decode_window_cursor(
            cursor,
            expected_origin=expected_origin,
            total_turns=head.total_turns,
        )
        if not valid or ordinal is None or ordinal > head.total_turns:
            return _stale("stale_cursor")
        turns, pages_read = _select_indexed_turns_backward(
            session_key, head, upper_ordinal=ordinal, limit=bounded_limit,
        )
        if turns is None:
            return _stale("unavailable")
        result = _load_turn_range(
            session_key, head, turns=turns, expected_origin=expected_origin,
            pages_read=pages_read, extra_records_decoded=delta_decoded,
        )
        return result if result is not None else _stale("unavailable")

    if direction == "after":
        ordinal, valid = decode_window_cursor(cursor, expected_origin=expected_origin)
        if not valid or ordinal is None or ordinal > head.total_turns:
            return _stale("stale_cursor")
        turns, pages_read = _select_indexed_turns_forward(
            session_key, head, start_ordinal=ordinal, limit=bounded_limit,
        )
        if turns is None:
            return _stale("unavailable")
        result = _load_turn_range(
            session_key, head, turns=turns, expected_origin=expected_origin,
            pages_read=pages_read, extra_records_decoded=delta_decoded,
        )
        return result if result is not None else _stale("unavailable")

    # direction == "around"
    target_turn: IndexedTurn | None = None
    target_pid: str | None = None
    pages_read = 0
    if prompt_locator is not None:
        decoded_loc = decode_prompt_locator(prompt_locator, expected_origin=expected_origin)
        if decoded_loc is None:
            return _stale("stale_cursor")
        page_idx, target_ordinal, loc_pid = decoded_loc
        if page_idx >= head.page_count or target_ordinal >= head.total_turns:
            return _stale("stale_cursor")
        page_turns = _read_page(session_key, page_idx)
        pages_read += 1
        if page_turns is None:
            return _stale("unavailable")
        candidate = next((t for t in page_turns if t.ordinal == target_ordinal), None)
        if candidate is None or not any(p.prompt_id == loc_pid for p in candidate.prompts):
            return _stale("stale_cursor")
        target_turn = candidate
        target_pid = loc_pid
    elif prompt_id is not None:
        for page_idx in range(head.page_count - 1, -1, -1):
            page_turns = _read_page(session_key, page_idx)
            pages_read += 1
            if page_turns is None:
                return _stale("unavailable")
            for turn in reversed(page_turns):
                if any(p.prompt_id == prompt_id for p in turn.prompts):
                    target_turn = turn
                    target_pid = prompt_id
                    break
            if target_turn is not None:
                break
        if target_turn is None:
            return _stale("not_found")
    else:
        return _stale("stale_cursor")

    # Build a symmetric context window around target_turn without duplicates.
    half_before = max(1, (bounded_limit - max(1, target_turn.ui_messages)) // 2)
    half_after = max(1, bounded_limit - max(1, target_turn.ui_messages) - half_before)
    before_turns, before_pages = _select_indexed_turns_backward(
        session_key, head, upper_ordinal=target_turn.ordinal, limit=half_before,
    )
    after_turns, after_pages = _select_indexed_turns_forward(
        session_key, head, start_ordinal=target_turn.ordinal + 1, limit=half_after,
    )
    pages_read += before_pages + after_pages
    if before_turns is None or after_turns is None:
        return _stale("unavailable")
    window_turns = [*before_turns, target_turn, *after_turns]
    result = _load_turn_range(
        session_key,
        head,
        turns=window_turns,
        expected_origin=expected_origin,
        pages_read=pages_read,
        extra_records_decoded=delta_decoded,
        target_prompt_id=target_pid,
    )
    return result if result is not None else _stale("unavailable")


def read_prompt_summary_range(
    session_key: str,
    *,
    expected_source: SourceState | None = None,
    limit: int = 100,
    before: str | None = None,
    allow_rebuild: bool = True,
) -> PromptSummaryRangeResult:
    """Read a bounded range of visible user prompts without opening JSONL chunks."""
    if expected_source is not None and expected_source.session_key != session_key:
        return PromptSummaryRangeResult(
            status="unavailable",
            entries=[],
            before_cursor=None,
            has_more_before=False,
            total_prompt_count=None,
            total_turn_count=None,
            origin=None,
            metrics=IndexReadMetrics(),
        )
    expected_origin = _expected_origin(expected_source)
    bounded_limit = max(1, min(1000, int(limit)))

    head, delta_decoded = _validate_or_catchup_head(
        session_key, expected_origin=expected_origin,
    )
    if head is None:
        if not allow_rebuild:
            return PromptSummaryRangeResult(
                status="unavailable",
                entries=[],
                before_cursor=None,
                has_more_before=False,
                total_prompt_count=None,
                total_turn_count=None,
                origin=expected_origin,
                metrics=IndexReadMetrics(),
            )
        head = rebuild_transcript_index(session_key, expected_source=expected_source)
        delta_decoded = 0
        if head is None:
            return PromptSummaryRangeResult(
                status="unavailable",
                entries=[],
                before_cursor=None,
                has_more_before=False,
                total_prompt_count=None,
                total_turn_count=None,
                origin=expected_origin,
                metrics=IndexReadMetrics(),
            )

    upper_prompt_ordinal, valid = decode_window_cursor(before, expected_origin=expected_origin)
    if not valid or (upper_prompt_ordinal is not None and upper_prompt_ordinal > head.total_prompts):
        return PromptSummaryRangeResult(
            status="stale_cursor",
            entries=[],
            before_cursor=None,
            has_more_before=False,
            total_prompt_count=head.total_prompts,
            total_turn_count=head.total_turns,
            origin=head.origin,
            metrics=IndexReadMetrics(records_decoded=delta_decoded),
        )

    upper = head.total_prompts if upper_prompt_ordinal is None else upper_prompt_ordinal
    if upper <= 0 or head.page_count == 0:
        return PromptSummaryRangeResult(
            status="empty",
            entries=[],
            before_cursor=None,
            has_more_before=False,
            total_prompt_count=head.total_prompts,
            total_turn_count=head.total_turns,
            origin=head.origin,
            metrics=IndexReadMetrics(records_decoded=delta_decoded),
        )

    entries: list[PromptSummaryItem] = []
    pages_read = 0
    page_cache: dict[int, list[IndexedTurn]] = {}

    def _load_page(idx: int) -> list[IndexedTurn] | None:
        nonlocal pages_read
        if idx in page_cache:
            return page_cache[idx]
        loaded = _read_page(session_key, idx)
        pages_read += 1
        if loaded is None:
            return None
        page_cache[idx] = loaded
        return loaded

    # Binary-search the last directory page whose starting prompt offset < upper.
    if upper >= head.total_prompts:
        start_page = head.page_count - 1
    else:
        lo, hi = 0, head.page_count - 1
        start_page = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            mid_turns = _load_page(mid)
            if mid_turns is None:
                return PromptSummaryRangeResult(
                    status="unavailable",
                    entries=[],
                    before_cursor=None,
                    has_more_before=False,
                    total_prompt_count=head.total_prompts,
                    total_turn_count=head.total_turns,
                    origin=head.origin,
                    metrics=IndexReadMetrics(
                        index_pages_read=pages_read, records_decoded=delta_decoded,
                    ),
                )
            if mid_turns and mid_turns[0].prompt_offset_before < upper:
                start_page = mid
                lo = mid + 1
            else:
                hi = mid - 1

    for page_idx in range(start_page, -1, -1):
        if len(entries) >= bounded_limit:
            break
        page_turns = _load_page(page_idx)
        if page_turns is None:
            return PromptSummaryRangeResult(
                status="unavailable",
                entries=[],
                before_cursor=None,
                has_more_before=False,
                total_prompt_count=head.total_prompts,
                total_turn_count=head.total_turns,
                origin=head.origin,
                metrics=IndexReadMetrics(index_pages_read=pages_read, records_decoded=delta_decoded),
            )
        if page_turns and page_turns[0].prompt_offset_before >= upper:
            continue
        for turn in reversed(page_turns):
            if len(entries) >= bounded_limit:
                break
            if turn.prompt_offset_before >= upper:
                continue
            for local_idx in range(len(turn.prompts) - 1, -1, -1):
                prompt_ordinal = turn.prompt_offset_before + local_idx
                if prompt_ordinal >= upper:
                    continue
                prompt = turn.prompts[local_idx]
                entries.append(PromptSummaryItem(
                    prompt_id=prompt.prompt_id,
                    turn_ordinal=turn.ordinal,
                    prompt_ordinal=prompt_ordinal,
                    label=prompt.label,
                    created_at_ms=prompt.created_at_ms,
                    locator=encode_prompt_locator(
                        origin=head.origin,
                        page=page_idx,
                        ordinal=turn.ordinal,
                        prompt_id=prompt.prompt_id,
                    ),
                ))
                if len(entries) >= bounded_limit:
                    break

    entries.reverse()
    if not entries:
        return PromptSummaryRangeResult(
            status="empty",
            entries=[],
            before_cursor=None,
            has_more_before=False,
            total_prompt_count=head.total_prompts,
            total_turn_count=head.total_turns,
            origin=head.origin,
            metrics=IndexReadMetrics(index_pages_read=pages_read, records_decoded=delta_decoded),
        )
    first_prompt_ordinal = entries[0].prompt_ordinal
    has_more = first_prompt_ordinal > 0
    return PromptSummaryRangeResult(
        status="ok",
        entries=entries,
        before_cursor=(
            encode_window_cursor(origin=head.origin, ordinal=first_prompt_ordinal)
            if has_more else None
        ),
        has_more_before=has_more,
        total_prompt_count=head.total_prompts,
        total_turn_count=head.total_turns,
        origin=head.origin,
        metrics=IndexReadMetrics(index_pages_read=pages_read, records_decoded=delta_decoded),
    )
