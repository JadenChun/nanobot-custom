"""Pure parsing and header patching for legacy receipt-bound session bytes.

This module accepts an already validated :class:`MemoryReceipt` and raw session
bytes. It performs no file access or runtime session integration.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable as _Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

if TYPE_CHECKING:
    from nanobot.agent.memory_recovery import MemoryReceipt

_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
_MAX_RECORDS = 250_000
_MAX_DEPTH = 128
_MAX_DECODED_NODES = 1_000_000
_MAX_COUNTER = (1 << 63) - 1
_CURSOR_FIELDS = ("last_archived", "last_consolidated")


class LegacyReceiptSourceError(ValueError):
    """The source bytes do not provide unambiguous receipt evidence."""

    _MESSAGE = "legacy receipt source is invalid"

    def __init__(self) -> None:
        super().__init__(self._MESSAGE)


@dataclass(frozen=True, slots=True)
class LegacyReceiptSourceResult:
    """Detached result of validating and optionally advancing a legacy source."""

    status: Literal["advanced", "already_advanced"]
    patched_bytes: bytes | None
    source_digest: str
    chunk_digest: str
    generation: int
    revision: int
    cursor_before: int
    cursor_after: int
    message_count: int


@dataclass(frozen=True, slots=True)
class _Record:
    data: dict[str, Any]
    start: int
    end: int
    ending: bytes


class _InvalidJSONError(ValueError):
    """Private signal for JSON structures outside the accepted format."""


def _invalid() -> NoReturn:
    raise LegacyReceiptSourceError()


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidJSONError
        result[key] = value
    return result


def _reject_constant(_value: str) -> NoReturn:
    raise _InvalidJSONError


def _validate_tree(value: object, node_budget: int) -> int:
    # The sole caller supplies JSON decoder output: object_pairs_hook receives
    # string keys at every depth, including dictionaries nested in lists.
    pending: list[tuple[object, int]] = [(value, 1)]
    decoded_nodes = 0
    while pending:
        current, depth = pending.pop()
        decoded_nodes += 1
        if depth > _MAX_DEPTH or decoded_nodes + len(pending) > node_budget:
            raise _InvalidJSONError
        if isinstance(current, dict):
            child_count = 2 * len(cast(dict[str, object], current))
            if decoded_nodes + len(pending) + child_count > node_budget:
                raise _InvalidJSONError
            for key, item in cast(dict[str, object], current).items():
                key.encode("utf-8", "strict")
                pending.append((key, depth + 1))
                pending.append((item, depth + 1))
        elif isinstance(current, list):
            if decoded_nodes + len(pending) + len(cast(list[object], current)) > node_budget:
                raise _InvalidJSONError
            pending.extend((item, depth + 1) for item in cast(list[object], current))
        elif isinstance(current, str):
            current.encode("utf-8", "strict")
        elif isinstance(current, float) and not math.isfinite(current):
            raise _InvalidJSONError
    return decoded_nodes


def _parse_record(line: bytes, node_budget: int) -> tuple[dict[str, Any], int]:
    text = line.decode("utf-8", "strict")
    decoded: object = json.loads(
        text,
        object_pairs_hook=_object_from_pairs,
        parse_constant=_reject_constant,
    )
    if not isinstance(decoded, dict):
        raise _InvalidJSONError
    decoded_nodes = _validate_tree(cast(object, decoded), node_budget)
    return cast(dict[str, Any], decoded), decoded_nodes


def _read_records(raw_bytes: bytes) -> tuple[list[_Record], int]:
    if not raw_bytes or len(raw_bytes) > _MAX_SOURCE_BYTES:
        raise _InvalidJSONError

    # Validate the byte stream once, including content in otherwise blank rows.
    raw_bytes.decode("utf-8", "strict")
    records: list[_Record] = []
    message_count = 0
    position = 0
    physical_records = 0
    decoded_nodes = 0
    header_seen = False
    provider_state_seen = False

    while position < len(raw_bytes):
        physical_records += 1
        if physical_records > _MAX_RECORDS:
            raise _InvalidJSONError

        line_start = position
        newline_index = raw_bytes.find(b"\n", position)
        if newline_index < 0:
            line_end = len(raw_bytes)
            content_end = line_end
            ending = b""
        else:
            line_end = newline_index + 1
            content_end = newline_index
            if content_end > line_start and raw_bytes[content_end - 1] == 0x0D:
                content_end -= 1
                ending = b"\r\n"
            else:
                ending = b"\n"

        line = raw_bytes[line_start:content_end]
        position = line_end
        if not line.strip(b" \t\r"):
            continue

        data, record_nodes = _parse_record(line, _MAX_DECODED_NODES - decoded_nodes)
        decoded_nodes += record_nodes
        record_type = data.get("_type")
        has_type = "_type" in data
        if record_type == "metadata":
            if header_seen or records:
                raise _InvalidJSONError
            header_seen = True
            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                raise _InvalidJSONError
            if "source_schema_version" in metadata or "source_schema_version" in data:
                raise _InvalidJSONError
        elif record_type == "provider_state":
            if not header_seen or provider_state_seen:
                raise _InvalidJSONError
            if not isinstance(data.get("state"), dict):
                raise _InvalidJSONError
            provider_state_seen = True
        elif has_type:
            # `_type` is reserved for structural session rows.
            raise _InvalidJSONError
        else:
            if not header_seen:
                raise _InvalidJSONError
            message_count += 1

        records.append(_Record(data, line_start, line_end, ending))

    if not header_seen:
        raise _InvalidJSONError
    return records, message_count


def _counter(header: dict[str, Any], field: str, default: int = 0) -> int:
    if field not in header:
        return default
    value = header[field]
    if type(value) is not int or not 0 <= value <= _MAX_COUNTER:
        raise _InvalidJSONError
    return value


def _receipt_evidence(
    trusted_session_key: str,
    receipt: MemoryReceipt,
) -> tuple[int, int, int, str]:
    if type(trusted_session_key) is not str or not trusted_session_key:
        raise _InvalidJSONError
    if type(receipt.session_key) is not str or receipt.session_key != trusted_session_key:
        raise _InvalidJSONError

    start = receipt.start_offset
    end = receipt.end_offset
    generation = receipt.session_generation
    if (
        type(start) is not int
        or type(end) is not int
        or type(generation) is not int
        or not 0 <= start <= _MAX_COUNTER
        or not start < end <= _MAX_COUNTER
        or not 0 <= generation <= _MAX_COUNTER
    ):
        raise _InvalidJSONError

    digest = receipt.chunk_digest
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise _InvalidJSONError
    return start, end, generation, digest


def _message_rows(records: list[_Record]) -> list[dict[str, Any]]:
    return [record.data for record in records if "_type" not in record.data]


def _chunk_digest(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def capture_legacy_receipt_chunk_digest() -> _Callable[[list[dict[str, object]]], str]:
    """Capture the current encoder callback without changing later import bindings."""
    return _chunk_digest


def patch_legacy_receipt_source(
    raw_bytes: bytes,
    trusted_session_key: str,
    receipt: MemoryReceipt,
) -> LegacyReceiptSourceResult:
    """Validate receipt evidence and return an immutable header-only patch.

    Legacy headers may omit ``revision`` and ``generation``; both then mean
    zero. If an advancing patch has no revision, it establishes ``revision: 1``.
    A header without either cursor alias means cursor zero; the first advance
    establishes the legacy ``last_consolidated`` field. An advancing keyless
    legacy header receives the trusted key so a retry can validate the patched
    current-format header. Existing cursor aliases are all advanced together.
    An already-advanced cursor returns no patch.
    """
    try:
        if not isinstance(cast(object, raw_bytes), bytes):
            raise _InvalidJSONError
        raw_bytes = bytes(raw_bytes)
        start, end, receipt_generation, chunk_digest = _receipt_evidence(
            trusted_session_key,
            receipt,
        )
        records, message_count = _read_records(raw_bytes)
        header_record = next(record for record in records if record.data.get("_type") == "metadata")
        header = header_record.data

        modern_header_fields = {"generation", "revision", "incarnation", "last_archived"}
        if "key" in header:
            if type(header["key"]) is not str or header["key"] != trusted_session_key:
                raise _InvalidJSONError
        elif modern_header_fields.intersection(header):
            raise _InvalidJSONError
        generation = _counter(header, "generation")
        revision = _counter(header, "revision")
        if "incarnation" in header and (
            type(header["incarnation"]) is not str or not header["incarnation"]
        ):
            raise _InvalidJSONError
        if generation != receipt_generation:
            raise _InvalidJSONError

        cursor_values = [_counter(header, name) for name in _CURSOR_FIELDS if name in header]
        if len(cursor_values) == 2 and cursor_values[0] != cursor_values[1]:
            raise _InvalidJSONError
        cursor = cursor_values[0] if cursor_values else 0
        if cursor > message_count:
            raise _InvalidJSONError
        if end > message_count:
            raise _InvalidJSONError

        messages = _message_rows(records)
        if _chunk_digest(messages[start:end]) != chunk_digest:
            raise _InvalidJSONError

        source_digest = hashlib.sha256(raw_bytes).hexdigest()
        if cursor >= end:
            return LegacyReceiptSourceResult(
                status="already_advanced",
                patched_bytes=None,
                source_digest=source_digest,
                chunk_digest=chunk_digest,
                generation=generation,
                revision=revision,
                cursor_before=cursor,
                cursor_after=cursor,
                message_count=message_count,
            )
        if cursor != start or revision == _MAX_COUNTER:
            raise _InvalidJSONError

        patched_header = dict(header)
        if "key" not in patched_header:
            patched_header["key"] = trusted_session_key
        cursor_fields = [name for name in _CURSOR_FIELDS if name in patched_header]
        if not cursor_fields:
            cursor_fields = ["last_consolidated"]
        for name in cursor_fields:
            patched_header[name] = end
        patched_header["revision"] = revision + 1

        serialized_header = json.dumps(
            patched_header,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        patched_bytes = (
            raw_bytes[: header_record.start]
            + serialized_header
            + header_record.ending
            + raw_bytes[header_record.end :]
        )
        if len(patched_bytes) > _MAX_OUTPUT_BYTES:
            raise _InvalidJSONError

        return LegacyReceiptSourceResult(
            status="advanced",
            patched_bytes=patched_bytes,
            source_digest=source_digest,
            chunk_digest=chunk_digest,
            generation=generation,
            revision=revision + 1,
            cursor_before=cursor,
            cursor_after=end,
            message_count=message_count,
        )
    except LegacyReceiptSourceError:
        raise
    except (AttributeError, OverflowError, RecursionError, TypeError, UnicodeError, ValueError):
        _invalid()
