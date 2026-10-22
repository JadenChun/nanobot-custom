"""Focused byte-contract tests for legacy receipt session evidence."""

from __future__ import annotations

import hashlib
import json

import pytest

from nanobot.agent.memory_recovery import MemoryReceipt
from nanobot.session.legacy_receipt_source import (
    LegacyReceiptSourceError,
    patch_legacy_receipt_source,
)

SESSION_KEY = "telegram:room-42"
MAX_COUNTER = (1 << 63) - 1
PINNED_CHUNK_DIGEST = "fb395bfb636f1cddf83d0c48a72041dff548f9c1d867f55b26fe604f5249a6fc"
PINNED_MESSAGES = [
    {
        "role": "user",
        "content": "Please remember café ☕.",
        "timestamp": "2024-03-10T12:00:00",
    },
    {
        "role": "assistant",
        "content": "I’ll remember it.",
        "metadata": {"locale": "日本語", "items": [1, True, None]},
    },
]


def _chunk_digest(messages: list[dict[str, object]]) -> str:
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _receipt(
    *,
    start: int = 0,
    end: int = 1,
    generation: int = 0,
    session_key: str | None = SESSION_KEY,
    chunk_digest: str | None = None,
) -> MemoryReceipt:
    return MemoryReceipt(
        version=1,
        chunk_digest=chunk_digest or _chunk_digest([{"role": "user", "content": "hello"}]),
        history_marker_id="marker-1",
        history_entry="remember this",
        memory_output="",
        base_memory_digest="0" * 64,
        output_memory_digest=hashlib.sha256(b"").hexdigest(),
        reference_only=False,
        created_at="2024-03-10T12:00:00Z",
        session_key=session_key,
        start_offset=start,
        end_offset=end,
        session_generation=generation,
    )


def _json_line(value: object, newline: bytes = b"\n") -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + newline


def _document(
    header: dict[str, object],
    rows: list[dict[str, object]],
    *,
    newline: bytes = b"\n",
    final_newline: bool = True,
) -> bytes:
    result = _json_line(header, newline)
    for index, row in enumerate(rows):
        row_newline = newline if final_newline or index < len(rows) - 1 else b""
        result += _json_line(row, row_newline)
    return result


def _assert_invalid(raw: bytes, receipt: MemoryReceipt | None = None, key: str = SESSION_KEY) -> None:
    with pytest.raises(LegacyReceiptSourceError) as error:
        patch_legacy_receipt_source(raw, key, receipt or _receipt())
    assert str(error.value) == "legacy receipt source is invalid"
    if raw:
        assert raw.decode("utf-8", errors="replace") not in repr(error.value)


def _pinned_legacy_source() -> tuple[bytes, bytes]:
    header = (
        b'{"_type":"metadata","created_at":"2024-03-10T12:00:00",'
        b'"metadata":{"legacy_note":{"nested":["keep",7]}},'
        b'"last_consolidated":0}\r\n'
    )
    first_message = (
        '{"role":"user","content":"Please remember café ☕.",'
        '"timestamp":"2024-03-10T12:00:00"}'
    ).encode("utf-8") + b"\r\n"
    provider_state = (
        '{"_type":"provider_state","state":{"opaque":"keep \\"exactly\\" ☂"}}'
    ).encode("utf-8") + b"\r\n"
    second_message = (
        '{"role":"assistant","content":"I’ll remember it.",'
        '"metadata":{"locale":"日本語","items":[1,true,null]}}'
    ).encode("utf-8")
    return header, header + first_message + provider_state + second_message


def test_pinned_historical_receipt_advances_legacy_staging_header() -> None:
    header, source = _pinned_legacy_source()
    assert _chunk_digest(PINNED_MESSAGES) == PINNED_CHUNK_DIGEST
    result = patch_legacy_receipt_source(
        source,
        SESSION_KEY,
        _receipt(start=0, end=2, generation=0, chunk_digest=PINNED_CHUNK_DIGEST),
    )

    assert result.status == "advanced"
    assert result.source_digest == hashlib.sha256(source).hexdigest()
    assert result.chunk_digest == PINNED_CHUNK_DIGEST
    assert result.generation == 0
    assert result.revision == 1
    assert (result.cursor_before, result.cursor_after, result.message_count) == (0, 2, 2)
    assert result.patched_bytes is not None
    patched_header, suffix = result.patched_bytes.split(b"\r\n", maxsplit=1)
    patched_header_data = json.loads(patched_header)
    assert patched_header_data["revision"] == 1
    assert patched_header_data["last_consolidated"] == 2
    assert patched_header_data["metadata"] == {"legacy_note": {"nested": ["keep", 7]}}
    assert suffix == source[len(header) :]


def test_pinned_legacy_patch_is_idempotent_when_retried() -> None:
    _header, source = _pinned_legacy_source()
    receipt = _receipt(start=0, end=2, generation=0, chunk_digest=PINNED_CHUNK_DIGEST)

    first = patch_legacy_receipt_source(source, SESSION_KEY, receipt)

    assert first.status == "advanced"
    assert first.patched_bytes is not None
    patched_header = json.loads(first.patched_bytes.split(b"\r\n", maxsplit=1)[0])
    assert patched_header["key"] == SESSION_KEY
    retry = patch_legacy_receipt_source(first.patched_bytes, SESSION_KEY, receipt)
    assert retry.status == "already_advanced"
    assert retry.patched_bytes is None
    assert retry.revision == first.revision == 1
    assert retry.cursor_before == retry.cursor_after == 2


def test_current_header_updates_both_cursor_aliases_and_only_header_bytes() -> None:
    messages = [
        {"role": "user", "content": "before"},
        {"role": "assistant", "content": "chunk one"},
        {"role": "user", "content": "chunk two"},
    ]
    header = {
        "_type": "metadata",
        "key": SESSION_KEY,
        "generation": 4,
        "revision": 12,
        "incarnation": "legacy-incarnation",
        "last_archived": 1,
        "last_consolidated": 1,
        "metadata": {"unknown": {"preserve": [1, "☂"]}},
    }
    raw = _document(header, messages, newline=b"\r\n", final_newline=False)
    original_suffix = raw.split(b"\r\n", maxsplit=1)[1]

    result = patch_legacy_receipt_source(
        raw,
        SESSION_KEY,
        _receipt(start=1, end=3, generation=4, chunk_digest=_chunk_digest(messages[1:])),
    )

    assert result.status == "advanced"
    assert result.revision == 13
    assert result.patched_bytes is not None
    first_line, suffix = result.patched_bytes.split(b"\r\n", maxsplit=1)
    updated = json.loads(first_line)
    assert updated["last_archived"] == updated["last_consolidated"] == 3
    assert updated["revision"] == 13
    assert updated["incarnation"] == "legacy-incarnation"
    assert updated["metadata"] == header["metadata"]
    assert suffix == original_suffix
    assert not result.patched_bytes.endswith(b"\n")


def test_missing_cursor_alias_is_established_as_legacy_last_consolidated() -> None:
    messages = [{"role": "user", "content": "hello"}]
    raw = _document({"_type": "metadata", "metadata": {}}, messages)

    result = patch_legacy_receipt_source(
        raw,
        SESSION_KEY,
        _receipt(chunk_digest=_chunk_digest(messages)),
    )

    assert result.patched_bytes is not None
    header = json.loads(result.patched_bytes.splitlines()[0])
    assert header["last_consolidated"] == 1
    assert header["revision"] == 1


def test_already_advanced_receipt_is_idempotent_and_returns_no_patch() -> None:
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "later one"},
        {"role": "user", "content": "later two"},
    ]
    raw = _document(
        {"_type": "metadata", "key": SESSION_KEY, "generation": 2,
         "revision": 9, "last_archived": 3},
        messages,
    )

    result = patch_legacy_receipt_source(
        raw,
        SESSION_KEY,
        _receipt(start=0, end=1, generation=2, chunk_digest=_chunk_digest(messages[:1])),
    )

    assert result.status == "already_advanced"
    assert result.patched_bytes is None
    assert result.cursor_before == result.cursor_after == 3
    assert result.revision == 9


def test_cursor_between_receipt_offsets_is_a_conflict() -> None:
    messages = [{"role": "user", "content": str(index)} for index in range(3)]
    raw = _document(
        {"_type": "metadata", "last_consolidated": 1},
        messages,
    )
    _assert_invalid(raw, _receipt(start=0, end=2, chunk_digest=_chunk_digest(messages[:2])))


@pytest.mark.parametrize(
    ("header_field", "value"),
    [
        ("revision", True),
        ("revision", -1),
        ("revision", "1"),
        ("generation", True),
        ("generation", -1),
        ("last_archived", False),
        ("last_consolidated", -1),
        ("last_consolidated", "0"),
    ],
)
def test_header_counters_require_nonnegative_exact_integers(
    header_field: str, value: object
) -> None:
    messages = [{"role": "user", "content": "hello"}]
    raw = _document(
        {"_type": "metadata", "key": SESSION_KEY, header_field: value},
        messages,
    )
    _assert_invalid(raw, _receipt(chunk_digest=_chunk_digest(messages)))


def test_conflicting_cursor_aliases_and_out_of_range_cursor_are_rejected() -> None:
    messages = [{"role": "user", "content": "hello"}]
    for header in (
        {"_type": "metadata", "key": SESSION_KEY, "last_archived": 0,
         "last_consolidated": 1},
        {"_type": "metadata", "key": SESSION_KEY, "last_archived": 2},
    ):
        _assert_invalid(
            _document(header, messages),
            _receipt(chunk_digest=_chunk_digest(messages)),
        )


def test_receipt_binding_requires_trusted_key_generation_offsets_and_digest() -> None:
    messages = [{"role": "user", "content": "hello"}]
    raw = _document(
        {"_type": "metadata", "key": "other:room", "generation": 3},
        messages,
    )
    receipt = _receipt(start=0, end=1, generation=3, chunk_digest=_chunk_digest(messages))
    _assert_invalid(raw, receipt)
    _assert_invalid(
        _document({"_type": "metadata", "key": SESSION_KEY, "generation": 4}, messages),
        receipt,
    )
    _assert_invalid(
        _document({"_type": "metadata", "key": SESSION_KEY}, messages),
        _receipt(start=0, end=None, chunk_digest=_chunk_digest(messages)),  # type: ignore[arg-type]
    )
    _assert_invalid(
        _document({"_type": "metadata", "key": SESSION_KEY}, messages),
        _receipt(chunk_digest="0" * 64),
    )
    _assert_invalid(
        _document({"_type": "metadata", "key": SESSION_KEY}, messages),
        _receipt(start=0, end=2, chunk_digest=_chunk_digest(messages)),
    )


def test_provider_state_is_opaque_and_does_not_count_as_a_message() -> None:
    message = {"role": "user", "content": "hello"}
    provider = {"_type": "provider_state", "state": {"opaque": [1, "x"]}}
    raw = _document(
        {"_type": "metadata", "key": SESSION_KEY, "last_archived": 0},
        [provider, message],
    )

    result = patch_legacy_receipt_source(
        raw,
        SESSION_KEY,
        _receipt(chunk_digest=_chunk_digest([message])),
    )

    assert result.message_count == 1
    assert result.patched_bytes is not None
    assert result.patched_bytes.splitlines()[1] == _json_line(provider).rstrip(b"\n")


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b" \n\t\n",
        b"{\"_type\":\"metadata\"}\n{\"role\":",
        b"\xff",
        b'{"_type":"metadata","metadata":{"nested":{"x":1,"x":2}}}\n',
        b'{"_type":"metadata","key":"telegram:room-42","revision":NaN}\n',
        b'{"_type":"metadata","key":"telegram:room-42","revision":Infinity}\n',
        b'{"_type":"metadata","key":"telegram:room-42","revision":1e999}\n',
        b'{"_type":"metadata","metadata":{"source_schema_version":1}}\n',
        b'{"_type":"metadata"}\n{"_type":"other"}\n',
        b'{"role":"user","content":"before"}\n{"_type":"metadata"}\n',
        b'{"_type":"metadata"}\n{"_type":"metadata"}\n',
        b'{"_type":"metadata"}\n'
        b'{"_type":"provider_state","state":{}}\n'
        b'{"_type":"provider_state","state":{}}\n',
    ],
)
def test_malformed_ambiguous_or_converted_sources_are_rejected(raw: bytes) -> None:
    _assert_invalid(raw)


def test_duplicate_message_keys_are_rejected_at_nested_depth() -> None:
    raw = (
        b'{"_type":"metadata"}\n'
        b'{"role":"user","content":"x","extra":{"a":{"b":1,"b":2}}}\n'
    )
    _assert_invalid(raw)


def test_receipt_offsets_must_be_exact_integers_and_nonempty() -> None:
    message = {"role": "user", "content": "hello"}
    raw = _document({"_type": "metadata"}, [message])
    for receipt in (
        _receipt(start=True, end=1, chunk_digest=_chunk_digest([message])),  # type: ignore[arg-type]
        _receipt(start=-1, end=1, chunk_digest=_chunk_digest([message])),
        _receipt(start=0, end=0, chunk_digest=_chunk_digest([])),
        _receipt(start=0, end=1, generation=True, chunk_digest=_chunk_digest([message])),  # type: ignore[arg-type]
    ):
        _assert_invalid(raw, receipt)


def test_input_and_output_limits_are_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    import nanobot.session.legacy_receipt_source as source_module

    message = {"role": "user", "content": "hello"}
    raw = _document({"_type": "metadata"}, [message])
    monkeypatch.setattr(source_module, "_MAX_SOURCE_BYTES", len(raw) - 1)
    _assert_invalid(raw, _receipt(chunk_digest=_chunk_digest([message])))

    monkeypatch.setattr(source_module, "_MAX_SOURCE_BYTES", len(raw))
    monkeypatch.setattr(source_module, "_MAX_OUTPUT_BYTES", len(raw) - 1)
    _assert_invalid(raw, _receipt(chunk_digest=_chunk_digest([message])))


def test_record_and_nesting_budgets_are_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    import nanobot.session.legacy_receipt_source as source_module

    messages = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}]
    raw = _document({"_type": "metadata"}, messages)
    monkeypatch.setattr(source_module, "_MAX_RECORDS", 2)
    _assert_invalid(raw, _receipt(start=0, end=2, chunk_digest=_chunk_digest(messages)))

    nested = {"role": "user", "content": {"one": {"two": "deep"}}}
    nested_raw = _document({"_type": "metadata"}, [nested])
    monkeypatch.setattr(source_module, "_MAX_RECORDS", 10)
    monkeypatch.setattr(source_module, "_MAX_DEPTH", 2)
    _assert_invalid(nested_raw, _receipt(chunk_digest=_chunk_digest([nested])))


def test_decoded_node_budget_is_cumulative_across_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nanobot.session.legacy_receipt_source as source_module

    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    raw = _document({"_type": "metadata"}, messages)
    monkeypatch.setattr(source_module, "_MAX_DECODED_NODES", 10)
    _assert_invalid(raw, _receipt(start=0, end=2, chunk_digest=_chunk_digest(messages)))


def test_decoded_node_budget_rejects_wide_shallow_arrays_before_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nanobot.session.legacy_receipt_source as source_module

    message = {"role": "user", "content": list(range(100))}
    raw = _document({"_type": "metadata"}, [message])
    monkeypatch.setattr(source_module, "_MAX_DECODED_NODES", 20)
    _assert_invalid(raw, _receipt(chunk_digest=_chunk_digest([message])))


def test_legacy_missing_key_is_allowed_only_when_modern_header_fields_are_absent() -> None:
    messages = [{"role": "user", "content": "hello"}]
    for field, value in (
        ("generation", 0),
        ("revision", 0),
        ("incarnation", "legacy-incarnation"),
        ("last_archived", 0),
    ):
        header = {"_type": "metadata", "last_consolidated": 0, field: value}
        _assert_invalid(
            _document(header, messages),
            _receipt(chunk_digest=_chunk_digest(messages)),
        )


def test_receipt_counter_boundaries_and_revision_overflow() -> None:
    messages = [{"role": "user", "content": "hello"}]
    digest = _chunk_digest(messages)
    maximum_header = {
        "_type": "metadata",
        "key": SESSION_KEY,
        "generation": MAX_COUNTER,
        "revision": MAX_COUNTER,
        "last_consolidated": 1,
    }
    current = patch_legacy_receipt_source(
        _document(maximum_header, messages),
        SESSION_KEY,
        _receipt(start=0, end=1, generation=MAX_COUNTER, chunk_digest=digest),
    )
    assert current.status == "already_advanced"
    assert current.generation == current.revision == MAX_COUNTER

    at_revision_limit = _document(
        {**maximum_header, "last_consolidated": 0},
        messages,
    )
    _assert_invalid(
        at_revision_limit,
        _receipt(start=0, end=1, generation=MAX_COUNTER, chunk_digest=digest),
    )

    for field in ("generation", "revision", "last_archived", "last_consolidated"):
        header = {
            "_type": "metadata",
            "key": SESSION_KEY,
            "generation": 0,
            "revision": 0,
            "last_consolidated": 0,
            field: MAX_COUNTER + 1,
        }
        _assert_invalid(
            _document(header, messages),
            _receipt(chunk_digest=digest),
        )

    _assert_invalid(
        _document({"_type": "metadata", "key": SESSION_KEY}, messages),
        _receipt(generation=MAX_COUNTER + 1, chunk_digest=digest),
    )
