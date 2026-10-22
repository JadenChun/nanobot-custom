"""Checked scalar views retain the original recovery refusal and read phases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from nanobot.agent import memory_recovery as codec
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent import memory_recovery_transaction_memory as memory_stage
from nanobot.agent.memory_recovery_manifest import MAX_INT

_FIELDS = ("start_offset", "end_offset", "session_generation", "session_key")


def _receipt(*, omit: str | None = None) -> codec.MemoryReceipt:
    payload: dict[str, object] = {
        "version": 1,
        "chunk_digest": hashlib.sha256(b"chunk").hexdigest(),
        "history_marker_id": "private marker",
        "history_entry": "private history",
        "memory_output": None,
        "base_memory_digest": hashlib.sha256(b"").hexdigest(),
        "output_memory_digest": None,
        "reference_only": True,
        "created_at": "2026-10-08",
        "session_key": "private session 🌙",
        "start_offset": 2,
        "end_offset": 7,
        "session_generation": 3,
    }
    if omit is not None:
        del payload[omit]
    return codec.decode_memory_receipt_bytes(json.dumps(payload).encode("utf-8"))


def _capture(receipt: codec.MemoryReceipt) -> transaction.CapturedTransactionPayload:
    return transaction.capture_transaction_payload(
        receipt, workspace_id="a" * 32, workspace_identity=(1, 2),
    )


@pytest.mark.parametrize("field", _FIELDS)
def test_missing_scalar_is_rejected_before_capture(field: str) -> None:
    receipt = _receipt(omit=field)
    assert getattr(receipt, field) is None
    with pytest.raises(transaction.MemoryRecoveryTransactionError) as caught:
        _capture(receipt)
    assert str(caught.value) == "memory recovery transaction payload is invalid"


@pytest.mark.parametrize("field", _FIELDS[:3])
@pytest.mark.parametrize("invalid", [None, True, -1, MAX_INT + 1, "2", 2.0])
def test_invalid_counter_bytes_keep_content_free_refusal(field: str, invalid: object) -> None:
    receipt = _receipt()
    payload = dict(receipt.raw_fields)
    payload[field] = invalid
    inconsistent = replace(receipt, raw_bytes=json.dumps(payload).encode("utf-8"))
    with pytest.raises(transaction.MemoryRecoveryTransactionError) as caught:
        _capture(inconsistent)
    assert str(caught.value) == "memory recovery transaction payload is invalid"


@pytest.mark.parametrize("invalid", [None, "", "   ", True, 1])
def test_invalid_key_bytes_keep_content_free_refusal(invalid: object) -> None:
    receipt = _receipt()
    payload = dict(receipt.raw_fields)
    payload["session_key"] = invalid
    inconsistent = replace(receipt, raw_bytes=json.dumps(payload).encode("utf-8"))
    with pytest.raises(transaction.MemoryRecoveryTransactionError) as caught:
        _capture(inconsistent)
    assert str(caught.value) == "memory recovery transaction payload is invalid"


def test_equal_offsets_still_fail_after_the_receipt_parser_accepts_them() -> None:
    receipt = _receipt()
    payload = dict(receipt.raw_fields)
    payload["end_offset"] = payload["start_offset"]
    parsed = codec.decode_memory_receipt_bytes(json.dumps(payload).encode("utf-8"))
    with pytest.raises(transaction.MemoryRecoveryTransactionError):
        _capture(parsed)


@pytest.mark.parametrize("start,end,generation", [(0, 1, 0), (2, 7, 3), (0, MAX_INT, MAX_INT)])
def test_valid_scalar_values_and_utf8_hash_are_preserved(
    start: int, end: int, generation: int,
) -> None:
    payload = dict(_receipt().raw_fields)
    payload.update(start_offset=start, end_offset=end, session_generation=generation)
    parsed = codec.decode_memory_receipt_bytes(json.dumps(payload).encode("utf-8"))
    captured = _capture(parsed)
    assert (captured.start_offset, captured.end_offset, captured.session_generation) == (
        start, end, generation,
    )
    assert captured.session_key_digest == hashlib.sha256("private session 🌙".encode()).hexdigest()
    assert captured.receipt_digest == hashlib.sha256(parsed.raw_bytes).hexdigest()
    assert transaction.decode_transaction_payload("CAPTURED", captured.to_payload()) == captured


def test_capture_retains_repeated_reads_and_validator_order(monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = _receipt()
    reads: list[str] = []
    phases: list[tuple[str, object]] = []
    original_get = codec.MemoryReceipt.__getattribute__
    original_counter = transaction._counter
    original_text = transaction._text

    def get(instance: codec.MemoryReceipt, name: str) -> object:
        if instance is receipt and name in _FIELDS:
            reads.append(name)
        return original_get(instance, name)

    def counter(value: object, *, positive: bool = False) -> int:
        phases.append(("counter", value))
        return original_counter(value, positive=positive)

    def text(value: object) -> str:
        phases.append(("text", value))
        return original_text(value)

    monkeypatch.setattr(transaction, "_receipt_from_exact_bytes", lambda value: receipt)
    monkeypatch.setattr(codec.MemoryReceipt, "__getattribute__", get)
    monkeypatch.setattr(transaction, "_counter", counter)
    monkeypatch.setattr(transaction, "_text", text)
    captured = _capture(receipt)
    assert captured.start_offset == 2
    assert reads == [
        "start_offset", "end_offset", "session_generation", "start_offset", "end_offset",
        "session_key", "session_key", "start_offset", "end_offset", "session_generation",
    ]
    assert phases == [
        ("counter", 1), ("counter", 2), ("counter", 2), ("counter", 7), ("counter", 3),
        ("text", "private session 🌙"),
    ]


@pytest.mark.parametrize("expected", [b"", b"private memory"])
def test_blob_guard_passes_the_exact_digest_once(
    monkeypatch: pytest.MonkeyPatch, expected: bytes,
) -> None:
    digest = hashlib.sha256(expected).hexdigest()
    descriptor = transaction.NewTargetDescriptor(True, len(expected), digest)
    reads: list[object] = []

    def read_blob(_self: object, value: object) -> bytes:
        reads.append(value)
        return expected

    authority = object.__new__(memory_stage.storage.WorkspaceMemoryAuthority)
    monkeypatch.setattr(memory_stage.storage.WorkspaceMemoryAuthority, "read_blob", read_blob)
    memory_stage._check_memory_blob(authority, descriptor, expected)
    assert reads == [digest]
    assert reads[0] is descriptor.blob_digest


@pytest.mark.parametrize(
    "expected,exists,size,digest,reached",
    [
        (None, False, 0, None, ["exists", "blob_digest", "size"]),
        (None, True, 0, None, ["exists"]),
        (None, False, 0, "f" * 64, ["exists", "blob_digest"]),
        (None, False, 1, None, ["exists", "blob_digest", "size"]),
        (b"memory", False, 6, None, ["exists"]),
        (b"memory", True, 6, None, ["exists", "blob_digest"]),
        (b"memory", True, 6, "f" * 64, ["exists", "blob_digest"]),
        (b"memory", True, 7, hashlib.sha256(b"memory").hexdigest(),
         ["exists", "blob_digest", "size"]),
    ],
)
def test_blob_guard_retains_short_circuits_without_reading_authority(
    monkeypatch: pytest.MonkeyPatch, expected: bytes | None, exists: bool,
    size: int, digest: str | None, reached: list[str],
) -> None:
    descriptor = transaction.NewTargetDescriptor(exists, size, digest)
    reads: list[str] = []
    original_get = transaction.NewTargetDescriptor.__getattribute__

    def get(instance: transaction.NewTargetDescriptor, name: str) -> object:
        if instance is descriptor and name in {"exists", "blob_digest", "size"}:
            reads.append(name)
        return original_get(instance, name)

    def forbidden_read(_self: object, value: object) -> bytes:
        pytest.fail("blob authority must not be reached by this descriptor")

    authority = object.__new__(memory_stage.storage.WorkspaceMemoryAuthority)
    monkeypatch.setattr(transaction.NewTargetDescriptor, "__getattribute__", get)
    monkeypatch.setattr(memory_stage.storage.WorkspaceMemoryAuthority, "read_blob", forbidden_read)
    if expected is None and not exists and size == 0 and digest is None:
        memory_stage._check_memory_blob(authority, descriptor, expected)
    else:
        with pytest.raises(memory_stage.MemoryRecoveryTransactionMemoryError) as caught:
            memory_stage._check_memory_blob(authority, descriptor, expected)
        assert str(caught.value) == "memory recovery transaction memory proof is unavailable"
    assert reads == reached


def test_blob_read_fault_retains_identity_and_the_second_digest_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = b"memory"
    digest = hashlib.sha256(expected).hexdigest()
    descriptor = transaction.NewTargetDescriptor(True, len(expected), digest)
    fault = RuntimeError("opaque blob read failure")
    reads: list[str] = []
    original_get = transaction.NewTargetDescriptor.__getattribute__

    def get(instance: transaction.NewTargetDescriptor, name: str) -> object:
        if instance is descriptor and name in {"exists", "blob_digest", "size"}:
            reads.append(name)
        return original_get(instance, name)

    def read_blob(_self: object, value: object) -> bytes:
        assert value is digest
        raise fault

    authority = object.__new__(memory_stage.storage.WorkspaceMemoryAuthority)
    monkeypatch.setattr(transaction.NewTargetDescriptor, "__getattribute__", get)
    monkeypatch.setattr(memory_stage.storage.WorkspaceMemoryAuthority, "read_blob", read_blob)
    with pytest.raises(RuntimeError) as caught:
        memory_stage._check_memory_blob(authority, descriptor, expected)
    assert caught.value is fault
    assert reads == ["exists", "blob_digest", "size", "blob_digest"]
