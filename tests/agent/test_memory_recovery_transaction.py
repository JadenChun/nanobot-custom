from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent.memory_recovery import MemoryReceipt
from nanobot.agent.memory_recovery_manifest import (
    MAX_INT,
    MemoryRecoveryManifest,
    derive_manifest_identity,
)
from nanobot.agent.memory_recovery_transaction import (
    CapturedTransactionPayload,
    MemoryRecoveryTransactionError,
    capture_transaction_payload,
    decode_transaction_payload,
    prepare_transaction_payload,
)

WORKSPACE_ID = "a" * 32


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _receipt(**changes: Any) -> MemoryReceipt:
    output = "private memory and transcript text 🌙"
    values: dict[str, Any] = {
        "version": 1,
        "chunk_digest": _digest(b"chunk"),
        "history_marker_id": "private-marker-id",
        "history_entry": "private history row",
        "memory_output": output,
        "base_memory_digest": _digest(b"old memory"),
        "output_memory_digest": _digest(output.encode()),
        "reference_only": False,
        "created_at": "2026-09-24T00:00:00Z",
        "session_key": "private-session-key",
        "start_offset": 4,
        "end_offset": 9,
        "session_generation": 2,
    }
    values.update(changes)
    raw_bytes = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return MemoryReceipt(
        version=values["version"],
        chunk_digest=values["chunk_digest"],
        history_marker_id=values["history_marker_id"],
        history_entry=values["history_entry"],
        memory_output=values["memory_output"],
        base_memory_digest=values["base_memory_digest"],
        output_memory_digest=values["output_memory_digest"],
        reference_only=values["reference_only"],
        created_at=values["created_at"],
        session_key=values.get("session_key"),
        start_offset=values.get("start_offset"),
        end_offset=values.get("end_offset"),
        session_generation=values.get("session_generation"),
        raw_fields=MappingProxyType(values),
        raw_bytes=raw_bytes,
    )


def _stat(size: int) -> dict[str, int | None]:
    return {
        "device": 1,
        "inode": 2,
        "mode": 0o100600,
        "uid": 501,
        "gid": 20,
        "links": 1,
        "size": size,
        "mtime_ns": 10,
        "ctime_ns": 11,
    }


def _prepared_inputs(
    captured: CapturedTransactionPayload, receipt: MemoryReceipt | None = None
) -> dict[str, Any]:
    if receipt is None:
        receipt = _receipt()
    names = ("history.jsonl", ".cursor", "MEMORY.md")
    targets: dict[str, Any] = {}
    for name in names:
        old_content = b"old memory" if name == "MEMORY.md" else f"old-{name}".encode()
        new_content = f"new-{name}".encode()
        if name == "MEMORY.md" and captured.base_memory_digest == _digest(b""):
            old_target = {"exists": False, "size": 0, "digest": None, "stat": None}
        else:
            old_target = {
                "exists": True,
                "size": len(old_content),
                "digest": _digest(old_content),
                "stat": _stat(len(old_content)),
            }
        if name == "MEMORY.md" and captured.reference_only:
            new_target = {
                "exists": old_target["exists"],
                "size": old_target["size"],
                "blob_digest": old_target["digest"],
            }
        elif name == "MEMORY.md":
            new_target = {
                "exists": True,
                "size": captured.output_memory_size,
                "blob_digest": captured.output_memory_digest,
            }
        else:
            new_target = {
                "exists": True,
                "size": len(new_content),
                "blob_digest": _digest(new_content),
            }
        targets[name] = {"old": old_target, "new": new_target}
    source = {
        "old_digest": _digest(b"old source"),
        "old_size": 10,
        "old_generation": 2,
        "old_revision": 3,
        "old_cursor": 4,
        "new_digest": _digest(b"new source"),
        "new_size": 10,
        "new_generation": 2,
        "new_revision": 4,
        "new_cursor": 9,
        "action": "advance",
    }
    return {
        "receipt_stat": _stat(len(receipt.raw_bytes)),
        "targets": targets,
        "source": source,
        "history_marker_digest": _digest(b"private-marker-id"),
        "history_entry_digest": _digest(b"private history row"),
        "history_row_digest": _digest(b"private history row\n"),
        "memory_action": "replace",
    }


def test_capture_payload_is_deterministic_bound_and_content_free() -> None:
    receipt = _receipt()
    first = capture_transaction_payload(
        receipt, workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    )
    second = capture_transaction_payload(
        receipt, workspace_id="b" * 32, workspace_identity=(99, 123)
    )
    payload = first.to_payload()
    assert first == second
    assert first.to_payload() == second.to_payload()
    assert first.identity == derive_manifest_identity("transactions", payload)
    assert payload["receipt_digest"] == _digest(receipt.raw_bytes)
    assert payload["source_format"] == "legacy_jsonl"
    assert payload["conversion_identity"] is None
    assert payload["output_memory_size"] == len(receipt.memory_output.encode("utf-8"))
    assert "workspace_id" not in payload
    assert "workspace_identity" not in payload
    assert decode_transaction_payload("CAPTURED", payload) == first
    serialized = json.dumps(payload)
    for secret in (
        receipt.session_key,
        receipt.history_marker_id,
        receipt.history_entry,
        receipt.created_at,
        receipt.memory_output,
    ):
        assert secret not in serialized


@pytest.mark.parametrize("workspace_id", ["workspace-1", "A" * 32, "a" * 31, "a" * 33])
def test_capture_requires_canonical_workspace_id(workspace_id: str) -> None:
    with pytest.raises(MemoryRecoveryTransactionError):
        capture_transaction_payload(
            _receipt(), workspace_id=workspace_id, workspace_identity=(1, 2)
        )


def test_changed_exact_receipt_bytes_change_portable_transaction_identity() -> None:
    receipt = _receipt()
    alternate_bytes = json.dumps(dict(receipt.raw_fields), indent=2).encode()
    alternate_receipt = replace(receipt, raw_bytes=alternate_bytes)
    first = capture_transaction_payload(
        receipt, workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    )
    second = capture_transaction_payload(
        alternate_receipt, workspace_id="b" * 32, workspace_identity=(90, 10)
    )
    assert first.receipt_digest != second.receipt_digest
    assert first.identity != second.identity


def test_captured_identity_does_not_depend_on_prepared_target_changes() -> None:
    captured = capture_transaction_payload(
        _receipt(), workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    )
    prepared_one = prepare_transaction_payload(captured, **_prepared_inputs(captured))
    inputs = _prepared_inputs(captured)
    inputs["targets"]["history.jsonl"]["new"]["blob_digest"] = _digest(b"different target")
    prepared_two = prepare_transaction_payload(captured, **inputs)
    assert captured.identity == derive_manifest_identity("transactions", captured.to_payload())
    assert prepared_one.to_payload() != prepared_two.to_payload()
    assert decode_transaction_payload("PREPARED", prepared_one.to_payload(), captured=captured) == prepared_one
    assert decode_transaction_payload("PREPARED", prepared_two.to_payload(), captured=captured) == prepared_two


def test_prepare_payload_round_trips_typed_descriptors() -> None:
    captured = capture_transaction_payload(
        _receipt(), workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    )
    prepared = prepare_transaction_payload(captured, **_prepared_inputs(captured))
    decoded = decode_transaction_payload("PREPARED", prepared.to_payload(), captured=captured)
    assert decoded == prepared
    assert decoded.targets["history.jsonl"].old.stat.device == 1
    assert decoded.targets["MEMORY.md"].new.blob_digest == captured.output_memory_digest
    assert decoded.source.old_digest == _digest(b"old source")
    assert decoded.receipt_digest == captured.receipt_digest


@pytest.mark.parametrize(
    "changes",
    [
        {"session_key": None},
        {"start_offset": None},
        {"end_offset": None},
        {"session_generation": None},
        {"start_offset": 9, "end_offset": 9},
        {"start_offset": 10, "end_offset": 9},
        {"chunk_digest": "not-a-digest"},
        {"base_memory_digest": ""},
        {"output_memory_digest": "f" * 64},
    ],
)
def test_capture_rejects_missing_or_invalid_session_evidence(changes: dict[str, Any]) -> None:
    with pytest.raises(MemoryRecoveryTransactionError) as raised:
        capture_transaction_payload(
            _receipt(**changes), workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
        )
    assert str(raised.value) == "memory recovery transaction payload is invalid"
    assert "private" not in str(raised.value)


def test_capture_rejects_mismatched_raw_receipt_evidence() -> None:
    receipt = _receipt()
    with pytest.raises(MemoryRecoveryTransactionError):
        capture_transaction_payload(
            replace(receipt, raw_bytes=b"different receipt bytes"),
            workspace_id=WORKSPACE_ID,
            workspace_identity=(1, 2),
        )


def test_strict_decoder_rejects_phase_unknown_fields_and_malformed_hashes() -> None:
    captured = capture_transaction_payload(
        _receipt(), workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    ).to_payload()
    for phase, payload in (
        ("OTHER", captured),
        ("CAPTURED", {**captured, "unknown_secret": "private"}),
        ("CAPTURED", {**captured, "receipt_digest": "bad"}),
        ("CAPTURED", {**captured, "schema_version": True}),
        ("CAPTURED", {**captured, "schema_version": 2}),
        ("CAPTURED", {**captured, "private_metadata": "x" * 5_000}),
        ("CAPTURED", {**captured, "workspace_id": WORKSPACE_ID}),
        ("CAPTURED", {**captured, "workspace_identity": {"device": 1, "inode": 2}}),
    ):
        with pytest.raises(MemoryRecoveryTransactionError) as raised:
            decode_transaction_payload(phase, payload)
        assert "private" not in str(raised.value)


def test_decoder_enforces_string_record_and_depth_caps_without_leaking() -> None:
    payload = capture_transaction_payload(
        _receipt(), workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    ).to_payload()
    with pytest.raises(MemoryRecoveryTransactionError):
        decode_transaction_payload("CAPTURED", {**payload, "source_format": "x" * 4097})
    with pytest.raises(MemoryRecoveryTransactionError):
        decode_transaction_payload("CAPTURED", json.dumps(payload) + " " * 65_536)
    with pytest.raises(MemoryRecoveryTransactionError):
        decode_transaction_payload("CAPTURED", b'{"schema_version":1,"schema_version":1}')

    nested: dict[str, Any] = {}
    cursor = nested
    for _ in range(10):
        child: dict[str, Any] = {}
        cursor["x"] = child
        cursor = child
    with pytest.raises(MemoryRecoveryTransactionError):
        decode_transaction_payload("CAPTURED", nested)


def test_prepared_receipt_digest_and_target_bindings_must_match() -> None:
    captured = capture_transaction_payload(
        _receipt(), workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    )
    prepared = prepare_transaction_payload(captured, **_prepared_inputs(captured)).to_payload()
    altered = dict(prepared)
    altered["receipt_digest"] = _digest(b"other receipt")
    with pytest.raises(MemoryRecoveryTransactionError):
        decode_transaction_payload("PREPARED", altered, captured=captured)

    altered_target = json.loads(json.dumps(prepared))
    altered_target["targets"]["MEMORY.md"]["old"]["stat"]["size"] += 1
    with pytest.raises(MemoryRecoveryTransactionError):
        decode_transaction_payload("PREPARED", altered_target, captured=captured)


def test_memory_target_old_digest_must_match_receipt_base() -> None:
    captured = capture_transaction_payload(
        _receipt(), workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    )
    inputs = _prepared_inputs(captured)
    inputs["targets"]["MEMORY.md"]["old"]["digest"] = _digest(b"unrelated memory")
    with pytest.raises(MemoryRecoveryTransactionError):
        prepare_transaction_payload(captured, **inputs)


def test_absent_memory_requires_empty_base_digest() -> None:
    nonempty_base = capture_transaction_payload(
        _receipt(), workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    )
    inputs = _prepared_inputs(nonempty_base)
    inputs["targets"]["MEMORY.md"]["old"] = {
        "exists": False,
        "size": 0,
        "digest": None,
        "stat": None,
    }
    with pytest.raises(MemoryRecoveryTransactionError):
        prepare_transaction_payload(nonempty_base, **inputs)

    empty_receipt = _receipt(base_memory_digest=_digest(b""))
    empty_base = capture_transaction_payload(
        empty_receipt, workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    )
    valid_inputs = _prepared_inputs(empty_base, empty_receipt)
    assert prepare_transaction_payload(empty_base, **valid_inputs)


@pytest.mark.parametrize(
    "target_update",
    [
        {"exists": False, "size": 0, "blob_digest": None},
        {"exists": True, "size": 1, "blob_digest": _digest(b"different memory")},
        {
            "exists": True,
            "size": 1,
            "blob_digest": _digest(_receipt().memory_output.encode("utf-8")),
        },
    ],
)
def test_replace_action_requires_memory_output_blob(
    target_update: dict[str, object],
) -> None:
    captured = capture_transaction_payload(
        _receipt(), workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    )
    inputs = _prepared_inputs(captured)
    inputs["targets"]["MEMORY.md"]["new"] = target_update
    with pytest.raises(MemoryRecoveryTransactionError):
        prepare_transaction_payload(captured, **inputs)


@pytest.mark.parametrize(
    "target_update",
    [
        {"exists": False, "size": 0, "blob_digest": None},
        {"exists": True, "size": 1, "blob_digest": _digest(b"different memory")},
        {
            "exists": True,
            "size": len(b"old-MEMORY.md") + 1,
            "blob_digest": _digest(b"old-MEMORY.md"),
        },
    ],
)
def test_preserve_action_cannot_change_memory_target(
    target_update: dict[str, object],
) -> None:
    receipt = _receipt(reference_only=True)
    captured = capture_transaction_payload(
        receipt, workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    )
    inputs = _prepared_inputs(captured, receipt)
    inputs["targets"]["MEMORY.md"]["new"] = target_update
    inputs["memory_action"] = "preserve"
    with pytest.raises(MemoryRecoveryTransactionError):
        prepare_transaction_payload(captured, **inputs)


def test_preserve_action_round_trips_an_unchanged_memory_target() -> None:
    receipt = _receipt(reference_only=True)
    captured = capture_transaction_payload(
        receipt, workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    )
    inputs = _prepared_inputs(captured, receipt)
    inputs["memory_action"] = "preserve"
    prepared = prepare_transaction_payload(captured, **inputs)
    decoded = decode_transaction_payload(
        "PREPARED", prepared.to_payload(), captured=captured
    )
    assert decoded.memory_action == "preserve"
    assert decoded.targets["MEMORY.md"].old.size == decoded.targets["MEMORY.md"].new.size
    assert (
        decoded.targets["MEMORY.md"].old.digest
        == decoded.targets["MEMORY.md"].new.blob_digest
    )


@pytest.mark.parametrize(
    "source_update",
    [
        {"new_revision": 3},
        {"new_revision": 2},
        {"old_revision": MAX_INT, "new_revision": MAX_INT},
        {"new_digest": _digest(b"old source")},
    ],
)
def test_advance_action_requires_forward_revision_and_changed_source(
    source_update: dict[str, int | str],
) -> None:
    captured = capture_transaction_payload(
        _receipt(), workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    )
    inputs = _prepared_inputs(captured)
    inputs["source"].update(source_update)
    with pytest.raises(MemoryRecoveryTransactionError):
        prepare_transaction_payload(captured, **inputs)


def test_manifest_chain_round_trips_through_typed_decoder(tmp_path: Path) -> None:
    captured = capture_transaction_payload(
        _receipt(), workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    )
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    namespace = authority.open_evidence_namespace("transactions", captured.identity)
    manifest = MemoryRecoveryManifest(namespace)
    try:
        first = manifest.create(captured.to_payload())
        assert decode_transaction_payload("CAPTURED", first.payload) == captured
        prepared = prepare_transaction_payload(captured, **_prepared_inputs(captured))
        second = manifest.append(
            predecessor_digest=first.digest,
            phase="PREPARED",
            payload=prepared.to_payload(),
        )
        assert decode_transaction_payload(
            second.phase, second.payload, captured=captured
        ) == prepared
    finally:
        namespace.close()
        authority.close()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["targets"]["MEMORY.md"]["new"].update(
            blob_digest=_digest(b"wrong output")
        ),
        lambda payload: payload["source"].update(new_revision=3),
        lambda payload: payload["source"].update(new_digest=_digest(b"old source")),
    ],
)
def test_typed_decoder_rejects_manifest_semantic_contradictions(
    tmp_path: Path, mutate: Any
) -> None:
    captured = capture_transaction_payload(
        _receipt(), workspace_id=WORKSPACE_ID, workspace_identity=(1, 2)
    )
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    namespace = authority.open_evidence_namespace("transactions", captured.identity)
    manifest = MemoryRecoveryManifest(namespace)
    try:
        first = manifest.create(captured.to_payload())
        prepared_payload = prepare_transaction_payload(
            captured, **_prepared_inputs(captured)
        ).to_payload()
        mutate(prepared_payload)
        second = manifest.append(
            predecessor_digest=first.digest,
            phase="PREPARED",
            payload=prepared_payload,
        )
        stored = manifest.load()[1]
        assert stored == second
        with pytest.raises(MemoryRecoveryTransactionError):
            decode_transaction_payload(
                stored.phase, stored.payload, captured=captured
            )
    finally:
        namespace.close()
        authority.close()
