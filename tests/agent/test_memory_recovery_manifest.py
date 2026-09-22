"""Focused tests for the immutable memory-recovery manifest protocol."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent.memory_recovery_manifest import (
    MAX_CHAIN_RECORDS,
    MemoryManifestError,
    MemoryRecoveryManifest,
    derive_manifest_identity,
)


def _identity(kind: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(b"nanobot-memory-manifest-v1\0" + kind.encode("ascii") + b"\0" + encoded).hexdigest()


def _open(tmp_path: Path, kind: str, payload: dict[str, object]) -> tuple[storage.WorkspaceMemoryAuthority, storage.ImmutableEvidenceNamespace, MemoryRecoveryManifest]:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    identity = derive_manifest_identity(kind, payload)
    namespace = authority.open_evidence_namespace(kind, identity)
    return authority, namespace, MemoryRecoveryManifest(namespace)


def _close(authority: storage.WorkspaceMemoryAuthority, namespace: storage.ImmutableEvidenceNamespace) -> None:
    namespace.close()
    authority.close()


def _record_bytes(
    *,
    kind: str,
    identity: str,
    sequence: int,
    predecessor_digest: str | None,
    phase: str,
    payload: dict[str, object],
    **extra: object,
) -> bytes:
    body: dict[str, object] = {
        "version": 1,
        "kind": kind,
        "identity": identity,
        "sequence": sequence,
        "predecessor_digest": predecessor_digest,
        "phase": phase,
        "payload": payload,
    }
    body.update(extra)
    return json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _fork_capable() -> bool:
    try:
        multiprocessing.get_context("fork")
    except (AttributeError, ValueError):
        return False
    return os.name == "posix"


def _append_child(
    workspace: str,
    kind: str,
    identity: str,
    predecessor_digest: str,
    payload: dict[str, object],
    barrier: Any,
    result: Any,
) -> None:
    try:
        barrier.wait(timeout=5)
        with storage.WorkspaceMemoryAuthority(Path(workspace)) as authority:
            namespace = authority.open_evidence_namespace(kind, identity)
            manifest = MemoryRecoveryManifest(namespace)
            record = manifest.append(
                predecessor_digest=predecessor_digest,
                phase="PREPARED",
                payload=payload,
            )
            result.put(("ok", record.digest))
    except BaseException as error:
        result.put(("error", type(error).__name__, str(error)))


def _inherited_process_child(
    authority: storage.WorkspaceMemoryAuthority,
    manifest: MemoryRecoveryManifest,
    kind: str,
    identity: str,
    result: Any,
) -> None:
    try:
        manifest.load()
    except BaseException as error:
        manifest_result = (type(error).__name__, getattr(error, "code", None), str(error))
    else:
        manifest_result = ("ok", None, None)
    try:
        authority.open_evidence_namespace(kind, identity)
    except BaseException as error:
        authority_result = (type(error).__name__, getattr(error, "code", None), str(error))
    else:
        authority_result = ("ok", None, None)
    result.put((manifest_result, authority_result))


def test_identity_is_domain_separated_and_canonical() -> None:
    payload = {"z": "é", "a": [True, None, -2]}
    assert derive_manifest_identity("transactions", payload) == _identity("transactions", payload)
    assert derive_manifest_identity("transactions", {"a": payload["a"], "z": "é"}) == derive_manifest_identity(
        "transactions", payload
    )
    assert derive_manifest_identity("transactions", payload) != derive_manifest_identity("conversions", payload)


def test_golden_genesis_bytes_and_identity(tmp_path: Path) -> None:
    captured = {"z": "é", "a": [True, None, -2]}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        record = manifest.create(captured)
        expected_identity = _identity("transactions", captured)
        expected = _record_bytes(
            kind="transactions",
            identity=expected_identity,
            sequence=0,
            predecessor_digest=None,
            phase="CAPTURED",
            payload=captured,
        )
        assert namespace.identity == expected_identity
        assert record.raw_bytes == expected
        assert record.digest == hashlib.sha256(expected).hexdigest()
        assert record.payload == captured
    finally:
        _close(authority, namespace)


def test_transaction_chain_accepts_every_prefix_and_full_sequence(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        record = manifest.create(captured)
        phases = (
            "PREPARED",
            "HISTORY_APPLIED",
            "MEMORY_APPLIED",
            "OFFSET_APPLIED",
            "RECEIPT_COMMITTED",
            "COMPLETE",
        )
        for phase in phases:
            record = manifest.append(
                predecessor_digest=record.digest,
                phase=phase,
                payload={"phase": phase},
            )
        loaded = manifest.load()
        assert len(loaded) == MAX_CHAIN_RECORDS
        assert tuple(item.phase for item in loaded) == ("CAPTURED", *phases)
        assert loaded[-1].phase == "COMPLETE"
    finally:
        _close(authority, namespace)


def test_create_and_append_produce_canonical_detached_records(tmp_path: Path) -> None:
    captured = {"chunk": "é", "nested": {"items": [1, False]}}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        first = manifest.create(captured)
        assert first.version == 1
        assert first.kind == "transactions"
        assert first.identity == namespace.identity
        assert first.sequence == 0
        assert first.predecessor_digest is None
        assert first.phase == "CAPTURED"
        assert first.digest == hashlib.sha256(first.raw_bytes).hexdigest()
        detached = first.payload
        detached["nested"]["items"].append("changed")  # type: ignore[index]
        assert first.payload == captured

        second = manifest.append(
            predecessor_digest=first.digest,
            phase="PREPARED",
            payload={"step": 1},
        )
        assert second.sequence == 1
        assert second.predecessor_digest == first.digest
        assert manifest.load() == (first, second)
    finally:
        _close(authority, namespace)


def test_complete_chain_and_terminal_append_are_rejected(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "conversions", captured)
    try:
        record = manifest.create(captured)
        phases = ("CONVERSION_PREPARED", "JSONL_APPLIED", "CURSORS_APPLIED", "LEGACY_FINALIZED", "COMPLETE")
        for phase in phases:
            record = manifest.append(predecessor_digest=record.digest, phase=phase, payload={"phase": phase})
        assert len(manifest.load()) == 6
        with pytest.raises(MemoryManifestError) as raised:
            manifest.append(predecessor_digest=record.digest, phase="COMPLETE", payload={})
        assert raised.value.code == "terminal"
        assert len(manifest.load()) == 6
    finally:
        _close(authority, namespace)


@pytest.mark.parametrize(
    "payload",
    [
        {"value": 1.0},
        {"value": float("nan")},
        {"value": b"bytes"},
        {"value": (1, 2)},
        {"value": 2**63},
        {"value": True, 1: "bad-key"},
    ],
)
def test_invalid_payloads_are_content_free(payload: dict[Any, Any]) -> None:
    with pytest.raises(MemoryManifestError) as raised:
        derive_manifest_identity("transactions", payload)
    assert raised.value.code == "invalid"
    assert str(raised.value) == "memory recovery manifest input is invalid"
    assert all(secret not in str(raised.value) for secret in ("value", "bytes", "nan"))


def test_signed_integer_endpoints_and_immediate_prohibited_values() -> None:
    accepted_min = {"value": -(2**63 - 1)}
    accepted_max = {"value": 2**63 - 1}
    assert derive_manifest_identity("transactions", accepted_min)
    assert derive_manifest_identity("transactions", accepted_max)
    for payload in ({"value": -(2**63)}, {"value": 2**63}):
        with pytest.raises(MemoryManifestError) as raised:
            derive_manifest_identity("transactions", payload)
        assert raised.value.code == "invalid"


@pytest.mark.parametrize("value", [-(2**63), 2**63])
def test_persisted_immediate_prohibited_integer_is_corrupt(tmp_path: Path, value: int) -> None:
    payload = {"value": value}
    payload_bytes = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    identity = hashlib.sha256(
        b"nanobot-memory-manifest-v1\0transactions\0" + payload_bytes
    ).hexdigest()
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    namespace = authority.open_evidence_namespace("transactions", identity)
    manifest = MemoryRecoveryManifest(namespace)
    try:
        namespace.publish_record(
            "genesis.json",
            _record_bytes(
                kind="transactions",
                identity=identity,
                sequence=0,
                predecessor_digest=None,
                phase="CAPTURED",
                payload=payload,
            ),
        )
        with pytest.raises(MemoryManifestError) as raised:
            manifest.load()
        assert raised.value.code == "corrupt"
    finally:
        _close(authority, namespace)


def test_exact_depth_node_key_and_string_boundaries() -> None:
    depth_payload: dict[str, object] = {}
    current = depth_payload
    for _ in range(6):
        child: dict[str, object] = {}
        current["child"] = child
        current = child
    current["value"] = 1
    assert derive_manifest_identity("transactions", depth_payload)
    current["child"] = {"value": 1}
    with pytest.raises(MemoryManifestError) as raised:
        derive_manifest_identity("transactions", depth_payload)
    assert raised.value.code == "invalid"

    exact_nodes = {f"k{index}": index for index in range(1_023)}
    assert derive_manifest_identity("transactions", exact_nodes)
    too_many_nodes = {f"k{index}": index for index in range(1_024)}
    with pytest.raises(MemoryManifestError) as raised:
        derive_manifest_identity("transactions", too_many_nodes)
    assert raised.value.code == "invalid"

    exact_key = {"é" * 64: 1}
    assert derive_manifest_identity("transactions", exact_key)
    too_long_key = {"é" * 64 + "a": 1}
    with pytest.raises(MemoryManifestError) as raised:
        derive_manifest_identity("transactions", too_long_key)
    assert raised.value.code == "invalid"

    exact_string = {"value": "é" * 2_048}
    assert derive_manifest_identity("transactions", exact_string)
    too_long_string = {"value": "é" * 2_048 + "a"}
    with pytest.raises(MemoryManifestError) as raised:
        derive_manifest_identity("transactions", too_long_string)
    assert raised.value.code == "invalid"


def test_exact_full_record_boundary_and_one_over_are_deterministic(tmp_path: Path) -> None:
    accepted = {"items": ["x" * 64] * 975 + ["x" * 15]}
    too_large = {"items": ["x" * 64] * 975 + ["x" * 16]}
    authority, namespace, manifest = _open(tmp_path, "transactions", accepted)
    try:
        record = manifest.create(accepted)
        assert len(record.raw_bytes) == 65_536
    finally:
        _close(authority, namespace)

    too_large_workspace = tmp_path / "too-large"
    too_large_workspace.mkdir()
    authority, namespace, manifest = _open(too_large_workspace, "transactions", too_large)
    try:
        with pytest.raises(MemoryManifestError) as raised:
            manifest.create(too_large)
        assert raised.value.code == "invalid"
    finally:
        _close(authority, namespace)


@pytest.mark.parametrize(
    "value",
    [
        {"value": type("DictSubclass", (dict,), {})()},
        {"value": type("ListSubclass", (list,), {})()},
        {"value": type("StringSubclass", (str,), {})("value")},
        {"value": type("IntSubclass", (int,), {})(1)},
    ],
)
def test_exact_type_rejections_and_reference_cycles_are_invalid(value: dict[str, object]) -> None:
    with pytest.raises(MemoryManifestError) as raised:
        derive_manifest_identity("transactions", value)
    assert raised.value.code == "invalid"

    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    with pytest.raises(MemoryManifestError) as raised:
        derive_manifest_identity("transactions", cycle)
    assert raised.value.code == "invalid"


@pytest.mark.parametrize(
    "raw",
    [
        b'{"version": 1, "kind": "transactions", "identity": "IDENTITY", "sequence": 0, "predecessor_digest": null, "phase": "CAPTURED", "payload": {}}',
        b'{"version":1,"kind":"transactions","identity":"IDENTITY","sequence":0,"predecessor_digest":null,"phase":"CAPTURED","payload":{"x":"\\u00e9"}}',
        b'{"version":1,"kind":"transactions","identity":"IDENTITY","sequence":0,"predecessor_digest":null,"phase":"CAPTURED","payload":{"x":1,"x":2}}',
        b'{"version":1,"kind":"transactions","identity":"IDENTITY","sequence":0,"predecessor_digest":null,"phase":"CAPTURED","payload":{"nested":{"x":1,"x":2}}}',
        b'{"version":1,"kind":"transactions","identity":"IDENTITY","sequence":0,"predecessor_digest":null,"phase":"CAPTURED","payload":{}}\xff',
    ],
)
def test_parser_rejects_noncanonical_duplicate_or_invalid_utf8_records(
    tmp_path: Path, raw: bytes
) -> None:
    captured: dict[str, object] = {}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        identity = namespace.identity
        namespace.publish_record(
            "genesis.json",
            raw.replace(b"IDENTITY", identity.encode("ascii")),
        )
        with pytest.raises(MemoryManifestError) as raised:
            manifest.load()
        assert raised.value.code == "corrupt"
        assert str(raised.value) == "memory recovery manifest evidence is invalid"
    finally:
        _close(authority, namespace)


@pytest.mark.parametrize(
    "payload",
    [
        {"k" * 129: 1},
        {"value": "x" * 4097},
        {"value": 2**63},
        {"value": -(2**63) - 1},
        {"value": [1] * 1024},
    ],
)
def test_payload_bounds_are_rejected_at_input_boundary(payload: dict[str, object]) -> None:
    with pytest.raises(MemoryManifestError) as raised:
        derive_manifest_identity("transactions", payload)
    assert raised.value.code == "invalid"
    assert str(raised.value) == "memory recovery manifest input is invalid"


def test_depth_and_record_size_bounds_are_rejected(tmp_path: Path) -> None:
    nested: dict[str, object] = {}
    current = nested
    for _ in range(8):
        child: dict[str, object] = {}
        current["child"] = child
        current = child
    with pytest.raises(MemoryManifestError) as raised:
        derive_manifest_identity("transactions", {"root": nested})
    assert raised.value.code == "invalid"

    captured = {str(index): "x" * 4096 for index in range(100)}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        with pytest.raises(MemoryManifestError) as raised:
            manifest.create(captured)
        assert raised.value.code == "invalid"
    finally:
        _close(authority, namespace)


def test_unknown_or_noncanonical_records_are_corrupt(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        identity = namespace.identity
        record = {
            "version": 1,
            "kind": "transactions",
            "identity": identity,
            "sequence": 0,
            "predecessor_digest": None,
            "phase": "CAPTURED",
            "payload": captured,
            "unknown": True,
        }
        namespace.publish_record("genesis.json", json.dumps(record).encode("utf-8"))
        with pytest.raises(MemoryManifestError) as raised:
            manifest.load()
        assert raised.value.code == "corrupt"
        assert str(raised.value) == "memory recovery manifest evidence is invalid"
        assert identity not in str(raised.value)
    finally:
        _close(authority, namespace)


@pytest.mark.parametrize(
    "override",
    [
        {"sequence": 1},
        {"predecessor_digest": "a" * 64},
        {"phase": "HISTORY_APPLIED"},
        {"kind": "conversions"},
        {"identity": "b" * 64},
        {"version": True},
        {"payload": []},
    ],
)
def test_reachable_transition_and_link_corruption_is_rejected(
    tmp_path: Path, override: dict[str, object]
) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        identity = namespace.identity
        body: dict[str, object] = {
            "version": 1,
            "kind": "transactions",
            "identity": identity,
            "sequence": 0,
            "predecessor_digest": None,
            "phase": "CAPTURED",
            "payload": captured,
        }
        body.update(override)
        namespace.publish_record(
            "genesis.json",
            json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(),
        )
        with pytest.raises(MemoryManifestError) as raised:
            manifest.load()
        assert raised.value.code == "corrupt"
    finally:
        _close(authority, namespace)


def test_missing_envelope_field_is_corrupt(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        body = {
            "version": 1,
            "kind": "transactions",
            "identity": namespace.identity,
            "sequence": 0,
            "predecessor_digest": None,
            "phase": "CAPTURED",
        }
        namespace.publish_record(
            "genesis.json",
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode(),
        )
        with pytest.raises(MemoryManifestError) as raised:
            manifest.load()
        assert raised.value.code == "corrupt"
    finally:
        _close(authority, namespace)


def test_genesis_identity_must_be_derived_from_its_payload(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        forged_payload = {"captured": False}
        namespace.publish_record(
            "genesis.json",
            _record_bytes(
                kind="transactions",
                identity=namespace.identity,
                sequence=0,
                predecessor_digest=None,
                phase="CAPTURED",
                payload=forged_payload,
            ),
        )
        with pytest.raises(MemoryManifestError) as raised:
            manifest.load()
        assert raised.value.code == "corrupt"
    finally:
        _close(authority, namespace)


def test_unreachable_orphan_slots_are_never_adopted(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        namespace.publish_record("next-" + "f" * 64 + ".json", b"orphan")
        assert manifest.load() == ()
        first = manifest.create(captured)
        assert manifest.load() == (first,)
    finally:
        _close(authority, namespace)


def test_terminal_successor_probe_rejects_any_present_slot(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "conversions", captured)
    try:
        record = manifest.create(captured)
        for phase in (
            "CONVERSION_PREPARED",
            "JSONL_APPLIED",
            "CURSORS_APPLIED",
            "LEGACY_FINALIZED",
            "COMPLETE",
        ):
            record = manifest.append(
                predecessor_digest=record.digest,
                phase=phase,
                payload={"phase": phase},
            )
        namespace.publish_record("next-" + record.digest + ".json", b"present")
        with pytest.raises(MemoryManifestError) as raised:
            manifest.load()
        assert raised.value.code == "corrupt"
    finally:
        _close(authority, namespace)


def test_create_and_append_retries_are_idempotent_including_historical_retry(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        first = manifest.create(captured)
        assert manifest.create(dict(captured)) == first
        second = manifest.append(
            predecessor_digest=first.digest,
            phase="PREPARED",
            payload={"step": 1},
        )
        third = manifest.append(
            predecessor_digest=second.digest,
            phase="HISTORY_APPLIED",
            payload={"step": 2},
        )
        assert manifest.append(
            predecessor_digest=first.digest,
            phase="PREPARED",
            payload={"step": 1},
        ) == second
        assert manifest.load() == (first, second, third)
    finally:
        _close(authority, namespace)


def test_create_retry_after_advancement_republishes_identical_genesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        first = manifest.create(captured)
        manifest.append(
            predecessor_digest=first.digest,
            phase="PREPARED",
            payload={"step": 1},
        )
        original_publish = namespace.publish_record
        calls: list[tuple[str, bytes]] = []

        def publish(slot: str, content: bytes) -> str:
            calls.append((slot, content))
            return original_publish(slot, content)

        monkeypatch.setattr(namespace, "publish_record", publish)
        assert manifest.create(dict(captured)) == first
        assert calls == [("genesis.json", first.raw_bytes)]
    finally:
        _close(authority, namespace)


def test_stale_conflict_and_invalid_transition_errors_are_content_free(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        first = manifest.create(captured)
        with pytest.raises(MemoryManifestError) as raised:
            manifest.append(
                predecessor_digest="a" * 64,
                phase="PREPARED",
                payload={},
            )
        assert raised.value.code == "stale"

        second = manifest.append(
            predecessor_digest=first.digest,
            phase="PREPARED",
            payload={"step": 1},
        )
        with pytest.raises(MemoryManifestError) as raised:
            manifest.append(
                predecessor_digest=first.digest,
                phase="PREPARED",
                payload={"step": 2},
            )
        assert raised.value.code == "conflict"
        with pytest.raises(MemoryManifestError) as raised:
            manifest.append(
                predecessor_digest=second.digest,
                phase="MEMORY_APPLIED",
                payload={},
            )
        assert raised.value.code == "invalid"
        assert set(raised.value.args) == {"memory recovery manifest input is invalid"}
    finally:
        _close(authority, namespace)


def test_terminal_append_returns_terminal_error(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "conversions", captured)
    try:
        record = manifest.create(captured)
        for phase in (
            "CONVERSION_PREPARED",
            "JSONL_APPLIED",
            "CURSORS_APPLIED",
            "LEGACY_FINALIZED",
            "COMPLETE",
        ):
            record = manifest.append(
                predecessor_digest=record.digest,
                phase=phase,
                payload={},
            )
        with pytest.raises(MemoryManifestError) as raised:
            manifest.append(predecessor_digest=record.digest, phase="COMPLETE", payload={})
        assert raised.value.code == "terminal"
    finally:
        _close(authority, namespace)


def test_create_repairs_pending_genesis_companion(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        raw = _record_bytes(
            kind="transactions",
            identity=namespace.identity,
            sequence=0,
            predecessor_digest=None,
            phase="CAPTURED",
            payload=captured,
        )
        namespace.publish_record("genesis.json", raw)
        (namespace.directory / "genesis.json").unlink()
        with pytest.raises(storage.MemoryStorageError) as raised:
            manifest.load()
        assert raised.value.code == "missing"
        assert manifest.create(captured).raw_bytes == raw
        assert len(manifest.load()) == 1
    finally:
        _close(authority, namespace)


def test_append_repairs_pending_candidate_and_maps_pending_conflict(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        first = manifest.create(captured)
        payload = {"step": 1}
        raw = _record_bytes(
            kind="transactions",
            identity=namespace.identity,
            sequence=1,
            predecessor_digest=first.digest,
            phase="PREPARED",
            payload=payload,
        )
        slot = "next-" + first.digest + ".json"
        namespace.publish_record(slot, raw)
        (namespace.directory / slot).unlink()
        with pytest.raises(storage.MemoryStorageError) as raised:
            manifest.load()
        assert raised.value.code == "missing"
        assert manifest.append(
            predecessor_digest=first.digest,
            phase="PREPARED",
            payload=payload,
        ).raw_bytes == raw

        other = {"step": 2}
        second = manifest.load()[-1]
        conflict_raw = _record_bytes(
            kind="transactions",
            identity=namespace.identity,
            sequence=2,
            predecessor_digest=second.digest,
            phase="HISTORY_APPLIED",
            payload=other,
        )
        conflict_slot = "next-" + second.digest + ".json"
        namespace.publish_record(conflict_slot, conflict_raw)
        (namespace.directory / conflict_slot).unlink()
        with pytest.raises(MemoryManifestError) as conflict:
            manifest.append(
                predecessor_digest=second.digest,
                phase="HISTORY_APPLIED",
                payload={"step": 3},
            )
        assert conflict.value.code == "conflict"
    finally:
        _close(authority, namespace)


def test_append_does_not_tolerate_pending_companion_at_reachable_non_candidate(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        first = manifest.create(captured)
        second = manifest.append(
            predecessor_digest=first.digest,
            phase="PREPARED",
            payload={"step": 1},
        )
        raw = namespace.read_record("genesis.json")
        assert raw is not None
        (namespace.directory / "genesis.json").unlink()
        with pytest.raises(storage.MemoryStorageError) as raised:
            manifest.append(
                predecessor_digest=second.digest,
                phase="HISTORY_APPLIED",
                payload={"step": 2},
            )
        assert raised.value.code == "missing"
    finally:
        _close(authority, namespace)


def test_terminal_successor_pending_companion_propagates_storage_missing(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "conversions", captured)
    try:
        record = manifest.create(captured)
        for phase in (
            "CONVERSION_PREPARED",
            "JSONL_APPLIED",
            "CURSORS_APPLIED",
            "LEGACY_FINALIZED",
            "COMPLETE",
        ):
            record = manifest.append(
                predecessor_digest=record.digest,
                phase=phase,
                payload={"phase": phase},
            )
        slot = "next-" + record.digest + ".json"
        namespace.publish_record(slot, b"pending terminal successor")
        (namespace.directory / slot).unlink()
        with pytest.raises(storage.MemoryStorageError) as raised:
            manifest.load()
        assert raised.value.code == "missing"
    finally:
        _close(authority, namespace)


def test_pending_companion_elsewhere_is_not_tolerated_by_create(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        first = manifest.create(captured)
        raw = _record_bytes(
            kind="transactions",
            identity=namespace.identity,
            sequence=1,
            predecessor_digest=first.digest,
            phase="PREPARED",
            payload={},
        )
        slot = "next-" + first.digest + ".json"
        namespace.publish_record(slot, raw)
        (namespace.directory / slot).unlink()
        with pytest.raises(storage.MemoryStorageError) as raised:
            manifest.create(captured)
        assert raised.value.code == "missing"
    finally:
        _close(authority, namespace)


def test_namespace_mutation_and_lifecycle_are_rejected_or_propagated(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        namespace.kind = "conversions"  # type: ignore[misc]
        with pytest.raises(MemoryManifestError) as raised:
            manifest.load()
        assert raised.value.code == "invalid"
        namespace.kind = "transactions"  # type: ignore[misc]
        namespace.close()
        with pytest.raises(storage.MemoryStorageError) as raised:
            manifest.load()
        assert raised.value.code == "closed"
    finally:
        authority.close()


@pytest.mark.parametrize(
    "error_factory",
    [
        storage.MemoryStorageDurabilityError,
        storage.MemoryStorageBindingError,
        storage.MemoryStorageMissingError,
        storage.MemoryStorageUnavailableError,
        lambda: storage.MemoryStorageError("closed"),
    ],
)
def test_non_conflict_storage_errors_propagate_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_factory: Any,
) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        expected = error_factory()

        def read_record(slot: str) -> bytes | None:
            raise expected

        monkeypatch.setattr(namespace, "read_record", read_record)
        with pytest.raises(type(expected)) as raised:
            manifest.load()
        assert raised.value is expected
        assert raised.value.code == expected.code
    finally:
        _close(authority, namespace)


def test_only_storage_conflict_maps_to_manifest_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    try:
        def read_record(slot: str) -> bytes | None:
            raise storage.MemoryStorageConflictError()

        monkeypatch.setattr(namespace, "read_record", read_record)
        with pytest.raises(MemoryManifestError) as raised:
            manifest.load()
        assert raised.value.code == "conflict"
        assert str(raised.value) == "memory recovery manifest evidence conflicts"
    finally:
        _close(authority, namespace)


@pytest.mark.skipif(not _fork_capable(), reason="manifest competition test requires multiprocessing fork")
def test_two_process_appenders_are_bounded_and_arbitrate_conflict(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    namespace = authority.open_evidence_namespace(
        "transactions", derive_manifest_identity("transactions", captured)
    )
    manifest = MemoryRecoveryManifest(namespace)
    first = manifest.create(captured)
    identity = namespace.identity
    namespace.close()
    authority.close()

    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    result = context.Queue()
    processes = [
        context.Process(
            target=_append_child,
            args=(
                str(tmp_path),
                "transactions",
                identity,
                first.digest,
                {"step": step},
                barrier,
                result,
            ),
        )
        for step in (1, 2)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=5)
        assert all(not process.is_alive() for process in processes)
        results = [result.get(timeout=2) for _ in processes]
        assert sorted(item[0] for item in results) == ["error", "ok"]
        assert all(
            item[2] == "memory recovery manifest evidence conflicts"
            for item in results
            if item[0] == "error"
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        assert len(MemoryRecoveryManifest(namespace).load()) == 2


@pytest.mark.skipif(not _fork_capable(), reason="inherited-process test requires multiprocessing fork")
def test_namespace_and_authority_reject_inherited_process_use(tmp_path: Path) -> None:
    captured: dict[str, object] = {"captured": True}
    authority, namespace, manifest = _open(tmp_path, "transactions", captured)
    context = multiprocessing.get_context("fork")
    result = context.Queue()
    process = context.Process(
        target=_inherited_process_child,
        args=(authority, manifest, "transactions", namespace.identity, result),
    )
    process.start()
    try:
        process.join(timeout=5)
        assert not process.is_alive()
        manifest_result, authority_result = result.get(timeout=2)
        assert manifest_result[0] == "MemoryStorageError"
        assert manifest_result[1] == "closed"
        assert authority_result[0] == "MemoryStorageError"
        assert authority_result[1] == "closed"
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        _close(authority, namespace)


def test_max_chain_records_is_public() -> None:
    assert MAX_CHAIN_RECORDS == 7
