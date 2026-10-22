"""Explicit offline application of frozen v2 receipt-tail conversion targets."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import NoReturn, cast

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_conversion as conversion
from nanobot.agent import memory_recovery_conversion_runtime as preparation
from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_target_mutation as mutation
from nanobot.agent import memory_recovery_target_verify as verification
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent.memory_recovery_manifest import ManifestRecord, MemoryRecoveryManifest

_TARGETS = ("history.jsonl", ".cursor", ".dream_cursor")
_RECEIPT = ".consolidation.pending.json"


class ConversionApplyError(RuntimeError):
    """Fixed, content-free failure at the offline application boundary."""

    def __init__(self) -> None:
        super().__init__("memory recovery conversion application is unavailable")


def _fail() -> NoReturn:
    raise ConversionApplyError() from None


def _chain(
    authority: storage.WorkspaceMemoryAuthority, kind: str, identity: str
) -> MemoryRecoveryManifest:
    return MemoryRecoveryManifest(authority.open_evidence_namespace(kind, identity))


def _old_bytes(
    authority: storage.WorkspaceMemoryAuthority, descriptor: conversion.OldFileDescriptor
) -> bytes | None:
    if not descriptor.exists:
        return None
    content = authority.read_blob(descriptor.blob_digest or "")
    if len(content) != descriptor.size or hashlib.sha256(content).hexdigest() != descriptor.digest:
        _fail()
    return content


def _descriptor_matches(
    snapshot: storage.MemoryTargetSnapshot,
    old: conversion.OldFileDescriptor,
    frozen: bytes | None,
) -> bool:
    info = snapshot.stat_identity
    expected = old.stat
    return (
        snapshot.exists == old.exists
        and snapshot.content == frozen
        and (
            (info is None and expected is None)
            or (
                info is not None
                and expected is not None
                and {
                    "device": info.device, "inode": info.inode, "mode": info.mode,
                    "uid": info.uid, "gid": info.gid, "links": info.links,
                    "size": info.size, "mtime_ns": info.mtime_ns, "ctime_ns": info.ctime_ns,
                } == expected.to_payload()
            )
        )
    )


def _output_bytes(
    authority: storage.WorkspaceMemoryAuthority, descriptor: conversion.NewFileDescriptor
) -> bytes | None:
    if not descriptor.exists:
        return None
    content = authority.read_blob(descriptor.blob_digest or "")
    if len(content) != descriptor.size or hashlib.sha256(content).hexdigest() != descriptor.blob_digest:
        _fail()
    return content


def _status(
    snapshot: storage.MemoryTargetSnapshot,
    old: conversion.OldFileDescriptor,
    frozen: bytes | None,
    output: bytes | None,
) -> str:
    if _descriptor_matches(snapshot, old, frozen):
        return "old"
    # A byte-identical output is an authorized no-op only at the original stat identity.
    if output == frozen or snapshot.content != output or output is None:
        _fail()
    info = snapshot.stat_identity
    old_info = old.stat
    if (
        info is None
        or not stat.S_ISREG(info.mode)
        or stat.S_IMODE(info.mode) != 0o600
        or info.uid != os.geteuid()
        or info.links != 1
        or (old_info is not None and (info.device, info.inode) == (old_info.device, old_info.inode))
    ):
        _fail()
    return "output"


def require_conversion_receipt_binding(
    authority: storage.WorkspaceMemoryAuthority,
    receipt_snapshot: storage.MemoryTargetSnapshot | quarantine.ValidatedQuarantineCopy,
) -> None:
    """Check the conversion receipt snapshot or current quarantine-copy binding."""

    _bound_receipt(authority, receipt_snapshot)


def _bound_receipt(
    authority: storage.WorkspaceMemoryAuthority,
    receipt_snapshot: storage.MemoryTargetSnapshot | quarantine.ValidatedQuarantineCopy,
) -> None:
    if type(receipt_snapshot) is quarantine.ValidatedQuarantineCopy:
        receipt_snapshot.require_current()
        if authority.snapshot_target(_RECEIPT).exists:
            _fail()
        return
    if type(receipt_snapshot) is not storage.MemoryTargetSnapshot:
        _fail()
    current = authority.snapshot_target(_RECEIPT)
    if (
        current.content != receipt_snapshot.content
        or current.stat_identity != receipt_snapshot.stat_identity
    ):
        _fail()


def _receipt_bytes(
    evidence: storage.MemoryTargetSnapshot | quarantine.ValidatedQuarantineCopy,
) -> bytes:
    if type(evidence) is quarantine.ValidatedQuarantineCopy:
        return evidence.raw_bytes
    if type(evidence) is not storage.MemoryTargetSnapshot or evidence.name != _RECEIPT or evidence.content is None:
        _fail()
    return evidence.content


def _receipt_stat(
    evidence: storage.MemoryTargetSnapshot | quarantine.ValidatedQuarantineCopy,
) -> dict[str, int | None]:
    if type(evidence) is quarantine.ValidatedQuarantineCopy:
        values = evidence.original_stat
        if len(values) != 9:
            _fail()
        return dict(zip(
            ("device", "inode", "mode", "uid", "gid", "links", "size", "mtime_ns", "ctime_ns"),
            values, strict=True,
        ))
    from nanobot.agent import memory_recovery_transaction_prepare as preparation
    # Supported consumers validate exact snapshot/copy evidence before projection.
    # The cast is inert: private fallback calls retain their original delegation.
    return preparation.receipt_snapshot_stat(cast(storage.MemoryTargetSnapshot, evidence))


def receipt_evidence_bytes(
    evidence: storage.MemoryTargetSnapshot | quarantine.ValidatedQuarantineCopy,
) -> bytes:
    """Project existing receipt evidence; this acquires no new authority."""
    return _receipt_bytes(evidence)


def receipt_evidence_stat(
    evidence: storage.MemoryTargetSnapshot | quarantine.ValidatedQuarantineCopy,
) -> dict[str, int | None]:
    """Project existing receipt evidence; this acquires no new authority."""
    return _receipt_stat(evidence)


def apply_prepared_conversion_targets(workspace: Path | str, *, workspace_id: str) -> ManifestRecord:
    """Verify frozen evidence and apply only the three conversion targets under one lock."""

    try:
        with (
            storage.open_workspace_memory(workspace) as authority,
            authority.operation() as operation,
        ):
            receipt_snapshot = authority.snapshot_target(_RECEIPT)
            if receipt_snapshot.content is None:
                _fail()
            with operation.borrow_pending_receipt_parent(receipt_snapshot) as scope:
                current = scope.snapshot_target()
                if (
                    current.content != receipt_snapshot.content
                    or current.stat_identity != receipt_snapshot.stat_identity
                ):
                    _fail()
            raw_receipt = receipt_snapshot.content
            receipt = receipt_codec.decode_memory_receipt_bytes(raw_receipt)
            captured_tx = transaction.capture_transaction_payload(
                receipt, workspace_id=workspace_id, workspace_identity=operation.workspace_identity
            )
            tx_records = _chain(authority, "transactions", captured_tx.identity).load()
            if len(tx_records) != 1 or tx_records[0].phase != "CAPTURED":
                _fail()
            tx_record = tx_records[0]
            loaded_tx = transaction.decode_transaction_payload("CAPTURED", tx_record.payload)
            if loaded_tx != captured_tx:
                _fail()
            v1 = conversion.capture_conversion_payload(captured_tx.identity)
            v2 = conversion.capture_receipt_tail_conversion_payload(captured_tx.identity)
            if _chain(authority, "conversions", v1.identity).load():
                _fail()
            manifest = _chain(authority, "conversions", v2.identity)
            records: tuple[ManifestRecord, ...] = manifest.load()
            if not 2 <= len(records) <= 4 or [record.phase for record in records] != [
                "CAPTURED", "CONVERSION_PREPARED", "JSONL_APPLIED", "CURSORS_APPLIED"
            ][: len(records)]:
                _fail()
            captured = conversion.decode_conversion_payload(records[0].payload, "CAPTURED")
            if captured != v2:
                _fail()
            prepared = conversion.decode_conversion_payload(
                records[1].payload, "CONVERSION_PREPARED", captured=v2
            )
            if prepared.captured != v2:
                _fail()
            jsonl = None
            if len(records) >= 3:
                jsonl = conversion.decode_conversion_payload(
                    records[2].payload, "JSONL_APPLIED", captured=v2, prepared=prepared
                )
            if len(records) == 4:
                conversion.decode_conversion_payload(
                    records[3].payload, "CURSORS_APPLIED", captured=v2,
                    prepared=prepared, jsonl_applied=jsonl,
                )
            if prepared.disposition == "convert":
                claim = prepared.receipt
                if (
                    claim is None
                    or prepared.transaction_captured_record_digest != tx_record.digest
                    or claim["size"] != len(raw_receipt)
                    or claim["blob_digest"] != hashlib.sha256(raw_receipt).hexdigest()
                    or authority.read_blob(claim["blob_digest"]) != raw_receipt
                ):
                    _fail()
            frozen_source = _old_bytes(authority, prepared.source)
            if frozen_source is None:
                _fail()
            frozen = {name: _old_bytes(authority, prepared.targets[name]) for name in _TARGETS}
            source_claim = prepared.source.to_payload()
            target_claims = {name: prepared.targets[name].to_payload() for name in _TARGETS}
            if prepared.disposition == "convert":
                rebuilt = conversion.prepare_conversion_payload(
                    v2, disposition="convert", source=source_claim, targets=target_claims,
                    source_bytes=frozen_source, target_contents=frozen,
                    backup=prepared.backup.to_payload() if prepared.backup is not None else None,
                    fallback_timestamp=prepared.fallback_timestamp, receipt=receipt,
                    transaction_captured=loaded_tx, transaction_captured_record=tx_record,
                )
                if prepared.outputs is None or prepared.canonical_row is None:
                    _fail()
                output = {
                    name: _output_bytes(authority, prepared.outputs[name]) for name in _TARGETS
                }
                derived = preparation.derive_conversion_output_bytes(prepared, frozen_source)
                if any(output[name] != derived[name] for name in _TARGETS):
                    _fail()
                row = derived["history.jsonl"].splitlines(keepends=True)[-1]
                canonical = prepared.canonical_row
                if (
                    canonical["size"] != len(row)
                    or canonical["blob_digest"] != hashlib.sha256(row).hexdigest()
                    or authority.read_blob(canonical["blob_digest"]) != row
                ):
                    _fail()
            elif prepared.disposition == "preserve_nonempty_jsonl":
                rebuilt = conversion.prepare_conversion_payload(
                    v2, disposition="preserve_nonempty_jsonl", source=source_claim,
                    targets=target_claims, source_bytes=None,
                )
                output = dict(frozen)
            else:
                _fail()
            # load supplies ManifestRecords; the unchanged 2–4 guard proves index 1 safe.
            if rebuilt.to_payload() != cast(tuple[ManifestRecord, ...], records)[1].payload:
                _fail()

            def observe() -> tuple[dict[str, storage.MemoryTargetSnapshot], dict[str, str]]:
                _bound_receipt(authority, receipt_snapshot)
                if not _descriptor_matches(
                    authority.snapshot_target("HISTORY.md"), prepared.source, frozen_source
                ):
                    _fail()
                snapshots = {name: authority.snapshot_target(name) for name in _TARGETS}
                states = {
                    name: _status(snapshots[name], prepared.targets[name], frozen[name], output[name])
                    for name in _TARGETS
                }
                # An unchanged byte-identical output is already satisfied at its old
                # stat identity, so later targets may have advanced past that no-op.
                if (
                    states["history.jsonl"] == "old"
                    and output["history.jsonl"] != frozen["history.jsonl"]
                    and (states[".cursor"] == "output" or states[".dream_cursor"] == "output")
                ) or (
                    states[".cursor"] == "old"
                    and output[".cursor"] != frozen[".cursor"]
                    and states[".dream_cursor"] == "output"
                ):
                    _fail()
                if len(records) >= 3 and states["history.jsonl"] == "old" and output["history.jsonl"] != frozen["history.jsonl"]:
                    _fail()
                if len(records) == 4 and any(
                    states[name] == "old" and output[name] != frozen[name] for name in _TARGETS
                ):
                    _fail()
                return snapshots, states

            def ensure(name: str) -> None:
                snapshots, states = observe()
                if states[name] == "old" and output[name] != frozen[name]:
                    expected = output[name]
                    if expected is None:
                        _fail()
                    try:
                        mutation.replace_target_if_unchanged(operation, snapshots[name], expected)
                    except mutation.MemoryTargetReplacementUncertainError:
                        refreshed, statuses = observe()
                        if statuses[name] == "old":
                            _fail()
                        snapshots, states = refreshed, statuses
                    else:
                        snapshots, states = observe()
                if output[name] is not None:
                    verification.durably_verify_target(operation, snapshots[name], output[name])
                observe()

            observe()
            if len(records) == 4:
                for name in _TARGETS:
                    ensure(name)
                return records[3]
            ensure("history.jsonl")
            if len(records) == 2:
                jsonl = conversion.jsonl_applied_conversion_payload(v2, prepared)
                observe()
                record = manifest.append(
                    predecessor_digest=records[1].digest,
                    phase="JSONL_APPLIED", payload=jsonl.to_payload(),
                )
                records = (*records, record)
            if jsonl is None:
                _fail()
            for name in (".cursor", ".dream_cursor"):
                ensure(name)
            phase = conversion.cursors_applied_conversion_payload(v2, prepared, jsonl)
            observe()
            return manifest.append(
                # load/append supply ManifestRecords; length 4 returned and length 2
                # appended once, so the unchanged paths prove index 2 safe.
                predecessor_digest=cast(tuple[ManifestRecord, ...], records)[2].digest,
                phase="CURSORS_APPLIED", payload=phase.to_payload(),
            )
    except ConversionApplyError:
        raise
    except Exception:
        _fail()


def open_conversion_manifest(
    authority: storage.WorkspaceMemoryAuthority, kind: str, identity: str
) -> MemoryRecoveryManifest:
    """Open the existing owner-bound recovery evidence namespace."""
    return _chain(authority, kind, identity)


def read_conversion_old_bytes(
    authority: storage.WorkspaceMemoryAuthority, descriptor: conversion.OldFileDescriptor
) -> bytes | None:
    """Read and check frozen original bytes through the held authority."""
    return _old_bytes(authority, descriptor)


def conversion_snapshot_matches_descriptor(
    snapshot: storage.MemoryTargetSnapshot,
    old: conversion.OldFileDescriptor,
    frozen: bytes | None,
) -> bool:
    """Compare a conversion snapshot with its frozen descriptor and bytes."""
    return _descriptor_matches(snapshot, old, frozen)


def read_conversion_output_bytes(
    authority: storage.WorkspaceMemoryAuthority, descriptor: conversion.NewFileDescriptor
) -> bytes | None:
    """Read and check a conversion output blob through the held authority."""
    return _output_bytes(authority, descriptor)


def classify_conversion_target(
    snapshot: storage.MemoryTargetSnapshot,
    old: conversion.OldFileDescriptor,
    frozen: bytes | None,
    output: bytes | None,
) -> str:
    """Require an original or authorized-output conversion target identity."""
    return _status(snapshot, old, frozen, output)


__all__ = [
    "ConversionApplyError",
    "apply_prepared_conversion_targets",
    "require_conversion_receipt_binding",
    "open_conversion_manifest",
    "read_conversion_old_bytes",
    "conversion_snapshot_matches_descriptor",
    "read_conversion_output_bytes",
    "classify_conversion_target",
]
