"""Shared recovery evidence interfaces delegate through real bound recovery state."""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_conversion as conversion
from nanobot.agent import memory_recovery_conversion_apply as application
from nanobot.agent import memory_recovery_conversion_finalize as finalization
from nanobot.agent import memory_recovery_conversion_runtime as conversion_runtime
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent import memory_recovery_transaction_history as history
from nanobot.agent import memory_recovery_transaction_memory as memory_stage
from nanobot.agent import memory_recovery_transaction_prepare as preparation
from nanobot.agent.memory_recovery_manifest import MemoryRecoveryManifest
from nanobot.session.manager import JsonlSessionStore
from tests.agent.test_memory_recovery_transaction_prepare import (
    _SESSION_KEY,
    _WORKSPACE_ID,
    _prepare,
    _workspace,
)

_INTERFACES = (
    (application, "open_conversion_manifest", "_chain", application.ConversionApplyError),
    (application, "read_conversion_old_bytes", "_old_bytes", application.ConversionApplyError),
    (application, "conversion_snapshot_matches_descriptor", "_descriptor_matches", application.ConversionApplyError),
    (application, "read_conversion_output_bytes", "_output_bytes", application.ConversionApplyError),
    (application, "classify_conversion_target", "_status", application.ConversionApplyError),
    (conversion_runtime, "derive_conversion_output_bytes", "_output_bytes", conversion_runtime.ConversionPreparationError),
    (finalization, "decode_conversion_finalization_evidence", "_decode", finalization.ConversionFinalizationError),
    (finalization, "require_conversion_finalization_binding", "_verify", finalization.ConversionFinalizationError),
    (preparation, "require_transaction_conversion_evidence", "_conversion_evidence", preparation.MemoryRecoveryTransactionPreparationError),
    (preparation, "read_verified_transaction_marker_row", "_verified_marker_row", preparation.MemoryRecoveryTransactionPreparationError),
    (preparation, "parse_recovery_session_records", "_session_records", preparation.MemoryRecoveryTransactionPreparationError),
    (preparation, "derive_recovery_source_proposal", "_source_proposal", preparation.MemoryRecoveryTransactionPreparationError),
    (preparation, "transaction_target_descriptor", "_target_descriptor", preparation.MemoryRecoveryTransactionPreparationError),
    (preparation, "require_current_transaction_snapshot", "_same_snapshot", preparation.MemoryRecoveryTransactionPreparationError),
    (history, "require_transaction_target_blob", "_check_new_target_blob", history.MemoryRecoveryTransactionHistoryError),
    (memory_stage, "transaction_snapshots_match", "_same_snapshot", memory_stage.MemoryRecoveryTransactionMemoryError),
    (memory_stage, "transaction_snapshot_matches_old", "_snapshot_matches_old", memory_stage.MemoryRecoveryTransactionMemoryError),
    (memory_stage, "transaction_snapshot_is_authorized_new", "_authorized_new_snapshot", memory_stage.MemoryRecoveryTransactionMemoryError),
    (memory_stage, "durably_verify_transaction_memory", "_durably_verify_memory", memory_stage.MemoryRecoveryTransactionMemoryError),
    (memory_stage, "require_transaction_memory_blob", "_check_memory_blob", memory_stage.MemoryRecoveryTransactionMemoryError),
)


def _live(paths):
    snapshots = {}
    for path in paths:
        before = storage.memory_target_stat_identity(path.lstat())
        content = path.read_bytes()
        after = storage.memory_target_stat_identity(path.lstat())
        assert after == before, f"{path} changed while evidence snapshot was read"
        snapshots[path] = (content, after)
    return snapshots


@pytest.fixture
def _evidence(tmp_path: Path):
    workspace, sessions_root, memory, receipt_path = _workspace(
        tmp_path, memory_output="updated memory\n",
    )
    _prepare(workspace, sessions_root)
    history.reconcile_memory_recovery_transaction_history(
        workspace, workspace_id=_WORKSPACE_ID, sessions_root=sessions_root,
    )
    memory_stage.reconcile_memory_recovery_transaction_memory(
        workspace, workspace_id=_WORKSPACE_ID, sessions_root=sessions_root,
    )
    session_path = (sessions_root / _WORKSPACE_ID
                    / f"{JsonlSessionStore.storage_key(_SESSION_KEY)}.jsonl")
    source = session_path.read_bytes()
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        snapshot = authority.snapshot_target(".consolidation.pending.json")
        assert snapshot.content is not None
        receipt = receipt_codec.decode_memory_receipt_bytes(snapshot.content)
        captured = transaction.capture_transaction_payload(
            receipt, workspace_id=_WORKSPACE_ID, workspace_identity=operation.workspace_identity,
        )
        tx_records = MemoryRecoveryManifest(
            authority.open_evidence_namespace("transactions", captured.identity),
        ).load()
        assert tuple(record.phase for record in tx_records) == (
            "CAPTURED", "PREPARED", "HISTORY_APPLIED", "MEMORY_APPLIED",
        )
        prepared = transaction.decode_transaction_payload(
            "PREPARED", tx_records[1].payload, captured=captured,
        )
        conversion_captured = conversion.capture_receipt_tail_conversion_payload(captured.identity)
        conversion_records = MemoryRecoveryManifest(
            authority.open_evidence_namespace("conversions", conversion_captured.identity),
        ).load()
        decoded = finalization._decode(conversion_records, conversion_captured)
        converted = decoded[0]
        assert converted.outputs is not None and converted.source.blob_digest is not None
        frozen_source = authority.read_blob(converted.source.blob_digest)
        derived = conversion_runtime._output_bytes(converted, frozen_source)
        current = authority.snapshot_target("MEMORY.md")
        assert current.content == b"updated memory\n" and current.stat_identity is not None
        history_bytes = authority.snapshot_target("history.jsonl").content
        cursor = authority.snapshot_target(".cursor")
        assert history_bytes is not None and cursor.content is not None
        marker_row = preparation._verified_marker_row(
            authority, converted, receipt, history_bytes, cursor.content,
        )
        header, events = preparation._session_records(source, _SESSION_KEY)
        proposal = preparation._source_proposal(source, receipt)
        assert header["last_archived"] == 1 and len(events) == 2
        assert proposal[0]["action"] == "advance" and proposal[0]["new_cursor"] == 2
        assert hashlib.sha256(proposal[1]).hexdigest() == prepared.source.new_digest
        absent = authority.snapshot_target("HISTORY.md")
        assert absent.content is None and absent.stat_identity is None
        old_absence = converted.targets[".cursor"]
        assert old_absence.exists is False
        paths = (receipt_path, session_path, memory / "history.jsonl", memory / ".cursor", memory / "MEMORY.md")
        recipes = {
            "open_conversion_manifest": ((authority, "conversions", conversion_captured.identity), {}, conversion_records),
            "read_conversion_old_bytes": ((authority, converted.source), {}, frozen_source),
            "conversion_snapshot_matches_descriptor": ((absent, old_absence, None), {}, True),
            "read_conversion_output_bytes": ((authority, converted.outputs["history.jsonl"]), {}, derived["history.jsonl"]),
            "classify_conversion_target": ((cursor, old_absence, None, cursor.content), {}, "output"),
            "derive_conversion_output_bytes": ((converted, frozen_source), {}, derived),
            "decode_conversion_finalization_evidence": ((conversion_records, conversion_captured), {}, decoded),
            "require_conversion_finalization_binding": ((authority, operation, snapshot, receipt, tx_records[0], captured, conversion_captured, converted), {"finalized": True, "read_only": False}, None),
            "require_transaction_conversion_evidence": ((authority, captured, tx_records[0], snapshot, receipt, operation), {"read_only": False}, converted),
            "read_verified_transaction_marker_row": ((authority, converted, receipt, history_bytes, cursor.content), {}, marker_row),
            "parse_recovery_session_records": ((source, _SESSION_KEY), {}, (header, events)),
            "derive_recovery_source_proposal": ((source, receipt), {}, proposal),
            "transaction_target_descriptor": ((current, current.content), {}, preparation._target_descriptor(current, current.content)),
            "require_current_transaction_snapshot": ((authority, current), {}, None),
            "require_transaction_target_blob": ((authority, prepared.targets["MEMORY.md"].new, current.content), {}, None),
            "transaction_snapshots_match": ((current, authority.snapshot_target("MEMORY.md")), {}, True),
            "transaction_snapshot_matches_old": ((current, prepared.targets["MEMORY.md"].old), {}, False),
            "transaction_snapshot_is_authorized_new": ((current, prepared.targets["MEMORY.md"].new, current.content, prepared.targets["MEMORY.md"].old), {}, True),
            "durably_verify_transaction_memory": ((operation, current, current.content), {}, current),
            "require_transaction_memory_blob": ((authority, prepared.targets["MEMORY.md"].new, current.content), {}, None),
        }
        assert len(recipes) == len(_INTERFACES) == 20
        yield {"recipes": recipes, "paths": paths, "authority": authority,
               "current": current, "memory": memory, "conversion_records": conversion_records,
               "conversion_captured": conversion_captured}


@pytest.mark.parametrize("module,public,private,error_type", _INTERFACES)
def test_supported_evidence_signatures_and_exports(module, public, private, error_type) -> None:
    operation = getattr(module, public)
    signature = inspect.signature(operation)
    assert signature.parameters == inspect.signature(getattr(module, private)).parameters
    assert signature.return_annotation is not inspect.Signature.empty
    assert public in module.__all__


@pytest.mark.parametrize("module,public,private,error_type", _INTERFACES)
def test_dynamic_evidence_forwarding_uses_real_helpers(
    _evidence, monkeypatch, module, public, private, error_type,
) -> None:
    operation = getattr(module, public)
    original = getattr(module, private)
    args, kwargs, expected = _evidence["recipes"][public]
    before = _live(_evidence["paths"])
    calls, results = [], []

    def first(*actual_args, **actual_kwargs):
        calls.append(("first", actual_args, actual_kwargs))
        result = original(*actual_args, **actual_kwargs)
        results.append(result)
        return result

    def second(*actual_args, **actual_kwargs):
        calls.append(("second", actual_args, actual_kwargs))
        result = original(*actual_args, **actual_kwargs)
        results.append(result)
        return result

    for index, helper in enumerate((first, second)):
        monkeypatch.setattr(module, private, helper)
        actual = operation(*args, **kwargs)
        assert actual is results[index]
        if public == "open_conversion_manifest":
            assert isinstance(actual, MemoryRecoveryManifest) and actual.load() == expected
        else:
            assert actual == expected
    assert calls == [("first", args, kwargs), ("second", args, kwargs)]
    assert all(all(actual is supplied for actual, supplied in zip(call[1], args, strict=True))
               for call in calls)
    assert _live(_evidence["paths"]) == before


def test_live_snapshot_ignores_access_time_only_change(
    tmp_path: Path, monkeypatch,
) -> None:
    path = tmp_path / "evidence.jsonl"
    path.write_bytes(b"evidence\n")
    access_time_ns = [0]
    read_bytes = Path.read_bytes
    lstat = Path.lstat

    def read_with_access_time_change(candidate: Path) -> bytes:
        content = read_bytes(candidate)
        if candidate == path:
            access_time_ns[0] += 1_000_000_000
        return content

    def lstat_with_access_time(candidate: Path):
        observed = lstat(candidate)
        if candidate != path:
            return observed
        return SimpleNamespace(
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_mode=observed.st_mode,
            st_nlink=observed.st_nlink,
            st_uid=observed.st_uid,
            st_gid=observed.st_gid,
            st_size=observed.st_size,
            st_atime=access_time_ns[0] / 1_000_000_000,
            st_atime_ns=access_time_ns[0],
            st_mtime=observed.st_mtime,
            st_mtime_ns=observed.st_mtime_ns,
            st_ctime=observed.st_ctime,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(Path, "read_bytes", read_with_access_time_change)
    monkeypatch.setattr(Path, "lstat", lstat_with_access_time)
    assert _live((path,)) == _live((path,))


def test_live_snapshot_detects_content_and_replacement_drift(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    path.write_bytes(b"original\n")
    before = _live((path,))

    path.write_bytes(b"modified\n")
    after_write = _live((path,))
    assert after_write[path][0] != before[path][0]

    retained = tmp_path / "retained.jsonl"
    path.rename(retained)
    path.write_bytes(before[path][0])
    after_replacement = _live((path,))
    assert after_replacement[path][0] == before[path][0]
    assert after_replacement[path][1] != before[path][1]


def test_live_snapshot_detects_nanosecond_timestamp_drift(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    path.write_bytes(b"unchanged\n")
    before = _live((path,))
    original_stat = path.lstat()
    requested_mtime_ns = original_stat.st_mtime_ns + 1_000_000_001

    os.utime(path, ns=(original_stat.st_atime_ns, requested_mtime_ns))
    changed_stat = path.lstat()
    assert changed_stat.st_mtime_ns != original_stat.st_mtime_ns

    after = _live((path,))
    assert after[path][0] == before[path][0]
    assert after[path][1] != before[path][1]


def test_live_snapshot_refuses_replacement_during_read(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "evidence.jsonl"
    path.write_bytes(b"before\n")
    retained = tmp_path / "retained.jsonl"
    read_bytes = Path.read_bytes

    def replace_during_read(candidate: Path) -> bytes:
        content = read_bytes(candidate)
        if candidate == path:
            candidate.rename(retained)
            candidate.write_bytes(b"after\n")
        return content

    monkeypatch.setattr(Path, "read_bytes", replace_during_read)
    with pytest.raises(AssertionError, match="changed while evidence snapshot was read"):
        _live((path,))


@pytest.mark.parametrize("module,public,private,error_type", _INTERFACES)
def test_evidence_forwarding_preserves_fault_identity(
    _evidence, monkeypatch, module, public, private, error_type,
) -> None:
    operation = getattr(module, public)
    args, kwargs, _ = _evidence["recipes"][public]
    before = _live(_evidence["paths"])
    for kind in (OSError, RuntimeError, KeyboardInterrupt, error_type):
        fault = kind() if kind is error_type else kind("evidence probe")
        calls = []

        def fail(*actual_args, **actual_kwargs):
            calls.append((actual_args, actual_kwargs))
            raise fault

        monkeypatch.setattr(module, private, fail)
        with pytest.raises(kind) as raised:
            operation(*args, **kwargs)
        assert raised.value is fault and calls == [(args, kwargs)]
        assert all(actual is supplied for actual, supplied in zip(calls[0][0], args, strict=True))
    assert _live(_evidence["paths"]) == before


def test_current_transaction_snapshot_refuses_real_drift(_evidence) -> None:
    operation = getattr(preparation, "require_current_transaction_snapshot")
    (_evidence["memory"] / "MEMORY.md").write_bytes(b"drifted memory\n")
    with pytest.raises(preparation.MemoryRecoveryTransactionPreparationError) as raised:
        operation(_evidence["authority"], _evidence["current"])
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("module,public,error_type", [
    (history, "require_transaction_target_blob", history.MemoryRecoveryTransactionHistoryError),
    (memory_stage, "require_transaction_memory_blob", memory_stage.MemoryRecoveryTransactionMemoryError),
])
def test_transaction_blob_interfaces_refuse_wrong_bytes(_evidence, module, public, error_type) -> None:
    operation = getattr(module, public)
    args, kwargs, _ = _evidence["recipes"][public]
    assert not kwargs
    with pytest.raises(error_type) as raised:
        operation(args[0], args[1], b"wrong bytes")
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("raw", [b"", b'{"_type":"not-metadata"}\n'])
def test_recovery_session_parser_refuses_malformed_data(raw) -> None:
    operation = getattr(preparation, "parse_recovery_session_records")
    with pytest.raises(preparation.MemoryRecoveryTransactionPreparationError):
        operation(raw, _SESSION_KEY)


def test_conversion_evidence_decoder_refuses_incomplete_chain(_evidence) -> None:
    operation = getattr(finalization, "decode_conversion_finalization_evidence")
    with pytest.raises(finalization.ConversionFinalizationError):
        operation(_evidence["conversion_records"][:3], _evidence["conversion_captured"])
