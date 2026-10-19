"""Finalization claims follow durable targets and a frozen legacy archive."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_conversion_apply as apply
from nanobot.agent import memory_recovery_conversion_finalize as finalize
from nanobot.agent import memory_recovery_conversion_runtime as prepare
from nanobot.agent.memory_recovery_manifest import MemoryRecoveryManifest

_WORKSPACE_ID = "a" * 32


def test_finalize_retains_none_receipt_private_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _workspace(tmp_path)
    calls = []

    def missing(evidence):
        calls.append(evidence)
        return None

    monkeypatch.setattr(apply, "_receipt_bytes", missing)
    with pytest.raises(finalize.ConversionFinalizationError):
        _finalize(tmp_path)
    assert len(calls) == 1
    assert (memory / "HISTORY.md").exists()
    assert not (memory / "HISTORY.md.bak").exists()


def _workspace(tmp_path: Path, *, jsonl: bytes | None = None) -> Path:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "HISTORY.md").write_bytes(b"# Existing\n\n")
    if jsonl is not None:
        (memory / "history.jsonl").write_bytes(jsonl)
    receipt = {
        "version": 1,
        "chunk_digest": hashlib.sha256(b"chunk").hexdigest(),
        "history_marker_id": "receipt-marker",
        "history_entry": "[2026-09-25 12:00] receipt text",
        "memory_output": None,
        "base_memory_digest": hashlib.sha256(b"base").hexdigest(),
        "output_memory_digest": None,
        "reference_only": True,
        "created_at": "2026-09-25T00:00:00Z",
        "session_key": "session",
        "start_offset": 1,
        "end_offset": 2,
        "session_generation": 1,
    }
    path = memory / ".consolidation.pending.json"
    path.write_bytes(json.dumps(receipt, sort_keys=True).encode())
    path.chmod(0o600)
    prepare.prepare_receipt_tail_conversion(tmp_path, workspace_id=_WORKSPACE_ID)
    apply.apply_prepared_conversion_targets(tmp_path, workspace_id=_WORKSPACE_ID)
    return memory


def _finalize(tmp_path: Path):
    return finalize.finalize_prepared_conversion(tmp_path, workspace_id=_WORKSPACE_ID)


def _phases(tmp_path: Path) -> list[str]:
    from nanobot.agent import memory_recovery as codec
    from nanobot.agent import memory_recovery_conversion as conversion
    from nanobot.agent import memory_recovery_storage as storage
    from nanobot.agent import memory_recovery_transaction as transaction

    with storage.open_workspace_memory(tmp_path) as authority, authority.operation() as operation:
        raw = authority.snapshot_target(".consolidation.pending.json").content
        receipt = codec._validate_payload(codec._decode(raw), raw)
        tx = transaction.capture_transaction_payload(
            receipt, workspace_id=_WORKSPACE_ID, workspace_identity=operation.workspace_identity,
        )
        captured = conversion.capture_receipt_tail_conversion_payload(tx.identity)
        return [record.phase for record in MemoryRecoveryManifest(
            authority.open_evidence_namespace("conversions", captured.identity)
        ).load()]


@pytest.mark.parametrize("jsonl", [None, b'{"cursor":999}\n'])
def test_complete_and_idempotent(tmp_path: Path, jsonl: bytes | None) -> None:
    memory = _workspace(tmp_path, jsonl=jsonl)
    original = (memory / "HISTORY.md").read_bytes()
    original_stat = (memory / "HISTORY.md").stat()
    receipt = (memory / ".consolidation.pending.json").read_bytes()
    result = _finalize(tmp_path)
    assert result.phase == "COMPLETE"
    assert _finalize(tmp_path).digest == result.digest
    assert _phases(tmp_path)[-2:] == ["LEGACY_FINALIZED", "COMPLETE"]
    assert (memory / ".consolidation.pending.json").read_bytes() == receipt
    if jsonl is None:
        assert not (memory / "HISTORY.md").exists()
        assert (memory / "HISTORY.md.bak").read_bytes() == original
        assert not (memory / "HISTORY.md.bak").is_symlink()
    else:
        assert not (memory / "HISTORY.md.bak").exists()
        assert (memory / "HISTORY.md").stat() == original_stat
        assert (memory / "HISTORY.md").read_bytes() == original


def test_preexisting_backup_collision(tmp_path: Path) -> None:
    memory = _workspace(tmp_path)
    (memory / "HISTORY.md.bak").write_bytes(b"collision")
    with pytest.raises(finalize.ConversionFinalizationError):
        _finalize(tmp_path)
    assert _phases(tmp_path)[-1] == "CURSORS_APPLIED"
    assert (memory / "HISTORY.md").exists()


@pytest.mark.parametrize("step", ["link", "unlink"])
def test_archive_uncertain_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    memory = _workspace(tmp_path)
    source = memory / "HISTORY.md"
    backup = memory / "HISTORY.md.bak"
    original_read = finalize.archive._read_entry
    injected = False

    def crash_after_real_mutation(*args, **kwargs):
        nonlocal injected
        result = original_read(*args, **kwargs)
        linked = (
            step == "link" and args[1] == "HISTORY.md"
            and source.exists() and backup.exists() and source.samefile(backup)
            and source.stat().st_nlink == 2
        )
        unlinked = (
            step == "unlink" and args[1] == "HISTORY.md.bak"
            and not source.exists() and backup.exists() and backup.stat().st_nlink == 1
        )
        if not injected and (linked or unlinked):
            injected = True
            raise OSError("interrupted archive")
        return result

    monkeypatch.setattr(finalize.archive, "_read_entry", crash_after_real_mutation)
    with pytest.raises(finalize.ConversionFinalizationError):
        _finalize(tmp_path)
    assert injected  # The actual archive helper crossed os.link/os.unlink.
    assert _phases(tmp_path)[-1] == "CURSORS_APPLIED"
    if step == "link":
        assert source.samefile(backup)
        assert source.stat().st_nlink == backup.stat().st_nlink == 2
    else:
        assert not source.exists()
        assert backup.stat().st_nlink == 1
    monkeypatch.setattr(finalize.archive, "_read_entry", original_read)
    assert _finalize(tmp_path).phase == "COMPLETE"
    assert not source.exists()
    assert backup.stat().st_nlink == 1
    assert _phases(tmp_path)[-2:] == ["LEGACY_FINALIZED", "COMPLETE"]


def test_existing_two_link_archive_replay(tmp_path: Path) -> None:
    memory = _workspace(tmp_path)
    source = memory / "HISTORY.md"
    backup = memory / "HISTORY.md.bak"
    os.link(source, backup)
    assert source.samefile(backup)
    assert source.stat().st_nlink == 2
    assert _phases(tmp_path)[-1] == "CURSORS_APPLIED"
    assert _finalize(tmp_path).phase == "COMPLETE"
    assert not source.exists()
    assert backup.stat().st_nlink == 1


@pytest.mark.parametrize("phase,after", [
    ("LEGACY_FINALIZED", False), ("LEGACY_FINALIZED", True),
    ("COMPLETE", False), ("COMPLETE", True),
])
def test_phase_append_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, after: bool,
) -> None:
    memory = _workspace(tmp_path)
    original = MemoryRecoveryManifest.append

    def crash(self, **kwargs):
        if kwargs["phase"] == phase and not after:
            raise RuntimeError("crash")
        result = original(self, **kwargs)
        if kwargs["phase"] == phase:
            raise RuntimeError("crash")
        return result

    monkeypatch.setattr(MemoryRecoveryManifest, "append", crash)
    with pytest.raises(finalize.ConversionFinalizationError):
        _finalize(tmp_path)
    phases = _phases(tmp_path)
    assert phases[-1] == (phase if after else (
        "CURSORS_APPLIED" if phase == "LEGACY_FINALIZED" else "LEGACY_FINALIZED"
    ))
    assert not (memory / "HISTORY.md").exists()
    monkeypatch.setattr(MemoryRecoveryManifest, "append", original)
    assert _finalize(tmp_path).phase == "COMPLETE"


@pytest.mark.parametrize("drift", ["receipt", "output", "backup", "copied"])
def test_drift_fails_closed(tmp_path: Path, drift: str) -> None:
    memory = _workspace(tmp_path, jsonl=b'{"cursor":999}\n' if drift == "copied" else None)
    if drift == "receipt":
        target = memory / ".consolidation.pending.json"
        target.write_bytes(b"tampered")
    elif drift == "output":
        (memory / "history.jsonl").write_bytes(b"tampered")
    elif drift == "backup":
        (memory / "HISTORY.md.bak").write_bytes(b"tampered")
    else:
        target = memory / "history.jsonl"
        data = target.read_bytes()
        target.unlink()
        target.write_bytes(data)
        target.chmod(0o600)
    with pytest.raises(finalize.ConversionFinalizationError):
        _finalize(tmp_path)
    if drift != "receipt":
        assert _phases(tmp_path)[-1] == "CURSORS_APPLIED"


def test_tampered_backup_after_completion(tmp_path: Path) -> None:
    memory = _workspace(tmp_path)
    _finalize(tmp_path)
    (memory / "HISTORY.md.bak").write_bytes(b"tampered")
    with pytest.raises(finalize.ConversionFinalizationError):
        _finalize(tmp_path)
    assert _phases(tmp_path)[-1] == "COMPLETE"


def test_receipt_replacement_same_bytes_replays_by_content(tmp_path: Path) -> None:
    memory = _workspace(tmp_path)
    target = memory / ".consolidation.pending.json"
    content = target.read_bytes()
    target.unlink()
    target.write_bytes(content)
    target.chmod(0o600)
    result = _finalize(tmp_path)
    assert result.phase == "COMPLETE"
    assert _finalize(tmp_path).digest == result.digest
    assert (memory / "history.jsonl").read_bytes().count(b"receipt text") == 1
    assert (memory / "HISTORY.md.bak").exists()
    assert not (memory / "HISTORY.md.bak.2").exists()
    assert target.read_bytes() == content
    # Conversion completion does not advance the independent receipt transaction.
    from nanobot.agent import memory_recovery as codec
    from nanobot.agent import memory_recovery_storage as storage
    from nanobot.agent import memory_recovery_transaction as transaction

    with storage.open_workspace_memory(tmp_path) as authority, authority.operation() as operation:
        receipt = codec._validate_payload(codec._decode(content), content)
        tx = transaction.capture_transaction_payload(
            receipt, workspace_id=_WORKSPACE_ID, workspace_identity=operation.workspace_identity,
        )
        phases = [record.phase for record in MemoryRecoveryManifest(
            authority.open_evidence_namespace("transactions", tx.identity)
        ).load()]
    assert phases == ["CAPTURED"]


def test_receipt_replacement_within_call_blocks_phase_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _workspace(tmp_path)
    target = memory / ".consolidation.pending.json"
    original = finalize.archive.archive_legacy_history_if_unchanged

    def replace(*args, **kwargs):
        result = original(*args, **kwargs)
        content = target.read_bytes()
        target.unlink()
        target.write_bytes(content)
        target.chmod(0o600)
        return result

    monkeypatch.setattr(finalize.archive, "archive_legacy_history_if_unchanged", replace)
    with pytest.raises(finalize.ConversionFinalizationError):
        _finalize(tmp_path)
    assert _phases(tmp_path)[-1] == "CURSORS_APPLIED"
    assert not (memory / "HISTORY.md").exists()
    monkeypatch.setattr(finalize.archive, "archive_legacy_history_if_unchanged", original)
    assert _finalize(tmp_path).phase == "COMPLETE"


def test_source_change_conflicts_before_archive(tmp_path: Path) -> None:
    memory = _workspace(tmp_path)
    (memory / "HISTORY.md").write_bytes(b"changed")
    with pytest.raises(finalize.ConversionFinalizationError):
        _finalize(tmp_path)
    assert _phases(tmp_path)[-1] == "CURSORS_APPLIED"
    assert not (memory / "HISTORY.md.bak").exists()


def test_target_drift_after_finalized_append_prevents_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _workspace(tmp_path)
    original = MemoryRecoveryManifest.append

    def drift(self, **kwargs):
        result = original(self, **kwargs)
        if kwargs["phase"] == "LEGACY_FINALIZED":
            (memory / ".cursor").write_bytes(b"changed")
        return result

    monkeypatch.setattr(MemoryRecoveryManifest, "append", drift)
    with pytest.raises(finalize.ConversionFinalizationError):
        _finalize(tmp_path)
    assert _phases(tmp_path)[-1] == "LEGACY_FINALIZED"
    assert not (memory / "HISTORY.md").exists()


def test_output_blob_drift_blocks_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from nanobot.agent import memory_recovery_storage as storage

    memory = _workspace(tmp_path)
    digest = hashlib.sha256((memory / "history.jsonl").read_bytes()).hexdigest()
    original = storage.WorkspaceMemoryAuthority.read_blob

    def corrupt(self, requested):
        if requested == digest:
            return b"corrupt output blob"
        return original(self, requested)

    monkeypatch.setattr(storage.WorkspaceMemoryAuthority, "read_blob", corrupt)
    with pytest.raises(finalize.ConversionFinalizationError):
        _finalize(tmp_path)
    assert _phases(tmp_path)[-1] == "CURSORS_APPLIED"
    assert (memory / "HISTORY.md").exists()


@pytest.mark.parametrize("replacement", ["copy", "symlink"])
def test_source_replacement_is_not_two_link_replay(tmp_path: Path, replacement: str) -> None:
    memory = _workspace(tmp_path)
    source = memory / "HISTORY.md"
    content = source.read_bytes()
    source.unlink()
    if replacement == "copy":
        source.write_bytes(content)
    else:
        source.symlink_to("history.jsonl")
    with pytest.raises(finalize.ConversionFinalizationError):
        _finalize(tmp_path)
    assert _phases(tmp_path)[-1] == "CURSORS_APPLIED"
    assert not (memory / "HISTORY.md.bak").exists()


@pytest.mark.parametrize("point", ["after_archive", "after_finalized_append"])
def test_backup_drift_never_claims_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, point: str,
) -> None:
    memory = _workspace(tmp_path)
    backup = memory / "HISTORY.md.bak"
    if point == "after_archive":
        original = finalize.archive.archive_legacy_history_if_unchanged
        injected = False

        def corrupt(*args, **kwargs):
            nonlocal injected
            result = original(*args, **kwargs)
            if not injected:
                injected = True
                backup.write_bytes(b"tampered")
            return result

        monkeypatch.setattr(finalize.archive, "archive_legacy_history_if_unchanged", corrupt)
    else:
        original_append = MemoryRecoveryManifest.append

        def corrupt(self, **kwargs):
            result = original_append(self, **kwargs)
            if kwargs["phase"] == "LEGACY_FINALIZED":
                backup.write_bytes(b"tampered")
            return result

        monkeypatch.setattr(MemoryRecoveryManifest, "append", corrupt)
    with pytest.raises(finalize.ConversionFinalizationError):
        _finalize(tmp_path)
    assert backup.read_bytes() == b"tampered"
    assert not (memory / "HISTORY.md").exists()
    assert _phases(tmp_path)[-1] == (
        "CURSORS_APPLIED" if point == "after_archive" else "LEGACY_FINALIZED"
    )
    assert "COMPLETE" not in _phases(tmp_path)
