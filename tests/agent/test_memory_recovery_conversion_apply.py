"""Offline target application replays authenticated frozen preparation evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_conversion_apply as apply
from nanobot.agent import memory_recovery_conversion_runtime as prepare
from nanobot.agent import memory_recovery_target_mutation as mutation
from nanobot.agent.memory_recovery_manifest import MemoryRecoveryManifest

_WORKSPACE_ID = "a" * 32


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
    receipt_path = memory / ".consolidation.pending.json"
    receipt_path.write_bytes(json.dumps(receipt, sort_keys=True).encode())
    receipt_path.chmod(0o600)
    prepare.prepare_receipt_tail_conversion(tmp_path, workspace_id=_WORKSPACE_ID)
    return memory


def _apply(tmp_path: Path):
    return apply.apply_prepared_conversion_targets(tmp_path, workspace_id=_WORKSPACE_ID)


def test_convert_and_idempotent_replay(tmp_path: Path) -> None:
    memory = _workspace(tmp_path)
    source = (memory / "HISTORY.md").read_bytes()
    receipt = (memory / ".consolidation.pending.json").read_bytes()
    result = _apply(tmp_path)
    assert result.phase == "CURSORS_APPLIED"
    assert _apply(tmp_path).digest == result.digest
    assert (memory / "history.jsonl").read_bytes().count(b"receipt text") == 1
    assert (memory / ".cursor").read_bytes() == b"2"
    assert (memory / ".dream_cursor").read_bytes() == b"2"
    assert (memory / "HISTORY.md").read_bytes() == source
    assert (memory / ".consolidation.pending.json").read_bytes() == receipt


def test_preserve_no_op_keeps_inode(tmp_path: Path) -> None:
    memory = _workspace(tmp_path, jsonl=b'{"cursor":999}\n')
    before = {name: (memory / name).stat().st_ino for name in ("history.jsonl", "HISTORY.md")}
    result = _apply(tmp_path)
    assert result.phase == "CURSORS_APPLIED"
    assert _apply(tmp_path).digest == result.digest
    assert {name: (memory / name).stat().st_ino for name in before} == before
    assert not (memory / ".cursor").exists()


@pytest.mark.parametrize("crash_at", [1, 2, 3])
def test_replay_after_each_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_at: int) -> None:
    memory = _workspace(tmp_path)
    original = mutation.replace_target_if_unchanged
    count = 0

    def crash(*args: object):
        nonlocal count
        count += 1
        result = original(*args)
        if count == crash_at:
            raise mutation.MemoryTargetReplacementUncertainError()
        return result

    monkeypatch.setattr(mutation, "replace_target_if_unchanged", crash)
    assert _apply(tmp_path).phase == "CURSORS_APPLIED"
    monkeypatch.setattr(mutation, "replace_target_if_unchanged", original)
    assert _apply(tmp_path).phase == "CURSORS_APPLIED"
    assert (memory / "history.jsonl").read_bytes().count(b"receipt text") == 1


@pytest.mark.parametrize("phase", ["JSONL_APPLIED", "CURSORS_APPLIED"])
def test_replay_after_phase_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    memory = _workspace(tmp_path)
    original = MemoryRecoveryManifest.append

    def crash(self: MemoryRecoveryManifest, **kwargs: object):
        result = original(self, **kwargs)
        if kwargs["phase"] == phase:
            raise RuntimeError("crash")
        return result

    monkeypatch.setattr(MemoryRecoveryManifest, "append", crash)
    with pytest.raises(apply.ConversionApplyError):
        _apply(tmp_path)
    monkeypatch.setattr(MemoryRecoveryManifest, "append", original)
    assert _apply(tmp_path).phase == "CURSORS_APPLIED"
    assert (memory / "history.jsonl").read_bytes().count(b"receipt text") == 1


def test_copied_old_inode_conflicts(tmp_path: Path) -> None:
    memory = _workspace(tmp_path, jsonl=b"")
    target = memory / "history.jsonl"
    old = target.read_bytes()
    target.unlink()
    target.write_bytes(old)
    with pytest.raises(apply.ConversionApplyError):
        _apply(tmp_path)


@pytest.mark.parametrize("crash_at", [1, 2, 3])
def test_before_rename_fails_closed_then_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_at: int
) -> None:
    memory = _workspace(tmp_path)
    original = mutation.replace_target_if_unchanged
    count = 0

    def crash(*args: object):
        nonlocal count
        count += 1
        if count == crash_at:
            raise mutation.MemoryTargetReplacementUncertainError()
        return original(*args)

    monkeypatch.setattr(mutation, "replace_target_if_unchanged", crash)
    with pytest.raises(apply.ConversionApplyError):
        _apply(tmp_path)
    monkeypatch.setattr(mutation, "replace_target_if_unchanged", original)
    assert _apply(tmp_path).phase == "CURSORS_APPLIED"
    assert (memory / "history.jsonl").read_bytes().count(b"receipt text") == 1


@pytest.mark.parametrize("phase", ["JSONL_APPLIED", "CURSORS_APPLIED"])
def test_replay_before_phase_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    _workspace(tmp_path)
    original = MemoryRecoveryManifest.append

    def crash(self: MemoryRecoveryManifest, **kwargs: object):
        if kwargs["phase"] == phase:
            raise RuntimeError("crash")
        return original(self, **kwargs)

    monkeypatch.setattr(MemoryRecoveryManifest, "append", crash)
    with pytest.raises(apply.ConversionApplyError):
        _apply(tmp_path)
    monkeypatch.setattr(MemoryRecoveryManifest, "append", original)
    assert _apply(tmp_path).phase == "CURSORS_APPLIED"


def test_first_cursor_only_replays_without_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = _workspace(tmp_path)
    original = mutation.replace_target_if_unchanged
    count = 0

    def crash(*args: object):
        nonlocal count
        count += 1
        if count == 3:
            raise mutation.MemoryTargetReplacementUncertainError()
        return original(*args)

    monkeypatch.setattr(mutation, "replace_target_if_unchanged", crash)
    with pytest.raises(apply.ConversionApplyError):
        _apply(tmp_path)
    before = (memory / ".cursor").stat().st_ino
    monkeypatch.setattr(mutation, "replace_target_if_unchanged", original)
    assert _apply(tmp_path).phase == "CURSORS_APPLIED"
    assert (memory / ".cursor").stat().st_ino == before


@pytest.mark.parametrize("target", ["HISTORY.md", "history.jsonl", ".cursor", ".dream_cursor"])
def test_changed_old_file_is_rejected(tmp_path: Path, target: str) -> None:
    memory = _workspace(tmp_path)
    path = memory / target
    path.write_bytes(b"changed")
    with pytest.raises(apply.ConversionApplyError):
        _apply(tmp_path)
    assert not (memory / "HISTORY.md.bak").exists()


@pytest.mark.parametrize("target", ["history.jsonl", ".cursor", ".dream_cursor"])
def test_copied_old_sidecar_inode_is_rejected(tmp_path: Path, target: str) -> None:
    memory = _workspace(tmp_path, jsonl=b"")
    path = memory / target
    if target != "history.jsonl":
        # Prepare a legitimate old cursor by first creating it before preparation.
        memory = tmp_path / "second" / "memory"
        memory.mkdir(parents=True)
        (memory / "HISTORY.md").write_bytes(b"# Existing\n\n")
        (memory / target).write_bytes(b"1")
        if target == ".dream_cursor":
            (memory / ".cursor").write_bytes(b"1")
        receipt = (tmp_path / "memory" / ".consolidation.pending.json").read_bytes()
        copied_receipt = memory / ".consolidation.pending.json"
        copied_receipt.write_bytes(receipt)
        copied_receipt.chmod(0o600)
        prepare.prepare_receipt_tail_conversion(tmp_path / "second", workspace_id=_WORKSPACE_ID)
        path = memory / target
    data = path.read_bytes()
    path.unlink()
    path.write_bytes(data)
    root = tmp_path if target == "history.jsonl" else tmp_path / "second"
    with pytest.raises(apply.ConversionApplyError):
        apply.apply_prepared_conversion_targets(root, workspace_id=_WORKSPACE_ID)


@pytest.mark.parametrize("mode", ["symlink", "unsafe"])
def test_unsafe_target_is_rejected(tmp_path: Path, mode: str) -> None:
    memory = _workspace(tmp_path)
    path = memory / "history.jsonl"
    if mode == "symlink":
        path.symlink_to(memory / "HISTORY.md")
    else:
        path.write_bytes(b"unsafe")
        path.chmod(0o666)
    with pytest.raises(apply.ConversionApplyError):
        _apply(tmp_path)


@pytest.mark.parametrize("claim", ["receipt", "source", "output", "canonical_row"])
def test_immutable_blob_tamper_is_rejected(tmp_path: Path, claim: str) -> None:
    _workspace(tmp_path)
    from nanobot.agent import memory_recovery as receipt_codec
    from nanobot.agent import memory_recovery_conversion as conversion
    from nanobot.agent import memory_recovery_storage as storage
    from nanobot.agent import memory_recovery_transaction as transaction

    with storage.open_workspace_memory(tmp_path) as authority:
        raw = authority.snapshot_target(".consolidation.pending.json").content
        assert raw is not None
        receipt = receipt_codec._validate_payload(receipt_codec._decode(raw), raw)
        with authority.operation() as operation:
            tx = transaction.capture_transaction_payload(
                receipt, workspace_id=_WORKSPACE_ID, workspace_identity=operation.workspace_identity
            )
        cv = conversion.capture_receipt_tail_conversion_payload(tx.identity)
        records = MemoryRecoveryManifest(
            authority.open_evidence_namespace("conversions", cv.identity)
        ).load()
        payload = records[1].payload
        digest = {
            "receipt": payload["receipt"]["blob_digest"],
            "source": payload["source"]["blob_digest"],
            "output": payload["outputs"]["history.jsonl"]["blob_digest"],
            "canonical_row": payload["canonical_row"]["blob_digest"],
        }[claim]
        blob = authority.blob_directory / digest
    blob.write_bytes(b"tampered")
    with pytest.raises(apply.ConversionApplyError):
        _apply(tmp_path)


def test_receipt_tamper_rejected(tmp_path: Path) -> None:
    memory = _workspace(tmp_path)
    receipt = memory / ".consolidation.pending.json"
    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(apply.ConversionApplyError):
        _apply(tmp_path)


def test_identical_cursor_outputs_keep_old_inodes(tmp_path: Path) -> None:
    memory = _workspace(tmp_path)
    # A separate state must include sidecars before preparation.
    other = tmp_path / "identical"
    memory = other / "memory"
    memory.mkdir(parents=True)
    (memory / "HISTORY.md").write_bytes(b"# Existing\n\n")
    receipt = memory / ".consolidation.pending.json"
    receipt.write_bytes((tmp_path / "memory" / ".consolidation.pending.json").read_bytes())
    receipt.chmod(0o600)
    for name in (".cursor", ".dream_cursor"):
        (memory / name).write_bytes(b"2")
    prepare.prepare_receipt_tail_conversion(other, workspace_id=_WORKSPACE_ID)
    before = {name: (memory / name).stat().st_ino for name in (".cursor", ".dream_cursor")}
    assert apply.apply_prepared_conversion_targets(other, workspace_id=_WORKSPACE_ID).phase == "CURSORS_APPLIED"
    assert {name: (memory / name).stat().st_ino for name in before} == before


def test_replay_rejects_changed_output(tmp_path: Path) -> None:
    memory = _workspace(tmp_path)
    assert _apply(tmp_path).phase == "CURSORS_APPLIED"
    (memory / ".cursor").write_bytes(b"changed")
    with pytest.raises(apply.ConversionApplyError):
        _apply(tmp_path)


def _mixed_noop_workspace(tmp_path: Path) -> tuple[Path, Path, int]:
    original = _workspace(tmp_path)
    root = tmp_path / "mixed"
    memory = root / "memory"
    memory.mkdir(parents=True)
    (memory / "HISTORY.md").write_bytes(b"# Existing\n\n")
    receipt = memory / ".consolidation.pending.json"
    receipt.write_bytes((original / ".consolidation.pending.json").read_bytes())
    receipt.chmod(0o600)
    cursor = memory / ".cursor"
    cursor.write_bytes(b"2")
    prepare.prepare_receipt_tail_conversion(root, workspace_id=_WORKSPACE_ID)
    return root, memory, cursor.stat().st_ino


def test_cursor_identical_dream_changing_replays_with_original_cursor_inode(
    tmp_path: Path,
) -> None:
    root, memory, cursor_inode = _mixed_noop_workspace(tmp_path)
    applied = apply.apply_prepared_conversion_targets(root, workspace_id=_WORKSPACE_ID)
    assert applied.phase == "CURSORS_APPLIED"
    assert apply.apply_prepared_conversion_targets(root, workspace_id=_WORKSPACE_ID).digest == applied.digest
    assert (memory / ".cursor").stat().st_ino == cursor_inode
    assert (memory / ".dream_cursor").read_bytes() == b"2"


def test_interruption_after_dream_rename_replays_mixed_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, memory, cursor_inode = _mixed_noop_workspace(tmp_path)
    original = mutation.replace_target_if_unchanged

    def interrupt(*args: object):
        result = original(*args)
        if args[1].name == ".dream_cursor":
            raise RuntimeError("injected after dream rename")
        return result

    monkeypatch.setattr(mutation, "replace_target_if_unchanged", interrupt)
    with pytest.raises(apply.ConversionApplyError):
        apply.apply_prepared_conversion_targets(root, workspace_id=_WORKSPACE_ID)
    monkeypatch.setattr(mutation, "replace_target_if_unchanged", original)
    applied = apply.apply_prepared_conversion_targets(root, workspace_id=_WORKSPACE_ID)
    assert applied.phase == "CURSORS_APPLIED"
    assert apply.apply_prepared_conversion_targets(root, workspace_id=_WORKSPACE_ID).digest == applied.digest
    assert (memory / ".cursor").stat().st_ino == cursor_inode
    assert (memory / ".dream_cursor").read_bytes() == b"2"
