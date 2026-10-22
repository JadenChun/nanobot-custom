"""MEMORY_APPLIED proves one authorized memory target and replays exactly once."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent import memory_recovery_conversion_apply as conversion_apply
from nanobot.agent import memory_recovery_conversion_finalize as conversion_finalize
from nanobot.agent import memory_recovery_conversion_runtime as conversion_prepare
from nanobot.agent import memory_recovery_manifest as transaction_manifest
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_target_mutation as mutation
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent import memory_recovery_transaction_history as history
from nanobot.agent import memory_recovery_transaction_memory as memory_stage
from nanobot.session.manager import JsonlSessionStore
from tests.agent.test_memory_recovery_transaction_prepare import (
    _SESSION_KEY,
    _WORKSPACE_ID,
    _chunk_digest,
    _prepare,
    _session_bytes,
    _session_events,
    _transaction_chain,
    _workspace,
    _write_session,
)


def _reconcile(workspace: Path, sessions_root: Path):
    return memory_stage.reconcile_memory_recovery_transaction_memory(
        workspace,
        workspace_id=_WORKSPACE_ID,
        sessions_root=sessions_root,
    )


def _prepare_history(workspace: Path, sessions_root: Path) -> None:
    _prepare(workspace, sessions_root)
    history.reconcile_memory_recovery_transaction_history(
        workspace,
        workspace_id=_WORKSPACE_ID,
        sessions_root=sessions_root,
    )


def _session_path(workspace: Path, sessions_root: Path) -> Path:
    del workspace
    return (
        sessions_root / _WORKSPACE_ID
        / f"{JsonlSessionStore.storage_key(_SESSION_KEY)}.jsonl"
    )


def _identity(path: Path) -> tuple[int, ...] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _live_state(paths: tuple[Path, ...]) -> dict[Path, tuple[bytes | None, tuple[int, ...] | None]]:
    return {
        path: (path.read_bytes() if path.exists() else None, _identity(path))
        for path in paths
    }


def _live_paths(workspace: Path, sessions_root: Path, memory: Path, receipt: Path):
    return (
        receipt,
        _session_path(workspace, sessions_root),
        memory / "history.jsonl",
        memory / ".cursor",
        memory / "MEMORY.md",
    )


def _workspace_without_memory(
    tmp_path: Path,
    *,
    memory_output: str | None,
    reference_only: bool,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir(mode=0o700)
    memory = workspace / "memory"
    memory.mkdir()
    (memory / "HISTORY.md").write_bytes(b"[2026-09-24 12:00] Prior note.\n\n")
    base_digest = hashlib.sha256(b"").hexdigest()
    output_digest = (
        None if memory_output is None
        else hashlib.sha256(memory_output.encode("utf-8")).hexdigest()
    )
    receipt: dict[str, Any] = {
        "version": 1,
        "chunk_digest": _chunk_digest(_session_events()[1:2]),
        "history_marker_id": "receipt-marker-prepare",
        "history_entry": "[2026-09-25 12:00] Prepared receipt entry.",
        "memory_output": memory_output,
        "base_memory_digest": base_digest,
        "output_memory_digest": output_digest,
        "reference_only": reference_only,
        "created_at": "2026-09-25T12:00:00Z",
        "session_key": _SESSION_KEY,
        "start_offset": 1,
        "end_offset": 2,
        "session_generation": 3,
    }
    receipt_path = memory / ".consolidation.pending.json"
    receipt_path.write_bytes(json.dumps(receipt, sort_keys=True).encode("utf-8"))
    receipt_path.chmod(0o600)
    _write_session(workspace, sessions_root, _session_bytes())
    conversion_prepare.prepare_receipt_tail_conversion(workspace, workspace_id=_WORKSPACE_ID)
    conversion_apply.apply_prepared_conversion_targets(workspace, workspace_id=_WORKSPACE_ID)
    conversion_finalize.finalize_prepared_conversion(workspace, workspace_id=_WORKSPACE_ID)
    return workspace, sessions_root, memory, receipt_path


def test_replace_is_durable_typed_and_replays_one_memory_phase(tmp_path: Path) -> None:
    workspace, sessions_root, memory, receipt = _workspace(
        tmp_path, memory_output="updated long-term memory\n",
    )
    _prepare_history(workspace, sessions_root)
    paths = _live_paths(workspace, sessions_root, memory, receipt)
    before = _live_state(paths)

    record = _reconcile(workspace, sessions_root)

    assert record.phase == "MEMORY_APPLIED"
    assert (memory / "MEMORY.md").read_bytes() == b"updated long-term memory\n"
    info = (memory / "MEMORY.md").lstat()
    assert stat.S_IMODE(info.st_mode) == 0o600
    assert info.st_nlink == 1
    _, records = _transaction_chain(workspace)
    assert [item.phase for item in records] == [
        "CAPTURED", "PREPARED", "HISTORY_APPLIED", "MEMORY_APPLIED",
    ]
    assert all(
        _live_state((path,))[path] == before[path]
        for path in paths
        if path != memory / "MEMORY.md"
    )

    after = _live_state(paths)
    replay = _reconcile(workspace, sessions_root)

    assert replay == record
    _, replayed = _transaction_chain(workspace)
    assert [item.phase for item in replayed] == [item.phase for item in records]
    assert _live_state(paths) == after


def test_reference_only_non_null_output_preserves_present_target_identity(tmp_path: Path) -> None:
    workspace, sessions_root, memory, receipt = _workspace(
        tmp_path,
        memory_output="text present only as a reference",
        reference_only=True,
    )
    _prepare_history(workspace, sessions_root)
    target = memory / "MEMORY.md"
    before = _live_state((target,))[target]

    record = _reconcile(workspace, sessions_root)

    assert record.payload["memory_action"] == "preserve"
    assert record.payload["memory_target"] == _transaction_chain(workspace)[1][1].payload[
        "targets"
    ]["MEMORY.md"]["new"]
    assert _live_state((target,))[target] == before
    assert [item.phase for item in _transaction_chain(workspace)[1]] == [
        "CAPTURED", "PREPARED", "HISTORY_APPLIED", "MEMORY_APPLIED",
    ]


def test_preserve_keeps_absent_memory_absent(tmp_path: Path) -> None:
    workspace, sessions_root, memory, receipt = _workspace_without_memory(
        tmp_path, memory_output=None, reference_only=True,
    )
    _prepare_history(workspace, sessions_root)
    target = memory / "MEMORY.md"
    assert not target.exists()

    record = _reconcile(workspace, sessions_root)

    assert record.payload["memory_action"] == "preserve"
    assert record.payload["memory_target"]["exists"] is False
    assert not target.exists()
    assert _identity(target) is None
    assert [item.phase for item in _transaction_chain(workspace)[1]] == [
        "CAPTURED", "PREPARED", "HISTORY_APPLIED", "MEMORY_APPLIED",
    ]


def test_replace_with_same_bytes_is_an_original_inode_no_op(tmp_path: Path) -> None:
    workspace, sessions_root, memory, receipt = _workspace(
        tmp_path, memory_output="base memory\n", reference_only=False,
    )
    _prepare_history(workspace, sessions_root)
    target = memory / "MEMORY.md"
    before = _live_state((target,))[target]

    _reconcile(workspace, sessions_root)

    assert _live_state((target,))[target] == before


def test_replace_can_create_memory_from_an_absent_old_target(tmp_path: Path) -> None:
    workspace, sessions_root, memory, receipt = _workspace_without_memory(
        tmp_path, memory_output="created memory\n", reference_only=False,
    )
    _prepare_history(workspace, sessions_root)
    target = memory / "MEMORY.md"
    assert not target.exists()

    _reconcile(workspace, sessions_root)

    assert target.read_bytes() == b"created memory\n"
    info = target.lstat()
    assert stat.S_IMODE(info.st_mode) == 0o600
    assert info.st_nlink == 1


@pytest.mark.parametrize(
    "drift", ["receipt", "receipt-copy", "source", "history", "old-copy"],
)
def test_drift_fails_without_memory_phase(
    tmp_path: Path,
    drift: str,
) -> None:
    workspace, sessions_root, memory, receipt = _workspace(
        tmp_path, memory_output="updated memory\n",
    )
    _prepare_history(workspace, sessions_root)
    original_receipt = receipt.read_bytes()
    target = memory / "MEMORY.md"
    paths = _live_paths(workspace, sessions_root, memory, receipt)
    if drift == "receipt":
        receipt.write_bytes(original_receipt + b" ")
    elif drift == "receipt-copy":
        copy = receipt.with_name("receipt.copy")
        copy.write_bytes(original_receipt)
        copy.chmod(0o600)
        os.replace(copy, receipt)
    elif drift == "source":
        source = _session_path(workspace, sessions_root)
        source.write_bytes(source.read_bytes() + b" ")
    elif drift == "history":
        (memory / "history.jsonl").write_bytes(
            (memory / "history.jsonl").read_bytes() + b" "
        )
    else:
        copy = target.with_name("MEMORY.copy")
        copy.write_bytes(target.read_bytes())
        copy.chmod(target.stat().st_mode & 0o777)
        os.replace(copy, target)
    before = _live_state(paths)

    with pytest.raises(memory_stage.MemoryRecoveryTransactionMemoryError):
        _reconcile(workspace, sessions_root)

    assert _live_state(paths) == before
    _, records = _transaction_chain(workspace, receipt_bytes=original_receipt)
    assert [item.phase for item in records] == [
        "CAPTURED", "PREPARED", "HISTORY_APPLIED",
    ]


def test_corrupt_output_blob_and_manifest_fail_closed(tmp_path: Path) -> None:
    workspace, sessions_root, memory, receipt = _workspace(
        tmp_path, memory_output="updated memory\n",
    )
    _prepare_history(workspace, sessions_root)
    original_receipt = receipt.read_bytes()
    manifest, records = _transaction_chain(workspace)
    blob_digest = records[1].payload["targets"]["MEMORY.md"]["new"]["blob_digest"]
    blob = workspace / "memory" / ".recovery" / "blobs" / "sha256" / blob_digest
    blob.write_bytes(b"corrupted output blob")
    target = memory / "MEMORY.md"
    before = _live_state((target,))[target]

    with pytest.raises(memory_stage.MemoryRecoveryTransactionMemoryError):
        _reconcile(workspace, sessions_root)

    assert _live_state((target,))[target] == before
    assert [item.phase for item in _transaction_chain(workspace, receipt_bytes=original_receipt)[1]] == [
        "CAPTURED", "PREPARED", "HISTORY_APPLIED",
    ]

    prepared_path = manifest._namespace.directory / f"next-{records[0].digest}.json"
    raw = prepared_path.read_bytes()
    prepared_path.write_bytes(raw.replace(b"PREPARED", b"PREPARED-TAMPERED"))
    with pytest.raises(memory_stage.MemoryRecoveryTransactionMemoryError):
        _reconcile(workspace, sessions_root)
    assert _live_state((target,))[target] == before


@pytest.mark.parametrize("fault", ["before-rename", "after-rename"])
def test_rename_fault_reconciles_only_exact_old_or_authorized_new(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    workspace, sessions_root, memory, receipt = _workspace(
        tmp_path, memory_output="updated memory\n",
    )
    _prepare_history(workspace, sessions_root)
    target = memory / "MEMORY.md"
    original_identity = _identity(target)

    if fault == "before-rename":
        def fail_before(*args, **kwargs):
            raise OSError("injected uncertain pre-rename failure")

        # The capability probe checks function identity; skip it so the fault
        # lands inside the already-qualified descriptor-relative rename call.
        monkeypatch.setattr(mutation, "_check_capabilities", lambda: None)
        monkeypatch.setattr(mutation.os, "rename", fail_before)
        with pytest.raises(memory_stage.MemoryRecoveryTransactionMemoryError):
            _reconcile(workspace, sessions_root)
        assert _identity(target) == original_identity
        assert [item.phase for item in _transaction_chain(workspace)[1]] == [
            "CAPTURED", "PREPARED", "HISTORY_APPLIED",
        ]
    else:
        original_verify = mutation._verify_replaced_snapshot

        def verify_then_fail(*args, **kwargs):
            original_verify(*args, **kwargs)
            raise OSError("injected post-rename failure")

        monkeypatch.setattr(mutation, "_verify_replaced_snapshot", verify_then_fail)
        record = _reconcile(workspace, sessions_root)
        assert record.phase == "MEMORY_APPLIED"
        assert target.read_bytes() == b"updated memory\n"
        assert _identity(target) != original_identity
        assert [item.phase for item in _transaction_chain(workspace)[1]] == [
            "CAPTURED", "PREPARED", "HISTORY_APPLIED", "MEMORY_APPLIED",
        ]


@pytest.mark.parametrize("drift", ["receipt", "source", "history"])
def test_drift_after_append_fails_and_replay_cannot_add_another_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    workspace, sessions_root, memory, receipt = _workspace(
        tmp_path, memory_output="updated memory\n",
    )
    _prepare_history(workspace, sessions_root)
    original_receipt = receipt.read_bytes()
    target = {
        "receipt": receipt,
        "source": _session_path(workspace, sessions_root),
        "history": memory / "history.jsonl",
    }[drift]
    original_append = transaction_manifest.MemoryRecoveryManifest.append
    injected = False

    def append_then_drift(self, **kwargs):
        nonlocal injected
        record = original_append(self, **kwargs)
        if kwargs.get("phase") == "MEMORY_APPLIED" and not injected:
            injected = True
            target.write_bytes(target.read_bytes() + b" ")
        return record

    monkeypatch.setattr(
        transaction_manifest.MemoryRecoveryManifest, "append", append_then_drift,
    )
    with pytest.raises(memory_stage.MemoryRecoveryTransactionMemoryError):
        _reconcile(workspace, sessions_root)
    assert injected
    _, records = _transaction_chain(workspace, receipt_bytes=original_receipt)
    assert [item.phase for item in records] == [
        "CAPTURED", "PREPARED", "HISTORY_APPLIED", "MEMORY_APPLIED",
    ]

    with pytest.raises(memory_stage.MemoryRecoveryTransactionMemoryError):
        _reconcile(workspace, sessions_root)
    _, replayed = _transaction_chain(workspace, receipt_bytes=original_receipt)
    assert [item.phase for item in replayed] == [item.phase for item in records]


def test_rejects_phase_after_memory_applied_without_live_changes(tmp_path: Path) -> None:
    workspace, sessions_root, memory, receipt = _workspace(
        tmp_path, memory_output="updated memory\n",
    )
    _prepare_history(workspace, sessions_root)
    _reconcile(workspace, sessions_root)
    manifest, records = _transaction_chain(workspace)
    captured = transaction.decode_transaction_payload("CAPTURED", records[0].payload)
    prepared = transaction.decode_transaction_payload(
        "PREPARED", records[1].payload, captured=captured,
    )
    history_applied = transaction.decode_transaction_payload(
        "HISTORY_APPLIED", records[2].payload,
        captured=captured,
        prepared=prepared,
    )
    memory_applied = transaction.decode_transaction_payload(
        "MEMORY_APPLIED", records[3].payload,
        captured=captured,
        prepared=prepared,
        history_applied=history_applied,
    )
    offset = transaction.offset_applied_transaction_payload(
        captured, prepared, history_applied, memory_applied,
    )
    with storage.open_workspace_memory(workspace) as authority, authority.operation():
        append_manifest = transaction_manifest.MemoryRecoveryManifest(
            authority.open_evidence_namespace("transactions", captured.identity)
        )
        append_manifest.append(
            predecessor_digest=records[3].digest,
            phase="OFFSET_APPLIED",
            payload=offset.to_payload(),
        )
    target = memory / "MEMORY.md"
    before = _live_state((target,))[target]

    with pytest.raises(memory_stage.MemoryRecoveryTransactionMemoryError):
        _reconcile(workspace, sessions_root)

    assert _live_state((target,))[target] == before
    assert [item.phase for item in _transaction_chain(workspace)[1]] == [
        "CAPTURED", "PREPARED", "HISTORY_APPLIED", "MEMORY_APPLIED", "OFFSET_APPLIED",
    ]
