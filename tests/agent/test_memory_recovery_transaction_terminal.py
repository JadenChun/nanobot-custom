"""The terminal coordinator appends typed commit/completion after full replay proof."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent import memory_recovery_transaction_history as history
from nanobot.agent import memory_recovery_transaction_memory as memory_stage
from nanobot.agent import memory_recovery_transaction_offset as offset_stage
from nanobot.agent import memory_recovery_transaction_terminal as terminal
from nanobot.agent.memory_recovery_manifest import MemoryRecoveryManifest
from nanobot.session.manager import JsonlSessionStore
from tests.agent.test_memory_recovery_transaction_memory import _workspace_without_memory
from tests.agent.test_memory_recovery_transaction_prepare import (
    _SESSION_KEY,
    _WORKSPACE_ID,
    _prepare,
    _transaction_chain,
    _workspace,
)


def _setup(tmp_path: Path, *, output: str | None = "updated memory\n", reference_only: bool = False):
    workspace, sessions, memory, receipt = _workspace(
        tmp_path, memory_output=output, reference_only=reference_only,
    )
    _prepare(workspace, sessions)
    history.reconcile_memory_recovery_transaction_history(
        workspace, workspace_id=_WORKSPACE_ID, sessions_root=sessions,
    )
    memory_stage.reconcile_memory_recovery_transaction_memory(
        workspace, workspace_id=_WORKSPACE_ID, sessions_root=sessions,
    )
    offset_stage.reconcile_memory_recovery_transaction_offset(
        workspace, workspace_id=_WORKSPACE_ID, sessions_root=sessions,
    )
    source = sessions / _WORKSPACE_ID / f"{JsonlSessionStore.storage_key(_SESSION_KEY)}.jsonl"
    return workspace, sessions, memory, receipt, source


def _reconcile(workspace: Path, sessions: Path):
    return terminal.reconcile_memory_recovery_transaction_terminal(
        workspace, workspace_id=_WORKSPACE_ID, sessions_root=sessions,
    )


def _phases(workspace: Path, *, receipt_bytes: bytes | None = None) -> list[str]:
    return [record.phase for record in _transaction_chain(workspace, receipt_bytes=receipt_bytes)[1]]


def _identity(path: Path) -> tuple[int, int, int, int, int, int, int, int, int]:
    value = path.lstat()
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
        value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns,
    )


def _live_state(workspace: Path, memory: Path, receipt: Path, source: Path) -> dict[Path, tuple[bytes, tuple[int, ...]]]:
    paths = (
        receipt,
        source,
        memory / "history.jsonl",
        memory / ".cursor",
        memory / "MEMORY.md",
        memory / "HISTORY.md.bak",
    )
    return {path: (path.read_bytes(), _identity(path)) for path in paths}


def test_terminal_appends_typed_commit_and_complete_and_replays_without_live_mutation(tmp_path: Path):
    workspace, sessions, memory, receipt, source = _setup(tmp_path)
    before = _live_state(workspace, memory, receipt, source)

    complete = _reconcile(workspace, sessions)
    _, records = _transaction_chain(workspace)

    assert complete.phase == "COMPLETE"
    assert records[5].phase == "RECEIPT_COMMITTED"
    assert records[5].predecessor_digest == records[4].digest
    assert records[5].payload["prepared_record_digest"] == records[1].digest
    assert records[5].payload["history_applied_record_digest"] == records[2].digest
    assert records[5].payload["memory_applied_record_digest"] == records[3].digest
    assert records[5].payload["offset_applied_record_digest"] == records[4].digest
    assert records[5].payload["receipt_present"] is True
    assert records[6].phase == "COMPLETE"
    assert records[6].predecessor_digest == records[5].digest
    assert records[6].payload["commit_record_digest"] == records[5].digest
    assert records[6].payload["marker_occurrences"] == 1
    assert complete.digest == records[6].digest
    assert _phases(workspace) == [
        "CAPTURED", "PREPARED", "HISTORY_APPLIED", "MEMORY_APPLIED", "OFFSET_APPLIED",
        "RECEIPT_COMMITTED", "COMPLETE",
    ]
    assert _live_state(workspace, memory, receipt, source) == before
    assert receipt.exists()
    assert _reconcile(workspace, sessions) == complete
    assert _phases(workspace)[-1] == "COMPLETE"
    assert _live_state(workspace, memory, receipt, source) == before


@pytest.mark.parametrize(
    ("phase", "failure_point", "expected_after_failure"),
    [
        ("RECEIPT_COMMITTED", "before", "OFFSET_APPLIED"),
        ("RECEIPT_COMMITTED", "after", "RECEIPT_COMMITTED"),
        ("COMPLETE", "before", "RECEIPT_COMMITTED"),
        ("COMPLETE", "after", "COMPLETE"),
    ],
)
def test_append_failures_replay_exact_terminal_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    failure_point: str,
    expected_after_failure: str,
):
    workspace, sessions, memory, receipt, source = _setup(tmp_path)
    before = _live_state(workspace, memory, receipt, source)
    original = MemoryRecoveryManifest.append
    injected = False
    fault_phase = phase

    def fault_once(self, *, predecessor_digest: str, phase: str, payload: dict[str, Any]):
        nonlocal injected
        if phase == fault_phase and not injected:
            injected = True
            if failure_point == "before":
                raise OSError("injected before terminal publication")
            original(self, predecessor_digest=predecessor_digest, phase=phase, payload=payload)
            raise OSError("injected after terminal publication")
        return original(self, predecessor_digest=predecessor_digest, phase=phase, payload=payload)
    monkeypatch.setattr(MemoryRecoveryManifest, "append", fault_once)
    with pytest.raises(terminal.MemoryRecoveryTransactionTerminalError):
        _reconcile(workspace, sessions)

    assert _phases(workspace)[-1] == expected_after_failure
    assert _live_state(workspace, memory, receipt, source) == before

    complete = _reconcile(workspace, sessions)

    assert complete.phase == "COMPLETE"
    assert _phases(workspace)[-1] == "COMPLETE"
    assert _live_state(workspace, memory, receipt, source) == before
    assert _reconcile(workspace, sessions) == complete


def test_post_commit_drift_blocks_complete_until_full_proof_is_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    workspace, sessions, memory, _, _ = _setup(tmp_path)
    cursor = memory / ".cursor"
    original_cursor = cursor.read_bytes()
    original_append = MemoryRecoveryManifest.append

    def append_then_drift(self, *, predecessor_digest: str, phase: str, payload: dict[str, Any]):
        record = original_append(self, predecessor_digest=predecessor_digest, phase=phase, payload=payload)
        if phase == "RECEIPT_COMMITTED":
            cursor.write_bytes(b"999")
        return record

    monkeypatch.setattr(MemoryRecoveryManifest, "append", append_then_drift)
    with pytest.raises(terminal.MemoryRecoveryTransactionTerminalError):
        _reconcile(workspace, sessions)
    assert _phases(workspace)[-1] == "RECEIPT_COMMITTED"

    monkeypatch.setattr(MemoryRecoveryManifest, "append", original_append)
    cursor.write_bytes(original_cursor)
    assert _reconcile(workspace, sessions).phase == "COMPLETE"


@pytest.mark.parametrize("drift", ["receipt", "source", "history", "cursor", "memory", "backup", "marker"])
def test_any_terminal_proof_drift_blocks_complete(tmp_path: Path, drift: str):
    workspace, sessions, memory, receipt, source = _setup(tmp_path)
    original_receipt = receipt.read_bytes()
    if drift == "receipt":
        receipt.write_bytes(original_receipt + b" ")
    elif drift == "source":
        source.write_bytes(source.read_bytes() + b" ")
    elif drift == "history":
        target = memory / "history.jsonl"
        target.write_bytes(target.read_bytes() + b"\n")
    elif drift == "cursor":
        (memory / ".cursor").write_bytes(b"999")
    elif drift == "memory":
        (memory / "MEMORY.md").write_bytes(b"drifted memory\n")
    elif drift == "backup":
        (memory / "HISTORY.md.bak").write_bytes(b"drifted backup\n")
    else:
        target = memory / "history.jsonl"
        rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
        rows[-1]["content"] = "marker removed"
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(terminal.MemoryRecoveryTransactionTerminalError):
        _reconcile(workspace, sessions)

    assert _phases(workspace, receipt_bytes=original_receipt)[-1] == "OFFSET_APPLIED"


def test_byte_identical_receipt_inode_replacement_fails_prepared_stat_binding(tmp_path: Path):
    workspace, sessions, memory, receipt, _ = _setup(tmp_path)
    raw = receipt.read_bytes()
    original_inode = receipt.stat().st_ino
    replacement = memory / "receipt-copy"
    replacement.write_bytes(raw)
    replacement.chmod(0o600)
    os.replace(replacement, receipt)
    assert receipt.stat().st_ino != original_inode

    with pytest.raises(terminal.MemoryRecoveryTransactionTerminalError):
        _reconcile(workspace, sessions)

    assert _phases(workspace)[-1] == "OFFSET_APPLIED"


@pytest.mark.parametrize("corruption", ["predecessor", "phase"])
def test_corrupt_manifest_link_or_phase_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
):
    workspace, sessions, _, _, _ = _setup(tmp_path)
    original = MemoryRecoveryManifest.load

    def corrupted_load(self):
        records = original(self)
        if records and records[0].kind == "transactions":
            changed = list(records)
            index = 2 if corruption == "predecessor" else 3
            if corruption == "predecessor":
                changed[index] = replace(changed[index], predecessor_digest="0" * 64)
            else:
                changed[index] = replace(changed[index], phase="PREPARED")
            return tuple(changed)
        return records

    monkeypatch.setattr(MemoryRecoveryManifest, "load", corrupted_load)
    with pytest.raises(terminal.MemoryRecoveryTransactionTerminalError):
        _reconcile(workspace, sessions)

    monkeypatch.setattr(MemoryRecoveryManifest, "load", original)
    assert _phases(workspace)[-1] == "OFFSET_APPLIED"


def test_reference_only_terminal_preserves_memory_target(tmp_path: Path):
    workspace, sessions, memory, receipt, source = _setup(
        tmp_path, output=None, reference_only=True,
    )
    memory_path = memory / "MEMORY.md"
    before = _live_state(workspace, memory, receipt, source)
    memory_before = (memory_path.read_bytes(), _identity(memory_path))

    complete = _reconcile(workspace, sessions)

    assert complete.phase == "COMPLETE"
    assert (memory_path.read_bytes(), _identity(memory_path)) == memory_before
    assert _live_state(workspace, memory, receipt, source) == before


def test_reference_only_terminal_preserves_absent_memory_target(tmp_path: Path):
    workspace, sessions, memory, _ = _workspace_without_memory(
        tmp_path, memory_output=None, reference_only=True,
    )
    _prepare(workspace, sessions)
    history.reconcile_memory_recovery_transaction_history(
        workspace, workspace_id=_WORKSPACE_ID, sessions_root=sessions,
    )
    memory_stage.reconcile_memory_recovery_transaction_memory(
        workspace, workspace_id=_WORKSPACE_ID, sessions_root=sessions,
    )
    offset_stage.reconcile_memory_recovery_transaction_offset(
        workspace, workspace_id=_WORKSPACE_ID, sessions_root=sessions,
    )

    assert not (memory / "MEMORY.md").exists()
    assert _reconcile(workspace, sessions).phase == "COMPLETE"
    assert not (memory / "MEMORY.md").exists()
