"""End-to-end, explicitly offline pending-receipt reconciliation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_conversion_apply as conversion_apply
from nanobot.agent import memory_recovery_conversion_finalize as conversion_finalize
from nanobot.agent import memory_recovery_conversion_runtime as conversion_prepare
from nanobot.agent import memory_recovery_coordinator as coordinator
from nanobot.agent import memory_recovery_transaction_history as transaction_history
from nanobot.agent import memory_recovery_transaction_memory as transaction_memory
from nanobot.agent import memory_recovery_transaction_offset as transaction_offset
from nanobot.agent import memory_recovery_transaction_prepare as transaction_prepare
from nanobot.agent import memory_recovery_transaction_terminal as transaction_terminal
from nanobot.agent.memory_recovery_manifest import MemoryRecoveryManifest
from nanobot.session.manager import JsonlSessionStore
from tests.state_stat import content_stat

KEY = "telegram:offline-receipt"
WORKSPACE_ID = "a" * 32


def _fixture(tmp_path: Path, *, jsonl: bytes | None = None) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    sessions = tmp_path / "sessions"
    sessions.mkdir(mode=0o700)
    state = workspace / ".nanobot"
    state.mkdir(mode=0o700)
    marker = state / "workspace-id"
    marker.write_text(WORKSPACE_ID + "\n")
    marker.chmod(0o600)
    namespace = sessions / WORKSPACE_ID
    namespace.mkdir(mode=0o700)
    binding = namespace / ".workspace"
    binding.write_text(str(workspace.resolve()) + "\n")
    binding.chmod(0o600)
    events = [
        {"role": "user", "content": "old", "_event_id": "event-1"},
        {"role": "assistant", "content": "new", "_event_id": "event-2"},
    ]
    header = {
        "_type": "metadata", "key": KEY, "created_at": "2026-09-25T00:00:00",
        "updated_at": "2026-09-25T00:01:00", "revision": 7, "generation": 3,
        "incarnation": "session-incarnation", "last_archived": 1,
        "last_consolidated": 1, "metadata": {"source_schema_version": 1},
    }
    source = namespace / f"{JsonlSessionStore.storage_key(KEY)}.jsonl"
    source.write_bytes(b"".join((json.dumps(row) + "\n").encode() for row in [header, *events]))
    source.chmod(0o600)
    memory = workspace / "memory"
    memory.mkdir()
    (memory / "HISTORY.md").write_bytes(b"[2026-09-24 12:00] Prior note.\n\n")
    (memory / "MEMORY.md").write_bytes(b"base memory\n")
    if jsonl is not None:
        (memory / "history.jsonl").write_bytes(jsonl)
    chunk = json.dumps(events[1:2], ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), default=str).encode()
    receipt = {
        "version": 1, "chunk_digest": hashlib.sha256(chunk).hexdigest(),
        "history_marker_id": "offline-marker", "history_entry": "[2026-09-25 12:00] Receipt.",
        "memory_output": "updated memory\n",
        "base_memory_digest": hashlib.sha256(b"base memory\n").hexdigest(),
        "output_memory_digest": hashlib.sha256(b"updated memory\n").hexdigest(),
        "reference_only": False, "created_at": "2026-09-25T12:00:00Z",
        "session_key": KEY, "start_offset": 1, "end_offset": 2, "session_generation": 3,
    }
    pending = memory / ".consolidation.pending.json"
    pending.write_bytes(json.dumps(receipt, sort_keys=True).encode())
    pending.chmod(0o600)
    return workspace, sessions, source


def test_reconcile_from_receipt_only_and_replay(tmp_path: Path) -> None:
    workspace, sessions, source = _fixture(tmp_path)
    memory = workspace / "memory"
    pending = memory / ".consolidation.pending.json"
    before = (pending.read_bytes(), content_stat(pending.stat()))
    result = coordinator.reconcile_pending_memory_receipt_offline(
        workspace, sessions_root=sessions,
    )
    assert result.phase == "COMPLETE"
    assert pending.read_bytes() == before[0]
    assert content_stat(pending.stat()) == before[1]
    assert (memory / "history.jsonl").read_bytes().count(b"offline-marker") == 1
    assert json.loads(source.read_bytes().splitlines()[0])["last_consolidated"] == 2
    targets = {p.name: (p.read_bytes(), content_stat(p.stat())) for p in (pending, source,
        memory / "history.jsonl", memory / "MEMORY.md", memory / ".cursor")}
    assert coordinator.reconcile_pending_memory_receipt_offline(
        workspace, sessions_root=sessions,
    ).phase == "COMPLETE"
    assert {p.name: (p.read_bytes(), content_stat(p.stat())) for p in (pending, source,
        memory / "history.jsonl", memory / "MEMORY.md", memory / ".cursor")} == targets


def test_absent_receipt_fails_without_target_changes(tmp_path: Path) -> None:
    workspace, sessions, source = _fixture(tmp_path)
    (workspace / "memory" / ".consolidation.pending.json").unlink()
    before = source.read_bytes()
    with pytest.raises(coordinator.OfflineMemoryRecoveryError, match="^memory recovery unavailable$"):
        coordinator.reconcile_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    assert source.read_bytes() == before


@pytest.mark.parametrize("module,name", [
    (conversion_prepare, "prepare_receipt_tail_conversion"),
    (conversion_apply, "apply_prepared_conversion_targets"),
    (conversion_finalize, "finalize_prepared_conversion"),
    (transaction_prepare, "prepare_memory_recovery_transaction"),
    (transaction_history, "reconcile_memory_recovery_transaction_history"),
    (transaction_memory, "reconcile_memory_recovery_transaction_memory"),
    (transaction_offset, "reconcile_memory_recovery_transaction_offset"),
    (transaction_terminal, "reconcile_memory_recovery_transaction_terminal"),
])
def test_restart_after_durable_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: object, name: str,
) -> None:
    workspace, sessions, _ = _fixture(tmp_path)
    original = getattr(module, name)

    def interrupted(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)
        raise RuntimeError("injected after durable stage")

    with monkeypatch.context() as patch:
        patch.setattr(module, name, interrupted)
        with pytest.raises(coordinator.OfflineMemoryRecoveryError):
            coordinator.reconcile_pending_memory_receipt_offline(
                workspace, sessions_root=sessions,
            )
    assert coordinator.reconcile_pending_memory_receipt_offline(
        workspace, sessions_root=sessions,
    ).phase == "COMPLETE"
    memory = workspace / "memory"
    assert (memory / "history.jsonl").read_bytes().count(b"offline-marker") == 1
    assert (memory / "MEMORY.md").read_bytes() == b"updated memory\n"


def test_corrupt_manifest_fails_closed(tmp_path: Path) -> None:
    workspace, sessions, source = _fixture(tmp_path)
    conversion_prepare.prepare_receipt_tail_conversion(workspace, workspace_id=WORKSPACE_ID)
    records = list((workspace / "memory" / ".recovery" / "transactions").rglob("genesis.json"))
    assert len(records) == 1
    records[0].write_bytes(b"invalid")
    before = source.read_bytes()
    with pytest.raises(coordinator.OfflineMemoryRecoveryError):
        coordinator.reconcile_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    assert source.read_bytes() == before
    assert (workspace / "memory" / "HISTORY.md").exists()


def test_unsafe_session_and_receipt_drift_fail_closed(tmp_path: Path) -> None:
    workspace, sessions, source = _fixture(tmp_path)
    pending = workspace / "memory" / ".consolidation.pending.json"
    before = pending.read_bytes()
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(coordinator.OfflineMemoryRecoveryError):
        coordinator.reconcile_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    assert pending.read_bytes() == before
    source.unlink()
    source.write_bytes(outside.read_bytes())
    source.chmod(0o600)
    altered = json.loads(before)
    altered["session_key"] = "telegram:missing"
    pending.write_bytes(json.dumps(altered).encode())
    with pytest.raises(coordinator.OfflineMemoryRecoveryError):
        coordinator.reconcile_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    assert (workspace / "memory" / "HISTORY.md").exists()


def test_preserve_nonempty_jsonl_requires_operator(tmp_path: Path) -> None:
    workspace, sessions, source = _fixture(tmp_path, jsonl=b'{"existing":true}\n')
    original = (source.read_bytes(), (workspace / "memory" / "history.jsonl").read_bytes())
    with pytest.raises(coordinator.OfflineMemoryRecoveryError):
        coordinator.reconcile_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    assert source.read_bytes() == original[0]
    assert (workspace / "memory" / "history.jsonl").read_bytes() == original[1]
    assert (workspace / "memory" / ".consolidation.pending.json").exists()


@pytest.mark.parametrize("terminal_phase", ["RECEIPT_COMMITTED", "COMPLETE"])
def test_restart_after_terminal_manifest_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_phase: str,
) -> None:
    workspace, sessions, source = _fixture(tmp_path)
    original = MemoryRecoveryManifest.append
    triggered = False

    def interrupted(self: MemoryRecoveryManifest, **kwargs: object):
        nonlocal triggered
        record = original(self, **kwargs)
        if self._kind == "transactions" and kwargs.get("phase") == terminal_phase and not triggered:
            triggered = True
            raise RuntimeError("injected after terminal append")
        return record

    with monkeypatch.context() as patch:
        patch.setattr(MemoryRecoveryManifest, "append", interrupted)
        with pytest.raises(coordinator.OfflineMemoryRecoveryError):
            coordinator.reconcile_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    assert triggered
    after = source.read_bytes()
    assert coordinator.reconcile_pending_memory_receipt_offline(
        workspace, sessions_root=sessions,
    ).phase == "COMPLETE"
    assert source.read_bytes() == after
    assert (workspace / "memory" / "history.jsonl").read_bytes().count(b"offline-marker") == 1


def test_drifted_receipt_after_conversion_fails_closed(tmp_path: Path) -> None:
    workspace, sessions, source = _fixture(tmp_path)
    conversion_prepare.prepare_receipt_tail_conversion(workspace, workspace_id=WORKSPACE_ID)
    pending = workspace / "memory" / ".consolidation.pending.json"
    altered = json.loads(pending.read_bytes())
    altered["history_entry"] = "different receipt entry"
    pending.write_bytes(json.dumps(altered, sort_keys=True).encode())
    before = source.read_bytes()
    with pytest.raises(coordinator.OfflineMemoryRecoveryError):
        coordinator.reconcile_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    assert source.read_bytes() == before
    assert (workspace / "memory" / "HISTORY.md").exists()


def test_session_header_mismatch_fails_before_conversion_targets(tmp_path: Path) -> None:
    workspace, sessions, source = _fixture(tmp_path)
    rows = source.read_bytes().splitlines()
    header = json.loads(rows[0])
    header["key"] = "telegram:other"
    source.write_bytes(b"".join([json.dumps(header).encode() + b"\n", *(
        row + b"\n" for row in rows[1:]
    )]))
    before = source.read_bytes()
    with pytest.raises(coordinator.OfflineMemoryRecoveryError):
        coordinator.reconcile_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    assert source.read_bytes() == before
    assert (workspace / "memory" / "HISTORY.md").exists()
    assert not (workspace / "memory" / "history.jsonl").exists()
