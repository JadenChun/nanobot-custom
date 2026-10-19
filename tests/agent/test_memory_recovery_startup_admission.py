"""Startup admission requires historical semantics, not structural quarantine alone."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.agent import memory_recovery as receipt_codec
from nanobot.agent import memory_recovery_coordinator as coordinator
from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_receipt_finalizer as finalizer
from nanobot.agent import memory_recovery_startup_admission as admission
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent.memory_recovery import MemoryReceiptRecoveryRequiredError
from nanobot.agent.memory_recovery_historical_seal import validate_historical_seals_under_gate
from nanobot.agent.memory_recovery_startup_admission import (
    StartupReceiptAdmissionError,
    admitted_startup_mutation,
    require_startup_receipt_admission,
)
from nanobot.session.manager import JsonlSessionStore
from tests.agent.test_memory_recovery_quarantine_storage import (
    _layout,
    _publish_v1_committed,
    _real_layout,
    _seal_real_receipt,
)
from tests.agent.test_memory_recovery_transaction_prepare import _SESSION_KEY, _WORKSPACE_ID


def _denied(action) -> None:
    with pytest.raises(StartupReceiptAdmissionError) as failure:
        action()
    assert isinstance(failure.value, MemoryReceiptRecoveryRequiredError)
    assert str(failure.value) == "memory recovery required"
    assert failure.value.__cause__ is None


def test_clean_absent_workspace_preflight_is_read_only(tmp_path: Path) -> None:
    workspace = tmp_path / "missing"
    require_startup_receipt_admission(workspace, sessions_root=tmp_path / "sessions")
    assert not workspace.exists()
    with admitted_startup_mutation(workspace, sessions_root=tmp_path / "sessions") as gate:
        if os.name == "posix":
            assert gate is not None
            gate.require_current()
        assert workspace.is_dir()


def test_clean_existing_workspace_preflight_does_not_create_lock(tmp_path: Path) -> None:
    require_startup_receipt_admission(tmp_path, sessions_root=tmp_path / "missing")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="real seal requires POSIX descriptor protocol")
def test_real_seal_admits_after_mutable_target_edits_and_rejects_wrong_root(tmp_path: Path) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    (workspace / "memory" / "MEMORY.md").write_bytes(b"later legitimate edit\n")
    require_startup_receipt_admission(workspace, sessions_root=sessions)
    with admitted_startup_mutation(workspace, sessions_root=sessions) as gate:
        assert gate is not None
        gate.require_current()
    _denied(lambda: require_startup_receipt_admission(workspace, sessions_root=tmp_path))
    assert quarantine.inspect_receipt_quarantine(workspace).sealed_identities == (
        binding.transaction_identity,
    )


@pytest.mark.skipif(os.name != "posix", reason="real seals require POSIX descriptor protocol")
def test_two_real_terminal_receipts_are_sealed_and_both_required_at_startup(
    tmp_path: Path,
) -> None:
    workspace, sessions, _, first = _real_layout(tmp_path)
    finalizer.finalize_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    memory = workspace / "memory"
    first_copy = (memory / ".recovery" / "receipt-quarantine"
                  / first.transaction_identity / "receipt.bin")
    assert first_copy.exists()

    # Simulate a second legacy-history recovery on the *same* workspace and
    # checked session. A fresh legacy source and empty JSONL are necessary:
    # the offline conversion deliberately rejects preserve-nonempty-JSONL.
    source = (sessions / _WORKSPACE_ID
              / f"{JsonlSessionStore.storage_key(_SESSION_KEY)}.jsonl")
    header, *events = [json.loads(line) for line in source.read_bytes().splitlines()]
    assert header["last_archived"] == header["last_consolidated"] == len(events) == 2
    header["revision"] += 1
    header["updated_at"] = "2026-09-26T00:01:00"
    followup = {"role": "assistant", "content": "second recovery event", "_event_id": "event-3"}
    events.append(followup)
    source.write_bytes(b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        for row in (header, *events)
    ))
    (memory / "history.jsonl").write_bytes(b"")
    (memory / "HISTORY.md").write_bytes(b"[2026-09-26 12:00] Another legacy note.\n\n")
    base = (memory / "MEMORY.md").read_bytes()
    output = b"second recovered memory\n"
    second_receipt = json.loads(first_copy.read_bytes())
    second_receipt.update({
        "chunk_digest": hashlib.sha256(json.dumps(
            [followup], ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "history_marker_id": "receipt-marker-second",
        "history_entry": "[2026-09-26 12:00] Second receipt entry.",
        "memory_output": output.decode("utf-8"),
        "base_memory_digest": hashlib.sha256(base).hexdigest(),
        "output_memory_digest": hashlib.sha256(output).hexdigest(),
        "created_at": "2026-09-26T12:00:00Z",
        "start_offset": 2,
        "end_offset": 3,
    })
    pending = memory / ".consolidation.pending.json"
    pending.write_bytes(json.dumps(second_receipt, sort_keys=True).encode("utf-8"))
    pending.chmod(0o600)
    complete = coordinator.reconcile_pending_memory_receipt_offline(
        workspace, sessions_root=sessions,
    )
    assert complete.phase == "COMPLETE"
    second_identity = complete.identity
    assert second_identity != first.transaction_identity
    finalizer.finalize_pending_memory_receipt_offline(workspace, sessions_root=sessions)
    assert not pending.exists()

    identities = tuple(sorted((first.transaction_identity, second_identity)))
    inspection = quarantine.inspect_receipt_quarantine(workspace)
    assert inspection.committed_identities == inspection.sealed_identities == identities
    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        evidence = validate_historical_seals_under_gate(gate, sessions)
    assert evidence.sealed_identities == identities
    assert len(set(evidence.semantic_seal_digests)) == 2
    require_startup_receipt_admission(workspace, sessions_root=sessions)

    # Admission must check both real semantic histories, not just the last
    # sealed identity. Restoring each corruption reestablishes admission.
    for identity in identities:
        genesis = memory / ".recovery" / "transactions" / identity / "genesis.json"
        original = genesis.read_bytes()
        genesis.write_bytes(b"corrupt frozen transaction\n")
        _denied(lambda: require_startup_receipt_admission(workspace, sessions_root=sessions))
        assert genesis.read_bytes() == b"corrupt frozen transaction\n"
        genesis.write_bytes(original)
        require_startup_receipt_admission(workspace, sessions_root=sessions)
    assert first_copy.exists()


@pytest.mark.skipif(os.name != "posix", reason="real seal requires POSIX descriptor protocol")
def test_semantically_corrupt_seal_is_not_admitted(tmp_path: Path) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    index = workspace / "memory" / ".recovery" / "transactions" / binding.transaction_identity
    manifest = next(index.glob("*.json"))
    manifest.write_bytes(b"corrupt frozen record\n")
    before = manifest.read_bytes()
    _denied(lambda: require_startup_receipt_admission(workspace, sessions_root=sessions))
    assert manifest.read_bytes() == before


@pytest.mark.skipif(os.name != "posix", reason="real seal requires POSIX descriptor protocol")
def test_missing_lock_and_copied_workspace_reject_without_repair(tmp_path: Path) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    copied = tmp_path / "copied"
    shutil.copytree(workspace, copied)
    _denied(lambda: require_startup_receipt_admission(copied, sessions_root=sessions))
    lock = workspace / ".memory-recovery.lock"
    assert lock.exists()
    lock.unlink()
    _denied(lambda: require_startup_receipt_admission(workspace, sessions_root=sessions))
    assert not lock.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX quarantine index")
@pytest.mark.parametrize("committed", [False, True])
def test_incomplete_and_v1_only_evidence_reject(tmp_path: Path, committed: bool) -> None:
    workspace, binding = _layout(tmp_path)
    pending = workspace / "memory" / ".consolidation.pending.json"
    with storage.open_workspace_memory_lock(workspace) as gate:
        with quarantine.ReceiptQuarantineStore(gate, binding) as store:
            store.prepare(pending.read_bytes())
            if committed:
                store.copy_receipt()
                pending.unlink()
                _publish_v1_committed(store)
    _denied(lambda: require_startup_receipt_admission(workspace, sessions_root=binding.sessions_root))
    assert quarantine.inspect_receipt_quarantine(workspace).committed_identities == (
        (binding.transaction_identity,) if committed else ()
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX quarantine index")
@pytest.mark.parametrize("entry", [None, "unknown"])
def test_empty_or_unknown_index_is_not_clean(tmp_path: Path, entry: str | None) -> None:
    workspace = tmp_path / "workspace"
    index = workspace / "memory" / ".recovery" / "receipt-quarantine"
    index.mkdir(parents=True, mode=0o700)
    (workspace / "memory").chmod(0o700)
    (workspace / "memory" / ".recovery").chmod(0o700)
    if entry:
        (index / entry).mkdir(mode=0o700)
    _denied(lambda: require_startup_receipt_admission(workspace, sessions_root=tmp_path))
    assert index.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX quarantine index")
def test_all_inspected_seals_must_be_semantically_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first, second = "a" * 64, "b" * 64
    observed = quarantine.QuarantineInspection(False, False, (first, second), (first, second))
    monkeypatch.setattr(admission, "_inspect", lambda path: observed)
    with storage.open_workspace_memory_lock(workspace):
        pass
    monkeypatch.setattr(admission, "validate_historical_seals_under_gate",
                        lambda gate, root: SimpleNamespace(sealed_identities=(first,)))
    _denied(lambda: require_startup_receipt_admission(workspace, sessions_root=tmp_path))
    monkeypatch.setattr(admission, "validate_historical_seals_under_gate",
                        lambda gate, root: SimpleNamespace(sealed_identities=(first, second)))
    require_startup_receipt_admission(workspace, sessions_root=tmp_path)

    def corrupted_second(gate, root):
        raise ValueError("sensitive semantic contents")

    monkeypatch.setattr(admission, "validate_historical_seals_under_gate", corrupted_second)
    _denied(lambda: require_startup_receipt_admission(workspace, sessions_root=tmp_path))


@pytest.mark.skipif(os.name != "posix", reason="POSIX gate race")
def test_receipt_arriving_between_preflight_and_gate_denies_before_yield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = storage.open_workspace_memory_lock

    def intervening_gate(path: Path):
        memory = workspace / "memory"
        memory.mkdir(mode=0o700)
        pending = memory / ".consolidation.pending.json"
        pending.write_bytes(b"new pending evidence")
        pending.chmod(0o600)
        return original(path)

    monkeypatch.setattr(storage, "open_workspace_memory_lock", intervening_gate)
    entered = False

    def attempt() -> None:
        nonlocal entered
        with admitted_startup_mutation(workspace, sessions_root=tmp_path):
            entered = True

    _denied(attempt)
    assert not entered
    assert (workspace / "memory" / ".consolidation.pending.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX gate race")
def test_index_arriving_between_preflight_and_gate_denies_before_yield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = storage.open_workspace_memory_lock

    def intervening_gate(path: Path):
        index = workspace / "memory" / ".recovery" / "receipt-quarantine"
        index.mkdir(parents=True, mode=0o700)
        (workspace / "memory").chmod(0o700)
        (workspace / "memory" / ".recovery").chmod(0o700)
        return original(path)

    monkeypatch.setattr(storage, "open_workspace_memory_lock", intervening_gate)
    entered = False

    def attempt() -> None:
        nonlocal entered
        with admitted_startup_mutation(workspace, sessions_root=tmp_path):
            entered = True

    _denied(attempt)
    assert not entered
    assert (workspace / "memory" / ".recovery" / "receipt-quarantine").exists()


def test_pending_and_unsupported_inspection_clean_still_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    memory = workspace / "memory"
    memory.mkdir(parents=True, mode=0o700)
    pending = memory / ".consolidation.pending.json"
    pending.write_bytes(b"pending")
    pending.chmod(0o600)
    monkeypatch.setattr(quarantine, "inspect_receipt_quarantine",
                        lambda workspace: quarantine.QuarantineInspection(True, False, ()))
    _denied(lambda: require_startup_receipt_admission(workspace, sessions_root=tmp_path))
    def attempt() -> None:
        with admitted_startup_mutation(workspace, sessions_root=tmp_path):
            pytest.fail("pending receipt admitted")

    _denied(attempt)


def test_unsupported_platform_clean_absence_and_pending_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(admission, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(quarantine, "_support", lambda: quarantine._fail())
    monkeypatch.setattr(receipt_codec, "_descriptor_flags", lambda: None)
    require_startup_receipt_admission(workspace, sessions_root=tmp_path)
    with admitted_startup_mutation(workspace, sessions_root=tmp_path) as gate:
        assert gate is None
    memory = workspace / "memory"
    memory.mkdir(mode=0o700)
    pending = memory / ".consolidation.pending.json"
    pending.symlink_to(tmp_path / "target")
    _denied(lambda: require_startup_receipt_admission(workspace, sessions_root=tmp_path))
