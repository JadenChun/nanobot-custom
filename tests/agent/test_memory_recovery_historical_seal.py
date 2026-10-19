"""Historical startup evidence is independent of later mutable target bytes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_conversion as conversion
from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_transaction_terminal as terminal
from nanobot.agent.memory_recovery_historical_seal import (
    HistoricalSealError,
    validate_historical_seals_under_gate,
)
from tests.agent.test_memory_recovery_quarantine_storage import _real_layout, _seal_real_receipt
from tests.agent.test_memory_recovery_transaction_terminal import _transaction_chain

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor protocol")


def test_sealed_history_survives_later_target_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    seal = _seal_real_receipt(workspace, sessions, receipt, binding)
    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        evidence = validate_historical_seals_under_gate(gate, sessions)
    assert evidence.sealed_identities == (binding.transaction_identity,)
    assert evidence.semantic_seal_digests == (seal,)

    (workspace / "memory" / "MEMORY.md").write_bytes(b"later legitimate memory\n")
    (workspace / "memory" / "history.jsonl").write_bytes(b"later history\n")
    (workspace / "memory" / ".cursor").write_bytes(b"99")
    session = sessions / binding.workspace_id
    for source in session.glob("*.jsonl"):
        source.write_bytes(source.read_bytes() + b"\n")
    calls = 0
    original_barrier = quarantine.establish_durable_quarantine_seals

    def counted_barrier(gate):
        nonlocal calls
        calls += 1
        return original_barrier(gate)

    def forbidden_terminal(*args, **kwargs):
        raise AssertionError("historical validation must not call current-target proof")

    monkeypatch.setattr(quarantine, "establish_durable_quarantine_seals", counted_barrier)
    monkeypatch.setattr(terminal, "prove_terminal_complete_under_locks", forbidden_terminal)
    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        assert validate_historical_seals_under_gate(gate, sessions) == evidence
    assert calls == 1


def test_missing_gate_never_admits(tmp_path: Path) -> None:
    with pytest.raises(HistoricalSealError):
        validate_historical_seals_under_gate(object(), tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(storage.MemoryStorageError):
        with storage.open_existing_workspace_memory_lock(workspace) as gate:
            validate_historical_seals_under_gate(gate, tmp_path)
    assert not (workspace / ".memory-recovery.lock").exists()


def test_clean_absence_under_existing_lock_does_not_create(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with storage.open_workspace_memory_lock(workspace):
        pass
    sessions = tmp_path / "sessions"
    sessions.mkdir(mode=0o700)
    before = tuple(workspace.iterdir())
    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        assert validate_historical_seals_under_gate(gate, sessions).sealed_identities == ()
    assert tuple(workspace.iterdir()) == before


def _sealed(tmp_path: Path):
    workspace, sessions, receipt, binding = _real_layout(tmp_path)
    _seal_real_receipt(workspace, sessions, receipt, binding)
    recovery = workspace / "memory" / ".recovery"
    quarantine_dir = recovery / "receipt-quarantine" / binding.transaction_identity
    return workspace, sessions, binding, recovery, quarantine_dir


def _payload(directory: Path, sequence: int) -> dict:
    slot = "genesis.json"
    for _ in range(sequence):
        raw = (directory / slot).read_bytes()
        slot = f"next-{hashlib.sha256(raw).hexdigest()}.json"
    return json.loads((directory / slot).read_bytes())["payload"]


@pytest.mark.parametrize("damage", [
    "copy", "committed", "transaction", "conversion", "blob", "old-source", "row",
    "receipt-blob", "seal", "stat",
    "v1", "workspace-id", "symlink", "hardlink", "unsafe", "incomplete", "unknown-slot",
])
def test_corrupt_or_competing_evidence_blocks(tmp_path: Path, damage: str) -> None:
    workspace, sessions, binding, recovery, quarantine_dir = _sealed(tmp_path)
    copy = quarantine_dir / "receipt.bin"
    committed = quarantine_dir / "committed.json"
    if damage == "copy":
        copy.write_bytes(copy.read_bytes() + b"x")
    elif damage == "committed":
        committed.write_bytes(b"{}\n")
    elif damage == "transaction":
        (recovery / "transactions" / binding.transaction_identity / "genesis.json").write_bytes(b"{}")
    elif damage == "conversion":
        identity = conversion.capture_receipt_tail_conversion_payload(binding.transaction_identity).identity
        (recovery / "conversions" / identity / "genesis.json").write_bytes(b"{}")
    elif damage == "blob":
        _, records = _transaction_chain(workspace, receipt_bytes=copy.read_bytes())
        digest = records[1].payload["source"]["new_digest"]
        (recovery / "blobs" / "sha256" / digest).write_bytes(b"corrupt")
    elif damage in {"old-source", "row"}:
        identity = conversion.capture_receipt_tail_conversion_payload(binding.transaction_identity).identity
        prepared = _payload(recovery / "conversions" / identity, 1)
        digest = (prepared["source"]["blob_digest"] if damage == "old-source"
                  else prepared["canonical_row"]["blob_digest"])
        (recovery / "blobs" / "sha256" / digest).write_bytes(b"corrupt")
    elif damage == "receipt-blob":
        (recovery / "blobs" / "sha256" / binding.receipt_digest).write_bytes(b"corrupt")
    elif damage == "seal":
        value = json.loads(committed.read_bytes())
        value["semantic_seal_digest"] = "0" * 64
        committed.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    elif damage == "stat":
        value = json.loads(committed.read_bytes())
        value["receipt_stat"][1] += 1
        committed.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    elif damage == "v1":
        v1 = conversion.capture_conversion_payload(binding.transaction_identity).identity
        (recovery / "conversions" / v1 / "genesis.json").write_bytes(b"{}")
    elif damage == "workspace-id":
        (workspace / ".nanobot" / "workspace-id").write_bytes(b"0" * 32 + b"\n")
    elif damage == "symlink":
        committed.unlink()
        committed.symlink_to("prepared.json")
    elif damage == "hardlink":
        (quarantine_dir / "extra-copy").hardlink_to(copy)
    elif damage == "unsafe":
        copy.chmod(0o644)
    elif damage == "incomplete":
        (recovery / "receipt-quarantine" / ("f" * 64)).mkdir(mode=0o700)
    elif damage == "unknown-slot":
        (recovery / "transactions" / binding.transaction_identity / "unexpected.json").write_bytes(b"{}")
    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        with pytest.raises(HistoricalSealError, match="^historical memory recovery seal is unavailable$"):
            validate_historical_seals_under_gate(gate, sessions)


def test_wrong_sessions_root_and_copied_workspace_binding_block(tmp_path: Path) -> None:
    workspace, sessions, binding, _, _ = _sealed(tmp_path)
    other = tmp_path / "other-sessions"
    other.mkdir(mode=0o700)
    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        with pytest.raises(HistoricalSealError):
            validate_historical_seals_under_gate(gate, other)
    marker = workspace / ".nanobot" / "workspace-id"
    marker.write_bytes(b"f" * 32 + b"\n")
    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        with pytest.raises(HistoricalSealError):
            validate_historical_seals_under_gate(gate, sessions)


def test_copied_workspace_inode_does_not_reuse_seal(tmp_path: Path) -> None:
    workspace, sessions, _, _, _ = _sealed(tmp_path)
    prior = tmp_path / "prior-workspace"
    workspace.rename(prior)
    shutil.copytree(prior, workspace)
    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        with pytest.raises(HistoricalSealError):
            validate_historical_seals_under_gate(gate, sessions)


@pytest.mark.parametrize("corrupt", [False, True])
def test_validation_never_creates_or_mutates_application_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool,
) -> None:
    workspace, sessions, _, _, quarantine_dir = _sealed(tmp_path)
    if corrupt:
        committed = quarantine_dir / "committed.json"
        value = json.loads(committed.read_bytes())
        value["semantic_seal_digest"] = hashlib.sha256(b"wrong").hexdigest()
        committed.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    calls: list[str] = []
    with storage.open_existing_workspace_memory_lock(workspace) as gate:
        with monkeypatch.context() as patch:
            original_open = os.open
            supported = os.supports_dir_fd

            def checked_open(path, flags, *args, **kwargs):
                if flags & (os.O_CREAT | os.O_TRUNC | os.O_APPEND | os.O_WRONLY | os.O_RDWR):
                    calls.append("open-write")
                    raise AssertionError("write open")
                return original_open(path, flags, *args, **kwargs)

            def forbidden(*args, **kwargs):
                calls.append("mutation")
                raise AssertionError("mutation")

            patch.setattr(os, "open", checked_open)
            for name in ("mkdir", "replace", "rename", "unlink", "chmod"):
                patch.setattr(os, name, forbidden)
            patch.setattr(os, "supports_dir_fd", supported | {checked_open, forbidden})
            if corrupt:
                with pytest.raises(HistoricalSealError):
                    validate_historical_seals_under_gate(gate, sessions)
            else:
                validate_historical_seals_under_gate(gate, sessions)
    assert not calls
