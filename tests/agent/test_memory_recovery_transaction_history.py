"""HISTORY_APPLIED reconciles completed conversion evidence without live writes."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_manifest as transaction_manifest
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_transaction as transaction
from nanobot.agent import memory_recovery_transaction_history as history
from nanobot.session.manager import JsonlSessionStore
from tests.agent.test_memory_recovery_transaction_prepare import (
    _SESSION_KEY,
    _WORKSPACE_ID,
    _prepare,
    _transaction_chain,
    _workspace,
)


def _live_paths(workspace: Path, sessions_root: Path, memory: Path, receipt: Path) -> tuple[Path, ...]:
    session = (
        sessions_root / _WORKSPACE_ID
        / f"{JsonlSessionStore.storage_key(_SESSION_KEY)}.jsonl"
    )
    return (receipt, session, memory / "history.jsonl", memory / ".cursor", memory / "MEMORY.md")


def _live_state(paths: tuple[Path, ...]) -> dict[Path, tuple[bytes, tuple[int, ...]]]:
    return {
        path: (
            path.read_bytes(),
            (
                (info := path.lstat()).st_dev,
                info.st_ino,
                info.st_mode,
                info.st_uid,
                info.st_gid,
                info.st_nlink,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            ),
        )
        for path in paths
    }


def _assert_live_state(paths: tuple[Path, ...], before: dict[Path, tuple[bytes, tuple[int, ...]]]) -> None:
    assert _live_state(paths) == before


def _reconcile(workspace: Path, sessions_root: Path):
    return history.reconcile_memory_recovery_transaction_history(
        workspace,
        workspace_id=_WORKSPACE_ID,
        sessions_root=sessions_root,
    )


def _tamper_target_copy(path: Path) -> None:
    replacement = path.with_name(path.name + ".replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(path.stat().st_mode & 0o777)
    os.replace(replacement, path)


def _tamper_marker(path: Path, *, mode: str) -> None:
    lines = path.read_bytes().splitlines()
    record = json.loads(lines[-1])
    marker = "<!-- nanobot-consolidation:receipt-marker-prepare -->"
    if mode == "duplicate":
        record["content"] += "\n" + marker
    elif mode == "conflicting":
        record["content"] = marker + "\nconflicting converted history entry"
    else:
        record["content"] = "marker removed"
    lines[-1] = json.dumps(record, ensure_ascii=False).encode("utf-8")
    path.write_bytes(b"\n".join(lines) + b"\n")


def _tamper_session(path: Path) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[2]["content"] = "source drift with the same generation, revision, and cursor"
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _tamper_receipt_copy(path: Path) -> None:
    replacement = path.with_name(path.name + ".replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.chmod(0o600)
    os.replace(replacement, path)


def _tamper_proposal_blob(workspace: Path) -> Path:
    _, records = _transaction_chain(workspace)
    digest = records[1].payload["targets"]["MEMORY.md"]["new"]["blob_digest"]
    blob = workspace / "memory" / ".recovery" / "blobs" / "sha256" / digest
    blob.write_bytes(b"tampered proposal bytes")
    return blob


def test_reconcile_is_typed_replayable_and_does_not_mutate_live_state(tmp_path: Path) -> None:
    workspace, sessions_root, memory, receipt = _workspace(
        tmp_path,
        memory_output="updated long-term memory\n",
    )
    _prepare(workspace, sessions_root)
    paths = _live_paths(workspace, sessions_root, memory, receipt)
    before = _live_state(paths)

    record = _reconcile(workspace, sessions_root)

    assert record.phase == "HISTORY_APPLIED"
    _, records = _transaction_chain(workspace)
    assert [item.phase for item in records] == ["CAPTURED", "PREPARED", "HISTORY_APPLIED"]
    _assert_live_state(paths, before)

    replay = _reconcile(workspace, sessions_root)

    assert replay.digest == record.digest
    _, replayed_records = _transaction_chain(workspace)
    assert [item.phase for item in replayed_records] == [
        "CAPTURED", "PREPARED", "HISTORY_APPLIED",
    ]
    _assert_live_state(paths, before)


@pytest.mark.parametrize(
    "tamper",
    [
        "session",
        "receipt-stat-copy",
        "receipt-byte-drift",
        "history-stat-copy",
        "cursor-stat-copy",
        "memory-stat-copy",
        "marker-duplicate",
        "marker-conflicting",
        "marker-missing",
        "proposal-blob",
    ],
)
def test_unproven_history_fails_without_live_mutations_or_phase_claim(
    tmp_path: Path, tamper: str,
) -> None:
    workspace, sessions_root, memory, receipt = _workspace(tmp_path)
    _prepare(workspace, sessions_root)
    paths = _live_paths(workspace, sessions_root, memory, receipt)
    session = paths[1]
    original_receipt = receipt.read_bytes()
    extra: Path | None = None

    if tamper == "session":
        _tamper_session(session)
    elif tamper == "receipt-stat-copy":
        _tamper_receipt_copy(receipt)
    elif tamper == "receipt-byte-drift":
        receipt.write_bytes(
            receipt.read_bytes().replace(b"Prepared receipt entry", b"Changed receipt entry")
        )
    elif tamper == "history-stat-copy":
        _tamper_target_copy(memory / "history.jsonl")
    elif tamper == "cursor-stat-copy":
        _tamper_target_copy(memory / ".cursor")
    elif tamper == "memory-stat-copy":
        _tamper_target_copy(memory / "MEMORY.md")
    elif tamper == "marker-duplicate":
        _tamper_marker(memory / "history.jsonl", mode="duplicate")
    elif tamper == "marker-conflicting":
        _tamper_marker(memory / "history.jsonl", mode="conflicting")
    elif tamper == "marker-missing":
        _tamper_marker(memory / "history.jsonl", mode="missing")
    else:
        extra = _tamper_proposal_blob(workspace)

    before = _live_state(paths)
    extra_before = extra.read_bytes() if extra is not None else None

    with pytest.raises(history.MemoryRecoveryTransactionHistoryError):
        _reconcile(workspace, sessions_root)

    _assert_live_state(paths, before)
    if extra is not None:
        assert extra.read_bytes() == extra_before
    _, records = _transaction_chain(workspace, receipt_bytes=original_receipt)
    assert [item.phase for item in records] == ["CAPTURED", "PREPARED"]


@pytest.mark.parametrize("interrupted_after_publish", [False, True])
def test_interrupted_history_append_is_replayable_without_live_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_after_publish: bool,
) -> None:
    workspace, sessions_root, memory, receipt = _workspace(tmp_path)
    _prepare(workspace, sessions_root)
    paths = _live_paths(workspace, sessions_root, memory, receipt)
    before = _live_state(paths)
    append = transaction_manifest.MemoryRecoveryManifest.append
    interrupted = False

    def interrupt(self, **kwargs):
        nonlocal interrupted
        if kwargs.get("phase") == "HISTORY_APPLIED" and not interrupted:
            interrupted = True
            if interrupted_after_publish:
                append(self, **kwargs)
            raise OSError("injected interrupted history append")
        return append(self, **kwargs)

    monkeypatch.setattr(transaction_manifest.MemoryRecoveryManifest, "append", interrupt)
    with pytest.raises(history.MemoryRecoveryTransactionHistoryError):
        _reconcile(workspace, sessions_root)
    assert interrupted
    _assert_live_state(paths, before)
    _, records = _transaction_chain(workspace)
    assert [item.phase for item in records] == (
        ["CAPTURED", "PREPARED", "HISTORY_APPLIED"]
        if interrupted_after_publish
        else ["CAPTURED", "PREPARED"]
    )

    monkeypatch.setattr(transaction_manifest.MemoryRecoveryManifest, "append", append)
    replay = _reconcile(workspace, sessions_root)

    assert replay.phase == "HISTORY_APPLIED"
    _, replayed_records = _transaction_chain(workspace)
    assert [item.phase for item in replayed_records] == [
        "CAPTURED", "PREPARED", "HISTORY_APPLIED",
    ]
    _assert_live_state(paths, before)


def test_manifest_tamper_fails_closed_without_live_mutation_or_successor(
    tmp_path: Path,
) -> None:
    workspace, sessions_root, memory, receipt = _workspace(tmp_path)
    _prepare(workspace, sessions_root)
    manifest, records = _transaction_chain(workspace)
    prepared_path = manifest._namespace.directory / f"next-{records[0].digest}.json"
    corrupted = prepared_path.read_bytes().replace(b"PREPARED", b"PREPARED-TAMPERED")
    prepared_path.write_bytes(corrupted)
    paths = _live_paths(workspace, sessions_root, memory, receipt)
    before = _live_state(paths)

    with pytest.raises(history.MemoryRecoveryTransactionHistoryError):
        _reconcile(workspace, sessions_root)

    _assert_live_state(paths, before)
    assert prepared_path.read_bytes() == corrupted
    assert not (manifest._namespace.directory / f"next-{records[1].digest}.json").exists()


def test_transaction_chain_beyond_history_prefix_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    workspace, sessions_root, memory, receipt = _workspace(tmp_path)
    _prepare(workspace, sessions_root)
    _, records = _transaction_chain(workspace)
    captured = transaction.decode_transaction_payload("CAPTURED", records[0].payload)
    prepared = transaction.decode_transaction_payload(
        "PREPARED", records[1].payload, captured=captured,
    )
    applied = transaction.history_applied_transaction_payload(captured, prepared)
    with storage.open_workspace_memory(workspace) as authority, authority.operation():
        manifest = transaction_manifest.MemoryRecoveryManifest(
            authority.open_evidence_namespace("transactions", captured.identity)
        )
        history_record = manifest.append(
            predecessor_digest=records[1].digest,
            phase="HISTORY_APPLIED",
            payload=applied.to_payload(),
        )
        memory_applied = transaction.memory_applied_transaction_payload(
            captured, prepared, applied,
        )
        manifest.append(
            predecessor_digest=history_record.digest,
            phase="MEMORY_APPLIED",
            payload=memory_applied.to_payload(),
        )
    paths = _live_paths(workspace, sessions_root, memory, receipt)
    before = _live_state(paths)
    _, recorded = _transaction_chain(workspace)
    phases_before = [item.phase for item in recorded]

    with pytest.raises(history.MemoryRecoveryTransactionHistoryError):
        _reconcile(workspace, sessions_root)

    _assert_live_state(paths, before)
    _, after = _transaction_chain(workspace)
    assert [item.phase for item in after] == phases_before


@pytest.mark.parametrize("drift", ["session", "receipt", "history"])
def test_append_boundary_drift_fails_closed_after_at_most_one_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    workspace, sessions_root, memory, receipt = _workspace(tmp_path)
    _prepare(workspace, sessions_root)
    paths = _live_paths(workspace, sessions_root, memory, receipt)
    before = _live_state(paths)
    original_receipt = receipt.read_bytes()
    target = {
        "session": paths[1],
        "receipt": paths[0],
        "history": paths[2],
    }[drift]
    original_append = transaction_manifest.MemoryRecoveryManifest.append
    injected = False

    def append_then_drift(self, **kwargs):
        nonlocal injected
        record = original_append(self, **kwargs)
        if kwargs.get("phase") == "HISTORY_APPLIED" and not injected:
            injected = True
            target.write_bytes(target.read_bytes() + b" ")
        return record

    monkeypatch.setattr(
        transaction_manifest.MemoryRecoveryManifest,
        "append",
        append_then_drift,
    )
    with pytest.raises(history.MemoryRecoveryTransactionHistoryError) as failure:
        _reconcile(workspace, sessions_root)

    assert str(failure.value) == "memory recovery transaction history proof is unavailable"
    assert injected
    after_drift = _live_state(paths)
    assert after_drift[target] != before[target]
    assert all(after_drift[path] == before[path] for path in paths if path != target)
    _, records = _transaction_chain(workspace, receipt_bytes=original_receipt)
    assert [item.phase for item in records] == ["CAPTURED", "PREPARED", "HISTORY_APPLIED"]

    with pytest.raises(history.MemoryRecoveryTransactionHistoryError):
        _reconcile(workspace, sessions_root)
    assert _live_state(paths) == after_drift
    _, replayed_records = _transaction_chain(workspace, receipt_bytes=original_receipt)
    assert [item.phase for item in replayed_records] == [
        "CAPTURED", "PREPARED", "HISTORY_APPLIED",
    ]
