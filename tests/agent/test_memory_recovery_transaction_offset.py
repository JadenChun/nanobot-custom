"""OFFSET_APPLIED publishes only after a checked, durable source proof."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_transaction_history as history
from nanobot.agent import memory_recovery_transaction_memory as memory_stage
from nanobot.agent import memory_recovery_transaction_offset as offset_stage
from nanobot.session.manager import JsonlSessionStore
from tests.agent.test_memory_recovery_transaction_memory import _workspace_without_memory
from tests.agent.test_memory_recovery_transaction_prepare import (
    _SESSION_KEY,
    _WORKSPACE_ID,
    _prepare,
    _transaction_chain,
    _workspace,
)


def _setup(tmp_path: Path, *, output: str | None = "new memory\n", reference_only: bool = False, absent: bool = False):
    fixture = _workspace_without_memory if absent else _workspace
    workspace, sessions, memory, receipt = fixture(
        tmp_path, memory_output=output, reference_only=reference_only,
    )
    _prepare(workspace, sessions)
    history.reconcile_memory_recovery_transaction_history(
        workspace, workspace_id=_WORKSPACE_ID, sessions_root=sessions,
    )
    memory_stage.reconcile_memory_recovery_transaction_memory(
        workspace, workspace_id=_WORKSPACE_ID, sessions_root=sessions,
    )
    source = sessions / _WORKSPACE_ID / f"{JsonlSessionStore.storage_key(_SESSION_KEY)}.jsonl"
    return workspace, sessions, memory, receipt, source


def _reconcile(workspace: Path, sessions: Path):
    return offset_stage.reconcile_memory_recovery_transaction_offset(
        workspace, workspace_id=_WORKSPACE_ID, sessions_root=sessions,
    )


def _phases(workspace: Path):
    return [record.phase for record in _transaction_chain(workspace)[1]]


def test_old_to_new_durable_replay_and_no_second_mutation(tmp_path: Path, monkeypatch):
    workspace, sessions, _, _, source = _setup(tmp_path)
    _, records = _transaction_chain(workspace)
    proposal = records[1].payload["source"]
    old = source.read_bytes()
    calls = 0
    original = offset_stage.open_checked_recovery_session_for_replace

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(offset_stage, "open_checked_recovery_session_for_replace", counted)
    record = _reconcile(workspace, sessions)
    assert record.phase == "OFFSET_APPLIED"
    assert source.read_bytes() != old
    assert len(source.read_bytes()) == proposal["new_size"]
    assert _phases(workspace) == [
        "CAPTURED", "PREPARED", "HISTORY_APPLIED", "MEMORY_APPLIED", "OFFSET_APPLIED",
    ]
    identity = source.stat()
    assert _reconcile(workspace, sessions) == record
    assert (source.stat().st_ino, source.stat().st_mtime_ns) == (identity.st_ino, identity.st_mtime_ns)
    assert calls == 1


def test_existing_new_before_publication_replays_without_replace(tmp_path: Path, monkeypatch):
    workspace, sessions, _, _, source = _setup(tmp_path)
    _, records = _transaction_chain(workspace)
    from nanobot.agent import memory_recovery_storage as storage

    with storage.open_workspace_memory(workspace) as authority:
        proposed = authority.read_blob(records[1].payload["source"]["new_digest"])
    assert proposed is not None
    temporary = source.with_suffix(".new")
    temporary.write_bytes(proposed)
    temporary.chmod(0o600)
    os.replace(temporary, source)

    def forbidden(*args, **kwargs):
        pytest.fail("source mutation on existing-new replay")

    monkeypatch.setattr(offset_stage, "open_checked_recovery_session_for_replace", forbidden)
    assert _reconcile(workspace, sessions).phase == "OFFSET_APPLIED"


def test_source_drift_and_receipt_drift_fail_before_phase(tmp_path: Path):
    workspace, sessions, _, receipt, source = _setup(tmp_path)
    source.write_bytes(source.read_bytes() + b" \n")
    with pytest.raises(offset_stage.MemoryRecoveryTransactionOffsetError):
        _reconcile(workspace, sessions)
    assert _phases(workspace)[-1] == "MEMORY_APPLIED"
    # Restore exact old bytes via the fixture source header is unnecessary; independent fixture.
    (tmp_path / "other").mkdir()
    other, other_sessions, _, other_receipt, _ = _setup(tmp_path / "other")
    before_phases = _phases(other)
    other_receipt.write_bytes(other_receipt.read_bytes() + b" ")
    with pytest.raises(offset_stage.MemoryRecoveryTransactionOffsetError):
        _reconcile(other, other_sessions)
    assert before_phases[-1] == "MEMORY_APPLIED"


def test_reference_only_absent_memory_and_noop(tmp_path: Path):
    workspace, sessions, memory, _, _ = _setup(
        tmp_path, output=None, reference_only=True, absent=True,
    )
    assert not (memory / "MEMORY.md").exists()
    assert _reconcile(workspace, sessions).phase == "OFFSET_APPLIED"


def test_corrupt_memory_fails_without_source_mutation(tmp_path: Path):
    workspace, sessions, memory, _, source = _setup(tmp_path)
    before = source.read_bytes()
    # A malformed MEMORY target suffices to prove pre-source refusal.
    (memory / "MEMORY.md").write_bytes(b"corrupt")
    with pytest.raises(offset_stage.MemoryRecoveryTransactionOffsetError):
        _reconcile(workspace, sessions)
    assert source.read_bytes() == before
    assert _phases(workspace)[-1] == "MEMORY_APPLIED"


def test_same_byte_inode_swap_during_replace_refuses_publication(tmp_path: Path, monkeypatch):
    workspace, sessions, _, _, source = _setup(tmp_path)
    original = offset_stage.open_checked_recovery_session_for_replace

    class Swapped:
        def __init__(self, delegate):
            self.delegate = delegate

        def __enter__(self):
            self.checked = self.delegate.__enter__()
            return self

        def __exit__(self, *args):
            return self.delegate.__exit__(*args)

        def read_source_bytes(self):
            raw = self.checked.read_source_bytes()
            replacement = source.with_suffix(".swap")
            replacement.write_bytes(raw)
            replacement.chmod(0o600)
            os.replace(replacement, source)
            return raw

        def replace_recovery_source_bytes(self, proposed):
            return self.checked.replace_recovery_source_bytes(proposed)

    monkeypatch.setattr(
        offset_stage, "open_checked_recovery_session_for_replace",
        lambda *args, **kwargs: Swapped(original(*args, **kwargs)),
    )
    with pytest.raises(offset_stage.MemoryRecoveryTransactionOffsetError):
        _reconcile(workspace, sessions)
    assert _phases(workspace)[-1] == "MEMORY_APPLIED"


def test_post_rename_uncertainty_reproves_and_publishes_once(tmp_path: Path, monkeypatch):
    workspace, sessions, _, _, _ = _setup(tmp_path)
    original = offset_stage.open_checked_recovery_session_for_replace

    class Uncertain:
        def __init__(self, delegate):
            self.delegate = delegate

        def __enter__(self):
            self.checked = self.delegate.__enter__()
            return self

        def __exit__(self, *args):
            return self.delegate.__exit__(*args)

        def read_source_bytes(self):
            return self.checked.read_source_bytes()

        def replace_recovery_source_bytes(self, proposed):
            self.checked.replace_recovery_source_bytes(proposed)
            raise OSError("after rename")

    monkeypatch.setattr(
        offset_stage, "open_checked_recovery_session_for_replace",
        lambda *args, **kwargs: Uncertain(original(*args, **kwargs)),
    )
    assert _reconcile(workspace, sessions).phase == "OFFSET_APPLIED"
    assert _phases(workspace).count("OFFSET_APPLIED") == 1


def test_pre_rename_failure_keeps_old_and_does_not_publish(tmp_path: Path, monkeypatch):
    workspace, sessions, _, _, source = _setup(tmp_path)
    old = source.read_bytes()
    original = offset_stage.open_checked_recovery_session_for_replace

    class Failed:
        def __init__(self, delegate):
            self.delegate = delegate

        def __enter__(self):
            self.checked = self.delegate.__enter__()
            return self

        def __exit__(self, *args):
            return self.delegate.__exit__(*args)

        def read_source_bytes(self):
            return self.checked.read_source_bytes()

        def replace_recovery_source_bytes(self, proposed):
            raise OSError("before rename")

    monkeypatch.setattr(
        offset_stage, "open_checked_recovery_session_for_replace",
        lambda *args, **kwargs: Failed(original(*args, **kwargs)),
    )
    with pytest.raises(offset_stage.MemoryRecoveryTransactionOffsetError):
        _reconcile(workspace, sessions)
    assert source.read_bytes() == old
    assert _phases(workspace)[-1] == "MEMORY_APPLIED"


def test_append_before_publication_failure_replays_new_without_replacement(
    tmp_path: Path, monkeypatch,
):
    workspace, sessions, _, _, source = _setup(tmp_path)
    from nanobot.agent.memory_recovery_manifest import MemoryRecoveryManifest

    original = MemoryRecoveryManifest.append
    calls = 0

    def fail_once(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("append before publication")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MemoryRecoveryManifest, "append", fail_once)
    with pytest.raises(offset_stage.MemoryRecoveryTransactionOffsetError):
        _reconcile(workspace, sessions)
    first_inode = source.stat().st_ino
    assert _phases(workspace)[-1] == "MEMORY_APPLIED"
    assert _reconcile(workspace, sessions).phase == "OFFSET_APPLIED"
    assert source.stat().st_ino == first_inode


def test_append_after_publication_failure_replays_existing_phase(tmp_path: Path, monkeypatch):
    workspace, sessions, _, _, source = _setup(tmp_path)
    from nanobot.agent.memory_recovery_manifest import MemoryRecoveryManifest

    original = MemoryRecoveryManifest.append
    calls = 0

    def fail_once(self, *args, **kwargs):
        nonlocal calls
        record = original(self, *args, **kwargs)
        calls += 1
        if calls == 1:
            raise OSError("append after publication")
        return record

    monkeypatch.setattr(MemoryRecoveryManifest, "append", fail_once)
    with pytest.raises(offset_stage.MemoryRecoveryTransactionOffsetError):
        _reconcile(workspace, sessions)
    first_inode = source.stat().st_ino
    assert _phases(workspace)[-1] == "OFFSET_APPLIED"
    assert _reconcile(workspace, sessions).phase == "OFFSET_APPLIED"
    assert source.stat().st_ino == first_inode
    assert calls == 1


def test_missing_source_proposal_blob_fails_before_mutation(tmp_path: Path):
    workspace, sessions, _, _, source = _setup(tmp_path)
    _, records = _transaction_chain(workspace)
    digest = records[1].payload["source"]["new_digest"]
    blob = workspace / "memory" / ".recovery" / "blobs" / "sha256" / digest
    assert blob.is_file()
    blob.unlink()
    before = source.read_bytes()
    with pytest.raises(offset_stage.MemoryRecoveryTransactionOffsetError):
        _reconcile(workspace, sessions)
    assert source.read_bytes() == before
    assert _phases(workspace)[-1] == "MEMORY_APPLIED"


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_durable_source_verify_fsync_fault_blocks_phase(tmp_path: Path, monkeypatch, kind: str):
    workspace, sessions, _, _, source = _setup(tmp_path)
    import nanobot.session.checked_source as checked_source

    original = checked_source.os.fsync
    hits: list[str] = []

    def fault(fd: int):
        import inspect
        import stat

        info = os.fstat(fd)
        in_verify = any(frame.function == "verify_recovery_source_bytes" for frame in inspect.stack())
        if in_verify and (
            (kind == "file" and stat.S_ISREG(info.st_mode))
            or (kind == "directory" and stat.S_ISDIR(info.st_mode))
        ):
            hits.append(kind)
            raise OSError("source durability fault")
        return original(fd)

    monkeypatch.setattr(checked_source.os, "fsync", fault)
    with pytest.raises(offset_stage.MemoryRecoveryTransactionOffsetError):
        _reconcile(workspace, sessions)
    monkeypatch.setattr(checked_source.os, "fsync", original)
    assert hits
    assert _phases(workspace)[-1] == "MEMORY_APPLIED"
    assert _reconcile(workspace, sessions).phase == "OFFSET_APPLIED"


def test_no_network_needed(tmp_path: Path, monkeypatch):
    workspace, sessions, _, _, _ = _setup(tmp_path)
    import socket

    def denied(*args, **kwargs):
        raise AssertionError("network access")

    monkeypatch.setattr(socket, "socket", denied)
    assert _reconcile(workspace, sessions).phase == "OFFSET_APPLIED"


def test_same_output_memory_noop_is_authorized(tmp_path: Path):
    workspace, sessions, memory, _, _ = _setup(tmp_path, output="base memory\n")
    before = (memory / "MEMORY.md").stat().st_ino
    assert _reconcile(workspace, sessions).phase == "OFFSET_APPLIED"
    assert (memory / "MEMORY.md").stat().st_ino == before


@pytest.mark.parametrize("name", ["history.jsonl", ".cursor"])
def test_corrupt_durable_history_blocks_source(tmp_path: Path, name: str):
    workspace, sessions, memory, _, source = _setup(tmp_path)
    old = source.read_bytes()
    (memory / name).write_bytes(b"corrupt")
    with pytest.raises(offset_stage.MemoryRecoveryTransactionOffsetError):
        _reconcile(workspace, sessions)
    assert source.read_bytes() == old
    assert _phases(workspace)[-1] == "MEMORY_APPLIED"
