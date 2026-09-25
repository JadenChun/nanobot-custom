"""Offline PREPARED proposals never mutate live receipt or session state."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent import memory_recovery_transaction_prepare as prepare
from nanobot.agent.memory_recovery_manifest import MemoryRecoveryManifest
from nanobot.session.manager import JsonlSessionStore

_WORKSPACE_ID = "a" * 32
_SESSION_KEY = "telegram:receipt-prepare"


def _session_events(*, end: int = 2) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "already summarized", "_event_id": "event-1"},
        {"role": "assistant", "content": "new summary", "_event_id": "event-2"},
    ][:end]


def _chunk_digest(events: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        events,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _session_bytes(
    *, revision: int = 7, generation: int = 3, cursor: int = 1, end: int = 2,
) -> bytes:
    header = {
        "_type": "metadata",
        "key": _SESSION_KEY,
        "created_at": "2026-09-25T00:00:00",
        "updated_at": "2026-09-25T00:01:00",
        "revision": revision,
        "generation": generation,
        "incarnation": "session-incarnation",
        "last_archived": cursor,
        "last_consolidated": cursor,
        "metadata": {"source_schema_version": 1},
    }
    events = _session_events(end=end)
    if cursor > len(events):
        header["last_archived"] = header["last_consolidated"] = len(events)
    return b"".join(
        (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        for record in (header, *events)
    )


def _write_session(workspace: Path, sessions_root: Path, content: bytes) -> Path:
    state = workspace / ".nanobot"
    state.mkdir(mode=0o700)
    (state / "workspace-id").write_text(f"{_WORKSPACE_ID}\n", encoding="ascii")
    (state / "workspace-id").chmod(0o600)
    namespace = sessions_root / _WORKSPACE_ID
    namespace.mkdir(mode=0o700)
    (namespace / ".workspace").write_text(f"{workspace.resolve()}\n", encoding="utf-8")
    (namespace / ".workspace").chmod(0o600)
    path = namespace / f"{JsonlSessionStore.storage_key(_SESSION_KEY)}.jsonl"
    path.write_bytes(content)
    path.chmod(0o600)
    return path


def _workspace(
    tmp_path: Path,
    *,
    memory_output: str | None = None,
    reference_only: bool | None = None,
    jsonl: bytes | None = None,
):
    from nanobot.agent import memory_recovery_conversion_apply as apply_conversion
    from nanobot.agent import memory_recovery_conversion_finalize as finalize_conversion
    from nanobot.agent import memory_recovery_conversion_runtime as prepare_conversion

    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir(mode=0o700)
    memory = workspace / "memory"
    memory.mkdir()
    (memory / "HISTORY.md").write_bytes(b"[2026-09-24 12:00] Prior note.\n\n")
    (memory / "MEMORY.md").write_bytes(b"base memory\n")
    base_digest = hashlib.sha256(b"base memory\n").hexdigest()
    output_digest = None
    if memory_output is not None:
        output_digest = hashlib.sha256(memory_output.encode("utf-8")).hexdigest()
    if reference_only is None:
        reference_only = memory_output is None
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
    if jsonl is not None:
        (memory / "history.jsonl").write_bytes(jsonl)
    prepare_conversion.prepare_receipt_tail_conversion(workspace, workspace_id=_WORKSPACE_ID)
    apply_conversion.apply_prepared_conversion_targets(workspace, workspace_id=_WORKSPACE_ID)
    finalize_conversion.finalize_prepared_conversion(workspace, workspace_id=_WORKSPACE_ID)
    return workspace, sessions_root, memory, receipt_path


def _prepare(workspace: Path, sessions_root: Path):
    return prepare.prepare_memory_recovery_transaction(
        workspace, workspace_id=_WORKSPACE_ID, sessions_root=sessions_root,
    )


def _transaction_chain(
    workspace: Path, *, receipt_bytes: bytes | None = None,
) -> tuple[MemoryRecoveryManifest, tuple[Any, ...]]:
    from nanobot.agent import memory_recovery as receipt_codec
    from nanobot.agent import memory_recovery_storage as storage
    from nanobot.agent import memory_recovery_transaction as transaction

    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        raw = receipt_bytes
        if raw is None:
            raw = authority.snapshot_target(".consolidation.pending.json").content
        assert raw is not None
        receipt = receipt_codec._validate_payload(receipt_codec._decode(raw), raw)
        captured = transaction.capture_transaction_payload(
            receipt,
            workspace_id=_WORKSPACE_ID,
            workspace_identity=operation.workspace_identity,
        )
        manifest = MemoryRecoveryManifest(
            authority.open_evidence_namespace("transactions", captured.identity)
        )
        return manifest, manifest.load()


def _identity(path: Path) -> tuple[int, int, int, int, int, int, int, int, int]:
    value = path.lstat()
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
        value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns,
    )


def test_source_chunk_digest_mismatch_fails_before_prepared_without_writes(
    tmp_path: Path,
) -> None:
    workspace, sessions_root, memory, receipt_path = _workspace(tmp_path)
    session_path = sessions_root / _WORKSPACE_ID / f"{JsonlSessionStore.storage_key(_SESSION_KEY)}.jsonl"
    records = [
        json.loads(line)
        for line in session_path.read_text(encoding="utf-8").splitlines()
    ]
    records[2]["content"] = "changed with the same generation, revision, cursor, and offsets"
    session_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    session_path.chmod(0o600)

    live_targets = (
        receipt_path,
        session_path,
        memory / "history.jsonl",
        memory / ".cursor",
        memory / "MEMORY.md",
    )
    before_identity = {path: _identity(path) for path in live_targets}
    before_bytes = {path: path.read_bytes() for path in live_targets}

    with pytest.raises(prepare.MemoryRecoveryTransactionPreparationError):
        _prepare(workspace, sessions_root)

    assert {path: _identity(path) for path in live_targets} == before_identity
    assert {path: path.read_bytes() for path in live_targets} == before_bytes
    _, transaction_records = _transaction_chain(workspace)
    assert [item.phase for item in transaction_records] == ["CAPTURED"]


def test_prepared_proposal_is_deterministic_and_does_not_apply_it(tmp_path: Path) -> None:
    workspace, sessions_root, memory, receipt_path = _workspace(
        tmp_path, memory_output="updated long-term memory\n",
    )
    session_path = sessions_root / _WORKSPACE_ID / f"{JsonlSessionStore.storage_key(_SESSION_KEY)}.jsonl"
    before = {path: _identity(path) for path in (
        receipt_path, session_path, memory / "history.jsonl", memory / ".cursor", memory / "MEMORY.md",
    )}
    before_bytes = {path: path.read_bytes() for path in before}

    record = _prepare(workspace, sessions_root)

    assert record.phase == "PREPARED"
    assert {path: _identity(path) for path in before} == before
    assert {path: path.read_bytes() for path in before} == before_bytes
    _, records = _transaction_chain(workspace)
    assert [item.phase for item in records] == ["CAPTURED", "PREPARED"]
    prepared = records[-1].payload
    assert prepared["receipt_stat"]["inode"] == before[receipt_path][1]
    assert prepared["source"]["old_revision"] == 7
    assert prepared["source"]["new_revision"] == 8
    assert prepared["source"]["old_generation"] == prepared["source"]["new_generation"] == 3
    assert prepared["source"]["old_cursor"] == 1
    assert prepared["source"]["new_cursor"] == 2
    assert prepared["source"]["action"] == "advance"
    assert prepared["memory_action"] == "replace"
    from nanobot.agent import memory_recovery_storage as storage

    with storage.open_workspace_memory(workspace) as authority, authority.operation():
        assert authority.read_blob(
            prepared["targets"]["MEMORY.md"]["new"]["blob_digest"]
        ) == b"updated long-term memory\n"

    assert _prepare(workspace, sessions_root).digest == record.digest
    _, replayed_records = _transaction_chain(workspace)
    assert [item.phase for item in replayed_records] == ["CAPTURED", "PREPARED"]


def test_prepared_replay_rejects_byte_identical_receipt_with_new_stat(tmp_path: Path) -> None:
    workspace, sessions_root, memory, receipt_path = _workspace(tmp_path)
    _prepare(workspace, sessions_root)
    raw = receipt_path.read_bytes()
    replacement = memory / "receipt-copy"
    replacement.write_bytes(raw)
    replacement.chmod(0o600)
    os.replace(replacement, receipt_path)

    with pytest.raises(prepare.MemoryRecoveryTransactionPreparationError):
        _prepare(workspace, sessions_root)

    _, records = _transaction_chain(workspace)
    assert [item.phase for item in records] == ["CAPTURED", "PREPARED"]


@pytest.mark.parametrize(
    ("revision", "generation", "cursor"),
    [(8, 3, 1), (7, 4, 1), (7, 3, 0)],
)
def test_source_state_drift_after_prepared_fails_closed(
    tmp_path: Path, revision: int, generation: int, cursor: int,
) -> None:
    workspace, sessions_root, _, _ = _workspace(tmp_path)
    _prepare(workspace, sessions_root)
    session_path = sessions_root / _WORKSPACE_ID / f"{JsonlSessionStore.storage_key(_SESSION_KEY)}.jsonl"
    session_path.write_bytes(_session_bytes(revision=revision, generation=generation, cursor=cursor))
    session_path.chmod(0o600)

    with pytest.raises(prepare.MemoryRecoveryTransactionPreparationError):
        _prepare(workspace, sessions_root)

    _, records = _transaction_chain(workspace)
    assert [item.phase for item in records] == ["CAPTURED", "PREPARED"]



def test_identical_receipt_copy_before_prepared_is_accepted(tmp_path: Path) -> None:
    workspace, sessions_root, memory, receipt_path = _workspace(tmp_path)
    raw = receipt_path.read_bytes()
    original_inode = receipt_path.stat().st_ino
    replacement = memory / "receipt-copy"
    replacement.write_bytes(raw)
    replacement.chmod(0o600)
    os.replace(replacement, receipt_path)
    assert receipt_path.stat().st_ino != original_inode

    record = _prepare(workspace, sessions_root)

    assert record.phase == "PREPARED"
    _, records = _transaction_chain(workspace)
    assert [item.phase for item in records] == ["CAPTURED", "PREPARED"]
    assert records[-1].payload["receipt_stat"]["inode"] == receipt_path.stat().st_ino


def test_changed_receipt_bytes_fail_without_a_prepared_record(tmp_path: Path) -> None:
    workspace, sessions_root, memory, receipt_path = _workspace(tmp_path)
    original_receipt = receipt_path.read_bytes()
    receipt = json.loads(original_receipt)
    receipt["created_at"] = "changed receipt content"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    receipt_path.chmod(0o600)

    with pytest.raises(prepare.MemoryRecoveryTransactionPreparationError) as raised:
        _prepare(workspace, sessions_root)

    assert str(raised.value) == "memory recovery transaction preparation is unavailable"
    assert "changed receipt content" not in str(raised.value)
    _, records = _transaction_chain(workspace, receipt_bytes=original_receipt)
    assert [item.phase for item in records] == ["CAPTURED"]


def test_source_proposal_is_blob_bound_and_targets_are_no_op_proposals(tmp_path: Path) -> None:
    workspace, sessions_root, memory, _ = _workspace(tmp_path)
    record = _prepare(workspace, sessions_root)
    payload = record.payload
    source_path = sessions_root / _WORKSPACE_ID / f"{JsonlSessionStore.storage_key(_SESSION_KEY)}.jsonl"
    current_session = source_path.read_bytes()

    from nanobot.agent import memory_recovery_storage as storage

    with storage.open_workspace_memory(workspace) as authority, authority.operation():
        proposed = authority.read_blob(payload["source"]["new_digest"])
        rows = [json.loads(line) for line in proposed.splitlines()]
        assert rows[0]["revision"] == 8
        assert rows[0]["generation"] == 3
        assert rows[0]["last_archived"] == rows[0]["last_consolidated"] == 2
        assert rows[1:] == [json.loads(line) for line in current_session.splitlines()][1:]
        for name in ("history.jsonl", ".cursor"):
            target = payload["targets"][name]
            assert target["old"]["digest"] == target["new"]["blob_digest"]
            assert authority.read_blob(target["new"]["blob_digest"]) == (memory / name).read_bytes()
        memory_target = payload["targets"]["MEMORY.md"]
        assert authority.read_blob(memory_target["new"]["blob_digest"]) == b"base memory\n"


def test_reference_only_string_output_preserves_memory_target(tmp_path: Path) -> None:
    workspace, sessions_root, memory, _ = _workspace(
        tmp_path,
        memory_output="referenced text not applied",
        reference_only=True,
    )
    original = (memory / "MEMORY.md").read_bytes()

    record = _prepare(workspace, sessions_root)

    target = record.payload["targets"]["MEMORY.md"]
    assert record.payload["memory_action"] == "preserve"
    assert target["old"]["digest"] == target["new"]["blob_digest"]
    assert (memory / "MEMORY.md").read_bytes() == original


def test_marker_corruption_fails_before_prepared(tmp_path: Path, *, duplicate: bool = False) -> None:
    workspace, sessions_root, memory, _ = _workspace(tmp_path)
    history = memory / "history.jsonl"
    raw = history.read_bytes()
    if duplicate:
        history.write_bytes(raw + raw.splitlines(keepends=True)[-1])
    else:
        rows = [json.loads(line) for line in raw.splitlines()]
        rows = [row for row in rows if "receipt-marker-prepare" not in row.get("content", "")]
        history.write_bytes(b"".join((json.dumps(row) + "\n").encode() for row in rows))

    with pytest.raises(prepare.MemoryRecoveryTransactionPreparationError):
        _prepare(workspace, sessions_root)

    _, records = _transaction_chain(workspace)
    assert [item.phase for item in records] == ["CAPTURED"]


@pytest.mark.parametrize("crash_after", ["blob", "manifest"])
def test_publication_crash_is_replayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_after: str,
) -> None:
    from nanobot.agent import memory_recovery_storage as storage

    workspace, sessions_root, _, _ = _workspace(tmp_path)
    if crash_after == "blob":
        original = storage.WorkspaceMemoryAuthority.publish_blob
        crashed = False

        def crash_after_blob(authority, content):
            nonlocal crashed
            digest = original(authority, content)
            if not crashed:
                crashed = True
                raise OSError("simulated interruption")
            return digest

        monkeypatch.setattr(storage.WorkspaceMemoryAuthority, "publish_blob", crash_after_blob)
        with pytest.raises(prepare.MemoryRecoveryTransactionPreparationError):
            _prepare(workspace, sessions_root)
        monkeypatch.setattr(storage.WorkspaceMemoryAuthority, "publish_blob", original)
        assert crashed
    else:
        original = MemoryRecoveryManifest.append
        crashed = False

        def crash_after_manifest(manifest, **kwargs):
            nonlocal crashed
            record = original(manifest, **kwargs)
            if kwargs["phase"] == "PREPARED" and not crashed:
                crashed = True
                raise OSError("simulated interruption")
            return record

        monkeypatch.setattr(MemoryRecoveryManifest, "append", crash_after_manifest)
        with pytest.raises(prepare.MemoryRecoveryTransactionPreparationError):
            _prepare(workspace, sessions_root)
        monkeypatch.setattr(MemoryRecoveryManifest, "append", original)
        assert crashed

    _, records = _transaction_chain(workspace)
    if crash_after == "blob":
        assert [item.phase for item in records] == ["CAPTURED"]
    else:
        assert [item.phase for item in records] == ["CAPTURED", "PREPARED"]
    assert _prepare(workspace, sessions_root).phase == "PREPARED"
    _, records = _transaction_chain(workspace)
    assert [item.phase for item in records] == ["CAPTURED", "PREPARED"]


def test_source_content_digest_drift_after_prepared_fails_closed(tmp_path: Path) -> None:
    workspace, sessions_root, _, _ = _workspace(tmp_path)
    _prepare(workspace, sessions_root)
    session_path = sessions_root / _WORKSPACE_ID / f"{JsonlSessionStore.storage_key(_SESSION_KEY)}.jsonl"
    records = [json.loads(line) for line in session_path.read_text(encoding="utf-8").splitlines()]
    records[2]["content"] = "changed with the same generation, revision, and cursor"
    session_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    session_path.chmod(0o600)

    with pytest.raises(prepare.MemoryRecoveryTransactionPreparationError):
        _prepare(workspace, sessions_root)

    _, records = _transaction_chain(workspace)
    assert [item.phase for item in records] == ["CAPTURED", "PREPARED"]


def test_unsafe_session_source_path_fails_before_prepared(tmp_path: Path) -> None:
    workspace, sessions_root, _, _ = _workspace(tmp_path)
    session_path = sessions_root / _WORKSPACE_ID / f"{JsonlSessionStore.storage_key(_SESSION_KEY)}.jsonl"
    moved = tmp_path / "moved-source.jsonl"
    os.replace(session_path, moved)
    session_path.symlink_to(moved)

    with pytest.raises(prepare.MemoryRecoveryTransactionPreparationError):
        _prepare(workspace, sessions_root)

    _, records = _transaction_chain(workspace)
    assert [item.phase for item in records] == ["CAPTURED"]
    assert moved.read_bytes() == _session_bytes()


def test_source_change_during_preparation_fails_before_prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nanobot.session import checked_source

    workspace, sessions_root, _, _ = _workspace(tmp_path)
    session_path = sessions_root / _WORKSPACE_ID / f"{JsonlSessionStore.storage_key(_SESSION_KEY)}.jsonl"
    original = checked_source.CheckedSessionAuthority.read_source_bytes
    changed = False

    def change_after_read(authority):
        nonlocal changed
        content = original(authority)
        if not changed:
            changed = True
            session_path.write_bytes(_session_bytes(revision=8))
            session_path.chmod(0o600)
        return content

    monkeypatch.setattr(checked_source.CheckedSessionAuthority, "read_source_bytes", change_after_read)
    with pytest.raises(prepare.MemoryRecoveryTransactionPreparationError):
        _prepare(workspace, sessions_root)

    assert changed
    _, records = _transaction_chain(workspace)
    assert [item.phase for item in records] == ["CAPTURED"]


def test_corrupt_completed_conversion_manifest_fails_before_prepared(tmp_path: Path) -> None:
    from nanobot.agent import memory_recovery as receipt_codec
    from nanobot.agent import memory_recovery_conversion as conversion
    from nanobot.agent import memory_recovery_storage as storage
    from nanobot.agent import memory_recovery_transaction as transaction

    workspace, sessions_root, memory, receipt_path = _workspace(tmp_path)
    with storage.open_workspace_memory(workspace) as authority, authority.operation() as operation:
        raw = authority.snapshot_target(".consolidation.pending.json").content
        assert raw is not None
        receipt = receipt_codec._validate_payload(receipt_codec._decode(raw), raw)
        captured = transaction.capture_transaction_payload(
            receipt,
            workspace_id=_WORKSPACE_ID,
            workspace_identity=operation.workspace_identity,
        )
        v2 = conversion.capture_receipt_tail_conversion_payload(captured.identity)
        namespace = authority.open_evidence_namespace("conversions", v2.identity)
        complete_paths = []
        for path in namespace.directory.glob("next-*.json"):
            value = json.loads(path.read_bytes())
            if value.get("phase") == "COMPLETE":
                complete_paths.append(path)
        assert len(complete_paths) == 1
        complete_paths[0].write_bytes(b"corrupt conversion evidence")

    with pytest.raises(prepare.MemoryRecoveryTransactionPreparationError):
        _prepare(workspace, sessions_root)

    _, records = _transaction_chain(workspace, receipt_bytes=receipt_path.read_bytes())
    assert [item.phase for item in records] == ["CAPTURED"]
    assert (memory / ".consolidation.pending.json").exists()

def test_nonempty_jsonl_conversion_policy_is_rejected_without_prepared(tmp_path: Path) -> None:
    workspace, sessions_root, memory, _ = _workspace(tmp_path, jsonl=b'{"cursor":999}\n')
    targets_before = {
        name: (memory / name).read_bytes() if (memory / name).exists() else None
        for name in ("history.jsonl", ".cursor", ".dream_cursor", "MEMORY.md")
    }

    with pytest.raises(prepare.MemoryRecoveryTransactionPreparationError):
        _prepare(workspace, sessions_root)

    _, records = _transaction_chain(workspace)
    assert [item.phase for item in records] == ["CAPTURED"]
    assert targets_before == {
        name: (memory / name).read_bytes() if (memory / name).exists() else None
        for name in targets_before
    }
