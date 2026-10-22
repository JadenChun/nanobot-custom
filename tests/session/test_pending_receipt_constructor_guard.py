"""Pending receipts and unsafe memory paths block ordinary session-store construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import nanobot.session.manager as session_manager_module
from nanobot.agent.memory_recovery import (
    MemoryReceiptRecoveryRequiredError,
    require_memory_receipt_recovery,
)
from nanobot.session.manager import JsonlSessionStore, SessionManager, SessionStore

_RECEIPT_NAME = ".consolidation.pending.json"
_SECRET = "constructor-guard-memory-secret"


def _receipt_bytes() -> bytes:
    memory_output = f"# Memory\n{_SECRET}\n"
    payload = {
        "version": 1,
        "chunk_digest": "a" * 64,
        "history_marker_id": "marker-1",
        "history_entry": "[2026-09-18 12:00] recovered summary",
        "memory_output": memory_output,
        "base_memory_digest": "b" * 64,
        "output_memory_digest": hashlib.sha256(memory_output.encode("utf-8")).hexdigest(),
        "reference_only": False,
        "created_at": "2026-09-18T12:00:00+00:00",
        "session_key": "cli:receipt",
        "start_offset": 0,
        "end_offset": 2,
        "session_generation": 4,
    }
    return json.dumps(payload).encode("utf-8")


def _session_bytes(key: str, content: str, updated_at: str = "2026-09-18T00:00:00") -> bytes:
    metadata = {
        "_type": "metadata",
        "key": key,
        "created_at": "2026-09-17T00:00:00",
        "updated_at": updated_at,
        "metadata": {},
        "last_consolidated": 0,
    }
    return (
        json.dumps(metadata, separators=(",", ":"))
        + "\n"
        + json.dumps({"role": "user", "content": content}, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _stat_snapshot(path: Path, *, follow_symlinks: bool = True) -> tuple[int, ...]:
    info = path.stat() if follow_symlinks else path.lstat()
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
    )


def _set_up_blocked_workspace(
    tmp_path: Path,
    receipt_kind: str,
) -> tuple[Path, Path, Path, Path | None]:
    workspace = tmp_path / "workspace"
    memory_dir = workspace / "memory"
    legacy_dir = workspace / "sessions"
    memory_dir.mkdir(parents=True)
    legacy_dir.mkdir()

    key = "telegram:legacy"
    legacy_session = legacy_dir / f"{JsonlSessionStore.storage_key(key)}.jsonl"
    legacy_session.write_bytes(_session_bytes(key, "legacy session"))

    receipt = memory_dir / _RECEIPT_NAME
    target: Path | None = None
    raw = _receipt_bytes()
    if receipt_kind == "malformed":
        raw = b'{"memory_output":"private malformed payload",broken'
    if receipt_kind == "symlink":
        target = tmp_path / "receipt-target.json"
        target.write_bytes(raw)
        target.chmod(0o600)
        receipt.symlink_to(target)
    else:
        receipt.write_bytes(raw)
        if receipt_kind == "unsafe":
            receipt.chmod(0o644)
    return workspace, legacy_session, receipt, target


@pytest.mark.parametrize("constructor", ["jsonl", "manager"])
@pytest.mark.parametrize("receipt_kind", ["valid", "malformed", "symlink", "unsafe"])
def test_constructors_reject_pending_receipts_before_workspace_or_session_mutation(
    tmp_path: Path,
    constructor: str,
    receipt_kind: str,
) -> None:
    workspace, legacy_session, receipt, target = _set_up_blocked_workspace(
        tmp_path,
        receipt_kind,
    )
    sessions_root = tmp_path / "private-sessions"
    memory_dir = workspace / "memory"
    legacy_dir = workspace / "sessions"
    paths = [workspace, memory_dir, legacy_dir, legacy_session, receipt]
    if target is not None:
        paths.append(target)
    before = {
        path: (
            _stat_snapshot(path, follow_symlinks=path != receipt or target is None),
            path.read_bytes() if path.is_file() and not path.is_symlink() else None,
        )
        for path in paths
    }

    with pytest.raises(MemoryReceiptRecoveryRequiredError) as raised:
        if constructor == "jsonl":
            JsonlSessionStore(workspace, sessions_root=sessions_root)
        else:
            SessionManager(workspace, sessions_root=sessions_root)

    assert str(raised.value) == "memory recovery required"
    assert raised.value.args == ("memory recovery required",)
    assert _SECRET not in str(raised.value)
    assert not sessions_root.exists()
    assert not (workspace / ".nanobot").exists()
    for path, (stat_before, contents_before) in before.items():
        follows_link = path != receipt or target is None
        assert _stat_snapshot(path, follow_symlinks=follows_link) == stat_before
        if contents_before is not None:
            assert path.read_bytes() == contents_before


def test_symlinked_memory_parent_without_receipt_fails_closed_before_session_side_effects(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    memory_target = tmp_path / "memory-target"
    memory_target.mkdir()
    memory_link = workspace / "memory"
    memory_link.symlink_to(memory_target, target_is_directory=True)

    legacy_dir = workspace / "sessions"
    legacy_dir.mkdir()
    key = "telegram:symlinked-memory"
    legacy_session = legacy_dir / f"{JsonlSessionStore.storage_key(key)}.jsonl"
    session_bytes = _session_bytes(key, "legacy session")
    legacy_session.write_bytes(session_bytes)

    sessions_root = tmp_path / "private-sessions"
    paths = (workspace, memory_target, memory_link, legacy_dir, legacy_session)
    before = {
        path: _stat_snapshot(path, follow_symlinks=path != memory_link)
        for path in paths
    }

    with pytest.raises(MemoryReceiptRecoveryRequiredError) as raised:
        JsonlSessionStore(workspace, sessions_root=sessions_root)

    assert raised.value.args == ("memory recovery required",)
    assert not (memory_link / _RECEIPT_NAME).exists()
    assert not sessions_root.exists()
    assert not (workspace / ".nanobot").exists()
    assert legacy_session.read_bytes() == session_bytes
    for path, stat_before in before.items():
        assert _stat_snapshot(path, follow_symlinks=path != memory_link) == stat_before


def test_injected_session_store_does_not_bypass_pending_receipt_preflight(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    memory_dir = workspace / "memory"
    memory_dir.mkdir(parents=True)
    receipt = memory_dir / _RECEIPT_NAME
    receipt_bytes = _receipt_bytes()
    receipt.write_bytes(receipt_bytes)
    receipt.chmod(0o600)
    receipt_before = _stat_snapshot(receipt)
    memory_dir_before = _stat_snapshot(memory_dir)
    injected_store = MagicMock(spec=SessionStore)
    sessions_root = tmp_path / "private-sessions"

    with pytest.raises(MemoryReceiptRecoveryRequiredError) as raised:
        SessionManager(
            workspace,
            store=injected_store,
            sessions_root=sessions_root,
        )

    assert raised.value.args == ("memory recovery required",)
    assert receipt.read_bytes() == receipt_bytes
    assert _stat_snapshot(receipt) == receipt_before
    assert _stat_snapshot(memory_dir) == memory_dir_before
    assert not sessions_root.exists()
    assert not (workspace / ".nanobot").exists()
    assert not injected_store.mock_calls


def test_receipt_arriving_after_initial_absence_check_blocks_session_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "memory").mkdir()
    legacy_dir = workspace / "sessions"
    legacy_dir.mkdir()
    key = "telegram:receipt-race"
    session_name = f"{JsonlSessionStore.storage_key(key)}.jsonl"
    source = legacy_dir / session_name
    source_bytes = _session_bytes(key, "legacy remains", "2026-09-18T00:00:00")
    source.write_bytes(source_bytes)
    source_snapshot = _stat_snapshot(source)

    # Seed a stable session namespace with a divergent destination so that
    # ordinary migration would arbitrate a conflict and alter one side.
    workspace_id = "a" * 32
    state_dir = workspace / ".nanobot"
    state_dir.mkdir()
    (state_dir / "workspace-id").write_text(f"{workspace_id}\n", encoding="utf-8")
    sessions_root = tmp_path / "private-sessions"
    namespace = sessions_root / workspace_id
    namespace.mkdir(parents=True)
    workspace_marker = namespace / ".workspace"
    workspace_marker.write_text(f"{workspace.resolve()}\n", encoding="utf-8")
    destination = namespace / session_name
    destination_bytes = _session_bytes(key, "canonical remains", "2026-09-19T00:00:00")
    destination.write_bytes(destination_bytes)
    destination_snapshot = _stat_snapshot(destination)

    receipt = workspace / "memory" / _RECEIPT_NAME
    real_preflight = require_memory_receipt_recovery
    calls = 0

    def preflight_then_arrive(path: Path) -> None:
        nonlocal calls
        calls += 1
        real_preflight(path)
        if calls == 1:
            receipt.write_bytes(_receipt_bytes())
            receipt.chmod(0o600)

    monkeypatch.setattr(
        session_manager_module,
        "require_memory_receipt_recovery",
        preflight_then_arrive,
        raising=False,
    )

    with patch.object(
        JsonlSessionStore,
        "_migrate_from_workspace",
        autospec=True,
        side_effect=AssertionError("legacy migration entered before second preflight"),
    ):
        with pytest.raises(MemoryReceiptRecoveryRequiredError) as raised:
            JsonlSessionStore(workspace, sessions_root=sessions_root)

    assert calls == 2
    assert str(raised.value) == "memory recovery required"
    assert receipt.read_bytes() == _receipt_bytes()
    assert source.read_bytes() == source_bytes
    assert _stat_snapshot(source) == source_snapshot
    assert destination.read_bytes() == destination_bytes
    assert _stat_snapshot(destination) == destination_snapshot
    assert not (namespace / ".migration-conflicts").exists()
    assert not list(namespace.glob("*.jsonl.tmp"))
    assert workspace_marker.exists()
