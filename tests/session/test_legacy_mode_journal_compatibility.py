"""A frozen original-writer journal remains readable and append-compatible."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nanobot.session.runtime_authority import OfflineSessionOperator
from nanobot.session.session_mode import SessionModeError


async def test_frozen_legacy_reset_journal_continues_without_rewriting_prefix(tmp_path):
    fixture_path = Path(__file__).parent / "fixtures" / "legacy-mode-reset-v1.json"
    fixture = json.loads(fixture_path.read_bytes())
    assert fixture["provenance"]["writer_commit"] == "70310b09731bcd2f370f75af5b5069184396ad86"
    assert fixture["provenance"]["writer_phase"] == "reset:target_installed"
    workspace = tmp_path / "workspace"
    state = workspace / ".nanobot"
    state.mkdir(mode=0o700, parents=True)
    root = tmp_path / "sessions"
    namespace = root / fixture["workspace_id"]
    namespace.mkdir(mode=0o700, parents=True)
    workspace.chmod(0o700)
    root.chmod(0o700)
    (namespace / ".workspace").write_text(str(workspace) + "\n")
    (state / "workspace-id").write_text(fixture["workspace_id"] + "\n")
    (state / "session-mode-binding.json").write_text(fixture["workspace_witness"])
    for relative, entry in fixture["files"].items():
        data = entry["utf8"].encode("utf-8")
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        path = namespace / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o600)
    for path in namespace.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
    for path in state.iterdir():
        path.chmod(0o600)
    journal_root = namespace / ".session-mode" / "journals"
    original_journals = {path: path.read_bytes() for path in journal_root.rglob("*.json")}
    operation = journal_root / fixture["pending_operation_id"]
    assert len(list(operation.glob("*.json"))) == 2

    lease = await OfflineSessionOperator(workspace, copied_sessions_root=root).acquire()
    try:
        manager = lease.manager
        key = fixture["session_key"]
        with pytest.raises(SessionModeError):
            manager.resolve_session_mode(key)
        resolver = manager._mode_resolver
        assert resolver is not None
        recovered = resolver.recover(key)
        assert recovered.mode == "canonical"
        assert recovered.generation == fixture["target_generation"]
        snapshot = manager._jsonl_store.source.load_source_snapshot(key)
        assert snapshot.events == [] and snapshot.state.generation == fixture["target_generation"]
        assert all(path.read_bytes() == raw for path, raw in original_journals.items())

        previous = None
        phases = []
        for path in sorted(operation.glob("*.json")):
            raw = path.read_bytes()
            row = json.loads(raw)
            assert "archive_identity" not in row and "archive_digest" not in row
            assert row["previous_record_digest"] == previous
            previous = hashlib.sha256(raw).hexdigest()
            phases.append(row["phase"])
        assert phases == ["prepared", "target_installed", "mode_committed", "complete"]
        assert manager.resolve_session_mode(key) == recovered
    finally:
        if lease.close is not None:
            await lease.close()
