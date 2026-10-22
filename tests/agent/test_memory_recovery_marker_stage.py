"""Locked staging of authenticated history recovery marker frames."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent import memory_recovery_marker_stage as staging
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_target_mutation as mutation


def _stage_capable() -> bool:
    if os.name != "posix":
        return False
    if any(
        not isinstance(getattr(os, name, None), int)
        for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")
    ):
        return False
    return {os.open, os.stat, os.mkdir, os.link, os.rename} <= getattr(
        os, "supports_dir_fd", set()
    ) and os.link in getattr(os, "supports_follow_symlinks", set())


pytestmark = pytest.mark.skipif(
    not _stage_capable(), reason="history staging requires POSIX no-follow dir-fd support"
)

_MARKER_ID = "a" * 24
_HISTORY_ENTRY = "[2026-09-25T00:00:00Z] Recovered history"
_MARKER = f"<!-- nanobot-consolidation:{_MARKER_ID} -->"
_TEMP_PREFIX = ".memory-recovery-replace-"


def _frame(entry: str = _HISTORY_ENTRY) -> bytes:
    return (f"{_MARKER}\n{entry}").rstrip().encode("utf-8") + b"\n\n"


def _write(memory: Path, name: str, content: bytes, mode: int = 0o644) -> Path:
    target = memory / name
    target.write_bytes(content)
    target.chmod(mode)
    return target


def _stages(memory: Path) -> list[Path]:
    return list(memory.glob(f"{_TEMP_PREFIX}*"))


def _stage(
    authority: storage.WorkspaceMemoryAuthority,
    marker_id: str = _MARKER_ID,
    entry: str = _HISTORY_ENTRY,
) -> staging.HistoryMarkerStageResult:
    with authority.operation() as operation:
        return staging.stage_history_marker(authority, operation, marker_id, entry)


def _enable_patched_rename(monkeypatch: pytest.MonkeyPatch, rename: Any) -> None:
    monkeypatch.setattr(os, "rename", rename)
    monkeypatch.setattr(os, "supports_dir_fd", set(os.supports_dir_fd) | {rename})


def test_stages_marker_into_absent_history_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        result = _stage(authority)

    assert result.disposition == "staged"
    assert result.snapshot.name == "HISTORY.md"
    assert result.snapshot.content == _frame()
    assert (memory / "HISTORY.md").read_bytes() == _frame()
    assert _stages(memory) == []


def test_stages_marker_after_nonempty_history_without_changing_prefix(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    original = b"# Earlier notes\r\n\r\nPrior recovery\n\n"
    history = _write(memory, "HISTORY.md", original)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        result = _stage(authority)

    assert result.disposition == "staged"
    assert result.snapshot.content == original + _frame()
    assert history.read_bytes() == original + _frame()
    assert _stages(memory) == []


def test_stages_entry_with_internal_blank_lines_exactly(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    entry = "First line\n\nSecond line"

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        result = _stage(authority, entry=entry)

    expected = _frame(entry)
    assert result.disposition == "staged"
    assert result.snapshot.content == expected
    assert (memory / "HISTORY.md").read_bytes() == expected
    assert _stages(memory) == []


def test_present_marker_returns_current_snapshot_without_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    history = _write(memory, "HISTORY.md", b"# Earlier notes\n\n" + _frame())
    before = history.stat()

    def reject_replacement(*args: Any, **kwargs: Any) -> None:
        pytest.fail("already-present marker attempted a replacement")

    monkeypatch.setattr(staging.target_mutation, "replace_target_if_unchanged", reject_replacement)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        result = _stage(authority)

    after = history.stat()
    assert result.disposition == "already_present"
    assert result.snapshot.content == b"# Earlier notes\n\n" + _frame()
    assert (after.st_dev, after.st_ino, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
    )
    assert _stages(memory) == []


def test_replay_after_staging_does_not_append_duplicate_frame(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        first = _stage(authority)
        second = _stage(authority)

    assert first.disposition == "staged"
    assert second.disposition == "already_present"
    assert second.snapshot.content == _frame()
    assert (memory / "HISTORY.md").read_bytes() == _frame()
    assert _stages(memory) == []


@pytest.mark.parametrize(
    "history",
    [
        b"# Earlier notes\n\n" + _frame() + b"Later note\n\n",
        _frame() + b"Intervening note\n\n" + _frame(),
        f"# Earlier notes\n\n<!-- nanobot-consolidation:{_MARKER_ID}".encode(),
    ],
    ids=("nonterminal", "duplicate", "partial"),
)
def test_ambiguous_marker_frames_fail_without_mutation_or_stage(
    tmp_path: Path, history: bytes
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = _write(memory, "HISTORY.md", history)
    before = target.stat()

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(ValueError, match="^invalid history marker frame$") as rejected:
            _stage(authority)

    after = target.stat()
    assert str(rejected.value) == "invalid history marker frame"
    assert target.read_bytes() == history
    assert (after.st_dev, after.st_ino, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
    )
    assert _stages(memory) == []


def test_uncertain_rename_before_effect_blocks_without_retry_and_keeps_private_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    history = _write(memory, "HISTORY.md", b"# Earlier notes\n\n")
    before = history.stat()

    def fail_before_rename(*args: Any, **kwargs: Any) -> None:
        raise OSError("secret injected rename failure")

    _enable_patched_rename(monkeypatch, fail_before_rename)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(mutation.MemoryTargetReplacementUncertainError) as rejected:
            _stage(authority)

    after = history.stat()
    stages = _stages(memory)
    assert str(rejected.value) == "memory target replacement outcome is uncertain"
    assert "secret" not in str(rejected.value)
    assert history.read_bytes() == b"# Earlier notes\n\n"
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert len(stages) == 1
    stage_info = stages[0].stat()
    assert stat.S_IMODE(stage_info.st_mode) == 0o600
    assert stat.S_ISREG(stage_info.st_mode)
    assert stage_info.st_nlink == 1


def test_uncertain_rename_after_effect_accepts_exact_proposal_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    history = _write(memory, "HISTORY.md", b"# Earlier notes\n\n")
    original_rename = os.rename
    calls = 0

    def rename_then_raise(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        original_rename(*args, **kwargs)
        raise OSError("secret post-rename diagnostic")

    _enable_patched_rename(monkeypatch, rename_then_raise)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        result = _stage(authority)

    assert calls == 1
    assert result.disposition == "staged"
    assert result.snapshot.content == b"# Earlier notes\n\n" + _frame()
    assert history.read_bytes() == b"# Earlier notes\n\n" + _frame()
    assert _stages(memory) == []


def test_changed_target_before_replace_fails_closed_without_overwrite_or_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    history = _write(memory, "HISTORY.md", b"# Original\n\n")
    original_replace = staging.target_mutation.replace_target_if_unchanged
    changed = b"# Concurrent change\n\n"

    def change_then_replace(operation: Any, snapshot: Any, proposal: bytes) -> Any:
        history.write_bytes(changed)
        history.chmod(0o644)
        return original_replace(operation, snapshot, proposal)

    monkeypatch.setattr(staging.target_mutation, "replace_target_if_unchanged", change_then_replace)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _stage(authority)

    assert rejected.value.code == "binding"
    assert history.read_bytes() == changed
    assert _stages(memory) == []


def test_staging_does_not_modify_receipt_or_other_recovery_targets(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    receipt = _write(memory, ".consolidation.pending.json", b"private receipt", 0o600)
    protected = {
        "history.jsonl": b"session history",
        ".cursor": b"cursor bytes",
        ".dream_cursor": b"dream cursor bytes",
    }
    protected_paths = {
        name: _write(memory, name, content, 0o600) for name, content in protected.items()
    }
    before = {
        name: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
        for name, path in {"receipt": receipt, **protected_paths}.items()
    }

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        result = _stage(authority)

    assert result.disposition == "staged"
    for name, path in {"receipt": receipt, **protected_paths}.items():
        info = path.stat()
        assert (path.read_bytes(), info.st_ino, info.st_mtime_ns) == before[name]
    assert _stages(memory) == []


def test_same_content_with_changed_stat_identity_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    original = b"# Existing history\n\n"
    history = _write(memory, "HISTORY.md", original)
    original_snapshot_target = storage.WorkspaceMemoryAuthority.snapshot_target
    calls = 0

    def replace_between_snapshots(
        authority: storage.WorkspaceMemoryAuthority, name: str
    ) -> storage.MemoryTargetSnapshot:
        nonlocal calls
        calls += 1
        if name == "HISTORY.md" and calls == 2:
            replacement = memory / "replacement-history"
            replacement.write_bytes(original)
            replacement.chmod(0o644)
            replacement.replace(history)
        return original_snapshot_target(authority, name)

    monkeypatch.setattr(
        storage.WorkspaceMemoryAuthority,
        "snapshot_target",
        replace_between_snapshots,
    )
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _stage(authority)

    assert rejected.value.code == "binding"
    assert history.read_bytes() == original
    assert _stages(memory) == []


def test_uncertain_rename_with_third_target_state_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    history = _write(memory, "HISTORY.md", b"# Original\n\n")
    changed = b"# Concurrent change\n\n"

    def change_then_fail_rename(*args: Any, **kwargs: Any) -> None:
        history.write_bytes(changed)
        history.chmod(0o644)
        raise OSError("secret uncertain rename diagnostic")

    _enable_patched_rename(monkeypatch, change_then_fail_rename)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _stage(authority)

    assert rejected.value.code == "binding"
    assert "secret" not in str(rejected.value)
    assert history.read_bytes() == changed
    stages = _stages(memory)
    assert len(stages) == 1
    assert stat.S_IMODE(stages[0].stat().st_mode) == 0o600


def test_uncertain_rename_requires_directory_fsync_before_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    history = _write(memory, "HISTORY.md", b"# Earlier notes\n\n")
    original_rename = os.rename
    original_fsync = os.fsync

    def rename_then_raise(*args: Any, **kwargs: Any) -> None:
        original_rename(*args, **kwargs)
        raise OSError("secret post-rename diagnostic")

    _enable_patched_rename(monkeypatch, rename_then_raise)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        def fail_directory_fsync(fd: int) -> None:
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("secret directory sync failure")
            original_fsync(fd)

        monkeypatch.setattr(os, "fsync", fail_directory_fsync)
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _stage(authority)

    assert rejected.value.code == "durability"
    assert "secret" not in str(rejected.value)
    assert history.read_bytes() == b"# Earlier notes\n\n" + _frame()
    assert _stages(memory) == []


def test_already_present_marker_requires_directory_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    expected = b"# Earlier notes\n\n" + _frame()
    history = _write(memory, "HISTORY.md", expected)
    before = history.stat()
    original_fsync = os.fsync

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        def fail_directory_fsync(fd: int) -> None:
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("secret directory sync failure")
            original_fsync(fd)

        monkeypatch.setattr(os, "fsync", fail_directory_fsync)
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _stage(authority)

    after = history.stat()
    assert rejected.value.code == "durability"
    assert "secret" not in str(rejected.value)
    assert history.read_bytes() == expected
    assert (after.st_dev, after.st_ino, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_mtime_ns,
    )
    assert _stages(memory) == []


def test_target_change_during_uncertain_reconciliation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    history = _write(memory, "HISTORY.md", b"# Earlier notes\n\n")
    changed = b"# Changed during reconciliation\n\n"
    original_rename = os.rename
    original_fsync = os.fsync

    def rename_then_raise(*args: Any, **kwargs: Any) -> None:
        original_rename(*args, **kwargs)
        raise OSError("secret post-rename diagnostic")

    def fsync_then_change_target(fd: int) -> None:
        original_fsync(fd)
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            history.write_bytes(changed)
            history.chmod(0o644)

    _enable_patched_rename(monkeypatch, rename_then_raise)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        monkeypatch.setattr(os, "fsync", fsync_then_change_target)
        with pytest.raises(storage.MemoryStorageError) as rejected:
            _stage(authority)

    assert rejected.value.code == "binding"
    assert "secret" not in str(rejected.value)
    assert history.read_bytes() == changed
    assert _stages(memory) == []
