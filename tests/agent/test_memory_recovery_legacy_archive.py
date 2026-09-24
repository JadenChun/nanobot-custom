from __future__ import annotations

import errno
import hashlib
import os
import stat
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_legacy_archive as archive
from nanobot.agent import memory_recovery_storage as storage


def _setup(tmp_path: Path):
    memory = tmp_path / "memory"
    memory.mkdir()
    source = memory / "HISTORY.md"
    source.write_bytes(b"[2026-09-24 12:00] USER: old\n")
    source.chmod(0o600)
    anchor = memory / "history.jsonl"
    anchor.write_bytes(b'{"role":"user","content":"migrated"}\n')
    anchor.chmod(0o600)
    return memory, source


def _issue(authority):
    snapshot = authority.snapshot_target("HISTORY.md")
    anchor_snapshot = authority.snapshot_target("history.jsonl")
    operation = authority.operation()
    operation.__enter__()
    return operation, snapshot, anchor_snapshot


def _archive(operation, anchor_snapshot, source_snapshot, name):
    assert source_snapshot.stat_identity is not None
    assert source_snapshot.content is not None
    return archive.archive_legacy_history_if_unchanged(
        operation, anchor_snapshot,
        expected_source_stat=source_snapshot.stat_identity,
        expected_digest=hashlib.sha256(source_snapshot.content).hexdigest(),
        expected_size=len(source_snapshot.content),
        backup_name=name,
    )


def _error(call, code):
    with pytest.raises(storage.MemoryStorageError) as caught:
        call()
    assert caught.value.code == code


def test_selects_first_free_name_including_dangling_symlinks(tmp_path: Path) -> None:
    memory, source = _setup(tmp_path)
    (memory / "HISTORY.md.bak").symlink_to("absent")
    (memory / "HISTORY.md.bak.2").write_bytes(b"occupied")
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation, snapshot, anchor_snapshot = _issue(authority)
        try:
            name = archive.choose_legacy_backup_name(operation, snapshot)
            assert name == "HISTORY.md.bak.3"
            _archive(operation, anchor_snapshot, snapshot, name)
        finally:
            operation.__exit__(None, None, None)
    assert not source.exists()
    assert (memory / name).read_bytes() == snapshot.content
    assert (memory / "HISTORY.md.bak").is_symlink()


def test_success_and_replay_one_link(tmp_path: Path) -> None:
    memory, source = _setup(tmp_path)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation, snapshot, anchor_snapshot = _issue(authority)
        try:
            name = archive.choose_legacy_backup_name(operation, snapshot)
            _archive(operation, anchor_snapshot, snapshot, name)
            _archive(operation, anchor_snapshot, snapshot, name)
        finally:
            operation.__exit__(None, None, None)
    assert not source.exists()
    backup = memory / name
    assert backup.read_bytes() == snapshot.content
    assert backup.stat().st_nlink == 1


@pytest.mark.parametrize("stage", ["link", "first_fsync", "unlink", "second_fsync"])
def test_attempted_mutations_uncertain_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    memory, source = _setup(tmp_path)
    real_link, real_unlink, real_fsync = os.link, os.unlink, os.fsync
    fired = False
    fsync_count = 0

    def link(*args, **kwargs):
        nonlocal fired
        real_link(*args, **kwargs)
        if stage == "link" and not fired:
            fired = True
            raise OSError(errno.EIO, "after link")

    def unlink(*args, **kwargs):
        nonlocal fired
        real_unlink(*args, **kwargs)
        if stage == "unlink" and not fired:
            fired = True
            raise OSError(errno.EIO, "after unlink")

    def fsync(fd):
        nonlocal fired, fsync_count
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            return real_fsync(fd)
        fsync_count += 1
        if fsync_count >= 1 and stage == "first_fsync" and not fired:
            fired = True
            raise OSError(errno.EIO, "after link fsync")
        if fsync_count >= 2 and stage == "second_fsync" and not fired:
            fired = True
            raise OSError(errno.EIO, "after unlink fsync")
        return real_fsync(fd)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation, snapshot, anchor_snapshot = _issue(authority)
        try:
            name = archive.choose_legacy_backup_name(operation, snapshot)
            with monkeypatch.context() as patch:
                patch.setattr(os, "link", link)
                patch.setattr(os, "unlink", unlink)
                patch.setattr(os, "fsync", fsync)
                patch.setattr(archive, "_capabilities", lambda: None)
                _error(lambda: _archive(operation, anchor_snapshot, snapshot, name), "uncertain")
            assert fired
            _archive(operation, anchor_snapshot, snapshot, name)
        finally:
            operation.__exit__(None, None, None)
    assert not source.exists()
    assert (memory / name).read_bytes() == snapshot.content


@pytest.mark.parametrize("drift", ["source", "backup", "extra_link", "collision"])
def test_changed_inputs_conflict(tmp_path: Path, drift: str) -> None:
    memory, source = _setup(tmp_path)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation, snapshot, anchor_snapshot = _issue(authority)
        try:
            name = archive.choose_legacy_backup_name(operation, snapshot)
            backup = memory / name
            if drift == "source":
                source.write_bytes(b"changed")
            elif drift == "backup":
                backup.write_bytes(b"foreign")
            elif drift == "extra_link":
                os.link(source, memory / "extra")
            else:
                backup.symlink_to("missing")
            _error(lambda: _archive(operation, anchor_snapshot, snapshot, name), "conflict")
        finally:
            operation.__exit__(None, None, None)
    assert source.exists()


def test_stale_authority_rejected(tmp_path: Path) -> None:
    _setup(tmp_path)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation, snapshot, anchor_snapshot = _issue(authority)
        name = archive.choose_legacy_backup_name(operation, snapshot)
        operation.__exit__(None, None, None)
        _error(lambda: _archive(operation, anchor_snapshot, snapshot, name), "closed")


def test_name_selection_rejects_changed_source(tmp_path: Path) -> None:
    _memory, source = _setup(tmp_path)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation, snapshot, anchor_snapshot = _issue(authority)
        try:
            source.write_bytes(b"changed")
            _error(lambda: archive.choose_legacy_backup_name(operation, snapshot), "conflict")
        finally:
            operation.__exit__(None, None, None)


def test_unsafe_source_symlink_and_foreign_backup_conflict(tmp_path: Path) -> None:
    memory, source = _setup(tmp_path)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation, snapshot, anchor_snapshot = _issue(authority)
        try:
            name = archive.choose_legacy_backup_name(operation, snapshot)
            source.unlink()
            source.symlink_to("missing")
            _error(lambda: _archive(operation, anchor_snapshot, snapshot, name), "conflict")
            source.unlink()
            (memory / name).write_bytes(b"foreign")
            _error(lambda: _archive(operation, anchor_snapshot, snapshot, name), "conflict")
        finally:
            operation.__exit__(None, None, None)


def test_replay_detects_backup_content_and_extra_link(tmp_path: Path) -> None:
    memory, source = _setup(tmp_path)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation, snapshot, anchor_snapshot = _issue(authority)
        try:
            name = archive.choose_legacy_backup_name(operation, snapshot)
            _archive(operation, anchor_snapshot, snapshot, name)
            backup = memory / name
            backup.write_bytes(b"tampered")
            _error(lambda: _archive(operation, anchor_snapshot, snapshot, name), "conflict")
        finally:
            operation.__exit__(None, None, None)
    assert not source.exists()

    other = tmp_path / "other"
    other.mkdir()
    memory, source = _setup(other)
    with storage.WorkspaceMemoryAuthority(other) as authority:
        operation, snapshot, anchor_snapshot = _issue(authority)
        try:
            name = archive.choose_legacy_backup_name(operation, snapshot)
            _archive(operation, anchor_snapshot, snapshot, name)
            os.link(memory / name, memory / "extra")
            _error(lambda: _archive(operation, anchor_snapshot, snapshot, name), "conflict")
        finally:
            operation.__exit__(None, None, None)


def test_source_fsync_failure_is_pre_link_durability_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    memory, source = _setup(tmp_path)
    original = os.fsync

    def fail_file(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "source fsync")
        return original(fd)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation, snapshot, anchor_snapshot = _issue(authority)
        try:
            name = archive.choose_legacy_backup_name(operation, snapshot)
            with monkeypatch.context() as patch:
                patch.setattr(os, "fsync", fail_file)
                _error(lambda: _archive(operation, anchor_snapshot, snapshot, name), "durability")
        finally:
            operation.__exit__(None, None, None)
    assert source.read_bytes() == snapshot.content
    assert not (memory / name).exists()


def test_link_collision_does_not_clobber_and_is_uncertain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    memory, source = _setup(tmp_path)
    real_link = os.link

    def race_link(src, dst, **kwargs):
        (memory / dst).write_bytes(b"racer")
        return real_link(src, dst, **kwargs)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation, snapshot, anchor_snapshot = _issue(authority)
        try:
            name = archive.choose_legacy_backup_name(operation, snapshot)
            with monkeypatch.context() as patch:
                patch.setattr(os, "link", race_link)
                patch.setattr(archive, "_capabilities", lambda: None)
                _error(lambda: _archive(operation, anchor_snapshot, snapshot, name), "uncertain")
            _error(lambda: _archive(operation, anchor_snapshot, snapshot, name), "conflict")
        finally:
            operation.__exit__(None, None, None)
    assert source.read_bytes() == snapshot.content
    assert (memory / name).read_bytes() == b"racer"


@pytest.mark.parametrize("state", ["two_links", "one_link"])
def test_replay_with_fresh_authority_uses_present_migrated_anchor(tmp_path: Path, state: str) -> None:
    memory, source = _setup(tmp_path)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation, source_snapshot, anchor_snapshot = _issue(authority)
        try:
            name = archive.choose_legacy_backup_name(operation, source_snapshot)
        finally:
            operation.__exit__(None, None, None)
    os.link(source, memory / name)
    if state == "one_link":
        source.unlink()
    migrated = (memory / "history.jsonl").read_bytes()
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        anchor_snapshot = authority.snapshot_target("history.jsonl")
        with authority.operation() as operation:
            _archive(operation, anchor_snapshot, source_snapshot, name)
    assert not source.exists()
    assert (memory / name).read_bytes() == source_snapshot.content
    assert (memory / name).stat().st_nlink == 1
    assert (memory / "history.jsonl").read_bytes() == migrated


def test_rejects_bad_frozen_evidence_and_wrong_anchor(tmp_path: Path) -> None:
    memory, source = _setup(tmp_path)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        operation, source_snapshot, anchor_snapshot = _issue(authority)
        try:
            name = archive.choose_legacy_backup_name(operation, source_snapshot)
            identity = source_snapshot.stat_identity
            assert identity is not None
            _error(
                lambda: archive.archive_legacy_history_if_unchanged(
                    operation, source_snapshot, expected_source_stat=identity,
                    expected_digest=hashlib.sha256(source_snapshot.content or b"").hexdigest(),
                    expected_size=identity.size, backup_name=name,
                ), "invalid",
            )
            for digest, size in (
                ("invalid", identity.size),
                ("0" * 64, identity.size + 1),
                ("0" * 64, identity.size),
            ):
                _error(
                    lambda: archive.archive_legacy_history_if_unchanged(
                        operation, anchor_snapshot, expected_source_stat=identity,
                        expected_digest=digest, expected_size=size, backup_name=name,
                    ), "invalid" if digest == "invalid" or size != identity.size else "conflict",
                )
        finally:
            operation.__exit__(None, None, None)
    assert source.exists()
    assert (memory / "history.jsonl").exists()
