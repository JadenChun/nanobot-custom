from __future__ import annotations

import errno
import os
import shutil
import stat
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import pytest

import nanobot.session.checked_source as checked_source
from nanobot.agent.memory_recovery_storage import WorkspaceMemoryAuthority
from nanobot.session.checked_source import (
    CheckedSessionAuthorityError,
    open_checked_recovery_session,
    open_checked_session,
)
from nanobot.session.manager import JsonlSessionStore
from nanobot.session.recovery_locator import RecoveryCandidate, locate_recovery_candidate

WORKSPACE_ID = "a" * 32
SESSION_KEY = "telegram:recovery"
PAYLOAD = b'{"role":"user","content":"exact recovery bytes"}\n\xff\x80\n'


def _layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    state = workspace / ".nanobot"
    state.mkdir(mode=0o700)
    (state / "workspace-id").write_bytes(f"{WORKSPACE_ID}\n".encode("ascii"))

    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir(mode=0o700)
    namespace = sessions_root / WORKSPACE_ID
    namespace.mkdir(mode=0o700)
    (namespace / ".workspace").write_bytes(f"{workspace}\n".encode("utf-8"))
    source = namespace / f"{JsonlSessionStore.storage_key(SESSION_KEY)}.jsonl"
    source.write_bytes(PAYLOAD)
    source.chmod(0o600)
    for path in (workspace, state, sessions_root, namespace):
        path.chmod(0o700)
    return workspace, sessions_root, source


def _candidate(workspace: Path, sessions_root: Path) -> RecoveryCandidate:
    return locate_recovery_candidate(workspace, sessions_root, SESSION_KEY)


@contextmanager
def _lease(workspace: Path) -> Iterator[object]:
    with WorkspaceMemoryAuthority(workspace) as authority:
        with authority.operation() as operation:
            yield operation


def _lock_paths(candidate: RecoveryCandidate) -> tuple[Path, Path, Path, Path]:
    return (
        candidate.sessions_root.parent / ".workspace-migration.lock",
        candidate.namespace_path / ".execution-locks",
        candidate.namespace_path
        / ".execution-locks"
        / f"{JsonlSessionStore.storage_key(SESSION_KEY)}.lock",
        candidate.namespace_path / ".session-files.lock",
    )


def _assert_error(action, code: str) -> CheckedSessionAuthorityError:
    with pytest.raises(CheckedSessionAuthorityError) as captured:
        action()
    error = captured.value
    assert error.code == code
    assert error.__cause__ is None
    assert error.__context__ is None
    assert str(error) == f"checked session authority: {code}"
    return error


def _source_signature(path: Path) -> tuple[int, int, int, int, int, int, int, int, int]:
    info = path.lstat()
    return (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _assert_no_session_lock_side_effects(candidate: RecoveryCandidate) -> None:
    migration, execution_directory, execution_lock, session_lock = _lock_paths(candidate)
    assert not migration.exists()
    assert not execution_directory.exists()
    assert not execution_lock.exists()
    assert not session_lock.exists()


def test_valid_candidate_reads_exact_bytes_and_only_creates_standard_private_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, sessions_root, source = _layout(tmp_path)
    candidate = _candidate(workspace, sessions_root)
    source_before = _source_signature(source)
    source_bytes = source.read_bytes()
    assert not (candidate.sessions_root.parent / ".workspace-migration.lock").exists()
    assert not (candidate.namespace_path / ".execution-locks").exists()
    assert not (candidate.namespace_path / ".session-files.lock").exists()

    def forbidden_constructor(*args, **kwargs) -> None:
        raise AssertionError("recovery checked reads must not construct JsonlSessionStore")

    monkeypatch.setattr(JsonlSessionStore, "__init__", forbidden_constructor)
    with _lease(workspace) as operation:
        with open_checked_recovery_session(operation, candidate) as authority:
            assert authority.read_source_bytes() == PAYLOAD
            _assert_error(lambda: authority.replace_source_bytes(b"replacement"), "invalid")

    migration, execution_directory, execution_lock, session_lock = _lock_paths(candidate)
    assert migration.is_file() and stat.S_IMODE(migration.stat().st_mode) == 0o600
    assert (
        execution_directory.is_dir() and stat.S_IMODE(execution_directory.stat().st_mode) == 0o700
    )
    assert execution_lock.is_file() and stat.S_IMODE(execution_lock.stat().st_mode) == 0o600
    assert session_lock.is_file() and stat.S_IMODE(session_lock.stat().st_mode) == 0o600
    assert source.read_bytes() == source_bytes == PAYLOAD
    assert _source_signature(source) == source_before
    assert sorted(path.name for path in candidate.namespace_path.iterdir()) == sorted(
        [".workspace", source.name, ".execution-locks", ".session-files.lock"]
    )


@pytest.mark.parametrize(
    "drift", ["source", "workspace-marker", "namespace-marker", "root", "namespace", "duplicate"]
)
def test_stale_candidate_is_rejected_before_any_lock_side_effect(
    tmp_path: Path, drift: str
) -> None:
    workspace, sessions_root, source = _layout(tmp_path)
    candidate = _candidate(workspace, sessions_root)
    if drift == "source":
        source.write_bytes(PAYLOAD + b"changed")
    elif drift == "workspace-marker":
        (workspace / ".nanobot" / "workspace-id").write_bytes(b"b" * 32 + b"\n")
    elif drift == "namespace-marker":
        (candidate.namespace_path / ".workspace").write_bytes(
            f"{tmp_path / 'elsewhere'}\n".encode()
        )
    elif drift == "root":
        moved = sessions_root.with_name("sessions-old")
        os.rename(sessions_root, moved)
        sessions_root.mkdir(mode=0o700)
        sessions_root.chmod(0o700)
    elif drift == "namespace":
        moved = candidate.namespace_path.with_name("namespace-old")
        os.rename(candidate.namespace_path, moved)
        candidate.namespace_path.mkdir(mode=0o700)
        candidate.namespace_path.chmod(0o700)
    else:
        duplicate = sessions_root / ("b" * 32)
        shutil.copytree(candidate.namespace_path, duplicate)
        duplicate.chmod(0o700)
        (duplicate / ".execution-locks").unlink(missing_ok=True)
        shutil.rmtree(duplicate / ".execution-locks", ignore_errors=True)

    with _lease(workspace) as operation:
        _assert_error(
            lambda: open_checked_recovery_session(operation, candidate).__enter__(), "changed"
        )

    _assert_no_session_lock_side_effects(candidate)
    assert not (sessions_root / ("b" * 32) / ".execution-locks").exists()


def test_duplicate_copy_created_after_candidate_is_revalidated_after_lock_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, sessions_root, source = _layout(tmp_path)
    candidate = _candidate(workspace, sessions_root)
    real_flock = checked_source.fcntl.flock
    injected = False

    def create_duplicate_and_report_busy(fd: int, operation: int) -> None:
        nonlocal injected
        if not injected and operation & checked_source.fcntl.LOCK_EX:
            injected = True
            duplicate = sessions_root / ("b" * 32)
            shutil.copytree(candidate.namespace_path, duplicate)
            duplicate.chmod(0o700)
            shutil.rmtree(duplicate / ".execution-locks", ignore_errors=True)
            raise BlockingIOError(errno.EAGAIN, "busy")
        real_flock(fd, operation)

    with _lease(workspace) as operation:
        monkeypatch.setattr(checked_source.fcntl, "flock", create_duplicate_and_report_busy)
        authority = open_checked_recovery_session(operation, candidate)
        _assert_error(authority.__enter__, "changed")
        authority.close()

    assert injected
    assert source.read_bytes() == PAYLOAD


def test_root_namespace_added_after_list_snapshot_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, sessions_root, _source = _layout(tmp_path)
    candidate = _candidate(workspace, sessions_root)
    duplicate = sessions_root / ("b" * 32)
    with _lease(workspace) as operation:
        authority = open_checked_recovery_session(operation, candidate)
        authority.__enter__()
        root_fd = authority._directories[-1].fd
        real_listdir = checked_source.os.listdir
        injected = False

        def add_duplicate_after_snapshot(path):
            nonlocal injected
            names = real_listdir(path)
            if path == root_fd and not injected:
                injected = True
                shutil.copytree(candidate.namespace_path, duplicate)
                duplicate.chmod(0o700)
            return names

        monkeypatch.setattr(checked_source.os, "listdir", add_duplicate_after_snapshot)
        monkeypatch.setattr(
            checked_source.os,
            "supports_fd",
            set(checked_source.os.supports_fd) | {add_duplicate_after_snapshot},
        )
        _assert_error(authority.require_current, "changed")
        authority.close()

    assert injected and duplicate.is_dir()


def test_source_symlink_after_entry_is_rejected_without_opening_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, sessions_root, source = _layout(tmp_path)
    candidate = _candidate(workspace, sessions_root)
    external_source = tmp_path / "outside.jsonl"
    external_source.write_bytes(b"outside bytes must not be read")
    displaced_source = source.with_suffix(".displaced")

    with _lease(workspace) as operation:
        authority = open_checked_recovery_session(operation, candidate)
        authority.__enter__()
        source.rename(displaced_source)
        source.symlink_to(external_source)
        opened_source: list[int] = []
        real_open = checked_source.os.open

        def track_source_open(name, flags, mode=0o777, *, dir_fd=None):
            if name == source.name:
                opened_source.append(flags)
            return real_open(name, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(checked_source.os, "open", track_source_open)
        _assert_error(authority.read_source_bytes, "unsafe")
        authority.close()

    assert not opened_source
    assert external_source.read_bytes() == b"outside bytes must not be read"


def test_mid_chunk_source_identity_change_discards_the_partial_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, sessions_root, source = _layout(tmp_path)
    source_bytes = b"x" * (checked_source._SOURCE_READ_CHUNK_SIZE + 17)
    source.write_bytes(source_bytes)
    candidate = _candidate(workspace, sessions_root)
    source_info = source.stat()
    source_identity = source_info.st_dev, source_info.st_ino
    real_read = checked_source.os.read
    injected = False

    def mutate_after_first_source_chunk(fd: int, size: int) -> bytes:
        nonlocal injected
        chunk = real_read(fd, size)
        info = os.fstat(fd)
        if chunk and not injected and (info.st_dev, info.st_ino) == source_identity:
            injected = True
            source.write_bytes(b"changed while reading")
        return chunk

    with _lease(workspace) as operation:
        authority = open_checked_recovery_session(operation, candidate)
        authority.__enter__()
        monkeypatch.setattr(checked_source.os, "read", mutate_after_first_source_chunk)
        _assert_error(authority.read_source_bytes, "changed")
        authority.close()

    assert injected
    assert source.read_bytes() == b"changed while reading"


def test_memory_lease_is_rechecked_after_a_busy_lock_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, sessions_root, source = _layout(tmp_path)
    candidate = _candidate(workspace, sessions_root)
    real_flock = checked_source.fcntl.flock

    with _lease(workspace) as operation:

        def close_lease_and_report_busy(fd: int, lock_operation: int) -> None:
            if lock_operation & checked_source.fcntl.LOCK_EX:
                operation.__exit__(None, None, None)
                raise BlockingIOError(errno.EAGAIN, "busy")
            real_flock(fd, lock_operation)

        monkeypatch.setattr(checked_source.fcntl, "flock", close_lease_and_report_busy)
        authority = open_checked_recovery_session(operation, candidate)
        _assert_error(authority.__enter__, "changed")
        authority.close()

    assert source.read_bytes() == PAYLOAD


@pytest.mark.parametrize(
    "drift", ["source", "workspace-marker", "namespace-marker", "root", "namespace"]
)
def test_candidate_change_after_entry_fails_before_returning_bytes(
    tmp_path: Path, drift: str
) -> None:
    workspace, sessions_root, source = _layout(tmp_path)
    candidate = _candidate(workspace, sessions_root)
    with _lease(workspace) as operation:
        authority = open_checked_recovery_session(operation, candidate)
        authority.__enter__()
        if drift == "source":
            source.write_bytes(PAYLOAD + b"later")
        elif drift == "workspace-marker":
            (workspace / ".nanobot" / "workspace-id").write_bytes(b"b" * 32 + b"\n")
        elif drift == "namespace-marker":
            (candidate.namespace_path / ".workspace").write_bytes(b"/other/workspace\n")
        elif drift == "root":
            os.rename(sessions_root, sessions_root.with_name("sessions-old"))
            sessions_root.mkdir(mode=0o700)
            sessions_root.chmod(0o700)
        else:
            os.rename(candidate.namespace_path, candidate.namespace_path.with_name("namespace-old"))
            candidate.namespace_path.mkdir(mode=0o700)
            candidate.namespace_path.chmod(0o700)
        _assert_error(authority.read_source_bytes, "changed")
        authority.close()


def test_recovery_acquires_migration_execution_then_namespace_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, sessions_root, _source = _layout(tmp_path)
    candidate = _candidate(workspace, sessions_root)
    execution_lock = (
        candidate.namespace_path
        / ".execution-locks"
        / f"{JsonlSessionStore.storage_key(SESSION_KEY)}.lock"
    )
    expected = {
        "migration": candidate.sessions_root.parent / ".workspace-migration.lock",
        "execution": execution_lock,
        "session-files": candidate.namespace_path / ".session-files.lock",
    }
    acquired: list[str] = []
    real_flock = checked_source.fcntl.flock

    def record_lock(fd: int, operation: int) -> None:
        if operation & checked_source.fcntl.LOCK_EX:
            info = os.fstat(fd)
            for name, path in expected.items():
                try:
                    path_info = path.stat()
                except FileNotFoundError:
                    continue
                if (info.st_dev, info.st_ino) == (path_info.st_dev, path_info.st_ino):
                    acquired.append(name)
        real_flock(fd, operation)

    with _lease(workspace) as operation:
        monkeypatch.setattr(checked_source.fcntl, "flock", record_lock)
        with open_checked_recovery_session(operation, candidate):
            pass

    assert acquired == ["migration", "execution", "session-files"]


def test_candidate_path_mismatch_is_rejected_before_lock_side_effects(tmp_path: Path) -> None:
    workspace, sessions_root, _source = _layout(tmp_path)
    candidate = _candidate(workspace, sessions_root)
    forged = replace(candidate, session_path=candidate.namespace_path / "other.jsonl")

    with _lease(workspace) as operation:
        _assert_error(lambda: open_checked_recovery_session(operation, forged), "invalid")

    _assert_no_session_lock_side_effects(candidate)


def test_wrong_workspace_lease_is_rejected_without_lock_side_effects(tmp_path: Path) -> None:
    workspace, sessions_root, _source = _layout(tmp_path / "first")
    other_workspace, _other_root, _other_source = _layout(tmp_path / "second")
    candidate = _candidate(workspace, sessions_root)

    with _lease(other_workspace) as operation:
        _assert_error(lambda: open_checked_recovery_session(operation, candidate), "invalid")

    _assert_no_session_lock_side_effects(candidate)


def test_inactive_lease_is_rejected_without_lock_side_effects(tmp_path: Path) -> None:
    workspace, sessions_root, _source = _layout(tmp_path)
    candidate = _candidate(workspace, sessions_root)
    with WorkspaceMemoryAuthority(workspace) as memory_authority:
        operation = memory_authority.operation()
        _assert_error(lambda: open_checked_recovery_session(operation, candidate), "invalid")
    _assert_no_session_lock_side_effects(candidate)


def test_memory_lease_is_rechecked_after_entry(tmp_path: Path) -> None:
    workspace, sessions_root, _source = _layout(tmp_path)
    candidate = _candidate(workspace, sessions_root)
    memory_authority = WorkspaceMemoryAuthority(workspace)
    memory_authority.__enter__()
    operation = memory_authority.operation()
    operation.__enter__()
    authority = open_checked_recovery_session(operation, candidate)
    authority.__enter__()
    operation.__exit__(None, None, None)
    _assert_error(authority.read_source_bytes, "changed")
    authority.close()
    memory_authority.__exit__(None, None, None)


def test_ordinary_checked_session_keeps_exact_store_source_read(tmp_path: Path) -> None:
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    key = "ordinary:exact"
    source = backend.get_session_path(key)
    source.write_bytes(b"ordinary source bytes\n")

    with open_checked_session(backend, key) as authority:
        assert authority.read_source_bytes() == b"ordinary source bytes\n"
