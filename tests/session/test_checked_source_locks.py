from __future__ import annotations

import errno
import multiprocessing
import os
import stat
import tempfile
import threading
from pathlib import Path

import pytest
from filelock import FileLock, Timeout

import nanobot.session.checked_source as checked_source
from nanobot.session.checked_source import (
    CheckedSessionAuthority,
    CheckedSessionAuthorityError,
    open_checked_session,
)
from nanobot.session.manager import JsonlSessionStore


def _backend(tmp_path: Path, name: str = "sessions") -> JsonlSessionStore:
    return JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / name)


def _lock_paths(backend: JsonlSessionStore, key: str) -> dict[str, Path]:
    return {
        "migration": backend.sessions_dir.parent / ".workspace-migration.lock",
        "execution": backend.sessions_dir / ".execution-locks" / f"{backend.storage_key(key)}.lock",
        "session": backend.sessions_dir / ".session-files.lock",
    }


def _prepare_lock_paths(backend: JsonlSessionStore, key: str) -> dict[str, Path]:
    paths = _lock_paths(backend, key)
    execution_dir = paths["execution"].parent
    execution_dir.mkdir(mode=0o700, exist_ok=True)
    execution_dir.chmod(0o700)
    for path in paths.values():
        path.parent.mkdir(mode=0o700, exist_ok=True)
        path.touch(exist_ok=True)
    return paths


def _assert_code(action, code: str) -> CheckedSessionAuthorityError:
    with pytest.raises(CheckedSessionAuthorityError) as captured:
        action()
    error = captured.value
    assert error.code == code
    assert error.__cause__ is None
    assert error.__context__ is None
    assert str(error) == f"checked session authority: {code}"
    return error


def _filelock_probe(path: str, result_queue) -> None:
    lock = FileLock(path, timeout=0)
    try:
        lock.acquire()
    except Timeout:
        result_queue.put(False)
    else:
        result_queue.put(True)
        lock.release()


def _hold_filelock(path: str, ready, release) -> None:
    lock = FileLock(path, timeout=0)
    with lock:
        ready.set()
        if not release.wait(10):
            raise Timeout("bounded lock holder timed out")


def test_checked_authority_is_factory_only_bound_and_lock_only(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    authority = open_checked_session(backend, "telegram:chat")
    assert type(authority) is CheckedSessionAuthority
    assert not hasattr(authority, "read_source")
    assert not hasattr(authority, "open_file")
    assert not hasattr(authority, "lock_file")
    with pytest.raises(TypeError):
        CheckedSessionAuthority(None, backend, "key", "a2V5", str(backend.sessions_dir), ())
    with pytest.raises(TypeError):
        import copy

        copy.copy(authority)
    with pytest.raises(TypeError):
        import pickle

        pickle.dumps(authority)
    authority.close()
    authority.close()
    _assert_code(authority.require_current, "closed")


def test_existing_supported_modes_are_unchanged_and_lock_files_persist(tmp_path: Path) -> None:
    key = "telegram:unchanged"
    backend = _backend(tmp_path)
    paths = _prepare_lock_paths(backend, key)
    for mode in (0o600, 0o604, 0o640, 0o644):
        path = paths["execution"]
        path.chmod(mode)
        before = path.stat()
        contents = path.read_bytes()
        with open_checked_session(backend, key) as authority:
            authority.require_current()
        after = path.stat()
        assert stat.S_IMODE(after.st_mode) == mode
        assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
        assert path.read_bytes() == contents == b""
    assert all(path.exists() for path in paths.values())


@pytest.mark.parametrize("lock_name", ["migration", "execution", "session"])
@pytest.mark.parametrize("mode", [0o600, 0o604, 0o640, 0o644])
def test_each_existing_lock_mode_is_accepted_without_mode_change(
    tmp_path: Path, lock_name: str, mode: int
) -> None:
    key = "matrix:key"
    backend = _backend(tmp_path)
    paths = _prepare_lock_paths(backend, key)
    path = paths[lock_name]
    path.chmod(mode)
    before = path.stat()
    with open_checked_session(backend, key) as authority:
        authority.require_current()
    after = path.stat()
    assert stat.S_IMODE(after.st_mode) == mode
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert after.st_nlink == 1 and after.st_size == 0


@pytest.mark.parametrize("lock_name", ["migration", "execution", "session"])
@pytest.mark.parametrize("unsafe_kind", ["symlink", "directory", "mode", "hardlink", "nonempty"])
def test_unsafe_existing_lock_entries_are_rejected_untouched(
    tmp_path: Path, lock_name: str, unsafe_kind: str
) -> None:
    key = "unsafe:key"
    backend = _backend(tmp_path)
    paths = _prepare_lock_paths(backend, key)
    path = paths[lock_name]
    path.unlink()
    marker = tmp_path / f"marker-{lock_name}-{unsafe_kind}"
    if unsafe_kind == "symlink":
        marker.write_bytes(b"target")
        path.symlink_to(marker)
    elif unsafe_kind == "directory":
        path.mkdir()
    elif unsafe_kind == "mode":
        path.touch()
        path.chmod(0o641)
    elif unsafe_kind == "hardlink":
        marker.write_bytes(b"")
        os.link(marker, path)
    else:
        path.write_bytes(b"must stay untouched")

    before = path.lstat()
    before_contents = path.read_bytes() if stat.S_ISREG(before.st_mode) else None
    before_target = os.readlink(path) if stat.S_ISLNK(before.st_mode) else None
    authority = open_checked_session(backend, key)
    _assert_code(authority.__enter__, "unsafe")
    after = path.lstat()
    assert (after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_size) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
    )
    if stat.S_ISREG(before.st_mode):
        assert path.read_bytes() == before_contents
    if stat.S_ISLNK(before.st_mode):
        assert os.readlink(path) == before_target
    authority.close()


@pytest.mark.parametrize("lock_name", ["execution", "session"])
def test_second_and_third_lock_failures_release_preceding_lock(tmp_path: Path, lock_name: str) -> None:
    key = "release:prior"
    backend = _backend(tmp_path)
    paths = _prepare_lock_paths(backend, key)
    failed_path = paths[lock_name]
    failed_path.unlink()
    failed_path.write_bytes(b"unsafe")
    authority = open_checked_session(backend, key)
    _assert_code(authority.__enter__, "unsafe")
    with FileLock(str(paths["migration"]), timeout=0):
        pass
    authority.close()


def test_missing_execution_directory_is_created_private_and_persistent(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "directory:normal"
    execution_dir = _lock_paths(backend, key)["execution"].parent
    assert not execution_dir.exists()
    with open_checked_session(backend, key) as authority:
        authority.require_current()
        assert stat.S_IMODE(execution_dir.stat().st_mode) == 0o700
    assert execution_dir.is_dir()
    assert stat.S_IMODE(execution_dir.stat().st_mode) == 0o700


def test_restrictive_umask_leaves_unsafe_execution_directory_without_repair(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "directory:umask"
    execution_dir = _lock_paths(backend, key)["execution"].parent
    authority = open_checked_session(backend, key)
    old_umask = os.umask(0o177)
    try:
        _assert_code(authority.__enter__, "unsafe")
    finally:
        os.umask(old_umask)
    parent_fd = os.open(backend.sessions_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        directory_stat = os.stat(".execution-locks", dir_fd=parent_fd, follow_symlinks=False)
        assert stat.S_IMODE(directory_stat.st_mode) == 0o600
        execution_fd = os.open(
            ".execution-locks", os.O_RDONLY | os.O_DIRECTORY, dir_fd=parent_fd
        )
        try:
            assert os.listdir(execution_fd) == []
        finally:
            os.close(execution_fd)
    finally:
        os.close(parent_fd)
    authority.close()
    execution_dir.chmod(0o700)


def test_safe_same_uid_directory_substitution_before_baseline_is_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "directory:adoption"
    execution_dir = _lock_paths(backend, key)["execution"].parent
    displaced = execution_dir.with_name(".execution-locks-displaced")
    real_mkdir = checked_source.os.mkdir
    replaced = False

    def substitute(name, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        real_mkdir(name, mode, dir_fd=dir_fd)
        if name == ".execution-locks" and not replaced:
            replaced = True
            os.rename(execution_dir, displaced)
            execution_dir.mkdir(mode=0o700)
            execution_dir.chmod(0o700)

    supported = set(checked_source.os.supports_dir_fd)
    monkeypatch.setattr(checked_source.os, "mkdir", substitute)
    monkeypatch.setattr(checked_source.os, "supports_dir_fd", supported | {substitute})
    with open_checked_session(backend, key) as authority:
        authority.require_current()
    assert replaced and execution_dir.is_dir() and displaced.is_dir()
    assert stat.S_IMODE(execution_dir.stat().st_mode) == 0o700


def _entered(backend: JsonlSessionStore, key: str) -> CheckedSessionAuthority:
    authority = open_checked_session(backend, key)
    authority.__enter__()
    return authority


@pytest.mark.parametrize("drift", ["root", "ancestor", "execution-edge", "root-mode"])
def test_directory_identity_owner_mode_and_edges_are_revalidated(
    tmp_path: Path, drift: str
) -> None:
    backend = _backend(tmp_path)
    key = "changed:directory"
    authority = _entered(backend, key)
    session_dir = backend.sessions_dir
    old_session = session_dir.with_name("sessions-old")
    old_parent = session_dir.parent.with_name("sessions-parent-old")
    replacement = session_dir.with_name("replacement")
    try:
        if drift == "root":
            os.rename(session_dir, old_session)
            session_dir.mkdir(mode=0o700)
        elif drift == "ancestor":
            parent = session_dir.parent
            os.rename(parent, old_parent)
            parent.mkdir(mode=0o700)
            session_dir.mkdir(mode=0o700)
        elif drift == "execution-edge":
            execution_dir = _lock_paths(backend, key)["execution"].parent
            os.rename(execution_dir, replacement)
            execution_dir.mkdir(mode=0o700)
        else:
            original_mode = stat.S_IMODE(session_dir.stat().st_mode)
            new_mode = 0o700 if original_mode != 0o700 else 0o710
            session_dir.chmod(new_mode)
        _assert_code(authority.require_current, "changed")
    finally:
        authority.close()
        if drift == "root-mode" and session_dir.exists():
            session_dir.chmod(0o755)


def test_opened_lock_descriptor_is_closed_after_fstat_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "fd:fstat"
    migration = _lock_paths(backend, key)["migration"]
    real_open = os.open
    real_fstat = os.fstat
    opened: list[int] = []
    injected = False

    def tracked_open(path, *args, **kwargs):
        descriptor = real_open(path, *args, **kwargs)
        if path == migration.name:
            opened.append(descriptor)
        return descriptor

    def failing_fstat(descriptor):
        nonlocal injected
        if opened and descriptor == opened[0] and not injected:
            injected = True
            raise OSError(errno.EIO, "untrusted detail")
        return real_fstat(descriptor)

    supported = set(checked_source.os.supports_dir_fd)
    monkeypatch.setattr(checked_source.os, "open", tracked_open)
    monkeypatch.setattr(checked_source.os, "supports_dir_fd", supported | {tracked_open})
    monkeypatch.setattr(checked_source.os, "fstat", failing_fstat)
    authority = open_checked_session(backend, key)
    error = _assert_code(authority.__enter__, "io")
    assert error.__cause__ is None and error.__context__ is None
    assert injected and opened
    with pytest.raises(OSError) as captured:
        real_fstat(opened[0])
    assert captured.value.errno == errno.EBADF


def test_opened_lock_descriptor_is_closed_after_flock_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "fd:flock"
    migration = _lock_paths(backend, key)["migration"]
    real_open = os.open
    real_fstat = os.fstat
    real_flock = checked_source.fcntl.flock
    opened: list[int] = []
    injected = False

    def tracked_open(path, *args, **kwargs):
        descriptor = real_open(path, *args, **kwargs)
        if path == migration.name:
            opened.append(descriptor)
        return descriptor

    def failing_flock(descriptor, flags):
        nonlocal injected
        if descriptor not in opened:
            return real_flock(descriptor, flags)
        if flags == checked_source.fcntl.LOCK_EX | checked_source.fcntl.LOCK_NB and not injected:
            injected = True
            raise OSError(errno.EIO, "untrusted detail")
        return real_flock(descriptor, flags)

    supported = set(checked_source.os.supports_dir_fd)
    monkeypatch.setattr(checked_source.os, "open", tracked_open)
    monkeypatch.setattr(checked_source.os, "supports_dir_fd", supported | {tracked_open})
    monkeypatch.setattr(checked_source.fcntl, "flock", failing_flock)
    authority = open_checked_session(backend, key)
    _assert_code(authority.__enter__, "io")
    assert injected and opened
    with pytest.raises(OSError) as captured:
        real_fstat(opened[0])
    assert captured.value.errno == errno.EBADF


@pytest.mark.parametrize("lock_name", ["migration", "execution", "session"])
def test_checked_lock_contends_with_filelock_in_separate_processes_both_ways(
    tmp_path: Path, lock_name: str
) -> None:
    key = "process:contention"
    backend = _backend(tmp_path)
    paths = _prepare_lock_paths(backend, key)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    authority = _entered(backend, key)
    contender = context.Process(target=_filelock_probe, args=(str(paths[lock_name]), result_queue))
    try:
        contender.start()
        assert result_queue.get(timeout=10) is False
        contender.join(10)
        assert contender.exitcode == 0
    finally:
        if contender.is_alive():
            contender.terminate()
            contender.join(5)
        authority.close()
        result_queue.close()

    ready, release = context.Event(), context.Event()
    holder = context.Process(target=_hold_filelock, args=(str(paths[lock_name]), ready, release))
    try:
        holder.start()
        assert ready.wait(10)
        blocked = open_checked_session(backend, key)
        _assert_code(blocked.__enter__, "busy")
        blocked.close()
    finally:
        release.set()
        holder.join(10)
        if holder.is_alive():
            holder.terminate()
            holder.join(5)
        assert holder.exitcode == 0


def test_wrong_thread_is_rejected_without_consuming_owner_authority(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    authority = _entered(backend, "thread:bound")
    results: list[str] = []

    def use_from_other_thread() -> None:
        try:
            authority.require_current()
        except CheckedSessionAuthorityError as error:
            assert error.__context__ is None and error.__cause__ is None
            results.append(error.code)

    thread = threading.Thread(target=use_from_other_thread)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    assert results == ["invalid"]
    authority.require_current()
    authority.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_child_invalidates_without_unlock_and_repeated_close_survives_fd_reuse(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    authority = _entered(backend, "fork:bound")
    migration = _lock_paths(backend, "fork:bound")["migration"]
    read_fd, write_fd = os.pipe()
    authority._mutex.acquire()
    try:
        child_pid = os.fork()
    except BaseException:
        authority._mutex.release()
        os.close(read_fd)
        os.close(write_fd)
        raise
    if child_pid == 0:
        try:
            os.close(read_fd)
            try:
                authority.require_current()
            except CheckedSessionAuthorityError as error:
                if error.code != "invalid" or error.__context__ is not None:
                    os._exit(3)
            reused_fd = os.open("/dev/null", os.O_RDONLY)
            authority.close()
            authority.close()
            os.fstat(reused_fd)
            os.close(reused_fd)
            os.write(write_fd, b"ok")
            os._exit(0)
        except BaseException:
            os._exit(4)
    os.close(write_fd)
    try:
        assert os.read(read_fd, 2) == b"ok"
        _, status = os.waitpid(child_pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        probe = os.open(migration, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                checked_source.fcntl.flock(
                    probe, checked_source.fcntl.LOCK_EX | checked_source.fcntl.LOCK_NB
                )
        finally:
            os.close(probe)
    finally:
        os.close(read_fd)
        authority._mutex.release()
        authority.close()


def test_successful_exit_revalidates_and_source_bytes_are_not_read_or_changed(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "source:opaque"
    source_path = backend.get_session_path(key)
    source_path.write_bytes(b"private source content\n")
    before = source_path.read_bytes()
    with open_checked_session(backend, key) as authority:
        authority.require_current()
    assert source_path.read_bytes() == before
    assert set(path.name for path in backend.sessions_dir.iterdir()) >= {
        source_path.name,
        ".session-files.lock",
        ".execution-locks",
    }


def test_public_factory_errors_are_content_free_without_exception_chains(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    error = _assert_code(lambda: open_checked_session(backend, ""), "invalid")
    assert str(tmp_path) not in str(error)


def test_new_persistent_lock_files_are_private_empty_and_never_removed(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "new:files"
    paths = _lock_paths(backend, key)
    paths["migration"].unlink()
    paths["session"].unlink()
    with open_checked_session(backend, key) as authority:
        authority.require_current()
    for path in paths.values():
        value = path.stat()
        assert stat.S_ISREG(value.st_mode)
        assert stat.S_IMODE(value.st_mode) == 0o600
        assert value.st_uid == os.geteuid() and value.st_nlink == 1 and value.st_size == 0
    assert all(path.exists() for path in paths.values())


def test_unsafe_same_uid_substitution_before_execution_baseline_is_left_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "directory:unsafe-adoption"
    execution_dir = _lock_paths(backend, key)["execution"].parent
    displaced = execution_dir.with_name(".execution-locks-displaced")
    target = execution_dir.with_name("execution-target")
    target.mkdir(mode=0o700)
    real_mkdir = checked_source.os.mkdir
    replaced = False

    def substitute(name, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        real_mkdir(name, mode, dir_fd=dir_fd)
        if name == ".execution-locks" and not replaced:
            replaced = True
            os.rename(execution_dir, displaced)
            execution_dir.symlink_to(target, target_is_directory=True)

    supported = set(checked_source.os.supports_dir_fd)
    monkeypatch.setattr(checked_source.os, "mkdir", substitute)
    monkeypatch.setattr(checked_source.os, "supports_dir_fd", supported | {substitute})
    authority = open_checked_session(backend, key)
    _assert_code(authority.__enter__, "unsafe")
    assert replaced and execution_dir.is_symlink() and displaced.is_dir()
    assert execution_dir.resolve() == target
    authority.close()


def test_successful_context_exit_rejects_post_baseline_substitution(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "exit:drift"
    authority = _entered(backend, key)
    execution_dir = _lock_paths(backend, key)["execution"].parent
    displaced = execution_dir.with_name("execution-locks-original")
    os.rename(execution_dir, displaced)
    execution_dir.mkdir(mode=0o700)
    execution_dir.chmod(0o700)
    _assert_code(lambda: authority.__exit__(None, None, None), "changed")
    authority.close()


def test_context_lifecycle_allows_only_one_entry(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    authority = open_checked_session(backend, "lifecycle:once")
    assert authority.__enter__() is authority
    _assert_code(authority.__enter__, "invalid")
    authority.require_current()
    authority.close()
    authority.close()
    _assert_code(authority.require_current, "closed")


def test_authority_import_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    import subprocess
    import sys

    script = """
import importlib
import os
import nanobot.session.checked_source as module

def forbidden(*args, **kwargs):
    raise AssertionError('module import attempted filesystem I/O')

for name in ('open', 'stat', 'mkdir', 'fstat', 'close'):
    setattr(os, name, forbidden)
importlib.reload(module)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_existing_execution_directory_with_wrong_mode_is_rejected_untouched(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "directory:existing-mode"
    execution_dir = _lock_paths(backend, key)["execution"].parent
    execution_dir.mkdir(mode=0o700)
    execution_dir.chmod(0o750)
    before = execution_dir.stat()
    authority = open_checked_session(backend, key)
    _assert_code(authority.__enter__, "unsafe")
    after = execution_dir.stat()
    assert stat.S_IMODE(after.st_mode) == 0o750
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert not _lock_paths(backend, key)["execution"].exists()
    authority.close()


def test_existing_execution_directory_symlink_is_rejected_without_following(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    key = "directory:symlink"
    execution_dir = _lock_paths(backend, key)["execution"].parent
    target = tmp_path / "external-locks"
    target.mkdir(mode=0o700)
    execution_dir.symlink_to(target, target_is_directory=True)
    before = execution_dir.lstat()
    authority = open_checked_session(backend, key)
    _assert_code(authority.__enter__, "unsafe")
    after = execution_dir.lstat()
    assert stat.S_ISLNK(after.st_mode)
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert list(target.iterdir()) == []
    authority.close()


def test_authority_entry_is_bound_to_issuing_thread_object(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    authority = open_checked_session(backend, "thread:entry")
    results: list[str] = []

    def enter_from_other_thread() -> None:
        try:
            authority.__enter__()
        except CheckedSessionAuthorityError as error:
            assert error.__context__ is None and error.__cause__ is None
            results.append(error.code)

    thread = threading.Thread(target=enter_from_other_thread)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    assert results == ["invalid"]
    with authority:
        authority.require_current()


def _fd_snapshot() -> frozenset[str]:
    return frozenset(os.listdir("/dev/fd"))


def _path_snapshot(path: Path) -> tuple[int, int, int, int, int, bytes]:
    value = path.lstat()
    contents = path.read_bytes() if stat.S_ISREG(value.st_mode) else b""
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        contents,
    )


def _assert_initial_directory_rejected_without_side_effects(
    backend: JsonlSessionStore, key: str
) -> None:
    source = backend.get_session_path(key)
    source.write_bytes(b"unchanged source payload")
    locks = _lock_paths(backend, key)
    lock_paths = (locks["migration"], locks["session"])
    before_locks = {path: _path_snapshot(path) for path in lock_paths}
    before_source = source.read_bytes()
    execution_dir = locks["execution"].parent
    assert not execution_dir.exists()
    before_fds = _fd_snapshot()
    authority = open_checked_session(backend, key)
    _assert_code(authority.__enter__, "unsafe")
    after_fds = _fd_snapshot()
    assert after_fds == before_fds
    assert {path: _path_snapshot(path) for path in lock_paths} == before_locks
    assert source.read_bytes() == before_source
    assert not execution_dir.exists()
    authority.close()


def test_initial_0777_namespace_is_rejected_without_lock_or_source_changes(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    original_mode = stat.S_IMODE(backend.sessions_dir.stat().st_mode)
    backend.sessions_dir.chmod(0o777)
    try:
        _assert_initial_directory_rejected_without_side_effects(backend, "policy:namespace")
    finally:
        backend.sessions_dir.chmod(original_mode)


def test_initial_0777_migration_parent_is_rejected_without_lock_or_source_changes(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    parent = backend.sessions_dir.parent
    original_mode = stat.S_IMODE(parent.stat().st_mode)
    parent.chmod(0o777)
    try:
        _assert_initial_directory_rejected_without_side_effects(backend, "policy:parent")
    finally:
        parent.chmod(original_mode)


def test_initial_nonsticky_writable_intermediate_ancestor_is_rejected(
    tmp_path: Path,
) -> None:
    intermediate = tmp_path / "world-writable-ancestor"
    intermediate.mkdir()
    intermediate.chmod(0o777)
    backend = JsonlSessionStore(
        tmp_path / "workspace",
        sessions_root=intermediate / "sessions-root",
    )
    _assert_initial_directory_rejected_without_side_effects(backend, "policy:ancestor")


def test_normal_0755_namespace_and_0700_migration_parent_are_accepted(
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    backend.sessions_dir.chmod(0o755)
    backend.sessions_dir.parent.chmod(0o700)
    assert stat.S_IMODE(backend.sessions_dir.stat().st_mode) == 0o755
    assert stat.S_IMODE(backend.sessions_dir.parent.stat().st_mode) == 0o700
    with open_checked_session(backend, "policy:normal") as authority:
        authority.require_current()


def test_root_owned_sticky_temp_ancestor_is_accepted_when_available() -> None:
    temp_root = Path("/tmp").resolve()
    try:
        value = temp_root.stat()
    except OSError as exc:
        pytest.skip(f"temporary root is unavailable: {type(exc).__name__}")
    if value.st_uid != 0 or not (stat.S_IMODE(value.st_mode) & stat.S_ISVTX):
        pytest.skip("no root-owned sticky temporary ancestor is available")
    with tempfile.TemporaryDirectory(prefix="checked-source-", dir=temp_root) as temporary:
        base = Path(temporary)
        backend = JsonlSessionStore(base / "workspace", sessions_root=base / "sessions")
        with open_checked_session(backend, "policy:sticky-temp") as authority:
            authority.require_current()


def test_initial_namespace_special_permission_bit_is_rejected(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    original_mode = stat.S_IMODE(backend.sessions_dir.stat().st_mode)
    backend.sessions_dir.chmod(0o1755)
    try:
        _assert_initial_directory_rejected_without_side_effects(
            backend, "policy:special-mode"
        )
    finally:
        backend.sessions_dir.chmod(original_mode)
