from __future__ import annotations

import errno
import multiprocessing
import os
import stat
import threading
from pathlib import Path

import pytest

import nanobot.session.checked_source as checked_source
from nanobot.session.checked_source import CheckedSessionAuthorityError, open_checked_session
from nanobot.session.manager import JsonlSessionStore

_MAX_SOURCE_BYTES = 64 * 1024 * 1024


def _backend(tmp_path: Path) -> JsonlSessionStore:
    return JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")


def _source(backend: JsonlSessionStore, key: str) -> Path:
    return backend.get_session_path(key)


def _entered(backend: JsonlSessionStore, key: str):
    authority = open_checked_session(backend, key)
    authority.__enter__()
    return authority


def _assert_error(action, code: str) -> CheckedSessionAuthorityError:
    with pytest.raises(CheckedSessionAuthorityError) as captured:
        action()
    error = captured.value
    assert error.code == code
    assert error.__cause__ is None and error.__context__ is None
    assert str(error) == f"checked session authority: {code}"
    return error


def test_replace_requires_read_and_bytes_argument(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    source = _source(backend, "replace:prerequisite")
    source.write_bytes(b"old")

    with open_checked_session(backend, "replace:prerequisite") as authority:
        _assert_error(lambda: authority.replace_source_bytes(b"new"), "invalid")
        _assert_error(lambda: authority.replace_source_bytes(b"x" * (_MAX_SOURCE_BYTES + 1)), "invalid")
        authority.read_source_bytes()
        _assert_error(lambda: authority.replace_source_bytes(bytearray(b"new")), "invalid")
        assert authority.replace_source_bytes(b"new") is None

    assert source.read_bytes() == b"new"


def test_replace_is_exact_private_and_one_shot(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "replace:success"
    source = _source(backend, key)
    old = b'{"role":"user"}\n\xff'
    new = b"replacement\x00bytes\n"
    source.write_bytes(old)
    old_inode = source.stat().st_ino

    with open_checked_session(backend, key) as authority:
        assert authority.read_source_bytes() == old
        assert authority.replace_source_bytes(new) is None
        _assert_error(lambda: authority.replace_source_bytes(b"again"), "invalid")
        authority.read_source_bytes()
        _assert_error(lambda: authority.replace_source_bytes(b"second attempt"), "invalid")

    assert source.read_bytes() == new
    assert source.stat().st_ino != old_inode
    assert stat.S_IMODE(source.stat().st_mode) == 0o600


def test_replace_rejects_oversized_input_without_creating_temp(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "replace:limit"
    source = _source(backend, key)
    source.write_bytes(b"old")
    oversized = b"x" * (_MAX_SOURCE_BYTES + 1)

    with open_checked_session(backend, key) as authority:
        authority.read_source_bytes()
        _assert_error(lambda: authority.replace_source_bytes(oversized), "limit")
        assert authority.replace_source_bytes(b"new") is None

    assert source.read_bytes() == b"new"
    assert not any(item.name.startswith(".checked-source-") for item in source.parent.iterdir())


def test_replace_accepts_exact_64_mib_limit(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "replace:at-limit"
    source = _source(backend, key)
    source.write_bytes(b"prior")
    replacement = b"x" * _MAX_SOURCE_BYTES

    with open_checked_session(backend, key) as authority:
        authority.read_source_bytes()
        assert authority.replace_source_bytes(replacement) is None

    assert source.stat().st_size == _MAX_SOURCE_BYTES
    with source.open("rb") as handle:
        assert handle.read(16) == b"x" * 16
        handle.seek(-16, os.SEEK_END)
        assert handle.read() == b"x" * 16



def test_same_byte_inode_substitution_after_read_is_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "replace:inode-race"
    source = _source(backend, key)
    source.write_bytes(b"same")

    temp_opened = False
    real_open = checked_source.os.open

    def track_temp_open(name, flags, mode=0o777, *, dir_fd=None):
        nonlocal temp_opened
        if flags & os.O_CREAT and flags & os.O_EXCL:
            temp_opened = True
        return real_open(name, flags, mode, dir_fd=dir_fd)

    with open_checked_session(backend, key) as authority:
        assert authority.read_source_bytes() == b"same"
        displaced = source.with_suffix(".old")
        os.replace(source, displaced)
        source.write_bytes(b"same")
        monkeypatch.setattr(checked_source.os, "open", track_temp_open)
        _assert_error(lambda: authority.replace_source_bytes(b"new"), "changed")

    assert not temp_opened
    assert source.read_bytes() == displaced.read_bytes() == b"same"
    assert source.stat().st_ino != displaced.stat().st_ino


def test_same_inode_mutation_after_read_is_changed(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "replace:content-race"
    source = _source(backend, key)
    source.write_bytes(b"before")

    with open_checked_session(backend, key) as authority:
        assert authority.read_source_bytes() == b"before"
        with source.open("r+b") as handle:
            handle.write(b"after!")
        _assert_error(lambda: authority.replace_source_bytes(b"new"), "changed")

    assert source.read_bytes() == b"after!"


def test_missing_source_after_read_is_changed_without_creating_temp(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "replace:missing-after-read"
    source = _source(backend, key)
    source.write_bytes(b"prior")

    with open_checked_session(backend, key) as authority:
        authority.read_source_bytes()
        source.unlink()
        _assert_error(lambda: authority.replace_source_bytes(b"new"), "changed")

    assert not source.exists()
    assert not any(item.name.startswith(".checked-source-") for item in source.parent.iterdir())



def test_source_race_before_second_full_check_retains_temp_without_replacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "replace:pre-rename-race"
    source = _source(backend, key)
    source.write_bytes(b"prior")
    real_write = checked_source.os.write
    real_replace = checked_source.os.replace
    real_open = checked_source.os.open
    temp_descriptors: set[int] = set()
    attempted_replace = False
    mutated = False

    def track_temp_open(name, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(name, flags, mode, dir_fd=dir_fd)
        if flags & os.O_CREAT and flags & os.O_EXCL:
            temp_descriptors.add(descriptor)
        return descriptor

    def mutate_after_temp_write(descriptor: int, data) -> int:
        nonlocal mutated
        count = real_write(descriptor, data)
        if descriptor in temp_descriptors and not mutated:
            source.write_bytes(b"raced")
            mutated = True
        return count

    def track_replace(*args, **kwargs):
        nonlocal attempted_replace
        attempted_replace = True
        return real_replace(*args, **kwargs)

    with open_checked_session(backend, key) as authority:
        authority.read_source_bytes()
        monkeypatch.setattr(checked_source.os, "open", track_temp_open)
        monkeypatch.setattr(checked_source.os, "write", mutate_after_temp_write)
        monkeypatch.setattr(checked_source.os, "replace", track_replace)
        _assert_error(lambda: authority.replace_source_bytes(b"new"), "changed")

    assert mutated and not attempted_replace
    assert source.read_bytes() == b"raced"
    assert list(source.parent.glob(".checked-source-*.tmp"))


def test_temporary_name_collision_preserves_existing_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "replace:collision"
    source = _source(backend, key)
    source.write_bytes(b"prior")
    names = iter(("collision", "collision", "replacement"))
    monkeypatch.setattr(checked_source.secrets, "token_hex", lambda _size: next(names))

    with open_checked_session(backend, key) as authority:
        collision = source.with_name(checked_source._temporary_name())
        collision.write_bytes(b"keep me")
        old_mode = stat.S_IMODE(collision.stat().st_mode)
        authority.read_source_bytes()
        assert authority.replace_source_bytes(b"new") is None

    assert collision.read_bytes() == b"keep me"
    assert stat.S_IMODE(collision.stat().st_mode) == old_mode
    assert source.read_bytes() == b"new"
    assert not source.with_name(".checked-source-replacement.tmp").exists()


def test_new_temp_descriptor_is_private_and_only_it_is_fchmodded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "replace:private-temp"
    source = _source(backend, key)
    source.write_bytes(b"prior")
    temp_descriptors: list[int] = []
    opened_flags: list[tuple[int, int]] = []
    chmod_descriptors: list[int] = []
    real_open = checked_source.os.open
    real_fchmod = checked_source.os.fchmod

    def track_open(name, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(name, flags, mode, dir_fd=dir_fd)
        if flags & os.O_CREAT and flags & os.O_EXCL:
            temp_descriptors.append(descriptor)
            opened_flags.append((flags, mode))
        return descriptor

    def track_fchmod(descriptor: int, mode: int) -> None:
        chmod_descriptors.append(descriptor)
        real_fchmod(descriptor, mode)

    with open_checked_session(backend, key) as authority:
        authority.read_source_bytes()
        monkeypatch.setattr(checked_source.os, "open", track_open)
        monkeypatch.setattr(checked_source.os, "fchmod", track_fchmod)
        old_umask = os.umask(0o777)
        try:
            assert authority.replace_source_bytes(b"new") is None
        finally:
            os.umask(old_umask)

    flags, mode = opened_flags[0]
    assert mode == 0o600
    assert flags & os.O_CREAT and flags & os.O_EXCL
    assert flags & os.O_NOFOLLOW and flags & os.O_CLOEXEC and flags & os.O_NONBLOCK
    assert chmod_descriptors == temp_descriptors
    assert stat.S_IMODE(source.stat().st_mode) == 0o600



def test_temp_path_substitution_is_rejected_without_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "replace:temp-substitution"
    source = _source(backend, key)
    source.write_bytes(b"prior")
    real_open = checked_source.os.open
    real_write = checked_source.os.write
    real_unlink = checked_source.os.unlink
    temp_names: list[str] = []
    substituted = False
    unlinks: list[str] = []

    def tracked_open(name, flags, mode=0o777, *, dir_fd=None):
        if flags & os.O_CREAT and flags & os.O_EXCL:
            temp_names.append(name)
        return real_open(name, flags, mode, dir_fd=dir_fd)

    def substitute_after_write(descriptor: int, data) -> int:
        nonlocal substituted
        result = real_write(descriptor, data)
        if not substituted and temp_names:
            temp_name = temp_names[-1]
            displaced = source.with_name(f"{temp_name}.displaced")
            os.replace(source.with_name(temp_name), displaced)
            source.with_name(temp_name).write_bytes(b"substitute")
            substituted = True
        return result

    def track_unlink(name, *, dir_fd=None):
        unlinks.append(name)
        return real_unlink(name, dir_fd=dir_fd)

    with open_checked_session(backend, key) as authority:
        authority.read_source_bytes()
        monkeypatch.setattr(checked_source.os, "open", tracked_open)
        monkeypatch.setattr(checked_source.os, "write", substitute_after_write)
        monkeypatch.setattr(checked_source.os, "unlink", track_unlink)
        _assert_error(lambda: authority.replace_source_bytes(b"new"), "changed")

    assert substituted and temp_names
    assert source.read_bytes() == b"prior"
    assert unlinks == []
    assert source.with_name(f"{temp_names[0]}.displaced").exists()
    assert source.with_name(temp_names[0]).read_bytes() == b"substitute"


def test_short_write_and_eintr_produce_exact_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "replace:short-write"
    source = _source(backend, key)
    source.write_bytes(b"prior")
    real_write = checked_source.os.write
    calls = 0

    def interrupted_then_short(descriptor: int, data) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.EINTR, "injected")
        return real_write(descriptor, data[:2])

    with open_checked_session(backend, key) as authority:
        authority.read_source_bytes()
        monkeypatch.setattr(checked_source.os, "write", interrupted_then_short)
        assert authority.replace_source_bytes(b"new contents") is None

    assert calls > 2
    assert source.read_bytes() == b"new contents"


def test_write_failure_is_pre_replace_io_and_leaves_source_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "replace:write-failure"
    source = _source(backend, key)
    source.write_bytes(b"prior")
    monkeypatch.setattr(
        checked_source.os,
        "write",
        lambda *_args: (_ for _ in ()).throw(OSError(errno.EIO, "injected")),
    )

    with open_checked_session(backend, key) as authority:
        authority.read_source_bytes()
        _assert_error(lambda: authority.replace_source_bytes(b"new"), "io")

    assert source.read_bytes() == b"prior"
    assert list(source.parent.glob(".checked-source-*.tmp"))


def test_temp_fsync_failure_is_pre_replace_io_and_leaves_source_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "replace:file-fsync"
    source = _source(backend, key)
    source.write_bytes(b"prior")
    monkeypatch.setattr(
        checked_source.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError(errno.EIO, "injected")),
    )

    with open_checked_session(backend, key) as authority:
        authority.read_source_bytes()
        _assert_error(lambda: authority.replace_source_bytes(b"new"), "io")

    assert source.read_bytes() == b"prior"
    assert list(source.parent.glob(".checked-source-*.tmp"))


def test_replace_failure_is_uncertain_and_never_restores_or_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "replace:syscall-failure"
    source = _source(backend, key)
    source.write_bytes(b"prior")
    real_unlink = checked_source.os.unlink
    unlinks: list[str] = []

    def track_unlink(name, *, dir_fd=None):
        unlinks.append(name)
        return real_unlink(name, dir_fd=dir_fd)

    with open_checked_session(backend, key) as authority:
        authority.read_source_bytes()
        monkeypatch.setattr(checked_source.os, "unlink", track_unlink)
        monkeypatch.setattr(
            checked_source.os,
            "replace",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError(errno.EIO, "injected")),
        )
        _assert_error(lambda: authority.replace_source_bytes(b"new"), "uncertain")

    assert source.read_bytes() == b"prior"
    assert unlinks == []
    assert list(source.parent.glob(".checked-source-*.tmp"))


def test_same_uid_temp_swap_at_replace_seam_is_uncertain_without_rollback_or_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "replace:replace-seam-swap"
    source = _source(backend, key)
    source.write_bytes(b"prior")
    real_replace = checked_source.os.replace
    real_unlink = checked_source.os.unlink
    unlinks: list[str] = []
    retained_names: list[str] = []
    substituted = False

    def swap_temp_then_replace(
        src, dst, *, src_dir_fd=None, dst_dir_fd=None
    ) -> None:
        nonlocal substituted
        if not substituted and src.startswith(".checked-source-") and dst == source.name:
            retained_name = f"{src}.retained"
            real_replace(src, retained_name, src_dir_fd=src_dir_fd, dst_dir_fd=src_dir_fd)
            retained_names.append(retained_name)
            source.with_name(src).write_bytes(b"same-uid substitute")
            substituted = True
        real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    def track_unlink(name, *, dir_fd=None):
        unlinks.append(name)
        return real_unlink(name, dir_fd=dir_fd)

    with open_checked_session(backend, key) as authority:
        authority.read_source_bytes()
        monkeypatch.setattr(checked_source.os, "replace", swap_temp_then_replace)
        monkeypatch.setattr(checked_source.os, "unlink", track_unlink)
        _assert_error(lambda: authority.replace_source_bytes(b"requested"), "uncertain")

    assert substituted
    assert source.read_bytes() == b"same-uid substitute"
    assert len(retained_names) == 1
    assert source.with_name(retained_names[0]).read_bytes() == b"requested"
    assert unlinks == []



def test_directory_fsync_failure_is_uncertain_after_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "replace:directory-fsync"
    source = _source(backend, key)
    source.write_bytes(b"prior")
    real_fsync = checked_source.os.fsync

    def fail_directory_sync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(errno.EIO, "injected")
        real_fsync(descriptor)

    with open_checked_session(backend, key) as authority:
        authority.read_source_bytes()
        monkeypatch.setattr(checked_source.os, "fsync", fail_directory_sync)
        _assert_error(lambda: authority.replace_source_bytes(b"new"), "uncertain")

    assert source.read_bytes() == b"new"


def _fork_replace_code(authority, result_queue) -> None:
    try:
        authority.replace_source_bytes(b"child")
    except CheckedSessionAuthorityError as error:
        result_queue.put((error.code, error.__cause__ is None, error.__context__ is None))


def test_temp_descriptor_is_detached_before_close_even_if_close_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    key = "replace:close-detach"
    source = _source(backend, key)
    source.write_bytes(b"prior")
    temp_descriptors: list[int] = []
    real_open = checked_source.os.open
    real_close = checked_source.os.close

    def track_open(name, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(name, flags, mode, dir_fd=dir_fd)
        if flags & os.O_CREAT and flags & os.O_EXCL:
            temp_descriptors.append(descriptor)
        return descriptor

    def close_then_raise(descriptor: int) -> None:
        real_close(descriptor)
        if temp_descriptors and descriptor == temp_descriptors[0]:
            raise OSError(errno.EIO, "injected")

    authority = _entered(backend, key)
    authority.read_source_bytes()
    monkeypatch.setattr(checked_source.os, "open", track_open)
    monkeypatch.setattr(checked_source.os, "close", close_then_raise)
    _assert_error(lambda: authority.replace_source_bytes(b"new"), "uncertain")
    assert source.read_bytes() == b"new"

    reused = os.open("/dev/null", os.O_RDONLY)
    try:
        assert reused == temp_descriptors[0]
        authority.close()
        os.fstat(reused)
    finally:
        real_close(reused)



def test_foreign_thread_and_fork_cannot_consume_replacement_evidence(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "replace:thread-fork"
    source = _source(backend, key)
    source.write_bytes(b"prior")
    authority = _entered(backend, key)
    authority.read_source_bytes()
    thread_codes: list[str] = []

    def replace_from_thread() -> None:
        try:
            authority.replace_source_bytes(b"thread")
        except CheckedSessionAuthorityError as error:
            thread_codes.append(error.code)

    thread = threading.Thread(target=replace_from_thread)
    thread.start()
    thread.join(5)
    assert not thread.is_alive() and thread_codes == ["invalid"]

    if "fork" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("fork")
        result_queue = context.Queue()
        process = context.Process(target=_fork_replace_code, args=(authority, result_queue))
        process.start()
        process.join(5)
        try:
            assert not process.is_alive() and process.exitcode == 0
            assert result_queue.get(timeout=2) == ("invalid", True, True)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            result_queue.close()

    assert authority.replace_source_bytes(b"parent") is None
    assert source.read_bytes() == b"parent"
    authority.close()
