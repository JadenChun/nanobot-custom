from __future__ import annotations

import errno
import inspect
import multiprocessing
import os
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import nanobot.session.checked_source as checked_source
from nanobot.session.checked_source import (
    CheckedSessionAuthorityError,
    open_checked_session,
)
from nanobot.session.manager import JsonlSessionStore

_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_READ_CHUNK = 64 * 1024


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
    assert error.__cause__ is None
    assert error.__context__ is None
    assert str(error) == f"checked session authority: {code}"
    return error


def _source_signature(path: Path) -> tuple[int, int, int, int, int, int, int, int, int]:
    value = path.lstat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def test_reads_exact_current_jsonl_and_detached_binary_bytes(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "telegram:room"
    source = _source(backend, key)
    payload = b'{"role":"user","content":"hello"}\n\xff\x80\n'
    source.write_bytes(payload)
    before = _source_signature(source)

    with open_checked_session(backend, key) as authority:
        result = authority.read_source_bytes()

    assert result == payload
    assert type(result) is bytes
    assert _source_signature(source) == before


def test_reads_empty_source_bytes_and_does_not_open_other_basename(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _backend(tmp_path)
    key = "empty:source"
    source = _source(backend, key)
    source.write_bytes(b"")
    other = backend.sessions_dir / "other.jsonl"
    other.write_bytes(b"not the requested source")
    called: list[str] = []
    real_open = checked_source.os.open

    def tracked_open(name, flags, mode=0o777, *, dir_fd=None):
        if name == source.name:
            called.append(name)
        return real_open(name, flags, mode, dir_fd=dir_fd)

    with open_checked_session(backend, key) as authority:
        monkeypatch.setattr(checked_source.os, "open", tracked_open)
        authority.read_source_bytes()

    assert called == [source.name]


def test_missing_canonical_source_fails_without_falling_back(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "source:missing"
    source = _source(backend, key)
    assert not source.exists()
    (backend.sessions_dir / "other.jsonl").write_bytes(b"must not be selected")

    with open_checked_session(backend, key) as authority:
        _assert_error(authority.read_source_bytes, "missing")


def test_reads_exactly_the_64_mib_limit(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "source:at-limit"
    source = _source(backend, key)
    with source.open("wb") as handle:
        handle.truncate(_MAX_SOURCE_BYTES)

    with open_checked_session(backend, key) as authority:
        result = authority.read_source_bytes()

    assert len(result) == _MAX_SOURCE_BYTES
    assert not any(result)


def test_rejects_initial_source_over_limit_before_open(tmp_path: Path, monkeypatch) -> None:
    backend = _backend(tmp_path)
    key = "source:over-limit"
    source = _source(backend, key)
    with source.open("wb") as handle:
        handle.truncate(_MAX_SOURCE_BYTES + 1)
    opened_source = False
    real_open = checked_source.os.open

    def tracked_open(name, flags, mode=0o777, *, dir_fd=None):
        nonlocal opened_source
        if name == source.name:
            opened_source = True
        return real_open(name, flags, mode, dir_fd=dir_fd)

    with open_checked_session(backend, key) as authority:
        monkeypatch.setattr(checked_source.os, "open", tracked_open)
        _assert_error(authority.read_source_bytes, "limit")

    assert not opened_source


def test_equal_byte_inode_substitution_before_open_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _backend(tmp_path)
    key = "source:before-open"
    source = _source(backend, key)
    payload = b"same exact bytes\n"
    source.write_bytes(payload)
    displaced = source.with_suffix(".old")
    real_open = checked_source.os.open
    substituted = False

    def replace_then_open(name, flags, mode=0o777, *, dir_fd=None):
        nonlocal substituted
        if name == source.name and not substituted:
            substituted = True
            os.replace(source, displaced)
            source.write_bytes(payload)
        return real_open(name, flags, mode, dir_fd=dir_fd)

    with open_checked_session(backend, key) as authority:
        monkeypatch.setattr(checked_source.os, "open", replace_then_open)
        _assert_error(authority.read_source_bytes, "changed")

    assert substituted
    assert source.read_bytes() == displaced.read_bytes() == payload
    assert source.stat().st_ino != displaced.stat().st_ino


def test_path_substitution_after_open_is_rejected(tmp_path: Path, monkeypatch) -> None:
    backend = _backend(tmp_path)
    key = "source:after-open"
    source = _source(backend, key)
    payload = b"same exact bytes\n"
    source.write_bytes(payload)
    displaced = source.with_suffix(".old")
    real_open = checked_source.os.open
    substituted = False

    def open_then_replace(name, flags, mode=0o777, *, dir_fd=None):
        nonlocal substituted
        descriptor = real_open(name, flags, mode, dir_fd=dir_fd)
        if name == source.name and not substituted:
            substituted = True
            os.replace(source, displaced)
            source.write_bytes(payload)
        return descriptor

    with open_checked_session(backend, key) as authority:
        monkeypatch.setattr(checked_source.os, "open", open_then_replace)
        _assert_error(authority.read_source_bytes, "changed")

    assert substituted
    assert source.read_bytes() == displaced.read_bytes() == payload


def test_same_inode_modification_after_first_chunk_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _backend(tmp_path)
    key = "source:modify-mid-read"
    source = _source(backend, key)
    source.write_bytes(b"a" * (_READ_CHUNK * 2))
    real_read = checked_source.os.read
    modified = False

    def read_then_modify(descriptor: int, size: int) -> bytes:
        nonlocal modified
        chunk = real_read(descriptor, size)
        if chunk and not modified:
            modified = True
            source.write_bytes(b"b" * (_READ_CHUNK * 2))
        return chunk

    with open_checked_session(backend, key) as authority:
        monkeypatch.setattr(checked_source.os, "read", read_then_modify)
        _assert_error(authority.read_source_bytes, "changed")

    assert modified
    assert source.read_bytes() == b"b" * (_READ_CHUNK * 2)


def test_same_inode_truncation_after_first_chunk_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _backend(tmp_path)
    key = "source:truncate-mid-read"
    source = _source(backend, key)
    source.write_bytes(b"a" * (_READ_CHUNK * 2))
    initial_inode = source.stat().st_ino
    real_read = checked_source.os.read
    truncated = False

    def read_then_truncate(descriptor: int, size: int) -> bytes:
        nonlocal truncated
        chunk = real_read(descriptor, size)
        if chunk and not truncated:
            truncated = True
            with source.open("wb"):
                pass
        return chunk

    with open_checked_session(backend, key) as authority:
        monkeypatch.setattr(checked_source.os, "read", read_then_truncate)
        _assert_error(authority.read_source_bytes, "changed")

    assert truncated
    assert source.stat().st_ino == initial_inode
    assert source.stat().st_size == 0


def test_early_eof_is_rejected_without_returning_partial_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _backend(tmp_path)
    key = "source:early-eof"
    source = _source(backend, key)
    source.write_bytes(b"some source bytes")

    with open_checked_session(backend, key) as authority:
        monkeypatch.setattr(checked_source.os, "read", lambda _fd, _size: b"")
        _assert_error(authority.read_source_bytes, "changed")


def test_growth_after_first_chunk_is_rejected(tmp_path: Path, monkeypatch) -> None:
    backend = _backend(tmp_path)
    key = "source:growth"
    source = _source(backend, key)
    source.write_bytes(b"a" * (_READ_CHUNK * 2))
    real_read = checked_source.os.read
    grew = False

    def read_then_grow(descriptor: int, size: int) -> bytes:
        nonlocal grew
        chunk = real_read(descriptor, size)
        if chunk and not grew:
            grew = True
            with source.open("ab") as handle:
                handle.write(b"growth")
        return chunk

    with open_checked_session(backend, key) as authority:
        monkeypatch.setattr(checked_source.os, "read", read_then_grow)
        _assert_error(authority.read_source_bytes, "changed")

    assert grew
    assert source.stat().st_size == (_READ_CHUNK * 2) + len(b"growth")


def test_short_reads_are_accumulated_and_eintr_is_retried(tmp_path: Path, monkeypatch) -> None:
    backend = _backend(tmp_path)
    key = "source:short-eintr"
    source = _source(backend, key)
    payload = bytes(range(251)) * 80
    source.write_bytes(payload)
    real_read = checked_source.os.read
    first = True

    def interrupted_then_short(descriptor: int, size: int) -> bytes:
        nonlocal first
        if first:
            first = False
            raise OSError(errno.EINTR, "private test detail")
        return real_read(descriptor, min(size, 13))

    with open_checked_session(backend, key) as authority:
        monkeypatch.setattr(checked_source.os, "read", interrupted_then_short)
        assert authority.read_source_bytes() == payload


def test_growth_cannot_make_reader_consume_more_than_limit_plus_one(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _backend(tmp_path)
    key = "source:bounded-read"
    source = _source(backend, key)
    source.write_bytes(b"small baseline")
    consumed = 0

    def oversized_stream(_descriptor: int, size: int) -> bytes:
        nonlocal consumed
        chunk = b"x" * size
        consumed += len(chunk)
        return chunk

    with open_checked_session(backend, key) as authority:
        monkeypatch.setattr(checked_source.os, "read", oversized_stream)
        _assert_error(authority.read_source_bytes, "limit")

    assert consumed == _MAX_SOURCE_BYTES + 1


def _prepare_unsafe_owner_mode_link_and_file_types(
    tmp_path: Path, monkeypatch, unsafe_kind: str
) -> None:
    backend = _backend(tmp_path)
    key = f"source:unsafe-{unsafe_kind}"
    source = _source(backend, key)
    source.write_bytes(b"unchanged")
    if unsafe_kind == "mode":
        source.chmod(0o664)
    elif unsafe_kind == "hardlink":
        os.link(source, source.with_suffix(".alias"))
    elif unsafe_kind == "symlink":
        source.unlink()
        source.symlink_to(tmp_path / "target")
        (tmp_path / "target").write_bytes(b"unchanged")
    elif unsafe_kind == "fifo":
        source.unlink()
        os.mkfifo(source)
    elif unsafe_kind == "directory":
        source.unlink()
        source.mkdir()

    if unsafe_kind in {"owner", "special-mode"}:
        real_stat = checked_source.os.stat

        def unsafe_stat(name, *args, **kwargs):
            result = real_stat(name, *args, **kwargs)
            if name == source.name and kwargs.get("dir_fd") is not None:
                mode = result.st_mode
                owner = result.st_uid
                if unsafe_kind == "owner":
                    owner = os.geteuid() + 1
                else:
                    mode |= stat.S_ISUID
                return SimpleNamespace(
                    st_mode=mode,
                    st_uid=owner,
                    st_nlink=result.st_nlink,
                    st_gid=result.st_gid,
                    st_dev=result.st_dev,
                    st_ino=result.st_ino,
                    st_size=result.st_size,
                    st_mtime_ns=result.st_mtime_ns,
                    st_ctime_ns=result.st_ctime_ns,
                )
            return result

    with open_checked_session(backend, key) as authority:
        if unsafe_kind in {"owner", "special-mode"}:
            monkeypatch.setattr(checked_source.os, "stat", unsafe_stat)
        _assert_error(authority.read_source_bytes, "unsafe")


@pytest.mark.parametrize(
    "unsafe_kind",
    ["owner", "mode", "special-mode", "hardlink", "symlink", "fifo", "directory"],
)
def test_source_file_safety_checks(tmp_path: Path, monkeypatch, unsafe_kind: str) -> None:
    _prepare_unsafe_owner_mode_link_and_file_types(
        tmp_path, monkeypatch, unsafe_kind
    )


def test_rejects_source_without_owner_read_permission(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "source:not-readable"
    source = _source(backend, key)
    source.write_bytes(b"unchanged")
    source.chmod(0o200)

    with open_checked_session(backend, key) as authority:
        _assert_error(authority.read_source_bytes, "unsafe")


def test_namespace_and_lock_identity_are_revalidated_between_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _backend(tmp_path)
    key = "source:lock-drift"
    source = _source(backend, key)
    source.write_bytes(b"a" * (_READ_CHUNK * 2))
    session_lock = backend.sessions_dir / ".session-files.lock"
    original_lock = session_lock.with_suffix(".held")
    real_read = checked_source.os.read
    drifted = False

    def read_then_replace_lock(descriptor: int, size: int) -> bytes:
        nonlocal drifted
        chunk = real_read(descriptor, size)
        if chunk and not drifted:
            drifted = True
            os.replace(session_lock, original_lock)
            session_lock.write_bytes(b"")
            session_lock.chmod(0o600)
        return chunk

    with open_checked_session(backend, key) as authority:
        monkeypatch.setattr(checked_source.os, "read", read_then_replace_lock)
        try:
            _assert_error(authority.read_source_bytes, "changed")
        finally:
            if original_lock.exists():
                session_lock.unlink(missing_ok=True)
                os.replace(original_lock, session_lock)

    assert drifted


def test_namespace_identity_is_revalidated_between_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _backend(tmp_path)
    key = "source:namespace-drift"
    source = _source(backend, key)
    source.write_bytes(b"a" * (_READ_CHUNK * 2))
    original_mode = stat.S_IMODE(backend.sessions_dir.stat().st_mode)
    real_read = checked_source.os.read
    drifted = False

    def read_then_chmod(descriptor: int, size: int) -> bytes:
        nonlocal drifted
        chunk = real_read(descriptor, size)
        if chunk and not drifted:
            drifted = True
            backend.sessions_dir.chmod(original_mode ^ 0o100)
        return chunk

    try:
        with open_checked_session(backend, key) as authority:
            monkeypatch.setattr(checked_source.os, "read", read_then_chmod)
            try:
                _assert_error(authority.read_source_bytes, "changed")
            finally:
                backend.sessions_dir.chmod(original_mode)
    finally:
        backend.sessions_dir.chmod(original_mode)

    assert drifted


def test_source_read_descriptor_is_closed_once_if_opened_fstat_fails(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _backend(tmp_path)
    key = "source:fd-cleanup"
    source = _source(backend, key)
    source.write_bytes(b"payload")
    real_open = checked_source.os.open
    real_fstat = checked_source.os.fstat
    real_close = checked_source.os.close
    source_fd: int | None = None
    source_close_count = 0

    def track_open(name, flags, mode=0o777, *, dir_fd=None):
        nonlocal source_fd
        descriptor = real_open(name, flags, mode, dir_fd=dir_fd)
        if name == source.name:
            source_fd = descriptor
        return descriptor

    def fail_source_fstat(descriptor: int):
        if descriptor == source_fd:
            raise OSError(errno.EIO, "private stat detail")
        return real_fstat(descriptor)

    def track_close(descriptor: int) -> None:
        nonlocal source_close_count
        if descriptor == source_fd:
            source_close_count += 1
        real_close(descriptor)

    with open_checked_session(backend, key) as authority:
        monkeypatch.setattr(checked_source.os, "open", track_open)
        monkeypatch.setattr(checked_source.os, "fstat", fail_source_fstat)
        monkeypatch.setattr(checked_source.os, "close", track_close)
        _assert_error(authority.read_source_bytes, "io")

    assert source_fd is not None
    assert source_close_count == 1


def test_read_io_error_is_content_free_and_closes_descriptor(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _backend(tmp_path)
    key = "source:read-io"
    source = _source(backend, key)
    source.write_bytes(b"payload")
    real_open = checked_source.os.open
    real_close = checked_source.os.close
    source_fd: int | None = None
    closed: list[int] = []

    def track_open(name, flags, mode=0o777, *, dir_fd=None):
        nonlocal source_fd
        descriptor = real_open(name, flags, mode, dir_fd=dir_fd)
        if name == source.name:
            source_fd = descriptor
        return descriptor

    def track_close(descriptor: int) -> None:
        if descriptor == source_fd:
            closed.append(descriptor)
        real_close(descriptor)

    def fail_read(_descriptor: int, _size: int) -> bytes:
        raise OSError(errno.EIO, "sensitive source failure")

    with open_checked_session(backend, key) as authority:
        monkeypatch.setattr(checked_source.os, "open", track_open)
        monkeypatch.setattr(checked_source.os, "close", track_close)
        monkeypatch.setattr(checked_source.os, "read", fail_read)
        error = _assert_error(authority.read_source_bytes, "io")

    assert source_fd is not None and closed == [source_fd]
    assert "sensitive" not in str(error)


def test_active_scope_exact_method_and_closed_lifecycle(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "source:scope"
    source = _source(backend, key)
    source.write_bytes(b"bound")
    authority = open_checked_session(backend, key)

    assert callable(authority.read_source_bytes)
    assert list(inspect.signature(authority.read_source_bytes).parameters) == []
    assert not hasattr(authority, "read_source")
    assert not hasattr(authority, "open_file")
    _assert_error(authority.read_source_bytes, "closed")
    authority.__enter__()
    assert authority.read_source_bytes() == b"bound"
    authority.close()
    _assert_error(authority.read_source_bytes, "closed")


def test_foreign_thread_cannot_read_active_source(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    key = "source:foreign-thread"
    _source(backend, key).write_bytes(b"bound")
    authority = _entered(backend, key)
    results: list[str] = []

    def read_foreign() -> None:
        try:
            authority.read_source_bytes()
        except CheckedSessionAuthorityError as error:
            assert error.__cause__ is None and error.__context__ is None
            results.append(error.code)

    thread = threading.Thread(target=read_foreign)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    assert results == ["invalid"]
    assert authority.read_source_bytes() == b"bound"
    authority.close()


def _forked_read_code(authority, result_queue) -> None:
    try:
        authority.read_source_bytes()
    except CheckedSessionAuthorityError as error:
        result_queue.put((error.code, error.__cause__ is None, error.__context__ is None))


def test_forked_process_cannot_read_or_release_parent_authority(tmp_path: Path) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork is unavailable")
    backend = _backend(tmp_path)
    key = "source:foreign-process"
    _source(backend, key).write_bytes(b"bound")
    authority = _entered(backend, key)
    context = multiprocessing.get_context("fork")
    result_queue = context.Queue()
    process = context.Process(target=_forked_read_code, args=(authority, result_queue))
    process.start()
    process.join(5)
    try:
        assert not process.is_alive()
        assert process.exitcode == 0
        assert result_queue.get(timeout=2) == ("invalid", True, True)
        assert authority.read_source_bytes() == b"bound"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        result_queue.close()
        authority.close()


def test_canonical_source_is_raw_and_read_does_not_call_ordinary_loader(
    tmp_path: Path, monkeypatch
) -> None:
    backend = _backend(tmp_path)
    key = "source:no-loader"
    source = _source(backend, key)
    payload = b'{"bad json but raw is allowed"\n'
    source.write_bytes(payload)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("ordinary session loader was called")

    monkeypatch.setattr(backend, "load_session", forbidden, raising=False)
    with open_checked_session(backend, key) as authority:
        assert authority.read_source_bytes() == payload
