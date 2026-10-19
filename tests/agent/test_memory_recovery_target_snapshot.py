"""Read-only snapshots of selected legacy memory targets."""

from __future__ import annotations

import copy
import errno
import multiprocessing
import os
import pickle
import stat
import threading
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent import memory_recovery_storage as storage


def test_r2_new_snapshot_operations_refuse_before_raw_fields_and_keep_unbound_seam(monkeypatch):
    raw = object.__new__(storage.MemoryTargetSnapshot)
    authority = object.__new__(storage.WorkspaceMemoryAuthority)
    for capability in (None, object()):
        with pytest.raises(storage.MemoryStorageError) as raised:
            raw.issuance_comparison(_capability=capability)
        assert raised.value.code == "invalid"
        with pytest.raises(storage.MemoryStorageError) as raised:
            raw.qualify_for_authority(authority, 1, False, _capability=capability)
        assert raised.value.code == "invalid"
        with pytest.raises(storage.MemoryStorageError) as raised:
            authority.require_issued_snapshot(raw, _capability=capability)
        assert raised.value.code == "invalid"
    with pytest.raises(AttributeError, match="_stat_identity"):
        raw.issuance_comparison(_capability=storage._SNAPSHOT_CAPABILITY)
    with pytest.raises(AttributeError, match="_authority"):
        raw.qualify_for_authority(authority, 1, False, _capability=storage._SNAPSHOT_CAPABILITY)

    class Snapshot(storage.MemoryTargetSnapshot):
        def issuance_comparison(self, **kwargs):
            raise AssertionError("new override must not replace original helper")

    subclass = Snapshot("MEMORY.md", None, None, authority, 1,
                        _capability=storage._SNAPSHOT_CAPABILITY)
    assert storage._target_snapshot_seal(subclass) == ("MEMORY.md", None, None, 1)
    with pytest.raises(storage.MemoryStorageError) as raised:
        authority._validate_snapshot(subclass, 1, False)
    assert raised.value.code == "invalid"
    calls = []
    monkeypatch.setattr(authority, "_require_issued_target_snapshot_locked", calls.append)
    assert authority.require_issued_snapshot(raw, _capability=storage._SNAPSHOT_CAPABILITY) is None
    assert calls == [raw]


def test_r2_original_seal_stat_first_and_exact_nine_fields():
    trace = []

    class Identity:
        def __getattribute__(self, name):
            trace.append(name)
            return {"device": 1, "inode": 2, "mode": 3, "uid": 4, "gid": 5,
                    "links": 6, "size": 7, "mtime_ns": 8, "ctime_ns": 9}[name]

    class Snapshot(storage.MemoryTargetSnapshot):
        def __getattribute__(self, name):
            trace.append(name)
            return super().__getattribute__(name)

    snapshot = object.__new__(Snapshot)
    for field, value in (("_name", "MEMORY.md"), ("_content", b"bytes"),
                         ("_stat_identity", Identity()), ("_epoch", 10)):
        object.__setattr__(snapshot, field, value)
    assert storage._target_snapshot_seal(snapshot) == (
        "MEMORY.md", b"bytes", (1, 2, 3, 4, 5, 6, 7, 8, 9), 10,
    )
    assert trace == ["_stat_identity", "device", "inode", "mode", "uid", "gid", "links",
                     "size", "mtime_ns", "ctime_ns", "_name", "_content", "_epoch"]
    with pytest.raises(AttributeError, match="_stat_identity"):
        storage._target_snapshot_seal(object.__new__(storage.MemoryTargetSnapshot))


@pytest.mark.parametrize("fault", [RuntimeError("seal"), KeyboardInterrupt("seal")])
def test_r2_original_dynamic_seal_registration_and_verification_order(fault, monkeypatch):
    authority = object.__new__(storage.WorkspaceMemoryAuthority)
    authority._pid = os.getpid()
    authority._lifecycle_lock = threading.RLock()
    authority._target_snapshot_issuance = {}
    snapshot = storage.MemoryTargetSnapshot("MEMORY.md", None, None, authority, 1,
                                            _capability=storage._SNAPSHOT_CAPABILITY)
    calls = []
    seal = ("patched", None, None, 99)

    def compute(actual):
        calls.append(actual)
        assert not authority._target_snapshot_issuance
        return seal

    monkeypatch.setattr(storage, "_target_snapshot_seal", compute)
    authority._register_target_snapshot_locked(snapshot)
    assert calls == [snapshot]
    assert authority._target_snapshot_issuance[id(snapshot)][1] is seal

    def failure(actual):
        calls.append(actual)
        assert authority._target_snapshot_issuance[id(actual)][0]() is actual
        raise fault

    monkeypatch.setattr(storage, "_target_snapshot_seal", failure)
    with pytest.raises(type(fault)) as raised:
        authority._require_issued_target_snapshot_locked(snapshot)
    assert raised.value is fault and calls == [snapshot, snapshot]
    clone = storage.MemoryTargetSnapshot("MEMORY.md", None, None, authority, 1,
                                        _capability=storage._SNAPSHOT_CAPABILITY)
    with pytest.raises(storage.MemoryStorageError) as raised:
        authority._require_issued_target_snapshot_locked(clone)
    assert raised.value.code == "invalid" and calls == [snapshot, snapshot]
    authority._target_snapshot_issuance.clear()

    def registration_failure(actual):
        assert actual is snapshot and not authority._target_snapshot_issuance
        raise fault

    monkeypatch.setattr(storage, "_target_snapshot_seal", registration_failure)
    with pytest.raises(type(fault)) as raised:
        authority._register_target_snapshot_locked(snapshot)
    # The registration failure must come from its live seal before assignment.
    assert raised.value is fault and not authority._target_snapshot_issuance


def test_r2_original_snapshot_shape_short_circuit_and_issuance_before_receipt_uid(monkeypatch):
    authority = object.__new__(storage.WorkspaceMemoryAuthority)
    calls = []
    fault = RuntimeError("issuance")

    def issuance(snapshot):
        calls.append("issuance")
        raise fault

    monkeypatch.setattr(authority, "_require_issued_target_snapshot_locked", issuance)
    monkeypatch.setattr(storage, "_uid", lambda: calls.append("uid") or 1)
    raw = object.__new__(storage.MemoryTargetSnapshot)
    object.__setattr__(raw, "_authority", authority)
    object.__setattr__(raw, "_epoch", 1)
    object.__setattr__(raw, "_name", object())
    with pytest.raises(storage.MemoryStorageError) as raised:
        authority._validate_snapshot(raw, 1, False)
    assert raised.value.code == "invalid" and not calls
    identity = storage.MemoryTargetStatIdentity(1, 2, stat.S_IFREG | 0o600, 1, 1, 1, 1, 1, 1)
    snapshot = storage.MemoryTargetSnapshot(".consolidation.pending.json", b"x", identity,
                                            authority, 1, _capability=storage._SNAPSHOT_CAPABILITY)
    with pytest.raises(RuntimeError) as raised:
        authority._validate_snapshot(snapshot, 1, True)
    assert raised.value is fault and calls == ["issuance"]

_TARGET_NAMES = (
    "MEMORY.md",
    "history.jsonl",
    "HISTORY.md",
    ".cursor",
    ".dream_cursor",
    ".consolidation.pending.json",
)


def _snapshot_capable() -> bool:
    if os.name != "posix":
        return False
    required_flags = ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")
    if any(not isinstance(getattr(os, name, None), int) for name in required_flags):
        return False
    required = {os.open, os.stat, os.mkdir, os.link}
    if not required <= getattr(os, "supports_dir_fd", set()):
        return False
    return os.link in getattr(os, "supports_follow_symlinks", set())


pytestmark = pytest.mark.skipif(
    not _snapshot_capable(), reason="target snapshots require POSIX no-follow dir-fd support"
)


def _snapshot_child(authority: storage.WorkspaceMemoryAuthority, result: Any) -> None:
    try:
        result.put(("ok", authority.snapshot_target("MEMORY.md").content))
    except storage.MemoryStorageError as error:
        result.put(("error", error.code, str(error)))


def test_snapshot_distinguishes_absent_and_empty_and_records_full_identity(
    tmp_path: Path,
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir(mode=0o755)
    memory.chmod(0o755)
    (memory / "MEMORY.md").write_bytes(b"")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        absent = authority.snapshot_target("history.jsonl")
        empty = authority.snapshot_target("MEMORY.md")

    assert absent.name == "history.jsonl"
    assert absent.content is None
    assert absent.exists is False
    assert absent.stat_identity is None
    assert empty.name == "MEMORY.md"
    assert empty.content == b""
    assert empty.exists is True
    identity = empty.stat_identity
    assert identity is not None
    observed = os.stat(memory / "MEMORY.md")
    assert (
        identity.device,
        identity.inode,
        identity.mode,
        identity.uid,
        identity.gid,
        identity.links,
        identity.size,
        identity.mtime_ns,
        identity.ctime_ns,
    ) == (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_uid,
        observed.st_gid,
        observed.st_nlink,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )
    assert stat.S_IMODE(memory.stat().st_mode) == 0o755


def test_snapshot_accepts_only_exact_allowlist_and_preserves_target_bytes(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    for index, name in enumerate(_TARGET_NAMES):
        target = memory / name
        target.write_bytes(f"target-{index}".encode())
        if name == ".consolidation.pending.json":
            target.chmod(0o600)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshots = [authority.snapshot_target(name) for name in _TARGET_NAMES]
        for invalid in (
            "memory.md",
            "../MEMORY.md",
            "MEMORY.md/child",
            ".memory-recovery.lock",
            "",
        ):
            with pytest.raises(storage.MemoryStorageError) as failure:
                authority.snapshot_target(invalid)
            assert failure.value.code == "invalid"

    assert [snapshot.name for snapshot in snapshots] == list(_TARGET_NAMES)
    assert [snapshot.content for snapshot in snapshots] == [
        f"target-{index}".encode() for index in range(len(_TARGET_NAMES))
    ]
    assert [(memory / name).read_bytes() for name in _TARGET_NAMES] == [
        f"target-{index}".encode() for index in range(len(_TARGET_NAMES))
    ]


def test_snapshot_target_is_immutable_and_cannot_be_copied_or_serialized(
    tmp_path: Path,
) -> None:
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "MEMORY.md").write_bytes(b"memory")

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        with pytest.raises((AttributeError, TypeError)):
            snapshot.content = b"changed"  # type: ignore[misc]
        with pytest.raises(TypeError):
            copy.copy(snapshot)
        with pytest.raises(TypeError):
            copy.deepcopy(snapshot)
        with pytest.raises(TypeError):
            pickle.dumps(snapshot)
        with pytest.raises(storage.MemoryStorageError) as forge:
            storage.MemoryTargetSnapshot("MEMORY.md", b"forged", None, authority, 1)
        assert forge.value.code == "invalid"
        assert repr(snapshot) == "<MemoryTargetSnapshot>"


def test_ordinary_target_allows_readable_0644_under_0755_memory(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    memory.chmod(0o755)
    target = memory / "MEMORY.md"
    target.write_bytes(b"legacy bytes")
    target.chmod(0o644)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")

    assert snapshot.content == b"legacy bytes"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert stat.S_IMODE(memory.stat().st_mode) == 0o755


@pytest.mark.parametrize("mode", [0o644, 0o400, 0o620, 0o20600, 0o4600])
def test_receipt_target_requires_exact_private_0600_mode(tmp_path: Path, mode: int) -> None:
    target = tmp_path / "memory" / ".consolidation.pending.json"
    target.parent.mkdir()
    target.write_bytes(b"receipt")
    target.chmod(mode)
    if stat.S_IMODE(target.stat().st_mode) != mode:
        pytest.skip("filesystem does not preserve special permission bits on regular files")
    before = target.stat()

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.snapshot_target(".consolidation.pending.json")

    assert failure.value.code == "corrupt"
    after = target.stat()
    assert (after.st_ino, after.st_mode, after.st_size) == (
        before.st_ino,
        before.st_mode,
        before.st_size,
    )
    assert target.read_bytes() == b"receipt"


def test_receipt_target_accepts_exact_private_0600_mode(tmp_path: Path) -> None:
    target = tmp_path / "memory" / ".consolidation.pending.json"
    target.parent.mkdir()
    target.write_bytes(b"receipt")
    target.chmod(0o600)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target(".consolidation.pending.json")

    assert snapshot.content == b"receipt"
    assert snapshot.stat_identity is not None
    assert stat.S_IMODE(snapshot.stat_identity.mode) == 0o600


@pytest.mark.parametrize(
    "target_kind", ["symlink", "fifo", "hardlink", "unsafe-mode", "unreadable"]
)
def test_snapshot_rejects_unsafe_target_entries_without_mutation(
    tmp_path: Path, target_kind: str
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    target = memory / "MEMORY.md"
    original = memory / "original"
    original.write_bytes(b"target bytes")
    if target_kind == "symlink":
        target.symlink_to(original)
    elif target_kind == "fifo":
        os.mkfifo(target)
    elif target_kind == "hardlink":
        os.link(original, target)
    else:
        target.write_bytes(b"target bytes")
        target.chmod(0o200 if target_kind == "unreadable" else 0o602)

    before = os.lstat(target)
    original_bytes = original.read_bytes()
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError):
            authority.snapshot_target("MEMORY.md")

    after = os.lstat(target)
    assert (after.st_dev, after.st_ino, after.st_mode, after.st_nlink) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
    )
    assert original.read_bytes() == original_bytes


def test_snapshot_rejects_targets_over_documented_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "memory" / "MEMORY.md"
    target.parent.mkdir()
    target.write_bytes(b"12345")
    monkeypatch.setattr(storage, "_MEMORY_TARGET_MAX_BYTES", 4)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.snapshot_target("MEMORY.md")

    assert failure.value.code == "corrupt"
    assert target.read_bytes() == b"12345"


def test_snapshot_accepts_target_at_documented_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "memory" / "MEMORY.md"
    target.parent.mkdir()
    target.write_bytes(b"1234")
    monkeypatch.setattr(storage, "_MEMORY_TARGET_MAX_BYTES", 4)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")

    assert snapshot.content == b"1234"
    assert target.read_bytes() == b"1234"


@pytest.mark.parametrize("error_number", [errno.EIO, errno.EMFILE])
def test_leaf_open_io_errors_are_reported_as_io_without_mutating_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_number: int
) -> None:
    target = tmp_path / "memory" / "MEMORY.md"
    target.parent.mkdir()
    target.write_bytes(b"memory")
    before = os.lstat(target)
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    real_open = storage.os.open

    def fail_target_open(name: str, flags: int, *args: Any, **kwargs: Any) -> int:
        if name == "MEMORY.md" and kwargs.get("dir_fd") == authority._memory_fd:
            raise OSError(error_number, "private open failure")
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", fail_target_open)
    try:
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.snapshot_target("MEMORY.md")
    finally:
        monkeypatch.undo()
        authority.close()

    assert failure.value.code == "io"
    assert str(failure.value) == "memory recovery storage operation failed"
    after = os.lstat(target)
    assert (after.st_dev, after.st_ino, after.st_mode, after.st_nlink) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
    )
    assert target.read_bytes() == b"memory"


@pytest.mark.parametrize(
    ("race", "expected_code"),
    [("disappearance", "binding"), ("symlink-swap", "corrupt")],
)
def test_stat_to_open_races_are_classified_without_changing_target_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race: str,
    expected_code: str,
) -> None:
    target = tmp_path / "memory" / "MEMORY.md"
    target.parent.mkdir()
    target.write_bytes(b"memory")
    held = tmp_path / "held-target"
    external = tmp_path / "external-target"
    external.write_bytes(b"external")
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    real_open = storage.os.open

    def race_target_open(name: str, flags: int, *args: Any, **kwargs: Any) -> int:
        if name == "MEMORY.md" and kwargs.get("dir_fd") == authority._memory_fd:
            target.rename(held)
            if race == "symlink-swap":
                target.symlink_to(external)
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", race_target_open)
    try:
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.snapshot_target("MEMORY.md")
        assert failure.value.code == expected_code
        if race == "disappearance":
            assert not target.exists()
            assert held.read_bytes() == b"memory"
        else:
            assert target.is_symlink()
            assert target.resolve() == external
    finally:
        monkeypatch.undo()
        if target.is_symlink():
            target.unlink()
        if held.exists():
            held.rename(target)
        authority.close()

    assert target.read_bytes() == b"memory"
    assert external.read_bytes() == b"external"


def test_same_bytes_replacement_during_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "memory" / "MEMORY.md"
    target.parent.mkdir()
    target.write_bytes(b"same bytes")
    real_read = storage.os.read
    replaced = False

    def replace_then_read(fd: int, size: int) -> bytes:
        nonlocal replaced
        if not replaced:
            replaced = True
            replacement = target.with_name("replacement")
            replacement.write_bytes(b"same bytes")
            replacement.chmod(target.stat().st_mode & 0o777)
            os.replace(replacement, target)
        return real_read(fd, size)

    monkeypatch.setattr(storage.os, "read", replace_then_read)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.snapshot_target("MEMORY.md")

    assert failure.value.code == "binding"
    assert target.read_bytes() == b"same bytes"


def test_same_bytes_in_place_rewrite_during_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "memory" / "MEMORY.md"
    target.parent.mkdir()
    target.write_bytes(b"same bytes")
    before = target.stat()
    real_read = storage.os.read
    rewritten = False

    def rewrite_then_read(fd: int, size: int) -> bytes:
        nonlocal rewritten
        if not rewritten:
            rewritten = True
            rewrite_fd = os.open(target, os.O_WRONLY)
            try:
                assert os.pwrite(rewrite_fd, b"same bytes", 0) == len(b"same bytes")
            finally:
                os.close(rewrite_fd)
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        return real_read(fd, size)

    monkeypatch.setattr(storage.os, "read", rewrite_then_read)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.snapshot_target("MEMORY.md")

    assert failure.value.code == "binding"
    assert target.read_bytes() == b"same bytes"


def test_mode_change_during_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "memory" / "MEMORY.md"
    target.parent.mkdir()
    target.write_bytes(b"memory")
    target.chmod(0o600)
    real_read = storage.os.read
    changed = False

    def chmod_then_read(fd: int, size: int) -> bytes:
        nonlocal changed
        if not changed:
            changed = True
            target.chmod(0o644)
        return real_read(fd, size)

    monkeypatch.setattr(storage.os, "read", chmod_then_read)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.snapshot_target("MEMORY.md")

    assert failure.value.code == "binding"
    assert target.read_bytes() == b"memory"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_snapshot_wrong_thread_fork_and_closed_authority_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "memory" / "MEMORY.md"
    target.parent.mkdir()
    target.write_bytes(b"memory")
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    other_thread: list[str] = []

    def snapshot_from_other_thread() -> None:
        try:
            authority.snapshot_target("MEMORY.md")
        except storage.MemoryStorageError as error:
            other_thread.append(error.code)

    worker = threading.Thread(target=snapshot_from_other_thread)
    worker.start()
    worker.join(timeout=3)
    try:
        assert not worker.is_alive()
        assert other_thread == ["closed"]

        if "fork" in multiprocessing.get_all_start_methods():
            context = multiprocessing.get_context("fork")
            result = context.Queue()
            child = context.Process(target=_snapshot_child, args=(authority, result))
            child.start()
            child.join(timeout=5)
            if child.exitcode is None:
                child.terminate()
                child.join(timeout=2)
            assert child.exitcode == 0
            assert result.get(timeout=1) == (
                "error",
                "closed",
                "memory recovery authority is closed",
            )

        authority.close()
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.snapshot_target("MEMORY.md")
        assert failure.value.code == "closed"
    finally:
        authority.close()


def test_snapshot_read_error_closes_target_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "memory" / "MEMORY.md"
    target.parent.mkdir()
    target.write_bytes(b"memory")
    opened_target_fds: list[int] = []
    real_open = storage.os.open
    real_read = storage.os.read

    def track_open(name: str, flags: int, *args: Any, **kwargs: Any) -> int:
        fd = real_open(name, flags, *args, **kwargs)
        if name == "MEMORY.md" and kwargs.get("dir_fd") is not None:
            opened_target_fds.append(fd)
        return fd

    def fail_read(fd: int, size: int) -> bytes:
        if fd in opened_target_fds:
            raise OSError(errno.EIO, "private failure")
        return real_read(fd, size)

    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    monkeypatch.setattr(storage.os, "open", track_open)
    monkeypatch.setattr(storage.os, "read", fail_read)
    try:
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.snapshot_target("MEMORY.md")
    finally:
        monkeypatch.undo()
        authority.close()

    assert failure.value.code == "io"
    assert len(opened_target_fds) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(opened_target_fds[0])
    assert closed.value.errno == errno.EBADF


def test_lifecycle_lock_prevents_concurrent_close_during_snapshot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "memory" / "MEMORY.md"
    target.parent.mkdir()
    target.write_bytes(b"memory")
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    close_attempted = threading.Event()
    close_completed = threading.Event()
    real_read = storage.os.read
    real_close = authority.close
    close_errors: list[BaseException] = []

    def close_from_worker() -> None:
        close_attempted.set()
        try:
            real_close()
        except BaseException as error:
            close_errors.append(error)
        finally:
            close_completed.set()

    closer: threading.Thread | None = None

    def start_close_then_read(fd: int, size: int) -> bytes:
        nonlocal closer
        if closer is None:
            closer = threading.Thread(target=close_from_worker)
            closer.start()
            assert close_attempted.wait(timeout=3)
            # The close thread has had a scheduling interval to attempt its
            # lifecycle acquisition. It must remain blocked until this read is
            # complete, otherwise it could detach the authority mid-snapshot.
            assert not close_completed.wait(timeout=0.1)
        return real_read(fd, size)

    monkeypatch.setattr(storage.os, "read", start_close_then_read)
    try:
        snapshot = authority.snapshot_target("MEMORY.md")
        if closer is not None:
            closer.join(timeout=3)
        assert snapshot.content == b"memory"
        assert close_completed.is_set()
        assert not close_errors
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.snapshot_target("MEMORY.md")
        assert failure.value.code == "closed"
    finally:
        monkeypatch.undo()
        authority.close()
        if closer is not None:
            closer.join(timeout=3)


def test_snapshot_rechecks_memory_directory_binding_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "MEMORY.md").write_bytes(b"memory")
    external = tmp_path / "external"
    external.mkdir()
    (external / "MEMORY.md").write_bytes(b"redirected")
    moved_memory = tmp_path / "memory-original"
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    real_read = storage.os.read
    swapped = False

    def replace_parent_then_read(fd: int, size: int) -> bytes:
        nonlocal swapped
        if not swapped:
            swapped = True
            memory.rename(moved_memory)
            memory.symlink_to(external, target_is_directory=True)
        return real_read(fd, size)

    monkeypatch.setattr(storage.os, "read", replace_parent_then_read)
    try:
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.snapshot_target("MEMORY.md")
        assert failure.value.code == "binding"
        assert (external / "MEMORY.md").read_bytes() == b"redirected"
    finally:
        monkeypatch.undo()
        if memory.is_symlink():
            memory.unlink()
        if moved_memory.exists():
            moved_memory.rename(memory)
        authority.close()


def test_snapshot_release_in_inherited_process_never_acquires_lifecycle_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        reference = authority._target_snapshot_issuance[id(snapshot)][0]
        callback = reference.__callback__
        assert callback is not None
        original_lock = authority._lifecycle_lock

        class InheritedLock:
            def __enter__(self) -> None:
                raise AssertionError("inherited lifecycle mutex acquired")

            def __exit__(self, *args: object) -> None:
                pass

        with monkeypatch.context() as inherited:
            inherited.setattr(authority, "_lifecycle_lock", InheritedLock())
            inherited.setattr(storage.os, "getpid", lambda: authority._pid + 1)
            callback(reference)
        assert authority._lifecycle_lock is original_lock
        assert id(snapshot) in authority._target_snapshot_issuance
