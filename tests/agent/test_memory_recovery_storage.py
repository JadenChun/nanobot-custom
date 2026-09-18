"""Focused B1a tests for descriptor-bound memory recovery storage."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent import memory_recovery_storage as storage


def _storage_capable() -> bool:
    if os.name != "posix":
        return False
    if any(not isinstance(getattr(os, name, None), int) for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")):
        return False
    if not all(function in getattr(os, "supports_dir_fd", set()) for function in (os.open, os.stat, os.mkdir, os.link)):
        return False
    if os.link not in getattr(os, "supports_follow_symlinks", set()):
        return False
    return hasattr(storage, "_fcntl") and storage._fcntl is not None


def _fork_capable() -> bool:
    if not _storage_capable():
        return False
    try:
        multiprocessing.get_context("fork")
    except (ValueError, AttributeError):
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _storage_capable(), reason="B1a storage requires POSIX no-follow dir-fd/fsync/flock capabilities"
)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _scope(workspace: Path) -> Path:
    return workspace / "memory" / ".recovery" / "blobs" / "sha256"


def _publish_child(
    workspace: str, content: bytes, barrier: Any, result: Any,
) -> None:
    try:
        barrier.wait(timeout=5)
        with storage.WorkspaceMemoryAuthority(Path(workspace)) as authority:
            result.put(("ok", authority.publish_blob(content)))
    except BaseException as exc:  # child reports only the public fixed text
        result.put(("error", type(exc).__name__, str(exc)))


def _publish_record_child(
    workspace: str,
    kind: str,
    identity: str,
    slot: str,
    content: bytes,
    barrier: Any,
    result: Any,
) -> None:
    try:
        barrier.wait(timeout=5)
        with storage.WorkspaceMemoryAuthority(Path(workspace)) as authority:
            namespace = authority.open_evidence_namespace(kind, identity)
            result.put(("ok", namespace.publish_record(slot, content)))
    except BaseException as exc:  # child reports only the public fixed text
        result.put(("error", type(exc).__name__, str(exc)))


def test_authority_creates_private_layout_and_reload_is_idempotent(tmp_path: Path) -> None:
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        assert authority.workspace == tmp_path.resolve()
        assert authority.blob_directory == _scope(tmp_path)
        assert authority.lock_path == tmp_path.resolve() / ".memory-recovery.lock"

    assert stat.S_IMODE((tmp_path / ".memory-recovery.lock").stat().st_mode) == 0o600
    for path in (
        tmp_path / "memory",
        tmp_path / "memory" / ".recovery",
        tmp_path / "memory" / ".recovery" / "blobs",
        _scope(tmp_path),
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    lock_inode = (tmp_path / ".memory-recovery.lock").stat().st_ino

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        assert authority.lock_path.stat().st_ino == lock_inode


def test_evidence_namespace_publishes_named_records(tmp_path: Path) -> None:
    identity = "a" * 64
    content = b"named evidence"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        assert namespace.kind == "transactions"
        assert namespace.identity == identity
        assert namespace.publish_record("genesis.json", content) == "genesis.json"
        assert namespace.read_record("genesis.json") == content
        namespace.close()

    record = tmp_path / "memory" / ".recovery" / "transactions" / identity / "genesis.json"
    companion = record.with_name(".install-genesis.json")
    assert record.stat().st_ino == companion.stat().st_ino
    assert stat.S_IMODE(record.stat().st_mode) == 0o600


@pytest.mark.parametrize("kind", ["transactions", "conversions"])
def test_evidence_namespace_layout_reload_and_missing_read(tmp_path: Path, kind: str) -> None:
    identity = "b" * 64
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace(kind, identity)
        assert namespace.directory == tmp_path / "memory" / ".recovery" / kind / identity
        assert stat.S_IMODE(namespace.directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(namespace.directory.parent.stat().st_mode) == 0o700
        assert namespace.read_record("genesis.json") is None
        namespace.close()

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with authority.open_evidence_namespace(kind, identity) as namespace:
            assert namespace.read_record("genesis.json") is None


@pytest.mark.parametrize("fail_at", range(4))
def test_each_evidence_namespace_acquisition_sync_failure_is_repaired_on_retry(
    tmp_path: Path, fail_at: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = "0" * 64
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        real_fsync = storage.os.fsync
        calls: list[tuple[int, int]] = []
        armed = True

        def fsync(fd: int) -> None:
            nonlocal armed
            info = os.fstat(fd)
            calls.append((info.st_dev, info.st_ino))
            if armed and len(calls) - 1 == fail_at:
                armed = False
                raise OSError("namespace acquisition sync fault")
            real_fsync(fd)

        monkeypatch.setattr(storage.os, "fsync", fsync)
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.open_evidence_namespace("transactions", identity)
        assert failure.value.code == "durability"
        calls.clear()
        with authority.open_evidence_namespace("transactions", identity) as namespace:
            assert namespace.directory.is_dir()
        assert len(calls) >= 4


@pytest.mark.parametrize("kind", ["", "Transactions", "transactions/extra", "transactions"])
def test_evidence_namespace_rejects_invalid_kind(tmp_path: Path, kind: str) -> None:
    identity = "c" * 64
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        if kind == "transactions":
            namespace = authority.open_evidence_namespace(kind, identity)
            assert namespace.kind == kind
            namespace.close()
        else:
            with pytest.raises(storage.MemoryStorageError) as failure:
                authority.open_evidence_namespace(kind, identity)
            assert failure.value.code == "invalid"


@pytest.mark.parametrize("identity", ["", "A" * 64, "d" * 63, "e" * 65, True])
def test_evidence_namespace_rejects_invalid_identity(tmp_path: Path, identity: Any) -> None:
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.open_evidence_namespace("transactions", identity)
        assert failure.value.code == "invalid"


@pytest.mark.parametrize(
    "slot",
    ["", "genesis", "genesis.json/extra", "next-" + "F" * 64 + ".json", "next-" + "a" * 63 + ".json", True],
)
def test_evidence_namespace_rejects_invalid_slots(tmp_path: Path, slot: Any) -> None:
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", "f" * 64)
        with pytest.raises(storage.MemoryStorageError) as failure:
            namespace.publish_record(slot, b"record")
        assert failure.value.code == "invalid"
        with pytest.raises(storage.MemoryStorageError) as failure:
            namespace.read_record(slot)
        assert failure.value.code == "invalid"


def test_evidence_namespace_successor_publish_retry_and_clean_missing(tmp_path: Path) -> None:
    identity = "1" * 64
    slot = "next-" + "2" * 64 + ".json"
    content = b"successor record\x00"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("conversions", identity)
        assert namespace.publish_record(slot, content) == slot
        assert namespace.publish_record(slot, content) == slot
        assert namespace.read_record(slot) == content
        assert namespace.read_record("genesis.json") is None

    record = tmp_path / "memory/.recovery/conversions" / identity / slot
    companion = record.with_name(".install-" + slot)
    assert record.stat().st_ino == companion.stat().st_ino
    assert record.stat().st_nlink == companion.stat().st_nlink == 2


def test_evidence_namespace_rejects_conflicting_record(tmp_path: Path) -> None:
    identity = "3" * 64
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        namespace.publish_record("genesis.json", b"first")
        with pytest.raises(storage.MemoryStorageError) as failure:
            namespace.publish_record("genesis.json", b"different")
        assert failure.value.code == "conflict"
        assert namespace.read_record("genesis.json") == b"first"


@pytest.mark.parametrize("content", [bytearray(b"record"), memoryview(b"record"), True, object()])
def test_evidence_namespace_rejects_non_bytes_content(tmp_path: Path, content: Any) -> None:
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", "2" * 64)
        with pytest.raises(storage.MemoryStorageError) as failure:
            namespace.publish_record("genesis.json", content)
        assert failure.value.code == "invalid"


def test_evidence_namespace_operations_after_close_fail_without_leaking_descriptors(
    tmp_path: Path,
) -> None:
    authority = storage.WorkspaceMemoryAuthority(tmp_path)
    authority.__enter__()
    namespace = authority.open_evidence_namespace("transactions", "3" * 64)
    namespace.close()
    with pytest.raises(storage.MemoryStorageError) as failure:
        namespace.read_record("genesis.json")
    assert failure.value.code == "closed"
    authority.close()


def test_two_process_record_publishers_are_bounded_and_immutable(tmp_path: Path) -> None:
    if not _fork_capable():
        pytest.skip("concurrent publisher test requires multiprocessing fork")
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    result = context.Queue()
    identity = "4" * 64
    slot = "genesis.json"
    processes = [
        context.Process(
            target=_publish_record_child,
            args=(str(tmp_path), "transactions", identity, slot, payload, barrier, result),
        )
        for payload in (b"same", b"same")
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=5)
        assert all(not process.is_alive() for process in processes)
        results = [result.get(timeout=2) for _ in processes]
        assert all(item == ("ok", slot) for item in results)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        assert namespace.read_record(slot) == b"same"


def test_two_process_record_publishers_conflict_without_hanging(tmp_path: Path) -> None:
    if not _fork_capable():
        pytest.skip("concurrent publisher test requires multiprocessing fork")
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    result = context.Queue()
    identity = "5" * 64
    slot = "genesis.json"
    processes = [
        context.Process(
            target=_publish_record_child,
            args=(str(tmp_path), "conversions", identity, slot, payload, barrier, result),
        )
        for payload in (b"first", b"second")
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=5)
        assert all(not process.is_alive() for process in processes)
        results = [result.get(timeout=2) for _ in processes]
        assert sorted(item[0] for item in results) == ["error", "ok"]
        assert all(item[2] == "memory recovery storage content conflicts" for item in results if item[0] == "error")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)


@pytest.mark.parametrize("kind", ["regular", "symlink", "fifo", "directory", "hardlink"])
def test_record_destination_collision_never_changes_existing_entry(
    tmp_path: Path, kind: str
) -> None:
    if kind == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("FIFO collision requires os.mkfifo")
    identity = "6" * 64
    content = b"record collision"
    other = b"different record"
    slot = "genesis.json"
    outside = tmp_path / "outside"
    outside.write_bytes(other)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        target = namespace.directory / slot
        if kind == "regular":
            target.write_bytes(other)
            target.chmod(0o600)
        elif kind == "symlink":
            target.symlink_to(outside)
        elif kind == "fifo":
            os.mkfifo(target, 0o600)
        elif kind == "directory":
            target.mkdir(mode=0o700)
        else:
            outside.write_bytes(content)
            os.link(outside, target)
            target.chmod(0o600)
        before = target.lstat()
        before_bytes = target.read_bytes() if kind in {"regular", "hardlink"} else None
        with pytest.raises(storage.MemoryStorageError):
            namespace.publish_record(slot, content)
        after = target.lstat()
        assert (after.st_dev, after.st_ino, after.st_mode, after.st_nlink) == (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
        )
        if before_bytes is not None:
            assert target.read_bytes() == before_bytes
        assert not (target.parent / (".install-" + slot)).exists()


def test_record_companion_collision_is_rejected_without_replacement(tmp_path: Path) -> None:
    identity = "7" * 64
    slot = "next-" + "8" * 64 + ".json"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("conversions", identity)
        companion = namespace.directory / (".install-" + slot)
        companion.write_bytes(b"wrong")
        companion.chmod(0o600)
        before = companion.read_bytes()
        with pytest.raises(storage.MemoryStorageError) as failure:
            namespace.publish_record(slot, b"expected")
        assert failure.value.code == "conflict"
        assert companion.read_bytes() == before
        assert not (namespace.directory / slot).exists()


@pytest.mark.parametrize("collision", ["regular", "exact", "symlink", "fifo", "directory"])
def test_record_link_collision_is_arbitrated_without_overwrite(
    tmp_path: Path, collision: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if collision == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("FIFO collision requires os.mkfifo")
    identity = "8" * 64
    slot = "genesis.json"
    content = b"link race content"
    other = b"link race other"
    outside = tmp_path / "outside"
    outside.write_bytes(other)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        real_link = storage.os.link
        injected = False

        def link(*args: Any, **kwargs: Any) -> Any:
            nonlocal injected
            target = namespace.directory / slot
            if not injected:
                injected = True
                if collision in {"regular", "exact"}:
                    target.write_bytes(content if collision == "exact" else other)
                    target.chmod(0o600)
                elif collision == "symlink":
                    target.symlink_to(outside)
                elif collision == "fifo":
                    os.mkfifo(target, 0o600)
                else:
                    target.mkdir(mode=0o700)
            return real_link(*args, **kwargs)

        monkeypatch.setattr(storage.os, "link", link)
        if collision == "exact":
            assert namespace.publish_record(slot, content) == slot
            assert namespace.read_record(slot) == content
        else:
            with pytest.raises(storage.MemoryStorageError) as failure:
                namespace.publish_record(slot, content)
            assert failure.value.code in {"corrupt", "conflict"}
        assert injected


def test_record_companion_only_install_is_finished_without_cleanup(tmp_path: Path) -> None:
    identity = "9" * 64
    slot = "genesis.json"
    content = b"companion recovery"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        namespace.publish_record(slot, content)
    record = tmp_path / "memory/.recovery/transactions" / identity / slot
    companion = record.with_name(".install-" + slot)
    record.unlink()
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        with pytest.raises(storage.MemoryStorageError) as failure:
            namespace.read_record(slot)
        assert failure.value.code == "missing"
        assert namespace.publish_record(slot, content) == slot
    assert record.read_bytes() == companion.read_bytes() == content
    assert record.stat().st_ino == companion.stat().st_ino


def test_record_directory_fsync_failure_is_repaired_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = "a" * 64
    slot = "genesis.json"
    content = b"directory retry"
    real_fsync = storage.os.fsync
    real_link = storage.os.link
    linked = False
    failed = False
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        record_fd = namespace._record_fd
        assert record_fd is not None

        def link(*args: Any, **kwargs: Any) -> Any:
            nonlocal linked
            result = real_link(*args, **kwargs)
            linked = True
            return result

        def fsync(fd: int) -> None:
            nonlocal failed
            if linked and fd == record_fd and not failed:
                failed = True
                raise OSError("injected record directory fsync failure")
            real_fsync(fd)

        monkeypatch.setattr(storage.os, "link", link)
        monkeypatch.setattr(storage.os, "fsync", fsync)
        with pytest.raises(storage.MemoryStorageError) as failure:
            namespace.publish_record(slot, content)
        assert failure.value.code == "durability"
        monkeypatch.setattr(storage.os, "link", real_link)
        monkeypatch.setattr(storage.os, "fsync", real_fsync)
        assert namespace.publish_record(slot, content) == slot


def test_copied_namespace_records_are_verified_by_content(tmp_path: Path) -> None:
    identity = "b" * 64
    slot = "next-" + "c" * 64 + ".json"
    content = b"copied namespace record"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("conversions", identity)
        namespace.publish_record(slot, content)
    copied = tmp_path.parent / (tmp_path.name + "-namespace-copy")
    shutil.copytree(tmp_path, copied, copy_function=shutil.copy2)
    try:
        record = copied / "memory/.recovery/conversions" / identity / slot
        companion = record.with_name(".install-" + slot)
        assert record.stat().st_nlink == companion.stat().st_nlink == 1
        with storage.WorkspaceMemoryAuthority(copied) as authority:
            namespace = authority.open_evidence_namespace("conversions", identity)
            assert namespace.read_record(slot) == content
    finally:
        shutil.rmtree(copied)


def test_copied_namespace_extra_hardlink_is_rejected(tmp_path: Path) -> None:
    identity = "d" * 64
    slot = "genesis.json"
    content = b"extra link"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        namespace.publish_record(slot, content)
    copied = tmp_path.parent / (tmp_path.name + "-namespace-links")
    shutil.copytree(tmp_path, copied, copy_function=shutil.copy2)
    try:
        record = copied / "memory/.recovery/transactions" / identity / slot
        os.link(record, record.with_name(".extra-link"))
        with storage.WorkspaceMemoryAuthority(copied) as authority:
            namespace = authority.open_evidence_namespace("transactions", identity)
            with pytest.raises(storage.MemoryStorageError) as failure:
                namespace.read_record(slot)
            assert failure.value.code == "corrupt"
    finally:
        shutil.rmtree(copied)


def test_named_read_rejects_independent_target_companion_content_mismatch(tmp_path: Path) -> None:
    identity = "e" * 64
    slot = "genesis.json"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        namespace.publish_record(slot, b"good")
        target = namespace.directory / slot
        companion = namespace.directory / (".install-" + slot)
        target.unlink()
        companion.unlink()
        target.write_bytes(b"good")
        target.chmod(0o600)
        companion.write_bytes(b"evil")
        companion.chmod(0o600)
        with pytest.raises(storage.MemoryStorageError) as failure:
            namespace.read_record(slot)
        assert failure.value.code == "corrupt"


def test_named_read_rechecks_first_state_through_directory_durability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = "f" * 64
    slot = "genesis.json"
    content = b"stable named record"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        namespace.publish_record(slot, content)
        target = namespace.directory / slot
        companion = namespace.directory / (".install-" + slot)
        record_fd = namespace._record_fd
        assert record_fd is not None
        real_fsync = storage.os.fsync
        replaced = False

        def fsync(fd: int) -> None:
            nonlocal replaced
            real_fsync(fd)
            if fd != record_fd or replaced:
                return
            replaced = True
            target.unlink()
            companion.unlink()
            target.write_bytes(content)
            target.chmod(0o600)
            companion.write_bytes(content)
            companion.chmod(0o600)

        monkeypatch.setattr(storage.os, "fsync", fsync)
        with pytest.raises(storage.MemoryStorageError) as failure:
            namespace.read_record(slot)
        assert replaced
        assert failure.value.code == "binding"


def test_clean_absence_rejects_companion_only_to_symlink_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = "0" * 64
    slot = "genesis.json"
    content = b"companion-only"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        companion = namespace.directory / (".install-" + slot)
        companion.write_bytes(content)
        companion.chmod(0o600)
        target = namespace.directory / slot
        real_state = storage._ImmutableSlotStore.state
        calls = 0

        def state(store: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            result = real_state(store, *args, **kwargs)
            calls += 1
            if calls == 1:
                target.symlink_to(tmp_path / "outside")
                companion.unlink()
            return result

        monkeypatch.setattr(storage._ImmutableSlotStore, "state", state)
        with pytest.raises(storage.MemoryStorageError) as failure:
            namespace.read_record(slot)
        assert failure.value.code in {"binding", "corrupt"}


def test_namespace_constructor_is_factory_only_and_authority_closes_live_children(
    tmp_path: Path,
) -> None:
    identity = "1" * 64
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError) as failure:
            storage.ImmutableEvidenceNamespace(authority, "transactions", identity)
        assert failure.value.code == "invalid"
        assert not (tmp_path / "memory/.recovery/transactions").exists()
        namespace = authority.open_evidence_namespace("transactions", identity)
        record_fd = namespace._record_fd
        kind_fd = namespace._kind_fd
        assert record_fd is not None and kind_fd is not None
        authority.close()
        assert namespace._record_fd is None
        assert namespace._kind_fd is None
        with pytest.raises(storage.MemoryStorageError) as failure:
            namespace.read_record("genesis.json")
        assert failure.value.code == "closed"


def test_failed_namespace_construction_closes_all_child_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not Path("/dev/fd").is_dir():
        pytest.skip("descriptor lifecycle probe requires /dev/fd")
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        before = {int(name) for name in os.listdir("/dev/fd") if name.isdigit()}
        real_sync = storage._sync_directory_boundary
        armed = True

        def sync(parent_fd: int, name: str, child_fd: int) -> None:
            nonlocal armed
            if armed and name == "transactions":
                armed = False
                raise storage.MemoryStorageDurabilityError()
            real_sync(parent_fd, name, child_fd)

        monkeypatch.setattr(storage, "_sync_directory_boundary", sync)
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.open_evidence_namespace("transactions", "2" * 64)
        assert failure.value.code == "durability"
        after = {int(name) for name in os.listdir("/dev/fd") if name.isdigit()}
        assert after == before


@pytest.mark.parametrize("mode", ["blob", "named"])
def test_read_rejects_exact_replacement_after_initial_state(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"exact replacement bytes"
    identity = "3" * 64
    slot = "genesis.json"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        if mode == "blob":
            target_directory = authority.blob_directory
            digest = _digest(content)
            authority.publish_blob(content)

            def read() -> bytes:
                return authority.read_blob(digest)

        else:
            namespace = authority.open_evidence_namespace("transactions", identity)
            target_directory = namespace.directory
            namespace.publish_record(slot, content)

            def read() -> bytes | None:
                return namespace.read_record(slot)

        target = target_directory / (digest if mode == "blob" else slot)
        companion = target_directory / (".install-" + target.name)
        real_durable = storage._ImmutableSlotStore._durable_success
        replaced = False

        def durable(store: Any, *args: Any, **kwargs: Any) -> bytes:
            nonlocal replaced
            if not replaced:
                replaced = True
                target.unlink()
                companion.unlink()
                target.write_bytes(content)
                target.chmod(0o600)
                companion.write_bytes(content)
                companion.chmod(0o600)
            return real_durable(store, *args, **kwargs)

        monkeypatch.setattr(storage._ImmutableSlotStore, "_durable_success", durable)
        with pytest.raises(storage.MemoryStorageError) as failure:
            read()
        assert replaced
        assert failure.value.code in {"binding", "corrupt"}


def test_identity_replacement_during_namespace_open_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = "e" * 64
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        recovery = tmp_path / "memory/.recovery"
        kind = recovery / "transactions"
        real_open = storage.os.open
        replaced = False

        def open_file(*args: Any, **kwargs: Any) -> int:
            nonlocal replaced
            fd = real_open(*args, **kwargs)
            if not replaced and args and args[0] == identity:
                replaced = True
                victim = kind / identity
                moved = kind / (identity + ".old")
                victim.rename(moved)
                victim.mkdir(mode=0o700)
            return fd

        monkeypatch.setattr(storage.os, "open", open_file)
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.open_evidence_namespace("transactions", identity)
        assert replaced
        assert failure.value.code in {"binding", "io"}


def test_kind_replacement_during_record_link_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = "f" * 64
    slot = "genesis.json"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        real_link = storage.os.link
        replaced = False
        kind = namespace.directory.parent

        def link(*args: Any, **kwargs: Any) -> Any:
            nonlocal replaced
            result = real_link(*args, **kwargs)
            if not replaced:
                replaced = True
                moved = kind.with_name("transactions.old")
                kind.rename(moved)
                kind.mkdir(mode=0o700)
            return result

        monkeypatch.setattr(storage.os, "link", link)
        with pytest.raises(storage.MemoryStorageError) as failure:
            namespace.publish_record(slot, b"kind swap")
        assert replaced
        assert failure.value.code == "binding"


@pytest.mark.parametrize("phase", ["file", "directory"])
def test_identity_replacement_during_record_fsync_fails_closed(
    tmp_path: Path, phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = "0" * 64
    slot = "genesis.json"
    content = b"fsync binding"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("conversions", identity)
        record_fd = namespace._record_fd
        assert record_fd is not None
        target_identity: tuple[int, int] | None = None
        real_fsync = storage.os.fsync
        replaced = False

        def replace_namespace() -> None:
            victim = namespace.directory
            moved = victim.with_name(identity + ".old")
            victim.rename(moved)
            victim.mkdir(mode=0o700)

        def fsync(fd: int) -> None:
            nonlocal replaced
            real_fsync(fd)
            if replaced:
                return
            info = os.fstat(fd)
            if phase == "directory" and fd == record_fd:
                replaced = True
                replace_namespace()
            elif phase == "file" and target_identity == (info.st_dev, info.st_ino):
                replaced = True
                replace_namespace()

        real_link = storage.os.link

        def link(*args: Any, **kwargs: Any) -> Any:
            nonlocal target_identity
            result = real_link(*args, **kwargs)
            target_identity = (namespace.directory / slot).stat().st_dev, (namespace.directory / slot).stat().st_ino
            return result

        monkeypatch.setattr(storage.os, "link", link)
        monkeypatch.setattr(storage.os, "fsync", fsync)
        with pytest.raises(storage.MemoryStorageError) as failure:
            namespace.publish_record(slot, content)
        assert replaced
        assert failure.value.code == "binding"


def test_identity_replacement_during_clean_missing_read_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = "1" * 64
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        namespace = authority.open_evidence_namespace("transactions", identity)
        real_stat = storage.os.stat
        replaced = False

        def stat_entry(*args: Any, **kwargs: Any) -> os.stat_result:
            nonlocal replaced
            try:
                return real_stat(*args, **kwargs)
            except FileNotFoundError:
                if not replaced and args and args[0] == "genesis.json":
                    replaced = True
                    victim = namespace.directory
                    moved = victim.with_name(identity + ".old")
                    victim.rename(moved)
                    victim.mkdir(mode=0o700)
                raise

        monkeypatch.setattr(storage.os, "stat", stat_entry)
        with pytest.raises(storage.MemoryStorageError) as failure:
            namespace.read_record("genesis.json")
        assert replaced
        assert failure.value.code == "binding"


@pytest.mark.parametrize("fail_at", range(10))
def test_each_layout_sync_failure_is_repaired_on_retry(
    tmp_path: Path, fail_at: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fsync = storage.os.fsync
    calls: list[tuple[int, int]] = []
    armed = True

    def fsync(fd: int) -> None:
        nonlocal armed
        info = os.fstat(fd)
        calls.append((info.st_dev, info.st_ino))
        if armed and len(calls) - 1 == fail_at:
            armed = False
            raise OSError("secret creation-sync fault")
        real_fsync(fd)

    monkeypatch.setattr(storage.os, "fsync", fsync)
    with pytest.raises(storage.MemoryStorageError) as failure:
        with storage.WorkspaceMemoryAuthority(tmp_path):
            pass
    assert failure.value.code == "durability"
    assert failure.value.__cause__ is None

    calls.clear()
    with storage.WorkspaceMemoryAuthority(tmp_path):
        pass
    expected_paths = [
        tmp_path / ".memory-recovery.lock",
        tmp_path,
        tmp_path / "memory",
        tmp_path / "memory" / ".recovery",
        tmp_path / "memory" / ".recovery" / "blobs",
        _scope(tmp_path),
    ]
    expected = {(path.stat().st_dev, path.stat().st_ino) for path in expected_paths}
    assert expected <= set(calls)
    assert len(calls) >= 10


def test_first_publish_read_and_repeat_keep_permanent_hardlinked_companion(tmp_path: Path) -> None:
    content = b"memory recovery bytes\x00" * 4096
    digest = _digest(content)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        assert authority.publish_blob(content) == digest
        assert authority.read_blob(digest) == content
        assert authority.publish_blob(content) == digest

    target = _scope(tmp_path) / digest
    companion = _scope(tmp_path) / f".install-{digest}"
    assert target.read_bytes() == content
    assert companion.read_bytes() == content
    assert target.stat().st_ino == companion.stat().st_ino
    assert target.stat().st_nlink == companion.stat().st_nlink == 2


def test_link_wrapper_replacing_companion_exact_bytes_fails_identity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"retained companion identity"
    digest = _digest(content)
    real_link = storage.os.link
    original_inode: int | None = None

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        scope = _scope(tmp_path)

        def link(*args: Any, **kwargs: Any) -> Any:
            nonlocal original_inode
            companion = scope / f".install-{digest}"
            replacement = scope / ".replacement"
            original_inode = companion.stat().st_ino
            replacement.write_bytes(content)
            replacement.chmod(0o600)
            companion.unlink()
            replacement.rename(companion)
            return real_link(*args, **kwargs)

        monkeypatch.setattr(storage.os, "link", link)
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.publish_blob(content)
    assert failure.value.code == "binding"
    assert failure.value.__cause__ is None
    assert original_inode is not None
    assert (scope / f".install-{digest}").stat().st_ino != original_inode


def test_companion_only_install_is_finished_without_cleanup(tmp_path: Path) -> None:
    content = b"interrupted install"
    digest = _digest(content)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        authority.publish_blob(content)
    target = _scope(tmp_path) / digest
    companion = _scope(tmp_path) / f".install-{digest}"
    target.unlink()

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        assert authority.publish_blob(content) == digest
    assert target.read_bytes() == content
    assert companion.read_bytes() == content
    assert target.stat().st_ino == companion.stat().st_ino


def test_retry_after_post_link_directory_fsync_failure_repairs_durability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"directory fsync retry"
    real_fsync = storage.os.fsync
    real_link = storage.os.link
    linked = False
    failed = False

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        digest_fd = authority._digest_fd
        assert digest_fd is not None

        def link(*args: Any, **kwargs: Any) -> Any:
            nonlocal linked
            result = real_link(*args, **kwargs)
            linked = True
            return result

        def fsync(fd: int) -> None:
            nonlocal failed
            if linked and fd == digest_fd and not failed:
                failed = True
                raise OSError("injected directory fsync failure")
            real_fsync(fd)

        monkeypatch.setattr(storage.os, "link", link)
        monkeypatch.setattr(storage.os, "fsync", fsync)
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.publish_blob(content)
        assert failure.value.code == "durability"
        monkeypatch.setattr(storage.os, "link", real_link)
        monkeypatch.setattr(storage.os, "fsync", real_fsync)
        assert authority.publish_blob(content) == _digest(content)


@pytest.mark.parametrize("operation", ["publish", "read"])
@pytest.mark.parametrize("slot", ["target", "companion"])
@pytest.mark.parametrize("replacement", ["corrupt", "exact"])
def test_final_directory_fsync_revalidates_blob_state(
    tmp_path: Path,
    operation: str,
    slot: str,
    replacement: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"final directory fsync window"
    digest = _digest(content)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        authority.publish_blob(content)

    scope = _scope(tmp_path)
    real_fsync = storage.os.fsync
    replaced = False
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        digest_fd = authority._digest_fd
        assert digest_fd is not None

        def fsync(fd: int) -> None:
            nonlocal replaced
            real_fsync(fd)
            if fd != digest_fd or replaced:
                return
            replaced = True
            victim = scope / digest if slot == "target" else scope / f".install-{digest}"
            replacement_path = scope / ".replacement-during-fsync"
            replacement_path.write_bytes(content if replacement == "exact" else b"corrupt during fsync")
            replacement_path.chmod(0o600)
            victim.unlink()
            replacement_path.rename(victim)

        monkeypatch.setattr(storage.os, "fsync", fsync)
        with pytest.raises(storage.MemoryStorageError) as failure:
            if operation == "publish":
                authority.publish_blob(content)
            else:
                authority.read_blob(digest)
        assert replaced
        assert failure.value.code in {"binding", "corrupt", "conflict"}
        assert failure.value.__cause__ is None


@pytest.mark.parametrize("operation", ["publish", "read"])
@pytest.mark.parametrize("replacement", ["corrupt", "exact"])
def test_target_file_fsync_revalidates_independent_companion(
    tmp_path: Path,
    operation: str,
    replacement: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"target file fsync replacement window"
    digest = _digest(content)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        authority.publish_blob(content)

    copied = tmp_path.parent / f"{tmp_path.name}-target-fsync-copy"
    shutil.copytree(tmp_path, copied, copy_function=shutil.copy2)
    try:
        scope = _scope(copied)
        target = scope / digest
        companion = scope / f".install-{digest}"
        assert target.stat().st_nlink == companion.stat().st_nlink == 1
        target_identity = (target.stat().st_dev, target.stat().st_ino)
        real_fsync = storage.os.fsync
        replaced = False

        def install_companion(payload: bytes, name: str) -> None:
            replacement_path = scope / name
            replacement_path.write_bytes(payload)
            replacement_path.chmod(0o600)
            companion.unlink()
            replacement_path.rename(companion)

        def fsync(fd: int) -> None:
            nonlocal replaced
            real_fsync(fd)
            info = os.fstat(fd)
            if replaced or (info.st_dev, info.st_ino) != target_identity:
                return
            replaced = True
            install_companion(
                content if replacement == "exact" else b"corrupt during target fsync",
                ".replacement-during-target-fsync",
            )

        monkeypatch.setattr(storage.os, "fsync", fsync)
        with storage.WorkspaceMemoryAuthority(copied) as authority:
            with pytest.raises(storage.MemoryStorageError) as failure:
                if operation == "publish":
                    authority.publish_blob(content)
                else:
                    authority.read_blob(digest)
            assert replaced
            assert failure.value.code in (
                {"binding"} if replacement == "exact" else {"corrupt", "conflict"}
            )
            assert failure.value.__cause__ is None
            assert "secret" not in str(failure.value)

            if operation == "publish":
                install_companion(content, ".repaired-companion")
                assert authority.publish_blob(content) == digest

    finally:
        shutil.rmtree(copied)


@pytest.mark.parametrize("kind", ["regular", "symlink", "fifo", "directory", "hardlink"])
def test_destination_collision_never_changes_existing_entry(tmp_path: Path, kind: str) -> None:
    if kind == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("FIFO collision requires os.mkfifo")
    content = b"expected bytes"
    other = b"different bytes"
    digest = _digest(content)
    target = _scope(tmp_path) / digest
    outside = tmp_path / "outside"
    outside.write_bytes(other)
    with storage.WorkspaceMemoryAuthority(tmp_path):
        pass
    if kind == "regular":
        target.write_bytes(other)
        target.chmod(0o600)
    elif kind == "symlink":
        target.symlink_to(outside)
    elif kind == "fifo":
        os.mkfifo(target, 0o600)
    elif kind == "directory":
        target.mkdir(mode=0o700)
    else:
        outside.write_bytes(content)
        os.link(outside, target)
        target.chmod(0o600)
    before = target.lstat()
    before_bytes = target.read_bytes() if kind in {"regular", "hardlink"} else None

    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError):
            authority.publish_blob(content)
    after = target.lstat()
    assert (after.st_dev, after.st_ino, after.st_mode, after.st_nlink) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
    )
    if before_bytes is not None:
        assert target.read_bytes() == before_bytes
    assert not (target.parent / f".install-{digest}").exists()


def test_companion_collision_is_rejected_without_replacement(tmp_path: Path) -> None:
    content = b"companion source"
    digest = _digest(content)
    with storage.WorkspaceMemoryAuthority(tmp_path):
        pass
    companion = _scope(tmp_path) / f".install-{digest}"
    companion.write_bytes(b"wrong")
    companion.chmod(0o600)
    before = companion.read_bytes()
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.publish_blob(content)
    assert failure.value.code == "conflict"
    assert companion.read_bytes() == before
    assert not (_scope(tmp_path) / digest).exists()


@pytest.mark.parametrize("syscall", ["open", "fstat", "stat", "read", "write", "fsync", "link"])
def test_blob_syscall_faults_are_fixed_content_free_errors(
    tmp_path: Path, syscall: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"secret-bearing syscall fault"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        real = getattr(storage.os, syscall)

        def fail(*args: Any, **kwargs: Any) -> Any:
            raise OSError(f"secret path and bytes from {syscall}")

        monkeypatch.setattr(storage.os, syscall, fail)
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.publish_blob(content)
        assert isinstance(failure.value, storage.MemoryStorageError)
        assert failure.value.__cause__ is None
        assert "secret" not in str(failure.value)
        assert "path" not in str(failure.value)
        assert "bytes" not in str(failure.value)
        monkeypatch.setattr(storage.os, syscall, real)


@pytest.mark.parametrize("component", ["memory", ".recovery", "blobs", "sha256"])
def test_replaced_layout_component_fails_closed(tmp_path: Path, component: str) -> None:
    content = b"bound layout"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        authority.publish_blob(content)
    original = tmp_path / "memory" / ".recovery" / "blobs" / "sha256"
    paths = {
        "memory": tmp_path / "memory",
        ".recovery": tmp_path / "memory" / ".recovery",
        "blobs": tmp_path / "memory" / ".recovery" / "blobs",
        "sha256": original,
    }
    victim = paths[component]
    moved = victim.with_name(victim.name + ".old")
    victim.rename(moved)
    victim.symlink_to(tmp_path)
    try:
        with pytest.raises(storage.MemoryStorageError) as failure:
            with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
                authority.publish_blob(content)
        assert failure.value.code in {"corrupt", "io", "binding"}
    finally:
        victim.unlink()
        moved.rename(victim)


def test_lock_replacement_is_detected_before_publication(tmp_path: Path) -> None:
    lock = tmp_path / ".memory-recovery.lock"
    moved = tmp_path / ".old-lock"
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        lock.rename(moved)
        lock.touch(mode=0o600)
        try:
            with pytest.raises(storage.MemoryStorageError) as failure:
                authority.publish_blob(b"must fail")
            assert failure.value.code in {"binding", "io"}
        finally:
            lock.unlink()
            moved.rename(lock)


def test_two_process_publishers_coexist_without_hanging(tmp_path: Path) -> None:
    if not _fork_capable():
        pytest.skip("concurrent publisher test requires multiprocessing fork")
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    result = context.Queue()
    processes = [
        context.Process(target=_publish_child, args=(str(tmp_path), payload, barrier, result))
        for payload in (b"same", b"same")
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=5)
        assert all(not process.is_alive() for process in processes)
        results = [result.get(timeout=2) for _ in processes]
        assert all(item[0] == "ok" and item[1] == _digest(b"same") for item in results)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)


def test_two_process_publishers_keep_different_digests(tmp_path: Path) -> None:
    if not _fork_capable():
        pytest.skip("concurrent publisher test requires multiprocessing fork")
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    result = context.Queue()
    payloads = (b"first digest", b"second digest")
    processes = [
        context.Process(target=_publish_child, args=(str(tmp_path), payload, barrier, result))
        for payload in payloads
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=5)
        assert all(not process.is_alive() for process in processes)
        results = [result.get(timeout=2) for _ in processes]
        assert {item[1] for item in results if item[0] == "ok"} == {_digest(value) for value in payloads}
        assert {_scope(tmp_path).joinpath(_digest(value)).read_bytes() for value in payloads} == set(payloads)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)


def test_copied_tree_with_independent_files_is_read_by_content(tmp_path: Path) -> None:
    content = b"copied immutable tree"
    digest = _digest(content)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        authority.publish_blob(content)
    copied = tmp_path.parent / f"{tmp_path.name}-copy"
    shutil.copytree(tmp_path, copied, copy_function=shutil.copy2)
    try:
        assert (copied / "memory/.recovery/blobs/sha256" / digest).stat().st_nlink == 1
        with storage.WorkspaceMemoryAuthority(copied) as authority:
            assert authority.read_blob(digest) == content
    finally:
        shutil.rmtree(copied)


def test_corruption_and_public_errors_are_content_free(tmp_path: Path) -> None:
    content = b"private bytes"
    digest = _digest(content)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        authority.publish_blob(content)
    target = _scope(tmp_path) / digest
    target.write_bytes(b"corrupt")
    target.chmod(0o600)
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        with pytest.raises(storage.MemoryStorageError) as failure:
            authority.read_blob(digest)
    assert failure.value.code == "corrupt"
    assert str(failure.value) == "memory recovery storage content is unsafe"
    assert digest not in str(failure.value)
    assert "private bytes" not in str(failure.value)
    assert failure.value.__cause__ is None


def test_unavailable_capability_fails_only_when_authority_is_invoked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    imported = storage.MemoryStorageError
    monkeypatch.setattr(storage.os, "O_NOFOLLOW", None)
    with pytest.raises(storage.MemoryStorageUnavailableError):
        storage.WorkspaceMemoryAuthority(tmp_path)
    assert imported is storage.MemoryStorageError


def test_no_mutable_replacement_or_cleanup_primitives_are_used() -> None:
    source = Path(storage.__file__).read_text(encoding="utf-8")
    for forbidden in ("os.replace(", "os.rename(", "os.chmod(", "os.unlink("):
        assert forbidden not in source
