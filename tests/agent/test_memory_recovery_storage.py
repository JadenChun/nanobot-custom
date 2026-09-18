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
