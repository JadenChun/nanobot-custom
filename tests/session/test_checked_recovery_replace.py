from __future__ import annotations

import errno
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import nanobot.session.checked_source as checked_source
from nanobot.session.checked_source import (
    CheckedSessionAuthorityError,
    open_checked_recovery_session,
    open_checked_recovery_session_for_replace,
)
from tests.session.test_checked_recovery_read import PAYLOAD, _candidate, _layout, _lease


def _error(action, code: str) -> None:
    with pytest.raises(CheckedSessionAuthorityError) as caught:
        action()
    assert caught.value.code == code


def test_read_only_factory_and_explicit_one_use_replace(tmp_path: Path) -> None:
    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    with _lease(workspace) as operation:
        with open_checked_recovery_session(operation, candidate) as authority:
            authority.read_source_bytes()
            _error(lambda: authority.replace_recovery_source_bytes(b"new"), "invalid")
        with open_checked_recovery_session_for_replace(operation, candidate) as authority:
            _error(lambda: authority.replace_source_bytes(b"new"), "invalid")
            _error(lambda: authority.replace_recovery_source_bytes(b"new"), "invalid")
            assert authority.read_source_bytes() == source.read_bytes()
            authority.replace_recovery_source_bytes(b"new")
            assert authority.read_source_bytes() == b"new"
            _error(lambda: authority.replace_recovery_source_bytes(b"again"), "invalid")
    assert source.read_bytes() == b"new"
    assert source.stat().st_mode & 0o777 == 0o600
    assert candidate.session_file_identity != checked_source._recovery_file_identity(source.stat())


def test_recovery_replace_uses_pinned_namespace_even_with_decoy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    original = os.replace
    calls = []

    def inspected(src, dst, **kwargs):
        calls.append(kwargs)
        assert kwargs["src_dir_fd"] == kwargs["dst_dir_fd"] == authority._recovery_namespace.fd
        return original(src, dst, **kwargs)

    with _lease(workspace) as operation:
        with open_checked_recovery_session_for_replace(operation, candidate) as authority:
            authority.read_source_bytes()
            monkeypatch.setattr(os, "replace", inspected)
            authority.replace_recovery_source_bytes(b"new")
    assert len(calls) == 1
    assert source.read_bytes() == b"new"


@pytest.mark.parametrize("mutation", ["bytes", "stat"])
def test_stale_source_retains_original_or_external_edit(tmp_path: Path, mutation: str) -> None:
    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    with _lease(workspace) as operation:
        authority = open_checked_recovery_session_for_replace(operation, candidate)
        authority.__enter__()
        try:
            authority.read_source_bytes()
            if mutation == "bytes":
                source.write_bytes(b"external")
            else:
                os.utime(source, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 1_000_000))
            _error(lambda: authority.replace_recovery_source_bytes(b"new"), "changed")
        finally:
            authority.close()
    assert source.read_bytes() != b"new"


def test_final_comparison_race_blocks_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    original = checked_source.CheckedSessionAuthority.read_source_bytes
    reads = 0
    renames = []

    def racing(self):
        nonlocal reads
        reads += 1
        if reads == 2:
            source.write_bytes(b"raced")
        return original(self)

    with _lease(workspace) as operation:
        authority = open_checked_recovery_session_for_replace(operation, candidate)
        authority.__enter__()
        try:
            assert authority.read_source_bytes()
            monkeypatch.setattr(checked_source.CheckedSessionAuthority, "read_source_bytes", racing)
            monkeypatch.setattr(os, "replace", lambda *a, **kw: renames.append((a, kw)))
            _error(lambda: authority.replace_recovery_source_bytes(b"new"), "changed")
        finally:
            authority.close()
    assert renames == []
    assert source.read_bytes() == b"raced"


@pytest.mark.parametrize("failure", ["rename", "postcheck", "fsync"])
def test_after_rename_is_uncertain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    original_replace = os.replace
    original_validate = checked_source.CheckedSessionAuthority._validate_all
    original_fsync = os.fsync
    attempted = False

    def replace(*args, **kwargs):
        nonlocal attempted
        attempted = True
        original_replace(*args, **kwargs)
        if failure == "rename":
            raise OSError(errno.EIO, "ambiguous")

    def validate(self):
        if attempted and failure == "postcheck":
            raise OSError(errno.EIO, "postcheck")
        return original_validate(self)

    def fsync(fd):
        if attempted and failure == "fsync":
            raise OSError(errno.EIO, "directory fsync")
        return original_fsync(fd)

    with _lease(workspace) as operation:
        authority = open_checked_recovery_session_for_replace(operation, candidate)
        authority.__enter__()
        try:
            authority.read_source_bytes()
            monkeypatch.setattr(os, "replace", replace)
            monkeypatch.setattr(checked_source.CheckedSessionAuthority, "_validate_all", validate)
            monkeypatch.setattr(os, "fsync", fsync)
            _error(lambda: authority.replace_recovery_source_bytes(b"new"), "uncertain")
        finally:
            authority.close()
    assert attempted
    assert source.read_bytes() == b"new"


def test_wrong_thread_and_closed_lifetime(tmp_path: Path) -> None:
    workspace, root, _ = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    with _lease(workspace) as operation:
        authority = open_checked_recovery_session_for_replace(operation, candidate)
        with authority:
            authority.read_source_bytes()
            with ThreadPoolExecutor(max_workers=1) as executor:
                assert executor.submit(lambda: _error(lambda: authority.replace_recovery_source_bytes(b"new"), "invalid")).result() is None
        _error(lambda: authority.replace_recovery_source_bytes(b"new"), "closed")


@pytest.mark.parametrize("drift", ["workspace-marker", "namespace-marker", "lock", "duplicate"])
def test_recovery_binding_drift_blocks_replace(tmp_path: Path, drift: str) -> None:
    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    with _lease(workspace) as operation:
        authority = open_checked_recovery_session_for_replace(operation, candidate)
        authority.__enter__()
        try:
            authority.read_source_bytes()
            if drift == "workspace-marker":
                (workspace / ".nanobot" / "workspace-id").write_bytes(b"b" * 32 + b"\n")
            elif drift == "namespace-marker":
                (candidate.namespace_path / ".workspace").write_bytes(b"/other\n")
            elif drift == "lock":
                lock = candidate.namespace_path / ".session-files.lock"
                lock.unlink()
                lock.write_bytes(b"")
                lock.chmod(0o600)
            else:
                duplicate = root / ("b" * 32)
                duplicate.mkdir(mode=0o700)
                (duplicate / ".workspace").write_bytes(f"{workspace}\n".encode())
            _error(lambda: authority.replace_recovery_source_bytes(b"new"), "unsafe" if drift == "lock" else "changed")
        finally:
            authority.close()
    assert source.read_bytes() != b"new"


def test_lease_drift_blocks_replace(tmp_path: Path) -> None:
    from nanobot.agent.memory_recovery_storage import WorkspaceMemoryAuthority

    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    with WorkspaceMemoryAuthority(workspace) as memory:
        operation = memory.operation()
        operation.__enter__()
        authority = open_checked_recovery_session_for_replace(operation, candidate)
        authority.__enter__()
        authority.read_source_bytes()
        operation.__exit__(None, None, None)
        _error(lambda: authority.replace_recovery_source_bytes(b"new"), "changed")
        authority.close()
    assert source.read_bytes() != b"new"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_fork_child_cannot_replace(tmp_path: Path) -> None:
    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    with _lease(workspace) as operation:
        with open_checked_recovery_session_for_replace(operation, candidate) as authority:
            authority.read_source_bytes()
            read_fd, write_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                try:
                    authority.replace_recovery_source_bytes(b"new")
                except CheckedSessionAuthorityError as error:
                    os.write(write_fd, error.code.encode())
                finally:
                    os._exit(0)
            os.close(write_fd)
            result = os.read(read_fd, 80)
            os.close(read_fd)
            os.waitpid(pid, 0)
            assert result == b"invalid"
    assert source.read_bytes() != b"new"


def test_namespace_decoy_rejected_before_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    replaced = []
    with _lease(workspace) as operation:
        authority = open_checked_recovery_session_for_replace(operation, candidate)
        authority.__enter__()
        try:
            authority.read_source_bytes()
            detached = root / "detached"
            os.rename(candidate.namespace_path, detached)
            candidate.namespace_path.mkdir(mode=0o700)
            (candidate.namespace_path / ".workspace").write_bytes(f"{workspace}\n".encode())
            decoy = candidate.namespace_path / source.name
            decoy.write_bytes(b"decoy")
            decoy.chmod(0o600)
            monkeypatch.setattr(os, "replace", lambda *a, **kw: replaced.append(a))
            _error(lambda: authority.replace_recovery_source_bytes(b"new"), "changed")
            assert decoy.read_bytes() == b"decoy"
            assert (detached / source.name).read_bytes() != b"new"
        finally:
            authority.close()
    assert not replaced


def test_verify_recovery_source_is_durable_and_does_not_consume_replace(
    tmp_path: Path,
) -> None:
    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    original_identity = checked_source._recovery_file_identity(source.stat())
    with _lease(workspace) as operation:
        with open_checked_recovery_session_for_replace(operation, candidate) as authority:
            assert authority.read_source_bytes() == source.read_bytes()
            assert authority.verify_recovery_source_bytes(source.read_bytes()) is None
            assert checked_source._recovery_file_identity(source.stat()) == original_identity
            authority.replace_recovery_source_bytes(b"new")
    assert source.read_bytes() == b"new"


@pytest.mark.parametrize("drift", ["bytes", "stat"])
def test_verify_recovery_source_rejects_candidate_drift(
    tmp_path: Path, drift: str
) -> None:
    workspace, root, source = _layout(tmp_path)
    if drift == "bytes":
        source.write_bytes(b"corrupt")
    candidate = _candidate(workspace, root)
    with _lease(workspace) as operation:
        authority = open_checked_recovery_session(operation, candidate)
        authority.__enter__()
        try:
            if drift == "stat":
                os.utime(
                    source,
                    ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns + 1_000_000),
                )
            _error(lambda: authority.verify_recovery_source_bytes(PAYLOAD), "changed")
        finally:
            authority.close()


@pytest.mark.parametrize("unsafe", ["symlink", "mode"])
def test_verify_recovery_source_rejects_unsafe_entries(
    tmp_path: Path, unsafe: str
) -> None:
    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    with _lease(workspace) as operation:
        authority = open_checked_recovery_session(operation, candidate)
        authority.__enter__()
        try:
            if unsafe == "symlink":
                target = source.with_suffix(".target")
                target.write_bytes(PAYLOAD)
                target.chmod(0o600)
                source.unlink()
                source.symlink_to(target)
            else:
                source.chmod(0o622)
            _error(lambda: authority.verify_recovery_source_bytes(PAYLOAD), "unsafe")
        finally:
            authority.close()


@pytest.mark.parametrize("failure", ["file", "directory"])
def test_verify_recovery_source_fsync_failure_is_content_free_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    original_fsync = os.fsync
    attempted = []

    def fail_fsync(descriptor: int) -> None:
        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        attempted.append("directory" if is_directory else "file")
        if attempted[-1] == failure:
            raise OSError(errno.EIO, "injected fsync failure")
        original_fsync(descriptor)

    with _lease(workspace) as operation:
        with open_checked_recovery_session(operation, candidate) as authority:
            monkeypatch.setattr(os, "fsync", fail_fsync)
            with pytest.raises(CheckedSessionAuthorityError) as caught:
                authority.verify_recovery_source_bytes(PAYLOAD)
            error = caught.value
            assert error.code == "io"
            assert error.__cause__ is None
            assert error.__context__ is None
            assert str(error) == "checked session authority: io"
    assert failure in attempted
    assert source.read_bytes() == PAYLOAD


def test_verify_recovery_source_rejects_namespace_identity_change(tmp_path: Path) -> None:
    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    with _lease(workspace) as operation:
        authority = open_checked_recovery_session(operation, candidate)
        authority.__enter__()
        try:
            detached = root / "detached"
            os.rename(candidate.namespace_path, detached)
            candidate.namespace_path.mkdir(mode=0o700)
            (candidate.namespace_path / ".workspace").write_bytes(
                f"{workspace}\n".encode()
            )
            decoy = candidate.namespace_path / source.name
            decoy.write_bytes(PAYLOAD)
            decoy.chmod(0o600)
            _error(lambda: authority.verify_recovery_source_bytes(PAYLOAD), "changed")
        finally:
            authority.close()


def test_fresh_authority_verifies_source_after_uncertain_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    original_replace = os.replace

    def replace_then_fail(*args, **kwargs) -> None:
        original_replace(*args, **kwargs)
        raise OSError(errno.EIO, "ambiguous rename result")

    with _lease(workspace) as operation:
        authority = open_checked_recovery_session_for_replace(operation, candidate)
        authority.__enter__()
        try:
            authority.read_source_bytes()
            monkeypatch.setattr(os, "replace", replace_then_fail)
            _error(lambda: authority.replace_recovery_source_bytes(b"new"), "uncertain")
        finally:
            authority.close()

        fresh_candidate = _candidate(workspace, root)
        identity_after_rename = checked_source._recovery_file_identity(source.stat())
        with open_checked_recovery_session(operation, fresh_candidate) as fresh_authority:
            assert fresh_authority.verify_recovery_source_bytes(b"new") is None
        assert source.read_bytes() == b"new"
        assert checked_source._recovery_file_identity(source.stat()) == identity_after_rename


def test_verify_recovery_source_preserves_platform_capability_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, root, _ = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    with _lease(workspace) as operation:
        with open_checked_recovery_session(operation, candidate) as authority:
            with monkeypatch.context() as capability_patch:
                capability_patch.setattr(
                    checked_source,
                    "_require_source_read_platform",
                    lambda: checked_source._fail("unavailable"),
                )
                _error(
                    lambda: authority.verify_recovery_source_bytes(PAYLOAD),
                    "unavailable",
                )


def test_verify_recovery_source_rejects_missing_fsync_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, root, _ = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    with _lease(workspace) as operation:
        with open_checked_recovery_session(operation, candidate) as authority:
            monkeypatch.setattr(os, "fsync", None)
            _error(
                lambda: authority.verify_recovery_source_bytes(PAYLOAD), "unavailable"
            )


def test_verify_recovery_source_maps_unsupported_directory_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, root, _ = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    original_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(errno.EINVAL, "unsupported directory fsync")
        original_fsync(descriptor)

    with _lease(workspace) as operation:
        with open_checked_recovery_session(operation, candidate) as authority:
            monkeypatch.setattr(os, "fsync", fail_directory_fsync)
            _error(
                lambda: authority.verify_recovery_source_bytes(PAYLOAD), "unavailable"
            )


def test_authority_after_directory_fsync_failure_requires_fresh_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, root, source = _layout(tmp_path)
    candidate = _candidate(workspace, root)
    original_replace = os.replace
    original_fsync = os.fsync
    rename_completed = False

    def replace(*args, **kwargs) -> None:
        nonlocal rename_completed
        original_replace(*args, **kwargs)
        rename_completed = True

    def fail_directory_fsync(descriptor: int) -> None:
        if rename_completed and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(errno.EIO, "directory fsync failed after rename")
        original_fsync(descriptor)

    with _lease(workspace) as operation:
        with open_checked_recovery_session_for_replace(operation, candidate) as authority:
            assert authority.read_source_bytes() == PAYLOAD
            with monkeypatch.context() as fault:
                fault.setattr(os, "replace", replace)
                fault.setattr(os, "fsync", fail_directory_fsync)
                _error(lambda: authority.replace_recovery_source_bytes(b"new"), "uncertain")

            uncertain_identity = checked_source._recovery_file_identity(source.stat())
            _error(lambda: authority.verify_recovery_source_bytes(b"new"), "invalid")
            assert source.read_bytes() == b"new"
            assert checked_source._recovery_file_identity(source.stat()) == uncertain_identity

        fresh_candidate = _candidate(workspace, root)
        before_fresh_verify = checked_source._recovery_file_identity(source.stat())
        with open_checked_recovery_session(operation, fresh_candidate) as fresh_authority:
            assert fresh_authority.verify_recovery_source_bytes(b"new") is None
        assert source.read_bytes() == b"new"
        assert checked_source._recovery_file_identity(source.stat()) == before_fresh_verify
