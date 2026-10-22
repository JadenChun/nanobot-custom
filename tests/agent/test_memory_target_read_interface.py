"""Supported recovery reads retain the private helpers' policy and fault seams."""

from __future__ import annotations

import errno
import inspect
import os
import stat
from pathlib import Path

import pytest

from nanobot.agent import memory_recovery_storage as storage

_INFO = object()
_FORWARDING = (
    ("memory_target_stat_identity", "_target_stat_identity", (_INFO,), {}, {}, True),
    ("require_safe_memory_target", "_valid_memory_target", ("MEMORY.md", _INFO), {}, {}, False),
    ("stat_recovery_descriptor", "_fstat", (37,), {}, {"code": "io"}, True),
    ("stat_recovery_descriptor", "_fstat", (37,), {"code": "binding"}, {"code": "binding"}, True),
    ("stat_recovery_entry", "_stat_entry", (37, "MEMORY.md"), {}, {}, True),
    ("close_recovery_descriptor", "_close", (37,), {}, {}, False),
)


def test_target_limit_and_supported_exports_are_exact() -> None:
    limit = getattr(storage, "MEMORY_TARGET_MAX_BYTES")
    assert limit is storage._MEMORY_TARGET_MAX_BYTES
    assert limit == 64 * 1024 * 1024
    names = {case[0] for case in _FORWARDING} | {"MEMORY_TARGET_MAX_BYTES"}
    assert names <= set(storage.__all__)


@pytest.mark.parametrize("public,private,args,kwargs,forwarded,returns", _FORWARDING)
def test_read_interfaces_forward_dynamically_and_exactly(
    monkeypatch, public, private, args, kwargs, forwarded, returns,
) -> None:
    operation = getattr(storage, public)
    result = object() if returns else None
    calls = []

    def first(*actual_args, **actual_kwargs):
        calls.append(("first", actual_args, actual_kwargs))
        return result

    def second(*actual_args, **actual_kwargs):
        calls.append(("second", actual_args, actual_kwargs))
        return result

    monkeypatch.setattr(storage, private, first)
    assert operation(*args, **kwargs) is result
    monkeypatch.setattr(storage, private, second)
    assert operation(*args, **kwargs) is result
    assert calls == [("first", args, forwarded), ("second", args, forwarded)]
    if public == "stat_recovery_descriptor":
        code = inspect.signature(operation).parameters["code"]
        assert code.kind is inspect.Parameter.KEYWORD_ONLY
        assert code.default == "io"


@pytest.mark.parametrize("public,private,args,kwargs,forwarded,returns", _FORWARDING)
@pytest.mark.parametrize("fault", [
    OSError("read"), RuntimeError("read"), KeyboardInterrupt("read"),
    storage.MemoryStorageError("binding"),
])
def test_read_interfaces_preserve_private_fault_identity(
    monkeypatch, public, private, args, kwargs, forwarded, returns, fault,
) -> None:
    operation = getattr(storage, public)

    def fail(*actual_args, **actual_kwargs):
        assert actual_args == args and actual_kwargs == forwarded
        raise fault

    monkeypatch.setattr(storage, private, fail)
    with pytest.raises(type(fault)) as raised:
        operation(*args, **kwargs)
    assert raised.value is fault


def test_real_named_and_descriptor_stats_keep_no_follow_and_exact_metadata(tmp_path: Path) -> None:
    named_stat = getattr(storage, "stat_recovery_entry")
    target = tmp_path / "MEMORY.md"
    target.write_bytes(b"durable memory")
    target.chmod(0o600)
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    target_fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        named = named_stat(parent_fd, target.name)
        descriptor = storage.stat_recovery_descriptor(target_fd)
        assert named is not None
        assert (named.st_dev, named.st_ino) == (descriptor.st_dev, descriptor.st_ino)
        identity = storage.memory_target_stat_identity(descriptor)
        assert identity == storage.MemoryTargetStatIdentity(
            device=descriptor.st_dev, inode=descriptor.st_ino, mode=descriptor.st_mode,
            uid=descriptor.st_uid, gid=descriptor.st_gid, links=descriptor.st_nlink,
            size=descriptor.st_size, mtime_ns=descriptor.st_mtime_ns,
            ctime_ns=descriptor.st_ctime_ns,
        )
        assert named_stat(parent_fd, "absent") is None
        observed_alias = named_stat(parent_fd, alias.name)
        assert observed_alias is not None and stat.S_ISLNK(observed_alias.st_mode)
        assert storage.require_safe_memory_target(target.name, descriptor) is None
        assert target.read_bytes() == b"durable memory"
    finally:
        os.close(target_fd)
        os.close(parent_fd)


@pytest.mark.parametrize("code", [None, "binding"])
def test_descriptor_stat_maps_real_bad_descriptor_with_original_code(code: str | None) -> None:
    descriptor_stat = getattr(storage, "stat_recovery_descriptor")
    kwargs = {} if code is None else {"code": code}
    with pytest.raises(storage.MemoryStorageError) as raised:
        descriptor_stat(-1, **kwargs)
    assert raised.value.code == ("io" if code is None else code)
    assert raised.value.__cause__ is None


def test_named_stat_retains_real_io_error_mapping() -> None:
    named_stat = getattr(storage, "stat_recovery_entry")
    with pytest.raises(storage.MemoryStorageError) as raised:
        named_stat(-1, "MEMORY.md")
    assert raised.value.code == "io"
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("name,mode,allowed", [
    ("MEMORY.md", 0o644, True), ("MEMORY.md", 0o400, True),
    ("MEMORY.md", 0o600, True), ("MEMORY.md", 0o660, False),
    ("MEMORY.md", 0o200, False), ("MEMORY.md", 0o1600, False),
    (".consolidation.pending.json", 0o600, True),
    (".consolidation.pending.json", 0o644, False),
])
def test_real_target_mode_policy_distinguishes_memory_and_private_receipts(
    tmp_path: Path, name: str, mode: int, allowed: bool,
) -> None:
    validate = getattr(storage, "require_safe_memory_target")
    target = tmp_path / name
    target.write_bytes(b"target")
    target.chmod(mode)
    before = target.lstat()
    assert stat.S_IMODE(before.st_mode) == mode
    if allowed:
        assert validate(name, before) is None
    else:
        with pytest.raises(storage.MemoryStorageError) as raised:
            validate(name, before)
        assert raised.value.code == "corrupt"
    assert target.lstat() == before


@pytest.mark.parametrize("name", ["MEMORY.md", ".consolidation.pending.json"])
@pytest.mark.parametrize("unsafe", ["directory", "symlink", "hardlink"])
def test_real_target_type_and_link_refusals(tmp_path: Path, name: str, unsafe: str) -> None:
    validate = getattr(storage, "require_safe_memory_target")
    target = tmp_path / name
    original = tmp_path / "original"
    original.write_bytes(b"sentinel")
    original.chmod(0o600)
    if unsafe == "directory":
        target.mkdir(mode=0o700)
    elif unsafe == "symlink":
        target.symlink_to(original)
    else:
        os.link(original, target)
    before = target.lstat(), original.lstat()
    with pytest.raises(storage.MemoryStorageError) as raised:
        validate(name, target.lstat())
    assert raised.value.code == "corrupt"
    assert (target.lstat(), original.lstat()) == before
    assert original.read_bytes() == b"sentinel"


def test_target_validation_retains_owner_mismatch_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate = getattr(storage, "require_safe_memory_target")
    target = tmp_path / "MEMORY.md"
    target.write_bytes(b"target")
    target.chmod(0o600)
    info = target.lstat()
    monkeypatch.setattr(storage, "_uid", lambda: info.st_uid + 1)
    with pytest.raises(storage.MemoryStorageError) as raised:
        validate(target.name, info)
    assert raised.value.code == "corrupt"
    assert target.lstat() == info


def test_cleanup_closes_real_owned_descriptor_and_retains_idempotence(tmp_path: Path) -> None:
    close = getattr(storage, "close_recovery_descriptor")
    target = tmp_path / "MEMORY.md"
    target.write_bytes(b"memory")
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    assert close(None) is None
    assert stat.S_ISREG(os.fstat(fd).st_mode)
    assert close(fd) is None
    with pytest.raises(OSError) as raised:
        os.fstat(fd)
    assert raised.value.errno == errno.EBADF
    assert close(fd) is None
    assert target.read_bytes() == b"memory"
