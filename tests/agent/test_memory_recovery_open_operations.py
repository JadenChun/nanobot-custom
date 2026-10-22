"""Descriptor compositions preserve consumer lookup, faults and effect order."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_storage as storage
from nanobot.agent import memory_recovery_target_verify as verifier
from nanobot.agent import memory_recovery_transaction_memory as memory_stage

OPERATIONS = (
    ("open_recovery_absolute_directory", "_dir_flags", False, False),
    ("open_recovery_relative_directory", "_dir_flags", True, False),
    ("open_recovery_read_leaf", "_leaf_flags", True, False),
    ("create_recovery_exclusive_leaf", "_write_flags", True, True),
)


@pytest.mark.parametrize("name,helper,relative,write", OPERATIONS)
@pytest.mark.parametrize("cloexec", [None, "not-int", 128])
def test_operation_exact_composition(monkeypatch, name, helper, relative, write, cloexec):
    operation = getattr(storage, name)  # Missing APIs fail here, not during collection.
    flags_os = SimpleNamespace(O_RDONLY=1, O_DIRECTORY=2, O_NOFOLLOW=4,
                               O_NONBLOCK=8, O_WRONLY=16, O_CREAT=32, O_EXCL=64)
    if cloexec is not None:
        flags_os.O_CLOEXEC = cloexec
    monkeypatch.setattr(storage, "os", flags_os)
    original = getattr(storage, helper)
    trace = []
    result = object()

    def flags():
        trace.append("flags")
        return original()

    def opener(*args, **kwargs):
        trace.append((args, kwargs))
        return result

    monkeypatch.setattr(storage, helper, flags)
    arguments = (opener, "name", 37) if relative else (opener, "name")
    assert operation(*arguments) is result
    expected = (16 | 32 | 64 | 4 | 8) if write else (1 | 4 | 8)
    if helper == "_dir_flags":
        expected |= 2
    if isinstance(cloexec, int):
        expected |= cloexec
    syscall = ("name", expected, 0o600) if write else ("name", expected)
    assert trace == ["flags", (syscall, {"dir_fd": 37} if relative else {})]


@pytest.mark.parametrize("name,helper,relative,write", OPERATIONS)
@pytest.mark.parametrize("kind", [OSError, ValueError, TypeError, RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("point", ["flags", "open"])
def test_operation_fault_identity(monkeypatch, name, helper, relative, write, kind, point):
    operation = getattr(storage, name)
    fault = kind("injected")
    trace = []

    def flags():
        trace.append("flags")
        if point == "flags":
            raise fault
        return 23

    def opener(*args, **kwargs):
        trace.append("open")
        raise fault

    monkeypatch.setattr(storage, helper, flags)
    with pytest.raises(kind) as raised:
        operation(*((opener, "name", 37) if relative else (opener, "name")))
    assert raised.value is fault
    assert trace == (["flags"] if point == "flags" else ["flags", "open"])


def _consumer(monkeypatch, site, trace):
    """Arrange all quarantine open sites without permitting real filesystem effects."""
    info = object()
    monkeypatch.setattr(quarantine, "_entry", lambda *args: info)
    monkeypatch.setattr(quarantine, "_private_dir", lambda *args, **kwargs: None)
    monkeypatch.setattr(quarantine, "_private_file", lambda *args: None)
    monkeypatch.setattr(quarantine, "_stat_tuple", lambda value: (1,))
    monkeypatch.setattr(quarantine, "_directory_tuple", lambda value: (1,))
    monkeypatch.setattr(quarantine, "_read", lambda *args: (b"x", info))
    proxy = SimpleNamespace(open=None, close=lambda fd: trace.append(("close", fd)),
                            fstat=lambda fd: info, stat=lambda *args, **kwargs: info,
                            fsync=lambda fd: trace.append(("fsync", fd)),
                            write=lambda fd, data: trace.append(("write", fd, data)) or len(data))
    monkeypatch.setattr(quarantine, "os", proxy)
    monkeypatch.setattr(verifier, "os", proxy)
    if site in {"root", "walk"}:
        monkeypatch.setattr(quarantine.stat, "S_ISDIR", lambda mode: True)
        proxy.stat = lambda *args, **kwargs: SimpleNamespace(st_mode=0)
        return proxy, "_dir_flags", lambda: quarantine._open_absolute(Path("/a") if site == "walk" else Path("/"))
    if site == "child":
        return proxy, "_dir_flags", lambda: quarantine._child(37, "name")
    if site == "read":
        # Restore the real read consumer; its first post-open fstat is our sentinel.
        monkeypatch.undo()
        return _read_consumer(monkeypatch, trace)
    if site == "sync":
        return proxy, "_leaf_flags", lambda: quarantine._sync_checked_file(37, "name", 10)
    if site == "verify":
        return proxy, "_leaf_flags", lambda: verifier._open_target(37, "name")
    store = object.__new__(quarantine.ReceiptQuarantineStore)
    store._identity_fd = 37
    monkeypatch.setattr(quarantine.ReceiptQuarantineStore, "_current", lambda self: trace.append("current"))
    monkeypatch.setattr(quarantine, "_entry", lambda *args: None)
    return proxy, "_write_flags", lambda: store._publish("name", b"x")


def _read_consumer(monkeypatch, trace):
    real_read = quarantine._read
    info = SimpleNamespace(st_size=1)
    monkeypatch.setattr(quarantine, "_entry", lambda *args: info)
    monkeypatch.setattr(quarantine, "_private_file", lambda *args: None)
    monkeypatch.setattr(quarantine, "_stat_tuple", lambda value: (1,))
    chunks = iter((b"x", b""))
    proxy = SimpleNamespace(open=None, close=lambda fd: trace.append(("close", fd)),
                            fstat=lambda fd: info, read=lambda *args: next(chunks))
    monkeypatch.setattr(quarantine, "os", proxy)
    return proxy, "_leaf_flags", lambda: real_read(37, "name", 10)


@pytest.mark.parametrize("site", ["root", "walk", "child", "read", "sync", "publish", "verify"])
def test_consumer_captures_opener_before_flags(monkeypatch, site):
    trace = []
    proxy, helper, invoke = _consumer(monkeypatch, site, trace)

    def original(*args, **kwargs):
        trace.append(("open", args, kwargs))
        return 41

    def replacement(*args, **kwargs):
        pytest.fail("late opener lookup")

    def flags():
        trace.append("flags")
        proxy.open = replacement
        return 23

    proxy.open = original
    monkeypatch.setattr(storage, helper, flags)
    if site == "walk":
        # The next component must capture a new opener; reset at its pre-open stat.
        proxy.stat = lambda *args, **kwargs: setattr(proxy, "open", original) or SimpleNamespace(st_mode=0)
    invoke()
    opens = [event for event in trace if isinstance(event, tuple) and event[0] == "open"]
    assert opens == (
        [("open", ("/", 23), {}), ("open", ("a", 23), {"dir_fd": 41})]
        if site == "walk" else
        [("open", ("/", 23), {})] if site == "root" else
        [("open", ("name", 23, 0o600), {"dir_fd": 37})] if site == "publish" else
        [("open", ("name", 23), {"dir_fd": 37})]
    )
    assert trace.count("flags") == len(opens)


@pytest.mark.parametrize("site", ["root", "walk", "child", "read", "sync", "publish", "verify"])
@pytest.mark.parametrize("kind", [OSError, ValueError, TypeError, RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("point", ["flags", "open"])
def test_consumer_fault_translation_and_close(monkeypatch, site, kind, point):
    trace = []
    proxy, helper, invoke = _consumer(monkeypatch, site, trace)
    fault = kind("injected")
    count = 0

    def flags():
        nonlocal count
        count += 1
        if point == "flags" and (site != "walk" or count == 2):
            raise fault
        return 23

    def opener(*args, **kwargs):
        if site == "walk" and args[0] == "/":
            return 41
        raise fault

    proxy.open = opener
    monkeypatch.setattr(storage, helper, flags)
    translated = kind in {OSError, ValueError, TypeError} and site != "sync"
    expected = (storage.MemoryStorageError if site == "verify" else quarantine.QuarantineError) if translated else kind
    with pytest.raises(expected) as raised:
        invoke()
    if not translated:
        assert raised.value is fault
    assert [event for event in trace if isinstance(event, tuple) and event[0] == "close"] == (
        [("close", 41)] if site == "walk" else []
    )


@pytest.mark.parametrize("number,code", [(errno.ENOENT, "binding"), (errno.EACCES, "corrupt"), (errno.EIO, "io")])
def test_verify_errno_translation(monkeypatch, number, code):
    fault = OSError(number, "injected")
    monkeypatch.setattr(verifier, "os", SimpleNamespace(open=lambda *args, **kwargs: (_ for _ in ()).throw(fault)))
    with pytest.raises(storage.MemoryStorageError) as raised:
        verifier._open_target(37, "name")
    assert raised.value.code == code


@pytest.mark.parametrize("boundary", [None, "write", "leaf-fsync", "parent-fsync", "reread"])
def test_publish_effect_order(monkeypatch, boundary):
    trace = []
    proxy, helper, invoke = _consumer(monkeypatch, "publish", trace)
    monkeypatch.setattr(storage, helper, lambda: trace.append("flags") or 23)
    proxy.open = lambda *args, **kwargs: trace.append("open") or 41

    def effect(name):
        trace.append(name)
        if name == boundary:
            raise OSError("injected")

    proxy.write = lambda fd, data: effect("write") or 1
    proxy.fsync = lambda fd: effect("leaf-fsync" if fd == 41 else "parent-fsync")
    monkeypatch.setattr(quarantine, "_read", lambda *args: effect("reread") or (b"x", object()))
    if boundary is None:
        invoke()
    else:
        with pytest.raises(quarantine.QuarantineError):
            invoke()
    order = ["current", "flags", "open", "write", "leaf-fsync", "current", "parent-fsync", "current", "reread"]
    assert trace == (order if boundary is None else order[:order.index(boundary) + 1]) + [("close", 41)]


@pytest.mark.parametrize("kind", [OSError, ValueError, TypeError, RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("point", ["flags", "open"])
def test_memory_consumer_fault_and_duplicate_cleanup(monkeypatch, tmp_path, kind, point):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "MEMORY.md").write_bytes(b"x")
    fault = kind("injected")
    with storage.WorkspaceMemoryAuthority(tmp_path) as authority:
        snapshot = authority.snapshot_target("MEMORY.md")
        with authority.operation() as operation:
            trace = []
            original_close = storage._close
            original_open = os.open
            proxy = SimpleNamespace(**vars(os))

            def opener(*args, **kwargs):
                trace.append("open")
                raise fault

            def flags():
                trace.append("flags")
                proxy.open = original_open  # Captured consumer callable must remain in force.
                if point == "flags":
                    raise fault
                return 23

            proxy.open = opener
            monkeypatch.setattr(memory_stage, "os", proxy)
            # Parent snapshot uses the helper too; install the fault only after duplicate.
            original_dup = storage.BoundMemoryTarget.duplicate_dir_fd

            def duplicate(scope):
                fd = original_dup(scope)
                trace.clear()
                monkeypatch.setattr(storage, "_leaf_flags", flags)
                return fd

            monkeypatch.setattr(storage.BoundMemoryTarget, "duplicate_dir_fd", duplicate)
            monkeypatch.setattr(storage, "_close", lambda fd: trace.append(("close", fd)) or original_close(fd))
            expected = KeyboardInterrupt if kind is KeyboardInterrupt else memory_stage.MemoryRecoveryTransactionMemoryError
            with pytest.raises(expected) as raised:
                memory_stage._durably_verify_memory(operation, snapshot, b"x")
            if kind is KeyboardInterrupt:
                assert raised.value is fault
            assert trace[:1 if point == "flags" else 2] == (["flags"] if point == "flags" else ["flags", "open"])
            assert trace[-2] == ("close", None)
            assert isinstance(trace[-1][1], int)
