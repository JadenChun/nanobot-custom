"""Quarantine owns its workspace pin; lock identity never replaces admission."""

import os
from contextlib import contextmanager

import pytest

from nanobot.agent import memory_recovery_quarantine_storage as quarantine
from nanobot.agent import memory_recovery_storage as storage
from tests.agent.test_memory_recovery_quarantine_storage import RECEIPT, _layout


@contextmanager
def opened_store(workspace, binding, mode):
    if mode == "gate":
        with storage.open_workspace_memory_lock(workspace) as owner:
            yield owner, quarantine.ReceiptQuarantineStore(owner, binding)
    else:
        with storage.open_workspace_memory(workspace) as owner, owner.operation() as operation:
            yield owner, quarantine.ReceiptQuarantineStore.from_operation(owner, operation, binding)


@pytest.mark.parametrize("mode", ["gate", "operation"])
def test_store_owns_distinct_workspace_pin_and_closes_only_its_pin(tmp_path, mode):
    workspace, binding = _layout(tmp_path)
    with opened_store(workspace, binding, mode) as (owner, store):
        original_fd = owner._workspace_fd
        with store:
            owned_fd = store._workspace_fd()
            assert owned_fd != original_fd
            assert os.fstat(owned_fd).st_ino == os.fstat(original_fd).st_ino
            store.prepare(RECEIPT)
        with pytest.raises(OSError):
            os.fstat(owned_fd)
        assert os.fstat(original_fd).st_ino == workspace.stat().st_ino


@pytest.mark.parametrize("mode", ["gate", "operation"])
def test_reopened_foreign_workspace_is_rejected_before_quarantine_creation(
    tmp_path, monkeypatch, mode,
):
    workspace, binding = _layout(tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir(mode=0o700)
    original_open = quarantine._open_absolute
    opened = []

    def substitute(path):
        fd = original_open(foreign if path == workspace else path)
        if path == workspace:
            opened.append(fd)
        return fd

    with opened_store(workspace, binding, mode) as (owner, store):
        with monkeypatch.context() as patch:
            patch.setattr(quarantine, "_open_absolute", substitute)
            with pytest.raises(quarantine.QuarantineError):
                with store:
                    raise AssertionError("foreign workspace admitted")
        assert opened
        for fd in opened:
            with pytest.raises(OSError):
                os.fstat(fd)
        assert list(foreign.iterdir()) == []
        assert not (workspace / "memory" / ".recovery" / "receipt-quarantine").exists()
        assert os.fstat(owner._workspace_fd).st_ino == workspace.stat().st_ino


@pytest.mark.parametrize("mode", ["gate", "operation"])
def test_failed_enter_closes_owned_workspace_without_closing_owner(tmp_path, monkeypatch, mode):
    workspace, binding = _layout(tmp_path)
    original_open = quarantine._open_absolute
    opened = []

    def fail_sessions(path):
        if path == binding.sessions_root:
            raise OSError("sessions unavailable")
        fd = original_open(path)
        if path == workspace:
            opened.append(fd)
        return fd

    with opened_store(workspace, binding, mode) as (owner, store):
        with monkeypatch.context() as patch:
            patch.setattr(quarantine, "_open_absolute", fail_sessions)
            with pytest.raises(OSError, match="sessions unavailable"):
                store.__enter__()
        assert opened
        for fd in opened:
            with pytest.raises(OSError):
                os.fstat(fd)
        assert os.fstat(owner._workspace_fd).st_ino == workspace.stat().st_ino
