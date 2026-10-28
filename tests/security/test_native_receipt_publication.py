"""Quiescence receipts publish atomically from admission readers' perspective."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import stat
import sys
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from nanobot.agent.tools import native_process
from nanobot.security import native_admission
from nanobot.security.folder_grants import (
    FolderGrantStore,
    NativeProcessesActiveError,
    NativeProcessesUnknownError,
)
from nanobot.security.native_admission import (
    NativeAdmissionLedger,
    QuiescenceReceipt,
    current_host_identity,
    process_start_time,
    write_quiescence_receipt,
)


def _receipt(launch_id: str) -> QuiescenceReceipt:
    identity = current_host_identity()
    assert identity.trusted
    starttime = process_start_time(os.getpid())
    assert starttime is not None
    host_id = identity.host_id
    boot_id = identity.boot_id
    pid_namespace = identity.pid_namespace
    assert host_id is not None and boot_id is not None and pid_namespace is not None
    return {
        "version": 1,
        "launch_id": launch_id,
        "host_id": host_id,
        "boot_id": boot_id,
        "pid_namespace": pid_namespace,
        "supervisor_pid": os.getpid(),
        "supervisor_starttime": starttime,
        "result": "quiescent",
        "completed_ns": time.time_ns(),
    }


def test_receipt_is_private_single_link_and_cannot_be_replayed(tmp_path: Path) -> None:
    ledger = NativeAdmissionLedger(tmp_path / "admissions.json")
    launch_id = uuid.uuid4().hex
    receipt = _receipt(launch_id)

    write_quiescence_receipt(ledger.receipt_dir, receipt)
    target = ledger.receipt_dir / f"{launch_id}.receipt"
    original = target.read_bytes()
    info = target.stat()
    assert stat.S_IMODE(info.st_mode) == 0o600
    assert info.st_nlink == 1
    assert info.st_uid == os.geteuid()

    with pytest.raises(FileExistsError):
        write_quiescence_receipt(ledger.receipt_dir, receipt)

    assert target.read_bytes() == original
    assert target.stat().st_nlink == 1


def test_activation_waits_for_receipt_publication_and_consumes_valid_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "agent-home"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    reservation = store.view.try_begin_native_process()
    assert reservation is not None
    starttime = process_start_time(os.getpid())
    assert starttime is not None
    reservation.acknowledge(os.getpid(), starttime)
    receipt = _receipt(reservation.launch_id)
    ledger = reservation.ledger

    paused = Event()
    release = Event()
    activation_started = Event()
    real_fsync_directory = native_admission._fsync_directory

    def pause_receipt_directory(path: Path) -> None:
        if path == ledger.receipt_dir and not paused.is_set():
            paused.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test did not release receipt publication")
        real_fsync_directory(path)

    monkeypatch.setattr(native_admission, "_fsync_directory", pause_receipt_directory)

    def publish() -> None:
        write_quiescence_receipt(ledger.receipt_dir, receipt)

    def activate():
        activation_started.set()
        return store.upsert(
            {"path": str(project), "file_access": "read_only", "commands": False},
            expected_revision=0,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        publisher = executor.submit(publish)
        assert paused.wait(timeout=3)
        target = ledger.receipt_dir / f"{reservation.launch_id}.receipt"
        nlink_during_publication = target.stat().st_nlink
        activation = executor.submit(activate)
        try:
            assert activation_started.wait(timeout=3)
            time.sleep(0.05)
            activation_waited_for_publication = not activation.done()
        finally:
            release.set()
        publisher.result(timeout=5)
        result = activation.result(timeout=8)

    assert nlink_during_publication == 1
    assert activation_waited_for_publication is True
    assert result.enabled is True
    assert store.activation_path.exists()
    assert store.view.is_managed is True
    assert ledger.reconcile() is False


def test_receipt_lock_timeout_is_retryable_and_does_not_latch_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        native_admission,
        "_RECEIPT_PUBLICATION_LOCK_TIMEOUT_SECONDS",
        0.05,
        raising=False,
    )
    home = tmp_path / "agent-home"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    reservation = store.view.try_begin_native_process()
    assert reservation is not None
    starttime = process_start_time(os.getpid())
    assert starttime is not None
    reservation.acknowledge(os.getpid(), starttime)
    receipt = _receipt(reservation.launch_id)
    ledger = reservation.ledger

    paused = Event()
    release = Event()
    real_fsync_directory = native_admission._fsync_directory

    def pause_receipt_directory(path: Path) -> None:
        if path == ledger.receipt_dir and not paused.is_set():
            paused.set()
            if not release.wait(timeout=8):
                raise RuntimeError("test did not release receipt publication")
        real_fsync_directory(path)

    monkeypatch.setattr(native_admission, "_fsync_directory", pause_receipt_directory)

    def publish() -> None:
        write_quiescence_receipt(ledger.receipt_dir, receipt)

    with ThreadPoolExecutor(max_workers=1) as executor:
        publisher = executor.submit(publish)
        assert paused.wait(timeout=3)
        activation_error: Exception | None = None
        try:
            store.upsert(
                {"path": str(project), "file_access": "read_only", "commands": False},
                expected_revision=0,
            )
        except Exception as exc:
            activation_error = exc
        finally:
            release.set()
        publisher.result(timeout=5)

    assert isinstance(activation_error, NativeProcessesActiveError)
    assert store.view.is_managed is False
    assert store.activation_path.exists() is False
    assert ledger._read_receipt(reservation.launch_id) is not None

    result = store.upsert(
        {"path": str(project), "file_access": "read_only", "commands": False},
        expected_revision=0,
    )
    assert result.enabled is True
    assert store.activation_path.exists()
    assert store.view.is_managed is True
    assert ledger.reconcile() is False


@pytest.mark.parametrize("invalid_kind", ["corrupt", "hardlink", "permissive-mode"])
def test_invalid_receipt_evidence_keeps_admission_blocked(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    ledger = NativeAdmissionLedger(tmp_path / "admissions.json")
    reservation = ledger.begin(managed=lambda: False)
    assert reservation is not None
    reservation.acknowledge(os.getpid(), process_start_time(os.getpid()) or 0)
    ledger.receipt_dir.mkdir(mode=0o700)
    target = ledger.receipt_dir / f"{reservation.launch_id}.receipt"

    if invalid_kind == "corrupt":
        target.write_text('{"version":1}\n', encoding="utf-8")
        os.chmod(target, 0o600)
    else:
        write_quiescence_receipt(ledger.receipt_dir, _receipt(reservation.launch_id))
        if invalid_kind == "hardlink":
            os.link(target, ledger.receipt_dir / "evidence-copy")
        else:
            os.chmod(target, 0o644)

    with pytest.raises(ValueError, match="receipt|singly linked"):
        ledger.reconcile()
    assert reservation.launch_id in ledger.path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_interrupted_receipt_write_stays_unknown_and_later_boot_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform != "linux":
        pytest.skip("native subreaper receipt publication is Linux-only")

    home = tmp_path / "agent-home"
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    home.mkdir()
    project.mkdir()
    store = FolderGrantStore(
        agent_home=home,
        instance_root=instance,
        path=instance / "webui" / "folder-grants.json",
    )
    reservation = store.view.try_begin_native_process()
    assert reservation is not None
    ledger = reservation.ledger

    marker = tmp_path / "publisher-paused-before-receipt-write"
    bootstrap = tmp_path / "helper-bootstrap.py"
    bootstrap.write_text(
        "import os,runpy,sys,time\n"
        "from pathlib import Path\n"
        "from nanobot.security import native_admission as admission\n"
        "real_fchmod=admission.os.fchmod\n"
        f"marker=Path({str(marker)!r})\n"
        "def pause_fchmod(fd,mode):\n"
        " real_fchmod(fd,mode)\n"
        " marker.write_text(str(os.getpid()),encoding='ascii')\n"
        " time.sleep(30)\n"
        "admission.os.fchmod=pause_fchmod\n"
        "sys.argv=[sys.argv[3]]+sys.argv[4:]\n"
        "runpy.run_module('nanobot.agent.tools.native_process_supervisor',run_name='__main__')\n",
        encoding="utf-8",
    )
    helper_launcher = tmp_path / "qa-frozen-helper"
    helper_launcher.write_text(
        "#!/bin/sh\n"
        f"exec {shlex.quote(sys.executable)} -I -B {shlex.quote(str(bootstrap))} \"$@\"\n",
        encoding="utf-8",
    )
    helper_launcher.chmod(0o700)
    monkeypatch.setattr(
        native_process,
        "sys",
        types.SimpleNamespace(platform=sys.platform, executable=str(helper_launcher)),
    )

    handle = await native_process.spawn_native_process(
        command=":",
        cwd=str(home),
        env=dict(os.environ),
        shell_program="/bin/bash",
        login=False,
        reservation=reservation,
    )
    try:
        assert await asyncio.wait_for(handle.wait(), timeout=3) == 0
        for _attempt in range(300):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        assert marker.exists()
        handle._supervisor.kill()
        await asyncio.wait_for(handle._supervisor.wait(), timeout=3)

        target = ledger.receipt_dir / f"{reservation.launch_id}.receipt"
        assert target.exists() is False
        [staged] = ledger.receipt_dir.iterdir()
        staged_info = staged.lstat()
        assert staged.name.startswith(f".{reservation.launch_id}.receipt.")
        assert staged.name.endswith(".tmp")
        assert stat.S_ISREG(staged_info.st_mode)
        assert staged_info.st_nlink == 1
        assert stat.S_IMODE(staged_info.st_mode) == 0o600
        assert staged_info.st_uid == os.geteuid()
        assert ledger.reconcile() is True
        assert list(ledger.receipt_dir.iterdir()) == []
        record = json.loads(ledger.path.read_text(encoding="utf-8"))["reservations"][0]
        assert record["state"] == "unknown"
        with pytest.raises(NativeProcessesUnknownError, match="native_processes_unknown"):
            store.upsert(
                {"path": str(project), "file_access": "read_only", "commands": False},
                expected_revision=0,
            )
        assert store.view.is_managed is False
        assert store.activation_path.exists() is False

        previous = current_host_identity()
        assert previous.host_id is not None and previous.boot_id is not None
        assert previous.pid_namespace is not None
        monkeypatch.setattr(
            native_admission,
            "current_host_identity",
            lambda: native_admission.HostIdentity(
                previous.host_id,
                "probe-next-boot",
                previous.pid_namespace,
            ),
        )
        assert ledger.reconcile() is False
        state = json.loads(ledger.path.read_text(encoding="utf-8"))
        assert state["reservations"] == []
        assert state["outcomes"][-1]["result"] == "unknown_after_boot"

        result = store.upsert(
            {"path": str(project), "file_access": "read_only", "commands": False},
            expected_revision=0,
        )
        assert result.enabled is True
        assert store.activation_path.exists()
    finally:
        if handle._supervisor.returncode is None:
            handle._supervisor.kill()
            await asyncio.wait_for(handle._supervisor.wait(), timeout=3)
        for stream in (handle.stdout, handle.stderr):
            if stream is not None:
                await asyncio.wait_for(stream.read(), timeout=3)
        await handle.close_control()
