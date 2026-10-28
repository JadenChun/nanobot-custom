"""Durable process admission is released only by validated lifecycle evidence."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from nanobot.security import native_admission
from nanobot.security.native_admission import (
    HostIdentity,
    NativeAdmissionActiveError,
    NativeAdmissionLedger,
)


def _configure_identity(monkeypatch: pytest.MonkeyPatch) -> HostIdentity:
    identity = HostIdentity("host-a", "boot-a", 101)
    monkeypatch.setattr(native_admission, "current_host_identity", lambda: identity)
    monkeypatch.setattr(native_admission, "process_start_time", lambda _pid: 77)
    monkeypatch.setattr(native_admission, "process_pid_namespace", lambda _pid: 101)
    return identity


def _active(ledger: NativeAdmissionLedger):
    reservation = ledger.begin(managed=lambda: False)
    assert reservation is not None
    reservation.acknowledge(4321, 77)
    return reservation


def test_same_boot_restart_retains_unacknowledged_launch_as_unknown(tmp_path, monkeypatch):
    _configure_identity(monkeypatch)
    ledger = NativeAdmissionLedger(tmp_path / "admissions.json")
    reservation = ledger.begin(managed=lambda: False)
    assert reservation is not None

    # A new gateway process has no in-memory record that the launch is local.
    monkeypatch.setattr(native_admission, "_LOCAL_LAUNCHES", {})
    restarted = NativeAdmissionLedger(ledger.path)

    assert restarted.reconcile() is True
    payload = restarted.path.read_text(encoding="utf-8")
    assert f'"launch_id":"{reservation.launch_id}"' in payload
    assert '"state":"unknown"' in payload
    with pytest.raises(NativeAdmissionActiveError):
        with restarted.activation_guard(managed=lambda: False):
            pass


def test_dead_or_pid_reused_supervisor_stays_blocked_in_same_boot(tmp_path, monkeypatch):
    _configure_identity(monkeypatch)
    ledger = NativeAdmissionLedger(tmp_path / "admissions.json")
    reservation = _active(ledger)
    monkeypatch.setattr(native_admission, "process_start_time", lambda _pid: 78)

    assert ledger.reconcile() is True
    assert '"state":"unknown"' in ledger.path.read_text(encoding="utf-8")
    restarted = NativeAdmissionLedger(ledger.path)
    with pytest.raises(NativeAdmissionActiveError):
        with restarted.activation_guard(managed=lambda: False):
            pass
    assert reservation.launch_id in ledger.path.read_text(encoding="utf-8")


def test_verified_later_boot_releases_live_barrier_but_records_unknown_outcome(
    tmp_path,
    monkeypatch,
):
    _configure_identity(monkeypatch)
    ledger = NativeAdmissionLedger(tmp_path / "admissions.json")
    reservation = _active(ledger)
    monkeypatch.setattr(
        native_admission,
        "current_host_identity",
        lambda: HostIdentity("host-a", "boot-b", 101),
    )

    assert ledger.reconcile() is False
    payload = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert payload["reservations"] == []
    assert payload["outcomes"][-1]["launch_id"] == reservation.launch_id
    assert payload["outcomes"][-1]["result"] == "unknown_after_boot"


@pytest.mark.parametrize(
    "identity",
    [
        HostIdentity(None, None, None),
        HostIdentity("host-b", "boot-b", 101),
        HostIdentity("host-a", "boot-b", 202),
    ],
)
def test_unavailable_host_or_different_execution_namespace_is_not_reboot_proof(
    tmp_path,
    monkeypatch,
    identity,
):
    _configure_identity(monkeypatch)
    ledger = NativeAdmissionLedger(tmp_path / "admissions.json")
    reservation = _active(ledger)
    monkeypatch.setattr(native_admission, "current_host_identity", lambda: identity)

    assert ledger.reconcile() is True
    assert reservation.launch_id in ledger.path.read_text(encoding="utf-8")
    assert '"state":"unknown"' in ledger.path.read_text(encoding="utf-8")


def test_corrupt_receipt_fails_closed_instead_of_clearing_barrier(tmp_path, monkeypatch):
    _configure_identity(monkeypatch)
    ledger = NativeAdmissionLedger(tmp_path / "admissions.json")
    reservation = _active(ledger)
    ledger.receipt_dir.mkdir(mode=0o700)
    (ledger.receipt_dir / f"{reservation.launch_id}.receipt").write_text(
        '{"version":1,"result":"quiescent"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="receipt"):
        ledger.reconcile()
    assert reservation.launch_id in ledger.path.read_text(encoding="utf-8")


def test_first_activation_and_native_admission_share_a_fence(tmp_path, monkeypatch):
    _configure_identity(monkeypatch)
    import nanobot.security.folder_grants as folder_grants

    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    store = folder_grants.FolderGrantStore(
        agent_home=home,
        instance_root=tmp_path / "instance",
        path=tmp_path / "instance" / "webui" / "folder-grants.json",
    )
    writing_activation = Event()
    release_activation = Event()
    original_write = folder_grants._atomic_write_activation

    def pause_activation(path):
        writing_activation.set()
        assert release_activation.wait(timeout=5)
        original_write(path)

    monkeypatch.setattr(folder_grants, "_atomic_write_activation", pause_activation)
    with ThreadPoolExecutor(max_workers=2) as executor:
        activation = executor.submit(
            store.upsert,
            {"path": str(project), "file_access": "read_only", "commands": False},
            expected_revision=0,
        )
        assert writing_activation.wait(timeout=5)
        admission = executor.submit(store.view.try_begin_native_process)
        try:
            assert admission.done() is False
        finally:
            release_activation.set()
        assert activation.result(timeout=5).enabled is True
        assert admission.result(timeout=5) is None


def test_owned_launch_upgrades_legacy_ledger_without_adopting_old_rows(tmp_path, monkeypatch):
    _configure_identity(monkeypatch)
    ledger = NativeAdmissionLedger(tmp_path / "admissions.json")
    legacy = ledger.begin(managed=lambda: False)
    assert legacy is not None
    before = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert before["version"] == 1
    assert "owner" not in before["reservations"][0]

    owner_tag = {
        "kind": "task_once",
        "runtime_id": "a" * 32,
        "owner_id": "turn-1",
        "epoch": 3,
        "session_key": "websocket:chat-a",
        "store_identity": "store-a",
        "incarnation": "incarnation-a",
        "generation": 0,
        "task_input_event_id": "event-a",
    }
    permission_owner = SimpleNamespace()
    permission_reference = object()
    owned = ledger.begin_owned(
        owner_tag,
        permission_owner=permission_owner,
        permission_reference=permission_reference,
    )
    after = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert after["version"] == 2
    assert after["reservations"][0]["owner"] is None
    assert after["reservations"][1]["owner"] == owner_tag
    assert owned.owner == owner_tag
    assert ledger.owner_counts(owner_tag) == (1, 0)

    older_generation = {**owner_tag, "generation": 4}
    previous = ledger.begin_owned(
        older_generation,
        permission_owner=permission_owner,
        permission_reference=permission_reference,
    )
    assert ledger.session_owner_counts(session_key="websocket:chat-a") == (2, 0)
    assert ledger.session_owner_counts(
        session_key="websocket:chat-a", kind="chat_full",
    ) == (0, 0)

    owned.mark_unproven()
    assert ledger.owner_counts(owner_tag) == (0, 1)
    previous.cancel_before_launch()
    assert ledger.session_owner_counts(session_key="websocket:chat-a") == (0, 1)
