"""Darwin process primitives and native-admission identity (A3-2M, WP2).

Read-only: nothing here sends a signal. The self-test launches two short
sandboxed sleeps that exit by themselves.
"""

from __future__ import annotations

import errno
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from kohaku.security import darwin_proc as darwin
from kohaku.security import native_admission as admission

on_darwin = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")


@pytest.fixture(autouse=True)
def _fresh_sandbox_check_cache():
    darwin._sandbox_check_function.cache_clear()
    yield
    darwin._sandbox_check_function.cache_clear()


# --- Cross-platform logic -------------------------------------------------


@pytest.mark.parametrize("bad", ["", "abc", "G" * 32, "A" * 32, "0" * 31, "0" * 33])
def test_launch_ids_are_strictly_validated(bad) -> None:
    with pytest.raises(ValueError):
        darwin.member_name(bad)
    with pytest.raises(ValueError):
        darwin.marker_rules(bad)


def test_marker_rules_allow_member_and_deny_fence() -> None:
    launch_id = darwin.new_launch_id()

    rules = darwin.marker_rules(launch_id).splitlines()

    assert rules == [
        f'(allow mach-lookup (global-name "dev.kohaku.launch.{launch_id}.member"))',
        f'(deny mach-lookup (global-name "dev.kohaku.launch.{launch_id}.fence"))',
    ]
    assert darwin.member_name(launch_id) != darwin.fence_name(launch_id)


def test_new_launch_ids_are_unique_and_valid() -> None:
    ids = {darwin.new_launch_id() for _ in range(50)}

    assert len(ids) == 50
    for launch_id in ids:
        darwin.member_name(launch_id)


def test_unqualified_abi_fails_closed_without_launching(monkeypatch) -> None:
    monkeypatch.setattr(darwin, "abi_supported", lambda: False)
    launched: list[object] = []
    monkeypatch.setattr(darwin.subprocess, "Popen", lambda *a, **k: launched.append(a))

    result = darwin.seatbelt_self_test()

    assert result.ok is False
    assert result.failures
    assert launched == []
    with pytest.raises(darwin.DarwinUnavailableError):
        darwin.is_sandboxed(os.getpid())


# --- macOS primitives -----------------------------------------------------


@on_darwin
def test_list_pids_contains_this_process_and_its_parent() -> None:
    pids = darwin.list_pids()

    assert os.getpid() in pids
    assert os.getppid() in pids


@on_darwin
def test_process_info_reports_identity_and_a_stable_start_stamp() -> None:
    first = darwin.process_info(os.getpid())
    second = darwin.process_info(os.getpid())

    assert first is not None and second is not None
    assert first.uid == os.geteuid()
    assert first.ppid == os.getppid()
    assert first.zombie is False
    assert first.start_us == second.start_us
    assert abs(first.start_us / 1_000_000 - time.time()) < 365 * 86400


@on_darwin
def test_process_info_distinguishes_different_processes_and_missing_pids() -> None:
    child = subprocess.Popen(["/bin/sleep", "1"])
    try:
        mine = darwin.process_info(os.getpid())
        theirs = darwin.process_info(child.pid)
        assert mine is not None and theirs is not None
        assert theirs.ppid == os.getpid()
        assert theirs.start_us >= mine.start_us
    finally:
        child.wait()
    assert darwin.process_info(2**22 - 1) is None
    assert darwin.process_start_time(2**22 - 1) is None


@on_darwin
def test_boot_and_host_identity_match_the_system() -> None:
    expected = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    assert darwin.boot_session_uuid() == expected
    assert str(uuid.UUID(darwin.host_uuid())) == darwin.host_uuid()


@on_darwin
def test_unsandboxed_supervisor_is_never_a_member() -> None:
    assert darwin.is_sandboxed(os.getpid()) is False
    assert darwin.is_member(os.getpid(), darwin.new_launch_id()) is False


@on_darwin
def test_whole_table_scan_with_fresh_ids_matches_nothing() -> None:
    # The zero-false-positive guard for the incident that killed unrelated
    # user processes: a launch that never existed must match nobody.
    for _ in range(3):
        assert darwin.scan_members(darwin.new_launch_id()) == []


@on_darwin
def test_rename_excl_publishes_once_and_never_replaces(tmp_path: Path) -> None:
    first, second, target = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    first.write_text("one")
    second.write_text("two")

    darwin.rename_excl(first, target)

    with pytest.raises(OSError) as raised:
        darwin.rename_excl(second, target)
    assert raised.value.errno == errno.EEXIST
    assert target.read_text() == "one"
    assert second.read_text() == "two"
    assert not first.exists()


@on_darwin
def test_self_test_passes_on_this_host() -> None:
    result = darwin.seatbelt_self_test()

    assert result.ok, result.failures
    assert result.failures == ()


@on_darwin
def test_self_test_detects_a_predicate_that_matches_everything(monkeypatch) -> None:
    monkeypatch.setattr(darwin, "_CHILD_SECONDS", 2)
    monkeypatch.setattr(darwin, "is_member", lambda pid, launch_id: True)

    result = darwin.seatbelt_self_test()

    assert result.ok is False
    assert any("foreign" in failure or "supervisor" in failure for failure in result.failures)


@on_darwin
def test_self_test_detects_a_predicate_that_matches_nothing(monkeypatch) -> None:
    monkeypatch.setattr(darwin, "_CHILD_SECONDS", 2)
    monkeypatch.setattr(darwin, "_ESTABLISH_SECONDS", 0.3)
    monkeypatch.setattr(darwin, "is_member", lambda pid, launch_id: False)

    result = darwin.seatbelt_self_test()

    assert result.ok is False
    assert any("never became a member" in failure for failure in result.failures)


# --- native admission on macOS -------------------------------------------


@on_darwin
def test_host_identity_is_trusted_with_one_pid_namespace() -> None:
    identity = admission.current_host_identity()

    assert identity.trusted is True
    assert identity.pid_namespace == 1
    assert identity.host_id == darwin.host_uuid()
    assert identity.boot_id == darwin.boot_session_uuid()


@on_darwin
def test_start_time_and_namespace_follow_pid_liveness() -> None:
    assert admission.process_start_time(os.getpid()) == darwin.process_start_time(os.getpid())
    assert admission.process_pid_namespace(os.getpid()) == 1
    assert admission.process_start_time(2**22 - 1) is None
    assert admission.process_pid_namespace(2**22 - 1) is None


@on_darwin
def test_later_boot_is_verified_only_for_the_same_host() -> None:
    identity = admission.current_host_identity()
    record = {
        "host_id": identity.host_id,
        "boot_id": "00000000-0000-0000-0000-000000000000",
        "pid_namespace": 1,
    }

    assert admission._verified_later_boot(record, identity) is True  # type: ignore[arg-type]
    assert admission._verified_later_boot(  # type: ignore[arg-type]
        {**record, "host_id": "another-host"}, identity,
    ) is False
    assert admission._verified_later_boot(  # type: ignore[arg-type]
        {**record, "boot_id": identity.boot_id}, identity,
    ) is False


@on_darwin
def test_receipts_validate_and_are_never_replaced(tmp_path: Path) -> None:
    identity = admission.current_host_identity()
    assert identity.host_id and identity.boot_id and identity.pid_namespace
    start = admission.process_start_time(os.getpid())
    assert start is not None
    receipt: admission.QuiescenceReceipt = {
        "version": 1,
        "launch_id": uuid.uuid4().hex,
        "host_id": identity.host_id,
        "boot_id": identity.boot_id,
        "pid_namespace": identity.pid_namespace,
        "supervisor_pid": os.getpid(),
        "supervisor_starttime": start,
        "result": "quiescent",
        "completed_ns": time.time_ns(),
    }
    directory = tmp_path / "receipts"

    admission.write_quiescence_receipt(directory, receipt)

    written = directory / f"{receipt['launch_id']}.receipt"
    assert written.is_file()
    assert oct(written.stat().st_mode & 0o777) == "0o600"
    with pytest.raises(OSError):
        admission.write_quiescence_receipt(directory, {**receipt, "completed_ns": 1})


# --- Membership verdicts and ancestry (used by the launch domain) -------------


@pytest.mark.parametrize(
    ("sandboxed", "denies_member", "denies_fence", "expected"),
    [
        (False, None, None, False),
        (None, False, True, None),
        (True, True, True, False),
        (True, False, False, False),
        (True, False, True, True),
        (True, None, True, None),
        (True, False, None, None),
    ],
)
def test_membership_separates_unknown_from_not_a_member(
    monkeypatch, sandboxed, denies_member, denies_fence, expected,
) -> None:
    verdicts = {
        darwin.member_name("a" * 32): denies_member,
        darwin.fence_name("a" * 32): denies_fence,
    }
    monkeypatch.setattr(darwin, "is_sandboxed", lambda pid: sandboxed)
    monkeypatch.setattr(darwin, "denies_global_name", lambda pid, name: verdicts[name])

    assert darwin.membership(10, "a" * 32) is expected
    assert darwin.is_member(10, "a" * 32) is (expected is True)


@on_darwin
def test_parent_pid_covers_processes_of_other_users() -> None:
    assert darwin.parent_pid(os.getpid()) == os.getppid()
    assert darwin.parent_pid(1) == 0
    assert darwin.parent_pid(99_999_999) is None
