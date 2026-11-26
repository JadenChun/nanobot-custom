"""Launch-domain tracking engine (A3-2M, WP3).

Every signalling test here runs against a fake process table and a fake
signaller. The autouse fixture makes any real signal fail the test. The one
real-host test only observes: its processes exit by themselves.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from kohaku.agent.tools import seatbelt_tracking as tracking
from kohaku.security import darwin_proc as darwin

LAUNCH = "a" * 32
OTHER_LAUNCH = "b" * 32
UID = 501
SUPERVISOR = 300
GATEWAY = 200
SHELL = 100
FLOOR = 1_000


@pytest.fixture(autouse=True)
def _no_real_signals(monkeypatch):
    def forbidden(*args: object) -> None:
        raise AssertionError(f"a unit test tried to send a real signal: {args}")

    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(os, "killpg", forbidden, raising=False)


@dataclass
class Proc:
    pid: int
    ppid: int = 1
    uid: int = UID
    start_us: int = FLOOR + 1
    launch: str | None = None
    unknown: bool = False
    hidden: bool = False


class FakeTable:
    def __init__(self, *procs: Proc) -> None:
        self.procs = {
            proc.pid: proc
            for proc in (
                Proc(SHELL, ppid=1, start_us=1),
                Proc(GATEWAY, ppid=SHELL, start_us=2),
                Proc(SUPERVISOR, ppid=GATEWAY, start_us=FLOOR),
                *procs,
            )
        }
        self.after_list: Callable[[], None] | None = None
        self.listings = 0
        self.broken = False

    def add(self, proc: Proc) -> None:
        self.procs[proc.pid] = proc

    def remove(self, pid: int) -> None:
        self.procs.pop(pid, None)

    def pids(self) -> list[int]:
        if self.broken:
            raise darwin.DarwinUnavailableError("scan failed")
        self.listings += 1
        listed = sorted(self.procs)
        if self.after_list is not None:
            self.after_list()
        return listed

    def membership(self, pid: int, launch_id: str) -> bool | None:
        proc = self.procs.get(pid)
        if proc is None:
            return False
        if proc.unknown:
            return None
        return proc.launch == launch_id

    def info(self, pid: int) -> darwin.ProcessInfo | None:
        proc = self.procs.get(pid)
        if proc is None or proc.hidden:
            return None
        return darwin.ProcessInfo(proc.pid, proc.ppid, proc.uid, 2, proc.start_us)

    def parent_pid(self, pid: int) -> int | None:
        proc = self.procs.get(pid)
        return None if proc is None else proc.ppid


class FakeSignaller:
    def __init__(self, table: FakeTable) -> None:
        self.table = table
        self.killed: list[int] = []
        self.denied: set[int] = set()

    def kill(self, pid: int) -> None:
        if pid in self.denied:
            raise PermissionError(pid)
        if pid not in self.table.procs:
            raise ProcessLookupError(pid)
        self.killed.append(pid)
        self.table.remove(pid)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.on_sleep: Callable[[], None] | None = None

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.on_sleep is not None:
            self.on_sleep()


def member(pid: int, **fields: object) -> Proc:
    return Proc(pid, launch=LAUNCH, **fields)  # type: ignore[arg-type]


def build(
    table: FakeTable,
    *,
    root_exited: Callable[[], bool] = lambda: True,
    protected: tuple[int, ...] = (),
) -> tuple[tracking.LaunchDomain, FakeSignaller, FakeClock]:
    signaller = FakeSignaller(table)
    clock = FakeClock()
    owner = table.info(SUPERVISOR)
    assert owner is not None
    domain = tracking.LaunchDomain(
        LAUNCH, table=table, signaller=signaller, owner=owner,
        root_exited=root_exited, protected=protected,
        clock=clock, sleep=clock.sleep,
    )
    return domain, signaller, clock


def test_profile_markers_belong_to_this_launch() -> None:
    domain, _, _ = build(FakeTable())

    assert domain.profile_markers() == darwin.marker_rules(LAUNCH)


def test_an_invalid_launch_id_is_refused() -> None:
    table = FakeTable()
    owner = table.info(SUPERVISOR)
    assert owner is not None
    with pytest.raises(ValueError):
        tracking.LaunchDomain(
            "nope", table=table, signaller=FakeSignaller(table), owner=owner,
            root_exited=lambda: True,
        )


def test_only_members_of_this_launch_are_signalled() -> None:
    table = FakeTable(
        member(400), member(401),
        Proc(500), Proc(501, launch=OTHER_LAUNCH),
    )
    domain, signaller, _ = build(table)

    result = domain.kill_until_quiescent(deadline=5.0)

    assert isinstance(result, tracking.Quiescent)
    assert signaller.killed == [400, 401]
    assert {500, 501} <= set(table.procs)


@pytest.mark.parametrize(
    ("proc", "reason"),
    [
        (member(400, uid=0), "foreign_uid_member"),
        (member(400, hidden=True), "unidentified_member"),
        (member(400, start_us=FLOOR - 1), "predates_launch"),
        (Proc(400, unknown=True), "unknown_membership"),
    ],
)
def test_unsafe_candidates_are_never_signalled_and_block_the_proof(proc, reason) -> None:
    table = FakeTable(proc, member(401))
    domain, signaller, _ = build(table)

    result = domain.kill_until_quiescent(deadline=1.0)

    assert result == tracking.NotQuiescent(reason, (400,))
    assert signaller.killed == [401]


def test_supervisor_ancestors_and_pid_one_are_protected_from_a_wrong_predicate() -> None:
    table = FakeTable(Proc(1, ppid=0, start_us=FLOOR + 1), Proc(600, ppid=SHELL))
    for proc in table.procs.values():
        proc.launch = LAUNCH
        proc.start_us = max(proc.start_us, FLOOR)
    table.procs[SUPERVISOR].start_us = FLOOR
    domain, signaller, _ = build(table)

    result = domain.kill_until_quiescent(deadline=1.0)

    assert isinstance(result, tracking.NotQuiescent)
    assert result.reason == "protected_member"
    assert set(result.pids) == {1, SHELL, GATEWAY, SUPERVISOR}
    assert signaller.killed == [600]


def test_explicitly_protected_processes_and_their_ancestors_are_not_signalled() -> None:
    table = FakeTable(member(700, ppid=1), member(701, ppid=700), member(702))
    domain, signaller, _ = build(table, protected=(701,))

    domain.kill_until_quiescent(deadline=1.0)

    assert signaller.killed == [702]


def test_former_ancestors_stay_protected_after_reparenting() -> None:
    table = FakeTable()
    domain, signaller, _ = build(table)
    table.procs[SUPERVISOR].ppid = 1
    table.procs[GATEWAY].launch = LAUNCH
    table.procs[GATEWAY].start_us = FLOOR + 5

    domain.kill_until_quiescent(deadline=1.0)

    assert signaller.killed == []


def test_unreadable_ancestry_signals_nothing() -> None:
    table = FakeTable(member(400))
    domain, signaller, _ = build(table)
    table.remove(GATEWAY)

    result = domain.kill_until_quiescent(deadline=1.0)

    assert result == tracking.NotQuiescent("ancestry_unavailable", (400,))
    assert signaller.killed == []


def test_a_reused_pid_is_skipped() -> None:
    table = FakeTable(member(400))
    domain, signaller, _ = build(table)
    original = table.membership

    def reuse_after_check(pid: int, launch_id: str) -> bool | None:
        verdict = original(pid, launch_id)
        if pid == 400 and table.listings:
            table.procs[400].start_us += 1
        return verdict

    table.membership = reuse_after_check  # type: ignore[method-assign]

    domain.kill_pass()

    assert signaller.killed == []


def test_membership_is_rechecked_immediately_before_signalling() -> None:
    table = FakeTable(member(400))
    domain, signaller, _ = build(table)
    answers = iter([True, False])
    original = table.membership
    table.membership = (  # type: ignore[method-assign]
        lambda pid, launch_id: next(answers) if pid == 400 else original(pid, launch_id)
    )

    domain.kill_pass()

    assert signaller.killed == []


def test_a_member_that_cannot_be_signalled_does_not_stop_the_others() -> None:
    table = FakeTable(member(400), member(401))
    domain, signaller, _ = build(table)
    signaller.denied.add(400)

    result = domain.kill_until_quiescent(deadline=1.0)

    assert result == tracking.NotQuiescent("unsignallable_member", (400,))
    assert signaller.killed == [401]


def test_children_forked_during_the_kill_loop_are_also_killed() -> None:
    table = FakeTable(member(400))
    domain, signaller, _ = build(table)
    original = signaller.kill

    def fork_then_die(pid: int) -> None:
        if pid < 420:
            table.add(member(pid + 1))
        original(pid)

    signaller.kill = fork_then_die  # type: ignore[method-assign]

    result = domain.kill_until_quiescent(deadline=10.0)

    assert isinstance(result, tracking.Quiescent)
    assert signaller.killed == list(range(400, 421))


def test_a_process_that_forks_and_exits_during_a_scan_is_still_found() -> None:
    table = FakeTable(member(400))
    domain, _, _ = build(table)

    def hop_once() -> None:
        table.after_list = None
        table.remove(400)
        table.add(member(401))

    table.after_list = hop_once

    assert domain.scan().members == (401,)


def test_a_table_that_never_settles_is_not_empty() -> None:
    table = FakeTable()
    domain, _, _ = build(table)
    table.after_list = lambda: table.add(Proc(max(table.procs) + 1))

    scan = domain.scan()

    assert scan.complete is False
    assert scan.empty is False


def test_quiescence_needs_two_empty_scans_a_settle_interval_apart() -> None:
    table = FakeTable()
    domain, _, clock = build(table)

    assert domain.poll() is False
    listings = table.listings
    assert domain.poll() is False
    assert table.listings == listings
    clock.now += tracking.SETTLE_SECONDS
    assert domain.poll() is True


def test_a_member_between_empty_scans_restarts_the_count() -> None:
    table = FakeTable()
    domain, _, clock = build(table)

    assert domain.poll() is False
    table.add(member(400))
    clock.now += tracking.SETTLE_SECONDS
    assert domain.poll() is False
    table.remove(400)
    clock.now += tracking.SETTLE_SECONDS
    assert domain.poll() is False
    clock.now += tracking.SETTLE_SECONDS
    assert domain.poll() is True


def test_an_empty_domain_is_not_quiescent_while_the_root_runs() -> None:
    domain, _, _ = build(FakeTable(), root_exited=lambda: False)

    assert domain.wait_quiescent(deadline=1.0) is False
    assert domain.kill_until_quiescent(deadline=2.0) == tracking.NotQuiescent("root_running", ())


def test_waiting_never_signals_and_returns_only_after_the_tree_exits() -> None:
    table = FakeTable(member(400))
    domain, signaller, clock = build(table)

    assert domain.wait_quiescent(deadline=1.0) is False

    clock.on_sleep = lambda: table.remove(400) if clock.now > 2.0 else None
    assert domain.wait_quiescent(deadline=5.0) is True
    assert clock.now > 2.0
    assert signaller.killed == []


def test_waiting_on_a_long_lived_tree_backs_off_between_scans() -> None:
    table = FakeTable(member(400))
    domain, _, clock = build(table)

    assert domain.wait_quiescent(deadline=60.0) is False
    assert table.listings <= 150

    table.remove(400)
    started = clock.now
    assert domain.wait_quiescent(deadline=clock.now + 5.0) is True
    assert clock.now - started <= 1.0


def test_killing_is_not_delayed_by_an_earlier_backoff() -> None:
    table = FakeTable(member(400))
    domain, signaller, clock = build(table)
    assert domain.wait_quiescent(deadline=60.0) is False
    started = clock.now

    assert isinstance(domain.kill_until_quiescent(clock.now + 5.0), tracking.Quiescent)
    assert signaller.killed == [400]
    assert clock.now - started <= 0.2


def test_a_failing_process_table_is_never_proof() -> None:
    table = FakeTable()
    domain, signaller, _ = build(table)
    table.broken = True

    assert domain.wait_quiescent(deadline=1.0) is False
    assert domain.kill_until_quiescent(deadline=2.0) == tracking.NotQuiescent(
        "scan_unavailable", (),
    )
    assert signaller.killed == []


@pytest.mark.parametrize("pid", [-1, 0, 1])
def test_the_host_signaller_refuses_groups_and_pid_one(pid, monkeypatch) -> None:
    sent: list[object] = []
    monkeypatch.setattr(os, "kill", lambda *args: sent.append(args))

    with pytest.raises(ValueError):
        tracking.HostSignaller().kill(pid)
    assert sent == []


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="POSIX only")
def test_the_host_signaller_sends_sigkill_to_one_process(monkeypatch) -> None:
    sent: list[object] = []
    monkeypatch.setattr(os, "kill", lambda *args: sent.append(args))

    tracking.HostSignaller().kill(4242)

    assert sent == [(4242, signal.SIGKILL)]


class _RefusingSignaller:
    def kill(self, pid: int) -> None:
        raise AssertionError(f"observation must not signal {pid}")


_ESCAPEE = """
import os, sys, time
if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        with open(sys.argv[1], "w") as handle:
            handle.write(str(os.getpid()))
        time.sleep(3)
    os._exit(0)
"""


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_the_host_domain_of_an_unused_launch_id_is_quiescent() -> None:
    domain = tracking.host_launch_domain(darwin.new_launch_id(), root_exited=lambda: True)

    assert domain.wait_quiescent(time.monotonic() + 10) is True


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_an_escaped_descendant_holds_the_domain_open_until_it_exits(tmp_path: Path) -> None:
    launch_id = darwin.new_launch_id()
    profile = tmp_path / "launch.sb"
    profile.write_text(
        "(version 1)\n(allow default)\n" + darwin.marker_rules(launch_id), encoding="ascii",
    )
    pid_file = tmp_path / "escapee.pid"
    table = tracking.HostProcTable()
    owner = table.info(os.getpid())
    assert owner is not None
    root = subprocess.Popen(
        [darwin.SANDBOX_EXEC, "-f", str(profile), sys.executable, "-c", _ESCAPEE, str(pid_file)],
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    domain = tracking.LaunchDomain(
        launch_id, table=table, signaller=_RefusingSignaller(), owner=owner,
        root_exited=lambda: root.poll() is not None,
    )
    assert root.wait(timeout=10) == 0
    started = time.monotonic()
    while not (pid_file.exists() and pid_file.read_text()):
        assert time.monotonic() - started < 5
        time.sleep(0.02)
    escapee = int(pid_file.read_text())

    assert escapee in domain.scan().members
    assert domain.wait_quiescent(time.monotonic() + 0.3) is False
    assert domain.wait_quiescent(time.monotonic() + 10) is True
    assert table.membership(escapee, launch_id) is False
