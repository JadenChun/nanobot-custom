"""Track one macOS launch tree through its Seatbelt markers (A3-2M, WP3).

A Seatbelt profile is inherited across fork and exec and cannot be removed, so
every descendant of a marked launch stays identifiable after it leaves the
process group or session, or is reparented to launchd.

Proof. The tree is quiescent when, after the root has exited, two complete scans
a settle interval apart find no member. A scan is complete only when a fresh
listing holds no pid it has not already checked, so a process that forks and
exits while being listed cannot slip between the listing and the check.

Signalling invariants. A mistaken membership test once killed unrelated
processes, so a signal needs all of these:

* K1: membership re-checked just before the signal, same effective uid, and not
  pid 0 or 1, the supervisor, a protected process, or an ancestor of those.
* K2: the same start stamp before the membership check and after it.
* K3: a bounded loop; anything that cannot be signalled or identified leaves the
  result not quiescent.
* K4: the process table and the signaller are injected, so tests use fakes.
* K5: no process older than the supervisor, which cannot be its descendant.
"""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from kohaku.security import darwin_proc

SETTLE_SECONDS = 0.05
_INTERVAL_SECONDS = 0.02
# A scan costs milliseconds, so waiting on a long-lived tree must slow down.
_MAX_BACKOFF_SECONDS = 0.5
_SCAN_ROUNDS = 64
_TABLE_ERRORS = (OSError, darwin_proc.DarwinUnavailableError)

BlockReason = Literal[
    "ancestry_unavailable",
    "protected_member",
    "unidentified_member",
    "unknown_membership",
    "foreign_uid_member",
    "predates_launch",
    "unsignallable_member",
]
NotQuiescentReason = BlockReason | Literal["scan_unavailable", "root_running", "deadline"]


class ProcTable(Protocol):
    def pids(self) -> list[int]: ...

    def membership(self, pid: int, launch_id: str) -> bool | None: ...

    def info(self, pid: int) -> darwin_proc.ProcessInfo | None: ...

    def parent_pid(self, pid: int) -> int | None: ...


class Signaller(Protocol):
    def kill(self, pid: int) -> None: ...


class HostProcTable:
    """The live process table of this host."""

    def pids(self) -> list[int]:
        return darwin_proc.list_pids()

    def membership(self, pid: int, launch_id: str) -> bool | None:
        return darwin_proc.membership(pid, launch_id)

    def info(self, pid: int) -> darwin_proc.ProcessInfo | None:
        return darwin_proc.process_info(pid)

    def parent_pid(self, pid: int) -> int | None:
        return darwin_proc.parent_pid(pid)


class HostSignaller:
    """Send SIGKILL to exactly one process, never to a group."""

    def kill(self, pid: int) -> None:
        if pid <= 1:
            raise ValueError("refusing to signal a process group or pid 1")
        os.kill(pid, signal.SIGKILL)


@dataclass(frozen=True)
class DomainScan:
    members: tuple[int, ...] = ()
    unknown: tuple[int, ...] = ()
    complete: bool = False

    @property
    def empty(self) -> bool:
        return self.complete and not self.members and not self.unknown


@dataclass(frozen=True)
class KillPass:
    scan: DomainScan
    blocked: tuple[tuple[int, BlockReason], ...] = ()


@dataclass(frozen=True)
class Quiescent:
    pass


@dataclass(frozen=True)
class NotQuiescent:
    reason: NotQuiescentReason
    pids: tuple[int, ...]


class LaunchDomain:
    """Every process carrying one launch's markers.

    owner is the supervising process: it supplies the uid that members must have
    and the start stamp that no member can precede. root_exited must report True
    only once the launched root is gone, because the root is not yet a member
    while sandbox-exec is still applying its profile.

    Deadlines are absolute values of clock.
    """

    def __init__(
        self,
        launch_id: str,
        *,
        table: ProcTable,
        signaller: Signaller,
        owner: darwin_proc.ProcessInfo,
        root_exited: Callable[[], bool],
        protected: Iterable[int] = (),
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._markers = darwin_proc.marker_rules(launch_id)
        self.launch_id = launch_id
        self._table = table
        self._signaller = signaller
        self._owner = owner
        self._root_exited = root_exited
        self._extra = tuple(protected)
        self._clock = clock
        self._sleep = sleep
        self._first_empty: float | None = None
        self._backoff = 0.0
        self._next_poll = 0.0
        # Remembered so that reparenting never removes protection.
        self._ancestors: frozenset[int] = frozenset()
        self._protected_pids()

    def profile_markers(self) -> str:
        return self._markers

    def scan(self) -> DomainScan:
        """Observe the domain without signalling anything."""
        checked: set[int] = set()
        members: list[int] = []
        unknown: list[int] = []
        try:
            for _ in range(_SCAN_ROUNDS):
                fresh = [pid for pid in self._table.pids() if pid not in checked]
                if not fresh:
                    return DomainScan(complete=True)
                for pid in fresh:
                    checked.add(pid)
                    verdict = self._table.membership(pid, self.launch_id)
                    if verdict is True:
                        members.append(pid)
                    elif verdict is None:
                        unknown.append(pid)
                if members or unknown:
                    return DomainScan(tuple(sorted(members)), tuple(sorted(unknown)), True)
        except _TABLE_ERRORS:
            return DomainScan()
        return DomainScan()

    def poll(self) -> bool:
        """Take at most one observation; True once quiescence is proven.

        Cheap to call in a tight loop: it skips the scan while settling, and
        scans less often the longer the tree stays alive.
        """
        if not self._root_exited():
            self._first_empty = None
            return False
        now = self._clock()
        first = self._first_empty
        if first is not None and now - first < SETTLE_SECONDS:
            return False
        if first is None and now < self._next_poll:
            return False
        if not self.scan().empty:
            self._first_empty = None
            self._backoff = min(
                max(self._backoff * 2, _INTERVAL_SECONDS), _MAX_BACKOFF_SECONDS,
            )
            self._next_poll = self._clock() + self._backoff
            return False
        self._backoff = self._next_poll = 0.0
        if first is None:
            self._first_empty = self._clock()
            return False
        return True

    def kill_pass(self) -> KillPass:
        """Signal every member that satisfies the invariants, once."""
        self._backoff = self._next_poll = 0.0
        scan = self.scan()
        blocked: dict[int, BlockReason] = {pid: "unknown_membership" for pid in scan.unknown}
        if scan.members:
            protected = self._protected_pids()
            for pid in scan.members:
                reason: BlockReason | None = (
                    "ancestry_unavailable" if protected is None
                    else self._signal_member(pid, protected)
                )
                if reason is not None:
                    blocked[pid] = reason
        return KillPass(scan, tuple(sorted(blocked.items())))

    def wait_quiescent(self, deadline: float) -> bool:
        """Wait, without signalling, for the tree to exit by itself."""
        while not self.poll():
            if self._clock() >= deadline:
                return False
            self._sleep(_INTERVAL_SECONDS)
        return True

    def kill_step(self, deadline: float) -> Quiescent | NotQuiescent | None:
        """Run one kill pass; None while the outcome is still open."""
        outcome = self.kill_pass()
        if outcome.scan.empty:
            if self.poll():
                return Quiescent()
        else:
            self._first_empty = None
        if self._clock() >= deadline:
            return self._not_quiescent(outcome)
        return None

    def kill_until_quiescent(self, deadline: float) -> Quiescent | NotQuiescent:
        """Kill members until quiescence is proven or the deadline passes."""
        while (result := self.kill_step(deadline)) is None:
            self._sleep(_INTERVAL_SECONDS)
        return result

    def _not_quiescent(self, outcome: KillPass) -> NotQuiescent:
        if outcome.blocked:
            reason = outcome.blocked[0][1]
            return NotQuiescent(
                reason, tuple(pid for pid, why in outcome.blocked if why == reason),
            )
        if not outcome.scan.complete:
            return NotQuiescent("scan_unavailable", ())
        if not self._root_exited():
            return NotQuiescent("root_running", outcome.scan.members)
        return NotQuiescent("deadline", outcome.scan.members)

    def _signal_member(self, pid: int, protected: frozenset[int]) -> BlockReason | None:
        """Kill one member; None when it was signalled or is no longer that member."""
        if pid <= 1 or pid in protected:
            return "protected_member"
        try:
            before = self._table.info(pid)
            if self._table.membership(pid, self.launch_id) is not True:
                return None
            if before is None:
                return "unidentified_member"
            if before.zombie:
                return None
            if before.uid != self._owner.uid:
                return "foreign_uid_member"
            if before.start_us < self._owner.start_us:
                return "predates_launch"
            latest = self._table.info(pid)
            if latest is None or latest.start_us != before.start_us:
                return None
            self._signaller.kill(pid)
        except ProcessLookupError:
            return None
        except _TABLE_ERRORS:
            return "unsignallable_member"
        return None

    def _protected_pids(self) -> frozenset[int] | None:
        """Pids that must never be signalled; None if the ancestry is unreadable."""
        found: set[int] = {0, 1}
        try:
            for origin in (self._owner.pid, *self._extra):
                pid = origin
                while pid > 1 and pid not in found:
                    found.add(pid)
                    parent = self._table.parent_pid(pid)
                    if parent is None:
                        # An extra protected process may have exited; a broken
                        # chain anywhere else means protection is incomplete.
                        if pid == origin and origin != self._owner.pid:
                            break
                        return None
                    pid = parent
        except _TABLE_ERRORS:
            return None
        self._ancestors |= found
        return self._ancestors


def host_launch_domain(
    launch_id: str,
    *,
    root_exited: Callable[[], bool],
    protected: Iterable[int] = (),
) -> LaunchDomain:
    """Build the domain for the current process acting as supervisor."""
    table = HostProcTable()
    owner = table.info(os.getpid())
    if owner is None:
        raise darwin_proc.DarwinUnavailableError("the supervisor identity is unavailable")
    return LaunchDomain(
        launch_id, table=table, signaller=HostSignaller(), owner=owner,
        root_exited=root_exited, protected=protected,
    )
