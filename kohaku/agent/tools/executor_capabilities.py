"""What this host can confine and prove for managed commands.

Managed command execution rests on two independent guarantees:

* confinement: an OS fence that limits what a launched command can touch;
* tree ownership: proof that every descendant of a launch has exited, so
  authority can be retired with nothing left running.

Linux gets confinement from bubblewrap and ownership from the subreaper
supervisor. macOS gets both from Seatbelt when the host passes the self-test. Anything else
reports neither, so shell admission fails closed there. Platform support is added
by teaching the two selectors below a new value, not by branching in call sites.

The legacy managed_bwrap_available and native_supervisor_available functions
remain the entry points that existing call sites and tests patch. The latter
now delegates here; the former is the bubblewrap probe itself.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from loguru import logger

Platform = Literal["linux", "darwin", "other"]
Confinement = Literal["bwrap", "seatbelt"]
TreeOwnership = Literal["linux_subreaper", "darwin_seatbelt"]


@dataclass(frozen=True)
class ExecutorCapabilities:
    """Immutable snapshot of the executor guarantees available on this host."""

    platform: Platform
    confinement: Confinement | None
    tree_ownership: TreeOwnership | None


def current_platform() -> Platform:
    if sys.platform == "linux":
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    return "other"


_RETRY_AFTER_FAILURE_SECONDS = 300.0


class CachedHostCheck:
    """A slow host check: success is kept for good, failure only for a cooldown.

    A transient failure, such as a timeout on a busy host, must not disable the
    executor until restart. Concurrent callers wait for one shared attempt.
    """

    def __init__(
        self,
        check: Callable[[], bool],
        *,
        cooldown: float = _RETRY_AFTER_FAILURE_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._check = check
        self._cooldown = cooldown
        self._clock = clock
        self._lock = threading.Lock()
        self._passed = False
        self._failed_at: float | None = None

    def __call__(self) -> bool:
        with self._lock:
            if self._passed:
                return True
            clock = self._clock or time.monotonic
            if self._failed_at is not None and clock() - self._failed_at < self._cooldown:
                return False
            self._passed = self._check()
            # The cooldown starts when the attempt ends, not when it began.
            self._failed_at = None if self._passed else clock()
            return self._passed

    def cache_clear(self) -> None:
        """Forget the last result, so the next call runs the check again."""
        with self._lock:
            self._passed = False
            self._failed_at = None


@CachedHostCheck
def _macos_self_test_passed() -> bool:
    """Run the host self-test; it takes a few seconds."""
    from kohaku.security.darwin_proc import seatbelt_self_test

    result = seatbelt_self_test()
    if not result.ok:
        logger.warning(
            "macOS Seatbelt executor disabled, self-test failed: {}",
            "; ".join(result.failures),
        )
    return result.ok


def macos_seatbelt_ready() -> bool:
    return _macos_self_test_passed()


def tree_ownership() -> TreeOwnership | None:
    """Name the mechanism that can prove a whole launch tree has exited."""
    platform = current_platform()
    if platform == "linux" and os.name == "posix":
        return "linux_subreaper"
    if platform == "darwin" and macos_seatbelt_ready():
        return "darwin_seatbelt"
    return None


def confinement() -> Confinement | None:
    """Name the OS fence that can wrap a managed command here."""
    # Resolve at call time so the sandbox module remains the patch point.
    from kohaku.agent.tools import sandbox

    platform = current_platform()
    if platform == "linux" and sandbox.managed_bwrap_available():
        return "bwrap"
    if platform == "darwin" and sandbox.managed_seatbelt_available():
        return "seatbelt"
    return None


def executor_capabilities() -> ExecutorCapabilities:
    return ExecutorCapabilities(
        platform=current_platform(),
        confinement=confinement(),
        tree_ownership=tree_ownership(),
    )


def managed_confinement_available() -> bool:
    return confinement() is not None


def tree_ownership_available() -> bool:
    return tree_ownership() is not None
