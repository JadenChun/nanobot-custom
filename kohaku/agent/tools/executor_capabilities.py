"""What this host can confine and prove for managed commands.

Managed command execution rests on two independent guarantees:

* confinement: an OS fence that limits what a launched command can touch;
* tree ownership: proof that every descendant of a launch has exited, so
  authority can be retired with nothing left running.

Linux gets confinement from bubblewrap and ownership from the subreaper
supervisor. Any other platform reports neither, so shell admission fails closed
there. Platform support is added by teaching the two selectors below a new
value, not by branching in call sites.

The legacy managed_bwrap_available and native_supervisor_available functions
remain the entry points that existing call sites and tests patch. The latter
now delegates here; the former is the bubblewrap probe itself.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

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


def tree_ownership() -> TreeOwnership | None:
    """Name the mechanism that can prove a whole launch tree has exited."""
    if current_platform() == "linux" and os.name == "posix":
        return "linux_subreaper"
    return None


def confinement() -> Confinement | None:
    """Name the OS fence that can wrap a managed command here."""
    if current_platform() == "linux":
        # Resolve at call time so the sandbox module remains the patch point.
        from kohaku.agent.tools import sandbox

        if sandbox.managed_bwrap_available():
            return "bwrap"
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
