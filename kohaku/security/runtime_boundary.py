"""Shared classification for instance runtime paths.

Private runtime state is the instance data tree minus the agent home, plus
explicit private roots. Media is a read-only input source and is admitted by
the caller; this module answers only the private-state question.
"""

from __future__ import annotations

from pathlib import Path


def is_private_runtime_path(
    path: Path,
    *,
    instance_root: Path,
    agent_home: Path | None,
    private_roots: tuple[Path, ...] = (),
) -> bool:
    """Return whether *path* is private runtime state.

    This is the single definition used by filesystem, exec, and sandbox gates:
    an explicit private root, or anything under the instance root that is not
    under the agent home. Callers that admit read-only media keep that narrow
    exception at their own boundary.
    """
    resolved = path.expanduser().resolve(strict=False)
    if any(resolved.is_relative_to(root) for root in private_roots):
        return True
    if not resolved.is_relative_to(instance_root):
        return False
    return agent_home is None or not resolved.is_relative_to(agent_home)
