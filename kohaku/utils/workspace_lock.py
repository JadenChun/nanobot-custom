"""Advisory lock shared by workspace memory writes and media retention."""

from __future__ import annotations

import os
import stat
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


class WorkspaceLockError(OSError):
    """The workspace memory lock could not be acquired or verified."""


def _check_binding(path: Path, fd: int) -> None:
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or not os.path.samestat(info, path.lstat())
    ):
        raise WorkspaceLockError("Invalid workspace memory lock")
    if os.name == "posix" and (info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600):
        raise WorkspaceLockError("Invalid workspace memory lock permissions")


def _set_lock(fd: int, *, acquire: bool) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK if acquire else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX if acquire else fcntl.LOCK_UN)


@contextmanager
def open_workspace_memory_lock(workspace: Path) -> Generator[None, None, None]:
    """Take the same exclusive lock that media retention uses."""
    fd: int | None = None
    locked = False
    try:
        try:
            path = workspace.resolve(strict=True) / ".memory-recovery.lock"
            flags = os.O_RDWR | os.O_CREAT
            if os.name == "posix":
                flags |= os.O_NOFOLLOW | os.O_NONBLOCK
            fd = os.open(path, flags, 0o600)
            _check_binding(path, fd)
            _set_lock(fd, acquire=True)
            locked = True
            _check_binding(path, fd)
        except WorkspaceLockError:
            raise
        except OSError as exc:
            raise WorkspaceLockError("Cannot acquire the workspace memory lock") from exc
        yield
    finally:
        if fd is not None:
            try:
                if locked:
                    _set_lock(fd, acquire=False)
            finally:
                os.close(fd)
