"""Dependency-free resource ownership failure shared by runtime lifecycles."""

from typing import cast


class RuntimeOwnedWorkError(RuntimeError):
    """Cleanup failed; owned effects still prohibit closing shared resources."""


def has_owned_work_error(error: BaseException) -> bool:
    """Return whether a cleanup error reports work that still needs its owner."""
    if isinstance(error, RuntimeOwnedWorkError):
        return True
    if isinstance(error, BaseExceptionGroup):
        group = cast(BaseExceptionGroup[BaseException], error)
        return any(has_owned_work_error(item) for item in group.exceptions)
    return False
