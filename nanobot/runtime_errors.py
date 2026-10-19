"""Dependency-free resource ownership failure shared by runtime lifecycles."""


class RuntimeOwnedWorkError(RuntimeError):
    """Cleanup failed; owned effects still prohibit closing shared resources."""
