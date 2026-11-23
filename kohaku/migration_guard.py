"""Compatibility fence for legacy entrypoints without local runtime admission.

Normal local foreground composition uses config-aware runtime preflight instead.
This no-argument fence remains for legacy desktop and administrative paths that
do not yet participate in that admission flow. There is deliberately no
environment-variable bypass.
"""

class MigrationIncompleteError(RuntimeError):
    """A legacy entrypoint has no applicable runtime admission contract."""


def require_runtime_ready() -> None:
    raise MigrationIncompleteError(
        "Migration candidate is not runnable: policy, provider and durable-state "
        "adapters require P2-P5 acceptance. Use the existing staging installation."
    )
