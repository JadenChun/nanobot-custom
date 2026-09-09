"""Activation boundary for the unfinished upstream migration candidate.

P2-P5 must replace this boundary with verified policy and state adapters before
runtime acceptance. There is deliberately no environment-variable bypass.
"""

class MigrationIncompleteError(RuntimeError):
    """Execution is unavailable until custom runtime contracts are integrated."""


def require_runtime_ready() -> None:
    raise MigrationIncompleteError(
        "Migration candidate is not runnable: policy, provider and durable-state "
        "adapters require P2-P5 acceptance. Use the existing staging installation."
    )
