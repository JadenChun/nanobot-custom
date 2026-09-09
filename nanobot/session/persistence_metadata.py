"""Shared metadata ownership rules for legacy and source session writers."""

from copy import deepcopy
from typing import Any

DURABLE_METADATA_KEYS = frozenset({
    "context_checkpoint", "source_schema_version", "execution_lease",
    "revision", "generation", "incarnation",
})
LEGACY_RECOVERY_KEYS = frozenset({
    "runtime_checkpoint", "pending_user_turn", "pending_user_followups", "webui_recovery",
})


def accepted_metadata_updates(updates: dict[str, Any], *, source: bool = False) -> dict[str, Any]:
    reserved = DURABLE_METADATA_KEYS | LEGACY_RECOVERY_KEYS if source else DURABLE_METADATA_KEYS
    return {name: deepcopy(value) for name, value in updates.items() if name not in reserved}
