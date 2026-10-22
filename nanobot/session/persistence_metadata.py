"""Shared metadata ownership rules for legacy and source session writers."""

from copy import deepcopy
from typing import Any

BUSY_INBOX_METADATA_KEY = "busy_inbox"
BUSY_RECOVERY_KEY_PREFIX = "h1-recovery:"
SOURCE_PENDING_FOLLOWUPS_METADATA_KEY = "source_pending_followups"


def is_busy_recovery_key(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith(BUSY_RECOVERY_KEY_PREFIX):
        return False
    digest = value[len(BUSY_RECOVERY_KEY_PREFIX):]
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


DURABLE_METADATA_KEYS = frozenset({
    "context_checkpoint", "source_schema_version", "execution_lease", "execution_recovery_required",
    "revision", "generation", "incarnation", "artifact_manifest", "user_activity_revision",
    "last_user_activity_at",
    BUSY_INBOX_METADATA_KEY,
    SOURCE_PENDING_FOLLOWUPS_METADATA_KEY,
})
LEGACY_RECOVERY_KEYS = frozenset({
    "runtime_checkpoint", "pending_user_turn", "pending_user_followups", "webui_recovery",
})


def accepted_metadata_updates(updates: dict[str, Any], *, source: bool = False) -> dict[str, Any]:
    reserved = DURABLE_METADATA_KEYS | LEGACY_RECOVERY_KEYS if source else DURABLE_METADATA_KEYS
    return {name: deepcopy(value) for name, value in updates.items() if name not in reserved}
