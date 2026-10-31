"""Remote task authority ceiling (M13).

Phone requests are intersected with the server-selected restricted task. A local
Full reference, unmanaged installation or stale task revision never becomes a
remote grant.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

ALLOWED_REMOTE_ACTIONS = frozenset({"chat.send", "chat.stop", "chat.read", "media.read", "attachment.write"})
DENIED_ACTION_PREFIXES = (
    "settings.",
    "system.",
    "workspace.",
    "folder.",
    "automation.",
    "skill.",
    "package.",
    "temporary.",
    "delete.",
    "native.",
    "full.",
    "shell.",
)


@dataclass(frozen=True)
class RemoteAuthorityDecision:
    allowed: bool
    reason: str
    action: str = ""
    task_turn_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "action": self.action,
            "task_turn_id": self.task_turn_id,
        }


class RemoteAuthorityCeiling:
    """Derive a bounded capability decision from the current projection."""

    def qualify(
        self,
        projection: Mapping[str, Any] | None,
        *,
        action: str,
        task_turn_id: str | None = None,
    ) -> RemoteAuthorityDecision:
        if any(action.startswith(prefix) for prefix in DENIED_ACTION_PREFIXES):
            return RemoteAuthorityDecision(False, "action_not_remote", action)
        if action not in ALLOWED_REMOTE_ACTIONS:
            return RemoteAuthorityDecision(False, "action_not_allowed", action)
        if projection is None:
            return RemoteAuthorityDecision(False, "unmanaged_installation", action)
        if projection.get("version") != 1:
            return RemoteAuthorityDecision(False, "projection_version_unsupported", action)
        mode = projection.get("mode")
        state = projection.get("state")
        if mode == "full":
            return RemoteAuthorityDecision(False, "local_full_not_remote", action)
        if mode != "restricted":
            return RemoteAuthorityDecision(False, "projection_mode_invalid", action)
        if state == "retiring":
            return RemoteAuthorityDecision(False, "permission_retiring", action)
        if state != "active":
            return RemoteAuthorityDecision(False, "projection_state_invalid", action)
        if action in {"chat.read", "media.read"}:
            return RemoteAuthorityDecision(True, "authorized_read", action)
        task = projection.get("task_access")
        if not isinstance(task, Mapping):
            return RemoteAuthorityDecision(False, "task_not_qualified", action)
        task_map = cast(Mapping[str, Any], task)
        if task_map.get("state") != "active":
            return RemoteAuthorityDecision(False, "task_not_active", action)
        bound_turn = task_map.get("task_turn_id")
        if not isinstance(bound_turn, str) or not bound_turn:
            return RemoteAuthorityDecision(False, "task_binding_missing", action)
        if task_turn_id is not None and task_turn_id != bound_turn:
            return RemoteAuthorityDecision(False, "task_revision_stale", action)
        return RemoteAuthorityDecision(True, "restricted_task", action, bound_turn)

    def capabilities(
        self,
        projection: Mapping[str, Any] | None,
        *,
        task_turn_id: str | None = None,
    ) -> dict[str, Any]:
        send = self.qualify(projection, action="chat.send", task_turn_id=task_turn_id)
        read = self.qualify(projection, action="chat.read", task_turn_id=task_turn_id)
        media = self.qualify(projection, action="media.read", task_turn_id=task_turn_id)
        return {
            "chat_send": send.allowed,
            "chat_read": read.allowed,
            "media_read": media.allowed,
            "reason": send.reason,
            "task_turn_id": send.task_turn_id,
        }

    def require(
        self,
        projection: Mapping[str, Any] | None,
        *,
        action: str,
        task_turn_id: str | None = None,
    ) -> RemoteAuthorityDecision:
        decision = self.qualify(
            projection,
            action=action,
            task_turn_id=task_turn_id,
        )
        if not decision.allowed:
            raise PermissionError(decision.reason)
        return decision
