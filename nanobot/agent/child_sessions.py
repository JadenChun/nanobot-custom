"""Durable delegated child admission for foreground work.

One admitted child owns a session identity, parent ancestry edge, runtime
profile, narrowed policy facts, attempt identity, and a finite tree slot.
Reservations are released only when the owner can prove the child did not
start; uncertain launches remain visible for reconciliation.

Runtime profiles are topology facts, not role names. The default profile is
``delegated`` for a native child. Foreground role policy (tool allowlist,
read-only promise, write scope) is carried separately as policy facts derived
from the live parent authority and the admitted assignment. Optional domain
roles such as the crawler are extensions, not core runtime profiles.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, cast

from loguru import logger

from nanobot.session.manager import Session, SessionManager
from nanobot.session.runtime_authority import (
    RUNTIME_SESSION_METADATA_KEY,
    RuntimeSessionMetadata,
)

DELEGATION_CHILD_METADATA_KEY = "delegation_child"
CHILD_SESSION_PREFIX = "delegated:"
DEFAULT_CHILD_PROFILE = "delegated"
SUPPORTED_CHILD_PROFILES = frozenset({DEFAULT_CHILD_PROFILE})
_LIVE_STATES = frozenset({"reserved", "active", "joining"})
_TERMINAL_STATES = frozenset({
    "completed", "failed", "uncertain", "stopped", "refused",
})


@dataclass(frozen=True, slots=True)
class DelegationTreePolicy:
    """Finite tree limits aligned with the accepted foreground delegation design.

    Delegation is synchronous and non-recursive: a child cannot create another
    delegated child. Two siblings are allowed under one parent, and the existing
    ``max_concurrent_subagents`` capacity bounds the live tree at four.
    """

    max_depth: int = 1
    max_live_children_per_parent: int = 2
    max_live_descendants: int = 4

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("delegation depth limit must be positive")
        if self.max_live_children_per_parent < 1:
            raise ValueError("delegation child limit must be positive")
        if self.max_live_descendants < self.max_live_children_per_parent:
            raise ValueError("delegation tree limit must hold one parent's children")


@dataclass(frozen=True, slots=True)
class ChildRefusal:
    """A pre-launch refusal that owns no live child or reservation."""

    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class ChildAdmission:
    """One committed child identity and its current tree state."""

    child_id: str
    session_key: str
    parent_key: str
    parent_incarnation: str
    root_key: str
    profile: str
    depth: int
    attempt_id: str
    assignment_digest: str
    policy_facts: dict[str, Any]
    runtime_generation: int
    state: str = "reserved"
    outcome: str | None = None
    transcript: tuple[dict[str, str], ...] = ()
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def live(self) -> bool:
        return self.state in _LIVE_STATES

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    def metadata(self) -> dict[str, Any]:
        return {
            "child_id": self.child_id,
            "session_key": self.session_key,
            "parent_key": self.parent_key,
            "parent_incarnation": self.parent_incarnation,
            "root_key": self.root_key,
            "profile": self.profile,
            "depth": self.depth,
            "attempt_id": self.attempt_id,
            "assignment_digest": self.assignment_digest,
            "policy_facts": dict(self.policy_facts),
            "runtime_generation": self.runtime_generation,
            "state": self.state,
            "outcome": self.outcome,
            "transcript": [dict(message) for message in self.transcript],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def assignment_digest(
    *,
    profile: str,
    assignment: str,
    policy_facts: Mapping[str, Any] | None = None,
) -> str:
    """Content-free stable identity for one admitted assignment."""
    payload = "\x1f".join([
        profile,
        assignment.strip(),
        repr(sorted((policy_facts or {}).items())),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DelegationTree:
    """Reserve, admit, and settle durable child identities under one session owner."""

    def __init__(
        self,
        sessions: SessionManager,
        *,
        policy: DelegationTreePolicy | None = None,
    ) -> None:
        self.sessions = sessions
        self.policy = policy or DelegationTreePolicy()
        self._lock = asyncio.Lock()
        self._children: dict[str, ChildAdmission] = {}
        self._attempts: dict[tuple[str, str, str], str] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._restore()

    def _restore(self) -> None:
        """Load durable child records; fence unresolved live states on restart."""
        try:
            sessions = self.sessions.list_sessions()
        except Exception:
            logger.exception("Delegated child recovery scan failed")
            return
        for info in sessions:
            key = info.get("key")
            if not isinstance(key, str) or not key.startswith(CHILD_SESSION_PREFIX):
                continue
            try:
                raw = self.sessions.read_session_metadata(key)
                metadata = cast(
                    dict[str, Any] | None,
                    raw.get("metadata") if raw is not None else None,
                )
                if metadata is None:
                    continue
                payload = metadata.get(DELEGATION_CHILD_METADATA_KEY)
                if not isinstance(payload, Mapping):
                    continue
                data = dict(cast(Mapping[str, Any], payload))
                admission = ChildAdmission(
                    child_id=cast(str, data["child_id"]),
                    session_key=cast(str, data["session_key"]),
                    parent_key=cast(str, data["parent_key"]),
                    parent_incarnation=cast(str, data["parent_incarnation"]),
                    root_key=cast(str, data["root_key"]),
                    profile=cast(str, data["profile"]),
                    depth=cast(int, data["depth"]),
                    attempt_id=cast(str, data["attempt_id"]),
                    assignment_digest=cast(str, data["assignment_digest"]),
                    policy_facts=dict(cast(Mapping[str, Any], data.get("policy_facts") or {})),
                    runtime_generation=cast(int, data["runtime_generation"]),
                    state=cast(str, data.get("state") or "uncertain"),
                    outcome=cast(str | None, data.get("outcome")),
                    transcript=tuple(
                        {
                            "role": str(
                                cast(dict[str, Any], item).get("role") or ""
                            ),
                            "content": str(
                                cast(dict[str, Any], item).get("content") or ""
                            ),
                        }
                        for item in cast(list[Any], data.get("transcript") or [])
                        if isinstance(item, dict)
                    ),
                    created_at=cast(float, data.get("created_at") or time.time()),
                    updated_at=cast(float, data.get("updated_at") or time.time()),
                )
            except (KeyError, TypeError, ValueError):
                logger.warning("Ignoring malformed delegated child record {}", key)
                continue
            if admission.live:
                admission = replace(
                    admission,
                    state="uncertain",
                    outcome="restart_unresolved",
                    updated_at=time.time(),
                )
                self._persist(admission)
            self._children[admission.child_id] = admission
            self._attempts[
                (
                    admission.parent_key,
                    admission.profile,
                    f"{admission.attempt_id}:{admission.assignment_digest}",
                )
            ] = admission.child_id

    def child(self, child_id: str) -> ChildAdmission | None:
        return self._children.get(child_id)

    def children_of(self, parent_key: str) -> tuple[ChildAdmission, ...]:
        return tuple(
            child for child in self._children.values()
            if child.parent_key == parent_key
        )

    def live_descendants(self, root_key: str) -> tuple[ChildAdmission, ...]:
        return tuple(
            child for child in self._children.values()
            if child.root_key == root_key and child.live
        )

    async def reserve(
        self,
        *,
        parent: Session,
        parent_key: str,
        assignment: str,
        attempt_id: str,
        profile: str = DEFAULT_CHILD_PROFILE,
        policy_facts: Mapping[str, Any] | None = None,
        runtime_generation: int = 0,
    ) -> ChildAdmission | ChildRefusal:
        """Reserve one child slot before launch, or return an existing attempt."""
        if profile not in SUPPORTED_CHILD_PROFILES:
            return ChildRefusal(
                "unsupported_profile",
                f"Unsupported child profile: {profile!r}",
            )
        facts = dict(cast(Mapping[str, Any], policy_facts or {}))
        digest = assignment_digest(
            profile=profile,
            assignment=assignment,
            policy_facts=facts,
        )
        attempt_key = (parent_key, profile, f"{attempt_id}:{digest}")
        async with self._lock:
            existing = self._attempts.get(attempt_key)
            if existing is not None:
                child = self._children.get(existing)
                if child is not None:
                    return child
            parent_state = self._runtime_state(parent)
            parent_depth = int(parent_state.get("delegation_depth", 0))
            depth = parent_depth + 1
            if depth > self.policy.max_depth:
                return ChildRefusal(
                    "depth_limit",
                    f"Delegation depth limit {self.policy.max_depth} reached",
                )
            root_key = str(
                parent.metadata.get("delegation_root_key") or parent_key
            )
            parent_children = [
                child for child in self._children.values()
                if child.parent_key == parent_key and child.live
            ]
            if len(parent_children) >= self.policy.max_live_children_per_parent:
                return ChildRefusal(
                    "child_limit",
                    "This parent already holds its live child limit",
                )
            live_tree = [
                child for child in self._children.values()
                if child.root_key == root_key and child.live
            ]
            if len(live_tree) >= self.policy.max_live_descendants:
                return ChildRefusal(
                    "tree_limit",
                    "The delegation tree already holds its live child limit",
                )
            child_id = uuid.uuid4().hex
            child_key = f"{CHILD_SESSION_PREFIX}{child_id}"
            admission = ChildAdmission(
                child_id=child_id,
                session_key=child_key,
                parent_key=parent_key,
                parent_incarnation=parent.incarnation,
                root_key=root_key,
                profile=profile,
                depth=depth,
                attempt_id=attempt_id,
                assignment_digest=digest,
                policy_facts=facts,
                runtime_generation=runtime_generation,
            )
            if not self._persist(admission, parent=parent):
                return ChildRefusal(
                    "session_unavailable",
                    "Could not persist the delegated child identity",
                )
            self._children[child_id] = admission
            self._attempts[attempt_key] = child_id
            return admission

    async def activate(self, child_id: str) -> ChildAdmission | None:
        return await self._transition(child_id, "active")

    async def begin_join(self, child_id: str) -> ChildAdmission | None:
        return await self._transition(child_id, "joining")

    async def complete(
        self,
        child_id: str,
        *,
        outcome: str,
        state: str = "completed",
    ) -> ChildAdmission | None:
        if state not in {"completed", "failed", "uncertain", "stopped"}:
            raise ValueError("delegated child terminal state is invalid")
        updated = await self._transition(child_id, state, outcome=outcome)
        self._tasks.pop(child_id, None)
        return updated

    async def release(self, child_id: str, *, reason: str = "not_started") -> ChildAdmission | None:
        """Release a reservation only when the child provably did not run."""
        updated = await self._transition(child_id, "refused", outcome=reason)
        self._tasks.pop(child_id, None)
        return updated

    async def attach_task(self, child_id: str, task: asyncio.Task[Any]) -> None:
        async with self._lock:
            self._tasks[child_id] = task

    def record_transcript(
        self,
        child_id: str,
        *,
        assignment: str,
        result_text: str | None,
    ) -> bool:
        """Store a bounded, display-only child transcript in session metadata."""
        child = self._children.get(child_id)
        if child is None:
            return False
        messages: list[dict[str, str]] = [
            {"role": "user", "content": assignment[:4000]},
        ]
        if result_text:
            messages.append({"role": "assistant", "content": result_text[:8000]})
        updated = replace(child, transcript=tuple(messages), updated_at=time.time())
        self._children[child_id] = updated
        return self._persist(updated)

    def record_exchange(
        self,
        child_id: str,
        *,
        assignment: str,
        result_text: str | None,
    ) -> bool:
        """Append the admitted assignment and terminal result to child history."""
        child = self._children.get(child_id)
        if child is None:
            return False
        try:
            session = self.sessions.get_or_create(child.session_key)
            session.add_message("user", assignment)
            if result_text:
                session.add_message("assistant", result_text)
            self.sessions.save(session)
            return True
        except Exception:
            logger.exception(
                "Delegated child {} history could not be recorded",
                child_id,
            )
            return False

    async def wait_child(
        self,
        child_id: str,
        *,
        timeout: float = 30.0,
    ) -> ChildAdmission | None:
        """Wait for one live child task without consuming its ownership."""
        child = self._children.get(child_id)
        if child is None:
            return None
        task = self._tasks.get(child_id)
        if task is None:
            return child
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        return self._children.get(child_id)

    async def stop_for_parent(self, parent_key: str) -> tuple[ChildAdmission, ...]:
        """Stop every live child admitted under one parent session."""
        async with self._lock:
            child_ids = [
                child.child_id for child in self._children.values()
                if child.parent_key == parent_key and child.live
            ]
        stopped: list[ChildAdmission] = []
        for child_id in child_ids:
            stopped.extend(await self.stop_subtree(child_id))
        return tuple(stopped)

    async def stop_subtree(self, child_id: str) -> tuple[ChildAdmission, ...]:
        """Cancel one child and its descendants; siblings remain independent."""
        async with self._lock:
            victims = self._subtree_locked(child_id)
        cancelled: list[ChildAdmission] = []
        for victim in victims:
            task = self._tasks.get(victim.child_id)
            if task is not None and not task.done():
                task.cancel()
            updated = await self.complete(
                victim.child_id,
                outcome="stopped_by_owner",
                state="stopped",
            )
            if updated is not None:
                cancelled.append(updated)
        return tuple(cancelled)

    def _subtree_locked(self, child_id: str) -> list[ChildAdmission]:
        root = self._children.get(child_id)
        if root is None:
            return []
        found = [root]
        pending = [root.session_key]
        seen = {root.child_id}
        while pending:
            parent_key = pending.pop()
            for child in self._children.values():
                if child.parent_key != parent_key or child.child_id in seen:
                    continue
                seen.add(child.child_id)
                found.append(child)
                pending.append(child.session_key)
        return found

    async def _transition(
        self,
        child_id: str,
        state: str,
        *,
        outcome: str | None = None,
    ) -> ChildAdmission | None:
        async with self._lock:
            current = self._children.get(child_id)
            if current is None:
                return None
            if current.terminal and state != current.state:
                return current
            updated = replace(
                current,
                state=state,
                outcome=outcome if outcome is not None else current.outcome,
                updated_at=time.time(),
            )
            self._children[child_id] = updated
        self._persist(updated)
        return updated

    @staticmethod
    def _runtime_state(session: Session) -> dict[str, Any]:
        raw = session.metadata.get(RUNTIME_SESSION_METADATA_KEY)
        if not isinstance(raw, Mapping):
            return {}
        return dict(cast(Mapping[str, Any], raw))

    def _persist(self, admission: ChildAdmission, *, parent: Session | None = None) -> bool:
        try:
            runtime = RuntimeSessionMetadata(
                profile=admission.profile,
                execution_kind="native",
                delegation_depth=admission.depth,
                parent_key=admission.parent_key,
                parent_incarnation=admission.parent_incarnation,
                backend_binding="native-runner",
                runtime_generation=admission.runtime_generation,
                recovery_disposition=self._recovery_disposition(admission),
            )
            return self.sessions.upsert_session_metadata(
                admission.session_key,
                {
                    DELEGATION_CHILD_METADATA_KEY: admission.metadata(),
                    RUNTIME_SESSION_METADATA_KEY: runtime.payload(),
                    "delegation_root_key": admission.root_key,
                },
                create_if_absent=True,
            )
        except Exception:
            logger.exception(
                "Delegated child {} could not be persisted",
                admission.child_id,
            )
            return False

    @staticmethod
    def _recovery_disposition(
        admission: ChildAdmission,
    ) -> Literal["none", "interrupted", "completed", "uncertain"]:
        if admission.state == "completed":
            return "completed"
        if admission.state in {"failed", "stopped", "refused"}:
            return "interrupted"
        if admission.state == "uncertain":
            return "uncertain"
        return "none"


__all__ = [
    "CHILD_SESSION_PREFIX",
    "ChildAdmission",
    "ChildRefusal",
    "DEFAULT_CHILD_PROFILE",
    "DELEGATION_CHILD_METADATA_KEY",
    "DelegationTree",
    "DelegationTreePolicy",
    "SUPPORTED_CHILD_PROFILES",
    "assignment_digest",
]
