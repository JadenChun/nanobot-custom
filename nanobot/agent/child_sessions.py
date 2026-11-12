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
import math
import re
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, cast

from loguru import logger

from nanobot.session.manager import Session, SessionManager
from nanobot.session.pending_followups import (
    PendingBackgroundResult,
    parse_pending_followup_records,
)
from nanobot.session.persistence_metadata import SOURCE_PENDING_FOLLOWUPS_METADATA_KEY
from nanobot.session.runtime_authority import (
    RUNTIME_SESSION_METADATA_KEY,
    RuntimeSessionMetadata,
    read_runtime_session,
)
from nanobot.session.source import new_source_event
from nanobot.session.source_store import SourceState

DELEGATION_CHILD_METADATA_KEY = "delegation_child"
CHILD_SESSION_PREFIX = "delegated:"
DEFAULT_CHILD_PROFILE = "delegated"
SUPPORTED_CHILD_PROFILES = frozenset({DEFAULT_CHILD_PROFILE})
_LIVE_STATES = frozenset({"reserved", "active", "joining", "waiting_approval", "stopping"})
_TERMINAL_STATES = frozenset({
    "completed", "failed", "uncertain", "stopped", "refused",
})
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
STOP_SETTLE_TIMEOUT = 10.0
_SHARED_TREE_LOCK = threading.RLock()
_SHARED_TREES: weakref.WeakKeyDictionary[
    SessionManager, weakref.ReferenceType[DelegationTree]
] = weakref.WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class DelegationTreePolicy:
    """Finite tree limits for durable delegated children.

    A child may create one nested child. Two siblings are allowed under one
    parent; each root and the gateway have separate finite child budgets.
    """

    max_depth: int = 2
    max_live_children_per_parent: int = 2
    max_live_descendants: int = 4
    max_live_children_per_gateway: int = 4

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("delegation depth limit must be positive")
        if self.max_live_children_per_parent < 1:
            raise ValueError("delegation child limit must be positive")
        if self.max_live_descendants < self.max_live_children_per_parent:
            raise ValueError("delegation tree limit must hold one parent's children")
        if self.max_live_children_per_gateway < 1:
            raise ValueError("gateway child limit must be positive")


@dataclass(frozen=True, slots=True)
class ChildRefusal:
    """A pre-launch refusal that owns no live child or reservation."""

    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class ChildControlState:
    """Expected source lifetimes and sequence for one existing child control."""

    child_id: str
    parent_key: str
    parent_incarnation: str
    parent_source_state: SourceState | None
    parent_present: bool
    root_key: str
    root_source_state: SourceState | None
    root_present: bool
    child_source_state: SourceState
    child_state_sequence: int
    control_epoch: int


@dataclass(frozen=True, slots=True)
class ChildStopAdmission:
    """One admitted child-only Stop bound to its observed child state."""

    expected: ChildControlState
    control_epoch: int


@dataclass(frozen=True, slots=True)
class OriginalChatStopAdmission:
    """One root Stop bound to its source lifetime and current active tasks."""

    parent_key: str
    parent_incarnation: str | None
    parent_source_state: SourceState | None
    control_epoch: int
    active_root_identity: tuple[asyncio.Task[Any], ...]
    target_turn_id: str | None = None
    pending_queue_identity: int | None = None


@dataclass(frozen=True, slots=True)
class BackgroundDeliveryIntent:
    """Exact parent lifetime and route captured when a background child starts."""

    parent_key: str
    parent_source_state: SourceState
    parent_run_id: str
    parent_tool_event_id: str
    parent_execution_id: str
    parent_dispatch_ordinal: int
    route_channel: str
    route_chat_id: str

    def __post_init__(self) -> None:
        if (
            not self.parent_key
            or not self.parent_run_id
            or not self.parent_tool_event_id
            or not self.parent_execution_id
            or type(self.parent_dispatch_ordinal) is not int
            or self.parent_dispatch_ordinal < 0
            or not self.route_channel
            or not self.route_chat_id
            or type(self.parent_source_state) is not SourceState
        ):
            raise ValueError("Background delivery intent is invalid")


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
    control_epoch: int = 0
    source_state: SourceState | None = None
    parent_source_state: SourceState | None = None
    root_source_state: SourceState | None = None
    run_sequence: int = 0
    state_sequence: int = 0
    current_run_id: str | None = None
    last_run_id: str | None = None
    holds_capacity: bool = True
    pending_approval_id: str | None = None
    pending_effect_id: str | None = None
    pending_summary: str | None = None
    blocked_run_id: str | None = None
    state: str = "reserved"
    outcome: str | None = None
    transcript: tuple[dict[str, str], ...] = ()
    background_delivery: BackgroundDeliveryIntent | None = None
    background_result: PendingBackgroundResult | None = None
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
            "control_epoch": self.control_epoch,
            "source_state": _source_state_payload(self.source_state),
            "parent_source_state": _source_state_payload(self.parent_source_state),
            "root_source_state": _source_state_payload(self.root_source_state),
            "run_sequence": self.run_sequence,
            "state_sequence": self.state_sequence,
            "current_run_id": self.current_run_id,
            "last_run_id": self.last_run_id,
            "holds_capacity": self.holds_capacity,
            "pending_approval_id": self.pending_approval_id,
            "pending_effect_id": self.pending_effect_id,
            "pending_summary": self.pending_summary,
            "blocked_run_id": self.blocked_run_id,
            "state": self.state,
            "outcome": self.outcome,
            "transcript": [dict(message) for message in self.transcript],
            "background_delivery": _background_delivery_payload(self.background_delivery),
            "background_result": (
                self.background_result.to_dict()
                if self.background_result is not None
                else None
            ),
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


def _source_state_payload(state: SourceState | None) -> dict[str, object] | None:
    if state is None:
        return None
    return {
        "store_identity": state.store_identity,
        "session_key": state.session_key,
        "incarnation": state.incarnation,
        "revision": state.revision,
        "generation": state.generation,
        "checkpoint_id": state.checkpoint_id,
        "checkpoint_fingerprint": state.checkpoint_fingerprint,
    }


def _background_delivery_payload(
    intent: BackgroundDeliveryIntent | None,
) -> dict[str, object] | None:
    if intent is None:
        return None
    return {
        "parent_key": intent.parent_key,
        "parent_source_state": _source_state_payload(intent.parent_source_state),
        "parent_run_id": intent.parent_run_id,
        "parent_tool_event_id": intent.parent_tool_event_id,
        "parent_execution_id": intent.parent_execution_id,
        "parent_dispatch_ordinal": intent.parent_dispatch_ordinal,
        "route_channel": intent.route_channel,
        "route_chat_id": intent.route_chat_id,
    }


def _parse_background_delivery(value: object) -> BackgroundDeliveryIntent:
    if not isinstance(value, Mapping):
        raise ValueError("child background delivery is invalid")
    data = cast(Mapping[str, object], value)
    strings = (
        "parent_key",
        "parent_run_id",
        "parent_tool_event_id",
        "parent_execution_id",
        "route_channel",
        "route_chat_id",
    )
    parsed: dict[str, str] = {}
    for name in strings:
        field_value = data.get(name)
        if not isinstance(field_value, str) or not field_value:
            raise ValueError("child background delivery is invalid")
        parsed[name] = field_value
    ordinal = data.get("parent_dispatch_ordinal")
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("child background delivery is invalid")
    parent_state = _parse_source_state(data.get("parent_source_state"))
    if parent_state is None:
        raise ValueError("child background delivery is invalid")
    return BackgroundDeliveryIntent(
        parent_key=parsed["parent_key"],
        parent_source_state=parent_state,
        parent_run_id=parsed["parent_run_id"],
        parent_tool_event_id=parsed["parent_tool_event_id"],
        parent_execution_id=parsed["parent_execution_id"],
        parent_dispatch_ordinal=ordinal,
        route_channel=parsed["route_channel"],
        route_chat_id=parsed["route_chat_id"],
    )


def _parse_source_state(value: object) -> SourceState | None:
    if not isinstance(value, Mapping):
        return None
    data = dict(cast(Mapping[str, object], value))
    store_identity = data.get("store_identity")
    session_key = data.get("session_key")
    incarnation = data.get("incarnation")
    revision = data.get("revision")
    generation = data.get("generation")
    checkpoint_id = data.get("checkpoint_id")
    checkpoint_fingerprint = data.get("checkpoint_fingerprint")
    if (
        not isinstance(store_identity, str)
        or not store_identity
        or not isinstance(session_key, str)
        or not session_key
        or not isinstance(incarnation, str)
        or not incarnation
        or type(revision) is not int
        or revision < 0
        or type(generation) is not int
        or generation < 0
        or (checkpoint_id is not None and not isinstance(checkpoint_id, str))
        or not isinstance(checkpoint_fingerprint, str)
    ):
        return None
    return SourceState(
        store_identity=store_identity,
        session_key=session_key,
        incarnation=incarnation,
        revision=revision,
        generation=generation,
        checkpoint_id=checkpoint_id,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )


def _parse_child_admission(key: str, data: Mapping[str, object]) -> ChildAdmission:
    def text(name: str) -> str:
        value = data.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"child field {name!r} is invalid")
        return value

    def optional_text(name: str) -> str | None:
        value = data.get(name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"child field {name!r} is invalid")
        return value

    def counter(name: str, *, default: int = 0, minimum: int = 0) -> int:
        value = data.get(name, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"child field {name!r} is invalid")
        return value

    child_id = text("child_id")
    session_key = text("session_key")
    parent_key = text("parent_key")
    parent_incarnation = text("parent_incarnation")
    root_key = text("root_key")
    profile = text("profile")
    if (
        key != session_key
        or session_key != f"{CHILD_SESSION_PREFIX}{child_id}"
        or not re.fullmatch(r"[0-9a-f]{32}", child_id)
        or parent_key == session_key
        or root_key == session_key
        or profile not in SUPPORTED_CHILD_PROFILES
    ):
        raise ValueError("child identity does not match its canonical source")

    digest = text("assignment_digest")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("child assignment digest is invalid")
    facts = data.get("policy_facts")
    if not isinstance(facts, Mapping):
        raise ValueError("child policy facts are invalid")

    source_state = _parse_source_state(data.get("source_state"))
    parent_source_state = _parse_source_state(data.get("parent_source_state"))
    root_source_state = _parse_source_state(data.get("root_source_state"))
    if (
        source_state is None
        or source_state.session_key != session_key
        or parent_source_state is None
        or parent_source_state.session_key != parent_key
        or root_source_state is None
        or root_source_state.session_key != root_key
        or source_state.store_identity != parent_source_state.store_identity
        or source_state.store_identity != root_source_state.store_identity
    ):
        raise ValueError("child source owner binding is invalid")

    state_value = data.get("state", "uncertain")
    if not isinstance(state_value, str) or state_value not in _LIVE_STATES | _TERMINAL_STATES:
        raise ValueError("child state is invalid")
    holds_value = data.get("holds_capacity")
    if holds_value is not None and type(holds_value) is not bool:
        raise ValueError("child capacity disposition is invalid")
    holds_capacity = (
        holds_value
        if type(holds_value) is bool
        else state_value in _LIVE_STATES or state_value == "uncertain"
    )

    raw_transcript = data.get("transcript", [])
    if not isinstance(raw_transcript, list):
        raise ValueError("child transcript is invalid")
    transcript: list[dict[str, str]] = []
    for raw_item in cast(list[object], raw_transcript):
        if not isinstance(raw_item, Mapping):
            raise ValueError("child transcript row is invalid")
        item = cast(Mapping[str, object], raw_item)
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("child transcript content is invalid")
        transcript.append({"role": role, "content": content})

    raw_background_delivery = data.get("background_delivery")
    background_delivery = (
        _parse_background_delivery(raw_background_delivery)
        if raw_background_delivery is not None
        else None
    )
    if background_delivery is not None and background_delivery.parent_key != parent_key:
        raise ValueError("child background delivery binding is invalid")
    raw_background_result = data.get("background_result")
    background_result = (
        PendingBackgroundResult.from_value(raw_background_result)
        if raw_background_result is not None
        else None
    )
    if background_result is not None and (
        background_result.child_id != child_id
        or background_result.child_key != session_key
        or background_result.parent_key != parent_key
    ):
        raise ValueError("child background result binding is invalid")

    def timestamp(name: str) -> float:
        value = data.get(name, time.time())
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"child timestamp {name!r} is invalid")
        return float(value)

    return ChildAdmission(
        child_id=child_id,
        session_key=session_key,
        parent_key=parent_key,
        parent_incarnation=parent_incarnation,
        root_key=root_key,
        profile=profile,
        depth=counter("depth", minimum=1),
        attempt_id=text("attempt_id"),
        assignment_digest=digest,
        policy_facts=dict(cast(Mapping[str, Any], facts)),
        runtime_generation=counter("runtime_generation"),
        control_epoch=counter("control_epoch"),
        source_state=source_state,
        parent_source_state=parent_source_state,
        root_source_state=root_source_state,
        run_sequence=counter("run_sequence"),
        state_sequence=counter("state_sequence"),
        current_run_id=optional_text("current_run_id"),
        last_run_id=optional_text("last_run_id"),
        holds_capacity=holds_capacity,
        pending_approval_id=optional_text("pending_approval_id"),
        pending_effect_id=optional_text("pending_effect_id"),
        pending_summary=optional_text("pending_summary"),
        blocked_run_id=optional_text("blocked_run_id"),
        state=state_value,
        outcome=optional_text("outcome"),
        transcript=tuple(transcript),
        background_delivery=background_delivery,
        background_result=background_result,
        created_at=timestamp("created_at"),
        updated_at=timestamp("updated_at"),
    )


class ChildLaunchFence:
    """Synchronous launch reservation that Stop can invalidate."""

    def __init__(self, tree: "DelegationTree", child_id: str) -> None:
        self._tree = tree
        self.child_id = child_id
        self._released = False

    @property
    def cancelled(self) -> bool:
        return self._tree.launch_cancelled(self.child_id)

    def attach(self, task: asyncio.Task[Any]) -> None:
        """Attach the backend task or cancel it before it can run."""
        self._tree.attach_launch_task(self.child_id, task)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._tree.release_launch(self.child_id)


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
        self._background_result_lock = asyncio.Lock()
        self._children: dict[str, ChildAdmission] = {}
        self._attempts: dict[tuple[str, str, str, str], str] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._launching: set[str] = set()
        self._launch_cancelled: set[str] = set()
        self._stopping_sessions: set[str] = set()
        self._control_epochs: dict[str, int] = {}
        self._original_stop_admissions: dict[str, OriginalChatStopAdmission] = {}
        self._recovery_error: str | None = None
        self._restore()

    @classmethod
    def shared(
        cls,
        sessions: SessionManager,
        *,
        policy: DelegationTreePolicy | None = None,
    ) -> DelegationTree:
        """Return the one live child tree for this session owner."""
        with _SHARED_TREE_LOCK:
            tree_ref = _SHARED_TREES.get(sessions)
            tree = tree_ref() if tree_ref is not None else None
            if tree is not None:
                if policy is not None and tree.policy != policy:
                    raise ValueError("shared delegated child policy does not match")
                return tree
            tree = cls(sessions, policy=policy)
            _SHARED_TREES[sessions] = weakref.ref(tree)
            return tree

    def _restore(self) -> None:
        """Load durable child records; fence unresolved live states on restart."""
        try:
            sessions = self.sessions.list_sessions()
        except Exception:
            logger.exception("Delegated child recovery scan failed")
            self._recovery_error = "session index scan failed"
            return
        for raw_info in cast(list[object], sessions):
            if not isinstance(raw_info, Mapping):
                self._recovery_error = "session index contains a malformed row"
                logger.warning("Delegated child recovery found a malformed session row")
                continue
            info = cast(Mapping[str, object], raw_info)
            key = info.get("key")
            if not isinstance(key, str):
                self._recovery_error = "session index contains a malformed key"
                logger.warning("Delegated child recovery found a session without a valid key")
                continue
            if not key.startswith(CHILD_SESSION_PREFIX):
                continue
            try:
                raw_value = self.sessions.read_session_metadata(key)
                if not isinstance(raw_value, Mapping):
                    raise ValueError("child source metadata is missing")
                raw = cast(Mapping[str, object], raw_value)
                metadata = raw.get("metadata")
                if not isinstance(metadata, Mapping):
                    raise ValueError("child source metadata is malformed")
                typed_metadata = cast(Mapping[str, object], metadata)
                payload = typed_metadata.get(DELEGATION_CHILD_METADATA_KEY)
                if not isinstance(payload, Mapping):
                    raise ValueError("delegated child record is missing")
                data = cast(Mapping[str, object], payload)
                admission = _parse_child_admission(key, data)
            except Exception:
                logger.warning("Ignoring admission for malformed delegated child {}", key)
                self._recovery_error = f"malformed child record: {key}"
                continue
            if admission.live:
                admission = replace(
                    admission,
                    state="uncertain",
                    outcome="restart_unresolved",
                    holds_capacity=True,
                    pending_approval_id=None,
                    pending_effect_id=None,
                    pending_summary=None,
                    blocked_run_id=None,
                    state_sequence=admission.state_sequence + 1,
                    updated_at=time.time(),
                )
                if not self._persist(admission):
                    self._recovery_error = f"child uncertainty could not be persisted: {key}"
            self._children[admission.child_id] = admission
            self._control_epochs[admission.root_key] = max(
                self._control_epochs.get(admission.root_key, 0),
                admission.control_epoch,
            )
            self._attempts[
                (
                    admission.parent_key,
                    admission.parent_incarnation,
                    admission.profile,
                    f"{admission.attempt_id}:{admission.assignment_digest}",
                )
            ] = admission.child_id

    @property
    def recovery_error(self) -> str | None:
        """Whether unresolved durable child state currently fences admission."""
        return self._recovery_error

    @property
    def background_result_lock(self) -> asyncio.Lock:
        """Serialize result recovery and journal admission across shared loops."""
        return self._background_result_lock

    async def reconcile_recovery(self) -> bool:
        """Repeat a failed startup scan after the owning store was repaired."""
        async with self._lock:
            if self._recovery_error is None:
                return True
            if self._tasks or self._launching:
                return False
            self._children.clear()
            self._attempts.clear()
            self._control_epochs.clear()
            self._recovery_error = None
            self._restore()
            return self._recovery_error is None

    def child(self, child_id: str) -> ChildAdmission | None:
        return self._children.get(child_id)

    def child_for_session(self, session_key: str) -> ChildAdmission | None:
        return next(
            (child for child in self._children.values() if child.session_key == session_key),
            None,
        )

    def children_of(
        self,
        parent_key: str,
        *,
        parent_incarnation: str | None = None,
    ) -> tuple[ChildAdmission, ...]:
        return tuple(
            child for child in self._children.values()
            if child.parent_key == parent_key
            and (
                parent_incarnation is None
                or child.parent_incarnation == parent_incarnation
            )
        )

    def is_stopping(self, child_id: str) -> bool:
        """Return whether Stop fenced this child or any owning ancestor."""
        child = self._children.get(child_id)
        if child is None:
            return True
        if child.session_key in self._stopping_sessions:
            return True
        parent_key = child.parent_key
        while parent_key:
            if parent_key in self._stopping_sessions:
                return True
            parent = next(
                (item for item in self._children.values() if item.session_key == parent_key),
                None,
            )
            if parent is None:
                break
            parent_key = parent.parent_key
        return False

    def live_descendants(
        self,
        root_key: str,
        *,
        parent_incarnation: str | None = None,
    ) -> tuple[ChildAdmission, ...]:
        return tuple(
            child for child in self._children.values()
            if child.root_key == root_key
            and child.live
            and (
                parent_incarnation is None
                or child.parent_incarnation == parent_incarnation
            )
        )

    def _source_state(self, key: str) -> SourceState | None:
        backend = self.sessions.source_backend(key)
        if backend is None:
            return None
        return backend.source.load_source_snapshot(key).state

    def source_state(self, key: str) -> SourceState | None:
        """Read the canonical source lifetime for one session key."""
        return self._source_state(key)

    def background_results(self) -> tuple[ChildAdmission, ...]:
        """Return every child that owns a durable background result record."""
        return tuple(
            child for child in self._children.values()
            if child.background_result is not None
        )

    def uncertain_children(self) -> tuple[ChildAdmission, ...]:
        """Return children that need explicit owner-evidence settlement."""
        return tuple(
            child for child in self._children.values()
            if child.state == "uncertain"
        )

    def background_delivery_candidates(self) -> tuple[ChildAdmission, ...]:
        """Return settled background children whose result write did not land."""
        return tuple(
            child for child in self._children.values()
            if child.background_delivery is not None
            and child.background_result is None
            and child.current_run_id is not None
            and (child.state == "uncertain" or child.terminal)
        )

    async def record_background_result(
        self,
        intent: PendingBackgroundResult,
    ) -> ChildAdmission | None:
        """Persist one bounded terminal result before the child releases capacity."""
        if type(intent) is not PendingBackgroundResult:
            raise ValueError("Invalid background child result")
        async with self._lock:
            current = self._children.get(intent.child_id)
            if (
                current is None
                or current.background_result is not None
                and not current.background_result.same_result(intent)
                or current.current_run_id != intent.child_run_id
                or current.session_key != intent.child_key
                or current.parent_key != intent.parent_key
            ):
                return None
            parent_state = self._source_state(intent.parent_key)
            stale_reason = (
                "parent_unavailable"
                if parent_state is None
                else "parent_source_changed"
                if (
                    parent_state.session_key != intent.parent_key
                    or parent_state.store_identity != intent.parent_store_identity
                    or parent_state.incarnation != intent.parent_incarnation
                    or parent_state.generation != intent.parent_generation
                )
                else None
            )
            if current.background_result is not None:
                existing = current.background_result
                if existing.state != "pending" or stale_reason is None:
                    return current
                updated = replace(
                    current,
                    background_result=replace(
                        existing,
                        state="stale",
                        stale_reason=stale_reason,
                    ),
                    updated_at=time.time(),
                )
            else:
                updated = replace(
                    current,
                    background_result=replace(
                        intent,
                        state="stale" if stale_reason is not None else "pending",
                        stale_reason=stale_reason,
                    ),
                    updated_at=time.time(),
                )
            if not self._persist(updated):
                return current
            self._children[intent.child_id] = updated
            return updated

    async def mark_background_result(
        self,
        child_id: str,
        *,
        state: Literal["admitted", "stale"],
        journal_id: str | None = None,
        stale_reason: str | None = None,
        published: bool | None = None,
    ) -> ChildAdmission | None:
        """Advance one result disposition without changing the child run."""
        async with self._lock:
            current = self._children.get(child_id)
            if current is None or current.background_result is None:
                return None
            existing = current.background_result
            if existing.state == "admitted" and state != "admitted":
                return current
            if journal_id is not None and (
                len(journal_id) != 32
                or any(character not in "0123456789abcdef" for character in journal_id)
            ):
                raise ValueError("Invalid background result journal identity")
            updated = replace(
                current,
                background_result=replace(
                    existing,
                    state=state,
                    journal_id=journal_id if journal_id is not None else existing.journal_id,
                    stale_reason=stale_reason if state == "stale" else None,
                    published=(
                        published if published is not None else existing.published
                    ),
                ),
                updated_at=time.time(),
            )
            if not self._persist(updated):
                return current
            self._children[child_id] = updated
            return updated

    def capture_child_control_state(self, child_id: str) -> ChildControlState | None:
        """Capture the source lifetimes and epoch seen by an existing child control."""
        child = self._children.get(child_id)
        if child is None or child.source_state is None:
            return None
        try:
            parent = self.sessions.read_session_snapshot(child.parent_key)
            root = self.sessions.read_session_snapshot(child.root_key)
            parent_state = self._source_state(child.parent_key)
            root_state = self._source_state(child.root_key)
            child_state = self._source_state(child.session_key)
        except Exception:
            logger.exception("Delegated child control source could not be read for {}", child_id)
            return None
        if child_state is None:
            return None
        return ChildControlState(
            child_id=child_id,
            parent_key=child.parent_key,
            parent_incarnation=child.parent_incarnation,
            parent_source_state=parent_state,
            parent_present=parent is not None,
            root_key=child.root_key,
            root_source_state=root_state,
            root_present=root is not None,
            child_source_state=child_state,
            child_state_sequence=child.state_sequence,
            control_epoch=child.control_epoch,
        )

    def _child_control_state_matches(self, expected: ChildControlState) -> bool:
        child = self._children.get(expected.child_id)
        if (
            child is None
            or child.parent_key != expected.parent_key
            or child.parent_incarnation != expected.parent_incarnation
            or child.root_key != expected.root_key
            or child.state_sequence != expected.child_state_sequence
            or child.control_epoch != expected.control_epoch
            or child.source_state is None
            or not child.source_state.same_origin(expected.child_source_state)
            or not self._same_origin_or_absent(
                child.parent_source_state,
                expected.parent_source_state,
            )
            or not self._same_origin_or_absent(
                child.root_source_state,
                expected.root_source_state,
            )
        ):
            return False
        try:
            parent = self.sessions.read_session_snapshot(expected.parent_key)
            root = self.sessions.read_session_snapshot(expected.root_key)
            parent_state = self._source_state(expected.parent_key)
            root_state = self._source_state(expected.root_key)
            child_state = self._source_state(child.session_key)
        except Exception:
            return False
        return (
            (parent is not None) == expected.parent_present
            and (root is not None) == expected.root_present
            and parent is not None
            and parent.incarnation == expected.parent_incarnation
            and self._same_origin_or_absent(parent_state, expected.parent_source_state)
            and self._same_origin_or_absent(root_state, expected.root_source_state)
            and child_state is not None
            and child_state.same_origin(expected.child_source_state)
        )

    @staticmethod
    def _same_origin_or_absent(
        current: SourceState | None,
        expected: SourceState | None,
    ) -> bool:
        return (
            current is None and expected is None
            or current is not None and expected is not None and current.same_origin(expected)
        )

    def begin_child_stop(self, expected: ChildControlState) -> ChildStopAdmission | None:
        """Fence one branch before Stop yields to cancellation or storage work."""
        if not self._child_control_state_matches(expected):
            return None
        child = self._children.get(expected.child_id)
        if child is None or not any(
            victim.holds_capacity for victim in self._subtree_locked(child.child_id)
        ):
            return None
        epoch = max(
            child.control_epoch,
            self._control_epochs.get(child.root_key, 0),
        ) + 1
        self._control_epochs[child.root_key] = epoch
        for victim in self._subtree_locked(child.child_id):
            self._stopping_sessions.add(victim.session_key)
        return ChildStopAdmission(expected=expected, control_epoch=epoch)

    def begin_original_chat_stop(
        self,
        *,
        parent_key: str,
        parent_incarnation: str | None,
        parent_source_state: SourceState | None,
        active_root_identity: tuple[asyncio.Task[Any], ...],
        target_turn_id: str | None,
        pending_queue_identity: int | None = None,
    ) -> OriginalChatStopAdmission | None:
        """Fence new child admission synchronously at the existing Stop entrypoint."""
        candidate = OriginalChatStopAdmission(
            parent_key=parent_key,
            parent_incarnation=parent_incarnation,
            parent_source_state=parent_source_state,
            control_epoch=self._control_epochs.get(parent_key, 0) + 1,
            active_root_identity=active_root_identity,
            target_turn_id=target_turn_id,
            pending_queue_identity=pending_queue_identity,
        )
        if not self._original_chat_stop_matches(candidate, active_root_identity):
            return None
        self._control_epochs[parent_key] = candidate.control_epoch
        self._original_stop_admissions[parent_key] = candidate
        self._stopping_sessions.add(parent_key)
        for task in active_root_identity:
            task.add_done_callback(
                lambda _done, admission=candidate: self._maybe_release_original_chat_stop(
                    admission,
                )
            )
        return candidate

    def validate_original_chat_stop(
        self,
        admission: OriginalChatStopAdmission,
        active_root_identity: tuple[asyncio.Task[Any], ...],
    ) -> bool:
        """Reject a stale Stop before it cancels a later source or root task."""
        return self._original_chat_stop_matches(admission, active_root_identity)

    def _original_chat_stop_matches(
        self,
        admission: OriginalChatStopAdmission,
        active_root_identity: tuple[asyncio.Task[Any], ...],
    ) -> bool:
        if (
            self._control_epochs.get(admission.parent_key, 0) + 1
            != admission.control_epoch
            and self._control_epochs.get(admission.parent_key, 0) != admission.control_epoch
        ):
            return False
        if not set(active_root_identity).issubset(admission.active_root_identity):
            return False
        try:
            parent = self.sessions.read_session_snapshot(admission.parent_key)
            source_state = self._source_state(admission.parent_key)
        except Exception:
            return False
        if admission.parent_incarnation is None:
            if parent is not None:
                return False
        elif parent is None or parent.incarnation != admission.parent_incarnation:
            return False
        return self._same_origin_or_absent(source_state, admission.parent_source_state)

    def abort_original_chat_stop(self, admission: OriginalChatStopAdmission) -> None:
        """Release only this Stop's root fence when its snapshot became stale."""
        if self._control_epochs.get(admission.parent_key) == admission.control_epoch:
            if self._original_stop_admissions.get(admission.parent_key) is admission:
                self._original_stop_admissions.pop(admission.parent_key, None)
                self._stopping_sessions.discard(admission.parent_key)

    def _maybe_release_original_chat_stop(
        self,
        admission: OriginalChatStopAdmission,
    ) -> None:
        """Release a root fence only after its captured owners truly settle."""
        if self._original_stop_admissions.get(admission.parent_key) is not admission:
            return
        if any(not task.done() for task in admission.active_root_identity):
            return
        for child in self._children.values():
            if (
                child.parent_key == admission.parent_key
                and child.parent_incarnation == admission.parent_incarnation
                and any(
                    victim.holds_capacity
                    for victim in self._subtree_locked(child.child_id)
                )
            ):
                return
        self._original_stop_admissions.pop(admission.parent_key, None)
        self._stopping_sessions.discard(admission.parent_key)

    async def fence_child_stop(
        self,
        admission: ChildStopAdmission,
    ) -> tuple[ChildAdmission, ...] | None:
        """Persist the stopping projection before cancelling an admitted branch."""
        async with self._lock:
            if not self._child_control_state_matches(admission.expected):
                return None
            victims = self._subtree_locked(admission.expected.child_id)
            return self._mark_stopping_locked(victims, admission.control_epoch)

    def child_session_keys_for_stop(self, child_id: str) -> tuple[str, ...]:
        """Return the exact child branch that an admitted Stop fenced."""
        return tuple(
            victim.session_key
            for victim in self._subtree_locked(child_id)
        )

    def child_session_keys_for_parent_stop(
        self,
        admission: OriginalChatStopAdmission,
    ) -> tuple[str, ...]:
        """Return source keys in the exact parent-incarnation subtree."""
        keys: set[str] = set()
        for child in self._children.values():
            if (
                child.parent_key == admission.parent_key
                and child.parent_incarnation == admission.parent_incarnation
            ):
                keys.update(
                    victim.session_key
                    for victim in self._subtree_locked(child.child_id)
                )
        return tuple(sorted(keys))

    async def fence_original_chat_stop(
        self,
        admission: OriginalChatStopAdmission,
        active_root_identity: tuple[asyncio.Task[Any], ...],
    ) -> tuple[ChildAdmission, ...] | None:
        """Persist the full subtree's stopping state before root cancellation."""
        async with self._lock:
            if not self.validate_original_chat_stop(admission, active_root_identity):
                return None
            direct = [
                child for child in self._children.values()
                if child.parent_key == admission.parent_key
                and child.parent_incarnation == admission.parent_incarnation
            ]
            victims: list[ChildAdmission] = []
            seen: set[str] = set()
            for child in direct:
                for victim in self._subtree_locked(child.child_id):
                    if victim.child_id not in seen:
                        seen.add(victim.child_id)
                        victims.append(victim)
            for victim in victims:
                self._stopping_sessions.add(victim.session_key)
            return self._mark_stopping_locked(victims, admission.control_epoch)

    def _mark_stopping_locked(
        self,
        victims: list[ChildAdmission],
        control_epoch: int,
    ) -> tuple[ChildAdmission, ...]:
        updates: list[ChildAdmission] = []
        for current in victims:
            if not current.holds_capacity:
                if not self._has_current_pending_child_followup(current):
                    continue
                if control_epoch <= current.control_epoch:
                    self._recovery_error = self._recovery_error or (
                        f"child Stop epoch did not advance: {current.child_id}"
                    )
                    continue
                updated = replace(
                    current,
                    control_epoch=control_epoch,
                    state_sequence=current.state_sequence + 1,
                    updated_at=time.time(),
                )
                if not self._persist(updated):
                    self._recovery_error = self._recovery_error or (
                        f"child Stop epoch could not be persisted: {current.child_id}"
                    )
                    continue
                self._children[current.child_id] = updated
                updates.append(updated)
                continue
            if current.state == "stopping" and current.control_epoch == control_epoch:
                updates.append(current)
                continue
            state = "uncertain" if current.state == "uncertain" else "stopping"
            updated = replace(
                current,
                state=state,
                control_epoch=control_epoch,
                state_sequence=current.state_sequence + 1,
                outcome=(current.outcome if state == "uncertain" else "stop_requested"),
                updated_at=time.time(),
            )
            if not self._persist(updated):
                self._recovery_error = self._recovery_error or (
                    f"child stopping state could not be persisted: {current.child_id}"
                )
                continue
            self._children[current.child_id] = updated
            updates.append(updated)
        return tuple(updates)

    def _has_current_pending_child_followup(self, child: ChildAdmission) -> bool:
        """Find a queued row that a terminal child could otherwise replay."""
        if child.source_state is None:
            return False
        backend = self.sessions.source_backend(child.session_key)
        if backend is None:
            return False
        try:
            snapshot = backend.source.load_source_snapshot(child.session_key)
            state = snapshot.state
            if (
                not state.same_origin(child.source_state)
                or state.store_identity != child.source_state.store_identity
                or state.incarnation != child.source_state.incarnation
                or state.generation != child.source_state.generation
            ):
                return False
            metadata_value: object = snapshot.header.get("metadata")
            if not isinstance(metadata_value, dict):
                return False
            metadata = cast(dict[str, object], metadata_value)
            records = parse_pending_followup_records(
                metadata.get(SOURCE_PENDING_FOLLOWUPS_METADATA_KEY, []),
            )
        except Exception:
            return False
        return any(
            intent is not None
            and intent.child_id == child.child_id
            and intent.child_key == child.session_key
            and intent.parent_key == child.parent_key
            and intent.parent_incarnation == child.parent_incarnation
            and intent.child_store_identity == state.store_identity
            and intent.child_incarnation == state.incarnation
            and intent.child_generation == state.generation
            and intent.child_control_epoch == child.control_epoch
            for intent in (record.child_followup for record in records)
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
        expected_parent_source_state: SourceState | None = None,
        background_delivery: BackgroundDeliveryIntent | None = None,
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
        parent_runtime = read_runtime_session(self.sessions, parent_key)
        if parent_runtime.present and not parent_runtime.executable:
            return ChildRefusal(
                "session_unavailable",
                "The parent runtime identity is stale or malformed",
            )
        if expected_parent_source_state is not None:
            try:
                current_parent = self.sessions.read_session_snapshot(parent_key)
                current_parent_state = self._source_state(parent_key)
            except Exception:
                current_parent = None
                current_parent_state = None
            if (
                parent.key != parent_key
                or parent.incarnation != expected_parent_source_state.incarnation
                or current_parent is None
                or current_parent.incarnation != expected_parent_source_state.incarnation
                or current_parent_state is None
                or not current_parent_state.same_origin(expected_parent_source_state)
            ):
                return ChildRefusal(
                    "stale_parent",
                    "The captured parent source lifetime is no longer current",
                )
        attempt_key = (
            parent_key,
            parent.incarnation,
            profile,
            f"{attempt_id}:{digest}",
        )
        async with self._lock:
            if self._recovery_error is not None:
                return ChildRefusal(
                    "reconciliation_required",
                    "Delegated child recovery must be reconciled before new work",
                )
            if parent_key in self._stopping_sessions:
                return ChildRefusal(
                    "session_stopping",
                    "The parent session is stopping its delegated subtree",
                )
            existing = self._attempts.get(attempt_key)
            if existing is not None:
                child = self._children.get(existing)
                if child is not None:
                    return child
            parent_depth = parent_runtime.metadata.delegation_depth
            depth = parent_depth + 1
            if depth > self.policy.max_depth:
                return ChildRefusal(
                    "depth_limit",
                    f"Delegation depth limit {self.policy.max_depth} reached",
                )
            root_key = str(
                parent.metadata.get("delegation_root_key") or parent_key
            )
            parent_backend = self.sessions.source_backend(parent_key)
            root_backend = self.sessions.source_backend(root_key)
            if parent_backend is None or root_backend is None or parent_backend is not root_backend:
                return ChildRefusal(
                    "session_unavailable",
                    "Durable delegation requires the parent's and root's canonical source owner",
                )
            try:
                parent_source_state = parent_backend.source.load_source_snapshot(parent_key).state
                root_source_state = root_backend.source.load_source_snapshot(root_key).state
            except Exception:
                logger.exception("Delegated parent source admission failed for {}", parent_key)
                return ChildRefusal(
                    "session_unavailable",
                    "The parent or root canonical source could not be read",
                )
            if (
                expected_parent_source_state is not None
                and not parent_source_state.same_origin(expected_parent_source_state)
            ):
                return ChildRefusal(
                    "stale_parent",
                    "The captured parent source lifetime is no longer current",
                )
            parent_children = [
                child for child in self._children.values()
                if child.parent_key == parent_key and child.holds_capacity
            ]
            if len(parent_children) >= self.policy.max_live_children_per_parent:
                return ChildRefusal(
                    "child_limit",
                    "This parent already holds its live child limit",
                )
            root_descendants = [
                child for child in self._children.values()
                if child.root_key == root_key and child.holds_capacity
            ]
            if len(root_descendants) >= self.policy.max_live_descendants:
                return ChildRefusal(
                    "root_tree_limit",
                    "This root chat already holds its live descendant limit",
                )
            gateway_children = [
                child for child in self._children.values()
                if child.holds_capacity
            ]
            if len(gateway_children) >= self.policy.max_live_children_per_gateway:
                return ChildRefusal(
                    "gateway_child_limit",
                    "The gateway already holds its live child limit",
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
                control_epoch=self._control_epochs.get(root_key, 0),
                parent_source_state=parent_source_state,
                root_source_state=root_source_state,
                background_delivery=background_delivery,
            )
            try:
                if self.sessions.has_managed_authority:
                    resolved = self.sessions.admit_persistent_write(child_key)
                    backend = self.sessions.source_backend(child_key)
                    if resolved.state != "active" or resolved.mode != "canonical" or backend is None:
                        return ChildRefusal(
                            "session_unavailable",
                            "The child session could not be admitted as canonical history",
                        )
                else:
                    # Offline source-backed sessions have no mode resolver, but
                    # still have one authoritative Jsonl source store. Admit the
                    # new key through that exact store before recording child
                    # metadata; custom stores return no parent source above.
                    legacy = self.sessions.get_or_create(child_key)
                    self.sessions.save(legacy)
                    parent_backend.source.upgrade_source_session(
                        child_key,
                        expected_revision=legacy.revision,
                        expected_generation=legacy.generation,
                        expected_incarnation=legacy.incarnation,
                    )
                    self.sessions.invalidate(child_key)
                    backend = self.sessions.source_backend(child_key)
                if backend is None or backend is not parent_backend:
                    return ChildRefusal(
                        "session_unavailable",
                        "The child session could not be admitted as canonical history",
                    )
                admission = replace(
                    admission,
                    source_state=backend.source.load_source_snapshot(child_key).state,
                )
            except Exception:
                logger.exception("Delegated child {} source admission failed", child_id)
                return ChildRefusal(
                    "session_unavailable",
                    "The child session could not be admitted as canonical history",
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

    async def register_run(self, child_id: str, run_id: str) -> ChildAdmission | None:
        """Persist the next safe run identity before a child turn is launched."""
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("delegated child run id is not filename safe")
        async with self._lock:
            current = self._children.get(child_id)
            if current is None or current.terminal or current.source_state is None:
                return None
            updated = replace(
                current,
                run_sequence=current.run_sequence + 1,
                state_sequence=current.state_sequence + 1,
                current_run_id=run_id,
                last_run_id=run_id,
                updated_at=time.time(),
            )
            if not self._persist(updated):
                return None
            self._children[child_id] = updated
            return updated

    async def begin_followup_run(
        self,
        child_id: str,
        *,
        run_id: str,
        expected_run_id: str,
        expected: ChildControlState,
    ) -> ChildAdmission | None:
        """Claim one new run on a settled healthy child source."""
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("delegated child run id is not filename safe")
        async with self._lock:
            current = self._children.get(child_id)
            if (
                current is None
                or current.state != "completed"
                or current.outcome != "completed"
                or current.holds_capacity
                or current.current_run_id != expected_run_id
                or expected.child_id != child_id
                or not self._child_control_state_matches(expected)
                or self.is_stopping(child_id)
                or self._recovery_error is not None
            ):
                return None
            parent_children = [
                child for child in self._children.values()
                if child.parent_key == current.parent_key and child.holds_capacity
            ]
            root_descendants = [
                child for child in self._children.values()
                if child.root_key == current.root_key and child.holds_capacity
            ]
            gateway_children = [
                child for child in self._children.values() if child.holds_capacity
            ]
            if (
                len(parent_children) >= self.policy.max_live_children_per_parent
                or len(root_descendants) >= self.policy.max_live_descendants
                or len(gateway_children) >= self.policy.max_live_children_per_gateway
            ):
                return None
            updated = replace(
                current,
                state="joining",
                state_sequence=current.state_sequence + 1,
                run_sequence=current.run_sequence + 1,
                current_run_id=run_id,
                last_run_id=run_id,
                holds_capacity=True,
                outcome=None,
                updated_at=time.time(),
            )
            if not self._persist(updated):
                return None
            self._children[child_id] = updated
            return updated

    async def wait_for_approval(
        self,
        child_id: str,
        *,
        approval_id: str,
        effect_id: str,
        summary: str,
        blocked_run_id: str,
    ) -> ChildAdmission | None:
        """Persist an actionable pending approval while retaining capacity."""
        async with self._lock:
            current = self._children.get(child_id)
            if (
                current is None
                or current.state not in {"active", "joining"}
                or not current.holds_capacity
                or current.current_run_id != blocked_run_id
            ):
                return current
            updated = replace(
                current,
                state="waiting_approval",
                state_sequence=current.state_sequence + 1,
                holds_capacity=True,
                pending_approval_id=approval_id,
                pending_effect_id=effect_id,
                pending_summary=summary[:4000],
                blocked_run_id=blocked_run_id,
                outcome="approval_required",
                updated_at=time.time(),
            )
            if not self._persist(updated):
                return None
            self._children[child_id] = updated
            return updated

    async def begin_approval_resume(
        self,
        child_id: str,
        *,
        approval_id: str,
        blocked_run_id: str,
        run_id: str,
        task: asyncio.Task[Any] | None = None,
    ) -> ChildAdmission | None:
        """Claim an authorized decision as one new serialized child run."""
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("delegated child run id is not filename safe")
        async with self._lock:
            current = self._children.get(child_id)
            if (
                current is None
                or current.state != "waiting_approval"
                or not current.holds_capacity
                or current.pending_approval_id != approval_id
                or current.blocked_run_id != blocked_run_id
                or current.current_run_id != blocked_run_id
                or self.is_stopping(child_id)
            ):
                return None
            updated = replace(
                current,
                state="joining",
                state_sequence=current.state_sequence + 1,
                run_sequence=current.run_sequence + 1,
                current_run_id=run_id,
                last_run_id=run_id,
                updated_at=time.time(),
            )
            if not self._persist(updated):
                return None
            self._children[child_id] = updated
            if task is not None:
                self._tasks[child_id] = task
                task.add_done_callback(
                    lambda done, child_id=child_id: self._forget_task(child_id, done)
                )
            return updated

    async def activate_approval_resume(
        self,
        child_id: str,
        *,
        approval_id: str,
        blocked_run_id: str,
        run_id: str,
    ) -> ChildAdmission | None:
        """Publish the consumed approval transition before provider dispatch."""
        async with self._lock:
            current = self._children.get(child_id)
            if (
                current is None
                or current.state != "joining"
                or current.current_run_id != run_id
                or current.pending_approval_id != approval_id
                or current.blocked_run_id != blocked_run_id
                or self.is_stopping(child_id)
            ):
                return None
            updated = replace(
                current,
                state="active",
                state_sequence=current.state_sequence + 1,
                pending_approval_id=None,
                pending_effect_id=None,
                pending_summary=None,
                blocked_run_id=None,
                outcome=None,
                updated_at=time.time(),
            )
            if not self._persist(updated):
                return None
            self._children[child_id] = updated
            return updated

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
        task = self._tasks.get(child_id)
        if task is not None and task.done():
            self._tasks.pop(child_id, None)
        return updated

    async def release(self, child_id: str, *, reason: str = "not_started") -> ChildAdmission | None:
        """Release a reservation only when the child provably did not run."""
        updated = await self._transition(child_id, "refused", outcome=reason)
        self._tasks.pop(child_id, None)
        return updated

    def begin_launch(self, child_id: str) -> ChildLaunchFence:
        """Reserve launch before any await that Stop could interleave."""
        self._launching.add(child_id)
        self._launch_cancelled.discard(child_id)
        return ChildLaunchFence(self, child_id)

    def launch_cancelled(self, child_id: str) -> bool:
        return child_id in self._launch_cancelled

    def attach_launch_task(self, child_id: str, task: asyncio.Task[Any]) -> None:
        if self.launch_cancelled(child_id):
            task.cancel()
            return
        self._tasks[child_id] = task
        task.add_done_callback(
            lambda done: self._forget_task(child_id, done)
        )

    def release_launch(self, child_id: str) -> None:
        self._launching.discard(child_id)
        self._launch_cancelled.discard(child_id)

    def _forget_task(self, child_id: str, task: asyncio.Task[Any]) -> None:
        if self._tasks.get(child_id) is task:
            self._tasks.pop(child_id, None)

    async def attach_task(self, child_id: str, task: asyncio.Task[Any]) -> None:
        async with self._lock:
            self._tasks[child_id] = task
            task.add_done_callback(
                lambda done: self._forget_task(child_id, done)
            )

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
        if not self._persist(updated):
            return False
        self._children[child_id] = updated
        return True

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
            backend = self.sessions.source_backend(child.session_key)
            if backend is not None:
                snapshot = backend.source.load_source_snapshot(child.session_key)
                metadata = snapshot.header["metadata"].get(DELEGATION_CHILD_METADATA_KEY)
                if (
                    not isinstance(metadata, dict)
                    or cast(dict[str, object], metadata).get("child_id") != child_id
                ):
                    return False
                messages = [new_source_event(
                    {"role": "user", "content": assignment}, run_id=child_id,
                )]
                if result_text:
                    messages.append(new_source_event(
                        {"role": "assistant", "content": result_text}, run_id=child_id,
                    ))
                backend.source.append_source_events(child.session_key, snapshot.state, messages)
                self.sessions.invalidate(child.session_key)
                return True

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

    async def stop_all(
        self,
        *,
        settle_timeout: float = STOP_SETTLE_TIMEOUT,
    ) -> tuple[ChildAdmission, ...]:
        """Stop every live child under this owner, including background work."""
        async with self._lock:
            child_ids = [
                child.child_id for child in self._children.values() if child.live
            ]
        stopped: list[ChildAdmission] = []
        for child_id in child_ids:
            stopped.extend(await self.stop_subtree(child_id, settle_timeout=settle_timeout))
        return tuple(stopped)

    async def stop_for_parent(
        self,
        parent_key: str,
        *,
        parent_incarnation: str | None = None,
        admission: OriginalChatStopAdmission | None = None,
        active_root_identity: tuple[asyncio.Task[Any], ...] = (),
        fenced: bool = False,
        settle_timeout: float = STOP_SETTLE_TIMEOUT,
        deadline: float | None = None,
    ) -> tuple[ChildAdmission, ...]:
        """Stop live children admitted under one exact parent incarnation."""
        if admission is None:
            try:
                parent = self.sessions.read_session_snapshot(parent_key)
                source_state = self._source_state(parent_key)
            except Exception:
                return ()
            if parent is None:
                return ()
            admission = self.begin_original_chat_stop(
                parent_key=parent_key,
                parent_incarnation=(parent_incarnation or parent.incarnation),
                parent_source_state=source_state,
                active_root_identity=active_root_identity,
                target_turn_id=None,
            )
        if admission is None:
            return ()
        if not fenced:
            updates = await self.fence_original_chat_stop(admission, active_root_identity)
            if updates is None:
                self.abort_original_chat_stop(admission)
                return ()
        child_ids = [
            child.child_id for child in self._children.values()
            if child.parent_key == parent_key
            and child.parent_incarnation == admission.parent_incarnation
            and any(
                victim.holds_capacity
                for victim in self._subtree_locked(child.child_id)
            )
        ]
        stopped: list[ChildAdmission] = []
        loop = asyncio.get_running_loop()
        settle_deadline = (
            deadline if deadline is not None else loop.time() + max(0.0, settle_timeout)
        )
        victims_by_root = [self._subtree_locked(child_id) for child_id in child_ids]
        for victims in victims_by_root:
            for victim in victims:
                if victim.child_id in self._launching:
                    self._launch_cancelled.add(victim.child_id)
                task = self._tasks.get(victim.child_id)
                if task is not None and not task.done() and not task.cancelling():
                    task.cancel()
        for child_id in child_ids:
            stopped.extend(await self.stop_subtree(
                child_id,
                settle_timeout=settle_timeout,
                fenced=True,
                deadline=settle_deadline,
                preserve_stopping_projection=True,
            ))
        self._maybe_release_original_chat_stop(admission)
        return tuple(stopped)

    async def stop_subtree(
        self,
        child_id: str,
        *,
        settle_timeout: float = STOP_SETTLE_TIMEOUT,
        fenced: bool = False,
        deadline: float | None = None,
        preserve_stopping_projection: bool = False,
    ) -> tuple[ChildAdmission, ...]:
        """Cancel one child and its descendants; siblings remain independent.

        A resistant runner keeps its stopping projection, task handle, and slot.
        """
        if not fenced:
            expected = self.capture_child_control_state(child_id)
            admission = self.begin_child_stop(expected) if expected is not None else None
            if admission is None:
                return ()
            updates = await self.fence_child_stop(admission)
            if updates is None:
                self.abort_child_stop(admission)
                return ()
        async with self._lock:
            victims = self._subtree_locked(child_id)
            self._stopping_sessions.update(victim.session_key for victim in victims)
        stopped: list[ChildAdmission] = []
        retain_fence = False
        try:
            loop = asyncio.get_running_loop()
            settle_deadline = (
                deadline if deadline is not None else loop.time() + max(0.0, settle_timeout)
            )
            tasks = {
                victim.child_id: self._tasks.get(victim.child_id)
                for victim in victims
            }
            for victim in victims:
                if victim.child_id in self._launching:
                    self._launch_cancelled.add(victim.child_id)
                task = tasks[victim.child_id]
                if task is not None and not task.done() and not task.cancelling():
                    task.cancel()
            for victim in victims:
                task = tasks[victim.child_id]
                settled = True
                if task is not None and not task.done():
                    remaining = max(0.0, settle_deadline - loop.time())
                    done, _pending = await asyncio.wait({task}, timeout=remaining)
                    settled = task in done
                if settled:
                    updated = await self.complete(
                        victim.child_id,
                        outcome="stopped_by_owner",
                        state="stopped",
                    )
                else:
                    updated = await self._transition(
                        victim.child_id,
                        "stopping" if preserve_stopping_projection else "uncertain",
                        outcome="stop_unsettled",
                    )
                if updated is not None:
                    stopped.append(updated)
                    if updated.holds_capacity:
                        retain_fence = True
                else:
                    retain_fence = True
        finally:
            if not retain_fence:
                async with self._lock:
                    self._stopping_sessions.difference_update(
                        victim.session_key for victim in victims
                    )
        return tuple(stopped)

    def abort_child_stop(self, admission: ChildStopAdmission) -> None:
        """Release a branch fence when its captured source state became stale."""
        child = self._children.get(admission.expected.child_id)
        if (
            child is None
            or self._control_epochs.get(child.root_key) != admission.control_epoch
        ):
            return
        for victim in self._subtree_locked(child.child_id):
            if victim.state != "stopping":
                self._stopping_sessions.discard(victim.session_key)

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

    async def settle_uncertain(
        self,
        child_id: str,
        *,
        state: Literal["completed", "failed", "stopped"],
        outcome: str,
    ) -> ChildAdmission | None:
        """Close one uncertain child only after the caller proves its run and source."""
        async with self._lock:
            current = self._children.get(child_id)
            if current is None or current.state != "uncertain":
                return None
            updated = replace(
                current,
                state=state,
                state_sequence=current.state_sequence + 1,
                holds_capacity=False,
                pending_approval_id=None,
                pending_effect_id=None,
                pending_summary=None,
                blocked_run_id=None,
                outcome=outcome,
                updated_at=time.time(),
            )
            if not self._persist(updated):
                return current
            self._children[child_id] = updated
        stop_admission = self._original_stop_admissions.get(updated.root_key)
        if stop_admission is not None:
            self._maybe_release_original_chat_stop(stop_admission)
        return updated

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
                state_sequence=current.state_sequence + 1,
                holds_capacity=state in _LIVE_STATES or state == "uncertain",
                pending_approval_id=None,
                pending_effect_id=None,
                pending_summary=None,
                blocked_run_id=None,
                outcome=outcome if outcome is not None else current.outcome,
                updated_at=time.time(),
            )
            if not self._persist(updated):
                return current
            self._children[child_id] = updated
        if not updated.holds_capacity:
            stop_admission = self._original_stop_admissions.get(updated.root_key)
            if stop_admission is not None:
                self._maybe_release_original_chat_stop(stop_admission)
        return updated

    def _persist(self, admission: ChildAdmission, *, parent: Session | None = None) -> bool:
        backend = self.sessions.source_backend(admission.session_key)
        expected = admission.source_state
        if backend is None or expected is None:
            self._recovery_error = self._recovery_error or (
                f"child source owner is unavailable: {admission.session_key}"
            )
            return False
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
            with self.sessions.locked_session_files():
                snapshot = backend.source.load_source_snapshot(admission.session_key)
                if not snapshot.state.same_origin(expected):
                    current = self._children.get(admission.child_id)
                    if current is not None and current.holds_capacity:
                        self._children[admission.child_id] = replace(
                            current,
                            state="uncertain",
                            outcome="child_source_lifetime_changed",
                            holds_capacity=True,
                            updated_at=time.time(),
                        )
                    self._recovery_error = self._recovery_error or (
                        f"child source lifetime changed: {admission.session_key}"
                    )
                    return False
                persisted = self.sessions.upsert_session_metadata(
                    admission.session_key,
                    {
                        DELEGATION_CHILD_METADATA_KEY: admission.metadata(),
                        RUNTIME_SESSION_METADATA_KEY: runtime.payload(),
                        "delegation_root_key": admission.root_key,
                    },
                    create_if_absent=False,
                )
            if not persisted:
                self._recovery_error = self._recovery_error or (
                    f"child metadata write failed: {admission.session_key}"
                )
            return persisted
        except Exception:
            logger.exception(
                "Delegated child {} could not be persisted",
                admission.child_id,
            )
            self._recovery_error = self._recovery_error or (
                f"child metadata write failed: {admission.session_key}"
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
        if admission.state in {"uncertain", "waiting_approval"}:
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
