"""Encode typed outbound events for the WebUI wire protocol."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

from nanobot.bus.outbound_events import (
    ContextCompactionEvent,
    DelegatedChildEvent,
    RecoveryStateEvent,
    RetryStatusEvent,
    StopTargetEvent,
    TurnEndEvent,
)
from nanobot.events import AgentEvent
from nanobot.webui.metadata import WEBUI_TURN_METADATA_KEY


class _ChatWirePayload(TypedDict):
    chat_id: str
    turn_id: NotRequired[str]


class RecoveryStateWirePayload(_ChatWirePayload):
    event: Literal["recovery_state"]
    status: str
    recovery_id: str
    attempts: int
    reason: NotRequired[str]
    can_continue: NotRequired[bool]


class RetryStatusWirePayload(_ChatWirePayload):
    event: Literal["retry_status"]
    state: str
    attempt: int
    max_attempts: NotRequired[int]
    error_kind: str
    retry_after_s: NotRequired[float]


class TurnEndWirePayload(_ChatWirePayload):
    event: Literal["turn_end"]
    latency_ms: NotRequired[int]
    goal_state: NotRequired[dict[str, Any]]
    usage: NotRequired[dict[str, int]]
    round_usages: NotRequired[list[dict[str, int]]]
    context_window_tokens: NotRequired[int]
    outcome: NotRequired[str]
    failure_kind: NotRequired[str]
    failure_error_kind: NotRequired[str]
    failure_attempts: NotRequired[int]
    failure_message: NotRequired[str]


class ContextCompactionWirePayload(_ChatWirePayload):
    event: Literal["context_compaction"]
    compaction_id: str
    phase: Literal["started", "succeeded", "failed", "cancelled"]


class DelegatedChildWirePayload(TypedDict):
    event: Literal["delegated_child"]
    chat_id: str
    parent_session_key: str
    child_session_key: str
    child_id: str
    profile: str
    state: str
    runtime_generation: int
    turn_id: str | None
    task_label: str
    task_label_truncated: bool
    result_summary: str | None
    result_summary_truncated: bool
    outcome: str | None
    parent_run_id: str | None
    child_run_id: str | None
    parent_store_identity: str | None
    parent_incarnation: str | None
    parent_generation: int
    root_key: str | None
    root_store_identity: str | None
    root_incarnation: str | None
    root_generation: int
    child_store_identity: str | None
    child_incarnation: str | None
    child_generation: int
    run_sequence: int
    state_sequence: int
    control_epoch: int
    holds_capacity: bool
    can_stop: bool
    owned_work: bool
    pending_approval_id: str | None
    blocked_run_id: str | None
    result_disposition: str | None


class StopTargetWirePayload(TypedDict):
    event: Literal["stop_target"]
    chat_id: str
    stop_target: dict[str, object] | None


WebUIWirePayload: TypeAlias = (
    RetryStatusWirePayload
    | ContextCompactionWirePayload
    | RecoveryStateWirePayload
    | TurnEndWirePayload
    | DelegatedChildWirePayload
    | StopTargetWirePayload
)
WebUIWirePersistence: TypeAlias = Literal[
    "transient",
    "turn_activity",
    "turn_complete",
]


def encode_recovery_state(
    chat_id: str,
    event: RecoveryStateEvent,
) -> RecoveryStateWirePayload:
    """Project one transient recovery transition onto its stable wire shape."""
    payload: RecoveryStateWirePayload = {
        "event": "recovery_state",
        "chat_id": chat_id,
        "status": event.status,
        "recovery_id": event.recovery_id,
        "attempts": event.attempts,
    }
    if event.reason:
        payload["reason"] = event.reason
    if event.can_continue is not None:
        payload["can_continue"] = event.can_continue
    return payload


def encode_context_compaction(
    chat_id: str,
    event: ContextCompactionEvent,
) -> ContextCompactionWirePayload:
    """Project one summary-free compaction transition onto its stable wire shape."""
    payload: ContextCompactionWirePayload = {
        "event": "context_compaction",
        "chat_id": chat_id,
        "compaction_id": event.compaction_id,
        "phase": event.phase,
    }
    return payload


def encode_delegated_child(
    chat_id: str,
    event: DelegatedChildEvent,
) -> DelegatedChildWirePayload:
    """Project one child activity snapshot with explicit nullable facts and counters."""
    return {
        "event": "delegated_child",
        "chat_id": chat_id,
        "parent_session_key": event.parent_session_key,
        "child_session_key": event.child_session_key,
        "child_id": event.child_id,
        "profile": event.profile,
        "state": event.state,
        "runtime_generation": event.runtime_generation,
        "turn_id": event.turn_id,
        "task_label": event.task_label,
        "task_label_truncated": event.task_label_truncated,
        "result_summary": event.result_summary,
        "result_summary_truncated": event.result_summary_truncated,
        "outcome": event.outcome,
        "parent_run_id": event.parent_run_id,
        "child_run_id": event.child_run_id,
        "parent_store_identity": event.parent_store_identity,
        "parent_incarnation": event.parent_incarnation,
        "parent_generation": event.parent_generation,
        "root_key": event.root_key,
        "root_store_identity": event.root_store_identity,
        "root_incarnation": event.root_incarnation,
        "root_generation": event.root_generation,
        "child_store_identity": event.child_store_identity,
        "child_incarnation": event.child_incarnation,
        "child_generation": event.child_generation,
        "run_sequence": event.run_sequence,
        "state_sequence": event.state_sequence,
        "control_epoch": event.control_epoch,
        "holds_capacity": event.holds_capacity,
        "can_stop": event.can_stop,
        "owned_work": event.owned_work,
        "pending_approval_id": event.pending_approval_id,
        "blocked_run_id": event.blocked_run_id,
        "result_disposition": event.result_disposition,
    }


def encode_stop_target(chat_id: str, event: StopTargetEvent) -> StopTargetWirePayload:
    """Encode a live Stop target as a transient, allowlisted event."""
    return {
        "event": "stop_target",
        "chat_id": chat_id,
        "stop_target": event.stop_target,
    }


@dataclass(frozen=True)
class NotificationProjection:
    """Allowlisted public payload and its durability policy."""

    payload: WebUIWirePayload
    persistence: WebUIWirePersistence = "transient"
    deliver_offline: bool = False
    attach_turn_metadata: bool = False


def project_notification(
    chat_id: str, event: AgentEvent | None, metadata: Mapping[str, object] | None = None,
) -> NotificationProjection | None:
    """Keep notification serialization and persistence decisions at one boundary."""
    if isinstance(event, ContextCompactionEvent):
        return NotificationProjection(
            encode_context_compaction(chat_id, event),
            persistence="transient" if event.phase == "started" else "turn_activity",
            deliver_offline=True,
            attach_turn_metadata=True,
        )
    if isinstance(event, RetryStatusEvent):
        return NotificationProjection(encode_retry_status(chat_id, event, metadata))
    if isinstance(event, RecoveryStateEvent):
        return NotificationProjection(encode_recovery_state(chat_id, event))
    return None


def encode_retry_status(
    chat_id: str,
    event: RetryStatusEvent,
    metadata: Mapping[str, object] | None = None,
) -> RetryStatusWirePayload:
    """Project one transient retry transition onto its stable wire shape."""
    payload: RetryStatusWirePayload = {
        "event": "retry_status",
        "chat_id": chat_id,
        "state": event.state,
        "attempt": event.attempt,
        "error_kind": event.error_kind,
    }
    turn_id = (metadata or {}).get(WEBUI_TURN_METADATA_KEY)
    if isinstance(turn_id, str) and turn_id:
        payload["turn_id"] = turn_id
    if event.max_attempts is not None:
        payload["max_attempts"] = event.max_attempts
    if event.next_retry_at is not None:
        payload["retry_after_s"] = max(0.0, event.next_retry_at - time.time())
    return payload
def encode_turn_end(
    chat_id: str,
    event: TurnEndEvent,
    metadata: Mapping[str, object] | None = None,
) -> TurnEndWirePayload:
    """Project a completed turn without leaking internal metadata onto the wire."""
    payload: TurnEndWirePayload = {
        "event": "turn_end",
        "chat_id": chat_id,
    }
    turn_id = (metadata or {}).get(WEBUI_TURN_METADATA_KEY)
    if isinstance(turn_id, str) and turn_id:
        payload["turn_id"] = turn_id
    if event.latency_ms is not None:
        payload["latency_ms"] = int(event.latency_ms)
    if event.goal_state is not None:
        payload["goal_state"] = event.goal_state
    if event.usage is not None:
        payload["usage"] = event.usage.to_turn_dict()
    if event.round_usages:
        payload["round_usages"] = [item.to_turn_dict() for item in event.round_usages]
    if event.context_window_tokens is not None:
        payload["context_window_tokens"] = int(event.context_window_tokens)
    if event.outcome != "completed":
        payload["outcome"] = event.outcome
    if event.failure_kind is not None:
        payload["failure_kind"] = event.failure_kind
    if event.failure_error_kind is not None:
        payload["failure_error_kind"] = event.failure_error_kind
    if event.failure_attempts is not None:
        payload["failure_attempts"] = int(event.failure_attempts)
    if event.failure_message is not None:
        payload["failure_message"] = event.failure_message
    return payload
