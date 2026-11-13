"""Typed outbound events carried by :class:`OutboundMessage`.

The message bus still transports :class:`nanobot.bus.events.OutboundMessage`
because channels need chat routing fields. Runtime/UI semantics live on the
message's explicit ``event`` field rather than in reserved metadata flags.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from nanobot.bus.events import OutboundMessage
from nanobot.events import AgentEvent
from nanobot.events import ContextCompactionEvent as ContextCompactionEvent
from nanobot.events import RecoveryStateEvent as RecoveryStateEvent
from nanobot.events import RetryStatusEvent as RetryStatusEvent
from nanobot.events import RetryWaitEvent as RetryWaitEvent
from nanobot.providers.base import LLMUsage

if TYPE_CHECKING:
    from nanobot.security.desktop_permissions import WorkspacePermissionProjection
    from nanobot.workspace_access_recovery import FolderAccessRequest, WorkspaceScopePayload


@dataclass(frozen=True)
class ProgressEvent(AgentEvent):
    content: str = ""
    tool_hint: bool = False
    reasoning: bool = False
    reasoning_delta: bool = False
    reasoning_end: bool = False
    stream_id: str | None = None
    tool_events: list[dict[str, Any]] | None = None
    file_edit_events: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class FileEditEvent(ProgressEvent):
    """File activity whose snapshot collection requires an interested consumer."""


@dataclass(frozen=True)
class StreamDeltaEvent(AgentEvent):
    content: str = ""
    stream_id: str | None = None


@dataclass(frozen=True)
class StreamEndEvent(AgentEvent):
    content: str = ""
    stream_id: str | None = None
    resuming: bool = False
    merge_next: bool = False


@dataclass(frozen=True)
class StreamedResponseEvent(AgentEvent):
    pass


@dataclass(frozen=True)
class TurnEndEvent(AgentEvent):
    latency_ms: int | None = None
    goal_state: dict[str, Any] | None = None
    usage: LLMUsage | None = None
    round_usages: tuple[LLMUsage, ...] = ()
    context_window_tokens: int | None = None
    outcome: str = "completed"
    failure_kind: str | None = None
    failure_error_kind: str | None = None
    failure_attempts: int | None = None
    failure_message: str | None = None


@dataclass(frozen=True)
class GoalStatusEvent(AgentEvent):
    status: str
    started_at: float | None = None


@dataclass(frozen=True)
class GoalStateSyncEvent(AgentEvent):
    goal_state: dict[str, Any]


@dataclass(frozen=True)
class MessageAdmittedEvent(AgentEvent):
    """The session owner committed one submitted input attempt."""

    turn_id: str | None = None
    attempt_id: str | None = None
    session_key: str | None = None
    session_incarnation: str | None = None
    active_turn_id: str | None = None
    starts_turn: bool = True


@dataclass(frozen=True)
class DelegatedChildEvent(AgentEvent):
    """Typed, durable reference from a parent activity row to one child."""

    parent_session_key: str = ""
    child_session_key: str = ""
    child_id: str = ""
    profile: str = "delegated"
    state: str = ""
    outcome: str | None = None
    runtime_generation: int = 0
    turn_id: str | None = None
    task_label: str = ""
    task_label_truncated: bool = False
    result_summary: str | None = None
    result_summary_truncated: bool = False
    parent_run_id: str | None = None
    child_run_id: str | None = None
    parent_store_identity: str | None = None
    parent_incarnation: str | None = None
    parent_generation: int = 0
    root_key: str | None = None
    root_store_identity: str | None = None
    root_incarnation: str | None = None
    root_generation: int = 0
    child_store_identity: str | None = None
    child_incarnation: str | None = None
    child_generation: int = 0
    run_sequence: int = 0
    state_sequence: int = 0
    control_epoch: int = 0
    holds_capacity: bool = False
    can_stop: bool = False
    owned_work: bool = False
    pending_approval_id: str | None = None
    blocked_run_id: str | None = None
    result_disposition: str | None = None


@dataclass(frozen=True)
class SessionUpdatedEvent(AgentEvent):
    scope: str | None = None
    workspace_scope: WorkspaceScopePayload | None = None
    workspace_access_request: FolderAccessRequest | None = None
    workspace_permissions: WorkspacePermissionProjection | None = None


@dataclass(frozen=True)
class UserInputEvent(AgentEvent):
    """A user-input row projected by an edge adapter."""

    content: str
    created_at_ms: int
    provenance: dict[str, Any]


@dataclass(frozen=True)
class WebUIAdmissionRejectedEvent(AgentEvent):
    """A previously acknowledged WebUI request lost its turn before injection."""

    detail: str
    reason: str
    turn_id: str | None = None


@dataclass(frozen=True)
class StopTargetEvent(AgentEvent):
    """Transient server-derived original-chat Stop target for current views."""

    stop_target: dict[str, object] | None


@dataclass(frozen=True)
class RuntimeModelUpdatedEvent(AgentEvent):
    model: str | None
    model_preset: str | None = None


@dataclass(frozen=True)
class TurnModelUpdatedEvent(AgentEvent):
    """The canonical preset and concrete model handling one chat turn."""

    model: str
    model_preset: str | None = None
    context_window_tokens: int | None = None
    fallback: bool = False


def outbound_message_for_event(
    *,
    channel: str,
    chat_id: str,
    event: AgentEvent,
    content: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OutboundMessage:
    """Build an :class:`OutboundMessage` for a typed event."""

    return OutboundMessage(
        channel=channel,
        chat_id=chat_id,
        content=_event_content(event) if content is None else content,
        event=event,
        metadata=dict(metadata or {}),
    )


def replace_outbound_event(
    msg: OutboundMessage,
    event: AgentEvent,
    *,
    content: str | None = None,
) -> OutboundMessage:
    """Return *msg* with a new event and optional content."""

    return replace(
        msg,
        content=_event_content(event) if content is None else content,
        event=event,
    )


def _event_content(event: AgentEvent) -> str:
    if isinstance(
        event,
        ProgressEvent | RetryWaitEvent | StreamDeltaEvent | StreamEndEvent | UserInputEvent,
    ):
        return event.content
    if isinstance(event, ContextCompactionEvent):
        if event.phase == "started":
            return "Compressing context…"
        if event.phase == "failed":
            return "Unable to compact context."
        if event.phase == "cancelled":
            return "Context compaction cancelled."
        return "Context compacted."
    return ""
