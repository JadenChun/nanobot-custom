"""Explicit audience policy for operation notifications."""

from typing import Literal

from kohaku.bus.outbound_events import (
    ExternalApprovalEvent,
    FileEditEvent,
    ProgressEvent,
    RetryStatusEvent,
    SessionUpdatedEvent,
    StreamDeltaEvent,
    StreamEndEvent,
)
from kohaku.events import AgentEvent, ContextCompactionEvent, RecoveryStateEvent, RetryWaitEvent

NotificationAudience = Literal["channel", "lifecycle", "interactive"]

NOTIFICATION_AUDIENCES: dict[type[AgentEvent], NotificationAudience] = {
    ProgressEvent: "lifecycle",
    FileEditEvent: "lifecycle",
    StreamDeltaEvent: "channel",
    StreamEndEvent: "channel",
    ContextCompactionEvent: "channel",
    RetryWaitEvent: "lifecycle",
    RecoveryStateEvent: "interactive",
    RetryStatusEvent: "interactive",
    ExternalApprovalEvent: "interactive",
    SessionUpdatedEvent: "interactive",
}


def notification_is_deliverable(
    event_type: type[AgentEvent], *, channel: str, publish_lifecycle: bool,
) -> bool:
    """Admit operation notifications to a channel only by explicit policy."""
    audience = NOTIFICATION_AUDIENCES.get(event_type)
    if audience is None:
        return False
    if audience == "lifecycle":
        return publish_lifecycle
    if audience == "interactive":
        return channel == "websocket" and (
            event_type not in {RetryStatusEvent, ExternalApprovalEvent, SessionUpdatedEvent}
            or publish_lifecycle
        )
    return True
