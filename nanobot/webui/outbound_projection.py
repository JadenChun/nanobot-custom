"""Project agent runtime events onto the WebUI wire protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.outbound_events import (
    ContextCompactionEvent,
    DelegatedChildEvent,
    GoalStateSyncEvent,
    GoalStatusEvent,
    MessageAdmittedEvent,
    ProgressEvent,
    RetryStatusEvent,
    RetryWaitEvent,
    RuntimeModelUpdatedEvent,
    SessionUpdatedEvent,
    TurnEndEvent,
    TurnModelUpdatedEvent,
    UserInputEvent,
    WebUIAdmissionRejectedEvent,
)
from nanobot.session.webui_turns import clear_websocket_turn_if_current
from nanobot.webui.metadata import (
    WEBSOCKET_TURN_OWNER_METADATA_KEY,
    WEBUI_SYSTEM_COMMAND_TURN_PREFIX,
    WEBUI_TURN_METADATA_KEY,
)
from nanobot.webui.outbound_wire import (
    DelegatedChildWirePayload,
    WebUIWirePayload,
    WebUIWirePersistence,
    encode_turn_end,
    project_notification,
)
from nanobot.webui.session_identity import webui_session_key
from nanobot.webui.session_projection import WebUISessionProjection
from nanobot.workspace_access_recovery import FolderAccessRequest, WorkspaceScopePayload

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection

    from nanobot.security.desktop_permissions import WorkspacePermissionProjection

class WebUIOutboundTransport(Protocol):
    """Wire operations required by the outbound application projector."""

    def webui_subscribers(self, chat_id: str) -> tuple[ServerConnection, ...]: ...

    async def send_runtime_model_updated(
        self,
        *,
        model_name: str | None,
        model_preset: str | None = None,
    ) -> None: ...

    async def send_turn_model_updated(
        self,
        chat_id: str,
        *,
        model_name: str,
        model_preset: str | None = None,
        context_window_tokens: int | None = None,
        fallback: bool = False,
    ) -> None: ...

    async def send_user_input(
        self,
        chat_id: str,
        *,
        content: str,
        created_at_ms: int,
        provenance: dict[str, Any],
    ) -> None: ...

    async def send_admission_rejected(
        self,
        chat_id: str,
        *,
        detail: str,
        reason: str,
        turn_id: str | None = None,
    ) -> None: ...

    async def send_payload(
        self,
        chat_id: str,
        payload: WebUIWirePayload,
        *,
        persistence: WebUIWirePersistence,
        metadata: dict[str, Any] | None = None,
        turn_owner: str | None = None,
    ) -> None: ...

    async def send_goal_state(self, chat_id: str, blob: dict[str, Any]) -> None: ...

    async def send_goal_status(
        self,
        chat_id: str,
        status: str,
        *,
        started_at: float | None = None,
        turn_id: str | None = None,
    ) -> None: ...

    async def send_session_updated(
        self,
        chat_id: str,
        *,
        scope: str | None = None,
        workspace_scope: WorkspaceScopePayload | None = None,
        workspace_access_request: FolderAccessRequest | None = None,
        workspace_permissions: WorkspacePermissionProjection | None = None,
    ) -> None: ...

    async def send_file_edit_events(
        self,
        chat_id: str,
        edits: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    async def send_projected_message(
        self,
        msg: OutboundMessage,
        progress_event: ProgressEvent | None,
    ) -> None: ...


class WebUIOutboundProjector:
    """Interpret runtime events without coupling that state machine to the channel."""

    def __init__(
        self,
        transport: WebUIOutboundTransport,
        session_projection: WebUISessionProjection,
    ) -> None:
        self._transport = transport
        self._session_projection = session_projection

    async def hydrate(self, chat_id: str) -> None:
        """Replay reconnect state through the existing stable wire operations."""
        for event in self._session_projection.hydration_events(
            webui_session_key(chat_id),
            chat_id,
        ):
            if event["event"] == "goal_state":
                await self._transport.send_goal_state(chat_id, event["goal_state"])
                continue
            await self._transport.send_goal_status(
                chat_id,
                "running",
                started_at=event["started_at"],
                turn_id=event.get("turn_id"),
            )

    async def send(self, msg: OutboundMessage) -> None:
        event = msg.event
        if isinstance(event, RetryWaitEvent):
            return
        progress_event = event if isinstance(event, ProgressEvent) else None
        if isinstance(event, RuntimeModelUpdatedEvent):
            await self._transport.send_runtime_model_updated(
                model_name=event.model,
                model_preset=event.model_preset,
            )
            return

        conns = list(self._transport.webui_subscribers(msg.chat_id))
        if not conns:
            quiet_events = (
                ProgressEvent,
                RetryStatusEvent,
                UserInputEvent,
                TurnEndEvent,
                SessionUpdatedEvent,
                GoalStatusEvent,
                GoalStateSyncEvent,
                ContextCompactionEvent,
                WebUIAdmissionRejectedEvent,
            )
            log = (
                logger.debug
                if isinstance(event, quiet_events)
                else logger.warning
            )
            log("no active subscribers for chat_id={}", msg.chat_id)

        if isinstance(event, TurnModelUpdatedEvent):
            if conns:
                await self._transport.send_turn_model_updated(
                    msg.chat_id,
                    model_name=event.model,
                    model_preset=event.model_preset,
                    context_window_tokens=event.context_window_tokens,
                    fallback=event.fallback,
                )
            return
        if isinstance(event, DelegatedChildEvent):
            if conns:
                payload = cast(DelegatedChildWirePayload, {
                    "event": "delegated_child",
                    "chat_id": msg.chat_id,
                    "parent_session_key": event.parent_session_key,
                    "child_session_key": event.child_session_key,
                    "child_id": event.child_id,
                    "profile": event.profile,
                    "state": event.state,
                    "runtime_generation": event.runtime_generation,
                    **({"outcome": event.outcome} if event.outcome else {}),
                    **({"turn_id": event.turn_id} if event.turn_id else {}),
                    **(
                        {"parent_run_id": event.parent_run_id}
                        if event.parent_run_id else {}
                    ),
                    **(
                        {"child_run_id": event.child_run_id}
                        if event.child_run_id else {}
                    ),
                    **(
                        {"parent_store_identity": event.parent_store_identity}
                        if event.parent_store_identity else {}
                    ),
                    **(
                        {"parent_incarnation": event.parent_incarnation}
                        if event.parent_incarnation else {}
                    ),
                    **(
                        {"parent_generation": event.parent_generation}
                        if event.parent_generation else {}
                    ),
                    **({"root_key": event.root_key} if event.root_key else {}),
                    **(
                        {"root_store_identity": event.root_store_identity}
                        if event.root_store_identity else {}
                    ),
                    **(
                        {"root_incarnation": event.root_incarnation}
                        if event.root_incarnation else {}
                    ),
                    **(
                        {"root_generation": event.root_generation}
                        if event.root_generation else {}
                    ),
                    **(
                        {"child_store_identity": event.child_store_identity}
                        if event.child_store_identity else {}
                    ),
                    **(
                        {"child_incarnation": event.child_incarnation}
                        if event.child_incarnation else {}
                    ),
                    **(
                        {"child_generation": event.child_generation}
                        if event.child_generation else {}
                    ),
                    **({"run_sequence": event.run_sequence} if event.run_sequence else {}),
                    **(
                        {"state_sequence": event.state_sequence}
                        if event.state_sequence else {}
                    ),
                    **(
                        {"control_epoch": event.control_epoch}
                        if event.control_epoch else {}
                    ),
                    "holds_capacity": event.holds_capacity,
                    "can_stop": event.can_stop,
                    "owned_work": event.owned_work,
                    **(
                        {"pending_approval_id": event.pending_approval_id}
                        if event.pending_approval_id else {}
                    ),
                    **(
                        {"blocked_run_id": event.blocked_run_id}
                        if event.blocked_run_id else {}
                    ),
                    **(
                        {"result_disposition": event.result_disposition}
                        if event.result_disposition else {}
                    ),
                })
                await self._transport.send_payload(
                    msg.chat_id,
                    cast(WebUIWirePayload, payload),
                    persistence="turn_activity",
                )
            return
        if isinstance(event, MessageAdmittedEvent):
            if conns:
                await self._transport.send_payload(
                    msg.chat_id,
                    cast(WebUIWirePayload, {
                        "event": "message_admitted",
                        "chat_id": msg.chat_id,
                        **({"turn_id": event.turn_id} if event.turn_id else {}),
                        **({"attempt_id": event.attempt_id} if event.attempt_id else {}),
                        **({"session_key": event.session_key} if event.session_key else {}),
                        **(
                            {"session_incarnation": event.session_incarnation}
                            if event.session_incarnation else {}
                        ),
                        **(
                            {"active_turn_id": event.active_turn_id}
                            if event.active_turn_id else {}
                        ),
                        "starts_turn": event.starts_turn,
                    }),
                    persistence="transient",
                )
            return
        if isinstance(event, UserInputEvent):
            if conns:
                await self._transport.send_user_input(
                    msg.chat_id,
                    content=event.content,
                    created_at_ms=event.created_at_ms,
                    provenance=event.provenance,
                )
            return
        if isinstance(event, WebUIAdmissionRejectedEvent):
            if conns:
                await self._transport.send_admission_rejected(
                    msg.chat_id,
                    detail=event.detail,
                    reason=event.reason,
                    turn_id=event.turn_id,
                )
            return
        notification = project_notification(msg.chat_id, event, msg.metadata)
        if notification is not None:
            if conns or notification.deliver_offline:
                kwargs: dict[str, Any] = (
                    {"metadata": msg.metadata} if notification.attach_turn_metadata else {}
                )
                await self._transport.send_payload(
                    msg.chat_id, notification.payload,
                    persistence=notification.persistence,
                    **kwargs,
                )
            return
        if isinstance(event, GoalStateSyncEvent):
            if conns:
                await self._transport.send_goal_state(
                    msg.chat_id,
                    event.goal_state or {"active": False},
                )
            return
        if isinstance(event, GoalStatusEvent):
            turn_id = (msg.metadata or {}).get(WEBUI_TURN_METADATA_KEY)
            current_turn_id = turn_id if isinstance(turn_id, str) else None
            turn_owner = (msg.metadata or {}).get(WEBSOCKET_TURN_OWNER_METADATA_KEY)
            current_turn_owner = turn_owner if isinstance(turn_owner, str) else None
            try:
                if conns and event.status in ("running", "idle"):
                    await self._transport.send_goal_status(
                        msg.chat_id,
                        event.status,
                        started_at=event.started_at,
                        turn_id=current_turn_id,
                    )
            finally:
                if event.status == "idle":
                    clear_websocket_turn_if_current(
                        msg.chat_id,
                        current_turn_owner,
                        preserve_persistence_failure=True,
                    )
            return
        if isinstance(event, TurnEndEvent):
            turn_id = (msg.metadata or {}).get(WEBUI_TURN_METADATA_KEY)
            session_update_scope = (
                "metadata"
                if isinstance(turn_id, str)
                and turn_id.startswith(WEBUI_SYSTEM_COMMAND_TURN_PREFIX)
                else "thread"
            )
            turn_owner = (msg.metadata or {}).get(WEBSOCKET_TURN_OWNER_METADATA_KEY)
            await self._transport.send_payload(
                msg.chat_id,
                encode_turn_end(msg.chat_id, event, msg.metadata),
                persistence="turn_complete",
                metadata=msg.metadata,
                turn_owner=turn_owner if isinstance(turn_owner, str) else None,
            )
            await self._transport.send_session_updated(msg.chat_id, scope=session_update_scope)
            return
        if isinstance(event, SessionUpdatedEvent):
            if conns:
                await self._transport.send_session_updated(
                    msg.chat_id,
                    scope=event.scope,
                    workspace_scope=event.workspace_scope,
                    workspace_access_request=event.workspace_access_request,
                    workspace_permissions=event.workspace_permissions,
                )
            return
        if progress_event and progress_event.file_edit_events:
            await self._transport.send_file_edit_events(
                msg.chat_id,
                progress_event.file_edit_events,
                msg.metadata,
            )
            return
        await self._transport.send_projected_message(msg, progress_event)
