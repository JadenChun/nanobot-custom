"""Application orchestration for typed WebUI WebSocket commands."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from loguru import logger
from websockets.asyncio.server import ServerConnection

from nanobot.agent.stop_target import StopExpectation, parse_stop_expectation
from nanobot.agent.turn import ApprovalDecision
from nanobot.attachments.models import AttachmentRecord
from nanobot.bus.events import (
    INBOUND_META_STOP_CLIENT,
    INBOUND_META_STOP_EXPECTATION,
    INBOUND_META_USER_SHELL,
)
from nanobot.command.builtin import USER_SHELL_COMMAND, builtin_command_starts_agent_turn
from nanobot.runtime_context import (
    RUNTIME_CONTEXT_INPUT_META,
    WEBUI_QUOTE_METADATA,
    RuntimeContextBlock,
    webui_quote_runtime_context,
)
from nanobot.security.workspace_access import (
    WORKSPACE_SCOPE_METADATA_KEY,
    WorkspaceScope,
    WorkspaceScopeError,
)
from nanobot.session.webui_turns import (
    activate_queued_websocket_turn,
    bind_queued_websocket_turn_scope,
    clear_websocket_turn_if_current,
    clear_websocket_turns,
    register_queued_websocket_turn_if_idle,
    websocket_turn_id,
    websocket_turn_reserved,
    websocket_turn_scope,
    websocket_turn_wall_started_at,
)
from nanobot.utils.helpers import safe_filename
from nanobot.webui.cli_apps_api import normalize_cli_app_mentions
from nanobot.webui.forking import handle_webui_fork_chat
from nanobot.webui.gateway_services import GatewayServices
from nanobot.webui.mcp_presets_api import normalize_mcp_preset_mentions
from nanobot.webui.metadata import (
    WEBSOCKET_TURN_OWNER_METADATA_KEY,
    WEBUI_INSTALLATION_METADATA_KEY,
    WEBUI_REQUEST_INTENT_METADATA_KEY,
    WEBUI_SOURCE_TOKEN_METADATA_KEY,
    WEBUI_TARGET_TURN_METADATA_KEY,
)
from nanobot.webui.replay_store import DurableWebUIReplay
from nanobot.webui.session_access import (
    SessionMention,
    WebuiSessionAccess,
    session_mentions_runtime_context,
)
from nanobot.webui.session_identity import (
    WEBUI_SESSION_STORAGE_PREFIX,
    is_valid_webui_chat_id,
    webui_session_key,
)
from nanobot.webui.sidebar_state import write_webui_sidebar_state
from nanobot.webui.temporary_chats import TemporaryChatError
from nanobot.webui.transcription_ws import webui_transcription_event
from nanobot.workspace_access_recovery import FolderAccessRequest, WorkspaceScopePayload

if TYPE_CHECKING:
    from nanobot.security.desktop_permissions import WorkspacePermissionProjection

_WEBUI_REQUEST_CACHE_TTL_S = 5 * 60.0
_WEBUI_REQUEST_CACHE_MAX = 256
_PREPARE_FULL_ACTION = "workspace.access.prepare_full"
_MOBILE_MUTATION_ACTIONS = frozenset({
    "sidebar.update",
    "recovery.continue",
    "recovery.dismiss",
})
_SOURCE_BOUND_PERMISSION_ACTIONS = frozenset({
    "workspace.access.add_folder",
    "workspace.access.allow_once",
    "workspace.access.cancel",
    "workspace.access.continue",
    "workspace.access.mode.set",
    "workspace.access.resolve",
})


REQUEST_SETTINGS_VERSION = 1


_REQUEST_SETTINGS_FIELDS = frozenset({
    "request_settings_version",
    "request_intent",
    "target_turn_id",
})

_MESSAGE_LEVEL_CONTROL_FIELDS = frozenset({
    "workspace_scope",
    "source_token",
    "full_confirmation_id",
})


def _is_stop_control(envelope: dict[str, Any]) -> bool:
    """Return whether the envelope is the exact Stop control command."""
    content = envelope.get("content")
    return isinstance(content, str) and content.strip().lower() == "/stop"


def _is_targeted_stop(envelope: dict[str, Any]) -> bool:
    """Return whether the envelope addresses Stop to one admitted turn."""
    if not _is_stop_control(envelope):
        return False
    target = envelope.get("target_turn_id")
    return isinstance(target, str) and bool(target)


def request_settings_error(
    envelope: dict[str, Any],
    *,
    active_turn_id: str | None,
) -> tuple[str, str] | None:
    """Validate the versioned request-settings envelope without side effects."""
    control_field = next(
        (field for field in _MESSAGE_LEVEL_CONTROL_FIELDS if field in envelope),
        None,
    )
    if control_field is not None:
        return "invalid_request_settings", control_field
    version = envelope.get("request_settings_version")
    negotiated = any(field in envelope for field in _REQUEST_SETTINGS_FIELDS)
    if version is None:
        # A bare legacy WebUI envelope remains compatible. Any client that
        # submits a v1 field must also negotiate the version explicitly.
        return ("update_required", "request_settings_version") if negotiated else None
    if type(version) is not int or version != REQUEST_SETTINGS_VERSION:
        return "update_required", "request_settings_version"
    intent = envelope.get("request_intent")
    if intent not in {"new", "steer"}:
        return "invalid_request_settings", "request_intent"
    target = envelope.get("target_turn_id")
    if intent == "new":
        # Stop is a turn control, not a new workspace request. Its target is
        # the admitted turn that the owner must verify before settling.
        if _is_stop_control(envelope):
            if target is not None and (not isinstance(target, str) or not target):
                return "invalid_request_settings", "target_turn_id"
        elif target is not None:
            return "invalid_request_settings", "target_turn_id"
    else:
        if (
            not isinstance(target, str)
            or not active_turn_id
            or target != active_turn_id
        ):
            return "not_admitted", "steering_target_changed"
    return None


def steering_target_error(
    envelope: dict[str, Any],
    *,
    chat_id: str,
    scope: WorkspaceScope,
) -> tuple[str, str] | None:
    """Reject guidance whose target turn or captured scope no longer matches."""
    target_turn_id = envelope.get("target_turn_id")
    if (
        not isinstance(target_turn_id, str)
        or websocket_turn_id(chat_id) != target_turn_id
    ):
        return "not_admitted", "steering_target_changed"
    current_scope = websocket_turn_scope(chat_id)
    if current_scope is None or scope.metadata() != current_scope:
        return "not_admitted", "steering_context_changed"
    return None


@dataclass(frozen=True)

class WebUIRequestResult:
    result: Any = None
    status: int | None = None
    message: str | None = None
    workspace_scope: Any = None
    workspace_permissions: Any = None
    # Authoritative effect disposition, distinct from response acknowledgement.
    # ``applied``/``partial`` mean the effect may exist even if the reply is lost.
    # ``not_applied`` is a pre-effect refusal. ``uncertain`` requires readback.
    outcome: str | None = None


@dataclass
class WebUIRequestOperation:
    action: str
    payload_digest: bytes
    task: asyncio.Task[WebUIRequestResult]
    source_binding: dict[str, object] | None = None
    completed_at: float | None = None


class WebUICommandTransport(Protocol):
    """Typed transport capabilities consumed by WebUI command orchestration."""

    def is_allowed(self, sender_id: str) -> bool: ...

    async def revalidate_mobile_connection(self, connection: Any) -> bool: ...

    def webui_subscribers(self, chat_id: str) -> tuple[ServerConnection, ...]: ...

    def webui_connection_chats(self, connection: ServerConnection) -> tuple[str, ...]: ...

    def webui_attach(self, connection: ServerConnection, chat_id: str) -> None: ...

    def webui_detach(self, connection: ServerConnection, chat_id: str) -> None: ...

    def webui_clear_connection_default(self, connection: ServerConnection) -> None: ...

    def webui_clear_stream_buffers(self, chat_id: str) -> None: ...

    async def webui_hydrate(self, chat_id: str) -> None: ...

    async def webui_send_event(
        self,
        connection: ServerConnection,
        event: str,
        **fields: Any,
    ) -> None: ...

    async def webui_send_raw(
        self,
        connection: ServerConnection,
        raw: str,
        *,
        label: str = "",
    ) -> None: ...

    async def webui_dispatch_message(
        self,
        *,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None,
        attachments: list[AttachmentRecord] | None,
        metadata: dict[str, Any],
        is_dm: bool,
        session_key: str | None,
        require_existing_session: bool,
        workspace_scope: WorkspaceScope,
        permission_reference: object | None = None,
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


def _typed_mobile_identity(connection: Any) -> Any | None:
    """Return only a real mobile identity, never a loose test mock."""
    identity = getattr(connection, "mobile_identity", None)
    if identity is None:
        return None
    installation_id = getattr(identity, "installation_id", None)
    scopes = getattr(identity, "scopes", None)
    if not isinstance(installation_id, str) or not installation_id:
        return None
    if not isinstance(scopes, (tuple, list, set, frozenset)):
        return None
    return identity


def _typed_owner_mobile_identity(connection: Any) -> Any | None:
    """Return a real mobile identity carrying the owner scope."""
    identity = _typed_mobile_identity(connection)
    if identity is None:
        return None
    scopes = getattr(identity, "scopes", None)
    if not isinstance(scopes, (tuple, list, set, frozenset)):
        return None
    return identity if "owner" in scopes else None


def _mobile_denial_message(connection: Any, action: str) -> str:
    """Name the documented local-only exception instead of a generic denial."""
    if _typed_mobile_identity(connection) is not None and (
        action.startswith("settings.remote_access.")
        or action == "workspace.pick_folder"
        or action in {"settings.channel.configure", "settings.channel.connect.start",
                      "settings.channel.connect.poll", "settings.channel.connect.cancel"}
    ):
        return "local_ui_required"
    return "access_denied"


def _mobile_mutation_allowed(connection: Any, action: str) -> bool:
    """Allow legacy chat organization and enabled owner operations.

    The operation registry is the single allowlist for owner devices; a legacy
    chat-only device keeps the three shared-organization actions.
    """
    identity = _typed_mobile_identity(connection)
    if identity is None:
        return False
    try:
        from nanobot.remote.operations import mobile_mutation_allowed

        return mobile_mutation_allowed(identity.scopes, action)
    except Exception:
        return False


class WebUICommandRouter:
    """Own WebUI command semantics while a transport host owns raw connections."""

    def __init__(
        self,
        transport: WebUICommandTransport,
        gateway: GatewayServices,
        *,
        replay_store: DurableWebUIReplay | None = None,
    ) -> None:
        self._transport = transport
        self.gateway = gateway
        self._http_router = gateway.http
        self._media = gateway.media
        self._ingress = gateway.ingress
        self._transcripts = gateway.transcripts
        self._workspaces = gateway.workspaces
        self._temporary_chats = gateway.temporary_chats
        self._session_projection = gateway.session_projection
        self._webui_connections = gateway.endpoint.webui_connections
        self._session_access = (
            WebuiSessionAccess(gateway.session_manager)
            if gateway.session_manager is not None
            else None
        )
        self.request_tasks: dict[
            tuple[ServerConnection, str],
            asyncio.Task[None],
        ] = {}
        self.request_operations: dict[str, WebUIRequestOperation] = {}
        self.request_locks: dict[ServerConnection, asyncio.Lock] = {}
        # A WebUI attempt ID is bound to its submitted payload. A repeated ID
        # with different input must fail; the same input may be recovered.
        self._message_attempts: dict[tuple[str, str], str] = {}
        self._message_attempt_limit = 4096
        if replay_store is None:
            settings = getattr(gateway, "settings", None)
            config = getattr(settings, "config", None)
            config_path = getattr(config, "path", None)
            if config_path is not None:
                replay_store = DurableWebUIReplay(
                    Path(config_path).parent / "webui" / "mutation-replay-v1.json"
                )
        self._durable_replay = replay_store

    def workspace_controls_available(self, connection: ServerConnection) -> bool:
        return (
            connection in self._webui_connections
            and self._http_router.workspace_controls_available(connection)
        )

    async def send_webui_protocol_error(
        self,
        connection: ServerConnection,
        detail: str,
    ) -> None:
        await self._transport.webui_send_event(connection, "error", detail=detail)

    async def attach_webui_fork(
        self,
        connection: ServerConnection,
        *,
        fork_id: str,
        fork_key: str,
    ) -> None:
        scope = self._workspaces.scope_for_session_key(fork_key)
        self._transport.webui_attach(connection, fork_id)
        await self._transport.webui_send_event(
            connection,
            "attached",
            chat_id=fork_id,
            **self._session_projection.attach_fields(fork_key),
        )
        await self._transport.webui_send_event(
            connection,
            "session_updated",
            chat_id=fork_id,
            scope="metadata",
            workspace_scope=self._workspaces.scope_payload(scope),
        )
        await self._transport.webui_hydrate(fork_id)

    async def discard_owned_chat(
        self,
        connection: ServerConnection,
        chat_id: str,
    ) -> None:
        await self._temporary_chats.discard(connection, chat_id)
        self._transport.webui_detach(connection, chat_id)
        clear_websocket_turns(chat_id)
        self._transport.webui_clear_stream_buffers(chat_id)

    async def cleanup_connection(self, connection: ServerConnection) -> None:
        """Release command-owned state associated with one transport connection."""
        chat_ids = self._transport.webui_connection_chats(connection)
        for chat_id in chat_ids:
            if self._temporary_chats.owns(connection, chat_id):
                await self.discard_owned_chat(connection, chat_id)
            else:
                self._transport.webui_detach(connection, chat_id)
        for chat_id in self._temporary_chats.chat_ids_for_owner(connection):
            await self.discard_owned_chat(connection, chat_id)
        self._transport.webui_clear_connection_default(connection)
        self.gateway.endpoint.discard_connection(connection)
        self.discard_request_lock_if_idle(connection)

    async def broadcast_webui_event(
        self,
        event: str,
        *,
        exclude: Any | None = None,
        **fields: Any,
    ) -> None:
        # Fan out to every accepted WebUI-protocol socket (desktop and mobile).
        # ``_webui_connections`` remains the trust/local-authority set; its
        # members are protocol connections too, so the union is deduplicated.
        targets = set(self._webui_connections)
        protocol = getattr(self.gateway.endpoint, "protocol_connections", None)
        if protocol:
            targets.update(protocol)
        if exclude is not None:
            targets.discard(exclude)
        for connection in tuple(targets):
            await self._transport.webui_send_event(connection, event, **fields)

    async def notify_state_changed(
        self,
        action: str,
        payload: Mapping[str, Any],
        *,
        origin: Any | None = None,
    ) -> None:
        """Invalidate the projections one successful mutation changed.

        The namespace table lives in ``nanobot.webui.state_sync``: new actions in
        a known namespace are covered automatically, unknown namespaces refresh
        every cache until they are classified. The origin client already holds
        the mutation response, so it is skipped.
        """
        from nanobot.webui.state_sync import mutation_chat_id, mutation_domains, notify_domains

        await notify_domains(
            self,
            mutation_domains(action),
            mutation_chat_id(payload),
            exclude=origin,
        )
        # ``sidebar`` mutations broadcast their full projection themselves.

    async def broadcast_user_message(
        self,
        origin: ServerConnection,
        chat_id: str,
        text: str,
        *,
        turn_id: str | None,
        starts_turn: bool,
        media_paths: list[str],
        media_names: list[str | None],
        cli_apps: list[dict[str, Any]],
        mcp_presets: list[dict[str, Any]],
        session_mentions: list[SessionMention],
    ) -> None:
        body: dict[str, Any] = {
            "event": "user_message",
            "chat_id": chat_id,
            "text": text,
            "starts_turn": starts_turn,
        }
        if turn_id is not None:
            body["turn_id"] = turn_id
        media = self._media.augment_transcript_user_media(media_paths)
        for attachment, name in zip(media, media_names, strict=False):
            if name:
                attachment["name"] = name
        if media:
            body["media_urls"] = media
        if cli_apps:
            body["cli_apps"] = cli_apps
        if mcp_presets:
            body["mcp_presets"] = mcp_presets
        if session_mentions:
            body["session_mentions"] = session_mentions
        active_turn_id = websocket_turn_id(chat_id)
        if active_turn_id is not None:
            body["active_turn_id"] = active_turn_id
        started_at = websocket_turn_wall_started_at(chat_id)
        if active_turn_id is not None and started_at is not None:
            body["started_at"] = started_at
        raw = json.dumps(body, ensure_ascii=False)
        for connection in self._transport.webui_subscribers(chat_id):
            if connection is not origin:
                await self._transport.webui_send_raw(connection, raw, label=" user_message ")

    async def workspace_scope_or_error(
        self,
        connection: ServerConnection,
        resolver: Callable[[], Any],
        *,
        chat_id: str | None = None,
        turn_id: str | None = None,
    ) -> Any | None:
        try:
            return resolver()
        except WorkspaceScopeError as exc:
            await self._transport.webui_send_event(
                connection,
                "error",
                detail="workspace_scope_rejected",
                reason=exc.message,
                **({"chat_id": chat_id} if chat_id else {}),
                **({"turn_id": turn_id} if turn_id else {}),
            )
            return None

    def _attachment_installation_id(self, connection: ServerConnection) -> str:
        identity = getattr(connection, "mobile_identity", None)
        installation = getattr(identity, "installation_id", None)
        if isinstance(installation, str) and installation:
            return installation
        return "local"

    def _attachment_source(self, connection: ServerConnection) -> str:
        return "mobile" if self._attachment_installation_id(connection) != "local" else "websocket"

    async def _handle_attachment_upload(
        self,
        connection: ServerConnection,
        envelope: dict[str, Any],
    ) -> None:
        request_id = envelope.get("request_id")
        request_field = {"request_id": request_id} if isinstance(request_id, str) else {}
        data_url = envelope.get("data_url")
        if not isinstance(data_url, str) or not data_url:
            raw = envelope.get("data_base64")
            mime = envelope.get("mime")
            if isinstance(raw, str) and raw and isinstance(mime, str) and mime:
                data_url = f"data:{mime};base64,{raw}"
        if not isinstance(data_url, str) or not data_url:
            await self._transport.webui_send_event(
                connection,
                "error",
                detail="attachment_rejected",
                reason="malformed",
                **request_field,
            )
            return
        name = envelope.get("name")
        item = {
            "data_url": data_url,
            "name": name if isinstance(name, str) else None,
        }
        source = self._attachment_source(connection)
        batch = self._media.attachment_ingress.accept_batch([item], source=source)
        if batch.rejection is not None:
            await self._transport.webui_send_event(
                connection,
                "error",
                detail="attachment_rejected",
                reason=batch.rejection,
                **request_field,
            )
            return
        records, rejection = self._media.attachment_ingress.commit_batch(batch.staged)
        if rejection is not None or not records:
            await self._transport.webui_send_event(
                connection,
                "error",
                detail="attachment_rejected",
                reason=rejection or "storage",
                **request_field,
            )
            return
        record = records[0]
        chat_id = envelope.get("chat_id")
        session_key = webui_session_key(chat_id) if is_valid_webui_chat_id(chat_id) else None
        self._media.upload_registry.add(
            record,
            installation_id=self._attachment_installation_id(connection),
            session_key=session_key,
            source=source,
        )
        await self._transport.webui_send_event(
            connection,
            "attachment_stored",
            id=record.id,
            name=record.name,
            mime=record.mime,
            bytes=record.bytes,
            **request_field,
        )

    async def dispatch(
        self,
        connection: ServerConnection,
        client_id: str,
        envelope: dict[str, Any],
    ) -> None:
        """Execute one typed WebUI command."""
        if _typed_mobile_identity(connection) is not None:
            if not await self._transport.revalidate_mobile_connection(connection):
                await self._transport.webui_send_event(
                    connection,
                    "error",
                    detail="access_expired",
                )
                try:
                    await connection.close(1008, "access expired")
                except Exception:
                    pass
                return
            from nanobot.remote.operations import mobile_frame_allowed

            identity = _typed_mobile_identity(connection)
            if identity is None or not mobile_frame_allowed(identity.scopes, envelope):
                await self._transport.webui_send_event(connection, "error", detail="scope_denied")
                return
        command_type = envelope.get("type")
        if command_type == "attachment":
            await self._handle_attachment_upload(connection, envelope)
            return
        if command_type == "webui_request":
            await self.start_webui_request(connection, envelope)
            return
        if command_type == "new_chat":
            new_id = str(uuid.uuid4())
            scope = await self.workspace_scope_or_error(
                connection,
                lambda: self._workspaces.scope_for_new_chat(
                    envelope,
                    controls_available=self.workspace_controls_available(connection),
                ),
            )
            if scope is None:
                return
            self._workspaces.stage_scope(new_id, scope)
            self._transport.webui_attach(connection, new_id)
            await self._transport.webui_send_event(
                connection,
                "attached",
                chat_id=new_id,
                **self._session_projection.attach_fields(webui_session_key(new_id)),
            )
            await self._transport.webui_send_event(
                connection,
                "session_updated",
                chat_id=new_id,
                scope="metadata",
                workspace_scope=self._workspaces.scope_payload(scope),
            )
            await self._transport.webui_hydrate(new_id)
            return
        if command_type == "new_temporary_chat":
            try:
                new_id = self._temporary_chats.create(
                    connection,
                    trusted_webui=connection in self._webui_connections,
                )
            except TemporaryChatError as exc:
                await self._transport.webui_send_event(connection, "error", detail=exc.detail)
                return
            self._transport.webui_attach(connection, new_id)
            await self._transport.webui_send_event(
                connection,
                "attached",
                chat_id=new_id,
                temporary=True,
            )
            return
        if command_type == "fork_chat":
            await handle_webui_fork_chat(self, connection, envelope)
            return
        if command_type == "discard_temporary_chat":
            chat_id = envelope.get("chat_id")
            if not is_valid_webui_chat_id(chat_id):
                await self._transport.webui_send_event(
                    connection,
                    "error",
                    detail="invalid temporary chat_id",
                )
                return
            try:
                await self.discard_owned_chat(connection, chat_id)
            except TemporaryChatError as exc:
                await self._transport.webui_send_event(
                    connection,
                    "error",
                    detail=exc.detail,
                    chat_id=chat_id,
                )
            return
        if command_type == "attach":
            chat_id = envelope.get("chat_id")
            if not is_valid_webui_chat_id(chat_id):
                await self._transport.webui_send_event(
                    connection,
                    "error",
                    detail="invalid chat_id",
                )
                return
            try:
                self._temporary_chats.validate_attach(chat_id)
            except TemporaryChatError as exc:
                await self._transport.webui_send_event(
                    connection,
                    "error",
                    detail=exc.detail,
                    chat_id=chat_id,
                )
                return
            self._transport.webui_attach(connection, chat_id)
            await self._transport.webui_send_event(
                connection,
                "attached",
                chat_id=chat_id,
                **self._session_projection.attach_fields(webui_session_key(chat_id)),
            )
            await self._transport.webui_hydrate(chat_id)
            return
        if command_type == "set_sidebar_state":
            if connection not in self._webui_connections:
                await self._transport.webui_send_event(connection, "error", detail="access_denied")
                return
            state = envelope.get("state")
            if not isinstance(state, dict):
                await self._transport.webui_send_event(
                    connection,
                    "error",
                    detail="invalid_sidebar_state",
                )
                return
            try:
                saved_state = await asyncio.to_thread(
                    write_webui_sidebar_state,
                    cast(dict[str, Any], state),
                )
            except (OSError, ValueError):
                await self._transport.webui_send_event(
                    connection,
                    "error",
                    detail="invalid_sidebar_state",
                )
                return
            await self.broadcast_webui_event("sidebar_state_updated", state=saved_state)
            return
        if command_type == "set_workspace_scope":
            chat_id = envelope.get("chat_id")
            if not is_valid_webui_chat_id(chat_id):
                await self._transport.webui_send_event(
                    connection,
                    "error",
                    detail="invalid chat_id",
                )
                return
            try:
                self._temporary_chats.validate_workspace_update(chat_id)
            except TemporaryChatError as exc:
                await self._transport.webui_send_event(
                    connection,
                    "error",
                    detail=exc.detail,
                    chat_id=chat_id,
                )
                return
            scope = await self.workspace_scope_or_error(
                connection,
                lambda: self._workspaces.scope_for_set_request(
                    envelope,
                    chat_id=chat_id,
                    chat_running=websocket_turn_wall_started_at(chat_id) is not None,
                    controls_available=self.workspace_controls_available(connection),
                ),
                chat_id=chat_id,
            )
            if scope is None:
                return
            self._workspaces.stage_scope(chat_id, scope)
            await self._transport.send_session_updated(
                chat_id,
                scope="metadata",
                workspace_scope=self._workspaces.scope_payload(scope),
            )
            await self._transport.webui_send_event(
                connection,
                "session_updated",
                chat_id=chat_id,
                scope="metadata",
                workspace_scope=self._workspaces.scope_payload(scope),
            )
            return
        if command_type == "transcribe_audio":
            event, payload = await webui_transcription_event(
                envelope,
                config_path=self.gateway.settings.config.path,
            )
            await self._transport.webui_send_event(connection, event, **payload)
            return
        if command_type == "message":
            await self._dispatch_message(connection, client_id, envelope)
            return
        await self._transport.webui_send_event(
            connection,
            "error",
            detail=f"unknown type: {command_type!r}",
        )

    def _bind_message_attempt(
        self,
        chat_id: str,
        turn_id: str | None,
        envelope: dict[str, Any],
        *,
        stop_expectation: StopExpectation | None = None,
    ) -> tuple[str, str] | None:
        """Bind one attempt ID to its payload; reject changed repeated input."""
        if not turn_id:
            return None
        payload = {
            key: envelope.get(key)
            for key in (
                "content",
                "media",
                "attachments",
                "cli_apps",
                "mcp_presets",
                "session_mentions",
                "quoted_context",
                "request_intent",
                "target_turn_id",
                "approval_decision",
                "stop_expectation",
                "user_shell",
                "webui",
            )
        }
        payload["stop_expectation"] = (
            stop_expectation.to_dict() if stop_expectation is not None else None
        )
        digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        key = (chat_id, turn_id)
        current = self._message_attempts.get(key)
        if current is None:
            if len(self._message_attempts) >= self._message_attempt_limit:
                # Bound the ledger without dropping a fresh in-flight attempt.
                for stale_key in list(self._message_attempts)[
                    : max(1, len(self._message_attempts) - self._message_attempt_limit + 1)
                ]:
                    self._message_attempts.pop(stale_key, None)
            self._message_attempts[key] = digest
            return None
        if current != digest:
            return ("not_admitted", "attempt_payload_changed")
        return None

    async def _dispatch_message(
        self,
        connection: ServerConnection,
        client_id: str,
        envelope: dict[str, Any],
    ) -> None:
        chat_id = envelope.get("chat_id")
        content = envelope.get("content")
        if not is_valid_webui_chat_id(chat_id):
            await self._transport.webui_send_event(connection, "error", detail="invalid chat_id")
            return
        raw_turn_id = envelope.get("turn_id")
        turn_id = raw_turn_id if isinstance(raw_turn_id, str) and raw_turn_id else None
        rejection_fields = {
            "chat_id": chat_id,
            **({"turn_id": turn_id} if turn_id else {}),
        }
        is_stop = _is_stop_control(envelope)
        raw_stop_expectation = envelope.get("stop_expectation")
        has_stop_expectation = "stop_expectation" in envelope
        duplicate_keys = getattr(envelope, "duplicate_keys", ())
        stop_expectation: StopExpectation | None = None
        if has_stop_expectation or "stop_expectation" in duplicate_keys:
            if not is_stop or "stop_expectation" in duplicate_keys:
                await self._transport.webui_send_event(
                    connection,
                    "error",
                    detail="not_admitted",
                    reason="invalid_stop_expectation",
                    **rejection_fields,
                )
                return
            try:
                stop_expectation = parse_stop_expectation(raw_stop_expectation)
            except (TypeError, ValueError):
                await self._transport.webui_send_event(
                    connection,
                    "error",
                    detail="not_admitted",
                    reason="invalid_stop_expectation",
                    **rejection_fields,
                )
                return
        attempt_error = self._bind_message_attempt(
            str(chat_id),
            turn_id,
            envelope,
            stop_expectation=stop_expectation,
        )
        if attempt_error is not None:
            detail, reason = attempt_error
            await self._transport.webui_send_event(
                connection,
                "error",
                detail=detail,
                reason=reason,
                **rejection_fields,
            )
            return
        if not self._transport.is_allowed(client_id):
            await self._transport.webui_send_event(
                connection,
                "error",
                detail="access_denied",
                **rejection_fields,
            )
            return
        request_intent = envelope.get("request_intent")
        temporary_owned = self._temporary_chats.owns(connection, chat_id)
        current_source_token = (
            None if temporary_owned else self._workspaces.source_token(chat_id)
        )
        settings_error = request_settings_error(
            envelope,
            active_turn_id=websocket_turn_id(chat_id),
        )
        if settings_error is not None:
            detail, reason = settings_error
            await self._transport.webui_send_event(
                connection,
                "error",
                detail=detail,
                reason=reason,
                **rejection_fields,
            )
            return
        approval_control = "approval_decision" in envelope or "approval_decision" in duplicate_keys
        if approval_control:
            raw_decision: object = envelope.get("approval_decision")
            try:
                if (
                    "approval_decision" in duplicate_keys
                    or not isinstance(raw_decision, dict)
                    or getattr(cast(dict[object, object], raw_decision), "duplicate_keys", ())
                    or set(cast(dict[object, object], raw_decision)) - {
                        "approval_id", "decision", "session_key", "session_incarnation",
                        "run_id", "turn_id", "effect_id", "generation",
                    }
                ):
                    raise ValueError("invalid approval control")
                ApprovalDecision.from_mapping(raw_decision)
                if (
                    is_stop
                    or envelope.get("webui") is not True
                    or connection not in self._webui_connections
                    or not turn_id
                    or turn_id == websocket_turn_id(chat_id)
                    or request_intent == "steer"
                    or temporary_owned
                ):
                    raise ValueError("invalid approval control context")
            except (TypeError, ValueError):
                await self._transport.webui_send_event(
                    connection, "error", detail="not_admitted",
                    reason="invalid_approval_decision", **rejection_fields,
                )
                return
        if (
            request_intent == "new"
            and not approval_control
            and not _is_targeted_stop(envelope)
            and stop_expectation is None
            and (
                websocket_turn_wall_started_at(chat_id) is not None
                or websocket_turn_reserved(chat_id)
            )
        ):
            await self._transport.webui_send_event(
                connection,
                "error",
                detail="not_admitted",
                reason="busy_new_request",
                **rejection_fields,
            )
            return
        if not isinstance(content, str):
            await self._transport.webui_send_event(
                connection,
                "error",
                detail="missing content",
                **rejection_fields,
            )
            return
        message_rejection = self._ingress.validate_text(content)
        if message_rejection is not None:
            await self._transport.webui_send_event(
                connection,
                "error",
                detail="message_rejected",
                reason=message_rejection,
                **rejection_fields,
            )
            return

        if approval_control:
            # Decisions carry no new parent input or workspace authority. Read
            # only the current scope; the canonical approval owner resolves the
            # exact child and revalidates its source after independent scheduling.
            try:
                scope = self._workspaces.scope_for_session_key(webui_session_key(chat_id))
                if (
                    current_source_token is None
                    or self._workspaces.source_token(chat_id) != current_source_token
                    or not self._transport.is_allowed(client_id)
                ):
                    raise ValueError("approval source changed")
            except Exception:
                await self._transport.webui_send_event(
                    connection, "error", detail="not_admitted",
                    reason="approval_source_unavailable", **rejection_fields,
                )
                return
            metadata = {
                "remote": getattr(connection, "remote_address", None),
                "webui": True,
                **self._transcripts.client_turn_metadata(turn_id),
                WEBUI_SOURCE_TOKEN_METADATA_KEY: current_source_token,
                "approval_decision": dict(cast(dict[str, Any], envelope["approval_decision"])),
            }
            self._transport.webui_attach(connection, chat_id)
            await self._transport.webui_dispatch_message(
                sender_id=client_id, chat_id=chat_id, content=content,
                media=None, attachments=None, metadata=metadata, is_dm=False,
                # Exact pending ownership is resolved before ordinary dispatch.
                # A valid settled root need not remain in the ephemeral cache.
                session_key=None, require_existing_session=False, workspace_scope=scope,
            )
            await self._transport.webui_send_event(
                connection, "message_accepted", chat_id=chat_id, turn_id=turn_id,
                starts_turn=False,
            )
            return

        try:
            temporary_policy = self._temporary_chats.message_policy(
                connection,
                chat_id,
                content,
            )
        except TemporaryChatError as exc:
            await self._transport.webui_send_event(
                connection,
                "error",
                detail=exc.detail,
                **rejection_fields,
            )
            return

        self._transport.webui_attach(connection, chat_id)
        if temporary_policy is None or temporary_policy.hydrate_transcript:
            await self._transport.webui_hydrate(chat_id)
        if (
            temporary_policy is None
            and self._workspaces.source_token(chat_id) != current_source_token
        ):
            await self._transport.webui_send_event(
                connection,
                "error",
                detail="source_changed",
                reason="source_token",
                **rejection_fields,
            )
            return

        is_webui = envelope.get("webui") is True
        trusted_webui = is_webui and connection in self._webui_connections
        is_user_shell = (
            trusted_webui and envelope.get("user_shell") is True and content.startswith("!")
        )
        queued_owner = None
        accepted = False
        expects_agent_turn = (
            is_webui and not is_user_shell and builtin_command_starts_agent_turn(content)
        )
        if expects_agent_turn:
            queued_owner = register_queued_websocket_turn_if_idle(
                chat_id,
                turn_id,
                publish=False,
            )
        if request_intent == "new" and expects_agent_turn and queued_owner is None:
            # The chat may have become busy while hydration was waiting. A new
            # v1 request must own the turn reservation before it can create any
            # transcript, default or inbound effect.
            await self._transport.webui_send_event(
                connection,
                "error",
                detail="not_admitted",
                reason="busy_new_request",
                **rejection_fields,
            )
            return
        try:
            raw_media = envelope.get("media")
            media_paths: list[str] = []
            media_records: tuple[AttachmentRecord, ...] = ()
            media_names: list[str | None] = []
            bound_records: list[AttachmentRecord] = []
            staged: list[Any] = []
            media_committed = False

            def discard_staged() -> None:
                if staged and not media_committed:
                    self._media.attachment_ingress.discard_batch(staged)
                    staged.clear()

            if raw_media is not None:
                if not isinstance(raw_media, list):
                    await self._transport.webui_send_event(
                        connection,
                        "error",
                        detail="attachment_rejected",
                        reason="malformed",
                        **rejection_fields,
                    )
                    return
                id_items: list[dict[str, Any]] = []
                data_items: list[Any] = []
                for item in cast(list[Any], raw_media):
                    if (
                        isinstance(item, dict)
                        and isinstance(cast(dict[str, Any], item).get("id"), str)
                    ):
                        id_items.append(cast(dict[str, Any], item))
                    else:
                        data_items.append(item)
                if id_items and len(id_items) + len(data_items) > self._media.attachment_limits.max_count:
                    await self._transport.webui_send_event(
                        connection,
                        "error",
                        detail="attachment_rejected",
                        reason="too_many_attachments",
                        **rejection_fields,
                    )
                    return
                installation_id = self._attachment_installation_id(connection)
                if id_items:
                    for item in id_items:
                        record = self._media.upload_registry.bind(
                            str(item["id"]),
                            installation_id=installation_id,
                            session_key=webui_session_key(chat_id),
                        )
                        if record is None:
                            discard_staged()
                            await self._transport.webui_send_event(
                                connection,
                                "error",
                                detail="attachment_rejected",
                                reason="expired",
                                **rejection_fields,
                            )
                            return
                        bound_records.append(record)
                        media_paths.append(str(self._media.attachment_store.path_for(record)))
                        media_names.append(record.name)
                if data_items:
                    batch = self._media.attachment_ingress.accept_batch(
                        data_items,
                        source=self._attachment_source(connection),
                    )
                    if batch.rejection is not None:
                        await self._transport.webui_send_event(
                            connection,
                            "error",
                            detail="attachment_rejected",
                            reason=batch.rejection,
                            **rejection_fields,
                        )
                        return
                    staged = list(batch.staged)
                    for item in data_items:
                        attachment = cast(dict[str, Any], item) if isinstance(item, dict) else {}
                        name = attachment.get("name")
                        media_names.append((safe_filename(name) or None) if isinstance(name, str) else None)

            if not content.strip() and not staged and not bound_records:
                discard_staged()
                await self._transport.webui_send_event(
                    connection,
                    "error",
                    detail="missing content",
                    **rejection_fields,
                )
                return
            local_controls = self.workspace_controls_available(connection)
            controls_available = local_controls or _typed_owner_mobile_identity(connection) is not None
            scope = await self.workspace_scope_or_error(
                connection,
                lambda: (
                    self._workspaces.scope_for_temporary_message(
                        envelope,
                        temporary_policy.workspace_scope,
                    )
                    if temporary_policy is not None
                    else self._workspaces.scope_for_message(
                        envelope,
                        chat_id=chat_id,
                        chat_running=websocket_turn_wall_started_at(chat_id) is not None,
                        controls_available=controls_available,
                    )
                ),
                chat_id=chat_id,
                turn_id=turn_id,
            )
            if scope is None:
                discard_staged()
                return
            if queued_owner is not None:
                bind_queued_websocket_turn_scope(chat_id, queued_owner, scope.metadata())
            if request_intent == "steer":
                error = steering_target_error(envelope, chat_id=chat_id, scope=scope)
                if error is not None:
                    detail, reason = error
                    await self._transport.webui_send_event(
                        connection,
                        "error",
                        detail=detail,
                        reason=reason,
                        **rejection_fields,
                    )
                    discard_staged()
                    return
            permission_reference = None
            if scope.access_mode == "full":
                mobile_identity = getattr(connection, "mobile_identity", None)
                mobile_installation = getattr(mobile_identity, "installation_id", None)
                if temporary_policy is not None:
                    permission_reference = self._workspaces.temporary_full_reference(
                        chat_id,
                        temporary_policy.source_binding,
                        request_turn_id=turn_id,
                    )
                else:
                    permission_reference = self._workspaces.ingress_full_reference(
                        chat_id,
                        local_ui=local_controls,
                        installation_id=(
                            mobile_installation
                            if isinstance(mobile_installation, str) and mobile_installation
                            else None
                        ),
                        request_turn_id=turn_id,
                    )
                if permission_reference is None:
                    await self._transport.webui_send_event(
                        connection,
                        "error",
                        detail="workspace_scope_rejected",
                        reason="full_access_unavailable",
                        **rejection_fields,
                    )
                    discard_staged()
                    return
                if temporary_policy is None:
                    self._workspaces.mark_full_request_admitted(chat_id)

            if queued_owner is not None:
                activate_queued_websocket_turn(chat_id, queued_owner)

            if not self._transport.is_allowed(client_id):
                await self._transport.webui_send_event(
                    connection,
                    "error",
                    detail="access_denied",
                    **rejection_fields,
                )
                discard_staged()
                return

            metadata: dict[str, Any] = {
                "remote": getattr(connection, "remote_address", None)
            }
            if temporary_policy is None and current_source_token is not None:
                metadata[WEBUI_SOURCE_TOKEN_METADATA_KEY] = current_source_token
            mobile_identity = getattr(connection, "mobile_identity", None)
            mobile_installation = getattr(mobile_identity, "installation_id", None)
            if isinstance(mobile_installation, str) and mobile_installation:
                metadata[WEBUI_INSTALLATION_METADATA_KEY] = mobile_installation
            if envelope.get("webui") is True:
                metadata["webui"] = True
                metadata.update(self._transcripts.client_turn_metadata(envelope.get("turn_id")))
            if isinstance(request_intent, str):
                metadata[WEBUI_REQUEST_INTENT_METADATA_KEY] = request_intent
            if is_stop:
                metadata[INBOUND_META_STOP_CLIENT] = True
                if stop_expectation is not None:
                    metadata[INBOUND_META_STOP_EXPECTATION] = stop_expectation
            target_turn_id = envelope.get("target_turn_id")
            if isinstance(target_turn_id, str):
                metadata[WEBUI_TARGET_TURN_METADATA_KEY] = target_turn_id
            approval_decision = envelope.get("approval_decision")
            if isinstance(approval_decision, dict):
                metadata["approval_decision"] = dict(
                    cast(dict[str, Any], approval_decision),
                )
            if is_user_shell:
                metadata[INBOUND_META_USER_SHELL] = True
            dispatch_content = (
                f"{USER_SHELL_COMMAND} {content[1:].lstrip()}" if is_user_shell else content
            )
            cli_apps = normalize_cli_app_mentions(envelope.get("cli_apps"))
            if cli_apps:
                metadata["cli_apps"] = cli_apps
            mcp_presets = normalize_mcp_preset_mentions(
                envelope.get("mcp_presets"),
                config_path=self.gateway.settings.config.path,
            )
            if mcp_presets:
                metadata["mcp_presets"] = mcp_presets
            session_mentions: list[SessionMention] = []
            if trusted_webui and self._session_access is not None:
                session_mentions = await asyncio.to_thread(
                    self._session_access.normalize_mentions,
                    envelope.get("session_mentions"),
                    exclude_session_key=webui_session_key(chat_id),
                )
                if session_mentions:
                    metadata["session_mentions"] = session_mentions
            metadata[WORKSPACE_SCOPE_METADATA_KEY] = scope.metadata()
            if queued_owner is not None:
                metadata[WEBSOCKET_TURN_OWNER_METADATA_KEY] = queued_owner

            if request_intent == "steer":
                error = steering_target_error(envelope, chat_id=chat_id, scope=scope)
                if error is not None:
                    detail, reason = error
                    await self._transport.webui_send_event(
                        connection,
                        "error",
                        detail=detail,
                        reason=reason,
                        **rejection_fields,
                    )
                    discard_staged()
                    return

            if (
                temporary_policy is None
                and self._workspaces.source_token(chat_id) != current_source_token
            ):
                await self._transport.webui_send_event(
                    connection,
                    "error",
                    detail="source_changed",
                    reason="source_token",
                    **rejection_fields,
                )
                discard_staged()
                return

            if staged:
                records, rejection = self._media.attachment_ingress.commit_batch(staged)
                if rejection is not None:
                    await self._transport.webui_send_event(
                        connection,
                        "error",
                        detail="attachment_rejected",
                        reason=rejection,
                        **rejection_fields,
                    )
                    discard_staged()
                    return
                media_paths.extend(
                    str(self._media.attachment_store.path_for(record))
                    for record in records
                )
                media_records = tuple([*bound_records, *records])
                media_committed = True
                staged.clear()
            else:
                media_records = tuple(bound_records)
            if temporary_policy is not None and media_paths:
                self._temporary_chats.register_media(connection, chat_id, media_paths, records=media_records)

            try:
                if is_webui and (
                    temporary_policy is None or temporary_policy.persist_transcript
                ):
                    self._transcripts.append_user_message(
                        chat_id,
                        content,
                        metadata=metadata,
                        media_paths=media_paths or None,
                        attachments=[record.to_dict() for record in media_records] or None,
                        cli_apps=cli_apps or None,
                        mcp_presets=mcp_presets or None,
                        session_mentions=session_mentions or None,
                    )
                if trusted_webui:
                    context_blocks: list[RuntimeContextBlock] = []
                    quote = webui_quote_runtime_context(
                        {WEBUI_QUOTE_METADATA: envelope.get("quoted_context")}
                    )
                    if quote is not None:
                        context_blocks.append(quote)
                    session_context = session_mentions_runtime_context(session_mentions)
                    if session_context is not None:
                        context_blocks.append(session_context)
                    if context_blocks:
                        metadata[RUNTIME_CONTEXT_INPUT_META] = context_blocks
                await self._transport.webui_dispatch_message(
                    sender_id=client_id,
                    chat_id=chat_id,
                    content=dispatch_content,
                    media=media_paths or None,
                    attachments=list(media_records) or None,
                    metadata=metadata,
                    is_dm=False,
                    session_key=(
                        temporary_policy.session_key if temporary_policy is not None else None
                    ),
                    require_existing_session=(
                        temporary_policy.require_existing_session
                        if temporary_policy is not None
                        else False
                    ),
                    workspace_scope=scope,
                    permission_reference=permission_reference,
                )
                if temporary_policy is None:
                    self._workspaces.persist_scope(chat_id, scope)
                    self._workspaces.remember_request_defaults(chat_id, scope)
                accepted = True
            finally:
                if not accepted and queued_owner is not None:
                    clear_websocket_turn_if_current(chat_id, queued_owner)
                if not media_committed:
                    discard_staged()

            if is_webui:
                await self.broadcast_user_message(
                    connection,
                    chat_id,
                    content,
                    turn_id=turn_id,
                    starts_turn=queued_owner is not None,
                    media_paths=media_paths,
                    media_names=media_names,
                    cli_apps=cli_apps,
                    mcp_presets=mcp_presets,
                    session_mentions=session_mentions,
                )
            if is_webui and turn_id:
                active_turn_id = websocket_turn_id(chat_id)
                started_at = websocket_turn_wall_started_at(chat_id)
                await self._transport.webui_send_event(
                    connection,
                    "message_accepted",
                    chat_id=chat_id,
                    turn_id=turn_id,
                    starts_turn=queued_owner is not None,
                    **(
                        {"active_turn_id": active_turn_id}
                        if active_turn_id is not None
                        else {}
                    ),
                    **(
                        {"started_at": started_at}
                        if active_turn_id is not None and started_at is not None
                        else {}
                    ),
                )

        finally:
            if not accepted and queued_owner is not None:
                clear_websocket_turn_if_current(chat_id, queued_owner)

    async def start_webui_request(
        self,
        connection: ServerConnection,
        envelope: dict[str, Any],
    ) -> None:
        request_id = envelope.get("request_id")
        if not isinstance(request_id, str) or re.fullmatch(
            r"[A-Za-z0-9._:-]{1,128}",
            request_id,
        ) is None:
            await self._transport.webui_send_event(
                connection,
                "error",
                detail="invalid webui request_id",
            )
            return
        action = envelope.get("action")
        payload = envelope.get("payload")
        if not isinstance(action, str) or re.fullmatch(
            r"[a-z][a-z0-9_.]{0,127}",
            action,
        ) is None:
            await self.send_webui_response(
                connection,
                request_id,
                status=400,
                message="invalid WebUI mutation action",
                outcome="not_applied",
            )
            return
        if not isinstance(payload, dict):
            await self.send_webui_response(
                connection,
                request_id,
                status=400,
                message="WebUI mutation payload must be an object",
                outcome="not_applied",
            )
            return
        if (
            connection not in self._webui_connections
            and not _mobile_mutation_allowed(connection, action)
        ):
            await self.send_webui_response(
                connection,
                request_id,
                status=403,
                message=_mobile_denial_message(connection, action),
                outcome="not_applied",
            )
            return

        authorization = self._http_router.authorize_webui_mutation(
            connection,
            action,
            cast(dict[str, Any], payload),
        )
        if authorization is not None:
            await self.send_webui_response(
                connection,
                request_id,
                status=authorization.status,
                message=authorization.message,
                outcome="not_applied",
            )
            return

        typed_payload = cast(dict[str, Any], payload)
        source_binding_captured = False
        if (
            action == _PREPARE_FULL_ACTION
            and self.gateway.session_manager is not None
            and is_valid_webui_chat_id(typed_payload.get("chat_id"))
        ):
            source_binding_captured, source_binding = self._prepare_admission_source_binding(
                typed_payload,
            )
            if not source_binding_captured:
                await self.send_webui_response(
                    connection,
                    request_id,
                    status=409,
                    message="source_changed",
                    outcome="not_applied",
                )
                return
        else:
            source_binding = self._permission_source_binding(typed_payload)
        if (
            action in _SOURCE_BOUND_PERMISSION_ACTIONS
            and self.gateway.session_manager is not None
            and source_binding is None
        ):
            await self.send_webui_response(
                connection,
                request_id,
                status=409,
                message="source_changed",
                outcome="not_applied",
            )
            return

        payload_digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).digest()
        self.prune_request_operations()
        operation = self.request_operations.get(request_id)
        is_replay = operation is not None
        if operation is not None and (
            operation.action != action or operation.payload_digest != payload_digest
        ):
            await self.send_webui_response(
                connection,
                request_id,
                status=409,
                message="request_id was already used for a different WebUI mutation",
                outcome="not_applied",
            )
            return
        if (
            operation is not None
            and action in _SOURCE_BOUND_PERMISSION_ACTIONS
            and operation.source_binding != source_binding
        ):
            await self.send_webui_response(connection, request_id, status=409, message="source_changed")
            return
        if operation is None and self._durable_replay is not None:
            try:
                durable_state, durable_entry = self._durable_replay.lookup(
                    request_id,
                    action,
                    payload_digest.hex(),
                    source_binding,
                )
            except Exception:
                logger.exception("WebUI replay ledger lookup failed")
                await self.send_webui_response(
                    connection,
                    request_id,
                    status=503,
                    message="replay_store_unavailable",
                    outcome="uncertain",
                )
                return
            if durable_state == "mismatch":
                await self.send_webui_response(
                    connection,
                    request_id,
                    status=409,
                    message="request_id was already used for a different WebUI mutation",
                    outcome="not_applied",
                )
                return
            if durable_state == "uncertain":
                await self.send_webui_response(
                    connection,
                    request_id,
                    status=409,
                    message="replay_uncertain",
                    outcome="uncertain",
                )
                return
            if durable_state == "completed":
                stored_value = durable_entry.get("result") if durable_entry else None
                stored: dict[str, Any] = (
                    cast(dict[str, Any], stored_value)
                    if isinstance(stored_value, dict)
                    else {}
                )
                cached = WebUIRequestResult(
                    status=stored.get("status"),
                    message=stored.get("message"),
                    result=stored.get("result"),
                    workspace_scope=stored.get("workspace_scope"),
                    workspace_permissions=stored.get("workspace_permissions"),
                    outcome=(
                        cast(str, stored["outcome"])
                        if stored.get("outcome") in {
                            "applied", "not_applied", "uncertain", "partial",
                        }
                        else None
                    ),
                )

                async def _cached_result() -> WebUIRequestResult:
                    return cached

                cached_task = asyncio.create_task(_cached_result())
                key = (connection, request_id)
                if key in self.request_tasks:
                    return
                delivery_task = asyncio.create_task(
                    self.deliver_webui_request(
                        connection,
                        request_id,
                        cached_task,
                        action=action,
                        payload=typed_payload,
                        sequence=True,
                        source_binding=source_binding,
                    )
                )
                self.request_tasks[key] = delivery_task
                return
        if operation is None:
            if self._durable_replay is not None:
                try:
                    started = self._durable_replay.begin(
                        request_id,
                        action,
                        payload_digest.hex(),
                        source_binding,
                    )
                except Exception:
                    logger.exception("WebUI replay ledger admission failed")
                    await self.send_webui_response(
                        connection,
                        request_id,
                        status=503,
                        message="replay_store_unavailable",
                        outcome="uncertain",
                    )
                    return
                if not started:
                    await self.send_webui_response(
                        connection,
                        request_id,
                        status=409,
                        message="replay_uncertain",
                        outcome="uncertain",
                    )
                    return

            async def execute_and_persist() -> WebUIRequestResult:
                try:
                    result = await self.execute_webui_request(
                        connection,
                        action,
                        typed_payload,
                        source_binding=source_binding,
                        source_binding_captured=source_binding_captured,
                    )
                except asyncio.CancelledError:
                    if self._durable_replay is not None:
                        try:
                            self._durable_replay.uncertain(request_id)
                        except Exception:
                            logger.exception("WebUI replay ledger could not mark uncertainty")
                    raise
                except Exception:
                    if self._durable_replay is not None:
                        try:
                            self._durable_replay.uncertain(request_id)
                        except Exception:
                            logger.exception("WebUI replay ledger could not mark uncertainty")
                    raise
                self._record_replay_result(request_id, result)
                return result

            operation_task = asyncio.create_task(execute_and_persist())
            new_operation = WebUIRequestOperation(
                action=action,
                payload_digest=payload_digest,
                task=operation_task,
                source_binding=source_binding,
            )
            operation = new_operation
            self.request_operations[request_id] = new_operation

            def mark_complete(_task: asyncio.Task[WebUIRequestResult]) -> None:
                current = self.request_operations.get(request_id)
                if current is not new_operation:
                    return
                new_operation.completed_at = time.monotonic()
                self.prune_request_operations()

            operation_task.add_done_callback(mark_complete)

        key = (connection, request_id)
        if key in self.request_tasks:
            return
        delivery_task = asyncio.create_task(
            self.deliver_webui_request(
                connection,
                request_id,
                operation.task,
                action=action,
                payload=typed_payload,
                sequence=is_replay,
                source_binding=operation.source_binding,
            )
        )
        self.request_tasks[key] = delivery_task

    def prune_request_operations(self) -> None:
        now = time.monotonic()
        for request_id, operation in tuple(self.request_operations.items()):
            if (
                operation.completed_at is not None
                and now - operation.completed_at >= _WEBUI_REQUEST_CACHE_TTL_S
            ):
                self.request_operations.pop(request_id, None)

        completed = sorted(
            (
                (operation.completed_at, request_id)
                for request_id, operation in self.request_operations.items()
                if operation.completed_at is not None
            ),
            key=lambda item: item[0],
        )
        for _, request_id in completed[:-_WEBUI_REQUEST_CACHE_MAX]:
            self.request_operations.pop(request_id, None)

    def _permission_source_binding(
        self,
        payload: dict[str, Any],
    ) -> dict[str, object] | None:
        sessions = self.gateway.session_manager
        chat_id = payload.get("chat_id")
        if sessions is None or not isinstance(chat_id, str):
            return None
        try:
            key = webui_session_key(chat_id)
            backend = sessions.source_backend(key)
            if backend is None:
                return None
            state = backend.source.load_source_snapshot(key).state
            return {
                "store_identity": state.store_identity,
                "incarnation": state.incarnation,
                "generation": state.generation,
            }
        except Exception:
            return None

    def _prepare_admission_source_binding(
        self,
        payload: dict[str, Any],
    ) -> tuple[bool, dict[str, object] | None]:
        """Capture an existing source, or explicit absence, for one prepare dispatch."""
        sessions = self.gateway.session_manager
        chat_id = payload.get("chat_id")
        if sessions is None or not isinstance(chat_id, str):
            return False, None
        try:
            if not sessions.has_managed_authority:
                return False, None
            key = webui_session_key(chat_id)
            resolved = sessions.resolve_session_mode(key)
            if resolved.state == "absent":
                return True, None
            if resolved.state != "active" or resolved.mode != "canonical":
                return False, None
            backend = sessions.source_backend(key)
            if backend is None:
                return False, None
            state = backend.source.load_source_snapshot(key).state
            return True, {
                "store_identity": state.store_identity,
                "incarnation": state.incarnation,
                "generation": state.generation,
            }
        except Exception:
            return False, None

    def discard_request_lock_if_idle(self, connection: ServerConnection) -> None:
        if connection in self._webui_connections:
            return
        if any(task_connection is connection for task_connection, _ in self.request_tasks):
            return
        self.request_locks.pop(connection, None)

    @staticmethod
    def _result_has_effect(result: WebUIRequestResult) -> bool:
        """Whether the owner may already have committed the mutation."""
        if result.outcome in {"applied", "partial"}:
            return True
        return (
            result.outcome is None
            and (result.status is None or 200 <= result.status < 300)
        )

    async def _invalidate_after_mutation(
        self,
        result: WebUIRequestResult,
        action: str,
        payload: dict[str, Any],
        *,
        origin: ServerConnection,
        request_id: str,
    ) -> None:
        """Invalidate projections for known effects even if the reply was lost."""
        if not self._result_has_effect(result):
            return
        try:
            await self.notify_state_changed(action, payload, origin=origin)
        except Exception:
            logger.exception(
                "WebUI mutation '{}' applied but projection refresh failed",
                action,
            )
            with suppress(Exception):
                await self._transport.webui_send_event(
                    origin,
                    "projection_error",
                    request_id=request_id,
                    action=action,
                    reason="projection_refresh_failed",
                )

    async def deliver_webui_request(
        self,
        connection: ServerConnection,
        request_id: str,
        operation_task: asyncio.Task[WebUIRequestResult],
        *,
        action: str,
        payload: dict[str, Any],
        sequence: bool = False,
        source_binding: dict[str, object] | None = None,
    ) -> None:
        try:
            delivery_error: BaseException | None = None
            if sequence:
                lock = self.request_locks.setdefault(connection, asyncio.Lock())
                async with lock:
                    result = await asyncio.shield(operation_task)
                    try:
                        await self._deliver_authorized_webui_result(
                            connection, request_id, action, payload, result, source_binding,
                        )
                    except BaseException as exc:
                        delivery_error = exc
                    await self._invalidate_after_mutation(
                        result, action, payload, origin=connection, request_id=request_id,
                    )
                    if delivery_error is not None:
                        raise delivery_error
                return
            result = await asyncio.shield(operation_task)
            try:
                await self._deliver_authorized_webui_result(
                    connection, request_id, action, payload, result, source_binding,
                )
            except BaseException as exc:
                delivery_error = exc
            # A known applied or partial effect invalidates the affected
            # projection even when the actor's response fails.
            await self._invalidate_after_mutation(
                result, action, payload, origin=connection, request_id=request_id,
            )
            if delivery_error is not None:
                raise delivery_error
        finally:
            self.request_tasks.pop((connection, request_id), None)
            self.discard_request_lock_if_idle(connection)

    async def _deliver_authorized_webui_result(
        self,
        connection: ServerConnection,
        request_id: str,
        action: str,
        payload: dict[str, Any],
        result: WebUIRequestResult,
        source_binding: dict[str, object] | None,
    ) -> None:
        if _typed_mobile_identity(connection) is not None:
            if not await self._transport.revalidate_mobile_connection(connection):
                return
        if connection not in self._webui_connections and not _mobile_mutation_allowed(
            connection, action
        ):
            await self.send_webui_response(
                connection,
                request_id,
                status=403,
                message=_mobile_denial_message(connection, action),
            )
            return
        if (
            action in _SOURCE_BOUND_PERMISSION_ACTIONS
            and self.gateway.session_manager is not None
            and self._permission_source_binding(payload) != source_binding
        ):
            await self.send_webui_response(
                connection,
                request_id,
                status=409,
                message="source_changed",
            )
            return
        authorization = self._http_router.authorize_webui_mutation(
            connection,
            action,
            payload,
        )
        if authorization is not None:
            await self.send_webui_response(
                connection,
                request_id,
                status=authorization.status,
                message=authorization.message,
            )
            return
        if action == _PREPARE_FULL_ACTION and result.status is None:
            failure = self._workspaces.validate_prepare_full_delivery(
                payload.get("chat_id"),
                result.result,
            )
            if failure is not None:
                await self.send_webui_response(
                    connection,
                    request_id,
                    status=503 if failure == "permission_state_unavailable" else 409,
                    message=failure,
                )
                return
        await self.send_webui_response(
            connection,
            request_id,
            result=result.result,
            status=result.status,
            message=result.message,
            workspace_scope=result.workspace_scope,
            workspace_permissions=result.workspace_permissions,
            outcome=result.outcome,
        )

    def _record_replay_result(
        self,
        request_id: str,
        result: WebUIRequestResult,
    ) -> None:
        """Persist a completed result, or uncertainty when the effect timed out."""
        if self._durable_replay is None:
            return
        try:
            if result.outcome == "uncertain":
                self._durable_replay.uncertain(request_id)
            else:
                self._durable_replay.complete(request_id, result)
        except Exception:
            logger.exception("WebUI replay ledger could not store the result")

    async def execute_webui_request(
        self,
        connection: ServerConnection,
        action: str,
        payload: dict[str, Any],
        *,
        source_binding: dict[str, object] | None = None,
        source_binding_captured: bool = False,
    ) -> WebUIRequestResult:
        if action == "delegation.child.control":
            return await self._control_delegated_child(connection, payload)
        timeout_s = 30.0
        try:
            from nanobot.remote.operations import load_registry

            spec = load_registry().mutations.get(action)
            if spec is not None:
                timeout_s = float(spec.timeout_s or 30.0)
                if spec.max_request_bytes is not None:
                    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    if len(encoded) > spec.max_request_bytes:
                        return WebUIRequestResult(
                            status=413,
                            message="request_too_large",
                            outcome="not_applied",
                        )
        except Exception:
            pass
        try:
            lock = self.request_locks.setdefault(connection, asyncio.Lock())
            async with lock:
                async def dispatch() -> Any:
                    if action in _SOURCE_BOUND_PERMISSION_ACTIONS:
                        return await self._http_router.dispatch_webui_mutation(
                            connection,
                            action,
                            payload,
                            expected_source_binding=source_binding,
                        )
                    if action == _PREPARE_FULL_ACTION and source_binding_captured:
                        return await self._http_router.dispatch_webui_mutation(
                            connection,
                            action,
                            payload,
                            expected_source_binding=source_binding,
                            check_expected_source_binding=True,
                        )
                    return await self._http_router.dispatch_webui_mutation(
                        connection,
                        action,
                        payload,
                    )

                try:
                    response = await asyncio.wait_for(dispatch(), timeout=timeout_s)
                except TimeoutError:
                    return WebUIRequestResult(
                        status=504,
                        message="timeout",
                        outcome="uncertain",
                    )
                status = response.status_code
                body = bytes(response.body).decode("utf-8", errors="replace").strip()
                if 200 <= status < 300:
                    try:
                        result = json.loads(body)
                    except json.JSONDecodeError:
                        return WebUIRequestResult(
                            status=502,
                            message="WebUI mutation returned an invalid response",
                            outcome="uncertain",
                        )
                    outcome = "applied"
                    if isinstance(result, dict):
                        declared = cast(dict[str, Any], result).get("outcome")
                        if declared in {"applied", "not_applied", "uncertain", "partial"}:
                            outcome = cast(str, declared)
                    if action == "sidebar.update" and isinstance(result, dict):
                        await self.broadcast_webui_event(
                            "sidebar_state_updated",
                            state=result,
                        )
                    return WebUIRequestResult(result=result, outcome=outcome)
                if action == "workspace.access.mode.set":
                    try:
                        decoded_value: object = json.loads(body)
                    except json.JSONDecodeError:
                        decoded_value = None
                    if isinstance(decoded_value, dict):
                        decoded = cast(dict[str, object], decoded_value)
                        error_value = decoded.get("error")
                        if isinstance(error_value, dict):
                            error = cast(dict[str, object], error_value)
                            error_message = error.get("message")
                            declared = decoded.get("outcome")
                            return WebUIRequestResult(
                                status=status,
                                message=(
                                    error_message
                                    if isinstance(error_message, str)
                                    else body or response.reason_phrase
                                ),
                                workspace_scope=decoded.get("workspace_scope"),
                                workspace_permissions=decoded.get("workspace_permissions"),
                                outcome=(
                                    cast(str, declared)
                                    if declared in {
                                        "applied", "not_applied", "uncertain", "partial",
                                    }
                                    else "not_applied"
                                ),
                            )
                return WebUIRequestResult(
                    status=status,
                    message=body or response.reason_phrase,
                    outcome="not_applied",
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("WebUI mutation '{}' failed", action)
            return WebUIRequestResult(
                status=500,
                message="WebUI mutation failed",
                outcome="uncertain",
            )

    async def _control_delegated_child(
        self,
        connection: ServerConnection,
        payload: dict[str, Any],
    ) -> WebUIRequestResult:
        """Authorize and route one child control from an attached parent chat."""
        parent_key = payload.get("parent_key")
        child_id = payload.get("child_id")
        control = payload.get("action")
        content = payload.get("content")
        runtime_generation = payload.get("runtime_generation")
        if not (
            isinstance(parent_key, str)
            and (
                parent_key.startswith(WEBUI_SESSION_STORAGE_PREFIX)
                or parent_key.startswith("delegated:")
            )
            and isinstance(child_id, str) and child_id
            and control in {"wait", "steer", "stop"}
            and (content is None or isinstance(content, str))
            and type(runtime_generation) is int
            and runtime_generation >= 0
        ):
            return WebUIRequestResult(
                status=400,
                message="invalid child control",
                outcome="not_applied",
            )
        sessions = self.gateway.session_manager
        if sessions is None:
            return WebUIRequestResult(
                status=503,
                message="delegation_control_unavailable",
                outcome="not_applied",
            )
        from nanobot.webui.ws_http import resolve_child_root_key

        root_key = resolve_child_root_key(sessions, parent_key)
        if root_key is None:
            return WebUIRequestResult(
                status=409,
                message="child_parent_stale",
                outcome="not_applied",
            )
        chat_id = root_key.split(":", 1)[1]
        if chat_id not in self._transport.webui_connection_chats(connection):
            return WebUIRequestResult(
                status=403,
                message="child_parent_not_attached",
                outcome="not_applied",
            )
        callback = getattr(self.gateway, "child_control", None)
        if callback is None:
            return WebUIRequestResult(
                status=503,
                message="delegation_control_unavailable",
                outcome="not_applied",
            )
        result = await callback(
            parent_key,
            child_id,
            str(control),
            content,
            runtime_generation,
        )
        if not result.get("ok"):
            return WebUIRequestResult(
                status=409,
                message=str(result.get("reason") or "child_control_failed"),
                outcome="not_applied",
            )
        return WebUIRequestResult(result=result, outcome="applied")

    async def send_webui_response(
        self,
        connection: ServerConnection,
        request_id: str,
        *,
        result: Any = None,
        status: int | None = None,
        message: str | None = None,
        workspace_scope: Any = None,
        workspace_permissions: Any = None,
        outcome: str | None = None,
    ) -> None:
        if status is None:
            await self._transport.webui_send_event(
                connection,
                "webui_response",
                request_id=request_id,
                ok=True,
                result=result,
                **({"outcome": outcome} if outcome is not None else {}),
            )
            return
        error: dict[str, Any] = {
            "status": status,
            "message": message or "WebUI mutation failed",
        }
        if workspace_scope is not None:
            error["workspace_scope"] = workspace_scope
        if workspace_permissions is not None:
            error["workspace_permissions"] = workspace_permissions
        if outcome is not None:
            error["outcome"] = outcome
        await self._transport.webui_send_event(
            connection,
            "webui_response",
            request_id=request_id,
            ok=False,
            error=error,
        )

    async def close(self) -> None:
        """Cancel command work and release application-owned gateway state."""
        delivery_tasks = tuple(self.request_tasks.values())
        operation_tasks = tuple(operation.task for operation in self.request_operations.values())
        for task in (*delivery_tasks, *operation_tasks):
            task.cancel()
        if delivery_tasks:
            await asyncio.gather(*delivery_tasks, return_exceptions=True)
        if operation_tasks:
            await asyncio.gather(*operation_tasks, return_exceptions=True)
        self.request_tasks.clear()
        self.request_locks.clear()
        self.request_operations.clear()
        self.gateway.tokens.clear()
        self.gateway.endpoint.clear()
        self._temporary_chats.close()
