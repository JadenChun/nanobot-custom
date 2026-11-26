"""Composition helpers for the embedded WebUI gateway."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger as default_logger

from kohaku.config.loader import get_config_path
from kohaku.context_repo import ContextRepoManager
from kohaku.context_repo.catalog import RepositoryCatalog
from kohaku.context_repo.policy import ResourceAccessPolicy
from kohaku.security.folder_grants import FolderGrantStore
from kohaku.webui.gateway_endpoint import WebUIGatewayEndpoint
from kohaku.webui.gateway_tokens import GatewayTokenStore
from kohaku.webui.ingress_policy import DEFAULT_WEBUI_INGRESS_POLICY, WebUIIngressPolicy
from kohaku.webui.media_gateway import WebUIMediaGateway
from kohaku.webui.session_projection import WebUISessionProjection
from kohaku.webui.settings_services import WebUISettingsServices
from kohaku.webui.temporary_chats import WebUITemporaryChats
from kohaku.webui.transcript import WebUITranscriptRecorder
from kohaku.webui.workspaces import WebUIWorkspaceController
from kohaku.webui.ws_http import GatewayHTTPHandler

if TYPE_CHECKING:
    from kohaku.agent.run_control import RunStopTarget
    from kohaku.agent.stop_target import StopTargetReader
    from kohaku.bus.outbound_events import RunState
    from kohaku.bus.queue import MessageBus
    from kohaku.channels.websocket.runtime import WebSocketConfig
    from kohaku.config.schema import RemoteAccessConfig
    from kohaku.cron.service import CronService
    from kohaku.session.manager import SessionManager
    from kohaku.triggers.local_store import LocalTriggerStore
    from kohaku.webui.remote_owner import GatewayRemoteOwner


@dataclass(frozen=True)
class GatewayServices:
    """Explicit dependencies shared by WebSocket transport and HTTP routes."""

    http: GatewayHTTPHandler
    endpoint: WebUIGatewayEndpoint
    settings: WebUISettingsServices
    tokens: GatewayTokenStore
    media: WebUIMediaGateway
    ingress: WebUIIngressPolicy
    transcripts: WebUITranscriptRecorder
    workspaces: WebUIWorkspaceController
    temporary_chats: WebUITemporaryChats
    session_projection: WebUISessionProjection
    session_manager: SessionManager | None
    child_control: Callable[[str, str, str, str | None], Awaitable[dict[str, Any]]] | None
    run_stop_target: Callable[[RunStopTarget], Awaitable[bool]] | None
    cron_service: CronService | None
    local_trigger_store: LocalTriggerStore | None
    cron_pending_job_ids: Callable[[str], set[str]] | None
    local_trigger_pending_ids: Callable[[str], set[str]] | None
    remote_owner: GatewayRemoteOwner | None = None


def build_gateway_services(
    *,
    config: WebSocketConfig,
    bus: MessageBus,
    session_manager: SessionManager | None,
    stop_target_for_key: StopTargetReader | None = None,
    run_state_for_key: Callable[[str], RunState | None] | None = None,
    run_stop_target: Callable[[RunStopTarget], Awaitable[bool]] | None = None,
    child_control: Callable[[str, str, str, str | None], Awaitable[dict[str, Any]]] | None = None,
    static_dist_path: Path | None = None,
    workspace_path: Path,
    context_manager: ContextRepoManager | None = None,
    default_restrict_to_workspace: bool,
    configured_exec_sandbox: bool = False,
    config_path: Path | None = None,
    runtime_model_name: Callable[[], str | None] | None,
    refresh_runtime_config: Callable[[], None] | None = None,
    runtime_surface: str,
    runtime_capabilities_overrides: dict[str, Any] | None,
    disabled_skills: set[str] | None = None,
    cron_service: CronService | None = None,
    local_trigger_store: LocalTriggerStore | None = None,
    cron_pending_job_ids: Callable[[str], set[str]] | None = None,
    local_trigger_pending_ids: Callable[[str], set[str]] | None = None,
    channel_feature_action: Callable[..., Any] | None = None,
    channel_runtime_status: Callable[[], dict[str, Any]] | None = None,
    mcp_runtime_status: Callable[[], Mapping[str, str]] | None = None,
    mcp_reload: Callable[[], Awaitable[dict[str, Any]]] | None = None,
    skill_state_action: Callable[[set[str]], None] | None = None,
    recovery_action: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
    logger: Any = default_logger,
    mobile_edge: Any | None = None,
    remote_access: "RemoteAccessConfig | None" = None,
) -> GatewayServices:
    settings = WebUISettingsServices.create(
        config_path or get_config_path(),
        rename_model_preset=(
            session_manager.rename_model_preset
            if session_manager is not None
            else None
        ),
        refresh_runtime_config=refresh_runtime_config,
    )
    tokens = GatewayTokenStore()
    ingress = DEFAULT_WEBUI_INGRESS_POLICY
    minimum_frame_bytes = ingress.minimum_full_policy_frame_bytes()
    if config.max_message_bytes < minimum_frame_bytes:
        logger.warning(
            "WebSocket maxMessageBytes={} is below the WebUI ingress policy capacity={}; "
            "policy-valid messages may still hit the transport frame guard",
            config.max_message_bytes,
            minimum_frame_bytes,
        )
    media = WebUIMediaGateway(
        workspace_path=workspace_path,
        logger=logger,
        attachment_limits=ingress.attachments,
    )
    transcripts = WebUITranscriptRecorder(log=logger, sessions=session_manager)
    sessions_root = session_manager.sessions_root if session_manager is not None else None
    instance_root = (
        config_path.expanduser().resolve(strict=False).parent
        if config_path is not None
        else None
    )
    grants = FolderGrantStore.for_workspace(
        workspace_path,
        sessions_root=sessions_root,
        instance_root=instance_root,
        config_path=config_path,
    )
    catalog = RepositoryCatalog.for_workspace(workspace_path, sessions_root=sessions_root)
    managed_context = context_manager or ContextRepoManager()
    workspaces = WebUIWorkspaceController(
        session_manager=session_manager,
        default_workspace=workspace_path,
        default_restrict_to_workspace=default_restrict_to_workspace,
        grant_store=grants,
        catalog=catalog,
        context_manager=managed_context,
        bus=bus,
        configured_exec_sandbox=configured_exec_sandbox,
    )
    focus_owner = workspaces.focus_owner
    skills_policy = focus_owner.policy if focus_owner is not None else ResourceAccessPolicy(
        workspace=workspace_path,
        context_manager=managed_context,
        restrict_to_workspace=default_restrict_to_workspace,
        folder_grants=grants.view,
        sandbox_default=True,
    )
    media.set_resource_policy(focus_owner.policy if focus_owner is not None else None)
    temporary_chats = WebUITemporaryChats(
        bus=bus,
        session_manager=session_manager,
        workspaces=workspaces,
        logger=logger,
    )
    workspaces.set_temporary_owners(
        binding=lambda chat_id: temporary_chats.binding(None, chat_id),
        set_mode=lambda chat_id, mode: temporary_chats.set_access_mode(chat_id, mode),
    )
    session_projection = WebUISessionProjection(
        session_manager,
        stop_target_for_key=stop_target_for_key,
        run_state_for_key=run_state_for_key,
        log=logger,
    )
    http = GatewayHTTPHandler(
        config=config,
        session_manager=session_manager,
        static_dist_path=static_dist_path,
        runtime_model_name=runtime_model_name,
        runtime_surface=runtime_surface,
        runtime_capabilities_overrides=runtime_capabilities_overrides,
        bus=bus,
        tokens=tokens,
        media=media,
        ingress=ingress,
        workspaces=workspaces,
        settings=settings,
        skills_workspace_path=workspace_path,
        context_manager=managed_context,
        resource_policy=skills_policy,
        default_restrict_to_workspace=default_restrict_to_workspace,
        disabled_skills=disabled_skills,
        cron_service=cron_service,
        local_trigger_store=local_trigger_store,
        cron_pending_job_ids=cron_pending_job_ids,
        local_trigger_pending_ids=local_trigger_pending_ids,
        channel_feature_action=channel_feature_action,
        channel_runtime_status=channel_runtime_status,
        mcp_runtime_status=mcp_runtime_status,
        mcp_reload=mcp_reload,
        skill_state_action=skill_state_action,
        recovery_action=recovery_action,
        stop_target_for_key=stop_target_for_key,
        run_state_for_key=run_state_for_key,
        log=logger,
    )
    remote_owner = None
    if remote_access is not None:
        from kohaku.remote.client import RemoteAccessClient
        from kohaku.remote.composition import build_mobile_edge, owner_socket_path
        from kohaku.remote.permissions import OwnerPermissionAdapter
        from kohaku.webui.remote_owner import GatewayRemoteOwner

        remote_owner = GatewayRemoteOwner(remote_access, settings.config)
        settings.config.set_remote_access_enabled = remote_owner.set_enabled
        if mobile_edge is None:
            mobile_edge = build_mobile_edge(
                remote_access,
                http,
                media,
                instance_root=settings.config.path.parent,
            )
        owner_client = RemoteAccessClient(
            owner_socket_path(remote_access, instance_root=settings.config.path.parent),
            timeout=remote_access.owner_timeout_seconds,
        )
        http.owner_permission_adapter = OwnerPermissionAdapter(owner_client)
        http.owner_audit_client = owner_client
    endpoint = WebUIGatewayEndpoint(
        config=config,
        http=http,
        tokens=tokens,
        mobile_edge=mobile_edge,
        mobile_enabled=remote_owner.is_enabled if remote_owner is not None else None,
    )
    return GatewayServices(
        http=http,
        endpoint=endpoint,
        settings=settings,
        tokens=tokens,
        media=media,
        ingress=ingress,
        transcripts=transcripts,
        workspaces=workspaces,
        temporary_chats=temporary_chats,
        session_projection=session_projection,
        session_manager=session_manager,
        child_control=child_control,
        run_stop_target=run_stop_target,
        cron_service=cron_service,
        local_trigger_store=local_trigger_store,
        cron_pending_job_ids=cron_pending_job_ids,
        local_trigger_pending_ids=local_trigger_pending_ids,
        remote_owner=remote_owner,
    )
