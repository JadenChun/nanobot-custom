"""Compatibility re-exports for the pure workspace recovery schema."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from nanobot.security.desktop_permissions import (
    DesktopPermissionOwner,
    PermissionSourceBinding,
    WorkspacePermissionConflictError,
    WorkspacePermissionReference,
    WorkspaceTaskAccessReference,
)
from nanobot.security.folder_grants import FolderGrantsUnavailableError
from nanobot.session.workspace_access_continuation import (
    WorkspaceAccessContinuationRejectedError,
)
from nanobot.workspace_access_recovery import (
    WORKSPACE_ACCESS_CONTINUATION_METADATA_KEY,
    FolderAccessActionResult,
    FolderAccessEvent,
    FolderAccessKnownTarget,
    FolderAccessNeed,
    FolderAccessNeedMetadata,
    FolderAccessOperation,
    FolderAccessReason,
    FolderAccessRequest,
    FolderAccessRequestStatus,
    FolderAccessTarget,
    FolderAccessTask,
    FolderAccessUnavailableReason,
    FolderAccessUnknownTarget,
    WorkspaceAccessActionIdentity,
    WorkspaceAccessActionPayload,
    WorkspaceAccessContinuationToken,
    WorkspaceAccessExpectedScope,
    WorkspaceAccessRecord,
    WorkspaceAccessResolvePayload,
    WorkspaceAccessSourceBinding,
    WorkspaceAccessTaskAnchor,
    WorkspaceOnceActionResult,
    WorkspaceOnceReceipt,
    WorkspaceSandboxPayload,
    WorkspaceScopeMetadata,
    WorkspaceScopePayload,
    parse_folder_access_need,
    parse_workspace_access_action_payload,
    parse_workspace_access_continuation_token,
    parse_workspace_access_once_payload,
    parse_workspace_access_record,
)

if TYPE_CHECKING:
    from nanobot.agent.loop import TurnContext
    from nanobot.agent.turn import TurnContext as RunTurnContext
    from nanobot.bus.queue import MessageBus
    from nanobot.session.manager import SessionManager
    from nanobot.session.workspace_access_continuation import (
        WorkspaceAccessContinuationReference,
    )


class WorkspaceAccessActionError(Exception):
    """Stable public error for a local workspace recovery action."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def admit_workspace_access_continuation(
    sessions: SessionManager,
    context: RunTurnContext,
    focus_owner: Any,
    reference: WorkspaceAccessContinuationReference,
) -> bool:
    """Validate and consume one private continuation inside its source owner."""
    request = context.request
    metadata: Mapping[str, object] = request.metadata if request is not None else {}
    try:
        from nanobot.agent.turn import HistoryMode, TurnSource
        from nanobot.session.pending_followups import parse_pending_followup_records
        from nanobot.session.persistence_metadata import SOURCE_PENDING_FOLLOWUPS_METADATA_KEY
        from nanobot.session.webui_turns import (
            mark_websocket_turn_continuation_admitted,
            queued_websocket_turn_matches_continuation,
        )
        from nanobot.session.workspace_access_continuation import (
            is_workspace_access_continuation_reference,
        )
        from nanobot.webui.metadata import (
            WEBSOCKET_TURN_OWNER_METADATA_KEY,
        )
        from nanobot.webui.metadata import (
            WEBUI_TURN_METADATA_KEY as PUBLIC_TURN_ID,
        )
        from nanobot.webui.session_identity import is_webui_session_key

        key = context.session_key
        writer = context.source_writer
        owner = context.execution_owner
        route = request.delivery_target or request.route if request is not None else None
        turn_id = metadata.get(PUBLIC_TURN_ID)
        lifecycle_owner = metadata.get(WEBSOCKET_TURN_OWNER_METADATA_KEY)
        if (
            not is_workspace_access_continuation_reference(reference)
            or request is None
            or request.source is not TurnSource.GATEWAY
            or request.history_mode is not HistoryMode.SESSION
            or request.content != "Continue task"
            or request.approval.granted
            or (request.approval_grant is not None and request.approval_grant.granted)
            or key is None
            or not is_webui_session_key(key)
            or request.session_key != key
            or route is None
            or route.channel != "websocket"
            or route.chat_id is None
            or route.chat_id != key.split(":", 1)[1]
            or metadata.get("webui") is not True
            or not isinstance(turn_id, str)
            or not turn_id
            or turn_id != reference.turn_id
            or not isinstance(lifecycle_owner, str)
            or not lifecycle_owner
            or writer is None
            or owner is None
            or writer.owner is not owner
            or reference.session_key != key
        ):
            raise WorkspaceAccessContinuationRejectedError()

        if not queued_websocket_turn_matches_continuation(
            route.chat_id,
            lifecycle_owner,
            reference,
        ):
            raise WorkspaceAccessContinuationRejectedError()

        owner.validate_dispatch()
        snapshot = owner.read_snapshot()
        header_metadata = cast(dict[str, object], snapshot.header["metadata"])
        events = cast(list[dict[str, object]], snapshot.events)
        raw_record = header_metadata.get("workspace_access_request")
        record = parse_workspace_access_record(raw_record)
        latest_user = next(
            (event for event in reversed(events) if event.get("role") == "user"),
            None,
        )
        task_input = next(
            (event for event in events
             if event.get("_event_id") == reference.task_input_event_id),
            None,
        )
        raw_pending = header_metadata.get(SOURCE_PENDING_FOLLOWUPS_METADATA_KEY, [])
        pending_followups = parse_pending_followup_records(raw_pending)
        expected_scope = record["expected_scope"]
        selected_path = record["selected_path"]
        actual_binding: PermissionSourceBinding = {
            "store_identity": snapshot.state.store_identity,
            "incarnation": snapshot.state.incarnation,
            "generation": snapshot.state.generation,
        }
        if (
            record["status"] != "continuing"
            or record["access_request_id"] != reference.access_request_id
            or record["revision"] != reference.revision
            or record["continuation_turn_id"] != reference.turn_id
            or record["source_binding"] != reference.source_binding
            or actual_binding != reference.source_binding
            or record["task"]["turn_id"] != reference.task_turn_id
            or record["task_input_event_id"] != reference.task_input_event_id
            or record["latest_input_event_id"] != reference.latest_input_event_id
            or latest_user is None
            or latest_user.get("_event_id") != reference.latest_input_event_id
            or latest_user.get("_run_id") != record["blocked_run_id"]
            or task_input is None
            or task_input.get("role") != "user"
            or task_input.get("_command") is True
            or header_metadata.get("workspace_scope") != reference.expected_scope
            or expected_scope != reference.expected_scope
            or metadata.get("workspace_scope") != reference.expected_scope
            or selected_path is None
            or selected_path != reference.selected_path
            or selected_path != reference.project_path
            or type(record["resolved_grants_revision"]) is not int
            or record["resolved_grants_revision"] != reference.grants_revision
            or pending_followups
        ):
            raise WorkspaceAccessContinuationRejectedError()

        selection = record.get("selection_authority")
        task_reference = reference.task_reference
        if selection is not None and selection["kind"] == "once":
            permission_owner = sessions.desktop_permissions
            if (
                task_reference is None
                or selection.get("runtime_id") != permission_owner.runtime_id
                or selection.get("authority_epoch") != task_reference.epoch
                or not permission_owner.authorizes_once(
                    task_reference,
                    session_key=key,
                    source_binding=actual_binding,
                    task_turn_id=record["task"]["turn_id"],
                    task_input_event_id=record["task_input_event_id"],
                )
            ):
                raise WorkspaceAccessContinuationRejectedError()
        elif task_reference is not None:
            raise WorkspaceAccessContinuationRejectedError()

        if focus_owner is None:
            raise RuntimeError("workspace continuation access owner is unavailable")

        path = Path(selected_path)
        if not path.is_dir() or path.resolve(strict=True) != path:
            raise WorkspaceAccessContinuationRejectedError()
        grants = focus_owner.grants.snapshot()
        if grants.revision != record["resolved_grants_revision"]:
            raise WorkspaceAccessContinuationRejectedError()
        # Canonical admission runs before the execution turn binds its
        # ContextVar focus. Once is carried only by this private continuation
        # reference, so pass the exact validated proof at the final admission
        # check instead of relying on the saved Always grant projection.
        access = focus_owner.effective_access(
            path,
            permission_reference=task_reference,
        )
        if not access["read"]:
            raise WorkspaceAccessContinuationRejectedError()

        replacement = parse_workspace_access_record({
            **record,
            "revision": record["revision"] + 1,
            "status": "continued",
        })
        sessions.commit_workspace_access_request(
            key,
            expected_binding=reference.source_binding,
            expected_record=record,
            replacement_record=replacement,
            expected_latest_input_event_id=reference.latest_input_event_id,
        )
        if not mark_websocket_turn_continuation_admitted(
            route.chat_id,
            lifecycle_owner,
            reference,
        ):
            raise RuntimeError("workspace continuation lifecycle owner is unavailable")
        context.workspace_task_reference = task_reference
        return True
    except WorkspaceAccessContinuationRejectedError:
        raise
    except Exception:
        raise RuntimeError("Workspace access continuation admission failed") from None


def persist_folder_access_request(
    sessions: SessionManager,
    context: TurnContext,
) -> FolderAccessRequest | None:
    """Persist a pending card only from a completed, owned canonical WebUI turn."""
    from nanobot.agent.loop import TurnKind
    from nanobot.agent.turn import HistoryMode, TurnSource
    from nanobot.session.source import SourceValidationError
    from nanobot.webui.metadata import WEBUI_TURN_METADATA_KEY
    from nanobot.webui.session_identity import is_webui_session_key

    result = context.run_result
    run_context = context.run_context
    request = run_context.request if run_context is not None else None
    writer = run_context.source_writer if run_context is not None else None
    request_context = context.request_context
    if (
        result is None
        or result.stop_reason != "folder_access_required"
        or request is None
        or request.source is not TurnSource.GATEWAY
        or request.history_mode is not HistoryMode.SESSION
        or context.kind is not TurnKind.USER
        or context.ephemeral
        or context.msg.channel != "websocket"
        or context.msg.metadata.get("webui") is not True
        or not context.msg.is_user_input
        or request_context is None
        or not request_context.persistent_session
        or writer is None
        or run_context is None
        or writer.failed
        or not is_webui_session_key(context.session_key)
        or request.session_key != context.session_key
        or context.msg.session_key != context.session_key
    ):
        return None
    turn_id = context.msg.metadata.get(WEBUI_TURN_METADATA_KEY)
    if not isinstance(turn_id, str) or not 1 <= len(turn_id) <= 128:
        return None
    try:
        need = parse_folder_access_need(result.policy_metadata.get("folder_access"))
        scope = request_context.workspace_scope
        if scope is None:
            return None
        task_input_id = run_context.source_input_event_id
        writer.owner.validate_dispatch()
        snapshot = writer.owner.read_snapshot()
        if type(task_input_id) is not str:
            return None
        input_event = next(
            (event for event in snapshot.events if event.get("_event_id") == task_input_id),
            None,
        )
        latest_user = next(
            (event for event in reversed(snapshot.events) if event.get("role") == "user"),
            None,
        )
        if (
            input_event is None
            or input_event.get("role") != "user"
            or input_event.get("_run_id") != writer.run_id
            or latest_user is None
            or latest_user.get("_event_id") != task_input_id
            or input_event.get("_command") is True
        ):
            return None
        source_backend = sessions.source_backend(context.session_key)
        if source_backend is None or not writer.store.matches_backend(source_backend):
            return None
        existing_raw = snapshot.header["metadata"].get("workspace_access_request")
        existing = (
            None
            if existing_raw is None
            else parse_workspace_access_record(existing_raw)
        )
        task_event_id = task_input_id
        task_turn_id = turn_id
        if (
            existing is not None
            and existing["status"] == "continued"
            and existing["continuation_turn_id"] == turn_id
        ):
            original_task = next(
                (event for event in snapshot.events
                 if event.get("_event_id") == existing["task_input_event_id"]),
                None,
            )
            if (
                original_task is None
                or original_task.get("role") != "user"
                or original_task.get("_command") is True
            ):
                return None
            task_event_id = existing["task_input_event_id"]
            task_turn_id = existing["task"]["turn_id"]
            input_event = original_task
        content = input_event.get("content")
        excerpt = " ".join(content.split())[:240] if isinstance(content, str) else ""
        if not excerpt:
            return None
        state = snapshot.state
        record: WorkspaceAccessRecord = {
            "version": 1,
            "access_request_id": uuid4().hex,
            "revision": 1,
            "task": {"turn_id": task_turn_id},
            "blocked_turn_id": turn_id,
            "tool_call_id": None,
            "created_at_ms": time.time_ns() // 1_000_000,
            "status": "pending",
            "reason": need["reason"],
            "operation": need["operation"],
            "target": need["target"],
            "purpose": need["purpose"],
            "selected_path": None,
            "continuation_turn_id": None,
            "source_binding": {
                "store_identity": state.store_identity,
                "incarnation": state.incarnation,
                "generation": state.generation,
            },
            "task_input_event_id": task_event_id,
            "latest_input_event_id": task_input_id,
            "blocked_run_id": writer.run_id,
            "expected_scope": {
                "project_path": str(scope.project_path),
                "access_mode": scope.access_mode,
            },
            "resolved_grants_revision": None,
        }
        if "folder_proposal" in need:
            record["folder_proposal"] = need["folder_proposal"]
        record = parse_workspace_access_record(record)
        committed_binding = sessions.commit_workspace_access_request(
            context.session_key,
            expected_binding=record["source_binding"],
            expected_record=existing,
            replacement_record=record,
            expected_latest_input_event_id=task_input_id,
        )
        _ = committed_binding
        public: FolderAccessRequest = {
            "version": 1,
            "access_request_id": record["access_request_id"],
            "revision": record["revision"],
            "chat_id": context.session_key.split(":", 1)[1],
            "task": {"turn_id": task_turn_id, "excerpt": excerpt},
            "blocked_turn_id": turn_id,
            "tool_call_id": None,
            "created_at_ms": record["created_at_ms"],
            "status": "pending",
            "reason": record["reason"],
            "operation": record["operation"],
            "target": record["target"],
            "purpose": record["purpose"],
            "selected_path": None,
            "selected_access": None,
            "can_continue": False,
            "unavailable_reason": None,
            "continuation_turn_id": None,
            "selected_duration": None,
            "task_access_state": None,
        }
        if "folder_proposal" in record:
            public["folder_proposal"] = record["folder_proposal"]
        return public
    except (OSError, RuntimeError, SourceValidationError, TypeError, ValueError):
        # The already-recorded safe final remains the whole outcome when the
        # independent recovery metadata transaction cannot be proven.
        return None


class WorkspaceAccessRecoveryOwner:
    """Operate on one validated source record and its owning WebUI focus."""

    def __init__(
        self,
        sessions: SessionManager,
        workspaces: Any,
        bus: MessageBus | None = None,
    ) -> None:
        self.sessions = sessions
        self.workspaces = workspaces
        self.bus = bus

    @staticmethod
    def _key(chat_id: object) -> tuple[str, str]:
        from nanobot.webui.session_identity import is_valid_webui_chat_id, webui_session_key

        if not is_valid_webui_chat_id(chat_id):
            raise WorkspaceAccessActionError(400, "invalid_access_request")
        try:
            key = webui_session_key(chat_id)
        except (TypeError, ValueError):
            raise WorkspaceAccessActionError(400, "invalid_access_request") from None
        return key, chat_id

    def _load(
        self,
        chat_id: object,
    ) -> tuple[str, str, Any, WorkspaceAccessRecord]:
        from nanobot.session.manager import SessionStorageError
        from nanobot.session.webui_turns import WEBUI_SESSION_METADATA_KEY

        key, checked_chat = self._key(chat_id)
        metadata_payload = self.sessions.read_session_metadata(key)
        if metadata_payload is None:
            raise WorkspaceAccessActionError(404, "session_not_found")
        raw_metadata: object = metadata_payload.get("metadata")
        if type(raw_metadata) is not dict:
            raise WorkspaceAccessActionError(404, "session_not_found")
        metadata = cast(dict[str, object], raw_metadata)
        if metadata.get(WEBUI_SESSION_METADATA_KEY) is not True:
            raise WorkspaceAccessActionError(404, "session_not_found")
        backend = self.sessions.source_backend(key)
        if backend is None:
            raise WorkspaceAccessActionError(503, "access_recovery_unavailable")
        try:
            snapshot = backend.source.load_source_snapshot(key)
        except (FileNotFoundError, SessionStorageError):
            raise WorkspaceAccessActionError(404, "session_not_found") from None
        except Exception:
            raise WorkspaceAccessActionError(503, "access_recovery_unavailable") from None
        header_metadata = cast(dict[str, object], snapshot.header["metadata"])
        raw = header_metadata.get("workspace_access_request")
        if raw is None:
            raise WorkspaceAccessActionError(404, "access_request_not_found")
        try:
            record = parse_workspace_access_record(raw)
        except (TypeError, ValueError):
            raise WorkspaceAccessActionError(503, "access_recovery_unavailable") from None
        return key, checked_chat, snapshot, record

    @staticmethod
    def _binding(snapshot: Any) -> WorkspaceAccessSourceBinding:
        return {
            "store_identity": snapshot.state.store_identity,
            "incarnation": snapshot.state.incarnation,
            "generation": snapshot.state.generation,
        }

    def _focus_metadata(self, snapshot: Any) -> WorkspaceScopeMetadata | None:
        metadata = cast(dict[str, object], snapshot.header["metadata"])
        raw = metadata.get("workspace_scope")
        if raw is None:
            return None
        if type(raw) is not dict:
            raise WorkspaceAccessActionError(409, "focus_changed")
        scope = cast(dict[str, object], raw)
        project_path = scope.get("project_path")
        access_mode = scope.get("access_mode")
        if (
            set(scope) != {"project_path", "access_mode"}
            or not isinstance(project_path, str)
            or not Path(project_path).is_absolute()
            or access_mode not in {"restricted", "full"}
        ):
            raise WorkspaceAccessActionError(409, "focus_changed")
        return cast(WorkspaceScopeMetadata, scope)

    def _validate_current(
        self,
        key: str,
        chat_id: str,
        snapshot: Any,
        record: WorkspaceAccessRecord,
    ) -> Any:
        from nanobot.session.webui_turns import websocket_turn_wall_started_at

        if self._binding(snapshot) != record["source_binding"]:
            raise WorkspaceAccessActionError(409, "source_changed")
        task_input = next(
            (event for event in snapshot.events if event.get("_event_id") == record["task_input_event_id"]),
            None,
        )
        latest_user = next(
            (event for event in reversed(snapshot.events) if event.get("role") == "user"),
            None,
        )
        if task_input is None or task_input.get("role") != "user":
            raise WorkspaceAccessActionError(409, "task_changed")
        if latest_user is None or latest_user.get("_event_id") != record["latest_input_event_id"]:
            raise WorkspaceAccessActionError(409, "task_changed")
        content = task_input.get("content")
        if not isinstance(content, str) or not content.strip():
            raise WorkspaceAccessActionError(409, "context_unavailable")
        if snapshot.header["metadata"].get("execution_recovery_required") is True:
            raise WorkspaceAccessActionError(409, "context_unavailable")
        if snapshot.header["metadata"].get("execution_lease") is not None:
            raise WorkspaceAccessActionError(409, "chat_running")
        if websocket_turn_wall_started_at(chat_id) is not None:
            raise WorkspaceAccessActionError(409, "chat_running")
        from nanobot.session.pending_followups import parse_pending_followup_records
        from nanobot.session.persistence_metadata import SOURCE_PENDING_FOLLOWUPS_METADATA_KEY

        raw_followups = snapshot.header["metadata"].get(SOURCE_PENDING_FOLLOWUPS_METADATA_KEY, [])
        try:
            if parse_pending_followup_records(raw_followups):
                raise WorkspaceAccessActionError(409, "chat_running")
        except WorkspaceAccessActionError:
            raise
        except (TypeError, ValueError):
            raise WorkspaceAccessActionError(409, "context_unavailable") from None

        current_scope = self.workspaces.scope_for_session_key(key)
        if {
            "project_path": str(current_scope.project_path),
            "access_mode": current_scope.access_mode,
        } != record["expected_scope"]:
            raise WorkspaceAccessActionError(409, "focus_changed")
        return current_scope

    def get(self, chat_id: str) -> FolderAccessRequest | None:
        try:
            key, checked_chat, snapshot, record = self._load(chat_id)
        except WorkspaceAccessActionError as exc:
            if exc.status == 404 and exc.message == "access_request_not_found":
                return None
            raise
        return self._public_request(key, checked_chat, snapshot, record)

    def workspace_permissions(
        self,
        chat_id: str,
        *,
        controls_available: bool,
    ) -> Any:
        """Return current local permission state bound to the canonical chat source."""
        key, _checked_chat = self._key(chat_id)
        metadata_payload = self.sessions.read_session_metadata(key)
        metadata = metadata_payload.get("metadata") if metadata_payload is not None else None
        if type(metadata) is not dict or cast(dict[str, object], metadata).get("webui") is not True:
            raise WorkspaceAccessActionError(404, "session_not_found")
        backend = self.sessions.source_backend(key)
        owner = self.workspaces.permission_owner
        if backend is None or owner is None:
            raise WorkspaceAccessActionError(503, "permission_state_unavailable")
        try:
            snapshot = backend.source.load_source_snapshot(key)
            source = self._binding(snapshot)
        except Exception:
            raise WorkspaceAccessActionError(503, "permission_state_unavailable") from None

        current_record: WorkspaceAccessRecord | None = None
        raw_record = cast(dict[str, object], snapshot.header["metadata"]).get(
            "workspace_access_request",
        )
        if raw_record is not None:
            try:
                record = parse_workspace_access_record(raw_record)
            except (TypeError, ValueError):
                raise WorkspaceAccessActionError(503, "access_recovery_unavailable") from None
            if record["source_binding"] == source:
                current_record = record

        if current_record is not None:
            return self._permission_projection(
                key,
                current_record,
                controls_available=controls_available,
            )

        grant_store = self.workspaces.grant_store
        if grant_store is not None:
            try:
                task_active, task_unknown = grant_store.view.session_native_process_counts(
                    session_key=key,
                    kind="task_once",
                )
            except FolderGrantsUnavailableError:
                raise WorkspaceAccessActionError(503, "permission_state_unavailable") from None
            owner.observe_task_source_residual_counts(
                session_key=key,
                source_binding=source,
                active_processes=task_active,
                unknown_processes=task_unknown,
            )

        configured_sandbox = self.workspaces._configured_exec_sandbox
        supervisor_available = self.workspaces._native_supervisor_available
        unavailable_reason = (
            "controls_unavailable" if not controls_available
            else "configured_sandbox" if configured_sandbox
            else "supervisor_unavailable" if not supervisor_available
            else None
        )
        return self.workspaces._permission_projection(
            key,
            source,
            can_enable_full=controls_available and unavailable_reason is None,
            unavailable_reason=unavailable_reason,
        )

    def preview_permission_context(
        self,
        chat_id: str,
    ) -> tuple[DesktopPermissionOwner, WorkspacePermissionReference | None]:
        """Return the current source-bound capability for one local preview request."""
        key, _checked_chat = self._key(chat_id)
        owner = self.workspaces.permission_owner
        backend = self.sessions.source_backend(key)
        if owner is None or backend is None:
            raise WorkspaceAccessActionError(503, "permission_state_unavailable")
        try:
            snapshot = backend.source.load_source_snapshot(key)
            source = self._binding(snapshot)
        except Exception:
            raise WorkspaceAccessActionError(503, "permission_state_unavailable") from None

        full = owner.full_reference(session_key=key, source_binding=source)
        if full is not None and owner.authorizes_full(
            full,
            session_key=key,
            source_binding=source,
        ):
            return owner, full

        raw_record = cast(dict[str, object], snapshot.header["metadata"]).get(
            "workspace_access_request",
        )
        if raw_record is None:
            return owner, None
        try:
            record = parse_workspace_access_record(raw_record)
        except (TypeError, ValueError):
            raise WorkspaceAccessActionError(503, "access_recovery_unavailable") from None
        selection = record.get("selection_authority")
        if (
            record["source_binding"] != source
            or selection is None
            or selection["kind"] != "once"
        ):
            return owner, None
        return owner, self._live_once_reference(key, record)

    def _public_request(
        self,
        key: str,
        chat_id: str,
        snapshot: Any,
        record: WorkspaceAccessRecord,
    ) -> FolderAccessRequest:
        events = cast(list[dict[str, object]], snapshot.events)
        task_input = next(
            (event for event in events if event.get("_event_id") == record["task_input_event_id"]),
            None,
        )
        raw_content: object = task_input.get("content") if task_input is not None else ""
        excerpt = " ".join(raw_content.split())[:240] if isinstance(raw_content, str) else ""
        selected_access = None
        unavailable: FolderAccessUnavailableReason | None = None
        can_continue = False
        selected_duration = None
        task_access_state = None
        selection = record.get("selection_authority")
        if selection is not None and selection["kind"] == "once":
            selected_duration = "once"
            try:
                if self._binding(snapshot) != record["source_binding"]:
                    task_access_state = "expired"
                else:
                    permission_projection = self._permission_projection(key, record)
                    task_access = permission_projection["task_access"]
                    task_access_state = (
                        task_access["state"] if task_access is not None else "expired"
                    )
            except WorkspaceAccessActionError:
                task_access_state = None
        if not excerpt:
            unavailable = "context_unavailable"
        if record["status"] in {"pending", "ready"} and unavailable is None:
            try:
                self._validate_current(key, chat_id, snapshot, record)
                if record["status"] == "ready":
                    selected_path = record["selected_path"]
                    if selected_path is None:
                        raise WorkspaceAccessActionError(409, "path_missing")
                    path = Path(selected_path)
                    if not path.is_dir():
                        unavailable = "path_missing"
                    else:
                        canonical = path.resolve(strict=True)
                        if canonical != path:
                            unavailable = "path_changed"
                        else:
                            focus_owner = self.workspaces.focus_owner
                            selection = record.get("selection_authority")
                            if selection is not None and selection["kind"] == "once":
                                reference = self._live_once_reference(key, record)
                                if reference is None:
                                    unavailable = "task_access_expired"
                                    task_access_state = "expired"
                                else:
                                    selected_access = focus_owner.effective_access(
                                        path,
                                        permission_reference=reference,
                                    )
                                    task_access_state = "active"
                                    if not selected_access["read"]:
                                        unavailable = "task_access_expired"
                                        task_access_state = "expired"
                                    else:
                                        can_continue = True
                                        selected_duration = "once"
                            else:
                                selected_access = focus_owner.effective_access(path)
                                grants_revision = self.workspaces.grant_snapshot().revision
                                if grants_revision != record["resolved_grants_revision"]:
                                    unavailable = "access_changed"
                                elif not selected_access["read"]:
                                    unavailable = "access_changed"
                                else:
                                    can_continue = True
                                    selected_duration = "always"
            except WorkspaceAccessActionError as exc:
                unavailable = cast(FolderAccessUnavailableReason | None, {
                    "task_changed": "task_changed",
                    "focus_changed": "focus_changed",
                    "source_changed": "source_changed",
                    "context_unavailable": "context_unavailable",
                    "chat_running": "context_unavailable",
                }.get(exc.message))
            except Exception:
                unavailable = "access_changed"
        elif record["status"] in {"continuing", "continued"}:
            selection = record.get("selection_authority")
            if selection is not None and selection["kind"] == "once":
                selected_duration = "once"
                path_value = record["selected_path"]
                if path_value is None:
                    unavailable = "path_missing"
                elif self._binding(snapshot) != record["source_binding"]:
                    unavailable = "source_changed"
                    task_access_state = "expired"
                else:
                    path = Path(path_value)
                    try:
                        canonical = path.resolve(strict=True)
                    except (OSError, RuntimeError, ValueError):
                        canonical = None
                    if canonical is None or not canonical.is_dir():
                        unavailable = "path_missing"
                    elif canonical != path:
                        unavailable = "path_changed"
                    else:
                        permissions = self._permission_projection(key, record)
                        task_access = permissions["task_access"]
                        reference = self._live_once_reference(key, record)
                        if reference is not None:
                            focus_owner = self.workspaces.focus_owner
                            selected_access = focus_owner.effective_access(
                                path,
                                permission_reference=reference,
                            )
                            if selected_access["read"]:
                                task_access_state = "active"
                                if record["status"] == "continuing":
                                    unavailable = "continuation_unknown"
                            else:
                                selected_access = None
                                task_access_state = "expired"
                                unavailable = "task_access_expired"
                        elif task_access is not None:
                            task_access_state = task_access["state"]
                            unavailable = (
                                "permission_retiring"
                                if task_access_state == "retiring"
                                else "task_access_expired"
                            )
                        else:
                            task_access_state = "expired"
                            unavailable = "task_access_expired"
            elif record["status"] == "continuing":
                unavailable = "continuation_unknown"
        public: FolderAccessRequest = {
            "version": 1,
            "access_request_id": record["access_request_id"],
            "revision": record["revision"],
            "chat_id": chat_id,
            "task": {"turn_id": record["task"]["turn_id"], "excerpt": excerpt},
            "blocked_turn_id": record["blocked_turn_id"],
            "tool_call_id": record["tool_call_id"],
            "created_at_ms": record["created_at_ms"],
            "status": record["status"],
            "reason": record["reason"],
            "operation": record["operation"],
            "target": record["target"],
            "purpose": record["purpose"],
            "selected_path": record["selected_path"],
            "selected_access": selected_access,
            "can_continue": can_continue,
            "unavailable_reason": unavailable,
            "continuation_turn_id": record["continuation_turn_id"],
            "selected_duration": selected_duration,
            "task_access_state": task_access_state,
        }
        if "folder_proposal" in record:
            public["folder_proposal"] = record["folder_proposal"]
        return public

    def _live_once_reference(
        self,
        key: str,
        record: WorkspaceAccessRecord,
    ) -> WorkspaceTaskAccessReference | None:
        selection = record.get("selection_authority")
        owner = self.workspaces.permission_owner
        if (
            selection is None
            or selection["kind"] != "once"
            or owner is None
            or selection.get("runtime_id") != owner.runtime_id
        ):
            return None
        reference = owner.task_reference(
            session_key=key,
            source_binding=record["source_binding"],
            task_turn_id=record["task"]["turn_id"],
            task_input_event_id=record["task_input_event_id"],
        )
        if (
            reference is None
            or reference.epoch != selection.get("authority_epoch")
            or not owner.authorizes_once(
                reference,
                session_key=key,
                source_binding=record["source_binding"],
                task_turn_id=record["task"]["turn_id"],
                task_input_event_id=record["task_input_event_id"],
            )
        ):
            return None
        return reference

    def _permission_projection(
        self,
        key: str,
        record: WorkspaceAccessRecord,
        *,
        controls_available: bool = True,
    ) -> Any:
        owner = self.workspaces.permission_owner
        if owner is None:
            raise WorkspaceAccessActionError(503, "permission_state_unavailable")
        self.workspaces._observe_full_retirement_activity(
            key,
            record["source_binding"],
        )
        selection = record.get("selection_authority")
        task_identity = (
            record["task"]["turn_id"],
            record["task_input_event_id"],
        ) if selection is not None and selection["kind"] == "once" else (None, None)
        grant_store = self.workspaces.grant_store
        if grant_store is not None:
            try:
                full_active, full_unknown = grant_store.view.session_native_process_counts(
                    session_key=key,
                    kind="chat_full",
                )
                owner.observe_full_residual_counts(
                    session_key=key,
                    source_binding=record["source_binding"],
                    active_processes=full_active,
                    unknown_processes=full_unknown,
                )
                if selection is not None and selection["kind"] == "once":
                    task_active, task_unknown = grant_store.view.tagged_native_process_counts({
                        "kind": "task_once",
                        "runtime_id": selection["runtime_id"],
                        "owner_id": record["task"]["turn_id"],
                        "epoch": selection["authority_epoch"],
                        "session_key": key,
                        "store_identity": record["source_binding"]["store_identity"],
                        "incarnation": record["source_binding"]["incarnation"],
                        "generation": record["source_binding"]["generation"],
                        "task_input_event_id": record["task_input_event_id"],
                    })
                    owner.observe_task_residual_counts(
                        session_key=key,
                        source_binding=record["source_binding"],
                        task_turn_id=task_identity[0] or "",
                        task_input_event_id=task_identity[1] or "",
                        active_processes=task_active,
                        unknown_processes=task_unknown,
                    )
            except (FolderGrantsUnavailableError, KeyError, TypeError, ValueError):
                raise WorkspaceAccessActionError(503, "permission_state_unavailable") from None
        elif selection is not None and selection["kind"] == "once":
            owner.observe_task_residual_counts(
                session_key=key,
                source_binding=record["source_binding"],
                task_turn_id=task_identity[0] or "",
                task_input_event_id=task_identity[1] or "",
                active_processes=0,
                unknown_processes=1,
            )
        return owner.projection(
            session_key=key,
            source_binding=record["source_binding"],
            task_turn_id=task_identity[0],
            task_input_event_id=task_identity[1],
            can_enable_full=(
                controls_available
                and not self.workspaces._configured_exec_sandbox
                and self.workspaces._native_supervisor_available
            ),
            full_access_unavailable_reason=(
                "controls_unavailable" if not controls_available
                else "configured_sandbox" if self.workspaces._configured_exec_sandbox
                else "supervisor_unavailable" if not self.workspaces._native_supervisor_available
                else None
            ),
        )

    def _action_result(
        self,
        key: str,
        chat_id: str,
        snapshot: Any,
        record: WorkspaceAccessRecord,
        scope: Any,
        *,
        permission_reference: object | None = None,
    ) -> FolderAccessActionResult:
        permissions = self._permission_projection(key, record)
        task_access = permissions["task_access"]
        permission_retiring = (
            permissions["state"] == "retiring"
            or (task_access is not None and task_access["state"] == "retiring")
        )
        return {
            "workspace_access_request": self._public_request(key, chat_id, snapshot, record),
            "workspace_scope": self.workspaces.scope_payload(
                scope,
                permission_reference=permission_reference,
                permission_retiring=permission_retiring,
            ),
            "workspace_permissions": permissions,
        }

    def allow_once(self, raw_payload: object) -> WorkspaceOnceActionResult:
        try:
            payload = parse_workspace_access_once_payload(raw_payload)
        except (TypeError, ValueError):
            raise WorkspaceAccessActionError(400, "invalid_access_request") from None
        key, chat_id, snapshot, record = self._load(payload.get("chat_id"))
        self._check_action_identity(payload, record)
        if record.get("folder_proposal") is not None:
            raise WorkspaceAccessActionError(400, "folder_proposal_requires_confirmation")
        if record["status"] not in {"pending", "ready"}:
            raise WorkspaceAccessActionError(409, "access_request_inactive")
        current_scope = self._validate_current(key, chat_id, snapshot, record)
        if current_scope.access_mode != "restricted":
            raise WorkspaceAccessActionError(409, "full_access_active")
        owner = self.workspaces.permission_owner
        if owner is None:
            raise WorkspaceAccessActionError(503, "permission_state_unavailable")
        permission_projection = self._permission_projection(key, record)
        task_access = permission_projection["task_access"]
        if (
            permission_projection["state"] == "retiring"
            or (task_access is not None and task_access["state"] == "retiring")
            or permission_projection["retirement"] is not None
        ):
            raise WorkspaceAccessActionError(409, "permission_retiring")
        if permission_projection["mode"] == "full":
            raise WorkspaceAccessActionError(409, "full_access_active")
        if (
            payload["expected_runtime_id"] != owner.runtime_id
            or payload["expected_permission_revision"] != permission_projection["revision"]
        ):
            raise WorkspaceAccessActionError(409, "permission_changed")
        grants = self._grant_snapshot()
        if grants.revision != payload["expected_grants_revision"]:
            raise WorkspaceAccessActionError(409, "grants_changed")
        candidate = Path(payload["path"]).expanduser()
        try:
            canonical = candidate.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise WorkspaceAccessActionError(400, "invalid_access_request") from None
        if not canonical.is_dir():
            raise WorkspaceAccessActionError(400, "invalid_access_request")
        focus_owner = self.workspaces.focus_owner
        if focus_owner is None:
            raise WorkspaceAccessActionError(503, "access_recovery_unavailable")
        if focus_owner.grants.is_private_path(canonical):
            raise WorkspaceAccessActionError(403, "folder_access_required")
        scope = self.workspaces._scope_for_mode(canonical, "restricted")
        old_focus = self._focus_metadata(snapshot)
        source = record["source_binding"]
        replacement_holder: list[WorkspaceAccessRecord] = []
        grant_store = self.workspaces.grant_store
        native_counts = None
        if grant_store is not None:
            def task_native_counts() -> tuple[int, int]:
                return grant_store.view.session_native_process_counts(
                    session_key=key,
                    kind="task_once",
                )

            native_counts = task_native_counts

        def persist(epoch: int) -> None:
            latest_grants = self._grant_snapshot()
            if latest_grants.revision != payload["expected_grants_revision"]:
                raise WorkspaceAccessActionError(409, "grants_changed")
            replacement = parse_workspace_access_record({
                **record,
                "revision": record["revision"] + 1,
                "status": "ready",
                "selected_path": str(canonical),
                "expected_scope": {
                    "project_path": str(scope.project_path),
                    "access_mode": "restricted",
                },
                "resolved_grants_revision": payload["expected_grants_revision"],
                "selection_authority": {
                    "kind": "once",
                    "runtime_id": owner.runtime_id,
                    "authority_epoch": epoch,
                },
            })
            self._commit_action(
                key,
                snapshot,
                record,
                replacement,
                expected_workspace_scope=old_focus,
                replacement_workspace_scope={
                    "project_path": str(scope.project_path),
                    "access_mode": "restricted",
                },
            )
            replacement_holder.append(replacement)

        try:
            reference = owner.commit_once(
                session_key=key,
                source_binding=source,
                task_turn_id=record["task"]["turn_id"],
                task_input_event_id=record["task_input_event_id"],
                path=canonical,
                file_access=payload["file_access"],
                commands=payload["commands"],
                persist=persist,
                process_counts=native_counts,
            )
        except WorkspacePermissionConflictError as exc:
            raise WorkspaceAccessActionError(409, exc.message) from None
        except WorkspaceAccessActionError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError):
            raise WorkspaceAccessActionError(409, "access_request_changed") from None
        replacement = replacement_holder[0]
        latest_snapshot = self.sessions.source_backend(key)
        if latest_snapshot is None:
            raise WorkspaceAccessActionError(503, "access_recovery_unavailable")
        committed_snapshot = latest_snapshot.source.load_source_snapshot(key)
        receipt: WorkspaceOnceReceipt = {
            "version": 1,
            "duration": "once",
            "runtime_id": owner.runtime_id,
            "permission_revision": owner.revision,
            "access_request_id": replacement["access_request_id"],
            "request_revision": replacement["revision"],
            "task_turn_id": replacement["task"]["turn_id"],
            "submitted_path": payload["path"],
            "canonical_path": str(canonical),
            "file_access": payload["file_access"],
            "commands": payload["commands"],
        }
        result = self._action_result(
            key,
            chat_id,
            committed_snapshot,
            replacement,
            scope,
            permission_reference=reference,
        )
        result["workspace_permissions"] = self._permission_projection(key, replacement)
        return {**result, "once_receipt": receipt}

    def cancel(self, raw_payload: object) -> FolderAccessActionResult:
        payload = parse_workspace_access_action_payload(raw_payload, resolve=False)
        key, chat_id, snapshot, record = self._load(payload.get("chat_id"))
        self._check_action_identity(payload, record)
        if record["status"] not in {"pending", "ready"}:
            raise WorkspaceAccessActionError(409, "access_request_inactive")
        self._validate_current(key, chat_id, snapshot, record)
        replacement = parse_workspace_access_record({
            **record,
            "revision": record["revision"] + 1,
            "status": "cancelled",
        })
        self._commit_action(key, snapshot, record, replacement)
        selection = record.get("selection_authority")
        if selection is not None and selection["kind"] == "once":
            owner = self.workspaces.permission_owner
            if owner is not None:
                native_counts = None
                grant_store = self.workspaces.grant_store
                if grant_store is not None:
                    def task_native_counts() -> tuple[int, int]:
                        return grant_store.view.session_native_process_counts(
                            session_key=key,
                            kind="task_once",
                        )

                    native_counts = task_native_counts
                owner.retire_once(
                    session_key=key,
                    source_binding=record["source_binding"],
                    task_turn_id=record["task"]["turn_id"],
                    task_input_event_id=record["task_input_event_id"],
                    reason="task_cancelled",
                    active_turn=False,
                    active_processes=0,
                    unknown_processes=0,
                    process_counts=native_counts,
                )
        backend = self.sessions.source_backend(key)
        if backend is None:
            raise WorkspaceAccessActionError(503, "access_recovery_unavailable")
        latest = backend.source.load_source_snapshot(key)
        return self._action_result(
            key, chat_id, latest, replacement, self.workspaces.scope_for_session_key(key),
        )

    def add_folder(self, raw_payload: object) -> FolderAccessActionResult:
        from typing import Literal

        from pydantic import BaseModel, ConfigDict, Field, StrictInt

        class Approval(BaseModel):
            model_config = ConfigDict(extra="forbid", strict=True)
            chat_id: str
            access_request_id: str
            expected_revision: StrictInt
            expected_grants_revision: StrictInt = Field(ge=0)
            expected_catalog_revision: StrictInt = Field(ge=0)
            file_access: Literal["read_only", "read_write"]
            agent_map: bool

        approval = Approval.model_validate(raw_payload)
        key, chat_id, snapshot, record = self._load(approval.chat_id)
        self._check_action_identity(approval.model_dump(), record)
        if record["status"] not in {"pending", "ready"}:
            raise WorkspaceAccessActionError(409, "access_request_inactive")
        self._validate_current(key, chat_id, snapshot, record)
        proposal = record.get("folder_proposal")
        if proposal is None:
            raise WorkspaceAccessActionError(400, "invalid_access_request")
        entry = {**proposal["entry"], "agent_map": approval.agent_map}
        try:
            self.workspaces.edit_folder({
                "operation_id": record["access_request_id"],
                "expected_grants_revision": approval.expected_grants_revision,
                "expected_catalog_revision": approval.expected_catalog_revision,
                "path": proposal["entry"]["path"], "entry": entry,
                "access": approval.file_access,
            })
        except (OSError, ValueError) as exc:
            raise WorkspaceAccessActionError(409, str(exc)) from exc
        return self.resolve({
            "chat_id": chat_id, "access_request_id": record["access_request_id"],
            "expected_revision": record["revision"], "path": proposal["entry"]["path"],
            "expected_grants_revision": self.workspaces.grant_snapshot().revision,
        }, confirmed_proposal=True)

    def resolve(self, raw_payload: object, *, confirmed_proposal: bool = False) -> FolderAccessActionResult:
        payload = cast(WorkspaceAccessResolvePayload, parse_workspace_access_action_payload(
            raw_payload,
            resolve=True,
        ))
        key, chat_id, snapshot, record = self._load(payload.get("chat_id"))
        self._check_action_identity(payload, record)
        if record.get("folder_proposal") is not None and not confirmed_proposal:
            raise WorkspaceAccessActionError(400, "folder_proposal_requires_confirmation")
        if record["status"] not in {"pending", "ready"}:
            raise WorkspaceAccessActionError(409, "access_request_inactive")
        current_scope = self._validate_current(key, chat_id, snapshot, record)
        grants = self._grant_snapshot()
        expected_grants_revision = payload.get("expected_grants_revision")
        if grants.revision != expected_grants_revision:
            raise WorkspaceAccessActionError(409, "grants_changed")
        path_value = payload.get("path")
        if not path_value.strip() or len(path_value) > 2048:
            raise WorkspaceAccessActionError(400, "invalid_access_request")
        candidate = Path(path_value).expanduser()
        if not candidate.is_absolute():
            raise WorkspaceAccessActionError(400, "invalid_access_request")
        try:
            canonical = candidate.resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise WorkspaceAccessActionError(400, "invalid_access_request") from None
        if not canonical.is_dir():
            raise WorkspaceAccessActionError(400, "invalid_access_request")
        focus_owner = self.workspaces.focus_owner
        if focus_owner is None:
            raise WorkspaceAccessActionError(503, "access_recovery_unavailable")
        try:
            access = focus_owner.effective_access(canonical)
        except Exception:
            raise WorkspaceAccessActionError(503, "folder_grants_unavailable") from None
        if not access["read"]:
            raise WorkspaceAccessActionError(403, "folder_access_required")
        try:
            scope = focus_owner.resolve_scope(
                current=current_scope,
                action="select",
                path=str(canonical),
            )
        except Exception as exc:
            if getattr(exc, "message", None) == "folder_access_required":
                raise WorkspaceAccessActionError(403, "folder_access_required") from None
            raise WorkspaceAccessActionError(400, "invalid_access_request") from None
        latest_grants = self._grant_snapshot()
        if latest_grants.revision != expected_grants_revision:
            raise WorkspaceAccessActionError(409, "grants_changed")
        old_focus = self._focus_metadata(snapshot)
        replacement = parse_workspace_access_record({
            **record,
            "revision": record["revision"] + 1,
            "status": "ready",
            "selected_path": str(scope.project_path),
            "expected_scope": {
                "project_path": str(scope.project_path),
                "access_mode": scope.access_mode,
            },
            "resolved_grants_revision": expected_grants_revision,
            "selection_authority": {"kind": "always"},
        })
        self._commit_action(
            key,
            snapshot,
            record,
            replacement,
            expected_workspace_scope=old_focus,
            replacement_workspace_scope={
                "project_path": str(scope.project_path),
                "access_mode": scope.access_mode,
            },
        )
        backend = self.sessions.source_backend(key)
        if backend is None:
            raise WorkspaceAccessActionError(503, "access_recovery_unavailable")
        latest = backend.source.load_source_snapshot(key)
        return self._action_result(key, chat_id, latest, replacement, scope)

    async def continue_request(self, payload: dict[str, Any]) -> FolderAccessActionResult:
        action = parse_workspace_access_action_payload(payload, resolve=False)
        if self.bus is None:
            raise WorkspaceAccessActionError(503, "access_recovery_unavailable")
        key, chat_id, snapshot, record = self._load(action.get("chat_id"))
        self._check_action_identity(action, record)
        if record["status"] != "ready":
            raise WorkspaceAccessActionError(409, "access_request_inactive")
        current_scope = self._validate_current(key, chat_id, snapshot, record)
        selected_path = record["selected_path"]
        if selected_path is None:
            raise WorkspaceAccessActionError(409, "access_request_inactive")
        path = Path(selected_path)
        if not path.is_dir():
            raise WorkspaceAccessActionError(409, "path_missing")
        try:
            if path.resolve(strict=True) != path:
                raise WorkspaceAccessActionError(409, "path_changed")
            focus_owner = self.workspaces.focus_owner
            if focus_owner is None:
                raise WorkspaceAccessActionError(503, "access_recovery_unavailable")
            grants = self._grant_snapshot()
            if grants.revision != record["resolved_grants_revision"]:
                raise WorkspaceAccessActionError(409, "access_changed")
            selection = record.get("selection_authority")
            task_reference = (
                self._live_once_reference(key, record)
                if selection is not None and selection["kind"] == "once"
                else None
            )
            if (
                selection is not None and selection["kind"] == "once"
                and task_reference is None
            ):
                raise WorkspaceAccessActionError(409, "task_access_expired")
            access = focus_owner.effective_access(
                path,
                permission_reference=task_reference,
            )
        except WorkspaceAccessActionError:
            raise
        except Exception:
            raise WorkspaceAccessActionError(503, "folder_grants_unavailable") from None
        if not access["read"] or current_scope.project_path != path:
            raise WorkspaceAccessActionError(409, "access_changed")

        from nanobot.bus.events import InboundMessage
        from nanobot.session.webui_turns import (
            bind_queued_websocket_turn_continuation,
            clear_websocket_turn_if_current,
            register_queued_websocket_turn_if_idle,
        )
        from nanobot.session.workspace_access_continuation import (
            attach_workspace_access_continuation,
        )
        from nanobot.webui.metadata import (
            WEBSOCKET_TURN_OWNER_METADATA_KEY,
            WEBUI_TURN_METADATA_KEY,
        )

        continuation_turn_id = uuid4().hex
        owner = register_queued_websocket_turn_if_idle(chat_id, continuation_turn_id)
        if owner is None:
            raise WorkspaceAccessActionError(409, "chat_running")

        try:
            replacement = parse_workspace_access_record({
                **record,
                "revision": record["revision"] + 1,
                "status": "continuing",
                "continuation_turn_id": continuation_turn_id,
            })
            self._commit_action(key, snapshot, record, replacement)
            backend = self.sessions.source_backend(key)
            if backend is None:
                raise WorkspaceAccessActionError(503, "access_recovery_unavailable")
            latest = backend.source.load_source_snapshot(key)
            result = self._action_result(
                key,
                chat_id,
                latest,
                replacement,
                current_scope,
                permission_reference=task_reference,
            )

            from nanobot.security.workspace_access import WORKSPACE_SCOPE_METADATA_KEY

            message = InboundMessage(
                channel="websocket",
                sender_id="webui",
                chat_id=chat_id,
                content="Continue task",
                metadata={
                    "webui": True,
                    "_wants_stream": True,
                    WEBUI_TURN_METADATA_KEY: continuation_turn_id,
                    WEBSOCKET_TURN_OWNER_METADATA_KEY: owner,
                    WORKSPACE_SCOPE_METADATA_KEY: replacement["expected_scope"],
                },
                session_key_override=key,
                require_existing_session=True,
            )
            reference = attach_workspace_access_continuation(
                message,
                replacement,
                task_reference=task_reference,
            )
            if not bind_queued_websocket_turn_continuation(chat_id, owner, reference):
                raise WorkspaceAccessActionError(503, "access_recovery_unavailable")
        except Exception:
            # Only release this queued reservation. A concurrent owner, if any,
            # remains authoritative for the chat.
            clear_websocket_turn_if_current(chat_id, owner)
            raise

        try:
            await self.bus.publish_inbound(message)
        except Exception:
            # The durable reservation remains consumed; a failed enqueue is
            # ambiguous and must never become a retryable Continue action.
            raise WorkspaceAccessActionError(503, "continuation_unknown") from None
        return result

    def _grant_snapshot(self) -> Any:
        try:
            return self.workspaces.grant_snapshot()
        except Exception:
            raise WorkspaceAccessActionError(503, "folder_grants_unavailable") from None

    @staticmethod
    def _check_action_identity(
        payload: Mapping[str, object],
        record: WorkspaceAccessRecord,
    ) -> None:
        if (
            payload.get("access_request_id") != record["access_request_id"]
            or payload.get("expected_revision") != record["revision"]
        ):
            raise WorkspaceAccessActionError(409, "access_request_changed")

    def _commit_action(
        self,
        key: str,
        snapshot: Any,
        expected_record: WorkspaceAccessRecord,
        replacement: WorkspaceAccessRecord,
        *,
        expected_workspace_scope: WorkspaceScopeMetadata | None = None,
        replacement_workspace_scope: WorkspaceScopeMetadata | None = None,
    ) -> None:
        from nanobot.session.manager import SessionWriteConflict

        try:
            self.sessions.commit_workspace_access_request(
                key,
                expected_binding=self._binding(snapshot),
                expected_record=expected_record,
                replacement_record=replacement,
                expected_latest_input_event_id=expected_record["latest_input_event_id"],
                expected_workspace_scope=expected_workspace_scope,
                replacement_workspace_scope=replacement_workspace_scope,
            )
        except SessionWriteConflict:
            raise WorkspaceAccessActionError(409, "access_request_changed") from None
        except Exception:
            raise WorkspaceAccessActionError(503, "access_recovery_unavailable") from None

    async def publish(self, chat_id: str, result: FolderAccessActionResult) -> None:
        if self.bus is None:
            return
        from nanobot.bus.outbound_events import SessionUpdatedEvent

        await self.bus.publish_event(
            SessionUpdatedEvent(
                scope="metadata",
                workspace_scope=result["workspace_scope"],
                workspace_access_request=result["workspace_access_request"],
                workspace_permissions=result["workspace_permissions"],
            ),
            channel="websocket",
            chat_id=chat_id,
        )

__all__ = [
    "admit_workspace_access_continuation",
    "FolderAccessActionResult",
    "FolderAccessEvent",
    "FolderAccessNeed",
    "FolderAccessNeedMetadata",
    "FolderAccessOperation",
    "FolderAccessReason",
    "FolderAccessRequest",
    "FolderAccessRequestStatus",
    "FolderAccessTarget",
    "FolderAccessTask",
    "FolderAccessUnavailableReason",
    "FolderAccessKnownTarget",
    "FolderAccessUnknownTarget",
    "WorkspaceAccessActionIdentity",
    "WorkspaceAccessActionPayload",
    "WorkspaceAccessContinuationToken",
    "WorkspaceAccessExpectedScope",
    "WorkspaceAccessRecord",
    "WorkspaceAccessResolvePayload",
    "WorkspaceAccessSourceBinding",
    "WorkspaceAccessTaskAnchor",
    "WorkspaceSandboxPayload",
    "WorkspaceScopeMetadata",
    "WorkspaceScopePayload",
    "parse_folder_access_need",
    "parse_workspace_access_action_payload",
    "parse_workspace_access_continuation_token",
    "parse_workspace_access_record",
    "WORKSPACE_ACCESS_CONTINUATION_METADATA_KEY",
]
