"""Persisted WebUI project workspace state."""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict, cast

from loguru import logger

from nanobot.config.paths import get_webui_dir
from nanobot.context_repo.catalog import RepositoryCatalog, RepositoryEntry
from nanobot.context_repo.discovery import EffectiveWorkspaceAccess, RepositoryDiscoveryPayload
from nanobot.context_repo.manager import ContextRepoManager
from nanobot.context_repo.policy import ResourceAccessPolicy
from nanobot.security.desktop_permissions import (
    DesktopPermissionOwner,
    PermissionFullUnavailableReason,
    PermissionSourceBinding,
    WorkspacePermissionConflictError,
    WorkspacePermissionProjection,
    WorkspacePermissionReference,
    WorkspacePermissionUnavailableError,
)
from nanobot.security.folder_grants import (
    FolderFileAccess,
    FolderGrantInput,
    FolderGrantSnapshot,
    FolderGrantStore,
    FolderGrantsUnavailableError,
    FolderGrantUpsertResult,
)
from nanobot.security.workspace_access import (
    WORKSPACE_SCOPE_METADATA_KEY,
    WorkspaceScope,
    WorkspaceScopeError,
    build_workspace_scope,
    validate_workspace_scope_payload,
)
from nanobot.session.workspace_focus import WorkspaceFocusOwner
from nanobot.webui.session_identity import webui_session_key

if TYPE_CHECKING:
    from nanobot.session.manager import SessionManager

from nanobot.workspace_access_recovery import (
    WorkspaceModeActionResult,
    WorkspaceModeConflict,
    WorkspaceModeConflictErrorPayload,
    WorkspaceSandboxPayload,
    WorkspaceScopePayload,
)

WEBUI_WORKSPACE_STATE_SCHEMA_VERSION = 2
_MAX_STATE_FILE_BYTES = 128 * 1024
_DEFAULT_ACCESS_MODES = {"default", "full"}
_LEGACY_RESTRICTED_DEFAULT_ACCESS_MODE = "restricted"
_WEBUI_SCOPE_CHANNEL = "websocket"
_MAX_DRAFT_SCOPES = 128


class WorkspaceGrantReceipt(TypedDict):
    """Operation-bound identity returned only with a successful grant upsert."""

    submitted_path: str
    canonical_path: str
    file_access: FolderFileAccess
    commands: bool
    grants_revision: int


def _scope_change_is_non_escalating(current: WorkspaceScope, requested: WorkspaceScope) -> bool:
    """Allow a remote request only when it keeps the project and does not add access."""
    return (
        requested.project_path == current.project_path
        and (not current.restrict_to_workspace or requested.restrict_to_workspace)
    )


def webui_workspace_state_path() -> Path:
    return get_webui_dir() / "workspace-state.json"


def default_webui_workspace_state() -> dict[str, Any]:
    return {
        "schema_version": WEBUI_WORKSPACE_STATE_SCHEMA_VERSION,
        "default_access_mode": "default",
        "updated_at": None,
    }


def normalize_webui_workspace_state(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    raw = cast(dict[str, Any], raw)
    state = default_webui_workspace_state()
    updated_at = raw.get("updated_at")
    state["updated_at"] = updated_at if isinstance(updated_at, str) else None
    default_access_mode = raw.get("default_access_mode")
    if default_access_mode in _DEFAULT_ACCESS_MODES:
        state["default_access_mode"] = default_access_mode
    return state


def read_webui_workspace_state() -> dict[str, Any]:
    path = webui_workspace_state_path()
    if not path.is_file():
        return default_webui_workspace_state()
    try:
        if path.stat().st_size > _MAX_STATE_FILE_BYTES:
            logger.warning("webui workspace state too large, ignoring: {}", path)
            return default_webui_workspace_state()
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("read webui workspace state failed {}: {}", path, e)
        return default_webui_workspace_state()
    return normalize_webui_workspace_state(raw)


def write_webui_workspace_state(raw: dict[str, Any]) -> dict[str, Any]:
    state = normalize_webui_workspace_state(raw)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    encoded = json.dumps(
        state,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > _MAX_STATE_FILE_BYTES:
        raise ValueError("workspace state is too large")

    path = webui_workspace_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "wb") as f:
        f.write(encoded)
        f.write(b"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return state
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return state


def read_webui_default_access_mode() -> str:
    state = read_webui_workspace_state()
    mode = state.get("default_access_mode")
    return mode if mode in _DEFAULT_ACCESS_MODES else "default"


def write_webui_default_access_mode(mode: str) -> bool:
    if mode == _LEGACY_RESTRICTED_DEFAULT_ACCESS_MODE:
        mode = "default"
    if mode not in _DEFAULT_ACCESS_MODES:
        raise ValueError("default access mode must be default or full")
    state = read_webui_workspace_state()
    changed = state.get("default_access_mode") != mode
    if changed:
        state["default_access_mode"] = mode
        write_webui_workspace_state(state)
    return changed


def default_scope_for_webui(
    default_workspace: Path,
    default_restrict_to_workspace: bool,
) -> WorkspaceScope:
    # Full is chat-scoped process-local authority in permission v2. A saved
    # legacy default is presentation only and cannot authorize a new chat.
    return build_workspace_scope(
        default_workspace,
        "restricted",
        source_channel=_WEBUI_SCOPE_CHANNEL,
    )


def workspaces_payload(
    *,
    default_workspace: Path,
    default_restrict_to_workspace: bool,
    controls_available: bool,
    folder_picker_available: bool = False,
) -> dict[str, Any]:
    default_access_mode = read_webui_default_access_mode()
    default_scope = default_scope_for_webui(default_workspace, default_restrict_to_workspace)
    full_access_unavailable_reason = "controls_unavailable" if not controls_available else None
    payload = {
        "schema_version": WEBUI_WORKSPACE_STATE_SCHEMA_VERSION,
        "agent_home": str(Path(default_workspace).expanduser().resolve(strict=False)),
        "managed_access": False,
        "management": None,
        "default_access_mode": default_access_mode,
        "default_scope": {
            **default_scope.payload(),
            "effective_access": {
                "policy": "legacy",
                "source": "agent_home",
                "grant_root": None,
                "boundary": "folder" if default_scope.restrict_to_workspace else "unrestricted",
                "exists": default_scope.project_path.is_dir(),
                "read": default_scope.project_path.is_dir(),
                "write": default_scope.project_path.is_dir(),
                "commands": {
                    "requested": True,
                    "available": True,
                    "backend": "native",
                    "reason": None,
                },
            },
        },
        "controls": {
            "permission_v2": True,
            "can_change_project": controls_available,
            "can_use_full_access": controls_available and full_access_unavailable_reason is None,
            "full_access_unavailable_reason": full_access_unavailable_reason,
            "can_pick_folder": folder_picker_available,
            "can_manage_folders": False,
            "can_manage_catalog": False,
            "can_discover_repositories": False,
        },
    }
    return payload


class WebUIWorkspaceController:
    """Own WebUI project scope persistence and validation."""

    def __init__(
        self,
        *,
        session_manager: SessionManager | None,
        default_workspace: Path,
        default_restrict_to_workspace: bool,
        grant_store: FolderGrantStore | None = None,
        catalog: RepositoryCatalog | None = None,
        context_manager: ContextRepoManager | None = None,
        bus: Any = None,
        configured_exec_sandbox: bool = False,
        native_supervisor_available: bool | None = None,
    ) -> None:
        self._sessions = session_manager
        if session_manager is not None:
            session_manager.set_permission_lifecycle_observer(
                self.retire_permissions_for_lifecycle,
            )
        self.permission_owner: DesktopPermissionOwner | None = (
            session_manager.desktop_permissions if session_manager is not None else None
        )
        self._default_workspace = default_workspace
        self._default_restrict_to_workspace = default_restrict_to_workspace
        self._configured_exec_sandbox = configured_exec_sandbox
        if native_supervisor_available is None:
            from nanobot.agent.tools.native_process import (
                native_supervisor_available as supervisor_available,
            )

            native_supervisor_available = supervisor_available()
        self._native_supervisor_available = native_supervisor_available
        self._draft_scopes: OrderedDict[str, WorkspaceScope] = OrderedDict()
        self.grant_store = grant_store
        self.catalog = catalog or RepositoryCatalog.for_workspace(
            default_workspace,
            sessions_root=(session_manager.sessions_root if session_manager is not None else None),
        )
        managed_context = context_manager or ContextRepoManager()
        self.focus_owner = (
            WorkspaceFocusOwner(
                sessions=session_manager,
                agent_home=default_workspace,
                grants=grant_store.view,
                policy=ResourceAccessPolicy(
                    workspace=default_workspace,
                    context_manager=managed_context,
                    restrict_to_workspace=default_restrict_to_workspace,
                    folder_grants=grant_store.view,
                ),
                catalog=self.catalog,
                bus=bus,
            )
            if session_manager is not None and grant_store is not None
            else None
        )

    def retire_permissions_for_lifecycle(
        self,
        session_key: str,
        reason: Literal["session_reset", "session_deleted"],
    ) -> None:
        """Fence Once and Full owners before reset or deletion can commit."""
        owner = self.permission_owner
        source = self._source_binding(session_key)
        if owner is None or source is None:
            return
        task_counts = None
        full_counts = None
        grant_store = self.grant_store
        if grant_store is not None:
            def task_count_snapshot() -> tuple[int, int]:
                return grant_store.view.session_native_process_counts(
                    session_key=session_key,
                    kind="task_once",
                )

            def full_count_snapshot() -> tuple[int, int]:
                return grant_store.view.session_native_process_counts(
                    session_key=session_key,
                    kind="chat_full",
                )

            task_counts = task_count_snapshot
            full_counts = full_count_snapshot
        owner.retire_source_for_lifecycle(
            session_key=session_key,
            source_binding=source,
            reason=reason,
            task_process_counts=task_counts,
            full_process_counts=full_counts,
        )

    def default_scope(self) -> WorkspaceScope:
        scope = default_scope_for_webui(
            self._default_workspace,
            self._default_restrict_to_workspace,
        )
        if self.focus_owner is not None and self.focus_owner.managed and scope.access_mode == "full":
            return build_workspace_scope(
                self._default_workspace, "restricted", source_channel=_WEBUI_SCOPE_CHANNEL,
            )
        return scope

    def full_access_unavailable_reason(self, *, controls_available: bool) -> str | None:
        if not controls_available:
            return "controls_unavailable"
        return None

    def clamp_effective_scope(self, scope: WorkspaceScope) -> WorkspaceScope:
        """Never restore a persisted Full scope as live permission."""
        if scope.access_mode != "full":
            return scope
        return build_workspace_scope(
            scope.project_path,
            "restricted",
            source_channel=_WEBUI_SCOPE_CHANNEL,
            preserve_path_identity=True,
        )

    def restricted_default_scope(self) -> WorkspaceScope:
        """Return the default workspace with access restricted for this request."""
        return build_workspace_scope(
            self._default_workspace,
            "restricted",
            source_channel=_WEBUI_SCOPE_CHANNEL,
        )

    def _scope_from_metadata_value(
        self,
        raw_scope: object,
        *,
        default_scope: WorkspaceScope | None = None,
        preserve_full: bool = False,
    ) -> WorkspaceScope:
        try:
            scope = validate_workspace_scope_payload(
                raw_scope,
                default_workspace=self._default_workspace,
                default_restrict_to_workspace=self._default_restrict_to_workspace,
                source_channel=_WEBUI_SCOPE_CHANNEL,
                allow_missing=bool(self.focus_owner and self.focus_owner.managed),
                preserve_path_identity=bool(self.focus_owner and self.focus_owner.managed),
            )
            return scope if preserve_full else self.clamp_effective_scope(scope)
        except WorkspaceScopeError:
            return default_scope if default_scope is not None else self.default_scope()

    def scope_for_indexed_metadata(
        self,
        raw_scope: object,
        *,
        scope_present: bool,
        default_scope: WorkspaceScope,
    ) -> WorkspaceScope:
        """Resolve a sidebar-only metadata snapshot without an authority-store read."""
        if not scope_present:
            return default_scope
        return self._scope_from_metadata_value(raw_scope, default_scope=default_scope)

    def scope_for_session_key(self, session_key: str) -> WorkspaceScope:
        draft = self._draft_scopes.get(session_key)
        if draft is not None:
            self._draft_scopes.move_to_end(session_key)
            return draft
        if self._sessions is None:
            return self.default_scope()
        data = self._sessions.read_session_metadata(session_key)
        if not isinstance(data, dict):
            return self.default_scope()
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict) or WORKSPACE_SCOPE_METADATA_KEY not in metadata:
            return self.default_scope()
        metadata_data = cast(dict[str, Any], metadata)
        scope = self._scope_from_metadata_value(
            cast(object, metadata_data.get(WORKSPACE_SCOPE_METADATA_KEY)),
            preserve_full=True,
        )
        if scope.access_mode == "full":
            source = self._source_binding(session_key)
            reference = (
                self.permission_owner.full_reference(
                    session_key=session_key,
                    source_binding=source,
                )
                if source is not None and self.permission_owner is not None
                else None
            )
            retiring = False
            if source is not None and self.permission_owner is not None:
                projection = self.permission_owner.projection(
                    session_key=session_key,
                    source_binding=source,
                )
                retiring = projection["state"] == "retiring"
            if reference is None and not retiring:
                return self.clamp_effective_scope(scope)
        return scope

    def folder_grants_allowed_for_session(self, session_key: str) -> bool:
        """Keep temporary sessions from inheriting this instance's external grants."""
        session = self._sessions.get_cached(session_key) if self._sessions is not None else None
        return session is None or session.policy.persist

    def payload(
        self,
        *,
        controls_available: bool,
        folder_picker_available: bool = False,
        management_available: bool = False,
    ) -> dict[str, Any]:
        payload = workspaces_payload(
            default_workspace=self._default_workspace,
            default_restrict_to_workspace=self._default_restrict_to_workspace,
            controls_available=controls_available,
            folder_picker_available=folder_picker_available,
        )
        owner = self.focus_owner
        if owner is not None:
            payload["managed_access"] = owner.managed
            payload["default_scope"] = self.scope_payload(self.default_scope())
            reason = self.full_access_unavailable_reason(controls_available=controls_available)
            payload["controls"]["full_access_unavailable_reason"] = reason
            payload["controls"]["can_use_full_access"] = controls_available and reason is None
            if management_available:
                try:
                    payload["management"] = owner.management_payload()
                except FolderGrantsUnavailableError:
                    raise
        can_manage = management_available and owner is not None
        payload["controls"]["can_manage_folders"] = can_manage
        payload["controls"]["can_manage_catalog"] = can_manage
        payload["controls"]["can_discover_repositories"] = can_manage
        return payload

    def scope_payload(
        self,
        scope: WorkspaceScope,
        *,
        permission_reference: WorkspacePermissionReference | None = None,
        permission_retiring: bool = False,
    ) -> WorkspaceScopePayload:
        owner = self.focus_owner
        if owner is not None:
            return owner.scope_payload(
                scope,
                permission_reference=permission_reference,
                permission_retiring=permission_retiring,
            )
        effective_access = cast(EffectiveWorkspaceAccess, {
            "policy": "legacy",
            "source": "agent_home" if scope.project_path == self._default_workspace.resolve(strict=False) else "none",
            "grant_root": None,
            "boundary": (
                "folder" if scope.restrict_to_workspace or self._default_restrict_to_workspace
                else "unrestricted"
            ),
            "exists": scope.project_path.is_dir(),
            "read": scope.project_path.is_dir(),
            "write": scope.project_path.is_dir(),
            "commands": {
                "requested": True,
                "available": True,
                "backend": "native",
                "reason": None,
            },
        })
        return {
            "project_path": str(scope.project_path),
            "project_name": scope.project_name,
            "access_mode": scope.access_mode,
            "effective_access": effective_access,
            "restrict_to_workspace": scope.restrict_to_workspace,
            "sandbox_status": cast(WorkspaceSandboxPayload, scope.sandbox_status.as_dict()),
        }

    def grant_snapshot(self) -> FolderGrantSnapshot:
        if self.grant_store is None:
            raise FolderGrantsUnavailableError("folder grants are unavailable")
        return self.grant_store.snapshot()

    def upsert_grant(self, payload: dict[str, Any]) -> FolderGrantSnapshot:
        return self._upsert_grant_result(payload).snapshot

    def upsert_grant_with_receipt(self, payload: dict[str, Any]) -> WorkspaceGrantReceipt:
        result = self._upsert_grant_result(payload)
        return {
            "submitted_path": result.submitted_path,
            "canonical_path": result.grant.path,
            "file_access": result.grant.file_access,
            "commands": result.grant.commands,
            "grants_revision": result.snapshot.revision,
        }

    def _upsert_grant_result(self, payload: dict[str, Any]) -> FolderGrantUpsertResult:
        if self.grant_store is None:
            raise FolderGrantsUnavailableError("folder grants are unavailable")
        expected_revision = _expected_revision(payload)
        grant_input = FolderGrantInput.model_validate({
            "path": payload.get("path"),
            "file_access": payload.get("file_access"),
            "commands": payload.get("commands"),
        })
        return self.grant_store.upsert_with_result(
            grant_input.model_dump(),
            expected_revision=expected_revision,
        )

    def revoke_grant(self, payload: dict[str, Any]) -> FolderGrantSnapshot:
        if self.grant_store is None:
            raise FolderGrantsUnavailableError("folder grants are unavailable")
        return self.grant_store.revoke(payload.get("path"), expected_revision=_expected_revision(payload))

    def remember_repository(self, payload: dict[str, Any]) -> RepositoryEntry:
        entry = RepositoryEntry.model_validate(payload.get("entry"))
        path = Path(entry.path).expanduser()
        try:
            canonical = path.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("repository path must be an existing Git working tree root") from exc
        if not canonical.is_dir():
            raise ValueError("repository path must be an existing Git working tree root")
        from nanobot.utils.git_sync import git_repo_root

        if git_repo_root(canonical) != canonical:
            raise ValueError("repository path must be the exact root of a Git working tree")
        canonical_entry = RepositoryEntry.model_validate({
            **entry.model_dump(),
            "path": str(canonical),
        })
        self.catalog.update(canonical_entry)
        return canonical_entry

    def forget_repository(self, name: str) -> None:
        entries = self.catalog.entries()
        entry = next((candidate for candidate in entries if candidate.name == name), None)
        if entry is None:
            return
        if any(name in candidate.knowledge_repos for candidate in entries):
            raise RepositoryInUseError("repository_in_use")
        self.catalog.update(None, remove=name)

    def discover_repositories(
        self,
        path: str,
        *,
        trusted_ui: bool = False,
    ) -> RepositoryDiscoveryPayload:
        if self.focus_owner is None:
            raise FolderGrantsUnavailableError("folder grants are unavailable")
        return self.focus_owner.discover(path, trusted_ui=trusted_ui)

    def scope_from_envelope(
        self,
        envelope: dict[str, Any],
        *,
        session_key: str | None,
        controls_available: bool,
    ) -> WorkspaceScope:
        current = self.scope_for_session_key(session_key) if session_key else self.default_scope()
        raw = envelope.get(WORKSPACE_SCOPE_METADATA_KEY)
        if raw is None:
            scope = current
        else:
            scope = validate_workspace_scope_payload(
                raw,
                default_workspace=self._default_workspace,
                default_restrict_to_workspace=self._default_restrict_to_workspace,
                source_channel=_WEBUI_SCOPE_CHANNEL,
            )
            if current.access_mode == "full" and scope.access_mode == "restricted":
                # Folder focus is presentation within an already proven Full chat;
                # only the explicit mode action can fence that proof.
                scope = self._scope_for_mode(scope.project_path, "full")
            if scope.access_mode == "full":
                if not controls_available or session_key is None or self.permission_owner is None:
                    raise WorkspaceScopeError("full_access_unavailable", status=409)
                source = self._source_binding(session_key)
                reference = self.permission_owner.full_reference(
                    session_key=session_key,
                    source_binding=source,
                ) if source is not None else None
                if reference is None:
                    raise WorkspaceScopeError("full_access_unavailable", status=409)
            if self.focus_owner is not None:
                self.focus_owner.validate_scope(scope)
        if scope.access_mode == "full" and not controls_available:
            raise WorkspaceScopeError("full_access_unavailable", status=409)
        if not controls_available and not _scope_change_is_non_escalating(current, scope):
            raise WorkspaceScopeError("workspace controls are localhost-only", status=403)
        return scope

    def ingress_full_reference(self, chat_id: str, *, local_ui: bool) -> object | None:
        """Return private chat authority only after local WebUI authentication."""
        from nanobot.webui.session_identity import is_valid_webui_chat_id

        if not local_ui or not is_valid_webui_chat_id(chat_id):
            return None
        owner = self.permission_owner
        sessions = self._sessions
        if owner is None or sessions is None:
            return None
        key = webui_session_key(chat_id)
        metadata_payload = sessions.read_session_metadata(key)
        if metadata_payload is None:
            return None
        metadata = metadata_payload.get("metadata")
        if type(metadata) is not dict or cast(dict[str, object], metadata).get("webui") is not True:
            return None
        session = sessions.get_cached(key)
        if session is not None and not session.policy.persist:
            return None
        source = self._source_binding(key)
        if source is None:
            return None
        reference = owner.full_reference(session_key=key, source_binding=source)
        if reference is None or not owner.authorizes_full(
            reference,
            session_key=key,
            source_binding=source,
        ):
            return None
        return reference

    def set_permission_mode(
        self,
        raw_payload: object,
        *,
        local_ui: bool,
    ) -> WorkspaceModeActionResult | WorkspaceModeConflict:
        """Apply explicit chat-scoped Full/Restricted authority for a local UI."""
        if not local_ui:
            raise WorkspacePermissionActionError(403, "local_ui_required")
        if type(raw_payload) is not dict:
            raise WorkspacePermissionActionError(400, "invalid_permission_request")
        payload = cast(dict[str, object], raw_payload)
        if set(payload) != {
            "chat_id", "mode", "expected_runtime_id", "expected_permission_revision",
        }:
            raise WorkspacePermissionActionError(400, "invalid_permission_request")
        chat_id = payload.get("chat_id")
        mode = payload.get("mode")
        runtime_id = payload.get("expected_runtime_id")
        expected_revision = payload.get("expected_permission_revision")
        from nanobot.webui.session_identity import is_valid_webui_chat_id

        if (
            not isinstance(chat_id, str)
            or not is_valid_webui_chat_id(chat_id)
            or mode not in {"restricted", "full"}
            or not isinstance(runtime_id, str)
            or len(runtime_id) != 32
            or any(char not in "0123456789abcdef" for char in runtime_id)
            or type(expected_revision) is not int
            or expected_revision < 0
        ):
            raise WorkspacePermissionActionError(400, "invalid_permission_request")
        if self._sessions is None or self.permission_owner is None:
            raise WorkspacePermissionActionError(503, "permission_state_unavailable")

        key = webui_session_key(chat_id)
        metadata_payload = self._sessions.read_session_metadata(key)
        if metadata_payload is None:
            raise WorkspacePermissionActionError(404, "session_not_found")
        metadata = metadata_payload.get("metadata")
        if type(metadata) is not dict or cast(dict[str, object], metadata).get("webui") is not True:
            raise WorkspacePermissionActionError(404, "session_not_found")
        session = self._sessions.get_cached(key)
        if session is not None and not session.policy.persist:
            raise WorkspacePermissionActionError(409, "full_access_unavailable")
        source = self._source_binding(key)
        if source is None:
            raise WorkspacePermissionActionError(503, "permission_state_unavailable")
        current_projection = self._permission_projection(
            key,
            source,
            can_enable_full=True,
            unavailable_reason=None,
        )
        if (
            runtime_id != self.permission_owner.runtime_id
            or expected_revision != current_projection["revision"]
        ):
            raise WorkspacePermissionActionError(409, "permission_changed")

        scope = self.scope_for_session_key(key)
        if (
            mode == "restricted"
            and current_projection["mode"] == "restricted"
            and current_projection["state"] == "active"
            and scope.access_mode == "restricted"
        ):
            return {
                "workspace_scope": self.scope_payload(scope),
                "workspace_permissions": current_projection,
            }
        if mode == "full":
            active_turn = _session_has_active_turn(self._sessions, key, chat_id)
            if active_turn:
                raise WorkspacePermissionActionError(409, "chat_running")
            grant_store = self.grant_store
            if self.focus_owner is None or grant_store is None:
                raise WorkspacePermissionActionError(503, "permission_state_unavailable")
            try:
                grant_store.snapshot()
                source_scope = self._scope_for_mode(scope.project_path, "full")

                def process_counts() -> tuple[int, int]:
                    return _sum_native_counts(
                        grant_store.view.session_native_process_counts(
                            session_key=key,
                            kind="task_once",
                        ),
                        grant_store.view.session_native_process_counts(
                            session_key=key,
                            kind="chat_full",
                        ),
                    )

                self.permission_owner.enable_full(
                    session_key=key,
                    source_binding=source,
                    persistent=session is None or session.policy.persist,
                    existing=True,
                    local=True,
                    idle=True,
                    configured_sandbox=self._configured_exec_sandbox,
                    supervisor_available=self._native_supervisor_available,
                    stores_healthy=True,
                    persist=lambda _epoch: self.persist_scope(
                        chat_id,
                        source_scope,
                        create_if_absent=False,
                    ),
                    process_counts=process_counts,
                )
            except WorkspacePermissionUnavailableError as exc:
                if exc.reason == "permission_retiring":
                    raise WorkspacePermissionActionError(409, "permission_retiring") from None
                raise WorkspacePermissionActionError(409, "full_access_unavailable") from None
            except WorkspacePermissionConflictError as exc:
                raise WorkspacePermissionActionError(409, exc.message) from None
            except FolderGrantsUnavailableError:
                raise WorkspacePermissionActionError(503, "folder_grants_unavailable") from None
        else:
            native_counts: Callable[[], tuple[int, int]] | None = None
            grant_store = self.grant_store
            if grant_store is not None:
                def full_native_counts() -> tuple[int, int]:
                    return grant_store.view.session_native_process_counts(
                        session_key=key,
                        kind="chat_full",
                    )

                native_counts = full_native_counts
            try:
                self.permission_owner.disable_full(
                    session_key=key,
                    source_binding=source,
                    active_turn=_session_has_active_turn(self._sessions, key, chat_id),
                    active_processes=0,
                    unknown_processes=0,
                    process_counts=native_counts,
                )
            except WorkspacePermissionConflictError as exc:
                current_scope = self._scope_for_mode(scope.project_path, "full")
                projection = self.permission_owner.projection(
                    session_key=key,
                    source_binding=source,
                    can_enable_full=False,
                    full_access_unavailable_reason="permission_retiring",
                )
                error_payload: WorkspaceModeConflictErrorPayload = {
                    "status": 409,
                    "message": exc.message,
                }
                conflict: WorkspaceModeConflict = {
                    "error": error_payload,
                    "workspace_scope": self.scope_payload(
                        current_scope,
                        permission_retiring=True,
                    ),
                    "workspace_permissions": projection,
                }
                return conflict
            restricted = self._scope_for_mode(scope.project_path, "restricted")
            try:
                self.persist_scope(chat_id, restricted, create_if_absent=False)
            except Exception:
                # The private reference was already fenced; a stale persisted
                # Full label is clamped on every subsequent read.
                raise WorkspacePermissionActionError(503, "permission_state_unavailable") from None

        new_scope = self.scope_for_session_key(key)
        final_source = self._source_binding(key)
        if final_source is None:
            raise WorkspacePermissionActionError(503, "permission_state_unavailable")
        projection = self._permission_projection(
            key,
            final_source,
            can_enable_full=(
                mode == "restricted"
                and not self._configured_exec_sandbox
                and self._native_supervisor_available
            ),
            unavailable_reason=(
                "configured_sandbox" if self._configured_exec_sandbox
                else "supervisor_unavailable" if not self._native_supervisor_available
                else None
            ),
        )
        full_reference = (
            self.permission_owner.full_reference(
                session_key=key,
                source_binding=final_source,
            )
            if mode == "full"
            else None
        )
        return {
            "workspace_scope": self.scope_payload(
                new_scope,
                permission_reference=full_reference,
            ),
            "workspace_permissions": projection,
        }

    def _source_binding(self, session_key: str) -> PermissionSourceBinding | None:
        if self._sessions is None:
            return None
        try:
            backend = self._sessions.source_backend(session_key)
            if backend is None:
                return None
            snapshot = backend.source.load_source_snapshot(session_key)
        except Exception:
            return None
        return {
            "store_identity": snapshot.state.store_identity,
            "incarnation": snapshot.state.incarnation,
            "generation": snapshot.state.generation,
        }

    def _observe_full_retirement_activity(
        self,
        session_key: str,
        source: PermissionSourceBinding,
    ) -> None:
        """Observe current turn state, conservatively retaining activity across source changes."""
        sessions = self._sessions
        owner = self.permission_owner
        if sessions is None or owner is None:
            return
        from nanobot.webui.session_identity import webui_chat_id

        chat_id = webui_chat_id(session_key)
        if chat_id is None:
            return
        active_turn = True
        if self._source_binding(session_key) == source:
            active_turn = _session_has_active_turn(sessions, session_key, chat_id)
            if self._source_binding(session_key) != source:
                # A changed or unreadable source cannot prove that its turn is idle.
                active_turn = True
        owner.observe_full_retirement_activity(
            session_key=session_key,
            source_binding=source,
            active_turn=active_turn,
        )

    def _scope_for_mode(self, path: Path, mode: str) -> WorkspaceScope:
        return build_workspace_scope(
            path,
            mode,
            source_channel=_WEBUI_SCOPE_CHANNEL,
            preserve_path_identity=bool(self.focus_owner and self.focus_owner.managed),
        )

    def _permission_projection(
        self,
        session_key: str,
        source: PermissionSourceBinding,
        *,
        can_enable_full: bool,
        unavailable_reason: PermissionFullUnavailableReason | None,
    ) -> WorkspacePermissionProjection:
        owner = self.permission_owner
        if owner is None:
            raise WorkspacePermissionActionError(503, "permission_state_unavailable")
        self._observe_full_retirement_activity(session_key, source)
        if self.grant_store is not None:
            try:
                active, unknown = self.grant_store.view.session_native_process_counts(
                    session_key=session_key,
                    kind="chat_full",
                )
            except FolderGrantsUnavailableError:
                raise WorkspacePermissionActionError(503, "permission_state_unavailable") from None
            owner.observe_full_residual_counts(
                session_key=session_key,
                source_binding=source,
                active_processes=active,
                unknown_processes=unknown,
            )
        return owner.projection(
            session_key=session_key,
            source_binding=source,
            can_enable_full=can_enable_full,
            full_access_unavailable_reason=unavailable_reason,
        )

    def scope_for_new_chat(
        self,
        envelope: dict[str, Any],
        *,
        controls_available: bool,
    ) -> WorkspaceScope:
        return self.scope_from_envelope(
            envelope,
            session_key=None,
            controls_available=controls_available,
        )

    def scope_for_set_request(
        self,
        envelope: dict[str, Any],
        *,
        chat_id: str,
        chat_running: bool,
        controls_available: bool,
    ) -> WorkspaceScope:
        if chat_running:
            raise WorkspaceScopeError("chat_running", status=409)
        return self.scope_from_envelope(
            envelope,
            session_key=webui_session_key(chat_id),
            controls_available=controls_available,
        )

    def scope_for_message(
        self,
        envelope: dict[str, Any],
        *,
        chat_id: str,
        chat_running: bool,
        controls_available: bool,
    ) -> WorkspaceScope:
        scope = self.scope_from_envelope(
            envelope,
            session_key=webui_session_key(chat_id),
            controls_available=controls_available,
        )
        if (
            WORKSPACE_SCOPE_METADATA_KEY in envelope
            and chat_running
            and scope.metadata() != self.scope_for_session_key(webui_session_key(chat_id)).metadata()
        ):
            raise WorkspaceScopeError("chat_running", status=409)
        return scope

    def persist_scope(
        self, chat_id: str, scope: WorkspaceScope, *, create_if_absent: bool = True
    ) -> None:
        """Persist after accepted message dispatch; staging may update existing state only."""
        session_key = webui_session_key(chat_id)
        if self._sessions is not None:
            self._sessions.upsert_session_metadata(
                session_key,
                {"webui": True, WORKSPACE_SCOPE_METADATA_KEY: scope.metadata()},
                create_if_absent=create_if_absent,
            )
        self._draft_scopes.pop(session_key, None)

    def stage_scope(self, chat_id: str, scope: WorkspaceScope) -> None:
        """Keep a new chat's scope transient until its first accepted message."""
        session_key = webui_session_key(chat_id)
        if (
            self._sessions is not None
            and self._sessions.read_session_metadata(session_key) is not None
        ):
            self.persist_scope(chat_id, scope, create_if_absent=False)
            return
        self._draft_scopes[session_key] = scope
        self._draft_scopes.move_to_end(session_key)
        while len(self._draft_scopes) > _MAX_DRAFT_SCOPES:
            self._draft_scopes.popitem(last=False)

    def discard_draft_scope(self, session_key: str) -> bool:
        """Discard the staged scope for a chat that has not persisted yet."""
        return self._draft_scopes.pop(session_key, None) is not None


class RepositoryInUseError(ValueError):
    """A catalog entry is still linked by another repository."""


class WorkspacePermissionActionError(Exception):
    """Stable permission-v2 mutation error without private source details."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _session_has_active_turn(sessions: SessionManager, key: str, chat_id: str) -> bool:
    from nanobot.session.pending_followups import parse_pending_followup_records
    from nanobot.session.persistence_metadata import SOURCE_PENDING_FOLLOWUPS_METADATA_KEY
    from nanobot.session.webui_turns import websocket_turn_wall_started_at

    if websocket_turn_wall_started_at(chat_id) is not None:
        return True
    try:
        backend = sessions.source_backend(key)
        if backend is None:
            return True
        snapshot = backend.source.load_source_snapshot(key)
        metadata = snapshot.header["metadata"]
        if metadata.get("execution_lease") is not None:
            return True
        return bool(parse_pending_followup_records(
            metadata.get(SOURCE_PENDING_FOLLOWUPS_METADATA_KEY, []),
        ))
    except Exception:
        return True


def _expected_revision(payload: dict[str, Any]) -> int:
    revision = payload.get("expected_revision")
    if type(revision) is not int or revision < 0:
        raise ValueError("expected_revision must be a nonnegative integer")
    return revision


def _sum_native_counts(
    first: tuple[int, int],
    second: tuple[int, int],
) -> tuple[int, int]:
    return first[0] + second[0], first[1] + second[1]
