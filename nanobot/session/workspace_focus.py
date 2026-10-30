"""One admitted owner for resolving and persisting WebUI workspace focus."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from loguru import logger

from nanobot.context_repo.catalog import RepositoryCatalog
from nanobot.context_repo.discovery import (
    EffectiveWorkspaceAccess,
    RepositoryDiscoveryPayload,
    WorkspaceAccessBoundary,
    discover_repositories,
)
from nanobot.context_repo.policy import FolderAccessNeededError, ResourceAccessPolicy
from nanobot.security.folder_grants import (
    FolderGrant,
    FolderGrantsUnavailableError,
    FolderGrantView,
)
from nanobot.security.workspace_access import (
    WORKSPACE_SCOPE_METADATA_KEY,
    WorkspaceScope,
    WorkspaceScopeError,
    build_workspace_scope,
    current_workspace_focus,
    current_workspace_scope,
)
from nanobot.workspace_access_recovery import (
    WorkspaceSandboxPayload,
    WorkspaceScopePayload,
    parse_folder_access_need,
)

if TYPE_CHECKING:
    from nanobot.agent.turn import ToolOutcome, TurnContext
    from nanobot.bus.queue import MessageBus
    from nanobot.security.desktop_permissions import (
        DesktopPermissionOwner,
        WorkspacePermissionReference,
        WorkspaceTaskAccessReference,
    )
    from nanobot.session.manager import SessionManager

_WEBSOCKET_CHANNEL = "websocket"


class WorkspaceFocusOwner:
    """Resolve path permissions and own same-session focus metadata updates."""

    def __init__(
        self,
        *,
        sessions: SessionManager,
        agent_home: Path,
        grants: FolderGrantView,
        policy: ResourceAccessPolicy,
        catalog: RepositoryCatalog,
        bus: MessageBus | None = None,
    ) -> None:
        self.sessions = sessions
        self.agent_home = agent_home.expanduser().resolve(strict=False)
        self.grants = grants
        self.policy = policy
        self.catalog = catalog
        self.bus = bus

    @property
    def managed(self) -> bool:
        # This owner serves the local WebUI, whose default is always Sandbox.
        return True

    def effective_access(
        self,
        path: str | Path,
        *,
        declared_grant: FolderGrant | None = None,
        retained_root: bool = False,
        permission_reference: WorkspacePermissionReference | None = None,
        permission_retiring: bool = False,
    ) -> EffectiveWorkspaceAccess:
        requested = Path(path).expanduser()
        logical_target = Path(os.path.abspath(requested))
        focus = current_workspace_focus()
        permission_owner = (
            focus.permission_owner if focus is not None else self.sessions.desktop_permissions
        )
        from nanobot.security.desktop_permissions import (
            WorkspaceFullAccessReference,
            WorkspaceTaskAccessReference,
        )

        if permission_reference is None:
            task_reference = focus.task_reference if focus is not None else None
            full_reference = focus.full_reference if focus is not None else None
        else:
            task_reference = (
                permission_reference
                if type(permission_reference) is WorkspaceTaskAccessReference
                else None
            )
            full_reference = (
                permission_reference
                if type(permission_reference) is WorkspaceFullAccessReference
                else None
            )
        task_live = bool(
            permission_owner is not None
            and task_reference is not None
            and permission_owner.is_current_task_reference(task_reference)
        )
        full_live = bool(
            permission_owner is not None
            and full_reference is not None
            and permission_owner.is_current_full_reference(full_reference)
        )
        managed_mode = self.managed or permission_retiring
        if not managed_mode and not task_live and not full_live:
            target = requested.resolve(strict=False)
            exists = target.is_dir()
            return self._legacy_access(target, exists)

        self.grants.snapshot()
        if declared_grant is not None:
            grant_root = Path(os.path.abspath(Path(declared_grant.path).expanduser()))
            try:
                grant_is_live = grant_root.is_dir() and grant_root.resolve(strict=True) == grant_root
            except (OSError, RuntimeError, ValueError):
                grant_is_live = False
            if not grant_is_live:
                return self._managed_none(logical_target, False, reason="path_changed")

        if retained_root:
            try:
                retained_is_live = (
                    logical_target.is_dir()
                    and logical_target.resolve(strict=True) == logical_target
                )
            except (OSError, RuntimeError, ValueError):
                retained_is_live = False
            if not retained_is_live:
                return self._managed_none(logical_target, False, reason="path_changed")

        target = logical_target if retained_root else requested.resolve(strict=False)
        exists = target.is_dir()
        if self.grants.is_private_path(target):
            return self._managed_none(target, exists)

        if full_live:
            from nanobot.agent.tools.native_process import native_supervisor_available

            commands_available = exists and native_supervisor_available()
            return {
                "boundary": "unrestricted",
                "policy": "managed" if managed_mode else "legacy",
                "source": "full",
                "grant_root": None,
                "exists": exists,
                "read": exists,
                "write": exists,
                "commands": {
                    "requested": True,
                    "available": commands_available,
                    "backend": "native" if commands_available else "none",
                    "reason": None if commands_available else "supervisor_unavailable",
                },
            }

        task_root = (
            self._task_root_for(target, task_reference)
            if task_reference is not None and task_live
            else None
        )
        if task_root is not None:
            root_path, file_access, _commands = task_root
            baseline_root = self._baseline_access_root(target, declared_grant)
            if baseline_root is None or len(Path(root_path).parts) >= len(baseline_root.parts):
                from nanobot.agent.tools.native_process import native_supervisor_available
                from nanobot.agent.tools.sandbox import managed_bwrap_available

                command_ready = (
                    exists
                    and managed_bwrap_available()
                    and native_supervisor_available()
                )
                command_reason = (
                    None if command_ready else
                    "supervisor_unavailable" if not native_supervisor_available() else
                    "sandbox_unavailable"
                )
                return {
                    "boundary": "folder",
                    "policy": "managed",
                    "source": "task",
                    "grant_root": root_path,
                    "exists": exists,
                    "read": exists,
                    "write": exists and file_access == "read_write",
                    "commands": {
                        "requested": True,
                        "available": command_ready,
                        "backend": "bwrap" if command_ready else "none",
                        "reason": command_reason,
                    },
                }

        grant = declared_grant
        if grant is not None:
            grant_root = Path(grant.path)
            try:
                grant_is_live = (
                    grant_root.is_dir()
                    and grant_root.resolve(strict=True) == grant_root
                    and _is_under(target, grant_root)
                )
            except (OSError, RuntimeError, ValueError):
                grant_is_live = False
            if not grant_is_live:
                grant = None
        if grant is None:
            grant = self.grants.effective_grant_for(target)

        if _is_under(target, self.agent_home):
            # A grant at least as specific as the workspace overrides its
            # built-in write rule; an ancestor grant does not. This matches
            # ResourceAccessPolicy's longest-root precedence.
            if grant is None or len(Path(grant.path).parts) < len(self.agent_home.parts):
                return self._managed_access(
                    target,
                    exists=exists,
                    source="agent_home",
                    grant_root=None,
                    read=True,
                    write=True,
                    commands_requested=True,
                )

        configured_read = any(_is_under(target, root.resolve(strict=False))
                              for root in self.policy.context_manager.read_roots())
        if configured_read:
            configured_write = any(_is_under(target, root.resolve(strict=False))
                                   for root in self.policy.context_manager.write_roots())
            return self._managed_access(
                target,
                exists=exists,
                source="configured",
                grant_root=None,
                read=True,
                write=configured_write,
                commands_requested=True,
            )

        if grant is None:
            return self._managed_access(target, exists=exists, source="none", grant_root=None,
                                        read=True, write=False, commands_requested=True)
        if not exists:
            return self._managed_access(
                target,
                exists=False,
                source="grant",
                grant_root=grant.path,
                read=False,
                write=False,
                commands_requested=False,
                fixed_reason="path_missing",
            )
        return self._managed_access(
            target,
            exists=True,
            source="grant",
            grant_root=grant.path,
            read=True,
            write=grant.file_access == "read_write",
            commands_requested=True,
        )

    def _task_root_for(
        self,
        target: Path,
        reference: WorkspaceTaskAccessReference,
    ) -> tuple[str, str, bool] | None:
        focus = current_workspace_focus()
        owner = (
            focus.permission_owner if focus is not None else self.sessions.desktop_permissions
        )
        from nanobot.security.desktop_permissions import WorkspaceTaskAccessReference

        if type(reference) is not WorkspaceTaskAccessReference or owner is None:
            return None
        matches = [
            row for row in owner.once_roots(reference)
            if _is_under(target, Path(row[0]))
        ]
        return max(matches, key=lambda row: len(Path(row[0]).parts), default=None)

    def _baseline_access_root(
        self,
        target: Path,
        declared_grant: FolderGrant | None,
    ) -> Path | None:
        if _is_under(target, self.agent_home):
            return self.agent_home
        configured_roots = [
            root.expanduser().resolve(strict=False)
            for root in self.policy.context_manager.read_roots()
            if _is_under(target, root.expanduser().resolve(strict=False))
        ]
        grant = declared_grant or self.grants.effective_grant_for(target)
        if grant is not None:
            root = Path(grant.path)
            if _is_under(target, root):
                configured_roots.append(root)
        return max(configured_roots, key=lambda root: len(root.parts), default=None)

    def scope_payload(
        self,
        scope: WorkspaceScope,
        *,
        permission_reference: WorkspacePermissionReference | None = None,
        permission_retiring: bool = False,
    ) -> WorkspaceScopePayload:
        access = self.effective_access(
            scope.project_path,
            retained_root=True,
            permission_reference=permission_reference,
            permission_retiring=permission_retiring,
        )
        access["boundary"] = (
            "folder" if permission_retiring else
            "unrestricted" if access["source"] == "full" else
            self._scope_boundary(scope)
        )
        return {
            "project_path": str(scope.project_path),
            "project_name": scope.project_name,
            "access_mode": scope.access_mode,
            "effective_access": access,
            "restrict_to_workspace": scope.restrict_to_workspace,
            "sandbox_status": cast(WorkspaceSandboxPayload, scope.sandbox_status.as_dict()),
        }

    def _scope_boundary(self, scope: WorkspaceScope | None = None) -> WorkspaceAccessBoundary:
        current = scope or current_workspace_scope()
        if self.managed or self.policy.restrict_to_workspace or (current and current.restrict_to_workspace):
            return "folder"
        return "unrestricted"

    def validate_scope(self, scope: WorkspaceScope) -> None:
        if not self.managed:
            return
        try:
            self.policy.validate_path(scope.project_path, "read")
        except PermissionError as exc:
            raise WorkspaceScopeError("folder_access_required", status=403) from exc

    async def request_access_from_tool(
        self,
        context: TurnContext,
        *,
        purpose: str,
        path: str | None,
    ) -> str | ToolOutcome:
        """Create a non-mutating need only for this live foreground source owner."""
        from nanobot.agent.tools.base import ToolResult
        from nanobot.agent.tools.context import current_request_context
        from nanobot.agent.tools.workspace import FOLDER_ACCESS_SAFE_FINAL
        from nanobot.agent.turn import HistoryMode, ToolOutcome, TurnSource
        from nanobot.webui.session_identity import is_webui_session_key

        request = current_request_context()
        session_key = context.session_key
        turn_request = context.request
        writer = context.source_writer
        owner = context.execution_owner
        if (
            request is None
            or request.channel != _WEBSOCKET_CHANNEL
            or not request.persistent_session
            or session_key is None
            or not is_webui_session_key(session_key)
            or request.session_key != session_key
            or not request.chat_id
            or turn_request is None
            or turn_request.source is not TurnSource.GATEWAY
            or turn_request.history_mode is not HistoryMode.SESSION
            or turn_request.session_key != session_key
            or writer is None
            or owner is None
            or writer.owner is not owner
        ):
            return ToolResult.error("Folder selection requires an owned persistent WebSocket chat")
        if self.sessions.read_session_metadata(session_key) is None:
            return ToolResult.error("Folder selection requires an existing chat session")
        try:
            owner.validate_dispatch()
            snapshot = owner.read_snapshot()
        except Exception:
            return ToolResult.error("Folder selection lost its active execution owner")
        input_id = context.source_input_event_id
        if (
            not isinstance(input_id, str)
            or not any(
                event.get("_event_id") == input_id and event.get("role") == "user"
                for event in snapshot.events
            )
        ):
            return ToolResult.error("Folder selection requires an owned source task")

        target: dict[str, object] = {"kind": "unknown", "path": None}
        if path is not None:
            try:
                candidate = Path(path).expanduser()
                if not candidate.is_absolute():
                    return ToolResult.error("Folder selection path must be absolute")
                candidate = candidate.resolve(strict=False)
                private_roots = (
                    self.grants.sandbox_private_roots
                    + (self.grants.sandbox_instance_root,)
                )
                broad = (
                    candidate == Path(candidate.anchor)
                    or candidate == Path.home().expanduser().resolve(strict=False)
                    or candidate == self.agent_home
                    or any(
                        root.expanduser().resolve(strict=False).is_relative_to(candidate)
                        for root in private_roots
                    )
                )
                if not broad:
                    target = {"kind": "directory", "path": str(candidate)}
            except (OSError, RuntimeError, ValueError):
                return ToolResult.error("Folder selection path is invalid")

        roots, truncated = self.policy.readable_roots_summary()
        need = parse_folder_access_need({
            "version": 1,
            "reason": "folder_selection_required",
            "operation": "locate",
            "target": target,
            "purpose": " ".join(purpose.split())[:240],
            "allowed_roots": roots,
            "roots_truncated": truncated,
        })
        return ToolOutcome(
            content=FOLDER_ACCESS_SAFE_FINAL,
            stop_reason="folder_access_required",
            policy_metadata={"folder_access": need},
        )

    def resolve_scope(
        self,
        *,
        current: WorkspaceScope,
        action: str,
        path: str = "",
        repository: str = "",
    ) -> WorkspaceScope:
        if action == "clear":
            if path or repository:
                raise WorkspaceScopeError("clear accepts no path or repository")
            target = self.agent_home
        elif action == "select":
            if bool(path.strip()) == bool(repository.strip()):
                raise WorkspaceScopeError("select requires exactly one path or repository")
            if repository.strip():
                try:
                    target = Path(self.catalog.get(repository.strip()).path)
                except (OSError, ValueError) as exc:
                    raise WorkspaceScopeError("repository not found in the catalog") from exc
            else:
                target = Path(path).expanduser()
        else:
            raise WorkspaceScopeError("action must be select or clear")

        if not target.is_absolute():
            raise WorkspaceScopeError("workspace path must be absolute")
        if repository.strip():
            # A saved shortcut keeps its canonical identity. The management view
            # already treats a replacement symlink (on the path or an ancestor)
            # as changed; selecting it must not follow the replacement.
            try:
                if not target.is_dir():
                    raise WorkspaceScopeError("workspace path must be an existing directory")
                if target.resolve(strict=True) != target:
                    raise WorkspaceScopeError("saved folder path changed; select it again")
            except WorkspaceScopeError:
                raise
            except (OSError, RuntimeError, ValueError) as exc:
                raise WorkspaceScopeError("workspace path must be an existing directory") from exc
        try:
            canonical = target.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkspaceScopeError("workspace path must be an existing directory") from exc
        if not canonical.is_dir():
            raise WorkspaceScopeError("workspace path must be an existing directory")
        if self.grants.is_private_path(canonical) and canonical != self.agent_home:
            raise WorkspaceScopeError("folder_access_required", status=403)
        scope = build_workspace_scope(
            canonical,
            current.access_mode,
            source_channel=_WEBSOCKET_CHANNEL,
        )
        self.validate_scope(scope)
        return scope

    async def select_from_tool(
        self,
        context: TurnContext,
        *,
        action: str,
        path: str = "",
        repository: str = "",
    ) -> str:
        """Persist an agent-requested focus only for its active WebSocket run."""
        from nanobot.agent.tools.base import ToolResult
        from nanobot.agent.tools.context import current_request_context
        from nanobot.agent.turn import TurnSource
        from nanobot.bus.outbound_events import SessionUpdatedEvent
        from nanobot.webui.session_identity import is_webui_session_key

        request = current_request_context()
        session_key = context.session_key
        turn_request = context.request
        if (
            request is None
            or request.channel != _WEBSOCKET_CHANNEL
            or not request.persistent_session
            or session_key is None
            or not is_webui_session_key(session_key)
            or request.session_key is None
            or request.session_key != session_key
            or not request.chat_id
            or turn_request is None
            or turn_request.source is not TurnSource.GATEWAY
            or turn_request.session_key != session_key
        ):
            return ToolResult.error("Workspace selection requires an owned persistent WebSocket chat")
        if self.sessions.read_session_metadata(session_key) is None:
            return ToolResult.error("Workspace selection requires an existing chat session")

        owner = context.execution_owner
        if owner is None:
            return ToolResult.error("Workspace selection requires an active execution owner")
        try:
            owner.validate_dispatch()
        except Exception:
            return ToolResult.error("Workspace selection lost its active execution owner")

        current = current_workspace_scope()
        if current is None:
            return ToolResult.error("Workspace selection has no active run focus")
        try:
            scope = self.resolve_scope(
                current=current,
                action=action,
                path=path,
                repository=repository,
            )
            commit = asyncio.create_task(asyncio.to_thread(
                self.sessions.upsert_session_metadata,
                session_key,
                {"webui": True, WORKSPACE_SCOPE_METADATA_KEY: scope.metadata()},
                create_if_absent=False,
                fsync=True,
            ))
            try:
                updated = await asyncio.shield(commit)
            except asyncio.CancelledError:
                updated = await asyncio.shield(commit)
                if updated:
                    focus = current_workspace_focus()
                    if focus is not None:
                        focus.replace(scope)
                raise
            if not updated:
                return ToolResult.error("Workspace selection owner was deleted before commit")
        except (OSError, RuntimeError, ValueError, FolderGrantsUnavailableError) as exc:
            return ToolResult.error(f"Workspace selection failed: {exc}")

        focus = current_workspace_focus()
        if focus is not None:
            focus.replace(scope)
        if self.bus is not None:
            try:
                await self.bus.publish_event(
                    SessionUpdatedEvent(
                        scope="metadata",
                        workspace_scope=self.scope_payload(scope),
                    ),
                    channel=_WEBSOCKET_CHANNEL,
                    chat_id=request.chat_id,
                )
            except Exception:
                logger.exception("Workspace selection committed but session update delivery failed")
        return json.dumps(
            {"workspace_scope": self.scope_payload(scope)},
            ensure_ascii=False,
        )

    def management_payload(self) -> dict[str, object]:
        snapshot = self.grants.snapshot()
        grant_rows: list[dict[str, object]] = []
        for grant in snapshot.grants:
            access = self.effective_access(grant.path, declared_grant=grant)
            grant_rows.append({
                "path": grant.path,
                "file_access": grant.file_access,
                "commands": grant.commands,
                "exists": access["exists"],
                "effective_access": access,
            })
        repositories: list[dict[str, object]] = []
        for entry in self.catalog.entries():
            path = Path(entry.path)
            access = self.effective_access(path, retained_root=True)
            repositories.append({
                **entry.model_dump(),
                "exists": access["exists"],
                "effective_access": access,
                "git": access["exists"] and (path / ".git").exists(),
            })
        return {
            "catalog_revision": self.catalog.snapshot().revision,
            "grants_revision": snapshot.revision,
            "grants": grant_rows,
            "repositories": repositories,
        }

    def repository_roots(self, *, limit: int = 64) -> dict[str, object]:
        snapshot = self.grants.snapshot()
        rows_by_path: dict[str, dict[str, object]] = {}
        for grant in snapshot.grants:
            access = self.effective_access(grant.path, declared_grant=grant)
            if not access["exists"] or not access["read"]:
                continue
            rows_by_path[grant.path] = {
                "path": grant.path,
                "file_access": grant.file_access,
                "commands": grant.commands,
                "exists": access["exists"],
                "effective_access": access,
            }

        task_owner, task_reference = self._current_local_task_reference()
        if task_owner is not None and task_reference is not None:
            for path, file_access, commands in task_owner.once_roots(task_reference):
                access = self.effective_access(path, permission_reference=task_reference)
                if not access["exists"] or not access["read"]:
                    continue
                rows_by_path[path] = {
                    "path": path,
                    "file_access": file_access,
                    "commands": commands,
                    "exists": access["exists"],
                    "effective_access": access,
                }
        rows = list(rows_by_path.values())[:limit]
        return {"agent_home": str(self.agent_home), "roots": rows}

    def _current_local_task_reference(
        self,
    ) -> tuple[DesktopPermissionOwner | None, WorkspaceTaskAccessReference | None]:
        """Return an active Once proof only for its current canonical local chat."""
        from nanobot.agent.tools.context import current_request_context
        from nanobot.security.desktop_permissions import WorkspaceTaskAccessReference
        from nanobot.webui.session_identity import webui_chat_id

        focus = current_workspace_focus()
        request = current_request_context()
        if (
            focus is None
            or focus.task_reference is None
            or request is None
            or request.channel != _WEBSOCKET_CHANNEL
            or not request.persistent_session
            or not request.chat_id
            or request.session_key is None
        ):
            return None, None
        reference = focus.task_reference
        permission_owner = focus.permission_owner
        if (
            type(reference) is not WorkspaceTaskAccessReference
            or request.session_key != reference.session_key
            or permission_owner is None
            or permission_owner is not self.sessions.desktop_permissions
            or request.chat_id != webui_chat_id(reference.session_key)
            or not permission_owner.is_current_task_reference(reference)
        ):
            return None, None
        backend = self.sessions.source_backend(reference.session_key)
        if backend is None:
            return None, None
        try:
            source = backend.source.load_source_snapshot(reference.session_key).state
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            return None, None
        current_binding = {
            "store_identity": source.store_identity,
            "incarnation": source.incarnation,
            "generation": source.generation,
        }
        if (
            source.session_key != reference.session_key
            or current_binding != reference.source_binding
        ):
            return None, None
        return permission_owner, reference

    def discover(
        self,
        path: str,
        *,
        require_access: bool = True,
        trusted_ui: bool = False,
    ) -> RepositoryDiscoveryPayload:
        if not path.strip():
            raise ValueError("discovery path is required")
        root = Path(path).expanduser()
        if not root.is_absolute():
            raise ValueError("discovery path must be absolute")
        managed = self.managed if trusted_ui else self.policy.managed_mode
        allow_folder_grants = True if trusted_ui else self.policy.folder_grants_allowed
        if require_access:
            try:
                self.policy.validate_path(
                    root,
                    "read",
                    managed=managed,
                    allow_folder_grants=allow_folder_grants,
                    target_kind="directory",
                )
            except FolderAccessNeededError:
                raise
            except PermissionError as exc:
                raise PermissionError("folder_access_required") from exc
        private_roots = tuple(
            root
            for root in self.grants.sandbox_private_roots
            if not _is_under(self.agent_home, root)
        )
        return discover_repositories(
            root,
            policy=self.policy,
            catalog=self.catalog,
            access_for_path=self.effective_access,
            extra_private_roots=private_roots,
            managed=managed,
            allow_folder_grants=allow_folder_grants,
        )

    def _legacy_access(self, target: Path, exists: bool) -> EffectiveWorkspaceAccess:
        read = self._allowed(target, "read")
        write = self._allowed(target, "write")
        commands = self._allowed(target, "exec")
        source = self._source(target)
        return {
            "boundary": self._scope_boundary(),
            "policy": "legacy",
            "source": source,
            "grant_root": None,
            "exists": exists,
            "read": exists and read,
            "write": exists and write,
            "commands": {
                "requested": True,
                "available": exists and commands,
                "backend": "native" if exists and commands else "none",
                "reason": None if exists and commands else "not_granted",
            },
        }

    def _managed_none(
        self,
        target: Path,
        exists: bool,
        *,
        reason: str | None = None,
    ) -> EffectiveWorkspaceAccess:
        _ = target
        return {
            "boundary": "folder",
            "policy": "managed",
            "source": "none",
            "grant_root": None,
            "exists": exists,
            "read": False,
            "write": False,
            "commands": {
                "requested": False,
                "available": False,
                "backend": "none",
                "reason": reason or ("path_missing" if not exists else "not_granted"),
            },
        }

    def _managed_access(
        self,
        target: Path,
        *,
        exists: bool,
        source: Literal["agent_home", "grant", "configured", "task", "none"],
        grant_root: str | None,
        read: bool,
        write: bool,
        commands_requested: bool,
        fixed_reason: str | None = None,
    ) -> EffectiveWorkspaceAccess:
        if exists:
            read = read and self._managed_path_allowed(target, "read")
            write = write and self._managed_path_allowed(target, "write")
        commands_available = False
        backend = "none"
        reason = fixed_reason
        if not exists and reason is None:
            reason = "path_missing"
        elif self.policy.context_manager.find_repo_for_path(target) is not None:
            reason = "managed_repository_policy"
        elif self.policy.context_manager.find_target_repo_for_path(target) is not None:
            reason = "managed_repository_policy"
        elif not commands_requested and reason is None:
            reason = "not_granted"
        elif commands_requested:
            from nanobot.agent.tools.sandbox import managed_bwrap_available

            if not managed_bwrap_available():
                reason = "unsupported_platform" if not _linux_host() else "sandbox_unavailable"
            else:
                commands_available = True
                backend = "bwrap"
                reason = None
        return {
            "boundary": "folder",
            "policy": "managed",
            "source": source,
            "grant_root": grant_root,
            "exists": exists,
            "read": exists and read,
            "write": exists and write,
            "commands": {
                "requested": commands_requested,
                "available": commands_available,
                "backend": backend,
                "reason": reason,
            },
        }

    def _managed_path_allowed(self, target: Path, action: str) -> bool:
        try:
            self.policy.validate_path(
                target,
                action,
                managed=True,
                allow_folder_grants=True,
                target_kind="directory",
            )
        except PermissionError:
            return False
        return True

    def _allowed(self, path: Path, action: str) -> bool:
        try:
            self.policy.validate_path(path, action)
            return True
        except PermissionError:
            return False

    def _source(self, path: Path) -> Literal["agent_home", "configured", "none"]:
        if _is_under(path, self.agent_home):
            return "agent_home"
        if any(_is_under(path, root.resolve(strict=False)) for root in self.policy.context_manager.read_roots()):
            return "configured"
        return "none"


def _linux_host() -> bool:
    import sys

    return sys.platform.startswith("linux")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False
