"""Resource access policy for workspace and managed context repositories."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, cast

from nanobot.context_repo.manager import ContextRepoManager, ManagedContextRepo, ManagedTargetRepo
from nanobot.security.folder_grants import (
    FolderFileAccess,
    FolderGrantsUnavailableError,
    FolderGrantView,
)
from nanobot.security.workspace_access import current_tool_workspace, current_workspace_scope
from nanobot.workspace_access_recovery import (
    FolderAccessOperation,
    FolderAccessReason,
    FolderAccessTarget,
)

if TYPE_CHECKING:
    from nanobot.security.desktop_permissions import (
        DesktopPermissionOwner,
        WorkspaceFullAccessReference,
        WorkspacePermissionReference,
        WorkspaceTaskAccessReference,
    )

_PATHISH_CHARS = {"/", "\\"}
_FolderTargetKind = Literal["directory", "file"]


class FolderAccessNeededError(PermissionError):
    """A narrow, typed missing-folder-capability decision safe for recovery."""

    reason: FolderAccessReason
    operation: FolderAccessOperation
    target: FolderAccessTarget

    def __init__(
        self,
        *,
        reason: FolderAccessReason,
        operation: FolderAccessOperation,
        target: FolderAccessTarget,
    ) -> None:
        super().__init__("folder access is required")
        self.reason = reason
        self.operation = operation
        self.target = cast(FolderAccessTarget, dict(target))


class ResourceAccessPolicy:
    """Validate file and shell access against workspace/context repo boundaries."""

    def __init__(
        self,
        *,
        workspace: Path,
        context_manager: ContextRepoManager | None = None,
        restrict_to_workspace: bool = False,
        folder_grants: FolderGrantView | None = None,
        managed_webui_only: bool = False,
        sandbox_default: bool = False,
    ):
        self.workspace = workspace.expanduser().resolve(strict=False)
        self.context_manager = context_manager or ContextRepoManager()
        self.restrict_to_workspace = restrict_to_workspace
        self.folder_grants = folder_grants
        self.managed_webui_only = managed_webui_only
        # The local WebUI starts in Sandbox, so its write boundary must hold
        # before any folder has been saved. Activation of the grant store is a
        # storage detail, not the mode switch.
        self.sandbox_default = sandbox_default and folder_grants is not None
        self.touched_paths: set[Path] = set()

    @property
    def managed_mode(self) -> bool:
        if self.folder_grants is None:
            return self._current_task_reference() is not None
        if self.sandbox_default:
            return True
        if self.managed_webui_only:
            from nanobot.agent.tools.context import current_request_context

            request = current_request_context()
            if request is not None and request.channel == "websocket":
                return True
            return self._current_task_reference() is not None
        return self.folder_grants.is_managed or self._current_task_reference() is not None

    @staticmethod
    def _current_task_reference() -> WorkspaceTaskAccessReference | None:
        from nanobot.security.workspace_access import current_workspace_focus

        focus = current_workspace_focus()
        if (
            focus is not None
            and focus.permission_owner is not None
            and focus.task_reference is not None
            and focus.permission_owner.is_current_task_reference(focus.task_reference)
        ):
            return focus.task_reference
        return None

    @staticmethod
    def _current_full_reference(
        permission_owner: DesktopPermissionOwner | None = None,
        permission_reference: WorkspacePermissionReference | None = None,
    ) -> tuple[DesktopPermissionOwner, WorkspaceFullAccessReference] | None:
        from nanobot.security.desktop_permissions import WorkspaceFullAccessReference

        if (
            type(permission_reference) is WorkspaceFullAccessReference
            and permission_owner is not None
            and permission_owner.is_current_full_reference(permission_reference)
        ):
            return permission_owner, permission_reference
        from nanobot.security.workspace_access import current_workspace_focus

        focus = current_workspace_focus()
        if (
            focus is not None
            and focus.permission_owner is not None
            and focus.full_reference is not None
            and focus.permission_owner.is_current_full_reference(focus.full_reference)
        ):
            return focus.permission_owner, focus.full_reference
        return None

    @property
    def folder_grants_allowed(self) -> bool:
        """Whether the active caller may use external UI folder grants."""
        if not self.managed_webui_only:
            return True
        from nanobot.agent.tools.context import current_request_context

        request = current_request_context()
        return bool(
            request
            and request.channel == "websocket"
            and request.persistent_session
        )

    def extra_read_dirs(self) -> list[Path]:
        return self.context_manager.read_roots()

    def extra_write_dirs(self) -> list[Path]:
        return self.context_manager.write_roots()

    def readable_roots_summary(self, *, limit: int = 8) -> tuple[list[str], bool]:
        """Return a small privacy-filtered summary, never a capability grant."""
        candidates: list[Path] = [self.workspace]
        candidates.extend(self.context_manager.read_roots())
        if self.folder_grants is not None and self.folder_grants.is_managed:
            try:
                candidates.extend(Path(grant.path) for grant in self.folder_grants.snapshot().grants)
            except FolderGrantsUnavailableError:
                pass
        result: list[str] = []
        for candidate in candidates:
            try:
                canonical = candidate.expanduser().resolve(strict=True)
                if not canonical.is_dir() or canonical in map(Path, result):
                    continue
                if self.folder_grants is not None and self.folder_grants.is_private_path(canonical):
                    continue
                self.validate_path(canonical, "read")
            except (OSError, RuntimeError, ValueError, PermissionError, FolderGrantsUnavailableError):
                continue
            result.append(str(canonical))
        return result[:limit], len(result) > limit

    def validate_path(
        self,
        path: Path,
        action: str,
        *,
        extra_read_roots: Iterable[Path] = (),
        extra_read_files: Iterable[Path] = (),
        managed: bool | None = None,
        allow_folder_grants: bool | None = None,
        target_kind: _FolderTargetKind | None = None,
        permission_owner: DesktopPermissionOwner | None = None,
        permission_reference: WorkspacePermissionReference | None = None,
    ) -> None:
        resolved = path.expanduser().resolve(strict=False)
        if action == "exec":
            self._deny_context_repo_exec(resolved)
        use_managed_policy = self.managed_mode if managed is None else managed
        if self._explicit_reference_is_live(permission_owner, permission_reference):
            use_managed_policy = True
        self.validate_current_workspace_identity(managed=use_managed_policy)
        if use_managed_policy:
            # A host process's environ/memory/fd tree can expose the gateway's
            # provider credentials and live authority despite filesystem masks.
            # Sandbox commands see their own PID namespace instead.
            if self._current_full_reference(permission_owner, permission_reference) is None:
                proc_parts = resolved.parts
                if len(proc_parts) >= 3 and proc_parts[1] == "proc" and proc_parts[2].isdigit():
                    raise PermissionError("Host process state is private to the Sandbox runtime; use confined commands for process inspection")
            self._validate_managed_private_path(resolved, action)
        if (
            not use_managed_policy
            and self.folder_grants is not None
            and self.folder_grants.is_private_path(resolved)
        ):
            raise PermissionError("Path is inside private nanobot runtime storage")
        if not use_managed_policy and self.folder_grants is None:
            try:
                from nanobot.config.paths import get_config_path

                instance_root = get_config_path().expanduser().parent.resolve(strict=False)
            except (OSError, RuntimeError, ValueError):
                instance_root = None
            if instance_root is not None and _is_under(resolved, instance_root) and not _is_under(
                resolved, self.workspace
            ):
                media_root = instance_root / "media"
                media_is_explicitly_readable = (
                    action in {"read", "list"}
                    and _is_under(resolved, media_root)
                    and any(
                        _is_under(resolved, root.expanduser().resolve(strict=False))
                        for root in extra_read_roots
                    )
                )
                if not media_is_explicitly_readable:
                    raise PermissionError("Path is inside private nanobot runtime storage")
        can_use_folder_grants = (
            self.folder_grants_allowed
            if allow_folder_grants is None
            else allow_folder_grants
        )
        target = self.context_manager.find_target_repo_for_path(resolved)
        repo = self.context_manager.find_repo_for_path(resolved)
        if target and (repo is None or len(str(target.path or "")) >= len(str(repo.path))):
            self._validate_target_path(target, resolved, action)
            return
        if repo:
            self._validate_context_path(repo, resolved, action)
            return

        if self._current_full_reference(permission_owner, permission_reference) is not None:
            view = self.folder_grants
            if view is not None:
                try:
                    view.snapshot()
                except FolderGrantsUnavailableError as exc:
                    raise PermissionError("Folder grants are unavailable") from exc
            return

        if use_managed_policy:
            self._validate_managed_path(
                resolved,
                action,
                extra_read_roots=extra_read_roots,
                extra_read_files=extra_read_files,
                allow_folder_grants=can_use_folder_grants,
                target_kind=target_kind,
                permission_owner=permission_owner,
                permission_reference=permission_reference,
            )

        if (
            not use_managed_policy
            and self.restrict_to_workspace
            and not _is_under(resolved, self.workspace)
        ):
            reason = "write_not_granted" if action in {"write", "edit"} else (
                "commands_not_granted" if action == "exec" else "folder_not_granted"
            )
            raise self._folder_access_needed(resolved, action, reason, target_kind=target_kind)

    def _validate_managed_path(
        self,
        path: Path,
        action: str,
        *,
        extra_read_roots: Iterable[Path],
        extra_read_files: Iterable[Path],
        allow_folder_grants: bool,
        target_kind: _FolderTargetKind | None,
        permission_owner: DesktopPermissionOwner | None = None,
        permission_reference: WorkspacePermissionReference | None = None,
    ) -> None:
        view = self.folder_grants
        if view is None:
            raise PermissionError("Managed folder access is unavailable")

        media_root: Path | None = None
        try:
            from nanobot.config.paths import get_media_dir

            media_root = get_media_dir().expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            pass
        if view.is_private_path(path):
            if not (
                action in {"read", "list"}
                and media_root is not None
                and _is_under(path, media_root)
            ):
                raise PermissionError("Path is inside private nanobot runtime storage")

        # Sandbox is a write boundary. Reads and program launches use normal
        # OS permissions; subprocess writes are enforced by the mount namespace.
        # Keep authority health checks even for reads so damaged state fails closed.
        try:
            view.snapshot()
            grant = view.effective_grant_for(path) if allow_folder_grants else None
        except FolderGrantsUnavailableError as exc:
            raise PermissionError("Folder grants are unavailable") from exc
        if action in {"read", "list", "exec"}:
            return
        if action not in {"write", "edit"}:
            raise PermissionError(f"Folder access does not permit {action} for {path}")

        choices: list[tuple[Path, bool, int]] = []
        if _is_under(path, self.workspace):
            choices.append((self.workspace, True, 0))
        for root in self.context_manager.write_roots():
            canonical = root.expanduser().resolve(strict=False)
            if _is_under(path, canonical):
                choices.append((canonical, True, 0))
        if grant is not None:
            choices.append((Path(grant.path), grant.file_access == "read_write", 1))
        task_choice = self._task_grant_for(
            path,
            permission_owner=permission_owner,
            permission_reference=permission_reference,
        )
        if task_choice is not None:
            task_root, file_access, _commands = task_choice
            choices.append((task_root, file_access == "read_write", 2))
        selected = max(choices, key=lambda row: (len(row[0].parts), row[2]), default=None)
        if selected is not None and selected[1]:
            return
        raise self._folder_access_needed(path, action, "write_not_granted", target_kind=target_kind)

    def _task_grant_for(
        self,
        path: Path,
        *,
        permission_owner: DesktopPermissionOwner | None = None,
        permission_reference: WorkspacePermissionReference | None = None,
    ) -> tuple[Path, FolderFileAccess, bool] | None:
        from nanobot.security.workspace_access import current_workspace_focus

        focus = current_workspace_focus()
        owner = focus.permission_owner if focus is not None else permission_owner
        reference = focus.task_reference if focus is not None else permission_reference
        from nanobot.security.desktop_permissions import WorkspaceTaskAccessReference

        if type(permission_reference) is WorkspaceTaskAccessReference:
            owner = permission_owner
            reference = permission_reference
        if (
            owner is None
            or type(reference) is not WorkspaceTaskAccessReference
            or not owner.is_current_task_reference(reference)
        ):
            return None
        task_reference = reference
        roots: tuple[tuple[str, FolderFileAccess, bool], ...] = owner.once_roots(
            task_reference,
        )
        matches: list[tuple[Path, FolderFileAccess, bool]] = [
            (Path(root), file_access, commands)
            for root, file_access, commands in roots
            if _is_under(path, Path(root))
        ]
        return max(matches, key=lambda row: len(row[0].parts), default=None)

    @staticmethod
    def _explicit_reference_is_live(
        permission_owner: DesktopPermissionOwner | None,
        permission_reference: WorkspacePermissionReference | None,
    ) -> bool:
        if permission_owner is None or permission_reference is None:
            return False
        from nanobot.security.desktop_permissions import (
            WorkspaceFullAccessReference,
            WorkspaceTaskAccessReference,
        )

        if type(permission_reference) is WorkspaceTaskAccessReference:
            return permission_owner.is_current_task_reference(permission_reference)
        if type(permission_reference) is WorkspaceFullAccessReference:
            return permission_owner.is_current_full_reference(permission_reference)
        return False

    def _baseline_root_for(self, path: Path, view: FolderGrantView) -> Path | None:
        roots: list[Path] = []
        if _is_under(path, self.workspace):
            roots.append(self.workspace)
        for candidate in (*self.context_manager.read_roots(), *self.context_manager.write_roots()):
            root = candidate.expanduser().resolve(strict=False)
            if _is_under(path, root):
                roots.append(root)
        try:
            grant = view.effective_grant_for(path)
        except FolderGrantsUnavailableError as exc:
            raise PermissionError("Folder grants are unavailable") from exc
        if grant is not None:
            roots.append(Path(grant.path))
        return max(roots, key=lambda root: len(root.parts), default=None)

    def _validate_managed_private_path(self, path: Path, action: str) -> None:
        """Keep configured repositories from widening private runtime access."""
        view = self.folder_grants
        if view is None or not view.is_private_path(path):
            return

        media_root: Path | None = None
        try:
            from nanobot.config.paths import get_media_dir

            media_root = get_media_dir().expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            pass
        if not (
            action in {"read", "list"}
            and media_root is not None
            and _is_under(path, media_root)
        ):
            raise PermissionError("Path is inside private nanobot runtime storage")

    @staticmethod
    def _folder_access_needed(
        path: Path,
        action: str,
        reason: FolderAccessReason,
        *,
        target_kind: _FolderTargetKind | None = None,
    ) -> FolderAccessNeededError:
        operations: dict[str, FolderAccessOperation] = {
            "read": "read", "list": "read", "write": "write", "edit": "write", "exec": "execute",
        }
        operation = operations.get(action)
        if operation is None:
            raise PermissionError("resource access denied")
        target: FolderAccessTarget = (
            {"kind": target_kind, "path": str(path)}
            if target_kind is not None
            else {"kind": "unknown", "path": None}
        )
        return FolderAccessNeededError(reason=reason, operation=operation, target=target)

    def record_touch(self, path: Path) -> None:
        self.touched_paths.add(path.expanduser().resolve(strict=False))

    def validate_current_workspace_identity(self, *, managed: bool | None = None) -> None:
        """Reject a retained managed focus whose acknowledged root moved or vanished."""
        use_managed_policy = self.managed_mode if managed is None else managed
        if not use_managed_policy:
            return
        scope = current_workspace_scope()
        if scope is None:
            return
        root = Path(os.path.abspath(scope.project_path.expanduser()))
        try:
            if not root.is_dir() or root.resolve(strict=True) != root:
                raise PermissionError(
                    "Active workspace path is unavailable or changed; select the folder again"
                )
        except (OSError, RuntimeError, ValueError) as exc:
            raise PermissionError(
                "Active workspace path is unavailable or changed; select the folder again"
            ) from exc

    def is_hidden(self, path: Path) -> bool:
        if self.managed_mode and self.folder_grants is not None:
            if self.folder_grants.is_private_path(path):
                return True
        target = self.context_manager.find_target_repo_for_path(path)
        if target:
            rel = target.rel_path(path)
            return bool(rel and target.is_protected(rel))
        repo = self.context_manager.find_repo_for_path(path)
        if not repo:
            return False
        rel = repo.rel_path(path)
        return bool(rel and repo.is_protected(rel))

    def validate_exec(self, command: str, cwd: str | Path) -> str | FolderAccessNeededError | None:
        cwd_path = Path(cwd).expanduser().resolve(strict=False)
        if self.managed_mode and self._current_full_reference() is None:
            try:
                self.validate_path(cwd_path, "exec", target_kind="directory")
            except FolderAccessNeededError as exc:
                return exc
            except PermissionError as exc:
                return f"Error: Command blocked by resource policy ({exc})"
            if self._is_managed_repository_path(cwd_path) or any(
                self._is_managed_repository_path(candidate)
                for candidate in self._extract_command_paths(command, cwd_path)
            ):
                return ("Error: Command blocked by resource policy "
                        "(managed_repository_policy: use file tools for configured repositories; "
                        "their protected paths and managed stores are unavailable to shell commands)")
            # Arguments are not execution grants. A compiler may read libraries
            # anywhere; the Sandbox applies write rules to every child process.
            return None
        command_paths = self._extract_command_paths(command, cwd_path)
        try:
            self._deny_context_repo_exec(cwd_path)
            for candidate in command_paths:
                self._deny_context_repo_exec(candidate)
        except PermissionError as exc:
            return f"Error: Command blocked by resource policy ({exc})"

        try:
            self.validate_current_workspace_identity()
        except PermissionError as exc:
            return f"Error: Active workspace is unavailable ({exc})"
        if self.managed_mode:
            if self._is_managed_repository_path(cwd_path):
                return (
                    "Error: Command blocked by resource policy "
                    "(managed_repository_policy: commands are unavailable in configured repositories)"
                )
            access = current_tool_workspace(self.workspace)
            focus = access.project_path or self.workspace
            try:
                self.validate_path(focus, "exec", target_kind="directory")
            except FolderAccessNeededError as exc:
                return exc
            except PermissionError as exc:
                return f"Error: Commands are unavailable for the active workspace ({exc})"
        try:
            self.validate_path(cwd_path, "exec", target_kind="directory")
        except FolderAccessNeededError as exc:
            return exc
        except PermissionError as exc:
            return f"Error: Command blocked by resource policy ({exc})"

        for candidate in command_paths:
            if self.managed_mode and self._is_managed_repository_path(candidate):
                return (
                    "Error: Command blocked by resource policy "
                    "(managed_repository_policy: commands cannot access configured repositories)"
                )
            target = self.context_manager.find_target_repo_for_path(candidate)
            repo = self.context_manager.find_repo_for_path(candidate)
            if target and (repo is None or len(str(target.path or "")) >= len(str(repo.path))):
                rel = target.rel_path(candidate)
                if rel and target.is_protected(rel):
                    return "Error: Command blocked by resource policy (protected target repo path referenced)"
                if rel and target.requires_proposal(rel):
                    return "Error: Command blocked by resource policy (target repo path requires a proposal)"
            elif repo:
                rel = repo.rel_path(candidate)
                if rel and repo.is_protected(rel):
                    return "Error: Command blocked by resource policy (protected context path referenced)"
            elif self.managed_mode:
                try:
                    self.validate_path(candidate, "exec")
                except FolderAccessNeededError as exc:
                    return FolderAccessNeededError(
                        reason=exc.reason,
                        operation="execute",
                        target={"kind": "unknown", "path": None},
                    )
                except PermissionError as exc:
                    return f"Error: Command blocked by resource policy ({exc})"
            elif self.restrict_to_workspace and not _is_under(candidate, self.workspace):
                if self._current_full_reference() is None:
                    return FolderAccessNeededError(
                        reason="folder_not_granted",
                        operation="execute",
                        target={"kind": "unknown", "path": None},
                    )
                try:
                    self.validate_path(candidate, "exec")
                except FolderAccessNeededError as exc:
                    return FolderAccessNeededError(
                        reason=exc.reason,
                        operation="execute",
                        target={"kind": "unknown", "path": None},
                    )
                except PermissionError as exc:
                    return f"Error: Command blocked by resource policy ({exc})"
        return None

    def _deny_context_repo_exec(self, path: Path) -> None:
        """Reject execution for any configured context repo that contains the path."""
        resolved = path.expanduser().resolve(strict=False)
        for repo in self.context_manager.repos:
            if not repo.allow_exec and repo.rel_path(resolved) is not None:
                raise PermissionError(
                    f"execution is disabled by context repo policy ({repo.name})"
                )

    def managed_repository_roots(self) -> tuple[Path, ...]:
        """Return canonical configured context and target roots for managed exec."""
        roots = {
            path.expanduser().resolve(strict=False)
            for path in self.context_manager.paths
        }
        roots.update(
            target.path.expanduser().resolve(strict=False)
            for target in self.context_manager.target_repos()
            if target.path is not None
        )
        return tuple(sorted(roots, key=lambda path: (len(path.parts), str(path))))

    def _is_managed_repository_path(self, path: Path) -> bool:
        return any(_is_under(path, root) for root in self.managed_repository_roots())

    def _validate_context_path(self, repo: ManagedContextRepo, path: Path, action: str) -> None:
        rel = repo.rel_path(path)
        if rel is None:
            raise PermissionError(f"Path {path} is outside context repo {repo.path}")
        if repo.is_protected(rel):
            raise PermissionError(f"Path {path} is protected by context repo policy")
        if action in {"write", "edit"}:
            if repo.read_only:
                raise PermissionError(f"Context repo {repo.name} is read-only")
            if repo.requires_proposal(rel):
                raise PermissionError(
                    f"Path {path} requires a proposal; write the proposed change under the configured proposal area first"
                )
            if repo.blocks_direct_store_edit(rel):
                raise PermissionError(
                    f"Path {path} is a managed store; use the registered store command instead"
                )
            if not repo.is_writable(rel):
                raise PermissionError(f"Path {path} is not in a managed writable area")

    def _validate_target_path(self, target: ManagedTargetRepo, path: Path, action: str) -> None:
        rel = target.rel_path(path)
        if rel is None:
            raise PermissionError(f"Path {path} is outside target repo {target.name}")
        if target.is_protected(rel):
            raise PermissionError(f"Path {path} is protected by target repo policy")
        if action in {"read", "list", "exec"} and not target.is_readable(rel):
            raise PermissionError(f"Path {path} is not in a readable target repo area")
        if action in {"write", "edit"}:
            if target.requires_proposal(rel):
                raise PermissionError(
                    f"Path {path} requires a proposal; write the proposed change under the configured proposal area first"
                )
            if not target.is_writable(rel):
                raise PermissionError(f"Path {path} is not in a writable target repo area")

    def _extract_command_paths(self, command: str, cwd: Path) -> list[Path]:
        paths: list[Path] = []
        tokens: list[str]
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = re.split(r"\s+", command)

        for token in tokens:
            cleaned = token.strip().strip("'\"")
            if not cleaned or any(marker in cleaned for marker in ("$", "*", "?", "|", ";", "&&", "||")):
                continue
            if _looks_like_path(cleaned):
                try:
                    expanded = os.path.expandvars(cleaned)
                    candidate = Path(expanded).expanduser()
                    if not candidate.is_absolute():
                        candidate = cwd / candidate
                    paths.append(candidate.resolve(strict=False))
                except Exception:
                    continue
        return paths


def _looks_like_path(token: str) -> bool:
    if token.startswith(("/", "~", "./", "../")):
        return True
    if re.match(r"^[A-Za-z]:\\", token):
        return True
    return any(char in token for char in _PATHISH_CHARS)


def _is_under(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory.resolve(strict=False))
        return True
    except ValueError:
        return False
