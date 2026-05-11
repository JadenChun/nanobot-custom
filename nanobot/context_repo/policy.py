"""Resource access policy for workspace and managed context repositories."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from nanobot.context_repo.manager import ContextRepoManager, ManagedContextRepo, ManagedTargetRepo


_PATHISH_CHARS = {"/", "\\"}


class ResourceAccessPolicy:
    """Validate file and shell access against workspace/context repo boundaries."""

    def __init__(
        self,
        *,
        workspace: Path,
        context_manager: ContextRepoManager | None = None,
        restrict_to_workspace: bool = False,
    ):
        self.workspace = workspace.expanduser().resolve(strict=False)
        self.context_manager = context_manager or ContextRepoManager()
        self.restrict_to_workspace = restrict_to_workspace
        self.touched_paths: set[Path] = set()

    def extra_read_dirs(self) -> list[Path]:
        return self.context_manager.read_roots()

    def extra_write_dirs(self) -> list[Path]:
        return self.context_manager.write_roots()

    def validate_path(self, path: Path, action: str) -> None:
        resolved = path.expanduser().resolve(strict=False)
        target = self.context_manager.find_target_repo_for_path(resolved)
        repo = self.context_manager.find_repo_for_path(resolved)
        if target and (repo is None or len(str(target.path or "")) >= len(str(repo.path))):
            self._validate_target_path(target, resolved, action)
            return
        if repo:
            self._validate_context_path(repo, resolved, action)
            return

        if self.restrict_to_workspace and not _is_under(resolved, self.workspace):
            raise PermissionError(f"Path {path} is outside allowed workspace or context repositories")

    def record_touch(self, path: Path) -> None:
        self.touched_paths.add(path.expanduser().resolve(strict=False))

    def is_hidden(self, path: Path) -> bool:
        target = self.context_manager.find_target_repo_for_path(path)
        if target:
            rel = target.rel_path(path)
            return bool(rel and target.is_protected(rel))
        repo = self.context_manager.find_repo_for_path(path)
        if not repo:
            return False
        rel = repo.rel_path(path)
        return bool(rel and repo.is_protected(rel))

    def validate_exec(self, command: str, cwd: str | Path) -> str | None:
        cwd_path = Path(cwd).expanduser().resolve(strict=False)
        try:
            self.validate_path(cwd_path, "exec")
        except PermissionError as exc:
            return f"Error: Command blocked by resource policy ({exc})"

        for candidate in self._extract_command_paths(command, cwd_path):
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
            elif self.restrict_to_workspace and not _is_under(candidate, self.workspace):
                return "Error: Command blocked by resource policy (path outside workspace/context repositories)"
        return None

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
