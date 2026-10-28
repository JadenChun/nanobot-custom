"""Bounded, permission-aware discovery of existing Git working trees."""

from __future__ import annotations

import re
import stat
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal, TypedDict

from nanobot.context_repo.catalog import RepositoryCatalog
from nanobot.context_repo.policy import ResourceAccessPolicy
from nanobot.security.folder_grants import FolderGrantsUnavailableError
from nanobot.utils.git_sync import git_repo_root

_MAX_DEPTH = 4
_MAX_VISITED = 512
_MAX_CANDIDATES = 32
_SKIPPED_DIRECTORY_NAMES = frozenset({
    "node_modules", ".venv", "venv", "__pycache__", "build", "dist", "target", "cache",
})
_SAFE_NAME = re.compile(r"[^a-z0-9_-]+")


WorkspaceAccessBoundary = Literal["folder", "unrestricted"]


class EffectiveCommandAccess(TypedDict):
    requested: bool
    available: bool
    backend: str
    reason: str | None


class EffectiveWorkspaceAccess(TypedDict):
    boundary: WorkspaceAccessBoundary
    policy: str
    source: Literal[
        "agent_home", "configured", "grant", "task", "full", "none",
    ]
    grant_root: str | None
    exists: bool
    read: bool
    write: bool
    commands: EffectiveCommandAccess


class RepositoryCandidate(TypedDict):
    path: str
    suggested_name: str
    known_name: str | None
    effective_access: EffectiveWorkspaceAccess


class RepositoryDiscoveryPayload(TypedDict):
    root: str
    candidates: list[RepositoryCandidate]
    visited_directories: int
    truncated: bool


def discover_repositories(
    root: str | Path,
    *,
    policy: ResourceAccessPolicy,
    catalog: RepositoryCatalog,
    access_for_path: Callable[[Path], EffectiveWorkspaceAccess],
    extra_private_roots: Iterable[Path] = (),
    managed: bool | None = None,
    allow_folder_grants: bool | None = None,
) -> RepositoryDiscoveryPayload:
    """Find Git roots under a readable directory without following symlinks."""
    try:
        canonical_root = Path(root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("discovery root must be an existing directory") from exc
    if not canonical_root.is_dir():
        raise ValueError("discovery root must be an existing directory")
    _require_read(
        policy,
        canonical_root,
        managed=managed,
        allow_folder_grants=allow_folder_grants,
    )

    known_paths = {entry.path: entry.name for entry in catalog.entries()}
    private_roots = tuple(path.expanduser().resolve(strict=False) for path in extra_private_roots)
    visited = 0
    truncated = False
    candidates: list[RepositoryCandidate] = []
    stack: list[tuple[Path, int]] = [(canonical_root, 0)]

    while stack:
        directory, depth = stack.pop()
        if any(_is_under(directory, private) for private in private_roots):
            continue
        if visited >= _MAX_VISITED:
            truncated = True
            break
        try:
            _require_read(
                policy,
                directory,
                managed=managed,
                allow_folder_grants=allow_folder_grants,
            )
        except PermissionError:
            continue
        visited += 1

        if _has_git_marker(directory) and git_repo_root(directory) == directory:
            candidates.append({
                "path": str(directory),
                "suggested_name": _suggested_name(directory.name),
                "known_name": known_paths.get(str(directory)),
                "effective_access": access_for_path(directory),
            })
            if len(candidates) >= _MAX_CANDIDATES:
                truncated = bool(stack)
                break
            continue

        if depth >= _MAX_DEPTH:
            if _has_expandable_children(directory):
                truncated = True
            continue
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name.casefold(), reverse=True)
        except OSError:
            continue
        for child in children:
            if child.name.startswith(".") or child.name.casefold() in _SKIPPED_DIRECTORY_NAMES:
                continue
            try:
                if child.is_symlink() or not child.is_dir():
                    continue
            except OSError:
                continue
            stack.append((child, depth + 1))

    return {
        "root": str(canonical_root),
        "candidates": candidates,
        "visited_directories": visited,
        "truncated": truncated,
    }


def _require_read(
    policy: ResourceAccessPolicy,
    path: Path,
    *,
    managed: bool | None,
    allow_folder_grants: bool | None,
) -> None:
    try:
        policy.validate_path(
            path,
            "read",
            managed=managed,
            allow_folder_grants=allow_folder_grants,
        )
    except (PermissionError, FolderGrantsUnavailableError) as exc:
        raise PermissionError(f"Discovery is not allowed for {path}") from exc


def _suggested_name(value: str) -> str:
    name = _SAFE_NAME.sub("-", value.casefold()).strip("-_")[:64].rstrip("-_")
    return name if name and name[0].isalnum() else "repository"


def _has_expandable_children(directory: Path) -> bool:
    try:
        return any(
            child.is_dir() and not child.is_symlink()
            and not child.name.startswith(".")
            and child.name.casefold() not in _SKIPPED_DIRECTORY_NAMES
            for child in directory.iterdir()
        )
    except OSError:
        return False


def _has_git_marker(directory: Path) -> bool:
    try:
        mode = (directory / ".git").lstat().st_mode
    except OSError:
        return False
    return stat.S_ISDIR(mode) or stat.S_ISREG(mode)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
