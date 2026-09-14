"""Configured, invocation-local artifact reader admission.

This module owns the boundary between inert configuration and private artifact
storage. It captures paths at loop construction, performs no private-store work
until an admitted source invocation asks for a reader, and never moves an
invalid root to a fallback location.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from nanobot.agent.tools.filesystem import ReadFileTool
from nanobot.security.workspace_access import current_tool_workspace
from nanobot.session.artifact import ArtifactValidationError
from nanobot.session.artifact_reader import ArtifactReader
from nanobot.session.artifact_store import ArtifactStorageError, PrivateArtifactStore

if TYPE_CHECKING:
    from nanobot.session.turn_writer import SourceTurnWriter


@dataclass(frozen=True, slots=True)
class ArtifactStoreSettings:
    """Captured storage settings for one loop lifetime."""

    root: Path
    max_blob_bytes: int
    max_namespace_bytes: int


@dataclass(frozen=True, slots=True)
class ArtifactLocationValidator:
    """Frozen root policy rechecked before the configured store is used."""

    root: Path
    forbidden_roots: tuple[Path, ...]

    def with_roots(self, roots: Iterable[Path | str]) -> ArtifactLocationValidator:
        return ArtifactLocationValidator(
            self.root,
            (*self.forbidden_roots, *(_resolved(path) for path in roots)),
        )

    def revalidate(self) -> None:
        _reject_symlink_components(self.root)
        for path in (*self.forbidden_roots, *_git_ancestors(self.root)):
            if _overlaps(self.root, path):
                raise ArtifactValidationError(
                    "Configured artifact root overlaps a readable repository or root"
                )


def _absolute_path(value: Path | str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path == Path(path.anchor):
        raise ArtifactValidationError("Configured artifact root is invalid")
    # Collapse lexical ``.`` / ``..`` only. Resolving an existing link would
    # turn a rejected configured alias into an accepted private location.
    return Path(os.path.abspath(os.fspath(path)))


def _resolved(value: Path | str) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _reject_symlink_components(path: Path) -> None:
    """Reject aliases before a store could create data through one."""
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ArtifactValidationError("Configured artifact root is unavailable") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ArtifactValidationError("Configured artifact root must not use symlinks")
        if index != len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ArtifactValidationError("Configured artifact root has a non-directory ancestor")
        if index == len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ArtifactValidationError("Configured artifact root is not a directory")


def _git_ancestors(path: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    current = path
    while True:
        marker = current / ".git"
        try:
            marker.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ArtifactValidationError("Configured artifact root cannot inspect repository state") from exc
        else:
            roots.append(current)
        if current.parent == current:
            return tuple(roots)
        current = current.parent


def _path_items(value: object) -> tuple[Path, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Path))


class ConfiguredArtifactReaderFactory:
    """Create a fresh reader only for an admitted source writer.

    Settings and known readable roots are captured once. Descriptor grants and
    readers remain invocation-local and are never cached by this factory.
    """

    __slots__ = ("_settings", "_location_validator")

    def __init__(
        self,
        *,
        root: Path | str,
        max_blob_bytes: int,
        max_namespace_bytes: int,
        static_roots: Iterable[Path | str] = (),
        media_root: Path | str | None = None,
    ) -> None:
        root_path = _absolute_path(root)
        if (
            type(max_blob_bytes) is not int
            or max_blob_bytes <= 0
            or type(max_namespace_bytes) is not int
            or max_namespace_bytes < max_blob_bytes
        ):
            raise ArtifactValidationError("Configured artifact quota is invalid")
        captured = [_resolved(path) for path in static_roots]
        if media_root is not None:
            captured.append(_resolved(media_root))
        self._settings = ArtifactStoreSettings(root_path, max_blob_bytes, max_namespace_bytes)
        self._location_validator = ArtifactLocationValidator(
            root_path,
            (*captured, _resolved(self._source_checkout())),
        )
        self._location_validator.revalidate()

    @property
    def settings(self) -> ArtifactStoreSettings:
        return self._settings

    @staticmethod
    def _source_checkout() -> Path:
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _reader_roots(writer: SourceTurnWriter, read_file: ReadFileTool) -> tuple[Path, ...]:
        backend = writer.store._backend
        roots = [backend.workspace, backend.sessions_dir, backend.sessions_dir.parent]
        for name in ("_workspace", "_allowed_dir"):
            value = getattr(read_file, name, None)
            if isinstance(value, Path):
                roots.append(value)
        roots.extend(_path_items(getattr(read_file, "_extra_read_allowed_dirs", ())))
        roots.extend(_path_items(getattr(read_file, "_extra_write_allowed_dirs", ())))
        roots.extend(_path_items(getattr(read_file, "_extra_read_allowed_files", ())))
        policy = read_file._resource_policy
        if policy is not None:
            roots.extend(policy.extra_read_dirs())
            roots.extend(policy.extra_write_dirs())
            manager = policy.context_manager
            roots.extend(target.path for target in manager.target_repos())
        access = current_tool_workspace(
            read_file._workspace,
            restrict_to_workspace=read_file._restrict_to_workspace,
            sandbox_restricts_workspace=read_file._sandbox_restricts_workspace,
        )
        if access.project_path is not None:
            roots.append(access.project_path)
        if access.allowed_root is not None:
            roots.append(access.allowed_root)
        return tuple(roots)

    def __call__(
        self,
        writer: SourceTurnWriter,
        isolated: bool,
        read_file: ReadFileTool,
    ) -> ArtifactReader | None:
        if type(read_file) is not ReadFileTool:
            return None
        try:
            location_validator = self._location_validator.with_roots(
                self._reader_roots(writer, read_file)
            )
            location_validator.revalidate()
            state = writer.owner.read_snapshot().state
            backend = writer.store._backend
            store = PrivateArtifactStore(
                self._settings.root,
                workspace_id=backend.sessions_dir.name,
                session_key=state.session_key,
                incarnation=state.incarnation,
                generation=state.generation,
                max_blob_bytes=self._settings.max_blob_bytes,
                max_namespace_bytes=self._settings.max_namespace_bytes,
                location_validator=location_validator.revalidate,
            )
            return ArtifactReader(
                writer,
                "isolated_stream" if isolated else "root_history",
                store,
                read_file,
                location_validator=location_validator.revalidate,
            )
        except (ArtifactStorageError, ArtifactValidationError, OSError):
            return None


__all__ = ["ArtifactStoreSettings", "ConfiguredArtifactReaderFactory"]
