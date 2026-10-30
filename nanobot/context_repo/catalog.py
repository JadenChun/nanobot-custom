"""Small agent-maintained index of existing repositories, outside their contents."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from nanobot.config.paths import get_config_path


class RepositoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    path: str = Field(max_length=2048)
    label: str = Field(default="", max_length=80)
    purpose: str = Field(default="", max_length=240)
    agent_map: bool = True
    kind: Literal["project", "knowledge"] = "project"
    knowledge_repos: list[str] = Field(default_factory=list, max_length=16)
    auto_sync: bool = False
    sync_paths: list[str] = Field(default_factory=list, max_length=64)

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not Path(value).expanduser().is_absolute() or ".." in Path(value).parts:
            raise ValueError("repository path must be absolute without '..'")
        return str(Path(value).expanduser())

    @model_validator(mode="after")
    def knowledge_sync_only(self) -> Self:
        if self.auto_sync and (self.kind != "knowledge" or not self.sync_paths):
            raise ValueError("automatic sync requires a knowledge repo and explicit sync paths")
        for pattern in self.sync_paths:
            if not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts or pattern.startswith(".git"):
                raise ValueError("sync paths must be relative repository paths or patterns")
        return self


class _CatalogData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1, 2] = 2
    revision: StrictInt = Field(default=0, ge=0)
    repositories: list[RepositoryEntry] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def valid_links(self) -> Self:
        entries = {entry.name: entry for entry in self.repositories}
        if len(entries) != len(self.repositories):
            raise ValueError("repository names must be unique")
        paths = {entry.path for entry in self.repositories}
        if len(paths) != len(self.repositories):
            raise ValueError("a repository path may only be registered once")
        for entry in self.repositories:
            for name in entry.knowledge_repos:
                if name == entry.name or name not in entries or entries[name].kind != "knowledge":
                    raise ValueError(f"invalid knowledge repository association: {name}")
        return self


class CatalogRevisionConflictError(ValueError):
    """Navigation metadata changed since the user opened the form."""


class RepositoryCatalog:
    def __init__(self, root: Path, *, runtime_root: Path | None = None):
        self.root = root
        self.path = root / "catalog.json"
        self.runtime_root = runtime_root or root

    @classmethod
    def for_workspace(cls, workspace: Path, *, sessions_root: Path | None = None) -> RepositoryCatalog:
        instance = sessions_root.parent if sessions_root is not None else get_config_path().parent
        identity = hashlib.sha256(str(workspace.expanduser().resolve()).encode()).hexdigest()[:24]
        return cls(instance / "repositories" / identity, runtime_root=instance)

    def snapshot(self) -> _CatalogData:
        if not self.path.exists():
            return _CatalogData()
        if self.path.is_symlink() or self.path.stat().st_size > 256 * 1024:
            raise ValueError("repository catalog must be a bounded regular file")
        data = _CatalogData.model_validate_json(self.path.read_bytes())
        # Canonical aliases become one location, retaining knowledge links.
        canonical: dict[str, RepositoryEntry] = {}
        aliases: dict[str, str] = {}
        for entry in data.repositories:
            path = Path(entry.path)
            normalized = str(path.resolve(strict=False)) if data.version == 1 and path.is_dir() else entry.path
            previous = canonical.get(normalized)
            if previous is None:
                canonical[normalized] = entry.model_copy(update={"path": normalized})
            else:
                aliases[entry.name] = previous.name
                canonical[normalized] = previous.model_copy(update={
                    "agent_map": previous.agent_map or entry.agent_map,
                    "purpose": previous.purpose or entry.purpose,
                    "kind": "knowledge" if "knowledge" in {previous.kind, entry.kind} else "project",
                    "knowledge_repos": list(dict.fromkeys([*previous.knowledge_repos, *entry.knowledge_repos])),
                    "auto_sync": previous.auto_sync or entry.auto_sync,
                    "sync_paths": list(dict.fromkeys([*previous.sync_paths, *entry.sync_paths])),
                })
        entries = [RepositoryEntry.model_validate({**entry.model_dump(), "knowledge_repos": list(dict.fromkeys(
            aliases.get(name, name) for name in entry.knowledge_repos if aliases.get(name, name) != entry.name
        ))}) for entry in canonical.values()]
        return _CatalogData(version=data.version, revision=data.revision, repositories=entries)

    def entries(self) -> list[RepositoryEntry]:
        return self.snapshot().repositories

    def get(self, name: str) -> RepositoryEntry:
        for entry in self.entries():
            if entry.name == name:
                return entry
        raise ValueError(f"unknown repository: {name}")

    def update(self, entry: RepositoryEntry | None, *, remove: str | None = None, expected_revision: int | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with FileLock(str(self.root / "catalog.lock"), timeout=5):
            current = self.snapshot()
            if expected_revision is not None and current.revision != expected_revision:
                raise CatalogRevisionConflictError("catalog_changed")
            name = entry.name if entry is not None else remove
            entries = [item for item in current.repositories if item.name != name]
            if entry is not None:
                entries.append(entry)
            data = _CatalogData(revision=current.revision + 1, repositories=sorted(entries, key=lambda item: item.name))
            if current.version == 1 and self.path.exists():
                backup = self.root / "catalog.pre-v2.json"
                if not backup.exists():
                    with backup.open("xb") as stream:
                        stream.write(self.path.read_bytes())
                        stream.flush()
                        os.fsync(stream.fileno())
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write((data.model_dump_json(indent=2) + "\n").encode())
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
                fd = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def prompt_summary(self, *, allowed_roots: list[Path] | None = None, access_for_path: Callable[[Path], str | None] | None = None) -> str:
        rows: list[str] = []
        for entry in self.entries():
            if not entry.agent_map:
                continue
            if len(rows) >= 24:
                rows.append('More folders are available with repository_catalog(action="list").')
                break
            if allowed_roots is not None and not any(Path(entry.path).is_relative_to(root) for root in allowed_roots):
                continue
            # JSON keeps paths/purposes on one metadata line; repo contents are
            # deliberately absent. The selected repository supplies its rules.
            access = access_for_path(Path(entry.path)) if access_for_path is not None else None
            if access_for_path is not None and access is None:
                continue
            row = entry.model_dump(exclude={"sync_paths", "auto_sync", "agent_map"})
            if access is not None:
                row["effective_access"] = access
            rows.append(json.dumps(row, ensure_ascii=True))
        return "\n".join(rows)
