"""Small agent-maintained index of existing repositories, outside their contents."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal, Self

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nanobot.config.paths import get_config_path


class RepositoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    path: str = Field(max_length=2048)
    purpose: str = Field(default="", max_length=240)
    kind: Literal["project", "knowledge"] = "project"
    knowledge_repos: list[str] = Field(default_factory=list, max_length=16)
    auto_sync: bool = False
    sync_paths: list[str] = Field(default_factory=list, max_length=64)

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        if not Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("repository path must be absolute without '..'")
        return value

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
    version: Literal[1] = 1
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

    def entries(self) -> list[RepositoryEntry]:
        if not self.path.exists():
            return []
        if self.path.is_symlink() or self.path.stat().st_size > 256 * 1024:
            raise ValueError("repository catalog must be a bounded regular file")
        return _CatalogData.model_validate_json(self.path.read_bytes()).repositories

    def get(self, name: str) -> RepositoryEntry:
        for entry in self.entries():
            if entry.name == name:
                return entry
        raise ValueError(f"unknown repository: {name}")

    def update(self, entry: RepositoryEntry | None, *, remove: str | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with FileLock(str(self.root / "catalog.lock"), timeout=5):
            name = entry.name if entry is not None else remove
            entries = [item for item in self.entries() if item.name != name]
            if entry is not None:
                entries.append(entry)
            data = _CatalogData(repositories=sorted(entries, key=lambda item: item.name))
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

    def prompt_summary(self, *, allowed_roots: list[Path] | None = None) -> str:
        rows: list[str] = []
        for entry in self.entries():
            if allowed_roots is not None and not any(Path(entry.path).is_relative_to(root) for root in allowed_roots):
                continue
            # JSON keeps paths/purposes on one metadata line; repo contents are
            # deliberately absent. The selected repository supplies its rules.
            rows.append(json.dumps(entry.model_dump(exclude={"sync_paths"}), ensure_ascii=True))
        return "\n".join(rows)
