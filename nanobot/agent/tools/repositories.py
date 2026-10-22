"""Generic repository references and deliberate knowledge write-back."""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from filelock import AsyncFileLock

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool
from nanobot.context_repo.catalog import RepositoryCatalog, RepositoryEntry
from nanobot.utils.git_sync import async_sync_context_repo, git_repo_root, matches_sync_path
from nanobot.utils.secret_scan import find_secrets_in_blob


class RepositoryCatalogTool(Tool):
    @property
    def name(self) -> str:
        return "repository_catalog"

    @property
    def description(self) -> str:
        return (
            "List, remember/update, or forget existing local Git repositories and their purposes. "
            "Use knowledge_repos to associate project repositories with designated knowledge repos. "
            "This stores references; use ordinary file tools with absolute paths to read context. "
            "Only knowledge repos may opt into auto_sync with explicit sync_paths."
        )

    _PARAMETERS: dict[str, Any] = {
        "type": "object", "required": ["action"], "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["list", "remember", "forget"]},
            "name": {"type": "string"}, "path": {"type": "string"},
            "purpose": {"type": "string"}, "kind": {"type": "string", "enum": ["project", "knowledge"]},
            "knowledge_repos": {"type": "array", "items": {"type": "string"}},
            "auto_sync": {"type": "boolean"}, "sync_paths": {"type": "array", "items": {"type": "string"}},
        },
    }

    @property
    def parameters(self) -> dict[str, Any]:
        return deepcopy(self._PARAMETERS)

    def __init__(self, catalog: RepositoryCatalog, reader: ReadFileTool):
        self.catalog, self.reader = catalog, reader

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return ctx.config.file.enable

    @classmethod
    def create(cls, ctx: ToolContext) -> RepositoryCatalogTool:
        root = ctx.sessions.sessions_root if ctx.sessions is not None else None
        return cls(RepositoryCatalog.for_workspace(Path(ctx.workspace), sessions_root=root), ReadFileTool.create(ctx))

    async def execute(self, action: str, name: str = "", path: str = "", purpose: str = "",
                      kind: str = "project", knowledge_repos: list[str] | None = None,
                      auto_sync: bool = False, sync_paths: list[str] | None = None, **kwargs: Any) -> str:
        try:
            if action == "list":
                visible: list[dict[str, Any]] = []
                for entry in self.catalog.entries():
                    try:
                        self.reader.resolve_read_path(entry.path)
                    except PermissionError:
                        continue
                    visible.append(entry.model_dump())
                return json.dumps(visible, ensure_ascii=False)
            if action == "forget":
                self.reader.resolve_read_path(self.catalog.get(name).path)
                await asyncio.to_thread(self.catalog.update, None, remove=name)
                return f"Forgot repository reference: {name}"
            if action != "remember":
                raise ValueError("unknown repository catalog action")
            if not Path(path).expanduser().is_absolute():
                raise ValueError("use an absolute repository path")
            resolved = self.reader.resolve_read_path(path)
            git_root = await asyncio.to_thread(git_repo_root, resolved)
            if git_root != resolved:
                raise ValueError("path must be the root of an existing Git working tree")
            private = self.catalog.runtime_root.resolve()
            if kind == "knowledge" and (resolved.is_relative_to(private) or private.is_relative_to(resolved)):
                raise ValueError("repository contents must be separate from runtime catalog storage")
            entry = RepositoryEntry.model_validate({
                "name": name, "path": str(resolved), "purpose": purpose, "kind": kind,
                "knowledge_repos": knowledge_repos or [], "auto_sync": auto_sync, "sync_paths": sync_paths or [],
            })
            await asyncio.to_thread(self.catalog.update, entry)
            return entry.model_dump_json()
        except (OSError, ValueError) as exc:
            return ToolResult.error(f"Repository catalog: {exc}")


class SaveKnowledgeTool(Tool):
    @property
    def name(self) -> str:
        return "save_knowledge"

    @property
    def description(self) -> str:
        return (
            "Automatically save selected durable facts or agreed decisions to a designated knowledge "
            "repository from repository_catalog. Appends a note once; never copies full runtime transcripts. "
            "Use the current project's knowledge_repos association and an explicit relative file path. "
            "Reports local save and remote sync separately; auto_sync only uses opted-in knowledge paths."
        )

    _PARAMETERS: dict[str, Any] = {
        "type": "object", "required": ["repository", "path", "content"], "additionalProperties": False,
        "properties": {"repository": {"type": "string"}, "path": {"type": "string"},
                       "content": {"type": "string", "minLength": 1, "maxLength": 32000}},
    }

    @property
    def parameters(self) -> dict[str, Any]:
        return deepcopy(self._PARAMETERS)

    def __init__(self, catalog: RepositoryCatalog, writer: WriteFileTool):
        self.catalog, self.writer = catalog, writer

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return ctx.config.file.enable

    @classmethod
    def create(cls, ctx: ToolContext) -> SaveKnowledgeTool:
        root = ctx.sessions.sessions_root if ctx.sessions is not None else None
        return cls(RepositoryCatalog.for_workspace(Path(ctx.workspace), sessions_root=root), WriteFileTool.create(ctx))

    async def execute(self, repository: str, path: str, content: str, **kwargs: Any) -> str:
        try:
            entry = self.catalog.get(repository)
            if entry.kind != "knowledge":
                raise ValueError("automatic knowledge writes require a designated knowledge repo")
            relative = Path(path)
            if not path or relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
                raise ValueError("knowledge path must stay within the selected repository")
            root = Path(entry.path).resolve()
            if root != Path(entry.path):
                raise ValueError("registered repository path changed; refresh its catalog entry")
            private = self.catalog.runtime_root.resolve()
            if root.is_relative_to(private) or private.is_relative_to(root):
                raise ValueError("knowledge must be separate from runtime catalog storage")
            destination = self.writer.resolve_write_path(str(root / relative))
            if not destination.is_relative_to(root):
                raise ValueError("knowledge destination escapes its repository")
            note = content.strip()
            if not note or len(note) > 32000:
                raise ValueError("knowledge note must contain 1–32000 characters")
            if find_secrets_in_blob(note):
                raise ValueError("knowledge note contains a probable credential")

            def append_note() -> bytes:
                prior = destination.read_text(encoding="utf-8") if destination.exists() else ""
                if len(prior) > 2 * 1024 * 1024:
                    raise ValueError("knowledge file is too large; choose a focused file")
                if prior.strip() == note or "\n\n" + note + "\n" in prior:
                    return prior.encode()
                return ((prior.rstrip() + "\n\n" if prior.strip() else "") + note + "\n").encode()

            locks = self.catalog.root / "knowledge-locks"
            locks.mkdir(parents=True, exist_ok=True, mode=0o700)
            identity = hashlib.sha256(str(destination).encode()).hexdigest()
            async with AsyncFileLock(str(locks / identity), timeout=10):
                await self.writer.publish_binary_output(
                    destination,
                    destination_request=lambda: str(root / relative),
                    read_bytes=append_note,
                    destination_changed_error=lambda: ValueError("knowledge destination changed"),
                )
                synced: bool | None = None
                if entry.auto_sync and matches_sync_path(relative.as_posix(), entry.sync_paths):
                    synced = await async_sync_context_repo(root, include_paths=entry.sync_paths)
            return json.dumps({"saved": True, "path": str(destination), "synced": synced,
                               "sync_state": "synced" if synced else "pending" if synced is False else "not_enabled"})
        except (OSError, ValueError) as exc:
            return ToolResult.error(f"Knowledge save: {exc}")
