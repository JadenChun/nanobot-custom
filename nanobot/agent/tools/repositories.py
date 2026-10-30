"""Generic repository references and deliberate knowledge write-back."""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from filelock import AsyncFileLock

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool
from nanobot.context_repo.catalog import RepositoryCatalog, RepositoryEntry
from nanobot.context_repo.policy import FolderAccessNeededError
from nanobot.security.workspace_access import current_workspace_scope
from nanobot.session.workspace_focus import WorkspaceFocusOwner
from nanobot.utils.git_sync import async_sync_context_repo, matches_sync_path
from nanobot.utils.secret_scan import find_secrets_in_blob

if TYPE_CHECKING:
    from nanobot.agent.turn import ToolOutcome


class RepositoryCatalogTool(Tool):
    @property
    def name(self) -> str:
        return "repository_catalog"

    @property
    def description(self) -> str:
        return (
            "List, discover, remember/update, or forget existing local folders and their purposes. In local WebUI chats, remember proposes a folder for user approval. "
            "Use knowledge_repos to associate project repositories with designated knowledge repos. "
            "Use roots then discover to find projects under authorized folders. Use select_workspace "
            "to change the current project; catalog metadata never grants access. "
            "Only knowledge repos may opt into auto_sync with explicit sync_paths."
        )

    _PARAMETERS: dict[str, Any] = {
        "type": "object", "required": ["action"], "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["list", "remember", "forget", "discover", "roots"]},
            "label": {"type": "string", "maxLength": 80},
            "name": {"type": "string"}, "path": {"type": "string"},
            "purpose": {"type": "string"}, "kind": {"type": "string", "enum": ["project", "knowledge"]},
            "knowledge_repos": {"type": "array", "items": {"type": "string"}},
            "agent_map": {"type": "boolean"},
            "file_access": {"type": "string", "enum": ["read_only", "read_write"]},
            "auto_sync": {"type": "boolean"}, "sync_paths": {"type": "array", "items": {"type": "string"}},
        },
    }

    @property
    def parameters(self) -> dict[str, Any]:
        return deepcopy(self._PARAMETERS)

    def __init__(
        self,
        catalog: RepositoryCatalog,
        reader: ReadFileTool,
        owner: WorkspaceFocusOwner | None = None,
    ):
        self.catalog, self.reader, self.owner = catalog, reader, owner

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return ctx.config.file.enable

    @classmethod
    def create(cls, ctx: ToolContext) -> RepositoryCatalogTool:
        root = ctx.sessions.sessions_root if ctx.sessions is not None else None
        return cls(
            RepositoryCatalog.for_workspace(Path(ctx.workspace), sessions_root=root),
            ReadFileTool.create(ctx),
            ctx.workspace_focus_owner,
        )

    async def execute(self, action: str, name: str = "", path: str = "", purpose: str = "",
                      kind: str = "project", knowledge_repos: list[str] | None = None,
                      auto_sync: bool = False, sync_paths: list[str] | None = None,
                      agent_map: bool = True, label: str = "", file_access: str = "read_only",
                      **kwargs: Any) -> str | ToolOutcome:
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
            if action == "roots":
                if not self._managed_webui_turn():
                    raise PermissionError("repository roots are available only to managed WebUI chats")
                if self.owner is None:
                    raise PermissionError("workspace access service is unavailable")
                return json.dumps(self.owner.repository_roots(), ensure_ascii=False)
            if action == "discover":
                if self.owner is None:
                    raise PermissionError("workspace access service is unavailable")
                scope = current_workspace_scope()
                start = path.strip() or (str(scope.project_path) if scope is not None else "")
                discovered = await asyncio.to_thread(self.owner.discover, start)
                return json.dumps(discovered, ensure_ascii=False)
            if action == "forget":
                self.reader.resolve_read_path(self.catalog.get(name).path)
                if any(name in entry.knowledge_repos for entry in self.catalog.entries()):
                    raise ValueError("repository_in_use")
                await asyncio.to_thread(self.catalog.update, None, remove=name)
                return f"Forgot repository reference: {name}"
            if action != "remember":
                raise ValueError("unknown repository catalog action")
            if not Path(path).expanduser().is_absolute():
                raise ValueError("use an absolute repository path")
            resolved = self.reader.resolve_read_path(path)
            if not resolved.is_dir():
                raise ValueError("path must be an existing folder")
            private = self.catalog.runtime_root.resolve()
            if kind == "knowledge" and (resolved.is_relative_to(private) or private.is_relative_to(resolved)):
                raise ValueError("repository contents must be separate from runtime catalog storage")
            entry = RepositoryEntry.model_validate({
                "name": name, "label": label, "path": str(resolved), "purpose": purpose, "kind": kind, "agent_map": agent_map,
                "knowledge_repos": knowledge_repos or [], "auto_sync": auto_sync, "sync_paths": sync_paths or [],
            })
            from nanobot.agent.tools.context import current_request_context

            request = current_request_context()
            if request is not None and request.channel == "websocket" and not request.persistent_session:
                raise ValueError("Saving folder references requires a persistent local chat")
            if self._managed_webui_turn():
                from nanobot.agent.tools.workspace import folder_access_outcome
                from nanobot.workspace_access_recovery import parse_folder_proposal

                policy = self.owner.policy if self.owner is not None else self.reader.resource_policy
                if policy is None:
                    raise ValueError("folder approval service is unavailable")
                proposal = parse_folder_proposal({"entry": entry.model_dump(), "requested_access": file_access})
                outcome = folder_access_outcome(FolderAccessNeededError(
                    reason="folder_selection_required", operation="select",
                    target={"kind": "directory", "path": str(resolved)},
                ), policy, purpose=purpose or f"Add {name} to the folder map")
                need = outcome.policy_metadata["folder_access"]
                from nanobot.workspace_access_recovery import parse_folder_access_need

                checked = parse_folder_access_need(need)
                checked["folder_proposal"] = proposal
                from nanobot.agent.turn import ToolOutcome

                return ToolOutcome(
                    content="Review the proposed folder and choose its Sandbox access in the approval card.",
                    stop_reason="folder_access_required", policy_metadata={"folder_access": checked},
                )
            await asyncio.to_thread(self.catalog.update, entry)
            return entry.model_dump_json()
        except FolderAccessNeededError as exc:
            policy = self.owner.policy if self.owner is not None else self.reader.resource_policy
            if policy is None:
                return ToolResult.error("Folder access is unavailable")
            from nanobot.agent.tools.workspace import folder_access_outcome

            return folder_access_outcome(exc, policy)
        except (OSError, ValueError) as exc:
            return ToolResult.error(f"Repository catalog: {exc}")

    @staticmethod
    def _managed_webui_turn() -> bool:
        from nanobot.agent.tools.context import current_request_context

        request = current_request_context()
        return bool(
            request
            and request.channel == "websocket"
            and request.persistent_session
        )


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
