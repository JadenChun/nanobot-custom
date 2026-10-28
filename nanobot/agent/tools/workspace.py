"""Agent-controlled selection of an already authorized WebUI project root."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.turn import ToolOutcome, TurnContext
from nanobot.context_repo.policy import FolderAccessNeededError, ResourceAccessPolicy
from nanobot.session.workspace_focus import WorkspaceFocusOwner
from nanobot.workspace_access_recovery import parse_folder_access_need

FOLDER_ACCESS_SAFE_FINAL = (
    "Folder access is needed to continue this task. Choose a folder or use an allowed project."
)


def folder_access_outcome(
    error: FolderAccessNeededError,
    policy: ResourceAccessPolicy,
    *,
    target: dict[str, object] | None = None,
    purpose: str | None = None,
) -> ToolOutcome:
    """Convert only the typed policy decision into the safe turn-stop envelope."""
    from nanobot.agent.tools.context import current_request_context

    request = current_request_context()
    display_purpose = purpose
    if display_purpose is None and request is not None:
        display_purpose = request.original_user_text
    if not isinstance(display_purpose, str) or not display_purpose.strip():
        display_purpose = "Continue the current task"
    display_purpose = " ".join(display_purpose.split())[:240].strip()
    roots, truncated = policy.readable_roots_summary()
    need = parse_folder_access_need({
        "version": 1,
        "reason": error.reason,
        "operation": error.operation,
        "target": target if target is not None else error.target,
        "purpose": display_purpose,
        "allowed_roots": roots,
        "roots_truncated": truncated,
    })
    return ToolOutcome(
        content=FOLDER_ACCESS_SAFE_FINAL,
        stop_reason="folder_access_required",
        policy_metadata={"folder_access": need},
    )


class SelectWorkspaceTool(Tool):
    """Select or clear the current project's focus for the owning WebUI chat."""

    _PARAMETERS: dict[str, Any] = {
        "type": "object",
        "required": ["action"],
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["select", "clear"]},
            "path": {"type": "string"},
            "repository": {"type": "string"},
        },
    }

    def __init__(self, owner: WorkspaceFocusOwner | None) -> None:
        self._owner = owner

    @property
    def name(self) -> str:
        return "select_workspace"

    @property
    def description(self) -> str:
        return (
            "Select an existing readable project folder or catalog repository for this chat, "
            "or clear focus back to agent home. Selection does not grant access."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return deepcopy(self._PARAMETERS)

    @property
    def exclusive(self) -> bool:
        return True

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return ctx.config.file.enable

    @classmethod
    def create(cls, ctx: ToolContext) -> SelectWorkspaceTool:
        return cls(ctx.workspace_focus_owner)

    async def execute(self, action: str, path: str = "", repository: str = "", **kwargs: Any) -> str:
        _ = action, path, repository, kwargs
        return ToolResult.error("Workspace selection requires an active owning WebSocket turn")

    async def execute_with_context(self, context: TurnContext, **kwargs: Any) -> str:
        if self._owner is None:
            return ToolResult.error("Workspace selection is unavailable")
        action = kwargs.get("action")
        path = kwargs.get("path", "")
        repository = kwargs.get("repository", "")
        if not isinstance(action, str) or not isinstance(path, str) or not isinstance(repository, str):
            return ToolResult.error("Workspace selection parameters are invalid")
        return await self._owner.select_from_tool(
            context,
            action=action,
            path=path,
            repository=repository,
        )


class RequestFolderAccessTool(Tool):
    """Ask the local user to identify an unknown or exact project folder."""

    _PARAMETERS: dict[str, Any] = {
        "type": "object",
        "required": ["purpose"],
        "additionalProperties": False,
        "properties": {
            "purpose": {"type": "string", "minLength": 1, "maxLength": 240},
            "path": {"type": "string", "minLength": 1, "maxLength": 2048},
        },
    }

    def __init__(self, owner: WorkspaceFocusOwner | None) -> None:
        self._owner = owner

    @property
    def name(self) -> str:
        return "request_folder_access"

    @property
    def description(self) -> str:
        return (
            "Request local folder selection when the requested project cannot be located "
            "within readable catalog roots. This requests no permission by itself."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return deepcopy(self._PARAMETERS)

    @property
    def exclusive(self) -> bool:
        return True

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return ctx.config.file.enable

    @classmethod
    def create(cls, ctx: ToolContext) -> RequestFolderAccessTool:
        return cls(ctx.workspace_focus_owner)

    async def execute(self, purpose: str = "", path: str | None = None, **kwargs: Any) -> str:
        _ = purpose, path, kwargs
        return ToolResult.error("Folder selection requires an active owning WebSocket turn")

    async def execute_with_context(self, context: TurnContext, **kwargs: Any) -> str | ToolOutcome:
        if self._owner is None:
            return ToolResult.error("Folder selection is unavailable")
        purpose = kwargs.get("purpose")
        path = kwargs.get("path")
        if not isinstance(purpose, str) or not 1 <= len(purpose) <= 240:
            return ToolResult.error("Folder selection purpose is invalid")
        if path is not None and (not isinstance(path, str) or not 1 <= len(path) <= 2048):
            return ToolResult.error("Folder selection path is invalid")
        return await self._owner.request_access_from_tool(
            context,
            purpose=purpose,
            path=path,
        )
