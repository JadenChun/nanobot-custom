"""Pure, bounded data edges for task-linked folder access recovery.

This module intentionally lives outside the eager ``nanobot.session`` package
so policy and tool imports can use the schema without loading session operators.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, TypeGuard, cast

if TYPE_CHECKING:
    from nanobot.context_repo.discovery import EffectiveWorkspaceAccess

FolderAccessReason = Literal[
    "folder_not_granted", "write_not_granted", "commands_not_granted",
    "folder_selection_required",
]
FolderAccessOperation = Literal["read", "write", "execute", "discover", "select", "locate"]
FolderAccessRequestStatus = Literal[
    "pending", "ready", "continuing", "continued", "cancelled", "invalidated",
]
FolderAccessDuration = Literal["once", "always"]
TaskAccessState = Literal["active", "retiring", "expired"]
PermissionFullUnavailableReason = Literal[
    "controls_unavailable", "session_required", "configured_sandbox",
    "supervisor_unavailable", "permission_state_unavailable", "permission_retiring",
]
FolderAccessUnavailableReason = Literal[
    "task_changed", "focus_changed", "source_changed", "session_reset",
    "session_deleted", "access_changed", "path_missing", "path_changed",
    "context_unavailable", "continuation_unknown", "task_access_expired",
    "permission_retiring",
]


class FolderAccessKnownTarget(TypedDict):
    kind: Literal["directory", "file"]
    path: str


class FolderAccessUnknownTarget(TypedDict):
    kind: Literal["unknown"]
    path: None


FolderAccessTarget = FolderAccessKnownTarget | FolderAccessUnknownTarget


class FolderProposalEntry(TypedDict):
    name: str
    label: str
    path: str
    purpose: str
    agent_map: bool
    kind: Literal["project", "knowledge"]
    knowledge_repos: list[str]
    auto_sync: bool
    sync_paths: list[str]


class FolderProposal(TypedDict):
    entry: FolderProposalEntry
    requested_access: Literal["read_only", "read_write"]


def parse_folder_proposal(raw: object) -> FolderProposal:
    from nanobot.context_repo.catalog import RepositoryEntry

    if not _has_exact_keys(raw, {"entry", "requested_access"}):
        raise ValueError("invalid folder proposal")
    access = raw["requested_access"]
    if access not in {"read_only", "read_write"}:
        raise ValueError("invalid folder proposal access")
    if len(json.dumps(raw, ensure_ascii=False).encode("utf-8")) > 8192:
        raise ValueError("folder proposal is too large")
    entry = RepositoryEntry.model_validate(raw["entry"])
    return {"entry": {
        "name": entry.name, "label": entry.label, "path": entry.path, "purpose": entry.purpose,
        "agent_map": entry.agent_map, "kind": entry.kind, "knowledge_repos": entry.knowledge_repos,
        "auto_sync": entry.auto_sync, "sync_paths": entry.sync_paths,
    }, "requested_access": cast(Literal["read_only", "read_write"], access)}


class FolderAccessNeed(TypedDict):
    version: Literal[1]
    reason: FolderAccessReason
    operation: FolderAccessOperation
    target: FolderAccessTarget
    purpose: str
    folder_proposal: NotRequired[FolderProposal]
    allowed_roots: list[str]
    roots_truncated: bool


class WorkspaceAccessSourceBinding(TypedDict):
    store_identity: str
    incarnation: str
    generation: int


class WorkspaceAccessTaskAnchor(TypedDict):
    turn_id: str


class WorkspaceAccessExpectedScope(TypedDict):
    project_path: str
    access_mode: Literal["restricted", "full"]


class WorkspaceScopeMetadata(TypedDict):
    project_path: str
    access_mode: Literal["restricted", "full"]


class WorkspaceAccessRecord(TypedDict):
    """Validated durable fields; derived public view fields are intentionally absent."""

    version: Literal[1]
    access_request_id: str
    revision: int
    task: WorkspaceAccessTaskAnchor
    blocked_turn_id: str
    tool_call_id: str | None
    created_at_ms: int
    status: FolderAccessRequestStatus
    reason: FolderAccessReason
    operation: FolderAccessOperation
    target: FolderAccessTarget
    purpose: str
    selected_path: str | None
    continuation_turn_id: str | None
    source_binding: WorkspaceAccessSourceBinding
    task_input_event_id: str
    latest_input_event_id: str
    blocked_run_id: str
    expected_scope: WorkspaceAccessExpectedScope
    resolved_grants_revision: int | None
    selection_authority: NotRequired[WorkspaceAccessSelectionAuthority | None]
    folder_proposal: NotRequired[FolderProposal]


class WorkspaceAccessAlwaysSelectionAuthority(TypedDict):
    kind: Literal["always"]


class WorkspaceAccessOnceSelectionAuthority(TypedDict):
    kind: Literal["once"]
    runtime_id: str
    authority_epoch: int


WorkspaceAccessSelectionAuthority = (
    WorkspaceAccessAlwaysSelectionAuthority | WorkspaceAccessOnceSelectionAuthority
)


class WorkspacePermissionFolder(TypedDict):
    path: str
    file_access: Literal["read_only", "read_write"]
    commands: bool


class WorkspacePermissionTaskAccess(TypedDict):
    task_turn_id: str
    state: TaskAccessState
    folders: list[WorkspacePermissionFolder]


class WorkspacePermissionRetirement(TypedDict):
    reason: Literal[
        "task_ended", "task_cancelled", "task_replaced", "mode_changed",
        "source_changed", "session_reset", "session_deleted", "restart",
    ]
    active_turn: bool
    active_processes: int
    unknown_processes: int


class WorkspacePermissionProjection(TypedDict):
    version: Literal[1]
    runtime_id: str
    revision: int
    mode: Literal["restricted", "full"]
    state: Literal["active", "retiring"]
    can_enable_full: bool
    full_access_unavailable_reason: PermissionFullUnavailableReason | None
    task_access: WorkspacePermissionTaskAccess | None
    retirement: WorkspacePermissionRetirement | None


class FolderAccessTask(TypedDict):
    turn_id: str
    excerpt: str


class WorkspaceSandboxPayload(TypedDict):
    restrict_to_workspace: bool
    workspace_root: str
    level: str
    enforced: bool
    provider: str
    provider_label: str
    summary: str


class WorkspaceScopePayload(TypedDict):
    project_path: str
    project_name: str
    access_mode: Literal["restricted", "full"]
    effective_access: EffectiveWorkspaceAccess
    restrict_to_workspace: bool
    sandbox_status: WorkspaceSandboxPayload


class FolderAccessRequest(TypedDict):
    version: Literal[1]
    access_request_id: str
    revision: int
    chat_id: str
    task: FolderAccessTask
    blocked_turn_id: str
    tool_call_id: str | None
    created_at_ms: int
    status: FolderAccessRequestStatus
    reason: FolderAccessReason
    operation: FolderAccessOperation
    target: FolderAccessTarget
    folder_proposal: NotRequired[FolderProposal]
    purpose: str
    selected_path: str | None
    selected_access: EffectiveWorkspaceAccess | None
    can_continue: bool
    unavailable_reason: FolderAccessUnavailableReason | None
    continuation_turn_id: str | None
    selected_duration: FolderAccessDuration | None
    task_access_state: TaskAccessState | None


class FolderAccessActionResult(TypedDict):
    workspace_access_request: FolderAccessRequest
    workspace_scope: WorkspaceScopePayload
    workspace_permissions: WorkspacePermissionProjection


class WorkspaceOnceReceipt(TypedDict):
    version: Literal[1]
    duration: Literal["once"]
    runtime_id: str
    permission_revision: int
    access_request_id: str
    request_revision: int
    task_turn_id: str
    submitted_path: str
    canonical_path: str
    file_access: Literal["read_only", "read_write"]
    commands: bool


class WorkspaceOnceActionResult(FolderAccessActionResult):
    once_receipt: WorkspaceOnceReceipt


class WorkspaceModeActionResult(TypedDict):
    workspace_scope: WorkspaceScopePayload
    workspace_permissions: WorkspacePermissionProjection


class WorkspaceModeConflictErrorPayload(TypedDict):
    status: Literal[409]
    message: Literal["chat_running", "native_processes_active", "native_processes_unknown"]


class WorkspaceModeConflict(TypedDict):
    error: WorkspaceModeConflictErrorPayload
    workspace_scope: WorkspaceScopePayload
    workspace_permissions: WorkspacePermissionProjection


class FolderAccessNeedMetadata(TypedDict):
    folder_access: FolderAccessNeed


class FolderAccessEvent(TypedDict):
    workspace_access_request: NotRequired[FolderAccessRequest]


class WorkspaceAccessActionIdentity(TypedDict):
    chat_id: str
    access_request_id: str
    expected_revision: int


class WorkspaceAccessResolvePayload(WorkspaceAccessActionIdentity):
    path: str
    expected_grants_revision: int


class WorkspaceAccessOncePayload(WorkspaceAccessActionIdentity):
    path: str
    file_access: Literal["read_only", "read_write"]
    commands: bool
    expected_grants_revision: int
    expected_runtime_id: str
    expected_permission_revision: int


WorkspaceAccessActionPayload = WorkspaceAccessActionIdentity | WorkspaceAccessResolvePayload

WORKSPACE_ACCESS_CONTINUATION_METADATA_KEY = "_workspace_access_continuation"


class WorkspaceAccessContinuationToken(TypedDict):
    access_request_id: str
    revision: int


_NEED_KEYS = frozenset({
    "version", "reason", "operation", "target", "purpose", "allowed_roots",
    "roots_truncated",
})
_REASONS = frozenset({
    "folder_not_granted", "write_not_granted", "commands_not_granted",
    "folder_selection_required",
})
_OPERATIONS = frozenset({"read", "write", "execute", "discover", "select", "locate"})
_MAX_PATH_CHARS = 2048
_MAX_PURPOSE_CHARS = 240
_MAX_RECORD_BYTES = 16 * 1024
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_UUID_HEX = re.compile(r"^[0-9a-f]{32}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_TURN_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _has_exact_keys(value: object, keys: Collection[str]) -> TypeGuard[dict[str, object]]:
    if type(value) is not dict:
        return False
    data = cast(dict[str, object], value)
    return set(data) == set(keys)


def _valid_absolute_path(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= _MAX_PATH_CHARS
        and "\x00" not in value
        and Path(value).is_absolute()
    )


def _valid_once_input_path(value: object) -> TypeGuard[str]:
    """Accept absolute Once input, including a safely expandable current home path."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_PATH_CHARS
        or "\x00" in value
    ):
        return False
    if value == "~" or value.startswith(("~/", "~\\")):
        try:
            return Path(value).expanduser().is_absolute()
        except (OSError, RuntimeError, ValueError):
            return False
    return _valid_absolute_path(value)


def parse_folder_access_need(raw: object) -> FolderAccessNeed:
    """Validate one exact need envelope at the policy-to-turn boundary."""
    if not _has_exact_keys(raw, _NEED_KEYS) and not _has_exact_keys(raw, _NEED_KEYS | {"folder_proposal"}):
        raise ValueError("invalid folder access need")
    value = raw
    version = value["version"]
    if type(version) is not int or version != 1:
        raise ValueError("invalid folder access need version")
    reason = value["reason"]
    operation = value["operation"]
    if not isinstance(reason, str) or reason not in _REASONS:
        raise ValueError("invalid folder access reason")
    if not isinstance(operation, str) or operation not in _OPERATIONS:
        raise ValueError("invalid folder access operation")

    target_raw = value["target"]
    if not _has_exact_keys(target_raw, {"kind", "path"}):
        raise ValueError("invalid folder access target")
    target_data = target_raw
    kind = target_data.get("kind")
    if kind == "unknown":
        if target_data["path"] is not None:
            raise ValueError("invalid unknown folder target")
        target: FolderAccessTarget = {"kind": "unknown", "path": None}
    elif kind in {"directory", "file"}:
        if not _valid_absolute_path(target_data["path"]):
            raise ValueError("invalid known folder target")
        target = {
            "kind": cast(Literal["directory", "file"], kind),
            "path": target_data["path"],
        }
    else:
        raise ValueError("invalid folder target kind")

    purpose = value["purpose"]
    if (
        not isinstance(purpose, str)
        or not 1 <= len(purpose) <= _MAX_PURPOSE_CHARS
        or _CONTROL_CHARACTERS.search(purpose)
    ):
        raise ValueError("invalid folder access purpose")
    roots = value["allowed_roots"]
    if type(roots) is not list:
        raise ValueError("invalid folder access roots")
    roots_value = cast(list[object], roots)
    if len(roots_value) > 8:
        raise ValueError("invalid folder access roots")
    checked_roots: list[str] = []
    for root in roots_value:
        if not _valid_absolute_path(root) or root in checked_roots:
            raise ValueError("invalid folder access root")
        checked_roots.append(root)
    truncated = value["roots_truncated"]
    if type(truncated) is not bool:
        raise ValueError("invalid folder access roots flag")

    result: FolderAccessNeed = {
        "version": 1,
        "reason": cast(FolderAccessReason, reason),
        "operation": cast(FolderAccessOperation, operation),
        "target": target,
        "purpose": purpose,
        "allowed_roots": checked_roots,
        "roots_truncated": truncated,
    }
    if "folder_proposal" in value:
        result["folder_proposal"] = parse_folder_proposal(value["folder_proposal"])
    return result


def parse_workspace_access_action_payload(
    raw: object,
    *,
    resolve: bool,
) -> WorkspaceAccessActionPayload:
    """Validate one exact local recovery action payload at its HTTP edge."""
    required = (
        {"chat_id", "access_request_id", "expected_revision", "path", "expected_grants_revision"}
        if resolve
        else {"chat_id", "access_request_id", "expected_revision"}
    )
    if not _has_exact_keys(raw, required):
        raise ValueError("invalid workspace access action payload")
    value = raw
    from nanobot.webui.session_identity import is_valid_webui_chat_id

    chat_id = value["chat_id"]
    request_id = value["access_request_id"]
    revision = value["expected_revision"]
    if (
        not isinstance(chat_id, str)
        or not is_valid_webui_chat_id(chat_id)
        or not isinstance(request_id, str)
        or _UUID_HEX.fullmatch(request_id) is None
        or type(revision) is not int
        or revision < 0
    ):
        raise ValueError("invalid workspace access action payload")
    identity: WorkspaceAccessActionIdentity = {
        "chat_id": chat_id,
        "access_request_id": request_id,
        "expected_revision": revision,
    }
    if not resolve:
        return identity
    path = value["path"]
    grants_revision = value["expected_grants_revision"]
    if (
        not isinstance(path, str)
        or not 1 <= len(path) <= _MAX_PATH_CHARS
        or "\x00" in path
        or type(grants_revision) is not int
        or grants_revision < 0
    ):
        raise ValueError("invalid workspace access action payload")
    return {
        **identity,
        "path": path,
        "expected_grants_revision": grants_revision,
    }


def parse_workspace_access_once_payload(raw: object) -> WorkspaceAccessOncePayload:
    """Validate the exact local Once-review payload without granting authority."""
    required = {
        "chat_id", "access_request_id", "expected_revision", "path", "file_access",
        "commands", "expected_grants_revision", "expected_runtime_id",
        "expected_permission_revision",
    }
    if not _has_exact_keys(raw, required):
        raise ValueError("invalid workspace access once payload")
    value = raw
    from nanobot.webui.session_identity import is_valid_webui_chat_id

    chat_id = value["chat_id"]
    request_id = value["access_request_id"]
    revision = value["expected_revision"]
    path = value["path"]
    file_access = value["file_access"]
    commands = value["commands"]
    grants_revision = value["expected_grants_revision"]
    runtime_id = value["expected_runtime_id"]
    permission_revision = value["expected_permission_revision"]
    if (
        not isinstance(chat_id, str)
        or not is_valid_webui_chat_id(chat_id)
        or not isinstance(request_id, str)
        or _UUID_HEX.fullmatch(request_id) is None
        or type(revision) is not int
        or revision < 1
        or not _valid_once_input_path(path)
        or file_access not in {"read_only", "read_write"}
        or type(commands) is not bool
        or type(grants_revision) is not int
        or grants_revision < 0
        or not isinstance(runtime_id, str)
        or _UUID_HEX.fullmatch(runtime_id) is None
        or type(permission_revision) is not int
        or permission_revision < 0
    ):
        raise ValueError("invalid workspace access once payload")
    return {
        "chat_id": chat_id,
        "access_request_id": request_id,
        "expected_revision": revision,
        "path": path,
        "file_access": cast(Literal["read_only", "read_write"], file_access),
        "commands": commands,
        "expected_grants_revision": grants_revision,
        "expected_runtime_id": runtime_id,
        "expected_permission_revision": permission_revision,
    }


def parse_workspace_access_source_binding(raw: object) -> WorkspaceAccessSourceBinding:
    """Validate one source-incarnation binding at its persistence boundary."""
    if not _has_exact_keys(raw, {"store_identity", "incarnation", "generation"}):
        raise ValueError("invalid workspace access source binding")
    store_identity = raw["store_identity"]
    incarnation = raw["incarnation"]
    generation = raw["generation"]
    if (
        not isinstance(store_identity, str)
        or _SHA256_HEX.fullmatch(store_identity) is None
        or not isinstance(incarnation, str)
        or not 1 <= len(incarnation) <= 128
        or _CONTROL_CHARACTERS.search(incarnation)
        or type(generation) is not int
        or generation < 0
    ):
        raise ValueError("invalid workspace access source binding")
    return {
        "store_identity": store_identity,
        "incarnation": incarnation,
        "generation": generation,
    }


def parse_workspace_scope_metadata(raw: object) -> WorkspaceScopeMetadata:
    """Validate the durable project focus shape used by recovery CAS."""
    if not _has_exact_keys(raw, {"project_path", "access_mode"}):
        raise ValueError("invalid workspace scope metadata")
    project_path = raw["project_path"]
    access_mode = raw["access_mode"]
    if (
        not _valid_absolute_path(project_path)
        or not isinstance(access_mode, str)
        or access_mode not in {"restricted", "full"}
    ):
        raise ValueError("invalid workspace scope metadata")
    return {
        "project_path": project_path,
        "access_mode": cast(Literal["restricted", "full"], access_mode),
    }


def parse_workspace_access_record(raw: object) -> WorkspaceAccessRecord:
    """Validate a bounded latest request before using it as durable authority."""
    keys = {
        "version", "access_request_id", "revision", "task", "blocked_turn_id",
        "tool_call_id", "created_at_ms", "status", "reason", "operation", "target",
        "purpose", "selected_path", "continuation_turn_id", "source_binding",
        "task_input_event_id", "latest_input_event_id", "blocked_run_id", "expected_scope",
        "resolved_grants_revision",
    }
    if type(raw) is not dict:
        raise ValueError("invalid workspace access record")
    value = cast(dict[str, object], raw)
    if frozenset(value) not in {
        frozenset(keys), frozenset(keys | {"selection_authority"}),
        frozenset(keys | {"folder_proposal"}),
        frozenset(keys | {"selection_authority", "folder_proposal"}),
    }:
        raise ValueError("invalid workspace access record")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("invalid workspace access record") from None
    if len(encoded) > _MAX_RECORD_BYTES:
        raise ValueError("workspace access record too large")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("invalid workspace access record version")
    request_id = value["access_request_id"]
    if not isinstance(request_id, str) or _UUID_HEX.fullmatch(request_id) is None:
        raise ValueError("invalid workspace access request id")
    revision = value["revision"]
    if type(revision) is not int or revision < 1:
        raise ValueError("invalid workspace access revision")
    created_at_ms = value["created_at_ms"]
    if type(created_at_ms) is not int or created_at_ms < 0:
        raise ValueError("invalid workspace access timestamp")

    task_raw = value["task"]
    if not _has_exact_keys(task_raw, {"turn_id"}):
        raise ValueError("invalid workspace access task")
    task_turn_id = task_raw["turn_id"]
    if not isinstance(task_turn_id, str) or _TURN_ID.fullmatch(task_turn_id) is None:
        raise ValueError("invalid workspace access task turn")
    blocked_turn_id = value["blocked_turn_id"]
    if not isinstance(blocked_turn_id, str) or _TURN_ID.fullmatch(blocked_turn_id) is None:
        raise ValueError("invalid workspace access blocked turn")

    tool_call_id = value["tool_call_id"]
    if tool_call_id is not None and (
        not isinstance(tool_call_id, str) or not 1 <= len(tool_call_id) <= 256
        or _CONTROL_CHARACTERS.search(tool_call_id)
    ):
        raise ValueError("invalid workspace access tool call")
    continuation_turn_id = value["continuation_turn_id"]
    if continuation_turn_id is not None and (
        not isinstance(continuation_turn_id, str)
        or _TURN_ID.fullmatch(continuation_turn_id) is None
    ):
        raise ValueError("invalid workspace access continuation turn")

    status = value["status"]
    if not isinstance(status, str) or status not in {
        "pending", "ready", "continuing", "continued", "cancelled", "invalidated",
    }:
        raise ValueError("invalid workspace access status")
    reason = value["reason"]
    operation = value["operation"]
    if not isinstance(reason, str) or reason not in _REASONS:
        raise ValueError("invalid workspace access reason")
    if not isinstance(operation, str) or operation not in _OPERATIONS:
        raise ValueError("invalid workspace access operation")
    target_raw = value["target"]
    if not _has_exact_keys(target_raw, {"kind", "path"}):
        raise ValueError("invalid workspace access target")
    target_data = target_raw
    target_kind = target_data.get("kind")
    if target_kind == "unknown":
        if target_data["path"] is not None:
            raise ValueError("invalid workspace access target")
        target: FolderAccessTarget = {"kind": "unknown", "path": None}
    elif isinstance(target_kind, str) and target_kind in {"directory", "file"}:
        target_path = target_data.get("path")
        if not _valid_absolute_path(target_path):
            raise ValueError("invalid workspace access target")
        target = {
            "kind": cast(Literal["directory", "file"], target_kind),
            "path": target_path,
        }
    else:
        raise ValueError("invalid workspace access target")

    purpose = value["purpose"]
    if (
        not isinstance(purpose, str) or not 1 <= len(purpose) <= _MAX_PURPOSE_CHARS
        or _CONTROL_CHARACTERS.search(purpose)
    ):
        raise ValueError("invalid workspace access purpose")
    selected_path = value["selected_path"]
    if selected_path is not None and not _valid_absolute_path(selected_path):
        raise ValueError("invalid selected workspace path")

    binding = parse_workspace_access_source_binding(value["source_binding"])

    event_ids: dict[str, str] = {}
    for field in ("task_input_event_id", "latest_input_event_id", "blocked_run_id"):
        identity = value[field]
        if not isinstance(identity, str) or _UUID_HEX.fullmatch(identity) is None:
            raise ValueError("invalid workspace access source event")
        event_ids[field] = identity

    scope = parse_workspace_scope_metadata(value["expected_scope"])

    grants_revision = value["resolved_grants_revision"]
    if grants_revision is not None and (type(grants_revision) is not int or grants_revision < 0):
        raise ValueError("invalid workspace access grant revision")

    selection_authority: WorkspaceAccessSelectionAuthority | None = None
    if "selection_authority" in value:
        selection_authority = parse_workspace_access_selection_authority(
            value["selection_authority"],
        )

    result: WorkspaceAccessRecord = {
        "version": 1,
        "access_request_id": request_id,
        "revision": revision,
        "task": {"turn_id": task_turn_id},
        "blocked_turn_id": blocked_turn_id,
        "tool_call_id": tool_call_id,
        "created_at_ms": created_at_ms,
        "status": cast(FolderAccessRequestStatus, status),
        "reason": cast(FolderAccessReason, reason),
        "operation": cast(FolderAccessOperation, operation),
        "target": target,
        "purpose": purpose,
        "selected_path": selected_path,
        "continuation_turn_id": continuation_turn_id,
        "source_binding": binding,
        "task_input_event_id": event_ids["task_input_event_id"],
        "latest_input_event_id": event_ids["latest_input_event_id"],
        "blocked_run_id": event_ids["blocked_run_id"],
        "expected_scope": scope,
        "resolved_grants_revision": grants_revision,
    }
    if "selection_authority" in value:
        result["selection_authority"] = selection_authority
    if "folder_proposal" in value:
        result["folder_proposal"] = parse_folder_proposal(value["folder_proposal"])
    return result


def parse_workspace_access_selection_authority(
    raw: object,
) -> WorkspaceAccessSelectionAuthority | None:
    """Validate the optional v2 selection marker; it never reconstructs proof."""
    if raw is None:
        return None
    if type(raw) is not dict:
        raise ValueError("invalid workspace access selection authority")
    value = cast(dict[str, object], raw)
    kind = value.get("kind")
    if kind == "always" and set(value) == {"kind"}:
        return {"kind": "always"}
    if kind != "once" or set(value) != {"kind", "runtime_id", "authority_epoch"}:
        raise ValueError("invalid workspace access selection authority")
    runtime_id = value.get("runtime_id")
    authority_epoch = value.get("authority_epoch")
    if (
        not isinstance(runtime_id, str)
        or _UUID_HEX.fullmatch(runtime_id) is None
        or type(authority_epoch) is not int
        or authority_epoch < 1
    ):
        raise ValueError("invalid workspace access selection authority")
    return {
        "kind": "once",
        "runtime_id": runtime_id,
        "authority_epoch": authority_epoch,
    }


def parse_workspace_access_continuation_token(
    value: object,
) -> WorkspaceAccessContinuationToken:
    if not _has_exact_keys(value, {"access_request_id", "revision"}):
        raise ValueError("invalid workspace access continuation token")
    access_request_id = value["access_request_id"]
    revision = value["revision"]
    if (
        not isinstance(access_request_id, str)
        or _UUID_HEX.fullmatch(access_request_id) is None
        or type(revision) is not int
        or revision < 1
    ):
        raise ValueError("invalid workspace access continuation token")
    return {"access_request_id": access_request_id, "revision": revision}


__all__ = [
    "FolderAccessActionResult",
    "FolderAccessEvent",
    "FolderAccessNeed",
    "FolderAccessNeedMetadata",
    "FolderAccessOperation",
    "FolderAccessReason",
    "FolderAccessDuration",
    "FolderAccessRequest",
    "FolderAccessRequestStatus",
    "FolderAccessTarget",
    "FolderAccessTask",
    "FolderAccessUnavailableReason",
    "FolderAccessKnownTarget",
    "FolderAccessUnknownTarget",
    "WorkspaceAccessSourceBinding",
    "WorkspaceAccessExpectedScope",
    "WorkspaceAccessActionIdentity",
    "WorkspaceAccessActionPayload",
    "WorkspaceAccessContinuationToken",
    "WorkspaceAccessResolvePayload",
    "WorkspaceAccessOncePayload",
    "WorkspaceAccessRecord",
    "WorkspaceAccessSelectionAuthority",
    "WorkspaceAccessTaskAnchor",
    "WorkspaceModeActionResult",
    "WorkspaceModeConflict",
    "WorkspaceOnceActionResult",
    "WorkspaceOnceReceipt",
    "WorkspacePermissionProjection",
    "PermissionFullUnavailableReason",
    "TaskAccessState",
    "WorkspaceScopeMetadata",
    "WorkspaceSandboxPayload",
    "WorkspaceScopePayload",
    "parse_folder_access_need",
    "parse_workspace_access_action_payload",
    "parse_workspace_access_once_payload",
    "parse_workspace_access_record",
    "parse_workspace_access_selection_authority",
    "parse_workspace_access_source_binding",
    "parse_workspace_scope_metadata",
    "parse_workspace_access_continuation_token",
    "WORKSPACE_ACCESS_CONTINUATION_METADATA_KEY",
]
