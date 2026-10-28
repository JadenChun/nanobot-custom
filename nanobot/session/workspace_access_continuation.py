"""Private, process-local authority for one canonical workspace Continue turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NoReturn, cast

from nanobot.webui.metadata import WEBUI_TURN_METADATA_KEY

if TYPE_CHECKING:
    from nanobot.bus.events import InboundMessage
    from nanobot.security.desktop_permissions import WorkspaceTaskAccessReference
    from nanobot.workspace_access_recovery import (
        WorkspaceAccessExpectedScope,
        WorkspaceAccessRecord,
        WorkspaceAccessSourceBinding,
    )

_ATTRIBUTE = "_nanobot_workspace_access_continuation"
_TOKEN = object()


class WorkspaceAccessContinuationRejectedError(RuntimeError):
    """A trusted continuation was definitively rejected before admission."""


@dataclass(frozen=True, slots=True, eq=False)
class WorkspaceAccessContinuationReference:
    """Unserializable authority copied only across trusted inbound replacements."""

    session_key: str
    store_identity: str
    incarnation: str
    generation: int
    access_request_id: str
    revision: int
    turn_id: str
    task_turn_id: str
    task_input_event_id: str
    latest_input_event_id: str
    project_path: str
    access_mode: Literal["restricted", "full"]
    selected_path: str
    grants_revision: int
    task_reference: WorkspaceTaskAccessReference | None
    _token: object = field(repr=False)

    def __reduce__(self) -> NoReturn:
        raise TypeError("workspace access continuation references cannot be serialized")

    @property
    def source_binding(self) -> WorkspaceAccessSourceBinding:
        return {
            "store_identity": self.store_identity,
            "incarnation": self.incarnation,
            "generation": self.generation,
        }

    @property
    def expected_scope(self) -> WorkspaceAccessExpectedScope:
        return {"project_path": self.project_path, "access_mode": self.access_mode}


def is_workspace_access_continuation_reference(value: object) -> bool:
    return (
        type(value) is WorkspaceAccessContinuationReference
        and getattr(value, "_token", None) is _TOKEN
    )


def attach_workspace_access_continuation(
    message: InboundMessage,
    record: WorkspaceAccessRecord,
    *,
    task_reference: WorkspaceTaskAccessReference | None = None,
) -> WorkspaceAccessContinuationReference:
    """Attach authority minted from the exact committed ``continuing`` record."""
    scope = record["expected_scope"]
    grants_revision = record["resolved_grants_revision"]
    selected_path = record["selected_path"]
    turn_id = record["continuation_turn_id"]
    if (
        record["status"] != "continuing"
        or turn_id is None
        or selected_path is None
        or type(grants_revision) is not int
    ):
        raise ValueError("workspace access request is not continuable")
    selection = record.get("selection_authority")
    if selection is not None and selection["kind"] == "once":
        if (
            task_reference is None
            or task_reference.session_key != message.session_key
            or task_reference.source_binding != record["source_binding"]
            or task_reference.task_turn_id != record["task"]["turn_id"]
            or task_reference.task_input_event_id != record["task_input_event_id"]
            or selection.get("authority_epoch") != task_reference.epoch
        ):
            raise ValueError("workspace Once authority is not continuable")
    elif task_reference is not None:
        raise ValueError("workspace Always continuation cannot carry task authority")
    binding = record["source_binding"]
    reference = WorkspaceAccessContinuationReference(
        session_key=message.session_key,
        store_identity=binding["store_identity"],
        incarnation=binding["incarnation"],
        generation=binding["generation"],
        access_request_id=record["access_request_id"],
        revision=record["revision"],
        turn_id=turn_id,
        task_turn_id=record["task"]["turn_id"],
        task_input_event_id=record["task_input_event_id"],
        latest_input_event_id=record["latest_input_event_id"],
        project_path=scope["project_path"],
        access_mode=scope["access_mode"],
        selected_path=selected_path,
        grants_revision=grants_revision,
        task_reference=task_reference,
        _token=_TOKEN,
    )
    setattr(message, _ATTRIBUTE, reference)
    return reference


def workspace_access_continuation(
    message: InboundMessage,
    *,
    session_key: str | None = None,
) -> WorkspaceAccessContinuationReference | None:
    """Read a valid object-bound reference; serialized metadata is never authority."""
    raw_reference = getattr(message, _ATTRIBUTE, None)
    if not is_workspace_access_continuation_reference(raw_reference):
        return None
    reference = cast(WorkspaceAccessContinuationReference, raw_reference)
    if (
        not message.is_user_input
        or message.channel != "websocket"
        or message.sender_id != "webui"
        or message.content != "Continue task"
        or message.media
        or not message.require_existing_session
        or message.session_key != reference.session_key
        or (session_key is not None and session_key != reference.session_key)
        or message.metadata.get("webui") is not True
        or message.metadata.get(WEBUI_TURN_METADATA_KEY) != reference.turn_id
        or message.metadata.get("workspace_scope") != {
            "project_path": reference.project_path,
            "access_mode": reference.access_mode,
        }
    ):
        return None
    return reference


def copy_workspace_access_continuation(
    source: InboundMessage,
    destination: InboundMessage,
) -> None:
    """Preserve only the exact private reference across trusted message copies."""
    reference = workspace_access_continuation(source)
    if reference is not None and destination.session_key == source.session_key:
        setattr(destination, _ATTRIBUTE, reference)


__all__ = [
    "WorkspaceAccessContinuationRejectedError",
    "WorkspaceAccessContinuationReference",
    "attach_workspace_access_continuation",
    "copy_workspace_access_continuation",
    "is_workspace_access_continuation_reference",
    "workspace_access_continuation",
]
