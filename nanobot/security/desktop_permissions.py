"""Process-local authority for explicit desktop workspace permissions.

Public projections in this module describe authority; only the opaque reference
objects returned by the owner can authorize work. References are intentionally
bound to their Python object identity and cannot be serialized.
"""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, NoReturn, TypedDict, cast

from nanobot.security.folder_grants import FolderFileAccess

if TYPE_CHECKING:
    from nanobot.security.native_admission import NativeOwnerTag

TaskAccessState = Literal["active", "retiring", "expired"]
PermissionMode = Literal["restricted", "full"]
PermissionState = Literal["active", "retiring"]
PermissionRetirementReason = Literal[
    "task_ended", "task_cancelled", "task_replaced", "mode_changed",
    "source_changed", "session_reset", "session_deleted", "restart",
]
PermissionFullUnavailableReason = Literal[
    "controls_unavailable", "session_required", "configured_sandbox",
    "supervisor_unavailable", "permission_state_unavailable", "permission_retiring",
]

_TOKEN = object()
_RUNTIME_ID = re.compile(r"^[0-9a-f]{32}$")
_MAX_ID_LENGTH = 128
_MAX_SESSION_KEY_LENGTH = 256


class PermissionSourceBinding(TypedDict):
    store_identity: str
    incarnation: str
    generation: int


class TaskAccessFolder(TypedDict):
    path: str
    file_access: FolderFileAccess
    commands: bool


class TaskAccessProjection(TypedDict):
    task_turn_id: str
    state: TaskAccessState
    folders: list[TaskAccessFolder]


class PermissionRetirementProjection(TypedDict):
    reason: PermissionRetirementReason
    active_turn: bool
    active_processes: int
    unknown_processes: int


class WorkspacePermissionProjection(TypedDict):
    version: Literal[1]
    runtime_id: str
    revision: int
    mode: PermissionMode
    state: PermissionState
    can_enable_full: bool
    full_access_unavailable_reason: PermissionFullUnavailableReason | None
    task_access: TaskAccessProjection | None
    retirement: PermissionRetirementProjection | None


class WorkspacePermissionUnavailableError(RuntimeError):
    """An explicit Full action cannot safely be enabled for this chat."""

    def __init__(self, reason: PermissionFullUnavailableReason) -> None:
        super().__init__(reason)
        self.reason = reason


class WorkspacePermissionConflictError(RuntimeError):
    """A fenced permission still has owned work that is not proven quiescent."""

    message: Literal[
        "chat_running", "native_processes_active", "native_processes_unknown",
    ]

    def __init__(self, message: Literal[
        "chat_running", "native_processes_active", "native_processes_unknown",
    ]) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True, slots=True)
class _Root:
    path: str
    file_access: FolderFileAccess
    commands: bool

    def projection(self) -> TaskAccessFolder:
        return {
            "path": self.path,
            "file_access": self.file_access,
            "commands": self.commands,
        }


@dataclass(frozen=True, slots=True, eq=False)
class WorkspaceTaskAccessReference:
    """Unserializable permission for one exact original primary task."""

    session_key: str
    source_binding: PermissionSourceBinding
    task_turn_id: str
    task_input_event_id: str
    epoch: int
    roots: tuple[_Root, ...]
    _token: object = field(repr=False)

    def __reduce__(self) -> NoReturn:
        raise TypeError("workspace task permission references cannot be serialized")

    def is_owner_issued(self) -> bool:
        return self._token is _TOKEN


@dataclass(frozen=True, slots=True, eq=False)
class WorkspaceFullAccessReference:
    """Unserializable Full permission for one canonical persistent chat source."""

    session_key: str
    source_binding: PermissionSourceBinding
    epoch: int
    _token: object = field(repr=False)

    def __reduce__(self) -> NoReturn:
        raise TypeError("workspace Full permission references cannot be serialized")

    def is_owner_issued(self) -> bool:
        return self._token is _TOKEN


WorkspacePermissionReference = WorkspaceTaskAccessReference | WorkspaceFullAccessReference


_MESSAGE_PERMISSION_ATTRIBUTE = "_nanobot_workspace_permission_reference"


def attach_full_permission_to_message(message: object, reference: object) -> None:
    """Attach a private Full proof to a transport-created inbound message."""
    if type(reference) is not WorkspaceFullAccessReference or not reference.is_owner_issued():
        raise TypeError("invalid workspace Full permission reference")
    setattr(message, _MESSAGE_PERMISSION_ATTRIBUTE, reference)


def full_permission_from_message(message: object) -> WorkspaceFullAccessReference | None:
    """Read only the nonserialized reference attached by a trusted local adapter."""
    reference = getattr(message, _MESSAGE_PERMISSION_ATTRIBUTE, None)
    if type(reference) is not WorkspaceFullAccessReference or not reference.is_owner_issued():
        return None
    return reference


def copy_full_permission_message_reference(source: object, destination: object) -> None:
    reference = full_permission_from_message(source)
    if reference is None:
        return
    if getattr(source, "session_key", None) != getattr(destination, "session_key", None):
        return
    attach_full_permission_to_message(destination, reference)


@dataclass(frozen=True, slots=True)
class _TaskIdentity:
    session_key: str
    store_identity: str
    incarnation: str
    generation: int
    task_turn_id: str
    task_input_event_id: str


@dataclass(frozen=True, slots=True)
class _SourceIdentity:
    session_key: str
    store_identity: str
    incarnation: str
    generation: int


@dataclass
class _Retirement:
    reason: PermissionRetirementReason
    active_turn: bool = False
    active_processes: int = 0
    unknown_processes: int = 0
    private_reference: object | None = None

    def projection(self) -> PermissionRetirementProjection:
        return {
            "reason": self.reason,
            "active_turn": self.active_turn,
            "active_processes": self.active_processes,
            "unknown_processes": self.unknown_processes,
        }


class DesktopPermissionOwner:
    """One lock-protected volatile authority owner per shared SessionManager."""

    def __init__(self) -> None:
        self._runtime_id = uuid.uuid4().hex
        self._revision = 0
        self._epoch = 0
        self._lock = threading.RLock()
        self._tasks: dict[_TaskIdentity, WorkspaceTaskAccessReference] = {}
        self._task_retirements: dict[_TaskIdentity, _Retirement] = {}
        self._full: dict[_SourceIdentity, WorkspaceFullAccessReference] = {}
        self._full_retirements: dict[_SourceIdentity, _Retirement] = {}
        self._native_stdin_closers: dict[object, dict[object, Callable[[], bool]]] = {}
        self._native_process_counts: dict[object, Callable[[], tuple[int, int]]] = {}
        self._full_residual_counts: dict[_SourceIdentity, tuple[int, int]] = {}
        self._task_residual_counts: dict[_TaskIdentity, tuple[int, int]] = {}
        self._task_source_residual_counts: dict[_SourceIdentity, tuple[int, int]] = {}
        self._last_observed_source: dict[str, _SourceIdentity] = {}

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def commit_once(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        task_turn_id: str,
        task_input_event_id: str,
        path: str | Path,
        file_access: FolderFileAccess,
        commands: bool,
        persist: Callable[[int], None],
        process_counts: Callable[[], tuple[int, int]] | None = None,
    ) -> WorkspaceTaskAccessReference:
        """Persist the selection under the authority lock, then activate it.

        ``persist`` receives the epoch that must be written into the durable
        selection marker. If that compare-and-swap raises, no private reference
        is published.
        """
        identity = _task_identity(
            session_key, source_binding, task_turn_id, task_input_event_id,
        )
        root = _validated_root(path, file_access, commands)
        with self._lock:
            current = self._tasks.get(identity)
            roots = list(current.roots) if current is not None else []
            existing = next((item for item in roots if item.path == root.path), None)
            if existing is not None and existing == root:
                assert current is not None
                persist(current.epoch)
                return current
            retirement = self._task_retirements.get(identity)
            residual = self._task_residual_counts.get(identity, (0, 0))
            previous_reference = current or (
                retirement.private_reference
                if retirement is not None
                and type(retirement.private_reference) is WorkspaceTaskAccessReference
                else None
            )
            if process_counts is not None:
                try:
                    active_processes, unknown_processes = process_counts()
                except Exception:
                    active_processes, unknown_processes = 0, 1
            else:
                active_processes, unknown_processes = 0, 0
            if previous_reference is not None:
                owned_active, owned_unknown = self._counts_for_reference_unlocked(previous_reference)
                active_processes = max(active_processes, owned_active)
                unknown_processes = max(unknown_processes, owned_unknown)
            active_processes = max(active_processes, residual[0])
            unknown_processes = max(unknown_processes, residual[1])
            if retirement is not None:
                if process_counts is None:
                    active_processes = max(active_processes, retirement.active_processes)
                    unknown_processes = max(unknown_processes, retirement.unknown_processes)
                if (
                    retirement.active_processes != active_processes
                    or retirement.unknown_processes != unknown_processes
                ):
                    retirement.active_processes = active_processes
                    retirement.unknown_processes = unknown_processes
                    self._changed()
                if retirement.active_turn:
                    raise WorkspacePermissionConflictError("chat_running")
            if unknown_processes:
                raise WorkspacePermissionConflictError("native_processes_unknown")
            if active_processes:
                raise WorkspacePermissionConflictError("native_processes_active")
            roots = [item for item in roots if item.path != root.path]
            roots.append(root)
            roots.sort(key=lambda item: item.path)
            epoch = self._next_epoch()
            persist(epoch)
            reference = WorkspaceTaskAccessReference(
                session_key=identity.session_key,
                source_binding=_copy_binding(source_binding),
                task_turn_id=identity.task_turn_id,
                task_input_event_id=identity.task_input_event_id,
                epoch=epoch,
                roots=tuple(roots),
                _token=_TOKEN,
            )
            self._tasks[identity] = reference
            if previous_reference is not None:
                self._native_stdin_closers.pop(previous_reference, None)
                self._native_process_counts.pop(previous_reference, None)
            self._task_retirements.pop(identity, None)
            self._task_residual_counts.pop(identity, None)
            self._changed()
            return reference

    def activate_once(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        task_turn_id: str,
        task_input_event_id: str,
        path: str | Path,
        file_access: FolderFileAccess,
        commands: bool,
    ) -> WorkspaceTaskAccessReference:
        """Activate a task root for internal callers that already committed CAS."""
        return self.commit_once(
            session_key=session_key,
            source_binding=source_binding,
            task_turn_id=task_turn_id,
            task_input_event_id=task_input_event_id,
            path=path,
            file_access=file_access,
            commands=commands,
            persist=lambda _epoch: None,
        )

    def task_reference(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        task_turn_id: str,
        task_input_event_id: str,
    ) -> WorkspaceTaskAccessReference | None:
        identity = _task_identity(
            session_key, source_binding, task_turn_id, task_input_event_id,
        )
        with self._lock:
            return self._tasks.get(identity)

    def authorizes_once(
        self,
        reference: object,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        task_turn_id: str,
        task_input_event_id: str,
    ) -> bool:
        if type(reference) is not WorkspaceTaskAccessReference:
            return False
        typed = reference
        if not typed.is_owner_issued():
            return False
        try:
            identity = _task_identity(
                session_key, source_binding, task_turn_id, task_input_event_id,
            )
        except (TypeError, ValueError):
            return False
        with self._lock:
            return (
                self._tasks.get(identity) is typed
                and typed.session_key == identity.session_key
                and typed.source_binding == _binding_from_identity(identity)
                and typed.task_turn_id == identity.task_turn_id
                and typed.task_input_event_id == identity.task_input_event_id
                and typed.epoch > 0
                and all(_root_still_canonical(root.path) for root in typed.roots)
            )

    def is_current_task_reference(self, reference: object) -> bool:
        if type(reference) is not WorkspaceTaskAccessReference:
            return False
        typed = reference
        if not typed.is_owner_issued():
            return False
        try:
            identity = _task_identity(
                typed.session_key,
                typed.source_binding,
                typed.task_turn_id,
                typed.task_input_event_id,
            )
        except (TypeError, ValueError):
            return False
        return self.authorizes_once(
            typed,
            session_key=identity.session_key,
            source_binding=_binding_from_identity(identity),
            task_turn_id=identity.task_turn_id,
            task_input_event_id=identity.task_input_event_id,
        )

    def once_roots(
        self,
        reference: WorkspaceTaskAccessReference,
    ) -> tuple[tuple[str, FolderFileAccess, bool], ...]:
        if type(reference) is not WorkspaceTaskAccessReference or not reference.is_owner_issued():
            return ()
        if not self.authorizes_once(
            reference,
            session_key=reference.session_key,
            source_binding=reference.source_binding,
            task_turn_id=reference.task_turn_id,
            task_input_event_id=reference.task_input_event_id,
        ):
            return ()
        return tuple((root.path, root.file_access, root.commands) for root in reference.roots)

    def enable_full(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        persistent: bool,
        existing: bool,
        local: bool,
        idle: bool,
        configured_sandbox: bool,
        supervisor_available: bool,
        stores_healthy: bool,
        persist: Callable[[int], None] | None = None,
        process_counts: Callable[[], tuple[int, int]] | None = None,
    ) -> WorkspaceFullAccessReference:
        source = _source_identity(session_key, source_binding)
        if not persistent or not existing:
            raise WorkspacePermissionUnavailableError("session_required")
        if not local:
            raise WorkspacePermissionUnavailableError("session_required")
        if configured_sandbox:
            raise WorkspacePermissionUnavailableError("configured_sandbox")
        if not supervisor_available:
            raise WorkspacePermissionUnavailableError("supervisor_unavailable")
        if not stores_healthy:
            raise WorkspacePermissionUnavailableError("permission_state_unavailable")
        if not idle:
            raise WorkspacePermissionConflictError("chat_running")
        with self._lock:
            if (
                source in self._full_retirements
                or self._full_residual_counts.get(source, (0, 0)) != (0, 0)
            ):
                raise WorkspacePermissionUnavailableError("permission_retiring")
            current = self._full.get(source)
            if current is not None:
                return current
            if process_counts is not None:
                try:
                    active_processes, unknown_processes = process_counts()
                except Exception:
                    active_processes, unknown_processes = 0, 1
                owned_active = 0
                owned_unknown = 0
                for identity, task_reference in self._tasks.items():
                    if _source_for_task(identity) == source:
                        active, unknown = self._counts_for_reference_unlocked(task_reference)
                        owned_active += active
                        owned_unknown += unknown
                active_processes = max(active_processes, owned_active)
                unknown_processes = max(unknown_processes, owned_unknown)
                if unknown_processes:
                    raise WorkspacePermissionConflictError("native_processes_unknown")
                if active_processes:
                    raise WorkspacePermissionConflictError("native_processes_active")
            else:
                active_processes = 0
                unknown_processes = 0
                for identity, task_reference in self._tasks.items():
                    if _source_for_task(identity) == source:
                        active, unknown = self._counts_for_reference_unlocked(task_reference)
                        active_processes += active
                        unknown_processes += unknown
                if unknown_processes:
                    raise WorkspacePermissionConflictError("native_processes_unknown")
                if active_processes:
                    raise WorkspacePermissionConflictError("native_processes_active")
            for identity, reference in tuple(self._tasks.items()):
                if _source_for_task(identity) == source:
                    self._close_native_stdin_unlocked(reference)
            epoch = self._next_epoch()
            if persist is not None:
                persist(epoch)
            self._expire_source_tasks(source)
            reference = WorkspaceFullAccessReference(
                session_key=source.session_key,
                source_binding=_binding_from_source(source),
                epoch=epoch,
                _token=_TOKEN,
            )
            self._full[source] = reference
            self._changed()
            return reference

    def full_reference(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
    ) -> WorkspaceFullAccessReference | None:
        try:
            source = _source_identity(session_key, source_binding)
        except (TypeError, ValueError):
            return None
        with self._lock:
            return self._full.get(source)

    def authorizes_full(
        self,
        reference: object,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
    ) -> bool:
        if type(reference) is not WorkspaceFullAccessReference:
            return False
        typed = reference
        if not typed.is_owner_issued():
            return False
        try:
            source = _source_identity(session_key, source_binding)
        except (TypeError, ValueError):
            return False
        with self._lock:
            return self._full.get(source) is typed and typed.epoch > 0

    def with_current_reference(
        self,
        reference: object,
        operation: Callable[[], None],
    ) -> bool:
        """Run a launch edge under the owner lock only while its proof is live."""
        with self._lock:
            if not (
                self.is_current_task_reference(reference)
                or self.is_current_full_reference(reference)
            ):
                return False
            operation()
            return True

    def native_owner_tag(self, reference: object) -> NativeOwnerTag | None:
        """Create a durable accounting label from a live private reference."""
        with self._lock:
            if type(reference) is WorkspaceTaskAccessReference:
                if not self.is_current_task_reference(reference):
                    return None
                binding = reference.source_binding
                return {
                    "kind": "task_once",
                    "runtime_id": self._runtime_id,
                    "owner_id": reference.task_turn_id,
                    "epoch": reference.epoch,
                    "session_key": reference.session_key,
                    "store_identity": binding["store_identity"],
                    "incarnation": binding["incarnation"],
                    "generation": binding["generation"],
                    "task_input_event_id": reference.task_input_event_id,
                }
            if type(reference) is WorkspaceFullAccessReference:
                if not self.is_current_full_reference(reference):
                    return None
                binding = reference.source_binding
                return {
                    "kind": "chat_full",
                    "runtime_id": self._runtime_id,
                    "owner_id": reference.session_key,
                    "epoch": reference.epoch,
                    "session_key": reference.session_key,
                    "store_identity": binding["store_identity"],
                    "incarnation": binding["incarnation"],
                    "generation": binding["generation"],
                    "task_input_event_id": None,
                }
        return None

    def is_current_full_reference(self, reference: object) -> bool:
        if type(reference) is not WorkspaceFullAccessReference:
            return False
        typed = reference
        if getattr(typed, "_token", None) is not _TOKEN:
            return False
        return self.authorizes_full(
            typed,
            session_key=typed.session_key,
            source_binding=typed.source_binding,
        )

    def disable_full(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        active_turn: bool,
        active_processes: int,
        unknown_processes: int,
        process_counts: Callable[[], tuple[int, int]] | None = None,
        reason: PermissionRetirementReason = "mode_changed",
    ) -> bool:
        """Fence Full; return True only when this explicit call completes it."""
        source = _source_identity(session_key, source_binding)
        active_processes = _count(active_processes)
        unknown_processes = _count(unknown_processes)
        with self._lock:
            reference = self._full.get(source)
            retirement = self._full_retirements.get(source)
            if reference is None and retirement is None and process_counts is None:
                return True
            close_failures = 0
            if retirement is None:
                reference = self._full.pop(source, None)
                retirement = _Retirement(reason, private_reference=reference)
                self._full_retirements[source] = retirement
                if reference is not None:
                    close_failures = self._close_native_stdin_unlocked(reference)
                self._changed()
            else:
                reference = retirement.private_reference
            if process_counts is not None:
                try:
                    active_processes, unknown_processes = process_counts()
                except Exception:
                    active_processes, unknown_processes = 0, 1
                active_processes = _count(active_processes)
                unknown_processes = _count(unknown_processes)
                if reference is not None:
                    owned_active, owned_unknown = self._counts_for_reference_unlocked(reference)
                    active_processes = max(active_processes, owned_active)
                    unknown_processes = max(unknown_processes, owned_unknown)
            elif reference is not None:
                owned_active, owned_unknown = self._counts_for_reference_unlocked(reference)
                active_processes = max(active_processes, owned_active)
                unknown_processes = max(unknown_processes, owned_unknown)
            unknown_processes = _count(unknown_processes + close_failures)
            retirement.active_turn = bool(active_turn)
            retirement.active_processes = active_processes
            retirement.unknown_processes = unknown_processes
            if unknown_processes:
                raise WorkspacePermissionConflictError("native_processes_unknown")
            if active_processes:
                raise WorkspacePermissionConflictError("native_processes_active")
            if active_turn:
                raise WorkspacePermissionConflictError("chat_running")
            self._full_retirements.pop(source, None)
            self._full_residual_counts.pop(source, None)
            if reference is not None:
                self._native_process_counts.pop(reference, None)
            self._changed()
            return True

    def retire_once(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        task_turn_id: str,
        task_input_event_id: str,
        reason: PermissionRetirementReason,
        active_turn: bool,
        active_processes: int,
        unknown_processes: int,
        process_counts: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        identity = _task_identity(
            session_key, source_binding, task_turn_id, task_input_event_id,
        )
        with self._lock:
            reference = self._tasks.pop(identity, None)
            retirement = self._task_retirements.get(identity)
            if reference is None and retirement is not None:
                reference = retirement.private_reference
            close_failures = 0
            if reference is not None and identity not in self._task_retirements:
                close_failures = self._close_native_stdin_unlocked(reference)
            if process_counts is not None:
                try:
                    active_processes, unknown_processes = process_counts()
                except Exception:
                    active_processes, unknown_processes = 0, 1
                active_processes = _count(active_processes)
                unknown_processes = _count(unknown_processes)
                if reference is not None:
                    owned_active, owned_unknown = self._counts_for_reference_unlocked(reference)
                    active_processes = max(active_processes, owned_active)
                    unknown_processes = max(unknown_processes, owned_unknown)
            elif reference is not None:
                owned_active, owned_unknown = self._counts_for_reference_unlocked(reference)
                active_processes = max(active_processes, owned_active)
                unknown_processes = max(unknown_processes, owned_unknown)
            unknown_processes = _count(unknown_processes + close_failures)
            retirement = _Retirement(
                reason,
                bool(active_turn),
                _count(active_processes),
                _count(unknown_processes),
                reference,
            )
            if reference is not None and process_counts is not None:
                self._native_process_counts[reference] = process_counts
            if retirement.active_turn or retirement.active_processes or retirement.unknown_processes:
                self._task_retirements[identity] = retirement
            else:
                self._task_retirements.pop(identity, None)
                if reference is not None:
                    self._native_process_counts.pop(reference, None)
            self._changed()

    def retire_task_reference(
        self,
        reference: WorkspaceTaskAccessReference,
        *,
        reason: PermissionRetirementReason,
        active_turn: bool,
    ) -> tuple[int, int]:
        """Fence one owner-issued task proof and return its current tree counts."""
        if type(reference) is not WorkspaceTaskAccessReference or not reference.is_owner_issued():
            return 0, 1
        identity = _task_identity(
            reference.session_key,
            reference.source_binding,
            reference.task_turn_id,
            reference.task_input_event_id,
        )
        with self._lock:
            current = self._tasks.get(identity)
            retirement = self._task_retirements.get(identity)
            if current is not reference and (
                retirement is None or retirement.private_reference is not reference
            ):
                return 0, 0
        self.retire_once(
            session_key=reference.session_key,
            source_binding=reference.source_binding,
            task_turn_id=reference.task_turn_id,
            task_input_event_id=reference.task_input_event_id,
            reason=reason,
            active_turn=active_turn,
            active_processes=0,
            unknown_processes=0,
        )
        projection = self.projection(
            session_key=reference.session_key,
            source_binding=reference.source_binding,
            task_turn_id=reference.task_turn_id,
            task_input_event_id=reference.task_input_event_id,
        )
        retirement_projection = projection["retirement"]
        if projection["task_access"] is not None and (
            projection["task_access"]["state"] == "retiring"
        ) and retirement_projection is not None:
            return (
                retirement_projection["active_processes"],
                retirement_projection["unknown_processes"],
            )
        return 0, 0

    def retire_tasks_for_source(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        reason: PermissionRetirementReason,
        active_turn: bool,
    ) -> tuple[int, int]:
        """Fence every live Once reference for one source lifetime."""
        active_processes = 0
        unknown_processes = 0
        source = _source_identity(session_key, source_binding)
        references = self.task_references_for_source(
            session_key=session_key,
            source_binding=source_binding,
        )
        for reference in references:
            self.retire_task_reference(reference, reason=reason, active_turn=active_turn)
        with self._lock:
            for identity, retirement in tuple(self._task_retirements.items()):
                if _source_for_task(identity) != source:
                    continue
                previous = (
                    retirement.reason,
                    retirement.active_turn,
                    retirement.active_processes,
                    retirement.unknown_processes,
                )
                retirement.active_turn = bool(active_turn)
                self._refresh_retirement_counts_unlocked(retirement)
                active_processes += retirement.active_processes
                unknown_processes += retirement.unknown_processes
                if (
                    not retirement.active_turn
                    and not retirement.active_processes
                    and not retirement.unknown_processes
                ):
                    self._task_retirements.pop(identity, None)
                    if retirement.private_reference is not None:
                        self._native_process_counts.pop(retirement.private_reference, None)
                else:
                    retirement.reason = reason
                current = (
                    retirement.reason,
                    retirement.active_turn,
                    retirement.active_processes,
                    retirement.unknown_processes,
                )
                if current != previous:
                    self._changed()
        return active_processes, unknown_processes

    def retire_source_for_lifecycle(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        reason: Literal["session_reset", "session_deleted", "source_changed"],
        task_process_counts: Callable[[], tuple[int, int]] | None = None,
        full_process_counts: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        """Fence chat-scoped authority and require exact tree reconciliation."""
        active, unknown = self.retire_tasks_for_source(
            session_key=session_key,
            source_binding=source_binding,
            reason=reason,
            active_turn=False,
        )
        if task_process_counts is not None:
            try:
                observed_active, observed_unknown = task_process_counts()
            except Exception:
                observed_active, observed_unknown = 0, 1
            active = max(active, _count(observed_active))
            unknown = max(unknown, _count(observed_unknown))
        if unknown:
            raise WorkspacePermissionConflictError("native_processes_unknown")
        if active:
            raise WorkspacePermissionConflictError("native_processes_active")
        source = _source_identity(session_key, source_binding)
        with self._lock:
            has_full_authority = (
                source in self._full or source in self._full_retirements
            )
        full_residual = (0, 0)
        if full_process_counts is not None:
            try:
                full_residual = full_process_counts()
            except Exception:
                full_residual = (0, 1)
        if not has_full_authority and full_residual == (0, 0):
            return
        try:
            self.disable_full(
                session_key=session_key,
                source_binding=source_binding,
                active_turn=False,
                active_processes=0,
                unknown_processes=0,
                process_counts=(lambda: full_residual) if full_process_counts is not None else None,
                reason=reason,
            )
        except WorkspacePermissionConflictError:
            raise

    def observe_full_residual_counts(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        active_processes: int,
        unknown_processes: int,
    ) -> None:
        """Project restart residuals without adopting their durable owner tags."""
        source = _source_identity(session_key, source_binding)
        counts = (_count(active_processes), _count(unknown_processes))
        with self._lock:
            old = self._full_residual_counts.get(source, (0, 0))
            if counts == (0, 0):
                if self._full_residual_counts.pop(source, None) is not None:
                    self._changed()
            elif old != counts:
                self._full_residual_counts[source] = counts
                self._changed()

    def observe_task_residual_counts(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        task_turn_id: str,
        task_input_event_id: str,
        active_processes: int,
        unknown_processes: int,
    ) -> None:
        """Project an exact old Once owner as residual accounting only."""
        identity = _task_identity(
            session_key, source_binding, task_turn_id, task_input_event_id,
        )
        counts = (_count(active_processes), _count(unknown_processes))
        with self._lock:
            old = self._task_residual_counts.get(identity, (0, 0))
            if counts == (0, 0):
                if self._task_residual_counts.pop(identity, None) is not None:
                    self._changed()
            elif old != counts:
                self._task_residual_counts[identity] = counts
                self._changed()

    def observe_task_source_residual_counts(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        active_processes: int,
        unknown_processes: int,
    ) -> None:
        """Project old-generation Once process counts without recreating task authority."""
        source = _source_identity(session_key, source_binding)
        counts = (_count(active_processes), _count(unknown_processes))
        with self._lock:
            old = self._task_source_residual_counts.get(source, (0, 0))
            if counts == (0, 0):
                if self._task_source_residual_counts.pop(source, None) is not None:
                    self._changed()
            elif old != counts:
                self._task_source_residual_counts[source] = counts
                self._changed()

    def update_retirement_counts(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        active_turn: bool,
        active_processes: int,
        unknown_processes: int,
    ) -> None:
        source = _source_identity(session_key, source_binding)
        with self._lock:
            retirement = self._full_retirements.get(source)
            if retirement is not None:
                retirement.active_turn = bool(active_turn)
                retirement.active_processes = _count(active_processes)
                retirement.unknown_processes = _count(unknown_processes)
                self._changed()

    def observe_full_retirement_activity(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        active_turn: bool,
    ) -> bool:
        """Refresh activity only on an existing Full retirement for this source."""
        source = _source_identity(session_key, source_binding)
        with self._lock:
            retirement = self._full_retirements.get(source)
            if retirement is None:
                return False
            if retirement.active_turn != bool(active_turn):
                retirement.active_turn = bool(active_turn)
                self._changed()
            return True

    def projection(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
        task_turn_id: str | None = None,
        task_input_event_id: str | None = None,
        can_enable_full: bool = False,
        full_access_unavailable_reason: PermissionFullUnavailableReason | None = "controls_unavailable",
    ) -> WorkspacePermissionProjection:
        source = _source_identity(session_key, source_binding)
        task_projection: TaskAccessProjection | None = None
        retirement_projection: PermissionRetirementProjection | None = None
        with self._lock:
            previous_source = self._last_observed_source.get(session_key)
            if previous_source is None:
                self._last_observed_source[session_key] = source
            elif previous_source != source:
                self._last_observed_source[session_key] = source
                self._changed()
            full = self._full.get(source)
            full_retirement = self._full_retirements.get(source)
            if full_retirement is not None:
                if self._refresh_retirement_counts_unlocked(full_retirement):
                    self._changed()
                mode: PermissionMode = "full"
                state: PermissionState = "retiring"
                retirement_projection = full_retirement.projection()
                can_enable_full = False
                full_access_unavailable_reason = "permission_retiring"
            elif full is not None:
                mode = "full"
                state = "active"
                can_enable_full = False
                full_access_unavailable_reason = None
            elif self._full_residual_counts.get(source, (0, 0)) != (0, 0):
                active_processes, unknown_processes = self._full_residual_counts[source]
                mode = "full"
                state = "retiring"
                can_enable_full = False
                full_access_unavailable_reason = "permission_retiring"
                retirement_projection = {
                    "reason": "restart",
                    "active_turn": False,
                    "active_processes": active_processes,
                    "unknown_processes": unknown_processes,
                }
            elif self._task_source_residual_counts.get(source, (0, 0)) != (0, 0):
                active_processes, unknown_processes = self._task_source_residual_counts[source]
                mode = "restricted"
                state = "retiring"
                can_enable_full = False
                full_access_unavailable_reason = "permission_retiring"
                retirement_projection = {
                    "reason": "restart",
                    "active_turn": False,
                    "active_processes": active_processes,
                    "unknown_processes": unknown_processes,
                }
            else:
                mode = "restricted"
                state = "active"
                if task_turn_id is not None and task_input_event_id is not None:
                    identity = _task_identity(
                        session_key, source_binding, task_turn_id, task_input_event_id,
                    )
                    task = self._tasks.get(identity)
                    if task is not None:
                        task_projection = {
                            "task_turn_id": task.task_turn_id,
                            "state": "active",
                            "folders": [root.projection() for root in task.roots],
                        }
                    else:
                        retiring = self._task_retirements.get(identity)
                        if retiring is not None:
                            changed = self._refresh_retirement_counts_unlocked(retiring)
                            if (
                                not retiring.active_turn
                                and not retiring.active_processes
                                and not retiring.unknown_processes
                            ):
                                self._task_retirements.pop(identity, None)
                                if retiring.private_reference is not None:
                                    self._native_process_counts.pop(
                                        retiring.private_reference,
                                        None,
                                    )
                                retiring = None
                                changed = True
                            if changed:
                                self._changed()
                        if retiring is not None:
                            task_projection = {
                                "task_turn_id": identity.task_turn_id,
                                "state": "retiring",
                                "folders": [],
                            }
                            retirement_projection = retiring.projection()
                        elif self._task_residual_counts.get(identity, (0, 0)) != (0, 0):
                            active_processes, unknown_processes = self._task_residual_counts[identity]
                            task_projection = {
                                "task_turn_id": identity.task_turn_id,
                                "state": "retiring",
                                "folders": [],
                            }
                            retirement_projection = {
                                "reason": "restart",
                                "active_turn": False,
                                "active_processes": active_processes,
                                "unknown_processes": unknown_processes,
                            }
                        else:
                            task_projection = {
                                "task_turn_id": identity.task_turn_id,
                                "state": "expired",
                                "folders": [],
                            }
            return {
                "version": 1,
                "runtime_id": self._runtime_id,
                "revision": self._revision,
                "mode": mode,
                "state": state,
                "can_enable_full": bool(can_enable_full),
                "full_access_unavailable_reason": full_access_unavailable_reason,
                "task_access": task_projection,
                "retirement": retirement_projection,
            }

    def restart(self) -> None:
        """Revoke all volatile proof; durable owner records are reconciled elsewhere."""
        with self._lock:
            for reference in tuple(self._native_stdin_closers):
                self._close_native_stdin_unlocked(reference)
            self._tasks.clear()
            self._task_retirements.clear()
            self._full.clear()
            self._full_retirements.clear()
            self._full_residual_counts.clear()
            self._task_residual_counts.clear()
            self._changed()

    def register_native_stdin(
        self,
        reference: object,
        process_handle: object,
        close_stdin: Callable[[], bool],
        process_counts: Callable[[], tuple[int, int]],
    ) -> bool:
        """Bind a launched handle's stdin to the exact permission epoch."""
        with self._lock:
            if not (
                self.is_current_task_reference(reference)
                or self.is_current_full_reference(reference)
            ):
                return False
            self._native_stdin_closers.setdefault(reference, {})[process_handle] = close_stdin
            self._native_process_counts[reference] = process_counts
            return True

    def unregister_native_stdin(self, reference: object, process_handle: object) -> None:
        """Forget a handle after its whole-tree supervisor exits."""
        with self._lock:
            registered = self._native_stdin_closers.get(reference)
            if registered is None:
                return
            registered.pop(process_handle, None)
            if not registered:
                self._native_stdin_closers.pop(reference, None)

    def task_references_for_source(
        self,
        *,
        session_key: str,
        source_binding: PermissionSourceBinding,
    ) -> tuple[WorkspaceTaskAccessReference, ...]:
        """Return current task capabilities for lifecycle fencing only."""
        try:
            source = _source_identity(session_key, source_binding)
        except (TypeError, ValueError):
            return ()
        with self._lock:
            return tuple(
                reference for identity, reference in self._tasks.items()
                if _source_for_task(identity) == source
            )

    def _close_native_stdin_unlocked(self, reference: object) -> int:
        registered = self._native_stdin_closers.pop(reference, {})
        pending: dict[object, Callable[[], bool]] = {}
        failures = 0
        for process_handle, close_stdin in registered.items():
            try:
                closed = close_stdin()
            except Exception:
                closed = False
            if not closed:
                failures += 1
                pending[process_handle] = close_stdin
        if pending:
            self._native_stdin_closers[reference] = pending
        return failures

    def _counts_for_reference_unlocked(self, reference: object) -> tuple[int, int]:
        process_counts = self._native_process_counts.get(reference)
        if process_counts is None:
            return 0, 0
        try:
            active_processes, unknown_processes = process_counts()
            return _count(active_processes), _count(unknown_processes)
        except Exception:
            return 0, 1

    def _refresh_retirement_counts_unlocked(self, retirement: _Retirement) -> bool:
        if retirement.private_reference is None:
            return False
        active_processes, unknown_processes = self._counts_for_reference_unlocked(
            retirement.private_reference,
        )
        changed = (
            retirement.active_processes != active_processes
            or retirement.unknown_processes != unknown_processes
        )
        retirement.active_processes = active_processes
        retirement.unknown_processes = unknown_processes
        return changed

    def _expire_source_tasks(self, source: _SourceIdentity) -> None:
        for identity in tuple(self._tasks):
            if _source_for_task(identity) == source:
                reference = self._tasks.pop(identity, None)
                if reference is not None:
                    self._close_native_stdin_unlocked(reference)
                self._task_retirements.pop(identity, None)

    def _next_epoch(self) -> int:
        self._epoch += 1
        return self._epoch

    def _changed(self) -> None:
        self._revision += 1


def validate_source_binding(value: object) -> PermissionSourceBinding:
    if type(value) is not dict:
        raise ValueError("invalid source binding")
    binding = cast(dict[str, object], value)
    if set(binding) != {"store_identity", "incarnation", "generation"}:
        raise ValueError("invalid source binding")
    store_identity = binding.get("store_identity")
    incarnation = binding.get("incarnation")
    generation = binding.get("generation")
    if (
        not isinstance(store_identity, str)
        or not 1 <= len(store_identity) <= _MAX_ID_LENGTH
        or not isinstance(incarnation, str)
        or not 1 <= len(incarnation) <= _MAX_ID_LENGTH
        or type(generation) is not int
        or generation < 0
    ):
        raise ValueError("invalid source binding")
    return {
        "store_identity": store_identity,
        "incarnation": incarnation,
        "generation": generation,
    }


def _task_identity(
    session_key: str,
    source_binding: PermissionSourceBinding,
    task_turn_id: str,
    task_input_event_id: str,
) -> _TaskIdentity:
    binding = validate_source_binding(dict(source_binding))
    if not 1 <= len(session_key) <= _MAX_SESSION_KEY_LENGTH:
        raise ValueError("invalid session key")
    for value in (task_turn_id, task_input_event_id):
        if not 1 <= len(value) <= _MAX_ID_LENGTH:
            raise ValueError("invalid task identity")
    return _TaskIdentity(
        session_key,
        binding["store_identity"],
        binding["incarnation"],
        binding["generation"],
        task_turn_id,
        task_input_event_id,
    )


def _source_identity(session_key: str, source_binding: PermissionSourceBinding) -> _SourceIdentity:
    binding = validate_source_binding(dict(source_binding))
    if not 1 <= len(session_key) <= _MAX_SESSION_KEY_LENGTH:
        raise ValueError("invalid session key")
    return _SourceIdentity(
        session_key,
        binding["store_identity"],
        binding["incarnation"],
        binding["generation"],
    )


def _binding_from_identity(identity: _TaskIdentity) -> PermissionSourceBinding:
    return {
        "store_identity": identity.store_identity,
        "incarnation": identity.incarnation,
        "generation": identity.generation,
    }


def _binding_from_source(source: _SourceIdentity) -> PermissionSourceBinding:
    return {
        "store_identity": source.store_identity,
        "incarnation": source.incarnation,
        "generation": source.generation,
    }


def _source_for_task(identity: _TaskIdentity) -> _SourceIdentity:
    return _SourceIdentity(
        identity.session_key,
        identity.store_identity,
        identity.incarnation,
        identity.generation,
    )


def _copy_binding(value: PermissionSourceBinding) -> PermissionSourceBinding:
    return validate_source_binding(dict(value))


def _validated_root(
    path: str | Path,
    file_access: FolderFileAccess,
    commands: bool,
) -> _Root:
    if file_access not in {"read_only", "read_write"} or type(commands) is not bool:
        raise ValueError("invalid folder capability")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ValueError("folder path must be absolute")
    canonical = candidate.resolve(strict=True)
    if not canonical.is_dir() or candidate != canonical:
        raise ValueError("folder path must be a canonical directory")
    return _Root(str(canonical), file_access, commands)


def _root_still_canonical(path: str) -> bool:
    root = Path(path)
    try:
        return root.is_dir() and root.resolve(strict=True) == root
    except (OSError, RuntimeError, ValueError):
        return False


def _count(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("process count must be a nonnegative integer")
    return value


__all__ = [
    "attach_full_permission_to_message",
    "copy_full_permission_message_reference",
    "DesktopPermissionOwner",
    "full_permission_from_message",
    "PermissionFullUnavailableReason",
    "PermissionRetirementReason",
    "PermissionSourceBinding",
    "TaskAccessFolder",
    "TaskAccessState",
    "WorkspaceFullAccessReference",
    "WorkspacePermissionConflictError",
    "WorkspacePermissionProjection",
    "WorkspacePermissionUnavailableError",
    "WorkspaceTaskAccessReference",
    "validate_source_binding",
]
