"""Narrow execution backend for one delegated child generation.

Both backends use the canonical child turn executor. They own task lifetime;
the executor selects the runner from the saved harness.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from kohaku.agent.child_sessions import BackgroundDeliveryIntent, ChildAdmission
from kohaku.agent.hook import AgentHook
from kohaku.agent.policy import ToolPolicy
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.agent.turn import ApprovalDecision, DeliveryTarget, TurnResult
from kohaku.bus.events import InboundMessage
from kohaku.harness.backend import ExternalToolPolicy
from kohaku.security.desktop_permissions import WorkspaceTaskAccessReference
from kohaku.session.pending_followups import PendingFollowupRecord
from kohaku.utils.llm_runtime import LLMRuntime

if TYPE_CHECKING:
    from kohaku.session.artifact_grants import CanonicalChildArtifactInput


class ChildBackendError(RuntimeError):
    """A child backend cannot start, resume, or control this generation."""


class ChildBackendUnavailableError(ChildBackendError):
    """The requested backend or generation is unavailable."""


@dataclass(frozen=True, slots=True)
class ChildBackendBinding:
    """Content-free proof for the current child execution generation."""

    backend: str = "native"
    runtime_generation: int = 0
    proof: str = ""

    def __post_init__(self) -> None:
        if self.backend not in {"native", "external"}:
            raise ValueError("child backend binding is invalid")
        if type(self.runtime_generation) is not int or self.runtime_generation < 0:
            raise ValueError("child backend generation is invalid")
        if type(self.proof) is not str:
            raise ValueError("child backend proof is invalid")


@dataclass(slots=True)
class ChildBackendHandle:
    """One live child execution owned by a backend."""

    child_id: str
    runtime_generation: int
    task: asyncio.Task[Any]


@dataclass(frozen=True, slots=True)
class ChildTurnRequest:
    """Trusted, scoped inputs for one canonical child turn."""

    admission: ChildAdmission
    run_id: str
    role: str
    instructions: str
    assignment: str
    tools: ToolRegistry
    runtime: LLMRuntime
    max_iterations: int
    hook: AgentHook
    policy: ToolPolicy | None = None
    system_prompt: str | None = None
    include_skill_context: bool = True
    approval_target: DeliveryTarget | None = None
    workspace_task_reference: WorkspaceTaskAccessReference | None = None
    approval_decision: ApprovalDecision | None = None
    artifact_input: CanonicalChildArtifactInput | None = None
    background_delivery: BackgroundDeliveryIntent | None = None
    followup_record: PendingFollowupRecord | None = None
    followup_records: tuple[PendingFollowupRecord, ...] = ()
    followup_records_to_admit: tuple[PendingFollowupRecord, ...] = ()
    followup_records_to_discard: tuple[PendingFollowupRecord, ...] = ()
    request_boundary_validator: Callable[[], bool] | None = None
    followup_boundary_validator: Callable[[], bool] | None = None
    followup_request_acknowledged: Callable[[tuple[str, ...]], None] | None = None
    external_policy: ExternalToolPolicy | None = None
    external_authority: Callable[[], bool] | None = None
    external_tool_is_current: Callable[[str], bool] | None = None
    prior_external_session_id: str | None = None


@dataclass(frozen=True, slots=True)
class ChildApprovalResume:
    """One typed approval decision bound to a new child run."""

    request: ChildTurnRequest
    approval_id: str
    blocked_run_id: str
    decision: ApprovalDecision


class ChildTurnExecutor(Protocol):
    """AgentLoop adapter that admits child work through execute_turn."""

    async def execute_child_turn(
        self,
        request: ChildTurnRequest,
        *,
        approval_resume: ChildApprovalResume | None = None,
    ) -> TurnResult: ...

    async def submit_child_followup(
        self,
        admission: ChildAdmission,
        record: PendingFollowupRecord,
        message: InboundMessage,
    ) -> str: ...


RunCall = Callable[[], Coroutine[Any, Any, Any]]


@runtime_checkable
class ChildBackend(Protocol):
    """Start, control, and settle one child generation."""

    async def start(
        self,
        admission: ChildAdmission,
        run: RunCall,
        binding: ChildBackendBinding,
        *,
        track: Callable[[asyncio.Task[Any]], None] | None = None,
    ) -> ChildBackendHandle: ...

    async def resume(
        self,
        admission: ChildAdmission,
        run: RunCall,
        binding: ChildBackendBinding,
    ) -> ChildBackendHandle: ...

    async def send(self, child_id: str, content: str) -> None: ...

    async def steer(self, child_id: str, content: str) -> None: ...

    async def interrupt(self, child_id: str) -> bool: ...

    async def wait(self, handle: ChildBackendHandle) -> Any: ...

    async def close(self, child_id: str) -> None: ...


class NativeChildBackend:
    """Run one child through the normal Kohaku runner."""

    backend_name = "native"

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    async def start(
        self,
        admission: ChildAdmission,
        run: RunCall,
        binding: ChildBackendBinding,
        *,
        track: Callable[[asyncio.Task[Any]], None] | None = None,
    ) -> ChildBackendHandle:
        if binding.backend != self.backend_name:
            raise ChildBackendUnavailableError("child backend binding does not match")
        if binding.runtime_generation != admission.runtime_generation:
            raise ChildBackendUnavailableError("child runtime generation is stale")
        existing = self._tasks.get(admission.child_id)
        if existing is not None and not existing.done():
            raise ChildBackendUnavailableError("child already has a live execution")
        task = asyncio.create_task(run())
        self._tasks[admission.child_id] = task
        task.add_done_callback(
            lambda finished, child_id=admission.child_id: self._forget(child_id, finished)
        )
        if track is not None:
            track(task)
        return ChildBackendHandle(
            child_id=admission.child_id,
            runtime_generation=admission.runtime_generation,
            task=task,
        )

    async def resume(
        self,
        admission: ChildAdmission,
        run: RunCall,
        binding: ChildBackendBinding,
    ) -> ChildBackendHandle:
        if binding.backend != self.backend_name:
            raise ChildBackendUnavailableError("child backend binding does not match")
        if (
            binding.runtime_generation != admission.runtime_generation
            or not binding.proof.strip()
        ):
            raise ChildBackendUnavailableError("child resume proof is missing or stale")
        # The native runner is process-local. A restarted child must be
        # reconciled as uncertain rather than replayed.
        raise ChildBackendUnavailableError("native child execution cannot resume after restart")

    async def send(self, child_id: str, content: str) -> None:
        raise ChildBackendUnavailableError("native child has no mid-run input")

    async def steer(self, child_id: str, content: str) -> None:
        raise ChildBackendUnavailableError("native child has no mid-run steering")

    def _forget(self, child_id: str, task: asyncio.Task[Any]) -> None:
        if self._tasks.get(child_id) is task:
            self._tasks.pop(child_id, None)

    async def interrupt(self, child_id: str) -> bool:
        task = self._tasks.pop(child_id, None)
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def wait(self, handle: ChildBackendHandle) -> Any:
        return await handle.task

    async def close(self, child_id: str) -> None:
        await self.interrupt(child_id)


class ClaudeChildBackend(NativeChildBackend):
    """Own Claude child tasks through the same canonical execution lifecycle."""

    backend_name = "external"


class ChildBackendRegistry:
    """Resolve the execution engine independently of the admitted role."""

    def __init__(self, native: NativeChildBackend | None = None) -> None:
        self._native = native or NativeChildBackend()
        self._claude = ClaudeChildBackend()

    @property
    def native(self) -> NativeChildBackend:
        return self._native

    def resolve(self, profile: str, harness: str = "native") -> ChildBackend:
        if profile == "delegated":
            if harness == "native":
                return self._native
            if harness == "claude_sdk":
                return self._claude
        raise ChildBackendUnavailableError(
            f"child backend for profile {profile!r} is not available"
        )


__all__ = [
    "ChildBackend",
    "ChildBackendBinding",
    "ChildBackendError",
    "ChildBackendHandle",
    "ChildBackendRegistry",
    "ChildBackendUnavailableError",
    "ChildApprovalResume",
    "ChildTurnExecutor",
    "ChildTurnRequest",
    "NativeChildBackend",
]
