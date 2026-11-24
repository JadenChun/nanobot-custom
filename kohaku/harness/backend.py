"""Typed turn, process, and approval contracts for external harnesses."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import uuid4

from kohaku.agent.permission_rules import extract_saveable_sandbox_command
from kohaku.agent.turn import ApprovalDecision
from kohaku.events import NO_EVENTS, EventSink
from kohaku.session.harness_binding import SessionHarnessBinding as SessionHarnessBinding

if TYPE_CHECKING:
    from kohaku.agent.hook import AgentHook
    from kohaku.llm_usage.context import LLMUsageSource
    from kohaku.session.turn_writer import SourceTurnWriter
    from kohaku.utils.llm_runtime import LLMRuntime


class LoopbackBridgeHandle(Protocol):
    @property
    def base_url(self) -> str: ...

    @property
    def auth_token(self) -> str: ...

    async def close(self) -> None: ...


BridgeFactory = Callable[["LLMRuntime", "LLMUsageSource | None"], Awaitable[LoopbackBridgeHandle]]


@dataclass(frozen=True, slots=True)
class ExternalToolPolicy:
    mode: Literal["sandbox", "full_access"] = "sandbox"
    # None admits the workspace. An empty tuple admits read-only tools.
    write_scope: tuple[Path, ...] | None = None
    sandbox_excluded_commands: tuple[str, ...] = ()
    sandbox_allow_local_binding: bool = True
    sandbox_allow_unix_sockets: tuple[str, ...] = ()

    @property
    def sandboxed(self) -> bool:
        return self.mode == "sandbox" or self.write_scope is not None


@dataclass(frozen=True, slots=True)
class ExternalTurnRequest:
    session_key: str
    run_id: str
    binding: SessionHarnessBinding
    workspace: Path
    prompt: str
    runtime: LLMRuntime
    hook: AgentHook
    writer: SourceTurnWriter
    is_authorized: Callable[[], bool]
    policy: ExternalToolPolicy = field(default_factory=ExternalToolPolicy)
    usage_source: LLMUsageSource | None = None
    prior_external_session_id: str | None = None
    on_session_id: Callable[[str], bool | None] | None = None
    on_approval: Callable[[PendingExternalApproval], Awaitable[None]] | None = None
    events: EventSink = NO_EVENTS
    private_roots: tuple[Path, ...] = ()
    on_approval_closed: Callable[[str], Awaitable[None]] | None = None
    retain_process: bool = True


@dataclass(frozen=True, slots=True)
class ExternalTurnOutcome:
    binding: SessionHarnessBinding
    content: str
    status: Literal["completed", "failed", "cancelled"]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExternalSessionStarted:
    session_id: str


@dataclass(frozen=True, slots=True)
class ExternalText:
    text: str


@dataclass(frozen=True, slots=True)
class ExternalReasoning:
    # None closes the current live reasoning segment.
    text: str | None


@dataclass(frozen=True, slots=True)
class ExternalToolCall:
    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class ExternalToolIntent:
    call: ExternalToolCall


@dataclass(frozen=True, slots=True)
class ExternalToolResult:
    call_id: str
    content: object
    is_error: bool = False
    dispatched: bool = True
    file_edit: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ExternalTurnFinished:
    error: str | None = None
    cancelled: bool = False
    final_text: str | None = None


@dataclass(frozen=True, slots=True)
class ExternalToolBatchReady:
    """The SDK finished reporting the current model response."""


ExternalEvent = (
    ExternalSessionStarted | ExternalText | ExternalReasoning | ExternalToolIntent
    | ExternalToolResult | ExternalTurnFinished | ExternalToolBatchReady
)
ExternalApprovalAction = Literal["approve", "approve_and_save", "deny"]
RequestApproval = Callable[[ExternalToolCall, str, bool], Awaitable[ExternalApprovalAction]]


class ExternalHarnessProcess(Protocol):
    def run(
        self, request: ExternalTurnRequest, approve: RequestApproval
    ) -> AsyncGenerator[ExternalEvent, None]: ...

    async def close(self) -> None: ...


class SessionHarnessBackend(Protocol):
    async def start(
        self, request: ExternalTurnRequest, *, resume: str | None, fork_session: bool
    ) -> ExternalHarnessProcess: ...


class ExternalApprovalSaveError(RuntimeError):
    """The owner could not persist the approved exclusion."""


class HarnessCapabilityError(RuntimeError):
    """The external harness is unavailable in this installation."""


@dataclass(frozen=True, slots=True)
class PendingExternalApproval:
    approval_id: str
    session_key: str
    session_incarnation: str
    run_id: str
    runtime_generation: int
    tool_call_id: str
    tool_name: str
    arguments_json: str
    reason: str
    save_exclusion_command: str | None


@dataclass(slots=True)
class _ApprovalWaiter:
    approval: PendingExternalApproval
    future: asyncio.Future[ExternalApprovalAction]
    is_authorized: Callable[[], bool]


class ExternalApprovalRegistry:
    """Own live approvals and settle each server-issued ID once."""

    def __init__(self, save_exclusion: Callable[[str], bool | None] | None = None) -> None:
        self._pending: dict[str, _ApprovalWaiter] = {}
        self._save_exclusion = save_exclusion

    async def request(
        self, request: ExternalTurnRequest, call: ExternalToolCall, reason: str, allow_save: bool
    ) -> ExternalApprovalAction:
        if request.on_approval is None or not request.is_authorized():
            return "deny"
        command = call.arguments.get("command")
        saveable = (
            extract_saveable_sandbox_command(command)
            if allow_save and call.name == "exec" and self._save_exclusion is not None
            and isinstance(command, str)
            else None
        )
        approval = PendingExternalApproval(
            uuid4().hex, request.session_key, request.writer.committed_state.incarnation,
            request.run_id, request.binding.runtime_generation,
            call.id, call.name, json.dumps(call.arguments, sort_keys=True), reason, saveable,
        )
        future: asyncio.Future[ExternalApprovalAction] = asyncio.get_running_loop().create_future()
        self._pending[approval.approval_id] = _ApprovalWaiter(
            approval, future, request.is_authorized,
        )
        try:
            await request.on_approval(approval)
            decision = await future
            return decision if request.is_authorized() else "deny"
        finally:
            self._pending.pop(approval.approval_id, None)
            future.cancel()
            if request.on_approval_closed is not None:
                await request.on_approval_closed(approval.approval_id)

    def resolve(
        self, decision: ApprovalDecision, *, save_exclusion_command: str | None = None,
    ) -> bool:
        waiter = self._pending.get(decision.approval_id)
        if waiter is None or waiter.future.done():
            return False
        approval = waiter.approval
        if (
            decision.session_key != approval.session_key or decision.run_id != approval.run_id
            or decision.generation != approval.runtime_generation
            or (decision.session_incarnation is not None
                and decision.session_incarnation != approval.session_incarnation)
            or (decision.effect_id is not None and decision.effect_id != approval.tool_call_id)
            or not waiter.is_authorized()
        ):
            return False
        if decision.decision == "approve_and_save":
            command = approval.save_exclusion_command
            save = self._save_exclusion
            if command is None or save is None or save_exclusion_command not in (None, command):
                return False
            if save(command) is False:
                raise ExternalApprovalSaveError("Kohaku could not save the Sandbox exclusion.")
        elif save_exclusion_command is not None:
            return False
        self._pending.pop(approval.approval_id)
        action: ExternalApprovalAction = (
            "approve_and_save" if decision.decision == "approve_and_save"
            else "approve" if decision.decision == "approve" else "deny"
        )
        waiter.future.set_result(action)
        return True

    def pending_tool_calls(self, run_id: str) -> frozenset[str]:
        return frozenset(
            waiter.approval.tool_call_id for waiter in self._pending.values()
            if waiter.approval.run_id == run_id and not waiter.future.done()
        )

    def cancel_for_run(self, run_id: str) -> None:
        self._cancel(lambda approval: approval.run_id == run_id)

    def cancel_for_session(self, session_key: str) -> None:
        self._cancel(lambda approval: approval.session_key == session_key)

    def _cancel(self, matches: Callable[[PendingExternalApproval], bool]) -> None:
        for key, waiter in tuple(self._pending.items()):
            if matches(waiter.approval):
                del self._pending[key]
                if not waiter.future.done():
                    waiter.future.set_result("deny")
