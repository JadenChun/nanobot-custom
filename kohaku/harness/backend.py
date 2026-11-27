"""Typed turn, process, and approval contracts for external harnesses."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from kohaku.agent.approvals import (
    PendingTurnApproval,
    TurnApprovalRegistry,
    TurnApprovalSaveError,
)
from kohaku.events import NO_EVENTS, EventSink
from kohaku.session.harness_binding import SessionHarnessBinding as SessionHarnessBinding

if TYPE_CHECKING:
    from kohaku.agent.auto_review import AutoReviewResult
    from kohaku.agent.hook import AgentHook
    from kohaku.llm_usage.context import LLMUsageSource
    from kohaku.session.turn_writer import SourceTurnWriter
    from kohaku.utils.llm_runtime import LLMRuntime


PendingExternalApproval = PendingTurnApproval
ExternalApprovalRegistry = TurnApprovalRegistry
ExternalApprovalSaveError = TurnApprovalSaveError


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
    exact_write_paths: frozenset[Path] = frozenset()
    allowed_tools: frozenset[str] | None = None

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
    max_turns: int | None = None
    approval_session_key: str | None = None
    authorize_tool: Callable[[str, dict[str, object]], bool] | None = None
    drain_steer: Callable[[Callable[[], Awaitable[None]], bool], Awaitable[list[str]]] | None = None
    auto_review: Callable[[ExternalToolCall, str], Awaitable[AutoReviewResult]] | None = None


@dataclass(frozen=True, slots=True)
class ExternalTurnOutcome:
    binding: SessionHarnessBinding
    content: str
    status: Literal["completed", "failed", "cancelled"]
    error: str | None = None
    tools_used: tuple[str, ...] = ()
    file_edits: tuple[dict[str, object], ...] = ()
    commands: tuple[tuple[str, str], ...] = ()


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


@dataclass(frozen=True, slots=True)
class ExternalSteerAvailable:
    """The active run has guidance in its mailbox."""


@dataclass(frozen=True, slots=True)
class ExternalQueryFinished:
    """One SDK response ended while another query remains active."""


ExternalEvent = (
    ExternalSessionStarted | ExternalText | ExternalReasoning | ExternalToolIntent
    | ExternalToolResult | ExternalTurnFinished | ExternalToolBatchReady
    | ExternalSteerAvailable | ExternalQueryFinished
)
ExternalApprovalAction = Literal["approve", "approve_and_save", "deny"]
RequestApproval = Callable[[ExternalToolCall, str, bool], Awaitable[ExternalApprovalAction]]


class ExternalHarnessProcess(Protocol):
    def run(
        self, request: ExternalTurnRequest, approve: RequestApproval
    ) -> AsyncGenerator[ExternalEvent, None]: ...

    async def close(self) -> None: ...


@runtime_checkable
class StreamingSteerProcess(Protocol):
    """External process that accepts user input while a query is active."""

    async def send_steer(self, prompt: str) -> bool: ...

    def notify_steer(self) -> None: ...


class SessionHarnessBackend(Protocol):
    async def start(
        self, request: ExternalTurnRequest, *, resume: str | None, fork_session: bool
    ) -> ExternalHarnessProcess: ...


class HarnessCapabilityError(RuntimeError):
    """The external harness is unavailable in this installation."""
