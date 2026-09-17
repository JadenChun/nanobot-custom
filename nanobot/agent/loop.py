"""Agent loop: the core processing engine."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import os
import re
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Coroutine, Iterable, Mapping
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    ExitStack,
    asynccontextmanager,
    nullcontext,
    suppress,
)
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol, TypeVar, cast

from loguru import logger

from nanobot.agent import context as agent_context
from nanobot.agent import model_presets as preset_helpers
from nanobot.agent.autocompact import AutoCompact
from nanobot.agent.automation_turns import publish_next_deferred_turn
from nanobot.agent.context import ContextBuilder, PersistedPromptContextResolver, TranscriptInput
from nanobot.agent.cron_turns import CronTurnCoordinator
from nanobot.agent.delegation import reset_current_turn_context, set_current_turn_context
from nanobot.agent.foreground import build_foreground_dependencies, validate_foreground_dependencies
from nanobot.agent.hook import AgentHook, AgentTurnHookFactory
from nanobot.agent.idle_lifecycle import IdleLifecycleIntegration
from nanobot.agent.memory import Consolidator
from nanobot.agent.model_runtime import ModelRuntimeResolver
from nanobot.agent.policy import RiskyActionPolicy
from nanobot.agent.run_store import RunStore
from nanobot.agent.runner import (
    _MAX_INJECTIONS_PER_TURN,
    AgentRunner,
    AgentRunResult,
    AgentRunSpec,
)
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.context import (
    ForkToolDependencies,
    RequestContext,
    bind_request_context,
    reset_request_context,
)
from nanobot.agent.tools.exec_session import ExecSessionManager
from nanobot.agent.tools.file_state import FileStateStore, bind_file_states, reset_file_states
from nanobot.agent.tools.message import capture_message_deliveries
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.runtime_control import AgentRuntimeControl
from nanobot.agent.trace import sanitize_trace
from nanobot.agent.turn import (
    DeliveryState,
    DeliveryTarget,
    HistoryMode,
    RunPolicyContext,
    RunRecord,
    RunStatus,
    SessionRunRef,
    TraceMode,
    TurnCallbacks,
    TurnRequest,
    TurnResult,
    TurnSource,
)
from nanobot.agent.turn import TurnContext as RunTurnContext
from nanobot.agent.turn_admission import ApprovalAdmission
from nanobot.agent.turn_delivery import (
    TurnDelivery,
    TurnDeliveryFactory,
)
from nanobot.agent.turn_delivery import TurnRoute as TurnRoute
from nanobot.agent.turn_hooks import AgentTurnHookSpec, build_agent_turn_hook
from nanobot.bus.events import INBOUND_META_USER_SHELL, InboundMessage, OutboundMessage
from nanobot.bus.outbound_events import (
    StreamDeltaEvent,
    StreamedResponseEvent,
    StreamEndEvent,
)
from nanobot.bus.queue import MessageBus
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.command.router import normalize_command_text
from nanobot.config.schema import AgentDefaults, ModelPresetConfig
from nanobot.cron.session_turns import cron_trigger
from nanobot.events import NO_EVENTS, AgentEvent, EventSink
from nanobot.llm_usage.context import source_from_request
from nanobot.providers.base import LLMProvider, LLMUsage, ProviderConversationState
from nanobot.providers.conversation_state import (
    ConversationRequestBinding,
    ProviderStateSettlement,
)
from nanobot.providers.factory import ProviderSnapshot
from nanobot.runtime_context import (
    RUNTIME_CONTEXT_HISTORY_META,
    RUNTIME_CONTEXT_MESSAGE_META,
    RuntimeContextBlock,
    RuntimeContextProvider,
    append_runtime_context,
    resolve_runtime_context,
    runtime_context_blocks_from_metadata,
)
from nanobot.security.workspace_access import (
    WorkspaceScopeResolver,
    bind_workspace_scope,
    reset_workspace_scope,
)
from nanobot.session import turn_continuation
from nanobot.session.automation_turns import automation_history_overrides
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.goal_state import (
    goal_state_runtime_lines,
    runner_wall_llm_timeout_s,
)
from nanobot.session.history_visibility import HIDDEN_HISTORY_META
from nanobot.session.keys import UNIFIED_SESSION_KEY, remember_last_channel
from nanobot.session.manager import SESSION_CACHE_MAX_SIZE, Session, SessionManager
from nanobot.session.model_selection import (
    SESSION_MODEL_PRESET_METADATA_KEY,
    model_preset_from_metadata,
)
from nanobot.session.recovery import (
    PENDING_FOLLOWUP_ID_KEY,
    RECOVERY_INBOUND_METADATA_KEY,
    RecoveryAdmission,
    acknowledge_pending_followups,
    record_pending_followup,
    restore_pending_interruption,
    restore_runtime_checkpoint,
)
from nanobot.session.source_projection import SourceNode, source_nodes
from nanobot.session.source_request import SourceTranscript
from nanobot.session.summary import (
    SessionSummary,
    SessionSummaryCheckpoint,
)
from nanobot.session.turn_writer import SourceTurnWriter
from nanobot.triggers.local_session_turns import local_trigger
from nanobot.triggers.local_turns import LocalTriggerTurnCoordinator
from nanobot.utils.cancellation import task_is_cancelling
from nanobot.utils.document import reference_non_image_attachments
from nanobot.utils.helpers import image_placeholder_text
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.progress_events import output_events
from nanobot.utils.runtime import (
    EMPTY_FINAL_RESPONSE_MESSAGE,
)

if TYPE_CHECKING:
    from nanobot.agent.tools.filesystem import ReadFileTool
    from nanobot.config.schema import (
        ChannelsConfig,
        Config,
        ProviderConfig,
        ToolsConfig,
    )
    from nanobot.cron.service import CronService
    from nanobot.session.artifact_reader import ArtifactReader
    from nanobot.triggers.local_store import LocalTriggerStore

_T = TypeVar("_T")
_SUBAGENT_PROVIDER_TASK_META = "subagent_provider_task_id"
_SUBAGENT_TERMINAL_WAIT_SECONDS = 300.0
_BUS_CLEANUP_TIMEOUT_SECONDS = 5.0
_FOREGROUND_IDLE_SOURCES = frozenset({
    TurnSource.DIRECT,
    TurnSource.GATEWAY,
    TurnSource.SDK,
    TurnSource.API,
})
_ACTIVITY_ADMITTED_ATTRIBUTE = "_nanobot_activity_admitted"
_ACTIVITY_NOTIFICATION_PENDING_ATTRIBUTE = "_nanobot_activity_notification_pending"


class ForegroundIdleIntegration(Protocol):
    """Complete, explicitly injected seam for foreground idle cancellation."""

    enabled: bool

    async def foreground_admitted(self, session_key: str) -> None:
        """Cancel/discard idle work after durable activity is committed."""


class TurnKind(Enum):
    USER = auto()
    SYSTEM = auto()


@dataclass
class TurnContext:
    msg: InboundMessage
    session_key: str
    turn_id: str
    runtime: LLMRuntime | None
    kind: TurnKind
    delivery: TurnDelivery
    original_user_text: str | None = None
    session: Session | None = None

    history: list[dict[str, Any]] = field(default_factory=list)
    transcript_input: TranscriptInput | None = None
    provider_state: ProviderConversationState | None = field(default=None, repr=False)
    provider_state_settlement: ProviderStateSettlement | None = field(default=None, repr=False)
    request_context: RequestContext | None = None
    runtime_context_blocks: list[RuntimeContextBlock] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    run_context: RunTurnContext | None = None
    run_result: AgentRunResult | None = None
    sent_messages: list[OutboundMessage] = field(default_factory=list)
    final_content: str | None = None
    all_messages: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    failure_error_kind: str | None = None
    streamed_content: bool = False

    input_persisted_early: bool = False
    trace_input_message: dict[str, Any] | None = None
    trace_detail_messages: list[dict[str, Any]] = field(default_factory=list)
    source_metadata_base: dict[str, Any] = field(default_factory=dict)
    save_skip: int = 0

    outbound: OutboundMessage | None = None
    suppress_response: bool = False

    events: EventSink = NO_EVENTS
    streaming: bool = False
    on_runtime_admitted: Callable[[LLMRuntime], Awaitable[None]] | None = None

    pending_queue: asyncio.Queue[InboundMessage] | None = None
    pending_summary: SessionSummary | None = None
    summary_checkpoint: SessionSummaryCheckpoint | None = None
    provider_compaction_applied: bool = False

    ephemeral: bool = False
    run_extra_hooks_for_ephemeral: bool = False
    hooks: list[AgentHook] = field(default_factory=list)
    hook_factories: list[AgentTurnHookFactory] = field(default_factory=list)
    turn_scopes: list[AbstractContextManager[Any]] = field(default_factory=list)
    tools: ToolRegistry | None = None

    turn_wall_started_at: float = field(default_factory=time.time)
    visible_run_started_at: float | None = None
    turn_latency_ms: int | None = None
    usage: LLMUsage | None = None

    @property
    def fresh_history(self) -> bool:
        request = self.run_context.request if self.run_context is not None else None
        return request is not None and request.history_mode is HistoryMode.FRESH

    def require_runtime(self) -> LLMRuntime:
        """Return the runtime established by the BUILD stage."""
        if self.runtime is None:
            raise RuntimeError("turn runtime is not initialized; BUILD must run before this stage")
        return self.runtime

    def require_session(self) -> Session:
        """Return the session established by the RESTORE stage."""
        if self.session is None:
            raise RuntimeError("turn session is not initialized; RESTORE must run before this stage")
        return self.session


@dataclass
class _AdmittedTurnIO:
    pending_queue: asyncio.Queue[InboundMessage] | None = None
    delivery: TurnDelivery | None = None
    result: TurnResult | None = None
    delivery_error: str | None = None
    message: InboundMessage | None = None


@dataclass
class _ActivityAdmissionRequest:
    message: InboundMessage
    result: asyncio.Future[_ForegroundActivityResult]


@dataclass(frozen=True)
class _ForegroundActivityResult:
    activity_committed: bool = False
    callback_succeeded: bool = False
    callback_error: BaseException | None = None


@dataclass(frozen=True)
class _AdmittedActivityMessage:
    """Internal identity proof for one committed queued activity update."""

    message: InboundMessage
    notification_succeeded: bool = False


@dataclass
class _SourceActivityAdmission:
    queue: asyncio.Queue[_ActivityAdmissionRequest | None]
    task: asyncio.Task[None]
    closed: bool = False

    async def submit(self, message: InboundMessage) -> _ForegroundActivityResult:
        """Admit one queued message and wait for its durable side effect."""

        if self.closed:
            return _ForegroundActivityResult()
        result = asyncio.get_running_loop().create_future()
        self.queue.put_nowait(_ActivityAdmissionRequest(message, result))
        return await asyncio.shield(result)

    async def close(self) -> None:
        """Drain admitted work and stop the owner-registered worker."""

        if self.closed:
            return
        self.closed = True
        self.queue.put_nowait(None)
        interrupted = False
        while not self.task.done():
            try:
                await asyncio.shield(self.task)
            except asyncio.CancelledError:
                interrupted = True
        self.task.result()
        if interrupted:
            raise asyncio.CancelledError


@dataclass(frozen=True)
class _TurnExecutionOptions:
    """Trusted adapter options; never reconstructed from inbound metadata."""

    ephemeral: bool = False
    run_extra_hooks_for_ephemeral: bool = False
    hook_factories: tuple[AgentTurnHookFactory, ...] = ()
    runtime: LLMRuntime | None = None
    on_runtime_admitted: Callable[[LLMRuntime], Awaitable[None]] | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    raise_errors: bool = False
    on_admission_end: Callable[[str], Awaitable[None]] | None = None
    admission_scope: Callable[[str], AbstractAsyncContextManager[_AdmittedTurnIO]] | None = None
    inbound: InboundMessage | None = None
    trusted_foreground_activity: bool = False
    activity_already_recorded: bool = False
    retry_activity_notification: bool = False
    admitted_activity_marker: _AdmittedActivityMessage | None = None


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    @property
    def tool_names(self) -> list[str]:
        return self.tools.tool_names

    @property
    def provider(self) -> LLMProvider:
        """Provider selected for future turn admissions."""
        return self.runtime_resolver.runtime.provider

    @property
    def model(self) -> str:
        """Model selected for future turn admissions."""
        return self.runtime_resolver.runtime.model

    @property
    def context_window_tokens(self) -> int:
        """Context limit selected for future turn admissions."""
        return self.runtime_resolver.runtime.context_window_tokens

    @property
    def model_presets(self) -> Mapping[str, ModelPresetConfig]:
        """Configured model presets exposed for selection and display."""
        return self.runtime_resolver.model_presets

    @property
    def model_preset(self) -> str | None:
        return self.runtime_resolver.model_preset

    @model_preset.setter
    def model_preset(self, name: str | None) -> None:
        self.set_model_preset(name)

    def llm_runtime(self) -> LLMRuntime:
        """Resolve the immutable default used to admit the next turn."""
        previous = self.runtime_resolver.runtime
        runtime = self.runtime_resolver.admit()
        if (
            runtime.model != previous.model
            or runtime.model_preset != previous.model_preset
            or runtime.snapshot_signature != previous.snapshot_signature
        ):
            self._publish_runtime_selection(runtime)
        return runtime

    def dream_runtime(self) -> LLMRuntime | None:
        """Resolve the optional preset used for Dream without changing defaults."""
        if not self.dream_model_preset:
            return None
        return self.runtime_resolver.resolve_preset(self.dream_model_preset)

    _RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"
    _PENDING_USER_TURN_KEY = "pending_user_turn"
    _PROVIDER_STATE_CHECKPOINT_VERSION_KEY = "provider_state_checkpoint_version"
    _PROVIDER_STATE_CHECKPOINT_VERSION = "v1"

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int | None = None,
        max_concurrent_subagents: int | None = None,
        context_window_tokens: int | None = None,
        max_tool_result_chars: int | None = None,
        provider_retry_mode: str = "standard",
        tool_hint_max_length: int | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        tool_registry: ToolRegistry | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        session_ttl_minutes: int = 0,
        hooks: list[AgentHook] | None = None,
        hook_factories: list[AgentTurnHookFactory] | None = None,
        unified_session: bool = False,
        disabled_skills: list[str] | None = None,
        tools_config: ToolsConfig | None = None,
        image_generation_provider_config: ProviderConfig | None = None,
        image_generation_provider_configs: dict[str, ProviderConfig] | None = None,
        provider_snapshot_loader: Callable[..., ProviderSnapshot] | None = None,
        provider_signature: tuple[object, ...] | None = None,
        model_presets: dict[str, ModelPresetConfig] | None = None,
        preset_catalog_loader: preset_helpers.PresetCatalogLoader | None = None,
        model_preset: str | None = None,
        dream_model_preset: str | None = None,
        preset_snapshot_loader: preset_helpers.PresetSnapshotLoader | None = None,
        turn_delivery_factory: TurnDeliveryFactory | None = None,
        runtime_model_publisher: Callable[[str, str | None], None] | None = None,
        restart_mode: str = "auto",
        local_trigger_store: LocalTriggerStore | None = None,
        idle_compact_check_interval_seconds: int = 0,
        recovery_admission: RecoveryAdmission | None = None,
        fork_dependencies: ForkToolDependencies | None = None,
        _owns_fork_dependencies: bool = False,
        artifact_reader_factory: Callable[
            [SourceTurnWriter, bool, ReadFileTool], ArtifactReader | None
        ] | None = None,
        idle_integration: ForegroundIdleIntegration | None = None,
    ):
        from nanobot.migration_guard import require_runtime_ready
        require_runtime_ready()
        from nanobot.config.schema import ToolsConfig

        _tc = tools_config or ToolsConfig()
        defaults = AgentDefaults()
        self.bus = bus
        self._recovery_admission = recovery_admission
        if turn_delivery_factory is not None:
            if turn_delivery_factory.bus is not bus:
                raise ValueError("turn delivery factory must use the agent message bus")
            self.turn_delivery_factory = turn_delivery_factory
        else:
            self.turn_delivery_factory = TurnDeliveryFactory(bus)
        self.runtime_event_publisher = self.turn_delivery_factory.runtime_event_publisher
        self.channels_config = channels_config
        self.restart_mode = restart_mode
        self._runtime_model_publisher = runtime_model_publisher
        # No production factory supplies this seam yet.  Keep it inert unless
        # callers inject both an explicit enabled integration and its callback.
        self._idle_integration = idle_integration
        self.workspace = workspace
        initial_model = model or provider.get_default_model()
        self.max_iterations = (
            max_iterations if max_iterations is not None else defaults.max_tool_iterations
        )
        initial_context_window = (
            context_window_tokens
            if context_window_tokens is not None
            else defaults.context_window_tokens
        )
        configured_presets = model_presets or {}
        self.runtime_resolver = ModelRuntimeResolver(
            LLMRuntime.capture(
                provider,
                initial_model,
                context_window_tokens=initial_context_window,
                snapshot_signature=provider_signature,
            ),
            model_presets=configured_presets,
            preset_catalog_loader=preset_catalog_loader,
            configured_default_preset=model_preset,
            provider_snapshot_loader=provider_snapshot_loader,
            preset_snapshot_loader=preset_snapshot_loader,
        )
        self.dream_model_preset = dream_model_preset
        self.max_tool_result_chars = (
            max_tool_result_chars
            if max_tool_result_chars is not None
            else defaults.max_tool_result_chars
        )
        self.provider_retry_mode = provider_retry_mode
        self.tool_hint_max_length = (
            tool_hint_max_length if tool_hint_max_length is not None
            else defaults.tool_hint_max_length
        )
        self.tools_config = _tc
        self.web_config = _tc.web
        self.exec_config = _tc.exec
        self._image_generation_provider_configs = dict(image_generation_provider_configs or {})
        if (
            image_generation_provider_config is not None
            and "openrouter" not in self._image_generation_provider_configs
        ):
            self._image_generation_provider_configs["openrouter"] = image_generation_provider_config
        self.cron_service = cron_service
        self.local_trigger_store = local_trigger_store
        self.restrict_to_workspace = restrict_to_workspace
        self.workspace_scopes = WorkspaceScopeResolver(
            default_workspace=workspace,
            default_restrict_to_workspace=restrict_to_workspace,
        )
        self._start_time = time.time()
        self._extra_hooks: list[AgentHook] = hooks or []
        self._hook_factories: list[AgentTurnHookFactory] = hook_factories or []

        if fork_dependencies is None:
            fork_dependencies = build_foreground_dependencies(
                workspace=workspace, runtime=self.runtime_resolver.runtime,
                tools_config=_tc, restrict_to_workspace=restrict_to_workspace)
            _owns_fork_dependencies = True
        validate_foreground_dependencies(fork_dependencies, workspace=workspace,
                                         restrict_to_workspace=restrict_to_workspace)
        self._fork_dependencies = fork_dependencies
        self._owned_foreground_manager = fork_dependencies.foreground_manager if _owns_fork_dependencies else None
        self.context = ContextBuilder(workspace, timezone=timezone, disabled_skills=disabled_skills,
                                      context_manager=fork_dependencies.resource_policy.context_manager)
        self.sessions = session_manager or SessionManager(workspace)
        self.approval_admission = ApprovalAdmission()
        self.run_store = RunStore(workspace, self.sessions)
        # One file-read/write tracker per logical session. The tool registry is
        # shared by this loop, so tools resolve the active state via contextvars.
        self._file_state_store = FileStateStore(max_sessions=SESSION_CACHE_MAX_SIZE)
        # SessionManager owns every durable deletion entrypoint, including the
        # WebUI and fork rollback paths.  Observe that boundary once instead of
        # duplicating cleanup in each consumer.
        self.sessions.set_delete_observer(self._file_state_store.discard)
        self.tools = tool_registry if tool_registry is not None else ToolRegistry()
        self._exec_session_manager = ExecSessionManager()
        self._artifact_reader_factory = artifact_reader_factory
        self.runner = AgentRunner(artifact_reader_factory=artifact_reader_factory)
        self.subagents = SubagentManager(
            workspace=workspace,
            bus=bus,
            tools_config=_tc,
            max_tool_result_chars=self.max_tool_result_chars,
            restrict_to_workspace=restrict_to_workspace,
            disabled_skills=disabled_skills,
            max_iterations=self.max_iterations,
            max_concurrent_subagents=max_concurrent_subagents,
            llm_wall_timeout_for_session=lambda sk: runner_wall_llm_timeout_s(self.sessions, sk),
        )
        self._unified_session = unified_session
        self._running = False
        self._closing = False
        self._closed = False
        self._loop_task: asyncio.Task[Any] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._close_initiator: asyncio.Task[Any] | None = None
        self._close_draining_tasks: set[asyncio.Task[Any]] = set()
        self._runtime_context_providers: list[RuntimeContextProvider] = []
        self._active_tasks: dict[str, set[asyncio.Task[Any]]] = {}
        self._discarding_sessions: set[str] = set()
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._close_lock = asyncio.Lock()
        self._session_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        # Per-session pending queues for mid-turn message injection.
        # When a session has an active task, new messages for that session
        # are routed here instead of creating a new task.
        self._pending_queues: dict[str, asyncio.Queue[InboundMessage]] = {}
        self._source_activity_admissions: dict[str, _SourceActivityAdmission] = {}
        # Keep the message object alive while a source cleanup republishes it;
        # object identity is the internal handoff proof for one already-admitted
        # pending message and cannot be forged through inbound metadata.
        self._admitted_activity_messages: dict[int, _AdmittedActivityMessage] = {}
        self._activity_admission_token = object()
        self._pending_republish: dict[str, list[InboundMessage]] = {}
        self._bus_cleanup_tasks: dict[str, set[asyncio.Future[Any]]] = {}
        self._preserve_inflight_turns_on_shutdown = False
        self._deferred_automation_turns: dict[str, list[InboundMessage]] = {}
        self._cron_turns = CronTurnCoordinator(
            publish_inbound=self.bus.publish_inbound,
            dispatch=self._dispatch,
            is_running=lambda: self._running,
            deferred_queues=self._deferred_automation_turns,
        )
        self._local_trigger_turns = LocalTriggerTurnCoordinator(
            publish_inbound=self.bus.publish_inbound,
            dispatch=self._dispatch,
            is_running=lambda: self._running,
            deferred_queues=self._deferred_automation_turns,
        )
        self._automation_turn_coordinators = (
            ("cron", self._cron_turns),
            ("local trigger", self._local_trigger_turns),
        )
        # NANOBOT_MAX_CONCURRENT_REQUESTS: unset or <=0 means unlimited.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "0"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.consolidator = Consolidator(
            store=self.context.memory,
            sessions=self.sessions,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            resolve_prompt_context=PersistedPromptContextResolver(
                workspace_scopes=self.workspace_scopes,
                unified_session=unified_session,
            ),
        )
        self.auto_compact = AutoCompact(
            sessions=self.sessions,
            consolidator=self.consolidator,
            session_ttl_minutes=session_ttl_minutes,
            bind_events=self._idle_events,
        )
        self._idle_compact_check_interval_s = idle_compact_check_interval_seconds
        self._next_idle_compact_check_at = time.monotonic()
        if model_preset:
            self.set_model_preset(model_preset, publish_update=False)
        self._register_default_tools(provider_snapshot_loader=provider_snapshot_loader)
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _ensure_lifecycle_state(self) -> None:
        """Lazily supply lifecycle fields for compatibility test hosts."""

        if not hasattr(self, "_closing"):
            self._closing = False
        if not hasattr(self, "_closed"):
            self._closed = False
        if not hasattr(self, "_loop_task"):
            self._loop_task = None
        if not hasattr(self, "_close_task"):
            self._close_task = None
        if not hasattr(self, "_close_initiator"):
            self._close_initiator = None
        if not hasattr(self, "_close_draining_tasks"):
            self._close_draining_tasks: set[asyncio.Task[Any]] = set()

    def _lifecycle_is_closing(self) -> bool:
        """Read the close fence safely on legacy/object-constructed hosts."""

        close_task = getattr(self, "_close_task", None)
        return bool(
            getattr(self, "_closing", False)
            or getattr(self, "_closed", False)
            or (close_task is not None and not close_task.done())
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        bus: MessageBus | None = None,
        *,
        tool_registry: ToolRegistry,
        **extra: Any,
    ) -> AgentLoop:
        """Create an AgentLoop from config with the common parameter set.

        The tool registry is caller-owned so application composition can share
        it with infrastructure such as an ``MCPProvider``.

        Extra keyword arguments are forwarded to ``AgentLoop.__init__``,
        allowing callers to override or extend the standard config-derived
        parameters (e.g. ``cron_service``, ``session_manager``).
        """
        from nanobot.providers.factory import make_provider

        if bus is None:
            bus = MessageBus()
        defaults = config.agents.defaults
        data_dir = config.runtime_data_dir
        if "session_manager" not in extra:
            extra["session_manager"] = SessionManager(
                config.workspace_path,
                sessions_root=data_dir / "sessions" if data_dir is not None else None,
            )
        provider = extra.pop("provider", None) or make_provider(config)
        resolved = config.resolve_preset()
        model = extra.pop("model", None) or resolved.model
        context_window_tokens = extra.pop("context_window_tokens", None) or resolved.context_window_tokens
        provider_snapshot_loader = extra.pop("provider_snapshot_loader", None)
        preset_snapshot_loader = extra.pop("preset_snapshot_loader", None) or preset_helpers.make_preset_snapshot_loader(
            config,
            provider_snapshot_loader,
        )
        if extra.get("fork_dependencies") is None:
            extra["fork_dependencies"] = build_foreground_dependencies(
                workspace=config.workspace_path,
                runtime=LLMRuntime.capture(provider, model, context_window_tokens=context_window_tokens),
                tools_config=config.tools, restrict_to_workspace=config.tools.restrict_to_workspace,
                config=config,
            )
            extra["_owns_fork_dependencies"] = True
        if config.artifacts.enabled and extra.get("artifact_reader_factory") is None:
            from nanobot.config.paths import get_media_dir
            from nanobot.session.artifact_factory import ConfiguredArtifactReaderFactory

            root = config.artifacts.root
            if root is None:
                if data_dir is None:
                    raise ValueError("Enabled artifacts require a loaded config path or explicit root")
                root = str(data_dir / "context-artifacts")
            dependencies = extra["fork_dependencies"]
            manager = dependencies.resource_policy.context_manager
            sessions = extra["session_manager"]
            session_roots: list[Path] = []
            sessions_dir = getattr(sessions, "sessions_dir", None)
            if isinstance(sessions_dir, Path):
                session_roots.extend((sessions_dir, sessions_dir.parent))
            static_roots = [
                config.workspace_path,
                *session_roots,
                *manager.read_roots(),
                *manager.write_roots(),
                *(target.path for target in manager.target_repos()),
            ]
            extra["artifact_reader_factory"] = ConfiguredArtifactReaderFactory(
                root=root,
                max_blob_bytes=config.artifacts.max_blob_bytes,
                max_namespace_bytes=config.artifacts.max_namespace_bytes,
                static_roots=static_roots,
                media_root=data_dir / "media" if data_dir is not None else get_media_dir(),
            )
        return cls(
            bus=bus,
            provider=provider,
            workspace=config.workspace_path,
            model=model,
            max_iterations=defaults.max_tool_iterations,
            max_concurrent_subagents=defaults.max_concurrent_subagents,
            context_window_tokens=context_window_tokens,
            max_tool_result_chars=defaults.max_tool_result_chars,
            provider_retry_mode=defaults.provider_retry_mode,
            tool_hint_max_length=defaults.tool_hint_max_length,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            channels_config=config.channels,
            timezone=defaults.timezone,
            unified_session=defaults.unified_session,
            disabled_skills=defaults.disabled_skills,
            session_ttl_minutes=defaults.session_ttl_minutes,
            idle_compact_check_interval_seconds=defaults.idle_compact_check_interval_seconds,
            tools_config=config.tools,
            model_presets=preset_helpers.configured_model_presets(config),
            model_preset=defaults.model_preset,
            dream_model_preset=defaults.dream.model_override,
            restart_mode=config.gateway.restart_mode,
            provider_snapshot_loader=provider_snapshot_loader,
            preset_snapshot_loader=preset_snapshot_loader,
            tool_registry=tool_registry,
            **extra,
        )

    def _sync_subagent_runtime_limits(self) -> None:
        """Keep subagent runtime limits aligned with mutable loop settings."""
        self.subagents.max_iterations = self.max_iterations

    def invalidate_runtime_config(self) -> None:
        """Invalidate runtime config for lazy refresh at the next admission."""
        self.runtime_resolver.invalidate()

    def refresh_runtime_config(self) -> LLMRuntime:
        """Refresh runtime config now and publish the canonical selection."""
        self.runtime_resolver.invalidate()
        runtime = self.runtime_resolver.admit()
        self._publish_runtime_selection(runtime)
        return runtime

    def runtime_for_session(
        self,
        session: Session,
        *,
        recover_removed: bool = True,
    ) -> LLMRuntime:
        """Resolve the immutable runtime selected by one session."""
        name = model_preset_from_metadata(session.metadata)
        if name is None:
            return self.llm_runtime()
        try:
            return self.runtime_resolver.resolve_preset(name)
        except KeyError:
            if not recover_removed or name in self.runtime_resolver.model_presets:
                raise
            logger.warning(
                "Session '{}' references removed model preset '{}'; falling back to default",
                session.key,
                name,
            )
            session.metadata.pop(SESSION_MODEL_PRESET_METADATA_KEY, None)
            self.sessions.save(session)
            return self.llm_runtime()

    def set_session_model_preset(
        self,
        session_key: str,
        name: str,
    ) -> LLMRuntime:
        """Validate and persist one session's preset selection."""
        runtime = self.runtime_resolver.resolve_preset(name)
        session = self.sessions.get_or_create(session_key)
        session.metadata[SESSION_MODEL_PRESET_METADATA_KEY] = runtime.model_preset
        if self.sessions.source_backend(session_key) is not None:
            self.sessions.update_session_metadata(session_key, {SESSION_MODEL_PRESET_METADATA_KEY: runtime.model_preset})
        else:
            self.sessions.save(session)
        return runtime

    def _publish_runtime_selection(
        self,
        runtime: LLMRuntime,
        *,
        publish_update: bool = True,
    ) -> None:
        if not publish_update:
            return
        if self._runtime_model_publisher is not None:
            self._runtime_model_publisher(runtime.model, runtime.model_preset)
        self.runtime_event_publisher.runtime_model_changed(
            runtime.model,
            runtime.model_preset,
        )

    def set_model_preset(
        self,
        name: str | None,
        *,
        publish_update: bool = True,
    ) -> LLMRuntime:
        """Select a named default runtime for future turns."""
        old_model = self.model
        runtime = self.runtime_resolver.select_preset(name)
        self._publish_runtime_selection(runtime, publish_update=publish_update)
        logger.info(
            "Runtime model switched for next turn: {} -> {}",
            old_model,
            runtime.model,
        )
        return runtime

    def set_runtime_model(self, model: str) -> LLMRuntime:
        """Select a model on the current provider for future turns."""
        return self.runtime_resolver.select_model(model)

    def set_runtime_context_window(self, context_window_tokens: int) -> LLMRuntime:
        """Select a context limit for future turns."""
        return self.runtime_resolver.select_context_window(context_window_tokens)

    def _register_default_tools(
        self,
        *,
        provider_snapshot_loader: Callable[..., ProviderSnapshot] | None,
    ) -> None:
        """Register the default set of tools via plugin loader."""
        from nanobot.agent.tools.context import ToolContext
        from nanobot.agent.tools.loader import ToolLoader

        ctx = ToolContext(
            config=self.tools_config,
            workspace=str(self.workspace),
            bus=self.bus,
            subagent_manager=self.subagents,
            cron_service=self.cron_service,
            exec_session_manager=self._exec_session_manager,
            sessions=self.sessions,
            provider_snapshot_loader=provider_snapshot_loader,
            image_generation_provider_configs=self._image_generation_provider_configs,
            timezone=self.context.timezone or "UTC",
            workspace_sandbox=self.workspace_scopes.sandbox_status,
            runtime_control=AgentRuntimeControl(self),
            fork_dependencies=self._fork_dependencies,
        )
        loader = ToolLoader()
        registered = loader.load(ctx, self.tools)

        logger.info("Registered {} tools: {}", len(registered), registered)

    def register_runtime_context_provider(
        self,
        provider: RuntimeContextProvider,
    ) -> Callable[[], None]:
        """Register a per-turn context provider and return an unsubscribe callback."""
        if provider in self._runtime_context_providers:
            return lambda: None
        self._runtime_context_providers.append(provider)

        def _unsubscribe() -> None:
            with suppress(ValueError):
                self._runtime_context_providers.remove(provider)

        return _unsubscribe

    async def submit_cron_turn(self, msg: InboundMessage) -> OutboundMessage | None:
        return await self._cron_turns.submit(msg)

    async def submit_local_trigger_turn(self, msg: InboundMessage) -> OutboundMessage | None:
        return await self._local_trigger_turns.submit(msg)

    def pending_cron_job_ids_for_session(self, session_key: str) -> set[str]:
        return self._cron_turns.pending_job_ids_for_session(session_key)

    def pending_local_trigger_ids_for_session(self, session_key: str) -> set[str]:
        return self._local_trigger_turns.pending_trigger_ids_for_session(session_key)

    async def _publish_next_deferred_automation_turn(self, session_key: str) -> None:
        await publish_next_deferred_turn(
            deferred_queues=self._deferred_automation_turns,
            publish_inbound=self.bus.publish_inbound,
            session_key=session_key,
        )

    def _persist_user_message_early(
        self,
        msg: InboundMessage,
        session: Session,
        runtime_context_blocks: list[RuntimeContextBlock] | None = None,
        *,
        run_id: str | None = None,
        trace_owner: TurnContext | None = None,
        **kwargs: Any,
    ) -> bool:
        """Persist the triggering user message before the turn starts.

        Returns True if the message was persisted.
        """
        if not turn_continuation.should_persist_user_message(msg.metadata):
            return False
        media_paths = [
            path
            for path in (msg.media or [])
            if isinstance(cast(object, path), str) and path
        ]
        content_value = cast(object, msg.content)
        has_text = isinstance(content_value, str) and content_value.strip()
        if has_text or media_paths or runtime_context_blocks:
            extra: dict[str, Any] = ({"media": list(media_paths)} if media_paths else {}) | agent_context.session_extra(msg.metadata)
            extra.update(kwargs)
            text = content_value if isinstance(content_value, str) else ""
            text_override, automation_extra = automation_history_overrides(msg.metadata)
            if text_override is not None:
                text = text_override
            extra.update(automation_extra)
            text, runtime_context_meta = append_runtime_context(
                text,
                runtime_context_blocks or (),
            )
            if runtime_context_meta is not None:
                extra[RUNTIME_CONTEXT_HISTORY_META] = runtime_context_meta
            extra.pop("_run_id", None)
            if run_id:
                extra["_run_id"] = run_id
            writer = (trace_owner.run_context.source_writer
                      if trace_owner is not None and trace_owner.run_context is not None else None)
            if writer is not None:
                request = trace_owner.run_context.request
                # Approval expansion/runtime instructions stay in the projection.
                original = request.content if request is not None else text
                _, automation = automation_history_overrides(msg.metadata)
                source_extra = {key: value for key, value in extra.items()
                                if key not in {RUNTIME_CONTEXT_HISTORY_META, "_run_id"}}
                retained_media = list(request.media) if request is not None else media_paths
                entry = writer.append_message({**source_extra, "role": "user", "content": original,
                    **({"media": retained_media} if retained_media else {}), **automation,
                    **({"_command": True} if kwargs.get("_command") else {})})
                trace_owner.run_context.source_input_event_id = entry["_event_id"]
                self._capture_trace_row(trace_owner, entry, is_input=True)
                return True
            session.add_message("user", text, **extra)
            self._capture_trace_row(trace_owner, session.messages[-1], is_input=True)
            self._mark_pending_user_turn(session)
            followup_id = msg.metadata.get(PENDING_FOLLOWUP_ID_KEY)
            if isinstance(followup_id, str) and followup_id:
                acknowledge_pending_followups(session, [followup_id])
            self.sessions.save(session)
            return True
        return False

    def _build_transcript_input(self, ctx: TurnContext) -> TranscriptInput:
        """Capture the persisted history and fresh input as separate transcript parts."""
        assert ctx.session is not None
        return TranscriptInput(
            history=ctx.history,
            current_message=ctx.msg.content,
            media=ctx.msg.media if ctx.kind is TurnKind.USER and ctx.msg.media else None,
            session_summary=ctx.pending_summary,
            runtime_context_blocks=ctx.runtime_context_blocks,
        )

    def _request_context_for_turn(self, ctx: TurnContext) -> RequestContext:
        assert ctx.session is not None
        scope = self.workspace_scopes.for_turn(
            channel=ctx.delivery.route.channel,
            message_metadata=ctx.msg.metadata,
            session_metadata=ctx.session.metadata,
        )
        return RequestContext(
            channel=ctx.delivery.route.channel,
            chat_id=ctx.delivery.route.chat_id,
            message_id=ctx.msg.metadata.get("message_id"),
            session_key=ctx.session_key,
            original_user_text=ctx.original_user_text,
            runtime=ctx.runtime,
            metadata=dict(ctx.msg.metadata or {}),
            attributes=dict(ctx.attributes),
            sender_id=ctx.msg.sender_id,
            turn_id=ctx.turn_id,
            workspace=scope.project_path,
        )

    async def _resolve_runtime_context_for_turn(
        self,
        ctx: TurnContext,
    ) -> list[RuntimeContextBlock]:
        assert ctx.request_context is not None
        return await self._resolve_runtime_context_for_request(
            ctx.request_context,
            ctx.tools if ctx.tools is not None else self.tools,
        )

    async def _resolve_runtime_context_for_request(
        self,
        request: RequestContext,
        tools: ToolRegistry,
    ) -> list[RuntimeContextBlock]:
        providers = [
            *tools.get_runtime_context_providers(),
            *self._runtime_context_providers,
        ]
        blocks = runtime_context_blocks_from_metadata(request.metadata)
        blocks.extend(await resolve_runtime_context(providers, request))
        skill_context = self.context.skills.build_explicit_skill_runtime_context(
            request.original_user_text or ""
        )
        if skill_context is not None and skill_context not in blocks:
            blocks.append(skill_context)
        return blocks

    async def _dispatch_command_inline(
        self,
        msg: InboundMessage,
        key: str,
        raw: str,
        dispatch_fn: Callable[[CommandContext], Awaitable[OutboundMessage | None]],
    ) -> None:
        """Dispatch a command directly from the run() loop and publish the result."""
        if normalize_command_text(raw).lower() == "/compact":
            # Compaction must wait for the active turn to commit its session.
            task = asyncio.create_task(self._dispatch(msg))
            self._track_active_task(key, task)
            # Enter dispatch's cancellation handler before consuming a queued /stop.
            await asyncio.sleep(0)
            return

        async def dispatch_and_publish() -> None:
            ctx = CommandContext(msg=msg, session=None, key=key, raw=raw, loop=self)
            result = await dispatch_fn(ctx)
            if result:
                await self.bus.publish_outbound(result)
            else:
                logger.warning("Command '{}' matched but dispatch returned None", raw)

        # A shell command may run for up to the configured exec timeout. Keep
        # the inbound consumer responsive when it runs beside an active turn.
        if (msg.metadata or {}).get(INBOUND_META_USER_SHELL) is True:
            self.schedule_background(dispatch_and_publish())
            return
        await dispatch_and_publish()

    async def execute_user_shell_command(self, ctx: CommandContext) -> OutboundMessage:
        """Execute one trusted user command with the active workspace policy."""
        metadata = dict(ctx.msg.metadata or {})
        tool = self.tools.get("exec")
        if tool is None:
            content = "Shell execution is disabled in this nanobot configuration."
        else:
            session = ctx.session or self.sessions.get_or_create(ctx.key)
            scope = self.workspace_scopes.for_turn(
                channel=ctx.msg.channel,
                message_metadata=metadata,
                session_metadata=session.metadata,
            )
            request_token = bind_request_context(RequestContext(
                channel=ctx.msg.channel,
                chat_id=ctx.msg.chat_id,
                message_id=metadata.get("message_id"),
                session_key=ctx.key,
                original_user_text=f"!{ctx.args.strip()}",
                runtime=ctx.runtime,
                metadata=metadata,
                sender_id=ctx.msg.sender_id,
                turn_id=metadata.get("webui_turn_id"),
                workspace=scope.project_path,
            ))
            workspace_token = bind_workspace_scope(scope)
            turn_scope_stack = ExitStack()
            try:
                for turn_scope in ctx.turn_scopes:
                    turn_scope_stack.enter_context(turn_scope)
                result = await tool.execute(
                    command=ctx.args.strip(),
                    working_dir=str(scope.project_path),
                )
                content = str(result)
            finally:
                turn_scope_stack.close()
                reset_workspace_scope(workspace_token)
                reset_request_context(request_token)
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=content,
            metadata={**metadata, "render_as": "text"},
        )

    def _track_active_task(self, key: str, task: asyncio.Task[Any]) -> None:
        """Track active session work until its task group becomes empty."""
        tasks = self._active_tasks.setdefault(key, set())
        tasks.add(task)
        task.add_done_callback(partial(self._active_task_done, key, tasks))

    def _active_task_done(
        self,
        key: str,
        tasks: set[asyncio.Task[Any]],
        task: asyncio.Task[Any],
    ) -> None:
        tasks.discard(task)
        if not tasks and self._active_tasks.get(key) is tasks:
            self._active_tasks.pop(key, None)

    async def _cancel_active_tasks(self, key: str) -> int:
        """Cancel and await all active work for *key*.

        Returns the total number of cancelled tasks, subagents, and exec sessions.
        """
        tasks = tuple(self._active_tasks.pop(key, set()))
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await t
        sub_cancelled = await self.subagents.cancel_by_session(key)
        exec_cancelled = await self._exec_session_manager.terminate_by_owner(key)
        return cancelled + sub_cancelled + exec_cancelled

    async def discard_session(self, key: str) -> None:
        """Stop active work for *key* and forget its cached session."""
        self._discarding_sessions.add(key)
        try:
            self.sessions.invalidate(key)
            await self._cancel_active_tasks(key)
        finally:
            self.discard_session_file_state(key)
            self._discarding_sessions.discard(key)

    def discard_session_file_state(self, key: str) -> None:
        """Forget ephemeral file-read state for a reset or removed session."""
        self._file_state_store.discard(key)

    def _effective_session_key(self, msg: InboundMessage) -> str:
        """Return the session key used for task routing and mid-turn injections."""
        if self._unified_session and not msg.session_key_override:
            return UNIFIED_SESSION_KEY
        return msg.session_key

    def _idle_events(
        self,
        session_key: str,
    ) -> EventSink:
        """Bind one idle compaction to its current user-facing destination."""
        session = self.sessions.get_or_create(session_key)
        return self.turn_delivery_factory.session_events(
            session_key, session.metadata,
        )

    def _configured_idle_integration(self) -> ForegroundIdleIntegration | None:
        """Return only a complete, explicitly enabled idle integration.

        The dormant coordinator is intentionally not constructed here.  This
        adapter accepts a caller-owned coordinator (or equivalent) only when
        it exposes the complete enabled flag and foreground notification
        callback.  Missing or partial wiring remains a no-op.  The private
        policy object is never consulted as an activation signal.
        """

        integration = getattr(self, "_idle_integration", None)
        if integration is None:
            return None
        if type(integration) is IdleLifecycleIntegration:
            try:
                if integration.enabled is not True:
                    return None
                callback = integration.foreground_admitted
            except Exception:
                return None
            return integration if callable(callback) else None
        try:
            enabled = getattr(integration, "enabled", None)
            callback = getattr(integration, "foreground_admitted", None)
        except Exception:
            return None
        if enabled is not True or not callable(callback):
            return None
        return cast(ForegroundIdleIntegration, integration)

    async def scan_idle_once(self) -> int:
        """Run one explicitly requested idle scan through injected lifecycle."""

        if self._lifecycle_is_closing():
            return 0
        integration = getattr(self, "_idle_integration", None)
        if integration is None:
            return 0
        try:
            enabled = getattr(integration, "enabled", None)
            scan = getattr(integration, "scan_once", None)
            closed = getattr(integration, "closed", False)
        except Exception:
            return 0
        if enabled is not True or closed is True or not callable(scan):
            return 0
        result = scan()
        if not inspect.isawaitable(result):
            return 0
        try:
            value = await cast(Awaitable[object], result)
        except asyncio.CancelledError:
            raise
        except Exception:
            return 0
        return value if type(value) is int and value >= 0 else 0

    @staticmethod
    def _inbound_turn_source(msg: InboundMessage) -> TurnSource:
        """Classify trusted bus provenance without treating system input as gateway input."""

        if cron_trigger(msg.metadata):
            return TurnSource.CRON
        if local_trigger(msg.metadata) is not None:
            return TurnSource.SYSTEM_COMPAT
        if msg.channel == "system" or not msg.is_user_input:
            return TurnSource.SYSTEM_COMPAT
        return TurnSource.GATEWAY

    @staticmethod
    def _trusted_foreground_inbound(msg: InboundMessage) -> bool:
        """Accept foreground activity only from trusted external user ingress."""

        return (
            msg.channel != "system"
            and msg.input_role != "system"
            and msg.is_user_input
            and not cron_trigger(msg.metadata)
            and local_trigger(msg.metadata) is None
            and RECOVERY_INBOUND_METADATA_KEY not in msg.metadata
            and msg.sender_id != "subagent"
            and not msg.sender_id.startswith("system:")
        )

    async def _record_foreground_activity(
        self,
        context: RunTurnContext,
        session_key: str,
        *,
        source: TurnSource,
        sender_id: str,
        trusted: bool,
    ) -> _ForegroundActivityResult:
        """Persist accepted foreground activity before any turn work starts."""

        if self._lifecycle_is_closing():
            return _ForegroundActivityResult(
                callback_error=RuntimeError("agent loop is closing"),
            )
        if (
            not trusted
            or source not in _FOREGROUND_IDLE_SOURCES
            or sender_id == "subagent"
        ):
            return _ForegroundActivityResult()
        integration = self._configured_idle_integration()
        if integration is None:
            return _ForegroundActivityResult()
        owner = context.execution_owner
        writer = context.source_writer
        # Activity belongs to the strict source substrate.  A legacy session
        # has no current ExecutionOwner and therefore fails closed without a
        # write or callback.
        if owner is None or writer is None:
            return _ForegroundActivityResult()
        owner.validate_dispatch()
        expected = owner.read_snapshot().state
        writer.store.record_user_activity(
            session_key,
            expected,
            accepted_foreground_user=True,
            owner=owner,
        )
        # The source replacement above fsyncs before returning.  Only then may
        # the coordinator cancel/discard dormant work; callback failures
        # propagate and prevent the turn from reaching BUILD/RUN.
        return await self._notify_foreground_activity(
            session_key,
            activity_committed=True,
        )

    async def _notify_foreground_activity(
        self,
        session_key: str,
        *,
        activity_committed: bool,
    ) -> _ForegroundActivityResult:
        """Notify injected idle work after a durable activity commit."""

        if self._lifecycle_is_closing():
            return _ForegroundActivityResult(
                activity_committed=activity_committed,
                callback_error=RuntimeError("agent loop is closing"),
            )
        integration = self._configured_idle_integration()
        if integration is None:
            return _ForegroundActivityResult(activity_committed=activity_committed)
        try:
            await integration.foreground_admitted(session_key)
        except BaseException as exc:
            return _ForegroundActivityResult(
                activity_committed=activity_committed,
                callback_error=exc,
            )
        return _ForegroundActivityResult(
            activity_committed=activity_committed,
            callback_succeeded=True,
        )

    def _remember_admitted_activity_message(
        self,
        msg: InboundMessage,
        *,
        notification_succeeded: bool = True,
    ) -> None:
        """Retain one internal identity marker until notification succeeds."""

        admitted_messages = getattr(self, "_admitted_activity_messages", None)
        if admitted_messages is None:
            admitted_messages = self._admitted_activity_messages = {}
        token = getattr(self, "_activity_admission_token", None)
        if token is not None:
            setattr(msg, _ACTIVITY_ADMITTED_ATTRIBUTE, token)
            setattr(
                msg,
                _ACTIVITY_NOTIFICATION_PENDING_ATTRIBUTE,
                not notification_succeeded,
            )
        admitted_messages[id(msg)] = _AdmittedActivityMessage(
            msg,
            notification_succeeded=notification_succeeded,
        )

    def _take_admitted_activity_message(
        self,
        msg: InboundMessage,
    ) -> _AdmittedActivityMessage | None:
        """Read an identity marker without consuming its retry authority."""

        admitted_messages = getattr(self, "_admitted_activity_messages", {})
        marker = admitted_messages.get(id(msg))
        already_recorded = marker is not None and marker.message is msg
        token = getattr(self, "_activity_admission_token", None)
        if not already_recorded and token is not None:
            already_recorded = (
                getattr(msg, _ACTIVITY_ADMITTED_ATTRIBUTE, None) is token
            )
        if already_recorded:
            pending = bool(
                getattr(msg, _ACTIVITY_NOTIFICATION_PENDING_ATTRIBUTE, False)
            )
            return marker or _AdmittedActivityMessage(
                msg,
                notification_succeeded=not pending,
            )
        return None

    def _ack_admitted_activity_message(self, msg: InboundMessage) -> None:
        """Consume a marker only after its notification callback succeeds."""

        admitted_messages = getattr(self, "_admitted_activity_messages", {})
        marker = admitted_messages.get(id(msg))
        if marker is not None and marker.message is msg:
            admitted_messages.pop(id(msg), None)
        with suppress(AttributeError):
            delattr(msg, _ACTIVITY_ADMITTED_ATTRIBUTE)
        with suppress(AttributeError):
            delattr(msg, _ACTIVITY_NOTIFICATION_PENDING_ATTRIBUTE)

    def _retain_admitted_activity_for_republish(
        self,
        session_key: str,
        msg: InboundMessage,
    ) -> None:
        """Queue the exact committed message for another notification retry."""

        pending_republish = getattr(self, "_pending_republish", None)
        if pending_republish is None:
            pending_republish = self._pending_republish = {}
        items = pending_republish.setdefault(session_key, [])
        if not any(item is msg for item in items):
            items.append(msg)

    def _clear_admitted_activity_messages(self) -> None:
        """Release markers when the loop shuts down without a republished turn."""

        admitted_messages = getattr(self, "_admitted_activity_messages", {})
        pending_republish = getattr(self, "_pending_republish", {})
        messages = [marker.message for marker in admitted_messages.values()]
        for pending_items in pending_republish.values():
            messages.extend(pending_items)
        for msg in messages:
            with suppress(AttributeError):
                delattr(msg, _ACTIVITY_ADMITTED_ATTRIBUTE)
            with suppress(AttributeError):
                delattr(msg, _ACTIVITY_NOTIFICATION_PENDING_ATTRIBUTE)
        admitted_messages.clear()
        pending_republish.clear()

    async def _admit_pending_activity(self, key: str, msg: InboundMessage) -> bool:
        """Use the active source owner to admit one queued user message."""

        admissions = getattr(self, "_source_activity_admissions", {})
        admission = admissions.get(key)
        if admission is None:
            return False
        result = await admission.submit(msg)
        if result.callback_error is not None:
            raise result.callback_error
        return result.callback_succeeded

    def _remember_session_route(
        self,
        session: Session,
        msg: InboundMessage,
        *,
        delivery: TurnDelivery,
        is_user_turn: bool,
    ) -> None:
        """Remember the latest user-facing destination, including channel threads."""
        if (
            not is_user_turn
            or msg.channel in {"cli", "system"}
            or msg.sender_id == "subagent"
        ):
            return
        _, automation_metadata = automation_history_overrides(msg.metadata)
        if automation_metadata:
            return
        delivery.remember_session_route(session.metadata)
        if self._unified_session and session.key == UNIFIED_SESSION_KEY:
            remember_last_channel(session.metadata, msg.channel, msg.chat_id)

    @staticmethod
    def _conversation_state_binding(
        *,
        session: Session | None,
        runtime: LLMRuntime,
        run_context: RunTurnContext,
        workspace: Path | None,
        tools: ToolRegistry,
        ephemeral: bool,
    ) -> ConversationRequestBinding | None:
        """Build the durable continuation scope from admitted, trusted inputs.

        Opaque provider state is valid only for a persisted ordinary session.
        This deliberately excludes fresh, source, and ephemeral runs, as well
        as any admission where the resource-policy or lock fence is incomplete.
        """
        if (
            session is None
            or ephemeral
            or run_context.request is None
            or run_context.request.history_mode is HistoryMode.FRESH
            or run_context.source_writer is not None
            or run_context.execution_owner is not None
            or not session.policy.persist
            or not isinstance(workspace, Path)
            or not isinstance(run_context.policy.workspace, Path)
            or run_context.policy.context_manager is None
            or not session.key
            or not session.incarnation
            or not run_context.lock_owner
            or type(session.generation) is not int
            or type(session.last_archived) is not int
            or not isinstance(runtime.model, str)
            or type(runtime.context_window_tokens) is not int
        ):
            return None
        context_manager = run_context.policy.context_manager
        try:
            read_roots = sorted(
                str(Path(path).expanduser().resolve(strict=False))
                for path in context_manager.read_roots()
            )
            write_roots = sorted(
                str(Path(path).expanduser().resolve(strict=False))
                for path in context_manager.write_roots()
            )
            tool_definitions = deepcopy(tools.get_definitions())
        except Exception:
            # A complete binding is safer than a best-effort one.  The
            # controller will use portable replay when a trusted input cannot
            # be read at admission.
            return None
        if not isinstance(tool_definitions, list):
            return None
        generation = runtime.generation
        workspace_path = str(workspace.expanduser().resolve(strict=False))
        policy_workspace = str(run_context.policy.workspace.expanduser().resolve(strict=False))
        grant = run_context.policy.approval_grant
        authority = {
            "workspace": policy_workspace,
            "read_roots": read_roots,
            "write_roots": write_roots,
            "approval_granted": bool(run_context.policy.approval_granted),
            "approval_grant": {
                "granted": bool(grant.granted) if grant is not None else False,
                "grant_id": grant.grant_id if grant is not None else None,
                "source": grant.source if grant is not None else None,
                "scope": grant.scope if grant is not None else None,
                "expires_at": grant.expires_at if grant is not None else None,
            },
            "allow_side_effects": bool(run_context.policy.allow_side_effects),
            "allow_external": bool(run_context.policy.allow_external),
            "disabled_tools": sorted(session.policy.disabled_tools),
            "policy_type": f"{type(run_context.policy).__module__}.{type(run_context.policy).__qualname__}",
        }
        runtime_prefix = {
            "provider": f"{type(runtime.provider).__module__}.{type(runtime.provider).__qualname__}",
            "model": runtime.model,
            "model_preset": runtime.model_preset,
            "context_window_tokens": runtime.context_window_tokens,
        }
        return ConversationRequestBinding(
            session_namespace="\x1f".join((workspace_path, session.key, session.incarnation)),
            authority_namespace=(
                f"{type(context_manager).__module__}.{type(context_manager).__qualname__}"
            ),
            execution_owner=run_context.lock_owner,
            session_generation=session.generation,
            # ``last_archived`` advances only when the replay projection is
            # replaced.  Ordinary append saves must retain the same scope so
            # the candidate can cover their next user delta.
            projection_epoch=session.last_archived,
            runtime_prefix=runtime_prefix,
            policy_prefix=authority,
            tools=tool_definitions,
            tool_choice={"type": "auto"},
            generation_settings={
                "model": runtime.model,
                "model_preset": runtime.model_preset,
                "provider": runtime_prefix["provider"],
                "temperature": generation.temperature,
                "max_tokens": generation.max_tokens,
                "reasoning_effort": generation.reasoning_effort,
                "context_window_tokens": runtime.context_window_tokens,
            },
        )

    async def _run_agent_loop(
        self,
        transcript_input: TranscriptInput,
        events: EventSink = NO_EVENTS,
        streaming: bool = False,
        *,
        runtime: LLMRuntime,
        session: Session | None = None,
        pending_queue: asyncio.Queue[InboundMessage] | None = None,
        ephemeral: bool = False,
        run_extra_hooks_for_ephemeral: bool = False,
        hooks: list[AgentHook] | None = None,
        hook_factories: list[AgentTurnHookFactory] | None = None,
        turn_scopes: list[AbstractContextManager[Any]] | None = None,
        tools: ToolRegistry | None = None,
        request_context: RequestContext | None = None,
        provider_state: ProviderConversationState | None = None,
        run_context: RunTurnContext | None = None,
    ) -> AgentRunResult:
        """Run the agent iteration loop.

        *events*: scoped operation and output events.
        *streaming*: request incremental output from the runner.
        ``resuming=True`` means the active turn continues. ``merge_next=True`` means
        the next text segment belongs to the same user-visible assistant message.

        Returns the complete result produced by ``AgentRunner``.
        """
        if run_context is None:
            raise ValueError("canonical run context is required for tool authorization")
        if run_context.policy.workspace is None or run_context.policy.context_manager is None:
            raise ValueError("canonical run context is missing workspace policy dependencies")
        fresh_history = (run_context.request is not None
                         and run_context.request.history_mode is HistoryMode.FRESH)
        source_writer = run_context.source_writer
        if fresh_history or source_writer is not None:
            pending_queue = None
            provider_state = None
        tool_policy = RiskyActionPolicy(
            workspace=run_context.policy.workspace,
            approval_granted=bool(run_context.policy.approval_granted),
            context_manager=run_context.policy.context_manager,
        )
        self._sync_subagent_runtime_limits()

        async def _checkpoint(payload: dict[str, Any]) -> None:
            if session is None or fresh_history or source_writer is not None:
                return
            public_payload = dict(payload)
            private_state = public_payload.pop("provider_state", None)
            public_payload.pop(self._PROVIDER_STATE_CHECKPOINT_VERSION_KEY, None)
            if "provider_state" in payload and (
                private_state is None
                or isinstance(private_state, ProviderConversationState)
            ):
                # Checkpoints may retain only a state committed before this
                # run.  A response candidate has no durable authority until
                # _persist_turn writes its transcript in the same fsync save.
                if session.provider_state is not None:
                    public_payload[self._PROVIDER_STATE_CHECKPOINT_VERSION_KEY] = (
                        self._PROVIDER_STATE_CHECKPOINT_VERSION
                    )
            self._set_runtime_checkpoint(session, public_payload)

        async def _drain_pending(
            *,
            limit: int = _MAX_INJECTIONS_PER_TURN,
            first_msg: InboundMessage | None = None,
        ) -> list[dict[str, Any]]:
            """Drain only messages that are already available."""
            if pending_queue is None:
                return []

            async def _to_user_message(pending_msg: InboundMessage) -> dict[str, Any]:
                content = pending_msg.content
                image_paths = pending_msg.media if pending_msg.media else None
                if image_paths:
                    content, image_paths = reference_non_image_attachments(
                        content,
                        image_paths,
                    )
                    image_paths = image_paths or None
                user_content = self.context.build_user_content(
                    content,
                    image_paths=image_paths,
                )
                row: dict[str, Any] = {"role": "user", "content": user_content}
                metadata_value = cast(object, pending_msg.metadata)
                metadata = (
                    pending_msg.metadata
                    if isinstance(metadata_value, dict)
                    else {}
                )
                if pending_msg.is_user_input:
                    scope = self.workspace_scopes.for_turn(
                        channel=pending_msg.channel,
                        message_metadata=metadata,
                        session_metadata=session.metadata if session is not None else None,
                    )
                    pending_request = RequestContext(
                        channel=pending_msg.channel,
                        chat_id=pending_msg.chat_id,
                        message_id=metadata.get("message_id"),
                        session_key=active_session_key,
                        original_user_text=pending_msg.content,
                        runtime=runtime,
                        metadata=dict(metadata),
                        attributes=dict(request_ctx.attributes),
                        sender_id=pending_msg.sender_id,
                        turn_id=request_ctx.turn_id,
                        workspace=scope.project_path,
                    )
                    blocks = await self._resolve_runtime_context_for_request(
                        pending_request,
                        effective_tools,
                    )
                    row["content"], runtime_marker = append_runtime_context(
                        user_content,
                        blocks,
                    )
                    if runtime_marker is not None:
                        row["_meta"] = {
                            RUNTIME_CONTEXT_MESSAGE_META: runtime_marker,
                        }
                if (
                    pending_msg.sender_id == "subagent"
                    and metadata.get("injected_event") == "subagent_result"
                ):
                    subagent_marker: dict[str, Any] = {"kind": "subagent_result"}
                    task_id = metadata.get("subagent_task_id")
                    if isinstance(task_id, str) and task_id:
                        subagent_marker["subagent_task_id"] = task_id
                        row["subagent_task_id"] = task_id
                    row[HIDDEN_HISTORY_META] = subagent_marker
                    row["injected_event"] = "subagent_result"
                followup_id = metadata.get(PENDING_FOLLOWUP_ID_KEY)
                if isinstance(followup_id, str) and followup_id:
                    row[PENDING_FOLLOWUP_ID_KEY] = followup_id
                return row

            items: list[dict[str, Any]] = []
            if first_msg is not None:
                items.append(await _to_user_message(first_msg))
            while len(items) < limit:
                try:
                    items.append(await _to_user_message(pending_queue.get_nowait()))
                except asyncio.QueueEmpty:
                    break

            return items

        terminal_wait_deadline: float | None = None

        async def _wait_for_pending(
            *,
            limit: int = _MAX_INJECTIONS_PER_TURN,
        ) -> list[dict[str, Any]]:
            """Wait for a pending result only when the runner is ready to exit."""
            nonlocal terminal_wait_deadline

            items = await _drain_pending(limit=limit)
            if (
                items
                or pending_queue is None
                or session is None
                or self.subagents.get_running_count_by_session(session.key) == 0
            ):
                return items

            now = asyncio.get_running_loop().time()
            if terminal_wait_deadline is None:
                terminal_wait_deadline = now + _SUBAGENT_TERMINAL_WAIT_SECONDS
            remaining = terminal_wait_deadline - now
            if remaining <= 0:
                return []

            try:
                msg = await asyncio.wait_for(pending_queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout waiting for sub-agent completion before session {} exits",
                    session.key,
                )
                return []

            return await _drain_pending(limit=limit, first_msg=msg)

        request_ctx = request_context or RequestContext(
            channel="cli",
            chat_id="direct",
            session_key=session.key if session is not None else None,
            runtime=runtime,
        )
        active_session_key = session.key if session else request_ctx.session_key
        request_metadata = request_ctx.metadata
        effective_scope = self.workspace_scopes.for_turn(
            channel=request_ctx.channel,
            message_metadata=request_metadata,
            session_metadata=session.metadata if session is not None else None,
        )
        transcript_builder = partial(
            self.context.build_transcript,
            channel=request_ctx.channel,
            workspace=effective_scope.project_path,
            include_memory=session.policy.persist if session is not None else True,
        )
        if request_context is None:
            request_ctx = dataclasses.replace(
                request_ctx,
                workspace=effective_scope.project_path,
            )
        effective_tools = tools if tools is not None else self.tools
        file_state_token = bind_file_states(self._file_state_store.for_session(active_session_key))
        request_token = bind_request_context(request_ctx)
        workspace_token = bind_workspace_scope(effective_scope)
        turn_scope_stack = ExitStack()
        # Compute lazily because create_goal may create goal metadata during this run.
        def _goal_continue() -> str | None:
            _goal_lines = goal_state_runtime_lines(session.metadata if session is not None else None)
            if not _goal_lines:
                return None
            return (
                "You have an active sustained goal:\n\n"
                + "\n".join(_goal_lines)
                + "\n\nPlease continue working toward the objective using your tools, "
                "or call update_goal with action='complete' if the work is truly finished."
            )

        session_metadata = {} if fresh_history else (session.metadata if session is not None else None)
        run_context_token = set_current_turn_context(run_context)
        try:
            for scope in turn_scopes or ():
                turn_scope_stack.enter_context(scope)
            hook = build_agent_turn_hook(AgentTurnHookSpec(
                events=events,
                streaming=streaming,
                channel=request_ctx.channel,
                chat_id=request_ctx.chat_id,
                message_id=request_ctx.message_id,
                metadata=request_metadata,
                attributes=dict(request_ctx.attributes),
                session_key=active_session_key,
                workspace=effective_scope.project_path,
                tool_hint_max_length=self.tool_hint_max_length,
                registered_hook_factories=self._hook_factories,
                turn_hook_factories=list(hook_factories or []),
                registered_hooks=self._extra_hooks,
                turn_hooks=list(hooks or []),
                ephemeral=ephemeral,
                run_extra_hooks_for_ephemeral=run_extra_hooks_for_ephemeral,
            ))
            source_transcript = None
            if source_writer is not None:
                if run_context.source_history is None:
                    raise ValueError("Source run requires explicitly bound history")
                source_transcript = self.context.build_source_transcript(
                    transcript_input, history=run_context.source_history,
                    input_event_id=run_context.source_input_event_id,
                    channel=request_ctx.channel, workspace=effective_scope.project_path,
                    include_memory=session.policy.persist if session is not None else True)
            run_context.artifact_read_allowed = (
                (
                    self._artifact_reader_factory is not None
                    or self.runner.artifact_reader_factory is not None
                )
                and session is not None and 'read_artifact' not in session.policy.disabled_tools
                and 'read_file' in effective_tools.tool_names)
            conversation_binding = self._conversation_state_binding(
                session=session,
                runtime=runtime,
                run_context=run_context,
                workspace=effective_scope.project_path,
                tools=effective_tools,
                ephemeral=ephemeral,
            )
            result = await self.runner.run(AgentRunSpec(
                artifact_read_allowed=run_context.artifact_read_allowed,
                initial_messages=None,
                source_transcript=source_transcript,
                source_isolated=fresh_history,
                source_initial_eligible_ids=frozenset(
                    node.event_id for node in run_context.source_history.nodes if isinstance(node, SourceNode)
                ) if run_context.source_history is not None else frozenset(),
                tools=effective_tools,
                tool_policy=tool_policy,
                turn_context=run_context,
                fail_on_tool_error=True,
                runtime=runtime,
                max_iterations=self.max_iterations,
                max_tool_result_chars=self.max_tool_result_chars,
                transcript_input=transcript_input,
                transcript_builder=transcript_builder,
                hook=hook,
                concurrent_tools=True,
                workspace=effective_scope.project_path,
                session_key=session.key if session else None,
                provider_retry_mode=self.provider_retry_mode,
                checkpoint_callback=_checkpoint,
                source_writer=source_writer,
                consolidate_history=(
                    partial(
                        self.consolidator.summarize_transcript,
                        runtime=runtime,
                        session_key=session.key,
                        tools=effective_tools.get_definitions(),
                    )
                    if session is not None and not ephemeral and not fresh_history and source_writer is None
                    else None
                ),
                consolidate_provider_compaction=(
                    partial(
                        self.consolidator.summarize_provider_compaction,
                        runtime=runtime,
                        session_key=session.key,
                        tools=effective_tools.get_definitions(),
                    )
                    if session is not None and not ephemeral and not fresh_history and source_writer is None
                    else None
                ),
                injection_callback=_drain_pending,
                terminal_injection_callback=_wait_for_pending,
                # Sustained goals may legitimately exceed NANOBOT_LLM_TIMEOUT_S; idle stall
                # is still capped by NANOBOT_STREAM_IDLE_TIMEOUT_S in streaming providers.
                llm_timeout_s=runner_wall_llm_timeout_s(
                    self.sessions,
                    session.key if session is not None else request_ctx.session_key,
                    metadata=session_metadata,
                    message_metadata=request_metadata,
                ),
                continuation_callback=None if fresh_history else _goal_continue,
                finalize_on_max_iterations=turn_continuation.should_finalize_on_max_iterations(
                    pending_queue_available=pending_queue is not None and session is not None,
                    session_metadata=session_metadata,
                    message_metadata=request_metadata,
                ),
                provider_state=provider_state,
                conversation_binding=conversation_binding,
                llm_usage_source=source_from_request(
                    active_session_key,
                    channel=request_ctx.channel,
                    metadata=request_metadata,
                ),
                events=events,
            ))
        finally:
            try:
                turn_scope_stack.close()
            finally:
                reset_current_turn_context(run_context_token)
                reset_workspace_scope(workspace_token)
                reset_request_context(request_token)
                reset_file_states(file_state_token)
        if session is not None and not ephemeral and (fresh_history or source_writer is not None):
            # Fresh and source executions never own a resumable provider
            # continuation, including one left over from an earlier mode.
            session.provider_state = None
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            should_stream = turn_continuation.should_stream_budget_response(
                stop_reason=result.stop_reason,
                pending_queue_available=pending_queue is not None and session is not None,
                session_metadata=session_metadata,
                message_metadata=request_metadata,
            )
            # Push final content through stream so streaming channels (e.g. Feishu)
            # update the card instead of leaving it empty.
            if events.publish is not None and streaming and should_stream:
                stream_content = (
                    result.pending_stream_content
                    if result.pending_stream_content is not None
                    else result.final_content or ""
                )
                await events.publish(StreamDeltaEvent(content=stream_content))
                await events.publish(StreamEndEvent())
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return result

    def _check_expired_sessions_if_due(self) -> None:
        """Scan idle sessions no more often than the configured interval."""
        now = time.monotonic()
        if now < self._next_idle_compact_check_at:
            return
        self._next_idle_compact_check_at = now + self._idle_compact_check_interval_s
        self.auto_compact.check_expired(
            self.schedule_background,
            self.runtime_for_session,
            active_session_keys=self._pending_queues.keys(),
        )

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        if self._lifecycle_is_closing():
            return
        self._loop_task = asyncio.current_task()
        self._running = True
        try:
            logger.info("Agent loop started")

            while self._running:
                try:
                    msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                except asyncio.TimeoutError:
                    self._check_expired_sessions_if_due()
                    continue
                except asyncio.CancelledError:
                    # Preserve real task cancellation so shutdown can complete cleanly.
                    # Only ignore non-task CancelledError signals that may leak from integrations.
                    if not self._running or task_is_cancelling():
                        raise
                    logger.warning(
                        "Ignoring leaked CancelledError while consuming inbound messages"
                    )
                    continue
                except Exception as e:
                    logger.warning("Error consuming inbound message: {}, continuing...", e)
                    continue

                # A concurrent close may have ended the run while this
                # message was waking the consumer.  Recheck the synchronous
                # fence before admitting any new turn.
                if self._lifecycle_is_closing() or not self._running:
                    break

                raw = msg.content.strip()
                effective_key = self._effective_session_key(msg)
                if await agent_context.handle_runtime_control(self, msg, self.tools):
                    continue
                if (
                    msg.require_existing_session
                    and self.sessions.get_cached(effective_key) is None
                ):
                    continue
                if msg.is_user_input:
                    await self.runtime_event_publisher.user_input_accepted(msg, effective_key)
                if msg.channel != "system" and self.commands.is_priority(raw):
                    await self._dispatch_command_inline(
                        msg, effective_key, raw,
                        self.commands.dispatch_priority,
                    )
                    continue
                deferred = False
                for label, coordinator in self._automation_turn_coordinators:
                    if coordinator.defer_if_active(
                        msg,
                        session_key=effective_key,
                        active_session_keys=self._pending_queues.keys(),
                    ):
                        logger.info(
                            "Deferred {} turn for active session {}",
                            label,
                            effective_key,
                        )
                        deferred = True
                        break
                if deferred:
                    continue
                routed_msg = msg
                if effective_key != msg.session_key:
                    routed_msg = dataclasses.replace(
                        msg,
                        session_key_override=effective_key,
                    )
                # A newer WebUI message must supersede an explicit recovery
                # before it is injected into that recovery's pending queue.
                # Without this admission point, a recovered turn could finish
                # first and only then observe the user's newer request.
                if (
                    effective_key in self._pending_queues
                    and msg.channel == "websocket"
                    and self._recovery_admission is not None
                    and not await self._recovery_admission.admit(routed_msg)
                ):
                    continue
                # If this session already has an active pending queue (i.e. a task
                # is processing this session), route the message there for mid-turn
                # injection instead of creating a competing task.
                if effective_key in self._pending_queues:
                    # Non-priority commands must not be queued for injection;
                    # dispatch them directly (same pattern as priority commands).
                    if msg.channel != "system" and self.commands.is_dispatchable_command(raw):
                        await self._dispatch_command_inline(
                            msg, effective_key, raw,
                            self.commands.dispatch,
                        )
                        continue
                    pending_msg = routed_msg
                    session = self.sessions.get_or_create(effective_key)
                    followup_id = record_pending_followup(session, pending_msg)
                    if followup_id is not None:
                        pending_msg = dataclasses.replace(
                            pending_msg,
                            metadata={
                                **pending_msg.metadata,
                                PENDING_FOLLOWUP_ID_KEY: followup_id,
                            },
                        )
                        self.sessions.save(session)
                    try:
                        self._pending_queues[effective_key].put_nowait(pending_msg)
                    except asyncio.QueueFull:
                        logger.warning(
                            "Pending queue full for session {}, falling back to queued task",
                            effective_key,
                        )
                        msg = pending_msg
                    else:
                        try:
                            await self._admit_pending_activity(effective_key, pending_msg)
                        except asyncio.CancelledError:
                            if task_is_cancelling():
                                raise
                            logger.warning(
                                "Pending activity admission was cancelled for {}",
                                effective_key,
                            )
                        except Exception:
                            # Keep the pending message durable and queued.  The
                            # source cleanup republishes it for a fresh owner,
                            # where the normal admission path can retry the
                            # callback before model work begins.
                            logger.exception(
                                "Pending activity admission failed for {}",
                                effective_key,
                            )
                        logger.info(
                            "Routed follow-up message to pending queue for session {}",
                            effective_key,
                        )
                        continue
                # Compute the effective session key before dispatching
                # This ensures /stop command can find tasks correctly when unified session is enabled
                task = asyncio.create_task(self._dispatch(msg))
                self._track_active_task(effective_key, task)
        finally:
            if getattr(self, "_loop_task", None) is asyncio.current_task():
                self._loop_task = None
            await self.aclose()

    def preserve_inflight_turns_on_shutdown(self) -> None:
        """Keep durable checkpoints when the owning gateway exits.

        Normal cancellation intentionally materializes partial output so a
        user-stopped turn leaves a readable conversation.  Gateway lifecycle
        shutdown is different: RecoveryCoordinator needs the checkpoint intact
        to safely offer the unfinished turn for explicit continuation later.
        """
        self._preserve_inflight_turns_on_shutdown = True

    @asynccontextmanager
    async def _bus_turn_scope(self, original: InboundMessage, key: str) -> AsyncIterator[_AdmittedTurnIO]:
        """Own bus delivery and the injection queue inside canonical admission."""
        msg = dataclasses.replace(original, session_key_override=key, metadata=dict(original.metadata))
        delivery = self.turn_delivery_factory.unrouted(msg, key)
        pending: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=20)
        io = _AdmittedTurnIO(pending_queue=pending, delivery=delivery)
        self._pending_queues[key] = pending
        current = asyncio.current_task()
        if current is not None and key != original.session_key:
            self._track_active_task(key, current)
        stage_entered = False

        def notify(error: BaseException | None = None) -> None:
            for _, coordinator in self._automation_turn_coordinators:
                try:
                    if error is not None:
                        coordinator.complete(msg, error=error)
                    else:
                        coordinator.complete(msg, response=io.result.outbound if io.result else None)
                except Exception:
                    logger.warning('Automation completion observer failed for {}', key)

        def continuing() -> bool:
            return turn_continuation.internal_continuation_pending((io.message or msg).metadata)

        async def bounded(awaitable: Awaitable[Any]) -> Any:
            operation = asyncio.ensure_future(awaitable)
            try:
                done, _ = await asyncio.wait({operation}, timeout=_BUS_CLEANUP_TIMEOUT_SECONDS)
                if done:
                    return operation.result()
                raise TimeoutError('bus cleanup deadline exceeded')
            finally:
                if not operation.done():
                    self._bus_cleanup_tasks.setdefault(key, set()).add(operation)
                    operation.cancel()
                    # Do not let cancellation-resistant collaborators hold admission.
                    # Consume eventual errors; admission waits for every late operation.
                    def finished(task: asyncio.Future[Any]) -> None:
                        if not task.cancelled():
                            task.exception()
                    operation.add_done_callback(finished)

        async def cleanup() -> None:
            if self._pending_queues.get(key) is pending:
                self._pending_queues.pop(key)
            items = self._pending_republish.pop(key, [])
            while not pending.empty():
                items.append(pending.get_nowait())
            for index, item in enumerate(items):
                try:
                    self.bus.publish_inbound_nowait(item)
                except Exception:
                    self._pending_republish[key] = items[index:]
                    logger.warning('Retained unpublished pending input for {}', key)
                    break
            try:
                if not continuing():
                    await bounded(delivery.idle())
            except Exception:
                logger.warning('Bus idle publication failed for {}', key)
            try:
                if not continuing():
                    self.runtime_event_publisher.clear_turn(key)
            except Exception:
                logger.warning('Bus turn clearing failed for {}', key)
            try:
                await bounded(self._publish_next_deferred_automation_turn(key))
            except Exception:
                logger.warning('Deferred automation publication failed for {}', key)

        try:
            delivery = self.turn_delivery_factory.create(msg, key, enable_stream=True)
            io.delivery = delivery
            stage_entered = True
            yield io
        except asyncio.CancelledError:
            notify(asyncio.CancelledError())
            try:
                await bounded(delivery.abort_stream())
            except Exception:
                logger.warning('Stream abort failed for {}', key)
            if stage_entered and key not in self._discarding_sessions and not self._preserve_inflight_turns_on_shutdown:
                try:
                    session = self.sessions.get_or_create(key)
                    if restore_runtime_checkpoint(session):
                        self._clear_pending_user_turn(session)
                        self.sessions.save(session)
                except Exception:
                    logger.warning('Cancelled checkpoint restoration failed for {}', key)
            if normalize_command_text(msg.content).lower() == '/compact':
                try:
                    await bounded(delivery.complete(None, publish_completion=True))
                except Exception:
                    logger.warning('Cancelled compact completion failed for {}', key)
            raise
        except Exception as exc:
            notify(exc)
            try:
                await bounded(delivery.fail(publish_completion=not continuing()))
            except Exception:
                logger.warning('Bus failure publication failed for {}', key)
            raise
        else:
            assert io.result is not None
            notify()
            try:
                await delivery.complete(io.result.outbound,
                    publish_completion=not continuing())
            except asyncio.CancelledError:
                io.delivery_error = 'CancelledError'
                try:
                    await bounded(delivery.abort_stream())
                except Exception:
                    logger.warning('Stream abort after delivery cancellation failed for {}', key)
                raise
            except Exception as exc:
                io.delivery_error = type(exc).__name__
                logger.warning('Computed turn delivery failed for {}: {}', key, type(exc).__name__)
        finally:
            # Finish admitted cleanup before releasing its lock, including repeated cancellation.
            task = asyncio.create_task(cleanup())
            cancelled = False
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
            task.result()
            if cancelled:
                raise asyncio.CancelledError()

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Adapt trusted bus ingress to canonical admission without another lock."""
        activity_marker = self._take_admitted_activity_message(msg)
        registration_key = self._effective_session_key(msg)
        msg = dataclasses.replace(msg, session_key_override=registration_key)
        recovery = self._recovery_admission
        task = asyncio.current_task()
        registered = False
        try:
            recovery_id = msg.metadata.get(RECOVERY_INBOUND_METADATA_KEY)
            if recovery is not None:
                if isinstance(recovery_id, str) and recovery_id and task is not None:
                    recovery.register_recovery_task(registration_key, task)
                    registered = True
                if not await recovery.admit(msg):
                    return
            request = TurnRequest(content=msg.content, session_key=registration_key,
                source=self._inbound_turn_source(msg),
                route=DeliveryTarget(channel=msg.channel, chat_id=msg.chat_id,
                                     message_id=msg.metadata.get('message_id')),
                sender_id=msg.sender_id, media=tuple(msg.media), metadata=dict(msg.metadata))
            await self.execute_turn(request, _options=_TurnExecutionOptions(
                inbound=msg, admission_scope=partial(self._bus_turn_scope, msg), raise_errors=True,
                trusted_foreground_activity=self._trusted_foreground_inbound(msg),
                activity_already_recorded=activity_marker is not None,
                retry_activity_notification=(
                    activity_marker is not None
                    and not activity_marker.notification_succeeded
                ),
                admitted_activity_marker=activity_marker,
            ))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception('Bus turn failed for {}', registration_key)
        finally:
            if registered and recovery is not None and task is not None:
                recovery.unregister_recovery_task(registration_key, task)

    async def aclose(self) -> None:
        """Stop active work, then close resources owned by the agent loop.

        A synchronous admission fence is installed before a shared teardown
        task is created.  Waiters are shielded from that task so a cancelled
        caller cannot abandon the cooperative cleanup owned by the loop.
        """
        self._ensure_lifecycle_state()
        current = asyncio.current_task()
        close_task = getattr(self, "_close_task", None)
        if current is close_task or (
            current is not None
            and current in self._close_draining_tasks
        ):
            # The close operation may be waiting for this active/background
            # task to finish.  A reentrant close from its finally block must
            # not await the operation that is waiting on it.
            return
        self._closing = True
        self._running = False
        if close_task is None:
            self._close_initiator = current
            draining = self._close_draining_tasks
            active_groups = getattr(self, "_active_tasks", {})
            draining.update(
                task
                for tasks in active_groups.values()
                for task in tasks
                if not task.done()
            )
            draining.update(
                task
                for task in getattr(self, "_background_tasks", set())
                if not task.done()
            )
            self._close_task = asyncio.create_task(self._aclose_operation())
            close_task = self._close_task
        task = close_task
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # The shared task is shielded and remains owned by the loop;
                # let a bounded gateway/SDK waiter return while a later owner
                # can still join the same cleanup operation.
                raise
            except BaseException:
                # Inspect the shared task below so repeated callers do not
                # rerun the already-owned cleanup operation.
                break
        if task.cancelled():
            raise asyncio.CancelledError
        error = task.exception()
        if error is not None:
            raise error

    async def _aclose_operation(self) -> None:
        """Run the one shared cooperative teardown operation."""

        errors: list[BaseException] = []
        active_task_groups = getattr(self, "_active_tasks", {})
        active_tasks = tuple({task for tasks in active_task_groups.values() for task in tasks})
        active_task_groups.clear()
        current_task = asyncio.current_task()
        initiating_task = getattr(self, "_close_initiator", None)
        loop_task = getattr(self, "_loop_task", None)
        draining = getattr(self, "_close_draining_tasks", set())
        active_tasks = tuple(
            task for task in active_tasks
            if task is not current_task
            and task is not initiating_task
            and task is not loop_task
            and not task.done()
        )
        draining.update(active_tasks)
        for task in active_tasks:
            task.cancel()
        try:
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            background_tasks = tuple(
                task for task in getattr(self, "_background_tasks", set())
                if task is not current_task
                and task is not initiating_task
                and not task.done()
            )
            draining.update(background_tasks)
            if background_tasks:
                for task in background_tasks:
                    task.cancel()
                await asyncio.gather(*background_tasks, return_exceptions=True)
        except BaseException as exc:
            errors.append(exc)
        finally:
            getattr(self, "_background_tasks", set()).clear()
        self._clear_admitted_activity_messages()

        idle_integration = getattr(self, "_idle_integration", None)
        close_idle = (
            getattr(idle_integration, "aclose", None)
            if type(idle_integration) is IdleLifecycleIntegration
            else None
        )
        if callable(close_idle):
            try:
                result = close_idle()
                if inspect.isawaitable(result):
                    await cast(Awaitable[object], result)
            except BaseException as exc:
                errors.append(exc)

        owned_foreground = getattr(self, "_owned_foreground_manager", None)
        self._owned_foreground_manager = None
        cleanup_steps = (
            *((owned_foreground.close,) if owned_foreground is not None else ()),
            self.subagents.close,
            self._exec_session_manager.close_all,
        )
        for cleanup in cleanup_steps:
            try:
                await cleanup()
            except BaseException as exc:
                errors.append(exc)
        self._closed = True
        draining.clear()
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("failed to close agent resources", errors)

    def schedule_background(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    @asynccontextmanager
    async def _admit_after_bus_cleanup(self, key: str) -> AsyncIterator[None]:
        """Wait outside admission while an earlier lifecycle operation can act.

        Keep the caller's original approval snapshot: replaying raw inbound text
        after this wait could approve a different, newer pending request.
        asyncio.wait does not cancel these owner-retained futures if the waiter
        is cancelled. Only the admitted owner clears a settled fence.
        """
        lock = self._get_session_lock(key)
        while True:
            async with lock:
                pending = {task for task in self._bus_cleanup_tasks.get(key, ())
                           if not task.done()}
                if not pending:
                    self._bus_cleanup_tasks.pop(key, None)
                    async with self._concurrency_gate or nullcontext():
                        yield
                    return
            await asyncio.wait(pending)

    @staticmethod
    def _request_route(request: TurnRequest) -> tuple[str, DeliveryTarget]:
        target = request.delivery_target or request.route
        channel = target.channel if target else ''
        chat_id = target.chat_id if target else ''
        key = request.session_key or (f'{channel}:{chat_id}' if channel and chat_id else 'cli:direct')
        if not channel or not chat_id:
            channel, separator, chat_id = key.partition(':')
            if not separator:
                channel, chat_id = 'cli', key
            target = DeliveryTarget(channel=channel, chat_id=chat_id,
                                    message_id=target.message_id if target else None)
        assert target is not None
        return key, target

    @asynccontextmanager
    async def _source_execution(self, key: str, context: RunTurnContext) -> AsyncIterator[None]:
        backend = self.sessions.source_backend(key)
        if backend is None:
            yield
            return
        async with acquire_execution(backend, key, run_id=context.run_id) as owner:
            # Recovery may only enter the explicit reset route. Never consume an
            # approval or request a model while uncertain effects remain.
            reset = (context.request is not None and context.request.source not in (TurnSource.CRON, TurnSource.HEARTBEAT)
                     and context.request.content.strip().lower() == "/new")
            if owner.recovery_required and not reset:
                raise RuntimeError("Source session requires recovery before execution")
            context.execution_owner = owner
            context.source_writer = SourceTurnWriter(backend.source, owner, run_id=context.run_id)
            activity_queue: asyncio.Queue[_ActivityAdmissionRequest | None] = asyncio.Queue()

            async def admit_pending_activity() -> None:
                while True:
                    item = await activity_queue.get()
                    if item is None:
                        return
                    try:
                        outcome = await self._record_foreground_activity(
                            context,
                            key,
                            source=self._inbound_turn_source(item.message),
                            sender_id=item.message.sender_id,
                            trusted=self._trusted_foreground_inbound(item.message),
                        )
                    except BaseException as exc:
                        outcome = _ForegroundActivityResult(callback_error=exc)
                    if outcome.activity_committed:
                        # Mark from the owner-registered worker itself so
                        # cancellation of the submitting bus task cannot
                        # lose the duplicate-suppression proof.
                        self._remember_admitted_activity_message(
                            item.message,
                            notification_succeeded=outcome.callback_succeeded,
                        )
                    if not item.result.done():
                        item.result.set_result(outcome)

            activity_task = asyncio.create_task(admit_pending_activity())
            owner.track_task(activity_task)
            activity_admission = _SourceActivityAdmission(activity_queue, activity_task)
            self._source_activity_admissions[key] = activity_admission
            try:
                yield
            finally:
                if self._source_activity_admissions.get(key) is activity_admission:
                    self._source_activity_admissions.pop(key, None)
                await activity_admission.close()
                self.sessions.invalidate(key)

    async def execute_turn(self, request: TurnRequest, *, _tools: ToolRegistry | None = None,
                           _options: _TurnExecutionOptions | None = None) -> TurnResult:
        """Admit one serialized canonical turn into the upstream stage pipeline."""
        if not isinstance(request, TurnRequest):
            raise TypeError('execute_turn expects a TurnRequest')
        if self._lifecycle_is_closing():
            raise RuntimeError("agent loop is closing")
        options = _options or _TurnExecutionOptions()
        deps = self._fork_dependencies
        if deps is None:
            raise ValueError('canonical admission requires foreground dependencies')
        key, target = self._request_route(request)
        snapshot = await self.approval_admission.snapshot(key, request.content)
        key = snapshot.effective_key
        run_id = request.run_id
        if not run_id or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', run_id):
            run_id = uuid.uuid4().hex
        grant = request.approval_grant or request.approval
        context = RunTurnContext(
            request=request, run_id=run_id, session_key=key,
            delegation_budget=deps.foreground_manager.new_budget(),
            policy=RunPolicyContext(workspace=deps.resource_policy.workspace,
                context_manager=deps.resource_policy.context_manager,
                approval_grant=grant, approval_granted=bool(grant.granted), source=request.source),
            delivery=DeliveryState(primary=target), lock_owner=f'main:{key}',
            metadata=dict(request.metadata),
        )
        context.trusted_foreground_activity = (
            options.trusted_foreground_activity
            and request.source in _FOREGROUND_IDLE_SOURCES
            and request.sender_id != "subagent"
        )
        record = RunRecord(run_id=run_id, source=request.source,
            session_ref=SessionRunRef(session_key=key, run_id=run_id),
            visible_session_key=(request.scheduled_link.visible_session_key
                if request.scheduled_link and request.scheduled_link.visible_session_key
                else f'{target.channel}:{target.chat_id}'),
            resumed_run_id=grant.resumed_run_id, metadata=dict(request.metadata),
            delivery_target=target, scheduled_link=request.scheduled_link)
        record.metadata.pop("source_stream_id", None)
        self.run_store.save(record)
        result: TurnResult | None = None
        try:
            async with self._admit_after_bus_cleanup(key), self._source_execution(key, context):
                if context.source_writer is not None:
                    record.metadata["source_stream_id"] = context.source_writer.stream_id
                try:
                    scope = options.admission_scope(key) if options.admission_scope else nullcontext(_AdmittedTurnIO())
                    async with scope as io:
                        action = await self.approval_admission.consume(snapshot)
                        activity_marker = options.admitted_activity_marker
                        if activity_marker is not None:
                            if not activity_marker.notification_succeeded:
                                activity = await self._notify_foreground_activity(
                                    key,
                                    activity_committed=True,
                                )
                                if activity.callback_error is not None:
                                    self._retain_admitted_activity_for_republish(
                                        key,
                                        activity_marker.message,
                                    )
                                    raise activity.callback_error
                            self._ack_admitted_activity_message(activity_marker.message)
                        if action in ('negative', 'stale'):
                            text = ('Cancelled the pending risky action.' if action == 'negative' else
                                    'That approval request is no longer current. Please reply to the latest approval request.')
                            result = TurnResult(run_id=run_id, status=RunStatus.CANCELLED,
                                content=text, stop_reason='cancelled' if action == 'negative' else 'approval_stale',
                                outbound=OutboundMessage(channel=target.channel, chat_id=target.chat_id or '', content=text))
                        else:
                            if not options.activity_already_recorded:
                                activity = await self._record_foreground_activity(
                                    context,
                                    key,
                                    source=request.source,
                                    sender_id=request.sender_id,
                                    trusted=context.trusted_foreground_activity,
                                )
                                if activity.callback_error is not None:
                                    raise activity.callback_error
                            elif options.retry_activity_notification and activity_marker is None:
                                activity = await self._notify_foreground_activity(
                                    key,
                                    activity_committed=True,
                                )
                                if activity.callback_error is not None:
                                    raise activity.callback_error
                            content = request.content
                            if action == 'affirmative' and snapshot.pending is not None:
                                context.policy.approval_granted = True
                                content = ('The user approved the previously blocked risky action. '
                                    f'Resume the task that required approval: {snapshot.pending.summary}.'
                                    f'\n\nOriginal approval reply: {request.content}')
                            routing_metadata = dict(request.metadata)
                            if target.message_id is not None:
                                routing_metadata["message_id"] = target.message_id
                            envelope = options.inbound or InboundMessage(channel=target.channel,
                                chat_id=target.chat_id or '', sender_id=request.sender_id, content=content)
                            msg = dataclasses.replace(envelope, channel=target.channel, chat_id=target.chat_id or '',
                                content=content, media=list(request.media), metadata=routing_metadata, session_key_override=key)
                            effective_tools = _tools if _tools is not None else (
                                self.tools.filtered(list(request.tool_names)) if request.tool_names is not None else self.tools)
                            io.message = msg
                            stage = await self._process_turn(msg, session_key=key, run_context=context,
                                delivery=io.delivery, pending_queue=io.pending_queue,
                                tools=effective_tools, hooks=list(request.hooks),
                                on_progress=request.callbacks.on_progress, on_stream=request.callbacks.on_stream,
                                on_stream_end=request.callbacks.on_stream_end,
                                ephemeral=options.ephemeral,
                                run_extra_hooks_for_ephemeral=options.run_extra_hooks_for_ephemeral,
                                hook_factories=list(options.hook_factories), runtime=options.runtime,
                                on_runtime_admitted=options.on_runtime_admitted, attributes=dict(options.attributes))
                            runner = stage.run_result
                            status = RunStatus.COMMAND if runner is None else {
                                'completed': RunStatus.COMPLETED, 'approval_required': RunStatus.APPROVAL_REQUIRED,
                                'policy_blocked': RunStatus.POLICY_BLOCKED, 'tool_error': RunStatus.TOOL_ERROR,
                                'max_iterations': RunStatus.MAX_ITERATIONS, 'cancelled': RunStatus.CANCELLED,
                            }.get(runner.stop_reason, RunStatus.ERROR)
                            result = TurnResult(run_id=run_id, status=status,
                                content=runner.final_content if runner else (stage.outbound.content if stage.outbound else None),
                                stop_reason=runner.stop_reason if runner else 'command', outbound=stage.outbound,
                                tools_used=list(runner.tools_used) if runner else [],
                                policy_metadata=dict(runner.policy_metadata) if runner else {},
                                error=runner.error if runner else None,
                                usage={k: v for k, v in runner.usage.to_dict().items()
                                       if isinstance(v, int) and not isinstance(v, bool)} if runner and runner.usage else {},
                                sent_messages=tuple(context.delivery.sent_messages),
                                llm_usage=runner.usage if runner else None)
                            if status is RunStatus.APPROVAL_REQUIRED:
                                await self.approval_admission.store(session_key=key, channel=target.channel,
                                    chat_id=target.chat_id or '', summary=str(result.policy_metadata.get('summary') or 'risky action'))
                        io.result = result
                    if io.delivery_error:
                        record.metadata['delivery_error_type'] = io.delivery_error
                finally:
                    if options.on_admission_end is not None:
                        try:
                            await options.on_admission_end(key)
                        except Exception:
                            logger.warning('Turn admission cleanup failed for run {}', run_id)
        except asyncio.CancelledError:
            context.cancelled = True
            if result is None:
                result = TurnResult(run_id=run_id, status=RunStatus.CANCELLED, stop_reason='cancelled',
                                    sent_messages=tuple(context.delivery.sent_messages))
            else:
                record.metadata['delivery_error_type'] = 'CancelledError'
            raise
        except Exception as exc:
            result = TurnResult(run_id=run_id, status=RunStatus.ERROR, stop_reason='error',
                error=str(exc), sent_messages=tuple(context.delivery.sent_messages))
            if options.raise_errors:
                raise
        finally:
            if result is not None:
                if request.trace_mode is TraceMode.SANITIZED:
                    result.messages = context.sanitized_messages or []
                record.status, record.stop_reason = result.status, result.stop_reason
                record.tools_used = tuple(result.tools_used)
                record.usage = dict(result.usage)
                record.error_type = result.status.value if result.error else None
                record.completed_at = record.updated_at = datetime.now(timezone.utc).isoformat()
                result.record = self.run_store.save(record)
        assert result is not None
        callback = request.callbacks.on_error if result.error else request.callbacks.on_complete
        if callback is not None:
            try:
                returned = callback(result)
                if hasattr(returned, '__await__'):
                    await returned
            except Exception:
                logger.warning('Canonical turn callback failed after run {}', run_id)
        return result

    async def _process_turn(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        pending_queue: asyncio.Queue[InboundMessage] | None = None,
        ephemeral: bool = False,
        run_extra_hooks_for_ephemeral: bool = False,
        hooks: list[AgentHook] | None = None,
        hook_factories: list[AgentTurnHookFactory] | None = None,
        tools: ToolRegistry | None = None,
        runtime: LLMRuntime | None = None,
        delivery: TurnDelivery | None = None,
        on_runtime_admitted: Callable[[LLMRuntime], Awaitable[None]] | None = None,
        attributes: Mapping[str, Any] | None = None,
        run_context: RunTurnContext | None = None,
    ) -> TurnContext:
        """Process a single inbound message and return the response."""
        kind = TurnKind.USER if msg.is_user_input else TurnKind.SYSTEM
        if kind is TurnKind.SYSTEM:
            destination = (
                msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
            )
            key = session_key or msg.session_key_override or f"{destination[0]}:{destination[1]}"
        else:
            key = session_key or msg.session_key
        if delivery is None:
            delivery = self.turn_delivery_factory.create(msg, key)
        elif delivery.session_key != key:
            raise ValueError("turn delivery session does not match the processing session")
        t0 = time.time()
        ctx = TurnContext(
            msg=msg,
            run_context=run_context,
            session=None,
            session_key=key,
            turn_id=f"{key}:{time.time_ns()}",
            runtime=runtime,
            kind=kind,
            delivery=delivery,
            original_user_text=(
                None
                if kind is TurnKind.SYSTEM
                or turn_continuation.internal_continuation_inbound(msg.metadata)
                else msg.content
            ),
            turn_wall_started_at=t0,
            visible_run_started_at=turn_continuation.internal_continuation_run_started_at(
                msg.metadata,
            ),
            events=output_events(
                default=delivery.events,
                on_progress=on_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
            ),
            streaming=on_stream is not None or delivery.streaming,
            on_runtime_admitted=on_runtime_admitted,
            pending_queue=pending_queue,
            ephemeral=ephemeral,
            run_extra_hooks_for_ephemeral=run_extra_hooks_for_ephemeral,
            hooks=list(hooks or []),
            hook_factories=list(hook_factories or []),
            tools=tools,
            attributes=dict(attributes or {}),
        )
        # A streaming callback may be present even when the final text comes from a
        # non-streaming recovery. Only the last completed segment can suppress the
        # regular outbound message.
        if ctx.streaming:
            publish = ctx.events.publish
            segment_streamed_content = False

            async def track_output(event: AgentEvent) -> None:
                nonlocal segment_streamed_content
                if isinstance(event, StreamDeltaEvent) and event.content:
                    segment_streamed_content = True
                elif isinstance(event, StreamEndEvent):
                    ctx.streamed_content = segment_streamed_content
                    segment_streamed_content = False
                if publish is not None:
                    await publish(event)

            ctx.events = EventSink(track_output, ctx.events.accepts)

        try:
            await self._run_turn_stage(ctx, "restore", self._restore_turn)
            await self._run_turn_stage(ctx, "compact", self._compact_session)
            if await self._run_turn_stage(ctx, "command", self._dispatch_command):
                return ctx
            await self._run_turn_stage(ctx, "build", self._build_turn)
            await self._run_turn_stage(ctx, "run", self._run_turn)
            await self._run_turn_stage(ctx, "save", self._persist_turn)
            await self._run_turn_stage(ctx, "respond", self._prepare_outbound)
            return ctx
        finally:
            if run_context is not None and run_context.request is not None and run_context.request.trace_mode is TraceMode.SANITIZED:
                rows = []
                source_writer = run_context.source_writer
                trace_rows = source_writer.committed_events if source_writer is not None else ctx.trace_detail_messages
                for row in trace_rows:
                    if row is ctx.trace_input_message:
                        # Replace exactly this invocation's triggering input.
                        row = {"role": row["role"], "content": run_context.request.content}
                    rows.append(row)
                run_context.sanitized_messages = sanitize_trace(rows)

    @staticmethod
    def _capture_trace_row(
        ctx: TurnContext | None, row: dict[str, Any], *, is_input: bool = False,
    ) -> None:
        """Capture exact appended rows before a fallible save, only when requested."""
        if ctx is None or ctx.run_context is None or ctx.run_context.request is None:
            return
        if ctx.run_context.request.trace_mode is not TraceMode.SANITIZED:
            return
        ctx.trace_detail_messages.append(row)
        if is_input:
            ctx.trace_input_message = row


    async def _run_turn_stage(
        self,
        ctx: TurnContext,
        name: str,
        handler: Callable[[TurnContext], Awaitable[_T]],
    ) -> _T:
        started_at = time.perf_counter()
        try:
            result = await handler(ctx)
        except Exception:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.debug(
                "[turn {}] Stage {} failed after {:.1f}ms",
                ctx.turn_id,
                name,
                duration_ms,
            )
            raise
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.debug(
            "[turn {}] Stage {} completed in {:.1f}ms",
            ctx.turn_id,
            name,
            duration_ms,
        )
        return result

    def _assemble_outbound(
        self,
        msg: InboundMessage,
        final_content: str,
        stop_reason: str,
        streamed_content: bool,
        *,
        log_content: bool = True,
        turn_latency_ms: int | None = None,
    ) -> OutboundMessage | None:
        """Assemble the final outbound message from turn results."""
        if log_content:
            preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
            logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        else:
            logger.info("Response to {}:{}: [content hidden]", msg.channel, msg.sender_id)

        event = None
        meta = dict(msg.metadata or {})
        if streamed_content and stop_reason not in {"error", "tool_error"}:
            event = StreamedResponseEvent()
        if turn_latency_ms is not None:
            meta["latency_ms"] = int(turn_latency_ms)

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            event=event,
            metadata=meta,
        )

    async def _restore_turn(self, ctx: TurnContext) -> None:
        """Restore checkpoint / pending user turn; reference non-image attachments."""
        msg = ctx.msg

        if ctx.kind is TurnKind.USER and msg.media:
            new_content, image_paths = reference_non_image_attachments(
                msg.content,
                msg.media,
            )
            ctx.msg = dataclasses.replace(msg, content=new_content, media=image_paths)
            msg = ctx.msg

        writer = ctx.run_context.source_writer if ctx.run_context is not None else None
        if writer is not None:
            admitted = ctx.session or self.sessions.get_cached(ctx.session_key)
            ctx.session = writer.owner.read_snapshot().to_session()
            if admitted is not None:
                ctx.session.policy = admitted.policy
            ctx.source_metadata_base = deepcopy(ctx.session.metadata)
        if ctx.session is None:
            if msg.require_existing_session:
                ctx.session = self.sessions.get_cached(ctx.session_key)
                if ctx.session is None:
                    raise RuntimeError("required session is not active")
            else:
                ctx.session = self.sessions.get_or_create(ctx.session_key)
        session = ctx.session
        ctx.ephemeral = ctx.ephemeral or not session.policy.persist
        tools = ctx.tools if ctx.tools is not None else self.tools
        if session.policy.disabled_tools:
            restricted = ToolRegistry()
            for name in tools.tool_names:
                tool = tools.get(name)
                if name not in session.policy.disabled_tools and tool:
                    restricted.register(tool)
            tools = restricted
        ctx.tools = tools

        if ctx.kind is TurnKind.SYSTEM:
            logger.info("Processing system message from {}", msg.sender_id)
        elif session.policy.log_content:
            preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)
        else:
            logger.info("Processing message from {}:{}: [content hidden]", msg.channel, msg.sender_id)

        self._remember_session_route(
            session,
            msg,
            delivery=ctx.delivery,
            is_user_turn=ctx.original_user_text is not None,
        )
        await ctx.delivery.started()
        if ctx.kind is TurnKind.USER:
            self.workspace_scopes.persist_message_scope(session, msg)

        if writer is not None or ctx.fresh_history:
            return
        if restore_runtime_checkpoint(session):
            self.sessions.save(session)
        if (
            RECOVERY_INBOUND_METADATA_KEY not in msg.metadata
            and restore_pending_interruption(session)
        ):
            self.sessions.save(session)

    async def _compact_session(self, ctx: TurnContext) -> None:
        if ctx.fresh_history or (ctx.run_context is not None and ctx.run_context.source_writer is not None):
            ctx.pending_summary = None
            return
        session = ctx.require_session()
        ctx.session, pending = self.auto_compact.prepare_session(
            session,
            ctx.session_key,
        )
        ctx.pending_summary = pending

    async def _dispatch_command(self, ctx: TurnContext) -> bool:
        request = ctx.run_context.request if ctx.run_context is not None else None
        if request is not None and request.source in (TurnSource.CRON, TurnSource.HEARTBEAT):
            return False
        if ctx.kind is TurnKind.SYSTEM or ctx.msg.channel == "system":
            return False
        if ctx.run_context is not None and ctx.run_context.source_writer is not None:
            return await self._dispatch_source_command(ctx)
        session = ctx.require_session()
        raw = ctx.msg.content.strip()
        _, automation_metadata = automation_history_overrides(ctx.msg.metadata)
        is_user_turn = (
            ctx.original_user_text is not None
            and not automation_metadata
            and ctx.msg.channel != "system"
            and ctx.msg.sender_id != "subagent"
        )
        cmd_ctx = CommandContext(
            msg=ctx.msg,
            session=session,
            key=ctx.session_key,
            raw=raw,
            loop=self,
            runtime=ctx.runtime,
            is_user_turn=is_user_turn,
            turn_scopes=ctx.turn_scopes,
        )
        result = await self.commands.dispatch(cmd_ctx)
        if cmd_ctx.raw.lower() == "/compact":
            # Compaction reports through events, not an archiveable command reply.
            return True
        if result is not None:
            ctx.outbound = result
            # Shortcut commands skip BUILD and SAVE, so we must persist the
            # turn here so WebUI history hydration after _turn_end sees the
            # message.  Mark messages with _command so get_history can filter
            # them out of LLM context.  /new is excluded because it
            # intentionally clears the session.
            if cmd_ctx.raw.lower() != "/new":
                ctx.input_persisted_early = self._persist_user_message_early(
                    ctx.msg, session, _command=True,
                    run_id=ctx.run_context.run_id if ctx.run_context else None,
                    trace_owner=ctx,
                )
                session.add_message(
                    "assistant", result.content, _command=True,
                    **({"_run_id": ctx.run_context.run_id} if ctx.run_context and ctx.run_context.run_id else {})
                )
                self._capture_trace_row(ctx, session.messages[-1])
                self._clear_pending_user_turn(session)
                self.sessions.save(session)
                if not ctx.ephemeral:
                    await self.runtime_event_publisher.session_turn_persisted(
                        ctx.msg,
                        ctx.session_key,
                        turn_id=ctx.turn_id,
                        attributes=ctx.attributes,
                    )
            return True
        return False

    async def _dispatch_source_command(self, ctx: TurnContext) -> bool:
        assert ctx.run_context is not None and ctx.run_context.source_writer is not None
        raw = normalize_command_text(ctx.msg.content).strip()
        if not raw.startswith('/'):
            return False
        writer = ctx.run_context.source_writer
        command = raw.split(maxsplit=1)[0].lower()
        if raw.lower() == '/new':
            await writer.owner.quiesce()
            writer.store.reset_source_session(writer.owner.key, writer.owner.read_snapshot().state, [], owner=writer.owner)
            self.discard_session_file_state(ctx.session_key)
            self.sessions.invalidate(ctx.session_key)
            text = 'New session started.'
        else:
            writer.owner.validate_dispatch()
            if command in {'/help', '/model', '/history', '/status'}:
                result = await self.commands.dispatch(CommandContext(
                    msg=ctx.msg, session=ctx.session, key=ctx.session_key, raw=raw,
                    loop=self, runtime=ctx.runtime, is_user_turn=True, turn_scopes=ctx.turn_scopes))
                text = result.content if result is not None else 'Command unavailable.'
            elif command == '/compact':
                text = 'Source checkpoint compaction is not available in this migration candidate yet.'
            else:
                text = 'This command is not available for source sessions in this migration candidate yet.'
            self._persist_user_message_early(ctx.msg, ctx.require_session(), trace_owner=ctx, _command=True)
            writer.append_message({'role': 'assistant', 'content': text, '_command': True})
        ctx.outbound = OutboundMessage(channel=ctx.msg.channel, chat_id=ctx.msg.chat_id, content=text,
                                       metadata=dict(ctx.msg.metadata))
        return True

    async def _build_source_turn(self, ctx: TurnContext) -> None:
        assert ctx.run_context is not None and ctx.run_context.source_writer is not None
        writer = ctx.run_context.source_writer
        session = ctx.require_session()
        runtime = ctx.runtime or self.runtime_for_session(session)
        ctx.runtime = runtime
        if ctx.on_runtime_admitted is not None:
            await ctx.on_runtime_admitted(runtime)
        nodes = () if ctx.fresh_history else source_nodes(writer.owner.read_snapshot().events)
        ctx.run_context.source_history = SourceTranscript(nodes)
        ctx.history = ctx.run_context.source_history.messages()
        ctx.pending_summary = None
        ctx.provider_state = None
        await ctx.delivery.runtime_admitted(runtime)
        ctx.request_context = self._request_context_for_turn(ctx)
        ctx.runtime_context_blocks = await self._resolve_runtime_context_for_turn(ctx)
        ctx.input_persisted_early = self._persist_user_message_early(
            ctx.msg, session, run_id=ctx.run_context.run_id, trace_owner=ctx)
        # Keep the full recovery history until the runner has admitted its
        # invocation-specific artifact verifier and checkpoint dependencies.
        if not ctx.fresh_history:
            ctx.run_context.source_history = SourceTranscript(tuple(
                node for node in source_nodes(writer.owner.read_snapshot().events)
                if node.event_id != ctx.run_context.source_input_event_id
            ))
            ctx.history = ctx.run_context.source_history.messages()
        ctx.transcript_input = self._build_transcript_input(ctx)

    async def _build_turn(self, ctx: TurnContext) -> None:
        if ctx.run_context is not None and ctx.run_context.source_writer is not None:
            await self._build_source_turn(ctx)
            return
        session = ctx.require_session()
        runtime = ctx.runtime
        if runtime is None:
            runtime = self.runtime_for_session(session)
            ctx.runtime = runtime
        if ctx.session_key.startswith("dream:"):
            logger.info(
                "Dream run using model={} (preset={})",
                runtime.model,
                runtime.model_preset or "default",
            )
        if ctx.on_runtime_admitted is not None:
            await ctx.on_runtime_admitted(runtime)
        if not ctx.ephemeral and not ctx.fresh_history:
            ctx.session, ctx.pending_summary = self.auto_compact.prepare_session(
                session,
                ctx.session_key,
            )
            session = ctx.require_session()
        is_subagent = ctx.kind is TurnKind.SYSTEM and ctx.msg.sender_id == "subagent"

        ctx.history = [] if ctx.fresh_history else session.get_history(extend_to_user=is_subagent)
        if ctx.fresh_history:
            ctx.pending_summary = None
            ctx.provider_state = None
            # The append changes the portable transcript: an old handle cannot survive it.
            session.provider_state = None
        stored_state = None if ctx.fresh_history else session.provider_state
        subagent_followup_persisted = False
        if is_subagent:
            # Keep the durable internal delivery as an assistant record, but
            # present this completion to the model as fresh follow-up input.
            # Providers without assistant-prefill support drop trailing
            # assistant messages, so using the persisted record as the current
            # prompt would hide an independently dispatched subagent result.
            subagent_followup_persisted = self._persist_subagent_followup(
                session,
                ctx.msg,
                run_id=ctx.run_context.run_id if ctx.run_context else None,
                trace_owner=ctx,
            )
            if subagent_followup_persisted:
                logger.debug("Subagent result persisted for session {}", ctx.session_key)
                # Establish a durable, replay-safe baseline before any fallible
                # provider compatibility or prompt assembly work. A compatible
                # staged state replaces this in a second atomic save below.
                session.provider_state = None
                self.sessions.save(session)
            ctx.input_persisted_early = True
        await ctx.delivery.runtime_admitted(runtime)

        ctx.request_context = self._request_context_for_turn(ctx)
        if ctx.kind is TurnKind.USER:
            ctx.runtime_context_blocks = await self._resolve_runtime_context_for_turn(ctx)
        staged_provider_state = False
        if stored_state is not None and runtime.provider.can_resume_conversation_state(
            stored_state,
            runtime.model,
        ):
            current_provider_message = self.context.build_current_message(
                ctx.msg.content,
                media=ctx.msg.media if ctx.kind is TurnKind.USER and ctx.msg.media else None,
                runtime_context_blocks=ctx.runtime_context_blocks,
            )
            task_id = ctx.msg.metadata.get("subagent_task_id") if is_subagent else None
            already_staged = False
            if isinstance(task_id, str) and task_id:
                internal_meta = current_provider_message.get("_meta")
                current_provider_message["_meta"] = {
                    **(
                        cast(dict[str, Any], internal_meta)
                        if isinstance(internal_meta, dict)
                        else {}
                    ),
                    _SUBAGENT_PROVIDER_TASK_META: task_id,
                }
                already_staged = any(
                    isinstance(message.get("_meta"), dict)
                    and cast(dict[str, Any], message["_meta"]).get(
                        _SUBAGENT_PROVIDER_TASK_META
                    )
                    == task_id
                    for message in stored_state.pending_messages
                )
            ctx.provider_state = (
                stored_state
                if already_staged
                else stored_state.with_pending_messages([
                    *stored_state.pending_messages,
                    current_provider_message,
                ])
            )
            if (
                not ctx.ephemeral
                and (ctx.kind is TurnKind.USER or subagent_followup_persisted)
            ):
                session.provider_state = ctx.provider_state
                staged_provider_state = True
        elif stored_state is not None:
            session.provider_state = None
        if ctx.kind is TurnKind.USER:
            ctx.input_persisted_early = self._persist_user_message_early(
                ctx.msg,
                session,
                runtime_context_blocks=ctx.runtime_context_blocks,
                run_id=ctx.run_context.run_id if ctx.run_context else None,
                trace_owner=ctx,
            )
            if staged_provider_state and not ctx.input_persisted_early:
                session.provider_state = stored_state
        elif subagent_followup_persisted and staged_provider_state:
            # Upgrade the replay-safe baseline to the resumable state before
            # prompt assembly and the first model checkpoint.
            self.sessions.save(session)
        ctx.transcript_input = self._build_transcript_input(ctx)


    async def _run_turn(self, ctx: TurnContext) -> None:
        runtime = ctx.require_runtime()
        if ctx.visible_run_started_at is None:
            ctx.visible_run_started_at = time.time()
        await ctx.delivery.running(started_at=ctx.visible_run_started_at)
        assert ctx.transcript_input is not None
        if ctx.run_context is not None:
            ctx.sent_messages = ctx.run_context.delivery.sent_messages
        with capture_message_deliveries(ctx.sent_messages) as message_sends:
            result = await self._run_agent_loop(
                ctx.transcript_input,
                runtime=runtime,
                streaming=ctx.streaming,
                session=ctx.session,
                pending_queue=None if ctx.fresh_history else ctx.pending_queue,
                ephemeral=ctx.ephemeral,
                run_extra_hooks_for_ephemeral=ctx.run_extra_hooks_for_ephemeral,
                hooks=ctx.hooks,
                hook_factories=ctx.hook_factories,
                turn_scopes=ctx.turn_scopes,
                tools=ctx.tools,
                request_context=ctx.request_context,
                provider_state=ctx.provider_state,
                run_context=ctx.run_context,
                events=ctx.events,
            )
        ctx.run_result = result
        ctx.provider_state_settlement = result.provider_state_settlement
        ctx.final_content = result.final_content
        ctx.all_messages = result.messages
        ctx.summary_checkpoint = None if ctx.fresh_history else result.summary_checkpoint
        ctx.provider_compaction_applied = result.provider_compaction_applied
        ctx.stop_reason = result.stop_reason
        ctx.failure_error_kind = result.failure_error_kind
        sent_to_primary = (ctx.kind is TurnKind.USER and
                           (ctx.delivery.route.channel, ctx.delivery.route.chat_id) in message_sends)
        if ctx.kind is TurnKind.USER and ctx.run_context is not None:
            final_text = (result.final_content or "").strip()
            sent_to_primary = bool(final_text) and any(
                (message.channel, message.chat_id) == (ctx.delivery.route.channel, ctx.delivery.route.chat_id)
                and isinstance(message.content, str) and message.content.strip() == final_text
                for message in ctx.sent_messages
            )
        if (
            ctx.kind is TurnKind.USER and sent_to_primary
            and (not result.had_injections or result.stop_reason == "empty_final_response")
        ):
            ctx.suppress_response = True
        ctx.usage = result.usage
        ctx.delivery.record_usage(result.round_usages)
        if (ctx.kind is TurnKind.USER and not ctx.fresh_history
                and (ctx.run_context is None or ctx.run_context.source_writer is None)):
            try:
                await turn_continuation.maybe_continue_turn(ctx)
            except BaseException:
                if ctx.provider_state_settlement is not None:
                    ctx.provider_state_settlement.discard()
                    ctx.provider_state_settlement = None
                raise

    async def _persist_turn(self, ctx: TurnContext) -> None:
        if ctx.run_context is not None and ctx.run_context.source_writer is not None:
            if ctx.provider_state_settlement is not None:
                ctx.provider_state_settlement.discard()
                ctx.provider_state_settlement = None
            writer = ctx.run_context.source_writer
            updates = {key: value for key, value in ctx.require_session().metadata.items()
                       if key not in ctx.source_metadata_base or value != ctx.source_metadata_base[key]}
            if ctx.usage is not None:
                updates['_last_usage'] = ctx.usage.to_dict()
            writer.store.update_metadata(writer.owner.key, updates)
            ctx.turn_latency_ms = max(0, int((time.time() - ctx.turn_wall_started_at) * 1000))
            ctx.delivery.record_latency(ctx.turn_latency_ms)
            if not ctx.ephemeral:
                await self.runtime_event_publisher.session_turn_persisted(
                    ctx.msg, ctx.session_key, turn_id=ctx.turn_id, attributes=ctx.attributes)
            return
        session = ctx.require_session()
        settlement = ctx.provider_state_settlement
        turn_continuation.prepare_save_boundary(ctx)

        if (
            ctx.kind is TurnKind.USER
            and (ctx.final_content is None or not ctx.final_content.strip())
            and not ctx.suppress_response
        ):
            ctx.final_content = EMPTY_FINAL_RESPONSE_MESSAGE

        latency_started_at = (
            ctx.visible_run_started_at
            if (
                ctx.kind is TurnKind.SYSTEM
                or turn_continuation.internal_continuation_inbound(ctx.msg.metadata)
            )
            and ctx.visible_run_started_at is not None
            else ctx.turn_wall_started_at
        )
        ctx.turn_latency_ms = max(0, int((time.time() - latency_started_at) * 1000))
        if ctx.usage is not None and not ctx.ephemeral:
            session.metadata["_last_usage"] = ctx.usage.to_dict()
        # A state from the prior request may not cross the transcript mutation
        # below.  Only the settlement candidate, obtained after that mutation,
        # can be written for this completed turn.
        session.provider_state = None
        try:
            transcript_rewritten = self._save_turn(
                session, ctx.all_messages, ctx.save_skip,
                turn_latency_ms=ctx.turn_latency_ms,
                summary_checkpoint=ctx.summary_checkpoint,
                input_persisted_early=ctx.input_persisted_early,
                run_id=ctx.run_context.run_id if ctx.run_context else None,
                trace_owner=ctx,
            )
        except BaseException:
            if settlement is not None:
                settlement.discard()
            ctx.provider_state_settlement = None
            self.sessions.invalidate(session.key)
            raise
        candidate = None
        if (
            settlement is not None
            and not ctx.ephemeral
            and not transcript_rewritten
            and not ctx.provider_compaction_applied
            and ctx.summary_checkpoint is None
        ):
            candidate = settlement.candidate_for_persistence()
        if candidate is None and settlement is not None:
            settlement.discard()
        if candidate is not None:
            session.provider_state = candidate
        ctx.delivery.record_latency(ctx.turn_latency_ms)
        self._clear_pending_user_turn(session)
        self._clear_runtime_checkpoint(session)
        try:
            self.sessions.save(session, fsync=candidate is not None)
        except BaseException:
            if settlement is not None:
                settlement.discard()
            ctx.provider_state_settlement = None
            session.provider_state = None
            self.sessions.invalidate(session.key)
            raise
        if candidate is not None:
            persisted = session.provider_state
            persisted_matches = (
                isinstance(persisted, ProviderConversationState)
                and persisted.to_private_record() == candidate.to_private_record()
            )
            if not persisted_matches or not settlement.commit_persisted(candidate):
                # A storage adapter that clears or substitutes the candidate
                # leaves the in-memory view uncertain.  Do not retry a CAS or
                # resurrect this handle after a completed atomic save.
                settlement.discard()
                session.provider_state = None
                self.sessions.invalidate(session.key)
            else:
                ctx.provider_state = candidate
        ctx.provider_state_settlement = None
        if not ctx.ephemeral:
            await self.runtime_event_publisher.session_turn_persisted(
                ctx.msg,
                ctx.session_key,
                turn_id=ctx.turn_id,
                attributes=ctx.attributes,
            )

    async def _prepare_outbound(self, ctx: TurnContext) -> None:
        ctx.delivery.record_stop_reason(
            ctx.stop_reason,
            failure_error_kind=ctx.failure_error_kind,
        )
        if ctx.suppress_response:
            ctx.outbound = None
            return
        if ctx.kind is TurnKind.SYSTEM:
            ctx.outbound = ctx.delivery.background_response(
                ctx.final_content,
                stop_reason=ctx.stop_reason,
                streamed=ctx.streamed_content,
                latency_ms=ctx.turn_latency_ms,
            )
            return
        ctx.outbound = self._assemble_outbound(
            ctx.delivery.delivery_message,
            cast(str, ctx.final_content),
            ctx.stop_reason,
            ctx.streamed_content,
            log_content=ctx.require_session().policy.log_content,
            turn_latency_ms=ctx.turn_latency_ms,
        )
        if ctx.ephemeral and ctx.outbound is not None:
            ctx.outbound.metadata["_stop_reason"] = ctx.stop_reason

    def _sanitize_persisted_blocks(
        self,
        content: list[object],
    ) -> list[object]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[object] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            block_data = cast(dict[str, Any], block)
            image_url = cast(dict[str, Any], block_data.get("image_url", {}))
            if block_data.get("type") == "image_url" and str(
                image_url.get("url", "")
            ).startswith("data:image/"):
                internal_meta = cast(dict[str, Any], block_data.get("_meta") or {})
                path = cast(str, internal_meta.get("path", ""))
                filtered.append(
                    {"type": "text", "text": image_placeholder_text(path)}
                )
                continue

            filtered.append(block_data)

        return filtered

    @staticmethod
    def _validated_checkpoint_boundary(
        checkpoint: SessionSummaryCheckpoint | None,
        *,
        skip: int,
        message_count: int,
        session_key: str,
    ) -> int | None:
        """Return a checkpoint boundary only when it belongs to this turn."""
        if checkpoint is None:
            return None
        boundary = checkpoint.transcript_boundary
        if skip - 1 <= boundary <= message_count:
            return boundary
        logger.warning(
            "Ignoring invalid summary boundary {} outside [{}, {}] for {}",
            boundary,
            skip - 1,
            message_count,
            session_key,
        )
        return None

    def _save_turn(
        self,
        session: Session,
        messages: list[dict[str, Any]],
        skip: int,
        *,
        turn_latency_ms: int | None = None,
        summary_checkpoint: SessionSummaryCheckpoint | None = None,
        input_persisted_early: bool = False,
        run_id: str | None = None,
        trace_owner: TurnContext | None = None,
    ) -> bool:
        """Commit new-turn messages and report whether model-visible replay changed."""
        transcript_rewritten = summary_checkpoint is not None
        declared_tool_call_ids = {
            str(tc["id"])
            for m in session.messages
            if m.get("role") == "assistant"
            for tc_value in cast(Iterable[object], m.get("tool_calls") or [])
            if isinstance(tc_value, dict)
            for tc in (cast(dict[str, Any], tc_value),)
            if tc.get("id")
        }
        fulfilled_tool_call_ids = {
            str(m["tool_call_id"])
            for m in session.messages
            if m.get("role") == "tool" and m.get("tool_call_id")
        }
        last_assistant_idx: int | None = None
        saved_followup_ids: set[str] = set()
        checkpoint_boundary = self._validated_checkpoint_boundary(
            summary_checkpoint,
            skip=skip,
            message_count=len(messages),
            session_key=session.key,
        )

        # The trigger input may already be the session tail while still being
        # the first message after the replacement checkpoint.
        if summary_checkpoint is not None and checkpoint_boundary == skip - 1:
            insert_at = len(session.messages) - (1 if input_persisted_early else 0)
            session.commit_summary_checkpoint(
                summary_checkpoint.summary,
                insert_at=insert_at,
            )

        for message_index, message in enumerate(messages[skip:], start=skip):
            # Insert against the raw transcript index before filtering the
            # message so persistence cleanup cannot shift the H/Δ boundary.
            if summary_checkpoint is not None and checkpoint_boundary == message_index:
                session.commit_summary_checkpoint(summary_checkpoint.summary)

            entry = dict(message)
            followup_id_value = cast(object, entry.pop(PENDING_FOLLOWUP_ID_KEY, None))
            followup_ids = (
                [followup_id_value]
                if isinstance(followup_id_value, str)
                else [
                    followup_id
                    for followup_id in cast(list[object], followup_id_value)
                    if isinstance(followup_id, str)
                ]
                if isinstance(followup_id_value, list)
                else []
            )
            internal_meta = cast(object, entry.pop("_meta", None))
            runtime_context_meta = (
                cast(dict[str, Any], internal_meta).get(
                    RUNTIME_CONTEXT_MESSAGE_META
                )
                if isinstance(internal_meta, dict)
                else None
            )
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                transcript_rewritten = True
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                tool_call_id = entry.get("tool_call_id")
                tool_call_id_str = str(tool_call_id) if tool_call_id else ""
                if (
                    not tool_call_id_str
                    or tool_call_id_str not in declared_tool_call_ids
                    or tool_call_id_str in fulfilled_tool_call_ids
                ):
                    # Undeclared tool results corrupt future provider requests.
                    logger.warning(
                        "Dropping invalid tool result {} from session {} during persistence",
                        tool_call_id_str or "(missing id)",
                        session.key,
                    )
                    transcript_rewritten = True
                    continue
                fulfilled_tool_call_ids.add(tool_call_id_str)
                # Preserve model-visible text for replay; redact only inline images.
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(
                        cast(list[object], content),
                    )
                    if not filtered:
                        # Preserve the tool_call/result pair after block filtering.
                        filtered = [
                            {"type": "text", "text": "[tool result omitted during persistence]"}
                        ]
                    entry["content"] = filtered
                    transcript_rewritten |= filtered != content
            elif role == "user":
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(
                        cast(list[object], content),
                    )
                    if not filtered:
                        transcript_rewritten = True
                        continue
                    entry["content"] = filtered
                    transcript_rewritten |= filtered != content
                if isinstance(runtime_context_meta, dict):
                    entry[RUNTIME_CONTEXT_HISTORY_META] = runtime_context_meta
            entry.setdefault("timestamp", datetime.now().isoformat())
            # Attribution comes only from canonical admission, never provider fields.
            entry.pop("_run_id", None)
            if run_id:
                entry["_run_id"] = run_id
            session.messages.append(entry)
            self._capture_trace_row(trace_owner, entry)
            if role == "user":
                saved_followup_ids.update(followup_id for followup_id in followup_ids if followup_id)
            if role == "assistant":
                last_assistant_idx = len(session.messages) - 1
                declared_tool_call_ids.update(
                    str(tc["id"])
                    for tc_value in cast(
                        Iterable[object],
                        entry.get("tool_calls") or [],
                    )
                    if isinstance(tc_value, dict)
                    for tc in (cast(dict[str, Any], tc_value),)
                    if tc.get("id")
                )
        if summary_checkpoint is not None and checkpoint_boundary == len(messages):
            session.commit_summary_checkpoint(summary_checkpoint.summary)
        if turn_latency_ms is not None and last_assistant_idx is not None:
            session.messages[last_assistant_idx]["latency_ms"] = int(turn_latency_ms)
        if saved_followup_ids:
            acknowledge_pending_followups(session, saved_followup_ids)
        session.updated_at = datetime.now()
        return transcript_rewritten

    def _persist_subagent_followup(
        self, session: Session, msg: InboundMessage, *, run_id: str | None = None,
        trace_owner: TurnContext | None = None,
    ) -> bool:
        """Persist subagent follow-ups before prompt assembly so history stays durable.

        Returns True if a new entry was appended; False if the follow-up was
        deduped (same ``subagent_task_id`` already in session) or carries no
        content worth persisting.
        """
        if not msg.content:
            return False
        metadata_value = cast(object, msg.metadata)
        task_id = (
            msg.metadata.get("subagent_task_id")
            if isinstance(metadata_value, dict)
            else None
        )
        if task_id and any(
            m.get("injected_event") == "subagent_result" and m.get("subagent_task_id") == task_id
            for m in session.messages
        ):
            return False
        session.add_message(
            "assistant",
            msg.content,
            sender_id=msg.sender_id,
            injected_event="subagent_result",
            subagent_task_id=task_id,
            **({"_run_id": run_id} if run_id else {}),
        )
        self._capture_trace_row(trace_owner, session.messages[-1], is_input=True)
        return True

    def _set_runtime_checkpoint(self, session: Session, payload: dict[str, Any]) -> None:
        """Persist the latest in-flight turn state into session metadata."""
        session.metadata[self._RUNTIME_CHECKPOINT_KEY] = payload
        self.sessions.save_runtime_checkpoint(session)

    def _mark_pending_user_turn(self, session: Session) -> None:
        session.metadata[self._PENDING_USER_TURN_KEY] = True

    def _clear_pending_user_turn(self, session: Session) -> None:
        session.metadata.pop(self._PENDING_USER_TURN_KEY, None)

    def _clear_runtime_checkpoint(self, session: Session) -> None:
        if self._RUNTIME_CHECKPOINT_KEY in session.metadata:
            session.metadata.pop(self._RUNTIME_CHECKPOINT_KEY, None)

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        sender_id: str = "user",
        media: list[str] | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        ephemeral: bool = False,
        _run_extra_hooks_for_ephemeral: bool = False,
        hooks: list[AgentHook] | None = None,
        hook_factories: list[AgentTurnHookFactory] | None = None,
        tools: ToolRegistry | None = None,
        persist_user_message: bool = True,
        runtime: LLMRuntime | None = None,
        on_runtime_admitted: Callable[[LLMRuntime], Awaitable[None]] | None = None,
        attributes: Mapping[str, Any] | None = None,
        *,
        source: TurnSource = TurnSource.DIRECT,
    ) -> OutboundMessage | None:
        """Process an external message directly and return the outbound payload."""
        result = await self._execute_direct(
            content, session_key=session_key, channel=channel, chat_id=chat_id,
            sender_id=sender_id, media=media, on_progress=on_progress,
            on_stream=on_stream, on_stream_end=on_stream_end, ephemeral=ephemeral,
            _run_extra_hooks_for_ephemeral=_run_extra_hooks_for_ephemeral,
            hooks=hooks, hook_factories=hook_factories, tools=tools,
            persist_user_message=persist_user_message, runtime=runtime,
            on_runtime_admitted=on_runtime_admitted, attributes=attributes,
            source=source,
        )
        return result.outbound

    async def _execute_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        sender_id: str = "user",
        media: list[str] | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        ephemeral: bool = False,
        _run_extra_hooks_for_ephemeral: bool = False,
        hooks: list[AgentHook] | None = None,
        hook_factories: list[AgentTurnHookFactory] | None = None,
        tools: ToolRegistry | None = None,
        persist_user_message: bool = True,
        runtime: LLMRuntime | None = None,
        on_runtime_admitted: Callable[[LLMRuntime], Awaitable[None]] | None = None,
        attributes: Mapping[str, Any] | None = None,
        *,
        source: TurnSource = TurnSource.DIRECT,
        trace_mode: TraceMode = TraceMode.NONE,
    ) -> TurnResult:
        """Execute a direct adapter request while preserving canonical results."""
        if channel == "system":
            raise ValueError("channel 'system' is reserved for internal messages")
        metadata: dict[str, Any] = {}
        if not persist_user_message:
            metadata[turn_continuation.SKIP_USER_PERSIST_META] = True
        msg = InboundMessage(
            channel=channel, sender_id=sender_id, chat_id=chat_id,
            content=content, media=media or [], metadata=metadata,
        )
        request = TurnRequest(content=content, session_key=session_key, source=source, trace_mode=trace_mode,
            route=DeliveryTarget(channel=channel, chat_id=chat_id), sender_id=sender_id,
            media=tuple(media or ()), metadata=metadata, hooks=tuple(hooks or ()),
            callbacks=TurnCallbacks(on_progress=on_progress, on_stream=on_stream, on_stream_end=on_stream_end))
        async def admission_ended(effective_key: str) -> None:
            try:
                await self.runtime_event_publisher.run_status_changed(msg, effective_key, "idle")
            finally:
                self.runtime_event_publisher.clear_turn(effective_key)

        options = _TurnExecutionOptions(ephemeral=ephemeral,
            run_extra_hooks_for_ephemeral=_run_extra_hooks_for_ephemeral,
            hook_factories=tuple(hook_factories or ()), runtime=runtime,
            on_runtime_admitted=on_runtime_admitted, attributes=dict(attributes or {}),
            raise_errors=True, on_admission_end=admission_ended,
            trusted_foreground_activity=(
                source in _FOREGROUND_IDLE_SOURCES
                and sender_id != "subagent"
                and not sender_id.startswith("system:")
            ))
        result = await self.execute_turn(request, _tools=tools, _options=options)
        return result

    def _get_session_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared lock while allowing idle session entries to expire."""
        lock = self._session_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_key] = lock
        return lock
