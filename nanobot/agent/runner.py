"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
import inspect
import os
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from nanobot.agent.context import TranscriptInput
from nanobot.agent.context_governance import (
    ContextCompactionState,
    ContextGovernanceConfig,
    ContextGovernor,
    HistoryConsolidator,
    ModelRequestState,
    ProviderCompactionConsolidator,
    SourceCompactionState,
    TranscriptBuilder,
)
from nanobot.agent.hook import AgentHook, AgentHookContext, AgentRunHookContext
from nanobot.agent.policy import ToolPolicy
from nanobot.agent.tools.execution import KnownToolResult, ToolResultCallback, execute_tool_calls
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import ToolOutcome, TurnContext
from nanobot.events import NO_EVENTS, EventSink
from nanobot.llm_usage.context import (
    LLMUsageSource,
    bind_llm_usage_source,
    reset_llm_usage_source,
    source_from_session_key,
)
from nanobot.providers.base import (
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ProviderConversationState,
)
from nanobot.providers.conversation_state import (
    ConversationRequestBinding,
    ProviderConversationStateController,
    ProviderStateSettlement,
)
from nanobot.session.source import SourceValidationError
from nanobot.session.source_projection import RuntimeNode
from nanobot.session.source_request import SourceRequestLedger, SourceRequestTicket
from nanobot.session.summary import SessionSummaryCheckpoint
from nanobot.session.turn_writer import SourceInjectionBinding, SourceTurnWriter
from nanobot.utils.helpers import (
    build_assistant_message,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    extract_reasoning,
    strip_reasoning_tags,
)
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.prompt_templates import render_template
from nanobot.utils.runtime import (
    EMPTY_FINAL_RESPONSE_MESSAGE,
    build_budget_exhausted_finalization_message,
    build_finalization_retry_message,
    build_length_recovery_message,
    is_blank_text,
)

if TYPE_CHECKING:
    from nanobot.agent.busy_steering import BusySteering
    from nanobot.agent.tools.filesystem import ReadFileTool
    from nanobot.session.artifact_grants import ArtifactGrantSet, ArtifactHandoffReceipt
    from nanobot.session.artifact_reader import ArtifactReader
    from nanobot.session.source_request import SourceTranscript


@dataclass(frozen=True)
class RecordedResponse:
    """A provider response and its exact mandatory source-write identity."""

    response: LLMResponse
    source_event_id: str | None


ContinuationCallback = Callable[[], str | None]
CheckpointCallback = Callable[[dict[str, Any]], Awaitable[None]]
InjectionCallback = Callable[..., Awaitable[Iterable[Any] | None]]

_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."
_ARREARAGE_ERROR_MESSAGE = (
    "The AI provider rejected the request because the API key is out of quota or the "
    "account is in arrears. Please top up / check the billing status of your API key and try again."
)
_PERSISTED_MODEL_ERROR_PLACEHOLDER = "[Assistant reply unavailable due to model error.]"
_MAX_EMPTY_RETRIES = 2
_MAX_LENGTH_RECOVERIES = 3
_MAX_INJECTIONS_PER_TURN = 3
_MAX_INJECTION_CYCLES = 5


def _restore_outer_whitespace(content: str, original: str | None) -> str:
    """Restore boundary whitespace stripped while cleaning one recovered segment."""
    if not original:
        return content
    leading_size = len(original) - len(original.lstrip())
    trailing_size = len(original) - len(original.rstrip())
    leading = original[:leading_size]
    trailing = original[-trailing_size:] if trailing_size else ""
    return f"{leading}{content}{trailing}"


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]] | None
    tools: ToolRegistry
    runtime: LLMRuntime
    max_iterations: int
    max_tool_result_chars: int
    tool_policy: ToolPolicy | None = None
    turn_context: TurnContext | None = None
    fail_on_tool_error: bool = False
    transcript_input: TranscriptInput | None = None
    transcript_builder: TranscriptBuilder | None = None
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    workspace: Path | None = None
    session_key: str | None = None
    provider_retry_mode: str = "standard"
    checkpoint_callback: CheckpointCallback | None = None
    consolidate_history: HistoryConsolidator | None = None
    consolidate_provider_compaction: ProviderCompactionConsolidator | None = None
    injection_callback: InjectionCallback | None = None
    terminal_injection_callback: InjectionCallback | None = None
    llm_timeout_s: float | None = None
    continuation_callback: ContinuationCallback | None = None
    finalize_on_max_iterations: bool = True
    provider_state: ProviderConversationState | None = None
    llm_usage_source: LLMUsageSource | None = None
    events: EventSink = NO_EVENTS
    tool_result_callback: ToolResultCallback | None = None
    source_writer: SourceTurnWriter | None = None
    source_transcript: SourceTranscript | None = None
    source_initial_eligible_ids: frozenset[str] = frozenset()
    source_isolated: bool = False
    artifact_read_allowed: bool = False
    conversation_binding: ConversationRequestBinding | None = field(default=None, repr=False)
    artifact_grants: ArtifactGrantSet | None = field(default=None, repr=False)
    artifact_reader_factory: Callable[
        [SourceTurnWriter, bool, ReadFileTool], ArtifactReader | None
    ] | None = field(default=None, repr=False)
    artifact_handoff_collector: object | None = field(default=None, repr=False)
    busy_steering: BusySteering | None = field(default=None, repr=False)


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: LLMUsage | None = None
    # One entry per runner-visible model round. Recovery dispatches needed to
    # produce that round's response are folded into the same usage value.
    round_usages: list[LLMUsage] = field(default_factory=list)
    stop_reason: str = "completed"
    policy_metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    failure_error_kind: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    had_injections: bool = False
    # Terminal tail to emit when the preceding final-content prefix was already streamed.
    pending_stream_content: str | None = None
    provider_state: ProviderConversationState | None = field(default=None, repr=False)
    summary_checkpoint: SessionSummaryCheckpoint | None = field(default=None, repr=False)
    provider_compaction_applied: bool = field(default=False, repr=False)
    # A process-local, one-shot authority.  The session owner must persist its
    # exact candidate before committing it; direct runner callers never adopt
    # it implicitly.
    provider_state_settlement: ProviderStateSettlement | None = field(default=None, repr=False)
    artifact_handoff_receipt: ArtifactHandoffReceipt | None = field(default=None, repr=False)


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self, *, artifact_reader_factory: Callable[
            [SourceTurnWriter, bool, ReadFileTool], ArtifactReader | None] | None = None) -> None:
        self.artifact_reader_factory = artifact_reader_factory
        self.context_governor = ContextGovernor()

    @staticmethod
    def _append_bound_message(messages: list[dict[str, Any]], message: dict[str, Any],
                              source: SourceRequestLedger | None,
                              event_id: str | None = None) -> None:
        """Append explicit replay provenance without recording source again."""
        if source is not None:
            if event_id is None:
                source.append_runtime(message)
            else:
                source.append_source(event_id, message)
        messages.append(message)

    @staticmethod
    def _append_injected_messages(
        messages: list[dict[str, Any]],
        injections: list[dict[str, Any]],
    ) -> None:
        """Append injected messages without rewriting the raw transcript."""
        messages.extend(injections)

    async def _try_drain_injections(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        assistant_message: dict[str, Any] | None,
        injection_cycles: int,
        *,
        conversation_state: ProviderConversationStateController | None = None,
        phase: str = "after error",
        iteration: int | None = None,
        allow_continuation: bool = False,
        wait_at_terminal: bool = False,
        source: SourceRequestLedger | None = None,
        assistant_event_id: str | None = None,
        allow_busy_steering: bool = True,
    ) -> tuple[bool, int]:
        """Drain pending injections. Returns (should_continue, updated_cycles).

        If injections are found and we haven't exceeded _MAX_INJECTION_CYCLES,
        append them to *messages* (and emit a checkpoint if *assistant_message*
        and *iteration* are both provided) and return (True, cycles+1) so the
        caller continues the iteration loop.  Otherwise return (False, cycles).
        """
        injections: list[dict[str, Any]] = []
        source_event_ids: list[str | None] = []
        real_injection = False

        def accept_injections(items: list[SourceInjectionBinding | dict[str, object]]) -> None:
            for item in items:
                if isinstance(item, SourceInjectionBinding):
                    if spec.source_writer is None:
                        raise SourceValidationError('Source injection requires its active writer')
                    binding = spec.source_writer.validate_injection_binding(item)
                    injections.append(dict(binding.message))
                    source_event_ids.append(binding.event_id)
                else:
                    injections.append(item)
                    source_event_ids.append(None)

        if injection_cycles < _MAX_INJECTION_CYCLES:
            if spec.busy_steering is not None:
                if allow_busy_steering:
                    for binding in spec.busy_steering.drain(limit=_MAX_INJECTIONS_PER_TURN):
                        injections.append(dict(binding.message))
                        source_event_ids.append(binding.source_event_id)
            else:
                accept_injections(await self._drain_injections(spec))
            real_injection = bool(injections)
        if not injections and allow_continuation and assistant_message is not None:
            continuation = self._build_continuation_message(spec)
            if continuation is not None:
                injections = [continuation]
        if (
            not injections
            and wait_at_terminal
            and injection_cycles < _MAX_INJECTION_CYCLES
            and spec.busy_steering is None
        ):
            accept_injections(await self._drain_injections(spec, terminal=True))
            real_injection = bool(injections)
        if not injections:
            return False, injection_cycles
        if real_injection and source is not None and (
                len(source_event_ids) != len(injections) or any(identity is None for identity in source_event_ids)):
            raise SourceValidationError('Source user injection requires a durable event binding')
        if real_injection:
            injection_cycles += 1
        if assistant_message is not None:
            self._append_bound_message(messages, assistant_message, source, assistant_event_id)
            if iteration is not None:
                checkpoint: dict[str, Any] = {
                    "phase": "final_response",
                    "iteration": iteration,
                    "model": spec.runtime.model,
                    "assistant_message": assistant_message,
                    "completed_tool_results": [],
                    "pending_tool_calls": [],
                }
                if conversation_state is not None:
                    checkpoint["provider_state"] = conversation_state.checkpoint(
                        messages
                    )
                await self._emit_checkpoint(
                    spec,
                    checkpoint,
                )
        if source is None:
            self._append_injected_messages(messages, injections)
        else:
            for index, message in enumerate(injections):
                self._append_bound_message(messages, message, source,
                    source_event_ids[index] if real_injection else None)
        if real_injection:
            logger.info(
                "Injected {} follow-up message(s) {} ({}/{})",
                len(injections), phase, injection_cycles, _MAX_INJECTION_CYCLES,
            )
        else:
            logger.info("Injected caller-requested continuation {}", phase)
        return True, injection_cycles

    @staticmethod
    def _build_continuation_message(spec: AgentRunSpec) -> dict[str, str] | None:
        callback = spec.continuation_callback
        if callback is None:
            return None
        try:
            content = callback()
        except Exception:
            logger.exception("continuation_callback failed")
            return None
        if content is None or not content.strip():
            return None
        return {"role": "user", "content": content}

    async def _drain_injections(
        self,
        spec: AgentRunSpec,
        *,
        terminal: bool = False,
    ) -> list[SourceInjectionBinding | dict[str, object]]:
        """Drain pending user messages via the injection callback.

        Returns normalized user messages (capped by
        ``_MAX_INJECTIONS_PER_TURN``), or an empty list when there is
        nothing to inject. Messages beyond the cap are logged so they
        are not silently lost.
        """
        callback = (
            spec.terminal_injection_callback
            if terminal
            else spec.injection_callback
        )
        if callback is None:
            return []
        try:
            signature = inspect.signature(callback)
            accepts_limit = (
                "limit" in signature.parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            )
            if accepts_limit:
                items = await callback(limit=_MAX_INJECTIONS_PER_TURN)
            else:
                items = await callback()
        except Exception:
            logger.exception("injection_callback failed")
            return []
        if not items:
            return []
        injected_messages: list[SourceInjectionBinding | dict[str, object]] = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, SourceInjectionBinding):
                if spec.source_writer is None:
                    raise SourceValidationError('Source injection requires its active writer')
                binding = spec.source_writer.validate_injection_binding(item)
                if self._has_injection_content(binding.message.get("content")):
                    injected_messages.append(binding)
                continue
            if isinstance(item, dict):
                message_item = cast(dict[str, Any], item)
                if message_item.get("role") == "user" and "content" in message_item:
                    if self._has_injection_content(message_item.get("content")):
                        injected_messages.append(message_item)
                continue
            content = getattr(item, "content") if hasattr(item, "content") else str(item)
            if self._has_injection_content(content):
                injected_messages.append({"role": "user", "content": content})
        if len(injected_messages) > _MAX_INJECTIONS_PER_TURN:
            dropped = len(injected_messages) - _MAX_INJECTIONS_PER_TURN
            logger.warning(
                "Injection callback returned {} messages, capping to {} ({} dropped)",
                len(injected_messages), _MAX_INJECTIONS_PER_TURN, dropped,
            )
            injected_messages = injected_messages[:_MAX_INJECTIONS_PER_TURN]
        return injected_messages

    @staticmethod
    def _has_injection_content(content: Any) -> bool:
        if content is None:
            return False
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return bool(cast(list[Any], content))
        return True

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        child_artifact_grants = (
            spec.artifact_grants
            if spec.source_writer is not None and spec.source_writer.parent_stream_id is not None
            else None
        )
        try:
            if spec.busy_steering is not None:
                if spec.injection_callback is not None or spec.terminal_injection_callback is not None:
                    raise ValueError("durable busy steering is exclusive with generic injection callbacks")
                spec.busy_steering.validate_context(spec.turn_context, spec.source_writer)
                if spec.source_writer is not None and spec.source_transcript is None:
                    raise ValueError("durable busy steering requires a source transcript")
            if (spec.source_writer is not None and (spec.turn_context is None
                    or spec.turn_context.execution_owner is not spec.source_writer.owner)):
                raise ValueError("Source recording requires the same dispatch owner")
            if (spec.turn_context is not None and spec.turn_context.execution_owner is not None
                    and spec.tool_result_callback is None and spec.source_writer is None):
                raise ValueError("Execution ownership requires durable tool-result recording")
            # Never inherit an invocation-bound reader from a parent registry.
            artifact_reader_factory = spec.artifact_reader_factory or self.artifact_reader_factory
            if artifact_reader_factory is not None or 'read_artifact' in spec.tools.tool_names:
                from nanobot.agent.tools.artifact import ReadArtifactTool
                from nanobot.agent.tools.filesystem import ReadFileTool

                tools = spec.tools.filtered([name for name in spec.tools.tool_names if name != 'read_artifact'])
                read_file = tools.get('read_file')
                if (spec.artifact_read_allowed and artifact_reader_factory is not None and spec.source_writer is not None
                        and type(read_file) is ReadFileTool):
                    from nanobot.agent.tools.artifact_handoff import ArtifactHandoffTool
                    from nanobot.session.artifact_grants import (
                        ArtifactGrantSet,
                        ArtifactHandoffCollector,
                    )

                    isolated = spec.source_isolated or spec.source_writer.parent_stream_id is not None
                    reader = artifact_reader_factory(spec.source_writer, isolated, read_file)
                    if reader is not None:
                        if reader.writer is not spec.source_writer or reader.mode != (
                                'isolated_stream' if isolated else 'root_history') or reader.read_file is not read_file:
                            raise ValueError('Artifact reader does not match admitted invocation')
                        grants = spec.artifact_grants or ArtifactGrantSet.for_reader(reader)
                        reader = replace(reader, grants=grants)
                        tools.register(ReadArtifactTool(reader))
                        if spec.source_writer.parent_stream_id is not None:
                            child_artifact_grants = grants
                            collector = ArtifactHandoffCollector(reader)
                            tools.register(ArtifactHandoffTool(collector))
                        else:
                            collector = None
                        if spec.turn_context is not None:
                            spec.turn_context.artifact_reader = reader
                            spec.turn_context.artifact_grants = grants
                    else:
                        collector = None
                else:
                    collector = None
                spec = replace(spec, tools=tools, artifact_handoff_collector=collector)
            hook = spec.hook or AgentHook()
            messages, compaction = self._initial_transcript_and_compaction(spec)
            context = AgentRunHookContext(messages=deepcopy(messages))
            llm_usage_source_token = bind_llm_usage_source(
                spec.llm_usage_source or source_from_session_key(spec.session_key)
            )
        except BaseException:
            if child_artifact_grants is not None:
                child_artifact_grants.close()
            raise

        try:
            request_state = self._make_request_state(spec, messages, compaction)
            if request_state.source is not None:
                self.context_governor.admit_source_checkpoint(request_state, request_state.source.transcript,
                    tool_definitions=spec.tools.get_definitions())
                context.messages = deepcopy(messages)
            await hook.before_run(context)
            result = await self._run_core(spec, hook, messages, request_state)
        except asyncio.CancelledError as exc:
            if spec.source_writer is not None:
                spec.source_writer.fail()
            context.messages = deepcopy(messages)
            context.stop_reason = "cancelled"
            context.error = None
            context.exception = exc
            raise
        except Exception as exc:
            if spec.source_writer is not None:
                spec.source_writer.fail()
            context.messages = deepcopy(messages)
            context.stop_reason = "error"
            context.error = f"Error: {type(exc).__name__}: {exc}"
            context.exception = exc
            await hook.on_error(context)
            raise
        else:
            context.messages = deepcopy(result.messages)
            context.final_content = result.final_content
            context.tools_used = list(result.tools_used)
            context.usage = result.usage
            context.stop_reason = result.stop_reason
            context.error = result.error
            context.tool_events = deepcopy(result.tool_events)
            context.had_injections = result.had_injections
            context.exception = None
            if context.error is not None:
                await hook.on_error(context)
            try:
                await hook.after_run(context)
            except BaseException:
                if result.provider_state_settlement is not None:
                    result.provider_state_settlement.discard()
                raise
            return result
        finally:
            try:
                context.messages = deepcopy(messages)
                if context.exception is None:
                    await hook.on_finally(context)
                else:
                    try:
                        await hook.on_finally(context)
                    except Exception:
                        logger.exception(
                            "AgentHook.on_finally error after {}",
                            context.stop_reason or "run exception",
                        )
            finally:
                try:
                    reset_llm_usage_source(llm_usage_source_token)
                finally:
                    if child_artifact_grants is not None:
                        child_artifact_grants.close()

    @staticmethod
    def _initial_transcript_and_compaction(
        spec: AgentRunSpec,
    ) -> tuple[list[dict[str, Any]], ContextCompactionState | None]:
        """Build the initial transcript and its optional compaction state."""
        if spec.source_transcript is not None:
            if spec.source_writer is None or spec.initial_messages is not None:
                raise ValueError("Source transcript requires writer and no unbound initial messages")
            return spec.source_transcript.messages(), None
        transcript_input = spec.transcript_input
        if transcript_input is not None:
            if spec.initial_messages is not None:
                raise ValueError("provide either transcript_input or initial_messages, not both")
            transcript_builder = spec.transcript_builder
            if transcript_builder is None:
                raise ValueError("transcript_builder is required with transcript_input")
            return ContextCompactionState.from_transcript(
                transcript_input,
                transcript_builder,
                spec.consolidate_history,
                spec.consolidate_provider_compaction,
            )
        if spec.initial_messages is None:
            raise ValueError("initial_messages is required without transcript_input")
        if spec.consolidate_history is not None:
            raise ValueError("consolidate_history requires transcript_input")
        return list(spec.initial_messages), None

    def _make_request_state(self, spec: AgentRunSpec, messages: list[dict[str, Any]],
                            compaction: ContextCompactionState | None) -> ModelRequestState:
        conversation_state = ProviderConversationStateController(
            provider=spec.runtime.provider,
            model=spec.runtime.model,
            messages=messages,
            state=spec.provider_state,
            session_id=spec.session_key,
        )
        governance_config = ContextGovernanceConfig(
            provider=spec.runtime.provider,
            model=spec.runtime.model,
            tools=spec.tools,
            workspace=spec.workspace,
            session_key=spec.session_key,
            max_tool_result_chars=spec.max_tool_result_chars,
            context_window_tokens=spec.runtime.context_window_tokens,
            max_tokens=spec.runtime.generation.max_tokens,
            preserve_source=spec.source_writer is not None,
        )
        return ModelRequestState(
            config=governance_config,
            conversation=conversation_state,
            conversation_binding=spec.conversation_binding,
            compaction=compaction,
            source_compaction=SourceCompactionState(spec.source_writer, spec.runtime, messages,
                spec.source_writer.stream_id if spec.source_isolated or spec.source_writer.parent_stream_id is not None else None)
                if spec.source_writer is not None else None,
            source=SourceRequestLedger(spec.source_transcript,
                initial_eligible_ids=spec.source_initial_eligible_ids)
                if spec.source_transcript is not None else None,
            events=spec.events,
        )


    async def _run_core(
        self,
        spec: AgentRunSpec,
        hook: AgentHook,
        messages: list[dict[str, Any]],
        request_state: ModelRequestState,
    ) -> AgentRunResult:
        final_content: str | None = None
        tools_used: list[str] = []
        usage: LLMUsage | None = None
        round_usages: list[LLMUsage] = []
        error: str | None = None
        failure_error_kind: str | None = None
        policy_metadata: dict[str, Any] = {}
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []
        external_lookup_counts: dict[str, int] = {}
        # Per-turn throttle for repeated attempts against the same outside target.
        workspace_violation_counts: dict[str, int] = {}
        empty_content_retries = 0
        # Segments from one uninterrupted length-recovery chain. Tool work or
        # injected user input starts a new logical answer and clears the chain.
        length_recovery_parts: list[str] = []
        had_injections = False
        injection_cycles = 0
        pending_stream_content: str | None = None
        provider_state_settlement: ProviderStateSettlement | None = None
        conversation_state = request_state.conversation
        governance_config = request_state.config

        for iteration in range(spec.max_iterations):
            if spec.busy_steering is not None:
                drained, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles, source=request_state.source,
                    phase="before model dispatch",
                )
                had_injections = had_injections or drained
            context = AgentHookContext(
                iteration=iteration,
                messages=messages,
                session_key=spec.session_key,
            )
            await hook.before_iteration(context)
            request_message_count = len(messages)
            request_messages = (
                request_state.compaction.request_messages(messages)
                if request_state.compaction is not None
                else messages
            )
            recorded_response, raw_usage = await self._request_model(
                spec,
                request_messages,
                hook,
                context,
                request_state=request_state,
                transcript=messages,
            )
            response = recorded_response.response
            assert request_state.messages is not None
            messages_for_model = request_state.messages
            if request_state.compaction is not None:
                request_state.compaction.accept_request(
                    messages_for_model,
                    raw_boundary=request_message_count,
                )
            context.response = response
            context.tool_calls = list(response.tool_calls)

            original_content = response.content
            reasoning_text, cleaned_content = extract_reasoning(
                response.reasoning_content,
                response.thinking_blocks,
                response.content,
            )
            response.content = cleaned_content
            round_usages.append(raw_usage)
            context.usage = raw_usage
            usage = self._merge_usage(usage, raw_usage)
            if reasoning_text and not context.streamed_reasoning:
                await hook.emit_reasoning(reasoning_text)
                await hook.emit_reasoning_end()
                context.streamed_reasoning = True

            if response.should_execute_tools:
                context.tool_calls = list(response.tool_calls)
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=True)

                assistant_message = build_assistant_message(
                    response.content or "",
                    tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                assistant_message = conversation_state.project_response_message(
                    assistant_message,
                    response,
                )
                messages.append(assistant_message)
                if request_state.source is not None:
                    if recorded_response.source_event_id is None:
                        raise SourceValidationError('Tool intent has no durable source identity')
                    request_state.source.append_source(recorded_response.source_event_id, assistant_message)
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "awaiting_tools",
                        "iteration": iteration,
                        "model": spec.runtime.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": [],
                        "pending_tool_calls": [tc.to_openai_tool_call() for tc in response.tool_calls],
                    },
                )

                await hook.before_execute_tools(context)

                result_ids: dict[int, str] = {}
                source_writer = spec.source_writer

                async def record_result(value: KnownToolResult) -> None:
                    assert source_writer is not None
                    event_id = await source_writer.record_tool_result_event(value)
                    result_ids[value.ordinal] = event_id
                    collector = spec.artifact_handoff_collector
                    if value.handoff_receipt is not None:
                        from nanobot.session.artifact_grants import (
                            ArtifactGrantSet,
                            ArtifactHandoffCollector,
                        )

                        if isinstance(collector, ArtifactHandoffCollector):
                            collector.record(value.handoff_receipt, event_id)
                        elif spec.turn_context is not None and isinstance(
                            spec.turn_context.artifact_grants, ArtifactGrantSet
                        ):
                            references = spec.turn_context.artifact_grants.accept_handoff(
                                value.handoff_receipt,
                                parent_result_event_id=event_id,
                            )
                            spec.turn_context.artifact_handoff_notices.extend(references)
                        else:
                            raise SourceValidationError('Artifact handoff has no parent grant authority')

                results, new_events = await execute_tool_calls(
                    spec.tools,
                    response.tool_calls,
                    concurrent=spec.concurrent_tools,
                    external_lookup_counts=external_lookup_counts,
                    workspace_violation_counts=workspace_violation_counts,
                    hook=hook,
                    context=context,
                    tool_policy=spec.tool_policy,
                    messages=messages,
                    turn_context=spec.turn_context,
                    fail_on_tool_error=spec.fail_on_tool_error,
                    tool_result_callback=(record_result if source_writer is not None
                                          else spec.tool_result_callback),
                )
                terminal = next((
                    r for r in results
                    if isinstance(r, ToolOutcome) and r.stop_reason
                    and r.stop_reason != "folder_access_required"
                ), None)
                if terminal is None:
                    terminal = next((r for r in results if isinstance(r, ToolOutcome)
                                     and r.stop_reason == "folder_access_required"), None)
                results = [r.content if isinstance(r, ToolOutcome) else r for r in results]
                tool_events.extend(new_events)
                tools_used.extend(
                    tool_call.name
                    for tool_call, event in zip(response.tool_calls, new_events)
                    if event.get("status") == "ok"
                )
                context.tool_results = list(results)
                context.tool_events = list(new_events)
                completed_tool_results: list[dict[str, Any]] = []
                for ordinal, (tool_call, result) in enumerate(zip(response.tool_calls, results)):
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": self.context_governor.normalize_tool_result(
                            governance_config,
                            tool_call.id,
                            tool_call.name,
                            result,
                        ),
                    }
                    messages.append(tool_message)
                    if request_state.source is not None:
                        identity = result_ids.get(ordinal)
                        if identity is None:
                            raise SourceValidationError('Tool result has no durable source identity')
                        request_state.source.append_source(identity, tool_message)
                    completed_tool_results.append(tool_message)
                if spec.turn_context is not None and spec.turn_context.artifact_handoff_notices:
                    references = tuple(spec.turn_context.artifact_handoff_notices)
                    spec.turn_context.artifact_handoff_notices.clear()
                    notice = {
                        "role": "system",
                        "content": (
                            "Verified child artifact handoff is available for this invocation only:\n"
                            + "\n".join(references)
                        ),
                    }
                    self._append_bound_message(messages, notice, request_state.source, None)
                checkpoint_model_messages = (
                    self.context_governor.prepare_messages_for_model(
                        governance_config,
                        messages,
                    )
                    if response.provider_state is not None
                    else None
                )
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "tools_completed",
                        "iteration": iteration,
                        "model": spec.runtime.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": completed_tool_results,
                        "pending_tool_calls": [],
                        "provider_state": conversation_state.checkpoint(
                            messages,
                            model_messages=checkpoint_model_messages,
                        ),
                    },
                )
                if terminal is not None:
                    stop_reason = terminal.stop_reason or "policy_blocked"
                    policy_metadata = dict(terminal.policy_metadata)
                    final_content = str(terminal.content or "")
                    context.stop_reason = stop_reason
                    context.final_content = final_content
                    if spec.source_writer is not None:
                        spec.source_writer.append_message(build_assistant_message(final_content))
                    break
                empty_content_retries = 0
                length_recovery_parts.clear()
                # Checkpoint 1: drain injections after tools, before next LLM call
                _drained, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles, source=request_state.source,
                    phase="after tool execution",
                    allow_busy_steering=iteration + 1 < spec.max_iterations,
                )
                if _drained:
                    had_injections = True
                await hook.after_iteration(context)
                continue

            if response.has_tool_calls:
                logger.warning(
                    "Ignoring tool calls under finish_reason='{}' for {}",
                    response.finish_reason,
                    spec.session_key or "default",
                )
                # The opaque response may still contain those ignored calls.
                # It cannot be replayed unless their matching tool results
                # were durably recorded.
                response.provider_state = None

            clean = hook.finalize_content(context, response.content)
            if (
                response.finish_reason
                not in {"error", "length", "refusal", "content_filter"}
                and is_blank_text(clean)
            ):
                empty_content_retries += 1
                if empty_content_retries < _MAX_EMPTY_RETRIES:
                    logger.warning(
                        "Empty response on turn {} for {} ({}/{}); retrying",
                        iteration,
                        spec.session_key or "default",
                        empty_content_retries,
                        _MAX_EMPTY_RETRIES,
                    )
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=False)
                    await hook.after_iteration(context)
                    continue
                logger.warning(
                    "Empty response on turn {} for {} after {} retries; attempting finalization",
                    iteration,
                    spec.session_key or "default",
                    empty_content_retries,
                )
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=False)
                recorded_response = await self._request_finalization_retry(
                    spec,
                    messages_for_model,
                    request_state=request_state,
                    transcript=messages,
                )
                response = recorded_response.response
                retry_usage = self._record_request_usage(spec, request_state, response)
                round_usages.append(retry_usage)
                usage = self._merge_usage(usage, retry_usage)
                raw_usage = self._merge_usage(raw_usage, retry_usage)
                context.response = response
                context.usage = raw_usage
                context.tool_calls = list(response.tool_calls)
                original_content = response.content
                clean = hook.finalize_content(context, response.content)

            if response.finish_reason == "length":
                if len(length_recovery_parts) < _MAX_LENGTH_RECOVERIES:
                    length_recovery_parts.append(
                        _restore_outer_whitespace(clean or "", original_content)
                    )
                    logger.info(
                        "Output truncated on turn {} for {} ({}/{}); continuing",
                        iteration,
                        spec.session_key or "default",
                        len(length_recovery_parts),
                        _MAX_LENGTH_RECOVERIES,
                    )
                    if hook.wants_streaming():
                        context.stream_continues_current_message = True
                        await hook.on_stream_end(context, resuming=True)
                    self._append_bound_message(messages, conversation_state.project_response_message(
                        build_assistant_message(
                            clean,
                            reasoning_content=response.reasoning_content,
                            thinking_blocks=response.thinking_blocks,
                        ),
                        response,
                    ), request_state.source, recorded_response.source_event_id)
                    self._append_bound_message(messages, build_length_recovery_message(clean or ""), request_state.source)
                    await hook.after_iteration(context)
                    continue

            # Some streaming providers recover with a complete response but no
            # content deltas. When an earlier length segment is already visible,
            # emit this terminal segment into the same stream; otherwise the
            # regular full response would duplicate the visible prefix.
            if (
                length_recovery_parts
                and hook.wants_streaming()
                and not context.streamed_content
                and response.finish_reason != "error"
                and not is_blank_text(clean)
            ):
                await hook.on_stream(
                    context,
                    _restore_outer_whitespace(clean or "", original_content),
                )
                context.streamed_content = True

            assistant_message: dict[str, Any] | None = None
            if response.finish_reason != "error" and not is_blank_text(clean):
                assistant_message = build_assistant_message(
                    clean,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                assistant_message = conversation_state.project_response_message(
                    assistant_message,
                    response,
                )

            # Check for mid-turn injections BEFORE signaling stream end.
            # If injections are found we keep the stream alive (resuming=True)
            # so streaming channels don't prematurely finalize the card.
            should_continue, injection_cycles = await self._try_drain_injections(
                spec, messages, assistant_message, injection_cycles,
                source=request_state.source, assistant_event_id=recorded_response.source_event_id,
                conversation_state=conversation_state,
                phase="after final response",
                iteration=iteration,
                allow_busy_steering=iteration + 1 < spec.max_iterations,
                allow_continuation=(
                    response.finish_reason not in {"refusal", "content_filter"}
                ),
                wait_at_terminal=(
                    assistant_message is not None
                    and response.finish_reason
                    not in {"error", "length", "refusal", "content_filter"}
                ),
            )
            if should_continue:
                had_injections = True

            if hook.wants_streaming():
                await hook.on_stream_end(context, resuming=should_continue)

            if should_continue:
                length_recovery_parts.clear()
                await hook.after_iteration(context)
                continue

            if response.finish_reason == "error":
                if LLMProvider.is_arrearage_response(response):
                    final_content = _ARREARAGE_ERROR_MESSAGE
                else:
                    final_content = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                stop_reason = "error"
                error = final_content
                placeholder = build_assistant_message(_PERSISTED_MODEL_ERROR_PLACEHOLDER)
                entry = spec.source_writer.append_message(placeholder) if spec.source_writer is not None else None
                if request_state.source is None:
                    self._append_model_error_placeholder(messages)
                else:
                    self._append_bound_message(messages, placeholder, request_state.source,
                        entry['_event_id'] if entry is not None else None)
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                should_continue, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles, source=request_state.source,
                    phase="after LLM error",
                    allow_busy_steering=iteration + 1 < spec.max_iterations,
                )
                if should_continue:
                    had_injections = True
                    length_recovery_parts.clear()
                    continue
                failure_error_kind = LLMProvider.public_error_kind(response)
                break
            if is_blank_text(clean):
                final_content = EMPTY_FINAL_RESPONSE_MESSAGE
                stop_reason = "empty_final_response"
                error = final_content
                placeholder = build_assistant_message(final_content)
                entry = spec.source_writer.append_message(placeholder) if spec.source_writer is not None else None
                if request_state.source is None:
                    self._append_final_message(messages, final_content)
                else:
                    self._append_bound_message(messages, placeholder, request_state.source,
                        entry['_event_id'] if entry is not None else None)
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                should_continue, injection_cycles = await self._try_drain_injections(
                    spec, messages, None, injection_cycles, source=request_state.source,
                    phase="after empty response",
                    allow_busy_steering=iteration + 1 < spec.max_iterations,
                )
                if should_continue:
                    had_injections = True
                    length_recovery_parts.clear()
                    continue
                break

            self._append_bound_message(messages,
                assistant_message
                or conversation_state.project_response_message(
                    build_assistant_message(
                        clean,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    ),
                    response,
                ), request_state.source, recorded_response.source_event_id,
            )
            await self._emit_checkpoint(
                spec,
                {
                    "phase": "final_response",
                    "iteration": iteration,
                    "model": spec.runtime.model,
                    "assistant_message": messages[-1],
                    "completed_tool_results": [],
                    "pending_tool_calls": [],
                    "provider_state": conversation_state.checkpoint(messages),
                },
            )
            if length_recovery_parts:
                final_content = (
                    "".join(length_recovery_parts)
                    + _restore_outer_whitespace(clean or "", original_content)
                ).strip()
            else:
                final_content = clean
            if response.finish_reason != "stop":
                stop_reason = response.finish_reason or "error"
            context.final_content = final_content
            context.stop_reason = stop_reason
            await hook.after_iteration(context)
            provider_state_settlement = conversation_state.observe_response(
                response,
                messages,
                adopt_candidate_state=spec.source_writer is None,
            )
            break
        else:
            stop_reason = "max_iterations"
            # Drain any remaining injections so they are appended to the
            # conversation history instead of being re-published as
            # independent inbound messages by _dispatch's finally block.
            # We include them before the no-tools finalization pass so the
            # final response can account for every known follow-up.
            drained_after_max_iterations, injection_cycles = await self._try_drain_injections(
                spec, messages, None, injection_cycles, source=request_state.source,
                phase="after max_iterations",
                allow_busy_steering=False,
            )
            if drained_after_max_iterations:
                had_injections = True
            terminal_content = None
            terminal_event_id = None
            if spec.finalize_on_max_iterations:
                terminal_content, usage, terminal_event_id = await self._try_finalize_after_max_iterations(
                    spec,
                    hook,
                    messages,
                    usage,
                    request_state=request_state,
                    round_usages=round_usages,
                )
            synthetic_terminal = terminal_content is None
            if terminal_content is None:
                terminal_content = self._max_iterations_fallback(spec)
            if length_recovery_parts:
                terminal_tail = f"\n\n{terminal_content.lstrip()}"
                final_content = (
                    "".join(length_recovery_parts).rstrip() + terminal_tail
                ).strip()
                pending_stream_content = terminal_tail
            else:
                final_content = terminal_content
            terminal_message = build_assistant_message(terminal_content)
            if spec.source_writer is not None and synthetic_terminal:
                terminal_event_id = spec.source_writer.append_message(terminal_message)['_event_id']
            if request_state.source is None:
                self._append_final_message(messages, terminal_content)
            else:
                self._append_bound_message(messages, terminal_message, request_state.source, terminal_event_id)

        artifact_handoff_receipt = None
        if spec.artifact_handoff_collector is not None:
            from nanobot.session.artifact_grants import ArtifactHandoffCollector

            collector = spec.artifact_handoff_collector
            if not isinstance(collector, ArtifactHandoffCollector):
                raise SourceValidationError('Invalid artifact handoff collector')
            artifact_handoff_receipt = collector.finalize(stop_reason=stop_reason)

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            round_usages=round_usages,
            stop_reason=stop_reason,
            policy_metadata=policy_metadata,
            error=error,
            failure_error_kind=failure_error_kind,
            tool_events=tool_events,
            had_injections=had_injections,
            pending_stream_content=pending_stream_content,
            # ``finish`` can manufacture a checkpoint from a previously
            # committed state.  A runner is not a durable boundary, so expose
            # only an explicit settlement staged for this terminal response.
            provider_state=None,
            provider_state_settlement=provider_state_settlement,
            artifact_handoff_receipt=artifact_handoff_receipt,
            summary_checkpoint=(
                request_state.compaction.summary_checkpoint
                if request_state.compaction is not None
                else None
            ),
            provider_compaction_applied=request_state.provider_compaction_applied,
        )

    @staticmethod
    def _record_source_response(spec: AgentRunSpec, response: LLMResponse) -> str | None:
        writer = spec.source_writer
        if writer is None or response.finish_reason == "error":
            return
        _, content = extract_reasoning(response.reasoning_content, response.thinking_blocks, response.content)
        if response.should_execute_tools:
            event = writer.begin_tool_batch(build_assistant_message(content or "",
                tool_calls=[call.to_openai_tool_call() for call in response.tool_calls]))
            return event["_event_id"]
        elif not is_blank_text(content):
            event = writer.append_message(build_assistant_message(content))
            return event["_event_id"]
        return None

    def _build_request_kwargs(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "model": spec.runtime.model,
            "retry_mode": spec.provider_retry_mode,
        }
        generation = spec.runtime.generation
        kwargs["temperature"] = generation.temperature
        kwargs["max_tokens"] = generation.max_tokens
        kwargs["reasoning_effort"] = generation.reasoning_effort
        return kwargs

    @staticmethod
    def _source_view(state: ModelRequestState, messages: list[dict[str, Any]],
                     view: SourceTranscript | None) -> SourceTranscript | None:
        if state.source is None:
            return None
        view = view or state.source.transcript
        if view.messages() != messages:
            raise SourceValidationError('Model request differs from bound source projection')
        return view

    @staticmethod
    def _settle_source_ticket(state: ModelRequestState, ticket: SourceRequestTicket | None,
                              response: LLMResponse, *, malformed: bool = False,
                              no_tools: bool = False) -> None:
        if ticket is None or state.source is None:
            return
        valid = response.finish_reason in {'stop', 'length', 'tool_calls', 'function_call'}
        if malformed or (no_tools and response.has_tool_calls):
            valid = False
        if valid:
            state.source.acknowledge(ticket)
        else:
            state.source.discard(ticket)

    def _record_and_settle(self, spec: AgentRunSpec, state: ModelRequestState,
                           ticket: SourceRequestTicket | None, response: LLMResponse,
                           *, malformed: bool = False, no_tools: bool = False) -> RecordedResponse:
        try:
            recorded = RecordedResponse(response, self._record_source_response(spec, response))
        except BaseException:
            if ticket is not None and state.source is not None:
                state.source.discard(ticket)
            raise
        self._settle_source_ticket(state, ticket, response, malformed=malformed, no_tools=no_tools)
        return recorded

    @staticmethod
    def _check_busy_stop(spec: AgentRunSpec) -> None:
        if spec.turn_context is not None and spec.turn_context.busy_turn is not None:
            spec.turn_context.busy_turn.check_stop()

    async def _request_model(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        hook: AgentHook,
        context: AgentHookContext,
        *,
        request_state: ModelRequestState,
        malformed_retry: bool = False,
        source_view: SourceTranscript | None = None,
        transcript: list[dict[str, Any]] | None,
    ) -> tuple[RecordedResponse, LLMUsage]:
        self._check_busy_stop(spec)
        source_view = self._source_view(request_state, messages, source_view)
        timeout_s = self._resolve_llm_timeout_s(spec)
        tool_definitions = spec.tools.get_definitions()
        messages, provider_context = await self.context_governor.prepare_request(
            request_state,
            messages,
            tool_definitions=tool_definitions,
            transcript=transcript,
            source_view=source_view,
        )
        source_view = self._source_view(request_state, messages, request_state.source_view)

        self._check_busy_stop(spec)
        ticket = request_state.source.issue(source_view) if request_state.source is not None and source_view is not None else None
        kwargs = self._build_request_kwargs(
            spec,
            messages,
            tools=tool_definitions,
        )
        wants_streaming = hook.wants_streaming()

        active_hosted_tools: dict[str, dict[str, Any]] = {}
        native_reasoning_open = False
        native_reasoning_close_task: asyncio.Task[None] | None = None
        request_started_at = 0.0
        first_output_at: float | None = None
        generation_started_at: float | None = None
        generation_elapsed_s = 0.0

        def _generation_delta(delta: str) -> None:
            nonlocal first_output_at, generation_started_at
            if not delta:
                return
            now = time.perf_counter()
            if first_output_at is None:
                first_output_at = now
            if generation_started_at is None:
                generation_started_at = now

        def _pause_generation() -> None:
            nonlocal generation_elapsed_s, generation_started_at
            if generation_started_at is None:
                return
            generation_elapsed_s += max(0.0, time.perf_counter() - generation_started_at)
            generation_started_at = None

        async def _close_native_reasoning() -> None:
            nonlocal native_reasoning_open, native_reasoning_close_task
            if native_reasoning_close_task is None:
                if not native_reasoning_open:
                    return
                native_reasoning_open = False
                native_reasoning_close_task = asyncio.create_task(
                    hook.emit_reasoning_end()
                )

            close_task = native_reasoning_close_task
            cancellation: asyncio.CancelledError | None = None
            while not close_task.done():
                try:
                    await asyncio.shield(close_task)
                except asyncio.CancelledError as exc:
                    cancellation = cancellation or exc
            try:
                close_task.result()
            finally:
                if native_reasoning_close_task is close_task:
                    native_reasoning_close_task = None
            if cancellation is not None:
                raise cancellation

        async def _provider_tool_event(event: dict[str, Any]) -> None:
            if event.get("kind") != "hosted_tool":
                return
            await _close_native_reasoning()
            await hook.on_provider_tool_event(context, event)
            call_id = event.get("call_id")
            if not call_id:
                return
            call_id = str(call_id)
            if event.get("phase") == "start":
                active_hosted_tools[call_id] = dict(event)
            elif event.get("phase") in {"end", "error"}:
                active_hosted_tools.pop(call_id, None)

        if wants_streaming:
            thinking_buf = ""

            async def _stream(delta: str) -> None:
                _generation_delta(delta)
                if delta:
                    context.streamed_content = True
                    await _close_native_reasoning()
                await hook.on_stream(context, delta)

            async def _thinking(delta: str) -> None:
                nonlocal native_reasoning_open, thinking_buf
                if not delta:
                    return
                _generation_delta(delta)
                prev_clean = strip_reasoning_tags(thinking_buf)
                thinking_buf += delta
                new_clean = strip_reasoning_tags(thinking_buf)
                incremental = new_clean[len(prev_clean):]
                if incremental:
                    context.streamed_reasoning = True
                    native_reasoning_open = True
                    await hook.emit_reasoning(incremental)

            async def _stream_recover() -> None:
                _pause_generation()
                await _close_native_reasoning()
                await hook.on_stream_end(context, resuming=True)

            coro = spec.runtime.provider.chat_stream_with_retry(
                **kwargs,
                provider_context=provider_context,
                on_content_delta=_stream,
                on_thinking_delta=_thinking,
                on_tool_call_delta=_provider_tool_event,
                on_stream_recover=_stream_recover,
            )
        else:
            coro = spec.runtime.provider.chat_with_retry(
                **kwargs,
                provider_context=provider_context,
            )

        # Streaming requests also have provider-level idle timeouts
        # (NANOBOT_STREAM_IDLE_TIMEOUT_S), but a stream that keeps producing
        # very slow deltas can still run forever. Use a more generous wall-clock
        # timeout for streaming while preserving NANOBOT_LLM_TIMEOUT_S=0 as an
        # opt-out for all LLM wall-clock timeouts.
        outer_timeout_s = (
            max(300.0, timeout_s * 2)
            if wants_streaming and timeout_s is not None
            else timeout_s
        )
        request_started_at = time.perf_counter()
        try:
            response = (
                await coro if outer_timeout_s is None
                else await asyncio.wait_for(coro, timeout=outer_timeout_s)
            )
            self._check_busy_stop(spec)
        except asyncio.CancelledError:
            if ticket is not None and request_state.source is not None:
                request_state.source.discard(ticket)
            _pause_generation()
            await _close_native_reasoning()
            raise
        except asyncio.TimeoutError:
            if outer_timeout_s is None:
                response = LLMResponse(
                    content="Error calling LLM: stream stalled",
                    finish_reason="error",
                    error_kind="timeout",
                )
            else:
                response = LLMResponse(
                    content=f"Error calling LLM: timed out after {outer_timeout_s:g}s",
                    finish_reason="error",
                    error_kind="timeout",
                )
        except Exception:
            if ticket is not None and request_state.source is not None:
                request_state.source.discard(ticket)
            raise
        dropped, all_dropped, original_finish_reason = self._drop_malformed_tool_calls(response)
        recorded = self._record_and_settle(spec, request_state, ticket, response, malformed=dropped > 0)
        _pause_generation()
        await _close_native_reasoning()
        if first_output_at is not None:
            response.ttft_ms = max(0, round((first_output_at - request_started_at) * 1000))
        if generation_elapsed_s > 0:
            response.generation_ms = max(1, round(generation_elapsed_s * 1000))
        await self.context_governor.summarize_provider_compaction(
            request_state,
            response,
            current_request_boundary=(len(transcript) if transcript is not None else None),
        )
        request_state.provider_compaction_applied |= response.provider_compaction_applied
        round_usage = self._record_request_usage(spec, request_state, response)
        # chat_stream_with_retry may recover internally, so only fail unfinished
        # hosted calls after the provider returns its final error response.
        if response.finish_reason == "error":
            for event in list(active_hosted_tools.values()):
                await _provider_tool_event({
                    **event,
                    "phase": "error",
                    "result": None,
                    "error": response.content
                    or "Model request failed before the provider-hosted tool completed.",
                })
        if (
            all_dropped
            and original_finish_reason in ("tool_calls", "function_call")
            and not malformed_retry
        ):
            logger.warning(
                "Retrying LLM request after all {} malformed tool call(s) were dropped",
                dropped,
            )
            note = self._malformed_tool_call_retry_note(response.content)
            retry_messages = [*messages, note]
            retry_view = source_view.with_runtime(RuntimeNode(uuid.uuid4().hex, note)) if source_view is not None else None
            retry_response, retry_usage = await self._request_model(
                spec, retry_messages, hook, context,
                request_state=request_state,
                malformed_retry=True,
                source_view=retry_view,
                transcript=None,
            )
            return retry_response, round_usage + retry_usage
        if (
            all_dropped
            and original_finish_reason in ("tool_calls", "function_call")
            and malformed_retry
        ):
            logger.warning(
                "Malformed tool calls persisted after retry; falling back to no-tools request",
            )
            note = self._malformed_tool_call_retry_note(response.content)
            fallback_messages = [*messages, note]
            fallback_view = source_view.with_runtime(RuntimeNode(uuid.uuid4().hex, note)) if source_view is not None else None
            fallback_response = await self._request_no_tools(
                spec,
                fallback_messages,
                request_state=request_state, source_view=fallback_view,
            )
            fallback_usage = self._record_request_usage(
                spec,
                request_state,
                fallback_response.response,
            )
            return fallback_response, round_usage + fallback_usage
        return recorded, round_usage

    @staticmethod
    def _drop_malformed_tool_calls(
        response: LLMResponse,
    ) -> tuple[int, bool, str | None]:
        """Strip tool calls whose name is missing/non-string from the response.

        Returns (dropped_count, all_dropped, original_finish_reason).

        A degenerate call (name=None or "") cannot be executed, and if it were
        persisted into the assistant message it would be replayed on every
        subsequent turn, causing upstream validation errors
        (``tool_use.name: Input should be a valid string``) that permanently
        wedge the session. Dropping it here keeps it out of execution, the
        assistant message, and the saved history in one place.
        """
        calls = getattr(response, "tool_calls", None)
        if not calls:
            return (0, False, getattr(response, "finish_reason", None))
        valid = [tc for tc in calls if tc.has_valid_name()]
        if len(valid) == len(calls):
            return (0, False, getattr(response, "finish_reason", None))
        dropped = len(calls) - len(valid)
        original_finish_reason = getattr(response, "finish_reason", None)
        logger.warning(
            "Dropped {} malformed tool call(s) with missing/non-string name "
            "from LLM response (finish_reason={!r})",
            dropped,
            original_finish_reason,
        )
        response.tool_calls = valid
        # The opaque candidate still contains every raw function_call item.
        # Advancing it after dropping even one call would replay an unmatched
        # call without a corresponding tool output on the next request.
        response.provider_state = None
        if not valid:
            response.finish_reason = "stop"
        return (dropped, not valid, original_finish_reason)

    @staticmethod
    def _malformed_tool_call_retry_note(assistant_text: str | None) -> dict[str, Any]:
        note = (
            "The previous model response attempted to call tools, but every tool call "
            "was malformed: the tool_use blocks had missing or non-string tool names. "
            "Do not answer with a promise to use tools. Either call the required tools again "
            "using valid tool names from the provided tool list and JSON object inputs, or give "
            "a final answer only if no tool is required."
        )
        if assistant_text:
            note += (
                f"\n\nPrevious assistant text before the malformed calls:\n"
                f"{assistant_text}"
            )
        return {"role": "user", "content": note}

    @staticmethod
    def _malformed_tool_call_retry_messages(messages: list[dict[str, Any]],
                                            assistant_text: str | None) -> list[dict[str, Any]]:
        return [*messages, AgentRunner._malformed_tool_call_retry_note(assistant_text)]

    async def _request_finalization_retry(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        request_state: ModelRequestState,
        transcript: list[dict[str, Any]],
    ) -> RecordedResponse:
        note = build_finalization_retry_message()
        retry_messages = [*messages, note]
        base_view = request_state.source_view
        view = base_view.with_runtime(RuntimeNode(uuid.uuid4().hex, note)) if base_view is not None else None
        response = await self._request_no_tools(
            spec,
            retry_messages,
            request_state=request_state, source_view=view,
            transcript=transcript,
        )
        request_state.conversation.observe_response(
            response.response,
            transcript,
            adopt_candidate_state=False,
        )
        return response

    @staticmethod
    def _finalization_retry_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        retry_messages = list(messages)
        retry_messages.append(build_finalization_retry_message())
        return retry_messages

    async def _try_finalize_after_max_iterations(
        self,
        spec: AgentRunSpec,
        hook: AgentHook,
        messages: list[dict[str, Any]],
        usage: LLMUsage | None,
        *,
        request_state: ModelRequestState,
        round_usages: list[LLMUsage],
    ) -> tuple[str | None, LLMUsage | None, str | None]:
        compaction = request_state.compaction
        request_messages = (
            compaction.request_messages(messages)
            if compaction is not None
            else messages
        )
        note = build_budget_exhausted_finalization_message()
        retry_messages = [*request_messages, note]
        base_view = request_state.source.transcript if request_state.source is not None else None
        view = base_view.with_runtime(RuntimeNode(uuid.uuid4().hex, note)) if base_view is not None else None
        try:
            recorded = await self._request_no_tools(
                spec,
                retry_messages,
                request_state=request_state, source_view=view,
                transcript=messages if compaction is not None else None,
            )
        except Exception:
            if spec.source_writer is not None:
                raise
            logger.exception(
                "Budget-exhausted finalization failed for {}; using fallback",
                spec.session_key or "default",
            )
            return None, usage, None

        response = recorded.response
        raw_usage = self._record_request_usage(spec, request_state, response)
        round_usages.append(raw_usage)
        usage = self._merge_usage(usage, raw_usage)
        if response.finish_reason == "error" or response.has_tool_calls:
            logger.warning(
                "Budget-exhausted finalization returned finish_reason='{}' "
                "with {} tool call(s) for {}; using fallback",
                response.finish_reason,
                len(response.tool_calls),
                spec.session_key or "default",
            )
            return None, usage, None

        context = AgentHookContext(
            iteration=spec.max_iterations,
            messages=messages,
            response=response,
            usage=raw_usage,
            session_key=spec.session_key,
        )
        clean = hook.finalize_content(context, response.content)
        if is_blank_text(clean):
            return None, usage, None
        return clean, usage, recorded.source_event_id

    async def _request_no_tools(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        request_state: ModelRequestState,
        transcript: list[dict[str, Any]] | None = None,
        source_view: SourceTranscript | None = None,
    ) -> RecordedResponse:
        self._check_busy_stop(spec)
        source_view = self._source_view(request_state, messages, source_view)
        messages, provider_context = await self.context_governor.prepare_request(
            request_state,
            messages,
            tool_definitions=None,
            transcript=transcript,
            source_view=source_view,
        )
        source_view = self._source_view(request_state, messages, request_state.source_view)
        self._check_busy_stop(spec)
        ticket = request_state.source.issue(source_view) if request_state.source is not None and source_view is not None else None
        kwargs = self._build_request_kwargs(
            spec,
            messages,
            tools=None,
        )
        coro = spec.runtime.provider.chat_with_retry(
            **kwargs,
            provider_context=provider_context,
        )
        timeout_s = self._resolve_llm_timeout_s(spec)
        try:
            response = (
                await coro
                if timeout_s is None
                else await asyncio.wait_for(coro, timeout=timeout_s)
            )
            self._check_busy_stop(spec)
        except asyncio.TimeoutError:
            response = LLMResponse(
                content=f"Error calling LLM: timed out after {timeout_s:g}s",
                finish_reason="error",
                error_kind="timeout",
            )
        except BaseException:
            if ticket is not None and request_state.source is not None:
                request_state.source.discard(ticket)
            raise
        recorded = self._record_and_settle(spec, request_state, ticket, response, no_tools=True)
        await self.context_governor.summarize_provider_compaction(
            request_state,
            response,
            current_request_boundary=(len(transcript) if transcript is not None else None),
        )
        request_state.provider_compaction_applied |= response.provider_compaction_applied
        return recorded

    @staticmethod
    def _resolve_llm_timeout_s(spec: AgentRunSpec) -> float | None:
        """Resolve the wall-clock limit shared by every model request path."""
        timeout_s = spec.llm_timeout_s
        if timeout_s is None:
            # Default to a finite timeout to avoid per-session lock starvation when an LLM
            # request hangs indefinitely (e.g. gateway/network stall).
            # Set NANOBOT_LLM_TIMEOUT_S=0 to disable.
            raw = os.environ.get("NANOBOT_LLM_TIMEOUT_S", "300").strip()
            try:
                timeout_s = float(raw)
            except (TypeError, ValueError):
                timeout_s = 300.0
        return timeout_s if timeout_s > 0 else None

    @staticmethod
    def _budget_exhausted_finalization_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        retry_messages = list(messages)
        retry_messages.append(build_budget_exhausted_finalization_message())
        return retry_messages

    @staticmethod
    def _max_iterations_fallback(spec: AgentRunSpec) -> str:
        if spec.max_iterations_message:
            return spec.max_iterations_message.format(
                max_iterations=spec.max_iterations,
            )
        return render_template(
            "agent/max_iterations_message.md",
            strip=True,
            max_iterations=spec.max_iterations,
        )

    def _usage_or_estimate(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        response: LLMResponse,
        *,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> LLMUsage:
        usage = response.usage
        if response.finish_reason == "error":
            if usage is None or usage.total_tokens == 0:
                usage = LLMUsage.empty_request()
        elif usage is None or usage.total_tokens == 0:
            usage = self._estimate_response_usage(
                spec,
                messages,
                response,
                tool_definitions=tool_definitions,
            )
        return usage.with_timing(
            generation_ms=response.generation_ms,
            ttft_ms=response.ttft_ms,
        )

    def _record_request_usage(
        self,
        spec: AgentRunSpec,
        state: ModelRequestState,
        response: LLMResponse,
    ) -> LLMUsage:
        assert state.messages is not None
        state.usage = self._usage_or_estimate(
            spec,
            state.messages,
            response,
            tool_definitions=state.tool_definitions,
        )
        return state.usage

    def _estimate_response_usage(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        response: LLMResponse,
        *,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> LLMUsage:
        prompt_tokens, _ = estimate_prompt_tokens_chain(
            spec.runtime.provider,
            spec.runtime.model,
            messages,
            tool_definitions,
        )
        assistant_message = build_assistant_message(
            response.content or "",
            tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
            reasoning_content=response.reasoning_content,
            thinking_blocks=response.thinking_blocks,
        )
        completion_tokens = estimate_message_tokens(assistant_message)
        return LLMUsage.estimated(
            input_tokens=max(0, prompt_tokens),
            output_tokens=max(0, completion_tokens),
        )

    @staticmethod
    def _merge_usage(
        left: LLMUsage | None,
        right: LLMUsage | None,
    ) -> LLMUsage | None:
        if left is None:
            return right
        if right is None:
            return left
        return left + right

    async def _emit_checkpoint(
        self,
        spec: AgentRunSpec,
        payload: dict[str, Any],
    ) -> None:
        callback = spec.checkpoint_callback
        if callback is not None:
            await callback(payload)

    @staticmethod
    def _append_final_message(messages: list[dict[str, Any]], content: str | None) -> None:
        if not content:
            return
        if (
            messages
            and messages[-1].get("role") == "assistant"
            and not messages[-1].get("tool_calls")
        ):
            if messages[-1].get("content") == content:
                return
            messages[-1] = build_assistant_message(content)
            return
        messages.append(build_assistant_message(content))

    @staticmethod
    def _append_model_error_placeholder(messages: list[dict[str, Any]]) -> None:
        if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("tool_calls"):
            return
        messages.append(build_assistant_message(_PERSISTED_MODEL_ERROR_PLACEHOLDER))
