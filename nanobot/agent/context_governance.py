"""Model-message governance and compaction for agent runner requests.

This module owns model-facing message shaping, request pressure, H/delta
compaction state, and tool-result content normalization. It may return copied
messages or persisted-result placeholders, but it must not mutate an existing
session history list in place.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from loguru import logger

from nanobot.agent.context import TranscriptInput
from nanobot.events import NO_EVENTS, ContextCompactionEvent, EventSink
from nanobot.providers.base import (
    GenerationSettings,
    LLMResponse,
    LLMUsage,
    ProviderCallContext,
    ProviderConversationState,
)
from nanobot.providers.conversation_state import (
    ConversationRequestBinding,
    ProviderConversationStateController,
    allows_conversation_message_merge,
)
from nanobot.providers.image_history import recent_image_messages
from nanobot.providers.provider_policy import RequestAccountingPolicy
from nanobot.providers.request_budget import (
    DEFAULT_FINALIZED_REQUEST_SAFETY_BUFFER,
    BackendContextOverflow,
    LocalUsageAnchor,
    PreparedRequestBudgetPreview,
    PreparedRequestRoute,
    ProviderRequestRejected,
    RequestPayloadBytesExceeded,
    TransportIdentity,
    TransportMeasurement,
    derive_provider_request_limits,
)
from nanobot.runtime_context import (
    RUNTIME_CONTEXT_MESSAGE_META,
    detach_runtime_context,
    reattach_runtime_context,
)
from nanobot.session.artifact_projection import ArtifactProjector
from nanobot.session.checkpoint_admission import CheckpointAdmission
from nanobot.session.history_visibility import is_hidden_history_message
from nanobot.session.source import SourceValidationError, prefix_digest
from nanobot.session.source_compaction import SourceCompactionCandidate
from nanobot.session.source_projection import CheckpointNode, RuntimeNode, SourceNode
from nanobot.session.source_request import RequestNode, SourceTranscript
from nanobot.session.summary import (
    SUMMARY_CONTINUATION_TEXT,
    SessionSummaryCheckpoint,
)
from nanobot.utils.helpers import (
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    find_legal_message_start,
    maybe_persist_tool_result,
    truncate_text,
    truncate_text_to_tokens,
)
from nanobot.utils.runtime import ensure_nonempty_tool_result

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.providers.base import LLMProvider
    from nanobot.session.checkpoint_dependencies import SummaryDependencyProof, SummaryEvidence
    from nanobot.session.source_request import SourceRequestLedger
    from nanobot.session.turn_writer import SourceTurnWriter
    from nanobot.utils.llm_runtime import LLMRuntime

TranscriptBuilder = Callable[[TranscriptInput], list[dict[str, Any]]]
HistoryConsolidator = Callable[
    [list[dict[str, Any]], str | None],
    Awaitable[str | None],
]
ProviderCompactionConsolidator = Callable[
    [ProviderConversationState, list[dict[str, Any]], str | None],
    Awaitable[str | None],
]

SNIP_SAFETY_BUFFER = DEFAULT_FINALIZED_REQUEST_SAFETY_BUFFER
# read_file has its own bound; exempt it to avoid persist->read->persist loops.
TOOL_RESULT_OFFLOAD_EXEMPT_TOOLS = frozenset({"read_file"})
BACKFILL_CONTENT = "[Tool result unavailable — call was interrupted or lost]"
PLACEHOLDER_TEXTS = frozenset({
    "[Previous assistant message omitted.]",
})


class ContextWindowExceededError(RuntimeError):
    """Raised before a locally fitted request that still exceeds its budget."""

    def __init__(
        self,
        *,
        session_key: str | None,
        estimated_tokens: int,
        input_budget: int,
        source: str,
    ) -> None:
        self.session_key = session_key
        self.estimated_tokens = estimated_tokens
        self.input_budget = input_budget
        self.source = source
        super().__init__(
            "Model input still exceeds the local context budget after request fitting "
            f"for {session_key or 'default'}: {estimated_tokens}/{input_budget} via {source}"
        )


class SourceCompactionUnavailableError(SourceValidationError):
    """A bounded summary cannot be prepared or completed for this request."""

    def __init__(self, message: str, *, reason: str = 'no_progress') -> None:
        self.reason = reason
        super().__init__(message)


def _tool_call_name_is_valid(tool_call: Any) -> bool:
    """Whether a persisted OpenAI-style tool_call carries a usable name.

    Mirrors ``ToolCallRequest.has_valid_name`` for the dict shape stored in
    message history: a degenerate call with ``name=None`` / ``""`` cannot be
    executed and is rejected by upstream APIs if replayed.
    """
    if not isinstance(tool_call, dict):
        return False
    tool_call_data = cast(dict[str, Any], tool_call)
    fn = tool_call_data.get("function")
    name = cast(dict[str, Any], fn).get("name") if isinstance(fn, dict) else tool_call_data.get("name")
    return isinstance(name, str) and bool(name)


@dataclass(slots=True)
class ContextGovernanceConfig:
    provider: LLMProvider
    model: str
    tools: ToolRegistry
    workspace: Path | None
    session_key: str | None
    max_tool_result_chars: int
    context_window_tokens: int | None = None
    max_tokens: int | None = None
    generation: GenerationSettings | None = None
    preserve_source: bool = False


@dataclass(slots=True)
class ContextCompactionState:
    """Track accepted provider input H separately from the unsent delta."""

    raw_messages: list[dict[str, Any]]
    accepted_messages: list[dict[str, Any]]
    raw_accepted_boundary: int
    active_summary: str | None
    transcript_input: TranscriptInput
    transcript_builder: TranscriptBuilder
    consolidate_history: HistoryConsolidator
    consolidate_provider_compaction: ProviderCompactionConsolidator | None
    summary_checkpoint: SessionSummaryCheckpoint | None = None

    @classmethod
    def from_transcript(
        cls,
        transcript_input: TranscriptInput,
        transcript_builder: TranscriptBuilder,
        consolidate_history: HistoryConsolidator | None,
        consolidate_provider_compaction: ProviderCompactionConsolidator | None,
    ) -> tuple[list[dict[str, Any]], ContextCompactionState | None]:
        """Build the raw transcript and its initial H/delta boundary."""
        messages = list(transcript_builder(transcript_input))
        if consolidate_history is None:
            return messages, None
        accepted_history_boundary = 1 + len(transcript_input.history)
        return messages, cls(
            raw_messages=messages,
            accepted_messages=deepcopy(messages[:accepted_history_boundary]),
            raw_accepted_boundary=accepted_history_boundary,
            active_summary=(
                transcript_input.session_summary["text"]
                if transcript_input.session_summary is not None
                else None
            ),
            transcript_input=transcript_input,
            transcript_builder=transcript_builder,
            consolidate_history=consolidate_history,
            consolidate_provider_compaction=consolidate_provider_compaction,
        )

    def request_messages(
        self,
        raw_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            *deepcopy(self.accepted_messages),
            *deepcopy(raw_messages[self.raw_accepted_boundary:]),
        ]

    def delta_after_accepted(
        self,
        request_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return deepcopy(request_messages[len(self.accepted_messages):])

    def accept_request(
        self,
        model_messages: list[dict[str, Any]],
        *,
        raw_boundary: int,
    ) -> None:
        """Advance H after the provider has received one request."""
        self.accepted_messages = deepcopy(model_messages)
        self.raw_accepted_boundary = raw_boundary


@dataclass(slots=True)
class SourceCompactionMetrics:
    scope: str
    before_tokens: int
    boundary: int = 0
    after_tokens: int | None = None
    calls: int = 0
    usage: LLMUsage | None = None
    usage_unknown: bool = False
    status: str = 'started'
    latency_ms: int = 0
    failure_reason: str | None = None
    root_reason: str | None = None
    prefix_mode: str | None = None
    prefix_fallback_reason: str | None = None
    prefix_source_nodes: int = 0
    prefix_estimated_tokens: int | None = None
    converted_prefix_match: bool | None = None
    payload_bytes: int | None = None
    selected_capacity: int | None = None
    completion_reserve: int | None = None
    safety_buffer: int | None = None
    measurement_source: str | None = None
    summary_generation_allowance: int | None = None
    summary_retained_tokens: int | None = None
    summary_reserve_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class SourceRecoveryOutcome:
    """One bounded recovery result or a content-free reason it was unavailable."""

    prepared_messages: list[dict[str, Any]] | None
    provider_context: ProviderCallContext | None
    root_reason: Literal['local_estimated_admission', 'backend_context_overflow']
    unavailable_reason: str | None = None

    @property
    def recovered(self) -> bool:
        return self.prepared_messages is not None and self.unavailable_reason is None


@dataclass(slots=True)
class SourceCompactionState:
    """Invocation-owned publication state; local scopes never write root metadata."""

    writer: SourceTurnWriter
    runtime: LLMRuntime
    raw_messages: list[dict[str, Any]]
    stream_id: str | None
    admission: CheckpointAdmission = field(default_factory=CheckpointAdmission)
    local_boundary: int = 0
    local_checkpoint: CheckpointNode | None = None
    artifact_projection_unavailable: bool = False
    attempts: list[SourceCompactionMetrics] = field(default_factory=list)
    idle_timeout_seconds: float = 300.0
    overall_timeout_seconds: float = 900.0
    pressure_trigger_ratio: float = 0.9
    target_ratio: float = 0.6
    summary_generation_tokens: int | None = None
    summary_retained_tokens: int | None = None


@dataclass(frozen=True)
class SourceCompactionAttempt:
    candidate: SourceCompactionCandidate
    projection_epoch: int
    request_notes: tuple[RuntimeNode, ...]
    tool_definitions: list[dict[str, Any]] | None


@dataclass(slots=True)
class ModelRequestState:
    """Context state shared by every provider request in one runner turn."""

    config: ContextGovernanceConfig
    conversation: ProviderConversationStateController
    usage: LLMUsage | None = None
    local_usage_anchor: LocalUsageAnchor | None = field(default=None, repr=False)
    messages: list[dict[str, Any]] | None = None
    tool_definitions: list[dict[str, Any]] | None = None
    compaction: ContextCompactionState | None = None
    source: SourceRequestLedger | None = None
    source_view: SourceTranscript | None = None
    source_compaction: SourceCompactionState | None = None
    artifact_projector: ArtifactProjector | None = None
    provider_compaction_applied: bool = False
    events: EventSink = NO_EVENTS
    # The loop creates this from its admitted session, policy, and execution
    # fence.  Request preparation fills in the effective tool policy for each
    # dispatch; callers without that authority intentionally stay stateless.
    conversation_binding: ConversationRequestBinding | None = None


class ContextGovernor:
    """Own model-request context while preserving persisted history."""

    @staticmethod
    def admit_source_checkpoint(state: ModelRequestState, view: SourceTranscript, *,
                                tool_definitions: list[dict[str, Any]] | None) -> SourceTranscript:
        from nanobot.agent.tools.artifact import ReadArtifactTool

        source, compaction = state.source, state.source_compaction
        if source is None or compaction is None:
            return view
        if source.has_pending_requests:
            raise SourceValidationError('Cannot admit checkpoint with pending requests')
        notes = source.validate_view(view)
        active, epoch = source.transcript, source.projection_epoch
        if compaction.raw_messages != active.messages():
            raise SourceValidationError('Checkpoint admission differs from runner transcript')
        admission = compaction.admission
        tool = state.config.tools.get('read_artifact')
        if type(tool) is ReadArtifactTool:
            if tool.reader.writer is not compaction.writer:
                raise SourceValidationError('Checkpoint verifier differs from source invocation')
            if admission.reader is None:
                admission.reader = tool.reader
        enabled = (type(tool) is ReadArtifactTool and tool.reader is admission.reader and any(
            definition.get('type') == 'function'
            and definition.get('function', {}).get('name') == 'read_artifact'
            for definition in tool_definitions or []))
        snapshot = compaction.writer.owner.read_snapshot()
        updated = active
        if not admission.initialized:
            if compaction.stream_id is None:
                updated = admission.select(snapshot, active)
            else:
                admission.initialized = True
        checkpoint = admission.checkpoint if compaction.stream_id is None else compaction.local_checkpoint
        if checkpoint is not None and checkpoint.artifact_references:
            if not admission.validate_dependencies(snapshot, updated, enabled=enabled, checkpoint=checkpoint):
                updated = source.recovery_transcript
                admission.reject()
                compaction.local_boundary, compaction.local_checkpoint = 0, None
        source.validate_transcript(updated)
        if (source.has_pending_requests or source.projection_epoch != epoch or source.transcript != active
                or compaction.raw_messages != active.messages()):
            raise SourceValidationError('Source projection changed during checkpoint admission')
        if updated != active:
            source.replace_projection(updated)
            compaction.raw_messages[:] = updated.messages()
            state.usage = None
            state.local_usage_anchor = None
            state.messages = None
            state.source_view = None
            state.tool_definitions = None
            state.conversation.replace_transcript(compaction.raw_messages)
        return updated.with_runtime(*notes)

    @staticmethod
    def project_source_artifacts(state: ModelRequestState, view: SourceTranscript, *,
                                 tool_definitions: list[dict[str, Any]] | None) -> SourceTranscript:
        from nanobot.agent.tools.artifact import ReadArtifactTool

        source, compaction = state.source, state.source_compaction
        if source is None or compaction is None:
            return view
        tool = state.config.tools.get('read_artifact')
        if state.artifact_projector is None and type(tool) is ReadArtifactTool:
            if tool.reader.writer is not compaction.writer:
                raise SourceValidationError('Artifact projector does not match source invocation')
            state.artifact_projector = ArtifactProjector(tool.reader)
        projector = state.artifact_projector
        if projector is None:
            if any(isinstance(node, SourceNode) and node.artifact is not None for node in view.nodes):
                raise SourceValidationError('Artifact projection lacks admitted verifier')
            return view
        if source.has_pending_requests:
            raise SourceValidationError('Cannot offload with an outstanding request ticket')
        notes = source.validate_view(view)
        active, epoch = source.transcript, source.projection_epoch
        if compaction.raw_messages != active.messages():
            raise SourceValidationError('Artifact projection differs from runner transcript')
        had_projected = any(
            isinstance(node, SourceNode) and node.artifact is not None for node in active.nodes
        )
        enabled = (type(tool) is ReadArtifactTool and tool.reader is projector.reader and any(
            definition.get('type') == 'function'
            and definition.get('function', {}).get('name') == 'read_artifact'
            for definition in tool_definitions or []))
        if not enabled and not any(
                isinstance(node, SourceNode) and node.artifact is not None for node in active.nodes):
            return active.with_runtime(*notes)
        updated = projector.prepare(active, enabled=enabled)
        compaction.artifact_projection_unavailable = had_projected and not any(
            isinstance(node, SourceNode) and node.artifact is not None for node in updated.nodes
        )
        source.validate_transcript(updated)
        if (source.has_pending_requests or source.projection_epoch != epoch or source.transcript != active
                or compaction.raw_messages != active.messages()):
            raise SourceValidationError('Source projection changed during artifact publication')
        if updated != active:
            source.replace_projection(updated)
            compaction.raw_messages[:] = updated.messages()
            state.usage = None
            state.local_usage_anchor = None
            state.conversation.replace_transcript(compaction.raw_messages)
        return updated.with_runtime(*notes)

    @staticmethod
    def capture_source_compaction(state: ModelRequestState, view: SourceTranscript, *,
                                  tool_definitions: list[dict[str, Any]] | None) -> SourceCompactionAttempt:
        source, compaction = state.source, state.source_compaction
        if source is None or compaction is None or source.has_pending_requests:
            raise SourceValidationError('Source compaction requires a quiescent request ledger')
        notes = source.validate_view(view)
        active = source.transcript
        if compaction.raw_messages != active.messages():
            raise SourceValidationError('Runner transcript differs from active source projection')
        candidate = SourceCompactionCandidate.capture(compaction.writer.owner.read_snapshot(), active,
            eligible_ids=source.eligible_ids, stream_id=compaction.stream_id,
            local_boundary=compaction.local_boundary, local_checkpoint=compaction.local_checkpoint,
            accepted_boundary=compaction.admission.boundary, accepted_checkpoint=compaction.admission.checkpoint)
        return SourceCompactionAttempt(candidate, source.projection_epoch, notes, deepcopy(tool_definitions))

    @staticmethod
    def _dependency_reader(state: ModelRequestState, attempt: SourceCompactionAttempt):
        """Return the exact reader and retrieval schema carried by the main request."""
        from nanobot.agent.tools.artifact import ReadArtifactTool

        compaction = state.source_compaction
        if compaction is None:
            raise SourceValidationError('Source compaction state is unavailable')
        tool = state.config.tools.get('read_artifact')
        if type(tool) is not ReadArtifactTool or tool.reader.writer is not compaction.writer:
            raise SourceValidationError('Dependent source compaction requires an admitted artifact reader')
        if not ContextGovernor._has_active_retrieval_schema(state, attempt):
            raise SourceValidationError('Dependent source compaction requires the active retrieval schema')
        candidate = attempt.candidate
        if candidate.stream_id is None:
            if tool.reader.mode != 'root_history' or compaction.writer.parent_stream_id is not None:
                raise SourceValidationError('Dependent root compaction requires root reader admission')
        elif tool.reader.mode != 'isolated_stream' or compaction.writer.stream_id != candidate.stream_id:
            raise SourceValidationError('Dependent local compaction requires isolated reader admission')
        return tool.reader

    @staticmethod
    def _has_active_retrieval_schema(state: ModelRequestState,
                                     attempt: SourceCompactionAttempt) -> bool:
        from nanobot.agent.tools.artifact import ReadArtifactTool

        tool = state.config.tools.get('read_artifact')
        return (type(tool) is ReadArtifactTool and attempt.tool_definitions is not None
                and any(definition == tool.to_schema() for definition in attempt.tool_definitions))

    @staticmethod
    def _preflight_summary_dependencies(state: ModelRequestState, attempt: SourceCompactionAttempt,
                                        proof: SummaryDependencyProof) -> None:
        """Validate dependencies under the blob lock immediately before transport."""
        from nanobot.session.checkpoint_dependencies import (
            derive_summary_references,
            validate_summary_dependencies,
        )

        source, compaction = state.source, state.source_compaction
        if source is None or compaction is None or source.has_pending_requests:
            raise SourceValidationError('Source projection changed during compaction')
        if (source.projection_epoch != attempt.projection_epoch
                or source.transcript != attempt.candidate.transcript
                or compaction.raw_messages != attempt.candidate.transcript.messages()
                or compaction.stream_id != attempt.candidate.stream_id):
            raise SourceValidationError('Source projection changed during compaction')
        compaction.writer.owner.validate_dispatch()
        references = derive_summary_references(proof, attempt.candidate.snapshot)
        if not references:
            return
        reader = ContextGovernor._dependency_reader(state, attempt)

        with reader.store.transaction() as blobs:
            with reader.writer.store.execution_transaction(reader.writer.owner.key,
                                       reader.writer.owner):
                current = reader.writer.owner.read_snapshot()
                validate_summary_dependencies(proof, current, reader, blobs)

    def publish_source_compaction(self, state: ModelRequestState, attempt: SourceCompactionAttempt, *,
                                  boundary: int, summary: str,
                                  auxiliary_usage: dict[str, int] | None,
                                  retained_text_tokens: int | None = None,
                                  proof: SummaryDependencyProof | None = None,
                                  stream: bool = False,
                                  tool_choice: str | dict[str, Any] | None = None,
                                  provider_context: ProviderCallContext | None = None,
                                  deadline_monotonic: float | None = None) -> SourceTranscript:
        """Validate the complete payload, commit under captured CAS, then publish.

        There is no await between durable publication and replacing the ledger/raw
        list. A failure never retries generation, refreshes CAS, or edits source.
        """
        source, compaction = state.source, state.source_compaction
        if source is None or compaction is None:
            raise SourceValidationError('Source compaction state is unavailable')
        candidate = attempt.candidate
        if (source.has_pending_requests or source.projection_epoch != attempt.projection_epoch
                or source.transcript != candidate.transcript
                or compaction.raw_messages != candidate.transcript.messages()
                or compaction.stream_id != candidate.stream_id):
            raise SourceValidationError('Source projection changed during compaction')
        retained = 2048 if retained_text_tokens is None else retained_text_tokens
        if (type(retained) is not int or retained <= 0
                or not isinstance(cast(object, summary), str) or not summary.strip()
                or len(summary) > 64_000
                or truncate_text_to_tokens(summary, retained) != summary):
            raise SourceValidationError('Invalid bounded source summary')
        if auxiliary_usage is not None and (not auxiliary_usage or any(
                type(value) is not int or value < 0 for value in auxiliary_usage.values())):
            raise SourceValidationError('Invalid source auxiliary usage')
        # Only the validated non-None proof branch can produce references, so
        # both later reference-bearing paths retain that same concrete proof.
        references: tuple[str, ...] = ()
        if proof is not None:
            from nanobot.session.checkpoint_dependencies import (
                SummaryDependencyProof,
                derive_summary_references,
            )

            if (type(proof) is not SummaryDependencyProof or proof.candidate != candidate
                    or not proof.chunks or proof.chunks[-1].evidence.boundary != boundary
                    or proof.chunks[-1].summary != summary):
                raise SourceValidationError('Source compaction proof does not match the final fold')
            references = derive_summary_references(proof, candidate.snapshot)
        elif ((candidate.previous_checkpoint is not None
               and candidate.previous_checkpoint.artifact_references)
              or candidate.summary_evidence(boundary).retrievals
              or candidate.summary_evidence(boundary).projections):
            raise SourceValidationError('Dependent source compaction requires a complete proof')
        checkpoint = CheckpointNode(uuid4().hex, summary, references)
        active = candidate.rebuild(boundary, checkpoint)
        # Validate local publication before any durable mutation.
        source.validate_transcript(active)
        view = active.with_runtime(*attempt.request_notes)
        source.validate_transcript(view)
        if deadline_monotonic is not None and asyncio.get_running_loop().time() >= deadline_monotonic:
            raise SourceCompactionUnavailableError(
                'Source compaction operation deadline expired before publication',
                reason='summary_overall_timeout',
            )
        prepared = self.ensure_request_fits(
            state.config,
            view.messages(),
            tool_definitions=attempt.tool_definitions,
            stream=stream,
            tool_choice=tool_choice,
            provider_context=provider_context,
        )
        writer = compaction.writer
        writer.owner.validate_dispatch()
        if deadline_monotonic is not None and asyncio.get_running_loop().time() >= deadline_monotonic:
            raise SourceCompactionUnavailableError(
                'Source compaction operation deadline expired before publication',
                reason='summary_overall_timeout',
            )
        captured = candidate.snapshot
        committed = None
        if candidate.stream_id is None:
            proposal = {
                'schema_version': 2 if references else 1, 'digest_version': 1, 'summary_format_version': 1,
                'session_generation': captured.state.generation,
                'source_revision': captured.state.revision,
                'checkpoint_id': checkpoint.checkpoint_id,
                'parent_checkpoint_id': captured.state.checkpoint_id,
                'producing_model': compaction.runtime.model,
                'producing_provider': compaction.runtime.provider.provider_name,
                'digest_algorithm': 'sha256', 'created_at': datetime.now(timezone.utc).isoformat(),
                'summary': summary, 'artifact_references': list(references), 'boundary': boundary,
                'boundary_event_id': captured.events[boundary - 1]['_event_id'],
                'prefix_digest': prefix_digest(captured.events, boundary),
                'auxiliary_usage': auxiliary_usage,
                'projection_scope': 'root', 'projection_version': 1,
            }
            if references:
                reader = self._dependency_reader(state, attempt)
                configured = writer.store.with_artifact_store(reader.store)
                with reader.store.transaction() as blobs:
                    committed = configured.commit_dependent_context_checkpoint(
                        writer.owner.key, captured.state, proposal, cast(SummaryDependencyProof, proof),
                        owner=writer.owner, reader=reader, blobs=blobs,
                    )
            else:
                committed = writer.store.commit_context_checkpoint(
                    writer.owner.key, captured.state, proposal, owner=writer.owner,
                )
        elif references:
            from nanobot.session.checkpoint_dependencies import validate_summary_dependencies

            reader = self._dependency_reader(state, attempt)
            with reader.store.transaction() as blobs:
                with reader.writer.store.execution_transaction(reader.writer.owner.key,
                                           reader.writer.owner):
                    current = writer.owner.read_snapshot()
                    before, after = captured.state, current.state
                    if ((before.store_identity, before.session_key, before.incarnation, before.generation)
                            != (after.store_identity, after.session_key, after.incarnation, after.generation)
                            or [event for event in captured.events
                                if event.get('_stream_id') == candidate.stream_id]
                            != [event for event in current.events
                                if event.get('_stream_id') == candidate.stream_id]):
                        raise SourceValidationError('Local source changed during compaction')
                    validate_summary_dependencies(cast(SummaryDependencyProof, proof), current, reader, blobs)
        else:
            current = writer.owner.read_snapshot()
            before, after = captured.state, current.state
            if ((before.store_identity, before.session_key, before.incarnation, before.generation)
                    != (after.store_identity, after.session_key, after.incarnation, after.generation)
                    or [event for event in captured.events if event.get('_stream_id') == candidate.stream_id]
                    != [event for event in current.events if event.get('_stream_id') == candidate.stream_id]):
                raise SourceValidationError('Local source changed during compaction')
        source.replace_projection(active)
        compaction.raw_messages[:] = active.messages()
        compaction.admission.initialized = True
        if candidate.stream_id is None:
            compaction.admission.boundary, compaction.admission.checkpoint = boundary, checkpoint
            if committed is None:
                raise SourceValidationError('Source checkpoint commit did not return a snapshot')
            compaction.admission.fingerprint = committed.state.checkpoint_fingerprint
        if candidate.stream_id is not None:
            compaction.local_boundary, compaction.local_checkpoint = boundary, checkpoint
        state.source_view = view
        state.messages = deepcopy(prepared)
        state.tool_definitions = deepcopy(attempt.tool_definitions)
        state.usage = None
        state.local_usage_anchor = None
        state.conversation.replace_transcript(compaction.raw_messages)
        return view

    @staticmethod
    def _safe_historical_prefix_envelope(body: dict[str, object]) -> bool:
        """Allow prefix reuse only for the fully converted function-tool surface."""
        if "functions" in body or "function_call" in body:
            return False
        raw_tools = body.get("tools")
        if raw_tools is None:
            tools: list[object] = []
        elif isinstance(raw_tools, list):
            tools = cast(list[object], raw_tools)
        else:
            return False
        for raw_tool in tools:
            if not isinstance(raw_tool, dict):
                return False
            tool = cast(dict[str, object], raw_tool)
            if set(tool) - {"type", "function"} or tool.get("type") != "function":
                return False
            raw_function = tool.get("function")
            if not isinstance(raw_function, dict):
                return False
            function = cast(dict[str, object], raw_function)
            if (
                set(function) - {"name", "description", "parameters", "strict"}
                or not isinstance(function.get("name"), str)
                or ("description" in function and not isinstance(function["description"], str))
                or ("parameters" in function and not isinstance(function["parameters"], dict))
                or ("strict" in function and type(function["strict"]) is not bool)
            ):
                return False
        choice = body.get("tool_choice")
        if choice is not None:
            if isinstance(choice, str):
                if choice not in {"auto", "none", "required"}:
                    return False
            elif isinstance(choice, dict):
                choice_mapping = cast(dict[str, object], choice)
                function_choice = choice_mapping.get("function")
                if (
                    set(choice_mapping) != {"type", "function"}
                    or choice_mapping.get("type") != "function"
                    or not isinstance(function_choice, dict)
                    or set(cast(dict[str, object], function_choice)) != {"name"}
                    or not isinstance(cast(dict[str, object], function_choice).get("name"), str)
                ):
                    return False
            else:
                return False
        if "parallel_tool_calls" in body and type(body["parallel_tool_calls"]) is not bool:
            return False
        return True

    @staticmethod
    def _summary_generation_allowance(
        compaction: SourceCompactionState,
        budget: int,
    ) -> int:
        """Resolve a summary-route generation budget from declared capability."""
        runtime = compaction.runtime
        requested = compaction.summary_generation_tokens
        if requested is None:
            requested = runtime.generation.max_tokens
        if type(requested) is not int or requested <= 0:
            requested = GenerationSettings().max_tokens
        declared = runtime.resolved_capabilities.max_output_tokens
        if type(declared) is int and declared > 0:
            requested = min(requested, declared)
        ceiling = runtime.context_window_tokens
        if type(ceiling) is not int or ceiling <= 0:
            return max(1, requested)
        if budget > 0:
            ceiling = min(ceiling, budget)
        return max(1, min(requested, max(1, ceiling - SNIP_SAFETY_BUFFER)))

    @staticmethod
    def _summary_envelope_tokens(compaction: SourceCompactionState) -> int:
        """Measure the fixed summary instruction and JSON envelope conservatively."""
        from nanobot.agent.memory import Consolidator

        payload = Consolidator.source_summary_messages(
            [{'role': 'user', 'content': ''}],
            None,
        )
        estimate, _ = estimate_prompt_tokens_chain(
            compaction.runtime.provider,
            compaction.runtime.model,
            payload,
            None,
        )
        return max(1, estimate)

    @classmethod
    def _summary_reserve(
        cls,
        config: ContextGovernanceConfig,
        compaction: SourceCompactionState,
        budget: int,
    ) -> int:
        """Reserve room for a complete summary plus likely next-step growth."""
        if budget <= 0:
            return 0
        allowance = cls._summary_generation_allowance(compaction, budget)
        envelope = cls._summary_envelope_tokens(compaction)
        max_tool_chars = config.max_tool_result_chars
        if type(max_tool_chars) is not int or max_tool_chars < 0:
            max_tool_chars = 0
        growth = min(max_tool_chars, max(1, budget // 8))
        reserve = envelope + allowance + SNIP_SAFETY_BUFFER + growth
        return min(reserve, max(0, budget - 1))

    async def compact_source_request(self, state: ModelRequestState, view: SourceTranscript, *,
                                     tool_definitions: list[dict[str, Any]] | None,
                                     force: bool = False,
                                     stream: bool = False,
                                     tool_choice: str | dict[str, Any] | None = None,
                                     provider_context: ProviderCallContext | None = None,
                                     recovery_root_reason: str | None = None) -> SourceTranscript:
        """Summarize eligible closed source chunks at 90% pressure, targeting 60%."""
        from nanobot.agent.memory import Consolidator
        from nanobot.session.checkpoint_dependencies import (
            ConsumedSummaryChunk,
            SummaryDependencyProof,
        )

        compaction = state.source_compaction
        budget = self.input_budget(state.config)
        if compaction is None or not state.config.context_window_tokens or budget <= 0:
            return view
        deadline = asyncio.get_running_loop().time() + compaction.overall_timeout_seconds

        def check_deadline() -> None:
            if asyncio.get_running_loop().time() >= deadline:
                raise SourceCompactionUnavailableError(
                    'Source compaction operation deadline expired',
                    reason='summary_overall_timeout',
                )

        check_deadline()
        if compaction.artifact_projection_unavailable:
            return view
        main_preview = self._prepared_budget_preview(
            state.config,
            view.messages(),
            tool_definitions=tool_definitions,
            stream=stream,
            tool_choice=tool_choice,
            provider_context=provider_context,
        )
        if main_preview is not None:
            estimated = main_preview.measurement.input_tokens
            measurement_source = main_preview.measurement.source
            budget = main_preview.measurement.input_limit_tokens or 0
        else:
            estimated, measurement_source = estimate_prompt_tokens_chain(
                state.config.provider,
                state.config.model,
                view.messages(),
                tool_definitions,
            )
            budget = self.input_budget(state.config)
        if main_preview is None and not budget:
            budget = self.input_budget(state.config)
        check_deadline()
        generation_allowance = self._summary_generation_allowance(compaction, budget)
        summary_reserve = self._summary_reserve(state.config, compaction, budget)
        if (
            not force
            and estimated < budget * compaction.pressure_trigger_ratio
            and estimated + summary_reserve < budget
        ):
            return view
        attempt = self.capture_source_compaction(state, view, tool_definitions=tool_definitions)
        check_deadline()
        candidate = attempt.candidate
        if not candidate.boundaries:
            return view  # The caller still enforces the hard limit before transport.
        final_boundary = candidate.boundaries[-1]
        final_evidence = candidate.summary_evidence(final_boundary)
        if not final_evidence.nodes:
            return view
        dependencies_present = bool(
            final_evidence.retrievals
            or final_evidence.projections
            or candidate.previous_checkpoint is not None and candidate.previous_checkpoint.artifact_references
        )
        if dependencies_present and not self._has_active_retrieval_schema(state, attempt):
            # The active main request cannot resolve a dependency-bearing
            # checkpoint. Keep canonical source and let the final main budget
            # guard decide whether the request can proceed.
            return view
        target = int(budget * compaction.target_ratio)
        preview = candidate.rebuild(final_boundary, CheckpointNode('preview', 'x')).with_runtime(*attempt.request_notes)
        fixed_preview = self._prepared_budget_preview(
            state.config,
            preview.messages(),
            tool_definitions=tool_definitions,
            stream=stream,
            tool_choice=tool_choice,
            provider_context=provider_context,
        )
        if fixed_preview is not None:
            fixed_tokens = fixed_preview.measurement.input_tokens
            budget = fixed_preview.measurement.input_limit_tokens or 0
        else:
            fixed_tokens, _ = estimate_prompt_tokens_chain(
                state.config.provider,
                state.config.model,
                preview.messages(),
                tool_definitions,
            )
        check_deadline()
        if fixed_tokens >= budget:
            self.ensure_request_fits(
                state.config,
                preview.messages(),
                tool_definitions=tool_definitions,
                stream=stream,
                tool_choice=tool_choice,
                provider_context=provider_context,
            )
            raise SourceCompactionUnavailableError(
                'No room for a bounded source summary',
                reason='fixed_prefix_exceeds_auxiliary_budget',
            )
        headroom = max(1, (target if fixed_tokens < target else budget) - fixed_tokens)
        retained_tokens = max(
            1,
            min(generation_allowance, headroom),
        )
        allowance = retained_tokens
        prepared_payloads: dict[tuple[int, int, str | None], list[dict[str, Any]]] = {}

        def prepared_payload(evidence: SummaryEvidence,
                             previous_summary: str | None) -> list[dict[str, Any]]:
            key = (evidence.after, evidence.boundary, previous_summary)
            payload = prepared_payloads.get(key)
            if payload is None:
                payload = Consolidator.source_summary_messages(
                    [node.message for node in evidence.nodes], previous_summary,
                )
                prepared_payloads[key] = payload
            return payload

        from nanobot.providers.openai_compat_provider import OpenAICompatProvider

        prefix_mode = 'canonical'
        prefix_fallback_reason: str | None = None
        prefix_source_nodes = 0
        prefix_estimated_tokens: int | None = None
        converted_prefix_match: bool | None = None
        prefix_selection = None
        prefix_payload: list[dict[str, Any]] | None = None
        prefix_preview: PreparedRequestBudgetPreview | None = None
        prefix_provider = compaction.runtime.provider
        supports_prefix_summary = (
            prefix_provider is state.config.provider
            and getattr(prefix_provider, 'supports_source_summary', False) is True
        )
        if not supports_prefix_summary:
            prefix_fallback_reason = 'summary_provider_missing_capability'
        elif main_preview is None:
            prefix_fallback_reason = 'provider_preview_unavailable'
        elif any(
            tool.get('type') != 'function'
            or not isinstance(tool.get('function'), dict)
            for tool in tool_definitions or []
        ):
            prefix_fallback_reason = 'unsupported_tool_envelope'
        else:
            prefix_selection = candidate.select_historical_prefix(final_boundary)
            prefix_fallback_reason = prefix_selection.fallback_reason
            if prefix_selection.transcript is not None:
                if prefix_selection.boundary is None:
                    raise SourceValidationError('Historical prefix selection has no boundary')
                from nanobot.agent.memory import Consolidator

                prefix_payload = Consolidator.source_prefix_summary_messages(
                    prefix_selection.transcript.messages(),
                )
                try:
                    prefix_preview = self._compatibility_auxiliary_preview(
                        compaction.runtime,
                        prefix_payload,
                        max_output_tokens=allowance,
                        tools=tool_definitions,
                        tool_choice=tool_choice,
                        mode='prefix',
                        idle_timeout_seconds=compaction.idle_timeout_seconds,
                        deadline_monotonic=deadline,
                    )
                except RequestPayloadBytesExceeded:
                    prefix_preview = None
                    prefix_fallback_reason = 'prefix_payload_bytes'
                except ProviderRequestRejected as exc:
                    if exc.reason != 'unsupported_auxiliary_envelope':
                        raise
                    prefix_preview = None
                    prefix_fallback_reason = exc.reason
                if prefix_preview is None and prefix_fallback_reason is None:
                    prefix_fallback_reason = 'provider_preview_unavailable'
                elif prefix_preview is not None and not all(
                    self._safe_historical_prefix_envelope(preview.effective_body)
                    for preview in (main_preview, prefix_preview)
                ):
                    prefix_preview = None
                    prefix_fallback_reason = 'unsupported_tool_envelope'
                elif prefix_preview is not None and not prefix_preview.measurement.fits:
                    prefix_fallback_reason = 'prefix_exceeds_auxiliary_budget'
                elif prefix_preview is not None:
                    instruction = prefix_payload[-1]
                    converted = prefix_preview.converted_messages
                    has_separate_instruction = bool(
                        len(converted) > 0
                        and OpenAICompatProvider.json_values_equal(
                            converted[-1], instruction,
                        )
                    )
                    prefix_rows = converted[:-1] if has_separate_instruction else ()
                    aligned_body_fields = {
                        'messages', 'stream', 'stream_options', 'max_tokens',
                        'max_completion_tokens', 'max_output_tokens',
                    }
                    main_envelope = {
                        key: value for key, value in main_preview.effective_body.items()
                        if key not in aligned_body_fields
                    }
                    prefix_envelope = {
                        key: value for key, value in prefix_preview.effective_body.items()
                        if key not in aligned_body_fields
                    }
                    converted_prefix_match = bool(
                        has_separate_instruction
                        and len(main_preview.converted_messages) >= len(prefix_rows)
                        and main_preview.converted_messages[:len(prefix_rows)] == prefix_rows
                        and main_preview.converted_tools == prefix_preview.converted_tools
                        and main_envelope == prefix_envelope
                    )
                    if converted_prefix_match:
                        prefix_mode = 'historical_prefix'
                        prefix_fallback_reason = None
                        prefix_source_nodes = prefix_selection.source_node_count
                        prefix_estimated_tokens = prefix_preview.measurement.input_tokens
                    else:
                        prefix_fallback_reason = 'converted_prefix_mismatch'

        metrics = SourceCompactionMetrics(
            'root' if candidate.stream_id is None else 'local',
            estimated,
            root_reason=recovery_root_reason,
            prefix_mode=prefix_mode,
            prefix_fallback_reason=prefix_fallback_reason,
            prefix_source_nodes=prefix_source_nodes,
            prefix_estimated_tokens=prefix_estimated_tokens,
            converted_prefix_match=converted_prefix_match,
            measurement_source=measurement_source,
            selected_capacity=(
                main_preview.measurement.total_limit_tokens
                if main_preview is not None
                else state.config.context_window_tokens
            ),
            completion_reserve=(
                main_preview.measurement.completion_reserve_tokens
                if main_preview is not None
                else ContextGovernor._captured_generation(state.config).max_tokens
            ),
            safety_buffer=(
                main_preview.measurement.safety_buffer_tokens
                if main_preview is not None
                else SNIP_SAFETY_BUFFER
            ),
            payload_bytes=(
                main_preview.measurement.measured_bytes if main_preview is not None else None
            ),
            summary_generation_allowance=generation_allowance,
            summary_retained_tokens=retained_tokens,
            summary_reserve_tokens=summary_reserve,
        )
        compaction.attempts.append(metrics)
        started = monotonic()
        compaction_id = uuid4().hex
        cursor = candidate.previous_boundary
        summary = candidate.previous_checkpoint.summary if candidate.previous_checkpoint is not None else None
        consumed: list[ConsumedSummaryChunk] = []

        async def emit_phase(
            phase: Literal['started', 'succeeded', 'failed', 'cancelled'],
        ) -> None:
            check_deadline()
            async with asyncio.timeout_at(deadline):
                await state.events.emit(
                    ContextCompactionEvent(compaction_id=compaction_id, phase=phase),
                )
            check_deadline()

        try:
            await emit_phase('started')
            # Reject impossible indivisible groups across the intended prefix before
            # spending any auxiliary calls. Local previews use the same finalized body
            # and token units as the eventual one-attempt dispatch.
            group_start = candidate.previous_boundary
            prefix_preflight_boundary = candidate.previous_boundary
            if prefix_mode == 'historical_prefix':
                if prefix_selection is None or prefix_selection.boundary is None:
                    raise SourceValidationError(
                        'Verified historical prefix selection lost its boundary',
                    )
                prefix_preflight_boundary = prefix_selection.boundary
            for boundary in candidate.boundaries:
                check_deadline()
                if boundary <= prefix_preflight_boundary:
                    # The verified native prefix was already checked as one exact
                    # finalized SDK body. Do not price its canonical JSON surrogate,
                    # which is not the body that will be dispatched.
                    group_start = boundary
                    check_deadline()
                    continue
                evidence = candidate.summary_evidence(boundary, after=group_start)
                if not evidence.nodes:
                    continue
                payload = prepared_payload(evidence, None)
                try:
                    group_preview = self._compatibility_auxiliary_preview(
                        compaction.runtime,
                        payload,
                        max_output_tokens=allowance,
                        tools=None,
                        tool_choice=None,
                        mode='canonical',
                        idle_timeout_seconds=compaction.idle_timeout_seconds,
                        deadline_monotonic=deadline,
                    )
                except RequestPayloadBytesExceeded as exc:
                    raise SourceCompactionUnavailableError(
                        'Indivisible source group exceeds the auxiliary payload limit',
                        reason=exc.reason,
                    ) from None
                except ProviderRequestRejected as exc:
                    raise SourceCompactionUnavailableError(
                        'Auxiliary request envelope was rejected',
                        reason=exc.reason,
                    ) from None
                if group_preview is not None:
                    if not group_preview.measurement.fits:
                        raise SourceCompactionUnavailableError(
                            'Indivisible source group exceeds auxiliary budget',
                            reason='indivisible_source_group_exceeds_auxiliary_budget',
                        )
                else:
                    tokens, _ = estimate_prompt_tokens_chain(
                        compaction.runtime.provider,
                        compaction.runtime.model,
                        payload,
                        None,
                    )
                    if (tokens + allowance + SNIP_SAFETY_BUFFER
                            > compaction.runtime.context_window_tokens):
                        raise SourceCompactionUnavailableError(
                            'Indivisible source group exceeds auxiliary budget',
                            reason='indivisible_source_group_exceeds_auxiliary_budget',
                        )
                group_start = boundary
                check_deadline()

            truncated_boundaries: set[int] = set()
            for fold_index in range(4):
                chosen = None
                evidence = None
                prepared = None
                call_mode: Literal['prefix', 'canonical'] = 'canonical'
                call_tools: list[dict[str, Any]] | None = None
                call_tool_choice: str | dict[str, Any] | None = None
                if (
                    fold_index == 0
                    and prefix_mode == 'historical_prefix'
                    and prefix_selection is not None
                    and prefix_payload is not None
                    and prefix_preview is not None
                ):
                    prefix_boundary = prefix_selection.boundary
                    if prefix_boundary is None:
                        raise SourceValidationError(
                            'Verified historical prefix selection lost its boundary',
                        )
                    chosen = prefix_boundary
                    evidence = candidate.summary_evidence(prefix_boundary, after=cursor)
                    prepared = prefix_payload
                    call_mode = 'prefix'
                    call_tools = tool_definitions
                    call_tool_choice = tool_choice
                # Search exact closed chunks, without truncating an indivisible
                # tool group or folding covered pin text into evidence again.
                chunk_rejection_reason: str | None = None
                if prepared is None:
                    for boundary in reversed(candidate.boundaries):
                        if boundary <= cursor or boundary in truncated_boundaries:
                            continue
                        selected = candidate.summary_evidence(boundary, after=cursor)
                        if not selected.nodes:
                            continue
                        payload = prepared_payload(selected, summary)
                        try:
                            chunk_preview = self._compatibility_auxiliary_preview(
                                compaction.runtime,
                                payload,
                                max_output_tokens=allowance,
                                tools=None,
                                tool_choice=None,
                                mode='canonical',
                                idle_timeout_seconds=compaction.idle_timeout_seconds,
                                deadline_monotonic=deadline,
                            )
                        except RequestPayloadBytesExceeded as exc:
                            chunk_rejection_reason = exc.reason
                            continue
                        except ProviderRequestRejected as exc:
                            raise SourceCompactionUnavailableError(
                                'Auxiliary request envelope was rejected',
                                reason=exc.reason,
                            ) from None
                        if chunk_preview is not None:
                            if chunk_preview.measurement.fits:
                                chosen, evidence, prepared = boundary, selected, payload
                                break
                        else:
                            tokens, _ = estimate_prompt_tokens_chain(
                                compaction.runtime.provider,
                                compaction.runtime.model,
                                payload,
                                None,
                            )
                            if (tokens + allowance + SNIP_SAFETY_BUFFER
                                    <= compaction.runtime.context_window_tokens):
                                chosen, evidence, prepared = boundary, selected, payload
                                break
                if chosen is None or evidence is None or prepared is None:
                    raise SourceCompactionUnavailableError(
                        'Eligible source chunk exceeds auxiliary budget',
                        reason=(chunk_rejection_reason
                                or 'eligible_source_chunk_exceeds_auxiliary_budget'),
                    )
                preflight = SummaryDependencyProof(candidate, tuple([
                    *consumed,
                    ConsumedSummaryChunk(evidence, summary, 'Preflight dependency validation.'),
                ]))
                self._preflight_summary_dependencies(state, attempt, preflight)
                check_deadline()
                calls_before = metrics.calls

                def record_auxiliary_attempt(_kind: str) -> None:
                    metrics.calls += 1

                try:
                    async with asyncio.timeout_at(deadline):
                        result = await Consolidator.summarize_prepared_source_chunk(
                            prepared,
                            runtime=compaction.runtime,
                            max_output_tokens=allowance,
                            retained_text_tokens=retained_tokens,
                            request_deadline_monotonic=deadline,
                            stream_idle_timeout_seconds=compaction.idle_timeout_seconds,
                            source_summary_mode=call_mode,
                            tools=call_tools,
                            tool_choice=call_tool_choice,
                            on_transport_attempt=record_auxiliary_attempt,
                        )
                except BaseException:
                    if metrics.calls > calls_before:
                        metrics.usage_unknown = True
                    raise
                observed_calls = metrics.calls - calls_before
                if result.calls > observed_calls:
                    metrics.calls += result.calls - observed_calls
                if (observed_calls or result.calls) and result.usage is None:
                    metrics.usage_unknown = True
                if result.usage is not None:
                    metrics.usage = result.usage if metrics.usage is None else metrics.usage + result.usage
                if result.summary is None or result.status != 'complete':
                    failure_reason = result.failure_reason or 'summary_provider_error'
                    if (
                        failure_reason == 'summary_truncated'
                        and fold_index < 3
                    ):
                        # Discard the truncated candidate and retry the same
                        # evidence cursor with the next smaller closed chunk.
                        truncated_boundaries.add(chosen)
                        check_deadline()
                        continue
                    raise SourceCompactionUnavailableError(
                        'Bounded source summary failed: ' + result.status,
                        reason=failure_reason,
                    )
                consumed.append(ConsumedSummaryChunk(evidence, summary, result.summary))
                summary, cursor = result.summary, chosen
                metrics.boundary = cursor
                rebuilt = candidate.rebuild(cursor, CheckpointNode('preview', summary)).with_runtime(*attempt.request_notes)
                rebuilt_preview = self._prepared_budget_preview(
                    state.config,
                    rebuilt.messages(),
                    tool_definitions=tool_definitions,
                    stream=stream,
                    tool_choice=tool_choice,
                    provider_context=provider_context,
                )
                if rebuilt_preview is not None:
                    after_tokens = rebuilt_preview.measurement.input_tokens
                    metrics.after_tokens = after_tokens
                else:
                    after_tokens, _ = estimate_prompt_tokens_chain(
                        state.config.provider,
                        state.config.model,
                        rebuilt.messages(),
                        tool_definitions,
                    )
                    metrics.after_tokens = after_tokens
                if (
                    not force
                    and after_tokens <= target
                ) or cursor == final_boundary:
                    break
            if summary is None or cursor == candidate.previous_boundary:
                raise SourceCompactionUnavailableError(
                    'No source compaction progress',
                    reason='no_progress',
                )
            usage = None if metrics.usage_unknown or metrics.usage is None else {
                key: value for key, value in metrics.usage.to_dict().items() if type(value) is int
            }
            check_deadline()
            published = self.publish_source_compaction(state, attempt, boundary=cursor,
                                                      summary=summary, auxiliary_usage=usage,
                                                      retained_text_tokens=retained_tokens,
                                                      proof=SummaryDependencyProof(candidate, tuple(consumed)),
                                                      stream=stream,
                                                      tool_choice=tool_choice,
                                                      provider_context=provider_context,
                                                      deadline_monotonic=deadline)
            metrics.status = 'succeeded'
        except TimeoutError as exc:
            metrics.status = 'failed'
            metrics.failure_reason = 'summary_overall_timeout'
            if asyncio.get_running_loop().time() < deadline:
                try:
                    await emit_phase('failed')
                except TimeoutError:
                    pass
            raise SourceCompactionUnavailableError(
                'Source compaction operation deadline expired',
                reason='summary_overall_timeout',
            ) from exc
        except BaseException as exc:
            metrics.status = 'cancelled' if isinstance(exc, asyncio.CancelledError) else 'failed'
            if isinstance(exc, SourceCompactionUnavailableError):
                metrics.failure_reason = exc.reason
            elif isinstance(exc, ProviderRequestRejected):
                metrics.failure_reason = exc.reason
            elif isinstance(exc, SourceValidationError):
                metrics.failure_reason = 'source_validation'
            else:
                metrics.failure_reason = (
                    'cancelled' if isinstance(exc, asyncio.CancelledError)
                    else 'summary_provider_error'
                )
            if asyncio.get_running_loop().time() < deadline:
                try:
                    await emit_phase(
                        'cancelled' if isinstance(exc, asyncio.CancelledError) else 'failed',
                    )
                except TimeoutError:
                    pass
            raise
        finally:
            metrics.latency_ms = max(0, int((monotonic() - started) * 1000))
            usage = metrics.usage
            logger.debug(
                'Source compaction provider={} model={} scope={} status={} boundary={} '
                'before_tokens={} after_tokens={} calls={} known_tokens={} usage_input_tokens={} '
                'usage_output_tokens={} cache_read_tokens={} cache_write_tokens={} usage_unknown={} '
                'measurement_source={} selected_capacity={} completion_reserve={} safety_buffer={} '
                'payload_bytes={} latency_ms={} root_reason={} failure_reason={} prefix_mode={} '
                'prefix_fallback_reason={} prefix_source_nodes={} prefix_estimated_tokens={} '
                'converted_prefix_match={} summary_generation_allowance={} '
                'summary_retained_tokens={} summary_reserve_tokens={}',
                state.config.provider.provider_name,
                state.config.model,
                metrics.scope,
                metrics.status,
                metrics.boundary,
                metrics.before_tokens,
                metrics.after_tokens,
                metrics.calls,
                usage.total_tokens if usage is not None else None,
                usage.input_tokens if usage is not None else None,
                usage.output_tokens if usage is not None else None,
                usage.cache_read_tokens if usage is not None else None,
                usage.cache_write_tokens if usage is not None else None,
                metrics.usage_unknown,
                metrics.measurement_source,
                metrics.selected_capacity,
                metrics.completion_reserve,
                metrics.safety_buffer,
                metrics.payload_bytes,
                metrics.latency_ms,
                metrics.root_reason,
                metrics.failure_reason,
                metrics.prefix_mode,
                metrics.prefix_fallback_reason,
                metrics.prefix_source_nodes,
                metrics.prefix_estimated_tokens,
                metrics.converted_prefix_match,
                metrics.summary_generation_allowance,
                metrics.summary_retained_tokens,
                metrics.summary_reserve_tokens,
            )
        if asyncio.get_running_loop().time() < deadline:
            await emit_phase('succeeded')
        return published

    @staticmethod
    def _is_finalized_overflow(measurement: TransportMeasurement | None) -> bool:
        """Accept only a coherent measured body rejected above its input limit."""
        if type(measurement) is not TransportMeasurement:
            return False
        identity = measurement.identity
        if type(identity) is not TransportIdentity or any(
            type(value) is not str or not value.strip()
            for value in (identity.provider, identity.transport, identity.model)
        ):
            return False
        input_tokens = measurement.input_tokens
        reserve = measurement.completion_reserve_tokens
        buffer = measurement.safety_buffer_tokens
        total = measurement.total_limit_tokens
        input_limit = measurement.input_limit_tokens
        if (
            type(input_tokens) is not int
            or type(reserve) is not int
            or type(buffer) is not int
            or type(total) is not int
            or type(input_limit) is not int
        ):
            return False
        return (
            identity.transport in {
                'openai_chat_completions',
                'openai_chat_completions_stream',
            }
            and measurement.policy == RequestAccountingPolicy.ESTIMATED_COMPAT.value
            and input_tokens >= 0
            and reserve >= 0
            and buffer >= 0
            and total > 0
            and input_limit > 0
            and input_limit == total - reserve - buffer
            and input_tokens > input_limit
            and type(measurement.source) is str
            and bool(measurement.source.strip())
        )

    async def recover_finalized_request(
        self,
        state: ModelRequestState,
        *,
        measurement: TransportMeasurement | None,
        backend_overflow: BackendContextOverflow | None = None,
        tool_definitions: list[dict[str, Any]] | None,
        transcript: list[dict[str, Any]] | None,
        source_view: SourceTranscript | None,
        stream: bool = False,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> SourceRecoveryOutcome:
        """Spend one forced source-compaction pass for one attributed rejection."""
        source, compaction = state.source, state.source_compaction
        root_reason: Literal['local_estimated_admission', 'backend_context_overflow'] = (
            'backend_context_overflow' if backend_overflow is not None
            else 'local_estimated_admission'
        )

        def unavailable(reason: Literal['no_eligible_evidence', 'no_progress']) -> SourceRecoveryOutcome:
            return SourceRecoveryOutcome(None, None, root_reason, reason)

        backend_valid = (
            type(backend_overflow) is BackendContextOverflow
            and type(backend_overflow.status_code) is int
            and backend_overflow.status_code == 400
            and type(backend_overflow.code) is int
            and backend_overflow.code == 400
            and backend_overflow.error_type == 'exceed_context_size_error'
            and type(backend_overflow.n_prompt_tokens) is int
            and backend_overflow.n_prompt_tokens > 0
            and type(backend_overflow.n_ctx) is int
            and backend_overflow.n_ctx > 0
            and backend_overflow.n_prompt_tokens >= backend_overflow.n_ctx
        )
        if (
            not state.config.preserve_source
            or source is None
            or compaction is None
            or not (
                backend_valid
                or backend_overflow is None and self._is_finalized_overflow(measurement)
            )
            or self.input_budget(state.config) <= 0
        ):
            return unavailable('no_eligible_evidence')
        if source.has_pending_requests:
            raise SourceValidationError('Finalized budget recovery requires a quiescent request ledger')
        view = source_view
        if view is None or state.source_view != view or state.messages is None:
            raise SourceValidationError('Finalized budget recovery requires the rejected source request view')
        source.validate_view(view)
        rejected_messages = deepcopy(state.messages)
        if view.messages() != rejected_messages:
            raise SourceValidationError('Rejected source view differs from prepared request messages')
        if compaction.raw_messages != source.transcript.messages():
            raise SourceValidationError('Runner transcript differs from active source projection')
        compaction.writer.owner.validate_dispatch()

        stream_id = compaction.stream_id
        previous_boundary = (
            compaction.admission.boundary if stream_id is None else compaction.local_boundary
        )
        previous_epoch = source.projection_epoch
        try:
            prepared, provider_context = await self.prepare_request(
                state,
                rejected_messages,
                tool_definitions=tool_definitions,
                transcript=transcript,
                source_view=view,
                force_source_compaction=True,
                stream=stream,
                tool_choice=tool_choice,
                recovery_root_reason=root_reason,
            )
        except SourceCompactionUnavailableError as exc:
            return SourceRecoveryOutcome(None, None, root_reason, exc.reason)
        except ContextWindowExceededError:
            return unavailable('no_progress')

        current_view = state.source_view
        if current_view is None:
            raise SourceValidationError('Source recovery preparation lost its source view')
        source.validate_view(current_view)
        if compaction.stream_id != stream_id:
            raise SourceValidationError('Finalized budget recovery changed source scope')
        current_boundary = (
            compaction.admission.boundary if stream_id is None else compaction.local_boundary
        )
        if (
            current_boundary <= previous_boundary
            or source.projection_epoch <= previous_epoch
            or prepared == rejected_messages
            or state.messages != prepared
            or current_view.messages() != prepared
        ):
            return unavailable('no_progress')
        return SourceRecoveryOutcome(prepared, provider_context, root_reason)

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [
                    cast(dict[str, Any], item)
                    if isinstance(item, dict)
                    else {"type": "text", "text": str(item)}
                    for item in cast(list[Any], value)
                ]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    @classmethod
    def _merge_adjacent_user_messages_for_model(
        cls,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge adjacent visible user messages only in the model-facing copy."""
        prepared: list[dict[str, Any]] = []
        for source in messages:
            injection = deepcopy(source)
            if (
                prepared
                and injection.get("role") == "user"
                and prepared[-1].get("role") == "user"
                and injection.get("content") != SUMMARY_CONTINUATION_TEXT
                and prepared[-1].get("content") != SUMMARY_CONTINUATION_TEXT
                and not is_hidden_history_message(injection)
                and not is_hidden_history_message(prepared[-1])
                and allows_conversation_message_merge(injection)
                and allows_conversation_message_merge(prepared[-1])
            ):
                merged = dict(prepared[-1])
                left_meta = merged.get("_meta")
                right_meta = injection.get("_meta")
                left_meta_dict = (
                    cast(dict[str, Any], left_meta) if isinstance(left_meta, dict) else None
                )
                right_meta_dict = (
                    cast(dict[str, Any], right_meta) if isinstance(right_meta, dict) else None
                )
                left_marker = (
                    left_meta_dict.get(RUNTIME_CONTEXT_MESSAGE_META)
                    if left_meta_dict is not None
                    else None
                )
                right_marker = (
                    right_meta_dict.get(RUNTIME_CONTEXT_MESSAGE_META)
                    if right_meta_dict is not None
                    else None
                )
                left_marker_dict = (
                    cast(dict[str, Any], left_marker) if isinstance(left_marker, dict) else None
                )
                right_marker_dict = (
                    cast(dict[str, Any], right_marker) if isinstance(right_marker, dict) else None
                )
                empty_sources: list[str] = []
                empty_blocks: list[dict[str, Any]] = []
                detached_left = (
                    detach_runtime_context(merged.get("content"), left_marker_dict)
                    if left_marker_dict is not None
                    else (merged.get("content"), empty_sources, empty_blocks)
                )
                detached_right = (
                    detach_runtime_context(injection.get("content"), right_marker_dict)
                    if right_marker_dict is not None
                    else (injection.get("content"), empty_sources, empty_blocks)
                )
                if detached_left is not None and detached_right is not None:
                    left_content, left_sources, left_blocks = detached_left
                    right_content, right_sources, right_blocks = detached_right
                    merged_content = cls._merge_message_content(left_content, right_content)
                    context_blocks = [*left_blocks, *right_blocks]
                    if context_blocks:
                        merged_content, marker = reattach_runtime_context(
                            merged_content,
                            [*left_sources, *right_sources],
                            context_blocks,
                        )
                        internal_meta = (
                            dict(left_meta_dict) if left_meta_dict is not None else {}
                        )
                        if right_meta_dict is not None:
                            for key, value in right_meta_dict.items():
                                internal_meta.setdefault(key, value)
                        internal_meta[RUNTIME_CONTEXT_MESSAGE_META] = marker
                        merged["_meta"] = internal_meta
                    merged["content"] = merged_content
                else:
                    merged["content"] = cls._merge_message_content(
                        merged.get("content"),
                        injection.get("content"),
                    )
                prepared[-1] = merged
                continue
            prepared.append(injection)
        return prepared

    def prepare_messages_for_model(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build the normalized model-facing copy of a raw transcript."""
        if config.preserve_source:
            return deepcopy(messages)
        # Select the image-bearing message before adjacent users are merged.
        governed = self.prepare_for_model(config, recent_image_messages(messages))
        return self._merge_adjacent_user_messages_for_model(governed)

    def prepare_for_model(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        updated = self.strip_placeholder_assistant_messages(messages)
        updated = self.strip_malformed_tool_calls(updated)
        updated = self.drop_orphan_tool_results(updated)
        updated = self.backfill_missing_tool_results(updated)
        return self.apply_tool_result_budget(config, updated)

    def fit_to_budget(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        *,
        tool_definitions: list[dict[str, Any]] | None,
        stream: bool = False,
        tool_choice: str | dict[str, Any] | None = None,
        provider_context: ProviderCallContext | None = None,
    ) -> list[dict[str, Any]]:
        """Fit a model-facing copy while keeping the source transcript intact."""
        updated = self.snip_history(
            config,
            messages,
            tool_definitions=tool_definitions,
            force=True,
        )
        updated = self.drop_orphan_tool_results(updated)
        updated = self.backfill_missing_tool_results(updated)
        return self.ensure_request_fits(
            config,
            updated,
            tool_definitions=tool_definitions,
            stream=stream,
            tool_choice=tool_choice,
            provider_context=provider_context,
        )

    @staticmethod
    def _captured_generation(config: ContextGovernanceConfig) -> GenerationSettings:
        for candidate in (config.generation, getattr(config.provider, 'generation', None)):
            if type(candidate) is not GenerationSettings:
                continue
            temperature = candidate.temperature
            if (
                type(temperature) not in (int, float)
                or not math.isfinite(temperature)
                or type(candidate.max_tokens) is not int
                or candidate.max_tokens < 0
                or (
                    candidate.reasoning_effort is not None
                    and type(candidate.reasoning_effort) is not str
                )
            ):
                continue
            return candidate
        return GenerationSettings()

    @staticmethod
    def _concrete_budget_preview(
        candidate: object,
    ) -> PreparedRequestBudgetPreview | None:
        """Accept only a fully shaped provider-owned request preview."""
        if type(candidate) is not PreparedRequestBudgetPreview:
            return None
        preview = candidate
        if type(preview.measurement) is not TransportMeasurement:
            return None
        measurement = preview.measurement
        identity = measurement.identity
        if type(identity) is not TransportIdentity or any(
            type(value) is not str or not value.strip()
            for value in (identity.provider, identity.transport, identity.model)
        ):
            return None
        for value in (
            measurement.input_tokens,
            measurement.completion_reserve_tokens,
            measurement.safety_buffer_tokens,
        ):
            if type(value) is not int or value < 0:
                return None
        for value in (
            measurement.total_limit_tokens,
            measurement.input_limit_tokens,
            measurement.measured_bytes,
            measurement.limit_bytes,
        ):
            if value is not None and (type(value) is not int or value < 0):
                return None
        if (
            measurement.input_limit_tokens is not None
            and measurement.total_limit_tokens is not None
            and measurement.input_limit_tokens > measurement.total_limit_tokens
        ):
            return None
        if (
            (measurement.measured_bytes is None) != (measurement.limit_bytes is None)
            or type(measurement.source) is not str
            or not measurement.source
            or type(measurement.policy) is not str
            or measurement.policy
            not in {
                RequestAccountingPolicy.STRICT.value,
                RequestAccountingPolicy.ESTIMATED_COMPAT.value,
            }
            or type(preview.envelope_binding) is not str
            or not preview.envelope_binding
        ):
            return None

        def string_keyed_dict(value: object) -> bool:
            return type(value) is dict and all(
                type(key) is str for key in cast(dict[object, object], value)
            )

        def converted_rows(value: object) -> bool:
            return type(value) is tuple and all(
                string_keyed_dict(row) for row in cast(tuple[object, ...], value)
            )

        if (
            not string_keyed_dict(preview.effective_body)
            or not string_keyed_dict(preview.sdk_kwargs)
            or not converted_rows(preview.converted_messages)
            or (
                preview.converted_tools is not None
                and not converted_rows(preview.converted_tools)
            )
        ):
            return None
        route = preview.route
        if route is not None:
            if type(route) is not PreparedRequestRoute:
                return None
            if (
                route.provider != identity.provider
                or route.model != identity.model
                or route.transport != identity.transport
                or route.accounting_policy != measurement.policy
                or route.context_window_tokens != measurement.total_limit_tokens
                or route.completion_reserve_tokens != measurement.completion_reserve_tokens
                or route.envelope_binding != preview.envelope_binding
            ):
                return None
        return preview

    def _attach_route_selection(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        *,
        tool_definitions: list[dict[str, Any]] | None,
        stream: bool,
        tool_choice: str | dict[str, Any] | None,
        provider_context: ProviderCallContext,
    ) -> ProviderCallContext:
        """Bind the exact prepared request to the leaf dispatch will use."""
        preview = self._prepared_budget_preview(
            config,
            messages,
            tool_definitions=tool_definitions,
            stream=stream,
            tool_choice=tool_choice,
            provider_context=provider_context,
        )
        if preview is None or preview.route is None:
            return provider_context
        return replace(provider_context, selected_route=preview.route)

    @classmethod
    def _prepared_budget_preview(
        cls,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        *,
        tool_definitions: list[dict[str, Any]] | None,
        stream: bool,
        tool_choice: str | dict[str, Any] | None,
        provider_context: ProviderCallContext | None = None,
        generation: GenerationSettings | None = None,
    ) -> PreparedRequestBudgetPreview | None:
        captured_generation = generation or cls._captured_generation(config)
        context = provider_context or ProviderCallContext(
            request_limits=derive_provider_request_limits(
                config.context_window_tokens,
                captured_generation.max_tokens,
            ),
            context_window_tokens=config.context_window_tokens,
        )
        preview_budget = getattr(config.provider, "preview_request_budget", None)
        if not callable(preview_budget):
            return None
        candidate = preview_budget(
            messages=messages,
            tools=tool_definitions,
            model=config.model,
            generation=captured_generation,
            provider_context=context,
            stream=stream,
            tool_choice=tool_choice,
        )
        return cls._concrete_budget_preview(candidate)

    @staticmethod
    def _compatibility_auxiliary_preview(
        runtime: LLMRuntime,
        messages: list[dict[str, Any]],
        *,
        max_output_tokens: int,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        mode: Literal['prefix', 'canonical'],
        idle_timeout_seconds: float,
        deadline_monotonic: float,
    ) -> PreparedRequestBudgetPreview | None:
        provider = runtime.provider
        if getattr(provider, 'supports_source_summary', False) is not True:
            return None
        context = ProviderCallContext(
            request_limits=derive_provider_request_limits(
                runtime.context_window_tokens,
                max_output_tokens,
            ),
            context_window_tokens=runtime.context_window_tokens,
            stream_idle_timeout_seconds=idle_timeout_seconds,
            request_deadline_monotonic=deadline_monotonic,
            single_attempt=True,
            source_summary_mode=mode,
        )
        generation = GenerationSettings(
            temperature=runtime.generation.temperature,
            max_tokens=max_output_tokens,
            reasoning_effort=(
                runtime.generation.reasoning_effort if mode == 'prefix' else 'none'
            ),
        )
        return provider.preview_request_budget(
            messages=messages,
            tools=tools,
            model=runtime.model,
            generation=generation,
            provider_context=context,
            stream=True,
            tool_choice=tool_choice,
        )

    def ensure_request_fits(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        *,
        tool_definitions: list[dict[str, Any]] | None,
        stream: bool = False,
        tool_choice: str | dict[str, Any] | None = None,
        provider_context: ProviderCallContext | None = None,
    ) -> list[dict[str, Any]]:
        """Validate an exact model request without dropping any messages."""
        preview = self._prepared_budget_preview(
            config,
            messages,
            tool_definitions=tool_definitions,
            stream=stream,
            tool_choice=tool_choice,
            provider_context=provider_context,
        )
        if preview is not None:
            measured = preview.measurement
            budget = measured.input_limit_tokens or 0
            estimated, source = measured.input_tokens, measured.source
            if measured.fits:
                return messages
        else:
            if not config.context_window_tokens:
                return messages
            budget = self.input_budget(config)
            estimated, source = estimate_prompt_tokens_chain(
                config.provider,
                config.model,
                messages,
                tool_definitions,
            )
        if budget > 0 and estimated <= budget:
            return messages
        raise ContextWindowExceededError(
            session_key=config.session_key,
            estimated_tokens=estimated,
            input_budget=budget,
            source=source,
        )

    def request_pressure(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        usage: LLMUsage | None,
        *,
        usage_matches_messages: bool,
        tool_definitions: list[dict[str, Any]] | None,
        request_context_tokens: int | None = None,
        stream: bool = False,
        tool_choice: str | dict[str, Any] | None = None,
        provider_context: ProviderCallContext | None = None,
    ) -> tuple[int, str] | None:
        """Return the authoritative measurement when a request is pressured."""
        if not config.context_window_tokens:
            return None
        preview = self._prepared_budget_preview(
            config,
            messages,
            tool_definitions=tool_definitions,
            stream=stream,
            tool_choice=tool_choice,
            provider_context=provider_context,
        )
        if preview is not None:
            measured = preview.measurement.input_tokens
            source = preview.measurement.source
            budget = preview.measurement.input_limit_tokens or 0
        else:
            budget = self.input_budget(config)
            if request_context_tokens is not None:
                measured = request_context_tokens
                source = "resumed provider state plus pending messages"
            elif (
                usage_matches_messages
                and usage is not None
                and usage.context_tokens is not None
            ):
                measured = usage.context_tokens
                source = "matching provider usage"
            else:
                measured, source = estimate_prompt_tokens_chain(
                    config.provider,
                    config.model,
                    messages,
                    tool_definitions,
                )
        if budget > 0 and measured < budget:
            return None
        return measured, source

    @staticmethod
    def _summary_transcript(
        compaction: ContextCompactionState,
        summary: str,
    ) -> list[dict[str, Any]]:
        """Rebuild only the stable system prefix around a replacement summary."""
        return compaction.transcript_builder(
            replace(
                compaction.transcript_input,
                history=[],
                current_message=None,
                media=None,
                session_summary={
                    "text": summary,
                    "last_active": datetime.now().astimezone().isoformat(),
                },
                runtime_context_blocks=None,
            )
        )

    async def summarize_provider_compaction(
        self,
        state: ModelRequestState,
        response: LLMResponse,
        *,
        current_request_boundary: int | None,
    ) -> None:
        """Materialize the exact input replaced by provider-native compaction."""
        compaction = state.compaction
        if (
            not response.provider_compaction_applied
            or response.provider_compaction_state is None
            or compaction is None
            or compaction.consolidate_provider_compaction is None
        ):
            return

        if response.provider_compaction_scope == "prior_context":
            accepted_messages = compaction.accepted_messages
            transcript_boundary = compaction.raw_accepted_boundary
        elif (
            response.provider_compaction_scope == "current_request"
            and state.messages is not None
            and current_request_boundary is not None
        ):
            accepted_messages = state.messages
            transcript_boundary = current_request_boundary
        else:
            logger.warning(
                "Ignoring provider compaction with missing request-boundary scope for {}",
                state.config.session_key or "default",
            )
            return

        compaction_id = uuid4().hex
        await state.events.emit(
            ContextCompactionEvent(compaction_id=compaction_id, phase="started"),
        )
        try:
            summary = await compaction.consolidate_provider_compaction(
                response.provider_compaction_state,
                deepcopy(accepted_messages),
                compaction.active_summary,
            )
        except (Exception, asyncio.CancelledError) as exc:
            await state.events.emit(
                ContextCompactionEvent(
                    compaction_id=compaction_id,
                    phase="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                ),
            )
            raise
        if not summary:
            await state.events.emit(
                ContextCompactionEvent(compaction_id=compaction_id, phase="failed"),
            )
            return
        compaction.active_summary = summary
        compaction.summary_checkpoint = SessionSummaryCheckpoint(
            summary=summary,
            transcript_boundary=transcript_boundary,
        )
        state.local_usage_anchor = None
        await state.events.emit(
            ContextCompactionEvent(
                compaction_id=compaction_id,
                phase="succeeded",
            ),
        )

    async def _compact_request_history(
        self,
        state: ModelRequestState,
        compaction: ContextCompactionState,
        messages: list[dict[str, Any]],
        pressure: tuple[int, str],
        *,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Replace accepted history H with a checkpoint while preserving delta."""
        measured, _source = pressure
        compaction_id = uuid4().hex
        await state.events.emit(
            ContextCompactionEvent(compaction_id=compaction_id, phase="started"),
        )
        try:
            delta_messages = compaction.delta_after_accepted(messages)
            consolidation_prefix = self.prepare_messages_for_model(
                state.config,
                compaction.accepted_messages,
            )
            summary = await compaction.consolidate_history(
                deepcopy(consolidation_prefix),
                compaction.active_summary,
            )
            if not summary:
                raise ContextWindowExceededError(
                    session_key=state.config.session_key,
                    estimated_tokens=measured,
                    input_budget=self.input_budget(state.config),
                    source=_source,
                )

            compaction.active_summary = summary
            prepared = self.prepare_messages_for_model(
                state.config,
                [
                    *self._summary_transcript(compaction, summary),
                    {"role": "user", "content": SUMMARY_CONTINUATION_TEXT},
                    *delta_messages,
                ],
            )
            # Responses-style state is append-only. Replacing H with a
            # checkpoint requires a fresh request; a successful response may
            # establish a new provider-owned state at the rewritten boundary.
            state.conversation.replace_transcript(compaction.raw_messages)
            state.usage = None
            state.local_usage_anchor = None
            prepared = self.ensure_request_fits(
                state.config,
                prepared,
                tool_definitions=tool_definitions,
            )
            compaction.summary_checkpoint = SessionSummaryCheckpoint(
                summary=summary,
                transcript_boundary=compaction.raw_accepted_boundary,
            )
        except (Exception, asyncio.CancelledError) as exc:
            await state.events.emit(
                ContextCompactionEvent(
                    compaction_id=compaction_id,
                    phase="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                ),
            )
            raise
        await state.events.emit(
            ContextCompactionEvent(
                compaction_id=compaction_id,
                phase="succeeded",
            ),
        )
        return prepared

    @staticmethod
    def project_source_images(
        state: ModelRequestState, view: SourceTranscript,
        transcript: list[dict[str, object]] | None,
    ) -> SourceTranscript:
        """Publish only the deterministic image projection, never request-only notes."""
        source = state.source
        if source is None:
            raise SourceValidationError('Image projection requires an active source ledger')
        notes = source.validate_view(view)
        if source.has_pending_requests:
            raise SourceValidationError('Cannot project images with an outstanding request ticket')
        active = source.transcript
        raw = state.source_compaction.raw_messages if state.source_compaction else transcript
        if raw is not None and raw != active.messages():
            raise SourceValidationError('Image projection differs from runner transcript')
        prepared = recent_image_messages(view.messages())
        nodes: list[RequestNode] = [
            node if isinstance(node, CheckpointNode) else replace(node, message=message)
            for node, message in zip(view.nodes, prepared, strict=True)
        ]
        updated = SourceTranscript(tuple(nodes[:len(active.nodes)]))
        extensions = tuple(
            replace(node, message=message)
            for node, message in zip(notes, prepared[len(active.nodes):], strict=True)
        )
        if updated != active:
            source.replace_projection(updated)
            if raw is not None:
                raw[:] = updated.messages()
            state.usage = None
            state.local_usage_anchor = None
            state.conversation.replace_transcript(updated.messages())
        return updated.with_runtime(*extensions)

    async def prepare_request(
        self,
        state: ModelRequestState,
        messages: list[dict[str, Any]],
        *,
        tool_definitions: list[dict[str, Any]] | None,
        transcript: list[dict[str, Any]] | None = None,
        source_view: SourceTranscript | None = None,
        force_source_compaction: bool = False,
        stream: bool = False,
        tool_choice: str | dict[str, Any] | None = None,
        recovery_root_reason: str | None = None,
    ) -> tuple[list[dict[str, Any]], ProviderCallContext | None]:
        """Prepare, compact or fit, and record the exact provider payload."""
        binding = self._binding_for_request(state, tool_definitions)
        generation = self._captured_generation(state.config)
        if (
            state.local_usage_anchor is not None
            and state.source is not None
            and state.local_usage_anchor.projection_epoch != state.source.projection_epoch
        ):
            state.local_usage_anchor = None
        request_context = ProviderCallContext(
            request_limits=derive_provider_request_limits(
                state.config.context_window_tokens,
                generation.max_tokens,
            ),
            context_window_tokens=state.config.context_window_tokens,
            local_usage_anchor=state.local_usage_anchor,
        )
        if state.config.preserve_source:
            if state.source is not None:
                source_view = source_view or state.source.transcript
                state.source.validate_view(source_view)
                if source_view.messages() != messages:
                    raise SourceValidationError('Model request differs from bound source projection')
            elif source_view is not None:
                raise SourceValidationError('Source view requires an active source ledger')
            if source_view is not None:
                source_view = self.admit_source_checkpoint(state, source_view, tool_definitions=tool_definitions)
                source_view = self.project_source_artifacts(state, source_view, tool_definitions=tool_definitions)
                source_view = self.project_source_images(state, source_view, transcript)
                request_context = replace(
                    request_context,
                    local_usage_anchor=state.local_usage_anchor,
                )
                source_view = await self.compact_source_request(
                    state,
                    source_view,
                    tool_definitions=tool_definitions,
                    force=force_source_compaction,
                    stream=stream,
                    tool_choice=tool_choice,
                    provider_context=request_context,
                    recovery_root_reason=recovery_root_reason,
                )
                messages = source_view.messages()
            else:
                messages = recent_image_messages(messages)
            prepared = self.ensure_request_fits(
                state.config,
                deepcopy(messages),
                tool_definitions=tool_definitions,
                stream=stream,
                tool_choice=tool_choice,
                provider_context=request_context,
            )
            if state.source_compaction is not None:
                state.source_compaction.writer.owner.validate_dispatch()
            state.source_view = deepcopy(source_view)
            state.messages = deepcopy(prepared)
            state.tool_definitions = deepcopy(tool_definitions)
            # Source checkpoints remain portable: never resume opaque provider
            # state, but preserve the same hard final-transport limits as a
            # normal runtime request.
            provider_context = state.conversation.prepare_request(
                messages,
                context_window_tokens=state.config.context_window_tokens,
                completion_reserve_tokens=state.config.max_tokens,
                resume_state=False,
                binding=None,
            )
            final_context = replace(
                provider_context or ProviderCallContext(),
                events=state.events,
                local_usage_anchor=state.local_usage_anchor,
            )
            final_context = self._attach_route_selection(
                state.config,
                prepared,
                tool_definitions=tool_definitions,
                stream=stream,
                tool_choice=tool_choice,
                provider_context=final_context,
            )
            return prepared, final_context
        prepared = self.prepare_messages_for_model(state.config, messages)
        image_view_changed = recent_image_messages(messages) != messages
        if image_view_changed:
            state.conversation.invalidate()
            state.local_usage_anchor = None
            request_context = replace(request_context, local_usage_anchor=None)
        model_messages: list[dict[str, Any]] | None = prepared
        supplemental_messages: list[dict[str, Any]] | None = None
        request_context_tokens = None
        if transcript is not None:
            if tool_definitions is None and not image_view_changed:
                model_messages = None
                supplemental_messages = [prepared[-1]]
            request_context_tokens = state.conversation.estimate_request_context_tokens(
                transcript,
                model_messages=model_messages,
                supplemental_messages=supplemental_messages,
                tool_definitions=tool_definitions,
                binding=binding,
            )
        usage_matches_messages = (
            state.messages is not None
            and prepared == state.messages
            and tool_definitions == state.tool_definitions
        )
        request_was_fitted = False
        compaction = state.compaction
        if compaction is None:
            pressure = self.request_pressure(
                state.config,
                prepared,
                state.usage,
                usage_matches_messages=usage_matches_messages,
                tool_definitions=tool_definitions,
                request_context_tokens=request_context_tokens,
                stream=stream,
                tool_choice=tool_choice,
                provider_context=request_context,
            )
            if pressure is not None:
                compaction_id = uuid4().hex
                await state.events.emit(
                    ContextCompactionEvent(compaction_id=compaction_id, phase="started"),
                )
                try:
                    prepared = self.fit_to_budget(
                        state.config,
                        prepared,
                        tool_definitions=tool_definitions,
                        stream=stream,
                        tool_choice=tool_choice,
                        provider_context=request_context,
                    )
                except Exception:
                    await state.events.emit(
                        ContextCompactionEvent(compaction_id=compaction_id, phase="failed"),
                    )
                    raise
                await state.events.emit(
                    ContextCompactionEvent(compaction_id=compaction_id, phase="succeeded"),
                )
                request_was_fitted = True
        else:
            pressure = self.request_pressure(
                state.config,
                prepared,
                state.usage,
                usage_matches_messages=usage_matches_messages,
                tool_definitions=tool_definitions,
                request_context_tokens=request_context_tokens,
                stream=stream,
                tool_choice=tool_choice,
                provider_context=request_context,
            )
            if pressure is not None:
                prepared = await self._compact_request_history(
                    state,
                    compaction,
                    messages,
                    pressure,
                    tool_definitions=tool_definitions,
                )
                model_messages = prepared
                supplemental_messages = None
        provider_context = (
            state.conversation.prepare_request(
                transcript,
                context_window_tokens=state.config.context_window_tokens,
                completion_reserve_tokens=state.config.max_tokens,
                model_messages=model_messages,
                supplemental_messages=supplemental_messages,
                resume_state=not request_was_fitted and not image_view_changed,
                binding=binding,
            )
            if transcript is not None
            else state.conversation.independent_request_context(
                context_window_tokens=state.config.context_window_tokens,
                completion_reserve_tokens=state.config.max_tokens,
            )
        )
        provider_context = replace(
            provider_context or ProviderCallContext(),
            events=state.events,
            local_usage_anchor=state.local_usage_anchor,
        )
        provider_context = self._attach_route_selection(
            state.config,
            prepared,
            tool_definitions=tool_definitions,
            stream=stream,
            tool_choice=tool_choice,
            provider_context=provider_context,
        )
        state.messages = deepcopy(prepared)
        state.tool_definitions = deepcopy(tool_definitions)
        return prepared, provider_context

    @staticmethod
    def _binding_for_request(
        state: ModelRequestState,
        tool_definitions: list[dict[str, Any]] | None,
    ) -> ConversationRequestBinding | None:
        """Return one complete binding for both estimate and dispatch.

        A no-tools finalization request has a different capability surface than
        a normal tool request.  It therefore receives its own value while the
        rest of the admitted binding remains unchanged.
        """
        binding = state.conversation_binding
        if binding is None:
            return None
        effective_tools: list[dict[str, Any]] = deepcopy(tool_definitions or [])
        tool_choice: dict[str, str] = {
            "type": "auto" if tool_definitions is not None else "none",
        }
        if binding.tools == effective_tools and binding.tool_choice == tool_choice:
            return binding
        return replace(binding, tools=effective_tools, tool_choice=tool_choice)

    @staticmethod
    def input_budget(config: ContextGovernanceConfig) -> int:
        context_window_tokens = config.context_window_tokens
        if type(context_window_tokens) is not int or context_window_tokens <= 0:
            return 0

        provider_max_tokens = ContextGovernor._captured_generation(config).max_tokens
        configured_max_tokens = config.max_tokens
        max_output = (
            configured_max_tokens
            if type(configured_max_tokens) is int and configured_max_tokens >= 0
            else provider_max_tokens
        )
        budget = context_window_tokens - max_output - SNIP_SAFETY_BUFFER
        return budget if budget > 0 else 0

    @staticmethod
    def normalize_tool_result(
        config: ContextGovernanceConfig,
        tool_call_id: str,
        tool_name: str,
        result: Any,
    ) -> Any:
        result = ensure_nonempty_tool_result(tool_name, result)
        if config.preserve_source:
            return result
        if tool_name in TOOL_RESULT_OFFLOAD_EXEMPT_TOOLS:
            return result

        def persist_text(text: str, call_id: str) -> str:
            content = maybe_persist_tool_result(
                config.workspace,
                config.session_key,
                call_id,
                text,
                max_chars=config.max_tool_result_chars,
            )
            # Persisted references must retain their complete paths.
            return truncate_text(content, config.max_tool_result_chars) if config.workspace is None else content

        original_result: object = result
        try:
            if isinstance(result, str):
                return persist_text(result, tool_call_id)
            if isinstance(result, list):
                result = cast(list[object], result)
                blocks: list[dict[str, Any]] = []
                for raw_block in result:
                    if not isinstance(raw_block, dict):
                        return result
                    block = cast(dict[str, Any], raw_block)
                    if not isinstance(block.get("type"), str):
                        return result
                    if block["type"] == "text" and not isinstance(block.get("text"), str):
                        return result
                    blocks.append(block)
                # Image redaction must not change how neighboring text is normalized.
                return [
                    {**block, "text": persist_text(block["text"], f"{tool_call_id}_text_{index}")}
                    if block["type"] == "text" else block
                    for index, block in enumerate(blocks)
                ]
        except Exception:
            logger.exception(
                "Tool result persist failed for {} in {}; using raw result",
                tool_call_id,
                config.session_key or "default",
            )
            return truncate_text(result, config.max_tool_result_chars) if isinstance(result, str) else original_result
        return result

    @staticmethod
    def strip_placeholder_assistant_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove assistant messages that are compaction placeholders.

        Messages like ``[Previous assistant message omitted.]`` carry no useful
        context for the model and can cause it to repeatedly attempt tool calls
        that previously failed, producing malformed responses in a loop.
        Consecutive same-role messages that result from removal are handled
        downstream by the provider's merge-consecutive logic. Only the
        model-facing copy is repaired; the persisted transcript is untouched
        (a copy is returned, or the same list object when nothing changes).
        """
        updated: list[dict[str, Any]] | None = None
        for idx, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                if updated is not None:
                    updated.append(msg)
                continue
            content = msg.get("content", "")
            text = content if isinstance(content, str) else ""
            is_placeholder = text.strip() in PLACEHOLDER_TEXTS
            has_tool_calls = bool(msg.get("tool_calls"))
            if is_placeholder and not has_tool_calls:
                if updated is None:
                    updated = list(messages[:idx])
                logger.debug(
                    "Stripping placeholder assistant message from history: {!r}",
                    text[:60],
                )
                continue
            if updated is not None:
                updated.append(msg)
        if updated is None:
            return messages
        return updated

    @staticmethod
    def strip_malformed_tool_calls(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop persisted assistant tool_calls whose name is missing/non-string.

        A degenerate tool call (``name=None`` or ``""``) that slipped into the
        saved history before this guard existed gets replayed on every turn and
        makes upstream APIs reject the whole request
        (``messages.content.N.tool_use.name: Input should be a valid string``),
        permanently wedging the session. Removing the bad call here lets the
        existing orphan-result cleanup drop its now-dangling tool result, so a
        polluted session self-heals on its next turn. The persisted transcript
        is left untouched; only the model-facing copy is repaired (a copy is
        returned, or the same list object when nothing changes).
        """
        updated: list[dict[str, Any]] | None = None
        for idx, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                if updated is not None:
                    updated.append(msg)
                continue
            calls = msg.get("tool_calls")
            if not calls:
                if updated is not None:
                    updated.append(msg)
                continue
            kept = [tc for tc in cast(list[Any], calls) if _tool_call_name_is_valid(tc)]
            if len(kept) == len(calls):
                if updated is not None:
                    updated.append(msg)
                continue
            if updated is None:
                updated = [dict(m) for m in messages[:idx]]
            logger.warning(
                "Stripping {} malformed tool_call(s) with missing/non-string "
                "name from assistant history before request",
                len(calls) - len(kept),
            )
            repaired = dict(msg)
            if kept:
                repaired["tool_calls"] = kept
            else:
                repaired.pop("tool_calls", None)
            # An assistant turn with neither content nor any valid tool call is
            # itself invalid upstream; drop it entirely in that case.
            has_content = bool(repaired.get("content"))
            if not kept and not has_content:
                continue
            updated.append(repaired)

        if updated is None:
            return messages
        return updated

    @staticmethod
    def drop_orphan_tool_results(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop invalid tool results before history is sent back to providers."""
        declared: set[str] = set()
        fulfilled: set[str] = set()
        updated: list[dict[str, Any]] | None = None
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in cast(list[Any], msg.get("tool_calls") or []):
                    if isinstance(tc, dict):
                        tool_call = cast(dict[str, Any], tc)
                        if tool_call.get("id"):
                            declared.add(str(tool_call["id"]))
            if role == "tool":
                tid = msg.get("tool_call_id")
                tid_str = str(tid) if tid else ""
                if not tid_str or tid_str not in declared or tid_str in fulfilled:
                    if updated is None:
                        updated = [dict(m) for m in messages[:idx]]
                    continue
                fulfilled.add(tid_str)
            if updated is not None:
                updated.append(dict(msg))

        if updated is None:
            return messages
        return updated

    @staticmethod
    def backfill_missing_tool_results(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Insert synthetic error results for assistant tool_calls with missing tool outputs."""
        declared: list[tuple[int, str, str]] = []
        fulfilled: set[str] = set()
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in cast(list[Any], msg.get("tool_calls") or []):
                    if isinstance(tc, dict):
                        name = ""
                        tool_call = cast(dict[str, Any], tc)
                        if tool_call.get("id"):
                            func = tool_call.get("function")
                            if isinstance(func, dict):
                                func_data = cast(dict[str, Any], func)
                                raw_name = func_data.get("name", "")
                                name = raw_name if isinstance(raw_name, str) else str(raw_name)
                            declared.append((idx, str(tool_call["id"]), name))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid:
                    fulfilled.add(str(tid))

        missing = [(ai, cid, name) for ai, cid, name in declared if cid not in fulfilled]
        if not missing:
            return messages

        updated = list(messages)
        offset = 0
        for assistant_idx, call_id, name in missing:
            insert_at = assistant_idx + 1 + offset
            while insert_at < len(updated) and updated[insert_at].get("role") == "tool":
                insert_at += 1
            updated.insert(insert_at, {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": BACKFILL_CONTENT,
            })
            offset += 1
        return updated

    def apply_tool_result_budget(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        updated = messages
        for idx, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            normalized = self.normalize_tool_result(
                config,
                str(message.get("tool_call_id") or f"tool_{idx}"),
                str(message.get("name") or "tool"),
                message.get("content"),
            )
            if normalized != message.get("content"):
                if updated is messages:
                    updated = [dict(m) for m in messages]
                updated[idx]["content"] = normalized
        return updated

    def snip_history(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        *,
        tool_definitions: list[dict[str, Any]] | None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        if not messages or not config.context_window_tokens:
            return messages

        budget = self.input_budget(config)
        if budget <= 0:
            return messages

        if not force:
            estimate, _ = estimate_prompt_tokens_chain(
                config.provider,
                config.model,
                messages,
                tool_definitions,
            )
            if estimate <= budget:
                return messages

        system_messages = [dict(msg) for msg in messages if msg.get("role") == "system"]
        non_system = [dict(msg) for msg in messages if msg.get("role") != "system"]
        if not non_system:
            return messages

        system_tokens = sum(estimate_message_tokens(msg) for msg in system_messages)
        fixed_tokens, _ = estimate_prompt_tokens_chain(
            config.provider,
            config.model,
            system_messages,
            tool_definitions,
        )
        remaining_budget = max(0, budget - max(system_tokens, fixed_tokens))
        kept: list[dict[str, Any]] = []
        kept_tokens = 0
        for message in reversed(non_system):
            msg_tokens = estimate_message_tokens(message)
            if kept and kept_tokens + msg_tokens > remaining_budget:
                break
            kept.append(message)
            kept_tokens += msg_tokens
        kept.reverse()

        return system_messages + self._legal_history_tail(kept, non_system)

    def _legal_history_tail(
        self,
        kept: list[dict[str, Any]],
        non_system: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fallback = kept if kept else (non_system[-1:] if non_system else [])
        kept = self._user_tail(kept) or self._user_tail(non_system, last=True) or fallback

        start = find_legal_message_start(kept)
        return kept[start:] if start else kept

    @staticmethod
    def _user_tail(messages: list[dict[str, Any]], *, last: bool = False) -> list[dict[str, Any]]:
        indexes = range(len(messages) - 1, -1, -1) if last else range(len(messages))
        for idx in indexes:
            if messages[idx].get("role") == "user":
                return messages[idx:]
        return []
