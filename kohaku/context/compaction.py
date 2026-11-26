"""Compaction for canonical source, legacy replay, and idle sessions."""

from __future__ import annotations

import asyncio
from bisect import bisect_right
from collections.abc import Awaitable, Callable
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from loguru import logger

from kohaku.agent import context_governance as governance
from kohaku.agent.context_governance import (
    SNIP_SAFETY_BUFFER,
    ContextCompactionState,
    ContextGovernanceConfig,
    ContextGovernor,
    ContextWindowExceededError,
    ModelRequestState,
    SourceCompactionMetrics,
    SourceCompactionState,
    SourceCompactionUnavailableError,
)
from kohaku.events import NO_EVENTS, ContextCompactionEvent, EventSink
from kohaku.providers.base import ProviderCallContext
from kohaku.providers.conversation_state import ProviderConversationStateController
from kohaku.providers.request_budget import (
    PreparedRequestBudgetPreview,
    ProviderRequestRejected,
    RequestPayloadBytesExceeded,
)
from kohaku.session.source import SourceValidationError
from kohaku.session.source_projection import CheckpointNode, source_nodes
from kohaku.session.source_request import SourceRequestLedger, SourceTranscript
from kohaku.session.summary import SUMMARY_CONTINUATION_TEXT, SessionSummaryCheckpoint

if TYPE_CHECKING:
    from kohaku.agent.tools.registry import ToolRegistry
    from kohaku.config.schema import IdleCompactionConfig, SourceCompactionConfig
    from kohaku.session.checkpoint_dependencies import SummaryEvidence
    from kohaku.session.manager import SessionManager
    from kohaku.session.turn_writer import SourceTurnWriter
    from kohaku.utils.llm_runtime import LLMRuntime


class CompactionEngine(ContextGovernor):
    """Compact model history through the owning source or legacy session."""

    async def compact_now(
        self,
        session_key: str,
        *,
        writer: SourceTurnWriter,
        runtime: LLMRuntime,
        tools: ToolRegistry,
        settings: SourceCompactionConfig,
        max_tool_result_chars: int,
        force: bool = True,
        events: EventSink = NO_EVENTS,
    ) -> bool:
        """Compact a persisted session under the caller's existing execution owner."""
        if writer.owner.key != session_key:
            raise ValueError("Compaction requires the current session owner.")
        snapshot = writer.owner.read_snapshot()
        nodes = source_nodes(snapshot.events)
        if not nodes:
            return False
        view = SourceTranscript(nodes)
        rows = view.messages()
        last_user_index = next(
            (idx for idx in range(len(nodes) - 1, -1, -1) if nodes[idx].message.get("role") == "user"),
            None,
        )
        checkpoint_boundary = (
            int(snapshot.checkpoint["boundary"])
            if snapshot.checkpoint is not None and isinstance(snapshot.checkpoint.get("boundary"), int)
            else 0
        )
        covered_ids = {event["_event_id"] for event in snapshot.events[:checkpoint_boundary]}
        prefix_nodes = (
            nodes[:last_user_index]
            if last_user_index is not None and last_user_index > 0
            else ()
        )
        # Keep the newest user turn uncompacted when earlier uncompacted history
        # exists; if all earlier turns are already covered by a checkpoint, allow
        # compacting the remaining completed turn.
        has_uncovered_prefix = any(node.event_id not in covered_ids for node in prefix_nodes)
        eligible_nodes = prefix_nodes if has_uncovered_prefix else nodes
        state = ModelRequestState(
            config=ContextGovernanceConfig(
                runtime.provider, runtime.model, tools, None, session_key, max_tool_result_chars,
                context_window_tokens=runtime.context_window_tokens,
                max_tokens=runtime.generation.max_tokens, generation=runtime.generation,
                preserve_source=True,
            ),
            conversation=ProviderConversationStateController(
                provider=runtime.provider, model=runtime.model, messages=rows, state=None,
            ),
            source=SourceRequestLedger(
                view, initial_eligible_ids=frozenset(n.event_id for n in eligible_nodes),
            ),
            source_compaction=SourceCompactionState(
                writer, runtime, rows, None,
                idle_timeout_seconds=settings.idle_timeout_seconds,
                overall_timeout_seconds=settings.overall_timeout_seconds,
                pressure_trigger_ratio=settings.pressure_trigger_ratio,
                target_ratio=settings.target_ratio,
                summary_generation_tokens=settings.summary_generation_tokens,
            ),
            events=events,
        )
        definitions = tools.get_definitions()
        view = self.admit_source_checkpoint(state, view, tool_definitions=definitions)
        await self.compact_source_request(
            state, view, tool_definitions=definitions, force=force,
        )
        return writer.owner.read_snapshot().state.checkpoint_id != snapshot.state.checkpoint_id

    async def compact_source_request(self: ContextGovernor, state: ModelRequestState, view: SourceTranscript, *,
                                     tool_definitions: list[dict[str, Any]] | None,
                                     force: bool = False,
                                     stream: bool = False,
                                     tool_choice: str | dict[str, Any] | None = None,
                                     provider_context: ProviderCallContext | None = None,
                                     recovery_root_reason: str | None = None) -> SourceTranscript:
        """Summarize eligible closed source chunks at the configured pressure."""
        from kohaku.agent.memory import Consolidator
        from kohaku.session.checkpoint_dependencies import (
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
            estimated, measurement_source = governance.estimate_prompt_tokens_chain(
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
            fixed_tokens, _ = governance.estimate_prompt_tokens_chain(
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
        from kohaku.providers.openai_compat_provider import OpenAICompatProvider

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
                from kohaku.agent.memory import Consolidator

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
            def fitting_chunk(
                after: int, previous: str | None, upper: int,
            ) -> tuple[int, SummaryEvidence | None, list[dict[str, Any]] | None, str | None]:
                low = bisect_right(candidate.boundaries, after)
                high = upper
                best: tuple[int, SummaryEvidence | None, list[dict[str, Any]] | None] = (after, None, None)
                rejection = None
                while low < high:
                    check_deadline()
                    index = (low + high) // 2
                    boundary = candidate.boundaries[index]
                    selected = candidate.summary_evidence(boundary, after=after)
                    if not selected.nodes:
                        low = index + 1
                        continue
                    payload = Consolidator.source_summary_messages(
                        [node.message for node in selected.nodes], previous,
                    )
                    try:
                        measurement = self._compatibility_auxiliary_preview(
                            compaction.runtime, payload, max_output_tokens=allowance,
                            tools=None, tool_choice=None, mode='canonical',
                            idle_timeout_seconds=compaction.idle_timeout_seconds,
                            deadline_monotonic=deadline,
                        )
                    except RequestPayloadBytesExceeded as exc:
                        rejection = exc.reason
                        high = index
                        continue
                    except ProviderRequestRejected as exc:
                        raise SourceCompactionUnavailableError(
                            'Auxiliary request envelope was rejected', reason=exc.reason,
                        ) from None
                    if measurement is None:
                        tokens, _ = governance.estimate_prompt_tokens_chain(
                            compaction.runtime.provider, compaction.runtime.model, payload, None,
                        )
                        fits = tokens + allowance + SNIP_SAFETY_BUFFER <= compaction.runtime.context_window_tokens
                    else:
                        fits = measurement.measurement.fits
                    if fits:
                        best = boundary, selected, payload
                        low = index + 1
                    else:
                        high = index
                return best[0], best[1], best[2], rejection

            # Preflight the partitions that can be reached within the fold limit.
            # A fitting prefix also admits its shorter closed groups.
            fold_limit = 4
            partition = cursor
            preflight_folds = fold_limit
            if prefix_mode == 'historical_prefix' and prefix_selection is not None:
                if prefix_selection.boundary is None:
                    raise SourceValidationError('Verified historical prefix selection lost its boundary')
                partition = prefix_selection.boundary
                preflight_folds -= 1
            for _ in range(preflight_folds):
                if partition == final_boundary:
                    break
                boundary, _, _, rejection = fitting_chunk(partition, None, len(candidate.boundaries))
                if boundary == partition:
                    raise SourceCompactionUnavailableError(
                        'Indivisible source group exceeds the auxiliary budget',
                        reason=rejection or 'indivisible_source_group_exceeds_auxiliary_budget',
                    )
                partition = boundary

            truncated_boundaries: set[int] = set()
            truncated_failure: str | None = None
            for fold_index in range(fold_limit):
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
                    upper = len(candidate.boundaries)
                    if truncated_boundaries:
                        upper = bisect_right(candidate.boundaries, min(truncated_boundaries) - 1)
                    chosen, evidence, prepared, chunk_rejection_reason = fitting_chunk(cursor, summary, upper)
                if chosen is None or evidence is None or prepared is None:
                    if truncated_failure is not None:
                        raise SourceCompactionUnavailableError(
                            'Bounded source summary failed: invalid_response',
                            reason=truncated_failure,
                        )
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
                        and fold_index < fold_limit - 1
                    ):
                        # Discard the truncated candidate and retry the same
                        # evidence cursor with the next smaller closed chunk.
                        truncated_boundaries.add(chosen)
                        truncated_failure = failure_reason
                        check_deadline()
                        continue
                    raise SourceCompactionUnavailableError(
                        'Bounded source summary failed: ' + result.status,
                        reason=failure_reason,
                    )
                consumed.append(ConsumedSummaryChunk(evidence, summary, result.summary))
                summary, cursor = result.summary, chosen
                truncated_boundaries.clear()
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
                    after_tokens, _ = governance.estimate_prompt_tokens_chain(
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

    async def compact_legacy_history(
        self: ContextGovernor,
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
            state.invalidate_context_usage()
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

    def snip_history(
        self: ContextGovernor,
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
            estimate, _ = governance.estimate_prompt_tokens_chain(
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

        system_tokens = sum(governance.estimate_message_tokens(msg) for msg in system_messages)
        fixed_tokens, _ = governance.estimate_prompt_tokens_chain(
            config.provider,
            config.model,
            system_messages,
            tool_definitions,
        )
        remaining_budget = max(0, budget - max(system_tokens, fixed_tokens))
        anchor = next((msg for msg in non_system if msg.get("role") == "user"), None)
        pin_anchor = anchor is not None and (config.session_key or "").startswith(("delegated:", "child:"))
        if pin_anchor and anchor is not None:
            remaining_budget = max(0, remaining_budget - governance.estimate_message_tokens(anchor))
        kept: list[dict[str, Any]] = []
        kept_tokens = 0
        for message in reversed(non_system):
            msg_tokens = governance.estimate_message_tokens(message)
            if kept and kept_tokens + msg_tokens > remaining_budget:
                break
            kept.append(message)
            kept_tokens += msg_tokens
        kept.reverse()

        tail = self._legal_history_tail(kept, non_system)
        if pin_anchor and anchor is not None and all(row is not anchor for row in tail):
            tail.insert(0, anchor)
        return system_messages + tail


async def run_idle_compaction(
    sessions: SessionManager,
    settings: IdleCompactionConfig,
    compact: Callable[[str, datetime, int], Awaitable[bool]],
) -> None:
    """Inspect idle canonical sessions during the gateway lifetime."""
    checked: dict[str, str] = {}
    while True:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=settings.idle_minutes)
        rows = await asyncio.to_thread(sessions.list_sessions)
        observed: set[str] = set()
        for row in rows:
            key: object = row.get("key")
            updated: object = row.get("updated_at")
            if not isinstance(key, str) or not isinstance(updated, str):
                continue
            if key.startswith(("dream:", "delegated:", "child:", "ephemeral:")):
                continue
            observed.add(key)
            if checked.get(key) == updated:
                continue
            try:
                stamp = datetime.fromisoformat(updated)
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                if stamp <= cutoff and await compact(key, cutoff, settings.min_messages):
                    checked[key] = updated
            except Exception:
                logger.exception("Could not compact an idle session.")
        checked = {key: stamp for key, stamp in checked.items() if key in observed}
        await asyncio.sleep(min(60, settings.idle_minutes * 60))

