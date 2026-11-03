"""Source-backed recovery from finalized provider request limits."""

from __future__ import annotations

import asyncio
import json
import struct
import zlib
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI

from nanobot.agent.context_governance import (
    ContextGovernanceConfig,
    ContextGovernor,
    ModelRequestState,
    SourceCompactionState,
)
from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.loop import AgentLoop
from nanobot.agent.runner import AgentRunner
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import TurnCallbacks, TurnRequest
from nanobot.config.schema import Config
from nanobot.events import AgentEvent, ContextCompactionEvent, EventSink
from nanobot.providers.base import (
    GenerationSettings,
    LLMResponse,
    LLMUsage,
    ProviderCallContext,
    ToolCallRequest,
)
from nanobot.providers.conversation_state import ProviderConversationStateController
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import find_by_name
from nanobot.providers.request_budget import (
    ESTIMATED_COMPAT_POLICY,
    BackendContextOverflow,
    FinalizedRequestBudgetExceeded,
    LocalRequestUsageReceipt,
    LocalUsageAnchor,
    ProviderRequestLimits,
    TransportIdentity,
    TransportMeasurement,
)
from nanobot.session.checkpoint_dependencies import SummaryDependencyProof
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import SessionWriteConflict
from nanobot.session.source import SourceValidationError, validate_execution_links
from nanobot.session.source_projection import CheckpointNode, RuntimeNode, SourceNode, source_nodes
from nanobot.session.source_request import (
    SourceRequestLedger,
    SourceRequestTicket,
    SourceTranscript,
)
from nanobot.session.turn_writer import SourceTurnWriter
from nanobot.utils.helpers import estimate_prompt_tokens_chain
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.runtime import build_budget_exhausted_finalization_message
from tests.agent.test_migration_engine import ScriptedProvider
from tests.agent.test_migration_engine import engine_config as engine_config
from tests.agent.test_source_pressure_compaction import pressure_state
from tests.agent.test_source_pressure_compaction import store as store

_MODEL = "Qwen3.8-27B-UD-Q6_K_XL"
_CONTEXT_WINDOW = 165_888
_COMPLETION_RESERVE = 8_192
_SAFETY_BUFFER = 1_024
_TEXT_RESULT_SIZE = 50_000


class _TextChunkTool(Tool):
    def __init__(self, result_size: int = _TEXT_RESULT_SIZE,
                 large_result_label: str | None = None) -> None:
        self.calls: list[str] = []
        self.result_size = result_size
        self.large_result_label = large_result_label

    @property
    def name(self) -> str:
        return "make_text_chunk"

    @property
    def description(self) -> str:
        return "Return deterministic local fixture evidence."

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        }

    async def execute(self, *, label: str) -> str:
        self.calls.append(label)
        size = 200_000 if label == self.large_result_label else self.result_size
        return f"{label}:" + ("x" * size)


def _png_bytes(width: int, height: int, padding: int) -> bytes:
    def chunk(name: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", zlib.crc32(name + data))
        )

    pixels = zlib.compress((b"\x00" + b"\x00" * (width * 3)) * height)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"tEXt", b"Comment\x00" + b"p" * padding)
        + chunk(b"IDAT", pixels)
        + chunk(b"IEND", b"")
    )


def _tool_call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _sdk_response(*, content: str | None = None,
                  tool_calls: list[dict[str, object]] | None = None) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}]}


def _stream_sdk_response(*, content: str | None = None,
                         tool_calls: list[dict[str, object]] | None = None,
                         reasoning_content: str | None = None):
    async def chunks():
        delta = SimpleNamespace(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=None,
        )
        if tool_calls:
            delta.tool_calls = [SimpleNamespace(
                index=index,
                id=call["id"],
                type="function",
                function=SimpleNamespace(
                    name=call["function"]["name"],
                    arguments=call["function"]["arguments"],
                ),
            ) for index, call in enumerate(tool_calls)]
        yield SimpleNamespace(choices=[SimpleNamespace(
            delta=delta,
            finish_reason="tool_calls" if tool_calls else "stop",
        )])

    return chunks()


class _CaptureHook(AgentHook):
    def __init__(self, streaming: bool) -> None:
        super().__init__()
        self.streaming = streaming
        self.content: list[str] = []
        self.reasoning: list[str] = []

    def wants_streaming(self) -> bool:
        return self.streaming

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        self.content.append(delta)

    async def emit_reasoning(self, reasoning_content: str | None) -> None:
        if reasoning_content:
            self.reasoning.append(reasoning_content)


def _text_content(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_text_content(item) for item in value)
    if isinstance(value, dict):
        return "".join(_text_content(item) for item in value.values())
    return ""


def _is_source_summary_request(messages: list[object]) -> bool:
    """Recognize canonical and replayed-prefix summary bodies in SDK fixtures."""
    if not messages:
        return False
    first = messages[0]
    last = messages[-1]
    first_content = first.get("content") if isinstance(first, dict) else None
    last_content = last.get("content") if isinstance(last, dict) else None
    return (
        isinstance(first_content, str)
        and first_content.startswith("Summarize historical evidence")
    ) or (
        isinstance(last_content, str)
        and last_content.startswith("Summarize only the historical user, assistant, and tool exchange")
    )


def _local_overflow_measurement(input_limit: int, *, stream: bool = False) -> TransportMeasurement:
    reserve = 256
    safety = _SAFETY_BUFFER
    total = input_limit + reserve + safety
    return TransportMeasurement(
        identity=TransportIdentity(
            "local_manager",
            "openai_chat_completions_stream" if stream else "openai_chat_completions",
            _MODEL,
        ),
        input_tokens=input_limit + 1,
        completion_reserve_tokens=reserve,
        safety_buffer_tokens=safety,
        total_limit_tokens=total,
        input_limit_tokens=input_limit,
        source=ESTIMATED_COMPAT_POLICY,
        policy=ESTIMATED_COMPAT_POLICY,
    )


def _upgrade_source_session(loop, key: str):
    session = loop.sessions.get_or_create(key)
    loop.sessions.save(session)
    backend = loop.sessions._jsonl_store
    backend.source.upgrade_source_session(
        key,
        expected_revision=session.revision,
        expected_generation=session.generation,
        expected_incarnation=session.incarnation,
    )
    loop.sessions.invalidate(key)
    return backend


@pytest.mark.parametrize(
    ("streaming", "finalization", "irreducible", "no_progress",
     "second_backend_overflow"),
    [(False, False, False, False, False), (True, False, False, False, False),
     (False, True, False, False, False), (False, False, True, False, False),
     (False, False, False, True, False), (True, False, False, False, True)],
    ids=["nonstreaming", "streaming", "no-tools-finalization", "large-local-tail",
         "structural-no-progress", "second-backend-overflow"],
)
async def test_source_turn_recovers_one_finalized_request_budget_rejection(
    engine_config: Config, tmp_path, monkeypatch, streaming: bool, finalization: bool,
    irreducible: bool, no_progress: bool, second_backend_overflow: bool,
) -> None:
    from nanobot.agent import runner as runner_module
    from nanobot.agent.loop import AgentLoop

    image_path = engine_config.workspace_path / "large-fixture.png"
    image_path.write_bytes(_png_bytes(1024, 1024, padding=225_000))
    assert image_path.stat().st_size >= 200_000

    chunk_tool = _TextChunkTool(large_result_label="chunk-3" if irreducible else None)
    tools = ToolRegistry()
    provider = OpenAICompatProvider(
        api_key="fixture",
        default_model=_MODEL,
        spec=find_by_name("local_manager"),
        provider_name="local_manager",
    )
    provider.generation = GenerationSettings(
        max_tokens=_COMPLETION_RESERVE,
        temperature=0,
        reasoning_effort="none",
    )
    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=tools,
        provider=provider,
        model=_MODEL,
        context_window_tokens=_CONTEXT_WINDOW,
    )
    loop.max_iterations = 4 if finalization else 10
    loop.tools.register(chunk_tool)
    backend = _upgrade_source_session(loop, "cli:budget-recovery")

    if no_progress:
        def no_op_publication(governor, request_state, attempt, **kwargs):
            assert request_state.source is not None
            return request_state.source.transcript

        monkeypatch.setattr(ContextGovernor, "publish_source_compaction", no_op_publication)

    source = backend.source
    source_type = type(source)
    source_operations: list[tuple[str, object]] = []
    issued: list[tuple[SourceRequestLedger, SourceRequestTicket, SourceTranscript,
                       frozenset[str]]] = []
    request_ledgers: list[SourceRequestLedger] = []
    latest_ticket: SourceRequestTicket | None = None
    original_issue = SourceRequestLedger.issue
    original_discard = SourceRequestLedger.discard
    original_acknowledge = SourceRequestLedger.acknowledge

    def issue(ledger: SourceRequestLedger, view):
        nonlocal latest_ticket
        ticket = original_issue(ledger, view)
        latest_ticket = ticket
        request_ledgers.append(ledger)
        issued.append((ledger, ticket, deepcopy(view), ledger.eligible_ids))
        source_operations.append(("issue", ticket))
        return ticket

    def discard(ledger: SourceRequestLedger, ticket):
        source_operations.append(("discard", ticket))
        original_discard(ledger, ticket)

    def acknowledge(ledger: SourceRequestLedger, ticket):
        source_operations.append(("acknowledge", ticket))
        original_acknowledge(ledger, ticket)

    monkeypatch.setattr(SourceRequestLedger, "issue", issue)
    monkeypatch.setattr(SourceRequestLedger, "discard", discard)
    monkeypatch.setattr(SourceRequestLedger, "acknowledge", acknowledge)

    original_append = source_type.append_source_events
    def append_events(store, key, expected, events, *, owner=None):
        source_operations.append(("write", tuple(event.get("_event_id") for event in events)))
        return original_append(store, key, expected, events, owner=owner)

    monkeypatch.setattr(source_type, "append_source_events", append_events)

    main_create_calls: list[dict[str, object]] = []
    auxiliary_create_calls: list[dict[str, object]] = []

    async def create(**kwargs: object) -> object:
        body = deepcopy(kwargs)
        messages = body.get("messages")
        assert isinstance(messages, list)
        if _is_source_summary_request(messages):
            auxiliary_create_calls.append(body)
            source_operations.append(("auxiliary", len(auxiliary_create_calls)))
            if body.get("stream") is True:
                return _stream_sdk_response(
                    content="Retained the image review and three evidence chunks."
                )
            return _sdk_response(content="Retained the image review and three evidence chunks.")

        main_create_calls.append(body)
        main_index = len(main_create_calls)
        if main_index == 1:
            response = _sdk_response(tool_calls=[_tool_call(
                "read-image", "read_file", {"path": str(image_path)},
            )], content="Earlier progress.")
        elif main_index in {2, 3, 4}:
            response = _sdk_response(tool_calls=[_tool_call(
                f"chunk-{main_index - 1}", "make_text_chunk", {"label": f"chunk-{main_index - 1}"},
            )])
        elif second_backend_overflow:
            raise BackendContextOverflow(
                status_code=400,
                code=400,
                n_prompt_tokens=227_502,
                n_ctx=165_888,
            )
        else:
            response = _sdk_response(content="Recovered after budget pressure.")
        if body.get("stream") is True:
            calls = response["choices"][0]["message"].get("tool_calls")
            return _stream_sdk_response(content=response["choices"][0]["message"].get("content"),
                                        tool_calls=calls,
                                        reasoning_content=(
                                            "Earlier reasoning." if main_index == 1 else None
                                        ))
        return response

    sdk_create = AsyncMock(side_effect=create)
    provider._ensure_client = AsyncMock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=sdk_create)),
    ))

    finalized_measurements: list[TransportMeasurement | None] = []
    guard_records: list[tuple[bool, dict[str, object],
                              TransportMeasurement | None,
                              SourceRequestTicket | None]] = []
    before_recovery_events: list[dict[str, object]] = []
    original_guard = provider.validate_finalized_request
    main_guard_calls = 0

    def observe_guard(
        body: dict[str, object],
        *,
        transport: str,
        model: str | None,
        provider_context: ProviderCallContext | None = None,
        completion_reserve_tokens: int | None = None,
        policy: str = "strict",
    ) -> TransportMeasurement | None:
        nonlocal main_guard_calls
        body_messages = body.get("messages")
        assert isinstance(body_messages, list)
        is_auxiliary = _is_source_summary_request(body_messages)
        try:
            result = original_guard(
                body,
                transport=transport,
                model=model,
                provider_context=provider_context,
                completion_reserve_tokens=completion_reserve_tokens,
                policy=policy,
            )
        except FinalizedRequestBudgetExceeded as exc:
            finalized_measurements.append(exc.measurement)
            guard_records.append((is_auxiliary, deepcopy(body), exc.measurement,
                                  None if is_auxiliary else latest_ticket))
            if not is_auxiliary:
                before_recovery_events.extend(deepcopy(
                    source.load_source_snapshot("cli:budget-recovery").events
                ))
            source_operations.append(("guard_rejected", exc.measurement))
            raise
        if not is_auxiliary:
            main_guard_calls += 1
            # Exercise the final local admission boundary after the four
            # completed tool-producing requests.  The measurement uses the
            # actual serialized local body and moves it one unit over the
            # captured boundary, leaving the completed tool work untouched.
            if main_guard_calls == 5:
                assert result is not None and result.input_limit_tokens is not None
                rejected = replace(
                    result,
                    input_tokens=result.input_limit_tokens + 1,
                    source=ESTIMATED_COMPAT_POLICY,
                    policy=ESTIMATED_COMPAT_POLICY,
                )
                exc = FinalizedRequestBudgetExceeded(rejected)
                finalized_measurements.append(rejected)
                guard_records.append((False, deepcopy(body), rejected, latest_ticket))
                before_recovery_events.extend(deepcopy(
                    source.load_source_snapshot("cli:budget-recovery").events
                ))
                source_operations.append(("guard_rejected", rejected))
                raise exc
        finalized_measurements.append(result)
        guard_records.append((is_auxiliary, deepcopy(body), result,
                              None if is_auxiliary else latest_ticket))
        source_operations.append(("guard_fitted", result))
        return result

    monkeypatch.setattr(provider, "validate_finalized_request", observe_guard)
    budget_error_measurements: list[TransportMeasurement | None] = []
    terminal_budget_errors: list[
        tuple[FinalizedRequestBudgetExceeded | BackendContextOverflow, str | None]
    ] = []
    original_budget_error = runner_module._context_budget_error_response

    def observe_budget_error(
        exc: FinalizedRequestBudgetExceeded | BackendContextOverflow,
        recovery_failure_reason: str | None = None,
    ):
        terminal_budget_errors.append((exc, recovery_failure_reason))
        budget_error_measurements.append(
            exc.measurement if isinstance(exc, FinalizedRequestBudgetExceeded) else None
        )
        return original_budget_error(exc, recovery_failure_reason)

    monkeypatch.setattr(runner_module, "_context_budget_error_response", observe_budget_error)

    streamed: list[str] = []
    capture_hook = _CaptureHook(streaming)

    async def on_stream(delta: str) -> None:
        streamed.append(delta)

    try:
        result = await loop.execute_turn(TurnRequest(
            content="Inspect the image and collect three evidence chunks.",
            session_key="cli:budget-recovery",
            callbacks=TurnCallbacks(on_stream=on_stream) if streaming else TurnCallbacks(),
            hooks=(capture_hook,),
        ))

        if irreducible:
            assert result.error is None
            assert result.content == "Recovered after budget pressure."
            assert len(main_create_calls) == 5
            assert budget_error_measurements == []
        elif second_backend_overflow:
            assert result.error is not None
            assert result.content != "Recovered after budget pressure."
            assert len(main_create_calls) == 5
            assert len(terminal_budget_errors) == 1
            terminal_error, recovery_reason = terminal_budget_errors[0]
            assert isinstance(terminal_error, FinalizedRequestBudgetExceeded)
            assert recovery_reason == "backend_context_overflow"
        elif no_progress:
            assert result.error is not None
            assert result.content != "Recovered after budget pressure."
            assert len(main_create_calls) == 4
            assert len(budget_error_measurements) == 1
        else:
            assert result.error is None, f"{result.error}; measurements={finalized_measurements!r}"
            assert result.content == "Recovered after budget pressure."
            assert len(main_create_calls) == 5
        assert 1 <= len(auxiliary_create_calls) <= 4
        assert chunk_tool.calls == ["chunk-1", "chunk-2", "chunk-3"]
        main_rejections = [record for record in guard_records
                           if not record[0] and record[2] is not None and not record[2].fits]
        assert len(main_rejections) == 1
        _, rejected_body, rejected_measurement, failed_ticket = main_rejections[0]
        assert rejected_measurement is not None and failed_ticket is not None
        rejected_messages = rejected_body.get("messages")
        assert isinstance(rejected_messages, list)
        logical_tokens, _ = estimate_prompt_tokens_chain(
            provider, _MODEL, rejected_messages, rejected_body.get("tools"),
        )
        assert logical_tokens < (_CONTEXT_WINDOW - _COMPLETION_RESERVE - _SAFETY_BUFFER) * 0.9

        failed_operation_index = next(
            index for index, (kind, value) in enumerate(source_operations)
            if kind == "guard_rejected" and value == rejected_measurement
        )
        failed_discard_index = next(
            index for index in range(failed_operation_index + 1, len(source_operations))
            if source_operations[index] == ("discard", failed_ticket)
        )
        first_auxiliary_index = next(
            index for index, (kind, _) in enumerate(source_operations)
            if kind == "auxiliary"
        )
        assert failed_discard_index < first_auxiliary_index
        assert sum(kind == "discard" and value == failed_ticket
                   for kind, value in source_operations) == 1
        assert all(value != failed_ticket for kind, value in source_operations
                   if kind == "acknowledge")
        if no_progress:
            assert len(main_rejections) == 1
            assert budget_error_measurements == [rejected_measurement]
            assert all(value != failed_ticket for kind, value in source_operations
                       if kind == "acknowledge")
            assert not any(ledger.has_pending_requests for ledger in request_ledgers)
            assert not any(
                record[3] is not None and record[3].projection_epoch > failed_ticket.projection_epoch
                for record in guard_records
            )
            snapshot = source.load_source_snapshot("cli:budget-recovery")
            before_by_id = {event["_event_id"]: event for event in before_recovery_events}
            after_by_id = {event["_event_id"]: event for event in snapshot.events}
            assert before_by_id.keys() <= after_by_id.keys()
            assert all(after_by_id[event_id] == event for event_id, event in before_by_id.items())
            assert len(snapshot.events) == len(before_recovery_events) + 1
            assert snapshot.events[-1]["content"] == "[Assistant reply unavailable due to model error.]"
            assert not validate_execution_links(snapshot.events)
            assert "Finalized provider request exceeds" not in _text_content(snapshot.events)
            return

        recovered_records = [record for record in guard_records
                             if not record[0] and record[2] is not None
                             and record[2].fits
                             and record[3] is not None
                             and record[3].projection_epoch > failed_ticket.projection_epoch]
        assert len(recovered_records) == 1
        _, recovered_body, recovered_measurement, recovered_ticket = recovered_records[0]
        assert recovered_measurement is not None and recovered_ticket is not None
        assert recovered_measurement.input_tokens < rejected_measurement.input_tokens
        assert recovered_measurement.fits
        assert recovered_ticket.ticket_id != failed_ticket.ticket_id
        assert recovered_ticket.projection_epoch == failed_ticket.projection_epoch + 1
        assert recovered_ticket.checkpoint_id is not None

        if second_backend_overflow:
            assert not any(value == recovered_ticket for kind, value in source_operations
                           if kind == "acknowledge")
            assert sum(kind == "discard" and value == recovered_ticket
                       for kind, value in source_operations) == 1
        else:
            assert any(value == recovered_ticket for kind, value in source_operations
                       if kind == "acknowledge")
        assert not any(ledger.has_pending_requests for ledger in request_ledgers)
        for index, (kind, value) in enumerate(source_operations):
            if kind == "acknowledge":
                assert source_operations[index - 1][0] == "write"

        failed_issued = next(row for row in issued if row[1] == failed_ticket)
        recovered_issued = next(row for row in issued if row[1] == recovered_ticket)
        _, _, failed_view, failed_eligible = failed_issued
        _, _, recovered_view, _ = recovered_issued
        failed_source_nodes = [node for node in failed_view.nodes if isinstance(node, SourceNode)]
        recovered_source_nodes = [node for node in recovered_view.nodes
                                  if isinstance(node, SourceNode)]
        pending_tail_nodes = [node for node in failed_source_nodes
                              if node.event_id not in failed_eligible]
        pending_tail_ids = {node.event_id for node in pending_tail_nodes}
        replayed_tail_nodes = [node for node in recovered_source_nodes
                               if node.event_id in pending_tail_ids]
        assert replayed_tail_nodes == pending_tail_nodes
        assert len(pending_tail_nodes) >= 2
        assert [node.message.get("role") for node in pending_tail_nodes][-2:] == ["assistant", "tool"]
        current_user_nodes = [node for node in failed_source_nodes
                              if node.message.get("role") == "user"
                              and "Inspect the image and collect three evidence chunks."
                              in _text_content(node.message)]
        assert len(current_user_nodes) == 1
        assert current_user_nodes[0] in recovered_source_nodes
        recovered_checkpoint = next(node for node in recovered_view.nodes
                                    if isinstance(node, CheckpointNode))
        assert recovered_checkpoint.checkpoint_id == recovered_ticket.checkpoint_id
        dispatched_messages = recovered_body.get("messages")
        assert isinstance(dispatched_messages, list)
        assert current_user_nodes[0].message in dispatched_messages
        assert all(node in recovered_source_nodes for node in pending_tail_nodes)
        pending_assistant = pending_tail_nodes[-2].message
        dispatched_assistants = [message for message in dispatched_messages
                                 if isinstance(message, dict)
                                 and message.get("role") == "assistant"
                                 and message.get("tool_calls")]
        assert dispatched_assistants[-1].get("tool_calls") == pending_assistant.get("tool_calls")
        pending_tool = pending_tail_nodes[-1].message
        dispatched_tool = next(message for message in dispatched_messages
                               if isinstance(message, dict)
                               and message.get("role") == "tool"
                               and message.get("tool_call_id") == pending_tool.get("tool_call_id"))
        assert {key: dispatched_tool.get(key) for key in
                ("role", "name", "tool_call_id", "content")} == {
            key: pending_tool.get(key) for key in
            ("role", "name", "tool_call_id", "content")
        }

        snapshot = source.load_source_snapshot("cli:budget-recovery")
        assert not validate_execution_links(snapshot.events)
        assert len({event["_event_id"] for event in snapshot.events}) == len(snapshot.events)
        before_by_id = {event["_event_id"]: event for event in before_recovery_events}
        after_by_id = {event["_event_id"]: event for event in snapshot.events}
        assert before_by_id.keys() <= after_by_id.keys()
        assert all(after_by_id[event_id] == event for event_id, event in before_by_id.items())
        assert "Finalized provider request exceeds" not in _text_content(snapshot.events)

        if finalization:
            final_messages = main_create_calls[-1].get("messages")
            assert isinstance(final_messages, list)
            assert sum(message == build_budget_exhausted_finalization_message()
                       for message in final_messages) == 1
            assert main_create_calls[-1].get("tools") is None
            assert chunk_tool.calls == ["chunk-1", "chunk-2", "chunk-3"]
        if streaming:
            assert capture_hook.content.count("Earlier progress.") == 1
            assert capture_hook.content.count("Recovered after budget pressure.") == (
                0 if second_backend_overflow else 1
            )
            assert capture_hook.reasoning == ["Earlier reasoning."]
            assert streamed.count("Earlier progress.") == 1
            assert streamed.count("Recovered after budget pressure.") == (
                0 if second_backend_overflow else 1
            )
        else:
            assert not capture_hook.content
            assert not capture_hook.reasoning
            assert not streamed
    finally:
        await loop.aclose()


@pytest.mark.parametrize("local", [False, True], ids=["root", "local"])
async def test_forced_recovery_compacts_all_eligible_prefix_after_logical_fit(store, local: bool):
    async with acquire_execution(store, "test:compact", run_id="forced-recovery") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="forced-recovery")
        state = pressure_state(writer, chunks=8, local=local)

        def count(messages, tools, model):
            is_auxiliary = messages[0]["content"].startswith("Summarize historical evidence")
            if is_auxiliary:
                return 100 + str(messages).count("chunk-") * 2_500, "fixture-auxiliary"
            return 4_000, "fixture-low-pressure"

        state.config.provider.estimate_prompt_tokens = count
        governor = ContextGovernor()
        before = owner.read_snapshot()
        view = state.source.transcript
        messages, _ = await governor.prepare_request(
            state,
            view.messages(),
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages,
            source_view=view,
        )
        assert messages == state.messages == state.source_view.messages()
        assert state.source_compaction.attempts == []
        assert state.config.provider.calls == []
        candidate = governor.capture_source_compaction(
            state,
            state.source_view,
            tool_definitions=None,
        ).candidate
        assert candidate.boundaries

        budget = governor.input_budget(state.config)
        measurement = _local_overflow_measurement(budget)
        recovered = await governor.recover_finalized_request(
            state,
            measurement=measurement,
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages,
            source_view=state.source_view,
        )

        assert recovered.recovered
        recovered_messages = recovered.prepared_messages
        assert recovered_messages is not None
        metrics = state.source_compaction.attempts[-1]
        assert metrics.calls > 1 and metrics.calls <= 4
        assert metrics.boundary == candidate.boundaries[-1]
        assert metrics.after_tokens <= budget * 0.6
        assert len(state.config.provider.calls) == metrics.calls
        assert state.source.projection_epoch == 1
        assert not state.source.has_pending_requests
        assert state.messages == state.source_view.messages() == recovered_messages
        assert recovered_messages.count({"role": "user", "content": "current instruction"}) == 1
        if local:
            assert owner.read_snapshot().checkpoint == before.checkpoint
            assert state.source_compaction.local_boundary == candidate.boundaries[-1]
            assert state.source_compaction.local_checkpoint is not None
        else:
            assert owner.read_snapshot().checkpoint is not None


@pytest.mark.parametrize(
    "case",
    ["no-measurement", "fitting", "inconsistent", "nonpositive-limit", "boolean",
     "no-source", "no-compactor", "no-eligible-prefix", "nonpositive-budget"],
)
async def test_unusable_final_measurement_or_source_never_spends_auxiliary_calls(store, case: str):
    async with acquire_execution(store, "test:compact", run_id="unusable-recovery") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="unusable-recovery")
        state = pressure_state(writer, chunks=2)
        state.config.provider.estimate_prompt_tokens = lambda *args: (100, "fixture-low-pressure")
        governor = ContextGovernor()
        view = state.source.transcript
        await governor.prepare_request(
            state,
            view.messages(),
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages,
            source_view=view,
        )
        budget = governor.input_budget(state.config)
        valid = TransportMeasurement(
            identity=TransportIdentity("fallback-leaf", "chat", _MODEL),
            input_tokens=budget + 1,
            completion_reserve_tokens=256,
            safety_buffer_tokens=1_024,
            total_limit_tokens=budget + 1_280,
            input_limit_tokens=budget,
            source="utf8-json-bytes",
        )
        measurement: TransportMeasurement | None = valid
        if case == "no-measurement":
            measurement = None
        elif case == "fitting":
            measurement = TransportMeasurement(
                identity=valid.identity,
                input_tokens=budget,
                completion_reserve_tokens=256,
                safety_buffer_tokens=1_024,
                total_limit_tokens=budget + 1_280,
                input_limit_tokens=budget,
                source=valid.source,
            )
        elif case == "inconsistent":
            measurement = TransportMeasurement(
                identity=valid.identity,
                input_tokens=budget + 1,
                completion_reserve_tokens=256,
                safety_buffer_tokens=1_024,
                total_limit_tokens=budget + 1_281,
                input_limit_tokens=budget,
                source=valid.source,
            )
        elif case == "nonpositive-limit":
            measurement = TransportMeasurement(
                identity=valid.identity,
                input_tokens=1,
                completion_reserve_tokens=0,
                safety_buffer_tokens=1_024,
                total_limit_tokens=1_024,
                input_limit_tokens=0,
                source=valid.source,
            )
        elif case == "boolean":
            measurement = TransportMeasurement(
                identity=valid.identity,
                input_tokens=True,
                completion_reserve_tokens=256,
                safety_buffer_tokens=1_024,
                total_limit_tokens=budget + 1_280,
                input_limit_tokens=budget,
                source=valid.source,
            )
        elif case == "no-source":
            state.source = None
        elif case == "no-compactor":
            state.source_compaction = None
        elif case == "no-eligible-prefix":
            state.source = SourceRequestLedger(
                state.source.transcript,
                initial_eligible_ids=frozenset(),
            )
            state.source_view = state.source.transcript
            state.messages = state.source_view.messages()
        elif case == "nonpositive-budget":
            state.config.context_window_tokens = 0

        epoch = state.source.projection_epoch if state.source is not None else None
        recovered = await governor.recover_finalized_request(
            state,
            measurement=measurement,
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages
            if state.source_compaction is not None else None,
            source_view=state.source_view,
        )

        assert not recovered.recovered
        assert recovered.unavailable_reason == "no_eligible_evidence"
        assert state.config.provider.calls == []
        assert state.source_compaction is None or state.source_compaction.attempts == []
        if state.source is not None:
            assert state.source.projection_epoch == epoch
            assert not state.source.has_pending_requests


@pytest.mark.parametrize(
    "response",
    [
        LLMResponse(content=""),
        LLMResponse(content="partial", finish_reason="length"),
        LLMResponse(content="invalid", finish_reason="error"),
        LLMResponse(
            content="tool request",
            tool_calls=[ToolCallRequest(id="summary-tool", name="write_file", arguments={})],
        ),
        TimeoutError("bounded summary timeout"),
    ],
    ids=["empty", "non-stop", "error", "tool-call", "timeout"],
)
async def test_unusable_summary_is_terminal_without_checkpoint_publication(store, response):
    async with acquire_execution(store, "test:compact", run_id="bad-summary") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="bad-summary")
        state = pressure_state(writer, chunks=3)
        state.config.provider.response = response
        state.config.provider.estimate_prompt_tokens = lambda *args: (100, "fixture-low-pressure")
        governor = ContextGovernor()
        view = state.source.transcript
        await governor.prepare_request(
            state,
            view.messages(),
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages,
            source_view=view,
        )
        before = owner.read_snapshot()
        active = state.source.transcript
        budget = governor.input_budget(state.config)
        measurement = _local_overflow_measurement(budget)

        recovered = await governor.recover_finalized_request(
            state,
            measurement=measurement,
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages,
            source_view=state.source_view,
        )

        assert not recovered.recovered
        if isinstance(response, TimeoutError):
            expected_reason = "summary_provider_error"
        elif response.has_tool_calls:
            expected_reason = "summary_tool_calls"
        elif response.finish_reason == "length":
            expected_reason = "summary_truncated"
        elif response.finish_reason == "error":
            expected_reason = "summary_provider_error"
        else:
            expected_reason = "summary_empty"
        assert recovered.unavailable_reason == expected_reason
        assert 1 <= len(state.config.provider.calls) <= 4
        assert state.source_compaction.attempts[-1].status == "failed"
        assert state.source_compaction.attempts[-1].calls == len(
            state.config.provider.calls,
        )
        assert owner.read_snapshot() == before
        assert state.source.transcript == active
        assert state.source.projection_epoch == 0
        assert not state.source.has_pending_requests


async def test_incomplete_real_sdk_summary_cannot_publish_source_checkpoint(store):
    from nanobot.providers.conversation_state import ProviderConversationStateController

    async with acquire_execution(store, "test:compact", run_id="incomplete-sdk-summary") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="incomplete-sdk-summary")
        state = pressure_state(writer, chunks=3, local=True)
        model = _MODEL
        generation = GenerationSettings(max_tokens=256, temperature=0.0)
        provider = OpenAICompatProvider(
            api_key="fixture-credential",
            api_base="http://manager.test/v1",
            default_model=model,
            provider_name="local_manager",
            spec=find_by_name("local_manager"),
            timeout=60.0,
        )
        provider.generation = generation
        state.config.provider = provider
        state.config.model = model
        state.config.max_tokens = generation.max_tokens
        state.config.generation = generation
        state.source_compaction.runtime = replace(
            state.source_compaction.runtime,
            provider=provider,
            model=model,
            generation=generation,
            context_window_tokens=12_000,
        )
        state.conversation = ProviderConversationStateController(
            provider=provider,
            model=model,
            messages=state.source_compaction.raw_messages,
            session_id=owner.key,
        )

        payload = (
            b'data: {"id":"probe","object":"chat.completion.chunk","created":1,'
            b'"model":"Qwen3.8-27B-UD-Q6_K_XL","choices":[{"index":0,"delta":'
            b'{"content":"incomplete historical fact"},"finish_reason":null}]}\n\n'
        )

        class IncompleteStream(httpx.AsyncByteStream):
            def __init__(self) -> None:
                self.closed = False

            async def __aiter__(self):
                yield payload

            async def aclose(self) -> None:
                self.closed = True

        stream = IncompleteStream()
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
                request=request,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            provider._client = AsyncOpenAI(
                api_key="fixture-credential",
                base_url="http://manager.test/v1",
                http_client=http_client,
                max_retries=0,
            )
            governor = ContextGovernor()
            await governor.prepare_request(
                state,
                state.source.transcript.messages(),
                tool_definitions=None,
                transcript=state.source_compaction.raw_messages,
                source_view=state.source.transcript,
            )
            before = owner.read_snapshot()
            active = state.source.transcript
            measurement = _local_overflow_measurement(governor.input_budget(state.config))

            recovered = await governor.recover_finalized_request(
                state,
                measurement=measurement,
                tool_definitions=None,
                transcript=state.source_compaction.raw_messages,
                source_view=state.source_view,
            )

        assert not recovered.recovered
        assert recovered.unavailable_reason == "summary_provider_error"
        assert len(requests) == 1
        assert state.source_compaction.attempts[-1].calls == 1
        assert owner.read_snapshot() == before
        assert state.source.transcript == active
        assert state.source.projection_epoch == 0
        assert not state.source.has_pending_requests
        assert stream.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra_body", "expect_prefix", "expected_tools"),
    [
        (
            {"tools": [{"type": "code_interpreter", "container": {"type": "auto"}}]},
            False,
            None,
        ),
        (
            {"tools": [{"type": "future_tool", "name": "unknown"}]},
            False,
            None,
        ),
        (
            {
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "configured_function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }],
                "tool_choice": "auto",
            },
            True,
            [{
                "type": "function",
                "function": {
                    "name": "configured_function",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
        ),
    ],
    ids=["configured-hosted-tool-falls-back", "unknown-tool-falls-back", "function-tool-kept"],
)
async def test_prefix_gate_uses_configured_effective_tool_envelope(
    store,
    extra_body: dict[str, object],
    expect_prefix: bool,
    expected_tools: list[dict[str, object]] | None,
) -> None:
    model = "Qwen3.8-27B-UD-Q6_K_XL"
    generation = GenerationSettings(temperature=0.2, max_tokens=128, reasoning_effort="none")
    provider = OpenAICompatProvider(
        api_key="fixture-credential",
        api_base="http://manager.test/v1",
        default_model=model,
        provider_name="local_manager",
        spec=find_by_name("local_manager"),
        extra_body=extra_body,
    )
    provider.generation = generation

    async with acquire_execution(store, "test:compact", run_id="effective-tools") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="effective-tools")
        writer.append_message({"role": "user", "content": "Historical objective."})
        writer.append_message({"role": "assistant", "content": "Historical answer."})
        writer.append_message({"role": "user", "content": "Current objective."})
        writer.append_message({"role": "assistant", "content": "PENDING_TAIL"})
        nodes = source_nodes(owner.read_snapshot().events)
        active = SourceTranscript((
            RuntimeNode("test-system", {"role": "system", "content": "System policy."}),
            *nodes,
        ))
        raw_messages = active.messages()
        ledger = SourceRequestLedger(
            active,
            initial_eligible_ids=frozenset(node.event_id for node in nodes[:2]),
        )
        runtime = LLMRuntime(provider, model, generation, 12_000)
        state = ModelRequestState(
            ContextGovernanceConfig(
                provider=provider,
                model=model,
                tools=ToolRegistry(),
                workspace=None,
                session_key=owner.key,
                max_tool_result_chars=16_000,
                context_window_tokens=12_000,
                max_tokens=generation.max_tokens,
                generation=generation,
                preserve_source=True,
            ),
            ProviderConversationStateController(
                provider=provider,
                model=model,
                messages=raw_messages,
                session_id=owner.key,
            ),
            source=ledger,
            source_compaction=SourceCompactionState(
                writer,
                runtime,
                raw_messages,
                None,
                overall_timeout_seconds=5.0,
            ),
        )
        before = owner.read_snapshot()
        captured: list[dict[str, object]] = []

        def sse_chunk(content: str | None, finish_reason: str | None) -> dict[str, object]:
            return {
                "id": "chatcmpl-effective-tools",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": finish_reason,
                }],
            }

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            captured.append(body)
            chunks = [
                sse_chunk("Historical summary.", None),
                sse_chunk(None, "stop"),
            ]
            payload = "".join(
                "data: " + json.dumps(chunk) + "\n\n" for chunk in chunks
            ) + "data: [DONE]\n\n"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=payload.encode(),
                request=request,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            provider._client = AsyncOpenAI(
                api_key="fixture-credential",
                base_url="http://manager.test/v1",
                http_client=http_client,
                max_retries=0,
            )
            result_view = await ContextGovernor().compact_source_request(
                state,
                state.source.transcript,
                tool_definitions=None,
                force=True,
                stream=True,
            )

        metrics = state.source_compaction.attempts[-1]
        assert metrics.status == "succeeded"
        assert metrics.calls == 1 and len(captured) == 1
        assert metrics.prefix_mode == ("historical_prefix" if expect_prefix else "canonical")
        if expect_prefix:
            assert metrics.prefix_fallback_reason is None
            assert metrics.converted_prefix_match is True
            assert captured[0]["tools"] == expected_tools
            assert captured[0]["tool_choice"] == "auto"
        else:
            assert metrics.prefix_fallback_reason == "unsupported_tool_envelope"
            assert metrics.converted_prefix_match is None
            assert "tools" not in captured[0]
            assert "tool_choice" not in captured[0]
        after = owner.read_snapshot()
        assert after.events == before.events
        assert after.checkpoint["boundary"] == metrics.boundary
        assert after.checkpoint["summary"] == "Historical summary."
        assert result_view == state.source_view
        assert not state.source.has_pending_requests


@pytest.mark.asyncio
async def test_from_config_run_keeps_admitted_source_deadlines_and_generation(
    engine_config: Config,
) -> None:
    idle_seconds = 37.0
    overall_seconds = 83.0
    engine_config.agents.defaults.source_compaction.idle_timeout_seconds = idle_seconds
    engine_config.agents.defaults.source_compaction.overall_timeout_seconds = overall_seconds
    provider = ScriptedProvider([LLMResponse(content="done")])
    initial_generation = GenerationSettings(
        temperature=0.25,
        max_tokens=73,
        reasoning_effort="low",
    )
    provider.generation = initial_generation
    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=ToolRegistry(),
        provider=provider,
        model=_MODEL,
        context_window_tokens=12_000,
    )
    source_key = "cli:source-config-capture"
    _upgrade_source_session(loop, source_key)
    admitted_runtime = loop.llm_runtime()
    assert admitted_runtime.generation == initial_generation
    captured_states: list[ModelRequestState] = []
    make_request_state = loop.runner._make_request_state

    def capture_admitted_state(spec, messages, compaction):
        state = make_request_state(spec, messages, compaction)
        if state.source_compaction is not None:
            captured_states.append(state)
        return state

    loop.runner._make_request_state = capture_admitted_state
    engine_config.agents.defaults.source_compaction.idle_timeout_seconds = 2.0
    engine_config.agents.defaults.source_compaction.overall_timeout_seconds = 3.0
    provider.generation = GenerationSettings(
        temperature=0.9,
        max_tokens=999,
        reasoning_effort="none",
    )
    try:
        result = await loop.process_direct(
            "Run with the captured source policy.",
            session_key=source_key,
            runtime=admitted_runtime,
        )
    finally:
        await loop.aclose()

    assert result is not None and result.content == "done"
    assert len(captured_states) == 1
    source_state = captured_states[0].source_compaction
    assert source_state is not None
    assert source_state.idle_timeout_seconds == idle_seconds
    assert source_state.overall_timeout_seconds == overall_seconds
    assert source_state.runtime.generation == initial_generation


async def test_cancelled_auxiliary_summary_propagates_without_publication(store):
    async with acquire_execution(store, "test:compact", run_id="cancel-summary") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="cancel-summary")
        state = pressure_state(writer, chunks=3)
        state.config.provider.response = asyncio.CancelledError()
        state.config.provider.estimate_prompt_tokens = lambda *args: (100, "fixture-low-pressure")
        governor = ContextGovernor()
        view = state.source.transcript
        await governor.prepare_request(
            state,
            view.messages(),
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages,
            source_view=view,
        )
        before = owner.read_snapshot()
        active = state.source.transcript
        budget = governor.input_budget(state.config)
        measurement = _local_overflow_measurement(budget)

        with pytest.raises(asyncio.CancelledError):
            await governor.recover_finalized_request(
                state,
                measurement=measurement,
                tool_definitions=None,
                transcript=state.source_compaction.raw_messages,
                source_view=state.source_view,
            )

        assert len(state.config.provider.calls) == 1
        assert state.source_compaction.attempts[-1].status == "cancelled"
        assert owner.read_snapshot() == before
        assert state.source.transcript == active
        assert state.source.projection_epoch == 0
        assert not state.source.has_pending_requests


async def test_source_conflict_during_forced_summary_propagates(store):
    async with acquire_execution(store, "test:compact", run_id="source-conflict") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="source-conflict")
        state = pressure_state(writer, chunks=3)
        state.config.provider.estimate_prompt_tokens = lambda *args: (100, "fixture-low-pressure")
        governor = ContextGovernor()
        view = state.source.transcript
        await governor.prepare_request(
            state,
            view.messages(),
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages,
            source_view=view,
        )
        before = owner.read_snapshot()
        active = state.source.transcript
        provider_stream = state.config.provider.chat_stream_with_context

        async def update_during_summary(*, provider_context, **kwargs):
            if provider_context.on_transport_attempt is not None:
                provider_context.on_transport_attempt("fixture-stream")
            store.source.update_metadata("test:compact", {"title": "concurrent update"})
            return await provider_stream(provider_context=provider_context, **kwargs)

        state.config.provider.chat_stream_with_context = update_during_summary
        budget = governor.input_budget(state.config)
        measurement = _local_overflow_measurement(budget)

        with pytest.raises((SourceValidationError, SessionWriteConflict)):
            await governor.recover_finalized_request(
                state,
                measurement=measurement,
                tool_definitions=None,
                transcript=state.source_compaction.raw_messages,
                source_view=state.source_view,
            )

        after = owner.read_snapshot()
        assert after.events == before.events
        assert after.checkpoint == before.checkpoint
        assert state.source.transcript == active
        assert state.source.projection_epoch == 0
        assert not state.source.has_pending_requests


@pytest.mark.parametrize(
    ("failure_phase", "published"),
    [("started", False), ("succeeded", True)],
    ids=["before-publication", "after-publication"],
)
async def test_compaction_callback_cancellation_preserves_publication_boundary(
    store, failure_phase: str, published: bool,
):
    async with acquire_execution(store, "test:compact", run_id="callback-cancel") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="callback-cancel")
        state = pressure_state(writer, chunks=3)
        state.config.provider.estimate_prompt_tokens = lambda *args: (100, "fixture-low-pressure")
        governor = ContextGovernor()
        view = state.source.transcript
        await governor.prepare_request(
            state,
            view.messages(),
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages,
            source_view=view,
        )
        before = owner.read_snapshot()
        active = state.source.transcript
        event_phases: list[str] = []

        async def publish(event: AgentEvent) -> None:
            if isinstance(event, ContextCompactionEvent):
                event_phases.append(event.phase)
                if event.phase == failure_phase:
                    raise asyncio.CancelledError("injected compaction callback cancellation")

        state.events = EventSink(publish=publish)
        budget = governor.input_budget(state.config)
        measurement = _local_overflow_measurement(budget)

        with pytest.raises(asyncio.CancelledError, match="injected compaction callback"):
            await governor.recover_finalized_request(
                state,
                measurement=measurement,
                tool_definitions=None,
                transcript=state.source_compaction.raw_messages,
                source_view=state.source_view,
            )

        after = owner.read_snapshot()
        assert not state.source.has_pending_requests
        assert event_phases[0] == "started"
        if published:
            assert event_phases == ["started", "succeeded"]
            assert after.events == before.events
            assert after.state.revision == before.state.revision + 1
            assert after.checkpoint is not None
            assert state.source.projection_epoch == 1
            assert state.source_compaction.raw_messages == state.source.transcript.messages()
            assert state.source_compaction.attempts[-1].status == "succeeded"
        else:
            assert event_phases == ["started", "cancelled"]
            assert after == before
            assert state.source.transcript == active
            assert state.source.projection_epoch == 0
            assert state.source_compaction.attempts[-1].status == "cancelled"


async def test_invalid_compaction_proof_propagates_without_publication(store, monkeypatch):
    async with acquire_execution(store, "test:compact", run_id="invalid-proof") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="invalid-proof")
        state = pressure_state(writer, chunks=3)
        state.config.provider.estimate_prompt_tokens = lambda *args: (100, "fixture-low-pressure")
        governor = ContextGovernor()
        view = state.source.transcript
        await governor.prepare_request(
            state,
            view.messages(),
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages,
            source_view=view,
        )
        before = owner.read_snapshot()
        active = state.source.transcript
        original_publish = ContextGovernor.publish_source_compaction

        def reject_incomplete_proof(governor, request_state, attempt, **kwargs):
            proof = kwargs.get("proof")
            assert isinstance(proof, SummaryDependencyProof)
            kwargs["proof"] = replace(proof, chunks=())
            return original_publish(governor, request_state, attempt, **kwargs)

        monkeypatch.setattr(
            ContextGovernor,
            "publish_source_compaction",
            reject_incomplete_proof,
        )
        budget = governor.input_budget(state.config)
        measurement = _local_overflow_measurement(budget)

        with pytest.raises(SourceValidationError, match="proof does not match"):
            await governor.recover_finalized_request(
                state,
                measurement=measurement,
                tool_definitions=None,
                transcript=state.source_compaction.raw_messages,
                source_view=state.source_view,
            )

        assert len(state.config.provider.calls) == 1
        assert state.source_compaction.attempts[-1].status == "failed"
        assert owner.read_snapshot() == before
        assert state.source.transcript == active
        assert state.source.projection_epoch == 0
        assert not state.source.has_pending_requests


async def test_auxiliary_actual_preflight_overflow_aborts_recovery(store):
    async with acquire_execution(store, "test:compact", run_id="aux-overflow") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="aux-overflow")
        state = pressure_state(writer, chunks=0)
        large_evidence = {"role": "assistant", "content": "evidence-" + "z" * 30_000}
        large_event = writer.append_message(large_evidence)
        state.source.append_source(large_event["_event_id"], large_evidence)
        issue = state.source.issue(state.source.transcript)
        state.source.acknowledge(issue)
        state.source_compaction.raw_messages[:] = state.source.transcript.messages()

        provider = OpenAICompatProvider(
            api_key="fixture",
            default_model=_MODEL,
            spec=find_by_name("local_manager"),
            provider_name="local_manager",
        )
        generation = GenerationSettings(
            max_tokens=2_048,
            temperature=0,
            reasoning_effort="none",
        )
        provider.generation = generation
        provider.estimate_prompt_tokens = lambda messages, tools, model: (
            (2_500 if messages[0].get("content", "").startswith(
                "Summarize historical evidence") else 2_000),
            "fixture-low-pressure",
        )
        sdk_create = AsyncMock(return_value=_sdk_response(content="must not dispatch"))
        provider._ensure_client = AsyncMock(return_value=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=sdk_create)),
        ))
        state.source_compaction.runtime = replace(
            state.source_compaction.runtime,
            provider=provider,
            model=_MODEL,
            generation=generation,
            context_window_tokens=16_384,
        )
        state.config.context_window_tokens = 16_384
        state.config.model = _MODEL
        state.config.provider.estimate_prompt_tokens = lambda *args: (2_000, "fixture-low-pressure")
        state.conversation.replace_transcript(state.source_compaction.raw_messages)
        governor = ContextGovernor()
        prepared, _ = await governor.prepare_request(
            state,
            state.source.transcript.messages(),
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages,
            source_view=state.source.transcript,
        )
        assert prepared == state.messages
        assert state.source_compaction.attempts == []
        before = owner.read_snapshot()
        active = state.source.transcript
        budget = governor.input_budget(state.config)
        rejected_main = _local_overflow_measurement(budget)
        recovered = await governor.recover_finalized_request(
            state,
            measurement=rejected_main,
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages,
            source_view=state.source_view,
        )

        assert not recovered.recovered
        assert recovered.unavailable_reason == "indivisible_source_group_exceeds_auxiliary_budget"
        assert sdk_create.await_count == 0
        assert state.source_compaction.attempts[-1].status == "failed"
        assert state.source_compaction.attempts[-1].calls == 0
        assert owner.read_snapshot() == before
        assert state.source.transcript == active
        assert state.source.projection_epoch == 0
        assert not state.source.has_pending_requests


class _UsageAnchorTicketLedger:
    def __init__(self) -> None:
        self.projection_epoch = 7
        self.acknowledged: list[object] = []
        self.discarded: list[object] = []

    def acknowledge(self, ticket: object) -> None:
        self.acknowledged.append(ticket)

    def discard(self, ticket: object) -> None:
        self.discarded.append(ticket)


def _usage_anchor_fixture(*, content: str | None = "usable", finish_reason: str = "stop"):
    provider = OpenAICompatProvider(
        api_key="fixture",
        default_model=_MODEL,
        spec=find_by_name("local_manager"),
        provider_name="local_manager",
    )
    receipt = LocalRequestUsageReceipt(
        provider_scope="private-client-scope",
        envelope_binding="private-envelope",
        model=_MODEL,
        limits_binding="private-limits",
        messages=({"role": "user", "content": "request"},),
    )
    response = LLMResponse(
        content=content,
        finish_reason=finish_reason,
        tool_calls=(
            [ToolCallRequest(id="call-1", name="make_text_chunk", arguments={})]
            if finish_reason in {"tool_calls", "function_call"}
            else []
        ),
        usage=LLMUsage.reported(
            input_tokens=900,
            output_tokens=4,
            cache_read_tokens=200,
        ),
        local_usage_receipt=receipt,
    )
    state = SimpleNamespace(
        source=_UsageAnchorTicketLedger(),
        messages=[{"role": "user", "content": "request"}],
        tool_definitions=None,
        usage=None,
        local_usage_anchor=LocalUsageAnchor(
            receipt=receipt,
            prompt_tokens=800,
            projection_epoch=7,
        ),
    )
    spec = SimpleNamespace(
        source_writer=None,
        runtime=SimpleNamespace(provider=provider),
    )
    return object.__new__(AgentRunner), spec, state, response


@pytest.mark.parametrize("case", ["malformed", "discarded_ticket", "no_tools_calls"])
def test_local_usage_is_recorded_but_unusable_or_discarded_responses_do_not_anchor(
    case: str,
) -> None:
    runner, spec, state, response = _usage_anchor_fixture(
        content="usable" if case != "discarded_ticket" else "provider error",
        finish_reason="error" if case == "discarded_ticket" else (
            "tool_calls" if case == "no_tools_calls" else "stop"
        ),
    )
    ticket = object()
    recorded = runner._record_and_settle(
        spec,
        state,
        ticket,
        response,
        malformed=case == "malformed",
        no_tools=case == "no_tools_calls",
    )
    usage = runner._record_request_usage(
        spec,
        state,
        response,
        recorded_response=recorded,
    )

    assert usage.input_tokens == 900
    assert usage.cache_read_tokens == 200
    assert state.usage is usage
    assert state.local_usage_anchor is None
    assert state.source.discarded == [ticket]


def test_local_usage_anchor_is_installed_after_usable_response_recording() -> None:
    runner, spec, state, response = _usage_anchor_fixture()
    recorded = runner._record_and_settle(spec, state, None, response)

    runner._record_request_usage(
        spec,
        state,
        response,
        recorded_response=recorded,
    )

    assert recorded.local_usage_anchor_eligible is True
    assert state.local_usage_anchor is not None
    assert state.local_usage_anchor.prompt_tokens == 900
    assert state.local_usage_anchor.projection_epoch == 7


@pytest.mark.asyncio
async def test_sdk_usage_anchor_waits_for_source_ack_and_floors_next_request(
    engine_config: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = ToolRegistry()
    chunk_tool = _TextChunkTool(result_size=5)
    tools.register(chunk_tool)
    provider = OpenAICompatProvider(
        api_key="fixture-credential",
        api_base="http://manager.test/v1",
        default_model=_MODEL,
        provider_name="local_manager",
        spec=find_by_name("local_manager"),
    )
    generation = GenerationSettings(
        temperature=0.0,
        max_tokens=256,
        reasoning_effort="none",
    )
    provider.generation = generation
    loop = AgentLoop.from_config(
        engine_config,
        tool_registry=tools,
        provider=provider,
        model=_MODEL,
        context_window_tokens=_CONTEXT_WINDOW,
    )
    loop.max_iterations = 4
    session_key = "cli:source-usage-anchor-sdk"
    _upgrade_source_session(loop, session_key)
    captured_bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert isinstance(body, dict)
        captured_bodies.append(body)
        request_number = len(captured_bodies)
        if request_number == 1:
            response_body = {
                "id": "chatcmpl-usage-anchor-tool",
                "object": "chat.completion",
                "created": 1,
                "model": _MODEL,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [_tool_call(
                            "anchor-tool-call",
                            "make_text_chunk",
                            {"label": "anchor"},
                        )],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {
                    "prompt_tokens": 10_000,
                    "completion_tokens": 4,
                    "total_tokens": 10_004,
                    "prompt_tokens_details": {"cached_tokens": 3_000},
                },
            }
        elif request_number == 2:
            response_body = {
                "id": "chatcmpl-usage-anchor-final",
                "object": "chat.completion",
                "created": 2,
                "model": _MODEL,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Finished."},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 10_100,
                    "completion_tokens": 2,
                    "total_tokens": 10_102,
                },
            }
        else:
            raise AssertionError("fixture should issue only the tool call and final request")
        return httpx.Response(200, json=response_body, request=request)

    captured_states: list[ModelRequestState] = []
    original_make_request_state = loop.runner._make_request_state

    def capture_request_state(spec, messages, compaction):
        state = original_make_request_state(spec, messages, compaction)
        if state.source is not None:
            captured_states.append(state)
        return state

    monkeypatch.setattr(loop.runner, "_make_request_state", capture_request_state)
    measurements: list[tuple[
        dict[str, object], ProviderCallContext | None, TransportMeasurement,
    ]] = []
    original_measure = provider._measure_local_body

    def capture_measurement(
        body: dict[str, object],
        *,
        transport: str,
        model: str | None,
        provider_context: ProviderCallContext | None = None,
        completion_reserve_tokens: int | None = None,
    ) -> TransportMeasurement:
        measurement = original_measure(
            body,
            transport=transport,
            model=model,
            provider_context=provider_context,
            completion_reserve_tokens=completion_reserve_tokens,
        )
        measurements.append((deepcopy(body), provider_context, measurement))
        return measurement

    monkeypatch.setattr(provider, "_measure_local_body", capture_measurement)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="http://manager.test/v1",
            http_client=http_client,
            max_retries=0,
        )
        try:
            result = await loop.process_direct(
                "Use the local fixture tool, then finish.",
                session_key=session_key,
                runtime=loop.llm_runtime(),
            )
        finally:
            await loop.aclose()

    assert result is not None and result.content == "Finished."
    assert chunk_tool.calls == ["anchor"]
    assert len(captured_bodies) == 2
    next_messages = captured_bodies[1].get("messages")
    assert isinstance(next_messages, list)
    assert any(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("tool_calls"), list)
        for message in next_messages
    )
    assert any(
        isinstance(message, dict) and message.get("role") == "tool"
        for message in next_messages
    )
    assert len(captured_states) == 1
    source_ledger = captured_states[0].source
    assert source_ledger is not None
    assert not source_ledger.has_pending_requests
    tool_call_node = next(
        node for node in source_ledger.transcript.nodes
        if isinstance(node, SourceNode)
        and node.message.get("role") == "assistant"
        and isinstance(node.message.get("tool_calls"), list)
    )
    assert tool_call_node.event_id in source_ledger.eligible_ids

    anchored = [
        (body, context, measurement)
        for body, context, measurement in measurements
        if context is not None
        and context.local_usage_anchor is not None
        and body.get("messages") == next_messages
    ]
    assert anchored
    anchor_observations = [
        (
            context.local_usage_anchor.prompt_tokens
            if isinstance(context.local_usage_anchor, LocalUsageAnchor)
            else None,
            measurement.source,
            measurement.input_tokens,
        )
        for _body, context, measurement in anchored
    ]
    assert all(
        isinstance(context.local_usage_anchor, LocalUsageAnchor)
        and context.local_usage_anchor.prompt_tokens == 10_000
        and measurement.source == "reported_anchor_plus_estimated_delta"
        and measurement.input_tokens > context.local_usage_anchor.prompt_tokens
        for _body, context, measurement in anchored
    ), anchor_observations


@pytest.mark.asyncio
async def test_source_failure_diagnostics_keep_safe_budget_and_usage_only(store) -> None:
    from loguru import logger

    credential_marker = "CREDENTIAL_PRIVATE_MARKER"
    image_marker = "IMAGE_PRIVATE_MARKER"
    reasoning_marker = "REASONING_PRIVATE_MARKER"
    source_marker = "SOURCE_PRIVATE_MARKER"
    model = _MODEL
    generation = GenerationSettings(
        temperature=0.0,
        max_tokens=128,
        reasoning_effort="none",
    )
    provider = OpenAICompatProvider(
        api_key=credential_marker,
        api_base="http://manager.test/v1",
        default_model=model,
        provider_name="local_manager",
        spec=find_by_name("local_manager"),
        extra_body={"metadata": {"image_payload": image_marker}},
    )
    runtime = LLMRuntime(provider, model, generation, 12_000)
    logs: list[str] = []

    async with acquire_execution(store, "test:compact", run_id="diagnostics") as owner:
        writer = SourceTurnWriter(store.source, owner, run_id="diagnostics")
        writer.append_message({
            "role": "user",
            "content": f"Historical evidence {source_marker}; image marker {image_marker}.",
        })
        writer.append_message({"role": "assistant", "content": "Prior answer."})
        writer.append_message({"role": "user", "content": "Current question."})
        writer.append_message({"role": "assistant", "content": "PENDING_PRIVATE_TAIL"})
        nodes = source_nodes(owner.read_snapshot().events)
        active = SourceTranscript((
            RuntimeNode("diagnostic-system", {"role": "system", "content": "System policy."}),
            *nodes,
        ))
        raw_messages = active.messages()
        source_ledger = SourceRequestLedger(
            active,
            initial_eligible_ids=frozenset(node.event_id for node in nodes[:2]),
        )
        state = ModelRequestState(
            ContextGovernanceConfig(
                provider=provider,
                model=model,
                tools=ToolRegistry(),
                workspace=None,
                session_key=owner.key,
                max_tool_result_chars=16_000,
                context_window_tokens=12_000,
                max_tokens=generation.max_tokens,
                generation=generation,
                preserve_source=True,
            ),
            ProviderConversationStateController(
                provider=provider,
                model=model,
                messages=raw_messages,
                session_id=owner.key,
            ),
            source=source_ledger,
            source_compaction=SourceCompactionState(
                writer,
                runtime,
                raw_messages,
                None,
                overall_timeout_seconds=5.0,
            ),
        )
        state.messages = deepcopy(raw_messages)
        state.source_view = active
        state.tool_definitions = None
        before = owner.read_snapshot()
        captured_bodies: list[dict[str, object]] = []
        sink_id = logger.add(logs.append, level="DEBUG", format="{message}")

        def sse_chunk(
            delta: dict[str, object],
            finish_reason: str | None,
        ) -> dict[str, object]:
            return {
                "id": "chatcmpl-diagnostic-failure",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }],
            }

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert isinstance(body, dict)
            captured_bodies.append(body)
            chunks = [
                sse_chunk({"reasoning_content": reasoning_marker}, None),
                sse_chunk({}, "stop"),
                {
                    "id": "chatcmpl-diagnostic-failure",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 71,
                        "completion_tokens": 0,
                        "total_tokens": 71,
                        "prompt_tokens_details": {"cached_tokens": 23},
                    },
                },
            ]
            payload = "".join(
                "data: " + json.dumps(chunk) + "\n\n" for chunk in chunks
            ) + "data: [DONE]\n\n"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=payload.encode(),
                request=request,
            )

        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
                provider._client = AsyncOpenAI(
                    api_key=credential_marker,
                    base_url="http://manager.test/v1",
                    http_client=http_client,
                    max_retries=0,
                )
                outcome = await ContextGovernor().recover_finalized_request(
                    state,
                    measurement=None,
                    backend_overflow=BackendContextOverflow(
                        status_code=400,
                        code=400,
                        n_prompt_tokens=227_502,
                        n_ctx=165_888,
                    ),
                    tool_definitions=None,
                    transcript=raw_messages,
                    source_view=active,
                    stream=True,
                )
        finally:
            logger.remove(sink_id)

        metrics = state.source_compaction.attempts[-1]
        assert outcome.root_reason == "backend_context_overflow"
        assert outcome.unavailable_reason == "summary_empty"
        assert metrics.root_reason == outcome.root_reason
        assert metrics.failure_reason == outcome.unavailable_reason
        assert metrics.calls == 1
        assert metrics.usage is not None
        assert metrics.usage.input_tokens == 71
        assert metrics.usage.cache_read_tokens == 23
        assert metrics.selected_capacity is not None
        assert metrics.completion_reserve == generation.max_tokens
        assert metrics.safety_buffer is not None
        assert metrics.payload_bytes is not None and metrics.payload_bytes > 0
        assert metrics.measurement_source is not None
        assert captured_bodies
        captured_body_text = json.dumps(captured_bodies[0])
        assert source_marker in captured_body_text
        assert image_marker in captured_body_text
        assert owner.read_snapshot() == before
        assert source_ledger.transcript == active
        assert not source_ledger.has_pending_requests

    diagnostic = "\n".join(logs)
    assert "root_reason=backend_context_overflow" in diagnostic
    assert "failure_reason=summary_empty" in diagnostic
    assert "calls=1" in diagnostic
    for safe_field in (
        "provider=local_manager",
        f"model={model}",
        "selected_capacity=",
        "completion_reserve=128",
        "safety_buffer=",
        "payload_bytes=",
        "measurement_source=estimated_compat",
        "prefix_mode=historical_prefix",
        "cache_read_tokens=23",
    ):
        assert safe_field in diagnostic
    for private_marker in (
        credential_marker,
        image_marker,
        reasoning_marker,
        source_marker,
        "PENDING_PRIVATE_TAIL",
    ):
        assert private_marker not in diagnostic
    assert credential_marker not in repr(outcome)


def test_local_usage_after_failed_mandatory_record_cannot_leave_an_anchor() -> None:
    runner, spec, state, response = _usage_anchor_fixture()

    class FailingSourceWriter:
        def append_message(self, _message: object) -> None:
            raise RuntimeError("mandatory source record failed")

    spec.source_writer = FailingSourceWriter()
    ticket = object()
    with pytest.raises(RuntimeError, match="mandatory source record failed"):
        runner._record_and_settle(spec, state, ticket, response)

    # Usage totals remain available to the caller even though persistence failed.
    usage = runner._record_request_usage(spec, state, response)
    assert usage.input_tokens == 900
    assert usage.cache_read_tokens == 200
    assert state.local_usage_anchor is None
    assert state.source.discarded == [ticket]


@pytest.mark.parametrize(
    "usage",
    [
        LLMUsage.estimated(input_tokens=900, output_tokens=4),
        LLMUsage(
            input_tokens=900,
            output_tokens=4,
            total_tokens=904,
            reported_tokens=400,
            estimated_tokens=504,
            context_tokens=900,
        ),
        replace(LLMUsage.reported(input_tokens=900, output_tokens=4), request_count=2),
    ],
    ids=["estimated", "mixed", "multiple-requests"],
)
def test_estimated_mixed_and_aggregate_usage_never_installs_local_anchor(
    usage: LLMUsage,
) -> None:
    runner, spec, state, response = _usage_anchor_fixture()
    response = replace(response, usage=usage)
    recorded = runner._record_and_settle(spec, state, None, response)

    recorded_usage = runner._record_request_usage(
        spec,
        state,
        response,
        recorded_response=recorded,
    )

    assert recorded.local_usage_anchor_eligible is True
    assert recorded_usage.request_count == usage.request_count
    assert recorded_usage.source == usage.source
    assert state.local_usage_anchor is None


@pytest.mark.asyncio
async def test_source_projection_epoch_invalidates_local_usage_anchor() -> None:
    provider = OpenAICompatProvider(
        api_key="fixture",
        default_model=_MODEL,
        spec=find_by_name("local_manager"),
        provider_name="local_manager",
    )
    messages = [{"role": "user", "content": "projection-bound request"}]
    request_context = ProviderCallContext(request_limits=ProviderRequestLimits(
        total_tokens=12_000,
        completion_reserve_tokens=128,
        safety_buffer_tokens=0,
    ))
    _kwargs, body = provider._prepare_chat_request(
        messages,
        None,
        _MODEL,
        128,
        0.0,
        "none",
        None,
        stream=False,
        provider_context=request_context,
    )
    receipt = provider._make_local_usage_receipt(body, request_context)
    assert isinstance(receipt, LocalRequestUsageReceipt)
    transcript = SourceTranscript((
        RuntimeNode("projection-system", {"role": "system", "content": "fixed"}),
        SourceNode("projection-source", messages[0]),
    ))
    ledger = SourceRequestLedger(
        transcript,
        initial_eligible_ids=frozenset({"projection-source"}),
    )
    anchor = LocalUsageAnchor(
        receipt=receipt,
        prompt_tokens=1_000,
        projection_epoch=ledger.projection_epoch,
    )
    ledger.replace_projection(transcript)
    assert ledger.projection_epoch == anchor.projection_epoch + 1
    config = ContextGovernanceConfig(
        provider=provider,
        model=_MODEL,
        tools=ToolRegistry(),
        workspace=None,
        session_key="test:projection-anchor",
        max_tool_result_chars=4096,
        context_window_tokens=12_000,
        max_tokens=128,
        generation=GenerationSettings(max_tokens=128),
    )
    state = ModelRequestState(
        config,
        ProviderConversationStateController(
            provider=provider,
            model=_MODEL,
            messages=messages,
            session_id="test:projection-anchor",
        ),
        source=ledger,
        local_usage_anchor=anchor,
    )

    _prepared, provider_context = await ContextGovernor().prepare_request(
        state,
        messages,
        tool_definitions=None,
    )

    assert state.local_usage_anchor is None
    assert provider_context is not None
    assert provider_context.local_usage_anchor is None
