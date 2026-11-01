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

import pytest

from nanobot.agent.context_governance import ContextGovernor
from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.turn import TurnCallbacks, TurnRequest
from nanobot.config.schema import Config
from nanobot.events import AgentEvent, ContextCompactionEvent, EventSink
from nanobot.providers.base import (
    GenerationSettings,
    LLMResponse,
    ProviderCallContext,
    ToolCallRequest,
)
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.request_budget import (
    FinalizedRequestBudgetExceeded,
    TransportIdentity,
    TransportMeasurement,
)
from nanobot.session.checkpoint_dependencies import SummaryDependencyProof
from nanobot.session.execution_owner import acquire_execution
from nanobot.session.manager import SessionWriteConflict
from nanobot.session.source import SourceValidationError, validate_execution_links
from nanobot.session.source_projection import CheckpointNode, SourceNode
from nanobot.session.source_request import (
    SourceRequestLedger,
    SourceRequestTicket,
    SourceTranscript,
)
from nanobot.session.turn_writer import SourceTurnWriter
from nanobot.utils.helpers import estimate_prompt_tokens_chain
from nanobot.utils.runtime import build_budget_exhausted_finalization_message
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
    ("streaming", "finalization", "irreducible", "no_progress"),
    [(False, False, False, False), (True, False, False, False),
     (False, True, False, False), (False, False, True, False),
     (False, False, False, True)],
    ids=["nonstreaming", "streaming", "no-tools-finalization", "irreducible-tail",
         "structural-no-progress"],
)
async def test_source_turn_recovers_one_finalized_request_budget_rejection(
    engine_config: Config, tmp_path, monkeypatch, streaming: bool, finalization: bool,
    irreducible: bool, no_progress: bool,
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
        prompt_head = messages[0].get("content", "") if messages else ""
        if isinstance(prompt_head, str) and prompt_head.startswith("Summarize historical evidence"):
            auxiliary_create_calls.append(body)
            source_operations.append(("auxiliary", len(auxiliary_create_calls)))
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

    def observe_guard(
        body: dict[str, object],
        *,
        transport: str,
        model: str | None,
        provider_context: ProviderCallContext | None = None,
        completion_reserve_tokens: int | None = None,
    ) -> TransportMeasurement | None:
        body_messages = body.get("messages")
        assert isinstance(body_messages, list)
        first_message = body_messages[0] if body_messages else {}
        first_content = (
            first_message.get("content") if isinstance(first_message, dict) else None
        )
        is_auxiliary = (
            isinstance(first_content, str)
            and first_content.startswith("Summarize historical evidence")
        )
        try:
            result = original_guard(
                body,
                transport=transport,
                model=model,
                provider_context=provider_context,
                completion_reserve_tokens=completion_reserve_tokens,
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
        finalized_measurements.append(result)
        guard_records.append((is_auxiliary, deepcopy(body), result,
                              None if is_auxiliary else latest_ticket))
        source_operations.append(("guard_fitted", result))
        return result

    monkeypatch.setattr(provider, "validate_finalized_request", observe_guard)
    budget_error_measurements: list[TransportMeasurement | None] = []
    original_budget_error = runner_module._context_budget_error_response

    def observe_budget_error(exc: FinalizedRequestBudgetExceeded):
        budget_error_measurements.append(exc.measurement)
        return original_budget_error(exc)

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
            assert result.error is not None
            assert result.content != "Recovered after budget pressure."
            assert len(main_create_calls) == 4
            assert len(budget_error_measurements) == 1
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
        assert len(main_rejections) == (2 if irreducible else 1)
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
                             and (record[2].fits is not irreducible)
                             and record[3] is not None
                             and record[3].projection_epoch > failed_ticket.projection_epoch]
        assert len(recovered_records) == 1
        _, recovered_body, recovered_measurement, recovered_ticket = recovered_records[0]
        assert recovered_measurement is not None and recovered_ticket is not None
        assert recovered_measurement.input_tokens < rejected_measurement.input_tokens
        assert recovered_measurement.fits is not irreducible
        assert recovered_ticket.ticket_id != failed_ticket.ticket_id
        assert recovered_ticket.projection_epoch == failed_ticket.projection_epoch + 1
        assert recovered_ticket.checkpoint_id is not None

        if irreducible:
            assert any(value == recovered_ticket for kind, value in source_operations
                       if kind == "discard")
            assert all(value != recovered_ticket for kind, value in source_operations
                       if kind == "acknowledge")
            assert budget_error_measurements == [rejected_measurement]
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
            assert capture_hook.content.count("Recovered after budget pressure.") == 1
            assert capture_hook.reasoning == ["Earlier reasoning."]
            assert streamed.count("Earlier progress.") == 1
            assert streamed.count("Recovered after budget pressure.") == 1
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
        measurement = TransportMeasurement(
            identity=TransportIdentity("local_manager", "chat", _MODEL),
            input_tokens=budget + 1,
            completion_reserve_tokens=256,
            safety_buffer_tokens=1_024,
            total_limit_tokens=budget + 1_280,
            input_limit_tokens=budget,
            source="utf8-json-bytes",
        )
        recovered = await governor.recover_finalized_request(
            state,
            measurement=measurement,
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages,
            source_view=state.source_view,
        )

        assert recovered is not None
        recovered_messages, _ = recovered
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

        assert recovered is None
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
        measurement = TransportMeasurement(
            identity=TransportIdentity("local_manager", "chat", _MODEL),
            input_tokens=budget + 1,
            completion_reserve_tokens=256,
            safety_buffer_tokens=1_024,
            total_limit_tokens=budget + 1_280,
            input_limit_tokens=budget,
            source="utf8-json-bytes",
        )

        recovered = await governor.recover_finalized_request(
            state,
            measurement=measurement,
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages,
            source_view=state.source_view,
        )

        assert recovered is None
        assert len(state.config.provider.calls) == 1
        assert state.source_compaction.attempts[-1].status == "failed"
        assert state.source_compaction.attempts[-1].calls == 1
        assert owner.read_snapshot() == before
        assert state.source.transcript == active
        assert state.source.projection_epoch == 0
        assert not state.source.has_pending_requests


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
        measurement = TransportMeasurement(
            identity=TransportIdentity("local_manager", "chat", _MODEL),
            input_tokens=budget + 1,
            completion_reserve_tokens=256,
            safety_buffer_tokens=1_024,
            total_limit_tokens=budget + 1_280,
            input_limit_tokens=budget,
            source="utf8-json-bytes",
        )

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
        provider_chat = state.config.provider.chat

        async def update_during_summary(*args, **kwargs):
            store.source.update_metadata("test:compact", {"title": "concurrent update"})
            return await provider_chat(*args, **kwargs)

        state.config.provider.chat = update_during_summary
        budget = governor.input_budget(state.config)
        measurement = TransportMeasurement(
            identity=TransportIdentity("local_manager", "chat", _MODEL),
            input_tokens=budget + 1,
            completion_reserve_tokens=256,
            safety_buffer_tokens=1_024,
            total_limit_tokens=budget + 1_280,
            input_limit_tokens=budget,
            source="utf8-json-bytes",
        )

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
        measurement = TransportMeasurement(
            identity=TransportIdentity("local_manager", "chat", _MODEL),
            input_tokens=budget + 1,
            completion_reserve_tokens=256,
            safety_buffer_tokens=1_024,
            total_limit_tokens=budget + 1_280,
            input_limit_tokens=budget,
            source="utf8-json-bytes",
        )

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
        measurement = TransportMeasurement(
            identity=TransportIdentity("local_manager", "chat", _MODEL),
            input_tokens=budget + 1,
            completion_reserve_tokens=256,
            safety_buffer_tokens=1_024,
            total_limit_tokens=budget + 1_280,
            input_limit_tokens=budget,
            source="utf8-json-bytes",
        )

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


async def test_auxiliary_actual_final_guard_overflow_aborts_recovery(store, monkeypatch):
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
        rejected_main = TransportMeasurement(
            identity=TransportIdentity("local_manager", "chat", _MODEL),
            input_tokens=budget + 1,
            completion_reserve_tokens=256,
            safety_buffer_tokens=1_024,
            total_limit_tokens=budget + 1_280,
            input_limit_tokens=budget,
            source="utf8-json-bytes",
        )
        auxiliary_rejections: list[TransportMeasurement] = []
        original_guard = provider.validate_finalized_request

        def observe_guard(
            body: dict[str, object],
            *,
            transport: str,
            model: str | None,
            provider_context: ProviderCallContext | None = None,
            completion_reserve_tokens: int | None = None,
        ) -> TransportMeasurement | None:
            try:
                return original_guard(
                    body,
                    transport=transport,
                    model=model,
                    provider_context=provider_context,
                    completion_reserve_tokens=completion_reserve_tokens,
                )
            except FinalizedRequestBudgetExceeded as exc:
                auxiliary_rejections.append(exc.measurement)
                raise

        monkeypatch.setattr(provider, "validate_finalized_request", observe_guard)

        recovered = await governor.recover_finalized_request(
            state,
            measurement=rejected_main,
            tool_definitions=None,
            transcript=state.source_compaction.raw_messages,
            source_view=state.source_view,
        )

        assert recovered is None
        assert len(auxiliary_rejections) == 1
        assert auxiliary_rejections[0].input_tokens > auxiliary_rejections[0].input_limit_tokens
        assert sdk_create.await_count == 0
        assert state.source_compaction.attempts[-1].status == "failed"
        assert state.source_compaction.attempts[-1].calls == 0
        assert owner.read_snapshot() == before
        assert state.source.transcript == active
        assert state.source.projection_epoch == 0
        assert not state.source.has_pending_requests
