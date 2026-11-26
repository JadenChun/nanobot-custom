"""Current governor, finalized budget, recovery and retained utility invariants."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from kohaku.agent.context_governance import ContextGovernor, ContextWindowExceededError
from kohaku.agent.hook import AgentHook
from kohaku.agent.memory import MemoryStore
from kohaku.agent.runner import AgentRunner, AgentRunSpec
from kohaku.agent.tools.execution import KnownToolResult
from kohaku.agent.tools.registry import ToolRegistry
from kohaku.providers.base import GenerationSettings, LLMProvider, LLMResponse
from kohaku.providers.request_budget import FinalizedRequestBudgetExceeded
from kohaku.session.execution_owner import acquire_execution
from kohaku.session.manager import SessionManager
from kohaku.session.source import (
    SourceValidationError,
    validate_closed_boundary,
    validate_execution_links,
)
from kohaku.session.source_projection import source_nodes
from kohaku.session.turn_writer import SourceTurnWriter
from kohaku.utils.llm_runtime import LLMRuntime
from tests.agent.p6_memory_contract_fixtures import (
    SummaryProvider,
    concrete_provider,
    file_snapshot,
    fitting_prefix,
    prepare_source,
    provider_limits,
    source_state,
    source_store,
)


class _DeterministicCounter:
    """Stable message counter independent of tokenizer/model availability."""

    @staticmethod
    def estimate_prompt_tokens(messages, tools, _model):
        total = len(messages) * 5 + len(json.dumps(tools or [], sort_keys=True)) // 10
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                total += max(1, (len(content) + 9) // 10)
            elif content is not None:
                total += max(1, len(json.dumps(content, sort_keys=True)) // 10)
            for call in message.get("tool_calls") or []:
                total += 7 + len(str(call.get("id", "")))
            if message.get("tool_call_id"):
                total += 3 + len(str(message["tool_call_id"]))
        return total, "deterministic-counter"


class _StreamingHook(AgentHook):
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def wants_streaming(self) -> bool:
        return self.enabled


def test_scheduled_ring_is_bounded_and_legacy_history_is_collapsed(tmp_path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("telegram:room")
    session.add_message("user", "ordinary question")
    session.add_message("assistant", "ordinary answer")
    for index in range(25):
        session.add_scheduled_run(
            job_id=f"job-{index}",
            run_id=f"run-{index}",
            instruction="instruction-" + "i" * 400,
            result="result-" + "r" * 400,
            detail_session_key=f"cron:job-{index}",
        )
    session.messages.extend([
        {"role": "user", "content": "[Scheduled task from cron:legacy]\nold instruction"},
        {"role": "assistant", "content": "old result"},
    ])
    manager.save(session)
    jsonl_before = manager._get_session_path(session.key).read_text()

    ring = session.recent_scheduled_runs()
    assert len(ring) == 20
    assert ring[0]["run_id"] == "run-5"
    assert len(ring[-1]["instruction"]) <= session.SCHEDULED_EXCERPT_CHARS
    assert len(ring[-1]["result"]) <= session.SCHEDULED_EXCERPT_CHARS

    model_history = session.get_model_history(max_messages=0)
    assert any(item.get("content") == "ordinary question" for item in model_history)
    assert not any(item.get("content") == "old result" for item in model_history)
    projections = [
        item for item in model_history
        if item.get("content", "").startswith("[Recent Scheduled Runs]")
    ]
    assert len(projections) == 1
    assert len(projections[0]["content"]) <= session.SCHEDULED_PROJECTION_CHARS

    assert manager._get_session_path(session.key).read_text() == jsonl_before


def test_session_managers_merge_append_and_scheduled_metadata(tmp_path) -> None:
    first_manager = SessionManager(tmp_path)
    second_manager = SessionManager(tmp_path)
    first = first_manager.get_or_create("cli:merge")
    first.add_message("user", "base")
    first_manager.save(first)

    # Both managers now hold the same prefix, then append independently.
    second = second_manager.get_or_create("cli:merge")
    second.add_message("assistant", "from-second")
    second.add_scheduled_run(
        job_id="job-second",
        run_id="run-second",
        instruction="second instruction",
        result="second result",
        detail_session_key="cron:job-second",
    )
    second_manager.save(second)

    first.add_message("assistant", "from-first")
    first.add_scheduled_run(
        job_id="job-first",
        run_id="run-first",
        instruction="first instruction",
        result="first result",
        detail_session_key="cron:job-first",
    )
    first_manager.save(first)

    merged_manager = SessionManager(tmp_path)
    merged = merged_manager.get_or_create("cli:merge")
    assert [message["content"] for message in merged.messages] == [
        "base", "from-second", "from-first",
    ]
    assert {item["run_id"] for item in merged.recent_scheduled_runs()} == {
        "run-second", "run-first",
    }


class _ProviderSpy(LLMProvider):
    """Named real runner boundary; chat/chat_stream are the synthetic transport."""

    def __init__(self):
        super().__init__(provider_name="q1-runner-fixture")
        self.generation = GenerationSettings(max_tokens=0)
        self.transport_calls = []

    def get_default_model(self):
        return "test-model"

    def estimate_prompt_tokens(self, messages, tools, model):
        return _DeterministicCounter.estimate_prompt_tokens(messages, tools, model)

    async def chat(self, messages, **kwargs):
        self.transport_calls.append("chat")
        return LLMResponse(content="fitted")

    async def chat_stream(self, messages, on_content_delta=None, **kwargs):
        self.transport_calls.append("stream")
        if on_content_delta is not None:
            await on_content_delta("fitted")
        return LLMResponse(content="fitted")


@pytest.mark.asyncio
@pytest.mark.parametrize("early_fitting", [False, True])
async def test_irreducible_source_group_has_zero_transport_and_no_partial_checkpoint(tmp_path, early_fitting):
    memory = MemoryStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    backend = source_store(tmp_path / "source")
    provider = SummaryProvider([LLMResponse(content="must not run")])
    main, transport_calls = concrete_provider()

    def count(messages, tools, model):
        if messages[0]["content"].startswith("Summarize historical evidence"):
            payload = json.loads(messages[1]["content"])
            return (30000 if any("oversized" in str(row) for row in payload["events"]) else 100), "fixture"
        return 5000 + len(messages) * 1000, "fixture"

    provider.estimate_prompt_tokens = main.estimate_prompt_tokens = count
    async with acquire_execution(backend, "test:compact", run_id="run") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="run")
        if early_fitting:
            writer.append_message({"role": "assistant", "content": "earlier fitting"})
        for row in [
            {"role": "user", "content": "oversized " + "x" * 5000},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "keep"},
            {"role": "assistant", "content": "later"},
        ]:
            writer.append_message(row)
        state = source_state(writer, provider, eligible_count=len(owner.read_snapshot().events))
        state.config.provider = main
        # Pressure must trigger, while the fixed rebuilt request still has room.
        state.config.context_window_tokens = 10000
        before, active = owner.read_snapshot(), state.source.transcript
        raw = deepcopy(state.source_compaction.raw_messages)
        durable = file_snapshot(tmp_path)
        with pytest.raises(SourceValidationError, match="Indivisible source group"):
            await prepare_source(state)
        assert provider.calls == 0
        assert transport_calls == []
        assert owner.read_snapshot() == before
        assert state.source.transcript == active
        assert state.source_compaction.raw_messages == raw
        assert not state.source.has_pending_requests and state.source.projection_epoch == 0
        assert file_snapshot(tmp_path) == durable
        assert (memory.memory_dir / ".consolidation.pending.json").exists() is False


@pytest.mark.asyncio
async def test_governor_selects_exact_fitting_prefix_without_splitting_closed_execution(tmp_path):
    rows = [row for index in range(101) for row in (
        {"role": "user", "content": f"turn-{index}"},
        {"role": "assistant", "content": f"answer-{index}"},
    )]
    chunk, after = await fitting_prefix(tmp_path / "source", rows, 200)
    assert (0, after.checkpoint["boundary"]) == (0, 200)
    assert len(chunk) == 200
    assert chunk[-1]["content"] == "answer-99"
    assert all(
        chunk[index]["role"] == "user"
        and chunk[index + 1]["role"] == "assistant"
        for index in range(0, len(chunk), 2)
    )
    assert chunk == rows[:200]
    validate_closed_boundary(after.events, 200)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_runner_rejects_irreducible_iteration_zero_before_transport(streaming):
    provider = _ProviderSpy()
    secret = "do-not-leak-" + "x" * 20_000
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": secret}]
    before = deepcopy(messages)
    with pytest.raises(ContextWindowExceededError) as raised:
        await AgentRunner().run(AgentRunSpec(
            initial_messages=messages, tools=ToolRegistry(),
            runtime=LLMRuntime(provider, "test-model", GenerationSettings(max_tokens=0), 64),
            max_iterations=1, max_tool_result_chars=16000, hook=_StreamingHook(streaming),
        ))
    assert provider.transport_calls == []
    assert secret not in str(raised.value)
    assert secret not in repr(raised.value)
    assert not _ProviderSpy._is_transient_error(str(raised.value))
    assert messages == before


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_runner_fitting_request_reaches_matching_transport(streaming):
    provider = _ProviderSpy()
    result = await AgentRunner().run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "fit"}], tools=ToolRegistry(),
        runtime=LLMRuntime(provider, "test-model", GenerationSettings(max_tokens=0), 4096),
        max_iterations=1, max_tool_result_chars=16000, hook=_StreamingHook(streaming),
    ))
    assert result.final_content == "fitted"
    assert provider.transport_calls == ["stream" if streaming else "chat"]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_budget_boundary_has_zero_or_one_transport_call(streaming):
    provider, transport_calls = concrete_provider()
    oversized = [{"role": "user", "content": "private-value-" + "x" * 50_000}]
    before = deepcopy(oversized)
    if streaming:
        with pytest.raises(FinalizedRequestBudgetExceeded):
            await provider.chat_stream_with_retry(messages=oversized, model="test-model",
                max_tokens=1, provider_context=provider_limits(2048))
    else:
        with pytest.raises(FinalizedRequestBudgetExceeded):
            await provider.chat_once(messages=oversized, model="test-model",
                max_tokens=1, provider_context=provider_limits(2048))
    assert transport_calls == []
    if streaming:
        result = await provider.chat_stream_with_retry(messages=[{"role": "user", "content": "fit"}],
            model="test-model", max_tokens=1, provider_context=provider_limits(2048))
    else:
        result = await provider.chat_once(messages=[{"role": "user", "content": "fit"}],
            model="test-model", max_tokens=1, provider_context=provider_limits(2048))
    assert result.content == "fitted"
    assert transport_calls == ["stream" if streaming else "chat"]
    assert oversized == before


def _pressure(writer, *, chunks=3):
    writer.append_message({"role": "user", "content": "objective"})
    for index in range(chunks):
        writer.append_message({"role": "assistant", "content": f"chunk-{index}"})
    writer.append_message({"role": "user", "content": "current instruction"})
    provider = SummaryProvider([LLMResponse(content="retained facts") for _ in range(12)])

    def count(messages, tools, model):
        text = str(messages)
        auxiliary = messages[0]["content"].startswith("Summarize historical evidence")
        return ((100 if auxiliary else 5200) + text.count("chunk-") * (6000 if auxiliary else 1500)), "fixture"

    provider.estimate_prompt_tokens = count
    state = source_state(writer, provider)
    return state, provider


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [5, 8])
async def test_four_call_ceiling_publishes_only_a_hard_fitting_partial_projection(tmp_path, chunks):
    backend = source_store(tmp_path / "source")
    async with acquire_execution(backend, "test:compact", run_id="run") as owner:
        state, provider = _pressure(SourceTurnWriter(backend.source, owner, run_id="run"), chunks=chunks)
        before, active = owner.read_snapshot(), state.source.transcript
        raw = deepcopy(state.source_compaction.raw_messages)
        durable = file_snapshot(tmp_path)
        if chunks == 8:
            with pytest.raises(ContextWindowExceededError):
                await prepare_source(state)
            assert owner.read_snapshot() == before
            assert state.source.transcript == active
            assert state.source_compaction.raw_messages == raw
            assert file_snapshot(tmp_path) == durable
        else:
            messages, _ = await prepare_source(state)
            assert owner.read_snapshot().events == before.events
            assert owner.read_snapshot().checkpoint["boundary"] > 0
            measured = provider.estimate_prompt_tokens(messages, None, "test-model")[0]
            assert ContextGovernor.input_budget(state.config) * 0.6 < measured <= ContextGovernor.input_budget(state.config)
        assert provider.calls == 4
        assert state.source_compaction.attempts[-1].calls == 4
        assert not state.source.has_pending_requests
        payloads = [json.loads(prompt[1]["content"]) for prompt in provider.prompts]
        assert [payload["previous_summary"] for payload in payloads] == [
            None, "retained facts", "retained facts", "retained facts",
        ]
        evidence = [row["content"] for payload in payloads for row in payload["events"]]
        assert [item for item in evidence if item.startswith("chunk-")] == [f"chunk-{i}" for i in range(4)]
        assert "current instruction" not in evidence


@pytest.mark.asyncio
async def test_failed_successful_failed_compaction_preserves_last_successful_checkpoint(tmp_path):
    backend = source_store(tmp_path / "source")
    async with acquire_execution(backend, "test:compact", run_id="run") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="run")
        state, provider = _pressure(writer)
        provider.responses.appendleft(LLMResponse(content="partial", finish_reason="error"))
        before, active = owner.read_snapshot(), state.source.transcript
        durable = file_snapshot(tmp_path)
        with pytest.raises(SourceValidationError, match="invalid_response"):
            await prepare_source(state)
        assert owner.read_snapshot() == before
        assert state.source.transcript == active
        assert file_snapshot(tmp_path) == durable
        await prepare_source(state)
        successful = owner.read_snapshot()
        assert successful.checkpoint["boundary"] > 0
        state.source.acknowledge(state.source.issue(state.source_view))
        for index in range(3, 6):
            row = {"role": "assistant", "content": f"chunk-{index}"}
            event = writer.append_message(row)
            state.source.append_source(event["_event_id"], row)
        state.source_compaction.raw_messages[:] = state.source.transcript.messages()
        state.source.acknowledge(state.source.issue(state.source.transcript))
        provider.responses.appendleft(LLMResponse(content="tool", finish_reason="error"))
        before_failure, active = owner.read_snapshot(), state.source.transcript
        raw = deepcopy(state.source_compaction.raw_messages)
        durable = file_snapshot(tmp_path)
        with pytest.raises(SourceValidationError, match="invalid_response"):
            await prepare_source(state)
        assert owner.read_snapshot() == before_failure
        assert owner.read_snapshot().checkpoint == successful.checkpoint
        assert state.source.transcript == active
        assert state.source_compaction.raw_messages == raw
        assert file_snapshot(tmp_path) == durable
        assert not state.source.has_pending_requests
        assert [attempt.status for attempt in state.source_compaction.attempts] == ["failed", "succeeded", "failed"]
        assert json.loads(provider.prompts[-1][1]["content"])["previous_summary"] == "retained facts"


@pytest.mark.asyncio
async def test_source_selection_never_splits_real_linked_tool_execution(tmp_path):
    backend = source_store(tmp_path / "source")
    provider = SummaryProvider([LLMResponse(content="tool facts")])
    async with acquire_execution(backend, "test:compact", run_id="run") as owner:
        writer = SourceTurnWriter(backend.source, owner, run_id="run")
        writer.append_message({"role": "user", "content": "objective"})
        intent = writer.begin_tool_batch({"role": "assistant", "content": "", "tool_calls": [
            {"id": "call", "type": "function", "function": {"name": "fixture_tool", "arguments": "{}"}},
        ]})
        await writer.record_tool_result(KnownToolResult(
            0, "call", "fixture_tool", "result", "result", {"name": "fixture_tool", "status": "ok"}, True,
        ))
        writer.append_message({"role": "assistant", "content": "closed"})
        writer.append_message({"role": "user", "content": "current instruction"})

        def count(messages, tools, model):
            auxiliary = messages[0]["content"].startswith("Summarize historical evidence")
            return (100 if auxiliary or any("Historical context summary" in str(row) for row in messages)
                    else 10000), "fixture"

        provider.estimate_prompt_tokens = count
        state = source_state(writer, provider)
        before = owner.read_snapshot()
        candidate = ContextGovernor().capture_source_compaction(state, state.source.transcript,
                                                               tool_definitions=None).candidate
        assert 2 not in candidate.boundaries  # Intent cannot be separated from its result.
        assert 3 in candidate.boundaries and 4 in candidate.boundaries
        messages, _ = await prepare_source(state)
        after = owner.read_snapshot()
        assert after.events == before.events
        assert after.checkpoint["boundary"] == 4
        validate_execution_links(after.events)
        validate_closed_boundary(after.events, after.checkpoint["boundary"])
        payload = json.loads(provider.prompts[0][1]["content"])
        assert payload["events"] == [node.message for node in source_nodes(before.events[:4])]
        assert intent["_execution_ids"] == [before.events[2]["_execution_id"]]
        assert messages[-1]["content"] == "current instruction"
        assert provider.calls == 1
