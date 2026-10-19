"""Fail-closed storage-free summary coverage with historical Q1 lineage."""

import asyncio
import json
from copy import deepcopy

import pytest

from nanobot.agent.memory import Consolidator, MemoryStore
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from tests.agent.p6_memory_contract_fixtures import SummaryProvider, file_snapshot, summary_runtime


def _make_messages(message_count=60):
    return [{"role": "user", "content": f"msg{i}", "timestamp": "2026-01-01 00:00"}
            for i in range(message_count)]


def _tool(arguments):
    return LLMResponse(content=None, tool_calls=[ToolCallRequest(
        id="call_1", name="save_memory", arguments=arguments,
    )])


async def _checked(tmp_path, responses, summaries, *, messages=None, previous=None,
                   cap=256, preexisting=False):
    store = MemoryStore(tmp_path, sessions_root=tmp_path / "sessions")
    if preexisting:
        store.memory_file.write_bytes(b"M0\n")
        store.history_file.write_bytes(b'{"preexisting":"history"}\n')
        (store.memory_dir / ".cursor").write_bytes(b"3\n")
        (store.memory_dir / ".dream_cursor").write_bytes(b"2\n")
        # This operation has no store dependency, even if a later constructor
        # would require offline receipt reconciliation. No recovery guard runs.
        (store.memory_dir / ".consolidation.pending.json").write_bytes(b'{"pending":"preserve"}\n')
    before_files = file_snapshot(tmp_path)
    source = messages if messages is not None else _make_messages()
    before_source, before_responses = deepcopy(source), deepcopy(responses)
    provider = SummaryProvider(responses)
    runtime = summary_runtime(provider)
    results = []
    for index, summary in enumerate(summaries):
        prior = previous[index] if previous is not None else None
        result = await Consolidator.summarize_source_chunk(
            source, prior, runtime=runtime, max_output_tokens=cap,
        )
        assert result.status == ("complete" if summary is not None else "invalid_response")
        assert result.summary == summary
        assert result.calls == 1
        assert provider.calls == index + 1  # No automatic retry consumes the next response.
        assert len(provider.responses) == len(responses) - index - 1
        assert file_snapshot(tmp_path) == before_files
        assert source == before_source
        payload = json.loads(provider.prompts[index][1]["content"])
        assert payload == {"previous_summary": prior, "events": before_source}
        kwargs = provider.kwargs[index]
        assert kwargs["tools"] is None
        assert kwargs["model"] == "test-model"
        assert kwargs["temperature"] == 0.25
        assert kwargs["reasoning_effort"] == "none"
        assert kwargs["max_tokens"] == cap
        assert "tool_choice" not in kwargs
        context = provider.contexts[index]
        assert context.conversation_state is None
        assert context.request_limits.total_tokens == 8192
        assert context.request_limits.completion_reserve_tokens == cap
        assert "[RAW]" not in str(result)
        results.append(result)
    assert responses == before_responses
    if not preexisting:
        assert not store.history_file.exists()
        assert not store.memory_file.exists()
    return store, provider, results


@pytest.mark.asyncio
@pytest.mark.parametrize("preexisting", [False, True])
async def test_plain_text_source_summary_is_storage_free(tmp_path, preexisting):
    text = "[2026-01-01] User discussed testing."
    _, provider, results = await _checked(tmp_path, [LLMResponse(content=text)], [text],
                                         preexisting=preexisting)
    assert results[0].summary == text
    assert len(json.loads(provider.prompts[0][1]["content"])["events"]) == 60


@pytest.mark.asyncio
async def test_tool_dict_output_is_rejected_without_coercion(tmp_path):
    arguments = {"history_entry": {"timestamp": "2026-01-01", "summary": "User discussed testing."},
                 "memory_update": {"facts": ["User likes testing"], "topics": ["testing"]}}
    raw = json.dumps(arguments)
    _, _, results = await _checked(tmp_path, [_tool(arguments), LLMResponse(content=raw)], [None, raw])
    assert results[0].summary is None and results[1].summary == raw


@pytest.mark.asyncio
async def test_tool_raw_json_output_is_rejected_without_parsing(tmp_path):
    raw = json.dumps({"history_entry": "[2026-01-01] User discussed testing.",
                     "memory_update": "# Memory\nUser likes testing."})
    _, _, results = await _checked(tmp_path, [_tool(raw), LLMResponse(content=raw)], [None, raw])
    assert results[1].summary == raw


@pytest.mark.asyncio
async def test_plain_text_no_tool_summary_succeeds(tmp_path):
    text = "I summarized the conversation."
    store, _, results = await _checked(tmp_path, [LLMResponse(content=text)], [text])
    assert results[0].status == "complete"
    assert not store.history_file.exists()


@pytest.mark.asyncio
async def test_empty_source_is_rejected_before_transport(tmp_path):
    store = MemoryStore(tmp_path, sessions_root=tmp_path / "sessions")
    before = file_snapshot(tmp_path)
    provider = SummaryProvider([LLMResponse(content="must not run")])
    with pytest.raises(ValueError, match="source evidence"):
        await Consolidator.summarize_source_chunk([], None, runtime=summary_runtime(provider))
    assert provider.calls == 0
    assert file_snapshot(tmp_path) == before
    assert not store.history_file.exists()


@pytest.mark.asyncio
async def test_tool_list_of_dict_output_is_rejected(tmp_path):
    _, _, results = await _checked(tmp_path, [_tool([{
        "history_entry": "[2026-01-01] User discussed testing.",
        "memory_update": "# Memory\nUser likes testing.",
    }])], [None])
    assert results[0].status == "invalid_response"


@pytest.mark.asyncio
async def test_tool_empty_list_output_is_rejected(tmp_path):
    _, _, results = await _checked(tmp_path, [_tool([])], [None])
    assert results[0].summary is None


@pytest.mark.asyncio
async def test_tool_non_dict_list_output_is_rejected(tmp_path):
    _, _, results = await _checked(tmp_path, [_tool(["string", "content"])], [None])
    assert results[0].summary is None


@pytest.mark.asyncio
async def test_tool_missing_history_output_is_rejected_without_writing(tmp_path):
    store, _, results = await _checked(tmp_path, [_tool({"memory_update": "# Memory\nOnly memory update"})],
                                       [None])
    assert results[0].status == "invalid_response"
    assert not store.history_file.exists()
    assert not store.memory_file.exists()


@pytest.mark.asyncio
async def test_tool_missing_memory_output_is_rejected_without_writing(tmp_path):
    text = "[2026-01-01] Partial output."
    store, _, results = await _checked(tmp_path, [_tool({"history_entry": text}), LLMResponse(content=text)],
                                       [None, text])
    assert results[0].summary is None and results[1].summary == text
    assert not store.history_file.exists()
    assert not store.memory_file.exists()


@pytest.mark.asyncio
async def test_null_summary_content_is_rejected_without_writing(tmp_path):
    store, _, results = await _checked(tmp_path, [
        _tool({"history_entry": None, "memory_update": "# Memory\nUser likes testing."}),
        LLMResponse(content=None),
    ], [None, None])
    assert all(result.status == "invalid_response" for result in results)
    assert not store.history_file.exists()
    assert not store.memory_file.exists()


@pytest.mark.asyncio
async def test_blank_summary_content_is_rejected_without_writing(tmp_path):
    store, _, results = await _checked(tmp_path, [
        _tool({"history_entry": "   ", "memory_update": "# Memory\nUser likes testing."}),
        LLMResponse(content="   "),
    ], [None, None])
    assert all(result.summary is None for result in results)
    assert not store.history_file.exists()
    assert not store.memory_file.exists()


@pytest.mark.asyncio
async def test_transient_source_summary_failure_is_single_attempt(tmp_path, monkeypatch):
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("nanobot.providers.base.asyncio.sleep", sleep)
    text = "[2026-01-01] User discussed testing."
    _, provider, results = await _checked(tmp_path, [
        LLMResponse(content="503 server error", finish_reason="error"), LLMResponse(content=text),
    ], [None, text])
    assert [result.status for result in results] == ["invalid_response", "complete"]
    assert provider.calls == 2
    assert delays == []


@pytest.mark.asyncio
async def test_source_summary_uses_captured_runtime_and_bounded_request(tmp_path):
    store = MemoryStore(tmp_path, sessions_root=tmp_path / "sessions")
    before = file_snapshot(tmp_path)
    provider = SummaryProvider([LLMResponse(content="facts")])
    provider.generation = GenerationSettings(max_tokens=8192, temperature=0.25, reasoning_effort="high")
    runtime = summary_runtime(provider)
    provider.generation = GenerationSettings(max_tokens=99, temperature=0.9, reasoning_effort="low")
    result = await Consolidator.summarize_source_chunk(_make_messages(), None, runtime=runtime,
                                                      max_output_tokens=256)
    assert result.status == "complete" and result.summary == "facts" and result.calls == 1
    assert provider.calls == 1
    kwargs = provider.kwargs[0]
    assert kwargs["model"] == "test-model"
    assert kwargs["temperature"] == 0.25
    assert kwargs["max_tokens"] == 256
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["tools"] is None
    assert provider.contexts[0].conversation_state is None
    assert provider.contexts[0].request_limits.total_tokens == 8192
    assert provider.contexts[0].request_limits.completion_reserve_tokens == 256
    assert file_snapshot(tmp_path) == before
    assert not store.memory_file.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [1, 256, 2048])
async def test_source_summary_validates_bounded_output_cap(tmp_path, cap):
    _, provider, results = await _checked(tmp_path, [LLMResponse(content="x")], ["x"], cap=cap)
    assert results[0].status == "complete"
    assert provider.kwargs[0]["max_tokens"] == cap
    assert provider.contexts[0].request_limits.completion_reserve_tokens == cap


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [0, -1, 2049, 8192, True])
async def test_invalid_summary_output_cap_has_zero_transport(tmp_path, cap):
    MemoryStore(tmp_path, sessions_root=tmp_path / "sessions")
    before = file_snapshot(tmp_path)
    provider = SummaryProvider([LLMResponse(content="must not run")])
    with pytest.raises(ValueError, match="limits"):
        await Consolidator.summarize_source_chunk(_make_messages(), None, runtime=summary_runtime(provider),
                                                  max_output_tokens=cap)
    assert provider.calls == 0
    assert file_snapshot(tmp_path) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, 60.01])
async def test_invalid_summary_timeout_has_zero_transport(tmp_path, timeout):
    MemoryStore(tmp_path, sessions_root=tmp_path / "sessions")
    before = file_snapshot(tmp_path)
    provider = SummaryProvider([LLMResponse(content="must not run")])
    with pytest.raises(ValueError, match="limits"):
        await Consolidator.summarize_source_chunk(_make_messages(), None, runtime=summary_runtime(provider),
                                                  timeout_seconds=timeout)
    assert provider.calls == 0
    assert file_snapshot(tmp_path) == before


@pytest.mark.asyncio
async def test_summary_timeout_cancels_the_single_transport(tmp_path):
    class WaitingProvider(SummaryProvider):
        async def chat(self, messages, **kwargs):
            self.calls += 1
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True

    MemoryStore(tmp_path, sessions_root=tmp_path / "sessions")
    before = file_snapshot(tmp_path)
    provider = WaitingProvider([])
    provider.cancelled = False
    result = await Consolidator.summarize_source_chunk(_make_messages(), None,
        runtime=summary_runtime(provider), timeout_seconds=0.01, max_output_tokens=256)
    assert result.status == "timeout" and result.summary is None and result.calls == 1
    assert provider.calls == 1 and provider.cancelled
    assert file_snapshot(tmp_path) == before


@pytest.mark.asyncio
async def test_tool_choice_error_does_not_retry_auxiliary_summary(tmp_path):
    _, provider, results = await _checked(tmp_path, [
        LLMResponse(content="tool_choice required is unsupported", finish_reason="error"),
        LLMResponse(content="Fallback worked."),
    ], [None, "Fallback worked."])
    assert results[0].summary is None and results[1].summary == "Fallback worked."
    assert all("tool_choice" not in kwargs for kwargs in provider.kwargs)


@pytest.mark.asyncio
async def test_repeated_stateless_summary_calls_have_no_tool_fallback_learning(tmp_path):
    _, provider, results = await _checked(tmp_path, [
        LLMResponse(content="tool_choice required is unsupported", finish_reason="error"),
        LLMResponse(content="summary2"), LLMResponse(content="summary3"),
    ], [None, "summary2", "summary3"], previous=[None, "explicit prior2", "explicit prior3"])
    assert [result.status for result in results] == ["invalid_response", "complete", "complete"]
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_unrelated_error_is_single_attempt_invalid_summary(tmp_path):
    _, provider, results = await _checked(tmp_path, [
        LLMResponse(content="reasoning_effort is unsupported", finish_reason="error"),
    ], [None])
    assert results[0].status == "invalid_response" and provider.calls == 1


@pytest.mark.asyncio
async def test_error_then_explicit_plain_summary_calls_are_independent(tmp_path):
    store, _, results = await _checked(tmp_path, [
        LLMResponse(content="tool_choice required is unsupported", finish_reason="error"),
        LLMResponse(content="Here is a summary."),
    ], [None, "Here is a summary."])
    assert results[1].summary == "Here is a summary."
    assert not store.history_file.exists()


@pytest.mark.asyncio
async def test_repeated_invalid_summaries_never_raw_archive(tmp_path):
    store, provider, results = await _checked(tmp_path, [
        _tool({"history_entry": "No tool call."}),
        LLMResponse(content="503", finish_reason="error"), _tool([]),
        LLMResponse(content="No tool call."),
    ], [None, None, None, "No tool call."], messages=_make_messages(10))
    assert [result.status for result in results[:3]] == ["invalid_response"] * 3
    assert results[3].summary == "No tool call."
    assert provider.calls == 4
    assert not store.memory_file.exists()


@pytest.mark.asyncio
async def test_length_summaries_never_publish_raw_fallback(tmp_path):
    store, provider, results = await _checked(tmp_path, [
        LLMResponse(content="partial", finish_reason="length") for _ in range(3)
    ], [None, None, None], messages=_make_messages(10))
    assert [result.status for result in results] == ["invalid_response"] * 3
    assert provider.calls == 3
    for kwargs in provider.kwargs:
        assert kwargs["max_tokens"] == 256
    assert not store.memory_file.exists()


@pytest.mark.asyncio
async def test_independent_summary_success_has_no_failure_counter_policy(tmp_path):
    _, provider, results = await _checked(tmp_path, [_tool([]), LLMResponse(content="OK."), _tool([])],
                                         [None, "OK.", None])
    assert [result.status for result in results] == ["invalid_response", "complete", "invalid_response"]
    assert [result.summary for result in results] == [None, "OK.", None]
    assert provider.calls == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [{"summary": "facts"}, ["facts"], "x" * 64001, "word " * 5000])
async def test_nonstring_or_oversized_plain_summary_is_rejected(tmp_path, content):
    _, _, results = await _checked(tmp_path, [LLMResponse(content=content)], [None])
    assert results[0].status == "invalid_response"


@pytest.mark.asyncio
async def test_original_source_shapes_remain_data_not_executable_calls(tmp_path):
    source = [
        {"role": "user", "content": {"summary": "User discussed testing."}},
        {"role": "assistant", "content": [{"type": "text", "text": "User likes testing"}],
         "tool_calls": [{"id": "old", "type": "function", "function": {
             "name": "save_memory", "arguments": json.dumps({"history_entry": "data"}),
         }}]},
        {"role": "tool", "tool_call_id": "old", "content": '{"facts":["testing"]}'},
    ]
    _, provider, results = await _checked(tmp_path, [LLMResponse(content="facts")], ["facts"],
                                         messages=source, previous=["prior summary"])
    assert results[0].summary == "facts"
    assert [message["role"] for message in provider.prompts[0]] == ["system", "user"]
    assert not any(message.get("tool_calls") for message in provider.prompts[0])


@pytest.mark.asyncio
async def test_preexisting_targets_survive_success_and_invalid_output(tmp_path):
    _, provider, results = await _checked(tmp_path, [
        _tool({"history_entry": "must not write"}), LLMResponse(content="facts"),
        LLMResponse(content="partial", finish_reason="length"),
    ], [None, "facts", None], preexisting=True)
    assert [result.status for result in results] == ["invalid_response", "complete", "invalid_response"]
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_summary_cancellation_propagates_without_storage_effects(tmp_path):
    MemoryStore(tmp_path, sessions_root=tmp_path / "sessions")
    before = file_snapshot(tmp_path)
    provider = SummaryProvider([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await Consolidator.summarize_source_chunk(_make_messages(), None,
            runtime=summary_runtime(provider), max_output_tokens=256)
    assert provider.calls == 1
    assert file_snapshot(tmp_path) == before
