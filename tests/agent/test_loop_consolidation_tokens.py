from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.providers.base import (
    GenerationSettings,
    LLMResponse,
    ProviderConversationState,
)
from nanobot.session.summary import SUMMARY_CONTINUATION_TEXT
from tests.agent import p6_admitted_fixtures

admitted_async = p6_admitted_fixtures.admitted_async


async def _make_loop(
    tmp_path,
    admitted_async,
    *,
    estimated_tokens: int,
    context_window_tokens: int,
    max_tokens: int = 0,
) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=max_tokens)
    provider.estimate_prompt_tokens.return_value = (estimated_tokens, "test-counter")
    _response = LLMResponse(content="ok", tool_calls=[])
    provider.chat_with_retry = AsyncMock(return_value=_response)
    provider.chat_stream_with_retry = AsyncMock(return_value=_response)

    def build(inputs):
        return AgentLoop(
            bus=inputs.bus, provider=provider, workspace=inputs.config.workspace_path,
            model="test-model", context_window_tokens=context_window_tokens,
            session_manager=inputs.sessions, tool_registry=inputs.tools,
            cron_service=inputs.cron, turn_delivery_factory=inputs.delivery,
            enable_legacy_auto_compact=False,
        )

    loop = await admitted_async(
        tmp_path, build, keys=("cli:test", "cli:native"),
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.consolidator._SAFETY_BUFFER = 0
    return loop


@pytest.mark.asyncio
async def test_runner_pressure_commits_summary_and_current_delta(tmp_path, admitted_async) -> None:
    loop = await _make_loop(
        tmp_path, admitted_async, estimated_tokens=100, context_window_tokens=1_624, max_tokens=100,
    )
    loop.provider.can_resume_conversation_state.return_value = False
    loop.schedule_background = lambda coro: coro.close()  # type: ignore[method-assign]

    session = loop.sessions.get_or_create("cli:test")
    session.messages = [
        {"role": role, "content": f"old-{role}-{turn}"}
        for turn in range(6)
        for role in ("user", "assistant")
    ]
    loop.sessions.save(session)

    def estimate(messages, _tools, _model):
        contents = [str(message.get("content")) for message in messages]
        if contents and "SNIP" in contents[-1]:
            return 300, "test-counter"
        if any(content.startswith("old-") for content in contents):
            return 600, "test-counter"
        return 100, "test-counter"

    loop.provider.estimate_prompt_tokens.side_effect = estimate
    loop.provider.chat_with_retry = AsyncMock(side_effect=[
        LLMResponse(content="Current checkpoint.", tool_calls=[]),
        LLMResponse(content="done", tool_calls=[]),
    ])

    result = await loop.process_direct("continue the task", session_key="cli:test")

    assert result.content == "done"
    assert loop.provider.chat_with_retry.await_count == 2
    model_request = loop.provider.chat_with_retry.await_args_list[1].kwargs["messages"]
    assert "Current checkpoint." in model_request[0]["content"]
    assert model_request[1]["content"] == SUMMARY_CONTINUATION_TEXT
    assert model_request[2]["content"] == "continue the task"

    reloaded = loop.sessions.get_or_create("cli:test")
    assert reloaded.messages[0]["content"] == "old-user-0"
    assert reloaded.metadata["_last_summary"]["text"] == "Current checkpoint."
    assert reloaded.messages[reloaded.last_archived]["content"] == (
        SUMMARY_CONTINUATION_TEXT
    )
    assert [message["content"] for message in reloaded.get_history()] == [
        SUMMARY_CONTINUATION_TEXT,
        "continue the task",
        "done",
    ]


@pytest.mark.asyncio
async def test_native_provider_compaction_commits_portable_terminal_checkpoint(
    tmp_path, admitted_async,
) -> None:
    loop = await _make_loop(tmp_path, admitted_async, estimated_tokens=100, context_window_tokens=2_000)
    session = loop.sessions.get_or_create("cli:native")
    session.messages = [
        {"role": "user", "content": "accepted history"},
        {"role": "assistant", "content": "accepted answer"},
    ]
    loop.sessions.save(session)
    compacted_state = ProviderConversationState(
        kind="openai_responses",
        provider="openai:test",
        model="test-model",
        version=1,
        payload={"items": [{"type": "compaction", "encrypted_content": "opaque"}]},
    )
    loop.provider.can_resume_conversation_state.return_value = True
    loop.provider.chat_with_retry = AsyncMock(return_value=LLMResponse(
        content="done",
        provider_state=compacted_state,
        provider_compaction_applied=True,
        provider_compaction_state=compacted_state,
        provider_compaction_scope="current_request",
    ))
    loop.consolidator.summarize_provider_compaction = AsyncMock(
        return_value="portable terminal checkpoint",
    )

    result = await loop.process_direct("continue", session_key="cli:native")

    assert result.content == "done"
    summarize = loop.consolidator.summarize_provider_compaction
    summarize.assert_awaited_once()
    assert summarize.await_args.args[0] == compacted_state
    accepted = summarize.await_args.args[1]
    accepted_contents = [message.get("content") for message in accepted]
    assert "accepted history" in accepted_contents
    assert "accepted answer" in accepted_contents
    assert "continue" in accepted_contents
    assert "done" not in accepted_contents
    reloaded = loop.sessions.get_or_create("cli:native")
    assert reloaded.provider_state is None
    assert reloaded.metadata["_last_summary"]["text"] == (
        "portable terminal checkpoint"
    )
    assert reloaded.messages[reloaded.last_archived]["content"] == (
        SUMMARY_CONTINUATION_TEXT
    )
    assert [message["content"] for message in reloaded.get_history()] == [
        SUMMARY_CONTINUATION_TEXT,
        "done",
    ]
