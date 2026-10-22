"""Preserve legacy mid-loop context guarantees through the current governor."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context_governance import ContextWindowExceededError
from nanobot.agent.runner import AgentRunner, AgentRunResult
from nanobot.providers.base import LLMProvider, LLMResponse
from tests.agent.runner_helpers import make_run_spec


def _messages() -> list[dict]:
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "old request"},
    ]
    for index in range(3):
        messages.extend((
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{index}", "type": "function",
                    "function": {"name": "dummy", "arguments": "{}"},
                }],
            },
            {
                "role": "tool", "tool_call_id": f"call_{index}",
                "name": "dummy", "content": f"result-{index}",
            },
        ))
    messages.append({"role": "user", "content": "current request"})
    return messages


async def _run(messages: list[dict], *, context_window_tokens: int) -> tuple[MagicMock, AgentRunResult]:
    provider = MagicMock(spec=LLMProvider)
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="done"))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    result = await AgentRunner().run(make_run_spec(
        provider,
        model="test-model",
        initial_messages=messages,
        tools=tools,
        max_iterations=1,
        max_tool_result_chars=2000,
        max_tokens=100,
        context_window_tokens=context_window_tokens,
    ))
    return provider, result


def _pressure_estimate(_provider, _model, messages, _tools):
    if any(message.get("content") == "result-0" for message in messages):
        return 1000, "fixture"
    return 100, "fixture"


def _message_tokens(message):
    if message.get("role") == "system" or message.get("content") == "current request":
        return 10
    return 300


async def test_under_budget_preserves_complete_model_context(monkeypatch) -> None:
    messages = _messages()
    original = deepcopy(messages)
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_prompt_tokens_chain",
        lambda *_args: (100, "fixture"),
    )

    provider, result = await _run(messages, context_window_tokens=1624)

    assert provider.chat_with_retry.await_args.kwargs["messages"] == original
    assert result.messages[:-1] == original
    assert messages == original


async def test_pressure_trims_model_view_but_preserves_source_history(monkeypatch) -> None:
    messages = _messages()
    original = deepcopy(messages)
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_prompt_tokens_chain",
        _pressure_estimate,
    )
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_message_tokens",
        _message_tokens,
    )

    provider, result = await _run(messages, context_window_tokens=1624)

    sent = provider.chat_with_retry.await_args.kwargs["messages"]
    assert sent == [original[0], original[-1]]
    assert result.messages[:-1] == original
    assert messages == original


async def test_intermediate_user_survives_removal_of_older_tool_group(monkeypatch) -> None:
    messages = _messages()[:-1]
    followup = {"role": "user", "content": "intermediate follow-up"}
    messages.insert(4, followup)
    original = deepcopy(messages)
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_prompt_tokens_chain",
        _pressure_estimate,
    )
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_message_tokens",
        lambda message: 300 if message.get("content") == "result-0" else 20,
    )

    provider, result = await _run(messages, context_window_tokens=1624)

    sent = provider.chat_with_retry.await_args.kwargs["messages"]
    assert followup in sent
    sent_call_ids = {call["id"] for message in sent for call in message.get("tool_calls", [])}
    assert all(message.get("tool_call_id") in sent_call_ids
               for message in sent if message.get("role") == "tool")
    assert result.messages[:-1] == original


async def test_unconfigured_context_window_does_not_trim(monkeypatch) -> None:
    messages = _messages()
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_prompt_tokens_chain",
        lambda *_args: (10000, "fixture"),
    )

    provider, _ = await _run(messages, context_window_tokens=0)

    assert provider.chat_with_retry.await_args.kwargs["messages"] == messages


async def test_irreducible_current_request_fails_before_provider(monkeypatch) -> None:
    provider = MagicMock(spec=LLMProvider)
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="unexpected"))
    tools = MagicMock()
    tools.get_definitions.return_value = []
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_prompt_tokens_chain",
        lambda *_args: (2000, "fixture"),
    )

    with pytest.raises(ContextWindowExceededError):
        await AgentRunner().run(make_run_spec(
            provider,
            model="test-model",
            initial_messages=[{"role": "system", "content": "policy"},
                              {"role": "user", "content": "current request"}],
            tools=tools,
            max_iterations=1,
            max_tool_result_chars=2000,
            max_tokens=100,
            context_window_tokens=1624,
        ))
    provider.chat_with_retry.assert_not_awaited()
