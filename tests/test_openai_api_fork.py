"""Direct handler checks that do not require an aiohttp loopback listener."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.turn import TurnResult, TurnSource
from nanobot.api.server import API_SESSION_KEY, handle_chat_completions
from nanobot.providers.base import LLMUsage


def _request(body: dict, agent: MagicMock) -> MagicMock:
    request = MagicMock()
    request.content_type = "application/json"
    request.json = AsyncMock(return_value=body)
    request.app = {
        "agent_loop": agent,
        "model_name": "test-model",
        "request_timeout": 1.0,
    }
    return request


@pytest.mark.asyncio
async def test_direct_handler_builds_one_api_canonical_turn() -> None:
    usage = LLMUsage.reported(input_tokens=9, output_tokens=4)
    agent = MagicMock()
    agent._execute_direct = AsyncMock(return_value=TurnResult(content="answer", llm_usage=usage))

    response = await handle_chat_completions(_request(
        {"messages": [{"role": "user", "content": "question"}]},
        agent,
    ))

    assert response.status == 200
    assert json.loads(response.body)["usage"] == {
        "prompt_tokens": 9,
        "completion_tokens": 4,
        "total_tokens": 13,
    }
    args = agent._execute_direct.await_args
    assert args.args == ("question",)
    assert args.kwargs["session_key"] == API_SESSION_KEY
    assert args.kwargs["source"] is TurnSource.API
    assert args.kwargs["channel"] == "api"
    assert args.kwargs["chat_id"] == "default"


@pytest.mark.asyncio
async def test_direct_handler_rejects_stream_before_turn_admission() -> None:
    agent = MagicMock()
    agent._execute_direct = AsyncMock()

    response = await handle_chat_completions(_request(
        {"messages": [{"role": "user", "content": "question"}], "stream": True},
        agent,
    ))

    assert response.status == 400
    assert "stream" in json.loads(response.body)["error"]["message"].lower()
    agent._execute_direct.assert_not_awaited()
