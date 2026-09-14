"""Regression checks for the retained non-streaming HTTP API contract."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from nanobot.agent.turn import TurnResult, TurnSource
from nanobot.api.server import create_app

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)

API_KEY = "secret"
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def _agent(content: str) -> MagicMock:
    agent = MagicMock()
    agent._execute_direct = AsyncMock(return_value=TurnResult(content=content))
    agent.aclose = AsyncMock()
    return agent


@pytest_asyncio.fixture
async def aiohttp_client():
    clients: list[TestClient] = []

    async def _make_client(app):
        client = TestClient(TestServer(app))
        await client.start_server()
        clients.append(client)
        return client

    try:
        yield _make_client
    finally:
        for client in clients:
            await client.close()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_stream_false_returns_json_from_the_canonical_result(aiohttp_client) -> None:
    agent = _agent("normal reply")
    client = await aiohttp_client(create_app(agent, model_name="m", api_key=API_KEY))

    response = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
    )

    assert response.status == 200
    assert response.content_type == "application/json"
    assert (await response.json())["choices"][0]["message"]["content"] == "normal reply"
    assert agent._execute_direct.await_args.kwargs["source"] is TurnSource.API


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_stream_default_is_false(aiohttp_client) -> None:
    agent = _agent("default reply")
    client = await aiohttp_client(create_app(agent, model_name="m", api_key=API_KEY))

    response = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status == 200
    assert (await response.json())["choices"][0]["message"]["content"] == "default reply"
    agent._execute_direct.assert_awaited_once()
