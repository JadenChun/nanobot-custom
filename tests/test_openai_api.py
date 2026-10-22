"""Focused checks for the non-streaming OpenAI-compatible HTTP API."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from nanobot.agent.turn import TraceMode, TurnResult, TurnSource
from nanobot.api.server import (
    API_CHAT_ID,
    API_SESSION_KEY,
    _chat_completion_response,
    _error_json,
    create_app,
    handle_chat_completions,
)
from nanobot.providers.base import LLMUsage
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE
from tests.api_http_client import InProcessApiClient

try:
    from aiohttp import web

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)

API_KEY = "secret"
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def _make_mock_agent(response_text: str = "mock response") -> MagicMock:
    agent = MagicMock()
    agent._execute_direct = AsyncMock(return_value=TurnResult(content=response_text))
    agent.aclose = AsyncMock()
    return agent


@pytest.fixture
def mock_agent() -> MagicMock:
    return _make_mock_agent()


@pytest.fixture
def app(mock_agent: MagicMock):
    return create_app(mock_agent, model_name="test-model", request_timeout=10.0, api_key=API_KEY)


@pytest_asyncio.fixture
async def aiohttp_client():
    clients: list[InProcessApiClient] = []

    async def _make_client(app: web.Application):
        client = InProcessApiClient(app)
        clients.append(client)
        await client.start()
        return client

    try:
        yield _make_client
    finally:
        for client in clients:
            await client.close()


def test_error_json() -> None:
    response = _error_json(400, "bad request")
    assert response.status == 400
    body = json.loads(response.body)
    assert body["error"] == {"message": "bad request", "type": "invalid_request_error", "code": 400}


def test_chat_completion_response_includes_usage() -> None:
    usage = LLMUsage.reported(input_tokens=70, output_tokens=7, total_tokens=175)
    response = _chat_completion_response("hello world", "test-model", usage)
    assert response["object"] == "chat.completion"
    assert response["model"] == "test-model"
    assert response["choices"][0]["message"]["content"] == "hello world"
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"] == {
        "prompt_tokens": 70,
        "completion_tokens": 7,
        "total_tokens": 175,
    }


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_api_key_protects_api_routes_but_not_health(aiohttp_client, mock_agent: MagicMock) -> None:
    client = await aiohttp_client(create_app(mock_agent, model_name="test-model", api_key=API_KEY))

    health = await client.get("/health")
    missing = await client.get("/v1/models")
    wrong = await client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    ok = await client.get("/v1/models", headers=AUTH_HEADERS)

    assert health.status == 200
    assert missing.status == 401
    assert wrong.status == 401
    assert ok.status == 200


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_models_and_health_keep_their_existing_payloads(aiohttp_client, mock_agent: MagicMock) -> None:
    client = await aiohttp_client(create_app(mock_agent, model_name="test-model", api_key=API_KEY))

    models = await client.get("/v1/models", headers=AUTH_HEADERS)
    health = await client.get("/health")

    assert models.status == 200
    assert (await models.json())["data"][0]["id"] == "test-model"
    assert health.status == 200
    assert (await health.json()) == {"status": "ok"}


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_api_routes_remain_open_without_a_configured_key(aiohttp_client, mock_agent: MagicMock) -> None:
    client = await aiohttp_client(create_app(mock_agent, model_name="test-model"))

    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status == 200
    mock_agent._execute_direct.assert_awaited_once()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_preparation_runs_before_each_admitted_turn(aiohttp_client, mock_agent: MagicMock) -> None:
    events: list[str] = []

    async def prepare() -> None:
        events.append("prepare")

    async def execute_direct(content: str, **_kwargs) -> TurnResult:
        events.append(f"execute:{content}")
        return TurnResult(content="ready")

    mock_agent._execute_direct = execute_direct
    client = await aiohttp_client(create_app(mock_agent, prepare_agent=prepare))
    response = await client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status == 200
    assert events == ["prepare", "execute:hello"]


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_json_schema_errors_do_not_admit_a_turn(aiohttp_client, app, mock_agent: MagicMock) -> None:
    client = await aiohttp_client(app)
    for body in (
        {"model": "test"},
        {"messages": [{"role": "system", "content": "not a user"}]},
        {"messages": [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]},
    ):
        response = await client.post("/v1/chat/completions", headers=AUTH_HEADERS, json=body)
        assert response.status == 400
    mock_agent._execute_direct.assert_not_awaited()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_json_stream_true_is_rejected_without_preparation_or_execution(aiohttp_client, mock_agent: MagicMock) -> None:
    prepared = False

    async def prepare() -> None:
        nonlocal prepared
        prepared = True

    client = await aiohttp_client(create_app(mock_agent, api_key=API_KEY, prepare_agent=prepare))
    response = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
    )

    assert response.status == 400
    assert "Streaming is not supported" in (await response.json())["error"]["message"]
    assert not prepared
    mock_agent._execute_direct.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_mismatch_returns_400_without_execution(mock_agent: MagicMock) -> None:
    request = MagicMock()
    request.json = AsyncMock(return_value={
        "model": "other-model",
        "messages": [{"role": "user", "content": "hello"}],
    })
    request.app = {"agent_loop": mock_agent, "model_name": "test-model", "request_timeout": 10.0}

    response = await handle_chat_completions(request)

    assert response.status == 400
    assert "test-model" in json.loads(response.body)["error"]["message"]
    mock_agent._execute_direct.assert_not_awaited()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_request_uses_canonical_result_content_usage_and_api_source(
    aiohttp_client,
    mock_agent: MagicMock,
) -> None:
    usage = LLMUsage.reported(input_tokens=100, output_tokens=50)
    mock_agent._execute_direct.return_value = TurnResult(
        content="canonical response",
        outbound=None,
        llm_usage=usage,
    )
    client = await aiohttp_client(create_app(mock_agent, model_name="test-model", api_key=API_KEY))

    response = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status == 200
    body = await response.json()
    assert body["choices"][0]["message"]["content"] == "canonical response"
    assert body["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }
    args = mock_agent._execute_direct.await_args
    assert args.args == ("hello",)
    assert args.kwargs == {
        "media": None,
        "session_key": API_SESSION_KEY,
        "channel": "api",
        "chat_id": API_CHAT_ID,
        "source": TurnSource.API,
    }
    assert args.kwargs.get("trace_mode", TraceMode.NONE) is TraceMode.NONE


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_optional_session_id_is_routed_to_its_api_session(aiohttp_client) -> None:
    requests = []

    async def execute_direct(content: str, **kwargs):
        requests.append((content, kwargs["session_key"]))
        return TurnResult(content=content)

    agent = _make_mock_agent()
    agent._execute_direct = execute_direct
    client = await aiohttp_client(create_app(agent, model_name="m", api_key=API_KEY))

    first = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "first"}]},
    )
    second = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "second"}], "session_id": "named"},
    )

    assert first.status == 200
    assert second.status == 200
    assert requests == [("first", API_SESSION_KEY), ("second", "api:named")]


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_media_is_forwarded_through_the_canonical_adapter(aiohttp_client, mock_agent: MagicMock) -> None:
    client = await aiohttp_client(create_app(mock_agent, model_name="m", api_key=API_KEY))
    response = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1n"}},
            ]}],
        },
    )

    assert response.status == 200
    args = mock_agent._execute_direct.await_args
    assert args.args == ("describe this",)
    assert len(args.kwargs["media"] or []) == 1
    assert args.kwargs["source"] is TurnSource.API


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_remote_media_returns_400_without_execution(aiohttp_client, mock_agent: MagicMock) -> None:
    client = await aiohttp_client(create_app(mock_agent, model_name="m", api_key=API_KEY))
    response = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        ]}]},
    )

    assert response.status == 400
    assert "remote image urls are not supported" in (await response.json())["error"]["message"].lower()
    mock_agent._execute_direct.assert_not_awaited()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_empty_canonical_result_uses_fallback_after_one_admitted_turn(aiohttp_client) -> None:
    calls = 0

    async def execute_direct(content: str, **_kwargs) -> TurnResult:
        nonlocal calls
        calls += 1
        return TurnResult(content="", outbound=None)

    agent = _make_mock_agent()
    agent._execute_direct = execute_direct
    client = await aiohttp_client(create_app(agent, model_name="m", api_key=API_KEY))

    response = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status == 200
    assert (await response.json())["choices"][0]["message"]["content"] == EMPTY_FINAL_RESPONSE_MESSAGE
    assert calls == 1


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_preparation_and_admission_share_the_request_timeout(aiohttp_client, mock_agent: MagicMock) -> None:
    async def prepare() -> None:
        await asyncio.Event().wait()

    client = await aiohttp_client(create_app(
        mock_agent,
        request_timeout=0.01,
        api_key=API_KEY,
        prepare_agent=prepare,
    ))
    response = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status == 504
    mock_agent._execute_direct.assert_not_awaited()


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_timeout_waiting_for_admission_does_not_poison_the_next_request(aiohttp_client) -> None:
    admission = asyncio.Lock()
    effects: list[str] = []

    async def execute_direct(content: str, **_kwargs) -> TurnResult:
        async with admission:
            effects.append(content)
            return TurnResult(content=content)

    agent = _make_mock_agent()
    agent._execute_direct = execute_direct
    client = await aiohttp_client(create_app(agent, model_name="m", api_key=API_KEY, request_timeout=0.02))
    await admission.acquire()
    try:
        queued = await client.post(
            "/v1/chat/completions",
            headers=AUTH_HEADERS,
            json={"messages": [{"role": "user", "content": "queued"}]},
        )
        assert queued.status == 504
        assert effects == []
    finally:
        admission.release()

    next_response = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "next"}]},
    )
    assert next_response.status == 200
    assert effects == ["next"]


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
async def test_adapter_failure_retains_the_existing_500_response(aiohttp_client, mock_agent: MagicMock) -> None:
    mock_agent._execute_direct.side_effect = RuntimeError("boom")
    client = await aiohttp_client(create_app(mock_agent, api_key=API_KEY))

    response = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status == 500
    assert (await response.json())["error"] == {
        "message": "Internal server error",
        "type": "server_error",
        "code": 500,
    }


def test_api_installs_no_second_session_lock(mock_agent: MagicMock) -> None:
    app = create_app(mock_agent, model_name="test", api_key=API_KEY)
    assert not any("session_lock" in str(key) for key in app.keys())
