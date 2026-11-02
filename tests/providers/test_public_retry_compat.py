"""Public retry compatibility when callers omit provider context.

Ordinary ``chat``/``chat_with_retry`` and streaming callers that omit
``provider_context`` must keep the normal Responses-to-Chat fallback. The
explicit once routes must remain single-attempt.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nanobot.providers.base import ProviderCallContext
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import find_by_name


def _chat_response(content: str = "from chat") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None, reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
    )


def _chat_stream(content: str = "from chat"):
    async def _stream():
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason=None,
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                delta=SimpleNamespace(
                    content=None,
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )],
            usage=None,
        )

    return _stream()


class _ResponsesUnsupportedError(Exception):
    def __init__(self) -> None:
        super().__init__("Responses endpoint not supported")
        self.status_code = 404
        self.response = SimpleNamespace(
            status_code=404,
            text="Responses endpoint not supported",
            headers={},
        )


def _provider() -> OpenAICompatProvider:
    return OpenAICompatProvider(
        api_key="sk-test-key",
        default_model="gpt-5-chat",
        spec=find_by_name("openai"),
    )


def _client() -> SimpleNamespace:
    return SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(side_effect=_ResponsesUnsupportedError())),
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=_chat_response()))
        ),
    )


@pytest.mark.asyncio
async def test_plain_chat_omitted_context_keeps_responses_fallback() -> None:
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as client_class:
        client = _client()
        client_class.return_value = client
        provider = _provider()

        result = await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-5-chat",
        )

    assert result.content == "from chat"
    assert client.responses.create.await_count == 1
    assert client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_chat_once_omitted_context_is_single_attempt() -> None:
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as client_class:
        client = _client()
        client_class.return_value = client
        provider = _provider()

        result = await provider.chat_once(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-5-chat",
        )

    assert result.finish_reason == "error"
    assert client.responses.create.await_count == 1
    assert client.chat.completions.create.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_context", [False, True], ids=["omitted", "explicit-default"])
async def test_chat_with_retry_keeps_responses_fallback(explicit_context: bool) -> None:
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as client_class:
        client = _client()
        client_class.return_value = client
        provider = _provider()

        kwargs = {}
        if explicit_context:
            kwargs["provider_context"] = ProviderCallContext()
        result = await provider.chat_with_retry(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-5-chat",
            **kwargs,
        )

    assert result.content == "from chat"
    assert client.responses.create.await_count == 1
    assert client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_context", [False, True], ids=["omitted", "explicit-default"])
async def test_chat_stream_with_retry_keeps_responses_fallback(
    explicit_context: bool,
) -> None:
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as client_class:
        client = _client()
        client.chat.completions.create = AsyncMock(return_value=_chat_stream())
        client_class.return_value = client
        provider = _provider()

        kwargs = {}
        if explicit_context:
            kwargs["provider_context"] = ProviderCallContext()
        result = await provider.chat_stream_with_retry(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-5-chat",
            **kwargs,
        )

    assert result.content == "from chat"
    assert client.responses.create.await_count == 1
    assert client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_chat_stream_once_omitted_context_is_single_attempt() -> None:
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as client_class:
        client = _client()
        client.chat.completions.create = AsyncMock(return_value=_chat_stream())
        client_class.return_value = client
        provider = _provider()

        result = await provider.chat_stream_once(
            messages=[{"role": "user", "content": "hello"}],
            model="gpt-5-chat",
        )

    assert result.finish_reason == "error"
    assert client.responses.create.await_count == 1
    assert client.chat.completions.create.await_count == 0
