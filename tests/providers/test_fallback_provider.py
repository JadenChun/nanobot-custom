"""Tests for FallbackProvider."""

import pytest

from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse
from nanobot.providers.fallback_provider import FallbackProvider


class ScriptedProvider(LLMProvider):
    """A provider that returns pre-scripted responses, tracking calls."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self._responses = list(responses)
        self.calls = 0
        self.last_model: str | None = None

    async def chat(self, *args, **kwargs) -> LLMResponse:
        self.calls += 1
        self.last_model = kwargs.get("model")
        return self._responses.pop(0)

    def get_default_model(self) -> str:
        return "scripted-model"


# ---------------------------------------------------------------------------
# Basic fallback tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_primary_succeeds_no_fallback() -> None:
    """When primary succeeds, fallback is never called."""
    primary = ScriptedProvider([LLMResponse(content="ok")])
    fallback = ScriptedProvider([LLMResponse(content="fallback")])

    provider = FallbackProvider([(primary, "model-a"), (fallback, "model-b")])

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert response.content == "ok"
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_fallback_on_quota_exhaustion() -> None:
    """Quota exhaustion on primary triggers fallback."""
    primary = ScriptedProvider([
        LLMResponse(
            content="Error: All configured API keys were rate-limited or out of quota after trying 1 keys.",
            finish_reason="error",
        ),
    ])
    fallback = ScriptedProvider([LLMResponse(content="fallback ok")])

    provider = FallbackProvider([(primary, "model-a"), (fallback, "model-b")])

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert response.content == "fallback ok"
    assert response.finish_reason == "stop"
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_fallback_on_402_payment_required() -> None:
    """HTTP 402 / payment required triggers fallback."""
    primary = ScriptedProvider([
        LLMResponse(content="Error: 402 Payment Required", finish_reason="error"),
    ])
    fallback = ScriptedProvider([LLMResponse(content="ok")])

    provider = FallbackProvider([(primary, "model-a"), (fallback, "model-b")])

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert response.content == "ok"
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_fallback_on_insufficient_quota() -> None:
    """'insufficient_quota' error triggers fallback."""
    primary = ScriptedProvider([
        LLMResponse(content="Error: insufficient_quota", finish_reason="error"),
    ])
    fallback = ScriptedProvider([LLMResponse(content="ok")])

    provider = FallbackProvider([(primary, "model-a"), (fallback, "model-b")])

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert response.content == "ok"


@pytest.mark.asyncio
async def test_no_fallback_on_non_quota_error() -> None:
    """Non-quota errors (e.g. 401 unauthorized) are returned immediately."""
    primary = ScriptedProvider([
        LLMResponse(content="401 unauthorized", finish_reason="error"),
    ])
    fallback = ScriptedProvider([LLMResponse(content="fallback")])

    provider = FallbackProvider([(primary, "model-a"), (fallback, "model-b")])

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert response.content == "401 unauthorized"
    assert response.finish_reason == "error"
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_all_providers_exhausted() -> None:
    """When all providers return quota errors, last error is returned."""
    primary = ScriptedProvider([
        LLMResponse(content="Error: out of quota", finish_reason="error"),
    ])
    fallback = ScriptedProvider([
        LLMResponse(content="Error: quota exceeded on fallback", finish_reason="error"),
    ])

    provider = FallbackProvider([(primary, "model-a"), (fallback, "model-b")])

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert response.finish_reason == "error"
    assert "quota exceeded on fallback" in response.content
    assert primary.calls == 1
    assert fallback.calls == 1


# ---------------------------------------------------------------------------
# Model override
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_override_per_provider() -> None:
    """Each provider in the chain receives its own model name."""
    primary = ScriptedProvider([
        LLMResponse(content="Error: out of quota", finish_reason="error"),
    ])
    fallback = ScriptedProvider([LLMResponse(content="ok")])

    provider = FallbackProvider([
        (primary, "qwen/qwen3-235b-a22b:free"),
        (fallback, "gemini-2.5-flash"),
    ])

    await provider.chat_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert primary.last_model == "qwen/qwen3-235b-a22b:free"
    assert fallback.last_model == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_explicit_model_overrides_provider_model() -> None:
    """An explicit model= kwarg overrides the per-provider model."""
    primary = ScriptedProvider([LLMResponse(content="ok")])

    provider = FallbackProvider([(primary, "default-model")])

    await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hi"}],
        model="explicit-model",
    )

    assert primary.last_model == "explicit-model"


@pytest.mark.asyncio
async def test_explicit_model_only_applies_to_primary_provider() -> None:
    """Fallback entries keep their configured model even when the caller passes model=."""
    primary = ScriptedProvider([
        LLMResponse(content="Error: out of quota", finish_reason="error"),
    ])
    fallback = ScriptedProvider([LLMResponse(content="ok")])

    provider = FallbackProvider([(primary, "primary-default"), (fallback, "fallback-model")])

    await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hi"}],
        model="explicit-primary-model",
    )

    assert primary.last_model == "explicit-primary-model"
    assert fallback.last_model == "fallback-model"


# ---------------------------------------------------------------------------
# Streaming fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_fallback_on_quota_exhaustion() -> None:
    """Streaming calls also fall back on quota exhaustion."""
    primary = ScriptedProvider([
        LLMResponse(content="Error: out of quota", finish_reason="error"),
    ])
    fallback = ScriptedProvider([LLMResponse(content="streamed ok")])

    provider = FallbackProvider([(primary, "model-a"), (fallback, "model-b")])

    response = await provider.chat_stream_with_retry(
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.content == "streamed ok"
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_streaming_no_fallback_on_non_quota_error() -> None:
    """Streaming non-quota errors are returned without fallback."""
    primary = ScriptedProvider([
        LLMResponse(content="401 unauthorized", finish_reason="error"),
    ])
    fallback = ScriptedProvider([LLMResponse(content="fallback")])

    provider = FallbackProvider([(primary, "model-a"), (fallback, "model-b")])

    response = await provider.chat_stream_with_retry(
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response.content == "401 unauthorized"
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_streaming_explicit_model_only_applies_to_primary_provider() -> None:
    """Streaming fallback entries keep their configured model even with model=."""
    primary = ScriptedProvider([
        LLMResponse(content="Error: out of quota", finish_reason="error"),
    ])
    fallback = ScriptedProvider([LLMResponse(content="streamed ok")])

    provider = FallbackProvider([(primary, "primary-default"), (fallback, "fallback-model")])

    await provider.chat_stream_with_retry(
        messages=[{"role": "user", "content": "hi"}],
        model="explicit-primary-model",
    )

    assert primary.last_model == "explicit-primary-model"
    assert fallback.last_model == "fallback-model"


# ---------------------------------------------------------------------------
# Generation settings propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_settings_propagation() -> None:
    """Setting generation on FallbackProvider propagates to all wrapped providers."""
    primary = ScriptedProvider([LLMResponse(content="ok")])
    fallback = ScriptedProvider([LLMResponse(content="ok")])

    provider = FallbackProvider([(primary, "model-a"), (fallback, "model-b")])
    gen = GenerationSettings(temperature=0.2, max_tokens=512, reasoning_effort="high")
    provider.generation = gen

    assert primary.generation == gen
    assert fallback.generation == gen
    assert provider.generation == gen


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_provider_no_fallback_available() -> None:
    """With only one provider, quota exhaustion returns the error."""
    primary = ScriptedProvider([
        LLMResponse(content="Error: out of quota", finish_reason="error"),
    ])

    provider = FallbackProvider([(primary, "model-a")])

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert response.finish_reason == "error"
    assert "out of quota" in response.content


@pytest.mark.asyncio
async def test_three_provider_chain() -> None:
    """Fallback chains through multiple providers."""
    p1 = ScriptedProvider([
        LLMResponse(content="Error: out of quota", finish_reason="error"),
    ])
    p2 = ScriptedProvider([
        LLMResponse(content="Error: billing limit reached", finish_reason="error"),
    ])
    p3 = ScriptedProvider([LLMResponse(content="third time's the charm")])

    provider = FallbackProvider([
        (p1, "model-1"), (p2, "model-2"), (p3, "model-3"),
    ])

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hi"}])

    assert response.content == "third time's the charm"
    assert p1.calls == 1
    assert p2.calls == 1
    assert p3.calls == 1


def test_empty_provider_list_raises() -> None:
    """FallbackProvider requires at least one provider."""
    with pytest.raises(ValueError, match="at least one provider"):
        FallbackProvider([])


@pytest.mark.asyncio
async def test_get_default_model() -> None:
    """get_default_model returns the primary provider's model."""
    primary = ScriptedProvider([])
    fallback = ScriptedProvider([])

    provider = FallbackProvider([(primary, "primary-model"), (fallback, "fallback-model")])

    assert provider.get_default_model() == "primary-model"
