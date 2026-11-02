"""One-attempt auxiliary stream routing through provider wrappers."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from nanobot.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderCallContext,
)
from nanobot.providers.fallback_provider import FallbackProvider
from nanobot.providers.legacy_fallback_provider import LegacyFallbackProvider
from nanobot.providers.pooled_provider import PooledProvider
from nanobot.providers.request_budget import LocalRequestUsageReceipt, ProviderRequestLimits


class StreamLeaf(LLMProvider):
    def __init__(self, model: str, *, api_key: str = "key") -> None:
        super().__init__(api_key=api_key, provider_name="fixture")
        self.model = model
        self.calls: list[dict[str, Any]] = []
        self.contexts: list[ProviderCallContext] = []
        self.response = LLMResponse(
            content="summary",
            local_usage_receipt=LocalRequestUsageReceipt(
                provider_scope="private-scope-digest",
                envelope_binding="envelope-digest",
                model=model,
                limits_binding="limits-digest",
                messages=(),
            ),
        )

    def get_default_model(self) -> str:
        return self.model

    async def chat(self, **kwargs: Any) -> LLMResponse:
        _ = kwargs
        return self.response

    def supports_native_compaction(self, model: str | None = None) -> bool:
        return True

    async def chat_stream(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(dict(kwargs))
        await kwargs["on_content_delta"]("summary") if kwargs.get("on_content_delta") else None
        return self.response

    async def chat_stream_with_context(
        self,
        *,
        provider_context: ProviderCallContext,
        **kwargs: Any,
    ) -> LLMResponse:
        self.contexts.append(provider_context)
        return await self.chat_stream(**kwargs)


def _context() -> ProviderCallContext:
    return ProviderCallContext(
        request_limits=ProviderRequestLimits(
            total_tokens=8_192,
            completion_reserve_tokens=256,
            safety_buffer_tokens=12,
        ),
        context_window_tokens=8_192,
        stream_idle_timeout_seconds=300.0,
        request_deadline_monotonic=time.monotonic() + 900.0,
        source_summary_mode="prefix",
        local_usage_anchor=None,
    )


@pytest.mark.asyncio
async def test_base_once_stream_is_observed_and_preserves_context() -> None:
    leaf = StreamLeaf("leaf-model")
    observed: list[object] = []
    leaf.set_llm_call_observer(observed.append)
    response = await leaf.chat_stream_once(
        messages=[{"role": "user", "content": "bounded"}],
        provider_context=_context(),
        model="leaf-model",
    )

    assert response.content == "summary"
    assert len(leaf.calls) == 1
    assert len(leaf.contexts) == 1
    context = leaf.contexts[0]
    assert context.single_attempt is True
    assert context.stream_idle_timeout_seconds == 300.0
    assert context.source_summary_mode == "prefix"
    assert len(observed) == 1


@pytest.mark.asyncio
async def test_fallback_once_uses_primary_even_when_circuit_is_open() -> None:
    primary = StreamLeaf("primary-model")
    fallback = StreamLeaf("fallback-model")
    factory_calls: list[object] = []
    wrapper = FallbackProvider(
        primary,
        [SimpleNamespace(model="fallback-model", context_window_tokens=64)],
        lambda preset: factory_calls.append(preset) or fallback,
        primary_context_window_tokens=512,
    )
    wrapper._primary_tripped_at = time.monotonic()

    response = await wrapper.chat_stream_once(
        messages=[{"role": "user", "content": "bounded"}],
        provider_context=_context(),
    )

    assert response.local_usage_receipt is None
    assert len(primary.calls) == 1
    assert fallback.calls == []
    assert factory_calls == []
    context = primary.contexts[0]
    assert context.request_limits is not None
    assert context.request_limits.model_context_window_tokens == 512
    assert context.request_deadline_monotonic is not None
    assert context.single_attempt is True


@pytest.mark.asyncio
async def test_legacy_once_uses_primary_model_and_drops_leaf_receipt() -> None:
    primary = StreamLeaf("primary-model")
    fallback = StreamLeaf("fallback-model")
    wrapper = LegacyFallbackProvider(
        primary,
        [SimpleNamespace(model="fallback-model", context_window_tokens=64)],
        lambda _preset: fallback,
        primary_context_window_tokens=256,
    )

    response = await wrapper.chat_stream_once(
        messages=[{"role": "user", "content": "bounded"}],
        provider_context=_context(),
        model=None,
    )

    assert response.local_usage_receipt is None
    assert len(primary.calls) == 1
    assert fallback.calls == []
    assert primary.contexts[0].request_limits is not None
    assert primary.contexts[0].request_limits.model_context_window_tokens == 256
    assert primary.contexts[0].single_attempt is True


@pytest.mark.asyncio
async def test_pool_once_paces_one_leaf_and_never_rotates_after_error() -> None:
    first = StreamLeaf("pool-model", api_key="first")
    second = StreamLeaf("pool-model", api_key="second")
    pool = PooledProvider([first, second], ["first", "second"])
    first.response = LLMResponse(
        content="error",
        finish_reason="error",
        error_status_code=429,
        local_usage_receipt=first.response.local_usage_receipt,
    )

    response = await pool.chat_stream_once(
        messages=[{"role": "user", "content": "bounded"}],
        provider_context=_context(),
    )

    assert response.finish_reason == "error"
    assert response.local_usage_receipt is None
    assert len(first.calls) == 1
    assert second.calls == []
    assert first.contexts[0].single_attempt is True
    assert first.contexts[0].stream_idle_timeout_seconds == 300.0


@pytest.mark.asyncio
async def test_nested_pool_and_fallback_once_still_reaches_one_primary() -> None:
    primary = StreamLeaf("primary-model", api_key="primary")
    fallback_leaf = StreamLeaf("fallback-model", api_key="fallback")
    fallback = FallbackProvider(
        primary,
        [SimpleNamespace(model="fallback-model", context_window_tokens=128)],
        lambda _preset: fallback_leaf,
    )
    pool = PooledProvider([fallback], ["primary"])

    response = await pool.chat_stream_once(
        messages=[{"role": "user", "content": "bounded"}],
        provider_context=_context(),
    )

    assert response.content == "summary"
    assert len(primary.calls) == 1
    assert fallback_leaf.calls == []
    assert primary.contexts[0].single_attempt is True
    assert primary.contexts[0].request_deadline_monotonic is not None
