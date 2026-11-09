"""Regression tests for prepared route fallback handoff and final limits."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from nanobot.agent.runner import AgentRunner
from nanobot.providers.base import LLMResponse, ProviderCallContext
from nanobot.providers.fallback_provider import FallbackProvider
from nanobot.providers.request_budget import ProviderRouteChanged
from tests.agent.runner_helpers import make_run_spec
from tests.providers.test_route_preview_contract import (
    FALLBACK,
    PRIMARY,
    _leaf,
    drive_primary_unavailable,
    preview,
)


def _fallback_preset(leaf: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        model=str(leaf["model"]),
        max_tokens=int(leaf["completion_reserve_tokens"]),
        temperature=0.2,
        reasoning_effort=None,
        context_window_tokens=int(leaf["context_window_tokens"]),
    )


def _rate_limit_response() -> LLMResponse:
    return LLMResponse(
        content="rate limit",
        finish_reason="error",
        error_status_code=429,
        error_kind="rate_limit",
    )


class _CountingProxy:
    """Count each provider entry point before delegating to the scripted leaf."""

    def __init__(self, leaf: Any) -> None:
        self._leaf = leaf
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._leaf, name)

    async def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return await self._leaf.chat(*args, **kwargs)

    async def chat_with_retry(self, *args: Any, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return await self._leaf.chat_with_retry(*args, **kwargs)

    async def chat_stream_with_retry(self, *args: Any, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return await self._leaf.chat_stream_with_retry(*args, **kwargs)


def _tools() -> MagicMock:
    tools = MagicMock()
    tools.get_definitions.return_value = []
    return tools


@pytest.mark.asyncio
async def test_prepared_primary_failure_reaches_fallback_once() -> None:
    primary = _CountingProxy(_leaf(PRIMARY, [_rate_limit_response()]))
    fallback = _CountingProxy(_leaf(FALLBACK, [LLMResponse(content="fallback ok")]))
    wrapper = FallbackProvider(
        primary,
        [_fallback_preset(FALLBACK)],
        lambda _preset: fallback,
    )

    result = await AgentRunner().run(
        make_run_spec(
            wrapper,
            model=str(PRIMARY["model"]),
            max_iterations=2,
            max_tool_result_chars=1000,
            tools=_tools(),
            initial_messages=[{"role": "user", "content": "hello"}],
        )
    )

    assert result.final_content == "fallback ok"
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_pre_dispatch_route_change_does_not_consume_skipped_primary() -> None:
    # The helper opens the circuit only after three fallback-eligible failures,
    # so the primary leaf must be scripted with those failures.
    primary = _CountingProxy(
        _leaf(PRIMARY, [_rate_limit_response() for _ in range(3)])
    )
    fallback = _CountingProxy(_leaf(FALLBACK))
    wrapper = FallbackProvider(
        primary,
        [_fallback_preset(FALLBACK)],
        lambda _preset: fallback,
    )

    prepared = preview(wrapper)
    assert prepared is not None
    assert prepared.route is not None
    assert prepared.route.fallback_index == 0

    await drive_primary_unavailable(wrapper)

    context = ProviderCallContext(selected_route=prepared.route)
    with pytest.raises(ProviderRouteChanged) as changed:
        await wrapper.chat_with_retry(
            messages=[{"role": "user", "content": "hello"}],
            model=str(PRIMARY["model"]),
            max_tokens=int(PRIMARY["completion_reserve_tokens"]),
            provider_context=context,
        )

    assert changed.value.actual_fallback_index == 1
    assert changed.value.consumed_through == -1

    primary.calls = 0
    fallback.calls = 0
    retry_context = replace(
        context,
        selected_route=None,
        route_candidate_index=1,
        route_consumed_through=-1,
    )
    response = await wrapper.chat_with_retry(
        messages=[{"role": "user", "content": "hello"}],
        model=str(PRIMARY["model"]),
        max_tokens=int(PRIMARY["completion_reserve_tokens"]),
        provider_context=retry_context,
    )

    assert response.content == "ok"
    assert primary.calls == 0
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_fallback_final_request_limits_reject_before_dispatch() -> None:
    primary_leaf = {
        **PRIMARY,
        "provider": "narrow-primary",
        "model": "narrow-primary-model",
        "context_window_tokens": 2000,
        "completion_reserve_tokens": 256,
    }
    fallback_leaf = {
        **FALLBACK,
        "provider": "narrow-fallback",
        "model": "narrow-fallback-model",
        "context_window_tokens": 500,
        "completion_reserve_tokens": 128,
    }
    primary = _CountingProxy(_leaf(primary_leaf, [_rate_limit_response()]))
    fallback = _CountingProxy(_leaf(fallback_leaf))
    wrapper = FallbackProvider(
        primary,
        [_fallback_preset(fallback_leaf)],
        lambda _preset: fallback,
        primary_context_window_tokens=2000,
    )

    # 400 words estimates to about 492 input tokens.  It fits the primary
    # window (2000 tokens) but exceeds the fallback input limit (500 - 128).
    result = await AgentRunner().run(
        make_run_spec(
            wrapper,
            model=str(primary_leaf["model"]),
            max_iterations=2,
            max_tool_result_chars=1000,
            tools=_tools(),
            initial_messages=[{"role": "user", "content": "word " * 400}],
        )
    )

    assert primary.calls == 1
    assert fallback.calls == 0
    assert result.stop_reason == "error" or result.error is not None
