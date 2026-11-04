"""Public-boundary contract tests for route-specific request preparation.

The user-outcome scenarios and expected results originate from an independent
requirements-only authoring pass.  This file only adapts those cases to the
concrete public provider boundary (fixtures, constructors, and runner wiring).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nanobot.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ProviderCallContext,
)
from nanobot.providers.fallback_provider import FallbackProvider
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.pooled_provider import PooledProvider
from nanobot.providers.provider_policy import ESTIMATED_COMPAT_POLICY, ProviderPolicy
from nanobot.providers.request_budget import ProviderRequestLimits

PRIMARY = {
    "provider": "primary-provider",
    "model": "primary-model",
    "context_window_tokens": 128_000,
    "completion_reserve_tokens": 8_000,
    "accounting_policy": "estimated_compat",
}

FALLBACK = {
    "provider": "fallback-provider",
    "model": "fallback-model",
    "context_window_tokens": 64_000,
    "completion_reserve_tokens": 4_096,
    "accounting_policy": "estimated_compat",
}

WIDE_PRIMARY = {
    **PRIMARY,
    "provider": "wide-provider",
    "model": "wide-model",
    "context_window_tokens": 200_000,
    "completion_reserve_tokens": 16_000,
}

NARROW_FALLBACK = {
    **FALLBACK,
    "provider": "narrow-provider",
    "model": "narrow-model",
    "context_window_tokens": 32_768,
    "completion_reserve_tokens": 2_048,
}

MESSAGES = [{"role": "user", "content": "Prepare an exact request budget."}]
AUXILIARY_MESSAGES = [
    {"role": "user", "content": "Summarize these source documents for context."}
]


class _ScriptedLeaf(OpenAICompatProvider):
    """A real provider leaf with scripted transport outcomes for circuit tests."""

    def __init__(
        self,
        leaf: dict[str, object],
        responses: list[LLMResponse],
        *,
        supports_source_summary: bool = True,
    ) -> None:
        super().__init__(
            api_key="fixture-credential",
            api_base="http://fixture.invalid/v1",
            default_model=str(leaf["model"]),
            provider_name=str(leaf["provider"]),
            policy=(
                ESTIMATED_COMPAT_POLICY
                if supports_source_summary
                else ProviderPolicy()
            ),
        )
        self._responses = list(responses)

    async def chat(self, *args, **kwargs) -> LLMResponse:
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="ok")

    async def chat_with_retry(self, *args, **kwargs) -> LLMResponse:
        return await self.chat(*args, **kwargs)

    async def chat_stream_with_retry(self, *args, **kwargs) -> LLMResponse:
        return await self.chat(*args, **kwargs)


def _generation(leaf: dict[str, object], max_tokens: int | None = None) -> GenerationSettings:
    return GenerationSettings(
        temperature=0.2,
        max_tokens=(
            int(leaf["completion_reserve_tokens"])
            if max_tokens is None
            else max_tokens
        ),
        reasoning_effort="medium",
    )


def _context(leaf: dict[str, object]) -> ProviderCallContext:
    return ProviderCallContext(
        request_limits=ProviderRequestLimits(
            total_tokens=int(leaf["context_window_tokens"]),
            completion_reserve_tokens=int(leaf["completion_reserve_tokens"]),
            safety_buffer_tokens=0,
        ),
        context_window_tokens=int(leaf["context_window_tokens"]),
    )


def _leaf(
    leaf: dict[str, object],
    responses: list[LLMResponse] | None = None,
    *,
    supports_source_summary: bool = True,
) -> _ScriptedLeaf:
    provider = _ScriptedLeaf(leaf, responses or [], supports_source_summary=supports_source_summary)
    provider.generation = _generation(leaf)
    return provider


def build_routing_wrapper(
    primary: dict[str, object],
    fallbacks: list[dict[str, object]],
    *,
    summary_support: list[bool] | None = None,
) -> FallbackProvider:
    support = summary_support or [True, *([True] * len(fallbacks))]
    primary_provider = _leaf(primary, supports_source_summary=support[0])
    fallback_providers = {
        str(leaf["model"]): _leaf(
            leaf,
            supports_source_summary=supports,
        )
        for leaf, supports in zip(fallbacks, support[1:], strict=True)
    }
    presets = [
        SimpleNamespace(
            model=str(leaf["model"]),
            max_tokens=int(leaf["completion_reserve_tokens"]),
            temperature=0.2,
            reasoning_effort=None,
            context_window_tokens=int(leaf["context_window_tokens"]),
        )
        for leaf in fallbacks
    ]
    return FallbackProvider(
        primary_provider,
        presets,
        lambda preset: fallback_providers[str(preset.model)],
    )


def build_pooled_wrapper(
    leaves: list[dict[str, object]],
    *,
    summary_support: list[bool] | None = None,
) -> PooledProvider:
    support = summary_support or [True] * len(leaves)
    return PooledProvider(
        [
            _leaf(leaf, supports_source_summary=supports)
            for leaf, supports in zip(leaves, support, strict=True)
        ],
        [f"fixture-key-{index}" for index in range(len(leaves))],
    )


async def drive_primary_unavailable(wrapper: FallbackProvider) -> None:
    """Make three consecutive fallback-eligible primary failures through chat."""
    for _ in range(3):
        await wrapper.chat_with_retry(
            messages=MESSAGES,
            model=str(PRIMARY["model"]),
            max_tokens=int(PRIMARY["completion_reserve_tokens"]),
        )


def preview(
    wrapper: LLMProvider,
    *,
    messages=None,
    model: str = "primary-model",
    gen: GenerationSettings | None = None,
    context: ProviderCallContext | None = None,
    leaf: dict[str, object] | None = None,
):
    selected = leaf or PRIMARY
    return wrapper.preview_request_budget(
        messages=MESSAGES if messages is None else messages,
        tools=None,
        model=model,
        generation=_generation(selected) if gen is None else gen,
        provider_context=_context(selected) if context is None else context,
        stream=False,
        tool_choice=None,
    )


def assert_preview_describes(
    prepared,
    *,
    purpose: str,
    leaf: dict[str, object],
    fallback_index: int,
) -> None:
    assert prepared is not None
    route = prepared.route
    measurement = prepared.measurement
    assert route.purpose == purpose
    assert route.fallback_index == fallback_index
    assert route.provider == leaf["provider"]
    assert route.model == leaf["model"]
    assert route.context_window_tokens == leaf["context_window_tokens"]
    assert route.completion_reserve_tokens == leaf["completion_reserve_tokens"]
    assert route.accounting_policy == leaf["accounting_policy"]
    assert route.route_token
    assert measurement.identity.provider == leaf["provider"]
    assert measurement.identity.model == leaf["model"]
    assert measurement.identity.transport
    assert isinstance(measurement.input_tokens, int)
    assert measurement.input_tokens > 0
    assert measurement.total_limit_tokens == leaf["context_window_tokens"]
    assert measurement.completion_reserve_tokens == leaf["completion_reserve_tokens"]
    assert measurement.policy == leaf["accounting_policy"]


def assert_route_is_not(prepared, other_leaf: dict[str, object]) -> None:
    assert prepared is not None
    assert prepared.route.provider != other_leaf["provider"]
    assert prepared.route.model != other_leaf["model"]
    assert prepared.measurement.total_limit_tokens != other_leaf["context_window_tokens"]


@pytest.mark.asyncio
async def test_r1_primary_available_main_preview_describes_primary() -> None:
    wrapper = build_routing_wrapper(PRIMARY, [FALLBACK])
    prepared = preview(wrapper)
    assert_preview_describes(
        prepared, purpose="main", leaf=PRIMARY, fallback_index=0,
    )
    assert_route_is_not(prepared, FALLBACK)


@pytest.mark.asyncio
async def test_r2_primary_unavailable_main_preview_describes_first_fallback() -> None:
    primary = _leaf(
        PRIMARY,
        [
            LLMResponse(
                content="rate limit",
                finish_reason="error",
                error_status_code=429,
                error_kind="rate_limit",
            )
            for _ in range(3)
        ],
    )
    primary_provider = primary
    fallback_provider = _leaf(FALLBACK)
    preset = SimpleNamespace(
        model=str(FALLBACK["model"]),
        max_tokens=int(FALLBACK["completion_reserve_tokens"]),
        temperature=0.2,
        reasoning_effort=None,
        context_window_tokens=int(FALLBACK["context_window_tokens"]),
    )
    wrapper = FallbackProvider(primary_provider, [preset], lambda _preset: fallback_provider)
    await drive_primary_unavailable(wrapper)
    prepared = preview(wrapper)
    assert_preview_describes(
        prepared, purpose="main", leaf=FALLBACK, fallback_index=1,
    )
    assert_route_is_not(prepared, PRIMARY)


@pytest.mark.asyncio
async def test_r3_source_summary_preview_stays_on_primary_while_primary_unavailable() -> None:
    primary = _leaf(
        PRIMARY,
        [
            LLMResponse(
                content="rate limit",
                finish_reason="error",
                error_status_code=429,
                error_kind="rate_limit",
            )
            for _ in range(3)
        ],
    )
    fallback_provider = _leaf(FALLBACK)
    preset = SimpleNamespace(
        model=str(FALLBACK["model"]),
        max_tokens=int(FALLBACK["completion_reserve_tokens"]),
        temperature=0.2,
        reasoning_effort=None,
        context_window_tokens=int(FALLBACK["context_window_tokens"]),
    )
    wrapper = FallbackProvider(primary, [preset], lambda _preset: fallback_provider)
    await drive_primary_unavailable(wrapper)

    auxiliary = preview(
        wrapper,
        messages=AUXILIARY_MESSAGES,
        gen=_generation(PRIMARY, max_tokens=512),
        context=ProviderCallContext(
            request_limits=ProviderRequestLimits(
                total_tokens=int(PRIMARY["context_window_tokens"]),
                completion_reserve_tokens=int(PRIMARY["completion_reserve_tokens"]),
                safety_buffer_tokens=0,
            ),
            context_window_tokens=int(PRIMARY["context_window_tokens"]),
            single_attempt=True,
            source_summary_mode="canonical",
        ),
        leaf=PRIMARY,
    )
    assert_preview_describes(
        auxiliary, purpose="source_summary", leaf=PRIMARY, fallback_index=0,
    )
    assert_route_is_not(auxiliary, FALLBACK)

    main = preview(wrapper, context=_context(PRIMARY), leaf=PRIMARY)
    assert main is not None
    assert main.route is not None
    assert main.route.fallback_index == 1


@pytest.mark.asyncio
async def test_r4_total_limit_tokens_follows_the_selected_route() -> None:
    primary = _leaf(
        WIDE_PRIMARY,
        [
            LLMResponse(
                content="rate limit",
                finish_reason="error",
                error_status_code=429,
                error_kind="rate_limit",
            )
            for _ in range(3)
        ],
    )
    fallback_provider = _leaf(NARROW_FALLBACK)
    preset = SimpleNamespace(
        model=str(NARROW_FALLBACK["model"]),
        max_tokens=int(NARROW_FALLBACK["completion_reserve_tokens"]),
        temperature=0.2,
        reasoning_effort=None,
        context_window_tokens=int(NARROW_FALLBACK["context_window_tokens"]),
    )
    wrapper = FallbackProvider(primary, [preset], lambda _preset: fallback_provider)

    primary_preview = preview(
        wrapper,
        model=str(WIDE_PRIMARY["model"]),
        context=_context(WIDE_PRIMARY),
        leaf=WIDE_PRIMARY,
    )
    assert_preview_describes(
        primary_preview, purpose="main", leaf=WIDE_PRIMARY, fallback_index=0,
    )
    assert primary_preview.measurement.total_limit_tokens == 200_000

    await drive_primary_unavailable(wrapper)

    fallback_preview = preview(
        wrapper,
        model=str(WIDE_PRIMARY["model"]),
        context=_context(WIDE_PRIMARY),
        leaf=WIDE_PRIMARY,
    )
    assert_preview_describes(
        fallback_preview, purpose="main", leaf=NARROW_FALLBACK, fallback_index=1,
    )
    assert fallback_preview.measurement.total_limit_tokens == 32_768
    assert fallback_preview.measurement.completion_reserve_tokens == 2_048


def test_r5_wrapper_source_summary_support_reflects_captured_primary() -> None:
    wrapper_a = build_routing_wrapper(
        {**PRIMARY, "provider": "supporting-primary"},
        [{**FALLBACK, "provider": "non-supporting-fallback"}],
        summary_support=[True, False],
    )
    supported = wrapper_a.supports_source_summary
    assert isinstance(supported, bool)
    assert supported is True

    wrapper_b = build_routing_wrapper(
        {**PRIMARY, "provider": "non-supporting-primary"},
        [{**FALLBACK, "provider": "supporting-fallback"}],
        summary_support=[False, True],
    )
    unsupported = wrapper_b.supports_source_summary
    assert isinstance(unsupported, bool)
    assert unsupported is False


def test_r5_pooled_wrapper_supports_source_summary_only_if_every_leaf_does() -> None:
    def leaf(index: int) -> dict[str, object]:
        return {
            **PRIMARY,
            "provider": f"pooled-provider-{index}",
            "model": f"pooled-model-{index}",
        }

    assert build_pooled_wrapper(
        [leaf(0), leaf(1)], summary_support=[True, True],
    ).supports_source_summary is True
    assert build_pooled_wrapper(
        [leaf(0), leaf(1)], summary_support=[True, False],
    ).supports_source_summary is False
    assert build_pooled_wrapper(
        [leaf(0), leaf(1)], summary_support=[False, False],
    ).supports_source_summary is False


@pytest.mark.asyncio
async def test_dispatch_reports_a_route_change_instead_of_using_a_stale_quote() -> None:
    from dataclasses import replace

    from nanobot.providers.base import ProviderRouteChanged

    primary = _leaf(PRIMARY)
    fallback_provider = _leaf(FALLBACK)
    preset = SimpleNamespace(
        model=str(FALLBACK["model"]),
        max_tokens=int(FALLBACK["completion_reserve_tokens"]),
        temperature=0.2,
        reasoning_effort=None,
        context_window_tokens=int(FALLBACK["context_window_tokens"]),
    )
    wrapper = FallbackProvider(primary, [preset], lambda _preset: fallback_provider)
    prepared_route = preview(wrapper)
    assert prepared_route is not None and prepared_route.route is not None
    stale = replace(prepared_route.route, fallback_index=1)
    context = ProviderCallContext(
        request_limits=ProviderRequestLimits(
            total_tokens=int(PRIMARY["context_window_tokens"]),
            completion_reserve_tokens=int(PRIMARY["completion_reserve_tokens"]),
            safety_buffer_tokens=0,
        ),
        context_window_tokens=int(PRIMARY["context_window_tokens"]),
        selected_route=stale,
    )
    with pytest.raises(ProviderRouteChanged):
        await wrapper.chat_with_retry(
            messages=MESSAGES,
            model=str(PRIMARY["model"]),
            max_tokens=int(PRIMARY["completion_reserve_tokens"]),
            provider_context=context,
        )
