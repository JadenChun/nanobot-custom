"""Provider policy resolution independent of provider and model names."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from nanobot.config.schema import Config, ProviderConfig
from nanobot.providers.base import ProviderCallContext
from nanobot.providers.capabilities import default_model_capabilities
from nanobot.providers.factory import build_provider_snapshot
from nanobot.providers.image_pricing import ImageOccurrence, image_request_pricing
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.provider_policy import (
    DEFAULT_ESTIMATED_COMPAT_MAX_REQUEST_BYTES,
    ESTIMATED_COMPAT_POLICY,
    ProviderPolicy,
    RequestAccountingPolicy,
    resolve_provider_policy,
)
from nanobot.providers.registry import (
    ProviderSpec,
    create_dynamic_spec,
    find_by_name,
)
from nanobot.providers.request_budget import (
    ProviderRequestLimits,
    RequestPayloadBytesExceeded,
    TransportIdentity,
    measure_finalized_request,
)

MODEL = "arbitrary-model"


def _spec(name: str, policy: ProviderPolicy) -> ProviderSpec:
    return ProviderSpec(name=name, keywords=(), env_key="", policy=policy)


def _provider(name: str, policy: ProviderPolicy) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        api_key="fixture",
        api_base=f"http://{name}.invalid/v1",
        default_model=MODEL,
        provider_name=name,
        spec=_spec(name, policy),
    )


def _body(provider: OpenAICompatProvider) -> dict[str, object]:
    return provider._build_kwargs(
        [{"role": "user", "content": "hello"}],
        None,
        MODEL,
        1,
        0.0,
        None,
        None,
    )


def _context(total: int = 8_192) -> ProviderCallContext:
    return ProviderCallContext(
        request_limits=ProviderRequestLimits(
            total_tokens=total,
            completion_reserve_tokens=1,
            safety_buffer_tokens=0,
        )
    )


def test_absent_policy_is_strict() -> None:
    policy = resolve_provider_policy()

    assert policy.request_accounting is RequestAccountingPolicy.STRICT
    assert policy.max_request_bytes is None
    assert policy.supports_usage_anchor is False
    assert policy.supports_source_summary is False
    assert policy.overflow_detection == "strict"
    assert policy.image_pricing == "auto"


def test_estimated_compat_enables_one_coherent_policy() -> None:
    policy = resolve_provider_policy(request_accounting="estimated_compat")

    assert policy.request_accounting is RequestAccountingPolicy.ESTIMATED_COMPAT
    assert policy.max_request_bytes == DEFAULT_ESTIMATED_COMPAT_MAX_REQUEST_BYTES
    assert policy.supports_usage_anchor is True
    assert policy.supports_source_summary is True
    assert policy.overflow_detection == "compat_400"
    assert policy.image_pricing == "auto"


def test_explicit_cap_and_strict_override() -> None:
    capped = resolve_provider_policy(
        request_accounting="estimated_compat",
        max_request_bytes=1_048_576,
    )
    assert capped.max_request_bytes == 1_048_576

    strict = resolve_provider_policy(
        ESTIMATED_COMPAT_POLICY,
        request_accounting="strict",
    )
    assert strict.request_accounting is RequestAccountingPolicy.STRICT
    assert strict.max_request_bytes is None
    assert strict.supports_usage_anchor is False
    assert strict.supports_source_summary is False
    assert strict.overflow_detection == "strict"
    assert strict.image_pricing == "local_generic"


def test_registry_preset_declares_local_compatibility() -> None:
    local = find_by_name("local_manager")
    assert local is not None
    assert local.policy == ESTIMATED_COMPAT_POLICY

    dynamic = create_dynamic_spec("acme-local")
    assert dynamic.policy.request_accounting is RequestAccountingPolicy.STRICT
    assert dynamic.policy.max_request_bytes is None


def test_provider_config_accepts_policy_overrides_and_rejects_invalid_cap() -> None:
    camel = ProviderConfig(
        requestAccounting="estimated_compat",
        maxRequestBytes=1_048_576,
    )
    snake = ProviderConfig(
        request_accounting="estimated_compat",
        max_request_bytes=1_048_576,
    )
    assert camel.request_accounting == snake.request_accounting == "estimated_compat"
    assert camel.max_request_bytes == snake.max_request_bytes == 1_048_576

    for value in (0, -1):
        with pytest.raises(ValidationError):
            ProviderConfig(maxRequestBytes=value)


def test_custom_provider_opts_into_policy_through_config_only() -> None:
    config = Config.model_validate({
        "providers": {
            "acme_local": {
                "apiBase": "http://127.0.0.1:9999/v1",
                "requestAccounting": "estimated_compat",
                "maxRequestBytes": 2_097_152,
            },
        },
        "modelPresets": {
            "primary": {
                "provider": "acme_local",
                "model": MODEL,
                "maxTokens": 128,
                "contextWindowTokens": 8_192,
            },
        },
    })

    snapshot = build_provider_snapshot(config, preset_name="primary")

    assert snapshot.policy is not None
    assert snapshot.policy.request_accounting is RequestAccountingPolicy.ESTIMATED_COMPAT
    assert snapshot.policy.max_request_bytes == 2_097_152
    assert snapshot.provider.policy == snapshot.policy


def test_custom_provider_without_policy_fields_stays_strict() -> None:
    config = Config.model_validate({
        "providers": {"acme_local": {"apiBase": "http://127.0.0.1:9999/v1"}},
        "modelPresets": {
            "primary": {
                "provider": "acme_local",
                "model": MODEL,
                "maxTokens": 128,
                "contextWindowTokens": 8_192,
            },
        },
    })

    snapshot = build_provider_snapshot(config, preset_name="primary")

    assert snapshot.policy == ProviderPolicy()


def test_accounting_is_selected_by_policy_not_provider_name() -> None:
    estimated_a = _provider("acme-local", ESTIMATED_COMPAT_POLICY)
    estimated_b = _provider("other-vendor", ESTIMATED_COMPAT_POLICY)
    strict = _provider("acme-local-strict", ProviderPolicy())

    assert estimated_a._request_budget_policy == "estimated_compat"
    assert estimated_b._request_budget_policy == "estimated_compat"
    assert strict._request_budget_policy == "strict"

    assert estimated_a.supports_usage_anchor is True
    assert estimated_a.supports_source_summary is True
    assert strict.supports_usage_anchor is False
    assert strict.supports_source_summary is False

    first = estimated_a.validate_finalized_request(
        _body(estimated_a),
        transport="openai_chat_completions",
        model=MODEL,
        provider_context=_context(),
        policy=estimated_a._request_budget_policy,
    )
    second = estimated_b.validate_finalized_request(
        _body(estimated_b),
        transport="openai_chat_completions",
        model=MODEL,
        provider_context=_context(),
        policy=estimated_b._request_budget_policy,
    )

    assert first is not None and second is not None
    assert first.input_tokens == second.input_tokens
    assert first.source == second.source
    assert first.limit_bytes == second.limit_bytes == (
        DEFAULT_ESTIMATED_COMPAT_MAX_REQUEST_BYTES
    )


def test_explicit_byte_cap_applies_to_arbitrary_provider_name() -> None:
    policy = replace(ESTIMATED_COMPAT_POLICY, max_request_bytes=1_048_576)
    provider = _provider("acme-local", policy)
    body = _body(provider)
    encoded_size = len(json.dumps(
        body,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8"))
    provider.policy = replace(provider.policy, max_request_bytes=encoded_size)

    measured = provider.validate_finalized_request(
        body,
        transport="openai_chat_completions",
        model=MODEL,
        provider_context=None,
        policy=provider._request_budget_policy,
    )
    assert measured is not None
    assert measured.limit_bytes == encoded_size

    provider.policy = replace(provider.policy, max_request_bytes=encoded_size - 1)
    with pytest.raises(RequestPayloadBytesExceeded):
        provider.validate_finalized_request(
            body,
            transport="openai_chat_completions",
            model=MODEL,
            provider_context=None,
            policy=provider._request_budget_policy,
        )


def test_image_pricing_mode_is_independent_of_provider_name() -> None:
    occurrence = ImageOccurrence(800, 600, "high", False, 20_000)
    local = image_request_pricing("acme-local", "unknown-model", mode="local_generic")
    assert local is not None
    prices = local.price((occurrence,))
    assert len(prices) == 1
    assert prices[0].visual_tokens > 0

    assert image_request_pricing("acme-local", "unknown-model", mode="none") is None
    assert image_request_pricing("acme-local", "unknown-model") is None


def test_capabilities_preserve_route_facts_with_explicit_pricing_mode() -> None:
    capabilities = default_model_capabilities(
        "acme-local",
        "unknown-model",
        context_window_tokens=32_768,
        max_output_tokens=4_096,
        input_modalities=("text", "image"),
        output_modalities=("text",),
        image_pricing_mode="local_generic",
    )

    assert capabilities.context_window_tokens == 32_768
    assert capabilities.max_output_tokens == 4_096
    assert capabilities.input_modalities == frozenset({"text", "image"})
    assert capabilities.output_modalities == frozenset({"text"})
    assert capabilities.image_pricing is not None


def test_default_capabilities_follow_registry_policy() -> None:
    local = default_model_capabilities(
        "local_manager",
        "unknown-local-model",
        context_window_tokens=8_192,
    )
    assert local.image_pricing is not None

    hosted_unknown = default_model_capabilities(
        "acme-hosted",
        "unknown-model",
        context_window_tokens=8_192,
    )
    assert hosted_unknown.image_pricing is None


def test_provider_capabilities_use_declared_image_policy() -> None:
    provider = _provider(
        "acme-local",
        ProviderPolicy(image_pricing="local_generic"),
    )

    capabilities = provider.model_capabilities(
        "unknown-model",
        context_window_tokens=32_768,
        max_output_tokens=4_096,
    )

    assert capabilities.context_window_tokens == 32_768
    assert capabilities.max_output_tokens == 4_096
    assert capabilities.image_pricing is not None


def test_measurement_accepts_arbitrary_estimated_provider_identity() -> None:
    identity = TransportIdentity(
        "acme-local",
        "openai_chat_completions",
        "unknown-model",
    )
    measurement = measure_finalized_request(
        identity,
        {"model": "unknown-model", "messages": [{"role": "user", "content": "hello"}]},
        ProviderRequestLimits(total_tokens=8_192, safety_buffer_tokens=0),
        policy="estimated_compat",
        max_request_bytes=1_048_576,
    )

    assert measurement.identity == identity
    assert measurement.policy == "estimated_compat"
    assert measurement.limit_bytes == 1_048_576
    assert measurement.source.startswith("estimated_compat")
