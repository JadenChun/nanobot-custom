"""Capability resolution and image pricing at the provider boundary."""

from __future__ import annotations

import base64
import struct
import zlib

import pytest

from nanobot.providers.capabilities import (
    ModelCapabilities,
    default_model_capabilities,
)
from nanobot.providers.image_pricing import ImageOccurrence, ImagePrice
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import ProviderModelSpec, ProviderSpec
from nanobot.providers.request_budget import (
    ProviderRequestLimits,
    TransportIdentity,
    measure_finalized_request,
)
from nanobot.utils.llm_runtime import LLMRuntime


def _png_data_uri(width: int = 64, height: int = 64) -> str:
    def chunk(name: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", zlib.crc32(name + data))
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixels = zlib.compress((b"\x00" + b"\x00" * (width * 3)) * height)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", pixels)
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def _image_body(model: str, width: int = 64, height: int = 64) -> dict[str, object]:
    return {
        "model": model,
        "max_tokens": 1,
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": _png_data_uri(width, height)},
            }],
        }],
    }


@pytest.mark.parametrize(
    "model",
    (
        "Qwen3.8-27B-UD-Q6_K_XL",
        "Qwen3.8-Flash-Next Gyro-M Abliterated (Strata)",
        "unknown-local-vision-model",
    ),
)
def test_local_vision_models_have_measurable_image_pricing(model: str) -> None:
    capabilities = default_model_capabilities(
        "local_manager",
        model,
        context_window_tokens=262_144,
    )

    assert capabilities.image_pricing is not None
    measurement = measure_finalized_request(
        TransportIdentity("local_manager", "openai_chat_completions_stream", model),
        _image_body(model),
        ProviderRequestLimits(
            total_tokens=262_144,
            completion_reserve_tokens=8_192,
            safety_buffer_tokens=1_024,
        ),
        capabilities=capabilities,
    )

    assert measurement.input_tokens > 0
    assert "inline_image" in measurement.source


def test_unknown_local_image_dimensions_use_bounded_occurrence_fallback() -> None:
    capabilities = default_model_capabilities(
        "local_manager",
        "unknown-local-vision-model",
        context_window_tokens=8_192,
    )
    assert capabilities.image_pricing is not None

    prices = capabilities.image_pricing.price((
        ImageOccurrence(
            width=None,
            height=None,
            detail=None,
            remote=False,
            encoded_bytes=2_000,
        ),
    ))

    assert len(prices) == 1
    assert prices[0].visual_tokens > 0


def test_route_pricing_answers_each_occurrence_in_order() -> None:
    class Pricing:
        def __init__(self) -> None:
            self.calls: list[ImageOccurrence] = []

        def price(self, images: object) -> list[ImagePrice]:
            sequence = list(images)  # type: ignore[arg-type]
            self.calls.extend(sequence)
            return [ImagePrice(101), ImagePrice(202)]

    pricing = Pricing()
    capabilities = ModelCapabilities(
        context_window_tokens=8_192,
        max_output_tokens=1_024,
        input_modalities=frozenset({"text", "image"}),
        output_modalities=frozenset({"text"}),
        image_pricing=pricing,
        capability_source="test:explicit",
    )

    body = _image_body("fixture")
    content = body["messages"][0]["content"]  # type: ignore[index]
    content.append({  # type: ignore[union-attr]
        "type": "image_url",
        "image_url": {"url": _png_data_uri(32, 32)},
    })
    measurement = measure_finalized_request(
        TransportIdentity("fixture", "openai_chat_completions", "fixture"),
        body,
        ProviderRequestLimits(total_tokens=8_192, safety_buffer_tokens=0),
        capabilities=capabilities,
    )

    assert measurement.input_tokens > 0
    assert len(pricing.calls) == 2
    assert [(item.width, item.height) for item in pricing.calls] == [(64, 64), (32, 32)]


def test_explicit_text_only_model_sanitizes_images() -> None:
    spec = ProviderSpec(
        name="fixture",
        keywords=(),
        env_key="",
        builtin_models=(
            ProviderModelSpec(
                id="text-only-fixture",
                input_modalities=("text",),
                output_modalities=("text",),
            ),
        ),
    )
    provider = OpenAICompatProvider(
        default_model="text-only-fixture",
        provider_name="fixture",
        spec=spec,
        api_base="http://127.0.0.1:1/v1",
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": _png_data_uri()}},
        ],
    }]

    sanitized = provider._sanitize_messages(messages, "text-only-fixture")

    assert isinstance(sanitized[0]["content"], str)
    assert "describe this" in sanitized[0]["content"]
    assert "image_url" not in sanitized[0]["content"]


def test_runtime_captures_catalog_capabilities() -> None:
    spec = ProviderSpec(
        name="fixture",
        keywords=(),
        env_key="",
        builtin_models=(
            ProviderModelSpec(
                id="vision-fixture",
                context_window=32_768,
                max_output_tokens=4_096,
                input_modalities=("text", "image"),
                output_modalities=("text",),
            ),
        ),
    )
    provider = OpenAICompatProvider(
        default_model="vision-fixture",
        provider_name="fixture",
        spec=spec,
        api_base="http://127.0.0.1:1/v1",
    )
    runtime = LLMRuntime.capture(
        provider,
        "vision-fixture",
        context_window_tokens=32_768,
    )
    capabilities = runtime.resolved_capabilities

    assert capabilities.context_window_tokens == 32_768
    assert capabilities.max_output_tokens == 4_096
    assert capabilities.input_modalities == frozenset({"text", "image"})
    assert capabilities.capability_source == "catalog"


def test_unknown_modalities_are_not_treated_as_text_only() -> None:
    spec = ProviderSpec(
        name="fixture",
        keywords=(),
        env_key="",
        builtin_models=(ProviderModelSpec(id="unknown-fixture"),),
    )
    provider = OpenAICompatProvider(
        default_model="unknown-fixture",
        provider_name="fixture",
        spec=spec,
        api_base="http://127.0.0.1:1/v1",
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": _png_data_uri()}},
        ],
    }]

    sanitized = provider._sanitize_messages(messages, "unknown-fixture")

    assert isinstance(sanitized[0]["content"], list)
    assert any(
        block.get("type") == "image_url"
        for block in sanitized[0]["content"]
        if isinstance(block, dict)
    )
