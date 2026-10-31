"""Provider dispatches validate their final concrete transport bodies."""

from __future__ import annotations

import base64
import json
import struct
import zlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import tiktoken

from nanobot.providers.anthropic_provider import AnthropicProvider
from nanobot.providers.azure_openai_provider import AzureOpenAIProvider
from nanobot.providers.base import LLMProvider, LLMResponse, LLMUsage, ProviderCallContext
from nanobot.providers.bedrock_provider import BedrockProvider
from nanobot.providers.fallback_provider import FallbackProvider
from nanobot.providers.openai_codex_provider import OpenAICodexProvider
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.openai_responses import build_responses_state
from nanobot.providers.pooled_provider import PooledProvider
from nanobot.providers.request_budget import (
    FinalizedRequestBudgetExceeded,
    ProviderRequestLimits,
    TransportIdentity,
    measure_finalized_request,
    validate_finalized_request,
)
from nanobot.providers.xai_grok_provider import (
    XAIGrokProvider,
    _XAIHTTPError,
    _XAIIncompleteHostedToolError,
)


def _png_data_uri(width: int, height: int, padding: int = 1_000_000) -> str:
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = (
        struct.pack(">I", len(ihdr_data))
        + b"IHDR"
        + ihdr_data
        + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data))
    )
    payload = b"\x89PNG\r\n\x1a\n" + ihdr + b"\x00" * padding
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def _limits(*, total: int) -> ProviderCallContext:
    return ProviderCallContext(request_limits=ProviderRequestLimits(
        total_tokens=total,
        completion_reserve_tokens=1,
        safety_buffer_tokens=0,
    ))


def _provider(create: AsyncMock) -> OpenAICompatProvider:
    provider = OpenAICompatProvider(api_key="fixture", default_model="fixture-model")
    provider._ensure_client = AsyncMock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    return provider


class _Leaf(LLMProvider):
    def __init__(self, *, fail_budget: bool = False) -> None:
        super().__init__(provider_name="leaf")
        self.calls: list[dict[str, object]] = []
        self.fail_budget = fail_budget

    async def chat(self, **kwargs: object) -> LLMResponse:
        self.calls.append(dict(kwargs))
        if self.fail_budget:
            raise FinalizedRequestBudgetExceeded()
        return LLMResponse(content="ok")

    async def chat_with_context(self, *, provider_context: ProviderCallContext, **kwargs: object) -> LLMResponse:
        return await self.chat(**kwargs, provider_context=provider_context)

    def get_default_model(self) -> str:
        return "leaf-model"


def _exact_context(provider: OpenAICompatProvider, *, messages: list[dict[str, object]],
                   tools: list[dict[str, object]] | None = None) -> ProviderCallContext:
    body = provider._build_kwargs(messages, tools, None, 1, 0.0, "high", None)
    measured = measure_finalized_request(
        TransportIdentity(provider.provider_name, "openai_chat_completions", body["model"]),
        body,
        ProviderRequestLimits(total_tokens=100_000, safety_buffer_tokens=0),
    )
    return ProviderCallContext(request_limits=ProviderRequestLimits(
        total_tokens=measured.input_tokens + measured.completion_reserve_tokens,
        safety_buffer_tokens=0,
    ))


def test_measurement_uses_tokenizer_estimate_with_a_safety_margin() -> None:
    identity = TransportIdentity("fixture", "fixture_transport", "fixture-model")
    body = {"model": "fixture-model", "input": ["ASCII", "emoji: 😀"], "max_tokens": 1}

    measurement = measure_finalized_request(
        identity,
        body,
        ProviderRequestLimits(total_tokens=100_000, safety_buffer_tokens=0),
    )

    encoded = json.dumps(
        body,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    exact = len(tiktoken.get_encoding("cl100k_base").encode(encoded.decode("utf-8")))
    assert measurement.source == "canonical_json_tiktoken"
    assert measurement.input_tokens >= exact
    assert measurement.input_tokens < len(encoded)


def test_measurement_fails_closed_to_the_byte_ceiling_without_a_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_name: str) -> object:
        raise RuntimeError("tokenizer unavailable")

    monkeypatch.setattr("nanobot.providers.request_budget.tiktoken.get_encoding", unavailable)
    identity = TransportIdentity("fixture", "fixture_transport", "fixture-model")
    body = {"model": "fixture-model", "input": ["plain text"], "max_tokens": 1}

    measurement = measure_finalized_request(
        identity,
        body,
        ProviderRequestLimits(total_tokens=100_000, safety_buffer_tokens=0),
    )

    encoded = json.dumps(
        body,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert measurement.input_tokens == len(encoded)
    assert measurement.source == "canonical_json_utf8_bytes"


def test_inline_image_is_measured_as_vision_tokens_not_base64_wire_bytes() -> None:
    identity = TransportIdentity("local", "openai_chat_completions", "vision-model")
    image = _png_data_uri(1024, 1024)
    body = {
        "model": "vision-model",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "compare these images"},
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "image_url", "image_url": {"url": _png_data_uri(1024, 1024)}},
        ]}],
    }

    measurement = measure_finalized_request(
        identity,
        body,
        ProviderRequestLimits(
            total_tokens=165_888,
            completion_reserve_tokens=8_192,
            safety_buffer_tokens=1_024,
        ),
    )

    assert measurement.fits
    assert measurement.input_tokens < 10_000
    assert "inline_image" in measurement.source
    assert measurement.input_tokens < len(image) // 100


def test_local_165k_window_accepts_long_text_with_parallel_images() -> None:
    identity = TransportIdentity("local_manager", "openai_chat_completions_stream", "local-model")
    content: list[dict[str, object]] = [
        {"type": "text", "text": "project evidence " * 30_000},
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": _png_data_uri(1024, 1024)}}
        for _ in range(3)
    )
    body = {
        "model": "local-model",
        "max_tokens": 8_192,
        "messages": [{"role": "user", "content": content}],
    }

    measurement = measure_finalized_request(
        identity,
        body,
        ProviderRequestLimits(
            total_tokens=165_888,
            completion_reserve_tokens=8_192,
            safety_buffer_tokens=1_024,
        ),
    )

    assert measurement.fits
    assert 60_000 < measurement.input_tokens < 90_000
    assert "+inline_image" in measurement.source


def test_text_embedded_media_data_uri_still_counts_as_text_tokens() -> None:
    identity = TransportIdentity("local", "openai_chat_completions", "text-model")
    image = _png_data_uri(1024, 1024)
    body = {
        "model": "text-model",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "untrusted page: " + image},
        ]}],
    }

    measurement = measure_finalized_request(
        identity,
        body,
        ProviderRequestLimits(total_tokens=100_000, safety_buffer_tokens=0),
    )

    assert not measurement.fits
    assert "inline_image" not in measurement.source


def test_remote_image_uses_a_bounded_vision_allowance() -> None:
    identity = TransportIdentity("local", "openai_chat_completions", "vision-model")
    body = {
        "model": "vision-model",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
        ]}],
    }

    measurement = measure_finalized_request(
        identity,
        body,
        ProviderRequestLimits(total_tokens=16_000, completion_reserve_tokens=1, safety_buffer_tokens=0),
    )

    assert not measurement.fits
    assert 16_000 < measurement.input_tokens < 17_000
    assert "remote_image" in measurement.source


def test_anthropic_image_sources_use_a_bounded_vision_allowance() -> None:
    identity = TransportIdentity("anthropic", "anthropic_messages", "vision-model")
    inline = _png_data_uri(1024, 1024)
    encoded = inline.split(";base64,", 1)[1]
    inline_body = {
        "model": "vision-model",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": [{
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": encoded},
        }]}],
    }
    inline_measurement = measure_finalized_request(
        identity,
        inline_body,
        ProviderRequestLimits(total_tokens=165_888, safety_buffer_tokens=1_024),
    )
    assert inline_measurement.fits
    assert inline_measurement.input_tokens < 10_000
    assert "inline_image" in inline_measurement.source

    remote_body = {
        "model": "vision-model",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": [{
            "type": "image",
            "source": {"type": "url", "url": "https://example.test/image.png"},
        }]}],
    }
    remote_measurement = measure_finalized_request(
        identity,
        remote_body,
        ProviderRequestLimits(total_tokens=16_000, safety_buffer_tokens=0),
    )
    assert not remote_measurement.fits
    assert "remote_image" in remote_measurement.source


@pytest.mark.asyncio
async def test_finalized_chat_body_over_budget_never_dispatches() -> None:
    create = AsyncMock()
    provider = _provider(create)

    with pytest.raises(FinalizedRequestBudgetExceeded):
        await provider.chat_with_context(
            provider_context=_limits(total=1),
            messages=[{"role": "user", "content": "large finalized request"}],
            tools=[{"type": "function", "function": {"name": "tool", "parameters": {}}}],
            max_tokens=1,
        )

    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalized_chat_body_is_detached_from_caller_messages() -> None:
    create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop",
        )], usage=None,
    ))
    provider = _provider(create)
    messages = [{"role": "user", "content": "hello"}]

    await provider.chat_with_context(
        provider_context=_limits(total=10_000), messages=messages, max_tokens=1,
    )

    assert messages == [{"role": "user", "content": "hello"}]
    assert create.await_args.kwargs["messages"] is not messages


@pytest.mark.asyncio
async def test_exact_finalized_limit_passes_and_one_token_over_never_dispatches() -> None:
    success = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")],
        usage=None,
    ))
    provider = _provider(success)
    provider._extra_body = {"response_format": {"type": "json_schema", "json_schema": {"name": "reply"}}}
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "image and structured output"},
        {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
    ]}]
    tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    context = _exact_context(provider, messages=messages, tools=tools)

    await provider.chat_with_context(
        provider_context=context, messages=messages, tools=tools, max_tokens=1,
        reasoning_effort="high",
    )
    success.assert_awaited_once()
    body = success.await_args.kwargs
    assert body["tools"] == tools
    assert body["extra_body"]["response_format"] == provider._extra_body["response_format"]
    assert body["reasoning_effort"] == "high"
    assert body["messages"][0]["content"][1]["type"] == "image_url"

    blocked_create = AsyncMock()
    blocked = _provider(blocked_create)
    blocked._extra_body = provider._extra_body
    over = replace(context.request_limits, total_tokens=context.request_limits.total_tokens - 1)
    with pytest.raises(FinalizedRequestBudgetExceeded):
        await blocked.chat_with_context(
            provider_context=replace(context, request_limits=over), messages=messages, tools=tools,
            max_tokens=1, reasoning_effort="high",
        )
    blocked_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_and_chat_once_budget_rejections_never_dispatch_or_retry() -> None:
    create = AsyncMock()
    provider = _provider(create)
    context = _limits(total=1)
    kwargs = {"messages": [{"role": "user", "content": "oversized"}], "max_tokens": 1}

    with pytest.raises(FinalizedRequestBudgetExceeded):
        await provider.chat_once(provider_context=context, **kwargs)
    with pytest.raises(FinalizedRequestBudgetExceeded):
        await provider.chat_stream_with_retry(provider_context=context, **kwargs)

    create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["anthropic", "azure", "bedrock"])
async def test_converted_stream_transport_hard_limit_blocks_sdk_dispatch(provider_name: str) -> None:
    context = _limits(total=1)
    if provider_name == "anthropic":
        stream = MagicMock()
        provider = AnthropicProvider(api_key="fixture")
        provider._client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
        with pytest.raises(FinalizedRequestBudgetExceeded):
            await provider.chat_stream_with_context(
                provider_context=context,
                messages=[{"role": "user", "content": "large"}],
                max_tokens=1,
            )
        stream.assert_not_called()
    elif provider_name == "azure":
        create = AsyncMock()
        provider = AzureOpenAIProvider(api_key="fixture", api_base="https://example.test")
        provider._client = SimpleNamespace(responses=SimpleNamespace(create=create))
        with pytest.raises(FinalizedRequestBudgetExceeded):
            await provider.chat_stream_with_context(
                provider_context=context,
                messages=[{"role": "user", "content": "large"}],
                max_tokens=1,
            )
        create.assert_not_awaited()
    else:
        client = SimpleNamespace(stream_calls=[])

        def converse_stream(**kwargs: object) -> object:
            client.stream_calls.append(kwargs)
            return {"stream": iter(())}

        client.converse_stream = converse_stream
        provider = BedrockProvider(region="fixture", client=client)
        with pytest.raises(FinalizedRequestBudgetExceeded):
            await provider.chat_stream_with_context(
                provider_context=context,
                messages=[{"role": "user", "content": "large"}],
                max_tokens=1,
            )
        assert client.stream_calls == []


def test_clamps_selected_model_window_and_rejects_opaque_finalized_bodies() -> None:
    identity = TransportIdentity("fixture", "fixture_transport", "override-model")
    body = {"model": "override-model", "max_tokens": 1, "input": ["abcdefghijk"]}
    with pytest.raises(FinalizedRequestBudgetExceeded) as exceeded:
        validate_finalized_request(identity, body, ProviderRequestLimits(
            total_tokens=100_000, model_context_window_tokens=1, safety_buffer_tokens=0,
        ))
    assert exceeded.value.measurement is not None
    assert exceeded.value.measurement.total_limit_tokens == 1

    with pytest.raises(FinalizedRequestBudgetExceeded):
        validate_finalized_request(identity, {"model": "override-model", "input": b"opaque"},
                                   ProviderRequestLimits(total_tokens=100_000))


@pytest.mark.asyncio
async def test_bedrock_binary_media_is_measured_and_hard_limited_before_dispatch() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def converse(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "stopReason": "end_turn",
            }

    def image_message(data: bytes) -> list[dict[str, object]]:
        encoded = base64.b64encode(data).decode("ascii")
        return [{"role": "user", "content": [{
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }]}]

    accepted_client = Client()
    accepted = BedrockProvider(region="fixture", client=accepted_client)
    accepted_result = await accepted.chat_with_context(
        provider_context=_limits(total=100_000),
        messages=image_message(b"tiny image"),
        max_tokens=1,
    )
    assert accepted_result.content == "ok"
    assert len(accepted_client.calls) == 1
    sent = accepted_client.calls[0]
    assert sent["messages"][0]["content"][0]["image"]["source"]["bytes"] == b"tiny image"

    blocked_client = Client()
    blocked = BedrockProvider(region="fixture", client=blocked_client)
    with pytest.raises(FinalizedRequestBudgetExceeded) as caught:
        await blocked.chat_with_context(
            provider_context=_limits(total=100),
            messages=image_message(b"x" * 1024),
            max_tokens=1,
        )
    assert caught.value.measurement is not None
    assert "bedrock_binary_media" in caught.value.measurement.source
    assert blocked_client.calls == []


@pytest.mark.asyncio
async def test_bedrock_redacted_reasoning_bytes_are_measured_without_mutating_replay() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def converse(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "stopReason": "end_turn",
            }

    original_bytes = b"redacted reasoning replay"
    encoded = base64.b64encode(original_bytes).decode("ascii")
    messages = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "thinking_blocks": [{
            "type": "redacted_thinking",
            "redactedContentBase64": encoded,
        }]},
        {"role": "user", "content": "continue"},
    ]
    probe = BedrockProvider(region="fixture", client=Client())
    body = probe._build_kwargs(messages, None, None, 1, 0.7, None, None)
    measurement = measure_finalized_request(
        TransportIdentity("bedrock", "bedrock_converse", str(body["modelId"])),
        body,
        ProviderRequestLimits(total_tokens=100_000, safety_buffer_tokens=0),
    )
    exact_context = ProviderCallContext(request_limits=ProviderRequestLimits(
        total_tokens=measurement.input_tokens + measurement.completion_reserve_tokens,
        safety_buffer_tokens=0,
    ))

    accepted_client = Client()
    accepted = BedrockProvider(region="fixture", client=accepted_client)
    accepted_result = await accepted.chat_with_context(
        provider_context=exact_context,
        messages=messages,
        max_tokens=1,
    )
    assert accepted_result.content == "ok"
    sent = accepted_client.calls[0]
    assert sent["messages"][1]["content"][0]["reasoningContent"]["redactedContent"] == original_bytes
    assert messages[1]["thinking_blocks"][0] == {
        "type": "redacted_thinking",
        "redactedContentBase64": encoded,
    }

    blocked_client = Client()
    blocked = BedrockProvider(region="fixture", client=blocked_client)
    one_token_over_context = ProviderCallContext(request_limits=ProviderRequestLimits(
        total_tokens=exact_context.request_limits.total_tokens - 1,
        safety_buffer_tokens=0,
    ))
    with pytest.raises(FinalizedRequestBudgetExceeded) as caught:
        await blocked.chat_with_context(
            provider_context=one_token_over_context,
            messages=messages,
            max_tokens=1,
        )
    assert caught.value.measurement is not None
    assert caught.value.measurement.over_by == 1
    assert "bedrock_redacted_reasoning" in caught.value.measurement.source
    assert blocked_client.calls == []
    assert messages[1]["thinking_blocks"][0]["redactedContentBase64"] == encoded


@pytest.mark.asyncio
async def test_fallback_and_pool_preserve_limits_and_budget_error_never_falls_back() -> None:
    primary = _Leaf()
    fallback = _Leaf()
    wrapper = FallbackProvider(
        primary,
        [SimpleNamespace(model="fallback-model", max_tokens=8, temperature=0.1,
                         context_window_tokens=128, reasoning_effort=None)],
        lambda _preset: fallback,
        primary_context_window_tokens=256,
    )
    context = ProviderCallContext(request_limits=ProviderRequestLimits(total_tokens=1_000))
    await wrapper.chat_with_context(provider_context=context, messages=[])
    primary_context = primary.calls[-1]["provider_context"]
    assert isinstance(primary_context, ProviderCallContext)
    assert primary_context.request_limits is not None
    assert primary_context.request_limits.model_context_window_tokens == 256

    pooled_leaf = _Leaf()
    pool = PooledProvider([pooled_leaf], ["key"])
    await pool.chat_with_context(provider_context=context, messages=[])
    pooled_context = pooled_leaf.calls[-1]["provider_context"]
    assert isinstance(pooled_context, ProviderCallContext)
    assert pooled_context.request_limits == context.request_limits

    guarded = FallbackProvider(
        _Leaf(fail_budget=True),
        [SimpleNamespace(model="fallback-model", max_tokens=8, temperature=0.1,
                         context_window_tokens=128, reasoning_effort=None)],
        lambda _preset: (_ for _ in ()).throw(AssertionError("fallback must not be created")),
    )
    with pytest.raises(FinalizedRequestBudgetExceeded):
        await guarded.chat_with_context(provider_context=context, messages=[])


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["anthropic", "azure", "bedrock"])
async def test_converted_transport_hard_limit_blocks_its_sdk_dispatch(provider_name: str) -> None:
    context = _limits(total=1)
    if provider_name == "anthropic":
        create = AsyncMock()
        provider = AnthropicProvider(api_key="fixture")
        provider._client = SimpleNamespace(messages=SimpleNamespace(create=create))
        with pytest.raises(FinalizedRequestBudgetExceeded):
            await provider.chat_with_context(
                provider_context=context, messages=[{"role": "user", "content": "large"}], max_tokens=1,
            )
        create.assert_not_awaited()
    elif provider_name == "azure":
        create = AsyncMock()
        provider = AzureOpenAIProvider(api_key="fixture", api_base="https://example.test")
        provider._client = SimpleNamespace(responses=SimpleNamespace(create=create))
        with pytest.raises(FinalizedRequestBudgetExceeded):
            await provider.chat_with_context(
                provider_context=context, messages=[{"role": "user", "content": "large"}], max_tokens=1,
            )
        create.assert_not_awaited()
    else:
        client = SimpleNamespace(calls=[])

        def converse(**kwargs: object) -> object:
            client.calls.append(kwargs)
            return {}

        client.converse = converse
        provider = BedrockProvider(region="fixture", client=client)
        with pytest.raises(FinalizedRequestBudgetExceeded):
            await provider.chat_with_context(
                provider_context=context, messages=[{"role": "user", "content": "large"}], max_tokens=1,
            )
        assert client.calls == []


@pytest.mark.asyncio
async def test_responses_to_chat_compatibility_fallback_revalidates_final_transport() -> None:
    class CompatibilityError(Exception):
        status_code = 400
        body = "responses unsupported"

    chat_create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")],
        usage=None,
    ))
    response_create = AsyncMock(side_effect=CompatibilityError())
    provider = _provider(chat_create)
    provider._ensure_client = AsyncMock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)),
        responses=SimpleNamespace(create=response_create),
    ))
    provider._should_use_responses_api = lambda *_: True
    provider._build_responses_body = lambda *_: {"model": "fixture-model", "input": ["small"], "max_output_tokens": 1}
    seen: list[str] = []
    original = provider.validate_finalized_request

    def validate(body: dict[str, object], **kwargs: object):
        seen.append(str(kwargs["transport"]))
        return original(body, **kwargs)

    provider.validate_finalized_request = validate  # type: ignore[method-assign]
    await provider.chat_with_context(
        provider_context=_limits(total=10_000), messages=[{"role": "user", "content": "small"}], max_tokens=1,
    )
    assert seen == ["openai_responses", "openai_chat_completions"]
    response_create.assert_awaited_once()
    chat_create.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["openai", "azure"])
async def test_compaction_compatibility_retry_revalidates_second_final_body(
    provider_name: str,
) -> None:
    class UnsupportedCompactionError(Exception):
        status_code = 400
        body = "context_management is unsupported"

    create = AsyncMock(side_effect=[UnsupportedCompactionError(), SimpleNamespace()])
    body = {
        "model": "fixture-model",
        "input": ["small"],
        "max_output_tokens": 1,
        "context_management": [{"type": "compaction", "compact_threshold": 10}],
    }
    context = _limits(total=10_000)
    if provider_name == "openai":
        provider = _provider(AsyncMock())
        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        create_call = provider._create_response_with_compaction_fallback
        args = (client, body, context)
    else:
        provider = AzureOpenAIProvider(api_key="fixture", api_base="https://example.test")
        provider._client = SimpleNamespace(responses=SimpleNamespace(create=create))
        create_call = provider._create_response_with_compaction_fallback
        args = (body, context)

    seen: list[bool] = []
    original = provider.validate_finalized_request

    def validate(request: dict[str, object], **kwargs: object) -> object:
        seen.append("context_management" in request)
        return original(request, **kwargs)

    provider.validate_finalized_request = validate  # type: ignore[method-assign]
    await create_call(*args)

    assert seen == [True, False]
    assert create.await_count == 2


@pytest.mark.asyncio
async def test_anthropic_streaming_required_reroute_revalidates_before_stream_dispatch() -> None:
    create = AsyncMock(side_effect=ValueError("Streaming is required for this request"))
    stream = MagicMock()
    provider = AnthropicProvider(api_key="fixture")
    provider._client = SimpleNamespace(messages=SimpleNamespace(create=create, stream=stream))
    context = _limits(total=10_000)
    original = provider.validate_finalized_request
    seen: list[str] = []

    def validate(body: dict[str, object], **kwargs: object) -> object:
        transport = str(kwargs["transport"])
        seen.append(transport)
        if transport == "anthropic_messages_stream":
            raise FinalizedRequestBudgetExceeded()
        return original(body, **kwargs)

    provider.validate_finalized_request = validate  # type: ignore[method-assign]
    with pytest.raises(FinalizedRequestBudgetExceeded):
        await provider.chat_with_context(
            provider_context=context,
            messages=[{"role": "user", "content": "reroute me"}],
            max_tokens=1,
        )

    assert seen == ["anthropic_messages", "anthropic_messages_stream"]
    create.assert_awaited_once()
    stream.assert_not_called()


@pytest.mark.asyncio
async def test_xai_hosted_tool_recovery_revalidates_each_generation_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nanobot.providers.xai_grok_provider.get_xai_oauth_token",
        lambda **_kwargs: SimpleNamespace(access="token"),
    )
    attempts = 0

    async def request(*_args: object, **_kwargs: object) -> tuple[
        str, list[object], str, LLMUsage | None, None,
    ]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _XAIIncompleteHostedToolError(
                [{"name": "x_search", "call_id": "search-1"}],
                usage=None,
            )
        return "recovered", [], "stop", None, None

    monkeypatch.setattr("nanobot.providers.xai_grok_provider._request_xai", request)
    provider = XAIGrokProvider()
    provider._supports_backend_search = AsyncMock(return_value=False)
    original = provider.validate_finalized_request
    validations = 0

    def validate(body: dict[str, object], **kwargs: object) -> object:
        nonlocal validations
        validations += 1
        return original(body, **kwargs)

    provider.validate_finalized_request = validate  # type: ignore[method-assign]
    response = await provider.chat_stream_with_retry(
        [{"role": "user", "content": "recover hosted search"}],
        provider_context=_limits(total=10_000),
        max_tokens=1,
        on_stream_recover=lambda: _complete(),
    )

    assert response.content == "recovered"
    assert attempts == 2
    assert validations == 2


async def _complete() -> None:
    return None


@pytest.mark.asyncio
async def test_xai_auth_refresh_revalidates_second_generation_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_calls = 0

    def token(*, force_refresh: bool = False, **_kwargs: object) -> SimpleNamespace:
        nonlocal token_calls
        token_calls += 1
        return SimpleNamespace(access="fresh" if force_refresh else "stale")

    monkeypatch.setattr("nanobot.providers.xai_grok_provider.get_xai_oauth_token", token)
    attempts = 0

    async def request(*_args: object, **_kwargs: object) -> tuple[
        str, list[object], str, LLMUsage | None, None,
    ]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _XAIHTTPError("unauthorized", status_code=401, should_retry=False)
        return "refreshed", [], "stop", None, None

    monkeypatch.setattr("nanobot.providers.xai_grok_provider._request_xai", request)
    provider = XAIGrokProvider()
    provider._supports_backend_search = AsyncMock(return_value=False)
    original = provider.validate_finalized_request
    validations = 0

    def validate(body: dict[str, object], **kwargs: object) -> object:
        nonlocal validations
        validations += 1
        return original(body, **kwargs)

    provider.validate_finalized_request = validate  # type: ignore[method-assign]
    response = await provider.chat_with_context(
        provider_context=_limits(total=10_000),
        messages=[{"role": "user", "content": "refresh token"}],
        max_tokens=1,
    )

    assert response.content == "refreshed"
    assert attempts == 2
    assert token_calls == 2
    assert validations == 2


@pytest.mark.asyncio
async def test_codex_and_xai_hard_limits_block_final_http_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    codex_send = AsyncMock()
    monkeypatch.setattr(
        "nanobot.providers.openai_codex_provider.get_codex_token",
        lambda **_kwargs: SimpleNamespace(account_id="account", access="token"),
    )
    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", codex_send)
    codex = OpenAICodexProvider()
    with pytest.raises(FinalizedRequestBudgetExceeded):
        await codex.chat_with_context(
            provider_context=_limits(total=1), messages=[{"role": "user", "content": "large"}], max_tokens=1,
        )
    codex_send.assert_not_awaited()

    xai_send = AsyncMock()
    monkeypatch.setattr(
        "nanobot.providers.xai_grok_provider.get_xai_oauth_token",
        lambda **_kwargs: SimpleNamespace(access="token"),
    )
    monkeypatch.setattr("nanobot.providers.xai_grok_provider._request_xai", xai_send)
    xai = XAIGrokProvider()
    xai._supports_backend_search = AsyncMock(return_value=False)
    with pytest.raises(FinalizedRequestBudgetExceeded):
        await xai.chat_with_context(
            provider_context=_limits(total=1), messages=[{"role": "user", "content": "large"}], max_tokens=1,
        )
    xai_send.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_native_compaction_and_generation_each_validate_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nanobot.providers.openai_codex_provider.get_codex_token",
        lambda **_kwargs: SimpleNamespace(account_id="account", access="token"),
    )
    provider = OpenAICodexProvider(default_model="openai-codex/gpt-5.6-sol")
    state_provider = provider._responses_state_provider()
    state = build_responses_state(
        provider=state_provider,
        model="gpt-5.6-sol",
        input_items=[{"type": "message", "role": "user", "content": "old"}],
        output_items=[{"type": "reasoning", "encrypted_content": "opaque"}],
        usage=LLMUsage.reported(input_tokens=90, output_tokens=1),
    )
    physical_bodies: list[dict[str, object]] = []

    async def request(
        _url: str,
        _headers: dict[str, str],
        body: dict[str, object],
        **_kwargs: object,
    ) -> LLMResponse:
        physical_bodies.append(body)
        assert body["input"][-1] == {"type": "compaction_trigger"}
        return LLMResponse(
            content=None,
            provider_state=build_responses_state(
                provider=state_provider,
                model="gpt-5.6-sol",
                input_items=body["input"],
                output_items=[{"type": "compaction", "encrypted_content": "x" * 16_384}],
                usage=LLMUsage.reported(input_tokens=95, output_tokens=1),
            ),
        )

    monkeypatch.setattr("nanobot.providers.openai_codex_provider._request_codex", request)
    context = ProviderCallContext(
        conversation_state=state.with_pending_messages([{"role": "user", "content": "new"}]),
        context_window_tokens=100,
        request_limits=ProviderRequestLimits(total_tokens=1_000, safety_buffer_tokens=0),
    )
    validations: list[list[object]] = []
    original = provider.validate_finalized_request

    def validate(body: dict[str, object], **kwargs: object) -> object:
        validations.append(list(body["input"]))
        return original(body, **kwargs)

    provider.validate_finalized_request = validate  # type: ignore[method-assign]
    with pytest.raises(FinalizedRequestBudgetExceeded):
        await provider.chat_with_context(
            provider_context=context,
            messages=[{"role": "user", "content": "new"}],
            max_tokens=1,
        )

    assert len(validations) == 2
    assert validations[0][-1] == {"type": "compaction_trigger"}
    assert any(item.get("type") == "compaction" for item in validations[1])
    assert len(physical_bodies) == 1


@pytest.mark.asyncio
async def test_budget_rejection_does_not_enter_compatibility_transport_fallback() -> None:
    chat_create = AsyncMock()
    responses_create = AsyncMock()
    provider = _provider(chat_create)
    provider._ensure_client = AsyncMock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)),
        responses=SimpleNamespace(create=responses_create),
    ))
    provider._should_use_responses_api = lambda *_: True
    provider._build_responses_body = lambda *_: {"model": "fixture-model", "input": ["opaque"]}
    provider._responses_failures = {"fixture-model:": 2}
    provider._responses_tripped_at = {"fixture-model:": 1.0}
    failures_before = dict(provider._responses_failures)
    tripped_before = dict(provider._responses_tripped_at)

    with pytest.raises(FinalizedRequestBudgetExceeded):
        await provider.chat_with_context(
            provider_context=_limits(total=1),
            messages=[{"role": "user", "content": "large"}], max_tokens=1,
        )

    responses_create.assert_not_awaited()
    chat_create.assert_not_awaited()
    assert provider._responses_failures == failures_before
    assert provider._responses_tripped_at == tripped_before


@pytest.mark.asyncio
async def test_budget_rejection_keeps_fallback_circuit_and_caller_image_intact() -> None:
    primary = _provider(AsyncMock())
    fallback = FallbackProvider(
        primary,
        [SimpleNamespace(model="fallback-model", max_tokens=1, temperature=0.1,
                         context_window_tokens=None, reasoning_effort=None)],
        lambda _preset: (_ for _ in ()).throw(AssertionError("fallback must not be created")),
    )
    messages = [{"role": "user", "content": [{
        "type": "image_url",
        "image_url": {"url": "https://example.test/image.png"},
    }]}]
    expected_messages = [{"role": "user", "content": [{
        "type": "image_url",
        "image_url": {"url": "https://example.test/image.png"},
    }]}]

    with pytest.raises(FinalizedRequestBudgetExceeded):
        await fallback.chat_stream_with_retry(
            messages=messages,
            provider_context=_limits(total=1),
            max_tokens=1,
        )

    assert messages == expected_messages
    assert fallback._primary_failures == 0
    assert fallback._primary_tripped_at is None
