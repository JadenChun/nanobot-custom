"""Evidence for the deliberately estimated Local Manager request policy."""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import AsyncOpenAI

from nanobot.agent.context_governance import (
    SNIP_SAFETY_BUFFER,
    ContextGovernanceConfig,
    ContextGovernor,
)
from nanobot.agent.memory import Consolidator
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers import request_budget as budget_module
from nanobot.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ProviderCallContext,
)
from nanobot.providers.fallback_provider import FallbackProvider
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.pooled_provider import PooledProvider
from nanobot.providers.registry import find_by_name
from nanobot.providers.request_budget import (
    LOCAL_MANAGER_POLICY,
    BackendContextOverflow,
    FinalizedRequestBudgetExceeded,
    GovernedRequestConflictError,
    LocalRequestUsageReceipt,
    LocalUsageAnchor,
    PreparedRequestBudgetPreview,
    ProviderRequestLimits,
    ProviderRequestRejected,
    RequestPayloadBytesExceeded,
    TransportIdentity,
    validate_finalized_request,
)
from nanobot.utils.llm_runtime import LLMRuntime

MODEL = "Qwen3.8-27B-UD-Q6_K_XL"
FIXTURE = Path(__file__).parent / "fixtures" / "local_manager_context_overflow.json"


def _governance_config(provider: LLMProvider) -> ContextGovernanceConfig:
    return ContextGovernanceConfig(
        provider=provider,
        model=MODEL,
        tools=ToolRegistry(),
        workspace=None,
        session_key="test:provider-preview",
        max_tool_result_chars=4096,
        context_window_tokens=8192,
        max_tokens=1,
        generation=GenerationSettings(max_tokens=1),
    )


def _png_data_uri(width: int, height: int, padding: int = 0) -> str:
    def chunk(name: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", zlib.crc32(name + data))
        )

    row = b"\x00" + b"\x00" * (width * 3)
    pixels = zlib.compress(row * height)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ancillary = chunk(b"tEXt", b"Comment\x00" + b"x" * padding) if padding else b""
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr_data)
        + ancillary
        + chunk(b"IDAT", pixels)
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _context(total: int = 8_192) -> ProviderCallContext:
    return ProviderCallContext(request_limits=ProviderRequestLimits(
        total_tokens=total,
        completion_reserve_tokens=1,
        safety_buffer_tokens=0,
    ))


def _local_provider(
    *,
    extra_body: dict[str, object] | None = None,
    timeout: float | None = None,
    provider_name: str = "local_manager",
    spec_name: str = "local_manager",
) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        api_key="fixture-credential",
        api_base="http://manager.test/v1",
        default_model=MODEL,
        provider_name=provider_name,
        spec=find_by_name(spec_name),
        extra_body=extra_body,
        timeout=timeout,
    )


def test_governor_declines_nonconcrete_provider_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MagicMock(spec=LLMProvider)
    provider.preview_request_budget.return_value = object()
    monkeypatch.setattr(
        "nanobot.agent.context_governance.estimate_prompt_tokens_chain",
        lambda *_args, **_kwargs: (8_000, "legacy_compatibility_estimate"),
    )

    pressure = ContextGovernor().request_pressure(
        _governance_config(provider),
        [{"role": "user", "content": "legacy provider"}],
        None,
        usage_matches_messages=False,
        tool_definitions=None,
    )

    assert pressure == (8_000, "legacy_compatibility_estimate")
    provider.preview_request_budget.assert_called_once()


def test_governor_falls_back_from_nonconcrete_captured_generation() -> None:
    config = _governance_config(MagicMock(spec=LLMProvider))
    config.generation = MagicMock()
    config.max_tokens = MagicMock()

    assert ContextGovernor._captured_generation(config) == GenerationSettings()
    assert ContextGovernor.input_budget(config) == (
        config.context_window_tokens - GenerationSettings().max_tokens - SNIP_SAFETY_BUFFER
    )


def test_governor_preserves_concrete_configured_generation_cap() -> None:
    config = _governance_config(MagicMock(spec=LLMProvider))
    config.generation = GenerationSettings(max_tokens=128)
    config.max_tokens = 64

    assert ContextGovernor._captured_generation(config) == config.generation
    assert ContextGovernor.input_budget(config) == (
        config.context_window_tokens - 64 - SNIP_SAFETY_BUFFER
    )


def test_governor_accepts_concrete_local_provider_preview() -> None:
    provider = _local_provider()
    messages = [{"role": "user", "content": "known local request"}]

    preview = ContextGovernor._prepared_budget_preview(
        _governance_config(provider),
        messages,
        tool_definitions=None,
        stream=False,
        tool_choice=None,
    )

    assert type(preview) is PreparedRequestBudgetPreview
    assert preview.measurement.policy == LOCAL_MANAGER_POLICY
    assert preview.effective_body["messages"] == messages


def _success_body(model: str = MODEL) -> dict[str, object]:
    return {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 2},
        },
    }


@pytest.mark.asyncio
async def test_mock_transport_body_matches_preview_and_success_receipt() -> None:
    configured_tool = {"type": "function", "function": {"name": "configured"}}
    requested_tool = {"type": "function", "function": {"name": "requested"}}
    provider = _local_provider(extra_body={
        "metadata": {"nested": {"safe": True}},
        "timeout": "gateway-body-timeout",
        "max_completion_tokens": 17,
        "tools": [configured_tool],
    })
    messages = [{"role": "user", "content": "request"}]
    tools = [requested_tool]
    generation = GenerationSettings(temperature=0.2, max_tokens=1, reasoning_effort=None)
    context = ProviderCallContext(request_limits=ProviderRequestLimits(
        total_tokens=8_192,
        completion_reserve_tokens=4,
        safety_buffer_tokens=0,
    ))
    preview = provider.preview_request_budget(
        messages=messages,
        tools=tools,
        model=MODEL,
        generation=generation,
        provider_context=context,
        stream=False,
    )
    assert preview is not None
    assert preview.measurement.policy == LOCAL_MANAGER_POLICY
    assert preview.measurement.completion_reserve_tokens == 17
    assert preview.effective_body["tools"] == [requested_tool, configured_tool]
    assert preview.effective_body["timeout"] == "gateway-body-timeout"

    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_success_body(), request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="http://manager.test/v1",
            http_client=http_client,
        )
        result = await provider.chat_with_context(
            provider_context=context,
            messages=messages,
            tools=tools,
            model=MODEL,
            max_tokens=generation.max_tokens,
            temperature=generation.temperature,
            reasoning_effort=generation.reasoning_effort,
        )

    assert result.finish_reason == "stop"
    assert result.local_usage_receipt is not None
    assert result.usage is not None and result.usage.input_tokens == 3
    assert result.usage.cache_read_tokens == 2
    assert captured == [preview.effective_body]
    assert "timeout" in captured[0]  # Configured extra_body data is still JSON.
    assert "fixture-credential" not in repr(result)


@pytest.mark.asyncio
async def test_local_exact_fit_and_one_over_match_narrow_selected_leaf() -> None:
    provider = _local_provider()
    messages = [{"role": "user", "content": "Exact local admission boundary."}]
    generation = GenerationSettings(
        temperature=0.15,
        max_tokens=17,
        reasoning_effort="none",
    )
    outer_limits = ProviderRequestLimits(
        total_tokens=100_000,
        completion_reserve_tokens=1,
        safety_buffer_tokens=0,
    )
    outer_context = ProviderCallContext(request_limits=outer_limits)
    full_preview = provider.preview_request_budget(
        messages=messages,
        tools=None,
        model=MODEL,
        generation=generation,
        provider_context=outer_context,
        stream=False,
    )
    assert full_preview is not None
    selected_leaf_capacity = (
        full_preview.measurement.input_tokens
        + full_preview.measurement.completion_reserve_tokens
    )

    def unavailable_fallback(_preset: object) -> LLMProvider:
        raise AssertionError("single-leaf boundary fixture must not construct a fallback")

    exact_leaf = FallbackProvider(
        provider,
        [],
        unavailable_fallback,
        primary_context_window_tokens=selected_leaf_capacity,
    )
    exact_preview = exact_leaf.preview_request_budget(
        messages=messages,
        tools=None,
        model=MODEL,
        generation=generation,
        provider_context=outer_context,
        stream=False,
    )
    assert exact_preview is not None
    assert exact_preview.effective_body == full_preview.effective_body
    assert exact_preview.measurement.input_limit_tokens == exact_preview.measurement.input_tokens
    assert exact_preview.measurement.fits

    config = _governance_config(exact_leaf)
    config.context_window_tokens = outer_limits.total_tokens
    config.max_tokens = generation.max_tokens
    config.generation = generation
    exact_pressure = ContextGovernor().request_pressure(
        config,
        messages,
        None,
        usage_matches_messages=False,
        tool_definitions=None,
        stream=False,
        tool_choice=None,
        provider_context=outer_context,
    )
    assert exact_pressure == (
        exact_preview.measurement.input_tokens,
        exact_preview.measurement.source,
    )

    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert isinstance(body, dict)
        captured.append(body)
        return httpx.Response(200, json=_success_body(), request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="http://manager.test/v1",
            http_client=http_client,
            max_retries=0,
        )
        exact_response = await exact_leaf.chat_with_context(
            provider_context=outer_context,
            messages=messages,
            model=MODEL,
            max_tokens=generation.max_tokens,
            temperature=generation.temperature,
            reasoning_effort=generation.reasoning_effort,
        )
        assert exact_response.finish_reason == "stop"

        one_over_capacity = selected_leaf_capacity - 1
        one_over_leaf = FallbackProvider(
            provider,
            [],
            unavailable_fallback,
            primary_context_window_tokens=one_over_capacity,
        )
        one_over_preview = one_over_leaf.preview_request_budget(
            messages=messages,
            tools=None,
            model=MODEL,
            generation=generation,
            provider_context=outer_context,
            stream=False,
        )
        assert one_over_preview is not None
        assert one_over_preview.measurement.input_limit_tokens == (
            one_over_preview.measurement.input_tokens - 1
        )
        assert one_over_preview.measurement.over_by == 1
        one_over_config = _governance_config(one_over_leaf)
        one_over_config.context_window_tokens = outer_limits.total_tokens
        one_over_config.max_tokens = generation.max_tokens
        one_over_config.generation = generation
        one_over_pressure = ContextGovernor().request_pressure(
            one_over_config,
            messages,
            None,
            usage_matches_messages=False,
            tool_definitions=None,
            stream=False,
            tool_choice=None,
            provider_context=outer_context,
        )
        assert one_over_pressure == (
            one_over_preview.measurement.input_tokens,
            one_over_preview.measurement.source,
        )
        with pytest.raises(FinalizedRequestBudgetExceeded) as rejected:
            await one_over_leaf.chat_with_context(
                provider_context=outer_context,
                messages=messages,
                model=MODEL,
                max_tokens=generation.max_tokens,
                temperature=generation.temperature,
                reasoning_effort=generation.reasoning_effort,
            )

    assert rejected.value.measurement is not None
    assert rejected.value.measurement.over_by == 1
    assert captured == [exact_preview.effective_body]


@pytest.mark.asyncio
async def test_sdk_only_timeout_is_not_body_but_extra_body_timeout_is() -> None:
    provider = _local_provider(extra_body={"timeout": {"gateway": 15}})
    sdk_kwargs, body = provider._prepare_chat_request(
        [{"role": "user", "content": "request"}],
        None,
        MODEL,
        2,
        0.1,
        None,
        None,
        stream=False,
        timeout=7.0,
    )
    assert sdk_kwargs["timeout"] == 7.0
    assert body["timeout"] == {"gateway": 15}

    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_success_body(), request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        sdk = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="http://manager.test/v1",
            http_client=http_client,
        )
        await sdk.chat.completions.create(**sdk_kwargs)
    assert captured == [body]
    assert captured[0]["timeout"] == {"gateway": 15}


class _TrackingStream(httpx.AsyncByteStream):
    def __init__(self, payload: bytes, *, block: asyncio.Event | None = None) -> None:
        self.payload = payload
        self.block = block
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.started.set()
        if self.block is not None:
            await self.block.wait()
        yield self.payload

    async def aclose(self) -> None:
        self.closed = True


class _DelayedEventStream(httpx.AsyncByteStream):
    def __init__(self, payloads: list[bytes], delay: float) -> None:
        self.payloads = payloads
        self.delay = delay
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.started.set()
        for payload in self.payloads:
            await asyncio.sleep(self.delay)
            yield payload

    async def aclose(self) -> None:
        self.closed = True


def _sse_payload() -> bytes:
    events = [
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        },
    ]
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"


def _responses_sse_payload(events: list[dict[str, object]]) -> bytes:
    return b"".join(
        b"event: " + str(event["type"]).encode() + b"\n"
        + b"data: " + json.dumps(event).encode() + b"\n\n"
        for event in events
    )


def _openai_responses_provider(*, extra_body: dict[str, object] | None = None) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        api_key="fixture-credential",
        api_base="https://api.openai.com/v1",
        default_model="gpt-5.6",
        provider_name="openai",
        spec=find_by_name("openai"),
        api_type="responses",
        extra_body=extra_body,
        timeout=60.0,
    )


@pytest.mark.asyncio
async def test_real_sdk_once_stream_uses_preview_body_and_closes_response() -> None:
    provider = _local_provider(extra_body={"timeout": "gateway-body-timeout"}, timeout=60.0)
    messages = [{"role": "user", "content": "request"}]
    generation = GenerationSettings(temperature=0.1, max_tokens=4)
    context = ProviderCallContext(
        request_limits=ProviderRequestLimits(total_tokens=8_192, safety_buffer_tokens=0),
        stream_idle_timeout_seconds=300.0,
        request_deadline_monotonic=asyncio.get_running_loop().time() + 900.0,
        source_summary_mode="prefix",
    )
    preview = provider.preview_request_budget(
        messages=messages,
        tools=None,
        model=MODEL,
        generation=generation,
        provider_context=context,
        stream=True,
    )
    assert preview is not None

    stream = _TrackingStream(_sse_payload())
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="http://manager.test/v1",
            http_client=http_client,
            max_retries=0,
        )
        response = await provider.chat_stream_once(
            provider_context=context,
            messages=messages,
            model=MODEL,
            max_tokens=generation.max_tokens,
            temperature=generation.temperature,
        )

    assert response.finish_reason == "stop"
    assert response.content == "ok"
    assert response.local_usage_receipt is not None
    assert captured == [preview.effective_body]
    assert captured[0]["timeout"] == "gateway-body-timeout"
    assert captured[0]["stream"] is True
    assert stream.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "payload", "expected_finish"),
    [
        (
            "canonical",
            b'data: {"id":"probe","object":"chat.completion.chunk","created":1,'
            b'"model":"Qwen3.8-27B-UD-Q6_K_XL","choices":[{"index":0,"delta":'
            b'{"content":"incomplete historical fact"},"finish_reason":null}]}\n\n',
            "error",
        ),
        ("canonical", b"data: [DONE]\n\n", "error"),
        (
            None,
            b'data: {"id":"probe","object":"chat.completion.chunk","created":1,'
            b'"model":"Qwen3.8-27B-UD-Q6_K_XL","choices":[{"index":0,"delta":'
            b'{"content":"ordinary main answer"},"finish_reason":null}]}\n\n',
            "stop",
        ),
    ],
    ids=["auxiliary-eof-after-text", "auxiliary-bare-done", "legacy-main-eof"],
)
async def test_auxiliary_chat_stream_requires_explicit_terminal_chunk(
    mode: str | None,
    payload: bytes,
    expected_finish: str,
) -> None:
    provider = _local_provider(timeout=60.0)
    stream = _TrackingStream(payload)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )

    context = ProviderCallContext(
        stream_idle_timeout_seconds=1.0,
        request_deadline_monotonic=asyncio.get_running_loop().time() + 3.0,
        source_summary_mode=mode,
        single_attempt=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="http://manager.test/v1",
            http_client=http_client,
            max_retries=0,
        )
        response = await provider.chat_stream_once(
            provider_context=context,
            messages=[{"role": "user", "content": "historical evidence"}],
            model=MODEL,
            max_tokens=32,
        )

    assert response.finish_reason == expected_finish
    if mode is not None:
        assert response.local_usage_receipt is None
        assert response.error_kind == "summary_incomplete_stream"
    assert len(requests) == 1
    assert stream.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_finish"),
    [("canonical", "error"), (None, "stop")],
    ids=["auxiliary-no-response-completed", "legacy-main-no-response-completed"],
)
async def test_responses_stream_requires_explicit_terminal_completion(
    mode: str | None,
    expected_finish: str,
) -> None:
    provider = _openai_responses_provider()
    payload = _responses_sse_payload([{
        "type": "response.output_text.delta",
        "delta": "incomplete historical fact",
    }])
    stream = _TrackingStream(payload)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )

    context = ProviderCallContext(
        stream_idle_timeout_seconds=1.0,
        request_deadline_monotonic=asyncio.get_running_loop().time() + 3.0,
        source_summary_mode=mode,
        single_attempt=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="https://api.openai.com/v1",
            http_client=http_client,
            max_retries=0,
        )
        response = await provider.chat_stream_once(
            provider_context=context,
            messages=[{"role": "user", "content": "historical evidence"}],
            model="gpt-5.6",
            max_tokens=32,
            reasoning_effort="none",
        )

    assert response.finish_reason == expected_finish
    if mode is not None:
        assert response.local_usage_receipt is None
        assert response.error_kind == "summary_incomplete_stream"
    assert len(requests) == 1
    assert stream.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("marker", "expected_finish", "expected_error"),
    [
        ("", "error", "summary_incomplete_stream"),
        (False, "error", "summary_incomplete_stream"),
        (0, "error", "summary_incomplete_stream"),
        ("stop", "stop", None),
    ],
    ids=["empty-string", "false", "zero", "valid-stop"],
)
async def test_auxiliary_chat_stream_rejects_malformed_terminal_markers(
    marker: object,
    expected_finish: str,
    expected_error: str | None,
) -> None:
    provider = _local_provider(timeout=60.0)
    payload = b"data: " + json.dumps({
        "id": "probe",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": [{
            "index": 0,
            "delta": {"content": "partial historical fact"},
            "finish_reason": marker,
        }],
    }).encode() + b"\n\n"
    stream = _TrackingStream(payload)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )

    context = ProviderCallContext(
        stream_idle_timeout_seconds=1.0,
        request_deadline_monotonic=asyncio.get_running_loop().time() + 3.0,
        source_summary_mode="canonical",
        single_attempt=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="http://manager.test/v1",
            http_client=http_client,
            max_retries=0,
        )
        response = await provider.chat_stream_once(
            provider_context=context,
            messages=[{"role": "user", "content": "historical evidence"}],
            model=MODEL,
            max_tokens=32,
        )

    assert response.finish_reason == expected_finish
    if expected_error is not None:
        assert response.error_kind == expected_error
        assert response.local_usage_receipt is None
    assert len(requests) == 1
    assert stream.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_type", "wire_status", "expected_finish"),
    [
        ("response.completed", "completed", "stop"),
        ("response.completed", "in_progress", "error"),
        ("response.completed", "unknown", "error"),
        ("response.completed", "", "error"),
        ("response.incomplete", "", "error"),
    ],
    ids=[
        "completed",
        "in-progress",
        "unknown",
        "empty-status",
        "incomplete-empty-status",
    ],
)
async def test_responses_stream_rejects_inconsistent_terminal_completion(
    event_type: str,
    wire_status: str,
    expected_finish: str,
) -> None:
    provider = _openai_responses_provider()
    payload = _responses_sse_payload([
        {"type": "response.output_text.delta", "delta": "partial historical fact"},
        {
            "type": event_type,
            "response": {
                "id": "resp_synthetic",
                "status": wire_status,
                "output": [],
            },
        },
    ])
    stream = _TrackingStream(payload)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )

    context = ProviderCallContext(
        stream_idle_timeout_seconds=1.0,
        request_deadline_monotonic=asyncio.get_running_loop().time() + 3.0,
        source_summary_mode="canonical",
        single_attempt=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="https://api.openai.com/v1",
            http_client=http_client,
            max_retries=0,
        )
        response = await provider.chat_stream_once(
            provider_context=context,
            messages=[{"role": "user", "content": "historical evidence"}],
            model="gpt-5.6",
            max_tokens=32,
            reasoning_effort="none",
        )

    assert response.finish_reason == expected_finish
    if expected_finish == "error":
        assert response.error_kind == "summary_incomplete_stream"
        assert response.local_usage_receipt is None
    assert len(requests) == 1
    assert stream.closed is True


def test_prefix_summary_preserves_trailing_user_instruction() -> None:
    provider = _local_provider()
    history = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "current"},
    ]
    instruction = {"role": "user", "content": "summarize the history"}
    payload = [*history, instruction]
    generation = GenerationSettings(
        temperature=0,
        max_tokens=64,
        reasoning_effort="none",
    )

    def preview(mode: str | None) -> PreparedRequestBudgetPreview:
        context = ProviderCallContext(
            request_limits=ProviderRequestLimits(
                total_tokens=8_192,
                completion_reserve_tokens=64,
                safety_buffer_tokens=0,
            ),
            context_window_tokens=8_192,
            stream_idle_timeout_seconds=300.0,
            source_summary_mode=mode,
            single_attempt=True,
        )
        result = provider.preview_request_budget(
            messages=payload,
            tools=None,
            model=MODEL,
            generation=generation,
            provider_context=context,
            stream=True,
        )
        assert result is not None
        return result

    prefix = preview("prefix")
    assert prefix.converted_messages[-1] == instruction
    assert len(prefix.converted_messages) == len(payload)

    canonical = preview("canonical")
    assert canonical.converted_messages[-1]["content"] == (
        "current\n\nsummarize the history"
    )
    assert len(canonical.converted_messages) == len(payload) - 1


@pytest.mark.asyncio
async def test_responses_first_idle_deadline_includes_delayed_headers_and_first_event() -> None:
    provider = _openai_responses_provider()
    delta = _responses_sse_payload([{
        "type": "response.output_text.delta",
        "delta": "partially delayed",
    }])
    completed = _responses_sse_payload([{
        "type": "response.completed",
        "response": {"id": "resp-idle", "status": "completed", "output": []},
    }])
    stream = _DelayedEventStream([delta, completed], 0.07)
    requests: list[httpx.Request] = []

    class DelayedHeadersTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            requests.append(request)
            await asyncio.sleep(0.07)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
                request=request,
            )

    context = ProviderCallContext(
        stream_idle_timeout_seconds=0.1,
        request_deadline_monotonic=asyncio.get_running_loop().time() + 1.0,
        source_summary_mode="canonical",
        single_attempt=True,
    )
    async with httpx.AsyncClient(transport=DelayedHeadersTransport()) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="https://api.openai.com/v1",
            http_client=http_client,
            max_retries=0,
            timeout=60.0,
        )
        response = await provider.chat_stream_once(
            provider_context=context,
            messages=[{"role": "user", "content": "historical evidence"}],
            model="gpt-5.6",
            max_tokens=32,
            reasoning_effort="none",
        )

    assert response.finish_reason == "error"
    assert response.error_kind == "summary_idle_timeout"
    assert len(requests) == 1
    timeout_extensions = requests[0].extensions["timeout"]
    assert timeout_extensions["read"] == pytest.approx(0.1, abs=0.02)


@pytest.mark.asyncio
async def test_responses_progress_resets_idle_but_not_absolute_deadline() -> None:
    provider = _openai_responses_provider()
    payloads = [
        _responses_sse_payload([{
            "type": "response.output_text.delta",
            "delta": f"part-{index} ",
        }])
        for index in range(3)
    ]
    payloads.append(_responses_sse_payload([{
        "type": "response.completed",
        "response": {"id": "resp-deadline", "status": "completed", "output": []},
    }]))
    stream = _DelayedEventStream(payloads, 0.05)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )

    operation_started = asyncio.get_running_loop().time()
    operation_budget = 0.12
    operation_deadline = operation_started + operation_budget
    context = ProviderCallContext(
        stream_idle_timeout_seconds=0.2,
        request_deadline_monotonic=operation_deadline,
        source_summary_mode="canonical",
        single_attempt=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="https://api.openai.com/v1",
            http_client=http_client,
            max_retries=0,
            timeout=60.0,
        )
        response = await provider.chat_stream_once(
            provider_context=context,
            messages=[{"role": "user", "content": "historical evidence"}],
            model="gpt-5.6",
            max_tokens=32,
            reasoning_effort="none",
        )

    assert response.finish_reason == "error"
    assert response.error_kind == "summary_overall_timeout"
    assert len(requests) == 1
    timeout_extensions = requests[0].extensions["timeout"]
    read_timeout = timeout_extensions["read"]
    captured_remaining = operation_deadline - operation_started
    assert read_timeout <= captured_remaining
    assert read_timeout == pytest.approx(captured_remaining, abs=0.02)


@pytest.mark.asyncio
async def test_responses_callback_failure_closes_retained_sdk_response() -> None:
    provider = _openai_responses_provider()
    payload = _responses_sse_payload([
        {"type": "response.output_text.delta", "delta": "summary text"},
        {
            "type": "response.completed",
            "response": {"id": "resp-callback", "status": "completed", "output": []},
        },
    ])
    stream = _TrackingStream(payload)
    responses: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )
        responses.append(response)
        return response

    context = ProviderCallContext(
        stream_idle_timeout_seconds=1.0,
        request_deadline_monotonic=asyncio.get_running_loop().time() + 3.0,
        source_summary_mode="canonical",
        single_attempt=True,
    )

    async def fail_callback(_delta: str) -> None:
        raise RuntimeError("callback fixture failure")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="https://api.openai.com/v1",
            http_client=http_client,
            max_retries=0,
        )
        result = await provider.chat_stream_once(
            provider_context=context,
            messages=[{"role": "user", "content": "historical evidence"}],
            model="gpt-5.6",
            max_tokens=32,
            reasoning_effort="none",
            on_content_delta=fail_callback,
        )

    assert result.finish_reason == "error"
    assert len(responses) == 1
    assert responses[0].is_closed
    assert stream.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "idle_timeout", "overall_timeout"])
async def test_responses_sdk_stream_closes_on_completion_or_timeout(outcome: str) -> None:
    provider = _openai_responses_provider()
    delta = _responses_sse_payload([{
        "type": "response.output_text.delta",
        "delta": "summary text",
    }])
    completed = _responses_sse_payload([{
        "type": "response.completed",
        "response": {"id": "resp-close", "status": "completed", "output": []},
    }])
    if outcome == "success":
        stream: httpx.AsyncByteStream = _TrackingStream(delta + completed)
        idle_timeout = 1.0
        deadline_seconds = 3.0
    elif outcome == "idle_timeout":
        stream = _DelayedEventStream([delta, completed], 0.08)
        idle_timeout = 0.02
        deadline_seconds = 1.0
    else:
        stream = _DelayedEventStream([delta, completed], 0.08)
        idle_timeout = 0.2
        deadline_seconds = 0.02
    retained_responses: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )
        retained_responses.append(response)
        return response

    context = ProviderCallContext(
        stream_idle_timeout_seconds=idle_timeout,
        request_deadline_monotonic=(
            asyncio.get_running_loop().time() + deadline_seconds
        ),
        source_summary_mode="canonical",
        single_attempt=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="https://api.openai.com/v1",
            http_client=http_client,
            max_retries=0,
        )
        result = await provider.chat_stream_once(
            provider_context=context,
            messages=[{"role": "user", "content": "historical evidence"}],
            model="gpt-5.6",
            max_tokens=32,
            reasoning_effort="none",
        )

    if outcome == "success":
        assert result.finish_reason == "stop"
    else:
        assert result.finish_reason == "error"
        assert result.error_kind == (
            "summary_idle_timeout" if outcome == "idle_timeout"
            else "summary_overall_timeout"
        )
    assert len(retained_responses) == 1
    assert retained_responses[0].is_closed
    assert getattr(stream, "closed", False) is True


@pytest.mark.asyncio
async def test_responses_sdk_stream_cancellation_closes_retained_response() -> None:
    provider = _openai_responses_provider()
    block = asyncio.Event()
    payload = _responses_sse_payload([{
        "type": "response.output_text.delta",
        "delta": "summary text",
    }])
    stream = _TrackingStream(payload, block=block)
    retained_responses: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )
        retained_responses.append(response)
        return response

    context = ProviderCallContext(
        stream_idle_timeout_seconds=2.0,
        request_deadline_monotonic=asyncio.get_running_loop().time() + 3.0,
        source_summary_mode="canonical",
        single_attempt=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="https://api.openai.com/v1",
            http_client=http_client,
            max_retries=0,
        )
        task = asyncio.create_task(provider.chat_stream_once(
            provider_context=context,
            messages=[{"role": "user", "content": "historical evidence"}],
            model="gpt-5.6",
            max_tokens=32,
            reasoning_effort="none",
        ))
        await asyncio.wait_for(stream.started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(retained_responses) == 1
    assert retained_responses[0].is_closed
    assert stream.closed


@pytest.mark.asyncio
async def test_real_sdk_stream_cancellation_closes_underlying_response() -> None:
    provider = _local_provider(timeout=60.0)
    block = asyncio.Event()
    stream = _TrackingStream(_sse_payload(), block=block)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="http://manager.test/v1",
            http_client=http_client,
            max_retries=0,
        )
        task = asyncio.create_task(provider.chat_stream_once(
            provider_context=ProviderCallContext(stream_idle_timeout_seconds=300.0),
            messages=[{"role": "user", "content": "request"}],
            model=MODEL,
            max_tokens=1,
        ))
        await asyncio.wait_for(stream.started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert stream.closed is True


@pytest.mark.asyncio
async def test_first_sdk_chunk_idle_budget_includes_response_header_wait() -> None:
    provider = _local_provider(timeout=60.0)
    request_count = 0

    class DelayedBody(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self):
            await asyncio.sleep(0.12)
            yield _sse_payload()

        async def aclose(self) -> None:
            self.closed = True

    stream = DelayedBody()

    class DelayedHeadersTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            await asyncio.sleep(0.12)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
                request=request,
            )

    context = ProviderCallContext(
        stream_idle_timeout_seconds=0.2,
        request_deadline_monotonic=asyncio.get_running_loop().time() + 2.0,
        source_summary_mode="canonical",
        single_attempt=True,
    )
    async with httpx.AsyncClient(transport=DelayedHeadersTransport()) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="http://manager.test/v1",
            http_client=http_client,
            max_retries=0,
        )
        result = await provider.chat_stream_once(
            provider_context=context,
            messages=[{"role": "user", "content": "request"}],
            model=MODEL,
            max_tokens=1,
        )

    assert result.finish_reason == "error"
    assert result.error_kind == "summary_idle_timeout"
    assert request_count == 1
    assert stream.closed is True


@pytest.mark.asyncio
async def test_foreground_summary_allows_progress_beyond_legacy_sixty_second_timeout(
    monkeypatch,
) -> None:
    loop = asyncio.get_running_loop()

    class AdvancingClock:
        def __init__(self, initial: float) -> None:
            self.current = initial

        def __call__(self) -> float:
            return self.current

        def advance(self, seconds: float) -> None:
            self.current += seconds

    clock = AdvancingClock(loop.time())
    monkeypatch.setattr(loop, "time", clock)

    def event(
        content: str | None,
        finish_reason: str | None = None,
        *,
        reasoning: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> bytes:
        data: dict[str, object] = {
            "id": "chatcmpl-progressing-summary",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [] if usage is not None else [{
                "index": 0,
                "delta": (
                    {"role": "assistant", "content": content}
                    if content
                    else {"reasoning_content": reasoning} if reasoning else {}
                ),
                "finish_reason": finish_reason,
            }],
        }
        if usage is not None:
            data["usage"] = usage
        return b"data: " + json.dumps(data).encode("utf-8") + b"\n\n"

    class ProgressingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self):
            for payload in (
                event("Historical "),
                event(None, reasoning="private reasoning progress"),
                event(None, usage={"prompt_tokens": 3, "completion_tokens": 1,
                                   "total_tokens": 4}),
                event("facts "),
                event("remain useful."),
                event(None, "stop"),
                b"data: [DONE]\n\n",
            ):
                clock.advance(20.0)
                yield payload

        async def aclose(self) -> None:
            self.closed = True

    stream = ProgressingStream()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
            request=request,
        )

    provider = _local_provider(timeout=60.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="http://manager.test/v1",
            http_client=http_client,
            max_retries=0,
        )
        start = clock.current
        result = await Consolidator.summarize_prepared_source_chunk(
            [{"role": "user", "content": "historical evidence"}],
            runtime=LLMRuntime(
                provider,
                MODEL,
                GenerationSettings(max_tokens=128, temperature=0.0),
                8_192,
            ),
            max_output_tokens=128,
            request_deadline_monotonic=start + 900.0,
            stream_idle_timeout_seconds=300.0,
            source_summary_mode="canonical",
        )

    assert result.status == "complete"
    assert result.summary == "Historical facts remain useful."
    assert result.calls == 1 and len(requests) == 1
    assert clock.current - start >= 60.0
    assert stream.closed is True


@pytest.mark.asyncio
async def test_stream_cleanup_is_bounded_when_close_stalls() -> None:
    class StalledClose:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = False
            self.never = asyncio.Event()

        async def aclose(self) -> None:
            self.started.set()
            try:
                await self.never.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    closeable = StalledClose()
    deadline = asyncio.get_running_loop().time() + 0.05
    await asyncio.wait_for(
        OpenAICompatProvider._close_sdk_stream(
            None,
            closeable,
            preserve_active_error=False,
            deadline_monotonic=deadline,
        ),
        timeout=0.5,
    )
    assert closeable.started.is_set()
    await asyncio.sleep(0)
    assert closeable.cancelled is True


@pytest.mark.asyncio
async def test_stream_cleanup_ignores_close_failure() -> None:
    class RaisingClose:
        async def aclose(self) -> None:
            raise RuntimeError("close failed")

    await OpenAICompatProvider._close_sdk_stream(
        None,
        RaisingClose(),
        preserve_active_error=False,
        deadline_monotonic=None,
    )


@pytest.mark.asyncio
async def test_new_cancellation_during_success_cleanup_is_preserved() -> None:
    class WaitingClose:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = False
            self.never = asyncio.Event()

        async def aclose(self) -> None:
            self.started.set()
            try:
                await self.never.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    closeable = WaitingClose()
    task = asyncio.create_task(OpenAICompatProvider._close_sdk_stream(
        None,
        closeable,
        preserve_active_error=False,
        deadline_monotonic=None,
    ))
    await asyncio.wait_for(closeable.started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert closeable.cancelled is True


@pytest.mark.asyncio
async def test_cleanup_cancellation_does_not_replace_active_provider_error() -> None:
    class WaitingClose:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = False
            self.never = asyncio.Event()

        async def aclose(self) -> None:
            self.started.set()
            try:
                await self.never.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    closeable = WaitingClose()
    task = asyncio.create_task(OpenAICompatProvider._close_sdk_stream(
        None,
        closeable,
        preserve_active_error=True,
        deadline_monotonic=None,
    ))
    await asyncio.wait_for(closeable.started.wait(), timeout=1.0)
    task.cancel()
    await asyncio.wait_for(task, timeout=1.0)
    await asyncio.sleep(0)
    assert closeable.cancelled is True


@pytest.mark.asyncio
async def test_sync_close_cancellation_preserves_active_error_only() -> None:
    class CancelledClose:
        def close(self) -> None:
            raise asyncio.CancelledError("close cancelled")

    await OpenAICompatProvider._close_sdk_stream(
        None,
        CancelledClose(),
        preserve_active_error=True,
        deadline_monotonic=None,
    )
    with pytest.raises(asyncio.CancelledError, match="close cancelled"):
        await OpenAICompatProvider._close_sdk_stream(
            None,
            CancelledClose(),
            preserve_active_error=False,
            deadline_monotonic=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_name", "spec_name"),
    [("vllm", "vllm"), ("local_manager", "openai"), ("custom", "local_manager")],
)
async def test_relaxed_policy_requires_both_local_manager_identities(
    provider_name: str,
    spec_name: str,
) -> None:
    provider = _local_provider(provider_name=provider_name, spec_name=spec_name)
    assert provider._request_budget_policy == "strict"
    body = provider._build_kwargs(
        [{"role": "user", "content": "界" * 2_000}],
        None,
        MODEL,
        1,
        0.0,
        None,
        None,
    )
    with pytest.raises(FinalizedRequestBudgetExceeded):
        provider.validate_finalized_request(
            body,
            transport="openai_chat_completions",
            model=MODEL,
            provider_context=_context(5_000),
            policy=provider._request_budget_policy,
        )


def test_local_cl100k_is_labeled_and_special_marker_text_is_counted_as_text() -> None:
    body = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": "<|endoftext|> function f(): return '界'",
        }],
    }
    measurement = validate_finalized_request(
        TransportIdentity("local_manager", "openai_chat_completions", MODEL),
        body,
        ProviderRequestLimits(total_tokens=10_000, safety_buffer_tokens=0),
        policy=LOCAL_MANAGER_POLICY,
    )
    assert measurement.source == LOCAL_MANAGER_POLICY
    assert measurement.policy == LOCAL_MANAGER_POLICY
    assert measurement.measured_bytes is not None
    assert measurement.measured_bytes > measurement.input_tokens


def test_reported_usage_floor_requires_same_converted_prefix_and_sdk_client_scope() -> None:
    provider = _local_provider()
    provider._client = AsyncOpenAI(
        api_key="fixture-credential",
        base_url="http://manager.test/v1",
    )
    context = ProviderCallContext(request_limits=ProviderRequestLimits(
        total_tokens=100_000,
        completion_reserve_tokens=1,
        safety_buffer_tokens=0,
    ))
    initial_messages = [{"role": "user", "content": "initial request"}]
    _kwargs, previous_body = provider._prepare_chat_request(
        initial_messages,
        None,
        MODEL,
        1,
        0.0,
        None,
        None,
        stream=False,
        provider_context=context,
    )
    receipt = provider._make_local_usage_receipt(previous_body, context)
    assert type(receipt) is LocalRequestUsageReceipt
    anchor = LocalUsageAnchor(receipt=receipt, prompt_tokens=5_000)
    anchored_context = replace(context, local_usage_anchor=anchor)

    appended_messages = initial_messages + [
        {"role": "assistant", "content": "a short completed reply"},
        {"role": "user", "content": "the next request"},
    ]
    _kwargs, appended_body = provider._prepare_chat_request(
        appended_messages,
        None,
        MODEL,
        1,
        0.0,
        None,
        None,
        stream=False,
        provider_context=anchored_context,
    )
    anchored = provider._measure_local_body(
        appended_body,
        transport="openai_chat_completions",
        model=MODEL,
        provider_context=anchored_context,
    )
    assert anchored.source == "reported_anchor_plus_estimated_delta"
    assert anchored.input_tokens > 5_000

    for role in ("system", "developer"):
        changed_messages = appended_messages + [{
            "role": role,
            "content": "new instruction after the usage receipt",
        }]
        _kwargs, changed_body = provider._prepare_chat_request(
            changed_messages,
            None,
            MODEL,
            1,
            0.0,
            None,
            None,
            stream=False,
            provider_context=anchored_context,
        )
        changed_anchor_measurement = provider._measure_local_body(
            changed_body,
            transport="openai_chat_completions",
            model=MODEL,
            provider_context=anchored_context,
        )
        full_measurement = provider._measure_local_body(
            changed_body,
            transport="openai_chat_completions",
            model=MODEL,
            provider_context=context,
        )
        assert changed_anchor_measurement.source != "reported_anchor_plus_estimated_delta"
        assert changed_anchor_measurement == full_measurement

    # Cached input is part of prompt_tokens; it is never added a second time.
    response = provider._parse(
        SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(
                prompt_tokens=3,
                completion_tokens=1,
                total_tokens=4,
                prompt_tokens_details=SimpleNamespace(cached_tokens=2),
            ),
            model=MODEL,
        )
    )
    assert response.usage is not None
    assert response.usage.input_tokens == 3
    assert response.usage.cache_read_tokens == 2

    old_scope = receipt.provider_scope
    provider._client.api_key = "rotated-credential"
    assert provider._local_scope_binding() != old_scope
    repriced = provider._measure_local_body(
        appended_body,
        transport="openai_chat_completions",
        model=MODEL,
        provider_context=anchored_context,
    )
    assert repriced.source == LOCAL_MANAGER_POLICY
    assert repriced.input_tokens < anchored.input_tokens

    provider._client = AsyncOpenAI(
        api_key="fixture-credential",
        base_url="http://manager.test/v1",
    )
    assert provider._local_scope_binding() != old_scope
    replaced_client = provider._measure_local_body(
        appended_body,
        transport="openai_chat_completions",
        model=MODEL,
        provider_context=anchored_context,
    )
    assert replaced_client.source == LOCAL_MANAGER_POLICY
    assert replaced_client.input_tokens < anchored.input_tokens

    provider._client.base_url = httpx.URL("http://manager.changed/v1/")
    assert provider._local_scope_binding() != old_scope


def test_reported_usage_floor_declines_changed_input_and_effective_envelopes() -> None:
    provider = _local_provider()
    context = ProviderCallContext(request_limits=ProviderRequestLimits(
        total_tokens=100_000,
        completion_reserve_tokens=1,
        safety_buffer_tokens=0,
    ))
    initial = [{"role": "user", "content": "original request"}]
    _kwargs, receipt_body = provider._prepare_chat_request(
        initial,
        None,
        MODEL,
        1,
        0.0,
        None,
        None,
        stream=False,
        provider_context=context,
    )
    receipt = provider._make_local_usage_receipt(receipt_body, context)
    assert isinstance(receipt, LocalRequestUsageReceipt)
    anchored_context = replace(
        context,
        local_usage_anchor=LocalUsageAnchor(receipt=receipt, prompt_tokens=10_000),
    )
    ordinary_append = initial + [
        {"role": "assistant", "content": "prior response"},
        {"role": "user", "content": "next request"},
    ]
    tool = [{
        "type": "function",
        "function": {"name": "changed_tool", "parameters": {"type": "object"}},
    }]
    media_message = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "original request"},
            {"type": "image_url", "image_url": {"url": _png_data_uri(64, 64)}},
        ],
    }, *ordinary_append[1:]]

    def assert_declined(
        messages: list[dict[str, object]],
        *,
        model: str = MODEL,
        tools: list[dict[str, object]] | None = None,
        max_tokens: int = 1,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        measured_context: ProviderCallContext = anchored_context,
    ) -> None:
        _sdk_kwargs, body = provider._prepare_chat_request(
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            reasoning_effort,
            None,
            stream=False,
            provider_context=measured_context,
        )
        anchored_measurement = provider._measure_local_body(
            body,
            transport="openai_chat_completions",
            model=model,
            provider_context=measured_context,
        )
        full_measurement = provider._measure_local_body(
            body,
            transport="openai_chat_completions",
            model=model,
            provider_context=replace(measured_context, local_usage_anchor=None),
        )
        assert anchored_measurement.source != "reported_anchor_plus_estimated_delta"
        assert anchored_measurement == full_measurement

    changed_input = [
        {"role": "user", "content": "replaced original request"},
        *ordinary_append[1:],
    ]
    deleted_row = ordinary_append[1:]
    merged_role = [
        {"role": "assistant", "content": "original request; prior response"},
        {"role": "user", "content": "next request"},
    ]
    assert_declined(changed_input)
    assert_declined(deleted_row)
    assert_declined(merged_role)
    assert_declined(ordinary_append, tools=tool)
    assert_declined(ordinary_append, model=f"{MODEL}-changed")
    assert_declined(ordinary_append, max_tokens=2)
    assert_declined(ordinary_append, temperature=0.25)
    assert_declined(ordinary_append, reasoning_effort="high")
    assert_declined(
        ordinary_append,
        measured_context=replace(
            anchored_context,
            request_limits=ProviderRequestLimits(
                total_tokens=100_001,
                completion_reserve_tokens=1,
                safety_buffer_tokens=0,
            ),
        ),
    )
    assert_declined(media_message)

    old_extra_body = dict(provider._extra_body)
    try:
        provider._extra_body = {"metadata": {"changed": True}}
        assert_declined(ordinary_append)
    finally:
        provider._extra_body = old_extra_body


@pytest.mark.asyncio
async def test_payload_byte_ceiling_is_independent_and_applies_without_token_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert budget_module.LOCAL_MANAGER_MAX_REQUEST_BYTES == 64 * 1024 * 1024
    metadata = "data:image/png;base64," + ("A" * 9_000)
    provider = _local_provider(extra_body={"metadata": {"inline": metadata}})
    messages = [{"role": "user", "content": "x"}]
    sdk_kwargs, body = provider._prepare_chat_request(
        messages, None, MODEL, 1, 0.0, None, None, stream=False,
    )
    encoded_size = len(json.dumps(
        body,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8"))
    monkeypatch.setattr(budget_module, "LOCAL_MANAGER_MAX_REQUEST_BYTES", encoded_size)
    measured = provider.validate_finalized_request(
        body,
        transport="openai_chat_completions",
        model=MODEL,
        provider_context=None,
        policy=LOCAL_MANAGER_POLICY,
    )
    assert measured is not None
    assert measured.input_limit_tokens is None
    assert measured.measured_bytes == encoded_size
    assert measured.limit_bytes == encoded_size

    attempts = AsyncMock(return_value=LLMResponse(content="ok"))
    provider._ensure_client = AsyncMock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=attempts)),
    ))
    monkeypatch.setattr(
        budget_module,
        "LOCAL_MANAGER_MAX_REQUEST_BYTES",
        encoded_size - 1,
    )
    with pytest.raises(RequestPayloadBytesExceeded) as exceeded:
        await provider.chat(messages=messages, max_tokens=1)
    assert exceeded.value.reason == "payload_bytes"
    assert exceeded.value.measured_bytes == encoded_size
    assert exceeded.value.limit_bytes == encoded_size - 1
    attempts.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra_body", "expected_field"),
    [
        ({"model": "other-model"}, "model"),
        ({"messages": [{"role": "user", "content": "different"}]}, "messages"),
    ],
)
async def test_governed_conflicting_model_or_messages_never_dispatch(
    extra_body: dict[str, object],
    expected_field: str,
) -> None:
    provider = _local_provider(extra_body=extra_body)
    create = AsyncMock()
    provider._ensure_client = AsyncMock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    with pytest.raises(GovernedRequestConflictError) as conflict:
        await provider.chat_with_context(
            provider_context=_context(),
            messages=[{"role": "user", "content": "original"}],
            model=MODEL,
            max_tokens=1,
        )
    assert conflict.value.field_name == expected_field
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_canonical_responses_summary_rejects_hosted_tools_before_dispatch() -> None:
    provider = OpenAICompatProvider(
        api_key="fixture-credential",
        api_base="https://api.openai.com/v1",
        default_model="gpt-5.6",
        provider_name="openai",
        spec=find_by_name("openai"),
        api_type="responses",
        extra_body={"tools": [{"type": "web_search"}]},
    )
    provider._hosted_web_search_enabled = lambda: True
    create = AsyncMock()
    provider._ensure_client = AsyncMock(return_value=SimpleNamespace(
        responses=SimpleNamespace(create=create),
    ))

    with pytest.raises(ProviderRequestRejected) as rejected:
        await provider.chat_stream_once(
            provider_context=ProviderCallContext(
                source_summary_mode="canonical",
                single_attempt=True,
            ),
            messages=[{"role": "user", "content": "summary evidence"}],
            model="gpt-5.6",
            max_tokens=128,
        )

    assert rejected.value.reason == "unsupported_auxiliary_envelope"
    create.assert_not_awaited()
    legacy_body = provider._build_responses_body(
        [{"role": "user", "content": "direct request"}],
        None,
        "gpt-5.6",
        128,
        0.0,
        None,
        None,
        provider_context=None,
    )
    assert any(
        isinstance(tool, dict) and tool.get("type") == "web_search"
        for tool in legacy_body.get("tools", [])
    )


def test_auxiliary_output_cap_above_summary_bound_is_rejected_in_preview() -> None:
    provider = _local_provider(extra_body={"max_completion_tokens": 2_049})
    context = ProviderCallContext(
        request_limits=ProviderRequestLimits(
            total_tokens=8_192,
            completion_reserve_tokens=2_048,
            safety_buffer_tokens=0,
        ),
        source_summary_mode="canonical",
    )

    with pytest.raises(ProviderRequestRejected) as rejected:
        provider.preview_request_budget(
            messages=[{"role": "user", "content": "summary evidence"}],
            tools=None,
            model=MODEL,
            generation=GenerationSettings(
                temperature=0.0,
                max_tokens=2_048,
                reasoning_effort="none",
            ),
            provider_context=context,
            stream=True,
        )

    assert rejected.value.reason == "unsupported_auxiliary_envelope"


@pytest.mark.asyncio
async def test_malformed_body_has_no_transport_and_unsupported_local_media_stays_closed() -> None:
    provider = _local_provider(extra_body={"metadata": {"not_finite": float("nan")}})
    create = AsyncMock()
    provider._ensure_client = AsyncMock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    with pytest.raises(ProviderRequestRejected) as malformed:
        await provider.chat_with_context(
            provider_context=_context(),
            messages=[{"role": "user", "content": "original"}],
            max_tokens=1,
        )
    assert malformed.value.reason == "invalid_request_envelope"
    create.assert_not_awaited()

    unsupported_body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [{"type": "video", "url": "x"}]}],
    }
    with pytest.raises(ProviderRequestRejected):
        validate_finalized_request(
            TransportIdentity("local_manager", "openai_chat_completions", MODEL),
            unsupported_body,
            _context().request_limits,
            policy=LOCAL_MANAGER_POLICY,
        )


@pytest.mark.parametrize("envelope", [False, True])
def test_pinned_backend_fixture_classifies_direct_and_one_error_envelope(
    envelope: bool,
) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["provenance"]["values_are_historical_actuals"] is True
    body = fixture["body"]
    payload = {"error": body} if envelope else body
    error = SimpleNamespace(status_code=400, body=payload)
    overflow = OpenAICompatProvider._backend_context_overflow(error)
    assert isinstance(overflow, BackendContextOverflow)
    assert overflow.status_code == 400
    assert overflow.code == 400
    assert overflow.error_type == "exceed_context_size_error"
    assert overflow.n_prompt_tokens == 227_502
    assert overflow.n_ctx == 165_888
    assert "227502" not in str(overflow)


@pytest.mark.parametrize(
    "error",
    [
        SimpleNamespace(status_code=400, body={"code": 400, "type": "other", "n_prompt_tokens": 2, "n_ctx": 1}),
        SimpleNamespace(status_code=413, body={"code": 400, "type": "exceed_context_size_error", "n_prompt_tokens": 2, "n_ctx": 1}),
        SimpleNamespace(status_code=400, body={"code": True, "type": "exceed_context_size_error", "n_prompt_tokens": 2, "n_ctx": 1}),
        SimpleNamespace(status_code=400, body={"code": 400, "type": "exceed_context_size_error", "n_prompt_tokens": "2", "n_ctx": 1}),
        SimpleNamespace(status_code=400, body={"code": 400, "type": "exceed_context_size_error", "n_ctx": 1}),
        SimpleNamespace(status_code=400, body={"code": 400, "type": "exceed_context_size_error", "n_prompt_tokens": 2}),
        SimpleNamespace(status_code=400, body={"code": 400, "type": "exceed_context_size_error", "n_prompt_tokens": 2, "n_ctx": True}),
        SimpleNamespace(status_code=400, body={
            "code": 400,
            "type": "exceed_context_size_error",
            "n_prompt_tokens": 3,
            "n_ctx": 1,
            "error": {
                "code": 400,
                "type": "exceed_context_size_error",
                "n_prompt_tokens": 2,
                "n_ctx": 1,
            },
        }),
        SimpleNamespace(status_code=400, body="context size exceeded"),
    ],
)
def test_generic_or_malformed_errors_do_not_classify_as_context_overflow(error: object) -> None:
    assert OpenAICompatProvider._backend_context_overflow(error) is None  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_call", [False, True], ids=["chat-once", "stream-once"])
@pytest.mark.parametrize("tools", [None, [{
    "type": "function",
    "function": {"name": "fixture_tool", "parameters": {"type": "object"}},
}],], ids=["no-tools", "with-tools"])
async def test_backend_overflow_from_real_sdk_is_one_observed_nonretryable_transport(
    stream_call: bool,
    tools: list[dict[str, object]] | None,
) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            fixture["status_code"],
            json=fixture["body"],
            request=request,
        )

    provider = _local_provider()
    observations: list[object] = []
    transport_attempts: list[str] = []
    provider.set_llm_call_observer(observations.append)
    context = replace(_context(), on_transport_attempt=transport_attempts.append)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="http://manager.test/v1",
            http_client=http_client,
            max_retries=0,
        )
        with pytest.raises(BackendContextOverflow):
            call = (
                provider.chat_stream_once
                if stream_call
                else provider.chat_once
            )
            await call(
                provider_context=context,
                messages=[{"role": "user", "content": "request"}],
                tools=tools,
                model=MODEL,
                max_tokens=1,
            )
    assert len(captured) == 1
    assert transport_attempts == ["main"]
    assert len(observations) == 1
    record = observations[0]
    assert getattr(record, "finish_reason") == "error"
    assert getattr(record, "error_status_code") == 400
    assert getattr(record, "error_kind") == "backend_context_overflow"
    assert "227502" not in repr(record) and "165888" not in repr(record)


@pytest.mark.asyncio
async def test_predispatch_finalized_rejection_has_no_overflow_observer_record() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_success_body(), request=request)

    provider = _local_provider()
    observations: list[object] = []
    attempts: list[str] = []
    provider.set_llm_call_observer(observations.append)
    context = ProviderCallContext(
        request_limits=ProviderRequestLimits(total_tokens=8, safety_buffer_tokens=0),
        on_transport_attempt=attempts.append,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="http://manager.test/v1",
            http_client=http_client,
            max_retries=0,
        )
        with pytest.raises(FinalizedRequestBudgetExceeded):
            await provider.chat_once(
                provider_context=context,
                messages=[{"role": "user", "content": "oversized preflight request"}],
                model=MODEL,
                max_tokens=1,
            )

    assert captured == []
    assert attempts == []
    assert observations == []


@pytest.mark.asyncio
async def test_backend_overflow_through_nested_wrappers_is_observed_once_without_rotation() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            fixture["status_code"],
            json=fixture["body"],
            request=request,
        )

    first_leaf = _local_provider()
    second_leaf = _local_provider()
    pool = PooledProvider([first_leaf, second_leaf], ["key-one", "key-two"])
    fallback = FallbackProvider(
        pool,
        [SimpleNamespace(model="never-selected", max_tokens=1, temperature=0.0)],
        lambda _preset: (_ for _ in ()).throw(AssertionError("one-shot overflow cannot fall back")),
    )
    observations: list[object] = []
    transport_attempts: list[str] = []
    fallback.set_llm_call_observer(observations.append)
    context = replace(_context(), on_transport_attempt=transport_attempts.append)
    order_before = pool._pool.request_order()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        for leaf in (first_leaf, second_leaf):
            leaf._client = AsyncOpenAI(
                api_key="fixture-credential",
                base_url="http://manager.test/v1",
                http_client=http_client,
                max_retries=0,
            )
        with pytest.raises(BackendContextOverflow):
            await fallback.chat_once(
                provider_context=context,
                messages=[{"role": "user", "content": "request"}],
                model=MODEL,
                max_tokens=1,
            )

    assert len(captured) == 1
    assert transport_attempts == ["main"]
    assert len(observations) == 1
    record = observations[0]
    assert getattr(record, "error_kind") == "backend_context_overflow"
    assert getattr(record, "error_status_code") == 400
    assert "227502" not in repr(record) and "165888" not in repr(record)
    assert fallback._primary_failures == 0
    assert pool._pool.request_order() == order_before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("summary_mode", "effective_cap", "should_reject"),
    [("canonical", 4096, True), ("canonical", 64, False), (None, 4096, False)],
    ids=["auxiliary-over-cap", "auxiliary-within-cap", "main-override-preserved"],
)
async def test_hosted_chat_effective_output_cap_is_checked_before_dispatch(
    summary_mode: str | None,
    effective_cap: int,
    should_reject: bool,
) -> None:
    provider = OpenAICompatProvider(
        api_key="fixture-credential",
        api_base="https://api.openai.com/v1",
        default_model="gpt-4o",
        provider_name="openai",
        spec=find_by_name("openai"),
        extra_body={"max_completion_tokens": effective_cap},
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_success_body(), request=request)

    context = None if summary_mode is None else ProviderCallContext(
        request_limits=ProviderRequestLimits(
            total_tokens=8_192,
            completion_reserve_tokens=128,
            safety_buffer_tokens=0,
        ),
        source_summary_mode=summary_mode,
        single_attempt=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="https://api.openai.com/v1",
            http_client=http_client,
            max_retries=0,
        )
        if should_reject:
            with pytest.raises(ProviderRequestRejected) as rejected:
                await provider.chat_once(
                    provider_context=context,
                    messages=[{"role": "user", "content": "summary evidence"}],
                    model="gpt-4o",
                    max_tokens=128,
                )
            assert rejected.value.reason == "unsupported_auxiliary_envelope"
        else:
            result = await provider.chat_once(
                provider_context=context,
                messages=[{"role": "user", "content": "summary evidence"}],
                model="gpt-4o",
                max_tokens=128,
            )
            assert result.finish_reason == "stop"

    assert len(requests) == (0 if should_reject else 1)
    if requests:
        assert json.loads(requests[0].content)["max_completion_tokens"] == effective_cap


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("summary_mode", "effective_cap", "should_reject"),
    [("canonical", 4096, True), ("canonical", 64, False), (None, 4096, False)],
    ids=["auxiliary-over-cap", "auxiliary-within-cap", "main-override-preserved"],
)
async def test_hosted_responses_effective_output_cap_is_checked_before_dispatch(
    summary_mode: str | None,
    effective_cap: int,
    should_reject: bool,
) -> None:
    provider = _openai_responses_provider(
        extra_body={"max_output_tokens": effective_cap},
    )
    response_events = _responses_sse_payload([
        {"type": "response.output_text.delta", "delta": "bounded summary"},
        {
            "type": "response.completed",
            "response": {"id": "resp-cap", "status": "completed", "output": []},
        },
    ])
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=response_events,
            request=request,
        )

    context = None if summary_mode is None else ProviderCallContext(
        request_limits=ProviderRequestLimits(
            total_tokens=8_192,
            completion_reserve_tokens=128,
            safety_buffer_tokens=0,
        ),
        stream_idle_timeout_seconds=1.0,
        request_deadline_monotonic=asyncio.get_running_loop().time() + 3.0,
        source_summary_mode=summary_mode,
        single_attempt=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        provider._client = AsyncOpenAI(
            api_key="fixture-credential",
            base_url="https://api.openai.com/v1",
            http_client=http_client,
            max_retries=0,
        )
        if should_reject:
            with pytest.raises(ProviderRequestRejected) as rejected:
                await provider.chat_stream_once(
                    provider_context=context,
                    messages=[{"role": "user", "content": "summary evidence"}],
                    model="gpt-5.6",
                    max_tokens=128,
                    reasoning_effort="none",
                )
            assert rejected.value.reason == "unsupported_auxiliary_envelope"
        else:
            result = await provider.chat_stream_once(
                provider_context=context,
                messages=[{"role": "user", "content": "summary evidence"}],
                model="gpt-5.6",
                max_tokens=128,
                reasoning_effort="none",
            )
            assert result.finish_reason == "stop"

    assert len(requests) == (0 if should_reject else 1)
    if requests:
        assert json.loads(requests[0].content)["max_output_tokens"] == effective_cap
