"""Exercise WP-2 request budgets, provider routing, and cache payloads."""

import json
import math
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import tiktoken
from openai import AsyncOpenAI

from kohaku.context.budget import TokenMeter
from kohaku.providers.azure_openai_provider import AzureOpenAIProvider
from kohaku.providers.base import GenerationSettings, ProviderCallContext
from kohaku.providers.bedrock_provider import BedrockProvider
from kohaku.providers.fallback_provider import FallbackProvider
from kohaku.providers.openai_compat_provider import OpenAICompatProvider
from kohaku.providers.registry import find_by_name
from kohaku.providers.request_budget import (
    FinalizedRequestBudgetExceeded,
    LocalUsageAnchor,
    ProviderRequestLimits,
    RequestPayloadBytesExceeded,
    TransportIdentity,
    validate_finalized_request,
)
from kohaku.providers.xai_grok_provider import XAIGrokProvider
from kohaku.utils.helpers import estimate_message_tokens, estimate_prompt_tokens_chain


@pytest.mark.parametrize(
    ("provider", "transport", "model"),
    [
        ("anthropic", "anthropic_messages", "claude-sonnet-4-20250514"),
        ("bedrock", "bedrock_converse", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
        ("bedrock", "bedrock_converse", "amazon.nova-pro-v1:0"),
    ],
)
def test_native_strict_60k_prompt_has_separate_token_and_byte_limits(provider, transport, model):
    text = " context" * 60_000
    if provider == "anthropic":
        body = {"model": model, "messages": [{"role": "user", "content": text}]}
    else:
        body = {"modelId": model, "messages": [{"role": "user", "content": [{"text": text}]}]}
    encoder = tiktoken.get_encoding("cl100k_base")
    assert len(encoder.encode(text)) == 60_000
    identity = TransportIdentity(provider, transport, model)
    limits = ProviderRequestLimits(total_tokens=80_000, safety_buffer_tokens=0)
    serialized = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    measured = validate_finalized_request(identity, body, limits)
    assert measured.input_tokens == math.ceil(len(encoder.encode(serialized)) * 1.15)
    assert measured.source == "strict_cl100k_calibrated"
    assert measured.measured_bytes > 240_000
    with pytest.raises(FinalizedRequestBudgetExceeded):
        validate_finalized_request(identity, body, ProviderRequestLimits(total_tokens=60_000))
    with pytest.raises(RequestPayloadBytesExceeded):
        validate_finalized_request(identity, body, limits, max_request_bytes=240_000)


def test_summary_growth_counts_tool_characters_as_tokens():
    reserve = TokenMeter.compute_summary_reserve(32_000, 8_000, 512, 300)
    without_growth = TokenMeter.compute_summary_reserve(32_000, 0, 512, 300)
    assert reserve - without_growth == 2_000


def test_prompt_counters_include_both_structured_thinking_text_shapes():
    blocks = [
        {"type": "thinking", "thinking": "First structured explanation with useful facts."},
        {"type": "text", "text": "Second structured explanation with useful facts."},
    ]
    base = {"role": "assistant", "content": None}
    message = {**base, "thinking_blocks": blocks}
    assert estimate_prompt_tokens_chain(object(), "fixture", [message])[0] > (
        estimate_prompt_tokens_chain(object(), "fixture", [base])[0]
    )
    for block in blocks:
        partial = {**base, "thinking_blocks": [block]}
        assert estimate_prompt_tokens_chain(object(), "fixture", [message])[0] > (
            estimate_prompt_tokens_chain(object(), "fixture", [partial])[0]
        )
        assert estimate_message_tokens(message) > estimate_message_tokens(partial)


def _sdk_chat_capture(provider):
    create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop",
        )],
        usage=None,
    ))
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return create


async def test_openrouter_retains_rolling_marker_before_empty_tool_call():
    provider = OpenAICompatProvider(
        api_key="fixture", api_base="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-sonnet-4", spec=find_by_name("openrouter"),
        provider_name="openrouter",
    )
    create = _sdk_chat_capture(provider)
    call = {"id": "call", "type": "function", "function": {"name": "fetch", "arguments": "{}"}}
    await provider.chat([
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Cache this text."},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call", "content": "Result"},
    ])
    messages = create.await_args.kwargs["messages"]
    assert messages[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert messages[1]["content"][-1]["text"] == "Cache this text."
    assert messages[2]["content"] is None


@pytest.mark.parametrize(
    ("model", "tool_markers"),
    [
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", True),
        ("us.anthropic.claude-3-7-sonnet-20250219-v1:0", True),
        ("global.anthropic.claude-sonnet-4-20250514-v1:0", True),
        ("amazon.nova-pro-v1:0", False),
    ],
)
async def test_bedrock_cache_points_match_model_placements(model, tool_markers):
    call = Mock(return_value={
        "output": {"message": {"content": [{"text": "done"}]}}, "stopReason": "end_turn",
    })
    provider = BedrockProvider(client=SimpleNamespace(converse=call), region="fixture")
    await provider.chat(
        [
            {"role": "system", "content": "Static instruction"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up"},
        ],
        tools=[
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "mcp_fixture_search"}},
        ],
        model=model,
    )
    request = call.call_args.kwargs
    marker = {"cachePoint": {"type": "default"}}
    assert request["system"][-1] == marker
    assert request["messages"][-2]["content"][-1] == marker
    assert sum(row == marker for row in request["toolConfig"]["tools"]) == (2 if tool_markers else 0)
    if tool_markers:
        assert request["toolConfig"]["tools"][1::2] == [marker, marker]


async def test_grok_headers_are_stable_and_session_scoped(monkeypatch):
    monkeypatch.setattr(
        "kohaku.providers.xai_grok_provider.get_xai_oauth_token",
        lambda *args, **kwargs: SimpleNamespace(access="fixture"),
    )
    transport = AsyncMock(return_value=("done", [], "stop", None, None))
    monkeypatch.setattr("kohaku.providers.xai_grok_provider._request_xai", transport)
    provider = XAIGrokProvider(extra_body={"tools": []})
    for session_id in ("first", "first", "second", "second"):
        await provider.chat_with_context(
            messages=[{"role": "user", "content": "hello"}],
            provider_context=ProviderCallContext(session_id=session_id),
        )
    pairs = [(call.args[1]["x-grok-conv-id"], call.args[1]["x-grok-session-id"])
             for call in transport.await_args_list]
    assert pairs[0] == pairs[1] and pairs[2] == pairs[3]
    assert pairs[0] != pairs[2]
    assert all(left == right for left, right in pairs)


async def test_prompt_cache_keys_use_only_supported_provider_payloads(monkeypatch):
    request = {"messages": [{"role": "user", "content": "hello"}]}
    context = ProviderCallContext(session_id="fixture session")
    expected = sha256(b"fixture session").hexdigest()
    for base, key in (
        ("https://api.openai.com/v1", expected),
        ("https://api.openai.com.example.test/v1", None),
    ):
        provider = OpenAICompatProvider(api_key="fixture", api_base=base)
        capture = _sdk_chat_capture(provider)
        await provider.chat_with_context(provider_context=context, **request)
        assert capture.await_args.kwargs.get("prompt_cache_key") == key

    responses = AsyncMock(return_value=SimpleNamespace(output=[], status="completed"))
    provider = OpenAICompatProvider(
        api_key="fixture", api_base="https://api.openai.com/v1",
        spec=find_by_name("openai"), api_type="responses",
    )
    provider._client = SimpleNamespace(responses=SimpleNamespace(create=responses))
    await provider.chat_with_context(provider_context=context, **request)
    assert responses.await_args.kwargs["prompt_cache_key"] == expected
    responses.reset_mock()

    monkeypatch.setattr("kohaku.providers.azure_openai_provider.AsyncOpenAI",
                        lambda **kwargs: SimpleNamespace(responses=SimpleNamespace(create=responses)))
    provider = AzureOpenAIProvider(api_key="fixture", api_base="https://fixture.openai.azure.com")
    await provider.chat_with_context(provider_context=context, **request)
    assert responses.await_args.kwargs["prompt_cache_key"] == expected


async def test_failover_route_does_not_calibrate_from_the_previous_backend():
    generation = GenerationSettings(max_tokens=256, temperature=0.1)
    spec = find_by_name("local_manager")
    primary = OpenAICompatProvider(
        api_key="primary", api_base="http://primary.test/v1", default_model="primary-model",
        spec=spec, provider_name="primary",
    )
    secondary = OpenAICompatProvider(
        api_key="secondary", api_base="http://secondary.test/v1", default_model="secondary-model",
        spec=spec, provider_name="secondary",
    )
    primary.generation = secondary.generation = generation
    entry = SimpleNamespace(
        provider="secondary", model="secondary-model", context_window_tokens=30_000,
        max_tokens=256, temperature=0.1, reasoning_effort=None,
    )
    route = FallbackProvider(primary, [entry], lambda preset: secondary)
    calls = []

    def transport(request):
        body = json.loads(request.content)
        calls.append((request.url.host, body))
        if request.url.host == "primary.test" and len(calls) > 1:
            return httpx.Response(401, json={"error": {
                "type": "authentication_error", "code": "invalid_api_key", "message": "bad key",
            }})
        return httpx.Response(200, json={
            "id": "chatcmpl-fixture", "object": "chat.completion", "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10_000, "completion_tokens": 2, "total_tokens": 10_002},
        })

    context = ProviderCallContext(request_limits=ProviderRequestLimits(total_tokens=30_000))
    first = [{"role": "user", "content": "First request"}]
    appended = [*first, {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "Next request"}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        for leaf in (primary, secondary):
            leaf._client = AsyncOpenAI(
                api_key=leaf.api_key, base_url=leaf.api_base, http_client=client, max_retries=0,
            )
        response = await route.chat_with_retry(first, provider_context=context)
        assert response.local_usage_receipt is not None and response.usage is not None
        anchored = replace(context, local_usage_anchor=LocalUsageAnchor(
            response.local_usage_receipt, response.usage.input_tokens,
        ))

        def preview(request_context):
            result = route.preview_request_budget(
                messages=appended, tools=None, model="primary-model", generation=generation,
                provider_context=request_context, stream=False,
            )
            assert result is not None
            return result.measurement

        primary_estimate = preview(anchored)
        assert primary_estimate.source == "reported_anchor_plus_estimated_delta"
        assert primary_estimate.input_tokens > 10_000

        result = await route.chat_with_retry(appended, provider_context=anchored)
        assert result.content == "Answer"
        assert [host for host, _ in calls] == ["primary.test", "primary.test", "secondary.test"]
        assert calls[-1][1]["model"] == "secondary-model"

        switched = replace(anchored, route_candidate_index=1)
        after_switch = preview(switched)
        assert after_switch == preview(replace(switched, local_usage_anchor=None))
        assert after_switch.identity.model == "secondary-model"
        assert after_switch.source == "estimated_compat"


async def test_openrouter_keeps_cache_marker_on_narrated_tool_turn_and_summary_block():
    from kohaku.session.summary import conversation_summary_block

    provider = OpenAICompatProvider(
        api_key="fixture", api_base="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-sonnet-4-5", spec=find_by_name("openrouter"),
        provider_name="openrouter",
    )
    capture = _sdk_chat_capture(provider)
    await provider.chat(
        messages=[
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": conversation_summary_block("## 1. Active Objective & Constraints\nKeep goal.")},
            {"role": "user", "content": "Check auth.py"},
            {
                "role": "assistant",
                "content": "Let me read auth.py now.",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "def login(): pass"},
        ],
        model="anthropic/claude-sonnet-4-5",
    )
    sent = capture.await_args.kwargs["messages"]
    # Narrated tool-call assistant turn is skipped so a surviving non-system message retains cache_control.
    non_system_marked = [
        m for m in sent[1:]
        if isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and b.get("cache_control") for b in m["content"])
    ]
    assert len(non_system_marked) == 1
    # Compacted summary remains a separate leading user block separated by a synthetic assistant turn.
    assert sent[1]["role"] == "user" and "<conversation_summary>" in str(sent[1]["content"])
    assert sent[2] == {"role": "assistant", "content": "Understood."}
    assert sent[3]["role"] == "user" and "Check auth.py" in str(sent[3]["content"])


async def test_anthropic_and_bedrock_preserve_history_cache_when_system_and_tools_split():
    from kohaku.context.types import SYSTEM_BLOCKS_META
    from kohaku.providers.anthropic_provider import AnthropicProvider

    system_blocks = [{"type": "text", "text": "Tier 1"}, {"type": "text", "text": "\n\n---\n\nTier 2"}]
    messages = [
        {"role": "system", "content": "Tier 1\n\n---\n\nTier 2", "_meta": {SYSTEM_BLOCKS_META: system_blocks}},
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Turn 2"},
    ]
    tools = [
        {"type": "function", "function": {"name": "read_file", "description": "r", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "mcp_s_t", "description": "m", "parameters": {"type": "object", "properties": {}}}},
    ]
    anth = AnthropicProvider(api_key="fixture", default_model="claude-sonnet-4-5")
    anth_kwargs = anth._build_kwargs(
        messages=messages, tools=tools, model="claude-sonnet-4-5",
        max_tokens=256, temperature=0.0, reasoning_effort=None, tool_choice=None, direct_tool_count=1,
    )
    assert anth_kwargs["messages"][-2]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert sum(1 for b in anth_kwargs["system"] if "cache_control" in b) == 2
    assert sum(1 for t in anth_kwargs["tools"] if "cache_control" in t) == 1

    bedrock = BedrockProvider(
        region="us-east-1", default_model="global.anthropic.claude-sonnet-4-20250514-v1:0",
        client=SimpleNamespace(),
    )
    bed_kwargs = bedrock._build_kwargs(
        messages=messages, tools=tools, model="global.anthropic.claude-sonnet-4-20250514-v1:0",
        max_tokens=256, temperature=0.0, reasoning_effort=None, tool_choice=None, direct_tool_count=1,
    )
    marker = {"cachePoint": {"type": "default"}}
    assert bed_kwargs["messages"][-2]["content"][-1] == marker
    assert sum(1 for b in bed_kwargs["system"] if b == marker) == 2
    assert sum(1 for t in bed_kwargs["toolConfig"]["tools"] if t == marker) == 1
