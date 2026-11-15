"""Recent images are a detached model view, not edits to conversation history."""

import asyncio
from base64 import b64decode
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.events import EventSink
from nanobot.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ProviderCallContext,
    ProviderConversationState,
    ToolCallRequest,
)
from nanobot.providers.fallback_provider import FallbackProvider
from nanobot.providers.image_history import (
    IMAGE_HISTORY_DEFERRED,
    IMAGE_HISTORY_OMISSION,
    recent_image_messages,
)
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.pooled_provider import PooledProvider
from nanobot.providers.provider_policy import ESTIMATED_COMPAT_POLICY
from nanobot.providers.request_budget import ProviderRequestLimits


def image(label):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + label},
            "_meta": {"path": "/private/" + label}}


def history():
    return [
        {"role": "user", "content": [image("old-a"), image("old-b"), {"type": "text", "text": "old"}]},
        {"role": "assistant", "content": "seen"},
        {"role": "user", "content": [image("new-a"), image("new-b"), {"type": "text", "text": "compare"}]},
        {"role": "user", "content": "What differs?"},
    ]


def images(messages):
    return [block for message in messages if isinstance(message.get("content"), list)
            for block in message["content"] if isinstance(block, dict) and block.get("type") == "image_url"]


def image_urls(messages):
    return [block["image_url"]["url"] for block in images(messages)]


async def saved_tool_image_batch(tmp_path, *, payloads=None):
    from nanobot.agent.tools.filesystem import ReadFileTool

    reader = ReadFileTool(workspace=tmp_path, restrict_to_workspace=True)
    if payloads is None:
        payloads = [b"\x89PNG\r\n\x1a\n" + bytes([index]) + b"candidate" for index in range(4)]
    calls = [
        ToolCallRequest(
            id=f"read-{index}",
            name="read_file",
            arguments={"path": f"candidate-{index}.png"},
        )
        for index in range(len(payloads))
    ]
    messages = [{
        "role": "assistant",
        "content": None,
        "tool_calls": [call.to_openai_tool_call() for call in calls],
    }]
    urls = []
    for index, (call, payload) in enumerate(zip(calls, payloads, strict=True)):
        (tmp_path / f"candidate-{index}.png").write_bytes(payload)
        result = await reader.execute(path=f"candidate-{index}.png")
        assert isinstance(result, list)
        urls.append(result[0]["image_url"]["url"])
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": result,
        })
    return reader, messages, calls, urls


def _capturing_local_provider(api_base):
    provider = OpenAICompatProvider(
        api_key="fixture",
        api_base=api_base,
        default_model="fixture-model",
        provider_name="vision-fixture",
        policy=ESTIMATED_COMPAT_POLICY,
    )
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return object()

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    provider._parse = lambda _response: LLMResponse(content="ok")
    return provider, requests


def test_window_is_message_scoped_detached_and_idempotent():
    original = history()
    before = deepcopy(original)
    prepared = recent_image_messages(original)
    assert images(prepared) == [image("new-a"), image("new-b")]
    assert prepared[0]["content"][:2] == [{"type": "text", "text": IMAGE_HISTORY_OMISSION}] * 2
    assert "/private/old" not in str(prepared)
    assert prepared[-1] == original[-1]
    assert recent_image_messages(prepared) == prepared
    prepared[2]["content"][0]["image_url"]["url"] = "mutated"
    assert original == before


@pytest.mark.parametrize("messages", [[], [{"role": "user", "content": "text"}], history()[2:]])
def test_under_window_content_unchanged(messages):
    assert recent_image_messages(messages) == messages


def test_tool_image_advances_window_without_changing_tool_structure_or_other_blocks():
    messages = history() + [
        {"role": "assistant", "tool_calls": [{"id": "call", "type": "function",
          "function": {"name": "capture", "arguments": "{}"}}], "content": None},
        {"role": "tool", "tool_call_id": "call", "name": "capture", "content": [
            {"type": "audio", "data": "unchanged"}, image("tool-a"), image("tool-b"), None]},
    ]
    prepared = recent_image_messages(messages)
    assert images(prepared) == [image("tool-a"), image("tool-b")]
    assert prepared[-2:] == messages[-2:]


def test_parallel_tool_image_results_keep_the_whole_batch():
    messages = history() + [
        {"role": "assistant", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "c", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        ], "content": None},
        {"role": "tool", "tool_call_id": "a", "name": "read_file", "content": [image("batch-a")]},
        {"role": "tool", "tool_call_id": "b", "name": "read_file", "content": [image("batch-b")]},
        {"role": "tool", "tool_call_id": "c", "name": "read_file", "content": [image("batch-c")]},
    ]

    prepared = recent_image_messages(messages)

    assert images(prepared) == [image("batch-a"), image("batch-b"), image("batch-c")]
    assert prepared[-3:] == messages[-3:]


def test_a_later_tool_batch_advances_the_whole_window():
    messages = history() + [
        {"role": "assistant", "tool_calls": [
            {"id": "old", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        ], "content": None},
        {"role": "tool", "tool_call_id": "old", "name": "read_file", "content": [image("old")]},
        {"role": "assistant", "tool_calls": [
            {"id": "new", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        ], "content": None},
        {"role": "tool", "tool_call_id": "new", "name": "read_file", "content": [image("new")]},
    ]

    prepared = recent_image_messages(messages)

    assert images(prepared) == [image("new")]
    assert prepared[-2:] == messages[-2:]


class CapturingProvider(LLMProvider):
    def __init__(self, outcomes=None, delta=False):
        super().__init__(provider_name="image-fixture")
        self.outcomes = list(outcomes or [LLMResponse(content="ok")])
        self.calls = []
        self.delta = delta

    def get_default_model(self):
        return "fixture"

    def supports_native_compaction(self, model=None):
        return True

    async def chat(self, messages, **kwargs):
        return await self.chat_with_context(messages, **kwargs)

    async def chat_with_context(self, messages, provider_context=None, **kwargs):
        body = {"messages": self._sanitize_empty_content(messages), "max_tokens": 256}
        measurement = self.validate_finalized_request(
            body, transport="chat_completions", model="fixture", provider_context=provider_context,
        )
        self.calls.append((deepcopy(messages), provider_context, body, measurement))
        result = self.outcomes.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def chat_stream_with_context(self, messages, on_content_delta=None, **kwargs):
        if self.delta and on_content_delta:
            await on_content_delta("visible")
        return await self.chat_with_context(messages, **kwargs)

    async def chat_stream(self, messages, **kwargs):
        return await self.chat_stream_with_context(messages, **kwargs)


def native_context():
    return ProviderCallContext(
        conversation_state=ProviderConversationState("fixture", "image-fixture", "fixture", 1,
            payload={"opaque": "old native image"}, pending_messages=history()),
        request_limits=ProviderRequestLimits(8192, 256), context_window_tokens=8192,
        session_id="test:images", events=EventSink(AsyncMock()),
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("pooled", [False, True])
async def test_entry_points_filter_before_final_transport_and_drop_stale_native_state(stream, pooled):
    leaf = CapturingProvider()
    provider = PooledProvider([leaf], ["fixture"]) if pooled else leaf
    messages, context = history(), native_context()
    before = deepcopy(messages)
    request = provider.chat_stream_with_retry if stream else provider.chat_with_retry
    result = await request(messages=messages, provider_context=context)
    assert result.content == "ok" and len(leaf.calls) == 1
    sent, received, body, measurement = leaf.calls[0]
    assert images(sent) == [image("new-a"), image("new-b")]
    assert "old-a" not in str(body) and "old-b" not in str(body)
    assert received.conversation_state is None
    assert received.request_limits == context.request_limits
    assert received.session_id == context.session_id
    assert received.events is context.events
    assert received.context_window_tokens == 8192
    assert measurement is not None and measurement.fits
    assert messages == before and context.conversation_state is not None


@pytest.mark.parametrize("stream", [False, True])
async def test_cancelled_request_preserves_history_and_context(stream):
    provider = CapturingProvider([asyncio.CancelledError()])
    messages, context = history(), native_context()
    before = deepcopy(messages)
    request = provider.chat_stream_with_retry if stream else provider.chat_with_retry
    with pytest.raises(asyncio.CancelledError):
        await request(messages=messages, provider_context=context)
    assert len(provider.calls) == 1 and messages == before
    assert context.conversation_state is not None


async def test_stream_failure_after_delivery_does_not_replay_images():
    provider = CapturingProvider([LLMResponse(content="503 transient", finish_reason="error")], delta=True)
    delivered = AsyncMock()
    messages = history()
    before = deepcopy(messages)
    response = await provider.chat_stream_with_retry(messages, on_content_delta=delivered)
    assert response.finish_reason == "error" and len(provider.calls) == 1
    delivered.assert_awaited_once_with("visible")
    assert messages == before


async def test_failed_image_fallback_never_restores_old_images_or_mutates_history():
    provider = CapturingProvider([LLMResponse(content="bad input", finish_reason="error"),
                                  LLMResponse(content="still bad", finish_reason="error")])
    messages = history()
    before = deepcopy(messages)
    response = await provider.chat_with_retry(messages)
    assert response.finish_reason == "error" and len(provider.calls) == 2
    assert images(provider.calls[0][0]) == [image("new-a"), image("new-b")]
    assert images(provider.calls[1][0]) == []
    assert "old-a" not in str(provider.calls) and "old-b" not in str(provider.calls)
    assert messages == before


async def test_local_provider_preview_dispatch_and_later_read_share_image_projection(tmp_path):
    reader, messages, calls, expected_urls = await saved_tool_image_batch(tmp_path)
    original = deepcopy(messages)
    local, local_requests = _capturing_local_provider("http://127.0.0.1:11434/v1")
    context = ProviderCallContext(
        request_limits=ProviderRequestLimits(8192, 128, safety_buffer_tokens=0),
    )
    generation = GenerationSettings(max_tokens=128, temperature=0.2, reasoning_effort="none")

    preview = local.preview_request_budget(
        messages=messages,
        tools=None,
        model="fixture-model",
        generation=generation,
        provider_context=context,
        stream=False,
    )

    assert preview is not None
    assert image_urls(list(preview.converted_messages)) == expected_urls[:1]
    assert IMAGE_HISTORY_DEFERRED in str(preview.converted_messages)
    assert len([message for message in preview.converted_messages if message.get("role") == "tool"]) == 4
    assert [message.get("tool_call_id") for message in preview.converted_messages
            if message.get("role") == "tool"] == [call.id for call in calls]

    response = await local.chat_with_retry(
        messages=messages,
        model="fixture-model",
        max_tokens=128,
        temperature=0.2,
        reasoning_effort="none",
        provider_context=context,
    )

    assert response.content == "ok"
    assert image_urls(local_requests[0]["messages"]) == expected_urls[:1]
    assert tuple(local_requests[0]["messages"]) == preview.converted_messages

    next_call = ToolCallRequest(
        id="read-next",
        name="read_file",
        arguments={"path": "candidate-2.png"},
    )
    reread_result = await reader.execute(path="candidate-2.png")
    later_messages = [
        *messages,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [next_call.to_openai_tool_call()],
        },
        {
            "role": "tool",
            "tool_call_id": next_call.id,
            "name": next_call.name,
            "content": reread_result,
        },
    ]
    await local.chat_with_retry(
        messages=later_messages,
        model="fixture-model",
        max_tokens=128,
        temperature=0.2,
        reasoning_effort="none",
        provider_context=context,
    )
    assert image_urls(local_requests[1]["messages"]) == [expected_urls[2]]

    user_uploads = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": url}}
            for url in expected_urls
        ],
    }]
    await local.chat_with_retry(
        messages=user_uploads,
        model="fixture-model",
        max_tokens=128,
        temperature=0.2,
        reasoning_effort="none",
        provider_context=context,
    )
    assert image_urls(local_requests[2]["messages"]) == expected_urls

    cloud, cloud_requests = _capturing_local_provider("https://images.example/v1")
    await cloud.chat_with_retry(
        messages=messages,
        model="fixture-model",
        max_tokens=128,
        temperature=0.2,
        reasoning_effort="none",
        provider_context=context,
    )
    assert image_urls(cloud_requests[0]["messages"]) == expected_urls
    assert messages == original


async def test_local_provider_can_review_three_small_images_together(tmp_path):
    png = b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a/wEAAAAASUVORK5CYII="
    )
    _reader, messages, _calls, expected_urls = await saved_tool_image_batch(
        tmp_path, payloads=[png, png, png],
    )
    local, requests = _capturing_local_provider("http://127.0.0.1:11434/v1")
    response = await local.chat_with_retry(messages=messages, model="fixture-model", max_tokens=128)

    assert response.content == "ok"
    assert image_urls(requests[0]["messages"]) == expected_urls
    assert len(expected_urls) == 3


async def test_local_fallback_leaf_applies_tool_image_cap(tmp_path):
    _reader, messages, _calls, expected_urls = await saved_tool_image_batch(tmp_path)
    primary = CapturingProvider([
        LLMResponse(
            content="temporary failure",
            finish_reason="error",
            error_status_code=503,
        ),
    ])
    local, local_requests = _capturing_local_provider("http://127.0.0.1:11434/v1")
    preset = SimpleNamespace(
        model="local-fallback-model",
        max_tokens=128,
        temperature=0.2,
        reasoning_effort="none",
        context_window_tokens=8192,
        provider="vision-fixture",
    )
    provider = FallbackProvider(primary, [preset], lambda _preset: local)

    response = await provider.chat(
        messages=messages,
        model="external-model",
        max_tokens=128,
        temperature=0.2,
        reasoning_effort="none",
        provider_context=ProviderCallContext(
            request_limits=ProviderRequestLimits(8192, 128, safety_buffer_tokens=0),
        ),
    )

    assert response.content == "ok"
    assert len(primary.calls) == 1
    assert image_urls(local_requests[0]["messages"]) == expected_urls[:1]
