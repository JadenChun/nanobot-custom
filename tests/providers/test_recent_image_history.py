"""Recent images are a detached model view, not edits to conversation history."""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from nanobot.events import EventSink
from nanobot.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderCallContext,
    ProviderConversationState,
)
from nanobot.providers.image_history import IMAGE_HISTORY_OMISSION, recent_image_messages
from nanobot.providers.pooled_provider import PooledProvider
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
