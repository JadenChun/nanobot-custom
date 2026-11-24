"""Exercise the bridge through its HTTP interface."""

import asyncio
import json
from urllib.parse import urlparse

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from kohaku.harness.anthropic_bridge import LoopbackAnthropicBridge
from kohaku.llm_usage.models import LLMCallRecord
from kohaku.providers.anthropic_provider import AnthropicProvider
from kohaku.providers.base import LLMProvider, LLMResponse, LLMUsage, ToolCallRequest
from kohaku.utils.llm_runtime import LLMRuntime

MODEL = "bound/model"
TOOL = {
    "name": "lookup",
    "description": "Look up an entry.",
    "input_schema": {"type": "object", "properties": {"key": {"type": "string"}}},
}


def runtime(provider):
    return LLMRuntime.capture(provider, MODEL, context_window_tokens=8192)


def payload(**fields):
    return {
        "model": "claude-haiku-compaction-alias",
        "messages": [{"role": "user", "content": "Look up an entry."}],
        "max_tokens": 128,
        **fields,
    }


def headers(bridge):
    return {"x-api-key": bridge.auth_token}


class ScriptedProvider(LLMProvider):
    def __init__(self, *responses):
        super().__init__(provider_name="scripted")
        self.responses = iter(responses)
        self.requests = []

    def get_default_model(self):
        return MODEL

    async def chat(self, **kwargs):
        self.requests.append(kwargs)
        return next(self.responses)


async def read_events(response):
    events = []
    for frame in (await response.text()).strip().split("\n\n"):
        event, data = frame.split("\n")
        event = event.removeprefix("event: ")
        data = json.loads(data.removeprefix("data: "))
        assert data["type"] == event
        events.append(data)
    return events


async def test_lifecycle_and_authentication():
    provider = ScriptedProvider(LLMResponse(content="One"), LLMResponse(content="Two"))
    bridge = LoopbackAnthropicBridge(runtime(provider))
    await bridge.start()
    try:
        async with LoopbackAnthropicBridge(runtime(provider)) as other, ClientSession() as http:
            url = urlparse(bridge.base_url)
            assert url.scheme == "http" and url.hostname == "127.0.0.1" and url.port
            assert urlparse(other.base_url).port != url.port
            assert bridge.auth_token != other.auth_token
            for endpoint in ("/v1/messages", "/v1/messages/count_tokens"):
                for credential in ({}, headers(other)):
                    async with http.post(
                        bridge.base_url + endpoint, headers=credential, json=payload(),
                    ) as result:
                        assert result.status == 401
                        assert (await result.json())["error"]["type"] == "authentication_error"
            for credential in (headers(bridge), {"Authorization": f"Bearer {bridge.auth_token}"}):
                async with http.post(
                    bridge.base_url + "/v1/messages", headers=credential, json=payload(),
                ) as result:
                    assert result.status == 200
    finally:
        await bridge.close()


async def test_non_streaming_tool_round_trip_and_bound_runtime():
    provider = ScriptedProvider(
        LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest("toolu_lookup", "lookup", {"key": "example"})],
            finish_reason="tool_calls",
        ),
        LLMResponse(content="Found it."),
    )
    records: list[LLMCallRecord] = []
    provider.set_llm_call_observer(records.append)
    initial = payload(
        system=[{"type": "text", "text": "Look up entries.", "cache_control": {"type": "ephemeral"}}],
        tools=[
            {"name": "DeferredToolPlaceholder", "input_schema": {"type": "object"}},
            {**TOOL, "cache_control": {"type": "ephemeral"}, "defer_loading": True},
        ],
        tool_choice={"type": "tool", "name": "lookup"},
        temperature=0.25,
        thinking={"type": "adaptive", "display": "updates"},
        output_config={"effort": "high"},
        context_management={"edits": []},
    )
    initial["messages"][0]["content"] = [
        {"type": "text", "text": "Look up an entry."},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "eA=="}},
    ]
    initial["messages"].append(
        {"role": "system", "content": [{"type": "text", "text": "Use the lookup tool."}]}
    )
    async with LoopbackAnthropicBridge(runtime(provider), usage_source="cron") as bridge:
        async with ClientSession() as http:
            url = bridge.base_url + "/v1/messages?beta=true"
            credential = {**headers(bridge), "anthropic-beta": "context-management-2025-06-27"}
            async with http.post(url, headers=credential, json=initial) as result:
                assert result.status == 200
                message = await result.json()
            assert message["stop_reason"] == "tool_use"
            assert message["content"] == [
                {"type": "tool_use", "id": "toolu_lookup", "name": "lookup", "input": {"key": "example"}},
            ]
            followup = {
                **initial,
                "model": "another-model-alias",
                "messages": [
                    *initial["messages"],
                    {"role": "assistant", "content": [
                        {"type": "thinking", "thinking": "Look up this key.", "signature": "signed"},
                        *message["content"],
                    ]},
                    {"role": "user", "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_lookup",
                            "content": [{"type": "text", "text": "The entry."}],
                        },
                    ]},
                ],
            }
            async with http.post(url, headers=credential, json=followup) as result:
                assert result.status == 200
                assert (await result.json())["content"] == [{"type": "text", "text": "Found it."}]

    first, second = provider.requests
    assert first["model"] == second["model"] == MODEL
    assert first["max_tokens"] == 128
    assert first["temperature"] == 0.25
    assert first["reasoning_effort"] == "high"
    assert first["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert first["tools"] == [{"type": "function", "function": {
        "name": "lookup", "description": TOOL["description"], "parameters": TOOL["input_schema"],
    }}]
    assert first["messages"][0] == {"role": "system", "content": "Look up entries.\nUse the lookup tool."}
    assert first["messages"][1]["content"][-1] == {
        "type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="},
    }
    assistant, tool_result = second["messages"][-2:]
    assert assistant["reasoning_content"] == "Look up this key."
    assert assistant["tool_calls"][0]["id"] == "toolu_lookup"
    assert tool_result == {"role": "tool", "tool_call_id": "toolu_lookup", "content": "The entry."}
    assert [record.source for record in records] == ["cron", "cron"]
    assert [record.model for record in records] == [MODEL, MODEL]


class StreamingProvider(ScriptedProvider):
    async def chat_stream(self, *, on_content_delta, on_thinking_delta, on_tool_call_delta, **kwargs):
        self.requests.append(kwargs)
        await on_thinking_delta("Pick ")
        await on_thinking_delta("a key.")
        await on_content_delta("Checking.")
        for piece in ('{"key":', '"example"}'):
            await on_tool_call_delta({
                "index": 0, "call_id": "toolu_lookup", "name": "lookup", "arguments_delta": piece,
            })
        return LLMResponse(
            content="Checking.",
            reasoning_content="Pick a key.",
            tool_calls=[ToolCallRequest("toolu_lookup", "lookup", {"key": "example"})],
            finish_reason="tool_calls",
            usage=LLMUsage.reported(input_tokens=19, output_tokens=7, cache_read_tokens=8),
        )


async def test_stream_events_and_usage():
    provider = StreamingProvider()
    async with LoopbackAnthropicBridge(runtime(provider)) as bridge, ClientSession() as http:
        async with http.post(
            bridge.base_url + "/v1/messages?beta=true", headers=headers(bridge),
            json=payload(stream=True, tools=[TOOL]),
        ) as result:
            assert result.status == 200
            assert result.content_type == "text/event-stream"
            events = await read_events(result)

    assert events[0]["type"] == "message_start"
    assert events[0]["message"]["content"] == []
    assert events[-2] == {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"input_tokens": 11, "output_tokens": 7, "cache_read_input_tokens": 8},
    }
    assert events[-1] == {"type": "message_stop"}
    events = events[1:-2]
    for index, start, deltas in (
        (0, {"type": "thinking", "thinking": "", "signature": ""}, [
            {"type": "thinking_delta", "thinking": text} for text in ("Pick ", "a key.")
        ]),
        (1, {"type": "text", "text": ""}, [{"type": "text_delta", "text": "Checking."}]),
        (2, {"type": "tool_use", "id": "toolu_lookup", "name": "lookup", "input": {}}, [
            {"type": "input_json_delta", "partial_json": piece} for piece in ('{"key":', '"example"}')
        ]),
    ):
        assert events[:len(deltas) + 2] == [
            {"type": "content_block_start", "index": index, "content_block": start},
            *({"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas),
            {"type": "content_block_stop", "index": index},
        ]
        events = events[len(deltas) + 2:]
    assert not events


async def test_count_tokens_estimates_without_a_provider_call():
    provider = ScriptedProvider()
    async with LoopbackAnthropicBridge(runtime(provider)) as bridge, ClientSession() as http:
        counts = []
        for text in ("A short sentence.", "A much longer sentence with more words. " * 20):
            async with http.post(
                bridge.base_url + "/v1/messages/count_tokens",
                headers=headers(bridge),
                json={"model": "alias", "messages": [{"role": "user", "content": text}]},
            ) as result:
                assert result.status == 200
                counts.append((await result.json())["input_tokens"])
        assert all(isinstance(count, int) for count in counts)
        assert counts[1] > counts[0] > 0
    assert not provider.requests


async def test_upstream_errors_use_anthropic_format():
    error = LLMResponse(
        content="Provider private detail",
        finish_reason="error",
        error_status_code=429,
        error_should_retry=False,
    )
    provider = ScriptedProvider(error, error)
    async with LoopbackAnthropicBridge(runtime(provider)) as bridge, ClientSession() as http:
        url = bridge.base_url + "/v1/messages"
        async with http.post(url, headers=headers(bridge), json=payload()) as result:
            assert result.status == 429
            body = await result.json()
            assert body["type"] == "error"
            assert body["error"]["type"] == "rate_limit_error"
            assert "Provider private detail" not in body["error"]["message"]
        async with http.post(url, headers=headers(bridge), json=payload(stream=True)) as result:
            events = await read_events(result)
        assert events[-1] == body
        assert not any(event["type"] == "message_stop" for event in events)


class WaitingProvider(ScriptedProvider):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def chat_stream(self, **kwargs):
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


async def test_disconnect_cancels_upstream_before_first_delta():
    provider = WaitingProvider()
    async with LoopbackAnthropicBridge(runtime(provider)) as bridge, ClientSession() as http:
        async with http.post(
            bridge.base_url + "/v1/messages", headers=headers(bridge), json=payload(stream=True),
        ) as result:
            await asyncio.wait_for(provider.started.wait(), 2)
            result.close()
            await asyncio.wait_for(provider.cancelled.wait(), 2)


class StoppingProvider(ScriptedProvider):
    async def chat_stream(self, *, on_content_delta, **kwargs):
        await on_content_delta("Answer <E")
        await on_content_delta("ND> trailing")
        return LLMResponse(
            content="Answer <END> trailing",
            usage=LLMUsage.reported(input_tokens=19, output_tokens=7),
        )


async def test_stop_sequence_split_across_text_deltas():
    provider = StoppingProvider(LLMResponse(content="Answer <END> trailing"))
    async with LoopbackAnthropicBridge(runtime(provider)) as bridge, ClientSession() as http:
        url = bridge.base_url + "/v1/messages"
        async with http.post(
            url, headers=headers(bridge), json=payload(stream=True, stop_sequences=["<END>"]),
        ) as result:
            events = await read_events(result)
        assert "".join(
            event["delta"]["text"]
            for event in events
            if event["type"] == "content_block_delta" and event["delta"]["type"] == "text_delta"
        ) == "Answer "
        assert events[-2]["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "<END>"}
        assert events[-2]["usage"] == {"input_tokens": 19, "output_tokens": 7}
        async with http.post(
            url, headers=headers(bridge), json=payload(stop_sequences=["<END>"]),
        ) as result:
            message = await result.json()
        assert message["content"] == [{"type": "text", "text": "Answer "}]
        assert message["stop_reason"] == "stop_sequence"
        assert message["stop_sequence"] == "<END>"


async def test_native_anthropic_signed_thinking_and_cache():
    requests = []

    async def upstream(request):
        body = await request.json()
        requests.append(body)
        return web.json_response({
            "id": "msg_native",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": [{"type": "text", "text": "Found it."}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

    app = web.Application()
    app.router.add_post("/v1/messages", upstream)
    cache = {"type": "ephemeral", "ttl": "1h"}
    thinking = {"type": "thinking", "thinking": "Look up this key.", "signature": "signed"}
    async with TestServer(app) as upstream_server:
        provider = AnthropicProvider(
            api_base=str(upstream_server.make_url("/")), api_key="scripted-upstream",
        )
        native_runtime = LLMRuntime.capture(provider, "claude-sonnet-4-6", context_window_tokens=8192)
        async with LoopbackAnthropicBridge(native_runtime) as bridge, ClientSession() as http:
            async with http.post(
                bridge.base_url + "/v1/messages",
                headers=headers(bridge),
                json=payload(
                    system=[
                        {"type": "text", "text": "Retain this prompt.", "cache_control": cache},
                        {"type": "text", "text": "Look up entries."},
                    ],
                    thinking={"type": "adaptive", "display": "updates"},
                    output_config={"effort": "high"},
                    tools=[TOOL],
                    messages=[
                        {"role": "user", "content": "Look up an entry."},
                        {"role": "assistant", "content": [
                            thinking,
                            {"type": "tool_use", "id": "toolu_lookup", "name": "lookup", "input": {}},
                        ]},
                        {"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_lookup", "content": "The entry."},
                        ]},
                    ],
                ),
            ) as result:
                assert result.status == 200
                assert (await result.json())["content"] == [{"type": "text", "text": "Found it."}]
    assert len(requests) == 1
    sent = requests[0]
    assert sent["model"] == "claude-sonnet-4-6"
    assert sent["system"][0]["cache_control"] == cache
    assert sent["messages"][1]["content"][0] == thinking
    assert sent["thinking"]["type"] == "adaptive"
