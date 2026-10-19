"""Routing wrappers retain their parent providers' public argument contracts."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nanobot.providers.base import LLMProvider, LLMResponse, ProviderCallContext
from nanobot.providers.legacy_fallback_provider import LegacyFallbackProvider
from nanobot.providers.pooled_provider import PooledProvider


class RecordingLeaf(LLMProvider):
    def __init__(self):
        super().__init__(provider_name="fixture")
        self.calls = []
        self.response = LLMResponse(content="ok")

    def get_default_model(self):
        return "fixture-model"

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

    async def chat_with_context(self, *, provider_context, **kwargs):
        return await self.chat(provider_context=provider_context, **kwargs)

    async def chat_stream(self, **kwargs):
        for name in ("on_content_delta", "on_thinking_delta", "on_tool_call_delta"):
            callback = kwargs.get(name)
            if callback is not None:
                await callback({"id": "fixture"} if name == "on_tool_call_delta" else "delta")
        return await self.chat(**kwargs)

    async def chat_stream_with_context(self, *, provider_context, **kwargs):
        return await self.chat_stream(provider_context=provider_context, **kwargs)


def wrapper(kind):
    first, second = RecordingLeaf(), RecordingLeaf()
    if kind == "legacy":
        provider = LegacyFallbackProvider(
            first,
            [SimpleNamespace(model="fallback-model", context_window_tokens=1000)],
            lambda _: second,
        )
    else:
        provider = PooledProvider([first, second], ["fixture-a", "fixture-b"])
    return provider, first, second


@pytest.mark.parametrize("kind", ["legacy", "pool"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("contextual", [False, True])
async def test_messages_positional_and_keyword_preserve_leaf_kwargs(kind, stream, contextual):
    provider, leaf, unused = wrapper(kind)
    messages = [{"role": "user", "content": "hello"}]
    context = ProviderCallContext(session_id="fixture-session")
    kwargs = {"custom_option": "fixture"}
    if contextual:
        kwargs["provider_context"] = context
    call = provider.chat_stream if stream else provider.chat
    assert await call(messages, **kwargs) is leaf.response
    assert await call(messages=messages, **kwargs) is leaf.response
    assert leaf.calls[0] == leaf.calls[1]
    assert set(leaf.calls[0]) == {"messages", *kwargs, *({"model"} if kind == "legacy" else set())}
    assert not unused.calls


@pytest.mark.parametrize("kind", ["legacy", "pool"])
@pytest.mark.parametrize("stream", [False, True])
async def test_keyword_context_entry_points_preserve_shape(kind, stream):
    provider, leaf, unused = wrapper(kind)
    call = provider.chat_stream_with_context if stream else provider.chat_with_context
    context = ProviderCallContext(session_id="fixture-session")
    assert await call(messages=[], provider_context=context, custom_option="fixture") is leaf.response
    assert set(leaf.calls[0]) == {
        "messages", "provider_context", "custom_option", *({"model"} if kind == "legacy" else set())
    }
    assert leaf.calls[0]["provider_context"].session_id == context.session_id
    assert not unused.calls


NORMAL_NAMES = ("tools", "model", "max_tokens", "temperature", "reasoning_effort", "tool_choice")
NORMAL_VALUES = ([{"type": "function"}], "fixture-model", 73, 0.2, "high", {"type": "auto"})
STREAM_NAMES = ("on_content_delta", "on_thinking_delta", "on_tool_call_delta")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("count", range(7))
async def test_pool_maps_only_supplied_positional_parameters(stream, count):
    provider, leaf, unused = wrapper("pool")
    context = ProviderCallContext(session_id="fixture-session")
    extra = {"provider_context": context, "custom_option": "fixture"}
    call = provider.chat_stream if stream else provider.chat
    assert await call([], *NORMAL_VALUES[:count], **extra) is leaf.response
    assert await call(messages=[], **dict(zip(NORMAL_NAMES[:count], NORMAL_VALUES[:count])), **extra) is leaf.response
    assert leaf.calls[0] == leaf.calls[1]
    assert set(leaf.calls[0]) == {"messages", *NORMAL_NAMES[:count], *extra}
    assert not unused.calls


@pytest.mark.parametrize("count", [1, 2, 3])
async def test_pool_positional_stream_callbacks_keep_delivery(count):
    provider, leaf, unused = wrapper("pool")
    callbacks = tuple(AsyncMock() for _ in STREAM_NAMES)
    values = (*NORMAL_VALUES, *callbacks[:count])
    names = (*NORMAL_NAMES, *STREAM_NAMES[:count])
    assert await provider.chat_stream([], *values, custom_option="fixture") is leaf.response
    assert await provider.chat_stream(messages=[], **dict(zip(names, values)), custom_option="fixture") is leaf.response
    for index, callback in enumerate(callbacks[:count]):
        expected = {"id": "fixture"} if index == 2 else "delta"
        assert callback.await_count == 2
        assert all(call.args == (expected,) for call in callback.await_args_list)
    for recorded in leaf.calls:
        assert set(recorded) == {"messages", *names, "custom_option"}
        assert {name: recorded[name] for name in NORMAL_NAMES} == dict(zip(NORMAL_NAMES, NORMAL_VALUES))
    assert not unused.calls


@pytest.mark.parametrize("kind", ["legacy", "pool"])
@pytest.mark.parametrize("stream", [False, True])
async def test_excess_positionals_fail_before_routing(kind, stream):
    provider, first, second = wrapper(kind)
    call = provider.chat_stream if stream else provider.chat
    excess = (None,) * (1 if kind == "legacy" else (10 if stream else 7))
    if kind == "pool":
        with patch.object(provider._pool, "request_order") as order, patch.object(provider._pool, "admit", new_callable=AsyncMock) as admit:
            with pytest.raises(TypeError):
                await call([], *excess)
            order.assert_not_called()
            admit.assert_not_awaited()
    else:
        with pytest.raises(TypeError):
            await call([], *excess)
        assert provider._cooldowns == [0.0, 0.0]
    assert not first.calls and not second.calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("name", ("messages", *NORMAL_NAMES, *STREAM_NAMES))
async def test_pool_duplicate_bindings_fail_before_routing(stream, name):
    if not stream and name in STREAM_NAMES:
        return
    provider, first, second = wrapper("pool")
    names = ("messages", *NORMAL_NAMES, *(STREAM_NAMES if stream else ()))
    args = ([], *((None,) * names.index(name)))
    call = provider.chat_stream if stream else provider.chat
    with patch.object(provider._pool, "request_order") as order, patch.object(provider._pool, "admit", new_callable=AsyncMock) as admit:
        with pytest.raises(TypeError):
            await call(*args, **{name: None})
        order.assert_not_called()
        admit.assert_not_awaited()
    assert not first.calls and not second.calls


@pytest.mark.parametrize("stream", [False, True])
async def test_legacy_duplicate_messages_fail_before_leaf_call(stream):
    provider, first, second = wrapper("legacy")
    call = provider.chat_stream if stream else provider.chat
    with pytest.raises(TypeError):
        await call([], messages=[])
    assert not first.calls and not second.calls
    assert provider._cooldowns == [0.0, 0.0]


@pytest.mark.parametrize("raises", [False, True])
async def test_base_delivery_tracker_preserves_variadic_captures_before_await(raises):
    from nanobot.providers.base import track_stream_delivery

    delivered = [False]
    seen = []
    callbacks = {}
    originals = []
    failure = RuntimeError("callback failed")
    for name in STREAM_NAMES:
        async def callback(*args, _name=name, **kwargs):
            assert delivered[0] is True
            seen.append((_name, args, kwargs))
            if raises:
                raise failure
        originals.append(callback)
        callbacks[name] = callback
    untouched = object()
    callbacks["custom_option"] = untouched
    track_stream_delivery(callbacks, delivered)
    assert delivered == [False]
    assert callbacks["custom_option"] is untouched
    for index, name in enumerate(STREAM_NAMES):
        tracked = callbacks[name]
        assert tracked.__kwdefaults__["_callback"] is originals[index]
        delivered[0] = False
        payload = object()
        if raises:
            with pytest.raises(RuntimeError) as caught:
                await tracked(payload, index, marker=payload)
            assert caught.value is failure
        else:
            await tracked(payload, index, marker=payload)
        assert delivered == [True]
        assert seen[-1] == (name, (payload, index), {"marker": payload})
    absent = {"on_content_delta": None}
    track_stream_delivery(absent, delivered)
    assert absent == {"on_content_delta": None}
