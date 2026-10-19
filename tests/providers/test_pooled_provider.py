import asyncio
from unittest.mock import AsyncMock

import pytest

from nanobot.providers.base import (
    LLMProvider,
    LLMResponse,
    ProviderCallContext,
    ProviderConversationState,
)
from nanobot.providers.pooled_provider import PooledProvider


class Leaf(LLMProvider):
    def __init__(self, key, responses):
        super().__init__(api_key=key, provider_name='fixture')
        self.responses = list(responses)
        self.calls = []
        self.delta = None

    def get_default_model(self):
        return 'fixture-model'

    def supports_native_compaction(self, model=None):
        return True

    def can_resume_conversation_state(self, state, model=None):
        return state.provider == 'fixture' and state.model == (model or self.get_default_model())

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    async def chat_with_context(self, *, provider_context, **kwargs):
        return await self.chat(provider_context=provider_context, **kwargs)

    async def chat_stream(self, **kwargs):
        if self.delta:
            await kwargs[self.delta]('emitted')
        return await self.chat(**kwargs)

    async def chat_stream_with_context(self, *, provider_context, **kwargs):
        return await self.chat_stream(provider_context=provider_context, **kwargs)


def test_generation_constructor_and_assignment_preserve_exact_leaf_identity():
    from nanobot.providers.base import GenerationSettings

    leaves = [Leaf(key, []) for key in ('a', 'b', 'c')]
    original = GenerationSettings(max_tokens=71, temperature=0.25, reasoning_effort='high')
    leaves[0].generation = original
    pool = PooledProvider(leaves, ['a', 'b', 'c'])

    assert 'generation' not in LLMProvider.__dict__
    assert pool.generation is original
    assert all(leaf.generation is original for leaf in leaves)
    replacement = GenerationSettings(max_tokens=83, temperature=0.5)
    pool.generation = replacement
    assert pool.generation is replacement
    assert all(leaf.generation is replacement for leaf in leaves)


def test_generation_identity_propagates_through_nested_legacy_pool():
    from types import SimpleNamespace

    from nanobot.providers.base import GenerationSettings
    from nanobot.providers.legacy_fallback_provider import LegacyFallbackProvider

    first, second, third = (Leaf(key, []) for key in ('a', 'b', 'c'))
    original = GenerationSettings(max_tokens=71)
    first.generation = original
    legacy = LegacyFallbackProvider(
        first, [SimpleNamespace(model='fallback')], lambda _: second,
    )
    pool = PooledProvider([legacy, third], ['a', 'c'])

    assert pool.generation is legacy.generation is original
    assert all(leaf.generation is original for leaf in (first, second, third))
    replacement = GenerationSettings(max_tokens=83)
    pool.generation = replacement
    assert pool.generation is legacy.generation is replacement
    assert all(leaf.generation is replacement for leaf in (first, second, third))


def error(status=429, text='quota exceeded', kind='rate_limit'):
    return LLMResponse(content=text, finish_reason='error', error_status_code=status, error_kind=kind)


async def test_rotation_uses_immutable_credentials_and_preserves_terminal_response():
    last = error(503, 'unavailable', 'server_error')
    first, second = Leaf('a', [error()]), Leaf('b', [last])
    pool = PooledProvider([first, second], ['a', 'b'])
    response = await pool.chat_with_retry(messages=[], model='fixture-model')
    assert response is last
    assert len(first.calls) == len(second.calls) == 1
    assert first.api_key == 'a' and second.api_key == 'b'


@pytest.mark.parametrize('response', [error(401, 'unauthorized', 'auth'), error(408, 'timeout', 'timeout')])
async def test_nonquota_errors_do_not_rotate(response):
    first, second = Leaf('a', [response]), Leaf('b', [])
    assert await PooledProvider([first, second], ['a', 'b']).chat(messages=[]) is response
    assert second.calls == []


@pytest.mark.parametrize('delta', ['on_content_delta', 'on_thinking_delta', 'on_tool_call_delta'])
async def test_delivered_delta_prevents_timeout_replay_and_rotation(delta):
    terminal = error(408, 'timed out', 'timeout')
    first, second = Leaf('a', [terminal]), Leaf('b', [])
    first.delta = delta
    callback = AsyncMock()
    result = await PooledProvider([first, second], ['a', 'b']).chat_stream_with_retry(messages=[], **{delta: callback})
    assert result is terminal
    assert len(first.calls) == 1 and second.calls == []
    callback.assert_awaited_once()


async def test_state_is_bound_to_actual_credential_and_context_identity_is_preserved():
    state = ProviderConversationState(kind='fixture', provider='fixture', model='fixture-model', version=1,
                                      payload={'opaque': 'private'})
    first, second = Leaf('a', [LLMResponse(content='first', provider_state=state), error()]), Leaf('b', [LLMResponse(content='second')])
    pool = PooledProvider([first, second], ['a', 'b'])
    initial = await pool.chat(messages=[])
    context = ProviderCallContext(conversation_state=initial.provider_state, session_id='session', context_window_tokens=12345)
    await pool.chat_with_context(messages=[], provider_context=context)
    own = first.calls[-1]['provider_context']
    foreign = second.calls[-1]['provider_context']
    assert own.conversation_state.payload == {'opaque': 'private'}
    assert foreign.conversation_state is None
    assert own.session_id == foreign.session_id == 'session'
    assert foreign.events is context.events
    assert foreign.context_window_tokens == 12345
    assert state.payload == {'opaque': 'private'}
    assert pool.can_resume_conversation_state(initial.provider_state)


async def test_cancelled_leaf_does_not_attempt_another_key():
    first, second = Leaf('a', [asyncio.CancelledError()]), Leaf('b', [])
    with pytest.raises(asyncio.CancelledError):
        await PooledProvider([first, second], ['a', 'b']).chat_with_retry(messages=[])
    assert second.calls == []


async def test_observer_counts_actual_leaf_attempts_once_and_success_promotes_key():
    first = Leaf('a', [error()])
    second = Leaf('b', [LLMResponse(content='ok'), LLMResponse(content='again')])
    pool = PooledProvider([first, second], ['a', 'b'])
    records = []
    pool.set_llm_call_observer(records.append)
    assert (await pool.chat_with_retry(messages=[])).content == 'ok'
    assert (await pool.chat_with_retry(messages=[])).content == 'again'
    assert len(records) == 3
    assert len(first.calls) == 1 and len(second.calls) == 2


async def test_exhaustion_notifies_once():
    pool = PooledProvider([Leaf('a', [error()]), Leaf('b', [error()])], ['a', 'b'])
    exhausted, status = AsyncMock(), AsyncMock()
    await pool.chat_with_retry(messages=[], on_retry_exhausted=exhausted, on_retry_status=status)
    exhausted.assert_awaited_once()
    status.assert_awaited_once()
    assert status.call_args.args[0].state == 'exhausted'


async def test_compaction_state_is_owned_and_detached():
    state = ProviderConversationState(kind='fixture', provider='fixture', model='fixture-model', version=1,
                                      payload={'nested': {'value': 1}})
    leaf = Leaf('a', [LLMResponse(content=None, provider_compaction_state=state), LLMResponse(content='ok')])
    pool = PooledProvider([leaf], ['a'])
    result = await pool.chat(messages=[])
    assert pool.can_resume_conversation_state(result.provider_compaction_state)
    await pool.chat_with_context(messages=[], provider_context=ProviderCallContext(
        conversation_state=result.provider_compaction_state))
    received = leaf.calls[-1]['provider_context'].conversation_state
    received.payload['nested']['value'] = 2
    assert state.payload['nested']['value'] == 1
    assert result.provider_compaction_state.payload['nested']['value'] == 1


@pytest.mark.parametrize('response,seconds', [
    (error(429, 'quota exceeded; retry after 10'), 60),
    (error(429, 'quota exceeded'), 1800),
    (error(503, 'unavailable'), 15),
    (error(408, 'timed out', 'timeout'), None),
])
def test_legacy_cooldown_classification(response, seconds):
    from nanobot.providers.pooled_provider import _rotation_cooldown
    assert _rotation_cooldown(response) == seconds


async def test_exhaustion_after_ordinary_retry_notifies_once():
    first = Leaf('a', [error(500, 'server failure', 'server_error'), error()])
    second = Leaf('b', [error()])
    pool = PooledProvider([first, second], ['a', 'b'])
    pool._sleep_with_heartbeat = AsyncMock()
    exhausted, status = AsyncMock(), AsyncMock()
    records = []
    pool.set_llm_call_observer(records.append)
    await pool.chat_with_retry(messages=[], on_retry_exhausted=exhausted, on_retry_status=status)
    exhausted.assert_awaited_once()
    assert [call.args[0].state for call in status.call_args_list].count('exhausted') == 1
    assert 'cleared' not in [call.args[0].state for call in status.call_args_list]
    assert len(records) == 3

    assert status.call_args.args[0].attempt == 3


async def test_reconstructed_pool_rebuilds_from_portable_history():
    state = ProviderConversationState(kind='fixture', provider='fixture', model='fixture-model', version=1,
                                      payload={'opaque': 'old-handle'})
    old = PooledProvider([Leaf('a', [LLMResponse(content='old', provider_state=state)])], ['a'])
    response = await old.chat(messages=[])
    leaf = Leaf('a', [LLMResponse(content='new')])
    new = PooledProvider([leaf], ['a'])
    messages = [{'role': 'user', 'content': 'portable history'}]
    assert not new.can_resume_conversation_state(response.provider_state)
    await new.chat_with_context(messages=messages, provider_context=ProviderCallContext(
        conversation_state=response.provider_state))
    assert leaf.calls[0]['provider_context'].conversation_state is None
    assert leaf.calls[0]['messages'] == messages


@pytest.mark.parametrize('delta', ['on_content_delta', 'on_thinking_delta', 'on_tool_call_delta'])
@pytest.mark.parametrize('kind', ['timeout', 'rate_limit'])
@pytest.mark.parametrize('pooled', [False, True])
@pytest.mark.parametrize('entry', ['retry', 'persistent', 'direct', 'context'])
async def test_nested_fallback_never_replays_delivered_delta(delta, kind, pooled, entry):
    from types import SimpleNamespace

    from nanobot.providers.fallback_provider import FallbackProvider
    terminal = error(408 if kind == 'timeout' else 429, kind, kind)
    leaf = Leaf('a', [terminal])
    leaf.delta = delta
    primary = PooledProvider([leaf], ['a']) if pooled else leaf
    fallback = Leaf('b', [])
    wrapper = FallbackProvider(primary, [SimpleNamespace(model='fallback')], lambda _: fallback)
    callback, recover = AsyncMock(), AsyncMock()
    kwargs = dict(messages=[], on_stream_recover=recover, **{delta: callback})
    if entry == 'direct':
        result = await wrapper.chat_stream(**kwargs)
    elif entry == 'context':
        result = await wrapper.chat_stream_with_context(provider_context=ProviderCallContext(), **kwargs)
    else:
        result = await wrapper.chat_stream_with_retry(
            retry_mode='persistent' if entry == 'persistent' else 'standard', **kwargs)
    assert result is terminal
    assert len(leaf.calls) == 1 and not fallback.calls
    callback.assert_awaited_once()
    recover.assert_not_awaited()


@pytest.mark.parametrize('delta', ['on_content_delta', 'on_thinking_delta', 'on_tool_call_delta'])
async def test_compat_responses_error_after_delivery_does_not_switch_transport(delta):
    from types import SimpleNamespace
    from unittest.mock import patch

    from nanobot.providers.openai_compat_provider import OpenAICompatProvider

    provider = OpenAICompatProvider(api_key='fixture', default_model='fixture')
    chat_create = AsyncMock()
    provider._ensure_client = AsyncMock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create))))
    provider._should_use_responses_api = lambda *args: True
    provider._build_responses_body = lambda *args: {'model': 'fixture'}
    provider._create_response_with_compaction_fallback = AsyncMock(return_value=object())
    provider._should_fallback_from_responses_error = lambda exc: True
    callback = AsyncMock()

    async def consume(stream, on_content_delta, **kwargs):
        selected = {'on_content_delta': on_content_delta,
                    'on_thinking_delta': kwargs['on_reasoning_delta'],
                    'on_tool_call_delta': kwargs['on_tool_call_delta']}[delta]
        await selected('delivered')
        raise ValueError('compatible transport error')

    with patch('nanobot.providers.openai_compat_provider.consume_sdk_stream', consume):
        result = await provider.chat_stream(messages=[], **{delta: callback})
    assert result.finish_reason == 'error'
    callback.assert_awaited_once()
    chat_create.assert_not_awaited()


@pytest.mark.parametrize('succeed', [False, True])
async def test_fallback_defers_pool_exhaustion_and_counts_leaf_attempts(succeed):
    from types import SimpleNamespace

    from nanobot.providers.fallback_provider import FallbackProvider

    primary = PooledProvider([Leaf('a', [error()]), Leaf('b', [error()])], ['a', 'b'])
    terminal = LLMResponse(content='ok') if succeed else error()
    fallback = PooledProvider([Leaf('c', [terminal])], ['c'])
    preset = SimpleNamespace(model='fallback', max_tokens=100, temperature=0.5, reasoning_effort=None)
    wrapper = FallbackProvider(primary, [preset], lambda _: fallback)
    exhausted, status = AsyncMock(), AsyncMock()
    records = []
    wrapper.set_llm_call_observer(records.append)
    result = await wrapper.chat_with_retry(messages=[], on_retry_exhausted=exhausted,
                                           on_retry_status=status)
    assert result.content == terminal.content
    assert result.error_status_code == terminal.error_status_code
    assert len(records) == 3
    if succeed:
        exhausted.assert_not_awaited()
        assert all(call.args[0].state != 'exhausted' for call in status.call_args_list)
    else:
        exhausted.assert_awaited_once()
        assert sum(call.args[0].state == 'exhausted' for call in status.call_args_list) == 1


async def test_concurrent_requests_keep_context_and_credentials_local():
    entered, release = asyncio.Event(), asyncio.Event()

    class ConcurrentLeaf(Leaf):
        async def chat(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 2:
                entered.set()
            await release.wait()
            session = kwargs['provider_context'].session_id
            return LLMResponse(content=session)

    leaf = ConcurrentLeaf('a', [])
    other = Leaf('b', [])
    pool = PooledProvider([leaf, other], ['a', 'b'])
    tasks = [asyncio.create_task(pool.chat_with_context(
        messages=[{'role': 'user', 'content': session}],
        provider_context=ProviderCallContext(session_id=session))) for session in ('one', 'two')]
    try:
        await asyncio.wait_for(entered.wait(), 1)
    finally:
        release.set()
    results = await asyncio.gather(*tasks)
    assert [result.content for result in results] == ['one', 'two']
    assert leaf.api_key == 'a' and other.api_key == 'b'
    assert not other.calls
    assert [call['messages'][0]['content'] for call in leaf.calls] == ['one', 'two']


@pytest.mark.parametrize('raises', [False, True])
async def test_pool_tracks_variadic_callbacks_before_await_and_captures_each(raises):
    terminal = error()
    names = ('on_content_delta', 'on_thinking_delta', 'on_tool_call_delta')
    calls = []

    async def content(*args, **kwargs):
        calls.append(('content', args, kwargs))
        if raises:
            raise ValueError('callback failed after delivery')

    async def thinking(*args, **kwargs):
        calls.append(('thinking', args, kwargs))
        if raises:
            raise ValueError('callback failed after delivery')

    async def tool(*args, **kwargs):
        calls.append(('tool', args, kwargs))
        if raises:
            raise ValueError('callback failed after delivery')

    class VariadicLeaf(Leaf):
        async def chat_stream(self, **kwargs):
            for name in names:
                try:
                    await kwargs[name]('first', {'second': name}, named=name)
                except ValueError:
                    pass
            return await self.chat(**kwargs)

    first, spare = VariadicLeaf('a', [terminal]), Leaf('b', [])
    pool = PooledProvider([first, spare], ['a', 'b'])
    result = await pool.chat_stream(messages=[], **dict(zip(names, (content, thinking, tool))))
    assert result is terminal
    assert spare.calls == []
    assert calls == [(label, ('first', {'second': name}), {'named': name})
                     for label, name in zip(('content', 'thinking', 'tool'), names)]


async def test_pool_exhaustion_keeps_last_slot_tagging_and_stop_cell():
    state = ProviderConversationState(kind='fixture', provider='fixture', model='fixture-model',
                                      version=1, payload={'opaque': 'last'})
    terminal = error()
    terminal.provider_state = state
    first, last = Leaf('a', [error()]), Leaf('b', [terminal])
    pool = PooledProvider([first, last], ['a', 'b'])
    stopped, attempts = [False], [0]
    result = await pool._attempt(dict(messages=[]), stream=False, observed=False,
                                 stopped=stopped, attempts=attempts)
    assert stopped == ['exhausted']
    assert attempts == [2]
    assert result.content == terminal.content
    assert result.provider_state is not state
    assert state.payload == {'opaque': 'last'}
    assert pool._leaf_state(0, result.provider_state, 'fixture-model') is None
    assert pool._leaf_state(1, result.provider_state, 'fixture-model').payload == state.payload
