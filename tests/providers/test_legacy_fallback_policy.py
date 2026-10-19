from types import SimpleNamespace

import pytest

from nanobot.config.schema import Config
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.providers.legacy_fallback_provider import LegacyFallbackProvider


class Leaf(LLMProvider):
    def __init__(self, responses):
        super().__init__(provider_name="fixture")
        self.responses = list(responses)
        self.calls = []

    def get_default_model(self):
        return 'primary'

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)

    async def chat_with_retry(self, **kwargs):
        return await self.chat(**kwargs)


def quota():
    return LLMResponse(content='quota exceeded', finish_reason='error', error_status_code=429)


def wrapper(*leaves):
    presets = [SimpleNamespace(model=f'fallback-{i}', context_window_tokens=1000) for i in range(1, len(leaves))]
    return LegacyFallbackProvider(leaves[0], presets, lambda p: leaves[presets.index(p)+1])


def test_generation_constructor_and_assignment_preserve_exact_candidate_identity():
    from nanobot.providers.base import GenerationSettings

    leaves = [Leaf([]) for _ in range(3)]
    original = GenerationSettings(max_tokens=71, temperature=0.25, reasoning_effort='high')
    leaves[0].generation = original
    provider = wrapper(*leaves)

    assert 'generation' not in LLMProvider.__dict__
    assert provider.generation is original
    assert all(leaf.generation is original for leaf in leaves)
    replacement = GenerationSettings(max_tokens=83, temperature=0.5)
    provider.generation = replacement
    assert provider.generation is replacement
    assert all(leaf.generation is replacement for leaf in leaves)


def test_generation_identity_propagates_through_nested_pool_legacy():
    from nanobot.providers.base import GenerationSettings
    from nanobot.providers.pooled_provider import PooledProvider

    first, second, third = (Leaf([]) for _ in range(3))
    original = GenerationSettings(max_tokens=71)
    first.generation = original
    pool = PooledProvider([first, second], ['a', 'b'])
    provider = wrapper(pool, third)

    assert provider.generation is pool.generation is original
    assert all(leaf.generation is original for leaf in (first, second, third))
    replacement = GenerationSettings(max_tokens=83)
    provider.generation = replacement
    assert provider.generation is pool.generation is replacement
    assert all(leaf.generation is replacement for leaf in (first, second, third))


async def test_cooling_primary_is_retried_after_ready_candidate_and_generation_override_is_preserved():
    primary = Leaf([quota(), LLMResponse(content='recovered')])
    second = Leaf([LLMResponse(content='ok'), quota()])
    provider = wrapper(primary, second)
    assert (await provider.chat_with_retry(messages=[], model='override', max_tokens=73)).content == 'ok'
    assert primary.calls[0]['model'] == 'override'
    assert second.calls[0]['model'] == 'fallback-1'
    assert second.calls[0]['max_tokens'] == 73
    assert (await provider.chat_with_retry(messages=[])).content == 'recovered'
    assert provider._cooldowns[0] == 0 and provider._cooldowns[1] > 0


@pytest.mark.parametrize('persistent', [False, True])
async def test_secondary_nonquota_stops_without_chain_replay(persistent):
    terminal = LLMResponse(content='server error', finish_reason='error', error_status_code=500)
    leaves = [Leaf([quota()]), Leaf([terminal]), Leaf([])]
    provider = wrapper(*leaves)
    result = await provider.chat_with_retry(messages=[], retry_mode='persistent' if persistent else 'standard')
    assert result is terminal
    assert [len(leaf.calls) for leaf in leaves] == [1, 1, 0]


async def test_direct_call_uses_primary_and_does_not_cool_or_failover():
    terminal = quota()
    first, second = Leaf([terminal]), Leaf([])
    provider = wrapper(first, second)
    assert await provider.chat(messages=[]) is terminal
    assert not second.calls and provider._cooldowns == [0, 0]


async def test_all_cooling_probes_earliest_expiry_and_caches_candidates():
    first, second = Leaf([]), Leaf([LLMResponse(content='ok')])
    provider = wrapper(first, second)
    # Finite future deadlines establish a deterministic earlier candidate.
    import time
    provider._cooldowns = [time.monotonic()+200, time.monotonic()+100]
    assert (await provider.chat_with_retry(messages=[])).content == 'ok'
    assert not first.calls
    assert provider._candidates[1] is second


async def test_generation_assignment_reaches_all_cached_leaves():
    from nanobot.providers.base import GenerationSettings
    first, second = Leaf([]), Leaf([])
    provider = wrapper(first, second)
    provider.generation = GenerationSettings(max_tokens=71)
    assert first.generation.max_tokens == second.generation.max_tokens == 71


async def test_fallback_owned_state_resumes_only_its_candidate():
    from nanobot.providers.base import ProviderCallContext, ProviderConversationState

    class StatefulLeaf(Leaf):
        def supports_native_compaction(self, model=None):
            return True

        def can_resume_conversation_state(self, state, model=None):
            return state.model == model

    state = ProviderConversationState(kind='fixture', provider='fixture', model='fallback-1', version=1,
                                      payload={'opaque': 'handle'})
    first = StatefulLeaf([quota(), LLMResponse(content='primary')])
    second = StatefulLeaf([LLMResponse(content='ok', provider_state=state), quota()])
    provider = wrapper(first, second)
    result = await provider.chat_with_retry(messages=[])
    assert provider.can_resume_conversation_state(result.provider_state, 'primary')
    context = ProviderCallContext(conversation_state=result.provider_state, session_id='session')
    await provider.chat_with_retry(messages=[], provider_context=context)
    own = second.calls[-1]['provider_context']
    foreign = first.calls[-1]['provider_context']
    assert own.conversation_state.payload == {'opaque': 'handle'}
    assert own.context_window_tokens == 1000
    assert foreign.conversation_state is None
    assert own.session_id == foreign.session_id == 'session'
    assert own.events is foreign.events is context.events


@pytest.mark.parametrize('delta', ['on_content_delta', 'on_thinking_delta', 'on_tool_call_delta'])
async def test_delivered_stream_never_advances_legacy_chain(delta):
    from unittest.mock import AsyncMock

    class StreamingLeaf(Leaf):
        async def chat_stream_with_retry(self, **kwargs):
            await kwargs[delta]('delivered')
            return await self.chat(**kwargs)

    terminal = quota()
    first, second = StreamingLeaf([terminal]), Leaf([])
    callback = AsyncMock()
    result = await wrapper(first, second).chat_stream_with_retry(messages=[], retry_mode='persistent',
                                                                **{delta: callback})
    assert result is terminal
    assert not second.calls
    callback.assert_awaited_once()


def test_factory_selects_legacy_policy_and_preserves_primary_generation():
    from nanobot.providers.factory import make_provider
    config = Config.model_validate({'agents': {'defaults': {
        'provider': 'openai', 'model': 'primary', 'maxTokens': 81,
        'fallback': [{'provider': 'openai', 'model': 'secondary'}]}},
        'providers': {'openai': {'apiKey': 'fixture'}}})
    provider = make_provider(config)
    assert isinstance(provider, LegacyFallbackProvider)
    assert len(provider._candidates) == 2
    assert all(candidate.generation.max_tokens == 81 for candidate in provider._candidates)


@pytest.mark.parametrize('text,status,token,eligible', [
    ('retry after 2 seconds', None, None, True),
    ('temporarily unavailable', None, None, True),
    ('billing limit', None, None, True),
    ('quota exceeded; retry after 2', None, None, False),
    ('quota exceeded; retry after 2', 429, None, True),
    ('quota exceeded', None, 'unrecognized', False),
    ('ordinary error', None, 'insufficient_quota', True),
    ('resource_exhausted', None, None, False),
])
def test_staging_classifier_precedence(text, status, token, eligible):
    response = LLMResponse(content=text, finish_reason='error', error_status_code=status, error_type=token)
    assert LegacyFallbackProvider._eligible(response) is eligible


def test_empty_legacy_chain_is_rejected():
    with pytest.raises(ValueError, match='at least one'):
        LegacyFallbackProvider(Leaf([]), [], lambda _: None)


@pytest.mark.parametrize('keyword_dependencies', [False, True])
@pytest.mark.parametrize('requested_mode', [None, 'standard', 'persistent'])
async def test_legacy_retry_forwarding_preserves_call_forms_and_forces_standard(
    monkeypatch, keyword_dependencies, requested_mode,
):
    from nanobot.providers.fallback_provider import FallbackProvider

    provider = wrapper(Leaf([]), Leaf([]))
    response = LLMResponse(content='terminal')
    observed = []

    async def call(provider, kwargs):
        return response

    async def callback(*args):
        return None

    def guard():
        return True

    async def capture(self, *args, **kwargs):
        observed.append((self, args, kwargs))
        return response

    monkeypatch.setattr(FallbackProvider, '_retry_with_fallback', capture)
    kwargs = {'messages': []}
    original_messages = []
    streamed = [False]
    options = {
        'on_retry_wait': callback,
        'on_retry_exhausted': callback,
        'on_retry_status': callback,
        'has_streamed': streamed,
        'on_stream_recover': callback,
        'persistent_retry_guard': guard,
    }
    if requested_mode is not None:
        options['retry_mode'] = requested_mode
    if keyword_dependencies:
        result = await provider._retry_with_fallback(
            call=call, kwargs=kwargs, original_messages=original_messages, **options,
        )
    else:
        result = await provider._retry_with_fallback(call, kwargs, original_messages, **options)

    assert result is response
    [(owner, args, forwarded)] = observed
    assert owner is provider
    dependencies = dict(zip(('call', 'kwargs', 'original_messages'), args)) | forwarded
    assert dependencies['call'] is call
    assert dependencies['kwargs'] is kwargs
    assert dependencies['original_messages'] is original_messages
    assert dependencies['retry_mode'] == 'standard'
    assert all(dependencies[key] is value for key, value in options.items() if key != 'retry_mode')


async def test_legacy_exhausted_finite_traversal_returns_last_candidate_response():
    terminal = quota()
    first, second = Leaf([quota()]), Leaf([terminal])
    provider = wrapper(first, second)

    assert await provider.chat_with_retry(messages=[]) is terminal
    assert [len(leaf.calls) for leaf in (first, second)] == [1, 1]
    assert all(expiry > 0 for expiry in provider._cooldowns)
