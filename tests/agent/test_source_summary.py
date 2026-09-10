"""Bounded auxiliary source summaries never archive or fall back to raw text."""

import asyncio
from unittest.mock import Mock

import pytest

from nanobot.agent.memory import Consolidator
from nanobot.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ToolCallRequest,
)
from nanobot.utils.llm_runtime import LLMRuntime


class SummaryProvider(LLMProvider):
    def __init__(self, response):
        super().__init__(provider_name='fixture')
        self.response = response
        self.calls = []

    def get_default_model(self):
        return 'fixture'

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if isinstance(self.response, BaseException):
            raise self.response
        if self.response == 'wait':
            await asyncio.Event().wait()
        return self.response


def runtime(provider, window=8192):
    return LLMRuntime(provider, 'fixture', GenerationSettings(max_tokens=256), window)


async def test_summary_is_observed_once_and_has_no_store_dependency():
    usage = LLMUsage(input_tokens=10, output_tokens=3, total_tokens=13, reported_tokens=13)
    provider = SummaryProvider(LLMResponse(content='Objective; evidence; next action.', usage=usage))
    observer = Mock()
    provider.set_llm_call_observer(observer)
    source = [{'role': 'user', 'content': 'objective'}]
    result = await Consolidator.summarize_source_chunk(source, 'prior facts', runtime=runtime(provider))
    assert result.status == 'complete' and result.calls == 1 and result.usage == usage
    assert result.provider == 'fixture' and result.model == 'fixture'
    assert len(provider.calls) == 1
    messages, kwargs = provider.calls[0]
    assert kwargs['tools'] is None and 'provider_context' not in kwargs
    assert 'prior facts' in messages[1]['content'] and 'objective' in messages[1]['content']
    assert source == [{'role': 'user', 'content': 'objective'}]
    observer.assert_called_once()


@pytest.mark.parametrize('response', [
    LLMResponse(content=''), LLMResponse(content='partial', finish_reason='length'),
    LLMResponse(content='error', finish_reason='error'),
    LLMResponse(content='tool', tool_calls=[ToolCallRequest(id='x', name='write_file', arguments={})]),
    LLMResponse(content='word ' * 5000), RuntimeError('provider failure'),
])
async def test_invalid_summary_has_no_raw_fallback_or_retry(response):
    provider = SummaryProvider(response)
    result = await Consolidator.summarize_source_chunk([{'role': 'user', 'content': 'raw evidence'}],
        None, runtime=runtime(provider), max_output_tokens=32)
    assert result.status == 'invalid_response' and result.summary is None
    assert result.calls == 1 and len(provider.calls) == 1


async def test_auxiliary_overflow_causes_zero_transport():
    provider = SummaryProvider(LLMResponse(content='should not run'))
    result = await Consolidator.summarize_source_chunk([{'role': 'tool', 'content': 'large ' * 1000}],
        None, runtime=runtime(provider, window=512))
    assert result.status == 'input_overflow' and result.calls == 0
    assert provider.calls == []


async def test_timeout_is_bounded_and_does_not_retry():
    provider = SummaryProvider('wait')
    result = await Consolidator.summarize_source_chunk([{'role': 'user', 'content': 'input'}],
        None, runtime=runtime(provider), timeout_seconds=0.01)
    assert result.status == 'timeout' and result.summary is None and len(provider.calls) == 1


async def test_cancellation_propagates_without_candidate():
    provider = SummaryProvider(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await Consolidator.summarize_source_chunk([{'role': 'user', 'content': 'input'}],
            None, runtime=runtime(provider))
    assert len(provider.calls) == 1


async def test_runtime_system_messages_cannot_enter_summary_evidence():
    provider = SummaryProvider(LLMResponse(content='should not run'))
    with pytest.raises(ValueError, match='source evidence'):
        await Consolidator.summarize_source_chunk([{'role': 'system', 'content': 'current policy'}],
            None, runtime=runtime(provider))
    assert provider.calls == []


async def test_summary_does_not_fallback_or_rotate_credentials():
    from nanobot.providers.fallback_provider import FallbackProvider
    from nanobot.providers.pooled_provider import PooledProvider

    primary = SummaryProvider(LLMResponse(content='rate limit', finish_reason='error', error_status_code=429))
    spare = SummaryProvider(LLMResponse(content='must not run'))
    pooled = PooledProvider([primary, spare], ['fixture-one', 'fixture-two'])
    factory = Mock(side_effect=AssertionError('unexpected fallback'))
    routed = FallbackProvider(pooled, [object()], factory)
    result = await Consolidator.summarize_source_chunk([{'role': 'user', 'content': 'input'}],
        None, runtime=runtime(routed))
    assert result.status == 'invalid_response' and result.calls == 1
    assert len(primary.calls) == 1 and spare.calls == []
    factory.assert_not_called()


async def test_auxiliary_disables_anthropic_output_budget_expansion():
    from nanobot.providers.anthropic_provider import AnthropicProvider

    provider = SummaryProvider(LLMResponse(content='facts'))
    captured = LLMRuntime(provider, 'claude-sonnet-4-5',
        GenerationSettings(max_tokens=256, reasoning_effort='high'), 8192)
    await Consolidator.summarize_source_chunk([{'role': 'user', 'content': 'input'}],
        None, runtime=captured, max_output_tokens=256)
    messages, kwargs = provider.calls[0]
    anthropic = object.__new__(AnthropicProvider)
    anthropic.default_model = captured.model
    anthropic.extra_headers = {}
    built = anthropic._build_kwargs(messages, None, captured.model, kwargs['max_tokens'],
        kwargs['temperature'], kwargs['reasoning_effort'], None)
    assert built['max_tokens'] == 256
    assert built.get('thinking', {}).get('type') != 'enabled'
