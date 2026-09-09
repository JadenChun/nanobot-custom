"""Custom credential and transport settings must reach the upstream factory."""
from unittest.mock import patch

import pytest

from nanobot.config.schema import Config, InlineFallbackConfig, ProviderConfig
from nanobot.providers.factory import make_provider, provider_signature, validate_provider_setup
from nanobot.providers.openai_compat_provider import OpenAICompatProvider


def configured():
    return Config.model_validate({'agents': {'defaults': {'model': 'openai/gpt-4o', 'provider': 'openai'}},
                                  'providers': {'openai': {'apiKey': 'scalar', 'apiKeys': ['first', 'second']}}})


def test_key_pool_only_configuration_selects_and_constructs_first_leaf():
    config = configured()
    config.providers.openai.api_key = None
    config.agents.defaults.provider = 'auto'
    assert config.get_provider_name() == 'openai'
    assert config.get_api_key() == 'first'
    validate_provider_setup(config)
    provider = make_provider(config)
    assert provider.api_key == 'first'
    assert config.providers.openai.api_key is None


def test_explicit_empty_key_list_member_does_not_restore_scalar():
    config = configured()
    config.providers.openai.api_keys = ['']
    assert config.providers.openai.effective_keys == []
    assert config.get_api_key() is None
    with pytest.raises(ValueError, match='No API key'):
        validate_provider_setup(config)
    assert ProviderConfig(api_key='scalar', api_keys=[]).effective_keys == ['scalar']


@pytest.mark.parametrize('field,value', [('api_keys', ['first', 'changed']), ('rate_limit', 12), ('timeout', 19)])
def test_secondary_key_and_transport_settings_invalidate_factory_signature(field, value):
    config = configured()
    before = provider_signature(config)
    setattr(config.providers.openai, field, value)
    assert provider_signature(config) != before


async def test_factory_timeout_reaches_lazy_sdk_client():
    config = configured()
    config.providers.openai.timeout = 17.0
    provider = make_provider(config)
    for leaf in provider._leaves:
        assert leaf._configured_timeout == 17.0
    with patch('nanobot.providers.openai_compat_provider.AsyncOpenAI') as constructor:
        for leaf in provider._leaves:
            await leaf._ensure_client()
    assert constructor.call_count == 2
    assert [call.kwargs['api_key'] for call in constructor.call_args_list] == ['first', 'second']
    assert all(call.kwargs['timeout'] == 17.0 for call in constructor.call_args_list)
    assert constructor.call_args.kwargs['max_retries'] == 0


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_invalid_explicit_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match='finite and positive'):
        OpenAICompatProvider(timeout=timeout)


def test_factory_default_timeout_precedes_upstream_environment(monkeypatch):
    monkeypatch.setenv("NANOBOT_OPENAI_COMPAT_TIMEOUT_S", "300")
    assert all(leaf._configured_timeout == 60.0 for leaf in make_provider(configured())._leaves)
    config = configured()
    config.providers.openai.timeout = 1.0
    assert all(leaf._configured_timeout == 5.0 for leaf in make_provider(config)._leaves)


def test_fallback_pool_changes_invalidate_signature():
    config = configured()
    config.agents.defaults.fallback_models = [InlineFallbackConfig(model="deepseek/deepseek-chat", provider="deepseek")]
    config.providers.deepseek.api_keys = ["fallback-first", "fallback-second"]
    before = provider_signature(config)
    config.providers.deepseek.api_keys[1] = "changed"
    assert provider_signature(config) != before
