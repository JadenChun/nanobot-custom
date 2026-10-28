"""Tests for the Local Inference Manager provider registration."""

from unittest.mock import patch

from nanobot.config.schema import Config, ProvidersConfig
from nanobot.providers.factory import make_provider, validate_provider_setup
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import PROVIDERS, find_by_name


def test_local_manager_config_field_exists() -> None:
    assert hasattr(ProvidersConfig(), "local_manager")


def test_local_manager_provider_in_registry() -> None:
    specs = {spec.name: spec for spec in PROVIDERS}
    spec = specs["local_manager"]

    assert spec.backend == "openai_compat"
    assert spec.env_key == ""
    assert spec.display_name == "Local Inference Manager"
    assert spec.is_local is True
    assert spec.detect_by_base_keyword == "127.0.0.1:8082"
    assert spec.default_api_base == "http://127.0.0.1:8082/v1"


def test_find_by_name_local_manager_accepts_camel_case() -> None:
    spec = find_by_name("localManager")

    assert spec is not None
    assert spec.name == "local_manager"


def test_local_manager_preset_routes_to_live_gateway() -> None:
    config = Config.model_validate(
        {
            "providers": {
                "localManager": {
                    "apiBase": "http://127.0.0.1:8082/v1",
                    "apiKey": "local-no-key",
                },
            },
            "modelPresets": {
                "local": {
                    "model": "Qwen3.8-Flash-Next Uncensored IQ2_XS (Strata)",
                    "provider": "local_manager",
                },
            },
            "agents": {"defaults": {"modelPreset": "local"}},
        }
    )

    validate_provider_setup(config)
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = make_provider(config)

    assert isinstance(provider, OpenAICompatProvider)
    assert provider.api_base == "http://127.0.0.1:8082/v1"
    assert provider.default_model == "Qwen3.8-Flash-Next Uncensored IQ2_XS (Strata)"
