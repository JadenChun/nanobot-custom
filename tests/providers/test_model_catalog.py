from dataclasses import replace

import httpx
import pytest

from kohaku.config.schema import ProviderConfig
from kohaku.providers.model_catalog import discover_provider_models
from kohaku.providers.oauth_model_catalog import OAuthModelCatalogSnapshot
from kohaku.providers.registry import ProviderModelSpec, ProviderSpec, find_by_name


def provider_spec(name: str) -> ProviderSpec:
    spec = find_by_name(name)
    assert spec is not None
    return spec


def test_discovers_local_manager_models() -> None:
    def get(url: str, *, headers: httpx.Headers, **_kwargs: object) -> httpx.Response:
        assert url == "http://127.0.0.1:8082/v1/models"
        assert "Authorization" not in headers
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={"data": [{"id": "profile-32k", "context_length": 32768}]},
        )

    result = discover_provider_models(provider_spec("local_manager"), ProviderConfig(), http_get=get)

    assert result["provider"] == "local_manager"
    assert result["status"] == "available"
    assert result["catalog_kind"] == "local"
    assert result["model_count"] == 1
    assert result["models"] == [
        {"id": "profile-32k", "label": None, "owned_by": None, "context_window": 32768},
    ]


def test_discovers_openai_compatible_models_with_resolved_headers(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_CATALOG_TEAM", "team-test")
    config = ProviderConfig(
        api_base="https://models.example.test/v1/",
        api_key="key-test",
        extra_headers={"X-Team": "${MODEL_CATALOG_TEAM}"},
    )

    def get(url: str, *, headers: httpx.Headers, **_kwargs: object) -> httpx.Response:
        assert url == "https://models.example.test/v1/models"
        assert headers["Authorization"] == "Bearer key-test"
        assert headers["X-Team"] == "team-test"
        return httpx.Response(200, request=httpx.Request("GET", url), json={"data": [{"id": "team-model"}]})

    result = discover_provider_models(provider_spec("openai"), config, http_get=get)

    assert result["status"] == "available"
    assert [model["id"] for model in result["models"]] == ["team-model"]


@pytest.mark.parametrize(
    ("provider", "api_base", "models_url", "extra_headers"),
    [
        ("anthropic", None, "https://api.anthropic.com/v1/models", {}),
        ("minimax_anthropic", None, "https://api.minimax.io/anthropic/v1/models", {}),
        (
            "anthropic",
            "https://models.example.test/v1/",
            "https://models.example.test/v1/models",
            {"Anthropic-Version": "test-version", "x-api-key": "override"},
        ),
        ("anthropic", "https://models.example.test/", "https://models.example.test/v1/models", {}),
    ],
)
def test_discovers_anthropic_models(provider, api_base, models_url, extra_headers) -> None:
    config = ProviderConfig(api_base=api_base, api_key="key-test", extra_headers=extra_headers)

    def get(url: str, *, headers: httpx.Headers, **_kwargs: object) -> httpx.Response:
        assert url == models_url
        assert headers["X-Api-Key"] == extra_headers.get("x-api-key", "key-test")
        assert headers["anthropic-version"] == extra_headers.get("Anthropic-Version", "2023-06-01")
        assert "Authorization" not in headers
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={"data": [{"id": "catalog-model", "display_name": "Catalog Model"}]},
        )

    result = discover_provider_models(provider_spec(provider), config, http_get=get)

    assert result["status"] == "available"
    assert result["catalog_kind"] == "official"
    assert [(model["id"], model["label"]) for model in result["models"]] == [
        ("catalog-model", "Catalog Model"),
    ]


def unexpected_http(*_args: object, **_kwargs: object) -> httpx.Response:
    pytest.fail("This catalog does not use the HTTP model list.")


def test_discovers_builtin_models() -> None:
    spec = replace(
        provider_spec("openai_codex"),
        model_catalog="builtin",
        builtin_models=(ProviderModelSpec(id="builtin", label="Built-in Model", context_window=200_000),),
    )

    result = discover_provider_models(spec, ProviderConfig(), http_get=unexpected_http)

    assert result["status"] == "available"
    assert result["catalog_kind"] == "builtin"
    assert result["model_count"] == 1
    assert result["models"] == [{
        "id": "builtin",
        "label": "Built-in Model",
        "description": None,
        "owned_by": spec.label,
        "context_window": 200_000,
    }]


def test_discovers_hybrid_oauth_models(monkeypatch) -> None:
    model = ProviderModelSpec(
        id="oauth-model",
        label="OAuth Model",
        context_window=200_000,
        reasoning_efforts=("medium", "high"),
        supports_backend_search=True,
    )
    monkeypatch.setattr(
        "kohaku.providers.model_catalog.get_oauth_model_catalog",
        lambda *_args, **_kwargs: OAuthModelCatalogSnapshot((model,), "remote", 123, "Catalog updated."),
    )

    result = discover_provider_models(
        provider_spec("openai_codex"), ProviderConfig(), http_get=unexpected_http,
    )

    assert result["catalog_kind"] == "hybrid"
    assert result["source"] == "remote"
    assert result["fetched_at"] == 123
    assert result["message"] == "Catalog updated."
    assert result["model_count"] == 1
    assert result["models"][0] == {
        "id": "oauth-model",
        "label": "OAuth Model",
        "description": None,
        "owned_by": "OpenAI Codex",
        "context_window": 200_000,
        "reasoning_efforts": ["medium", "high"],
        "supports_backend_search": True,
    }


@pytest.mark.parametrize("provider", ["bedrock", "azure_openai"])
def test_unsupported_providers_do_not_fetch_models(provider) -> None:
    result = discover_provider_models(
        provider_spec(provider), ProviderConfig(), http_get=unexpected_http,
    )
    assert result["catalog_kind"] == result["status"] == "unsupported"
    assert result["models"] == []
