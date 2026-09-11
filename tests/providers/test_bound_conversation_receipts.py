"""Concrete Responses transport receipt and settlement boundary tests."""

from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.providers.azure_openai_provider import AzureOpenAIProvider
from nanobot.providers.fallback_provider import FallbackProvider
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.pooled_provider import PooledProvider
from nanobot.providers.registry import find_by_name


def _response(*, status: str = "completed") -> MagicMock:
    response = MagicMock()
    response.model_dump.return_value = {
        "status": status,
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}],
        }],
        "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
    }
    return response


async def _completed_stream():
    delta = MagicMock(type="response.output_text.delta", delta="hello")
    completed = MagicMock(type="response.completed", response=_response())
    for event in (delta, completed):
        yield event


def _openai(
    *,
    api_key: str = "static-key",
    api_base: str | None = None,
    api_type: str = "responses",
    extra_headers: dict[str, str] | None = None,
    extra_query: dict[str, str] | None = None,
) -> OpenAICompatProvider:
    spec = find_by_name("openai")
    assert spec is not None
    return OpenAICompatProvider(
        api_key=api_key,
        api_base=api_base,
        default_model="gpt-5.6",
        spec=spec,
        api_type=api_type,
        extra_headers=extra_headers,
        extra_query=extra_query,
    )


def _set_responses_client(provider: object, result: object) -> AsyncMock:
    create = AsyncMock(return_value=result)
    provider._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return create


@pytest.mark.asyncio
async def test_explicit_direct_openai_responses_stamps_normal_and_stream_receipts() -> None:
    provider = _openai(api_base="https://API.OpenAI.com/v1/")
    create = _set_responses_client(provider, _response())

    result = await provider.chat([{"role": "user", "content": "hello"}])
    receipt = result.provider_state.transport_receipt if result.provider_state is not None else None
    assert provider.supports_bound_conversation_state()
    assert receipt is not None
    assert receipt.provider == "openai"
    assert receipt.account == sha256(b"static-key").hexdigest()
    assert receipt.endpoint == "https://api.openai.com/v1"
    assert receipt.protocol == "openai.responses.v1"
    assert receipt.model == create.call_args.kwargs["model"] == "gpt-5.6"
    assert "static-key" not in str(result.provider_state.to_private_record())

    _set_responses_client(provider, _completed_stream())
    streamed = await provider.chat_stream([{"role": "user", "content": "again"}])
    assert streamed.provider_state is not None
    assert streamed.provider_state.transport_receipt == receipt


@pytest.mark.asyncio
async def test_static_azure_responses_stamps_normal_and_stream_receipts() -> None:
    provider = AzureOpenAIProvider(
        api_key="azure-static-key",
        api_base="https://RES.OpenAI.Azure.com/",
        default_model="deployment-a",
    )
    create = _set_responses_client(provider, _response())

    result = await provider.chat(
        [{"role": "user", "content": "hello"}], model="deployment-b"
    )
    receipt = result.provider_state.transport_receipt if result.provider_state is not None else None
    assert provider.supports_bound_conversation_state()
    assert receipt is not None
    assert receipt.provider == "azure_openai"
    assert receipt.account == sha256(b"azure-static-key").hexdigest()
    assert receipt.endpoint == "https://res.openai.azure.com/openai/v1"
    assert receipt.protocol == "azure_openai.responses.v1"
    assert receipt.model == create.call_args.kwargs["model"] == "deployment-b"
    assert "azure-static-key" not in str(result.provider_state.to_private_record())

    _set_responses_client(provider, _completed_stream())
    streamed = await provider.chat_stream([{"role": "user", "content": "again"}], model="deployment-b")
    assert streamed.provider_state is not None
    assert streamed.provider_state.transport_receipt == receipt


def test_receipt_eligibility_rejects_auto_oauth_mutable_and_rotated_configurations() -> None:
    explicit = _openai()
    expected = explicit.conversation_state_transport_receipt("gpt-5.6")
    assert expected is not None
    assert expected != _openai(api_key="rotated-key").conversation_state_transport_receipt("gpt-5.6")
    assert expected != _openai(api_base="https://api.openai.com/v2").conversation_state_transport_receipt("gpt-5.6")

    assert not _openai(api_type="auto").supports_bound_conversation_state()
    assert not OpenAICompatProvider(
        api_key="oauth-token",
        api_base="https://api.githubcopilot.com",
        default_model="gpt-5.6",
        spec=find_by_name("github_copilot"),
    ).supports_bound_conversation_state()
    assert not OpenAICompatProvider(
        api_key="static-key",
        default_model="gpt-5.6",
        spec=find_by_name("openai"),
        api_type="responses",
        extra_headers={"OpenAI-Organization": "mutable-account"},
    ).supports_bound_conversation_state()
    assert not OpenAICompatProvider(
        api_key="static-key",
        default_model="gpt-5.6",
        spec=find_by_name("openai"),
        api_type="responses",
        extra_query={"api-key": "mutable-secret"},
    ).supports_bound_conversation_state()
    assert not OpenAICompatProvider(
        api_key="static-key",
        default_model="gpt-5.6",
        spec=find_by_name("openai"),
        api_type="responses",
        extra_body={"instructions": "replace the portable history"},
    ).supports_bound_conversation_state()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra_headers", "extra_query"),
    [
        ({"Authorization": "Bearer overridden-account"}, None),
        ({"OpenAI-Organization": "overridden-organization"}, None),
        ({}, None),
        (None, {"api-key": "overridden-account"}),
        (None, {}),
    ],
)
async def test_supplied_transport_overrides_remain_ineligible_after_source_mutation(
    monkeypatch: pytest.MonkeyPatch,
    extra_headers: dict[str, str] | None,
    extra_query: dict[str, str] | None,
) -> None:
    source = extra_headers if extra_headers is not None else extra_query
    assert source is not None
    expected_headers = dict(extra_headers or {})
    expected_query = dict(extra_query or {})
    provider = _openai(extra_headers=extra_headers, extra_query=extra_query)

    source.clear()
    assert provider.extra_headers == expected_headers
    assert provider._extra_query == expected_query
    assert not provider.supports_bound_conversation_state()

    client_factory = MagicMock()
    monkeypatch.setattr(
        "nanobot.providers.openai_compat_provider.AsyncOpenAI",
        client_factory,
    )
    await provider._ensure_client()

    source["after-client-init"] = "must-not-affect-provider"
    assert provider.extra_headers == expected_headers
    assert provider._extra_query == expected_query
    assert not provider.supports_bound_conversation_state()
    assert client_factory.call_args is not None
    client_kwargs = client_factory.call_args.kwargs
    if extra_headers is not None:
        for name, value in expected_headers.items():
            assert client_kwargs["default_headers"][name] == value
    if extra_query is not None and expected_query:
        assert client_kwargs["default_query"] == expected_query


@pytest.mark.asyncio
async def test_auto_incomplete_and_wrapped_responses_never_expose_bound_receipts() -> None:
    auto = _openai(api_type="auto")
    _set_responses_client(auto, _response())
    auto_result = await auto.chat([{"role": "user", "content": "hello"}])
    assert auto_result.provider_state is not None
    assert auto_result.provider_state.transport_receipt is None

    incomplete = _openai()
    _set_responses_client(incomplete, _response(status="incomplete"))
    incomplete_result = await incomplete.chat([{"role": "user", "content": "hello"}])
    assert incomplete_result.provider_state is None

    leaf = _openai()
    _set_responses_client(leaf, _response())
    pool = PooledProvider([leaf], ["static-key"])
    pooled_result = await pool.chat([{"role": "user", "content": "hello"}])
    assert pooled_result.provider_state is not None
    assert pooled_result.provider_state.transport_receipt is None
    assert not pool.supports_bound_conversation_state()

    fallback_leaf = _openai()
    _set_responses_client(fallback_leaf, _response())
    fallback = FallbackProvider(fallback_leaf, [], lambda _preset: fallback_leaf)
    fallback_result = await fallback.chat([{"role": "user", "content": "hello"}])
    assert fallback_result.provider_state is not None
    assert fallback_result.provider_state.transport_receipt is None
    assert not fallback.supports_bound_conversation_state()
