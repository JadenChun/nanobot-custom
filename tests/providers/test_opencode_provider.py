"""Tests for the OpenCode Zen and OpenCode Go provider registrations."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nanobot.config.schema import Config, ProvidersConfig
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.registry import PROVIDERS, find_by_name


def test_opencode_config_fields_exist() -> None:
    config = ProvidersConfig()

    assert hasattr(config, "opencode")
    assert hasattr(config, "opencode_zen")
    assert hasattr(config, "opencode_go")


def test_opencode_specs_use_openai_compatible_gateways() -> None:
    specs = {spec.name: spec for spec in PROVIDERS}

    zen = specs["opencode"]
    assert zen.backend == "openai_compat"
    assert zen.env_key == "OPENCODE_API_KEY"
    assert zen.display_name == "OpenCode Zen"
    assert zen.is_gateway is True
    assert zen.detect_by_base_keyword == "opencode.ai/zen"
    assert zen.default_api_base == "https://opencode.ai/zen/v1"
    assert "opencode" in zen.strip_model_prefixes

    zen_compat = specs["opencode_zen"]
    assert zen_compat.env_key == "OPENCODE_API_KEY"
    assert zen_compat.default_api_base == zen.default_api_base

    go = specs["opencode_go"]
    assert go.backend == "openai_compat"
    assert go.env_key == "OPENCODE_API_KEY"
    assert go.display_name == "OpenCode Go"
    assert go.is_gateway is True
    assert go.detect_by_base_keyword == "opencode.ai/zen/go"
    assert go.default_api_base == "https://opencode.ai/zen/go/v1"
    assert "opencode-go" in go.strip_model_prefixes


def test_find_by_name_opencode_providers() -> None:
    canonical = find_by_name("opencode")
    assert canonical is not None
    assert canonical.name == "opencode"

    zen = find_by_name("opencode_zen")
    assert zen is not None
    assert zen.name == "opencode_zen"

    go = find_by_name("opencode-go")
    assert go is not None
    assert go.name == "opencode_go"


def test_opencode_forced_providers_use_default_api_base() -> None:
    zen_config = Config.model_validate(
        {
            "providers": {"opencode": {"apiKey": "opencode-key"}},
            "agents": {"defaults": {"provider": "opencode", "model": "opencode/o3"}},
        }
    )

    assert zen_config.get_provider_name() == "opencode"
    assert zen_config.get_api_key() == "opencode-key"
    assert zen_config.get_api_base() == "https://opencode.ai/zen/v1"

    legacy_zen_config = Config.model_validate(
        {
            "providers": {"opencodeZen": {"apiKey": "opencode-key"}},
            "agents": {"defaults": {"provider": "opencode_zen", "model": "opencode/o3"}},
        }
    )

    assert legacy_zen_config.get_provider_name() == "opencode_zen"
    assert legacy_zen_config.get_api_key() == "opencode-key"
    assert legacy_zen_config.get_api_base() == "https://opencode.ai/zen/v1"

    go_config = Config.model_validate(
        {
            "providers": {"opencodeGo": {"apiKey": "opencode-key"}},
            "agents": {"defaults": {"provider": "opencode_go", "model": "opencode-go/o3"}},
        }
    )

    assert go_config.get_provider_name() == "opencode_go"
    assert go_config.get_api_key() == "opencode-key"
    assert go_config.get_api_base() == "https://opencode.ai/zen/go/v1"


def test_opencode_prefixes_are_stripped_before_request() -> None:
    zen_provider = OpenAICompatProvider(
        api_key=None,
        default_model="opencode/o3",
        spec=find_by_name("opencode"),
    )
    zen_kwargs = zen_provider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model="opencode/o3",
        max_tokens=1024,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    )
    assert zen_kwargs["model"] == "o3"

    go_provider = OpenAICompatProvider(
        api_key=None,
        default_model="opencode-go/o3",
        spec=find_by_name("opencode_go"),
    )
    go_kwargs = go_provider._build_kwargs(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        model="opencode-go/o3",
        max_tokens=1024,
        temperature=0.7,
        reasoning_effort=None,
        tool_choice=None,
    )
    assert go_kwargs["model"] == "o3"


def _completion():
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None),
            finish_reason="stop",
        )],
        usage=None,
    )


async def _chunks():
    yield SimpleNamespace(
        choices=[SimpleNamespace(
            delta=SimpleNamespace(content="ok"), finish_reason="stop", index=0,
        )],
        usage=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("provider_name,model", [
    ("opencode", "opencode/deepseek-v4-flash"),
    ("opencode_zen", "opencode-zen/deepseek-reasoner"),
    ("opencode_go", "opencode-go/deepseek-v4-pro"),
    ("opencode_go", "deepseek-v4-flash"),
    ("deepseek", "deepseek-reasoner"),
])
@pytest.mark.parametrize("effort", [None, "high", "none", "minimal"])
async def test_deepseek_history_reaches_transport_without_mutating_source(
    stream, provider_name, model, effort,
) -> None:
    messages = [
        {"role": "user", "content": "research"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        {"role": "assistant", "content": "prior", "reasoning_content": "kept"},
        {"role": "user", "content": "continue"},
    ]
    original = deepcopy(messages)
    create = AsyncMock(return_value=_chunks() if stream else _completion())
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as client:
        client.return_value.chat.completions.create = create
        provider = OpenAICompatProvider(
            api_key="fixture", spec=find_by_name(provider_name), default_model=model,
        )
        call = provider.chat_stream if stream else provider.chat
        response = await call(messages=messages, reasoning_effort=effort)
    assert response.content == "ok"
    assert create.await_count == 1
    sent = create.call_args.kwargs["messages"]
    if effort in (None, "high"):
        assert sent[1]["reasoning_content"] == ""
    else:
        assert "reasoning_content" not in sent[1]
    assert sent[3]["reasoning_content"] == "kept"
    assert sent[1]["content"] == (None if provider_name == "deepseek" else "")
    assert sent[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert messages == original


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("provider_name,prefix", [
    ("opencode", "opencode/"),
    ("opencode_zen", "opencode-zen/"),
    ("opencode_go", "opencode-go/"),
    ("opencode_go", ""),
])
@pytest.mark.parametrize("api_type", ["auto", "chat_completions", "responses"])
async def test_opencode_minimax_rejected_before_transport(
    stream, provider_name, prefix, api_type,
) -> None:
    create = AsyncMock(return_value=_completion())
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as client:
        client.return_value.chat.completions.create = create
        provider = OpenAICompatProvider(
            api_key="fixture", spec=find_by_name(provider_name),
            default_model="kimi-k2.6", api_type=api_type,
        )
        call = provider.chat_stream_with_retry if stream else provider.chat_with_retry
        response = await call(
            messages=[{"role": "user", "content": "hello"}],
            model=prefix + "minimax-m2.7",
        )
        assert response.finish_reason == "error"
        assert "MiniMax models use the Anthropic-style /messages endpoint" in response.content
        create.assert_not_awaited()
        client.return_value.responses.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name,model", [
    ("openrouter", "minimax/minimax-m2.7"),
    ("opencode_go", "opencode-go/kimi-k2.6"),
    ("opencode", "opencode/deepseek-chat"),
])
async def test_other_gateway_models_keep_existing_transport_contract(provider_name, model) -> None:
    create = AsyncMock(return_value=_completion())
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as client:
        client.return_value.chat.completions.create = create
        provider = OpenAICompatProvider(
            api_key="fixture", spec=find_by_name(provider_name), default_model=model,
        )
        response = await provider.chat(messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "prior"},
            {"role": "user", "content": "continue"},
        ])
    assert response.content == "ok"
    assert create.await_count == 1
    assert "reasoning_content" not in create.call_args.kwargs["messages"][1]
