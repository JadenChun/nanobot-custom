"""Tests for OpenAICompatProvider spec-driven behavior.

Validates that:
- OpenRouter (no strip) keeps model names intact.
- AiHubMix (strip_model_prefix=True) strips provider prefixes.
- Standard providers pass model names through as-is.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from nanobot.config.schema import Config, ModelPresetConfig
from nanobot.providers.factory import make_provider
from nanobot.providers.legacy_fallback_provider import LegacyFallbackProvider
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.providers.pooled_provider import PooledProvider, _rotation_cooldown
from nanobot.providers.registry import find_by_name


def _gemini_provider(*, api_keys, api_base, default_model, spec, timeout=None):
    """Exercise the production config/factory composition, not a mutable leaf."""
    assert spec == find_by_name("gemini")
    config = Config.model_validate({
        "providers": {"gemini": {
            "apiKey": "unused-single-key", "apiKeys": api_keys,
            "apiBase": api_base,
            **({"timeout": timeout} if timeout is not None else {}),
        }},
        "agents": {"defaults": {"provider": "gemini", "model": default_model}},
    })
    provider = make_provider(config)
    if len(api_keys) > 1:
        assert isinstance(provider, PooledProvider)
        assert [leaf.api_key for leaf in provider._leaves] == api_keys
        assert all(isinstance(leaf, OpenAICompatProvider) for leaf in provider._leaves)
    else:
        assert isinstance(provider, OpenAICompatProvider)
        assert provider.api_key == api_keys[0]
    return provider


def _fake_chat_response(content: str = "ok") -> SimpleNamespace:
    """Build a minimal OpenAI chat completion response."""
    message = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
    )
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage)


def _fake_tool_call_response() -> SimpleNamespace:
    """Build a minimal chat response that includes Gemini-style extra_content."""
    function = SimpleNamespace(
        name="exec",
        arguments='{"cmd":"ls"}',
        provider_specific_fields={"inner": "value"},
    )
    tool_call = SimpleNamespace(
        id="call_123",
        index=0,
        type="function",
        function=function,
        extra_content={"google": {"thought_signature": "signed-token"}},
    )
    message = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
        reasoning_content=None,
    )
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_openrouter_spec_is_gateway() -> None:
    spec = find_by_name("openrouter")
    assert spec is not None
    assert spec.is_gateway is True
    assert spec.default_api_base == "https://openrouter.ai/api/v1"


def test_openrouter_sets_default_attribution_headers() -> None:
    spec = find_by_name("openrouter")
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as mock_client:
        provider = OpenAICompatProvider(
            api_key="sk-or-test-key",
            api_base="https://openrouter.ai/api/v1",
            default_model="anthropic/claude-sonnet-4-5",
            spec=spec,
        )
        provider._build_client()

    headers = mock_client.call_args.kwargs["default_headers"]
    assert headers["HTTP-Referer"] == "https://github.com/HKUDS/nanobot"
    assert headers["X-OpenRouter-Title"] == "nanobot"
    assert headers["X-OpenRouter-Categories"] == "cli-agent,personal-agent"
    assert "x-session-affinity" in headers


def test_openrouter_user_headers_override_default_attribution() -> None:
    spec = find_by_name("openrouter")
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as mock_client:
        provider = OpenAICompatProvider(
            api_key="sk-or-test-key",
            api_base="https://openrouter.ai/api/v1",
            default_model="anthropic/claude-sonnet-4-5",
            extra_headers={
                "HTTP-Referer": "https://nanobot.ai",
                "X-OpenRouter-Title": "Nanobot Pro",
                "X-Custom-App": "enabled",
            },
            spec=spec,
        )
        provider._build_client()

    headers = mock_client.call_args.kwargs["default_headers"]
    assert headers["HTTP-Referer"] == "https://nanobot.ai"
    assert headers["X-OpenRouter-Title"] == "Nanobot Pro"
    assert headers["X-OpenRouter-Categories"] == "cli-agent,personal-agent"
    assert headers["X-Custom-App"] == "enabled"


@pytest.mark.asyncio
async def test_openrouter_keeps_model_name_intact() -> None:
    """OpenRouter gateway keeps the full model name (gateway does its own routing)."""
    mock_create = AsyncMock(return_value=_fake_chat_response())
    spec = find_by_name("openrouter")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as mock_client:
        client_instance = mock_client.return_value
        client_instance.chat.completions.create = mock_create

        provider = OpenAICompatProvider(
            api_key="sk-or-test-key",
            api_base="https://openrouter.ai/api/v1",
            default_model="anthropic/claude-sonnet-4-5",
            spec=spec,
        )
        await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="anthropic/claude-sonnet-4-5",
        )

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "anthropic/claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_aihubmix_strips_model_prefix() -> None:
    """AiHubMix strips the provider prefix (strip_model_prefix=True)."""
    mock_create = AsyncMock(return_value=_fake_chat_response())
    spec = find_by_name("aihubmix")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as mock_client:
        client_instance = mock_client.return_value
        client_instance.chat.completions.create = mock_create

        provider = OpenAICompatProvider(
            api_key="sk-aihub-test-key",
            api_base="https://aihubmix.com/v1",
            default_model="claude-sonnet-4-5",
            spec=spec,
        )
        await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="anthropic/claude-sonnet-4-5",
        )

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_opencode_go_strips_model_prefix() -> None:
    """OpenCode Go accepts opencode-go/<model-id> locally but wants the raw model ID upstream."""
    mock_create = AsyncMock(return_value=_fake_chat_response())
    spec = find_by_name("opencode-go")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as mock_client:
        client_instance = mock_client.return_value
        client_instance.chat.completions.create = mock_create

        provider = OpenAICompatProvider(
            api_key="opencode-test-key",
            api_base="https://opencode.ai/zen/go/v1",
            default_model="opencode-go/kimi-k2.6",
            spec=spec,
        )
        await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="opencode-go/kimi-k2.6",
        )

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "kimi-k2.6"


@pytest.mark.asyncio
async def test_opencode_deepseek_fills_missing_tool_reasoning_history() -> None:
    """A missing gateway field must not break the next DeepSeek tool continuation."""
    mock_create = AsyncMock(return_value=_fake_chat_response())
    spec = find_by_name("opencode-go")
    messages = [
        {"role": "user", "content": "research"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "result"},
    ]

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as mock_client:
        mock_client.return_value.chat.completions.create = mock_create
        provider = OpenAICompatProvider(
            api_key="opencode-test-key",
            api_base="https://opencode.ai/zen/go/v1",
            default_model="deepseek-v4-flash",
            spec=spec,
        )
        await provider.chat(messages=messages, model="deepseek-v4-flash")

    assistant = mock_create.call_args.kwargs["messages"][1]
    assert assistant["reasoning_content"] == ""
    assert assistant["content"] == ""
    assert "reasoning_content" not in messages[1]


@pytest.mark.asyncio
async def test_non_deepseek_does_not_invent_tool_reasoning_history() -> None:
    mock_create = AsyncMock(return_value=_fake_chat_response())
    spec = find_by_name("opencode-go")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as mock_client:
        mock_client.return_value.chat.completions.create = mock_create
        provider = OpenAICompatProvider(
            api_key="opencode-test-key",
            api_base="https://opencode.ai/zen/go/v1",
            default_model="kimi-k2.6",
            spec=spec,
        )
        await provider.chat(
            messages=[{
                "role": "user", "content": "run the tool",
            }, {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }],
            }, {
                "role": "tool", "tool_call_id": "call_123", "content": "result",
            }],
            model="kimi-k2.6",
        )

    assistant = mock_create.call_args.kwargs["messages"][1]
    assert "reasoning_content" not in assistant


@pytest.mark.asyncio
async def test_opencode_go_minimax_returns_clear_error() -> None:
    spec = find_by_name("opencode-go")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = OpenAICompatProvider(
            api_key="opencode-test-key",
            api_base="https://opencode.ai/zen/go/v1",
            default_model="opencode-go/minimax-m2.7",
            spec=spec,
        )
        result = await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="opencode-go/minimax-m2.7",
        )

    assert result.finish_reason == "error"
    assert "MiniMax models use the Anthropic-style /messages endpoint" in (result.content or "")


@pytest.mark.asyncio
async def test_standard_provider_passes_model_through() -> None:
    """Standard provider (e.g. deepseek) passes model name through as-is."""
    mock_create = AsyncMock(return_value=_fake_chat_response())
    spec = find_by_name("deepseek")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as mock_client:
        client_instance = mock_client.return_value
        client_instance.chat.completions.create = mock_create

        provider = OpenAICompatProvider(
            api_key="sk-deepseek-test-key",
            default_model="deepseek-chat",
            spec=spec,
        )
        await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="deepseek-chat",
        )

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "deepseek-chat"


@pytest.mark.asyncio
async def test_openai_compat_preserves_extra_content_on_tool_calls() -> None:
    """Gemini extra_content (thought signatures) must survive parse→serialize round-trip."""
    mock_create = AsyncMock(return_value=_fake_tool_call_response())
    spec = find_by_name("gemini")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI") as mock_client:
        client_instance = mock_client.return_value
        client_instance.chat.completions.create = mock_create

        provider = OpenAICompatProvider(
            api_key="test-key",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-3.1-pro-preview",
            spec=spec,
        )
        result = await provider.chat(
            messages=[{"role": "user", "content": "run exec"}],
            model="google/gemini-3.1-pro-preview",
        )

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.extra_content == {"google": {"thought_signature": "signed-token"}}
    assert tool_call.function_provider_specific_fields == {"inner": "value"}

    serialized = tool_call.to_openai_tool_call()
    assert serialized["extra_content"] == {"google": {"thought_signature": "signed-token"}}
    assert serialized["function"]["provider_specific_fields"] == {"inner": "value"}


def test_openai_model_passthrough() -> None:
    """OpenAI models pass through unchanged."""
    spec = find_by_name("openai")
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI"):
        provider = OpenAICompatProvider(
            api_key="sk-test-key",
            default_model="gpt-4o",
            spec=spec,
        )
    assert provider.get_default_model() == "gpt-4o"


@pytest.mark.asyncio
async def test_gemini_rotates_to_next_key_on_rate_limit() -> None:
    class _RateLimitError(Exception):
        status_code = 429

    spec = find_by_name("gemini")
    first = SimpleNamespace()
    first.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=_RateLimitError("quota exceeded")),
    ))
    second = SimpleNamespace()
    second.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=_fake_chat_response("ok from backup key")),
    ))

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=[first, second]) as mock_client:
        provider = _gemini_provider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        result = await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="google/gemini-2.5-pro",
        )

    assert result.content == "ok from backup key"
    assert mock_client.call_args_list[0].kwargs["api_key"] == "gem-key-1"
    assert mock_client.call_args_list[1].kwargs["api_key"] == "gem-key-2"


@pytest.mark.asyncio
async def test_gemini_rotation_does_not_mutate_process_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RateLimitError(Exception):
        status_code = 429

    spec = find_by_name("gemini")
    first = SimpleNamespace()
    first.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=_RateLimitError("quota exceeded")),
    ))
    second = SimpleNamespace()
    second.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=_fake_chat_response("ok from backup key")),
    ))

    monkeypatch.setenv("GEMINI_API_KEY", "stale-key")

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=[first, second]):
        provider = _gemini_provider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        result = await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="google/gemini-2.5-pro",
        )

    assert result.content == "ok from backup key"
    assert os.environ["GEMINI_API_KEY"] == "stale-key"
    assert [leaf.api_key for leaf in provider._leaves] == ["gem-key-1", "gem-key-2"]


@pytest.mark.asyncio
async def test_gemini_does_not_rotate_on_non_rate_limit_error() -> None:
    class _AuthError(Exception):
        status_code = 401

    spec = find_by_name("gemini")
    client = SimpleNamespace()
    client.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=_AuthError("unauthorized")),
    ))

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", return_value=client) as mock_client:
        provider = _gemini_provider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        result = await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="google/gemini-2.5-pro",
        )

    assert result.finish_reason == "error"
    assert mock_client.call_count == 1


@pytest.mark.asyncio
async def test_gemini_reports_when_all_keys_are_exhausted() -> None:
    class _RateLimitError(Exception):
        status_code = 429

    exhausted, status = AsyncMock(), AsyncMock()
    spec = find_by_name("gemini")
    first = SimpleNamespace()
    first.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=_RateLimitError("quota exceeded on key 1")),
    ))
    second = SimpleNamespace()
    second.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=_RateLimitError("quota exceeded on key 2")),
    ))

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=[first, second]):
        provider = _gemini_provider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        result = await provider.chat_with_retry(
            messages=[{"role": "user", "content": "hello"}],
            model="google/gemini-2.5-pro",
            on_retry_exhausted=exhausted, on_retry_status=status,
        )

    assert result.finish_reason == "error"
    assert result.error_status_code == 429
    assert "quota exceeded on key 2" in result.content
    assert first.chat.completions.create.await_count == 1
    assert second.chat.completions.create.await_count == 1
    exhausted.assert_awaited_once()
    status.assert_awaited_once()
    assert status.call_args.args[0].state == "exhausted"
    assert status.call_args.args[0].attempt == 2


@pytest.mark.asyncio
async def test_gemini_concurrent_requests_keep_key_rotation_request_local() -> None:
    class _RateLimitError(Exception):
        status_code = 429

    spec = find_by_name("gemini")
    call_keys = {}
    started_on_primary = 0
    started_on_backup = False
    release_primary = asyncio.Event()

    def _client_factory(*, api_key, **kwargs):
        async def _create(**req_kwargs):
            nonlocal started_on_primary, started_on_backup
            prompt = req_kwargs["messages"][0]["content"]
            call_keys.setdefault(prompt, []).append(api_key)
            if api_key == "gem-key-1":
                started_on_primary += 1
                if started_on_primary >= 2 or started_on_backup:
                    release_primary.set()
                await release_primary.wait()
                raise _RateLimitError("quota exceeded on key 1")
            started_on_backup = True
            release_primary.set()
            return _fake_chat_response("ok from backup key")

        client = SimpleNamespace()
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))
        return client

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=_client_factory):
        provider = _gemini_provider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        first, second = await asyncio.wait_for(asyncio.gather(
            provider.chat(
                messages=[{"role": "user", "content": "hello a"}],
                model="google/gemini-2.5-pro",
            ),
            provider.chat(
                messages=[{"role": "user", "content": "hello b"}],
                model="google/gemini-2.5-pro",
            ),
        ), timeout=2.0)

    assert first.finish_reason == "stop"
    assert second.finish_reason == "stop"
    assert first.content == "ok from backup key"
    assert second.content == "ok from backup key"

    assert call_keys == {
        "hello a": ["gem-key-1", "gem-key-2"],
        "hello b": ["gem-key-1", "gem-key-2"],
    }


@pytest.mark.asyncio
async def test_gemini_prefers_last_successful_key_on_follow_up_request() -> None:
    class _RateLimitError(Exception):
        status_code = 429

    spec = find_by_name("gemini")
    call_keys: list[str] = []

    def _client_factory(*, api_key, **kwargs):
        async def _create(**req_kwargs):
            call_keys.append(api_key)
            if call_keys == ["gem-key-1"]:
                raise _RateLimitError("quota exceeded on key 1")
            return _fake_chat_response(f"ok from {api_key}")

        client = SimpleNamespace()
        client.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))
        return client

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=_client_factory):
        provider = _gemini_provider(
            api_keys=["gem-key-1", "gem-key-2", "gem-key-3"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        first = await provider.chat(
            messages=[{"role": "user", "content": "hello first"}],
            model="google/gemini-2.5-pro",
        )
        second = await provider.chat(
            messages=[{"role": "user", "content": "hello second"}],
            model="google/gemini-2.5-pro",
        )

    assert first.content == "ok from gem-key-2"
    assert second.content == "ok from gem-key-2"
    assert call_keys == ["gem-key-1", "gem-key-2", "gem-key-2"]


@pytest.mark.asyncio
async def test_gemini_stream_timeout_retries_same_credential(monkeypatch) -> None:
    class _TimedOutStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Event().wait()

    async def recovered_stream():
        yield SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="ok after stream timeout"),
            finish_reason="stop", index=0,
        )], usage=None)

    create = AsyncMock(side_effect=[_TimedOutStream(), recovered_stream()])
    first = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("nanobot.providers.base.asyncio.sleep", sleep)
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", return_value=first) as client:
        provider = _gemini_provider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro", timeout=0.01,
            spec=find_by_name("gemini"),
        )
        assert all(leaf._configured_timeout == 5.0 for leaf in provider._leaves)
        result = await provider.chat_stream_with_retry(
            messages=[{"role": "user", "content": "hello"}],
        )
    assert result.content == "ok after stream timeout"
    assert create.await_count == 2
    assert client.call_count == 1
    assert client.call_args.kwargs["api_key"] == "gem-key-1"
    assert provider._leaves[1]._client is None
    assert delays == [1]


@pytest.mark.asyncio
async def test_gemini_rotates_on_service_unavailable_503() -> None:
    class _ServiceUnavailableError(Exception):
        status_code = 503

    spec = find_by_name("gemini")
    first = SimpleNamespace()
    first.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=_ServiceUnavailableError("high demand")),
    ))
    second = SimpleNamespace()
    second.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=_fake_chat_response("ok after 503 rotate")),
    ))

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=[first, second]):
        provider = _gemini_provider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        result = await provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            model="google/gemini-2.5-pro",
        )

    assert result.content == "ok after 503 rotate"


@pytest.mark.asyncio
async def test_single_key_structured_rate_limit_reaches_legacy_fallback(monkeypatch) -> None:
    """Fallback must work from the real leaf response, without pool-error prose."""
    class _RateLimitError(Exception):
        status_code = 429

    sleep = AsyncMock()
    monkeypatch.setattr("nanobot.providers.base.asyncio.sleep", sleep)
    primary_create = AsyncMock(side_effect=_RateLimitError("429 Too Many Requests"))
    backup_create = AsyncMock(return_value=_fake_chat_response("fallback succeeded"))
    primary_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=primary_create)))
    backup_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=backup_create)))
    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=[primary_client, backup_client]):
        provider = _gemini_provider(
            api_keys=["only-key"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro", spec=find_by_name("gemini"),
        )
        result = await provider.chat(messages=[{"role": "user", "content": "hello"}])
        assert result.finish_reason == "error"
        assert result.error_status_code == 429
        assert "429 Too Many Requests" in result.content
        backup = OpenAICompatProvider(
            api_key="backup", default_model="gemini-backup", spec=find_by_name("gemini"),
        )
        wrapper = LegacyFallbackProvider(
            provider, [ModelPresetConfig(model="gemini-backup", provider="gemini")],
            lambda preset: backup,
        )
        recovered = await wrapper.chat_with_retry(messages=[{"role": "user", "content": "hello"}])
    assert recovered.content == "fallback succeeded"
    # One inspected direct response, then four primary attempts before fallback.
    assert primary_create.await_count == 5
    assert backup_create.await_count == 1
    assert [call.args[0] for call in sleep.await_args_list] == [1, 2, 4]


def test_local_timeout_not_classified_as_rate_limit() -> None:
    response = OpenAICompatProvider._handle_error(TimeoutError(""))
    assert response.finish_reason == "error"
    assert response.error_kind == "timeout"
    assert response.error_status_code is None
    assert _rotation_cooldown(response) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", [False, True])
async def test_chat_stream_does_not_rotate_after_delivering_content(retry) -> None:
    """Once content has been streamed to the user, a mid-stream failure must
    not rotate to another key and concatenate a second response."""

    class _RateLimitError(Exception):
        status_code = 429

    class _PartialThenFailStream:
        """Yields one chunk with content then raises a rate-limit error."""

        def __init__(self) -> None:
            self._yielded = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._yielded:
                self._yielded = True
                delta = SimpleNamespace(content="partial-hello")
                choice = SimpleNamespace(delta=delta, finish_reason=None, index=0)
                return SimpleNamespace(choices=[choice], usage=None)
            raise _RateLimitError("429 mid-stream")

    class _BackupStream:
        def __init__(self) -> None:
            self._done = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._done:
                raise StopAsyncIteration
            self._done = True
            delta = SimpleNamespace(content="should-not-see-this")
            choice = SimpleNamespace(delta=delta, finish_reason="stop", index=0)
            usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
            return SimpleNamespace(choices=[choice], usage=usage)

    spec = find_by_name("gemini")
    first = SimpleNamespace()
    first.chat = SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=_PartialThenFailStream()),
    ))
    second = SimpleNamespace()
    second_create = AsyncMock(return_value=_BackupStream())
    second.chat = SimpleNamespace(completions=SimpleNamespace(create=second_create))

    deltas: list[str] = []

    async def _on_delta(text: str) -> None:
        deltas.append(text)

    with patch("nanobot.providers.openai_compat_provider.AsyncOpenAI", side_effect=[first, second]):
        provider = _gemini_provider(
            api_keys=["gem-key-1", "gem-key-2"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="google/gemini-2.5-pro",
            spec=spec,
        )
        call = provider.chat_stream_with_retry if retry else provider.chat_stream
        result = await call(
            messages=[{"role": "user", "content": "hello"}],
            model="google/gemini-2.5-pro",
            on_content_delta=_on_delta,
        )

    # User received only the first key's partial content — no second key content.
    assert deltas == ["partial-hello"]
    assert first.chat.completions.create.await_count == 1
    assert second_create.await_count == 0
    # The response should reflect the error; it must NOT contain the backup content.
    assert "should-not-see-this" not in (result.content or "")
