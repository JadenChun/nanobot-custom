"""H03-F5 C1-C5: one attempt observation per physical transport dispatch.

The existing transport observer is ``ProviderCallContext.on_transport_attempt``.
Dispatch counts (``D``) are measured independently at an instrumented transport;
attempt events (``A``) are recorded by the observer. Every case asserts ``A == D``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kohaku.providers.azure_openai_provider import AzureOpenAIProvider
from kohaku.providers.base import ProviderCallContext
from kohaku.providers.openai_compat_provider import OpenAICompatProvider
from kohaku.providers.registry import find_by_name
from kohaku.providers.xai_grok_provider import (
    XAIGrokProvider,
    _XAIIncompleteHostedToolError,
)


class _TransportLog:
    """Record observer events and dispatches in one ordered sequence."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, object]] = []
        self.dispatches = 0

    def observe(self, kind: str) -> None:
        self.entries.append(("attempt", kind))

    def dispatch(self) -> int:
        self.dispatches += 1
        self.entries.append(("dispatch", self.dispatches))
        return self.dispatches


class _UnsupportedCompactionError(Exception):
    status_code = 400
    body = {"error": {"message": "Unknown parameter: context_management"}}


class _ResponsesCompatibilityError(Exception):
    def __init__(self) -> None:
        super().__init__("Unknown parameter: context_management")
        self.status_code = 400
        self.response = SimpleNamespace(
            status_code=400,
            text="Unknown parameter: context_management",
            headers={},
        )


def _sdk_response(content: str) -> MagicMock:
    response = MagicMock()
    response.model_dump = MagicMock(return_value={
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        }],
        "status": "completed",
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    })
    return response


def _azure_provider(create: Any, model: str = "gpt-5.6") -> AzureOpenAIProvider:
    provider = AzureOpenAIProvider(
        api_key="fixture",
        api_base="https://example.test",
        default_model=model,
    )
    provider._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return provider


def _paired(entries: list[tuple[str, object]]) -> list[tuple[tuple[str, object], ...]]:
    """Group the log into (attempt, dispatch) pairs, preserving order."""
    assert len(entries) % 2 == 0, entries
    return list(zip(entries[0::2], entries[1::2], strict=True))


@pytest.mark.asyncio
async def test_c1_azure_compaction_fallback_observes_each_dispatch() -> None:
    log = _TransportLog()

    async def create(**_kwargs: Any) -> Any:
        if log.dispatch() == 1:
            raise _UnsupportedCompactionError()
        return _sdk_response("compaction fallback")

    provider = _azure_provider(create)
    context = ProviderCallContext(
        context_window_tokens=200_000,
        on_transport_attempt=log.observe,
    )

    response = await provider.chat(
        [{"role": "user", "content": "large request"}],
        provider_context=context,
    )

    assert log.dispatches == 2
    assert [entry for entry in log.entries if entry[0] == "attempt"] == [
        ("attempt", "main"),
        ("attempt", "main"),
    ]
    assert _paired(log.entries) == [
        (("attempt", "main"), ("dispatch", 1)),
        (("attempt", "main"), ("dispatch", 2)),
    ]
    assert response.content == "compaction fallback"
    assert response.finish_reason == "stop"
    assert provider.provider_name == "azure_openai"


@pytest.mark.asyncio
async def test_c2_openai_compatible_compaction_fallback_observes_each_dispatch() -> None:
    log = _TransportLog()

    async def create(**_kwargs: Any) -> Any:
        if log.dispatch() == 1:
            raise _ResponsesCompatibilityError()
        return _sdk_response("compaction fallback")

    provider = OpenAICompatProvider(
        api_key="fixture",
        default_model="gpt-5.6",
        spec=find_by_name("openai"),
    )
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=MagicMock())),
        responses=SimpleNamespace(create=create),
    )
    context = ProviderCallContext(
        context_window_tokens=200_000,
        on_transport_attempt=log.observe,
    )

    response = await provider.chat_with_context(
        messages=[{"role": "user", "content": "large request"}],
        provider_context=context,
    )

    assert log.dispatches == 2
    assert _paired(log.entries) == [
        (("attempt", "main"), ("dispatch", 1)),
        (("attempt", "main"), ("dispatch", 2)),
    ]
    assert response.content == "compaction fallback"
    assert provider.provider_name == "openai"


@pytest.mark.asyncio
async def test_c3_xai_recovery_observes_primary_and_recovery_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "kohaku.providers.xai_grok_provider.get_xai_oauth_token",
        lambda **_kwargs: SimpleNamespace(
            access="subscription-token",
            refresh="refresh-token",
            expires=0,
            account_id="account",
        ),
    )
    log = _TransportLog()

    async def fake_request(_url: str, _headers: dict[str, str], _body: dict[str, Any],
                           **_kwargs: Any) -> Any:
        if log.dispatch() == 1:
            raise _XAIIncompleteHostedToolError(
                [{"name": "x_search", "call_id": "search-1"}],
                usage=None,
            )
        return "recovered answer", [], "stop", None, None

    monkeypatch.setattr("kohaku.providers.xai_grok_provider._request_xai", fake_request)
    provider = XAIGrokProvider()
    provider._extra_body = {"tools": []}
    context = ProviderCallContext(on_transport_attempt=log.observe)

    response = await provider.chat(
        [{"role": "user", "content": "Search X"}],
        provider_context=context,
    )

    assert log.dispatches == 2
    assert _paired(log.entries) == [
        (("attempt", "main"), ("dispatch", 1)),
        (("attempt", "main"), ("dispatch", 2)),
    ]
    assert response.content == "recovered answer"
    assert response.finish_reason == "stop"
    assert provider.provider_name == "xai_grok"


@pytest.mark.asyncio
async def test_c4_outer_retry_observes_each_dispatch_once() -> None:
    log = _TransportLog()

    async def create(**_kwargs: Any) -> Any:
        if log.dispatch() == 1:
            raise TimeoutError("upstream stalled")
        return _sdk_response("retried")

    provider = _azure_provider(create)
    provider._CHAT_RETRY_DELAYS = (0,)
    context = ProviderCallContext(on_transport_attempt=log.observe)

    response = await provider.chat_with_retry(
        [{"role": "user", "content": "hello"}],
        provider_context=context,
    )

    assert log.dispatches == 2
    assert _paired(log.entries) == [
        (("attempt", "main"), ("dispatch", 1)),
        (("attempt", "main"), ("dispatch", 2)),
    ]
    assert response.content == "retried"
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_c5_observation_does_not_change_request_route_or_usage() -> None:
    baseline_log = _TransportLog()
    observed_log = _TransportLog()

    def _transport(log: _TransportLog) -> Any:
        async def dispatch(**_kwargs: Any) -> Any:
            log.dispatch()
            return _sdk_response("control answer")

        return AsyncMock(side_effect=dispatch)

    baseline = _azure_provider(_transport(baseline_log))
    observed = _azure_provider(_transport(observed_log))
    records: list[Any] = []
    observed.set_llm_call_observer(records.append)
    context = ProviderCallContext(on_transport_attempt=observed_log.observe)

    baseline_response = await baseline.chat_with_retry(
        [{"role": "user", "content": "ordinary request"}],
        model="gpt-5.6",
    )
    observed_response = await observed.chat_with_retry(
        [{"role": "user", "content": "ordinary request"}],
        model="gpt-5.6",
        provider_context=context,
    )

    assert baseline_log.dispatches == 1
    assert observed_log.dispatches == 1
    assert observed_log.entries == [("attempt", "main"), ("dispatch", 1)]
    assert len(records) == 1

    baseline_kwargs = baseline._client.responses.create.call_args_list[0].kwargs
    observed_kwargs = observed._client.responses.create.call_args_list[0].kwargs
    assert baseline_kwargs == observed_kwargs
    assert baseline_response.content == observed_response.content == "control answer"
    assert baseline_response.finish_reason == observed_response.finish_reason == "stop"
    assert baseline_response.usage == observed_response.usage
    assert observed.provider_name == "azure_openai"
