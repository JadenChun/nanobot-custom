"""Hard finalized-request limits for memory archive calls."""

from __future__ import annotations

import pytest

from nanobot.agent.memory import MemoryArchiver, MemoryStore
from nanobot.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ProviderConversationState,
)
from nanobot.providers.request_budget import FinalizedRequestBudgetExceeded
from nanobot.utils.llm_runtime import LLMRuntime


class _ArchiveProvider(LLMProvider):
    def __init__(self, response: LLMResponse | BaseException) -> None:
        super().__init__(provider_name="archive-fixture")
        self.response = response
        self.contexts = []

    def get_default_model(self) -> str:
        return "archive-model"

    def can_resume_conversation_state(
        self,
        state: ProviderConversationState,
        model: str | None = None,
    ) -> bool:
        return True

    async def chat(self, **_kwargs) -> LLMResponse:
        raise AssertionError("archive calls must retain their provider context")

    async def chat_with_context(self, *, provider_context, **_kwargs) -> LLMResponse:
        self.contexts.append(provider_context)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def _runtime(provider: LLMProvider) -> LLMRuntime:
    return LLMRuntime(
        provider,
        "archive-model",
        GenerationSettings(max_tokens=64),
        4_096,
    )


@pytest.mark.asyncio
async def test_stateful_memory_archive_carries_hard_limits(tmp_path) -> None:
    provider = _ArchiveProvider(LLMResponse(content="archived evidence"))
    archiver = MemoryArchiver(MemoryStore(tmp_path), lambda **_kwargs: [], lambda: [])
    state = ProviderConversationState(
        kind="fixture",
        provider="archive-fixture",
        model="archive-model",
        version=1,
        payload={"context_tokens": 1},
    )

    summary = await archiver.archive(
        [{"role": "user", "content": "evidence"}],
        runtime=_runtime(provider),
        session_key="memory:fixture",
        history=[{"role": "system", "content": "system"}],
        request_tools=[],
        provider_state=state,
    )

    assert summary == "archived evidence"
    context = provider.contexts[0]
    assert context.request_limits is not None
    assert context.request_limits.total_tokens == 4_096
    assert context.request_limits.completion_reserve_tokens == 64


@pytest.mark.asyncio
async def test_memory_archive_rejects_an_oversized_resumed_context_before_call(tmp_path) -> None:
    provider = _ArchiveProvider(LLMResponse(content="must not run"))
    archiver = MemoryArchiver(MemoryStore(tmp_path), lambda **_kwargs: [], lambda: [])
    state = ProviderConversationState(
        kind="fixture",
        provider="archive-fixture",
        model="archive-model",
        version=1,
        payload={"context_tokens": 4_000},
    )

    summary = await archiver.archive(
        [{"role": "user", "content": "evidence"}],
        runtime=_runtime(provider),
        session_key="memory:fixture",
        history=[{"role": "system", "content": "system"}],
        request_tools=[],
        provider_state=state,
    )

    assert summary is None
    assert provider.contexts == []


@pytest.mark.asyncio
async def test_memory_archive_rebuilds_unknown_state_as_stateless_request(tmp_path) -> None:
    provider = _ArchiveProvider(LLMResponse(content="archived evidence"))
    archiver = MemoryArchiver(MemoryStore(tmp_path), lambda **_kwargs: [], lambda: [])
    state = ProviderConversationState(
        kind="fixture",
        provider="archive-fixture",
        model="archive-model",
        version=1,
        payload={},
    )

    summary = await archiver.archive(
        [{"role": "user", "content": "evidence"}],
        runtime=_runtime(provider),
        session_key="memory:fixture",
        history=[{"role": "system", "content": "system"}],
        request_tools=[],
        provider_state=state,
    )

    assert summary == "archived evidence"
    assert len(provider.contexts) == 1
    assert provider.contexts[0].conversation_state is None


@pytest.mark.asyncio
async def test_memory_final_budget_rejection_does_not_create_a_checkpoint(tmp_path) -> None:
    provider = _ArchiveProvider(FinalizedRequestBudgetExceeded())
    store = MemoryStore(tmp_path)
    archiver = MemoryArchiver(store, lambda **_kwargs: [], lambda: [])

    summary = await archiver.archive(
        [{"role": "user", "content": "evidence"}],
        runtime=_runtime(provider),
        session_key="memory:fixture",
        history=[{"role": "system", "content": "system"}],
        request_tools=[],
    )

    assert summary is None
    assert not store.history_file.exists()
