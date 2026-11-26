"""Shared logical prompt metering and source compaction reserves."""

from __future__ import annotations

from typing import Any

from kohaku.providers.base import LLMProvider
from kohaku.providers.request_budget import DEFAULT_FINALIZED_REQUEST_SAFETY_BUFFER
from kohaku.utils.helpers import (
    count_text_tokens,
    estimate_prompt_tokens_chain,
    truncate_text_to_tokens,
)


class TokenMeter:
    """Measure model input and keep completion reserves in token units."""

    count_text = staticmethod(count_text_tokens)
    truncate_text = staticmethod(truncate_text_to_tokens)

    @staticmethod
    def estimate_messages_tokens(
        provider: object,
        model: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[int, str]:
        return estimate_prompt_tokens_chain(provider, model, messages, tools)

    @staticmethod
    def unwrap_anchor_provider(provider: LLMProvider) -> LLMProvider:
        while True:
            leaf: object = getattr(provider, "anchor_provider", None)
            if not isinstance(leaf, LLMProvider):
                return provider
            provider = leaf

    @staticmethod
    def compute_summary_reserve(
        budget: int, max_tool_result_chars: int, generation_allowance: int, envelope_tokens: int,
    ) -> int:
        if budget <= 0:
            return 0
        max_tool_tokens = (
            max(0, max_tool_result_chars // 4) if type(max_tool_result_chars) is int else 0
        )
        growth = min(max_tool_tokens, max(1, budget // 8))
        reserve = envelope_tokens + generation_allowance + DEFAULT_FINALIZED_REQUEST_SAFETY_BUFFER
        return min(reserve + growth, budget - 1)
