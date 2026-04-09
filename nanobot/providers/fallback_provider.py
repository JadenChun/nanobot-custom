"""Fallback provider — wraps multiple providers in a priority chain."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse


class FallbackProvider(LLMProvider):
    """Wraps multiple providers with automatic fallback on quota exhaustion.

    Each call always starts with the primary provider (index 0). If the
    primary returns a quota-exhaustion error, the next provider in the
    chain is tried, and so on. This ensures automatic recovery once the
    primary provider's quota resets.

    Rate-limit retries are handled internally by each wrapped provider's
    ``chat_with_retry()`` / ``chat_stream_with_retry()`` — the fallback
    layer only catches errors that survive those retries.
    """

    def __init__(self, providers: list[tuple[LLMProvider, str]]) -> None:
        """
        Args:
            providers: Ordered list of ``(provider_instance, model_name)``
                tuples. The first entry is the primary; the rest are fallbacks.
        """
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider")
        self._providers = providers
        super().__init__()

    # ------------------------------------------------------------------
    # Generation settings — propagate to all wrapped providers
    # ------------------------------------------------------------------

    @property  # type: ignore[override]
    def generation(self) -> GenerationSettings:
        return self._providers[0][0].generation

    @generation.setter
    def generation(self, value: GenerationSettings) -> None:
        for provider, _ in self._providers:
            provider.generation = value

    # ------------------------------------------------------------------
    # Fallback logic
    # ------------------------------------------------------------------

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = LLMProvider._SENTINEL,
        temperature: object = LLMProvider._SENTINEL,
        reasoning_effort: object = LLMProvider._SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        last_response: LLMResponse | None = None

        for idx, (provider, provider_model) in enumerate(self._providers):
            effective_model = model if idx == 0 and model is not None else provider_model

            response = await provider.chat_with_retry(
                messages=messages,
                tools=tools,
                model=effective_model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            )

            if response.finish_reason != "error":
                if idx > 0:
                    logger.info("Fallback provider #{} ({}) succeeded", idx, provider_model)
                return response

            last_response = response

            if not self._is_quota_exhaustion(response.content):
                return response

            if idx + 1 < len(self._providers):
                next_provider, next_model = self._providers[idx + 1]
                logger.warning(
                    "Provider {} (model={}) quota exhausted, falling back to {} (model={}): {}",
                    type(provider).__name__, provider_model,
                    type(next_provider).__name__, next_model,
                    (response.content or "")[:120],
                )

        return last_response or LLMResponse(
            content="Error: All fallback providers exhausted.",
            finish_reason="error",
        )

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = LLMProvider._SENTINEL,
        temperature: object = LLMProvider._SENTINEL,
        reasoning_effort: object = LLMProvider._SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        last_response: LLMResponse | None = None

        delivered_any = False
        effective_delta = on_content_delta
        if on_content_delta is not None:
            user_delta = on_content_delta

            async def _tracked_delta(text: str) -> None:
                nonlocal delivered_any
                delivered_any = True
                await user_delta(text)

            effective_delta = _tracked_delta

        for idx, (provider, provider_model) in enumerate(self._providers):
            effective_model = model if idx == 0 and model is not None else provider_model

            response = await provider.chat_stream_with_retry(
                messages=messages,
                tools=tools,
                model=effective_model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
                on_content_delta=effective_delta,
            )

            if response.finish_reason != "error":
                if idx > 0:
                    logger.info("Fallback provider #{} ({}) succeeded (streaming)", idx, provider_model)
                return response

            last_response = response

            if delivered_any:
                # Primary already streamed content to the user — falling back to
                # a secondary provider would concatenate a second response on top.
                return response

            if not self._is_quota_exhaustion(response.content):
                return response

            if idx + 1 < len(self._providers):
                next_provider, next_model = self._providers[idx + 1]
                logger.warning(
                    "Provider {} (model={}) quota exhausted, falling back to {} (model={}) (streaming): {}",
                    type(provider).__name__, provider_model,
                    type(next_provider).__name__, next_model,
                    (response.content or "")[:120],
                )

        return last_response or LLMResponse(
            content="Error: All fallback providers exhausted.",
            finish_reason="error",
        )

    # ------------------------------------------------------------------
    # ABC compliance — delegate to primary provider
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        provider, provider_model = self._providers[0]
        return await provider.chat(
            messages=messages,
            tools=tools,
            model=model or provider_model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )

    def get_default_model(self) -> str:
        return self._providers[0][1]
