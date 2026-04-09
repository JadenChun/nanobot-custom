"""Fallback provider — wraps multiple providers in a priority chain."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse


class FallbackProvider(LLMProvider):
    """Wraps multiple providers with automatic fallback on quota exhaustion.

    Each call prefers the primary provider (index 0). When a provider returns
    a quota-exhaustion error, it is marked cooling for
    ``_PROVIDER_QUOTA_COOLDOWN_S`` and subsequent requests skip it until the
    cooldown expires — at which point it is probed again and, if it succeeds,
    restored as the preferred provider. Non-quota errors short-circuit the
    chain (falling further would likely just reproduce the same failure).

    Rate-limit retries are handled internally by each wrapped provider's
    ``chat_with_retry()`` / ``chat_stream_with_retry()`` — the fallback
    layer only catches errors that survive those retries.
    """

    # Free-tier quotas typically refresh on a daily-ish cadence, so a 12h
    # cooldown keeps us from hammering an exhausted provider while still
    # letting us re-probe it roughly twice per day.
    _PROVIDER_QUOTA_COOLDOWN_S = 12 * 3600.0

    def __init__(self, providers: list[tuple[LLMProvider, str]]) -> None:
        """
        Args:
            providers: Ordered list of ``(provider_instance, model_name)``
                tuples. The first entry is the primary; the rest are fallbacks.
        """
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider")
        self._providers = providers
        self._provider_retry_after: list[float] = [0.0] * len(providers)
        super().__init__()

    # ------------------------------------------------------------------
    # Provider cooldown helpers
    # ------------------------------------------------------------------

    def _provider_walk_order(self) -> list[int]:
        """Return provider indices to try this request, ready ones first.

        Ready providers keep their configured order. Cooling providers are
        appended after them, sorted by earliest expiry so the one closest
        to recovery is probed first if every ready provider errors out.
        """
        now = time.monotonic()
        ready: list[int] = []
        cooling: list[int] = []
        for idx in range(len(self._providers)):
            if self._provider_retry_after[idx] <= now:
                ready.append(idx)
            else:
                cooling.append(idx)
        cooling.sort(key=lambda i: self._provider_retry_after[i])
        return ready + cooling

    def _mark_provider_exhausted(self, idx: int, provider_model: str) -> None:
        """Put a provider into its quota-exhausted cooldown window."""
        self._provider_retry_after[idx] = time.monotonic() + self._PROVIDER_QUOTA_COOLDOWN_S
        logger.warning(
            "Provider #{} ({} model={}) marked exhausted; cooling down for {:.0f}h",
            idx,
            type(self._providers[idx][0]).__name__,
            provider_model,
            self._PROVIDER_QUOTA_COOLDOWN_S / 3600.0,
        )

    def _clear_provider_cooldown_if_recovered(self, idx: int, provider_model: str) -> None:
        """Called after a successful call — lift the cooldown on a probe hit."""
        if self._provider_retry_after[idx] > 0.0:
            logger.info(
                "Provider #{} ({} model={}) recovered from cooldown",
                idx,
                type(self._providers[idx][0]).__name__,
                provider_model,
            )
            self._provider_retry_after[idx] = 0.0

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
        order = self._provider_walk_order()

        for pos, idx in enumerate(order):
            provider, provider_model = self._providers[idx]
            # The caller-provided model override only applies to the primary
            # (index 0). When we fall back to a different provider, we must
            # use that provider's configured model — it likely can't serve
            # the primary's model id.
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
                self._clear_provider_cooldown_if_recovered(idx, provider_model)
                if idx != 0:
                    logger.info("Fallback provider #{} ({}) succeeded", idx, provider_model)
                return response

            last_response = response

            if not self._is_quota_exhaustion(response.content):
                return response

            self._mark_provider_exhausted(idx, provider_model)

            if pos + 1 < len(order):
                next_idx = order[pos + 1]
                next_provider, next_model = self._providers[next_idx]
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

        order = self._provider_walk_order()

        for pos, idx in enumerate(order):
            provider, provider_model = self._providers[idx]
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
                self._clear_provider_cooldown_if_recovered(idx, provider_model)
                if idx != 0:
                    logger.info("Fallback provider #{} ({}) succeeded (streaming)", idx, provider_model)
                return response

            last_response = response

            if delivered_any:
                # Primary already streamed content to the user — falling back to
                # a secondary provider would concatenate a second response on top.
                return response

            if not self._is_quota_exhaustion(response.content):
                return response

            self._mark_provider_exhausted(idx, provider_model)

            if pos + 1 < len(order):
                next_idx = order[pos + 1]
                next_provider, next_model = self._providers[next_idx]
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
