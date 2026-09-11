"""Staging fallback policy on the upstream provider call surfaces."""
from __future__ import annotations

import secrets
import time
from copy import deepcopy
from dataclasses import replace

from nanobot.providers.base import LLMResponse, ProviderCallContext
from nanobot.providers.fallback_provider import FallbackProvider

_OWNER = '_nanobot_legacy_candidate'


class LegacyFallbackProvider(FallbackProvider):
    """One ordered traversal per request; only quota/rate failures advance."""

    def __init__(self, primary, fallback_presets, provider_factory, **kwargs):
        if not fallback_presets:
            raise ValueError("Legacy fallback requires at least one fallback candidate")
        generation = primary.generation
        self._candidates = (primary, *(provider_factory(preset) for preset in fallback_presets))
        self._owners = tuple(secrets.token_hex(32) for _ in self._candidates)
        self._cooldowns = [0.0] * len(self._candidates)
        super().__init__(primary, fallback_presets, provider_factory, **kwargs)
        self.generation = generation

    @property
    def generation(self):
        return self._candidates[0].generation

    @generation.setter
    def generation(self, value):
        for candidate in self._candidates:
            candidate.generation = value

    def set_llm_call_observer(self, observer):
        super().set_llm_call_observer(observer)
        for candidate in self._candidates:
            candidate.set_llm_call_observer(observer)

    def _model(self, slot, override=None):
        return (override or self._primary.get_default_model()) if slot == 0 else self._fallback_presets[slot - 1].model

    def _state(self, slot, state, model):
        if state is None or state.payload.get(_OWNER) != self._owners[slot]:
            return None
        clean = replace(state, payload=deepcopy({k: v for k, v in state.payload.items() if k != _OWNER}),
                        pending_messages=deepcopy(state.pending_messages))
        return clean if self._candidates[slot].can_resume_conversation_state(clean, model) else None

    def can_resume_conversation_state(self, state, model=None):
        return any(self._state(slot, state, self._model(slot, model)) is not None
                   for slot in range(len(self._candidates)))

    def supports_native_compaction(self, model=None):
        return any(candidate.supports_native_compaction(self._model(slot, model))
                   for slot, candidate in enumerate(self._candidates))

    def _primary_call_context(self, provider_context, model):
        # Capacity and state are resolved only after the destination is selected.
        return provider_context

    def _kwargs(self, slot, kwargs):
        local = dict(kwargs)
        local['model'] = self._model(slot, kwargs.get('model'))
        context = local.get('provider_context')
        if isinstance(context, ProviderCallContext):
            capacity = (self._primary_context_window_tokens or context.context_window_tokens) if slot == 0 else self._fallback_presets[slot - 1].context_window_tokens
            request_limits = context.request_limits
            if request_limits is not None and capacity is not None:
                request_limits = request_limits.for_model_window(capacity)
            local['provider_context'] = replace(
                context, conversation_state=self._state(slot, context.conversation_state, local['model']),
                request_limits=request_limits,
                context_window_tokens=capacity if self._candidates[slot].supports_native_compaction(local['model']) else None)
        return local

    def _tag(self, slot, response):
        states = {}
        for field in ('provider_state', 'provider_compaction_state'):
            state = getattr(response, field)
            if state is not None:
                states[field] = replace(state, payload={**deepcopy(state.payload), _OWNER: self._owners[slot]},
                                        pending_messages=deepcopy(state.pending_messages))
        return replace(response, **states) if states else response

    async def chat(self, **kwargs):
        return self._tag(0, await self._primary.chat(**self._kwargs(0, kwargs)))

    async def chat_with_context(self, *, provider_context, **kwargs):
        return self._tag(0, await self._primary.chat_with_context(
            **self._kwargs(0, dict(kwargs, provider_context=provider_context))))

    async def chat_stream(self, **kwargs):
        kwargs.pop('on_stream_recover', None)
        return self._tag(0, await self._primary.chat_stream(**self._kwargs(0, kwargs)))

    async def chat_stream_with_context(self, *, provider_context, **kwargs):
        kwargs.pop('on_stream_recover', None)
        return self._tag(0, await self._primary.chat_stream_with_context(
            **self._kwargs(0, dict(kwargs, provider_context=provider_context))))

    async def _retry_with_fallback(self, *args, **kwargs):
        # Staging retries individual candidates, never the whole chain.
        kwargs['retry_mode'] = 'standard'
        return await super()._retry_with_fallback(*args, **kwargs)

    _NON_RETRYABLE_429_TOKENS = frozenset({
        "insufficient_quota",
        "quota_exceeded",
        "quota_exhausted",
        "billing_hard_limit_reached",
        "insufficient_balance",
        "credit_balance_too_low",
        "billing_not_active",
        "payment_required",
    })
    # Semantic tokens that indicate a transient rate limit (should retry).
    _RETRYABLE_429_TOKENS = frozenset({
        "rate_limit_exceeded",
        "rate_limit_error",
        "too_many_requests",
        "request_limit_exceeded",
        "overloaded_error",
    })
    # Text markers for fallback classification when no structured tokens available.
    _QUOTA_EXHAUSTION_TEXT_MARKERS = (
        "insufficient_quota",
        "insufficient quota",
        "out of quota",
        "quota exceeded",
        "quota exhausted",
        "billing hard limit",
        "billing_hard_limit_reached",
        "billing limit",
        "billing not active",
        "insufficient balance",
        "credit balance too low",
        "payment required",
        "out of credits",
        "exceeded your current quota",
        "all api keys exhausted",
        "all configured api keys were rate-limited or out of quota",
    )
    _RATE_LIMIT_TEXT_MARKERS = (
        "rate limit",
        "rate_limit",
        "too many requests",
        "retry after",
        "try again in",
        "temporarily unavailable",
        "overloaded",
        "concurrency limit",
    )

    @classmethod
    def _is_quota_exhaustion(cls, content: str | None) -> bool:
        """True when error indicates permanent quota exhaustion (not transient rate limit)."""
        err = (content or "").lower()
        # Check for quota exhaustion markers first.
        has_quota_marker = any(marker in err for marker in cls._QUOTA_EXHAUSTION_TEXT_MARKERS)
        if not has_quota_marker:
            return False
        # "all configured api keys" is a definitive exhaustion signal even
        # though the message text also contains "rate-limited".
        if "all configured api keys" in err or "all api keys exhausted" in err:
            return True
        # For other messages, if rate limit markers are present, it's transient.
        if any(marker in err for marker in cls._RATE_LIMIT_TEXT_MARKERS):
            return False
        return True

    @classmethod
    def _is_quota_exhaustion_response(cls, response: LLMResponse) -> bool:
        """Structured check: is this response a permanent quota exhaustion?"""
        tokens = {t for t in (response.error_type, response.error_code) if t}
        if tokens:
            return any(t in cls._NON_RETRYABLE_429_TOKENS for t in tokens)
        return cls._is_quota_exhaustion(response.content)

    @classmethod
    def _is_rate_limit_response(cls, response: LLMResponse) -> bool:
        """True when the response indicates a transient rate limit (not quota exhaustion)."""
        if response.error_status_code == 429 and not cls._is_quota_exhaustion_response(response):
            return True
        content = (response.content or "").lower()
        has_rate_limit = any(marker in content for marker in cls._RATE_LIMIT_TEXT_MARKERS)
        has_quota = any(marker in content for marker in cls._QUOTA_EXHAUSTION_TEXT_MARKERS)
        return has_rate_limit and not has_quota

    @classmethod
    def _eligible(cls, response):
        return cls._is_quota_exhaustion_response(response) or cls._is_rate_limit_response(response)

    async def _try_with_fallback(self, call, kwargs, has_streamed,
                                 on_stream_recover=None, on_fallback_attempt=None):
        now = time.monotonic()
        ready = [slot for slot, expiry in enumerate(self._cooldowns) if expiry <= now]
        cooling = sorted((slot for slot, expiry in enumerate(self._cooldowns) if expiry > now),
                         key=lambda slot: (self._cooldowns[slot], slot))
        for position, slot in enumerate((*ready, *cooling)):
            if position and on_fallback_attempt:
                await on_fallback_attempt()
            response, _ = await self._call_provider(call, self._candidates[slot], self._kwargs(slot, kwargs))
            response = self._tag(slot, response)
            if response.finish_reason != 'error':
                self._cooldowns[slot] = 0.0
                if slot:
                    await self._notify_fallback_model(self._model(slot))
                return response
            if (has_streamed is not None and has_streamed[0]) or not self._eligible(response):
                return response
            self._cooldowns[slot] = max(self._cooldowns[slot], time.monotonic() + 300.0)
        return response
