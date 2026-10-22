"""Credential-isolated provider leaves sharing one key pool coordinator."""
from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any, Literal

from nanobot.events import RetryStatusEvent
from nanobot.providers.base import (
    GenerationSettings,
    LLMCallObserver,
    LLMProvider,
    LLMResponse,
    ProviderCallContext,
    ProviderConversationState,
    RetryEventCallback,
    RetryStatusCallback,
)
from nanobot.providers.key_pool import KeyPoolCoordinator

_OWNER = '_nanobot_credential_owner'

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



def _rotation_cooldown(response: LLMResponse) -> float | None:
    if response.finish_reason != 'error':
        return None
    text = (response.content or '').lower()
    status = response.error_status_code
    if status not in (429, 503) and ('timeout' in text or 'timed out' in text or response.error_kind == 'timeout'):
        return None
    eligible = status in (429, 503) or any(marker in text for marker in (
        'rate limit', 'rate_limit', 'quota', 'resource_exhausted', '429', 'unavailable', 'high demand', '503'))
    if not eligible:
        return None
    if status == 503 or any(marker in text for marker in ('high demand', 'unavailable', 'timeout', 'timed out')):
        return 15.0
    quota = any(marker in text for marker in _QUOTA_EXHAUSTION_TEXT_MARKERS)
    definitive = 'all configured api keys' in text or 'all api keys exhausted' in text
    transient = any(marker in text for marker in _RATE_LIMIT_TEXT_MARKERS)
    if (quota and (definitive or not transient)) or any(
        marker in text for marker in ('current quota', 'plan and billing', 'resource_exhausted')
    ):
        return 1800.0
    return 60.0


class PooledProvider(LLMProvider):
    """Keep each credential/client immutable; state remains owned by its leaf."""

    def __init__(self, leaves: list[LLMProvider], keys: list[str], rate_limit: int = 0) -> None:
        if not leaves or len(leaves) != len(keys):
            raise ValueError('Each key pool slot requires exactly one provider leaf')
        self._leaves: tuple[LLMProvider, ...] = tuple(leaves)
        generation = leaves[0].generation
        super().__init__(api_key=leaves[0].api_key, api_base=leaves[0].api_base,
                         provider_name=leaves[0].provider_name)
        self.generation = generation
        self._owners: tuple[str, ...] = tuple(secrets.token_hex(32) for _ in keys)
        self._pool: KeyPoolCoordinator = KeyPoolCoordinator(len(leaves), rate_limit)

    @property
    def generation(self) -> GenerationSettings:
        return self._leaves[0].generation

    @generation.setter
    def generation(self, value: GenerationSettings) -> None:
        for leaf in self._leaves:
            leaf.generation = value

    def get_default_model(self) -> str:
        return self._leaves[0].get_default_model()

    def set_llm_call_observer(self, observer: LLMCallObserver | None) -> None:
        super().set_llm_call_observer(observer)
        for leaf in self._leaves:
            leaf.set_llm_call_observer(observer)

    def supports_native_compaction(self, model: str | None = None) -> bool:
        return any(leaf.supports_native_compaction(model) for leaf in self._leaves)

    def _leaf_state(self, slot: int, state: ProviderConversationState | None,
                    model: str | None) -> ProviderConversationState | None:
        if state is None or state.payload.get(_OWNER) != self._owners[slot]:
            return None
        clean = replace(state, payload=deepcopy({key: value for key, value in state.payload.items() if key != _OWNER}),
                        pending_messages=deepcopy(state.pending_messages))
        return clean if self._leaves[slot].can_resume_conversation_state(clean, model) else None

    def can_resume_conversation_state(self, state: ProviderConversationState, model: str | None = None) -> bool:
        return any(self._leaf_state(slot, state, model) is not None for slot in range(len(self._leaves)))

    def _tag_response(self, slot: int, response: LLMResponse) -> LLMResponse:
        """Keep pool ownership while removing leaf-bound continuation metadata."""
        states: dict[str, ProviderConversationState] = {}
        for field in ('provider_state', 'provider_compaction_state'):
            state = getattr(response, field)
            if state is not None:
                states[field] = replace(
                    state,
                    payload={**deepcopy(state.payload), _OWNER: self._owners[slot]},
                    pending_messages=deepcopy(state.pending_messages),
                    conversation_binding=None,
                    transport_receipt=None,
                    continuation_metadata_present=False,
                )
        return replace(response, **states) if states else response

    async def _attempt(self, kwargs: dict[str, Any], *, stream: bool, observed: bool,
                       stopped: list[bool | Literal['exhausted']],
                       attempts: list[int] | None = None) -> LLMResponse:
        order = self._pool.request_order()
        call_kwargs = dict(kwargs)
        delivered = False
        for name in ('on_content_delta', 'on_thinking_delta', 'on_tool_call_delta'):
            callback: Callable[..., Awaitable[None]] | None = call_kwargs.get(name)
            if callback is not None:
                async def track(*args: Any, _callback: Callable[..., Awaitable[None]] = callback,
                                **kw: Any) -> None:
                    nonlocal delivered
                    delivered = True
                    await _callback(*args, **kw)
                call_kwargs[name] = track
        slot: int | None = None
        response: LLMResponse | None = None
        for slot in order:
            leaf = self._leaves[slot]
            local = dict(call_kwargs)
            context = local.get('provider_context')
            if isinstance(context, ProviderCallContext):
                local['provider_context'] = replace(
                    context, conversation_state=self._leaf_state(slot, context.conversation_state, local.get('model')),
                    context_window_tokens=context.context_window_tokens if leaf.supports_native_compaction(local.get('model')) else None,
                )
            await self._pool.admit()
            if attempts is not None:
                attempts[0] += 1
            if observed:
                response = await (leaf._safe_chat_stream(**local) if stream else leaf._safe_chat(**local))
            else:
                contextual = isinstance(local.get('provider_context'), ProviderCallContext)
                call = (leaf.chat_stream_with_context if contextual else leaf.chat_stream) if stream else (
                    leaf.chat_with_context if contextual else leaf.chat)
                response = await call(**local)
            response = self._tag_response(slot, response)
            if response.finish_reason != 'error':
                self._pool.succeeded(slot)
                return response
            cooldown = _rotation_cooldown(response)
            if delivered:
                stopped[0] = True
                return response
            if cooldown is None:
                return response
            self._pool.failed(slot, cooldown)
        stopped[0] = "exhausted"
        # Nonempty leaves create matching finite cooldowns; request_order's
        # ready/cooling partition therefore visits at least one credential slot.
        assert slot is not None and response is not None
        return self._tag_response(slot, response)

    async def chat_once(self, **kwargs: Any) -> LLMResponse:
        """One paced credential attempt; do not rotate and repeat a summary."""
        slot = self._pool.request_order()[0]
        await self._pool.admit()
        response = await self._leaves[slot].chat_once(**kwargs)
        if response.finish_reason != 'error':
            self._pool.succeeded(slot)
        else:
            cooldown = _rotation_cooldown(response)
            if cooldown is not None:
                self._pool.failed(slot, cooldown)
        return self._tag_response(slot, response)

    @staticmethod
    def _bind_positional_args(
        args: tuple[object, ...], kwargs: dict[str, Any], *, stream: bool,
    ) -> None:
        # Match the base call order without materializing omitted defaults.
        names = ('tools', 'model', 'max_tokens', 'temperature', 'reasoning_effort', 'tool_choice')
        if stream:
            names += ('on_content_delta', 'on_thinking_delta', 'on_tool_call_delta')
        method = 'chat_stream' if stream else 'chat'
        if len(args) > len(names):
            raise TypeError(f'{method}() accepts at most {len(names) + 1} positional arguments')
        for name in names[:len(args)]:
            if name in kwargs:
                raise TypeError(f"{method}() got multiple values for argument '{name}'")
        kwargs.update(zip(names, args))

    async def chat(
        self, messages: list[dict[str, Any]], *args: object, **kwargs: Any,
    ) -> LLMResponse:
        self._bind_positional_args(args, kwargs, stream=False)
        kwargs["messages"] = messages
        return await self._attempt(kwargs, stream=False, observed=False, stopped=[False])

    async def chat_with_context(self, *, provider_context: ProviderCallContext,
                                **kwargs: Any) -> LLMResponse:
        return await self.chat(provider_context=provider_context, **kwargs)

    async def chat_stream(
        self, messages: list[dict[str, Any]], *args: object, **kwargs: Any,
    ) -> LLMResponse:
        self._bind_positional_args(args, kwargs, stream=True)
        kwargs["messages"] = messages
        return await self._attempt(kwargs, stream=True, observed=False, stopped=[False])

    async def chat_stream_with_context(self, *, provider_context: ProviderCallContext,
                                       **kwargs: Any) -> LLMResponse:
        return await self.chat_stream(provider_context=provider_context, **kwargs)

    async def _run_chat_with_retry(self, kw: dict[str, Any], original_messages: list[dict[str, Any]],
                                   *, stream: bool, retry_mode: str,
                                   on_retry_wait: RetryEventCallback | None,
                                   on_retry_exhausted: RetryEventCallback | None,
                                   on_retry_status: RetryStatusCallback | None,
                                   should_retry_guard: Callable[[], bool] | None = None,
                                   on_stream_recover: Callable[[], Awaitable[None]] | None = None) -> LLMResponse:
        stopped: list[bool | Literal['exhausted']] = [False]
        attempts: list[int] = [0]
        async def call(**kwargs: Any) -> LLMResponse:
            return await self._attempt(kwargs, stream=stream, observed=True, stopped=stopped, attempts=attempts)
        async def status(event: RetryStatusEvent) -> None:
            if stopped[0] != "exhausted" and on_retry_status:
                await on_retry_status(event)

        result = await self._run_with_retry(
            call, kw, original_messages, retry_mode=retry_mode, on_retry_wait=on_retry_wait,
            on_retry_exhausted=on_retry_exhausted, on_retry_status=status,
            should_retry_guard=lambda: not stopped[0] and (should_retry_guard is None or should_retry_guard()),
            on_stream_recover=on_stream_recover,
        )
        if stopped[0] == "exhausted":
            if on_retry_exhausted:
                await on_retry_exhausted("All configured provider credentials failed.")
            if on_retry_status:
                await on_retry_status(RetryStatusEvent(
                    state="exhausted", attempt=attempts[0], max_attempts=attempts[0],
                    error_kind=self.public_error_kind(result)))
        return result
