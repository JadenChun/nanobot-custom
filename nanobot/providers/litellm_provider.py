"""LiteLLM provider implementation for multi-provider support."""

import asyncio
import collections
import json
import json_repair
import logging
import os
import re
import time
from typing import Any

import litellm
from litellm import acompletion

logger = logging.getLogger(__name__)

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.registry import find_by_model, find_gateway


# Standard OpenAI chat-completion message keys; extras (e.g. reasoning_content) are stripped for strict providers.
_ALLOWED_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name"})


class LiteLLMProvider(LLMProvider):
    """
    LLM provider using LiteLLM for multi-provider support.
    
    Supports OpenRouter, Anthropic, OpenAI, Gemini, MiniMax, and many other providers through
    a unified interface.  Provider-specific logic is driven by the registry
    (see providers/registry.py) — no if-elif chains needed here.
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        api_base: str | None = None,
        default_model: str = "anthropic/claude-opus-4-5",
        extra_headers: dict[str, str] | None = None,
        provider_name: str | None = None,
        rate_limit: int = 0,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}

        # Round-robin key rotation state
        self._api_keys: list[str] = api_keys if api_keys else ([api_key] if api_key else [])
        self._key_index: int = 0
        self.api_key = self._api_keys[0] if self._api_keys else None

        # Proactive rate limiting (sliding window, requests per minute)
        self._rate_limit = rate_limit  # 0 = unlimited
        self._request_timestamps: collections.deque[float] = collections.deque()

        # Detect gateway / local deployment.
        # provider_name (from config key) is the primary signal;
        # api_key / api_base are fallback for auto-detection.
        self._gateway = find_gateway(provider_name, self.api_key, api_base)

        # Configure environment variables (uses first key)
        if self.api_key:
            self._setup_env(self.api_key, api_base, default_model)

        if api_base:
            litellm.api_base = api_base

        # Disable LiteLLM logging noise
        litellm.suppress_debug_info = True
        # Drop unsupported parameters for providers (e.g., gpt-5 rejects some params)
        litellm.drop_params = True
    
    def _next_key(self) -> str | None:
        """Advance to the next API key in round-robin order. Returns the new key, or None if only one key."""
        if len(self._api_keys) <= 1:
            return None
        self._key_index = (self._key_index + 1) % len(self._api_keys)
        self.api_key = self._api_keys[self._key_index]
        return self.api_key

    # Rate-limit retry configuration
    RATE_LIMIT_MAX_RETRIES = 5
    RATE_LIMIT_BASE_DELAY = 10.0   # seconds
    RATE_LIMIT_MAX_DELAY = 120.0   # seconds

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Return True if the exception indicates a rate-limit or quota error."""
        if isinstance(exc, litellm.RateLimitError):
            return True
        status = getattr(exc, "status_code", None)
        if status == 429:
            return True
        msg = str(exc).lower()
        return any(kw in msg for kw in ("rate limit", "rate_limit", "quota", "resource_exhausted", "429"))

    @staticmethod
    def _parse_retry_delay(exc: Exception) -> float | None:
        """Extract a suggested retry delay (in seconds) from the error message, if present."""
        msg = str(exc)
        # Match patterns like "retry in 53.676334989s" or "retryDelay": "53s"
        match = re.search(r'retry\s*(?:in|Delay["\s:]*)\s*"?(\d+(?:\.\d+)?)\s*s', msg, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None

    async def _wait_for_rate_limit(self) -> None:
        """Sleep if necessary to stay within the configured rate limit (requests/minute)."""
        if self._rate_limit <= 0:
            return

        now = time.monotonic()
        window = 60.0  # 1 minute sliding window

        # Evict timestamps older than the window
        while self._request_timestamps and self._request_timestamps[0] <= now - window:
            self._request_timestamps.popleft()

        if len(self._request_timestamps) >= self._rate_limit:
            # Must wait until the oldest request in the window expires
            oldest = self._request_timestamps[0]
            delay = oldest + window - now + 1.0  # +1s safety buffer
            if delay > 0:
                logger.info(
                    "Rate limit: %d/%d requests in window. Waiting %.1fs before next request…",
                    len(self._request_timestamps),
                    self._rate_limit,
                    delay,
                )
                await asyncio.sleep(delay)

        self._request_timestamps.append(time.monotonic())

    def _setup_env(self, api_key: str, api_base: str | None, model: str) -> None:
        """Set environment variables based on detected provider."""
        spec = self._gateway or find_by_model(model)
        if not spec:
            return
        if not spec.env_key:
            # OAuth/provider-only specs (for example: openai_codex)
            return

        # Gateway/local overrides existing env; standard provider doesn't
        if self._gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)

        # Resolve env_extras placeholders:
        #   {api_key}  → user's API key
        #   {api_base} → user's api_base, falling back to spec.default_api_base
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key)
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)
    
    def _resolve_model(self, model: str) -> str:
        """Resolve model name by applying provider/gateway prefixes."""
        if self._gateway:
            # Gateway mode: apply gateway prefix, skip provider-specific prefixes
            prefix = self._gateway.litellm_prefix
            if self._gateway.strip_model_prefix:
                model = model.split("/")[-1]
            if prefix and not model.startswith(f"{prefix}/"):
                model = f"{prefix}/{model}"
            return model
        
        # Standard mode: auto-prefix for known providers
        spec = find_by_model(model)
        if spec and spec.litellm_prefix:
            model = self._canonicalize_explicit_prefix(model, spec.name, spec.litellm_prefix)
            if not any(model.startswith(s) for s in spec.skip_prefixes):
                model = f"{spec.litellm_prefix}/{model}"

        return model

    @staticmethod
    def _canonicalize_explicit_prefix(model: str, spec_name: str, canonical_prefix: str) -> str:
        """Normalize explicit provider prefixes like `github-copilot/...`."""
        if "/" not in model:
            return model
        prefix, remainder = model.split("/", 1)
        if prefix.lower().replace("-", "_") != spec_name:
            return model
        return f"{canonical_prefix}/{remainder}"
    
    def _supports_cache_control(self, model: str) -> bool:
        """Return True when the provider supports cache_control on content blocks."""
        if self._gateway is not None:
            return self._gateway.supports_prompt_caching
        spec = find_by_model(model)
        return spec is not None and spec.supports_prompt_caching

    def _apply_cache_control(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Return copies of messages and tools with cache_control injected."""
        new_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg["content"]
                if isinstance(content, str):
                    new_content = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
                else:
                    new_content = list(content)
                    new_content[-1] = {**new_content[-1], "cache_control": {"type": "ephemeral"}}
                new_messages.append({**msg, "content": new_content})
            else:
                new_messages.append(msg)

        new_tools = tools
        if tools:
            new_tools = list(tools)
            new_tools[-1] = {**new_tools[-1], "cache_control": {"type": "ephemeral"}}

        return new_messages, new_tools

    def _apply_model_overrides(self, model: str, kwargs: dict[str, Any]) -> None:
        """Apply model-specific parameter overrides from the registry."""
        model_lower = model.lower()
        spec = find_by_model(model)
        if spec:
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    return
    
    @staticmethod
    def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip non-standard keys and ensure assistant messages have a content key."""
        sanitized = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if k in _ALLOWED_MSG_KEYS}
            # Strict providers require "content" even when assistant only has tool_calls.
            # Also fix None content — some providers reject assistant messages
            # that have neither meaningful content nor tool_calls.
            if clean.get("role") == "assistant":
                if "content" not in clean or clean["content"] is None:
                    clean["content"] = ""
            sanitized.append(clean)
        return sanitized

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a chat completion request via LiteLLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions in OpenAI format.
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4-5').
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        original_model = model or self.default_model
        model = self._resolve_model(original_model)

        if self._supports_cache_control(original_model):
            messages, tools = self._apply_cache_control(messages, tools)

        # Clamp max_tokens to at least 1 — negative or zero values cause
        # LiteLLM to reject the request with "max_tokens must be at least 1".
        max_tokens = max(1, max_tokens)
        
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._sanitize_messages(self._sanitize_empty_content(messages)),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Apply model-specific overrides (e.g. kimi-k2.5 temperature)
        self._apply_model_overrides(model, kwargs)
        
        # Pass api_key directly — more reliable than env vars alone
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        # Pass api_base for custom endpoints
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        # Pass extra headers (e.g. APP-Code for AiHubMix)
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        # Retry loop: rotate API keys on rate-limit / quota errors,
        # then fall back to exponential backoff with retry.
        keys_tried = 0
        total_keys = max(len(self._api_keys), 1)
        last_error: Exception | None = None
        rate_limit_retries = 0

        while True:
            kwargs["api_key"] = self.api_key
            try:
                await self._wait_for_rate_limit()
                response = await acompletion(**kwargs)
                return self._parse_response(response)
            except Exception as e:
                last_error = e
                if not self._is_rate_limit_error(e):
                    return LLMResponse(
                        content=f"Error calling LLM: {str(e)}",
                        finish_reason="error",
                    )

                # Try rotating to next API key first
                if keys_tried < total_keys - 1 and self._next_key() is not None:
                    keys_tried += 1
                    continue

                # All keys exhausted (or single key) — use backoff retry
                rate_limit_retries += 1
                if rate_limit_retries > self.RATE_LIMIT_MAX_RETRIES:
                    return LLMResponse(
                        content=(
                            f"Error calling LLM: Rate limit exceeded after "
                            f"{self.RATE_LIMIT_MAX_RETRIES} retries. Last error: {last_error}"
                        ),
                        finish_reason="error",
                    )

                # Use server-suggested delay if available, otherwise exponential backoff
                suggested = self._parse_retry_delay(e)
                if suggested is not None:
                    delay = min(suggested + 2.0, self.RATE_LIMIT_MAX_DELAY)
                else:
                    delay = min(
                        self.RATE_LIMIT_BASE_DELAY * (2 ** (rate_limit_retries - 1)),
                        self.RATE_LIMIT_MAX_DELAY,
                    )

                logger.warning(
                    "Rate limited (attempt %d/%d). Retrying in %.1fs…",
                    rate_limit_retries,
                    self.RATE_LIMIT_MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)

                # Reset key rotation for the next round
                keys_tried = 0
    
    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse LiteLLM response into our standard format."""
        if not hasattr(response, "choices") or not response.choices:
            raise RuntimeError("LLM returned no choices in response.")
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json_repair.loads(args)
                
                tool_calls.append(ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        
        reasoning_content = getattr(message, "reasoning_content", None) or None
        
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )
    
    def get_default_model(self) -> str:
        """Get the default model."""
        return self.default_model
