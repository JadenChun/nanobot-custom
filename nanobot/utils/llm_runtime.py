"""Immutable execution settings for one LLM turn."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from nanobot.providers.base import GenerationSettings, LLMProvider
from nanobot.providers.capabilities import ModelCapabilities, default_model_capabilities

if TYPE_CHECKING:
    from nanobot.providers.factory import ProviderSnapshot


@dataclass(frozen=True, slots=True)
class LLMRuntime:
    """One captured provider/model configuration used for an entire execution.

    The provider itself is stateful, but all mutable selection and generation
    values are copied into this frozen value.  Consumers must use these fields
    instead of consulting ``provider.generation`` after admission.
    """

    provider: LLMProvider
    model: str
    generation: GenerationSettings
    context_window_tokens: int
    model_preset: str | None = None
    snapshot_signature: tuple[object, ...] | None = None
    capabilities: ModelCapabilities | None = None

    @classmethod
    def capture(
        cls,
        provider: LLMProvider,
        model: str,
        *,
        context_window_tokens: int,
        model_preset: str | None = None,
        snapshot_signature: tuple[object, ...] | None = None,
        capabilities: ModelCapabilities | None = None,
    ) -> LLMRuntime:
        """Capture provider defaults without retaining mutable generation state."""
        defaults = GenerationSettings()
        generation = getattr(provider, "generation", defaults)
        max_tokens = getattr(generation, "max_tokens", defaults.max_tokens)
        return cls(
            provider=provider,
            model=model,
            generation=GenerationSettings(
                temperature=getattr(generation, "temperature", defaults.temperature),
                max_tokens=max_tokens,
                reasoning_effort=getattr(
                    generation,
                    "reasoning_effort",
                    defaults.reasoning_effort,
                ),
            ),
            context_window_tokens=context_window_tokens,
            model_preset=model_preset,
            snapshot_signature=snapshot_signature,
            capabilities=(
                capabilities
                if capabilities is not None
                else _resolve_provider_capabilities(
                    provider,
                    model,
                    context_window_tokens=context_window_tokens,
                    max_output_tokens=max_tokens,
                )
            ),
        )

    @property
    def resolved_capabilities(self) -> ModelCapabilities:
        """Return the immutable capability snapshot for this run."""
        if self.capabilities is not None:
            return self.capabilities
        return _resolve_provider_capabilities(
            self.provider,
            self.model,
            context_window_tokens=self.context_window_tokens,
            max_output_tokens=self.generation.max_tokens,
        )

    def with_generation_overrides(
        self,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMRuntime:
        """Return a derived runtime for explicit per-run generation overrides."""
        generation = self.generation
        return replace(
            self,
            generation=GenerationSettings(
                temperature=(
                    generation.temperature if temperature is None else temperature
                ),
                max_tokens=generation.max_tokens if max_tokens is None else max_tokens,
                reasoning_effort=(
                    generation.reasoning_effort
                    if reasoning_effort is None
                    else reasoning_effort
                ),
            ),
        )


def _resolve_provider_capabilities(
    provider: LLMProvider,
    model: str,
    *,
    context_window_tokens: int,
    max_output_tokens: int | None,
) -> ModelCapabilities:
    resolver = getattr(provider, "model_capabilities", None)
    if callable(resolver):
        try:
            candidate = resolver(
                model,
                context_window_tokens=context_window_tokens,
                max_output_tokens=max_output_tokens,
            )
        except Exception:
            candidate = None
        if isinstance(candidate, ModelCapabilities):
            return candidate
    return default_model_capabilities(
        getattr(provider, "provider_name", "unknown"),
        model,
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
    )


def runtime_from_provider_snapshot(
    snapshot: ProviderSnapshot,
) -> LLMRuntime:
    """Convert a provider factory snapshot into the canonical runtime value."""
    if snapshot.generation is not None:
        return LLMRuntime(
            provider=snapshot.provider,
            model=snapshot.model,
            generation=snapshot.generation,
            context_window_tokens=snapshot.context_window_tokens,
            model_preset=snapshot.model_preset,
            snapshot_signature=snapshot.signature,
            capabilities=snapshot.capabilities,
        )
    return LLMRuntime.capture(
        snapshot.provider,
        snapshot.model,
        context_window_tokens=snapshot.context_window_tokens,
        model_preset=snapshot.model_preset,
        snapshot_signature=snapshot.signature,
        capabilities=snapshot.capabilities,
    )
