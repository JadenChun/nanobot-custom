"""Immutable model capability facts resolved for one provider/model route."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from nanobot.providers.image_pricing import ImageRequestPricing, image_request_pricing


class _ModelMetadata(Protocol):
    @property
    def max_output_tokens(self) -> int | None: ...

    @property
    def input_modalities(self) -> tuple[str, ...] | None: ...

    @property
    def output_modalities(self) -> tuple[str, ...] | None: ...


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Capabilities consumed by request admission, accounting, and diagnostics."""

    context_window_tokens: int
    max_output_tokens: int | None
    input_modalities: frozenset[str] | None
    output_modalities: frozenset[str]
    image_pricing: ImageRequestPricing | None
    capability_source: str

    @property
    def accepts_images(self) -> bool | None:
        """Return True/False when declared, or None when the modality is unknown."""
        if self.input_modalities is None:
            return None
        return "image" in self.input_modalities


def default_model_capabilities(
    provider: str,
    model: str,
    *,
    context_window_tokens: int,
    max_output_tokens: int | None = None,
    input_modalities: Iterable[str] | None = None,
    output_modalities: Iterable[str] | None = None,
    capability_source: str = "fallback",
) -> ModelCapabilities:
    """Resolve generic route capabilities from the provider/model identity.

    Provider adapters may replace this with catalog metadata. A missing
    modality declaration remains unknown; it is not negative capability.
    """
    return ModelCapabilities(
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        input_modalities=(
            None if input_modalities is None else frozenset(input_modalities)
        ),
        output_modalities=frozenset(output_modalities or ("text",)),
        image_pricing=image_request_pricing(provider, model),
        capability_source=capability_source,
    )


def model_capabilities_from_metadata(
    provider: str,
    model: str,
    metadata: _ModelMetadata | None,
    *,
    context_window_tokens: int,
    max_output_tokens: int | None = None,
    capability_source: str = "catalog",
) -> ModelCapabilities:
    """Layer curated provider metadata over the generic route defaults."""
    if metadata is None:
        return default_model_capabilities(
            provider,
            model,
            context_window_tokens=context_window_tokens,
            max_output_tokens=max_output_tokens,
        )
    return default_model_capabilities(
        provider,
        model,
        context_window_tokens=context_window_tokens,
        max_output_tokens=(
            metadata.max_output_tokens
            if metadata.max_output_tokens is not None
            else max_output_tokens
        ),
        input_modalities=metadata.input_modalities,
        output_modalities=metadata.output_modalities,
        capability_source=capability_source,
    )


__all__ = [
    "ModelCapabilities",
    "default_model_capabilities",
    "model_capabilities_from_metadata",
]
