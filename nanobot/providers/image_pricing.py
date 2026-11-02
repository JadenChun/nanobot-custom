"""Provider-owned request-image pricing."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from nanobot.providers.provider_policy import ImagePricingMode
from nanobot.utils.media_tokens import (
    MAX_IMAGE_TOKENS,
    estimate_image_tokens,
    unknown_image_tokens,
)

_IMAGE_FRAMING_TOKENS = 32
_OPENAI_IMAGE_BASES = (
    "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1",
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "gpt-5.4-mini",
    "gpt-5.4",
)


@dataclass(frozen=True, slots=True)
class ImageOccurrence:
    """One image occurrence in a model-visible request."""

    width: int | None
    height: int | None
    detail: str | None
    remote: bool
    encoded_bytes: int
    text: str = ""


@dataclass(frozen=True, slots=True)
class ImagePrice:
    """Visual tokens plus any text substituted for one image occurrence."""

    visual_tokens: int
    text: str = ""


class ImageRequestPricing(Protocol):
    """Synchronous route-owned pricing for a sequence of image occurrences."""

    def price(self, images: Sequence[ImageOccurrence]) -> Sequence[ImagePrice]:
        ...


class _FunctionPricing:
    def __init__(self, counter: Callable[[ImageOccurrence], int]) -> None:
        self._counter = counter

    def price(self, images: Sequence[ImageOccurrence]) -> Sequence[ImagePrice]:
        return [ImagePrice(self._counter(image)) for image in images]


class _RejectPricing:
    def __init__(self, message: str) -> None:
        self._message = message

    def price(self, images: Sequence[ImageOccurrence]) -> Sequence[ImagePrice]:
        _ = images
        raise ValueError(self._message)


def image_request_pricing(
    provider: str,
    model: str,
    *,
    mode: ImagePricingMode = "auto",
) -> ImageRequestPricing | None:
    """Return route-owned image pricing for the declared pricing mode.

    ``local_generic`` covers arbitrary local model aliases; ``auto`` keeps
    exact OpenAI/Claude model-family formulas; ``none`` delegates to the
    bounded generic fallback in request measurement.
    """
    if mode == "none":
        return None
    if mode == "local_generic":
        return _FunctionPricing(_local_generic_tokens)
    if _openai_image_model(provider, model) is not None:
        return _FunctionPricing(lambda occurrence: _openai_tokens(provider, model, occurrence))
    if provider in {"openai", "azure_openai", "openai_codex"} and _openai_image_family(
        _selected_model(provider, model)
    ):
        return _RejectPricing("unsupported image model alias")
    if _claude_image_model(provider, model) is not None:
        return _FunctionPricing(
            lambda occurrence: _claude_tokens(provider, model, occurrence)
        )
    return None


def _local_generic_tokens(occurrence: ImageOccurrence) -> int:
    effective_detail = _detail(occurrence.detail)
    if effective_detail not in {"low", "high", "auto"}:
        raise ValueError("unsupported local image detail")
    if occurrence.remote:
        return MAX_IMAGE_TOKENS
    if occurrence.width is not None and occurrence.height is not None:
        if occurrence.width > 0 and occurrence.height > 0:
            return estimate_image_tokens(occurrence.width, occurrence.height)
    return unknown_image_tokens(occurrence.encoded_bytes)


def _openai_tokens(provider: str, model: str, occurrence: ImageOccurrence) -> int:
    openai_model = _openai_image_model(provider, model)
    if openai_model is None:
        raise ValueError("no OpenAI image accounting policy for selected model")
    effective_detail = _detail(occurrence.detail)
    if openai_model in {"gpt-4o", "gpt-4o-mini"}:
        base, tile = (85, 170) if openai_model == "gpt-4o" else (2_833, 5_667)
        if effective_detail == "low":
            return base
        if effective_detail not in {"high", "auto"}:
            raise ValueError("unsupported image detail for selected model")
        if occurrence.width is None or occurrence.height is None or occurrence.remote:
            return base + 8 * tile
        width, height = occurrence.width, occurrence.height
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions are invalid")
        scale = min(1.0, 2_048 / max(width, height))
        scaled_width = max(1, math.ceil(width * scale))
        scaled_height = max(1, math.ceil(height * scale))
        short_edge = min(scaled_width, scaled_height)
        scale = min(1.0, 768 / short_edge)
        scaled_width = max(1, math.ceil(scaled_width * scale))
        scaled_height = max(1, math.ceil(scaled_height * scale))
        tile_count = math.ceil(scaled_width / 512) * math.ceil(scaled_height / 512)
        return base + tile_count * tile

    if openai_model.startswith(("gpt-5.6-", "gpt-5.5")):
        if effective_detail == "low":
            return 308
        if effective_detail == "high":
            return 3_000
        if effective_detail in {"auto", "original"}:
            return 36_000 if openai_model.startswith("gpt-5.6-") else 12_000
        raise ValueError("unsupported image detail for selected model")
    if openai_model in {"gpt-5.4", "gpt-5.4-mini"}:
        if effective_detail == "low":
            return 7_373
        if effective_detail in {"high", "auto"}:
            return 3_000
        if effective_detail == "original":
            return 12_000
        raise ValueError("unsupported image detail for selected model")
    if openai_model == "gpt-4.1-mini":
        if effective_detail in {"low", "high", "auto"}:
            return 9_954
        raise ValueError("unsupported image detail for selected model")
    if openai_model == "gpt-4.1-nano":
        if effective_detail in {"low", "high", "auto"}:
            return 73_800
        raise ValueError("unsupported image detail for selected model")
    if openai_model == "gpt-4.1":
        base, tile = 85, 170
        if effective_detail == "low":
            return base
        if effective_detail not in {"high", "auto"}:
            raise ValueError("unsupported image detail for selected model")
        if occurrence.width is None or occurrence.height is None or occurrence.remote:
            return base + 8 * tile
        width, height = occurrence.width, occurrence.height
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions are invalid")
        scale = min(1.0, 2_048 / max(width, height))
        width = max(1, math.ceil(width * scale))
        height = max(1, math.ceil(height * scale))
        scale = min(1.0, 768 / min(width, height))
        width = max(1, math.ceil(width * scale))
        height = max(1, math.ceil(height * scale))
        return base + math.ceil(width / 512) * math.ceil(height / 512) * tile
    raise ValueError("unsupported OpenAI image model")


def _claude_tokens(provider: str, model: str, occurrence: ImageOccurrence) -> int:
    normalized = _claude_image_model(provider, model)
    if normalized is None:
        raise ValueError("no Claude image accounting policy for selected model")
    effective_detail = _detail(occurrence.detail)
    if normalized == "claude-opus-4-7":
        if effective_detail in {"auto", "high"}:
            return 4_784 + _IMAGE_FRAMING_TOKENS
        if effective_detail == "standard":
            return 1_568 + _IMAGE_FRAMING_TOKENS
        raise ValueError("unsupported image detail for selected Claude model")
    if effective_detail in {"auto", "standard"}:
        return 1_568 + _IMAGE_FRAMING_TOKENS
    raise ValueError("unsupported image detail for selected Claude model")


def _detail(value: str | None) -> str:
    effective = "auto" if value is None else value.strip().lower()
    if value is not None and effective not in {"low", "high", "auto", "original"}:
        raise ValueError("unsupported image detail")
    return effective


def _selected_model(provider: str, model: str) -> str:
    lowered = model.strip().lower()
    prefixes: tuple[str, ...] = ()
    if provider == "openai_codex":
        prefixes = ("openai-codex/", "openai_codex/")
    elif provider == "github_copilot":
        prefixes = ("github-copilot/",)
    elif provider == "bedrock":
        prefixes = (
            "global.anthropic.", "anthropic.", "us.anthropic.", "eu.anthropic.",
            "apac.anthropic.",
        )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return lowered[len(prefix):]
    return lowered


def _openai_image_model(provider: str, model: str) -> str | None:
    selected = _selected_model(provider, model)
    if selected in _OPENAI_IMAGE_BASES:
        return selected
    return _valid_dated_model(selected, _OPENAI_IMAGE_BASES)


def _openai_image_family(selected: str) -> bool:
    return any(
        selected == base or selected.startswith(f"{base}-")
        for base in _OPENAI_IMAGE_BASES
    )


def _claude_image_model(provider: str, model: str) -> str | None:
    selected = _selected_model(provider, model)
    claude_bases = tuple(
        f"claude-{family}-4-{version}"
        for version in ("5", "6")
        for family in ("haiku", "sonnet", "opus")
    ) + ("claude-opus-4-7",)
    if selected in claude_bases:
        return selected
    for base in claude_bases:
        suffix = selected[len(base):] if selected.startswith(f"{base}-") else ""
        if not suffix:
            continue
        dated = suffix.removeprefix("-").removesuffix("-v1:0")
        if len(dated) == 8 and dated.isdigit():
            try:
                date.fromisoformat(f"{dated[:4]}-{dated[4:6]}-{dated[6:8]}")
            except ValueError:
                continue
            return base
        if suffix == "-v1:0":
            return base
    return None


def _valid_dated_model(model: str, bases: tuple[str, ...]) -> str | None:
    for base in bases:
        prefix = f"{base}-"
        if not model.startswith(prefix):
            continue
        suffix = model[len(prefix):]
        try:
            date.fromisoformat(suffix)
        except ValueError:
            continue
        if len(suffix) == 10:
            return base
    return None


__all__ = [
    "ImageOccurrence",
    "ImagePrice",
    "ImageRequestPricing",
    "image_request_pricing",
]
