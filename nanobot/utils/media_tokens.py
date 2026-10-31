"""Token allowances for inline media in provider requests.

Provider tokenizers do not consume base64 image bytes as text.  Counting the
wire payload would overstate a normal 1 MiB image by roughly three orders of
magnitude, so media is measured with a bounded vision allowance instead.  The
allowance is deliberately conservative and provider-agnostic: it never claims
an exact tokenizer result, but it keeps the final request gate useful for
multimodal conversations.
"""

from __future__ import annotations

import base64
import math
import struct
from collections.abc import Mapping
from datetime import date
from typing import cast

# These generic values are retained for logical estimates that do not have a
# selected transport identity. Final request measurement uses model policies
# below and never treats these as universal provider limits.
MAX_IMAGE_TOKENS = 16_384
MAX_AUDIO_TOKENS = 8_192
MAX_VIDEO_TOKENS = 32_768
MAX_DOCUMENT_TOKENS = 16_384
_MIN_IMAGE_TOKENS = 128
_IMAGE_FRAMING_TOKENS = 32
_SMALL_UNKNOWN_MEDIA_BYTES = 8 * 1024
_MEDIUM_UNKNOWN_MEDIA_BYTES = 128 * 1024

_IMAGE_TYPES = frozenset({"image", "image_url", "input_image", "output_image"})
_AUDIO_TYPES = frozenset({"audio", "audio_url", "input_audio", "output_audio"})
_VIDEO_TYPES = frozenset({"video", "video_url", "input_video", "output_video"})
_DOCUMENT_TYPES = frozenset({"document", "input_file", "file"})

_MAX_DATA_URI_HEADER_CHARS = 4_096
_MAX_HEADER_BASE64_CHARS = 512 * 1024
_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


def _media_family(mime: str) -> str | None:
    lowered = mime.strip().lower()
    if lowered.startswith("image/"):
        return "image"
    if lowered.startswith("audio/"):
        return "audio"
    if lowered.startswith("video/"):
        return "video"
    return None


def _decode_prefix(payload: str) -> bytes:
    chunk = payload[:_MAX_HEADER_BASE64_CHARS]
    chunk += "=" * (-len(chunk) % 4)
    try:
        return base64.b64decode(chunk, validate=False)
    except (ValueError, TypeError):
        return b""


def _png_dimensions(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 24 or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    width, height = struct.unpack(">II", raw[16:24])
    return (width, height) if width and height else None


def _gif_dimensions(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 10 or raw[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    width, height = struct.unpack("<HH", raw[6:10])
    return (width, height) if width and height else None


def _webp_dimensions(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 30 or not raw.startswith(b"RIFF") or raw[8:12] != b"WEBP":
        return None
    chunk = raw[12:16]
    if chunk == b"VP8X":
        width = 1 + int.from_bytes(raw[24:27], "little")
        height = 1 + int.from_bytes(raw[27:30], "little")
        return (width, height) if width and height else None
    if chunk == b"VP8 " and raw[23:26] == b"\x9d\x01\x2a":
        width = int.from_bytes(raw[26:28], "little") & 0x3FFF
        height = int.from_bytes(raw[28:30], "little") & 0x3FFF
        return (width, height) if width and height else None
    if chunk == b"VP8L" and raw[20] == 0x2F:
        bits = int.from_bytes(raw[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return (width, height) if width and height else None
    return None


def _jpeg_dimensions(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 4 or not raw.startswith(b"\xff\xd8"):
        return None
    offset = 2
    while offset + 4 <= len(raw):
        if raw[offset] != 0xFF:
            offset += 1
            continue
        marker = raw[offset + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        if marker == 0xDA:
            return None
        length = int.from_bytes(raw[offset + 2 : offset + 4], "big")
        if length < 2:
            return None
        if marker in _SOF_MARKERS and offset + 9 <= len(raw):
            height = int.from_bytes(raw[offset + 5 : offset + 7], "big")
            width = int.from_bytes(raw[offset + 7 : offset + 9], "big")
            return (width, height) if width and height else None
        offset += 2 + length
    return None


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    """Read image dimensions from a bounded prefix without decoding pixels."""
    for reader in (_png_dimensions, _gif_dimensions, _webp_dimensions, _jpeg_dimensions):
        dimensions = reader(data)
        if dimensions is not None:
            return dimensions
    return None


def estimate_image_tokens(width: int, height: int) -> int:
    """Return a conservative vision-token allowance for one image.

    The two terms cover Qwen-style 28x28 visual patches and Claude-style
    area/750 accounting.  The result is capped at the documented maximum
    visual context of Qwen2/2.5/3-VL.
    """
    if width <= 0 or height <= 0:
        return MAX_IMAGE_TOKENS
    patches = math.ceil(width / 28) * math.ceil(height / 28)
    area = math.ceil(width * height / 750)
    return min(
        MAX_IMAGE_TOKENS,
        max(_MIN_IMAGE_TOKENS, patches, area) + _IMAGE_FRAMING_TOKENS,
    )


def _valid_dated_model(model: str, bases: tuple[str, ...]) -> str | None:
    """Return a canonical base only for a complete supported dated name."""
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


def _selected_model(provider: str, model: str) -> str:
    """Strip only routing prefixes owned by the selected provider."""
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
    openai_models = (
        "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1",
        "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "gpt-5.4-mini",
        "gpt-5.4",
    )
    if selected in openai_models:
        return selected
    return _valid_dated_model(selected, openai_models)


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


def model_image_tokens(
    provider: str,
    model: str,
    dimensions: tuple[int, int] | None,
    detail: str | None,
    *,
    remote: bool = False,
) -> int:
    """Return a bounded image allowance for one selected provider/model.

    A missing dimension pair means the image is remote, opaque, or a file
    reference. Policies either provide a documented upper bound or reject it.
    """
    selected = _selected_model(provider, model)
    effective_detail = "auto" if detail is None else detail.strip().lower()
    if detail is not None and effective_detail not in {"low", "high", "auto", "original"}:
        raise ValueError("unsupported image detail")

    if provider == "local_manager" and selected == "qwen3.8-27b-ud-q6_k_xl":
        if effective_detail not in {"low", "high", "auto"}:
            raise ValueError("unsupported local image detail")
        if dimensions is None:
            raise ValueError("local image dimensions are unavailable")
        width, height = dimensions
        if width <= 0 or height <= 0:
            raise ValueError("local image dimensions are invalid")
        patches = math.ceil(width / 28) * math.ceil(height / 28)
        area = math.ceil(width * height / 750)
        return max(_MIN_IMAGE_TOKENS, patches, area) + _IMAGE_FRAMING_TOKENS

    openai_model = _openai_image_model(provider, model)
    if openai_model is not None:
        if openai_model in {"gpt-4o", "gpt-4o-mini"}:
            base, tile = (85, 170) if openai_model == "gpt-4o" else (2_833, 5_667)
            if effective_detail == "low":
                return base
            if effective_detail not in {"high", "auto"}:
                raise ValueError("unsupported image detail for selected model")
            if dimensions is None or remote:
                return base + 8 * tile
            width, height = dimensions
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
            # The base model follows the documented 4o high-resolution tile rule.
            base, tile = 85, 170
            if effective_detail == "low":
                return base
            if effective_detail not in {"high", "auto"}:
                raise ValueError("unsupported image detail for selected model")
            if dimensions is None or remote:
                return base + 8 * tile
            width, height = dimensions
            if width <= 0 or height <= 0:
                raise ValueError("image dimensions are invalid")
            scale = min(1.0, 2_048 / max(width, height))
            width = max(1, math.ceil(width * scale))
            height = max(1, math.ceil(height * scale))
            scale = min(1.0, 768 / min(width, height))
            width = max(1, math.ceil(width * scale))
            height = max(1, math.ceil(height * scale))
            return base + math.ceil(width / 512) * math.ceil(height / 512) * tile

    claude_model = _claude_image_model(provider, model)
    if claude_model is not None:
        if claude_model == "claude-opus-4-7":
            if effective_detail in {"auto", "high"}:
                return 4_784 + _IMAGE_FRAMING_TOKENS
            if effective_detail == "standard":
                return 1_568 + _IMAGE_FRAMING_TOKENS
            raise ValueError("unsupported image detail for selected Claude model")
        if effective_detail in {"auto", "standard"}:
            return 1_568 + _IMAGE_FRAMING_TOKENS
        raise ValueError("unsupported image detail for selected Claude model")

    raise ValueError("no image accounting policy for selected model")


def unknown_image_tokens(encoded_bytes: int) -> int:
    """Bound an image whose container format could not be parsed.

    Payload size is only a fallback signal: a few kilobytes cannot encode a
    high-resolution photo, while a large unknown container must retain the
    full vision allowance.
    """
    if encoded_bytes < _SMALL_UNKNOWN_MEDIA_BYTES:
        return _MIN_IMAGE_TOKENS + _IMAGE_FRAMING_TOKENS
    if encoded_bytes < _MEDIUM_UNKNOWN_MEDIA_BYTES:
        return 1024
    return MAX_IMAGE_TOKENS


def media_token_ceiling(kind: str) -> int:
    """Return the maximum local allowance for one media item."""
    if kind == "image":
        return MAX_IMAGE_TOKENS
    if kind == "audio":
        return MAX_AUDIO_TOKENS
    if kind == "video":
        return MAX_VIDEO_TOKENS
    if kind == "document":
        return MAX_DOCUMENT_TOKENS
    return MAX_IMAGE_TOKENS


def _parse_data_uri(value: str) -> tuple[str, str] | None:
    """Parse a data URI with one bounded header split and one parameter pass."""
    start = len(value) - len(value.lstrip())
    end = len(value.rstrip())
    comma = value.find(",", start, min(end, start + _MAX_DATA_URI_HEADER_CHARS + 1))
    if comma < 0:
        return None
    header = value[start:comma]
    payload = value[comma + 1 : end]
    if not header[:5].lower() == "data:":
        return None
    components = header[5:].split(";")
    mime = components[0].strip()
    if not mime or not any(component.strip().lower() == "base64" for component in components[1:]):
        return None
    return mime, payload


def inline_media_details(value: str) -> tuple[str, tuple[int, int] | None, int] | None:
    """Return MIME family, decoded image dimensions, and decoded prefix size."""
    parsed = _parse_data_uri(value)
    if parsed is None:
        return None
    mime, payload = parsed
    family = _media_family(mime)
    if family is None:
        return None
    if family != "image":
        return family, None, 0
    raw_prefix = _decode_prefix(payload)
    dimensions = image_dimensions(raw_prefix)
    return family, dimensions, len(raw_prefix)


def inline_media_tokens(value: str) -> tuple[int, str] | None:
    """Estimate tokens for a ``data:<mime>;base64,...`` payload."""
    details = inline_media_details(value)
    if details is None:
        return None
    family, dimensions, encoded_bytes = details
    if family != "image":
        return media_token_ceiling(family), family
    if dimensions is None:
        return unknown_image_tokens(encoded_bytes), family
    return estimate_image_tokens(*dimensions), family


def remote_media_tokens(kind: str) -> int:
    """Return the allowance for media whose dimensions are not locally known."""
    return media_token_ceiling(kind)


def binary_media_tokens(data: bytes, kind: str) -> int:
    """Estimate tokens for provider-native binary media (for example Bedrock)."""
    if kind == "image":
        dimensions = image_dimensions(data)
        if dimensions is not None:
            return estimate_image_tokens(*dimensions)
        return unknown_image_tokens(len(data))
    return media_token_ceiling(kind)


def _url_from_block(block: Mapping[str, object]) -> str | None:
    image = block.get("image_url")
    if isinstance(image, str) and image:
        return image
    if isinstance(image, Mapping):
        url = cast(Mapping[str, object], image).get("url")
        if isinstance(url, str) and url:
            return url
    source = block.get("source")
    if isinstance(source, Mapping):
        source_map = cast(Mapping[str, object], source)
        url = source_map.get("url")
        if isinstance(url, str) and url:
            return url
    url = block.get("url")
    return url if isinstance(url, str) and url else None


def _data_from_block(block: Mapping[str, object]) -> tuple[str, str] | None:
    source = block.get("source")
    if isinstance(source, Mapping):
        source_map = cast(Mapping[str, object], source)
        media_type = source_map.get("media_type")
        data = source_map.get("data")
        if (
            source_map.get("type") == "base64"
            and isinstance(media_type, str)
            and isinstance(data, str)
        ):
            family = _media_family(media_type)
            if family is not None:
                return data, family
    data = block.get("data")
    media_type = block.get("media_type")
    if isinstance(data, str) and isinstance(media_type, str):
        family = _media_family(media_type)
        if family is not None:
            return data, family
    return None


def media_block_tokens(block: Mapping[str, object]) -> tuple[int, str] | None:
    """Return a token allowance for one structured media content block."""
    raw_type = block.get("type")
    block_type = raw_type.strip().lower() if isinstance(raw_type, str) else ""
    if block_type in _IMAGE_TYPES:
        direct = _data_from_block(block)
        if direct is not None:
            payload, family = direct
            estimate = inline_media_tokens(f"data:{family}/unknown;base64,{payload}")
            if estimate is not None:
                return estimate
        url = _url_from_block(block)
        if url:
            estimate = inline_media_tokens(url)
            return estimate or (remote_media_tokens("image"), "image")
        return remote_media_tokens("image"), "image"
    if block_type in _AUDIO_TYPES:
        return remote_media_tokens("audio"), "audio"
    if block_type in _VIDEO_TYPES:
        return remote_media_tokens("video"), "video"
    if block_type in _DOCUMENT_TYPES:
        return remote_media_tokens("document"), "document"
    if "image" in block and isinstance(block.get("image"), Mapping):
        nested = cast(Mapping[str, object], block["image"])
        source = nested.get("source")
        if isinstance(source, Mapping):
            payload = cast(Mapping[str, object], source).get("bytes")
            if isinstance(payload, bytes):
                return binary_media_tokens(payload, "image"), "image"
    return None


__all__ = [
    "MAX_AUDIO_TOKENS",
    "MAX_DOCUMENT_TOKENS",
    "MAX_IMAGE_TOKENS",
    "MAX_VIDEO_TOKENS",
    "binary_media_tokens",
    "estimate_image_tokens",
    "image_dimensions",
    "inline_media_details",
    "inline_media_tokens",
    "media_block_tokens",
    "media_token_ceiling",
    "model_image_tokens",
    "remote_media_tokens",
    "unknown_image_tokens",
]
