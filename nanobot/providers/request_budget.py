"""Deterministic hard limits for finalized provider transport requests.

The logical prompt reducer works on Chat-style history.  This module is the
last boundary before a provider SDK receives a concrete request body, so it
counts the fully converted body, including provider extras and tool schemas.
It intentionally has no provider imports and fails closed for values that
cannot be represented deterministically.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from typing import cast

import tiktoken
from tiktoken.model import MODEL_TO_ENCODING

from nanobot.utils.media_tokens import (
    image_dimensions,
    inline_media_details,
    model_image_tokens,
)

DEFAULT_FINALIZED_REQUEST_SAFETY_BUFFER = 1024
TOKEN_ESTIMATE_SAFETY_RATIO = 1.15
_BEDROCK_BINARY_FRAMING_BYTES = 64
_MODEL_ENCODING_BASES: dict[str, str] = {
    "gpt-4-turbo": "cl100k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4.1-mini": "o200k_base",
    "gpt-4.1-nano": "o200k_base",
    "gpt-5.4": "o200k_base",
    "gpt-5.4-mini": "o200k_base",
    "gpt-5.5": "o200k_base",
    "gpt-5.6-sol": "o200k_base",
    "gpt-5.6-terra": "o200k_base",
    "gpt-5.6-luna": "o200k_base",
}
_DATED_ENCODING_BASES = (
    "gpt-4-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4.1", "gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "gpt-5.6-sol",
    "gpt-5.6-terra", "gpt-5.6-luna",
)
_UNSUPPORTED_IMAGE_BLOCK_TYPES = frozenset({
    "audio", "audio_url", "input_audio", "output_audio", "video", "video_url",
    "input_video", "output_video", "document", "file", "input_file", "output_file",
})


@dataclass(frozen=True, slots=True)
class ProviderRequestLimits:
    """Configured total context capacity for one provider request.

    ``total_tokens`` includes the completion reserve.  A concrete transport's
    output cap supplies that reserve unless the caller explicitly provides a
    larger one.  ``model_context_window_tokens`` lets routing wrappers clamp a
    caller's configured capacity to the selected leaf model's known window.
    """

    total_tokens: int
    completion_reserve_tokens: int | None = None
    safety_buffer_tokens: int = DEFAULT_FINALIZED_REQUEST_SAFETY_BUFFER
    model_context_window_tokens: int | None = None

    def __post_init__(self) -> None:
        for name, value, allow_none in (
            ("total_tokens", self.total_tokens, False),
            ("completion_reserve_tokens", self.completion_reserve_tokens, True),
            ("safety_buffer_tokens", self.safety_buffer_tokens, False),
            ("model_context_window_tokens", self.model_context_window_tokens, True),
        ):
            if value is None and allow_none:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be positive")

    @property
    def effective_total_tokens(self) -> int:
        if self.model_context_window_tokens is None:
            return self.total_tokens
        return min(self.total_tokens, self.model_context_window_tokens)

    def for_model_window(self, window: int | None) -> "ProviderRequestLimits":
        """Return an equivalent limit clamped to one selected concrete model."""
        if window is None:
            return self
        if isinstance(window, bool) or not isinstance(cast(object, window), int) or window <= 0:
            raise ValueError("model context window must be a positive integer")
        current = self.model_context_window_tokens
        return replace(self, model_context_window_tokens=min(current, window) if current else window)


def derive_provider_request_limits(
    total_tokens: int | None,
    completion_reserve_tokens: int | None,
) -> ProviderRequestLimits | None:
    """Build runtime hard limits when a concrete total context capacity is known."""
    if (
        isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or total_tokens <= 0
        or isinstance(completion_reserve_tokens, bool)
        or not isinstance(completion_reserve_tokens, int)
        or completion_reserve_tokens < 0
    ):
        return None
    return ProviderRequestLimits(
        total_tokens=total_tokens,
        completion_reserve_tokens=completion_reserve_tokens,
    )


@dataclass(frozen=True, slots=True)
class TransportIdentity:
    """The provider protocol and selected model used for one concrete body."""

    provider: str
    transport: str
    model: str

    def __post_init__(self) -> None:
        for name, value in (("provider", self.provider), ("transport", self.transport), ("model", self.model)):
            if not isinstance(cast(object, value), str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class TransportMeasurement:
    """A content-free measurement of one finalized request body."""

    identity: TransportIdentity
    input_tokens: int
    completion_reserve_tokens: int
    safety_buffer_tokens: int
    total_limit_tokens: int
    input_limit_tokens: int
    source: str

    @property
    def fits(self) -> bool:
        return self.input_tokens <= self.input_limit_tokens

    @property
    def over_by(self) -> int:
        return max(0, self.input_tokens - self.input_limit_tokens)


class FinalizedRequestBudgetExceeded(Exception):  # noqa: N818 - public contract name
    """A local, non-retryable rejection before provider transport dispatch."""

    DEFAULT_MESSAGE = "Finalized provider request exceeds its context budget."

    def __init__(self, measurement: TransportMeasurement | None = None) -> None:
        self.measurement = measurement
        super().__init__(self.DEFAULT_MESSAGE)


def _completion_reserve(body: Mapping[str, object]) -> int:
    """Read one concrete generation cap from each supported request shape."""
    values: list[object] = [
        body.get("max_tokens"),
        body.get("max_completion_tokens"),
        body.get("max_output_tokens"),
    ]
    inference = body.get("inferenceConfig")
    if isinstance(inference, Mapping):
        values.append(cast(Mapping[object, object], inference).get("maxTokens"))
    output = body.get("output")
    if isinstance(output, Mapping):
        values.append(cast(Mapping[object, object], output).get("max_tokens"))
    parsed: list[int] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("finalized completion reserve is invalid")
        parsed.append(value)
    return max(parsed, default=0)


def known_tiktoken_encoding(provider: str, model: str) -> str | None:
    """Return an installed encoding only for an exact known model identity.

    Provider routing prefixes are stripped only for their owning providers.
    Dated snapshots must append a syntactically valid ISO date to an explicitly
    supported canonical model base; arbitrary model-name prefixes are ignored.
    """
    selected = model.strip().lower()
    if provider == "openai_codex":
        for prefix in ("openai-codex/", "openai_codex/"):
            if selected.startswith(prefix):
                selected = selected[len(prefix):]
                break
    elif provider == "github_copilot" and selected.startswith("github-copilot/"):
        selected = selected[len("github-copilot/"):]

    exact = MODEL_TO_ENCODING.get(selected) or _MODEL_ENCODING_BASES.get(selected)
    if exact is not None:
        return exact
    for base in _DATED_ENCODING_BASES:
        suffix = selected.removeprefix(f"{base}-")
        if suffix == selected or len(suffix) != 10:
            continue
        try:
            date.fromisoformat(suffix)
        except ValueError:
            continue
        return MODEL_TO_ENCODING.get(base) or _MODEL_ENCODING_BASES.get(base)
    return None


def _is_index(value: str) -> bool:
    return value.isdigit()


def _is_message_content_block(path: tuple[str, ...]) -> bool:
    return (
        len(path) == 4
        and path[0] == "messages"
        and _is_index(path[1])
        and path[2] == "content"
        and _is_index(path[3])
    )


def _is_nested_content_block(path: tuple[str, ...], marker: str) -> bool:
    return (
        len(path) == 6
        and path[0] == "messages"
        and _is_index(path[1])
        and path[2] == "content"
        and _is_index(path[3])
        and path[4] == marker
        and _is_index(path[5])
    )


def _is_responses_content_block(path: tuple[str, ...]) -> bool:
    return (
        len(path) == 4
        and path[0] == "input"
        and _is_index(path[1])
        and path[2] in {"content", "output"}
        and _is_index(path[3])
    )


def _is_bedrock_tool_result_block(path: tuple[str, ...]) -> bool:
    return (
        len(path) == 7
        and path[0] == "messages"
        and _is_index(path[1])
        and path[2] == "content"
        and _is_index(path[3])
        and path[4] == "toolResult"
        and path[5] == "content"
        and _is_index(path[6])
    )


def _protocol_block_kind(
    identity: TransportIdentity,
    path: tuple[str, ...],
    value: Mapping[str, object],
    media_context: str | None,
) -> str | None:
    """Recognize media only at a supported protocol content-block boundary."""
    block_type = value.get("type")
    lowered = block_type.lower() if isinstance(block_type, str) else ""
    transport = identity.transport
    direct_chat_block = (
        transport in {"openai_chat_completions", "openai_chat_completions_stream"}
        and _is_message_content_block(path)
    )
    direct_responses_block = (
        transport in {
            "openai_responses", "azure_openai_responses", "openai_codex_responses",
            "xai_grok_responses",
        }
        and _is_responses_content_block(path)
        and media_context in {"responses_message", "responses_function_output"}
    )
    direct_anthropic_block = (
        transport in {"anthropic_messages", "anthropic_messages_stream"}
        and _is_message_content_block(path)
    )
    nested_anthropic_block = (
        transport in {"anthropic_messages", "anthropic_messages_stream"}
        and _is_nested_content_block(path, "content")
        and media_context == "anthropic_tool_result"
    )
    direct_bedrock_block = (
        transport in {"bedrock_converse", "bedrock_converse_stream"}
        and _is_message_content_block(path)
    )
    nested_bedrock_block = (
        transport in {"bedrock_converse", "bedrock_converse_stream"}
        and _is_bedrock_tool_result_block(path)
        and media_context == "bedrock_tool_result"
    )
    content_block_boundary = (
        direct_chat_block
        or direct_responses_block
        or direct_anthropic_block
        or nested_anthropic_block
        or direct_bedrock_block
        or nested_bedrock_block
    )
    if lowered in _UNSUPPORTED_IMAGE_BLOCK_TYPES and content_block_boundary:
        raise ValueError("unsupported finalized request media block")

    if transport in {"openai_chat_completions", "openai_chat_completions_stream"}:
        if direct_chat_block and lowered == "image_url":
            image_url = value.get("image_url")
            if isinstance(image_url, str):
                return "openai_chat_image"
            if isinstance(image_url, Mapping):
                url = cast(Mapping[str, object], image_url).get("url")
                if isinstance(url, str):
                    return "openai_chat_image"
            raise ValueError("unsupported OpenAI image block")
        return None

    if transport in {
        "openai_responses", "azure_openai_responses", "openai_codex_responses",
        "xai_grok_responses",
    }:
        if direct_responses_block and lowered == "input_image":
            image_url = value.get("image_url")
            file_id = value.get("file_id")
            if isinstance(image_url, str) or isinstance(file_id, str):
                return "responses_image"
            raise ValueError("unsupported Responses image block")
        return None

    if transport in {"anthropic_messages", "anthropic_messages_stream"}:
        if lowered == "tool_result" and direct_anthropic_block:
            return "anthropic_tool_result"
        if lowered == "image" and (direct_anthropic_block or nested_anthropic_block):
            return "anthropic_image"
        return None

    if transport in {"bedrock_converse", "bedrock_converse_stream"}:
        actual_block = direct_bedrock_block or nested_bedrock_block
        if actual_block:
            for key in ("audio", "document", "video"):
                if key in value:
                    raise ValueError("unsupported finalized request media block")
            if "toolResult" in value and direct_bedrock_block:
                return "bedrock_tool_result"
            if "image" in value and "toolUse" not in value:
                return "bedrock_image"
        return None
    return None


def _child_media_context(
    identity: TransportIdentity,
    path: tuple[str, ...],
    parent: Mapping[str, object],
    key: str,
    item: object,
    block_kind: str | None,
    media_context: str | None,
    media_detail: str | None,
) -> tuple[str | None, str | None]:
    """Pass media context only through the owning protocol field."""
    if identity.transport in {
        "openai_responses", "azure_openai_responses", "openai_codex_responses",
        "xai_grok_responses",
    } and len(path) == 2 and path[0] == "input" and _is_index(path[1]):
        item_type = parent.get("type")
        role = parent.get("role")
        if (
            key == "content"
            and role in ("user", "assistant")
            and (item_type is None or item_type == "message")
        ):
            return "responses_message", None
        if parent.get("type") == "function_call_output" and key == "output":
            return "responses_function_output", None

    if block_kind == "anthropic_tool_result" and key == "content":
        return "anthropic_tool_result", None
    if block_kind == "bedrock_tool_result" and key == "toolResult":
        return "bedrock_tool_result_wrapper", None
    if media_context == "bedrock_tool_result_wrapper" and key == "content":
        return "bedrock_tool_result", None

    if block_kind == "openai_chat_image" and key == "image_url":
        return "openai_chat_image_url", _chat_image_detail(item)
    if block_kind == "anthropic_image" and key == "source":
        if not isinstance(item, Mapping):
            raise ValueError("unsupported Anthropic image source")
        source = cast(Mapping[str, object], item)
        if source.get("type") == "base64":
            mime = source.get("media_type")
            if (
                isinstance(mime, str)
                and mime.lower().startswith("image/")
                and isinstance(source.get("data"), str)
            ):
                return "anthropic_image_source", None
        elif source.get("type") == "url" and isinstance(source.get("url"), str):
            return "anthropic_image_source", None
        raise ValueError("unsupported Anthropic image source")
    if media_context == "anthropic_image_source":
        source_type = parent.get("type")
        if source_type == "base64" and key == "data":
            mime = parent.get("media_type")
            if isinstance(mime, str) and mime.lower().startswith("image/"):
                return "anthropic_image_data", None
        if source_type == "url" and key == "url":
            return "anthropic_image_url", None
    if block_kind == "bedrock_image" and key == "image":
        if not isinstance(item, Mapping):
            raise ValueError("unsupported Bedrock image source")
        image = cast(Mapping[str, object], item)
        source = image.get("source")
        if not isinstance(source, Mapping) or not isinstance(
            cast(Mapping[str, object], source).get("bytes"), bytes,
        ):
            raise ValueError("unsupported Bedrock image source")
        return "bedrock_image", _image_detail(parent, image)

    if media_context == "bedrock_image" and key == "source" and isinstance(item, Mapping):
        return "bedrock_image_source", media_detail
    if media_context == "bedrock_image_source" and key == "bytes":
        return "bedrock_image_bytes", media_detail
    return None, None


def _image_detail(parent: Mapping[str, object], image_url: object) -> str | None:
    if isinstance(image_url, Mapping):
        detail = cast(Mapping[str, object], image_url).get("detail")
        if detail is not None:
            if not isinstance(detail, str):
                raise ValueError("unsupported image detail")
            return detail
    detail = parent.get("detail")
    if detail is not None and not isinstance(detail, str):
        raise ValueError("unsupported image detail")
    return detail


def _chat_image_detail(image_url: object) -> str | None:
    """Read OpenAI Chat detail only from the owning image_url field."""
    if not isinstance(image_url, Mapping):
        return None
    detail = cast(Mapping[str, object], image_url).get("detail")
    if detail is not None and not isinstance(detail, str):
        raise ValueError("unsupported image detail")
    return detail


def _context_detail(context: str, parent: Mapping[str, object]) -> str | None:
    if context != "openai_chat_image_url":
        return None
    detail = parent.get("detail")
    if detail is not None and not isinstance(detail, str):
        raise ValueError("unsupported image detail")
    return detail


def _media_context_for_list(
    identity: TransportIdentity,
    path: tuple[str, ...],
    media_context: str | None,
) -> str | None:
    """Keep media context only on the exact list owned by a protocol field."""
    if media_context in {"responses_message", "responses_function_output"}:
        expected_key = "content" if media_context == "responses_message" else "output"
        if (
            identity.transport in {
                "openai_responses", "azure_openai_responses",
                "openai_codex_responses", "xai_grok_responses",
            }
            and len(path) == 3
            and path[0] == "input"
            and _is_index(path[1])
            and path[2] == expected_key
        ):
            return media_context
    if media_context == "anthropic_tool_result" and (
        identity.transport in {"anthropic_messages", "anthropic_messages_stream"}
        and len(path) == 5
        and path[0] == "messages"
        and _is_index(path[1])
        and path[2] == "content"
        and _is_index(path[3])
        and path[4] == "content"
    ):
        return media_context
    if media_context == "bedrock_tool_result" and (
        identity.transport in {"bedrock_converse", "bedrock_converse_stream"}
        and len(path) == 6
        and path[0] == "messages"
        and _is_index(path[1])
        and path[2] == "content"
        and _is_index(path[3])
        and path[4] == "toolResult"
        and path[5] == "content"
    ):
        return media_context
    return None


def _media_mime_hint(mapping: Mapping[str, object]) -> str | None:
    mime = mapping.get("media_type")
    return mime if isinstance(mime, str) else None


def _measured_image_string(
    value: str,
    identity: TransportIdentity,
    detail: str | None,
    mime_hint: str | None = None,
) -> tuple[int, str]:
    """Measure one known image payload under the selected model policy."""
    candidate = value if mime_hint is None else f"data:{mime_hint};base64,{value}"
    parsed = inline_media_details(candidate)
    if parsed is not None:
        family, dimensions, _encoded_bytes = parsed
        if family != "image":
            raise ValueError("unsupported finalized request media family")
        return model_image_tokens(
            identity.provider, identity.model, dimensions, detail,
        ), "inline_image"
    lowered = value.strip().lower()
    if lowered.startswith(("http://", "https://")):
        return model_image_tokens(identity.provider, identity.model, None, detail, remote=True), (
            "remote_image"
        )
    raise ValueError("unsupported finalized request image value")


def _bedrock_binary_kind(
    identity: TransportIdentity,
    path: tuple[str, ...],
    media_context: str | None,
) -> str | None:
    if identity.provider != "bedrock":
        return None
    direct_image_bytes = (
        len(path) == 7
        and path[0] == "messages"
        and _is_index(path[1])
        and path[2] == "content"
        and _is_index(path[3])
        and path[4:] == ("image", "source", "bytes")
    )
    tool_result_image_bytes = (
        len(path) == 10
        and path[0] == "messages"
        and _is_index(path[1])
        and path[2] == "content"
        and _is_index(path[3])
        and path[4] == "toolResult"
        and path[5] == "content"
        and _is_index(path[6])
        and path[7:] == ("image", "source", "bytes")
    )
    if (
        media_context == "bedrock_image_bytes"
        and (direct_image_bytes or tool_result_image_bytes)
    ):
        return "bedrock_binary_image"
    if (
        len(path) == 6
        and path[0] == "messages"
        and _is_index(path[1])
        and path[2] == "content"
        and _is_index(path[3])
        and path[4:] == ("reasoningContent", "redactedContent")
    ):
        return "bedrock_redacted_reasoning"
    return None


def _canonicalize_finalized_value(
    value: object,
    identity: TransportIdentity,
    path: tuple[str, ...] = (),
    media_context: str | None = None,
    media_detail: str | None = None,
) -> tuple[object, int, frozenset[str]]:
    """Return canonical JSON data and supplemental token/byte units.

    Bedrock Converse accepts raw ``bytes`` only below documented media sources
    and replayed ``reasoningContent.redactedContent`` blocks. Their content is
    replaced with a length-only marker so measurement never retains binary
    data. Every other non-JSON value remains opaque and therefore fails closed.

    Inline media is replaced with a marker and contributes a bounded vision
    allowance rather than its base64 wire length, which is not a text-token
    measurement. Text is measured separately with a tokenizer.
    """
    if isinstance(value, bytes):
        kind = _bedrock_binary_kind(identity, path, media_context)
        if kind is None:
            raise ValueError("unsupported binary finalized request value")
        if kind == "bedrock_binary_image":
            units = model_image_tokens(
                identity.provider, identity.model, image_dimensions(value), media_detail,
            )
            return (
                {"__nanobot_bedrock_binary_bytes__": len(value)},
                units,
                frozenset({"bedrock_binary_media"}),
            )
        # Redacted reasoning is opaque token content, not decodable media.
        transport_bytes = 4 * ((len(value) + 2) // 3) + _BEDROCK_BINARY_FRAMING_BYTES
        return (
            {"__nanobot_bedrock_binary_bytes__": len(value)},
            transport_bytes,
            frozenset({"bedrock_redacted_reasoning"}),
        )
    if isinstance(value, dict):
        canonical: dict[str, object] = {}
        supplemental_units = 0
        measurement_kinds: set[str] = set()
        mapping = cast(Mapping[str, object], value)
        block_kind = _protocol_block_kind(identity, path, mapping, media_context)
        for raw_key, item in cast(dict[object, object], value).items():
            if not isinstance(raw_key, str):
                raise ValueError("finalized provider request keys must be strings")
            key = raw_key
            measured_image: tuple[int, str] | None = None
            if block_kind in {"openai_chat_image", "responses_image"}:
                if key == "image_url" and isinstance(item, str):
                    detail = (
                        _chat_image_detail(item)
                        if block_kind == "openai_chat_image"
                        else _image_detail(mapping, item)
                    )
                    measured_image = _measured_image_string(
                        item, identity, detail,
                    )
                elif block_kind == "responses_image" and key == "file_id" and isinstance(item, str):
                    measured_image = (
                        model_image_tokens(
                            identity.provider, identity.model, None, _image_detail(mapping, item),
                            remote=True,
                        ),
                        "remote_image",
                    )
            elif media_context == "openai_chat_image_url":
                selected_detail = (
                    media_detail
                    if media_detail is not None
                    else _context_detail(media_context, mapping)
                )
                if key == "url" and isinstance(item, str):
                    measured_image = _measured_image_string(
                        item, identity, selected_detail,
                    )
            elif media_context == "anthropic_image_source":
                source_type = mapping.get("type")
                if source_type == "url" and key == "url" and isinstance(item, str):
                    measured_image = _measured_image_string(item, identity, None)
                elif source_type == "base64" and key == "data" and isinstance(item, str):
                    measured_image = _measured_image_string(
                        item, identity, None, _media_mime_hint(mapping),
                    )
            if measured_image is not None:
                tokens, source_suffix = measured_image
                canonical[key] = {"__nanobot_inline_media__": "image"}
                supplemental_units += tokens
                measurement_kinds.add(source_suffix)
                continue

            child_context, child_detail = _child_media_context(
                identity, path, mapping, key, item, block_kind, media_context, media_detail,
            )
            converted, item_units, item_kinds = _canonicalize_finalized_value(
                item,
                identity,
                (*path, key),
                child_context,
                child_detail,
            )
            canonical[key] = converted
            supplemental_units += item_units
            measurement_kinds.update(item_kinds)
        return canonical, supplemental_units, frozenset(measurement_kinds)
    if isinstance(value, (list, tuple)):
        canonical_list: list[object] = []
        supplemental_units = 0
        measurement_kinds = set()
        list_media_context = _media_context_for_list(identity, path, media_context)
        for index, item in enumerate(cast(list[object] | tuple[object, ...], value)):
            converted, item_units, item_kinds = _canonicalize_finalized_value(
                item,
                identity,
                (*path, str(index)),
                list_media_context,
                media_detail if list_media_context is not None else None,
            )
            canonical_list.append(converted)
            supplemental_units += item_units
            measurement_kinds.update(item_kinds)
        return canonical_list, supplemental_units, frozenset(measurement_kinds)
    if value is None or isinstance(value, str | int | float | bool):
        return value, 0, frozenset()
    raise ValueError("unsupported finalized provider request value")


def _text_token_measurement(
    encoded: bytes,
    identity: TransportIdentity,
) -> tuple[int, str]:
    """Estimate text tokens, failing closed to the older byte ceiling.

    Unknown model names and unavailable encodings use UTF-8 bytes, which are a
    conservative ceiling that cannot assign a nearby model's vocabulary to an
    arbitrary alias.
    """
    try:
        encoding_name = known_tiktoken_encoding(identity.provider, identity.model)
        if encoding_name is None:
            raise ValueError("no tokenizer mapping for selected model")
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = len(encoding.encode(encoded.decode("utf-8")))
        return max(1, math.ceil(tokens * TOKEN_ESTIMATE_SAFETY_RATIO)), (
            "canonical_json_tiktoken"
        )
    except Exception:
        return max(1, len(encoded)), "canonical_json_utf8_bytes"


def _serialized_input_measurement(
    identity: TransportIdentity,
    body: Mapping[str, object],
) -> tuple[int, str]:
    """Count a content-free canonical body with conservative token units.

    Text is measured with a tokenizer and a safety ratio. Inline and
    provider-native media are replaced by a bounded vision allowance because
    base64 bytes are not text tokens. Bedrock redacted-reasoning bytes retain
    their Base64-size ceiling.
    """
    if not isinstance(body, dict):
        raise ValueError("finalized provider request is not a JSON object")
    try:
        canonical, supplemental_units, measurement_kinds = _canonicalize_finalized_value(
            body, identity,
        )
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("finalized provider request cannot be measured") from exc
    text_units, source = _text_token_measurement(encoded, identity)
    for kind in sorted(measurement_kinds):
        source += f"+{kind}"
    return max(1, text_units + supplemental_units), source


def measure_finalized_request(
    identity: TransportIdentity,
    body: Mapping[str, object],
    limits: ProviderRequestLimits,
    *,
    completion_reserve_tokens: int | None = None,
) -> TransportMeasurement:
    """Measure a detached finalized body against the selected transport limit."""
    if not isinstance(cast(object, identity), TransportIdentity):
        raise ValueError("transport identity is invalid")
    if not isinstance(cast(object, body), Mapping):
        raise ValueError("finalized provider request is invalid")
    if not isinstance(cast(object, limits), ProviderRequestLimits):
        raise ValueError("provider request limits are invalid")
    if completion_reserve_tokens is not None and (
        isinstance(completion_reserve_tokens, bool)
        or not isinstance(cast(object, completion_reserve_tokens), int)
        or completion_reserve_tokens < 0
    ):
        raise ValueError("finalized completion reserve is invalid")
    concrete_reserve = max(_completion_reserve(body), completion_reserve_tokens or 0)
    reserve = max(concrete_reserve, limits.completion_reserve_tokens or 0)
    total = limits.effective_total_tokens
    input_limit = total - reserve - limits.safety_buffer_tokens
    input_tokens, source = _serialized_input_measurement(identity, body)
    return TransportMeasurement(
        identity=identity,
        input_tokens=input_tokens,
        completion_reserve_tokens=reserve,
        safety_buffer_tokens=limits.safety_buffer_tokens,
        total_limit_tokens=total,
        input_limit_tokens=input_limit,
        source=source,
    )


def validate_finalized_request(
    identity: TransportIdentity,
    body: Mapping[str, object],
    limits: ProviderRequestLimits,
    *,
    completion_reserve_tokens: int | None = None,
) -> TransportMeasurement:
    """Return a valid measurement or raise without exposing request content."""
    try:
        measurement = measure_finalized_request(
            identity, body, limits, completion_reserve_tokens=completion_reserve_tokens,
        )
    except ValueError as exc:
        raise FinalizedRequestBudgetExceeded() from exc
    if not measurement.fits:
        raise FinalizedRequestBudgetExceeded(measurement)
    return measurement
