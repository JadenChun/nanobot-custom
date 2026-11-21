"""Deterministic hard limits for finalized provider transport requests.

The logical prompt reducer works on Chat-style history.  This module is the
last boundary before a provider SDK receives a concrete request body, so it
counts the fully converted body, including provider extras and tool schemas.
It intentionally has no provider imports and fails closed for values that
cannot be represented deterministically.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Literal, cast

import tiktoken
from tiktoken.model import MODEL_TO_ENCODING

from nanobot.providers.capabilities import (
    ModelCapabilities,
    default_model_capabilities,
)
from nanobot.providers.image_pricing import ImageOccurrence
from nanobot.providers.provider_policy import (
    DEFAULT_ESTIMATED_COMPAT_MAX_REQUEST_BYTES,
    RequestAccountingPolicy,
)
from nanobot.utils.media_tokens import (
    MAX_IMAGE_TOKENS,
    estimate_image_tokens,
    image_dimensions,
    inline_media_details,
    unknown_image_tokens,
)

DEFAULT_FINALIZED_REQUEST_SAFETY_BUFFER = 1024
TOKEN_ESTIMATE_SAFETY_RATIO = 1.15
ESTIMATED_COMPAT_POLICY = RequestAccountingPolicy.ESTIMATED_COMPAT.value
RequestBudgetPolicy = Literal["strict", "estimated_compat"]
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
    total_limit_tokens: int | None
    input_limit_tokens: int | None
    source: str
    measured_bytes: int | None = None
    limit_bytes: int | None = None
    policy: str = "strict"

    @property
    def fits(self) -> bool:
        return self.input_limit_tokens is None or self.input_tokens <= self.input_limit_tokens

    @property
    def over_by(self) -> int:
        if self.input_limit_tokens is None:
            return 0
        return max(0, self.input_tokens - self.input_limit_tokens)


@dataclass(frozen=True, slots=True)
class PreparedRequestRoute:
    """Content-free identity of the concrete route selected for one request.

    The governor uses this to keep pressure accounting, routing and dispatch on
    the same leaf.  It carries no endpoint, credential, prompt or image bytes.
    """

    purpose: Literal["main", "source_summary"]
    provider: str
    model: str
    transport: str
    accounting_policy: str
    context_window_tokens: int | None
    completion_reserve_tokens: int
    native_compaction: bool
    envelope_binding: str = field(repr=False)
    fallback_index: int = 0
    route_token: str = field(init=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("provider", self.provider),
            ("model", self.model),
            ("transport", self.transport),
            ("accounting_policy", self.accounting_policy),
            ("envelope_binding", self.envelope_binding),
        ):
            if not isinstance(cast(object, value), str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.purpose not in {"main", "source_summary"}:
            raise ValueError("unsupported request purpose")
        window = self.context_window_tokens
        if (
            (
                window is not None
                and (type(window) is not int or window <= 0)
            )
            or type(self.completion_reserve_tokens) is not int
            or self.completion_reserve_tokens < 0
            or type(self.native_compaction) is not bool
            or type(self.fallback_index) is not int
            or self.fallback_index < 0
        ):
            raise ValueError("invalid request route facts")
        digest = hashlib.sha256(
            json.dumps(
                [
                    self.purpose,
                    self.provider,
                    self.model,
                    self.transport,
                    self.accounting_policy,
                    self.context_window_tokens,
                    self.completion_reserve_tokens,
                    self.native_compaction,
                    self.envelope_binding,
                    self.fallback_index,
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "route_token", digest)

    @classmethod
    def from_preview(
        cls,
        preview: "PreparedRequestBudgetPreview",
        *,
        purpose: Literal["main", "source_summary"] = "main",
        fallback_index: int = 0,
        native_compaction: bool = False,
    ) -> "PreparedRequestRoute":
        """Describe the leaf/request already measured by a provider preview."""
        measurement = preview.measurement
        return cls(
            purpose=purpose,
            provider=measurement.identity.provider,
            model=measurement.identity.model,
            transport=measurement.identity.transport,
            accounting_policy=measurement.policy,
            context_window_tokens=measurement.total_limit_tokens,
            completion_reserve_tokens=measurement.completion_reserve_tokens,
            native_compaction=native_compaction,
            envelope_binding=preview.envelope_binding,
            fallback_index=fallback_index,
        )


@dataclass(frozen=True, slots=True)
class PreparedRequestBudgetPreview:
    """Private converted request view used to align local pressure with dispatch."""

    measurement: TransportMeasurement
    envelope_binding: str
    converted_messages: tuple[dict[str, object], ...] = field(repr=False)
    converted_tools: tuple[dict[str, object], ...] | None = field(repr=False)
    effective_body: dict[str, object] = field(repr=False)
    sdk_kwargs: dict[str, object] = field(repr=False)
    route: PreparedRequestRoute | None = None


@dataclass(frozen=True, slots=True)
class LocalRequestUsageReceipt:
    """Invocation-only binding for one successful estimated-compatible request."""

    provider_scope: str
    envelope_binding: str
    model: str
    limits_binding: str
    messages: tuple[dict[str, object], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class LocalUsageAnchor:
    """A validated reported prompt count paired with its private request receipt."""

    receipt: LocalRequestUsageReceipt = field(repr=False)
    prompt_tokens: int
    projection_epoch: int | None = None


class ProviderRequestRejected(Exception):  # noqa: N818 - stable contract name
    """Content-free, non-retryable rejection before or at provider transport."""

    DEFAULT_MESSAGE = "Provider request was rejected."

    def __init__(self, *, reason: str = "invalid_request_envelope") -> None:
        self.reason = reason
        super().__init__(self.DEFAULT_MESSAGE)


class FinalizedRequestBudgetExceeded(ProviderRequestRejected):  # noqa: N818
    """A local, non-retryable rejection before provider transport dispatch."""

    DEFAULT_MESSAGE = "Finalized provider request exceeds its context budget."

    def __init__(
        self,
        measurement: TransportMeasurement | None = None,
        *,
        reason: str = "estimated_admission",
    ) -> None:
        self.measurement = measurement
        ProviderRequestRejected.__init__(self, reason=reason)
        self.args = (self.DEFAULT_MESSAGE,)


class ProviderRouteChanged(ProviderRequestRejected):
    """Dispatch selected a different leaf than the captured preparation route."""

    def __init__(
        self,
        *,
        expected: str,
        actual: str,
        actual_provider: str | None = None,
        actual_model: str | None = None,
        actual_fallback_index: int | None = None,
        consumed_through: int = -1,
    ) -> None:
        self.expected = expected
        self.actual = actual
        self.actual_provider = actual_provider
        self.actual_model = actual_model
        self.actual_fallback_index = actual_fallback_index
        self.consumed_through = consumed_through
        super().__init__(reason="route_changed")


class RequestPayloadBytesExceeded(ProviderRequestRejected):
    """The complete JSON request exceeds its independent byte ceiling."""

    def __init__(self, measured_bytes: int, limit_bytes: int) -> None:
        self.measured_bytes = measured_bytes
        self.limit_bytes = limit_bytes
        super().__init__(reason="payload_bytes")


class GovernedRequestConflictError(ProviderRequestRejected):
    """An extra body would change a source-governed model or message history."""

    def __init__(self, field_name: str) -> None:
        self.field_name = field_name
        super().__init__(reason="invalid_request_envelope")


class BackendContextOverflow(ProviderRequestRejected):
    """A local backend returned its pinned structured context overflow result."""

    def __init__(self, *, status_code: int, code: int, n_prompt_tokens: int, n_ctx: int) -> None:
        self.status_code = status_code
        self.code = code
        self.error_type = "exceed_context_size_error"
        self.n_prompt_tokens = n_prompt_tokens
        self.n_ctx = n_ctx
        super().__init__(reason="backend_context_overflow")


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
    detail: str | None,
    mime_hint: str | None = None,
) -> tuple[ImageOccurrence, str]:
    """Return one image occurrence and its measurement source."""
    candidate = value if mime_hint is None else f"data:{mime_hint};base64,{value}"
    parsed = inline_media_details(candidate)
    if parsed is not None:
        family, dimensions, encoded_bytes = parsed
        if family != "image":
            raise ValueError("unsupported finalized request media family")
        return (
            ImageOccurrence(
                width=dimensions[0] if dimensions is not None else None,
                height=dimensions[1] if dimensions is not None else None,
                detail=detail,
                remote=False,
                encoded_bytes=encoded_bytes,
            ),
            "inline_image",
        )
    lowered = value.strip().lower()
    if lowered.startswith(("http://", "https://")):
        return (
            ImageOccurrence(
                width=None,
                height=None,
                detail=detail,
                remote=True,
                encoded_bytes=0,
            ),
            "remote_image",
        )
    raise ValueError("unsupported finalized request image value")


def _price_image_occurrences(
    images: list[tuple[ImageOccurrence, str]],
    capabilities: ModelCapabilities | None,
) -> tuple[int, frozenset[str]]:
    if not images:
        return 0, frozenset()
    occurrences = [occurrence for occurrence, _source in images]
    kinds = frozenset(source for _occurrence, source in images)
    pricing = capabilities.image_pricing if capabilities is not None else None
    if pricing is not None:
        prices = pricing.price(occurrences)
        if len(prices) != len(occurrences):
            raise ValueError("image pricing returned an invalid occurrence count")
        if any(price.visual_tokens < 0 for price in prices):
            raise ValueError("image pricing returned a negative allowance")
        return sum(price.visual_tokens for price in prices), kinds
    units = 0
    for occurrence in occurrences:
        if occurrence.remote:
            units += MAX_IMAGE_TOKENS
        elif occurrence.width is not None and occurrence.height is not None:
            if occurrence.width > 0 and occurrence.height > 0:
                units += estimate_image_tokens(occurrence.width, occurrence.height)
            else:
                units += unknown_image_tokens(occurrence.encoded_bytes)
        else:
            units += unknown_image_tokens(occurrence.encoded_bytes)
    return units, kinds


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


def _finish_image_batch(
    canonical: object,
    units: int,
    kinds: frozenset[str],
    images: list[tuple[ImageOccurrence, str]],
    capabilities: ModelCapabilities,
    root: bool,
) -> tuple[object, int, frozenset[str]]:
    if not root:
        return canonical, units, kinds
    image_units, image_kinds = _price_image_occurrences(images, capabilities)
    return canonical, units + image_units, kinds | image_kinds


def _canonicalize_finalized_value(
    value: object,
    identity: TransportIdentity,
    path: tuple[str, ...] = (),
    media_context: str | None = None,
    media_detail: str | None = None,
    capabilities: ModelCapabilities | None = None,
    images: list[tuple[ImageOccurrence, str]] | None = None,
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
    if capabilities is None:
        capabilities = default_model_capabilities(
            identity.provider,
            identity.model,
            context_window_tokens=0,
        )
    root = images is None
    if images is None:
        images = []
    if isinstance(value, bytes):
        kind = _bedrock_binary_kind(identity, path, media_context)
        if kind is None:
            raise ValueError("unsupported binary finalized request value")
        if kind == "bedrock_binary_image":
            dimensions = image_dimensions(value)
            images.append((
                ImageOccurrence(
                    width=dimensions[0] if dimensions is not None else None,
                    height=dimensions[1] if dimensions is not None else None,
                    detail=media_detail,
                    remote=False,
                    encoded_bytes=len(value),
                ),
                "bedrock_binary_media",
            ))
            return _finish_image_batch(
                {"__nanobot_bedrock_binary_bytes__": len(value)},
                0,
                frozenset({"bedrock_binary_media"}),
                images,
                capabilities,
                root,
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
            measured_image: tuple[ImageOccurrence, str] | None = None
            if block_kind in {"openai_chat_image", "responses_image"}:
                if key == "image_url" and isinstance(item, str):
                    detail = (
                        _chat_image_detail(item)
                        if block_kind == "openai_chat_image"
                        else _image_detail(mapping, item)
                    )
                    measured_image = _measured_image_string(item, detail)
                elif block_kind == "responses_image" and key == "file_id" and isinstance(item, str):
                    measured_image = (
                        ImageOccurrence(
                            width=None,
                            height=None,
                            detail=_image_detail(mapping, item),
                            remote=True,
                            encoded_bytes=0,
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
                    measured_image = _measured_image_string(item, selected_detail)
            elif media_context == "anthropic_image_source":
                source_type = mapping.get("type")
                if source_type == "url" and key == "url" and isinstance(item, str):
                    measured_image = _measured_image_string(item, None)
                elif source_type == "base64" and key == "data" and isinstance(item, str):
                    measured_image = _measured_image_string(
                        item, None, _media_mime_hint(mapping),
                    )
            if measured_image is not None:
                images.append(measured_image)
                canonical[key] = {"__nanobot_inline_media__": "image"}
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
                capabilities,
                images,
            )
            canonical[key] = converted
            supplemental_units += item_units
            measurement_kinds.update(item_kinds)
        return _finish_image_batch(
            canonical,
            supplemental_units,
            frozenset(measurement_kinds),
            images,
            capabilities,
            root,
        )
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
                capabilities,
                images,
            )
            canonical_list.append(converted)
            supplemental_units += item_units
            measurement_kinds.update(item_kinds)
        return _finish_image_batch(
            canonical_list,
            supplemental_units,
            frozenset(measurement_kinds),
            images,
            capabilities,
            root,
        )
    if value is None or isinstance(value, str | int | float | bool):
        return _finish_image_batch(
            value,
            0,
            frozenset(),
            images,
            capabilities,
            root,
        )
    raise ValueError("unsupported finalized provider request value")


def _text_token_measurement(
    encoded: bytes,
    identity: TransportIdentity,
    policy: RequestBudgetPolicy,
) -> tuple[int, str]:
    """Estimate text tokens, failing closed to the older byte ceiling.

    Unknown model names and unavailable encodings use UTF-8 bytes, which are a
    conservative ceiling that cannot assign a nearby model's vocabulary to an
    arbitrary alias.
    """
    if policy == ESTIMATED_COMPAT_POLICY:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = len(encoding.encode(encoded.decode("utf-8"), disallowed_special=()))
            return max(1, math.ceil(tokens * TOKEN_ESTIMATE_SAFETY_RATIO)), (
                ESTIMATED_COMPAT_POLICY
            )
        except Exception:
            return max(1, len(encoded)), (
                "estimated_compat_unavailable_utf8_byte_ceiling"
            )
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
    policy: RequestBudgetPolicy,
    capabilities: ModelCapabilities | None = None,
) -> tuple[int, str, int | None]:
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
            body, identity, capabilities=capabilities,
        )
        request_json = (
            json.dumps(
                body,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            if policy == ESTIMATED_COMPAT_POLICY
            else b""
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
    text_units, source = _text_token_measurement(encoded, identity, policy)
    for kind in sorted(measurement_kinds):
        source += f"+{kind}"
    return (
        max(1, text_units + supplemental_units),
        source,
        len(request_json) if policy == ESTIMATED_COMPAT_POLICY else None,
    )


def _estimated_compat_byte_limit(max_request_bytes: int | None) -> int:
    raw = cast(object, max_request_bytes)
    if raw is None:
        return DEFAULT_ESTIMATED_COMPAT_MAX_REQUEST_BYTES
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError("provider request byte limit is invalid")
    return raw


def measure_finalized_request(
    identity: TransportIdentity,
    body: Mapping[str, object],
    limits: ProviderRequestLimits | None,
    *,
    completion_reserve_tokens: int | None = None,
    policy: RequestBudgetPolicy = "strict",
    capabilities: ModelCapabilities | None = None,
    max_request_bytes: int | None = None,
) -> TransportMeasurement:
    """Measure a detached finalized body against the selected transport limit."""
    if not isinstance(cast(object, identity), TransportIdentity):
        raise ValueError("transport identity is invalid")
    if not isinstance(cast(object, body), Mapping):
        raise ValueError("finalized provider request is invalid")
    if limits is not None and not isinstance(cast(object, limits), ProviderRequestLimits):
        raise ValueError("provider request limits are invalid")
    if policy not in {"strict", ESTIMATED_COMPAT_POLICY}:
        raise ValueError("provider request measurement policy is invalid")
    if policy == "strict" and limits is None:
        raise ValueError("provider request limits are invalid")
    if policy == ESTIMATED_COMPAT_POLICY:
        if identity.transport not in {
            "openai_chat_completions", "openai_chat_completions_stream",
        }:
            raise ValueError(
                "estimated compatibility policy is unsupported for this transport"
            )
        byte_limit = _estimated_compat_byte_limit(max_request_bytes)
    else:
        byte_limit = None
    if completion_reserve_tokens is not None and (
        isinstance(completion_reserve_tokens, bool)
        or not isinstance(cast(object, completion_reserve_tokens), int)
        or completion_reserve_tokens < 0
    ):
        raise ValueError("finalized completion reserve is invalid")
    concrete_reserve = max(_completion_reserve(body), completion_reserve_tokens or 0)
    configured_reserve = limits.completion_reserve_tokens if limits is not None else 0
    reserve = max(concrete_reserve, completion_reserve_tokens or 0, configured_reserve or 0)
    total = limits.effective_total_tokens if limits is not None else None
    safety = limits.safety_buffer_tokens if limits is not None else 0
    input_limit = max(0, total - reserve - safety) if total is not None else None
    if capabilities is None:
        capabilities = default_model_capabilities(
            identity.provider,
            identity.model,
            context_window_tokens=total or 0,
            max_output_tokens=reserve,
        )
    input_tokens, source, measured_bytes = _serialized_input_measurement(
        identity, body, policy, capabilities,
    )
    return TransportMeasurement(
        identity=identity,
        input_tokens=input_tokens,
        completion_reserve_tokens=reserve,
        safety_buffer_tokens=safety,
        total_limit_tokens=total,
        input_limit_tokens=input_limit,
        source=source,
        measured_bytes=measured_bytes,
        limit_bytes=byte_limit,
        policy=policy,
    )


def validate_finalized_request(
    identity: TransportIdentity,
    body: Mapping[str, object],
    limits: ProviderRequestLimits | None,
    *,
    completion_reserve_tokens: int | None = None,
    policy: RequestBudgetPolicy = "strict",
    capabilities: ModelCapabilities | None = None,
    max_request_bytes: int | None = None,
) -> TransportMeasurement:
    """Return a valid measurement or raise without exposing request content."""
    try:
        measurement = measure_finalized_request(
            identity,
            body,
            limits,
            completion_reserve_tokens=completion_reserve_tokens,
            policy=policy,
            capabilities=capabilities,
            max_request_bytes=max_request_bytes,
        )
    except ValueError as exc:
        if policy == ESTIMATED_COMPAT_POLICY:
            raise ProviderRequestRejected(reason="invalid_request_envelope") from exc
        raise FinalizedRequestBudgetExceeded() from exc
    if (
        measurement.limit_bytes is not None
        and measurement.measured_bytes is not None
        and measurement.measured_bytes > measurement.limit_bytes
    ):
        raise RequestPayloadBytesExceeded(
            measurement.measured_bytes,
            measurement.limit_bytes,
        )
    if not measurement.fits:
        reason = "local_estimated_admission" if policy == ESTIMATED_COMPAT_POLICY else (
            "estimated_admission"
        )
        raise FinalizedRequestBudgetExceeded(measurement, reason=reason)
    return measurement
