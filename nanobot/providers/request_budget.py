"""Deterministic hard limits for finalized provider transport requests.

The logical prompt reducer works on Chat-style history.  This module is the
last boundary before a provider SDK receives a concrete request body, so it
counts the fully converted body, including provider extras and tool schemas.
It intentionally has no provider imports and fails closed for values that
cannot be represented deterministically.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import cast

DEFAULT_FINALIZED_REQUEST_SAFETY_BUFFER = 1024
_BEDROCK_BINARY_MEDIA_KEYS = frozenset({"image", "document", "video"})
_BEDROCK_BINARY_FRAMING_BYTES = 64


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


def _supported_bedrock_binary_kind(
    identity: TransportIdentity,
    path: tuple[str, ...],
) -> str | None:
    """Return the documented Bedrock Converse binary shape at ``path``."""
    if identity.provider != "bedrock":
        return None
    if len(path) >= 3 and path[-3] in _BEDROCK_BINARY_MEDIA_KEYS and path[-2:] == (
        "source",
        "bytes",
    ):
        return "bedrock_binary_media"
    if (
        len(path) >= 6
        and path[0] == "messages"
        and path[-4] == "content"
        and path[-3].isdigit()
        and path[-2:] == ("reasoningContent", "redactedContent")
    ):
        return "bedrock_redacted_reasoning"
    return None


def _canonicalize_finalized_value(
    value: object,
    identity: TransportIdentity,
    path: tuple[str, ...] = (),
) -> tuple[object, int, frozenset[str]]:
    """Return canonical JSON data and Bedrock binary transport bytes.

    Bedrock Converse accepts raw ``bytes`` only below documented media sources
    and replayed ``reasoningContent.redactedContent`` blocks. Their content is
    replaced with a length-only marker so measurement never retains binary
    data. Every other non-JSON value remains opaque and therefore fails closed.
    """
    if isinstance(value, bytes):
        kind = _supported_bedrock_binary_kind(identity, path)
        if kind is None:
            raise ValueError("unsupported binary finalized request value")
        # AWS serializes binary SDK values separately from JSON. Reserve its
        # worst-case Base64 wire expansion plus fixed binary framing; this is
        # a conservative byte ceiling, not a provider tokenization estimate.
        transport_bytes = 4 * ((len(value) + 2) // 3) + _BEDROCK_BINARY_FRAMING_BYTES
        return {"__nanobot_bedrock_binary_bytes__": len(value)}, transport_bytes, frozenset({kind})
    if isinstance(value, dict):
        canonical: dict[str, object] = {}
        binary_transport_bytes = 0
        binary_kinds: set[str] = set()
        for key, item in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                raise ValueError("finalized provider request keys must be strings")
            converted, item_binary_bytes, item_binary_kinds = _canonicalize_finalized_value(
                item,
                identity,
                (*path, key),
            )
            canonical[key] = converted
            binary_transport_bytes += item_binary_bytes
            binary_kinds.update(item_binary_kinds)
        return canonical, binary_transport_bytes, frozenset(binary_kinds)
    if isinstance(value, (list, tuple)):
        canonical_list: list[object] = []
        binary_transport_bytes = 0
        binary_kinds = set()
        for index, item in enumerate(cast(list[object] | tuple[object, ...], value)):
            converted, item_binary_bytes, item_binary_kinds = _canonicalize_finalized_value(
                item,
                identity,
                (*path, str(index)),
            )
            canonical_list.append(converted)
            binary_transport_bytes += item_binary_bytes
            binary_kinds.update(item_binary_kinds)
        return canonical_list, binary_transport_bytes, frozenset(binary_kinds)
    if value is None or isinstance(value, str | int | float | bool):
        return value, 0, frozenset()
    raise ValueError("unsupported finalized provider request value")


def _serialized_input_measurement(
    identity: TransportIdentity,
    body: Mapping[str, object],
) -> tuple[int, str]:
    """Count a content-free canonical body with a conservative byte ceiling.

    UTF-8 bytes are used as token-equivalent units instead of dividing by four:
    a supported text tokenizer can consume no more than one token per input
    byte, while a four-byte heuristic can undercount Unicode and adversarial
    tokenization.  This is intentionally conservative and does not claim an
    exact provider tokenizer result. Supported Bedrock media and redacted
    reasoning binary fields add their Base64-size ceiling and fixed framing
    allowance without inferring provider tokenization from file contents.
    """
    if not isinstance(body, dict):
        raise ValueError("finalized provider request is not a JSON object")
    try:
        canonical, binary_transport_bytes, binary_kinds = _canonicalize_finalized_value(body, identity)
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("finalized provider request cannot be measured") from exc
    source = "canonical_json_utf8_bytes"
    for kind in sorted(binary_kinds):
        source += f"+{kind}"
    return max(1, len(encoded) + binary_transport_bytes), source


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
