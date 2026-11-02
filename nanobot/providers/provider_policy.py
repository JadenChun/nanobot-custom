"""Provider-owned runtime policy, separate from model capability facts.

A provider route declares how its requests should be accounted for, whether it
supports locally reported usage, how it signals context overflow, and which
image-pricing policy applies.  Generic agent and budget code consumes these
values instead of inferring behavior from a provider or model name.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal, cast

DEFAULT_ESTIMATED_COMPAT_MAX_REQUEST_BYTES = 64 * 1024 * 1024

ImagePricingMode = Literal["auto", "local_generic", "none"]
OverflowDetectionPolicy = Literal["strict", "compat_400"]


class RequestAccountingPolicy(StrEnum):
    """How a provider's request body is measured before dispatch."""

    STRICT = "strict"
    ESTIMATED_COMPAT = "estimated_compat"


@dataclass(frozen=True, slots=True)
class ProviderPolicy:
    """Declared runtime behavior for one provider route."""

    request_accounting: RequestAccountingPolicy = RequestAccountingPolicy.STRICT
    max_request_bytes: int | None = None
    supports_usage_anchor: bool = False
    supports_source_summary: bool = False
    overflow_detection: OverflowDetectionPolicy = "strict"
    image_pricing: ImagePricingMode = "auto"

    def __post_init__(self) -> None:
        accounting = cast(object, self.request_accounting)
        if not isinstance(accounting, RequestAccountingPolicy):
            object.__setattr__(
                self,
                "request_accounting",
                RequestAccountingPolicy(self.request_accounting),
            )
        max_request_bytes = cast(object, self.max_request_bytes)
        if max_request_bytes is not None and (
            isinstance(max_request_bytes, bool)
            or not isinstance(max_request_bytes, int)
            or max_request_bytes <= 0
        ):
            raise ValueError("max_request_bytes must be a positive integer")
        if self.overflow_detection not in {"strict", "compat_400"}:
            raise ValueError("unsupported overflow detection policy")
        if self.image_pricing not in {"auto", "local_generic", "none"}:
            raise ValueError("unsupported image pricing policy")

    def signature(self) -> tuple[object, ...]:
        """Return a primitive tuple suitable for runtime fingerprinting."""
        return (
            self.request_accounting.value,
            self.max_request_bytes,
            self.supports_usage_anchor,
            self.supports_source_summary,
            self.overflow_detection,
            self.image_pricing,
        )


ESTIMATED_COMPAT_POLICY = ProviderPolicy(
    request_accounting=RequestAccountingPolicy.ESTIMATED_COMPAT,
    max_request_bytes=DEFAULT_ESTIMATED_COMPAT_MAX_REQUEST_BYTES,
    supports_usage_anchor=True,
    supports_source_summary=True,
    overflow_detection="compat_400",
    image_pricing="local_generic",
)


def resolve_provider_policy(
    default: ProviderPolicy | None = None,
    *,
    request_accounting: str | None = None,
    max_request_bytes: int | None = None,
) -> ProviderPolicy:
    """Resolve provider config overrides over a declared provider preset.

    Selecting ``estimated_compat`` enables the compatible accounting features
    as one coherent policy.  Selecting ``strict`` disables them, so the two
    modes cannot be mixed accidentally.
    """
    base = default if default is not None else ProviderPolicy()
    accounting = (
        base.request_accounting
        if request_accounting is None
        else RequestAccountingPolicy(request_accounting)
    )
    cap = (
        max_request_bytes
        if max_request_bytes is not None
        else base.max_request_bytes
    )
    if accounting is RequestAccountingPolicy.STRICT:
        return replace(
            base,
            request_accounting=RequestAccountingPolicy.STRICT,
            max_request_bytes=None,
            supports_usage_anchor=False,
            supports_source_summary=False,
            overflow_detection="strict",
        )
    if cap is None:
        cap = DEFAULT_ESTIMATED_COMPAT_MAX_REQUEST_BYTES
    return replace(
        base,
        request_accounting=RequestAccountingPolicy.ESTIMATED_COMPAT,
        max_request_bytes=cap,
        supports_usage_anchor=True,
        supports_source_summary=True,
        overflow_detection="compat_400",
    )


__all__ = [
    "DEFAULT_ESTIMATED_COMPAT_MAX_REQUEST_BYTES",
    "ESTIMATED_COMPAT_POLICY",
    "ImagePricingMode",
    "OverflowDetectionPolicy",
    "ProviderPolicy",
    "RequestAccountingPolicy",
    "resolve_provider_policy",
]
