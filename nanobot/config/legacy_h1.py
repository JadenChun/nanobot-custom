"""Narrow compatibility validation for the retired Telegram H1 settings.

The feature is no longer part of the runtime schema. Existing configuration
files may still contain a disabled block; loading it is tolerated but the block
is removed from the validated model. Enabled or malformed blocks fail before
gateway work starts instead of being silently ignored.
"""

from __future__ import annotations

from typing import cast

from pydantic import ConfigDict, Field, StrictBool, ValidationError

from nanobot.config_base import Base

RETIRED_TELEGRAM_BUSY_KEYS = ("telegramBusy", "telegram_busy")


class RetiredH1ConfigError(ValueError):
    """A retired Telegram H1 configuration block requires operator action."""


class _RetiredBusyControlConfig(Base):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    max_concurrent: int = Field(default=2, ge=1, le=16)
    timeout_seconds: float = Field(default=3.0, gt=0, le=30, allow_inf_nan=False)
    max_calls_per_window: int = Field(default=30, ge=1, le=1000)
    window_seconds: float = Field(default=60.0, gt=0, le=3600, allow_inf_nan=False)
    max_input_bytes: int = Field(default=16384, ge=1, le=65536)
    max_output_tokens: int = Field(default=128, ge=1, le=1024)
    max_response_bytes: int = Field(default=4096, ge=1, le=16384)
    minimum_confidence: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)


class _RetiredBusyInboxConfig(Base):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    max_pending: int = Field(default=256, ge=1, le=8192)
    max_records: int = Field(default=4096, ge=1, le=65536)
    max_bytes: int = Field(default=4194304, ge=256, le=67108864)
    max_clarifications: int = Field(default=8, ge=1, le=32)


class _RetiredTelegramBusyConfig(Base):
    model_config = ConfigDict(extra="forbid")

    enabled: StrictBool = False
    session_key: str = Field(default="", max_length=256)
    owner_sender_ids: list[str] = Field(default_factory=list, max_length=64)
    owner_chat_ids: list[str] = Field(default_factory=list, max_length=64)
    control_provider: str = ""
    control_model: str = ""
    tool_scope: list[str] | None = Field(default=None, max_length=4096)
    automation_scope: list[str] | None = Field(default=None, max_length=4096)
    limits: _RetiredBusyControlConfig = Field(default_factory=_RetiredBusyControlConfig)
    inbox: _RetiredBusyInboxConfig = Field(default_factory=_RetiredBusyInboxConfig)


def _retired_message(action: str) -> str:
    return (
        f"gateway.telegramBusy is retired (legacy Telegram H1 busy-input): {action}. "
        "Remove the block or set enabled=false; no H1 routing or inbox starts."
    )


def consume_retired_telegram_busy(gateway: object) -> object:
    """Return *gateway* without the retired key, or raise a safe explanation."""
    if not isinstance(gateway, dict):
        return gateway
    values = cast(dict[str, object], gateway)
    present = [key for key in RETIRED_TELEGRAM_BUSY_KEYS if key in values]
    if not present:
        return cast(object, gateway)
    if len(present) > 1:
        raise RetiredH1ConfigError(_retired_message("duplicate camel-case and snake-case blocks"))
    key = present[0]
    try:
        parsed = _RetiredTelegramBusyConfig.model_validate(values[key])
    except ValidationError:
        raise RetiredH1ConfigError(
            _retired_message("the retired block is malformed")
        ) from None
    if parsed.enabled:
        raise RetiredH1ConfigError(
            _retired_message("the retired busy-input feature is enabled")
        )
    cleaned = values.copy()
    cleaned.pop(key)
    return cleaned


def retired_telegram_busy_error(settings: object) -> RetiredH1ConfigError | None:
    """Validate one gateway mapping and return its safe failure, if any."""
    try:
        consume_retired_telegram_busy(settings)
    except RetiredH1ConfigError as error:
        return error
    return None


__all__ = [
    "RETIRED_TELEGRAM_BUSY_KEYS",
    "RetiredH1ConfigError",
    "consume_retired_telegram_busy",
    "retired_telegram_busy_error",
]
