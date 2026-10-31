"""Versioned, host-neutral contracts for the Nanobot mobile client (M06).

This module deliberately contains no gateway, agent, session or permission
implementation. It is the single source of truth for wire versions, route
scopes, limits and mutation/replay semantics consumed by the remote owner and
the mobile client.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MOBILE_PROTOCOL_VERSION = 1
BRIDGE_VERSION = 1
ACCESS_TTL_SECONDS = 10 * 60
CHALLENGE_TTL_SECONDS = 120
REFRESH_IDLE_SECONDS = 30 * 24 * 60 * 60
MAX_WS_FRAME_BYTES = 64 * 1024
MAX_ATTACHMENT_BYTES = 256 * 1024
MAX_MEDIA_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_REQUEST_BODY_BYTES = 1 * 1024 * 1024

SUPPORTED_SCOPES = frozenset(
    {
        "chat.read",
        "chat.write",
        "media.read",
        "attachment.write",
        "device.self.logout",
    }
)
MANDATORY_CAPABILITIES = frozenset({"chat"})
OPTIONAL_CAPABILITIES = frozenset({"media", "attachments", "lifecycle"})

_INSTALLATION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_PRINCIPAL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ACTION_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class ContractError(ValueError):
    """A protocol input is invalid or outside the accepted contract."""


class MutationMode(str, Enum):
    SUBMIT = "submit"
    REPLAY_ONLY = "replay_only"


class MutationDecision(str, Enum):
    APPLIED = "applied"
    REPLAYED = "replayed"
    OUTCOME_UNKNOWN = "outcome_unknown"
    DENIED = "denied"


class RouteScope(str, Enum):
    CHAT_READ = "chat.read"
    CHAT_WRITE = "chat.write"
    MEDIA_READ = "media.read"
    ATTACHMENT_WRITE = "attachment.write"
    DEVICE_SELF_LOGOUT = "device.self.logout"
    LOCAL_ADMIN = "local.admin"


class Capabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chat: bool
    media: bool = False
    attachments: bool = False
    lifecycle: bool = False

    def validate_mandatory(self) -> None:
        for capability in MANDATORY_CAPABILITIES:
            if not getattr(self, capability):
                raise ContractError(f"mandatory capability is missing: {capability}")


class HandshakeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mobile_protocol: int
    bridge_version: int
    installation_id: str
    origin: str
    capabilities: Capabilities

    @field_validator("installation_id")
    @classmethod
    def _installation(cls, value: str) -> str:
        return validate_installation_id(value)

    @field_validator("origin")
    @classmethod
    def _origin(cls, value: str) -> str:
        return validate_origin(value)

    @model_validator(mode="after")
    def _versions(self) -> "HandshakeRequest":
        if self.mobile_protocol != MOBILE_PROTOCOL_VERSION:
            raise ContractError("mobile_protocol is not supported")
        if self.bridge_version != BRIDGE_VERSION:
            raise ContractError("bridge_version is not supported")
        self.capabilities.validate_mandatory()
        return self


class HandshakeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mobile_protocol: int = MOBILE_PROTOCOL_VERSION
    gateway_generation: int = Field(ge=1)
    server_time: int
    principal: str
    scopes: tuple[str, ...]
    capabilities: Capabilities
    limits: dict[str, int]

    @field_validator("scopes")
    @classmethod
    def _scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        validate_scopes(value)
        return value


class RefreshRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    refresh_token: str = Field(min_length=16, max_length=512)
    installation_id: str

    @field_validator("installation_id")
    @classmethod
    def _installation(cls, value: str) -> str:
        return validate_installation_id(value)


class RefreshResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    access_token: str
    refresh_token: str
    expires_in: int = Field(ge=1, le=ACCESS_TTL_SECONDS)
    refresh_expires_in: int = Field(ge=1, le=REFRESH_IDLE_SECONDS)


class MutationEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    installation_id: str
    principal: str
    action: str
    digest: str
    gateway_generation: int = Field(ge=1)
    epoch: int = Field(ge=0)
    mode: MutationMode
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id")
    @classmethod
    def _request_id(cls, value: str) -> str:
        if not _REQUEST_ID_RE.fullmatch(value):
            raise ContractError("request_id is not a stable opaque identifier")
        return value

    @field_validator("installation_id")
    @classmethod
    def _installation(cls, value: str) -> str:
        return validate_installation_id(value)

    @field_validator("principal")
    @classmethod
    def _principal(cls, value: str) -> str:
        if not _PRINCIPAL_RE.fullmatch(value):
            raise ContractError("principal is invalid")
        return value

    @field_validator("action")
    @classmethod
    def _action(cls, value: str) -> str:
        if not _ACTION_RE.fullmatch(value):
            raise ContractError("action is invalid")
        return value

    @field_validator("digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ContractError("digest must be a lowercase SHA-256 hex string")
        return value


class MutationOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: MutationDecision
    request_id: str
    effect_id: str | None = None
    effect_count: int = Field(default=0, ge=0)
    http_status: int = 200


class PairingChallenge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pairing_id: str
    origin: str
    installation_id: str
    nonce: str = Field(min_length=16, max_length=256)
    label: str = Field(min_length=1, max_length=64)
    expires_at: int
    chat_consent: bool
    media_consent: bool

    @field_validator("origin")
    @classmethod
    def _origin(cls, value: str) -> str:
        return validate_origin(value)

    @field_validator("installation_id")
    @classmethod
    def _installation(cls, value: str) -> str:
        return validate_installation_id(value)


class DeviceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    installation_id: str
    label: str = Field(min_length=1, max_length=64)
    scopes: tuple[str, ...]
    created_at: int
    last_seen_at: int
    revoked_at: int | None = None

    @field_validator("scopes")
    @classmethod
    def _scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        validate_scopes(value)
        return value


@dataclass(frozen=True)
class OperationKey:
    """Immutable operation identity; access versions never replace it."""

    installation_id: str
    generation: int
    request_id: str

    def __post_init__(self) -> None:
        validate_installation_id(self.installation_id)
        if self.generation < 1:
            raise ContractError("generation must be positive")
        if not _REQUEST_ID_RE.fullmatch(self.request_id):
            raise ContractError("request_id is invalid")


@dataclass(frozen=True)
class RouteSpec:
    method: str
    path: str
    scope: str
    replay: Literal["none", "submit", "replay_only"] = "none"
    websocket: bool = False
    local_admin: bool = False


ROUTES: tuple[RouteSpec, ...] = (
    RouteSpec("GET", "/mobile/v1/handshake", "chat.read"),
    RouteSpec("POST", "/mobile/v1/bootstrap", "chat.read"),
    RouteSpec("GET", "/mobile/v1/read", "chat.read"),
    RouteSpec("POST", "/mobile/v1/refresh", "chat.read"),
    RouteSpec("POST", "/mobile/v1/mutation", "chat.write", replay="submit"),
    RouteSpec("GET", "/mobile/v1/media/{name}", "media.read"),
    RouteSpec("POST", "/mobile/v1/attachment", "attachment.write"),
    RouteSpec("GET", "/mobile/v1/status", "chat.read"),
    RouteSpec("POST", "/mobile/v1/logout", "device.self.logout"),
    RouteSpec("GET", "/mobile/v1/ws", "chat.read", websocket=True),
    RouteSpec("POST", "/remote/v1/pair/challenge", "local.admin", local_admin=True),
    RouteSpec("POST", "/remote/v1/pair/collect", "local.admin", local_admin=True),
    RouteSpec("POST", "/remote/v1/pair/approve", "local.admin", local_admin=True),
    RouteSpec("GET", "/remote/v1/devices", "local.admin", local_admin=True),
    RouteSpec("POST", "/remote/v1/devices/{id}/revoke", "local.admin", local_admin=True),
    RouteSpec("GET", "/remote/v1/status", "local.admin", local_admin=True),
)


def validate_origin(value: str) -> str:
    from urllib.parse import urlsplit

    if not value or value != value.strip():
        raise ContractError("origin must be a trimmed absolute URL")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ContractError("origin must use https with a host")
    if parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ContractError("origin must not carry credentials, path, query or fragment")
    return value.rstrip("/")


def validate_installation_id(value: str) -> str:
    if not _INSTALLATION_RE.fullmatch(value):
        raise ContractError("installation_id is invalid")
    return value


def validate_scopes(value: tuple[str, ...] | list[str]) -> None:
    unknown = set(value) - SUPPORTED_SCOPES
    if unknown:
        raise ContractError(f"unsupported scope: {sorted(unknown)[0]}")


def stable_operation_digest(principal: str, action: str, payload: dict[str, Any]) -> str:
    """Return a canonical digest independent of request/access versions."""

    if not _PRINCIPAL_RE.fullmatch(principal):
        raise ContractError("principal is invalid")
    if not _ACTION_RE.fullmatch(action):
        raise ContractError("action is invalid")
    canonical = json.dumps(
        {"principal": principal, "action": action, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def admit_mutation(
    envelope: MutationEnvelope,
    *,
    current_generation: int,
    current_epoch: int,
    record: OperationKey | None,
    record_digest: str | None,
    record_principal: str | None,
) -> MutationOutcome:
    """Apply the M12 replay rule without executing any effect.

    A new effect is admitted only for a current-epoch ``submit``. A
    ``replay_only`` request can only join an existing record. A missing or
    incompatible record for ``replay_only`` is an outcome-unknown response.
    """

    if envelope.gateway_generation != current_generation:
        return MutationOutcome(
            decision=MutationDecision.DENIED,
            request_id=envelope.request_id,
            http_status=409,
        )
    if envelope.mode is MutationMode.SUBMIT and envelope.epoch != current_epoch:
        return MutationOutcome(
            decision=MutationDecision.DENIED,
            request_id=envelope.request_id,
            http_status=409,
        )
    if record is not None:
        if (
            record.installation_id != envelope.installation_id
            or record.request_id != envelope.request_id
            or record.generation != envelope.gateway_generation
            or record_digest != envelope.digest
            or record_principal != envelope.principal
        ):
            return MutationOutcome(
                decision=MutationDecision.DENIED,
                request_id=envelope.request_id,
                http_status=409,
            )
        return MutationOutcome(
            decision=MutationDecision.REPLAYED,
            request_id=envelope.request_id,
            effect_id=envelope.request_id,
            effect_count=1,
            http_status=200,
        )
    if envelope.mode is MutationMode.REPLAY_ONLY:
        return MutationOutcome(
            decision=MutationDecision.OUTCOME_UNKNOWN,
            request_id=envelope.request_id,
            effect_count=0,
            http_status=409,
        )
    return MutationOutcome(
        decision=MutationDecision.APPLIED,
        request_id=envelope.request_id,
        effect_id=envelope.request_id,
        effect_count=1,
        http_status=200,
    )


def challenge_expired(challenge: PairingChallenge, now: int | None = None) -> bool:
    return (now if now is not None else int(time.time())) >= challenge.expires_at


def default_limits() -> dict[str, int]:
    return {
        "access_ttl_seconds": ACCESS_TTL_SECONDS,
        "challenge_ttl_seconds": CHALLENGE_TTL_SECONDS,
        "refresh_idle_seconds": REFRESH_IDLE_SECONDS,
        "max_ws_frame_bytes": MAX_WS_FRAME_BYTES,
        "max_attachment_bytes": MAX_ATTACHMENT_BYTES,
        "max_media_response_bytes": MAX_MEDIA_RESPONSE_BYTES,
        "max_request_body_bytes": MAX_REQUEST_BODY_BYTES,
    }
