"""Typed owner delegation over the independent remote-access owner (M30).

The owner process is the only authority for device identity. A gateway may ask
for a named delegated administration operation by presenting the opaque device
access credential; it never supplies identity, locality, native capability or
scope assertions. The owner validates the credential and returns a typed,
short-lived, single-use admission bound to device/operation/digest.

Local-only enrollment actions (initial enable/re-enable, new-device challenge
creation/approval and scope upgrade) are intentionally not members of
``DelegatedAdminOperation``: callers cannot request them over IPC at all.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, cast

from nanobot.remote.auth import AuthRejectedError, TokenService
from nanobot.remote.protocol import OWNER_SCOPE, validate_installation_id
from nanobot.remote.store import RemoteAccessStore

MAX_ARGUMENT_BYTES = 4 * 1024
MAX_ADMISSION_TTL_SECONDS = 30.0
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

# Fields that would let a caller assert authority it does not have.
_FORBIDDEN_ARGUMENT_KEYS = frozenset(
    {
        "is_local",
        "is_native",
        "local",
        "native",
        "principal",
        "scope",
        "scopes",
        "installation_id",
        "device_id",
        "admin_token",
        "access_token",
        "refresh_token",
    }
)


class DelegationError(RuntimeError):
    """A delegated operation is malformed, unauthorized or no longer current."""

    def __init__(self, code: str, message: str, *, status: int = 403) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class DelegatedAdminOperation(str, Enum):
    """The complete enumerated surface a gateway may request from the owner."""

    DEVICE_LIST = "device.list"
    DEVICE_RENAME = "device.rename"
    DEVICE_REVOKE = "device.revoke"
    DEVICE_SELF_REVOKE = "device.self_revoke"
    ENTRY_STATUS = "entry.status"
    ENTRY_UPDATE = "entry.update"
    ENTRY_DISABLE = "entry.disable"
    GRANT_PROPOSAL = "grant.proposal"
    GRANT_COMMIT = "grant.commit"
    GRANT_RELEASE = "grant.release"
    FULL_PROPOSAL = "full.proposal"
    FULL_COMMIT = "full.commit"
    FULL_RELEASE = "full.release"


# Operations that only a full owner capability may perform.
_OWNER_ONLY = frozenset(set(DelegatedAdminOperation) - {DelegatedAdminOperation.DEVICE_SELF_REVOKE})

# Bounded argument keys per operation. Unknown keys are rejected, not ignored.
_ALLOWED_ARGUMENTS: dict[DelegatedAdminOperation, frozenset[str]] = {
    DelegatedAdminOperation.DEVICE_LIST: frozenset(),
    DelegatedAdminOperation.DEVICE_RENAME: frozenset({"target_installation_id", "label"}),
    DelegatedAdminOperation.DEVICE_REVOKE: frozenset({"target_installation_id"}),
    DelegatedAdminOperation.DEVICE_SELF_REVOKE: frozenset(),
    DelegatedAdminOperation.ENTRY_STATUS: frozenset(),
    DelegatedAdminOperation.ENTRY_UPDATE: frozenset({"entry_id", "settings"}),
    DelegatedAdminOperation.ENTRY_DISABLE: frozenset({"entry_id"}),
    DelegatedAdminOperation.GRANT_PROPOSAL: frozenset({"chat_id", "target", "extent"}),
    DelegatedAdminOperation.GRANT_COMMIT: frozenset({"proposal_id"}),
    DelegatedAdminOperation.GRANT_RELEASE: frozenset({"proposal_id"}),
    DelegatedAdminOperation.FULL_PROPOSAL: frozenset({"chat_id"}),
    DelegatedAdminOperation.FULL_COMMIT: frozenset({"proposal_id"}),
    DelegatedAdminOperation.FULL_RELEASE: frozenset({"proposal_id"}),
}


@dataclass(frozen=True)
class AuthenticatedDeviceContext:
    """Owner-built identity; never constructed from caller-supplied fields."""

    installation_id: str
    principal: str
    scopes: tuple[str, ...]
    record_revision: int
    expires_at: float

    @property
    def is_owner(self) -> bool:
        return OWNER_SCOPE in self.scopes


@dataclass(frozen=True)
class DelegationAdmission:
    """Owner-issued internal handle; must never cross owner IPC to the phone."""

    handle: str
    installation_id: str
    operation: DelegatedAdminOperation
    digest: str
    record_revision: int
    expires_at: float


def _canonical_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_arguments(
    operation: DelegatedAdminOperation,
    arguments: object,
    *,
    store: RemoteAccessStore | None = None,
    device_installation_id: str | None = None,
) -> dict[str, Any]:
    """Return a bounded copy of the typed arguments for ``operation``."""
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise DelegationError("invalid_arguments", "arguments must be an object", status=400)
    payload = cast(dict[str, Any], arguments)
    allowed = _ALLOWED_ARGUMENTS[operation]
    unknown = set(payload) - allowed
    if unknown:
        raise DelegationError(
            "invalid_arguments",
            f"arguments contain unlisted fields: {sorted(unknown)[0]}",
            status=400,
        )
    forbidden = set(payload) & _FORBIDDEN_ARGUMENT_KEYS
    if forbidden:
        raise DelegationError(
            "invalid_arguments",
            f"arguments may not assert authority: {sorted(forbidden)[0]}",
            status=400,
        )
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_ARGUMENT_BYTES:
        raise DelegationError("arguments_too_large", "arguments exceed the bounded size", status=413)

    def required_string(key: str, *, max_length: int = 128) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip() or len(value) > max_length:
            raise DelegationError("invalid_arguments", f"{key} is required", status=400)
        return value

    if operation is DelegatedAdminOperation.DEVICE_RENAME:
        required_string("target_installation_id")
        validate_installation_id(required_string("target_installation_id"))
        required_string("label", max_length=64)
    elif operation is DelegatedAdminOperation.DEVICE_REVOKE:
        required_string("target_installation_id")
        validate_installation_id(required_string("target_installation_id"))
    elif operation is DelegatedAdminOperation.ENTRY_UPDATE:
        required_string("entry_id", max_length=128)
        settings = payload.get("settings")
        if not isinstance(settings, dict):
            raise DelegationError("invalid_arguments", "settings must be an object", status=400)
        settings_payload = cast(dict[str, Any], settings)
        if len(settings_payload) > 8:
            raise DelegationError("invalid_arguments", "too many settings", status=400)
        for key, value in settings_payload.items():
            if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key):
                raise DelegationError("invalid_arguments", "invalid setting key", status=400)
            if not isinstance(value, (str, bool, int)):
                raise DelegationError("invalid_arguments", "invalid setting value", status=400)
    elif operation is DelegatedAdminOperation.ENTRY_DISABLE:
        required_string("entry_id", max_length=128)
    elif operation in {DelegatedAdminOperation.GRANT_PROPOSAL}:
        required_string("chat_id", max_length=128)
        required_string("target", max_length=4096)
        if payload.get("extent") not in {"read", "write"}:
            raise DelegationError("invalid_arguments", "extent must be read or write", status=400)
    elif operation in {DelegatedAdminOperation.FULL_PROPOSAL}:
        required_string("chat_id", max_length=128)
    elif operation in {
        DelegatedAdminOperation.GRANT_COMMIT,
        DelegatedAdminOperation.GRANT_RELEASE,
        DelegatedAdminOperation.FULL_COMMIT,
        DelegatedAdminOperation.FULL_RELEASE,
    }:
        required_string("proposal_id", max_length=128)
    elif store is not None and device_installation_id is not None:
        # Argument-free operations still bind the store for symmetry.
        del store, device_installation_id
    return dict(payload)


class OwnerDelegationService:
    """Owner-side authentication and admission minting for delegated operations."""

    def __init__(
        self,
        store: RemoteAccessStore,
        tokens: TokenService,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.tokens = tokens
        self.clock = clock
        self._lock = threading.RLock()
        self._admissions: dict[str, DelegationAdmission] = {}

    def authenticate(self, access_token: str) -> AuthenticatedDeviceContext:
        if not isinstance(access_token, str) or not access_token:
            raise DelegationError("missing_access", "access credential is required", status=401)
        try:
            grant = self.tokens.validate_access(access_token)
        except AuthRejectedError as exc:
            raise DelegationError("invalid_access", str(exc), status=401) from exc
        device = self.store.get_device(grant.installation_id)
        if device is None or device.revoked:
            raise DelegationError("device_revoked", "installation is revoked", status=401)
        snapshot = self.store.snapshot()
        return AuthenticatedDeviceContext(
            installation_id=device.installation_id,
            principal=f"owner:{device.installation_id}",
            scopes=tuple(device.scopes),
            record_revision=int(snapshot["revision"]),
            expires_at=grant.expires_at,
        )

    def authorize(
        self,
        *,
        access_token: str,
        operation: str,
        request_id: str,
        digest: str,
        arguments: object = None,
    ) -> tuple[AuthenticatedDeviceContext, DelegationAdmission, dict[str, Any]]:
        try:
            op = DelegatedAdminOperation(operation)
        except ValueError as exc:
            raise DelegationError(
                "unknown_operation",
                "operation is not an enumerated delegated action",
                status=404,
            ) from exc
        if not isinstance(request_id, str) or not _REQUEST_ID_RE.fullmatch(request_id):
            raise DelegationError("invalid_request_id", "request_id is invalid", status=400)
        if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
            raise DelegationError("invalid_digest", "digest must be a lowercase SHA-256", status=400)
        context = self.authenticate(access_token)
        if op in _OWNER_ONLY and not context.is_owner:
            raise DelegationError("owner_required", "owner capability is required", status=403)
        payload = validate_arguments(
            op,
            arguments,
            store=self.store,
            device_installation_id=context.installation_id,
        )
        expected_digest = _canonical_digest(
            {
                "operation": op.value,
                "request_id": request_id,
                "arguments": payload,
                "installation_id": context.installation_id,
            }
        )
        if not secrets.compare_digest(expected_digest, digest):
            raise DelegationError("digest_mismatch", "operation digest does not match", status=409)
        now = self.clock()
        admission = DelegationAdmission(
            handle=secrets.token_urlsafe(24),
            installation_id=context.installation_id,
            operation=op,
            digest=digest,
            record_revision=context.record_revision,
            expires_at=now + MAX_ADMISSION_TTL_SECONDS,
        )
        with self._lock:
            self._purge_expired()
            self._admissions[admission.handle] = admission
        return context, admission, payload

    def consume(
        self,
        *,
        handle: str,
        installation_id: str,
        operation: str,
        digest: str,
    ) -> AuthenticatedDeviceContext:
        with self._lock:
            self._purge_expired()
            admission = self._admissions.pop(handle, None)
        if admission is None:
            raise DelegationError("admission_missing", "admission is not current", status=409)
        if (
            admission.installation_id != installation_id
            or admission.operation.value != operation
            or not secrets.compare_digest(admission.digest, digest)
        ):
            raise DelegationError("admission_mismatch", "admission is bound to another operation", status=409)
        context = self.authenticate_device(admission.installation_id)
        if context.record_revision != admission.record_revision:
            raise DelegationError("authority_changed", "device authority changed", status=409)
        return context

    def authenticate_device(self, installation_id: str) -> AuthenticatedDeviceContext:
        device = self.store.get_device(installation_id)
        if device is None or device.revoked:
            raise DelegationError("device_revoked", "installation is revoked", status=401)
        snapshot = self.store.snapshot()
        return AuthenticatedDeviceContext(
            installation_id=device.installation_id,
            principal=f"owner:{device.installation_id}",
            scopes=tuple(device.scopes),
            record_revision=int(snapshot["revision"]),
            expires_at=self.clock(),
        )

    def rename_device(self, installation_id: str, label: str) -> dict[str, Any]:
        device = self.store.rename_device(installation_id, label)
        return {"installation_id": device.installation_id, "label": device.label}

    def revoke_device(self, installation_id: str) -> dict[str, Any]:
        device = self.store.revoke_device(installation_id)
        return {"installation_id": device.installation_id, "revoked": device.revoked}

    def _purge_expired(self) -> None:
        now = self.clock()
        for handle, admission in list(self._admissions.items()):
            if now >= admission.expires_at:
                self._admissions.pop(handle, None)


def current_digest(operation: str, request_id: str, arguments: dict[str, Any], installation_id: str) -> str:
    """Helper shared by tests and gateway clients; never trusts caller identity."""
    return _canonical_digest(
        {
            "operation": operation,
            "request_id": request_id,
            "arguments": arguments,
            "installation_id": installation_id,
        }
    )
