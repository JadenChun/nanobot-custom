"""Opaque rotating device credentials for the remote owner (M08)."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Callable

from nanobot.remote.protocol import (
    ACCESS_TTL_SECONDS,
    OWNER_SCOPE,
    REFRESH_IDLE_SECONDS,
    validate_scopes,
)
from nanobot.remote.store import RemoteAccessStore, RemoteDevice, StoreError


class AuthError(RuntimeError):
    """Base class for credential failures."""


class AuthRejectedError(AuthError):
    """A credential is missing, expired, revoked or out of scope."""


class AuthConflictError(AuthError):
    """A device already exists or the owner refused a duplicate operation."""


@dataclass(frozen=True)
class AccessGrant:
    installation_id: str
    scopes: tuple[str, ...]
    expires_at: float


@dataclass(frozen=True)
class IssuedPair:
    access_token: str
    refresh_token: str
    expires_in: int
    refresh_expires_in: int
    scopes: tuple[str, ...]


class TokenService:
    """In-memory access grants plus durable hashed refresh credentials."""

    def __init__(self, store: RemoteAccessStore, *, clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.clock = clock
        self._lock = threading.RLock()
        self._access: dict[str, AccessGrant] = {}

    def issue(
        self, installation_id: str, scopes: tuple[str, ...], label: str = "mobile"
    ) -> IssuedPair:
        validate_scopes(scopes)
        with self._lock:
            state = self.store.load()
            if state is None or not state.enabled:
                raise AuthRejectedError("remote access is disabled")
            if self.store.get_device(installation_id) is not None:
                raise AuthConflictError("installation already has a device record")
            access_token = f"nbm_at_{secrets.token_urlsafe(32)}"
            refresh_token = f"nbm_rt_{secrets.token_urlsafe(32)}"
            try:
                self.store.add_device(
                    installation_id=installation_id,
                    label=label,
                    scopes=scopes,
                    refresh_token=refresh_token,
                )
            except StoreError as exc:
                raise AuthConflictError(str(exc)) from exc
            expires_at = self.clock() + ACCESS_TTL_SECONDS
            self._access[_token_hash(access_token)] = AccessGrant(
                installation_id=installation_id,
                scopes=tuple(scopes),
                expires_at=expires_at,
            )
            return IssuedPair(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=ACCESS_TTL_SECONDS,
                refresh_expires_in=_refresh_expires_in(scopes),
                scopes=tuple(scopes),
            )

    def issue_device(
        self,
        installation_id: str,
        scopes: tuple[str, ...],
        *,
        device_fingerprint: str,
        device_generation: int,
        origin: str = "",
        label: str = "mobile",
    ) -> IssuedPair:
        """Issue one proof-bound access grant without a bearer refresh authority."""
        validate_scopes(scopes)
        if not device_fingerprint or device_generation < 1:
            raise AuthRejectedError("device proof binding is invalid")
        with self._lock:
            state = self.store.load()
            if state is None or not state.enabled:
                raise AuthRejectedError("remote access is disabled")
            if self.store.get_device(installation_id) is not None:
                raise AuthConflictError("installation already has a device record")
            access_token = f"nbm_at_{secrets.token_urlsafe(32)}"
            try:
                self.store.add_device(
                    installation_id=installation_id,
                    label=label,
                    scopes=scopes,
                    refresh_token="",
                    refresh_expires_at=0.0,
                )
            except StoreError as exc:
                raise AuthConflictError(str(exc)) from exc
            self.store.set_device_proof(
                installation_id,
                fingerprint=device_fingerprint,
                generation=device_generation,
                origin=origin,
            )
            expires_at = self.clock() + ACCESS_TTL_SECONDS
            self._access[_token_hash(access_token)] = AccessGrant(
                installation_id=installation_id,
                scopes=tuple(scopes),
                expires_at=expires_at,
            )
            return IssuedPair(
                access_token=access_token,
                refresh_token="",
                expires_in=ACCESS_TTL_SECONDS,
                refresh_expires_in=0,
                scopes=tuple(scopes),
            )

    def reissue_device(self, installation_id: str) -> IssuedPair:
        """Issue a fresh access token for an already proof-bound device."""
        with self._lock:
            state = self.store.load()
            if state is None or not state.enabled:
                raise AuthRejectedError("remote access is disabled")
            device = state.devices.get(installation_id)
            if device is None or device.revoked:
                raise AuthRejectedError("installation is revoked")
            if self.store.get_device_proof(installation_id) is None:
                raise AuthRejectedError("installation has no device proof binding")
            access_token = f"nbm_at_{secrets.token_urlsafe(32)}"
            expires_at = self.clock() + ACCESS_TTL_SECONDS
            self._remove_installation_grants(installation_id)
            self._access[_token_hash(access_token)] = AccessGrant(
                installation_id=installation_id,
                scopes=tuple(device.scopes),
                expires_at=expires_at,
            )
            return IssuedPair(
                access_token=access_token,
                refresh_token="",
                expires_in=ACCESS_TTL_SECONDS,
                refresh_expires_in=0,
                scopes=tuple(device.scopes),
            )

    def validate_access(self, access_token: str) -> AccessGrant:
        with self._lock:
            self._purge_expired()
            grant = self._access.get(_token_hash(access_token))
            if grant is None:
                raise AuthRejectedError("access credential is not current")
            state = self.store.load()
            if state is None or not state.enabled:
                raise AuthRejectedError("remote access is disabled")
            device = state.devices.get(grant.installation_id)
            if device is None or device.revoked:
                self._access.pop(_token_hash(access_token), None)
                raise AuthRejectedError("installation is revoked")
            if self.clock() >= grant.expires_at:
                self._access.pop(_token_hash(access_token), None)
                raise AuthRejectedError("access credential has expired")
            if not device.scopes:
                raise AuthRejectedError("installation has no scopes")
            return AccessGrant(
                installation_id=grant.installation_id,
                scopes=tuple(device.scopes),
                expires_at=grant.expires_at,
            )

    def refresh(self, installation_id: str, refresh_token: str) -> IssuedPair:
        with self._lock:
            device = self.store.validate_refresh(installation_id, refresh_token)
            if device is None:
                raise AuthRejectedError("refresh credential is not current")
            access_token = f"nbm_at_{secrets.token_urlsafe(32)}"
            next_refresh = f"nbm_rt_{secrets.token_urlsafe(32)}"
            try:
                rotated = self.store.rotate_refresh(installation_id, next_refresh)
            except StoreError as exc:
                raise AuthRejectedError(str(exc)) from exc
            self._remove_installation_grants(installation_id)
            self._access[_token_hash(access_token)] = AccessGrant(
                installation_id=installation_id,
                scopes=tuple(rotated.scopes),
                expires_at=self.clock() + ACCESS_TTL_SECONDS,
            )
            return IssuedPair(
                access_token=access_token,
                refresh_token=next_refresh,
                expires_in=ACCESS_TTL_SECONDS,
                refresh_expires_in=_device_refresh_expires_in(rotated, self.clock()),
                scopes=tuple(rotated.scopes),
            )

    def revoke(self, installation_id: str) -> bool:
        with self._lock:
            try:
                device = self.store.revoke_device(installation_id)
            except StoreError as exc:
                raise AuthRejectedError(str(exc)) from exc
            self._remove_installation_grants(installation_id)
            return device.revoked

    def logout(self, access_token: str) -> bool:
        with self._lock:
            grant = self._access.get(_token_hash(access_token))
            if grant is None:
                raise AuthRejectedError("access credential is not current")
            return self.revoke(grant.installation_id)

    def _remove_installation_grants(self, installation_id: str) -> None:
        for key, grant in list(self._access.items()):
            if grant.installation_id == installation_id:
                self._access.pop(key, None)

    def _purge_expired(self) -> None:
        now = self.clock()
        for key, grant in list(self._access.items()):
            if now >= grant.expires_at:
                self._access.pop(key, None)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _refresh_expires_in(scopes: tuple[str, ...]) -> int:
    """Owner sessions live until revoke; legacy chat-only keeps the idle window."""

    return 0 if OWNER_SCOPE in scopes else REFRESH_IDLE_SECONDS


def _device_refresh_expires_in(device: RemoteDevice, now: float) -> int:
    if device.refresh_expires_at == 0:
        return 0
    remaining = int(device.refresh_expires_at - now)
    return max(1, min(REFRESH_IDLE_SECONDS, remaining))


def constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))
