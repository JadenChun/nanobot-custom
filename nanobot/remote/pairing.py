"""Single-winner pairing and comparison-code flow for the remote owner (M08)."""

from __future__ import annotations

import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from nanobot.remote.auth import (
    AuthConflictError,
    AuthError,
    AuthRejectedError,
    IssuedPair,
    TokenService,
    constant_time_equal,
)
from nanobot.remote.protocol import (
    CHALLENGE_TTL_SECONDS,
    ContractError,
    PairingChallenge,
    validate_installation_id,
    validate_origin,
    validate_scopes,
)
from nanobot.remote.store import RemoteAccessStore


class PairingError(RuntimeError):
    """Base class for pairing failures."""


class PairingDisabledError(PairingError):
    pass


class PairingNotFoundError(PairingError):
    pass


class PairingExpiredError(PairingError):
    pass


class PairingRejectedError(PairingError):
    pass


class PairingBusyError(PairingError):
    pass


class PairingRateLimitedError(PairingError):
    pass


class PairingConsumedError(PairingError):
    pass


@dataclass(frozen=True)
class PairingOffer:
    pairing_id: str
    origin: str
    label: str
    nonce: str
    expires_at: int
    collector_secret: str
    comparison_code: str
    chat_consent: bool
    media_consent: bool
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class PairingPending:
    pairing_id: str
    comparison_code: str
    expires_at: int
    installation_id: str


@dataclass(frozen=True)
class PairingApproved:
    pairing_id: str
    installation_id: str
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class PairingResult:
    status: str
    pairing_id: str
    installation_id: str
    access_token: str
    refresh_token: str
    expires_in: int
    refresh_expires_in: int
    scopes: tuple[str, ...]


@dataclass
class _Challenge:
    offer: PairingOffer
    scopes: tuple[str, ...]
    challenge: PairingChallenge | None = None
    pending_installation: str | None = None
    pending_label: str | None = None
    approved: bool = False
    delivered: bool = False
    result: IssuedPair | None = None
    attempts: list[float] = field(default_factory=list)


class PairingService:
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
        self._challenges: dict[str, _Challenge] = {}

    def create_challenge(
        self,
        *,
        origin: str,
        label: str,
        chat_consent: bool,
        media_consent: bool,
        scopes: tuple[str, ...] | None = None,
        ttl_seconds: int = CHALLENGE_TTL_SECONDS,
    ) -> PairingOffer:
        origin = validate_origin(origin)
        if not chat_consent:
            raise PairingRejectedError("chat consent is required")
        if ttl_seconds <= 0 or ttl_seconds > CHALLENGE_TTL_SECONDS:
            raise PairingRejectedError("challenge lifetime is outside the accepted bound")
        state = self.store.load()
        if state is None or not state.enabled:
            raise PairingDisabledError("remote access is disabled")
        if scopes is None:
            computed = ["chat.read", "chat.write"]
            if media_consent:
                computed.extend(["media.read", "attachment.write"])
            scopes = tuple(computed)
        try:
            validate_scopes(scopes)
        except ContractError as exc:
            raise PairingRejectedError(str(exc)) from exc
        now = int(self.clock())
        offer = PairingOffer(
            pairing_id=str(uuid.uuid4()),
            origin=origin,
            label=label[:64] or "mobile",
            nonce=secrets.token_urlsafe(32),
            expires_at=now + ttl_seconds,
            collector_secret=secrets.token_urlsafe(32),
            comparison_code=f"{secrets.randbelow(1_000_000):06d}",
            chat_consent=chat_consent,
            media_consent=media_consent,
            scopes=tuple(scopes),
        )
        with self._lock:
            self._challenges[offer.pairing_id] = _Challenge(offer=offer, scopes=tuple(scopes))
        return offer

    def collect(
        self,
        *,
        pairing_id: str,
        nonce: str,
        collector_secret: str,
        installation_id: str,
        label: str,
    ) -> PairingPending:
        with self._lock:
            challenge = self._require_live(pairing_id)
            self._rate_limit(challenge)
            if not constant_time_equal(nonce, challenge.offer.nonce):
                raise PairingRejectedError("challenge nonce is not current")
            if not constant_time_equal(collector_secret, challenge.offer.collector_secret):
                raise PairingRejectedError("collector secret is not current")
            if challenge.pending_installation is not None or challenge.approved:
                raise PairingBusyError("challenge already has a pending request")
            try:
                validate_installation_id(installation_id)
            except ContractError as exc:
                raise PairingRejectedError(str(exc)) from exc
            if self.store.get_device(installation_id) is not None:
                raise PairingBusyError("installation already has a device; revoke it before re-pairing")
            challenge.pending_installation = installation_id
            challenge.pending_label = label[:64] or challenge.offer.label
            challenge.challenge = PairingChallenge(
                pairing_id=challenge.offer.pairing_id,
                origin=challenge.offer.origin,
                installation_id=installation_id,
                nonce=challenge.offer.nonce,
                label=challenge.pending_label,
                expires_at=challenge.offer.expires_at,
                chat_consent=challenge.offer.chat_consent,
                media_consent=challenge.offer.media_consent,
            )
            return PairingPending(
                pairing_id=challenge.offer.pairing_id,
                comparison_code=challenge.offer.comparison_code,
                expires_at=challenge.offer.expires_at,
                installation_id=installation_id,
            )

    def approve(
        self,
        *,
        pairing_id: str,
        comparison_code: str,
        scopes: tuple[str, ...] | None = None,
    ) -> PairingApproved:
        with self._lock:
            challenge = self._require_live(pairing_id)
            self._rate_limit(challenge)
            if challenge.pending_installation is None or challenge.challenge is None:
                raise PairingRejectedError("no pending request to approve")
            if challenge.approved:
                raise PairingBusyError("challenge has already been approved")
            if not constant_time_equal(comparison_code, challenge.offer.comparison_code):
                raise PairingRejectedError("comparison code does not match")
            approved_scopes = challenge.scopes if scopes is None else tuple(scopes)
            if not set(approved_scopes).issubset(set(challenge.scopes)):
                raise PairingRejectedError("approval cannot add scopes")
            try:
                pair = self.tokens.issue(
                    challenge.pending_installation,
                    approved_scopes,
                    label=challenge.pending_label or challenge.offer.label,
                )
            except (AuthConflictError, AuthRejectedError, AuthError) as exc:
                raise PairingRejectedError(str(exc)) from exc
            challenge.approved = True
            challenge.result = pair
            return PairingApproved(
                pairing_id=challenge.offer.pairing_id,
                installation_id=challenge.pending_installation,
                scopes=pair.scopes,
            )

    def poll(
        self,
        *,
        pairing_id: str,
        nonce: str,
        collector_secret: str,
    ) -> PairingResult | PairingPending:
        with self._lock:
            challenge = self._require_live(pairing_id)
            if not constant_time_equal(nonce, challenge.offer.nonce):
                raise PairingRejectedError("challenge nonce is not current")
            if not constant_time_equal(collector_secret, challenge.offer.collector_secret):
                raise PairingRejectedError("collector secret is not current")
            if challenge.pending_installation is None:
                raise PairingRejectedError("no pending request is bound to this challenge")
            if not challenge.approved or challenge.result is None:
                return PairingPending(
                    pairing_id=challenge.offer.pairing_id,
                    comparison_code=challenge.offer.comparison_code,
                    expires_at=challenge.offer.expires_at,
                    installation_id=challenge.pending_installation,
                )
            if challenge.delivered:
                raise PairingConsumedError("pairing result was already delivered")
            challenge.delivered = True
            return PairingResult(
                status="approved",
                pairing_id=challenge.offer.pairing_id,
                installation_id=challenge.pending_installation,
                access_token=challenge.result.access_token,
                refresh_token=challenge.result.refresh_token,
                expires_in=challenge.result.expires_in,
                refresh_expires_in=challenge.result.refresh_expires_in,
                scopes=challenge.result.scopes,
            )

    def cancel(self, pairing_id: str) -> bool:
        with self._lock:
            return self._challenges.pop(pairing_id, None) is not None

    def _require_live(self, pairing_id: str) -> _Challenge:
        challenge = self._challenges.get(pairing_id)
        if challenge is None:
            raise PairingNotFoundError("pairing challenge is not known")
        if self.clock() >= challenge.offer.expires_at:
            self._challenges.pop(pairing_id, None)
            raise PairingExpiredError("pairing challenge has expired")
        return challenge

    def _rate_limit(self, challenge: _Challenge) -> None:
        now = self.clock()
        challenge.attempts = [attempt for attempt in challenge.attempts if now - attempt < 60]
        if len(challenge.attempts) >= 5:
            raise PairingRateLimitedError("too many pairing attempts")
        challenge.attempts.append(now)
