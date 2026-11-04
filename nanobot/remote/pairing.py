"""Single-winner owner pairing with a QR challenge and 8-digit fallback code.

The QR path collects with the high-entropy challenge nonce/collector secret and
needs no further comparison code. The manual path collects with the full HTTPS
origin plus the 8-digit code. Both paths still require a local desktop approval,
and only the new owner pairing API is exposed by IPC/settings.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Callable
from urllib.parse import urlencode

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
    OWNER_PAIRING_SCOPES,
    OWNER_SCOPE,
    PAIRING_CODE_DIGITS,
    PAIRING_TRANSPORT,
    ContractError,
    PairingChallenge,
    validate_installation_id,
    validate_origin,
    validate_scopes,
)
from nanobot.remote.store import RemoteAccessStore

_CODE_RE = re.compile(r"^\d{8}$")
_RENEWAL_WINDOW_SECONDS = 5 * 60
_SETUP_LIMIT_SECONDS = 30 * 60


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


class PairingOwnerRequiredError(PairingRejectedError):
    """The new pairing API only accepts owner/full-control offers."""


class PairingCodeInvalidError(PairingRejectedError):
    """The 8-digit fallback code is missing, wrong or expired."""


class PairingDeniedError(PairingRejectedError):
    """The desktop denied this pairing request."""


@dataclass(frozen=True)
class PairingOffer:
    pairing_id: str
    origin: str
    label: str
    nonce: str
    expires_at: int
    collector_secret: str
    pairing_code: str
    comparison_code: str
    chat_consent: bool
    media_consent: bool
    scopes: tuple[str, ...]
    full_control_consent: bool = False
    shared_history_consent: bool = False
    transport: str = PAIRING_TRANSPORT

    @property
    def qr_link(self) -> str:
        query = urlencode(
            {
                "origin": self.origin,
                "challenge": f"{self.pairing_id}.{self.nonce}.{self.collector_secret}",
                "transport": self.transport,
            }
        )
        return f"nanobot://pair?{query}"


@dataclass(frozen=True)
class PairingPending:
    pairing_id: str
    comparison_code: str
    expires_at: int
    installation_id: str
    state: str = "pending_approval"
    label: str = ""
    platform: str | None = None
    model: str | None = None
    origin: str = ""
    transport: str = PAIRING_TRANSPORT
    scopes: tuple[str, ...] = ()
    requested_at: int = 0
    pairing_code: str = ""
    nonce: str = ""
    collector_secret: str = ""
    created_at: int = 0
    renewable: bool = False


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
    owner_only: bool = False
    challenge: PairingChallenge | None = None
    pending_installation: str | None = None
    pending_label: str | None = None
    pending_platform: str | None = None
    pending_model: str | None = None
    pending_method: str | None = None
    state: str = "offer_created"
    requested_at: int = 0
    approved: bool = False
    denied: bool = False
    delivered: bool = False
    result: IssuedPair | None = None
    attempts: list[float] = field(default_factory=list)
    created_at: int = 0


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
        self._code_attempts: dict[str, list[float]] = {}

    def create_challenge(
        self,
        *,
        origin: str,
        label: str,
        chat_consent: bool,
        media_consent: bool,
        scopes: tuple[str, ...] | None = None,
        full_control_consent: bool = False,
        shared_history_consent: bool = False,
        ttl_seconds: int = CHALLENGE_TTL_SECONDS,
        owner_only: bool = False,
        transport: str = PAIRING_TRANSPORT,
    ) -> PairingOffer:
        origin = validate_origin(origin)
        if transport != PAIRING_TRANSPORT:
            raise PairingRejectedError("only the tailscale transport is supported")
        if owner_only:
            if not full_control_consent:
                raise PairingOwnerRequiredError(
                    "owner pairing requires full-control consent"
                )
            if scopes is not None:
                try:
                    validate_scopes(scopes)
                except ContractError as exc:
                    raise PairingRejectedError(str(exc)) from exc
                if OWNER_SCOPE not in scopes:
                    raise PairingOwnerRequiredError(
                        "owner pairing cannot use chat-only scopes"
                    )
                scopes = tuple(dict.fromkeys((*OWNER_PAIRING_SCOPES, *scopes)))
            else:
                scopes = OWNER_PAIRING_SCOPES
            chat_consent = True
            media_consent = True
        else:
            if not chat_consent:
                raise PairingRejectedError("chat consent is required")
            if scopes is None:
                computed = ["chat.read", "chat.write"]
                if media_consent:
                    computed.extend(["media.read", "attachment.write"])
                if full_control_consent:
                    computed.append(OWNER_SCOPE)
                scopes = tuple(computed)
            elif full_control_consent and OWNER_SCOPE not in scopes:
                raise PairingRejectedError("full-control consent requires the owner scope")
            elif OWNER_SCOPE in scopes and not full_control_consent:
                raise PairingRejectedError("owner scope requires full-control consent")
            try:
                validate_scopes(scopes)
            except ContractError as exc:
                raise PairingRejectedError(str(exc)) from exc
        if shared_history_consent and not full_control_consent:
            raise PairingRejectedError("shared-history consent requires full-control consent")
        if ttl_seconds <= 0 or ttl_seconds > CHALLENGE_TTL_SECONDS:
            raise PairingRejectedError("challenge lifetime is outside the accepted bound")
        state = self.store.load()
        if state is None or not state.enabled:
            raise PairingDisabledError("remote access is disabled")
        now = int(self.clock())
        with self._lock:
            self._prune_locked()
            code = self._new_code_locked()
            offer = PairingOffer(
                pairing_id=str(uuid.uuid4()),
                origin=origin,
                label=_clean_label(label) or "mobile",
                nonce=secrets.token_urlsafe(32),
                expires_at=now + ttl_seconds,
                collector_secret=secrets.token_urlsafe(32),
                pairing_code=code,
                # Deprecated alias kept for older internal callers/tests.
                comparison_code=code,
                chat_consent=chat_consent,
                media_consent=media_consent,
                scopes=tuple(scopes),
                full_control_consent=full_control_consent,
                shared_history_consent=shared_history_consent,
                transport=transport,
            )
            self._challenges[offer.pairing_id] = _Challenge(
                offer=offer,
                scopes=tuple(scopes),
                owner_only=owner_only,
                created_at=now,
            )
        return offer

    def collect(
        self,
        *,
        pairing_id: str,
        nonce: str,
        collector_secret: str,
        installation_id: str,
        label: str,
        platform: str | None = None,
        model: str | None = None,
    ) -> PairingPending:
        with self._lock:
            challenge = self._require_live(pairing_id)
            self._rate_limit(challenge)
            if not constant_time_equal(nonce, challenge.offer.nonce):
                raise PairingRejectedError("challenge nonce is not current")
            if not constant_time_equal(collector_secret, challenge.offer.collector_secret):
                raise PairingRejectedError("collector secret is not current")
            return self._bind_collector(
                challenge,
                installation_id=installation_id,
                label=label,
                platform=platform,
                model=model,
                method="qr",
            )

    def collect_by_code(
        self,
        *,
        origin: str,
        pairing_code: str,
        installation_id: str,
        label: str,
        platform: str | None = None,
        model: str | None = None,
    ) -> PairingPending:
        """Collect with the full HTTPS origin plus the 8-digit fallback code."""

        try:
            origin = validate_origin(origin)
        except ContractError as exc:
            raise PairingCodeInvalidError("pairing code is not current") from exc
        with self._lock:
            self._prune_locked()
            self._rate_limit_code(origin)
            challenge = self._find_by_code_locked(pairing_code)
            if challenge is None or challenge.offer.origin != origin:
                raise PairingCodeInvalidError("pairing code is not current")
            if self.clock() >= challenge.offer.expires_at:
                challenge.state = "expired"
                raise PairingExpiredError("pairing challenge has expired")
            if challenge.denied:
                raise PairingDeniedError("pairing request was denied")
            self._code_attempts.pop(origin, None)
            return self._bind_collector(
                challenge,
                installation_id=installation_id,
                label=label,
                platform=platform,
                model=model,
                method="code",
            )

    def approve(
        self,
        *,
        pairing_id: str,
        comparison_code: str | None = None,
        scopes: tuple[str, ...] | None = None,
        pairing_code: str | None = None,
    ) -> PairingApproved:
        code = pairing_code if pairing_code is not None else comparison_code
        with self._lock:
            challenge = self._require_live(pairing_id)
            self._rate_limit(challenge)
            if challenge.pending_installation is None or challenge.challenge is None:
                raise PairingRejectedError("no pending request to approve")
            if challenge.approved:
                raise PairingBusyError("challenge has already been approved")
            if challenge.denied:
                raise PairingDeniedError("pairing request was denied")
            if code is not None:
                if not constant_time_equal(code, challenge.offer.pairing_code):
                    raise PairingCodeInvalidError("pairing code does not match")
            elif challenge.pending_method == "code":
                raise PairingCodeInvalidError("pairing code is required")
            approved_scopes = challenge.scopes if scopes is None else tuple(scopes)
            if not set(approved_scopes).issubset(set(challenge.scopes)):
                raise PairingRejectedError("approval cannot add scopes")
            if challenge.owner_only and OWNER_SCOPE not in approved_scopes:
                raise PairingOwnerRequiredError("owner scope cannot be removed")
            try:
                pair = self.tokens.issue(
                    challenge.pending_installation,
                    approved_scopes,
                    label=challenge.pending_label or challenge.offer.label,
                )
            except (AuthConflictError, AuthRejectedError, AuthError) as exc:
                raise PairingRejectedError(str(exc)) from exc
            challenge.approved = True
            challenge.state = "approved"
            challenge.result = pair
            return PairingApproved(
                pairing_id=challenge.offer.pairing_id,
                installation_id=challenge.pending_installation,
                scopes=pair.scopes,
            )

    def deny(self, pairing_id: str) -> PairingPending:
        with self._lock:
            challenge = self._challenges.get(pairing_id)
            if challenge is None:
                raise PairingNotFoundError("pairing challenge is not known")
            if self.clock() >= challenge.offer.expires_at:
                challenge.state = "expired"
                raise PairingExpiredError("pairing challenge has expired")
            if challenge.approved:
                raise PairingBusyError("challenge has already been approved")
            challenge.denied = True
            challenge.state = "denied"
            return self._pending_payload(challenge)

    def status(self, pairing_id: str) -> PairingPending:
        with self._lock:
            challenge = self._challenges.get(pairing_id)
            if challenge is None:
                raise PairingNotFoundError("pairing challenge is not known")
            if (
                self.clock() >= challenge.offer.expires_at
                and challenge.state not in {"approved", "denied"}
            ):
                challenge.state = "expired"
            return self._pending_payload(challenge)

    def pending(self) -> list[PairingPending]:
        with self._lock:
            self._prune_locked()
            now = self.clock()
            result: list[PairingPending] = []
            for challenge in self._challenges.values():
                if challenge.state in {"expired", "denied"}:
                    continue
                if challenge.delivered:
                    continue
                if now >= challenge.offer.expires_at and challenge.state != "approved":
                    challenge.state = "expired"
                    continue
                result.append(self._pending_payload(challenge))
            result.sort(key=lambda item: item.requested_at)
            return result

    def list_pending(self) -> list[PairingPending]:
        return self.pending()

    def pending_status(self, pairing_id: str) -> PairingPending:
        return self.status(pairing_id)

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
            if challenge.denied:
                raise PairingDeniedError("pairing request was denied")
            if not challenge.approved or challenge.result is None:
                return self._pending_payload(challenge)
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

    def renew(self, pairing_id: str) -> PairingPending:
        """Extend one pending desktop-local offer within the overall setup limit.

        Never renews after deny/cancel/approval/consumption. The collector/device
        binding, nonce, code and scopes are retained; only the deadline moves.
        """
        with self._lock:
            challenge = self._challenges.get(pairing_id)
            if challenge is None:
                raise PairingNotFoundError("pairing challenge is not known")
            if challenge.denied:
                raise PairingDeniedError("pairing challenge was denied")
            if (
                challenge.state in {"approved", "consumed"}
                or challenge.approved
                or challenge.delivered
                or challenge.result is not None
            ):
                raise PairingConsumedError("pairing challenge can no longer be renewed")
            now = int(self.clock())
            limit = challenge.created_at + _SETUP_LIMIT_SECONDS
            if now >= limit:
                raise PairingExpiredError("pairing setup limit has expired")
            renewed_until = min(now + _RENEWAL_WINDOW_SECONDS, limit)
            if renewed_until <= now:
                raise PairingExpiredError("pairing setup limit has expired")
            challenge.offer = replace(challenge.offer, expires_at=renewed_until)
            if challenge.challenge is not None:
                challenge.challenge = challenge.challenge.model_copy(
                    update={"expires_at": renewed_until},
                )
            if challenge.state == "expired":
                challenge.state = (
                    "pending_approval"
                    if challenge.pending_installation is not None
                    else "offer_created"
                )
            return self._pending_payload(challenge)

    def _renewable(self, challenge: _Challenge) -> bool:
        if challenge.denied or challenge.approved or challenge.delivered:
            return False
        if challenge.state in {"approved", "consumed"} or challenge.result is not None:
            return False
        return int(self.clock()) < challenge.created_at + _SETUP_LIMIT_SECONDS

    def cancel(self, pairing_id: str) -> bool:
        with self._lock:
            return self._challenges.pop(pairing_id, None) is not None

    def _bind_collector(
        self,
        challenge: _Challenge,
        *,
        installation_id: str,
        label: str,
        platform: str | None,
        model: str | None,
        method: str,
    ) -> PairingPending:
        if challenge.pending_installation is not None or challenge.approved:
            raise PairingBusyError("challenge already has a pending request")
        try:
            validate_installation_id(installation_id)
        except ContractError as exc:
            raise PairingRejectedError(str(exc)) from exc
        if self.store.get_device(installation_id) is not None:
            raise PairingBusyError("installation already has a device; revoke it before re-pairing")
        challenge.pending_installation = installation_id
        challenge.pending_label = _clean_label(label) or challenge.offer.label
        challenge.pending_platform = _clean_optional(platform)
        challenge.pending_model = _clean_optional(model)
        challenge.pending_method = method
        challenge.state = "pending_approval"
        challenge.requested_at = int(self.clock())
        challenge.challenge = PairingChallenge(
            pairing_id=challenge.offer.pairing_id,
            origin=challenge.offer.origin,
            installation_id=installation_id,
            nonce=challenge.offer.nonce,
            label=challenge.pending_label,
            expires_at=challenge.offer.expires_at,
            chat_consent=challenge.offer.chat_consent,
            media_consent=challenge.offer.media_consent,
            full_control_consent=challenge.offer.full_control_consent,
            shared_history_consent=challenge.offer.shared_history_consent,
            transport=challenge.offer.transport,
        )
        return self._pending_payload(challenge)

    def _pending_payload(self, challenge: _Challenge) -> PairingPending:
        pending = challenge.pending_installation is not None
        return PairingPending(
            pairing_id=challenge.offer.pairing_id,
            comparison_code=challenge.offer.pairing_code,
            expires_at=challenge.offer.expires_at,
            installation_id=challenge.pending_installation or "",
            state=challenge.state,
            label=challenge.pending_label or challenge.offer.label,
            platform=challenge.pending_platform,
            model=challenge.pending_model,
            origin=challenge.offer.origin,
            transport=challenge.offer.transport,
            scopes=challenge.offer.scopes if not pending else challenge.scopes,
            requested_at=challenge.requested_at,
            pairing_code=challenge.offer.pairing_code,
            nonce=challenge.offer.nonce,
            collector_secret=challenge.offer.collector_secret,
            created_at=challenge.created_at,
            renewable=self._renewable(challenge),
        )

    def _new_code_locked(self) -> str:
        self._prune_locked()
        live = {
            challenge.offer.pairing_code
            for challenge in self._challenges.values()
            if self.clock() < challenge.offer.expires_at
        }
        for _attempt in range(1000):
            code = f"{secrets.randbelow(10**PAIRING_CODE_DIGITS):0{PAIRING_CODE_DIGITS}d}"
            if code not in live:
                return code
        raise PairingRejectedError("could not allocate a unique pairing code")

    def _find_by_code_locked(self, pairing_code: str) -> _Challenge | None:
        if not _CODE_RE.fullmatch(pairing_code):
            return None
        for challenge in self._challenges.values():
            if constant_time_equal(pairing_code, challenge.offer.pairing_code):
                return challenge
        return None

    def _rate_limit_code(self, origin: str) -> None:
        now = self.clock()
        attempts = [
            attempt
            for attempt in self._code_attempts.get(origin, [])
            if now - attempt < 60
        ]
        if len(attempts) >= 5:
            self._code_attempts[origin] = attempts
            raise PairingRateLimitedError("too many pairing code attempts")
        attempts.append(now)
        self._code_attempts[origin] = attempts

    def _prune_locked(self) -> None:
        now = self.clock()
        for pairing_id, challenge in list(self._challenges.items()):
            # Eligible expired offers remain only until the overall setup limit.
            if now <= challenge.created_at + _SETUP_LIMIT_SECONDS:
                continue
            self._challenges.pop(pairing_id, None)
        for origin, attempts in list(self._code_attempts.items()):
            kept = [attempt for attempt in attempts if now - attempt < 60]
            if kept:
                self._code_attempts[origin] = kept
            else:
                self._code_attempts.pop(origin, None)

    def _require_live(self, pairing_id: str) -> _Challenge:
        challenge = self._challenges.get(pairing_id)
        if challenge is None:
            raise PairingNotFoundError("pairing challenge is not known")
        if challenge.denied:
            raise PairingDeniedError("pairing request was denied")
        if self.clock() >= challenge.offer.expires_at:
            if challenge.state != "approved":
                challenge.state = "expired"
            raise PairingExpiredError("pairing challenge has expired")
        return challenge

    def _rate_limit(self, challenge: _Challenge) -> None:
        now = self.clock()
        challenge.attempts = [attempt for attempt in challenge.attempts if now - attempt < 60]
        if len(challenge.attempts) >= 5:
            raise PairingRateLimitedError("too many pairing attempts")
        challenge.attempts.append(now)


def _clean_label(value: str) -> str:
    return " ".join(value.split())[:64]


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split())[:64]
    return cleaned or None
