"""Single-winner owner pairing with a QR challenge and 8-digit fallback code.

The QR path collects with the high-entropy challenge nonce/collector secret and
needs no further comparison code. The manual path collects with the full HTTPS
origin plus the 8-digit code. Both paths still require a local desktop approval,
and only the new owner pairing API is exposed by IPC/settings.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, cast
from urllib.parse import urlencode

from nanobot.remote.auth import (
    AuthConflictError,
    AuthError,
    AuthRejectedError,
    IssuedPair,
    TokenService,
    constant_time_equal,
)
from nanobot.remote.device_proof import (
    DEVICE_PROOF_TTL_SECONDS,
    DeviceProofClaims,
    DeviceProofError,
    device_proof_fingerprint,
    verify_device_proof,
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


class PairingDeviceRevokedError(PairingRejectedError):
    """The owner revoked the device that matches this recovery request."""


class PairingBusyError(PairingError):
    pass


class PairingRateLimitedError(PairingError):
    pass


class PairingConsumedError(PairingError):
    pass


class PairingUpgradeRequiredError(PairingRejectedError):
    """A legacy client tried to pair without the device-key contract."""


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
    device_fingerprint: str | None = None
    device_generation: int = 0
    proof_challenge: dict[str, object] | None = None


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
class _RecoveryChallenge:
    installation_id: str
    fingerprint: str
    generation: int
    origin: str
    public_key_der: bytes
    challenge_id: str
    nonce: str
    expires_at: int
    consumed: bool = False
    issued: IssuedPair | None = None


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
    device_public_key_der: bytes | None = None
    device_fingerprint: str | None = None
    device_generation: int = 0
    proof_challenge_id: str = ""
    proof_nonce: str = ""
    proof_expires_at: int = 0
    proof_consumed: bool = False
    proof_verified_at: int = 0


class PairingService:
    def __init__(
        self,
        store: RemoteAccessStore,
        tokens: TokenService,
        *,
        clock: Callable[[], float] = time.time,
        state_path: Path | None = None,
    ) -> None:
        self.store = store
        self.tokens = tokens
        self.clock = clock
        self._lock = threading.RLock()
        self._challenges: dict[str, _Challenge] = {}
        self._code_attempts: dict[str, list[float]] = {}
        self._recoveries: dict[str, _RecoveryChallenge] = {}
        self._state_path = state_path or (store.root / "pairing-pending.json")
        self._load_state()

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
            self._save_locked()
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
        device_public_key_der: bytes | None = None,
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
                device_public_key_der=device_public_key_der,
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
        device_public_key_der: bytes | None = None,
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
                device_public_key_der=device_public_key_der,
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
            challenge.scopes = tuple(approved_scopes)
            if challenge.device_public_key_der is None:
                raise PairingUpgradeRequiredError(
                    "This server requires device-key pairing; upgrade the app and re-pair"
                )
            challenge.approved = True
            challenge.state = "approved"
            challenge.result = None
            self._save_locked()
            return PairingApproved(
                pairing_id=challenge.offer.pairing_id,
                installation_id=challenge.pending_installation,
                scopes=approved_scopes,
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
            self._save_locked()
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
            self._save_locked()
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
            cancelled = self._challenges.pop(pairing_id, None) is not None
            if cancelled:
                self._save_locked()
            return cancelled

    def _bind_collector(
        self,
        challenge: _Challenge,
        *,
        installation_id: str,
        label: str,
        platform: str | None,
        model: str | None,
        method: str,
        device_public_key_der: bytes | None = None,
    ) -> PairingPending:
        if challenge.pending_installation is not None or challenge.approved:
            raise PairingBusyError("challenge already has a pending request")
        if device_public_key_der is None:
            raise PairingUpgradeRequiredError(
                "This server requires device-key pairing; upgrade the app and re-pair"
            )
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
        try:
            fingerprint = device_proof_fingerprint(device_public_key_der)
        except DeviceProofError as exc:
            raise PairingRejectedError(str(exc)) from exc
        now = int(self.clock())
        challenge.device_public_key_der = bytes(device_public_key_der)
        challenge.device_fingerprint = fingerprint
        challenge.device_generation = 1
        challenge.proof_challenge_id = uuid.uuid4().hex
        challenge.proof_nonce = secrets.token_urlsafe(32)
        challenge.proof_expires_at = min(
            challenge.offer.expires_at,
            now + DEVICE_PROOF_TTL_SECONDS,
        )
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
        self._save_locked()
        return self._pending_payload(challenge)

    def begin_device_recovery(
        self,
        *,
        installation_id: str,
        device_public_key_der: bytes,
    ) -> dict[str, object]:
        """Mint a fresh proof challenge for an already approved v2 device."""
        try:
            fingerprint = device_proof_fingerprint(device_public_key_der)
        except DeviceProofError as exc:
            raise PairingRejectedError(str(exc)) from exc
        with self._lock:
            state = self.store.load()
            device = self.store.get_device(installation_id)
            if state is None or not state.enabled or device is None:
                raise PairingRejectedError("installation is not recoverable")
            binding = self.store.get_device_proof(installation_id)
            if binding is None or binding[0] != fingerprint:
                raise PairingRejectedError("device proof is not current")
            if device.revoked:
                raise PairingDeviceRevokedError("installation is not recoverable")
            now = int(self.clock())
            recovery = _RecoveryChallenge(
                installation_id=installation_id,
                fingerprint=fingerprint,
                generation=binding[1],
                origin=binding[2],
                public_key_der=bytes(device_public_key_der),
                challenge_id=uuid.uuid4().hex,
                nonce=secrets.token_urlsafe(32),
                expires_at=now + DEVICE_PROOF_TTL_SECONDS,
            )
            self._recoveries[installation_id] = recovery
            return DeviceProofClaims(
                instance_id=state.installation_id,
                origin=binding[2],
                installation_id=installation_id,
                device_fingerprint=fingerprint,
                device_generation=binding[1],
                challenge_id=recovery.challenge_id,
                nonce=recovery.nonce,
                expires_at=recovery.expires_at,
            ).payload()

    def complete_device_recovery(
        self,
        *,
        installation_id: str,
        claims: DeviceProofClaims,
        signature_der: bytes,
    ) -> IssuedPair:
        """Verify a fresh recovery proof, consume it, and issue access."""
        with self._lock:
            recovery = self._recoveries.get(installation_id)
            if recovery is None or recovery.consumed:
                raise PairingConsumedError("device recovery challenge is not current")
            if self.clock() >= recovery.expires_at:
                raise PairingExpiredError("device recovery challenge has expired")
            state = self.store.load()
            if state is None or not state.enabled:
                raise PairingRejectedError("remote access is disabled")
            device = self.store.get_device(installation_id)
            if device is None or device.revoked:
                raise PairingRejectedError("installation is revoked")
            try:
                verify_device_proof(
                    claims,
                    public_key_der=recovery.public_key_der,
                    signature_der=signature_der,
                    expected_instance_id=state.installation_id,
                    expected_origin=recovery.origin,
                    expected_installation_id=installation_id,
                    expected_device_generation=recovery.generation,
                    expected_challenge_id=recovery.challenge_id,
                    expected_nonce=recovery.nonce,
                    now=self.clock(),
                )
            except DeviceProofError as exc:
                raise PairingRejectedError(exc.reason) from exc
            recovery.consumed = True
            try:
                pair = self.tokens.reissue_device(installation_id)
            except AuthRejectedError as exc:
                raise PairingRejectedError(str(exc)) from exc
            recovery.issued = pair
            return pair

    def proof_challenge_claims(self, pairing_id: str) -> dict[str, object] | None:
        """Return the exact claims the bound device must sign, if one is bound."""
        with self._lock:
            challenge = self._require_live(pairing_id)
            return self._proof_claims_locked(challenge)

    def complete_device_proof(
        self,
        *,
        pairing_id: str,
        claims: DeviceProofClaims,
        signature_der: bytes,
    ) -> IssuedPair:
        """Verify one current device proof, consume it, and issue access."""
        with self._lock:
            challenge = self._require_live(pairing_id)
            if challenge.pending_installation is None:
                raise PairingRejectedError("no pending request is bound to this challenge")
            if not challenge.approved:
                raise PairingRejectedError("pairing approval is required before proof")
            if challenge.device_public_key_der is None or challenge.device_fingerprint is None:
                raise PairingRejectedError("device proof is not bound to this challenge")
            if challenge.proof_consumed:
                raise PairingConsumedError("device proof challenge was already consumed")
            now = self.clock()
            if now >= challenge.proof_expires_at:
                raise PairingExpiredError("device proof challenge has expired")
            state = self.store.load()
            instance_id = state.installation_id if state is not None else ""
            try:
                verify_device_proof(
                    claims,
                    public_key_der=challenge.device_public_key_der,
                    signature_der=signature_der,
                    expected_instance_id=instance_id,
                    expected_origin=challenge.offer.origin,
                    expected_installation_id=challenge.pending_installation,
                    expected_device_generation=challenge.device_generation,
                    expected_challenge_id=challenge.proof_challenge_id,
                    expected_nonce=challenge.proof_nonce,
                    now=now,
                )
            except DeviceProofError as exc:
                raise PairingRejectedError(exc.reason) from exc
            if challenge.result is not None:
                raise PairingBusyError("pairing result is already issued")
            try:
                pair = self.tokens.issue_device(
                    challenge.pending_installation,
                    challenge.scopes,
                    device_fingerprint=challenge.device_fingerprint,
                    device_generation=challenge.device_generation,
                    origin=challenge.offer.origin,
                    label=challenge.pending_label or challenge.offer.label,
                )
            except (AuthConflictError, AuthRejectedError, AuthError) as exc:
                raise PairingRejectedError(str(exc)) from exc
            challenge.proof_consumed = True
            challenge.proof_verified_at = int(now)
            challenge.result = pair
            self._save_locked()
            return pair

    def _proof_claims_locked(self, challenge: _Challenge) -> dict[str, object] | None:
        if (
            challenge.device_public_key_der is None
            or challenge.device_fingerprint is None
            or challenge.pending_installation is None
        ):
            return None
        state = self.store.load()
        instance_id = state.installation_id if state is not None else ""
        return DeviceProofClaims(
            instance_id=instance_id,
            origin=challenge.offer.origin,
            installation_id=challenge.pending_installation,
            device_fingerprint=challenge.device_fingerprint,
            device_generation=challenge.device_generation,
            challenge_id=challenge.proof_challenge_id,
            nonce=challenge.proof_nonce,
            expires_at=challenge.proof_expires_at,
        ).payload()

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
            device_fingerprint=challenge.device_fingerprint,
            device_generation=challenge.device_generation,
            proof_challenge=self._proof_claims_locked(challenge),
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

    def _load_state(self) -> None:
        """Restore unexpired collections and non-bearer initial approvals."""
        try:
            raw: object = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        entries_value = cast(object, cast(dict[str, Any], raw).get("challenges"))
        if not isinstance(entries_value, list):
            return
        now = self.clock()
        for raw_item in cast(list[object], entries_value):
            if not isinstance(raw_item, dict):
                continue
            value = cast(dict[str, Any], cast(object, raw_item))
            offer_value = value.get("offer")
            if not isinstance(offer_value, dict):
                continue
            offer_data = cast(dict[str, Any], cast(object, offer_value))
            scopes = _json_str_list(offer_data.get("scopes"))
            if scopes is None:
                continue
            try:
                offer = PairingOffer(
                    pairing_id=_json_text(offer_data.get("pairing_id")),
                    origin=_json_text(offer_data.get("origin")),
                    label=_json_text(offer_data.get("label")),
                    nonce=_json_text(offer_data.get("nonce")),
                    expires_at=_json_int(offer_data.get("expires_at")),
                    collector_secret=_json_text(offer_data.get("collector_secret")),
                    pairing_code=_json_text(offer_data.get("pairing_code")),
                    comparison_code=_json_text(offer_data.get("comparison_code")),
                    chat_consent=_json_bool(offer_data.get("chat_consent")),
                    media_consent=_json_bool(offer_data.get("media_consent")),
                    scopes=scopes,
                    full_control_consent=_json_bool(offer_data.get("full_control_consent")),
                    shared_history_consent=_json_bool(offer_data.get("shared_history_consent")),
                    transport=_json_text(
                        offer_data.get("transport"), PAIRING_TRANSPORT
                    ),
                )
                if not offer.pairing_id or now >= offer.expires_at:
                    continue
                challenge_scopes = _json_str_list(value.get("scopes"))
                if challenge_scopes is None:
                    continue
                key_raw = _json_text(value.get("device_public_key_der"))
                challenge = _Challenge(
                    offer=offer,
                    scopes=challenge_scopes,
                    owner_only=_json_bool(value.get("owner_only")),
                    pending_installation=_json_optional_text(
                        value.get("pending_installation")
                    ),
                    pending_label=_json_optional_text(value.get("pending_label")),
                    pending_platform=_json_optional_text(value.get("pending_platform")),
                    pending_model=_json_optional_text(value.get("pending_model")),
                    pending_method=_json_optional_text(value.get("pending_method")),
                    state=_json_text(value.get("state"), "offer_created"),
                    requested_at=_json_int(value.get("requested_at")),
                    device_public_key_der=(
                        base64.b64decode(key_raw, validate=True)
                        if key_raw else None
                    ),
                    device_fingerprint=_json_optional_text(
                        value.get("device_fingerprint")
                    ),
                    device_generation=_json_int(value.get("device_generation")),
                    proof_challenge_id=_json_text(value.get("proof_challenge_id")),
                    proof_nonce=_json_text(value.get("proof_nonce")),
                    proof_expires_at=_json_int(value.get("proof_expires_at")),
                    proof_consumed=_json_bool(value.get("proof_consumed")),
                    proof_verified_at=_json_int(value.get("proof_verified_at")),
                    created_at=_json_int(value.get("created_at")),
                )
                if challenge.pending_installation is not None:
                    challenge.challenge = PairingChallenge(
                        pairing_id=offer.pairing_id,
                        origin=offer.origin,
                        installation_id=challenge.pending_installation,
                        nonce=offer.nonce,
                        label=challenge.pending_label or offer.label,
                        expires_at=offer.expires_at,
                        chat_consent=offer.chat_consent,
                        media_consent=offer.media_consent,
                        full_control_consent=offer.full_control_consent,
                        shared_history_consent=offer.shared_history_consent,
                        transport=offer.transport,
                    )
                # Issued results are never persisted or replayed. An approval
                # without issuance retains only its device binding and deadline.
                if challenge.proof_consumed:
                    continue
                challenge.denied = challenge.state == "denied"
                if challenge.state == "approved":
                    device = self.store.get_device(challenge.pending_installation or "")
                    if device is not None and device.revoked:
                        continue
                    if (
                        challenge.device_public_key_der is None
                        or challenge.pending_installation is None
                        or device_proof_fingerprint(challenge.device_public_key_der) != challenge.device_fingerprint
                        or not set(challenge.scopes).issubset(offer.scopes)
                    ):
                        continue
                    challenge.approved = True
                    # No challenge admitted before a restart can be replayed.
                    challenge.proof_challenge_id = uuid.uuid4().hex
                    challenge.proof_nonce = secrets.token_urlsafe(32)
                    challenge.proof_expires_at = min(int(now) + DEVICE_PROOF_TTL_SECONDS, offer.expires_at)
                self._challenges[offer.pairing_id] = challenge
            except (KeyError, TypeError, ValueError):
                continue

    def _save_locked(self) -> None:
        """Persist pending bindings/approvals; never write issued bearer tokens."""
        entries: list[dict[str, object]] = []
        for challenge in self._challenges.values():
            if challenge.proof_consumed or challenge.result:
                continue
            entries.append({
                "offer": {
                    "pairing_id": challenge.offer.pairing_id,
                    "origin": challenge.offer.origin,
                    "label": challenge.offer.label,
                    "nonce": challenge.offer.nonce,
                    "expires_at": challenge.offer.expires_at,
                    "collector_secret": challenge.offer.collector_secret,
                    "pairing_code": challenge.offer.pairing_code,
                    "comparison_code": challenge.offer.comparison_code,
                    "chat_consent": challenge.offer.chat_consent,
                    "media_consent": challenge.offer.media_consent,
                    "scopes": list(challenge.offer.scopes),
                    "full_control_consent": challenge.offer.full_control_consent,
                    "shared_history_consent": challenge.offer.shared_history_consent,
                    "transport": challenge.offer.transport,
                },
                "scopes": list(challenge.scopes),
                "owner_only": challenge.owner_only,
                "pending_installation": challenge.pending_installation,
                "pending_label": challenge.pending_label,
                "pending_platform": challenge.pending_platform,
                "pending_model": challenge.pending_model,
                "pending_method": challenge.pending_method,
                "state": challenge.state,
                "requested_at": challenge.requested_at,
                "created_at": challenge.created_at,
                "device_public_key_der": (
                    base64.b64encode(challenge.device_public_key_der).decode("ascii")
                    if challenge.device_public_key_der is not None else None
                ),
                "device_fingerprint": challenge.device_fingerprint,
                "device_generation": challenge.device_generation,
                "proof_challenge_id": challenge.proof_challenge_id,
                "proof_nonce": challenge.proof_nonce,
                "proof_expires_at": challenge.proof_expires_at,
                "proof_consumed": challenge.proof_consumed,
                "proof_verified_at": challenge.proof_verified_at,
            })
        self._state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self._state_path.with_name(f".{self._state_path.name}.tmp")
        temporary.write_text(
            json.dumps({"version": 1, "challenges": entries}, ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            os.chmod(temporary, 0o600)
            os.replace(temporary, self._state_path)
        except OSError:
            temporary.unlink(missing_ok=True)

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


def _json_text(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _json_optional_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _json_int(value: object, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _json_bool(value: object, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _json_str_list(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    return tuple(str(item) for item in cast(list[object], value))
