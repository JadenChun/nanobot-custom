"""Canonical device-proof claims and P-256 verification for mobile access."""

from __future__ import annotations

import hashlib
import json
import base64
import re
from dataclasses import dataclass
from typing import Any, Mapping

PROOF_PROTOCOL_VERSION = 2
PROOF_PURPOSE = "remote.access"
DEVICE_PROOF_TTL_SECONDS = 120
MAX_PROOF_INTEGER = (1 << 63) - 1
PROOF_CLAIM_KEYS = frozenset(
    {
        "version",
        "purpose",
        "instance_id",
        "origin",
        "installation_id",
        "device_fingerprint",
        "device_generation",
        "challenge_id",
        "nonce",
        "expires_at",
    }
)


class DeviceProofError(ValueError):
    """A device proof is invalid for one fixed, content-free reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_text(value: object, reason: str) -> str:
    if not isinstance(value, str) or not value:
        raise DeviceProofError(reason)
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise DeviceProofError(reason) from None
    return value


def _require_int(value: object, reason: str) -> int:
    if type(value) is not int or not 0 < value <= MAX_PROOF_INTEGER:
        raise DeviceProofError(reason)
    return value


@dataclass(frozen=True, slots=True)
class DeviceProofClaims:
    """The exact fields signed by an Android device key."""

    instance_id: str
    origin: str
    installation_id: str
    device_fingerprint: str
    device_generation: int
    challenge_id: str
    nonce: str
    expires_at: int
    version: int = PROOF_PROTOCOL_VERSION
    purpose: str = PROOF_PURPOSE

    def __post_init__(self) -> None:
        _require_int(self.version, "unsupported_proof_version")
        if self.version != PROOF_PROTOCOL_VERSION:
            raise DeviceProofError("unsupported_proof_version")
        if self.purpose != PROOF_PURPOSE:
            raise DeviceProofError("unsupported_proof_purpose")
        _require_text(self.instance_id, "invalid_instance")
        _require_text(self.origin, "invalid_origin")
        _require_text(self.installation_id, "invalid_installation")
        _require_text(self.device_fingerprint, "invalid_device_key")
        _require_int(self.device_generation, "invalid_device_generation")
        _require_text(self.challenge_id, "invalid_challenge")
        _require_text(self.nonce, "invalid_nonce")
        _require_int(self.expires_at, "invalid_expiry")

    def payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "purpose": self.purpose,
            "instance_id": self.instance_id,
            "origin": self.origin,
            "installation_id": self.installation_id,
            "device_fingerprint": self.device_fingerprint,
            "device_generation": self.device_generation,
            "challenge_id": self.challenge_id,
            "nonce": self.nonce,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> DeviceProofClaims:
        if set(payload) != set(PROOF_CLAIM_KEYS):
            raise DeviceProofError("invalid_proof_shape")
        try:
            return cls(
                version=_require_int(payload["version"], "unsupported_proof_version"),
                purpose=_require_text(payload["purpose"], "unsupported_proof_purpose"),
                instance_id=_require_text(payload["instance_id"], "invalid_instance"),
                origin=_require_text(payload["origin"], "invalid_origin"),
                installation_id=_require_text(
                    payload["installation_id"], "invalid_installation"
                ),
                device_fingerprint=_require_text(
                    payload["device_fingerprint"], "invalid_device_key"
                ),
                device_generation=_require_int(
                    payload["device_generation"], "invalid_device_generation"
                ),
                challenge_id=_require_text(
                    payload["challenge_id"], "invalid_challenge"
                ),
                nonce=_require_text(payload["nonce"], "invalid_nonce"),
                expires_at=_require_int(payload["expires_at"], "invalid_expiry"),
            )
        except KeyError:
            raise DeviceProofError("invalid_proof_shape") from None


def canonical_device_proof_bytes(claims: DeviceProofClaims) -> bytes:
    """Return the exact UTF-8 bytes the device signs."""
    return json.dumps(
        claims.payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def device_proof_fingerprint(public_key_der: bytes) -> str:
    """Validate one P-256 SubjectPublicKeyInfo and return its SHA-256 digest."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    if not public_key_der:
        raise DeviceProofError("invalid_device_key")
    try:
        key = serialization.load_der_public_key(bytes(public_key_der))
    except (TypeError, ValueError):
        raise DeviceProofError("invalid_device_key") from None
    if not isinstance(key, ec.EllipticCurvePublicKey) or key.curve.name != "secp256r1":
        raise DeviceProofError("unsupported_device_key")
    if key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo) != public_key_der:
        raise DeviceProofError("invalid_device_key")
    return hashlib.sha256(bytes(public_key_der)).hexdigest()


def decode_proof_base64url(value: str) -> bytes:
    """Decode only the canonical unpadded base64url representation."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise DeviceProofError("invalid_proof_encoding")
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except ValueError:
        raise DeviceProofError("invalid_proof_encoding") from None
    if base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=") != value:
        raise DeviceProofError("invalid_proof_encoding")
    return decoded


def verify_device_proof(
    claims: DeviceProofClaims,
    *,
    public_key_der: bytes,
    signature_der: bytes,
    expected_instance_id: str,
    expected_origin: str,
    expected_installation_id: str,
    expected_device_generation: int,
    expected_challenge_id: str,
    expected_nonce: str,
    now: float,
) -> str:
    """Verify one P-256/SHA-256 DER proof and return the device fingerprint."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, utils

    if claims.instance_id != expected_instance_id:
        raise DeviceProofError("wrong_instance")
    if claims.origin != expected_origin:
        raise DeviceProofError("wrong_origin")
    if claims.installation_id != expected_installation_id:
        raise DeviceProofError("wrong_installation")
    if claims.device_generation != expected_device_generation:
        raise DeviceProofError("stale_device_generation")
    if claims.challenge_id != expected_challenge_id:
        raise DeviceProofError("wrong_challenge")
    if claims.nonce != expected_nonce:
        raise DeviceProofError("wrong_nonce")

    fingerprint = device_proof_fingerprint(public_key_der)
    if claims.device_fingerprint != fingerprint:
        raise DeviceProofError("wrong_device_key")
    if now >= claims.expires_at:
        raise DeviceProofError("expired_proof")
    if not signature_der:
        raise DeviceProofError("invalid_signature")
    try:
        r, s = utils.decode_dss_signature(signature_der)
        if r <= 0 or s <= 0 or utils.encode_dss_signature(r, s) != signature_der:
            raise ValueError("non-canonical signature")
    except ValueError:
        raise DeviceProofError("invalid_signature") from None

    try:
        key = serialization.load_der_public_key(bytes(public_key_der))
    except (TypeError, ValueError):
        raise DeviceProofError("invalid_device_key") from None
    if not isinstance(key, ec.EllipticCurvePublicKey) or key.curve.name != "secp256r1":
        raise DeviceProofError("unsupported_device_key")
    try:
        key.verify(
            bytes(signature_der),
            canonical_device_proof_bytes(claims),
            ec.ECDSA(hashes.SHA256()),
        )
    except InvalidSignature:
        raise DeviceProofError("invalid_signature") from None
    return fingerprint


__all__ = [
    "DEVICE_PROOF_TTL_SECONDS",
    "DeviceProofClaims",
    "DeviceProofError",
    "PROOF_CLAIM_KEYS",
    "PROOF_PROTOCOL_VERSION",
    "PROOF_PURPOSE",
    "canonical_device_proof_bytes",
    "device_proof_fingerprint",
    "verify_device_proof",
]
