"""Shared device-proof fixtures for remote pairing tests."""

from __future__ import annotations

import base64

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from nanobot.remote.device_proof import (
    DeviceProofClaims,
    canonical_device_proof_bytes,
)

_KEYS: dict[str, ec.EllipticCurvePrivateKey] = {}


def encode_proof_base64url(value: bytes) -> str:
    """Encode one proof binary value with canonical unpadded base64url."""
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def new_device_key() -> tuple[ec.EllipticCurvePrivateKey, bytes]:
    key = ec.generate_private_key(ec.SECP256R1())
    public_der = key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return key, public_der


def sign_claims(
    key: ec.EllipticCurvePrivateKey,
    claims: DeviceProofClaims,
) -> bytes:
    return key.sign(canonical_device_proof_bytes(claims), ec.ECDSA(hashes.SHA256()))


def complete_proof(pairing, pairing_id: str, key=None) -> object:
    """Complete the bound proof for an approved challenge."""
    key = key or _KEYS[pairing_id]
    payload = pairing.proof_challenge_claims(pairing_id)
    assert payload is not None
    claims = DeviceProofClaims.from_payload(payload)
    return pairing.complete_device_proof(
        pairing_id=pairing_id,
        claims=claims,
        signature_der=sign_claims(key, claims),
    )


def collect_with_key(pairing, **kwargs):
    key, public_der = new_device_key()
    pending = pairing.collect(device_public_key_der=public_der, **kwargs)
    _KEYS[pending.pairing_id] = key
    return pending


def collect_by_code_with_key(pairing, **kwargs):
    key, public_der = new_device_key()
    pending = pairing.collect_by_code(device_public_key_der=public_der, **kwargs)
    _KEYS[pending.pairing_id] = key
    return pending


def approve_and_prove(pairing, **kwargs):
    approved = pairing.approve(**kwargs)
    pairing_id = str(kwargs["pairing_id"])
    complete_proof(pairing, pairing_id)
    return approved
