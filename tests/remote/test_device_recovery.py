"""Regression tests for A3-7 proof-bound access recovery."""

from __future__ import annotations

import dataclasses

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from nanobot.remote.device_proof import (
    DeviceProofClaims,
    canonical_device_proof_bytes,
)
from nanobot.remote.pairing import (
    PairingConsumedError,
    PairingRejectedError,
    PairingService,
    RemoteAccessStore,
    TokenService,
)

NOW = 1_800_000_000.0
ORIGIN = "https://fixture.example:8443"
INSTALLATION_ID = "install-0001"


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _keypair():
    key = ec.generate_private_key(ec.SECP256R1())
    public_der = key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return key, public_der


def _sign(key, claims: DeviceProofClaims) -> bytes:
    return key.sign(canonical_device_proof_bytes(claims), ec.ECDSA(hashes.SHA256()))


def _pairing_service(root):
    clock = _Clock(NOW)
    store = RemoteAccessStore(root)
    store.load_or_create()
    store.set_enabled(True)
    tokens = TokenService(store, clock=clock)
    return store, tokens, clock


def _create_proof_bound_device(service: PairingService, key, public_der: bytes) -> None:
    offer = service.create_challenge(
        origin=ORIGIN,
        label="phone",
        chat_consent=True,
        media_consent=False,
    )
    service.collect(
        pairing_id=offer.pairing_id,
        nonce=offer.nonce,
        collector_secret=offer.collector_secret,
        installation_id=INSTALLATION_ID,
        label="phone",
        device_public_key_der=public_der,
    )
    service.approve(pairing_id=offer.pairing_id, comparison_code=offer.comparison_code)
    claims = DeviceProofClaims.from_payload(service.proof_challenge_claims(offer.pairing_id))
    service.complete_device_proof(
        pairing_id=offer.pairing_id,
        claims=claims,
        signature_der=_sign(key, claims),
    )


def test_proof_bound_device_can_recover_lost_access_token(tmp_path):
    store, tokens, clock = _pairing_service(tmp_path / "remote-access")
    service = PairingService(store, tokens, clock=clock)
    key, public_der = _keypair()
    _create_proof_bound_device(service, key, public_der)

    claims = DeviceProofClaims.from_payload(
        service.begin_device_recovery(
            installation_id=INSTALLATION_ID,
            device_public_key_der=public_der,
        )
    )
    pair = service.complete_device_recovery(
        installation_id=INSTALLATION_ID,
        claims=claims,
        signature_der=_sign(key, claims),
    )

    assert tokens.validate_access(pair.access_token).installation_id == INSTALLATION_ID
    assert pair.refresh_token == ""
    assert pair.refresh_expires_in == 0

    with pytest.raises(PairingConsumedError):
        service.complete_device_recovery(
            installation_id=INSTALLATION_ID,
            claims=claims,
            signature_der=_sign(key, claims),
        )


def test_recovery_survives_pairing_service_restart(tmp_path):
    store, tokens, clock = _pairing_service(tmp_path / "remote-access")
    service = PairingService(store, tokens, clock=clock)
    key, public_der = _keypair()
    _create_proof_bound_device(service, key, public_der)

    restarted = PairingService(store, tokens, clock=clock)
    claims = DeviceProofClaims.from_payload(
        restarted.begin_device_recovery(
            installation_id=INSTALLATION_ID,
            device_public_key_der=public_der,
        )
    )
    pair = restarted.complete_device_recovery(
        installation_id=INSTALLATION_ID,
        claims=claims,
        signature_der=_sign(key, claims),
    )

    assert tokens.validate_access(pair.access_token).installation_id == INSTALLATION_ID
    assert pair.refresh_token == ""
    assert pair.refresh_expires_in == 0


def test_recovery_rejects_wrong_key_wrong_origin_and_revoked_device(tmp_path):
    store, tokens, clock = _pairing_service(tmp_path / "remote-access")
    service = PairingService(store, tokens, clock=clock)
    key, public_der = _keypair()
    _create_proof_bound_device(service, key, public_der)

    _wrong_key, wrong_public_der = _keypair()
    with pytest.raises(PairingRejectedError, match="not current"):
        service.begin_device_recovery(
            installation_id=INSTALLATION_ID,
            device_public_key_der=wrong_public_der,
        )

    claims = DeviceProofClaims.from_payload(
        service.begin_device_recovery(
            installation_id=INSTALLATION_ID,
            device_public_key_der=public_der,
        )
    )
    wrong_origin_claims = dataclasses.replace(claims, origin="https://other.example")
    with pytest.raises(PairingRejectedError, match="wrong_origin"):
        service.complete_device_recovery(
            installation_id=INSTALLATION_ID,
            claims=wrong_origin_claims,
            signature_der=_sign(key, wrong_origin_claims),
        )

    store.revoke_device(INSTALLATION_ID)
    with pytest.raises(PairingRejectedError, match="not recoverable"):
        service.begin_device_recovery(
            installation_id=INSTALLATION_ID,
            device_public_key_der=public_der,
        )
