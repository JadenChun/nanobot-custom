"""Regression tests for A3-7 device proofs."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from nanobot.remote.device_proof import (
    DeviceProofClaims,
    DeviceProofError,
    canonical_device_proof_bytes,
    device_proof_fingerprint,
    verify_device_proof,
)
from nanobot.remote.pairing import (
    PairingConsumedError,
    PairingExpiredError,
    PairingRejectedError,
    PairingService,
    RemoteAccessStore,
    TokenService,
)

NOW = 1_800_000_000.0
ORIGIN = "https://fixture.example:8443"
INSTALLATION_ID = "install-0001"


class _MovableClock:
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


def _claims(public_der: bytes) -> DeviceProofClaims:
    return DeviceProofClaims(
        instance_id="instance-0001",
        origin=ORIGIN,
        installation_id=INSTALLATION_ID,
        device_fingerprint=device_proof_fingerprint(public_der),
        device_generation=1,
        challenge_id="challenge-0001",
        nonce="nonce-0001",
        expires_at=int(NOW) + 120,
    )


def _sign(key, claims: DeviceProofClaims) -> bytes:
    return key.sign(canonical_device_proof_bytes(claims), ec.ECDSA(hashes.SHA256()))


def _expected_kwargs(claims: DeviceProofClaims) -> dict[str, object]:
    return {
        "expected_instance_id": claims.instance_id,
        "expected_origin": claims.origin,
        "expected_installation_id": claims.installation_id,
        "expected_device_generation": claims.device_generation,
        "expected_challenge_id": claims.challenge_id,
        "expected_nonce": claims.nonce,
    }


def _pairing_service(root):
    clock = _MovableClock(NOW)
    store = RemoteAccessStore(root)
    store.load_or_create()
    store.set_enabled(True)
    tokens = TokenService(store, clock=clock)
    return PairingService(store, tokens, clock=clock), tokens, clock


def _start_pairing(service, public_der: bytes):
    offer = service.create_challenge(
        origin=ORIGIN,
        label="phone",
        chat_consent=True,
        media_consent=False,
    )
    pending = service.collect(
        pairing_id=offer.pairing_id,
        nonce=offer.nonce,
        collector_secret=offer.collector_secret,
        installation_id=INSTALLATION_ID,
        label="phone",
        device_public_key_der=public_der,
    )
    service.approve(pairing_id=offer.pairing_id, comparison_code=offer.comparison_code)
    claims_mapping = service.proof_challenge_claims(offer.pairing_id)
    assert claims_mapping is not None
    return offer, pending, DeviceProofClaims(**claims_mapping)


def test_canonical_bytes_are_sorted_and_compact():
    _key, public_der = _keypair()
    claims = _claims(public_der)

    expected = json.dumps(
        claims.payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    assert canonical_device_proof_bytes(claims) == expected


def test_valid_proof_verifies_and_fingerprint_matches():
    key, public_der = _keypair()
    fingerprint = device_proof_fingerprint(public_der)
    assert fingerprint == hashlib.sha256(public_der).hexdigest()

    claims = _claims(public_der)
    signature = _sign(key, claims)

    verified = verify_device_proof(
        claims,
        public_key_der=public_der,
        signature_der=signature,
        now=NOW,
        **_expected_kwargs(claims),
    )

    assert verified == fingerprint
    assert claims.device_fingerprint == fingerprint


def test_expired_proof_is_rejected():
    key, public_der = _keypair()
    claims = _claims(public_der)
    signature = _sign(key, claims)

    with pytest.raises(DeviceProofError) as excinfo:
        verify_device_proof(
            claims,
            public_key_der=public_der,
            signature_der=signature,
            now=claims.expires_at,
            **_expected_kwargs(claims),
        )

    assert excinfo.value.reason == "expired_proof"


def test_wrong_key_signature_is_rejected():
    _registered_key, public_der = _keypair()
    claims = _claims(public_der)

    other_key, _other_public_der = _keypair()
    signature = _sign(other_key, claims)

    with pytest.raises(DeviceProofError) as excinfo:
        verify_device_proof(
            claims,
            public_key_der=public_der,
            signature_der=signature,
            now=NOW,
            **_expected_kwargs(claims),
        )

    assert excinfo.value.reason == "invalid_signature"


def test_wrong_origin_instance_and_generation_are_rejected():
    key, public_der = _keypair()
    claims = _claims(public_der)
    signature = _sign(key, claims)
    expected = _expected_kwargs(claims)

    with pytest.raises(DeviceProofError) as origin_excinfo:
        verify_device_proof(
            dataclasses.replace(claims, origin="https://other.example:8443"),
            public_key_der=public_der,
            signature_der=signature,
            now=NOW,
            **expected,
        )
    assert origin_excinfo.value.reason == "wrong_origin"

    with pytest.raises(DeviceProofError) as instance_excinfo:
        verify_device_proof(
            dataclasses.replace(claims, instance_id="instance-0002"),
            public_key_der=public_der,
            signature_der=signature,
            now=NOW,
            **expected,
        )
    assert instance_excinfo.value.reason == "wrong_instance"

    with pytest.raises(DeviceProofError) as generation_excinfo:
        verify_device_proof(
            dataclasses.replace(claims, device_generation=2),
            public_key_der=public_der,
            signature_der=signature,
            now=NOW,
            **expected,
        )
    assert generation_excinfo.value.reason == "stale_device_generation"


def test_pairing_proof_issues_access_once_and_replay_is_consumed(tmp_path):
    service, tokens, _clock = _pairing_service(tmp_path / "remote-access")
    key, public_der = _keypair()
    offer, _pending, claims = _start_pairing(service, public_der)
    signature = _sign(key, claims)

    pair = service.complete_device_proof(
        pairing_id=offer.pairing_id,
        claims=claims,
        signature_der=signature,
    )

    grant = tokens.validate_access(pair.access_token)
    assert grant.installation_id == INSTALLATION_ID

    with pytest.raises(PairingConsumedError):
        service.complete_device_proof(
            pairing_id=offer.pairing_id,
            claims=claims,
            signature_der=signature,
        )


def test_pairing_rejects_wrong_key_and_wrong_origin_before_issuance(tmp_path):
    service, tokens, _clock = _pairing_service(tmp_path / "wrong-key")
    key, public_der = _keypair()
    offer, _pending, claims = _start_pairing(service, public_der)
    wrong_key, _wrong_public_der = _keypair()

    with pytest.raises(PairingRejectedError) as key_excinfo:
        service.complete_device_proof(
            pairing_id=offer.pairing_id,
            claims=claims,
            signature_der=_sign(wrong_key, claims),
        )
    assert "invalid_signature" in str(key_excinfo.value)

    pair = service.complete_device_proof(
        pairing_id=offer.pairing_id,
        claims=claims,
        signature_der=_sign(key, claims),
    )
    assert tokens.validate_access(pair.access_token).installation_id == INSTALLATION_ID

    service, tokens, _clock = _pairing_service(tmp_path / "wrong-origin")
    key, public_der = _keypair()
    offer, _pending, claims = _start_pairing(service, public_der)
    wrong_origin_claims = dataclasses.replace(claims, origin="https://other.example:8443")

    with pytest.raises(PairingRejectedError) as origin_excinfo:
        service.complete_device_proof(
            pairing_id=offer.pairing_id,
            claims=wrong_origin_claims,
            signature_der=_sign(key, wrong_origin_claims),
        )
    assert "wrong_origin" in str(origin_excinfo.value)

    pair = service.complete_device_proof(
        pairing_id=offer.pairing_id,
        claims=claims,
        signature_der=_sign(key, claims),
    )
    assert tokens.validate_access(pair.access_token).installation_id == INSTALLATION_ID


def test_pairing_expired_proof_is_rejected(tmp_path):
    service, _tokens, clock = _pairing_service(tmp_path / "remote-access")
    key, public_der = _keypair()
    offer, pending, claims = _start_pairing(service, public_der)
    signature = _sign(key, claims)

    clock.now = pending.proof_challenge["expires_at"] + 1
    assert clock.now < offer.expires_at

    with pytest.raises(PairingExpiredError):
        service.complete_device_proof(
            pairing_id=offer.pairing_id,
            claims=claims,
            signature_der=signature,
        )
