"""Regression tests for A3-7 pairing persistence across service restarts."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.remote.auth import AuthRejectedError, TokenService
from nanobot.remote.device_proof import DeviceProofClaims
from nanobot.remote.pairing import (
    PairingBusyError,
    PairingDeviceRevokedError,
    PairingNotFoundError,
    PairingOffer,
    PairingService,
)
from nanobot.remote.store import RemoteAccessStore
from tests.remote.proof_helpers import (
    approve_and_prove,
    collect_with_key,
    new_device_key,
    sign_claims,
)

ORIGIN = "https://fixture.example:8443"


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def setup(
    tmp_path: Path,
) -> tuple[RemoteAccessStore, TokenService, PairingService, Clock]:
    store = RemoteAccessStore(tmp_path / "remote-access")
    store.load_or_create()
    store.set_enabled(True)
    clock = Clock()
    tokens = TokenService(store, clock=clock)
    return store, tokens, PairingService(store, tokens, clock=clock), clock


def offer(pairing: PairingService) -> PairingOffer:
    return pairing.create_challenge(
        origin=ORIGIN,
        label="phone",
        chat_consent=True,
        media_consent=False,
    )


def test_pending_collection_survives_restart_and_cancel_clears_it(
    tmp_path: Path,
) -> None:
    store, tokens, pairing, clock = setup(tmp_path)
    created = offer(pairing)
    collect_with_key(pairing,
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
        installation_id="install-0001",
        label="Jaden's phone",
    )

    restarted = PairingService(store, tokens, clock=clock)
    restored = restarted.status(created.pairing_id)
    assert restored.installation_id == "install-0001"
    with pytest.raises(PairingBusyError):
        restarted.collect(
            pairing_id=created.pairing_id,
            nonce=created.nonce,
            collector_secret=created.collector_secret,
            installation_id="install-0001",
            label="second scanner",
        )

    assert restarted.cancel(created.pairing_id) is True

    after_cancel = PairingService(store, tokens, clock=clock)
    with pytest.raises(PairingNotFoundError):
        after_cancel.status(created.pairing_id)


def test_approved_result_is_not_replayed_after_restart(tmp_path: Path) -> None:
    store, tokens, pairing, clock = setup(tmp_path)
    created = offer(pairing)
    collect_with_key(pairing,
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
        installation_id="install-0001",
        label="Jaden's phone",
    )
    approve_and_prove(pairing,
        pairing_id=created.pairing_id,
        comparison_code=created.comparison_code,
    )
    delivered = pairing.poll(
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
    )
    assert delivered.status == "approved"

    restarted = PairingService(store, tokens, clock=clock)
    with pytest.raises(PairingNotFoundError):
        restarted.status(created.pairing_id)


def test_approved_unissued_survives_restart_with_fresh_proof(tmp_path: Path) -> None:
    from nanobot.remote.device_proof import DeviceProofClaims
    from tests.remote.proof_helpers import new_device_key, sign_claims

    store, tokens, pairing, clock = setup(tmp_path)
    created = offer(pairing)
    key, public = new_device_key()
    pairing.collect(pairing_id=created.pairing_id, nonce=created.nonce, collector_secret=created.collector_secret, installation_id="install-0001", label="phone", device_public_key_der=public)
    pairing.approve(pairing_id=created.pairing_id, comparison_code=created.comparison_code)
    before = pairing.proof_challenge_claims(created.pairing_id)
    assert store.get_device("install-0001") is None
    restarted = PairingService(store, TokenService(store, clock=clock), clock=clock)
    assert restarted.status(created.pairing_id).state == "approved"
    after = restarted.proof_challenge_claims(created.pairing_id)
    assert after["nonce"] != before["nonce"]
    claims = DeviceProofClaims.from_payload(after)
    issued = restarted.complete_device_proof(pairing_id=created.pairing_id, claims=claims, signature_der=sign_claims(key, claims))
    saved = restarted._state_path.read_text()
    assert issued.access_token not in saved
    assert "access_token" not in saved
    assert PairingService(store, tokens, clock=clock).pending() == []


@pytest.mark.parametrize("disposition", ["cancel", "expired", "revoked"])
def test_approved_restart_does_not_revive_cancel_expiry_or_revocation(tmp_path, disposition):
    from nanobot.remote.device_proof import DeviceProofClaims
    from nanobot.remote.pairing import PairingError
    from tests.remote.proof_helpers import new_device_key, sign_claims

    store, tokens, pairing, clock = setup(tmp_path)
    created = offer(pairing)
    key, public = new_device_key()
    pairing.collect(pairing_id=created.pairing_id, nonce=created.nonce, collector_secret=created.collector_secret, installation_id="install-0001", label="phone", device_public_key_der=public)
    pairing.approve(pairing_id=created.pairing_id, comparison_code=created.comparison_code)
    if disposition == "cancel":
        pairing.cancel(created.pairing_id)
    elif disposition == "expired":
        clock.now = created.expires_at
    else:
        # A separately current device revocation must also fence a pending approval.
        tokens.issue_device("install-0001", ("chat.read",), device_fingerprint="a" * 64, device_generation=1, origin=ORIGIN)
        tokens.revoke("install-0001")
    restarted = PairingService(store, tokens, clock=clock)
    with pytest.raises(PairingError):
        raw = restarted.proof_challenge_claims(created.pairing_id)
        claims = DeviceProofClaims.from_payload(raw)
        restarted.complete_device_proof(pairing_id=created.pairing_id, claims=claims, signature_der=sign_claims(key, claims))


def test_cancel_old_pairing_preserves_new_pairing_for_same_installation(tmp_path: Path) -> None:
    store, tokens, pairing, _ = setup(tmp_path)
    installation = "install-0001"

    def pair() -> tuple[PairingOffer, str]:
        created = offer(pairing)
        collect_with_key(
            pairing,
            pairing_id=created.pairing_id,
            nonce=created.nonce,
            collector_secret=created.collector_secret,
            installation_id=installation,
            label="phone",
        )
        approve_and_prove(
            pairing, pairing_id=created.pairing_id, comparison_code=created.comparison_code
        )
        result = pairing.poll(
            pairing_id=created.pairing_id,
            nonce=created.nonce,
            collector_secret=created.collector_secret,
        )
        assert result.status == "approved"
        return created, result.access_token

    previous, previous_access = pair()
    store.revoke_device(installation)
    _, current_access = pair()

    pairing.cancel_attempt(
        pairing_id=previous.pairing_id,
        nonce=previous.nonce,
        collector_secret=previous.collector_secret,
    )

    assert tokens.validate_access(current_access).installation_id == installation
    with pytest.raises(AuthRejectedError):
        tokens.validate_access(previous_access)


def test_cancel_issued_pairing_survives_owner_restart(tmp_path: Path) -> None:
    store, _, pairing, clock = setup(tmp_path)
    installation = "install-0001"
    created = offer(pairing)
    key, public = new_device_key()
    pairing.collect(
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
        installation_id=installation,
        label="phone",
        device_public_key_der=public,
    )
    pairing.approve(pairing_id=created.pairing_id, comparison_code=created.comparison_code)
    raw = pairing.proof_challenge_claims(created.pairing_id)
    assert raw is not None
    claims = DeviceProofClaims.from_payload(raw)
    pairing.complete_device_proof(
        pairing_id=created.pairing_id, claims=claims, signature_der=sign_claims(key, claims)
    )

    restarted = PairingService(store, TokenService(store, clock=clock), clock=clock)
    assert restarted.cancel_attempt(
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
    )
    with pytest.raises(PairingDeviceRevokedError):
        restarted.begin_device_recovery(installation_id=installation, device_public_key_der=public)
