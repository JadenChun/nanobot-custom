from __future__ import annotations

import re
import threading
from pathlib import Path

import pytest

from nanobot.remote.auth import TokenService
from nanobot.remote.pairing import (
    PairingBusyError,
    PairingCodeInvalidError,
    PairingConsumedError,
    PairingDeniedError,
    PairingExpiredError,
    PairingNotFoundError,
    PairingOwnerRequiredError,
    PairingRateLimitedError,
    PairingRejectedError,
    PairingService,
)
from nanobot.remote.protocol import OWNER_PAIRING_SCOPES
from nanobot.remote.store import RemoteAccessStore
from tests.remote.proof_helpers import (
    approve_and_prove,
    collect_by_code_with_key,
    collect_with_key,
)


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def setup(tmp_path: Path, clock: Clock | None = None) -> tuple[RemoteAccessStore, TokenService, PairingService]:
    store = RemoteAccessStore(tmp_path / "remote-access")
    store.load_or_create()
    store.set_enabled(True)
    clock = clock or Clock()
    tokens = TokenService(store, clock=clock)
    return store, tokens, PairingService(store, tokens, clock=clock)


def offer(pairing: PairingService, **overrides):
    values = {
        "origin": "https://fixture.example:8443",
        "label": "phone",
        "chat_consent": True,
        "media_consent": False,
    }
    values.update(overrides)
    return pairing.create_challenge(**values)


def test_pairing_flow_and_one_time_result(tmp_path) -> None:
    store, tokens, pairing = setup(tmp_path)
    created = offer(pairing)
    pending = collect_with_key(pairing,
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
        installation_id="install-0001",
        label="Jaden's phone",
    )
    assert pending.comparison_code == created.comparison_code
    assert isinstance(
        pairing.poll(
            pairing_id=created.pairing_id,
            nonce=created.nonce,
            collector_secret=created.collector_secret,
        ),
        type(pending),
    )
    approved = approve_and_prove(pairing, pairing_id=created.pairing_id, comparison_code=created.comparison_code)
    assert approved.installation_id == "install-0001"
    result = pairing.poll(
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
    )
    assert result.status == "approved"
    assert tokens.validate_access(result.access_token).installation_id == "install-0001"
    with pytest.raises(PairingConsumedError):
        pairing.poll(
            pairing_id=created.pairing_id,
            nonce=created.nonce,
            collector_secret=created.collector_secret,
        )
    with pytest.raises(PairingBusyError):
        collect_with_key(pairing,
            pairing_id=created.pairing_id,
            nonce=created.nonce,
            collector_secret=created.collector_secret,
            installation_id="install-0002",
            label="second scanner",
        )


def test_pairing_negatives_and_rate_limit(tmp_path) -> None:
    store, tokens, pairing = setup(tmp_path)
    created = offer(pairing)
    with pytest.raises(PairingRejectedError):
        collect_with_key(pairing,
            pairing_id=created.pairing_id,
            nonce="wrong",
            collector_secret=created.collector_secret,
            installation_id="install-0001",
            label="phone",
        )
    with pytest.raises(PairingRejectedError):
        collect_with_key(pairing,
            pairing_id=created.pairing_id,
            nonce=created.nonce,
            collector_secret="wrong",
            installation_id="install-0001",
            label="phone",
        )
    for _ in range(3):
        with pytest.raises(PairingRejectedError):
            collect_with_key(pairing,
                pairing_id=created.pairing_id,
                nonce="wrong",
                collector_secret="wrong",
                installation_id="install-0001",
                label="phone",
            )
    with pytest.raises(PairingRateLimitedError):
        collect_with_key(pairing,
            pairing_id=created.pairing_id,
            nonce="wrong",
            collector_secret="wrong",
            installation_id="install-0001",
            label="phone",
        )


def test_pairing_expiry_and_restart(tmp_path) -> None:
    clock = Clock()
    store, tokens, pairing = setup(tmp_path, clock)
    created = offer(pairing)
    clock.advance(301)
    with pytest.raises(PairingExpiredError):
        collect_with_key(pairing,
            pairing_id=created.pairing_id,
            nonce=created.nonce,
            collector_secret=created.collector_secret,
            installation_id="install-0001",
            label="phone",
        )

    fresh = offer(pairing)
    pending = collect_with_key(pairing,
        pairing_id=fresh.pairing_id,
        nonce=fresh.nonce,
        collector_secret=fresh.collector_secret,
        installation_id="install-0001",
        label="phone",
    )
    approve_and_prove(pairing, pairing_id=fresh.pairing_id, comparison_code=pending.comparison_code)
    result = pairing.poll(
        pairing_id=fresh.pairing_id,
        nonce=fresh.nonce,
        collector_secret=fresh.collector_secret,
    )
    restarted = PairingService(store, tokens, clock=clock)
    with pytest.raises(PairingNotFoundError):
        restarted.collect(
            pairing_id=fresh.pairing_id,
            nonce=fresh.nonce,
            collector_secret=fresh.collector_secret,
            installation_id="install-0001",
            label="phone",
        )
    assert restarted.tokens.validate_access(result.access_token).installation_id == "install-0001"


def test_pairing_consent_scope_reduction_and_wrong_code(tmp_path) -> None:
    store, tokens, pairing = setup(tmp_path)
    created = offer(pairing, media_consent=True)
    assert "media.read" in created.scopes
    assert "attachment.write" in created.scopes
    with pytest.raises(PairingRejectedError):
        approve_and_prove(pairing, pairing_id=created.pairing_id, comparison_code="000000")
    pending = collect_with_key(pairing,
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
        installation_id="install-0001",
        label="phone",
    )
    with pytest.raises(PairingRejectedError):
        approve_and_prove(pairing, pairing_id=created.pairing_id, comparison_code="000000")
    approved = approve_and_prove(pairing,
        pairing_id=created.pairing_id,
        comparison_code=pending.comparison_code,
        scopes=("chat.read",),
    )
    assert approved.scopes == ("chat.read",)
    assert store.get_device("install-0001").scopes == ("chat.read",)


def test_concurrent_collect_and_approve_have_one_winner(tmp_path) -> None:
    store, tokens, pairing = setup(tmp_path)
    created = offer(pairing)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def collect(index: int) -> None:
        barrier.wait()
        try:
            collect_with_key(pairing,
                pairing_id=created.pairing_id,
                nonce=created.nonce,
                collector_secret=created.collector_secret,
                installation_id=f"install-{index:04d}",
                label="phone",
            )
            outcomes.append("ok")
        except PairingBusyError:
            outcomes.append("busy")

    threads = [threading.Thread(target=collect, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["busy", "ok"]
    pending = pairing.poll(
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
    )
    assert pending.installation_id in {"install-0000", "install-0001"}

    approve_outcomes: list[str] = []
    approve_barrier = threading.Barrier(2)

    def approve() -> None:
        approve_barrier.wait()
        try:
            approve_and_prove(pairing, pairing_id=created.pairing_id, comparison_code=created.comparison_code)
            approve_outcomes.append("ok")
        except PairingBusyError:
            approve_outcomes.append("busy")

    approvers = [threading.Thread(target=approve) for _ in range(2)]
    for thread in approvers:
        thread.start()
    for thread in approvers:
        thread.join()
    assert sorted(approve_outcomes) == ["busy", "ok"]
    assert len(store.snapshot()["devices"]) == 1


def test_owner_only_offer_requires_consent_and_complete_scopes(tmp_path) -> None:
    _store, _tokens, pairing = setup(tmp_path)
    with pytest.raises(PairingOwnerRequiredError):
        offer(pairing, owner_only=True, full_control_consent=False)
    with pytest.raises(PairingOwnerRequiredError):
        offer(
            pairing,
            owner_only=True,
            full_control_consent=True,
            scopes=("chat.read", "chat.write"),
        )
    created = offer(pairing, owner_only=True, full_control_consent=True)
    assert set(OWNER_PAIRING_SCOPES).issubset(created.scopes)
    assert created.media_consent is True
    assert created.transport == "tailscale"


def test_eight_digit_code_is_unique_and_supports_manual_approval(tmp_path) -> None:
    _store, _tokens, pairing = setup(tmp_path)
    offers = [offer(pairing) for _index in range(25)]
    codes = [created.pairing_code for created in offers]
    assert all(re.fullmatch(r"\d{8}", code) for code in codes)
    assert len(set(codes)) == len(codes)
    assert all(created.comparison_code == created.pairing_code for created in offers)

    created = offers[0]
    pending = collect_by_code_with_key(pairing,
        origin=created.origin,
        pairing_code=created.pairing_code,
        installation_id="install-9001",
        label="Pixel 10",
        platform="Android 16",
        model="Pixel 10",
    )
    assert pending.state == "pending_approval"
    assert pending.pairing_code == created.pairing_code
    assert pending.platform == "Android 16"
    assert pending.model == "Pixel 10"
    assert pending.origin == created.origin

    with pytest.raises(PairingCodeInvalidError):
        approve_and_prove(pairing, pairing_id=created.pairing_id)
    with pytest.raises(PairingCodeInvalidError):
        approve_and_prove(pairing, pairing_id=created.pairing_id, pairing_code="00000000")
    approved = approve_and_prove(pairing,
        pairing_id=created.pairing_id,
        pairing_code=created.pairing_code,
    )
    assert approved.installation_id == "install-9001"
    assert pairing.status(created.pairing_id).state == "approved"


def test_wrong_origin_and_repeated_code_attempts_are_rejected(tmp_path) -> None:
    _store, _tokens, pairing = setup(tmp_path)
    created = offer(pairing)
    with pytest.raises(PairingCodeInvalidError):
        collect_by_code_with_key(pairing,
            origin="https://other.example",
            pairing_code=created.pairing_code,
            installation_id="install-9002",
            label="phone",
        )
    for _attempt in range(5):
        with pytest.raises(PairingCodeInvalidError):
            collect_by_code_with_key(pairing,
                origin=created.origin,
                pairing_code="00000000",
                installation_id="install-9002",
                label="phone",
            )
    with pytest.raises(PairingRateLimitedError):
        collect_by_code_with_key(pairing,
            origin=created.origin,
            pairing_code="00000000",
            installation_id="install-9002",
            label="phone",
        )


def test_qr_approval_needs_no_comparison_code_and_pending_metadata(tmp_path) -> None:
    _store, _tokens, pairing = setup(tmp_path)
    created = offer(pairing, owner_only=True, full_control_consent=True)
    status = pairing.status(created.pairing_id)
    assert status.state == "offer_created"
    assert status.label == "phone"

    pending = collect_with_key(pairing,
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
        installation_id="install-9003",
        label="Pixel 10",
        platform="Android 16",
    )
    assert pending.state == "pending_approval"
    pending_list = pairing.pending()
    assert len(pending_list) == 1
    assert pending_list[0].state == "pending_approval"
    assert pending_list[0].platform == "Android 16"

    approved = approve_and_prove(pairing, pairing_id=created.pairing_id)
    assert approved.installation_id == "install-9003"
    assert set(OWNER_PAIRING_SCOPES).issubset(approved.scopes)


def test_deny_and_expiry_states(tmp_path) -> None:
    clock = Clock()
    _store, _tokens, pairing = setup(tmp_path, clock)
    created = offer(pairing, owner_only=True, full_control_consent=True)
    collect_with_key(pairing,
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
        installation_id="install-9004",
        label="phone",
    )
    denied = pairing.deny(created.pairing_id)
    assert denied.state == "denied"
    assert pairing.status(created.pairing_id).state == "denied"
    assert pairing.pending() == []
    with pytest.raises(PairingDeniedError):
        approve_and_prove(pairing, pairing_id=created.pairing_id)

    expired = offer(pairing, owner_only=True, full_control_consent=True)
    clock.advance(301)
    assert pairing.status(expired.pairing_id).state == "expired"
    with pytest.raises(PairingExpiredError):
        collect_with_key(pairing,
            pairing_id=expired.pairing_id,
            nonce=expired.nonce,
            collector_secret=expired.collector_secret,
            installation_id="install-9005",
            label="phone",
        )


def test_renew_extends_an_expired_offer_within_the_setup_limit(tmp_path) -> None:
    clock = Clock()
    _store, _tokens, pairing = setup(tmp_path, clock)
    created = offer(pairing)
    clock.advance(301)
    assert pairing.status(created.pairing_id).state == "expired"

    renewed = pairing.renew(created.pairing_id)
    assert renewed.expires_at > int(clock.now)
    assert renewed.expires_at <= created.expires_at - 300 + 1800
    assert renewed.renewable is True
    assert renewed.state == "offer_created"
    assert pairing.status(created.pairing_id).expires_at == renewed.expires_at


def test_renew_keeps_the_collected_device_binding_and_polling(tmp_path) -> None:
    clock = Clock()
    _store, _tokens, pairing = setup(tmp_path, clock)
    created = offer(pairing)
    pending = collect_with_key(pairing,
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
        installation_id="install-0001",
        label="phone",
    )
    clock.advance(301)
    renewed = pairing.renew(created.pairing_id)
    assert renewed.state == "pending_approval"
    assert renewed.installation_id == pending.installation_id
    assert isinstance(
        pairing.poll(
            pairing_id=created.pairing_id,
            nonce=created.nonce,
            collector_secret=created.collector_secret,
        ),
        type(pending),
    )


def test_renew_is_refused_after_terminal_or_overall_limit(tmp_path) -> None:
    clock = Clock()
    _store, _tokens, pairing = setup(tmp_path, clock)
    created = offer(pairing)
    collect_with_key(pairing,
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
        installation_id="install-0001",
        label="phone",
    )
    approve_and_prove(pairing, pairing_id=created.pairing_id, comparison_code=created.comparison_code)
    with pytest.raises(PairingConsumedError):
        pairing.renew(created.pairing_id)

    late = offer(pairing)
    clock.advance(1801)
    with pytest.raises(PairingExpiredError):
        pairing.renew(late.pairing_id)
