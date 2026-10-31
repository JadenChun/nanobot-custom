from __future__ import annotations

import threading
from pathlib import Path

import pytest

from nanobot.remote.auth import TokenService
from nanobot.remote.pairing import (
    PairingBusyError,
    PairingConsumedError,
    PairingExpiredError,
    PairingNotFoundError,
    PairingRateLimitedError,
    PairingRejectedError,
    PairingService,
)
from nanobot.remote.store import RemoteAccessStore


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
    pending = pairing.collect(
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
    approved = pairing.approve(pairing_id=created.pairing_id, comparison_code=created.comparison_code)
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
        pairing.collect(
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
        pairing.collect(
            pairing_id=created.pairing_id,
            nonce="wrong",
            collector_secret=created.collector_secret,
            installation_id="install-0001",
            label="phone",
        )
    with pytest.raises(PairingRejectedError):
        pairing.collect(
            pairing_id=created.pairing_id,
            nonce=created.nonce,
            collector_secret="wrong",
            installation_id="install-0001",
            label="phone",
        )
    for _ in range(3):
        with pytest.raises(PairingRejectedError):
            pairing.collect(
                pairing_id=created.pairing_id,
                nonce="wrong",
                collector_secret="wrong",
                installation_id="install-0001",
                label="phone",
            )
    with pytest.raises(PairingRateLimitedError):
        pairing.collect(
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
    clock.advance(121)
    with pytest.raises(PairingExpiredError):
        pairing.collect(
            pairing_id=created.pairing_id,
            nonce=created.nonce,
            collector_secret=created.collector_secret,
            installation_id="install-0001",
            label="phone",
        )

    fresh = offer(pairing)
    pending = pairing.collect(
        pairing_id=fresh.pairing_id,
        nonce=fresh.nonce,
        collector_secret=fresh.collector_secret,
        installation_id="install-0001",
        label="phone",
    )
    pairing.approve(pairing_id=fresh.pairing_id, comparison_code=pending.comparison_code)
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
        pairing.approve(pairing_id=created.pairing_id, comparison_code="000000")
    pending = pairing.collect(
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
        installation_id="install-0001",
        label="phone",
    )
    with pytest.raises(PairingRejectedError):
        pairing.approve(pairing_id=created.pairing_id, comparison_code="000000")
    approved = pairing.approve(
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
            pairing.collect(
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
            pairing.approve(pairing_id=created.pairing_id, comparison_code=created.comparison_code)
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
