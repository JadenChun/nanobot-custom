"""Regression tests for A3-7 pairing persistence across service restarts."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.remote.auth import TokenService
from nanobot.remote.pairing import (
    PairingBusyError,
    PairingNotFoundError,
    PairingOffer,
    PairingService,
)
from nanobot.remote.store import RemoteAccessStore

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
    pairing.collect(
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
    pairing.collect(
        pairing_id=created.pairing_id,
        nonce=created.nonce,
        collector_secret=created.collector_secret,
        installation_id="install-0001",
        label="Jaden's phone",
    )
    pairing.approve(
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
