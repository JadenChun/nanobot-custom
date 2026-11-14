"""Regression tests for A3-7 bearer cutover and exact Serve binding.

Covers AC7.1 and AC7.4 (keyless owner offers are refused, legacy owner bearer
devices are revoked at startup) and AC7.2 (the Serve target must match the
expected port and the local Tailscale host exactly).
"""

from pathlib import Path

import pytest

from nanobot.remote import tailscale
from nanobot.remote.auth import AuthUpgradeRequiredError, TokenService
from nanobot.remote.client import OwnerRequestError, RemoteAccessClient
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.remote.pairing import (
    PairingService,
    PairingUpgradeRequiredError,
)
from nanobot.remote.store import RemoteAccessStore

ORIGIN = "https://fixture.example:8443"


def _pairing(tmp_path: Path):
    store = RemoteAccessStore(tmp_path / "remote-access")
    store.load_or_create()
    store.set_enabled(True)
    tokens = TokenService(store)
    return store, tokens, PairingService(store, tokens)


def _owner_offer(pairing):
    return pairing.create_challenge(
        origin=ORIGIN,
        label="phone",
        chat_consent=True,
        media_consent=False,
        owner_only=True,
        full_control_consent=True,
    )


def _serve_shape(host: str, port: int, path: str = "/mobile/v1") -> dict:
    return {
        "TCP": {"443": {"HTTPS": True}},
        "Web": {
            f"{host}:443": {
                "Handlers": {
                    path: {"Proxy": f"http://127.0.0.1:{port}/mobile/v1"},
                },
            },
        },
    }


def _fake_run_json_factory(serve_shape):
    def fake_run_json(*args, **kwargs):
        if args and isinstance(args[0], (list, tuple)):
            argv = tuple(args[0])
        else:
            argv = tuple(arg for arg in args if isinstance(arg, str))
        if "serve" in argv:
            return serve_shape()
        if "status" in argv:
            return {
                "BackendState": "Running",
                "Self": {"DNSName": "desktop.tailnet.ts.net."},
            }
        raise AssertionError(f"unexpected tailscale command: {argv!r}")

    return fake_run_json


def test_keyless_owner_offer_is_refused(tmp_path):
    """AC7.1: an owner offer without a device proof binding is refused."""
    store, _tokens, pairing = _pairing(tmp_path)
    offer = _owner_offer(pairing)

    with pytest.raises(PairingUpgradeRequiredError):
        pairing.collect(
            pairing_id=offer.pairing_id,
            nonce=offer.nonce,
            collector_secret=offer.collector_secret,
            installation_id="install-keyless",
            label="Keyless phone",
        )

    assert store.get_device("install-keyless") is None
    assert store.snapshot()["devices"] == []


def test_startup_revokes_legacy_owner_bearer_and_refresh_is_refused(tmp_path):
    """AC7.4: a legacy bearer-only owner device is revoked and cannot refresh."""
    store, tokens, _pairing_service = _pairing(tmp_path)

    pair = tokens.issue(
        "install-legacy-owner",
        ("owner", "chat.read", "chat.write"),
        "legacy owner",
    )

    assert store.get_device_proof("install-legacy-owner") is None
    access_before_cutover = tokens.refresh(
        "install-legacy-owner", pair.refresh_token
    ).access_token
    assert access_before_cutover

    revoked = store.revoke_legacy_bearer_devices()

    assert "install-legacy-owner" in revoked
    assert store.get_device("install-legacy-owner").revoked is True
    with pytest.raises(AuthUpgradeRequiredError):
        tokens.refresh("install-legacy-owner", pair.refresh_token)


def test_exact_serve_target_parsing(monkeypatch):
    """AC7.2: Serve must map the expected port on the local Tailscale host."""
    serve = {"host": "desktop.tailnet.ts.net", "port": 8767}

    monkeypatch.setattr(
        tailscale.shutil, "which", lambda name: "/usr/bin/tailscale"
    )
    monkeypatch.setattr(
        tailscale,
        "_run_json",
        _fake_run_json_factory(
            lambda: _serve_shape(serve["host"], serve["port"])
        ),
    )

    status = tailscale.tailscale_status(expected_port=8767)

    assert status["connected"] is True
    assert status["serve_mapped"] is True
    assert status["serve_target"] == "http://127.0.0.1:8767/mobile/v1"
    assert status["origin"] == "https://desktop.tailnet.ts.net"

    assert tailscale.tailscale_status(expected_port=9999)["serve_mapped"] is False

    serve["host"] = "other.tailnet.ts.net"
    assert tailscale.tailscale_status(expected_port=8767)["serve_mapped"] is False


def test_owner_offer_creation_refuses_mismatched_origin(tmp_path):
    """AC7.2: the owner refuses an offer for an unverified Serve origin."""
    owner = RemoteAccessOwner(
        tmp_path / "remote-access",
        serve_verifier=lambda origin: origin == ORIGIN,
    )
    owner.start()
    try:
        client = RemoteAccessClient(
            owner.socket_path,
            admin_token=owner.store.admin_token(),
        )
        client.set_enabled(True)

        with pytest.raises(OwnerRequestError) as rejected:
            client.create_pairing(
                origin="https://wrong.example",
                label="phone",
                chat_consent=True,
                media_consent=False,
                full_control_consent=True,
                shared_history_consent=True,
            )
        assert rejected.value.code == "serve_unverified"

        offer = client.create_pairing(
            origin=ORIGIN,
            label="phone",
            chat_consent=True,
            media_consent=False,
            full_control_consent=True,
            shared_history_consent=True,
        )
        assert offer["origin"] == ORIGIN
    finally:
        owner.stop()
