from __future__ import annotations

from pathlib import Path

from nanobot.remote.management import RemoteAccessManager
from nanobot.remote.owner import RemoteAccessOwner


def test_local_manager_pair_devices_and_toggle(tmp_path: Path) -> None:
    owner = RemoteAccessOwner(tmp_path / "remote-access", serve_verifier=lambda _origin: True)
    owner.start()
    token = owner.store.admin_token()
    manager = RemoteAccessManager(owner.socket_path, token)
    try:
        assert manager.status()["enabled"] is False
        assert manager.set_enabled(True) is True
        assert manager.devices() == []
        offer = manager.create_pairing(
            origin="https://fixture.example:8443",
            label="phone",
            chat_consent=True,
        )
        assert offer["pairing_id"]
        assert manager.cancel_pairing(offer["pairing_id"]) is True
        assert manager.set_enabled(False) is False
    finally:
        owner.stop()
