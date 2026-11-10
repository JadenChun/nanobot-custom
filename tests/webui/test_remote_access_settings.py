from __future__ import annotations

import json
from pathlib import Path

from websockets.http11 import Request as WsRequest

from nanobot.config.loader import save_config
from nanobot.config.schema import Config
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.webui.settings_remote import handle_remote_access
from nanobot.webui.settings_services import WebUISettingsConfig
from tests.remote.proof_helpers import new_device_key


def _request(path: str, payload: dict | None = None) -> WsRequest:
    request = WsRequest(path, {})
    if payload is not None:
        setattr(request, "_nanobot_webui_mutation_payload", payload)
    return request


def test_remote_access_settings_routes(tmp_path: Path) -> None:
    owner = RemoteAccessOwner(tmp_path / "remote-access", serve_verifier=lambda _origin: True)
    owner.start()
    config_path = tmp_path / "config.json"
    config = Config.model_validate(
        {
            "remoteAccess": {
                "enabled": False,
                "stateDir": str(owner.store.root),
                "socketPath": str(owner.socket_path),
            }
        }
    )
    save_config(config, config_path)
    settings = WebUISettingsConfig(config_path)
    try:
        response = handle_remote_access(settings, _request("/api/settings/remote-access"), "/api/settings/remote-access")
        assert response.status_code == 200
        assert json.loads(response.body)["enabled"] is False
        enabled = handle_remote_access(
            settings,
            _request("/api/settings/remote-access/enable", {"enabled": True}),
            "/api/settings/remote-access/enable",
        )
        assert enabled.status_code == 200
        assert settings.load().remote_access.enabled is True
        offer = handle_remote_access(
            settings,
            _request(
                "/api/settings/remote-access/pairing/start",
                {
                    "origin": "https://fixture.example:8443",
                    "label": "phone",
                    "fullControlConsent": True,
                },
            ),
            "/api/settings/remote-access/pairing/start",
        )
        assert offer.status_code == 200
        pairing = json.loads(offer.body)
        assert pairing["pairing_id"]
        pair = owner.tokens.issue("settings-install", ("chat.read",))
        assert pair.access_token
        revoked = handle_remote_access(
            settings,
            _request("/api/settings/remote-access/revoke", {"installation_id": "settings-install"}),
            "/api/settings/remote-access/revoke",
        )
        assert revoked.status_code == 200
        assert owner.store.get_device("settings-install").revoked
    finally:
        owner.stop()


def test_owner_pairing_start_requires_consent_and_exposes_pending(tmp_path: Path) -> None:
    owner = RemoteAccessOwner(tmp_path / "remote-access", serve_verifier=lambda _origin: True)
    owner.start()
    owner.store.set_enabled(True)
    config_path = tmp_path / "config.json"
    config = Config.model_validate(
        {
            "remoteAccess": {
                "enabled": True,
                "stateDir": str(owner.store.root),
                "socketPath": str(owner.socket_path),
            }
        }
    )
    save_config(config, config_path)
    settings = WebUISettingsConfig(config_path)
    try:
        rejected = handle_remote_access(
            settings,
            _request(
                "/api/settings/remote-access/pairing/start",
                {"origin": "https://fixture.example:8443", "label": "phone"},
            ),
            "/api/settings/remote-access/pairing/start",
        )
        assert rejected.status_code == 400

        offer = handle_remote_access(
            settings,
            _request(
                "/api/settings/remote-access/pairing/start",
                {
                    "origin": "https://fixture.example:8443",
                    "label": "Pixel 10",
                    "fullControlConsent": True,
                },
            ),
            "/api/settings/remote-access/pairing/start",
        )
        assert offer.status_code == 200
        pairing = json.loads(offer.body)
        assert pairing["state"] == "offer_created"
        assert pairing["transport"] == "tailscale"
        assert pairing["expires_in"] > 0
        assert len(pairing["pairing_code"]) == 8
        assert pairing["qr_link"].startswith("nanobot://pair?")

        pending = handle_remote_access(
            settings,
            _request("/api/settings/remote-access/pairing/pending"),
            "/api/settings/remote-access/pairing/pending",
        )
        assert pending.status_code == 200
        assert json.loads(pending.body)["pending"][0]["pairing_id"] == pairing["pairing_id"]

        _key, public_der = new_device_key()
        owner.pairing.collect(
            pairing_id=pairing["pairing_id"],
            nonce=pairing["nonce"],
            collector_secret=pairing["collector_secret"],
            installation_id="settings-owner-1",
            label="Pixel 10",
            platform="Android 16",
            device_public_key_der=public_der,
        )
        pending = handle_remote_access(
            settings,
            _request("/api/settings/remote-access/pairing/pending"),
            "/api/settings/remote-access/pairing/pending",
        )
        current = json.loads(pending.body)["pending"][0]
        assert current["state"] == "pending_approval"
        assert current["device"]["platform"] == "Android 16"
        assert current["origin"] == "https://fixture.example:8443"

        approved = handle_remote_access(
            settings,
            _request(
                "/api/settings/remote-access/pairing/approve",
                {"pairingId": pairing["pairing_id"]},
            ),
            "/api/settings/remote-access/pairing/approve",
        )
        assert approved.status_code == 200
        assert json.loads(approved.body)["state"] == "approved"

        manual = json.loads(
            handle_remote_access(
                settings,
                _request(
                    "/api/settings/remote-access/pairing/start",
                    {
                        "origin": "https://fixture.example:8443",
                        "label": "Manual phone",
                        "fullControlConsent": True,
                    },
                ),
                "/api/settings/remote-access/pairing/start",
            ).body
        )
        _manual_key, manual_public = new_device_key()
        owner.pairing.collect_by_code(
            origin="https://fixture.example:8443",
            pairing_code=manual["pairing_code"],
            installation_id="settings-owner-2",
            label="Manual phone",
            device_public_key_der=manual_public,
        )
        missing_code = handle_remote_access(
            settings,
            _request(
                "/api/settings/remote-access/pairing/approve",
                {"pairingId": manual["pairing_id"]},
            ),
            "/api/settings/remote-access/pairing/approve",
        )
        assert missing_code.status_code == 403
        assert json.loads(missing_code.body)["error"] == "pairing_code_invalid"
        approved_manual = handle_remote_access(
            settings,
            _request(
                "/api/settings/remote-access/pairing/approve",
                {
                    "pairingId": manual["pairing_id"],
                    "pairingCode": manual["pairing_code"],
                },
            ),
            "/api/settings/remote-access/pairing/approve",
        )
        assert approved_manual.status_code == 200
        denied = json.loads(
            handle_remote_access(
                settings,
                _request(
                    "/api/settings/remote-access/pairing/start",
                    {
                        "origin": "https://fixture.example:8443",
                        "label": "Denied phone",
                        "fullControlConsent": True,
                    },
                ),
                "/api/settings/remote-access/pairing/start",
            ).body
        )
        _deny_key, deny_public = new_device_key()
        owner.pairing.collect(
            pairing_id=denied["pairing_id"],
            nonce=denied["nonce"],
            collector_secret=denied["collector_secret"],
            installation_id="settings-owner-3",
            label="Denied phone",
            device_public_key_der=deny_public,
        )
        denied_response = handle_remote_access(
            settings,
            _request(
                "/api/settings/remote-access/pairing/deny",
                {"pairingId": denied["pairing_id"]},
            ),
            "/api/settings/remote-access/pairing/deny",
        )
        assert denied_response.status_code == 200
        assert json.loads(denied_response.body)["state"] == "denied"
    finally:
        owner.stop()


def test_tailscale_readiness_route(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    save_config(Config(), config_path)
    settings = WebUISettingsConfig(config_path)
    monkeypatch.setattr(
        "nanobot.remote.tailscale.tailscale_status",
        lambda: {
            "available": True,
            "connected": True,
            "serve_mapped": True,
            "dns_name": "desktop.tailnet.ts.net",
            "origin": "https://desktop.tailnet.ts.net",
            "error": "",
        },
    )
    response = handle_remote_access(
        settings,
        _request("/api/settings/remote-access/tailscale"),
        "/api/settings/remote-access/tailscale",
    )
    assert response.status_code == 200
    assert json.loads(response.body)["origin"] == "https://desktop.tailnet.ts.net"
