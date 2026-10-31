from __future__ import annotations

import json
from pathlib import Path

from websockets.http11 import Request as WsRequest

from nanobot.config.loader import save_config
from nanobot.config.schema import Config
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.webui.settings_remote import handle_remote_access
from nanobot.webui.settings_services import WebUISettingsConfig


def _request(path: str, payload: dict | None = None) -> WsRequest:
    request = WsRequest(path, {})
    if payload is not None:
        setattr(request, "_nanobot_webui_mutation_payload", payload)
    return request


def test_remote_access_settings_routes(tmp_path: Path) -> None:
    owner = RemoteAccessOwner(tmp_path / "remote-access")
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
                {"origin": "https://fixture.example:8443", "label": "phone"},
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
