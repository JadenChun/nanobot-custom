"""Local-only remote-access settings routes (M18)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from websockets.http11 import Request as WsRequest
from websockets.http11 import Response

from nanobot.config.schema import Config, RemoteAccessConfig
from nanobot.remote.client import OwnerRequestError, OwnerUnavailableError
from nanobot.remote.management import RemoteAccessManager
from nanobot.remote.store import RemoteAccessStore
from nanobot.webui.http_utils import http_error, http_json_response
from nanobot.webui.settings_services import WebUISettingsConfig

_PREFIX = "/api/settings/remote-access"


def owner_paths(config: RemoteAccessConfig) -> tuple[Path, Path]:
    if config.state_dir:
        root = Path(config.state_dir).expanduser()
    else:
        from nanobot.config.paths import get_data_dir

        root = get_data_dir() / "remote-access"
    socket = Path(config.socket_path).expanduser() if config.socket_path else root / "owner.sock"
    return root, socket


def _manager(config: WebUISettingsConfig) -> tuple[RemoteAccessManager, Config]:
    cfg = config.load()
    root, socket = owner_paths(cfg.remote_access)
    token = RemoteAccessStore(root).admin_token()
    return RemoteAccessManager(socket, token), cfg


def _status(manager: RemoteAccessManager, cfg: Config) -> dict[str, Any]:
    try:
        owner = manager.status()
        return {
            "enabled": cfg.remote_access.enabled,
            "installation_id": owner.get("installation_id", ""),
            "devices": owner.get("devices", []),
            "owner_available": True,
            "revision": owner.get("revision", 0),
        }
    except (OwnerUnavailableError, OwnerRequestError, OSError) as exc:
        return {
            "enabled": cfg.remote_access.enabled,
            "installation_id": "",
            "devices": [],
            "owner_available": False,
            "error": str(exc),
        }


def handle_remote_access(
    config: WebUISettingsConfig,
    request: WsRequest,
    path: str,
) -> Response:
    manager, cfg = _manager(config)
    if path == _PREFIX:
        return http_json_response(_status(manager, cfg))
    raw_body = getattr(request, "_nanobot_webui_mutation_payload", None)
    if not isinstance(raw_body, dict):
        return http_error(400, "invalid remote-access payload")
    body = cast(dict[str, Any], raw_body)
    try:
        if path == f"{_PREFIX}/enable":
            enabled = bool(body.get("enabled"))
            config.update(lambda current: setattr(current.remote_access, "enabled", enabled))
            manager.set_enabled(enabled)
            return http_json_response({"enabled": enabled})
        if path == f"{_PREFIX}/revoke":
            installation_id = body.get("installation_id")
            if not isinstance(installation_id, str):
                return http_error(400, "installation_id is required")
            actor = getattr(request, "_nanobot_owner_installation", None)
            return http_json_response(
                manager.revoke(
                    installation_id,
                    actor_installation_id=actor if isinstance(actor, str) else "local",
                )
            )
        if path == f"{_PREFIX}/pairing/start":
            origin = body.get("origin")
            label = body.get("label", "mobile")
            if not isinstance(origin, str):
                return http_error(400, "origin is required")
            return http_json_response(
                manager.create_pairing(
                    origin=origin,
                    label=str(label),
                    chat_consent=True,
                    media_consent=bool(body.get("mediaConsent", False)),
                    full_control_consent=bool(body.get("fullControlConsent", False)),
                    shared_history_consent=bool(body.get("sharedHistoryConsent", False)),
                )
            )
        if path == f"{_PREFIX}/pairing/approve":
            pairing_id = body.get("pairingId")
            comparison_code = body.get("comparisonCode")
            if not isinstance(pairing_id, str) or not isinstance(comparison_code, str):
                return http_error(400, "pairing id and comparison code are required")
            return http_json_response(
                manager.approve_pairing(
                    pairing_id=pairing_id,
                    comparison_code=comparison_code,
                )
            )
        return http_error(404, "remote-access route not found")
    except (OwnerUnavailableError, OwnerRequestError, OSError) as exc:
        return http_error(503, str(exc))
