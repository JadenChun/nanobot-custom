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
            "enabled": bool(owner.get("enabled", cfg.remote_access.enabled)),
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
    if path == f"{_PREFIX}/tailscale":
        from nanobot.remote.tailscale import tailscale_status

        return http_json_response(tailscale_status())
    if path == f"{_PREFIX}/pairing/pending":
        try:
            return http_json_response(manager.pending_pairings())
        except OwnerUnavailableError as exc:
            return http_json_response(
                {"error": "owner_unavailable", "message": str(exc)},
                status=503,
            )
        except OwnerRequestError as exc:
            return _owner_error_response(exc)
    raw_body = getattr(request, "_nanobot_webui_mutation_payload", None)
    if not isinstance(raw_body, dict):
        return http_error(400, "invalid remote-access payload")
    body = cast(dict[str, Any], raw_body)
    try:
        if path == f"{_PREFIX}/enable":
            enabled = bool(body.get("enabled"))
            manager.set_enabled(enabled)
            config.update(lambda current: setattr(current.remote_access, "enabled", enabled))
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
            if body.get("fullControlConsent") is not True:
                return http_json_response(
                    {
                        "error": "owner_required",
                        "message": "fullControlConsent=true is required for owner pairing",
                    },
                    status=400,
                )
            return http_json_response(
                manager.create_pairing(
                    origin=origin,
                    label=str(label),
                    chat_consent=True,
                    media_consent=True,
                    full_control_consent=True,
                    shared_history_consent=bool(body.get("sharedHistoryConsent", False)),
                )
            )
        if path == f"{_PREFIX}/pairing/approve":
            pairing_id = body.get("pairingId")
            decision = body.get("decision", "approve")
            if not isinstance(pairing_id, str):
                return http_error(400, "pairing id is required")
            if decision == "deny":
                return http_json_response(manager.deny_pairing(pairing_id))
            code = body.get("pairingCode", body.get("comparisonCode"))
            if code is not None and not isinstance(code, str):
                return http_error(400, "pairing code must be a string")
            return http_json_response(
                manager.approve_pairing(
                    pairing_id=pairing_id,
                    comparison_code=code if isinstance(code, str) else None,
                    pairing_code=code if isinstance(code, str) else None,
                )
            )
        if path == f"{_PREFIX}/pairing/deny":
            pairing_id = body.get("pairingId")
            if not isinstance(pairing_id, str):
                return http_error(400, "pairing id is required")
            return http_json_response(manager.deny_pairing(pairing_id))
        if path == f"{_PREFIX}/pairing/cancel":
            pairing_id = body.get("pairingId")
            if not isinstance(pairing_id, str):
                return http_error(400, "pairing id is required")
            return http_json_response({"cancelled": manager.cancel_pairing(pairing_id)})
        return http_error(404, "remote-access route not found")
    except OwnerUnavailableError as exc:
        return http_json_response(
            {"error": "owner_unavailable", "message": str(exc)},
            status=503,
        )
    except OwnerRequestError as exc:
        return _owner_error_response(exc)
    except OSError as exc:
        return http_error(503, str(exc))


_OWNER_ERROR_STATUS = {
    "owner_required": 403,
    "pairing_expired": 410,
    "pairing_code_invalid": 403,
    "pairing_denied": 403,
    "pairing_rate_limited": 429,
    "pairing_not_found": 404,
    "pairing_busy": 409,
    "pairing_consumed": 410,
    "remote_access_disabled": 503,
}


def _owner_error_response(exc: OwnerRequestError) -> Response:
    return http_json_response(
        {"error": exc.code, "message": str(exc)},
        status=_OWNER_ERROR_STATUS.get(exc.code, 503),
    )
