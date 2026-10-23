"""Configuration-aware private desktop adapter for the existing gateway owner."""

from __future__ import annotations

import http.client
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit, urlunsplit

from nanobot import __version__
from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.config.loader import load_config, resolve_config_env_vars, set_config_path
from nanobot.config.schema import Config
from nanobot.gateway.runtime import (
    DesktopAdmissionResult,
    DesktopGatewayIdentity,
    GatewayDesktopLifecycleError,
    GatewayInstance,
    GatewayRuntime,
    GatewayStartOptions,
)
from nanobot.providers.factory import validate_provider_setup
from nanobot.runtime_composition import RuntimePreflightError, runtime_preflight
from nanobot.security.network import is_loopback_host

_CAPABILITY = "managed-gateway-v1"
_SHORT_OPERATION_TIMEOUT_S = 3.0
_HTTP_TIMEOUT_S = 1.0
_MAX_JSON_BYTES = 64 * 1024
_MAX_HTML_BYTES = 1024 * 1024
_MAX_ASSET_BYTES = 16 * 1024 * 1024
_ASSET_REF = re.compile(rb"(?:src|href)=[\"'](/assets/[^\"']+)[\"']")


class DesktopError(ValueError):
    """Stable, content-free error code returned over the private host pipe."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class _Selection:
    config_path: Path
    workspace_selector: str | None
    workspace: Path
    config: Config
    websocket: WebSocketConfig
    instance: GatewayInstance
    options: GatewayStartOptions
    sessions_root: Path
    secret: str
    origin: str
    health_host: str
    health_port: int


class DesktopAdapter:
    """Own one selected gateway adapter session and its client lease."""

    def __init__(self) -> None:
        self._selection: _Selection | None = None
        self._runtime: GatewayRuntime | None = None

    def handle(self, op: str, params: dict[str, object]) -> dict[str, object]:
        if op == "hello":
            return self._hello(params)
        selection = self._selection
        runtime = self._runtime
        if selection is None or runtime is None:
            raise DesktopError("hello_required")
        if op == "start":
            handle = params.get("handle")
            if isinstance(handle, str):
                result = runtime.adopt_desktop_handle(selection.options, handle)
                if not result.ok:
                    if result.message == "gateway_stopped":
                        return self._result("stopped", result.message, None, handle)
                    raise DesktopError(result.message)
                return self._admission_response(result, selection)
            return self._start()
        if op == "status":
            return self._status(cast(str, params.get("handle")))
        if op == "bootstrap":
            return self._bootstrap(cast(str, params.get("handle")))
        if op == "stop":
            return self._stop(cast(str, params.get("handle")))
        raise DesktopError("operation_unsupported")

    def close(self) -> None:
        runtime = self._runtime
        if runtime is not None:
            runtime.release_desktop_lease()
        self._selection = None
        self._runtime = None

    def _hello(self, params: dict[str, object]) -> dict[str, object]:
        raw_config = params.get("config_path")
        raw_workspace = params.get("workspace")
        if not isinstance(raw_config, str) or not Path(raw_config).is_absolute():
            raise DesktopError("config_unavailable")
        if raw_workspace is not None and (
            not isinstance(raw_workspace, str) or not Path(raw_workspace).is_absolute()
        ):
            raise DesktopError("workspace_invalid")
        try:
            config_path = Path(raw_config).expanduser().resolve(strict=True)
            if not config_path.is_file():
                raise OSError
            with config_path.open("rb"):
                pass
        except OSError:
            raise DesktopError("config_unavailable") from None

        existing = self._selection
        workspace_selector = (
            str(Path(raw_workspace).expanduser().resolve(strict=False))
            if isinstance(raw_workspace, str)
            else None
        )
        if existing is not None:
            if (
                config_path != existing.config_path
                or workspace_selector != existing.workspace_selector
            ):
                raise DesktopError("selection_locked")
            self._check_preflight(existing)
            return self._hello_result(existing)

        try:
            config = resolve_config_env_vars(
                load_config(config_path), config_path=config_path
            )
            set_config_path(config_path)
            if workspace_selector is not None:
                config.agents.defaults.workspace = workspace_selector
            workspace = config.workspace_path.expanduser().resolve(strict=False)
            config.agents.defaults.workspace = str(workspace)
            websocket = WebSocketConfig.model_validate(config.channels.websocket)
        except Exception:
            raise DesktopError("config_invalid") from None

        if (
            not is_loopback_host(config.gateway.host)
            or not is_loopback_host(websocket.host)
            or websocket.unix_socket_path
            or websocket.public_ws_url
            or websocket.ssl_certfile
            or websocket.ssl_keyfile
        ):
            raise DesktopError("loopback_required")
        if not websocket.enabled:
            raise DesktopError("websocket_disabled")
        if config.gateway.port == websocket.port:
            raise DesktopError("gateway_listener_conflict")
        secret = websocket.token_issue_secret.strip() or websocket.token.strip()
        if not secret:
            raise DesktopError("webui_auth_required")
        if not (1 <= config.gateway.port <= 65535 and 1 <= websocket.port <= 65535):
            raise DesktopError("config_invalid")

        dist = Path(__file__).resolve().parents[1] / "web" / "dist"
        if not (dist / "index.html").is_file():
            raise DesktopError("webui_assets_unavailable")
        try:
            validate_provider_setup(config)
        except Exception:
            raise DesktopError("provider_unavailable") from None

        instance = GatewayInstance.resolve(
            config_path=config_path,
            workspace=workspace_selector,
        )
        origin_host = _url_host(websocket.host)
        origin = urlunsplit(("http", f"{origin_host}:{websocket.port}", "", "", ""))
        sessions_root = (config.runtime_data_dir or config_path.parent) / "sessions"
        sessions_root = sessions_root.expanduser().resolve(strict=False)
        selection = _Selection(
            config_path=config_path,
            workspace_selector=workspace_selector,
            workspace=workspace,
            config=config,
            websocket=websocket,
            instance=instance,
            options=instance.start_options(
                port=config.gateway.port,
                desktop_managed=True,
                desktop_listener_ports=(websocket.port,),
            ),
            sessions_root=sessions_root,
            secret=secret,
            origin=origin,
            health_host=config.gateway.host,
            health_port=config.gateway.port,
        )
        self._check_preflight(selection)
        self._selection = selection
        self._runtime = GatewayRuntime(paths=instance.paths, python_executable=sys.executable)
        return self._hello_result(selection)

    @staticmethod
    def _hello_result(selection: _Selection) -> dict[str, object]:
        return {
            "protocol_version": 1,
            "backend_version": __version__,
            "python_executable": sys.executable,
            "capabilities": [_CAPABILITY],
            "config_path": str(selection.config_path),
            "workspace": str(selection.workspace),
        }

    @staticmethod
    def _check_preflight(selection: _Selection) -> None:
        try:
            validate_provider_setup(selection.config)
            runtime_preflight(selection.config, sessions_root=selection.sessions_root)
        except RuntimePreflightError:
            raise DesktopError("runtime_preflight_refused") from None
        except Exception:
            raise DesktopError("startup_admission_refused") from None

    def _start(self) -> dict[str, object]:
        selection = self._need_selection()
        runtime = self._need_runtime()
        self._check_preflight(selection)
        result = runtime.start_desktop(
            selection.options,
            preflight=lambda: self._check_preflight(selection),
        )
        if not result.ok:
            return self._result("unavailable", result.message, selection.origin, None)
        return self._admission_response(result, selection)

    @staticmethod
    def _admission_response(
        result: DesktopAdmissionResult,
        selection: _Selection,
    ) -> dict[str, object]:
        identity = result.identity
        if identity is None:
            return {
                "state": "unavailable",
                "code": "gateway_identity_unavailable",
                "origin": selection.origin,
                "handle": None,
            }
        return {
            "state": "starting",
            "code": "gateway_starting",
            "origin": selection.origin,
            "handle": identity.handle,
        }

    def _status(self, handle: str) -> dict[str, object]:
        selection = self._need_selection()
        runtime = self._need_runtime()
        deadline = time.monotonic() + _SHORT_OPERATION_TIMEOUT_S
        try:
            identity = runtime.desktop_identity(
                selection.options,
                handle=handle,
                lock_timeout_s=0.25,
            )
        except GatewayDesktopLifecycleError as exc:
            raise DesktopError(exc.code) from None
        if identity is None:
            return self._result("stopped", "gateway_stopped", None, handle)
        ready, code = self._ready(selection, identity, deadline=deadline)
        state = "running" if ready else "starting" if identity.starting else "unavailable"
        return self._result(state, "gateway_ready" if ready else code, selection.origin, handle)

    def _bootstrap(self, handle: str) -> dict[str, object]:
        selection = self._need_selection()
        runtime = self._need_runtime()
        deadline = time.monotonic() + _SHORT_OPERATION_TIMEOUT_S
        try:
            identity = runtime.desktop_identity(
                selection.options,
                handle=handle,
                lock_timeout_s=0.25,
            )
        except GatewayDesktopLifecycleError as exc:
            raise DesktopError(exc.code) from None
        if identity is None:
            raise DesktopError("gateway_stopped")
        ready, code = self._ready(selection, identity, deadline=deadline)
        if not ready:
            raise DesktopError(code)
        status, payload = _request_json(
            selection.websocket.host,
            selection.websocket.port,
            "/webui/bootstrap",
            headers={"Authorization": f"Bearer {selection.secret}"},
            deadline=deadline,
        )
        if status != 200 or not isinstance(payload, dict):
            raise DesktopError("gateway_unavailable")
        payload_object = cast(dict[str, object], payload)
        ws_url = payload_object.get("ws_url")
        if not isinstance(ws_url, str) or not _valid_ws_url(ws_url, selection.websocket):
            raise DesktopError("gateway_origin_mismatch")
        return {"origin": selection.origin, "secret": selection.secret, "ws_url": ws_url}

    def _stop(self, handle: str) -> dict[str, object]:
        selection = self._need_selection()
        runtime = self._need_runtime()
        result = runtime.stop_desktop(selection.options, handle, timeout_s=20.0, lock_timeout_s=1.0)
        if result.ok:
            return self._result("stopped", "gateway_stopped", None, handle)
        return self._result("unavailable", result.message, selection.origin, handle)

    def _ready(
        self,
        selection: _Selection,
        identity: DesktopGatewayIdentity,
        *,
        deadline: float | None = None,
    ) -> tuple[bool, str]:
        deadline = deadline if deadline is not None else time.monotonic() + _SHORT_OPERATION_TIMEOUT_S
        if identity.starting:
            return False, "gateway_starting"
        if (
            not identity.gateway_id
            or identity.backend_version != __version__
            or identity.capability != _CAPABILITY
            or identity.python_executable != sys.executable
        ):
            return False, "gateway_identity_unavailable"
        try:
            health_status, health = _request_json(
                selection.health_host,
                selection.health_port,
                "/health",
                deadline=deadline,
            )
        except DesktopError as exc:
            return False, exc.code
        if health_status != 200 or not isinstance(health, dict):
            return False, "gateway_unavailable"
        health_object = cast(dict[str, object], health)
        if health_object.get("ready") is not True:
            return False, "gateway_unavailable"
        try:
            _probe_assets(selection, deadline=deadline)
        except DesktopError as exc:
            return False, exc.code
        try:
            terminal_status, terminal = _request_json(
                selection.websocket.host,
                selection.websocket.port,
                "/webui/terminal",
                headers={"Authorization": f"Bearer {selection.secret}"},
                deadline=deadline,
            )
        except DesktopError as exc:
            return False, exc.code
        if terminal_status in {401, 403}:
            return False, "webui_auth_failed"
        if terminal_status != 200 or not isinstance(terminal, dict):
            return False, "webui_unavailable"
        terminal_object = cast(dict[str, object], terminal)
        if (
            type(terminal_object.get("protocolVersion")) is not int
            or terminal_object.get("protocolVersion") != 1
            or terminal_object.get("gatewayId") != identity.gateway_id
        ):
            return False, "gateway_identity_mismatch"
        return True, "gateway_ready"

    def _need_selection(self) -> _Selection:
        if self._selection is None:
            raise DesktopError("hello_required")
        return self._selection

    def _need_runtime(self) -> GatewayRuntime:
        if self._runtime is None:
            raise DesktopError("hello_required")
        return self._runtime

    @staticmethod
    def _result(
        state: str,
        code: str,
        origin: str | None,
        handle: str | None,
    ) -> dict[str, object]:
        return {"state": state, "code": code, "origin": origin, "handle": handle}


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _request_bytes(
    host: str,
    port: int,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    max_bytes: int,
    deadline: float | None = None,
) -> tuple[int, bytes]:
    timeout = _HTTP_TIMEOUT_S
    if deadline is not None:
        timeout = min(timeout, deadline - time.monotonic())
        if timeout <= 0:
            raise DesktopError("gateway_unavailable")
    connection = http.client.HTTPConnection(host.strip("[]"), port, timeout=timeout)
    try:
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise DesktopError("http_response_too_large")
        return response.status, body
    except DesktopError:
        raise
    except (OSError, http.client.HTTPException, TimeoutError, ValueError):
        raise DesktopError("gateway_unavailable") from None
    finally:
        connection.close()


def _request_json(
    host: str,
    port: int,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    deadline: float | None = None,
) -> tuple[int, object]:
    status, body = _request_bytes(
        host,
        port,
        path,
        headers=headers,
        max_bytes=_MAX_JSON_BYTES,
        deadline=deadline,
    )
    try:
        value: object = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, None
    return status, value


def _probe_assets(selection: _Selection, *, deadline: float) -> None:
    status, body = _request_bytes(
        selection.websocket.host,
        selection.websocket.port,
        "/",
        max_bytes=_MAX_HTML_BYTES,
        deadline=deadline,
    )
    if status != 200 or b"<html" not in body.lower():
        raise DesktopError("webui_assets_unavailable")
    refs = _ASSET_REF.findall(body)
    if not refs or len(refs) > 32:
        raise DesktopError("webui_assets_unavailable")
    for raw_path in refs:
        path = raw_path.decode("ascii", errors="strict")
        asset_status, _asset = _request_bytes(
            selection.websocket.host,
            selection.websocket.port,
            path,
            max_bytes=_MAX_ASSET_BYTES,
            deadline=deadline,
        )
        if asset_status != 200:
            raise DesktopError("webui_assets_unavailable")


def _valid_ws_url(value: str, websocket: WebSocketConfig) -> bool:
    try:
        parsed = urlsplit(value)
        expected_host = websocket.host.strip("[]").casefold()
        return (
            parsed.scheme == "ws"
            and (parsed.hostname or "").casefold() == expected_host
            and parsed.port == websocket.port
            and parsed.path == websocket.path
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False

