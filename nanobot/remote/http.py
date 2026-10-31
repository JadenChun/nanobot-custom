"""Loopback-only pairing listener for the staged remote owner (M09)."""

from __future__ import annotations

import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, cast

from nanobot.remote.pairing import (
    PairingBusyError,
    PairingConsumedError,
    PairingDisabledError,
    PairingError,
    PairingExpiredError,
    PairingNotFoundError,
    PairingRateLimitedError,
    PairingRejectedError,
    PairingService,
)

MAX_REQUEST_BYTES = 64 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class RemoteHTTPListener:
    """Minimal /remote/v1 pairing listener; no data or local-admin routes."""

    def __init__(
        self,
        pairing: PairingService,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        if host not in LOOPBACK_HOSTS:
            raise ValueError("remote listener must bind to loopback")
        self.pairing = pairing
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    @property
    def running(self) -> bool:
        return self._server is not None

    def start(self) -> int:
        if self._server is not None:
            return self.bound_port
        handler = _make_handler(self.pairing)
        server = ThreadingHTTPServer((self.host, self.port), handler)
        server.daemon_threads = True
        self._server = server
        self._thread = Thread(target=server.serve_forever, name="remote-pairing-http", daemon=True)
        self._thread.start()
        return self.bound_port

    @property
    def bound_port(self) -> int:
        if self._server is None:
            raise RuntimeError("listener is not running")
        return int(self._server.server_address[1])

    def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)


def _make_handler(pairing: PairingService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "nanobot-remote"
        sys_version = ""

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming
            path = _normalized_path(self.path)
            if path == "/health":
                self._write(HTTPStatus.OK, {"ok": True, "version": 1})
                return
            self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler naming
            path = _normalized_path(self.path)
            if path not in {"/pair/collect", "/pair/poll"}:
                self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                payload = self._read_json()
            except ValueError as exc:
                self._write(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            try:
                if path.endswith("/collect"):
                    result = pairing.collect(
                        pairing_id=_required(payload, "pairing_id"),
                        nonce=_required(payload, "nonce"),
                        collector_secret=_required(payload, "collector_secret"),
                        installation_id=_required(payload, "installation_id"),
                        label=_required(payload, "label"),
                    )
                else:
                    result = pairing.poll(
                        pairing_id=_required(payload, "pairing_id"),
                        nonce=_required(payload, "nonce"),
                        collector_secret=_required(payload, "collector_secret"),
                    )
            except PairingRateLimitedError:
                self._write(HTTPStatus.TOO_MANY_REQUESTS, {"error": "rate_limited"})
            except PairingNotFoundError:
                self._write(HTTPStatus.NOT_FOUND, {"error": "pairing_not_found"})
            except PairingExpiredError:
                self._write(HTTPStatus.GONE, {"error": "pairing_expired"})
            except PairingConsumedError:
                self._write(HTTPStatus.GONE, {"error": "pairing_consumed"})
            except PairingBusyError:
                self._write(HTTPStatus.CONFLICT, {"error": "pairing_busy"})
            except PairingRejectedError:
                self._write(HTTPStatus.FORBIDDEN, {"error": "pairing_rejected"})
            except PairingDisabledError:
                self._write(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "remote_access_disabled"})
            except (ValueError, PairingError):
                self._write(HTTPStatus.BAD_REQUEST, {"error": "bad_request"})
            else:
                self._write(HTTPStatus.OK, asdict(result))
            return

        def _read_json(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise ValueError("invalid content length") from exc
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("request body is outside the accepted size")
            body = self.rfile.read(length)
            try:
                value = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("request body is not JSON") from exc
            if not isinstance(value, dict):
                raise ValueError("request body must be an object")
            return cast(dict[str, Any], value)

        def _write(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def _normalized_path(raw: str) -> str:
    path = raw.split("?", 1)[0]
    if path == "/remote/v1":
        return "/"
    if path.startswith("/remote/v1/"):
        return path[len("/remote/v1") :]
    return path


def _required(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} is required")
    return value
