"""Isolated mobile data plane for live M14 qualification.

This is a bounded staging server, not the production gateway. It exercises the
same owner validation, mobile edge routes, refresh rotation and replay cache
against a loopback listener.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from websockets.datastructures import Headers
from websockets.http11 import Request as WsRequest

from nanobot.remote.client import RemoteAccessClient
from nanobot.remote.edge import EdgeAuthError, MediaResult, MobileGatewayEdge, OwnerAccessValidator
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.remote.protocol import MutationEnvelope
from nanobot.remote.replay import OperationReplayCache

MAX_BODY = 1024 * 1024


class MobileDataServer:
    def __init__(self, owner: RemoteAccessOwner, *, port: int = 0) -> None:
        self.owner = owner
        self.client = RemoteAccessClient(owner.socket_path, timeout=2.0)
        validator = OwnerAccessValidator(self.client, cache_ttl=0.0)
        self.replay = OperationReplayCache()
        self.effects = 0
        self.edge = MobileGatewayEdge(
            validator,
            generation_provider=lambda: 1,
            socket_epoch=1,
            read_handler=self._read,
            media_handler=self._media,
            replay_cache=self.replay,
            mutation_handler=self.mutation,
        )
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

    async def _read(self, _identity: Any, _query: dict[str, list[str]]) -> dict[str, Any]:
        return {"sessions": []}

    async def _media(self, _identity: Any, _sig: str, _payload: str, _range: str | None) -> MediaResult:
        return MediaResult(404, b"not found", "text/plain", {})

    async def mutation(self, _identity: Any, _envelope: MutationEnvelope) -> dict[str, Any]:
        self.effects += 1
        return {"effect": self.effects}

    def run(self, coroutine: Any) -> Any:
        if self._loop is None:
            raise RuntimeError("data server loop is not running")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout=30)

    def start(self) -> int:
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._loop_thread = threading.Thread(target=loop.run_forever, name="mobile-data-loop", daemon=True)
        self._loop_thread.start()
        handler = _make_handler(self)
        server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="mobile-data", daemon=True)
        self._thread.start()
        return int(server.server_address[1])

    def stop(self) -> None:
        loop = self._loop
        self._loop = None
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)
        loop_thread = self._loop_thread
        self._loop_thread = None
        if loop_thread is not None and loop_thread.is_alive():
            loop_thread.join(timeout=3)


def _make_handler(data: MobileDataServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "nanobot-mobile-data"
        sys_version = ""

        def do_GET(self) -> None:  # noqa: N802
            path = _normalize(self.path)
            if not path.startswith("/mobile/v1/"):
                self._json(404, {"error": "not_found"})
                return
            request = WsRequest(
                "/mobile/v1/" + path[len("/mobile/v1/") :],
                Headers([(key, value) for key, value in self.headers.items()]),
            )
            response = data.run(data.edge.handle(None, request))
            if response is None:
                self._json(404, {"error": "not_found"})
                return
            self._response(response)

        def do_POST(self) -> None:  # noqa: N802
            path = _normalize(self.path)
            if not path.startswith("/mobile/v1/"):
                self._json(404, {"error": "not_found"})
                return
            token = _bearer(self.headers.get("Authorization"))
            if token is None:
                self._json(401, {"error": "missing_access"})
                return
            try:
                identity = data.run(data.edge.validator.validate(token))
            except EdgeAuthError as exc:
                self._json(exc.status, {"error": exc.code})
                return
            try:
                body = self._body()
            except ValueError:
                self._json(400, {"error": "bad_body"})
                return
            if path == "/mobile/v1/refresh":
                refresh_token = body.get("refresh_token")
                if not isinstance(refresh_token, str):
                    self._json(400, {"error": "refresh_token_required"})
                    return
                try:
                    result = data.client.refresh_access(identity.installation_id, refresh_token)
                except Exception:
                    self._json(401, {"error": "refresh_rejected"})
                    return
                self._json(200, result)
                return
            if path == "/mobile/v1/mutation":
                raw_envelope = body.get("envelope")
                if not isinstance(raw_envelope, dict):
                    self._json(400, {"error": "invalid_envelope"})
                    return
                try:
                    envelope = MutationEnvelope.model_validate(raw_envelope)
                except ValueError:
                    self._json(400, {"error": "invalid_envelope"})
                    return
                if envelope.installation_id != identity.installation_id:
                    self._json(403, {"error": "installation_mismatch"})
                    return
                outcome = data.run(
                    data.replay.execute(
                        envelope,
                        current_generation=1,
                        current_epoch=1,
                        authorize=_always_true,
                        effect=lambda envelope: data.mutation(None, envelope),
                    )
                )
                self._json(200, outcome.model_dump(mode="json"))
                return
            if path == "/mobile/v1/logout":
                try:
                    data.client.revoke(identity.installation_id)
                except Exception:
                    self._json(503, {"error": "revoke_unavailable"})
                    return
                self._json(200, {"installation_id": identity.installation_id, "erased": True})
                return
            self._json(404, {"error": "not_found"})

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY:
                raise ValueError("body size")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("body object")
            return cast(dict[str, Any], value)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _response(self, response: Any) -> None:
            self.send_response(int(response.status_code))
            for key, value in response.headers.raw_items():
                if key.lower() in {"content-length", "connection", "date"}:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(response.body)))
            self.end_headers()
            self.wfile.write(response.body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def _normalize(path: str) -> str:
    path = path.split("?", 1)[0]
    if path == "/mobile/v1":
        return "/mobile/v1"
    if path.startswith(("/handshake", "/read", "/refresh", "/mutation", "/logout", "/media/")):
        return "/mobile/v1" + path
    return path


def _bearer(value: str | None) -> str | None:
    if not value or not value.lower().startswith("bearer "):
        return None
    token = value[7:].strip()
    return token or None


async def _always_true() -> bool:
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8125)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()
    owner = RemoteAccessOwner(args.state_dir, args.socket)
    owner.start()
    owner.store.set_enabled(True)
    data = MobileDataServer(owner, port=args.port)
    port = data.start()
    args.ready_file.write_text(
        json.dumps({"data": f"http://127.0.0.1:{port}", "socket": str(owner.socket_path)}),
        encoding="utf-8",
    )
    print(f"MOBILE_DATA_READY {port}", flush=True)
    stop = threading.Event()

    def _stop(_signum: int, _frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        stop.wait()
    finally:
        data.stop()
        owner.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
