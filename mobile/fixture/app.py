"""Deterministic non-agent fixture for the Nanobot mobile client (M02).

This module intentionally imports no Nanobot agent, gateway, provider or session
code. It exists only to exercise transport, credential, media and replay behavior
against synthetic data on loopback.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

FIXTURE_VERSION = "fixture-v1"
MAX_ATTACHMENT_BYTES = 256 * 1024
MAX_WS_FRAME_BYTES = 64 * 1024


class MutationRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    value: str = Field(default="", max_length=1024)


class AttachmentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    media_type: str = Field(default="application/octet-stream", max_length=128)
    content_base64: str = Field(max_length=400_000)


class AdminMode(BaseModel):
    mode: str = ""
    value: float | str | None = None


def _now() -> float:
    return time.time()


def _token() -> str:
    return secrets.token_urlsafe(32)


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


@dataclass
class FixtureState:
    instance_id: str = "fixture-instance"
    installation_id: str = "fixture-installation"
    bootstrap_token: str = "fixture-bootstrap"
    admin_token: str = "fixture-admin"
    access_ttl: int = 300
    refresh_ttl: int = 3600
    clock: Callable[[], float] = _now
    gateway_generation: int = 1
    access_token: str = ""
    refresh_token: str = ""
    access_expires_at: float = 0.0
    refresh_expires_at: float = 0.0
    revoked: bool = False
    mode: str = ""
    mode_value: float | str | None = None
    effects: int = 0
    handled: dict[str, dict[str, Any]] = field(default_factory=dict)
    reset_count: int = 0

    def reset(self) -> None:
        self.access_token = ""
        self.refresh_token = ""
        self.access_expires_at = 0.0
        self.refresh_expires_at = 0.0
        self.revoked = False
        self.mode = ""
        self.mode_value = None
        self.effects = 0
        self.handled.clear()
        self.reset_count += 1

    def issue_pair(self) -> tuple[str, str]:
        self.access_token = _token()
        self.refresh_token = _token()
        self.access_expires_at = self.clock() + self.access_ttl
        self.refresh_expires_at = self.clock() + self.refresh_ttl
        return self.access_token, self.refresh_token

    def require_access(self, token: str | None) -> None:
        if self.revoked:
            raise _error(403, "revoked", "installation is revoked")
        if not token or token != self.access_token:
            raise _error(401, "bad_access", "access credential is not current")
        if self.clock() >= self.access_expires_at:
            raise _error(401, "expired_access", "access credential has expired")

    def require_refresh(self, token: str) -> None:
        if self.revoked:
            raise _error(403, "revoked", "installation is revoked")
        if token != self.refresh_token:
            raise _error(401, "bad_refresh", "refresh credential is not current")
        if self.clock() >= self.refresh_expires_at:
            raise _error(401, "expired_refresh", "refresh credential has expired")


def _media_payloads() -> dict[str, tuple[str, bytes]]:
    png = bytes.fromhex("89504E470D0A1A0A0000000D49484452") + b"\x00\x00\x00\x01fixture-png"
    mp3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"fixture-mp3-audio"
    mp4 = b"\x00\x00\x00\x18ftypmp42" + b"fixture-mp4-video"
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
        b'<script>globalThis.__fixtureXss=1</script><rect width="16" height="16"/></svg>'
    )
    html = b"<!doctype html><html><body><script>globalThis.__fixtureXss=1</script></body></html>"
    return {
        "sample.png": ("image/png", png),
        "sample.mp3": ("audio/mpeg", mp3),
        "sample.mp4": ("video/mp4", mp4),
        "untrusted.svg": ("image/svg+xml", svg),
        "untrusted.html": ("text/html", html),
    }


MEDIA = _media_payloads()


def _parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    if not header:
        return None
    if not header.startswith("bytes=") or "," in header:
        raise RangeNotSatisfiableError(size)
    spec = header[6:].strip()
    if "-" not in spec:
        raise RangeNotSatisfiableError(size)
    start_text, end_text = spec.split("-", 1)
    try:
        if start_text == "":
            suffix = int(end_text)
            if suffix <= 0:
                raise ValueError
            start = max(0, size - suffix)
            end = size - 1
        else:
            start = int(start_text)
            end = size - 1 if end_text == "" else int(end_text)
    except ValueError as exc:
        raise RangeNotSatisfiableError(size) from exc
    if start < 0 or start >= size or end < start:
        raise RangeNotSatisfiableError(size)
    return start, min(end, size - 1)


class RangeNotSatisfiableError(Exception):
    def __init__(self, size: int) -> None:
        super().__init__("range is not satisfiable")
        self.size = size


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return None
    return value


def create_app(state: FixtureState | None = None) -> FastAPI:
    state = state or FixtureState()
    app = FastAPI(title="Nanobot mobile fixture", version=FIXTURE_VERSION)
    app.state.fixture = state

    def require_admin(token: str | None) -> None:
        if token != state.admin_token:
            raise _error(403, "admin_denied", "fixture admin token is required")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "fixture": True,
            "version": FIXTURE_VERSION,
            "instance_id": state.instance_id,
            "gateway_generation": state.gateway_generation,
        }

    @app.post("/fixture/v1/bootstrap")
    async def bootstrap(
        request: Request,
        x_fixture_bootstrap: str | None = Header(default=None),
    ) -> dict[str, Any]:
        if x_fixture_bootstrap != state.bootstrap_token:
            raise _error(401, "bad_bootstrap", "bootstrap credential is not valid")
        body = await _json_body(request)
        installation = body.get("installation_id", state.installation_id)
        if installation != state.installation_id:
            raise _error(409, "instance_mismatch", "installation does not belong to this fixture")
        if state.mode == "bootstrap_expired":
            raise _error(401, "bootstrap_expired", "bootstrap challenge has expired")
        if state.mode == "bootstrap_revoked" or state.revoked:
            raise _error(403, "revoked", "installation is revoked")
        access, refresh = state.issue_pair()
        return {
            "fixture": True,
            "instance_id": state.instance_id,
            "installation_id": state.installation_id,
            "gateway_generation": state.gateway_generation,
            "access_token": access,
            "refresh_token": refresh,
            "access_expires_in": state.access_ttl,
            "refresh_expires_in": state.refresh_ttl,
            "capabilities": {"chat": True, "media": True, "lifecycle": False},
        }

    @app.post("/fixture/v1/refresh")
    async def refresh(request: Request) -> dict[str, Any]:
        body = await _json_body(request)
        token = str(body.get("refresh_token", ""))
        state.require_refresh(token)
        if state.mode == "refresh_expired":
            raise _error(401, "expired_refresh", "refresh credential has expired")
        access, refresh_token = state.issue_pair()
        if state.mode == "refresh_lost_response":
            raise _error(503, "response_lost", "refresh rotated but the response was lost")
        return {
            "access_token": access,
            "refresh_token": refresh_token,
            "access_expires_in": state.access_ttl,
            "refresh_expires_in": state.refresh_ttl,
        }

    @app.get("/fixture/v1/read")
    async def read(
        authorization: str | None = Header(default=None),
        cursor: str | None = Query(default=None),
    ) -> dict[str, Any]:
        state.require_access(_bearer(authorization))
        if state.mode == "read_wrong_instance":
            raise _error(409, "instance_mismatch", "gateway instance does not match")
        items = [
            {"id": "synthetic-1", "role": "user", "text": "fixture history one"},
            {"id": "synthetic-2", "role": "assistant", "text": "fixture history two"},
        ]
        return {"items": items, "cursor": None if cursor is None else "end"}

    @app.post("/fixture/v1/mutation")
    async def mutation(request: Request, authorization: str | None = Header(default=None)) -> Any:
        state.require_access(_bearer(authorization))
        body = MutationRequest.model_validate(await _json_body(request))
        previous = state.handled.get(body.request_id)
        if previous is not None:
            return {**previous, "replayed": True}
        state.effects += 1
        result = {
            "request_id": body.request_id,
            "effect": state.effects,
            "value": body.value,
        }
        state.handled[body.request_id] = result
        if state.mode == "mutation_delayed":
            await asyncio.sleep(float(state.mode_value or 0.1))
        if state.mode == "mutation_lost_response":
            return StreamingResponse(_drop_after_effect(), media_type="application/json")
        if state.mode == "mutation_lost_inflight":
            raise _error(503, "response_lost", "effect applied but the response was lost")
        return result

    @app.get("/fixture/v1/counter")
    async def counter() -> dict[str, Any]:
        return {"effects": state.effects, "handled": len(state.handled), "resets": state.reset_count}

    @app.get("/fixture/v1/media/{name}")
    async def media(name: str, request: Request, authorization: str | None = Header(default=None)) -> Response:
        state.require_access(_bearer(authorization))
        if state.mode == "media_revoked":
            raise _error(403, "revoked", "media access was revoked")
        entry = MEDIA.get(name)
        if entry is None:
            raise _error(404, "not_found", "unknown fixture media")
        media_type, payload = entry
        try:
            ranged = _parse_range(request.headers.get("range"), len(payload))
        except RangeNotSatisfiableError as exc:
            return Response(
                status_code=416,
                headers={
                    "Content-Range": f"bytes */{exc.size}",
                    "Accept-Ranges": "bytes",
                    "Cache-Control": "no-store",
                },
            )
        etag = '"' + hashlib.sha256(payload).hexdigest()[:16] + '"'
        if ranged is None:
            return Response(
                payload,
                media_type=media_type,
                headers={"Accept-Ranges": "bytes", "ETag": etag, "Cache-Control": "no-store"},
            )
        start, end = ranged
        chunk = payload[start : end + 1]
        return Response(
            chunk,
            status_code=206,
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                "Content-Length": str(len(chunk)),
                "ETag": etag,
                "Cache-Control": "no-store",
            },
        )

    @app.post("/fixture/v1/attachment")
    async def attachment(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        state.require_access(_bearer(authorization))
        body = AttachmentRequest.model_validate(await _json_body(request))
        try:
            payload = base64.b64decode(body.content_base64, validate=True)
        except ValueError as exc:
            raise _error(400, "bad_attachment", "attachment is not valid base64") from exc
        if len(payload) > MAX_ATTACHMENT_BYTES:
            raise _error(413, "attachment_too_large", "attachment exceeds the fixture limit")
        return {
            "name": body.name,
            "media_type": body.media_type,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    @app.get("/fixture/v1/redirect")
    async def redirect(to: str = Query(min_length=1, max_length=2048)) -> Response:
        return Response(status_code=302, headers={"Location": to})

    @app.websocket("/fixture/v1/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            await websocket.send_json(
                {
                    "type": "hello",
                    "fixture": True,
                    "generation": state.gateway_generation,
                }
            )
            if state.mode == "ws_disconnect":
                await websocket.close(code=1008)
                return
            while True:
                message = await websocket.receive_text()
                if len(message.encode("utf-8")) > MAX_WS_FRAME_BYTES:
                    await websocket.send_json({"type": "error", "code": "frame_too_large"})
                    continue
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "code": "malformed_json"})
                    continue
                kind = payload.get("type")
                if kind == "echo":
                    await websocket.send_json({"type": "echo", "value": payload.get("value")})
                elif kind == "stream":
                    count = min(int(payload.get("count", 3)), 20)
                    for index in range(count):
                        await websocket.send_json({"type": "stream", "index": index})
                elif kind == "oversize":
                    await websocket.send_json({"type": "oversize", "value": "x" * (MAX_WS_FRAME_BYTES + 1)})
                elif kind == "binary":
                    await websocket.send_bytes(b"\xff\xfe\x00fixture")
                else:
                    await websocket.send_json({"type": "error", "code": "unknown_frame"})
        except WebSocketDisconnect:
            return

    @app.post("/fixture/v1/admin/reset")
    async def admin_reset(x_fixture_admin: str | None = Header(default=None)) -> dict[str, Any]:
        require_admin(x_fixture_admin)
        state.reset()
        return {"ok": True, "resets": state.reset_count}

    @app.post("/fixture/v1/admin/mode")
    async def admin_mode(body: AdminMode, x_fixture_admin: str | None = Header(default=None)) -> dict[str, Any]:
        require_admin(x_fixture_admin)
        state.mode = body.mode
        state.mode_value = body.value
        if body.mode == "expire_access":
            state.access_expires_at = state.clock() - 1
        if body.mode == "expire_refresh":
            state.refresh_expires_at = state.clock() - 1
        if body.mode == "revoke":
            state.revoked = True
        return {"ok": True, "mode": state.mode}

    @app.post("/fixture/v1/admin/expire")
    async def admin_expire(
        what: str = Query(pattern="^(access|refresh)$"),
        x_fixture_admin: str | None = Header(default=None),
    ) -> dict[str, Any]:
        require_admin(x_fixture_admin)
        if what == "access":
            state.access_expires_at = state.clock() - 1
        else:
            state.refresh_expires_at = state.clock() - 1
        return {"ok": True, "expired": what}

    return app


def _drop_after_effect() -> Any:
    yield b'{"effect":"applied","response":"lost"}'
    raise ConnectionResetError("fixture lost the response after applying the effect")


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        value = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _error(400, "bad_json", "body must be valid JSON") from exc
    if not isinstance(value, dict):
        raise _error(400, "bad_json", "body must be a JSON object")
    return value
