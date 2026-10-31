"""Authenticated `/mobile/v1` gateway edge over the independent owner IPC (M11).

The edge is intentionally a separate front door: legacy WebUI, desktop, settings
and native routes are untouched. Every mobile request is validated against the
owner's current access grant and scope; unknown routes deny.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import re
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Awaitable, Callable, cast

from websockets.http11 import Request as WsRequest
from websockets.http11 import Response

from nanobot.remote.client import OwnerRequestError, OwnerUnavailableError, RemoteAccessClient
from nanobot.remote.protocol import (
    BRIDGE_VERSION,
    MOBILE_PROTOCOL_VERSION,
    MutationEnvelope,
    MutationOutcome,
    default_limits,
)
from nanobot.remote.replay import OperationReplayCache
from nanobot.webui.http_utils import (
    bearer_token,
    http_json_response,
    parse_request_path,
)

_MEDIA_RE = re.compile(r"^/mobile/v1/media/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)$")
_LOCAL_ROUTES = {
    "/mobile/v1/handshake": "chat.read",
    "/mobile/v1/read": "chat.read",
    "/mobile/v1/status": "chat.read",
}
_PAIRING_ROUTES = {"/mobile/v1/pair/collect", "/mobile/v1/pair/poll"}
_PAIRING_STATUS = {
    "pairing_rate_limited": 429,
    "pairing_not_found": 404,
    "pairing_expired": 410,
    "pairing_consumed": 410,
    "pairing_busy": 409,
    "remote_access_disabled": 503,
    "pairing_rejected": 403,
    "bad_request": 400,
}


class EdgeAuthError(RuntimeError):
    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class AccessIdentity:
    installation_id: str
    scopes: tuple[str, ...]
    expires_at: float
    generation: int

    def allows(self, scope: str) -> bool:
        return scope in self.scopes


@dataclass(frozen=True)
class MediaResult:
    status: int
    body: bytes
    media_type: str
    headers: dict[str, str]


class OwnerAccessValidator:
    """Cache a validated owner grant for a short, tested revocation bound."""

    def __init__(
        self,
        client: RemoteAccessClient,
        *,
        cache_ttl: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.cache_ttl = cache_ttl
        self.clock = clock
        self._lock = asyncio.Lock()
        self._cache: dict[str, tuple[float, AccessIdentity]] = {}

    async def validate(self, access_token: str) -> AccessIdentity:
        key = hashlib.sha256(access_token.encode("utf-8")).hexdigest()
        now = self.clock()
        async with self._lock:
            cached = self._cache.get(key)
            if cached is not None and cached[0] > now:
                return cached[1]
        try:
            result = await asyncio.to_thread(self.client.validate_access, access_token)
        except OwnerUnavailableError as exc:
            raise EdgeAuthError(503, "owner_unavailable") from exc
        except OwnerRequestError as exc:
            raise EdgeAuthError(401, "invalid_access") from exc
        if not result.get("valid"):
            async with self._lock:
                self._cache.pop(key, None)
            raise EdgeAuthError(401, "invalid_access")
        scopes_raw = result.get("scopes")
        if not isinstance(scopes_raw, list):
            raise EdgeAuthError(401, "invalid_access")
        raw_scopes = cast(list[Any], scopes_raw)
        if not all(isinstance(scope, str) for scope in raw_scopes):
            raise EdgeAuthError(401, "invalid_access")
        scopes = tuple(scope for scope in raw_scopes if isinstance(scope, str))
        identity = AccessIdentity(
            installation_id=str(result.get("installation_id", "")),
            scopes=scopes,
            expires_at=float(result.get("expires_at", 0)),
            generation=int(result.get("generation", 1)),
        )
        if not identity.installation_id:
            raise EdgeAuthError(401, "invalid_access")
        async with self._lock:
            self._cache[key] = (now + self.cache_ttl, identity)
        return identity

    async def invalidate(self, installation_id: str | None = None) -> None:
        async with self._lock:
            if installation_id is None:
                self._cache.clear()
                return
            for key, (_expiry, identity) in list(self._cache.items()):
                if identity.installation_id == installation_id:
                    self._cache.pop(key, None)


class MobileGatewayEdge:
    def __init__(
        self,
        validator: OwnerAccessValidator,
        *,
        generation_provider: Callable[[], int],
        socket_epoch: int | None = None,
        read_handler: Callable[[AccessIdentity, dict[str, list[str]]], Awaitable[dict[str, Any]]] | None = None,
        media_handler: Callable[[AccessIdentity, str, str, str | None], Awaitable[MediaResult]] | None = None,
        attachment_handler: Callable[[AccessIdentity, bytes], Awaitable[dict[str, Any]]] | None = None,
        logout_handler: Callable[[AccessIdentity], Awaitable[dict[str, Any]]] | None = None,
        replay_cache: OperationReplayCache | None = None,
        mutation_handler: Callable[[AccessIdentity, MutationEnvelope], Awaitable[dict[str, Any]]] | None = None,
        authorize_operation: Callable[[AccessIdentity], Awaitable[bool]] | None = None,
    ) -> None:
        self.validator = validator
        self.generation_provider = generation_provider
        self.socket_epoch = socket_epoch if socket_epoch is not None else int(time.time())
        self.read_handler = read_handler
        self.media_handler = media_handler
        self.attachment_handler = attachment_handler
        self.logout_handler = logout_handler
        self.replay_cache = replay_cache
        self.mutation_handler = mutation_handler
        self.authorize_operation = authorize_operation

    async def authorize_websocket(self, connection: Any, request: WsRequest) -> Response | None:
        token = bearer_token(request.headers)
        if not token:
            return http_json_response({"error": "missing_access"}, status=401)
        try:
            identity = await self.validator.validate(token)
        except EdgeAuthError as exc:
            return http_json_response({"error": exc.code}, status=exc.status)
        setattr(connection, "mobile_identity", identity)
        return None

    async def handle(self, _connection: Any, request: WsRequest) -> Response | None:
        path, query = parse_request_path(request.path)
        if path != "/mobile/v1" and not path.startswith("/mobile/v1/"):
            return None
        if path in _PAIRING_ROUTES:
            return await self._handle_pairing(request, path)
        token = bearer_token(request.headers)
        if not token:
            return http_json_response({"error": "missing_access"}, status=401)
        try:
            identity = await self.validator.validate(token)
        except EdgeAuthError as exc:
            return http_json_response({"error": exc.code}, status=exc.status)

        route = _LOCAL_ROUTES.get(path)
        if route is not None:
            scope = route
            if not identity.allows(scope):
                return http_json_response({"error": "scope_denied"}, status=403)
            if path == "/mobile/v1/handshake":
                return http_json_response(self._handshake(identity))
            if path == "/mobile/v1/status":
                return http_json_response(
                    {
                        "state": "ready",
                        "installation_id": identity.installation_id,
                        "generation": self.generation_provider(),
                        "socket_epoch": self.socket_epoch,
                    }
                )
            if path == "/mobile/v1/read":
                if self.read_handler is None:
                    return http_json_response({"error": "read_unavailable"}, status=503)
                return http_json_response(await self.read_handler(identity, query))

        media_match = _MEDIA_RE.match(path)
        if media_match is not None:
            if not identity.allows("media.read"):
                return http_json_response({"error": "scope_denied"}, status=403)
            if self.media_handler is None:
                return http_json_response({"error": "media_unavailable"}, status=503)
            result = await self.media_handler(
                identity,
                media_match.group(1),
                media_match.group(2),
                request.headers.get("Range"),
            )
            if result.status < 200 or result.status >= 300:
                return http_json_response({"error": "media_denied"}, status=result.status)
            if len(result.body) > self._media_limit():
                return http_json_response({"error": "media_too_large"}, status=413)
            headers = [
                ("Content-Type", result.media_type),
                ("Cache-Control", "no-store"),
                *result.headers.items(),
            ]
            reason = HTTPStatus(result.status).phrase
            return Response(result.status, reason, _headers(headers), result.body)
        return http_json_response({"error": "not_found"}, status=404)

    async def _handle_pairing(self, request: WsRequest, path: str) -> Response:
        headers = request.headers

        def header(name: str) -> str:
            value = headers.get(name)
            return value.strip() if isinstance(value, str) else ""

        pairing_id = header("X-Nanobot-Pairing-Id")
        nonce = header("X-Nanobot-Pairing-Nonce")
        collector_secret = header("X-Nanobot-Pairing-Secret")
        if not pairing_id or not nonce or not collector_secret:
            return http_json_response({"error": "bad_request"}, status=400)
        try:
            if path.endswith("/collect"):
                installation_id = header("X-Nanobot-Installation")
                label = header("X-Nanobot-Label") or "mobile"
                if not installation_id:
                    return http_json_response({"error": "bad_request"}, status=400)
                result = await asyncio.to_thread(
                    self.validator.client.collect_pairing,
                    pairing_id=pairing_id,
                    nonce=nonce,
                    collector_secret=collector_secret,
                    installation_id=installation_id,
                    label=label,
                )
            else:
                result = await asyncio.to_thread(
                    self.validator.client.poll_pairing,
                    pairing_id=pairing_id,
                    nonce=nonce,
                    collector_secret=collector_secret,
                )
        except OwnerUnavailableError:
            return http_json_response({"error": "owner_unavailable"}, status=503)
        except OwnerRequestError as exc:
            status = _PAIRING_STATUS.get(exc.code, 400)
            return http_json_response({"error": exc.code}, status=status)
        return http_json_response(result)

    def _handshake(self, identity: AccessIdentity) -> dict[str, Any]:
        capabilities = {
            "chat": True,
            "media": "media.read" in identity.scopes,
            "attachments": "attachment.write" in identity.scopes,
            "lifecycle": False,
        }
        return {
            "mobile_protocol": MOBILE_PROTOCOL_VERSION,
            "bridge_version": BRIDGE_VERSION,
            "gateway_generation": self.generation_provider(),
            "socket_epoch": self.socket_epoch,
            "server_time": int(time.time()),
            "principal": f"mobile:{identity.installation_id}",
            "installation_id": identity.installation_id,
            "scopes": list(identity.scopes),
            "capabilities": capabilities,
            "limits": default_limits(),
        }

    def authorize_ws_frame(self, identity: AccessIdentity, epoch: int, scope: str) -> None:
        if epoch != self.socket_epoch:
            raise EdgeAuthError(409, "epoch_mismatch")
        if not identity.allows(scope):
            raise EdgeAuthError(403, "scope_denied")

    async def handle_ws_frame(
        self, identity: AccessIdentity, epoch: int, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle mutation-bearing mobile WS frames with the same identity/epoch."""
        frame_type = payload.get("type")
        if frame_type == "attachment":
            self.authorize_ws_frame(identity, epoch, "attachment.write")
            if self.attachment_handler is None:
                return {"ok": False, "error": "attachment_unavailable"}
            raw = payload.get("data_base64")
            if not isinstance(raw, str):
                return {"ok": False, "error": "invalid_attachment"}
            try:
                body = base64.b64decode(raw, validate=True)
            except ValueError:
                return {"ok": False, "error": "invalid_attachment"}
            if len(body) > self._attachment_limit():
                return {"ok": False, "error": "attachment_too_large"}
            return {"ok": True, "result": await self.attachment_handler(identity, body)}
        if frame_type == "mutation":
            self.authorize_ws_frame(identity, epoch, "chat.write")
            if self.replay_cache is None or self.mutation_handler is None:
                return {"ok": False, "error": "mutation_unavailable"}
            raw_envelope = payload.get("envelope")
            if not isinstance(raw_envelope, dict):
                return {"ok": False, "error": "invalid_envelope"}
            try:
                envelope = MutationEnvelope.model_validate(raw_envelope)
            except ValueError:
                return {"ok": False, "error": "invalid_envelope"}
            if envelope.installation_id != identity.installation_id:
                return {"ok": False, "error": "installation_mismatch"}

            async def authorize() -> bool:
                if self.authorize_operation is not None:
                    return await self.authorize_operation(identity)
                return identity.allows("chat.write")

            outcome: MutationOutcome = await self.replay_cache.execute(
                envelope,
                current_generation=self.generation_provider(),
                current_epoch=epoch,
                authorize=authorize,
                effect=lambda item: self.mutation_handler(identity, item)
                if self.mutation_handler is not None
                else _empty_effect(item),
            )
            return {"ok": True, "result": outcome.model_dump(mode="json")}
        if frame_type == "logout":
            self.authorize_ws_frame(identity, epoch, "device.self.logout")
            if self.logout_handler is None:
                return {"ok": False, "error": "logout_unavailable"}
            result = await self.logout_handler(identity)
            await self.validator.invalidate(identity.installation_id)
            return {"ok": True, "result": result}
        return {"ok": False, "error": "unknown_frame"}

    def _attachment_limit(self) -> int:
        return default_limits()["max_attachment_bytes"]

    def _media_limit(self) -> int:
        return default_limits()["max_media_response_bytes"]


def _headers(items: list[tuple[str, str]]) -> Any:
    from websockets.datastructures import Headers

    return Headers(items)


async def _empty_effect(_envelope: MutationEnvelope) -> dict[str, Any]:
    return {}
