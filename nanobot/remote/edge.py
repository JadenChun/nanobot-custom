"""Authenticated `/mobile/v1` gateway edge over the independent owner IPC (M11).

The edge is intentionally a separate front door: legacy WebUI, desktop, settings
and native routes are untouched. Every mobile request is validated against the
owner's current access grant and scope; unknown routes deny.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Awaitable, Callable, cast
from urllib.parse import parse_qs, unquote, urlsplit

from websockets.http11 import Request as WsRequest
from websockets.http11 import Response

from nanobot.remote.client import OwnerRequestError, OwnerUnavailableError, RemoteAccessClient
from nanobot.remote.device_proof import decode_proof_base64url
from nanobot.remote.operations import OperationRegistryError, load_registry
from nanobot.remote.protocol import (
    BRIDGE_VERSION,
    MOBILE_PROTOCOL_VERSION,
    OWNER_SCOPE,
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
MOBILE_API_TIMEOUT_S = 30.0
# Safety ceiling only: the thread projection is paged and bounded, so a normal
# session stays far below this. It exists to stop a pathological payload from
# exhausting the phone, not to decide which sessions are viewable.
MOBILE_API_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_THREAD_RE = re.compile(r"^/mobile/v1/sessions/([^/]+)/thread$")
_MOBILE_API_ROUTES: tuple[tuple[re.Pattern[str], frozenset[str]], ...] = (
    (re.compile(r"^/api/commands$"), frozenset()),
    (re.compile(r"^/api/sessions/[^/]+/file-preview$"), frozenset({"path", "probe"})),
    (re.compile(r"^/api/sessions/[^/]+/generated-assets$"), frozenset({"limit", "cursor"})),
    (re.compile(r"^/api/generated-assets$"), frozenset({"limit", "cursor"})),
    (re.compile(r"^/api/sessions/[^/]+/automations$"), frozenset()),
    (re.compile(r"^/api/webui/sidebar-state$"), frozenset()),
)
_LOCAL_ROUTES = {
    "/mobile/v1/handshake": "chat.read",
    "/mobile/v1/read": "chat.read",
    "/mobile/v1/status": "chat.read",
    "/mobile/v1/model-settings": "chat.read",
    "/mobile/v1/workspaces": "chat.read",
}
_PAIRING_ROUTES = {
    "/mobile/v1/pair/collect",
    "/mobile/v1/pair/poll",
    "/mobile/v1/pair/code",
    "/mobile/v1/pair/prove",
    "/mobile/v1/pair/recover",
    "/mobile/v1/pair/recover/complete",
}
_REFRESH_ROUTE = "/mobile/v1/refresh"
_LEGACY_MUTATION_ACTIONS = frozenset(
    {
        "session.delete",
        "sidebar.update",
        "recovery.continue",
        "recovery.dismiss",
    }
)
_PAIRING_STATUS = {
    "pairing_rate_limited": 429,
    "pairing_not_found": 404,
    "pairing_expired": 410,
    "pairing_consumed": 410,
    "pairing_busy": 409,
    "pairing_code_invalid": 403,
    "pairing_denied": 403,
    "remote_access_disabled": 503,
    "pairing_rejected": 403,
    "owner_required": 403,
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
    origin: str = ""

    def allows(self, scope: str) -> bool:
        return scope in self.scopes

    @property
    def is_owner(self) -> bool:
        return OWNER_SCOPE in self.scopes


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
        return await self._validate(access_token, use_cache=True)

    async def validate_uncached(self, access_token: str) -> AccessIdentity:
        """Recheck current owner state without a warm admission-cache answer."""
        return await self._validate(access_token, use_cache=False)

    async def _validate(
        self,
        access_token: str,
        *,
        use_cache: bool,
    ) -> AccessIdentity:
        key = hashlib.sha256(access_token.encode("utf-8")).hexdigest()
        now = self.clock()
        if use_cache:
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
            origin=str(result.get("origin", "")),
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


def _allowed_mobile_api_path(raw_path: str) -> str | None:
    """Return the exact allowlisted chat read path, including its query."""
    if not raw_path or len(raw_path) > 4096:
        return None
    parsed = urlsplit(raw_path)
    if parsed.scheme or parsed.netloc or parsed.fragment:
        return None
    query = parse_qs(parsed.query, keep_blank_values=True)
    for pattern, allowed_query in _MOBILE_API_ROUTES:
        if pattern.fullmatch(parsed.path):
            if set(query) - allowed_query:
                return None
            return raw_path
    return None


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
        model_settings_handler: Callable[[AccessIdentity], Awaitable[dict[str, Any]]] | None = None,
        workspaces_handler: Callable[[AccessIdentity], Awaitable[dict[str, Any]]] | None = None,
        thread_handler: Callable[
            [AccessIdentity, str, dict[str, list[str]]], Awaitable[tuple[int, dict[str, Any]]]
        ] | None = None,
        api_handler: Callable[[AccessIdentity, str], Awaitable[tuple[int, dict[str, Any]]]] | None = None,
        feature_handler: Callable[[AccessIdentity, str], Awaitable[tuple[int, dict[str, Any]]]] | None = None,
        standard_handler: Callable[
            [AccessIdentity, str, str], Awaitable[tuple[int, dict[str, Any]]]
        ] | None = None,
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
        self.model_settings_handler = model_settings_handler
        self.workspaces_handler = workspaces_handler
        self.thread_handler = thread_handler
        self.api_handler = api_handler
        self.feature_handler = feature_handler
        self.standard_handler = standard_handler

    async def authorize_websocket(self, connection: Any, request: WsRequest) -> Response | None:
        token = bearer_token(request.headers)
        if not token:
            return http_json_response({"error": "missing_access"}, status=401)
        try:
            identity = await self.validator.validate_uncached(token)
            self._authorize_target(request, identity.origin)
            if not identity.allows("chat.read"):
                raise EdgeAuthError(403, "scope_denied")
        except EdgeAuthError as exc:
            return http_json_response({"error": exc.code}, status=exc.status)
        setattr(connection, "mobile_identity", identity)
        setattr(connection, "mobile_access_token", token)
        return None

    async def revalidate_mobile_access(
        self,
        identity: AccessIdentity,
        access_token: str,
    ) -> AccessIdentity:
        """Revalidate current expiry/revocation for an established socket."""
        if time.time() >= identity.expires_at:
            raise EdgeAuthError(401, "access_expired")
        if not access_token:
            raise EdgeAuthError(401, "missing_access")
        fresh = await self.validator.validate_uncached(access_token)
        if fresh.installation_id != identity.installation_id:
            raise EdgeAuthError(403, "installation_mismatch")
        if not fresh.origin or fresh.origin != identity.origin:
            raise EdgeAuthError(403, "origin_mismatch")
        if not fresh.allows("chat.read"):
            raise EdgeAuthError(403, "scope_denied")
        return fresh

    @staticmethod
    def _authorize_target(request: WsRequest, origin: str) -> None:
        """The owner-bound HTTPS target proves native requests without Origin."""
        try:
            expected = urlsplit(origin)
            host = request.headers.get("Host", "")
            supplied = request.headers.get("Origin")
            if (
                expected.scheme != "https" or not expected.netloc
                or expected.path or expected.query or expected.fragment
                or not host or host.lower() != expected.netloc.lower()
                or (supplied is not None and supplied != origin)
            ):
                raise EdgeAuthError(403, "origin_mismatch")
        except (ValueError, KeyError):
            raise EdgeAuthError(403, "origin_mismatch") from None

    async def _authorized_delivery(
        self,
        token: str,
        identity: AccessIdentity,
        response: Response,
        *,
        scope: str | None = None,
    ) -> Response:
        """Recheck current owner authorization before releasing private bytes."""
        try:
            fresh = await self.validator.validate_uncached(token)
        except EdgeAuthError as exc:
            return http_json_response({"error": exc.code}, status=exc.status)
        if fresh.installation_id != identity.installation_id:
            return http_json_response({"error": "installation_mismatch"}, status=403)
        if not fresh.origin or fresh.origin != identity.origin:
            return http_json_response({"error": "origin_mismatch"}, status=403)
        if scope is not None and not fresh.allows(scope):
            return http_json_response({"error": "scope_denied"}, status=403)
        return response

    async def handle(self, _connection: Any, request: WsRequest) -> Response | None:
        path, query = parse_request_path(request.path)
        if path != "/mobile/v1" and not path.startswith("/mobile/v1/"):
            return None
        if path in _PAIRING_ROUTES:
            try:
                return await self._handle_pairing(request, path)
            except EdgeAuthError as exc:
                return http_json_response({"error": exc.code}, status=exc.status)
        if path == _REFRESH_ROUTE:
            return await self._handle_refresh(request)
        token = bearer_token(request.headers)
        if not token:
            return http_json_response({"error": "missing_access"}, status=401)
        try:
            identity = await self.validator.validate_uncached(token)
            self._authorize_target(request, identity.origin)
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
                try:
                    payload = await asyncio.wait_for(
                        self.read_handler(identity, query),
                        timeout=self._timeout("sessions.list"),
                    )
                except TimeoutError:
                    return http_json_response({"error": "read_timeout"}, status=504)
                if not self._fits(payload, "sessions.list"):
                    return http_json_response({"error": "response_too_large"}, status=413)
                return await self._authorized_delivery(
                    token, identity, http_json_response(payload), scope=scope,
                )
            if path == "/mobile/v1/model-settings":
                if self.model_settings_handler is None:
                    return http_json_response({"error": "model_settings_unavailable"}, status=503)
                return await self._authorized_delivery(
                    token,
                    identity,
                    http_json_response(await self.model_settings_handler(identity)),
                    scope=scope,
                )
            if path == "/mobile/v1/workspaces":
                if self.workspaces_handler is None:
                    return http_json_response({"error": "workspaces_unavailable"}, status=503)
                return await self._authorized_delivery(
                    token,
                    identity,
                    http_json_response(await self.workspaces_handler(identity)),
                    scope=scope,
                )

        if path == "/mobile/v1/dispatch":
            if not identity.is_owner:
                return http_json_response({"error": "owner_required"}, status=403)
            if self.standard_handler is None:
                return http_json_response({"error": "dispatch_unavailable"}, status=503)
            raw_path = (query.get("path") or [""])[0]
            spec = load_registry().match_read(raw_path)
            if spec is None or not spec.enabled or not identity.allows(spec.scope):
                return http_json_response({"error": "scope_denied"}, status=403)
            try:
                status, payload = await asyncio.wait_for(
                    self.standard_handler(identity, "GET", raw_path),
                    timeout=self._timeout("sessions.thread"),
                )
            except TimeoutError:
                return http_json_response({"error": "dispatch_timeout"}, status=504)
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if len(encoded) > MOBILE_API_MAX_RESPONSE_BYTES:
                return http_json_response({"error": "response_too_large"}, status=413)
            return await self._authorized_delivery(
                token, identity, http_json_response(payload, status=status),
                scope="chat.read",
            )

        if path == "/mobile/v1/feature":
            if not identity.is_owner:
                return http_json_response({"error": "owner_required"}, status=403)
            if self.feature_handler is None:
                return http_json_response({"error": "feature_unavailable"}, status=503)
            raw_path = (query.get("path") or [""])[0]
            try:
                spec = load_registry().match_read(raw_path)
            except OperationRegistryError:
                spec = None
            if spec is None or not spec.enabled:
                return http_json_response({"error": "path_not_allowed"}, status=400)
            try:
                status, payload = await asyncio.wait_for(
                    self.feature_handler(identity, raw_path),
                    timeout=spec.timeout_s or 30,
                )
            except TimeoutError:
                return http_json_response({"error": "feature_timeout"}, status=504)
            if spec.max_response_bytes is not None:
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if len(encoded) > spec.max_response_bytes:
                    return http_json_response({"error": "response_too_large"}, status=413)
            return await self._authorized_delivery(
                token, identity, http_json_response(payload, status=status),
                scope=OWNER_SCOPE,
            )

        if path == "/mobile/v1/api":
            if not identity.allows("chat.read"):
                return http_json_response({"error": "scope_denied"}, status=403)
            if self.api_handler is None:
                return http_json_response({"error": "api_unavailable"}, status=503)
            raw_path = (query.get("path") or [""])[0]
            allowed_path = _allowed_mobile_api_path(raw_path)
            if allowed_path is None:
                return http_json_response({"error": "path_not_allowed"}, status=400)
            try:
                status, payload = await asyncio.wait_for(
                    self.api_handler(identity, allowed_path),
                    timeout=MOBILE_API_TIMEOUT_S,
                )
            except TimeoutError:
                return http_json_response({"error": "api_timeout"}, status=504)
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if len(encoded) > MOBILE_API_MAX_RESPONSE_BYTES:
                return http_json_response({"error": "response_too_large"}, status=413)
            return await self._authorized_delivery(
                token, identity, http_json_response(payload, status=status),
                scope="chat.read",
            )

        thread_match = _THREAD_RE.match(path)
        if thread_match is not None:
            if not identity.allows("chat.read"):
                return http_json_response({"error": "scope_denied"}, status=403)
            if self.thread_handler is None:
                return http_json_response({"error": "thread_unavailable"}, status=503)
            key = unquote(thread_match.group(1))
            try:
                status, payload = await asyncio.wait_for(
                    self.thread_handler(identity, key, query),
                    timeout=self._timeout("sessions.thread"),
                )
            except TimeoutError:
                return http_json_response({"error": "thread_timeout"}, status=504)
            if not self._fits(payload, "sessions.thread"):
                return http_json_response({"error": "response_too_large"}, status=413)
            return await self._authorized_delivery(
                token, identity, http_json_response(payload, status=status),
                scope="chat.read",
            )

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
            return await self._authorized_delivery(
                token,
                identity,
                Response(result.status, reason, _headers(headers), result.body),
                scope="media.read",
            )
        return http_json_response({"error": "not_found"}, status=404)

    async def _handle_refresh(self, request: WsRequest) -> Response:
        headers = request.headers

        def header(name: str) -> str:
            value = headers.get(name)
            return value.strip() if isinstance(value, str) else ""

        installation_id = header("X-Nanobot-Installation")
        refresh_token = header("X-Nanobot-Refresh-Token")
        if not installation_id or not refresh_token:
            return http_json_response({"error": "bad_request"}, status=400)
        try:
            result = await asyncio.to_thread(
                self.validator.client.refresh_access,
                installation_id,
                refresh_token,
            )
        except OwnerUnavailableError:
            return http_json_response({"error": "owner_unavailable"}, status=503)
        except OwnerRequestError as exc:
            if exc.code == "upgrade_required":
                return http_json_response({"error": "upgrade_required"}, status=426)
            code = exc.code if exc.code in {"device_revoked", "invalid_refresh"} else "invalid_refresh"
            return http_json_response({"error": code}, status=401)
        return http_json_response(result)

    async def _handle_pairing(self, request: WsRequest, path: str) -> Response:
        headers = request.headers

        def header(name: str) -> str:
            value = headers.get(name)
            return value.strip() if isinstance(value, str) else ""

        installation_id = header("X-Nanobot-Installation")
        label = header("X-Nanobot-Label") or "mobile"
        platform = header("X-Nanobot-Platform") or None
        model = header("X-Nanobot-Model") or None
        device_public_key = header("X-Nanobot-Device-Key") or None
        if path.endswith("/recover"):
            installation_id = header("X-Nanobot-Installation")
            device_public_key = header("X-Nanobot-Device-Key")
            if not installation_id or not device_public_key:
                return http_json_response({"error": "bad_request"}, status=400)
            try:
                result = await asyncio.to_thread(
                    self.validator.client.begin_pairing_recovery,
                    installation_id=installation_id,
                    device_public_key=device_public_key,
                )
            except OwnerUnavailableError:
                return http_json_response({"error": "owner_unavailable"}, status=503)
            except OwnerRequestError as exc:
                status = _PAIRING_STATUS.get(exc.code, 400)
                return http_json_response({"error": exc.code}, status=status)
            return await self._pairing_delivery(request, result)
        if path.endswith("/recover/complete"):
            installation_id = header("X-Nanobot-Installation")
            signature = header("X-Nanobot-Device-Signature")
            encoded_claims = header("X-Nanobot-Proof-Claims")
            if not installation_id or not signature or not encoded_claims:
                return http_json_response({"error": "bad_request"}, status=400)
            try:
                decoded = decode_proof_base64url(encoded_claims)
                parsed = json.loads(decoded.decode("utf-8"))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                return http_json_response({"error": "bad_request"}, status=400)
            if not isinstance(parsed, dict):
                return http_json_response({"error": "bad_request"}, status=400)
            self._authorize_target(request, str(parsed.get("origin", "")))
            try:
                result = await asyncio.to_thread(
                    self.validator.client.complete_pairing_recovery,
                    installation_id=installation_id,
                    claims=cast(dict[str, object], parsed),
                    signature=signature,
                )
            except OwnerUnavailableError:
                return http_json_response({"error": "owner_unavailable"}, status=503)
            except OwnerRequestError as exc:
                status = _PAIRING_STATUS.get(exc.code, 400)
                return http_json_response({"error": exc.code}, status=status)
            return await self._pairing_delivery(request, result)
        if path.endswith("/prove"):
            pairing_id = header("X-Nanobot-Pairing-Id")
            signature = header("X-Nanobot-Device-Signature")
            encoded_claims = header("X-Nanobot-Proof-Claims")
            claims: dict[str, object] | None = None
            if encoded_claims:
                try:
                    decoded = decode_proof_base64url(encoded_claims)
                    parsed = json.loads(decoded.decode("utf-8"))
                except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                    return http_json_response({"error": "bad_request"}, status=400)
                if isinstance(parsed, dict):
                    claims = cast(dict[str, object], parsed)
            if claims is None:
                try:
                    raw_body = cast(bytes, getattr(request, "body", b""))
                    payload = json.loads(raw_body.decode("utf-8"))
                except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
                    return http_json_response({"error": "bad_request"}, status=400)
                if not isinstance(payload, dict):
                    return http_json_response({"error": "bad_request"}, status=400)
                body = cast(dict[str, Any], payload)
                pairing_id = body.get("pairing_id") if not pairing_id else pairing_id
                candidate = body.get("claims")
                claims = cast(dict[str, object], candidate) if isinstance(candidate, dict) else None
                signature = body.get("signature") if not signature else signature
            if (
                not isinstance(pairing_id, str)
                or not pairing_id
                or claims is None
                or not isinstance(signature, str)
                or not signature
            ):
                return http_json_response({"error": "bad_request"}, status=400)
            self._authorize_target(request, str(claims.get("origin", "")))
            try:
                result = await asyncio.to_thread(
                    self.validator.client.prove_pairing,
                    pairing_id=pairing_id,
                    claims=claims,
                    signature=signature,
                )
            except OwnerUnavailableError:
                return http_json_response({"error": "owner_unavailable"}, status=503)
            except OwnerRequestError as exc:
                status = _PAIRING_STATUS.get(exc.code, 400)
                return http_json_response({"error": exc.code}, status=status)
            return await self._pairing_delivery(request, result)
        if path.endswith("/code"):
            origin = "https://" + header("Host")
            self._authorize_target(request, origin)
            forwarded_origin = header("X-Nanobot-Origin")
            if forwarded_origin and forwarded_origin != origin:
                raise EdgeAuthError(403, "origin_mismatch")
            pairing_code = header("X-Nanobot-Pairing-Code") or header("X-Nanobot-Code")
            if not origin or not pairing_code or not installation_id:
                return http_json_response({"error": "bad_request"}, status=400)
            try:
                result = await asyncio.to_thread(
                    self.validator.client.collect_pairing_code,
                    origin=origin,
                    pairing_code=pairing_code,
                    installation_id=installation_id,
                    label=label,
                    platform=platform,
                    model=model,
                    device_public_key=device_public_key,
                )
            except OwnerUnavailableError:
                return http_json_response({"error": "owner_unavailable"}, status=503)
            except OwnerRequestError as exc:
                status = _PAIRING_STATUS.get(exc.code, 400)
                return http_json_response({"error": exc.code}, status=status)
            return await self._pairing_delivery(request, result)
        pairing_id = header("X-Nanobot-Pairing-Id")
        nonce = header("X-Nanobot-Pairing-Nonce")
        collector_secret = header("X-Nanobot-Pairing-Secret")
        if not pairing_id or not nonce or not collector_secret:
            return http_json_response({"error": "bad_request"}, status=400)
        try:
            if path.endswith("/collect"):
                if not installation_id:
                    return http_json_response({"error": "bad_request"}, status=400)
                result = await asyncio.to_thread(
                    self.validator.client.collect_pairing,
                    pairing_id=pairing_id,
                    nonce=nonce,
                    collector_secret=collector_secret,
                    installation_id=installation_id,
                    label=label,
                    platform=platform,
                    model=model,
                    device_public_key=device_public_key,
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
        return await self._pairing_delivery(request, result)

    async def _pairing_delivery(self, request: WsRequest, result: dict[str, Any]) -> Response:
        origin = result.get("origin")
        token = result.get("access_token")
        if isinstance(token, str) and token:
            fresh = await self.validator.validate_uncached(token)
            origin = fresh.origin
        if not isinstance(origin, str) or not origin:
            proof = result.get("proof_challenge")
            origin = proof.get("origin") if isinstance(proof, dict) else ""
        self._authorize_target(request, origin if isinstance(origin, str) else "")
        return http_json_response(result)

    def _handshake(self, identity: AccessIdentity) -> dict[str, Any]:
        capabilities = {
            "chat": True,
            "media": "media.read" in identity.scopes,
            "attachments": "attachment.write" in identity.scopes,
            "lifecycle": False,
            "owner": identity.is_owner,
            "request_settings_version": 1,
            "standard_dispatch": True,
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
            try:
                spec = load_registry().mutations.get(envelope.action)
            except OperationRegistryError:
                spec = None
            if spec is None or not spec.enabled:
                return {"ok": False, "error": "operation_not_allowed"}
            if not identity.is_owner and envelope.action not in _LEGACY_MUTATION_ACTIONS:
                return {"ok": False, "error": "owner_required"}
            try:
                self.authorize_ws_frame(identity, epoch, "chat.write")
            except EdgeAuthError as exc:
                return {"ok": False, "error": exc.code}

            async def authorize() -> bool:
                if self.authorize_operation is not None:
                    return await self.authorize_operation(identity)
                return identity.allows("chat.write")

            async def apply_effect(item: MutationEnvelope) -> dict[str, Any]:
                if self.mutation_handler is None:
                    return await _empty_effect(item)
                return await asyncio.wait_for(
                    self.mutation_handler(identity, item),
                    timeout=float(spec.timeout_s or 30),
                )

            try:
                outcome: MutationOutcome = await self.replay_cache.execute(
                    envelope,
                    current_generation=self.generation_provider(),
                    current_epoch=epoch,
                    authorize=authorize,
                    effect=apply_effect,
                )
            except TimeoutError:
                return {"ok": False, "error": "timeout"}
            return {"ok": True, "result": outcome.model_dump(mode="json")}
        if frame_type == "logout":
            self.authorize_ws_frame(identity, epoch, "device.self.logout")
            if self.logout_handler is None:
                return {"ok": False, "error": "logout_unavailable"}
            result = await self.logout_handler(identity)
            await self.validator.invalidate(identity.installation_id)
            return {"ok": True, "result": result}
        return {"ok": False, "error": "unknown_frame"}

    @staticmethod
    def _timeout(operation: str) -> float:
        spec = load_registry().by_id(operation)
        return float(spec.timeout_s or 30) if spec else 30.0

    @staticmethod
    def _fits(payload: Any, operation: str) -> bool:
        spec = load_registry().by_id(operation)
        if spec is None or spec.max_response_bytes is None:
            return True
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return len(encoded) <= spec.max_response_bytes

    def _attachment_limit(self) -> int:
        return default_limits()["max_attachment_bytes"]

    def _media_limit(self) -> int:
        return default_limits()["max_media_response_bytes"]


def _headers(items: list[tuple[str, str]]) -> Any:
    from websockets.datastructures import Headers

    return Headers(items)


async def _empty_effect(_envelope: MutationEnvelope) -> dict[str, Any]:
    return {}
