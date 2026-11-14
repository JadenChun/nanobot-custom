"""HTTP and handshake composition for the WebUI gateway listener."""

from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from websockets.asyncio.server import ServerConnection
from websockets.http11 import Request as WsRequest

from nanobot.webui.gateway_tokens import GatewayTokenStore
from nanobot.webui.http_utils import (
    is_trusted_proxy_authenticated_request,
    normalize_config_path,
    parse_request_path,
    query_first,
)
from nanobot.webui.ws_http import GatewayHTTPHandler, attached_mobile_identity

if TYPE_CHECKING:
    from nanobot.channels.websocket.runtime import WebSocketConfig
    from nanobot.remote.edge import MobileGatewayEdge


def is_websocket_upgrade(request: WsRequest) -> bool:
    """Return whether a request contains a complete WebSocket upgrade handshake."""
    upgrade = request.headers.get("Upgrade") or request.headers.get("upgrade")
    connection = request.headers.get("Connection") or request.headers.get("connection")
    return bool(
        upgrade
        and "websocket" in upgrade.lower()
        and connection
        and "upgrade" in connection.lower()
    )


class WebUIGatewayEndpoint:
    """Compose HTTP routing and WebSocket authentication on one listener."""

    def __init__(
        self,
        *,
        config: WebSocketConfig,
        http: GatewayHTTPHandler,
        tokens: GatewayTokenStore,
        mobile_edge: "MobileGatewayEdge | None" = None,
    ) -> None:
        self._config = config
        self._http = http
        self._tokens = tokens
        self._mobile_edge = mobile_edge
        http.mobile_revalidator = self.revalidate_mobile_connection
        self.webui_connections: set[ServerConnection] = set()
        # Every accepted WebUI-protocol socket, desktop and mobile. Kept apart
        # from ``webui_connections`` (a trust/local-authority set) so broadcasts
        # cannot widen authorization.
        self.protocol_connections: set[ServerConnection] = set()

    async def process_request(
        self,
        connection: ServerConnection,
        request: WsRequest,
        *,
        is_allowed: Callable[[str], bool],
    ) -> Any:
        """Route one listener request to a WS handshake or the HTTP application."""
        got, query = parse_request_path(request.path)
        if got == "/mobile/v1/ws" and is_websocket_upgrade(request):
            if self._mobile_edge is None:
                return connection.respond(404, "Not Found")
            response = await self._mobile_edge.authorize_websocket(connection, request)
            if response is None:
                self.protocol_connections.add(connection)
            return response
        if got == "/mobile/v1" or got.startswith("/mobile/v1/"):
            if self._mobile_edge is None:
                return connection.respond(404, "Not Found")
            return await self._mobile_edge.handle(connection, request)
        expected_ws = normalize_config_path(self._config.path)
        if got == expected_ws and is_websocket_upgrade(request):
            terminal_version = query_first(query, "terminal_protocol")
            if terminal_version is not None and (
                terminal_version != "1"
                or query_first(query, "terminal_instance") != self._tokens.instance_id
            ):
                return connection.respond(409, "Terminal instance or protocol changed")
            client_id = query_first(query, "client_id") or ""
            if len(client_id) > 128:
                client_id = client_id[:128]
            if not is_allowed(client_id):
                return connection.respond(403, "Forbidden")
            response = self.authorize_websocket_handshake(connection, query, request.headers)
            if response is None:
                self.protocol_connections.add(connection)
            return response
        return await self._http.dispatch(connection, request)

    def authorize_websocket_handshake(
        self,
        connection: ServerConnection,
        query: dict[str, list[str]],
        headers: Any = None,
    ) -> Any:
        """Authorize a WebSocket upgrade and remember trusted WebUI connections."""
        if is_trusted_proxy_authenticated_request(connection, headers or {}, self._config):
            self.webui_connections.add(connection)
            return None

        supplied = query_first(query, "token")
        static_token = self._config.token.strip()
        if static_token:
            if supplied and hmac.compare_digest(supplied, static_token):
                return None
            if supplied and self.consume_issued_token(connection, supplied):
                return None
            return connection.respond(401, "Unauthorized")

        if self._config.websocket_requires_token:
            if supplied and self.consume_issued_token(connection, supplied):
                return None
            return connection.respond(401, "Unauthorized")

        if supplied:
            self.consume_issued_token(connection, supplied)
        return None

    def consume_issued_token(self, connection: ServerConnection, token: str) -> bool:
        """Consume one issued token and record its WebUI audience when present."""
        audience = self._tokens.take_issued_token_audience(token)
        if audience == "webui":
            self.webui_connections.add(connection)
        return audience is not None

    def is_webui_connection(self, connection: ServerConnection) -> bool:
        return connection in self.webui_connections

    async def revalidate_mobile_connection(self, connection: ServerConnection) -> bool:
        """Revalidate current owner authority for one established mobile socket."""
        identity = attached_mobile_identity(connection)
        if identity is None:
            return True
        from nanobot.remote.edge import AccessIdentity

        if not isinstance(identity, AccessIdentity):
            return False
        token = getattr(connection, "mobile_access_token", None)
        if self._mobile_edge is None or not isinstance(token, str) or not token:
            return False
        try:
            request = getattr(connection, "request", None)
            if not isinstance(request, WsRequest):
                return False
            self._mobile_edge.authorize_target(request, identity.origin)
            fresh = await self._mobile_edge.revalidate_mobile_access(identity, token)
        except Exception:
            from dataclasses import replace

            setattr(connection, "mobile_identity", replace(identity, scopes=()))
            return False
        setattr(connection, "mobile_identity", fresh)
        return True

    def discard_connection(self, connection: ServerConnection) -> None:
        self.webui_connections.discard(connection)
        self.protocol_connections.discard(connection)

    def clear(self) -> None:
        self.webui_connections.clear()
        self.protocol_connections.clear()
