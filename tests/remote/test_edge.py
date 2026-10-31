from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request as WsRequest
from websockets.http11 import Response

from nanobot.remote.client import RemoteAccessClient
from nanobot.remote.edge import (
    AccessIdentity,
    EdgeAuthError,
    MediaResult,
    MobileGatewayEdge,
    OwnerAccessValidator,
)
from nanobot.remote.owner import RemoteAccessOwner


def token_headers(token: str) -> Headers:
    return Headers([("Authorization", f"Bearer {token}")])


def request(path: str, token: str | None = None, extra: list[tuple[str, str]] | None = None) -> WsRequest:
    headers = Headers(extra or [])
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return WsRequest(path, headers)


def body(response: Response) -> dict[str, Any]:
    value = json.loads(response.body)
    assert isinstance(value, dict)
    return value


@pytest.fixture
def edge_setup(tmp_path: Path):
    owner = RemoteAccessOwner(tmp_path / "remote-access")
    owner.start()
    owner.store.set_enabled(True)
    pair = owner.tokens.issue(
        "install-0001",
        ("chat.read", "chat.write", "media.read", "attachment.write", "device.self.logout"),
    )
    validator = OwnerAccessValidator(
        RemoteAccessClient(owner.socket_path, timeout=2.0),
        cache_ttl=0.0,
    )
    edge = MobileGatewayEdge(
        validator,
        generation_provider=lambda: int(owner.store.snapshot()["revision"]),
        socket_epoch=7,
        read_handler=lambda identity, query: _read(identity, query),
        media_handler=lambda identity, name, range_header: _media(identity, name, range_header),
        attachment_handler=lambda identity, data: _attachment(identity, data),
        logout_handler=lambda identity: _logout(owner, identity),
    )
    try:
        yield owner, edge, pair
    finally:
        owner.stop()


async def _read(identity: AccessIdentity, query: dict[str, list[str]]) -> dict[str, Any]:
    return {"items": [identity.installation_id], "cursor": query.get("cursor", [None])[0]}


async def _media(identity: AccessIdentity, name: str, range_header: str | None) -> MediaResult:
    assert identity.installation_id == "install-0001"
    return MediaResult(
        status=206,
        body=b"PNG!",
        media_type="image/png",
        headers={"Content-Range": "bytes 0-3/4", "X-Range": range_header or "none"},
    )


async def _attachment(identity: AccessIdentity, data: bytes) -> dict[str, Any]:
    return {"installation_id": identity.installation_id, "bytes": len(data)}


async def _logout(owner: RemoteAccessOwner, identity: AccessIdentity) -> dict[str, Any]:
    owner.tokens.revoke(identity.installation_id)
    return {"installation_id": identity.installation_id, "erased": True}


@pytest.mark.asyncio
async def test_edge_handshake_read_media_scopes_and_unknown_routes(edge_setup) -> None:
    _owner, edge, pair = edge_setup
    response = await edge.handle(None, request("/mobile/v1/handshake", pair.access_token))
    assert response is not None
    assert response.status_code == 200
    payload = body(response)
    assert payload["mobile_protocol"] == 1
    assert payload["bridge_version"] == 1
    assert payload["principal"] == "mobile:install-0001"
    assert payload["socket_epoch"] == 7
    assert payload["capabilities"]["chat"] is True
    assert payload["limits"]["max_attachment_bytes"] == 256 * 1024

    read = await edge.handle(None, request("/mobile/v1/read?cursor=next", pair.access_token))
    assert read is not None and read.status_code == 200
    assert body(read)["cursor"] == "next"
    assert body(read)["items"] == ["install-0001"]

    media = await edge.handle(
        None,
        request("/mobile/v1/media/sample.png", pair.access_token, [("Range", "bytes=0-3")]),
    )
    assert media is not None and media.status_code == 206
    assert media.body == b"PNG!"
    assert media.headers["Content-Range"] == "bytes 0-3/4"
    assert media.headers["X-Range"] == "bytes=0-3"

    assert (await edge.handle(None, request("/mobile/v1/read"))).status_code == 401
    assert (await edge.handle(None, request("/mobile/v1/read", "wrong"))).status_code == 401
    assert (await edge.handle(None, request("/mobile/v1/nope", pair.access_token))).status_code == 404
    assert (await edge.handle(None, request("/api/bootstrap", pair.access_token))) is None


@pytest.mark.asyncio
async def test_edge_scope_denial_and_ws_frame_epoch(edge_setup) -> None:
    owner, edge, _pair = edge_setup
    limited = owner.tokens.issue("install-0002", ("chat.read",))
    response = await edge.handle(None, request("/mobile/v1/handshake", limited.access_token))
    assert response is not None and response.status_code == 200
    media = await edge.handle(None, request("/mobile/v1/media/sample.png", limited.access_token))
    assert media is not None and media.status_code == 403

    identity = AccessIdentity("install-0001", ("chat.read", "attachment.write"), 0, 2)
    with pytest.raises(EdgeAuthError) as mismatch:
        edge.authorize_ws_frame(identity, 8, "chat.read")
    assert mismatch.value.status == 409
    with pytest.raises(EdgeAuthError) as denied:
        edge.authorize_ws_frame(identity, 7, "media.read")
    assert denied.value.status == 403

    attachment = await edge.handle_ws_frame(
        identity,
        7,
        {"type": "attachment", "data_base64": base64.b64encode(b"hello").decode()},
    )
    assert attachment["ok"] is True
    assert attachment["result"]["bytes"] == 5
    unknown = await edge.handle_ws_frame(identity, 7, {"type": "unknown"})
    assert unknown["ok"] is False


@pytest.mark.asyncio
async def test_edge_revocation_owner_loss_and_logout(edge_setup) -> None:
    owner, edge, pair = edge_setup
    assert (await edge.handle(None, request("/mobile/v1/handshake", pair.access_token))).status_code == 200
    owner.tokens.revoke("install-0001")
    await edge.validator.invalidate("install-0001")
    revoked = await edge.handle(None, request("/mobile/v1/handshake", pair.access_token))
    assert revoked is not None and revoked.status_code == 401

    second = owner.tokens.issue("install-0003", ("chat.read", "device.self.logout"))
    logout = await edge.handle(None, request("/mobile/v1/logout", second.access_token))
    assert logout is not None and logout.status_code == 404
    frame = await edge.handle_ws_frame(
        AccessIdentity("install-0003", ("device.self.logout",), 0, 3),
        7,
        {"type": "logout"},
    )
    assert frame["ok"] is True
    after = await edge.handle(None, request("/mobile/v1/handshake", second.access_token))
    assert after is not None and after.status_code == 401

    owner.stop()
    unavailable = await edge.handle(None, request("/mobile/v1/handshake", pair.access_token))
    assert unavailable is not None and unavailable.status_code == 503
