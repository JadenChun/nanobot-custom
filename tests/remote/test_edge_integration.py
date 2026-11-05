from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request as WsRequest

from nanobot.bus.queue import MessageBus
from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.config.schema import RemoteAccessConfig
from nanobot.remote.edge import AccessIdentity
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.webui.gateway_services import build_gateway_services

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def headers(token: str, extra: list[tuple[str, str]] | None = None) -> Headers:
    values = Headers(extra or [])
    values["Authorization"] = f"Bearer {token}"
    return values


@pytest.mark.asyncio
async def test_real_gateway_edge_read_media_attachment_logout(tmp_path: Path) -> None:
    owner = RemoteAccessOwner(tmp_path / "remote-access")
    owner.start()
    owner.store.set_enabled(True)
    pair = owner.tokens.issue(
        "install-edge-1",
        ("chat.read", "chat.write", "media.read", "attachment.write", "device.self.logout"),
    )
    services = build_gateway_services(
        config=WebSocketConfig(host="127.0.0.1", port=29999, path="/"),
        bus=MessageBus(),
        session_manager=None,
        static_dist_path=None,
        workspace_path=tmp_path,
        default_restrict_to_workspace=False,
        runtime_model_name=None,
        runtime_surface="mobile",
        runtime_capabilities_overrides=None,
        remote_access=RemoteAccessConfig(
            enabled=True,
            state_dir=str(owner.store.root),
            socket_path=str(owner.socket_path),
        ),
    )
    edge = services.endpoint._mobile_edge
    assert edge is not None
    try:
        handshake = await services.endpoint.process_request(
            None,
            WsRequest("/mobile/v1/handshake", headers(pair.access_token)),
            is_allowed=lambda _client: True,
        )
        assert handshake is not None and handshake.status_code == 200
        handshake_payload = json.loads(handshake.body)
        assert handshake_payload["capabilities"]["media"] is True
        epoch = int(handshake_payload["socket_epoch"])

        read = await services.endpoint.process_request(
            None,
            WsRequest("/mobile/v1/read", headers(pair.access_token)),
            is_allowed=lambda _client: True,
        )
        assert read is not None and read.status_code == 200
        assert json.loads(read.body) == {"sessions": []}

        media_file = tmp_path / "sample.png"
        media_file.write_bytes(PNG)
        signed = services.media.sign_or_stage_media_path(media_file)
        assert signed is not None
        url = signed["url"]
        assert url.startswith("/api/media/")
        parts = url.split("/")
        mobile_path = f"/mobile/v1/media/{parts[3]}/{parts[4]}"
        media = await services.endpoint.process_request(
            None,
            WsRequest(mobile_path, headers(pair.access_token, [("Range", "bytes=0-7")])),
            is_allowed=lambda _client: True,
        )
        assert media is not None
        assert media.status_code in {200, 206}
        assert media.body

        attachment = {
            "name": "pixel.png",
            "media_type": "image/png",
            "data_base64": base64.b64encode(PNG).decode(),
        }
        stored = await edge.handle_ws_frame(
            AccessIdentity("install-edge-1", ("attachment.write",), 0, 1),
            epoch,
            {"type": "attachment", "data_base64": base64.b64encode(json.dumps(attachment).encode()).decode()},
        )
        assert stored["ok"] is True
        assert stored["result"]["id"].startswith("att_")
        assert stored["result"]["name"] == "pixel.png"

        logged_out = await edge.handle_ws_frame(
            AccessIdentity("install-edge-1", ("device.self.logout",), 0, 1),
            epoch,
            {"type": "logout"},
        )
        assert logged_out["ok"] is True
        assert owner.store.get_device("install-edge-1").revoked
        denied = await services.endpoint.process_request(
            None,
            WsRequest("/mobile/v1/handshake", headers(pair.access_token)),
            is_allowed=lambda _client: True,
        )
        assert denied is not None and denied.status_code == 401
    finally:
        owner.stop()
