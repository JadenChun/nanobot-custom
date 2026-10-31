"""Gateway composition for the authenticated mobile edge (M11)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

from websockets.datastructures import Headers
from websockets.http11 import Request as WsRequest

from nanobot.config.paths import get_data_dir
from nanobot.config.schema import RemoteAccessConfig
from nanobot.remote.client import RemoteAccessClient
from nanobot.remote.edge import MediaResult, MobileGatewayEdge, OwnerAccessValidator
from nanobot.webui.media_gateway import WebUIMediaGateway


def owner_socket_path(config: RemoteAccessConfig) -> Path:
    if config.socket_path:
        return Path(config.socket_path).expanduser()
    if config.state_dir:
        return Path(config.state_dir).expanduser() / "owner.sock"
    return get_data_dir() / "remote-access" / "owner.sock"


def build_mobile_edge(
    config: RemoteAccessConfig,
    http: Any,
    media: WebUIMediaGateway,
) -> MobileGatewayEdge:
    client = RemoteAccessClient(
        owner_socket_path(config),
        timeout=config.owner_timeout_seconds,
    )
    validator = OwnerAccessValidator(client)

    async def read_handler(_identity: Any, _query: dict[str, list[str]]) -> dict[str, Any]:
        payload = http.mobile_read_payload()
        return cast(dict[str, Any], payload) if isinstance(payload, dict) else {"sessions": []}

    async def media_handler(identity: Any, sig: str, payload: str, range_header: str | None) -> MediaResult:
        del identity
        headers = Headers([("Range", range_header)] if range_header else [])
        response = media.serve_signed_media(
            sig,
            payload,
            request=WsRequest(f"/api/media/{sig}/{payload}", headers),
        )
        content_type = ""
        for key, value in response.headers.raw_items():
            if key.lower() == "content-type":
                content_type = value
                break
        return MediaResult(
            status=response.status_code,
            body=bytes(response.body),
            media_type=content_type or "application/octet-stream",
            headers={
                key: value
                for key, value in response.headers.raw_items()
                if key.lower() in {"accept-ranges", "content-range", "etag", "cache-control"}
            },
        )

    async def attachment_handler(_identity: Any, body: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("attachment body is not JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("attachment body must be an object")
        payload_dict = cast(dict[str, Any], payload)
        data = payload_dict.get("data_base64")
        name = payload_dict.get("name", "attachment")
        media_type = payload_dict.get("media_type", "application/octet-stream")
        if not isinstance(data, str) or not isinstance(name, str) or not isinstance(media_type, str):
            raise ValueError("attachment fields are invalid")
        paths, rejection = media.store_inbound_attachments(
            [{"data_url": f"data:{media_type};base64,{data}", "name": name}]
        )
        if rejection is not None:
            return {"error": rejection}
        return {"paths": paths}

    async def logout_handler(identity: Any) -> dict[str, Any]:
        await asyncio.to_thread(client.revoke, identity.installation_id)
        return {"installation_id": identity.installation_id, "erased": True}

    return MobileGatewayEdge(
        validator,
        generation_provider=lambda: 1,
        read_handler=read_handler,
        media_handler=media_handler,
        attachment_handler=attachment_handler,
        logout_handler=logout_handler,
    )
