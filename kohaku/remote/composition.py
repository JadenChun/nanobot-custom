"""Gateway composition for the authenticated mobile edge (M11)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

from websockets.datastructures import Headers
from websockets.http11 import Request as WsRequest

from kohaku.config.paths import get_data_dir
from kohaku.config.schema import RemoteAccessConfig
from kohaku.remote.client import RemoteAccessClient
from kohaku.remote.edge import MediaResult, MobileGatewayEdge, OwnerAccessValidator
from kohaku.remote.protocol import MAX_MEDIA_RESPONSE_BYTES
from kohaku.webui.media_gateway import WebUIMediaGateway


def remote_access_root(
    config: RemoteAccessConfig,
    *,
    instance_root: Path | str | None = None,
) -> Path:
    """Return the owner state root for this instance.

    An explicit ``state_dir`` wins. Otherwise the root is
    ``<instance-root>/remote-access`` where the instance root is the directory
    of the explicit config path. Callers without an explicit instance keep the
    process-global data directory.
    """
    if config.state_dir:
        return Path(config.state_dir).expanduser()
    base = Path(instance_root).expanduser() if instance_root is not None else get_data_dir()
    return base / "remote-access"


def owner_socket_path(
    config: RemoteAccessConfig,
    *,
    instance_root: Path | str | None = None,
) -> Path:
    if config.socket_path:
        return Path(config.socket_path).expanduser()
    return remote_access_root(config, instance_root=instance_root) / "owner.sock"


def build_mobile_edge(
    config: RemoteAccessConfig,
    http: Any,
    media: WebUIMediaGateway,
    *,
    instance_root: Path | str | None = None,
) -> MobileGatewayEdge:
    client = RemoteAccessClient(
        owner_socket_path(config, instance_root=instance_root),
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
            max_response_bytes=MAX_MEDIA_RESPONSE_BYTES,
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

    async def attachment_handler(identity: Any, body: bytes) -> dict[str, Any]:
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
        records, rejection = media.store_inbound_records(
            [{"data_url": f"data:{media_type};base64,{data}", "name": name}],
            source="mobile",
        )
        if rejection is not None:
            return {"error": rejection}
        if not records:
            return {"error": "decode"}
        record = records[0]
        media.upload_registry.add(
            record,
            installation_id=str(identity.installation_id),
            session_key=None,
            source="mobile",
        )
        return {
            "id": record.id,
            "name": record.name,
            "mime": record.mime,
            "bytes": record.bytes,
        }

    async def logout_handler(identity: Any) -> dict[str, Any]:
        await asyncio.to_thread(client.revoke, identity.installation_id)
        return {"installation_id": identity.installation_id, "erased": True}

    async def model_settings_handler(_identity: Any) -> dict[str, Any]:
        return await asyncio.to_thread(http.mobile_model_settings_payload)

    async def workspaces_handler(identity: Any) -> dict[str, Any]:
        return await asyncio.to_thread(
            http.mobile_workspaces_payload,
            owner=bool(getattr(identity, "is_owner", False)),
        )

    async def thread_handler(
        identity: Any,
        key: str,
        query: dict[str, list[str]],
    ) -> tuple[int, dict[str, Any]]:
        return await asyncio.to_thread(
            http.mobile_thread_payload,
            key,
            query,
            owner=bool(getattr(identity, "is_owner", False)),
        )

    async def delegated_children_handler(
        _identity: object,
        key: str,
        query: dict[str, list[str]],
    ) -> tuple[int, dict[str, Any]]:
        return await asyncio.to_thread(
            http.mobile_delegated_children_payload,
            key,
            query,
        )

    async def api_handler(identity: Any, path: str) -> tuple[int, dict[str, Any]]:
        return await http.mobile_api_payload(
            path,
            owner=bool(getattr(identity, "is_owner", False)),
            installation_id=str(getattr(identity, "installation_id", "") or ""),
        )

    async def feature_handler(identity: Any, path: str) -> tuple[int, dict[str, Any]]:
        return await http.mobile_feature_payload(path, identity)

    async def standard_handler(
        identity: Any,
        method: str,
        path: str,
    ) -> tuple[int, dict[str, Any]]:
        return await http.mobile_standard_payload(identity, method, path)

    return MobileGatewayEdge(
        validator,
        generation_provider=lambda: 1,
        read_handler=read_handler,
        media_handler=media_handler,
        attachment_handler=attachment_handler,
        logout_handler=logout_handler,
        model_settings_handler=model_settings_handler,
        workspaces_handler=workspaces_handler,
        thread_handler=thread_handler,
        delegated_children_handler=delegated_children_handler,
        api_handler=api_handler,
        feature_handler=feature_handler,
        standard_handler=standard_handler,
    )
