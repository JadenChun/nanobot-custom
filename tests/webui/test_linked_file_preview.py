"""Public API cases authored independently in linked_preview_cases (CP-01–CP-07)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from nanobot.bus.queue import MessageBus
from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.config.loader import get_config_path, set_config_path
from nanobot.config.schema import RemoteAccessConfig
from nanobot.remote.owner import RemoteAccessOwner
from nanobot.remote.protocol import OWNER_PAIRING_SCOPES
from nanobot.session.manager import SessionManager
from nanobot.webui.gateway_services import GatewayServices, build_gateway_services
from nanobot.webui.transcript import append_transcript_object

ORIGIN = "https://127.0.0.1:29999"
ORIGIN_HOST = "127.0.0.1:29999"


@dataclass
class PreviewClient:
    root: Path
    project: Path
    instance: Path
    services: GatewayServices
    desktop_token: str
    mobile_token: str
    key: str = "websocket:linked-preview"

    def share(self, text: str, paths: list[Path] = (), *, key: str | None = None) -> None:
        key = key or self.key
        turn = str(len(text)) + ":" + text[:16]
        append_transcript_object(key, {"event": "user", "text": "Show the file", "turn_id": turn})
        if paths:
            append_transcript_object(key, {
                "event": "message", "kind": "progress", "turn_id": turn,
                "tool_events": [{
                    "phase": "end", "name": "exec", "call_id": turn,
                    "result": "\n".join(str(path) for path in paths), "error": None,
                }],
            })
        append_transcript_object(key, {"event": "stream_end", "text": text, "turn_id": turn})
        append_transcript_object(key, {"event": "turn_end", "turn_id": turn})

    async def request(self, path: str, *, probe: bool = False, mobile: bool = False):
        query = {"path": path, **({"probe": "1"} if probe else {})}
        url = f"/api/sessions/{quote(self.key, safe='')}/file-preview?{urlencode(query)}"
        token = self.desktop_token
        if mobile:
            url = "/mobile/v1/api?" + urlencode({"path": url})
            token = self.mobile_token
        return await self.services.endpoint.process_request(
            None,
            Request(
                url,
                Headers({
                    "Authorization": f"Bearer {token}",
                    "Host": ORIGIN_HOST,
                    "Origin": ORIGIN,
                }),
            ),
            is_allowed=lambda _client: True,
        )


@pytest.fixture
def preview_client(tmp_path: Path):
    previous_config = get_config_path()
    project = tmp_path / "chat project"
    instance = tmp_path / "instance"
    project.mkdir()
    instance.mkdir()
    config_path = instance / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    set_config_path(config_path)
    owner = RemoteAccessOwner(instance / "remote-access")
    owner.start()
    owner.store.set_enabled(True)
    try:
        pair = owner.tokens.issue_device(
            "preview-owner-device",
            OWNER_PAIRING_SCOPES,
            device_fingerprint="a" * 64,
            device_generation=1,
            origin=ORIGIN,
        )
        services = build_gateway_services(
            config=WebSocketConfig(host="127.0.0.1", port=29999, path="/"),
            bus=MessageBus(),
            session_manager=SessionManager(project, sessions_root=instance / "sessions"),
            static_dist_path=None,
            workspace_path=project,
            default_restrict_to_workspace=True,
            runtime_model_name=None,
            runtime_surface="mobile",
            runtime_capabilities_overrides=None,
            remote_access=RemoteAccessConfig(
                enabled=True, state_dir=str(owner.store.root), socket_path=str(owner.socket_path),
            ),
        )
        yield PreviewClient(
            tmp_path, project, instance, services,
            services.tokens.issue_api_token(60), pair.access_token,
        )
    finally:
        owner.stop()
        set_config_path(previous_config)


def assert_preview(response, target: Path, project: Path, content: str) -> dict:
    assert response.status_code == 200, response.body
    payload = json.loads(response.body)
    assert {"path", "display_path", "project_path", "language", "content", "size", "truncated"} <= payload.keys()
    assert payload["path"] == str(target)
    assert payload["project_path"] == str(project)
    assert payload["content"] == content
    return payload


@pytest.mark.parametrize("relative", ["../external area/report v1.txt", "external area/report v1.txt"])
async def test_cp01_relative_shared_file_opens_on_desktop_and_mobile(preview_client, relative):
    client = preview_client
    target = client.root / "external area" / "report v1.txt"
    target.parent.mkdir()
    target.write_text("outside-marker\n", encoding="utf-8")
    client.share(f"Here is `{relative}`", [target])
    before = client.services.workspaces.scope_for_session_key(client.key).metadata()

    desktop = assert_preview(await client.request(relative), target, client.project, "outside-marker\n")
    mobile = assert_preview(await client.request(relative, mobile=True), target, client.project, "outside-marker\n")

    assert mobile == desktop
    assert client.services.workspaces.scope_for_session_key(client.key).metadata() == before


async def test_cp02_absolute_markdown_link_needs_no_tool_output(preview_client):
    client = preview_client
    target = client.root / "external-absolute.txt"
    target.write_text("absolute-marker\n", encoding="utf-8")
    client.share(f"[external-absolute.txt](<{target}>)")
    before = client.services.workspaces.scope_for_session_key(client.key).metadata()

    desktop = assert_preview(await client.request(str(target)), target, client.project, "absolute-marker\n")
    mobile = assert_preview(await client.request(str(target), mobile=True), target, client.project, "absolute-marker\n")

    assert mobile == desktop
    assert client.services.workspaces.scope_for_session_key(client.key).metadata() == before


async def test_cp03_ambiguous_relative_reference_returns_no_file_content(preview_client):
    client = preview_client
    paths = [client.root / side / "draft.txt" for side in ("left", "right")]
    for path, marker in zip(paths, ("left-marker\n", "right-marker\n"), strict=True):
        path.parent.mkdir()
        path.write_text(marker, encoding="utf-8")
    client.share("Open `draft.txt`", paths)

    response = await client.request("draft.txt")

    assert response.status_code == 409
    assert b"left-marker" not in response.body
    assert b"right-marker" not in response.body


@pytest.mark.parametrize("source", ["user", "other-chat", "unmentioned"])
async def test_cp04_external_preview_requires_a_reference_in_this_chat(preview_client, source):
    client = preview_client
    target = client.root / f"external-{source}.txt"
    target.write_text("scope-marker\n", encoding="utf-8")
    if source == "user":
        append_transcript_object(client.key, {"event": "user", "text": f"Open `{target}`"})
    elif source == "other-chat":
        client.share(f"Open `{target}`", [target], key="websocket:other-chat")

    response = await client.request(str(target))

    assert response.status_code == 403
    assert b"scope-marker" not in response.body


async def test_cp05_private_instance_file_stays_denied(preview_client):
    client = preview_client
    target = client.instance / "credentials" / "private.txt"
    target.parent.mkdir()
    target.write_text("private-marker\n", encoding="utf-8")
    client.share(f"Open `{target}`", [target])

    response = await client.request(str(target))

    assert response.status_code == 403
    assert b"private-marker" not in response.body


@pytest.mark.parametrize("kind,available", [("text", True), ("missing", False), ("external", False), ("binary", False)])
async def test_cp06_probe_returns_only_availability(preview_client, kind, available):
    client = preview_client
    target = (client.root if kind == "external" else client.project) / f"{kind}.txt"
    if kind != "missing":
        target.write_bytes(b"\0binary" if kind == "binary" else b"marker\n")

    response = await client.request(str(target), probe=True)

    assert response.status_code == 200
    assert json.loads(response.body) == {"available": available}


@pytest.mark.parametrize("kind,status", [("text", 200), ("missing", 404), ("binary", 415)])
async def test_cp07_existing_project_preview_and_errors(preview_client, kind, status):
    client = preview_client
    target = client.project / ("inside.txt" if kind == "text" else f"{kind}.txt")
    if kind != "missing":
        target.write_bytes(b"\0binary" if kind == "binary" else b"project-marker\n")

    response = await client.request(str(target))

    assert response.status_code == status
    if kind == "text":
        assert_preview(response, target, client.project, "project-marker\n")
