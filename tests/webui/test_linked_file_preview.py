"""Public API cases authored independently in linked_preview_cases (CP-01–CP-07)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from kohaku.bus.queue import MessageBus
from kohaku.channels.websocket.runtime import WebSocketConfig
from kohaku.config.loader import get_config_path, set_config_path
from kohaku.config.schema import RemoteAccessConfig
from kohaku.remote.owner import RemoteAccessOwner
from kohaku.remote.protocol import OWNER_PAIRING_SCOPES
from kohaku.session.manager import SessionManager
from kohaku.webui.gateway_services import GatewayServices, build_gateway_services
from kohaku.webui.transcript import append_transcript_object

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

    async def fetch_media(self, media_url: str):
        return await self.services.endpoint.process_request(
            None,
            Request(
                media_url,
                Headers({
                    "Authorization": f"Bearer {self.desktop_token}",
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


@pytest.mark.parametrize(
    "filename,raw_bytes,expected_kind,expected_mime",
    [
        ("INV-000009.pdf", b"%PDF-1.7\n\x00invoice-content", "pdf", "application/pdf"),
        ("demo.mp4", b"\x00\x00\x00\x18ftypmp42\x00video", "video", "video/mp4"),
        ("voice.mp3", b"ID3\x03\x00\x00\x00\x00audio", "audio", "audio/mpeg"),
    ],
)
async def test_media_and_pdf_preview_end_to_end_with_signed_media_route(
    preview_client: PreviewClient,
    filename: str,
    raw_bytes: bytes,
    expected_kind: str,
    expected_mime: str,
) -> None:
    target = preview_client.project / filename
    target.write_bytes(raw_bytes)

    probe = await preview_client.request(str(target), probe=True)
    assert probe.status_code == 200
    assert json.loads(probe.body) == {"available": True}

    resp = await preview_client.request(str(target))
    assert resp.status_code == 200, resp.body
    payload = json.loads(resp.body)
    assert payload["kind"] == expected_kind
    assert payload["mime"] == expected_mime
    assert payload["media_url"].startswith("/api/media/")

    media_resp = await preview_client.fetch_media(payload["media_url"])
    assert media_resp.status_code == 200
    assert bytes(media_resp.body) == raw_bytes
    assert media_resp.headers["content-type"].startswith(expected_mime)
    assert media_resp.headers.get("x-content-type-options") == "nosniff"


@pytest.mark.parametrize(
    "filename,raw_bytes,expected_kind,expected_mime",
    [
        ("INV-000009.pdf", b"%PDF-1.7\n\x00shared-pdf", "pdf", "application/pdf"),
        ("chart.png", b"\x89PNG\r\n\x1a\n\x00shared-png", "image", "image/png"),
    ],
)
async def test_external_assistant_shared_pdf_and_image_open_and_fetch_media(
    preview_client: PreviewClient,
    filename: str,
    raw_bytes: bytes,
    expected_kind: str,
    expected_mime: str,
) -> None:
    shared_dir = preview_client.root / "shared"
    shared_dir.mkdir(exist_ok=True)
    target = shared_dir / filename
    target.write_bytes(raw_bytes)
    preview_client.share(f"Saved `{target}`", [target])

    probe = await preview_client.request(str(target), probe=True)
    assert probe.status_code == 200
    assert json.loads(probe.body) == {"available": True}

    resp = await preview_client.request(str(target))
    assert resp.status_code == 200, resp.body
    payload = json.loads(resp.body)
    assert payload["kind"] == expected_kind
    assert payload["path"] == str(target)

    media_resp = await preview_client.fetch_media(payload["media_url"])
    assert media_resp.status_code == 200
    assert bytes(media_resp.body) == raw_bytes
    assert media_resp.headers["content-type"].startswith(expected_mime)


async def test_relative_path_with_spaces_from_quoted_command_and_multiline_stdout(
    preview_client: PreviewClient,
) -> None:
    target = (
        preview_client.root
        / "Documents"
        / "Freelance"
        / "Records"
        / "T&C"
        / "company website"
        / "docs"
        / "INV-000009.pdf"
    )
    target.parent.mkdir(parents=True)
    pdf_bytes = b"%PDF-1.7\n\x00invoice-9"
    target.write_bytes(pdf_bytes)

    rel_ref = "Documents/Freelance/Records/T&C/company website/docs/INV-000009.pdf"
    turn = "turn-inv-9"
    append_transcript_object(
        preview_client.key,
        {"event": "user", "text": "create invoice", "turn_id": turn},
    )
    append_transcript_object(
        preview_client.key,
        {
            "event": "message",
            "kind": "progress",
            "turn_id": turn,
            "tool_events": [
                {
                    "phase": "end",
                    "name": "exec",
                    "call_id": "call-render",
                    "arguments": {"command": f'/usr/bin/render-pdf INV-000009 --out "{target}"'},
                    "result": (
                        "InvoiceShelf up: http://localhost:8724\n"
                        f"Rendered INV-000009 -> {target} (878,937 bytes)\n\n"
                        "STDERR:\ncontainer warning"
                    ),
                    "error": None,
                }
            ],
        },
    )
    append_transcript_object(
        preview_client.key,
        {"event": "stream_end", "text": f"PDF saved:\n`{rel_ref}`", "turn_id": turn},
    )
    append_transcript_object(preview_client.key, {"event": "turn_end", "turn_id": turn})

    probe = await preview_client.request(rel_ref, probe=True)
    assert probe.status_code == 200
    assert json.loads(probe.body) == {"available": True}

    resp = await preview_client.request(rel_ref)
    assert resp.status_code == 200, resp.body
    payload = json.loads(resp.body)
    assert payload["kind"] == "pdf"
    assert payload["path"] == str(target)
    assert payload["mime"] == "application/pdf"

    media_resp = await preview_client.fetch_media(payload["media_url"])
    assert media_resp.status_code == 200
    assert bytes(media_resp.body) == pdf_bytes
    assert media_resp.headers["content-type"].startswith("application/pdf")


async def test_consolidated_transcript_text_ref_hydrates_for_external_preview(
    tmp_path: Path,
) -> None:
    from kohaku.session.execution_owner import acquire_execution
    from kohaku.session.runtime_authority import DurableSessionAuthority
    from kohaku.session.turn_writer import SourceTurnWriter
    from kohaku.webui.transcript import TRANSCRIPT_TEXT_SOURCE_REF_KEY

    previous_config = get_config_path()
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    external_dir = tmp_path / "external"
    project.mkdir()
    instance.mkdir()
    external_dir.mkdir()
    config_path = instance / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    set_config_path(config_path)

    target = external_dir / "INV-000009.pdf"
    pdf_bytes = b"%PDF-1.7\n\x00consolidated-pdf"
    target.write_bytes(pdf_bytes)

    lease = await DurableSessionAuthority(
        project,
        sessions_root=instance / "sessions",
    ).acquire()
    try:
        manager = lease.manager
        key = "websocket:consolidated-preview"
        assert manager.upsert_session_metadata(key, {"webui": True}, create_if_absent=True)
        backend = manager.source_backend(key)
        assert backend is not None

        async with acquire_execution(backend, key, run_id="run-consolidated") as owner:
            writer = SourceTurnWriter(backend.source, owner, run_id="run-consolidated")
            u_ev = writer.append_message({"role": "user", "content": "generate invoice"})
            a_ev = writer.append_message({"role": "assistant", "content": f"Saved `{target}`"})

        turn = "turn-consolidated"
        append_transcript_object(
            key,
            {
                "event": "user",
                "turn_id": turn,
                TRANSCRIPT_TEXT_SOURCE_REF_KEY: str(u_ev["_event_id"]),
            },
        )
        append_transcript_object(
            key,
            {
                "event": "stream_end",
                "turn_id": turn,
                TRANSCRIPT_TEXT_SOURCE_REF_KEY: str(a_ev["_event_id"]),
            },
        )
        append_transcript_object(key, {"event": "turn_end", "turn_id": turn})

        services = build_gateway_services(
            config=WebSocketConfig(host="127.0.0.1", port=29998, path="/"),
            bus=MessageBus(),
            session_manager=manager,
            static_dist_path=None,
            workspace_path=project,
            default_restrict_to_workspace=True,
            runtime_model_name=None,
            runtime_surface="desktop",
            runtime_capabilities_overrides=None,
        )
        token = services.tokens.issue_api_token(60)

        async def _req(path: str, *, probe: bool = False):
            q = {"path": path, **({"probe": "1"} if probe else {})}
            url = f"/api/sessions/{quote(key, safe='')}/file-preview?{urlencode(q)}"
            return await services.endpoint.process_request(
                None,
                Request(url, Headers({"Authorization": f"Bearer {token}", "Host": ORIGIN_HOST})),
                is_allowed=lambda _client: True,
            )

        probe = await _req(str(target), probe=True)
        assert probe.status_code == 200
        assert json.loads(probe.body) == {"available": True}

        resp = await _req(str(target))
        assert resp.status_code == 200, resp.body
        payload = json.loads(resp.body)
        assert payload["kind"] == "pdf"
        assert payload["path"] == str(target)
    finally:
        await lease.close()
        set_config_path(previous_config)
