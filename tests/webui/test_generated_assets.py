"""Generated asset gallery projection and route tests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Request

from nanobot.bus.queue import MessageBus
from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.config.loader import get_config_path, set_config_path
from nanobot.config.paths import get_media_dir
from nanobot.session.manager import SessionManager
from nanobot.webui.gateway_services import build_gateway_services
from nanobot.webui.generated_asset_index import (
    delete_generated_asset_index,
    generated_asset_index_path,
    load_or_build_generated_asset_index,
)
from nanobot.webui.generated_assets import (
    build_generated_assets_payload,
    build_generated_assets_payload_from_groups,
    collect_generated_asset_groups,
)
from nanobot.webui.transcript import (
    append_transcript_object,
    delete_webui_transcript,
    fork_transcript_before_user_index,
    read_transcript_lines,
)


def _event(
    *,
    call_id: str = "call-1",
    name: str = "mcp_comfy_generate_image",
    result: object,
    arguments: dict[str, object] | None = None,
    structured: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "phase": "end",
        "name": name,
        "call_id": call_id,
        "arguments": arguments or {"prompt": "a test prompt"},
        "result": json.dumps(result) if not isinstance(result, str) else result,
        "error": None,
    }
    if structured is not None:
        event["media_artifacts"] = structured
    return event


def _artifact(path: Path, *, artifact_id: str = "img-1", **extra: object) -> dict[str, object]:
    return {
        "id": artifact_id,
        "path": str(path),
        "mime": "image/png",
        "name": path.name,
        **extra,
    }


def _media_file(root: Path, name: str = "img-1.png") -> Path:
    path = root / "generated" / "2026-11-03" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png-bytes")
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "id": path.stem,
                "prompt": "sidecar prompt",
                "model": "qwen",
                "provider": "mcp:comfy",
                "created_at": "2026-11-03T14:39:12+08:00",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_collects_groups_in_tool_call_order_and_deduplicates(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    duplicate = tmp_path / "first.png"
    second = tmp_path / "second.png"
    records = [
        {
            "event": "message",
            "kind": "progress",
            "turn_id": "turn-1",
            "created_at_ms": 1000,
            "tool_events": [
                _event(result={"artifacts": [_artifact(first)]}),
                _event(
                    call_id="call-2",
                    result={"artifacts": [_artifact(second, artifact_id="img-2")]},
                ),
            ],
        },
        {
            "event": "message",
            "kind": "progress",
            "turn_id": "turn-1",
            "created_at_ms": 2000,
            "tool_events": [
                _event(result={"artifacts": [_artifact(duplicate)]}),
            ],
        },
    ]

    groups = collect_generated_asset_groups(records)

    assert [group["toolCallId"] for group in groups] == ["call-1", "call-2"]
    assert [group["sequence"] for group in groups] == [1, 2]
    assert len(groups[0]["artifacts"]) == 1
    assert groups[0]["startedAt"] == 1000
    assert groups[1]["artifacts"][0]["id"] == "img-2"


def test_keeps_generated_candidates_and_drops_reference_images(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"
    records = [
        {
            "event": "message",
            "kind": "progress",
            "turn_id": "turn-1",
            "tool_events": [
                _event(
                    result={
                        "artifacts": [
                            _artifact(reference, artifact_id="reference"),
                            _artifact(candidate, artifact_id="candidate"),
                        ],
                        "text": json.dumps(
                            {
                                "candidate_count": 1,
                                "candidates": [
                                    {"candidate": 1, "prompt": "candidate prompt"}
                                ],
                            }
                        ),
                    }
                )
            ],
        }
    ]

    groups = collect_generated_asset_groups(records)

    assert [item["id"] for item in groups[0]["artifacts"]] == ["candidate"]
    assert groups[0]["artifacts"][0]["candidate"] == 1
    assert groups[0]["artifacts"][0]["prompt"] == "candidate prompt"


def test_structured_media_artifacts_take_precedence(tmp_path: Path) -> None:
    structured = tmp_path / "structured.png"
    parsed = tmp_path / "parsed.png"
    records = [
        {
            "event": "message",
            "kind": "progress",
            "turn_id": "turn-1",
            "tool_events": [
                _event(
                    result={"artifacts": [_artifact(parsed, artifact_id="parsed")]},
                    structured=[_artifact(structured, artifact_id="structured")],
                )
            ],
        }
    ]

    groups = collect_generated_asset_groups(records)

    assert [item["id"] for item in groups[0]["artifacts"]] == ["structured"]


def test_payload_uses_structured_media_artifacts_without_result_payload(
    tmp_path: Path,
) -> None:
    media_root = tmp_path / "media"
    path = _media_file(media_root)
    records = [
        {
            "event": "message",
            "kind": "progress",
            "turn_id": "turn-1",
            "tool_events": [
                _event(
                    result="not json",
                    structured=[_artifact(path, artifact_id="structured-route")],
                )
            ],
        }
    ]

    payload = build_generated_assets_payload(
        records,
        session_key="websocket:assets",
        media_root=media_root,
        sign_path=lambda item: {"url": f"/media/{item.name}"},
    )

    assert payload["total"] == 1
    assert payload["groups"][0]["artifacts"][0]["id"] == "structured-route"


def test_payload_signs_media_enriches_sidecar_and_skips_unsafe_paths(
    tmp_path: Path,
) -> None:
    media_root = tmp_path / "media"
    valid = _media_file(media_root)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    missing = media_root / "generated" / "missing.png"
    records = [
        {
            "event": "message",
            "kind": "progress",
            "turn_id": "turn-1",
            "created_at_ms": 1000,
            "tool_events": [
                _event(
                    result={
                        "artifacts": [
                            _artifact(valid, artifact_id="img-valid"),
                            _artifact(outside, artifact_id="img-outside"),
                            _artifact(missing, artifact_id="img-missing"),
                        ]
                    }
                )
            ],
        }
    ]

    seen: list[Path] = []

    def sign_path(path: Path) -> dict[str, str]:
        seen.append(path)
        return {"url": f"/api/media/signed/{path.name}", "name": path.name}

    payload = build_generated_assets_payload(
        records,
        session_key="websocket:assets",
        media_root=media_root,
        sign_path=sign_path,
    )

    assert seen == [valid.resolve()]
    assert payload["schemaVersion"] == 1
    assert payload["sessionKey"] == "websocket:assets"
    assert payload["total"] == 1
    group = payload["groups"][0]
    assert group["toolCallId"] == "call-1"
    assert group["sequence"] == 1
    artifact = group["artifacts"][0]
    assert artifact["id"] == "img-valid"
    assert artifact["url"] == "/api/media/signed/img-1.png"
    assert artifact["path"] == str(valid.resolve())
    assert artifact["bytes"] == len(b"png-bytes")
    assert artifact["prompt"] == "sidecar prompt"
    assert artifact["model"] == "qwen"
    assert artifact["provider"] == "mcp:comfy"
    assert artifact["createdAt"] == int(
        datetime.fromisoformat("2026-11-03T14:39:12+08:00").timestamp() * 1000
    )


def test_payload_rejects_symlink_escape_from_media_root(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    link = media_root / "link.png"
    link.symlink_to(outside)
    records = [
        {
            "event": "message",
            "kind": "progress",
            "turn_id": "turn-1",
            "tool_events": [
                _event(result={"artifacts": [_artifact(link, artifact_id="escape")]})
            ],
        }
    ]

    payload = build_generated_assets_payload(
        records,
        session_key="websocket:assets",
        media_root=media_root,
        sign_path=lambda item: {"url": f"/media/{item.name}"},
    )

    assert payload["total"] == 0
    assert payload["groups"] == []


def test_payload_accepts_legacy_json_sidecars_and_iso_created_at(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    path = media_root / "generated" / "2026-11-03" / "legacy.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"legacy")
    records = [
        {
            "event": "message",
            "kind": "progress",
            "turn_id": "turn-1",
            "tool_events": [
                _event(
                    result={
                        "artifacts": [
                            _artifact(
                                path,
                                artifact_id="legacy",
                                created_at="2026-11-03T14:39:12+08:00",
                            )
                        ]
                    }
                )
            ],
        }
    ]

    payload = build_generated_assets_payload(
        records,
        session_key="websocket:assets",
        media_root=media_root,
        sign_path=lambda item: {"url": f"/media/{item.name}"},
    )

    artifact = payload["groups"][0]["artifacts"][0]
    assert artifact["id"] == "legacy"
    assert artifact["prompt"] == ""
    assert artifact["createdAt"] == int(
        datetime.fromisoformat("2026-11-03T14:39:12+08:00").timestamp() * 1000
    )


def test_payload_paginates_groups_and_reports_next_cursor(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    paths = [_media_file(media_root, f"img-{index}.png") for index in range(3)]
    records = [
        {
            "event": "message",
            "kind": "progress",
            "turn_id": f"turn-{index}",
            "tool_events": [
                _event(
                    call_id=f"call-{index}",
                    result={
                        "artifacts": [
                            _artifact(path, artifact_id=f"img-{index}")
                        ]
                    },
                )
            ],
        }
        for index, path in enumerate(paths)
    ]
    groups = collect_generated_asset_groups(records)

    first = build_generated_assets_payload_from_groups(
        groups,
        session_key="websocket:assets",
        media_root=media_root,
        sign_path=lambda item: {"url": f"/media/{item.name}"},
        limit=2,
    )
    second = build_generated_assets_payload_from_groups(
        groups,
        session_key="websocket:assets",
        media_root=media_root,
        sign_path=lambda item: {"url": f"/media/{item.name}"},
        cursor=2,
        limit=2,
    )

    assert first["total"] == 3
    assert [group["sequence"] for group in first["groups"]] == [1, 2]
    assert first["next"] == "2"
    assert [group["sequence"] for group in second["groups"]] == [3]
    assert second["next"] is None


def test_index_reuses_signature_and_is_deleted_with_transcript(tmp_path: Path) -> None:
    previous = get_config_path()
    set_config_path(tmp_path / "config.json")
    try:
        key = "websocket:indexed-assets"
        path = _media_file(tmp_path / "media")
        records = [
            {
                "event": "message",
                "kind": "progress",
                "turn_id": "turn-1",
                "tool_events": [
                    _event(result={"artifacts": [_artifact(path, artifact_id="img-index")]})
                ],
            }
        ]
        calls = 0

        def load_records() -> list[dict[str, object]]:
            nonlocal calls
            calls += 1
            return records

        first = load_or_build_generated_asset_index(
            key,
            signature=(("active", 1, 1),),
            records_loader=load_records,
        )
        second = load_or_build_generated_asset_index(
            key,
            signature=(("active", 1, 1),),
            records_loader=load_records,
        )
        third = load_or_build_generated_asset_index(
            key,
            signature=(("active", 2, 2),),
            records_loader=load_records,
        )

        assert len(first) == len(second) == len(third) == 1
        assert calls == 2
        assert generated_asset_index_path(key).is_file()

        append_transcript_object(key, {"event": "user", "text": "clear me"})
        assert delete_webui_transcript(key) is True
        assert not generated_asset_index_path(key).exists()
        assert delete_generated_asset_index(key) is False
    finally:
        set_config_path(previous)


@pytest.fixture
def assets_client(tmp_path: Path):
    previous = get_config_path()
    project = tmp_path / "project"
    instance = tmp_path / "instance"
    project.mkdir()
    instance.mkdir()
    config_path = instance / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    set_config_path(config_path)
    try:
        services = build_gateway_services(
            config=WebSocketConfig(host="127.0.0.1", port=29999, path="/"),
            bus=MessageBus(),
            session_manager=SessionManager(project, sessions_root=instance / "sessions"),
            static_dist_path=None,
            workspace_path=project,
            default_restrict_to_workspace=True,
            runtime_model_name=None,
            runtime_surface="webui",
            runtime_capabilities_overrides=None,
        )
        yield project, instance, services, services.tokens.issue_api_token(60)
    finally:
        set_config_path(previous)


async def test_route_returns_session_assets_and_requires_auth(assets_client) -> None:
    _project, instance, services, token = assets_client
    path = _media_file(instance / "media")
    key = "websocket:generated-assets"
    append_transcript_object(
        key,
        {
            "event": "message",
            "kind": "progress",
            "turn_id": "turn-1",
            "created_at_ms": 1000,
            "tool_events": [
                _event(
                    result={
                        "artifacts": [
                            _artifact(path, artifact_id="img-route"),
                        ]
                    }
                )
            ],
        },
    )

    response = await services.endpoint.process_request(
        None,
        Request(
            f"/api/sessions/{key}/generated-assets",
            Headers({"Authorization": f"Bearer {token}"}),
        ),
        is_allowed=lambda _client: True,
    )

    assert response.status_code == 200, response.body
    payload = json.loads(response.body)
    assert payload["sessionKey"] == key
    assert payload["total"] == 1
    artifact = payload["groups"][0]["artifacts"][0]
    assert artifact["id"] == "img-route"
    assert artifact["url"].startswith("/api/media/")

    unauthorized = await services.endpoint.process_request(
        None,
        Request(f"/api/sessions/{key}/generated-assets", Headers({})),
        is_allowed=lambda _client: True,
    )
    assert unauthorized.status_code == 401


async def test_route_paginates_generated_assets(assets_client) -> None:
    _project, instance, services, token = assets_client
    paths = [
        _media_file(instance / "media", f"route-page-{index}.png")
        for index in range(2)
    ]
    key = "websocket:generated-assets-page"
    append_transcript_object(
        key,
        {
            "event": "message",
            "kind": "progress",
            "turn_id": "turn-1",
            "tool_events": [
                _event(
                    call_id=f"call-{index}",
                    result={
                        "artifacts": [
                            _artifact(path, artifact_id=f"route-page-{index}")
                        ]
                    },
                )
                for index, path in enumerate(paths)
            ],
        },
    )

    first = await services.endpoint.process_request(
        None,
        Request(
            f"/api/sessions/{key}/generated-assets?limit=1",
            Headers({"Authorization": f"Bearer {token}"}),
        ),
        is_allowed=lambda _client: True,
    )
    second = await services.endpoint.process_request(
        None,
        Request(
            f"/api/sessions/{key}/generated-assets?limit=1&cursor=1",
            Headers({"Authorization": f"Bearer {token}"}),
        ),
        is_allowed=lambda _client: True,
    )

    assert first.status_code == 200, first.body
    first_payload = json.loads(first.body)
    assert first_payload["total"] == 2
    assert len(first_payload["groups"]) == 1
    assert first_payload["next"] == "1"
    second_payload = json.loads(second.body)
    assert len(second_payload["groups"]) == 1
    assert second_payload["next"] is None


async def test_global_generated_asset_library_pages_across_retained_sessions(
    assets_client,
) -> None:
    from urllib.parse import urlencode

    _project, instance, services, token = assets_client
    asset_indices = {0, 100, 200}
    expected_ids = set()
    for index in range(201):
        session = f"library-page-{index:03d}"
        append_transcript_object(
            f"websocket:{session}",
            {"event": "user", "text": "hello", "turn_id": "turn-1"},
        )
        if index not in asset_indices:
            continue
        artifact_id = f"library-page-{index:03d}"
        path = _media_file(instance / "media", f"{artifact_id}.png")
        expected_ids.add(artifact_id)
        append_transcript_object(
            f"websocket:{session}",
            {
                "event": "message",
                "kind": "progress",
                "turn_id": "turn-1",
                "tool_events": [
                    _event(
                        call_id=f"call-{index}",
                        result={"artifacts": [_artifact(path, artifact_id=artifact_id)]},
                    )
                ],
            },
        )

    first = await services.endpoint.process_request(
        None,
        Request("/api/generated-assets", Headers({"Authorization": f"Bearer {token}"})),
        is_allowed=lambda _client: True,
    )
    assert first.status_code == 200, first.body
    first_payload = json.loads(first.body)
    assert first_payload["complete"] is False
    assert first_payload["sessionsScanned"] == 200
    assert first_payload["totalScope"] == "page"
    assert json.loads(first_payload["next"]) == ["websocket:library-page-200", 0]
    first_ids = {
        artifact["id"]
        for group in first_payload["groups"]
        for artifact in group["artifacts"]
    }
    assert first_ids == {"library-page-000", "library-page-100"}

    second = await services.endpoint.process_request(
        None,
        Request(
            f"/api/generated-assets?{urlencode({'cursor': first_payload['next']})}",
            Headers({"Authorization": f"Bearer {token}"}),
        ),
        is_allowed=lambda _client: True,
    )
    assert second.status_code == 200, second.body
    second_payload = json.loads(second.body)
    assert second_payload["next"] is None
    assert second_payload["complete"] is True
    assert second_payload["sessionsScanned"] == 1
    second_ids = {
        artifact["id"]
        for group in second_payload["groups"]
        for artifact in group["artifacts"]
    }
    assert second_ids == {"library-page-200"}

    keys = [
        group["sessionKey"]
        for payload in (first_payload, second_payload)
        for group in payload["groups"]
    ]
    assert len(keys) == len(set(keys)) == 3
    assert first_ids | second_ids == expected_ids
    assert len(first_ids) + len(second_ids) == len(expected_ids)


@pytest.mark.parametrize("session_count", [0, 200, 205])
async def test_library_traverses_all_sessions_without_duplicate_groups(
    assets_client, session_count: int,
) -> None:
    from urllib.parse import urlencode

    _project, instance, services, token = assets_client
    path = _media_file(instance / "media", "library-page.png")
    expected: set[tuple[str, str]] = set()
    for index in range(session_count):
        key = f"websocket:page-{index:03d}"
        # Multiple groups in a session exercise the within-session continuation.
        events = [_event(call_id=f"call-{n}", result={"artifacts": [_artifact(path)]})
                  for n in range(2)]
        append_transcript_object(key, {"event": "message", "tool_events": events})
        expected.update((key, f"legacy:0|call-{n}") for n in range(2))

    cursor = None
    seen: list[tuple[str, str]] = []
    while True:
        query = {"limit": "37"}
        if cursor is not None:
            query["cursor"] = cursor
        response = await services.endpoint.process_request(
            None,
            Request(f"/api/generated-assets?{urlencode(query)}",
                    Headers({"Authorization": f"Bearer {token}"})),
            is_allowed=lambda _client: True,
        )
        assert response.status_code == 200, response.body
        payload = json.loads(response.body)
        assert payload["totalScope"] == "page"
        assert payload["sessionsScanned"] <= 200
        assert len(payload["groups"]) <= 37
        assert payload["total"] == sum(len(group["artifacts"]) for group in payload["groups"])
        seen.extend((group["sessionKey"], group["id"]) for group in payload["groups"])
        cursor = payload["next"]
        assert payload["complete"] == (cursor is None)
        if cursor is None:
            break
    assert len(seen) == len(set(seen))
    assert set(seen) == expected


async def test_library_continues_past_a_full_empty_session_batch(assets_client) -> None:
    from urllib.parse import urlencode

    _project, instance, services, token = assets_client
    for index in range(200):
        append_transcript_object(f"websocket:empty-{index:03d}", {"event": "user", "text": "hi"})
    path = _media_file(instance / "media", "recent.png")
    append_transcript_object("websocket:z-recent", {
        "event": "message",
        "tool_events": [_event(result={"artifacts": [_artifact(path)]})],
    })
    headers = Headers({"Authorization": f"Bearer {token}"})
    first = await services.endpoint.process_request(
        None, Request("/api/generated-assets", headers), is_allowed=lambda _client: True,
    )
    payload = json.loads(first.body)
    assert payload["groups"] == []
    assert payload["sessionsScanned"] == 200
    assert payload["complete"] is False
    second = await services.endpoint.process_request(
        None, Request(f"/api/generated-assets?{urlencode({'cursor': payload['next']})}", headers),
        is_allowed=lambda _client: True,
    )
    payload = json.loads(second.body)
    assert payload["groups"][0]["sessionKey"] == "websocket:z-recent"
    assert payload["sessionsScanned"] == 1
    assert payload["complete"] is True


@pytest.mark.parametrize("cursor", ["-1", "nonsense", '["key", true]', '["key", -1]'])
async def test_library_rejects_invalid_cursors(assets_client, cursor: str) -> None:
    from urllib.parse import urlencode

    _project, _instance, services, token = assets_client
    response = await services.endpoint.process_request(
        None,
        Request(f"/api/generated-assets?{urlencode({'cursor': cursor})}",
                Headers({"Authorization": f"Bearer {token}"})),
        is_allowed=lambda _client: True,
    )
    assert response.status_code == 400


def test_forked_session_keeps_generated_asset_references(assets_client) -> None:
    _project, instance, _services, _token = assets_client
    path = _media_file(instance / "media")
    source = "websocket:source-assets"
    append_transcript_object(source, {"event": "user", "text": "first", "turn_id": "turn-1"})
    append_transcript_object(
        source,
        {
            "event": "message",
            "kind": "progress",
            "turn_id": "turn-1",
            "created_at_ms": 1000,
            "tool_events": [
                _event(result={"artifacts": [_artifact(path, artifact_id="img-fork")]})
            ],
        },
    )
    append_transcript_object(source, {"event": "turn_end", "turn_id": "turn-1"})
    append_transcript_object(source, {"event": "user", "text": "second", "turn_id": "turn-2"})

    assert fork_transcript_before_user_index(source, "websocket:forked-assets", 1) is True

    payload = build_generated_assets_payload(
        read_transcript_lines("websocket:forked-assets"),
        session_key="websocket:forked-assets",
        media_root=get_media_dir(),
        sign_path=lambda item: {"url": f"/media/{item.name}"},
    )

    assert payload["total"] == 1
    assert payload["groups"][0]["artifacts"][0]["id"] == "img-fork"
