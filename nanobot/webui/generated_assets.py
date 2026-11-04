"""Session-scoped generated media assets for the WebUI.

The first implementation derives asset groups from persisted WebUI transcript tool
events. Structured ``media_artifacts`` emitted by newer tools take precedence; the
result JSON is retained as a compatibility path for existing sessions.
"""

from __future__ import annotations

import json
import mimetypes
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from nanobot.utils.artifacts import (
    parse_embedded_text_payload,
    select_candidate_artifacts,
)

SCHEMA_VERSION = 1
_MAX_SIDECAR_BYTES = 256 * 1024


def collect_generated_asset_groups(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collect artifact groups from transcript records without touching the filesystem."""
    groups: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    sequence = 0

    for record_index, record in enumerate(records):
        raw_events = record.get("tool_events")
        if not isinstance(raw_events, list):
            continue
        turn_id = record.get("turn_id")
        turn = str(turn_id) if isinstance(turn_id, str) and turn_id else f"legacy:{record_index}"
        started_at = _created_at_ms(record.get("created_at_ms"))

        for event_index, raw_event in enumerate(cast(list[Any], raw_events)):
            if not isinstance(raw_event, dict):
                continue
            event = cast(dict[str, Any], raw_event)
            if event.get("phase") != "end" or event.get("error"):
                continue
            artifacts = _event_artifacts(event)
            if not artifacts:
                continue

            name = event.get("name")
            tool_name = name if isinstance(name, str) and name else "tool"
            call_id = event.get("call_id")
            call = str(call_id) if isinstance(call_id, str) and call_id else f"{name}:{event_index}"
            key = f"{turn}|{call}"
            existing = by_key.get(key)
            if existing is None:
                sequence += 1
                arguments = event.get("arguments")
                group: dict[str, Any] = {
                    "id": key,
                    "turnId": turn,
                    "toolCallId": call,
                    "toolName": tool_name,
                    "sequence": sequence,
                    "startedAt": started_at,
                    "arguments": (
                        dict(cast(Mapping[str, Any], arguments))
                        if isinstance(arguments, Mapping)
                        else {}
                    ),
                    "artifacts": [],
                }
                by_key[key] = group
                groups.append(group)
            else:
                group = existing

            seen = {_artifact_key(item) for item in cast(list[dict[str, Any]], group["artifacts"])}
            for artifact in artifacts:
                marker = _artifact_key(artifact)
                if marker in seen:
                    continue
                seen.add(marker)
                cast(list[dict[str, Any]], group["artifacts"]).append(artifact)

    return [group for group in groups if group["artifacts"]]


def build_generated_assets_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    session_key: str,
    media_root: Path,
    sign_path: Callable[[Path], Mapping[str, str] | None],
    cursor: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return the stable API payload for one session's generated assets."""
    return build_generated_assets_payload_from_groups(
        collect_generated_asset_groups(records),
        session_key=session_key,
        media_root=media_root,
        sign_path=sign_path,
        cursor=cursor,
        limit=limit,
    )


def build_generated_assets_payload_from_groups(
    raw_groups: Sequence[Mapping[str, Any]],
    *,
    session_key: str,
    media_root: Path,
    sign_path: Callable[[Path], Mapping[str, str] | None],
    cursor: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Finalize indexed raw groups and return one page of signed asset groups."""
    root = media_root.resolve()
    groups: list[dict[str, Any]] = []
    total = 0

    for raw_group in raw_groups:
        artifacts: list[dict[str, Any]] = []
        for raw_artifact in cast(list[dict[str, Any]], raw_group.get("artifacts") or []):
            artifact = _resolve_artifact(raw_artifact, root, sign_path)
            if artifact is not None:
                artifacts.append(artifact)
        if not artifacts:
            continue
        total += len(artifacts)
        group = {key: value for key, value in raw_group.items() if key != "artifacts"}
        group["sequence"] = len(groups) + 1
        group["artifacts"] = artifacts
        groups.append(group)

    start = max(0, cursor)
    page = groups[start:] if limit is None else groups[start:start + max(0, limit)]
    next_cursor = start + len(page)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "sessionKey": session_key,
        "total": total,
        "groups": page,
        "next": str(next_cursor) if next_cursor < len(groups) else None,
    }


def _event_artifacts(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    structured = event.get("media_artifacts")
    if isinstance(structured, list):
        items = [_artifact_item(item) for item in cast(list[Any], structured)]
        if items:
            return select_candidate_artifacts(items, None)

    result = event.get("result")
    payload: Mapping[str, Any]
    if isinstance(result, str):
        text = result.strip()
        if not text or len(text) > 16 * 1024 * 1024:
            return []
        try:
            parsed: Any = json.loads(text)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, Mapping):
            return []
        payload = cast(Mapping[str, Any], parsed)
    elif isinstance(result, Mapping):
        payload = cast(Mapping[str, Any], result)
    else:
        return []

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return []
    items = [_artifact_item(item) for item in cast(list[Any], artifacts)]
    return select_candidate_artifacts(items, parse_embedded_text_payload(payload.get("text")))


def _artifact_item(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    item = cast(Mapping[str, Any], value)
    raw_path = item.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return {}
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return {}
    name = item.get("name")
    if not isinstance(name, str) or not name:
        name = path.name
    mime = item.get("mime")
    if not isinstance(mime, str) or not mime:
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    kind = item.get("kind")
    if not isinstance(kind, str) or not kind:
        kind = _kind_for_mime(mime)
    artifact_id = item.get("id")
    if not isinstance(artifact_id, str) or not artifact_id:
        artifact_id = path.stem
    return {
        "id": artifact_id,
        "kind": kind,
        "mime": mime,
        "name": name,
        "path": str(path),
        "prompt": _optional_text(item.get("prompt")),
        "model": _optional_text(item.get("model")),
        "provider": _optional_text(item.get("provider")),
        "sourceImages": _string_list(item.get("source_images")),
        "createdAt": _created_at_ms(item.get("created_at")),
    }


def _resolve_artifact(
    raw: dict[str, Any],
    media_root: Path,
    sign_path: Callable[[Path], Mapping[str, str] | None],
) -> dict[str, Any] | None:
    if not raw:
        return None
    path = _safe_media_file(raw.get("path"), media_root)
    if path is None:
        return None
    sidecar = _read_sidecar(path)
    mime = raw.get("mime") if isinstance(raw.get("mime"), str) else None
    if not mime or mime == "application/octet-stream":
        mime = _optional_text(sidecar.get("mime")) or mimetypes.guess_type(path.name)[0]
    mime = mime or "application/octet-stream"
    kind = raw.get("kind") if isinstance(raw.get("kind"), str) else None
    if not kind or kind == "file":
        kind = _kind_for_mime(mime)

    signed = sign_path(path)
    if not signed or not signed.get("url"):
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None

    created_at = _created_at_ms(sidecar.get("created_at")) or raw.get("createdAt")
    return {
        "id": raw.get("id") or path.stem,
        "kind": kind,
        "mime": mime,
        "name": raw.get("name") or path.name,
        "url": signed["url"],
        "path": str(path),
        "bytes": size,
        "createdAt": created_at,
        "prompt": raw.get("prompt") or _optional_text(sidecar.get("prompt")),
        "model": raw.get("model") or _optional_text(sidecar.get("model")),
        "provider": raw.get("provider") or _optional_text(sidecar.get("provider")),
        "sourceImages": raw.get("sourceImages") or _string_list(sidecar.get("source_images")),
    }


def _safe_media_file(raw: Any, media_root: Path) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return None
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_file():
        return None
    try:
        resolved.relative_to(media_root)
    except ValueError:
        return None
    return resolved


def _read_sidecar(path: Path) -> dict[str, Any]:
    sidecar_path = path.with_suffix(".json")
    try:
        if not sidecar_path.is_file() or sidecar_path.stat().st_size > _MAX_SIDECAR_BYTES:
            return {}
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return cast(dict[str, Any], data) if isinstance(data, dict) else {}


def _artifact_key(artifact: Mapping[str, Any]) -> str:
    artifact_id = artifact.get("id")
    if isinstance(artifact_id, str) and artifact_id:
        return f"id:{artifact_id}"
    path = artifact.get("path")
    return f"path:{os.path.normpath(str(path))}" if isinstance(path, str) else ""


def _kind_for_mime(mime: str) -> str:
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("video/"):
        return "video"
    return "file"


def _created_at_ms(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value >= 0:
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            return None
    return None


def _optional_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in cast(list[Any], value) if isinstance(item, str)]
