"""Persisted per-session projection of generated media asset groups."""

from __future__ import annotations

import json
import os
import uuid
from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from loguru import logger

from nanobot.config.paths import get_webui_dir
from nanobot.session.manager import SessionManager
from nanobot.webui.generated_assets import (
    SCHEMA_VERSION,
    build_generated_assets_payload_from_groups,
    collect_generated_asset_groups,
)
from nanobot.webui.session_identity import (
    WEBUI_SESSION_STORAGE_PREFIX,
    webui_session_key,
)
from nanobot.webui.transcript import read_transcript_lines, webui_transcript_signature

INDEX_VERSION = 1


def generated_asset_index_path(session_key: str) -> Path:
    return get_webui_dir() / f"{SessionManager.storage_key(session_key)}.generated-assets.json"


def read_generated_asset_index(session_key: str) -> dict[str, Any] | None:
    path = generated_asset_index_path(session_key)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return cast(dict[str, Any], data) if isinstance(data, dict) else None


def write_generated_asset_index(
    session_key: str,
    *,
    signature: Sequence[Sequence[int | str]],
    groups: list[dict[str, Any]],
) -> None:
    path = generated_asset_index_path(session_key)
    payload = {
        "version": INDEX_VERSION,
        "sessionKey": session_key,
        "signature": [list(item) for item in signature],
        "groups": groups,
    }
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        from nanobot.attachments._media import fsync_directory
        from nanobot.webui.transcript import media_publication

        with media_publication(payload):
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(path)
            fsync_directory(path.parent)
    except OSError:
        logger.warning("failed to persist generated asset index for {}", session_key)
        tmp.unlink(missing_ok=True)


def load_or_build_generated_asset_index(
    session_key: str,
    *,
    signature: Sequence[Sequence[int | str]],
    records_loader: Callable[[], Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    expected = [list(item) for item in signature]
    cached = read_generated_asset_index(session_key)
    if (
        cached is not None
        and cached.get("version") == INDEX_VERSION
        and cached.get("sessionKey") == session_key
        and cached.get("signature") == expected
        and isinstance(cached.get("groups"), list)
    ):
        return [
            cast(dict[str, Any], group)
            for group in cast(list[Any], cached["groups"])
            if isinstance(group, dict)
        ]

    groups = collect_generated_asset_groups(records_loader())
    write_generated_asset_index(session_key, signature=signature, groups=groups)
    return groups


def delete_generated_asset_index(session_key: str) -> bool:
    path = generated_asset_index_path(session_key)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning("failed to delete generated asset index for {}", session_key)
        return False
    return True


def webui_session_keys(webui_dir: Path) -> list[str]:
    """Derive WebSocket session keys from persisted transcript files."""
    keys: set[str] = set()
    if not webui_dir.is_dir():
        return []
    persisted_prefix = SessionManager.safe_key(WEBUI_SESSION_STORAGE_PREFIX)
    for path in webui_dir.iterdir():
        stem: str | None = None
        if path.is_file() and path.suffix == ".jsonl":
            stem = path.stem
        elif path.is_dir() and path.name.endswith(".segments"):
            stem = path.name.removesuffix(".segments")
        if stem is None:
            continue
        decoded = SessionManager.decode_storage_key(stem)
        if decoded is not None and decoded.startswith(WEBUI_SESSION_STORAGE_PREFIX):
            keys.add(decoded)
            continue
        if not stem.startswith(persisted_prefix):
            continue
        chat_id = stem[len(persisted_prefix):]
        if chat_id:
            keys.add(webui_session_key(chat_id))
    return sorted(keys)


def build_generated_asset_library_page(
    webui_dir: Path,
    *,
    media_root: Path,
    sign_path: Callable[[Path], Mapping[str, str] | None],
    cursor: str | None = None,
    limit: int = 50,
    max_sessions: int = 200,
) -> dict[str, Any]:
    """Read a bounded session page, continuing within a session when necessary.

    Library totals count returned artifacts only. A global count would require
    reading every transcript. The opaque cursor anchors to a session key so
    inserting/removing earlier sessions cannot shift later pages.
    """
    anchor, offset = _library_cursor(cursor)
    keys = webui_session_keys(webui_dir)
    start = bisect_left(keys, anchor)
    groups: list[dict[str, Any]] = []
    next_cursor: str | None = None
    scanned = 0
    for index in range(start, min(len(keys), start + max_sessions)):
        session_key = keys[index]
        scanned += 1
        session_groups = load_or_build_generated_asset_index(
            session_key,
            signature=webui_transcript_signature(session_key),
            records_loader=lambda key=session_key: read_transcript_lines(key),
        )
        group_offset = offset if session_key == anchor else 0
        payload = build_generated_assets_payload_from_groups(
            session_groups,
            session_key=session_key,
            media_root=media_root,
            sign_path=sign_path,
            cursor=group_offset,
            limit=limit - len(groups),
        )
        page: list[dict[str, Any]] = payload["groups"]
        for group in page:
            item = dict(group)
            item["sessionKey"] = session_key
            groups.append(item)
        if payload["next"] is not None:
            next_cursor = json.dumps([session_key, group_offset + len(page)])
            break
        if index + 1 < len(keys):
            next_cursor = json.dumps([keys[index + 1], 0])
        else:
            next_cursor = None
        if len(groups) >= limit:
            break
    return {
        "schemaVersion": SCHEMA_VERSION,
        "sessionKey": "*",
        "groups": groups,
        "total": sum(len(group["artifacts"]) for group in groups),
        "totalScope": "page",
        "complete": next_cursor is None,
        "sessionsScanned": scanned,
        "next": next_cursor,
    }


def _library_cursor(raw: str | None) -> tuple[str, int]:
    if raw is None or not raw.strip():
        return "", 0
    if len(raw) > 4096:
        raise ValueError("invalid cursor")
    try:
        value: object = json.loads(raw)
    except ValueError as exc:
        raise ValueError("invalid cursor") from exc
    if not isinstance(value, list):
        raise ValueError("invalid cursor")
    parts = cast(list[object], value)
    if len(parts) != 2:
        raise ValueError("invalid cursor")
    key, offset = parts
    if not isinstance(key, str) or not key or type(offset) is not int or offset < 0:
        raise ValueError("invalid cursor")
    return key, offset
