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

from kohaku.config.paths import get_webui_dir
from kohaku.session.manager import SessionManager
from kohaku.webui.generated_assets import (
    SCHEMA_VERSION,
    build_generated_assets_payload_from_groups,
    collect_generated_asset_groups,
)
from kohaku.webui.session_identity import (
    WEBUI_SESSION_STORAGE_PREFIX,
    webui_session_key,
)
from kohaku.webui.transcript import read_transcript_lines, webui_transcript_signature

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
        from kohaku.attachments._media import fsync_directory
        from kohaku.webui.transcript import media_publication

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
    return [key for _, key in _webui_session_positions(webui_dir, newest_first=False)]


def _webui_session_positions(
    webui_dir: Path, *, newest_first: bool,
) -> list[tuple[int, str]]:
    positions: dict[str, int] = {}
    if not webui_dir.is_dir():
        return []
    persisted_prefix = SessionManager.safe_key(WEBUI_SESSION_STORAGE_PREFIX)
    for path in webui_dir.iterdir():
        stem: str | None = None
        if path.is_file() and path.suffix == ".jsonl":
            stem = path.stem
            if newest_first:
                base, marker, generation = stem.rpartition(".active.")
                if marker and len(generation) == 16 and (webui_dir / f"{base}.segments").is_dir():
                    stem = base
        elif path.is_dir() and path.name.endswith(".segments"):
            stem = path.name.removesuffix(".segments")
        if stem is None:
            continue
        decoded = SessionManager.decode_storage_key(stem)
        if decoded is not None and decoded.startswith(WEBUI_SESSION_STORAGE_PREFIX):
            key = decoded
        elif stem.startswith(persisted_prefix) and (chat_id := stem[len(persisted_prefix):]):
            key = webui_session_key(chat_id)
        else:
            continue
        rank = 0
        if newest_first:
            try:
                rank = -path.stat().st_mtime_ns
            except OSError:
                continue
        positions[key] = min(rank, positions.get(key, rank))
    return sorted((rank, key) for key, rank in positions.items())


def build_generated_asset_library_page(
    webui_dir: Path,
    *,
    media_root: Path,
    sign_path: Callable[[Path], Mapping[str, str] | None],
    cursor: str | None = None,
    limit: int = 50,
    max_sessions: int = 200,
    newest_first: bool = False,
) -> dict[str, Any]:
    """Read a bounded session page, continuing within a session when necessary.

    Library totals count returned artifacts only. A global count would require
    reading every transcript. The cursor anchors to the session's sort position.
    """
    anchor, offset = _library_cursor(cursor, newest_first=newest_first)
    positions = _webui_session_positions(webui_dir, newest_first=newest_first)
    start = bisect_left(positions, anchor) if anchor is not None else 0
    groups: list[dict[str, Any]] = []
    next_cursor: str | None = None
    scanned = 0
    for index in range(start, min(len(positions), start + max_sessions)):
        position = positions[index]
        _, session_key = position
        scanned += 1
        session_groups = load_or_build_generated_asset_index(
            session_key,
            signature=webui_transcript_signature(session_key),
            records_loader=lambda key=session_key: read_transcript_lines(key),
        )
        group_offset = offset if position == anchor else 0
        payload = build_generated_assets_payload_from_groups(
            session_groups,
            session_key=session_key,
            media_root=media_root,
            sign_path=sign_path,
            cursor=group_offset,
            limit=limit - len(groups),
            newest_first=newest_first,
        )
        page: list[dict[str, Any]] = payload["groups"]
        for group in page:
            item = dict(group)
            item["sessionKey"] = session_key
            groups.append(item)
        if payload["next"] is not None:
            next_cursor = _encode_library_cursor(
                position, group_offset + len(page), newest_first=newest_first,
            )
            break
        if index + 1 < len(positions):
            next_cursor = _encode_library_cursor(
                positions[index + 1], 0, newest_first=newest_first,
            )
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


def _encode_library_cursor(
    position: tuple[int, str], offset: int, *, newest_first: bool,
) -> str:
    rank, key = position
    return json.dumps([key, offset, rank] if newest_first else [key, offset])


def _library_cursor(
    raw: str | None, *, newest_first: bool,
) -> tuple[tuple[int, str] | None, int]:
    if raw is None or not raw.strip():
        return None, 0
    if len(raw) > 4096:
        raise ValueError("invalid cursor")
    try:
        value: object = json.loads(raw)
    except ValueError as exc:
        raise ValueError("invalid cursor") from exc
    if not isinstance(value, list):
        raise ValueError("invalid cursor")
    parts = cast(list[object], value)
    if len(parts) != (3 if newest_first else 2):
        raise ValueError("invalid cursor")
    key, offset = parts[:2]
    if not isinstance(key, str) or not key or type(offset) is not int or offset < 0:
        raise ValueError("invalid cursor")
    rank = parts[2] if newest_first else 0
    if type(rank) is not int:
        raise ValueError("invalid cursor")
    return (rank, key), offset
