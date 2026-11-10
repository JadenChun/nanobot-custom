"""Persisted per-session projection of generated media asset groups."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from loguru import logger

from nanobot.config.paths import get_webui_dir
from nanobot.session.manager import SessionManager
from nanobot.webui.generated_assets import collect_generated_asset_groups
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
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
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


def _load_generated_asset_groups(
    session_keys: Sequence[str],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for session_key in session_keys:
        session_groups = load_or_build_generated_asset_index(
            session_key,
            signature=webui_transcript_signature(session_key),
            records_loader=lambda key=session_key: read_transcript_lines(key),
        )
        for group in session_groups:
            item = dict(group)
            item["sessionKey"] = session_key
            groups.append(item)
    return groups


def load_all_generated_asset_page(
    webui_dir: Path,
    *,
    session_cursor: int = 0,
    max_sessions: int = 200,
) -> tuple[list[dict[str, Any]], int | None]:
    """Return one bounded session page and the next stable session cursor."""
    keys = webui_session_keys(webui_dir)
    start = max(0, session_cursor)
    page_keys = keys[start:start + max(0, max_sessions)]
    groups = _load_generated_asset_groups(page_keys)
    next_cursor = start + len(page_keys)
    return groups, (next_cursor if next_cursor < len(keys) else None)


def load_all_generated_asset_groups(
    webui_dir: Path,
    *,
    max_sessions: int = 200,
    session_cursor: int = 0,
) -> list[dict[str, Any]]:
    """Return indexed groups for one bounded session page."""
    groups, _next = load_all_generated_asset_page(
        webui_dir,
        session_cursor=session_cursor,
        max_sessions=max_sessions,
    )
    return groups
