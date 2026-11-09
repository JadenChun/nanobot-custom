"""Retention and orphan cleanup for generated media artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.config.paths import get_media_dir, get_webui_dir
from nanobot.config.schema import Config

_JSON_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)+)"')


def collect_referenced_media_evidence(
    webui_dir: Path,
    media_root: Path,
) -> tuple[set[Path], bool]:
    """Collect references and report whether every evidence file was readable."""
    root = media_root.resolve()
    referenced: set[Path] = set()
    complete = True
    for path in webui_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            complete = False
            continue
        for match in _JSON_STRING_RE.finditer(text):
            try:
                value = json.loads(f'"{match.group(1)}"')
            except json.JSONDecodeError:
                continue
            if not isinstance(value, str) or not value.startswith("/"):
                continue
            candidate = Path(value)
            try:
                resolved = candidate.resolve(strict=False)
                resolved.relative_to(root)
            except (OSError, ValueError):
                continue
            referenced.add(resolved)
    return referenced, complete


def collect_referenced_media_paths(webui_dir: Path, media_root: Path) -> set[Path]:
    """Collect absolute media paths mentioned by readable WebUI evidence."""
    return collect_referenced_media_evidence(webui_dir, media_root)[0]


def garbage_collect_generated_media(
    *,
    artifact_root: Path,
    referenced: set[Path],
    retention_days: int = 0,
    orphan_grace_days: int = 7,
    max_bytes: int = 0,
    now: datetime | None = None,
    dry_run: bool = False,
    evidence_complete: bool = True,
) -> dict[str, Any]:
    """Delete unreferenced/expired generated media and enforce a byte quota.

    A retained reference always blocks reclamation. If any reference evidence
    is missing or unreadable, no deletion is attempted.
    """
    now_ts = (now or datetime.now().astimezone()).timestamp()
    root = artifact_root.resolve()
    assets: list[tuple[Path, float, int]] = []
    if root.is_dir():
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() == ".json":
                continue
            try:
                info = path.stat()
            except OSError:
                continue
            assets.append((path, (now_ts - info.st_mtime) / 86400, info.st_size))

    to_delete: list[tuple[Path, int, str]] = []
    if evidence_complete:
        for path, age_days, size in assets:
            try:
                resolved = path.resolve()
            except OSError:
                evidence_complete = False
                break
            if resolved in referenced:
                continue
            if retention_days > 0 and age_days >= retention_days:
                to_delete.append((path, size, "retention"))
            elif age_days >= orphan_grace_days:
                to_delete.append((path, size, "orphan"))
    if not evidence_complete:
        # Missing or unreadable reference evidence can hide a live file.
        to_delete = []

    deleted_paths = {path for path, _size, _reason in to_delete}
    deleted_count = len(to_delete)
    deleted_bytes = sum(size for _path, size, _reason in to_delete)
    if not dry_run:
        for path, _size, _reason in to_delete:
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)

    remaining = [
        (path, size)
        for path, _age, size in assets
        if path not in deleted_paths
    ]
    kept_bytes = sum(size for _path, size in remaining)
    if max_bytes > 0 and kept_bytes > max_bytes:
        for path, size in sorted(
            remaining,
            key=lambda item: item[0].stat().st_mtime,
        ):
            if kept_bytes <= max_bytes:
                break
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in referenced:
                continue
            if not dry_run:
                path.unlink(missing_ok=True)
                path.with_suffix(".json").unlink(missing_ok=True)
            kept_bytes -= size
            deleted_bytes += size
            deleted_count += 1
    quota_unmet = max(0, kept_bytes - max_bytes) if max_bytes > 0 else 0

    return {
        "scanned": len(assets),
        "deleted": deleted_count,
        "deleted_bytes": deleted_bytes,
        "kept": len(assets) - deleted_count,
        "kept_bytes": kept_bytes,
        "quota_unmet_bytes": quota_unmet,
        "dry_run": dry_run,
        "evidence_complete": evidence_complete,
    }


def run_generated_media_gc(
    config: Config,
    *,
    dry_run: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    media_root = get_media_dir()
    referenced, evidence_complete = collect_referenced_media_evidence(
        get_webui_dir(), media_root
    )
    return garbage_collect_generated_media(
        artifact_root=media_root / "generated",
        referenced=referenced,
        retention_days=config.tools.generated_media_retention_days,
        orphan_grace_days=config.tools.generated_media_orphan_grace_days,
        max_bytes=config.tools.generated_media_max_bytes,
        now=now,
        dry_run=dry_run,
        evidence_complete=evidence_complete,
    )


def iter_generated_media_files(artifact_root: Path) -> Iterable[Path]:
    """Expose generated media files for diagnostics without including sidecars."""
    if not artifact_root.is_dir():
        return
    for path in artifact_root.rglob("*"):
        if path.is_file() and path.suffix.lower() != ".json":
            yield path
