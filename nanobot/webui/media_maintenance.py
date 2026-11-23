"""Generated-media diagnostics and maintenance owned by AttachmentStore."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.attachments._inventory import References, inspect_webui, walk_files
from nanobot.attachments._media import MediaRetentionError, generation
from nanobot.attachments.store import AttachmentStore
from nanobot.config.paths import inspect_media_dir
from nanobot.config.schema import Config


def collect_referenced_media_evidence(webui_dir: Path, media_root: Path) -> tuple[set[Path], bool]:
    """Nonrepairing diagnostics only; caller sets never grant deletion authority."""
    references = References(media_root.resolve(strict=False))
    try:
        inspect_webui(webui_dir, references)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        return references.paths, False
    return references.paths, True


def collect_referenced_media_paths(webui_dir: Path, media_root: Path) -> set[Path]:
    return collect_referenced_media_evidence(webui_dir, media_root)[0]


def garbage_collect_generated_media(
    *, artifact_root: Path, referenced: set[Path] | None = None,
    retention_days: int = 0, orphan_grace_days: int = 7, max_bytes: int = 0,
    now: datetime | None = None, dry_run: bool = False, evidence_complete: bool = True,
) -> dict[str, Any]:
    """All destructive branches require one complete store-owned inventory.

    Legacy caller evidence can add protection or refuse work, never authorize it.
    A supported declared state still reclaims genuine orphans and enforces quota.
    """
    root = Path(artifact_root).expanduser().absolute()
    configured = inspect_media_dir()
    media_root = configured if root.is_relative_to(configured) else root.parent
    store = AttachmentStore(media_root / "attachments", media_root=media_root)
    now_ts = (now or datetime.now().astimezone()).timestamp()
    assets: list[tuple[Path, float, int, float]] = []
    reasons: list[str] = []
    failures: list[dict[str, str]] = []
    deleted: set[Path] = set()
    deleted_bytes = 0
    complete = False
    assets_complete = True
    try:
        if root.exists() or root.is_symlink():
            for path in walk_files(root):
                if path.suffix.lower() == ".json":
                    continue
                info = path.lstat()
                assets.append((path, (now_ts - info.st_mtime) / 86400, info.st_size, info.st_mtime))
    except (OSError, ValueError) as exc:
        reasons.append(f"{type(exc).__name__}: {exc}")
        assets_complete = False
    try:
        with store.custody.maintenance() as (gate, evidence):
            complete = evidence.complete and evidence_complete and not reasons
            reasons.extend(evidence.reasons)
            if not evidence_complete:
                reasons.append("caller reported incomplete evidence")
            if complete:
                # Candidate age, size and generation belong to the same
                # fenced inventory; pre-admission diagnostics are not proof.
                fresh: list[tuple[Path, float, int, float]] = []
                expected_assets = dict(evidence.assets)
                if root.exists() or root.is_symlink():
                    for path in walk_files(root):
                        if path.suffix.lower() == ".json":
                            continue
                        info = path.lstat()
                        if generation(info) != expected_assets.get(path):
                            raise MediaRetentionError("generated asset changed after inventory")
                        fresh.append((path, (now_ts - info.st_mtime) / 86400, info.st_size, info.st_mtime))
                assets = fresh
            protected = set(evidence.paths) | (referenced or set())
            protected.update(path for _token, path in evidence.pending)
            protected.update(path for path, _age, _size, _mtime in assets if path.stem in evidence.hashes)
            candidates: dict[Path, int] = {}
            if complete:
                for path, age, size, _mtime in assets:
                    if path not in protected and ((retention_days > 0 and age >= retention_days) or age >= orphan_grace_days):
                        candidates[path] = size
                kept = sum(size for path, _age, size, _mtime in assets if path not in candidates)
                if max_bytes > 0 and kept > max_bytes:
                    for path, _age, size, _mtime in sorted(assets, key=lambda row: row[3]):
                        if kept <= max_bytes:
                            break
                        if path not in protected and path not in candidates:
                            candidates[path] = size
                            kept -= size
                for intent in store.custody.records("deletions"):
                    path = Path(str(intent["path"]))
                    if path.is_relative_to(root) and path not in protected:
                        candidates.setdefault(path, 0)
                for path, size in candidates.items():
                    try:
                        gate.require(store.custody)
                        if dry_run or store.custody.delete(gate, evidence, path):
                            deleted.add(path)
                            deleted_bytes += size
                    except OSError as exc:
                        failures.append({"path": str(path), "reason": type(exc).__name__})
                        if not path.exists() and size:
                            deleted.add(path)
                            deleted_bytes += size
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        reasons.append(f"{type(exc).__name__}: {exc}")
        complete = False
    kept_bytes = sum(size for path, _age, size, _mtime in assets if path not in deleted)
    removed_count = sum(path in deleted for path, _age, _size, _mtime in assets)
    return {
        "scanned": len(assets), "deleted": removed_count, "deleted_bytes": deleted_bytes,
        "kept": len(assets) - removed_count, "kept_bytes": kept_bytes,
        "quota_unmet_bytes": max(0, kept_bytes - max_bytes) if max_bytes > 0 else 0,
        "quota_unknown": bool(max_bytes > 0 and not assets_complete),
        "dry_run": dry_run, "evidence_complete": complete, "reasons": reasons,
        "partial": bool(failures), "retry_required": bool(failures), "failed": failures,
    }


def run_generated_media_gc(config: Config, *, dry_run: bool = False, now: datetime | None = None) -> dict[str, Any]:
    return garbage_collect_generated_media(
        artifact_root=inspect_media_dir() / "generated",
        retention_days=config.tools.generated_media_retention_days,
        orphan_grace_days=config.tools.generated_media_orphan_grace_days,
        max_bytes=config.tools.generated_media_max_bytes, now=now, dry_run=dry_run,
    )


def iter_generated_media_files(artifact_root: Path) -> Iterable[Path]:
    if not artifact_root.is_dir():
        return
    for path in walk_files(artifact_root):
        if path.suffix.lower() != ".json":
            yield path
