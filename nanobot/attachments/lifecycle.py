"""Reference-aware retention for the content-addressed attachment store."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from nanobot.attachments.store import AttachmentStore

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _record_hashes(value: object) -> Iterable[str]:
    if not isinstance(value, list):
        return ()
    hashes: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        sha256 = item.get("sha256")
        if isinstance(sha256, str) and _HASH_RE.match(sha256):
            hashes.append(sha256)
    return hashes


def _media_hashes(value: object) -> Iterable[str]:
    if not isinstance(value, list):
        return ()
    hashes: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        stem = Path(item).name.split(".", 1)[0]
        if _HASH_RE.match(stem):
            hashes.append(stem)
    return hashes


def reference_evidence(
    session_files: Iterable[Path],
) -> tuple[set[str], bool]:
    """Collect hashes and report whether every canonical source was readable."""
    referenced: set[str] = set()
    complete = True
    for path in session_files:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(event, dict):
                        continue
                    referenced.update(_record_hashes(event.get("attachments")))
                    referenced.update(_media_hashes(event.get("media")))
        except OSError:
            complete = False
            continue
    return referenced, complete


def referenced_hashes(session_files: Iterable[Path]) -> set[str]:
    """Collect hashes referenced by readable canonical session events."""
    return reference_evidence(session_files)[0]


def prune_unreferenced(
    store: AttachmentStore,
    sessions_dir: Path,
    *,
    grace_seconds: float = 7 * 24 * 60 * 60,
    dry_run: bool = False,
) -> list[Path]:
    """Delete stored blobs no live session references.

    The grace period protects a blob between staging and the next session save.
    """
    session_files = list(Path(sessions_dir).rglob("*.jsonl"))
    referenced, evidence_complete = reference_evidence(session_files)
    if not evidence_complete:
        # A missing or unreadable session can hide a live reference.
        return []
    now = time.time()
    removed: list[Path] = []
    for path in store.root.rglob("*"):
        if not path.is_file() or path.suffix == ".json":
            continue
        sha256 = path.stem
        if not _HASH_RE.match(sha256) or sha256 in referenced:
            continue
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        if now - modified < grace_seconds:
            continue
        if not dry_run:
            try:
                path.unlink()
            except OSError:
                continue
        removed.append(path)
    if not dry_run:
        for directory in sorted(store.root.rglob("*"), reverse=True):
            if directory.is_dir():
                try:
                    directory.rmdir()
                except OSError:
                    pass
    return removed


@dataclass(frozen=True)
class MigrationReport:
    files_scanned: int = 0
    files_changed: int = 0
    events_migrated: int = 0
    records_created: int = 0
    events_skipped: int = 0


def _migrate_event(
    event: dict[str, object],
    *,
    store: AttachmentStore,
    ingress: object,
    source: str,
) -> tuple[dict[str, object], int] | None:
    media = event.get("media")
    if not isinstance(media, list) or not media or event.get("attachments"):
        return None
    from nanobot.attachments.ingress import AttachmentIngress, AttachmentIngressError
    from nanobot.config.paths import get_media_dir

    if not isinstance(ingress, AttachmentIngress):
        return None
    media_root = get_media_dir()
    records: list[dict[str, object]] = []
    for raw_path in media:
        if not isinstance(raw_path, str) or not raw_path:
            return None
        try:
            resolved = Path(raw_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if not resolved.is_relative_to(media_root):
            return None
        record = None
        if resolved.is_relative_to(store.root):
            record = store.record_for_existing_path(resolved, source=source)
        if record is None:
            try:
                staged = ingress.accept_path(resolved, source=source)
                committed, rejection = ingress.commit_batch([staged])
                if rejection is None and committed:
                    record = committed[0]
            except (AttachmentIngressError, OSError):
                record = None
        if record is None:
            return None
        records.append(record.to_dict())
    migrated = dict(event)
    migrated.pop("media", None)
    migrated["attachments"] = records
    return migrated, len(records)


def migrate_legacy_media(
    sessions_dir: Path,
    store: AttachmentStore,
    *,
    dry_run: bool = False,
) -> MigrationReport:
    """Rewrite legacy ``media`` events into canonical ``attachments`` records."""
    from nanobot.attachments.ingress import AttachmentIngress
    from nanobot.attachments.policy import CHANNEL_ATTACHMENT_POLICY

    ingress = AttachmentIngress(store, policy=CHANNEL_ATTACHMENT_POLICY)
    files_scanned = 0
    files_changed = 0
    events_migrated = 0
    records_created = 0
    events_skipped = 0
    for session_file in sorted(Path(sessions_dir).rglob("*.jsonl")):
        files_scanned += 1
        try:
            lines = session_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        changed = False
        output: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                output.append(line)
                continue
            try:
                event = json.loads(stripped)
            except (TypeError, ValueError):
                output.append(line)
                continue
            if not isinstance(event, dict) or event.get("role") != "user":
                output.append(line)
                continue
            result = _migrate_event(
                event,
                store=store,
                ingress=ingress,
                source="legacy",
            )
            if result is None:
                if isinstance(event.get("media"), list) and event.get("media"):
                    events_skipped += 1
                output.append(line)
                continue
            migrated, created = result
            output.append(json.dumps(migrated, ensure_ascii=False))
            changed = True
            events_migrated += 1
            records_created += created
        if changed:
            files_changed += 1
            if not dry_run:
                backup_root = Path(sessions_dir).parent / "attachment-migration-backups"
                relative = session_file.relative_to(sessions_dir)
                backup = backup_root / relative.with_name(
                    f"{relative.name}.bak-{int(time.time())}"
                )
                backup.parent.mkdir(parents=True, exist_ok=True)
                backup.write_bytes(session_file.read_bytes())
                temp = session_file.with_name(f".{session_file.name}.{time.time_ns()}.tmp")
                try:
                    temp.write_text("\n".join(output) + "\n", encoding="utf-8")
                    temp.chmod(0o600)
                    os.replace(temp, session_file)
                finally:
                    temp.unlink(missing_ok=True)
    return MigrationReport(
        files_scanned=files_scanned,
        files_changed=files_changed,
        events_migrated=events_migrated,
        records_created=records_created,
        events_skipped=events_skipped,
    )
