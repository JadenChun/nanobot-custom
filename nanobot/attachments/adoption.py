"""Adopt adapter-acquired media into the canonical attachment store."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanobot.attachments.ingress import AttachmentIngress, AttachmentIngressError
from nanobot.attachments.models import AttachmentRecord
from nanobot.attachments.policy import CHANNEL_ATTACHMENT_POLICY
from nanobot.attachments.store import AttachmentStore


@dataclass(frozen=True, slots=True)
class AdoptionResult:
    records: tuple[AttachmentRecord, ...]
    paths: tuple[str, ...]
    replacements: tuple[tuple[str, str], ...]


def adopt_media_paths(
    media: Sequence[object],
    *,
    source: str,
    logger: Any = None,
    existing: tuple[AttachmentRecord, ...] = (),
) -> AdoptionResult:
    """Copy adapter media under the media root into the canonical store.

    Paths outside the media root (workspace/test attachments) are preserved.
    The original adapter file is intentionally left in place for its own
    lifecycle. Records already bound to the turn are reused by canonical
    identity so an upload-first ID and its canonical path do not become two
    records, while distinct acquisition paths remain distinct attachments.
    """
    from nanobot.config.paths import get_media_dir

    media_root = get_media_dir()
    store = AttachmentStore(
        media_root / "attachments",
        media_root=media_root,
        logger=logger,
    )
    ingress = AttachmentIngress(
        store,
        policy=CHANNEL_ATTACHMENT_POLICY,
        logger=logger,
    )
    by_identity: dict[tuple[str, str], AttachmentRecord] = {
        (record.sha256, record.storage_key or str(store.path_for(record))): record
        for record in existing
        if record.sha256
    }
    records: list[AttachmentRecord] = []
    paths: list[str] = []
    replacements: list[tuple[str, str]] = []
    for raw_path in media:
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = Path(raw_path).expanduser()
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            paths.append(str(path))
            continue
        record = None
        if resolved.is_relative_to(store.root):
            record = store.record_for_existing_path(resolved, source=source)
        elif not resolved.is_relative_to(media_root):
            record = store.record_for_existing_path(
                resolved,
                source=source,
                allow_outside=True,
            )
        if record is None:
            try:
                staged = ingress.accept_path(resolved, source=source)
                committed, rejection = ingress.commit_batch([staged])
                if rejection is None and committed:
                    record = committed[0]
            except (AttachmentIngressError, OSError):
                record = None
        if record is None:
            paths.append(str(resolved))
            continue
        canonical = store.resolve(record)
        if canonical is None:
            paths.append(str(resolved))
            continue
        identity = (record.sha256, record.storage_key or str(canonical))
        existing_record = by_identity.get(identity)
        if existing_record is None:
            by_identity[identity] = record
            records.append(record)
        else:
            record = existing_record
            canonical = store.resolve(record) or canonical
        canonical_text = str(canonical)
        paths.append(canonical_text)
        if str(resolved) != canonical_text:
            replacements.append((str(resolved), canonical_text))
    return AdoptionResult(
        records=tuple(records),
        paths=tuple(paths),
        replacements=tuple(replacements),
    )
