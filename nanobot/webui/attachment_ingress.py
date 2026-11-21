"""Compatibility wrapper for the canonical attachment ingress.

New code should import from :mod:`nanobot.attachments`. This module remains for
existing WebUI tests and callers until the W6 removal gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.attachments.ingress import AttachmentIngress, extract_data_url_mime
from nanobot.attachments.models import AttachmentRecord
from nanobot.attachments.policy import (
    AttachmentIngressLimits,
    AttachmentPolicy,
    AttachmentRejection,
)
from nanobot.attachments.store import AttachmentStore
from nanobot.webui.ingress_policy import DEFAULT_WEBUI_INGRESS_POLICY

AttachmentIngressResult = tuple[list[str], AttachmentRejection | None]

__all__ = [
    "AttachmentIngressResult",
    "extract_data_url_mime",
    "store_inbound_attachments",
    "store_inbound_records",
]


def store_inbound_attachments(
    media: list[Any],
    *,
    media_dir: Path,
    logger: Any,
    limits: AttachmentIngressLimits = DEFAULT_WEBUI_INGRESS_POLICY.attachments,
    store_root: Path | None = None,
    media_root: Path | None = None,
    source: str = "websocket",
) -> AttachmentIngressResult:
    """Validate and persist one inbound batch through the canonical service."""
    records, rejection = store_inbound_records(
        media,
        media_dir=media_dir,
        logger=logger,
        limits=limits,
        store_root=store_root,
        media_root=media_root,
        source=source,
    )
    if rejection is not None:
        return [], rejection
    store = AttachmentStore(
        Path(store_root or media_dir),
        media_root=Path(media_root or media_dir),
        logger=logger,
    )
    return [str(store.path_for(record)) for record in records], None


def store_inbound_records(
    media: list[Any],
    *,
    media_dir: Path,
    logger: Any,
    limits: AttachmentIngressLimits = DEFAULT_WEBUI_INGRESS_POLICY.attachments,
    store_root: Path | None = None,
    media_root: Path | None = None,
    source: str = "websocket",
) -> tuple[list[AttachmentRecord], AttachmentRejection | None]:
    """Validate and persist one inbound batch, returning canonical records."""
    root = Path(store_root or media_dir)
    base = Path(media_root or media_dir)
    store = AttachmentStore(root, media_root=base, logger=logger)
    ingress = AttachmentIngress(
        store,
        policy=AttachmentPolicy(limits=limits),
        logger=logger,
    )
    result = ingress.accept_batch(media, source=source)
    if result.rejection is not None:
        return [], result.rejection
    records, rejection = ingress.commit_batch(result.staged)
    return list(records), rejection
