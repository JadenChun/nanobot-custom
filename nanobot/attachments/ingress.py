"""One validation and staging implementation for inbound attachments."""

from __future__ import annotations

import mimetypes
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from nanobot.attachments.models import (
    AttachmentBatchResult,
    AttachmentInput,
    AttachmentRecord,
    StagedAttachment,
)
from nanobot.attachments.policy import (
    DEFAULT_ATTACHMENT_POLICY,
    IMAGE_MIME_ALLOWED,
    VIDEO_MIME_ALLOWED,
    AttachmentIngressLimits,
    AttachmentPolicy,
    AttachmentRejection,
)
from nanobot.attachments.store import AttachmentStore
from nanobot.utils.helpers import detect_image_mime
from nanobot.utils.media_decode import FileSizeExceeded, decode_data_url

_DATA_URL_MIME_RE = re.compile(r"^data:([^;,]+)(?:;[^,]*)*;base64,", re.DOTALL)


class AttachmentIngressError(Exception):
    """Raised when a non-data-URL attachment violates the supplied policy."""

    def __init__(self, reason: AttachmentRejection) -> None:
        super().__init__(reason)
        self.reason = reason


def extract_data_url_mime(url: Any) -> str | None:
    """Return the normalized MIME from a base64 data URL, else ``None``."""
    if not isinstance(url, str):
        return None
    match = _DATA_URL_MIME_RE.match(url)
    if match is None:
        return None
    return match.group(1).strip().lower() or None


def _coerce_input(item: Any) -> AttachmentInput | None:
    if isinstance(item, AttachmentInput):
        return item
    if isinstance(item, dict):
        value = cast(dict[str, Any], item)
        data_url = value.get("data_url")
        if not isinstance(data_url, str) or not data_url:
            return None
        name = value.get("name")
        return AttachmentInput(data_url=data_url, name=name if isinstance(name, str) else None)
    return None


class AttachmentIngress:
    """Validate a batch, stage it, then commit it after turn admission."""

    def __init__(
        self,
        store: AttachmentStore,
        *,
        policy: AttachmentPolicy = DEFAULT_ATTACHMENT_POLICY,
        logger: Any = None,
    ) -> None:
        self.store = store
        self.policy = policy
        self.logger = logger

    def accept_bytes(
        self,
        data: bytes,
        *,
        name: str | None,
        mime: str | None,
        source: str,
        policy: AttachmentPolicy | None = None,
    ) -> StagedAttachment:
        active_policy = policy or self.policy
        resolved_mime = (
            mime
            or detect_image_mime(data[:16])
            or mimetypes.guess_type(name or "")[0]
            or "application/octet-stream"
        )
        if resolved_mime not in active_policy.allowed_mime:
            raise AttachmentIngressError("mime")
        limit = (
            active_policy.max_video_bytes
            if resolved_mime in VIDEO_MIME_ALLOWED
            else active_policy.limits.max_file_bytes
        )
        if len(data) > limit:
            raise AttachmentIngressError("size")
        try:
            return self.store.stage_bytes(
                data,
                mime=resolved_mime,
                name=name,
                source=source,
            )
        except (OSError, ValueError) as exc:
            raise AttachmentIngressError("storage") from exc

    def accept_path(
        self,
        path: Path,
        *,
        name: str | None = None,
        mime: str | None = None,
        source: str,
        policy: AttachmentPolicy | None = None,
    ) -> StagedAttachment:
        active_policy = policy or self.policy
        resolved = Path(path).expanduser().resolve(strict=True)
        with resolved.open("rb") as handle:
            head = handle.read(16)
        resolved_mime = (
            mime
            or detect_image_mime(head)
            or mimetypes.guess_type(name or resolved.name)[0]
            or "application/octet-stream"
        )
        if resolved_mime not in active_policy.allowed_mime:
            raise AttachmentIngressError("mime")
        limit = (
            active_policy.max_video_bytes
            if resolved_mime in VIDEO_MIME_ALLOWED
            else active_policy.limits.max_file_bytes
        )
        try:
            return self.store.stage_path(
                resolved,
                mime=resolved_mime,
                name=name or resolved.name,
                source=source,
                max_bytes=limit,
            )
        except FileSizeExceeded as exc:
            raise AttachmentIngressError("size") from exc
        except OSError as exc:
            raise AttachmentIngressError("storage") from exc

    def accept_batch(
        self,
        media: Sequence[Any],
        *,
        source: str,
        policy: AttachmentPolicy | None = None,
    ) -> AttachmentBatchResult:
        active_policy = policy or self.policy
        limits = active_policy.limits
        staged: list[StagedAttachment] = []
        image_count = 0
        video_count = 0
        document_count = 0
        total_bytes = 0

        def reject(reason: AttachmentRejection) -> AttachmentBatchResult:
            self.discard_batch(staged)
            return AttachmentBatchResult(rejection=reason)

        for item in media:
            attachment = _coerce_input(item)
            if attachment is None:
                return reject("malformed")
            mime = extract_data_url_mime(attachment.data_url)
            if mime is None:
                return reject("decode")
            if mime not in active_policy.allowed_mime:
                return reject("mime")
            if mime in IMAGE_MIME_ALLOWED:
                image_count += 1
            elif mime in VIDEO_MIME_ALLOWED:
                video_count += 1
            else:
                # Documents and any extra policy-allowed MIME (for example the
                # API's generic application/octet-stream multipart uploads).
                document_count += 1
            if image_count > limits.max_count:
                return reject("too_many_images")
            if video_count > active_policy.max_videos:
                return reject("too_many_videos")
            if image_count + document_count > limits.max_count:
                return reject("too_many_attachments")

            decoded = decode_data_url(attachment.data_url)
            if decoded is None:
                return reject("decode")
            _, raw = decoded
            max_bytes = (
                active_policy.max_video_bytes
                if mime in VIDEO_MIME_ALLOWED
                else limits.max_file_bytes
            )
            if len(raw) > max_bytes:
                return reject("size")
            try:
                staged_item = self.store.stage_bytes(
                    raw,
                    mime=mime,
                    name=attachment.name,
                    source=source,
                )
            except (OSError, ValueError):
                return reject("storage")
            staged.append(staged_item)
            if mime not in VIDEO_MIME_ALLOWED:
                total_bytes += len(raw)
                if total_bytes > limits.max_total_bytes:
                    return reject("total_size")

        return AttachmentBatchResult(staged=tuple(staged))

    def commit_batch(
        self,
        staged: Sequence[StagedAttachment],
    ) -> tuple[tuple[AttachmentRecord, ...], AttachmentRejection | None]:
        records: list[AttachmentRecord] = []
        for item in staged:
            try:
                records.append(self.store.commit(item))
            except OSError as exc:
                if self.logger is not None:
                    self.logger.warning("attachment commit failed: {}", exc)
                self.discard_batch(staged[len(records):])
                return tuple(records), "storage"
        return tuple(records), None

    def discard_batch(self, staged: Sequence[StagedAttachment]) -> None:
        for item in staged:
            self.store.discard(item)
        self.store.cleanup_staging()

    def accept_and_commit(
        self,
        media: Sequence[Any],
        *,
        source: str,
        policy: AttachmentPolicy | None = None,
    ) -> tuple[list[str], AttachmentRejection | None]:
        """Compatibility convenience: accept, commit, and return paths."""
        result = self.accept_batch(media, source=source, policy=policy)
        if result.rejection is not None:
            return [], result.rejection
        records, rejection = self.commit_batch(result.staged)
        if rejection is not None:
            return [], rejection
        return [str(self.store.path_for(record)) for record in records], None


def ingress_for_limits(
    store: AttachmentStore,
    limits: AttachmentIngressLimits,
    *,
    logger: Any = None,
) -> AttachmentIngress:
    return AttachmentIngress(
        store,
        policy=AttachmentPolicy(limits=limits),
        logger=logger,
    )
