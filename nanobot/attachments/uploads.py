"""Short-lived upload records for upload-first attachment messages."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from nanobot.attachments.models import AttachmentRecord

DEFAULT_UPLOAD_TTL_SECONDS = 60 * 60


@dataclass(frozen=True, slots=True)
class PendingUpload:
    record: AttachmentRecord
    installation_id: str
    session_key: str | None
    created_at: float
    expires_at: float
    source: str


class UploadRegistry:
    """Process-local upload IDs awaiting a message binding.

    Uploads are short-lived and intentionally in-memory: a restart invalidates
    unbound uploads, and the client may retry. Committed blobs remain in the
    store until lifecycle cleanup.
    """

    def __init__(self, *, ttl_seconds: int = DEFAULT_UPLOAD_TTL_SECONDS) -> None:
        self._ttl = max(1, ttl_seconds)
        self._lock = threading.Lock()
        self._uploads: dict[str, PendingUpload] = {}

    def add(
        self,
        record: AttachmentRecord,
        *,
        installation_id: str,
        session_key: str | None,
        source: str,
    ) -> PendingUpload:
        now = time.time()
        upload = PendingUpload(
            record=record,
            installation_id=installation_id,
            session_key=session_key,
            created_at=now,
            expires_at=now + self._ttl,
            source=source,
        )
        with self._lock:
            self._sweep_locked(now)
            self._uploads[record.id] = upload
        return upload

    def bind(
        self,
        attachment_id: str,
        *,
        installation_id: str,
        session_key: str,
    ) -> AttachmentRecord | None:
        now = time.time()
        with self._lock:
            self._sweep_locked(now)
            upload = self._uploads.get(attachment_id)
            if upload is None:
                return None
            if upload.installation_id != installation_id:
                return None
            if upload.session_key is not None and upload.session_key != session_key:
                return None
            if upload.session_key is None:
                self._uploads[attachment_id] = PendingUpload(
                    record=upload.record,
                    installation_id=upload.installation_id,
                    session_key=session_key,
                    created_at=upload.created_at,
                    expires_at=upload.expires_at,
                    source=upload.source,
                )
            return upload.record

    def get(self, attachment_id: str) -> PendingUpload | None:
        with self._lock:
            self._sweep_locked(time.time())
            return self._uploads.get(attachment_id)

    def clear(self) -> None:
        with self._lock:
            self._uploads.clear()

    def sweep(self) -> None:
        with self._lock:
            self._sweep_locked(time.time())

    def _sweep_locked(self, now: float) -> None:
        expired = [
            attachment_id
            for attachment_id, upload in self._uploads.items()
            if upload.expires_at <= now
        ]
        for attachment_id in expired:
            self._uploads.pop(attachment_id, None)
