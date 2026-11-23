"""Attachment ingress policy shared by every transport adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AttachmentRejection = Literal[
    "malformed",
    "too_many_images",
    "too_many_videos",
    "too_many_attachments",
    "total_size",
    "mime",
    "size",
    "decode",
    "storage",
    "expired",
    "foreign",
]

_IMAGE_MIME_ALLOWED: frozenset[str] = frozenset({
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
})

_VIDEO_MIME_ALLOWED: frozenset[str] = frozenset({
    "video/mp4",
    "video/webm",
    "video/quicktime",
})

_DOCUMENT_MIME_ALLOWED: frozenset[str] = frozenset({
    "application/json",
    "application/pdf",
    "application/toml",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/x-yaml",
    "application/xhtml+xml",
    "application/xml",
    "application/yaml",
    "text/csv",
    "text/html",
    "text/markdown",
    "text/plain",
    "text/xml",
    "text/yaml",
})

UPLOAD_MIME_ALLOWED: frozenset[str] = (
    _IMAGE_MIME_ALLOWED | _VIDEO_MIME_ALLOWED | _DOCUMENT_MIME_ALLOWED
)

IMAGE_MIME_ALLOWED = _IMAGE_MIME_ALLOWED
VIDEO_MIME_ALLOWED = _VIDEO_MIME_ALLOWED
DOCUMENT_MIME_ALLOWED = _DOCUMENT_MIME_ALLOWED

_AUDIO_MIME_ALLOWED: frozenset[str] = frozenset({
    "audio/mpeg",
    "audio/mp4",
    "audio/ogg",
    "audio/wav",
    "audio/webm",
    "audio/x-m4a",
    "audio/mpga",
})
AUDIO_MIME_ALLOWED = _AUDIO_MIME_ALLOWED


@dataclass(frozen=True)
class AttachmentIngressLimits:
    max_count: int = 4
    max_file_bytes: int = 6 * 1024 * 1024
    max_total_bytes: int = 24 * 1024 * 1024


@dataclass(frozen=True)
class AttachmentPolicy:
    """Per-source attachment limits and MIME ceiling."""

    limits: AttachmentIngressLimits = field(default_factory=AttachmentIngressLimits)
    allowed_mime: frozenset[str] = UPLOAD_MIME_ALLOWED
    max_videos: int = 1
    max_video_bytes: int = 20 * 1024 * 1024


DEFAULT_ATTACHMENT_POLICY = AttachmentPolicy()

CHANNEL_ATTACHMENT_POLICY = AttachmentPolicy(
    limits=AttachmentIngressLimits(
        max_count=16,
        max_file_bytes=24 * 1024 * 1024,
        max_total_bytes=96 * 1024 * 1024,
    ),
    allowed_mime=(
        UPLOAD_MIME_ALLOWED
        | AUDIO_MIME_ALLOWED
        | frozenset({"application/octet-stream"})
    ),
    max_videos=4,
    max_video_bytes=24 * 1024 * 1024,
)
