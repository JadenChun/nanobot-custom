"""Server-owned attachment records and ingress values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from nanobot.attachments.policy import AttachmentRejection

ATTACHMENT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class AttachmentInput:
    """One raw attachment offered by a transport adapter."""

    data_url: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class StagedAttachment:
    """A validated attachment waiting for turn admission."""

    token: str
    temp_path: Path
    mime: str
    name: str | None
    bytes: int
    sha256: str
    source: str


@dataclass(frozen=True, slots=True)
class AttachmentRecord:
    """The persisted identity of one inbound attachment."""

    id: str
    sha256: str
    mime: str
    bytes: int
    name: str
    source: str
    created_at: float
    storage_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": ATTACHMENT_SCHEMA_VERSION,
            "id": self.id,
            "sha256": self.sha256,
            "mime": self.mime,
            "bytes": self.bytes,
            "name": self.name,
            "source": self.source,
            "created_at": self.created_at,
        }
        if self.storage_key:
            payload["storage_key"] = self.storage_key
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AttachmentRecord | None:
        """Parse a persisted record; ignore unknown fields for forward compatibility."""
        try:
            attachment_id = value["id"]
            sha256 = value["sha256"]
            mime = value["mime"]
            size = value["bytes"]
            name = value["name"]
            source = value["source"]
            created_at = value["created_at"]
            storage_key = value.get("storage_key", "")
        except KeyError:
            return None
        if not all(isinstance(item, str) for item in (
            attachment_id, sha256, mime, name, source, storage_key,
        )):
            return None
        if not isinstance(size, int) or size < 0:
            return None
        if not isinstance(created_at, (int, float)):
            return None
        if not attachment_id or not name:
            return None
        if not sha256 and not storage_key:
            return None
        return cls(
            id=attachment_id,
            sha256=sha256,
            mime=mime,
            bytes=size,
            name=name,
            source=source,
            created_at=float(created_at),
            storage_key=storage_key,
        )


@dataclass(frozen=True, slots=True)
class AttachmentBatchResult:
    """Staged items or a policy rejection for one inbound batch."""

    staged: tuple[StagedAttachment, ...] = ()
    rejection: AttachmentRejection | None = None
