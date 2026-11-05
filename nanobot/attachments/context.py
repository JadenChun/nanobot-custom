"""Single model-visible projection for inbound attachments."""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from nanobot.attachments.models import AttachmentRecord
from nanobot.attachments.store import AttachmentStore
from nanobot.config.paths import get_media_dir
from nanobot.utils.helpers import detect_image_mime
from nanobot.utils.media_decode import mime_extension


class AttachmentResolver(Protocol):
    """Resolve a persisted attachment record to a readable local file."""

    def resolve(self, record: AttachmentRecord) -> Path | None: ...


class StoreAttachmentResolver:
    def __init__(self, store: AttachmentStore) -> None:
        self.store = store

    def resolve(self, record: AttachmentRecord) -> Path | None:
        return self.store.resolve(record)


class FilesystemAttachmentResolver:
    """Resolve records without requiring a store instance (agent default)."""

    def __init__(
        self,
        *,
        media_root: Path | None = None,
        attachments_root: Path | None = None,
    ) -> None:
        self.media_root = (media_root or get_media_dir()).expanduser().resolve(strict=False)
        self.attachments_root = (
            attachments_root or (get_media_dir() / "attachments")
        ).expanduser().resolve(strict=False)

    def resolve(self, record: AttachmentRecord) -> Path | None:
        if record.storage_key:
            candidate = Path(record.storage_key)
            if not candidate.is_absolute():
                candidate = self.media_root / candidate
        else:
            if not record.sha256:
                return None
            candidate = (
                self.attachments_root
                / record.sha256[:2]
                / f"{record.sha256}{mime_extension(record.mime)}"
            )
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if not resolved.is_file():
            return None
        if not resolved.is_relative_to(self.media_root):
            if not (record.storage_key and Path(record.storage_key).is_absolute()):
                return None
        return resolved


class AttachmentContext:
    """Format attachments for the model without any transport-specific logic."""

    def __init__(
        self,
        resolver: AttachmentResolver | None = None,
        *,
        logger: Any = None,
    ) -> None:
        self.resolver = resolver or FilesystemAttachmentResolver()
        self.logger = logger

    @staticmethod
    def image_reference(path: str | Path) -> str:
        return f"[image: {path}]"

    @staticmethod
    def attachment_reference(path: str | Path) -> str:
        return f"[Attachment: {path}]"

    @classmethod
    def unavailable_reference(cls, *, is_image: bool, name: str) -> str:
        label = "image" if is_image else "Attachment"
        return f"[{label}: unavailable ({name})]"

    def reference_for_record(self, record: AttachmentRecord) -> str:
        is_image = record.mime.startswith("image/")
        path = self.resolver.resolve(record)
        if path is None:
            return self.unavailable_reference(is_image=is_image, name=record.name)
        return (
            self.image_reference(path)
            if is_image
            else self.attachment_reference(path)
        )

    def append_references(self, text: str, references: Iterable[str]) -> str:
        missing: list[str] = []
        for reference in references:
            if not reference or reference in text or reference in missing:
                continue
            missing.append(reference)
        if not missing:
            return text
        suffix = "\n".join(missing)
        return f"{text}\n\n{suffix}" if text else suffix

    def user_content_from_paths(
        self,
        text: str,
        image_paths: Sequence[str] | None,
    ) -> str | list[dict[str, Any]]:
        """Compatibility path projection used until sessions persist records."""
        if not image_paths:
            return text
        image_blocks: list[dict[str, Any]] = []
        image_refs: list[str] = []
        for path in image_paths:
            candidate = Path(path)
            if not candidate.is_file():
                continue
            raw = candidate.read_bytes()
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(raw).decode()
            image_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
                "_meta": {"path": str(candidate)},
            })
            image_refs.append(self.image_reference(candidate))
        if not image_blocks:
            return text
        content: list[dict[str, Any]] = [
            *image_blocks,
            {"type": "text", "text": text},
        ]
        missing = [ref for ref in image_refs if ref not in text]
        if missing:
            content.append({"type": "text", "text": "\n".join(missing)})
        return content

    def user_content(
        self,
        text: str,
        records: Sequence[AttachmentRecord],
    ) -> str | list[dict[str, Any]]:
        """Build model content for records from the canonical store."""
        if not records:
            return text
        image_blocks: list[dict[str, Any]] = []
        references: list[str] = []
        seen: set[str] = set()
        for record in records:
            if record.id in seen:
                continue
            seen.add(record.id)
            path = self.resolver.resolve(record)
            if path is None:
                references.append(
                    self.unavailable_reference(
                        is_image=record.mime.startswith("image/"),
                        name=record.name,
                    )
                )
                continue
            if record.mime.startswith("image/"):
                raw = path.read_bytes()
                mime = detect_image_mime(raw) or record.mime
                b64 = base64.b64encode(raw).decode()
                image_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                    "_meta": {"path": str(path)},
                })
                references.append(self.image_reference(path))
            else:
                references.append(self.attachment_reference(path))
        if not image_blocks:
            return self.append_references(text, references)
        content: list[dict[str, Any]] = [
            *image_blocks,
            {"type": "text", "text": text},
        ]
        missing = [ref for ref in references if ref not in text]
        if missing:
            content.append({"type": "text", "text": "\n".join(missing)})
        return content


def content_with_media_breadcrumbs(
    role: str | None,
    content: Any,
    media: Any,
) -> Any:
    """Append persisted user-media breadcrumbs to plain-text content."""
    if role != "user" or not isinstance(content, str) or not isinstance(media, list):
        return content
    context = AttachmentContext()
    breadcrumbs: list[str] = []
    for path in cast(list[object], media):
        if not isinstance(path, str) or not path:
            continue
        reference = context.image_reference(path)
        if reference in content or reference in breadcrumbs:
            continue
        breadcrumbs.append(reference)
    if not breadcrumbs:
        return content
    suffix = "\n".join(breadcrumbs)
    return f"{content}\n{suffix}" if content else suffix


def media_paths_for_message(
    message: Mapping[str, Any],
    *,
    resolver: AttachmentResolver | None = None,
) -> list[str]:
    """Return resolved paths from canonical attachment records only."""
    paths: list[str] = []
    records = message.get("attachments")
    if isinstance(records, list):
        context = AttachmentContext(resolver)
        for raw in cast(list[object], records):
            if not isinstance(raw, Mapping):
                continue
            record = AttachmentRecord.from_dict(cast(Mapping[str, Any], raw))
            if record is None:
                continue
            resolved = context.resolver.resolve(record)
            if resolved is not None:
                candidate = str(resolved)
            elif record.storage_key:
                candidate = record.storage_key
            else:
                continue
            if candidate not in paths:
                paths.append(candidate)
    return paths
