"""Artifact persistence helpers for generated media."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import mimetypes
import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

from nanobot.attachments._media import atomic_write_bytes
from nanobot.attachments.store import AttachmentStore
from nanobot.config.paths import get_media_dir
from nanobot.utils.helpers import detect_image_mime, ensure_dir

_DATA_IMAGE_RE = re.compile(r"^data:(image/[A-Za-z0-9.+-]+);base64,(.*)$", re.DOTALL)
_MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

class ArtifactError(ValueError):
    """Raised when an artifact cannot be safely decoded or stored."""


def decode_image_data_url(data_url: str) -> tuple[bytes, str]:
    """Decode a base64 image data URL and return ``(bytes, mime)``."""
    match = _DATA_IMAGE_RE.match(data_url.strip())
    if match is None:
        raise ArtifactError("expected a base64 image data URL")

    declared_mime, encoded = match.groups()
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ArtifactError("invalid base64 image payload") from exc

    detected_mime = detect_image_mime(raw)
    if detected_mime is None:
        raise ArtifactError("unsupported or unrecognized image data")
    if declared_mime != detected_mime:
        declared_mime = detected_mime
    return raw, declared_mime


def _safe_relative_dir(save_dir: str) -> Path:
    normalized = save_dir.replace("\\", "/").strip("/")
    if not normalized:
        raise ArtifactError("save_dir must not be empty")
    rel = PurePosixPath(normalized)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ArtifactError("save_dir must be a safe relative path")
    return Path(*rel.parts)


def _artifact_root(save_dir: str, media_root: Path | None = None) -> Path:
    root_base = (media_root if media_root is not None else get_media_dir()).resolve()
    root = (root_base / _safe_relative_dir(save_dir)).resolve()
    try:
        root.relative_to(root_base)
    except ValueError as exc:
        raise ArtifactError("artifact directory escapes media root") from exc
    return root


def store_generated_image_artifact(
    data_url: str,
    *,
    prompt: str,
    model: str,
    source_images: list[str] | None = None,
    save_dir: str = "generated",
    provider: str = "openrouter",
    created_at: datetime | None = None,
    media_root: Path | None = None,
) -> dict[str, Any]:
    """Persist a generated image and sidecar metadata under the media root."""
    raw, mime = decode_image_data_url(data_url)
    ext = _MIME_EXTENSIONS.get(mime)
    if ext is None:
        raise ArtifactError(f"unsupported image MIME type: {mime}")

    now = created_at or datetime.now().astimezone()
    day_dir = ensure_dir(_artifact_root(save_dir, media_root) / now.strftime("%Y-%m-%d"))
    artifact_id = f"img_{uuid.uuid4().hex[:12]}"
    image_path = day_dir / f"{artifact_id}{ext}"
    metadata_path = day_dir / f"{artifact_id}.json"

    metadata: dict[str, Any] = {
        "id": artifact_id,
        "kind": "image",
        "name": image_path.name,
        "path": str(image_path),
        "mime": mime,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "prompt": prompt,
        "model": model,
        "provider": provider,
        "source_images": list(source_images or []),
        "created_at": now.isoformat(),
    }
    base = (media_root if media_root is not None else get_media_dir()).resolve()
    store = AttachmentStore(base / "attachments", media_root=base)
    store.custody.declare("generated", _artifact_root(save_dir, base))
    with store.custody.fence(create=True) as gate:
        store.custody.pin(gate, publication_id=artifact_id, path=image_path, sha256=metadata["sha256"])
        atomic_write_bytes(image_path, raw)
        atomic_write_bytes(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2).encode())
    return metadata


def store_generated_file_artifact(
    source_path: Path,
    *,
    kind: str,
    prompt: str,
    model: str,
    provider: str,
    allowed_roots: Sequence[Path],
    save_dir: str = "generated",
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Copy a trusted local MCP output file into the managed media library."""
    if not allowed_roots:
        raise ArtifactError("no trusted media output roots are configured")
    try:
        source = source_path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ArtifactError("generated file is unavailable") from exc
    if not source.is_file():
        raise ArtifactError("generated file is not a regular file")
    roots = [root.expanduser().resolve() for root in allowed_roots]
    if not any(source.is_relative_to(root) for root in roots):
        raise ArtifactError("generated file is outside the trusted media output roots")

    mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    extension = source.suffix.lower()
    if not extension:
        extension = mimetypes.guess_extension(mime) or ".bin"
    now = created_at or datetime.now().astimezone()
    subdir = "videos" if kind == "video" else "files"
    day_dir = ensure_dir(_artifact_root(save_dir) / subdir / now.strftime("%Y-%m-%d"))
    artifact_id = f"{'vid' if kind == 'video' else 'file'}_{uuid.uuid4().hex[:12]}"
    target = day_dir / f"{artifact_id}{extension}"
    metadata_path = day_dir / f"{artifact_id}.json"

    base = get_media_dir().resolve()
    store = AttachmentStore(base / "attachments", media_root=base)
    staged = store.stage_path(source, mime=mime, name=source.name, source=provider)
    metadata: dict[str, Any] = {
        "id": artifact_id,
        "kind": kind,
        "name": target.name,
        "path": str(target),
        "mime": mime,
        "bytes": staged.bytes,
        "sha256": staged.sha256,
        "prompt": prompt,
        "model": model,
        "provider": provider,
        "source_images": [],
        "created_at": now.isoformat(),
    }
    store.custody.declare("generated", _artifact_root(save_dir, base))
    with store.custody.fence(create=True) as gate:
        store.custody.pin(gate, publication_id=artifact_id, path=target, sha256=staged.sha256,
                          token=staged.token)
        atomic_write_bytes(target, staged.temp_path.read_bytes())
        atomic_write_bytes(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2).encode())
        store.discard(staged)
    return metadata


def generated_image_tool_result(artifacts: list[dict[str, Any]]) -> str:
    """Return the compact structured result exposed to the LLM."""
    return json.dumps(
        {
            "artifacts": artifacts,
            "next_step": (
                "Use these artifact paths as reference_images for follow-up edits. "
                "Call the message tool with the artifact paths in the media parameter "
                "to deliver the images to the user. To inspect or copy one from a "
                "Sandbox shell, pass its exact path in the exec tool's 'input_files' "
                "list and write only to an authorized workspace folder. Keep raw paths "
                "internal unless the user asks for debug details."
            ),
        },
        ensure_ascii=False,
    )


_MEDIA_ARTIFACT_REF_KEYS = (
    "id",
    "kind",
    "mime",
    "name",
    "path",
    "bytes",
    "sha256",
)


def media_artifact_ref(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Keep durable asset identity without duplicating prompt text in the event."""
    return {
        key: artifact[key]
        for key in _MEDIA_ARTIFACT_REF_KEYS
        if key in artifact
    }


def parse_embedded_text_payload(value: Any) -> Mapping[str, Any] | None:
    """Parse a JSON payload that may be followed by human-readable labels."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > 16 * 1024 * 1024:
        return None
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed, _end = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            return None
    return cast(Mapping[str, Any], parsed) if isinstance(parsed, Mapping) else None


def select_candidate_artifacts(
    artifacts: list[dict[str, Any]],
    payload: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Keep generated candidates, not the reference/critic image blocks.

    MCP image batches put critic/reference images before the generated
    candidates. The payload reports ``candidate_count``, so the trailing items
    are the generated candidates. If the count is unavailable, keep every item.
    """
    clean = [item for item in artifacts if item]
    if not clean:
        return []
    candidate_count = payload.get("candidate_count") if payload is not None else None
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        candidate_count = None
    raw_candidates = payload.get("candidates") if payload is not None else None
    candidates = cast(list[Any], raw_candidates) if isinstance(raw_candidates, list) else []
    if candidate_count is None and candidates:
        candidate_count = len(candidates)
    if candidate_count is None or not 0 < candidate_count <= len(clean):
        return clean

    selected = [dict(item) for item in clean[-candidate_count:]]
    for index, item in enumerate(selected):
        candidate = candidates[index] if index < len(candidates) else None
        if isinstance(candidate, Mapping):
            candidate_map = cast(Mapping[str, Any], candidate)
            item.setdefault("candidate", candidate_map.get("candidate") or index + 1)
            candidate_prompt = candidate_map.get("prompt")
            if isinstance(candidate_prompt, str) and candidate_prompt.strip():
                item["prompt"] = candidate_prompt
        else:
            item.setdefault("candidate", index + 1)
    return selected
