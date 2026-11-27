"""Signed media helpers for the WebUI HTTP surface."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import mimetypes
import os
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from websockets.http11 import Request as WsRequest
from websockets.http11 import Response

from kohaku.config.paths import get_media_dir
from kohaku.utils.helpers import safe_filename
from kohaku.webui.http_utils import (
    case_insensitive_header as _case_insensitive_header,
)
from kohaku.webui.http_utils import (
    http_error as _http_error,
)
from kohaku.webui.http_utils import (
    http_response as _http_response,
)

MediaDirProvider = Callable[[str | None], Path]
SignedMediaPath = Callable[[Path], dict[str, str] | None]


def b64url_encode(data: bytes) -> str:
    """URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(value: str) -> bytes:
    """Reverse of :func:`b64url_encode`; caller handles decode errors."""
    pad = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + pad)


def _default_media_dir(channel: str | None = None) -> Path:
    return get_media_dir(channel)


# Deterministic extension-to-MIME lookup so OS MIME registries never fail to
# classify safe media or PDFs consistently across Linux, macOS, and Windows.
_MEDIA_EXTENSION_MIMES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
    ".avif": "image/avif",
    ".ico": "image/x-icon",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".ogv": "video/ogg",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".weba": "audio/webm",
    ".opus": "audio/opus",
    ".pdf": "application/pdf",
}


def guess_media_mime(name: str) -> str | None:
    """Return a deterministic MIME type for known media/PDF suffixes or fall back."""
    ext = Path(name).suffix.lower()
    if ext in _MEDIA_EXTENSION_MIMES:
        return _MEDIA_EXTENSION_MIMES[ext]
    mime, _ = mimetypes.guess_type(name)
    return mime


# Allowed MIME types we actually serve from the media endpoint. Anything
# outside this set is degraded to ``application/octet-stream`` so an
# attacker who somehow gets a signed URL for an unexpected file type can't
# trick the browser into sniffing executable content.
_MEDIA_ALLOWED_MIMES: frozenset[str] = frozenset({
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/svg+xml",
    "image/bmp",
    "image/avif",
    "image/x-icon",
    "image/vnd.microsoft.icon",
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/ogg",
    "video/x-matroska",
    "video/x-msvideo",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/mp4",
    "audio/x-m4a",
    "audio/flac",
    "audio/x-flac",
    "audio/aac",
    "audio/webm",
    "audio/opus",
    "application/pdf",
})


def is_allowed_media_mime(mime: str | None) -> bool:
    """Return True when ``mime`` may be served inline from the signed media route."""
    return mime in _MEDIA_ALLOWED_MIMES
_SVG_MEDIA_HEADERS: tuple[tuple[str, str], ...] = (
    (
        "Content-Security-Policy",
        "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; sandbox",
    ),
)

_BYTE_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def _parse_single_byte_range(range_header: str, size: int) -> tuple[int, int]:
    """Parse a single HTTP byte range for signed media responses."""
    if size <= 0 or "," in range_header:
        raise ValueError("invalid byte range")
    m = _BYTE_RANGE_RE.fullmatch(range_header.strip())
    if m is None:
        raise ValueError("invalid byte range")
    start_text, end_text = m.groups()
    if not start_text and not end_text:
        raise ValueError("invalid byte range")
    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise ValueError("invalid byte range")
        start = max(size - suffix_length, 0)
        end = size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
        if start >= size or start > end:
            raise ValueError("invalid byte range")
        end = min(end, size - 1)
    return start, end


def sign_media_path(
    abs_path: Path,
    *,
    secret: bytes,
    media_dir: MediaDirProvider = _default_media_dir,
) -> str | None:
    """Return a signed ``/api/media/<sig>/<payload>`` URL for a media-root path."""
    try:
        media_root = media_dir(None).resolve()
        rel = abs_path.resolve().relative_to(media_root)
    except (OSError, ValueError):
        return None
    payload = b64url_encode(rel.as_posix().encode("utf-8"))
    mac = hmac.new(secret, payload.encode("ascii"), hashlib.sha256).digest()[:16]
    return f"/api/media/{b64url_encode(mac)}/{payload}"


def sign_or_stage_media_path(
    path: Path,
    *,
    secret: bytes,
    media_dir: MediaDirProvider = _default_media_dir,
    logger: Any | None = None,
    authorize_source: Callable[[Path], bool] | None = None,
) -> dict[str, str] | None:
    """Sign an existing media-root path, or stage an arbitrary file before signing."""
    from kohaku.attachments._media import atomic_write_bytes
    from kohaku.attachments.store import AttachmentStore

    base = media_dir(None).resolve(strict=True)
    store = AttachmentStore(base / "attachments", media_root=base)
    staged_input = None
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            return None
        managed = resolved.is_relative_to(base)
        if not managed and authorize_source is not None and not authorize_source(resolved):
            return None
        publication_id = f"media_{uuid.uuid4().hex}"
        if managed:
            target = resolved
        else:
            source_stat = resolved.stat()
            target_dir = media_dir("websocket")
            safe_name = safe_filename(path.name) or "attachment"
            source_version = "\0".join((os.path.normcase(str(resolved)), str(source_stat.st_size),
                                         str(source_stat.st_mtime_ns), str(source_stat.st_ctime_ns)))
            source_digest = hashlib.sha256(source_version.encode("utf-8")).hexdigest()[:20]
            target = target_dir / f"{source_digest}-{safe_name}"
            # The staged name is derived from the source identity (path, size,
            # mtime, ctime), so an existing target already holds this exact
            # version. Re-staging would copy and hash the whole file again on
            # every transcript read; a media-heavy chat paid tens of megabytes
            # per open. Reuse the staged copy instead and only stage when it is
            # genuinely missing.
            if not target.is_file():
                staged_input = store.stage_path(resolved, source="websocket-outbound")
        with store.custody.fence(create=True) as gate:
            if managed:
                record = store.record_for_existing_path(target, source="websocket-outbound")
                if record is None:
                    return None
                digest = record.sha256
            elif staged_input is None:
                record = store.record_for_existing_path(
                    target, source="websocket-outbound", allow_outside=False,
                )
                if record is None:
                    # The cached copy vanished between the check and the fence;
                    # fall back to staging so the read still succeeds.
                    staged_input = store.stage_path(resolved, source="websocket-outbound")
                    digest = staged_input.sha256
                else:
                    digest = record.sha256
            else:
                digest = staged_input.sha256
            store.custody.pin(gate, publication_id=publication_id, path=target, sha256=digest)
            if staged_input is not None:
                atomic_write_bytes(target, staged_input.temp_path.read_bytes())
            signed = sign_media_path(target, secret=secret, media_dir=media_dir)
            if signed is None:
                return None
            # This public publication ID is not the private custody token and
            # cannot authorize cleanup. It survives markdown-only projection.
            return {"url": f"{signed}?publication={publication_id}", "name": path.name,
                    "id": publication_id}
    except OSError as exc:
        if logger is not None:
            logger.warning("failed to stage outbound media {}: {}", path, exc)
        return None
    finally:
        if staged_input is not None:
            store.discard(staged_input)


def stage_managed_media_path(
    path: Path,
    *,
    media_dir: MediaDirProvider = _default_media_dir,
    authorize_source: Callable[[Path], bool] | None = None,
) -> Path | None:
    """Return the managed copy of ``path``, staging it once when it is unmanaged.

    A read-only projection must not mint a custody publication record: those
    accumulate permanently, and every media sign then re-scans all of them.
    Stage the bytes under the deterministic source-version name if they are not
    already there, and let the caller sign the managed path directly.
    """
    from kohaku.attachments.store import AttachmentStore

    base = media_dir(None).resolve(strict=True)
    store = AttachmentStore(base / "attachments", media_root=base)
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            return None
        if resolved.is_relative_to(base):
            return resolved
        if authorize_source is not None and not authorize_source(resolved):
            return None
        source_stat = resolved.stat()
        target_dir = media_dir("websocket")
        safe_name = safe_filename(path.name) or "attachment"
        source_version = "\0".join((os.path.normcase(str(resolved)), str(source_stat.st_size),
                                     str(source_stat.st_mtime_ns), str(source_stat.st_ctime_ns)))
        source_digest = hashlib.sha256(source_version.encode("utf-8")).hexdigest()[:20]
        target = target_dir / f"{source_digest}-{safe_name}"
        if target.is_file():
            return target
        staged_input = store.stage_path(resolved, source="websocket-outbound")
        try:
            from kohaku.attachments._media import atomic_write_bytes

            atomic_write_bytes(target, staged_input.temp_path.read_bytes())
        finally:
            store.discard(staged_input)
        return target if target.is_file() else None
    except OSError as exc:
        logger.warning("failed to stage projected media {}: {}", path, exc)
        return None


def sign_existing_media_path(
    path: Path, *, secret: bytes, media_dir: MediaDirProvider = _default_media_dir,
) -> dict[str, str] | None:
    """Project an already managed asset without creating publication custody."""
    try:
        resolved = path.resolve(strict=True)
        root = media_dir(None).resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            return None
        signed = sign_media_path(resolved, secret=secret, media_dir=media_dir)
        return {"url": signed, "name": path.name} if signed is not None else None
    except (OSError, ValueError):
        return None


def media_attachment_kind(name: str) -> str:
    """Infer the WebUI media attachment kind from a filename."""
    mime = guess_media_mime(name)
    if mime and mime.startswith("video/"):
        return "video"
    if mime and mime.startswith("image/"):
        return "image"
    return "file"


def signed_media_attachments(
    paths: list[str],
    *,
    sign_path: SignedMediaPath,
) -> list[dict[str, Any]]:
    """Map persisted media paths to WebUI attachment dicts with fresh signed URLs."""
    out: list[dict[str, Any]] = []
    for pstr in paths:
        path = Path(pstr)
        att = sign_path(path)
        if att is None:
            continue
        url = att.get("url")
        if not url:
            continue
        name = att.get("name") or path.name
        item = {"kind": media_attachment_kind(name), "url": url, "name": name}
        if att.get("id"):
            item["id"] = att["id"]
        out.append(item)
    return out


def serve_signed_media(
    sig: str,
    payload: str,
    *,
    secret: bytes,
    request: WsRequest | None = None,
    max_response_bytes: int | None = None,
    media_dir: MediaDirProvider = _default_media_dir,
) -> Response:
    """Serve a signed media URL, including browser-friendly byte ranges."""
    try:
        provided_mac = b64url_decode(sig)
    except (ValueError, binascii.Error):
        return _http_error(401, "invalid signature")
    expected_mac = hmac.new(secret, payload.encode("ascii"), hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(expected_mac, provided_mac):
        return _http_error(401, "invalid signature")
    try:
        rel_bytes = b64url_decode(payload)
        rel_str = rel_bytes.decode("utf-8")
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return _http_error(400, "invalid payload")
    try:
        media_root = media_dir(None).resolve()
        candidate = (media_root / rel_str).resolve()
        candidate.relative_to(media_root)
    except (OSError, ValueError):
        return _http_error(404, "not found")
    if not candidate.is_file():
        return _http_error(404, "not found")

    mime = guess_media_mime(candidate.name)
    if mime is None or not is_allowed_media_mime(mime):
        mime = "application/octet-stream"
    common_headers = [
        ("Accept-Ranges", "bytes"),
        ("Cache-Control", "private, max-age=31536000, immutable"),
        ("X-Content-Type-Options", "nosniff"),
    ]
    if mime == "image/svg+xml":
        common_headers.extend(_SVG_MEDIA_HEADERS)
    try:
        size = candidate.stat().st_size
    except OSError:
        return _http_error(500, "read error")

    range_header = _case_insensitive_header(request.headers, "Range") if request else ""
    if range_header:
        try:
            start, end = _parse_single_byte_range(range_header, size)
        except ValueError:
            return _http_response(
                b"range not satisfiable",
                status=416,
                extra_headers=[
                    ("Accept-Ranges", "bytes"),
                    ("Content-Range", f"bytes */{size}"),
                    ("X-Content-Type-Options", "nosniff"),
                ],
            )
        try:
            if max_response_bytes is not None:
                end = min(end, start + max_response_bytes - 1)
            length = end - start + 1
            with candidate.open("rb") as fh:
                fh.seek(start)
                body = fh.read(length)
        except OSError:
            return _http_error(500, "read error")
        return _http_response(
            body,
            status=206,
            content_type=mime,
            extra_headers=[
                *common_headers,
                ("Content-Range", f"bytes {start}-{end}/{size}"),
            ],
        )

    if max_response_bytes is not None and size > max_response_bytes:
        return _http_error(413, "media too large")
    try:
        body = candidate.read_bytes()
    except OSError:
        return _http_error(500, "read error")
    return _http_response(body, content_type=mime, extra_headers=common_headers)
