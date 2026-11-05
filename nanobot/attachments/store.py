"""Content-addressed storage for inbound attachments."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from nanobot.attachments.models import ATTACHMENT_SCHEMA_VERSION, AttachmentRecord, StagedAttachment
from nanobot.utils.helpers import detect_image_mime, safe_filename
from nanobot.utils.media_decode import FileSizeExceeded, mime_extension

_STAGE_CHUNK_BYTES = 1024 * 1024


def _ensure_private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass
    return path


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, payload: bytes) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        temp.unlink(missing_ok=True)


class AttachmentStore:
    """Store committed attachments by content hash.

    ``root`` is the content-addressed attachments directory. ``media_root`` is
    the parent media root used for legacy ``storage_key`` records and signed
    media projection.
    """

    def __init__(
        self,
        root: Path,
        *,
        media_root: Path | None = None,
        staging_root: Path | None = None,
        logger: Any = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.media_root = Path(media_root or self.root).expanduser().resolve(strict=False)
        self.staging_root = Path(
            staging_root or (self.media_root / ".staging")
        ).expanduser().resolve(strict=False)
        self.logger = logger

    def stage_bytes(
        self,
        data: bytes,
        *,
        mime: str,
        name: str | None,
        source: str,
    ) -> StagedAttachment:
        """Write bytes to staging and return their pending identity."""
        if not data:
            raise ValueError("cannot stage empty attachment")
        sha256 = hashlib.sha256(data).hexdigest()
        extension = mime_extension(mime)
        token = uuid.uuid4().hex
        _ensure_private_dir(self.staging_root)
        temp_path = self.staging_root / f"{token}{extension}"
        with temp_path.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            temp_path.chmod(0o600)
        except OSError:
            pass
        return StagedAttachment(
            token=token,
            temp_path=temp_path,
            mime=mime,
            name=name,
            bytes=len(data),
            sha256=sha256,
            source=source,
        )

    def stage_path(
        self,
        path: Path,
        *,
        mime: str | None = None,
        name: str | None = None,
        source: str,
        max_bytes: int | None = None,
    ) -> StagedAttachment:
        """Copy an acquired file into staging while hashing it."""
        resolved = Path(path).expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise FileNotFoundError(str(resolved))
        display_name = name or resolved.name
        with resolved.open("rb") as head_handle:
            head = head_handle.read(16)
        resolved_mime = (
            mime
            or detect_image_mime(head)
            or mimetypes.guess_type(display_name)[0]
            or "application/octet-stream"
        )
        extension = mime_extension(resolved_mime)
        token = uuid.uuid4().hex
        _ensure_private_dir(self.staging_root)
        temp_path = self.staging_root / f"{token}{extension}"
        digest = hashlib.sha256()
        total = 0
        try:
            with resolved.open("rb") as source_handle, temp_path.open("wb") as target_handle:
                while True:
                    chunk = source_handle.read(_STAGE_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if max_bytes is not None and total > max_bytes:
                        raise FileSizeExceeded(
                            f"File exceeds {max_bytes // (1024 * 1024)}MB limit"
                        )
                    digest.update(chunk)
                    target_handle.write(chunk)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            temp_path.chmod(0o600)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            self._maybe_remove_staging_root()
            raise
        return StagedAttachment(
            token=token,
            temp_path=temp_path,
            mime=resolved_mime,
            name=display_name,
            bytes=total,
            sha256=digest.hexdigest(),
            source=source,
        )

    def commit(self, staged: StagedAttachment) -> AttachmentRecord:
        """Move a staged attachment into the content-addressed store."""
        extension = mime_extension(staged.mime)
        target_dir = _ensure_private_dir(self.root / staged.sha256[:2])
        target = target_dir / f"{staged.sha256}{extension}"
        if target.is_file():
            staged.temp_path.unlink(missing_ok=True)
        else:
            try:
                os.replace(staged.temp_path, target)
            except OSError:
                temp_target = target_dir / f".{staged.sha256}.{uuid.uuid4().hex}.tmp"
                try:
                    shutil.copyfile(staged.temp_path, temp_target)
                    with temp_target.open("rb") as handle:
                        os.fsync(handle.fileno())
                    os.replace(temp_target, target)
                finally:
                    temp_target.unlink(missing_ok=True)
                    staged.temp_path.unlink(missing_ok=True)
            try:
                target.chmod(0o600)
            except OSError:
                pass
            _fsync_directory(target_dir)
        self._write_sidecar(target_dir, staged)
        self._maybe_remove_staging_root()
        display_name = safe_filename(staged.name or "")[:128]
        if not display_name:
            display_name = f"attachment{extension}"
        return AttachmentRecord(
            id=f"att_{uuid.uuid4().hex}",
            sha256=staged.sha256,
            mime=staged.mime,
            bytes=staged.bytes,
            name=display_name,
            source=staged.source,
            created_at=time.time(),
        )

    def discard(self, staged: StagedAttachment) -> None:
        """Remove a staged attachment that never reached a turn."""
        staged.temp_path.unlink(missing_ok=True)
        self._maybe_remove_staging_root()

    def cleanup_staging(self) -> None:
        """Remove an empty staging directory after a rejected batch."""
        self._maybe_remove_staging_root()

    def path_for(self, record: AttachmentRecord) -> Path:
        """Return the current storage path for a record."""
        if record.storage_key:
            candidate = Path(record.storage_key)
            if not candidate.is_absolute():
                candidate = self.media_root / candidate
            return candidate.resolve(strict=False)
        return (
            self.root
            / record.sha256[:2]
            / f"{record.sha256}{mime_extension(record.mime)}"
        )

    def resolve(self, record: AttachmentRecord) -> Path | None:
        """Resolve a readable path inside the media root, or ``None``."""
        try:
            resolved = self.path_for(record).resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if not resolved.is_file():
            return None
        if not resolved.is_relative_to(self.media_root):
            # Legacy/channel records may carry an absolute acquisition path
            # (workspace or test attachment); only that explicit form may
            # resolve outside the media root.
            if not (record.storage_key and Path(record.storage_key).is_absolute()):
                return None
        return resolved

    def record_for_existing_path(
        self,
        path: Path,
        *,
        source: str,
        name: str | None = None,
        mime: str | None = None,
        allow_outside: bool = False,
    ) -> AttachmentRecord | None:
        """Create a read-only legacy record for an already stored media file."""
        try:
            resolved = Path(path).expanduser().resolve(strict=True)
            outside = not resolved.is_relative_to(self.media_root)
        except (OSError, RuntimeError):
            return None
        if not resolved.is_file():
            return None
        digest = hashlib.sha256()
        total = 0
        try:
            with resolved.open("rb") as handle:
                while True:
                    chunk = handle.read(_STAGE_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    digest.update(chunk)
        except OSError:
            return None
        resolved_mime = (
            mime
            or self._head_image_mime(resolved)
            or mimetypes.guess_type(name or resolved.name)[0]
            or "application/octet-stream"
        )
        sha256 = digest.hexdigest()
        canonical = (
            self.root / sha256[:2] / f"{sha256}{mime_extension(resolved_mime)}"
        ).resolve(strict=False)
        if canonical == resolved:
            storage_key = ""
        elif outside:
            if not allow_outside:
                return None
            storage_key = str(resolved)
        else:
            storage_key = resolved.relative_to(self.media_root).as_posix()
        display_name = safe_filename(name or resolved.name)[:128] or resolved.name
        return AttachmentRecord(
            id=f"att_{uuid.uuid4().hex}",
            sha256=sha256,
            mime=resolved_mime,
            bytes=total,
            name=display_name,
            source=source,
            created_at=resolved.stat().st_mtime,
            storage_key=storage_key,
        )

    @staticmethod
    def _head_image_mime(path: Path) -> str | None:
        try:
            with path.open("rb") as handle:
                return detect_image_mime(handle.read(16))
        except OSError:
            return None

    def _write_sidecar(self, target_dir: Path, staged: StagedAttachment) -> None:
        sidecar = target_dir / f"{staged.sha256}.json"
        payload = json.dumps(
            {
                "version": ATTACHMENT_SCHEMA_VERSION,
                "sha256": staged.sha256,
                "mime": staged.mime,
                "bytes": staged.bytes,
                "created_at": time.time(),
            },
            sort_keys=True,
        ).encode("utf-8")
        if sidecar.is_file():
            try:
                if sidecar.read_bytes() == payload:
                    return
            except OSError:
                pass
        _atomic_write(sidecar, payload)

    def _maybe_remove_staging_root(self) -> None:
        try:
            next(self.staging_root.iterdir())
        except StopIteration:
            try:
                self.staging_root.rmdir()
            except OSError:
                pass
        except OSError:
            pass
