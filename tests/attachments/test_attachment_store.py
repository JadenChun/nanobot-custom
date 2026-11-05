from __future__ import annotations

import base64
from pathlib import Path

import pytest

from nanobot.attachments.models import AttachmentRecord
from nanobot.attachments.store import AttachmentStore
from nanobot.utils.media_decode import FileSizeExceeded

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yF9kAAAAASUVORK5CYII="
)


def _store(tmp_path: Path) -> AttachmentStore:
    media_root = tmp_path / "media"
    media_root.mkdir()
    return AttachmentStore(
        media_root / "attachments",
        media_root=media_root,
    )


def test_commit_is_content_addressed_and_writes_sidecar(tmp_path: Path) -> None:
    store = _store(tmp_path)
    staged = store.stage_bytes(_PNG, mime="image/png", name="photo.png", source="websocket")

    record = store.commit(staged)

    path = store.path_for(record)
    assert path == store.root / record.sha256[:2] / f"{record.sha256}.png"
    assert path.read_bytes() == _PNG
    assert path.parent.parent == store.root
    assert record.name == "photo.png"
    assert record.mime == "image/png"
    assert record.bytes == len(_PNG)
    assert (path.parent / f"{record.sha256}.json").is_file()


def test_identical_bytes_dedupe_blob_but_keep_distinct_records(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.commit(
        store.stage_bytes(_PNG, mime="image/png", name="one.png", source="websocket")
    )
    second = store.commit(
        store.stage_bytes(_PNG, mime="image/png", name="two.png", source="api")
    )

    assert first.id != second.id
    assert first.sha256 == second.sha256
    assert store.path_for(first) == store.path_for(second)
    assert store.resolve(first) == store.resolve(second)


def test_discard_removes_staged_bytes_and_empty_staging_dir(tmp_path: Path) -> None:
    store = _store(tmp_path)
    staged = store.stage_bytes(b"data", mime="text/plain", name="a.txt", source="websocket")
    assert staged.temp_path.is_file()

    store.discard(staged)

    assert not staged.temp_path.exists()
    assert not store.staging_root.exists()


def test_stage_path_enforces_limit_and_removes_partial_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source = tmp_path / "large.bin"
    source.write_bytes(b"x" * 64)

    with pytest.raises(FileSizeExceeded):
        store.stage_path(source, mime="application/octet-stream", source="api", max_bytes=32)

    assert not store.staging_root.exists()


def test_legacy_record_resolves_existing_media_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    legacy = store.media_root / "websocket" / "old.png"
    legacy.parent.mkdir()
    legacy.write_bytes(_PNG)

    record = store.record_for_existing_path(legacy, source="legacy")

    assert record is not None
    assert record.storage_key == "websocket/old.png"
    assert store.resolve(record) == legacy.resolve()


def test_resolve_rejects_missing_and_outside_media_root(tmp_path: Path) -> None:
    store = _store(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(_PNG)
    record = AttachmentRecord(
        id="att_test",
        sha256="",
        mime="image/png",
        bytes=len(_PNG),
        name="outside.png",
        source="legacy",
        created_at=0,
        storage_key="../outside.png",
    )
    assert store.resolve(record) is None

    missing = AttachmentRecord(
        id="att_missing",
        sha256="a" * 64,
        mime="image/png",
        bytes=1,
        name="missing.png",
        source="websocket",
        created_at=0,
    )
    assert store.resolve(missing) is None
