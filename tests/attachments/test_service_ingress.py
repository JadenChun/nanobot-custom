from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock

from nanobot.attachments.ingress import AttachmentIngress
from nanobot.attachments.policy import AttachmentIngressLimits, AttachmentPolicy
from nanobot.attachments.store import AttachmentStore

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yF9kAAAAASUVORK5CYII="
)


def _data_url(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


def _store(tmp_path: Path) -> AttachmentStore:
    media_root = tmp_path / "media"
    media_root.mkdir(exist_ok=True)
    return AttachmentStore(media_root / "attachments", media_root=media_root)


def _ingress(tmp_path: Path, limits: AttachmentIngressLimits | None = None) -> AttachmentIngress:
    return AttachmentIngress(
        _store(tmp_path),
        policy=AttachmentPolicy(limits=limits or AttachmentIngressLimits()),
        logger=MagicMock(),
    )


def test_accepts_image_and_document_and_commits_records(tmp_path: Path) -> None:
    ingress = _ingress(tmp_path)

    result = ingress.accept_batch(
        [
            {"data_url": _data_url("image/png", _PNG), "name": "photo.png"},
            {"data_url": _data_url("text/csv", b"a,b"), "name": "report.csv"},
        ],
        source="websocket",
    )

    assert result.rejection is None
    records, rejection = ingress.commit_batch(result.staged)
    assert rejection is None
    assert [record.mime for record in records] == ["image/png", "text/csv"]
    assert [record.name for record in records] == ["photo.png", "report.csv"]
    assert all(ingress.store.resolve(record) for record in records)


def test_invalid_batch_discards_earlier_staged_files(tmp_path: Path) -> None:
    ingress = _ingress(tmp_path)

    result = ingress.accept_batch(
        [
            {"data_url": _data_url("image/png", _PNG)},
            {"data_url": _data_url("image/svg+xml", b"<svg/>")},
        ],
        source="websocket",
    )

    assert result.rejection == "mime"
    assert result.staged == ()
    assert not ingress.store.staging_root.exists()


def test_document_and_image_counts_are_policy_limited(tmp_path: Path) -> None:
    ingress = _ingress(
        tmp_path,
        AttachmentIngressLimits(max_count=1, max_file_bytes=1024, max_total_bytes=4096),
    )

    result = ingress.accept_batch(
        [
            {"data_url": _data_url("image/png", _PNG)},
            {"data_url": _data_url("text/plain", b"x")},
        ],
        source="websocket",
    )

    assert result.rejection == "too_many_attachments"
    assert not ingress.store.staging_root.exists()


def test_total_size_rolls_back_batch(tmp_path: Path) -> None:
    ingress = _ingress(
        tmp_path,
        AttachmentIngressLimits(max_file_bytes=1024, max_total_bytes=6),
    )

    result = ingress.accept_batch(
        [
            {"data_url": _data_url("text/plain", b"1234")},
            {"data_url": _data_url("text/plain", b"5678")},
        ],
        source="websocket",
    )

    assert result.rejection == "total_size"
    assert result.staged == ()
    assert not ingress.store.staging_root.exists()


def test_same_bytes_from_any_source_share_hash(tmp_path: Path) -> None:
    websocket = _ingress(tmp_path)
    mobile = _ingress(tmp_path)

    first = websocket.accept_batch(
        [{"data_url": _data_url("image/png", _PNG), "name": "photo.png"}],
        source="websocket",
    )
    second = mobile.accept_batch(
        [{"data_url": _data_url("image/png", _PNG), "name": "photo.png"}],
        source="mobile",
    )

    assert first.rejection is None and second.rejection is None
    first_records, _ = websocket.commit_batch(first.staged)
    second_records, _ = mobile.commit_batch(second.staged)
    assert first_records[0].sha256 == second_records[0].sha256
    assert websocket.store.path_for(first_records[0]) == mobile.store.path_for(
        second_records[0]
    )
    assert first_records[0].source == "websocket"
    assert second_records[0].source == "mobile"
