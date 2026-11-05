from __future__ import annotations

import base64
from pathlib import Path

from nanobot.attachments.context import (
    AttachmentContext,
    StoreAttachmentResolver,
    content_with_media_breadcrumbs,
    media_paths_for_message,
)
from nanobot.attachments.store import AttachmentStore

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yF9kAAAAASUVORK5CYII="
)


def _store(tmp_path: Path) -> AttachmentStore:
    media_root = tmp_path / "media"
    media_root.mkdir()
    return AttachmentStore(media_root / "attachments", media_root=media_root)


def test_image_record_builds_vision_block_and_path_reference(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.commit(
        store.stage_bytes(_PNG, mime="image/png", name="photo.png", source="mobile")
    )
    context = AttachmentContext(StoreAttachmentResolver(store))

    content = context.user_content("edit this", [record])

    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "edit this"}
    assert content[2]["type"] == "text"
    assert f"[image: {store.path_for(record)}]" in content[2]["text"]


def test_non_image_record_appends_attachment_reference(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.commit(
        store.stage_bytes(b"a,b", mime="text/csv", name="report.csv", source="api")
    )
    context = AttachmentContext(StoreAttachmentResolver(store))

    content = context.user_content("read this", [record])

    assert content == f"read this\n\n[Attachment: {store.path_for(record)}]"


def test_missing_record_emits_unavailable_reference(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.commit(
        store.stage_bytes(_PNG, mime="image/png", name="photo.png", source="mobile")
    )
    store.path_for(record).unlink()
    context = AttachmentContext(StoreAttachmentResolver(store))

    content = context.user_content("edit this", [record])

    assert content == "edit this\n\n[image: unavailable (photo.png)]"


def test_references_are_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.commit(
        store.stage_bytes(_PNG, mime="image/png", name="photo.png", source="mobile")
    )
    reference = f"[image: {store.path_for(record)}]"
    context = AttachmentContext(StoreAttachmentResolver(store))

    assert context.append_references(f"look\n{reference}", [reference]) == f"look\n{reference}"


def test_breadcrumbs_are_idempotent() -> None:
    assert content_with_media_breadcrumbs(
        "user",
        "edit\n[image: /media/photo.jpg]",
        ["/media/photo.jpg"],
    ) == "edit\n[image: /media/photo.jpg]"


def test_history_paths_use_records_when_media_is_absent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.commit(
        store.stage_bytes(_PNG, mime="image/png", name="photo.png", source="mobile")
    )

    paths = media_paths_for_message(
        {"attachments": [record.to_dict()]},
        resolver=StoreAttachmentResolver(store),
    )

    assert paths == [str(store.path_for(record))]


def test_history_paths_ignore_legacy_media_without_records() -> None:
    assert media_paths_for_message({"media": ["/m/legacy.png"]}) == []
