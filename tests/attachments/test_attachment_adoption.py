from __future__ import annotations

import base64
from pathlib import Path

import pytest

from nanobot.attachments.adoption import adopt_media_paths
from nanobot.attachments.store import AttachmentStore
from nanobot.bus.events import InboundMessage

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yF9kAAAAASUVORK5CYII="
)


def _store(tmp_path: Path) -> AttachmentStore:
    media_root = tmp_path / "media"
    media_root.mkdir(exist_ok=True)
    return AttachmentStore(media_root / "attachments", media_root=media_root)


def test_adoption_reuses_bound_record_for_canonical_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    record = store.commit(
        store.stage_bytes(_PNG, mime="image/png", name="photo.png", source="mobile")
    )
    blob = store.path_for(record)
    monkeypatch.setattr("nanobot.config.paths.get_media_dir", lambda *_: store.media_root)

    result = adopt_media_paths(
        [str(blob)],
        source="websocket",
        existing=(record,),
    )

    assert result.records == ()
    assert result.paths == (str(blob),)
    assert result.replacements == ()


def test_adoption_dedupes_repeated_paths_within_one_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    channel_file = store.media_root / "telegram" / "photo.png"
    channel_file.parent.mkdir()
    channel_file.write_bytes(_PNG)
    monkeypatch.setattr("nanobot.config.paths.get_media_dir", lambda *_: store.media_root)

    result = adopt_media_paths(
        [str(channel_file), str(channel_file)],
        source="telegram",
    )

    assert len(result.records) == 1
    assert len(result.paths) == 2
    assert result.paths[0] == result.paths[1]


def test_inbound_message_with_bound_record_and_path_keeps_one_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    record = store.commit(
        store.stage_bytes(_PNG, mime="image/png", name="photo.png", source="mobile")
    )
    blob = store.path_for(record)
    monkeypatch.setattr("nanobot.config.paths.get_media_dir", lambda *_: store.media_root)

    message = InboundMessage(
        channel="websocket",
        sender_id="u",
        chat_id="c",
        content="edit this",
        media=[str(blob)],
        attachments=(record,),
    )

    assert len(message.attachments) == 1
    assert message.attachments[0].id == record.id
    assert message.media == [str(blob)]


def test_adoption_keeps_distinct_workspace_files_with_identical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first = workspace / "first.png"
    second = workspace / "second.png"
    first.write_bytes(_PNG)
    second.write_bytes(_PNG)
    monkeypatch.setattr("nanobot.config.paths.get_media_dir", lambda *_: store.media_root)

    result = adopt_media_paths(
        [str(first), str(second)],
        source="sdk",
    )

    assert len(result.records) == 2
    assert {record.storage_key for record in result.records} == {
        str(first.resolve()),
        str(second.resolve()),
    }
