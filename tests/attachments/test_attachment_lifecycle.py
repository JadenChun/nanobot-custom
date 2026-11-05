from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import pytest

from nanobot.attachments.lifecycle import (
    migrate_legacy_media,
    prune_unreferenced,
    referenced_hashes,
)
from nanobot.attachments.store import AttachmentStore

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yF9kAAAAASUVORK5CYII="
)


def _store(tmp_path: Path) -> AttachmentStore:
    media_root = tmp_path / "media"
    media_root.mkdir(exist_ok=True)
    return AttachmentStore(media_root / "attachments", media_root=media_root)


def _old(path: Path) -> None:
    old = time.time() - 30 * 24 * 60 * 60
    os.utime(path, (old, old))


def test_referenced_hashes_include_records_and_media_paths(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    sha = "a" * 64
    other = "b" * 64
    event = {
        "role": "user",
        "content": "hi",
        "attachments": [{"sha256": sha}],
        "media": [f"/media/attachments/{other[:2]}/{other}.png"],
    }
    (sessions / "session.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

    assert referenced_hashes([sessions / "session.jsonl"]) == {sha, other}


def test_prune_keeps_referenced_and_recent_blobs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    referenced = store.commit(
        store.stage_bytes(_PNG, mime="image/png", name="keep.png", source="websocket")
    )
    orphan = store.commit(
        store.stage_bytes(_PNG + b"x", mime="image/png", name="drop.png", source="websocket")
    )
    recent = store.commit(
        store.stage_bytes(_PNG + b"y", mime="image/png", name="recent.png", source="websocket")
    )
    referenced_path = store.path_for(referenced)
    orphan_path = store.path_for(orphan)
    recent_path = store.path_for(recent)
    _old(referenced_path)
    _old(orphan_path)
    _old(recent_path)

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "session.jsonl").write_text(
        json.dumps({"role": "user", "content": "hi", "attachments": [referenced.to_dict()]}) + "\n",
        encoding="utf-8",
    )
    # ``recent`` is unreferenced but inside the grace window.
    now = time.time()
    os.utime(recent_path, (now, now))

    removed = prune_unreferenced(store, sessions, grace_seconds=60)

    assert orphan_path in removed
    assert referenced_path.is_file()
    assert recent_path.is_file()
    assert not orphan_path.exists()


def test_prune_dry_run_keeps_files(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.commit(
        store.stage_bytes(_PNG, mime="image/png", name="drop.png", source="websocket")
    )
    path = store.path_for(record)
    _old(path)
    sessions = tmp_path / "sessions"
    sessions.mkdir()

    removed = prune_unreferenced(store, sessions, grace_seconds=0, dry_run=True)

    assert path in removed
    assert path.is_file()


def test_migrate_legacy_media_rewrites_events_and_keeps_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_root = tmp_path / "media"
    channel = media_root / "telegram"
    channel.mkdir(parents=True)
    image = channel / "legacy.png"
    image.write_bytes(_PNG)
    store = AttachmentStore(media_root / "attachments", media_root=media_root)
    monkeypatch.setattr("nanobot.config.paths.get_media_dir", lambda *_: media_root)

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    session_file = sessions / "session.jsonl"
    event = {
        "role": "user",
        "content": "look",
        "media": [str(image)],
    }
    session_file.write_text(json.dumps(event) + "\n", encoding="utf-8")

    report = migrate_legacy_media(sessions, store, dry_run=False)

    assert report.events_migrated == 1
    assert report.records_created == 1
    migrated = json.loads(session_file.read_text(encoding="utf-8").strip())
    assert "media" not in migrated
    assert len(migrated["attachments"]) == 1
    record = migrated["attachments"][0]
    assert record["source"] == "legacy"
    canonical = store.path_for(store.record_for_existing_path(image, source="legacy"))
    assert canonical.is_file()
    backup_root = tmp_path / "attachment-migration-backups"
    assert any(path.name.startswith("session.jsonl.bak-") for path in backup_root.rglob("*"))
    assert session_file.stat().st_mode & 0o777 == 0o600


def test_migrate_legacy_media_dry_run_does_not_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_root = tmp_path / "media"
    channel = media_root / "slack"
    channel.mkdir(parents=True)
    image = channel / "legacy.png"
    image.write_bytes(_PNG)
    store = AttachmentStore(media_root / "attachments", media_root=media_root)
    monkeypatch.setattr("nanobot.config.paths.get_media_dir", lambda *_: media_root)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    session_file = sessions / "session.jsonl"
    original = json.dumps({"role": "user", "content": "look", "media": [str(image)]}) + "\n"
    session_file.write_text(original, encoding="utf-8")

    report = migrate_legacy_media(sessions, store, dry_run=True)

    assert report.events_migrated == 1
    assert session_file.read_text(encoding="utf-8") == original
    assert not list(sessions.glob("*.bak-*"))
