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
    reference_evidence,
    referenced_hashes,
)
from nanobot.attachments.store import AttachmentStore
from nanobot.config.loader import set_config_path
from nanobot.session.manager import JsonlSessionStore, Session
from tests.attachments.test_media_publication_retention import tree

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+yF9kAAAAASUVORK5CYII="
)


def _store(tmp_path: Path) -> AttachmentStore:
    set_config_path(tmp_path / "config.json")
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


def _published(store: AttachmentStore, backend: JsonlSessionStore, record, *, keep: bool) -> None:
    session = Session("telegram:" + record.id, messages=[{
        "role": "user", "content": "hi", "attachments": [record.to_dict()],
    }])
    backend.save(session)
    snapshot = backend.source.upgrade_source_session(session.key,
        expected_revision=session.revision, expected_generation=session.generation,
        expected_incarnation=session.incarnation)
    if not keep:
        backend.source.delete_source_session(session.key, snapshot.state)


def test_prune_keeps_referenced_and_recent_blobs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    referenced = store.commit(store.stage_bytes(_PNG, mime="image/png", name="keep.png", source="websocket"))
    orphan = store.commit(store.stage_bytes(_PNG + b"x", mime="image/png", name="drop.png", source="websocket"))
    recent = store.commit(store.stage_bytes(_PNG + b"y", mime="image/png", name="recent.png", source="websocket"))
    _published(store, backend, referenced, keep=True)
    _published(store, backend, orphan, keep=False)
    _published(store, backend, recent, keep=False)
    referenced_path, orphan_path, recent_path = [store.path_for(item) for item in (referenced, orphan, recent)]
    _old(referenced_path)
    _old(orphan_path)
    now = time.time()
    os.utime(recent_path, (now, now))
    removed = prune_unreferenced(store, backend.sessions_root, grace_seconds=60)
    assert orphan_path in removed
    assert referenced_path.is_file()
    assert recent_path.is_file()
    assert not orphan_path.exists()
    assert not orphan_path.with_suffix(".json").exists()


def test_prune_dry_run_keeps_files(tmp_path: Path) -> None:
    store = _store(tmp_path)
    backend = JsonlSessionStore(tmp_path / "workspace", sessions_root=tmp_path / "sessions")
    record = store.commit(store.stage_bytes(_PNG, mime="image/png", name="drop.png", source="websocket"))
    _published(store, backend, record, keep=False)
    path = store.path_for(record)
    _old(path)
    before = tree(tmp_path)
    removed = prune_unreferenced(store, backend.sessions_root, grace_seconds=0, dry_run=True)
    assert path in removed
    assert path.is_file()
    assert tree(tmp_path) == before


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

    before = tree(tmp_path)
    with pytest.raises(RuntimeError, match="retired"):
        migrate_legacy_media(sessions, store, dry_run=False)
    assert tree(tmp_path) == before


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

    before = tree(tmp_path)
    with pytest.raises(RuntimeError, match="retired"):
        migrate_legacy_media(sessions, store, dry_run=True)
    assert session_file.read_text(encoding="utf-8") == original
    assert not list(sessions.glob("*.bak-*"))
    assert tree(tmp_path) == before


def test_unreadable_session_evidence_prevents_attachment_prune(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    record = store.commit(
        store.stage_bytes(_PNG, mime="image/png", name="live.png", source="websocket")
    )
    blob = store.path_for(record)
    _old(blob)

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    session_file = sessions / "session.jsonl"
    session_file.write_text(
        json.dumps({"role": "user", "content": "hi", "attachments": [record.to_dict()]}) + "\n",
        encoding="utf-8",
    )

    import nanobot.attachments._inventory as inventory
    import nanobot.attachments.lifecycle as lifecycle
    original_read = lifecycle.read_bytes

    def _unreadable_read(self: Path, *args: object, **kwargs: object):
        if self == session_file:
            raise OSError("evidence unavailable")
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(lifecycle, "read_bytes", _unreadable_read)
    monkeypatch.setattr(inventory, "read_bytes", _unreadable_read)

    referenced, complete = reference_evidence([session_file])

    assert complete is False
    removed = prune_unreferenced(store, sessions, grace_seconds=0)

    assert removed == []
    assert blob.is_file()
