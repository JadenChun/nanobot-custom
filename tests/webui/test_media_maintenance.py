from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nanobot.webui.media_maintenance import (
    collect_referenced_media_paths,
    garbage_collect_generated_media,
)

NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


def _asset(root: Path, name: str, *, size: int, age_days: int) -> Path:
    path = root / "2026-05-01" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    path.with_suffix(".json").write_text(
        json.dumps({"id": path.stem, "kind": "image", "path": str(path)}),
        encoding="utf-8",
    )
    timestamp = (NOW - timedelta(days=age_days)).timestamp()
    os.utime(path, (timestamp, timestamp))
    os.utime(path.with_suffix(".json"), (timestamp, timestamp))
    return path


def test_orphan_cleanup_deletes_only_unreferenced_old_assets(tmp_path: Path) -> None:
    root = tmp_path / "generated"
    referenced = _asset(root, "kept.png", size=10, age_days=20)
    orphan = _asset(root, "orphan.png", size=20, age_days=20)

    result = garbage_collect_generated_media(
        artifact_root=root,
        referenced={referenced.resolve()},
        orphan_grace_days=7,
        now=NOW,
    )

    assert referenced.is_file()
    assert not orphan.exists()
    assert not orphan.with_suffix(".json").exists()
    assert result["deleted"] == 1
    assert result["deleted_bytes"] == 20
    assert result["kept"] == 1


def test_retention_deletes_referenced_assets(tmp_path: Path) -> None:
    root = tmp_path / "generated"
    old = _asset(root, "old.png", size=10, age_days=40)

    result = garbage_collect_generated_media(
        artifact_root=root,
        referenced={old.resolve()},
        retention_days=30,
        now=NOW,
    )

    assert not old.exists()
    assert result["deleted"] == 1


def test_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    root = tmp_path / "generated"
    orphan = _asset(root, "orphan.png", size=20, age_days=20)

    result = garbage_collect_generated_media(
        artifact_root=root,
        referenced=set(),
        orphan_grace_days=7,
        now=NOW,
        dry_run=True,
    )

    assert orphan.is_file()
    assert result["deleted"] == 1
    assert result["dry_run"] is True


def test_quota_never_deletes_referenced_assets(tmp_path: Path) -> None:
    root = tmp_path / "generated"
    referenced = _asset(root, "kept.png", size=80, age_days=1)
    orphan_old = _asset(root, "old.png", size=40, age_days=20)
    orphan_new = _asset(root, "new.png", size=40, age_days=2)

    result = garbage_collect_generated_media(
        artifact_root=root,
        referenced={referenced.resolve()},
        orphan_grace_days=7,
        max_bytes=60,
        now=NOW,
    )

    assert referenced.is_file()
    assert not orphan_old.exists()
    assert not orphan_new.exists()
    assert result["quota_unmet_bytes"] == 20


def test_collect_referenced_media_paths_reads_transcripts_and_indexes(tmp_path: Path) -> None:
    webui = tmp_path / "webui"
    webui.mkdir()
    media_root = tmp_path / "media"
    first = media_root / "generated" / "a.png"
    second = media_root / "generated" / "b.mp4"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    (webui / "session.jsonl").write_text(
        json.dumps({"tool_events": [{"media_artifacts": [{"path": str(first)}]}]}),
        encoding="utf-8",
    )
    (webui / "session.generated-assets.json").write_text(
        json.dumps({"groups": [{"artifacts": [{"path": str(second)}]}]}),
        encoding="utf-8",
    )

    referenced = collect_referenced_media_paths(webui, media_root)

    assert referenced == {first.resolve(), second.resolve()}
