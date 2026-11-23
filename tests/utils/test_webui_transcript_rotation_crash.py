"""Crash-safety tests for WebUI transcript rotation (A3-8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import kohaku.webui.transcript as transcript_module
from kohaku.webui.transcript import (
    active_transcript_path,
    append_transcript_object,
    read_transcript_lines,
    webui_transcript_path,
    webui_transcript_segments_dir,
)

_QUESTION_PREFIX = "question"


def _force_small_transcript_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kohaku.webui.transcript._MAX_TRANSCRIPT_FILE_BYTES", 520)
    monkeypatch.setattr("kohaku.webui.transcript._ACTIVE_TRANSCRIPT_ROTATE_BYTES", 520)
    monkeypatch.setattr("kohaku.webui.transcript._TARGET_ACTIVE_TRANSCRIPT_BYTES", 260)


def _force_large_transcript_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kohaku.webui.transcript._MAX_TRANSCRIPT_FILE_BYTES", 10_000_000)
    monkeypatch.setattr("kohaku.webui.transcript._ACTIVE_TRANSCRIPT_ROTATE_BYTES", 10_000_000)
    monkeypatch.setattr("kohaku.webui.transcript._TARGET_ACTIVE_TRANSCRIPT_BYTES", 5_000_000)


def _append_numbered_turn(key: str, chat_id: str, idx: int) -> None:
    append_transcript_object(
        key,
        {"event": "user", "chat_id": chat_id, "text": f"{_QUESTION_PREFIX} {idx} " + ("x" * 24)},
    )
    append_transcript_object(
        key,
        {"event": "message", "chat_id": chat_id, "text": f"answer {idx} " + ("y" * 24)},
    )
    append_transcript_object(key, {"event": "turn_end", "chat_id": chat_id})


def _question_text(idx: int) -> str:
    return f"{_QUESTION_PREFIX} {idx} " + ("x" * 24)


def _active_path_and_bytes(key: str) -> tuple[Path, bytes]:
    path = active_transcript_path(key)
    return path, path.read_bytes()


def _user_texts(lines: list[dict[str, Any]]) -> list[str]:
    return [str(line.get("text") or "") for line in lines if line.get("event") == "user"]


def _segment_files(key: str) -> list[str]:
    directory = webui_transcript_segments_dir(key)
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.glob("*.jsonl"))


def _read_manifest(key: str) -> dict[str, Any]:
    paths = sorted(webui_transcript_segments_dir(key).glob("manifest.json"))
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["version"] == 3
    entries = payload["segments"]
    assert isinstance(entries, list)
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids))
    for entry in entries:
        path = webui_transcript_segments_dir(key) / f"{entry['id']}.jsonl"
        assert path.is_file()
        assert entry["bytes"] == path.stat().st_size
        assert entry["turn_count"] > 0
    active = payload["active"]
    assert isinstance(active, str)
    assert webui_transcript_segments_dir(key).parent.joinpath(active).is_file()
    return payload


def _segment_question_texts(key: str, manifest: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for entry in manifest["segments"]:
        path = webui_transcript_segments_dir(key) / f"{entry['id']}.jsonl"
        texts.extend(_user_texts(transcript_module._read_transcript_file(path)))
    return texts


def test_rotation_manifest_failure_leaves_old_active_readable_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("kohaku.config.paths.get_data_dir", lambda: tmp_path)
    key = "websocket:crash-manifest"
    chat = "crash-manifest"
    turn_count = 5

    _force_large_transcript_budget(monkeypatch)
    for idx in range(1, turn_count + 1):
        _append_numbered_turn(key, chat, idx)

    old_active, old_bytes = _active_path_and_bytes(key)
    _force_small_transcript_budget(monkeypatch)

    original_manifest_writer = transcript_module._write_segment_manifest

    def fail_manifest_write(*args: object, **kwargs: object) -> None:
        raise RuntimeError("manifest failure")

    monkeypatch.setattr(transcript_module, "_write_segment_manifest", fail_manifest_write)
    try:
        with pytest.raises(RuntimeError):
            transcript_module._rotate_active_transcript_if_needed(key)
    finally:
        monkeypatch.setattr(
            transcript_module,
            "_write_segment_manifest",
            original_manifest_writer,
        )

    # The manifest commit is the rotation point: the old active file must stay
    # intact when the commit fails.
    assert old_active.read_bytes() == old_bytes

    lines = read_transcript_lines(key)
    user_texts = [
        str(line.get("text") or "") for line in lines if line.get("event") == "user"
    ]
    for idx in range(1, turn_count + 1):
        assert user_texts.count(_question_text(idx)) == 1


def test_rotation_ignores_stale_legacy_active_after_manifest_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("kohaku.config.paths.get_data_dir", lambda: tmp_path)
    key = "websocket:crash-legacy"
    chat = "crash-legacy"
    turn_count = 5

    _force_small_transcript_budget(monkeypatch)
    for idx in range(1, turn_count + 1):
        _append_numbered_turn(key, chat, idx)

    manifest = json.loads(
        (webui_transcript_segments_dir(key) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["version"] == 3
    assert isinstance(manifest["active"], str)

    active = active_transcript_path(key)
    assert manifest["active"] == active.name

    # Simulate the crash after the manifest commit but before the old legacy
    # active file was deleted: that file still contains an old duplicate turn.
    legacy_path = webui_transcript_path(key)
    duplicate_question = _question_text(1)
    if legacy_path.is_file():
        stale_text = legacy_path.read_text(encoding="utf-8")
    else:
        stale_text = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in (
                {"event": "user", "chat_id": chat, "text": duplicate_question},
                {"event": "turn_end", "chat_id": chat},
            )
        )
    legacy_path.write_text(stale_text, encoding="utf-8")

    lines = read_transcript_lines(key)
    user_texts = [
        str(line.get("text") or "") for line in lines if line.get("event") == "user"
    ]
    assert user_texts.count(duplicate_question) == 1


def test_crash_before_tail_publication_reads_each_turn_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U4: crash after four published segments and before the active tail."""
    monkeypatch.setattr("kohaku.config.paths.get_data_dir", lambda: tmp_path)
    key = "websocket:crash-segment-tail"
    chat = "crash-segment-tail"
    turn_count = 5

    _force_large_transcript_budget(monkeypatch)
    for idx in range(1, turn_count + 1):
        _append_numbered_turn(key, chat, idx)

    old_active, old_bytes = _active_path_and_bytes(key)
    segments_dir = webui_transcript_segments_dir(key)
    _force_small_transcript_budget(monkeypatch)

    original_writer = transcript_module._write_records_to_path
    crashed = False

    def fail_before_tail_write(path: Path, rows: list[dict[str, Any]]) -> None:
        nonlocal crashed
        if (
            not crashed
            and path.parent == old_active.parent
            and ".active." in path.name
            and path.name != old_active.name
        ):
            crashed = True
            raise OSError("simulated crash before the active tail write")
        original_writer(path, rows)

    monkeypatch.setattr(transcript_module, "_write_records_to_path", fail_before_tail_write)
    with pytest.raises(OSError):
        transcript_module._rotate_active_transcript_if_needed(key)
    assert crashed

    # Crash boundary: four segments are published, no tail and no manifest.
    published = [path for path in segments_dir.glob("[0-9]*.jsonl")]
    assert len(published) == 4
    assert not (segments_dir / "manifest.json").is_file()
    assert old_active.read_bytes() == old_bytes

    original_reader = transcript_module._read_transcript_file
    old_active_reads = 0

    def count_old_active_reads(path: Path) -> list[dict[str, Any]]:
        nonlocal old_active_reads
        if path == old_active:
            old_active_reads += 1
        return original_reader(path)

    monkeypatch.setattr(transcript_module, "_read_transcript_file", count_old_active_reads)

    lines = read_transcript_lines(key)
    texts = _user_texts(lines)
    expected = [_question_text(idx) for idx in range(1, turn_count + 1)]
    # Original turns, original order, each turn once.
    assert texts == expected
    assert len(texts) == turn_count

    # Exactly one valid manifest, referenced segments appear once, no segment
    # repeats an old tail turn.
    assert len(sorted(tmp_path.rglob("manifest.json"))) == 1
    manifest = _read_manifest(key)
    manifest_ids = [entry["id"] for entry in manifest["segments"]]
    assert len(manifest_ids) == len(set(manifest_ids))
    assert sorted(f"{segment_id}.jsonl" for segment_id in manifest_ids) == _segment_files(key)
    segment_texts = _segment_question_texts(key, manifest)
    assert len(segment_texts) == len(set(segment_texts))
    active_texts = _user_texts(
        transcript_module._read_transcript_file(active_transcript_path(key))
    )
    assert set(active_texts).isdisjoint(segment_texts)
    assert old_active_reads <= 1
    assert not (segments_dir / "rotation.json").is_file()


def test_normal_rotation_and_post_tail_crash_read_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U5: Part A normal rotation, Part B crash after tail and manifest."""
    monkeypatch.setattr("kohaku.config.paths.get_data_dir", lambda: tmp_path)
    turn_count = 5
    expected = [_question_text(idx) for idx in range(1, turn_count + 1)]

    # Part A: normal rotation, then read.
    key_a = "websocket:rotation-normal"
    _force_large_transcript_budget(monkeypatch)
    for idx in range(1, turn_count + 1):
        _append_numbered_turn(key_a, "rotation-normal", idx)
    old_active_a, _ = _active_path_and_bytes(key_a)
    _force_small_transcript_budget(monkeypatch)
    transcript_module._rotate_active_transcript_if_needed(key_a)

    manifest_a = _read_manifest(key_a)
    active_a = active_transcript_path(key_a)
    # The new active tail is a fresh generation. The store keeps the last turns
    # in it by design, so it is not empty; assert it repeats no segmented turn.
    assert active_a.is_file()
    assert active_a.name == manifest_a["active"]
    assert active_a.name != old_active_a.name
    assert active_a.name.startswith(f"{webui_transcript_path(key_a).stem}.active.")
    segment_texts_a = _segment_question_texts(key_a, manifest_a)
    assert set(_user_texts(transcript_module._read_transcript_file(active_a))).isdisjoint(
        segment_texts_a
    )

    segment_files_a = _segment_files(key_a)
    lines_a = read_transcript_lines(key_a)
    assert _user_texts(lines_a) == expected
    assert len(_user_texts(lines_a)) == turn_count
    assert _segment_files(key_a) == segment_files_a
    assert _read_manifest(key_a) == manifest_a
    assert len(segment_texts_a) == len(set(segment_texts_a))
    assert sorted(f"{entry['id']}.jsonl" for entry in manifest_a["segments"]) == segment_files_a

    # Part B: the new tail and the manifest are committed, then the process
    # crashes before the rotation intent and the old active are removed.
    key_b = "websocket:post-tail-crash"
    _force_large_transcript_budget(monkeypatch)
    for idx in range(1, turn_count + 1):
        _append_numbered_turn(key_b, "post-tail-crash", idx)
    old_active_b, old_bytes_b = _active_path_and_bytes(key_b)
    _force_small_transcript_budget(monkeypatch)
    transcript_module._rotate_active_transcript_if_needed(key_b)

    manifest_b = _read_manifest(key_b)
    segment_files_b = _segment_files(key_b)
    old_active_b.write_bytes(old_bytes_b)
    transcript_module._write_transcript_rotation_intent(
        key_b,
        source_active=old_active_b.name,
        target_active=str(manifest_b["active"]),
        committed_segments=[str(entry["id"]) for entry in manifest_b["segments"]],
    )
    assert old_active_b.is_file()

    lines_b = read_transcript_lines(key_b)
    assert _user_texts(lines_b) == expected
    assert len(_user_texts(lines_b)) == turn_count
    assert _read_manifest(key_b) == manifest_b
    assert _segment_files(key_b) == segment_files_b
    assert not old_active_b.is_file()
    assert not (
        webui_transcript_segments_dir(key_b) / "rotation.json"
    ).is_file()
