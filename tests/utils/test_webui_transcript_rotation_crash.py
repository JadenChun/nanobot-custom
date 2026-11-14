"""Crash-safety tests for WebUI transcript rotation (A3-8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import nanobot.webui.transcript as transcript_module
from nanobot.webui.transcript import (
    active_transcript_path,
    append_transcript_object,
    read_transcript_lines,
    webui_transcript_path,
    webui_transcript_segments_dir,
)

_QUESTION_PREFIX = "question"


def _force_small_transcript_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nanobot.webui.transcript._MAX_TRANSCRIPT_FILE_BYTES", 520)
    monkeypatch.setattr("nanobot.webui.transcript._ACTIVE_TRANSCRIPT_ROTATE_BYTES", 520)
    monkeypatch.setattr("nanobot.webui.transcript._TARGET_ACTIVE_TRANSCRIPT_BYTES", 260)


def _force_large_transcript_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nanobot.webui.transcript._MAX_TRANSCRIPT_FILE_BYTES", 10_000_000)
    monkeypatch.setattr("nanobot.webui.transcript._ACTIVE_TRANSCRIPT_ROTATE_BYTES", 10_000_000)
    monkeypatch.setattr("nanobot.webui.transcript._TARGET_ACTIVE_TRANSCRIPT_BYTES", 5_000_000)


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


def test_rotation_manifest_failure_leaves_old_active_readable_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)
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
    monkeypatch.setattr("nanobot.config.paths.get_data_dir", lambda: tmp_path)
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
