"""Tests for the video-understanding storyboard helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "nanobot"
        / "skills"
        / "video-understanding"
        / "scripts"
        / "extract_keyframes.py"
    )
    spec = importlib.util.spec_from_file_location("video_understanding_extract_keyframes", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_ocr_text_collapses_noise():
    module = _load_module()

    cleaned = module._clean_ocr_text("Reminder\n\nset \x0c  for   5pm")

    assert cleaned == "Reminder set for 5pm"


def test_storyboard_text_includes_ocr_excerpt():
    module = _load_module()
    manifest = {
        "source": {"path": "/tmp/demo.mp4", "duration_seconds": 3.0},
        "ocr_enabled": True,
        "semantic_storyboard": {
            "generator": "fallback",
            "overall_summary": "Demo flow",
            "video_flow": "Open app then create reminder",
            "frames": [],
        },
        "frames": [
            {
                "index": 1,
                "timestamp": "00:00:00.000",
                "reason": "start",
                "path": "frames/frame-001.jpg",
                "ocr_excerpt": "Create reminder for drink water",
            },
            {
                "index": 2,
                "timestamp": "00:00:02.000",
                "reason": "scene",
                "path": "frames/frame-002.jpg",
                "ocr_excerpt": "",
            },
        ],
    }

    storyboard = module._build_storyboard_text(manifest)

    assert "ocr_enabled: true" in storyboard
    assert "text=Create reminder for drink water" in storyboard
    assert "text=none" in storyboard


def test_fallback_semantic_storyboard_uses_ocr_excerpt():
    module = _load_module()
    frames = [
        {
            "index": 1,
            "timestamp": "00:00:00.000",
            "reason": "start",
            "ocr_excerpt": "Create reminder for drink water",
        },
        {
            "index": 2,
            "timestamp": "00:00:03.000",
            "reason": "end",
            "ocr_excerpt": "",
        },
    ]

    semantic = module._build_fallback_semantic_storyboard(frames)

    assert semantic["generator"] == "fallback"
    assert semantic["frames"][0]["summary"] == "Screen shows: Create reminder for drink water"
    assert semantic["frames"][0]["viewer_value"] == "setup"
    assert semantic["frames"][1]["viewer_value"] == "cta"


def test_semantic_storyboard_text_contains_semantic_summary():
    module = _load_module()
    manifest = {
        "source": {"path": "/tmp/demo.mp4"},
        "semantic_storyboard": {
            "generator": "fallback",
            "overall_summary": "The app opens and shows a reminder flow.",
            "video_flow": "Open app, create reminder, confirm final state.",
            "frames": [
                {
                    "index": 1,
                    "timestamp": "00:00:00.000",
                    "reason": "start",
                    "viewer_value": "setup",
                    "confidence": 0.4,
                    "summary": "Opening screen introduces the reminder form.",
                }
            ],
        },
    }

    storyboard = module._build_semantic_storyboard_text(manifest)

    assert "SEMANTIC VIDEO STORYBOARD" in storyboard
    assert "overall_summary: The app opens and shows a reminder flow." in storyboard
    assert "summary=Opening screen introduces the reminder form." in storyboard


def test_event_selection_preserves_endpoints_and_frame_limit():
    module = _load_module()
    events = module._build_events(
        30.0, [2.0, 2.1, 10.0, 21.0],
        min_spacing_seconds=0.75, max_gap_seconds=6.0, max_frames=5,
    )
    assert len(events) == 5
    assert events[0] == {"time": 0.0, "reason": "start", "priority": 3}
    assert events[-1] == {"time": 29.95, "reason": "end", "priority": 3}
    assert all(left["time"] < right["time"] for left, right in zip(events, events[1:]))


def test_main_writes_storyboards_and_reuses_cache(tmp_path, monkeypatch, capsys):
    module = _load_module()
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"offline fixture")
    output = tmp_path / "storyboard"
    monkeypatch.setattr(module.sys, "argv", [
        "extract_keyframes.py", str(video), "--output-dir", str(output),
        "--no-ocr", "--no-semantic",
    ])
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(module, "_probe_duration", lambda path: 3.0)
    monkeypatch.setattr(module, "_detect_scene_times", lambda path, threshold: [1.0])
    extracted = []

    def extract(path, timestamp, target, width):
        extracted.append(timestamp)
        target.write_bytes(b"offline frame")

    monkeypatch.setattr(module, "_extract_frame", extract)
    assert module.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["frame_count"] == len(extracted) == 3
    assert manifest["semantic_storyboard"]["generator"] == "fallback"
    for name in ("storyboard.md", "storyboard.txt", "semantic-storyboard.md", "semantic-storyboard.txt"):
        assert (output / name).read_text().strip()
    assert module.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "cached"
    assert len(extracted) == 3
