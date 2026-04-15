"""Tests for the video-understanding storyboard helpers."""

from __future__ import annotations

import importlib.util
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
