#!/usr/bin/env python3
"""Extract timestamped key frames from a local video into a reusable storyboard directory."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PTS_TIME_RE = re.compile(r"pts_time:([0-9.]+)")
WHITESPACE_RE = re.compile(r"\s+")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def _format_timestamp(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _timestamp_token(seconds: float) -> str:
    return _format_timestamp(seconds).replace(":", "-").replace(".", "_")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return slug or "video"


def _default_output_dir(video_path: Path) -> Path:
    digest = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return Path("artifacts") / "video-understanding" / f"{_slugify(video_path.stem)}-{digest}"


def _probe_duration(video_path: Path) -> float:
    result = _run([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ])
    return float(result.stdout.strip())


def _detect_scene_times(video_path: Path, scene_threshold: float) -> list[float]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene,{scene_threshold})',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode not in {0, 255}:
        raise RuntimeError(result.stderr.strip() or "ffmpeg scene detection failed")

    times: list[float] = []
    for match in PTS_TIME_RE.finditer(result.stderr):
        times.append(float(match.group(1)))
    return times


def _add_gap_frames(events: list[dict[str, object]], max_gap_seconds: float) -> list[dict[str, object]]:
    if not events:
        return []

    filled: list[dict[str, object]] = [events[0]]
    for event in events[1:]:
        previous = filled[-1]
        prev_time = float(previous["time"])
        current_time = float(event["time"])
        while current_time - prev_time > max_gap_seconds:
            prev_time += max_gap_seconds
            filled.append({"time": prev_time, "reason": "gap", "priority": 1})
        filled.append(event)
    return filled


def _enforce_min_spacing(events: list[dict[str, object]], min_spacing_seconds: float) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for event in sorted(events, key=lambda item: (float(item["time"]), -int(item["priority"]))):
        if not selected:
            selected.append(event)
            continue

        previous = selected[-1]
        if float(event["time"]) - float(previous["time"]) < min_spacing_seconds:
            if int(event["priority"]) > int(previous["priority"]):
                selected[-1] = event
            continue

        selected.append(event)
    return selected


def _cap_events(events: list[dict[str, object]], max_frames: int) -> list[dict[str, object]]:
    if len(events) <= max_frames:
        return events
    if max_frames <= 1:
        return events[:1]
    if max_frames == 2:
        return [events[0], events[-1]]

    interior = events[1:-1]
    slots = max_frames - 2
    sampled = [interior[math.floor(i * len(interior) / slots)] for i in range(slots)]
    return [events[0], *sampled, events[-1]]


def _build_events(
    duration_seconds: float,
    scene_times: list[float],
    *,
    min_spacing_seconds: float,
    max_gap_seconds: float,
    max_frames: int,
) -> list[dict[str, object]]:
    end_time = max(duration_seconds - 0.05, 0.0)
    raw_events: list[dict[str, object]] = [
        {"time": 0.0, "reason": "start", "priority": 3},
        *({"time": time, "reason": "scene", "priority": 2} for time in scene_times if 0.0 < time < end_time),
        {"time": end_time, "reason": "end", "priority": 3},
    ]
    with_gaps = _add_gap_frames(sorted(raw_events, key=lambda item: float(item["time"])), max_gap_seconds)
    deduped = _enforce_min_spacing(with_gaps, min_spacing_seconds)
    return _cap_events(deduped, max_frames)


def _extract_frame(video_path: Path, timestamp_seconds: float, output_path: Path, width: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale='min(iw,{width})':-2" if width > 0 else "null"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{timestamp_seconds:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        vf,
        "-q:v",
        "2",
        "-y",
        str(output_path),
    ]
    _run(cmd)


def _load_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _cache_valid(
    manifest: dict | None,
    video_path: Path,
    params: dict[str, object],
    output_dir: Path,
) -> bool:
    if not manifest:
        return False
    source = manifest.get("source") or {}
    expected = {
        "path": str(video_path.resolve()),
        "size_bytes": video_path.stat().st_size,
        "mtime_ns": video_path.stat().st_mtime_ns,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            return False
    if manifest.get("parameters") != params:
        return False

    for frame in manifest.get("frames") or []:
        path = output_dir / str(frame.get("path") or "")
        if not path.is_file():
            return False
    return True


def _clean_ocr_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.replace("\x0c", " ")).strip()


def _truncate_text(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _default_viewer_value(reason: str) -> str:
    return {
        "start": "setup",
        "gap": "waiting",
        "scene": "proof",
        "end": "cta",
    }.get(reason, "proof")


def _default_semantic_summary(frame: dict[str, object]) -> str:
    excerpt = str(frame.get("ocr_excerpt") or "").strip()
    reason = str(frame.get("reason") or "scene")
    if excerpt:
        return f"Screen shows: {excerpt}"
    return {
        "start": "Opening frame that establishes the initial state.",
        "gap": "Checkpoint frame during a quieter stretch of the video.",
        "scene": "Visual state changes here.",
        "end": "Ending frame or final visible state.",
    }.get(reason, "Meaningful frame from the video.")


def _ocr_frame(frame_path: Path) -> dict[str, object]:
    if not shutil.which("tesseract"):
        return {"enabled": False, "text": "", "excerpt": "", "word_count": 0}

    try:
        result = _run([
            "tesseract",
            str(frame_path),
            "stdout",
            "--psm",
            "11",
            "quiet",
        ])
    except Exception:
        return {"enabled": True, "text": "", "excerpt": "", "word_count": 0}

    text = _clean_ocr_text(result.stdout)
    return {
        "enabled": True,
        "text": text,
        "excerpt": _truncate_text(text) if text else "",
        "word_count": len(text.split()) if text else 0,
    }


def _sample_frames_for_semantics(frames: list[dict[str, object]], max_frames: int) -> list[dict[str, object]]:
    if len(frames) <= max_frames:
        return frames
    if max_frames <= 1:
        return frames[:1]
    if max_frames == 2:
        return [frames[0], frames[-1]]

    interior = frames[1:-1]
    slots = max_frames - 2
    sampled = [interior[math.floor(i * len(interior) / slots)] for i in range(slots)]
    return [frames[0], *sampled, frames[-1]]


def _encode_frame_data_url(frame_path: Path) -> str:
    encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _build_fallback_semantic_storyboard(frames: list[dict[str, object]]) -> dict[str, object]:
    semantic_frames = []
    for frame in frames:
        semantic_frames.append({
            "index": frame["index"],
            "timestamp": frame["timestamp"],
            "reason": frame["reason"],
            "viewer_value": _default_viewer_value(str(frame["reason"])),
            "summary": _default_semantic_summary(frame),
            "confidence": 0.35 if frame.get("ocr_excerpt") else 0.2,
        })
    return {
        "generator": "fallback",
        "overall_summary": "OCR-backed storyboard generated without a semantic vision model.",
        "video_flow": "Use the frame summaries as a rough semantic map, then inspect images for high-stakes details.",
        "frames": semantic_frames,
    }


def _generate_semantic_storyboard(
    frames: list[dict[str, object]],
    output_dir: Path,
    *,
    semantic_enabled: bool,
    semantic_model: str,
    semantic_max_frames: int,
) -> dict[str, object]:
    if not semantic_enabled:
        return _build_fallback_semantic_storyboard(frames)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _build_fallback_semantic_storyboard(frames)

    try:
        from openai import OpenAI
    except Exception:
        return _build_fallback_semantic_storyboard(frames)

    sampled_frames = _sample_frames_for_semantics(frames, semantic_max_frames)
    content: list[dict[str, object]] = [{
        "type": "text",
        "text": (
            "Analyze these storyboard frames from a source video. "
            "Return strict JSON with keys overall_summary, video_flow, and frames. "
            "frames must be a list of objects with keys index, summary, viewer_value, and confidence. "
            "viewer_value must be one of setup, waiting, proof, payoff, cta, transition. "
            "Use OCR excerpts as hints but rely on the frame images for meaning."
        ),
    }]
    for frame in sampled_frames:
        frame_path = output_dir / str(frame["path"])
        content.append({
            "type": "text",
            "text": (
                f"Frame {frame['index']} at {frame['timestamp']} "
                f"(reason={frame['reason']}, ocr={frame.get('ocr_excerpt') or 'none'})"
            ),
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": _encode_frame_data_url(frame_path)},
        })

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=semantic_model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        payload = json.loads(raw)
        semantic_frames = []
        by_index = {int(frame["index"]): frame for frame in frames}
        for item in payload.get("frames") or []:
            index = int(item.get("index"))
            source = by_index.get(index)
            if not source:
                continue
            semantic_frames.append({
                "index": index,
                "timestamp": source["timestamp"],
                "reason": source["reason"],
                "viewer_value": str(item.get("viewer_value") or _default_viewer_value(str(source["reason"]))),
                "summary": str(item.get("summary") or _default_semantic_summary(source)),
                "confidence": float(item.get("confidence") or 0.5),
            })
        if not semantic_frames:
            return _build_fallback_semantic_storyboard(frames)
        return {
            "generator": semantic_model,
            "overall_summary": str(payload.get("overall_summary") or "").strip()
            or "Semantic storyboard generated from sampled frames.",
            "video_flow": str(payload.get("video_flow") or "").strip()
            or "Review frame summaries in order to understand the visible progression.",
            "frames": semantic_frames,
        }
    except Exception:
        return _build_fallback_semantic_storyboard(frames)


def _build_storyboard_markdown(manifest: dict[str, object]) -> str:
    frames = manifest["frames"]
    lines = [
        "# Video Storyboard",
        "",
        f"Source: `{manifest['source']['path']}`",
        f"Duration: `{_format_timestamp(float(manifest['source']['duration_seconds']))}`",
        f"Frames: `{len(frames)}`",
        f"OCR Enabled: `{str(manifest.get('ocr_enabled', False)).lower()}`",
        "",
        "Read `storyboard.txt` first if you want the most compact text-only version for a smaller model.",
        "",
        "## Quick Read",
    ]

    for frame in frames:
        excerpt = frame.get("ocr_excerpt") or "none"
        lines.append(
            f"- `#{frame['index']}` `{frame['timestamp']}` `{frame['reason']}` "
            f"`{frame['path']}` text: {excerpt}"
        )
    lines.extend([
        "",
        "## Frame Notes",
    ])
    for frame in frames:
        lines.extend([
            "",
            f"### Frame {frame['index']}",
            f"- Time: `{frame['timestamp']}`",
            f"- Reason: `{frame['reason']}`",
            f"- File: `{frame['path']}`",
            f"- OCR Text: {frame.get('ocr_excerpt') or 'none'}",
        ])
    return "\n".join(lines) + "\n"


def _build_storyboard_text(manifest: dict[str, object]) -> str:
    frames = manifest["frames"]
    lines = [
        "VIDEO STORYBOARD",
        f"source: {manifest['source']['path']}",
        f"duration: {_format_timestamp(float(manifest['source']['duration_seconds']))}",
        f"frame_count: {len(frames)}",
        f"ocr_enabled: {str(manifest.get('ocr_enabled', False)).lower()}",
        "frames:",
    ]
    for frame in frames:
        excerpt = frame.get("ocr_excerpt") or "none"
        lines.append(
            f"{frame['index']} | {frame['timestamp']} | {frame['reason']} | "
            f"{frame['path']} | text={excerpt}"
        )
    return "\n".join(lines) + "\n"


def _build_semantic_storyboard_markdown(manifest: dict[str, object]) -> str:
    semantic = manifest["semantic_storyboard"]
    lines = [
        "# Semantic Video Storyboard",
        "",
        f"Source: `{manifest['source']['path']}`",
        f"Generator: `{semantic['generator']}`",
        "",
        "## Overall Summary",
        semantic["overall_summary"],
        "",
        "## Flow",
        semantic["video_flow"],
        "",
        "## Beats",
    ]
    for frame in semantic["frames"]:
        lines.extend([
            "",
            f"### Frame {frame['index']}",
            f"- Time: `{frame['timestamp']}`",
            f"- Reason: `{frame['reason']}`",
            f"- Viewer Value: `{frame['viewer_value']}`",
            f"- Summary: {frame['summary']}",
            f"- Confidence: `{frame['confidence']}`",
        ])
    return "\n".join(lines) + "\n"


def _build_semantic_storyboard_text(manifest: dict[str, object]) -> str:
    semantic = manifest["semantic_storyboard"]
    lines = [
        "SEMANTIC VIDEO STORYBOARD",
        f"source: {manifest['source']['path']}",
        f"generator: {semantic['generator']}",
        f"overall_summary: {semantic['overall_summary']}",
        f"video_flow: {semantic['video_flow']}",
        "frames:",
    ]
    for frame in semantic["frames"]:
        lines.append(
            f"{frame['index']} | {frame['timestamp']} | {frame['reason']} | "
            f"value={frame['viewer_value']} | confidence={frame['confidence']} | summary={frame['summary']}"
        )
    return "\n".join(lines) + "\n"


def _write_storyboards(output_dir: Path, manifest: dict[str, object]) -> None:
    (output_dir / "storyboard.md").write_text(_build_storyboard_markdown(manifest), encoding="utf-8")
    (output_dir / "storyboard.txt").write_text(_build_storyboard_text(manifest), encoding="utf-8")
    (output_dir / "semantic-storyboard.md").write_text(
        _build_semantic_storyboard_markdown(manifest),
        encoding="utf-8",
    )
    (output_dir / "semantic-storyboard.txt").write_text(
        _build_semantic_storyboard_text(manifest),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", help="Path to the source video file")
    parser.add_argument("--output-dir", help="Directory to write manifest, storyboard, and frames into")
    parser.add_argument("--scene-threshold", type=float, default=0.025, help="FFmpeg scene-change threshold")
    parser.add_argument("--min-spacing-seconds", type=float, default=0.75, help="Minimum time between kept frames")
    parser.add_argument("--max-gap-seconds", type=float, default=6.0, help="Maximum uncovered gap between frames")
    parser.add_argument("--max-frames", type=int, default=36, help="Maximum number of extracted frames")
    parser.add_argument("--width", type=int, default=1280, help="Maximum output frame width")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR even if tesseract is available")
    parser.add_argument("--no-semantic", action="store_true", help="Skip semantic storyboard generation")
    parser.add_argument("--semantic-model", default="gpt-5.4-mini", help="Model to use for semantic storyboard generation")
    parser.add_argument("--semantic-max-frames", type=int, default=12, help="Maximum number of frames to send to the semantic model")
    parser.add_argument("--force", action="store_true", help="Ignore matching cache and rebuild")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    if not video_path.is_file():
        print(json.dumps({"error": f"Video not found: {video_path}"}))
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(video_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ocr_enabled = bool(shutil.which("tesseract")) and not args.no_ocr
    semantic_enabled = not args.no_semantic
    params = {
        "scene_threshold": args.scene_threshold,
        "min_spacing_seconds": args.min_spacing_seconds,
        "max_gap_seconds": args.max_gap_seconds,
        "max_frames": args.max_frames,
        "width": args.width,
        "ocr_enabled": ocr_enabled,
        "semantic_enabled": semantic_enabled,
        "semantic_model": args.semantic_model,
        "semantic_max_frames": args.semantic_max_frames,
    }
    manifest_path = output_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    if not args.force and _cache_valid(manifest, video_path, params, output_dir):
        print(json.dumps({"status": "cached", "output_dir": str(output_dir), "manifest": str(manifest_path)}))
        return 0

    duration_seconds = _probe_duration(video_path)
    scene_times = _detect_scene_times(video_path, args.scene_threshold)
    events = _build_events(
        duration_seconds,
        scene_times,
        min_spacing_seconds=args.min_spacing_seconds,
        max_gap_seconds=args.max_gap_seconds,
        max_frames=args.max_frames,
    )

    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    manifest_frames: list[dict[str, object]] = []
    for index, event in enumerate(events, start=1):
        timestamp_seconds = float(event["time"])
        reason = str(event["reason"])
        filename = f"frame-{index:03d}-{_timestamp_token(timestamp_seconds)}-{reason}.jpg"
        frame_path = frames_dir / filename
        _extract_frame(video_path, timestamp_seconds, frame_path, args.width)
        ocr = _ocr_frame(frame_path) if ocr_enabled else {
            "enabled": False,
            "text": "",
            "excerpt": "",
            "word_count": 0,
        }
        manifest_frames.append({
            "index": index,
            "timestamp_seconds": round(timestamp_seconds, 3),
            "timestamp": _format_timestamp(timestamp_seconds),
            "reason": reason,
            "path": str(Path("frames") / filename),
            "ocr_text": ocr["text"],
            "ocr_excerpt": ocr["excerpt"],
            "ocr_word_count": ocr["word_count"],
        })

    manifest = {
        "source": {
            "path": str(video_path),
            "size_bytes": video_path.stat().st_size,
            "mtime_ns": video_path.stat().st_mtime_ns,
            "duration_seconds": round(duration_seconds, 3),
        },
        "parameters": params,
        "ocr_enabled": ocr_enabled,
        "semantic_enabled": semantic_enabled,
        "frame_count": len(manifest_frames),
        "frames": manifest_frames,
    }
    manifest["semantic_storyboard"] = _generate_semantic_storyboard(
        manifest_frames,
        output_dir,
        semantic_enabled=semantic_enabled,
        semantic_model=args.semantic_model,
        semantic_max_frames=args.semantic_max_frames,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_storyboards(output_dir, manifest)

    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
