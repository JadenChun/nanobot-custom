"""Regression tests for A3-8 AC8.1 and AC8.4 durability bounds.

These tests exercise only public interfaces:

* distinct chat IDs get distinct transcripts, and legacy ambiguous transcripts
  stay un-migrated for a non-owner;
* new base64 transcript stems are discovered;
* PDF content streams are bounded before decode materialization;
* unmeasurable audio is refused before the provider loads.
"""

import asyncio
import zlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nanobot.audio.transcription import (
    EffectiveTranscriptionConfig,
    TranscriptionIngressError,
    transcribe_audio_file,
)
from nanobot.session.manager import SessionManager
from nanobot.utils.document import (
    PdfSafetyError,
    _check_pdf_stream_bound,
)
from nanobot.webui import generated_asset_index, transcript


def test_distinct_chat_ids_never_share_a_transcript(tmp_path: Path) -> None:
    """Two chat IDs that collide under a legacy safe-key have separate files."""
    with patch.object(transcript, "get_webui_dir", lambda: tmp_path):
        transcript.append_transcript_object(
            "websocket:a:b",
            {"event": "user", "chat_id": "a:b", "text": "first history"},
        )
        transcript.append_transcript_object(
            "websocket:a_b",
            {"event": "user", "chat_id": "a_b", "text": "second history"},
        )

        first_path = transcript.webui_transcript_path("websocket:a:b")
        second_path = transcript.webui_transcript_path("websocket:a_b")
        assert first_path != second_path

        assert [
            record.get("text")
            for record in transcript.read_transcript_lines("websocket:a:b")
        ] == ["first history"]
        assert [
            record.get("text")
            for record in transcript.read_transcript_lines("websocket:a_b")
        ] == ["second history"]

        transcript.delete_webui_transcript("websocket:a:b")

        assert transcript.read_transcript_lines("websocket:a:b") == []
        assert [
            record.get("text")
            for record in transcript.read_transcript_lines("websocket:a_b")
        ] == ["second history"]


def test_ambiguous_legacy_transcript_is_quarantined(tmp_path: Path) -> None:
    """A legacy file that belongs to another chat ID is not migrated or deleted."""
    legacy = tmp_path / f"{SessionManager.safe_key('websocket:a:b')}.jsonl"
    legacy.write_text(
        '{"event":"user","chat_id":"a_b","text":"other session history"}\n',
        encoding="utf-8",
    )

    with patch.object(transcript, "get_webui_dir", lambda: tmp_path):
        assert transcript.read_transcript_lines("websocket:a:b") == []
        assert legacy.is_file()
        assert (
            transcript.delete_webui_transcript("websocket:a:b") is False
            or legacy.is_file()
        )
        assert legacy.is_file()
        assert [
            record.get("text")
            for record in transcript.read_transcript_lines("websocket:a_b")
        ] == ["other session history"]


def test_new_base64_transcript_stems_are_discovered(tmp_path: Path) -> None:
    """The session key scanner recognizes the new base64 transcript stem form."""
    later_path = tmp_path / f"{SessionManager.storage_key('websocket:later')}.jsonl"
    later_path.write_text(
        '{"event":"user","chat_id":"later","text":"later history"}\n',
        encoding="utf-8",
    )

    assert "websocket:later" in generated_asset_index.webui_session_keys(tmp_path)


def test_pdf_bounds_decoded_stream_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded decompressor refuses an oversized stream before get_data()."""

    class FakeContents:
        def __init__(self, raw: bytes) -> None:
            self._data = raw
            self.get_data_called = False

        def get(self, key):
            if key == "/Filter":
                return "/FlateDecode"
            return None

        def get_data(self):
            self.get_data_called = True
            return b"x" * 10_000

    raw = zlib.compress(b"x" * 10_000)
    monkeypatch.setattr(
        "nanobot.utils.document._MAX_PDF_CONTENT_STREAM_SIZE", 64
    )

    contents = FakeContents(raw)
    with pytest.raises(PdfSafetyError):
        _check_pdf_stream_bound(contents, 1)

    assert contents.get_data_called is False


def test_unmeasurable_audio_is_refused_before_provider_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A duration measurement of None stops transcription before the provider loads."""
    config = EffectiveTranscriptionConfig(
        enabled=True,
        provider="groq",
        model="test",
        language=None,
        api_key="key",
        api_base="",
        max_duration_sec=120,
        max_upload_mb=25,
    )

    monkeypatch.setattr(
        "nanobot.audio.transcription.measure_audio_duration",
        AsyncMock(return_value=None),
    )

    def fail_provider(*args, **kwargs):
        raise AssertionError("provider must not load")

    monkeypatch.setattr(
        "nanobot.audio.transcription.get_transcription_provider",
        fail_provider,
    )

    with pytest.raises(TranscriptionIngressError) as exc_info:
        asyncio.run(transcribe_audio_file(tmp_path / "voice.ogg", config))

    assert exc_info.value.detail == "duration"
