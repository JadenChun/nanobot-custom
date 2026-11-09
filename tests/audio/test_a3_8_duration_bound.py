"""A3-8: measured audio duration must bound ingress before provider dispatch."""

from __future__ import annotations

import shutil
from unittest.mock import AsyncMock

import pytest

from nanobot.audio import transcription
from nanobot.audio.transcription import (
    EffectiveTranscriptionConfig,
    TranscriptionIngressError,
    measure_audio_duration,
    transcribe_audio_data_url,
    transcribe_audio_file,
)


def _config() -> EffectiveTranscriptionConfig:
    return EffectiveTranscriptionConfig(
        enabled=True,
        provider="groq",
        model="test",
        language=None,
        api_key="key",
        api_base="",
        max_duration_sec=120,
        max_upload_mb=25,
    )


@pytest.mark.asyncio
async def test_transcribe_file_rejects_measured_duration_before_dispatch(
    tmp_path, monkeypatch
) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"\x00" * 8)

    monkeypatch.setattr(
        transcription,
        "measure_audio_duration",
        AsyncMock(return_value=121.5),
    )

    def _forbidden_provider_load(_provider: str) -> None:
        raise AssertionError("provider must not load")

    monkeypatch.setattr(
        transcription,
        "get_transcription_provider",
        _forbidden_provider_load,
    )

    with pytest.raises(TranscriptionIngressError) as exc:
        await transcribe_audio_file(audio_path, _config())
    assert exc.value.detail == "duration"


@pytest.mark.asyncio
async def test_transcribe_data_url_ignores_short_declared_duration_when_measured_is_long(
    tmp_path, monkeypatch
) -> None:
    saved_path = tmp_path / "saved.mp3"
    saved_path.write_bytes(b"\x00" * 8)
    monkeypatch.setattr(
        transcription,
        "save_base64_data_url",
        lambda *args, **kwargs: str(saved_path),
    )
    monkeypatch.setattr(
        transcription,
        "measure_audio_duration",
        AsyncMock(return_value=999.0),
    )

    data_url = "data:audio/mpeg;base64,AAAA"
    with pytest.raises(TranscriptionIngressError) as exc:
        await transcribe_audio_data_url(data_url, _config(), duration_ms=1000)
    assert exc.value.detail == "duration"


@pytest.mark.asyncio
async def test_measure_audio_duration_returns_none_when_ffprobe_is_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    assert await measure_audio_duration(tmp_path / "missing.mp3") is None
