"""Rate-limit recovery: reschedule a client job after the 5-hour window resets."""

import time

from nanobot.cli import commands


def _patch_usage(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr(
        "nanobot.providers.codex_auth.get_codex_usage", lambda: payload
    )


def test_looks_like_rate_limit() -> None:
    assert commands._looks_like_codex_rate_limit(
        "Error calling Codex: ChatGPT rate limit triggered. Please try again shortly."
    )
    assert commands._looks_like_codex_rate_limit("error: rate_limited")
    assert not commands._looks_like_codex_rate_limit("unknown tool error")


def test_reschedules_after_primary_window(monkeypatch) -> None:
    reset_at = int(time.time()) + 3600
    _patch_usage(
        monkeypatch,
        {
            "rate_limit_reached_type": "primary",
            "rate_limit": {
                "primary_window": {"reset_at": reset_at, "used_percent": 100},
                "secondary_window": {"used_percent": 74},
            },
        },
    )
    retry = commands._codex_rate_limit_retry_at(current_retries=0)
    assert retry == reset_at * 1000 + commands._RATE_LIMIT_RETRY_BUFFER_MS


def test_skips_when_weekly_window_is_the_limit(monkeypatch) -> None:
    reset_at = int(time.time()) + 3600
    _patch_usage(
        monkeypatch,
        {
            "rate_limit_reached_type": "secondary",
            "rate_limit": {
                "primary_window": {"reset_at": reset_at, "used_percent": 100},
                "secondary_window": {"used_percent": 100},
            },
        },
    )
    assert commands._codex_rate_limit_retry_at(current_retries=0) is None


def test_skips_when_secondary_exhausted_without_type(monkeypatch) -> None:
    reset_at = int(time.time()) + 3600
    _patch_usage(
        monkeypatch,
        {
            "rate_limit_reached_type": None,
            "rate_limit": {
                "primary_window": {"reset_at": reset_at, "used_percent": 100},
                "secondary_window": {"used_percent": 100},
            },
        },
    )
    assert commands._codex_rate_limit_retry_at(current_retries=0) is None


def test_skips_when_retry_budget_exhausted(monkeypatch) -> None:
    reset_at = int(time.time()) + 3600
    _patch_usage(
        monkeypatch,
        {
            "rate_limit_reached_type": "primary",
            "rate_limit": {
                "primary_window": {"reset_at": reset_at, "used_percent": 100},
                "secondary_window": {"used_percent": 10},
            },
        },
    )
    assert commands._codex_rate_limit_retry_at(
        current_retries=commands._RATE_LIMIT_RETRY_MAX
    ) is None


def test_no_reschedule_when_usage_unreadable(monkeypatch) -> None:
    def boom():
        raise RuntimeError("no token")

    monkeypatch.setattr("nanobot.providers.codex_auth.get_codex_usage", boom)
    assert commands._codex_rate_limit_retry_at(current_retries=0) is None
