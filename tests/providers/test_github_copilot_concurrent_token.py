"""Regression tests for concurrent token refresh in GitHubCopilotProvider (#4677)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest

from nanobot.providers import copilot_auth
from nanobot.providers import github_copilot_provider as gc


def _install_identity(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NANOBOT_COPILOT_AUTH_PATH", str(tmp_path / "copilot.json"))
    authority = copilot_auth.get_copilot_authority()
    generation = authority.begin_login()
    authority.commit_login("github-token", generation)


@pytest.mark.asyncio
async def test_concurrent_token_refresh_fetches_once(monkeypatch, tmp_path: Path):
    """Two concurrent _get_copilot_access_token calls should trigger only one
    HTTP fetch when the token is expired, not two."""
    _install_identity(monkeypatch, tmp_path)

    fetch_count = 0

    def fake_get(url, *, headers, timeout):
        nonlocal fetch_count
        fetch_count += 1
        time.sleep(0.05)
        return httpx.Response(
            200,
            json={"token": "copilot-token", "refresh_in": 1500},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(copilot_auth.httpx, "get", fake_get)

    provider = gc.GitHubCopilotProvider()

    token_a, token_b = await asyncio.gather(
        provider._get_copilot_access_token(),
        provider._get_copilot_access_token(),
    )

    assert token_a == "copilot-token"
    assert token_b == "copilot-token"
    assert fetch_count == 1, (
        f"Expected exactly 1 token fetch under concurrency, got {fetch_count}"
    )


@pytest.mark.asyncio
async def test_second_call_returns_cached_token_while_first_in_flight(
    monkeypatch, tmp_path: Path
):
    """If task A is mid-fetch inside the lock, task B should wait, then find
    the cached token and skip the HTTP call entirely."""
    _install_identity(monkeypatch, tmp_path)

    fetch_count = 0

    def fake_get(url, *, headers, timeout):
        nonlocal fetch_count
        fetch_count += 1
        time.sleep(0.1)
        return httpx.Response(
            200,
            json={"token": "cached-token", "refresh_in": 1500},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(copilot_auth.httpx, "get", fake_get)

    provider = gc.GitHubCopilotProvider()

    # Start task A, let it acquire the lock and begin the HTTP fetch.
    task_a = asyncio.create_task(provider._get_copilot_access_token())
    await asyncio.sleep(0.02)  # task A is now inside the lock, mid-fetch

    # Task B starts while A is still in flight.
    token_b = await provider._get_copilot_access_token()
    token_a = await task_a

    assert token_a == "cached-token"
    assert token_b == "cached-token"
    assert fetch_count == 1


def test_provider_has_no_instance_local_bearer_authority() -> None:
    provider = gc.GitHubCopilotProvider()
    assert not hasattr(provider, "_copilot_token_lock")
    assert not hasattr(provider, "_copilot_access_token")
