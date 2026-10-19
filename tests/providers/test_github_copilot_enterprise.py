"""Regression tests for GitHub Enterprise / Copilot for Business endpoint overrides (#4220)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from nanobot.providers import copilot_auth
from nanobot.providers import github_copilot_provider as gc


def test_resolve_falls_back_to_default_without_env(monkeypatch):
    monkeypatch.delenv("NANOBOT_COPILOT_BASE_URL", raising=False)
    assert gc._resolve("NANOBOT_COPILOT_BASE_URL", gc.DEFAULT_COPILOT_BASE_URL) == (
        gc.DEFAULT_COPILOT_BASE_URL
    )


def test_resolve_uses_env_override_and_strips(monkeypatch):
    monkeypatch.setenv("NANOBOT_COPILOT_TOKEN_URL", "  https://api.acme.ghe.com/copilot_internal/v2/token  ")
    assert gc._resolve("NANOBOT_COPILOT_TOKEN_URL", gc.DEFAULT_COPILOT_TOKEN_URL) == (
        "https://api.acme.ghe.com/copilot_internal/v2/token"
    )


def test_blank_env_override_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("NANOBOT_COPILOT_BASE_URL", "   ")
    assert gc._resolve("NANOBOT_COPILOT_BASE_URL", gc.DEFAULT_COPILOT_BASE_URL) == (
        gc.DEFAULT_COPILOT_BASE_URL
    )


def test_provider_api_base_honors_env_override(monkeypatch):
    monkeypatch.setenv("NANOBOT_COPILOT_BASE_URL", "https://copilot-api.acme.ghe.com")
    provider = gc.GitHubCopilotProvider()
    assert provider.api_base == "https://copilot-api.acme.ghe.com"


def test_login_uses_enterprise_endpoint_overrides(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("NANOBOT_COPILOT_AUTH_PATH", str(tmp_path / "copilot.json"))
    monkeypatch.setenv("NANOBOT_GITHUB_COPILOT_CLIENT_ID", "enterprise-client-id")
    monkeypatch.setenv("NANOBOT_GITHUB_DEVICE_CODE_URL", "https://ghe.example/login/device/code")
    monkeypatch.setenv(
        "NANOBOT_GITHUB_ACCESS_TOKEN_URL",
        "https://ghe.example/login/oauth/access_token",
    )
    monkeypatch.setenv("NANOBOT_GITHUB_USER_URL", "https://api.ghe.example/user")
    monkeypatch.setattr(gc.webbrowser, "open", lambda _url: None)

    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, *, headers, data):
            calls.append(("post", url, data))
            if url.endswith("/device/code"):
                return FakeResponse(
                    {
                        "device_code": "device-code",
                        "user_code": "user-code",
                        "verification_uri": "https://ghe.example/device",
                        "interval": 1,
                        "expires_in": 60,
                    }
                )
            return FakeResponse({"access_token": "github-token", "expires_in": 3600})

        def get(self, url, *, headers):
            calls.append(("get", url, headers))
            return FakeResponse({"login": "enterprise-user"})

    monkeypatch.setattr(gc.httpx, "Client", FakeClient)
    token = gc.login_github_copilot(print_fn=lambda _message: None)

    assert token.access == "github-token"
    assert copilot_auth.get_copilot_identity().account_id == "enterprise-user"
    assert calls[0] == (
        "post",
        "https://ghe.example/login/device/code",
        {"client_id": "enterprise-client-id", "scope": gc.GITHUB_COPILOT_SCOPE},
    )
    assert calls[1][0:2] == ("post", "https://ghe.example/login/oauth/access_token")
    assert calls[1][2]["client_id"] == "enterprise-client-id"
    assert calls[2][0:2] == ("get", "https://api.ghe.example/user")


@pytest.mark.asyncio
async def test_copilot_token_exchange_uses_enterprise_endpoint_override(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("NANOBOT_COPILOT_AUTH_PATH", str(tmp_path / "copilot.json"))
    monkeypatch.setenv(
        "NANOBOT_COPILOT_TOKEN_URL",
        "https://api.ghe.example/copilot_internal/v2/token",
    )
    authority = copilot_auth.get_copilot_authority()
    generation = authority.begin_login()
    authority.commit_login("github-token", generation)

    calls = []

    def fake_get(url, *, headers, timeout):
        calls.append((url, headers))
        return httpx.Response(
            200,
            json={"token": "copilot-token", "refresh_in": 1200},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(copilot_auth.httpx, "get", fake_get)

    provider = gc.GitHubCopilotProvider()

    assert await provider._get_copilot_access_token() == "copilot-token"
    assert calls[0][0] == "https://api.ghe.example/copilot_internal/v2/token"
