from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest
from oauth_cli_kit.models import OAuthToken

import kohaku.providers.openai_codex_oauth as codex_oauth
from kohaku.providers.codex_auth import (
    get_codex_credential_view,
    get_codex_login_status,
    logout_codex,
)
from kohaku.providers.oauth_credentials import CredentialStatus
from kohaku.providers.openai_codex_oauth import (
    OpenAICodexOAuthError,
    OpenAICodexOAuthInputError,
    complete_openai_codex_oauth_login,
    start_openai_codex_oauth_login,
)


@pytest.fixture(autouse=True)
def _isolated_authority(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KOHAKU_CODEX_AUTH_PATH", str(tmp_path / "codex.json"))


def _authorization_url(state: str = "expected-state") -> str:
    return f"{codex_oauth.OPENAI_CODEX_PROVIDER.authorize_url}?{urlencode({'state': state})}"


def _wait_for_completion(flow) -> OAuthToken:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        token = complete_openai_codex_oauth_login(flow)
        if token is not None:
            return token
        time.sleep(0.01)
    pytest.fail("OAuth flow did not finish")


def _fake_interactive_login(
    captured: dict[str, object],
    *,
    error: Exception | None = None,
) -> Callable[..., OAuthToken]:
    def login(
        *,
        print_fn,
        prompt_fn,
        provider,
        proxy,
        open_browser,
        storage,
    ) -> OAuthToken:
        captured.update(
            provider=provider,
            proxy=proxy,
            open_browser=open_browser,
        )
        print_fn("Open this URL:")
        print_fn(_authorization_url())
        if not open_browser:
            captured["callback_url"] = prompt_fn("Paste callback URL")
        if error is not None:
            raise error
        return OAuthToken(
            access="access-token",
            refresh="refresh-token",
            expires=2_000_000_000_000,
            account_id="acct-test",
        )

    return login


def test_authorization_url_comes_from_oauth_cli_kit() -> None:
    flow = start_openai_codex_oauth_login(
        timeout_s=2,
        open_browser=False,
    )
    try:
        params = parse_qs(urlsplit(flow.authorization_url).query)
        assert params["response_type"] == ["code"]
        assert params["client_id"] == [codex_oauth.OPENAI_CODEX_PROVIDER.client_id]
        assert params["redirect_uri"] == [codex_oauth.OPENAI_CODEX_PROVIDER.redirect_uri]
        assert params["code_challenge_method"] == ["S256"]
        assert params["code_challenge"]
        assert params["state"]
    finally:
        flow.cancel()


def test_local_flow_delegates_browser_and_callback_to_public_oauth_cli_kit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        codex_oauth,
        "login_oauth_interactive",
        _fake_interactive_login(captured),
    )
    flow = start_openai_codex_oauth_login(timeout_s=5)

    try:
        token = _wait_for_completion(flow)
    finally:
        flow.cancel()

    assert token.account_id == "acct-test"
    assert captured == {
        "provider": codex_oauth.OPENAI_CODEX_PROVIDER,
        "proxy": None,
        "open_browser": True,
    }


def test_remote_flow_delegates_pasted_callback_to_public_oauth_cli_kit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        codex_oauth,
        "login_oauth_interactive",
        _fake_interactive_login(captured),
    )
    flow = start_openai_codex_oauth_login(
        proxy="http://127.0.0.1:7890",
        timeout_s=5,
        open_browser=False,
    )
    callback_url = (
        "http://localhost:1455/auth/callback?"
        + urlencode({"code": "authorization-code", "state": "expected-state"})
    )
    try:
        assert complete_openai_codex_oauth_login(flow) is None
        with pytest.raises(OpenAICodexOAuthInputError, match="full callback URL"):
            complete_openai_codex_oauth_login(flow, "authorization-code")
        token = complete_openai_codex_oauth_login(flow, callback_url)
        if token is None:
            token = _wait_for_completion(flow)
    finally:
        flow.cancel()

    assert token is not None
    assert token.account_id == "acct-test"
    assert captured == {
        "provider": codex_oauth.OPENAI_CODEX_PROVIDER,
        "proxy": "http://127.0.0.1:7890",
        "open_browser": False,
        "callback_url": callback_url,
    }


def test_remote_flow_rejects_callback_from_another_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        codex_oauth,
        "login_oauth_interactive",
        _fake_interactive_login(captured),
    )
    flow = start_openai_codex_oauth_login(
        timeout_s=5,
        open_browser=False,
    )
    callback_url = (
        "http://localhost:1455/auth/callback?"
        + urlencode({"code": "authorization-code", "state": "wrong-state"})
    )
    try:
        with pytest.raises(OpenAICodexOAuthInputError, match="does not belong"):
            complete_openai_codex_oauth_login(flow, callback_url)
        assert "callback_url" not in captured
    finally:
        flow.cancel()


def test_remote_flow_reports_authorization_denial_without_exchanging_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        codex_oauth,
        "login_oauth_interactive",
        _fake_interactive_login(captured),
    )
    flow = start_openai_codex_oauth_login(
        timeout_s=5,
        open_browser=False,
    )
    callback_url = (
        "http://localhost:1455/auth/callback?"
        + urlencode({"error": "access_denied", "state": "expected-state"})
    )
    try:
        with pytest.raises(OpenAICodexOAuthError, match="authorization server"):
            complete_openai_codex_oauth_login(flow, callback_url)
    finally:
        flow.cancel()

    assert "callback_url" not in captured


def test_dependency_error_is_bounded_and_does_not_expose_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        codex_oauth,
        "login_oauth_interactive",
        _fake_interactive_login(
            captured,
            error=RuntimeError("Token exchange failed: 400 secret-code upstream-body"),
        ),
    )
    flow = start_openai_codex_oauth_login(
        timeout_s=5,
        open_browser=False,
    )
    callback_url = (
        "http://localhost:1455/auth/callback?"
        + urlencode({"code": "secret-code", "state": "expected-state"})
    )
    try:
        with pytest.raises(OpenAICodexOAuthError) as exc:
            token = complete_openai_codex_oauth_login(flow, callback_url)
            if token is None:
                _wait_for_completion(flow)
    finally:
        flow.cancel()

    assert str(exc.value) == "OpenAI Codex OAuth token exchange failed with HTTP 400."
    assert "secret-code" not in str(exc.value)


def test_remote_flow_expires_while_waiting_for_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        codex_oauth,
        "login_oauth_interactive",
        _fake_interactive_login({}),
    )
    flow = start_openai_codex_oauth_login(
        timeout_s=0.05,
        open_browser=False,
    )
    try:
        time.sleep(0.08)
        with pytest.raises(OpenAICodexOAuthError, match="expired"):
            complete_openai_codex_oauth_login(flow)
    finally:
        flow.cancel()


def _gated_login(
    exchange_started: threading.Event,
    release_exchange: threading.Event,
    *,
    access: str,
) -> Callable[..., OAuthToken]:
    def login(**kwargs) -> OAuthToken:
        kwargs["print_fn"]("Open this URL:")
        kwargs["print_fn"](_authorization_url())
        exchange_started.set()
        assert release_exchange.wait(timeout=2)
        return OAuthToken(
            access=access,
            refresh=f"{access}-refresh",
            expires=2_000_000_000_000,
            account_id="acct-test",
        )

    return login


def test_cancelled_login_cannot_publish_after_the_exchange_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange_started = threading.Event()
    release_exchange = threading.Event()
    monkeypatch.setattr(
        codex_oauth,
        "login_oauth_interactive",
        _gated_login(exchange_started, release_exchange, access="late-access"),
    )

    flow = start_openai_codex_oauth_login(timeout_s=5)
    try:
        assert exchange_started.wait(timeout=2)
        first_generation = flow.login_generation
        flow.cancel()
        release_exchange.set()
        flow._thread.join(timeout=2)

        assert not Path(os.environ["KOHAKU_CODEX_AUTH_PATH"]).exists()
        assert get_codex_login_status() is CredentialStatus.ABSENT
        with pytest.raises(OpenAICodexOAuthError, match="cancelled"):
            flow.complete()
    finally:
        release_exchange.set()
        flow.cancel()

    monkeypatch.setattr(
        codex_oauth,
        "login_oauth_interactive",
        _fake_interactive_login({}),
    )
    newer = start_openai_codex_oauth_login(timeout_s=5)
    try:
        assert newer.login_generation > first_generation
        newer_token = _wait_for_completion(newer)
    finally:
        newer.cancel()

    assert newer_token.access == "access-token"
    assert get_codex_credential_view().token.access == "access-token"
    assert get_codex_login_status() is CredentialStatus.AVAILABLE


def test_inflight_login_cannot_publish_after_logout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange_started = threading.Event()
    release_exchange = threading.Event()
    monkeypatch.setattr(
        codex_oauth,
        "login_oauth_interactive",
        _gated_login(exchange_started, release_exchange, access="late-access"),
    )

    flow = start_openai_codex_oauth_login(timeout_s=5)
    try:
        assert exchange_started.wait(timeout=2)
        logout_codex()
        release_exchange.set()
        flow._thread.join(timeout=2)

        assert not Path(os.environ["KOHAKU_CODEX_AUTH_PATH"]).exists()
        assert get_codex_login_status() is CredentialStatus.LOGGED_OUT
        with pytest.raises(OpenAICodexOAuthError):
            flow.complete()
    finally:
        release_exchange.set()
        flow.cancel()


def test_expired_login_cannot_publish_after_the_exchange_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange_started = threading.Event()
    release_exchange = threading.Event()
    monkeypatch.setattr(
        codex_oauth,
        "login_oauth_interactive",
        _gated_login(exchange_started, release_exchange, access="late-access"),
    )

    flow = start_openai_codex_oauth_login(timeout_s=5)
    try:
        assert exchange_started.wait(timeout=2)
        flow._expires_at = 0
        with pytest.raises(OpenAICodexOAuthError, match="expired"):
            flow.complete()
        release_exchange.set()
        flow._thread.join(timeout=2)

        assert not Path(os.environ["KOHAKU_CODEX_AUTH_PATH"]).exists()
        assert get_codex_login_status() is CredentialStatus.ABSENT
    finally:
        release_exchange.set()
        flow.cancel()
