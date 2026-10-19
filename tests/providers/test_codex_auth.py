from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from nanobot.providers import codex_auth
from nanobot.providers.codex_auth import (
    _load_official_codex_token,
    _nanobot_token_path,
    _official_codex_auth_path,
    _refresh_token,
    format_codex_usage,
    get_codex_usage,
    get_token,
)


def _jwt(payload: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{header}.{body}.sig"


def test_load_official_codex_token_from_tokens_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex-home"))
    auth_path = _official_codex_auth_path()
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    access = _jwt(
        {
            "exp": 4_102_444_800,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"},
        }
    )
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access,
                    "refresh_token": "refresh_123",
                    "account_id": "acct_123",
                },
                "last_refresh": "2026-04-12T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    token = _load_official_codex_token(auth_path)

    assert token is not None
    assert token.access == access
    assert token.refresh == "refresh_123"
    assert token.account_id == "acct_123"
    assert token.expires == 4_102_444_800_000


def test_get_token_imports_official_codex_auth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    access = _jwt(
        {
            "exp": 4_102_444_800,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_home"},
        }
    )
    auth_path = tmp_path / ".codex" / "auth.json"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access,
                    "refresh_token": "refresh_home",
                    "account_id": "acct_home",
                }
            }
        ),
        encoding="utf-8",
    )
    official_before = auth_path.read_bytes()

    token = get_token()
    token_path = _nanobot_token_path()

    assert token.access == access
    assert token.refresh == "refresh_home"
    assert token.account_id == "acct_home"
    saved = json.loads(token_path.read_text(encoding="utf-8"))
    assert saved["version"] == 2
    assert saved["token"]["account_id"] == "acct_home"
    assert saved["token"]["refresh"] == "refresh_home"
    assert auth_path.read_bytes() == official_before


@pytest.mark.parametrize("use_override", [True, False])
def test_authority_path_rejects_final_symlink_without_touching_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_override: bool
) -> None:
    target = tmp_path / "target.json"
    target.write_text("sentinel", encoding="utf-8")
    before = target.read_bytes()
    link = (
        tmp_path / "codex.json"
        if use_override
        else tmp_path / ".nanobot" / "auth" / "openai-codex.json"
    )
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    if use_override:
        monkeypatch.setenv("NANOBOT_CODEX_AUTH_PATH", str(link))
    else:
        monkeypatch.delenv("NANOBOT_CODEX_AUTH_PATH", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(codex_auth.CredentialError, match="invalid credential store"):
        codex_auth.get_codex_authority()

    assert target.read_bytes() == before
    assert link.is_symlink()


def test_refresh_token_keeps_existing_refresh_when_endpoint_omits_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Response:
        status_code = 200
        text = '{"access_token":"new_access"}'

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "access_token": _jwt(
                    {
                        "exp": 4_102_444_900,
                        "https://api.openai.com/auth": {"chatgpt_account_id": "acct_refresh"},
                    }
                )
            }

    monkeypatch.setattr("nanobot.providers.codex_auth.httpx.post", lambda *args, **kwargs: _Response())

    token = _refresh_token("refresh_existing", "acct_refresh")

    assert token.refresh == "refresh_existing"
    assert token.account_id == "acct_refresh"
    assert token.expires == 4_102_444_900_000


def test_get_codex_usage_uses_oauth_account_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Token:
        access = "access_123"
        account_id = "acct_123"

    class _Response:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"plan_type": "plus", "rate_limit": {"primary_window": {"used_percent": 25}}}

    captured: dict[str, object] = {}

    def _get(url: str, **kwargs: object) -> _Response:
        captured["url"] = url
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr("nanobot.providers.codex_auth.get_token", lambda: _Token())
    monkeypatch.setattr("nanobot.providers.codex_auth.httpx.get", _get)

    payload = get_codex_usage()

    assert payload["plan_type"] == "plus"
    assert captured["url"] == "https://chatgpt.com/backend-api/wham/usage"
    assert captured["headers"] == {
        "Authorization": "Bearer access_123",
        "ChatGPT-Account-ID": "acct_123",
        "originator": "codex_cli_rs",
        "Accept": "application/json",
    }


def test_format_codex_usage_renders_windows_and_reset_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nanobot.providers.codex_auth.time.time", lambda: 1_700_000_000)

    content = format_codex_usage(
        {
            "plan_type": "plus",
            "rate_limit": {
                "primary_window": {
                    "used_percent": 25,
                    "limit_window_seconds": 900,
                    "reset_at": 1_700_003_600,
                }
            },
        }
    )

    assert "Codex quota (plus):" in content
    assert "Primary: 75% remaining" in content
    assert "window 15m" not in content
    assert "Primary: 75% remaining\nReset at 2023-11-15 07:13 MYT" in content


@pytest.mark.parametrize('raw', [1700000000, 1700000000.0, '1700000000', 1700000000000])
def test_absolute_expiry_never_becomes_duration(raw):
    assert codex_auth._coerce_expires_ms(raw, None) == 1700000000000


def test_explicit_refresh_duration_is_relative(monkeypatch):
    monkeypatch.setattr(codex_auth, '_now_ms', lambda: 1700000000000)
    assert codex_auth._coerce_expires_ms(3600, None, duration=True) == 1700003600000


@pytest.mark.parametrize('expiry,expected', [({'expires_at': 1700000000}, 1700000000000),
                                            ({'expires_in': 0, 'expires_at': 1800000000}, 1700000000000)])
def test_refresh_expiry_field_semantics(monkeypatch, expiry, expected):
    from types import SimpleNamespace
    payload = {'access_token': 'synthetic', 'account_id': 'fixture', **expiry}
    monkeypatch.setattr(codex_auth, '_now_ms', lambda: 1700000000000)
    monkeypatch.setattr(codex_auth.httpx, 'post', lambda *args, **kwargs:
                        SimpleNamespace(status_code=200, json=lambda: payload))
    assert _refresh_token('synthetic-refresh', 'fixture').expires == expected


@pytest.mark.parametrize("value", [0, False, "", [], {}])
def test_first_value_keeps_falsey_non_none_value(value) -> None:
    payload = {17: "unrelated", "preferred": value, "fallback": "not-selected"}
    assert codex_auth._first_value(payload, "missing", "preferred", "fallback") is value
    assert codex_auth._first_value({"preferred": None, "fallback": value},
                                   "preferred", "fallback") is value


@pytest.mark.parametrize("payload", [None, [], "text", 7, False])
def test_jwt_decoder_keeps_non_dictionary_refusal(payload) -> None:
    token = _jwt(payload)
    assert codex_auth._decode_jwt_payload(token) == {}
    assert codex_auth._account_id_from_access_token(token) is None


@pytest.mark.parametrize("token_block", [None, [], "not-a-block", {}, {"access_token": 7}])
def test_official_token_block_selection_retains_dictionary_precedence(
    tmp_path: Path, token_block
) -> None:
    path = tmp_path / "synthetic-official.json"
    payload = {
        "tokens": token_block,
        "access_token": "synthetic-access",
        "refresh_token": "synthetic-refresh",
        "account_id": "synthetic-account",
        "expires": 4_102_444_800_000,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    token = _load_official_codex_token(path)
    if isinstance(token_block, dict):
        assert token is None
    else:
        assert token is not None
        assert token.access == "synthetic-access"
        assert token.account_id == "synthetic-account"
    assert path.read_bytes() == before


@pytest.mark.parametrize("payload", [None, [], "text", 7, False])
def test_usage_keeps_non_dictionary_response_refusal(monkeypatch, payload) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(codex_auth, "get_token", lambda: SimpleNamespace(
        access="synthetic-access", account_id="synthetic-account"))
    response = codex_auth.httpx.Response(
        200, content=json.dumps(payload), request=codex_auth.httpx.Request(
            "GET", "https://example.invalid/synthetic"))
    monkeypatch.setattr(codex_auth.httpx, "get", lambda *_args, **_kwargs: response)
    with pytest.raises(codex_auth.CodexUsageError, match="unexpected quota response"):
        get_codex_usage()


def test_usage_retains_response_identity_and_optional_header_failure(monkeypatch) -> None:
    from types import SimpleNamespace

    token = SimpleNamespace(access="synthetic-access", account_id="synthetic-account")
    payload = {"rate_limit": {}, "unrecognized": [1, None, False]}
    captured = []

    def get(_url, *, headers, timeout):
        captured.append((headers, timeout))
        codex_auth.httpx.Headers(headers)
        return SimpleNamespace(status_code=200, json=lambda: payload)

    monkeypatch.setattr(codex_auth, "get_token", lambda: token)
    monkeypatch.setattr(codex_auth.httpx, "get", get)
    assert get_codex_usage() is payload
    assert captured[0][0]["ChatGPT-Account-ID"] == "synthetic-account"
    token.account_id = None
    with pytest.raises(TypeError):
        get_codex_usage()
    assert captured[-1][0]["ChatGPT-Account-ID"] is None
